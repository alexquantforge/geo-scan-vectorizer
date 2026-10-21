#!/usr/bin/env python3
import cv2, json, argparse, os, re
import itertools
import numpy as np
from pathlib import Path

ALIGNED_DOT_MIN_COUNT = 5
ALIGNED_DOT_TOLERANCE = 6.0
ALIGNED_DOT_MAX_GAP = 110.0
ALIGNED_DOT_MIN_SPAN = 40.0
DOT_CANDIDATE_MERGE_DISTANCE = 9.0


def split_into_tiles(img, cols, rows, overlap):
    H,W=img.shape[:2]
    tile_w=(W+overlap*(cols-1))//cols; tile_h=(H+overlap*(rows-1))//rows
    tiles=[]
    for r in range(rows):
        for c in range(cols):
            x0=c*(tile_w-overlap); y0=r*(tile_h-overlap)
            x1=min(x0+tile_w,W); y1=min(y0+tile_h,H)
            x0=max(0,x1-tile_w) if x1==W else x0
            y0=max(0,y1-tile_h) if y1==H else y0
            tiles.append((img[y0:y1, x0:x1].copy(), x0,y0, f"r{r+1}c{c+1}_x{x0}_y{y0}"))
    return tiles

def line_has_dark_support(ink_support, x1, y1, x2, y2, min_ratio=0.35):
    """Require a candidate segment to follow actual dark ink, not just texture edges."""
    h, w = ink_support.shape[:2]
    length = float(np.hypot(x2 - x1, y2 - y1))
    steps = max(12, int(np.ceil(length * 1.5)))
    xs = np.rint(np.linspace(x1, x2, steps)).astype(int)
    ys = np.rint(np.linspace(y1, y2, steps)).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not np.any(valid):
        return False
    support_ratio = np.mean(ink_support[ys[valid], xs[valid]] > 0)
    return support_ratio >= min_ratio


def blue_cyan_mask(tile):
    """Mask colored blue/cyan marks without whitening neutral dark ink.

    Antialiased black strokes on cyan paper can inherit a blue hue. Preserve
    pixels that are both dark and nearly neutral, including after mask dilation.
    Strongly chromatic blue/cyan overlay lines remain masked.
    """
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array([90, 25, 60]), np.array([128, 255, 255])
    )
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    channels = tile.astype(np.int16)
    channel_spread = channels.max(axis=2) - channels.min(axis=2)
    dark_neutral_ink = (gray <= 110) & (channel_spread <= 55)
    mask[dark_neutral_ink] = 0
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask[dark_neutral_ink] = 0
    return mask


def detect_dashed_curve_segments(gray, binary):
    """Join dark, similarly oriented components into oblique dashed curves.

    Long horizontal/vertical grid lines are removed first. The component support
    threshold rejects pale paper texture, while the path constraints favor a
    smooth run of dark dashes over isolated symbols and lettering.
    """
    height, width = gray.shape[:2]
    if min(height, width) < 80:
        return []

    # Find only prominent, nearly axis-aligned ink lines (typically the grid).
    dark_ink = np.where(gray <= 90, 255, 0).astype(np.uint8)
    grid_edges = cv2.Canny(dark_ink, 50, 150)
    min_grid_length = max(120, int(0.45 * max(height, width)))
    hough_threshold = max(50, int(0.12 * min(height, width)))
    grid_lines = cv2.HoughLinesP(
        grid_edges,
        1,
        np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_grid_length,
        maxLineGap=20,
    )
    grid_mask = np.zeros_like(binary)
    if grid_lines is not None:
        for x1, y1, x2, y2 in np.asarray(grid_lines).reshape(-1, 4):
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            horizontal = angle <= 5 or angle >= 175
            vertical = 85 <= angle <= 95
            if length < min_grid_length or not (horizontal or vertical):
                continue
            cv2.line(grid_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 7)

    components = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    line_support = cv2.dilate(
        (gray <= 110).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1
    )
    components = cv2.bitwise_and(components, cv2.bitwise_not(grid_mask))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        components, connectivity=8
    )

    nodes = []
    for label in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[label])
        major = max(box_width, box_height)
        minor = max(1, min(box_width, box_height))
        if not (6 <= major <= 65 and area >= 18 and major / minor >= 1.1):
            continue

        # Work inside each small component box rather than scanning the full
        # label image once per component.
        local_labels = labels[y : y + box_height, x : x + box_width]
        local_y, local_x = np.nonzero(local_labels == label)
        if len(local_x) < 3:
            continue
        dark_ratio = float(
            np.mean(gray[y + local_y, x + local_x] <= 110)
        )
        if dark_ratio < 0.12:
            continue

        points = np.column_stack((local_x + x, local_y + y)).astype(np.float32)
        _, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
        axis = eigenvectors[0]
        axis /= np.linalg.norm(axis)
        center_x, center_y = centroids[label]
        nodes.append(
            {
                "x": float(center_x),
                "y": float(center_y),
                "axis": axis,
                "dark_ratio": dark_ratio,
            }
        )

    if len(nodes) < 6:
        return []
    nodes.sort(key=lambda node: (node["y"], node["x"]))

    # Build a forward graph in the dominant vertical direction. The target
    # boundary is oblique, so the overall slope constraint rejects the long
    # near-vertical grid remnants that survive local thresholding.
    successors = [[] for _ in nodes]
    predecessors = [[] for _ in nodes]
    max_vertical_gap = 120
    max_distance = 140
    for index, first in enumerate(nodes):
        for next_index in range(index + 1, len(nodes)):
            second = nodes[next_index]
            delta_y = second["y"] - first["y"]
            if delta_y > max_vertical_gap:
                break
            if delta_y < 5:
                continue

            delta_x = second["x"] - first["x"]
            distance = float(np.hypot(delta_x, delta_y))
            if distance > max_distance or abs(delta_x) > 120:
                continue
            direction = np.array([delta_x, delta_y], dtype=np.float32) / distance
            align_first = abs(float(np.dot(first["axis"], direction)))
            align_second = abs(float(np.dot(second["axis"], direction)))
            if align_first < np.cos(np.deg2rad(35)):
                continue
            if align_second < np.cos(np.deg2rad(40)):
                continue
            if not line_has_dark_support(
                line_support,
                first["x"],
                first["y"],
                second["x"],
                second["y"],
                min_ratio=0.5,
            ):
                continue

            angle = float(np.degrees(np.arctan2(delta_y, delta_x)))
            successors[index].append(
                (next_index, distance, align_first, align_second, angle)
            )
            predecessors[next_index].append(index)

    # Dynamic programming over pairs preserves the incoming direction so a
    # zig-zag through adjacent text strokes cannot win just by having many blobs.
    states = {}
    for index in range(len(nodes)):
        for next_index, distance, align_first, align_second, angle in successors[index]:
            edge_score = 1.0 + align_first + align_second - 0.006 * distance
            states[(index, next_index)] = (edge_score, [index, next_index], angle, 0.0)

            for previous_index in predecessors[index]:
                previous = states.get((previous_index, index))
                if previous is None:
                    continue
                previous_score, path, previous_angle, total_turn = previous
                turn = abs((angle - previous_angle + 180) % 360 - 180)
                if turn > 75:
                    continue
                score = (
                    previous_score
                    + edge_score
                    - 0.04 * turn
                )
                if score > states[(index, next_index)][0]:
                    states[(index, next_index)] = (
                        score,
                        path + [next_index],
                        angle,
                        total_turn + turn,
                    )

    candidates = []
    for score, path, _, total_turn in states.values():
        first = nodes[path[0]]
        last = nodes[path[-1]]
        vertical_span = last["y"] - first["y"]
        if len(path) < 6 or vertical_span < 150:
            continue
        slope_ratio = abs(last["x"] - first["x"]) / max(vertical_span, 1.0)
        mean_turn = total_turn / max(len(path) - 2, 1)
        if not (0.2 <= slope_ratio <= 1.5) or mean_turn > 9:
            continue
        candidates.append((score + 0.02 * vertical_span, path))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    selected_paths = []
    used_nodes = set()
    for _, path in candidates:
        overlap = len(set(path) & used_nodes) / len(path)
        if overlap > 0.5:
            continue
        selected_paths.append(path)
        used_nodes.update(path)

    segments = []
    for path in selected_paths:
        for first_index, second_index in zip(path, path[1:]):
            first = nodes[first_index]
            second = nodes[second_index]
            length = float(np.hypot(second["x"] - first["x"], second["y"] - first["y"]))
            angle = float(
                np.degrees(
                    np.arctan2(second["y"] - first["y"], second["x"] - first["x"])
                )
            )
            segments.append(
                (first["x"], first["y"], second["x"], second["y"], length, angle)
            )
    return segments


def detect_small_circles(gray, binary):
    """Find small, closed ring symbols while rejecting isolated texture specks.

    Prefer a circular outer contour around a centered, light hole. A fallback
    also accepts a small light hole when an attached stroke distorts the outer
    contour, but only if a dark annulus is strongly supported around it. Bright
    core checks reject filled dots; non-maximum suppression removes duplicates.
    """
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    candidates = []

    def ring_evidence(
        center_x,
        center_y,
        radius,
        inner_x,
        inner_y,
        inner_radius,
        sample_count=48,
        patch_radius=2,
    ):
        core_radius = max(1.0, inner_radius * 0.55)
        core_x0 = max(0, int(np.floor(inner_x - core_radius)))
        core_x1 = min(gray.shape[1], int(np.ceil(inner_x + core_radius)) + 1)
        core_y0 = max(0, int(np.floor(inner_y - core_radius)))
        core_y1 = min(gray.shape[0], int(np.ceil(inner_y + core_radius)) + 1)
        core_patch = gray[core_y0:core_y1, core_x0:core_x1]
        core_y, core_x = np.ogrid[core_y0:core_y1, core_x0:core_x1]
        core_mask = (core_x - inner_x) ** 2 + (core_y - inner_y) ** 2 <= core_radius ** 2
        core_pixels = core_patch[core_mask]
        if core_pixels.size < 3:
            return None
        core_brightness = float(np.median(core_pixels))

        supported_angles = 0
        dark_ring_pixels = []
        for angle in np.linspace(0, 2 * np.pi, sample_count, endpoint=False):
            sample_x = int(round(center_x + radius * np.cos(angle)))
            sample_y = int(round(center_y + radius * np.sin(angle)))
            y_start = max(0, sample_y - patch_radius)
            y_end = min(gray.shape[0], sample_y + patch_radius + 1)
            x_start = max(0, sample_x - patch_radius)
            x_end = min(gray.shape[1], sample_x + patch_radius + 1)
            patch = gray[y_start:y_end, x_start:x_end]
            if patch.size:
                dark_pixels = patch[patch <= 110]
                if dark_pixels.size:
                    supported_angles += 1
                    dark_ring_pixels.extend(dark_pixels.tolist())

        if not dark_ring_pixels:
            return None
        dark_support = supported_angles / sample_count
        ring_brightness = float(np.median(dark_ring_pixels))
        return core_brightness, dark_support, ring_brightness

    for index, contour in enumerate(contours):
        child_index = int(hierarchy[index][2])
        if child_index < 0:
            continue

        outer_area = float(cv2.contourArea(contour))
        outer_perimeter = float(cv2.arcLength(contour, True))
        if outer_area <= 0 or outer_perimeter <= 0:
            continue
        outer_circularity = 4 * np.pi * outer_area / (outer_perimeter ** 2)
        _, _, box_width, box_height = cv2.boundingRect(contour)
        aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
        (outer_x, outer_y), outer_radius = cv2.minEnclosingCircle(contour)
        if not (
            5.5 <= outer_radius <= 20.0
            and aspect_ratio <= 1.4
            and outer_circularity >= 0.5
        ):
            continue

        matching_holes = []
        while child_index >= 0:
            hole = contours[child_index]
            hole_area = float(cv2.contourArea(hole))
            hole_perimeter = float(cv2.arcLength(hole, True))
            if hole_area > 0 and hole_perimeter > 0:
                hole_circularity = 4 * np.pi * hole_area / (hole_perimeter ** 2)
                _, _, hole_width, hole_height = cv2.boundingRect(hole)
                hole_aspect = max(hole_width, hole_height) / max(
                    1, min(hole_width, hole_height)
                )
                (inner_x, inner_y), inner_radius = cv2.minEnclosingCircle(hole)
                center_offset = float(np.hypot(inner_x - outer_x, inner_y - outer_y))
                radius_ratio = inner_radius / max(outer_radius, 1.0)
                if (
                    0.3 <= radius_ratio <= 0.9
                    and center_offset <= 4.0
                    and hole_aspect <= 1.5
                    and hole_circularity >= 0.3
                ):
                    matching_holes.append(
                        (
                            hole_circularity
                            - 0.05 * center_offset
                            - abs(radius_ratio - 0.5),
                            inner_x,
                            inner_y,
                            inner_radius,
                            center_offset,
                            radius_ratio,
                        )
                    )
            child_index = int(hierarchy[child_index][0])

        if not matching_holes:
            continue
        hole_score, inner_x, inner_y, inner_radius, center_offset, radius_ratio = max(
            matching_holes, key=lambda item: item[0]
        )

        center_x = (outer_x + inner_x) / 2.0
        center_y = (outer_y + inner_y) / 2.0
        radius = (outer_radius + inner_radius) / 2.0 + 1.0

        # The interior must be genuinely light, with a dark annulus around it.
        evidence = ring_evidence(
            center_x,
            center_y,
            radius - 0.8,
            inner_x,
            inner_y,
            inner_radius,
        )
        if evidence is None:
            continue
        core_brightness, dark_support, ring_brightness = evidence
        if dark_support < 0.5:
            continue
        if core_brightness < 130 or core_brightness - ring_brightness < 30:
            continue

        score = (
            outer_circularity
            + dark_support
            + 0.1 * hole_score
            - 0.05 * center_offset
            - 0.2 * abs(radius_ratio - 0.5)
        )
        candidates.append((score, center_x, center_y, radius))

    # A circle may touch a nearby stroke, making the outer contour elongated.
    # In that case use its small, bright hole as the center proposal and verify
    # a dark annulus locally instead of requiring the whole component to be round.
    for hole_index, hole in enumerate(contours):
        parent_index = int(hierarchy[hole_index][3])
        if parent_index < 0:
            continue

        ancestor_count = 0
        ancestor_index = parent_index
        while ancestor_index >= 0:
            ancestor_count += 1
            ancestor_index = int(hierarchy[ancestor_index][3])
        if ancestor_count % 2 == 0:
            continue

        hole_area = float(cv2.contourArea(hole))
        hole_perimeter = float(cv2.arcLength(hole, True))
        if hole_area <= 0 or hole_perimeter <= 0:
            continue
        hole_circularity = 4 * np.pi * hole_area / (hole_perimeter ** 2)
        _, _, hole_width, hole_height = cv2.boundingRect(hole)
        hole_aspect = max(hole_width, hole_height) / max(1, min(hole_width, hole_height))
        (inner_x, inner_y), inner_radius = cv2.minEnclosingCircle(hole)
        if not (
            2.5 <= inner_radius <= 8.5
            and hole_aspect <= 1.5
            and hole_circularity >= 0.5
        ):
            continue

        parent = contours[parent_index]
        parent_area = float(cv2.contourArea(parent))
        parent_perimeter = float(cv2.arcLength(parent, True))
        _, _, parent_width, parent_height = cv2.boundingRect(parent)
        parent_aspect = max(parent_width, parent_height) / max(
            1, min(parent_width, parent_height)
        )
        _, parent_radius = cv2.minEnclosingCircle(parent)
        parent_circularity = (
            4 * np.pi * parent_area / (parent_perimeter ** 2)
            if parent_perimeter > 0
            else 0.0
        )
        if max(parent_width, parent_height) > 40 or parent_area > 700:
            continue
        if (
            5.5 <= parent_radius <= 20.0
            and parent_aspect <= 1.4
            and parent_circularity >= 0.5
        ):
            continue  # The primary outer-contour pass already handles this ring.

        best_fallback = None
        min_radius = max(inner_radius + 0.8, 3.5)
        max_radius = min(18.0, inner_radius * 2.4)
        preferred_radius = inner_radius + 2.0
        for ring_radius in np.arange(min_radius, max_radius + 0.01, 0.5):
            evidence = ring_evidence(
                inner_x,
                inner_y,
                ring_radius,
                inner_x,
                inner_y,
                inner_radius,
                sample_count=72,
                patch_radius=1,
            )
            if evidence is None:
                continue
            core_brightness, dark_support, ring_brightness = evidence
            if dark_support < 0.65:
                continue
            if core_brightness < 130 or core_brightness - ring_brightness < 30:
                continue
            rank = (
                dark_support,
                -ring_brightness,
                -abs(ring_radius - preferred_radius),
            )
            if best_fallback is None or rank > best_fallback[0]:
                best_fallback = (rank, ring_radius, dark_support)

        if best_fallback is not None:
            _, ring_radius, dark_support = best_fallback
            score = dark_support - 0.01 * abs(ring_radius - preferred_radius)
            candidates.append((1.0 + score, inner_x, inner_y, ring_radius))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    circles = []
    for _, center_x, center_y, radius in candidates:
        if any(
            np.hypot(center_x - other_x, center_y - other_y) < 5.0
            and abs(radius - other_radius) < 3.0
            for other_x, other_y, other_radius in circles
        ):
            continue
        circles.append((float(center_x), float(center_y), float(radius)))
    return circles


def detect_dot_candidates(tile, gray):
    """Extract compact dark components before checking whether they form a row."""
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    colored_overlay = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 60)
    colored_overlay = cv2.dilate(
        colored_overlay.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    )

    dot_gray = gray.copy()
    dot_gray[colored_overlay > 0] = 255
    dot_gray = cv2.GaussianBlur(dot_gray, (3, 3), 0)
    binary = np.where(dot_gray <= 80, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    candidates = []
    for label in range(1, component_count):
        x, y, width, height, area = (int(value) for value in stats[label])
        max_dimension = max(width, height)
        min_dimension = min(width, height)
        aspect_ratio = max_dimension / max(1, min_dimension)
        if not (
            4 <= max_dimension <= 18
            and 3 <= min_dimension <= 16
            and 12 <= area <= 180
            and aspect_ratio <= 2.0
        ):
            continue

        component = (labels[y : y + height, x : x + width] == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * float(cv2.contourArea(contour)) / perimeter**2
        fill_ratio = area / float(width * height)
        if circularity < 0.15 or fill_ratio < 0.25:
            continue

        candidates.append(
            {
                "x": float(centroids[label][0]),
                "y": float(centroids[label][1]),
                "radius": max(2.5, max_dimension / 2.0 + 1.0),
                "area": area,
            }
        )
    return candidates


def merge_dot_candidates(candidates):
    """Merge nearby component fragments so one printed mark counts as one dot."""
    if not candidates:
        return []

    parents = list(range(len(candidates)))

    def find_root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for first, second in itertools.combinations(range(len(candidates)), 2):
        first_candidate = candidates[first]
        second_candidate = candidates[second]
        distance = float(
            np.hypot(
                first_candidate["x"] - second_candidate["x"],
                first_candidate["y"] - second_candidate["y"],
            )
        )
        if distance <= DOT_CANDIDATE_MERGE_DISTANCE:
            first_root = find_root(first)
            second_root = find_root(second)
            if first_root != second_root:
                parents[second_root] = first_root

    clusters = {}
    for index, candidate in enumerate(candidates):
        clusters.setdefault(find_root(index), []).append(candidate)

    merged = []
    for cluster in clusters.values():
        total_area = sum(candidate["area"] for candidate in cluster)
        merged.append(
            {
                "x": sum(candidate["x"] * candidate["area"] for candidate in cluster)
                / total_area,
                "y": sum(candidate["y"] * candidate["area"] for candidate in cluster)
                / total_area,
                "radius": max(candidate["radius"] for candidate in cluster),
                "area": total_area,
            }
        )
    return merged


def group_aligned_dot_candidates(candidates):
    """Return the individual dot centers in near-straight groups of at least five."""
    points = merge_dot_candidates(candidates)
    if len(points) < ALIGNED_DOT_MIN_COUNT:
        return []

    # Pair hypotheses act as a deterministic RANSAC pass; the perpendicular
    # tolerance allows scan variation without joining nearby rows.
    tolerance = ALIGNED_DOT_TOLERANCE
    max_gap = ALIGNED_DOT_MAX_GAP
    min_span = ALIGNED_DOT_MIN_SPAN
    group_candidates = {}

    def add_chunk(chunk):
        if len(chunk) < ALIGNED_DOT_MIN_COUNT:
            return
        span = chunk[-1][0] - chunk[0][0]
        if span < min_span:
            return
        indices = tuple(sorted(point_index for _, point_index in chunk))
        group_candidates[indices] = max(group_candidates.get(indices, 0.0), span)

    for first, second in itertools.combinations(range(len(points)), 2):
        first_point = points[first]
        second_point = points[second]
        delta_x = second_point["x"] - first_point["x"]
        delta_y = second_point["y"] - first_point["y"]
        seed_distance = float(np.hypot(delta_x, delta_y))
        if seed_distance < 24.0:
            continue

        unit_x = delta_x / seed_distance
        unit_y = delta_y / seed_distance
        normal_x = -unit_y
        normal_y = unit_x
        inliers = []
        for point_index, point in enumerate(points):
            offset_x = point["x"] - first_point["x"]
            offset_y = point["y"] - first_point["y"]
            perpendicular_distance = abs(offset_x * normal_x + offset_y * normal_y)
            if perpendicular_distance <= tolerance:
                projection = offset_x * unit_x + offset_y * unit_y
                inliers.append((projection, point_index))

        inliers.sort()
        chunk = []
        for item in inliers:
            if chunk and item[0] - chunk[-1][0] > max_gap:
                add_chunk(chunk)
                chunk = []
            chunk.append(item)
        add_chunk(chunk)

    ordered_groups = sorted(
        group_candidates.items(), key=lambda item: (-len(item[0]), -item[1])
    )
    accepted_groups = []
    for indices, _ in ordered_groups:
        index_set = set(indices)
        if any(
            len(index_set & set(accepted)) / min(len(indices), len(accepted)) >= 0.8
            for accepted in accepted_groups
        ):
            continue
        accepted_groups.append(indices)

    selected_indices = sorted(
        {index for group in accepted_groups for index in group},
        key=lambda index: (points[index]["y"], points[index]["x"]),
    )
    return [points[index] for index in selected_indices]


def detect_aligned_dots(tile, gray):
    """Return compact dark marks belonging to a near-straight row of 5+ dots."""
    return [
        (point["x"], point["y"], point["radius"])
        for point in group_aligned_dot_candidates(detect_dot_candidates(tile, gray))
    ]


def detect_all(tile, x0,y0, dot_candidates_out=None):
    # Ignore colored blue/cyan marks, but retain dark neutral ink on cyan paper.
    blue_mask = blue_cyan_mask(tile)
    tile_for_detection = tile.copy()
    tile_for_detection[blue_mask > 0] = (255, 255, 255)
    gray=cv2.cvtColor(tile_for_detection, cv2.COLOR_BGR2GRAY)
    dark_pixels=(gray <= 110).astype(np.uint8)
    ink_support=cv2.dilate(dark_pixels, np.ones((7,7),np.uint8), iterations=1)
    clahe=cv2.createCLAHE(2.0,(8,8)); eq=clahe.apply(gray)
    th=cv2.adaptiveThreshold(eq,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,12)
    edges=cv2.Canny(cv2.morphologyEx(th,cv2.MORPH_OPEN, np.ones((3,3),np.uint8)),50,150)
    # A smaller opening preserves thin strokes for the curved-contour pass.
    curve_edges=cv2.Canny(cv2.morphologyEx(th,cv2.MORPH_OPEN, np.ones((2,2),np.uint8)),50,150)
    out=[]
    lsd=cv2.createLineSegmentDetector()
    lines,_,_,_=lsd.detect(gray)
    if lines is not None:
        for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):
            length=float(np.hypot(x2-x1,y2-y1))
            if length<35 or not line_has_dark_support(ink_support,x1,y1,x2,y2): continue
            ang=float(np.degrees(np.arctan2(y2-y1,x2-x1)))
            out.append((x1+x0,y1+y0,x2+x0,y2+y0,length,ang,"straight", float(x1),float(y1),float(x2),float(y2)))
    hlines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=70,minLineLength=50,maxLineGap=4)
    if hlines is not None:
        for x1,y1,x2,y2 in np.asarray(hlines).reshape(-1,4):
            length=float(np.hypot(x2-x1,y2-y1)); ang=float(np.degrees(np.arctan2(y2-y1,x2-x1)))
            if not line_has_dark_support(ink_support,x1,y1,x2,y2): continue
            out.append((x1+x0,y1+y0,x2+x0,y2+y0,length,ang,"straight", float(x1),float(y1),float(x2),float(y2)))
    contours,_=cv2.findContours(curve_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.arcLength(cnt, False) < 80: continue
        approx=cv2.approxPolyDP(cnt, 2.0, False)
        if len(approx)<2: continue
        pts=approx.reshape(-1,2)
        for i in range(len(pts)-1):
            x1,y1=pts[i]; x2,y2=pts[i+1]
            length=float(np.hypot(x2-x1,y2-y1))
            if length<25 or not line_has_dark_support(ink_support,x1,y1,x2,y2): continue
            ang=float(np.degrees(np.arctan2(y2-y1,x2-x1)))
            out.append((float(x1+x0),float(y1+y0),float(x2+x0),float(y2+y0),length,ang,"curved", float(x1),float(y1),float(x2),float(y2)))
    for x1, y1, x2, y2, length, ang in detect_dashed_curve_segments(gray, th):
        out.append(
            (
                x1 + x0,
                y1 + y0,
                x2 + x0,
                y2 + y0,
                length,
                ang,
                "dashed_curve",
                x1,
                y1,
                x2,
                y2,
            )
        )
    dot_candidates = detect_dot_candidates(tile, gray)
    if dot_candidates_out is None:
        aligned_dots = [
            (point["x"], point["y"], point["radius"])
            for point in group_aligned_dot_candidates(dot_candidates)
        ]
    else:
        aligned_dots = []
        dot_candidates_out.extend(
            {
                **candidate,
                "x": candidate["x"] + x0,
                "y": candidate["y"] + y0,
            }
            for candidate in dot_candidates
        )
    small_circles = detect_small_circles(gray, th)
    for center_x, center_y, radius in small_circles:
        if any(
            np.hypot(center_x - dot_x, center_y - dot_y)
            <= max(7.0, (radius + dot_radius) / 2.0)
            for dot_x, dot_y, dot_radius in aligned_dots
        ):
            continue
        global_x, global_y = center_x + x0, center_y + y0
        out.append(
            (
                global_x,
                global_y,
                global_x,
                global_y,
                2 * np.pi * radius,
                0.0,
                "small_circle",
                center_x,
                center_y,
                center_x,
                center_y,
                radius,
            )
        )

    for center_x, center_y, radius in aligned_dots:
        global_x, global_y = center_x + x0, center_y + y0
        out.append(
            (
                global_x,
                global_y,
                global_x,
                global_y,
                2 * np.pi * radius,
                0.0,
                "aligned_dot",
                center_x,
                center_y,
                center_x,
                center_y,
                radius,
            )
        )
    return out

def main():
    ap=argparse.ArgumentParser(description="Детектор линий для геологического разреза - координаты от левого верхнего угла")
    ap.add_argument("image")
    ap.add_argument("--tiles", default="3x2")
    ap.add_argument("--overlap", type=int, default=250)
    ap.add_argument("--output", default="lines.json")
    ap.add_argument("--debug-dir", default="debug")
    args=ap.parse_args()
    img=cv2.imread(args.image); H,W=img.shape[:2]
    print(f"Исходник {W}x{H} origin top-left (0,0)")
    cols,rows=map(int,args.tiles.split("x"))
    tiles=split_into_tiles(img,cols,rows,args.overlap)
    os.makedirs(args.debug_dir, exist_ok=True)
    all_lines=[]; seen=set()
    colors = {
        "straight": (0, 0, 255),
        "curved": (0, 255, 0),
        "dashed_curve": (255, 0, 255),
        "small_circle": (0, 255, 255),
        "aligned_dot": (255, 255, 255),
    }
    for tile,x0,y0,tid in tiles:
        cand=detect_all(tile,x0,y0)
        filtered=[]
        for candidate in cand:
            x1,y1,x2,y2,length,ang,typ,lx1,ly1,lx2,ly2 = candidate[:11]
            radius = candidate[11] if len(candidate) > 11 else None
            if typ in {"small_circle", "aligned_dot"}:
                key=(typ,round(x1/2),round(y1/2),round(radius/2))
            else:
                key=(round(x1/2),round(y1/2),round(x2/2),round(y2/2),typ)
            if key in seen: continue
            seen.add(key)
            filtered.append({"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),"length":round(length,1),"angle":round(ang,1),"type":typ,"tile":tid,"radius":None if radius is None else round(float(radius),1)})
        all_lines.extend(filtered)
        counts = {
            kind: sum(line["type"] == kind for line in filtered)
            for kind in ("straight", "curved", "dashed_curve", "small_circle", "aligned_dot")
        }
        print(
            f"  {tid}: straight {counts['straight']} curved {counts['curved']} "
            f"dashed_curve {counts['dashed_curve']} small_circle {counts['small_circle']} "
            f"aligned_dot {counts['aligned_dot']}"
        )
        dbg=tile.copy()
        for f in filtered:
            lx1,ly1=f["x1"]-x0, f["y1"]-y0; lx2,ly2=f["x2"]-x0, f["y2"]-y0
            col = colors.get(f["type"], (0, 0, 255))
            if f["type"] in {"small_circle", "aligned_dot"}:
                radius = max(1, int(round(f["radius"])))
                if f["type"] == "aligned_dot":
                    cv2.circle(dbg, (int(lx1), int(ly1)), radius, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.circle(dbg, (int(lx1), int(ly1)), radius, col, 2, cv2.LINE_AA)
            else:
                cv2.line(dbg,(int(lx1),int(ly1)),(int(lx2),int(ly2)),col,2)
        cv2.imwrite(os.path.join(args.debug_dir, f"debug_{tid}.jpg"), dbg)
    all_lines=sorted(all_lines, key=lambda x: x["length"], reverse=True)
    with open(args.output,"w",encoding="utf-8") as f: json.dump({"image":args.image,"image_size":{"W":W,"H":H,"origin":"top-left"},"tiling":args.tiles,"lines_count":len(all_lines),"lines":all_lines}, f, ensure_ascii=False, indent=2)
    print(f"\nГотово! Найдено объектов: {len(all_lines)}")
    from collections import Counter; print(Counter(x["type"] for x in all_lines))
    csv=Path(args.output).with_suffix(".csv")
    with open(csv,"w",encoding="utf-8") as f:
        f.write("x1,y1,x2,y2,length,angle,type,tile,radius\n")
        for l in all_lines: f.write(f"{l['x1']},{l['y1']},{l['x2']},{l['y2']},{l['length']},{l['angle']},{l['type']},{l['tile']},{l['radius']}\n")
    print(f"CSV: {csv}")

if __name__=="__main__": main()
