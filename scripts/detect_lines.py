#!/usr/bin/env python3
import cv2, json, argparse, os, re
import numpy as np
from pathlib import Path

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


def detect_all(tile, x0,y0):
    # Blue/cyan marks are not a target class: mask them before the grayscale detectors.
    hsv=cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    blue_mask=cv2.inRange(hsv, np.array([90,25,60]), np.array([128,255,255]))
    blue_mask=cv2.dilate(blue_mask, np.ones((3,3),np.uint8), iterations=1)
    tile_for_detection=tile.copy()
    tile_for_detection[blue_mask > 0]=(255,255,255)
    gray=cv2.cvtColor(tile_for_detection, cv2.COLOR_BGR2GRAY)
    dark_pixels=(gray <= 110).astype(np.uint8)
    ink_support=cv2.dilate(dark_pixels, np.ones((7,7),np.uint8), iterations=1)
    clahe=cv2.createCLAHE(2.0,(8,8)); eq=clahe.apply(gray)
    th=cv2.adaptiveThreshold(eq,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,12)
    edges=cv2.Canny(cv2.morphologyEx(th,cv2.MORPH_OPEN, np.ones((3,3),np.uint8)),50,150)
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
    contours,_=cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    }
    for tile,x0,y0,tid in tiles:
        cand=detect_all(tile,x0,y0)
        filtered=[]
        for x1,y1,x2,y2,length,ang,typ,lx1,ly1,lx2,ly2 in cand:
            key=(round(x1/2),round(y1/2),round(x2/2),round(y2/2),typ)
            if key in seen: continue
            seen.add(key)
            filtered.append({"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),"length":round(length,1),"angle":round(ang,1),"type":typ,"tile":tid})
        all_lines.extend(filtered)
        counts = {kind: sum(line["type"] == kind for line in filtered) for kind in ("straight", "curved", "dashed_curve")}
        print(f"  {tid}: straight {counts['straight']} curved {counts['curved']} dashed_curve {counts['dashed_curve']}")
        dbg=tile.copy()
        for f in filtered:
            lx1,ly1=f["x1"]-x0, f["y1"]-y0; lx2,ly2=f["x2"]-x0, f["y2"]-y0
            col = colors.get(f["type"], (0, 0, 255))
            cv2.line(dbg,(int(lx1),int(ly1)),(int(lx2),int(ly2)),col,2)
        cv2.imwrite(os.path.join(args.debug_dir, f"debug_{tid}.jpg"), dbg)
    all_lines=sorted(all_lines, key=lambda x: x["length"], reverse=True)
    with open(args.output,"w",encoding="utf-8") as f: json.dump({"image":args.image,"image_size":{"W":W,"H":H,"origin":"top-left"},"tiling":args.tiles,"lines_count":len(all_lines),"lines":all_lines}, f, ensure_ascii=False, indent=2)
    print(f"\nГотово! {len(all_lines)} отрезков (будет много цифр как просил)")
    from collections import Counter; print(Counter(x["type"] for x in all_lines))
    csv=Path(args.output).with_suffix(".csv")
    with open(csv,"w",encoding="utf-8") as f:
        f.write("x1,y1,x2,y2,length,angle,type,tile\n")
        for l in all_lines: f.write(f"{l['x1']},{l['y1']},{l['x2']},{l['y2']},{l['length']},{l['angle']},{l['type']},{l['tile']}\n")
    print(f"CSV: {csv}")

if __name__=="__main__": main()
