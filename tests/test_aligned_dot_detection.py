import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np

from scripts.detect_lines import (
    detect_dot_candidates,
    filter_dot_candidates_near_lines,
    group_aligned_dot_candidates,
    merge_dot_candidates,
)
from scripts.detect_tile_lines import (
    Tile,
    _TiledDotCandidateGray,
    detect_tiles,
    draw_overlay,
    group_dot_candidates_in_windows,
    write_line_files,
)


class AlignedDotDetectionTests(unittest.TestCase):
    def test_equal_spacing_and_distance_from_existing_line_are_required(self) -> None:
        regular_points = [
            {"x": float(x), "y": 100.0, "radius": 5.0, "area": 30}
            for x in (30, 80, 130, 180, 230, 280)
        ]
        irregular_points = [
            {"x": float(x), "y": 180.0, "radius": 5.0, "area": 30}
            for x in (30, 70, 130, 180, 250, 300)
        ]
        too_close_points = [
            {"x": float(x), "y": 220.0, "radius": 5.0, "area": 30}
            for x in (30, 55, 80, 105, 130, 155)
        ]
        minimum_spacing_points = [
            {"x": float(x), "y": 260.0, "radius": 5.0, "area": 30}
            for x in (30, 70, 110, 150, 190, 230)
        ]
        widely_spaced_points = [
            {"x": x, "y": y, "radius": 5.0, "area": 30}
            for x, y in (
                (19.5, 0.0),
                (92.5, 129.0),
                (167.0, 258.4),
                (224.1, 363.7),
                (285.3, 470.0),
                (362.8, 601.5),
            )
        ]

        self.assertEqual(len(group_aligned_dot_candidates(regular_points)), 6)
        self.assertEqual(len(group_aligned_dot_candidates(minimum_spacing_points)), 6)
        self.assertEqual(len(group_aligned_dot_candidates(widely_spaced_points)), 6)
        self.assertEqual(group_aligned_dot_candidates(too_close_points), [])
        self.assertEqual(group_aligned_dot_candidates(irregular_points), [])
        self.assertEqual(
            filter_dot_candidates_near_lines(regular_points, [(0, 100, 350, 100)]), []
        )

    def test_global_grouping_reports_window_progress(self) -> None:
        candidates = [
            {"x": float(x), "y": 100.0, "radius": 5.0, "area": 30}
            for x in (30, 80, 130, 180, 230, 280)
        ]
        output = StringIO()
        with redirect_stdout(output):
            aligned_dots = group_dot_candidates_in_windows(
                candidates, tile_size=500, report_progress=True
            )

        self.assertEqual(len(aligned_dots), 6)
        self.assertIn("Глобальная группировка точек", output.getvalue())
        self.assertIn("окно 1/", output.getvalue())
        self.assertIn("готово; найдено точек=6", output.getvalue())

    def test_point_filter_reports_index_and_candidate_progress(self) -> None:
        candidates = [
            {"x": float(x), "y": 100.0, "radius": 5.0, "area": 30}
            for x in (30, 80, 130, 180, 230, 280)
        ]
        index_progress = []
        candidate_progress = []

        filtered = filter_dot_candidates_near_lines(
            candidates,
            [(0.0, 100.0, 350.0, 100.0)],
            progress_callback=lambda done, total: candidate_progress.append(
                (done, total)
            ),
            index_progress_callback=lambda done, total: index_progress.append(
                (done, total)
            ),
        )

        self.assertEqual(filtered, [])
        self.assertEqual(index_progress[-1], (1, 1))
        self.assertEqual(candidate_progress[-1], (6, 6))

    def test_dot_candidates_keep_a_clearance_from_detected_lines(self) -> None:
        near_line = {"x": 100.0, "y": 115.0, "radius": 5.0, "area": 30}
        clear_of_line = {"x": 200.0, "y": 121.0, "radius": 5.0, "area": 30}

        filtered = filter_dot_candidates_near_lines(
            [near_line, clear_of_line], [(0.0, 100.0, 300.0, 100.0)]
        )

        self.assertEqual(filtered, [clear_of_line])

    def test_dotted_marks_on_line_are_kept_but_solid_line_is_rejected(self) -> None:
        source = np.full((180, 360, 3), 225, dtype=np.uint8)
        centers = [(40 + 50 * index, 90) for index in range(6)]
        for center in centers:
            cv2.circle(source, center, 4, (90, 90, 90), -1, cv2.LINE_AA)
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        candidates = detect_dot_candidates(source, gray)
        line_segment = [(40.0, 90.0, 290.0, 90.0)]

        kept_dots = filter_dot_candidates_near_lines(
            candidates, line_segment, gray_image=gray
        )
        self.assertEqual(len(candidates), 6)
        self.assertEqual(len(kept_dots), 6)
        self.assertEqual(len(group_aligned_dot_candidates(kept_dots)), 6)

        solid_gray = gray.copy()
        cv2.line(solid_gray, (20, 90), (310, 90), 20, 2, cv2.LINE_AA)
        self.assertEqual(
            filter_dot_candidates_near_lines(
                candidates, line_segment, gray_image=solid_gray
            ),
            [],
        )

    def test_global_line_filter_reads_support_across_tile_seams(self) -> None:
        source = np.full((80, 100, 3), 225, dtype=np.uint8)
        cv2.line(source, (20, 40), (80, 40), (20, 20, 20), 2, cv2.LINE_AA)
        candidate = {"x": 50.0, "y": 40.0, "radius": 4.5, "area": 30}

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tiles = []
            for x0, x1 in ((0, 49), (50, 99)):
                path = root / f"{x0:05d}_00000_{x1:05d}_00079.png"
                self.assertTrue(cv2.imwrite(str(path), source[:, x0 : x1 + 1]))
                tiles.append(Tile(path, x0, 0, x1, 79))

            gray_source = _TiledDotCandidateGray(tiles, width=100, height=80)
            filtered = filter_dot_candidates_near_lines(
                [candidate], [(20.0, 40.0, 80.0, 40.0)], gray_image=gray_source
            )

        self.assertEqual(filtered, [])

    def test_spatial_merge_keeps_transitive_clusters_and_threshold(self) -> None:
        candidates = [
            {"x": float(x), "y": 0.0, "radius": 3.0, "area": 1}
            for x in (0.0, 8.0, 17.0, 30.0, 39.1)
        ]

        merged = merge_dot_candidates(candidates)

        self.assertEqual(len(merged), 3)
        self.assertAlmostEqual(merged[0]["x"], 25.0 / 3.0)
        self.assertEqual([point["x"] for point in merged[1:]], [30.0, 39.1])

    def test_parallel_rows_allow_skipped_marks_but_keep_six_point_minimum(self) -> None:
        direction = np.array([0.4, 0.916515], dtype=np.float64)
        direction /= np.linalg.norm(direction)

        def make_row(origin: tuple[float, float], spacing: float, slots: list[int]):
            candidates = []
            for index, slot in enumerate(slots):
                center = np.asarray(origin) + direction * spacing * slot
                jitter = np.array([(-1) ** index * 0.8, (index % 3 - 1) * 0.6])
                candidates.append(
                    {
                        "x": float(center[0] + jitter[0]),
                        "y": float(center[1] + jitter[1]),
                        "radius": 5.0,
                        "area": 30,
                    }
                )
            return candidates

        first_row = make_row((250, 100), 60, [0, 1, 4, 5, 6, 9, 10])
        parallel_row = make_row((140, 148), 70, list(range(8)))
        five_point_decoy = make_row((400, 10), 55, list(range(5)))
        candidates = first_row + parallel_row + five_point_decoy

        aligned_dots = group_aligned_dot_candidates(candidates)
        self.assertEqual(len(aligned_dots), len(first_row) + len(parallel_row))
        for expected in first_row + parallel_row:
            self.assertLess(
                min(
                    np.hypot(dot["x"] - expected["x"], dot["y"] - expected["y"])
                    for dot in aligned_dots
                ),
                2.0,
            )
        self.assertTrue(
            all(
                min(
                    np.hypot(dot["x"] - decoy["x"], dot["y"] - decoy["y"])
                    for dot in aligned_dots
                )
                > 2.0
                for decoy in five_point_decoy
            )
        )

    def test_dense_parallel_rows_are_grouped_without_pairwise_fallback(self) -> None:
        direction = np.array([0.6, 0.8], dtype=np.float64)
        normal = np.array([-direction[1], direction[0]])
        candidates = []
        for row in range(10):
            for column in range(15):
                center = np.array([800.0, 800.0]) + direction * 70 * column
                center += normal * 100 * row
                candidates.append(
                    {
                        "x": float(center[0]),
                        "y": float(center[1]),
                        "radius": 5.0,
                        "area": 30,
                    }
                )

        self.assertEqual(len(group_aligned_dot_candidates(candidates)), 150)

    def test_tiled_pipeline_groups_a_periodic_row_with_two_missing_marks(self) -> None:
        source_image = np.full((900, 1000, 3), 220, dtype=np.uint8)
        direction = np.array([0.4, 0.916515], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        origin = np.array([320.0, 50.0])
        global_centers = [
            tuple(np.rint(origin + direction * 60 * slot).astype(int))
            for slot in (0, 1, 4, 5, 6, 9, 10)
        ]
        for center in global_centers:
            cv2.circle(source_image, center, 6, (10, 10, 10), 2, cv2.LINE_AA)
        ring_center = (100, 800)
        cv2.circle(source_image, ring_center, 8, (10, 10, 10), 2, cv2.LINE_AA)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tiles = []
            for y0 in (0, 500):
                for x0 in (0, 500):
                    x1 = min(x0 + 500, 1000) - 1
                    y1 = min(y0 + 500, 900) - 1
                    tile_path = root / f"{x0:05d}_{y0:05d}_{x1:05d}_{y1:05d}.png"
                    tile_image = source_image[y0 : y1 + 1, x0 : x1 + 1]
                    self.assertTrue(cv2.imwrite(str(tile_path), tile_image))
                    tiles.append(Tile(tile_path, x0, y0, x1, y1))

            records = detect_tiles(tiles, width=1000, height=900)
            aligned_dots = [
                record for record in records if record["type"] == "aligned_dot"
            ]
            self.assertEqual(len(aligned_dots), len(global_centers))
            actual_centers = {(record["x1"], record["y1"]) for record in aligned_dots}
            self.assertEqual(actual_centers, set(global_centers))
            small_circles = [
                record for record in records if record["type"] == "small_circle"
            ]
            self.assertEqual(len(small_circles), 1)
            self.assertLess(
                np.hypot(
                    small_circles[0]["x1"] - ring_center[0],
                    small_circles[0]["y1"] - ring_center[1],
                ),
                2.0,
            )
            self.assertTrue(
                all(
                    0 <= record["x1"] < 1000 and 0 <= record["y1"] < 900
                    for record in aligned_dots
                )
            )

    def test_tiled_pipeline_requires_six_regular_points(self) -> None:
        tile_image = np.full((500, 500, 3), 220, dtype=np.uint8)
        aligned_centers = [
            (50, 50),
            (90, 90),
            (130, 130),
            (170, 171),
            (210, 210),
            (250, 250),
        ]
        five_point_row = [(50, 350), (100, 350), (150, 350), (200, 350), (250, 350)]
        four_point_row = [(60, 430), (110, 430), (160, 430), (210, 430)]
        for center in aligned_centers + five_point_row + four_point_row:
            cv2.circle(tile_image, center, 6, (10, 10, 10), 2, cv2.LINE_AA)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tile_path = root / "00500_00100_00999_00599.png"
            self.assertTrue(cv2.imwrite(str(tile_path), tile_image))
            tile = Tile(tile_path, x0=500, y0=100, x1=999, y1=599)

            records = detect_tiles([tile], width=1000, height=700)
            aligned_dots = [
                record for record in records if record["type"] == "aligned_dot"
            ]
            self.assertEqual(len(aligned_dots), len(aligned_centers))

            expected_global_centers = [
                (x + 500, y + 100) for x, y in aligned_centers
            ]
            actual_centers = {(record["x1"], record["y1"]) for record in aligned_dots}
            for expected_x, expected_y in expected_global_centers:
                self.assertLess(
                    min(
                        np.hypot(record["x1"] - expected_x, record["y1"] - expected_y)
                        for record in aligned_dots
                    ),
                    2.0,
                )
            repeated_centers = {
                (record["x2"], record["y2"]) for record in aligned_dots
            }
            self.assertEqual(actual_centers, repeated_centers)
            self.assertTrue(all(record["radius"] is not None for record in aligned_dots))
            self.assertTrue(all(record["tile"] == tile_path.name for record in aligned_dots))

            output_directory = root / "output"
            output_directory.mkdir()
            json_path, csv_path = write_line_files(
                output_directory,
                image_path=root / "source.png",
                tiles_dir=root / "tiles",
                width=1000,
                height=700,
                tiles_count=1,
                lines=records,
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["counts_by_type"]["aligned_dot"], len(aligned_centers)
            )
            with csv_path.open(encoding="utf-8-sig", newline="") as file:
                csv_dots = [
                    row for row in csv.DictReader(file) if row["type"] == "aligned_dot"
                ]
            self.assertEqual(len(csv_dots), len(aligned_centers))

            source_image = np.full((700, 1000, 3), 220, dtype=np.uint8)
            source_image[100:600, 500:1000] = tile_image
            overlay_path, preview_path = draw_overlay(
                source_image, records, preview_max_side=500, output_dir=output_directory
            )
            overlay = cv2.imread(str(overlay_path))
            first_dot = aligned_dots[0]
            outline_x = first_dot["x1"] + round(first_dot["radius"])
            outline_pixel = overlay[first_dot["y1"], outline_x]
            self.assertGreater(int(outline_pixel.min()), 230)
            self.assertTrue(preview_path.is_file())

    def test_widely_spaced_row_crossing_tile_boundaries_is_grouped_globally(self) -> None:
        source_image = np.full((750, 1000, 3), 220, dtype=np.uint8)
        global_centers = [
            (300, 50),
            (372, 175),
            (445, 300),
            (517, 425),
            (590, 550),
            (663, 675),
        ]
        for center in global_centers:
            cv2.circle(source_image, center, 6, (10, 10, 10), 2, cv2.LINE_AA)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tiles = []
            for y0 in (0, 500):
                for x0 in (0, 500):
                    x1 = min(x0 + 500, 1000) - 1
                    y1 = min(y0 + 500, 750) - 1
                    tile_path = root / f"{x0:05d}_{y0:05d}_{x1:05d}_{y1:05d}.png"
                    tile_image = source_image[y0 : y1 + 1, x0 : x1 + 1]
                    self.assertTrue(cv2.imwrite(str(tile_path), tile_image))
                    tiles.append(Tile(tile_path, x0, y0, x1, y1))

            records = detect_tiles(tiles, width=1000, height=750)
            aligned_dots = [
                record for record in records if record["type"] == "aligned_dot"
            ]
            self.assertEqual(len(aligned_dots), len(global_centers))
            for expected_x, expected_y in global_centers:
                self.assertLess(
                    min(
                        np.hypot(record["x1"] - expected_x, record["y1"] - expected_y)
                        for record in aligned_dots
                    ),
                    2.0,
                )


if __name__ == "__main__":
    unittest.main()
