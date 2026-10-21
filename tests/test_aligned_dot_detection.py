import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.detect_lines import (
    filter_dot_candidates_near_lines,
    group_aligned_dot_candidates,
)
from scripts.detect_tile_lines import Tile, detect_tiles, draw_overlay, write_line_files


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
        self.assertEqual(len(group_aligned_dot_candidates(widely_spaced_points)), 6)
        self.assertEqual(group_aligned_dot_candidates(irregular_points), [])
        self.assertEqual(
            filter_dot_candidates_near_lines(regular_points, [(0, 100, 350, 100)]), []
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
