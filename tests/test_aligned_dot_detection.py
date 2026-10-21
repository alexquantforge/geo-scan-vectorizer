import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.detect_tile_lines import Tile, detect_tiles, draw_overlay, write_line_files


class AlignedDotDetectionTests(unittest.TestCase):
    def test_tiled_pipeline_marks_every_dot_in_rows_of_five_or_more(self) -> None:
        tile_image = np.full((500, 500, 3), 220, dtype=np.uint8)
        aligned_centers = [
            (50, 50),
            (90, 90),
            (130, 130),
            (170, 171),
            (210, 210),
        ]
        four_point_row = [(60, 350), (105, 350), (150, 350), (195, 350)]
        for center in aligned_centers + four_point_row:
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

    def test_row_crossing_tile_boundary_is_grouped_globally(self) -> None:
        left_image = np.full((500, 500, 3), 220, dtype=np.uint8)
        right_image = np.full((500, 500, 3), 220, dtype=np.uint8)
        global_centers = [(400, 100), (455, 127), (510, 154), (565, 181), (620, 208)]
        for global_x, y in global_centers:
            image = left_image if global_x < 500 else right_image
            x = global_x if global_x < 500 else global_x - 500
            cv2.circle(image, (x, y), 6, (10, 10, 10), 2, cv2.LINE_AA)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left_path = root / "00000_00000_00499_00499.png"
            right_path = root / "00500_00000_00999_00499.png"
            self.assertTrue(cv2.imwrite(str(left_path), left_image))
            self.assertTrue(cv2.imwrite(str(right_path), right_image))
            tiles = [
                Tile(left_path, x0=0, y0=0, x1=499, y1=499),
                Tile(right_path, x0=500, y0=0, x1=999, y1=499),
            ]

            records = detect_tiles(tiles, width=1000, height=500)
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
