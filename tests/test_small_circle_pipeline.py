import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.detect_tile_lines import Tile, detect_tiles, draw_overlay, write_line_files


class SmallCirclePipelineTests(unittest.TestCase):
    def test_tiled_circle_metadata_and_overlay_use_global_coordinates(self) -> None:
        tile_image = np.full((500, 500, 3), 220, dtype=np.uint8)
        cv2.circle(tile_image, (120, 200), 8, (10, 10, 10), 2, cv2.LINE_AA)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tile_path = root / "00500_00100_00999_00599.png"
            self.assertTrue(cv2.imwrite(str(tile_path), tile_image))
            tile = Tile(tile_path, x0=500, y0=100, x1=999, y1=599)

            records = detect_tiles([tile], width=1000, height=700)
            circles = [record for record in records if record["type"] == "small_circle"]
            self.assertEqual(len(circles), 1)
            circle = circles[0]
            self.assertLess(np.hypot(circle["x1"] - 620, circle["y1"] - 300), 2)
            self.assertEqual((circle["x1"], circle["y1"]), (circle["x2"], circle["y2"]))
            self.assertGreaterEqual(circle["radius"], 7)
            self.assertLessEqual(circle["radius"], 12)

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
            json_circle = next(
                record for record in payload["lines"] if record["type"] == "small_circle"
            )
            self.assertEqual(json_circle["radius"], circle["radius"])

            with csv_path.open(encoding="utf-8-sig", newline="") as file:
                csv_circle = next(
                    row for row in csv.DictReader(file) if row["type"] == "small_circle"
                )
            self.assertAlmostEqual(float(csv_circle["radius"]), circle["radius"])

            source_image = np.full((700, 1000, 3), 220, dtype=np.uint8)
            overlay_path, preview_path = draw_overlay(
                source_image, records, preview_max_side=500, output_dir=output_directory
            )
            overlay = cv2.imread(str(overlay_path))
            yellow_pixel = overlay[circle["y1"], circle["x1"] + round(circle["radius"])]
            self.assertGreater(int(yellow_pixel[1]), 150)
            self.assertGreater(int(yellow_pixel[2]), 150)
            self.assertTrue(preview_path.is_file())


if __name__ == "__main__":
    unittest.main()
