import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.vectorize_scan import (
    detect_grid,
    grid_boundaries,
    prepare_output_directory,
    split_into_grid,
    vectorize,
)


class GridBoundariesTests(unittest.TestCase):
    def test_boundaries_cover_the_image_without_gaps(self) -> None:
        for length, parts in ((24030, 10), (13244, 2), (1000, 7), (5, 5)):
            edges = grid_boundaries(length, parts)
            self.assertEqual(len(edges), parts + 1)
            self.assertEqual(edges[0], 0)
            self.assertEqual(edges[-1], length)
            self.assertTrue(all(b > a for a, b in zip(edges, edges[1:])))

    def test_rejects_more_parts_than_pixels(self) -> None:
        with self.assertRaises(ValueError):
            grid_boundaries(3, 4)


class GridSplitTests(unittest.TestCase):
    def test_default_layout_creates_ten_top_and_ten_bottom_tiles(self) -> None:
        width, height = 1000, 400
        image = np.full((height, width, 3), 240, dtype=np.uint8)
        cv2.line(image, (0, 200), (width - 1, 200), (20, 20, 20), 2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "scan.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            output = root / "result"

            tiles, found_width, found_height = split_into_grid(
                source, output, columns=10, rows=2
            )

            self.assertEqual((found_width, found_height), (width, height))
            self.assertEqual(len(tiles), 20)
            self.assertEqual(len([tile for tile in tiles if tile.row == 1]), 10)
            self.assertEqual(len([tile for tile in tiles if tile.row == 2]), 10)
            self.assertEqual(
                len(list((output / "tiles" / "row_01").glob("*.png"))), 10
            )
            self.assertEqual(
                len(list((output / "tiles" / "row_02").glob("*.png"))), 10
            )

            covered = 0
            for tile in tiles:
                tile_image = cv2.imread(str(tile.path))
                self.assertIsNotNone(tile_image)
                self.assertEqual(
                    (tile_image.shape[1], tile_image.shape[0]),
                    (tile.width, tile.height),
                )
                covered += tile.width * tile.height
            self.assertEqual(covered, width * height)

            top = [tile for tile in tiles if tile.row == 1]
            self.assertEqual([tile.column for tile in top], list(range(1, 11)))
            self.assertEqual(top[0].x0, 0)
            self.assertEqual(top[-1].x1, width - 1)
            self.assertEqual(top[1].x0, top[0].x1 + 1)

    def test_overwrite_refuses_to_delete_foreign_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result"
            output.mkdir()
            (output / "notes.txt").write_text("keep me", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_directory(output, overwrite=True)
            self.assertTrue((output / "notes.txt").is_file())


class VectorizePipelineTests(unittest.TestCase):
    @staticmethod
    def make_scan() -> np.ndarray:
        width, height = 1200, 800
        image = np.full((height, width, 3), 242, dtype=np.uint8)
        for y in (140, 260):
            cv2.line(image, (30, y), (width - 30, y), (20, 20, 20), 2, cv2.LINE_AA)
        cv2.circle(image, (900, 560), 9, (10, 10, 10), 2, cv2.LINE_AA)
        return image

    def test_pipeline_writes_tiles_rows_tables_and_overlays_to_subfolders(
        self,
    ) -> None:
        image = self.make_scan()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "scan.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            output = root / "scan_vectorized"

            result = vectorize(
                source,
                output_dir=output,
                columns=4,
                rows=2,
                preview_max_side=500,
            )

            tiles = result["tiles"]
            self.assertEqual(len(tiles), 8)
            self.assertEqual(len(result["tile_overlays"]), 8)
            self.assertEqual(len(result["row_strips"]), 2)

            for row_name in ("row_01", "row_02"):
                self.assertEqual(
                    len(list((output / "tiles" / row_name).glob("*.png"))), 4
                )
                self.assertEqual(
                    len(list((output / "overlays" / "tiles" / row_name).glob("*.jpg"))),
                    4,
                )
                self.assertEqual(
                    len(list((output / "tables" / "tiles" / row_name).glob("*.csv"))),
                    4,
                )
                self.assertEqual(
                    len(list((output / "tables" / "tiles" / row_name).glob("*.json"))),
                    4,
                )
                self.assertTrue((output / "rows" / f"{row_name}.jpg").is_file())
                self.assertTrue((output / "tables" / "rows" / f"{row_name}.csv").is_file())
                self.assertTrue((output / "tables" / "rows" / f"{row_name}.json").is_file())

            self.assertTrue((output / "tables" / "lines.csv").is_file())
            self.assertTrue((output / "tables" / "lines.json").is_file())
            self.assertTrue((output / "tables" / "tiles_summary.csv").is_file())
            self.assertTrue((output / "overlays" / "overlay_full.jpg").is_file())
            self.assertTrue((output / "overlays" / "overlay_preview.jpg").is_file())
            self.assertTrue((output / "run.json").is_file())

            strip = cv2.imread(str(output / "rows" / "row_01.jpg"))
            self.assertIsNotNone(strip)
            self.assertEqual(strip.shape[1], image.shape[1])

            full_overlay = cv2.imread(str(output / "overlays" / "overlay_full.jpg"))
            self.assertEqual(full_overlay.shape, image.shape)
            preview = cv2.imread(str(output / "overlays" / "overlay_preview.jpg"))
            self.assertEqual(max(preview.shape[:2]), 500)

            with (output / "tables" / "tiles_summary.csv").open(
                encoding="utf-8-sig", newline=""
            ) as file:
                summary_rows = list(csv.DictReader(file))
            self.assertEqual(len(summary_rows), 8)
            self.assertEqual(
                sum(int(row["total"]) for row in summary_rows),
                result["lines_count"],
            )

            combined = json.loads(
                (output / "tables" / "lines.json").read_text(encoding="utf-8")
            )
            self.assertEqual(combined["tile_count"], 8)
            self.assertEqual(combined["grid"], {"columns": 4, "rows": 2})
            self.assertEqual(combined["lines_count"], result["lines_count"])

            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metadata["tiles"]), 8)
            self.assertEqual(metadata["grid"]["tile_count"], 8)

            known_tiles = {tile.path.name: tile for tile in tiles}
            for tile in tiles:
                payload = json.loads(
                    (
                        output
                        / "tables"
                        / "tiles"
                        / tile.row_name
                        / f"{tile.path.stem}.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(payload["tile"]["name"], tile.path.name)
                for line in payload["lines"]:
                    self.assertEqual(line["tile"], tile.path.name)
                    self.assertEqual(line["x1_local"], line["x1"] - tile.x0)
                    self.assertEqual(line["y1_local"], line["y1"] - tile.y0)
                    self.assertGreaterEqual(line["x1_local"], 0)
                    self.assertLessEqual(line["x1_local"], tile.width)
                    self.assertGreaterEqual(line["y1_local"], 0)
                    self.assertLessEqual(line["y1_local"], tile.height)

            # Every detected object belongs to exactly one existing tile table.
            total_from_tables = 0
            for name in known_tiles:
                table = next(
                    (output / "tables" / "tiles").glob(f"row_*/{Path(name).stem}.csv")
                )
                with table.open(encoding="utf-8-sig", newline="") as file:
                    total_from_tables += len(list(csv.DictReader(file)))
            self.assertEqual(total_from_tables, result["lines_count"])
            self.assertGreater(result["lines_count"], 0)


class ParallelDetectionTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "fork"), "параллельный режим требует fork")
    def test_workers_produce_the_same_records_as_sequential_run(self) -> None:
        width, height = 800, 400
        image = np.full((height, width, 3), 240, dtype=np.uint8)
        for y in (120, 260):
            cv2.line(image, (20, y), (width - 20, y), (20, 20, 20), 2, cv2.LINE_AA)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "scan.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            output = root / "result"
            tiles, found_width, found_height = split_into_grid(
                source, output, columns=4, rows=2
            )
            sequential = detect_grid(tiles, found_width, found_height, workers=1)
            parallel = detect_grid(tiles, found_width, found_height, workers=2)

            self.assertGreater(len(sequential), 0)
            self.assertEqual(
                [line for line in sequential if "tile" in line],
                [line for line in parallel if "tile" in line],
            )


if __name__ == "__main__":
    unittest.main()
