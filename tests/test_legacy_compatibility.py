"""Гарантии обратной совместимости старого пайплайна 500×500.

Старые скрипты (`split_scan_tiles.py`, `detect_lines.py`,
`detect_tile_lines.py`) и их публичные функции должны работать без изменений,
а новый режим сетки не должен менять их результаты.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.detect_tile_lines import Tile, detect_tiles
from scripts.vectorize_scan import split_into_grid


class LegacyDetectTilesSignatureTests(unittest.TestCase):
    def make_tiles(self, root: Path) -> list[Tile]:
        """Create four 500×500 tiles with horizontal ink, as the old splitter does."""
        tiles = []
        for row in range(2):
            for column in range(2):
                image = np.full((500, 500, 3), 235, dtype=np.uint8)
                cv2.line(
                    image,
                    (20, 150 + 100 * row),
                    (480, 150 + 100 * row),
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )
                x0, y0 = column * 500, row * 500
                path = root / f"{x0:05d}_{y0:05d}_{x0 + 499:05d}_{y0 + 499:05d}.png"
                self.assertTrue(cv2.imwrite(str(path), image))
                tiles.append(Tile(path, x0, y0, x0 + 499, y0 + 499))
        return tiles

    def test_old_positional_calls_still_work_and_match_new_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tiles = self.make_tiles(Path(temporary_directory))

            # Сигнатуры, которые использовались до появления новых параметров.
            three_args = detect_tiles(tiles, 1000, 1000)
            four_args = detect_tiles(tiles, 1000, 1000, 500)
            explicit_defaults = detect_tiles(
                tiles,
                1000,
                1000,
                500,
                grouping_stride=None,
                workers=1,
                report_every=None,
            )

            self.assertGreater(len(three_args), 0)
            self.assertEqual(three_args, four_args)
            self.assertEqual(three_args, explicit_defaults)

    def test_new_grid_mode_does_not_change_legacy_tile_names(self) -> None:
        image = np.full((400, 1000, 3), 240, dtype=np.uint8)
        cv2.line(image, (0, 200), (999, 200), (20, 20, 20), 2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "scan.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            tiles, _, _ = split_into_grid(source, root / "out", columns=10, rows=2)
            for tile in tiles:
                self.assertRegex(
                    tile.path.name,
                    r"^\d+_\d+_\d+_\d+\.png$",
                    "имена тайлов должны остаться в схеме x0_y0_x1_y1.png",
                )


if __name__ == "__main__":
    unittest.main()
