import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.split_jpg_sections import (
    GRID_TILES_DIR,
    PLAIN_TILES_DIR,
    apply_soft_grid,
    find_jpg_images,
    process_directory,
    process_image,
    tile_starts,
)


class TileStartsTests(unittest.TestCase):
    def test_overlapping_tiles_cover_the_full_dimension(self) -> None:
        starts = tile_starts(5504, tile_size=1000, overlap=100)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 4504)
        self.assertTrue(all(b > a for a, b in zip(starts, starts[1:])))
        self.assertTrue(all(b - a <= 1000 for a, b in zip(starts, starts[1:])))

    def test_zero_overlap_leaves_only_the_last_tile_smaller(self) -> None:
        self.assertEqual(tile_starts(2100, tile_size=1000, overlap=0), [0, 1000, 2000])

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tile_starts(2000, tile_size=1000, overlap=1000)


class SplitJpgSectionTests(unittest.TestCase):
    def test_each_source_tile_is_saved_plain_and_with_grid(self) -> None:
        image = np.full((1200, 1400, 3), 200, dtype=np.uint8)
        cv2.line(image, (0, 600), (1399, 600), (30, 30, 30), 2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "section.jpg"
            self.assertTrue(cv2.imwrite(str(source), image))

            result = process_image(source, tile_size=1000, overlap=100)

            self.assertEqual(result["output_dir"].name, "section_1000x1000")
            self.assertEqual((result["columns"], result["rows"]), (2, 2))
            self.assertEqual(result["tile_count"], 4)

            plain_paths = sorted(
                (result["output_dir"] / PLAIN_TILES_DIR).glob("*.png")
            )
            grid_paths = sorted(
                (result["output_dir"] / GRID_TILES_DIR).glob("*.png")
            )
            self.assertEqual(len(plain_paths), 4)
            self.assertEqual(len(grid_paths), 4)
            self.assertEqual([path.name for path in plain_paths], [path.name for path in grid_paths])
            self.assertEqual(plain_paths[0].name, "0000_0000_0999_0999.png")

            plain = cv2.imread(str(plain_paths[0]), cv2.IMREAD_COLOR)
            gridded = cv2.imread(str(grid_paths[0]), cv2.IMREAD_COLOR)
            self.assertEqual(plain.shape, (1000, 1000, 3))
            self.assertEqual(gridded.shape, (1000, 1000, 3))
            # Both versions have the requested filename badge in the upper-left.
            self.assertTrue(np.all(plain[0, 0] > 230))
            self.assertTrue(np.all(gridded[0, 0] > 230))
            # The fine grid visibly changes a pixel that is away from the title.
            self.assertFalse(np.array_equal(plain[100, 80], gridded[100, 80]))

            tile_map = cv2.imread(str(result["tile_map"]), cv2.IMREAD_COLOR)
            self.assertEqual(tile_map.shape, (1200, 1400, 3))
            # The map shows a bright pink outline at a tile boundary.
            pink_pixel = tile_map[150, 400]
            self.assertGreater(int(pink_pixel[0]), 100)
            self.assertLess(int(pink_pixel[1]), 100)
            self.assertGreater(int(pink_pixel[2]), 100)

    def test_no_overlap_allows_smaller_right_and_bottom_edge_tiles(self) -> None:
        image = np.full((1050, 1100, 3), 180, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "partial.JPG"
            self.assertTrue(cv2.imwrite(str(source), image))

            result = process_image(source, tile_size=1000, overlap=0)
            paths = sorted((result["output_dir"] / PLAIN_TILES_DIR).glob("*.png"))
            self.assertEqual(len(paths), 4)

            sizes = {
                (cv2.imread(str(path)).shape[1], cv2.imread(str(path)).shape[0])
                for path in paths
            }
            self.assertEqual(sizes, {(1000, 1000), (100, 1000), (1000, 50), (100, 50)})

    def test_find_images_is_case_insensitive_and_can_recurse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            (root / "a.JPG").write_bytes(b"not decoded by this test")
            (nested / "b.jpeg").write_bytes(b"not decoded by this test")
            (root / "ignore.png").write_bytes(b"not a JPG")

            self.assertEqual([path.name for path in find_jpg_images(root)], ["a.JPG"])
            self.assertEqual(
                {path.name for path in find_jpg_images(root, recursive=True)},
                {"a.JPG", "b.jpeg"},
            )

    def test_grid_has_contrastive_minor_dashed_half_and_major_lines(self) -> None:
        source = np.full((100, 100, 3), (185, 195, 225), dtype=np.uint8)
        grid = apply_soft_grid(source)

        # The 10 px line and 50 px major line are visible over the warm paper.
        self.assertFalse(np.array_equal(grid[5, 10], source[5, 10]))
        self.assertFalse(np.array_equal(grid[5, 50], source[5, 50]))
        # The 25 px half-step is dashed: first dash is drawn, then a gap.
        self.assertFalse(np.array_equal(grid[5, 25], source[5, 25]))
        np.testing.assert_array_equal(grid[13, 25], source[13, 25])
        # The coordinate glyphs have a white halo around their colored strokes.
        self.assertTrue(np.any(np.all(grid[25:36, 50:80] == 255, axis=2)))

    def test_directory_batch_creates_one_output_folder_per_jpg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in ("first.jpg", "second.JPG"):
                self.assertTrue(
                    cv2.imwrite(
                        str(root / name),
                        np.full((12, 18, 3), 190, dtype=np.uint8),
                    )
                )

            results = process_directory(
                root, tile_size=10, overlap=0, additional_tile_size=None
            )

            self.assertEqual(len(results), 2)
            self.assertEqual(
                {result["output_dir"].name for result in results},
                {"first_10x10", "second_10x10"},
            )
            self.assertTrue(
                all(
                    len(list((result["output_dir"] / PLAIN_TILES_DIR).glob("*.png")))
                    == 4
                    for result in results
                )
            )
            self.assertTrue(all(result["tile_map"].is_file() for result in results))

    def test_default_batch_also_creates_500px_tile_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "scan.jpg"
            self.assertTrue(
                cv2.imwrite(
                    str(source),
                    np.full((600, 600, 3), 190, dtype=np.uint8),
                )
            )

            results = process_directory(root, overlap=0)

            self.assertEqual(
                {result["output_dir"].name for result in results},
                {"scan_1000x1000", "scan_500x500"},
            )
            by_size = {result["output_dir"].name: result for result in results}
            self.assertEqual(by_size["scan_1000x1000"]["tile_count"], 1)
            self.assertEqual(by_size["scan_500x500"]["tile_count"], 4)
            for result in results:
                for subdir in (PLAIN_TILES_DIR, GRID_TILES_DIR):
                    self.assertEqual(
                        len(list((result["output_dir"] / subdir).glob("*.png"))),
                        result["tile_count"],
                    )
                self.assertTrue(result["tile_map"].is_file())

    def test_overwrite_rebuilds_tile_folders_and_tile_map(self) -> None:
        image = np.full((12, 12, 3), 190, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "section.jpg"
            self.assertTrue(cv2.imwrite(str(source), image))

            first = process_image(source, tile_size=10, overlap=0)
            with self.assertRaises(FileExistsError):
                process_image(source, tile_size=10, overlap=0)
            second = process_image(source, tile_size=10, overlap=0, overwrite=True)

            self.assertEqual(first["tile_map"], second["tile_map"])
            self.assertTrue(second["tile_map"].is_file())

    def test_grid_rejects_non_divisible_steps(self) -> None:
        with self.assertRaises(ValueError):
            apply_soft_grid(
                np.zeros((100, 100, 3), dtype=np.uint8),
                minor_step=10,
                major_step=45,
            )


if __name__ == "__main__":
    unittest.main()
