#!/usr/bin/env python3
"""Split JPG geological sections into coordinate-named PNG tiles.

For each source image, create ``<stem>_<tile-size>x<tile-size>/`` beside it,
with plain tiles in ``tiles/``, soft-grid copies in ``tiles_with_grid/``, and
``tile_map.jpg`` showing pink tile outlines and coordinate filenames. The grid
is local to each tile: 10 px solid, 25 px dashed, and 50 px major lines.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    raise SystemExit(
        "Не найден OpenCV или NumPy. Установите зависимости: "
        "python -m pip install -r requirements-detection.txt"
    ) from exc


DEFAULT_INPUT_DIR = Path(
    "/home/alex/2025-12-03-1636-backup/PycharmProjects/geo-scan-vectorizer/"
    "Графика_разрезы/new_project"
)
DEFAULT_TILE_SIZE = 1000
DEFAULT_ADDITIONAL_TILE_SIZE = 500
DEFAULT_OVERLAP = 100
PLAIN_TILES_DIR = "tiles"
GRID_TILES_DIR = "tiles_with_grid"
OUTPUT_SUBDIRS = (PLAIN_TILES_DIR, GRID_TILES_DIR)
OUTPUT_FILES = ("tile_map.jpg",)

GRID_MINOR_STEP = 10
GRID_HALF_STEP = 25
GRID_MAJOR_STEP = 50
GRID_DASH_LENGTH = 8
GRID_DASH_GAP = 8
GRID_MINOR_ALPHA = 0.38
GRID_HALF_ALPHA = 0.45
GRID_MAJOR_ALPHA = 0.35

# OpenCV colors use BGR order. The requested minor RGB (0, 110, 180) becomes
# BGR (180, 110, 0); the half-step is a dark indigo-purple.
MINOR_GRID_COLOR = (180, 110, 0)
HALF_GRID_COLOR = (100, 45, 65)
MAJOR_GRID_COLOR = (255, 0, 255)  # magenta
Y_COORDINATE_COLOR = (180, 45, 0)  # deep blue
TILE_MAP_COLOR = (203, 0, 255)  # vivid pink


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидается целое число") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return result


def non_negative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидается целое число") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("значение не может быть отрицательным")
    return result


def find_jpg_images(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Find JPG/JPEG files directly in ``input_dir`` (or recursively)."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"не найдена папка с изображениями: {input_dir}")

    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        ),
        key=lambda path: str(path).casefold(),
    )


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Calculate tile origins that cover the full dimension without gaps.

    For overlapping tiles, distribute the starts across the full-size range.
    This avoids an almost-duplicate final tile when the image edge falls just
    beyond a regular stride. The requested overlap is used when the geometry
    allows it; otherwise the fewest full-size tiles are used. With
    ``overlap=0``, the last tile may be smaller instead of overlapping.
    """
    if length <= 0:
        raise ValueError("размер изображения должен быть больше нуля")
    if tile_size <= 0:
        raise ValueError("размер тайла должен быть больше нуля")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("перекрытие должно быть от 0 до размера тайла - 1")

    if length <= tile_size:
        return [0]
    if overlap == 0:
        return list(range(0, length, tile_size))

    last_start = length - tile_size
    stride = tile_size - overlap
    min_intervals = (last_start + tile_size - 1) // tile_size
    max_intervals = last_start // stride
    intervals = max_intervals if max_intervals >= min_intervals else min_intervals
    return [round(index * last_start / intervals) for index in range(intervals + 1)]


def coordinate_tile_name(
    x0: int, y0: int, x1: int, y1: int, digits: int
) -> str:
    """Use the repository's inclusive ``x0_y0_x1_y1.png`` naming scheme."""
    return (
        f"{x0:0{digits}d}_{y0:0{digits}d}_"
        f"{x1:0{digits}d}_{y1:0{digits}d}.png"
    )


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    """Create output folders, refusing to remove files not made by this script."""
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"путь результата не является папкой: {output_dir}")

    if output_dir.exists():
        entries = list(output_dir.iterdir())
        allowed_entries = set(OUTPUT_SUBDIRS) | set(OUTPUT_FILES)
        unexpected = [entry for entry in entries if entry.name not in allowed_entries]
        if unexpected:
            names = ", ".join(entry.name for entry in unexpected[:5])
            raise FileExistsError(
                f"не буду изменять папку с посторонними файлами: {output_dir} "
                f"(например: {names})"
            )
        if entries and not overwrite:
            raise FileExistsError(
                f"папка уже содержит результаты: {output_dir}\n"
                "Чтобы пересоздать их, добавьте --overwrite."
            )
        if overwrite:
            for entry in entries:
                if entry.is_symlink() or not entry.is_dir():
                    entry.unlink()
                else:
                    shutil.rmtree(entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_SUBDIRS:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def draw_filename_label(
    image: Any,
    filename: str,
    x: int = 0,
    y: int = 0,
    max_width: int | None = None,
    max_height: int | None = None,
) -> Any:
    """Draw a small filename badge at ``(x, y)`` without changing other pixels."""
    image_height, image_width = image.shape[:2]
    left = max(0, min(x, image_width - 1))
    top = max(0, min(y, image_height - 1))
    allowed_width = min(
        image_width - left,
        max_width if max_width is not None else image_width - left,
    )
    allowed_height = min(
        image_height - top,
        max_height if max_height is not None else image_height - top,
    )
    if allowed_width <= 0 or allowed_height <= 0:
        return image

    font = cv2.FONT_HERSHEY_PLAIN
    scale = 0.8
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        filename, font, scale, thickness
    )
    max_text_width = max(1, allowed_width - 6)
    max_text_height = max(1, allowed_height - 6)
    if text_width > max_text_width:
        scale *= max_text_width / text_width
    if text_height + baseline > max_text_height:
        scale *= max_text_height / max(text_height + baseline, 1)
    (text_width, text_height), baseline = cv2.getTextSize(
        filename, font, scale, thickness
    )

    badge_width = min(allowed_width, text_width + 6)
    badge_height = min(allowed_height, text_height + baseline + 6)
    badge = image[top : top + badge_height, left : left + badge_width]
    white = np.full_like(badge, 255)
    cv2.addWeighted(white, 0.82, badge, 0.18, 0, dst=badge)

    baseline_y = min(top + badge_height - 1, top + text_height + 3)
    cv2.putText(
        image,
        filename,
        (left + 3, baseline_y),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return image


def add_filename_label(image: Any, filename: str) -> Any:
    """Draw a small, readable tile filename in the upper-left corner."""
    return draw_filename_label(image, filename)


def write_tile_map(
    image: Any,
    tiles: list[tuple[int, int, int, int, str]],
    output_path: Path,
) -> Path:
    """Draw pink tile boundaries and tile filenames into the scan buffer."""
    # The source array is no longer needed after cropping, so drawing in place
    # avoids allocating a second full-size image for large geological scans.
    for x0, y0, x1, y1, _filename in tiles:
        cv2.rectangle(
            image,
            (x0, y0),
            (x1, y1),
            TILE_MAP_COLOR,
            3,
            cv2.LINE_AA,
        )

    for x0, y0, x1, y1, filename in tiles:
        draw_filename_label(
            image,
            filename,
            x=x0 + 4,
            y=y0 + 4,
            max_width=max(1, x1 - x0 - 6),
            max_height=max(1, y1 - y0 - 6),
        )

    success, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    if not success:
        raise OSError(f"не удалось закодировать карту разбивки: {output_path}")
    output_path.write_bytes(encoded.tobytes())
    return output_path


def draw_dashed_grid_line(
    image: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    dash_length: int = GRID_DASH_LENGTH,
    dash_gap: int = GRID_DASH_GAP,
) -> None:
    """Draw one horizontal or vertical dashed segment into ``image``."""
    x1, y1 = start
    x2, y2 = end
    if y1 == y2:
        line_length = abs(x2 - x1)
        direction = 1 if x2 >= x1 else -1
        for offset in range(0, line_length + 1, dash_length + dash_gap):
            end_offset = min(offset + dash_length - 1, line_length)
            cv2.line(
                image,
                (x1 + direction * offset, y1),
                (x1 + direction * end_offset, y1),
                color,
                1,
                cv2.LINE_AA,
            )
    elif x1 == x2:
        line_length = abs(y2 - y1)
        direction = 1 if y2 >= y1 else -1
        for offset in range(0, line_length + 1, dash_length + dash_gap):
            end_offset = min(offset + dash_length - 1, line_length)
            cv2.line(
                image,
                (x1, y1 + direction * offset),
                (x1, y1 + direction * end_offset),
                color,
                1,
                cv2.LINE_AA,
            )
    else:
        raise ValueError("линия сетки должна быть горизонтальной или вертикальной")


def put_coordinate_label(
    image: Any,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.8,
) -> None:
    """Draw a coordinate with a one-pixel white halo for contrast."""
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_PLAIN,
        scale,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_PLAIN,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def apply_soft_grid(
    image: Any,
    filename: str | None = None,
    minor_step: int = GRID_MINOR_STEP,
    half_step: int = GRID_HALF_STEP,
    major_step: int = GRID_MAJOR_STEP,
) -> Any:
    """Draw contrastive 10 px, dashed 25 px and magenta 50 px grid lines."""
    if minor_step <= 0 or half_step <= 0 or major_step <= 0:
        raise ValueError("шаг сетки должен быть больше нуля")
    if major_step % minor_step != 0 or major_step % half_step != 0:
        raise ValueError("основной шаг должен делиться на мелкий и полушаг без остатка")

    height, width = image.shape[:2]
    result = image.copy()

    # Fine 10 px grid: saturated blue-turquoise, blended at 38% opacity.
    minor_overlay = result.copy()
    for x in range(minor_step, width, minor_step):
        if x % major_step != 0:
            cv2.line(
                minor_overlay,
                (x, 0),
                (x, height - 1),
                MINOR_GRID_COLOR,
                1,
                cv2.LINE_AA,
            )
    for y in range(minor_step, height, minor_step):
        if y % major_step != 0:
            cv2.line(
                minor_overlay,
                (0, y),
                (width - 1, y),
                MINOR_GRID_COLOR,
                1,
                cv2.LINE_AA,
            )
    result = cv2.addWeighted(
        minor_overlay, GRID_MINOR_ALPHA, result, 1.0 - GRID_MINOR_ALPHA, 0
    )

    # Dashed 25 px half-step grid, excluding 50 px major intersections.
    half_overlay = result.copy()
    for x in range(half_step, width, half_step):
        if x % major_step != 0:
            draw_dashed_grid_line(
                half_overlay, (x, 0), (x, height - 1), HALF_GRID_COLOR
            )
    for y in range(half_step, height, half_step):
        if y % major_step != 0:
            draw_dashed_grid_line(
                half_overlay, (0, y), (width - 1, y), HALF_GRID_COLOR
            )
    result = cv2.addWeighted(
        half_overlay, GRID_HALF_ALPHA, result, 1.0 - GRID_HALF_ALPHA, 0
    )

    # Magenta 50 px reference grid is drawn last and remains easy to see.
    major_overlay = result.copy()
    for x in range(major_step, width, major_step):
        cv2.line(
            major_overlay,
            (x, 0),
            (x, height - 1),
            MAJOR_GRID_COLOR,
            1,
            cv2.LINE_AA,
        )
    for y in range(major_step, height, major_step):
        cv2.line(
            major_overlay,
            (0, y),
            (width - 1, y),
            MAJOR_GRID_COLOR,
            1,
            cv2.LINE_AA,
        )
    result = cv2.addWeighted(
        major_overlay, GRID_MAJOR_ALPHA, result, 1.0 - GRID_MAJOR_ALPHA, 0
    )

    # Put x labels a few pixels into each 50 px cell at 250 px intervals;
    # y labels sit below horizontal lines. Their rows are staggered to avoid
    # colliding with x labels and with the tile-name badge.
    x_label_rows = list(range(0, height, 250))
    for x in range(major_step, width, major_step):
        for row_top in x_label_rows:
            label_y = min(height - 2, row_top + 35)
            put_coordinate_label(
                result,
                str(x),
                (min(width - 1, x + 4), label_y),
                MAJOR_GRID_COLOR,
            )

    y_label_columns = sorted(
        {
            column
            for column in [0, *range(250, width, 250), max(0, width - 28)]
            if column < width
        }
    )
    for y in range(major_step, height, major_step):
        for column_left in y_label_columns:
            label_y = min(height - 2, y + 14)
            put_coordinate_label(
                result,
                str(y),
                (min(width - 1, column_left + 4), label_y),
                Y_COORDINATE_COLOR,
            )

    if filename:
        result = add_filename_label(result, filename)
    return result


def _write_png(path: Path, image: Any) -> None:
    """Save PNG through an in-memory encoder (also works with Unicode paths)."""
    success, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1]
    )
    if not success:
        raise OSError(f"не удалось закодировать PNG: {path}")
    path.write_bytes(encoded.tobytes())


def process_image(
    image_path: Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Split one image and save plain and grid versions of every tile."""
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"не найден исходный файл: {image_path}")
    if tile_size <= 0:
        raise ValueError("размер тайла должен быть больше нуля")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("перекрытие должно быть от 0 до размера тайла - 1")

    # imdecode handles non-ASCII filenames on Windows as well as Linux.
    encoded_source = image_path.read_bytes()
    image = cv2.imdecode(
        np.frombuffer(encoded_source, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError(f"OpenCV не смог прочитать JPG: {image_path}")

    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"некорректный размер изображения: {width}x{height}")

    output_dir = image_path.parent / f"{image_path.stem}_{tile_size}x{tile_size}"
    prepare_output_directory(output_dir, overwrite)

    x_starts = tile_starts(width, tile_size, overlap)
    y_starts = tile_starts(height, tile_size, overlap)
    digits = max(1, len(str(max(width - 1, height - 1))))
    tile_bounds: list[tuple[int, int, int, int, str]] = []

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + tile_size, width) - 1
            y1 = min(y0 + tile_size, height) - 1
            tile_name = coordinate_tile_name(x0, y0, x1, y1, digits)
            tile = image[y0 : y1 + 1, x0 : x1 + 1]
            if tile.size == 0:
                raise ValueError(
                    f"получился пустой тайл для координат {x0},{y0}—{x1},{y1}"
                )

            plain_tile = add_filename_label(tile.copy(), tile_name)
            grid_tile = apply_soft_grid(tile, filename=tile_name)
            _write_png(output_dir / PLAIN_TILES_DIR / tile_name, plain_tile)
            _write_png(output_dir / GRID_TILES_DIR / tile_name, grid_tile)
            tile_bounds.append((x0, y0, x1, y1, tile_name))

    tile_map_path = write_tile_map(
        image, tile_bounds, output_dir / "tile_map.jpg"
    )
    return {
        "image_path": image_path,
        "output_dir": output_dir,
        "width": width,
        "height": height,
        "tile_count": len(tile_bounds),
        "columns": len(x_starts),
        "rows": len(y_starts),
        "tile_map": tile_map_path,
    }


def process_directory(
    input_dir: Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    recursive: bool = False,
    overwrite: bool = False,
    additional_tile_size: int | None = DEFAULT_ADDITIONAL_TILE_SIZE,
) -> list[dict[str, Any]]:
    """Process JPG/JPEG files at the primary and optional secondary sizes."""
    if tile_size <= 0:
        raise ValueError("размер тайла должен быть больше нуля")
    if additional_tile_size is not None and additional_tile_size < 0:
        raise ValueError("дополнительный размер тайла не может быть отрицательным")

    tile_sizes = [tile_size]
    if additional_tile_size not in (None, 0, tile_size):
        tile_sizes.append(additional_tile_size)
    for size in tile_sizes:
        if overlap < 0 or overlap >= size:
            raise ValueError(
                f"перекрытие должно быть от 0 до {size - 1} px для тайла {size}x{size}"
            )

    input_dir = input_dir.expanduser().resolve()
    images = find_jpg_images(input_dir, recursive=recursive)
    if not images:
        return []

    # Two source images with the same stem and parent would target the same
    # output folder for a given size; detect all collisions before writing.
    output_paths = [
        (image_path.parent / f"{image_path.stem}_{size}x{size}").resolve()
        for image_path in images
        for size in tile_sizes
    ]
    normalized_output_paths = [str(path).casefold() for path in output_paths]
    if len(normalized_output_paths) != len(set(normalized_output_paths)):
        raise ValueError(
            "найдены JPG-файлы с одинаковым именем без расширения в одной папке; "
            "для них получилась бы одна папка результатов"
        )

    results = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] Обрабатываю: {image_path}", flush=True)
        for size in tile_sizes:
            print(f"  Нарезка на {size}×{size} px...", flush=True)
            result = process_image(
                image_path,
                tile_size=size,
                overlap=overlap,
                overwrite=overwrite,
            )
            results.append(result)
            print(
                f"    {result['width']}×{result['height']} px → "
                f"{result['tile_count']} тайлов "
                f"({result['columns']}×{result['rows']}); "
                f"результаты: {result['output_dir']}",
                flush=True,
            )
            print(f"    Карта разбивки: {result['tile_map']}", flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Найти JPG/JPEG-сканы и создать PNG-тайлы в двух размерах "
            "(по умолчанию 1000x1000 и 500x500). Для каждого размера создаются "
            "tiles/, tiles_with_grid/ и tile_map.jpg с границами и именами файлов. "
            "Имена PNG содержат включительные координаты x0_y0_x1_y1."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"папка с JPG/JPEG (по умолчанию: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--tile-size",
        type=positive_int,
        default=DEFAULT_TILE_SIZE,
        help="основной размер тайла (по умолчанию: 1000; также создаётся размер --also-tile-size)",
    )
    parser.add_argument(
        "--also-tile-size",
        type=non_negative_int,
        default=DEFAULT_ADDITIONAL_TILE_SIZE,
        help="второй размер тайла (по умолчанию: 500; 0 отключает второй размер)",
    )
    parser.add_argument(
        "--overlap",
        type=non_negative_int,
        default=DEFAULT_OVERLAP,
        help=(
            "перекрытие соседних тайлов в пикселях "
            "(по умолчанию: 100; задайте 0, чтобы не перекрывать тайлы)"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="искать JPG/JPEG также во вложенных папках",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="пересоздать папки результатов, созданные этим скриптом",
    )
    args = parser.parse_args()

    try:
        results = process_directory(
            args.input_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            recursive=args.recursive,
            overwrite=args.overwrite,
            additional_tile_size=args.also_tile_size or None,
        )
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")

    if not results:
        print(f"В папке не найдены JPG/JPEG-файлы: {args.input_dir}")
        return 0

    total_tiles = sum(result["tile_count"] for result in results)
    source_count = len({result["image_path"] for result in results})
    print(
        f"Готово: обработано изображений — {source_count}, "
        f"вариантов размера — {len(results)}, тайлов — {total_tiles}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
