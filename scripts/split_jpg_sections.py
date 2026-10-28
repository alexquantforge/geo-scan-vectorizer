#!/usr/bin/env python3
"""Split JPG geological sections into coordinate-named PNG tiles.

For each source image, create ``<stem>_<tile-size>x<tile-size>/`` beside it,
with the plain tiles in ``tiles/`` and soft-grid copies in ``tiles_with_grid/``.
The grid is local to each tile, at 10 px and 50 px intervals.
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
DEFAULT_OVERLAP = 100
PLAIN_TILES_DIR = "tiles"
GRID_TILES_DIR = "tiles_with_grid"
OUTPUT_SUBDIRS = (PLAIN_TILES_DIR, GRID_TILES_DIR)

# OpenCV colors use BGR order.
MINOR_GRID_COLOR = (255, 210, 0)  # light cyan
MAJOR_GRID_COLOR = (255, 0, 255)  # magenta
Y_COORDINATE_COLOR = (220, 0, 0)  # blue


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
        unexpected = [entry for entry in entries if entry.name not in OUTPUT_SUBDIRS]
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


def add_filename_label(image: Any, filename: str) -> Any:
    """Draw a small, readable tile filename in the upper-left corner."""
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return image

    font = cv2.FONT_HERSHEY_PLAIN
    scale = 0.8
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        filename, font, scale, thickness
    )
    max_text_width = max(1, width - 8)
    if text_width > max_text_width:
        scale *= max_text_width / text_width
        (text_width, text_height), baseline = cv2.getTextSize(
            filename, font, scale, thickness
        )

    right = min(width - 1, text_width + 7)
    bottom = min(height - 1, text_height + baseline + 6)
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (right, bottom), (255, 255, 255), -1)
    cv2.addWeighted(panel, 0.82, image, 0.18, 0, dst=image)

    baseline_y = min(height - 1, text_height + 3)
    cv2.putText(
        image,
        filename,
        (3, baseline_y),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return image


def apply_soft_grid(
    image: Any,
    filename: str | None = None,
    minor_step: int = 10,
    major_step: int = 50,
) -> Any:
    """Return a tile with a light 10 px grid, 50 px grid and coordinate labels."""
    if minor_step <= 0 or major_step <= 0:
        raise ValueError("шаг сетки должен быть больше нуля")
    if major_step % minor_step != 0:
        raise ValueError("крупный шаг сетки должен делиться на мелкий")

    height, width = image.shape[:2]
    result = image.copy()

    # Fine lines, except where the stronger 50 px lines will be drawn.
    minor_overlay = result.copy()
    for x in range(minor_step, width, minor_step):
        if x % major_step != 0:
            cv2.line(
                minor_overlay,
                (x, 0),
                (x, height - 1),
                MINOR_GRID_COLOR,
                1,
            )
    for y in range(minor_step, height, minor_step):
        if y % major_step != 0:
            cv2.line(
                minor_overlay,
                (0, y),
                (width - 1, y),
                MINOR_GRID_COLOR,
                1,
            )
    result = cv2.addWeighted(minor_overlay, 0.18, result, 0.82, 0)

    # Stronger 50 px reference lines.
    major_overlay = result.copy()
    for x in range(major_step, width, major_step):
        cv2.line(
            major_overlay,
            (x, 0),
            (x, height - 1),
            MAJOR_GRID_COLOR,
            1,
        )
    for y in range(major_step, height, major_step):
        cv2.line(
            major_overlay,
            (0, y),
            (width - 1, y),
            MAJOR_GRID_COLOR,
            1,
        )
    result = cv2.addWeighted(major_overlay, 0.35, result, 0.65, 0)

    # Coordinate labels on the 50 px grid, matching the supplied example.
    text_overlay = result.copy()
    y_positions = sorted(
        {
            position
            for position in [12, *range(248, height, 250), height - 4]
            if 0 <= position < height
        }
    )
    x_positions = sorted(
        {
            position
            for position in [2, *range(252, width, 250), width - 28]
            if 0 <= position < width
        }
    )
    for x in range(major_step, width, major_step):
        for y_position in y_positions:
            cv2.putText(
                text_overlay,
                str(x),
                (x + 2, y_position),
                cv2.FONT_HERSHEY_PLAIN,
                0.65,
                MAJOR_GRID_COLOR,
                1,
                cv2.LINE_AA,
            )
    for y in range(major_step, height, major_step):
        for x_position in x_positions:
            cv2.putText(
                text_overlay,
                str(y),
                (x_position, y - 2),
                cv2.FONT_HERSHEY_PLAIN,
                0.65,
                Y_COORDINATE_COLOR,
                1,
                cv2.LINE_AA,
            )
    result = cv2.addWeighted(text_overlay, 0.65, result, 0.35, 0)

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
    tile_count = 0

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
            tile_count += 1

    return {
        "image_path": image_path,
        "output_dir": output_dir,
        "width": width,
        "height": height,
        "tile_count": tile_count,
        "columns": len(x_starts),
        "rows": len(y_starts),
    }


def process_directory(
    input_dir: Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    recursive: bool = False,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Process all JPG/JPEG files found in a folder."""
    input_dir = input_dir.expanduser().resolve()
    images = find_jpg_images(input_dir, recursive=recursive)
    if not images:
        return []

    # Two different extensions with the same stem would otherwise target the
    # same output folder; report that clearly before writing any results.
    output_paths = [
        (path.parent / f"{path.stem}_{tile_size}x{tile_size}").resolve()
        for path in images
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
        result = process_image(
            image_path,
            tile_size=tile_size,
            overlap=overlap,
            overwrite=overwrite,
        )
        results.append(result)
        print(
            f"  {result['width']}×{result['height']} px → "
            f"{result['tile_count']} тайлов "
            f"({result['columns']}×{result['rows']}); "
            f"результаты: {result['output_dir']}",
            flush=True,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Найти JPG/JPEG-сканы и нарезать каждый на PNG-тайлы. "
            "Для каждого изображения создаются папки tiles/ и tiles_with_grid/; "
            "в именах тайлов указаны включительные координаты x0_y0_x1_y1."
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
        help="ширина и высота тайла в пикселях (по умолчанию: 1000)",
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
        )
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")

    if not results:
        print(f"В папке не найдены JPG/JPEG-файлы: {args.input_dir}")
        return 0

    total_tiles = sum(result["tile_count"] for result in results)
    print(
        f"Готово: обработано изображений — {len(results)}, "
        f"тайлов — {total_tiles}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
