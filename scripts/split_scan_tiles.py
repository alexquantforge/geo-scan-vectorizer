#!/usr/bin/env python3
"""Split a scan into non-overlapping, coordinate-named image tiles."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    raise SystemExit(
        "Не найден Pillow. Установите зависимость командой: python -m pip install Pillow"
    ) from exc


DEFAULT_IMAGE = Path(
    "/home/alex/2025-12-03-1636-backup/PycharmProjects/geo-scan-vectorizer/"
    "Графика_разрезы/Графика_разрезы/РЛ-III(Г).jpg"
)
DEFAULT_TILE_SIZE = 500

# Pillow's default decompression-bomb limit is lower than the stated scan size
# (24030 x 13244 ~= 318 million pixels). Keep a finite limit, but allow this scan.
MAX_IMAGE_PIXELS = 400_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидается целое число") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return parsed


def output_directory_for(image_path: Path, tile_size: int) -> Path:
    return image_path.parent / f"{image_path.stem}_tiles_{tile_size}_by_{tile_size}"


def coordinate_filename(
    x0: int, y0: int, x1: int, y1: int, digits: int, extension: str
) -> str:
    """Return a filename with inclusive pixel bounds: x0_y0_x1_y1."""
    return (
        f"{x0:0{digits}d}_{y0:0{digits}d}_"
        f"{x1:0{digits}d}_{y1:0{digits}d}.{extension}"
    )


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"путь для результата не является папкой: {output_dir}")

    if output_dir.exists():
        entries = list(output_dir.iterdir())
        if entries and not overwrite:
            raise FileExistsError(
                f"папка уже существует и не пуста: {output_dir}\n"
                "Чтобы пересоздать тайлы, добавьте флаг --overwrite."
            )
        if overwrite:
            tile_pattern = re.compile(r"\d+_\d+_\d+_\d+\.png$")
            unrelated = [
                entry
                for entry in entries
                if not entry.is_file() or tile_pattern.fullmatch(entry.name) is None
            ]
            if unrelated:
                names = ", ".join(entry.name for entry in unrelated[:5])
                raise FileExistsError(
                    f"не буду очищать папку с посторонними файлами: {output_dir} "
                    f"(например: {names})"
                )
            for entry in entries:
                entry.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)


def split_image(
    image_path: Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, int, int, int, int]:
    if tile_size <= 0:
        raise ValueError("размер тайла должен быть больше нуля")

    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"не найден исходный файл: {image_path}")

    if output_dir is None:
        output_dir = output_directory_for(image_path, tile_size)
    else:
        output_dir = output_dir.expanduser().resolve()

    with Image.open(image_path) as image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"некорректный размер изображения: {width}x{height}")
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"изображение слишком большое ({width}x{height} пикселей); "
                f"лимит программы — {MAX_IMAGE_PIXELS:,} пикселей"
            )

        columns = (width + tile_size - 1) // tile_size
        rows = (height + tile_size - 1) // tile_size
        digits = max(1, len(str(max(width - 1, height - 1))))

        prepare_output_directory(output_dir, overwrite)
        image.load()

        for y0 in range(0, height, tile_size):
            y1 = min(y0 + tile_size, height) - 1
            for x0 in range(0, width, tile_size):
                x1 = min(x0 + tile_size, width) - 1
                tile = image.crop((x0, y0, x1 + 1, y1 + 1))
                # PNG avoids adding another lossy JPEG compression step.
                if tile.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                    tile = tile.convert("RGB")
                filename = coordinate_filename(x0, y0, x1, y1, digits, "png")
                tile.save(output_dir / filename, format="PNG", compress_level=1)

    return output_dir, width, height, columns, rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Разрезать скан на неперекрывающиеся тайлы. "
            "Имена файлов содержат координаты x0_y0_x1_y1 включительно; "
            "начало координат — верхний левый пиксель (0, 0)."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"исходное изображение (по умолчанию: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--tile-size",
        type=positive_int,
        default=DEFAULT_TILE_SIZE,
        help=f"ширина и высота тайла в пикселях (по умолчанию: {DEFAULT_TILE_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="папка для тайлов (по умолчанию — соседняя с изображением)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="пересоздать тайлы, удалив только PNG-файлы с координатными именами",
    )
    args = parser.parse_args()

    try:
        output_dir, width, height, columns, rows = split_image(
            args.image,
            tile_size=args.tile_size,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")

    print(f"Исходник: {args.image} ({width}x{height} px)")
    print(f"Папка тайлов: {output_dir}")
    print(
        f"Готово: {columns} столбцов × {rows} строк = {columns * rows} тайлов; "
        f"размер — до {args.tile_size}x{args.tile_size} px."
    )
    print("Координаты в имени: x0_y0_x1_y1, x1 и y1 включены; начало — (0, 0).")
    print("Крайние тайлы справа/снизу меньше 500 px, без белого дополнения.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
