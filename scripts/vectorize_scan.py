#!/usr/bin/env python3
"""Split a scan into a row grid of tiles and vectorize every tile.

The default layout is 10 columns × 2 rows = 20 tiles ("10 top, 10 bottom").
Every artifact is written into its own subfolder:

    <scan>_vectorized/
      tiles/row_01/*.png ...     source tiles (10 per row)
      rows/row_01.jpg ...        the 10 tiles of a row assembled in a row
      tables/tiles/row_01/...    per-tile CSV + JSON coordinate tables
      tables/rows/row_01.csv ... per-row aggregate tables
      tables/lines.csv/json      all detected objects
      tables/tiles_summary.csv   one row per tile with counts
      overlays/tiles/row_01/...  per-tile overlays (20 files)
      overlays/overlay_full.jpg  one combined overlay for the whole scan
      overlays/overlay_preview.jpg
      run.json                   run metadata and tile map
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    raise SystemExit(
        "Не найден OpenCV. Установите зависимости: "
        "python -m pip install -r requirements-detection.txt"
    ) from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    raise SystemExit(
        "Не найден Pillow. Установите зависимость командой: python -m pip install Pillow"
    ) from exc

try:
    # Running the file as a script puts its directory on sys.path.
    from detect_tile_lines import DEFAULT_IMAGE, LINE_COLORS, Tile, detect_tiles
except ModuleNotFoundError as exc:  # pragma: no cover - script vs package import
    if exc.name != "detect_tile_lines":
        raise
    try:
        # Package imports (for example, from tests) need a relative import.
        from .detect_tile_lines import (
            DEFAULT_IMAGE,
            LINE_COLORS,
            Tile,
            detect_tiles,
        )
    except ImportError as relative_exc:  # pragma: no cover - depends on invocation
        raise SystemExit(
            "Не удалось загрузить scripts/detect_tile_lines.py. "
            "Запускайте скрипт из репозитория или через PyCharm."
        ) from relative_exc


DEFAULT_COLUMNS = 10
DEFAULT_ROWS = 2
DEFAULT_PREVIEW_MAX_SIDE = 3000
DEFAULT_ROW_STRIP_QUALITY = 92
LINE_TYPES = ("straight", "curved", "dashed_curve", "small_circle", "aligned_dot")
OUTPUT_SUBDIRS = ("tiles", "rows", "tables", "overlays")
OUTPUT_MARKER = "run.json"

# Pillow's default decompression-bomb limit is lower than a full 24030×13244
# scan (~318 million pixels). Keep a finite limit, but allow that scan.
MAX_IMAGE_PIXELS = 400_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@dataclass(frozen=True)
class GridTile:
    """A tile of a rectangular grid with inclusive pixel bounds."""

    path: Path
    row: int
    column: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def row_name(self) -> str:
        return f"row_{self.row:02d}"


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидается целое число") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return result


def grid_boundaries(length: int, parts: int) -> list[int]:
    """Return ``parts + 1`` evenly rounded boundaries covering ``length`` px.

    The last boundary is exactly ``length``, so the tiles always cover the
    source image without gaps or overlaps, even when the size is not divisible.
    """
    if length <= 0:
        raise ValueError(f"некорректный размер изображения: {length}")
    if parts <= 0:
        raise ValueError("количество тайлов должно быть больше нуля")
    if parts > length:
        raise ValueError(
            f"нельзя разбить {length} px на {parts} частей: частей больше, чем пикселей"
        )
    return [round(index * length / parts) for index in range(parts + 1)]


def default_output_dir(image_path: Path) -> Path:
    return image_path.parent / f"{image_path.stem}_vectorized"


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    """Create the output folder, refusing to delete anything we did not write."""
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"путь для результата не является папкой: {output_dir}")

    if output_dir.exists():
        entries = list(output_dir.iterdir())
        if entries and not overwrite:
            raise FileExistsError(
                f"папка уже существует и не пуста: {output_dir}\n"
                "Чтобы пересоздать результаты, добавьте флаг --overwrite."
            )
        if entries:
            allowed = set(OUTPUT_SUBDIRS) | {OUTPUT_MARKER}
            unexpected = [entry for entry in entries if entry.name not in allowed]
            if unexpected:
                names = ", ".join(entry.name for entry in unexpected[:5])
                raise FileExistsError(
                    f"не буду очищать папку с посторонними файлами: {output_dir} "
                    f"(например: {names})"
                )
            for entry in entries:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)


def split_into_grid(
    image_path: Path,
    output_dir: Path,
    columns: int = DEFAULT_COLUMNS,
    rows: int = DEFAULT_ROWS,
) -> tuple[list[GridTile], int, int]:
    """Crop the scan into a columns×rows grid and save lossless PNG tiles."""
    if not image_path.is_file():
        raise FileNotFoundError(f"не найден исходный файл: {image_path}")

    with Image.open(image_path) as image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"некорректный размер изображения: {width}x{height}")
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"изображение слишком большое ({width}x{height} пикселей); "
                f"лимит программы — {MAX_IMAGE_PIXELS:,} пикселей"
            )

        x_edges = grid_boundaries(width, columns)
        y_edges = grid_boundaries(height, rows)
        digits = max(1, len(str(max(width - 1, height - 1))))
        tiles: list[GridTile] = []
        image.load()

        for row_index in range(rows):
            row_dir = output_dir / "tiles" / f"row_{row_index + 1:02d}"
            row_dir.mkdir(parents=True, exist_ok=True)
            y0, y1 = y_edges[row_index], y_edges[row_index + 1] - 1
            for column_index in range(columns):
                x0, x1 = x_edges[column_index], x_edges[column_index + 1] - 1
                tile = image.crop((x0, y0, x1 + 1, y1 + 1))
                # PNG avoids adding another lossy JPEG compression step.
                if tile.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                    tile = tile.convert("RGB")
                name = (
                    f"{x0:0{digits}d}_{y0:0{digits}d}_"
                    f"{x1:0{digits}d}_{y1:0{digits}d}.png"
                )
                path = row_dir / name
                tile.save(path, format="PNG", compress_level=1)
                tiles.append(
                    GridTile(
                        path=path,
                        row=row_index + 1,
                        column=column_index + 1,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                    )
                )

    return tiles, width, height


def tiles_by_row(tiles: Iterable[GridTile]) -> dict[int, list[GridTile]]:
    grouped: dict[int, list[GridTile]] = {}
    for tile in tiles:
        grouped.setdefault(tile.row, []).append(tile)
    for row_tiles in grouped.values():
        row_tiles.sort(key=lambda item: item.column)
    return dict(sorted(grouped.items()))


def write_row_strips(
    tiles: list[GridTile],
    rows_dir: Path,
    quality: int = DEFAULT_ROW_STRIP_QUALITY,
) -> list[Path]:
    """Assemble the 10 tiles of every row into one horizontal strip image."""
    rows_dir.mkdir(parents=True, exist_ok=True)
    strips: list[Path] = []
    for row_index, row_tiles in tiles_by_row(tiles).items():
        strip_height = max(tile.height for tile in row_tiles)
        strip_width = sum(tile.width for tile in row_tiles)
        strip = np.empty((strip_height, strip_width, 3), dtype=np.uint8)
        offset_x = 0
        for tile in row_tiles:
            tile_image = cv2.imread(str(tile.path), cv2.IMREAD_COLOR)
            if tile_image is None:
                raise ValueError(f"OpenCV не смог прочитать тайл: {tile.path}")
            strip[0 : tile.height, offset_x : offset_x + tile.width] = tile_image
            offset_x += tile.width
            del tile_image
        strip_path = rows_dir / f"row_{row_index:02d}.jpg"
        if not cv2.imwrite(
            str(strip_path), strip, [cv2.IMWRITE_JPEG_QUALITY, quality]
        ):
            raise OSError(f"не удалось сохранить полосу ряда: {strip_path}")
        strips.append(strip_path)
        del strip
    return strips


def default_grouping_stride(tiles: list[GridTile]) -> int:
    """Pick dot-grouping windows scaled to the tile size, bounded to 500..2000."""
    if not tiles:
        return 500
    smallest_side = min(min(tile.width, tile.height) for tile in tiles)
    return max(500, min(2000, smallest_side))


def detect_grid(
    tiles: list[GridTile],
    width: int,
    height: int,
    grouping_stride: int | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Run the existing detector over all grid tiles and merge coordinates."""
    detect_input = [
        Tile(path=tile.path, x0=tile.x0, y0=tile.y0, x1=tile.x1, y1=tile.y1)
        for tile in tiles
    ]
    return detect_tiles(
        detect_input,
        width,
        height,
        grouping_stride=grouping_stride or default_grouping_stride(tiles),
        workers=workers,
    )


def record_with_local_coordinates(
    line: dict[str, Any], tile: GridTile
) -> dict[str, Any]:
    """Return a copy of a record with tile-local pixel coordinates added."""
    record = dict(line)
    record["x1_local"] = line["x1"] - tile.x0
    record["y1_local"] = line["y1"] - tile.y0
    record["x2_local"] = line["x2"] - tile.x0
    record["y2_local"] = line["y2"] - tile.y0
    return record


def counts_by_type(lines: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(line["type"] for line in lines)
    return {line_type: counts.get(line_type, 0) for line_type in LINE_TYPES}


def lines_for_tile(lines: list[dict[str, Any]], tile: GridTile) -> list[dict[str, Any]]:
    return [line for line in lines if line["tile"] == tile.path.name]


def write_csv(
    path: Path, lines: list[dict[str, Any]], include_local: bool = True
) -> None:
    fieldnames = [
        "id",
        "tile",
        "x1",
        "y1",
        "x2",
        "y2",
        "length",
        "angle",
        "type",
        "radius",
    ]
    if include_local:
        fieldnames += ["x1_local", "y1_local", "x2_local", "y2_local"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_tables(
    tables_dir: Path,
    tiles: list[GridTile],
    lines: list[dict[str, Any]],
    image_path: Path,
    width: int,
    height: int,
    columns: int,
    rows: int,
) -> dict[str, Any]:
    """Write per-tile, per-row and combined coordinate tables."""
    tiles_dir = tables_dir / "tiles"
    rows_dir = tables_dir / "rows"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(parents=True, exist_ok=True)

    lines_by_tile: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    tile_payloads: list[dict[str, Any]] = []

    for tile in tiles:
        tile_lines = [
            record_with_local_coordinates(line, tile)
            for line in lines_for_tile(lines, tile)
        ]
        lines_by_tile[tile.path.name] = tile_lines

        row_dir = tiles_dir / tile.row_name
        row_dir.mkdir(parents=True, exist_ok=True)
        stem = tile.path.stem
        write_csv(row_dir / f"{stem}.csv", tile_lines)
        write_json(
            row_dir / f"{stem}.json",
            {
                "source_image": str(image_path),
                "image_size": {"width": width, "height": height},
                "coordinate_system": "pixel coordinates, origin at top-left (0, 0)",
                "tile": {
                    "name": tile.path.name,
                    "row": tile.row,
                    "column": tile.column,
                    "x0": tile.x0,
                    "y0": tile.y0,
                    "x1": tile.x1,
                    "y1": tile.y1,
                    "width": tile.width,
                    "height": tile.height,
                },
                "lines_count": len(tile_lines),
                "counts_by_type": counts_by_type(tile_lines),
                "lines": tile_lines,
            },
        )

        counts = counts_by_type(tile_lines)
        summary = {
            "row": tile.row,
            "column": tile.column,
            "tile": tile.path.name,
            "x0": tile.x0,
            "y0": tile.y0,
            "x1": tile.x1,
            "y1": tile.y1,
            "width": tile.width,
            "height": tile.height,
            **counts,
            "total": len(tile_lines),
        }
        summary_rows.append(summary)
        tile_payloads.append(
            {
                "name": tile.path.name,
                "row": tile.row,
                "column": tile.column,
                "x0": tile.x0,
                "y0": tile.y0,
                "x1": tile.x1,
                "y1": tile.y1,
                "width": tile.width,
                "height": tile.height,
                "lines_count": len(tile_lines),
                "counts_by_type": counts,
            }
        )

    with (tables_dir / "tiles_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "row",
                "column",
                "tile",
                "x0",
                "y0",
                "x1",
                "y1",
                "width",
                "height",
                *LINE_TYPES,
                "total",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Per-row aggregate tables.
    row_payloads = []
    for row_index, row_tiles in tiles_by_row(tiles).items():
        row_tiles_by_name = {tile.path.name: tile for tile in row_tiles}
        row_lines = [
            record_with_local_coordinates(line, row_tiles_by_name[line["tile"]])
            for line in lines
            if line["tile"] in row_tiles_by_name
        ]
        write_csv(rows_dir / f"row_{row_index:02d}.csv", row_lines)
        row_payload = {
            "source_image": str(image_path),
            "image_size": {"width": width, "height": height},
            "row": row_index,
            "tile_count": len(row_tiles),
            "lines_count": len(row_lines),
            "counts_by_type": counts_by_type(row_lines),
            "lines": row_lines,
        }
        write_json(rows_dir / f"row_{row_index:02d}.json", row_payload)
        row_payloads.append(
            {
                "row": row_index,
                "tile_count": len(row_tiles),
                "lines_count": len(row_lines),
                "counts_by_type": counts_by_type(row_lines),
            }
        )

    # Combined tables: keep the global coordinates only.
    combined = [
        {key: value for key, value in line.items() if not key.endswith("_local")}
        for line in lines
    ]
    write_csv(tables_dir / "lines.csv", combined, include_local=False)
    write_json(
        tables_dir / "lines.json",
        {
            "source_image": str(image_path),
            "image_size": {"width": width, "height": height},
            "coordinate_system": "pixel coordinates, origin at top-left (0, 0)",
            "grid": {"columns": columns, "rows": rows},
            "tile_count": len(tiles),
            "lines_count": len(lines),
            "counts_by_type": counts_by_type(lines),
            "tiles": tile_payloads,
            "rows": row_payloads,
            "lines": combined,
        },
    )
    return {
        "lines_count": len(lines),
        "counts_by_type": counts_by_type(lines),
        "tiles": tile_payloads,
    }


def draw_line(
    image: Any,
    line: dict[str, Any],
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    """Draw one record on a BGR image, translating global to local pixels."""
    color = LINE_COLORS.get(line["type"], (0, 0, 255))
    x1 = int(round(line["x1"])) - offset_x
    y1 = int(round(line["y1"])) - offset_y
    x2 = int(round(line["x2"])) - offset_x
    y2 = int(round(line["y2"])) - offset_y
    if line["type"] in {"small_circle", "aligned_dot"}:
        radius = max(1, int(round(float(line["radius"] or 1))))
        cv2.circle(image, (x1, y1), radius, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(image, (x1, y1), radius, color, 2, cv2.LINE_AA)
    else:
        cv2.line(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def write_tile_overlays(
    overlays_dir: Path, tiles: list[GridTile], lines: list[dict[str, Any]]
) -> list[Path]:
    """Save one overlay image per tile into overlays/tiles/row_XX/."""
    paths: list[Path] = []
    lines_by_tile = {tile.path.name: lines_for_tile(lines, tile) for tile in tiles}
    for tile in tiles:
        tile_image = cv2.imread(str(tile.path), cv2.IMREAD_COLOR)
        if tile_image is None:
            raise ValueError(f"OpenCV не смог прочитать тайл: {tile.path}")
        for line in lines_by_tile[tile.path.name]:
            draw_line(tile_image, line, offset_x=tile.x0, offset_y=tile.y0)
        row_dir = overlays_dir / "tiles" / tile.row_name
        row_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = row_dir / f"{tile.path.stem}.jpg"
        if not cv2.imwrite(
            str(overlay_path), tile_image, [cv2.IMWRITE_JPEG_QUALITY, 95]
        ):
            raise OSError(f"не удалось сохранить overlay тайла: {overlay_path}")
        paths.append(overlay_path)
        del tile_image
    return paths


def write_combined_overlay(
    overlays_dir: Path,
    image_path: Path,
    lines: list[dict[str, Any]],
    preview_max_side: int,
) -> tuple[Path, Path]:
    """Draw every object on the full scan and save a preview as well."""
    overlays_dir.mkdir(parents=True, exist_ok=True)
    source_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise ValueError(f"OpenCV не смог прочитать исходное изображение: {image_path}")
    height, width = source_image.shape[:2]

    # Draw in place: the original scan is no longer needed after this point,
    # which halves peak memory for a full 24030×13244 overlay.
    for line in lines:
        draw_line(source_image, line)

    full_path = overlays_dir / "overlay_full.jpg"
    if not cv2.imwrite(
        str(full_path), source_image, [cv2.IMWRITE_JPEG_QUALITY, 95]
    ):
        raise OSError(f"не удалось сохранить общий overlay: {full_path}")

    scale = min(1.0, preview_max_side / max(width, height))
    if scale < 1.0:
        preview_width = max(1, round(width * scale))
        preview_height = max(1, round(height * scale))
        preview = cv2.resize(
            source_image,
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = source_image
    preview_path = overlays_dir / "overlay_preview.jpg"
    if not cv2.imwrite(
        str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 92]
    ):
        raise OSError(f"не удалось сохранить превью overlay: {preview_path}")
    return full_path, preview_path


def write_run_metadata(
    output_dir: Path,
    image_path: Path,
    width: int,
    height: int,
    columns: int,
    rows: int,
    tiles: list[GridTile],
    lines: list[dict[str, Any]],
    row_strips: list[Path],
    summary: dict[str, Any],
) -> Path:
    payload = {
        "source_image": str(image_path),
        "image_size": {"width": width, "height": height},
        "coordinate_system": "pixel coordinates, origin at top-left (0, 0)",
        "tile_naming": "x0_y0_x1_y1.png with inclusive bounds",
        "grid": {
            "columns": columns,
            "rows": rows,
            "tile_count": len(tiles),
            "layout": f"{columns} top + {columns} bottom"
            if rows == 2
            else f"{columns} per row × {rows} rows",
        },
        "tiles": [
            {
                "name": tile.path.name,
                "relative_path": str(tile.path.relative_to(output_dir)),
                "row": tile.row,
                "column": tile.column,
                "x0": tile.x0,
                "y0": tile.y0,
                "x1": tile.x1,
                "y1": tile.y1,
                "width": tile.width,
                "height": tile.height,
            }
            for tile in tiles
        ],
        "row_strips": [str(path.relative_to(output_dir)) for path in row_strips],
        "lines_count": len(lines),
        "counts_by_type": counts_by_type(lines),
        "tiles_summary": summary["tiles"],
    }
    path = output_dir / OUTPUT_MARKER
    write_json(path, payload)
    return path


def vectorize(
    image_path: Path,
    output_dir: Path | None = None,
    columns: int = DEFAULT_COLUMNS,
    rows: int = DEFAULT_ROWS,
    overwrite: bool = False,
    row_strips: bool = True,
    preview_max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
    grouping_stride: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run the full split → detect → tables → overlays pipeline."""
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"не найден исходный файл: {image_path}")
    if columns <= 0 or rows <= 0:
        raise ValueError("количество столбцов и строк должно быть больше нуля")
    if preview_max_side <= 0:
        raise ValueError("размер превью должен быть больше нуля")
    if workers <= 0:
        raise ValueError("количество процессов должно быть больше нуля")

    if output_dir is None:
        output_dir = default_output_dir(image_path)
    else:
        output_dir = output_dir.expanduser().resolve()

    prepare_output_directory(output_dir, overwrite)
    print(f"Исходник: {image_path}")
    print(
        f"Разбиваю на сетку {columns}×{rows} = {columns * rows} тайлов "
        f"в {output_dir / 'tiles'}...",
        flush=True,
    )
    tiles, width, height = split_into_grid(
        image_path, output_dir, columns=columns, rows=rows
    )
    print(
        f"Готово: {len(tiles)} тайлов, размер скана {width}x{height} px.",
        flush=True,
    )

    strip_paths: list[Path] = []
    if row_strips:
        print("Собираю ряды тайлов в полосы...", flush=True)
        strip_paths = write_row_strips(tiles, output_dir / "rows")

    print("Запускаю детектор по тайлам...", flush=True)
    lines = detect_grid(
        tiles,
        width,
        height,
        grouping_stride=grouping_stride,
        workers=workers,
    )

    print("Записываю таблицы координат...", flush=True)
    summary = write_tables(
        output_dir / "tables",
        tiles,
        lines,
        image_path,
        width,
        height,
        columns,
        rows,
    )

    print("Сохраняю overlay каждого тайла...", flush=True)
    tile_overlay_paths = write_tile_overlays(output_dir / "overlays", tiles, lines)

    print("Сохраняю общий overlay...", flush=True)
    full_overlay, preview_overlay = write_combined_overlay(
        output_dir / "overlays",
        image_path,
        lines,
        preview_max_side=preview_max_side,
    )

    metadata_path = write_run_metadata(
        output_dir,
        image_path,
        width,
        height,
        columns,
        rows,
        tiles,
        lines,
        strip_paths,
        summary,
    )

    result = {
        "output_dir": output_dir,
        "tiles": tiles,
        "lines": lines,
        "lines_count": len(lines),
        "counts_by_type": counts_by_type(lines),
        "tile_overlays": tile_overlay_paths,
        "row_strips": strip_paths,
        "full_overlay": full_overlay,
        "preview_overlay": preview_overlay,
        "tables_dir": output_dir / "tables",
        "metadata": metadata_path,
    }
    print("")
    print(f"Готово. Найдено объектов: {len(lines)}")
    for line_type, count in counts_by_type(lines).items():
        print(f"  {line_type}: {count}")
    print(f"Тайлы: {output_dir / 'tiles'}")
    print(f"Полосы рядов: {output_dir / 'rows'}")
    print(f"Таблицы по тайлам: {output_dir / 'tables' / 'tiles'}")
    print(f"Сводная таблица: {output_dir / 'tables' / 'tiles_summary.csv'}")
    print(f"Общая таблица: {output_dir / 'tables' / 'lines.csv'}")
    print(f"Overlay тайлов: {output_dir / 'overlays' / 'tiles'} ({len(tile_overlay_paths)} шт.)")
    print(f"Общий overlay: {full_overlay}")
    print(f"Превью: {preview_overlay}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Разбить скан на сетку тайлов (по умолчанию 10 сверху и 10 снизу), "
            "найти линии и точки, сохранить таблицы координат по каждому тайлу, "
            "overlay каждого тайла и один общий. Результаты раскладываются "
            "по подпапкам tiles/, rows/, tables/, overlays/."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"исходный скан (по умолчанию: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--columns",
        type=positive_int,
        default=DEFAULT_COLUMNS,
        help=f"количество тайлов в ряду (по умолчанию: {DEFAULT_COLUMNS})",
    )
    parser.add_argument(
        "--rows",
        type=positive_int,
        default=DEFAULT_ROWS,
        help=f"количество рядов тайлов (по умолчанию: {DEFAULT_ROWS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="папка результатов (по умолчанию — соседняя <скан>_vectorized)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="пересоздать результаты в папке, созданной этой программой",
    )
    parser.add_argument(
        "--no-row-strips",
        action="store_true",
        help="не собирать 10 тайлов каждого ряда в полосу rows/row_XX.jpg",
    )
    parser.add_argument(
        "--preview-max-side",
        type=positive_int,
        default=DEFAULT_PREVIEW_MAX_SIDE,
        help=f"максимальная сторона общего превью (по умолчанию: {DEFAULT_PREVIEW_MAX_SIDE})",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=1,
        help=(
            "сколько тайлов обрабатывать параллельно (по умолчанию 1); "
            "каждый процесс держит один тайл в памяти"
        ),
    )
    parser.add_argument(
        "--dot-window-stride",
        type=positive_int,
        default=None,
        help=(
            "шаг окон глобальной группировки точек, px; "
            "по умолчанию подбирается по размеру тайла"
        ),
    )
    args = parser.parse_args()

    try:
        vectorize(
            image_path=args.image,
            output_dir=args.output_dir,
            columns=args.columns,
            rows=args.rows,
            overwrite=args.overwrite,
            row_strips=not args.no_row_strips,
            preview_max_side=args.preview_max_side,
            grouping_stride=args.dot_window_stride,
            workers=args.jobs,
        )
    except (OSError, ValueError, RuntimeError, Image.DecompressionBombError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
