#!/usr/bin/env python3
"""Detect line segments in coordinate-named tiles and draw them on the source scan."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    raise SystemExit(
        "Не найден OpenCV. Установите зависимости: "
        "python -m pip install -r requirements-detection.txt"
    ) from exc

try:
    # Running the file as a script puts its directory on sys.path.
    from detect_lines import (
        ALIGNED_DOT_MAX_GAP,
        detect_all,
        filter_dot_candidates_near_lines,
        group_aligned_dot_candidates,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - script vs package import
    if exc.name != "detect_lines":
        raise
    try:
        # Package imports (for example, from tests) need a relative import.
        from .detect_lines import (
            ALIGNED_DOT_MAX_GAP,
            detect_all,
            filter_dot_candidates_near_lines,
            group_aligned_dot_candidates,
        )
    except ImportError as relative_exc:  # pragma: no cover - depends on invocation
        raise SystemExit(
            "Не удалось загрузить scripts/detect_lines.py. "
            "Запускайте скрипт из репозитория или через PyCharm."
        ) from relative_exc


DEFAULT_IMAGE = Path(
    "/home/alex/2025-12-03-1636-backup/PycharmProjects/geo-scan-vectorizer/"
    "Графика_разрезы/Графика_разрезы/РЛ-III(Г).jpg"
)
DEFAULT_TILE_SIZE = 500
TILE_NAME_RE = re.compile(
    r"(?P<x0>\d+)_(?P<y0>\d+)_(?P<x1>\d+)_(?P<y1>\d+)\.(?:png|jpe?g|tiff?)$",
    re.IGNORECASE,
)

# OpenCV colors are BGR.
LINE_COLORS = {
    "straight": (0, 0, 255),      # red
    "curved": (0, 255, 0),        # green
    "dashed_curve": (255, 0, 255),  # magenta
    "small_circle": (0, 255, 255),  # yellow
    "aligned_dot": (255, 255, 255),  # white, outlined in black
}


@dataclass(frozen=True)
class Tile:
    path: Path
    x0: int
    y0: int
    x1: int
    y1: int


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидается целое число") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return result


def default_tiles_dir(image_path: Path, tile_size: int) -> Path:
    return image_path.parent / f"{image_path.stem}_tiles_{tile_size}_by_{tile_size}"


def default_output_dir(image_path: Path) -> Path:
    return image_path.parent / f"{image_path.stem}_line_detection"


def list_tiles(tiles_dir: Path) -> list[Tile]:
    if not tiles_dir.is_dir():
        raise FileNotFoundError(f"не найдена папка с тайлами: {tiles_dir}")

    tiles: list[Tile] = []
    for path in tiles_dir.iterdir():
        if not path.is_file():
            continue
        match = TILE_NAME_RE.fullmatch(path.name)
        if match is None:
            continue
        tiles.append(
            Tile(
                path=path,
                x0=int(match.group("x0")),
                y0=int(match.group("y0")),
                x1=int(match.group("x1")),
                y1=int(match.group("y1")),
            )
        )

    tiles.sort(key=lambda tile: (tile.y0, tile.x0, tile.path.name))
    if not tiles:
        raise FileNotFoundError(
            f"в папке нет тайлов с именами x0_y0_x1_y1.png: {tiles_dir}"
        )
    return tiles


def validate_tiles(tiles: list[Tile], width: int, height: int, tile_size: int) -> None:
    expected_origins = {
        (x0, y0)
        for y0 in range(0, height, tile_size)
        for x0 in range(0, width, tile_size)
    }
    actual_origins = {(tile.x0, tile.y0) for tile in tiles}

    duplicates = len(actual_origins) != len(tiles)
    missing = expected_origins - actual_origins
    unexpected = actual_origins - expected_origins
    if duplicates or missing or unexpected:
        details = []
        if duplicates:
            details.append("есть повторяющиеся координаты начала тайла")
        if missing:
            details.append(f"не хватает тайлов: {sorted(missing)[:5]}")
        if unexpected:
            details.append(f"лишние координаты: {sorted(unexpected)[:5]}")
        raise ValueError("состав тайлов не совпадает с исходником: " + "; ".join(details))

    for tile in tiles:
        expected_x1 = min(tile.x0 + tile_size, width) - 1
        expected_y1 = min(tile.y0 + tile_size, height) - 1
        if (tile.x1, tile.y1) != (expected_x1, expected_y1):
            raise ValueError(
                f"неверные границы в имени {tile.path.name}; ожидалось "
                f"{tile.x0}_{tile.y0}_{expected_x1}_{expected_y1}"
            )


def canonical_line_key(line: dict[str, Any]) -> tuple[Any, ...]:
    """Deduplicate near-identical segments or circle centers and radii."""
    if line["type"] in {"small_circle", "aligned_dot"}:
        return (
            line["type"],
            round(line["x1"] / 2),
            round(line["y1"] / 2),
            round(line["radius"] / 2),
        )
    p1 = (round(line["x1"] / 2), round(line["y1"] / 2))
    p2 = (round(line["x2"] / 2), round(line["y2"] / 2))
    endpoints = tuple(sorted((p1, p2)))
    return (line["type"], *endpoints)


def group_dot_candidates_in_windows(
    candidates: list[dict[str, float]], tile_size: int
) -> list[dict[str, float]]:
    """Group points in overlapping global windows so rows can cross tile seams."""
    if not candidates:
        return []

    stride = tile_size
    window_size = max(2 * tile_size, tile_size + 4 * ALIGNED_DOT_MAX_GAP + 1)
    windows: dict[tuple[int, int], list[dict[str, float]]] = {}
    windows_back = max(0, math.ceil(window_size / stride) - 1)

    for candidate in candidates:
        x_cell = int(candidate["x"] // stride) * stride
        y_cell = int(candidate["y"] // stride) * stride
        for x_step in range(windows_back + 1):
            x_start = x_cell - x_step * stride
            if x_start < 0 or not x_start <= candidate["x"] < x_start + window_size:
                continue
            for y_step in range(windows_back + 1):
                y_start = y_cell - y_step * stride
                if y_start < 0 or not y_start <= candidate["y"] < y_start + window_size:
                    continue
                windows.setdefault((x_start, y_start), []).append(candidate)

    aligned_dots = []
    for window_candidates in windows.values():
        for dot in group_aligned_dot_candidates(window_candidates):
            duplicate = next(
                (
                    existing
                    for existing in aligned_dots
                    if math.hypot(existing["x"] - dot["x"], existing["y"] - dot["y"])
                    < 4.0
                ),
                None,
            )
            if duplicate is None:
                aligned_dots.append(dot.copy())
            elif dot["radius"] > duplicate["radius"]:
                duplicate["radius"] = dot["radius"]

    return sorted(aligned_dots, key=lambda dot: (dot["y"], dot["x"]))


def detect_tiles(
    tiles: list[Tile], width: int, height: int, tile_size: int = DEFAULT_TILE_SIZE
) -> list[dict[str, Any]]:
    all_lines: list[dict[str, Any]] = []
    all_dot_candidates: list[dict[str, float]] = []
    seen: set[tuple[Any, ...]] = set()

    for index, tile_info in enumerate(tiles, start=1):
        tile_image = cv2.imread(str(tile_info.path), cv2.IMREAD_COLOR)
        if tile_image is None:
            raise ValueError(f"OpenCV не смог прочитать тайл: {tile_info.path}")

        tile_height, tile_width = tile_image.shape[:2]
        expected_size = (tile_info.y1 - tile_info.y0 + 1, tile_info.x1 - tile_info.x0 + 1)
        if (tile_height, tile_width) != expected_size:
            raise ValueError(
                f"размер тайла {tile_info.path.name} — {tile_width}x{tile_height}, "
                f"а по имени координат должен быть {expected_size[1]}x{expected_size[0]}"
            )

        try:
            candidates = detect_all(
                tile_image,
                tile_info.x0,
                tile_info.y0,
                dot_candidates_out=all_dot_candidates,
            )
        except cv2.error as exc:
            raise RuntimeError(f"ошибка детектора на тайле {tile_info.path.name}: {exc}") from exc

        for candidate in candidates:
            x1, y1, x2, y2, length, angle, line_type = candidate[:7]
            if line_type == "aligned_dot":
                # Aligned dots are regrouped globally after all 500 px tiles
                # have contributed their compact-point candidates.
                continue
            radius = float(candidate[11]) if len(candidate) > 11 else None
            record = {
                "x1": max(0, min(width - 1, int(round(x1)))),
                "y1": max(0, min(height - 1, int(round(y1)))),
                "x2": max(0, min(width - 1, int(round(x2)))),
                "y2": max(0, min(height - 1, int(round(y2)))),
                "length": round(float(length), 1),
                "angle": round(float(angle), 1),
                "type": str(line_type),
                "tile": tile_info.path.name,
                "radius": None if radius is None else round(radius, 1),
            }
            key = canonical_line_key(record)
            if key in seen:
                continue
            seen.add(key)
            all_lines.append(record)

        if index % 25 == 0 or index == len(tiles):
            print(f"Обработано тайлов: {index}/{len(tiles)}")

    existing_line_segments = [
        (line["x1"], line["y1"], line["x2"], line["y2"])
        for line in all_lines
        if line["type"] in {"straight", "curved", "dashed_curve"}
        and line["length"] >= 35.0
    ]
    candidates_off_lines = filter_dot_candidates_near_lines(
        all_dot_candidates, existing_line_segments
    )
    aligned_dots = group_dot_candidates_in_windows(candidates_off_lines, tile_size)
    if aligned_dots:
        all_lines = [
            line
            for line in all_lines
            if not (
                line["type"] == "small_circle"
                and any(
                    math.hypot(line["x1"] - dot["x"], line["y1"] - dot["y"])
                    <= max(7.0, (line["radius"] + dot["radius"]) / 2.0)
                    for dot in aligned_dots
                )
            )
        ]
        seen = {canonical_line_key(line) for line in all_lines}

        for dot in aligned_dots:
            x = max(0, min(width - 1, int(round(dot["x"]))))
            y = max(0, min(height - 1, int(round(dot["y"]))))
            tile_name = next(
                (
                    tile.path.name
                    for tile in tiles
                    if tile.x0 <= x <= tile.x1 and tile.y0 <= y <= tile.y1
                ),
                tiles[0].path.name,
            )
            radius = float(dot["radius"])
            record = {
                "x1": x,
                "y1": y,
                "x2": x,
                "y2": y,
                "length": round(2 * math.pi * radius, 1),
                "angle": 0.0,
                "type": "aligned_dot",
                "tile": tile_name,
                "radius": round(radius, 1),
            }
            key = canonical_line_key(record)
            if key not in seen:
                seen.add(key)
                all_lines.append(record)

    all_lines.sort(key=lambda line: (-line["length"], line["y1"], line["x1"]))
    for line_id, record in enumerate(all_lines, start=1):
        record["id"] = line_id
    return all_lines


def draw_overlay(
    source_image: Any,
    lines: list[dict[str, Any]],
    preview_max_side: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    height, width = source_image.shape[:2]
    overlay = source_image.copy()

    for line in lines:
        color = LINE_COLORS.get(line["type"], (0, 0, 255))
        start = (line["x1"], line["y1"])
        end = (line["x2"], line["y2"])
        if line["type"] in {"small_circle", "aligned_dot"}:
            radius = max(1, int(round(line["radius"])))
            cv2.circle(overlay, start, radius, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(overlay, start, radius, color, 2, cv2.LINE_AA)
        else:
            cv2.line(overlay, start, end, color, 2, cv2.LINE_AA)

    full_path = output_dir / "overlay.jpg"
    if not cv2.imwrite(str(full_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"не удалось сохранить наложение: {full_path}")

    scale = min(1.0, preview_max_side / max(width, height))
    if scale < 1.0:
        preview_width = max(1, round(width * scale))
        preview_height = max(1, round(height * scale))
        preview = cv2.resize(
            source_image,
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA,
        )
        for line in lines:
            color = LINE_COLORS.get(line["type"], (0, 0, 255))
            start = (round(line["x1"] * scale), round(line["y1"] * scale))
            end = (round(line["x2"] * scale), round(line["y2"] * scale))
            if line["type"] in {"small_circle", "aligned_dot"}:
                radius = max(1, round(line["radius"] * scale))
                cv2.circle(preview, start, radius, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.circle(preview, start, radius, color, 2, cv2.LINE_AA)
            else:
                cv2.line(preview, start, end, color, 2, cv2.LINE_AA)
    else:
        preview = overlay

    preview_path = output_dir / "overlay_preview.jpg"
    if not cv2.imwrite(
        str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 92]
    ):
        raise OSError(f"не удалось сохранить превью наложения: {preview_path}")
    return full_path, preview_path


def write_line_files(
    output_dir: Path,
    image_path: Path,
    tiles_dir: Path,
    width: int,
    height: int,
    tiles_count: int,
    lines: list[dict[str, Any]],
) -> tuple[Path, Path]:
    counts = Counter(line["type"] for line in lines)
    payload = {
        "source_image": str(image_path),
        "tiles_directory": str(tiles_dir),
        "image_size": {"width": width, "height": height},
        "coordinate_system": "pixel coordinates, origin at top-left (0, 0)",
        "tile_count": tiles_count,
        "lines_count": len(lines),
        "counts_by_type": dict(sorted(counts.items())),
        "lines": lines,
    }

    json_path = output_dir / "lines.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "lines.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "x1", "y1", "x2", "y2", "length", "angle", "type", "tile", "radius"],
        )
        writer.writeheader()
        writer.writerows(lines)
    return json_path, csv_path


def run(
    image_path: Path,
    tiles_dir: Path | None,
    output_dir: Path | None,
    tile_size: int,
    preview_max_side: int,
) -> None:
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"не найден исходник: {image_path}")

    if tiles_dir is None:
        tiles_dir = default_tiles_dir(image_path, tile_size)
    else:
        tiles_dir = tiles_dir.expanduser().resolve()
    if output_dir is None:
        output_dir = default_output_dir(image_path)
    else:
        output_dir = output_dir.expanduser().resolve()

    source_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise ValueError(f"OpenCV не смог прочитать исходное изображение: {image_path}")
    height, width = source_image.shape[:2]

    tiles = list_tiles(tiles_dir)
    validate_tiles(tiles, width, height, tile_size)
    print(f"Исходник: {image_path} ({width}x{height} px)")
    print(f"Тайлы: {tiles_dir} ({len(tiles)} шт.)")
    print("Запускаю тот же детектор, что используется в scripts/detect_lines.py")

    lines = detect_tiles(tiles, width, height, tile_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = write_line_files(
        output_dir, image_path, tiles_dir, width, height, len(tiles), lines
    )
    overlay_path, preview_path = draw_overlay(
        source_image, lines, preview_max_side, output_dir
    )

    print(f"Найдено объектов: {len(lines)}")
    for line_type, count in sorted(Counter(line["type"] for line in lines).items()):
        print(f"  {line_type}: {count}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Полноразмерное наложение: {overlay_path}")
    print(f"Превью: {preview_path}")
    print(
        "Цвета: straight — красный, curved — зелёный, dashed_curve — пурпурный, "
        "small_circle — жёлтый, aligned_dot — белый с чёрной каймой."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Обработать координатно названные тайлы тем же детектором, что в "
            "scripts/detect_lines.py; сохранить координаты отрезков и наложение на скан."
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
        "--tiles-dir",
        type=Path,
        default=None,
        help="папка с тайлами; по умолчанию соседняя папка *_tiles_500_by_500",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="папка результатов; по умолчанию соседняя папка *_line_detection",
    )
    parser.add_argument(
        "--tile-size",
        type=positive_int,
        default=DEFAULT_TILE_SIZE,
        help=f"размер исходных тайлов (по умолчанию: {DEFAULT_TILE_SIZE})",
    )
    parser.add_argument(
        "--preview-max-side",
        type=positive_int,
        default=3000,
        help="максимальная сторона уменьшенного превью в пикселях (по умолчанию: 3000)",
    )
    args = parser.parse_args()

    try:
        run(
            image_path=args.image,
            tiles_dir=args.tiles_dir,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            preview_max_side=args.preview_max_side,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
