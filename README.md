# geo-scan-vectorizer
Vectorizes Soviet-era geological scans (24030×13244) to DXF/DWG: tiled skeleton + LSD wavy horizons, short tics & dotted fills, with multimodal REL fallback.

## Pipeline
- Input: `original_scan.jpg 24030×13244` (upper 24030×6778 geological + lower technological)
- Tiling: 5×4 = 20 tiles 4806×3311 via `vips crop`
- Per tile: skeleton 1px (adaptiveThreshold 101,12 + iterative thinning) → wavy green; LSD 12-50px ±18° → short red tics; Otsu fat dots 15-110px → purple
- Assembly: `vips arrayjoin` → `assembled_lines_white_vips.jpg 13M` + preview 10% + overlay 40/60
- Example: `r1c1_22_green_purple.jpg` (1001×847) — 203 straight + 564 curved + 42 fat dots 1:1

## Usage
```bash
vips crop original.jpg tile.jpg x y w h
python scripts/process20b.py  # 20 tiles
vips arrayjoin ... --across 5
```

## Outputs
- `tiles20/` raw tiles
- `tiles20_out/` white lines (green/red/purple)
- `assembled_lines_white_vips.jpg` / `assembled_overlay_preview.jpg`
