#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_CAPTURE_ROOT = Path("F:/aw_cap")
DEFAULT_SUMMARY = Path("F:/aw_cap_summary.csv")
DEFAULT_OUTPUT_DIR = Path("F:/aw_cap/test_fig")
DEFAULT_TICKS = [220, 230, 240, 250, 260]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def first_ok_uav_view(rows: list[dict[str, str]], episode: str) -> str:
    views = sorted(
        {
            str(row.get("output_dir") or "")
            for row in rows
            if str(row.get("episode") or "") == episode
            and str(row.get("view") or "") == "uav_event_chain"
            and str(row.get("status") or "") == "ok"
            and str(row.get("output_dir") or "")
        }
    )
    if not views:
        raise RuntimeError(f"No ok uav_event_chain rows found for episode {episode!r}")
    return views[0]


def depth_to_preview(depth_path: Path, preview_path: Path) -> Path:
    import numpy as np
    from PIL import Image

    depth = np.load(depth_path)
    finite = np.isfinite(depth)
    if finite.any():
        values = depth[finite]
        near = float(np.percentile(values, 2.0))
        far = float(np.percentile(values, 98.0))
        if far <= near:
            far = near + 1.0
        normalized = np.clip((depth - near) / (far - near), 0.0, 1.0)
        normalized[~finite] = 0.0
        preview = (normalized * 255.0).astype(np.uint8)
    else:
        preview = np.zeros_like(depth, dtype=np.uint8)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview, mode="L").save(preview_path)
    return preview_path


def ensure_depth_preview(depth_path: Path) -> Path:
    sidecar_path = depth_path.with_suffix(".json")
    if sidecar_path.exists():
        sidecar = read_json(sidecar_path)
        preview = str(sidecar.get("depth_preview_path") or "").strip()
        if preview and Path(preview).exists():
            return Path(preview)
    preview_path = depth_path.with_name(f"{depth_path.stem}__depth_preview.png")
    if preview_path.exists():
        return preview_path
    return depth_to_preview(depth_path, preview_path)


def build_pre_figure(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    rows = load_summary_rows(args.summary)
    view_root = Path(args.uav_view_root) if args.uav_view_root else Path(first_ok_uav_view(rows, args.episode))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ticks = [int(tick) for tick in args.ticks]
    cell_w = int(args.cell_width)
    cell_h = int(args.cell_height)
    label_h = 42
    row_label_w = 90

    try:
        font = ImageFont.truetype("arial.ttf", 22)
        label_font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    canvas = Image.new("RGB", (row_label_w + cell_w * len(selected_ticks), (cell_h + label_h) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    created_depth_previews: list[str] = []
    source_paths: dict[str, dict[int, str]] = {"rgb": {}, "depth": {}, "seg": {}}

    for row_index, modality in enumerate(("rgb", "depth", "seg")):
        y0 = row_index * (cell_h + label_h)
        draw.text((14, y0 + cell_h // 2 - 12), modality.upper(), fill=(20, 20, 20), font=label_font)
        for col_index, tick in enumerate(selected_ticks):
            x0 = row_label_w + col_index * cell_w
            stem = f"tick_{tick:06d}"
            if modality == "rgb":
                source_path = view_root / "rgb" / f"{stem}.png"
                image_path = source_path
            elif modality == "depth":
                source_path = view_root / "depth" / f"{stem}.npy"
                image_path = ensure_depth_preview(source_path)
                created_depth_previews.append(str(image_path))
            else:
                source_path = view_root / "seg" / f"{stem}__palette.png"
                image_path = source_path
                if not image_path.exists():
                    source_path = view_root / "seg" / f"{stem}.png"
                    image_path = source_path
            if not image_path.exists():
                raise RuntimeError(f"Missing {modality} source for tick {tick}: {image_path}")
            source_paths[modality][tick] = str(source_path)
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            paste_x = x0 + (cell_w - image.width) // 2
            paste_y = y0 + (cell_h - image.height) // 2
            canvas.paste(image, (paste_x, paste_y))
            draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline=(210, 210, 210), width=1)
            text = f"tick={tick}"
            bbox = draw.textbbox((0, 0), text, font=font)
            draw.text((x0 + (cell_w - (bbox[2] - bbox[0])) // 2, y0 + cell_h + 8), text, fill=(20, 20, 20), font=font)

    figure_path = output_dir / str(args.figure_name)
    canvas.save(figure_path)
    metadata = {
        "figure_path": str(figure_path),
        "capture_root": str(args.capture_root),
        "summary": str(args.summary),
        "episode": str(args.episode),
        "uav_view_root": str(view_root),
        "ticks": selected_ticks,
        "layout": "3 rows (RGB, depth preview, segmentation palette) x 5 differentiated ticks",
        "source_paths": source_paths,
        "created_depth_previews": sorted(set(created_depth_previews)),
    }
    metadata_path = output_dir / "pre_figure_manifest.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 3x5 Pre figure from existing AeroWorld multimodal UAV outputs.")
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode", default="L1-1_v1__seed00")
    parser.add_argument("--uav-view-root", type=Path, default=None)
    parser.add_argument("--ticks", nargs="+", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--cell-width", type=int, default=384)
    parser.add_argument("--cell-height", type=int, default=216)
    parser.add_argument("--figure-name", default="pre_multimodal_ticks_3x5.png")
    return parser.parse_args()


def main() -> int:
    metadata = build_pre_figure(parse_args())
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
