#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("F:/aw_cap")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_depth_preview(depth_path: Path, *, update_sidecar: bool) -> Path:
    import numpy as np
    from PIL import Image

    preview_path = depth_path.with_name(f"{depth_path.stem}__depth_preview.png")
    depth = np.load(depth_path)
    finite = np.isfinite(depth)
    if finite.any():
        values = depth[finite]
        near = float(np.percentile(values, 2.0))
        far = float(np.percentile(values, 98.0))
        if far <= near:
            far = near + 1.0
        preview = np.clip((depth - near) / (far - near), 0.0, 1.0)
        preview[~finite] = 0.0
        preview_u8 = (preview * 255.0).astype(np.uint8)
    else:
        near = 0.0
        far = 0.0
        preview_u8 = np.zeros_like(depth, dtype=np.uint8)
    Image.fromarray(preview_u8, mode="L").save(preview_path)

    if update_sidecar:
        sidecar_path = depth_path.with_suffix(".json")
        if sidecar_path.exists():
            sidecar = read_json(sidecar_path)
            sidecar["depth_preview_path"] = str(preview_path)
            sidecar["depth_preview_for_debug_only"] = False
            sidecar["depth_preview_percentile_clip"] = [2.0, 98.0]
            sidecar["depth_preview_near_m"] = near
            sidecar["depth_preview_far_m"] = far
            write_json(sidecar_path, sidecar)
    return preview_path


def update_depth_sidecar(depth_path: Path, preview_path: Path) -> None:
    import numpy as np

    sidecar_path = depth_path.with_suffix(".json")
    if not sidecar_path.exists():
        return
    depth = np.load(depth_path)
    finite = np.isfinite(depth)
    if finite.any():
        values = depth[finite]
        near = float(np.percentile(values, 2.0))
        far = float(np.percentile(values, 98.0))
        if far <= near:
            far = near + 1.0
    else:
        near = 0.0
        far = 0.0
    sidecar = read_json(sidecar_path)
    sidecar["depth_preview_path"] = str(preview_path)
    sidecar["depth_preview_for_debug_only"] = False
    sidecar["depth_preview_percentile_clip"] = [2.0, 98.0]
    sidecar["depth_preview_near_m"] = near
    sidecar["depth_preview_far_m"] = far
    write_json(sidecar_path, sidecar)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write grayscale preview PNGs next to AirSim depth .npy outputs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-update-sidecar", dest="update_sidecar", action="store_false", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    depth_paths = sorted(Path(args.root).rglob("depth/tick_*.npy"))
    written: list[str] = []
    skipped = 0
    for depth_path in depth_paths:
        preview_path = depth_path.with_name(f"{depth_path.stem}__depth_preview.png")
        if preview_path.exists() and not args.overwrite:
            if bool(args.update_sidecar):
                update_depth_sidecar(depth_path, preview_path)
            skipped += 1
            continue
        written.append(str(write_depth_preview(depth_path, update_sidecar=bool(args.update_sidecar))))
    print(json.dumps({"root": str(args.root), "depth_files": len(depth_paths), "written": len(written), "skipped": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
