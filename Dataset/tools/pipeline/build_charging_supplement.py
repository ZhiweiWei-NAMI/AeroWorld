#!/usr/bin/env python3
"""Build or check deterministic charging service plans for formal episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.semantic_truth.charging_supplement import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    check_charging_supplement,
    materialize_charging_supplement,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes-root",
        type=Path,
        default=PROJECT_ROOT / "aw_data" / "render_ready_episodes_capture_filtered",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    available = {
        path.name: path
        for path in args.episodes_root.resolve().iterdir()
        if path.is_dir() and (path / "episode_manifest.json").is_file()
    }
    selected = sorted(available) if args.all else sorted(set(args.episode))
    if not selected:
        raise SystemExit("select --episode or --all")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise SystemExit(f"unknown episodes: {missing}")
    mismatches: list[str] = []
    summaries: list[dict[str, object]] = []
    for episode_id in selected:
        episode_root = available[episode_id]
        if args.check:
            mismatches.extend(
                check_charging_supplement(episode_root, args.output_root)
            )
        else:
            summaries.append(
                materialize_charging_supplement(episode_root, args.output_root)
            )
    result = {
        "mode": "check" if args.check else "write",
        "episode_count": len(selected),
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "summaries": summaries,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
