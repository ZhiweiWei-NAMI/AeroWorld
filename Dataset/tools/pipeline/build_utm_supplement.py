#!/usr/bin/env python3
"""Build deterministic L0 UTM logs for selected formal episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.semantic_simulation.utm_state import (  # noqa: E402
    UtmStateError,
    build_utm_episode_artifacts,
    check_utm_episode_artifacts,
    write_utm_episode_artifacts,
)
from Dataset.semantic_truth.provenance import canonical_json  # noqa: E402


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-root",
        type=Path,
        default=PROJECT_ROOT / "aw_data" / "render_ready_episodes_capture_filtered",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "aw_data" / "utm_supplement",
    )
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not args.episode_id:
        raise UtmStateError("at least one --episode-id is required; bulk 210-episode execution is forbidden")
    results: list[dict[str, object]] = []
    for episode_id in sorted(set(args.episode_id)):
        episode_root = args.episodes_root.resolve() / episode_id
        if not episode_root.is_dir():
            raise UtmStateError(f"episode directory is missing: {episode_root}")
        artifacts = build_utm_episode_artifacts(
            episode_root,
            args.output_root.resolve() / episode_id,
        )
        if args.check:
            mismatches = check_utm_episode_artifacts(artifacts)
            if mismatches:
                raise UtmStateError(f"{episode_id}: UTM outputs are stale: {mismatches}")
        else:
            write_utm_episode_artifacts(artifacts)
        results.append(
            {
                "episode_id": episode_id,
                "record_counts": artifacts.summary["record_counts"],
                "input_digest": artifacts.summary["input_digest"],
            }
        )
    print(
        canonical_json(
            {
                "status": "ok",
                "mode": "check" if args.check else "write",
                "episode_count": len(results),
                "episodes": results,
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UtmStateError as exc:
        print(f"UTM supplement failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
