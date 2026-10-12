#!/usr/bin/env python3
"""Build objective semantic truth corpus artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.semantic_truth.objective_pipeline import (  # noqa: E402
    DEFAULT_EPISODES_ROOT,
    DEFAULT_OUTPUT_ROOT,
    ObjectivePipelineError,
    build_corpus,
    selected_episode_roots,
)
from Dataset.semantic_truth.provenance import canonical_json  # noqa: E402


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build objective semantic corpus artifacts."
    )
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_EPISODES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    selected_episode_ids = list(args.episode)
    if bool(selected_episode_ids) == bool(args.all):
        raise ObjectivePipelineError(
            "select exactly one mode: --all or at least one --episode"
        )
    episodes_root = args.episodes_root.absolute()
    output_root = args.output_root.absolute()
    default_episodes = DEFAULT_EPISODES_ROOT.resolve()
    default_output = DEFAULT_OUTPUT_ROOT.resolve()
    formal_input = bool(args.all) and episodes_root.resolve() == default_episodes
    formal_output = formal_input and output_root.resolve() == default_output
    if output_root.resolve() == default_output and not formal_output:
        raise ObjectivePipelineError(
            "canonical objective output requires --all with the default formal episodes root"
        )
    corpus_mode = (
        "formal_full"
        if formal_output
        else "selected_diagnostic"
        if selected_episode_ids
        else "custom_all"
    )
    roots = selected_episode_roots(
        episodes_root,
        episodes=selected_episode_ids,
        all_episodes=bool(args.all),
        formal=formal_input,
    )
    result = build_corpus(
        roots,
        output_root,
        corpus_mode=corpus_mode,
        check=bool(args.check),
        workers=int(args.workers),
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ObjectivePipelineError as exc:
        print(f"objective semantic corpus build failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
