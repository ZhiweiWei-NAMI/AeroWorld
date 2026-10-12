#!/usr/bin/env python3
"""Build deterministic low-altitude domain-state supplement artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.semantic_simulation.compute_comm import (  # noqa: E402
    build_communication_state_rows,
)
from Dataset.semantic_simulation.domain_state import (  # noqa: E402
    DomainStateSimulationError,
    SCHEMA_VERSION,
    build_episode_artifacts,
    check_episode_outputs,
    load_domain_state_profile,
    write_episode_outputs,
)
from Dataset.semantic_truth.provenance import canonical_json  # noqa: E402


DEFAULT_PROFILE_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "domain_state_supplement_profile.json"
)
DEFAULT_COMPUTE_COMM_PROFILE_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "compute_comm_supplement_profile.json"
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    default_profile = load_domain_state_profile(DEFAULT_PROFILE_PATH)
    parser = argparse.ArgumentParser(
        description="Build mechanism-based domain-state observations for semantic truth rules."
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument(
        "--episodes-root",
        type=Path,
        default=Path(default_profile["default_input_root"]),
        help="Root containing render-ready capture-filtered episode directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(default_profile["default_output_root"]),
        help="Root for domain-state supplement outputs.",
    )
    parser.add_argument(
        "--episode-id",
        action="append",
        default=[],
        help="Specific episode id to process. May be supplied multiple times.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every matched episode. Without this flag, only one pilot episode is built.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if deterministic outputs are missing or stale.",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def selected_episode_roots(args: argparse.Namespace) -> list[Path]:
    root = args.episodes_root
    if not root.is_dir():
        raise DomainStateSimulationError(f"episodes root is missing: {root}")
    if args.episode_id:
        episodes = [root / episode_id for episode_id in args.episode_id]
    else:
        episodes = [
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "episode_manifest.json").is_file()
        ]
    missing = [str(path) for path in episodes if not path.is_dir()]
    if missing:
        raise DomainStateSimulationError(f"selected episode directories are missing: {missing}")
    episodes.sort(key=lambda path: path.name)
    if args.limit > 0:
        episodes = episodes[: args.limit]
    if not args.all and not args.episode_id:
        episodes = episodes[:1]
    if not episodes:
        raise DomainStateSimulationError("no episodes matched the requested selection")
    return episodes


def _process_episode(job: tuple[Path, Path, Path, bool]) -> dict[str, object]:
    episode_root, output_root, profile_path, check = job
    output_dir = output_root / episode_root.name
    communication_rows = build_communication_state_rows(
        episode_root.resolve(),
        DEFAULT_COMPUTE_COMM_PROFILE_PATH,
    )
    artifacts = build_episode_artifacts(
        episode_root.resolve(),
        output_dir.resolve(),
        profile_path,
        communication_rows=communication_rows,
    )
    if check:
        mismatches = check_episode_outputs(artifacts)
        if mismatches:
            raise DomainStateSimulationError(
                f"{episode_root.name}: domain-state outputs are not up to date: {mismatches}"
            )
    else:
        write_episode_outputs(artifacts)
    return {
        "episode_id": artifacts.episode_id,
        "output_dir": str(output_dir.resolve()),
        "record_counts": artifacts.summary["record_counts"],
        "artifact_bytes": sum(len(text.encode("utf-8")) for text in artifacts.files.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.workers < 1:
        raise DomainStateSimulationError("--workers must be at least 1")
    profile_path = args.profile.resolve()
    profile = load_domain_state_profile(profile_path)
    roots = selected_episode_roots(args)
    jobs = [(root, args.output_root.resolve(), profile_path, bool(args.check)) for root in roots]
    if args.workers == 1:
        processed = [_process_episode(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            processed = list(executor.map(_process_episode, jobs))
    processed.sort(key=lambda item: str(item["episode_id"]))
    totals: dict[str, int] = {}
    for item in processed:
        for name, value in dict(item["record_counts"]).items():
            totals[name] = totals.get(name, 0) + int(value)
    corpus_summary = {
        "schema_name": "domain_state_supplement_corpus_summary",
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile["profile_id"],
        "parameter_governance": profile["parameter_governance"],
        "episode_count": len(processed),
        "record_counts": dict(sorted(totals.items())),
        "artifact_bytes": sum(int(item["artifact_bytes"]) for item in processed),
        "episodes": processed,
    }
    full_corpus = bool(args.all and not args.episode_id and args.limit == 0)
    if full_corpus:
        summary_path = args.output_root.resolve() / "corpus_summary.json"
        summary_text = json.dumps(
            corpus_summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
        if args.check:
            if not summary_path.is_file() or summary_path.read_text(encoding="utf-8-sig") != summary_text:
                raise DomainStateSimulationError(f"corpus summary is missing or stale: {summary_path}")
        else:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(summary_text, encoding="utf-8", newline="\n")
    print(
        canonical_json(
            {
                "status": "ok",
                "mode": "check" if args.check else "write",
                "processed_episode_count": len(processed),
                "record_counts": dict(sorted(totals.items())),
                "artifact_bytes": corpus_summary["artifact_bytes"],
                "episodes": processed,
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DomainStateSimulationError as exc:
        print(f"domain-state supplement failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
