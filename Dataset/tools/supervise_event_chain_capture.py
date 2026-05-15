#!/usr/bin/env python3
"""Long-run supervisor for formal event-chain capture.

This script intentionally delegates capture work to
``run_semantic_event_chain_every10.py``.  It adds episode-by-episode retry and
fallback chunk sizing without duplicating UE, AirSim, or render-host logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "Dataset" / "tools" / "run_semantic_event_chain_every10.py"
DEFAULT_EPISODES_ROOT = ROOT / "Dataset" / "render_ready_episodes_capture_filtered"
DEFAULT_OUTPUT_ROOT = Path("F:/aw_cap")
DEFAULT_SUMMARY = Path("F:/aw_cap_summary.csv")


def is_f_drive_path(path: Path) -> bool:
    text = str(path.resolve() if path.is_absolute() else path).replace("\\", "/").lower()
    return text.startswith("f:/")


def parse_csv_ints(value: str) -> list[int]:
    values: list[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("fallback chunk sizes must be positive")
        values.append(parsed)
    return values


def episode_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"Episode root does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "render_host_config.json").exists())


def selected_episode_indexes(root: Path, names: Sequence[str], start_index: int, limit: int) -> list[tuple[int, Path]]:
    dirs = episode_dirs(root)
    if names:
        wanted = list(names)
        by_name = {path.name: (index, path) for index, path in enumerate(dirs)}
        missing = sorted(name for name in wanted if name not in by_name)
        if missing:
            raise SystemExit(f"Missing capture-filtered episodes: {missing}")
        selected = [by_name[name] for name in wanted]
    else:
        selected = [(index, path) for index, path in enumerate(dirs)]
        selected = selected[max(0, int(start_index)) :]
        if limit > 0:
            selected = selected[:limit]
    return selected


def base_runner_command(args: argparse.Namespace, *, episode_index: int, chunk_size: int) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--episodes-root",
        str(args.episodes_root),
        "--output-root",
        str(args.output_root),
        "--summary",
        str(args.summary),
        "--rules",
        str(args.rules),
        "--contract",
        str(args.contract),
        "--capture-presets",
        str(args.capture_presets),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--start-index",
        str(episode_index),
        "--limit",
        "1",
        "--tick-start",
        "0",
        "--tick-end",
        "900",
        "--tick-step",
        "5",
        "--simulation-tick-stride",
        "5",
        "--capture-ticks-per-host-run",
        str(chunk_size),
        "--max-working-set-gb",
        "18.0",
        "--max-private-memory-gb",
        "18.0",
        "--uav-capture-backend",
        "editor_hook",
        "--uav-scene-control-backend",
        "truth_frame_scene_sync",
    ]
    if args.append_summary:
        command.append("--append-summary")
    if args.resume_completed_ok:
        command.append("--resume-completed-ok")
    if args.continue_on_error:
        command.append("--continue-on-error")
    return command


def validate_supervisor_args(args: argparse.Namespace) -> None:
    if not RUNNER.exists():
        raise SystemExit(f"Missing formal runner: {RUNNER}")
    if not is_f_drive_path(args.output_root):
        raise SystemExit(f"Formal output root must be on F:, got {args.output_root}")
    if not is_f_drive_path(args.summary):
        raise SystemExit(f"Formal summary must be on F:, got {args.summary}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Supervise formal event-chain capture episode by episode.")
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_EPISODES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--rules", type=Path, default=ROOT / "Config" / "LowAltitude" / "semantic_stencil_rules.json")
    parser.add_argument("--contract", type=Path, default=ROOT / "Config" / "LowAltitude" / "semantic_capture_runtime_contract.json")
    parser.add_argument("--capture-presets", type=Path, default=ROOT / "Plugins" / "SumoImporter" / "Scripts" / "episode_capture_presets.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--episode", action="append", default=[], help="Capture one named filtered episode; repeatable.")
    parser.add_argument("--capture-ticks-per-host-run", type=int, default=16)
    parser.add_argument("--fallback-capture-ticks-per-host-run", type=parse_csv_ints, default=parse_csv_ints("8,4,2,1"))
    parser.add_argument("--append-summary", dest="append_summary", action="store_true", default=True)
    parser.add_argument("--no-append-summary", dest="append_summary", action="store_false")
    parser.add_argument("--resume-completed-ok", dest="resume_completed_ok", action="store_true", default=True)
    parser.add_argument("--no-resume-completed-ok", dest="resume_completed_ok", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Print selected episodes and retry profile without touching UE.")
    args = parser.parse_args(argv)

    args.episodes_root = args.episodes_root.resolve()
    args.output_root = args.output_root.resolve()
    args.summary = args.summary.resolve()
    args.rules = args.rules.resolve()
    args.contract = args.contract.resolve()
    args.capture_presets = args.capture_presets.resolve()
    validate_supervisor_args(args)

    names = [name for group in args.episode for name in str(group).split(",") if name.strip()]
    selected = selected_episode_indexes(args.episodes_root, names, args.start_index, args.limit)
    chunk_sizes = [int(args.capture_ticks_per_host_run), *list(args.fallback_capture_ticks_per_host_run)]
    chunk_sizes = list(dict.fromkeys(size for size in chunk_sizes if size > 0))
    plan = {
        "episodes_root": str(args.episodes_root),
        "output_root": str(args.output_root),
        "summary": str(args.summary),
        "episode_count": len(selected),
        "chunk_retry_profile": chunk_sizes,
        "episodes": [{"index": index, "name": path.name} for index, path in selected],
    }
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    failures: list[dict[str, str]] = []
    for ordinal, (episode_index, path) in enumerate(selected, start=1):
        print(f"[supervisor] {ordinal}/{len(selected)} e{episode_index:02d} {path.name}", flush=True)
        last_return_code = 1
        for chunk_size in chunk_sizes:
            command = base_runner_command(args, episode_index=episode_index, chunk_size=chunk_size)
            print(f"[supervisor] running chunk_size={chunk_size}", flush=True)
            completed = subprocess.run(command, cwd=ROOT)
            last_return_code = int(completed.returncode)
            if last_return_code == 0:
                break
            print(f"[supervisor] failed rc={last_return_code}; trying smaller chunks if available", flush=True)
        if last_return_code != 0:
            failures.append({"episode": path.name, "return_code": str(last_return_code)})
            if not args.continue_on_error:
                print(json.dumps({"ok": False, "failures": failures}, ensure_ascii=False, indent=2), file=sys.stderr)
                return last_return_code

    print(json.dumps({"ok": not failures, "episode_count": len(selected), "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
