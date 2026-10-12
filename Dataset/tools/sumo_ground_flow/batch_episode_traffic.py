"""Batch scheduler for per-episode SUMO traffic generation and review.

This wrapper intentionally drives the existing episode-local tools through
subprocesses so a single stuck SUMO/TraCI run cannot poison the full 210-episode
batch.  It writes a compact status ledger under the SUMO output root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EPISODES_ROOT = ROOT / "Dataset" / "episodes"
DEFAULT_OUTPUT_ROOT = ROOT / "aw_data" / "sumo_outputs" / "episode_vehicle_authority"
DEFAULT_STATUS_PATH = DEFAULT_OUTPUT_ROOT / "_batch_traffic_status.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _episode_ids(episodes_root: Path, requested: Sequence[str]) -> list[str]:
    if requested:
        return sorted({str(item) for item in requested if str(item)})
    return [
        path.name
        for path in sorted(Path(episodes_root).iterdir())
        if path.is_dir() and (path / "episode_manifest.json").exists()
    ]


def _run_command(command: Sequence[str], *, cwd: Path, timeout_s: float) -> dict[str, Any]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            list(command),
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=float(timeout_s),
            check=False,
        )
        return {
            "returncode": int(proc.returncode),
            "elapsed_s": round(time.monotonic() - start, 3),
            "stdout_tail": proc.stdout[-6000:],
            "stderr_tail": proc.stderr[-6000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "elapsed_s": round(time.monotonic() - start, 3),
            "stdout_tail": (exc.stdout or "")[-6000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-6000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def _parse_json_from_stdout(text: str) -> Any:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _classify_failure(step: str, command_result: dict[str, Any], parsed: Any = None) -> str:
    if command_result.get("timed_out"):
        return f"{step}_timeout"
    stderr = str(command_result.get("stderr_tail") or "")
    stdout = str(command_result.get("stdout_tail") or "")
    text = f"{stderr}\n{stdout}"
    errors = []
    if isinstance(parsed, dict):
        errors = [str(item) for item in parsed.get("errors") or []]
        text = "\n".join([text, *errors])
    lowered = text.lower()
    if "semantic vehicle sample" in lowered or "fixed semantic lane" in lowered:
        return "semantic_lane_projection"
    if "accepted background vehicles below target" in lowered or "accepted explicit background vehicles below target" in lowered:
        if "inactive candidates" in lowered and "filtered candidates: 0" in lowered:
            return "accepted_background_below_target_inactive"
        return "accepted_background_below_target_route_filter"
    if "roi background coverage missing" in lowered:
        return "roi_background_missing"
    if "protected windows" in lowered or "protected-window" in lowered or "during protected windows" in lowered:
        return "protected_window_intrusion"
    if "semantic vehicles visible outside" in lowered or "outside their control windows" in lowered:
        return "semantic_outside_control_window"
    if "opposite sumo lane direction" in lowered or "reverse lane" in lowered:
        return "semantic_lane_direction"
    if "no roi-visible sumo endpoint candidates" in lowered or "outside-roi depart" in lowered:
        return "plan_endpoint_candidate"
    if "sumo_findroute_rejected_background_vehicle" in lowered:
        return "sumo_route_filter"
    if command_result.get("returncode") not in (0, None):
        return f"{step}_failed"
    return "ok"


def _step_commands(
    episode_id: str,
    *,
    episodes_root: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, list[str]]:
    python = sys.executable
    output_dir = output_root / episode_id
    run_command = [
        python,
        "-m",
        "Dataset.tools.sumo_ground_flow.run_traffic",
        "--output-dir",
        str(output_root),
        "--semantic-plan-root",
        str(episodes_root),
        "--episode",
        episode_id,
    ]
    if overwrite:
        run_command.append("--overwrite")
    visualize_command = [
        python,
        "-m",
        "Dataset.tools.sumo_ground_flow.visualize_traffic",
        "--output-dir",
        str(output_dir),
        "--bounds-mode",
        "episode-expanded-roi",
        "--episode-only",
        "--frame-stride",
        "5",
        "--output-suffix",
        "roi_extended_episode_only",
        "--overwrite",
    ]
    return {
        "plan": [
            python,
            "-m",
            "Dataset.tools.sumo_ground_flow.explicit_vehicle_plan",
            "--episodes-root",
            str(episodes_root),
            "--episode",
            episode_id,
        ],
        "run": run_command,
        "validate": [
            python,
            "-m",
            "Dataset.tools.sumo_ground_flow.validate_traffic_output",
            str(output_dir),
        ],
        "visualize": visualize_command,
    }


def process_episode(
    episode_id: str,
    *,
    episodes_root: Path,
    output_root: Path,
    phases: set[str],
    overwrite: bool,
    visualize_policy: str,
    timeout_s: float,
) -> dict[str, Any]:
    commands = _step_commands(
        episode_id,
        episodes_root=episodes_root,
        output_root=output_root,
        overwrite=overwrite,
    )
    result: dict[str, Any] = {
        "episode": episode_id,
        "ok": False,
        "failure_category": "",
        "steps": {},
    }

    for step in ("plan", "run"):
        if step not in phases:
            continue
        step_result = _run_command(commands[step], cwd=ROOT, timeout_s=timeout_s)
        result["steps"][step] = step_result
        if step_result.get("returncode") != 0:
            result["failure_category"] = _classify_failure(step, step_result)
            return result

    run_steps_ok = not any(
            result.get("steps", {}).get(step, {}).get("returncode") not in (0, None)
            for step in ("plan", "run")
            if step in result.get("steps", {})
    )

    validate_needs_visualization = "validate" in phases and "visualize" in phases
    should_visualize_before_validate = "visualize" in phases and (
        visualize_policy == "always"
        or (visualize_policy == "on-run-success" and run_steps_ok)
        or (visualize_policy == "on-ok" and validate_needs_visualization and run_steps_ok)
    )
    if should_visualize_before_validate:
        step_result = _run_command(commands["visualize"], cwd=ROOT, timeout_s=timeout_s)
        result["steps"]["visualize"] = step_result
        if step_result.get("returncode") != 0 and not result.get("failure_category"):
            result["failure_category"] = _classify_failure("visualize", step_result)

    validation_payload = None
    if "validate" in phases:
        step_result = _run_command(commands["validate"], cwd=ROOT, timeout_s=timeout_s)
        validation_payload = _parse_json_from_stdout(str(step_result.get("stdout_tail") or ""))
        result["steps"]["validate"] = step_result
        result["validation"] = validation_payload
        if step_result.get("returncode") != 0:
            result["failure_category"] = _classify_failure("validate", step_result, validation_payload)
        elif isinstance(validation_payload, dict) and not validation_payload.get("ok"):
            result["failure_category"] = _classify_failure("validate", step_result, validation_payload)

    should_visualize_after_validate = "visualize" in phases and not should_visualize_before_validate and (
        visualize_policy == "on-ok" and not result.get("failure_category")
    )
    if should_visualize_after_validate:
        step_result = _run_command(commands["visualize"], cwd=ROOT, timeout_s=timeout_s)
        result["steps"]["visualize"] = step_result
        if step_result.get("returncode") != 0 and not result.get("failure_category"):
            result["failure_category"] = _classify_failure("visualize", step_result)

    gif_path = output_root / episode_id / "sumo_topo_vehicles_roi_extended_episode_only.gif"
    vis_json_path = output_root / episode_id / "sumo_traffic_visualization_roi_extended_episode_only.json"
    result["has_episode_only_gif"] = gif_path.exists()
    result["has_episode_only_visualization_json"] = vis_json_path.exists()
    visualization_required = "visualize" in phases
    result["ok"] = not bool(result.get("failure_category")) and (
        bool(result["has_episode_only_gif"]) if visualization_required else True
    )
    if visualization_required and not result["ok"] and not result.get("failure_category"):
        result["failure_category"] = "missing_episode_only_gif"
    return result


def _summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        key = "ok" if result.get("ok") else str(result.get("failure_category") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    examples: dict[str, list[str]] = {}
    for result in results:
        key = "ok" if result.get("ok") else str(result.get("failure_category") or "unknown")
        examples.setdefault(key, [])
        if len(examples[key]) < 20:
            examples[key].append(str(result.get("episode")))
    return {
        "total": len(results),
        "counts": counts,
        "examples": examples,
    }


def _flush_status(status_path: Path, results: Sequence[dict[str, Any]]) -> None:
    _write_json(
        status_path,
        {
            "schema": "aero_sumo_batch_traffic_status_v1",
            "updated_at_unix_s": round(time.time(), 3),
            "summary": _summarize(results),
            "results": list(results),
        },
    )


def _parse_phases(values: Sequence[str]) -> set[str]:
    if not values or "all" in values:
        return {"plan", "run", "validate", "visualize"}
    allowed = {"plan", "run", "validate", "visualize"}
    phases = {str(value) for value in values}
    unknown = sorted(phases - allowed)
    if unknown:
        raise ValueError(f"unknown phases: {unknown}; allowed={sorted(allowed)}")
    return phases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch generate and review episode-local SUMO traffic.")
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_EPISODES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--phase", action="append", default=[], help="all, plan, run, validate, visualize")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--visualize-policy",
        choices=("on-ok", "on-run-success", "always"),
        default="on-run-success",
    )
    parser.add_argument("--flush-every", type=int, default=1)
    args = parser.parse_args(argv)

    episodes = _episode_ids(args.episodes_root, args.episode)
    phases = _parse_phases(args.phase)
    results: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    flush_every = max(1, int(args.flush_every))

    def run_one(episode_id: str) -> dict[str, Any]:
        return process_episode(
            episode_id,
            episodes_root=args.episodes_root,
            output_root=args.output_root,
            phases=phases,
            overwrite=bool(args.overwrite),
            visualize_policy=str(args.visualize_policy),
            timeout_s=float(args.timeout_s),
        )

    if workers == 1:
        for episode_id in episodes:
            result = run_one(episode_id)
            results.append(result)
            print(json.dumps({k: result.get(k) for k in ("episode", "ok", "failure_category")}, ensure_ascii=False), flush=True)
            if len(results) % flush_every == 0:
                _flush_status(args.status_path, results)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(run_one, episode_id): episode_id for episode_id in episodes}
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    episode_id = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "episode": episode_id,
                            "ok": False,
                            "failure_category": "batch_scheduler_exception",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    results.append(result)
                    print(json.dumps({k: result.get(k) for k in ("episode", "ok", "failure_category")}, ensure_ascii=False), flush=True)
                    if len(results) % flush_every == 0:
                        _flush_status(args.status_path, results)

    _flush_status(args.status_path, results)
    summary = _summarize(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["counts"].get("ok", 0) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
