#!/usr/bin/env python3
"""Create a deterministic multi-pass capture schedule.

The scheduler deliberately emits one camera/modality per task. Ground top-view
tasks are grouped by site because each site has one active overview camera.
UAV tasks are per episode and per vehicle camera so a multi-UAV scenario does
not capture multiple onboard cameras in the same render pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from batch_render_dataset import discover_source_episodes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = PROJECT_ROOT / "DynamicCityCreatorEx.uproject"
DEFAULT_UE_ROOT = Path(r"E:\UE_5.2")
DEFAULT_RENDER_READY_ROOT = Path(r"D:\AeroWorldCapture\render_ready_episodes")
DEFAULT_OUTPUT_ROOT = Path(r"D:\AeroWorldCapture\full_collection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan deterministic full dataset capture tasks.")
    parser.add_argument("--episodes-root", type=Path, default=Path("Dataset/episodes"))
    parser.add_argument("--render-ready-root", type=Path, default=DEFAULT_RENDER_READY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ue-root", type=Path, default=DEFAULT_UE_ROOT)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--ground-chunk-size", type=int, default=3)
    parser.add_argument("--tick-stride", type=int, default=10)
    parser.add_argument("--uav-modalities", action="append", default=None)
    parser.add_argument("--startup-timeout-s", type=int, default=900)
    parser.add_argument("--rpc-timeout-s", type=int, default=180)
    parser.add_argument("--batch-timeout-s", type=int, default=1800)
    parser.add_argument("--include-private", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_site_id(render_episode_dir: Path) -> str:
    plan = load_json(render_episode_dir / "scenario_plan.json")
    summary = dict(plan.get("compiled_plan_summary") or {})
    site_contracts = dict(summary.get("site_contracts") or {})
    if site_contracts:
        return str(sorted(site_contracts)[0])
    manifest = load_json(render_episode_dir / "episode_manifest.json")
    return str(manifest.get("site_id") or "site.intersection_a")


def overview_camera_for_site(site_id: str) -> str:
    normalized = str(site_id).strip().lower()
    if normalized.endswith("intersection_b") or normalized.endswith("_b"):
        return "ground_b_overview_top"
    return "ground_a_overview_top"


def runtime_uav_entity_ids(render_episode_dir: Path) -> list[str]:
    roster = load_json(render_episode_dir / "global_entity_roster.json")
    if isinstance(roster, dict):
        entries = roster.get("entities") or []
    elif isinstance(roster, list):
        entries = roster
    else:
        entries = []
    return sorted(
        str(entity.get("entity_id") or "")
        for entity in entries
        if isinstance(entity, dict)
        and str(entity.get("entity_id") or "")
        and str(entity.get("mode") or "").strip() == "runtime_multirotor"
    )


def chunked(values: list[str], chunk_size: int) -> list[list[str]]:
    size = max(1, int(chunk_size))
    return [values[index : index + size] for index in range(0, len(values), size)]


def run_command(args: argparse.Namespace, task: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "Dataset" / "tools" / "run_auto_pie_capture.py"),
        "--ue-root",
        str(args.ue_root),
        "--project",
        str(args.project),
        "--episodes-root",
        str(args.episodes_root),
        "--render-ready-root",
        str(args.render_ready_root),
        "--output-root",
        str(task["output_root"]),
        "--chunk-size",
        str(int(task["chunk_size"])),
        "--tick-stride",
        str(int(args.tick_stride)),
        "--camera-role",
        str(task["camera_role"]),
        "--camera-id",
        str(task["camera_id"]),
        "--modality",
        str(task["modality"]),
        "--startup-timeout-s",
        str(int(args.startup_timeout_s)),
        "--rpc-timeout-s",
        str(int(args.rpc_timeout_s)),
        "--batch-timeout-s",
        str(int(args.batch_timeout_s)),
        "--retries",
        "0",
        "--overwrite",
    ]
    if args.include_private:
        command.append("--include-private")
    for episode_id in task["episodes"]:
        command.extend(["--episode", str(episode_id)])
    return command


def main() -> None:
    args = parse_args()
    episode_dirs = discover_source_episodes(
        args.episodes_root,
        list(args.episode or []),
        include_private=bool(args.include_private),
    )
    episode_dirs = episode_dirs[int(args.start) : args.end]
    if int(args.max_episodes) > 0:
        episode_dirs = episode_dirs[: int(args.max_episodes)]
    episode_ids = [path.name for path in episode_dirs]

    render_ready_root = Path(args.render_ready_root)
    output_root = Path(args.output_root)
    tasks: list[dict[str, Any]] = []

    requested_modalities = args.uav_modalities if args.uav_modalities is not None else ["rgb", "depth", "seg"]
    uav_modalities = [str(value).strip() for value in requested_modalities if str(value).strip()]
    for repeat_index in range(max(1, int(args.repeat_count))):
        repeat_id = f"repeat_{repeat_index:02d}"
        episodes_by_ground_camera: dict[str, list[str]] = {}
        for episode_id in episode_ids:
            render_dir = render_ready_root / episode_id
            camera_id = overview_camera_for_site(episode_site_id(render_dir))
            episodes_by_ground_camera.setdefault(camera_id, []).append(episode_id)

        for camera_id, grouped_episodes in sorted(episodes_by_ground_camera.items()):
            for chunk_index, chunk in enumerate(chunked(sorted(grouped_episodes), int(args.ground_chunk_size))):
                task = {
                    "task_id": f"{repeat_id}__ground_top_rgb__{camera_id}__chunk_{chunk_index:03d}",
                    "repeat_id": repeat_id,
                    "pass_id": "ground_top_rgb",
                    "camera_role": "ground",
                    "camera_id": camera_id,
                    "modality": "rgb",
                    "chunk_size": len(chunk),
                    "episodes": chunk,
                    "output_root": str(output_root / repeat_id / "ground_rgb_top"),
                    "one_camera_per_run": True,
                    "coverage_policy": "episode_event_bbox_aligned_overview",
                }
                task["command"] = run_command(args, task)
                tasks.append(task)

        for episode_id in sorted(episode_ids):
            render_dir = render_ready_root / episode_id
            for entity_id in runtime_uav_entity_ids(render_dir):
                camera_id = f"{entity_id}__nadir_down"
                for modality in uav_modalities:
                    task = {
                        "task_id": f"{repeat_id}__uav_nadir_{modality}__{episode_id}__{camera_id}",
                        "repeat_id": repeat_id,
                        "pass_id": f"uav_nadir_{modality}",
                        "camera_role": "uav",
                        "camera_id": camera_id,
                        "modality": modality,
                        "chunk_size": 1,
                        "episodes": [episode_id],
                        "output_root": str(output_root / repeat_id / f"uav_nadir_{modality}"),
                        "one_camera_per_run": True,
                        "coverage_policy": "single_runtime_uav_nadir_camera",
                    }
                    task["command"] = run_command(args, task)
                    tasks.append(task)

    schedule = {
        "schema": "aeroworld_capture_schedule_v1",
        "output_root": str(output_root),
        "render_ready_root": str(render_ready_root),
        "episode_count": len(episode_ids),
        "repeat_count": max(1, int(args.repeat_count)),
        "tick_stride": int(args.tick_stride),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    schedule_path = output_root / "capture_schedule.json"
    schedule_path.write_text(json.dumps(schedule, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"schedule_path": str(schedule_path), "task_count": len(tasks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
