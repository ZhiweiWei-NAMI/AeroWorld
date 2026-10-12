"""Monitor SUMO semantic vehicle review artifacts for every episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EPISODES_ROOT = ROOT / "Dataset" / "episodes"
DEFAULT_REVIEW_ROOT = Path("G:/aw_cap/_meta/sumo_semantic_vehicle_review")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def episode_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "episode_manifest.json").exists())


def inspect_episode(episode_dir: Path, review_root: Path) -> dict[str, Any]:
    episode_id = episode_dir.name
    plan_path = episode_dir / "sumo_semantic_vehicle_plan.json"
    review_dir = review_root / episode_id
    md_path = review_dir / "semantic_vehicle_actions.md"
    action_json_path = review_dir / "semantic_vehicle_actions.json"
    gif_path = review_dir / "roi_vehicle_trajectories.gif"
    gif_meta_path = review_dir / "roi_vehicle_trajectories_review.json"
    issues: list[str] = []
    warnings: list[str] = []
    plan: dict[str, Any] = {}
    if not plan_path.exists():
        issues.append("missing_plan_json")
    else:
        try:
            plan = read_json(plan_path)
        except Exception as exc:
            issues.append(f"invalid_plan_json:{exc}")
    if not md_path.exists():
        issues.append("missing_review_markdown")
    if not action_json_path.exists():
        issues.append("missing_review_action_json")
    else:
        try:
            action_json = read_json(action_json_path)
            if plan and action_json.get("episode_id") != plan.get("episode_id"):
                issues.append("review_action_json_episode_mismatch")
        except Exception as exc:
            issues.append(f"invalid_review_action_json:{exc}")
    if not gif_path.exists():
        issues.append("missing_roi_trajectory_gif")
    gif_meta: dict[str, Any] = {}
    if not gif_meta_path.exists():
        issues.append("missing_roi_trajectory_gif_metadata")
    else:
        try:
            gif_meta = read_json(gif_meta_path)
        except Exception as exc:
            issues.append(f"invalid_roi_trajectory_gif_metadata:{exc}")
    if plan:
        if plan.get("policy", {}).get("all_vehicle_authority") != "sumo":
            issues.append("vehicle_authority_not_sumo")
        if any(str(issue) for issue in plan.get("issues") or []):
            issues.append(f"plan_route_or_instruction_issues:{len(plan.get('issues') or [])}")
        for event in plan.get("events") or []:
            status = str(event.get("status") or "")
            if status in {"vehicle_referenced_without_instruction", "needs_route_projection_fix"}:
                issues.append(f"event_not_accepted:{event.get('event_id')}:{status}")
            for inst in event.get("instructions") or []:
                call = str(inst.get("traci_call") or "")
                if call.startswith("ue.") or "render_ready.post_insert" in call:
                    issues.append(f"forbidden_instruction:{event.get('event_id')}:{call}")
                inst_type = str(inst.get("instruction_type") or "")
                args = dict(inst.get("args") or {})
                if inst_type == "vehicle.setRoute" and not list(args.get("edges") or []):
                    issues.append(f"empty_route_edges:{event.get('event_id')}:{args.get('vehID')}")
                if inst_type == "vehicle.setStop":
                    issues.append(f"legacy_setStop_instruction:{event.get('event_id')}:{args.get('vehID')}")
                if inst_type == "vehicle.moveToXY_series":
                    samples = list(args.get("samples") or [])
                    if str(args.get("control_mode") or "") != "forced_xy":
                        issues.append(f"forced_xy_missing_control_mode:{event.get('event_id')}:{args.get('vehID')}")
                    if not samples:
                        issues.append(f"forced_xy_missing_samples:{event.get('event_id')}:{args.get('vehID')}")
        if gif_meta:
            track_source = str(gif_meta.get("track_source") or "")
            track_counts = dict(gif_meta.get("track_counts") or {})
            if track_source != "render_ready_truth_frames":
                issues.append(f"gif_track_source_not_render_ready_truth_frames:{track_source or '<missing>'}")
            if int(track_counts.get("sumo_vehicle") or 0) <= 0:
                issues.append("gif_without_sumo_vehicle_tracks")
            if int(track_counts.get("plan_vehicle") or 0) != int(plan.get("vehicle_count") or 0):
                warnings.append(
                    "gif_plan_vehicle_count_mismatch:"
                    f"{int(track_counts.get('plan_vehicle') or 0)}!={int(plan.get('vehicle_count') or 0)}"
                )
            if int(plan.get("semantic_vehicle_count") or 0) > 0 and int(track_counts.get("semantic_vehicle") or 0) <= 0:
                issues.append("semantic_vehicle_episode_without_gif_semantic_vehicle_tracks")
            if int(track_counts.get("semantic_vehicle") or 0) != int(plan.get("semantic_vehicle_count") or 0):
                issues.append(
                    "gif_semantic_vehicle_count_mismatch:"
                    f"{int(track_counts.get('semantic_vehicle') or 0)}!={int(plan.get('semantic_vehicle_count') or 0)}"
                )
        has_uav_event = any(event.get("uav_targets") for event in plan.get("events") or [])
        if has_uav_event and gif_meta:
            track_counts = dict(gif_meta.get("track_counts") or {})
            if int(track_counts.get("uav") or 0) <= 0:
                issues.append("uav_event_without_gif_uav_tracks")
    return {
        "episode_id": episode_id,
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "plan_path": str(plan_path),
        "markdown_path": str(md_path),
        "action_json_path": str(action_json_path),
        "gif_path": str(gif_path),
        "gif_metadata_path": str(gif_meta_path),
        "gif_track_counts": dict(gif_meta.get("track_counts") or {}),
        "vehicle_count": int(plan.get("vehicle_count") or 0),
        "semantic_vehicle_count": int(plan.get("semantic_vehicle_count") or 0),
        "event_count": len(plan.get("events") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_EPISODES_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REVIEW_ROOT / "monitor_summary.json")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    rows = [inspect_episode(path, args.review_root) for path in episode_dirs(args.episodes_root.resolve())]
    summary = {
        "episodes": len(rows),
        "ok_episodes": sum(1 for row in rows if row["ok"]),
        "error_episodes": sum(1 for row in rows if not row["ok"]),
        "warning_episodes": sum(1 for row in rows if row.get("warnings")),
        "vehicles": sum(int(row["vehicle_count"]) for row in rows),
        "semantic_vehicles": sum(int(row["semantic_vehicle_count"]) for row in rows),
        "events": sum(int(row["event_count"]) for row in rows),
        "errors": [row for row in rows if not row["ok"]],
        "warnings": [row for row in rows if row.get("warnings")],
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_error and summary["error_episodes"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
