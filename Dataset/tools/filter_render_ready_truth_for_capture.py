#!/usr/bin/env python3
"""Materialize formal UE replay truth packages without corrective filtering.

The full render-ready truth is the authoritative semantic record.  Formal UE
capture still consumes ``Dataset/render_ready_episodes_capture_filtered`` for
historical path compatibility, but this tool now copies the full truth through
unchanged and only rewrites package paths / render-host config for that formal
root.  Dynamic P/V/U visibility mistakes must be exposed by validators and fixed
in generators, not hidden by deleting frame records after conversion.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

try:
    import orjson
except ModuleNotFoundError:  # pragma: no cover - optional speedup.
    orjson = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "Dataset" / "render_ready_episodes"
DEFAULT_OUTPUT_ROOT = ROOT / "Dataset" / "render_ready_episodes_capture_filtered"
PVU_CATEGORIES = {"pedestrian", "vehicle", "uav"}
CAPTURE_VISIBILITY_PADDING_M = 60.0
CAPTURE_SYNC_ROI_MARGIN_M = 25.0
CAPTURE_SYNC_POLICY = "full_truth_passthrough_generator_authoritative_v1"
REGENERATED_FILES = {
    "truth_frames.jsonl",
    "trajectories.jsonl",
    "global_entity_roster.json",
    "episode_manifest.json",
    "scenario_plan.json",
}

if str(ROOT / "Dataset" / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "Dataset" / "tools"))

from sumo_ground_flow.truth_integration import VisibilityGeometry  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dumps_jsonl_bytes(payload: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def loads_jsonl_bytes(payload: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload.decode("utf-8-sig"))


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_manifest_path(value: Any, episode_dir: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise RuntimeError(f"{episode_dir.name}: manifest path is empty")
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [
        (ROOT / path).resolve(),
        (episode_dir / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def position_enu_from_entity(entity: dict[str, Any]) -> list[float] | None:
    pose = entity.get("truth_pose") or {}
    position = pose.get("position_enu_m") or pose.get("position_m") or entity.get("pos_enu")
    if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) < 2:
        return None
    try:
        return [
            float(position[0]),
            float(position[1]),
            float(position[2] if len(position) > 2 else 0.0),
        ]
    except (TypeError, ValueError):
        return None


def velocity_enu_from_entity(entity: dict[str, Any]) -> list[float]:
    pose = entity.get("truth_pose") or {}
    velocity = pose.get("velocity_enu_mps") or entity.get("vel_mps") or [0.0, 0.0, 0.0]
    values = list(velocity) if isinstance(velocity, Sequence) and not isinstance(velocity, (str, bytes)) else []
    return [
        float(values[0] if len(values) > 0 else 0.0),
        float(values[1] if len(values) > 1 else 0.0),
        float(values[2] if len(values) > 2 else 0.0),
    ]


def yaw_from_entity(entity: dict[str, Any]) -> float | None:
    pose = entity.get("truth_pose") or {}
    rotation = pose.get("rotation_deg") or {}
    if isinstance(rotation, dict):
        value = rotation.get("yaw_deg", rotation.get("yaw"))
        if value is not None:
            return float(value)
    value = entity.get("yaw_deg")
    return float(value) if value is not None else None


def event_contract(event_script: dict[str, Any]) -> dict[str, Any]:
    params = dict(event_script.get("parameters") or {})
    contract = dict(params.get("semantic_event_contract") or {})
    if not contract:
        raise RuntimeError("event_script.parameters.semantic_event_contract is missing")
    return contract


def capture_polygon_from_event_script(event_script: dict[str, Any]) -> list[list[float]]:
    contract = event_contract(event_script)
    boundary = dict(contract.get("capture_boundary") or {})
    polygon: list[list[float]] = []
    for point in boundary.get("polygon_enu_m") or []:
        if isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) >= 2:
            polygon.append([float(point[0]), float(point[1])])
    if not polygon:
        raise RuntimeError("capture boundary polygon_enu_m is missing")
    return polygon


def inspect_entity_from_roster(roster_entities: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for entity in roster_entities:
        if dict(entity.get("contract_inspect_uav") or {}):
            return entity
    for entity in roster_entities:
        role_values = {
            str(entity.get("role") or "").lower(),
            str(entity.get("semantic_role") or "").lower(),
            str(entity.get("uav_corridor_role") or "").lower(),
        }
        if "u_inspect" in role_values or "inspect_observer" in role_values:
            return entity
    raise RuntimeError("inspect UAV contract is missing from global_entity_roster")


def inspect_route_from_entity(entity: dict[str, Any]) -> list[list[float]]:
    inspect_contract = dict(entity.get("contract_inspect_uav") or {})
    for key in ("loop_route_enu_m", "repaired_route_enu_m", "planned_route_enu_m"):
        candidate = inspect_contract.get(key)
        if isinstance(candidate, list) and candidate:
            return [[float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else 0.0)] for point in candidate]
    candidate = entity.get("route_waypoints_enu_m")
    if isinstance(candidate, list) and candidate:
        return [[float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else 0.0)] for point in candidate]
    raise RuntimeError(f"{entity.get('entity_id')}: inspect route is missing")


def visibility_from_episode(source_episode_dir: Path, manifest: dict[str, Any], roster_entities: Sequence[dict[str, Any]]) -> VisibilityGeometry:
    event_script_path = resolve_manifest_path(manifest.get("source_event_script_path"), source_episode_dir)
    event_script = read_json(event_script_path)
    inspect_entity = inspect_entity_from_roster(roster_entities)
    inspect_contract = dict(inspect_entity.get("contract_inspect_uav") or {})
    route = inspect_route_from_entity(inspect_entity)
    source_contract = {
        "capture_boundary_polygon_enu_m": capture_polygon_from_event_script(event_script),
        "inspect_route_enu_m": route,
        "inspect_contract": inspect_contract,
    }
    visibility = VisibilityGeometry.from_contract(source_contract)
    return replace(
        visibility,
        padding_m=max(float(visibility.padding_m), CAPTURE_VISIBILITY_PADDING_M),
    )


def entity_category(entity: dict[str, Any]) -> str:
    return str(entity.get("entity_category") or entity.get("category") or entity.get("label_class") or "").strip().lower()


def trajectory_row_from_entity(frame: dict[str, Any], entity: dict[str, Any]) -> dict[str, Any]:
    position = position_enu_from_entity(entity) or [0.0, 0.0, 0.0]
    row: dict[str, Any] = {
        "tick": int(frame.get("tick") or 0),
        "frame_id": frame.get("frame_id"),
        "sim_time_s": float(frame.get("sim_time_s") or 0.0),
        "entity_id": entity.get("entity_id"),
        "label_class": entity.get("label_class"),
        "asset_id": entity.get("logical_asset_id") or entity.get("asset_id"),
        "entity_category": entity.get("entity_category"),
        "entity_kind": entity.get("entity_kind"),
        "entity_type": entity.get("entity_type"),
        "pos_enu": position,
        "vel_mps": velocity_enu_from_entity(entity),
        "yaw_deg": yaw_from_entity(entity),
        "state": entity.get("state"),
        "activity_type": (entity.get("annotations") or {}).get("activity_type"),
        "source": entity.get("source"),
        "category": entity_category(entity),
    }
    preserve_keys = (
        "role",
        "semantic_role",
        "background_role",
        "capture_boundary_id",
        "sumo_segment",
        "sumo_vehicle",
        "sumo_visibility",
        "uav_segment",
        "uav_global_flow",
        "uav_visibility",
        "mission_type",
        "task_id",
    )
    for key in preserve_keys:
        if key in entity:
            row[key] = copy.deepcopy(entity[key])
    return {key: value for key, value in row.items() if value is not None}


def update_frame_summaries(frame: dict[str, Any], kept_entities: Sequence[dict[str, Any]]) -> None:
    counts = Counter(entity_category(entity) for entity in kept_entities)
    frame["roster_summary"] = {
        "total": len(kept_entities),
        "by_category": dict(sorted(counts.items())),
    }
    kept_ids = {str(entity.get("entity_id") or "") for entity in kept_entities}
    motion_state = frame.get("entity_motion_state")
    if isinstance(motion_state, dict):
        frame["entity_motion_state"] = {
            key: value for key, value in motion_state.items() if str(key) in kept_ids
        }
    if isinstance(frame.get("sumo_semantics"), dict):
        frame["sumo_semantics"]["active_selected_vehicle_count"] = sum(
            1 for entity in kept_entities if entity_category(entity) == "vehicle" and entity.get("source") == "sumo_traci"
        )
    if isinstance(frame.get("uav_global_flow"), dict):
        frame["uav_global_flow"]["active_selected_uav_count"] = sum(
            1 for entity in kept_entities if entity_category(entity) == "uav" and entity.get("source") == "uav_global_flow"
        )


def capture_visibility_bbox_enu_m(visibility: VisibilityGeometry) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = [0.0, float(visibility.altitude_m)]
    for point in visibility.capture_polygon_enu_m:
        xs.append(float(point[0]))
        ys.append(float(point[1]))
    for point in visibility.inspect_route_enu_m:
        xs.append(float(point[0]))
        ys.append(float(point[1]))
        zs.append(float(point[2] if len(point) > 2 else visibility.altitude_m))
    if not xs or not ys:
        return []
    margin_m = max(
        CAPTURE_SYNC_ROI_MARGIN_M,
        float(visibility.padding_m),
        float(visibility.footprint_half_width_m),
        float(visibility.footprint_half_height_m),
    )
    return [
        round(min(xs) - margin_m, 3),
        round(min(ys) - margin_m, 3),
        round(max(xs) + margin_m, 3),
        round(max(ys) + margin_m, 3),
        round(max(zs) if zs else 0.0, 3),
    ]


def copy_static_episode_files(source_episode_dir: Path, output_episode_dir: Path) -> None:
    output_episode_dir.mkdir(parents=True, exist_ok=True)
    for item in source_episode_dir.iterdir():
        if item.name in REGENERATED_FILES:
            continue
        target = output_episode_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        elif item.is_file():
            shutil.copy2(item, target)


def update_artifact_paths(payload: dict[str, Any], output_episode_dir: Path) -> None:
    artifacts = {
        "scenario_plan": repo_relative(output_episode_dir / "scenario_plan.json"),
        "global_entity_roster": repo_relative(output_episode_dir / "global_entity_roster.json"),
        "truth_frames": repo_relative(output_episode_dir / "truth_frames.jsonl"),
        "event_trace": repo_relative(output_episode_dir / "event_trace.jsonl"),
        "trajectories": repo_relative(output_episode_dir / "trajectories.jsonl"),
        "weather_meta": repo_relative(output_episode_dir / "weather_meta.jsonl"),
        "dynamic_labels": repo_relative(output_episode_dir / "dynamic_labels.jsonl"),
        "episode_manifest": repo_relative(output_episode_dir / "episode_manifest.json"),
    }
    payload["artifacts"] = artifacts
    payload["canonical_artifacts"] = copy.deepcopy(artifacts)


def update_render_host_config(output_episode_dir: Path, manifest: dict[str, Any]) -> None:
    path = output_episode_dir / "render_host_config.json"
    if path.exists():
        config = read_json(path)
    else:
        template_path = ROOT / "Plugins" / "SumoImporter" / "Scripts" / "episode_render_host_config.json"
        config = read_json(template_path) if template_path.exists() else {}
    config["episode_dir"] = repo_relative(output_episode_dir)
    config["output_dir"] = f"F:/aw_cap/_direct_render_host_capture_filtered/{output_episode_dir.name}"
    config["map_id"] = str(manifest.get("map_id") or config.get("map_id") or "donghu_road_topo")
    config["truth_frame_coordinate_space"] = "map_enu"
    source_event_script_path = str(manifest.get("source_event_script_path") or "").strip()
    if source_event_script_path:
        config["event_script_path"] = source_event_script_path
    for section in ("vehicle_lane_offsets", "entity_overlap_filter", "pedestrian_roadside_projection"):
        payload = dict(config.get(section) or {})
        payload["enabled"] = False
        config[section] = payload
    road_topology = dict(config.get("road_topology_snap") or {})
    if road_topology:
        road_topology["preserve_truth_xy"] = True
        config["road_topology_snap"] = road_topology
    strategy = dict(config.get("batch_strategy") or {})
    scenario_plan_path = output_episode_dir / "scenario_plan.json"
    if scenario_plan_path.exists():
        scenario_plan = read_json(scenario_plan_path)
        site_contracts = dict((scenario_plan.get("compiled_plan_summary") or {}).get("site_contracts") or {})
        if site_contracts:
            strategy["sites"] = sorted(site_contracts)
    config["batch_strategy"] = strategy
    write_json(path, config)


def update_scenario_package(output_episode_dir: Path) -> None:
    path = output_episode_dir / "scenario_package.json"
    if not path.exists():
        return
    package = read_json(path)
    package["root_dir"] = repo_relative(output_episode_dir)
    package["truth_frames"] = repo_relative(output_episode_dir / "truth_frames.jsonl")
    package["weather_meta"] = repo_relative(output_episode_dir / "weather_meta.jsonl")
    package["scenario_plan"] = repo_relative(output_episode_dir / "scenario_plan.json")
    package["episode_manifest"] = repo_relative(output_episode_dir / "episode_manifest.json")
    write_json(path, package)


def update_manifest(
    manifest: dict[str, Any],
    *,
    output_episode_dir: Path,
    source_episode_dir: Path,
    frame_count: int,
    trajectory_count: int,
    roster_entities: Sequence[dict[str, Any]],
    visibility: VisibilityGeometry,
    filter_stats: dict[str, Any],
    roi_bbox_enu_m: list[float],
) -> dict[str, Any]:
    updated = copy.deepcopy(manifest)
    counts = dict(updated.get("record_counts") or {})
    counts["truth_frames"] = frame_count
    counts["trajectories"] = trajectory_count
    counts["global_entity_roster"] = len(roster_entities)
    updated["record_counts"] = counts
    updated["canonical_record_counts"] = copy.deepcopy(counts)
    category_counts = Counter(entity_category(entity) for entity in roster_entities)
    updated["node_counts"] = {
        "all_nodes": len(roster_entities),
        "dynamic_nodes": sum(category_counts.get(category, 0) for category in PVU_CATEGORIES),
        "static_nodes": len(roster_entities) - sum(category_counts.get(category, 0) for category in PVU_CATEGORIES),
    }
    generation = dict(updated.get("generation") or {})
    generation["truth_filter_source_root"] = repo_relative(source_episode_dir)
    generation["truth_filter_contract"] = CAPTURE_SYNC_POLICY
    generation["capture_truth_sync_policy"] = CAPTURE_SYNC_POLICY
    generation["trajectory_source"] = "source_trajectories_passthrough"
    generation["trajectory_contract"] = "full_truth_trajectory_passthrough_v1"
    updated["generation"] = generation
    capture_sync_payload = {
        "policy": CAPTURE_SYNC_POLICY,
        "filtering_enabled": False,
        "dynamic_pvu_filtering_enabled": False,
        "coordinate_policy": "copy_truth_pose_map_enu_without_transform",
        "visibility_policy": "metadata_only_generator_must_produce_visible_truth",
        "visibility_padding_policy": "expanded_for_editor_hook_uav_rgb_capture_margin_v1",
        "visibility_padding_m": CAPTURE_VISIBILITY_PADDING_M,
        "visibility_geometry": visibility.as_dict(),
        "formal_roi_bbox_enu_m": roi_bbox_enu_m,
        "stats": filter_stats,
    }
    updated["capture_visible_truth_filter"] = copy.deepcopy(capture_sync_payload)
    updated["capture_truth_sync"] = copy.deepcopy(capture_sync_payload)
    update_artifact_paths(updated, output_episode_dir)
    return updated


def update_scenario_plan(
    scenario_plan: dict[str, Any],
    *,
    roster_entities: Sequence[dict[str, Any]],
    visibility: VisibilityGeometry,
    filter_stats: dict[str, Any],
    roi_bbox_enu_m: list[float],
) -> dict[str, Any]:
    updated = copy.deepcopy(scenario_plan)
    updated["global_entity_roster"] = list(roster_entities)
    category_counts = Counter(entity_category(entity) for entity in roster_entities)
    summary = dict(updated.get("compiled_plan_summary") or {})
    summary["entity_counts_by_category"] = dict(sorted(category_counts.items()))
    for contract in (summary.get("site_contracts") or {}).values():
        if isinstance(contract, dict):
            contract["entity_count"] = len(roster_entities)
    for window in (summary.get("roi_windows") or {}).values():
        if isinstance(window, dict):
            window["bbox_enu_m"] = list(roi_bbox_enu_m)
            window["bbox_source"] = "capture_visibility_geometry_passthrough"
    capture_sync_payload = {
        "policy": CAPTURE_SYNC_POLICY,
        "filtering_enabled": False,
        "dynamic_pvu_filtering_enabled": False,
        "coordinate_policy": "copy_truth_pose_map_enu_without_transform",
        "visibility_policy": "metadata_only_generator_must_produce_visible_truth",
        "visibility_padding_policy": "expanded_for_editor_hook_uav_rgb_capture_margin_v1",
        "visibility_padding_m": CAPTURE_VISIBILITY_PADDING_M,
        "visibility_geometry": visibility.as_dict(),
        "formal_roi_bbox_enu_m": roi_bbox_enu_m,
        "stats": filter_stats,
    }
    summary["capture_visible_truth_filter"] = copy.deepcopy(capture_sync_payload)
    summary["capture_truth_sync"] = copy.deepcopy(capture_sync_payload)
    updated["compiled_plan_summary"] = summary
    updated["capture_visible_truth_filter"] = copy.deepcopy(summary["capture_visible_truth_filter"])
    updated["capture_truth_sync"] = copy.deepcopy(summary["capture_truth_sync"])
    nested_plan = updated.get("scenario_plan")
    if isinstance(nested_plan, dict):
        nested_plan["summary"] = copy.deepcopy(summary)
    return updated


def filter_episode(source_episode_dir: Path, output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    start = time.perf_counter()
    output_episode_dir = output_root / source_episode_dir.name
    if output_episode_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_episode_dir} exists; pass --overwrite")
        shutil.rmtree(output_episode_dir)
    copy_static_episode_files(source_episode_dir, output_episode_dir)

    manifest = read_json(source_episode_dir / "episode_manifest.json")
    roster_root = read_json(source_episode_dir / "global_entity_roster.json")
    roster_entities = list(roster_root.get("entities") or [])
    visibility = visibility_from_episode(source_episode_dir, manifest, roster_entities)

    input_truth_path = source_episode_dir / "truth_frames.jsonl"
    input_trajectory_path = source_episode_dir / "trajectories.jsonl"
    output_truth_path = output_episode_dir / "truth_frames.jsonl"
    output_trajectory_path = output_episode_dir / "trajectories.jsonl"
    frame_count = 0
    truth_entity_count = 0
    trajectory_count = 0
    dynamic_pvu_records = 0
    truth_entity_ids: set[str] = set()

    shutil.copy2(input_truth_path, output_truth_path)
    if input_trajectory_path.exists():
        shutil.copy2(input_trajectory_path, output_trajectory_path)
        with output_trajectory_path.open("rb") as handle:
            trajectory_count = sum(1 for line in handle if line.strip())
    else:
        with output_trajectory_path.open("wb") as traj_out, input_truth_path.open("rb") as source:
            for line_number, line in enumerate(source, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    frame = loads_jsonl_bytes(stripped)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{input_truth_path}:{line_number}: invalid JSONL row: {exc}") from exc
                for entity in frame.get("entities") or []:
                    if not isinstance(entity, dict) or entity_category(entity) not in PVU_CATEGORIES:
                        continue
                    traj_out.write(dumps_jsonl_bytes(trajectory_row_from_entity(frame, entity)))
                    trajectory_count += 1

    with input_truth_path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                frame = loads_jsonl_bytes(stripped)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{input_truth_path}:{line_number}: invalid JSONL row: {exc}") from exc
            entities = list(frame.get("entities") or [])
            truth_entity_count += len(entities)
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                category = entity_category(entity)
                entity_id = str(entity.get("entity_id") or "")
                if entity_id:
                    truth_entity_ids.add(entity_id)
                if category in PVU_CATEGORIES:
                    dynamic_pvu_records += 1
            frame_count += 1

    synced_roster = roster_entities
    synced_roster_root = copy.deepcopy(roster_root)
    synced_roster_root["entities"] = synced_roster
    filter_stats = {
        "input_truth_entities": truth_entity_count,
        "output_truth_entities": truth_entity_count,
        "removed_truth_entities": 0,
        "dynamic_pvu_records": dynamic_pvu_records,
        "trajectory_rows": trajectory_count,
        "truth_entity_id_count": len(truth_entity_ids),
        "roster_entities": len(synced_roster),
        "dynamic_filter_rule": "none_full_truth_records_preserved",
        "semantic_context_rule": "preserve_full_roster_without_visibility_filter",
        "generator_authority_rule": "validators_must_fail_generation_errors_instead_of_deleting_truth",
    }
    roi_bbox_enu_m = capture_visibility_bbox_enu_m(visibility)
    write_json(output_episode_dir / "global_entity_roster.json", synced_roster_root)
    write_json(
        output_episode_dir / "episode_manifest.json",
        update_manifest(
            manifest,
            output_episode_dir=output_episode_dir,
            source_episode_dir=source_episode_dir,
            frame_count=frame_count,
            trajectory_count=trajectory_count,
            roster_entities=synced_roster,
            visibility=visibility,
            filter_stats=filter_stats,
            roi_bbox_enu_m=roi_bbox_enu_m,
        ),
    )
    scenario_plan = read_json(source_episode_dir / "scenario_plan.json")
    write_json(
        output_episode_dir / "scenario_plan.json",
        update_scenario_plan(
            scenario_plan,
            roster_entities=synced_roster,
            visibility=visibility,
            filter_stats=filter_stats,
            roi_bbox_enu_m=roi_bbox_enu_m,
        ),
    )
    update_render_host_config(output_episode_dir, manifest)
    update_scenario_package(output_episode_dir)
    elapsed_s = time.perf_counter() - start
    return {
        "episode": source_episode_dir.name,
        "elapsed_s": round(elapsed_s, 3),
        "input_truth_mb": round(input_truth_path.stat().st_size / 1024 / 1024, 3),
        "output_truth_mb": round(output_truth_path.stat().st_size / 1024 / 1024, 3),
        "trajectory_mb": round(output_trajectory_path.stat().st_size / 1024 / 1024, 3),
        **filter_stats,
    }


def parse_episode_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for item in value.split(","):
            name = item.strip()
            if name:
                names.append(name)
    return names


def select_episode_dirs(input_root: Path, names: list[str]) -> list[Path]:
    all_dirs = sorted(path for path in input_root.iterdir() if path.is_dir() and (path / "truth_frames.jsonl").exists())
    if not names:
        return all_dirs
    by_name = {path.name: path for path in all_dirs}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"Unknown episode(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync full render-ready truth into the formal UE replay root without filtering.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    episode_dirs = select_episode_dirs(input_root, parse_episode_names(args.episode))
    if args.limit and args.limit > 0:
        episode_dirs = episode_dirs[: args.limit]
    if not episode_dirs:
        raise SystemExit("No render-ready episodes selected")

    output_root.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers or 1))
    if workers == 1 or len(episode_dirs) <= 1:
        results = [filter_episode(path, output_root, overwrite=bool(args.overwrite)) for path in episode_dirs]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_episode = {
                executor.submit(filter_episode, path, output_root, overwrite=bool(args.overwrite)): path
                for path in episode_dirs
            }
            for future in as_completed(future_to_episode):
                results.append(future.result())
        results.sort(key=lambda item: str(item["episode"]))
    summary = {
        "ok": True,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "episode_count": len(results),
        "total_elapsed_s": round(sum(float(item["elapsed_s"]) for item in results), 3),
        "total_input_truth_mb": round(sum(float(item["input_truth_mb"]) for item in results), 3),
        "total_output_truth_mb": round(sum(float(item["output_truth_mb"]) for item in results), 3),
        "total_trajectory_mb": round(sum(float(item["trajectory_mb"]) for item in results), 3),
        "total_removed_truth_entities": sum(int(item["removed_truth_entities"]) for item in results),
        "results": results,
    }
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))
    if len(results) <= 10:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
