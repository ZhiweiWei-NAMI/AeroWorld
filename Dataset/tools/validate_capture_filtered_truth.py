#!/usr/bin/env python3
"""Validate the formal UE replay truth synchronization package.

``Dataset/render_ready_episodes_capture_filtered`` is kept as the formal capture
input path.  It is allowed to remove positioned entities outside the capture
boundary expanded by 60m, then trajectories must be regenerated from the
filtered truth frames. SUMO-authored event-controlled and source-required
vehicles preserve their complete source lifecycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "aw_data" / "render_ready_episodes_capture_filtered"
PVU_CATEGORIES = {"pedestrian", "vehicle", "uav"}
DISABLED_RENDER_FLAGS = (
    ("vehicle_lane_offsets", "enabled"),
    ("entity_overlap_filter", "enabled"),
    ("pedestrian_roadside_projection", "enabled"),
)
ALLOWED_DYNAMIC_FILTER_RULES = {
    "drop_positioned_entities_outside_capture_boundary_plus_60m",
    "drop_positioned_entities_outside_capture_boundary_plus_60m_and_jittery_sumo_background_vehicles",
    "drop_positioned_entities_outside_capture_boundary_plus_60m_and_jittery_sumo_background_vehicles_and_road_semantic_closed_lanes",
    "drop_positioned_entities_outside_capture_boundary_plus_60m_and_jittery_sumo_background_vehicles_and_road_semantic_closed_lanes_preserve_required_sumo_lifecycle",
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "Dataset" / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "Dataset" / "tools"))

from Dataset.semantic_truth.provenance import digest_file  # noqa: E402
from filter_render_ready_truth_for_capture import (  # noqa: E402
    DEFAULT_LOGICAL_ASSET_YAW_OFFSETS_DEG,
    REQUIRED_SUMO_LIFECYCLE_PRESERVATION_POLICY,
    entity_category,
    entity_within_capture_runtime,
    is_capture_required_sumo_vehicle,
    loads_jsonl_bytes,
    read_json,
    visibility_from_episode,
)
from semantic_event_contract import get_contract  # noqa: E402


def parse_episode_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for item in value.split(","):
            name = item.strip()
            if name:
                names.append(name)
    return names


def selected_episode_dirs(input_root: Path, names: list[str], limit: int) -> list[Path]:
    all_dirs = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and (path / "truth_frames.jsonl").exists()
    )
    if names:
        wanted = set(names)
        by_name = {path.name: path for path in all_dirs}
        missing = sorted(name for name in wanted if name not in by_name)
        if missing:
            raise SystemExit(f"Missing formal capture episodes: {missing}")
        all_dirs = [by_name[name] for name in names]
    if limit > 0:
        all_dirs = all_dirs[:limit]
    return all_dirs


def source_episode_dir(formal_dir: Path, manifest: dict[str, Any]) -> Path:
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("episode manifest requires a generation object")
    raw = str(generation.get("truth_filter_source_root") or "").strip()
    if not raw:
        raise ValueError(
            "episode manifest requires generation.truth_filter_source_root"
        )
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if resolved.name != formal_dir.name:
        raise ValueError(
            "generation.truth_filter_source_root must name the validated episode"
        )
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"generation.truth_filter_source_root does not exist: {resolved}"
        )
    return resolved


def check_render_config(formal_dir: Path, errors: list[str]) -> None:
    config_path = formal_dir / "render_host_config.json"
    if not config_path.exists():
        errors.append(f"{formal_dir.name}: missing render_host_config.json")
        return
    config = read_json(config_path)
    for section, key in DISABLED_RENDER_FLAGS:
        payload = dict(config.get(section) or {})
        if payload.get(key) is not False:
            errors.append(
                f"{formal_dir.name}: render_host_config must set {section}.{key}=false"
            )
    road_topology = dict(config.get("road_topology_snap") or {})
    if (
        road_topology.get("enabled") is True
        and road_topology.get("preserve_truth_xy") is not True
    ):
        errors.append(
            f"{formal_dir.name}: road_topology_snap must preserve truth XY when enabled"
        )
    rotation_offsets = dict(config.get("entity_rotation_offsets_deg") or {})
    logical_asset_offsets = dict(rotation_offsets.get("logical_asset_yaw_deg") or {})
    for logical_asset_id, expected_yaw in DEFAULT_LOGICAL_ASSET_YAW_OFFSETS_DEG.items():
        if float(logical_asset_offsets.get(logical_asset_id, 999.0)) != float(
            expected_yaw
        ):
            errors.append(
                f"{formal_dir.name}: render_host_config missing logical_asset_yaw_deg "
                f"{logical_asset_id}={expected_yaw}"
            )


def sha256_file(path: Path) -> str:
    return digest_file(path).removeprefix("sha256:")


def truth_stats(path: Path) -> dict[str, Any]:
    frame_count = 0
    entity_records = 0
    dynamic_pvu_records = 0
    dynamic_pvu_ids: set[str] = set()
    dynamic_id_sets: list[set[str]] = []
    vehicle_frame_count = 0
    max_vehicle_count = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                frame = loads_jsonl_bytes(stripped)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSONL row: {exc}"
                ) from exc
            frame_count += 1
            current_dynamic_ids: set[str] = set()
            entities = frame.get("entities") or []
            entity_records += len(entities) if isinstance(entities, list) else 0
            current_vehicle_count = 0
            for entity in entities if isinstance(entities, list) else []:
                if not isinstance(entity, dict):
                    continue
                category = entity_category(entity)
                if category not in PVU_CATEGORIES:
                    continue
                if category == "vehicle":
                    current_vehicle_count += 1
                entity_id = str(entity.get("entity_id") or "")
                dynamic_pvu_records += 1
                if entity_id:
                    dynamic_pvu_ids.add(entity_id)
                    current_dynamic_ids.add(entity_id)
            if current_vehicle_count > 0:
                vehicle_frame_count += 1
                max_vehicle_count = max(max_vehicle_count, current_vehicle_count)
            dynamic_id_sets.append(current_dynamic_ids)
    return {
        "frame_count": frame_count,
        "entity_records": entity_records,
        "dynamic_pvu_records": dynamic_pvu_records,
        "dynamic_pvu_ids": dynamic_pvu_ids,
        "dynamic_id_sets": dynamic_id_sets,
        "vehicle_frame_count": vehicle_frame_count,
        "max_vehicle_count": max_vehicle_count,
    }


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def check_no_filter_policy(
    formal_dir: Path, manifest: dict[str, Any], errors: list[str]
) -> None:
    payload = dict(
        manifest.get("capture_truth_sync")
        or manifest.get("capture_visible_truth_filter")
        or {}
    )
    if not payload:
        errors.append(
            f"{formal_dir.name}: missing capture_truth_sync/capture_visible_truth_filter metadata"
        )
        return
    if payload.get("filtering_enabled") is not True:
        errors.append(
            f"{formal_dir.name}: formal capture sync must set filtering_enabled=true"
        )
    if payload.get("dynamic_pvu_filtering_enabled") is not True:
        errors.append(
            f"{formal_dir.name}: formal capture sync must set dynamic_pvu_filtering_enabled=true"
        )
    stats = dict(payload.get("stats") or {})
    if str(stats.get("dynamic_filter_rule") or "") not in ALLOWED_DYNAMIC_FILTER_RULES:
        errors.append(
            f"{formal_dir.name}: unexpected dynamic_filter_rule={stats.get('dynamic_filter_rule')!r}"
        )


def has_required_sumo_lifecycle_retention(entity: dict[str, Any]) -> bool:
    if not is_capture_required_sumo_vehicle(entity):
        return False
    runtime_visibility = dict(entity.get("runtime_visibility") or {})
    sumo_vehicle = dict(entity.get("sumo_vehicle") or {})
    return (
        runtime_visibility.get("retained_outside_runtime_boundary") is True
        and str(runtime_visibility.get("retention_policy") or "")
        == REQUIRED_SUMO_LIFECYCLE_PRESERVATION_POLICY
    ) or str(
        sumo_vehicle.get("required_lifecycle_preservation_policy") or ""
    ) == REQUIRED_SUMO_LIFECYCLE_PRESERVATION_POLICY


def check_runtime_crop(
    formal_dir: Path,
    source_dir: Path,
    manifest: dict[str, Any],
    source_entities: list[dict[str, Any]],
    errors: list[str],
) -> None:
    visibility = visibility_from_episode(source_dir, manifest, source_entities)
    violations: list[str] = []
    truth_path = formal_dir / "truth_frames.jsonl"
    with truth_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            frame = loads_jsonl_bytes(stripped)
            tick = int(frame.get("tick") or 0)
            for entity in frame.get("entities") or []:
                if not isinstance(entity, dict):
                    continue
                if entity_within_capture_runtime(entity, visibility):
                    continue
                if has_required_sumo_lifecycle_retention(entity):
                    continue
                violations.append(f"tick={tick} entity={entity.get('entity_id')}")
                if len(violations) >= 10:
                    break
            if len(violations) >= 10:
                break
    if violations:
        errors.append(
            f"{formal_dir.name}: entities outside capture boundary +60m remain in formal truth: {violations}"
        )


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _read_sumo_source_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    sumo_traffic = dict(manifest.get("sumo_traffic") or {})
    source = dict(sumo_traffic.get("source") or {})
    raw_path = str(source.get("manifest") or "").strip()
    if not raw_path:
        return {}
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {}
    try:
        return dict(read_json(path) or {})
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _sumo_only_vehicle_authority(manifest: dict[str, Any]) -> bool:
    authority = dict(manifest.get("source_vehicle_authority") or {})
    if authority.get("enabled") is not True:
        return False
    authority_text = " ".join(
        str(authority.get(key) or "")
        for key in ("policy", "authority", "vehicle_source_policy")
    ).lower()
    return "sumo" in authority_text


def _is_background_only_sumo_episode(
    manifest: dict[str, Any],
    sumo_manifest: dict[str, Any],
    planned_semantic_vehicle_count: int,
) -> bool:
    sumo_traffic = dict(manifest.get("sumo_traffic") or {})
    plans = (
        dict(sumo_traffic.get("explicit_vehicle_plan") or {}),
        dict(sumo_manifest.get("explicit_vehicle_plan") or {}),
    )
    for plan in plans:
        traffic_case = dict(plan.get("traffic_case") or {})
        if str(traffic_case.get("case_type") or "") == "background_only":
            return traffic_case.get("has_semantic_vehicle") is not True
    if planned_semantic_vehicle_count > 0:
        return False
    source_authority = dict(manifest.get("source_vehicle_authority") or {})
    plan = dict(source_authority.get("explicit_vehicle_plan") or {})
    audit = dict(plan.get("selection_audit") or {})
    return (
        _safe_int(audit.get("planned_semantic_vehicle_count")) <= 0
        and _safe_int(plan.get("vehicle_count")) > 0
    )


def _capture_road_geometry_absent(
    manifest: dict[str, Any],
    sumo_manifest: dict[str, Any],
) -> bool:
    roi_lane_coverage = dict(sumo_manifest.get("roi_lane_coverage") or {})
    if roi_lane_coverage:
        return (
            _safe_int(roi_lane_coverage.get("target_edge_count")) <= 0
            and _safe_int(roi_lane_coverage.get("target_anchor_count")) <= 0
        )

    sumo_traffic = dict(manifest.get("sumo_traffic") or {})
    selection = dict(sumo_traffic.get("selection") or {})
    return (
        _safe_int(selection.get("min_visible_vehicle_target")) <= 0
        and _safe_int(selection.get("max_visible_vehicle_target")) <= 0
        and selection.get("expanded_to_nearest") is not True
    )


def _vehicle_context_geometry_exempted(
    manifest: dict[str, Any],
    formal_stats: dict[str, Any],
    planned_semantic_vehicle_count: int,
    planned_source_required_vehicle_count: int,
    selected_count: int,
) -> tuple[bool, str]:
    if selected_count > 0 or _safe_int(formal_stats.get("vehicle_frame_count")) > 0:
        return False, ""
    if not _sumo_only_vehicle_authority(manifest):
        return False, ""
    if planned_source_required_vehicle_count > 0:
        return False, ""
    sumo_manifest = _read_sumo_source_manifest(manifest)
    if not _is_background_only_sumo_episode(
        manifest, sumo_manifest, planned_semantic_vehicle_count
    ):
        return False, ""
    if not _capture_road_geometry_absent(manifest, sumo_manifest):
        return False, ""
    roi_lane_coverage = dict(sumo_manifest.get("roi_lane_coverage") or {})
    if roi_lane_coverage:
        reason = (
            f"roi_lane_coverage target_edge_count={_safe_int(roi_lane_coverage.get('target_edge_count'))}, "
            f"target_anchor_count={_safe_int(roi_lane_coverage.get('target_anchor_count'))}"
        )
    else:
        reason = "SUMO selection declares zero visible vehicle target and no nearest expansion"
    return True, reason


def check_vehicle_roi_contract(
    formal_dir: Path,
    manifest: dict[str, Any],
    formal_stats: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    scenario_id = str(manifest.get("scenario_id") or formal_dir.name.split("__", 1)[0])
    contract = get_contract(scenario_id)
    sumo_traffic = dict(manifest.get("sumo_traffic") or {})
    audit = dict(sumo_traffic.get("explicit_selection_audit") or {})
    planned_semantic_vehicle_count = _safe_int(
        audit.get("planned_semantic_vehicle_count")
    )
    planned_source_required_vehicle_count = _safe_int(
        audit.get("planned_source_presence_required_vehicle_count")
    )
    planned_required_vehicle_count = _safe_int(
        audit.get("planned_capture_required_vehicle_count")
    )
    selection = dict(sumo_traffic.get("selection") or {})
    selected_count = _safe_int(selection.get("selected_count"))
    geometry_exempted, geometry_reason = _vehicle_context_geometry_exempted(
        manifest,
        formal_stats,
        planned_semantic_vehicle_count,
        planned_source_required_vehicle_count,
        selected_count,
    )
    if planned_required_vehicle_count <= 0:
        if (
            int(contract.vehicle) > 0
            and int(formal_stats.get("vehicle_frame_count") or 0) <= 0
        ):
            suffix = f"; {geometry_reason}" if geometry_exempted else ""
            warnings.append(
                f"{formal_dir.name}: source contract declares vehicle context ({contract.vehicle}), "
                "but no source-required or event-controlled SUMO vehicle is planned"
                f"{suffix}"
            )
        return
    if geometry_exempted:
        warnings.append(
            f"{formal_dir.name}: required SUMO vehicle plan requires {planned_required_vehicle_count} vehicle(s), "
            "but background-only SUMO vehicle authority selected no ROI vehicles because capture-road geometry is absent; "
            f"downgraded to warning ({geometry_reason})"
        )
        return
    if selected_count <= 0:
        errors.append(
            f"{formal_dir.name}: required SUMO vehicle plan requires {planned_required_vehicle_count} vehicle(s), "
            "but runtime ROI selection selected_count=0"
        )
    if int(formal_stats.get("vehicle_frame_count") or 0) <= 0:
        errors.append(
            f"{formal_dir.name}: required SUMO vehicle plan requires {planned_required_vehicle_count} vehicle(s), "
            "but truth has zero frames with visible vehicle entities"
        )


def validate_episode(formal_dir: Path, max_errors: int) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required = (
        "episode_manifest.json",
        "global_entity_roster.json",
        "truth_frames.jsonl",
        "trajectories.jsonl",
        "scenario_plan.json",
    )
    for name in required:
        if not (formal_dir / name).exists():
            errors.append(f"{formal_dir.name}: missing {name}")
    if errors:
        return {
            "episode": formal_dir.name,
            "ok": False,
            "errors": errors,
            "warnings": warnings,
        }

    manifest = read_json(formal_dir / "episode_manifest.json")
    try:
        source_dir = source_episode_dir(formal_dir, manifest)
    except (FileNotFoundError, ValueError) as error:
        errors.append(f"{formal_dir.name}: {error}")
        return {
            "episode": formal_dir.name,
            "ok": False,
            "errors": errors,
            "warnings": warnings,
        }
    for name in required:
        if not (source_dir / name).exists():
            errors.append(f"{formal_dir.name}: source missing {name}: {source_dir}")
    if errors:
        return {
            "episode": formal_dir.name,
            "ok": False,
            "errors": errors[:max_errors],
            "warnings": warnings,
        }

    check_no_filter_policy(formal_dir, manifest, errors)
    check_render_config(formal_dir, errors)

    source_roster = read_json(source_dir / "global_entity_roster.json")
    formal_roster = read_json(formal_dir / "global_entity_roster.json")
    source_entities = list(source_roster.get("entities") or [])
    formal_entities = list(formal_roster.get("entities") or [])
    source_ids = [
        str(entity.get("entity_id") or "")
        for entity in source_entities
        if isinstance(entity, dict)
    ]
    formal_ids = [
        str(entity.get("entity_id") or "")
        for entity in formal_entities
        if isinstance(entity, dict)
    ]
    extra_roster_ids = sorted(set(formal_ids) - set(source_ids))
    if extra_roster_ids:
        errors.append(
            f"{formal_dir.name}: global_entity_roster has ids absent from source: {extra_roster_ids[:10]}"
        )

    source_truth_path = source_dir / "truth_frames.jsonl"
    formal_truth_path = formal_dir / "truth_frames.jsonl"
    formal_traj_path = formal_dir / "trajectories.jsonl"
    formal_truth_sha = sha256_file(formal_truth_path)
    formal_traj_sha = sha256_file(formal_traj_path)

    source_stats = truth_stats(source_truth_path)
    formal_stats = truth_stats(formal_truth_path)
    check_vehicle_roi_contract(formal_dir, manifest, formal_stats, errors, warnings)
    check_runtime_crop(formal_dir, source_dir, manifest, source_entities, errors)
    if source_stats["frame_count"] != formal_stats["frame_count"]:
        errors.append(
            f"{formal_dir.name}: frame count changed source={source_stats['frame_count']} formal={formal_stats['frame_count']}"
        )
    if formal_stats["entity_records"] > source_stats["entity_records"]:
        errors.append(
            f"{formal_dir.name}: formal truth has more entity records than source "
            f"source={source_stats['entity_records']} formal={formal_stats['entity_records']}"
        )
    if formal_stats["dynamic_pvu_records"] > source_stats["dynamic_pvu_records"]:
        errors.append(
            f"{formal_dir.name}: formal truth has more dynamic P/V/U records than source "
            f"source={source_stats['dynamic_pvu_records']} formal={formal_stats['dynamic_pvu_records']}"
        )
    pvu_extra = sorted(
        formal_stats["dynamic_pvu_ids"] - source_stats["dynamic_pvu_ids"]
    )
    if pvu_extra:
        errors.append(
            f"{formal_dir.name}: formal dynamic P/V/U ids absent from source: {pvu_extra[:20]}"
        )
    formal_traj_rows = count_jsonl_rows(formal_traj_path)
    if formal_traj_rows != int(formal_stats["entity_records"]):
        errors.append(
            f"{formal_dir.name}: trajectories.jsonl must be regenerated from filtered truth "
            f"formal_rows={formal_traj_rows} truth_entities={formal_stats['entity_records']}"
        )
    if int(formal_stats["dynamic_pvu_records"]) <= 0:
        warnings.append(f"{formal_dir.name}: no dynamic P/V/U records in formal truth")

    return {
        "episode": formal_dir.name,
        "ok": not errors,
        "errors": errors[:max_errors],
        "warnings": warnings,
        "frame_count": int(formal_stats["frame_count"]),
        "entity_records": int(formal_stats["entity_records"]),
        "dynamic_pvu_records": int(formal_stats["dynamic_pvu_records"]),
        "dynamic_pvu_ids": len(formal_stats["dynamic_pvu_ids"]),
        "source_episode_dir": str(source_dir),
        "truth_sha256": formal_truth_sha,
        "trajectories_sha256": formal_traj_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate formal capture truth sync packages."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-errors", type=int, default=50)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    episodes = selected_episode_dirs(
        input_root, parse_episode_names(args.episode), int(args.limit or 0)
    )
    if not episodes:
        raise SystemExit("No formal capture episodes selected")

    results = [
        validate_episode(path, max(1, int(args.max_errors or 50))) for path in episodes
    ]
    failed = [result for result in results if not result["ok"]]
    summary = {
        "ok": not failed,
        "input_root": str(input_root),
        "episode_count": len(results),
        "failed_episode_count": len(failed),
        "dynamic_pvu_records": sum(
            int(result.get("dynamic_pvu_records") or 0) for result in results
        ),
        "dynamic_pvu_ids": sum(
            int(result.get("dynamic_pvu_ids") or 0) for result in results
        ),
        "entity_records": sum(
            int(result.get("entity_records") or 0) for result in results
        ),
        "failed_episodes": failed[:20],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
