#!/usr/bin/env python3
"""Validate capture-filtered truth before UE replay.

The full render-ready truth remains the semantic source of truth.  This
validator checks the smaller UE replay view: every dynamic pedestrian,
vehicle, and UAV record that remains in the filtered truth must be observable
in that episode's capture visibility geometry.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "Dataset" / "render_ready_episodes_capture_filtered"
DEFAULT_SOURCE_ROOT = ROOT / "Dataset" / "render_ready_episodes"
PVU_CATEGORIES = {"pedestrian", "vehicle", "uav"}
DISABLED_RENDER_FLAGS = (
    ("vehicle_lane_offsets", "enabled"),
    ("entity_overlap_filter", "enabled"),
    ("pedestrian_roadside_projection", "enabled"),
)

if str(ROOT / "Dataset" / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "Dataset" / "tools"))

from filter_render_ready_truth_for_capture import (  # noqa: E402
    entity_category,
    entity_initial_or_truth_position,
    is_filterable_global_infrastructure,
    loads_jsonl_bytes,
    position_enu_from_entity,
    read_json,
    visibility_segment_distance_m,
    visibility_segment_is_observable,
    visibility_from_episode,
)


def parse_episode_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for item in value.split(","):
            name = item.strip()
            if name:
                names.append(name)
    return names


def selected_episode_dirs(input_root: Path, names: list[str], limit: int) -> list[Path]:
    all_dirs = sorted(path for path in input_root.iterdir() if path.is_dir() and (path / "truth_frames.jsonl").exists())
    if names:
        wanted = set(names)
        by_name = {path.name: path for path in all_dirs}
        missing = sorted(name for name in wanted if name not in by_name)
        if missing:
            raise SystemExit(f"Missing capture-filtered episodes: {missing}")
        all_dirs = [by_name[name] for name in names]
    if limit > 0:
        all_dirs = all_dirs[:limit]
    return all_dirs


def source_episode_dir(filtered_dir: Path, source_root: Path, manifest: dict[str, Any]) -> Path:
    generation = dict(manifest.get("generation") or {})
    raw = str(generation.get("truth_filter_source_root") or "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.exists():
            return candidate.resolve()
    return (source_root / filtered_dir.name).resolve()


def check_render_config(filtered_dir: Path, errors: list[str]) -> None:
    config_path = filtered_dir / "render_host_config.json"
    if not config_path.exists():
        errors.append(f"{filtered_dir.name}: missing render_host_config.json")
        return
    config = read_json(config_path)
    for section, key in DISABLED_RENDER_FLAGS:
        payload = dict(config.get(section) or {})
        if payload.get(key) is not False:
            errors.append(f"{filtered_dir.name}: render_host_config must set {section}.{key}=false")
    road_topology = dict(config.get("road_topology_snap") or {})
    if road_topology.get("enabled") is True and road_topology.get("preserve_truth_xy") is not True:
        errors.append(f"{filtered_dir.name}: road_topology_snap must preserve truth XY when enabled")


def dynamic_segment_ok(
    position_by_frame: dict[int, dict[str, list[float]]],
    frame_indexes: list[int],
    current_offset: int,
    entity_id: str,
    visibility: Any,
) -> bool:
    current_index = frame_indexes[current_offset]
    current = position_by_frame[current_index].get(entity_id)
    if current is None:
        return False
    if visibility.is_observable(current):
        return True
    for neighbor_offset in (current_offset - 1, current_offset + 1):
        if neighbor_offset < 0 or neighbor_offset >= len(frame_indexes):
            continue
        neighbor = position_by_frame[frame_indexes[neighbor_offset]].get(entity_id)
        if neighbor is not None and visibility_segment_is_observable(current, neighbor, visibility):
            return True
    return False


def dynamic_segment_distance(
    position_by_frame: dict[int, dict[str, list[float]]],
    frame_indexes: list[int],
    current_offset: int,
    entity_id: str,
    visibility: Any,
) -> float:
    current_index = frame_indexes[current_offset]
    current = position_by_frame[current_index].get(entity_id)
    if current is None:
        return float("inf")
    distances = [visibility.observation_distance_m(current)]
    for neighbor_offset in (current_offset - 1, current_offset + 1):
        if neighbor_offset < 0 or neighbor_offset >= len(frame_indexes):
            continue
        neighbor = position_by_frame[frame_indexes[neighbor_offset]].get(entity_id)
        if neighbor is not None:
            distances.append(visibility_segment_distance_m(current, neighbor, visibility))
    return min(distances)


def validate_episode(filtered_dir: Path, source_root: Path, max_errors: int) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = filtered_dir / "episode_manifest.json"
    roster_path = filtered_dir / "global_entity_roster.json"
    truth_path = filtered_dir / "truth_frames.jsonl"
    if not manifest_path.exists():
        errors.append(f"{filtered_dir.name}: missing episode_manifest.json")
        return {"episode": filtered_dir.name, "ok": False, "errors": errors, "warnings": warnings}
    if not roster_path.exists():
        errors.append(f"{filtered_dir.name}: missing global_entity_roster.json")
        return {"episode": filtered_dir.name, "ok": False, "errors": errors, "warnings": warnings}

    manifest = read_json(manifest_path)
    source_dir = source_episode_dir(filtered_dir, source_root, manifest)
    if not source_dir.exists():
        errors.append(f"{filtered_dir.name}: source render-ready episode not found: {source_dir}")
        return {"episode": filtered_dir.name, "ok": False, "errors": errors, "warnings": warnings}
    source_manifest = read_json(source_dir / "episode_manifest.json")
    source_roster_root = read_json(source_dir / "global_entity_roster.json")
    source_roster_entities = list(source_roster_root.get("entities") or [])
    visibility = visibility_from_episode(source_dir, source_manifest, source_roster_entities)

    filter_payload = dict(manifest.get("capture_visible_truth_filter") or {})
    if not filter_payload:
        errors.append(f"{filtered_dir.name}: missing episode_manifest.capture_visible_truth_filter")
    else:
        visibility_payload = dict(filter_payload.get("visibility_geometry") or {})
        padding_m = visibility_payload.get("padding_m", filter_payload.get("visibility_padding_m"))
        if padding_m is not None:
            try:
                visibility = replace(visibility, padding_m=max(float(visibility.padding_m), float(padding_m)))
            except (TypeError, ValueError):
                errors.append(f"{filtered_dir.name}: invalid capture visibility padding_m={padding_m!r}")

    check_render_config(filtered_dir, errors)

    roster = read_json(roster_path)
    roster_entities = list(roster.get("entities") or [])
    roster_dynamic_ids = {
        str(entity.get("entity_id") or "")
        for entity in roster_entities
        if entity_category(entity) in PVU_CATEGORIES and str(entity.get("entity_id") or "")
    }
    roster_global_infra = [
        entity for entity in roster_entities if isinstance(entity, dict) and is_filterable_global_infrastructure(entity)
    ]
    for entity in roster_global_infra:
        position = entity_initial_or_truth_position(entity)
        if position is None:
            errors.append(f"{filtered_dir.name}: global infrastructure {entity.get('entity_id')} missing position")
            continue
        if not visibility.is_observable(position):
            distance = visibility.observation_distance_m(position)
            errors.append(
                f"{filtered_dir.name}: global infrastructure {entity.get('entity_id')} is outside capture visibility "
                f"distance={distance:.3f}m padding={visibility.padding_m:.3f}m"
            )

    frames: list[dict[str, Any]] = []
    with truth_path.open("rb") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            frames.append(loads_jsonl_bytes(stripped))
    frame_indexes = list(range(len(frames)))
    position_by_frame: dict[int, dict[str, list[float]]] = {}
    for frame_index, frame in enumerate(frames):
        positions: dict[str, list[float]] = {}
        for entity in frame.get("entities") or []:
            if not isinstance(entity, dict) or entity_category(entity) not in PVU_CATEGORIES:
                continue
            entity_id = str(entity.get("entity_id") or "")
            position = position_enu_from_entity(entity)
            if entity_id and position is not None:
                positions[entity_id] = position
        position_by_frame[frame_index] = positions

    frame_count = 0
    pvu_records = 0
    pvu_ids: set[str] = set()
    global_infra_records = 0
    for frame_offset, frame in enumerate(frames):
        frame_count += 1
        tick = int(frame.get("tick") or 0)
        for entity in frame.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            category = entity_category(entity)
            entity_id = str(entity.get("entity_id") or "")
            if category in PVU_CATEGORIES:
                position = position_enu_from_entity(entity)
                if position is None:
                    errors.append(f"{filtered_dir.name}: tick {tick} dynamic entity {entity_id} missing position")
                elif entity_id and not dynamic_segment_ok(
                    position_by_frame,
                    frame_indexes,
                    frame_offset,
                    entity_id,
                    visibility,
                ):
                    distance = dynamic_segment_distance(
                        position_by_frame,
                        frame_indexes,
                        frame_offset,
                        entity_id,
                        visibility,
                    )
                    errors.append(
                        f"{filtered_dir.name}: tick {tick} dynamic {category} {entity_id} has no point/adjacent segment in capture visibility "
                        f"distance={distance:.3f}m padding={visibility.padding_m:.3f}m"
                    )
                else:
                    pvu_records += 1
                    if entity_id:
                        pvu_ids.add(entity_id)
            elif is_filterable_global_infrastructure(entity):
                global_infra_records += 1
                position = entity_initial_or_truth_position(entity)
                if position is None or not visibility.is_observable(position):
                    distance = visibility.observation_distance_m(position) if position is not None else float("inf")
                    errors.append(
                        f"{filtered_dir.name}: tick {tick} global infrastructure {entity_id} outside capture visibility "
                        f"distance={distance:.3f}m padding={visibility.padding_m:.3f}m"
                    )
            if len(errors) >= max_errors:
                break
        if len(errors) >= max_errors:
            break

    missing_from_truth = sorted(roster_dynamic_ids - pvu_ids)
    if missing_from_truth:
        errors.append(
            f"{filtered_dir.name}: dynamic roster ids missing from filtered truth records: {missing_from_truth[:10]}"
        )
    if frame_count <= 0:
        errors.append(f"{filtered_dir.name}: empty truth_frames.jsonl")
    if pvu_records <= 0:
        warnings.append(f"{filtered_dir.name}: no dynamic P/V/U records remain after filtering")

    return {
        "episode": filtered_dir.name,
        "ok": not errors,
        "errors": errors[:max_errors],
        "warnings": warnings,
        "frame_count": frame_count,
        "dynamic_pvu_records": pvu_records,
        "dynamic_pvu_ids": len(pvu_ids),
        "global_infrastructure_records": global_infra_records,
        "source_episode_dir": str(source_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate capture-filtered UE replay truth.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-errors", type=int, default=50)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    source_root = args.source_root.resolve()
    episodes = selected_episode_dirs(input_root, parse_episode_names(args.episode), int(args.limit or 0))
    if not episodes:
        raise SystemExit("No capture-filtered episodes selected")

    results = [validate_episode(path, source_root, max(1, int(args.max_errors or 50))) for path in episodes]
    failed = [result for result in results if not result["ok"]]
    summary = {
        "ok": not failed,
        "input_root": str(input_root),
        "source_root": str(source_root),
        "episode_count": len(results),
        "failed_episode_count": len(failed),
        "dynamic_pvu_records": sum(int(result.get("dynamic_pvu_records") or 0) for result in results),
        "dynamic_pvu_ids": sum(int(result.get("dynamic_pvu_ids") or 0) for result in results),
        "global_infrastructure_records": sum(int(result.get("global_infrastructure_records") or 0) for result in results),
        "failed_episodes": failed[:20],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
