"""Lightweight integration checks for render-ready truth frames.

This validator intentionally does not repeat the heavy generation-time checks
for SUMO geometry, UAV obstacle clearance, or sensor coverage.  It verifies
that each render-ready episode actually carries the already-validated global
traffic/UAV products into TRUTH with the required entities and per-frame
contract fields.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from map_spatial_index import PEDESTRIAN_ROAD_BUFFER_M, MapSpatialIndex  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - supports package imports from repo root.
    from Dataset.tools.map_spatial_index import PEDESTRIAN_ROAD_BUFFER_M, MapSpatialIndex  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "Dataset" / "render_ready_episodes"
DEFAULT_UAV_VALIDATION = ROOT / "Dataset" / "uav_outputs" / "donghu_uav_flow_270s" / "uav_flow_validation.json"
DEFAULT_SUMO_MANIFEST = ROOT / "Dataset" / "sumo_outputs" / "donghu_traffic_270s" / "sumo_traffic_manifest.json"
GROUND_FLOW_MIN_VISIBLE_MOTION_RATIO = 0.85
GROUND_FLOW_SPEED_EPS_MPS = 0.05
BACKGROUND_PEDESTRIAN_MAX_SPEED_MPS = 6.0
BACKGROUND_PEDESTRIAN_SEMANTIC_IDLE = {"phone_call", "chatting"}
_SPATIAL_INDEX_CACHE: MapSpatialIndex | None = None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                try:
                    yield line_number, json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc


def category_from_entity(entity: dict[str, Any]) -> str:
    category = str(entity.get("entity_category") or "").lower()
    if category:
        return category
    label_class = str(entity.get("label_class") or "").lower()
    if label_class in {"pedestrian", "vehicle", "uav", "facility"}:
        return label_class
    return ""


def position_from_entity(entity: dict[str, Any]) -> tuple[float, float, float] | None:
    pose = entity.get("truth_pose") or {}
    position = pose.get("position_enu_m") or pose.get("position")
    if not isinstance(position, list):
        for key in ("initial_position_enu_m", "position_enu_m", "center_enu_m"):
            value = entity.get(key)
            if isinstance(value, list):
                position = value
                break
    if not isinstance(position, list):
        placement = dict(entity.get("placement") or {})
        for key in ("resolved_position_enu_m", "position_enu_m", "center_enu_m"):
            value = placement.get(key)
            if isinstance(value, list):
                position = value
                break
    if not isinstance(position, list) or len(position) < 2:
        return None
    try:
        z = float(position[2]) if len(position) >= 3 else 0.0
        return float(position[0]), float(position[1]), z
    except (TypeError, ValueError):
        return None


def velocity_xy_speed_from_entity(entity: dict[str, Any]) -> float:
    pose = entity.get("truth_pose") or {}
    velocity = pose.get("velocity_enu_mps") or entity.get("vel_mps") or [0.0, 0.0, 0.0]
    if not isinstance(velocity, list) or len(velocity) < 2:
        return 0.0
    try:
        return math.hypot(float(velocity[0]), float(velocity[1]))
    except (TypeError, ValueError):
        return 0.0


def activity_type_from_entity(entity: dict[str, Any]) -> str:
    annotations = dict(entity.get("annotations") or {})
    facets = dict(annotations.get("state_facets") or {})
    activity = dict(facets.get("activity") or {})
    return str(
        annotations.get("activity_type")
        or activity.get("activity_type")
        or entity.get("activity_type")
        or entity.get("state")
        or ""
    ).strip().lower()


def is_visible_background_pedestrian(entity: dict[str, Any]) -> bool:
    if category_from_entity(entity) != "pedestrian":
        return False
    if str(entity.get("role") or "") != "semantic_background_pedestrian":
        return False
    presence = dict(entity.get("render_presence") or {})
    return presence.get("visibility_state") == "visible" and presence.get("offstage") is not True


def validate_background_pedestrian_motion(episode_dir: Path, episode_id: str) -> tuple[list[str], dict[str, Any]]:
    truth_path = episode_dir / "truth_frames.jsonl"
    errors: list[str] = []
    stats: dict[str, dict[str, Any]] = {}
    previous: dict[str, tuple[int, tuple[float, float, float]]] = {}
    for _, frame in iter_jsonl(truth_path):
        tick = int(frame.get("tick") if frame.get("tick") is not None else frame.get("frame_seq") or 0)
        for entity in frame.get("entities") or []:
            if not isinstance(entity, dict) or not is_visible_background_pedestrian(entity):
                continue
            entity_id = str(entity.get("entity_id") or "")
            if not entity_id:
                continue
            contract = dict(entity.get("ground_flow_contract") or {})
            row = stats.setdefault(
                entity_id,
                {
                    "visible": 0,
                    "moving": 0,
                    "min_motion_ratio": max(
                        GROUND_FLOW_MIN_VISIBLE_MOTION_RATIO,
                        float(contract.get("min_visible_motion_ratio") or GROUND_FLOW_MIN_VISIBLE_MOTION_RATIO),
                    ),
                    "first_nonsemantic_stop": None,
                    "first_speed_error": None,
                },
            )
            row["visible"] = int(row["visible"]) + 1
            speed = velocity_xy_speed_from_entity(entity)
            if speed > GROUND_FLOW_SPEED_EPS_MPS:
                row["moving"] = int(row["moving"]) + 1
            else:
                activity = activity_type_from_entity(entity)
                if activity not in BACKGROUND_PEDESTRIAN_SEMANTIC_IDLE and row["first_nonsemantic_stop"] is None:
                    row["first_nonsemantic_stop"] = {
                        "tick": tick,
                        "activity_type": activity,
                        "speed_mps": round(speed, 4),
                    }
            position = position_from_entity(entity)
            if position is None:
                errors.append(f"{episode_id}: background pedestrian {entity_id} missing position at tick {tick}")
                continue
            prior = previous.get(entity_id)
            if prior is not None:
                prior_tick, prior_position = prior
                dt_s = max(1e-6, float(tick - prior_tick) / 10.0)
                implied_speed = math.dist(position[:2], prior_position[:2]) / dt_s
                if implied_speed > BACKGROUND_PEDESTRIAN_MAX_SPEED_MPS and row["first_speed_error"] is None:
                    row["first_speed_error"] = {
                        "from_tick": prior_tick,
                        "to_tick": tick,
                        "speed_mps": round(implied_speed, 3),
                    }
            previous[entity_id] = (tick, position)

    for entity_id, row in sorted(stats.items()):
        visible = int(row["visible"])
        if visible <= 0:
            errors.append(f"{episode_id}: background pedestrian {entity_id} is never visible")
            continue
        moving_ratio = float(row["moving"]) / float(visible)
        min_motion_ratio = float(row["min_motion_ratio"])
        if moving_ratio + 1e-6 < min_motion_ratio:
            errors.append(
                f"{episode_id}: background pedestrian {entity_id} moving ratio too small: "
                f"{moving_ratio:.3f} < {min_motion_ratio:.3f}"
            )
        if row["first_nonsemantic_stop"] is not None:
            stop = dict(row["first_nonsemantic_stop"] or {})
            errors.append(
                f"{episode_id}: background pedestrian {entity_id} stopped without semantic idle at tick "
                f"{stop.get('tick')}: activity_type={stop.get('activity_type')!r}"
            )
        if row["first_speed_error"] is not None:
            speed_error = dict(row["first_speed_error"] or {})
            errors.append(
                f"{episode_id}: background pedestrian {entity_id} implied speed too high between "
                f"tick {speed_error.get('from_tick')} and {speed_error.get('to_tick')}: "
                f"{float(speed_error.get('speed_mps') or 0.0):.3f}m/s > {BACKGROUND_PEDESTRIAN_MAX_SPEED_MPS:.3f}m/s"
            )
    return errors, {
        "checked": True,
        "background_pedestrian_count": len(stats),
        "visible_records": sum(int(row["visible"]) for row in stats.values()),
        "moving_records": sum(int(row["moving"]) for row in stats.values()),
        "min_moving_ratio": min(
            (float(row["moving"]) / float(row["visible"]) for row in stats.values() if int(row["visible"]) > 0),
            default=None,
        ),
    }


def is_global_uav(entity: dict[str, Any]) -> bool:
    return category_from_entity(entity) == "uav" and str(entity.get("source") or "") == "uav_global_flow"


def is_global_pad(entity: dict[str, Any]) -> bool:
    return category_from_entity(entity) == "facility" and str(entity.get("source") or "") == "uav_global_flow"


def spatial_index() -> MapSpatialIndex:
    global _SPATIAL_INDEX_CACHE
    if _SPATIAL_INDEX_CACHE is None:
        _SPATIAL_INDEX_CACHE = MapSpatialIndex.default(ROOT)
    return _SPATIAL_INDEX_CACHE


def check_global_pad_spatial(label: str, position: tuple[float, float, float], errors: list[str]) -> float | None:
    try:
        spatial = spatial_index()
        errors.extend(
            spatial.validation_errors_for_point(
                position,
                context=label,
                allow_road=False,
                allow_green=True,
                road_buffer_m=PEDESTRIAN_ROAD_BUFFER_M,
            )
        )
        clearance = spatial.nearest_lane_clearance(position)
    except Exception as exc:
        errors.append(f"{label}: unable to validate landing pad spatial grounding: {exc}")
        return None
    return clearance


def expected_selected_uav_count(plan_uav: dict[str, Any]) -> int:
    selection = dict(plan_uav.get("selection") or {})
    ids = selection.get("uav_ids")
    if isinstance(ids, list):
        return len(ids)
    return int(selection.get("selected_count") or 0)


def expected_visible_global_pad_count(plan_uav: dict[str, Any]) -> int:
    return int(plan_uav.get("pad_count") or 0)


def validate_episode(args: tuple[str, int, float, float, int, str]) -> dict[str, Any]:
    episode_dir_raw, min_active_uavs, min_global_uav_track_span_m, max_stationary_run_s, entity_sample_step_ticks, truth_check_mode = args
    episode_dir = Path(episode_dir_raw)
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = episode_dir / "episode_manifest.json"
    scenario_plan_path = episode_dir / "scenario_plan.json"
    roster_path = episode_dir / "global_entity_roster.json"
    truth_path = episode_dir / "truth_frames.jsonl"
    trajectory_path = episode_dir / "trajectories.jsonl"
    for path in (manifest_path, scenario_plan_path, roster_path, truth_path, trajectory_path):
        if not path.exists():
            errors.append(f"missing artifact: {path.name}")
    if errors:
        return episode_result(episode_dir, errors=errors, warnings=warnings)

    manifest = read_json(manifest_path)
    scenario_plan = read_json(scenario_plan_path)
    roster = read_json(roster_path)
    episode_id = str(manifest.get("episode_id") or episode_dir.name)
    duration_ticks = int(manifest.get("duration_ticks") or 0)
    expected_frames = duration_ticks + 1 if duration_ticks >= 0 else 0
    manifest_record_counts = dict(manifest.get("record_counts") or {})

    manifest_sumo = dict(manifest.get("sumo_traffic") or {})
    manifest_uav = dict(manifest.get("uav_global_flow") or {})
    plan_sumo = dict(scenario_plan.get("sumo_traffic") or {})
    plan_uav = dict(scenario_plan.get("uav_global_flow") or {})
    if manifest_sumo.get("enabled") is not True or plan_sumo.get("enabled") is not True:
        errors.append("SUMO traffic integration is not enabled in manifest/scenario_plan")
    if manifest_uav.get("enabled") is not True or plan_uav.get("enabled") is not True:
        errors.append("UAV global flow integration is not enabled in manifest/scenario_plan")
    if manifest.get("uav_crosses_boundary") is not True:
        errors.append("manifest uav_crosses_boundary is not true")
    if manifest.get("inspect_observes_boundary") is not True:
        errors.append("manifest inspect_observes_boundary is not true")
    expected_global_uav_roster_count = expected_selected_uav_count(plan_uav)
    expected_global_pad_count = expected_visible_global_pad_count(plan_uav)
    pedestrian_motion_errors, pedestrian_motion_summary = validate_background_pedestrian_motion(episode_dir, episode_id)
    errors.extend(pedestrian_motion_errors[:40])
    if len(pedestrian_motion_errors) > 40:
        errors.append(f"{len(pedestrian_motion_errors) - 40} additional background pedestrian motion errors omitted")

    roster_entities = roster.get("entities") if isinstance(roster, dict) else roster
    roster_categories: set[str] = set()
    global_pad_roster_count = 0
    global_uav_roster_count = 0
    global_pad_road_clearance_min_m = float("inf")
    for entity in roster_entities if isinstance(roster_entities, list) else []:
        if not isinstance(entity, dict):
            continue
        category = category_from_entity(entity)
        if category:
            roster_categories.add(category)
        if is_global_pad(entity):
            global_pad_roster_count += 1
            position = position_from_entity(entity)
            if position is None:
                errors.append(f"global roster pad {entity.get('entity_id')} missing position")
            else:
                clearance = check_global_pad_spatial(
                    f"{episode_id}: roster pad {entity.get('entity_id')}",
                    position,
                    errors,
                )
                if clearance is not None:
                    global_pad_road_clearance_min_m = min(global_pad_road_clearance_min_m, clearance)
        if is_global_uav(entity):
            global_uav_roster_count += 1

    if global_pad_roster_count != expected_global_pad_count:
        errors.append(f"global UAV pad roster count {global_pad_roster_count} != selected visible pad_count {expected_global_pad_count}")
    if global_uav_roster_count != expected_global_uav_roster_count:
        errors.append(f"global UAV roster count {global_uav_roster_count} != selected UAV count {expected_global_uav_roster_count}")

    if truth_check_mode == "manifest":
        truth_record_count = int(manifest_record_counts.get("truth_frames") or 0)
        trajectory_record_count = int(manifest_record_counts.get("trajectories") or 0)
        if expected_frames and truth_record_count != expected_frames:
            errors.append(f"manifest record_counts.truth_frames {truth_record_count} != expected {expected_frames}")
        if trajectory_record_count <= 0:
            errors.append(f"manifest record_counts.trajectories {trajectory_record_count} is not positive")
        first_frame: dict[str, Any] | None = None
        with truth_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    first_frame = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    errors.append(f"truth_frames.jsonl:{line_number}: first frame invalid JSON: {exc}")
                break
        if first_frame is None:
            errors.append("truth_frames.jsonl has no frames")
            return episode_result(
                episode_dir,
                episode_id=episode_id,
                errors=errors,
                warnings=warnings,
                frame_count=truth_record_count,
                roster_categories=sorted(roster_categories),
                frame_categories=[],
                global_pad_roster_count=global_pad_roster_count,
                global_uav_roster_count=global_uav_roster_count,
                global_pad_road_clearance_min_m=round(global_pad_road_clearance_min_m, 3)
                if math.isfinite(global_pad_road_clearance_min_m)
                else None,
                truth_check_mode=truth_check_mode,
                background_pedestrian_motion=pedestrian_motion_summary,
            )

        frame_categories: set[str] = set()
        global_uav_count = 0
        global_pad_count = 0
        for entity in first_frame.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            category = category_from_entity(entity)
            if category:
                frame_categories.add(category)
            position = position_from_entity(entity)
            if position is None:
                errors.append(f"first frame entity {entity.get('entity_id')} missing truth_pose.position_enu_m")
                continue
            if is_global_uav(entity):
                global_uav_count += 1
            elif is_global_pad(entity):
                global_pad_count += 1
                clearance = check_global_pad_spatial(
                    f"{episode_id}: first frame pad {entity.get('entity_id')}",
                    position,
                    errors,
                )
                if clearance is not None:
                    global_pad_road_clearance_min_m = min(global_pad_road_clearance_min_m, clearance)
        uav_payload = dict(first_frame.get("uav_global_flow") or {})
        sumo_payload = dict(first_frame.get("sumo_semantics") or {})
        payload_uav_count = int(uav_payload.get("active_selected_uav_count") or 0)
        if first_frame.get("schema_name") != "truth_frame":
            errors.append("first frame schema_name is not truth_frame")
        if first_frame.get("uav_crosses_boundary") is not True:
            errors.append("first frame uav_crosses_boundary is not true")
        if first_frame.get("inspect_observes_boundary") is not True:
            errors.append("first frame inspect_observes_boundary is not true")
        if sumo_payload.get("enabled") is not True:
            errors.append("first frame sumo_semantics.enabled is not true")
        if uav_payload.get("enabled") is not True:
            errors.append("first frame uav_global_flow.enabled is not true")
        if payload_uav_count > expected_global_uav_roster_count:
            errors.append(f"first frame active_selected_uav_count {payload_uav_count} > selected UAV count {expected_global_uav_roster_count}")
        if global_uav_count != payload_uav_count:
            errors.append(f"first frame global UAV entity count {global_uav_count} != active_selected_uav_count {payload_uav_count}")
        if global_pad_count != expected_global_pad_count:
            errors.append(f"first frame global pad count {global_pad_count} != selected visible pad_count {expected_global_pad_count}")
        first_trajectory_row: dict[str, Any] | None = None
        with trajectory_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    first_trajectory_row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    errors.append(f"trajectories.jsonl:{line_number}: first row invalid JSON: {exc}")
                break
        if first_trajectory_row is None:
            errors.append("trajectories.jsonl has no rows")
        else:
            if "frame_id" not in first_trajectory_row or "sim_time_s" not in first_trajectory_row:
                errors.append("trajectories.jsonl is not truth-frame-derived; missing frame_id/sim_time_s")
            if "pos_enu" not in first_trajectory_row or "vel_mps" not in first_trajectory_row:
                errors.append("trajectories.jsonl first row missing pos_enu/vel_mps")
        return episode_result(
            episode_dir,
            episode_id=episode_id,
            errors=errors,
            warnings=warnings,
            frame_count=truth_record_count,
            roster_categories=sorted(roster_categories),
            frame_categories=sorted(frame_categories),
            global_pad_roster_count=global_pad_roster_count,
            global_uav_roster_count=global_uav_roster_count,
            global_pad_road_clearance_min_m=round(global_pad_road_clearance_min_m, 3)
            if math.isfinite(global_pad_road_clearance_min_m)
            else None,
            min_frame_global_uavs=payload_uav_count,
            min_frame_global_pads=global_pad_count,
            min_frame_sumo_vehicles=0,
            max_frame_sumo_vehicles=int(dict(plan_sumo.get("selection") or {}).get("selected_count") or 0),
            traffic_light_frames=1 if dict(first_frame.get("sumo_traffic_light_states") or {}) else 0,
            active_incident_frames=0,
            sampled_entity_frames=1,
            entity_sample_step_ticks=entity_sample_step_ticks,
            truth_check_mode=truth_check_mode,
            global_uav_track_count=global_uav_roster_count,
            static_global_uav_track_count=0,
            background_pedestrian_motion=pedestrian_motion_summary,
        )

    frame_count = 0
    min_frame_global_uavs = None
    min_frame_global_pads = None
    min_frame_sumo_vehicles = None
    max_frame_sumo_vehicles = 0
    frame_categories: set[str] = set()
    traffic_light_frames = 0
    active_incident_frames = 0
    global_uav_tracks: dict[str, list[tuple[int, float, float, float]]] = {}
    frame_errors: list[str] = []
    sampled_entity_frames = 0

    with truth_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            frame_count += 1
            tick = frame_count - 1
            if tick % max(1, int(entity_sample_step_ticks)) != 0:
                continue
            try:
                frame = json.loads(stripped)
            except json.JSONDecodeError as exc:
                frame_errors.append(f"line {line_number}: invalid JSONL row: {exc}")
                continue
            sampled_entity_frames += 1
            tick = int(frame.get("tick") if frame.get("tick") is not None else frame.get("frame_seq", tick))
            if frame.get("schema_name") != "truth_frame":
                frame_errors.append(f"tick {tick}: schema_name is not truth_frame")
            if frame.get("uav_crosses_boundary") is not True:
                frame_errors.append(f"tick {tick}: uav_crosses_boundary is not true")
            if frame.get("inspect_observes_boundary") is not True:
                frame_errors.append(f"tick {tick}: inspect_observes_boundary is not true")

            sumo_payload = dict(frame.get("sumo_semantics") or {})
            uav_payload = dict(frame.get("uav_global_flow") or {})
            if sumo_payload.get("enabled") is not True:
                frame_errors.append(f"tick {tick}: sumo_semantics.enabled is not true")
            if uav_payload.get("enabled") is not True:
                frame_errors.append(f"tick {tick}: uav_global_flow.enabled is not true")
            payload_uav_count = int(uav_payload.get("active_selected_uav_count") or 0)
            min_frame_global_uavs = payload_uav_count if min_frame_global_uavs is None else min(min_frame_global_uavs, payload_uav_count)
            if payload_uav_count > expected_global_uav_roster_count:
                frame_errors.append(f"tick {tick}: active_selected_uav_count {payload_uav_count} > selected UAV count {expected_global_uav_roster_count}")

            entities = frame.get("entities") or []
            if not isinstance(entities, list):
                frame_errors.append(f"tick {tick}: entities is not a list")
                entities = []
            global_uav_count = 0
            global_pad_count = 0
            sumo_vehicle_count = 0
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                category = category_from_entity(entity)
                if category:
                    frame_categories.add(category)
                position = position_from_entity(entity)
                if position is None:
                    frame_errors.append(f"tick {tick}: entity {entity.get('entity_id')} missing truth_pose.position_enu_m")
                    continue
                if is_global_uav(entity):
                    global_uav_count += 1
                    entity_id = str(entity.get("entity_id") or "")
                    if entity_id:
                        global_uav_tracks.setdefault(entity_id, []).append((tick, position[0], position[1], position[2]))
                elif is_global_pad(entity):
                    global_pad_count += 1
                    clearance = check_global_pad_spatial(
                        f"{episode_id}: tick {tick} pad {entity.get('entity_id')}",
                        position,
                        frame_errors,
                    )
                    if clearance is not None:
                        global_pad_road_clearance_min_m = min(global_pad_road_clearance_min_m, clearance)
                elif category == "vehicle" and str(entity.get("source") or "") == "sumo_traci":
                    sumo_vehicle_count += 1

            min_frame_global_pads = global_pad_count if min_frame_global_pads is None else min(min_frame_global_pads, global_pad_count)
            min_frame_sumo_vehicles = sumo_vehicle_count if min_frame_sumo_vehicles is None else min(min_frame_sumo_vehicles, sumo_vehicle_count)
            max_frame_sumo_vehicles = max(max_frame_sumo_vehicles, sumo_vehicle_count)
            if global_uav_count != payload_uav_count:
                frame_errors.append(f"tick {tick}: sampled global UAV entity count {global_uav_count} != active_selected_uav_count {payload_uav_count}")
            if global_pad_count != expected_global_pad_count:
                frame_errors.append(f"tick {tick}: sampled global pad count {global_pad_count} != selected visible pad_count {expected_global_pad_count}")
            if len(dict(frame.get("sumo_traffic_light_states") or {})) > 0:
                traffic_light_frames += 1
            if len(list(frame.get("sumo_active_incidents") or [])) > 0:
                active_incident_frames += 1

    if frame_errors:
        errors.extend(frame_errors[:40])
        if len(frame_errors) > 40:
            errors.append(f"{len(frame_errors) - 40} additional frame errors omitted")
    if expected_frames and frame_count != expected_frames:
        errors.append(f"truth frame count {frame_count} != expected {expected_frames}")
    if traffic_light_frames != sampled_entity_frames:
        errors.append(f"sampled traffic light state frames {traffic_light_frames} != sampled entity frames {sampled_entity_frames}")
    if max_frame_sumo_vehicles <= 0:
        errors.append("no SUMO vehicle appears in any truth frame")

    static_tracks: list[dict[str, Any]] = []
    sample_period_s = max(0.1, float(entity_sample_step_ticks) / 10.0)
    stationary_run_limit = max(1, int(math.ceil(max_stationary_run_s / sample_period_s)))
    for entity_id, points in global_uav_tracks.items():
        if len(points) < 3:
            continue
        xs = [point[1] for point in points]
        ys = [point[2] for point in points]
        zs = [point[3] for point in points]
        xy_span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        z_span = max(zs) - min(zs)
        longest_stationary_run = 0
        current_run = 0
        previous = points[0]
        for point in points[1:]:
            step = math.sqrt((point[1] - previous[1]) ** 2 + (point[2] - previous[2]) ** 2 + (point[3] - previous[3]) ** 2)
            if step < 0.02:
                current_run += 1
                longest_stationary_run = max(longest_stationary_run, current_run)
            else:
                current_run = 0
            previous = point
        if xy_span < min_global_uav_track_span_m and z_span < min_global_uav_track_span_m:
            static_tracks.append(
                {
                    "entity_id": entity_id,
                    "sample_count": len(points),
                    "xy_span_m": round(xy_span, 3),
                    "z_span_m": round(z_span, 3),
                    "longest_stationary_run_ticks": longest_stationary_run,
                }
            )
        elif longest_stationary_run > stationary_run_limit:
            static_tracks.append(
                {
                    "entity_id": entity_id,
                    "sample_count": len(points),
                    "xy_span_m": round(xy_span, 3),
                    "z_span_m": round(z_span, 3),
                    "longest_stationary_run_ticks": longest_stationary_run,
                }
            )
    if static_tracks:
        errors.append(f"global UAV static/long-stationary tracks: {static_tracks[:20]}")
        if len(static_tracks) > 20:
            errors.append(f"{len(static_tracks) - 20} additional static/long-stationary UAV tracks omitted")

    return episode_result(
        episode_dir,
        episode_id=episode_id,
        errors=errors,
        warnings=warnings,
        frame_count=frame_count,
        roster_categories=sorted(roster_categories),
        frame_categories=sorted(frame_categories),
        global_pad_roster_count=global_pad_roster_count,
        global_uav_roster_count=global_uav_roster_count,
        global_pad_road_clearance_min_m=round(global_pad_road_clearance_min_m, 3)
        if math.isfinite(global_pad_road_clearance_min_m)
        else None,
        min_frame_global_uavs=min_frame_global_uavs or 0,
        min_frame_global_pads=min_frame_global_pads or 0,
        min_frame_sumo_vehicles=min_frame_sumo_vehicles or 0,
        max_frame_sumo_vehicles=max_frame_sumo_vehicles,
        traffic_light_frames=traffic_light_frames,
        active_incident_frames=active_incident_frames,
        sampled_entity_frames=sampled_entity_frames,
        entity_sample_step_ticks=entity_sample_step_ticks,
        global_uav_track_count=len(global_uav_tracks),
        static_global_uav_track_count=len(static_tracks),
        background_pedestrian_motion=pedestrian_motion_summary,
    )


def episode_result(
    episode_dir: Path,
    *,
    episode_id: str | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    errors = errors or []
    warnings = warnings or []
    return {
        "episode": episode_id or episode_dir.name,
        "episode_dir": str(episode_dir),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        **extra,
    }


def validate_generation_outputs(uav_validation_path: Path, sumo_manifest_path: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    payload: dict[str, Any] = {}
    if not uav_validation_path.exists():
        errors.append(f"missing UAV validation output: {uav_validation_path}")
    else:
        uav_validation = read_json(uav_validation_path)
        payload["uav_validation"] = uav_validation
        if uav_validation.get("ok") is not True:
            errors.append("UAV global flow generation validation is not ok")
        if int(uav_validation.get("active_uav_count_min") or 0) < 50:
            errors.append("UAV global flow active_uav_count_min < 50")
        if float(uav_validation.get("sample_period_s") or 0.0) != 0.5:
            errors.append("UAV global flow sample_period_s is not 0.5")
    if not sumo_manifest_path.exists():
        errors.append(f"missing SUMO manifest: {sumo_manifest_path}")
    else:
        sumo_manifest = read_json(sumo_manifest_path)
        payload["sumo_manifest"] = sumo_manifest
        if float(sumo_manifest.get("duration_s") or 0.0) != 270.0:
            errors.append("SUMO duration_s is not 270.0")
        if float(sumo_manifest.get("sample_period_s") or 0.0) != 0.5:
            errors.append("SUMO sample_period_s is not 0.5")
        if int(sumo_manifest.get("max_vehicles") or 0) != 200:
            errors.append("SUMO max_vehicles is not 200")
        mapper = dict(sumo_manifest.get("coordinate_mapper") or {})
        if float(mapper.get("max_error_m") or 999.0) > 0.01:
            errors.append("SUMO->UE coordinate mapper max_error_m is above 0.01m")
    return errors, payload


def parse_episode_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for name in value.split(","):
            stripped = name.strip()
            if stripped:
                names.append(stripped)
    return names


def select_episode_dirs(input_root: Path, episode_names: list[str]) -> list[Path]:
    all_episode_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not episode_names:
        return all_episode_dirs
    by_name = {path.name: path for path in all_episode_dirs}
    missing = [name for name in episode_names if name not in by_name]
    if missing:
        raise SystemExit(f"Unknown episode(s): {', '.join(missing)}")
    return [by_name[name] for name in episode_names]


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight render-ready TRUTH integration validator.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--uav-validation", type=Path, default=DEFAULT_UAV_VALIDATION)
    parser.add_argument("--sumo-manifest", type=Path, default=DEFAULT_SUMO_MANIFEST)
    parser.add_argument("--min-active-uavs", type=int, default=50)
    parser.add_argument("--min-global-uav-track-span-m", type=float, default=2.0)
    parser.add_argument("--max-stationary-run-s", type=float, default=5.0)
    parser.add_argument(
        "--entity-sample-step-ticks",
        type=int,
        default=50,
        help="Parse full entity arrays only every N ticks; frame-level contract counters are still checked on every line.",
    )
    parser.add_argument(
        "--truth-check-mode",
        choices=("manifest", "sampled"),
        default="manifest",
        help="Use manifest for fast full-dataset checks, or sampled to scan truth_frames.jsonl every entity sample step.",
    )
    parser.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--summary-path", type=Path, default=None)
    args = parser.parse_args()

    generation_errors, generation_payload = validate_generation_outputs(args.uav_validation, args.sumo_manifest)
    episode_dirs = select_episode_dirs(args.input_root, parse_episode_names(args.episode))
    worker_count = max(1, int(args.workers or 1))
    task_args = [
        (
            str(path),
            int(args.min_active_uavs),
            float(args.min_global_uav_track_span_m),
            float(args.max_stationary_run_s),
            max(1, int(args.entity_sample_step_ticks)),
            str(args.truth_check_mode),
        )
        for path in episode_dirs
    ]
    episode_results: list[dict[str, Any]] = []
    if worker_count == 1 or len(task_args) <= 1:
        episode_results = [validate_episode(item) for item in task_args]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(validate_episode, item) for item in task_args]
            for future in as_completed(futures):
                episode_results.append(future.result())
        episode_results.sort(key=lambda item: str(item["episode"]))

    failed = [item for item in episode_results if not item["ok"]]
    summary = {
        "ok": not generation_errors and not failed,
        "input_root": str(args.input_root),
        "episode_count": len(episode_results),
        "failed_episode_count": len(failed),
        "workers": worker_count,
        "min_active_uavs": int(args.min_active_uavs),
        "entity_sample_step_ticks": max(1, int(args.entity_sample_step_ticks)),
        "truth_check_mode": str(args.truth_check_mode),
        "generation_errors": generation_errors,
        "generation_validation": {
            "uav_ok": bool(dict(generation_payload.get("uav_validation") or {}).get("ok")),
            "uav_active_uav_count_min": dict(generation_payload.get("uav_validation") or {}).get("active_uav_count_min"),
            "uav_sample_period_s": dict(generation_payload.get("uav_validation") or {}).get("sample_period_s"),
            "sumo_duration_s": dict(generation_payload.get("sumo_manifest") or {}).get("duration_s"),
            "sumo_sample_period_s": dict(generation_payload.get("sumo_manifest") or {}).get("sample_period_s"),
            "sumo_max_vehicles": dict(generation_payload.get("sumo_manifest") or {}).get("max_vehicles"),
            "sumo_coordinate_mapper": dict(dict(generation_payload.get("sumo_manifest") or {}).get("coordinate_mapper") or {}),
        },
        "aggregate": {
            "min_frame_global_uavs": min((int(item.get("min_frame_global_uavs") or 0) for item in episode_results), default=0),
            "min_frame_global_pads": min((int(item.get("min_frame_global_pads") or 0) for item in episode_results), default=0),
            "max_frame_sumo_vehicles": max((int(item.get("max_frame_sumo_vehicles") or 0) for item in episode_results), default=0),
            "total_active_incident_frames": sum(int(item.get("active_incident_frames") or 0) for item in episode_results),
            "total_static_global_uav_tracks": sum(int(item.get("static_global_uav_track_count") or 0) for item in episode_results),
            "total_background_pedestrian_visible_records": sum(
                int(dict(item.get("background_pedestrian_motion") or {}).get("visible_records") or 0) for item in episode_results
            ),
            "min_background_pedestrian_moving_ratio": min(
                (
                    float(dict(item.get("background_pedestrian_motion") or {}).get("min_moving_ratio"))
                    for item in episode_results
                    if dict(item.get("background_pedestrian_motion") or {}).get("min_moving_ratio") is not None
                ),
                default=None,
            ),
        },
        "failed_episodes": failed[:40],
        "episodes": episode_results,
    }
    summary_path = args.summary_path or (args.input_root / "truth_light_validation_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    printable = {key: value for key, value in summary.items() if key != "episodes"}
    printable["summary_path"] = str(summary_path)
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
