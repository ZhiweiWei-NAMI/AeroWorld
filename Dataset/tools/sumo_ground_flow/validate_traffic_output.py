"""Validate SUMO traffic exports against the capture-oriented contract."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from .explicit_vehicle_plan import validate_source_presence_manifest_contract
except ImportError:  # pragma: no cover - direct script execution from this directory.
    from explicit_vehicle_plan import validate_source_presence_manifest_contract


REQUIRED_VEHICLE_FIELDS = {
    "vehicle_id",
    "vehicle_type",
    "route_id",
    "sumo_edge_id",
    "sumo_lane_id",
    "lane_position_m",
    "sumo_xy_m",
    "sumo_front_bumper_xy_m",
    "sumo_position_reference",
    "truth_front_bumper_enu_m",
    "truth_position_enu_m",
    "truth_position_reference",
    "truth_yaw_deg",
    "center_lane_position_m",
    "speed_mps",
    "accel_mps2",
    "signals",
    "dimensions_m",
    "source",
}

CORE_BACKGROUND_VEHICLE_FIELDS = {
    "vehicle_id",
    "sumo_edge_id",
    "sumo_lane_id",
    "sumo_xy_m",
    "truth_position_enu_m",
    "source",
}

REQUIRED_TRAFFIC_LIGHT_FIELDS = {
    "tls_id",
    "program_id",
    "phase_index",
    "state",
    "next_switch_s",
    "controlled_links",
}

ROOT = Path(__file__).resolve().parents[3]
LANE_META_CSV = ROOT / "Config" / "LowAltitude" / "Maps" / "donghu_road_topo" / "traffic_bundle" / "lane_meta.csv"
MULTILANE_MIN_GLOBAL_SAMPLES = 250
EXPECTED_TIME_AXIS_POLICY = "episode_local_vehicle_sumo_after_warmup_v1"
EXPECTED_WARMUP_S = 90.0
EXPECTED_CAPTURE_DURATION_S = 90.0
EXPECTED_STEP_LENGTH_S = 0.1
EXPECTED_SAMPLE_PERIOD_S = 0.1
MIN_CAPTURE_START_VEHICLE_RATIO = 0.80
MIN_CAPTURE_WINDOW_VEHICLE_RATIO = 0.70
TRAFFIC_SLOT_TICKS = 100
TRAFFIC_SLOT_COUNT = 9
MIN_ROI_BACKGROUND_FRAMES_PER_NONPROTECTED_SLOT = 5
MIN_BACKGROUND_ONLY_ROI_BACKGROUND_TICKS = 100
NONTRAFFIC_MAIN_ROAD_EVENT_SPACE_CLASS = "nontraffic_main_road_grid"
EXPECTED_EPISODE_GIF_FRAME_COUNT = 181
EXPECTED_EPISODE_GIF_TICK_START = 0
EXPECTED_EPISODE_GIF_TICK_END = 900
EXPECTED_EPISODE_GIF_TICK_STRIDE = 5
EXPECTED_EPISODE_GIF_SUFFIX = "roi_extended_episode_only"


def _classify_error(message: str) -> str:
    text = message.lower()
    if "sumo_traffic_manifest.json" in text or "traffic manifest" in text:
        return "missing_manifest" if "missing" in text else "manifest_contract"
    if "sumo_topo_vehicles" in text or "gif" in text or "visualization" in text:
        return "missing_gif_or_manifest" if "missing" in text else "gif_gate"
    if "roi background coverage missing" in text or "background-only roi traffic too sparse" in text:
        return "roi_background_missing"
    if "protected window" in text or "protected windows" in text:
        return "protected_window_intrusion"
    if "outside their control windows" in text:
        return "semantic_outside_control"
    if "opposite sumo lane direction" in text or "lane_position decreases" in text:
        return "semantic_reverse_lane"
    return "contract"


def _error_records(errors: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "category": _classify_error(error),
            "message": error,
        }
        for error in errors
    ]


def _error_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(record.get("category") or "contract") for record in records))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _find_event_space_class(value: Any) -> str:
    if isinstance(value, dict):
        if str(value.get("rule") or "") == "spatial_grid_assignment" and value.get("event_space_class"):
            return str(value.get("event_space_class") or "")
        if value.get("event_space_class"):
            return str(value.get("event_space_class") or "")
        for child in value.values():
            found = _find_event_space_class(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_event_space_class(child)
            if found:
                return found
    return ""


def _scene_event_space_class_summary(explicit_plan: dict[str, Any]) -> dict[str, Any]:
    episode_id = str(explicit_plan.get("episode_id") or "")
    summary: dict[str, Any] = {
        "policy": "scene_setup_spatial_grid_assignment_event_space_class_v1",
        "episode_id": episode_id,
        "event_space_class": "",
        "exempts_background_only_roi_tick_gate": False,
    }
    if not episode_id:
        summary["load_status"] = "missing_episode_id"
        return summary

    episode_manifest_path = ROOT / "Dataset" / "episodes" / episode_id / "episode_manifest.json"
    summary["episode_manifest_path"] = str(episode_manifest_path)
    if not episode_manifest_path.exists():
        summary["load_status"] = "missing_episode_manifest"
        return summary

    try:
        episode_manifest = _load_json(episode_manifest_path)
        scene_setup_value = str(episode_manifest.get("source_scene_setup_path") or "")
        if not scene_setup_value:
            summary["load_status"] = "missing_source_scene_setup_path"
            return summary
        scene_setup_path = _resolve_project_path(scene_setup_value)
        summary["scene_setup_path"] = str(scene_setup_path)
        if not scene_setup_path.exists():
            summary["load_status"] = "missing_scene_setup"
            return summary
        scene_setup = _load_json(scene_setup_path)
        event_space_class = _find_event_space_class(scene_setup)
    except (OSError, json.JSONDecodeError) as exc:
        summary["load_status"] = "load_error"
        summary["load_error"] = str(exc)
        return summary

    summary["load_status"] = "ok"
    summary["event_space_class"] = event_space_class
    summary["exempts_background_only_roi_tick_gate"] = event_space_class == NONTRAFFIC_MAIN_ROAD_EVENT_SPACE_CLASS
    return summary


def _point_in_bbox(point: Sequence[float], bbox: Sequence[float]) -> bool:
    return (
        len(point) >= 2
        and len(bbox) >= 4
        and float(bbox[0]) <= float(point[0]) <= float(bbox[2])
        and float(bbox[1]) <= float(point[1]) <= float(bbox[3])
    )


def _episode_tick(frame: dict[str, Any], *, capture_start_s: float, tick_hz: float) -> int:
    return int(round((float(frame.get("sim_time_s") or 0.0) - float(capture_start_s)) * float(tick_hz)))


def _load_explicit_plan_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    plan_meta = dict(manifest.get("explicit_vehicle_plan") or {})
    path_value = str(plan_meta.get("path") or "")
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    return _load_json(path)


def _visualization_gate_summary(output_dir: Path) -> dict[str, Any]:
    gif_path = output_dir / f"sumo_topo_vehicles_{EXPECTED_EPISODE_GIF_SUFFIX}.gif"
    manifest_path = output_dir / f"sumo_traffic_visualization_{EXPECTED_EPISODE_GIF_SUFFIX}.json"
    frames_dir = output_dir / f"visualization_frames_{EXPECTED_EPISODE_GIF_SUFFIX}"
    failures: list[str] = []
    summary: dict[str, Any] = {
        "policy": "episode_only_roi_extended_gif_0_900_every5_v1",
        "gif_path": str(gif_path),
        "manifest_path": str(manifest_path),
        "frames_dir": str(frames_dir),
        "gif_exists": gif_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "frames_dir_exists": frames_dir.exists(),
        "expected_png_frame_count": EXPECTED_EPISODE_GIF_FRAME_COUNT,
        "expected_episode_tick_first": EXPECTED_EPISODE_GIF_TICK_START,
        "expected_episode_tick_last": EXPECTED_EPISODE_GIF_TICK_END,
        "expected_episode_tick_stride": EXPECTED_EPISODE_GIF_TICK_STRIDE,
        "failures": failures,
    }
    if not gif_path.exists():
        failures.append(f"missing episode-only ROI+extended GIF: {gif_path}")
    if not manifest_path.exists():
        failures.append(f"missing episode-only ROI+extended visualization manifest: {manifest_path}")
        return summary

    try:
        visualization = _load_json(manifest_path)
    except json.JSONDecodeError as exc:
        failures.append(f"invalid episode-only ROI+extended visualization manifest: {manifest_path}: {exc}")
        return summary

    summary.update(
        {
            "episode_only": bool(visualization.get("episode_only")),
            "bounds_mode": visualization.get("bounds_mode"),
            "png_frame_count": int(visualization.get("png_frame_count") or 0),
            "visualized_frame_stride": int(visualization.get("visualized_frame_stride") or 0),
            "episode_tick_first": visualization.get("episode_tick_first"),
            "episode_tick_last": visualization.get("episode_tick_last"),
            "episode_tick_stride": visualization.get("episode_tick_stride"),
            "episode_tick_count": int(visualization.get("episode_tick_count") or 0),
            "overlay_has_roi_bbox": bool(dict(visualization.get("overlay") or {}).get("bbox_enu_m")),
            "overlay_has_expanded_bbox": bool(dict(visualization.get("overlay") or {}).get("expanded_bbox_enu_m")),
        }
    )
    if not summary["episode_only"]:
        failures.append("episode-only ROI+extended visualization manifest has episode_only=false")
    if summary["bounds_mode"] != "episode-expanded-roi":
        failures.append(
            "episode-only ROI+extended visualization must use bounds_mode=episode-expanded-roi; "
            f"got {summary['bounds_mode']}"
        )
    if summary["png_frame_count"] != EXPECTED_EPISODE_GIF_FRAME_COUNT:
        failures.append(
            "episode-only ROI+extended GIF must have 181 PNG frames for ticks 0..900 every 5; "
            f"got {summary['png_frame_count']}"
        )
    if summary["visualized_frame_stride"] != EXPECTED_EPISODE_GIF_TICK_STRIDE:
        failures.append(
            "episode-only ROI+extended GIF must use frame stride 5 over 0.1s truth frames; "
            f"got {summary['visualized_frame_stride']}"
        )
    if summary["episode_tick_first"] != EXPECTED_EPISODE_GIF_TICK_START:
        failures.append(
            "episode-only ROI+extended GIF must start at episode_tick=0; "
            f"got {summary['episode_tick_first']}"
        )
    if summary["episode_tick_last"] != EXPECTED_EPISODE_GIF_TICK_END:
        failures.append(
            "episode-only ROI+extended GIF must end at episode_tick=900; "
            f"got {summary['episode_tick_last']}"
        )
    if summary["episode_tick_stride"] != EXPECTED_EPISODE_GIF_TICK_STRIDE:
        failures.append(
            "episode-only ROI+extended GIF manifest must record episode_tick_stride=5; "
            f"got {summary['episode_tick_stride']}"
        )
    if summary["episode_tick_count"] != EXPECTED_EPISODE_GIF_FRAME_COUNT:
        failures.append(
            "episode-only ROI+extended GIF manifest must record episode_tick_count=181; "
            f"got {summary['episode_tick_count']}"
        )
    if not summary["overlay_has_roi_bbox"] or not summary["overlay_has_expanded_bbox"]:
        failures.append("episode-only ROI+extended visualization manifest must include ROI and expanded bbox overlays")
    if frames_dir.exists():
        png_count = len(list(frames_dir.glob("sumo_topo_vehicles_*.png")))
        summary["frames_dir_png_count"] = png_count
        if png_count != EXPECTED_EPISODE_GIF_FRAME_COUNT:
            failures.append(
                "episode-only ROI+extended frames directory must contain 181 PNG frames; "
                f"got {png_count}"
            )
    return summary


def _roi_background_summary(
    frames: Sequence[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    explicit_plan: dict[str, Any],
    capture_start_s: float,
    capture_duration_s: float,
) -> dict[str, Any]:
    spatial_scope = dict(explicit_plan.get("spatial_scope") or {})
    roi_bbox = list(spatial_scope.get("bbox_enu_m") or [])
    expanded_bbox = list(spatial_scope.get("expanded_bbox_enu_m") or roi_bbox)
    tick_hz = float(manifest.get("tick_hz") or 10.0)
    roi_edges = {str(edge_id) for edge_id in spatial_scope.get("roi_visible_sumo_edges") or spatial_scope.get("core_sumo_edges") or []}
    traffic_case = dict(explicit_plan.get("traffic_case") or {})
    semantic_plan_vehicle_ids = {
        str(vehicle.get("vehicle_id") or "")
        for vehicle in explicit_plan.get("vehicles") or []
        if str(vehicle.get("traffic_role") or "") == "semantic_vehicle" and str(vehicle.get("vehicle_id") or "")
    }
    has_semantic_vehicle_case = bool(traffic_case.get("has_semantic_vehicle")) or bool(semantic_plan_vehicle_ids)
    protected_constraints: list[dict[str, Any]] = []
    protected_ranges: list[list[int]] = []
    if has_semantic_vehicle_case:
        for constraint in explicit_plan.get("semantic_traffic_constraints") or []:
            protected_edges = {str(edge_id) for edge_id in constraint.get("protected_edges") or []}
            tick_range = [int(value) for value in constraint.get("tick_range") or []]
            if len(tick_range) >= 2 and (not roi_edges or protected_edges & roi_edges):
                protected_ranges.append([tick_range[0], tick_range[1]])
                protected_constraints.append(
                    {
                        "constraint_id": str(constraint.get("constraint_id") or constraint.get("event_id") or ""),
                        "tick_range": [tick_range[0], tick_range[1]],
                        "protected_edges": protected_edges,
                        "protected_bbox_enu_m": list(constraint.get("protected_bbox_enu_m") or []),
                        "allowed_vehicle_ids": [str(value) for value in constraint.get("allowed_vehicle_ids") or []],
                    }
                )

    slot_unprotected_frames: Counter[int] = Counter()
    slot_roi_background_frames: Counter[int] = Counter()
    slot_roi_background_unique: dict[int, set[str]] = defaultdict(set)
    slot_roi_background_matching_frames: Counter[int] = Counter()
    slot_roi_background_matching_unique: dict[int, set[str]] = defaultdict(set)
    protected_roi_background_frames = 0
    protected_roi_background_ids: set[str] = set()
    roi_background_frames = 0
    roi_background_ids: set[str] = set()
    semantic_windows: dict[str, tuple[int, int]] = {}
    planned_slot_by_vehicle_id: dict[str, int] = {}
    for vehicle in explicit_plan.get("vehicles") or []:
        vehicle_id = str(vehicle.get("vehicle_id") or "")
        if str(vehicle.get("traffic_role") or "") == "deterministic_background_vehicle" and vehicle_id:
            try:
                planned_slot_by_vehicle_id[vehicle_id] = int(vehicle.get("traffic_slot_index"))
            except (TypeError, ValueError):
                pass
        if str(vehicle.get("traffic_role") or "") != "semantic_vehicle":
            continue
        ticks = [int(control.get("tick") or 0) for control in vehicle.get("semantic_controls") or []]
        if vehicle_id and ticks:
            semantic_windows[vehicle_id] = (min(ticks), max(ticks))

    semantic_outside_window_frames = 0
    semantic_outside_window_ids: set[str] = set()
    capture_end_s = float(capture_start_s) + float(capture_duration_s)
    for frame in frames:
        sim_time_s = float(frame.get("sim_time_s") or 0.0)
        if sim_time_s < float(capture_start_s) - 1e-9 or sim_time_s > capture_end_s + 1e-9:
            continue
        tick = max(0, min(900, _episode_tick(frame, capture_start_s=capture_start_s, tick_hz=tick_hz)))
        slot = max(0, min(TRAFFIC_SLOT_COUNT - 1, tick // TRAFFIC_SLOT_TICKS))
        active_protected_edges: set[str] = set()
        for constraint in protected_constraints:
            start_tick, end_tick = [int(value) for value in constraint.get("tick_range") or [0, -1]]
            if start_tick <= tick <= end_tick:
                active_protected_edges.update(str(edge_id) for edge_id in constraint.get("protected_edges") or [])
        frame_has_unprotected_roi_edge = True
        if roi_edges and active_protected_edges:
            frame_has_unprotected_roi_edge = bool(roi_edges - active_protected_edges)
        if frame_has_unprotected_roi_edge:
            slot_unprotected_frames[slot] += 1
        roi_background_seen_this_frame = False
        roi_background_matching_seen_this_frame = False
        for vehicle in frame.get("vehicles") or []:
            vehicle_id = str(vehicle.get("vehicle_id") or "")
            if vehicle_id in semantic_windows:
                start_tick, end_tick = semantic_windows[vehicle_id]
                point = vehicle.get("truth_position_enu_m") or []
                if (tick < start_tick or tick > end_tick) and expanded_bbox and _point_in_bbox(point, expanded_bbox):
                    semantic_outside_window_frames += 1
                    semantic_outside_window_ids.add(vehicle_id)
                continue
            point = vehicle.get("truth_position_enu_m") or []
            if roi_bbox and _point_in_bbox(point, roi_bbox):
                vehicle_edge = str(vehicle.get("sumo_edge_id") or vehicle.get("edge_id") or "")
                violating_constraint = False
                for constraint in protected_constraints:
                    start_tick, end_tick = [int(value) for value in constraint.get("tick_range") or [0, -1]]
                    if tick < start_tick or tick > end_tick:
                        continue
                    protected_edges = set(str(edge_id) for edge_id in constraint.get("protected_edges") or [])
                    if protected_edges:
                        if vehicle_edge and vehicle_edge in protected_edges:
                            violating_constraint = True
                            break
                        continue
                    protected_bbox = list(constraint.get("protected_bbox_enu_m") or [])
                    if protected_bbox and _point_in_bbox(point, protected_bbox):
                        violating_constraint = True
                        break
                if violating_constraint:
                    protected_roi_background_frames += 1
                    protected_roi_background_ids.add(vehicle_id)
                else:
                    roi_background_seen_this_frame = True
                    slot_roi_background_unique[slot].add(vehicle_id)
                    if planned_slot_by_vehicle_id.get(vehicle_id) == slot:
                        roi_background_matching_seen_this_frame = True
                        slot_roi_background_matching_unique[slot].add(vehicle_id)
        if roi_background_seen_this_frame and frame_has_unprotected_roi_edge:
            slot_roi_background_frames[slot] += 1
            roi_background_frames += 1
            roi_background_ids.update(slot_roi_background_unique.get(slot, set()))
        if roi_background_matching_seen_this_frame and frame_has_unprotected_roi_edge:
            slot_roi_background_matching_frames[slot] += 1

    slots: list[dict[str, Any]] = []
    failing_slots: list[int] = []
    enforce_matching_slot_vehicle = not protected_constraints
    for slot in range(TRAFFIC_SLOT_COUNT):
        unprotected_frames = int(slot_unprotected_frames.get(slot, 0))
        roi_frames = int(slot_roi_background_frames.get(slot, 0))
        matching_frames = int(slot_roi_background_matching_frames.get(slot, 0))
        matching_unique = slot_roi_background_matching_unique.get(slot, set())
        minimum = min(MIN_ROI_BACKGROUND_FRAMES_PER_NONPROTECTED_SLOT, unprotected_frames)
        missing_frames = unprotected_frames > 0 and roi_frames < minimum
        missing_matching_slot_vehicle = (
            enforce_matching_slot_vehicle
            and unprotected_frames > 0
            and not matching_unique
        )
        if missing_frames or missing_matching_slot_vehicle:
            failing_slots.append(slot)
        failure_category = "roi_background_missing" if missing_frames or missing_matching_slot_vehicle else ""
        slots.append(
            {
                "slot_index": slot,
                "failure_category": failure_category,
                "unprotected_frame_count": unprotected_frames,
                "roi_background_frame_count": roi_frames,
                "roi_background_unique_count": len(slot_roi_background_unique.get(slot, set())),
                "roi_background_vehicle_ids": sorted(slot_roi_background_unique.get(slot, set()))[:20],
                "roi_background_matching_slot_frame_count": matching_frames,
                "roi_background_matching_slot_unique_count": len(matching_unique),
                "roi_background_matching_slot_vehicle_ids": sorted(matching_unique)[:20],
                "minimum_required_roi_background_frames": minimum,
                "minimum_required_matching_slot_vehicle_count": 1 if enforce_matching_slot_vehicle and unprotected_frames > 0 else 0,
            }
        )
    return {
        "policy": "explicit_roi_background_required_outside_protected_windows_v2",
        "roi_bbox_enu_m": roi_bbox,
        "expanded_bbox_enu_m": expanded_bbox,
        "traffic_case": traffic_case,
        "has_semantic_vehicle_case": bool(has_semantic_vehicle_case),
        "protected_tick_ranges": protected_ranges,
        "protected_constraint_count": len(protected_constraints),
        "roi_background_frame_count": roi_background_frames,
        "roi_background_unique_count": len(roi_background_ids),
        "roi_background_vehicle_ids": sorted(roi_background_ids)[:50],
        "background_only_minimum_roi_background_ticks": MIN_BACKGROUND_ONLY_ROI_BACKGROUND_TICKS,
        "slots": slots,
        "failing_nonprotected_slots": failing_slots,
        "protected_roi_background_frame_count": protected_roi_background_frames,
        "protected_roi_background_vehicle_ids": sorted(protected_roi_background_ids)[:20],
        "semantic_outside_control_window_frame_count": semantic_outside_window_frames,
        "semantic_outside_control_window_vehicle_ids": sorted(semantic_outside_window_ids),
    }


def _semantic_lane_direction_summary(explicit_plan: dict[str, Any]) -> dict[str, Any]:
    vehicles: list[dict[str, Any]] = []
    failures: list[str] = []
    for vehicle in explicit_plan.get("vehicles") or []:
        if str(vehicle.get("traffic_role") or "") != "semantic_vehicle":
            continue
        vehicle_id = str(vehicle.get("vehicle_id") or "")
        controls = sorted(
            list(vehicle.get("semantic_controls") or []),
            key=lambda item: int(item.get("tick") or 0),
        )
        previous_tick: int | None = None
        previous_s: float | None = None
        reverse_segments: list[dict[str, Any]] = []
        forward_segments = 0
        stationary_segments = 0
        for control in controls:
            projection = dict(control.get("semantic_lane_projection") or {})
            if "lane_position_m" not in projection:
                continue
            tick = int(control.get("tick") or 0)
            lane_s = float(projection.get("lane_position_m") or 0.0)
            if previous_s is not None and previous_tick is not None:
                delta_s = lane_s - previous_s
                if delta_s < -0.25:
                    reverse_segments.append(
                        {
                            "from_tick": previous_tick,
                            "to_tick": tick,
                            "delta_lane_position_m": round(float(delta_s), 6),
                        }
                    )
                elif delta_s > 0.25:
                    forward_segments += 1
                else:
                    stationary_segments += 1
            previous_tick = tick
            previous_s = lane_s
        status = "ok"
        if reverse_segments:
            status = "wrong_way"
            failures.append(
                f"{vehicle_id} semantic lane_position decreases on {len(reverse_segments)} control segments"
            )
        vehicles.append(
            {
                "vehicle_id": vehicle_id,
                "status": status,
                "control_count": len(controls),
                "forward_segment_count": forward_segments,
                "stationary_segment_count": stationary_segments,
                "reverse_segment_count": len(reverse_segments),
                "reverse_segments": reverse_segments[:10],
            }
        )
    return {
        "policy": "semantic_controls_must_follow_sumo_lane_position_v1",
        "vehicle_count": len(vehicles),
        "vehicles": vehicles,
        "failures": failures,
    }


def _load_lane_meta(path: Path = LANE_META_CSV) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            edge_id = str(row.get("edge_id") or "")
            if not edge_id.startswith("cg_edge_"):
                continue
            direction_role = str(row.get("direction_role") or "")
            if direction_role not in {"f", "r"}:
                continue
            try:
                physical_lane_index = int(float(row.get("physical_lane_index") or 0))
            except (TypeError, ValueError):
                physical_lane_index = 0
            rows[edge_id] = {
                "source_road_id": str(row.get("source_road_id") or ""),
                "direction_role": direction_role,
                "physical_lane_index": physical_lane_index,
            }
    return rows


def _multilane_summary(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lane_meta = _load_lane_meta()
    expected_groups: dict[tuple[str, str], set[int]] = defaultdict(set)
    for meta in lane_meta.values():
        expected_groups[(str(meta["source_road_id"]), str(meta["direction_role"]))].add(int(meta["physical_lane_index"]))
    observed_groups: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for frame in frames:
        for vehicle in frame.get("vehicles") or []:
            edge_id = str(vehicle.get("sumo_edge_id") or "")
            meta = lane_meta.get(edge_id)
            if not meta:
                continue
            key = (str(meta["source_road_id"]), str(meta["direction_role"]))
            observed_groups[key][int(meta["physical_lane_index"])] += 1

    checked: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []
    for key, expected_lanes in sorted(expected_groups.items(), key=lambda item: (int(item[0][0] or 0), item[0][1])):
        if len(expected_lanes) < 2:
            continue
        observed = observed_groups.get(key, Counter())
        total = sum(observed.values())
        if total < MULTILANE_MIN_GLOBAL_SAMPLES:
            continue
        dominant = max(observed.values()) / float(total) if observed else 0.0
        record = {
            "source_road_id": key[0],
            "direction_role": key[1],
            "expected_physical_lanes": sorted(expected_lanes),
            "observed_lane_sample_counts": {str(k): int(v) for k, v in sorted(observed.items())},
            "total_vehicle_samples": int(total),
            "used_physical_lane_count": len(observed),
            "dominant_lane_share": round(dominant, 4),
        }
        checked.append(record)
        if len(observed) < min(2, len(expected_lanes)):
            failures.append(
                f"road={key[0]} direction={key[1]} uses {len(observed)}/{len(expected_lanes)} physical lanes"
            )
        elif dominant >= 0.90:
            warnings.append(f"road={key[0]} direction={key[1]} dominant lane share {dominant:.3f}")
    return {
        "policy": "episode_local_sumo_multilane_distribution_audit_v1",
        "groups_checked": len(checked),
        "groups": checked[:80],
        "failures": failures,
        "warnings": warnings,
    }


def validate_output(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "sumo_traffic_manifest.json"
    incident_plan_path = output_dir / "sumo_incident_plan.json"
    frames_path = output_dir / "sumo_traffic_frames.jsonl"
    errors: list[str] = []
    warnings: list[str] = []
    visualization_gate = _visualization_gate_summary(output_dir)
    for path in (manifest_path, incident_plan_path, frames_path):
        if not path.exists():
            errors.append(f"missing required output: {path}")
    errors.extend(str(item) for item in visualization_gate.get("failures") or [])
    if errors:
        records = _error_records(errors)
        return {
            "ok": False,
            "errors": errors,
            "error_records": records,
            "error_counts_by_category": _error_counts(records),
            "warnings": warnings,
            "visualization_gate": visualization_gate,
        }

    manifest = _load_json(manifest_path)
    plan = _load_json(incident_plan_path)
    frames = _load_jsonl(frames_path)
    duration_s = float(manifest.get("duration_s") or 0.0)
    warmup_s = float(manifest.get("warmup_s") or 0.0)
    capture_start_s = float(manifest.get("capture_start_s") or warmup_s)
    capture_duration_s = float(manifest.get("capture_duration_s") or 0.0)
    step_length_s = float(manifest.get("step_length_s") or 0.0)
    sample_period_s = float(manifest.get("sample_period_s") or 0.0)
    sample_every_ticks = int(manifest.get("sample_every_ticks") or 0)
    expected_frame_count = int(round(duration_s / sample_period_s)) + 1 if sample_period_s > 0 else 0
    max_vehicles = int(manifest.get("max_vehicles") or 0)
    explicit_vehicle_plan = dict(manifest.get("explicit_vehicle_plan") or {})
    explicit_activation = dict(manifest.get("explicit_vehicle_activation") or {})
    explicit_mode = bool(explicit_vehicle_plan.get("enabled"))
    loaded_explicit_plan = _load_explicit_plan_from_manifest(manifest) if explicit_mode else {}
    if not explicit_mode:
        errors.append("traffic manifest must enable the explicit vehicle plan")
    traffic_case = dict(
        dict(manifest.get("explicit_vehicle_plan") or {}).get("traffic_case")
        or dict(loaded_explicit_plan.get("traffic_case") or {})
    )
    semantic_plan_vehicle_ids = {
        str(vehicle.get("vehicle_id") or "")
        for vehicle in loaded_explicit_plan.get("vehicles") or []
        if str(vehicle.get("traffic_role") or "") == "semantic_vehicle" and str(vehicle.get("vehicle_id") or "")
    }
    semantic_constraint_count = len(list(loaded_explicit_plan.get("semantic_traffic_constraints") or []))
    has_semantic_vehicle_case = bool(traffic_case.get("has_semantic_vehicle")) or bool(semantic_plan_vehicle_ids)
    background_only_profile = bool(explicit_mode and loaded_explicit_plan and not has_semantic_vehicle_case)
    scene_event_space_class = _scene_event_space_class_summary(loaded_explicit_plan) if loaded_explicit_plan else {}
    background_only_roi_tick_gate_exempted = bool(
        background_only_profile
        and scene_event_space_class.get("event_space_class") == NONTRAFFIC_MAIN_ROAD_EVENT_SPACE_CLASS
    )
    validation_profile = (
        "background_only_gif_gate"
        if background_only_profile
        else "semantic_vehicle_full_gate"
        if explicit_mode
        else "invalid_non_explicit_traffic"
    )
    if background_only_profile:
        if str(traffic_case.get("case_type") or "") != "background_only":
            errors.append(f"background-only explicit plan must record traffic_case.case_type=background_only; got {traffic_case}")
        if semantic_constraint_count:
            errors.append(
                f"background-only explicit plan must not contain protected semantic traffic constraints; got {semantic_constraint_count}"
            )

    if step_length_s <= 0.0:
        errors.append(f"step_length_s must be positive, got {step_length_s}")
    elif abs(step_length_s - EXPECTED_STEP_LENGTH_S) > 1e-9:
        errors.append(f"step_length_s must be {EXPECTED_STEP_LENGTH_S}, got {step_length_s}")
    if sample_period_s <= 0.0:
        errors.append(f"sample_period_s must be positive, got {sample_period_s}")
    elif abs(sample_period_s - EXPECTED_SAMPLE_PERIOD_S) > 1e-9:
        errors.append(f"sample_period_s must be {EXPECTED_SAMPLE_PERIOD_S}, got {sample_period_s}")
    if abs(sample_period_s - step_length_s) > 1e-9:
        errors.append(f"sample_period_s must match step_length_s to avoid render-time interpolation, got {sample_period_s} vs {step_length_s}")
    if sample_every_ticks != 1:
        errors.append(f"sample_every_ticks must be 1 for step-aligned truth, got {sample_every_ticks}")
    if str(manifest.get("time_axis_policy") or "") != EXPECTED_TIME_AXIS_POLICY:
        errors.append(f"time_axis_policy must be {EXPECTED_TIME_AXIS_POLICY}; got {manifest.get('time_axis_policy')}")
    if abs(warmup_s - EXPECTED_WARMUP_S) > 1e-9:
        errors.append(f"warmup_s must be {EXPECTED_WARMUP_S}, got {warmup_s}")
    if abs(capture_start_s - warmup_s) > 1e-9:
        errors.append(f"capture_start_s must equal warmup_s, got {capture_start_s} vs {warmup_s}")
    if abs(capture_duration_s - EXPECTED_CAPTURE_DURATION_S) > 1e-9:
        errors.append(f"capture_duration_s must be {EXPECTED_CAPTURE_DURATION_S}, got {capture_duration_s}")
    if abs(duration_s - (warmup_s + capture_duration_s)) > 1e-9:
        errors.append(f"duration_s must equal warmup_s + capture_duration_s, got {duration_s}")
    if len(frames) != expected_frame_count:
        errors.append(f"expected {expected_frame_count} frames, got {len(frames)}")
    if frames and abs(float(frames[0].get("sim_time_s") or 0.0) - 0.0) > 1e-9:
        errors.append("first frame must be sim_time_s=0.0")
    if frames and abs(float(frames[-1].get("sim_time_s") or 0.0) - duration_s) > 1e-6:
        errors.append(f"last frame must be sim_time_s={duration_s}, got {frames[-1].get('sim_time_s')}")

    for prev, current in zip(frames, frames[1:]):
        dt = round(float(current["sim_time_s"]) - float(prev["sim_time_s"]), 6)
        if abs(dt - sample_period_s) > 1e-6:
            errors.append(
                f"non-sample-period frame delta at {prev.get('sim_time_s')} -> {current.get('sim_time_s')}: {dt}"
            )
            break
        tick_delta = int(current["tick"]) - int(prev["tick"])
        if tick_delta != sample_every_ticks:
            errors.append(
                f"non-sample-every-ticks frame delta at {prev.get('tick')} -> {current.get('tick')}: {tick_delta}"
            )
            break

    traffic_light_counts = [len(frame.get("traffic_lights") or []) for frame in frames]
    if not traffic_light_counts or min(traffic_light_counts) <= 0:
        message = "every sampled frame must include traffic light records"
        if background_only_profile:
            warnings.append(message)
        else:
            errors.append(message)
    expected_traffic_light_count = int(manifest.get("traffic_light_count") or 0)
    if expected_traffic_light_count > 0 and traffic_light_counts and (
        min(traffic_light_counts) != expected_traffic_light_count
        or max(traffic_light_counts) != expected_traffic_light_count
    ):
        message = (
            f"traffic light count must stay {expected_traffic_light_count}; "
            f"min={min(traffic_light_counts)}, max={max(traffic_light_counts)}"
        )
        if background_only_profile:
            warnings.append(message)
        else:
            errors.append(message)
    for frame in frames[: min(5, len(frames))]:
        for traffic_light in frame.get("traffic_lights") or []:
            missing = REQUIRED_TRAFFIC_LIGHT_FIELDS - set(traffic_light)
            if missing:
                message = f"traffic light record missing fields: {sorted(missing)}"
                if background_only_profile:
                    warnings.append(message)
                else:
                    errors.append(message)
                break

    vehicle_counts = [len(frame.get("vehicles") or []) for frame in frames]
    required_vehicle_fields = CORE_BACKGROUND_VEHICLE_FIELDS if background_only_profile else REQUIRED_VEHICLE_FIELDS
    optional_vehicle_schema_warning_count = 0
    for frame in frames:
        for vehicle in frame.get("vehicles") or []:
            missing = required_vehicle_fields - set(vehicle)
            if missing:
                errors.append(f"vehicle record missing fields: {sorted(missing)}")
                break
            if background_only_profile:
                optional_missing = REQUIRED_VEHICLE_FIELDS - set(vehicle) - missing
                if optional_missing and optional_vehicle_schema_warning_count < 10:
                    warnings.append(f"vehicle record missing non-gating fields: {sorted(optional_missing)}")
                    optional_vehicle_schema_warning_count += 1
            if vehicle.get("source") != "sumo_traci":
                errors.append(f"vehicle {vehicle.get('vehicle_id')} has non-SUMO source {vehicle.get('source')}")
                break
            if len(vehicle.get("sumo_xy_m") or []) != 2 or len(vehicle.get("truth_position_enu_m") or []) != 3:
                errors.append(f"vehicle {vehicle.get('vehicle_id')} has invalid coordinate payload")
                break
            if vehicle.get("sumo_position_reference") != "front_bumper_center":
                message = (
                    f"vehicle {vehicle.get('vehicle_id')} has invalid sumo_position_reference "
                    f"{vehicle.get('sumo_position_reference')}"
                )
                if background_only_profile:
                    warnings.append(message)
                    continue
                errors.append(message)
                break
            if vehicle.get("truth_position_reference") not in {
                "vehicle_body_center",
                "vehicle_body_center_on_sumo_lane_shape",
                "vehicle_body_center_from_front_bumper",
            }:
                message = (
                    f"vehicle {vehicle.get('vehicle_id')} has invalid truth_position_reference "
                    f"{vehicle.get('truth_position_reference')}"
                )
                if background_only_profile:
                    warnings.append(message)
                    continue
                errors.append(message)
                break
        if errors:
            break

    roi_background = (
        _roi_background_summary(
            frames,
            manifest=manifest,
            explicit_plan=loaded_explicit_plan,
            capture_start_s=capture_start_s,
            capture_duration_s=capture_duration_s,
        )
        if loaded_explicit_plan
        else {}
    )
    semantic_lane_direction = _semantic_lane_direction_summary(loaded_explicit_plan) if loaded_explicit_plan else {}
    if max_vehicles > 0 and vehicle_counts and max(vehicle_counts) > max_vehicles:
        message = f"vehicle count exceeded max_vehicles={max_vehicles}: observed={max(vehicle_counts)}"
        if background_only_profile:
            warnings.append(message)
        else:
            errors.append(message)
    capture_frames = [
        frame
        for frame in frames
        if capture_start_s - 1e-9 <= float(frame.get("sim_time_s") or 0.0) <= capture_start_s + capture_duration_s + 1e-9
    ]
    try:
        planned_source_presence_required_ids = validate_source_presence_manifest_contract(
            loaded_explicit_plan,
            manifest,
        )
    except RuntimeError as exc:
        planned_source_presence_required_ids = set()
        errors.append(str(exc))
    source_presence_missing_capture_frames: dict[str, int] = {}
    for vehicle_id in sorted(planned_source_presence_required_ids):
        missing_frame_count = sum(
            1
            for frame in capture_frames
            if vehicle_id
            not in {
                str(vehicle.get("vehicle_id") or "")
                for vehicle in frame.get("vehicles") or []
            }
        )
        if missing_frame_count:
            source_presence_missing_capture_frames[vehicle_id] = missing_frame_count
    if source_presence_missing_capture_frames:
        errors.append(
            "source-required vehicles are missing from capture frames: "
            f"{source_presence_missing_capture_frames}"
        )
    capture_vehicle_counts = [len(frame.get("vehicles") or []) for frame in capture_frames]
    capture_start_frame = min(
        frames,
        key=lambda frame: abs(float(frame.get("sim_time_s") or 0.0) - capture_start_s),
    ) if frames else None
    capture_start_vehicle_count = len(capture_start_frame.get("vehicles") or []) if capture_start_frame else 0
    if explicit_mode:
        accepted_target = int(explicit_activation.get("accepted_background_vehicle_target") or 0)
        accepted_count = int(explicit_activation.get("accepted_background_vehicle_count") or 0)
        if accepted_target > 0 and accepted_count < accepted_target:
            warnings.append(
                f"accepted explicit background vehicles below non-gating target: {accepted_count} < {accepted_target}"
            )
        if capture_vehicle_counts and min(capture_vehicle_counts) <= 0:
            errors.append("explicit capture window has an empty vehicle frame")
        failing_roi_slots = [int(slot) for slot in roi_background.get("failing_nonprotected_slots") or []]
        roi_background_tick_count = int(roi_background.get("roi_background_frame_count") or 0)
        if background_only_profile:
            if roi_background_tick_count <= MIN_BACKGROUND_ONLY_ROI_BACKGROUND_TICKS:
                if background_only_roi_tick_gate_exempted:
                    warnings.append(
                        "background-only ROI tick gate exempted by "
                        f"event_space_class={NONTRAFFIC_MAIN_ROAD_EVENT_SPACE_CLASS}: "
                        f"ROI has background vehicles in {roi_background_tick_count} formal ticks"
                    )
                else:
                    errors.append(
                        "background-only ROI traffic too sparse: "
                        f"ROI has background vehicles in {roi_background_tick_count} formal ticks, "
                        f"must be > {MIN_BACKGROUND_ONLY_ROI_BACKGROUND_TICKS}"
                    )
            if failing_roi_slots:
                warnings.append(
                    "background-only slot coverage diagnostic only: "
                    f"slots {failing_roi_slots} missed the former per-slot matching gate"
                )
        elif failing_roi_slots:
            errors.append(
                "ROI background coverage missing in non-protected slots: "
                f"{failing_roi_slots}; extended-only traffic is not a pass"
            )
        protected_roi_frames = int(roi_background.get("protected_roi_background_frame_count") or 0)
        if has_semantic_vehicle_case and protected_roi_frames > 0:
            errors.append(
                f"background vehicles entered ROI during protected windows: {protected_roi_frames} frame-observations"
            )
        semantic_leak_frames = int(roi_background.get("semantic_outside_control_window_frame_count") or 0)
        if has_semantic_vehicle_case and semantic_leak_frames > 0:
            errors.append(
                "semantic vehicles visible outside their control windows: "
                f"{semantic_leak_frames} frame-observations"
            )
        semantic_direction_failures = [str(item) for item in semantic_lane_direction.get("failures") or []]
        if has_semantic_vehicle_case and semantic_direction_failures:
            errors.append(
                "semantic vehicles moved opposite SUMO lane direction: "
                f"{semantic_direction_failures[:10]}"
            )
    else:
        if max_vehicles > 0 and capture_start_vehicle_count < int(round(max_vehicles * MIN_CAPTURE_START_VEHICLE_RATIO)):
            errors.append(
                f"capture start has too few vehicles after warm-up: "
                f"{capture_start_vehicle_count} < {MIN_CAPTURE_START_VEHICLE_RATIO:.0%} of max_vehicles={max_vehicles}"
            )
        if max_vehicles > 0 and capture_vehicle_counts and min(capture_vehicle_counts) < int(round(max_vehicles * MIN_CAPTURE_WINDOW_VEHICLE_RATIO)):
            errors.append(
                f"capture window vehicle count dipped too low: "
                f"min={min(capture_vehicle_counts)} < {MIN_CAPTURE_WINDOW_VEHICLE_RATIO:.0%} of max_vehicles={max_vehicles}"
            )

    semantic_plan = dict(manifest.get("semantic_vehicle_plans") or {})
    expected_semantic_source_count = int(semantic_plan.get("source_entity_count") or 0)
    semantic_capture_start_ids = sorted(
        str(vehicle.get("source_entity_id") or "")
        for vehicle in (capture_start_frame.get("vehicles") if capture_start_frame else []) or []
        if vehicle.get("semantic_vehicle") or str(vehicle.get("vehicle_id") or "") in semantic_plan_vehicle_ids
    )
    semantic_capture_start_ids = [value for value in semantic_capture_start_ids if value]
    if explicit_mode and semantic_capture_start_ids:
        warnings.append(
            "explicit capture tick 0 contains semantic source vehicles. "
            "Semantic actors should normally be absent until their event/control window unless the event requires otherwise."
        )
    elif not explicit_mode and expected_semantic_source_count > 0 and not semantic_capture_start_ids:
        errors.append(
            "capture tick 0 has no semantic source vehicles after warm-up; "
            f"expected at least one source entity from {expected_semantic_source_count} planned semantic entities"
        )
    elif not explicit_mode and expected_semantic_source_count > 0 and len(set(semantic_capture_start_ids)) < expected_semantic_source_count:
        warnings.append(
            f"capture tick 0 has {len(set(semantic_capture_start_ids))}/{expected_semantic_source_count} "
            f"planned semantic source entities present: {sorted(set(semantic_capture_start_ids))}. "
            "This is not a vehicle-flow failure by itself; event realization checks should evaluate the exact "
            "event target at its activation/evidence ticks."
        )

    incidents = list(plan.get("incidents") or [])
    classes = {str(item.get("accident_class")) for item in incidents}
    active_ids = {str(item.get("incident_id")) for frame in frames for item in frame.get("active_incidents") or []}
    if background_only_profile:
        if incidents:
            warnings.append(
                f"background-only GIF gate skipped incident activation checks for {len(incidents)} incident(s)"
            )
    else:
        incident_ids = {str(item.get("incident_id")) for item in incidents}
        missing_incidents = sorted(incident_ids - active_ids)
        if missing_incidents:
            errors.append(f"incidents never became active in sampled frames: {missing_incidents[:10]}")

        active_classes = {str(item.get("accident_class")) for frame in frames for item in frame.get("active_incidents") or []}
        missing_classes = sorted(classes - active_classes)
        if missing_classes:
            errors.append(f"accident classes never became active: {missing_classes}")

    # An all-red intervention must be checked even in a background-only episode.
    all_red_incidents = [item for item in incidents if item.get("accident_class") == "traffic_light_all_red_fault"]
    for incident in all_red_incidents:
        traffic_light = dict(incident.get("traffic_light") or {})
        tls_id = str(traffic_light.get("traffic_light_id") or "")
        if not tls_id:
            errors.append(f"all-red incident lacks traffic light id: {incident.get('incident_id')}")
            continue
        observed = False
        for frame in frames:
            sim_time_s = float(frame.get("sim_time_s") or 0.0)
            in_window = (float(incident["start_s"]) <= sim_time_s and
                         sim_time_s + 1e-9 < float(incident["end_s"]))
            active_rows = [item for item in frame.get("active_incidents") or []
                           if item.get("incident_id") == incident.get("incident_id")]
            if not in_window:
                if active_rows:
                    errors.append(f"all-red incident remains active outside its window: {incident.get('incident_id')} at {sim_time_s}")
                continue
            observed = True
            if len(active_rows) != 1 or any(
                item.get("traffic_light_id") != tls_id or
                item.get("accident_class") != incident["accident_class"] or
                item.get("start_s") != incident["start_s"] or
                item.get("end_s") != incident["end_s"]
                for item in active_rows
            ):
                errors.append(f"all-red incident row differs from plan: {incident.get('incident_id')} at {sim_time_s}")
            states = [item.get("state") for item in frame.get("traffic_lights") or []
                      if item.get("tls_id") == tls_id]
            if len(states) != 1 or not isinstance(states[0], str) or set(states[0].lower()) != {"r"}:
                errors.append(f"all-red controller effect is absent: {incident.get('incident_id')} at {sim_time_s}")
        if not observed:
            errors.append(f"all-red incident has no sampled active frame: {incident.get('incident_id')}")

    if background_only_profile:
        multilane = {
            "policy": "episode_local_sumo_multilane_distribution_audit_v1",
            "skipped": True,
            "skip_reason": "background_only_gif_gate_non_gating_warning",
            "groups_checked": 0,
            "groups": [],
            "failures": [],
            "warnings": [],
        }
    else:
        multilane = _multilane_summary(frames)
    if multilane["failures"]:
        warnings.extend(f"multi-lane utilization episode-local underuse: {item}" for item in multilane["failures"][:10])
    if multilane["warnings"]:
        warnings.extend(f"multi-lane utilization warning: {item}" for item in multilane["warnings"][:10])

    records = _error_records(errors)
    summary = {
        "ok": not errors,
        "validation_profile": validation_profile,
        "errors": errors,
        "error_records": records,
        "error_counts_by_category": _error_counts(records),
        "warnings": warnings,
        "frame_count": len(frames),
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "capture_start_s": capture_start_s,
        "capture_duration_s": capture_duration_s,
        "sample_period_s": sample_period_s,
        "traffic_light_count_min": min(traffic_light_counts) if traffic_light_counts else 0,
        "traffic_light_count_max": max(traffic_light_counts) if traffic_light_counts else 0,
        "vehicle_count_min": min(vehicle_counts) if vehicle_counts else 0,
        "vehicle_count_max": max(vehicle_counts) if vehicle_counts else 0,
        "vehicle_count_final": vehicle_counts[-1] if vehicle_counts else 0,
        "capture_start_vehicle_count": capture_start_vehicle_count,
        "capture_window_vehicle_count_min": min(capture_vehicle_counts) if capture_vehicle_counts else 0,
        "capture_window_vehicle_count_max": max(capture_vehicle_counts) if capture_vehicle_counts else 0,
        "accepted_background_vehicle_target": int(explicit_activation.get("accepted_background_vehicle_target") or 0),
        "accepted_background_vehicle_count": int(explicit_activation.get("accepted_background_vehicle_count") or 0),
        "filtered_background_vehicle_count": int(dict(manifest.get("explicit_vehicle_route_filter") or {}).get("filtered_background_vehicle_count") or 0),
        "inactive_background_vehicle_count": int(explicit_activation.get("inactive_background_vehicle_count") or 0),
        "traffic_case": traffic_case,
        "scene_event_space_class": scene_event_space_class,
        "background_only_roi_tick_gate_exempted": background_only_roi_tick_gate_exempted,
        "traffic_tuning": dict(dict(manifest.get("explicit_vehicle_plan") or {}).get("traffic_tuning") or {}),
        "roi_background_coverage": roi_background,
        "semantic_lane_direction": semantic_lane_direction,
        "semantic_capture_start_source_entities": sorted(set(semantic_capture_start_ids)),
        "source_presence_required_vehicle_count": len(planned_source_presence_required_ids),
        "source_presence_capture_frame_count": len(capture_frames),
        "source_presence_missing_capture_frames": source_presence_missing_capture_frames,
        "visualization_gate": visualization_gate,
        "incident_count": len(incidents),
        "active_incident_count": len(active_ids),
        "accident_classes": sorted(classes),
        "multi_lane_utilization": multilane,
    }
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SUMO traffic output files.")
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_output(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
