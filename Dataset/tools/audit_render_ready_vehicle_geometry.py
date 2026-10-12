from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from render_ready_vehicle_lanes import VehicleLaneProjector
except ModuleNotFoundError:  # pragma: no cover - supports package imports.
    from Dataset.tools.render_ready_vehicle_lanes import VehicleLaneProjector

try:
    from sumo_ground_flow.validate_road_derived_network import (
        DEFAULT_MANIFEST as DEFAULT_ROAD_DERIVED_MANIFEST,
        DEFAULT_TRAFFIC_BUNDLE_DIR,
        ROAD_SURFACE_NUMERIC_EPSILON_M,
        _load_road_surface,
        _read_json,
    )
except ModuleNotFoundError:  # pragma: no cover - supports package imports.
    from Dataset.tools.sumo_ground_flow.validate_road_derived_network import (
        DEFAULT_MANIFEST as DEFAULT_ROAD_DERIVED_MANIFEST,
        DEFAULT_TRAFFIC_BUNDLE_DIR,
        ROAD_SURFACE_NUMERIC_EPSILON_M,
        _load_road_surface,
        _read_json,
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RENDER_READY_ROOT = ROOT / "Dataset" / "render_ready_episodes"
DEFAULT_MAP_ID = "donghu_road_topo"
STRICT_SOURCE_VEHICLE_POLICY = "sumo_only_vehicle_authority_replaces_source_vehicle_truth_v2"
MAX_MOVING_LANE_DISTANCE_M = 5.0
MAX_DIRECTION_SAMPLE_DISTANCE_M = 2.25
MIN_FORWARD_COS = -0.2
MIN_BACKGROUND_FORWARD_COS = 0.45
MIN_BACKGROUND_MOTION_COS = 0.35
MIN_MOVING_SPEED_FOR_KINEMATIC_AUDIT_MPS = 1.0
MAX_BACKGROUND_ASSIGNED_LANE_DISTANCE_M = 1.35
MAX_BACKGROUND_SAME_LANE_YAW_DELTA_DEG = 65.0
MAX_BACKGROUND_STATIONARY_RUN = 3
OVERLAP_BROADPHASE_DISTANCE_M = 6.0
ROAD_SURFACE_TOLERANCE_M = 0.75


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc


def _position(entity: dict[str, Any]) -> list[float] | None:
    pose = dict(entity.get("truth_pose") or {})
    value = pose.get("position_enu_m")
    if not isinstance(value, list) or len(value) < 2:
        return None
    return [float(value[0]), float(value[1]), float(value[2] if len(value) > 2 else 0.0)]


def _velocity(entity: dict[str, Any]) -> list[float]:
    pose = dict(entity.get("truth_pose") or {})
    value = pose.get("velocity_enu_mps")
    if not isinstance(value, list) or len(value) < 2:
        return [0.0, 0.0, 0.0]
    return [float(value[0]), float(value[1]), float(value[2] if len(value) > 2 else 0.0)]


def _yaw_deg(entity: dict[str, Any]) -> float:
    pose = dict(entity.get("truth_pose") or {})
    rotation = dict(pose.get("rotation_deg") or {})
    try:
        return float(rotation.get("yaw_deg") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _angle_delta_deg(a_deg: float, b_deg: float) -> float:
    return abs((float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0)


def _cos_delta(a_deg: float, b_deg: float) -> float:
    return math.cos(math.radians(float(a_deg) - float(b_deg)))


def _is_background_sumo_vehicle(entity: dict[str, Any]) -> bool:
    if str(entity.get("source") or "") != "sumo_traci":
        return False
    tags = {str(tag) for tag in (entity.get("tags") or [])}
    background_vehicle = dict(entity.get("background_vehicle") or {})
    sumo_vehicle = dict(entity.get("sumo_vehicle") or {})
    return (
        str(entity.get("background_role") or "") == "sumo_background_traffic"
        or str(background_vehicle.get("control_role") or "") == "background"
        or str(sumo_vehicle.get("control_role") or "") == "background"
        or "background_traffic" in tags
    )


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _dimensions_m(entity: dict[str, Any]) -> tuple[float, float]:
    sumo_vehicle = dict(entity.get("sumo_vehicle") or {})
    dimensions = dict(sumo_vehicle.get("dimensions_m") or {})
    try:
        length = float(dimensions.get("length") or 5.0)
        width = float(dimensions.get("width") or 1.8)
    except (TypeError, ValueError):
        return 5.0, 1.8
    return max(0.1, length), max(0.1, width)


def _oriented_box_corners(position: list[float], yaw_deg: float, length_m: float, width_m: float) -> list[tuple[float, float]]:
    yaw_rad = math.radians(float(yaw_deg))
    ux = (math.cos(yaw_rad), math.sin(yaw_rad))
    uy = (-math.sin(yaw_rad), math.cos(yaw_rad))
    half_length = 0.5 * float(length_m)
    half_width = 0.5 * float(width_m)
    corners: list[tuple[float, float]] = []
    for along in (-half_length, half_length):
        for lateral in (-half_width, half_width):
            corners.append(
                (
                    float(position[0]) + along * ux[0] + lateral * uy[0],
                    float(position[1]) + along * ux[1] + lateral * uy[1],
                )
            )
    return corners


def _projection(points: list[tuple[float, float]], axis: tuple[float, float]) -> tuple[float, float]:
    values = [point[0] * axis[0] + point[1] * axis[1] for point in points]
    return min(values), max(values)


def _overlap_on_axis(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _oriented_boxes_overlap(
    a_position: list[float],
    a_yaw_deg: float,
    a_length_m: float,
    a_width_m: float,
    b_position: list[float],
    b_yaw_deg: float,
    b_length_m: float,
    b_width_m: float,
) -> bool:
    a_corners = _oriented_box_corners(a_position, a_yaw_deg, a_length_m, a_width_m)
    b_corners = _oriented_box_corners(b_position, b_yaw_deg, b_length_m, b_width_m)
    axes = []
    for yaw in (a_yaw_deg, b_yaw_deg):
        yaw_rad = math.radians(float(yaw))
        axes.append((math.cos(yaw_rad), math.sin(yaw_rad)))
        axes.append((-math.sin(yaw_rad), math.cos(yaw_rad)))
    return all(_overlap_on_axis(_projection(a_corners, axis), _projection(b_corners, axis)) for axis in axes)


def load_default_road_surface() -> tuple[Any | None, Any | None]:
    try:
        manifest = _read_json(DEFAULT_ROAD_DERIVED_MANIFEST)
        net_xml_value = dict(manifest.get("outputs") or {}).get("net_xml")
        net_xml = Path(net_xml_value) if net_xml_value else None
        return _load_road_surface(DEFAULT_TRAFFIC_BUNDLE_DIR, ROAD_SURFACE_TOLERANCE_M, net_xml)
    except Exception:
        return None, None


def audit_episode(
    episode_dir: Path,
    projector: VehicleLaneProjector,
    *,
    road_surface: Any | None = None,
    prepared_road_surface: Any | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = episode_dir / "episode_manifest.json"
    truth_path = episode_dir / "truth_frames.jsonl"
    roster_path = episode_dir / "global_entity_roster.json"
    if not manifest_path.exists() or not truth_path.exists() or not roster_path.exists():
        return {"episode_id": episode_dir.name, "ok": False, "errors": ["missing render-ready artifacts"], "warnings": []}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_vehicle_authority = dict(manifest.get("source_vehicle_authority") or {})
    strict_sumo_authority = (
        source_vehicle_authority.get("enabled") is True
        and source_vehicle_authority.get("policy") == STRICT_SOURCE_VEHICLE_POLICY
    )
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    for entry in roster.get("entities") or []:
        if str(entry.get("entity_category") or entry.get("label_class") or "").lower() != "vehicle":
            continue
        if strict_sumo_authority and str(entry.get("source") or "") != "sumo_traci":
            errors.append(f"roster non-SUMO vehicle under SUMO authority: {entry.get('entity_id')}")

    frame_count = 0
    vehicle_frame_count = 0
    max_lane_distance_m = 0.0
    max_assigned_lane_distance_m = 0.0
    opposite_count = 0
    overlap_count = 0
    background_lane_center_count = 0
    background_kinematic_count = 0
    overlap_examples: list[str] = []
    opposite_examples: list[str] = []
    distance_examples: list[str] = []
    background_lane_examples: list[str] = []
    background_kinematic_examples: list[str] = []
    previous_by_entity: dict[str, dict[str, Any]] = {}
    stationary_run_by_entity: dict[str, int] = defaultdict(int)
    for frame in iter_jsonl(truth_path):
        frame_count += 1
        tick = int(frame.get("tick") or 0)
        frame_vehicles: list[tuple[str, list[float], float, float, float]] = []
        for entity in frame.get("entities") or []:
            if str(entity.get("entity_category") or entity.get("label_class") or "").lower() != "vehicle":
                continue
            entity_id = str(entity.get("entity_id") or "")
            if strict_sumo_authority and str(entity.get("source") or "") != "sumo_traci":
                errors.append(f"tick {tick}: non-SUMO vehicle rendered under SUMO authority: {entity_id}")
                continue
            position = _position(entity)
            if position is None:
                errors.append(f"tick {tick}: vehicle {entity_id} missing truth position")
                continue
            if prepared_road_surface is not None and road_surface is not None:
                try:
                    from shapely.geometry import Point

                    point = Point(float(position[0]), float(position[1]))
                    if not prepared_road_surface.covers(point) and road_surface.distance(point) > ROAD_SURFACE_NUMERIC_EPSILON_M:
                        errors.append(f"tick {tick}: vehicle {entity_id} is outside road-derived road surface")
                except Exception:
                    pass
            vehicle_frame_count += 1
            length_m, width_m = _dimensions_m(entity)
            entity_yaw_deg = _yaw_deg(entity)
            frame_vehicles.append((entity_id, position, entity_yaw_deg, length_m, width_m))
            sample = projector._nearest_sample(position, yaw_deg=None)
            lane_distance_m = math.hypot(position[0] - sample.x_m, position[1] - sample.y_m)
            max_lane_distance_m = max(max_lane_distance_m, lane_distance_m)
            if lane_distance_m > MAX_MOVING_LANE_DISTANCE_M:
                if len(distance_examples) < 12:
                    distance_examples.append(f"{entity_id}@tick{tick}:{lane_distance_m:.3f}m")
                errors.append(f"tick {tick}: vehicle {entity_id} is {lane_distance_m:.3f}m from nearest road-derived lane")
            velocity = _velocity(entity)
            speed = math.hypot(velocity[0], velocity[1])
            sumo_vehicle = dict(entity.get("sumo_vehicle") or {})
            edge_id = str(sumo_vehicle.get("sumo_edge_id") or "")
            lane_id = str(sumo_vehicle.get("sumo_lane_id") or "")
            center_lane_position = _float_or_none(
                sumo_vehicle.get("center_lane_position_m", sumo_vehicle.get("lane_position_m"))
            )
            is_internal_edge = edge_id.startswith(":")
            is_background_sumo = _is_background_sumo_vehicle(entity)
            if is_background_sumo and not is_internal_edge and center_lane_position is not None and center_lane_position >= 0.0:
                assigned_sample = projector._sample_for_vehicle(
                    position,
                    edge_id=edge_id,
                    lane_id=lane_id,
                    lane_position_m=center_lane_position,
                    yaw_deg=None,
                )
                assigned_lane_distance_m = math.hypot(position[0] - assigned_sample.x_m, position[1] - assigned_sample.y_m)
                max_assigned_lane_distance_m = max(max_assigned_lane_distance_m, assigned_lane_distance_m)
                if speed >= 0.5 and assigned_lane_distance_m > MAX_BACKGROUND_ASSIGNED_LANE_DISTANCE_M:
                    background_lane_center_count += 1
                    if len(background_lane_examples) < 12:
                        background_lane_examples.append(
                            f"{entity_id}@tick{tick}:{assigned_lane_distance_m:.3f}m lane={lane_id}"
                        )
                    errors.append(
                        f"tick {tick}: background vehicle {entity_id} is {assigned_lane_distance_m:.3f}m from assigned lane center {lane_id}"
                    )
            if speed >= 0.5:
                forward_cos_threshold = MIN_BACKGROUND_FORWARD_COS if is_background_sumo else MIN_FORWARD_COS
                if edge_id.startswith(":"):
                    velocity_yaw_deg = math.degrees(math.atan2(float(velocity[1]), float(velocity[0])))
                    alignment = _cos_delta(entity_yaw_deg, velocity_yaw_deg)
                    if alignment < forward_cos_threshold:
                        opposite_count += 1
                        if len(opposite_examples) < 12:
                            opposite_examples.append(f"{entity_id}@tick{tick}:internal_cos={alignment:.3f}")
                        errors.append(
                            f"tick {tick}: vehicle {entity_id} moves against SUMO internal connector velocity cos={alignment:.3f}"
                        )
                    continue
                if center_lane_position is not None and center_lane_position < 0.0:
                    velocity_yaw_deg = math.degrees(math.atan2(float(velocity[1]), float(velocity[0])))
                    alignment = _cos_delta(entity_yaw_deg, velocity_yaw_deg)
                    if alignment < forward_cos_threshold:
                        opposite_count += 1
                        if len(opposite_examples) < 12:
                            opposite_examples.append(f"{entity_id}@tick{tick}:edge_entry_cos={alignment:.3f}")
                        errors.append(
                            f"tick {tick}: vehicle {entity_id} moves against entry velocity cos={alignment:.3f}"
                        )
                    continue
                direction_sample = projector._sample_for_vehicle(
                    position,
                    edge_id=edge_id,
                    lane_id=lane_id,
                    lane_position_m=center_lane_position,
                    yaw_deg=None,
                )
                direction_distance_m = math.hypot(position[0] - direction_sample.x_m, position[1] - direction_sample.y_m)
                if direction_distance_m > MAX_DIRECTION_SAMPLE_DISTANCE_M:
                    direction_sample = sample
                    direction_yaw_candidates = [direction_sample.yaw_deg]
                else:
                    direction_yaw_candidates = projector.direction_yaw_candidates_for_vehicle(
                        position,
                        edge_id=edge_id,
                        lane_id=lane_id,
                        lane_position_m=center_lane_position,
                        yaw_deg=None,
                    )
                alignment = max(_cos_delta(entity_yaw_deg, candidate_yaw) for candidate_yaw in direction_yaw_candidates)
                if alignment < forward_cos_threshold:
                    opposite_count += 1
                    if len(opposite_examples) < 12:
                        opposite_examples.append(f"{entity_id}@tick{tick}:cos={alignment:.3f}")
                    errors.append(f"tick {tick}: vehicle {entity_id} moves against road-derived lane direction cos={alignment:.3f}")
            previous = previous_by_entity.get(entity_id)
            if is_background_sumo and previous is not None:
                previous_tick = int(previous["tick"])
                tick_delta = tick - previous_tick
                if 0 < tick_delta <= 10:
                    previous_position = previous["position"]
                    dx = position[0] - previous_position[0]
                    dy = position[1] - previous_position[1]
                    displacement_m = math.hypot(dx, dy)
                    dt_s = tick_delta / max(1.0, float(frame.get("tick_hz") or 10.0))
                    expected_m = max(speed, float(previous.get("speed") or 0.0)) * dt_s
                    if (
                        speed >= MIN_MOVING_SPEED_FOR_KINEMATIC_AUDIT_MPS
                        and expected_m >= 0.20
                        and displacement_m < max(0.03, expected_m * 0.15)
                    ):
                        stationary_run_by_entity[entity_id] += 1
                        if stationary_run_by_entity[entity_id] >= MAX_BACKGROUND_STATIONARY_RUN:
                            background_kinematic_count += 1
                            if len(background_kinematic_examples) < 12:
                                background_kinematic_examples.append(
                                    f"{entity_id}@tick{tick}:frozen speed={speed:.2f} disp={displacement_m:.3f}"
                                )
                            errors.append(
                                f"tick {tick}: background vehicle {entity_id} is visually stationary while speed={speed:.3f}mps"
                            )
                    else:
                        stationary_run_by_entity[entity_id] = 0
                    if displacement_m >= 0.12 and speed >= MIN_MOVING_SPEED_FOR_KINEMATIC_AUDIT_MPS:
                        displacement_yaw_deg = math.degrees(math.atan2(dy, dx))
                        yaw_motion_alignment = _cos_delta(entity_yaw_deg, displacement_yaw_deg)
                        if yaw_motion_alignment < MIN_BACKGROUND_MOTION_COS:
                            background_kinematic_count += 1
                            if len(background_kinematic_examples) < 12:
                                background_kinematic_examples.append(
                                    f"{entity_id}@tick{tick}:yaw_vs_disp={yaw_motion_alignment:.3f}"
                                )
                            errors.append(
                                f"tick {tick}: background vehicle {entity_id} visual yaw does not follow displacement cos={yaw_motion_alignment:.3f}"
                            )
                        velocity_yaw_deg = math.degrees(math.atan2(float(velocity[1]), float(velocity[0])))
                        velocity_motion_alignment = _cos_delta(velocity_yaw_deg, displacement_yaw_deg)
                        if velocity_motion_alignment < MIN_BACKGROUND_MOTION_COS:
                            background_kinematic_count += 1
                            if len(background_kinematic_examples) < 12:
                                background_kinematic_examples.append(
                                    f"{entity_id}@tick{tick}:velocity_vs_disp={velocity_motion_alignment:.3f}"
                                )
                            errors.append(
                                f"tick {tick}: background vehicle {entity_id} velocity does not follow displacement cos={velocity_motion_alignment:.3f}"
                            )
                    if (
                        not is_internal_edge
                        and str(previous.get("lane_id") or "") == lane_id
                        and speed >= MIN_MOVING_SPEED_FOR_KINEMATIC_AUDIT_MPS
                        and float(previous.get("speed") or 0.0) >= MIN_MOVING_SPEED_FOR_KINEMATIC_AUDIT_MPS
                    ):
                        yaw_delta_deg = _angle_delta_deg(entity_yaw_deg, float(previous.get("yaw_deg") or entity_yaw_deg))
                        if yaw_delta_deg > MAX_BACKGROUND_SAME_LANE_YAW_DELTA_DEG:
                            background_kinematic_count += 1
                            if len(background_kinematic_examples) < 12:
                                background_kinematic_examples.append(
                                    f"{entity_id}@tick{tick}:same_lane_yaw_delta={yaw_delta_deg:.1f}"
                                )
                            errors.append(
                                f"tick {tick}: background vehicle {entity_id} yaw jumps {yaw_delta_deg:.1f}deg on lane {lane_id}"
                            )
            previous_by_entity[entity_id] = {
                "tick": tick,
                "position": position,
                "yaw_deg": entity_yaw_deg,
                "speed": speed,
                "lane_id": lane_id,
            }
        for index, (a_id, a_pos, a_yaw, a_length, a_width) in enumerate(frame_vehicles):
            for b_id, b_pos, b_yaw, b_length, b_width in frame_vehicles[index + 1 :]:
                distance_m = math.hypot(a_pos[0] - b_pos[0], a_pos[1] - b_pos[1])
                if distance_m > OVERLAP_BROADPHASE_DISTANCE_M:
                    continue
                if _oriented_boxes_overlap(a_pos, a_yaw, a_length, a_width, b_pos, b_yaw, b_length, b_width):
                    overlap_count += 1
                    if len(overlap_examples) < 12:
                        overlap_examples.append(f"{a_id}/{b_id}@tick{tick}:{distance_m:.3f}m")
                    errors.append(f"tick {tick}: vehicle boxes overlap {a_id}/{b_id} center_distance={distance_m:.3f}m")

    return {
        "episode_id": episode_dir.name,
        "ok": not errors,
        "errors": errors[:200],
        "error_count": len(errors),
        "warnings": warnings,
        "summary": {
            "frame_count": frame_count,
            "vehicle_frame_count": vehicle_frame_count,
            "max_lane_distance_m": round(max_lane_distance_m, 6),
            "max_assigned_lane_distance_m": round(max_assigned_lane_distance_m, 6),
            "opposite_direction_count": opposite_count,
            "overlap_count": overlap_count,
            "background_lane_center_count": background_lane_center_count,
            "background_kinematic_count": background_kinematic_count,
            "distance_examples": distance_examples,
            "opposite_examples": opposite_examples,
            "overlap_examples": overlap_examples,
            "background_lane_examples": background_lane_examples,
            "background_kinematic_examples": background_kinematic_examples,
        },
    }


def audit_root(render_ready_root: Path, map_id: str) -> dict[str, Any]:
    projector = VehicleLaneProjector(ROOT, map_id)
    road_surface, prepared_road_surface = load_default_road_surface()
    episodes = sorted(path for path in render_ready_root.iterdir() if path.is_dir())
    results = [
        audit_episode(path, projector, road_surface=road_surface, prepared_road_surface=prepared_road_surface)
        for path in episodes
    ]
    errors: list[str] = []
    for result in results:
        if not result["ok"]:
            for error in result.get("errors") or []:
                errors.append(f"{result['episode_id']}: {error}")
    return {
        "ok": not errors,
        "episode_count": len(results),
        "errors": errors[:500],
        "error_count": len(errors),
        "episodes": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit render-ready vehicle geometry against road-derived lanes.")
    parser.add_argument("--render-ready-root", type=Path, default=DEFAULT_RENDER_READY_ROOT)
    parser.add_argument("--map-id", default=DEFAULT_MAP_ID)
    parser.add_argument("--episode", type=Path)
    args = parser.parse_args()
    projector = VehicleLaneProjector(ROOT, args.map_id)
    road_surface, prepared_road_surface = load_default_road_surface()
    if args.episode:
        result = audit_episode(
            args.episode.resolve(),
            projector,
            road_surface=road_surface,
            prepared_road_surface=prepared_road_surface,
        )
    else:
        result = audit_root(args.render_ready_root.resolve(), args.map_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
