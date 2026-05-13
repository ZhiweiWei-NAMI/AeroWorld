"""Generate Dataset/episodes from grounded scene_setup/event_script pairs.

The render pipeline consumes Dataset/episodes, so this generator must keep those
episodes aligned with the scenario files that validation checks. It uses
scene_setup.json as the authoritative entity/asset/initial-pose source and
event_script.json as the event/motion source.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from pedestrian_activity_catalog import get_activity, normalize_activity_type


TICK_HZ = 10
DEFAULT_DURATION_TICKS = 900
AMBIENT_GROUND_MOTION_MIN_PREFIX_TICKS = 50
AMBIENT_GROUND_MOTION_SPAN_M = {
    "pedestrian": 10.5,
    "vehicle": 26.0,
}
AMBIENT_GROUND_MOTION_SPEED_MPS = {
    "pedestrian": 1.25,
    "vehicle": 5.0,
}
INSPECT_UAV_LOOP_SPEED_MPS = 5.0
INSPECT_UAV_MIN_MOTION_RATIO = 1.05
GROUND_MOTION_SPEED_EPS_MPS = 0.05
WEATHER_PROFILES: dict[str, dict[str, Any]] = {
    "clear": {"condition": "clear", "rain": 0.0, "fog": 0.0, "fog_density": 0.0, "wind_speed": 2.0, "visibility_m": 20000.0, "visibility": 20000.0},
    "rain": {"condition": "rain", "rain": 0.55, "fog": 0.0, "fog_density": 0.0, "wind_speed": 4.0, "visibility_m": 2200.0, "visibility": 2200.0, "wetness": 0.75},
    "fog": {"condition": "fog", "rain": 0.0, "fog": 0.6, "fog_density": 0.6, "wind_speed": 2.0, "visibility_m": 650.0, "visibility": 650.0},
    "wind": {"condition": "wind", "rain": 0.0, "fog": 0.0, "fog_density": 0.0, "wind_speed": 12.5, "visibility_m": 20000.0, "visibility": 20000.0},
    "dusk": {"condition": "dusk", "rain": 0.0, "fog": 0.0, "fog_density": 0.0, "wind_speed": 2.0, "visibility_m": 12000.0, "visibility": 12000.0},
    "heat": {"condition": "heat", "rain": 0.0, "fog": 0.0, "fog_density": 0.0, "wind_speed": 2.0, "visibility_m": 18000.0, "visibility": 18000.0},
    "light smoke": {"condition": "light smoke", "rain": 0.0, "fog": 0.18, "fog_density": 0.18, "wind_speed": 2.0, "visibility_m": 3500.0, "visibility": 3500.0, "dust": 0.25},
}
PRESERVED_ENTITY_FIELDS = (
    "task_id",
    "role",
    "state_sequence",
    "semantic_role",
    "background_role",
    "contract_scenario_id",
    "contract_inspect_uav",
    "background_vehicle",
    "background_pedestrian",
    "ground_flow_contract",
    "contract_facility",
    "contract_logical_sidecar",
    "uav_corridor_role",
    "uav_corridor",
    "assigned_altitude_m",
    "inspect_altitude_code",
    "inspect_altitude_m",
    "min_path_length_m",
    "full_episode_presence",
    "lifecycle",
    "activation_tick",
    "spawn_policy",
    "category",
    "route_waypoints_enu_m",
    "task_kind",
    "task_state",
    "inspect_altitude_code",
    "inspect_altitude_m",
    "min_path_length_m",
    "full_episode_presence",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def project_relative(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(dataset_root.parent.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def remove_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


def resolve_param(value: Any, params: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$param."):
        return params.get(value[len("$param.") :], value)
    if isinstance(value, list):
        return [resolve_param(item, params) for item in value]
    if isinstance(value, dict):
        return {key: resolve_param(item, params) for key, item in value.items()}
    return value


def vector3(value: Any, default: Sequence[float] = (0.0, 0.0, 0.0)) -> list[float]:
    values = list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []
    return [
        float(values[0] if len(values) > 0 else default[0]),
        float(values[1] if len(values) > 1 else default[1]),
        float(values[2] if len(values) > 2 else default[2]),
    ]


def distance3(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def path_length_m(points: Sequence[Sequence[float]]) -> float:
    return sum(distance3(a, b) for a, b in zip(points, points[1:]))


def extend_loop_route(
    start: list[float],
    route: list[list[float]],
    *,
    velocity_mps: float,
    duration_ticks: int,
    min_motion_ratio: float,
) -> list[list[float]]:
    base: list[list[float]] = []
    for point in [start, *route]:
        candidate = vector3(point)
        if base and distance3(base[-1], candidate) < 0.05:
            continue
        base.append(candidate)
    if len(base) < 2:
        return [vector3(point) for point in route]
    if distance3(base[0], base[-1]) >= 0.05:
        base.append(list(base[0]))
    result = [list(point) for point in base[1:]]
    cycle = [list(point) for point in base[1:]]
    current = list(result[-1])
    current_length_m = path_length_m([start, *result])
    target_length_m = float(velocity_mps) * (float(duration_ticks) / float(TICK_HZ)) * float(min_motion_ratio)
    while current_length_m < target_length_m:
        progressed = False
        for point in cycle:
            if distance3(current, point) < 0.05:
                continue
            result.append(list(point))
            current_length_m += distance3(current, point)
            current = list(point)
            progressed = True
            if current_length_m >= target_length_m:
                break
        if not progressed:
            break
    return result


def stable_unit_interval(seed_text: str) -> float:
    digest = hashlib.sha256(str(seed_text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _stable_signed_unit(seed_text: str) -> float:
    return stable_unit_interval(seed_text) * 2.0 - 1.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _explicit_variation_fields(action: dict[str, Any], params: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    keys = ("trajectory_variation", "stagger", "density_profile")
    resolved: dict[str, Any] = {}
    explicit = False
    action_params = dict(action.get("params") or {}) if isinstance(action.get("params"), dict) else {}
    for key in keys:
        if key in action:
            resolved[key] = resolve_param(action.get(key), params)
            explicit = True
            continue
        if key in action_params:
            resolved[key] = resolve_param(action_params.get(key), params)
            explicit = True
            continue
        if key in params:
            resolved[key] = resolve_param(params.get(key), params)
            explicit = True
    return explicit, resolved


def _variation_settings(label_class: str, action: dict[str, Any], params: dict[str, Any]) -> dict[str, float] | None:
    explicit, fields = _explicit_variation_fields(action, params)
    if not explicit:
        return None

    max_tick_offset_ticks = 0
    tick_bias_ticks = 0
    velocity_jitter_ratio = 0.0
    velocity_bias = 0.0
    lateral_offset_m = 0.0
    longitudinal_offset_m = 0.0
    pedestrian_lateral_offset_m: float | None = None
    vehicle_lateral_offset_m: float | None = None
    pedestrian_longitudinal_offset_m: float | None = None
    vehicle_longitudinal_offset_m: float | None = None
    min_segment_m = 0.25

    trajectory_variation = fields.get("trajectory_variation")
    if isinstance(trajectory_variation, dict):
        max_tick_offset_ticks = max(
            0,
            _to_int(
                trajectory_variation.get(
                    "max_tick_offset_ticks",
                    trajectory_variation.get("tick_offset_ticks", max_tick_offset_ticks),
                ),
                max_tick_offset_ticks,
            ),
        )
        velocity_jitter_ratio = max(
            0.0,
            _to_float(
                trajectory_variation.get(
                    "velocity_jitter_ratio",
                    trajectory_variation.get("velocity_scale_jitter", velocity_jitter_ratio),
                ),
                velocity_jitter_ratio,
            ),
        )
        velocity_bias += _to_float(trajectory_variation.get("velocity_bias", 0.0), 0.0)
        lateral_offset_m = abs(_to_float(trajectory_variation.get("lateral_offset_m", lateral_offset_m), lateral_offset_m))
        longitudinal_offset_m = abs(
            _to_float(
                trajectory_variation.get(
                    "longitudinal_offset_m",
                    trajectory_variation.get("path_offset_m", longitudinal_offset_m),
                ),
                longitudinal_offset_m,
            )
        )
        if "pedestrian_lateral_offset_m" in trajectory_variation:
            pedestrian_lateral_offset_m = abs(_to_float(trajectory_variation.get("pedestrian_lateral_offset_m"), 0.0))
        if "vehicle_lateral_offset_m" in trajectory_variation:
            vehicle_lateral_offset_m = abs(_to_float(trajectory_variation.get("vehicle_lateral_offset_m"), 0.0))
        if "pedestrian_longitudinal_offset_m" in trajectory_variation:
            pedestrian_longitudinal_offset_m = abs(_to_float(trajectory_variation.get("pedestrian_longitudinal_offset_m"), 0.0))
        if "vehicle_longitudinal_offset_m" in trajectory_variation:
            vehicle_longitudinal_offset_m = abs(_to_float(trajectory_variation.get("vehicle_longitudinal_offset_m"), 0.0))
        min_segment_m = max(0.0, _to_float(trajectory_variation.get("min_segment_m", min_segment_m), min_segment_m))
    elif trajectory_variation is True:
        max_tick_offset_ticks = max(max_tick_offset_ticks, 6)
        velocity_jitter_ratio = max(velocity_jitter_ratio, 0.10)
    elif isinstance(trajectory_variation, (int, float)):
        max_tick_offset_ticks = max(max_tick_offset_ticks, abs(_to_int(trajectory_variation, 0)))

    stagger = fields.get("stagger")
    if isinstance(stagger, dict):
        max_tick_offset_ticks = max(
            max_tick_offset_ticks,
            abs(_to_int(stagger.get("max_tick_offset_ticks", stagger.get("tick_offset_ticks", 0)), 0)),
        )
        tick_bias_ticks += _to_int(stagger.get("tick_bias_ticks", stagger.get("tick_bias", 0)), 0)
    elif isinstance(stagger, (int, float)):
        max_tick_offset_ticks = max(max_tick_offset_ticks, abs(_to_int(stagger, 0)))
    elif stagger is True:
        max_tick_offset_ticks = max(max_tick_offset_ticks, 6)

    density_profile = fields.get("density_profile")
    if isinstance(density_profile, dict):
        tick_bias_ticks += _to_int(density_profile.get("tick_bias_ticks", density_profile.get("tick_bias", 0)), 0)
        velocity_bias += _to_float(density_profile.get("velocity_bias", density_profile.get("speed_bias", 0.0)), 0.0)
        max_tick_offset_ticks = max(
            max_tick_offset_ticks,
            abs(_to_int(density_profile.get("max_tick_offset_ticks", density_profile.get("tick_offset_ticks", 0)), 0)),
        )
        velocity_jitter_ratio = max(
            velocity_jitter_ratio,
            max(
                0.0,
                _to_float(
                    density_profile.get("velocity_jitter_ratio", density_profile.get("speed_jitter_ratio", 0.0)),
                    0.0,
                ),
            ),
        )
        lateral_offset_m = max(lateral_offset_m, abs(_to_float(density_profile.get("lateral_offset_m", 0.0), 0.0)))
        longitudinal_offset_m = max(
            longitudinal_offset_m,
            abs(
                _to_float(
                    density_profile.get("longitudinal_offset_m", density_profile.get("path_offset_m", 0.0)),
                    0.0,
                )
            ),
        )
    else:
        density_text = str(density_profile or "").strip().lower()
        if density_text in {"dense", "high", "crowded"}:
            tick_bias_ticks += 2
            velocity_bias -= 0.08
            max_tick_offset_ticks = max(max_tick_offset_ticks, 8)
            velocity_jitter_ratio = max(velocity_jitter_ratio, 0.04)
            lateral_offset_m = max(lateral_offset_m, 0.12)
            longitudinal_offset_m = max(longitudinal_offset_m, 0.25)
        elif density_text in {"sparse", "low", "light"}:
            tick_bias_ticks -= 2
            velocity_bias += 0.08
            max_tick_offset_ticks = max(max_tick_offset_ticks, 12)
            velocity_jitter_ratio = max(velocity_jitter_ratio, 0.06)
            lateral_offset_m = max(lateral_offset_m, 0.18)
            longitudinal_offset_m = max(longitudinal_offset_m, 0.45)
        elif density_text in {"medium", "normal", "balanced"}:
            max_tick_offset_ticks = max(max_tick_offset_ticks, 6)
            velocity_jitter_ratio = max(velocity_jitter_ratio, 0.03)
            lateral_offset_m = max(lateral_offset_m, 0.10)
            longitudinal_offset_m = max(longitudinal_offset_m, 0.20)

    if label_class == "pedestrian":
        max_lateral_offset_m = abs(pedestrian_lateral_offset_m if pedestrian_lateral_offset_m is not None else lateral_offset_m)
        max_longitudinal_offset_m = abs(
            pedestrian_longitudinal_offset_m if pedestrian_longitudinal_offset_m is not None else longitudinal_offset_m
        )
    elif label_class == "vehicle":
        if vehicle_lateral_offset_m is not None:
            max_lateral_offset_m = abs(vehicle_lateral_offset_m)
        else:
            max_lateral_offset_m = abs(lateral_offset_m) * 0.35
        max_longitudinal_offset_m = abs(
            vehicle_longitudinal_offset_m if vehicle_longitudinal_offset_m is not None else longitudinal_offset_m
        )
    else:
        max_lateral_offset_m = 0.0
        max_longitudinal_offset_m = 0.0

    return {
        "max_tick_offset_ticks": float(max_tick_offset_ticks),
        "tick_bias_ticks": float(tick_bias_ticks),
        "velocity_jitter_ratio": float(velocity_jitter_ratio),
        "velocity_bias": float(velocity_bias),
        "max_lateral_offset_m": float(max_lateral_offset_m),
        "max_longitudinal_offset_m": float(max_longitudinal_offset_m),
        "min_segment_m": float(min_segment_m),
    }


def _path_has_segment(start_pos: Sequence[float], waypoints: list[Any], min_segment_m: float) -> bool:
    current = vector3(start_pos)
    for waypoint in waypoints:
        target = vector3(waypoint, current)
        if distance3(current, target) >= min_segment_m:
            return True
        current = target
    return False


def _path_planar_basis(
    start_pos: Sequence[float],
    waypoints: list[Any],
    min_segment_m: float,
) -> tuple[list[float], list[float], float] | None:
    current = vector3(start_pos)
    for waypoint in waypoints:
        target = vector3(waypoint, current)
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        planar = math.hypot(dx, dy)
        if planar >= min_segment_m:
            tangent = [dx / planar, dy / planar, 0.0]
            normal = [-dy / planar, dx / planar, 0.0]
            return tangent, normal, planar
        current = target
    return None


def _apply_waypoint_offsets(
    start_pos: Sequence[float],
    waypoints: list[Any],
    max_lateral_offset_m: float,
    max_longitudinal_offset_m: float,
    seed_text: str,
    min_segment_m: float,
) -> list[list[float]]:
    normalized = [vector3(waypoint) for waypoint in waypoints]
    if max_lateral_offset_m <= 0.0 and max_longitudinal_offset_m <= 0.0:
        return normalized
    basis = _path_planar_basis(start_pos, normalized, min_segment_m)
    if basis is None:
        return normalized
    tangent, normal, first_segment_m = basis
    lateral_offset = max_lateral_offset_m * _stable_signed_unit(f"{seed_text}|lateral") if max_lateral_offset_m > 0.0 else 0.0
    longitudinal_limit = min(max_longitudinal_offset_m, first_segment_m * 0.4)
    longitudinal_offset = (
        longitudinal_limit * _stable_signed_unit(f"{seed_text}|longitudinal") if longitudinal_limit > 0.0 else 0.0
    )
    varied = [
        [
            point[0] + normal[0] * lateral_offset + tangent[0] * longitudinal_offset,
            point[1] + normal[1] * lateral_offset + tangent[1] * longitudinal_offset,
            point[2],
        ]
        for point in normalized
    ]
    if varied:
        varied[-1] = list(normalized[-1])
    return varied


def _apply_move_variation(
    *,
    scenario_id: str,
    entity_id: str,
    action_id: str,
    label_class: str,
    tick: int,
    velocity_mps: float,
    start_pos: Sequence[float],
    waypoints: list[Any],
    action: dict[str, Any],
    params: dict[str, Any],
) -> tuple[int, float, list[list[float]]]:
    settings = _variation_settings(label_class, action, params)
    normalized_waypoints = [vector3(waypoint) for waypoint in waypoints]
    if settings is None:
        return tick, velocity_mps, normalized_waypoints
    if not normalized_waypoints:
        return tick, velocity_mps, normalized_waypoints

    min_segment_m = max(0.0, float(settings["min_segment_m"]))
    if min_segment_m > 0.0 and not _path_has_segment(start_pos, normalized_waypoints, min_segment_m):
        return tick, velocity_mps, normalized_waypoints

    seed_root = f"{scenario_id}|{entity_id}|{action_id}"
    tick_offset = int(settings["tick_bias_ticks"])
    max_tick_offset = int(settings["max_tick_offset_ticks"])
    if max_tick_offset > 0:
        tick_offset += int(round(_stable_signed_unit(f"{seed_root}|tick") * max_tick_offset))
    varied_tick = max(0, tick + tick_offset)

    velocity_scale = 1.0 + float(settings["velocity_bias"])
    velocity_jitter_ratio = float(settings["velocity_jitter_ratio"])
    if velocity_jitter_ratio > 0.0:
        velocity_scale += _stable_signed_unit(f"{seed_root}|velocity") * velocity_jitter_ratio
    velocity_scale = max(0.1, velocity_scale)
    varied_velocity = max(0.1, float(velocity_mps) * velocity_scale)

    varied_waypoints = _apply_waypoint_offsets(
        start_pos,
        normalized_waypoints,
        float(settings["max_lateral_offset_m"]),
        float(settings["max_longitudinal_offset_m"]),
        seed_root,
        max(min_segment_m, 1e-6),
    )
    return varied_tick, varied_velocity, varied_waypoints


def label_class_for(entity_id: str, logical_asset_id: str, category: str) -> str:
    category_text = str(category).lower()
    asset_text = str(logical_asset_id).lower()
    text = f"{entity_id} {asset_text} {category_text}".lower()
    if category_text == "uav" or asset_text.startswith("uav."):
        return "uav"
    if asset_text.startswith("semantic.uav_corridor.") or "uav_corridor" in text or "highaltitudecorridor" in text:
        return "uav_corridor"
    if category_text == "vehicle" or asset_text.startswith("vehicle."):
        return "vehicle"
    if category_text == "pedestrian" or asset_text.startswith("pedestrian."):
        return "pedestrian"
    if "signal_light" in asset_text:
        return "traffic_light"
    if "base_tower" in asset_text or "tower" in text or "station" in text:
        return "radio_tower"
    if "charger" in asset_text:
        return "charging_pile"
    if asset_text.startswith("trigger.") or "no_fly" in text or "hazard" in text or "zone" in text:
        return "no_fly_zone"
    if asset_text.startswith("facility."):
        return "facility"
    if asset_text.startswith("prop."):
        return "prop"
    return "other"


def position_from_placement(entity: dict[str, Any]) -> list[float]:
    placement = dict(entity.get("placement") or {})
    mode = str(entity.get("placement_mode") or "").lower()
    for key in ("resolved_position_enu_m", "position_enu_m", "center_enu_m"):
        if isinstance(placement.get(key), list):
            return vector3(placement[key])
    if mode == "polygon_prism" and isinstance(placement.get("polygon_enu_m"), list):
        points = [vector3(point) for point in placement["polygon_enu_m"] if isinstance(point, list)]
        if points:
            return [
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
                float(placement.get("base_z_m", points[0][2])),
            ]
    logical_asset_id = str(entity.get("logical_asset_id") or "")
    category = str(entity.get("category") or "")
    if logical_asset_id.startswith("pedestrian.") or category == "pedestrian":
        raise RuntimeError(f"Pedestrian entity lacks resolved placement: {entity.get('entity_id')}")
    return [50.0, 20.0, 0.0]


def _state_label(state: str, *, pedestrian: bool, moving: bool = False) -> str:
    if pedestrian:
        return normalize_activity_type(state, moving=moving)
    return str(state or ("moving" if moving else "idle"))


def _append_frame(
    frames: list[tuple[int, list[float], str]],
    tick: int,
    pos: list[float],
    state: str,
    *,
    pedestrian: bool,
    moving: bool = False,
) -> None:
    state = _state_label(state, pedestrian=pedestrian, moving=moving)
    if frames and frames[-1][0] == tick:
        frames[-1] = (tick, list(pos), state)
        return
    frames.append((tick, list(pos), state))


def keyframes_for(
    initial_pos: list[float],
    schedules: list[dict[str, Any]],
    initial_state: str,
    *,
    pedestrian: bool = False,
) -> list[tuple[int, list[float], str]]:
    current_activity = _state_label(initial_state, pedestrian=pedestrian)
    frames = [(0, list(initial_pos), current_activity)]
    current_pos = list(initial_pos)
    current_tick = 0
    for schedule in sorted(schedules, key=lambda item: int(item.get("tick", 0))):
        schedule_type = str(schedule.get("type") or "move")
        start_tick = max(current_tick, int(schedule.get("tick", 0)))
        if start_tick > current_tick:
            _append_frame(frames, start_tick, current_pos, current_activity, pedestrian=pedestrian)
            current_tick = start_tick
        if schedule_type == "activity":
            current_activity = _state_label(str(schedule.get("activity_type") or current_activity), pedestrian=pedestrian)
            _append_frame(frames, start_tick, current_pos, current_activity, pedestrian=pedestrian)
            continue
        velocity = max(0.1, float(schedule.get("velocity_mps") or 1.0))
        waypoints = list(schedule.get("waypoints_enu_m", []) or [])
        if not waypoints:
            continue
        moving_activity = _state_label(str(schedule.get("activity_type") or current_activity), pedestrian=pedestrian, moving=True)
        post_activity = _state_label(
            str(schedule.get("post_activity_type") or ("waiting" if pedestrian else moving_activity)),
            pedestrian=pedestrian,
        )
        _append_frame(frames, start_tick, current_pos, moving_activity, pedestrian=pedestrian, moving=True)
        if schedule.get("start_pos_enu") is not None:
            current_pos = vector3(schedule.get("start_pos_enu"), current_pos)
        for waypoint_index, waypoint in enumerate(waypoints):
            target = vector3(waypoint, current_pos)
            distance = distance3(current_pos, target)
            if distance <= 1e-6:
                continue
            current_tick += max(1, int(math.ceil(distance / velocity * TICK_HZ)))
            current_pos = target
            frame_activity = post_activity if waypoint_index == len(waypoints) - 1 else moving_activity
            _append_frame(frames, current_tick, current_pos, frame_activity, pedestrian=pedestrian)
            current_activity = frame_activity
    return frames


def sample_keyframes(frames: list[tuple[int, list[float], str]], tick: int) -> tuple[list[float], list[float], str]:
    if not frames:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], "idle"
    frames = sorted(frames, key=lambda item: item[0])
    if tick <= frames[0][0]:
        return list(frames[0][1]), [0.0, 0.0, 0.0], frames[0][2]
    for previous, current in zip(frames, frames[1:]):
        if previous[0] <= tick <= current[0]:
            span = max(1, current[0] - previous[0])
            alpha = (tick - previous[0]) / float(span)
            pos = [previous[1][i] + (current[1][i] - previous[1][i]) * alpha for i in range(3)]
            dt_s = span / float(TICK_HZ)
            vel = [(current[1][i] - previous[1][i]) / dt_s for i in range(3)]
            if tick == current[0]:
                state = current[2]
            else:
                state = previous[2] if math.sqrt(sum(value * value for value in vel)) > 0.05 else current[2]
            return pos, vel, state
    return list(frames[-1][1]), [0.0, 0.0, 0.0], frames[-1][2]


def _row_xy_distance_from(row: dict[str, Any], start: Sequence[float]) -> float:
    pos = row.get("pos_enu")
    if not isinstance(pos, list) or len(pos) < 2:
        return 0.0
    return math.hypot(float(pos[0]) - float(start[0]), float(pos[1]) - float(start[1]))


def _entity_yaw_deg(entity: dict[str, Any]) -> float:
    scene_setup = dict(entity.get("scene_setup") or {})
    placement = dict(scene_setup.get("placement") or {})
    rotation = dict(placement.get("rotation_deg") or {})
    try:
        return float(rotation.get("yaw_deg") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ambient_ground_target(entity_id: str, entity: dict[str, Any], start: Sequence[float], label_class: str) -> list[float]:
    yaw_rad = math.radians(_entity_yaw_deg(entity))
    direction = 1.0 if int(hashlib.sha256(entity_id.encode("utf-8")).hexdigest()[:2], 16) % 2 == 0 else -1.0
    span_m = AMBIENT_GROUND_MOTION_SPAN_M[label_class]
    return [
        float(start[0]) + direction * span_m * math.cos(yaw_rad),
        float(start[1]) + direction * span_m * math.sin(yaw_rad),
        float(start[2] if len(start) > 2 else 0.0),
    ]


def _ambient_position(
    tick: int,
    end_tick: int,
    start: Sequence[float],
    target: Sequence[float],
    *,
    return_to_start: bool,
    label_class: str,
) -> tuple[list[float], list[float]]:
    if end_tick <= 0:
        return vector3(start), [0.0, 0.0, 0.0]
    if return_to_start:
        half = max(1.0, float(end_tick) / 2.0)
        if tick <= half:
            alpha = float(tick) / half
            sign = 1.0
            speed = distance3(start, target) / (half / float(TICK_HZ))
        else:
            alpha = 1.0 - (float(tick) - half) / half
            sign = -1.0
            speed = distance3(start, target) / (half / float(TICK_HZ))
        alpha = max(0.0, min(1.0, alpha))
    else:
        span = max(1e-6, distance3(start, target))
        speed = AMBIENT_GROUND_MOTION_SPEED_MPS[label_class]
        period = max(2.0, 2.0 * span / speed * float(TICK_HZ))
        phase = float(tick % int(math.ceil(period))) / period
        if phase <= 0.5:
            alpha = phase * 2.0
            sign = 1.0
        else:
            alpha = (1.0 - phase) * 2.0
            sign = -1.0
    pos = [float(start[i]) + (float(target[i]) - float(start[i])) * alpha for i in range(3)]
    norm = max(1e-6, distance3(start, target))
    vel = [(float(target[i]) - float(start[i])) / norm * speed * sign for i in range(3)]
    if return_to_start and tick >= end_tick:
        vel = [0.0, 0.0, 0.0]
    return pos, vel


def _row_xy_speed(row: dict[str, Any]) -> float:
    velocity = row.get("vel_mps")
    if not isinstance(velocity, list):
        return 0.0
    return math.hypot(
        float(velocity[0] if len(velocity) > 0 else 0.0),
        float(velocity[1] if len(velocity) > 1 else 0.0),
    )


def _motion_periods(entity_rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    periods: list[tuple[int, int]] = []
    start: int | None = None
    previous_tick: int | None = None
    for row in entity_rows:
        tick = int(row.get("tick", 0))
        moving = _row_xy_speed(row) > GROUND_MOTION_SPEED_EPS_MPS
        if moving and start is None:
            start = tick
        elif not moving and start is not None:
            periods.append((start, int(previous_tick if previous_tick is not None else tick)))
            start = None
        previous_tick = tick
    if start is not None:
        periods.append((start, int(previous_tick if previous_tick is not None else start)))
    return periods


def _last_motion_state(entity_rows: list[dict[str, Any]]) -> tuple[list[float], list[float], int] | None:
    for row in reversed(entity_rows):
        if _row_xy_speed(row) > GROUND_MOTION_SPEED_EPS_MPS:
            return (
                vector3(row.get("pos_enu") or [0.0, 0.0, 0.0]),
                vector3(row.get("vel_mps") or [0.0, 0.0, 0.0]),
                int(row.get("tick", 0)),
            )
    return None


def _continuous_activity_state(label_class: str) -> str:
    if label_class == "pedestrian":
        return normalize_activity_type("walking", moving=True)
    return "moving"


def _write_continuous_ground_row(
    row: dict[str, Any],
    *,
    label_class: str,
    pos: Sequence[float],
    vel: Sequence[float],
) -> None:
    row["pos_enu"] = [round(float(value), 6) for value in pos]
    row["vel_mps"] = [round(float(value), 6) for value in vel]
    state = _continuous_activity_state(label_class)
    row["state"] = state
    row.update(row_activity_payload(label_class, state))


def _has_continuous_ground_flow_contract(entity: dict[str, Any]) -> bool:
    contract = dict(entity.get("ground_flow_contract") or {})
    if not contract:
        scene_setup = dict(entity.get("scene_setup") or {})
        contract = dict(scene_setup.get("ground_flow_contract") or {})
    return str(contract.get("policy") or "") == "continuous_capture_ground_flow_v1"


def _ground_flow_point(
    *,
    tick: int,
    label_class: str,
    anchor_tick: int,
    anchor_pos: Sequence[float],
    unit: Sequence[float],
    span_m: float,
    speed_mps: float,
) -> tuple[list[float], list[float]]:
    period_ticks = max(2, int(round((2.0 * span_m / max(0.1, speed_mps)) * TICK_HZ)))
    phase = ((int(tick) - int(anchor_tick)) % period_ticks) / float(period_ticks)
    if phase <= 0.5:
        offset = phase * 2.0 * span_m
        sign = 1.0
    else:
        offset = (1.0 - phase) * 2.0 * span_m
        sign = -1.0
    pos = [
        float(anchor_pos[0]) + float(unit[0]) * offset,
        float(anchor_pos[1]) + float(unit[1]) * offset,
        float(anchor_pos[2] if len(anchor_pos) > 2 else 0.0),
    ]
    vel = [float(unit[0]) * speed_mps * sign, float(unit[1]) * speed_mps * sign, 0.0]
    return pos, vel


def _unit_from_motion_or_entity(
    *,
    entity_id: str,
    entity: dict[str, Any],
    anchor_pos: Sequence[float],
    preferred_vel: Sequence[float] | None,
    label_class: str,
) -> list[float]:
    if preferred_vel is not None:
        norm = math.hypot(float(preferred_vel[0]), float(preferred_vel[1]))
        if norm > GROUND_MOTION_SPEED_EPS_MPS:
            return [float(preferred_vel[0]) / norm, float(preferred_vel[1]) / norm, 0.0]
    target = _ambient_ground_target(entity_id, entity, anchor_pos, label_class)
    dx = float(target[0]) - float(anchor_pos[0])
    dy = float(target[1]) - float(anchor_pos[1])
    norm = max(1e-6, math.hypot(dx, dy))
    return [dx / norm, dy / norm, 0.0]


def _smooth_ground_jumps(
    entity_rows: list[dict[str, Any]],
    *,
    label_class: str,
    max_step_m: float,
) -> None:
    for index in range(1, len(entity_rows)):
        previous = entity_rows[index - 1]
        current = entity_rows[index]
        previous_pos = vector3(previous.get("pos_enu") or [0.0, 0.0, 0.0])
        current_pos = vector3(current.get("pos_enu") or previous_pos)
        step = distance3(previous_pos, current_pos)
        if step <= max_step_m:
            continue
        dx = current_pos[0] - previous_pos[0]
        dy = current_pos[1] - previous_pos[1]
        norm = max(1e-6, math.hypot(dx, dy))
        current_pos = [
            previous_pos[0] + dx / norm * max_step_m,
            previous_pos[1] + dy / norm * max_step_m,
            previous_pos[2],
        ]
        tick_delta = max(1, int(current.get("tick", index)) - int(previous.get("tick", index - 1)))
        vel = [
            (current_pos[0] - previous_pos[0]) / (tick_delta / float(TICK_HZ)),
            (current_pos[1] - previous_pos[1]) / (tick_delta / float(TICK_HZ)),
            0.0,
        ]
        _write_continuous_ground_row(current, label_class=label_class, pos=current_pos, vel=vel)


def apply_ambient_ground_motion(
    rows: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
    duration_ticks: int,
) -> None:
    rows_by_entity: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        entity_id = str(row.get("entity_id") or "")
        if entity_id:
            rows_by_entity.setdefault(entity_id, []).append(row)
    for entity_id, entity_rows in rows_by_entity.items():
        entity_rows.sort(key=lambda item: int(item.get("tick", 0)))
        if not entity_rows:
            continue
        label_class = str(entity_rows[0].get("label_class") or "")
        if label_class not in AMBIENT_GROUND_MOTION_SPAN_M:
            continue
        entity = entities.get(entity_id)
        if not entity:
            continue
        if _has_continuous_ground_flow_contract(entity):
            continue
        if not (
            bool(entity.get("background_vehicle"))
            or bool(entity.get("background_pedestrian"))
            or str(entity.get("role") or "") in {"semantic_background_vehicle", "semantic_background_pedestrian"}
        ):
            continue
        start = vector3(entity_rows[0].get("pos_enu") or entity.get("pos") or [0.0, 0.0, 0.0])
        target = _ambient_ground_target(entity_id, entity, start, label_class)
        for row in entity_rows:
            tick = int(row.get("tick", 0))
            pos, vel = _ambient_position(
                tick,
                int(duration_ticks),
                start,
                target,
                return_to_start=False,
                label_class=label_class,
            )
            _write_continuous_ground_row(row, label_class=label_class, pos=pos, vel=vel)


def keep_ground_entities_moving(
    rows: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
    duration_ticks: int,
) -> None:
    rows_by_entity: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        entity_id = str(row.get("entity_id") or "")
        if entity_id:
            rows_by_entity.setdefault(entity_id, []).append(row)
    for entity_id, entity_rows in rows_by_entity.items():
        entity_rows.sort(key=lambda item: int(item.get("tick", 0)))
        if not entity_rows:
            continue
        label_class = str(entity_rows[0].get("label_class") or "")
        if label_class not in AMBIENT_GROUND_MOTION_SPAN_M:
            continue
        entity = entities.get(entity_id) or {}
        if _has_continuous_ground_flow_contract(entity):
            continue
        moving_flags = [_row_xy_speed(row) > GROUND_MOTION_SPEED_EPS_MPS for row in entity_rows]
        moving_ratio = sum(1 for flag in moving_flags if flag) / float(len(entity_rows))
        tick0_moving = _row_xy_speed(entity_rows[0]) > GROUND_MOTION_SPEED_EPS_MPS
        if moving_ratio >= 0.85 and tick0_moving:
            continue
        speed = AMBIENT_GROUND_MOTION_SPEED_MPS[label_class]
        span = AMBIENT_GROUND_MOTION_SPAN_M[label_class]
        first_moving_index = next((idx for idx, flag in enumerate(moving_flags) if flag), None)
        if first_moving_index is None:
            anchor_pos = vector3(entity_rows[0].get("pos_enu") or entity.get("pos") or [0.0, 0.0, 0.0])
            unit = _unit_from_motion_or_entity(
                entity_id=entity_id,
                entity=entity,
                anchor_pos=anchor_pos,
                preferred_vel=None,
                label_class=label_class,
            )
            for row in entity_rows:
                pos, vel = _ground_flow_point(
                    tick=int(row.get("tick", 0)),
                    label_class=label_class,
                    anchor_tick=0,
                    anchor_pos=anchor_pos,
                    unit=unit,
                    span_m=span,
                    speed_mps=speed,
                )
                _write_continuous_ground_row(row, label_class=label_class, pos=pos, vel=vel)
            continue

        first_moving = entity_rows[first_moving_index]
        first_tick = int(first_moving.get("tick", 0))
        first_pos = vector3(first_moving.get("pos_enu") or [0.0, 0.0, 0.0])
        first_vel = vector3(first_moving.get("vel_mps") or [0.0, 0.0, 0.0])
        prefix_unit = _unit_from_motion_or_entity(
            entity_id=entity_id,
            entity=entity,
            anchor_pos=first_pos,
            preferred_vel=first_vel,
            label_class=label_class,
        )
        for row in entity_rows[:first_moving_index]:
            tick = int(row.get("tick", 0))
            dt = float(first_tick - tick) / float(TICK_HZ)
            pos = [
                first_pos[0] - prefix_unit[0] * speed * dt,
                first_pos[1] - prefix_unit[1] * speed * dt,
                first_pos[2],
            ]
            vel = [prefix_unit[0] * speed, prefix_unit[1] * speed, 0.0]
            _write_continuous_ground_row(row, label_class=label_class, pos=pos, vel=vel)

        previous_moving_pos = first_pos
        previous_moving_vel = first_vel
        previous_moving_tick = first_tick
        silent_run: list[dict[str, Any]] = []
        for row, moving in zip(entity_rows[first_moving_index + 1 :], moving_flags[first_moving_index + 1 :]):
            if moving:
                if silent_run:
                    next_tick = int(row.get("tick", 0))
                    next_pos = vector3(row.get("pos_enu") or previous_moving_pos)
                    next_vel = vector3(row.get("vel_mps") or previous_moving_vel)
                    gap = max(1, next_tick - previous_moving_tick)
                    gap_speed_mps = distance3(previous_moving_pos, next_pos) / (gap / float(TICK_HZ))
                    for silent in silent_run:
                        alpha = (int(silent.get("tick", 0)) - previous_moving_tick) / float(gap)
                        if distance3(previous_moving_pos, next_pos) > 0.05 and gap_speed_mps <= speed * 1.6:
                            pos = [
                                previous_moving_pos[i] + (next_pos[i] - previous_moving_pos[i]) * alpha
                                for i in range(3)
                            ]
                            vel = [(next_pos[i] - previous_moving_pos[i]) / (gap / float(TICK_HZ)) for i in range(3)]
                        else:
                            unit = _unit_from_motion_or_entity(
                                entity_id=entity_id,
                                entity=entity,
                                anchor_pos=previous_moving_pos,
                                preferred_vel=previous_moving_vel,
                                label_class=label_class,
                            )
                            pos, vel = _ground_flow_point(
                                tick=int(silent.get("tick", 0)),
                                label_class=label_class,
                                anchor_tick=previous_moving_tick,
                                anchor_pos=previous_moving_pos,
                                unit=unit,
                                span_m=span,
                                speed_mps=speed,
                            )
                        _write_continuous_ground_row(silent, label_class=label_class, pos=pos, vel=vel)
                    silent_run = []
                previous_moving_pos = vector3(row.get("pos_enu") or previous_moving_pos)
                previous_moving_vel = vector3(row.get("vel_mps") or previous_moving_vel)
                previous_moving_tick = int(row.get("tick", 0))
            else:
                silent_run.append(row)
        if silent_run:
            unit = _unit_from_motion_or_entity(
                entity_id=entity_id,
                entity=entity,
                anchor_pos=previous_moving_pos,
                preferred_vel=previous_moving_vel,
                label_class=label_class,
            )
            for silent in silent_run:
                pos, vel = _ground_flow_point(
                    tick=int(silent.get("tick", 0)),
                    label_class=label_class,
                    anchor_tick=previous_moving_tick,
                    anchor_pos=previous_moving_pos,
                    unit=unit,
                    span_m=span,
                    speed_mps=speed,
                )
                _write_continuous_ground_row(silent, label_class=label_class, pos=pos, vel=vel)
        _smooth_ground_jumps(
            entity_rows,
            label_class=label_class,
            max_step_m=0.4 if label_class == "pedestrian" else 1.0,
        )


def keep_inspect_uavs_looping(
    rows: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
    duration_ticks: int,
) -> None:
    rows_by_entity: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        entity_id = str(row.get("entity_id") or "")
        if entity_id:
            rows_by_entity.setdefault(entity_id, []).append(row)
    for entity_id, entity in entities.items():
        if not bool(entity.get("contract_inspect_uav")) and str(entity.get("role") or "") != "U_inspect":
            continue
        entity_rows = rows_by_entity.get(entity_id)
        if not entity_rows:
            continue
        entity_rows.sort(key=lambda item: int(item.get("tick", 0)))
        route = [vector3(point) for point in (entity.get("route_waypoints_enu_m") or [])]
        if len(route) < 2:
            continue
        start = vector3(route[0])
        full_route = extend_loop_route(
            start,
            route[1:],
            velocity_mps=INSPECT_UAV_LOOP_SPEED_MPS,
            duration_ticks=duration_ticks,
            min_motion_ratio=INSPECT_UAV_MIN_MOTION_RATIO,
        )
        frames = keyframes_for(
            start,
            [
                {
                    "type": "move",
                    "tick": 0,
                    "waypoints_enu_m": full_route,
                    "velocity_mps": INSPECT_UAV_LOOP_SPEED_MPS,
                    "activity_type": "inspect_racetrack",
                    "post_activity_type": "inspect_racetrack",
                    "source": "contract_inspect_uav.loop_route_enu_m",
                    "start_pos_enu": start,
                }
            ],
            "inspect_racetrack",
            pedestrian=False,
        )
        for row in entity_rows:
            tick = int(row.get("tick", 0))
            pos, vel, state = sample_keyframes(frames, tick)
            row["pos_enu"] = [round(float(value), 6) for value in pos]
            row["vel_mps"] = [round(float(value), 6) for value in vel]
            row["state"] = state
            row.update(row_activity_payload("uav", state))


def preserved_fields_from(*sources: dict[str, Any]) -> dict[str, Any]:
    preserved: dict[str, Any] = {}
    for field in PRESERVED_ENTITY_FIELDS:
        for source in sources:
            if field in source and source[field] not in (None, ""):
                preserved[field] = copy.deepcopy(source[field])
                break
            for nested_key in ("initial_state", "visual_state"):
                nested = source.get(nested_key)
                if isinstance(nested, dict) and field in nested and nested[field] not in (None, ""):
                    preserved[field] = copy.deepcopy(nested[field])
                    break
            if field in preserved:
                break
    return preserved


def weather_payload(profile: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    key = str(profile or "clear").strip().lower()
    payload = copy.deepcopy(WEATHER_PROFILES.get(key))
    if payload is None:
        raise RuntimeError(f"Unknown deterministic weather profile: {profile}")
    payload["condition"] = str(profile or payload.get("condition") or key)
    for field, value in dict(overrides or {}).items():
        payload[field] = value
        if field == "visibility_m":
            payload["visibility"] = value
        elif field == "visibility":
            payload["visibility_m"] = value
        elif field == "fog":
            payload["fog_density"] = value
        elif field == "fog_density":
            payload["fog"] = value
    if "visibility_m" not in payload and "visibility" in payload:
        payload["visibility_m"] = payload["visibility"]
    if "visibility" not in payload and "visibility_m" in payload:
        payload["visibility"] = payload["visibility_m"]
    if "fog" not in payload and "fog_density" in payload:
        payload["fog"] = payload["fog_density"]
    if "fog_density" not in payload and "fog" in payload:
        payload["fog_density"] = payload["fog"]
    return payload


def initial_weather_state(scene_setup: dict[str, Any]) -> dict[str, Any]:
    profile = dict(scene_setup.get("weather_profile") or {})
    return weather_payload(str(profile.get("initial") or "clear"))


def scheduled_weather_transitions(scene_setup: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    transitions: dict[int, list[dict[str, Any]]] = {}
    for transition in (scene_setup.get("weather_profile") or {}).get("transitions") or []:
        tick = _to_int(transition.get("tick"), 0)
        if tick < 0:
            raise RuntimeError(f"Weather transition tick is negative: {transition}")
        transitions.setdefault(tick, []).append(
            weather_payload(str(transition.get("profile") or "clear"), dict(transition.get("overrides") or {}))
        )
    return transitions


def row_activity_payload(label_class: str, state: str) -> dict[str, Any]:
    if label_class == "pedestrian":
        activity = get_activity(state)
        return {
            "activity_type": activity.activity_type,
            "animation_hint": activity.animation_hint,
            "posture": activity.posture,
            "social_state": activity.social_state,
        }
    return {
        "activity_type": state,
        "animation_hint": state,
        "posture": "standing",
        "social_state": "solo",
    }


def route_motion_schedule(
    *,
    scenario_id: str,
    entity_id: str,
    label_class: str,
    initial_pos: Sequence[float],
    route_waypoints: list[Any],
    initial_state: str,
    ground_flow_contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not route_waypoints:
        return []
    velocity = 4.0
    if label_class == "uav":
        velocity = 5.0
    elif label_class == "vehicle":
        velocity = 6.0
    elif label_class == "pedestrian":
        velocity = 1.25
    ground_flow = dict(ground_flow_contract or {})
    if str(ground_flow.get("policy") or "") == "continuous_capture_ground_flow_v1":
        velocity = max(0.1, _to_float(ground_flow.get("speed_mps", velocity), velocity))
        route_waypoints = [vector3(waypoint) for waypoint in route_waypoints]
    activity_type = initial_state
    post_activity_type = initial_state
    if label_class == "pedestrian":
        activity_type = normalize_activity_type("walking", moving=True)
        post_activity_type = normalize_activity_type(initial_state or "waiting")
        if str(ground_flow.get("policy") or "") == "continuous_capture_ground_flow_v1":
            post_activity_type = activity_type
    elif str(ground_flow.get("policy") or "") == "continuous_capture_ground_flow_v1":
        post_activity_type = activity_type
    return [
        {
            "type": "move",
            "tick": 0,
            "waypoints_enu_m": [vector3(waypoint) for waypoint in route_waypoints],
            "velocity_mps": velocity,
            "activity_type": activity_type,
            "post_activity_type": post_activity_type,
            "source": "scene_setup.route_waypoints_enu_m",
            "action_id": f"{scenario_id}.{entity_id}.route_waypoints",
            "start_pos_enu": vector3(initial_pos),
        }
    ]


def build_entities(scene_setup: dict[str, Any], script: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    scenario_id = str(script.get("scenario_id") or "")
    event_move_targets = {
        str(action.get("entity_id") or "")
        for event in script.get("events") or []
        for action in event.get("actions") or []
        if str(action.get("type") or "") == "move_entity"
    }
    for scene_entity in scene_setup.get("entities") or []:
        entity_id = str(scene_entity.get("entity_id") or scene_entity.get("instance_id") or "")
        if not entity_id:
            continue
        logical_asset_id = str(scene_entity.get("logical_asset_id") or "")
        category = str(scene_entity.get("category") or "")
        initial_state = dict(scene_entity.get("initial_state") or {})
        activation_tick = int(scene_entity.get("activation_tick") or 0)
        spawn_policy = str(scene_entity.get("spawn_policy") or "").lower()
        initial_activity = (
            initial_state.get("activity_type")
            or (initial_state.get("state_facets") or {}).get("activity", {}).get("activity_type")
            or initial_state.get("mode")
            or "idle"
        )
        if logical_asset_id.startswith("pedestrian.") or category == "pedestrian":
            initial_activity = normalize_activity_type(str(initial_activity or "waiting"))
        label_class = label_class_for(entity_id, logical_asset_id, category)
        position = position_from_placement(scene_entity)
        preserved = preserved_fields_from(scene_entity)
        role = str(preserved.get("role") or "")
        auto_route = bool(scene_entity.get("route_waypoints_enu_m")) and (
            entity_id not in event_move_targets
            or bool(scene_entity.get("contract_inspect_uav"))
            or bool(scene_entity.get("background_vehicle"))
            or bool(scene_entity.get("background_pedestrian"))
            or role in {"U_inspect", "semantic_background_vehicle", "semantic_background_pedestrian"}
            or entity_id.startswith("uav_observer_")
        )
        entities[entity_id] = {
            "entity_id": entity_id,
            "pos": position,
            "label_class": label_class,
            "asset_id": logical_asset_id,
            "state": str(initial_activity or "idle"),
            "active_from": activation_tick if activation_tick > 0 or spawn_policy == "event_script_only" else 0,
            "scene_setup": copy.deepcopy(scene_entity),
            "schedules": (
                route_motion_schedule(
                    scenario_id=scenario_id,
                    entity_id=entity_id,
                    label_class=label_class,
                    initial_pos=position,
                    route_waypoints=list(scene_entity.get("route_waypoints_enu_m") or []),
                    initial_state=str(initial_activity or "idle"),
                    ground_flow_contract=dict(scene_entity.get("ground_flow_contract") or {}),
                )
                if auto_route
                else []
            ),
            **preserved,
        }
    return entities


class EpisodeStateEngine:
    def __init__(self, scene_setup: dict[str, Any], script: dict[str, Any], script_path: Path, duration_ticks: int) -> None:
        self.scene_setup = scene_setup
        self.script = script
        self.script_path = script_path
        self.duration_ticks = int(duration_ticks)
        self.scenario_id = str(script.get("scenario_id") or script_path.parent.name)
        self.params = dict(script.get("parameters") or {})
        self.entities = build_entities(scene_setup, script)
        self.keyframes: dict[str, list[tuple[int, list[float], str]]] = {
            entity_id: self._frames_for_entity(entity)
            for entity_id, entity in self.entities.items()
        }
        self.weather = initial_weather_state(scene_setup)
        self.weather_transitions = scheduled_weather_transitions(scene_setup)
        self.trajectory_rows: list[dict[str, Any]] = []
        self.weather_rows: list[dict[str, Any]] = []
        self.executed_actions: list[dict[str, Any]] = []

    def _frames_for_entity(self, entity: dict[str, Any]) -> list[tuple[int, list[float], str]]:
        return keyframes_for(
            vector3(entity["pos"]),
            list(entity.get("schedules") or []),
            str(entity.get("state") or "idle"),
            pedestrian=str(entity.get("label_class") or "") == "pedestrian",
        )

    def _refresh_entity_frames(self, entity_id: str) -> None:
        self.keyframes[entity_id] = self._frames_for_entity(self.entities[entity_id])

    def _entity_sample(self, entity_id: str, tick: int) -> tuple[list[float], list[float], str]:
        entity = self.entities[entity_id]
        if tick < int(entity.get("active_from") or 0):
            return vector3(entity["pos"]), [0.0, 0.0, 0.0], "offstage"
        return sample_keyframes(self.keyframes[entity_id], tick)

    def _entity_pos_at(self, entity_id: str, tick: int) -> list[float]:
        return self._entity_sample(entity_id, tick)[0]

    def _append_action_result(self, action: dict[str, Any], status: str, **extra: Any) -> dict[str, Any]:
        result = {"status": status, **extra}
        self.executed_actions.append(
            {
                "tick": int(extra.get("tick", -1)),
                "action_id": str(action.get("action_id") or ""),
                "type": str(action.get("type") or ""),
                "entity_id": str(action.get("entity_id") or action.get("ped_id") or ""),
                "result": result,
            }
        )
        return result

    def _handle_move_entity(self, action: dict[str, Any], tick: int) -> dict[str, Any]:
        entity_id = str(resolve_param(action.get("entity_id", ""), self.params) or "")
        if entity_id not in self.entities:
            raise RuntimeError(f"move_entity references undeclared entity {entity_id}")
        entity = self.entities[entity_id]
        label_class = str(entity.get("label_class") or "")
        waypoints_raw = resolve_param(action.get("waypoints_enu_m", []), self.params)
        waypoints = [vector3(point) for point in waypoints_raw] if isinstance(waypoints_raw, list) else []
        if not waypoints:
            raise RuntimeError(f"move_entity action has no waypoints: {action.get('action_id')}")
        action_id = str(action.get("action_id") or f"{self.scenario_id}.{entity_id}.move_at_{tick}")
        activity_type = action.get("activity_type")
        post_activity_type = action.get("post_activity_type")
        if label_class == "pedestrian":
            activity_type = normalize_activity_type(str(activity_type or "walking"), moving=True)
            post_activity_type = normalize_activity_type(str(post_activity_type or "waiting"))
        current_pos = self._entity_pos_at(entity_id, tick)
        varied_tick, varied_velocity, varied_waypoints = _apply_move_variation(
            scenario_id=self.scenario_id,
            entity_id=entity_id,
            action_id=action_id,
            label_class=label_class,
            tick=tick,
            velocity_mps=_to_float(action.get("velocity_mps", 1.0), 1.0),
            start_pos=current_pos,
            waypoints=waypoints,
            action=action,
            params=self.params,
        )
        schedule = {
            "type": "move",
            "tick": varied_tick,
            "waypoints_enu_m": varied_waypoints,
            "velocity_mps": varied_velocity,
            "activity_type": activity_type,
            "post_activity_type": post_activity_type,
            "source": "event_script.move_entity",
            "source_event_tick": tick,
            "action_id": action_id,
            "start_pos_enu": current_pos,
        }
        entity.setdefault("schedules", []).append(schedule)
        self._refresh_entity_frames(entity_id)
        return self._append_action_result(
            action,
            "ok",
            tick=tick,
            scheduled_tick=varied_tick,
            entity_id=entity_id,
            path_length_m=round(path_length_m([current_pos, *varied_waypoints]), 6),
        )

    def _handle_set_visual_state(self, action: dict[str, Any], tick: int) -> dict[str, Any]:
        entity_id = str(resolve_param(action.get("entity_id", ""), self.params) or "")
        if entity_id not in self.entities:
            raise RuntimeError(f"set_visual_state references undeclared entity {entity_id}")
        visual_state = dict(action.get("visual_state") or {})
        mode = str(visual_state.get("mode") or action.get("mode") or "")
        if mode:
            self.entities[entity_id]["state"] = mode
            self.entities[entity_id].setdefault("schedules", []).append(
                {
                    "type": "activity",
                    "tick": tick,
                    "activity_type": mode,
                    "source": "event_script.set_visual_state",
                    "action_id": str(action.get("action_id") or ""),
                }
            )
            self._refresh_entity_frames(entity_id)
        self.entities[entity_id].update(preserved_fields_from(visual_state))
        return self._append_action_result(action, "ok", tick=tick, entity_id=entity_id, mode=mode)

    def _handle_set_pedestrian_activity(self, action: dict[str, Any], tick: int) -> dict[str, Any]:
        entity_id = str(resolve_param(action.get("entity_id", action.get("ped_id", "")), self.params) or "")
        if entity_id not in self.entities:
            raise RuntimeError(f"set_pedestrian_activity references undeclared entity {entity_id}")
        activity_type = normalize_activity_type(str(action.get("activity_type") or "waiting"))
        self.entities[entity_id]["state"] = activity_type
        self.entities[entity_id].setdefault("schedules", []).append(
            {
                "type": "activity",
                "tick": tick,
                "activity_type": activity_type,
                "source": "event_script.set_pedestrian_activity",
                "action_id": str(action.get("action_id") or ""),
            }
        )
        self._refresh_entity_frames(entity_id)
        return self._append_action_result(action, "ok", tick=tick, entity_id=entity_id, activity_type=activity_type)

    def _handle_set_weather(self, action: dict[str, Any], tick: int) -> dict[str, Any]:
        profile = str(action.get("profile") or "clear")
        self.weather = weather_payload(profile, dict(action.get("overrides") or {}))
        return self._append_action_result(action, "ok", tick=tick, profile=profile)

    def _handle_spawn_entity(self, action: dict[str, Any], tick: int) -> dict[str, Any]:
        entity_id = str(resolve_param(action.get("entity_id", ""), self.params) or "")
        if entity_id not in self.entities:
            raise RuntimeError(f"spawn_entity references undeclared entity {entity_id}")
        position = vector3(resolve_param(action.get("position_enu_m", self.entities[entity_id]["pos"]), self.params))
        entity = self.entities[entity_id]
        entity["pos"] = position
        entity["active_from"] = tick
        if action.get("asset_id"):
            entity["asset_id"] = str(resolve_param(action.get("asset_id"), self.params) or entity.get("asset_id") or "")
        entity.setdefault("schedules", []).append(
            {
                "type": "activity",
                "tick": tick,
                "activity_type": str((action.get("visual_state") or {}).get("mode") or entity.get("state") or "spawned"),
                "source": "event_script.spawn_entity",
                "action_id": str(action.get("action_id") or ""),
            }
        )
        self._refresh_entity_frames(entity_id)
        return self._append_action_result(action, "ok", tick=tick, entity_id=entity_id)

    def _handle_capture_screenshot(self, action: dict[str, Any], tick: int) -> dict[str, Any]:
        return self._append_action_result(action, "ok", tick=tick, capture_id=str(action.get("action_id") or action.get("camera_id") or ""))

    def _handle_noop(self, action: dict[str, Any], tick: int) -> dict[str, Any]:
        atype = str(action.get("type") or "")
        if atype in {"sequence", "remove_entity", "play_animation", "spawn_crowd"}:
            return self._append_action_result(action, "ok", tick=tick)
        raise RuntimeError(f"Unhandled deterministic event action type: {atype}")

    def _row_for_entity(self, entity_id: str, tick: int) -> dict[str, Any]:
        entity = self.entities[entity_id]
        pos, vel, state = self._entity_sample(entity_id, tick)
        row = {
            "tick": tick,
            "entity_id": entity_id,
            "label_class": entity["label_class"],
            "asset_id": entity["asset_id"],
            "pos_enu": [round(float(value), 6) for value in pos],
            "vel_mps": [round(float(value), 6) for value in vel],
            "state": state,
            **row_activity_payload(str(entity["label_class"]), state),
        }
        row.update(preserved_fields_from(entity))
        return row

    def _record_tick_rows(self, tick: int) -> list[dict[str, Any]]:
        rows = [self._row_for_entity(entity_id, tick) for entity_id in sorted(self.entities)]
        self.trajectory_rows.extend(rows)
        weather_row = {"tick": tick, **copy.deepcopy(self.weather)}
        self.weather_rows.append(weather_row)
        return rows

    def run(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Plugins" / "SumoImporter" / "Scripts"))
        from donghu_core.event_script_interpreter import EventScriptInterpreter

        interpreter = EventScriptInterpreter(self.script_path)
        current_tick = 0
        interpreter.register_handler("move_entity", lambda action: self._handle_move_entity(action, current_tick))
        interpreter.register_handler("set_visual_state", lambda action: self._handle_set_visual_state(action, current_tick))
        interpreter.register_handler("set_pedestrian_activity", lambda action: self._handle_set_pedestrian_activity(action, current_tick))
        interpreter.register_handler("set_weather", lambda action: self._handle_set_weather(action, current_tick))
        interpreter.register_handler("spawn_entity", lambda action: self._handle_spawn_entity(action, current_tick))
        interpreter.register_handler("capture_screenshot", lambda action: self._handle_capture_screenshot(action, current_tick))
        interpreter.register_handler("remove_entity", lambda action: self._handle_noop(action, current_tick))
        interpreter.register_handler("sequence", lambda action: self._handle_noop(action, current_tick))
        interpreter.register_handler("play_animation", lambda action: self._handle_noop(action, current_tick))
        interpreter.register_handler("spawn_crowd", lambda action: self._handle_noop(action, current_tick))

        for tick in range(self.duration_ticks + 1):
            current_tick = tick
            for payload in self.weather_transitions.get(tick, []):
                self.weather = payload
            rows = self._record_tick_rows(tick)
            for row in rows:
                interpreter.update_entity_state(row["entity_id"], row["pos_enu"], {}, row["vel_mps"])
                if str(row.get("label_class") or "") == "pedestrian":
                    interpreter.update_entity_activity(row["entity_id"], str(row.get("activity_type") or ""))
            interpreter.update_weather_state(self.weather_rows[-1])
            interpreter.tick(tick)

        event_log = interpreter.get_event_log()
        fired_event_ids = {
            remove_prefix(str(row.get("topic") or row.get("source_event_id") or ""), f"evt_{self.scenario_id}_")
            for row in event_log
        }
        logged_event_topics = {str(row.get("topic") or "") for row in event_log}
        missing: list[str] = []
        for event_def in self.script.get("events") or []:
            event_id = str(event_def.get("event_id") or "")
            if not event_id:
                continue
            log_event = dict(event_def.get("log_event") or {})
            topic = str(log_event.get("topic") or event_id)
            if topic not in logged_event_topics and event_id not in fired_event_ids:
                missing.append(event_id)
        if missing:
            raise RuntimeError(f"{self.scenario_id}: declared events did not fire in deterministic episode simulation: {missing}")

        apply_ambient_ground_motion(self.trajectory_rows, self.entities, self.duration_ticks)
        keep_ground_entities_moving(self.trajectory_rows, self.entities, self.duration_ticks)
        keep_inspect_uavs_looping(self.trajectory_rows, self.entities, self.duration_ticks)

        roster: dict[str, dict[str, Any]] = {}
        for entity_id, entity in sorted(self.entities.items()):
            roster[entity_id] = {
                "entity_id": entity_id,
                "label_class": entity["label_class"],
                "asset_id": entity["asset_id"],
                **preserved_fields_from(entity),
            }
        return self.trajectory_rows, self.weather_rows, event_log, roster


def generate_episode(script_path: Path, episode_dir: Path, dataset_root: Path, seed: int = 0, skip_graphs: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)

    script = read_json(script_path)
    scene_setup = read_json(script_path.with_name("scene_setup.json")) if script_path.with_name("scene_setup.json").exists() else {}
    scenario_id = str(script.get("scenario_id") or script_path.parent.name)
    duration = int(script.get("parameters", {}).get("duration_ticks") or DEFAULT_DURATION_TICKS)

    engine = EpisodeStateEngine(scene_setup, script, script_path, duration)
    trajectories, weather, event_log, roster = engine.run()
    write_jsonl(episode_dir / "trajectories.jsonl", trajectories)
    write_jsonl(episode_dir / "weather_meta.jsonl", weather)
    write_jsonl(episode_dir / "event_trace.jsonl", event_log)
    write_json(episode_dir / "global_entity_roster.json", roster)

    manifest = {
        "episode_id": episode_dir.name,
        "scenario_id": scenario_id,
        "duration_ticks": duration,
        "seed": seed,
        "n_events": len(event_log),
        "n_entities": len(roster),
        "source_event_script_path": project_relative(script_path, dataset_root),
        "source_scene_setup_path": project_relative(script_path.with_name("scene_setup.json"), dataset_root),
        "generator": "Dataset/tools/batch_generate.py",
    }
    write_json(episode_dir / "episode_manifest.json", manifest)

    summary: dict[str, Any] = {"n_frames": len({row["tick"] for row in trajectories}), "n_unique_entities": len(roster)}
    if not skip_graphs:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from generate_graph_labels import generate_frame_graphs

        frame_graphs, summary = generate_frame_graphs(episode_dir)
        graphs_dir = episode_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(graphs_dir / "frame_graphs.jsonl", frame_graphs)
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate grounded episodes")
    parser.add_argument("--dataset-root", default="Dataset")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--episode", action="append", default=[], help="Scenario id / episode name to generate. Repeatable.")
    parser.add_argument("--skip-graphs", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    scenario_scripts = sorted((dataset_root / "scenarios").rglob("event_script.json"))
    if args.episode:
        wanted = {str(value).replace("__seed00", "") for value in args.episode}
        scenario_scripts = [path for path in scenario_scripts if path.parent.name in wanted or str(read_json(path).get("scenario_id")) in wanted]
    print(f"Found {len(scenario_scripts)} scenarios")

    total_episodes = 0
    total_events = 0
    for script_path in scenario_scripts:
        scenario_name = script_path.parent.name
        for seed in range(max(1, int(args.seeds))):
            episode_dir = dataset_root / "episodes" / f"{scenario_name}__seed{seed:02d}"
            manifest, summary = generate_episode(script_path, episode_dir, dataset_root, seed, bool(args.skip_graphs))
            total_episodes += 1
            total_events += int(manifest["n_events"])
            print(
                f"  [OK] {episode_dir.name}: {manifest['n_events']} events, "
                f"{summary.get('n_frames', 0)} frames, {summary.get('n_unique_entities', 0)} entities"
            )
    print(f"\nGenerated {total_episodes} episodes with {total_events} total events")


if __name__ == "__main__":
    main()
