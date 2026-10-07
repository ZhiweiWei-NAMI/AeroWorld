"""Build deterministic per-episode SUMO vehicle contracts.

The contract is the only source of vehicle truth for formal render-ready
episodes.  SUMO consumes these routes to produce coordinates; it does not invent
background traffic at runtime.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .incident_plan import DEFAULT_SUMO_NET_XML
from .planner import SumoEdge, SumoGroundFlowPlanner
from .road_semantic_rules import resolve_scene_road_semantics


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EPISODES_ROOT = ROOT / "Dataset" / "episodes"
EXPLICIT_PLAN_FILENAME = "sumo_explicit_vehicle_plan.json"
REVIEW_JSON_FILENAME = "vehicle_plan_review.json"
REVIEW_MD_FILENAME = "vehicle_plan_review.md"
TRAFFIC_TUNING_FILENAME = "episode_traffic_tuning.json"
SCHEMA = "aero_explicit_sumo_vehicle_plan_v2"
VEHICLE_SOURCE_POLICY = "explicit_episode_traffic_schedule_only_v2"
EVENT_SCRIPT_ONLY_SPAWN_POLICY = "event_script_only"
DURATION_TICKS = 900
TICK_HZ = 10
FORMAL_WARMUP_TICKS = 900
SOURCE_PRESENCE_MANIFEST_FIELDS = frozenset(
    {
        "source_presence_required_vehicle_count",
        "source_presence_required_vehicle_ids",
        "source_presence_capture_lifecycle_by_vehicle_id",
    }
)
MIN_ORDINARY_VEHICLES = 8
MIN_VEHICLE_EVENT_VEHICLES = 12
MIN_COMPLEX_VEHICLES = 16
TRAFFIC_SLOT_TICKS = 100
TRAFFIC_SLOT_COUNT = 9
CORE_CROSSING_TICKS = 25
ESTIMATED_UPSTREAM_TRAVEL_TICKS = 70
ROI_VISIBLE_FLOW_LEAD_TICKS = ESTIMATED_UPSTREAM_TRAVEL_TICKS * 4
CONTEXT_ROI_VISIBLE_FLOW_LEAD_TICKS = ESTIMATED_UPSTREAM_TRAVEL_TICKS * 6
CONTEXT_ROI_LATE_SLOT_EXTRA_LEAD_TICKS = ESTIMATED_UPSTREAM_TRAVEL_TICKS
ROI_BOUNDARY_APPROACH_RELEASE_LEAD_TICKS = ESTIMATED_UPSTREAM_TRAVEL_TICKS
ROI_POST_PROTECTED_RELEASE_LEAD_TICKS = 0
PROTECTED_WINDOW_PRE_ROUTE_GUARD_TICKS = ESTIMATED_UPSTREAM_TRAVEL_TICKS * 3
PROTECTED_WINDOW_POST_ROUTE_GUARD_TICKS = ROI_POST_PROTECTED_RELEASE_LEAD_TICKS
PROTECTED_WINDOW_ROUTE_GUARD_TICKS = PROTECTED_WINDOW_PRE_ROUTE_GUARD_TICKS
MIN_CORE_ROI_WIDTH_M = 120.0
MIN_CORE_ROI_HEIGHT_M = 150.0
PRE_EVENT_VISIBLE_MARGIN_TICKS = 50
POST_EVENT_VISIBLE_MARGIN_TICKS = 50
SEMANTIC_BOUNDARY_ENTRY_LEAD_TICKS = 2
SEMANTIC_PRE_ACTIVATION_TICKS = 10
SEMANTIC_EXIT_FRONT_BUMPER_MARGIN_M = 3.0
SEMANTIC_DISPATCH_APPROACH_DISTANCE_M = 30.0
MIN_SEMANTIC_QUEUE_GAP_M = 8.0
SLOT_BACKGROUND_TARGETS = {
    "uav_only_or_context": {0: 6, 1: 5, 2: 6},
    "road_or_vehicle_semantic_event": {0: 6, 1: 5, 2: 6},
    "complex_emergency_weather_or_combo": {0: 10, 1: 8, 2: 10},
}
CONTEXT_FREE_FLOW_BACKGROUND_TARGETS = {0: 8, 1: 6, 2: 8}
CONTEXT_FREE_FLOW_ROI_SMOOTHING_ENTRY_TICKS = (40, 276, 500, 700, 760, 820, 880)
CONTEXT_FREE_FLOW_ROI_SMOOTHING_STOP_TICKS = 45
EMERGENCY_CORRIDOR_BACKGROUND_TARGETS = {0: 10, 1: 8, 2: 10}
EMERGENCY_CORRIDOR_ACCEPTED_BACKGROUND_TARGETS = {0: 50, 1: 40, 2: 50}
DEFAULT_ROUTE_MIN_EDGES = 10
DEFAULT_ROUTE_MAX_EDGES = 28
ROI_DEPART_MARGIN_M = 4.0
DEFAULT_EXPLICIT_VEHICLE_LENGTH_M = 5.0
ROI_BODY_CENTER_INSET_M = DEFAULT_EXPLICIT_VEHICLE_LENGTH_M * 0.5
MAX_CONNECTED_START_EDGE_ATTEMPTS = 160
OFF_ROI_ENTRY_SAMPLES = tuple(sample_index / 40.0 for sample_index in range(41))
MAX_SEMANTIC_LANE_PROJECTION_ERROR_M = 3.0
BACKGROUND_REPLENISHMENT_FRACTION = 0.40
TRAFFIC_TUNING_PATH = Path(__file__).with_name(TRAFFIC_TUNING_FILENAME)
_TRAFFIC_TUNING_CACHE: dict[str, Any] | None = None


@dataclass(frozen=True)
class SeedTrafficProfile:
    seed_index: int
    seed_label: str
    profile_id: str
    direction_bias: str
    peak_multiplier: float
    slot_entry_jitter_ticks: tuple[int, ...]
    target_speed_mps: float
    direction_distribution: dict[str, float]


SEED_TRAFFIC_PROFILES: dict[int, SeedTrafficProfile] = {
    0: SeedTrafficProfile(
        seed_index=0,
        seed_label="seed00",
        profile_id="morning_peak",
        direction_bias="inbound_to_roi",
        peak_multiplier=1.0,
        slot_entry_jitter_ticks=(0, -3, 2, -1, 4, -2, 3, 0, -4, 1),
        target_speed_mps=7.0,
        direction_distribution={"inbound": 0.7, "outbound": 0.2, "cross": 0.1},
    ),
    1: SeedTrafficProfile(
        seed_index=1,
        seed_label="seed01",
        profile_id="midday",
        direction_bias="balanced_cross_traffic",
        peak_multiplier=1.0,
        slot_entry_jitter_ticks=(0, 4, -4, 2, -2, 0, 3, -3),
        target_speed_mps=8.5,
        direction_distribution={"inbound": 0.34, "outbound": 0.33, "cross": 0.33},
    ),
    2: SeedTrafficProfile(
        seed_index=2,
        seed_label="seed02",
        profile_id="evening_peak",
        direction_bias="outbound_from_roi",
        peak_multiplier=1.0,
        slot_entry_jitter_ticks=(0, 3, -2, 1, -4, 2, -1, 4, -3, 0),
        target_speed_mps=6.8,
        direction_distribution={"inbound": 0.2, "outbound": 0.7, "cross": 0.1},
    ),
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _load_traffic_tuning_config(path: Path = TRAFFIC_TUNING_PATH) -> dict[str, Any]:
    global _TRAFFIC_TUNING_CACHE
    if _TRAFFIC_TUNING_CACHE is not None:
        return copy.deepcopy(_TRAFFIC_TUNING_CACHE)
    if not path.exists():
        _TRAFFIC_TUNING_CACHE = {
            "defaults": {},
            "scenario_overrides": {},
            "episode_overrides": {},
        }
        return copy.deepcopy(_TRAFFIC_TUNING_CACHE)
    loaded = _read_json(path)
    _TRAFFIC_TUNING_CACHE = loaded if isinstance(loaded, dict) else {}
    return copy.deepcopy(_TRAFFIC_TUNING_CACHE)


def _traffic_tuning_for_episode(episode_id: str, scenario_id: str) -> dict[str, Any]:
    config = _load_traffic_tuning_config()
    tuning = dict(config.get("defaults") or {})
    scenario_override = dict(dict(config.get("scenario_overrides") or {}).get(str(scenario_id), {}) or {})
    episode_override = dict(dict(config.get("episode_overrides") or {}).get(str(episode_id), {}) or {})
    for override in (scenario_override, episode_override):
        for key, value in override.items():
            tuning[key] = value
    tuning["source"] = {
        "config_path": str(TRAFFIC_TUNING_PATH),
        "scenario_id": str(scenario_id),
        "episode_id": str(episode_id),
        "has_scenario_override": bool(scenario_override),
        "has_episode_override": bool(episode_override),
    }
    return tuning


def _semantic_lane_projection_override(traffic_tuning: dict[str, Any], source_entity_id: str) -> dict[str, Any]:
    overrides = traffic_tuning.get("semantic_lane_projection_overrides")
    if not isinstance(overrides, dict):
        return {}
    source_override = overrides.get(str(source_entity_id))
    if isinstance(source_override, dict):
        return dict(source_override)
    wildcard = overrides.get("*")
    return dict(wildcard) if isinstance(wildcard, dict) else {}


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip().lower())
    return token.strip("_") or "vehicle"


def _ordered_unique_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def episode_scenario_id(episode_id: str) -> str:
    return re.sub(r"__seed\d+$", "", str(episode_id))


def episode_seed_index(episode_id: str, manifest: dict[str, Any] | None = None) -> int:
    if manifest is not None and manifest.get("seed") not in (None, ""):
        return int(manifest["seed"])
    match = re.search(r"__seed(\d+)$", str(episode_id))
    return int(match.group(1)) if match else 0


def semantic_internal_vehicle_id(source_entity_id: str, seed_index: int) -> str:
    return f"{source_entity_id}__seed{seed_index:02d}"


def explicit_plan_path(episode_dir: Path) -> Path:
    return Path(episode_dir) / EXPLICIT_PLAN_FILENAME


def review_json_path(episode_dir: Path) -> Path:
    return Path(episode_dir) / REVIEW_JSON_FILENAME


def review_md_path(episode_dir: Path) -> Path:
    return Path(episode_dir) / REVIEW_MD_FILENAME


def resolve_manifest_path(project_root: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(project_root) / path


def load_explicit_vehicle_plan(episode_dir_or_root: Path, episode_id: str | None = None) -> dict[str, Any]:
    root = Path(episode_dir_or_root)
    path = root / str(episode_id) / EXPLICIT_PLAN_FILENAME if episode_id else explicit_plan_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Explicit SUMO vehicle plan is required and missing: {path}")
    plan = _read_json(path)
    if str(plan.get("schema") or "") != SCHEMA:
        raise ValueError(f"{path}: unsupported explicit SUMO vehicle plan schema {plan.get('schema')!r}")
    return plan


def planned_vehicle_ids(plan: dict[str, Any]) -> set[str]:
    return {str(vehicle.get("vehicle_id") or "") for vehicle in plan.get("vehicles") or [] if str(vehicle.get("vehicle_id") or "")}


def planned_source_vehicle_ids(plan: dict[str, Any]) -> set[str]:
    return {
        str(vehicle.get("source_entity_id") or "")
        for vehicle in plan.get("vehicles") or []
        if str(vehicle.get("source_entity_id") or "")
    }


def is_script_controlled_vehicle(entity: Mapping[str, Any]) -> bool:
    """Return True for vehicles whose motion authority is the episode script.

    ``semantic_vehicle`` is a semantic role, not by itself a motion-authority
    declaration.  Script-only/spatially kinematic vehicles stay out of SUMO;
    source vehicles whose contract is only ``state_animation`` still need an
    explicit SUMO route so their physical presence is represented.  A bare
    semantic role with no motion contract remains script-controlled and thus
    fails closed, preserving the conservative source default.
    """
    if str(entity.get("spawn_policy") or "").casefold() == EVENT_SCRIPT_ONLY_SPAWN_POLICY:
        return True
    role = str(entity.get("role") or "").casefold()
    if role == "semantic_event":
        return True
    if str(entity.get("traffic_role") or "").casefold() == "semantic_vehicle":
        return True
    if bool(entity.get("semantic_actor") is True):
        return True
    if role == "semantic_vehicle":
        motion_contract = entity.get("motion_contract")
        if isinstance(motion_contract, Mapping):
            motion_kind = str(motion_contract.get("motion_kind") or "").casefold()
            if motion_kind == "state_animation":
                return False
        return True
    return False


def planned_script_controlled_source_vehicle_ids(
    plan: dict[str, Any],
) -> set[str]:
    """Source entity ids for script-controlled vehicles present in a plan."""
    return {
        str(vehicle.get("source_entity_id") or "")
        for vehicle in plan.get("vehicles") or []
        if str(vehicle.get("source_entity_id") or "")
        and is_script_controlled_vehicle(vehicle)
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def _position3(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        z = float(value[2]) if len(value) >= 3 else 0.0
        return [float(value[0]), float(value[1]), z]
    except (TypeError, ValueError):
        return None


def _dedupe_points(points: Sequence[Sequence[float]]) -> list[list[float]]:
    unique: list[list[float]] = []
    seen: set[tuple[float, float, float]] = set()
    for point in points:
        pos = _position3(point)
        if pos is None:
            continue
        key = (round(pos[0], 3), round(pos[1], 3), round(pos[2], 3))
        if key in seen:
            continue
        seen.add(key)
        unique.append([round(pos[0], 6), round(pos[1], 6), round(pos[2], 6)])
    return unique


def _entity_position(entity: dict[str, Any]) -> list[float] | None:
    placement = dict(entity.get("placement") or {})
    for key in ("resolved_position_enu_m", "position_enu_m"):
        point = _position3(placement.get(key))
        if point is not None:
            return point
    waypoints = entity.get("route_waypoints_enu_m") or []
    if waypoints:
        return _position3(waypoints[0])
    return None


def _capture_boundary(script: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("capture_boundary"), dict):
                visit(value["capture_boundary"])
            if isinstance(value.get("polygon_enu_m"), list):
                candidates.append(value)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    visit(child)

    visit(script)
    return dict(candidates[0]) if candidates else {}


def _bbox_from_points(points: Sequence[Sequence[float]], padding_m: float = 0.0) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        pos = _position3(point)
        if pos is None:
            continue
        xs.append(pos[0])
        ys.append(pos[1])
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round(min(xs) - padding_m, 6),
        round(min(ys) - padding_m, 6),
        round(max(xs) + padding_m, 6),
        round(max(ys) + padding_m, 6),
    ]


def _bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5)


def _expand_bbox_to_minimum(bbox: Sequence[float], min_width_m: float, min_height_m: float) -> list[float]:
    if len(bbox) < 4:
        return [float(value) for value in bbox]
    xmin, ymin, xmax, ymax = [float(value) for value in bbox[:4]]
    center_x = (xmin + xmax) * 0.5
    center_y = (ymin + ymax) * 0.5
    half_width = max(xmax - xmin, float(min_width_m)) * 0.5
    half_height = max(ymax - ymin, float(min_height_m)) * 0.5
    return [
        round(center_x - half_width, 6),
        round(center_y - half_height, 6),
        round(center_x + half_width, 6),
        round(center_y + half_height, 6),
    ]


def _point_in_bbox(point: Sequence[float], bbox: Sequence[float]) -> bool:
    return float(bbox[0]) <= float(point[0]) <= float(bbox[2]) and float(bbox[1]) <= float(point[1]) <= float(bbox[3])


def _point_in_polygon_xy(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    x = float(point[0])
    y = float(point[1])
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = float(polygon[index][0]), float(polygon[index][1])
        x2, y2 = float(polygon[(index + 1) % count][0]), float(polygon[(index + 1) % count][1])
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def _distance_point_to_segment_xy(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    px = float(point[0])
    py = float(point[1])
    ax = float(a[0])
    ay = float(a[1])
    bx = float(b[0])
    by = float(b[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    qx = ax + dx * t
    qy = ay + dy * t
    return math.hypot(px - qx, py - qy)


def _distance_to_polygon_xy(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> float:
    if not polygon:
        return float("inf")
    if _point_in_polygon_xy(point, polygon):
        return 0.0
    return min(
        _distance_point_to_segment_xy(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _distance_to_bbox_xy(point: Sequence[float], bbox: Sequence[float]) -> float:
    if len(bbox) < 4:
        return float("inf")
    x = float(point[0])
    y = float(point[1])
    dx = max(float(bbox[0]) - x, 0.0, x - float(bbox[2]))
    dy = max(float(bbox[1]) - y, 0.0, y - float(bbox[3]))
    return math.hypot(dx, dy)


def _point_in_spatial_scope(point: Sequence[float], spatial_scope: dict[str, Any]) -> bool:
    polygon = spatial_scope.get("capture_boundary_polygon_enu_m") or []
    if polygon:
        padding = float(spatial_scope.get("expanded_boundary_padding_m") or 60.0)
        return _distance_to_polygon_xy(point, polygon) <= padding + 1e-6
    bbox = spatial_scope.get("expanded_bbox_enu_m") or []
    return bool(bbox) and _point_in_bbox(point, bbox)


def _core_spatial_scope(spatial_scope: dict[str, Any]) -> dict[str, Any]:
    core = dict(spatial_scope)
    core["expanded_boundary_padding_m"] = 0.0
    bbox = list(core.get("bbox_enu_m") or [])
    if bbox:
        core["expanded_bbox_enu_m"] = [float(value) for value in bbox]
    return core


def _edge_allows_vehicle(edge: SumoEdge) -> bool:
    vehicle_tokens = {"passenger", "delivery", "truck", "bus", "taxi", "motorcycle", "moped", "emergency"}
    if edge.allow:
        return bool(edge.allow & vehicle_tokens)
    forbidden = {"footway", "pedestrian", "path", "steps", "rail", "tram"}
    if edge.disallow & vehicle_tokens:
        return False
    return not any(token in str(edge.edge_type) for token in forbidden)


def _edge_direction_role(edge: SumoEdge, center_xy: tuple[float, float]) -> str:
    if len(edge.shape_xy) < 2:
        return "cross"
    sx, sy = edge.shape_xy[0]
    ex, ey = edge.shape_xy[-1]
    cd0 = math.hypot(float(sx) - center_xy[0], float(sy) - center_xy[1])
    cd1 = math.hypot(float(ex) - center_xy[0], float(ey) - center_xy[1])
    if cd0 > cd1 + 2.0:
        return "inbound"
    if cd1 > cd0 + 2.0:
        return "outbound"
    return "cross"


def _edge_distance_to_center(edge: SumoEdge, center_xy: tuple[float, float]) -> float:
    return min(math.hypot(float(x) - center_xy[0], float(y) - center_xy[1]) for x, y in edge.shape_xy)


def _project_point_to_edge(edge: SumoEdge, point: Sequence[float]) -> dict[str, Any]:
    px = float(point[0])
    py = float(point[1])
    best_distance = float("inf")
    best_s = 0.0
    best_xy = edge.shape_xy[0]
    cumulative = 0.0
    for a, b in zip(edge.shape_xy, edge.shape_xy[1:]):
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dx = bx - ax
        dy = by - ay
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (length * length)))
        qx = ax + t * dx
        qy = ay + t * dy
        distance = math.hypot(px - qx, py - qy)
        if distance < best_distance:
            best_distance = distance
            best_s = cumulative + t * length
            best_xy = (qx, qy)
        cumulative += length
    return {
        "lane_position_m": round(float(best_s), 6),
        "distance_m": round(float(best_distance), 6),
        "xy_enu_m": [round(float(best_xy[0]), 6), round(float(best_xy[1]), 6)],
    }


def _point_at_edge_s(edge: SumoEdge, s_m: float) -> tuple[float, float]:
    remaining = max(0.0, float(s_m))
    last = edge.shape_xy[0]
    for current in edge.shape_xy[1:]:
        length = math.hypot(float(current[0]) - float(last[0]), float(current[1]) - float(last[1]))
        if length > 1e-9 and remaining <= length:
            t = remaining / length
            return (
                float(last[0]) + (float(current[0]) - float(last[0])) * t,
                float(last[1]) + (float(current[1]) - float(last[1])) * t,
            )
        remaining -= length
        last = current
    return float(edge.shape_xy[-1][0]), float(edge.shape_xy[-1][1])


def _depart_s_outside_bbox(edge: SumoEdge, bbox: Sequence[float], candidate_s: float) -> float | None:
    upper = max(0.0, float(edge.length_m))
    if upper <= 0.0:
        return None
    if not bbox or len(bbox) < 4:
        return max(0.0, min(float(candidate_s), upper))
    base = max(0.0, min(float(candidate_s), upper))
    margin_s = min(max(ROI_DEPART_MARGIN_M, base), max(0.0, upper - ROI_DEPART_MARGIN_M))
    probes: list[float] = []
    for value in (
        base,
        margin_s,
        base - 0.1,
        base - 0.25,
        base - 0.5,
        base - 1.0,
        base - ROI_DEPART_MARGIN_M,
        base - ROI_DEPART_MARGIN_M * 1.5,
        base - ROI_DEPART_MARGIN_M * 2.0,
        base + 0.1,
        base + 0.25,
        base + 0.5,
        base + 1.0,
        base + ROI_DEPART_MARGIN_M,
        base + ROI_DEPART_MARGIN_M * 1.5,
        base + ROI_DEPART_MARGIN_M * 2.0,
        0.0,
        upper,
    ):
        s_value = round(max(0.0, min(float(value), upper)), 6)
        if s_value not in probes:
            probes.append(s_value)
    for sample_index in range(41):
        s_value = round(upper * float(sample_index) / 40.0, 6)
        if s_value not in probes:
            probes.append(s_value)
    for s_value in probes:
        point = _point_at_edge_s(edge, s_value)
        if not _point_in_bbox(point, bbox) and _distance_to_bbox_xy(point, bbox) >= ROI_DEPART_MARGIN_M:
            return s_value
    return None


def _depart_s_before_bbox_with_clearance(edge: SumoEdge, bbox: Sequence[float], candidate_s: float) -> float | None:
    upper = max(0.0, min(float(candidate_s), float(edge.length_m)))
    probes: list[float] = []
    for value in (
        upper,
        upper - 0.25,
        upper - 0.5,
        upper - 1.0,
        upper - ROI_DEPART_MARGIN_M,
        upper - ROI_DEPART_MARGIN_M * 1.5,
        upper - ROI_DEPART_MARGIN_M * 2.0,
        0.0,
    ):
        s_value = round(max(0.0, min(float(value), upper)), 6)
        if s_value not in probes:
            probes.append(s_value)
    for sample_index in range(40, -1, -1):
        s_value = round(upper * float(sample_index) / 40.0, 6)
        if s_value not in probes:
            probes.append(s_value)
    for s_value in probes:
        point = _point_at_edge_s(edge, s_value)
        if not _point_in_bbox(point, bbox) and _distance_to_bbox_xy(point, bbox) >= ROI_DEPART_MARGIN_M:
            return s_value
    return None


def _edge_depart_s_outside_roi(
    edge: SumoEdge,
    spatial_scope: dict[str, Any],
    center_xy: tuple[float, float],
) -> float | None:
    best: tuple[float, float] | None = None
    cumulative = 0.0
    bbox = spatial_scope.get("bbox_enu_m") or []
    if bbox and edge.shape_xy and not any(_point_in_bbox(point, bbox) for point in edge.shape_xy):
        best_vertex: tuple[float, float] | None = None
        s_value = 0.0
        previous = edge.shape_xy[0]
        for index, point in enumerate(edge.shape_xy):
            if index > 0:
                s_value += math.hypot(float(point[0]) - float(previous[0]), float(point[1]) - float(previous[1]))
            score = math.hypot(float(point[0]) - center_xy[0], float(point[1]) - center_xy[1])
            if best_vertex is None or score < best_vertex[0]:
                best_vertex = (score, s_value)
            previous = point
        if best_vertex is None:
            return None
        return _depart_s_outside_bbox(edge, bbox, best_vertex[1])
    for a, b in zip(edge.shape_xy, edge.shape_xy[1:]):
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        length = math.hypot(bx - ax, by - ay)
        if length <= 1e-9:
            continue
        for t in OFF_ROI_ENTRY_SAMPLES:
            qx = ax + (bx - ax) * t
            qy = ay + (by - ay) * t
            if (bbox and _point_in_bbox([qx, qy], bbox)) or (not bbox and _point_in_spatial_scope([qx, qy], spatial_scope)):
                continue
            score = math.hypot(qx - center_xy[0], qy - center_xy[1])
            s_value = cumulative + t * length
            if best is None or score < best[0]:
                best = (score, s_value)
        cumulative += length
    if best is None:
        return None
    return _depart_s_outside_bbox(edge, bbox, best[1])


def _edge_depart_s_before_bbox(edge: SumoEdge, bbox: Sequence[float]) -> float | None:
    if not bbox or len(bbox) < 4 or len(edge.shape_xy) < 2:
        return None
    previous = edge.shape_xy[0]
    cumulative = 0.0
    last_outside_s: float | None = None
    if not _point_in_bbox(previous, bbox):
        last_outside_s = 0.0
    for current in edge.shape_xy[1:]:
        ax, ay = float(previous[0]), float(previous[1])
        bx, by = float(current[0]), float(current[1])
        length = math.hypot(bx - ax, by - ay)
        if length <= 1e-9:
            previous = current
            continue
        for sample_index in range(1, 41):
            t = sample_index / 40.0
            qx = ax + (bx - ax) * t
            qy = ay + (by - ay) * t
            s_value = cumulative + length * t
            if _point_in_bbox([qx, qy], bbox):
                if last_outside_s is None:
                    return None
                return _depart_s_before_bbox_with_clearance(edge, bbox, last_outside_s)
            last_outside_s = s_value
        cumulative += length
        previous = current
    return None


def _segment_bbox_interval(
    a: Sequence[float],
    b: Sequence[float],
    bbox: Sequence[float],
) -> tuple[float, float] | None:
    if len(bbox) < 4:
        return None
    min_x, min_y, max_x, max_y = [float(value) for value in bbox[:4]]
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    dy = by - ay
    p = (-dx, dx, -dy, dy)
    q = (ax - min_x, max_x - ax, ay - min_y, max_y - ay)
    u1 = 0.0
    u2 = 1.0
    for pi, qi in zip(p, q):
        if abs(pi) <= 1e-12:
            if qi < 0.0:
                return None
            continue
        r = qi / pi
        if pi < 0.0:
            u1 = max(u1, r)
        else:
            u2 = min(u2, r)
        if u1 > u2:
            return None
    return u1, u2


def _segment_intersects_bbox(a: Sequence[float], b: Sequence[float], bbox: Sequence[float]) -> bool:
    return _segment_bbox_interval(a, b, bbox) is not None


def _edge_intersects_bbox(edge: SumoEdge, bbox: Sequence[float]) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    return any(
        _segment_intersects_bbox(a, b, bbox)
        for a, b in zip(edge.shape_xy, edge.shape_xy[1:])
    )


def _edge_intersects_scope(edge: SumoEdge, spatial_scope: dict[str, Any]) -> bool:
    bbox = spatial_scope.get("expanded_bbox_enu_m") or spatial_scope.get("bbox_enu_m") or []
    if bbox:
        return _edge_intersects_bbox(edge, bbox)
    return any(_point_in_spatial_scope(point, spatial_scope) for point in edge.shape_xy)


def _edge_has_non_roi_point(edge: SumoEdge, spatial_scope: dict[str, Any]) -> bool:
    bbox = spatial_scope.get("bbox_enu_m") or []
    if bbox:
        return bool(edge.shape_xy) and any(not _point_in_bbox(point, bbox) for point in edge.shape_xy)
    return bool(edge.shape_xy) and not any(_point_in_spatial_scope(point, spatial_scope) for point in edge.shape_xy)


def _endpoint_edge_candidates(
    planner: SumoGroundFlowPlanner,
    spatial_scope: dict[str, Any],
    center_xy: tuple[float, float],
    direction_role: str | None,
    *,
    for_from: bool,
) -> list[SumoEdge]:
    core_scope = _core_spatial_scope(spatial_scope)
    preferred = str(direction_role or "")
    ranked: list[tuple[int, float, str, SumoEdge]] = []
    for edge in planner.edges.values():
        if not _edge_allows_vehicle(edge) or len(edge.shape_xy) < 2:
            continue
        if not _edge_has_non_roi_point(edge, spatial_scope):
            continue
        if for_from and _edge_depart_s_outside_roi(edge, spatial_scope, center_xy) is None:
            continue
        role = _edge_direction_role(edge, center_xy)
        if for_from:
            role_score = 0 if role == preferred else 1
        else:
            opposite = {"inbound": "outbound", "outbound": "inbound", "cross": "cross"}.get(preferred, preferred)
            role_score = 0 if role == opposite else 1
        core_score = 0 if _edge_intersects_scope(edge, core_scope) else 1
        ranked.append((role_score + core_score, _edge_distance_to_center(edge, center_xy), edge.edge_id, edge))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in ranked]


def _candidate_from_edges(planner: SumoGroundFlowPlanner, spatial_scope: dict[str, Any], direction_role: str) -> list[SumoEdge]:
    center_xy = _bbox_center(spatial_scope.get("bbox_enu_m") or spatial_scope.get("expanded_bbox_enu_m") or [0.0, 0.0, 0.0, 0.0])
    return _endpoint_edge_candidates(planner, spatial_scope, center_xy, direction_role, for_from=True)


def _candidate_to_edges(planner: SumoGroundFlowPlanner, spatial_scope: dict[str, Any], direction_role: str) -> list[SumoEdge]:
    center_xy = _bbox_center(spatial_scope.get("bbox_enu_m") or spatial_scope.get("expanded_bbox_enu_m") or [0.0, 0.0, 0.0, 0.0])
    return _endpoint_edge_candidates(planner, spatial_scope, center_xy, direction_role, for_from=False)


def _reverse_adjacency(planner: SumoGroundFlowPlanner) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for src, destinations in planner.adjacency.items():
        for dst in destinations:
            reverse.setdefault(str(dst), []).append(str(src))
    return reverse


def _adjacent_vehicle_edges(
    planner: SumoGroundFlowPlanner,
    edge_id: str,
    reverse_adjacency: dict[str, list[str]],
    *,
    incoming: bool,
) -> list[SumoEdge]:
    candidates: list[SumoEdge] = []
    seen: set[str] = set()
    frontier = list(reverse_adjacency.get(edge_id, [])) if incoming else list(planner.adjacency.get(edge_id, []))
    for adjacent_id in frontier:
        edge = planner.edges.get(str(adjacent_id))
        if edge is not None and _edge_allows_vehicle(edge) and edge.edge_id not in seen:
            candidates.append(edge)
            seen.add(edge.edge_id)
            continue
        next_ids = reverse_adjacency.get(str(adjacent_id), []) if incoming else planner.adjacency.get(str(adjacent_id), [])
        for next_id in next_ids:
            edge = planner.edges.get(str(next_id))
            if edge is not None and _edge_allows_vehicle(edge) and edge.edge_id not in seen:
                candidates.append(edge)
                seen.add(edge.edge_id)
    return candidates


def _unique_edges(edges: Sequence[SumoEdge]) -> list[SumoEdge]:
    unique: list[SumoEdge] = []
    seen_edges: set[str] = set()
    for edge in edges:
        if edge.edge_id in seen_edges:
            continue
        seen_edges.add(edge.edge_id)
        unique.append(edge)
    return unique


def _sorted_edge_ids(edge_ids: Sequence[str]) -> list[str]:
    return sorted({str(edge_id) for edge_id in edge_ids if str(edge_id)})


def _edge_id_variants(edge_id: str) -> list[str]:
    value = str(edge_id or "").strip()
    if not value:
        return []
    variants = [value]
    for suffix in ("_f_pl0", "_f_pl1", "_r_pl0", "_r_pl1"):
        variants.append(f"{value}{suffix}")
    return [item for index, item in enumerate(variants) if item and item not in variants[:index]]


def _semantic_corridor_points_for_entity(episode_dir: Path, source_entity_id: str) -> list[list[float]]:
    points: list[list[float]] = []
    for row in _read_jsonl(Path(episode_dir) / "event_realization.jsonl"):
        for action in row.get("action_realizations") or []:
            if not isinstance(action, dict) or str(action.get("entity_id") or "") != source_entity_id:
                continue
            for point in action.get("waypoints_enu_m") or []:
                pos = _position3(point)
                if pos is not None:
                    points.append(pos)
            terminal = _position3(action.get("terminal_enu_m"))
            if terminal is not None:
                points.append(terminal)
        snapshots = row.get("source_truth_snapshots_by_tick")
        if isinstance(snapshots, dict):
            for by_entity in snapshots.values():
                if not isinstance(by_entity, dict):
                    continue
                snapshot = by_entity.get(source_entity_id)
                if not isinstance(snapshot, dict):
                    continue
                point = _position3(snapshot.get("position_enu_m"))
                if point is not None:
                    points.append(point)
    return _dedupe_points(points)


def _rank_semantic_edges_for_points(
    planner: SumoGroundFlowPlanner,
    points: Sequence[Sequence[float]],
) -> list[tuple[float, float, float, str, SumoEdge]]:
    ranked: list[tuple[float, float, float, str, SumoEdge]] = []
    for edge in planner.edges.values():
        if not _edge_allows_vehicle(edge) or len(edge.shape_xy) < 2:
            continue
        distances = [_project_point_to_edge(edge, point)["distance_m"] for point in points]
        mean_distance = float(sum(distances) / max(1, len(distances)))
        max_distance = float(max(distances)) if distances else 0.0
        score = mean_distance + max_distance * 0.25
        ranked.append((score, max_distance, mean_distance, edge.edge_id, edge))
    ranked.sort(key=lambda item: (item[0], item[3]))
    return ranked


def _best_semantic_edge_for_points(
    candidates: Sequence[SumoEdge],
    points: Sequence[Sequence[float]],
) -> tuple[float, float, SumoEdge] | None:
    best: tuple[float, float, SumoEdge] | None = None
    for edge in candidates:
        distances = [float(_project_point_to_edge(edge, point)["distance_m"]) for point in points]
        mean_distance = sum(distances) / max(1, len(distances))
        max_distance = max(distances) if distances else 0.0
        score = mean_distance + max_distance * 0.25
        if best is None or score < best[0]:
            best = (score, max_distance, edge)
    return best


def _semantic_corridor_for_vehicle(
    *,
    planner: SumoGroundFlowPlanner,
    episode_dir: Path,
    source_entity: dict[str, Any],
    source_entity_id: str,
    traffic_tuning: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not source_entity_id:
        return None
    points = _semantic_corridor_points_for_entity(episode_dir, source_entity_id)
    if not points:
        return None
    override = _semantic_lane_projection_override(dict(traffic_tuning or {}), source_entity_id)
    override_limit_m = _optional_float(override.get("max_projection_error_m"))
    if (
        override_limit_m is not None
        and float(override_limit_m) > MAX_SEMANTIC_LANE_PROJECTION_ERROR_M
    ):
        raise RuntimeError(
            f"semantic vehicle {source_entity_id} configures projection tolerance "
            f"{float(override_limit_m):.3f}m above the physical threshold "
            f"{MAX_SEMANTIC_LANE_PROJECTION_ERROR_M:.3f}m; repair the authored route instead of "
            "relaxing lane authority"
        )
    forced_edge_id = str(override.get("edge_id") or "").strip()
    placement = dict(source_entity.get("placement") or {})
    candidate_ids: list[str] = []
    for key in ("edge_id", "source_edge_id_hint"):
        candidate_ids.extend(_edge_id_variants(str(placement.get(key) or "")))
    candidates: list[SumoEdge] = []
    if forced_edge_id:
        forced_edge = planner.edges.get(forced_edge_id)
        if forced_edge is None or not _edge_allows_vehicle(forced_edge):
            raise RuntimeError(
                f"semantic vehicle {source_entity_id} configured lane {forced_edge_id} is missing or not vehicle-legal"
            )
        candidates.append(forced_edge)
    else:
        for edge_id in candidate_ids:
            edge = planner.edges.get(edge_id)
            if edge is not None and _edge_allows_vehicle(edge):
                candidates.append(edge)
    ranked_edges: list[tuple[float, float, float, str, SumoEdge]] = []
    if not candidates:
        ranked_edges = _rank_semantic_edges_for_points(planner, points)
        candidates = [item[4] for item in ranked_edges[:8]]
    best = _best_semantic_edge_for_points(candidates, points)
    if best is not None and best[1] > MAX_SEMANTIC_LANE_PROJECTION_ERROR_M and not forced_edge_id:
        if not ranked_edges:
            ranked_edges = _rank_semantic_edges_for_points(planner, points)
        by_edge_id = {edge.edge_id: edge for edge in candidates}
        for _score, _max_distance, _mean_distance, _edge_id, edge in ranked_edges[:8]:
            by_edge_id.setdefault(edge.edge_id, edge)
        best = _best_semantic_edge_for_points(list(by_edge_id.values()), points)
    if best is None:
        return None
    edge = best[2]
    projection_error_tolerance_m = MAX_SEMANTIC_LANE_PROJECTION_ERROR_M
    corridor_source = (
        str(override.get("source") or "scenario_specific_semantic_lane_edge_override")
        if forced_edge_id
        else "scene_placement_edge_or_event_point_projection"
    )
    if (
        not forced_edge_id
        and any(token in source_entity_id.lower() for token in ("yield", "civilian"))
        and edge.edge_id.endswith("_pl0")
    ):
        sibling_edge = planner.edges.get(edge.edge_id[:-4] + "_pl1")
        if sibling_edge is not None and _edge_allows_vehicle(sibling_edge):
            sibling_distances = [float(_project_point_to_edge(sibling_edge, point)["distance_m"]) for point in points]
            sibling_max_distance = max(sibling_distances) if sibling_distances else 0.0
            if sibling_max_distance <= MAX_SEMANTIC_LANE_PROJECTION_ERROR_M:
                sibling_mean_distance = sum(sibling_distances) / max(1, len(sibling_distances))
                edge = sibling_edge
                best = (sibling_mean_distance + sibling_max_distance * 0.25, sibling_max_distance, sibling_edge)
                corridor_source = "adjacent_lane_for_yield_clearance"
    max_projection_error = float(best[1])
    if max_projection_error > projection_error_tolerance_m:
        raise RuntimeError(
            f"semantic vehicle {source_entity_id} authored corridor is {max_projection_error:.3f}m from "
            f"legal SUMO edge {edge.edge_id}, above the physical threshold "
            f"{MAX_SEMANTIC_LANE_PROJECTION_ERROR_M:.3f}m; repair the source geometry instead of snapping it"
        )
    override_reason = ""
    if override_limit_m is not None:
        corridor_source = str(override.get("source") or "scenario_specific_semantic_lane_projection_override")
        override_reason = str(override.get("reason") or "")
    bootstrap_depart_pos_m = _optional_float(override.get("bootstrap_depart_pos_m"))
    return {
        "policy": "generator_fixed_semantic_lane_corridor_v1",
        "edge_id": edge.edge_id,
        "lane_id": edge.lane_id,
        "lane_index": 0,
        "projection_error_max_m": round(float(best[1]), 6),
        "projection_error_score_m": round(float(best[0]), 6),
        "projection_error_tolerance_m": round(float(projection_error_tolerance_m), 6),
        "source": corridor_source,
        "override_reason": override_reason,
        "bootstrap_depart_pos_m": round(float(bootstrap_depart_pos_m), 6) if bootstrap_depart_pos_m is not None else None,
    }


def _role_and_type(entity: dict[str, Any], semantic_actor: bool) -> tuple[str, str, str]:
    role = "semantic_event" if semantic_actor else "background"
    asset = str(entity.get("logical_asset_id") or entity.get("asset_id") or "")
    by_asset = {
        "vehicle.emergency.ambulance.v1": ("aero_ambulance", "emergency"),
        "vehicle.emergency.police_suv.v1": ("aero_police", "emergency"),
        "vehicle.emergency.suv.v1": ("aero_emergency", "emergency"),
        "vehicle.service.box.v1": ("aero_delivery", "delivery"),
        "vehicle.ground.boxcar.v1": ("aero_passenger", "passenger"),
    }
    if asset:
        if asset not in by_asset:
            raise ValueError(f"unsupported source vehicle logical asset: {asset}")
        type_id, vehicle_class = by_asset[asset]
        return role, type_id, vehicle_class
    if entity.get("entity_id"):
        raise ValueError(f"source vehicle {entity['entity_id']} lacks logical asset")
    return role, "aero_passenger", "passenger"


def _source_presence_required(
    entity: dict[str, Any] | None,
    *,
    semantic_actor: bool = False,
) -> bool:
    return (
        bool(entity)
        and not semantic_actor
        and str(entity.get("role") or "") == "semantic_vehicle"
    )


def validate_source_presence_manifest_contract(
    explicit_plan: Mapping[str, Any],
    sumo_manifest: Mapping[str, Any],
) -> set[str]:
    episode_id = str(explicit_plan.get("episode_id") or "<unknown>")
    planned_ids = {
        str(vehicle.get("vehicle_id") or "")
        for vehicle in explicit_plan.get("vehicles") or []
        if vehicle.get("source_presence_required") is True
        and str(vehicle.get("vehicle_id") or "")
    }
    activation = sumo_manifest.get("explicit_vehicle_activation")
    if not isinstance(activation, dict):
        raise RuntimeError(f"{episode_id}: SUMO manifest lacks explicit_vehicle_activation")
    missing_fields = sorted(SOURCE_PRESENCE_MANIFEST_FIELDS - set(activation))
    if missing_fields:
        raise RuntimeError(
            f"{episode_id}: SUMO manifest lacks source-presence fields: {missing_fields}"
        )

    raw_ids = activation["source_presence_required_vehicle_ids"]
    if not isinstance(raw_ids, list) or any(
        not isinstance(vehicle_id, str) or not vehicle_id for vehicle_id in raw_ids
    ):
        raise RuntimeError(
            f"{episode_id}: source_presence_required_vehicle_ids must be a list of non-empty strings"
        )
    declared_ids = set(raw_ids)
    if len(declared_ids) != len(raw_ids):
        raise RuntimeError(
            f"{episode_id}: source_presence_required_vehicle_ids contains duplicates"
        )
    if declared_ids != planned_ids:
        raise RuntimeError(
            f"{episode_id}: source-required vehicle manifest/plan mismatch: "
            f"planned={sorted(planned_ids)} declared={sorted(declared_ids)}"
        )

    declared_count = activation["source_presence_required_vehicle_count"]
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        raise RuntimeError(
            f"{episode_id}: source_presence_required_vehicle_count must be an integer"
        )
    if declared_count != len(planned_ids):
        raise RuntimeError(
            f"{episode_id}: source_presence_required_vehicle_count={declared_count} "
            f"does not match planned count={len(planned_ids)}"
        )

    lifecycle = activation["source_presence_capture_lifecycle_by_vehicle_id"]
    if not isinstance(lifecycle, dict) or set(lifecycle) != planned_ids:
        lifecycle_ids = sorted(lifecycle) if isinstance(lifecycle, dict) else []
        raise RuntimeError(
            f"{episode_id}: source-presence lifecycle IDs do not match the plan: "
            f"planned={sorted(planned_ids)} lifecycle={lifecycle_ids}"
        )
    capture_start_s = float(sumo_manifest.get("capture_start_s") or 0.0)
    capture_end_s = capture_start_s + float(
        sumo_manifest.get("capture_duration_s") or 0.0
    )
    for vehicle_id in sorted(planned_ids):
        record = lifecycle[vehicle_id]
        if not isinstance(record, dict):
            raise RuntimeError(
                f"{episode_id}: source-presence lifecycle for {vehicle_id} must be an object"
            )
        first_active_s = record.get("first_active_s")
        last_active_s = record.get("last_active_s")
        required_interval = record.get("required_active_interval_s")
        if (
            not isinstance(first_active_s, (int, float))
            or isinstance(first_active_s, bool)
            or not isinstance(last_active_s, (int, float))
            or isinstance(last_active_s, bool)
            or not isinstance(required_interval, list)
            or len(required_interval) != 2
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in required_interval
            )
            or abs(float(required_interval[0]) - capture_start_s) > 1e-6
            or abs(float(required_interval[1]) - capture_end_s) > 1e-6
            or float(first_active_s) > capture_start_s + 1e-6
            or float(last_active_s) < capture_end_s - 1e-6
        ):
            raise RuntimeError(
                f"{episode_id}: source-presence lifecycle for {vehicle_id} does not cover "
                f"the formal capture interval [{capture_start_s}, {capture_end_s}]"
            )
    return planned_ids


def _scenario_classification(scenario_id: str, script: dict[str, Any], bindings: Sequence[dict[str, Any]]) -> tuple[str, int]:
    text = _json_text(script)
    adverse_weather = _has_adverse_weather(scenario_id, script)
    complex_tokens = (
        "forced_landing",
        "hazmat",
        "crowd",
        "evacuation",
        "collision",
        "c2loss",
        "roadwork",
        "lockdown",
        "congestion",
        "gridlock",
        "traffic_jam",
        "pileup",
    )
    vehicle_tokens = (
        "traffic_light",
        "ground_vehicle",
        "car",
        "ambulance",
        "police",
        "yield",
        "crash",
        "collision",
        "lane",
        "signal",
    )
    if scenario_id.startswith(("X", "L5")) or adverse_weather or any(token in text for token in complex_tokens):
        return "complex_emergency_weather_or_combo", MIN_COMPLEX_VEHICLES
    if scenario_id.startswith(("L2", "L3", "L4")) or bindings or any(token in text for token in vehicle_tokens):
        return "road_or_vehicle_semantic_event", MIN_VEHICLE_EVENT_VEHICLES
    return "uav_only_or_context", MIN_ORDINARY_VEHICLES


def _has_adverse_weather(scenario_id: str, script: dict[str, Any]) -> bool:
    scenario_text = str(scenario_id).lower()
    if any(token in scenario_text for token in ("rain", "fog", "wind", "smoke")):
        return True
    parameters = dict(script.get("parameters") or {})
    contract = dict(parameters.get("semantic_event_contract") or {})
    candidates = [
        parameters.get("weather"),
        parameters.get("weather_condition"),
        contract.get("weather"),
    ]
    weather_profile = parameters.get("weather_profile")
    if isinstance(weather_profile, dict):
        candidates.extend([weather_profile.get("condition"), weather_profile.get("weather")])
    adverse_values = {"rain", "fog", "wind", "dusk", "heat", "light smoke", "smoke", "low_visibility"}
    return any(str(value or "").strip().lower() in adverse_values for value in candidates)


def _traffic_flow_mode(scenario_id: str, script: dict[str, Any], scenario_class: str) -> str:
    text = _json_text(script)
    if any(token in text for token in ("ambulance", "police", "emergency", "yield_vehicle", "civilian_yield")):
        return "emergency_corridor_with_yielding_flow"
    if any(token in text for token in ("congestion", "gridlock", "traffic_jam", "all_red", "lane_closure", "roadwork")):
        return "queue_congestion"
    if _has_adverse_weather(scenario_id, script) or any(token in text for token in ("weather_speed_degradation", "slowdown", "low_visibility")):
        return "weather_slowdown_flow"
    if scenario_id.startswith(("L2", "L3", "L4", "X")) or scenario_class != "uav_only_or_context":
        return "intersection_event_flow"
    return "context_free_flow"


def _flow_group_for_vehicle(flow_mode: str, direction_role: str, role: str, index: int) -> str:
    if flow_mode == "queue_congestion":
        return f"queue_{direction_role}_{index // 4:02d}"
    if flow_mode == "weather_slowdown_flow":
        return f"weather_slow_{direction_role}_{index // 5:02d}"
    if flow_mode == "emergency_corridor_with_yielding_flow":
        if "ambulance" in role or "police" in role or "emergency" in role or index % 5 == 0:
            return f"emergency_corridor_{direction_role}"
        return f"yielding_platoon_{direction_role}_{index // 4:02d}"
    if flow_mode == "intersection_event_flow":
        return f"intersection_platoon_{direction_role}_{index // 5:02d}"
    return f"context_flow_{direction_role}_{index // 6:02d}"


def _flow_speed_target(profile: SeedTrafficProfile, flow_mode: str, index: int) -> float:
    if flow_mode == "queue_congestion":
        return 2.0 + (index % 3) * 0.5
    if flow_mode == "weather_slowdown_flow":
        return 4.0 + (index % 3) * 0.6
    if flow_mode == "emergency_corridor_with_yielding_flow":
        return 9.5 if index % 5 == 0 else 3.5 + (index % 2) * 0.7
    return max(2.5, profile.target_speed_mps + ((index % 3) - 1) * 0.7)


def _event_ticks(event_def: dict[str, Any]) -> list[int]:
    ticks: list[int] = []
    for key, value in event_def.items():
        if not str(key).endswith("tick") and key not in {"tick", "start_tick", "end_tick"}:
            continue
        try:
            ticks.append(int(value))
        except (TypeError, ValueError):
            continue
    return [max(0, min(DURATION_TICKS, tick)) for tick in ticks]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _episode_vehicle_event_bindings(
    episode_dir: Path,
    source_vehicle_ids: Sequence[str],
    seed_index: int,
) -> list[dict[str, Any]]:
    source_set = {str(source_id) for source_id in source_vehicle_ids if str(source_id)}
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for path_name in ("event_realization.jsonl", "event_trace.jsonl"):
        for row in _read_jsonl(Path(episode_dir) / path_name):
            target_ids = [str(item) for item in row.get("target_ids") or [] if str(item) in source_set]
            scope = dict(row.get("scope") or {})
            for item in scope.get("entities") or []:
                if str(item) in source_set and str(item) not in target_ids:
                    target_ids.append(str(item))
            if not target_ids:
                continue
            event_id = str(row.get("event_id") or row.get("chain_id") or row.get("topic") or row.get("source_event_id") or "").strip()
            if not event_id:
                continue
            ticks = _event_ticks(dict(row))
            payload = row.get("payload")
            if isinstance(payload, dict):
                ticks.extend(_event_ticks(payload))
            ticks = sorted(set(max(0, min(DURATION_TICKS, int(tick))) for tick in ticks))
            for source_id in target_ids:
                key = (event_id, source_id)
                binding = merged.setdefault(
                    key,
                    {
                        "event_id": event_id,
                        "source_entity_id": source_id,
                        "vehicle_id": semantic_internal_vehicle_id(source_id, seed_index),
                        "binding_source": "episode_event_trace_vehicle_target",
                        "evidence_ticks": [],
                    },
                )
                binding["evidence_ticks"] = sorted(set([*binding.get("evidence_ticks", []), *ticks]))
    return list(merged.values())


def _protected_bbox_for_event(points: Sequence[Sequence[float]], padding_m: float) -> list[float]:
    return _bbox_from_points(points, padding_m=float(padding_m))


def _protected_edges_for_bbox(planner: SumoGroundFlowPlanner, bbox: Sequence[float]) -> list[str]:
    return sorted(
        edge.edge_id
        for edge in planner.edges.values()
        if _edge_allows_vehicle(edge) and len(edge.shape_xy) >= 2 and _edge_intersects_bbox(edge, bbox)
    )


def _row_tick_range(row: dict[str, Any], pre_margin: int = PRE_EVENT_VISIBLE_MARGIN_TICKS, post_margin: int = POST_EVENT_VISIBLE_MARGIN_TICKS) -> list[int]:
    ticks = _event_ticks(row)
    payload = row.get("payload")
    if isinstance(payload, dict):
        ticks.extend(_event_ticks(payload))
    for action in row.get("action_realizations") or []:
        if isinstance(action, dict):
            ticks.extend(_event_ticks(action))
    if not ticks:
        return [0, DURATION_TICKS]
    start = max(0, min(ticks) - int(pre_margin))
    end = min(DURATION_TICKS, max(ticks) + int(post_margin))
    return [start, end]


def _event_semantic_vehicle_source_ids(
    event_def: dict[str, Any],
    row: dict[str, Any],
    source_vehicle_set: set[str],
) -> list[str]:
    if not source_vehicle_set:
        return []
    source_ids: set[str] = set()
    target_ids = [str(item) for item in row.get("target_ids") or [] if str(item)]
    source_ids.update(target_id for target_id in target_ids if target_id in source_vehicle_set)
    actions = [
        *[dict(action) for action in event_def.get("actions") or [] if isinstance(action, dict)],
        *[dict(action) for action in row.get("action_realizations") or [] if isinstance(action, dict)],
    ]
    for action in actions:
        entity_id = str(action.get("entity_id") or "")
        if entity_id in source_vehicle_set:
            source_ids.add(entity_id)
    snapshots = row.get("source_truth_snapshots_by_tick")
    if isinstance(snapshots, dict):
        for by_entity in snapshots.values():
            if not isinstance(by_entity, dict):
                continue
            source_ids.update(str(entity_id) for entity_id in by_entity.keys() if str(entity_id) in source_vehicle_set)
    return sorted(source_ids)


def _event_semantic_vehicle_points(
    event_def: dict[str, Any],
    row: dict[str, Any],
    semantic_source_ids: Sequence[str],
) -> list[list[float]]:
    source_set = {str(source_id) for source_id in semantic_source_ids if str(source_id)}
    if not source_set:
        return []
    points: list[list[float]] = []
    snapshots = row.get("source_truth_snapshots_by_tick")
    if isinstance(snapshots, dict):
        for by_entity in snapshots.values():
            if not isinstance(by_entity, dict):
                continue
            for source_id in sorted(source_set):
                snapshot = by_entity.get(source_id)
                if not isinstance(snapshot, dict) or snapshot.get("present") is False:
                    continue
                point = _position3(snapshot.get("position_enu_m"))
                if point is not None:
                    points.append(point)
    point_keys = ("position_enu_m", "terminal_enu_m", "target_enu_m", "start_enu_m", "end_enu_m", "anchor_enu_m")
    for action in [
        *[dict(item) for item in event_def.get("actions") or [] if isinstance(item, dict)],
        *[dict(item) for item in row.get("action_realizations") or [] if isinstance(item, dict)],
    ]:
        if str(action.get("entity_id") or "") not in source_set:
            continue
        for key in point_keys:
            point = _position3(action.get(key))
            if point is not None:
                points.append(point)
    return _dedupe_points(points)


def _event_vehicle_related_review(
    event_id: str,
    event_def: dict[str, Any],
    row: dict[str, Any],
    *,
    source_vehicle_set: set[str],
) -> dict[str, Any]:
    target_ids = [str(item) for item in row.get("target_ids") or [] if str(item)]
    script_actions = [dict(action) for action in event_def.get("actions") or [] if isinstance(action, dict)]
    realized_actions = [dict(action) for action in row.get("action_realizations") or [] if isinstance(action, dict)]
    action_types = {
        str(action.get("type") or action.get("action_type") or "")
        for action in [*script_actions, *realized_actions]
        if str(action.get("type") or action.get("action_type") or "")
    }
    semantic_event_vehicle_ids = _event_semantic_vehicle_source_ids(event_def, row, source_vehicle_set)
    reasons: list[str] = []
    if semantic_event_vehicle_ids:
        reasons.append(f"semantic vehicle participant(s): {semantic_event_vehicle_ids[:6]}")
        reasons.append("protection follows realized semantic vehicle spacetime")
    elif source_vehicle_set:
        reasons.append("no semantic vehicle participant in this event row")
    else:
        reasons.append("no semantic vehicle actor in this episode")
    return {
        "event_id": event_id,
        "target_ids": target_ids,
        "vehicle_related": bool(semantic_event_vehicle_ids),
        "should_protect_traffic": bool(semantic_event_vehicle_ids),
        "semantic_event_vehicle_source_ids": semantic_event_vehicle_ids,
        "action_types": sorted(action_types),
        "reason": "; ".join(reasons),
    }


def _semantic_traffic_constraints(
    *,
    episode_dir: Path,
    script: dict[str, Any],
    scene_entities: Sequence[dict[str, Any]],
    semantic_source_vehicle_ids: Sequence[str],
    seed_index: int,
    planner: SumoGroundFlowPlanner,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semantic_source_vehicle_set = {str(source_id) for source_id in semantic_source_vehicle_ids if str(source_id)}
    event_defs = {
        str(event.get("event_id") or event.get("topic") or ""): dict(event)
        for event in script.get("events") or []
        if isinstance(event, dict)
    }
    constraints: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for row in _read_jsonl(Path(episode_dir) / "event_realization.jsonl"):
        event_id = str(row.get("event_id") or row.get("topic") or "").strip()
        if not event_id:
            continue
        event_def = event_defs.get(event_id, {})
        review = _event_vehicle_related_review(
            event_id,
            event_def,
            row,
            source_vehicle_set=semantic_source_vehicle_set,
        )
        review["traffic_case"] = (
            "semantic_vehicle_protected_flow"
            if semantic_source_vehicle_set
            else "background_only_no_semantic_vehicle_flow"
        )
        review["semantic_source_vehicle_ids"] = sorted(semantic_source_vehicle_set)
        if not semantic_source_vehicle_set:
            review["vehicle_related"] = False
            review["should_protect_traffic"] = False
            review["reason"] = (
                f"{review.get('reason')}; no semantic vehicle actor in this episode, "
                "so no protected traffic window is generated"
            )
        if not semantic_source_vehicle_set:
            reviews.append(review)
            continue
        vehicle_targets = [str(item) for item in review.get("semantic_event_vehicle_source_ids") or [] if str(item) in semantic_source_vehicle_set]
        if not review["should_protect_traffic"]:
            reviews.append(review)
            continue
        points = _event_semantic_vehicle_points(event_def, row, vehicle_targets)
        if not points:
            review["should_protect_traffic"] = False
            review["reason"] = f"{review.get('reason')}; no realized semantic vehicle spacetime points found"
            reviews.append(review)
            continue
        padding = 15.0 if vehicle_targets else 8.0
        bbox = _protected_bbox_for_event(points, padding)
        allowed_ids = sorted(semantic_internal_vehicle_id(source_id, seed_index) for source_id in vehicle_targets)
        if not allowed_ids:
            review["should_protect_traffic"] = False
            review["reason"] = f"{review.get('reason')}; no semantic allowed vehicle ids resolved"
            reviews.append(review)
            continue
        constraints.append(
            {
                "constraint_id": f"{_safe_token(event_id)}_protected_traffic_window",
                "event_id": event_id,
                "tick_range": _row_tick_range(row),
                "protected_bbox_enu_m": bbox,
                "protected_edges": _protected_edges_for_bbox(planner, bbox),
                "allowed_vehicle_ids": allowed_ids,
                "background_policy": "route_must_avoid_protected_edges",
                "source": "event_realization_vehicle_window",
                "vehicle_related": bool(review["vehicle_related"]),
                "vehicle_related_reason": str(review["reason"]),
            }
        )
        reviews.append(review)

    return constraints, reviews


def _state_patch_flag_is_true(value: Any, flag_name: str) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) == str(flag_name) and child is True:
                return True
            if _state_patch_flag_is_true(child, flag_name):
                return True
    elif isinstance(value, list):
        return any(_state_patch_flag_is_true(child, flag_name) for child in value)
    return False


def _semantic_dispatch_motion_lifecycle(
    *,
    episode_dir: Path,
    script: dict[str, Any],
    source_entity_id: str,
) -> dict[str, Any] | None:
    """Resolve orchestration-gated responder motion from authored and realized actions.

    The same authored event must contain an exact responder ``move_entity`` action
    plus either a ``dispatch_active`` runtime orchestration signal or an exact
    responder ``spawn_entity`` action.  A successful gate realization controls
    only when SUMO may make the responder physically present.  Neither that gate
    nor the event label is semantic truth.  Semantic dispatch truth remains
    grounded in the successful movement realization and its physical
    deployment/approach samples.
    """

    source_id = str(source_entity_id or "")
    if not source_id:
        return None
    rows_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(Path(episode_dir) / "event_realization.jsonl"):
        event_id = str(row.get("event_id") or row.get("topic") or "").strip()
        if event_id:
            rows_by_event.setdefault(event_id, []).append(row)

    for event_def in script.get("events") or []:
        if not isinstance(event_def, dict):
            continue
        event_id = str(event_def.get("event_id") or event_def.get("topic") or "").strip()
        if not event_id:
            continue
        authored_actions = [dict(action) for action in event_def.get("actions") or [] if isinstance(action, dict)]
        movement_actions = [
            action
            for action in authored_actions
            if str(action.get("type") or action.get("action_type") or "") == "move_entity"
            and str(action.get("entity_id") or "") == source_id
        ]
        dispatch_actions = [
            action
            for action in authored_actions
            if str(action.get("type") or action.get("action_type") or "") == "set_runtime_state"
            and _state_patch_flag_is_true(action.get("state_patch"), "dispatch_active")
        ]
        spawn_actions = [
            action
            for action in authored_actions
            if str(action.get("type") or action.get("action_type") or "") == "spawn_entity"
            and str(action.get("entity_id") or "") == source_id
        ]
        if not movement_actions or (not dispatch_actions and not spawn_actions):
            continue

        movement_action_ids = {
            str(action.get("action_id") or "")
            for action in movement_actions
            if str(action.get("action_id") or "")
        }
        if not movement_action_ids:
            raise RuntimeError(
                f"semantic vehicle {source_id} dispatch-gated movement in {event_id} has no authored action_id"
            )
        dispatch_action_ids = {
            str(action.get("action_id") or "")
            for action in dispatch_actions
            if str(action.get("action_id") or "")
        }
        spawn_action_ids = {
            str(action.get("action_id") or "")
            for action in spawn_actions
            if str(action.get("action_id") or "")
        }
        if not dispatch_action_ids and not spawn_action_ids:
            raise RuntimeError(
                f"semantic vehicle {source_id} orchestration gate in {event_id} has no authored action_id"
            )
        matching_rows: list[dict[str, Any]] = []
        for row in rows_by_event.get(event_id, []):
            realized_actions = [dict(action) for action in row.get("action_realizations") or [] if isinstance(action, dict)]
            realized_movement_ids = {
                str(action.get("action_id") or "")
                for action in realized_actions
                if str(action.get("action_type") or action.get("type") or "") == "move_entity"
                and str(action.get("entity_id") or "") == source_id
                and str(action.get("status") or "") == "ok"
            }
            if movement_action_ids & realized_movement_ids:
                matching_rows.append(row)
        if not matching_rows:
            raise RuntimeError(
                f"semantic vehicle {source_id} has authored dispatch-gated movement in {event_id} "
                "but no successful movement realization"
            )
        if len(matching_rows) != 1:
            raise RuntimeError(
                f"semantic vehicle {source_id} has {len(matching_rows)} dispatch-gated movement realizations "
                f"for {event_id}; exactly one is required"
            )
        row = matching_rows[0]
        realized_actions = [dict(action) for action in row.get("action_realizations") or [] if isinstance(action, dict)]
        realized_movement_actions = [
            action
            for action in realized_actions
            if str(action.get("action_id") or "") in movement_action_ids
            and str(action.get("action_type") or action.get("type") or "") == "move_entity"
            and str(action.get("entity_id") or "") == source_id
            and str(action.get("status") or "") == "ok"
        ]
        successful_movement_action_ids = {
            str(action.get("action_id") or "") for action in realized_movement_actions
        }
        if not successful_movement_action_ids:
            raise RuntimeError(
                f"semantic vehicle {source_id} has no exact successful movement action in {event_id}"
            )
        realized_dispatch_actions = [
            action
            for action in realized_actions
            if str(action.get("action_id") or "") in dispatch_action_ids
            and str(action.get("action_type") or action.get("type") or "") == "set_runtime_state"
            and str(action.get("status") or "") == "ok"
        ]
        realized_spawn_actions = [
            action
            for action in realized_actions
            if str(action.get("action_id") or "") in spawn_action_ids
            and str(action.get("action_type") or action.get("type") or "") == "spawn_entity"
            and str(action.get("entity_id") or "") == source_id
            and str(action.get("status") or "") == "ok"
        ]
        if len(realized_spawn_actions) > 1 or len(realized_dispatch_actions) > 1:
            raise RuntimeError(
                f"semantic vehicle {source_id} has ambiguous successful orchestration gates in {event_id}"
            )
        # When both actions succeed, the authored dispatch-state transition is
        # the orchestration gate for the semantic response.  The spawn is only
        # a physical SUMO presence operation and must not replace the event's
        # explicit dispatch evidence.  A spawn remains the gate only for
        # scripts that do not author a dispatch-state action.
        if len(realized_dispatch_actions) == 1:
            activation_realization = realized_dispatch_actions[0]
            activation_action_type = "set_runtime_state"
            activation_action_ids = {str(activation_realization.get("action_id") or "")}
            activation_entity_ids = {
                str(activation_realization.get("entity_id") or "")
            }
            activation_source_policy = "successful_dispatch_active_runtime_orchestration_realization"
        elif len(realized_spawn_actions) == 1:
            activation_realization = realized_spawn_actions[0]
            activation_action_type = "spawn_entity"
            activation_action_ids = {str(activation_realization.get("action_id") or "")}
            activation_entity_ids = {source_id}
            activation_source_policy = "successful_responder_spawn_entity_realization"
        else:
            raise RuntimeError(
                f"semantic vehicle {source_id} requires one successful responder spawn or dispatch-state "
                f"orchestration realization in {event_id}"
            )
        activation_evidence_tick = _optional_int(activation_realization.get("evidence_tick"))
        if activation_evidence_tick is None:
            raise RuntimeError(
                f"semantic vehicle {source_id} {activation_action_type} orchestration realization in "
                f"{event_id} has no evidence_tick"
            )
        activation_evidence_tick = max(0, min(DURATION_TICKS, int(activation_evidence_tick)))
        if activation_evidence_tick >= DURATION_TICKS:
            raise RuntimeError(
                f"semantic vehicle {source_id} orchestration evidence tick {activation_evidence_tick} "
                "leaves no formal tick for physical activation"
            )

        source_sample_ticks: set[int] = set()
        snapshots = row.get("source_truth_snapshots_by_tick")
        if isinstance(snapshots, dict):
            for tick_text, by_entity in snapshots.items():
                if not isinstance(by_entity, dict):
                    continue
                snapshot = by_entity.get(source_id)
                if not isinstance(snapshot, dict) or snapshot.get("present") is False:
                    continue
                if _position3(snapshot.get("position_enu_m")) is not None:
                    source_sample_ticks.add(int(tick_text))
        for action in realized_movement_actions:
            if _position3(action.get("terminal_enu_m")) is None:
                continue
            terminal_tick = _optional_int(
                action.get("terminal_tick")
                or action.get("result_tick")
                or action.get("evidence_tick")
            )
            if terminal_tick is not None:
                source_sample_ticks.add(int(terminal_tick))
        if not source_sample_ticks:
            raise RuntimeError(
                f"semantic vehicle {source_id} dispatch-gated movement in {event_id} has no physical position samples"
            )

        movement_speeds = [
            float(action.get("velocity_mps") or 0.0)
            for action in movement_actions
            if str(action.get("action_id") or "") in successful_movement_action_ids
            and float(action.get("velocity_mps") or 0.0) > 0.0
        ]
        if not movement_speeds:
            raise RuntimeError(
                f"semantic vehicle {source_id} dispatch-gated movement in {event_id} has no positive velocity_mps"
            )
        movement_speed_mps = min(movement_speeds)
        approach_duration_ticks = max(
            1,
            int(math.ceil(SEMANTIC_DISPATCH_APPROACH_DISTANCE_M / movement_speed_mps * TICK_HZ)),
        )
        control_start_tick = activation_evidence_tick + 1
        first_source_tick = min(source_sample_ticks)
        first_authority_sample_tick = control_start_tick + approach_duration_ticks
        sample_tick_offset = max(0, first_authority_sample_tick - first_source_tick)
        retimed_sample_ticks = sorted(int(tick) + sample_tick_offset for tick in source_sample_ticks)
        if not retimed_sample_ticks or max(retimed_sample_ticks) > DURATION_TICKS:
            raise RuntimeError(
                f"semantic vehicle {source_id} dispatch-gated movement in {event_id} exceeds formal tick "
                f"{DURATION_TICKS} after orchestration-aligned retiming"
            )
        control_end_tick = min(DURATION_TICKS, max(retimed_sample_ticks) + POST_EVENT_VISIBLE_MARGIN_TICKS)
        if activation_action_type == "spawn_entity":
            activation_specific_fields = {
                "spawn_entity_action_ids": sorted(activation_action_ids),
                "spawn_entity_ids": sorted(activation_entity_ids),
                "spawn_entity_evidence_tick": int(activation_evidence_tick),
            }
        else:
            activation_specific_fields = {
                "dispatch_state_action_ids": sorted(activation_action_ids),
                "dispatch_state_entity_ids": sorted(activation_entity_ids),
                "dispatch_state_evidence_tick": int(activation_evidence_tick),
                "dispatch_state_evidence_role": "sumo_orchestration_activation_only_not_semantic_truth",
            }
        return {
            "policy": "responder_orchestration_gated_physical_motion_authority_v3",
            "source_policy": (
                "successful_move_entity_physical_samples_gated_by_" + activation_source_policy
            ),
            "activation_orchestration_evidence_role": "sumo_activation_only_not_semantic_truth",
            "activation_orchestration_action_type": activation_action_type,
            "activation_orchestration_action_ids": sorted(activation_action_ids),
            "activation_orchestration_entity_ids": sorted(activation_entity_ids),
            "activation_orchestration_evidence_tick": int(activation_evidence_tick),
            "semantic_truth_authority": "successful_move_entity_physical_deployment_and_continuous_approach",
            "authority_event_ids": [event_id],
            "movement_action_ids": sorted(successful_movement_action_ids),
            **activation_specific_fields,
            "strict_absent_before_tick": int(activation_evidence_tick),
            "control_start_tick": int(control_start_tick),
            "control_end_tick": int(control_end_tick),
            "arrival_not_before_tick": int(first_authority_sample_tick),
            "source_sample_ticks": sorted(int(tick) for tick in source_sample_ticks),
            "retimed_sample_ticks": retimed_sample_ticks,
            "sample_tick_offset": int(sample_tick_offset),
            "approach_distance_m": float(SEMANTIC_DISPATCH_APPROACH_DISTANCE_M),
            "approach_duration_ticks": int(approach_duration_ticks),
            "physical_motion_speed_mps": round(float(movement_speed_mps), 6),
        }
    return None


def _semantic_control_samples(
    *,
    episode_dir: Path,
    source_entity_id: str,
    authority_event_ids: Sequence[str] = (),
    movement_action_ids: Sequence[str] = (),
    tick_offset: int = 0,
) -> list[dict[str, Any]]:
    samples: dict[int, dict[str, Any]] = {}
    authority_event_set = {str(event_id) for event_id in authority_event_ids if str(event_id)}
    movement_action_set = {str(action_id) for action_id in movement_action_ids if str(action_id)}
    rows = _read_jsonl(Path(episode_dir) / "event_realization.jsonl")
    for row in rows:
        event_id = str(row.get("event_id") or row.get("topic") or "").strip()
        if authority_event_set and event_id not in authority_event_set:
            continue
        realized_actions = [
            dict(action)
            for action in row.get("action_realizations") or []
            if isinstance(action, dict)
        ]
        successful_movement_actions = [
            action
            for action in realized_actions
            if str(action.get("action_type") or action.get("type") or "") == "move_entity"
            and str(action.get("entity_id") or "") == source_entity_id
            and str(action.get("status") or "") == "ok"
            and str(action.get("action_id") or "")
            and (
                not movement_action_set
                or str(action.get("action_id") or "") in movement_action_set
            )
        ]
        if movement_action_set and not successful_movement_actions:
            continue
        row_movement_action_ids = sorted(
            {str(action.get("action_id") or "") for action in successful_movement_actions}
        )
        snapshots = row.get("source_truth_snapshots_by_tick")
        if isinstance(snapshots, dict):
            for tick_text, by_entity in snapshots.items():
                if not isinstance(by_entity, dict):
                    continue
                snapshot = by_entity.get(source_entity_id)
                if not isinstance(snapshot, dict) or snapshot.get("present") is False:
                    continue
                point = _position3(snapshot.get("position_enu_m"))
                if point is not None:
                    source_tick = int(tick_text)
                    control_tick = source_tick + int(tick_offset)
                    if not 0 <= control_tick <= DURATION_TICKS:
                        raise RuntimeError(
                            f"semantic vehicle {source_entity_id} sample tick {source_tick} becomes "
                            f"out-of-range control tick {control_tick}"
                        )
                    samples[control_tick] = {
                        "position_enu_m": [point[0], point[1], point[2]],
                        "source_tick": source_tick,
                        "authority_event_id": event_id,
                        "movement_action_ids": row_movement_action_ids,
                    }
        for action in successful_movement_actions:
            terminal = _position3(action.get("terminal_enu_m"))
            if terminal is not None:
                source_tick = int(action.get("terminal_tick") or action.get("evidence_tick") or action.get("result_tick") or row.get("evidence_tick") or 0)
                control_tick = source_tick + int(tick_offset)
                if not 0 <= control_tick <= DURATION_TICKS:
                    raise RuntimeError(
                        f"semantic vehicle {source_entity_id} terminal tick {source_tick} becomes "
                        f"out-of-range control tick {control_tick}"
                    )
                samples[control_tick] = {
                    "position_enu_m": terminal,
                    "source_tick": source_tick,
                    "authority_event_id": event_id,
                    "movement_action_ids": [str(action.get("action_id") or "")],
                }
    if not samples:
        return []
    ordered_ticks = sorted(samples)
    return [
        {
            "tick": int(tick),
            "position_enu_m": [round(float(value), 6) for value in samples[tick]["position_enu_m"]],
            "source_tick": int(samples[tick]["source_tick"]),
            "authority_event_id": str(samples[tick]["authority_event_id"]),
            "movement_action_ids": [str(action_id) for action_id in samples[tick]["movement_action_ids"]],
            "retimed_by_ticks": int(tick_offset),
        }
        for tick in ordered_ticks
    ]


def _semantic_lane_controls(
    *,
    planner: SumoGroundFlowPlanner,
    raw_samples: Sequence[dict[str, Any]],
    semantic_corridor: dict[str, Any] | None,
    source_entity_id: str = "",
    active_start_tick: int | None = None,
    active_end_tick: int | None = None,
    visible_bbox_enu_m: Sequence[float] = (),
    enforce_active_start_tick: bool = False,
    hold_after_last_sample: bool = False,
) -> list[dict[str, Any]]:
    if not raw_samples or not semantic_corridor:
        return []
    edge_id = str(semantic_corridor.get("edge_id") or "")
    edge = planner.edges.get(edge_id)
    if edge is None:
        return []
    lane_index = int(semantic_corridor.get("lane_index") or 0)
    configured_projection_tolerance_m = float(
        semantic_corridor.get("projection_error_tolerance_m") or MAX_SEMANTIC_LANE_PROJECTION_ERROR_M
    )
    if configured_projection_tolerance_m > MAX_SEMANTIC_LANE_PROJECTION_ERROR_M:
        raise RuntimeError(
            f"semantic vehicle {source_entity_id or '<unknown>'} corridor configures projection tolerance "
            f"{configured_projection_tolerance_m:.3f}m above the physical threshold "
            f"{MAX_SEMANTIC_LANE_PROJECTION_ERROR_M:.3f}m"
        )
    projection_error_tolerance_m = MAX_SEMANTIC_LANE_PROJECTION_ERROR_M
    nearest_authority_by_tick: dict[int, tuple[str, float]] = {}
    nearest_authority_sequence: list[tuple[int, str, float]] = []
    for sample in raw_samples:
        tick = int(sample.get("tick") or 0)
        point = _position3(sample.get("position_enu_m"))
        if point is None:
            continue
        nearest_ranked_edges = _rank_semantic_edges_for_points(planner, [point])
        if not nearest_ranked_edges:
            continue
        _nearest_score, nearest_distance, _nearest_mean, nearest_edge_id, _nearest_edge = (
            nearest_ranked_edges[0]
        )
        nearest_authority_by_tick[tick] = (str(nearest_edge_id), float(nearest_distance))
        if not nearest_authority_sequence or nearest_authority_sequence[-1][1] != str(nearest_edge_id):
            nearest_authority_sequence.append((tick, str(nearest_edge_id), float(nearest_distance)))
    if (
        len(nearest_authority_sequence) > 1
        and all(
            distance <= MAX_SEMANTIC_LANE_PROJECTION_ERROR_M
            for _tick, _edge_id, distance in nearest_authority_sequence
        )
    ):
        transition_text = " -> ".join(
            f"{authority_edge_id}@{authority_tick}"
            for authority_tick, authority_edge_id, _distance in nearest_authority_sequence
        )
        raise RuntimeError(
            f"semantic vehicle {source_entity_id or '<unknown>'} authored physical path requires segmented "
            f"lane authority ({transition_text}); refusing to disguise legal segment transitions as fixed-lane "
            f"control on {edge_id}"
        )
    projected_by_tick: dict[int, dict[str, Any]] = {}
    for sample in raw_samples:
        tick = int(sample.get("tick") or 0)
        point = _position3(sample.get("position_enu_m"))
        if point is None:
            continue
        projection = _project_point_to_edge(edge, point)
        distance = float(projection["distance_m"])
        if distance > projection_error_tolerance_m:
            nearest_authority = nearest_authority_by_tick.get(tick)
            if nearest_authority is None:
                raise RuntimeError(
                    f"semantic vehicle {source_entity_id or '<unknown>'} authored physical path at control tick "
                    f"{tick} (source tick {sample.get('source_tick')}) has no legal SUMO vehicle lane; "
                    "refusing an ungrounded lane snap"
                )
            nearest_edge_id, nearest_distance = nearest_authority
            if float(nearest_distance) > MAX_SEMANTIC_LANE_PROJECTION_ERROR_M:
                raise RuntimeError(
                    f"semantic vehicle {source_entity_id or '<unknown>'} authored physical path leaves legal "
                    f"SUMO vehicle lanes at control tick {tick} (source tick {sample.get('source_tick')}): "
                    f"nearest legal edge {nearest_edge_id} is {float(nearest_distance):.3f}m away, above "
                    f"physical threshold {MAX_SEMANTIC_LANE_PROJECTION_ERROR_M:.3f}m; refusing to relax "
                    "the threshold or fabricate an observed lane position"
                )
            raise RuntimeError(
                f"semantic vehicle {source_entity_id or '<unknown>'} authored physical path requires "
                f"segmented lane authority at control tick {tick} (source tick {sample.get('source_tick')}): "
                f"fixed edge {edge_id} is {distance:.3f}m away while nearest legal edge "
                f"{nearest_edge_id} is {float(nearest_distance):.3f}m away; refusing to disguise the "
                "segment transition as fixed-lane control"
            )
        projected_by_tick[tick] = {
            "tick": tick,
            "lane_position_m": float(projection["lane_position_m"]),
            "position_enu_m": [float(value) for value in projection["xy_enu_m"]],
            "projection_error_m": distance,
            "source_tick": sample.get("source_tick"),
            "authority_event_id": str(sample.get("authority_event_id") or ""),
            "movement_action_ids": [str(action_id) for action_id in sample.get("movement_action_ids") or []],
            "retimed_by_ticks": int(sample.get("retimed_by_ticks") or 0),
            "observed_physical_sample": sample.get("source_tick") is not None,
            "derivation_policy": "authoritative_physical_movement_sample_v1",
            "derivation_source_ticks": (
                [int(sample["source_tick"])] if sample.get("source_tick") is not None else []
            ),
        }
    if not projected_by_tick:
        return []

    def derived_provenance(policy: str, *parents: dict[str, Any]) -> dict[str, Any]:
        event_ids = {
            str(parent.get("authority_event_id") or "")
            for parent in parents
            if str(parent.get("authority_event_id") or "")
        }
        action_id_sets = [
            {
                str(action_id)
                for action_id in parent.get("movement_action_ids") or []
                if str(action_id)
            }
            for parent in parents
        ]
        populated_action_id_sets = [action_ids for action_ids in action_id_sets if action_ids]
        if enforce_active_start_tick:
            if len(event_ids) != 1 or not populated_action_id_sets:
                raise RuntimeError(
                    f"semantic vehicle {source_entity_id or '<unknown>'} synthetic control lacks unique "
                    "event/movement authority"
                )
            if any(action_ids != populated_action_id_sets[0] for action_ids in populated_action_id_sets[1:]):
                raise RuntimeError(
                    f"semantic vehicle {source_entity_id or '<unknown>'} synthetic control has inconsistent "
                    "movement action authority"
                )
        movement_ids = sorted(set().union(*populated_action_id_sets)) if populated_action_id_sets else []
        derivation_source_ticks = sorted(
            {
                int(source_tick)
                for parent in parents
                for source_tick in (
                    [parent.get("source_tick")]
                    if parent.get("source_tick") is not None
                    else list(parent.get("derivation_source_ticks") or [])
                )
                if source_tick is not None
            }
        )
        return {
            "source_tick": None,
            "authority_event_id": sorted(event_ids)[0] if event_ids else "",
            "movement_action_ids": movement_ids,
            "retimed_by_ticks": max(
                (int(parent.get("retimed_by_ticks") or 0) for parent in parents),
                default=0,
            ),
            "observed_physical_sample": False,
            "derivation_policy": str(policy),
            "derivation_source_ticks": derivation_source_ticks,
        }

    ticks = sorted(projected_by_tick)
    first_tick = ticks[0]
    first_s = float(projected_by_tick[first_tick]["lane_position_m"])
    travel_sign = 1.0
    previous_s = first_s
    for tick in ticks[1:]:
        current_s = float(projected_by_tick[tick]["lane_position_m"])
        if abs(current_s - previous_s) > 0.25:
            travel_sign = 1.0 if current_s > previous_s else -1.0
            break
        previous_s = current_s
    if travel_sign < 0.0:
        raise RuntimeError(
            f"semantic vehicle {source_entity_id or '<unknown>'} authored physical path moves opposite "
            f"the legal direction of SUMO edge {edge.edge_id}; refusing to reverse the observed sample "
            "order or fabricate forward lane positions"
        )
    active_tick = max(0, first_tick - PRE_EVENT_VISIBLE_MARGIN_TICKS - 5)
    if active_start_tick is not None:
        requested_active_tick = max(0, min(DURATION_TICKS, int(active_start_tick)))
        if enforce_active_start_tick:
            if requested_active_tick > first_tick:
                raise RuntimeError(
                    f"semantic vehicle {source_entity_id or '<unknown>'} strict activation tick "
                    f"{requested_active_tick} is after its first physical sample tick {first_tick}"
                )
            active_tick = max(active_tick, requested_active_tick)
        else:
            active_tick = max(0, min(active_tick, requested_active_tick))
    entry_distance_target_m = 20.0 if "yield" in str(source_entity_id).lower() else 30.0
    entry_distance_m = min(entry_distance_target_m, max(8.0, edge.length_m * 0.45))
    entry_s = first_s - travel_sign * entry_distance_m
    entry_s = max(0.5, min(max(0.5, edge.length_m - 0.5), entry_s))
    if visible_bbox_enu_m and _point_in_bbox(_point_at_edge_s(edge, entry_s), visible_bbox_enu_m):
        scan_s = entry_s
        for _ in range(int(math.ceil(edge.length_m * 2.0)) + 2):
            scan_s = max(0.5, min(max(0.5, edge.length_m - 0.5), scan_s - travel_sign * 0.5))
            if not _point_in_bbox(_point_at_edge_s(edge, scan_s), visible_bbox_enu_m):
                entry_s = scan_s
                break
    first_authoritative_sample = projected_by_tick[first_tick]
    for tick in range(active_tick, first_tick):
        alpha = 0.0 if first_tick == active_tick else (tick - active_tick) / float(first_tick - active_tick)
        lane_s = entry_s + (first_s - entry_s) * alpha
        x, y = _point_at_edge_s(edge, lane_s)
        projected_by_tick.setdefault(
            tick,
            {
                "tick": tick,
                "lane_position_m": round(float(lane_s), 6),
                "position_enu_m": [round(float(x), 6), round(float(y), 6), 0.0],
                "projection_error_m": 0.0,
                "synthetic_entry_sample": True,
                **derived_provenance(
                    "synthetic_lane_entry_from_first_authoritative_movement_sample_v1",
                    first_authoritative_sample,
                ),
            },
        )

    dense_ticks = sorted(projected_by_tick)
    for start_tick, end_tick in zip(dense_ticks, dense_ticks[1:]):
        if end_tick - start_tick <= 1:
            continue
        start_s = float(projected_by_tick[start_tick]["lane_position_m"])
        end_s = float(projected_by_tick[end_tick]["lane_position_m"])
        interpolation_provenance = derived_provenance(
            "linear_lane_interpolation_between_authoritative_movement_samples_v1",
            projected_by_tick[start_tick],
            projected_by_tick[end_tick],
        )
        for tick in range(start_tick + 1, end_tick):
            alpha = (tick - start_tick) / float(end_tick - start_tick)
            lane_s = start_s + (end_s - start_s) * alpha
            x, y = _point_at_edge_s(edge, lane_s)
            projected_by_tick.setdefault(
                tick,
                {
                    "tick": tick,
                    "lane_position_m": round(float(lane_s), 6),
                    "position_enu_m": [round(float(x), 6), round(float(y), 6), 0.0],
                    "projection_error_m": 0.0,
                    "synthetic_interpolated_sample": True,
                    **interpolation_provenance,
                },
            )

    if active_end_tick is not None:
        end_control_tick = max(0, min(DURATION_TICKS, int(active_end_tick)))
        dense_ticks = sorted(projected_by_tick)
        last_tick = dense_ticks[-1]
        exit_min_s = 0.5
        exit_max_s = max(exit_min_s, float(edge.length_m) - SEMANTIC_EXIT_FRONT_BUMPER_MARGIN_M)

        def exit_boundary_reached(candidate_s: float, step_s: float) -> bool:
            if step_s >= 0.0:
                return float(candidate_s) > exit_max_s
            return float(candidate_s) < exit_min_s

        if end_control_tick > last_tick and hold_after_last_sample:
            last_sample = projected_by_tick[last_tick]
            last_s = float(last_sample["lane_position_m"])
            x, y = _point_at_edge_s(edge, last_s)
            for tick in range(last_tick + 1, end_control_tick + 1):
                projected_by_tick.setdefault(
                    tick,
                    {
                        "tick": tick,
                        "lane_position_m": round(float(last_s), 6),
                        "position_enu_m": [round(float(x), 6), round(float(y), 6), 0.0],
                        "projection_error_m": 0.0,
                        "synthetic_post_event_hold_sample": True,
                        **derived_provenance(
                            "synthetic_post_event_hold_from_last_authoritative_movement_sample_v1",
                            last_sample,
                        ),
                    },
                )
        elif end_control_tick > last_tick:
            last_s = float(projected_by_tick[last_tick]["lane_position_m"])
            previous_s = last_s
            for previous_tick in reversed(dense_ticks[:-1]):
                candidate_s = float(projected_by_tick[previous_tick]["lane_position_m"])
                if abs(last_s - candidate_s) > 0.05:
                    previous_s = candidate_s
                    break
            step_s = max(-0.8, min(0.8, last_s - previous_s))
            if abs(step_s) <= 0.05:
                step_s = 0.35 * travel_sign
            for tick in range(last_tick + 1, end_control_tick + 1):
                candidate_s = last_s + step_s * float(tick - last_tick)
                if exit_boundary_reached(candidate_s, step_s):
                    break
                lane_s = max(exit_min_s, min(exit_max_s, candidate_s))
                x, y = _point_at_edge_s(edge, lane_s)
                projected_by_tick.setdefault(
                    tick,
                    {
                        "tick": tick,
                        "lane_position_m": round(float(lane_s), 6),
                        "position_enu_m": [round(float(x), 6), round(float(y), 6), 0.0],
                        "projection_error_m": 0.0,
                        "synthetic_exit_sample": True,
                    },
                )
        if visible_bbox_enu_m and not hold_after_last_sample:
            dense_ticks = sorted(projected_by_tick)
            last_tick = dense_ticks[-1]
            last_position = projected_by_tick[last_tick]["position_enu_m"]
            if _point_in_bbox(last_position, visible_bbox_enu_m):
                last_s = float(projected_by_tick[last_tick]["lane_position_m"])
                previous_s = last_s
                for previous_tick in reversed(dense_ticks[:-1]):
                    candidate_s = float(projected_by_tick[previous_tick]["lane_position_m"])
                    if abs(last_s - candidate_s) > 0.05:
                        previous_s = candidate_s
                        break
                step_s = max(-0.8, min(0.8, last_s - previous_s))
                if abs(step_s) <= 0.05:
                    step_s = 0.35 * travel_sign
                for tick in range(last_tick + 1, DURATION_TICKS + 1):
                    candidate_s = last_s + step_s * float(tick - last_tick)
                    if exit_boundary_reached(candidate_s, step_s):
                        break
                    lane_s = max(exit_min_s, min(exit_max_s, candidate_s))
                    x, y = _point_at_edge_s(edge, lane_s)
                    projected_by_tick.setdefault(
                        tick,
                        {
                            "tick": tick,
                            "lane_position_m": round(float(lane_s), 6),
                            "position_enu_m": [round(float(x), 6), round(float(y), 6), 0.0],
                            "projection_error_m": 0.0,
                            "synthetic_exit_sample": True,
                        },
                    )
                    if not _point_in_bbox([x, y], visible_bbox_enu_m):
                        break

    for sample in projected_by_tick.values():
        lane_s = max(0.5, min(max(0.5, edge.length_m - 0.5), float(sample["lane_position_m"])))
        x, y = _point_at_edge_s(edge, lane_s)
        sample["lane_position_m"] = float(lane_s)
        sample["position_enu_m"] = [float(x), float(y), 0.0]

    lane_min_s = 0.5
    lane_max_s = max(lane_min_s, float(edge.length_m) - 0.5)
    previous_forward_s: float | None = None
    for tick in sorted(projected_by_tick):
        sample = projected_by_tick[tick]
        lane_s = max(lane_min_s, min(lane_max_s, float(sample["lane_position_m"])))
        if previous_forward_s is not None:
            delta_s = lane_s - previous_forward_s
            if delta_s < -0.05:
                raise RuntimeError(
                    f"semantic vehicle {source_entity_id or '<unknown>'} projected control path reverses "
                    f"on SUMO edge {edge.edge_id} at tick {tick}: lane position changes from "
                    f"{previous_forward_s:.3f}m to {lane_s:.3f}m; refusing to mirror the reverse delta "
                    "into fabricated forward motion"
                )
            elif delta_s < 0.0:
                lane_s = previous_forward_s
                correction = "sub_5cm_projection_jitter_clamp"
            else:
                correction = ""
            if correction:
                existing = str(sample.get("direction_correction") or "")
                sample["direction_correction"] = f"{existing}+{correction}" if existing else correction
        x, y = _point_at_edge_s(edge, lane_s)
        sample["lane_position_m"] = float(lane_s)
        sample["position_enu_m"] = [float(x), float(y), 0.0]
        previous_forward_s = lane_s

    controls: list[dict[str, Any]] = []
    previous_s: float | None = None
    previous_angle: float | None = None
    for tick in sorted(projected_by_tick):
        sample = projected_by_tick[tick]
        lane_s = float(sample["lane_position_m"])
        position = list(sample["position_enu_m"])
        angle = None
        if previous_s is not None and abs(lane_s - previous_s) > 0.05:
            angle = _sumo_angle_from_points(_point_at_edge_s(edge, previous_s), _point_at_edge_s(edge, lane_s))
        if angle is None:
            angle = previous_angle
        if angle is None:
            angle = _sumo_angle_at_edge_s(edge, lane_s)
        controls.append(
            {
                "tick": int(tick),
                "instruction_type": "vehicle.moveToXY",
                "semantic_lane_projection": {
                    "edge_id": edge.edge_id,
                    "lane_id": edge.lane_id,
                    "lane_index": lane_index,
                    "lane_position_m": round(float(sample["lane_position_m"]), 6),
                    "projection_error_m": round(float(sample.get("projection_error_m") or 0.0), 6),
                    "synthetic_entry_sample": bool(sample.get("synthetic_entry_sample")),
                    "synthetic_interpolated_sample": bool(sample.get("synthetic_interpolated_sample")),
                    "synthetic_exit_sample": bool(sample.get("synthetic_exit_sample")),
                    "synthetic_post_event_hold_sample": bool(sample.get("synthetic_post_event_hold_sample")),
                    "source_tick": sample.get("source_tick"),
                    "authority_event_id": str(sample.get("authority_event_id") or ""),
                    "movement_action_ids": [str(action_id) for action_id in sample.get("movement_action_ids") or []],
                    "retimed_by_ticks": int(sample.get("retimed_by_ticks") or 0),
                    "observed_physical_sample": bool(sample.get("observed_physical_sample")),
                    "derivation_policy": str(sample.get("derivation_policy") or ""),
                    "derivation_source_ticks": [
                        int(source_tick) for source_tick in sample.get("derivation_source_ticks") or []
                    ],
                    "direction_correction": str(sample.get("direction_correction") or ""),
                },
                "args": {
                    "edgeID": edge.edge_id,
                    "lane": lane_index,
                    "x": round(float(position[0]), 6),
                    "y": round(float(position[1]), 6),
                    "angle": angle,
                    "keepRoute": 0,
                    "matchThreshold": projection_error_tolerance_m + 1.0,
                },
            }
        )
        previous_s = lane_s
        previous_angle = angle
    return controls


def _sumo_angle_at_edge_s(edge: SumoEdge, s_m: float) -> float:
    window_m = 0.5
    s0 = max(0.0, min(float(edge.length_m), float(s_m) - window_m))
    s1 = max(0.0, min(float(edge.length_m), float(s_m) + window_m))
    if abs(s1 - s0) <= 1e-9:
        s0 = max(0.0, min(float(edge.length_m), float(s_m) - 1.0))
        s1 = max(0.0, min(float(edge.length_m), float(s_m) + 1.0))
    angle = _sumo_angle_from_points(_point_at_edge_s(edge, s0), _point_at_edge_s(edge, s1))
    return 90.0 if angle is None else angle


def _sumo_angle_from_points(a: Sequence[float], b: Sequence[float]) -> float | None:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    if abs(dx) + abs(dy) <= 1e-9:
        return None
    return round((90.0 - math.degrees(math.atan2(dy, dx))) % 360.0, 6)


def _separate_same_lane_semantic_entries(
    planner: SumoGroundFlowPlanner,
    vehicles: Sequence[dict[str, Any]],
) -> None:
    """Preserve authored queue spacing while multiple actors enter one lane.

    Each actor is projected independently before this point.  When several
    actors share a corridor, the independent ROI-boundary scan can collapse
    their synthetic entry controls and bootstrap positions to the same lane
    coordinate.  The first observed samples already encode the authoritative
    longitudinal ordering, so use that ordering for the synthetic prefix too.
    """

    vehicles_by_edge: dict[str, list[dict[str, Any]]] = {}
    for vehicle in vehicles:
        if str(vehicle.get("traffic_role") or "") != "semantic_vehicle":
            continue
        edge_id = str(dict(vehicle.get("semantic_corridor") or {}).get("edge_id") or "")
        if edge_id and vehicle.get("semantic_controls"):
            vehicles_by_edge.setdefault(edge_id, []).append(vehicle)

    for edge_id, lane_vehicles in vehicles_by_edge.items():
        if len(lane_vehicles) < 2:
            continue
        edge = planner.edges.get(edge_id)
        if edge is None:
            raise RuntimeError(f"semantic corridor edge {edge_id} is absent from the SUMO network")

        ordered: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
        for vehicle in lane_vehicles:
            controls = sorted(vehicle.get("semantic_controls") or [], key=lambda item: int(item.get("tick") or 0))
            observed = [
                control
                for control in controls
                if bool(dict(control.get("semantic_lane_projection") or {}).get("observed_physical_sample"))
            ]
            if not observed:
                continue
            first_observed = observed[0]
            projection = dict(first_observed.get("semantic_lane_projection") or {})
            ordered.append(
                (
                    float(projection.get("lane_position_m") or 0.0),
                    int(first_observed.get("tick") or 0),
                    vehicle,
                    first_observed,
                )
            )
        if len(ordered) < 2:
            continue
        ordered.sort(key=lambda item: (item[0], str(item[2].get("vehicle_id") or "")))

        # Fail on an authored collision.  Only the synthetic prefix may be
        # changed; observed source samples remain immutable.
        observed_by_tick: dict[int, list[tuple[float, str]]] = {}
        for _first_s, _first_tick, vehicle, _first_observed in ordered:
            for control in vehicle.get("semantic_controls") or []:
                projection = dict(control.get("semantic_lane_projection") or {})
                if not bool(projection.get("observed_physical_sample")):
                    continue
                observed_by_tick.setdefault(int(control.get("tick") or 0), []).append(
                    (
                        float(projection.get("lane_position_m") or 0.0),
                        str(vehicle.get("vehicle_id") or ""),
                    )
                )
        for tick, samples in observed_by_tick.items():
            samples.sort()
            for (rear_s, rear_id), (front_s, front_id) in zip(samples, samples[1:]):
                if front_s - rear_s < MIN_SEMANTIC_QUEUE_GAP_M:
                    raise RuntimeError(
                        f"authored semantic vehicles {rear_id} and {front_id} are only "
                        f"{front_s - rear_s:.3f}m apart on {edge_id} at tick {tick}"
                    )

        minimum_observed_s = ordered[0][0]
        current_entry_s = min(
            float(dict((vehicle.get("semantic_controls") or [])[0].get("semantic_lane_projection") or {}).get("lane_position_m") or 0.5)
            for _first_s, _first_tick, vehicle, _first_observed in ordered
        )
        desired_entry_by_vehicle_id = {
            str(vehicle.get("vehicle_id") or ""): current_entry_s + first_observed_s - minimum_observed_s
            for first_observed_s, _first_tick, vehicle, _first_observed in ordered
        }
        max_desired_entry_s = max(desired_entry_by_vehicle_id.values())
        lane_max_s = max(0.5, float(edge.length_m) - 0.5)
        if max_desired_entry_s > lane_max_s:
            shift = max_desired_entry_s - lane_max_s
            desired_entry_by_vehicle_id = {
                vehicle_id: max(0.5, desired_s - shift)
                for vehicle_id, desired_s in desired_entry_by_vehicle_id.items()
            }

        # Bootstrap all cars outside the ROI at distinct longitudinal slots.
        # The leading car keeps the existing outside-ROI boundary position;
        # followers queue behind it at a deterministic safe gap.
        leading_bootstrap_s = max(
            float(vehicle.get("semantic_bootstrap_depart_pos_m") or 0.5)
            for _first_s, _first_tick, vehicle, _first_observed in ordered
        )
        for queue_index, (_first_s, _first_tick, vehicle, first_observed) in enumerate(reversed(ordered)):
            vehicle["semantic_bootstrap_depart_pos_m"] = round(
                max(0.5, leading_bootstrap_s - queue_index * MIN_SEMANTIC_QUEUE_GAP_M),
                6,
            )

        for first_observed_s, first_observed_tick, vehicle, _first_observed in ordered:
            controls = sorted(vehicle.get("semantic_controls") or [], key=lambda item: int(item.get("tick") or 0))
            entry_tick = int(controls[0].get("tick") or 0)
            desired_entry_s = desired_entry_by_vehicle_id[str(vehicle.get("vehicle_id") or "")]
            for control in controls:
                tick = int(control.get("tick") or 0)
                if tick >= first_observed_tick:
                    break
                projection = dict(control.get("semantic_lane_projection") or {})
                if not bool(projection.get("synthetic_entry_sample")):
                    continue
                alpha = (
                    0.0
                    if first_observed_tick == entry_tick
                    else (tick - entry_tick) / float(first_observed_tick - entry_tick)
                )
                lane_s = desired_entry_s + (first_observed_s - desired_entry_s) * alpha
                x_m, y_m = _point_at_edge_s(edge, lane_s)
                projection["lane_position_m"] = round(float(lane_s), 6)
                control["semantic_lane_projection"] = projection
                args = dict(control.get("args") or {})
                args["x"] = round(float(x_m), 6)
                args["y"] = round(float(y_m), 6)
                args["angle"] = _sumo_angle_at_edge_s(edge, lane_s)
                control["args"] = args


def _semantic_bindings(script: dict[str, Any], source_vehicle_ids: Sequence[str], seed_index: int) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for event_def in script.get("events") or []:
        event_id = str(event_def.get("event_id") or event_def.get("topic") or "").strip()
        if not event_id:
            continue
        event_text = _json_text(event_def)
        matched = [source_id for source_id in source_vehicle_ids if source_id.lower() in event_text]
        for source_id in matched:
            bindings.append(
                {
                    "event_id": event_id,
                    "source_entity_id": source_id,
                    "vehicle_id": semantic_internal_vehicle_id(source_id, seed_index),
                    "binding_source": "event_script_entity_reference",
                    "evidence_ticks": sorted(set(_event_ticks(dict(event_def)))),
                }
            )
    return bindings


def _merge_semantic_bindings(bindings: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for binding in bindings:
        event_id = str(binding.get("event_id") or "")
        source_id = str(binding.get("source_entity_id") or "")
        if not event_id or not source_id:
            continue
        key = (event_id, source_id)
        current = merged.setdefault(key, dict(binding))
        current["evidence_ticks"] = sorted(
            set([int(tick) for tick in current.get("evidence_ticks") or []])
            | set(int(tick) for tick in binding.get("evidence_ticks") or [])
        )
        sources = set(str(current.get("binding_source") or "").split("+"))
        sources.add(str(binding.get("binding_source") or ""))
        current["binding_source"] = "+".join(sorted(source for source in sources if source))
    return list(merged.values())


def _slot_background_count(scenario_class: str, seed_index: int, flow_mode: str = "") -> int:
    if str(flow_mode or "") == "emergency_corridor_with_yielding_flow":
        return int(EMERGENCY_CORRIDOR_BACKGROUND_TARGETS.get(int(seed_index), 10))
    if str(flow_mode or "") == "context_free_flow":
        return int(CONTEXT_FREE_FLOW_BACKGROUND_TARGETS.get(int(seed_index), CONTEXT_FREE_FLOW_BACKGROUND_TARGETS.get(int(seed_index) % 3, 8)))
    by_seed = SLOT_BACKGROUND_TARGETS.get(str(scenario_class), SLOT_BACKGROUND_TARGETS["road_or_vehicle_semantic_event"])
    return int(by_seed.get(int(seed_index), by_seed.get(int(seed_index) % 3, 6)))


def _accepted_background_count_target(scenario_class: str, seed_index: int, flow_mode: str = "") -> int:
    if str(flow_mode or "") == "emergency_corridor_with_yielding_flow":
        return int(EMERGENCY_CORRIDOR_ACCEPTED_BACKGROUND_TARGETS.get(int(seed_index), 50))
    per_slot = _slot_background_count(scenario_class, seed_index, flow_mode)
    return int(max(1, round(per_slot * TRAFFIC_SLOT_COUNT * 0.82)))


def _slot_bounds(slot_index: int) -> tuple[int, int]:
    start = int(slot_index) * TRAFFIC_SLOT_TICKS
    end = DURATION_TICKS if int(slot_index) == TRAFFIC_SLOT_COUNT - 1 else min(DURATION_TICKS, start + TRAFFIC_SLOT_TICKS - 1)
    return start, end


def _slot_entry_ticks(slot_index: int, count: int, profile: SeedTrafficProfile) -> list[int]:
    start, end = _slot_bounds(slot_index)
    usable_start = start + 8
    usable_end = max(usable_start, end - 8)
    if count <= 1:
        base_ticks = [(usable_start + usable_end) // 2]
    else:
        span = usable_end - usable_start
        base_ticks = [int(round(usable_start + span * index / float(count - 1))) for index in range(count)]
    jitter = profile.slot_entry_jitter_ticks or (0,)
    return [
        max(start, min(end, int(base_tick) + int(jitter[index % len(jitter)])))
        for index, base_tick in enumerate(base_ticks)
    ]


def _traffic_slots_payload(
    *,
    scenario_class: str,
    flow_mode: str,
    profile: SeedTrafficProfile,
) -> list[dict[str, Any]]:
    target = _slot_background_count(scenario_class, profile.seed_index, flow_mode)
    slots: list[dict[str, Any]] = []
    for slot_index in range(TRAFFIC_SLOT_COUNT):
        start, end = _slot_bounds(slot_index)
        entry_ticks = _slot_entry_ticks(slot_index, target, profile)
        slots.append(
            {
                "slot_index": slot_index,
                "start_tick": start,
                "end_tick": end,
                "duration_ticks": end - start + 1,
                "target_background_vehicle_count": target,
                "minimum_core_roi_vehicle_passages": target,
                "direction_distribution": dict(profile.direction_distribution),
                "core_entry_ticks": entry_ticks,
                "vehicle_ids": [],
            }
        )
    return slots


def _roi_route_crossing_cap_per_slot(
    *,
    entry_tick_count: int,
    roi_visible_edge_count: int,
    traffic_tuning: dict[str, Any],
) -> int:
    explicit_cap = _optional_int(traffic_tuning.get("roi_route_crossing_cap_per_slot"))
    if explicit_cap is not None:
        return max(0, min(int(entry_tick_count), explicit_cap))
    narrow_threshold = _optional_int(traffic_tuning.get("narrow_roi_visible_edge_threshold"))
    narrow_cap = _optional_int(traffic_tuning.get("narrow_roi_route_crossing_cap_per_slot"))
    if narrow_threshold is not None and narrow_cap is not None and int(roi_visible_edge_count) <= narrow_threshold:
        return max(0, min(int(entry_tick_count), narrow_cap))
    return max(0, int(entry_tick_count))


def _optional_int_sequence(value: Any, default_values: Sequence[int]) -> list[int]:
    if value is None:
        return [int(item) for item in default_values]
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_items = list(value)
    else:
        return [int(item) for item in default_values]
    parsed: list[int] = []
    for item in raw_items:
        parsed_item = _optional_int(item)
        if parsed_item is not None:
            parsed.append(int(parsed_item))
    return parsed


def _context_free_flow_smoothing_entry_ticks(traffic_tuning: dict[str, Any]) -> list[int]:
    return sorted(
        {
            max(0, min(DURATION_TICKS, int(tick)))
            for tick in _optional_int_sequence(
                traffic_tuning.get("context_free_flow_smoothing_entry_ticks"),
                CONTEXT_FREE_FLOW_ROI_SMOOTHING_ENTRY_TICKS,
            )
        }
    )


def _context_free_flow_smoothing_stop_ticks(traffic_tuning: dict[str, Any]) -> int:
    configured = _optional_int(traffic_tuning.get("context_free_flow_smoothing_stop_ticks"))
    if configured is None:
        return int(CONTEXT_FREE_FLOW_ROI_SMOOTHING_STOP_TICKS)
    return max(0, int(configured))


def _roi_route_crossing_smoothing_max_per_slot(traffic_tuning: dict[str, Any]) -> int:
    configured = _optional_int(traffic_tuning.get("roi_route_crossing_smoothing_max_per_slot"))
    if configured is None:
        return 2
    return max(0, int(configured))


def _protected_window_pre_route_guard_ticks(traffic_tuning: dict[str, Any]) -> int:
    configured = _optional_int(traffic_tuning.get("protected_window_pre_route_guard_ticks"))
    if configured is None:
        return int(PROTECTED_WINDOW_PRE_ROUTE_GUARD_TICKS)
    return max(int(CORE_CROSSING_TICKS), int(configured))


def _protected_window_post_route_guard_ticks(traffic_tuning: dict[str, Any]) -> int:
    configured = _optional_int(traffic_tuning.get("protected_window_post_route_guard_ticks"))
    if configured is None:
        return int(PROTECTED_WINDOW_POST_ROUTE_GUARD_TICKS)
    return max(0, int(configured))


def _traffic_slot_index_for_tick(tick: int) -> int:
    return max(0, min(TRAFFIC_SLOT_COUNT - 1, int(tick) // int(TRAFFIC_SLOT_TICKS)))


def _must_visible_ticks(entry_target_tick: int, binding_ticks: Sequence[int]) -> list[int]:
    ticks = {
        max(0, int(entry_target_tick) - 5),
        int(entry_target_tick),
        min(DURATION_TICKS, int(entry_target_tick) + CORE_CROSSING_TICKS),
    }
    for tick in binding_ticks:
        ticks.add(max(0, min(DURATION_TICKS, int(tick) - PRE_EVENT_VISIBLE_MARGIN_TICKS)))
        ticks.add(max(0, min(DURATION_TICKS, int(tick))))
        ticks.add(max(0, min(DURATION_TICKS, int(tick) + POST_EVENT_VISIBLE_MARGIN_TICKS)))
    return sorted(tick for tick in ticks if 0 <= int(tick) <= DURATION_TICKS)


def _flow_groups_payload(vehicles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for vehicle in vehicles:
        group_id = str(vehicle.get("flow_group_id") or "")
        if not group_id:
            continue
        groups.setdefault(group_id, []).append(dict(vehicle))
    payload: list[dict[str, Any]] = []
    for group_id, members in sorted(groups.items()):
        depart_ticks = [int(vehicle.get("depart_tick") or 0) for vehicle in members]
        payload.append(
            {
                "flow_group_id": group_id,
                "flow_mode": str(members[0].get("flow_mode") or ""),
                "direction_role": str(members[0].get("direction_role") or ""),
                "vehicle_ids": [str(vehicle.get("vehicle_id") or "") for vehicle in members],
                "vehicle_count": len(members),
                "depart_tick_min": min(depart_ticks),
                "depart_tick_max": max(depart_ticks),
                "speed_target_mps_min": round(
                    min(float(dict(vehicle.get("speed_profile") or {}).get("target_mps") or 0.0) for vehicle in members),
                    6,
                ),
                "speed_target_mps_max": round(
                    max(float(dict(vehicle.get("speed_profile") or {}).get("target_mps") or 0.0) for vehicle in members),
                    6,
                ),
            }
        )
    return payload


def _edge_s_inside_bbox(edge: SumoEdge, bbox: Sequence[float]) -> float | None:
    if len(bbox) < 4:
        return None
    min_x, min_y, max_x, max_y = [float(value) for value in bbox[:4]]
    inset_bbox = [
        min_x + ROI_BODY_CENTER_INSET_M,
        min_y + ROI_BODY_CENTER_INSET_M,
        max_x - ROI_BODY_CENTER_INSET_M,
        max_y - ROI_BODY_CENTER_INSET_M,
    ]
    if inset_bbox[0] > inset_bbox[2] or inset_bbox[1] > inset_bbox[3]:
        return None
    clipped_spans: list[tuple[float, float]] = []
    cumulative_s = 0.0
    for start, end in zip(edge.shape_xy, edge.shape_xy[1:]):
        segment_length_m = math.hypot(
            float(end[0]) - float(start[0]),
            float(end[1]) - float(start[1]),
        )
        if segment_length_m <= 1e-9:
            continue
        interval = _segment_bbox_interval(start, end, inset_bbox)
        if interval is not None:
            entry_fraction, exit_fraction = interval
            clipped_spans.append(
                (
                    cumulative_s + entry_fraction * segment_length_m,
                    cumulative_s + exit_fraction * segment_length_m,
                )
            )
        cumulative_s += segment_length_m
    if not clipped_spans:
        return None
    merged_spans: list[list[float]] = []
    for start_s, end_s in clipped_spans:
        if merged_spans and start_s <= merged_spans[-1][1] + 1e-6:
            merged_spans[-1][1] = max(merged_spans[-1][1], end_s)
        else:
            merged_spans.append([start_s, end_s])
    longest_start_s, longest_end_s = max(
        merged_spans,
        key=lambda span: (span[1] - span[0], -span[0]),
    )
    lane_s = 0.5 * (longest_start_s + longest_end_s)
    return round(max(0.0, min(lane_s, float(edge.length_m))), 6)


def _estimated_roi_route_lead_ticks(
    planner: SumoGroundFlowPlanner,
    endpoint: dict[str, Any],
    bbox: Sequence[float],
    speed_mps: float,
) -> int:
    if len(bbox) < 4:
        return ESTIMATED_UPSTREAM_TRAVEL_TICKS * 2
    depart_xy = endpoint.get("from_xy_enu_m") or []
    if not isinstance(depart_xy, Sequence) or isinstance(depart_xy, (str, bytes)) or len(depart_xy) < 2:
        return ESTIMATED_UPSTREAM_TRAVEL_TICKS * 2
    route_to_roi = [
        str(edge_id)
        for edge_id in endpoint.get("route_to_roi_edges") or []
        if str(edge_id) in planner.edges
    ]
    if not route_to_roi:
        raise RuntimeError("ROI crossing endpoint is missing its directed route-to-ROI contract")
    travel_s = 0.0
    for edge_index, edge_id in enumerate(route_to_roi):
        edge = planner.edges[edge_id]
        start_s = float(endpoint.get("from_pos") or 0.0) if edge_index == 0 else 0.0
        entry_s = _edge_s_inside_bbox(edge, bbox)
        end_s = float(edge.length_m)
        reaches_roi = entry_s is not None and float(entry_s) >= start_s
        if reaches_roi:
            end_s = float(entry_s)
        travel_distance_m = max(0.0, end_s - start_s)
        governed_speed_mps = max(
            1.0,
            min(float(speed_mps), float(edge.speed_mps) if float(edge.speed_mps) > 0.0 else float(speed_mps)),
        )
        travel_s += travel_distance_m / governed_speed_mps
        if reaches_roi:
            break
    free_flow_ticks = int(math.ceil(travel_s * float(TICK_HZ)))
    # A route is released early enough for one full signal cycle and queue
    # discharge.  A planned stop on the ROI edge prevents an early vehicle
    # from crossing before its assigned slot.
    signal_and_queue_ticks = max(100, int(math.ceil(0.5 * free_flow_ticks)))
    return max(1, free_flow_ticks + signal_and_queue_ticks)


def _estimated_pre_capture_spawn_s(
    route_lead_ticks: int,
    *,
    coverage_scope: str = "",
    traffic_tuning: dict[str, Any] | None = None,
) -> float:
    lead_ticks = max(1, int(route_lead_ticks))
    tuning = dict(traffic_tuning or {})
    if str(coverage_scope) == "roi_background_route_crossing_flow":
        lead_fraction = _optional_float(
            tuning.get(
                "future_vehicle_entry_lead_fraction",
                tuning.get("route_crossing_pre_spawn_lead_fraction"),
            )
        )
        if lead_fraction is None:
            lead_fraction = 0.2
        reserve_ticks = _optional_int(
            tuning.get(
                "future_vehicle_entry_reserve_ticks",
                tuning.get("route_crossing_pre_spawn_reserve_ticks"),
            )
        )
        if reserve_ticks is None:
            reserve_ticks = 0
        activation_ticks = max(2, int(math.ceil(float(lead_ticks) * lead_fraction + float(reserve_ticks))))
    else:
        activation_ticks = max(2, int(math.ceil(float(lead_ticks) * 0.2)))
    return round(float(activation_ticks) / float(TICK_HZ), 6)


def _vehicle_plan_record(
    *,
    planner: SumoGroundFlowPlanner,
    episode_dir: Path,
    scenario_id: str,
    seed_index: int,
    profile: SeedTrafficProfile,
    source_entity: dict[str, Any] | None,
    vehicle_index: int,
    endpoint: dict[str, Any],
    direction_role: str,
    flow_mode: str,
    semantic_actor: bool,
    binding_ticks: Sequence[int],
    traffic_slot_index: int | None,
    core_entry_tick: int,
    core_exit_tick: int,
    forbidden_edges: Sequence[str],
    coverage_scope: str = "extended_context_flow",
    required_edges: Sequence[str] = (),
    spawn_scope: str = "outside_roi",
    visible_bbox_enu_m: Sequence[float] = (),
    semantic_corridor: dict[str, Any] | None = None,
    long_roi_release_lead: bool = False,
    roi_visible_release_lead_ticks: int | None = None,
    traffic_tuning: dict[str, Any] | None = None,
    semantic_lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_entity_id = str((source_entity or {}).get("entity_id") or "")
    source_presence_required = _source_presence_required(
        source_entity,
        semantic_actor=semantic_actor,
    )
    lifecycle = dict(semantic_lifecycle or {}) if semantic_actor else {}
    strict_absent_before_tick = _optional_int(lifecycle.get("strict_absent_before_tick"))
    if source_entity_id:
        vehicle_id = semantic_internal_vehicle_id(source_entity_id, seed_index)
    else:
        slot_token = f"slot{int(traffic_slot_index):02d}" if traffic_slot_index is not None else "slotxx"
        vehicle_id = f"traffic_bg_{_safe_token(scenario_id)}_{profile.seed_label}_{slot_token}_{vehicle_index + 1:03d}"
    role, type_id, vehicle_class = _role_and_type(source_entity or {"vehicle_id": vehicle_id}, semantic_actor)
    speed_mps = _flow_speed_target(profile, flow_mode, vehicle_index)
    no_protected_context_flow = (
        not semantic_actor
        and not forbidden_edges
        and str(flow_mode or "") == "context_free_flow"
    )
    if roi_visible_release_lead_ticks is not None:
        visible_lead_ticks = max(1, int(roi_visible_release_lead_ticks))
    elif no_protected_context_flow:
        visible_lead_ticks = CONTEXT_ROI_VISIBLE_FLOW_LEAD_TICKS
        if str(coverage_scope) == "roi_background_visible_flow" and traffic_slot_index is not None:
            visible_lead_ticks += max(0, int(traffic_slot_index) - 6) * CONTEXT_ROI_LATE_SLOT_EXTRA_LEAD_TICKS
    elif (
        str(coverage_scope) == "roi_background_visible_flow"
        and str(spawn_scope) == "outside_roi_boundary_inside_extended_approach"
    ):
        visible_lead_ticks = ROI_BOUNDARY_APPROACH_RELEASE_LEAD_TICKS
    elif long_roi_release_lead:
        visible_lead_ticks = ROI_VISIBLE_FLOW_LEAD_TICKS
    else:
        visible_lead_ticks = ESTIMATED_UPSTREAM_TRAVEL_TICKS
    if str(coverage_scope) == "roi_background_route_crossing_flow":
        route_lead_ticks = _estimated_roi_route_lead_ticks(planner, endpoint, visible_bbox_enu_m, speed_mps)
    else:
        route_lead_ticks = ESTIMATED_UPSTREAM_TRAVEL_TICKS * 2
    if str(coverage_scope) == "roi_background_visible_flow":
        release_tick = max(-int(visible_lead_ticks), int(core_entry_tick) - int(visible_lead_ticks))
    elif str(coverage_scope) == "roi_background_route_crossing_flow":
        release_tick = max(-int(route_lead_ticks), int(core_entry_tick) - int(route_lead_ticks))
    elif str(coverage_scope) == "roi_background_local_component_flow":
        release_tick = int(core_entry_tick)
    else:
        release_tick = max(-int(ESTIMATED_UPSTREAM_TRAVEL_TICKS * 2), int(core_entry_tick) - ESTIMATED_UPSTREAM_TRAVEL_TICKS)
    if semantic_actor:
        semantic_release_overrides = dict(dict(traffic_tuning or {}).get("semantic_release_tick_overrides") or {})
        release_override = _optional_int(semantic_release_overrides.get(source_entity_id))
        if release_override is not None:
            release_tick = int(release_override)
        if strict_absent_before_tick is not None:
            release_tick = int(strict_absent_before_tick)
    projection = {
        "edge_id": endpoint["from_edge"],
        "lane_id": endpoint["from_lane"],
        "lane_index": 0,
        "lane_position_m": round(float(endpoint["from_pos"]), 6),
        "xy_enu_m": list(endpoint["from_xy_enu_m"]),
    }
    flow_group_id = _flow_group_for_vehicle(flow_mode, direction_role, role, vehicle_index)
    if strict_absent_before_tick is not None:
        must_visible_ticks = sorted(
            {
                int(strict_absent_before_tick),
                int(lifecycle.get("control_start_tick") or core_entry_tick),
                int(lifecycle.get("control_end_tick") or core_exit_tick),
                *[int(tick) for tick in lifecycle.get("retimed_sample_ticks") or []],
            }
        )
    else:
        must_visible_ticks = _must_visible_ticks(core_entry_tick, binding_ticks)
    semantic_samples = (
        _semantic_control_samples(
            episode_dir=episode_dir,
            source_entity_id=source_entity_id,
            authority_event_ids=[str(event_id) for event_id in lifecycle.get("authority_event_ids") or []],
            movement_action_ids=[str(action_id) for action_id in lifecycle.get("movement_action_ids") or []],
            tick_offset=int(lifecycle.get("sample_tick_offset") or 0),
        )
        if semantic_actor
        else []
    )
    if semantic_actor and strict_absent_before_tick is not None and not semantic_samples:
        raise RuntimeError(
            f"semantic vehicle {source_entity_id} dispatch-gated lifecycle has no authoritative physical samples"
        )
    control_start_tick = (
        int(lifecycle.get("control_start_tick") or core_entry_tick)
        if strict_absent_before_tick is not None
        else (max(0, min(must_visible_ticks) - SEMANTIC_BOUNDARY_ENTRY_LEAD_TICKS) if must_visible_ticks else None)
    )
    control_end_tick = (
        int(lifecycle.get("control_end_tick") or core_exit_tick)
        if strict_absent_before_tick is not None
        else (min(DURATION_TICKS, max(must_visible_ticks) + POST_EVENT_VISIBLE_MARGIN_TICKS) if must_visible_ticks else None)
    )
    semantic_controls = _semantic_lane_controls(
        planner=planner,
        raw_samples=semantic_samples,
        semantic_corridor=semantic_corridor,
        source_entity_id=source_entity_id,
        active_start_tick=control_start_tick,
        active_end_tick=control_end_tick,
        visible_bbox_enu_m=visible_bbox_enu_m,
        enforce_active_start_tick=strict_absent_before_tick is not None,
        hold_after_last_sample=semantic_actor,
    )
    semantic_bootstrap_depart_pos = None
    if semantic_actor and semantic_corridor:
        corridor_edge = planner.edges.get(str(semantic_corridor.get("edge_id") or ""))
        if corridor_edge is not None:
            semantic_offset_index = 1 if any(token in source_entity_id.lower() for token in ("yield", "civilian")) else 0
            first_control_s = min(
                (
                    float(dict(control.get("semantic_lane_projection") or {}).get("lane_position_m") or 0.0)
                    for control in semantic_controls
                ),
                default=None,
            )
            outside_limit = _edge_depart_s_before_bbox(corridor_edge, visible_bbox_enu_m) if visible_bbox_enu_m else None
            if outside_limit is None:
                outside_limit = float(corridor_edge.length_m) - 2.0
            bootstrap_depart_pos_override_m = _optional_float(dict(semantic_corridor).get("bootstrap_depart_pos_m"))
            if bootstrap_depart_pos_override_m is not None:
                desired_depart_pos = float(bootstrap_depart_pos_override_m)
                semantic_bootstrap_depart_pos = max(
                    0.5,
                    min(float(desired_depart_pos), max(0.5, float(corridor_edge.length_m) - 0.5)),
                )
            elif first_control_s is not None:
                if semantic_offset_index:
                    desired_depart_pos = min(float(first_control_s) + 4.0, float(outside_limit))
                else:
                    desired_depart_pos = min(float(first_control_s), float(outside_limit))
                semantic_bootstrap_depart_pos = max(0.5, min(float(desired_depart_pos), float(outside_limit)))
            else:
                desired_depart_pos = min(max(0.5, 2.0 + float(semantic_offset_index) * 8.0), float(outside_limit))
                semantic_bootstrap_depart_pos = max(0.5, min(float(desired_depart_pos), float(outside_limit)))
    active_by_tick = min(must_visible_ticks or [0])
    if strict_absent_before_tick is not None:
        active_by_tick = int(strict_absent_before_tick)
    elif binding_ticks:
        active_by_tick = min(
            active_by_tick,
            max(0, min(int(tick) for tick in binding_ticks) - PRE_EVENT_VISIBLE_MARGIN_TICKS),
        )
    if semantic_actor:
        pre_capture_spawn_s = 0.0
    elif source_presence_required:
        pre_capture_spawn_s = FORMAL_WARMUP_TICKS / float(TICK_HZ)
    else:
        pre_capture_spawn_s = _estimated_pre_capture_spawn_s(
            route_lead_ticks,
            coverage_scope=str(coverage_scope),
            traffic_tuning=dict(traffic_tuning or {}),
        )
    source_presence_contract = dict(endpoint.get("source_presence_contract") or {})
    if source_presence_required:
        planned_add_capture_tick = max(
            -FORMAL_WARMUP_TICKS,
            int(release_tick) - FORMAL_WARMUP_TICKS,
        )
        source_presence_contract.update(
            {
                "formal_warmup_ticks": FORMAL_WARMUP_TICKS,
                "pre_capture_spawn_ticks": FORMAL_WARMUP_TICKS,
                "planned_add_capture_tick": int(planned_add_capture_tick),
                "insertion_reserve_before_capture_ticks": int(-planned_add_capture_tick),
            }
        )
    return {
        "vehicle_id": vehicle_id,
        "source_entity_id": source_entity_id,
        "source_presence_required": source_presence_required,
        "role": role,
        "traffic_role": "semantic_vehicle" if semantic_actor else "deterministic_background_vehicle",
        "spawn_policy": (
            "semantic_absent_until_orchestration_evidence_v2"
            if strict_absent_before_tick is not None
            else ("semantic_absent_until_control_window_v1" if semantic_actor else "warmup_spawn_visible_flow_entry_v2")
        ),
        "spawn_phase": "semantic_control_window" if semantic_actor else "warmup_or_capture_release",
        "pre_capture_spawn_s": round(float(pre_capture_spawn_s), 6),
        "traffic_slot_index": traffic_slot_index,
        "traffic_coverage_scope": "semantic_event_actor" if semantic_actor else str(coverage_scope),
        "active_by_tick": int(active_by_tick),
        "expected_core_entry_tick": int(core_entry_tick),
        "expected_core_exit_tick": int(core_exit_tick),
        "release_tick": int(release_tick),
        "release_time_s": round(release_tick / float(TICK_HZ), 6),
        **(
            {
                "roi_approach_contract": {
                    "policy": "directed_route_free_flow_plus_signal_queue_allowance_v1",
                    "route_to_roi_edges": [
                        str(edge_id) for edge_id in endpoint.get("route_to_roi_edges") or []
                    ],
                    "lead_ticks": int(route_lead_ticks),
                }
            }
            if str(coverage_scope) == "roi_background_route_crossing_flow"
            else {}
        ),
        **(
            {"source_presence_contract": source_presence_contract}
            if source_presence_required and source_presence_contract
            else {}
        ),
        "hold_policy": {
            "policy": "warmup_spawn_hold_upstream_until_release_tick_v1",
            "hold_edge": endpoint["from_edge"],
            "hold_lane_index": 0,
            "hold_pos_m": round(float(endpoint["from_pos"]), 6),
            "release_tick": int(release_tick),
        },
        "type_id": type_id,
        "vehicle_class": vehicle_class,
        "depart_tick": 0,
        "depart_time_s": 0.0,
        "from_edge": endpoint["from_edge"],
        "from_edge_candidates": list(endpoint.get("from_edge_candidates") or [endpoint["from_edge"]]),
        "from_edge_depart_pos_by_edge": dict(
            endpoint.get("from_edge_depart_pos_by_edge")
            or {str(endpoint["from_edge"]): round(float(endpoint["from_pos"]), 6)}
        ),
        "from_lane": endpoint["from_lane"],
        "from_lane_index": int(endpoint.get("from_lane_index") or 0),
        "from_pos": round(float(endpoint["from_pos"]), 6),
        "from_projection": projection,
        "to_edge": endpoint["to_edge"],
        "to_edge_candidates": list(endpoint.get("to_edge_candidates") or []),
        "forbidden_edges": sorted(set(str(edge_id) for edge_id in forbidden_edges if str(edge_id))),
        "required_edges": sorted(set(str(edge_id) for edge_id in required_edges if str(edge_id))),
        "depart_projection": projection,
        "route_policy": "sumo_findRoute_from_to_v1",
        "direction_role": direction_role,
        "flow_mode": flow_mode,
        "flow_group_id": flow_group_id,
        "speed_profile": {
            "policy": f"{profile.profile_id}_{flow_mode}_fixed_target_speed_v1",
            "target_mps": round(speed_mps, 6),
            "max_speed_mps": round(speed_mps, 6),
            "depart_speed": "0",
        },
        "spacing_profile": {
            "policy": "explicit_traffic_slot_entry_spacing_v2",
            "platoon_or_queue_group": flow_group_id,
        },
        "must_be_visible_ticks": must_visible_ticks,
        "semantic_actor": bool(semantic_actor),
        "semantic_source": "scene_vehicle_binding" if semantic_actor else "deterministic_traffic_slot_schedule",
        "semantic_corridor": semantic_corridor if semantic_actor else None,
        "semantic_lifecycle": lifecycle if semantic_actor and lifecycle else None,
        "semantic_bootstrap_depart_pos_m": round(float(semantic_bootstrap_depart_pos), 6) if semantic_bootstrap_depart_pos is not None else None,
        "semantic_controls": semantic_controls,
        "sumo_control": {
            "source_policy": VEHICLE_SOURCE_POLICY,
            "route_source": "sumo_findRoute",
            "spawn_scope": str(spawn_scope),
            "required_edges": sorted(set(str(edge_id) for edge_id in required_edges if str(edge_id))),
            "core_crossing_target_ticks": [int(core_entry_tick), int(core_exit_tick)],
            "traffic_tuning_source": dict(dict(traffic_tuning or {}).get("source") or {}),
        },
    }


def _build_spatial_scope(script: dict[str, Any], scene_entities: Sequence[dict[str, Any]]) -> dict[str, Any]:
    boundary = _capture_boundary(script)
    polygon = []
    for point in boundary.get("polygon_enu_m") or []:
        pos = _position3(point)
        if pos is not None:
            polygon.append([round(pos[0], 6), round(pos[1], 6)])
    if polygon:
        bbox = _bbox_from_points(polygon)
    else:
        positions = [pos for entity in scene_entities if (pos := _entity_position(entity)) is not None]
        bbox = _bbox_from_points(positions, padding_m=60.0)
    bbox = _expand_bbox_to_minimum(bbox, MIN_CORE_ROI_WIDTH_M, MIN_CORE_ROI_HEIGHT_M)
    padding = float(boundary.get("expanded_boundary_padding_m") or 60.0)
    expanded_bbox = [round(float(bbox[0]) - padding, 6), round(float(bbox[1]) - padding, 6), round(float(bbox[2]) + padding, 6), round(float(bbox[3]) + padding, 6)]
    return {
        "roi_id": str(boundary.get("boundary_id") or boundary.get("source_entity_id") or "episode_capture_roi"),
        "capture_boundary_polygon_enu_m": polygon,
        "bbox_enu_m": bbox,
        "expanded_boundary_padding_m": round(padding, 6),
        "expanded_bbox_enu_m": expanded_bbox,
    }


def _merge_scene_and_roster_entities(scene_setup: dict[str, Any], roster: dict[str, Any]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for entity in scene_setup.get("entities") or []:
        if isinstance(entity, dict) and entity.get("entity_id"):
            entity_id = str(entity["entity_id"])
            if entity_id in by_id:
                raise ValueError(f"duplicate scene entity ID: {entity_id}")
            by_id[entity_id] = dict(entity)
    for entity_id, entry in roster.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("entity_id") != entity_id:
            raise ValueError(
                f"roster key and entity ID disagree: {entity_id!r} != "
                f"{entry.get('entity_id')!r}"
            )
        merged = dict(by_id.get(str(entity_id)) or {})
        scene_asset = str(merged.get("logical_asset_id") or merged.get("asset_id") or "")
        roster_asset = str(entry.get("logical_asset_id") or entry.get("asset_id") or "")
        if scene_asset and roster_asset and scene_asset != roster_asset:
            raise ValueError(
                f"scene and roster vehicle assets disagree for {entity_id}: "
                f"{scene_asset!r} != {roster_asset!r}"
            )
        merged.update(dict(entry))
        merged.setdefault("entity_id", str(entity_id))
        by_id[str(entity_id)] = merged
    return list(by_id.values())


def _vehicle_entities(scene_entities: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    vehicles = []
    for entity in scene_entities:
        category = str(entity.get("category") or entity.get("label_class") or "")
        if category == "vehicle" and not is_script_controlled_vehicle(entity):
            vehicles.append(dict(entity))
    return sorted(vehicles, key=lambda item: str(item.get("entity_id") or ""))


def build_explicit_vehicle_plan(
    episode_dir: Path,
    *,
    net_xml: Path = DEFAULT_SUMO_NET_XML,
    project_root: Path = ROOT,
    planner: SumoGroundFlowPlanner | None = None,
) -> dict[str, Any]:
    episode_dir = Path(episode_dir)
    manifest = _read_json(episode_dir / "episode_manifest.json")
    episode_id = str(manifest.get("episode_id") or episode_dir.name)
    scenario_id = str(manifest.get("scenario_id") or episode_scenario_id(episode_id))
    seed_index = episode_seed_index(episode_id, manifest)
    profile = SEED_TRAFFIC_PROFILES.get(seed_index, SEED_TRAFFIC_PROFILES[seed_index % 3])
    duration_ticks = int(manifest.get("duration_ticks") or DURATION_TICKS)
    if duration_ticks != DURATION_TICKS:
        raise ValueError(f"{episode_id}: explicit SUMO vehicle plan requires duration_ticks={DURATION_TICKS}, got {duration_ticks}")

    script_path = resolve_manifest_path(project_root, str(manifest.get("source_event_script_path") or ""))
    scene_path = resolve_manifest_path(project_root, str(manifest.get("source_scene_setup_path") or ""))
    script = _read_json(script_path)
    scene_setup = _read_json(scene_path) if scene_path.exists() else {}
    roster = _read_json(episode_dir / "global_entity_roster.json")
    scene_entities = _merge_scene_and_roster_entities(scene_setup, roster)
    source_vehicles = _vehicle_entities(scene_entities)
    source_vehicle_ids = [str(entity.get("entity_id") or "") for entity in source_vehicles]
    bindings = _merge_semantic_bindings(
        [
            *_semantic_bindings(script, source_vehicle_ids, seed_index),
            *_episode_vehicle_event_bindings(episode_dir, source_vehicle_ids, seed_index),
        ]
    )
    semantic_sources = {str(binding["source_entity_id"]) for binding in bindings}
    has_semantic_vehicle_case = bool(semantic_sources)
    scenario_class, minimum_vehicle_count = _scenario_classification(scenario_id, script, bindings)
    flow_mode = _traffic_flow_mode(scenario_id, script, scenario_class)
    traffic_tuning = _traffic_tuning_for_episode(episode_id, scenario_id)
    semantic_lifecycles_by_source = {
        source_id: lifecycle
        for source_id in sorted(semantic_sources)
        if (
            lifecycle := _semantic_dispatch_motion_lifecycle(
                episode_dir=episode_dir,
                script=script,
                source_entity_id=source_id,
            )
        )
        is not None
    }
    traffic_slots = _traffic_slots_payload(scenario_class=scenario_class, flow_mode=flow_mode, profile=profile)
    traffic_slot_by_index = {int(slot.get("slot_index") or 0): slot for slot in traffic_slots}
    scheduled_background_count = sum(int(slot["target_background_vehicle_count"]) for slot in traffic_slots)
    target_vehicle_count = len(source_vehicles) + scheduled_background_count
    planner = planner or SumoGroundFlowPlanner(net_xml)
    road_semantics_obj = resolve_scene_road_semantics(scene_entities, episode_id=episode_id)
    road_semantics = road_semantics_obj.as_dict()
    road_closed_edge_set = {edge_id for edge_id in road_semantics_obj.closed_edges if edge_id in planner.edges}
    road_closed_edges = sorted(road_closed_edge_set)
    road_closed_lanes = sorted(lane_id for lane_id in road_semantics_obj.closed_lanes)
    semantic_corridors_by_source: dict[str, dict[str, Any]] = {}
    for entity in source_vehicles:
        source_id = str(entity.get("entity_id") or "")
        if source_id not in semantic_sources:
            continue
        corridor = _semantic_corridor_for_vehicle(
            planner=planner,
            episode_dir=episode_dir,
            source_entity=entity,
            source_entity_id=source_id,
            traffic_tuning=traffic_tuning,
        )
        if corridor and corridor.get("edge_id"):
            semantic_corridors_by_source[source_id] = corridor
    spatial_scope = _build_spatial_scope(script, scene_entities)
    spatial_scope["road_semantics"] = road_semantics
    spatial_scope["road_semantic_closed_sumo_edges"] = road_closed_edges
    spatial_scope["road_semantic_closed_sumo_lanes"] = road_closed_lanes
    center_xy = _bbox_center(spatial_scope["bbox_enu_m"] or spatial_scope["expanded_bbox_enu_m"])
    roi_bbox = spatial_scope.get("bbox_enu_m") or []
    roi_visible_edges = [
        edge_id
        for edge_id in _protected_edges_for_bbox(planner, roi_bbox)
        if edge_id in planner.edges
        and _edge_s_inside_bbox(planner.edges[edge_id], roi_bbox) is not None
    ]
    spatial_scope["roi_visible_sumo_edges"] = roi_visible_edges
    spatial_scope["roi_visible_sumo_lanes"] = sorted(
        str(planner.edges[edge_id].lane_id)
        for edge_id in roi_visible_edges
        if edge_id in planner.edges and str(planner.edges[edge_id].lane_id)
    )
    core_edges = list(roi_visible_edges)
    spatial_scope["core_sumo_edges"] = core_edges
    spatial_scope["core_sumo_lanes"] = sorted(
        str(planner.edges[edge_id].lane_id)
        for edge_id in core_edges
        if edge_id in planner.edges and str(planner.edges[edge_id].lane_id)
    )
    expanded_edges = _protected_edges_for_bbox(
        planner,
        spatial_scope.get("expanded_bbox_enu_m") or spatial_scope.get("bbox_enu_m") or [],
    )
    spatial_scope["expanded_sumo_edges"] = expanded_edges
    spatial_scope["expanded_sumo_lanes"] = sorted(
        str(planner.edges[edge_id].lane_id)
        for edge_id in expanded_edges
        if edge_id in planner.edges and str(planner.edges[edge_id].lane_id)
    )
    semantic_traffic_constraints, semantic_traffic_reviews = _semantic_traffic_constraints(
        episode_dir=episode_dir,
        script=script,
        scene_entities=scene_entities,
        semantic_source_vehicle_ids=sorted(semantic_sources),
        seed_index=seed_index,
        planner=planner,
    )
    semantic_corridor_edges_by_vehicle_id = {
        semantic_internal_vehicle_id(source_id, seed_index): str(corridor.get("edge_id") or "")
        for source_id, corridor in semantic_corridors_by_source.items()
        if str(corridor.get("edge_id") or "")
    }
    for constraint in semantic_traffic_constraints:
        allowed_vehicle_ids = {str(vehicle_id) for vehicle_id in constraint.get("allowed_vehicle_ids") or [] if str(vehicle_id)}
        corridor_edges = sorted(
            {
                edge_id
                for vehicle_id, edge_id in semantic_corridor_edges_by_vehicle_id.items()
                if vehicle_id in allowed_vehicle_ids and edge_id
            }
        )
        if not corridor_edges:
            continue
        bbox_candidate_edges = sorted(
            set(str(edge_id) for edge_id in constraint.get("protected_edges") or [] if str(edge_id))
        )
        constraint["protected_edges"] = corridor_edges
        constraint["spatial_bbox_candidate_edges"] = bbox_candidate_edges
        constraint["semantic_corridor_protected_edges"] = corridor_edges
    semantic_core_edges = {
        str(edge_id)
        for constraint in semantic_traffic_constraints
        if str(constraint.get("source") or "") == "event_realization_vehicle_window"
        for edge_id in constraint.get("protected_edges") or []
    }
    core_edges = sorted(set(core_edges) | semantic_core_edges)
    spatial_scope["core_sumo_edges"] = core_edges
    spatial_scope["core_sumo_lanes"] = sorted(
        str(planner.edges[edge_id].lane_id)
        for edge_id in core_edges
        if edge_id in planner.edges and str(planner.edges[edge_id].lane_id)
    )

    direction_order = {
        "inbound_to_roi": ("inbound", "cross", "outbound"),
        "outbound_from_roi": ("outbound", "cross", "inbound"),
        "balanced_cross_traffic": ("cross", "inbound", "outbound"),
    }[profile.direction_bias]
    all_directions = ("inbound", "outbound", "cross")
    reverse_adjacency = _reverse_adjacency(planner)
    from_pools = {
        direction: [
            edge
            for edge in _endpoint_edge_candidates(planner, spatial_scope, center_xy, direction, for_from=True)
            if edge.edge_id not in road_closed_edge_set
        ]
        for direction in all_directions
    }
    to_pools = {
        direction: [
            edge
            for edge in _endpoint_edge_candidates(planner, spatial_scope, center_xy, direction, for_from=False)
            if edge.edge_id not in road_closed_edge_set
        ]
        for direction in all_directions
    }
    from_pos_by_edge = {
        edge.edge_id: _edge_depart_s_outside_roi(edge, spatial_scope, center_xy)
        for pool in from_pools.values()
        for edge in pool
    }
    roi_entry_edges = [
        planner.edges[edge_id]
        for edge_id in roi_visible_edges
        if edge_id in planner.edges and _edge_depart_s_before_bbox(planner.edges[edge_id], spatial_scope.get("bbox_enu_m") or []) is not None
    ]
    roi_pos_by_edge = {
        edge.edge_id: _edge_depart_s_before_bbox(edge, spatial_scope.get("bbox_enu_m") or [])
        for edge in roi_entry_edges
    }
    def roi_route_crossing_depart_pos(edge: SumoEdge) -> float | None:
        if roi_bbox and _edge_intersects_bbox(edge, roi_bbox):
            return _edge_depart_s_before_bbox(edge, roi_bbox)
        return _edge_depart_s_outside_roi(edge, spatial_scope, center_xy)

    roi_required_band_by_edge = {
        str(roi_edge_id): [str(roi_edge_id)]
        for roi_edge_id in roi_visible_edges
    }
    roi_route_required_edges = _sorted_edge_ids(
        edge_id
        for edge_ids in roi_required_band_by_edge.values()
        for edge_id in edge_ids
    )
    spatial_scope["roi_route_required_sumo_edges"] = roi_route_required_edges

    vehicles: list[dict[str, Any]] = []
    endpoint_cursor = 0

    def next_endpoint(preferred_direction: str, vehicle_index: int, *, prefer_outside_expanded: bool = False) -> dict[str, Any]:
        nonlocal endpoint_cursor
        directions = [preferred_direction, *direction_order, "inbound", "outbound", "cross"]
        seen: set[str] = set()

        def endpoint_priority(edge: SumoEdge) -> tuple[int, float, str]:
            depart_pos = from_pos_by_edge.get(edge.edge_id)
            if depart_pos is None:
                return (3, 0.0, edge.edge_id)
            xy = _point_at_edge_s(edge, float(depart_pos))
            expanded_bbox = spatial_scope.get("expanded_bbox_enu_m") or []
            roi_bbox_local = spatial_scope.get("bbox_enu_m") or []
            if expanded_bbox and not _point_in_bbox(xy, expanded_bbox):
                bucket = 0
            elif roi_bbox_local and not _point_in_bbox(xy, roi_bbox_local):
                bucket = 1
            else:
                bucket = 2
            distance = math.hypot(float(xy[0]) - center_xy[0], float(xy[1]) - center_xy[1])
            return (bucket, -distance, edge.edge_id)

        for direction in directions:
            if not direction or direction in seen:
                continue
            seen.add(direction)
            from_candidates = from_pools.get(direction) or []
            to_candidates = to_pools.get(direction) or []
            if not from_candidates or not to_candidates:
                continue
            start_edge = from_candidates[(endpoint_cursor + vehicle_index) % len(from_candidates)]
            depart_pos = from_pos_by_edge.get(start_edge.edge_id)
            if depart_pos is None:
                continue
            destination_edges = [edge for edge in to_candidates if edge.edge_id != start_edge.edge_id]
            if not destination_edges:
                continue
            destination = destination_edges[((endpoint_cursor + vehicle_index) // max(1, len(from_candidates))) % len(destination_edges)]
            depart_xy = _point_at_edge_s(start_edge, depart_pos)
            candidate_from_edges = _unique_edges(
                [
                    start_edge,
                    *(
                        sorted(from_candidates, key=endpoint_priority)[:80]
                        if prefer_outside_expanded
                        else []
                    ),
                    *from_candidates[:80],
                ]
            )
            candidate_depart_pos_by_edge = {
                edge.edge_id: round(float(from_pos_by_edge[edge.edge_id]), 6)
                for edge in candidate_from_edges
                if from_pos_by_edge.get(edge.edge_id) is not None
            }
            candidate_edge_ids = [
                start_edge.edge_id,
                *[
                    edge.edge_id
                    for edge in candidate_from_edges
                    if edge.edge_id != start_edge.edge_id and edge.edge_id in candidate_depart_pos_by_edge
                ],
            ]
            endpoint_cursor += 1
            return {
                "from_edge": start_edge.edge_id,
                "from_edge_candidates": candidate_edge_ids,
                "from_edge_depart_pos_by_edge": candidate_depart_pos_by_edge,
                "from_lane": start_edge.lane_id,
                "from_lane_index": 0,
                "from_pos": round(float(depart_pos), 6),
                "from_xy_enu_m": [round(float(depart_xy[0]), 6), round(float(depart_xy[1]), 6)],
                "to_edge": destination.edge_id,
                "to_edge_candidates": [destination.edge_id, *[edge.edge_id for edge in destination_edges[:120] if edge.edge_id != destination.edge_id]],
                "direction_role": _edge_direction_role(start_edge, center_xy),
            }
        raise RuntimeError(f"{episode_id}: no off-ROI from/to endpoint pair for explicit vehicle plan")

    directed_path_cache: dict[tuple[str, str], list[str]] = {}

    def directed_path(start_edge_id: str, target_edge_id: str) -> list[str]:
        cache_key = (str(start_edge_id), str(target_edge_id))
        cached = directed_path_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        if start_edge_id == target_edge_id:
            directed_path_cache[cache_key] = [str(start_edge_id)]
            return [str(start_edge_id)]
        distances: dict[str, float] = {str(start_edge_id): 0.0}
        predecessor: dict[str, str] = {}
        queue: list[tuple[float, str]] = [(0.0, str(start_edge_id))]
        while queue:
            cost, current_edge_id = heapq.heappop(queue)
            if cost > distances.get(current_edge_id, math.inf) + 1e-12:
                continue
            if current_edge_id == target_edge_id:
                break
            for adjacent_edge_id in planner.adjacency.get(current_edge_id, []):
                adjacent_edge_id = str(adjacent_edge_id)
                adjacent_edge = planner.edges.get(adjacent_edge_id)
                if (
                    adjacent_edge is None
                    or not _edge_allows_vehicle(adjacent_edge)
                    or adjacent_edge_id in road_closed_edge_set
                ):
                    continue
                edge_speed_mps = max(1.0, float(adjacent_edge.speed_mps))
                adjacent_cost = cost + float(adjacent_edge.length_m) / edge_speed_mps
                if adjacent_cost + 1e-12 >= distances.get(adjacent_edge_id, math.inf):
                    continue
                distances[adjacent_edge_id] = adjacent_cost
                predecessor[adjacent_edge_id] = current_edge_id
                heapq.heappush(queue, (adjacent_cost, adjacent_edge_id))
        if target_edge_id not in distances:
            directed_path_cache[cache_key] = []
            return []
        path = [str(target_edge_id)]
        while path[-1] != str(start_edge_id):
            path.append(predecessor[path[-1]])
        path.reverse()
        directed_path_cache[cache_key] = path
        return list(path)

    def approach_time_s(start_edge: SumoEdge, roi_edge: SumoEdge, path: Sequence[str]) -> float:
        depart_pos = roi_route_crossing_depart_pos(start_edge)
        if depart_pos is None or not path:
            return math.inf
        total_s = 0.0
        for edge_index, edge_id in enumerate(path):
            edge = planner.edges[edge_id]
            start_s = float(depart_pos) if edge_index == 0 else 0.0
            end_s = float(edge.length_m)
            if edge_id == roi_edge.edge_id:
                roi_entry_s = _edge_s_inside_bbox(edge, roi_bbox)
                if roi_entry_s is not None and float(roi_entry_s) >= start_s:
                    end_s = float(roi_entry_s)
            total_s += max(0.0, end_s - start_s) / max(1.0, float(edge.speed_mps))
            if edge_id == roi_edge.edge_id:
                break
        return total_s

    corridor_edge_cache: dict[tuple[str, bool], list[SumoEdge]] = {}
    roi_pair_cache: dict[
        tuple[tuple[str, ...], bool],
        list[tuple[float, float, SumoEdge, SumoEdge, SumoEdge, list[str]]],
    ] = {}

    def next_roi_endpoint(
        preferred_direction: str,
        vehicle_index: int,
        roi_edge_ids: Sequence[str] | None = None,
        *,
        full_capture_presence_required: bool = False,
        forbidden_edges: Sequence[str] = (),
        target_speed_mps: float | None = None,
        core_entry_tick: int | None = None,
    ) -> dict[str, Any]:
        selected_roi_edge_ids = [str(edge_id) for edge_id in (roi_edge_ids or roi_visible_edges) if str(edge_id) in planner.edges]
        if not selected_roi_edge_ids:
            raise RuntimeError(f"{episode_id}: no ROI-visible SUMO endpoint candidates for explicit vehicle plan")
        preferred = str(preferred_direction or "")
        roi_bbox = spatial_scope.get("bbox_enu_m") or []

        def corridor_edges(roi_edge_id: str, *, incoming: bool) -> list[SumoEdge]:
            cache_key = (str(roi_edge_id), bool(incoming))
            if cache_key in corridor_edge_cache:
                return corridor_edge_cache[cache_key]
            adjacency = reverse_adjacency if incoming else planner.adjacency
            queue: list[tuple[str, int]] = [(roi_edge_id, 0)]
            visited = {roi_edge_id}
            candidates: list[tuple[int, float, str, SumoEdge]] = []
            while queue:
                current_edge_id, depth = queue.pop(0)
                if depth >= 16:
                    continue
                for candidate_id in adjacency.get(current_edge_id, []):
                    candidate_id = str(candidate_id)
                    if candidate_id in visited:
                        continue
                    visited.add(candidate_id)
                    queue.append((candidate_id, depth + 1))
                    edge = planner.edges.get(candidate_id)
                    if edge is None or not _edge_allows_vehicle(edge) or edge.edge_id in road_closed_edge_set:
                        continue
                    if incoming:
                        depart_s = roi_route_crossing_depart_pos(edge)
                        if depart_s is None:
                            continue
                        depart_xy = _point_at_edge_s(edge, depart_s)
                        if roi_bbox and _point_in_bbox(depart_xy, roi_bbox):
                            continue
                    elif roi_bbox and not any(not _point_in_bbox(point, roi_bbox) for point in edge.shape_xy):
                        continue
                    candidates.append((depth + 1, _edge_distance_to_center(edge, center_xy), edge.edge_id, edge))
                if len(candidates) >= 80:
                    break
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            result = [item[3] for item in candidates]
            corridor_edge_cache[cache_key] = result
            return result

        pair_cache_key = (tuple(sorted(selected_roi_edge_ids)), bool(full_capture_presence_required))
        raw_connected_pairs = roi_pair_cache.get(pair_cache_key)
        if raw_connected_pairs is None:
            raw_connected_pairs = []
            for roi_edge_id in selected_roi_edge_ids:
                roi_edge = planner.edges.get(roi_edge_id)
                if roi_edge is None:
                    continue
                starts = corridor_edges(roi_edge_id, incoming=True)
                roi_depart_s = roi_route_crossing_depart_pos(roi_edge)
                if roi_depart_s is not None:
                    roi_depart_xy = _point_at_edge_s(roi_edge, roi_depart_s)
                    if not roi_bbox or not _point_in_bbox(roi_depart_xy, roi_bbox):
                        starts = _unique_edges([roi_edge, *starts])
                destinations = _unique_edges(
                    [roi_edge, *corridor_edges(roi_edge_id, incoming=False)]
                )
                roi_distance_m = _edge_distance_to_center(roi_edge, center_xy)
                for start in starts[:40]:
                    route_to_roi = directed_path(start.edge_id, roi_edge.edge_id)
                    if not route_to_roi:
                        continue
                    route_approach_time_s = approach_time_s(start, roi_edge, route_to_roi)
                    destination_limit = 80 if full_capture_presence_required else 40
                    for destination_edge in destinations[:destination_limit]:
                        if start.edge_id == destination_edge.edge_id:
                            continue
                        raw_connected_pairs.append(
                            (
                                route_approach_time_s,
                                roi_distance_m,
                                start,
                                destination_edge,
                                roi_edge,
                                route_to_roi,
                            )
                        )
            roi_pair_cache[pair_cache_key] = raw_connected_pairs
        connected_pairs = [
            (
                0 if _edge_direction_role(start, center_xy) == preferred else 1,
                route_approach_time_s,
                roi_distance_m,
                start,
                destination_edge,
                roi_edge,
                route_to_roi,
            )
            for route_approach_time_s, roi_distance_m, start, destination_edge, roi_edge, route_to_roi in raw_connected_pairs
        ]
        connected_pairs.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3].edge_id,
                item[4].edge_id,
                item[5].edge_id,
            )
        )
        primary_corridors: list[
            tuple[int, float, float, SumoEdge, SumoEdge, SumoEdge, list[str]]
        ] = []
        repeated_destinations: list[
            tuple[int, float, float, SumoEdge, SumoEdge, SumoEdge, list[str]]
        ] = []
        seen_corridors: set[tuple[str, str]] = set()
        for pair in connected_pairs:
            corridor_key = (pair[3].edge_id, pair[5].edge_id)
            if corridor_key in seen_corridors:
                repeated_destinations.append(pair)
                continue
            seen_corridors.add(corridor_key)
            primary_corridors.append(pair)
        connected_pairs = [*primary_corridors, *repeated_destinations]
        source_presence_contract: dict[str, Any] | None = None
        if full_capture_presence_required:
            if target_speed_mps is None or core_entry_tick is None:
                raise RuntimeError(
                    f"{episode_id}: full-capture source presence requires target speed and core entry tick"
                )
            forbidden_edge_set = {str(edge_id) for edge_id in forbidden_edges if str(edge_id)}
            modeled_pairs: list[
                tuple[
                    tuple[int, float, float, SumoEdge, SumoEdge, SumoEdge, list[str]],
                    dict[str, Any],
                ]
            ] = []
            maximum_full_route_ticks = 0
            for pair in connected_pairs:
                _, _, _, candidate_start, candidate_destination, candidate_roi_edge, candidate_route_to_roi = pair
                full_route_edges = directed_path(candidate_start.edge_id, candidate_destination.edge_id)
                if not full_route_edges or set(full_route_edges) & forbidden_edge_set:
                    continue
                candidate_required_edges = set(
                    roi_required_band_by_edge.get(candidate_roi_edge.edge_id)
                    or [candidate_roi_edge.edge_id]
                )
                if not set(full_route_edges) & candidate_required_edges:
                    continue
                candidate_depart_pos = roi_route_crossing_depart_pos(candidate_start)
                if candidate_depart_pos is None:
                    continue
                candidate_depart_xy = _point_at_edge_s(candidate_start, candidate_depart_pos)
                candidate_endpoint = {
                    "from_pos": float(candidate_depart_pos),
                    "from_xy_enu_m": [float(candidate_depart_xy[0]), float(candidate_depart_xy[1])],
                    "route_to_roi_edges": list(candidate_route_to_roi),
                }
                approach_lead_ticks = _estimated_roi_route_lead_ticks(
                    planner,
                    candidate_endpoint,
                    roi_bbox,
                    float(target_speed_mps),
                )
                planned_release_tick = max(
                    -int(approach_lead_ticks),
                    int(core_entry_tick) - int(approach_lead_ticks),
                )
                full_route_travel_s = 0.0
                for edge_index, edge_id in enumerate(full_route_edges):
                    edge = planner.edges[edge_id]
                    start_s = float(candidate_depart_pos) if edge_index == 0 else 0.0
                    governed_speed_mps = max(
                        1.0,
                        min(
                            float(target_speed_mps),
                            float(edge.speed_mps) if float(edge.speed_mps) > 0.0 else float(target_speed_mps),
                        ),
                    )
                    full_route_travel_s += max(0.0, float(edge.length_m) - start_s) / governed_speed_mps
                full_route_free_flow_ticks = int(math.ceil(full_route_travel_s * float(TICK_HZ)))
                minimum_required_ticks = int(DURATION_TICKS - planned_release_tick + 1)
                maximum_full_route_ticks = max(maximum_full_route_ticks, full_route_free_flow_ticks)
                if full_route_free_flow_ticks < minimum_required_ticks:
                    continue
                modeled_pairs.append(
                    (
                        pair,
                        {
                            "policy": "full_capture_presence_from_route_travel_lower_bound_v1",
                            "capture_tick_range": [0, DURATION_TICKS],
                            "planned_release_tick": int(planned_release_tick),
                            "approach_lead_ticks": int(approach_lead_ticks),
                            "full_route_edges": list(full_route_edges),
                            "full_route_free_flow_ticks": int(full_route_free_flow_ticks),
                            "minimum_required_route_ticks": int(minimum_required_ticks),
                            "speed_upper_bound_mps": round(float(target_speed_mps), 6),
                        },
                    )
                )
            if not modeled_pairs:
                raise RuntimeError(
                    f"{episode_id}: source-required context vehicle has no directed route whose modeled "
                    f"lifetime covers capture ticks 0..{DURATION_TICKS}; "
                    f"maximum_full_route_ticks={maximum_full_route_ticks}"
                )
            connected_pairs = [item[0] for item in modeled_pairs]
            source_presence_contract = modeled_pairs[vehicle_index % len(modeled_pairs)][1]
        local_component_source = False
        if connected_pairs:
            _, _, _, start_edge, destination, roi_edge, route_to_roi = connected_pairs[vehicle_index % len(connected_pairs)]
            if full_capture_presence_required:
                from_candidate_edges = [start_edge]
                target_edges = [destination]
            else:
                same_roi_pairs = [item for item in connected_pairs if item[5].edge_id == roi_edge.edge_id]
                from_candidate_edges = _unique_edges(
                    [start_edge, *[item[3] for item in same_roi_pairs]]
                )[:220]
                target_edges = _unique_edges(
                    [destination, *[item[4] for item in same_roi_pairs]]
                )[:220]
        else:
            # Some imported road components are wholly contained by the
            # formal ROI and have no directed connection to an outside edge.
            # Such a component has a physical local traffic source at its
            # first edge; it must not be represented as a fabricated outside
            # approach on an unrelated road component.
            local_pairs: list[tuple[SumoEdge, SumoEdge]] = []
            selected_set = set(selected_roi_edge_ids)
            for source_edge_id in selected_roi_edge_ids:
                source_edge = planner.edges.get(source_edge_id)
                if source_edge is None:
                    continue
                for destination_edge_id in planner.adjacency.get(source_edge_id, []):
                    destination_edge = planner.edges.get(str(destination_edge_id))
                    if (
                        destination_edge is not None
                        and destination_edge.edge_id in selected_set
                        and _edge_allows_vehicle(destination_edge)
                        and _edge_s_inside_bbox(source_edge, roi_bbox) is not None
                    ):
                        local_pairs.append((source_edge, destination_edge))
            if not local_pairs:
                local_pairs = [
                    (edge, edge)
                    for edge_id in selected_roi_edge_ids
                    for edge in [planner.edges.get(edge_id)]
                    if edge is not None
                    and _edge_allows_vehicle(edge)
                    and all(_point_in_bbox(point, roi_bbox) for point in edge.shape_xy)
                ]
            if not local_pairs:
                raise RuntimeError(f"{episode_id}: ROI road component has no physical local traffic source")
            local_pairs.sort(key=lambda item: (item[0].edge_id, item[1].edge_id))
            start_edge, destination = local_pairs[vehicle_index % len(local_pairs)]
            roi_edge = start_edge
            route_to_roi = [start_edge.edge_id]
            from_candidate_edges = [start_edge]
            target_edges = [destination]
            local_component_source = True
        required_edge_hint = list(roi_required_band_by_edge.get(roi_edge.edge_id) or [roi_edge.edge_id])
        depart_pos = (
            _edge_s_inside_bbox(start_edge, roi_bbox)
            if local_component_source
            else roi_route_crossing_depart_pos(start_edge)
        )
        if depart_pos is None and not local_component_source:
            depart_pos = roi_pos_by_edge.get(start_edge.edge_id)
        if depart_pos is None:
            raise RuntimeError(
                f"{episode_id}: selected ROI background edge {start_edge.edge_id} "
                "has no valid depart point for its declared source scope"
            )
        depart_xy = _point_at_edge_s(start_edge, depart_pos)
        candidate_depart_pos_by_edge: dict[str, float] = {}
        for edge in from_candidate_edges:
            candidate_depart_pos = (
                _edge_s_inside_bbox(edge, roi_bbox)
                if local_component_source
                else roi_route_crossing_depart_pos(edge)
            )
            if candidate_depart_pos is None and not local_component_source:
                candidate_depart_pos = roi_pos_by_edge.get(edge.edge_id)
            if candidate_depart_pos is None:
                continue
            candidate_xy = _point_at_edge_s(edge, candidate_depart_pos)
            if not local_component_source and roi_bbox and _point_in_bbox(candidate_xy, roi_bbox):
                continue
            candidate_depart_pos_by_edge[edge.edge_id] = round(float(candidate_depart_pos), 6)
        if start_edge.edge_id not in candidate_depart_pos_by_edge:
            raise RuntimeError(f"{episode_id}: selected ROI background edge {start_edge.edge_id} has no outside-ROI candidate depart point")
        candidate_edge_ids = [
            start_edge.edge_id,
            *[
                edge.edge_id
                for edge in from_candidate_edges
                if edge.edge_id != start_edge.edge_id and edge.edge_id in candidate_depart_pos_by_edge
            ],
        ]
        return {
            "from_edge": start_edge.edge_id,
            "from_edge_candidates": candidate_edge_ids,
            "from_edge_depart_pos_by_edge": candidate_depart_pos_by_edge,
            "from_lane": start_edge.lane_id,
            "from_lane_index": 0,
            "from_pos": round(float(depart_pos), 6),
            "from_xy_enu_m": [round(float(depart_xy[0]), 6), round(float(depart_xy[1]), 6)],
            "to_edge": destination.edge_id,
            "to_edge_candidates": [destination.edge_id, *[edge.edge_id for edge in target_edges[:120] if edge.edge_id != destination.edge_id]],
            "direction_role": _edge_direction_role(start_edge, center_xy),
            "required_edge_hint": required_edge_hint,
            "roi_local_component_source": bool(local_component_source),
            "route_to_roi_edges": route_to_roi,
            **(
                {"source_presence_contract": source_presence_contract}
                if source_presence_contract is not None
                else {}
            ),
        }

    def direction_sequence(count: int) -> list[str]:
        weights = dict(profile.direction_distribution)
        counts = {direction: max(0, int(round(float(weights.get(direction, 0.0)) * int(count)))) for direction in ("inbound", "outbound", "cross")}
        while sum(counts.values()) < count:
            direction = max(("inbound", "outbound", "cross"), key=lambda item: (weights.get(item, 0.0), -counts[item], item))
            counts[direction] += 1
        while sum(counts.values()) > count:
            direction = max(("inbound", "outbound", "cross"), key=lambda item: (counts[item], -weights.get(item, 0.0), item))
            counts[direction] -= 1
        ordered: list[str] = []
        for direction in direction_order:
            ordered.extend([direction] * counts.get(direction, 0))
        for direction in ("inbound", "outbound", "cross"):
            if direction not in direction_order:
                ordered.extend([direction] * counts.get(direction, 0))
        return ordered[:count]

    binding_ticks_by_source: dict[str, list[int]] = {}
    for binding in bindings:
        binding_ticks_by_source.setdefault(str(binding["source_entity_id"]), []).extend(int(tick) for tick in binding.get("evidence_ticks") or [])

    semantic_reserved_edges: set[str] = set()
    episode_wide_protected_edges = sorted(
        {
            str(edge_id)
            for constraint in semantic_traffic_constraints
            if str(constraint.get("background_policy") or "") == "route_must_avoid_protected_edges_episode_wide"
            for edge_id in constraint.get("protected_edges") or []
            if str(edge_id)
        }
    )
    episode_wide_protected_edge_set = set(episode_wide_protected_edges)
    for constraint in semantic_traffic_constraints:
        # Signals and queues allow zero speed, so no finite clearance bound can
        # be proved from target speed. Treat the route as active for the whole
        # observation window and prove protection through route exclusion.
        constraint["background_protected_edge_traversal_ticks"] = DURATION_TICKS
        constraint["background_protected_edge_traversal_model"] = (
            "episode_window_upper_bound_without_positive_speed_floor_v1"
        )

    def roi_stop_candidates_for_edges(edge_ids: Sequence[str]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_edges: set[str] = set()
        for edge_id in edge_ids:
            edge = planner.edges.get(str(edge_id))
            if edge is None or edge.edge_id in seen_edges:
                continue
            seen_edges.add(edge.edge_id)
            stop_pos = _edge_s_inside_bbox(edge, spatial_scope.get("bbox_enu_m") or [])
            if stop_pos is None:
                continue
            candidates.append(
                {
                    "edgeID": edge.edge_id,
                    "pos": round(float(stop_pos), 6),
                    "laneIndex": 0,
                }
            )
        return candidates

    def add_roi_smoothing_stop(vehicle: dict[str, Any], edge_ids: Sequence[str]) -> None:
        if str(vehicle.get("traffic_role") or "") != "deterministic_background_vehicle":
            return
        if str(vehicle.get("traffic_coverage_scope") or "") not in {
            "roi_background_visible_flow",
            "roi_background_route_crossing_flow",
        }:
            return
        candidates = roi_stop_candidates_for_edges(edge_ids)
        if not candidates:
            return
        vehicle["roi_smoothing_stop"] = {
            "policy": "sumo_setStop_short_roi_queue_for_slot_coverage_v1",
            "duration_ticks": _context_free_flow_smoothing_stop_ticks(traffic_tuning),
            "target_tick": int(vehicle.get("expected_core_entry_tick") or 0),
            "candidates": candidates,
        }

    def apply_protected_edge_lifetime_guards(vehicle: dict[str, Any]) -> None:
        if str(vehicle.get("traffic_role") or "") != "deterministic_background_vehicle":
            return
        release_tick = int(vehicle.get("release_tick") or 0)
        expected_entry_tick = int(vehicle.get("expected_core_entry_tick") or 0)
        expected_exit_tick = int(
            vehicle.get("expected_core_exit_tick") or expected_entry_tick
        )
        forbidden_edges = {
            str(edge_id) for edge_id in vehicle.get("forbidden_edges") or [] if str(edge_id)
        }
        lifetime_guards: list[dict[str, Any]] = []
        for constraint in semantic_traffic_constraints:
            tick_range = [int(value) for value in constraint.get("tick_range") or []]
            protected_edges = {
                str(edge_id) for edge_id in constraint.get("protected_edges") or [] if str(edge_id)
            }
            if len(tick_range) < 2 or not protected_edges:
                continue
            protected_traversal_ticks = int(
                constraint.get("background_protected_edge_traversal_ticks") or 0
            )
            physical_active_end_tick = max(
                expected_exit_tick,
                expected_entry_tick + protected_traversal_ticks,
            )
            if release_tick > tick_range[1] or physical_active_end_tick < tick_range[0]:
                continue
            forbidden_edges.update(protected_edges)
            lifetime_guards.append(
                {
                    "constraint_id": str(constraint.get("constraint_id") or ""),
                    "protected_edges": sorted(protected_edges),
                    "protected_tick_range": tick_range,
                    "physical_active_tick_range": [
                        release_tick,
                        physical_active_end_tick,
                    ],
                    "policy": "release_to_route_and_protected_edge_clearance_envelope_v1",
                }
            )
        vehicle["forbidden_edges"] = sorted(forbidden_edges)
        if lifetime_guards:
            vehicle["protected_edge_lifetime_guards"] = lifetime_guards

    def keep_roi_release_after_protected_window(vehicle: dict[str, Any]) -> None:
        if str(vehicle.get("traffic_role") or "") != "deterministic_background_vehicle":
            return
        if str(vehicle.get("traffic_coverage_scope") or "") not in {
            "roi_background_visible_flow",
            "roi_background_route_crossing_flow",
        }:
            return
        apply_protected_edge_lifetime_guards(vehicle)
        required_edges = {str(edge_id) for edge_id in vehicle.get("required_edges") or [] if str(edge_id)}
        if not required_edges:
            return
        release_tick = int(vehicle.get("release_tick") or 0)
        expected_entry_tick = int(vehicle.get("expected_core_entry_tick") or 0)
        latest_protected_end: int | None = None
        for constraint in semantic_traffic_constraints:
            protected_edges = {str(edge_id) for edge_id in constraint.get("protected_edges") or [] if str(edge_id)}
            if not (required_edges & protected_edges):
                continue
            tick_range = [int(value) for value in constraint.get("tick_range") or []]
            if len(tick_range) < 2:
                continue
            protected_end = int(tick_range[1])
            if release_tick <= protected_end < expected_entry_tick:
                latest_protected_end = max(latest_protected_end or protected_end, protected_end)
        if latest_protected_end is None:
            return
        adjusted_release_tick = min(
            DURATION_TICKS,
            max(release_tick, latest_protected_end + 1 - ROI_POST_PROTECTED_RELEASE_LEAD_TICKS),
        )
        vehicle["release_tick"] = int(adjusted_release_tick)
        vehicle["release_time_s"] = round(adjusted_release_tick / float(TICK_HZ), 6)
        hold_policy = dict(vehicle.get("hold_policy") or {})
        hold_policy["release_tick"] = int(adjusted_release_tick)
        vehicle["hold_policy"] = hold_policy
        vehicle["release_adjustment"] = {
            "policy": "roi_vehicle_release_after_required_edge_protection_v1",
            "original_release_tick": int(release_tick),
            "protected_end_tick": int(latest_protected_end),
        }

    def forbidden_edges_for_window(semantic_actor: bool, entry_tick: int, exit_tick: int) -> list[str]:
        if semantic_actor:
            return []
        forbidden: set[str] = set(road_closed_edge_set)
        check_entry_tick = int(entry_tick) - _protected_window_post_route_guard_ticks(traffic_tuning)
        check_exit_tick = int(exit_tick) + _protected_window_pre_route_guard_ticks(traffic_tuning)
        forbidden.update(episode_wide_protected_edges)
        for constraint in semantic_traffic_constraints:
            tick_range = list(constraint.get("tick_range") or [])
            if len(tick_range) < 2:
                continue
            protected_traversal_ticks = int(
                constraint.get("background_protected_edge_traversal_ticks") or 0
            )
            physical_exit_tick = max(
                check_exit_tick,
                int(entry_tick) + protected_traversal_ticks,
            )
            if (
                check_entry_tick <= int(tick_range[1])
                and physical_exit_tick >= int(tick_range[0])
            ):
                forbidden.update(str(edge_id) for edge_id in constraint.get("protected_edges") or [])
        return sorted(forbidden)

    for index, entity in enumerate(source_vehicles):
        source_id = str(entity.get("entity_id") or "")
        binding_ticks = sorted(set(binding_ticks_by_source.get(source_id, [])))
        semantic_actor = source_id in semantic_sources
        source_presence_required = _source_presence_required(
            entity,
            semantic_actor=semantic_actor,
        )
        semantic_lifecycle = semantic_lifecycles_by_source.get(source_id) if semantic_actor else None
        if semantic_lifecycle:
            core_entry_tick = int(semantic_lifecycle["control_start_tick"])
            core_exit_tick = int(semantic_lifecycle["control_end_tick"])
        elif binding_ticks:
            core_entry_tick = max(0, min(DURATION_TICKS, min(binding_ticks) - PRE_EVENT_VISIBLE_MARGIN_TICKS))
            core_exit_tick = max(core_entry_tick + CORE_CROSSING_TICKS, min(DURATION_TICKS, max(binding_ticks) + POST_EVENT_VISIBLE_MARGIN_TICKS))
        else:
            slot_index = index % TRAFFIC_SLOT_COUNT
            slot_ticks = list(traffic_slots[slot_index]["core_entry_ticks"])
            core_entry_tick = int(slot_ticks[index % len(slot_ticks)])
            core_exit_tick = min(DURATION_TICKS, core_entry_tick + CORE_CROSSING_TICKS)
        preferred_direction = direction_order[index % len(direction_order)]
        semantic_corridor = semantic_corridors_by_source.get(source_id) if semantic_actor else None
        forbidden_edges = forbidden_edges_for_window(semantic_actor, core_entry_tick, core_exit_tick)
        endpoint = (
            next_roi_endpoint(
                preferred_direction,
                index,
                full_capture_presence_required=True,
                forbidden_edges=forbidden_edges,
                target_speed_mps=_flow_speed_target(profile, flow_mode, index),
                core_entry_tick=core_entry_tick,
            )
            if source_presence_required
            else next_endpoint(preferred_direction, index, prefer_outside_expanded=semantic_actor)
        )
        if semantic_corridor and semantic_corridor.get("edge_id"):
            semantic_reserved_edges.add(str(semantic_corridor["edge_id"]))
        vehicle = _vehicle_plan_record(
            planner=planner,
            episode_dir=episode_dir,
            scenario_id=scenario_id,
            seed_index=seed_index,
            profile=profile,
            source_entity=entity,
            vehicle_index=index,
            endpoint=endpoint,
            direction_role=str(endpoint.get("direction_role") or preferred_direction),
            flow_mode=flow_mode,
            semantic_actor=semantic_actor,
            binding_ticks=binding_ticks,
            traffic_slot_index=None,
            core_entry_tick=core_entry_tick,
            core_exit_tick=core_exit_tick,
            forbidden_edges=forbidden_edges,
            coverage_scope=(
                "semantic_event_actor"
                if semantic_actor
                else "roi_background_route_crossing_flow"
                if source_presence_required
                else "source_background_context_flow"
            ),
            required_edges=(
                list(endpoint.get("required_edge_hint") or [])
                if source_presence_required
                else []
            ),
            spawn_scope="semantic_control_window" if semantic_actor else "outside_roi",
            visible_bbox_enu_m=spatial_scope.get("bbox_enu_m") or [],
            semantic_corridor=semantic_corridor,
            long_roi_release_lead=bool(set(forbidden_edges) & episode_wide_protected_edge_set),
            traffic_tuning=traffic_tuning,
            semantic_lifecycle=semantic_lifecycle,
        )
        if source_presence_required:
            add_roi_smoothing_stop(vehicle, vehicle.get("required_edges") or [])
            apply_protected_edge_lifetime_guards(vehicle)
        vehicles.append(vehicle)

    _separate_same_lane_semantic_entries(planner, vehicles)

    for slot in traffic_slots:
        slot_index = int(slot["slot_index"])
        entry_ticks = list(slot["core_entry_ticks"])
        direction_slots = direction_sequence(len(entry_ticks))
        for local_index, core_entry_tick in enumerate(entry_ticks):
            index = len(vehicles)
            preferred_direction = direction_slots[local_index % len(direction_slots)] if direction_slots else direction_order[index % len(direction_order)]
            core_exit_tick = min(DURATION_TICKS, int(core_entry_tick) + CORE_CROSSING_TICKS)
            forbidden_edges = forbidden_edges_for_window(False, int(core_entry_tick), core_exit_tick)
            all_roi_visible_edges = [str(edge_id) for edge_id in spatial_scope.get("roi_visible_sumo_edges") or [] if str(edge_id)]
            all_roi_required_edges = [str(edge_id) for edge_id in spatial_scope.get("roi_route_required_sumo_edges") or all_roi_visible_edges if str(edge_id)]
            narrow_roi_visible_flow = len(all_roi_visible_edges) <= 2
            roi_vehicle_cap_for_slot = _roi_route_crossing_cap_per_slot(
                entry_tick_count=len(entry_ticks),
                roi_visible_edge_count=len(all_roi_visible_edges),
                traffic_tuning=traffic_tuning,
            )
            roi_visible_edges_for_endpoint = [
                str(edge_id)
                for edge_id in all_roi_visible_edges
                if str(edge_id) and str(edge_id) not in set(forbidden_edges)
            ]
            roi_required_edges = [
                str(edge_id)
                for edge_id in all_roi_required_edges
                if str(edge_id) and str(edge_id) not in set(forbidden_edges)
            ]
            assigned_roi_required_edges = (
                _sorted_edge_ids(
                    [
                        roi_required_edges[local_index % len(roi_required_edges)],
                        *roi_required_edges,
                    ]
                )
                if roi_required_edges
                else []
            )
            roi_background_allowed = bool(roi_visible_edges_for_endpoint and roi_required_edges) and local_index < roi_vehicle_cap_for_slot
            if roi_background_allowed:
                endpoint = next_roi_endpoint(preferred_direction, local_index, roi_visible_edges_for_endpoint)
                local_component_source = bool(endpoint.get("roi_local_component_source"))
                coverage_scope = (
                    "roi_background_local_component_flow"
                    if local_component_source
                    else "roi_background_route_crossing_flow"
                )
                required_edges = list(endpoint.get("required_edge_hint") or assigned_roi_required_edges)
                spawn_scope = "roi_local_component_source" if local_component_source else "outside_roi"
            else:
                endpoint = next_endpoint(preferred_direction, index)
                coverage_scope = "extended_context_flow_protected_window"
                required_edges = list(spatial_scope.get("expanded_sumo_edges") or spatial_scope.get("core_sumo_edges") or [])
                spawn_scope = "outside_roi"
            vehicle = _vehicle_plan_record(
                planner=planner,
                episode_dir=episode_dir,
                scenario_id=scenario_id,
                seed_index=seed_index,
                profile=profile,
                source_entity=None,
                vehicle_index=index,
                endpoint=endpoint,
                direction_role=str(endpoint.get("direction_role") or preferred_direction),
                flow_mode=flow_mode,
                semantic_actor=False,
                binding_ticks=[],
                traffic_slot_index=slot_index,
                core_entry_tick=int(core_entry_tick),
                core_exit_tick=core_exit_tick,
                forbidden_edges=forbidden_edges,
                coverage_scope=coverage_scope,
                required_edges=required_edges,
                spawn_scope=spawn_scope,
                visible_bbox_enu_m=spatial_scope.get("bbox_enu_m") or [],
                long_roi_release_lead=bool(set(forbidden_edges) & episode_wide_protected_edge_set),
                traffic_tuning=traffic_tuning,
            )
            if (
                local_index < _roi_route_crossing_smoothing_max_per_slot(traffic_tuning)
                and not narrow_roi_visible_flow
            ):
                add_roi_smoothing_stop(vehicle, required_edges)
            keep_roi_release_after_protected_window(vehicle)
            vehicles.append(vehicle)
            slot["vehicle_ids"].append(str(vehicle["vehicle_id"]))

    roi_visible_slot_fill_ticks = _optional_int_sequence(
        traffic_tuning.get("roi_visible_slot_fill_ticks"),
        [],
    )
    if roi_visible_slot_fill_ticks:
        roi_visible_slot_fill_from_edges = [
            str(edge_id)
            for edge_id in traffic_tuning.get("roi_visible_slot_fill_from_edges") or []
            if str(edge_id)
        ]
        roi_visible_slot_fill_to_edges = [
            str(edge_id)
            for edge_id in traffic_tuning.get("roi_visible_slot_fill_to_edges") or []
            if str(edge_id)
        ]
        roi_fill_visible_edges = [
            str(edge_id)
            for edge_id in spatial_scope.get("roi_visible_sumo_edges") or []
            if str(edge_id)
        ]
        roi_fill_required_edges = [
            str(edge_id)
            for edge_id in spatial_scope.get("roi_route_required_sumo_edges") or roi_fill_visible_edges
            if str(edge_id)
        ]
        roi_visible_fill_lead_ticks = _optional_int(
            traffic_tuning.get("roi_visible_slot_fill_release_lead_ticks")
        )
        for fill_index, fill_tick in enumerate(roi_visible_slot_fill_ticks):
            if not (roi_fill_visible_edges and roi_fill_required_edges):
                continue
            index = len(vehicles)
            core_entry_tick = max(0, min(DURATION_TICKS, int(fill_tick)))
            slot_index = _traffic_slot_index_for_tick(core_entry_tick)
            preferred_direction = direction_order[(index + fill_index) % len(direction_order)]
            core_exit_tick = min(DURATION_TICKS, core_entry_tick + CORE_CROSSING_TICKS)
            forbidden_edges = forbidden_edges_for_window(False, core_entry_tick, core_exit_tick)
            visible_edges_for_endpoint = [
                edge_id for edge_id in roi_fill_visible_edges if edge_id not in set(forbidden_edges)
            ]
            required_edges = [
                edge_id for edge_id in roi_fill_required_edges if edge_id not in set(forbidden_edges)
            ]
            if not (visible_edges_for_endpoint and required_edges):
                continue
            assigned_required_edges = _sorted_edge_ids(
                [required_edges[(index + fill_index) % len(required_edges)], *required_edges]
            )
            endpoint = next_roi_endpoint(preferred_direction, fill_index, visible_edges_for_endpoint)
            if roi_visible_slot_fill_from_edges:
                forced_from_edge_id = roi_visible_slot_fill_from_edges[
                    fill_index % len(roi_visible_slot_fill_from_edges)
                ]
                forced_from_edge = planner.edges.get(forced_from_edge_id)
                if forced_from_edge is not None:
                    forced_depart_pos = roi_route_crossing_depart_pos(forced_from_edge)
                    if forced_depart_pos is not None:
                        forced_depart_xy = _point_at_edge_s(forced_from_edge, forced_depart_pos)
                        if not (roi_bbox and _point_in_bbox(forced_depart_xy, roi_bbox)):
                            depart_pos_by_edge = dict(endpoint.get("from_edge_depart_pos_by_edge") or {})
                            depart_pos_by_edge[forced_from_edge.edge_id] = round(float(forced_depart_pos), 6)
                            endpoint["from_edge"] = forced_from_edge.edge_id
                            endpoint["from_lane"] = forced_from_edge.lane_id
                            endpoint["from_lane_index"] = 0
                            endpoint["from_pos"] = round(float(forced_depart_pos), 6)
                            endpoint["from_xy_enu_m"] = [
                                round(float(forced_depart_xy[0]), 6),
                                round(float(forced_depart_xy[1]), 6),
                            ]
                            endpoint["from_edge_depart_pos_by_edge"] = depart_pos_by_edge
                            endpoint["from_edge_candidates"] = [
                                forced_from_edge.edge_id,
                                *[
                                    str(edge_id)
                                    for edge_id in endpoint.get("from_edge_candidates") or []
                                    if str(edge_id) and str(edge_id) != forced_from_edge.edge_id
                                ],
                            ]
                            endpoint["direction_role"] = _edge_direction_role(forced_from_edge, center_xy)
            if roi_visible_slot_fill_to_edges:
                forced_to_edge_id = roi_visible_slot_fill_to_edges[
                    fill_index % len(roi_visible_slot_fill_to_edges)
                ]
                if forced_to_edge_id in planner.edges and forced_to_edge_id != str(endpoint.get("from_edge") or ""):
                    endpoint["to_edge"] = forced_to_edge_id
                    endpoint["to_edge_candidates"] = [
                        forced_to_edge_id,
                        *[
                            str(edge_id)
                            for edge_id in endpoint.get("to_edge_candidates") or []
                            if str(edge_id) and str(edge_id) != forced_to_edge_id
                        ],
                    ]
            vehicle = _vehicle_plan_record(
                planner=planner,
                episode_dir=episode_dir,
                scenario_id=scenario_id,
                seed_index=seed_index,
                profile=profile,
                source_entity=None,
                vehicle_index=index,
                endpoint=endpoint,
                direction_role=str(endpoint.get("direction_role") or preferred_direction),
                flow_mode=flow_mode,
                semantic_actor=False,
                binding_ticks=[],
                traffic_slot_index=slot_index,
                core_entry_tick=core_entry_tick,
                core_exit_tick=core_exit_tick,
                forbidden_edges=forbidden_edges,
                coverage_scope="roi_background_visible_flow",
                required_edges=list(endpoint.get("required_edge_hint") or assigned_required_edges),
                spawn_scope="outside_roi_boundary_inside_extended_approach",
                visible_bbox_enu_m=spatial_scope.get("bbox_enu_m") or [],
                long_roi_release_lead=False,
                roi_visible_release_lead_ticks=roi_visible_fill_lead_ticks,
                traffic_tuning=traffic_tuning,
            )
            vehicle["coverage_smoothing_role"] = "scenario_roi_visible_slot_fill"
            if not bool(traffic_tuning.get("roi_visible_slot_fill_disable_smoothing_stop")):
                add_roi_smoothing_stop(vehicle, vehicle.get("required_edges") or assigned_required_edges)
            keep_roi_release_after_protected_window(vehicle)
            vehicles.append(vehicle)
            if slot_index in traffic_slot_by_index:
                traffic_slot_by_index[slot_index].setdefault("vehicle_ids", []).append(str(vehicle["vehicle_id"]))

    if str(flow_mode or "") == "context_free_flow":
        roi_required_edges = [str(edge_id) for edge_id in spatial_scope.get("roi_route_required_sumo_edges") or spatial_scope.get("roi_visible_sumo_edges") or [] if str(edge_id)]
        smoothing_roi_edges = [str(edge_id) for edge_id in spatial_scope.get("roi_visible_sumo_edges") or [] if str(edge_id)]
        smoothing_max_per_slot = _optional_int(traffic_tuning.get("context_free_flow_smoothing_max_per_slot"))
        smoothing_counts_by_slot: Counter[int] = Counter()
        for smoothing_entry_tick in _context_free_flow_smoothing_entry_ticks(traffic_tuning):
            if not roi_required_edges:
                continue
            index = len(vehicles)
            preferred_direction = direction_order[index % len(direction_order)]
            core_entry_tick = max(0, min(DURATION_TICKS, int(smoothing_entry_tick)))
            slot_index = _traffic_slot_index_for_tick(core_entry_tick)
            if smoothing_max_per_slot is not None and smoothing_counts_by_slot[slot_index] >= int(smoothing_max_per_slot):
                continue
            core_exit_tick = min(DURATION_TICKS, core_entry_tick + CORE_CROSSING_TICKS)
            forbidden_edges = forbidden_edges_for_window(False, core_entry_tick, core_exit_tick)
            if set(forbidden_edges) - road_closed_edge_set:
                continue
            assigned_roi_required_edges = _sorted_edge_ids([roi_required_edges[index % len(roi_required_edges)], *roi_required_edges])
            endpoint = next_roi_endpoint(
                preferred_direction,
                smoothing_counts_by_slot[slot_index],
                smoothing_roi_edges or assigned_roi_required_edges,
            )
            vehicle = _vehicle_plan_record(
                planner=planner,
                episode_dir=episode_dir,
                scenario_id=scenario_id,
                seed_index=seed_index,
                profile=profile,
                source_entity=None,
                vehicle_index=index,
                endpoint=endpoint,
                direction_role=str(endpoint.get("direction_role") or preferred_direction),
                flow_mode=flow_mode,
                semantic_actor=False,
                binding_ticks=[],
                traffic_slot_index=slot_index,
                core_entry_tick=core_entry_tick,
                core_exit_tick=core_exit_tick,
                forbidden_edges=forbidden_edges,
                coverage_scope="roi_background_route_crossing_flow",
                required_edges=assigned_roi_required_edges,
                spawn_scope="outside_roi",
                visible_bbox_enu_m=spatial_scope.get("bbox_enu_m") or [],
                long_roi_release_lead=False,
                traffic_tuning=traffic_tuning,
            )
            vehicle["coverage_smoothing_role"] = "context_free_flow_roi_midlate_gap_fill"
            add_roi_smoothing_stop(vehicle, assigned_roi_required_edges)
            vehicles.append(vehicle)
            smoothing_counts_by_slot[slot_index] += 1
            if slot_index in traffic_slot_by_index:
                traffic_slot_by_index[slot_index].setdefault("vehicle_ids", []).append(str(vehicle["vehicle_id"]))

    accepted_target = _accepted_background_count_target(scenario_class, profile.seed_index, flow_mode)
    replenishment_count = max(6, int(math.ceil(float(accepted_target) * BACKGROUND_REPLENISHMENT_FRACTION)))
    replenishment_directions = direction_sequence(replenishment_count)
    for local_index in range(replenishment_count):
        index = len(vehicles)
        preferred_direction = replenishment_directions[local_index % len(replenishment_directions)] if replenishment_directions else direction_order[index % len(direction_order)]
        core_entry_tick = max(0, min(DURATION_TICKS, 20 + int(local_index * max(4, DURATION_TICKS // max(1, replenishment_count)))))
        core_exit_tick = min(DURATION_TICKS, core_entry_tick + CORE_CROSSING_TICKS)
        endpoint = next_endpoint(preferred_direction, index)
        forbidden_edges = forbidden_edges_for_window(False, core_entry_tick, core_exit_tick)
        vehicle = _vehicle_plan_record(
            planner=planner,
            episode_dir=episode_dir,
            scenario_id=scenario_id,
            seed_index=seed_index,
            profile=profile,
            source_entity=None,
            vehicle_index=index,
            endpoint=endpoint,
            direction_role=str(endpoint.get("direction_role") or preferred_direction),
            flow_mode=flow_mode,
            semantic_actor=False,
            binding_ticks=[],
            traffic_slot_index=None,
            core_entry_tick=core_entry_tick,
            core_exit_tick=core_exit_tick,
            forbidden_edges=forbidden_edges,
            coverage_scope="accepted_background_replenishment_context_flow",
            required_edges=[],
            spawn_scope="outside_roi",
            visible_bbox_enu_m=spatial_scope.get("bbox_enu_m") or [],
            long_roi_release_lead=bool(set(forbidden_edges) & episode_wide_protected_edge_set),
            traffic_tuning=traffic_tuning,
        )
        vehicles.append(vehicle)

    for vehicle in vehicles:
        apply_protected_edge_lifetime_guards(vehicle)

    endpoint_edges = sorted(
        {
            edge_id
            for vehicle in vehicles
            for edge_id in [
                vehicle.get("from_edge"),
                vehicle.get("to_edge"),
                *(vehicle.get("from_edge_candidates") or []),
                *(vehicle.get("to_edge_candidates") or []),
                *(vehicle.get("required_edges") or []),
            ]
            if str(edge_id)
        }
    )
    allowed_edges = sorted(set(endpoint_edges) | set(core_edges) | set(expanded_edges))
    spatial_scope["allowed_sumo_edges"] = allowed_edges
    spatial_scope["allowed_sumo_lanes"] = sorted(
        {
            str(planner.edges[edge_id].lane_id)
            for edge_id in allowed_edges
            if edge_id in planner.edges and str(planner.edges[edge_id].lane_id)
        }
    )
    plan = {
        "schema": SCHEMA,
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "traffic_geometry": planner.traffic_geometry_provenance,
        "seed_profile": {
            "seed_index": profile.seed_index,
            "seed_label": profile.seed_label,
            "profile_id": profile.profile_id,
            "direction_bias": profile.direction_bias,
        },
        "duration_ticks": DURATION_TICKS,
        "tick_hz": TICK_HZ,
        "vehicle_source_policy": VEHICLE_SOURCE_POLICY,
        "scenario_vehicle_class": scenario_class,
        "traffic_flow_mode": flow_mode,
        "traffic_case": {
            "policy": "semantic_vehicle_first_principle_v1",
            "case_type": "semantic_vehicle" if has_semantic_vehicle_case else "background_only",
            "has_semantic_vehicle": bool(has_semantic_vehicle_case),
            "semantic_source_vehicle_ids": sorted(semantic_sources),
            "protection_policy": (
                "protect_only_semantic_vehicle_spacetime_windows"
                if has_semantic_vehicle_case
                else "no_protected_edges_background_flow_only"
            ),
        },
        "traffic_tuning": traffic_tuning,
        "minimum_vehicle_count": target_vehicle_count,
        "minimum_background_vehicle_count": scheduled_background_count,
        "scenario_roster_floor": minimum_vehicle_count,
        "target_vehicle_count": target_vehicle_count,
        "spatial_scope": spatial_scope,
        "traffic_profile": {
            "profile_id": profile.profile_id,
            "seed_label": profile.seed_label,
            "flow_mode": flow_mode,
            "schedule_policy": "nine_100_tick_slots_core_roi_passage_v2",
            "slot_count": TRAFFIC_SLOT_COUNT,
            "slot_duration_ticks": TRAFFIC_SLOT_TICKS,
            "core_crossing_duration_ticks": CORE_CROSSING_TICKS,
            "background_vehicles_per_slot": _slot_background_count(scenario_class, profile.seed_index, flow_mode),
            "accepted_background_vehicle_target": _accepted_background_count_target(
                scenario_class,
                profile.seed_index,
                flow_mode,
            ),
            "seed_semantics": {
                "seed00": "morning_peak_inbound_to_core_depart_early_dense",
                "seed01": "midday_balanced_depart_spread",
                "seed02": "evening_peak_outbound_from_core_reverse_of_seed00",
            },
            "direction_distribution": profile.direction_distribution,
            "target_speed_mps": profile.target_speed_mps,
            "slot_entry_jitter_ticks": list(profile.slot_entry_jitter_ticks),
        },
        "traffic_slots": traffic_slots,
        "semantic_traffic_constraints": semantic_traffic_constraints,
        "semantic_traffic_reviews": semantic_traffic_reviews,
        "road_semantics": road_semantics,
        "semantic_reserved_edges": sorted(semantic_reserved_edges),
        "flow_groups": _flow_groups_payload(vehicles),
        "vehicles": vehicles,
        "semantic_bindings": bindings,
        "review_artifacts": {
            "json": REVIEW_JSON_FILENAME,
            "markdown": REVIEW_MD_FILENAME,
        },
    }
    validate_explicit_vehicle_plan(plan, planner=planner)
    return plan


def validate_explicit_vehicle_plan(plan: dict[str, Any], *, planner: SumoGroundFlowPlanner) -> None:
    episode_id = str(plan.get("episode_id") or "<unknown>")
    vehicles = list(plan.get("vehicles") or [])
    minimum = int(plan.get("minimum_vehicle_count") or 0)
    if len(vehicles) < minimum:
        raise RuntimeError(f"{episode_id}: explicit vehicle plan has {len(vehicles)} vehicles, below minimum {minimum}")
    ids = [str(vehicle.get("vehicle_id") or "") for vehicle in vehicles]
    duplicates = sorted(vehicle_id for vehicle_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"{episode_id}: duplicate explicit SUMO vehicle ids: {duplicates[:8]}")
    spatial_scope = dict(plan.get("spatial_scope") or {})
    vehicles_by_id = {str(vehicle.get("vehicle_id") or ""): dict(vehicle) for vehicle in vehicles}
    semantic_vehicle_ids = {
        str(vehicle.get("vehicle_id") or "")
        for vehicle in vehicles
        if str(vehicle.get("traffic_role") or "") == "semantic_vehicle" and str(vehicle.get("vehicle_id") or "")
    }
    traffic_case = dict(plan.get("traffic_case") or {})
    has_semantic_vehicle_case = bool(traffic_case.get("has_semantic_vehicle")) or bool(semantic_vehicle_ids)
    if not has_semantic_vehicle_case and plan.get("semantic_traffic_constraints"):
        raise RuntimeError(f"{episode_id}: background-only traffic case must not generate protected semantic traffic constraints")
    for constraint in plan.get("semantic_traffic_constraints") or []:
        allowed_vehicle_ids = {str(vehicle_id) for vehicle_id in constraint.get("allowed_vehicle_ids") or [] if str(vehicle_id)}
        if not allowed_vehicle_ids:
            raise RuntimeError(f"{episode_id}: protected constraint {constraint.get('constraint_id')} lacks semantic allowed vehicle ids")
        if not (allowed_vehicle_ids & semantic_vehicle_ids):
            raise RuntimeError(f"{episode_id}: protected constraint {constraint.get('constraint_id')} is not tied to a semantic vehicle")
    for slot in plan.get("traffic_slots") or []:
        slot_index = int(slot.get("slot_index") or 0)
        vehicle_ids = [str(vehicle_id) for vehicle_id in slot.get("vehicle_ids") or []]
        for vehicle_id in vehicle_ids:
            vehicle = vehicles_by_id.get(vehicle_id)
            if not vehicle:
                raise RuntimeError(f"{episode_id}: traffic slot {slot_index} references missing vehicle {vehicle_id}")
            if str(vehicle.get("traffic_role") or "") != "deterministic_background_vehicle":
                raise RuntimeError(f"{episode_id}: traffic slot {slot_index} includes non-background vehicle {vehicle_id}")
    for vehicle in vehicles:
        vehicle_id = str(vehicle.get("vehicle_id") or "")
        if not isinstance(vehicle.get("source_presence_required"), bool):
            raise RuntimeError(
                f"{episode_id}: {vehicle_id} must declare boolean source_presence_required"
            )
        if vehicle.get("source_presence_required") and not str(vehicle.get("source_entity_id") or ""):
            raise RuntimeError(
                f"{episode_id}: {vehicle_id} requires source presence without source_entity_id"
            )
        if (
            vehicle.get("source_presence_required")
            and str(vehicle.get("traffic_role") or "") != "semantic_vehicle"
            and str(vehicle.get("traffic_coverage_scope") or "")
            != "roi_background_route_crossing_flow"
        ):
            raise RuntimeError(
                f"{episode_id}: {vehicle_id} required source context must use an ROI-crossing route"
            )
        if vehicle.get("source_presence_required"):
            source_presence_contract = dict(vehicle.get("source_presence_contract") or {})
            if (
                str(source_presence_contract.get("policy") or "")
                != "full_capture_presence_from_route_travel_lower_bound_v1"
                or list(source_presence_contract.get("capture_tick_range") or [])
                != [0, DURATION_TICKS]
            ):
                raise RuntimeError(
                    f"{episode_id}: {vehicle_id} lacks a full-capture source presence model"
                )
            if (
                list(vehicle.get("from_edge_candidates") or [])
                != [str(vehicle.get("from_edge") or "")]
                or list(vehicle.get("to_edge_candidates") or [])
                != [str(vehicle.get("to_edge") or "")]
            ):
                raise RuntimeError(
                    f"{episode_id}: {vehicle_id} source presence route must use exact endpoints"
                )
            full_route_edges = [
                str(edge_id)
                for edge_id in source_presence_contract.get("full_route_edges") or []
            ]
            if (
                not full_route_edges
                or full_route_edges[0] != str(vehicle.get("from_edge") or "")
                or full_route_edges[-1] != str(vehicle.get("to_edge") or "")
            ):
                raise RuntimeError(
                    f"{episode_id}: {vehicle_id} source presence route endpoints do not match its model"
                )
            if set(full_route_edges) & set(str(edge_id) for edge_id in vehicle.get("forbidden_edges") or []):
                raise RuntimeError(
                    f"{episode_id}: {vehicle_id} source presence route intersects forbidden edges"
                )
            if not set(full_route_edges) & set(str(edge_id) for edge_id in vehicle.get("required_edges") or []):
                raise RuntimeError(
                    f"{episode_id}: {vehicle_id} source presence route misses required ROI edges"
                )
            release_tick = int(vehicle.get("release_tick") or 0)
            modeled_ticks = int(source_presence_contract.get("full_route_free_flow_ticks") or 0)
            minimum_ticks = int(source_presence_contract.get("minimum_required_route_ticks") or 0)
            planned_add_capture_tick = int(
                source_presence_contract.get("planned_add_capture_tick") or 0
            )
            if (
                int(source_presence_contract.get("planned_release_tick") or 0) != release_tick
                or minimum_ticks != DURATION_TICKS - release_tick + 1
                or modeled_ticks < minimum_ticks
                or int(source_presence_contract.get("formal_warmup_ticks") or 0)
                != FORMAL_WARMUP_TICKS
                or int(source_presence_contract.get("pre_capture_spawn_ticks") or 0)
                != FORMAL_WARMUP_TICKS
                or abs(float(vehicle.get("pre_capture_spawn_s") or 0.0) - FORMAL_WARMUP_TICKS / float(TICK_HZ))
                > 1e-9
                or planned_add_capture_tick
                != max(-FORMAL_WARMUP_TICKS, release_tick - FORMAL_WARMUP_TICKS)
                or int(source_presence_contract.get("insertion_reserve_before_capture_ticks") or 0)
                != -planned_add_capture_tick
                or planned_add_capture_tick >= 0
            ):
                raise RuntimeError(
                    f"{episode_id}: {vehicle_id} modeled activation and route lifetime do not cover "
                    f"capture ticks 0..{DURATION_TICKS}"
                )
        if vehicle.get("route_edges"):
            raise RuntimeError(f"{episode_id}: {vehicle_id} must not write route_edges in v2 explicit generation contract")
        from_edge = str(vehicle.get("from_edge") or "")
        to_edge = str(vehicle.get("to_edge") or "")
        if not from_edge or from_edge not in planner.edges:
            raise RuntimeError(f"{episode_id}: {vehicle_id} references unknown from_edge {from_edge!r}")
        if not to_edge or to_edge not in planner.edges:
            raise RuntimeError(f"{episode_id}: {vehicle_id} references unknown to_edge {to_edge!r}")
        declared_coverage_scope = str(vehicle.get("traffic_coverage_scope") or "")
        if from_edge == to_edge and declared_coverage_scope != "roi_background_local_component_flow":
            raise RuntimeError(f"{episode_id}: {vehicle_id} from_edge and to_edge must be distinct")
        from_edge_candidates = [str(edge_id) for edge_id in vehicle.get("from_edge_candidates") or []]
        if from_edge not in from_edge_candidates:
            raise RuntimeError(f"{episode_id}: {vehicle_id} from_edge must be included in from_edge_candidates")
        from_edge_depart_pos_by_edge = dict(vehicle.get("from_edge_depart_pos_by_edge") or {})
        missing_depart_candidates = [
            edge_id
            for edge_id in from_edge_candidates
            if edge_id not in from_edge_depart_pos_by_edge
        ]
        if missing_depart_candidates:
            raise RuntimeError(
                f"{episode_id}: {vehicle_id} from_edge_candidates missing candidate-specific depart positions: "
                f"{missing_depart_candidates[:8]}"
            )
        from_xy = dict(vehicle.get("from_projection") or vehicle.get("depart_projection") or {}).get("xy_enu_m")
        coverage_scope = str(vehicle.get("traffic_coverage_scope") or "")
        if coverage_scope in {"roi_background_visible_flow", "roi_background_route_crossing_flow"}:
            roi_bbox = spatial_scope.get("bbox_enu_m") or []
            if not from_xy or (
                roi_bbox
                and (
                    _point_in_bbox(from_xy, roi_bbox)
                    or _distance_to_bbox_xy(from_xy, roi_bbox) < ROI_DEPART_MARGIN_M
                )
            ):
                raise RuntimeError(f"{episode_id}: {vehicle_id} ROI-background from point must be outside ROI bbox")
            for candidate_edge_id in from_edge_candidates:
                candidate_edge = planner.edges.get(candidate_edge_id)
                if candidate_edge is None:
                    continue
                try:
                    candidate_xy = _point_at_edge_s(candidate_edge, float(from_edge_depart_pos_by_edge[candidate_edge_id]))
                except (TypeError, ValueError):
                    raise RuntimeError(f"{episode_id}: {vehicle_id} invalid candidate depart position for {candidate_edge_id}")
                if roi_bbox and (
                    _point_in_bbox(candidate_xy, roi_bbox)
                    or _distance_to_bbox_xy(candidate_xy, roi_bbox) < ROI_DEPART_MARGIN_M
                ):
                    raise RuntimeError(f"{episode_id}: {vehicle_id} candidate {candidate_edge_id} depart point is inside or too close to ROI bbox")
            roi_required_edges = set(
                str(edge_id)
                for edge_id in (
                    spatial_scope.get("roi_route_required_sumo_edges")
                    or spatial_scope.get("roi_visible_sumo_edges")
                    or []
                )
            )
            required_edges = set(str(edge_id) for edge_id in vehicle.get("required_edges") or [])
            if roi_required_edges and not (required_edges & roi_required_edges):
                raise RuntimeError(f"{episode_id}: {vehicle_id} ROI-background vehicle is missing ROI required_edges")
            if coverage_scope == "roi_background_route_crossing_flow":
                approach_contract = dict(vehicle.get("roi_approach_contract") or {})
                route_to_roi = [
                    str(edge_id)
                    for edge_id in approach_contract.get("route_to_roi_edges") or []
                ]
                if not route_to_roi or route_to_roi[0] != from_edge:
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} lacks a directed route-to-ROI approach contract"
                    )
                if required_edges and not (set(route_to_roi) & required_edges):
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} route-to-ROI contract misses required ROI edges"
                    )
                lead_ticks = int(approach_contract.get("lead_ticks") or 0)
                route_derived_release_tick = max(
                    -lead_ticks,
                    int(vehicle.get("expected_core_entry_tick") or 0) - lead_ticks,
                )
                release_adjustment = dict(vehicle.get("release_adjustment") or {})
                if release_adjustment:
                    if (
                        str(release_adjustment.get("policy") or "")
                        != "roi_vehicle_release_after_required_edge_protection_v1"
                        or int(release_adjustment.get("original_release_tick") or 0)
                        != route_derived_release_tick
                    ):
                        raise RuntimeError(
                            f"{episode_id}: {vehicle_id} protection release adjustment lacks route-derived evidence"
                        )
                    protected_end_tick = int(release_adjustment.get("protected_end_tick") or 0)
                    expected_release_tick = max(
                        route_derived_release_tick,
                        protected_end_tick + 1 - ROI_POST_PROTECTED_RELEASE_LEAD_TICKS,
                    )
                else:
                    expected_release_tick = route_derived_release_tick
                if lead_ticks <= 0 or int(vehicle.get("release_tick") or 0) != expected_release_tick:
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} release tick is not derived from its route lead"
                    )
        elif coverage_scope == "roi_background_local_component_flow":
            if str(dict(vehicle.get("sumo_control") or {}).get("spawn_scope") or "") != "roi_local_component_source":
                raise RuntimeError(f"{episode_id}: {vehicle_id} local ROI component lacks its source declaration")
            if not from_xy or not _point_in_bbox(from_xy, spatial_scope.get("bbox_enu_m") or []):
                raise RuntimeError(f"{episode_id}: {vehicle_id} local ROI component source must be inside the ROI")
            if to_edge != from_edge and to_edge not in planner.adjacency.get(from_edge, []):
                raise RuntimeError(f"{episode_id}: {vehicle_id} local ROI component source is not a directed edge pair")
        elif not from_xy or ((spatial_scope.get("bbox_enu_m") or []) and _point_in_bbox(from_xy, spatial_scope.get("bbox_enu_m") or [])):
            raise RuntimeError(f"{episode_id}: {vehicle_id} from point must be outside ROI bbox")
        to_edge_candidates = [str(edge_id) for edge_id in vehicle.get("to_edge_candidates") or []]
        if to_edge not in to_edge_candidates:
            raise RuntimeError(f"{episode_id}: {vehicle_id} to_edge must be included in to_edge_candidates")
        depart_tick = int(vehicle.get("depart_tick") or 0)
        for tick in vehicle.get("must_be_visible_ticks") or []:
            if int(tick) < depart_tick or int(tick) > DURATION_TICKS:
                raise RuntimeError(f"{episode_id}: {vehicle_id} invalid must-visible tick {tick}")
        if str(vehicle.get("traffic_role") or "") == "semantic_vehicle":
            corridor = dict(vehicle.get("semantic_corridor") or {})
            corridor_edge = str(corridor.get("edge_id") or "")
            if not corridor_edge or corridor_edge not in planner.edges:
                raise RuntimeError(f"{episode_id}: {vehicle_id} semantic vehicle missing fixed semantic corridor edge")
            binding_ticks = [
                int(tick)
                for binding in plan.get("semantic_bindings") or []
                if str(binding.get("vehicle_id") or "") == vehicle_id
                for tick in binding.get("evidence_ticks") or []
            ]
            lifecycle = dict(vehicle.get("semantic_lifecycle") or {})
            strict_absent_before_tick = _optional_int(lifecycle.get("strict_absent_before_tick"))
            control_ticks = [
                int(control.get("tick") or 0)
                for control in vehicle.get("semantic_controls") or []
            ]
            if not control_ticks:
                raise RuntimeError(f"{episode_id}: {vehicle_id} semantic vehicle missing lane-fixed semantic controls")
            must_visible_ticks = [int(tick) for tick in vehicle.get("must_be_visible_ticks") or []]
            if must_visible_ticks and max(control_ticks) < max(must_visible_ticks):
                raise RuntimeError(
                    f"{episode_id}: {vehicle_id} last semantic control tick {max(control_ticks)} precedes "
                    f"must-visible tick {max(must_visible_ticks)}"
                )
            if must_visible_ticks and int(vehicle.get("active_by_tick") or 0) > min(must_visible_ticks):
                raise RuntimeError(
                    f"{episode_id}: {vehicle_id} active_by_tick is later than its first must-visible tick"
                )
            if strict_absent_before_tick is not None:
                if int(vehicle.get("active_by_tick") or -1) != int(strict_absent_before_tick):
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} active_by_tick must equal orchestration evidence tick "
                        f"{strict_absent_before_tick}"
                    )
                if int(vehicle.get("release_tick") or -1) != int(strict_absent_before_tick):
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} release_tick must equal orchestration evidence tick "
                        f"{strict_absent_before_tick}"
                    )
                if min(control_ticks) <= int(strict_absent_before_tick):
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} semantic controls must start after orchestration evidence tick "
                        f"{strict_absent_before_tick}"
                    )
                lifecycle_control_start = _optional_int(lifecycle.get("control_start_tick"))
                if lifecycle_control_start is not None and min(control_ticks) != int(lifecycle_control_start):
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} semantic controls start at {min(control_ticks)}, "
                        f"not lifecycle control_start_tick {lifecycle_control_start}"
                    )
                lifecycle_control_end = _optional_int(lifecycle.get("control_end_tick"))
                if lifecycle_control_end is not None and max(control_ticks) != int(lifecycle_control_end):
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} semantic controls end at {max(control_ticks)}, "
                        f"not lifecycle control_end_tick {lifecycle_control_end}"
                    )
                if any(int(tick) < int(strict_absent_before_tick) for tick in vehicle.get("must_be_visible_ticks") or []):
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} must-visible contract precedes orchestration evidence tick "
                        f"{strict_absent_before_tick}"
                    )
                authority_event_ids = {
                    str(event_id) for event_id in lifecycle.get("authority_event_ids") or [] if str(event_id)
                }
                movement_action_ids = {
                    str(action_id) for action_id in lifecycle.get("movement_action_ids") or [] if str(action_id)
                }
                if len(authority_event_ids) != 1 or not movement_action_ids:
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} dispatch lifecycle lacks one exact event/action authority"
                    )
                authoritative_event_id = next(iter(authority_event_ids))
                lifecycle_source_ticks = {
                    int(tick) for tick in lifecycle.get("source_sample_ticks") or []
                }
                projections = [
                    dict(control.get("semantic_lane_projection") or {})
                    for control in vehicle.get("semantic_controls") or []
                ]
                observed_control_count = 0
                for projection in projections:
                    projection_event_id = str(projection.get("authority_event_id") or "")
                    projection_action_ids = {
                        str(action_id)
                        for action_id in projection.get("movement_action_ids") or []
                        if str(action_id)
                    }
                    if (
                        projection_event_id != authoritative_event_id
                        or projection_action_ids != movement_action_ids
                    ):
                        raise RuntimeError(
                            f"{episode_id}: {vehicle_id} semantic control is not bound to an exact successful "
                            "movement action realization"
                        )
                    synthetic_control = any(
                        bool(projection.get(flag_name))
                        for flag_name in (
                            "synthetic_entry_sample",
                            "synthetic_interpolated_sample",
                            "synthetic_exit_sample",
                            "synthetic_post_event_hold_sample",
                        )
                    )
                    source_tick = _optional_int(projection.get("source_tick"))
                    derivation_source_ticks = {
                        int(tick) for tick in projection.get("derivation_source_ticks") or []
                    }
                    derivation_policy = str(projection.get("derivation_policy") or "")
                    observed_physical_sample = bool(projection.get("observed_physical_sample"))
                    if synthetic_control:
                        if source_tick is not None or observed_physical_sample:
                            raise RuntimeError(
                                f"{episode_id}: {vehicle_id} synthetic semantic control must not masquerade "
                                "as an observed physical sample"
                            )
                        if not derivation_policy or not derivation_source_ticks:
                            raise RuntimeError(
                                f"{episode_id}: {vehicle_id} synthetic semantic control lacks explicit derivation"
                            )
                    else:
                        if source_tick is None or not observed_physical_sample:
                            raise RuntimeError(
                                f"{episode_id}: {vehicle_id} non-synthetic semantic control lacks observed source tick"
                            )
                        observed_control_count += 1
                        if derivation_source_ticks != {int(source_tick)}:
                            raise RuntimeError(
                                f"{episode_id}: {vehicle_id} observed semantic control has inconsistent source derivation"
                            )
                    if lifecycle_source_ticks and not derivation_source_ticks.issubset(lifecycle_source_ticks):
                        raise RuntimeError(
                            f"{episode_id}: {vehicle_id} semantic control derivation references non-authoritative ticks"
                        )
                if observed_control_count <= 0:
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} dispatch lifecycle has no observed physical semantic controls"
                    )
            elif binding_ticks and int(vehicle.get("active_by_tick") or 0) > min(binding_ticks) - PRE_EVENT_VISIBLE_MARGIN_TICKS:
                raise RuntimeError(f"{episode_id}: {vehicle_id} active_by_tick is too late for semantic before window")
            previous_control_s: float | None = None
            for control in vehicle.get("semantic_controls") or []:
                args = dict(control.get("args") or {})
                if str(args.get("edgeID") or "") != corridor_edge:
                    raise RuntimeError(f"{episode_id}: {vehicle_id} semantic control does not use fixed corridor edge {corridor_edge}")
                projection = dict(control.get("semantic_lane_projection") or {})
                configured_projection_tolerance = float(
                    corridor.get("projection_error_tolerance_m")
                    or MAX_SEMANTIC_LANE_PROJECTION_ERROR_M
                )
                if configured_projection_tolerance > MAX_SEMANTIC_LANE_PROJECTION_ERROR_M:
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} semantic corridor projection tolerance "
                        f"{configured_projection_tolerance:.3f}m exceeds physical threshold "
                        f"{MAX_SEMANTIC_LANE_PROJECTION_ERROR_M:.3f}m"
                    )
                projection_tolerance = MAX_SEMANTIC_LANE_PROJECTION_ERROR_M
                if float(projection.get("projection_error_m") or 0.0) > projection_tolerance:
                    raise RuntimeError(f"{episode_id}: {vehicle_id} semantic control projection error exceeds threshold")
                control_s = float(projection.get("lane_position_m") or 0.0)
                if previous_control_s is not None and control_s + 0.05 < previous_control_s:
                    raise RuntimeError(f"{episode_id}: {vehicle_id} semantic controls reverse on fixed corridor edge {corridor_edge}")
                previous_control_s = control_s
        else:
            forbidden = set(str(edge_id) for edge_id in vehicle.get("forbidden_edges") or [])
            episode_wide_protected_edges = {
                str(edge_id)
                for constraint in plan.get("semantic_traffic_constraints") or []
                if str(constraint.get("background_policy") or "") == "route_must_avoid_protected_edges_episode_wide"
                for edge_id in constraint.get("protected_edges") or []
                if str(edge_id)
            }
            missing = sorted(episode_wide_protected_edges - forbidden)
            if missing:
                raise RuntimeError(f"{episode_id}: {vehicle_id} background route is missing episode-wide protected-edge forbids")
            entry_tick = int(vehicle.get("expected_core_entry_tick") or vehicle.get("core_entry_tick") or 0)
            exit_tick = int(vehicle.get("expected_core_exit_tick") or vehicle.get("core_exit_tick") or entry_tick)
            release_tick = int(vehicle.get("release_tick") or 0)
            traffic_tuning = dict(plan.get("traffic_tuning") or {})
            check_entry_tick = entry_tick - _protected_window_post_route_guard_ticks(traffic_tuning)
            check_exit_tick = exit_tick + _protected_window_pre_route_guard_ticks(traffic_tuning)
            for constraint in plan.get("semantic_traffic_constraints") or []:
                tick_range = list(constraint.get("tick_range") or [])
                protected_edges = set(str(edge_id) for edge_id in constraint.get("protected_edges") or [])
                protected_traversal_ticks = int(
                    constraint.get("background_protected_edge_traversal_ticks") or 0
                )
                physical_active_end_tick = max(
                    check_exit_tick,
                    check_entry_tick + protected_traversal_ticks,
                )
                if (
                    len(tick_range) >= 2
                    and release_tick <= int(tick_range[1])
                    and physical_active_end_tick >= int(tick_range[0])
                ):
                    missing = sorted(protected_edges - forbidden)
                    if missing:
                        raise RuntimeError(
                            f"{episode_id}: {vehicle_id} physical release-to-clearance envelope overlaps "
                            f"{constraint.get('constraint_id')} without forbidden protected edges"
                        )
    vehicle_id_set = set(ids)
    road_semantics = dict(plan.get("road_semantics") or {})
    road_closed_edges = {str(edge_id) for edge_id in road_semantics.get("closed_edges") or [] if str(edge_id)}
    for vehicle in vehicles:
        vehicle_id = str(vehicle.get("vehicle_id") or "")
        forbidden = {str(edge_id) for edge_id in vehicle.get("forbidden_edges") or [] if str(edge_id)}
        if str(vehicle.get("traffic_role") or "") == "semantic_vehicle":
            continue
        missing = sorted(road_closed_edges - forbidden)
        if missing:
            raise RuntimeError(f"{episode_id}: {vehicle_id} background route is missing road-closure forbids {missing[:8]}")
    for binding in plan.get("semantic_bindings") or []:
        vehicle_id = str(binding.get("vehicle_id") or "")
        if vehicle_id not in vehicle_id_set:
            raise RuntimeError(f"{episode_id}: semantic binding references missing SUMO vehicle {vehicle_id}")


def build_vehicle_plan_review(plan: dict[str, Any]) -> dict[str, Any]:
    vehicles = list(plan.get("vehicles") or [])
    role_counts = Counter(str(vehicle.get("role") or "") for vehicle in vehicles)
    direction_counts = Counter(str(vehicle.get("direction_role") or "") for vehicle in vehicles)
    return {
        "episode_id": plan.get("episode_id"),
        "scenario_id": plan.get("scenario_id"),
        "duration_ticks": plan.get("duration_ticks"),
        "tick_hz": plan.get("tick_hz"),
        "vehicle_source_policy": plan.get("vehicle_source_policy"),
        "seed_profile": plan.get("seed_profile"),
        "traffic_profile": plan.get("traffic_profile"),
        "traffic_case": plan.get("traffic_case"),
        "traffic_tuning": plan.get("traffic_tuning"),
        "road_semantics": plan.get("road_semantics") or {},
        "scenario_vehicle_class": plan.get("scenario_vehicle_class"),
        "traffic_flow_mode": plan.get("traffic_flow_mode"),
        "minimum_vehicle_count": plan.get("minimum_vehicle_count"),
        "minimum_background_vehicle_count": plan.get("minimum_background_vehicle_count"),
        "vehicle_count": len(vehicles),
        "role_counts": dict(sorted(role_counts.items())),
        "direction_counts": dict(sorted(direction_counts.items())),
        "traffic_slots": plan.get("traffic_slots") or [],
        "semantic_traffic_reviews": plan.get("semantic_traffic_reviews") or [],
        "semantic_traffic_constraints": plan.get("semantic_traffic_constraints") or [],
        "flow_groups": plan.get("flow_groups") or [],
        "spatial_scope": plan.get("spatial_scope"),
        "semantic_bindings": plan.get("semantic_bindings"),
        "vehicles": [
            {
                "vehicle_id": vehicle.get("vehicle_id"),
                "role": vehicle.get("role"),
                "type_id": vehicle.get("type_id"),
                "traffic_slot_index": vehicle.get("traffic_slot_index"),
                "expected_core_entry_tick": vehicle.get("expected_core_entry_tick"),
                "expected_core_exit_tick": vehicle.get("expected_core_exit_tick"),
                "release_tick": vehicle.get("release_tick"),
                "spawn_phase": vehicle.get("spawn_phase"),
                "active_by_tick": vehicle.get("active_by_tick"),
                "from_edge": vehicle.get("from_edge"),
                "to_edge": vehicle.get("to_edge"),
                "direction_role": vehicle.get("direction_role"),
                "flow_group_id": vehicle.get("flow_group_id"),
                "flow_mode": vehicle.get("flow_mode"),
                "forbidden_edge_count": len(vehicle.get("forbidden_edges") or []),
                "semantic_control_count": len(vehicle.get("semantic_controls") or []),
                "must_be_visible_ticks": vehicle.get("must_be_visible_ticks"),
            }
            for vehicle in vehicles
        ],
    }


def build_vehicle_plan_review_markdown(review: dict[str, Any]) -> str:
    lines = [
        f"# Vehicle Plan Review: {review['episode_id']}",
        "",
        f"- Scenario: {review['scenario_id']}",
        f"- Vehicle source policy: {review['vehicle_source_policy']}",
        f"- Time scope: ticks 0..{review['duration_ticks']} at {review['tick_hz']} Hz",
        f"- Seed profile: {dict(review.get('seed_profile') or {}).get('seed_label')} / {dict(review.get('seed_profile') or {}).get('profile_id')}",
        f"- Traffic flow mode: {review.get('traffic_flow_mode')}",
        f"- Traffic case: {json.dumps(review.get('traffic_case') or {}, ensure_ascii=False, sort_keys=True)}",
        f"- Traffic tuning: {json.dumps(review.get('traffic_tuning') or {}, ensure_ascii=False, sort_keys=True)}",
        f"- Vehicle count: {review['vehicle_count']} (minimum {review['minimum_vehicle_count']})",
        f"- Background schedule count: {review.get('minimum_background_vehicle_count')}",
        f"- Role counts: {json.dumps(review.get('role_counts') or {}, ensure_ascii=False, sort_keys=True)}",
        f"- Direction counts: {json.dumps(review.get('direction_counts') or {}, ensure_ascii=False, sort_keys=True)}",
        f"- Traffic slots: {len(review.get('traffic_slots') or [])}",
        f"- Semantic traffic reviews: {len(review.get('semantic_traffic_reviews') or [])}",
        f"- Semantic traffic constraints: {len(review.get('semantic_traffic_constraints') or [])}",
        f"- Flow groups: {len(review.get('flow_groups') or [])}",
        "",
        "## Semantic Traffic Reviews",
        "",
        "| event_id | vehicle_related | should_protect | reason |",
        "|---|---:|---:|---|",
    ]
    for item in review.get("semantic_traffic_reviews") or []:
        lines.append(
            f"| {item.get('event_id')} | {bool(item.get('vehicle_related'))} | "
            f"{bool(item.get('should_protect_traffic'))} | {item.get('reason')} |"
        )
    lines.extend(
        [
        "",
        "## Traffic Slots",
        "",
        "| slot | tick range | target bg | vehicles | core entry ticks |",
        "|---:|---|---:|---|---|",
        ]
    )
    for slot in review.get("traffic_slots") or []:
        lines.append(
            f"| {slot.get('slot_index')} | {slot.get('start_tick')}..{slot.get('end_tick')} | "
            f"{slot.get('target_background_vehicle_count')} | {len(slot.get('vehicle_ids') or [])} | "
            f"{','.join(str(tick) for tick in slot.get('core_entry_ticks') or [])} |"
        )
    lines.extend(
        [
        "",
        "## Vehicles",
        "",
        "| vehicle_id | role | type | slot | release_tick | expected_core_entry..exit | flow_group | from_edge | to_edge | direction | forbidden_edges | controls | must_visible_ticks |",
        "|---|---|---|---:|---:|---|---|---|---|---|---|",
        ]
    )
    for vehicle in review.get("vehicles") or []:
        lines.append(
            "| {vehicle_id} | {role} | {type_id} | {slot} | {release_tick} | {core_entry}..{core_exit} | {flow_group_id} | {from_edge} | {to_edge} | {direction_role} | {forbidden} | {controls} | {ticks} |".format(
                vehicle_id=vehicle.get("vehicle_id"),
                role=vehicle.get("role"),
                type_id=vehicle.get("type_id"),
                slot=vehicle.get("traffic_slot_index"),
                release_tick=vehicle.get("release_tick"),
                core_entry=vehicle.get("expected_core_entry_tick"),
                core_exit=vehicle.get("expected_core_exit_tick"),
                flow_group_id=vehicle.get("flow_group_id"),
                from_edge=vehicle.get("from_edge"),
                to_edge=vehicle.get("to_edge"),
                direction_role=vehicle.get("direction_role"),
                forbidden=vehicle.get("forbidden_edge_count"),
                controls=vehicle.get("semantic_control_count"),
                ticks=",".join(str(tick) for tick in vehicle.get("must_be_visible_ticks") or []),
            )
        )
    lines.extend(["", "## Semantic Traffic Constraints", ""])
    constraints = review.get("semantic_traffic_constraints") or []
    if not constraints:
        lines.append("- None")
    else:
        for constraint in constraints:
            lines.append(
                f"- {constraint.get('constraint_id')}: ticks {constraint.get('tick_range')} "
                f"edges={len(constraint.get('protected_edges') or [])} allowed={constraint.get('allowed_vehicle_ids')}"
            )
    lines.extend(["", "## Semantic Bindings", ""])
    bindings = review.get("semantic_bindings") or []
    if not bindings:
        lines.append("- None")
    else:
        for binding in bindings:
            lines.append(
                f"- {binding.get('event_id')}: {binding.get('vehicle_id')} "
                f"(source {binding.get('source_entity_id')}, ticks {binding.get('evidence_ticks')})"
            )
    return "\n".join(lines) + "\n"


def write_explicit_vehicle_plan(episode_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    review = build_vehicle_plan_review(plan)
    _write_json(explicit_plan_path(episode_dir), plan)
    _write_json(review_json_path(episode_dir), review)
    review_md_path(episode_dir).write_text(build_vehicle_plan_review_markdown(review), encoding="utf-8")
    return review


def instructions_from_explicit_plan(
    plan: dict[str, Any],
    *,
    warmup_s: float,
    tick_hz: float = TICK_HZ,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    episode_id = str(plan["episode_id"])
    seed_index = int(dict(plan.get("seed_profile") or {}).get("seed_index") or episode_seed_index(episode_id))
    for index, vehicle in enumerate(plan.get("vehicles") or []):
        vehicle_id = str(vehicle["vehicle_id"])
        source_entity_id = str(vehicle.get("source_entity_id") or "")
        route_id = f"{episode_id}.{vehicle_id}.explicit_route"
        route_sequence = -200000 + index * 5
        has_semantic_controls = bool(vehicle.get("semantic_controls"))
        first_semantic_control_tick = min(
            (int(control.get("tick") or 0) for control in vehicle.get("semantic_controls") or []),
            default=None,
        )
        last_semantic_control_tick = max(
            (int(control.get("tick") or 0) for control in vehicle.get("semantic_controls") or []),
            default=None,
        )
        semantic_corridor = dict(vehicle.get("semantic_corridor") or {})
        semantic_lifecycle = dict(vehicle.get("semantic_lifecycle") or {})
        strict_absent_before_tick = _optional_int(semantic_lifecycle.get("strict_absent_before_tick"))
        semantic_contract_end_ticks = [
            int(tick) for tick in vehicle.get("must_be_visible_ticks") or []
        ]
        lifecycle_control_end_tick = _optional_int(semantic_lifecycle.get("control_end_tick"))
        if lifecycle_control_end_tick is not None:
            semantic_contract_end_ticks.append(int(lifecycle_control_end_tick))
        if (
            has_semantic_controls
            and last_semantic_control_tick is not None
            and semantic_contract_end_ticks
            and last_semantic_control_tick < max(semantic_contract_end_ticks)
        ):
            raise RuntimeError(
                f"{episode_id}: {vehicle_id} last semantic control tick {last_semantic_control_tick} precedes "
                f"visibility contract end tick {max(semantic_contract_end_ticks)}"
            )
        semantic_corridor_edge = str(semantic_corridor.get("edge_id") or "")
        semantic_bootstrap_depart_pos_m = _optional_float(vehicle.get("semantic_bootstrap_depart_pos_m"))
        route_from_edge = str(vehicle.get("from_edge") or "")
        route_from_candidates = [str(edge_id) for edge_id in vehicle.get("from_edge_candidates") or []]
        route_from_depart_pos_by_edge = {
            str(edge_id): float(depart_pos)
            for edge_id, depart_pos in dict(vehicle.get("from_edge_depart_pos_by_edge") or {}).items()
            if str(edge_id)
        }
        if has_semantic_controls and semantic_corridor_edge and semantic_bootstrap_depart_pos_m is not None:
            route_from_depart_pos_by_edge[semantic_corridor_edge] = float(semantic_bootstrap_depart_pos_m)
        if route_from_edge and route_from_edge not in route_from_depart_pos_by_edge:
            route_from_depart_pos_by_edge[route_from_edge] = float(vehicle.get("from_pos") or 0.0)
        semantic_bootstrap_edges = (
            [semantic_corridor_edge]
            if has_semantic_controls and semantic_corridor_edge
            else []
        )
        records.append(
            {
                "sequence": route_sequence,
                "time_s": 0.0,
                "tick": 0,
                "episode_id": episode_id,
                "seed_index": seed_index,
                "source_entity_id": source_entity_id,
                "internal_vehicle_id": vehicle_id,
                "instruction_type": "route.findRoute",
                "traci_call": "traci.simulation.findRoute",
                "time_axis": "warmup",
                "episode_tick": None,
                "episode_time_s": None,
                "active_required": False,
                "event_id": "",
                "intent": "explicit_vehicle_plan_sumo_resolved_route",
                "target_role": str(vehicle.get("role") or ""),
                "args": {
                    "vehicle_id": vehicle_id,
                    "route_id": route_id,
                    "from_edge": route_from_edge,
                    "from_edge_candidates": route_from_candidates,
                    "from_edge_depart_pos_by_edge": route_from_depart_pos_by_edge,
                    "to_edge": str(vehicle.get("to_edge") or ""),
                    "to_edge_candidates": [str(edge_id) for edge_id in vehicle.get("to_edge_candidates") or []],
                    "typeID": str(vehicle.get("type_id") or "aero_passenger"),
                    "forbidden_edges": [str(edge_id) for edge_id in vehicle.get("forbidden_edges") or []],
                    "semantic_bootstrap_edges": semantic_bootstrap_edges,
                    "semantic_bootstrap_depart_pos_m": semantic_bootstrap_depart_pos_m,
                    "required_edges": (
                        [semantic_corridor_edge]
                        if has_semantic_controls and semantic_corridor_edge
                        else [str(edge_id) for edge_id in vehicle.get("required_edges") or []]
                    ),
                },
            }
        )
        release_tick = int(vehicle.get("release_tick") or 0)
        pre_capture_spawn_s = float(vehicle.get("pre_capture_spawn_s") or 0.0)
        semantic_first_control_time_s: float | None = None
        if has_semantic_controls and first_semantic_control_tick is not None:
            if strict_absent_before_tick is not None:
                if first_semantic_control_tick <= int(strict_absent_before_tick):
                    raise RuntimeError(
                        f"{episode_id}: {vehicle_id} first semantic control tick {first_semantic_control_tick} "
                        f"must follow orchestration evidence tick {strict_absent_before_tick}"
                    )
                add_tick = int(strict_absent_before_tick)
            else:
                requested_add_tick = first_semantic_control_tick - SEMANTIC_PRE_ACTIVATION_TICKS
                add_tick = min(release_tick, requested_add_tick)
            semantic_pre_activation_ticks = first_semantic_control_tick - add_tick
            add_time_s = float(warmup_s) + add_tick / float(tick_hz)
            semantic_first_control_time_s = float(warmup_s) + first_semantic_control_tick / float(tick_hz)
        else:
            release_time_nominal_s = float(warmup_s) + release_tick / float(tick_hz)
            spawn_scope = str(dict(vehicle.get("sumo_control") or {}).get("spawn_scope") or "")
            if spawn_scope in {"outside_roi", "outside_expanded_roi"}:
                pre_spawn_s = max(0.0, float(pre_capture_spawn_s))
                add_time_s = max(0.0, release_time_nominal_s - pre_spawn_s)
            elif spawn_scope == "outside_roi_boundary_inside_extended_approach":
                pre_spawn_s = max(0.0, float(pre_capture_spawn_s))
                add_time_s = max(0.0, release_time_nominal_s - pre_spawn_s)
            else:
                add_time_s = max(0.0, release_time_nominal_s)
        nominal_release_abs_time_s = float(warmup_s) + release_tick / float(tick_hz)
        if has_semantic_controls and semantic_first_control_time_s is not None:
            release_abs_time_s = max(add_time_s + 0.1, semantic_first_control_time_s - 0.05)
        else:
            release_abs_time_s = max(add_time_s + 0.1, nominal_release_abs_time_s)
        add_axis = "capture" if add_time_s >= float(warmup_s) else "warmup"
        add_episode_tick = int(round((add_time_s - float(warmup_s)) * float(tick_hz))) if add_axis == "capture" else None
        release_axis = "capture" if release_abs_time_s >= float(warmup_s) else "warmup"
        release_episode_tick = (
            int(round((release_abs_time_s - float(warmup_s)) * float(tick_hz)))
            if release_axis == "capture"
            else None
        )
        metadata = {
            "explicit_vehicle_plan": True,
            "vehicle_source_policy": VEHICLE_SOURCE_POLICY,
            "vehicle_id": vehicle_id,
            "source_entity_id": source_entity_id,
            "role": vehicle.get("role"),
            "traffic_role": vehicle.get("traffic_role"),
            "direction_role": vehicle.get("direction_role"),
            "traffic_slot_index": vehicle.get("traffic_slot_index"),
            "expected_core_entry_tick": vehicle.get("expected_core_entry_tick"),
            "expected_core_exit_tick": vehicle.get("expected_core_exit_tick"),
            "release_tick": release_tick,
            "spawn_phase": vehicle.get("spawn_phase"),
            "traffic_coverage_scope": vehicle.get("traffic_coverage_scope"),
            "required_edges": [str(edge_id) for edge_id in vehicle.get("required_edges") or []],
            "semantic_control_window_ticks": (
                [first_semantic_control_tick, last_semantic_control_tick]
                if first_semantic_control_tick is not None and last_semantic_control_tick is not None
                else []
            ),
            "semantic_bootstrap_depart_pos_m": vehicle.get("semantic_bootstrap_depart_pos_m"),
            "semantic_lifecycle": semantic_lifecycle or None,
            "semantic_activation_policy": (
                "absent_before_orchestration_evidence_then_lane_control_v2"
                if strict_absent_before_tick is not None
                else "outside_roi_route_endpoint_before_lane_control_v1"
                if has_semantic_controls
                else ""
            ),
            "semantic_pre_activation_ticks": int(semantic_pre_activation_ticks) if has_semantic_controls and first_semantic_control_tick is not None else None,
            "semantic_add_tick": int(add_tick) if has_semantic_controls and first_semantic_control_tick is not None else None,
            "semantic_activation_depart_edge": route_from_edge if has_semantic_controls else "",
            "semantic_activation_depart_pos_m": float(vehicle.get("from_pos") or 0.0) if has_semantic_controls else None,
            "route_source": "sumo_findRoute",
            "protected_window_role": "allowed_semantic_vehicle" if str(vehicle.get("traffic_role") or "") == "semantic_vehicle" else "background_forbidden_from_protected_edges",
            "seed_profile": plan.get("seed_profile"),
            "logical_asset_id": _logical_asset_for_type(str(vehicle.get("type_id") or "")),
        }
        records.append(
            {
                "sequence": route_sequence + 1,
                "time_s": round(add_time_s, 6),
                "tick": int(round(add_time_s * float(tick_hz))),
                "episode_id": episode_id,
                "seed_index": seed_index,
                "source_entity_id": source_entity_id,
                "internal_vehicle_id": vehicle_id,
                "instruction_type": "vehicle.addFull",
                "traci_call": "traci.vehicle.addFull",
                "time_axis": add_axis,
                "episode_tick": add_episode_tick,
                "episode_time_s": round(add_episode_tick / float(tick_hz), 6) if add_episode_tick is not None else None,
                "active_required": False,
                "event_id": "",
                "intent": (
                    "explicit_semantic_vehicle_orchestration_activation"
                    if strict_absent_before_tick is not None
                    else "explicit_vehicle_plan_warmup_spawn"
                ),
                "target_role": str(vehicle.get("role") or ""),
                "args": {
                    "vehID": vehicle_id,
                    "routeID": route_id,
                    "typeID": str(vehicle.get("type_id") or "aero_passenger"),
                    "depart": "now",
                    "departLane": (
                        str(int(semantic_corridor.get("lane_index") or vehicle.get("from_lane_index") or 0))
                        if has_semantic_controls
                        else "free"
                    ),
                    "departPos": (
                        f"{float(semantic_bootstrap_depart_pos_m if semantic_bootstrap_depart_pos_m is not None else vehicle.get('from_pos') or 0.0):.3f}"
                        if has_semantic_controls
                        else f"{float(vehicle.get('from_pos') or 0.0):.3f}"
                    ),
                    "departSpeed": str(dict(vehicle.get("speed_profile") or {}).get("depart_speed") or "0"),
                    "metadata": metadata,
                },
            }
        )
        hold_until_time_s = release_abs_time_s
        if hold_until_time_s > add_time_s + 0.15:
            initial_hold_time_s = add_time_s + 0.05
            initial_hold_axis = "capture" if initial_hold_time_s >= float(warmup_s) else "warmup"
            initial_hold_episode_tick = (
                int(round((initial_hold_time_s - float(warmup_s)) * float(tick_hz)))
                if initial_hold_axis == "capture"
                else None
            )
            records.append(
                {
                    "sequence": route_sequence + 2,
                    "time_s": round(initial_hold_time_s, 6),
                    "tick": int(round(initial_hold_time_s * float(tick_hz))),
                    "episode_id": episode_id,
                    "seed_index": seed_index,
                    "source_entity_id": source_entity_id,
                    "internal_vehicle_id": vehicle_id,
                    "instruction_type": "vehicle.setSpeed",
                    "traci_call": "traci.vehicle.setSpeed",
                    "time_axis": initial_hold_axis,
                    "episode_tick": initial_hold_episode_tick,
                    "episode_time_s": (
                        round(initial_hold_episode_tick / float(tick_hz), 6)
                        if initial_hold_episode_tick is not None
                        else None
                    ),
                    "active_required": False,
                    "event_id": "",
                    "intent": "explicit_vehicle_plan_off_roi_speed_hold",
                    "target_role": str(vehicle.get("role") or ""),
                    "args": {
                        "vehID": vehicle_id,
                        "speed": 0.0,
                    },
                }
            )
            records.append(
                {
                    "sequence": route_sequence + 3,
                    "time_s": round(release_abs_time_s + 0.05, 6),
                    "tick": int(round((release_abs_time_s + 0.05) * float(tick_hz))),
                    "episode_id": episode_id,
                    "seed_index": seed_index,
                    "source_entity_id": source_entity_id,
                    "internal_vehicle_id": vehicle_id,
                    "instruction_type": "vehicle.setSpeed",
                    "traci_call": "traci.vehicle.setSpeed",
                    "time_axis": release_axis,
                    "episode_tick": release_episode_tick,
                    "episode_time_s": (
                        round(release_episode_tick / float(tick_hz), 6)
                        if release_episode_tick is not None
                        else None
                    ),
                    "active_required": False,
                    "event_id": "",
                    "intent": "explicit_vehicle_plan_release_from_off_roi_speed_hold",
                    "target_role": str(vehicle.get("role") or ""),
                    "args": {
                        "vehID": vehicle_id,
                        "speed": -1.0,
                    },
                }
            )
            if has_semantic_controls and first_semantic_control_tick is not None:
                hold_time_s = add_time_s + 0.5
                hold_index = 0
                while hold_time_s < release_abs_time_s - 1e-9:
                    hold_axis = "capture" if hold_time_s >= float(warmup_s) else "warmup"
                    hold_episode_tick = (
                        int(round((hold_time_s - float(warmup_s)) * float(tick_hz)))
                        if hold_axis == "capture"
                        else None
                    )
                    records.append(
                        {
                            "sequence": route_sequence + 10 + hold_index,
                            "time_s": round(hold_time_s, 6),
                            "tick": int(round(hold_time_s * float(tick_hz))),
                            "episode_id": episode_id,
                            "seed_index": seed_index,
                            "source_entity_id": source_entity_id,
                            "internal_vehicle_id": vehicle_id,
                            "instruction_type": "vehicle.setSpeed",
                            "traci_call": "traci.vehicle.setSpeed",
                            "time_axis": hold_axis,
                            "episode_tick": hold_episode_tick,
                            "episode_time_s": (
                                round(hold_episode_tick / float(tick_hz), 6)
                                if hold_episode_tick is not None
                                else None
                            ),
                            "active_required": False,
                            "event_id": "",
                            "intent": "explicit_semantic_vehicle_pre_event_hold_until_lane_control",
                            "target_role": str(vehicle.get("role") or ""),
                            "args": {
                                "vehID": vehicle_id,
                                "speed": 0.0,
                            },
                        }
                    )
                    hold_index += 1
                    hold_time_s += 0.5
        max_speed_mps = float(dict(vehicle.get("speed_profile") or {}).get("max_speed_mps") or 0.0)
        if max_speed_mps > 0.0 and not has_semantic_controls:
            records.append(
                {
                    "sequence": route_sequence + 4,
                    "time_s": round(release_abs_time_s + 0.1, 6),
                    "tick": int(round((release_abs_time_s + 0.1) * float(tick_hz))),
                    "episode_id": episode_id,
                    "seed_index": seed_index,
                    "source_entity_id": source_entity_id,
                    "internal_vehicle_id": vehicle_id,
                    "instruction_type": "vehicle.setMaxSpeed",
                    "traci_call": "traci.vehicle.setMaxSpeed",
                    "time_axis": release_axis,
                    "episode_tick": min(DURATION_TICKS, release_tick + 1) if release_tick >= 0 else None,
                    "episode_time_s": round((release_tick + 1) / float(tick_hz), 6) if release_tick >= 0 else None,
                    "active_required": False,
                    "event_id": "",
                    "intent": "explicit_vehicle_plan_speed_profile",
                    "target_role": str(vehicle.get("role") or ""),
                    "args": {
                        "vehID": vehicle_id,
                        "speed": max_speed_mps,
                    },
                }
            )
        smoothing_stop = dict(vehicle.get("roi_smoothing_stop") or {})
        smoothing_stop_candidates = [
            dict(candidate)
            for candidate in smoothing_stop.get("candidates") or []
            if str(dict(candidate).get("edgeID") or "")
        ]
        if smoothing_stop_candidates and not has_semantic_controls:
            first_candidate = smoothing_stop_candidates[0]
            stop_duration_s = max(
                0.1,
                float(smoothing_stop.get("duration_ticks") or 0.0) / float(tick_hz),
            )
            target_tick = int(smoothing_stop.get("target_tick") or vehicle.get("expected_core_entry_tick") or release_tick)
            target_time_s = float(warmup_s) + max(0, min(DURATION_TICKS, target_tick)) / float(tick_hz)
            stop_until_s = target_time_s + stop_duration_s
            first_stop_time_s = max(add_time_s + 0.2, release_abs_time_s + 0.2)
            for stop_attempt in range(8):
                stop_time_s = first_stop_time_s + float(stop_attempt)
                if stop_time_s > float(warmup_s) + DURATION_TICKS / float(tick_hz) + 1e-9:
                    break
                stop_axis = "capture" if stop_time_s >= float(warmup_s) else "warmup"
                stop_episode_tick = (
                    int(round((stop_time_s - float(warmup_s)) * float(tick_hz)))
                    if stop_axis == "capture"
                    else None
                )
                records.append(
                    {
                        "sequence": route_sequence + 20 + stop_attempt,
                        "time_s": round(stop_time_s, 6),
                        "tick": int(round(stop_time_s * float(tick_hz))),
                        "episode_id": episode_id,
                        "seed_index": seed_index,
                        "source_entity_id": source_entity_id,
                        "internal_vehicle_id": vehicle_id,
                        "instruction_type": "vehicle.setStop",
                        "traci_call": "traci.vehicle.setStop",
                        "time_axis": stop_axis,
                        "episode_tick": stop_episode_tick,
                        "episode_time_s": round(stop_episode_tick / float(tick_hz), 6) if stop_episode_tick is not None else None,
                        "active_required": False,
                        "event_id": "",
                        "intent": "explicit_vehicle_plan_roi_context_smoothing_stop",
                        "target_role": str(vehicle.get("role") or ""),
                        "args": {
                            "vehID": vehicle_id,
                            "edgeID": str(first_candidate.get("edgeID") or ""),
                            "pos": float(first_candidate.get("pos") or 0.0),
                            "laneIndex": int(first_candidate.get("laneIndex") or 0),
                            "duration": stop_duration_s,
                            "until": stop_until_s,
                            "stop_candidates": smoothing_stop_candidates,
                        },
                    }
                )
        for control_index, control in enumerate(vehicle.get("semantic_controls") or []):
            control_tick = int(control.get("tick") or 0)
            control_active_required = control_tick in set(int(tick) for tick in vehicle.get("must_be_visible_ticks") or [])
            records.append(
                {
                    "sequence": route_sequence + 50 + control_index,
                    "time_s": round(float(warmup_s) + control_tick / float(tick_hz), 6),
                    "tick": int(round(float(warmup_s) * float(tick_hz))) + control_tick,
                    "episode_id": episode_id,
                    "seed_index": seed_index,
                    "source_entity_id": source_entity_id,
                    "internal_vehicle_id": vehicle_id,
                    "instruction_type": str(control.get("instruction_type") or "vehicle.moveToXY"),
                    "traci_call": "traci.vehicle.moveToXY",
                    "time_axis": "capture",
                    "episode_tick": control_tick,
                    "episode_time_s": round(control_tick / float(tick_hz), 6),
                    "active_required": control_active_required,
                    "event_id": "",
                    "intent": "explicit_semantic_vehicle_control",
                    "target_role": str(vehicle.get("role") or ""),
                    "args": {"vehID": vehicle_id, **dict(control.get("args") or {})},
                }
            )
        if has_semantic_controls and last_semantic_control_tick is not None and last_semantic_control_tick < DURATION_TICKS:
            remove_tick = min(DURATION_TICKS, last_semantic_control_tick + 1)
            records.append(
                {
                    "sequence": route_sequence + 5000,
                    "time_s": round(float(warmup_s) + remove_tick / float(tick_hz), 6),
                    "tick": int(round(float(warmup_s) * float(tick_hz))) + remove_tick,
                    "episode_id": episode_id,
                    "seed_index": seed_index,
                    "source_entity_id": source_entity_id,
                    "internal_vehicle_id": vehicle_id,
                    "instruction_type": "vehicle.remove",
                    "traci_call": "traci.vehicle.remove",
                    "time_axis": "capture",
                    "episode_tick": remove_tick,
                    "episode_time_s": round(remove_tick / float(tick_hz), 6),
                    "active_required": False,
                    "event_id": "",
                    "intent": "explicit_semantic_vehicle_remove_after_boundary_exit",
                    "target_role": str(vehicle.get("role") or ""),
                    "args": {"vehID": vehicle_id},
                }
            )
    return sorted(records, key=lambda item: (float(item.get("time_s") or 0.0), int(item.get("sequence") or 0)))


def _logical_asset_for_type(type_id: str) -> str:
    if "ambulance" in type_id:
        return "vehicle.emergency.ambulance.v1"
    if "police" in type_id:
        return "vehicle.emergency.police_suv.v1"
    if "emergency" in type_id:
        return "vehicle.emergency.suv.v1"
    if "delivery" in type_id:
        return "vehicle.service.box.v1"
    return "vehicle.ground.boxcar.v1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic explicit SUMO vehicle plans for existing episodes.")
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_EPISODES_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--net-xml", type=Path, default=DEFAULT_SUMO_NET_XML)
    args = parser.parse_args(argv)
    planner = SumoGroundFlowPlanner(args.net_xml)
    root = Path(args.episodes_root)
    episode_ids = set(str(item) for item in args.episode if str(item))
    episode_dirs = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and (path / "episode_manifest.json").exists()
        and (not episode_ids or path.name in episode_ids)
    ]
    for episode_dir in episode_dirs:
        plan = build_explicit_vehicle_plan(episode_dir, net_xml=args.net_xml, planner=planner)
        review = write_explicit_vehicle_plan(episode_dir, plan)
        print(
            json.dumps(
                {
                    "episode_id": review["episode_id"],
                    "vehicle_count": review["vehicle_count"],
                    "minimum_vehicle_count": review["minimum_vehicle_count"],
                    "seed_profile": dict(review.get("seed_profile") or {}).get("profile_id"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
