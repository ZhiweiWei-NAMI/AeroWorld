"""Validate scene_setup/event_script spatial grounding.

This complements validate_coverage.py.  Coverage only proves that scenario
names exist; this tool checks that generated scenarios are executable against
the concrete asset catalog, traffic bundle, and event-script interpreter
contracts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_spatial_index import MapSpatialIndex  # noqa: E402
from pedestrian_activity_catalog import get_activity, normalize_activity_type, validate_local_animation_assets  # noqa: E402
from uav_corridor_planner import BuildingObstacleIndex, UAV_CORRIDOR_LOGICAL_ASSET_ID  # noqa: E402
from semantic_event_contract import all_contracts, get_contract, required_intent_sequence_matches  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "Dataset"
DEFAULT_ASSET_CATALOG = ROOT / "Config" / "LowAltitude" / "asset_catalog.json"
DEFAULT_TRAFFIC_BUNDLE = ROOT / "Config" / "LowAltitude" / "Maps" / "donghu_road_topo" / "traffic_bundle"
DEFAULT_ROAD_GEOJSON = ROOT / "Content" / "Maps" / "donghu_road_topo" / "road" / "road.geojson"
DEFAULT_BUILDING_GEOJSON = ROOT / "Content" / "Maps" / "donghu_road_topo" / "building" / "building.geojson"
DEFAULT_RENDER_READY_ROOT = ROOT / "Dataset" / "render_ready_episodes"
DEFAULT_RENDER_HOST_CONFIG = ROOT / "Plugins" / "SumoImporter" / "Scripts" / "episode_render_host_config.json"
LANE_HALF_WIDTH_M = 1.9
PEDESTRIAN_ROAD_BUFFER_M = 0.25
CROWD_ROAD_BUFFER_M = 0.5
POSITION_MATCH_TOLERANCE_M = 1.25
MOVE_START_TOLERANCE_GROUND_M = 0.75
MOVE_START_TOLERANCE_UAV_M = 1.5
UAV_INITIAL_ALTITUDE_MAX_M = 5.0
UAV_MISSION_ALTITUDE_MIN_M = 18.0
UAV_TERMINAL_ALTITUDE_MAX_M = 8.0
SEMANTIC_ALLOWED_STATIC_STATES = {
    "available",
    "idle",
    "waiting",
    "queued",
    "blocked",
    "blocked_by_barrier",
    "held",
    "stopped",
    "hold",
    "traffic_flow",
    "traffic_slow",
    "cautious_flow",
    "braking",
    "evacuating",
    "retreat",
    "frozen",
    "landing",
    "touchdown",
    "landed",
    "preflight_on_pad",
    "inspect_racetrack",
    "orbit",
    "patrol",
}
BACKGROUND_VEHICLE_ALLOWED_STATES = {
    "blocked",
    "blocked_by_barrier",
    "braking",
    "cautious_flow",
    "detour",
    "held",
    "queued",
    "responder",
    "stopped",
    "traffic_flow",
    "traffic_slow",
    "yielding",
}
BACKGROUND_PEDESTRIAN_ALLOWED_STATES = {
    "chatting",
    "evacuating",
    "medical_incident",
    "observing",
    "retreat",
    "waiting",
    "walking",
}
INITIAL_OVERLAP_MIN_M = {
    ("pedestrian", "pedestrian"): 0.75,
    ("vehicle", "vehicle"): 2.2,
    ("pedestrian", "vehicle"): 1.5,
}
PEDESTRIAN_VEHICLE_DYNAMIC_CLEARANCE_MIN_M = 4.0
PEDESTRIAN_VEHICLE_COLLISION_EVENT_TYPE = "intentional_pedestrian_vehicle_collision"
PEDESTRIAN_VEHICLE_INTERACTION_EVENT_TYPE = "intentional_pedestrian_vehicle_interaction"
PEDESTRIAN_VEHICLE_ALLOWED_EVENT_TYPES = {
    PEDESTRIAN_VEHICLE_COLLISION_EVENT_TYPE,
    PEDESTRIAN_VEHICLE_INTERACTION_EVENT_TYPE,
}
PEDESTRIAN_VEHICLE_CLEARANCE_CHECK = "pedestrian_vehicle_dynamic_clearance"
GROUND_FLOW_MIN_VISIBLE_MOTION_RATIO = 0.85
GROUND_FLOW_SPEED_EPS_MPS = 0.05
GROUND_FLOW_MIN_XY_SPAN_M = {
    "pedestrian": 8.0,
    "vehicle": 18.0,
}
RENDER_READY_MAX_SPEED_MPS = {
    "pedestrian": 6.0,
    "vehicle": 45.0,
    "uav": 90.0,
}
FORBIDDEN_GROUND_FLOW_LOOP_POLICIES = {"bounce_between_route_waypoints"}
GROUND_FLOW_PINGPONG_REVERSAL_LIMIT = 2
GROUND_FLOW_PINGPONG_SPAN_M = {
    "pedestrian": 14.0,
    "vehicle": 40.0,
}
ROI_MARGIN_M = 1000.0
SCRIPT_TICK_HZ = 10.0
DELAY_SAFETY_FACTOR = 1.1
MAX_WAYPOINT_SEGMENT_M = {
    "uav": 300.0,
    "vehicle": 80.0,
    "pedestrian": 20.0,
    "asset": 120.0,
}
_BUILDING_INDEX_CACHE: BuildingObstacleIndex | None = None
LANDING_PAD_LOGICAL_ASSET_ID = "facility.landing_pad.visible.v1"
RENDER_CONFIG_DISABLED_FLAGS = (
    ("entity_overlap_filter", "enabled"),
    ("vehicle_lane_offsets", "enabled"),
    ("pedestrian_roadside_projection", "enabled"),
)


@dataclass(frozen=True)
class LaneSample:
    edge_id: str
    lane_index: int
    s_m: float
    x_m: float
    y_m: float
    yaw_deg: float


@dataclass(frozen=True)
class RoadLaneMetadata:
    lanes: int
    width_m: float


def load_road_lane_metadata(path: Path) -> dict[str, RoadLaneMetadata]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    result: dict[str, RoadLaneMetadata] = {}
    for feature in payload.get("features") or []:
        properties = dict(feature.get("properties") or {})
        road_id = properties.get("id")
        if road_id is None:
            continue
        try:
            lanes = max(1, int(float(properties.get("lanes") or 1)))
            width_m = max(0.0, float(properties.get("width") or 0.0))
        except (TypeError, ValueError):
            continue
        result[f"cg_edge_{road_id}"] = RoadLaneMetadata(lanes=lanes, width_m=width_m)
    return result


ROAD_LANE_METADATA = load_road_lane_metadata(DEFAULT_ROAD_GEOJSON)


def physical_vehicle_lane_offsets(edge_id: str) -> list[float]:
    metadata = ROAD_LANE_METADATA.get(edge_id)
    if metadata is None or metadata.lanes <= 1:
        return [0.0]
    lane_spacing_m = metadata.width_m / float(metadata.lanes) if metadata.width_m > 0.0 else LANE_HALF_WIDTH_M * 2.0
    return [
        ((float(index) + 0.5) - 0.5 * float(metadata.lanes)) * lane_spacing_m
        for index in range(metadata.lanes)
    ]


class LaneResolver:
    def __init__(self, lane_samples_csv: Path) -> None:
        self.samples: list[LaneSample] = []
        self.by_edge: dict[str, list[LaneSample]] = {}
        with lane_samples_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"edge_id", "lane_index", "s_m", "x_m", "y_m", "yaw_deg"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{lane_samples_csv} is not a lane center sample CSV; missing columns {sorted(missing)}. "
                    "Use traffic_bundle/lane_center_samples.csv, not lane_meta.csv."
                )
            for row in reader:
                sample = LaneSample(
                    edge_id=row["edge_id"],
                    lane_index=int(row["lane_index"]),
                    s_m=float(row["s_m"]),
                    x_m=float(row["x_m"]),
                    y_m=float(row["y_m"]),
                    yaw_deg=float(row.get("yaw_deg") or 0.0),
                )
                self.samples.append(sample)
                self.by_edge.setdefault(sample.edge_id, []).append(sample)
        for edge_samples in self.by_edge.values():
            edge_samples.sort(key=lambda item: item.s_m)
        if not self.samples:
            raise ValueError(f"{lane_samples_csv} contains no lane samples")
        xs = [sample.x_m for sample in self.samples]
        ys = [sample.y_m for sample in self.samples]
        self.x_min = min(xs) - ROI_MARGIN_M
        self.x_max = max(xs) + ROI_MARGIN_M
        self.y_min = min(ys) - ROI_MARGIN_M
        self.y_max = max(ys) + ROI_MARGIN_M

    def nearest(self, pos: list[float]) -> LaneSample:
        return min(self.samples, key=lambda item: (item.x_m - pos[0]) ** 2 + (item.y_m - pos[1]) ** 2)

    def resolve(self, edge_id: str, s_m: float) -> LaneSample:
        if edge_id not in self.by_edge:
            raise KeyError(edge_id)
        return min(self.by_edge[edge_id], key=lambda item: abs(item.s_m - s_m))

    def in_roi(self, pos: list[float]) -> bool:
        return self.x_min <= pos[0] <= self.x_max and self.y_min <= pos[1] <= self.y_max

    def roi_label(self) -> str:
        return f"x=[{self.x_min:.1f},{self.x_max:.1f}], y=[{self.y_min:.1f},{self.y_max:.1f}]"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    return rows


def asset_ids(catalog_path: Path) -> set[str]:
    data = load_json(catalog_path)
    return {item["logical_asset_id"] for item in data.get("assets", [])}


def building_ids(building_geojson_path: Path) -> set[str]:
    data = load_json(building_geojson_path)
    ids: set[str] = set()
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        source_id = props.get("id_origin") or props.get("id")
        if source_id is not None:
            ids.add(f"building_geojson_{source_id}")
    return ids


def pos3(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return None


def entity_pos(entity: dict[str, Any]) -> list[float] | None:
    placement = entity.get("placement", {})
    for key in ("resolved_position_enu_m", "position_enu_m", "center_enu_m"):
        found = pos3(placement.get(key))
        if found:
            return found
    if entity.get("placement_mode") == "polygon_prism":
        polygon = placement.get("polygon_enu_m") or []
        if polygon:
            x = sum(point[0] for point in polygon) / len(polygon)
            y = sum(point[1] for point in polygon) / len(polygon)
            z = float(placement.get("base_z_m", 0.0))
            return [x, y, z]
    return None


def entity_mission_start_pos(entity: dict[str, Any]) -> list[float] | None:
    lifecycle = entity.get("lifecycle") or {}
    return pos3(lifecycle.get("mission_start_enu_m")) or entity_pos(entity)


def uav_role(entity: dict[str, Any]) -> str:
    initial_state = dict(entity.get("initial_state") or {})
    if str(initial_state.get("role") or "") == "U_inspect":
        return "inspect"
    corridor_role = str(entity.get("uav_corridor_role") or initial_state.get("uav_corridor_role") or "")
    if corridor_role == "inspect_observer":
        return "inspect"
    if corridor_role == "observer" or "observer" in str(initial_state.get("semantic_role") or ""):
        return "observer"
    return "mission"


def is_inspect_uav(entity: dict[str, Any]) -> bool:
    return uav_role(entity) == "inspect"


def is_observer_uav(entity: dict[str, Any]) -> bool:
    return uav_role(entity) == "observer"


def is_mission_uav(entity: dict[str, Any]) -> bool:
    return uav_role(entity) == "mission"


def uav_action_is_terminal_or_low_profile(action_id: Any) -> bool:
    lowered = str(action_id or "").lower()
    return any(
        token in lowered
        for token in (
            "landing",
            "touchdown",
            "debris",
            "crash",
            "forced",
            "falling",
            "descent",
            "to_pad",
            "ped_descent",
            "nearmiss",
            "near_miss",
            "pull_up",
            "recovery",
            "takeoff",
            "liftoff",
            "launch",
        )
    )


def dist_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_oriented_frustum_footprint_xy(
    point: list[float],
    a: list[float],
    b: list[float],
    half_width_m: float,
    half_height_m: float,
) -> bool:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(point[0]), float(point[1])
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return abs(px - ax) <= half_width_m and abs(py - ay) <= half_height_m
    ux = dx / length
    uy = dy / length
    rel_x = px - ax
    rel_y = py - ay
    along = rel_x * ux + rel_y * uy
    cross = abs(-rel_x * uy + rel_y * ux)
    return -half_height_m <= along <= length + half_height_m and cross <= half_width_m


def dist3(a: list[float], b: list[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def point_in_polygon_xy(point: list[float], polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(polygon) - 1
    for i, pi in enumerate(polygon):
        pj = polygon[j]
        yi, yj = float(pi[1]), float(pj[1])
        xi, xj = float(pi[0]), float(pj[0])
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def segment_intersects_polygon_xy(a: list[float], b: list[float], polygon: list[list[float]]) -> bool:
    if point_in_polygon_xy(a, polygon) or point_in_polygon_xy(b, polygon):
        return True
    for edge_a, edge_b in zip(polygon, polygon[1:] + polygon[:1]):
        if segments_intersect_xy(a, b, edge_a, edge_b):
            return True
    return False


def _orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (float(b[1]) - float(a[1])) * (float(c[0]) - float(b[0])) - (float(b[0]) - float(a[0])) * (float(c[1]) - float(b[1]))


def _on_segment(a: list[float], b: list[float], c: list[float]) -> bool:
    return (
        min(float(a[0]), float(c[0])) - 1e-6 <= float(b[0]) <= max(float(a[0]), float(c[0])) + 1e-6
        and min(float(a[1]), float(c[1])) - 1e-6 <= float(b[1]) <= max(float(a[1]), float(c[1])) + 1e-6
    )


def segments_intersect_xy(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) <= 1e-6 and _on_segment(a, c, b):
        return True
    if abs(o2) <= 1e-6 and _on_segment(a, d, b):
        return True
    if abs(o3) <= 1e-6 and _on_segment(c, a, d):
        return True
    if abs(o4) <= 1e-6 and _on_segment(c, b, d):
        return True
    return False


def yaw_delta_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def rotation_yaw(rotation: dict[str, Any] | None) -> float | None:
    if not isinstance(rotation, dict):
        return None
    if "yaw_deg" in rotation or "yaw" in rotation:
        return float(rotation.get("yaw_deg", rotation.get("yaw", 0.0)))
    return None


def polygon_centroid(points: list[list[float]]) -> list[float] | None:
    if not points:
        return None
    return [
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    ]


def motion_class(asset_id: str) -> str:
    if asset_id.startswith("uav."):
        return "uav"
    if asset_id.startswith("vehicle."):
        return "vehicle"
    if asset_id.startswith("pedestrian."):
        return "pedestrian"
    return "asset"


def placement_allows_green(entity: dict[str, Any]) -> bool:
    placement = dict(entity.get("placement") or {})
    if placement.get("allow_green") is True:
        return True
    semantics = str(placement.get("placement_semantics") or "").lower()
    if semantics == "sidewalk_or_plaza":
        return False
    text = " ".join(
        str(value or "").lower()
        for value in (
            semantics,
            entity.get("placement_mode"),
            entity.get("category"),
            entity.get("entity_id"),
        )
    )
    return any(token in text for token in ("gather", "evacuation", "crowd", "park", "plaza", "safe"))


def action_allows_green(action_id: Any) -> bool:
    text = str(action_id or "").lower()
    return any(token in text for token in ("crowd", "evac", "gather", "safe"))


def is_crossing_action(action_id: Any) -> bool:
    text = str(action_id or "").lower()
    return any(token in text for token in ("crosswalk", "roadway", "retreat", "jaywalk"))


def path_length_m(waypoints: list[list[float]]) -> float:
    return sum(dist3(a, b) for a, b in zip(waypoints, waypoints[1:]))


def point_at_path_fraction(waypoints: list[list[float]], fraction: float) -> list[float] | None:
    if not waypoints:
        return None
    if len(waypoints) == 1:
        return list(waypoints[0])
    total = path_length_m(waypoints)
    if total <= 1e-6:
        return list(waypoints[0])
    target = max(0.0, min(1.0, fraction)) * total
    traversed = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        segment = dist3(a, b)
        if segment <= 1e-6:
            continue
        if traversed + segment >= target:
            alpha = (target - traversed) / segment
            return [a[i] + (b[i] - a[i]) * alpha for i in range(3)]
        traversed += segment
    return list(waypoints[-1])


def action_move_duration_ticks(action: dict[str, Any]) -> int:
    waypoints = [pos3(item) for item in action.get("waypoints_enu_m", [])]
    waypoints = [item for item in waypoints if item]
    if len(waypoints) < 2:
        return 0
    velocity = max(0.1, float(action.get("velocity_mps", 1.0)))
    return int(math.ceil(path_length_m(waypoints) / velocity * SCRIPT_TICK_HZ * DELAY_SAFETY_FACTOR))


def check_roi_point(scenario_id: str, label: str, point: list[float] | None, lanes: LaneResolver, issues: list[str]) -> None:
    if point and not lanes.in_roi(point):
        issues.append(f"{scenario_id}: {label} outside traffic_bundle ROI {lanes.roi_label()}: {point}")


def lane_normal(sample: LaneSample) -> tuple[float, float]:
    yaw_rad = math.radians(sample.yaw_deg)
    return -math.sin(yaw_rad), math.cos(yaw_rad)


def offset_from_lane(sample: LaneSample, lateral_m: float, z_m: float = 0.0) -> list[float]:
    nx, ny = lane_normal(sample)
    return [sample.x_m + lateral_m * nx, sample.y_m + lateral_m * ny, z_m]


def nearest_lane_clearance(lanes: LaneResolver, point: list[float]) -> float:
    sample = lanes.nearest(point)
    return dist_xy(point, [sample.x_m, sample.y_m, 0.0])


def lateral_from_lane(sample: LaneSample, point: list[float]) -> float:
    nx, ny = lane_normal(sample)
    return (float(point[0]) - sample.x_m) * nx + (float(point[1]) - sample.y_m) * ny


def check_vehicle_physical_lane_point(
    scenario_id: str,
    context: str,
    point: list[float],
    lanes: LaneResolver,
    issues: list[str],
) -> None:
    sample = lanes.nearest(point)
    physical_offsets = physical_vehicle_lane_offsets(sample.edge_id)
    lateral = lateral_from_lane(sample, point)
    if physical_offsets and min(abs(lateral - value) for value in physical_offsets) > 0.35:
        issues.append(
            f"{scenario_id}: {context} lateral {lateral:.2f}m does not match physical lane centers "
            f"{[round(value, 2) for value in physical_offsets]} on {sample.edge_id}"
        )


def check_landing_pad_spatial(context: str, position: list[float], spatial: MapSpatialIndex, issues: list[str]) -> None:
    for error in spatial.validation_errors_for_point(
        position,
        context=context,
        allow_road=False,
        allow_green=True,
        road_buffer_m=PEDESTRIAN_ROAD_BUFFER_M,
    ):
        issues.append(error)


def action_iter(script: dict[str, Any]):
    for event in script.get("events", []):
        for action in event.get("actions", []):
            yield event, action


def validation_event_field(event: dict[str, Any], field: str) -> Any:
    value = event.get(field)
    if value not in (None, "", []):
        return value
    for nested_key in ("metadata", "payload"):
        nested = event.get(nested_key)
        if isinstance(nested, dict):
            value = nested.get(field)
            if value not in (None, "", []):
                return value
    return None


def validation_skip_checks(event: dict[str, Any]) -> set[str]:
    checks = validation_event_field(event, "validation_skip_checks")
    if isinstance(checks, (list, tuple, set)):
        return {str(item) for item in checks}
    if isinstance(checks, str) and checks:
        return {checks}
    return set()


def event_allows_validation_skip(event: dict[str, Any], check_name: str, issues: list[str], scenario_id: str) -> bool:
    event_type = str(validation_event_field(event, "validation_event_type") or "")
    skip_checks = validation_skip_checks(event)
    if not event_type and check_name not in skip_checks:
        return False
    if event_type != "intentional_motion_discontinuity":
        issues.append(
            f"{scenario_id}: event {event.get('event_id')} has unsupported validation_event_type={event_type!r}; "
            "only intentional_motion_discontinuity may skip motion continuity checks"
        )
        return False
    if check_name not in skip_checks:
        return False
    if not str(validation_event_field(event, "validation_reason") or "").strip():
        issues.append(f"{scenario_id}: event {event.get('event_id')} skips {check_name} without validation_reason")
        return False
    return True


def known_pedestrian_activity(value: Any, *, moving: bool = False) -> str | None:
    try:
        return get_activity(str(value or ""), moving=moving).activity_type
    except ValueError:
        return None


def event_entity_refs(event: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for target in event.get("log_event", {}).get("target_ids", []):
        if target:
            refs.add(str(target))
    for action in event.get("actions", []):
        for key in ("entity_id", "ped_id"):
            value = action.get(key)
            if value:
                refs.add(str(value))
    return refs


def referenced_events(script: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for trigger in script.get("triggers", []):
        if trigger.get("type") in {"event_fired_after", "event_fired"}:
            value = trigger.get("event_id") or trigger.get("event_ref")
            if value:
                refs.add(str(value))
    return refs


def contract_stage_terms(text: str) -> list[list[str]]:
    stages: list[list[str]] = []
    for raw_stage in str(text or "").split(">"):
        options: list[str] = []
        for raw_option in raw_stage.split("/"):
            normalized = " ".join(token for token in str(raw_option).lower().replace("-", " ").replace("_", " ").split() if token)
            if normalized:
                options.append(normalized)
        if options:
            stages.append(options)
    return stages


def sequence_contains_terms(sequence: Any, terms: list[str]) -> bool:
    if not isinstance(sequence, list):
        return False
    hay = " ".join(str(item).lower() for item in sequence)
    return all(term in hay for term in terms)


def sequence_matches_allowed_states(sequence: Any, allowed_states: set[str]) -> bool:
    if not isinstance(sequence, list) or not sequence:
        return False
    normalized = [str(item).strip().lower() for item in sequence if str(item).strip()]
    return bool(normalized) and all(item in allowed_states for item in normalized)


def as_tuple_str(value: Any) -> tuple[str, ...]:
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return ()


def close_enough(a: float, b: float, tolerance: float = 0.001) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def route_is_closed(route: list[list[float]], tolerance_m: float = 1.0) -> bool:
    return len(route) >= 4 and dist3(route[0], route[-1]) <= tolerance_m


def route_fixed_altitude(route: list[list[float]], altitude_m: float, tolerance_m: float = 0.001) -> bool:
    return bool(route) and all(close_enough(point[2], altitude_m, tolerance_m) for point in route)


def event_field_haystack(event: dict[str, Any]) -> str:
    payload = dict(event.get("payload") or {})
    parts = [
        event.get("event_id"),
        event.get("topic"),
        event.get("source_event_id"),
        event.get("source_topic"),
        event.get("intent"),
        event.get("intent_stage"),
        event.get("causal_predecessor_intent"),
        payload.get("title"),
        payload.get("category"),
        payload.get("phase"),
        payload.get("source_kind"),
        payload.get("event_id"),
    ]
    return " ".join(str(part or "").lower() for part in parts)


def declared_ids(scene: dict[str, Any]) -> set[str]:
    return {entity["entity_id"] for entity in scene.get("entities", [])}


def is_ground_asset(asset_id: str) -> bool:
    if asset_id == UAV_CORRIDOR_LOGICAL_ASSET_ID:
        return False
    return asset_id.startswith(("vehicle.", "pedestrian.", "prop.", "facility.", "semantic."))


def check_entity_references(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    ids = declared_ids(scene)
    for trigger in script.get("triggers", []):
        for key in ("entity_a", "entity_b"):
            value = trigger.get(key)
            if value and value not in ids:
                issues.append(f"{script['scenario_id']}: trigger {trigger['trigger_id']} references undeclared {value}")
        if trigger.get("type") == "composite":
            trigger_ids = {item["trigger_id"] for item in script.get("triggers", [])}
            for child in trigger.get("children", []):
                if child not in trigger_ids:
                    issues.append(f"{script['scenario_id']}: composite trigger {trigger['trigger_id']} has missing child {child}")

    for event, action in action_iter(script):
        for key in ("entity_id", "ped_id"):
            value = action.get(key)
            if value and value not in ids:
                issues.append(f"{script['scenario_id']}: action {action.get('action_id')} references undeclared {value}")
        if "$param." in str(action.get("action_id", "")):
            issues.append(f"{script['scenario_id']}: action_id contains unexpanded parameter: {action.get('action_id')}")
        for target_id in event.get("log_event", {}).get("target_ids", []):
            if target_id not in ids:
                issues.append(f"{script['scenario_id']}: log_event target_id is not declared: {target_id}")


def check_contract_payload(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    contract = get_contract(scene.get("scenario_id") or script.get("scenario_id") or scene.get("episode_id") or script.get("episode_id") or "")
    scene_rule_payloads = [
        dict(rule.get("contract") or {})
        for rule in scene.get("validation_rules", [])
        if rule.get("rule") == "semantic_event_contract"
    ]
    if len(scene_rule_payloads) != 1:
        issues.append(f"{script['scenario_id']}: scene validation_rules must contain exactly one semantic_event_contract rule")
    elif scene_rule_payloads[0] != script.get("parameters", {}).get("semantic_event_contract"):
        issues.append(f"{script['scenario_id']}: semantic_event_contract rule does not match event_script parameters")
    payload = dict(script.get("parameters", {}).get("semantic_event_contract") or {})
    if str(payload.get("schema") or "") != "low_altitude_event_chain_contract_v1":
        issues.append(f"{script['scenario_id']}: missing semantic_event_contract schema")
        return
    if dict(payload.get("exact_counts") or {}) != contract.counts:
        issues.append(f"{script['scenario_id']}: exact_counts do not match contract")
    if str(payload.get("required_event") or "") != contract.required_event:
        issues.append(f"{script['scenario_id']}: required_event does not match contract")
    if as_tuple_str(payload.get("required_intents")) != contract.required_intents:
        issues.append(f"{script['scenario_id']}: required_intents do not match contract")
    capture_boundary = dict(payload.get("capture_boundary") or {})
    if not capture_boundary:
        issues.append(f"{script['scenario_id']}: semantic_event_contract missing capture_boundary")
    else:
        expected_boundary = contract.capture_boundary
        boundary_id = str(capture_boundary.get("boundary_id") or "")
        source_entity_id = str(capture_boundary.get("source_entity_id") or "")
        scene_ids = declared_ids(scene)
        if not boundary_id:
            issues.append(f"{script['scenario_id']}: capture_boundary.boundary_id missing")
        elif boundary_id not in scene_ids and source_entity_id not in scene_ids:
            issues.append(f"{script['scenario_id']}: capture_boundary must resolve to a declared scene entity")
        if str(capture_boundary.get("geometry_source") or "") != expected_boundary.geometry_source:
            issues.append(f"{script['scenario_id']}: capture_boundary.geometry_source does not match contract")
        if as_tuple_str(capture_boundary.get("anchor_entity_roles")) != expected_boundary.anchor_entity_roles:
            issues.append(f"{script['scenario_id']}: capture_boundary.anchor_entity_roles do not match contract")
        if str(capture_boundary.get("boundary_role") or "") != expected_boundary.boundary_role:
            issues.append(f"{script['scenario_id']}: capture_boundary.boundary_role does not match contract")
        if str(capture_boundary.get("z_policy") or "") != expected_boundary.z_policy:
            issues.append(f"{script['scenario_id']}: capture_boundary.z_policy does not match contract")
    inspect = dict(payload.get("inspect") or {})
    if not inspect:
        issues.append(f"{script['scenario_id']}: semantic_event_contract missing inspect")
    else:
        if str(inspect.get("role") or "") != contract.inspect.role:
            issues.append(f"{script['scenario_id']}: inspect.role does not match contract")
        if str(inspect.get("altitude_code") or "") != contract.inspect.altitude_code:
            issues.append(f"{script['scenario_id']}: inspect.altitude_code does not match contract")
        if not close_enough(float(inspect.get("altitude_m") or -1.0), contract.inspect.altitude_m):
            issues.append(f"{script['scenario_id']}: inspect.altitude_m does not match contract")
        if float(inspect.get("min_path_length_m") or 0.0) != contract.inspect.min_path_length_m:
            issues.append(f"{script['scenario_id']}: inspect.min_path_length_m does not match contract")
        if str(inspect.get("required_presence") or "") != contract.inspect.required_presence:
            issues.append(f"{script['scenario_id']}: inspect.required_presence does not match contract")
        if str(inspect.get("motion_policy") or "") != contract.inspect.motion_policy:
            issues.append(f"{script['scenario_id']}: inspect.motion_policy does not match contract")
        if str(inspect.get("corridor_policy") or "") != contract.inspect.corridor_policy:
            issues.append(f"{script['scenario_id']}: inspect.corridor_policy does not match contract")
        if bool(inspect.get("fov_coverage_required")) != contract.inspect.fov_coverage_required:
            issues.append(f"{script['scenario_id']}: inspect.fov_coverage_required does not match contract")
        if bool(inspect.get("sensor_profile_required")) != contract.inspect.sensor_profile_required:
            issues.append(f"{script['scenario_id']}: inspect.sensor_profile_required does not match contract")
    pad_boundary_policy = payload.get("pad_boundary_policy")
    if isinstance(pad_boundary_policy, str):
        issues.append(f"{script['scenario_id']}: pad_boundary_policy must be explicit object, not legacy string")
    else:
        pad_boundary_policy = dict(pad_boundary_policy or {})
        if str(pad_boundary_policy.get("default") or "") != contract.pad_policy.default:
            issues.append(f"{script['scenario_id']}: pad_boundary_policy.default does not match contract")
        if as_tuple_str(pad_boundary_policy.get("inside_required_for")) != contract.pad_policy.inside_required_for:
            issues.append(f"{script['scenario_id']}: pad_boundary_policy.inside_required_for does not match contract")
    if bool(payload.get("uav_boundary_crossing_required")) != contract.uav_boundary_crossing_required:
        issues.append(f"{script['scenario_id']}: uav_boundary_crossing_required does not match contract")
    if bool(payload.get("inspect_fov_coverage_required")) != contract.inspect.fov_coverage_required:
        issues.append(f"{script['scenario_id']}: inspect_fov_coverage_required does not match contract")
    background = dict(payload.get("background_semantics") or {})
    if str(background.get("vehicle_role") or "") != contract.vehicle_role:
        issues.append(f"{script['scenario_id']}: background vehicle_role does not match contract")
    if str(background.get("pedestrian_role") or "") != contract.pedestrian_role:
        issues.append(f"{script['scenario_id']}: background pedestrian_role does not match contract")
    if str(payload.get("weather") or "") != contract.weather:
        issues.append(f"{script['scenario_id']}: weather does not match contract")
    determinism = dict(payload.get("determinism") or {})
    for key in ("fallback", "guessing", "compatibility_paths"):
        if str(determinism.get(key) or "") != "forbidden":
            issues.append(f"{script['scenario_id']}: deterministic contract must mark {key}=forbidden")


def check_event_intent_contract(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    contract = get_contract(scene.get("scenario_id") or script.get("scenario_id") or "")
    events = [dict(event) for event in script.get("events", [])]
    for index, event in enumerate(events):
        event_id = str(event.get("event_id") or event.get("topic") or index)
        payload = dict(event.get("payload") or {})
        if not event.get("intent") and not payload.get("intent"):
            issues.append(f"{script['scenario_id']}: every event must declare explicit intent: {event_id}")
        if not event.get("intent_stage"):
            issues.append(f"{script['scenario_id']}: event missing intent_stage: {event_id}")
        if not event.get("causal_chain_id"):
            issues.append(f"{script['scenario_id']}: event missing causal_chain_id: {event_id}")
        if "causal_predecessor_intent" not in event:
            issues.append(f"{script['scenario_id']}: event missing causal_predecessor_intent field: {event_id}")
        if not isinstance(event.get("target_roles"), list) or not event.get("target_roles"):
            issues.append(f"{script['scenario_id']}: event missing target_roles: {event_id}")
    ok, matched = required_intent_sequence_matches(contract.required_intents, events)
    if not ok:
        issues.append(f"{script['scenario_id']}: required_intents declaration order mismatch: {contract.required_intents} (matched={matched})")


def check_scene_contract_counts(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    contract = get_contract(scene.get("scenario_id") or script.get("scenario_id") or "")
    entities = scene.get("entities", [])
    counts = {
        "uav": 0,
        "vehicle": 0,
        "pedestrian": 0,
        "facility": 0,
        "logical": 0,
    }
    background_vehicle_semantics: list[str] = []
    background_pedestrian_semantics: list[str] = []
    inspect_entities: list[dict[str, Any]] = []
    for entity in entities:
        asset_id = str(entity.get("logical_asset_id") or "")
        category = str(entity.get("category") or "")
        entity_id = str(entity.get("entity_id") or "")
        if asset_id.startswith("uav."):
            counts["uav"] += 1
        elif asset_id.startswith("vehicle.") or category == "vehicle":
            counts["vehicle"] += 1
        elif asset_id.startswith("pedestrian.") or category == "pedestrian":
            counts["pedestrian"] += 1
        elif entity_id.startswith("pad_home_"):
            pass
        elif asset_id.startswith("facility.") or category in {"facility", "traffic_signal"}:
            counts["facility"] += 1
        elif asset_id == UAV_CORRIDOR_LOGICAL_ASSET_ID or asset_id.startswith("trigger.") or category in {"airspace_constraint", "hazard_zone", "crowd_anchor", "airspace_corridor"}:
            counts["logical"] += 1
        initial_state = dict(entity.get("initial_state") or {})
        role = str(initial_state.get("role") or "")
        state_sequence = initial_state.get("state_sequence")
        semantic_role = str(initial_state.get("semantic_role") or "")
        if role == "U_inspect":
            inspect_entities.append(entity)
        is_background_vehicle = role == "semantic_background_vehicle" or entity_id.startswith("bg_vehicle_")
        is_background_pedestrian = role == "semantic_background_pedestrian" or entity_id.startswith("bg_ped_")
        if is_background_vehicle:
            if role != "semantic_background_vehicle":
                issues.append(f"{script['scenario_id']}: background vehicle {entity_id} role must be semantic_background_vehicle")
            background_vehicle_semantics.append(semantic_role)
            if not sequence_matches_allowed_states(state_sequence, BACKGROUND_VEHICLE_ALLOWED_STATES):
                issues.append(f"{script['scenario_id']}: background vehicle {entity_id} lacks exact semantic traffic state sequence")
            if semantic_role != contract.vehicle_role:
                issues.append(f"{script['scenario_id']}: background vehicle {entity_id} semantic_role must match contract vehicle_role")
        if is_background_pedestrian:
            if role != "semantic_background_pedestrian":
                issues.append(f"{script['scenario_id']}: background pedestrian {entity_id} role must be semantic_background_pedestrian")
            background_pedestrian_semantics.append(semantic_role)
            if not sequence_matches_allowed_states(state_sequence, BACKGROUND_PEDESTRIAN_ALLOWED_STATES):
                issues.append(f"{script['scenario_id']}: background pedestrian {entity_id} lacks exact semantic pedestrian state sequence")
            if semantic_role != contract.pedestrian_role:
                issues.append(f"{script['scenario_id']}: background pedestrian {entity_id} semantic_role must match contract pedestrian_role")
    for key, expected in contract.counts.items():
        actual = counts[key]
        if actual != expected:
            issues.append(f"{script['scenario_id']}: source scene count mismatch for {key}: expected {expected}, got {actual}")
    if len(inspect_entities) != 1:
        issues.append(f"{script['scenario_id']}: source scene must contain exactly one U_inspect")
    else:
        inspect = inspect_entities[0]
        initial_state = dict(inspect.get("initial_state") or {})
        inspect_contract = dict(inspect.get("contract_inspect_uav") or {})
        initial_corridor_role = str(initial_state.get("uav_corridor_role") or "")
        top_corridor_role = str(inspect.get("uav_corridor_role") or "")
        if initial_corridor_role != top_corridor_role:
            issues.append(f"{script['scenario_id']}: U_inspect initial_state.uav_corridor_role must equal top-level uav_corridor_role")
        if top_corridor_role != "inspect_observer":
            issues.append(f"{script['scenario_id']}: U_inspect uav_corridor_role must be inspect_observer")
        if float(initial_state.get("assigned_altitude_m") or initial_state.get("inspect_altitude_m") or initial_state.get("altitude_m") or 0.0) != float(contract.inspect_altitude_m):
            issues.append(f"{script['scenario_id']}: U_inspect altitude mismatch")
        if str(initial_state.get("inspect_altitude_code") or "") != contract.inspect_code:
            issues.append(f"{script['scenario_id']}: U_inspect inspect_altitude_code mismatch")
        if str(inspect_contract.get("corridor_policy") or "") != contract.inspect.corridor_policy:
            issues.append(f"{script['scenario_id']}: U_inspect contract_inspect_uav.corridor_policy must be {contract.inspect.corridor_policy}")
        fixed_altitude = float(inspect_contract.get("fixed_altitude_m") or inspect_contract.get("inspect_altitude_m") or -1.0)
        if not close_enough(fixed_altitude, contract.inspect.altitude_m):
            issues.append(f"{script['scenario_id']}: U_inspect contract fixed altitude must equal contract.inspect.altitude_m")
        if not sequence_contains_terms(initial_state.get("state_sequence"), ["inspect", "orbit"]) and not sequence_contains_terms(initial_state.get("state_sequence"), ["inspect", "racetrack"]):
            issues.append(f"{script['scenario_id']}: U_inspect state sequence must include inspect/orbit or inspect/racetrack")
        route = [pos3(point) for point in inspect.get("route_waypoints_enu_m") or []]
        route = [point for point in route if point]
        if len(route) >= 2:
            if not route_fixed_altitude(route, contract.inspect_altitude_m):
                issues.append(f"{script['scenario_id']}: U_inspect.route_waypoints_enu_m must contain only fixed inspect altitude points")
            if not route_is_closed(route):
                issues.append(f"{script['scenario_id']}: U_inspect.route_waypoints_enu_m must be a closed loop")
            if path_length_m(route) < 80.0:
                issues.append(f"{script['scenario_id']}: U_inspect route is too short")
        else:
            issues.append(f"{script['scenario_id']}: U_inspect route missing")
        contract_route = [
            pos3(point)
            for point in inspect_contract.get("loop_route_enu_m")
            or inspect_contract.get("repaired_route_enu_m")
            or inspect_contract.get("planned_route_enu_m")
            or []
        ]
        contract_route = [point for point in contract_route if point]
        if len(contract_route) < 2 or path_length_m(contract_route) < 80.0:
            issues.append(f"{script['scenario_id']}: U_inspect contract route is missing or too short")
        elif not route_fixed_altitude(contract_route, contract.inspect_altitude_m):
            issues.append(f"{script['scenario_id']}: U_inspect contract route must contain only fixed inspect altitude points")
    if background_vehicle_semantics and any(not item for item in background_vehicle_semantics):
        issues.append(f"{script['scenario_id']}: background vehicle semantic_role missing")
    if background_pedestrian_semantics and any(not item for item in background_pedestrian_semantics):
        issues.append(f"{script['scenario_id']}: background pedestrian semantic_role missing")


def check_assets(scene: dict[str, Any], known_assets: set[str], issues: list[str]) -> None:
    for entity in scene.get("entities", []):
        asset_id = entity.get("logical_asset_id")
        if asset_id not in known_assets:
            issues.append(f"{scene['scenario_id']}: unknown logical_asset_id {asset_id} on {entity.get('entity_id')}")


def check_dynamic_spawn_policy(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    spawned_ids = {
        str(action.get("entity_id"))
        for _, action in action_iter(script)
        if action.get("type") == "spawn_entity" and action.get("entity_id")
    }
    for entity in scene.get("entities", []):
        activation_tick = int(entity.get("activation_tick", 0) or 0)
        if activation_tick <= 0:
            continue
        entity_id = entity["entity_id"]
        if entity.get("enabled") is not False:
            issues.append(f"{scene['scenario_id']}: dynamic entity {entity_id} activation_tick={activation_tick} must set enabled=false")
        if entity.get("spawn_policy") != "event_script_only":
            issues.append(f"{scene['scenario_id']}: dynamic entity {entity_id} must set spawn_policy=event_script_only")
        if entity_id not in spawned_ids:
            issues.append(f"{scene['scenario_id']}: dynamic entity {entity_id} has no spawn_entity action")


def check_spawn_schema(scene: dict[str, Any], script: dict[str, Any], known_assets: set[str], issues: list[str]) -> None:
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    for _, action in action_iter(script):
        if action.get("type") != "spawn_entity":
            continue
        action_id = action.get("action_id")
        entity_id = str(action.get("entity_id") or "")
        asset_id = str(action.get("asset_id") or "")
        if asset_id not in known_assets:
            issues.append(f"{script['scenario_id']}: spawn_entity {action_id} asset_id is not a catalog logical_asset_id: {asset_id}")
        entity = entities.get(entity_id)
        if not entity:
            continue
        if entity.get("logical_asset_id") != asset_id:
            issues.append(f"{script['scenario_id']}: spawn_entity {action_id} asset_id {asset_id} does not match scene_setup logical_asset_id {entity.get('logical_asset_id')}")
        if int(entity.get("activation_tick", 0) or 0) <= 0:
            issues.append(f"{script['scenario_id']}: spawn_entity {action_id} targets non-dynamic scene entity {entity_id}")
        scene_yaw = rotation_yaw((entity.get("placement") or {}).get("rotation_deg"))
        action_yaw = rotation_yaw(action.get("rotation_deg"))
        if scene_yaw is not None and action_yaw is not None and yaw_delta_deg(scene_yaw, action_yaw) > 1.0:
            issues.append(f"{script['scenario_id']}: spawn_entity {action_id} yaw {action_yaw:.1f} differs from scene_setup yaw {scene_yaw:.1f}")


def check_placements(
    scene: dict[str, Any],
    lanes: LaneResolver,
    spatial: MapSpatialIndex,
    known_buildings: set[str],
    issues: list[str],
) -> None:
    roadwork_laterals: list[float] = []
    for entity in scene.get("entities", []):
        placement = entity.get("placement", {})
        mode = entity.get("placement_mode")
        position = entity_pos(entity)
        if position is None:
            issues.append(f"{scene['scenario_id']}: {entity['entity_id']} has no resolved position")
            continue
        check_roi_point(scene["scenario_id"], f"entity {entity['entity_id']} position", position, lanes, issues)
        asset_id = entity.get("logical_asset_id", "")
        category = entity.get("category", "")
        if str(asset_id) == LANDING_PAD_LOGICAL_ASSET_ID:
            check_landing_pad_spatial(
                f"{scene['scenario_id']}: landing pad {entity['entity_id']}",
                position,
                spatial,
                issues,
            )
        if str(asset_id).startswith("pedestrian."):
            initial_state = dict(entity.get("initial_state") or {})
            activity_state = dict((initial_state.get("state_facets") or {}).get("activity") or {})
            initial_activity = (
                initial_state.get("activity_type")
                or activity_state.get("activity_type")
                or initial_state.get("mode")
            )
            if not known_pedestrian_activity(initial_activity):
                issues.append(f"{scene['scenario_id']}: pedestrian {entity['entity_id']} has unknown initial activity_type {initial_activity!r}")
            allow_green = placement_allows_green(entity)
            for error in spatial.validation_errors_for_point(
                position,
                context=f"{scene['scenario_id']}: pedestrian {entity['entity_id']} position",
                allow_road=(mode == "crosswalk_anchor"),
                allow_green=allow_green,
            ):
                issues.append(error)
            route = [pos3(point) for point in entity.get("route_waypoints_enu_m") or []]
            route = [point for point in route if point]
            for index, point in enumerate(route):
                for error in spatial.validation_errors_for_point(
                    point,
                    context=f"{scene['scenario_id']}: pedestrian {entity['entity_id']} route waypoint {index}",
                    allow_road=(mode == "crosswalk_anchor"),
                    allow_green=allow_green,
                ):
                    issues.append(error)
            for index, (a, b) in enumerate(zip(route, route[1:])):
                for error in spatial.validation_errors_for_segment(
                    a,
                    b,
                    context=f"{scene['scenario_id']}: pedestrian {entity['entity_id']} route segment {index}",
                    allow_road=(mode == "crosswalk_anchor"),
                    allow_green=allow_green,
                ):
                    issues.append(error)
        if asset_id.startswith("uav."):
            if is_inspect_uav(entity) or is_observer_uav(entity):
                continue
            lifecycle = entity.get("lifecycle") or {}
            mission_start = pos3(lifecycle.get("mission_start_enu_m"))
            corridor_lifecycle = bool(lifecycle.get("corridor_lifecycle") or entity.get("uav_corridor"))
            if lifecycle:
                if position[2] > UAV_INITIAL_ALTITUDE_MAX_M and not corridor_lifecycle:
                    issues.append(f"{scene['scenario_id']}: UAV {entity['entity_id']} lifecycle initial z is too high: z={position[2]}")
                if not mission_start or mission_start[2] < UAV_MISSION_ALTITUDE_MIN_M:
                    issues.append(f"{scene['scenario_id']}: UAV {entity['entity_id']} lifecycle mission start is below mission altitude")
            elif position[2] < 20.0:
                issues.append(f"{scene['scenario_id']}: UAV {entity['entity_id']} starts below 20m without lifecycle takeoff metadata: z={position[2]}")
        if is_ground_asset(asset_id) and not asset_id.startswith("uav.") and abs(position[2]) > 5.0:
            if category not in {"traffic_signal", "airspace_constraint", "hazard_zone", "facade_anchor"} and not entity.get("initial_state", {}).get("mode") == "attached":
                issues.append(f"{scene['scenario_id']}: ground entity {entity['entity_id']} has unexpected z={position[2]}")

        if mode == "lane_anchor":
            edge_id = placement.get("edge_id")
            if edge_id not in lanes.by_edge:
                issues.append(f"{scene['scenario_id']}: lane_anchor edge_id does not exist: {edge_id}")
                continue
            sample = lanes.resolve(edge_id, float(placement.get("longitudinal_s", 0.0)))
            lateral = float(placement.get("resolved_lateral_from_center_m", placement.get("lateral_offset_m", 0.0)))
            expected = offset_from_lane(sample, lateral, position[2])
            if dist_xy(position, expected) > POSITION_MATCH_TOLERANCE_M:
                issues.append(f"{scene['scenario_id']}: lane_anchor {entity['entity_id']} resolved position does not match edge/s/lateral")
            if asset_id.startswith("prop.roadwork."):
                roadwork_laterals.append(lateral)
                required = float(placement.get("lane_half_width_m", LANE_HALF_WIDTH_M)) + abs(float(placement.get("lateral_offset_m", 0.0))) - 0.05
                if abs(lateral) < required:
                    issues.append(f"{scene['scenario_id']}: roadwork {entity['entity_id']} is not outside lane edge")
            elif asset_id.startswith("vehicle."):
                semantics = str(placement.get("placement_semantics") or "")
                physical_offsets = [
                    float(value)
                    for value in (
                        placement.get("physical_lane_center_offsets_m")
                        or physical_vehicle_lane_offsets(str(edge_id))
                    )
                ]
                road_width_m = float(placement.get("road_width_m") or 0.0)
                if road_width_m <= 0.0:
                    metadata = ROAD_LANE_METADATA.get(str(edge_id))
                    road_width_m = float(metadata.width_m if metadata else LANE_HALF_WIDTH_M * 2.0)
                if semantics == "physical_vehicle_lane_center":
                    road_half_width_m = max(LANE_HALF_WIDTH_M, road_width_m * 0.5)
                    if abs(lateral) > road_half_width_m + 0.05:
                        issues.append(f"{scene['scenario_id']}: vehicle {entity['entity_id']} physical lane offset is outside road width")
                elif abs(lateral) > LANE_HALF_WIDTH_M:
                    issues.append(f"{scene['scenario_id']}: vehicle {entity['entity_id']} lane_anchor is outside drivable lane")
                if physical_offsets and min(abs(lateral - value) for value in physical_offsets) > 0.25:
                    issues.append(f"{scene['scenario_id']}: vehicle {entity['entity_id']} lane offset does not match physical road lane centers")
                if (
                    semantics != "physical_vehicle_lane_center"
                    and len([value for value in physical_offsets if abs(value) > 1e-6]) > 0
                ):
                    issues.append(f"{scene['scenario_id']}: vehicle {entity['entity_id']} lacks physical_vehicle_lane_center placement semantics")

        if mode == "sidewalk_anchor":
            edge_id = placement.get("lane_edge_id")
            if edge_id not in lanes.by_edge:
                issues.append(f"{scene['scenario_id']}: sidewalk_anchor lane_edge_id does not exist: {edge_id}")
                continue
            sample = lanes.resolve(edge_id, float(placement.get("longitudinal_s", 0.0)))
            lateral = float(placement.get("resolved_lateral_from_center_m", 0.0))
            expected = offset_from_lane(sample, lateral, position[2])
            if dist_xy(position, expected) > POSITION_MATCH_TOLERANCE_M:
                issues.append(f"{scene['scenario_id']}: sidewalk_anchor {entity['entity_id']} resolved position does not match edge/s/lateral")
            required = float(placement.get("lane_half_width_m", LANE_HALF_WIDTH_M)) + float(placement.get("offset_from_curb_m", 0.0)) - 0.05
            if abs(lateral) < required:
                issues.append(f"{scene['scenario_id']}: pedestrian {entity['entity_id']} is inside roadway clearance")

        if mode == "crosswalk_anchor":
            start = pos3(placement.get("resolved_position_enu_m"))
            road = pos3(placement.get("roadway_center_position_enu_m"))
            opposite = pos3(placement.get("opposite_curb_position_enu_m"))
            check_roi_point(scene["scenario_id"], f"crosswalk {entity['entity_id']} roadway point", road, lanes, issues)
            check_roi_point(scene["scenario_id"], f"crosswalk {entity['entity_id']} opposite curb", opposite, lanes, issues)
            if not (start and road and opposite):
                issues.append(f"{scene['scenario_id']}: crosswalk_anchor {entity['entity_id']} lacks start/road/opposite resolved positions")
            else:
                if nearest_lane_clearance(lanes, start) <= LANE_HALF_WIDTH_M:
                    issues.append(f"{scene['scenario_id']}: crosswalk start {entity['entity_id']} is in roadway instead of curb side")
                for error in spatial.validation_errors_for_point(
                    start,
                    context=f"{scene['scenario_id']}: crosswalk {entity['entity_id']} start",
                    allow_road=False,
                    allow_green=False,
                ):
                    issues.append(error)
                for error in spatial.validation_errors_for_point(
                    road,
                    context=f"{scene['scenario_id']}: crosswalk {entity['entity_id']} roadway center",
                    allow_road=True,
                    allow_green=False,
                ):
                    issues.append(error)
                for error in spatial.validation_errors_for_point(
                    opposite,
                    context=f"{scene['scenario_id']}: crosswalk {entity['entity_id']} opposite curb",
                    allow_road=False,
                    allow_green=False,
                ):
                    issues.append(error)
                for label, a, b in (("start_to_road", start, road), ("road_to_opposite", road, opposite)):
                    for error in spatial.validation_errors_for_segment(
                        a,
                        b,
                        context=f"{scene['scenario_id']}: crosswalk {entity['entity_id']} {label}",
                        allow_road=True,
                        allow_green=False,
                    ):
                        issues.append(error)

        if mode == "facade_anchor":
            building_id = placement.get("building_id")
            if building_id not in known_buildings:
                issues.append(f"{scene['scenario_id']}: facade_anchor {entity['entity_id']} uses unknown building_id {building_id}")
            if not placement.get("building_source"):
                issues.append(f"{scene['scenario_id']}: facade_anchor {entity['entity_id']} lacks building_source")

        if mode == "polygon_prism":
            polygon = [pos3(point) for point in placement.get("polygon_enu_m", [])]
            polygon = [point for point in polygon if point]
            if len(polygon) < 3:
                issues.append(f"{scene['scenario_id']}: polygon_prism {entity['entity_id']} has fewer than 3 valid vertices")
            else:
                for index, point in enumerate(polygon):
                    check_roi_point(scene["scenario_id"], f"polygon {entity['entity_id']} vertex {index}", point, lanes, issues)
                centroid = polygon_centroid(polygon)
                resolved = pos3(placement.get("resolved_position_enu_m"))
                if centroid and resolved and dist_xy(centroid, resolved) > 5.0:
                    issues.append(
                        f"{scene['scenario_id']}: polygon_prism {entity['entity_id']} centroid is {dist_xy(centroid, resolved):.1f}m from resolved_position_enu_m"
                    )

    if roadwork_laterals and len({1 if item > 0 else -1 for item in roadwork_laterals}) > 1:
        issues.append(f"{scene['scenario_id']}: roadwork props are split across both lane sides")


def check_event_positions(scene: dict[str, Any], script: dict[str, Any], lanes: LaneResolver, spatial: MapSpatialIndex, issues: list[str]) -> None:
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    current = {entity_id: entity_pos(entity) for entity_id, entity in entities.items()}
    asset = {entity_id: entity.get("logical_asset_id", "") for entity_id, entity in entities.items()}

    for event, action in action_iter(script):
        action_type = action.get("type")
        entity_id = action.get("entity_id")
        if action_type == "play_animation":
            issues.append(f"{script['scenario_id']}: raw pedestrian play_animation action is forbidden in Dataset scenarios: {action.get('action_id')}")
        if action_type == "set_pedestrian_activity":
            activity_type = action.get("activity_type")
            normalized = known_pedestrian_activity(activity_type)
            if not normalized:
                issues.append(f"{script['scenario_id']}: set_pedestrian_activity {action.get('action_id')} has unknown activity_type {activity_type!r}")
            if entity_id not in entities:
                issues.append(f"{script['scenario_id']}: set_pedestrian_activity {action.get('action_id')} references undeclared pedestrian {entity_id}")
            elif not asset[entity_id].startswith("pedestrian."):
                issues.append(f"{script['scenario_id']}: set_pedestrian_activity {action.get('action_id')} target is not pedestrian: {entity_id}")
        if action_type == "spawn_entity" and entity_id in entities:
            spawn_pos = pos3(action.get("position_enu_m"))
            expected = entity_pos(entities[entity_id])
            check_roi_point(script["scenario_id"], f"spawn_entity {entity_id} position", spawn_pos, lanes, issues)
            if spawn_pos and expected and dist3(spawn_pos, expected) > POSITION_MATCH_TOLERANCE_M:
                issues.append(f"{script['scenario_id']}: spawn_entity {entity_id} position diverges from scene_setup resolved position")
            current[entity_id] = spawn_pos or expected

        if action_type == "move_entity" and entity_id in entities:
            waypoints = [pos3(item) for item in action.get("waypoints_enu_m", [])]
            waypoints = [item for item in waypoints if item]
            if not waypoints:
                issues.append(f"{script['scenario_id']}: move_entity {action.get('action_id')} has no valid waypoints")
                continue
            for index, point in enumerate(waypoints):
                check_roi_point(script["scenario_id"], f"move_entity {action.get('action_id')} waypoint {index}", point, lanes, issues)
            max_segment = MAX_WAYPOINT_SEGMENT_M[motion_class(asset[entity_id])]
            for index, (a, b) in enumerate(zip(waypoints, waypoints[1:])):
                segment_m = dist_xy(a, b)
                if segment_m > max_segment and not event_allows_validation_skip(
                    event,
                    "move_segment_continuity",
                    issues,
                    script["scenario_id"],
                ):
                    issues.append(
                        f"{script['scenario_id']}: move_entity {action.get('action_id')} segment {index}->{index + 1} is {segment_m:.1f}m, exceeds {motion_class(asset[entity_id])} limit {max_segment:.1f}m"
                    )
            expected_start = current.get(entity_id)
            if expected_start:
                tolerance = MOVE_START_TOLERANCE_UAV_M if asset[entity_id].startswith("uav.") else MOVE_START_TOLERANCE_GROUND_M
                if dist3(waypoints[0], expected_start) > tolerance and not event_allows_validation_skip(
                    event,
                    "move_start_continuity",
                    issues,
                    script["scenario_id"],
                ):
                    issues.append(f"{script['scenario_id']}: move_entity {action.get('action_id')} starts {dist3(waypoints[0], expected_start):.1f}m from current {entity_id} position")
            if asset[entity_id].startswith("uav."):
                low_altitude_allowed = any(token in str(action.get("action_id", "")) for token in ("debris", "touchdown", "landing"))
                for point in waypoints[:-1]:
                    if point[2] < 3.0 and not low_altitude_allowed:
                        issues.append(f"{script['scenario_id']}: UAV waypoint too low before terminal landing in {action.get('action_id')}: {point}")
            if asset[entity_id].startswith("vehicle."):
                for point in waypoints:
                    if abs(point[2]) > 0.75:
                        issues.append(f"{script['scenario_id']}: vehicle waypoint z is not ground level in {action.get('action_id')}: {point}")
                    check_vehicle_physical_lane_point(
                        script["scenario_id"],
                        f"vehicle move_entity {action.get('action_id')} waypoint",
                        point,
                        lanes,
                        issues,
                    )
            if asset[entity_id].startswith("pedestrian."):
                for field_name, moving in (("activity_type", True), ("post_activity_type", False)):
                    if field_name not in action:
                        issues.append(f"{script['scenario_id']}: pedestrian move_entity {action.get('action_id')} missing {field_name}")
                        continue
                    normalized = known_pedestrian_activity(action.get(field_name), moving=moving)
                    if not normalized:
                        issues.append(f"{script['scenario_id']}: pedestrian move_entity {action.get('action_id')} has unknown {field_name}={action.get(field_name)!r}")
                        continue
                    if field_name == "activity_type" and not get_activity(normalized).moving:
                        issues.append(f"{script['scenario_id']}: pedestrian move_entity {action.get('action_id')} uses non-moving activity_type={normalized}")
                    if field_name == "post_activity_type" and get_activity(normalized).moving:
                        issues.append(f"{script['scenario_id']}: pedestrian move_entity {action.get('action_id')} post_activity_type must be stationary, got {normalized}")
                crossing_context = is_crossing_action(action.get("action_id"))
                allow_green = action_allows_green(action.get("action_id"))
                for index, point in enumerate(waypoints):
                    if abs(point[2]) > 0.75:
                        issues.append(f"{script['scenario_id']}: pedestrian waypoint z is not ground level in {action.get('action_id')}: {point}")
                    for error in spatial.validation_errors_for_point(
                        point,
                        context=f"{script['scenario_id']}: pedestrian {entity_id} waypoint {index} in {action.get('action_id')}",
                        allow_road=crossing_context,
                        allow_green=allow_green,
                    ):
                        issues.append(error)
                    if not crossing_context and nearest_lane_clearance(lanes, point) < LANE_HALF_WIDTH_M + PEDESTRIAN_ROAD_BUFFER_M:
                        issues.append(f"{script['scenario_id']}: pedestrian non-crossing waypoint is inside roadway in {action.get('action_id')}: {point}")
                for index, (a, b) in enumerate(zip(waypoints, waypoints[1:])):
                    for error in spatial.validation_errors_for_segment(
                        a,
                        b,
                        context=f"{script['scenario_id']}: pedestrian {entity_id} segment {index} in {action.get('action_id')}",
                        allow_road=crossing_context,
                        allow_green=allow_green,
                    ):
                        issues.append(error)
            current[entity_id] = waypoints[-1]

        if action_type == "move_entity" and entity_id in entities:
            if entities[entity_id].get("category") in {"traffic_signal", "facility"} and asset[entity_id].startswith(("prop.traffic_control.signal_light", "facility.radio.base_tower")):
                issues.append(f"{script['scenario_id']}: non-movable infrastructure has move_entity action: {entity_id}")

        if action_type == "spawn_crowd":
            issues.append(f"{script['scenario_id']}: raw spawn_crowd action is forbidden; use explicit pedestrian cohort entities: {action.get('group_id')}")
            origin = pos3(action.get("spawn_origin_enu_m"))
            extent = action.get("spawn_box_extent_cm") or [0, 0, 0]
            check_roi_point(script["scenario_id"], f"spawn_crowd {action.get('group_id')} origin", origin, lanes, issues)
            if origin:
                for error in spatial.validation_errors_for_spawn_envelope(
                    origin,
                    extent,
                    context=f"{script['scenario_id']}: spawn_crowd {action.get('group_id')}",
                    allow_green=True,
                ):
                    issues.append(error)
                half_extent = max(float(extent[0]), float(extent[1])) / 200.0
                required = LANE_HALF_WIDTH_M + half_extent + CROWD_ROAD_BUFFER_M
                clearance = nearest_lane_clearance(lanes, origin)
                if clearance < required:
                    issues.append(f"{script['scenario_id']}: spawn_crowd {action.get('group_id')} overlaps roadway clearance ({clearance:.2f}m < {required:.2f}m)")


def check_cameras(scene: dict[str, Any], lanes: LaneResolver, issues: list[str]) -> None:
    for camera in scene.get("cameras", []):
        placement = camera.get("placement", {})
        position = pos3(placement.get("position_enu_m") or placement.get("resolved_position_enu_m"))
        check_roi_point(scene["scenario_id"], f"camera {camera.get('camera_id')} position", position, lanes, issues)


def check_event_delay_physics(script: dict[str, Any], issues: list[str]) -> None:
    events = {event["event_id"]: event for event in script.get("events", [])}
    for trigger in script.get("triggers", []):
        if trigger.get("type") != "event_fired_after":
            continue
        prior_event = events.get(trigger.get("event_id"))
        if not prior_event:
            continue
        required_ticks = max(
            (action_move_duration_ticks(action) for action in prior_event.get("actions", []) if action.get("type") == "move_entity"),
            default=0,
        )
        delay_ticks = int(trigger.get("delay_ticks", 0) or 0)
        if required_ticks > delay_ticks and not event_allows_validation_skip(
            prior_event,
            "event_delay_physics",
            issues,
            script["scenario_id"],
        ):
            issues.append(
                f"{script['scenario_id']}: trigger {trigger.get('trigger_id')} delay_ticks={delay_ticks} is shorter than prior move duration {required_ticks} ticks"
            )


def collect_scene_points(scene: dict[str, Any], script: dict[str, Any]) -> list[tuple[str, list[float]]]:
    points: list[tuple[str, list[float]]] = []
    for entity in scene.get("entities", []):
        position = entity_pos(entity)
        if position:
            points.append((f"entity {entity['entity_id']}", position))
        for waypoint in entity.get("route_waypoints_enu_m") or []:
            found = pos3(waypoint)
            if found:
                points.append((f"entity {entity['entity_id']} route waypoint", found))
        placement = entity.get("placement") or {}
        for index, vertex in enumerate(placement.get("polygon_enu_m") or []):
            found = pos3(vertex)
            if found:
                points.append((f"polygon {entity['entity_id']} vertex {index}", found))
    for camera in scene.get("cameras", []):
        placement = camera.get("placement", {})
        found = pos3(placement.get("position_enu_m") or placement.get("resolved_position_enu_m"))
        if found:
            points.append((f"camera {camera.get('camera_id')}", found))
    for _, action in action_iter(script):
        for key in ("position_enu_m", "spawn_origin_enu_m"):
            found = pos3(action.get(key))
            if found:
                points.append((f"action {action.get('action_id') or action.get('group_id')} {key}", found))
        for index, waypoint in enumerate(action.get("waypoints_enu_m") or []):
            found = pos3(waypoint)
            if found:
                points.append((f"action {action.get('action_id')} waypoint {index}", found))
    return points


def check_local_bounds(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    bounds = scene.get("local_bounds") or {}
    if bounds:
        source = str(bounds.get("source") or bounds.get("semantic_role") or bounds.get("boundary_role") or "").lower()
        if source and source != "export_envelope_only":
            issues.append(f"{scene['scenario_id']}: local_bounds must be export_bounds/export_envelope_only, not semantic boundary source={source}")
        for forbidden_key in ("capture_boundary_id", "uav_boundary_crossing_required", "pad_boundary_policy", "inspect_fov_coverage_required"):
            if forbidden_key in bounds:
                issues.append(f"{scene['scenario_id']}: local_bounds must not carry semantic boundary field {forbidden_key}")
    if "local_bounds" in dict(script.get("parameters") or {}):
        issues.append(f"{scene['scenario_id']}: event_script parameters must not use local_bounds as semantic boundary")
    contract_payload = dict(dict(script.get("parameters") or {}).get("semantic_event_contract") or {})
    contract_boundary = dict(contract_payload.get("capture_boundary") or {})
    if str(contract_boundary.get("geometry_source") or "") == "local_bounds":
        issues.append(f"{scene['scenario_id']}: capture_boundary.geometry_source must not be local_bounds")
    center = pos3(bounds.get("center_enu_m"))
    radius = float(bounds.get("radius_m", 0.0) or 0.0)
    if not center or radius <= 0.0:
        issues.append(f"{scene['scenario_id']}: scene_setup missing local_bounds.center_enu_m/radius_m")
        return


def check_weather_bootstrap(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    weather_triggers = [trigger for trigger in script.get("triggers", []) if trigger.get("type") == "weather_state"]
    if not weather_triggers:
        return
    tick_bootstraps: set[str] = set()
    trigger_by_id = {trigger["trigger_id"]: trigger for trigger in script.get("triggers", [])}
    for event, action in action_iter(script):
        if action.get("type") != "set_weather":
            continue
        trigger = trigger_by_id.get(event.get("trigger_ref"), {})
        if trigger.get("type") == "tick":
            profile = action.get("profile")
            if profile:
                tick_bootstraps.add(str(profile))
            for key in (action.get("overrides") or {}).keys():
                tick_bootstraps.add(str(key))
    transitions = scene.get("weather_profile", {}).get("transitions", [])
    for transition in transitions:
        profile = transition.get("profile")
        if profile:
            tick_bootstraps.add(str(profile))
        for key in (transition.get("overrides") or {}).keys():
            tick_bootstraps.add(str(key))
    for trigger in weather_triggers:
        parameter = str(trigger.get("parameter"))
        if parameter not in tick_bootstraps:
            issues.append(f"{script['scenario_id']}: weather_state trigger for {parameter} has no tick/weather_profile bootstrap")


def check_proximity_metrics(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    category = {entity["entity_id"]: entity.get("category", "") for entity in scene.get("entities", [])}
    for trigger in script.get("triggers", []):
        if trigger.get("type") != "entity_proximity":
            continue
        a = trigger.get("entity_a")
        b = trigger.get("entity_b")
        cat_a = category.get(a)
        cat_b = category.get(b)
        mobile = {"pedestrian", "vehicle", "uav"}
        if "uav" in {cat_a, cat_b} and cat_a in mobile and cat_b in mobile:
            metric = trigger.get("metric", "xy")
            if metric not in {"3d", "xy_plus_z"}:
                issues.append(f"{script['scenario_id']}: proximity trigger {trigger['trigger_id']} needs 3d/xy_plus_z metric")


def check_activity_causality(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    trigger_entity_refs: dict[str, set[str]] = {}
    for trigger in script.get("triggers", []):
        refs = {str(trigger.get(key)) for key in ("entity_a", "entity_b") if trigger.get(key)}
        if refs:
            trigger_entity_refs[str(trigger.get("trigger_id") or "")] = refs

    events = list(script.get("events", []))
    refs_by_event = [
        event_entity_refs(event) | trigger_entity_refs.get(str(event.get("trigger_ref") or ""), set())
        for event in events
    ]
    downstream_event_ids = referenced_events(script)
    for event_index, event in enumerate(events):
        log_targets = {str(target) for target in event.get("log_event", {}).get("target_ids", []) if target}
        later_refs: set[str] = set()
        for refs in refs_by_event[event_index + 1 :]:
            later_refs.update(refs)
        group_chain_related = bool(log_targets & later_refs)
        event_has_concrete_action = any(
            action.get("type") in {"move_entity", "capture_screenshot", "set_visual_state", "set_weather", "spawn_entity"}
            for action in event.get("actions", [])
        )
        event_id = str(event.get("event_id") or "")
        for action in event.get("actions", []):
            if action.get("type") != "set_pedestrian_activity":
                continue
            entity_id = str(action.get("entity_id") or "")
            if entity_id not in entities or not str(entities[entity_id].get("logical_asset_id", "")).startswith("pedestrian."):
                continue
            if entity_id not in log_targets:
                issues.append(
                    f"{script['scenario_id']}: semantic pedestrian action {action.get('action_id')} must include {entity_id} in log_event.target_ids"
                )
            chain_related = (
                event_id in downstream_event_ids
                or entity_id in later_refs
                or event_has_concrete_action
                or group_chain_related
            )
            if not chain_related:
                issues.append(
                    f"{script['scenario_id']}: semantic pedestrian action {action.get('action_id')} is not connected to a downstream trigger/action chain"
                )


def mobile_overlap_class(entity: dict[str, Any]) -> str | None:
    asset_id = str(entity.get("logical_asset_id", ""))
    if asset_id.startswith("pedestrian."):
        return "pedestrian"
    if asset_id.startswith("vehicle."):
        return "vehicle"
    return None


def check_initial_mobile_overlaps(scene: dict[str, Any], issues: list[str]) -> None:
    rows: list[tuple[str, str, list[float]]] = []
    for entity in scene.get("entities", []):
        if int(entity.get("activation_tick", 0) or 0) > 0 or entity.get("enabled") is False:
            continue
        cls = mobile_overlap_class(entity)
        pos = entity_pos(entity)
        if cls and pos:
            rows.append((str(entity["entity_id"]), cls, pos))
    for index, (entity_a, class_a, pos_a) in enumerate(rows):
        for entity_b, class_b, pos_b in rows[index + 1 :]:
            key = tuple(sorted((class_a, class_b)))
            threshold = INITIAL_OVERLAP_MIN_M.get(key)
            if threshold is None:
                continue
            distance = dist_xy(pos_a, pos_b)
            if distance < threshold:
                issues.append(
                    f"{scene['scenario_id']}: initial {class_a}/{class_b} overlap {entity_a} vs {entity_b}: {distance:.2f}m < {threshold:.2f}m"
                )


def scene_route_points(entity: dict[str, Any]) -> list[list[float]]:
    start = entity_pos(entity)
    if not start:
        return []
    route = [
        [float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else 0.0)]
        for point in entity.get("route_waypoints_enu_m") or []
        if isinstance(point, list) and len(point) >= 2
    ]
    return [start, *route]


def route_position_at_tick(route: list[list[float]], speed_mps: float, tick_value: int) -> list[float]:
    if not route:
        return [0.0, 0.0, 0.0]
    if len(route) == 1:
        return list(route[0])
    remaining = max(0.0, float(speed_mps) * (float(tick_value) / SCRIPT_TICK_HZ))
    current = list(route[0])
    for target in route[1:]:
        segment_len = dist_xy(current, target)
        if segment_len <= 1e-6:
            current = list(target)
            continue
        if remaining <= segment_len:
            alpha = remaining / segment_len
            return [
                current[0] + (target[0] - current[0]) * alpha,
                current[1] + (target[1] - current[1]) * alpha,
                current[2] + (target[2] - current[2]) * alpha,
            ]
        remaining -= segment_len
        current = list(target)
    return list(route[-1])


def route_duration_ticks(route: list[list[float]], speed_mps: float) -> int:
    if len(route) < 2:
        return 0
    return int(math.ceil(path_length_m(route) / max(0.1, float(speed_mps)) * SCRIPT_TICK_HZ))


def explicit_pedestrian_vehicle_collision_allowed(
    scenario_id: str,
    events: list[dict[str, Any]],
    issues: list[str] | None = None,
) -> bool:
    allowed = False
    for event in events:
        event_type = str(validation_event_field(event, "validation_event_type") or "")
        skip_checks = validation_skip_checks(event)
        if event_type not in PEDESTRIAN_VEHICLE_ALLOWED_EVENT_TYPES and PEDESTRIAN_VEHICLE_CLEARANCE_CHECK not in skip_checks:
            continue
        event_id = event.get("event_id") or event.get("source_event_id") or "<unknown>"
        if event_type not in PEDESTRIAN_VEHICLE_ALLOWED_EVENT_TYPES:
            if issues is not None:
                issues.append(
                    f"{scenario_id}: event {event_id} may skip {PEDESTRIAN_VEHICLE_CLEARANCE_CHECK} only with "
                    f"one of validation_event_type={sorted(PEDESTRIAN_VEHICLE_ALLOWED_EVENT_TYPES)!r}"
                )
            continue
        if PEDESTRIAN_VEHICLE_CLEARANCE_CHECK not in skip_checks:
            if issues is not None:
                issues.append(f"{scenario_id}: event {event_id} lacks {PEDESTRIAN_VEHICLE_CLEARANCE_CHECK} skip check")
            continue
        if not str(validation_event_field(event, "validation_reason") or "").strip():
            if issues is not None:
                issues.append(f"{scenario_id}: event {event_id} skips {PEDESTRIAN_VEHICLE_CLEARANCE_CHECK} without validation_reason")
            continue
        allowed = True
    return allowed


def check_scene_pedestrian_vehicle_dynamic_clearance(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    scenario_id = str(scene.get("scenario_id") or script.get("scenario_id") or "")
    if explicit_pedestrian_vehicle_collision_allowed(scenario_id, list(script.get("events") or []), issues):
        return
    pedestrians = [
        entity
        for entity in scene.get("entities") or []
        if str(entity.get("logical_asset_id") or "").startswith("pedestrian.")
    ]
    vehicles = [
        entity
        for entity in scene.get("entities") or []
        if str(entity.get("logical_asset_id") or "").startswith("vehicle.")
    ]
    worst: tuple[float, str, str, int] | None = None
    for pedestrian in pedestrians:
        ped_route = scene_route_points(pedestrian)
        if len(ped_route) < 2:
            continue
        ped_speed = float((pedestrian.get("ground_flow_contract") or {}).get("speed_mps") or 1.25)
        ped_duration = route_duration_ticks(ped_route, ped_speed)
        for vehicle in vehicles:
            vehicle_route = scene_route_points(vehicle)
            if len(vehicle_route) < 2:
                continue
            vehicle_speed = float((vehicle.get("ground_flow_contract") or {}).get("speed_mps") or 6.0)
            max_tick = min(ped_duration, route_duration_ticks(vehicle_route, vehicle_speed))
            for tick_value in range(0, max_tick + 1):
                distance = dist_xy(
                    route_position_at_tick(ped_route, ped_speed, tick_value),
                    route_position_at_tick(vehicle_route, vehicle_speed, tick_value),
                )
                if worst is None or distance < worst[0]:
                    worst = (distance, str(pedestrian.get("entity_id") or ""), str(vehicle.get("entity_id") or ""), tick_value)
                if distance < PEDESTRIAN_VEHICLE_DYNAMIC_CLEARANCE_MIN_M:
                    issues.append(
                        f"{scenario_id}: pedestrian/vehicle dynamic clearance too small at tick {tick_value}: "
                        f"{pedestrian.get('entity_id')} vs {vehicle.get('entity_id')} "
                        f"{distance:.2f}m < {PEDESTRIAN_VEHICLE_DYNAMIC_CLEARANCE_MIN_M:.2f}m"
                    )
                    return


def _background_vehicle_clearance_radius(entity: dict[str, Any]) -> float:
    asset_id = str(entity.get("logical_asset_id", ""))
    category = str(entity.get("category", ""))
    placement = dict(entity.get("placement") or {})
    mode = str(entity.get("placement_mode") or "")
    if mode == "box_volume":
        center = pos3(placement.get("center_enu_m") or placement.get("resolved_position_enu_m"))
        extent = pos3(placement.get("extent_m"))
        if center and extent and center[2] - extent[2] <= 8.0:
            return max(extent[0], extent[1]) + 5.0
        return 0.0
    if asset_id.startswith("vehicle."):
        return 5.3
    if asset_id.startswith("pedestrian."):
        return 5.8
    if asset_id.startswith("uav."):
        return 11.0
    if asset_id == "facility.landing_pad.visible.v1":
        return 13.0
    if asset_id.startswith("prop.roadwork."):
        return 6.0
    if asset_id == "semantic.spawn_zone" or category == "crowd_anchor":
        return 14.0
    if asset_id.startswith("trigger.") or category in {"hazard_zone", "airspace_constraint"}:
        return 10.0
    return 0.0


def check_background_vehicle_clearance_rule(
    scene: dict[str, Any],
    script: dict[str, Any],
    rule: dict[str, Any],
    issues: list[str],
) -> bool:
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    requested_ids = [str(entity_id) for entity_id in rule.get("entity_ids", []) if str(entity_id)]
    if not requested_ids:
        requested_ids = [
            str(entity["entity_id"])
            for entity in scene.get("entities", [])
            if str(entity.get("entity_id") or "").startswith("bg_vehicle_")
        ]
    min_count = int(rule.get("min_count", 0) or 0)
    ok = True
    if len(requested_ids) < min_count:
        issues.append(f"{script['scenario_id']}: background vehicle count {len(requested_ids)} < {min_count}")
        ok = False

    event_points: list[list[float]] = []
    for _event, action in action_iter(script):
        if action.get("type") == "move_entity":
            for point in action.get("waypoints_enu_m", []) or []:
                parsed = pos3(point)
                if parsed and parsed[2] <= 12.0:
                    event_points.append(parsed)
        else:
            parsed = pos3(action.get("position_enu_m") or action.get("spawn_origin_enu_m"))
            if parsed and parsed[2] <= 12.0:
                event_points.append(parsed)

    for entity_id in requested_ids:
        target = entities.get(entity_id)
        if not target:
            issues.append(f"{script['scenario_id']}: missing background vehicle {entity_id}")
            ok = False
            continue
        asset_id = str(target.get("logical_asset_id", ""))
        if not asset_id.startswith("vehicle."):
            issues.append(f"{script['scenario_id']}: background vehicle {entity_id} is not a vehicle asset")
            ok = False
        if target.get("placement_mode") != "lane_anchor":
            issues.append(f"{script['scenario_id']}: background vehicle {entity_id} is not lane anchored")
            ok = False
        target_pos = entity_pos(target)
        if not target_pos:
            issues.append(f"{script['scenario_id']}: background vehicle {entity_id} lacks resolved position")
            ok = False
            continue
        for other_id, other in entities.items():
            if other_id == entity_id:
                continue
            other_pos = entity_pos(other)
            if not other_pos:
                continue
            radius = _background_vehicle_clearance_radius(other)
            if radius <= 0.0:
                continue
            distance = dist_xy(target_pos, other_pos)
            if distance < radius:
                issues.append(
                    f"{script['scenario_id']}: background vehicle {entity_id} is too close to {other_id}: {distance:.2f}m < {radius:.2f}m"
                )
                ok = False
                break
        for point in event_points:
            distance = dist_xy(target_pos, point)
            if distance < 4.0:
                issues.append(
                    f"{script['scenario_id']}: background vehicle {entity_id} is too close to low event action point: {distance:.2f}m < 4.00m"
                )
                ok = False
                break
    return ok


def capture_boundary_payload(scene: dict[str, Any], script: dict[str, Any]) -> dict[str, Any]:
    params = dict(script.get("parameters") or {})
    boundary = dict(params.get("capture_boundary") or {})
    if boundary:
        return boundary
    contract = dict(params.get("semantic_event_contract") or {})
    return dict(contract.get("capture_boundary") or {})


def boundary_polygon(scene: dict[str, Any], script: dict[str, Any]) -> list[list[float]]:
    boundary = capture_boundary_payload(scene, script)
    polygon = boundary.get("polygon_enu_m") or []
    result: list[list[float]] = []
    for point in polygon:
        if isinstance(point, list) and len(point) >= 2:
            result.append([float(point[0]), float(point[1])])
    if result:
        return result
    source_id = str(boundary.get("source_entity_id") or "")
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    source = entities.get(source_id)
    if source:
        placement = dict(source.get("placement") or {})
        for point in placement.get("polygon_enu_m") or []:
            if isinstance(point, list) and len(point) >= 2:
                result.append([float(point[0]), float(point[1])])
        if result:
            return result
        center = entity_pos(source)
        extent = placement.get("extent_m") or placement.get("size_m")
        if center and isinstance(extent, list) and len(extent) >= 2:
            half_x = float(extent[0]) * (0.5 if "size_m" in placement and "extent_m" not in placement else 1.0)
            half_y = float(extent[1]) * (0.5 if "size_m" in placement and "extent_m" not in placement else 1.0)
            return [
                [center[0] - half_x, center[1] - half_y],
                [center[0] + half_x, center[1] - half_y],
                [center[0] + half_x, center[1] + half_y],
                [center[0] - half_x, center[1] + half_y],
            ]
    return []


def boundary_center(boundary: dict[str, Any], polygon: list[list[float]]) -> list[float] | None:
    center = pos3(boundary.get("center_enu_m"))
    if center:
        return center
    if polygon:
        return [sum(point[0] for point in polygon) / len(polygon), sum(point[1] for point in polygon) / len(polygon), 0.0]
    return None


def boundary_samples(boundary: dict[str, Any], polygon: list[list[float]]) -> list[list[float]]:
    samples: list[list[float]] = [[point[0], point[1], 0.0] for point in polygon]
    if polygon:
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            samples.append([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, 0.0])
    center = boundary_center(boundary, polygon)
    if center:
        samples.append(center)
    for key in ("key_conflict_point_enu_m", "terminal_landing_point_enu_m", "forced_landing_point_enu_m"):
        point = pos3(boundary.get(key))
        if point:
            samples.append(point)
    unique: list[list[float]] = []
    seen: set[tuple[float, float, float]] = set()
    for sample in samples:
        key = (round(sample[0], 3), round(sample[1], 3), round(sample[2], 3))
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def route_crosses_boundary(route: list[list[float]], polygon: list[list[float]]) -> bool:
    if not route or not polygon:
        return False
    if any(point_in_polygon_xy(point, polygon) for point in route):
        return True
    return any(segment_intersects_polygon_xy(a, b, polygon) for a, b in zip(route, route[1:]))


def entity_route_points(entity: dict[str, Any], script: dict[str, Any]) -> list[list[float]]:
    points: list[list[float]] = []
    start = entity_mission_start_pos(entity) or entity_pos(entity)
    if start:
        points.append(start)
    for waypoint in entity.get("route_waypoints_enu_m") or []:
        found = pos3(waypoint)
        if found:
            points.append(found)
    entity_id = str(entity.get("entity_id") or "")
    for _, action in action_iter(script):
        if action.get("type") != "move_entity" or str(action.get("entity_id") or action.get("ped_id") or "") != entity_id:
            continue
        for waypoint in action.get("waypoints_enu_m") or []:
            found = pos3(waypoint)
            if found:
                points.append(found)
    return points


def inspect_loop_route(entity: dict[str, Any]) -> list[list[float]]:
    contract = dict(entity.get("contract_inspect_uav") or {})
    raw = contract.get("loop_route_enu_m") or contract.get("repaired_route_enu_m") or contract.get("planned_route_enu_m") or entity.get("route_waypoints_enu_m") or []
    return [point for point in (pos3(item) for item in raw) if point]


def sensor_fov_deg(contract: dict[str, Any]) -> float:
    profile = dict(contract.get("sensor_profile") or {})
    for source in (contract, profile):
        for key in ("hfov_deg", "FOV_Degrees", "sensor_fov_deg"):
            if key in source:
                return float(source.get(key) or 0.0)
    return 0.0


def sensor_profile(contract: dict[str, Any]) -> dict[str, Any]:
    profile = dict(contract.get("sensor_profile") or {})
    if not profile:
        return {}
    return profile


def inspect_route_covers_boundary(route: list[list[float]], samples: list[list[float]], profile: dict[str, Any], altitude_m: float) -> bool:
    if not route or not samples or altitude_m <= 0.0:
        return False
    hfov_deg = float(profile.get("hfov_deg") or profile.get("FOV_Degrees") or profile.get("fov_degrees") or 0.0)
    width = float(profile.get("width") or 0.0)
    height = float(profile.get("height") or 0.0)
    rotation = dict(profile.get("fixed_rotation_offset_deg") or {})
    pitch_deg = float(rotation.get("pitch_deg", 0.0))
    if hfov_deg <= 0.0 or width <= 0.0 or height <= 0.0 or pitch_deg > -60.0:
        return False
    half_width_m = math.tan(math.radians(hfov_deg / 2.0)) * altitude_m
    vfov_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov_deg / 2.0)) * (height / width)))
    half_height_m = math.tan(math.radians(vfov_deg / 2.0)) * altitude_m
    segments = list(zip(route, route[1:]))
    return all(
        any(point_in_oriented_frustum_footprint_xy(sample, a, b, half_width_m, half_height_m) for a, b in segments)
        for sample in samples
    )


def visible_physical_entity(entity: dict[str, Any]) -> bool:
    if entity.get("enabled") is False:
        return False
    asset_id = str(entity.get("logical_asset_id") or "")
    category = str(entity.get("category") or "")
    if asset_id == UAV_CORRIDOR_LOGICAL_ASSET_ID:
        return False
    if asset_id.startswith(("uav.", "vehicle.", "pedestrian.", "facility.", "prop.")):
        return True
    return category in {"uav", "vehicle", "pedestrian", "facility", "traffic_signal", "airspace_constraint", "hazard_zone"}


def locomotion_entity(entity: dict[str, Any]) -> bool:
    asset_id = str(entity.get("logical_asset_id") or "")
    category = str(entity.get("category") or "")
    return asset_id.startswith(("uav.", "vehicle.", "pedestrian.")) or category in {"uav", "vehicle", "pedestrian"}


def entity_motion_displacement(entity: dict[str, Any], script: dict[str, Any]) -> float:
    points = entity_route_points(entity, script)
    if len(points) >= 2:
        return path_length_m(points)
    return 0.0


def non_locomotion_has_animation(entity: dict[str, Any], script: dict[str, Any]) -> bool:
    entity_id = str(entity.get("entity_id") or "")
    motion = dict(entity.get("motion_contract") or {})
    if str(motion.get("motion_kind") or "") in {"transform_animation", "state_animation"}:
        return True
    for _, action in action_iter(script):
        if str(action.get("entity_id") or action.get("ped_id") or "") != entity_id:
            continue
        if action.get("type") in {"animate_entity_transform", "set_facility_state", "set_logical_boundary_state", "set_visual_state", "play_animation"}:
            return True
    return False


def check_all_entities_motion_contract(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    sid = str(scene.get("scenario_id") or script.get("scenario_id") or "")
    for entity in scene.get("entities", []):
        if not visible_physical_entity(entity):
            continue
        entity_id = str(entity.get("entity_id") or "")
        motion = dict(entity.get("motion_contract") or {})
        if not motion:
            issues.append(f"{sid}: visible entity missing motion_contract: {entity_id}")
            continue
        if motion.get("required") is not True:
            issues.append(f"{sid}: visible entity motion_contract.required must be true: {entity_id}")
        if str(motion.get("motion_kind") or "") not in {"kinematic_path", "transform_animation", "state_animation"}:
            issues.append(f"{sid}: visible entity motion_contract.motion_kind invalid: {entity_id}")
        if motion.get("semantic_link_required") is not True:
            issues.append(f"{sid}: visible entity motion_contract.semantic_link_required must be true: {entity_id}")
        if not motion.get("linked_intents"):
            issues.append(f"{sid}: visible entity motion_contract.linked_intents missing: {entity_id}")
        if locomotion_entity(entity) and str(motion.get("motion_kind") or "") == "kinematic_path":
            min_displacement = float(motion.get("min_displacement_m") if "min_displacement_m" in motion else 0.5)
            displacement = entity_motion_displacement(entity, script)
            if displacement < min_displacement:
                issues.append(f"{sid}: locomotion entity displacement too small: {entity_id} {displacement:.2f}m < {min_displacement:.2f}m")
        elif not non_locomotion_has_animation(entity, script):
            issues.append(f"{sid}: non-locomotion visible entity lacks transform/state/material animation: {entity_id}")


def check_ground_flow_contracts(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    sid = str(scene.get("scenario_id") or script.get("scenario_id") or "")
    for entity in scene.get("entities", []):
        state = dict(entity.get("initial_state") or {})
        role = str(state.get("role") or entity.get("role") or "")
        if role not in {"semantic_background_vehicle", "semantic_background_pedestrian"}:
            continue
        entity_id = str(entity.get("entity_id") or "")
        contract = dict(entity.get("ground_flow_contract") or {})
        if not contract:
            issues.append(f"{sid}: background ground-flow entity missing ground_flow_contract: {entity_id}")
            continue
        if str(contract.get("policy") or "") != "continuous_capture_ground_flow_v1":
            issues.append(f"{sid}: background ground-flow entity has invalid policy: {entity_id}")
        loop_policy = str(contract.get("loop_policy") or "")
        if loop_policy in FORBIDDEN_GROUND_FLOW_LOOP_POLICIES:
            issues.append(f"{sid}: background ground-flow entity uses forbidden loop_policy={loop_policy}: {entity_id}")


def check_capture_boundary_contract(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    sid = str(scene.get("scenario_id") or script.get("scenario_id") or "")
    contract = get_contract(sid)
    boundary = capture_boundary_payload(scene, script)
    polygon = boundary_polygon(scene, script)
    if not boundary:
        issues.append(f"{sid}: missing capture_boundary contract")
        return
    if not polygon:
        issues.append(f"{sid}: capture_boundary lacks polygon_enu_m")
        return
    if str(boundary.get("geometry_source") or "") == "local_bounds":
        issues.append(f"{sid}: capture_boundary must not use local_bounds as geometry_source")
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    uavs = [entity for entity in entities.values() if str(entity.get("logical_asset_id") or "").startswith("uav.")]
    if contract.uav_boundary_crossing_required:
        for uav in uavs:
            if is_inspect_uav(uav):
                continue
            if not route_crosses_boundary(entity_route_points(uav, script), polygon):
                issues.append(f"{sid}: UAV route does not cross capture boundary: {uav['entity_id']}")
    inspect_entities = [uav for uav in uavs if is_inspect_uav(uav)]
    if len(inspect_entities) == 1:
        inspect = inspect_entities[0]
        loop_route = inspect_loop_route(inspect)
        if not loop_route:
            issues.append(f"{sid}: U_inspect lacks loop route for capture boundary observation")
        else:
            altitudes = {round(float(point[2]), 3) for point in loop_route}
            contract_altitude = float(contract.inspect_altitude_m)
            if len(altitudes) != 1 or abs(next(iter(altitudes)) - contract_altitude) > 0.001:
                issues.append(f"{sid}: U_inspect loop route is not fixed at {contract_altitude:.1f}m")
            if not route_is_closed(loop_route):
                issues.append(f"{sid}: U_inspect loop route must be closed")
            inspect_contract = dict(inspect.get("contract_inspect_uav") or {})
            profile = sensor_profile(inspect_contract)
            if contract.inspect.fov_coverage_required:
                if not profile:
                    issues.append(f"{sid}: U_inspect FoV is required but missing contract_inspect_uav.sensor_profile")
                elif not inspect_route_covers_boundary(loop_route, boundary_samples(boundary, polygon), profile, contract_altitude):
                    issues.append(f"{sid}: U_inspect sensor-profile frustum does not cover capture boundary samples")
    pad_policy = dict(dict(script.get("parameters") or {}).get("semantic_event_contract") or {}).get("pad_boundary_policy")
    if not isinstance(pad_policy, dict):
        pad_policy = {}
    inside_required_for = set(as_tuple_str(pad_policy.get("inside_required_for"))) or set(contract.pad_policy.inside_required_for)
    if inside_required_for:
        pad_inside_required = inside_required_for.intersection({"pad_contention", "priority_landing_arbitration", "terminal_pad_queue"})
        if pad_inside_required:
            pads = [
                entity
                for entity in entities.values()
                if str(entity.get("logical_asset_id") or "") == "facility.landing_pad.visible.v1"
                and not str(entity.get("entity_id") or "").startswith("pad_home_")
            ]
            if not pads:
                issues.append(f"{sid}: pad boundary policy requires event pads but none are declared")
            for pad in pads:
                position = entity_pos(pad)
                if position and not point_in_polygon_xy(position, polygon):
                    issues.append(f"{sid}: event pad is outside capture boundary: {pad['entity_id']}")
        if inside_required_for.intersection({"conflict_point", "terminal_landing_zone"}):
            relevant_points: list[list[float]] = []
            for event_def, action in action_iter(script):
                action_parts = [
                    action.get("action_id"),
                    action.get("type"),
                    action.get("intent"),
                    action.get("intent_stage"),
                    action.get("entity_id"),
                    action.get("target_entity_id"),
                ]
                text = event_field_haystack(event_def) + " " + " ".join(str(part or "").lower() for part in action_parts)
                if not any(
                    token in text
                    for token in (
                        "conflict",
                        "collision",
                        "near",
                        "proximity",
                        "convergence",
                        "brake",
                        "forced",
                        "emergency",
                        "landing",
                        "descent",
                        "touchdown",
                    )
                ):
                    continue
                for waypoint in action.get("waypoints_enu_m") or []:
                    point = pos3(waypoint)
                    if point:
                        relevant_points.append(point)
            if not any(point_in_polygon_xy(point, polygon) for point in relevant_points):
                issues.append(f"{sid}: conflict/terminal event point is not inside capture boundary")


def check_evacuation_pedestrian_spacing(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    scenario_id = str(script.get("scenario_id") or scene.get("scenario_id") or "")
    if not (scenario_id.startswith("L4-8") or scenario_id.startswith("X6")):
        return
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    routes: list[tuple[str, list[list[float]]]] = []
    for _event, action in action_iter(script):
        if action.get("type") != "move_entity":
            continue
        entity_id = str(action.get("entity_id") or "")
        entity = entities.get(entity_id)
        if not entity or not str(entity.get("logical_asset_id", "")).startswith("pedestrian."):
            continue
        waypoints = [pos3(item) for item in action.get("waypoints_enu_m", [])]
        waypoints = [item for item in waypoints if item]
        if len(waypoints) >= 2:
            routes.append((entity_id, waypoints))
    if len(routes) < 2:
        return
    min_allowed_m = INITIAL_OVERLAP_MIN_M[("pedestrian", "pedestrian")]
    for step in range(21):
        fraction = step / 20.0
        samples = [(entity_id, point_at_path_fraction(route, fraction)) for entity_id, route in routes]
        samples = [(entity_id, point) for entity_id, point in samples if point is not None]
        for index, (id_a, point_a) in enumerate(samples):
            for id_b, point_b in samples[index + 1 :]:
                distance = dist_xy(point_a, point_b)
                if distance < min_allowed_m:
                    issues.append(
                        f"{scenario_id}: evacuation pedestrian routes overlap at fraction={fraction:.2f}: "
                        f"{id_a} vs {id_b}: {distance:.2f}m < {min_allowed_m:.2f}m"
                    )
                    return


def check_lifecycle_closure(scene: dict[str, Any], script: dict[str, Any], issues: list[str]) -> None:
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    trigger_by_id = {trigger["trigger_id"]: trigger for trigger in script.get("triggers", [])}
    mobile_ids = {
        entity_id
        for entity_id, entity in entities.items()
        if str(entity.get("logical_asset_id", "")).startswith(("uav.", "vehicle.", "pedestrian."))
    }
    move_actions_by_entity: dict[str, list[dict[str, Any]]] = {}
    for event, action in action_iter(script):
        action_type = action.get("type")
        entity_id = action.get("entity_id")
        if action_type == "move_entity" and entity_id in entities:
            move_actions_by_entity.setdefault(entity_id, []).append(action)
        if action_type == "spawn_entity" and entity_id in mobile_ids:
            issues.append(f"{script['scenario_id']}: mobile entity {entity_id} uses mid-script spawn_entity; declare it at scene start instead")
        if action_type == "remove_entity" and entity_id in mobile_ids:
            issues.append(f"{script['scenario_id']}: mobile entity {entity_id} uses remove_entity and would disappear from camera")
        if action_type == "clear_crowd":
            issues.append(f"{script['scenario_id']}: clear_crowd {action.get('group_id')} would make visible pedestrians disappear during capture")
        if action_type == "spawn_crowd":
            issues.append(f"{script['scenario_id']}: Dataset scenarios must use explicit pedestrian cohort entities, not spawn_crowd {action.get('group_id')}")
            trigger = trigger_by_id.get(event.get("trigger_ref"), {})
            tick_value = trigger.get("tick", -1)
            if trigger.get("type") != "tick" or int(tick_value if tick_value is not None else -1) != 0:
                issues.append(f"{script['scenario_id']}: spawn_crowd {action.get('group_id')} is not initialized at tick 0")
            if str(script["scenario_id"]).startswith(("L4-8", "X6")):
                issues.append(f"{script['scenario_id']}: evacuation scenes must use explicit movable pedestrians, not static spawn_crowd {action.get('group_id')}")

    for entity_id, entity in entities.items():
        asset_id = str(entity.get("logical_asset_id", ""))
        if not asset_id.startswith("uav."):
            continue
        if is_inspect_uav(entity) or is_observer_uav(entity):
            continue
        start = entity_pos(entity)
        lifecycle = entity.get("lifecycle") or {}
        corridor_lifecycle = bool(lifecycle.get("corridor_lifecycle") or entity.get("uav_corridor"))
        pad_id = lifecycle.get("home_pad_entity_id")
        home_hover = pos3(lifecycle.get("home_hover_enu_m"))
        mission_start = pos3(lifecycle.get("mission_start_enu_m"))
        if not start:
            issues.append(f"{script['scenario_id']}: UAV {entity_id} lacks resolved initial position")
            continue
        if start[2] > UAV_INITIAL_ALTITUDE_MAX_M and not corridor_lifecycle:
            issues.append(f"{script['scenario_id']}: UAV {entity_id} starts at {start[2]:.1f}m; expected visible low-altitude pad/preflight start")
        if (not pad_id or pad_id not in entities) and not corridor_lifecycle:
            issues.append(f"{script['scenario_id']}: UAV {entity_id} lacks declared lifecycle home pad")
        pad_pos = entity_pos(entities[pad_id]) if pad_id in entities else None
        if pad_pos and not home_hover:
            home_hover = [pad_pos[0], pad_pos[1], 3.0]
        if pad_pos and home_hover and dist_xy(home_hover, pad_pos) > 1.5:
            issues.append(
                f"{script['scenario_id']}: UAV {entity_id} lifecycle home_hover is not above home pad "
                f"{pad_id}: {dist_xy(home_hover, pad_pos):.2f}m"
            )
        if pad_pos and dist_xy(start, pad_pos) > 1.5 and not corridor_lifecycle:
            issues.append(
                f"{script['scenario_id']}: UAV {entity_id} initial position is not above home pad "
                f"{pad_id}: {dist_xy(start, pad_pos):.2f}m"
            )
        if not mission_start or mission_start[2] < UAV_MISSION_ALTITUDE_MIN_M:
            issues.append(f"{script['scenario_id']}: UAV {entity_id} lacks high-altitude lifecycle mission_start_enu_m")

        moves = move_actions_by_entity.get(entity_id, [])
        if not moves:
            issues.append(f"{script['scenario_id']}: UAV {entity_id} has no move_entity lifecycle")
            continue
        first = moves[0]
        first_waypoints = [pos3(point) for point in first.get("waypoints_enu_m", [])]
        first_waypoints = [point for point in first_waypoints if point]
        if "takeoff" not in str(first.get("action_id", "")).lower():
            issues.append(f"{script['scenario_id']}: first UAV move for {entity_id} is not an explicit takeoff action")
        if first_waypoints:
            if dist3(first_waypoints[0], start) > MOVE_START_TOLERANCE_UAV_M:
                issues.append(f"{script['scenario_id']}: UAV {entity_id} takeoff does not start at declared pad/preflight position")
            if pad_pos and dist_xy(first_waypoints[0], pad_pos) > 1.5:
                issues.append(
                    f"{script['scenario_id']}: UAV {entity_id} takeoff first waypoint is not above home pad "
                    f"{pad_id}: {dist_xy(first_waypoints[0], pad_pos):.2f}m"
                )
            if first_waypoints[0][2] > UAV_TERMINAL_ALTITUDE_MAX_M:
                issues.append(f"{script['scenario_id']}: UAV {entity_id} takeoff first waypoint starts too high at {first_waypoints[0][2]:.1f}m")
            if max(point[2] for point in first_waypoints) < UAV_MISSION_ALTITUDE_MIN_M:
                issues.append(f"{script['scenario_id']}: UAV {entity_id} takeoff never reaches mission altitude")

        last = moves[-1]
        last_waypoints = [pos3(point) for point in last.get("waypoints_enu_m", [])]
        last_waypoints = [point for point in last_waypoints if point]
        if last_waypoints:
            final_z = last_waypoints[-1][2]
            terminal_action = str(last.get("action_id", "")).lower()
            terminal_by_name = any(token in terminal_action for token in ("landing", "touchdown", "debris", "crash"))
            if final_z > UAV_TERMINAL_ALTITUDE_MAX_M and not terminal_by_name and not corridor_lifecycle:
                issues.append(f"{script['scenario_id']}: UAV {entity_id} ends at {final_z:.1f}m without landing/crash terminal action")
            if pad_pos and "landing_return" in terminal_action:
                if dist_xy(last_waypoints[-1], pad_pos) > 1.5:
                    issues.append(
                        f"{script['scenario_id']}: UAV {entity_id} landing_return does not finish above home pad "
                        f"{pad_id}: {dist_xy(last_waypoints[-1], pad_pos):.2f}m"
                    )
                if final_z > UAV_TERMINAL_ALTITUDE_MAX_M:
                    issues.append(f"{script['scenario_id']}: UAV {entity_id} landing_return finishes too high at {final_z:.1f}m")


def execute_validation_rules(scene: dict[str, Any], script: dict[str, Any], known_assets: set[str], issues: list[str]) -> None:
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    events = script.get("events", [])
    actions = [action for _, action in action_iter(script)]

    def original_or_repaired_waypoints(action: dict[str, Any]) -> list[list[float]]:
        validation = dict(action.get("uav_corridor_validation") or {})
        raw_points = validation.get("original_waypoints_enu_m") or action.get("waypoints_enu_m") or []
        return [point for point in (pos3(item) for item in raw_points) if point]

    def forced_descent_profile_ok(action: dict[str, Any]) -> bool:
        points = original_or_repaired_waypoints(action)
        return bool(points) and min(point[2] for point in points) <= 3.5 and max(point[2] for point in points) >= 29.0

    for rule in scene.get("validation_rules", []):
        name = rule.get("rule")
        ok = True
        if name == "entity_resolvable":
            ok = rule.get("entity_id") in entities
        elif name == "asset_in_catalog":
            ok = rule.get("logical_asset_id") in known_assets
        elif name == "event_chain_min":
            ok = len(events) >= int(rule.get("min_count", 0))
        elif name == "cross_layer_event_chain_min":
            ok = len(events) >= int(rule.get("min_count", 5))
        elif name == "uav_start_outside_constraint":
            constraints = [entity for entity in entities.values() if entity.get("logical_asset_id", "").startswith("trigger.")]
            uavs = [entity for entity in entities.values() if entity.get("logical_asset_id", "").startswith("uav.") and is_mission_uav(entity)]
            for uav in uavs:
                uav_pos = entity_mission_start_pos(uav)
                for constraint in constraints:
                    placement = constraint.get("placement", {})
                    center = entity_pos(constraint)
                    extent = placement.get("extent_m") or [0.0, 0.0, 0.0]
                    if uav_pos and center and all(abs(uav_pos[i] - center[i]) <= float(extent[i]) / 2.0 for i in range(3)):
                        ok = False
        elif name == "roadwork_asset_count":
            counts = {"barrier": 0, "fence": 0, "cone": 0}
            for entity in entities.values():
                asset = entity.get("logical_asset_id", "")
                counts["barrier"] += int(asset == "prop.roadwork.barrier.v1")
                counts["fence"] += int(asset == "prop.roadwork.construction_fence.v1")
                counts["cone"] += int(asset == "prop.roadwork.traffic_cone.v1")
            ok = counts["barrier"] >= int(rule.get("barrier_min", 0)) and counts["fence"] >= int(rule.get("fence_min", 0)) and counts["cone"] >= int(rule.get("cone_min", 0))
        elif name == "lane_anchor_required":
            prefix = rule.get("asset_prefix", "")
            ok = all(entity.get("placement_mode") == "lane_anchor" for entity in entities.values() if entity.get("logical_asset_id", "").startswith(prefix))
        elif name == "dynamic_spawn_required":
            target = rule.get("entity_id")
            ok = any(action.get("type") == "spawn_entity" and action.get("entity_id") == target for action in actions)
        elif name == "hazmat_min_entities":
            counts = {
                "pedestrian": sum(1 for entity in entities.values() if entity.get("logical_asset_id", "").startswith("pedestrian.")),
                "ambulance": sum(1 for entity in entities.values() if entity.get("logical_asset_id") == "vehicle.emergency.ambulance.v1"),
                "uav": sum(1 for entity in entities.values() if entity.get("logical_asset_id", "").startswith("uav.")),
            }
            ok = counts["pedestrian"] >= int(rule.get("pedestrian_min", 0)) and counts["ambulance"] >= int(rule.get("ambulance_min", 0)) and counts["uav"] >= int(rule.get("uav_min", 0))
        elif name == "uav_start_separation":
            uav_positions = [
                entity_mission_start_pos(entity)
                for entity in entities.values()
                if entity.get("logical_asset_id", "").startswith("uav.") and is_mission_uav(entity)
            ]
            uav_positions = [item for item in uav_positions if item]
            ok = len(uav_positions) >= 2 and dist_xy(uav_positions[0], uav_positions[1]) >= float(rule.get("min_horizontal_distance_m", 0.0)) and abs(uav_positions[0][2] - uav_positions[1][2]) > 0.5
        elif name == "facade_approach_distance":
            facade_positions = [entity_pos(entity) for entity in entities.values() if entity.get("category") == "facade_anchor"]
            uav_waypoints = [
                point
                for action in actions
                if action.get("type") == "move_entity" and str(action.get("entity_id", "")).startswith("uav_")
                for point in original_or_repaired_waypoints(action)
            ]
            ok = bool(facade_positions and uav_waypoints) and min(dist_xy(point, facade_positions[0]) for point in uav_waypoints) <= float(rule.get("max_distance_m", 3.0)) + 2.0
        elif name == "forced_landing_descent_profile":
            uav_moves = [action for action in actions if action.get("type") == "move_entity" and "forced_descent" in action.get("action_id", "")]
            ok = any(forced_descent_profile_ok(action) for action in uav_moves)
        elif name == "spatial_grid_assignment":
            assignment = dict((script.get("parameters") or {}).get("spatial_grid_assignment") or {})
            event_space_class = str(rule.get("event_space_class") or "")
            max_error_m = 80.0 if event_space_class == "traffic_incident_grid" else 140.0
            ok = (
                bool(assignment)
                and str(rule.get("policy") or "") == str(assignment.get("policy") or "")
                and event_space_class == str(assignment.get("event_space_class") or "")
                and abs(float(rule.get("target_error_m") or 0.0) - float(assignment.get("target_error_m") or 0.0)) <= 1e-3
                and float(assignment.get("target_error_m") or 0.0) <= max_error_m
            )
        elif name == "trajectory_intersection_required":
            uav_points = [
                point
                for action in actions
                if action.get("type") == "move_entity" and str(action.get("entity_id", "")).startswith("uav_")
                for point in original_or_repaired_waypoints(action)
            ]
            vehicle_points = [
                point
                for action in actions
                if action.get("type") == "move_entity" and str(action.get("entity_id", "")).startswith("car_")
                for point in original_or_repaired_waypoints(action)
            ]
            ok = any(up and vp and dist_xy(up, vp) <= 4.0 for up in uav_points for vp in vehicle_points)
        elif name == "crowd_spawn_count_min":
            ok = any(action.get("type") == "spawn_crowd" and int(action.get("count", 0)) >= int(rule.get("min_count", 0)) for action in actions)
        elif name == "explicit_pedestrian_count_min":
            ok = sum(1 for entity in entities.values() if str(entity.get("logical_asset_id", "")).startswith("pedestrian.")) >= int(rule.get("min_count", 0))
        elif name == "ambulance_lights_on":
            target = entities.get(rule.get("entity_id"))
            ok = bool(target and target.get("initial_state", {}).get("lights_on") is True)
        elif name == "environment_trigger_kind":
            sid = script.get("scenario_id", "")
            has_weather = any(trigger.get("type") == "weather_state" for trigger in script.get("triggers", []))
            ok = has_weather if sid.startswith(("L5-1", "L5-2", "L5-3")) else not has_weather
        elif name == "digital_anomaly_chain":
            ok = any(action.get("type") == "move_entity" and str(action.get("entity_id", "")).startswith("uav_") for action in actions)
        elif name == "uav_corridor_contract":
            params = dict(script.get("parameters") or {})
            target_uav_count = int(rule.get("target_uav_count") or 0)
            expected_segment_count = int(rule.get("corridor_segment_count") or -1)
            expected_materialized_count = int(rule.get("materialized_corridor_scene_count") or -1)
            actual_uav_count = sum(
                1 for entity in entities.values() if str(entity.get("logical_asset_id") or "").startswith("uav.")
            )
            actual_segment_count = len(params.get("uav_corridor_segment_details") or [])
            actual_materialized_count = sum(
                1 for entity in entities.values() if str(entity.get("logical_asset_id") or "") == UAV_CORRIDOR_LOGICAL_ASSET_ID
            )
            ok = (
                target_uav_count > 0
                and actual_uav_count == target_uav_count
                and actual_segment_count == expected_segment_count
                and actual_materialized_count == expected_materialized_count
            )
        elif name == "semantic_event_contract":
            ok = dict(rule.get("contract") or {}) == dict(dict(script.get("parameters") or {}).get("semantic_event_contract") or {})
        elif name == "uav_boundary_crossing_required":
            polygon = boundary_polygon(scene, script)
            ok = bool(polygon) and all(
                route_crosses_boundary(entity_route_points(entity, script), polygon)
                for entity in entities.values()
                if str(entity.get("logical_asset_id") or "").startswith("uav.") and not is_inspect_uav(entity)
            )
        elif name == "pad_inside_boundary_required":
            polygon = boundary_polygon(scene, script)
            ok = False
            if polygon:
                for entity in entities.values():
                    if str(entity.get("logical_asset_id") or "") != "facility.landing_pad.visible.v1":
                        continue
                    if str(entity.get("entity_id") or "").startswith("pad_home_"):
                        continue
                    position = entity_pos(entity)
                    if position and point_in_polygon_xy(position, polygon):
                        ok = True
                        break
        elif name == "background_vehicle_clearance":
            ok = check_background_vehicle_clearance_rule(scene, script, rule, issues)
        else:
            ok = False
        if not ok:
            issues.append(f"{script['scenario_id']}: validation_rule failed or unsupported: {name}")


def check_render_config_flags(config_path: Path, issues: list[str]) -> None:
    if not config_path.exists():
        issues.append(f"{config_path}: missing render config")
        return
    config = load_json(config_path)
    for section, key in RENDER_CONFIG_DISABLED_FLAGS:
        payload = dict(config.get(section) or {})
        if payload.get(key) is not False:
            issues.append(f"{config_path}: Dataset render config must set {section}.{key}=false")
    road_topology = dict(config.get("road_topology_snap") or {})
    if road_topology.get("enabled") is True and road_topology.get("preserve_truth_xy") is not True:
        issues.append(f"{config_path}: road_topology_snap must preserve truth XY when enabled")


def check_render_ready_configs(render_ready_root: Path, issues: list[str]) -> None:
    if not render_ready_root.exists():
        return
    for config_path in sorted(render_ready_root.rglob("render_host_config.json")):
        check_render_config_flags(config_path, issues)


def _is_render_ready_source_contract_entity(entity: dict[str, Any]) -> bool:
    entity_id = str(entity.get("entity_id") or "")
    source = str(entity.get("source") or "")
    background_role = str(entity.get("background_role") or "")
    tags = {str(tag) for tag in entity.get("tags") or []}
    if entity_id.startswith(("sumo_vehicle_", "global_uav_", "global_pad_", "pad_home_")):
        return False
    if source in {"sumo_traci", "uav_global_flow"}:
        return False
    if background_role in {"sumo_background_traffic", "donghu_global_uav_flow", "global_uav_pad"}:
        return False
    if {"sumo_traci", "uav_global_flow", "background_traffic"} & tags:
        return False
    if entity.get("sumo_vehicle") or entity.get("uav_global_flow") or entity.get("uav_global_pad"):
        return False
    return True


def _render_ready_entity_counts(episode_dir: Path) -> dict[str, int]:
    roster_path = episode_dir / "global_entity_roster.json"
    if not roster_path.exists():
        return {}
    roster = load_json(roster_path)
    counts = {"uav": 0, "vehicle": 0, "pedestrian": 0, "facility": 0, "logical": 0}
    for entity in roster.get("entities") or []:
        if not _is_render_ready_source_contract_entity(entity):
            continue
        entity_id = str(entity.get("entity_id") or "")
        category = str(entity.get("entity_category") or "")
        asset_id = str(entity.get("logical_asset_id") or "")
        role = str(entity.get("role") or entity.get("background_role") or "")
        kind = str(entity.get("entity_kind") or "")
        if entity_id.startswith("pad_home_"):
            continue
        if (
            asset_id == UAV_CORRIDOR_LOGICAL_ASSET_ID
            or asset_id.startswith(("trigger.", "semantic.trigger_box."))
            or asset_id == "semantic.spawn_zone"
            or category in {"airspace_constraint", "hazard_zone", "crowd_anchor", "airspace_corridor"}
            or kind.startswith(("airspace_corridor.", "facility.no_fly_zone", "facility.hazmat_proxy"))
            or role == "semantic_logical_sidecar"
        ):
            counts["logical"] += 1
            continue
        if category == "uav":
            counts["uav"] += 1
        elif category == "vehicle":
            counts["vehicle"] += 1
        elif category == "pedestrian":
            counts["pedestrian"] += 1
        elif category in {"facility", "traffic_light"} or kind.startswith(("facility.", "traffic_light.")):
            counts["facility"] += 1
    return counts


def ground_flow_direction_reversals(points: list[list[float]]) -> int:
    reversals = 0
    previous_unit: tuple[float, float] | None = None
    for a, b in zip(points, points[1:]):
        dx = float(b[0]) - float(a[0])
        dy = float(b[1]) - float(a[1])
        norm = math.hypot(dx, dy)
        if norm <= 0.05:
            continue
        unit = (dx / norm, dy / norm)
        if previous_unit is not None:
            dot = previous_unit[0] * unit[0] + previous_unit[1] * unit[1]
            if dot < -0.95:
                reversals += 1
        previous_unit = unit
    return reversals


def check_render_ready_ground_flow(episode_name: str, truths: list[dict[str, Any]], issues: list[str]) -> None:
    stats: dict[str, dict[str, Any]] = {}
    for frame in truths:
        for entity in frame.get("entities") or []:
            category = str(entity.get("entity_category") or "")
            if category not in {"pedestrian", "vehicle"}:
                continue
            role = str(entity.get("role") or "")
            if role not in {"semantic_background_pedestrian", "semantic_background_vehicle"}:
                continue
            entity_id = str(entity.get("entity_id") or "")
            if not entity_id:
                continue
            render_presence = dict(entity.get("render_presence") or {})
            visible = render_presence.get("visibility_state") == "visible" and render_presence.get("offstage") is not True
            row = stats.setdefault(
                entity_id,
                {
                    "category": category,
                    "visible": 0,
                    "moving": 0,
                    "tick0_moving": False,
                    "xs": [],
                    "ys": [],
                    "ticks": [],
                    "positions": [],
                },
            )
            if not visible:
                continue
            row["visible"] += 1
            velocity = ((entity.get("truth_pose") or {}).get("velocity_enu_mps") or [0.0, 0.0, 0.0])
            speed = math.hypot(
                float(velocity[0] if len(velocity) > 0 else 0.0),
                float(velocity[1] if len(velocity) > 1 else 0.0),
            )
            if speed > GROUND_FLOW_SPEED_EPS_MPS:
                row["moving"] += 1
                if int(frame.get("tick") or frame.get("frame_seq") or 0) == 0:
                    row["tick0_moving"] = True
            position = ((entity.get("truth_pose") or {}).get("position_enu_m") or [])
            if len(position) >= 2:
                row["ticks"].append(int(frame.get("tick") or frame.get("frame_seq") or 0))
                row["xs"].append(float(position[0]))
                row["ys"].append(float(position[1]))
                row["positions"].append([float(position[0]), float(position[1]), float(position[2] if len(position) > 2 else 0.0)])
    for entity_id, row in sorted(stats.items()):
        visible = int(row["visible"])
        if visible <= 0:
            issues.append(f"{episode_name}: background ground-flow actor is never visible: {entity_id}")
            continue
        positions = list(row["positions"])
        ticks = list(row["ticks"])
        max_speed = RENDER_READY_MAX_SPEED_MPS[str(row["category"])]
        for prev_tick, tick, prev_pos, pos in zip(ticks, ticks[1:], positions, positions[1:]):
            dt_s = max(1e-6, (int(tick) - int(prev_tick)) / SCRIPT_TICK_HZ)
            speed_mps = dist_xy(prev_pos, pos) / dt_s
            if speed_mps > max_speed:
                issues.append(
                    f"{episode_name}: background {row['category']} {entity_id} implied speed too high between "
                    f"tick {prev_tick} and {tick}: {speed_mps:.2f}m/s > {max_speed:.2f}m/s"
                )
                break
        xs = list(row["xs"])
        ys = list(row["ys"])
        xy_span = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else 0.0
        reversals = ground_flow_direction_reversals(positions)
        if reversals > GROUND_FLOW_PINGPONG_REVERSAL_LIMIT and xy_span <= GROUND_FLOW_PINGPONG_SPAN_M[str(row["category"])]:
            issues.append(
                f"{episode_name}: background {row['category']} {entity_id} appears to ping-pong in a short span: "
                f"{reversals} reversals over {xy_span:.3f}m"
            )


def check_render_ready_pedestrian_vehicle_clearance(
    episode_name: str,
    truths: list[dict[str, Any]],
    events: list[dict[str, Any]],
    issues: list[str],
) -> None:
    if explicit_pedestrian_vehicle_collision_allowed(episode_name, events, issues):
        return
    worst: tuple[float, int, str, str] | None = None
    for frame in truths:
        tick = int(frame.get("tick") or frame.get("frame_seq") or 0)
        pedestrians: list[tuple[str, list[float]]] = []
        vehicles: list[tuple[str, list[float]]] = []
        for entity in frame.get("entities") or []:
            category = str(entity.get("entity_category") or "")
            if category not in {"pedestrian", "vehicle"}:
                continue
            pose = dict(entity.get("truth_pose") or {})
            position = pos3(pose.get("position_enu_m"))
            if not position:
                continue
            row = (str(entity.get("entity_id") or ""), position)
            if category == "pedestrian":
                pedestrians.append(row)
            else:
                vehicles.append(row)
        for ped_id, ped_pos in pedestrians:
            for vehicle_id, vehicle_pos in vehicles:
                distance = dist_xy(ped_pos, vehicle_pos)
                if worst is None or distance < worst[0]:
                    worst = (distance, tick, ped_id, vehicle_id)
    if worst is not None and worst[0] < PEDESTRIAN_VEHICLE_DYNAMIC_CLEARANCE_MIN_M:
        issues.append(
            f"{episode_name}: render-ready pedestrian/vehicle dynamic clearance too small at tick {worst[1]}: "
            f"{worst[2]} vs {worst[3]} {worst[0]:.2f}m < {PEDESTRIAN_VEHICLE_DYNAMIC_CLEARANCE_MIN_M:.2f}m"
        )


def check_render_ready_contract(render_ready_root: Path, issues: list[str]) -> None:
    if not render_ready_root.exists():
        return
    for episode_dir in sorted(path for path in render_ready_root.iterdir() if path.is_dir()):
        scenario_plan_path = episode_dir / "scenario_plan.json"
        truth_frames_path = episode_dir / "truth_frames.jsonl"
        event_trace_path = episode_dir / "event_trace.jsonl"
        dynamic_labels_path = episode_dir / "dynamic_labels.jsonl"
        roster_path = episode_dir / "global_entity_roster.json"
        manifest_path = episode_dir / "episode_manifest.json"
        if not scenario_plan_path.exists() or not truth_frames_path.exists() or not event_trace_path.exists() or not dynamic_labels_path.exists() or not roster_path.exists() or not manifest_path.exists():
            issues.append(f"{episode_dir}: missing render-ready artifacts")
            continue
        scenario_plan = load_json(scenario_plan_path)
        manifest = load_json(manifest_path)
        scenario_id = str(scenario_plan.get("scenario_id") or scenario_plan.get("episode_id") or episode_dir.name)
        if "__seed" in scenario_id:
            scenario_id = scenario_id.split("__seed", 1)[0]
        contract = get_contract(scenario_id)
        generation = dict(manifest.get("generation") or {})
        if "generated_at" in manifest or "generated_at" in generation:
            issues.append(f"{episode_dir.name}: render-ready manifest must not contain generated_at")
        if not generation.get("source_contract_hash"):
            issues.append(f"{episode_dir.name}: render-ready manifest missing generation.source_contract_hash")
        for key in ("capture_boundary_id", "uav_crosses_boundary", "inspect_observes_boundary", "pad_boundary_policy"):
            if key not in scenario_plan:
                issues.append(f"{episode_dir.name}: scenario_plan missing {key}")
        if scenario_plan.get("uav_crosses_boundary") is not True:
            issues.append(f"{episode_dir.name}: scenario_plan.uav_crosses_boundary must be true")
        if scenario_plan.get("inspect_observes_boundary") is not True:
            issues.append(f"{episode_dir.name}: scenario_plan.inspect_observes_boundary must be true")
        roster = load_json(roster_path).get("entities") or []
        counts = _render_ready_entity_counts(episode_dir)
        for key, expected in contract.counts.items():
            actual = counts.get(key, 0)
            if key in {"pedestrian", "vehicle", "uav"}:
                if actual > expected:
                    issues.append(f"{episode_dir.name}: render-ready {key} count exceeds source contract: expected <= {expected}, got {actual}")
            elif actual != expected:
                issues.append(f"{episode_dir.name}: render-ready {key} count mismatch: expected {expected}, got {actual}")
        inspect_entities = [entity for entity in roster if str(entity.get("role") or "") == "U_inspect" or str((entity.get("task_id") or "")).endswith(".u_inspect")]
        if len(inspect_entities) != 1:
            issues.append(f"{episode_dir.name}: render-ready must contain exactly one U_inspect")
        else:
            inspect = inspect_entities[0]
            if str(inspect.get("inspect_altitude_code") or "") != contract.inspect_code:
                issues.append(f"{episode_dir.name}: render-ready U_inspect altitude code mismatch")
            if float(inspect.get("min_path_length_m") or 0.0) < 80.0:
                issues.append(f"{episode_dir.name}: render-ready U_inspect min path length too short")
        events = load_jsonl(event_trace_path)
        truths = load_jsonl(truth_frames_path)
        if not truths:
            issues.append(f"{episode_dir.name}: empty truth_frames")
        else:
            check_render_ready_ground_flow(episode_dir.name, truths, issues)
            check_render_ready_pedestrian_vehicle_clearance(episode_dir.name, truths, events, issues)
            for frame in truths[:3]:
                for key in ("capture_boundary_id", "uav_crosses_boundary", "inspect_observes_boundary", "pad_boundary_policy"):
                    if key not in frame:
                        issues.append(f"{episode_dir.name}: truth_frame missing {key}")
                if frame.get("capture_boundary_id") != scenario_plan.get("capture_boundary_id"):
                    issues.append(f"{episode_dir.name}: truth_frame capture_boundary_id mismatch")
                if frame.get("uav_crosses_boundary") is not True:
                    issues.append(f"{episode_dir.name}: truth_frame.uav_crosses_boundary must be true")
                if frame.get("inspect_observes_boundary") is not True:
                    issues.append(f"{episode_dir.name}: truth_frame.inspect_observes_boundary must be true")
                if frame.get("pad_boundary_policy") != scenario_plan.get("pad_boundary_policy"):
                    issues.append(f"{episode_dir.name}: truth_frame pad_boundary_policy mismatch")
                frame_counts = dict((frame.get("roster_summary") or {}).get("by_category") or {})
                for key, expected in contract.counts.items():
                    if int(frame_counts.get(key, 0) or 0) < 0:
                        issues.append(f"{episode_dir.name}: invalid truth_frame count payload for {key}")
                        break
        labels = load_jsonl(dynamic_labels_path)
        event_pairs = [(str(row.get("topic") or row.get("source_event_id") or row.get("event_id") or ""), int(row.get("tick", 0) or 0)) for row in events]
        label_pairs = [(str(row.get("topic") or row.get("source_event_id") or row.get("event_id") or ""), int(row.get("tick", 0) or 0)) for row in labels]
        if event_pairs != label_pairs:
            issues.append(f"{episode_dir.name}: event_trace/dynamic_labels mismatch")
        ok, matched = required_intent_sequence_matches(contract.required_intents, events)
        if not ok:
            issues.append(f"{episode_dir.name}: render-ready required_intents order mismatch: {contract.required_intents} (matched={matched})")


def building_index(spatial: MapSpatialIndex) -> BuildingObstacleIndex:
    global _BUILDING_INDEX_CACHE
    if _BUILDING_INDEX_CACHE is None:
        _BUILDING_INDEX_CACHE = BuildingObstacleIndex(spatial)
    return _BUILDING_INDEX_CACHE


def corridor_defs(scene: dict[str, Any]) -> list[dict[str, Any]]:
    corridors = []
    for entity in scene.get("entities", []):
        if str(entity.get("logical_asset_id") or "") != UAV_CORRIDOR_LOGICAL_ASSET_ID:
            continue
        placement = dict(entity.get("placement") or {})
        center = pos3(placement.get("center_enu_m") or placement.get("resolved_position_enu_m"))
        size = pos3(placement.get("size_m") or placement.get("scale_xyz"))
        if not center or not size:
            extent = pos3(placement.get("extent_m"))
            if center and extent:
                size = [extent[0] * 2.0, extent[1] * 2.0, extent[2] * 2.0]
        if not center or not size:
            continue
        rotation = dict(placement.get("rotation_deg") or {})
        corridors.append(
            {
                "entity_id": entity.get("entity_id"),
                "center": center,
                "size": size,
                "yaw": float(rotation.get("yaw_deg", rotation.get("yaw", 0.0))),
            }
        )
    return corridors


def corridor_defs_from_contract(script: dict[str, Any]) -> list[dict[str, Any]]:
    corridors = []
    params = dict(script.get("parameters") or {})
    for item in params.get("uav_corridor_segment_details") or []:
        if not isinstance(item, dict):
            continue
        a = pos3(item.get("segment_start_enu_m"))
        b = pos3(item.get("segment_end_enu_m"))
        if not a or not b:
            continue
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]
        length_m = max(1.0, math.sqrt(dx * dx + dy * dy + dz * dz))
        corridors.append(
            {
                "entity_id": f"{item.get('entity_id', 'uav')}:contract:{item.get('segment_index', 0)}",
                "center": [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5],
                "size": [length_m, 8.0, max(8.0, abs(dz) + 4.0)],
                "yaw": math.degrees(math.atan2(dy, dx)),
            }
        )
    return corridors


def point_in_corridor(point: list[float], corridor: dict[str, Any], tolerance_m: float = 1.5) -> bool:
    center = corridor["center"]
    size = corridor["size"]
    yaw = math.radians(float(corridor["yaw"]))
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
    local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return (
        abs(local_x) <= size[0] * 0.5 + tolerance_m
        and abs(local_y) <= size[1] * 0.5 + tolerance_m
        and abs(point[2] - center[2]) <= size[2] * 0.5 + tolerance_m
    )


def check_uav_corridor_contract(scene: dict[str, Any], script: dict[str, Any], spatial: MapSpatialIndex, issues: list[str]) -> None:
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    uavs = [entity for entity in entities.values() if str(entity.get("logical_asset_id") or "").startswith("uav.")]
    corridors = corridor_defs(scene) + corridor_defs_from_contract(script)
    sid = str(scene.get("scenario_id") or script.get("scenario_id") or "")
    params = dict(script.get("parameters") or {})
    allow_building_impact = bool(params.get("allow_building_impact", False))
    if not uavs:
        issues.append(f"{sid}: uav_corridor_contract failed: scene has no UAV")
    if not corridors:
        issues.append(f"{sid}: uav_corridor_contract failed: scene has no semantic UAV corridor")

    bindex = building_index(spatial)
    for uav in uavs:
        if is_inspect_uav(uav):
            continue
        pos = entity_mission_start_pos(uav)
        if pos and not allow_building_impact and bindex.point_collision(pos):
            issues.append(f"{sid}: UAV mission/start point intersects building volume: {uav['entity_id']} {pos}")
        if is_mission_uav(uav) and pos and pos[2] >= UAV_MISSION_ALTITUDE_MIN_M and corridors and not any(point_in_corridor(pos, corridor) for corridor in corridors):
            issues.append(f"{sid}: UAV mission/start point is outside high-altitude corridors: {uav['entity_id']} {pos}")

    starts = [(entity["entity_id"], entity_mission_start_pos(entity)) for entity in uavs if is_mission_uav(entity)]
    starts = [(entity_id, pos) for entity_id, pos in starts if pos]
    for index, (a_id, a_pos) in enumerate(starts):
        for b_id, b_pos in starts[index + 1:]:
            if dist_xy(a_pos, b_pos) < 8.0 and abs(a_pos[2] - b_pos[2]) < 6.0:
                issues.append(f"{sid}: UAV corridor slot separation failed: {a_id} {a_pos} vs {b_id} {b_pos}")

    asset_by_id = {entity_id: str(entity.get("logical_asset_id") or "") for entity_id, entity in entities.items()}
    for _, action in action_iter(script):
        if action.get("type") != "move_entity":
            continue
        entity_id = str(action.get("entity_id") or action.get("ped_id") or "")
        if not asset_by_id.get(entity_id, "").startswith("uav."):
            continue
        entity = entities.get(entity_id)
        terminal_or_low_profile = uav_action_is_terminal_or_low_profile(action.get("action_id"))
        points = [pos3(point) for point in action.get("waypoints_enu_m") or []]
        points = [point for point in points if point]
        for point in points:
            if not allow_building_impact and bindex.point_collision(point):
                issues.append(f"{sid}: UAV waypoint intersects building volume in {action.get('action_id')}: {entity_id} {point}")
            if (
                entity
                and is_mission_uav(entity)
                and not terminal_or_low_profile
                and point[2] >= UAV_MISSION_ALTITUDE_MIN_M
                and corridors
                and not any(point_in_corridor(point, corridor) for corridor in corridors)
            ):
                issues.append(f"{sid}: UAV high waypoint outside corridor in {action.get('action_id')}: {entity_id} {point}")
        for a, b in zip(points, points[1:]):
            if not allow_building_impact and bindex.segment_collision(a, b):
                issues.append(f"{sid}: UAV route segment intersects building volume in {action.get('action_id')}: {entity_id} {a}->{b}")


def validate_scenario(
    scene_path: Path,
    lanes: LaneResolver,
    spatial: MapSpatialIndex,
    known_assets: set[str],
    known_buildings: set[str],
) -> list[str]:
    scene = load_json(scene_path)
    script_path = scene_path.with_name("event_script.json")
    if not script_path.exists():
        return [f"{scene_path.parent}: missing event_script.json"]
    script = load_json(script_path)
    issues: list[str] = []
    check_contract_payload(scene, script, issues)
    check_event_intent_contract(scene, script, issues)
    check_scene_contract_counts(scene, script, issues)
    check_entity_references(scene, script, issues)
    check_assets(scene, known_assets, issues)
    check_dynamic_spawn_policy(scene, script, issues)
    check_spawn_schema(scene, script, known_assets, issues)
    check_placements(scene, lanes, spatial, known_buildings, issues)
    check_initial_mobile_overlaps(scene, issues)
    check_scene_pedestrian_vehicle_dynamic_clearance(scene, script, issues)
    check_cameras(scene, lanes, issues)
    check_event_positions(scene, script, lanes, spatial, issues)
    check_event_delay_physics(script, issues)
    check_evacuation_pedestrian_spacing(scene, script, issues)
    check_local_bounds(scene, script, issues)
    check_weather_bootstrap(scene, script, issues)
    check_proximity_metrics(scene, script, issues)
    check_activity_causality(scene, script, issues)
    check_lifecycle_closure(scene, script, issues)
    check_capture_boundary_contract(scene, script, issues)
    check_all_entities_motion_contract(scene, script, issues)
    check_ground_flow_contracts(scene, script, issues)
    check_uav_corridor_contract(scene, script, spatial, issues)
    execute_validation_rules(scene, script, known_assets, issues)
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate grounded scene_setup/event_script pairs")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--asset-catalog", default=str(DEFAULT_ASSET_CATALOG))
    parser.add_argument("--traffic-bundle", default=str(DEFAULT_TRAFFIC_BUNDLE))
    parser.add_argument("--building-geojson", default=str(DEFAULT_BUILDING_GEOJSON))
    parser.add_argument("--render-ready-root", default=str(DEFAULT_RENDER_READY_ROOT))
    parser.add_argument("--render-host-config", default=str(DEFAULT_RENDER_HOST_CONFIG))
    parser.add_argument("--skip-render-ready-configs", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N scenario files for debugging")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    scenarios_root = dataset_root / "scenarios"
    traffic_bundle = Path(args.traffic_bundle).resolve()
    lanes = LaneResolver(traffic_bundle / "lane_center_samples.csv")
    spatial = MapSpatialIndex.default(ROOT)
    known_assets = asset_ids(Path(args.asset_catalog).resolve())
    known_buildings = building_ids(Path(args.building_geojson).resolve())

    scene_paths = sorted(scenarios_root.rglob("scene_setup.json"))
    if args.limit:
        scene_paths = scene_paths[: args.limit]
    all_issues: list[str] = []
    all_issues.extend(f"pedestrian activity catalog: {issue}" for issue in validate_local_animation_assets(ROOT))
    for scene_path in scene_paths:
        all_issues.extend(validate_scenario(scene_path, lanes, spatial, known_assets, known_buildings))
    check_render_config_flags(Path(args.render_host_config).resolve(), all_issues)
    if not args.skip_render_ready_configs:
        check_render_ready_configs(Path(args.render_ready_root).resolve(), all_issues)
        check_render_ready_contract(Path(args.render_ready_root).resolve(), all_issues)

    print("=" * 72)
    print("Scene Grounding Validation")
    print("=" * 72)
    print(f"Scenarios checked: {len(scene_paths)}")
    print(f"Issues: {len(all_issues)}")
    if all_issues:
        for issue in all_issues[:200]:
            print(f"  - {issue}")
        if len(all_issues) > 200:
            print(f"  ... {len(all_issues) - 200} additional issues")
    else:
        print("All scene grounding checks PASSED.")
    sys.exit(1 if all_issues else 0)


if __name__ == "__main__":
    main()
