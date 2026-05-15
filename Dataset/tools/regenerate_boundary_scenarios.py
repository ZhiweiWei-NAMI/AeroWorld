"""Regenerate semantic P0 scenario specs and compiled event scripts.

This generator replaces the old placeholder specs with concrete
SpecCompiler ScenarioSpec definitions.  It also emits scene_setup.json files
because the boundary document requires semantic placement, asset resolution,
and validation metadata that event_script_v1 does not carry.
"""

from __future__ import annotations

import argparse
import json
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec_compiler import (  # noqa: E402
    ActionSpec,
    EntitySpec,
    EventStepSpec,
    ScenarioSpec,
    SpecCompiler,
    TriggerSpec,
    WaypointSpec,
)
from map_spatial_index import (  # noqa: E402
    MapSpatialIndex,
    PEDESTRIAN_ROAD_BUFFER_M,
    PlannedSidewalkAnchor,
    SpatialValidationError,
)
from pedestrian_activity_catalog import get_activity, normalize_activity_type  # noqa: E402
from uav_corridor_planner import (  # noqa: E402
    HighAltitudeCorridorPlanner,
    UAV_ALTITUDE_LAYERS_M,
    UAV_CORRIDOR_LOGICAL_ASSET_ID,
)
from semantic_event_contract import (  # noqa: E402
    EpisodeContract,
    canonical_intent_from_text,
    contract_payload,
    get_contract,
)
from sumo_ground_flow import (  # noqa: E402
    SpatialAssignment,
    SpatialEventGridPlanner,
    SumoGroundFlowPlanner,
    SumoRouteError,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "Dataset"
SCENARIOS_ROOT = DATASET_ROOT / "scenarios"
ASSET_CATALOG = ROOT / "Config" / "LowAltitude" / "asset_catalog.json"
LANE_SAMPLES_CSV = (
    ROOT
    / "Config"
    / "LowAltitude"
    / "Maps"
    / "donghu_road_topo"
    / "traffic_bundle"
    / "lane_center_samples.csv"
)
SUMO_NET_XML = ROOT / "Plugins" / "SumoImporter" / "Maps" / "donghu_road_topo" / "source" / "map.net.xml"
BUILDING_GEOJSON = ROOT / "Content" / "Maps" / "donghu_road_topo" / "building" / "building.geojson"
CAPTURE_PRESETS = ROOT / "Plugins" / "SumoImporter" / "Scripts" / "episode_capture_presets.json"
WORLD_OFFSET_X_M = 7000.0
WORLD_OFFSET_Y_M = 6200.0
CURRENT_WORLD_OFFSET_X_M = WORLD_OFFSET_X_M
CURRENT_WORLD_OFFSET_Y_M = WORLD_OFFSET_Y_M
LANE_HALF_WIDTH_M = 1.9
SIDEWALK_MIN_OFFSET_FROM_CURB_M = 1.2
GATHERING_MIN_OFFSET_FROM_CURB_M = 4.5
ROADWORK_MIN_OFFSET_FROM_EDGE_M = 1.5
LANDING_PAD_MIN_OFFSET_FROM_CURB_M = GATHERING_MIN_OFFSET_FROM_CURB_M
GROUND_Z_M = 0.0
UAV_PAD_HOVER_Z_M = 3.0
UAV_TAKEOFF_ENTRY_TICK = 30
UAV_LANDING_AFTER_SCENARIO_DELAY_TICKS = 80
UAV_LANDING_ALTITUDE_MAX_M = 8.0
SCRIPT_TICK_HZ = 10.0
DELAY_SAFETY_FACTOR = 1.1
UAV_ASSET_CYCLE = (
    "uav.inspect.quad.v1",
    "uav.airsim.flying_pawn.v1",
    "uav.airsim.cv_pawn.v1",
)

MAP_REF = {
    "map_id": "donghu_road_topo",
    "geo_reference": {"lat": 30.5609, "lon": 114.3627, "alt": 24.0},
    "coordinate_frame": "ENU",
}

ALLOWED_ASSETS = {
    "uav.inspect.quad.v1",
    "uav.airsim.flying_pawn.v1",
    "uav.airsim.cv_pawn.v1",
    "vehicle.ground.boxcar.v1",
    "vehicle.emergency.suv.v1",
    "vehicle.emergency.police_suv.v1",
    "vehicle.emergency.ambulance.v1",
    "vehicle.service.box.v1",
    "pedestrian.cityops.basic.v1",
    "prop.roadwork.barrier.v1",
    "prop.roadwork.construction_fence.v1",
    "prop.roadwork.traffic_cone.v1",
    "prop.traffic_control.police_sign.v1",
    "prop.traffic_control.signal_light.v1",
    "prop.incident.police_tape.v1",
    "prop.service.delivery_bag.v1",
    "prop.service.backpack.v1",
    "prop.misc.phone.v1",
    "prop.misc.umbrella.v1",
    "facility.landing_pad.visible.v1",
    "facility.radio.base_tower.v1",
    "facility.charger.cityops.v1",
    "facility.barrier.basic",
    "trigger.no_fly.box.v1",
    "trigger.hazard.construction.box.v1",
    "trigger.hazard.generic.box.v1",
    "semantic.landing_pad",
    "semantic.spawn_zone",
    "semantic.asset_anchor",
    UAV_CORRIDOR_LOGICAL_ASSET_ID,
}

BACKGROUND_VEHICLE_ASSETS = (
    "vehicle.ground.boxcar.v1",
    "vehicle.emergency.suv.v1",
    "vehicle.service.box.v1",
)
BACKGROUND_VEHICLE_FLOW_SPEED_MPS = 6.0
BACKGROUND_PEDESTRIAN_FLOW_SPEED_MPS = 1.25
GROUND_FLOW_MIN_VISIBLE_MOTION_RATIO = 0.85
GROUND_FLOW_ROUTE_DURATION_TICKS = 900
BACKGROUND_VEHICLE_MIN_ROUTE_SPAN_M = 24.0
BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M = 9.0
SUMO_TO_TRAFFIC_BUNDLE_MAX_SNAP_M = 15.0
FORCED_LANDING_DESCENT_SPEED_MPS = 8.0
FORCED_LANDING_TOUCHDOWN_SPEED_MPS = 2.0


def default_uav_sensor_fov_deg() -> float:
    presets = json.loads(CAPTURE_PRESETS.read_text(encoding="utf-8-sig"))
    cameras = list((presets.get("uav_cameras") or {}).get("default") or [])
    if not cameras:
        raise RuntimeError(f"No default UAV camera preset found in {CAPTURE_PRESETS}")
    fov = float(dict(cameras[0]).get("fov_degrees") or 0.0)
    if fov <= 0.0:
        raise RuntimeError(f"Default UAV camera preset lacks positive fov_degrees in {CAPTURE_PRESETS}")
    return fov


def default_uav_sensor_profile() -> dict[str, Any]:
    presets = json.loads(CAPTURE_PRESETS.read_text(encoding="utf-8-sig"))
    cameras = list((presets.get("uav_cameras") or {}).get("default") or [])
    if not cameras:
        raise RuntimeError(f"No default UAV camera preset found in {CAPTURE_PRESETS}")
    profile = dict(cameras[0])
    fov = float(profile.get("fov_degrees") or 0.0)
    width = int(profile.get("width") or 0)
    height = int(profile.get("height") or 0)
    rotation = dict(profile.get("fixed_rotation_offset_deg") or {})
    if fov <= 0.0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Default UAV camera preset lacks fov/width/height in {CAPTURE_PRESETS}")
    if float(rotation.get("pitch_deg", 0.0)) > -60.0:
        raise RuntimeError(f"Default UAV camera preset must be downward looking for inspect coverage: {CAPTURE_PRESETS}")
    return {
        "source": "Plugins/SumoImporter/Scripts/episode_capture_presets.json:uav_cameras.default[0]",
        "camera_name": str(profile.get("camera_name") or ""),
        "capture_backend": str(profile.get("capture_backend") or ""),
        "width": width,
        "height": height,
        "hfov_deg": fov,
        "fixed_rotation_offset_deg": rotation,
    }


DEFAULT_UAV_SENSOR_FOV_DEG = default_uav_sensor_fov_deg()
DEFAULT_UAV_SENSOR_PROFILE = default_uav_sensor_profile()

SEMANTIC_STATIC_STATES = {
    "queued",
    "waiting",
    "waiting_at_crosswalk",
    "blocked",
    "blocked_by_barrier",
    "medical_incident",
    "bystander",
    "observing",
    "shelter_wait",
}


@dataclass
class ScenarioBundle:
    scenario_id: str
    directory: Path
    category: str
    description: str
    spec_entities: list[dict[str, Any]]
    scene_entities: list[dict[str, Any]]
    parameters: dict[str, Any]
    events: list[dict[str, Any]]
    weather_profile: dict[str, Any] | None = None
    cameras: list[dict[str, Any]] | None = None
    validation_rules: list[dict[str, Any]] | None = None
    duration_ticks: int = 900


@dataclass(frozen=True)
class LaneSample:
    edge_id: str
    lane_index: int
    s_m: float
    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float


@dataclass(frozen=True)
class BackgroundVehiclePolicy:
    add_count: int
    min_uav_count: int = 0
    reason: str = ""


BACKGROUND_VEHICLE_POLICY_BY_PREFIX: tuple[tuple[str, BackgroundVehiclePolicy], ...] = (
    ("L2-2_", BackgroundVehiclePolicy(1, reason="urban canyon road context")),
    ("L2-5_", BackgroundVehiclePolicy(2, reason="traffic-control queue context")),
    ("L3-1_", BackgroundVehiclePolicy(1, reason="roadwork detour traffic context")),
    ("L3-3_", BackgroundVehiclePolicy(0, min_uav_count=3, reason="hazmat evacuation road context")),
    ("L4-3_", BackgroundVehiclePolicy(2, reason="forced landing near crowd road context")),
    ("L4-5_", BackgroundVehiclePolicy(2, reason="pedestrian near-miss road context")),
    ("L4-6_", BackgroundVehiclePolicy(1, reason="jaywalk conflict background traffic")),
    ("L4-7_", BackgroundVehiclePolicy(1, reason="emergency response background traffic")),
    ("L4-8_", BackgroundVehiclePolicy(2, reason="crowd evacuation perimeter traffic")),
    ("L4-9_", BackgroundVehiclePolicy(1, reason="intersection traffic context")),
    ("L4-10_", BackgroundVehiclePolicy(1, reason="ambulance priority background traffic")),
    ("L4-11_", BackgroundVehiclePolicy(1, reason="safe-stop lane context")),
    ("L5-", BackgroundVehiclePolicy(1, reason="weather road traffic context")),
    ("X1_", BackgroundVehiclePolicy(2, reason="rain forced-landing crowd road context")),
    ("X3_", BackgroundVehiclePolicy(1, reason="emergency response road context")),
    ("X6_", BackgroundVehiclePolicy(2, min_uav_count=3, reason="crowd evacuation lockdown road context")),
)


class LaneSampleResolver:
    def __init__(self, path: Path) -> None:
        self.samples: list[LaneSample] = []
        self.by_edge: dict[str, list[LaneSample]] = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"edge_id", "lane_index", "s_m", "x_m", "y_m", "yaw_deg"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} is not a lane center sample CSV; missing columns {sorted(missing)}. "
                    "Use traffic_bundle/lane_center_samples.csv, not lane_meta.csv."
                )
            for row in reader:
                sample = LaneSample(
                    edge_id=str(row["edge_id"]),
                    lane_index=int(row["lane_index"]),
                    s_m=float(row["s_m"]),
                    x_m=float(row["x_m"]),
                    y_m=float(row["y_m"]),
                    z_m=float(row.get("z_m") or 0.0),
                    yaw_deg=float(row.get("yaw_deg") or 0.0),
                )
                self.samples.append(sample)
                self.by_edge.setdefault(sample.edge_id, []).append(sample)
        for edge_samples in self.by_edge.values():
            edge_samples.sort(key=lambda item: item.s_m)

    def nearest_to_xy(self, x_m: float, y_m: float) -> LaneSample:
        return min(self.samples, key=lambda item: (item.x_m - x_m) ** 2 + (item.y_m - y_m) ** 2)

    def resolve_edge_s(self, edge_id: str, s_m: float) -> LaneSample:
        edge_samples = self.by_edge.get(edge_id)
        if not edge_samples:
            raise ValueError(f"Unknown lane edge_id: {edge_id}")
        return min(edge_samples, key=lambda item: abs(item.s_m - s_m))


class BuildingCatalog:
    def __init__(self, path: Path) -> None:
        self.path = path
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        ids: list[str] = []
        for feature in data.get("features", []):
            props = feature.get("properties") or {}
            source_id = props.get("id_origin") or props.get("id")
            if source_id is not None:
                ids.append(f"building_geojson_{source_id}")
        if not ids:
            raise ValueError(f"No building IDs found in {path}")
        self.ids = ids

    def id_for(self, index: int) -> str:
        return self.ids[index % len(self.ids)]

    @property
    def source_ref(self) -> str:
        return str(self.path.relative_to(ROOT)).replace("\\", "/")


SPATIAL = MapSpatialIndex.default(ROOT)
LANES = SPATIAL.lanes
SUMO_GROUND_FLOW = SumoGroundFlowPlanner(
    SUMO_NET_XML,
    max_start_snap_m=SUMO_TO_TRAFFIC_BUNDLE_MAX_SNAP_M,
)
BUILDINGS = BuildingCatalog(BUILDING_GEOJSON)
UAV_CORRIDORS = HighAltitudeCorridorPlanner(SPATIAL)


def p(x: float, y: float, z: float = 0.0) -> list[float]:
    return [round(x + CURRENT_WORLD_OFFSET_X_M, 3), round(y + CURRENT_WORLD_OFFSET_Y_M, 3), round(z, 3)]


def q(x: float, y: float, z: float = 0.0) -> list[float]:
    return [round(x, 3), round(y, 3), round(z, 3)]


def reset_world_event_origin() -> None:
    global CURRENT_WORLD_OFFSET_X_M, CURRENT_WORLD_OFFSET_Y_M
    CURRENT_WORLD_OFFSET_X_M = WORLD_OFFSET_X_M
    CURRENT_WORLD_OFFSET_Y_M = WORLD_OFFSET_Y_M


def set_world_event_origin(x_m: float, y_m: float) -> None:
    global CURRENT_WORLD_OFFSET_X_M, CURRENT_WORLD_OFFSET_Y_M
    CURRENT_WORLD_OFFSET_X_M = float(x_m)
    CURRENT_WORLD_OFFSET_Y_M = float(y_m)


def dist_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _lane_normal(sample: LaneSample) -> tuple[float, float]:
    yaw_rad = math.radians(sample.yaw_deg)
    return -math.sin(yaw_rad), math.cos(yaw_rad)


def _side_sign_for_desired(sample: LaneSample, desired_pos_enu: list[float]) -> float:
    nx, ny = _lane_normal(sample)
    dx = float(desired_pos_enu[0]) - sample.x_m
    dy = float(desired_pos_enu[1]) - sample.y_m
    return 1.0 if dx * nx + dy * ny >= 0.0 else -1.0


def _offset_from_lane(sample: LaneSample, lateral_m: float, z_m: float = GROUND_Z_M) -> list[float]:
    nx, ny = _lane_normal(sample)
    return q(sample.x_m + lateral_m * nx, sample.y_m + lateral_m * ny, z_m)


def road_center_point(pos_enu: list[float], z_m: float = GROUND_Z_M) -> list[float]:
    sample = LANES.nearest_to_xy(float(pos_enu[0]), float(pos_enu[1]))
    return _offset_from_lane(sample, 0.0, z_m)


def sidewalk_point(pos_enu: list[float], offset_from_curb_m: float = SIDEWALK_MIN_OFFSET_FROM_CURB_M) -> list[float]:
    anchor = SPATIAL.plan_sidewalk_anchor(
        pos_enu,
        offset_from_curb_m=offset_from_curb_m,
        allow_green=False,
        placement_semantics="sidewalk",
    )
    return anchor.position_enu_m


def _validate_landing_pad_position(pos_enu: list[float], *, context: str) -> float:
    errors = SPATIAL.validation_errors_for_point(
        pos_enu,
        context=context,
        allow_road=False,
        allow_green=True,
        road_buffer_m=PEDESTRIAN_ROAD_BUFFER_M,
    )
    road_clearance_m = SPATIAL.nearest_lane_clearance(pos_enu)
    if errors:
        raise RuntimeError(f"Illegal landing pad placement for {context}: {errors[:4]}")
    return round(road_clearance_m, 3)


def landing_pad_anchor(
    desired_pos_enu: list[float],
    *,
    context: str,
    offset_from_curb_m: float = LANDING_PAD_MIN_OFFSET_FROM_CURB_M,
) -> PlannedSidewalkAnchor:
    anchor = SPATIAL.plan_sidewalk_anchor(
        desired_pos_enu,
        offset_from_curb_m=offset_from_curb_m,
        allow_green=True,
        placement_semantics="uav_pad",
    )
    _validate_landing_pad_position(anchor.position_enu_m, context=context)
    return anchor


def landing_pad_pose(
    desired_pos_enu: list[float],
    *,
    context: str,
    offset_from_curb_m: float = LANDING_PAD_MIN_OFFSET_FROM_CURB_M,
) -> tuple[list[float], float, PlannedSidewalkAnchor]:
    anchor = landing_pad_anchor(
        desired_pos_enu,
        context=context,
        offset_from_curb_m=offset_from_curb_m,
    )
    return anchor.position_enu_m, float(anchor.sample.yaw_deg), anchor


def gathering_point(pos_enu: list[float], extent_cm: list[float] | None = None) -> list[float]:
    return SPATIAL.plan_crowd_zone(pos_enu, extent_cm or [0.0, 0.0, 0.0], allow_green=True)


def scene_pos(scene: dict[str, Any]) -> list[float]:
    placement = scene["placement"]
    return placement.get("resolved_position_enu_m") or placement.get("position_enu_m") or placement.get("center_enu_m")


def shifted_sidewalk_point(base_pos_enu: list[float], dx_m: float, dy_m: float, offset_from_curb_m: float = SIDEWALK_MIN_OFFSET_FROM_CURB_M) -> list[float]:
    base_sample = LANES.nearest_to_xy(float(base_pos_enu[0]), float(base_pos_enu[1]))
    yaw_rad = math.radians(base_sample.yaw_deg)
    s_delta = dx_m * math.cos(yaw_rad) + dy_m * math.sin(yaw_rad)
    s_delta = max(-18.0, min(18.0, s_delta))
    sign = _side_sign_for_desired(base_sample, base_pos_enu)
    min_s, max_s = LANES.edge_s_bounds(base_sample.edge_id)
    target_s = max(min_s, min(max_s, base_sample.s_m + s_delta))
    errors: list[str] = []
    for delta_s in (0.0, -2.0, 2.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0):
        sample = LANES.resolve_edge_s(base_sample.edge_id, max(min_s, min(max_s, target_s + delta_s)))
        for extra_offset in (0.0, 0.8, 1.6, 3.0, 5.0):
            offset = max(abs(offset_from_curb_m), SIDEWALK_MIN_OFFSET_FROM_CURB_M) + extra_offset
            point = _offset_from_lane(sample, sign * (LANE_HALF_WIDTH_M + offset), GROUND_Z_M)
            point_errors = SPATIAL.validation_errors_for_point(
                point,
                context="sidewalk_shifted_target",
                allow_road=False,
                allow_green=True,
            )
            if not point_errors:
                try:
                    SPATIAL.plan_sidewalk_route(
                        [base_pos_enu, point],
                        allow_green=True,
                        context="sidewalk_shifted_target_reachability",
                    )
                except SpatialValidationError as exc:
                    errors.append(str(exc))
                    continue
                return point
            errors.extend(point_errors[:2])
    raise RuntimeError(f"Unable to plan shifted sidewalk target from {base_pos_enu}: {errors[:6]}")


def rot(yaw: float = 0.0, pitch: float | None = None, roll: float | None = None) -> dict[str, float]:
    r = {"yaw_deg": round(yaw, 3)}
    if pitch is not None:
        r["pitch_deg"] = round(pitch, 3)
    if roll is not None:
        r["roll_deg"] = round(roll, 3)
    return r


def tick(t: int) -> dict[str, Any]:
    return {"type": "tick", "tick": t}


def weather(parameter: str, operator: str, value: float, sustain_ticks: int = 0) -> dict[str, Any]:
    out = {
        "type": "weather_state",
        "weather_parameter": parameter,
        "weather_operator": operator,
        "weather_value": value,
    }
    if sustain_ticks:
        out["sustain_ticks"] = sustain_ticks
    return out


def prox(
    a: str,
    b: str,
    distance: float,
    min_true_ticks: int = 2,
    *,
    metric: str = "3d",
    horizontal_distance_m: float | None = None,
    vertical_distance_m: float | None = None,
) -> dict[str, Any]:
    out = {
        "type": "entity_proximity",
        "entity_a": a,
        "entity_b": b,
        "distance_m": distance,
        "proximity_operator": "lte",
        "min_true_ticks": min_true_ticks,
    }
    if metric:
        out["metric"] = metric
    if horizontal_distance_m is not None:
        out["horizontal_distance_m"] = horizontal_distance_m
    if vertical_distance_m is not None:
        out["vertical_distance_m"] = vertical_distance_m
    return out


def fired(event_id: str, delay_ticks: int = 30) -> dict[str, Any]:
    if delay_ticks <= 0:
        return {"type": "event_fired", "event_ref": event_id}
    return {"type": "event_fired_after", "event_ref": event_id, "delay_ticks": delay_ticks}


def composite(operator: str, children: list[str]) -> dict[str, Any]:
    return {"type": "composite", "composite_operator": operator, "composite_children": children}


def action(action_id: str, action_type: str, **params: Any) -> dict[str, Any]:
    return {"type": action_type, "params": {"action_id": action_id, **params}}


def _is_pedestrian_entity_id(entity_id: str) -> bool:
    clean = str(entity_id or "").lower()
    return clean.startswith(("ped", "pedestrian"))


def _is_crossing_move(action_id: str) -> bool:
    clean = str(action_id or "").lower()
    return any(token in clean for token in ("crosswalk", "roadway", "retreat", "jaywalk"))


def _move_allows_green(action_id: str) -> bool:
    clean = str(action_id or "").lower()
    return any(token in clean for token in ("crowd", "evac", "gather", "safe"))


def _planned_pedestrian_waypoints(action_id: str, entity_id: str, waypoints: list[list[float]]) -> list[list[float]]:
    if not _is_pedestrian_entity_id(entity_id):
        return waypoints
    if _is_crossing_move(action_id):
        return SPATIAL.plan_crossing_path(waypoints, context=f"{action_id} {entity_id}")
    return SPATIAL.plan_sidewalk_route(
        waypoints,
        allow_green=_move_allows_green(action_id),
        context=f"{action_id} {entity_id}",
    )


def move(
    action_id: str,
    entity_id: str,
    waypoints: list[list[float]],
    velocity: float,
    activity_type: str | None = None,
    post_activity_type: str | None = None,
) -> dict[str, Any]:
    waypoints = _planned_pedestrian_waypoints(action_id, entity_id, waypoints)
    params: dict[str, Any] = {
        "entity_id": entity_id,
        "waypoints_enu_m": waypoints,
        "velocity_mps": velocity,
    }
    if _is_pedestrian_entity_id(entity_id):
        inferred = activity_type
        if inferred is None:
            if _is_crossing_move(action_id):
                inferred = "crossing"
            elif "evac" in action_id.lower():
                inferred = "evacuating"
            elif "clear" in action_id.lower() or "evade" in action_id.lower() or "retreat" in action_id.lower():
                inferred = "walking"
            else:
                inferred = "walking"
        params["activity_type"] = normalize_activity_type(inferred, moving=True)
        params["post_activity_type"] = normalize_activity_type(post_activity_type or "waiting")
    return action(
        action_id,
        "move_entity",
        **params,
    )


def visual(action_id: str, entity_id: str, mode: str, **state: Any) -> dict[str, Any]:
    return action(action_id, "set_visual_state", entity_id=entity_id, visual_state={"mode": mode, **state})


def spawn(action_id: str, entity_id: str, asset_id: str, pos_enu: list[float], yaw: float = 0.0, **extra: Any) -> dict[str, Any]:
    params = {
        "entity_id": entity_id,
        "asset_id": asset_id,
        "position_enu_m": pos_enu,
        "rotation_deg": rot(yaw),
    }
    params.update(extra)
    return action(action_id, "spawn_entity", **params)


def scene_yaw(scene: dict[str, Any]) -> float:
    rotation = dict((scene.get("placement") or {}).get("rotation_deg") or {})
    return float(rotation.get("yaw_deg", rotation.get("yaw", 0.0)))


def spawn_from_scene(action_id: str, scene: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return spawn(
        action_id,
        scene["entity_id"],
        scene["logical_asset_id"],
        scene_pos(scene),
        scene_yaw(scene),
        **extra,
    )


def remove(action_id: str, entity_id: str) -> dict[str, Any]:
    return action(action_id, "remove_entity", entity_id=entity_id)


def screenshot(action_id: str, camera_id: str = "demo_high_overview") -> dict[str, Any]:
    return action(action_id, "capture_screenshot", camera_id=camera_id)


def play(action_id: str, ped_id: str) -> dict[str, Any]:
    return ped_activity(action_id, ped_id, "medical_incident")


def ped_activity(action_id: str, ped_id: str, activity_type: str) -> dict[str, Any]:
    activity_type = normalize_activity_type(activity_type)
    return action(action_id, "set_pedestrian_activity", entity_id=ped_id, activity_type=activity_type)


def crowd(action_id: str, group_id: str, count: int, origin: list[float], extent_cm: list[float], seed: int) -> dict[str, Any]:
    SPATIAL.validate_spawn_envelope(
        origin,
        extent_cm,
        context=f"spawn_crowd {group_id}",
        allow_green=True,
    )
    return action(
        action_id,
        "spawn_crowd",
        group_id=group_id,
        count=count,
        spawn_origin_enu_m=origin,
        spawn_box_extent_cm=extent_cm,
        seed=seed,
    )


def clear_crowd(action_id: str, group_id: str) -> dict[str, Any]:
    return action(action_id, "clear_crowd", group_id=group_id)


def set_weather(action_id: str, profile: str, **overrides: float) -> dict[str, Any]:
    return action(action_id, "set_weather", profile=profile, overrides=overrides)


def safe_id(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _event_intent(event_id: str, title: str, category: str) -> str:
    for value in (title, event_id, category):
        intent = canonical_intent_from_text(value)
        if intent:
            return intent
    raise RuntimeError(f"Unable to assign explicit intent for event {event_id}")


def _target_role_from_id(entity_id: str) -> str:
    lowered = entity_id.lower()
    if "u_inspect" in lowered:
        return "inspect_observer"
    if lowered.startswith("uav") or lowered.startswith("u_"):
        return "mission_uav"
    if lowered.startswith("ped") or "_ped" in lowered or "crowd" in lowered:
        return "pedestrian"
    if lowered.startswith(("car", "veh", "ambulance", "police")) or "_vehicle" in lowered:
        return "vehicle"
    if "pad" in lowered:
        return "landing_pad"
    if any(token in lowered for token in ("nfz", "hazard", "zone", "boundary", "anchor")):
        return "capture_boundary"
    if any(token in lowered for token in ("tower", "charger", "facility")):
        return "facility"
    return "semantic_context"


def _target_roles(targets: list[str]) -> list[str]:
    roles = sorted({_target_role_from_id(str(target)) for target in targets if str(target)})
    return roles or ["semantic_context"]


def finalize_event_semantics(events: list[dict[str, Any]], scenario_id: str) -> None:
    predecessor = ""
    for index, item in enumerate(events):
        locked_intent = bool(item.pop("_intent_locked", False))
        if locked_intent and item.get("intent"):
            intent = str(item["intent"])
        else:
            intent = _event_intent(
                str(item.get("event_id") or index),
                str(item.get("log_title") or ""),
                str(item.get("log_category") or ""),
            )
        item["intent"] = intent
        item["intent_stage"] = str(item.get("intent_stage") or f"{index:02d}.{intent}")
        item["causal_chain_id"] = str(item.get("causal_chain_id") or f"{scenario_id}.semantic_event_chain")
        item["causal_predecessor_intent"] = predecessor
        item["target_roles"] = list(item.get("target_roles") or _target_roles(list(item.get("log_target_ids") or [])))
        predecessor = intent


def event(
    event_id: str,
    trigger: dict[str, Any],
    actions: list[dict[str, Any]],
    priority: int,
    scenario_id: str,
    title: str,
    category: str,
    targets: list[str],
    severity: str = "info",
    overlay: str | None = None,
    emits: list[str] | None = None,
    intent: str | None = None,
) -> dict[str, Any]:
    event_intent = intent or _event_intent(event_id, title, category)
    intent_stage = f"{priority:02d}.{event_intent}"
    return {
        "event_id": event_id,
        "trigger": trigger,
        "actions": actions,
        "priority": priority,
        "max_fire_count": 1,
        "on_fire_emit": emits or [],
        "intent": event_intent,
        "_intent_locked": intent is not None,
        "intent_stage": intent_stage,
        "causal_chain_id": f"{scenario_id}.semantic_event_chain",
        "causal_predecessor_intent": "",
        "target_roles": _target_roles(targets),
        "log_topic": f"evt_{scenario_id}_{event_id}",
        "log_category": category,
        "log_title": title,
        "log_severity": severity,
        "log_overlay": overlay or category,
        "log_target_ids": targets,
    }


def world_entity(
    entity_id: str,
    asset: str,
    category: str,
    pos_enu: list[float],
    yaw: float = 0.0,
    mode: str | None = None,
    activation_tick: int = 0,
    route: list[list[float]] | None = None,
    visual_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = visual_state.copy() if visual_state else {}
    if mode and "mode" not in state:
        state["mode"] = mode
    if asset.startswith("pedestrian.") or category == "pedestrian":
        raw_activity = str(state.get("activity_type") or state.get("mode") or "waiting")
        if raw_activity == "walking" and not route:
            raw_activity = "waiting"
        activity = get_activity(raw_activity)
        state.update(
            {
                "mode": activity.activity_type,
                "activity_type": activity.activity_type,
                "animation_hint": activity.animation_hint,
                "posture": activity.posture,
                "social_state": activity.social_state,
            }
        )
    default_mode = str(state.get("mode") or mode or "idle")
    if "task_id" not in state:
        state["task_id"] = f"{entity_id}.task"
    if "role" not in state:
        if asset.startswith("uav.") or category == "uav":
            state["role"] = "mission_uav"
        elif asset.startswith("vehicle.") or category == "vehicle":
            state["role"] = "semantic_vehicle"
        elif asset.startswith("pedestrian.") or category == "pedestrian":
            state["role"] = "semantic_pedestrian"
        elif asset.startswith("facility.") or category in {"facility", "traffic_signal"}:
            state["role"] = "semantic_facility"
        else:
            state["role"] = "semantic_context"
    if "state_sequence" not in state:
        state["state_sequence"] = [default_mode]
    if "semantic_role" not in state:
        state["semantic_role"] = default_mode
    spec = {
        "entity_id": entity_id,
        "asset_id": asset,
        "initial_pos_enu": pos_enu,
        "initial_rotation_deg": [0.0, 0.0, yaw],
        "movement_waypoints": route or [],
        "visual_state": state or None,
    }
    scene = {
        "entity_id": entity_id,
        "logical_asset_id": asset,
        "category": category,
        "placement_mode": "world_pose",
        "placement": {"position_enu_m": pos_enu, "resolved_position_enu_m": pos_enu, "rotation_deg": rot(yaw)},
        "route_waypoints_enu_m": route or [],
        "initial_state": state or {},
        "activation_tick": activation_tick,
    }
    if activation_tick > 0:
        scene["enabled"] = False
        scene["spawn_policy"] = "event_script_only"
    return spec, scene


def semantic_state(
    *,
    task_id: str,
    role: str,
    state_sequence: list[str],
    semantic_role: str,
    mode: str,
    **extra: Any,
) -> dict[str, Any]:
    state = {
        "mode": mode,
        "task_id": task_id,
        "role": role,
        "semantic_role": semantic_role,
        "state_sequence": state_sequence,
    }
    state.update(extra)
    return state


def apply_task_state(
    spec: dict[str, Any],
    scene: dict[str, Any],
    *,
    task_id: str,
    role: str,
    state_sequence: list[str],
    semantic_role: str,
    mode: str,
    **extra: Any,
) -> None:
    state = semantic_state(
        task_id=task_id,
        role=role,
        state_sequence=state_sequence,
        semantic_role=semantic_role,
        mode=mode,
        **extra,
    )
    spec["visual_state"] = {**(spec.get("visual_state") or {}), **state}
    scene["initial_state"] = {**(scene.get("initial_state") or {}), **state}


def add_with_task(
    target_specs: list[dict[str, Any]],
    target_scenes: list[dict[str, Any]],
    pair: tuple[dict[str, Any], dict[str, Any]],
    *,
    task_id: str,
    role: str,
    state_sequence: list[str],
    semantic_role: str,
    mode: str,
    **extra: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, scene = pair
    apply_task_state(
        spec,
        scene,
        task_id=task_id,
        role=role,
        state_sequence=state_sequence,
        semantic_role=semantic_role,
        mode=mode,
        **extra,
    )
    target_specs.append(spec)
    target_scenes.append(scene)
    return spec, scene


def lane_entity(
    entity_id: str,
    asset: str,
    category: str,
    edge_id: str,
    s: float,
    lateral: float,
    pos_enu: list[float],
    yaw: float = 0.0,
    mode: str | None = None,
    activation_tick: int = 0,
    route: list[list[float]] | None = None,
    visual_state: dict[str, Any] | None = None,
    prefer_edge_hint: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample = LANES.resolve_edge_s(edge_id, s) if prefer_edge_hint else LANES.nearest_to_xy(float(pos_enu[0]), float(pos_enu[1]))
    sign = _side_sign_for_desired(sample, pos_enu)
    requested_lateral = float(lateral)
    placement_semantics = "lane_center"
    if asset.startswith("prop.roadwork."):
        sign = 1.0 if float(lateral) >= 0.0 else -1.0
        offset_from_edge = max(abs(requested_lateral), ROADWORK_MIN_OFFSET_FROM_EDGE_M)
        requested_lateral = sign * (LANE_HALF_WIDTH_M + offset_from_edge)
        placement_semantics = "roadwork_shoulder"
    pos_enu = _offset_from_lane(sample, requested_lateral, float(pos_enu[2] if len(pos_enu) > 2 else GROUND_Z_M))
    yaw = sample.yaw_deg
    spec, scene = world_entity(
        entity_id, asset, category, pos_enu, yaw, mode, activation_tick, route, visual_state
    )
    scene["placement_mode"] = "lane_anchor"
    scene["placement"] = {
        "edge_id": sample.edge_id,
        "lane_index": sample.lane_index,
        "longitudinal_s": round(sample.s_m, 3),
        "lateral_offset_m": round(lateral, 3),
        "lane_half_width_m": LANE_HALF_WIDTH_M,
        "resolved_lateral_from_center_m": round(requested_lateral, 3),
        "placement_semantics": placement_semantics,
        "resolved_position_enu_m": pos_enu,
        "rotation_deg": rot(yaw),
        "source_edge_id_hint": edge_id,
        "source_longitudinal_s_hint": round(s, 3),
    }
    return spec, scene


def sidewalk_entity(
    entity_id: str,
    asset: str,
    category: str,
    edge_id: str,
    s: float,
    offset: float,
    pos_enu: list[float],
    yaw: float = 0.0,
    mode: str | None = None,
    route: list[list[float]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor = SPATIAL.plan_sidewalk_anchor(
        pos_enu,
        edge_id_hint=edge_id,
        s_hint=s,
        offset_from_curb_m=offset,
        allow_green=False,
        placement_semantics="sidewalk",
    )
    if dist_xy(anchor.position_enu_m, pos_enu) > 80.0:
        anchor = SPATIAL.plan_sidewalk_anchor(
            pos_enu,
            offset_from_curb_m=offset,
            allow_green=False,
            placement_semantics="sidewalk",
        )
    sample = anchor.sample
    offset_from_curb = anchor.offset_from_curb_m
    resolved_lateral = anchor.resolved_lateral_from_center_m
    pos_enu = anchor.position_enu_m
    yaw = sample.yaw_deg
    spec, scene = world_entity(entity_id, asset, category, pos_enu, yaw, mode, route=route)
    scene["placement_mode"] = "sidewalk_anchor"
    scene["placement"] = {
        "lane_edge_id": sample.edge_id,
        "longitudinal_s": round(sample.s_m, 3),
        "offset_from_curb_m": round(offset_from_curb, 3),
        "lane_half_width_m": LANE_HALF_WIDTH_M,
        "resolved_lateral_from_center_m": round(resolved_lateral, 3),
        "placement_semantics": anchor.placement_semantics,
        "resolved_position_enu_m": pos_enu,
        "rotation_deg": rot(yaw),
        "source_lane_edge_id_hint": edge_id,
        "source_longitudinal_s_hint": round(s, 3),
    }
    return spec, scene


def crosswalk_entity(
    entity_id: str,
    asset: str,
    crosswalk_id: str,
    side: str,
    pos_enu: list[float],
    yaw: float = 0.0,
    route: list[list[float]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    crossing = SPATIAL.plan_crossing_route(pos_enu, offset_from_curb_m=SIDEWALK_MIN_OFFSET_FROM_CURB_M)
    sample = crossing.sample
    start_pos = crossing.start_position_enu_m
    road_pos = crossing.roadway_center_position_enu_m
    opposite_pos = crossing.opposite_curb_position_enu_m
    spec, scene = world_entity(entity_id, asset, "pedestrian", start_pos, sample.yaw_deg, "walking", route=route)
    scene["placement_mode"] = "crosswalk_anchor"
    scene["placement"] = {
        "crosswalk_id": crosswalk_id,
        "side": side,
        "lane_edge_id": sample.edge_id,
        "longitudinal_s": round(sample.s_m, 3),
        "lane_half_width_m": LANE_HALF_WIDTH_M,
        "offset_from_curb_m": SIDEWALK_MIN_OFFSET_FROM_CURB_M,
        "resolved_lateral_from_center_m": crossing.resolved_lateral_from_center_m,
        "resolved_position_enu_m": start_pos,
        "roadway_center_position_enu_m": road_pos,
        "opposite_curb_position_enu_m": opposite_pos,
        "placement_semantics": "crosswalk_curb_start",
        "rotation_deg": rot(sample.yaw_deg),
    }
    return spec, scene


def facade_entity(
    entity_id: str,
    building_id: str,
    pos_enu: list[float],
    normal: list[float],
    stand_off: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, scene = world_entity(entity_id, "semantic.asset_anchor", "facade_anchor", pos_enu, 0.0)
    scene["placement_mode"] = "facade_anchor"
    scene["placement"] = {
        "building_id": building_id,
        "building_source": BUILDINGS.source_ref,
        "building_id_property": "properties.id_origin",
        "facade_anchor_method": "semantic_resolved_position",
        "outward_normal_enu": normal,
        "stand_off_m": stand_off,
        "position_enu_m": pos_enu,
        "resolved_position_enu_m": pos_enu,
    }
    return spec, scene


def box_entity(
    entity_id: str,
    asset: str,
    category: str,
    center: list[float],
    extent: list[float],
    activation_tick: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, scene = world_entity(entity_id, asset, category, center, 0.0, activation_tick=activation_tick)
    scene["placement_mode"] = "box_volume"
    scene["placement"] = {"center_enu_m": center, "resolved_position_enu_m": center, "extent_m": extent}
    return spec, scene


def polygon_entity(
    entity_id: str,
    asset: str,
    category: str,
    polygon: list[list[float]],
    base_z: float,
    height: float,
    center: list[float],
    activation_tick: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, scene = world_entity(entity_id, asset, category, center, 0.0, activation_tick=activation_tick)
    scene["placement_mode"] = "polygon_prism"
    scene["placement"] = {
        "polygon_enu_m": polygon,
        "base_z_m": base_z,
        "height_m": height,
        "resolved_position_enu_m": center,
    }
    return spec, scene


def pad_entity(
    entity_id: str,
    pos_enu: list[float],
    pad_instance_id: str,
    approach_side: str,
    yaw: float = 0.0,
    anchor: PlannedSidewalkAnchor | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    road_clearance_m = _validate_landing_pad_position(pos_enu, context=f"pad_entity {entity_id}")
    spec, scene = world_entity(entity_id, "facility.landing_pad.visible.v1", "facility", pos_enu, yaw, "available")
    scene["placement_mode"] = "pad_anchor"
    placement = {
        "pad_instance_id": pad_instance_id,
        "approach_side": approach_side,
        "position_enu_m": pos_enu,
        "resolved_position_enu_m": pos_enu,
        "rotation_deg": rot(yaw),
        "road_clearance_m": road_clearance_m,
    }
    if anchor is not None:
        placement.update(
            {
                "anchor_edge_id": anchor.sample.edge_id,
                "anchor_lane_s_m": round(float(anchor.sample.s_m), 3),
                "offset_from_curb_m": round(float(anchor.offset_from_curb_m), 3),
                "resolved_lateral_from_center_m": round(float(anchor.resolved_lateral_from_center_m), 3),
                "placement_semantics": anchor.placement_semantics,
            }
        )
    scene["placement"] = placement
    return spec, scene


def add_evacuation_ped_cohort(
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    sid: str,
    prefix: str,
    start_origin: list[float],
    safe_origin: list[float],
    count: int = 8,
) -> list[tuple[str, list[float], list[list[float]]]]:
    def cohort_sidewalk_point(origin: list[float], index: int) -> tuple[LaneSample, float, float, list[float]]:
        base = LANES.nearest_to_xy(float(origin[0]), float(origin[1]))
        col = index % 4
        row = index // 4
        s_offset = (col - 1.5) * 3.5
        offset_from_curb = GATHERING_MIN_OFFSET_FROM_CURB_M + row * 4.0 - col * 0.5
        anchor = SPATIAL.plan_sidewalk_anchor(
            [origin[0], origin[1], GROUND_Z_M],
            edge_id_hint=base.edge_id,
            s_hint=base.s_m + s_offset,
            offset_from_curb_m=offset_from_curb,
            allow_green=True,
            placement_semantics="sidewalk_evacuation_cohort",
        )
        return anchor.sample, anchor.offset_from_curb_m, anchor.resolved_lateral_from_center_m, anchor.position_enu_m

    direction_sample = LANES.nearest_to_xy(float(start_origin[0]), float(start_origin[1]))
    direction_yaw_rad = math.radians(direction_sample.yaw_deg)
    direction_dx = float(safe_origin[0]) - float(start_origin[0])
    direction_dy = float(safe_origin[1]) - float(start_origin[1])
    evacuation_direction = 1.0 if direction_dx * math.cos(direction_yaw_rad) + direction_dy * math.sin(direction_yaw_rad) >= 0.0 else -1.0

    def cohort_route(
        start_sample: LaneSample,
        target_sample: LaneSample,
        side_sign: float,
        offset_from_curb: float,
        start: list[float],
        target: list[float],
    ) -> list[list[float]]:
        s0 = float(start_sample.s_m)
        s1 = float(target_sample.s_m)
        steps = max(1, int(math.ceil(abs(s1 - s0) / 18.0)))
        points: list[list[float]] = [start]
        for step in range(1, steps):
            s_m = s0 + (s1 - s0) * step / float(steps)
            sample = LANES.resolve_edge_s(start_sample.edge_id, s_m)
            points.append(_offset_from_lane(sample, side_sign * (LANE_HALF_WIDTH_M + offset_from_curb), GROUND_Z_M))
        points.append(target)
        deduped: list[list[float]] = []
        for point in points:
            if not deduped or math.hypot(deduped[-1][0] - point[0], deduped[-1][1] - point[1]) > 0.001:
                deduped.append(point)
        for segment_index, (a, b) in enumerate(zip(deduped, deduped[1:])):
            SPATIAL.validate_segment(
                a,
                b,
                context=f"sidewalk_evacuation_route {start_sample.edge_id} segment {segment_index}",
                allow_road=False,
                allow_green=True,
            )
        return deduped

    def cohort_safe_route(start_sample: LaneSample, start_offset: float, start: list[float]) -> tuple[list[float], list[list[float]]]:
        side_sign = _side_sign_for_desired(start_sample, start)
        min_s, max_s = LANES.edge_s_bounds(start_sample.edge_id)
        errors: list[str] = []
        for dir_sign in (evacuation_direction,):
            for travel_m in (24.0, 21.0, 18.0, 15.0, 12.0, 9.0, 6.0):
                target_s = max(min_s, min(max_s, start_sample.s_m + dir_sign * travel_m))
                target_sample = LANES.resolve_edge_s(start_sample.edge_id, target_s)
                for extra_offset in (0.0, 0.8, 1.6, 3.0, 5.0):
                    target = _offset_from_lane(
                        target_sample,
                        side_sign * (LANE_HALF_WIDTH_M + start_offset + extra_offset),
                        GROUND_Z_M,
                    )
                    point_errors = SPATIAL.validation_errors_for_point(
                        target,
                        context="sidewalk_evacuation_safe_target",
                        allow_road=False,
                        allow_green=True,
                    )
                    if not point_errors and math.hypot(start[0] - target[0], start[1] - target[1]) >= 4.0:
                        try:
                            route = cohort_route(start_sample, target_sample, side_sign, start_offset + extra_offset, start, target)
                            return target, route
                        except SpatialValidationError as exc:
                            errors.append(str(exc))
                            continue
                    errors.extend(point_errors[:2])
        raise RuntimeError(f"Unable to plan same-edge evacuation target from {start}: {errors[:6]}")

    cohort: list[tuple[str, list[float], list[list[float]]]] = []
    for index in range(count):
        start_sample, start_offset, start_lateral, start = cohort_sidewalk_point(start_origin, index)
        _safe, route = cohort_safe_route(start_sample, start_offset, start)
        ped_id = f"{prefix}_{sid}_{index:02d}"
        spec, ped_scene = world_entity(ped_id, "pedestrian.cityops.basic.v1", "pedestrian", start, start_sample.yaw_deg, "standing")
        ped_scene["placement_mode"] = "sidewalk_anchor"
        ped_scene["placement"] = {
            "lane_edge_id": start_sample.edge_id,
            "longitudinal_s": round(start_sample.s_m, 3),
            "offset_from_curb_m": round(start_offset, 3),
            "lane_half_width_m": LANE_HALF_WIDTH_M,
            "resolved_lateral_from_center_m": round(start_lateral, 3),
            "placement_semantics": "sidewalk_evacuation_cohort",
            "resolved_position_enu_m": start,
            "rotation_deg": rot(start_sample.yaw_deg),
        }
        add(specs, scenes, (spec, ped_scene))
        cohort.append((ped_id, start, route))
    return cohort


def evacuation_move_actions(cohort: list[tuple[str, list[float], list[list[float]]]], action_prefix: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for index, (ped_id, start, route) in enumerate(cohort):
        if not route or math.hypot(route[0][0] - start[0], route[0][1] - start[1]) > 0.001:
            raise RuntimeError(f"{action_prefix}_{index:02d} {ped_id}: evacuation route does not start at scene position")
        actions.append(move(f"{action_prefix}_{index:02d}", ped_id, route, 1.4))
    return actions


def add_static_ped_cohort(
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    sid: str,
    prefix: str,
    origin: list[float],
    count: int,
    *,
    activity_type: str = "chatting",
    allow_green: bool = True,
) -> list[str]:
    base = LANES.nearest_to_xy(float(origin[0]), float(origin[1]))
    min_s, max_s = LANES.edge_s_bounds(base.edge_id)
    occupied = [
        scene_pos(scene)
        for scene in scenes
        if str(scene.get("logical_asset_id", "")).startswith("pedestrian.")
    ]
    ids: list[str] = []
    for index in range(count):
        anchor = None
        errors: list[str] = []
        s_offsets = [0.0, -4.0, 4.0, -8.0, 8.0, -12.0, 12.0, -16.0, 16.0, -20.0, 20.0, -24.0, 24.0]
        offset_values = [GATHERING_MIN_OFFSET_FROM_CURB_M + 1.35 * row for row in range(9)]
        for offset_from_curb in offset_values:
            for s_offset in s_offsets:
                try:
                    candidate = SPATIAL.plan_sidewalk_anchor(
                        [origin[0], origin[1], GROUND_Z_M],
                        edge_id_hint=base.edge_id,
                        s_hint=max(min_s, min(max_s, base.s_m + s_offset)),
                        offset_from_curb_m=offset_from_curb,
                        allow_green=allow_green,
                        placement_semantics="sidewalk_static_ped_cohort",
                    )
                except SpatialValidationError as exc:
                    errors.append(str(exc))
                    continue
                if all(dist_xy(candidate.position_enu_m, other) >= 0.9 for other in occupied):
                    anchor = candidate
                    occupied.append(candidate.position_enu_m)
                    break
            if anchor is not None:
                break
        if anchor is None:
            raise RuntimeError(f"Unable to place static pedestrian cohort member {prefix}_{sid}_{index:02d}: {errors[:6]}")
        ped_id = f"{prefix}_{sid}_{index:02d}"
        spec, ped_scene = world_entity(
            ped_id,
            "pedestrian.cityops.basic.v1",
            "pedestrian",
            anchor.position_enu_m,
            anchor.sample.yaw_deg,
            activity_type,
        )
        ped_scene["placement_mode"] = "sidewalk_anchor"
        ped_scene["placement"] = {
            "lane_edge_id": anchor.sample.edge_id,
            "longitudinal_s": round(anchor.sample.s_m, 3),
            "offset_from_curb_m": round(anchor.offset_from_curb_m, 3),
            "lane_half_width_m": LANE_HALF_WIDTH_M,
            "resolved_lateral_from_center_m": round(anchor.resolved_lateral_from_center_m, 3),
            "placement_semantics": "sidewalk_static_ped_cohort",
            "resolved_position_enu_m": anchor.position_enu_m,
            "rotation_deg": rot(anchor.sample.yaw_deg),
            "cohort_id": f"{prefix}_{sid}",
        }
        add(specs, scenes, (spec, ped_scene))
        ids.append(ped_id)
    return ids


def background_vehicle_policy_for(scenario_id: str) -> BackgroundVehiclePolicy | None:
    for prefix, policy in BACKGROUND_VEHICLE_POLICY_BY_PREFIX:
        if scenario_id.startswith(prefix):
            return policy
    return None


def _entity_matches_category(scene: dict[str, Any], category: str) -> bool:
    asset = str(scene.get("logical_asset_id") or "")
    scene_category = str(scene.get("category") or "")
    entity_id = str(scene.get("entity_id") or "")
    if category == "uav":
        return asset.startswith("uav.")
    if category == "vehicle":
        return asset.startswith("vehicle.") or scene_category == "vehicle"
    if category == "pedestrian":
        return asset.startswith("pedestrian.") or scene_category == "pedestrian"
    if category == "facility":
        if entity_id.startswith("pad_home_"):
            return False
        return asset.startswith("facility.") or scene_category in {"facility", "traffic_signal"}
    if category == "logical":
        return (
            asset == UAV_CORRIDOR_LOGICAL_ASSET_ID
            or asset.startswith("trigger.")
            or scene_category in {"airspace_constraint", "hazard_zone", "crowd_anchor", "airspace_corridor"}
        )
    raise ValueError(f"Unknown contract category: {category}")


def _scene_counts(scenes: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "uav": sum(1 for scene in scenes if _entity_matches_category(scene, "uav")),
        "vehicle": sum(1 for scene in scenes if _entity_matches_category(scene, "vehicle")),
        "pedestrian": sum(1 for scene in scenes if _entity_matches_category(scene, "pedestrian")),
        "facility": sum(1 for scene in scenes if _entity_matches_category(scene, "facility")),
        "logical": sum(1 for scene in scenes if _entity_matches_category(scene, "logical")),
    }


def _contract_anchor_points(scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[list[float]]:
    points = [
        point
        for point in collect_scene_bound_points(ScenarioBundle("", Path("."), "", "", [], scenes, {}, events))
        if len(point) >= 3
    ]
    return points or [[WORLD_OFFSET_X_M, WORLD_OFFSET_Y_M, GROUND_Z_M]]


def _contract_center(scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[float]:
    points = _contract_anchor_points(scenes, events)
    return [
        round(sum(point[0] for point in points) / len(points), 3),
        round(sum(point[1] for point in points) / len(points), 3),
        round(sum(point[2] for point in points) / len(points), 3),
    ]


def _find_free_sidewalk_anchor(
    origin: list[float],
    occupied: list[list[float]],
    *,
    semantic: str,
    allow_green: bool = True,
) -> Any:
    for anchor in _iter_free_sidewalk_anchors(origin, occupied, semantic=semantic, allow_green=allow_green):
        occupied.append(anchor.position_enu_m)
        return anchor
    raise RuntimeError(f"Unable to place sidewalk semantic actor near {origin}")


def _iter_free_sidewalk_anchors(
    origin: list[float],
    occupied: list[list[float]],
    *,
    semantic: str,
    allow_green: bool = True,
):
    base = LANES.nearest_to_xy(float(origin[0]), float(origin[1]))
    nearby_bases: list[LaneSample] = [base]
    seen_edges = {base.edge_id}
    for sample in sorted(LANES.samples, key=lambda item: (item.x_m - float(origin[0])) ** 2 + (item.y_m - float(origin[1])) ** 2):
        if sample.edge_id in seen_edges:
            continue
        min_edge_s, max_edge_s = LANES.edge_s_bounds(sample.edge_id)
        if max_edge_s - min_edge_s < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M:
            continue
        seen_edges.add(sample.edge_id)
        nearby_bases.append(sample)
        if len(nearby_bases) >= 24:
            break
    def point_errors(point: list[float]) -> list[str]:
        probe = SPATIAL._point_geometry(point)
        local_errors: list[str] = []
        if not SPATIAL.bounds_prepared.covers(probe):
            local_errors.append(f"{semantic} candidate outside bounds.geojson: {point}")
        if SPATIAL.water_prepared.covers(probe):
            local_errors.append(f"{semantic} candidate inside water.geojson: {point}")
        if SPATIAL.building_prepared.covers(probe):
            local_errors.append(f"{semantic} candidate inside building.geojson: {point}")
        if not allow_green and SPATIAL.green_prepared.covers(probe):
            local_errors.append(f"{semantic} candidate inside green.geojson without explicit green allowance: {point}")
        return local_errors

    for base_sample in nearby_bases:
        min_s, max_s = LANES.edge_s_bounds(base_sample.edge_id)
        preferred_sign = _side_sign_for_desired(base_sample, origin)
        for s_offset in (0.0, 6.0, -6.0, 12.0, -12.0, 18.0, -18.0, 28.0, -28.0, 42.0, -42.0, 60.0, -60.0):
            sample = LANES.resolve_edge_s(base_sample.edge_id, max(min_s, min(max_s, base_sample.s_m + s_offset)))
            for sign in (preferred_sign, -preferred_sign):
                for offset_from_curb in (SIDEWALK_MIN_OFFSET_FROM_CURB_M, 2.4, 3.6, 4.8, 6.2, 8.0, 10.0, 12.0):
                    lateral = sign * (LANE_HALF_WIDTH_M + offset_from_curb)
                    point = _offset_from_lane(sample, lateral, GROUND_Z_M)
                    current_errors = point_errors(point)
                    if current_errors:
                        continue
                    if all(dist_xy(point, other) >= 1.0 for other in occupied):
                        yield PlannedSidewalkAnchor(
                            position_enu_m=point,
                            sample=sample,
                            offset_from_curb_m=offset_from_curb,
                            resolved_lateral_from_center_m=lateral,
                            placement_semantics=semantic,
                        )


def _find_free_lane_sample(
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    occupied: list[list[float]],
    *,
    mode: str = "traffic_flow",
    index: int = 0,
) -> LaneSample:
    blockers = _background_vehicle_blockers(scenes, events)
    for min_s_gap_m, occupied_gap_m, blocker_scale in ((6.0, 7.0, 1.0), (4.5, 5.5, 0.85), (3.5, 4.5, 0.72)):
        for sample in _background_vehicle_candidate_samples(scenes, events):
            if not _lane_has_route_span(sample, BACKGROUND_VEHICLE_MIN_ROUTE_SPAN_M):
                continue
            candidate_pos = _offset_from_lane(sample, 0.0, GROUND_Z_M)
            if SPATIAL.validation_errors_for_point(
                candidate_pos,
                context="semantic contract vehicle candidate",
                allow_road=True,
                allow_green=False,
            ):
                continue
            if _near_existing_lane_vehicle(sample, scenes, min_s_gap_m=min_s_gap_m):
                continue
            if any(dist_xy(candidate_pos, point) < clearance_m * blocker_scale for _label, point, clearance_m in blockers):
                continue
            if any(dist_xy(candidate_pos, other) < occupied_gap_m for other in occupied):
                continue
            if not _semantic_vehicle_route(sample, mode, index):
                continue
            occupied.append(candidate_pos)
            return sample
    raise RuntimeError("Unable to place required semantic background vehicle without collision")


def _vehicle_mode_from_role(role: str) -> str:
    text = role.lower()
    if any(token in text for token in ("queue", "queued")):
        return "queued"
    if "yield" in text:
        return "yielding"
    if "brake" in text or "stop" in text:
        return "braking"
    if "response" in text or "ambulance" in text or "emergency" in text:
        return "responder"
    if "detour" in text:
        return "detour"
    if "blocked" in text:
        return "blocked_by_barrier"
    if "slow" in text or "cautious" in text or "fog" in text or "rain" in text:
        return "cautious_flow"
    return "traffic_flow"


def _pedestrian_mode_from_role(role: str) -> str:
    text = role.lower()
    if "evac" in text:
        return "evacuating"
    if "retreat" in text or "evade" in text or "clear" in text:
        return "evacuating"
    if "fallen" in text:
        return "medical_incident"
    if "shelter" in text:
        return "evacuating"
    if "wait" in text:
        return "waiting"
    if "bystander" in text or "observ" in text or "witness" in text:
        return "observing"
    return "walking"


def _moving_background_vehicle_mode(mode: str) -> str:
    if mode in {"queued", "braking", "yielding", "blocked_by_barrier", "blocked", "held", "stopped"}:
        return "traffic_slow"
    return mode or "traffic_flow"


def _dedupe_route_points(points: list[list[float]]) -> list[list[float]]:
    deduped: list[list[float]] = []
    for point in points:
        if not point:
            continue
        current = q(float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else GROUND_Z_M))
        if deduped and dist_xy(deduped[-1], current) < 0.05 and abs(float(deduped[-1][2]) - float(current[2])) < 0.05:
            continue
        deduped.append(current)
    return deduped


def _route_xy_span(start: list[float], route: list[list[float]]) -> float:
    points = [start, *route]
    if len(points) < 2:
        return 0.0
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _lane_has_route_span(sample: LaneSample, min_span_m: float) -> bool:
    min_s, max_s = LANES.edge_s_bounds(sample.edge_id)
    return (max_s - sample.s_m) >= min_span_m or (sample.s_m - min_s) >= min_span_m


def _snap_sumo_route_to_traffic_bundle(route: list[list[float]], *, lateral_m: float = 0.0) -> list[list[float]]:
    snapped: list[list[float]] = []
    for point in route:
        sample = LANES.nearest_to_xy(float(point[0]), float(point[1]))
        lane_point = _offset_from_lane(sample, lateral_m, float(point[2] if len(point) > 2 else GROUND_Z_M))
        if dist_xy(point, lane_point) > SUMO_TO_TRAFFIC_BUNDLE_MAX_SNAP_M:
            return []
        if SPATIAL.validation_errors_for_point(
            lane_point,
            context="SUMO ground-flow traffic-bundle snap",
            allow_road=True,
            allow_green=False,
        ):
            return []
        snapped.append(lane_point)
    return _dedupe_route_points(snapped)


def _sumo_vehicle_route(start: list[float], index: int) -> list[list[float]]:
    min_span = BACKGROUND_VEHICLE_MIN_ROUTE_SPAN_M + index * 2.0
    try:
        route = SUMO_GROUND_FLOW.plan_vehicle_route_enu(
            start,
            min_xy_span_m=min_span,
            min_path_length_m=max(32.0, min_span + 8.0),
            max_edges=24,
        )
    except SumoRouteError:
        return []
    snapped = _snap_sumo_route_to_traffic_bundle(route)
    if not snapped or _route_xy_span(start, snapped) < min_span:
        return []
    return snapped


def _ground_flow_contract(
    kind: str,
    velocity_mps: float,
    route: list[list[float]],
    *,
    route_source: str,
) -> dict[str, Any]:
    route_points = _dedupe_route_points([list(point) for point in route])
    min_xy_span_m = (
        BACKGROUND_VEHICLE_MIN_ROUTE_SPAN_M
        if kind == "vehicle"
        else BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M
    )
    return {
        "policy": "continuous_capture_ground_flow_v1",
        "actor_kind": kind,
        "required": True,
        "min_visible_motion_ratio": GROUND_FLOW_MIN_VISIBLE_MOTION_RATIO,
        "min_xy_span_m": min_xy_span_m,
        "speed_mps": float(velocity_mps),
        "route_duration_ticks": GROUND_FLOW_ROUTE_DURATION_TICKS,
        "loop_policy": "single_pass_route_waypoints",
        "route_source": route_source,
        "planned_path_length_m": round(path_length_m(route_points), 3) if len(route_points) >= 2 else 0.0,
    }


def _semantic_vehicle_route(sample: LaneSample, _mode: str, index: int) -> list[list[float]]:
    start = _offset_from_lane(sample, 0.0, GROUND_Z_M)
    return _sumo_vehicle_route(start, index)


def _semantic_ped_route(anchor: Any, mode: str, index: int) -> list[list[float]]:
    min_s, max_s = LANES.edge_s_bounds(anchor.sample.edge_id)
    travel_m = 5.5 + index * 1.4
    if mode in {"evacuating"}:
        travel_m = 12.0 + index * 1.2
    travel_m = max(travel_m, BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M + 1.5 + index * 1.0)
    side_sign = _side_sign_for_desired(anchor.sample, anchor.position_enu_m)
    for signed_travel_m in (travel_m, -travel_m, travel_m * 0.75, -travel_m * 0.75):
        target_sample = LANES.resolve_edge_s(
            anchor.sample.edge_id,
            max(min_s, min(max_s, anchor.sample.s_m + signed_travel_m)),
        )
        target = _offset_from_lane(
            target_sample,
            side_sign * (LANE_HALF_WIDTH_M + anchor.offset_from_curb_m),
            GROUND_Z_M,
        )
        if dist_xy(anchor.position_enu_m, target) < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M:
            continue
        try:
            SPATIAL.validate_segment(
                anchor.position_enu_m,
                target,
                context="semantic contract pedestrian route",
                allow_road=False,
                allow_green=True,
            )
            return [target]
        except SpatialValidationError:
            continue
    return []


def _add_contract_vehicle(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    contract: EpisodeContract,
    index: int,
    occupied: list[list[float]],
) -> str:
    mode = _vehicle_mode_from_role(contract.vehicle_role)
    moving_mode = _moving_background_vehicle_mode(mode)
    sample = _find_free_lane_sample(scenes, events, occupied, mode=moving_mode, index=index)
    entity_id = f"bg_vehicle_{safe_id(scenario_id)}_{index + 1:02d}"
    asset = BACKGROUND_VEHICLE_ASSETS[index % len(BACKGROUND_VEHICLE_ASSETS)]
    start = _offset_from_lane(sample, 0.0, GROUND_Z_M)
    route = _semantic_vehicle_route(sample, moving_mode, index)
    if not route:
        raise RuntimeError(f"{scenario_id}: background vehicle {entity_id} lacks a valid continuous flow route")
    spec, scene = lane_entity(
        entity_id,
        asset,
        "vehicle",
        sample.edge_id,
        sample.s_m,
        0.0,
        start,
        sample.yaw_deg,
        moving_mode,
        route=route,
        visual_state=semantic_state(
            task_id=f"{scenario_id}.vehicle_background.{index + 1:02d}",
            role="semantic_background_vehicle",
            state_sequence=[moving_mode, moving_mode],
            semantic_role=contract.vehicle_role,
            mode=moving_mode,
            background_role="semantic_context",
        ),
        prefer_edge_hint=True,
    )
    scene["background_vehicle"] = {
        "policy": "semantic_event_contract_v1",
        "semantic_role": contract.vehicle_role,
        "contract_scenario_id": scenario_id,
    }
    scene["ground_flow_contract"] = _ground_flow_contract(
        "vehicle",
        BACKGROUND_VEHICLE_FLOW_SPEED_MPS,
        [start, *route],
        route_source="sumo_net_projected_to_traffic_bundle",
    )
    add(specs, scenes, (spec, scene))
    return entity_id


def _add_contract_pedestrian(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    contract: EpisodeContract,
    index: int,
    occupied: list[list[float]],
) -> str:
    origin = _contract_center(scenes, events)
    mode = _pedestrian_mode_from_role(contract.pedestrian_role)
    entity_id = f"bg_ped_{safe_id(scenario_id)}_{index + 1:02d}"
    anchor = None
    route: list[list[float]] = []
    checked = 0
    for candidate in _iter_free_sidewalk_anchors(
        origin,
        occupied,
        semantic="semantic_event_background_pedestrian",
        allow_green=True,
    ):
        checked += 1
        candidate_route = _semantic_ped_route(candidate, mode, index)
        if candidate_route:
            anchor = candidate
            route = candidate_route
            occupied.append(anchor.position_enu_m)
            break
        if checked >= 512:
            break
    if anchor is None or not route:
        raise RuntimeError(
            f"{scenario_id}: background pedestrian {entity_id} lacks a valid continuous sidewalk flow route "
            f"(checked {checked} anchors)"
        )
    moving_mode = "walking"
    spec, scene = world_entity(
        entity_id,
        "pedestrian.cityops.basic.v1",
        "pedestrian",
        anchor.position_enu_m,
        anchor.sample.yaw_deg,
        moving_mode,
        route=route,
        visual_state=semantic_state(
            task_id=f"{scenario_id}.pedestrian_background.{index + 1:02d}",
            role="semantic_background_pedestrian",
            state_sequence=[moving_mode, moving_mode],
            semantic_role=contract.pedestrian_role,
            mode=moving_mode,
            background_role="semantic_context",
        ),
    )
    scene["placement_mode"] = "sidewalk_anchor"
    scene["placement"] = {
        "lane_edge_id": anchor.sample.edge_id,
        "longitudinal_s": round(anchor.sample.s_m, 3),
        "offset_from_curb_m": round(anchor.offset_from_curb_m, 3),
        "lane_half_width_m": LANE_HALF_WIDTH_M,
        "resolved_lateral_from_center_m": round(anchor.resolved_lateral_from_center_m, 3),
        "placement_semantics": "semantic_event_background_pedestrian",
        "allow_green": True,
        "resolved_position_enu_m": anchor.position_enu_m,
        "rotation_deg": rot(anchor.sample.yaw_deg),
        "contract_scenario_id": scenario_id,
    }
    scene["background_pedestrian"] = {
        "policy": "semantic_event_contract_v1",
        "semantic_role": contract.pedestrian_role,
        "contract_scenario_id": scenario_id,
    }
    scene["ground_flow_contract"] = _ground_flow_contract(
        "pedestrian",
        BACKGROUND_PEDESTRIAN_FLOW_SPEED_MPS,
        [anchor.position_enu_m, *route],
        route_source="traffic_bundle_sidewalk",
    )
    add(specs, scenes, (spec, scene))
    return entity_id


def _inspect_racetrack_points(center: list[float], altitude_m: float, index_seed: int) -> tuple[list[float], list[list[float]]]:
    radius_x = 18.0 + (index_seed % 3) * 2.0
    radius_y = 11.0 + (index_seed % 2) * 2.0
    start = q(center[0] - radius_x, center[1] - radius_y, altitude_m)
    route = [
        q(center[0] + radius_x, center[1] - radius_y, altitude_m),
        q(center[0] + radius_x, center[1] + radius_y, altitude_m),
        q(center[0] - radius_x, center[1] + radius_y, altitude_m),
        q(center[0] - radius_x, center[1] - radius_y, altitude_m),
    ]
    return start, route


def _inspect_boundary_sample_loop(scenes: list[dict[str, Any]], events: list[dict[str, Any]], altitude_m: float) -> tuple[list[float], list[list[float]]] | None:
    boundary = _capture_boundary_from_points("inspect", scenes, events)
    polygon = [
        [float(point[0]), float(point[1])]
        for point in boundary.get("polygon_enu_m") or []
        if isinstance(point, list) and len(point) >= 2
    ]
    if len(polygon) < 3:
        return None
    center = list(boundary.get("center_enu_m") or [])
    if len(center) < 2:
        center = [
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        ]
    hfov = float(DEFAULT_UAV_SENSOR_PROFILE.get("hfov_deg") or DEFAULT_UAV_SENSOR_FOV_DEG)
    width = float(DEFAULT_UAV_SENSOR_PROFILE.get("width") or 1.0)
    height = float(DEFAULT_UAV_SENSOR_PROFILE.get("height") or 1.0)
    half_width_m = math.tan(math.radians(hfov / 2.0)) * float(altitude_m)
    vfov = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov / 2.0)) * (height / width)))
    half_height_m = math.tan(math.radians(vfov / 2.0)) * float(altitude_m)
    inset_margin_m = max(2.0, min(half_width_m, half_height_m) * 0.35)
    start = q(float(center[0]), float(center[1]), altitude_m)
    samples: list[list[float]] = []
    for index, point in enumerate(polygon):
        nxt = polygon[(index + 1) % len(polygon)]
        for sample in (point, [(point[0] + nxt[0]) * 0.5, (point[1] + nxt[1]) * 0.5]):
            dx = float(sample[0]) - float(center[0])
            dy = float(sample[1]) - float(center[1])
            length = math.hypot(dx, dy)
            if length > inset_margin_m:
                scale = (length - inset_margin_m) / length
                samples.append(q(float(center[0]) + dx * scale, float(center[1]) + dy * scale, altitude_m))
            else:
                samples.append(q(float(sample[0]), float(sample[1]), altitude_m))
    samples.append(start)
    route = samples + [start]
    if _route_length_m([start, *route]) < 80.0:
        radius_x = 18.0
        radius_y = 12.0
        route.extend(
            [
                q(start[0] + radius_x, start[1], altitude_m),
                q(start[0], start[1] + radius_y, altitude_m),
                q(start[0] - radius_x, start[1], altitude_m),
                q(start[0], start[1] - radius_y, altitude_m),
                start,
            ]
        )
    return start, route


def _route_length_m(points: list[list[float]]) -> float:
    length = 0.0
    for a, b in zip(points, points[1:]):
        length += math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2
        )
    return length


def _visible_physical_scene(scene: dict[str, Any]) -> bool:
    if scene.get("enabled") is False:
        return False
    asset = str(scene.get("logical_asset_id") or "")
    category = str(scene.get("category") or "")
    if asset == UAV_CORRIDOR_LOGICAL_ASSET_ID:
        return False
    if asset.startswith(("uav.", "vehicle.", "pedestrian.", "facility.", "prop.")):
        return True
    return category in {"uav", "vehicle", "pedestrian", "facility", "traffic_signal", "airspace_constraint", "hazard_zone"}


def _motion_points_for_entity(scene: dict[str, Any], events: list[dict[str, Any]]) -> list[list[float]]:
    entity_id = str(scene.get("entity_id") or "")
    points: list[list[float]] = []
    start = scene_pos(scene)
    if start:
        points.append(start)
    for waypoint in scene.get("route_waypoints_enu_m") or []:
        if isinstance(waypoint, list) and len(waypoint) >= 3:
            points.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
    for event_def in events:
        for action_def in event_def.get("actions") or []:
            params = action_params(action_def)
            if str(params.get("entity_id") or params.get("ped_id") or "") != entity_id:
                continue
            for waypoint in params.get("waypoints_enu_m") or []:
                if isinstance(waypoint, list) and len(waypoint) >= 3:
                    points.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
    return points


def _linked_intents_for_entity(scene: dict[str, Any], events: list[dict[str, Any]], contract: EpisodeContract) -> list[str]:
    entity_id = str(scene.get("entity_id") or "")
    linked: list[str] = []
    for event_def in events:
        intent = str(event_def.get("intent") or "")
        if not intent:
            continue
        targets = {str(target) for target in event_def.get("log_target_ids") or []}
        action_targets = {
            str(action_params(action).get("entity_id") or action_params(action).get("ped_id") or "")
            for action in event_def.get("actions") or []
        }
        if entity_id in targets or entity_id in action_targets:
            if intent not in linked:
                linked.append(intent)
    if linked:
        return linked
    return list(contract.required_intents or ("semantic_context",))


def apply_motion_contracts(scenes: list[dict[str, Any]], events: list[dict[str, Any]], contract: EpisodeContract) -> None:
    for scene in scenes:
        if not _visible_physical_scene(scene):
            continue
        points = _motion_points_for_entity(scene, events)
        displacement_m = _route_length_m(points) if len(points) >= 2 else 0.0
        asset = str(scene.get("logical_asset_id") or "")
        category = str(scene.get("category") or "")
        can_locomote = asset.startswith(("uav.", "vehicle.", "pedestrian.")) or category in {"uav", "vehicle", "pedestrian"}
        motion_kind = "kinematic_path" if can_locomote and displacement_m >= 0.5 else "state_animation"
        scene["motion_contract"] = {
            "schema": "semantic_motion_contract_v1",
            "required": True,
            "motion_kind": motion_kind,
            "semantic_link_required": True,
            "linked_intents": _linked_intents_for_entity(scene, events, contract),
            "min_displacement_m": 0.5 if motion_kind == "kinematic_path" else 0.0,
            "planned_displacement_m": round(displacement_m, 3),
        }


def _capture_boundary_entity(scene: dict[str, Any]) -> bool:
    capture_contract = dict(scene.get("capture_contract") or {})
    if str(capture_contract.get("boundary_role") or "") == "capture_boundary":
        return True
    asset_id = str(scene.get("logical_asset_id") or "")
    category = str(scene.get("category") or "")
    return asset_id.startswith("trigger.") or category in {"airspace_constraint", "hazard_zone"}


def _mobile_capture_boundary_subject(scene: dict[str, Any]) -> bool:
    asset_id = str(scene.get("logical_asset_id") or "")
    category = str(scene.get("category") or "")
    return asset_id.startswith(("uav.", "vehicle.", "pedestrian.")) or category in {"uav", "vehicle", "pedestrian"}


def _capture_boundary_anchor_entity_id(scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> str:
    scenes_by_id = {str(scene.get("entity_id") or ""): scene for scene in scenes}
    candidate_ids: list[str] = []
    for event_def in events:
        for target in event_def.get("log_target_ids") or []:
            candidate_ids.append(str(target))
        for action_def in event_def.get("actions") or []:
            params = action_params(action_def)
            candidate_ids.append(str(params.get("entity_id") or params.get("ped_id") or ""))
    candidate_ids.extend(str(scene.get("entity_id") or "") for scene in scenes)
    seen: set[str] = set()
    for entity_id in candidate_ids:
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        scene = scenes_by_id.get(entity_id)
        if not scene:
            continue
        asset_id = str(scene.get("logical_asset_id") or "")
        if asset_id == UAV_CORRIDOR_LOGICAL_ASSET_ID or asset_id.startswith("semantic."):
            continue
        if str((scene.get("initial_state") or {}).get("role") or "") == "U_inspect":
            continue
        if scene_pos(scene):
            return entity_id
    raise RuntimeError("Unable to anchor capture boundary to an existing event entity")


def _scene_polygon_xy(scene: dict[str, Any]) -> list[list[float]]:
    placement = dict(scene.get("placement") or {})
    polygon = placement.get("polygon_enu_m") or []
    result: list[list[float]] = []
    for point in polygon:
        if isinstance(point, list) and len(point) >= 2:
            result.append([round(float(point[0]), 3), round(float(point[1]), 3)])
    if result:
        return result
    center = scene_pos(scene)
    extent = placement.get("extent_m") or placement.get("size_m")
    if not center or not isinstance(extent, list) or len(extent) < 2:
        return []
    half_x = float(extent[0]) * (0.5 if "size_m" in placement and "extent_m" not in placement else 1.0)
    half_y = float(extent[1]) * (0.5 if "size_m" in placement and "extent_m" not in placement else 1.0)
    return [
        q(center[0] - half_x, center[1] - half_y, 0.0)[:2],
        q(center[0] + half_x, center[1] - half_y, 0.0)[:2],
        q(center[0] + half_x, center[1] + half_y, 0.0)[:2],
        q(center[0] - half_x, center[1] + half_y, 0.0)[:2],
    ]


def _capture_boundary_focus_points(scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[list[float]]:
    boundary_scenes = [scene for scene in scenes if _capture_boundary_entity(scene)]
    points: list[list[float]] = []
    for scene in boundary_scenes:
        polygon = _scene_polygon_xy(scene)
        if polygon:
            z = float((scene_pos(scene) or [0.0, 0.0, GROUND_Z_M])[2])
            points.extend(q(point[0], point[1], z) for point in polygon)
        else:
            position = scene_pos(scene)
            if position:
                points.append(position)
    if points:
        return points

    scenes_by_id = {str(scene.get("entity_id") or ""): scene for scene in scenes}

    def add_scene_point(entity_id: str) -> None:
        scene = scenes_by_id.get(entity_id)
        if not scene:
            return
        if str((scene.get("initial_state") or {}).get("role") or "") == "U_inspect":
            return
        if _mobile_capture_boundary_subject(scene):
            return
        position = scene_pos(scene)
        if position:
            points.append(position)

    for event_def in events:
        if str(event_def.get("event_id") or "").startswith("lifecycle_"):
            continue
        for target in event_def.get("log_target_ids") or []:
            add_scene_point(str(target))
        for action_def in event_def.get("actions") or []:
            params = action_params(action_def)
            add_scene_point(str(params.get("entity_id") or params.get("ped_id") or ""))
            for waypoint in params.get("waypoints_enu_m") or []:
                if isinstance(waypoint, list) and len(waypoint) >= 3:
                    points.append(q(float(waypoint[0]), float(waypoint[1]), float(waypoint[2])))
    return points or _contract_anchor_points(scenes, events)


def _capture_boundary_focus_center(scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[float]:
    points = _capture_boundary_focus_points(scenes, events)
    return [
        round(sum(point[0] for point in points) / len(points), 3),
        round(sum(point[1] for point in points) / len(points), 3),
        round(sum(point[2] for point in points) / len(points), 3),
    ]


def _capture_boundary_from_points(scenario_id: str, scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    boundary_scenes = [scene for scene in scenes if _capture_boundary_entity(scene)]
    if boundary_scenes:
        primary = boundary_scenes[0]
        center = scene_pos(primary) or _contract_center(scenes, events)
        polygon = _scene_polygon_xy(primary)
        return {
            "boundary_id": str(primary.get("entity_id") or f"capture_boundary_{safe_id(scenario_id)}"),
            "source": "scene_entity",
            "source_entity_id": str(primary.get("entity_id") or ""),
            "center_enu_m": q(center[0], center[1], center[2] if len(center) > 2 else 0.0),
            "polygon_enu_m": polygon,
        }
    points = _capture_boundary_focus_points(scenes, events)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    margin = 5.0
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin
    center = q((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, GROUND_Z_M)
    source_entity_id = _capture_boundary_anchor_entity_id(scenes, events)
    return {
        "boundary_id": f"capture_boundary_{safe_id(scenario_id)}",
        "source": "event_entity_envelope",
        "source_entity_id": source_entity_id,
        "center_enu_m": center,
        "polygon_enu_m": [
            [round(min_x, 3), round(min_y, 3)],
            [round(max_x, 3), round(min_y, 3)],
            [round(max_x, 3), round(max_y, 3)],
            [round(min_x, 3), round(max_y, 3)],
        ],
    }


def _point_in_polygon_xy(point: list[float], polygon: list[list[float]]) -> bool:
    x, y = float(point[0]), float(point[1])
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


def _segments_intersect_xy(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    def ccw(p1: list[float], p2: list[float], p3: list[float]) -> bool:
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def _route_crosses_polygon_xy(route: list[list[float]], polygon: list[list[float]]) -> bool:
    if not route or not polygon:
        return False
    if any(_point_in_polygon_xy(point, polygon) for point in route):
        return True
    return any(
        _segments_intersect_xy(a, b, polygon[index], polygon[(index + 1) % len(polygon)])
        for a, b in zip(route, route[1:])
        for index in range(len(polygon))
    )


def ensure_uav_routes_cross_capture_boundary(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    capture_boundary: dict[str, Any],
    contract: EpisodeContract,
) -> None:
    polygon = [
        [float(point[0]), float(point[1])]
        for point in capture_boundary.get("polygon_enu_m") or []
        if isinstance(point, list) and len(point) >= 2
    ]
    if not polygon:
        raise RuntimeError(f"{scenario_id}: capture boundary polygon missing before UAV crossing enforcement")
    center = list(capture_boundary.get("center_enu_m") or [])
    if len(center) < 2:
        center = [
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        ]
    spec_by_id = {str(spec.get("entity_id") or ""): spec for spec in specs}
    crossing_z = max(float(contract.inspect_altitude_m), 22.0)
    crossing_waypoint = q(float(center[0]), float(center[1]), crossing_z)
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    half_x = max(1.0, 0.5 * (max(xs) - min(xs)))
    half_y = max(1.0, 0.5 * (max(ys) - min(ys)))
    for scene in scenes:
        if not str(scene.get("logical_asset_id") or "").startswith("uav."):
            continue
        initial_state = dict(scene.get("initial_state") or {})
        if str(initial_state.get("role") or "") == "U_inspect":
            continue
        route_points = _motion_points_for_entity(scene, events)
        if _route_crosses_polygon_xy(route_points, polygon):
            continue
        route = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in scene.get("route_waypoints_enu_m") or []
            if isinstance(point, list) and len(point) >= 3
        ]
        route = [crossing_waypoint, *route]
        corridor_role = str(scene.get("uav_corridor_role") or initial_state.get("uav_corridor_role") or "")
        if corridor_role == "observer":
            current = scene_pos(scene) or crossing_waypoint
            dx = float(current[0]) - float(center[0])
            dy = float(current[1]) - float(center[1])
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                dx, dy, length = 1.0, 0.0, 1.0
            ux = dx / length
            uy = dy / length
            edge_scale = min(
                half_x / max(abs(ux), 1e-6),
                half_y / max(abs(uy), 1e-6),
            )
            approach = q(
                float(center[0]) + ux * (edge_scale + 18.0),
                float(center[1]) + uy * (edge_scale + 18.0),
                crossing_z,
            )
            update_world_entity_position(spec_by_id.get(str(scene.get("entity_id") or ""), {}), scene, approach)
        scene["route_waypoints_enu_m"] = route
        scene["boundary_crossing_injected"] = {
            "policy": "semantic_event_contract_v1",
            "capture_boundary_id": capture_boundary.get("boundary_id"),
            "reason": "non_inspect_uav_route_must_cross_capture_boundary",
        }
        spec = spec_by_id.get(str(scene.get("entity_id") or ""))
        if spec is not None:
            spec["movement_waypoints"] = route


def _scenario_pad_boundary_policy(scenario_id: str) -> str:
    inside_required_for = set(get_contract(scenario_id).pad_policy.inside_required_for)
    if inside_required_for.intersection({"pad_contention", "priority_landing_arbitration", "terminal_pad_queue"}):
        return "event_pads_inside_capture_boundary"
    return "pads_may_be_outside_capture_boundary"


def _inspect_route_candidates(center: list[float], altitude_m: float) -> list[list[float]]:
    candidates: list[list[float]] = []
    base = q(center[0], center[1], altitude_m)
    candidates.append(base)
    candidates.append(road_center_point(q(center[0], center[1], GROUND_Z_M), altitude_m))
    offsets = (
        (18.0, 0.0),
        (-18.0, 0.0),
        (0.0, 18.0),
        (0.0, -18.0),
        (24.0, 12.0),
        (-24.0, 12.0),
        (24.0, -12.0),
        (-24.0, -12.0),
        (36.0, 0.0),
        (0.0, 36.0),
        (-36.0, 0.0),
        (0.0, -36.0),
    )
    for dx, dy in offsets:
        candidates.append(road_center_point(q(center[0] + dx, center[1] + dy, GROUND_Z_M), altitude_m))
    return candidates


def _find_clear_inspect_lane_route(center: list[float], altitude_m: float) -> tuple[list[float], list[list[float]]] | None:
    nearby_samples = sorted(
        LANES.samples,
        key=lambda sample: (sample.x_m - float(center[0])) ** 2 + (sample.y_m - float(center[1])) ** 2,
    )[:120]
    ordered_samples: list[LaneSample] = []
    seen_edges: set[str] = set()
    for sample in nearby_samples:
        if sample.edge_id in seen_edges:
            continue
        seen_edges.add(sample.edge_id)
        ordered_samples.append(sample)
    ordered_samples.extend(nearby_samples)
    for sample in ordered_samples:
        min_s, max_s = LANES.edge_s_bounds(sample.edge_id)
        for span_m in (90.0, 70.0, 110.0, 130.0, 55.0):
            for bias_m in (0.0, -24.0, 24.0, -48.0, 48.0, -72.0, 72.0):
                s0 = max(min_s, min(max_s, sample.s_m - span_m * 0.5 + bias_m))
                s1 = max(min_s, min(max_s, sample.s_m + span_m * 0.5 + bias_m))
                if abs(s1 - s0) < 42.0:
                    continue
                start_sample = LANES.resolve_edge_s(sample.edge_id, s0)
                end_sample = LANES.resolve_edge_s(sample.edge_id, s1)
                for lateral_m in (0.0, 8.0, -8.0, 16.0, -16.0, 24.0, -24.0, 36.0, -36.0):
                    opposite_lateral_m = lateral_m + (8.0 if lateral_m <= 0.0 else -8.0)
                    loop = [
                        _offset_from_lane(start_sample, lateral_m, altitude_m),
                        _offset_from_lane(end_sample, lateral_m, altitude_m),
                        _offset_from_lane(end_sample, opposite_lateral_m, altitude_m),
                        _offset_from_lane(start_sample, opposite_lateral_m, altitude_m),
                        _offset_from_lane(start_sample, lateral_m, altitude_m),
                    ]
                    if not UAV_CORRIDORS.buildings.route_clear_at_altitude(loop, altitude_m):
                        continue
                    if _route_length_m(loop) >= 80.0:
                        return loop[0], loop[1:]
    return None


def _find_inspect_route(scenario_id: str, scenes: list[dict[str, Any]], events: list[dict[str, Any]], contract: EpisodeContract) -> tuple[list[float], list[list[float]]]:
    center = _capture_boundary_focus_center(scenes, events)
    boundary_loop = _inspect_boundary_sample_loop(scenes, events, contract.inspect_altitude_m)
    if boundary_loop is not None:
        start, route = boundary_loop
        planned = [start, *route]
        if _route_length_m(planned) >= 80.0:
            return start, route
    seen: set[tuple[float, float, float]] = set()
    for index_seed, candidate_center in enumerate(_inspect_route_candidates(center, contract.inspect_altitude_m)):
        key = (round(float(candidate_center[0]), 3), round(float(candidate_center[1]), 3), round(float(candidate_center[2]), 3))
        if key in seen:
            continue
        seen.add(key)
        for attempt_seed in range(8):
            start, route = _inspect_racetrack_points(candidate_center, contract.inspect_altitude_m, index_seed + attempt_seed)
            planned = [start, *route]
            if not UAV_CORRIDORS.buildings.route_clear_at_altitude(planned, contract.inspect_altitude_m):
                continue
            if _route_length_m(planned) < 80.0:
                continue
            return start, route
    lane_route = _find_clear_inspect_lane_route(center, contract.inspect_altitude_m)
    if lane_route is not None:
        return lane_route
    raise RuntimeError(
        f"{scenario_id}: unable to place deterministic U_inspect route at {contract.inspect_code}"
    )


def _add_contract_inspect_uav(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    contract: EpisodeContract,
) -> str:
    if any((scene.get("initial_state") or {}).get("role") == "U_inspect" for scene in scenes):
        return str(next(scene["entity_id"] for scene in scenes if (scene.get("initial_state") or {}).get("role") == "U_inspect"))
    start, route = _find_inspect_route(scenario_id, scenes, events, contract)
    entity_id = f"u_inspect_{safe_id(scenario_id)}"
    spec, scene = world_entity(
        entity_id,
        "uav.inspect.quad.v1",
        "uav",
        start,
        45.0,
        "inspect_racetrack",
        route=route,
        visual_state=semantic_state(
            task_id=f"{scenario_id}.u_inspect",
            role="U_inspect",
            state_sequence=["takeoff", "inspect_racetrack", "orbit", "land"],
            semantic_role="long-lived UAV inspect view replacing high overview",
            mode="inspect_racetrack",
            assigned_altitude_m=contract.inspect_altitude_m,
            uav_corridor_role="inspect_observer",
            inspect_altitude_code=contract.inspect_code,
            min_path_length_m=80.0,
            full_episode_presence=True,
        ),
    )
    scene["uav_corridor_role"] = "inspect_observer"
    scene["contract_inspect_uav"] = {
        "policy": "semantic_event_contract_v1",
        "corridor_role": "inspect_observer",
        "corridor_policy": contract.inspect.corridor_policy,
        "motion_policy": "fixed_altitude_loop_observes_capture_boundary",
        "inspect_altitude_m": contract.inspect_altitude_m,
        "inspect_altitude_code": contract.inspect_code,
        "sensor_fov_deg": DEFAULT_UAV_SENSOR_FOV_DEG,
        "sensor_fov_source": "Plugins/SumoImporter/Scripts/episode_capture_presets.json:uav_cameras.default[0].fov_degrees",
        "sensor_profile": dict(DEFAULT_UAV_SENSOR_PROFILE),
        "min_path_length_m": 80.0,
        "fixed_altitude_loop": True,
        "planned_route_enu_m": [start, *route],
        "loop_route_enu_m": [start, *route],
    }
    add(specs, scenes, (spec, scene))
    return entity_id


def _add_contract_facility(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    contract: EpisodeContract,
    index: int,
) -> str:
    center = _contract_center(scenes, events)
    angle = math.radians((index * 73) % 360)
    offset = 10.0 + index * 3.0
    desired_pos = q(center[0] + math.cos(angle) * offset, center[1] + math.sin(angle) * offset, GROUND_Z_M)
    facility_cycle = (
        ("facility.landing_pad.visible.v1", "facility", "available"),
        ("facility.charger.cityops.v1", "facility", "available"),
        ("facility.radio.base_tower.v1", "facility", "online"),
        ("facility.barrier.basic", "facility", "available"),
    )
    asset, category, mode = facility_cycle[index % len(facility_cycle)]
    entity_id = f"contract_facility_{safe_id(scenario_id)}_{index + 1:02d}"
    state = semantic_state(
        task_id=f"{scenario_id}.facility_context.{index + 1:02d}",
        role="semantic_facility",
        state_sequence=[mode],
        semantic_role="visible facility context required by low-altitude event-chain contract",
        mode=mode,
    )
    if asset == "facility.landing_pad.visible.v1":
        pad_pos, pad_yaw, pad_anchor = landing_pad_pose(
            desired_pos,
            context=f"{scenario_id} contract landing pad {index + 1:02d}",
        )
        spec, scene = pad_entity(
            entity_id,
            pad_pos,
            f"contract_{safe_id(scenario_id)}_{index + 1:02d}",
            "contract",
            pad_yaw,
            pad_anchor,
        )
        spec["visual_state"] = {**(spec.get("visual_state") or {}), **state}
        scene["initial_state"] = {**(scene.get("initial_state") or {}), **state}
    else:
        pos = road_center_point(desired_pos)
        spec, scene = world_entity(
            entity_id,
            asset,
            category,
            pos,
            float((index * 45) % 360),
            mode,
            visual_state=state,
        )
    scene["contract_facility"] = {
        "policy": "semantic_event_contract_v1",
        "contract_scenario_id": scenario_id,
    }
    add(specs, scenes, (spec, scene))
    return entity_id


def apply_semantic_event_contract(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> EpisodeContract:
    contract = get_contract(scenario_id)
    payload = contract_payload(contract)
    parameters["semantic_event_contract"] = payload
    parameters["target_uav_count"] = max(int(parameters.get("target_uav_count") or 0), contract.uav)
    parameters["target_vehicle_count"] = contract.vehicle
    parameters["target_pedestrian_count"] = contract.pedestrian
    parameters["target_facility_count"] = contract.facility
    parameters["target_logical_count"] = contract.logical
    parameters.setdefault("_pending_validation_rules", []).append(
        {
            "rule": "semantic_event_contract",
            "contract": payload,
            "description": "Episode must match the low-altitude semantic event-chain contract exactly after generation",
        }
    )

    occupied = [scene_pos(scene) for scene in scenes if scene_pos(scene)]
    _add_contract_inspect_uav(scenario_id, specs, scenes, events, contract)

    counts = _scene_counts(scenes)
    while counts["vehicle"] < contract.vehicle:
        _add_contract_vehicle(
            scenario_id,
            specs,
            scenes,
            events,
            contract,
            counts["vehicle"],
            occupied,
        )
        counts = _scene_counts(scenes)
    while counts["pedestrian"] < contract.pedestrian:
        _add_contract_pedestrian(
            scenario_id,
            specs,
            scenes,
            events,
            contract,
            counts["pedestrian"],
            occupied,
        )
        counts = _scene_counts(scenes)
    while counts["facility"] < contract.facility:
        _add_contract_facility(
            scenario_id,
            specs,
            scenes,
            events,
            contract,
            counts["facility"],
        )
        counts = _scene_counts(scenes)
    ensure_contract_background_roles(scenario_id, specs, scenes, events, contract)
    ensure_background_ground_flow_routes(scenario_id, specs, scenes)
    return contract


def ensure_contract_background_roles(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    contract: EpisodeContract,
) -> None:
    spec_by_id = {str(spec.get("entity_id") or ""): spec for spec in specs}
    event_target_ids: set[str] = set()
    for event_def in events:
        severity = str(event_def.get("log_severity") or "").lower()
        if severity not in {"warning", "critical"}:
            continue
        for target_id in event_def.get("log_target_ids") or []:
            if target_id:
                event_target_ids.add(str(target_id))
        for action_def in event_def.get("actions") or []:
            params = action_params(action_def)
            target_id = str(params.get("entity_id") or params.get("ped_id") or "")
            if target_id:
                event_target_ids.add(target_id)
    missing = {
        "vehicle": not any(
            str((scene.get("initial_state") or {}).get("role") or "") == "semantic_background_vehicle"
            for scene in scenes
        ),
        "pedestrian": not any(
            str((scene.get("initial_state") or {}).get("role") or "") == "semantic_background_pedestrian"
            for scene in scenes
        ),
    }
    role_payload = {
        "vehicle": {
            "role": "semantic_background_vehicle",
            "semantic_role": contract.vehicle_role,
            "state_sequence": [_vehicle_mode_from_role(contract.vehicle_role), "traffic_flow"],
            "background_key": "background_vehicle",
            "task_suffix": "vehicle_background.01",
        },
        "pedestrian": {
            "role": "semantic_background_pedestrian",
            "semantic_role": contract.pedestrian_role,
            "state_sequence": [_pedestrian_mode_from_role(contract.pedestrian_role), _pedestrian_mode_from_role(contract.pedestrian_role)],
            "background_key": "background_pedestrian",
            "task_suffix": "pedestrian_background.01",
        },
    }
    for kind, is_missing in missing.items():
        if not is_missing:
            continue
        candidates = [
            scene
            for scene in scenes
            if str(scene.get("category") or "") == kind
            or str(scene.get("logical_asset_id") or "").startswith(f"{kind}.")
        ]
        if not candidates:
            raise RuntimeError(f"{scenario_id}: contract requires background {kind}, but no {kind} scene exists")
        non_event_candidates = [
            scene
            for scene in candidates
            if str(scene.get("entity_id") or "") not in event_target_ids
        ]
        selected = sorted(non_event_candidates or candidates, key=lambda item: str(item.get("entity_id") or ""))[0]
        payload = role_payload[kind]
        state = dict(selected.get("initial_state") or {})
        state.update(
            {
                "task_id": f"{scenario_id}.{payload['task_suffix']}",
                "role": payload["role"],
                "semantic_role": payload["semantic_role"],
                "state_sequence": payload["state_sequence"],
                "mode": payload["state_sequence"][0],
            }
        )
        selected["initial_state"] = state
        selected[payload["background_key"]] = {
            "policy": "semantic_event_contract_v1",
            "semantic_role": payload["semantic_role"],
            "contract_scenario_id": scenario_id,
        }
        spec = spec_by_id.get(str(selected.get("entity_id") or ""))
        if spec is not None:
            spec["visual_state"] = {**(spec.get("visual_state") or {}), **state}


def _pedestrian_flow_seed_route(scene: dict[str, Any], index: int) -> list[list[float]]:
    start = scene_pos(scene)
    if not start:
        return []
    placement = dict(scene.get("placement") or {})
    edge_id = str(placement.get("lane_edge_id") or "")
    if edge_id and edge_id in LANES.by_edge:
        s_m = float(placement.get("longitudinal_s") or LANES.nearest_to_xy(start[0], start[1]).s_m)
        sample = LANES.resolve_edge_s(edge_id, s_m)
    else:
        sample = LANES.nearest_to_xy(start[0], start[1])
    lateral = float(placement.get("resolved_lateral_from_center_m") or 0.0)
    side_sign = 1.0 if lateral >= 0.0 else -1.0
    offset_from_curb = max(float(placement.get("offset_from_curb_m") or SIDEWALK_MIN_OFFSET_FROM_CURB_M), SIDEWALK_MIN_OFFSET_FROM_CURB_M)
    min_s, max_s = LANES.edge_s_bounds(sample.edge_id)
    travel_m = max(7.0 + index * 1.5, BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M + 1.5 + index * 1.0)
    for signed_travel_m in (travel_m, -travel_m, travel_m * 0.75, -travel_m * 0.75):
        target_sample = LANES.resolve_edge_s(sample.edge_id, max(min_s, min(max_s, sample.s_m + signed_travel_m)))
        target = _offset_from_lane(
            target_sample,
            side_sign * (LANE_HALF_WIDTH_M + offset_from_curb),
            GROUND_Z_M,
        )
        if dist_xy(start, target) < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M:
            continue
        try:
            SPATIAL.validate_segment(
                start,
                target,
                context="background pedestrian continuous flow route",
                allow_road=False,
                allow_green=bool(placement.get("allow_green", True)),
            )
            return [target]
        except SpatialValidationError:
            continue
    return []


def ensure_background_ground_flow_routes(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
) -> None:
    spec_by_id = {str(spec.get("entity_id") or ""): spec for spec in specs}
    for index, scene in enumerate(scenes):
        entity_id = str(scene.get("entity_id") or "")
        state = dict(scene.get("initial_state") or {})
        role = str(state.get("role") or "")
        if role not in {"semantic_background_vehicle", "semantic_background_pedestrian"}:
            continue
        start = scene_pos(scene)
        if not start:
            raise RuntimeError(f"{scenario_id}: background ground-flow actor lacks start position: {entity_id}")
        current_route = [
            q(float(item[0]), float(item[1]), float(item[2] if len(item) > 2 else GROUND_Z_M))
            for item in scene.get("route_waypoints_enu_m") or []
            if isinstance(item, list) and len(item) >= 2
        ]
        if role == "semantic_background_vehicle":
            mode = _moving_background_vehicle_mode(str(state.get("mode") or "traffic_flow"))
            if not current_route or _route_xy_span(start, current_route) < BACKGROUND_VEHICLE_MIN_ROUTE_SPAN_M:
                placement = dict(scene.get("placement") or {})
                edge_id = str(placement.get("edge_id") or "")
                sample = (
                    LANES.resolve_edge_s(edge_id, float(placement.get("longitudinal_s") or 0.0))
                    if edge_id in LANES.by_edge
                    else LANES.nearest_to_xy(start[0], start[1])
                )
                current_route = _semantic_vehicle_route(sample, mode, index)
            route = current_route
            if not route or _route_xy_span(start, route) < BACKGROUND_VEHICLE_MIN_ROUTE_SPAN_M:
                raise RuntimeError(f"{scenario_id}: background vehicle {entity_id} lacks continuous flow route")
            state.update({"mode": mode, "state_sequence": [mode, mode]})
            scene["ground_flow_contract"] = _ground_flow_contract(
                "vehicle",
                BACKGROUND_VEHICLE_FLOW_SPEED_MPS,
                [start, *route],
                route_source="sumo_net_projected_to_traffic_bundle",
            )
        else:
            mode = "walking"
            if not current_route or _route_xy_span(start, current_route) < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M:
                current_route = _pedestrian_flow_seed_route(scene, index)
            route = current_route
            if not route or _route_xy_span(start, route) < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M:
                raise RuntimeError(f"{scenario_id}: background pedestrian {entity_id} lacks continuous sidewalk flow route")
            state.update({"mode": mode, "activity_type": mode, "state_sequence": [mode, mode]})
            scene["ground_flow_contract"] = _ground_flow_contract(
                "pedestrian",
                BACKGROUND_PEDESTRIAN_FLOW_SPEED_MPS,
                [start, *route],
                route_source="traffic_bundle_sidewalk",
            )
        scene["initial_state"] = state
        scene["route_waypoints_enu_m"] = route
        spec = spec_by_id.get(entity_id)
        if spec is not None:
            spec["movement_waypoints"] = route
            spec["visual_state"] = {**(spec.get("visual_state") or {}), **state}


def _scene_extent_radius_m(scene: dict[str, Any]) -> float:
    placement = dict(scene.get("placement") or {})
    asset = str(scene.get("logical_asset_id") or "")
    category = str(scene.get("category") or "")
    mode = str(scene.get("placement_mode") or "")
    if mode == "box_volume":
        extent = placement.get("extent_m") or [0.0, 0.0, 0.0]
        try:
            z_center = float((placement.get("center_enu_m") or placement.get("resolved_position_enu_m") or [0.0, 0.0, 0.0])[2])
            z_extent = float(extent[2] if len(extent) > 2 else 0.0)
        except (TypeError, ValueError):
            z_center = 0.0
            z_extent = 0.0
        if z_center - z_extent > 8.0:
            return 0.0
        return max(float(extent[0] if len(extent) > 0 else 0.0), float(extent[1] if len(extent) > 1 else 0.0))
    if mode == "polygon_prism":
        polygon = placement.get("polygon_enu_m") or []
        center = scene_pos(scene)
        if center and polygon:
            return max((dist_xy(center, point) for point in polygon if isinstance(point, list) and len(point) >= 2), default=0.0)
    if asset.startswith("vehicle."):
        return 2.3
    if asset.startswith("pedestrian."):
        return 0.8
    if asset.startswith("uav."):
        return 3.0
    if asset.startswith("prop.roadwork."):
        return 2.0
    if asset == "semantic.spawn_zone" or category == "crowd_anchor":
        return 8.0
    if asset == "facility.landing_pad.visible.v1":
        return 5.0
    if asset.startswith("facility.") or asset.startswith("prop."):
        return 3.0
    return 0.0


def _background_vehicle_blockers(scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[tuple[str, list[float], float]]:
    blockers: list[tuple[str, list[float], float]] = []
    for scene in scenes:
        position = scene_pos(scene)
        if not position:
            continue
        asset = str(scene.get("logical_asset_id") or "")
        radius = _scene_extent_radius_m(scene)
        if radius <= 0.0:
            continue
        margin = 2.0
        if asset.startswith("vehicle."):
            margin = 3.0
        elif asset.startswith("pedestrian."):
            margin = 5.0
        elif asset.startswith("uav."):
            margin = 8.0
        elif asset == "facility.landing_pad.visible.v1":
            margin = 8.0
        elif asset.startswith("trigger.") or str(scene.get("category") or "") in {"hazard_zone", "airspace_constraint"}:
            margin = 5.0
        elif asset.startswith("prop.roadwork."):
            margin = 4.0
        elif asset == "semantic.spawn_zone" or str(scene.get("category") or "") == "crowd_anchor":
            margin = 6.0
        blockers.append((str(scene.get("entity_id") or ""), position, radius + margin))

    for point in event_action_points(events):
        if point[2] <= 6.0:
            blockers.append(("event_action_ground_point", point, 4.0))
        elif point[2] <= 12.0:
            blockers.append(("event_action_low_air_point", point, 5.0))
    return blockers


def _near_existing_lane_vehicle(sample: LaneSample, scenes: list[dict[str, Any]], min_s_gap_m: float = 7.5) -> bool:
    for scene in scenes:
        if not str(scene.get("logical_asset_id") or "").startswith("vehicle."):
            continue
        placement = dict(scene.get("placement") or {})
        if scene.get("placement_mode") == "lane_anchor" and placement.get("edge_id") == sample.edge_id:
            try:
                if abs(float(placement.get("longitudinal_s")) - float(sample.s_m)) < min_s_gap_m:
                    return True
            except (TypeError, ValueError):
                pass
        position = scene_pos(scene)
        if position and dist_xy([sample.x_m, sample.y_m, GROUND_Z_M], position) < 5.0:
            return True
    return False


def _background_vehicle_candidate_samples(scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[LaneSample]:
    reference_points = [point for point in collect_scene_bound_points(ScenarioBundle("", Path("."), "", "", [], scenes, {}, events)) if point[2] <= 90.0]
    if not reference_points:
        reference_points = [[WORLD_OFFSET_X_M, WORLD_OFFSET_Y_M, GROUND_Z_M]]
    cx = sum(point[0] for point in reference_points) / len(reference_points)
    cy = sum(point[1] for point in reference_points) / len(reference_points)
    nearby_raw = sorted(LANES.samples, key=lambda sample: (sample.x_m - cx) ** 2 + (sample.y_m - cy) ** 2)
    nearby: list[LaneSample] = []
    seen_nearby_edges: set[str] = set()
    for sample in nearby_raw:
        if sample.edge_id in seen_nearby_edges:
            continue
        seen_nearby_edges.add(sample.edge_id)
        nearby.append(sample)
        if len(nearby) >= 160:
            break
    nearby.extend(nearby_raw[:240])
    result: list[LaneSample] = []
    seen: set[tuple[str, int]] = set()
    for base in nearby:
        min_s, max_s = LANES.edge_s_bounds(base.edge_id)
        for offset_m in (0.0, -18.0, 18.0, -32.0, 32.0, -48.0, 48.0, -64.0, 64.0, -82.0, 82.0):
            target_s = max(min_s, min(max_s, base.s_m + offset_m))
            sample = LANES.resolve_edge_s(base.edge_id, target_s)
            key = (sample.edge_id, int(round(sample.s_m)))
            if key in seen:
                continue
            seen.add(key)
            result.append(sample)
    return result


def add_background_vehicles(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> None:
    policy = background_vehicle_policy_for(scenario_id)
    if policy is None or policy.add_count <= 0:
        return
    existing_bg = [scene for scene in scenes if str(scene.get("entity_id") or "").startswith("bg_vehicle_")]
    if existing_bg:
        return

    blockers = _background_vehicle_blockers(scenes, events)
    selected: list[tuple[LaneSample, list[list[float]]]] = []
    attempts = 0
    for sample in _background_vehicle_candidate_samples(scenes, events):
        attempts += 1
        if not _lane_has_route_span(sample, BACKGROUND_VEHICLE_MIN_ROUTE_SPAN_M):
            continue
        candidate_pos = _offset_from_lane(sample, 0.0, GROUND_Z_M)
        if SPATIAL.validation_errors_for_point(
            candidate_pos,
            context=f"{scenario_id}: background vehicle candidate",
            allow_road=True,
            allow_green=False,
        ):
            continue
        if _near_existing_lane_vehicle(sample, scenes):
            continue
        if any(dist_xy(candidate_pos, point) < clearance_m for _label, point, clearance_m in blockers):
            continue
        if any(dist_xy(candidate_pos, _offset_from_lane(other, 0.0, GROUND_Z_M)) < 8.0 for other, _route in selected):
            continue
        route = _semantic_vehicle_route(sample, "traffic_flow", len(selected))
        if not route:
            continue
        selected.append((sample, route))
        if len(selected) >= policy.add_count:
            break
    if len(selected) < policy.add_count:
        raise RuntimeError(
            f"{scenario_id}: unable to place {policy.add_count} background vehicles without overlay/collision conflicts "
            f"(placed {len(selected)}, checked {attempts} lane samples)"
        )

    added_ids: list[str] = []
    for index, (sample, route) in enumerate(selected):
        entity_id = f"bg_vehicle_{safe_id(scenario_id)}_{index + 1:02d}"
        asset = BACKGROUND_VEHICLE_ASSETS[index % len(BACKGROUND_VEHICLE_ASSETS)]
        spec, scene = lane_entity(
            entity_id,
            asset,
            "vehicle",
            sample.edge_id,
            sample.s_m,
            0.0,
            _offset_from_lane(sample, 0.0, GROUND_Z_M),
            sample.yaw_deg,
            "traffic_flow",
            route=route,
            visual_state={
                "mode": "traffic_flow",
                "state_sequence": ["traffic_flow", "traffic_flow"],
                "background_role": "road_context",
            },
            prefer_edge_hint=True,
        )
        scene["background_vehicle"] = {
            "policy": "road_context_non_event_actor_v1",
            "reason": policy.reason,
            "min_clearance_rule": "background_vehicle_clearance",
        }
        start = _offset_from_lane(sample, 0.0, GROUND_Z_M)
        scene["ground_flow_contract"] = _ground_flow_contract(
            "vehicle",
            BACKGROUND_VEHICLE_FLOW_SPEED_MPS,
            [start, *route],
            route_source="sumo_net_projected_to_traffic_bundle",
        )
        add(specs, scenes, (spec, scene))
        added_ids.append(entity_id)

    parameters["background_vehicle_policy"] = {
        "policy": "road_context_non_event_actor_v1",
        "requested_count": policy.add_count,
        "added_entity_ids": added_ids,
        "reason": policy.reason,
    }
    parameters.setdefault("_pending_validation_rules", []).append(
        {
            "rule": "background_vehicle_clearance",
            "min_count": policy.add_count,
            "entity_ids": added_ids,
            "description": "Background road vehicles must stay clear of overlays, logical regions, pads, pedestrians, and event motion points",
        }
    )


def camera_for(cx: float, cy: float, z: float = 75.0) -> list[dict[str, Any]]:
    return [
        {
            "camera_id": "demo_high_overview",
            "placement_mode": "world_pose",
            "placement": {
                "position_enu_m": q(cx, cy, z),
                "rotation_deg": {"pitch_deg": -70.0, "yaw_deg": 0.0},
            },
            "fov_deg": 90.0,
        }
    ]


def base_rules(entity_ids: list[str], scenario_id: str) -> list[dict[str, Any]]:
    rules = []
    for entity_id in entity_ids[:2]:
        rules.append(
            {
                "rule": "entity_resolvable",
                "entity_id": entity_id,
                "description": f"{entity_id} is declared before event_script references it in {scenario_id}",
            }
        )
    return rules


def asset_rules(scene_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rule": "asset_in_catalog",
            "entity_id": e["entity_id"],
            "logical_asset_id": e["logical_asset_id"],
            "description": "Asset ID must match Config/LowAltitude/asset_catalog.json",
        }
        for e in scene_entities[:2]
    ]


def action_params(action_def: dict[str, Any]) -> dict[str, Any]:
    params = dict(action_def.get("params") or action_def)
    params.setdefault("type", action_def.get("type"))
    return params


def write_action_params(action_def: dict[str, Any], params: dict[str, Any]) -> None:
    if isinstance(action_def.get("params"), dict):
        action_def["params"].update(params)
    else:
        action_def.update(params)


def path_length_m(waypoints: list[list[float]]) -> float:
    length = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        length += math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (float(a[2]) - float(b[2])) ** 2)
    return length


def move_duration_ticks(action_def: dict[str, Any]) -> int:
    params = action_params(action_def)
    waypoints = params.get("waypoints_enu_m") or []
    if len(waypoints) < 2:
        return 0
    velocity = max(0.1, float(params.get("velocity_mps", 1.0)))
    return int(math.ceil(path_length_m(waypoints) / velocity * SCRIPT_TICK_HZ * DELAY_SAFETY_FACTOR))


def enforce_physical_delays(events: list[dict[str, Any]]) -> None:
    actions_by_event = {event_def["event_id"]: list(event_def.get("actions") or []) for event_def in events}
    for event_def in events:
        trigger = event_def.get("trigger") or {}
        if trigger.get("type") != "event_fired_after":
            continue
        prior_event_id = trigger.get("event_ref")
        if not prior_event_id:
            continue
        required_delay = max(
            (move_duration_ticks(action_def) for action_def in actions_by_event.get(prior_event_id, []) if action_params(action_def).get("type") == "move_entity"),
            default=0,
        )
        if required_delay > int(trigger.get("delay_ticks", 0)):
            trigger["delay_ticks"] = required_delay


def event_action_points(events: list[dict[str, Any]]) -> list[list[float]]:
    points: list[list[float]] = []
    for event_def in events:
        for action_def in event_def.get("actions") or []:
            params = action_params(action_def)
            for key in ("position_enu_m", "spawn_origin_enu_m"):
                point = params.get(key)
                if isinstance(point, list) and len(point) >= 3:
                    points.append([float(point[0]), float(point[1]), float(point[2])])
            for waypoint in params.get("waypoints_enu_m") or []:
                if isinstance(waypoint, list) and len(waypoint) >= 3:
                    points.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
    return points


def _entity_motion_points(entity_id: str, scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[list[float]]:
    points: list[list[float]] = []
    for scene in scenes:
        if str(scene.get("entity_id") or "") != entity_id:
            continue
        position = scene_pos(scene)
        if position:
            points.append(position)
        for waypoint in scene.get("route_waypoints_enu_m") or []:
            if isinstance(waypoint, list) and len(waypoint) >= 3:
                points.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
    for event_def in events:
        for action_def in event_def.get("actions") or []:
            params = action_params(action_def)
            if params.get("type") != "move_entity" or str(params.get("entity_id") or "") != entity_id:
                continue
            for waypoint in params.get("waypoints_enu_m") or []:
                if isinstance(waypoint, list) and len(waypoint) >= 3:
                    points.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
    return points


def normalize_proximity_trigger_reachability(
    scenario_id: str,
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> None:
    adjustments: list[dict[str, Any]] = []
    for event_def in events:
        trigger = event_def.get("trigger") or {}
        if trigger.get("type") != "entity_proximity":
            continue
        entity_a = str(trigger.get("entity_a") or "")
        entity_b = str(trigger.get("entity_b") or "")
        points_a = _entity_motion_points(entity_a, scenes, events)
        points_b = _entity_motion_points(entity_b, scenes, events)
        if not points_a or not points_b:
            raise RuntimeError(f"{scenario_id}: proximity trigger {event_def.get('event_id')} lacks deterministic motion points")
        metric = str(trigger.get("metric") or "3d").lower()
        best: tuple[float, float, float] | None = None
        for a in points_a:
            for b in points_b:
                horizontal = math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
                vertical = abs(float(a[2]) - float(b[2]))
                dist3 = math.sqrt(horizontal * horizontal + vertical * vertical)
                value = horizontal if metric == "xy" else dist3
                if metric == "xy_plus_z":
                    value = max(horizontal, vertical)
                if best is None or value < best[0]:
                    best = (value, horizontal, vertical)
        if best is None:
            raise RuntimeError(f"{scenario_id}: proximity trigger {event_def.get('event_id')} has no deterministic distance sample")
        current_distance = float(trigger.get("distance_m") or 0.0)
        if metric == "xy_plus_z":
            current_h = float(trigger.get("horizontal_distance_m") or current_distance)
            current_v = float(trigger.get("vertical_distance_m") or current_distance)
            new_h = max(current_h, round(best[1] + 2.0, 3))
            new_v = max(current_v, round(best[2] + 2.0, 3))
            if new_h > 50.0 or new_v > 50.0:
                raise RuntimeError(f"{scenario_id}: proximity trigger {event_def.get('event_id')} cannot be normalized within 50m")
            if new_h != current_h or new_v != current_v:
                trigger["original_horizontal_distance_m"] = current_h
                trigger["original_vertical_distance_m"] = current_v
                trigger["horizontal_distance_m"] = new_h
                trigger["vertical_distance_m"] = new_v
                trigger["distance_m"] = max(float(trigger.get("distance_m") or 0.0), new_h)
                adjustments.append({"event_id": event_def.get("event_id"), "entity_a": entity_a, "entity_b": entity_b, "horizontal_distance_m": new_h, "vertical_distance_m": new_v})
            continue
        new_distance = max(current_distance, round(best[0] + 3.0, 3))
        if new_distance > 50.0:
            raise RuntimeError(f"{scenario_id}: proximity trigger {event_def.get('event_id')} cannot be normalized within 50m")
        if new_distance != current_distance:
            trigger["original_distance_m"] = current_distance
            trigger["distance_m"] = new_distance
            adjustments.append({"event_id": event_def.get("event_id"), "entity_a": entity_a, "entity_b": entity_b, "distance_m": new_distance})
    if adjustments:
        parameters["proximity_reachability_normalization"] = adjustments


def collect_scene_bound_points(bundle: ScenarioBundle) -> list[list[float]]:
    points: list[list[float]] = []
    for entity in bundle.scene_entities:
        position = scene_pos(entity)
        if position:
            points.append(position)
        for waypoint in entity.get("route_waypoints_enu_m") or []:
            if isinstance(waypoint, list) and len(waypoint) >= 3:
                points.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
    points.extend(event_action_points(bundle.events))
    return points


def default_cameras_for_bundle(scene_entities: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[list[float]] = []
    for entity in scene_entities:
        position = scene_pos(entity)
        if position:
            points.append(position)
        for waypoint in entity.get("route_waypoints_enu_m") or []:
            if isinstance(waypoint, list) and len(waypoint) >= 3:
                points.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
    points.extend(event_action_points(events))
    if not points:
        return camera_for(WORLD_OFFSET_X_M, WORLD_OFFSET_Y_M)
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    radius = max(math.hypot(point[0] - cx, point[1] - cy) for point in points)
    altitude = max(75.0, min(180.0, radius * 1.45 + 45.0))
    return camera_for(cx, cy, altitude)


def uav_home_pad_pose(mission_start_enu: list[float], slot_index: int) -> tuple[list[float], float]:
    sample = LANES.nearest_to_xy(float(mission_start_enu[0]), float(mission_start_enu[1]))
    base_offset = LANE_HALF_WIDTH_M + GATHERING_MIN_OFFSET_FROM_CURB_M + 2.0
    side = 1.0 if slot_index % 2 == 0 else -1.0
    ring = slot_index // 2
    lateral = side * (base_offset + ring * 4.0)
    desired_position = _offset_from_lane(sample, lateral, GROUND_Z_M)
    pad_position, yaw, _ = landing_pad_pose(
        desired_position,
        context=f"home UAV pad slot {slot_index}",
        offset_from_curb_m=GATHERING_MIN_OFFSET_FROM_CURB_M + ring * 4.0,
    )
    return pad_position, yaw


def update_world_entity_position(spec: dict[str, Any], scene: dict[str, Any], position_enu: list[float]) -> None:
    spec["initial_pos_enu"] = position_enu
    placement = scene.setdefault("placement", {})
    placement["position_enu_m"] = position_enu
    placement["resolved_position_enu_m"] = position_enu


def terminal_uav_action(action_id: str, final_z_m: float) -> bool:
    lowered = action_id.lower()
    if final_z_m <= UAV_LANDING_ALTITUDE_MAX_M:
        return True
    return any(token in lowered for token in ("touchdown", "landing", "debris", "crash"))


def _scene_uav_corridor_role(scene: dict[str, Any]) -> str:
    initial_state = dict(scene.get("initial_state") or {})
    if str(initial_state.get("role") or "") == "U_inspect":
        return "inspect_observer"
    return str(scene.get("uav_corridor_role") or initial_state.get("uav_corridor_role") or "")


def _scene_uses_mission_lifecycle(scene: dict[str, Any]) -> bool:
    corridor_role = _scene_uav_corridor_role(scene)
    return corridor_role not in {"inspect_observer", "observer"}


def add_uav_lifecycle(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    spec_by_id = {spec["entity_id"]: spec for spec in specs}
    uav_scenes = [
        scene
        for scene in scenes
        if str(scene.get("logical_asset_id", "")).startswith("uav.")
        and not scene.get("lifecycle", {}).get("auto_lifecycle_applied")
        and _scene_uses_mission_lifecycle(scene)
    ]
    if not uav_scenes:
        return

    original_positions: dict[str, list[float]] = {}
    current_positions: dict[str, list[float]] = {}
    last_event_for_uav: dict[str, str] = {}
    last_action_for_uav: dict[str, str] = {}
    last_scenario_event_id = ""
    for scene in uav_scenes:
        entity_id = str(scene["entity_id"])
        start_position = scene_pos(scene)
        if start_position:
            original_positions[entity_id] = start_position
            current_positions[entity_id] = start_position

    for event_def in events:
        event_id = str(event_def["event_id"])
        if event_id and not event_id.startswith("lifecycle_"):
            last_scenario_event_id = event_id
        for action_def in event_def.get("actions") or []:
            params = action_params(action_def)
            if params.get("type") != "move_entity":
                continue
            entity_id = str(params.get("entity_id") or "")
            if entity_id not in current_positions:
                continue
            waypoints = [wp for wp in params.get("waypoints_enu_m") or [] if isinstance(wp, list) and len(wp) >= 3]
            if not waypoints:
                continue
            current_positions[entity_id] = [float(waypoints[-1][0]), float(waypoints[-1][1]), float(waypoints[-1][2])]
            last_event_for_uav[entity_id] = event_id
            last_action_for_uav[entity_id] = str(params.get("action_id") or "")

    lifecycle_events: list[dict[str, Any]] = []
    landing_events: list[dict[str, Any]] = []
    for slot_index, scene in enumerate(uav_scenes):
        entity_id = str(scene["entity_id"])
        spec = spec_by_id.get(entity_id)
        mission_start = original_positions.get(entity_id)
        if spec is None or not mission_start:
            continue

        suffix = safe_id(entity_id)
        pad_id = f"pad_home_{suffix}"
        pad_position, pad_yaw = uav_home_pad_pose(mission_start, slot_index)
        pad_hover = q(pad_position[0], pad_position[1], UAV_PAD_HOVER_Z_M)
        climb = q(pad_position[0], pad_position[1], max(18.0, min(float(mission_start[2]) - 4.0, 24.0)))

        add(specs, scenes, pad_entity(pad_id, pad_position, f"home_{suffix}", "departure", pad_yaw))
        update_world_entity_position(spec, scene, pad_hover)
        scene["initial_state"] = {**(scene.get("initial_state") or {}), "mode": "preflight_on_pad"}
        scene["route_waypoints_enu_m"] = [mission_start] + list(scene.get("route_waypoints_enu_m") or [])
        scene["lifecycle"] = {
            "auto_lifecycle_applied": True,
            "home_pad_entity_id": pad_id,
            "home_hover_enu_m": pad_hover,
            "mission_start_enu_m": mission_start,
            "requires_takeoff": True,
            "requires_landing_or_terminal_resolution": True,
        }
        spec["movement_waypoints"] = [mission_start] + list(spec.get("movement_waypoints") or [])
        spec["visual_state"] = {**(spec.get("visual_state") or {}), "mode": "preflight_on_pad"}

        lifecycle_events.append(
            event(
                f"lifecycle_takeoff_{suffix}",
                tick(UAV_TAKEOFF_ENTRY_TICK + slot_index * 8),
                [move(f"move_{suffix}_takeoff_entry", entity_id, [pad_hover, climb, mission_start], 5.0)],
                0,
                scenario_id,
                "UAV takes off from visible home pad and enters mission airspace",
                "uav_lifecycle",
                [entity_id, pad_id],
                "info",
            )
        )

        final_position = current_positions.get(entity_id, mission_start)
        final_z = float(final_position[2])
        last_action = last_action_for_uav.get(entity_id, "")
        if not terminal_uav_action(last_action, final_z):
            landing_altitude = max(final_z, 36.0)
            approach = q(final_position[0], final_position[1], landing_altitude)
            pad_approach = q(pad_position[0], pad_position[1], landing_altitude)
            landing_after_event = last_scenario_event_id or last_event_for_uav.get(entity_id, f"lifecycle_takeoff_{suffix}")
            landing_events.append(
                event(
                    f"lifecycle_landing_{suffix}",
                    fired(landing_after_event, UAV_LANDING_AFTER_SCENARIO_DELAY_TICKS + slot_index * 8),
                    [move(f"move_{suffix}_landing_return", entity_id, [final_position, approach, pad_approach, pad_hover], 4.0)],
                    9,
                    scenario_id,
                    "UAV returns to the visible home pad and lands after mission resolution",
                    "uav_lifecycle",
                    [entity_id, pad_id],
                    "info",
                    intent="landing_or_terminal_resolution",
                )
            )

    if lifecycle_events:
        events[:0] = lifecycle_events
    if landing_events:
        events.extend(landing_events)


def local_bounds_for_bundle(bundle: ScenarioBundle) -> dict[str, Any]:
    points = collect_scene_bound_points(bundle)
    if not points:
        return {"center_enu_m": [WORLD_OFFSET_X_M, WORLD_OFFSET_Y_M, 0.0], "radius_m": 120.0}
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    radius = max(math.hypot(point[0] - cx, point[1] - cy) for point in points)
    return {
        "center_enu_m": q(cx, cy, 0.0),
        "radius_m": round(max(120.0, radius + 160.0), 3),
        "source": "export_envelope_only",
    }


def make_bundle(
    scenario_id: str,
    directory: Path,
    category: str,
    description: str,
    spec_entities: list[dict[str, Any]],
    scene_entities: list[dict[str, Any]],
    parameters: dict[str, Any],
    events: list[dict[str, Any]],
    weather_profile: dict[str, Any] | None = None,
    cameras: list[dict[str, Any]] | None = None,
    validation_rules: list[dict[str, Any]] | None = None,
) -> ScenarioBundle:
    contract = apply_semantic_event_contract(scenario_id, spec_entities, scene_entities, events, parameters)
    ensure_uav_corridor_population(scenario_id, spec_entities, scene_entities, events, parameters)
    add_uav_lifecycle(scenario_id, spec_entities, scene_entities, events)
    provisional_capture_boundary = _capture_boundary_from_points(scenario_id, scene_entities, events)
    ensure_uav_routes_cross_capture_boundary(
        scenario_id,
        spec_entities,
        scene_entities,
        events,
        provisional_capture_boundary,
        contract,
    )
    apply_uav_corridor_contract(scenario_id, spec_entities, scene_entities, events, parameters)
    normalize_proximity_trigger_reachability(scenario_id, scene_entities, events, parameters)
    capture_boundary_policy = _scenario_pad_boundary_policy(scenario_id)
    capture_boundary = {
        **_capture_boundary_from_points(scenario_id, scene_entities, events),
        "uav_boundary_crossing_required": True,
        "pad_boundary_policy": capture_boundary_policy,
        "inspect_fov_coverage_required": True,
        "inspect_entity_role": "U_inspect",
    }
    parameters["capture_boundary"] = capture_boundary
    if isinstance(parameters.get("semantic_event_contract"), dict):
        parameters["semantic_event_contract"] = {
            **parameters["semantic_event_contract"],
            "capture_boundary": {
                **dict(parameters["semantic_event_contract"].get("capture_boundary") or {}),
                **capture_boundary,
            },
            "uav_boundary_crossing_required": True,
            "pad_boundary_policy": {
                "default": contract.pad_policy.default,
                "inside_required_for": list(contract.pad_policy.inside_required_for),
            },
            "inspect_fov_coverage_required": True,
        }
    for scene in scene_entities:
        if str((scene.get("initial_state") or {}).get("role") or "") == "U_inspect":
            scene["contract_inspect_uav"] = {
                **(scene.get("contract_inspect_uav") or {}),
                "observed_capture_boundary_id": capture_boundary["boundary_id"],
                "inspect_fov_coverage_required": True,
            }
    ensure_contract_logical_count(scenario_id, spec_entities, scene_entities, events, parameters)
    final_contract_payload = dict(parameters.get("semantic_event_contract") or {})
    pending_validation_rules = []
    for rule in list(parameters.pop("_pending_validation_rules", [])):
        if rule.get("rule") == "semantic_event_contract":
            pending_validation_rules.append({**rule, "contract": final_contract_payload})
        else:
            pending_validation_rules.append(rule)
    ids = [e["entity_id"] for e in scene_entities]
    rules = base_rules(ids, scenario_id) + asset_rules(scene_entities)
    rules.extend(pending_validation_rules)
    rules.append(
        {
            "rule": "uav_boundary_crossing_required",
            "boundary_id": capture_boundary["boundary_id"],
            "description": "Mission/observer UAV routes must pass through the semantic capture boundary; pads may remain outside unless the pad policy says otherwise",
        }
    )
    if capture_boundary_policy == "event_pads_inside_capture_boundary":
        rules.append(
            {
                "rule": "pad_inside_boundary_required",
                "boundary_id": capture_boundary["boundary_id"],
                "description": "Pad-contention scenarios require event pads inside the semantic capture boundary",
            }
        )
    if validation_rules:
        rules.extend(validation_rules)
    if len(rules) < 2 and ids:
        rules.append({"rule": "entity_resolvable", "entity_id": ids[0], "description": "Entity must resolve"})
    enforce_physical_delays(events)
    finalize_event_semantics(events, scenario_id)
    apply_motion_contracts(scene_entities, events, contract)
    final_counts = _scene_counts(scene_entities)
    for key in ("uav", "vehicle", "pedestrian", "facility", "logical"):
        expected = contract.counts[key]
        actual = final_counts[key]
        if actual != expected:
            raise RuntimeError(f"{scenario_id}: semantic contract {key} count mismatch after generation: expected {expected}, got {actual}")
    inspect_uavs = [
        scene
        for scene in scene_entities
        if str((scene.get("initial_state") or {}).get("role") or "") == "U_inspect"
    ]
    if len(inspect_uavs) != 1:
        raise RuntimeError(f"{scenario_id}: semantic contract requires exactly one U_inspect actor, got {len(inspect_uavs)}")
    return ScenarioBundle(
        scenario_id=scenario_id,
        directory=directory,
        category=category,
        description=description,
        spec_entities=spec_entities,
        scene_entities=scene_entities,
        parameters=parameters,
        events=events,
        weather_profile=weather_profile or {"initial": "clear", "transitions": []},
        cameras=cameras or default_cameras_for_bundle(scene_entities, events),
        validation_rules=rules,
    )


def add(target_specs: list[dict[str, Any]], target_scenes: list[dict[str, Any]], pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, scene = pair
    target_specs.append(spec)
    target_scenes.append(scene)
    return pair


def _is_uav_scene(scene: dict[str, Any]) -> bool:
    return str(scene.get("logical_asset_id") or "").startswith("uav.")


def _is_corridor_scene(scene: dict[str, Any]) -> bool:
    return str(scene.get("logical_asset_id") or "") == UAV_CORRIDOR_LOGICAL_ASSET_ID


def _uav_action_is_low_or_terminal(action_id: str, waypoints: list[list[float]]) -> bool:
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
        )
    )


def _collect_corridor_reference_points(scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[list[float]]:
    points = []
    for scene in scenes:
        if _is_corridor_scene(scene):
            continue
        try:
            points.append(scene_pos(scene))
        except Exception:
            pass
        for waypoint in scene.get("route_waypoints_enu_m") or []:
            if isinstance(waypoint, list) and len(waypoint) >= 3:
                points.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
    points.extend(event_action_points(events))
    return points


def _assigned_uav_altitude(entity_id: str, ordinal: int) -> float:
    return float(UAV_ALTITUDE_LAYERS_M[ordinal % len(UAV_ALTITUDE_LAYERS_M)])


def _repairable_altitudes(entity_id: str, ordinal: int) -> list[float]:
    preferred = _assigned_uav_altitude(entity_id, ordinal)
    return [preferred] + [float(value) for value in UAV_ALTITUDE_LAYERS_M if float(value) != preferred]


def _event_uav_actions(events: list[dict[str, Any]], entity_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for event_def in events:
        for action_def in event_def.get("actions") or []:
            params = action_params(action_def)
            if params.get("type") == "move_entity" and str(params.get("entity_id") or "") == entity_id:
                result.append(action_def)
    return result


def _uav_route_points(scene: dict[str, Any], actions: list[dict[str, Any]]) -> list[list[float]]:
    points: list[list[float]] = []
    start = scene_pos(scene)
    if start:
        points.append(start)
    for waypoint in scene.get("route_waypoints_enu_m") or []:
        if isinstance(waypoint, list) and len(waypoint) >= 3:
            points.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
    for action_def in actions:
        for waypoint in action_params(action_def).get("waypoints_enu_m") or []:
            if isinstance(waypoint, list) and len(waypoint) >= 3:
                points.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
    return points


def _choose_uav_altitude(entity_id: str, ordinal: int, route_points: list[list[float]], used_altitudes: set[float]) -> tuple[float, bool]:
    if "intruder" in entity_id.lower():
        candidates = [28.0, 30.0, 34.0, 42.0] + [float(value) for value in UAV_ALTITUDE_LAYERS_M if float(value) not in {28.0, 30.0, 34.0, 42.0}]
    else:
        candidates = _repairable_altitudes(entity_id, ordinal)
    candidates = [item for item in candidates if item not in used_altitudes] + [item for item in candidates if item in used_altitudes]
    for altitude_m in candidates:
        if UAV_CORRIDORS.buildings.route_clear_at_altitude(route_points, altitude_m):
            return altitude_m, False
    for altitude_m in candidates:
        try:
            UAV_CORRIDORS.buildings.repair_route_at_altitude(route_points, altitude_m)
            return altitude_m, True
        except RuntimeError:
            continue
    return candidates[-1], True


def _repair_uav_waypoints(waypoints: list[list[float]], altitude_m: float) -> list[list[float]]:
    return UAV_CORRIDORS.buildings.repair_route_at_altitude(waypoints, altitude_m)


def _repair_uav_profile_waypoints(scenario_id: str, action_id: str, waypoints: list[list[float]]) -> list[list[float]]:
    if not waypoints:
        raise RuntimeError(f"{scenario_id}: low-altitude UAV action {action_id} has no deterministic waypoints")
    normalized = [q(float(point[0]), float(point[1]), float(point[2])) for point in waypoints]
    current = UAV_CORRIDORS.buildings.repair_point(normalized[0], float(normalized[0][2]))
    repaired = [current]
    for raw_target in normalized[1:]:
        target = UAV_CORRIDORS.buildings.repair_point(raw_target, float(raw_target[2]))
        leg = UAV_CORRIDORS.buildings.repair_segment(current, target)
        if not leg:
            leg = _repair_uav_profile_segment_with_bridge(current, target)
        if not leg:
            raise RuntimeError(
                f"{scenario_id}: low-altitude UAV action {action_id} cannot be repaired at profile altitude "
                f"from {current} to {target}"
            )
        for point in leg:
            if not repaired or dist_xy(repaired[-1], point) > 0.05 or abs(repaired[-1][2] - point[2]) > 0.05:
                repaired.append(point)
        current = repaired[-1]
    for a, b in zip(repaired, repaired[1:]):
        if not UAV_CORRIDORS.buildings.air_segment_clear(a, b):
            raise RuntimeError(
                f"{scenario_id}: low-altitude UAV action {action_id} still intersects an obstacle after profile repair "
                f"on segment {a}->{b}"
            )
    return repaired


def _repair_uav_profile_segment_with_bridge(current: list[float], target: list[float]) -> list[list[float]]:
    high_altitude = max(float(current[2]), float(target[2]))
    candidate_altitudes: list[float] = []
    for value in (high_altitude, 42.0, 50.0, 60.0, 80.0, 36.0, 30.0, 24.0, 18.0, 12.0, float(target[2])):
        rounded = round(max(float(value), float(target[2])), 3)
        if rounded not in candidate_altitudes:
            candidate_altitudes.append(rounded)
    bridge_points: list[list[float]] = []
    if float(current[2]) != float(target[2]):
        bridge_points.append(q(float(current[0]), float(current[1]), float(target[2])))
    bridge_points.append(q(float(current[0]), float(current[1]), max(candidate_altitudes)))
    bridge_points.append(q(float(target[0]), float(target[1]), max(candidate_altitudes)))
    bridge_points.append(q(float(target[0]), float(target[1]), float(target[2])))
    horizontal_candidates = [
        q(float(current[0]), float(target[1]), max(candidate_altitudes)),
        q(float(target[0]), float(current[1]), max(candidate_altitudes)),
        q((float(current[0]) + float(target[0])) * 0.5, float(current[1]), max(candidate_altitudes)),
        q(float(current[0]), (float(current[1]) + float(target[1])) * 0.5, max(candidate_altitudes)),
        q((float(current[0]) + float(target[0])) * 0.5, float(target[1]), max(candidate_altitudes)),
        q(float(target[0]), (float(current[1]) + float(target[1])) * 0.5, max(candidate_altitudes)),
    ]
    for candidate in horizontal_candidates:
        if UAV_CORRIDORS.buildings.air_point_clear(candidate):
            bridge_points.append(candidate)
    unique_bridge_points: list[list[float]] = []
    for point in bridge_points:
        if not unique_bridge_points or dist_xy(unique_bridge_points[-1], point) > 0.05 or abs(unique_bridge_points[-1][2] - point[2]) > 0.05:
            unique_bridge_points.append(point)
    for bridge in unique_bridge_points:
        if UAV_CORRIDORS.buildings.air_segment_clear(current, bridge) and UAV_CORRIDORS.buildings.air_segment_clear(bridge, target):
            return [bridge, target]
    if len(unique_bridge_points) >= 2:
        for first in unique_bridge_points:
            if not UAV_CORRIDORS.buildings.air_segment_clear(current, first):
                continue
            for second in unique_bridge_points:
                if dist_xy(first, second) <= 0.05 and abs(first[2] - second[2]) <= 0.05:
                    continue
                if UAV_CORRIDORS.buildings.air_segment_clear(first, second) and UAV_CORRIDORS.buildings.air_segment_clear(second, target):
                    return [first, second, target]
    return []


def _set_uav_altitude_metadata(
    spec: dict[str, Any],
    scene: dict[str, Any],
    altitude_m: float,
    repaired_route: list[list[float]],
    *,
    used_lateral_bypass: bool,
) -> None:
    if not repaired_route:
        return
    route_waypoints = list(repaired_route[1:] or [repaired_route[0]])
    if str((scene.get("initial_state") or {}).get("role") or "") == "U_inspect":
        route_waypoints = list(repaired_route)
    scene["route_waypoints_enu_m"] = route_waypoints
    scene["initial_state"] = {
        **(scene.get("initial_state") or {}),
        "assigned_altitude_m": altitude_m,
        "uav_corridor_role": scene.get("uav_corridor_role")
        or (scene.get("initial_state") or {}).get("uav_corridor_role", "mission_uav"),
    }
    scene["uav_corridor"] = {
        "assigned_altitude_m": altitude_m,
        "altitude_layers_m": list(UAV_ALTITUDE_LAYERS_M),
        "used_lateral_bypass": used_lateral_bypass,
    }
    if isinstance(scene.get("lifecycle"), dict):
        scene["lifecycle"] = {
            **scene["lifecycle"],
            "home_hover_enu_m": list(repaired_route[0]),
            "mission_start_enu_m": list(repaired_route[0]),
            "assigned_altitude_m": altitude_m,
            "waypoint_repair_applied": True,
        }
    spec["movement_waypoints"] = route_waypoints
    spec["visual_state"] = {
        **(spec.get("visual_state") or {}),
        "assigned_altitude_m": altitude_m,
    }
    update_world_entity_position(spec, scene, list(repaired_route[0]))


def _replace_uav_action_waypoints(
    action_def: dict[str, Any],
    repaired_waypoints: list[list[float]],
    altitude_m: float,
    *,
    validation_status: str = "waypoints_repaired_for_building_clearance",
) -> None:
    params = action_params(action_def)
    original_waypoints = params.get("waypoints_enu_m") or []
    params["waypoints_enu_m"] = repaired_waypoints
    params["uav_corridor_validation"] = {
        "status": validation_status,
        "assigned_altitude_m": altitude_m,
        "original_waypoints_enu_m": original_waypoints,
        "preserved_action_type": "move_entity",
    }
    write_action_params(action_def, params)


def _corridor_entity_from_segment(
    scenario_id: str,
    entity_id: str,
    segment_index: int,
    a: list[float],
    b: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    length_m = max(1.0, math.sqrt(dx * dx + dy * dy + dz * dz))
    mid = q((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)
    yaw_deg = round(math.degrees(math.atan2(dy, dx)), 3)
    corridor_id = f"corridor_{safe_id(scenario_id)}_{safe_id(entity_id)}_{segment_index:02d}"
    spec = {
        "entity_id": corridor_id,
        "asset_id": UAV_CORRIDOR_LOGICAL_ASSET_ID,
        "initial_pos_enu": mid,
        "initial_rotation_deg": [0.0, 0.0, yaw_deg],
        "movement_waypoints": [],
        "visual_state": {"mode": "semantic_corridor"},
    }
    scene = {
        "entity_id": corridor_id,
        "logical_asset_id": UAV_CORRIDOR_LOGICAL_ASSET_ID,
        "category": "airspace_corridor",
        "placement_mode": "box_volume",
        "placement": {
            "center_enu_m": mid,
            "resolved_position_enu_m": mid,
            "extent_m": [round(length_m * 0.5, 3), 4.0, 4.0],
            "size_m": [round(length_m, 3), 8.0, 8.0],
            "rotation_deg": {"pitch_deg": 0.0, "yaw_deg": yaw_deg, "roll_deg": 0.0},
            "scale_xyz": [round(length_m, 3), 8.0, 8.0],
            "source_uav_entity_id": entity_id,
            "segment_start_enu_m": a,
            "segment_end_enu_m": b,
            "altitude_layer_m": round((a[2] + b[2]) * 0.5, 3),
        },
        "initial_state": {
            "mode": "semantic_corridor",
            "semantic_class": "uav_corridor",
            "custom_stencil_only": False,
        },
        "query_tags": ["UAVCorridor", "HighAltitudeCorridor", "event_semantic"],
        "activation_tick": 0,
        "enabled": True,
    }
    return spec, scene


def _contract_logical_sidecar(
    scenario_id: str,
    index: int,
    anchor: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    entity_id = f"contract_logical_{safe_id(scenario_id)}_{index + 1:02d}"
    angle = math.radians((index * 47) % 360)
    radius = 6.0 + (index % 4) * 2.0
    position = q(
        anchor[0] + math.cos(angle) * radius,
        anchor[1] + math.sin(angle) * radius,
        GROUND_Z_M,
    )
    spec, scene = world_entity(
        entity_id,
        "semantic.asset_anchor",
        "crowd_anchor",
        position,
        float((index * 29) % 360),
        "semantic_context_anchor",
        visual_state=semantic_state(
            task_id=f"{scenario_id}.logical_sidecar.{index + 1:02d}",
            role="semantic_logical_sidecar",
            state_sequence=["semantic_context_anchor"],
            semantic_role="deterministic logical sidecar required by low-altitude event-chain contract",
            mode="semantic_context_anchor",
        ),
    )
    scene["contract_logical_sidecar"] = {
        "policy": "semantic_event_contract_v1",
        "contract_scenario_id": scenario_id,
        "logical_index": index + 1,
    }
    return spec, scene


def ensure_contract_logical_count(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> None:
    target = int(parameters.get("target_logical_count") or 0)
    current = _scene_counts(scenes)["logical"]
    if current > target:
        raise RuntimeError(f"{scenario_id}: logical sidecars exceed contract before filler: expected {target}, got {current}")
    anchor = _contract_center(scenes, events)
    while current < target:
        add(specs, scenes, _contract_logical_sidecar(scenario_id, current, anchor))
        current = _scene_counts(scenes)["logical"]


def ensure_uav_corridor_population(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> None:
    existing_uav_count = sum(1 for scene in scenes if _is_uav_scene(scene))
    default_target = max(1, existing_uav_count)
    target_uav_count = max(1, min(5, int(parameters.get("target_uav_count") or default_target)))
    target_uav_count = max(target_uav_count, min(5, existing_uav_count))
    parameters["target_uav_count"] = target_uav_count
    if existing_uav_count >= target_uav_count:
        return

    reference_points = _collect_corridor_reference_points(scenes, events)
    slots = UAV_CORRIDORS.find_slots(
        reference_points,
        count=target_uav_count - existing_uav_count,
        slot_prefix=f"{safe_id(scenario_id)}_observer_slot",
    )
    for observer_index in range(existing_uav_count, target_uav_count):
        slot = slots[observer_index - existing_uav_count]
        entity_id = f"uav_observer_{safe_id(scenario_id)}_{observer_index + 1}"
        if any(scene.get("entity_id") == entity_id for scene in scenes):
            continue
        asset = UAV_ASSET_CYCLE[observer_index % len(UAV_ASSET_CYCLE)]
        spec, scene = world_entity(
            entity_id,
            asset,
            "uav",
            list(slot.start_enu_m),
            slot.yaw_deg,
            "corridor_observer",
            route=[list(slot.end_enu_m)],
            visual_state={"mode": "corridor_observer"},
        )
        scene["uav_corridor_role"] = "observer"
        specs.append(spec)
        scenes.append(scene)


def apply_uav_corridor_contract(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> None:
    parameters.setdefault("uav_corridor_policy", "repair_waypoints_then_visualize_corridors_v1")
    parameters.setdefault("allow_building_impact", False)
    target_uav_count = int(parameters.get("target_uav_count") or max(1, sum(1 for scene in scenes if _is_uav_scene(scene))))

    spec_by_id = {spec["entity_id"]: spec for spec in specs}
    uav_scenes = [scene for scene in scenes if _is_uav_scene(scene)]
    repaired_routes_by_entity: dict[str, list[list[float]]] = {}
    original_routes_by_entity: dict[str, list[list[float]]] = {}
    assigned_altitudes: dict[str, float] = {}
    used_lateral_bypass: dict[str, bool] = {}
    used_altitudes: set[float] = set()
    for ordinal, scene in enumerate(uav_scenes):
        entity_id = str(scene["entity_id"])
        spec = spec_by_id.get(entity_id)
        if spec is None:
            continue
        original_route = []
        start_for_original = scene_pos(scene)
        if start_for_original:
            original_route.append(start_for_original)
        for waypoint in scene.get("route_waypoints_enu_m") or []:
            if isinstance(waypoint, list) and len(waypoint) >= 3:
                original_route.append([float(waypoint[0]), float(waypoint[1]), float(waypoint[2])])
        original_routes_by_entity[entity_id] = original_route
        actions = _event_uav_actions(events, entity_id)
        route_points = _uav_route_points(scene, actions)
        initial_state = dict(scene.get("initial_state") or {})
        if str(initial_state.get("role") or "") == "U_inspect":
            altitude_m = float(
                initial_state.get("assigned_altitude_m")
                or initial_state.get("inspect_altitude_m")
                or initial_state.get("altitude_m")
                or 28.0
            )
            needs_lateral_bypass = False
        else:
            altitude_m, needs_lateral_bypass = _choose_uav_altitude(entity_id, ordinal, route_points, used_altitudes)
        used_altitudes.add(altitude_m)
        assigned_altitudes[entity_id] = altitude_m
        used_lateral_bypass[entity_id] = needs_lateral_bypass

        base_start = scene_pos(scene)
        repaired_start = _repair_uav_waypoints([base_start], altitude_m)[0] if base_start else []
        if repaired_start:
            update_world_entity_position(spec, scene, repaired_start)
        repaired_route = [repaired_start] if repaired_start else []
        for action_def in actions:
            original = [
                [float(point[0]), float(point[1]), float(point[2])]
                for point in action_params(action_def).get("waypoints_enu_m") or []
                if isinstance(point, list) and len(point) >= 3
            ]
            if not original:
                continue
            if repaired_route:
                route_input = [repaired_route[-1]] + (original[1:] if len(original) > 1 else [original[0]])
            else:
                route_input = original
            action_id = str(action_params(action_def).get("action_id") or action_def.get("action_id") or "")
            if _uav_action_is_low_or_terminal(action_id, original):
                action_waypoints = _repair_uav_profile_waypoints(scenario_id, action_id, route_input)
                validation_status = "low_altitude_waypoints_preserved_and_building_clear"
            else:
                action_waypoints = _repair_uav_waypoints(route_input, altitude_m)
                if route_input and action_waypoints:
                    route_start = route_input[0]
                    if dist_xy(route_start, action_waypoints[0]) <= 0.05 and abs(route_start[2] - action_waypoints[0][2]) > 0.05:
                        action_waypoints = [route_start] + action_waypoints
                validation_status = "waypoints_repaired_for_building_clearance"
            _replace_uav_action_waypoints(action_def, action_waypoints, altitude_m, validation_status=validation_status)
            if action_waypoints:
                for point in action_waypoints:
                    if not repaired_route or dist_xy(repaired_route[-1], point) > 0.05 or abs(repaired_route[-1][2] - point[2]) > 0.05:
                        repaired_route.append(point)
        if str(initial_state.get("role") or "") == "U_inspect":
            refreshed_loop = _inspect_boundary_sample_loop(scenes, events, altitude_m)
            if refreshed_loop is not None:
                refreshed_start, refreshed_route = refreshed_loop
                planned_route = [refreshed_start, *refreshed_route]
                update_world_entity_position(spec, scene, refreshed_start)
            else:
                planned_route = [
                    [float(point[0]), float(point[1]), float(point[2])]
                    for point in (scene.get("contract_inspect_uav") or {}).get("planned_route_enu_m", [])
                    if isinstance(point, list) and len(point) >= 3
                ]
                if not planned_route:
                    planned_route = original_routes_by_entity.get(entity_id, [])
            if _route_length_m(planned_route) < 80.0:
                raise RuntimeError(
                    f"{scenario_id}: U_inspect planned route is shorter than the deterministic contract minimum: "
                    f"{entity_id} length={_route_length_m(planned_route):.3f}m"
                )
            inspect_route = planned_route
            if _route_length_m(inspect_route) < 80.0:
                raise RuntimeError(
                    f"{scenario_id}: U_inspect repaired route is shorter than the deterministic contract minimum: "
                    f"{entity_id} length={_route_length_m(inspect_route):.3f}m"
                )
            inspect_loop_route = list(inspect_route)
            scene["contract_inspect_uav"] = {
                **(scene.get("contract_inspect_uav") or {}),
                "repaired_route_enu_m": inspect_route,
                "loop_route_enu_m": inspect_loop_route,
                "repaired_path_length_m": round(_route_length_m(inspect_route), 3),
                "fixed_altitude_loop": True,
            }
            repaired_route = list(inspect_route)
            if _route_length_m(repaired_route) < 80.0:
                raise RuntimeError(
                    f"{scenario_id}: U_inspect final generated route is shorter than the deterministic contract minimum: "
                    f"{entity_id} length={_route_length_m(repaired_route):.3f}m"
                )
        if len(repaired_route) == 1 and scene.get("route_waypoints_enu_m"):
            extra = _repair_uav_waypoints(scene.get("route_waypoints_enu_m") or [], altitude_m)
            repaired_route.extend(extra)
        _set_uav_altitude_metadata(
            spec,
            scene,
            altitude_m,
            repaired_route,
            used_lateral_bypass=needs_lateral_bypass,
        )
        repaired_routes_by_entity[entity_id] = repaired_route

    existing_corridors = {scene["entity_id"] for scene in scenes if _is_corridor_scene(scene)}
    existing_logical_count = _scene_counts(scenes)["logical"]
    target_logical_count = int(parameters.get("target_logical_count") or existing_logical_count)
    allowed_corridor_scene_count = target_logical_count - existing_logical_count
    if allowed_corridor_scene_count < 0:
        raise RuntimeError(
            f"{scenario_id}: existing logical sidecars exceed contract before UAV corridor visualization: "
            f"expected {target_logical_count}, got {existing_logical_count}"
        )
    materialized_corridor_count = 0
    corridor_metadata: list[dict[str, Any]] = []
    for entity_id, route in repaired_routes_by_entity.items():
        for segment_index, (a, b) in enumerate(zip(route, route[1:])):
            if dist_xy(a, b) <= 0.05 and abs(a[2] - b[2]) <= 0.05:
                continue
            corridor_metadata.append(
                {
                    "entity_id": entity_id,
                    "segment_index": segment_index,
                    "segment_start_enu_m": a,
                    "segment_end_enu_m": b,
                    "assigned_altitude_m": assigned_altitudes.get(entity_id),
                }
            )
            if materialized_corridor_count >= allowed_corridor_scene_count:
                continue
            spec, scene = _corridor_entity_from_segment(scenario_id, entity_id, segment_index, a, b)
            if scene["entity_id"] not in existing_corridors:
                specs.append(spec)
                scenes.append(scene)
                existing_corridors.add(scene["entity_id"])
                materialized_corridor_count += 1

    corridor_rule = {
        "rule": "uav_corridor_contract",
        "target_uav_count": target_uav_count,
        "corridor_segment_count": len(corridor_metadata),
        "materialized_corridor_scene_count": materialized_corridor_count,
        "allow_building_impact": bool(parameters.get("allow_building_impact", False)),
        "description": "UAV waypoints are repaired for building clearance; corridor scene entities are capped by the exact episode logical-sidecar contract",
    }
    parameters["uav_corridor_segment_count"] = corridor_rule["corridor_segment_count"]
    parameters["target_uav_count"] = target_uav_count
    parameters["uav_altitude_layers_m"] = list(UAV_ALTITUDE_LAYERS_M)
    parameters["uav_assigned_altitudes_m"] = assigned_altitudes
    parameters["uav_lateral_bypass_used"] = used_lateral_bypass
    parameters["uav_corridor_segments"] = [
        {
            "entity_id": entity_id,
            "point_count": len(route),
            "assigned_altitude_m": assigned_altitudes.get(entity_id),
            "used_lateral_bypass": used_lateral_bypass.get(entity_id, False),
        }
        for entity_id, route in repaired_routes_by_entity.items()
    ]
    parameters["uav_corridor_segment_details"] = corridor_metadata
    for scene in uav_scenes:
        scene.setdefault("validation_tags", []).append("uav_corridor_contract")
    if not any(rule.get("rule") == "uav_corridor_contract" for rule in parameters.setdefault("_pending_validation_rules", [])):
        parameters["_pending_validation_rules"].append(corridor_rule)


def l_path(layer: str, subdir: str, scenario_id: str) -> Path:
    return SCENARIOS_ROOT / layer / subdir / scenario_id


def x_path(dirname: str) -> Path:
    return SCENARIOS_ROOT / "X_cross_layer" / dirname


def build_l1(scenario_id: str, idx: int) -> ScenarioBundle:
    ox = 38.0 + idx * 8.5
    oy = 12.0 + idx * 4.0
    specs: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    uav = f"uav_{scenario_id.lower().replace('-', '_')}_primary"
    zone = f"nfz_{scenario_id.lower().replace('-', '_')}"
    start = p(ox, oy, 30 + (idx % 3) * 2)
    near = p(ox + 19, oy + 10, start[2])
    center = p(ox + 26, oy + 14, 28)
    safe = p(ox - 5, oy + 25, 34)
    approach_mid = q(near[0] - 4, near[1] - 2, near[2])
    avoidance_hold = q(near[0] - 6, near[1] + 4, near[2] + 4)
    asset = "trigger.no_fly.box.v1" if scenario_id.startswith(("L1-1", "L1-3")) else "trigger.hazard.generic.box.v1"
    add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 35, "patrol", route=[near, safe]))
    if scenario_id.startswith("L1-1"):
        polygon = [
            q(center[0] - 9, center[1] - 7, 22),
            q(center[0] + 11, center[1] - 6, 22),
            q(center[0] + 12, center[1] + 8, 22),
            q(center[0] - 8, center[1] + 9, 22),
        ]
        add(specs, scenes, polygon_entity(zone, asset, "airspace_constraint", polygon, 20.0, 26.0, center))
    else:
        add(specs, scenes, box_entity(zone, asset, "airspace_constraint", center, [14.0, 10.0, 14.0]))
    events = [
        event(
            "approach_boundary",
            tick(220),
            [move("move_boundary_approach", uav, [start, approach_mid, near], 8.0)],
            1,
            scenario_id,
            "UAV approaches constrained airspace",
            "uav_mission",
            [uav, zone],
            "info",
        ),
        event(
            "boundary_conflict",
            prox(uav, zone, 14.0, 3, metric="xy"),
            [
                move("move_boundary_avoidance", uav, [near, avoidance_hold], 3.0),
                visual("hover_boundary_hold", uav, "hover"),
            ],
            2,
            scenario_id,
            "Airspace boundary conflict detected",
            "airspace",
            [uav, zone],
            "warning",
        ),
        event(
            "return_safe_airspace",
            fired("boundary_conflict"),
            [move("move_boundary_return_safe", uav, [avoidance_hold, safe], 9.0)],
            3,
            scenario_id,
            "UAV returns to safe airspace",
            "uav_mission",
            [uav, zone],
            "info",
        ),
    ]
    if scenario_id.startswith("L1-3"):
        intruder = f"intruder_{scenario_id.lower().replace('-', '_')}"
        intr_start = p(ox + 38, oy + 6, 29)
        intr_near = q(center[0] + 3, center[1] - 2, 29)
        add(specs, scenes, world_entity(intruder, "uav.airsim.flying_pawn.v1", "uav", intr_start, 215, "noncooperative", route=[intr_near]))
        events[0]["actions"].append(move("move_intruder_converge", intruder, [intr_start, intr_near], 10.0))
        events[1]["trigger"] = prox(intruder, zone, 14.0, 3, metric="xy")
        events[1]["log_title"] = "Noncooperative intruder enters constrained airspace"
        if scenario_id.startswith("L1-3_v2"):
            events[1]["log_title"] = "Fast intruder crossing constrained airspace"
        events[1]["log_target_ids"].append(intruder)
    events[1]["require_conditions"] = ["trig_approach_boundary"]
    if scenario_id.startswith("L1-4"):
        uav_b = f"uav_{scenario_id.lower().replace('-', '_')}_secondary"
        uav_c = f"uav_{scenario_id.lower().replace('-', '_')}_tertiary"
        add(specs, scenes, world_entity(uav_b, "uav.airsim.cv_pawn.v1", "uav", p(ox + 6, oy - 18, 32), 20, "patrol"))
        add(specs, scenes, world_entity(uav_c, "uav.airsim.flying_pawn.v1", "uav", p(ox - 12, oy - 10, 34), 45, "patrol"))
        secondary_conflict = q(center[0] - 4, center[1] - 2, 32)
        tertiary_conflict = q(center[0] - 2, center[1] + 1, 34)
        events[0]["actions"].extend(
            [
                move("move_corridor_secondary", uav_b, [p(ox + 6, oy - 18, 32), secondary_conflict], 7.0),
                move("move_corridor_tertiary", uav_c, [p(ox - 12, oy - 10, 34), tertiary_conflict], 6.5),
            ]
        )
        events[1]["actions"].append(move("move_corridor_reroute_secondary", uav_b, [secondary_conflict, q(center[0] + 16, center[1] + 16, 38)], 8.5))
        events[2]["actions"].append(move("move_corridor_resume_tertiary", uav_c, [tertiary_conflict, q(center[0] - 18, center[1] + 14, 34)], 7.5))
    if scenario_id.startswith("L1-2"):
        events[1]["actions"][0] = move("move_altitude_recover", uav, [near, q(near[0] - 3, near[1] + 4, 42), q(near[0] - 8, near[1] + 8, 34)], 4.0)
        events[1]["log_title"] = "Altitude deviation from assigned corridor"
    return make_bundle(
        scenario_id,
        l_path("L1_airspace", "interaction", scenario_id),
        "airspace",
        {
            "L1-1_v1": "Geofence boundary approach and RTH",
            "L1-1_v2": "Geofence edge intrusion with alternate RTH",
            "L1-2_v1": "Altitude deviation from assigned corridor",
            "L1-3_v1": "Noncooperative intruder in operational airspace",
            "L1-3_v2": "Fast intruder crossing constrained airspace",
            "L1-4_v1": "Corridor congestion with multi-UAV separation",
            "L1-4_v2": "Dense corridor congestion and staged recovery",
        }[scenario_id],
        specs,
        scenes,
        {"approach_tick": 220, "conflict_distance_m": 14.0, "resolution_tick": 420},
        events,
        validation_rules=[
            {
                "rule": "uav_start_outside_constraint",
                "entity_id": uav,
                "constraint_id": zone,
                "description": "UAV start is outside the no-fly or hazard volume",
            },
            {
                "rule": "event_chain_min",
                "min_count": 3,
                "description": "Airspace scenario must include approach, conflict, and recovery",
            },
        ],
    )


def build_l2(scenario_id: str, idx: int) -> ScenarioBundle:
    ox = 48.0 + idx * 7.0
    oy = -18.0 + idx * 5.5
    specs: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    sid = scenario_id.lower().replace("-", "_")
    events: list[dict[str, Any]]
    params = {"failure_tick": 260, "resolution_tick": 520}
    if scenario_id.startswith("L2-1"):
        tower = f"tower_{sid}"
        uav = f"uav_{sid}"
        start = p(ox, oy, 31)
        degraded = p(ox + 10, oy + 10, 30)
        home = p(ox - 11, oy + 2, 31)
        add(specs, scenes, world_entity(tower, "facility.radio.base_tower.v1", "facility", p(ox + 6, oy + 4, 0), 0, "online"))
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 30, "patrol", route=[degraded, home]))
        degraded_loiter = q(degraded[0] + 3, degraded[1] + 5, 34)
        events = [
            event("station_degraded", tick(260), [visual("set_tower_degraded", tower, "degraded")], 1, scenario_id, "Communication station degraded", "infrastructure", [tower], "warning"),
            event("uav_link_response", fired("station_degraded"), [move("move_uav_backup_loiter", uav, [start, degraded, degraded_loiter], 4.0)], 2, scenario_id, "UAV changes behavior under degraded C2", "uav_mission", [uav, tower], "warning"),
            event("backup_link_restore", fired("uav_link_response"), [visual("set_tower_backup", tower, "backup_link"), move("move_uav_resume_patrol", uav, [degraded_loiter, home], 7.0)], 3, scenario_id, "Backup link restores service", "infrastructure", [uav, tower], "info"),
        ]
        desc = "Communication station failure with backup link recovery"
    elif scenario_id.startswith("L2-2"):
        tower = f"tower_{sid}"
        uav = f"uav_{sid}"
        facade = f"facade_{sid}"
        start = p(ox, oy, 30)
        planned = p(ox + 14, oy + 4, 28)
        drift = p(ox + 28, oy + 14, 24)
        corrected = p(ox + 18, oy + 20, 30)
        add(specs, scenes, world_entity(tower, "facility.radio.base_tower.v1", "facility", p(ox + 4, oy + 2, 0), 0, "online"))
        add(specs, scenes, facade_entity(facade, BUILDINGS.id_for(20 + idx), p(ox + 30, oy + 16, 18), [0.8, -0.6, 0.0]))
        add(specs, scenes, world_entity(uav, "uav.airsim.cv_pawn.v1", "uav", start, 45, "corridor_follow", route=[planned, drift, corrected]))
        events = [
            event("gnss_anomaly", tick(250), [visual("set_tower_multipath", tower, "multipath_warning")], 1, scenario_id, "GNSS multipath anomaly in urban canyon", "digital_layer", [uav, tower], "warning"),
            event("route_drift", fired("gnss_anomaly"), [move("move_uav_multipath_drift", uav, [start, planned, drift], 6.0)], 2, scenario_id, "UAV drifts off planned route by more than 10m", "uav_mission", [uav, facade], "warning"),
            event("visual_relocalization", prox(uav, facade, 8.0, 3, metric="xy"), [visual("set_uav_visual_reloc", uav, "visual_relocalization"), screenshot("capture_gnss_drift")], 3, scenario_id, "Visual relocalization engages near facade", "uav_mission", [uav, facade], "warning"),
            event("route_corrected", fired("visual_relocalization"), [move("move_uav_corrected_route", uav, [drift, corrected], 5.0)], 4, scenario_id, "UAV corrects the route after visual relocalization", "uav_mission", [uav], "info"),
        ]
        desc = "GNSS urban canyon multipath drift and visual correction"
    elif scenario_id.startswith("L2-3"):
        uav = f"uav_{sid}"
        charger = f"charger_primary_{sid}"
        backup = f"charger_backup_{sid}"
        start = p(ox - 6, oy + 8, 28)
        inspect = p(ox + 3, oy + 1, 10)
        backup_app = p(ox + 25, oy + 10, 10)
        add(specs, scenes, world_entity(charger, "facility.charger.cityops.v1", "facility", p(ox + 2, oy, 0), 90, "available"))
        add(specs, scenes, world_entity(backup, "facility.charger.cityops.v1", "facility", p(ox + 27, oy + 12, 0), 90, "available"))
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 110, "low_battery", route=[inspect, backup_app]))
        events = [
            event("charger_unavailable", tick(240), [visual("set_primary_charger_offline", charger, "unavailable")], 1, scenario_id, "Primary charger becomes unavailable", "infrastructure", [charger], "warning"),
            event("uav_inspects_charger", fired("charger_unavailable"), [move("move_uav_charger_inspection", uav, [start, p(ox, oy + 3, 18), inspect], 4.0), screenshot("capture_charger_fault")], 2, scenario_id, "UAV checks the unavailable charger", "uav_mission", [uav, charger], "warning"),
            event("reroute_backup_charger", fired("uav_inspects_charger"), [move("move_uav_backup_charger", uav, [inspect, p(ox + 14, oy + 6, 18), backup_app], 6.0), visual("set_backup_charger_reserved", backup, "reserved")], 3, scenario_id, "UAV reroutes to backup charger", "uav_mission", [uav, backup], "info"),
        ]
        desc = "Charger unavailable with reroute to backup charger"
    elif scenario_id.startswith("L2-4"):
        pad = f"pad_shared_{sid}"
        uav_a = f"uav_priority_{sid}"
        uav_b = f"uav_divert_{sid}"
        start_a = p(ox - 12, oy + 12, 32)
        start_b = p(ox + 18, oy - 10, 30)
        hold_b = p(ox + 14, oy + 14, 36)
        pad_pos, pad_yaw, pad_anchor = landing_pad_pose(
            p(ox + 2, oy + 2, 0),
            context=f"{scenario_id} shared landing pad",
        )
        pad_hover = q(pad_pos[0], pad_pos[1], 5.0)
        pad_touchdown = q(pad_pos[0], pad_pos[1], 1.2)
        add(specs, scenes, pad_entity(pad, pad_pos, f"pad_{sid}", "east", pad_yaw, pad_anchor))
        add(specs, scenes, world_entity(uav_a, "uav.inspect.quad.v1", "uav", start_a, 75, "emergency_landing", route=[pad_hover]))
        add(specs, scenes, world_entity(uav_b, "uav.airsim.flying_pawn.v1", "uav", start_b, 295, "landing_request", route=[hold_b]))
        events = [
            event("dual_pad_approach", tick(230), [move("move_priority_uav_to_pad", uav_a, [start_a, p(ox - 4, oy + 6, 22), pad_hover], 5.0), move("move_second_uav_to_pad_hold", uav_b, [start_b, p(ox + 6, oy - 2, 24), hold_b], 5.5)], 1, scenario_id, "Two UAVs request the same landing pad", "uav_mission", [uav_a, uav_b, pad], "warning"),
            event("priority_arbitration", prox(uav_a, pad, 7.0, 2), [visual("set_pad_reserved_priority", pad, "reserved"), visual("set_second_uav_hold", uav_b, "hold")], 2, scenario_id, "Pad priority arbitration reserves pad for emergency UAV", "infrastructure", [uav_a, uav_b, pad], "warning"),
            event("second_uav_diverted", fired("priority_arbitration"), [move("move_second_uav_diversion", uav_b, [hold_b, p(ox + 34, oy + 22, 34)], 8.0), move("move_priority_uav_landing", uav_a, [pad_hover, pad_touchdown], 1.5)], 3, scenario_id, "Second UAV diverts while priority UAV lands", "uav_mission", [uav_a, uav_b, pad], "info"),
        ]
        params["contention_distance_m"] = 7.0
        desc = "Landing pad emergency contention and priority diversion"
    else:
        signal = "traffic_signal_l2_5_v1"
        police = "police_unit_l2_5_v1"
        car_a = "civilian_car_l2_5_a"
        car_b = "civilian_car_l2_5_b"
        start_a = p(ox - 25, oy, 0)
        start_b = p(ox + 4, oy - 24, 0)
        add(specs, scenes, world_entity(signal, "prop.traffic_control.signal_light.v1", "traffic_signal", p(ox, oy, 4), 0, "green_cycle"))
        add(specs, scenes, lane_entity(car_a, "vehicle.ground.boxcar.v1", "vehicle", "cg_edge_7", 82, 0.0, start_a, 80, "moving", route=[p(ox - 6, oy, 0), p(ox + 8, oy + 1, 0)]))
        add(specs, scenes, lane_entity(car_b, "vehicle.ground.boxcar.v1", "vehicle", "cg_edge_8", 54, 0.0, start_b, 5, "moving", route=[p(ox + 2, oy - 5, 0), p(ox + 2, oy + 9, 0)]))
        add(specs, scenes, lane_entity(police, "vehicle.emergency.police_suv.v1", "vehicle", "cg_edge_12", 38, 0.0, p(ox - 36, oy - 16, 0), 70, "response", visual_state={"mode": "response", "lights_on": True}))
        events = [
            event("signal_all_red_fault", tick(210), [visual("set_signal_all_red", signal, "all_red_fault")], 1, scenario_id, "Traffic signal all-red fault", "infrastructure", [signal], "critical"),
            event("traffic_confusion", fired("signal_all_red_fault"), [move("move_car_a_confused_stop", car_a, [start_a, p(ox - 5, oy, 0)], 3.0), move("move_car_b_confused_stop", car_b, [start_b, p(ox + 2, oy - 4, 0)], 3.0)], 2, scenario_id, "Traffic queues form under all-red fault", "vehicle", [car_a, car_b, signal], "warning"),
            event("police_arrival", fired("traffic_confusion"), [move("move_police_to_intersection", police, [p(ox - 36, oy - 16, 0), p(ox - 7, oy - 4, 0)], 12.0), visual("set_police_manual_control", police, "manual_directing", lights_on=True)], 3, scenario_id, "Police arrives for manual traffic control", "vehicle", [police, signal], "warning"),
            event("manual_flow_restore", fired("police_arrival"), [move("move_car_a_released", car_a, [p(ox - 5, oy, 0), p(ox + 16, oy + 1, 0)], 6.0), move("move_car_b_released", car_b, [p(ox + 2, oy - 4, 0), p(ox + 2, oy + 15, 0)], 5.5)], 4, scenario_id, "Police restores manual vehicle flow", "vehicle", [police, car_a, car_b], "info"),
        ]
        params["primary_id"] = signal
        params["secondary_id"] = police
        desc = "Traffic signal all-red fault with police intervention"
    return make_bundle(
        scenario_id,
        l_path("L2_infrastructure", "interaction" if scenario_id == "L2-5_v1" else "failure", scenario_id),
        "infrastructure",
        desc,
        specs,
        scenes,
        params,
        events,
        validation_rules=[
            {
                "rule": "event_chain_min",
                "min_count": 3,
                "description": "Infrastructure scenarios include fault, agent response, and recovery or control",
            }
        ],
    )


def build_l3(scenario_id: str, idx: int) -> ScenarioBundle:
    ox = 36.0 + idx * 12.0
    oy = 44.0 + idx * 5.0
    specs: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    sid = scenario_id.lower().replace("-", "_")
    if scenario_id == "L3-1_v1":
        uav = "uav_roadwork_inspector_l3_1"
        car = "blocked_car_l3_1"
        prop_entities: list[dict[str, Any]] = []
        work_base = LANES.nearest_to_xy(*p(ox, oy, 0)[:2])
        work_edge = work_base.edge_id
        work_s = work_base.s_m
        for i in range(3):
            ent = f"barrier_l3_1_{i + 1:02d}"
            pos = p(ox + i * 4.5, oy, 0)
            _, scene = add(specs, scenes, lane_entity(ent, "prop.roadwork.barrier.v1", "roadwork_prop", work_edge, work_s + i * 5, 1.5, pos, 90, "staged", activation_tick=220, prefer_edge_hint=True))
            prop_entities.append(scene)
        for i, sx in enumerate([-3.0, 12.5]):
            ent = f"fence_l3_1_{i + 1:02d}"
            pos = p(ox + sx, oy + 1.6, 0)
            _, scene = add(specs, scenes, lane_entity(ent, "prop.roadwork.construction_fence.v1", "roadwork_prop", work_edge, work_s - 2 + i * 18, 1.5, pos, 90, "staged", activation_tick=220, prefer_edge_hint=True))
            prop_entities.append(scene)
        for i in range(5):
            ent = f"cone_l3_1_{i + 1:02d}"
            pos = p(ox - 2 + i * 3.6, oy - 2.4, 0)
            _, scene = add(specs, scenes, lane_entity(ent, "prop.roadwork.traffic_cone.v1", "roadwork_prop", work_edge, work_s - 4 + i * 4, 1.5, pos, 90, "staged", activation_tick=220, prefer_edge_hint=True))
            prop_entities.append(scene)
        _, car_scene = add(specs, scenes, lane_entity(car, "vehicle.ground.boxcar.v1", "vehicle", work_edge, work_s + 12, 0.0, p(ox + 4, oy - 5, 0), 90, "blocked", prefer_edge_hint=True))
        car_start = scene_pos(car_scene)
        car_mid = road_center_point(p(ox + 2, oy - 14, 0))
        car_end = road_center_point(p(ox + 22, oy - 18, 0))
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", p(ox - 18, oy - 12, 32), 65, "inspection"))
        spawn_actions = [spawn_from_scene(f"spawn_{entity_scene['entity_id']}", entity_scene) for entity_scene in prop_entities]
        events = [
            event("roadwork_barriers_spawn", tick(220), spawn_actions, 1, scenario_id, "Roadwork barriers and cones appear", "dynamic_constraint", [e["entity_id"] for e in prop_entities[:4]], "warning"),
            event("lane_closure_active", fired("roadwork_barriers_spawn"), [visual("set_blocked_car_waiting", car, "blocked"), screenshot("capture_lane_closure")], 2, scenario_id, "Lane closure becomes active", "vehicle", [car], "warning"),
            event("vehicle_detour", fired("lane_closure_active"), [move("move_blocked_car_detour", car, [car_start, car_mid, car_end], 6.0)], 3, scenario_id, "Blocked vehicle detours around construction", "vehicle", [car], "info"),
            event("uav_inspection_report", fired("vehicle_detour"), [move("move_uav_roadwork_scan", uav, [p(ox - 18, oy - 12, 32), p(ox + 4, oy - 4, 28), p(ox + 15, oy + 3, 28)], 5.0), screenshot("capture_roadwork_report")], 4, scenario_id, "UAV reports lane closure geometry", "uav_mission", [uav], "info"),
        ]
        desc = "Road construction lane closure with vehicle detour"
        rules = [
            {"rule": "roadwork_asset_count", "barrier_min": 3, "fence_min": 2, "cone_min": 5, "description": "Roadwork setup includes required barrier, fence, and cone counts"},
            {"rule": "lane_anchor_required", "asset_prefix": "prop.roadwork", "description": "Roadwork props use lane_anchor placement at lateral offset 1.5m"},
        ]
    elif scenario_id.startswith("L3-2"):
        uav = f"uav_{sid}"
        nfz = f"temporary_nfz_{sid}"
        start = p(ox - 10, oy - 12, 31)
        approach = p(ox + 8, oy + 2, 31)
        center = p(ox + 14, oy + 5, 28)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 45, "patrol", route=[approach]))
        _, nfz_scene = add(specs, scenes, box_entity(nfz, "trigger.no_fly.box.v1", "airspace_constraint", center, [12, 9, 15], activation_tick=280))
        events = [
            event("temporary_nfz_declared", tick(280), [spawn_from_scene("spawn_temporary_nfz", nfz_scene, visual_state={"mode": "active"})], 1, scenario_id, "Temporary no-fly zone declared mid-operation", "dynamic_constraint", [nfz], "warning"),
            event("uav_approaches_temporary_nfz", fired("temporary_nfz_declared"), [move("move_uav_to_nfz_edge", uav, [start, approach], 7.0)], 2, scenario_id, "UAV approaches newly declared NFZ", "uav_mission", [uav, nfz], "warning"),
            event("nfz_proximity_alert", prox(uav, nfz, 12.0, 3, metric="xy"), [move("move_uav_nfz_reroute", uav, [approach, p(ox - 2, oy + 18, 34), p(ox + 24, oy + 28, 34)], 8.5), screenshot("capture_temporary_nfz")], 3, scenario_id, "UAV reroutes around temporary NFZ", "uav_mission", [uav, nfz], "info"),
        ]
        desc = "Temporary no-fly zone activation and UAV reroute"
        rules = [{"rule": "dynamic_spawn_required", "entity_id": nfz, "description": "Temporary NFZ is spawned by event, not pre-activated statically"}]
    else:
        uav = f"uav_{sid}"
        hazard = f"hazmat_zone_{sid}"
        ped_a = f"ped_hazmat_{sid}_a"
        ped_b = f"ped_hazmat_{sid}_b"
        ambulance = f"ambulance_{sid}"
        center = p(ox + 8, oy + 6, 3)
        _, hazard_scene = add(specs, scenes, box_entity(hazard, "trigger.hazard.generic.box.v1", "hazard_zone", center, [13, 10, 4], activation_tick=240))
        _, ped_a_scene = add(specs, scenes, sidewalk_entity(ped_a, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_15", 48, 1.2, p(ox + 5, oy + 3, 0), 0, "walking"))
        _, ped_b_scene = add(specs, scenes, sidewalk_entity(ped_b, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_15", 54, 1.2, p(ox + 11, oy + 4, 0), 180, "walking"))
        ped_a_start = scene_pos(ped_a_scene)
        ped_b_start = scene_pos(ped_b_scene)
        ped_a_safe = shifted_sidewalk_point(ped_a_start, -12, 12, 2.0)
        ped_b_safe = shifted_sidewalk_point(ped_b_start, 12, 12, 2.0)
        _, ambulance_scene = add(specs, scenes, lane_entity(ambulance, "vehicle.emergency.ambulance.v1", "vehicle", "cg_edge_16", 26, 0.0, p(ox - 26, oy - 10, 0), 70, "response", visual_state={"mode": "response", "lights_on": True}))
        ambulance_start = scene_pos(ambulance_scene)
        ambulance_arrival = road_center_point(p(ox - 4, oy - 2, 0))
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", p(ox - 8, oy - 15, 33), 45, "monitor"))
        events = [
            event("hazmat_leak", tick(240), [spawn_from_scene("spawn_hazmat_zone", hazard_scene, visual_state={"mode": "isolation_active"})], 1, scenario_id, "Hazmat leak declares isolation zone", "dynamic_constraint", [hazard], "critical"),
            event("ambulance_arrival", fired("hazmat_leak"), [move("move_ambulance_hazmat", ambulance, [ambulance_start, ambulance_arrival], 11.0)], 2, scenario_id, "Ambulance arrives at isolation perimeter", "vehicle", [ambulance, hazard], "warning"),
            event("pedestrian_evacuation", fired("ambulance_arrival"), [move("move_ped_a_safe", ped_a, [ped_a_start, ped_a_safe], 1.6, activity_type="evacuating"), move("move_ped_b_safe", ped_b, [ped_b_start, ped_b_safe], 1.4, activity_type="evacuating")], 3, scenario_id, "Pedestrians evacuate from hazmat zone", "pedestrian", [ped_a, ped_b, hazard], "warning"),
            event("uav_external_monitor", fired("ambulance_arrival"), [move("move_uav_hazmat_orbit", uav, [p(ox - 8, oy - 15, 33), p(ox + 2, oy - 5, 30), p(ox + 18, oy + 18, 30)], 5.0), screenshot("capture_hazmat_monitor")], 4, scenario_id, "UAV monitors hazmat zone from outside", "uav_mission", [uav, hazard], "info"),
        ]
        desc = "Hazmat leak isolation, evacuation, and UAV monitoring"
        rules = [{"rule": "hazmat_min_entities", "pedestrian_min": 2, "ambulance_min": 1, "uav_min": 1, "description": "Hazmat scenario includes pedestrians, ambulance, and UAV"}]
    return make_bundle(
        scenario_id,
        l_path("L3_dynamic_constraints", "interaction", scenario_id),
        "dynamic_constraints",
        desc,
        specs,
        scenes,
        {"incident_tick": 240 if scenario_id.startswith("L3-3") else 220},
        events,
        validation_rules=rules,
    )


def build_l4(scenario_id: str, idx: int) -> ScenarioBundle:
    ox = 42.0 + idx * 5.5
    oy = 82.0 + idx * 3.0
    specs: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    sid = scenario_id.lower().replace("-", "_")
    rules: list[dict[str, Any]] = []
    params: dict[str, Any] = {"incident_tick": 260, "resolution_tick": 520}
    if scenario_id.startswith("L4-1_"):
        uav_a = f"uav_conflict_{sid}_a"
        uav_b = f"uav_conflict_{sid}_b"
        a0, b0 = p(ox - 20, oy, 25), p(ox + 18, oy + 12, 30)
        meet = p(ox, oy + 6, 28)
        add(specs, scenes, world_entity(uav_a, "uav.inspect.quad.v1", "uav", a0, 70, "patrol"))
        add(specs, scenes, world_entity(uav_b, "uav.airsim.flying_pawn.v1", "uav", b0, 250, "patrol"))
        events = [
            event("converging_approach", tick(240), [move("move_uav_a_converge", uav_a, [a0, p(ox - 5, oy + 4, 25), meet], 8.0), move("move_uav_b_converge", uav_b, [b0, p(ox + 6, oy + 8, 30), meet], 7.5)], 1, scenario_id, "Two UAVs converge from separated starts", "uav_mission", [uav_a, uav_b], "warning"),
            event("uav_conflict_resolution", prox(uav_a, uav_b, 8.0, 2, metric="xy_plus_z", horizontal_distance_m=8.0, vertical_distance_m=35.0), [visual("set_uav_a_hover", uav_a, "hover"), move("move_uav_b_altitude_reroute", uav_b, [meet, p(ox + 12, oy + 18, 36)], 9.0), screenshot("capture_uav_conflict")], 2, scenario_id, "UAV-UAV proximity conflict triggers separation", "uav_mission", [uav_a, uav_b], "critical"),
            event("resume_patrols", fired("uav_conflict_resolution"), [move("move_uav_a_resume", uav_a, [meet, p(ox - 18, oy + 20, 27)], 6.0), move("move_uav_b_resume", uav_b, [p(ox + 12, oy + 18, 36), p(ox + 24, oy + 22, 32)], 7.0)], 3, scenario_id, "UAVs separate and resume patrol", "uav_mission", [uav_a, uav_b], "info"),
        ]
        desc = "UAV-UAV converging conflict and separation"
        rules.append({"rule": "uav_start_separation", "min_horizontal_distance_m": 15.0, "description": "UAV starts are separated and altitudes differ"})
    elif scenario_id.startswith("L4-2_"):
        uav = f"uav_facade_{sid}"
        facade = f"facade_anchor_{sid}"
        start = p(ox - 25, oy - 9, 31)
        near = p(ox + 2, oy + 6, 18)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 55, "navigation_fault"))
        add(specs, scenes, facade_entity(facade, BUILDINGS.id_for(40 + idx), p(ox + 4, oy + 7, 16), [1.0, -0.4, 0.0]))
        events = [
            event("facade_approach", tick(250), [move("move_uav_facade_approach", uav, [start, p(ox - 12, oy - 2, 26), near], 7.0)], 1, scenario_id, "UAV approaches building facade on faulty route", "uav_mission", [uav, facade], "warning"),
            event("facade_proximity", prox(uav, facade, 8.0, 3), [move("move_uav_facade_evasion", uav, [near, p(ox - 4, oy + 14, 34)], 10.0), screenshot("capture_facade_evasion")], 2, scenario_id, "Emergency evasion near facade", "uav_mission", [uav, facade], "critical"),
            event("facade_recovery", fired("facade_proximity"), [move("move_uav_facade_recover", uav, [p(ox - 4, oy + 14, 34), p(ox - 22, oy + 18, 32)], 7.0)], 3, scenario_id, "UAV recovers after facade near strike", "uav_mission", [uav], "info"),
        ]
        desc = "UAV building facade near strike"
        rules.append({"rule": "facade_approach_distance", "max_distance_m": 3.0, "description": "UAV trajectory nearest point is within 3m of facade anchor"})
    elif scenario_id.startswith("L4-3_"):
        uav = f"uav_forced_landing_{sid}"
        ped_a = f"ped_landing_{sid}_a"
        ped_b = f"ped_landing_{sid}_b"
        _, ped_a_scene = add(specs, scenes, sidewalk_entity(ped_a, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_19", 24, GATHERING_MIN_OFFSET_FROM_CURB_M, p(ox + 2, oy + 2, 0), 0, "chatting"))
        _, ped_b_scene = add(specs, scenes, sidewalk_entity(ped_b, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_19", 28, GATHERING_MIN_OFFSET_FROM_CURB_M, p(ox + 7, oy + 2, 0), 180, "chatting"))
        ped_a_start = scene_pos(ped_a_scene)
        ped_b_start = scene_pos(ped_b_scene)
        landing_x = (ped_a_start[0] + ped_b_start[0]) / 2.0
        landing_y = (ped_a_start[1] + ped_b_start[1]) / 2.0
        start = q(landing_x - 28.0, landing_y - 22.0, 30)
        descent_mid = q(landing_x - 12.0, landing_y - 9.0, 18)
        low = q(landing_x, landing_y, 8)
        land = q(landing_x, landing_y, 3)
        crowd_origin = gathering_point(q(landing_x, landing_y, GROUND_Z_M), [650, 420, 0])
        extra_crowd_ids = add_static_ped_cohort(specs, scenes, sid, "ped_landing_crowd", crowd_origin, 6, activity_type="chatting")
        crowd_ids = [ped_a, ped_b, *extra_crowd_ids]
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 45, "fault"))
        ped_a_evade = shifted_sidewalk_point(ped_a_start, -8, 8, GATHERING_MIN_OFFSET_FROM_CURB_M)
        ped_b_evade = shifted_sidewalk_point(ped_b_start, 8, 8, GATHERING_MIN_OFFSET_FROM_CURB_M)
        events = [
            event("crowd_present", tick(0), [ped_activity(f"set_{ped_id}_crowd_chatting", ped_id, "chatting") for ped_id in crowd_ids], 0, scenario_id, "Ground crowd is visible before the UAV fault begins", "pedestrian", crowd_ids, "info"),
            event("forced_landing_fault", tick(230), [visual("set_uav_forced_landing_fault", uav, "propulsion_fault")], 1, scenario_id, "UAV fault appears near a ground crowd", "uav_mission", [uav, ped_a, ped_b], "critical"),
            event("descent_to_low_altitude", fired("forced_landing_fault"), [move("move_uav_forced_descent", uav, [start, descent_mid, low, land], FORCED_LANDING_DESCENT_SPEED_MPS)], 2, scenario_id, "UAV descends from 30m to low height", "uav_mission", [uav], "critical"),
            event("crowd_proximity_response", prox(uav, ped_a, 7.5, 2, metric="xy_plus_z", horizontal_distance_m=7.5, vertical_distance_m=10.0), [move("move_ped_a_evade_landing", ped_a, [ped_a_start, ped_a_evade], 2.0, post_activity_type="quarrel"), move("move_ped_b_evade_landing", ped_b, [ped_b_start, ped_b_evade], 2.0, post_activity_type="quarrel"), screenshot("capture_forced_landing")], 3, scenario_id, "Crowd evades forced landing zone and yells at the forced landing site", "pedestrian", [ped_a, ped_b, uav], "warning"),
            event("safe_landing", fired("crowd_proximity_response"), [move("move_uav_safe_touchdown", uav, [land, q(land[0], land[1], 0.8)], FORCED_LANDING_TOUCHDOWN_SPEED_MPS)], 4, scenario_id, "UAV completes safe landing", "uav_mission", [uav], "info"),
        ]
        desc = "UAV forced landing near crowd"
        rules.append({"rule": "forced_landing_descent_profile", "z_sequence": [30, 8, 3], "description": "UAV descends through low-altitude forced landing profile"})
    elif scenario_id.startswith("L4-4_"):
        uav = f"uav_falls_vehicle_{sid}"
        car = f"car_under_uav_{sid}"
        u0 = p(ox - 18, oy - 5, 20)
        c0 = p(ox + 4, oy - 18, 0)
        impact_u = p(ox + 2, oy + 2, 2.6)
        impact_c = p(ox + 2, oy + 2, 0)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", u0, 70, "unstable"))
        _, car_scene = add(specs, scenes, lane_entity(car, "vehicle.ground.boxcar.v1", "vehicle", "cg_edge_20", 46, 0, c0, 0, "moving"))
        car_start = scene_pos(car_scene)
        events = [
            event("crossing_trajectories", tick(240), [move("move_uav_falling_cross", uav, [u0, p(ox - 7, oy - 1, 12), impact_u], 5.5), move("move_car_crossing_path", car, [car_start, p(ox + 3, oy - 7, 0), impact_c], 7.0)], 1, scenario_id, "UAV and vehicle trajectories cross", "uav_mission", [uav, car], "critical"),
            event("vehicle_roof_contact", prox(uav, car, 3.0, 1), [move("move_vehicle_emergency_brake", car, [impact_c, p(ox + 2.5, oy + 2.5, 0)], 0.5), move("move_uav_debris_ground", uav, [impact_u, p(ox + 4, oy + 3, 0.4)], 1.0), visual("set_uav_debris", uav, "debris"), screenshot("capture_uav_vehicle_contact")], 2, scenario_id, "UAV contacts vehicle roof and vehicle brakes", "vehicle", [uav, car], "critical"),
        ]
        desc = "UAV falls onto moving ground vehicle"
        rules.append({"rule": "trajectory_intersection_required", "description": "UAV and vehicle trajectories intersect rather than running parallel"})
    elif scenario_id.startswith("L4-5_"):
        uav = f"uav_ped_nearmiss_{sid}"
        ped = f"ped_nearmiss_{sid}"
        _, ped_scene = add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_21", 35, 1.2, p(ox + 2, oy + 1, 0), 15, "walking"))
        ped_start = scene_pos(ped_scene)
        start = q(ped_start[0] - 1.394, ped_start[1] + 15.939, 30)
        descent_mid = q(ped_start[0] - 0.697, ped_start[1] + 7.97, 15)
        over = q(ped_start[0], ped_start[1], 5)
        pull_up = q(ped_start[0] + 1.389, ped_start[1] + 7.878, 24)
        ped_clear = shifted_sidewalk_point(ped_start, 10, 10, 1.8)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 45, "descent"))
        events = [
            event("uav_low_descent", tick(230), [ped_activity("set_ped_phone_distraction", ped, "phone_call"), move("move_uav_ped_descent", uav, [start, descent_mid, over], 4.5)], 1, scenario_id, "UAV descends toward pedestrian head height while pedestrian is distracted by a phone call", "uav_mission", [uav, ped], "warning"),
            event("pedestrian_near_miss", prox(uav, ped, 5.0, 2, metric="xy_plus_z", horizontal_distance_m=3.0, vertical_distance_m=6.0), [visual("set_uav_hover_nearmiss", uav, "hover"), move("move_uav_pull_up_after_nearmiss", uav, [over, pull_up], 6.0), screenshot("capture_ped_nearmiss")], 2, scenario_id, "UAV near-miss with pedestrian triggers pull-up", "uav_mission", [uav, ped], "critical"),
            event("pedestrian_clears_area", fired("pedestrian_near_miss"), [move("move_ped_clear_nearmiss", ped, [ped_start, ped_clear], 1.5)], 3, scenario_id, "Pedestrian clears UAV operating area", "pedestrian", [ped], "info"),
        ]
        desc = "UAV low-altitude pedestrian near-miss"
    elif scenario_id.startswith("L4-6_"):
        ped = f"ped_jaywalk_{sid}"
        car = f"car_jaywalk_{sid}"
        p0 = p(ox - 8, oy + 2, 0)
        c0 = p(ox + 20, oy - 4, 0)
        _, ped_scene = add(specs, scenes, crosswalk_entity(ped, "pedestrian.cityops.basic.v1", f"crosswalk_{sid}", "west", p0, 90))
        p0 = scene_pos(ped_scene)
        road = ped_scene["placement"]["roadway_center_position_enu_m"]
        retreat = ped_scene["placement"]["opposite_curb_position_enu_m"]
        _, car_scene = add(specs, scenes, lane_entity(car, "vehicle.ground.boxcar.v1", "vehicle", "cg_edge_22", 58, 0, c0, 250, "moving"))
        car_start = scene_pos(car_scene)
        car_conflict = road
        car_stop = q(road[0] + 1.0, road[1] + 0.3, GROUND_Z_M)
        events = [
            event("ped_enters_roadway", tick(220), [move("move_ped_from_crosswalk_to_lane", ped, [p0, road], 1.6, activity_type="texting_walk"), move("move_car_toward_crosswalk", car, [car_start, car_conflict], 8.0)], 1, scenario_id, "Texting pedestrian moves from sidewalk into roadway", "pedestrian", [ped, car], "warning"),
            event("vehicle_ped_proximity", prox(ped, car, 4.0, 2, metric="xy_plus_z", horizontal_distance_m=4.0, vertical_distance_m=1.5), [move("move_car_hard_brake_ped", car, [car_conflict, car_stop], 0.8), screenshot("capture_ped_vehicle_conflict")], 2, scenario_id, "Vehicle brakes for pedestrian conflict", "vehicle", [ped, car], "critical"),
            event("ped_retreats", fired("vehicle_ped_proximity"), [move("move_ped_retreat_sidewalk", ped, [road, retreat], 1.8)], 3, scenario_id, "Pedestrian retreats from roadway", "pedestrian", [ped], "info"),
        ]
        desc = "Pedestrian jaywalk vehicle conflict"
    elif scenario_id.startswith("L4-7_"):
        ped = f"ped_fall_{sid}"
        uav = f"uav_detect_{sid}"
        ambulance = f"ambulance_fall_{sid}"
        _, ped_scene = add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_23", 42, 1.2, p(ox + 3, oy + 2, 0), 0, "walking"))
        ped_start = scene_pos(ped_scene)
        uav_start = q(ped_start[0] - 28.0, ped_start[1] - 20.0, 31)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 55, "patrol"))
        ambulance_arrival = road_center_point(ped_start)
        arrival_sample = LANES.nearest_to_xy(ambulance_arrival[0], ambulance_arrival[1])
        ambulance_s = max(0.0, arrival_sample.s_m - 42.0)
        ambulance_start_hint = _offset_from_lane(LANES.resolve_edge_s(arrival_sample.edge_id, ambulance_s), 0.0, GROUND_Z_M)
        _, ambulance_scene = add(
            specs,
            scenes,
            lane_entity(
                ambulance,
                "vehicle.emergency.ambulance.v1",
                "vehicle",
                arrival_sample.edge_id,
                ambulance_s,
                0.0,
                ambulance_start_hint,
                arrival_sample.yaw_deg,
                "response",
                visual_state={"mode": "response", "lights_on": True},
                prefer_edge_hint=True,
            ),
        )
        ambulance_start = scene_pos(ambulance_scene)
        uav_observe = q(ped_start[0], ped_start[1], 16)
        events = [
            event("pedestrian_fall", tick(230), [play("play_pedestrian_fall", ped)], 1, scenario_id, "Pedestrian fall animation starts", "pedestrian", [ped], "critical"),
            event("uav_detects_fall", fired("pedestrian_fall"), [move("move_uav_detect_fall", uav, [uav_start, uav_observe], 5.0), screenshot("capture_fall_detection")], 2, scenario_id, "UAV detects fallen pedestrian", "uav_mission", [uav, ped], "warning"),
            event("ambulance_response", fired("uav_detects_fall"), [move("move_ambulance_to_fall", ambulance, [ambulance_start, ambulance_arrival], 12.0)], 3, scenario_id, "Ambulance responds to detected fall", "vehicle", [ambulance, ped], "warning", intent="ambulance"),
            event("incident_documented", fired("ambulance_response"), [screenshot("capture_fall_response")], 4, scenario_id, "UAV documents emergency response", "uav_mission", [uav, ambulance, ped], "info"),
        ]
        desc = "Pedestrian fall detected by UAV and response chain"
    elif scenario_id.startswith("L4-8_"):
        uav = f"uav_crowd_monitor_{sid}"
        uav_start = p(ox - 18, oy - 10, 34)
        crowd_extent = [900, 600, 0]
        safe_extent = [1000, 500, 0]
        crowd_origin = gathering_point(p(ox + 3, oy + 3, 0), crowd_extent)
        safe_origin = gathering_point(p(ox + 24, oy + 19, 0), safe_extent)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 50, "monitor"))
        add(specs, scenes, world_entity(f"crowd_anchor_{sid}", "semantic.spawn_zone", "crowd_anchor", crowd_origin, 0))
        cohort = add_evacuation_ped_cohort(specs, scenes, sid, "ped_evac", crowd_origin, safe_origin, 8)
        events = [
            event("crowd_generated", tick(0), [], 1, scenario_id, "Explicit pedestrian cohort is present in evacuation zone at episode start", "pedestrian", [f"crowd_anchor_{sid}"], "info"),
            event("evacuation_triggered", tick(220), evacuation_move_actions(cohort, "move_crowd_evacuation"), 2, scenario_id, "Visible pedestrian cohort walks to safe evacuation zone", "pedestrian", [f"crowd_anchor_{sid}"], "warning"),
            event("uav_evacuation_monitor", fired("evacuation_triggered"), [move("move_uav_monitor_evacuation", uav, [uav_start, q(crowd_origin[0], crowd_origin[1], 30), q(safe_origin[0], safe_origin[1], 30)], 5.0), screenshot("capture_crowd_evacuation")], 3, scenario_id, "UAV monitors evacuation movement", "uav_mission", [uav], "info"),
            event("crowd_safe_hold", fired("uav_evacuation_monitor"), [screenshot("capture_crowd_safe_hold")], 4, scenario_id, "Crowd evacuation completes without removing visible agents", "pedestrian", [uav], "info"),
        ]
        desc = "Crowd evacuation with UAV monitoring"
        rules.append({"rule": "explicit_pedestrian_count_min", "min_count": 8, "description": "Crowd evacuation scenario uses at least 8 individually controlled pedestrians"})
    elif scenario_id.startswith("L4-9_"):
        car_a = f"car_intersection_{sid}_a"
        car_b = f"car_intersection_{sid}_b"
        a0 = p(ox - 24, oy, 0)
        b0 = p(ox + 2, oy + 24, 0)
        junction = p(ox, oy + 2, 0)
        _, car_a_scene = add(specs, scenes, lane_entity(car_a, "vehicle.ground.boxcar.v1", "vehicle", "cg_edge_25", 65, 0.0, a0, 80, "moving"))
        _, car_b_scene = add(specs, scenes, lane_entity(car_b, "vehicle.emergency.suv.v1", "vehicle", "cg_edge_26", 40, 0.0, b0, 185, "moving"))
        a0 = scene_pos(car_a_scene)
        b0 = scene_pos(car_b_scene)
        events = [
            event("vehicles_enter_intersection", tick(230), [move("move_car_a_intersection", car_a, [a0, p(ox - 7, oy + 1, 0), junction], 8.0), move("move_car_b_intersection", car_b, [b0, p(ox + 1, oy + 9, 0), junction], 7.5)], 1, scenario_id, "Two vehicles enter the same intersection from different lanes", "vehicle", [car_a, car_b], "warning"),
            event("vehicle_collision_warning", prox(car_a, car_b, 6.0, 2), [move("move_car_a_brake", car_a, [junction, p(ox - 1, oy + 2, 0)], 0.5), move("move_car_b_brake", car_b, [junction, p(ox + 1, oy + 3, 0)], 0.5), screenshot("capture_vehicle_conflict")], 2, scenario_id, "Vehicle-vehicle proximity triggers simultaneous braking", "vehicle", [car_a, car_b], "critical"),
        ]
        desc = "Vehicle-vehicle intersection conflict"
    elif scenario_id.startswith("L4-10_"):
        ambulance = f"ambulance_priority_{sid}"
        car = f"civilian_yield_{sid}"
        uav = f"uav_priority_monitor_{sid}"
        a0 = p(ox - 28, oy - 4, 0)
        c0 = p(ox - 6, oy - 2, 0)
        add(specs, scenes, lane_entity(ambulance, "vehicle.emergency.ambulance.v1", "vehicle", "cg_edge_27", 42, 0.0, a0, 76, "response", visual_state={"mode": "response", "lights_on": True}))
        add(specs, scenes, lane_entity(car, "vehicle.ground.boxcar.v1", "vehicle", "cg_edge_27", 58, 0.0, c0, 76, "moving"))
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", p(ox - 12, oy + 12, 32), 60, "monitor"))
        events = [
            event("ambulance_priority_approach", tick(220), [visual("set_ambulance_lights_on", ambulance, "response", lights_on=True), move("move_ambulance_priority", ambulance, [a0, p(ox - 12, oy - 3, 0), p(ox + 12, oy - 1, 0)], 13.0)], 1, scenario_id, "Ambulance approaches with lights active", "vehicle", [ambulance, car], "warning"),
            event("civilian_vehicle_yields", fired("ambulance_priority_approach"), [move("move_civilian_yield", car, [c0, p(ox - 3, oy + 4, 0), p(ox - 1, oy + 8, 0)], 3.0)], 2, scenario_id, "Civilian vehicle yields to ambulance", "vehicle", [ambulance, car], "info"),
            event("uav_priority_capture", fired("civilian_vehicle_yields"), [move("move_uav_priority_monitor", uav, [p(ox - 12, oy + 12, 32), p(ox + 6, oy + 10, 28)], 5.0), screenshot("capture_ambulance_priority")], 3, scenario_id, "UAV monitors priority passage", "uav_mission", [uav, ambulance], "info"),
        ]
        desc = "Ambulance priority passage with civilian yield"
        rules.append({"rule": "ambulance_lights_on", "entity_id": ambulance, "description": "Ambulance starts and moves with lights_on true"})
    else:
        av = f"av_fault_{sid}"
        uav = f"uav_av_report_{sid}"
        start = p(ox - 16, oy - 1, 0)
        stop = p(ox + 2, oy, 0)
        add(specs, scenes, lane_entity(av, "vehicle.ground.boxcar.v1", "vehicle", "cg_edge_28", 47, 0.0, start, 82, "autonomous"))
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", p(ox - 20, oy + 11, 33), 70, "patrol"))
        events = [
            event("av_sensor_fault", tick(230), [visual("set_av_sensor_fault", av, "sensor_fault")], 1, scenario_id, "Autonomous vehicle sensor fault detected", "vehicle", [av], "warning"),
            event("av_safe_stop", fired("av_sensor_fault"), [move("move_av_safe_stop", av, [start, p(ox - 5, oy - 0.4, 0), stop], 2.0), visual("set_av_safe_stop", av, "safe_stop")], 2, scenario_id, "AV performs safe stop in lane", "vehicle", [av], "critical"),
            event("lane_blockage_report", fired("av_safe_stop"), [move("move_uav_blockage_report", uav, [p(ox - 20, oy + 11, 33), p(ox, oy + 6, 25)], 5.0), screenshot("capture_av_lane_blockage")], 3, scenario_id, "UAV reports lane blockage", "uav_mission", [uav, av], "warning"),
            event("av_recovery_hold", fired("lane_blockage_report"), [visual("set_av_hazard_lights", av, "blocked_lane", lights_on=True)], 4, scenario_id, "AV remains in safe stop with hazard state", "vehicle", [av], "info"),
        ]
        desc = "AV sensor failure, safe stop, and UAV report"
    subdir = "collision"
    if scenario_id.startswith(("L4-3_", "L4-11_")):
        subdir = "failure"
    elif scenario_id.startswith(("L4-7_", "L4-8_", "L4-10_")):
        subdir = "interaction"
    return make_bundle(
        scenario_id,
        l_path("L4_agents", subdir, scenario_id),
        "agents",
        desc,
        specs,
        scenes,
        params,
        events,
        validation_rules=rules,
    )


def build_l5(scenario_id: str, idx: int) -> ScenarioBundle:
    ox = 30.0 + idx * 8.0
    oy = 146.0 + idx * 3.5
    specs: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    sid = scenario_id.lower().replace("-", "_")
    uav = f"uav_weather_{sid}"
    start = p(ox, oy, 32)
    add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 35, "patrol"))
    car = f"weather_car_{sid}"
    ped = f"weather_ped_{sid}"
    _, car_scene = add(specs, scenes, lane_entity(car, "vehicle.ground.boxcar.v1", "vehicle", "cg_edge_29", 36, 0.0, p(ox + 8, oy - 8, 0), 90, "moving"))
    car_start = scene_pos(car_scene)
    car_rain_end = road_center_point(p(ox + 15, oy - 7, 0))
    add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_29", 40, 1.2, p(ox + 2, oy + 5, 0), 0, "walking"))
    if scenario_id.startswith("L5-1"):
        threshold = 0.5
        trigger = weather("rain", "gte", threshold, 5)
        profile = {"initial": "clear", "transitions": [{"tick": 180, "profile": "rain", "overrides": {"rain": 0.55, "visibility_m": 2200.0}}]}
        events = [
            event("weather_onset", tick(180), [set_weather("set_rain_onset", "rain", rain=0.55, visibility_m=2200.0)], 1, scenario_id, "Rain onset is applied before threshold confirmation", "weather", [uav, car], "info"),
            event("rain_condition_met", trigger, [set_weather("set_rain_moderate", "rain", rain=0.55, visibility_m=2200.0)], 1, scenario_id, "Rain threshold reached", "weather", [uav, car], "warning"),
            event("rain_speed_reduction", fired("rain_condition_met"), [move("move_uav_rain_slowdown", uav, [start, p(ox + 10, oy + 8, 32)], 3.0), move("move_car_wet_slowdown", car, [car_start, car_rain_end], 3.0)], 2, scenario_id, "UAV and vehicle slow under rain", "uav_mission", [uav, car], "warning"),
            event("rain_intensifies", fired("rain_speed_reduction"), [set_weather("set_rain_heavy_visibility", "rain", rain=0.82, visibility_m=900.0)], 3, scenario_id, "Rain intensifies and visibility drops", "weather", [uav, car], "critical"),
            event("rain_recovery", fired("rain_intensifies"), [move("move_uav_rain_recover", uav, [p(ox + 10, oy + 8, 32), p(ox - 6, oy + 16, 34)], 5.0), set_weather("set_rain_recovering", "clear", rain=0.2, visibility_m=5000.0)], 4, scenario_id, "Rain recovers and UAV exits degraded area", "weather", [uav], "info"),
        ]
        desc = "Rain cascade with UAV and ground traffic slowdown"
    elif scenario_id.startswith("L5-2"):
        trigger = weather("fog", "gte", 0.5, 5)
        profile = {"initial": "clear", "transitions": [{"tick": 190, "profile": "fog", "overrides": {"fog": 0.6, "visibility_m": 650.0}}]}
        events = [
            event("weather_onset", tick(190), [set_weather("set_fog_onset", "fog", fog=0.6, visibility_m=650.0)], 1, scenario_id, "Fog onset is applied before threshold confirmation", "weather", [uav], "info"),
            event("fog_condition_met", trigger, [set_weather("set_fog_dense", "fog", fog=0.62, visibility_m=650.0)], 1, scenario_id, "Fog threshold reached", "weather", [uav], "warning"),
            event("visual_navigation_failure", fired("fog_condition_met"), [visual("set_uav_visual_nav_degraded", uav, "visual_nav_degraded"), move("move_uav_fog_slow_hold", uav, [start, p(ox + 6, oy + 4, 32)], 2.5)], 2, scenario_id, "Visual navigation degrades in fog", "uav_mission", [uav], "critical"),
            event("fog_worsens", fired("visual_navigation_failure"), [set_weather("set_fog_worse_visibility", "fog", fog=0.85, visibility_m=260.0), screenshot("capture_fog_failure")], 3, scenario_id, "Fog worsens and visibility collapses", "weather", [uav], "critical"),
            event("fog_mission_abort", fired("fog_worsens"), [move("move_uav_fog_abort", uav, [p(ox + 6, oy + 4, 32), p(ox - 8, oy + 12, 34)], 4.0)], 4, scenario_id, "UAV aborts mission under dense fog", "uav_mission", [uav], "warning"),
        ]
        desc = "Sudden fog visual navigation failure"
    elif scenario_id.startswith("L5-3"):
        trigger = weather("wind_speed", "gte", 12.0, 4)
        profile = {"initial": "clear", "transitions": [{"tick": 210, "profile": "wind", "overrides": {"wind_speed": 12.5}}]}
        bag = f"payload_bag_{sid}"
        add(specs, scenes, world_entity(bag, "prop.service.delivery_bag.v1", "prop", p(ox + 1, oy + 1, 28), 0, "attached"))
        events = [
            event("weather_onset", tick(210), [set_weather("set_wind_onset", "wind", wind_speed=12.5)], 1, scenario_id, "Wind onset is applied before threshold confirmation", "weather", [uav, bag], "info"),
            event("wind_condition_met", trigger, [set_weather("set_crosswind", "wind", wind_speed=12.5)], 1, scenario_id, "Crosswind threshold reached", "weather", [uav, bag], "warning"),
            event("payload_swing", fired("wind_condition_met"), [move("move_payload_swing_path", bag, [p(ox + 1, oy + 1, 28), p(ox + 4, oy - 3, 27), p(ox - 2, oy + 5, 29)], 2.0), visual("set_uav_attitude_unstable", uav, "attitude_unstable")], 2, scenario_id, "Payload swings and UAV attitude becomes abnormal", "uav_mission", [uav, bag], "warning"),
            event("wind_gust_intensifies", fired("payload_swing"), [set_weather("set_wind_gust", "wind", wind_speed=18.0), move("move_uav_wind_hold", uav, [start, p(ox + 3, oy + 5, 33)], 2.0), screenshot("capture_wind_payload")], 3, scenario_id, "Crosswind intensifies into gust", "weather", [uav, bag], "critical"),
            event("wind_recovery", fired("wind_gust_intensifies"), [move("move_uav_wind_recovery", uav, [p(ox + 3, oy + 5, 33), p(ox - 8, oy + 10, 34)], 5.0), set_weather("set_wind_recover", "clear", wind_speed=6.0)], 4, scenario_id, "Wind eases and UAV exits gust corridor", "uav_mission", [uav], "info"),
        ]
        desc = "Crosswind payload swing and UAV attitude anomaly"
    elif scenario_id == "L5-4_v1":
        profile = {"initial": "clear", "transitions": []}
        events = [
            event("dusk_light_shift", tick(260), [visual("set_camera_underexposed", uav, "camera_underexposed"), screenshot("capture_dusk_underexposure")], 1, scenario_id, "Dusk light shift underexposes camera", "environment", [uav], "warning"),
            event("infrared_switch", fired("dusk_light_shift"), [visual("set_uav_infrared_mode", uav, "infrared_navigation"), move("move_uav_ir_continue", uav, [start, p(ox + 12, oy + 8, 32)], 5.0)], 2, scenario_id, "UAV switches to infrared navigation", "uav_mission", [uav], "info"),
            event("low_light_termination", fired("infrared_switch"), [screenshot("capture_infrared_validation")], 3, scenario_id, "Low-light segment terminates after validation", "uav_mission", [uav], "info"),
        ]
        desc = "Dusk light shift and camera underexposure"
    else:
        profile = {"initial": "clear", "transitions": []}
        events = [
            event("high_temperature_process", tick(260), [visual("set_uav_battery_derating", uav, "battery_derating")], 1, scenario_id, "High temperature causes battery derating", "environment", [uav], "warning"),
            event("range_shortened", fired("high_temperature_process"), [move("move_uav_shortened_range", uav, [start, p(ox + 8, oy + 5, 32)], 4.0)], 2, scenario_id, "UAV range shortens under derating", "uav_mission", [uav], "warning"),
            event("thermal_abort", fired("range_shortened"), [move("move_uav_thermal_abort", uav, [p(ox + 8, oy + 5, 32), p(ox - 5, oy + 9, 31)], 4.5), screenshot("capture_temperature_abort")], 3, scenario_id, "UAV terminates high-temperature mission", "uav_mission", [uav], "info"),
        ]
        desc = "High temperature battery derating and shortened range"
    return make_bundle(
        scenario_id,
        l_path("L5_environment", "interaction", scenario_id),
        "environment",
        desc,
        specs,
        scenes,
        {"weather_threshold_tick": 180 if scenario_id.startswith(("L5-1", "L5-2", "L5-3")) else 260},
        events,
        weather_profile=profile,
        validation_rules=[
            {
                "rule": "environment_trigger_kind",
                "description": "Rain/fog/wind use weather_state; light and temperature use tick simulation",
            }
        ],
    )


def build_l6(scenario_id: str, idx: int) -> ScenarioBundle:
    ox = 34.0 + idx * 8.0
    oy = 196.0 + idx * 3.2
    specs: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    sid = scenario_id.lower().replace("-", "_")
    tower = f"tower_{sid}"
    uav = f"uav_digital_{sid}"
    start = p(ox, oy, 32)
    add(specs, scenes, world_entity(tower, "facility.radio.base_tower.v1", "facility", p(ox + 8, oy + 4, 0), 0, "online"))
    add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 35, "patrol"))
    params = {"anomaly_tick": 240, "recovery_tick": 520}
    if scenario_id.startswith("L6-1"):
        current = p(ox + 14, oy + 8, 31)
        home = p(ox - 10, oy - 6, 30)
        events = [
            event("c2_loss", tick(240), [visual("set_tower_c2_loss", tower, "link_lost"), move("move_uav_until_c2_loss", uav, [start, current], 6.0)], 1, scenario_id, "C2 link loss occurs", "digital_layer", [uav, tower], "critical"),
            event("rth_engaged", fired("c2_loss"), [visual("set_uav_rth", uav, "return_to_home"), move("move_uav_rth_home", uav, [current, p(ox + 2, oy + 1, 34), home], 7.0)], 2, scenario_id, "UAV return-to-home safety action", "uav_mission", [uav, tower], "warning"),
            event("c2_recovered", fired("rth_engaged"), [visual("set_tower_link_restored", tower, "online"), visual("set_uav_rth_complete", uav, "home_hold")], 3, scenario_id, "C2 link recovered after RTH", "digital_layer", [uav, tower], "info"),
        ]
        desc = "C2 link loss with return-to-home"
    elif scenario_id.startswith("L6-2"):
        delayed = p(ox + 12, oy + 6, 31)
        events = [
            event("c2_degradation", tick(240), [visual("set_tower_c2_degraded", tower, "degraded")], 1, scenario_id, "Intermittent C2 degradation starts", "digital_layer", [tower, uav], "warning"),
            event("uav_delayed_response", fired("c2_degradation"), [move("move_uav_delayed_slowdown", uav, [start, delayed], 3.0), visual("set_uav_command_delay", uav, "command_delay")], 2, scenario_id, "UAV slows due to delayed command link", "uav_mission", [uav], "warning"),
            event("backup_link_lock", fired("uav_delayed_response"), [visual("set_tower_backup_lock", tower, "backup_link"), move("move_uav_backup_resume", uav, [delayed, p(ox + 2, oy + 18, 32)], 5.5)], 3, scenario_id, "Backup link reduces degradation", "digital_layer", [uav, tower], "info"),
            event("nominal_c2_restore", fired("backup_link_lock"), [visual("set_tower_nominal", tower, "online")], 4, scenario_id, "Nominal C2 restored", "digital_layer", [tower], "info"),
        ]
        desc = "Intermittent C2 degradation with slowdown and backup link"
    elif scenario_id.startswith("L6-3"):
        planned = p(ox + 12, oy + 8, 31)
        spoofed = p(ox + 27, oy + 18, 31)
        corrected = p(ox + 8, oy + 22, 32)
        events = [
            event("spoofing_start", tick(240), [visual("set_gnss_spoofing_detected", tower, "gnss_spoofing")], 1, scenario_id, "GNSS spoofing starts", "digital_layer", [uav, tower], "critical"),
            event("spoofed_route_offset", fired("spoofing_start"), [move("move_uav_spoofed_offset", uav, [start, planned, spoofed], 6.0)], 2, scenario_id, "UAV deviates from planned route by at least 10m", "uav_mission", [uav], "critical"),
            event("safety_route_lock", fired("spoofed_route_offset"), [visual("set_uav_nav_lock", uav, "navigation_lock"), move("move_uav_spoof_corrected", uav, [spoofed, corrected], 4.0), screenshot("capture_spoof_correction")], 3, scenario_id, "Safety mechanism locks route and corrects spoofing", "digital_layer", [uav, tower], "warning"),
            event("spoof_recovery", fired("safety_route_lock"), [visual("set_tower_gnss_normal", tower, "online")], 4, scenario_id, "GNSS spoofing cleared", "digital_layer", [tower, uav], "info"),
        ]
        params["spoof_offset_min_m"] = 10.0
        desc = "GNSS spoofing route hijack with actual offset"
    elif scenario_id.startswith("L6-4"):
        uav_b = f"uav_digital_{sid}_secondary"
        add(specs, scenes, world_entity(uav_b, "uav.airsim.flying_pawn.v1", "uav", p(ox - 12, oy + 7, 34), 50, "patrol"))
        events = [
            event("wideband_jamming", tick(240), [visual("set_tower_jamming", tower, "jammed"), visual("set_uav_a_jammed", uav, "jammed"), visual("set_uav_b_jammed", uav_b, "jammed")], 1, scenario_id, "Wideband jamming affects two UAVs", "digital_layer", [uav, uav_b, tower], "critical"),
            event("multi_uav_safe_hold", fired("wideband_jamming"), [move("move_uav_a_jam_hold", uav, [start, p(ox + 4, oy + 4, 35)], 2.0), move("move_uav_b_jam_hold", uav_b, [p(ox - 12, oy + 7, 34), p(ox - 7, oy + 12, 36)], 2.0)], 2, scenario_id, "Both UAVs enter safe hold under jamming", "uav_mission", [uav, uav_b], "warning"),
            event("alternate_channel", fired("multi_uav_safe_hold"), [visual("set_tower_alternate_channel", tower, "alternate_channel"), move("move_uav_a_channel_recover", uav, [p(ox + 4, oy + 4, 35), p(ox + 15, oy + 12, 33)], 5.5), move("move_uav_b_channel_recover", uav_b, [p(ox - 7, oy + 12, 36), p(ox - 18, oy + 18, 34)], 5.5)], 3, scenario_id, "Alternate channel recovers both UAVs", "digital_layer", [uav, uav_b, tower], "info"),
        ]
        desc = "Wideband jamming affecting multiple UAVs"
    else:
        gcs = f"gcs_anchor_{sid}"
        add(specs, scenes, world_entity(gcs, "semantic.asset_anchor", "ground_station", p(ox + 5, oy - 5, 0), 0, "gcs_online"))
        abnormal = p(ox + 20, oy - 12, 32)
        locked = p(ox + 3, oy + 7, 34)
        events = [
            event("gcs_intrusion", tick(240), [visual("set_gcs_intrusion", gcs, "intrusion_detected")], 1, scenario_id, "Ground control station intrusion detected", "digital_layer", [gcs, uav], "critical"),
            event("abnormal_uav_behavior", fired("gcs_intrusion"), [move("move_uav_intrusion_abnormal", uav, [start, abnormal], 9.0), visual("set_uav_abnormal_commanded", uav, "abnormal_command")], 2, scenario_id, "UAV follows abnormal command path", "uav_mission", [uav, gcs], "critical"),
            event("command_lockout", fired("abnormal_uav_behavior"), [visual("set_gcs_locked", gcs, "locked_out"), move("move_uav_command_lock", uav, [abnormal, locked], 4.0)], 3, scenario_id, "Command lockout and safe route intervention", "digital_layer", [uav, gcs], "warning"),
            event("secure_recovery", fired("command_lockout"), [visual("set_gcs_secure", gcs, "secure"), screenshot("capture_gcs_recovery")], 4, scenario_id, "Ground control station secured", "digital_layer", [uav, gcs], "info"),
        ]
        desc = "Ground control station intrusion and abnormal UAV behavior"
    return make_bundle(
        scenario_id,
        l_path("L6_digital_layer", "failure", scenario_id),
        "digital_layer",
        desc,
        specs,
        scenes,
        params,
        events,
        validation_rules=[
            {
                "rule": "digital_anomaly_chain",
                "description": "Digital layer anomaly produces actual UAV movement and recovery action",
            }
        ],
    )


def build_x(short_id: str, dirname: str, idx: int) -> ScenarioBundle:
    ox = 42.0 + idx * 16.0
    oy = 244.0 + idx * 7.0
    specs: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    weather_profile = {"initial": "clear", "transitions": []}
    params = {"cross_layer": True}
    if short_id == "X1":
        scenario_id = dirname
        uav, tower, ped = "uav_x1_forced_landing", "tower_x1_c2", "ped_x1_crowd_anchor"
        start = p(ox, oy, 32)
        add(specs, scenes, world_entity(tower, "facility.radio.base_tower.v1", "facility", p(ox + 8, oy + 2, 0), 0, "online"))
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 30, "rain_patrol"))
        _, ped_scene = add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_31", 42, GATHERING_MIN_OFFSET_FROM_CURB_M, p(ox + 16, oy + 9, 0), 0, "observing"))
        ped_start = scene_pos(ped_scene)
        crowd_origin = gathering_point(ped_start, [800, 500, 0])
        extra_crowd_ids = add_static_ped_cohort(specs, scenes, "x1", "ped_x1_crowd", crowd_origin, 9, activity_type="observing")
        crowd_ids = [ped, *extra_crowd_ids]
        low = q(ped_start[0], ped_start[1], 8)
        landing = q(ped_start[0], ped_start[1], 3)
        ped_evade = shifted_sidewalk_point(ped_start, 12, 10, GATHERING_MIN_OFFSET_FROM_CURB_M)
        weather_profile = {"initial": "clear", "transitions": [{"tick": 180, "profile": "rain", "overrides": {"rain": 0.62, "visibility_m": 1600.0}}]}
        events = [
            event("crowd_present", tick(0), [ped_activity(f"set_{ped_id}_crowd_observing", ped_id, "observing") for ped_id in crowd_ids], 0, scenario_id, "Crowd is visible near the future forced-landing area at episode start", "pedestrian", crowd_ids, "info"),
            event("rain_onset", tick(180), [set_weather("set_x1_rain_onset", "rain", rain=0.62, visibility_m=1600.0)], 1, scenario_id, "Rain onset is applied before C2 coupling", "weather", [uav, tower], "info"),
            event("rain_threshold", weather("rain", "gte", 0.5, 5), [set_weather("set_x1_rain", "rain", rain=0.65, visibility_m=1500.0)], 1, scenario_id, "Rain threshold triggers L5 degradation", "weather", [uav, tower], "warning"),
            event("c2_loss_tick", tick(260), [visual("set_x1_tower_stressed", tower, "rain_stressed")], 2, scenario_id, "C2 loss timing condition becomes true", "digital_layer", [tower], "warning"),
            event("c2_loss_after_rain", composite("AND", ["rain_threshold", "c2_loss_tick"]), [visual("set_x1_tower_c2_loss", tower, "link_lost"), move("move_x1_uav_degraded", uav, [start, p(ox + 9, oy + 4, 26)], 6.0)], 3, scenario_id, "Rain and C2 condition combine into C2 loss", "digital_layer", [uav, tower], "critical"),
            event("forced_landing_descent", fired("c2_loss_after_rain"), [move("move_x1_forced_landing", uav, [p(ox + 9, oy + 4, 26), low, landing], 6.0)], 4, scenario_id, "C2 loss causes forced landing near crowd", "uav_mission", [uav, ped], "critical"),
            event("crowd_reacts_to_landing", prox(uav, ped, 5.0, 2), [move("move_x1_ped_evade", ped, [ped_start, ped_evade], 2.0), screenshot("capture_x1_forced_landing")], 5, scenario_id, "Crowd reacts to forced landing", "pedestrian", [uav, ped], "warning"),
            event("x1_recovery", fired("crowd_reacts_to_landing"), [visual("set_x1_tower_restored", tower, "online"), visual("set_x1_uav_landed", uav, "landed_safe")], 6, scenario_id, "Cross-layer chain recovers", "digital_layer", [uav, tower], "info"),
        ]
        desc = "Rain to C2 loss to forced landing cross-layer chain"
    elif short_id == "X2":
        scenario_id = dirname
        uav, nfz = "uav_x2_spoofed", "nfz_x2_geofence"
        start, spoofed = p(ox, oy, 32), p(ox + 28, oy + 12, 32)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 45, "mission"))
        add(specs, scenes, box_entity(nfz, "trigger.no_fly.box.v1", "airspace_constraint", p(ox + 31, oy + 14, 28), [12, 10, 15]))
        events = [
            event("spoofing_start", tick(230), [visual("set_x2_spoof_flag", uav, "gnss_spoofed")], 1, scenario_id, "GNSS spoofing starts", "digital_layer", [uav], "critical"),
            event("trajectory_offset", fired("spoofing_start"), [move("move_x2_spoof_offset", uav, [start, p(ox + 13, oy + 5, 32), spoofed], 7.0)], 2, scenario_id, "Spoofing offsets UAV trajectory", "uav_mission", [uav], "critical"),
            event("geofence_violation", prox(uav, nfz, 10.0, 2), [visual("set_x2_geofence_alert", uav, "geofence_alert"), screenshot("capture_x2_geofence_violation")], 3, scenario_id, "Spoofed route approaches NFZ", "airspace", [uav, nfz], "critical"),
            event("safety_correction", fired("geofence_violation"), [move("move_x2_corrected_route", uav, [spoofed, p(ox + 8, oy + 24, 34)], 8.0)], 4, scenario_id, "Safety correction pulls UAV out of NFZ approach", "uav_mission", [uav, nfz], "warning"),
            event("route_recovered", fired("safety_correction"), [visual("set_x2_nav_restored", uav, "mission_recovered")], 5, scenario_id, "Navigation recovers", "digital_layer", [uav], "info"),
        ]
        desc = "GNSS spoofing to geofence violation chain"
    elif short_id == "X3":
        scenario_id = dirname
        ped, uav, ambulance, tape = "ped_x3_fall", "uav_x3_detect", "ambulance_x3", "police_tape_x3"
        _, ped_scene = add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_32", 50, 1.2, p(ox + 4, oy + 3, 0), 0, "walking"))
        ped_start = scene_pos(ped_scene)
        uav_start = q(ped_start[0] - 28.0, ped_start[1] - 20.0, 32)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 60, "patrol"))
        ambulance_arrival = road_center_point(ped_start)
        arrival_sample = LANES.nearest_to_xy(ambulance_arrival[0], ambulance_arrival[1])
        ambulance_s = max(0.0, arrival_sample.s_m - 42.0)
        ambulance_start_hint = _offset_from_lane(LANES.resolve_edge_s(arrival_sample.edge_id, ambulance_s), 0.0, GROUND_Z_M)
        _, ambulance_scene = add(
            specs,
            scenes,
            lane_entity(
                ambulance,
                "vehicle.emergency.ambulance.v1",
                "vehicle",
                arrival_sample.edge_id,
                ambulance_s,
                0.0,
                ambulance_start_hint,
                70,
                "response",
                visual_state={"mode": "response", "lights_on": True},
                prefer_edge_hint=True,
            ),
        )
        ambulance_start = scene_pos(ambulance_scene)
        uav_observe = q(ped_start[0], ped_start[1], 16)
        _, tape_scene = add(specs, scenes, world_entity(tape, "prop.incident.police_tape.v1", "prop", shifted_sidewalk_point(ped_start, 1.5, 1.0, 1.5), 0, "staged", activation_tick=420))
        tape_pos = scene_pos(tape_scene)
        events = [
            event("pedestrian_fall", tick(220), [play("play_x3_ped_fall", ped)], 1, scenario_id, "Pedestrian falls", "pedestrian", [ped], "critical"),
            event("uav_detection", fired("pedestrian_fall"), [move("move_x3_uav_detect", uav, [uav_start, uav_observe], 5.0), screenshot("capture_x3_detection")], 2, scenario_id, "UAV detects pedestrian fall", "uav_mission", [uav, ped], "warning"),
            event("ambulance_dispatch", fired("uav_detection"), [move("move_x3_ambulance_dispatch", ambulance, [ambulance_start, ambulance_arrival], 12.0)], 3, scenario_id, "Ambulance is dispatched", "vehicle", [ambulance, ped], "warning"),
            event("isolation_setup", fired("ambulance_dispatch"), [spawn_from_scene("spawn_x3_police_tape", tape_scene)], 4, scenario_id, "Responder sets incident isolation tape", "dynamic_constraint", [tape, ped], "warning"),
            event("emergency_documented", fired("isolation_setup"), [screenshot("capture_x3_response")], 5, scenario_id, "Emergency response documented", "uav_mission", [uav, ambulance, ped], "info"),
        ]
        desc = "Pedestrian fall to emergency response chain"
    elif short_id == "X4":
        scenario_id = dirname
        uav_a, uav_b = "uav_x4_a", "uav_x4_b"
        a0, b0 = p(ox - 18, oy, 29), p(ox + 20, oy + 10, 31)
        meet = p(ox + 1, oy + 5, 30)
        add(specs, scenes, world_entity(uav_a, "uav.inspect.quad.v1", "uav", a0, 70, "patrol"))
        add(specs, scenes, world_entity(uav_b, "uav.airsim.flying_pawn.v1", "uav", b0, 250, "patrol"))
        weather_profile = {"initial": "clear", "transitions": [{"tick": 180, "profile": "fog", "overrides": {"fog": 0.62, "visibility_m": 700.0}}]}
        events = [
            event("fog_onset", tick(180), [set_weather("set_x4_fog_onset", "fog", fog=0.62, visibility_m=700.0)], 1, scenario_id, "Fog onset is applied before conflict chain", "weather", [uav_a, uav_b], "info"),
            event("fog_threshold", weather("fog", "gte", 0.5, 5), [set_weather("set_x4_fog", "fog", fog=0.65, visibility_m=700.0)], 1, scenario_id, "Fog reaches operational threshold", "weather", [uav_a, uav_b], "warning"),
            event("visibility_drop", fired("fog_threshold"), [visual("set_x4_uav_a_visual_degraded", uav_a, "visual_degraded"), visual("set_x4_uav_b_visual_degraded", uav_b, "visual_degraded")], 2, scenario_id, "Visibility drops for both UAVs", "weather", [uav_a, uav_b], "warning"),
            event("fog_converging_routes", fired("visibility_drop"), [move("move_x4_uav_a_converge", uav_a, [a0, meet], 6.0), move("move_x4_uav_b_converge", uav_b, [b0, meet], 6.0)], 3, scenario_id, "Fog causes converging route conflict", "uav_mission", [uav_a, uav_b], "critical"),
            event("fog_uav_conflict", prox(uav_a, uav_b, 8.0, 2, metric="xy_plus_z", horizontal_distance_m=8.0, vertical_distance_m=35.0), [visual("set_x4_uav_a_hover", uav_a, "hover"), move("move_x4_uav_b_evasion", uav_b, [meet, p(ox + 14, oy + 18, 36)], 8.0), screenshot("capture_x4_conflict")], 4, scenario_id, "UAV proximity conflict under fog", "uav_mission", [uav_a, uav_b], "critical"),
            event("fog_conflict_recovered", fired("fog_uav_conflict"), [move("move_x4_uav_a_resume", uav_a, [meet, p(ox - 16, oy + 18, 30)], 5.5)], 5, scenario_id, "UAVs recover after fog conflict", "uav_mission", [uav_a, uav_b], "info"),
        ]
        desc = "Fog to multi-UAV conflict cross-layer chain"
    elif short_id == "X5":
        scenario_id = dirname
        tower, pad_a, pad_b = "tower_x5_comm", "pad_x5_primary", "pad_x5_backup"
        uav_a, uav_b = "uav_x5_priority", "uav_x5_second"
        add(specs, scenes, world_entity(tower, "facility.radio.base_tower.v1", "facility", p(ox + 5, oy + 1, 0), 0, "online"))
        pad_a_pos, pad_a_yaw, pad_a_anchor = landing_pad_pose(
            p(ox + 2, oy + 4, 0),
            context=f"{scenario_id} primary landing pad",
        )
        pad_b_pos, pad_b_yaw, pad_b_anchor = landing_pad_pose(
            p(ox + 26, oy + 8, 0),
            context=f"{scenario_id} backup landing pad",
        )
        pad_a_hover = q(pad_a_pos[0], pad_a_pos[1], 6.0)
        add(specs, scenes, pad_entity(pad_a, pad_a_pos, "pad_x5_a", "north", pad_a_yaw, pad_a_anchor))
        add(specs, scenes, pad_entity(pad_b, pad_b_pos, "pad_x5_b", "east", pad_b_yaw, pad_b_anchor))
        add(specs, scenes, world_entity(uav_a, "uav.inspect.quad.v1", "uav", p(ox - 18, oy + 12, 33), 60, "landing_request"))
        add(specs, scenes, world_entity(uav_b, "uav.airsim.flying_pawn.v1", "uav", p(ox + 18, oy - 12, 31), 280, "landing_request"))
        events = [
            event("station_failure", tick(220), [visual("set_x5_tower_failed", tower, "failed")], 1, scenario_id, "Communication station failure", "infrastructure", [tower], "critical"),
            event("backup_pad_unavailable", fired("station_failure"), [visual("set_x5_backup_pad_unavailable", pad_b, "unavailable")], 2, scenario_id, "Only one landing pad remains available", "infrastructure", [pad_a, pad_b], "warning"),
            event("dual_uav_pad_contention", fired("backup_pad_unavailable"), [move("move_x5_uav_a_to_pad", uav_a, [p(ox - 18, oy + 12, 33), pad_a_hover], 5.0), move("move_x5_uav_b_to_pad_hold", uav_b, [p(ox + 18, oy - 12, 31), p(ox + 8, oy + 10, 34)], 5.0)], 3, scenario_id, "Two UAVs contend for one available pad", "uav_mission", [uav_a, uav_b, pad_a], "critical"),
            event("pad_priority_arbitration", prox(uav_a, pad_a, 7.0, 2), [visual("set_x5_pad_reserved", pad_a, "reserved"), visual("set_x5_uav_b_hold", uav_b, "hold")], 4, scenario_id, "Priority arbitration reserves pad for first UAV", "infrastructure", [uav_a, uav_b, pad_a], "warning"),
            event("second_uav_reroute", fired("pad_priority_arbitration"), [move("move_x5_uav_b_reroute", uav_b, [p(ox + 8, oy + 10, 34), p(ox + 34, oy + 24, 34)], 8.0)], 5, scenario_id, "Second UAV reroutes after contention", "uav_mission", [uav_b], "info"),
            event("station_recovered", fired("second_uav_reroute"), [visual("set_x5_tower_restored", tower, "online")], 6, scenario_id, "Communication station recovers", "infrastructure", [tower], "info"),
        ]
        desc = "Communication failure to pad contention chain"
    else:
        scenario_id = dirname
        uav, nfz, anchor = "uav_x6_monitor", "nfz_x6_lockdown", "crowd_anchor_x6"
        uav_start = p(ox - 18, oy - 10, 34)
        crowd_extent = [900, 600, 0]
        safe_extent = [900, 500, 0]
        crowd_origin = gathering_point(p(ox + 4, oy + 3, 0), crowd_extent)
        safe_origin = gathering_point(p(ox + 28, oy + 18, 0), safe_extent)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 55, "monitor"))
        add(specs, scenes, world_entity(anchor, "semantic.spawn_zone", "crowd_anchor", crowd_origin, 0))
        cohort = add_evacuation_ped_cohort(specs, scenes, "x6", "ped_x6_evac", crowd_origin, safe_origin, 8)
        _, nfz_scene = add(specs, scenes, box_entity(nfz, "trigger.no_fly.box.v1", "airspace_constraint", p(ox + 6, oy + 4, 28), [18, 13, 16], activation_tick=310))
        events = [
            event("crowd_present", tick(0), [], 0, scenario_id, "Explicit pedestrian cohort is visible before evacuation begins", "pedestrian", [anchor], "info"),
            event("crowd_evacuation", tick(220), evacuation_move_actions(cohort, "move_x6_crowd_evacuation"), 1, scenario_id, "Crowd evacuation begins with visible pedestrian movement", "pedestrian", [anchor], "warning"),
            event("crowd_safe_zone_move", fired("crowd_evacuation"), [screenshot("capture_x6_crowd_safe_zone")], 2, scenario_id, "Crowd reaches safe zone without mid-script disappearance", "pedestrian", [anchor], "warning"),
            event("airspace_lockdown", fired("crowd_safe_zone_move"), [spawn_from_scene("spawn_x6_nfz", nfz_scene, visual_state={"mode": "active"})], 3, scenario_id, "Airspace lockdown declares temporary NFZ", "dynamic_constraint", [nfz], "critical"),
            event("uav_reroute_lockdown", fired("airspace_lockdown"), [move("move_x6_uav_avoid_nfz", uav, [uav_start, p(ox - 5, oy + 14, 36), p(ox + 22, oy + 28, 36)], 7.0), screenshot("capture_x6_lockdown")], 4, scenario_id, "UAV reroutes around lockdown NFZ", "uav_mission", [uav, nfz], "warning"),
            event("lockdown_cleared", fired("uav_reroute_lockdown"), [visual("set_x6_nfz_standdown", nfz, "standdown")], 5, scenario_id, "Crowd evacuation and airspace lockdown clear", "dynamic_constraint", [uav, nfz], "info"),
        ]
        desc = "Crowd evacuation to airspace lockdown chain"
    return make_bundle(
        scenario_id,
        x_path(dirname),
        "cross_layer",
        desc,
        specs,
        scenes,
        params,
        events,
        weather_profile=weather_profile,
        validation_rules=[
            {
                "rule": "cross_layer_event_chain_min",
                "min_count": 5,
                "description": "Cross-layer scenario has at least five causal steps",
            }
        ],
    )


def trigger_from_dict(data: dict[str, Any]) -> TriggerSpec:
    allowed = set(TriggerSpec.__dataclass_fields__)
    cleaned = {key: value for key, value in data.items() if key in allowed}
    if "distance_m" not in cleaned and "original_distance_m" in data:
        cleaned["distance_m"] = data["original_distance_m"]
    if "horizontal_distance_m" not in cleaned and "original_horizontal_distance_m" in data:
        cleaned["horizontal_distance_m"] = data["original_horizontal_distance_m"]
    if "vertical_distance_m" not in cleaned and "original_vertical_distance_m" in data:
        cleaned["vertical_distance_m"] = data["original_vertical_distance_m"]
    return TriggerSpec(**cleaned)


def action_from_dict(data: dict[str, Any]) -> ActionSpec:
    return ActionSpec(data["type"], data.get("params", {}))


def event_from_dict(data: dict[str, Any]) -> EventStepSpec:
    return EventStepSpec(
        event_id=data["event_id"],
        trigger=trigger_from_dict(data["trigger"]),
        actions=[action_from_dict(a) for a in data.get("actions", [])],
        on_fire_emit=data.get("on_fire_emit", []),
        priority=data.get("priority", 10),
        max_fire_count=data.get("max_fire_count", 1),
        cooldown_ticks=data.get("cooldown_ticks", 0),
        require_conditions=data.get("require_conditions", []),
        log_topic=data.get("log_topic", ""),
        log_category=data.get("log_category", ""),
        log_title=data.get("log_title", ""),
        log_severity=data.get("log_severity", "info"),
        log_overlay=data.get("log_overlay", ""),
        log_target_ids=data.get("log_target_ids", []),
        intent=data.get("intent", ""),
        intent_stage=data.get("intent_stage", ""),
        causal_chain_id=data.get("causal_chain_id", ""),
        causal_predecessor_intent=data.get("causal_predecessor_intent", ""),
        target_roles=data.get("target_roles", []),
    )


def spec_from_bundle(bundle: ScenarioBundle) -> ScenarioSpec:
    entities = [
        EntitySpec(
            entity_id=e["entity_id"],
            asset_id=e["asset_id"],
            initial_pos_enu=e["initial_pos_enu"],
            initial_rotation_deg=e.get("initial_rotation_deg", [0.0, 0.0, 0.0]),
            movement_waypoints=[WaypointSpec(w) for w in e.get("movement_waypoints", [])],
            visual_state=e.get("visual_state"),
        )
        for e in bundle.spec_entities
    ]
    return ScenarioSpec(
        scenario_id=bundle.scenario_id,
        category=bundle.category,
        description=bundle.description,
        duration_ticks=bundle.duration_ticks,
        entities=entities,
        event_chain=[event_from_dict(e) for e in bundle.events],
        parameters=bundle.parameters,
    )


def scene_setup_from_bundle(bundle: ScenarioBundle) -> dict[str, Any]:
    return {
        "$schema": "scene_setup_v1",
        "scenario_id": bundle.scenario_id,
        "description": bundle.description,
        "map_ref": MAP_REF,
        "local_bounds": local_bounds_for_bundle(bundle),
        "entities": bundle.scene_entities,
        "spawn_sequencing": [
            {"entity_id": e["entity_id"], "tick": e.get("activation_tick", 0)}
            for e in bundle.scene_entities
        ],
        "weather_profile": bundle.weather_profile,
        "cameras": bundle.cameras,
        "validation_rules": bundle.validation_rules,
    }


def py_repr(value: Any, indent: int = 0) -> str:
    return pformat(value, width=118)


def write_spec_py(bundle: ScenarioBundle) -> str:
    scene_setup = scene_setup_from_bundle(bundle)
    spec_data = {
        "scenario_id": bundle.scenario_id,
        "category": bundle.category,
        "description": bundle.description,
        "duration_ticks": bundle.duration_ticks,
        "parameters": bundle.parameters,
        "entities": bundle.spec_entities,
        "event_chain": bundle.events,
    }
    return f'''"""Concrete ScenarioSpec for {bundle.scenario_id}.

Generated from Dataset/tools/regenerate_boundary_scenarios.py.
This file is intentionally self-contained: running it recompiles
event_script.json from the ScenarioSpec below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve()
while _TOOLS.name != "Dataset" and _TOOLS.parent != _TOOLS:
    _TOOLS = _TOOLS.parent
_TOOLS = _TOOLS / "tools"
sys.path.insert(0, str(_TOOLS))

from spec_compiler import (
    ActionSpec,
    EntitySpec,
    EventStepSpec,
    ScenarioSpec,
    SpecCompiler,
    TriggerSpec,
    WaypointSpec,
)


SCENE_SETUP = {py_repr(scene_setup)}


SPEC_DATA = {py_repr(spec_data)}


def _trigger(data):
    return TriggerSpec(**data)


def _action(data):
    return ActionSpec(data["type"], data.get("params", {{}}))


def _event(data):
    return EventStepSpec(
        event_id=data["event_id"],
        trigger=_trigger(data["trigger"]),
        actions=[_action(a) for a in data.get("actions", [])],
        on_fire_emit=data.get("on_fire_emit", []),
        priority=data.get("priority", 10),
        max_fire_count=data.get("max_fire_count", 1),
        cooldown_ticks=data.get("cooldown_ticks", 0),
        require_conditions=data.get("require_conditions", []),
        log_topic=data.get("log_topic", ""),
        log_category=data.get("log_category", ""),
        log_title=data.get("log_title", ""),
        log_severity=data.get("log_severity", "info"),
        log_overlay=data.get("log_overlay", ""),
        log_target_ids=data.get("log_target_ids", []),
        intent=data.get("intent", ""),
        intent_stage=data.get("intent_stage", ""),
        causal_chain_id=data.get("causal_chain_id", ""),
        causal_predecessor_intent=data.get("causal_predecessor_intent", ""),
        target_roles=data.get("target_roles", []),
    )


def build_spec():
    return ScenarioSpec(
        scenario_id=SPEC_DATA["scenario_id"],
        category=SPEC_DATA["category"],
        description=SPEC_DATA["description"],
        duration_ticks=SPEC_DATA["duration_ticks"],
        parameters=SPEC_DATA["parameters"],
        entities=[
            EntitySpec(
                entity_id=e["entity_id"],
                asset_id=e["asset_id"],
                initial_pos_enu=e["initial_pos_enu"],
                initial_rotation_deg=e.get("initial_rotation_deg", [0.0, 0.0, 0.0]),
                movement_waypoints=[WaypointSpec(w) for w in e.get("movement_waypoints", [])],
                visual_state=e.get("visual_state"),
            )
            for e in SPEC_DATA["entities"]
        ],
        event_chain=[_event(e) for e in SPEC_DATA["event_chain"]],
    )


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    spec = build_spec()
    compiled = SpecCompiler().compile(spec)
    with open(here / "event_script.json", "w", encoding="utf-8") as f:
        json.dump(compiled, f, indent=2, ensure_ascii=False)
        f.write("\\n")
    with open(here / "scene_setup.json", "w", encoding="utf-8") as f:
        json.dump(SCENE_SETUP, f, indent=2, ensure_ascii=False)
        f.write("\\n")
'''


def _bundle_capture_center(bundle: ScenarioBundle) -> list[float]:
    boundary = dict(bundle.parameters.get("capture_boundary") or {})
    center = boundary.get("center_enu_m")
    if isinstance(center, list) and len(center) >= 2:
        return [float(center[0]), float(center[1]), float(center[2] if len(center) > 2 else GROUND_Z_M)]
    return _capture_boundary_focus_center(bundle.scene_entities, bundle.events)


def build_spatially_assigned_bundle(
    scenario_id: str,
    builder: Callable[[], ScenarioBundle],
    assignment: SpatialAssignment,
    provisional_center: list[float] | None = None,
) -> ScenarioBundle:
    if provisional_center is None:
        reset_world_event_origin()
        provisional = builder()
        provisional_center = _bundle_capture_center(provisional)
    target = assignment.target_center_enu_m
    target_origin_x = WORLD_OFFSET_X_M + float(target[0]) - float(provisional_center[0])
    target_origin_y = WORLD_OFFSET_Y_M + float(target[1]) - float(provisional_center[1])
    set_world_event_origin(target_origin_x, target_origin_y)
    try:
        bundle = builder()
    finally:
        reset_world_event_origin()
    actual_center = _bundle_capture_center(bundle)
    target_error_m = dist_xy(actual_center, target)
    assignment_payload = assignment.to_dict()
    assignment_payload.update(
        {
            "provisional_capture_center_enu_m": q(
                provisional_center[0],
                provisional_center[1],
                provisional_center[2] if len(provisional_center) > 2 else GROUND_Z_M,
            ),
            "applied_world_origin_enu_m": q(target_origin_x, target_origin_y, GROUND_Z_M),
            "actual_capture_center_enu_m": q(
                actual_center[0],
                actual_center[1],
                actual_center[2] if len(actual_center) > 2 else GROUND_Z_M,
            ),
            "target_error_m": round(target_error_m, 6),
        }
    )
    bundle.parameters["spatial_grid_assignment"] = assignment_payload
    bundle.validation_rules = list(bundle.validation_rules or [])
    bundle.validation_rules.append(
        {
            "rule": "spatial_grid_assignment",
            "policy": "city_grid_main_road_event_coverage_v1",
            "event_space_class": assignment.event_space_class,
            "target_center_enu_m": assignment_payload["target_center_enu_m"],
            "actual_capture_center_enu_m": assignment_payload["actual_capture_center_enu_m"],
            "target_error_m": assignment_payload["target_error_m"],
            "description": "Episode location is assigned from the city main-road spatial grid; traffic incidents bind to SUMO incident anchors while nontraffic EPI avoid incident grids",
        }
    )
    if assignment.event_space_class == "traffic_incident_grid" and target_error_m > 80.0:
        raise RuntimeError(
            f"{scenario_id}: traffic incident capture center is {target_error_m:.1f}m from SUMO anchor"
        )
    if assignment.event_space_class == "nontraffic_main_road_grid" and target_error_m > 140.0:
        raise RuntimeError(
            f"{scenario_id}: nontraffic capture center is {target_error_m:.1f}m from assigned main-road grid"
        )
    return bundle


def collect_bundles(selected_scenarios: set[str] | None = None) -> list[ScenarioBundle]:
    selected = set(selected_scenarios or set())

    def include(scenario_id: str, *aliases: str) -> bool:
        return not selected or scenario_id in selected or any(alias in selected for alias in aliases)

    ids_l1 = ["L1-1_v1", "L1-1_v2", "L1-2_v1", "L1-3_v1", "L1-3_v2", "L1-4_v1", "L1-4_v2"]
    ids_l2 = ["L2-1_v1", "L2-1_v2", "L2-2_v1", "L2-2_v2", "L2-3_v1", "L2-3_v2", "L2-4_v1", "L2-4_v2", "L2-5_v1"]
    ids_l3 = ["L3-1_v1", "L3-2_v1", "L3-2_v2", "L3-3_v1", "L3-3_v2"]
    ids_l4 = [
        "L4-1_v1",
        "L4-1_v2",
        "L4-2_v1",
        "L4-2_v2",
        "L4-3_v1",
        "L4-3_v2",
        "L4-3_v3",
        "L4-4_v1",
        "L4-4_v2",
        "L4-5_v1",
        "L4-5_v2",
        "L4-5_v3",
        "L4-6_v1",
        "L4-6_v2",
        "L4-7_v1",
        "L4-7_v2",
        "L4-8_v1",
        "L4-8_v2",
        "L4-9_v1",
        "L4-9_v2",
        "L4-10_v1",
        "L4-10_v2",
        "L4-11_v1",
        "L4-11_v2",
    ]
    ids_l5 = ["L5-1_v1", "L5-1_v2", "L5-1_v3", "L5-2_v1", "L5-2_v2", "L5-3_v1", "L5-3_v2", "L5-4_v1", "L5-5_v1"]
    ids_l6 = ["L6-1_v1", "L6-1_v2", "L6-2_v1", "L6-2_v2", "L6-3_v1", "L6-3_v2", "L6-4_v1", "L6-4_v2", "L6-5_v1", "L6-5_v2"]
    x_ids = [
        ("X1", "X1_rain_to_c2loss_to_forced_landing"),
        ("X2", "X2_gnss_spoof_to_geofence_violation"),
        ("X3", "X3_pedestrian_fall_to_emergency_response"),
        ("X4", "X4_fog_to_uav_conflict"),
        ("X5", "X5_comm_failure_to_pad_contention"),
        ("X6", "X6_crowd_evacuation_to_airspace_lockdown"),
    ]
    all_scenario_ids = [*ids_l1, *ids_l2, *ids_l3, *ids_l4, *ids_l5, *ids_l6, *(dirname for _short_id, dirname in x_ids)]
    spatial_grid = SpatialEventGridPlanner(planner=SUMO_GROUND_FLOW)
    spatial_assignments = spatial_grid.assign_scenarios(all_scenario_ids)
    nonincident_cells = spatial_grid.nontraffic_cells_for_assignments(spatial_assignments)
    used_nontraffic_cell_ids: set[str] = set()
    if selected:
        for script_path in SCENARIOS_ROOT.rglob("event_script.json"):
            try:
                existing_script = json.loads(script_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            existing_id = str(existing_script.get("scenario_id") or script_path.parent.name)
            is_selected = (
                existing_id in selected
                or script_path.parent.name in selected
                or any(existing_id.startswith(f"{token}_") for token in selected)
            )
            if is_selected:
                continue
            assignment = dict((existing_script.get("parameters") or {}).get("spatial_grid_assignment") or {})
            if str(assignment.get("event_space_class") or "") != "nontraffic_main_road_grid":
                continue
            grid_cell = dict(assignment.get("grid_cell") or {})
            cell_id = str(grid_cell.get("cell_id") or "")
            if cell_id:
                used_nontraffic_cell_ids.add(cell_id)
    bundles: list[ScenarioBundle] = []

    def nontraffic_assignment_for_cell(scenario_id: str, cell: Any) -> SpatialAssignment:
        return SpatialAssignment(
            scenario_id=scenario_id,
            event_space_class="nontraffic_main_road_grid",
            target_center_enu_m=list(cell.representative_enu_m),
            grid_cell=cell,
            source="main_road_grid.nonincident_cell",
            exclusion_radius_m=spatial_grid.accident_exclusion_radius_m,
        )

    def append_assigned(scenario_id: str, builder: Callable[[], ScenarioBundle]) -> None:
        primary = spatial_assignments[scenario_id]
        candidates = [primary]
        if primary.event_space_class == "nontraffic_main_road_grid":
            start_index = next(
                (
                    index
                    for index, cell in enumerate(nonincident_cells)
                    if cell.cell_id == primary.grid_cell.cell_id
                ),
                0,
            )
            for offset in range(1, len(nonincident_cells)):
                cell = nonincident_cells[(start_index + offset) % len(nonincident_cells)]
                if cell.cell_id == primary.grid_cell.cell_id:
                    continue
                candidates.append(nontraffic_assignment_for_cell(scenario_id, cell))
            candidates = [
                candidate
                for candidate in candidates
                if candidate.grid_cell.cell_id not in used_nontraffic_cell_ids
            ] + [
                candidate
                for candidate in candidates
                if candidate.grid_cell.cell_id in used_nontraffic_cell_ids
            ]
        errors: list[str] = []
        reset_world_event_origin()
        provisional = builder()
        provisional_center = _bundle_capture_center(provisional)
        for assignment in candidates:
            try:
                bundle = build_spatially_assigned_bundle(
                    scenario_id,
                    builder,
                    assignment,
                    provisional_center=provisional_center,
                )
                bundles.append(bundle)
                if assignment.event_space_class == "nontraffic_main_road_grid":
                    used_nontraffic_cell_ids.add(assignment.grid_cell.cell_id)
                return
            except RuntimeError as exc:
                errors.append(str(exc))
                if primary.event_space_class == "traffic_incident_grid":
                    break
        raise RuntimeError(
            f"{scenario_id}: unable to place scenario in assigned spatial grid candidates; "
            f"errors={errors[:4]}"
        )

    for i, scenario_id in enumerate(ids_l1):
        if include(scenario_id):
            append_assigned(scenario_id, lambda scenario_id=scenario_id, i=i: build_l1(scenario_id, i))
    for i, scenario_id in enumerate(ids_l2):
        if include(scenario_id):
            append_assigned(scenario_id, lambda scenario_id=scenario_id, i=i: build_l2(scenario_id, i))
    for i, scenario_id in enumerate(ids_l3):
        if include(scenario_id):
            append_assigned(scenario_id, lambda scenario_id=scenario_id, i=i: build_l3(scenario_id, i))
    for i, scenario_id in enumerate(ids_l4):
        if include(scenario_id):
            append_assigned(scenario_id, lambda scenario_id=scenario_id, i=i: build_l4(scenario_id, i))
    for i, scenario_id in enumerate(ids_l5):
        if include(scenario_id):
            append_assigned(scenario_id, lambda scenario_id=scenario_id, i=i: build_l5(scenario_id, i))
    for i, scenario_id in enumerate(ids_l6):
        if include(scenario_id):
            append_assigned(scenario_id, lambda scenario_id=scenario_id, i=i: build_l6(scenario_id, i))
    for i, (short_id, dirname) in enumerate(x_ids):
        if include(dirname, short_id):
            append_assigned(dirname, lambda short_id=short_id, dirname=dirname, i=i: build_x(short_id, dirname, i))
    return bundles


def compile_bundle(bundle: ScenarioBundle) -> dict[str, Any]:
    return SpecCompiler().compile(spec_from_bundle(bundle))


def validate_bundle(bundle: ScenarioBundle, compiled: dict[str, Any], catalog_ids: set[str]) -> list[str]:
    errors: list[str] = []
    scene_ids = {e["entity_id"] for e in bundle.scene_entities}
    scene_assets = {e["entity_id"]: e["logical_asset_id"] for e in bundle.scene_entities}
    for e in bundle.scene_entities:
        asset = e["logical_asset_id"]
        if asset not in ALLOWED_ASSETS:
            errors.append(f"{bundle.scenario_id}: asset outside boundary list: {asset}")
        if asset not in catalog_ids:
            errors.append(f"{bundle.scenario_id}: asset missing from catalog: {asset}")
    for e in compiled["events"]:
        for a in e.get("actions", []):
            action_id = a.get("action_id")
            if not action_id:
                errors.append(f"{bundle.scenario_id}: action without action_id in {e['event_id']}")
            if isinstance(action_id, str) and "$param." in action_id:
                errors.append(f"{bundle.scenario_id}: action_id contains parameter ref: {action_id}")
            ent_id = a.get("entity_id") or a.get("ped_id")
            if ent_id and ent_id not in scene_ids:
                errors.append(f"{bundle.scenario_id}: action references undeclared entity {ent_id}")
            if a["type"] in {"play_animation", "spawn_crowd"}:
                errors.append(f"{bundle.scenario_id}: Dataset generation must not emit {a['type']}")
            if a["type"] == "set_pedestrian_activity":
                activity = a.get("activity_type")
                try:
                    normalize_activity_type(str(activity or ""))
                except ValueError as exc:
                    errors.append(f"{bundle.scenario_id}: {exc}")
                if ent_id and not scene_assets.get(ent_id, "").startswith("pedestrian."):
                    errors.append(f"{bundle.scenario_id}: set_pedestrian_activity target is not pedestrian: {ent_id}")
            if a["type"] == "move_entity":
                asset = scene_assets.get(ent_id, "")
                if asset in {"prop.traffic_control.signal_light.v1", "facility.radio.base_tower.v1"}:
                    errors.append(f"{bundle.scenario_id}: static infrastructure moved: {ent_id}")
                for wp in a.get("waypoints_enu_m", []):
                    z = wp[2]
                    if asset.startswith("uav.") and z < 0.5 and not bundle.scenario_id.startswith("L4-4_"):
                        errors.append(f"{bundle.scenario_id}: UAV waypoint has ground z: {ent_id} {wp}")
                    if asset.startswith("vehicle.") and abs(z) > 0.6:
                        errors.append(f"{bundle.scenario_id}: vehicle waypoint not ground level: {ent_id} {wp}")
                    if asset.startswith("pedestrian.") and abs(z) > 0.6:
                        errors.append(f"{bundle.scenario_id}: pedestrian waypoint not ground level: {ent_id} {wp}")
            if a["type"] == "spawn_entity":
                ent = a.get("entity_id")
                if ent and ent not in scene_ids:
                    errors.append(f"{bundle.scenario_id}: spawn_entity references undeclared entity {ent}")
                asset = a.get("asset_id")
                if asset not in ALLOWED_ASSETS:
                    errors.append(f"{bundle.scenario_id}: spawn asset outside boundary list: {asset}")
        if e.get("trigger_ref") not in {t["trigger_id"] for t in compiled["triggers"]}:
            errors.append(f"{bundle.scenario_id}: event trigger_ref missing: {e.get('trigger_ref')}")
    trigger_ids = {t["trigger_id"] for t in compiled["triggers"]}
    for t in compiled["triggers"]:
        if t["type"] == "entity_proximity":
            for key in ("entity_a", "entity_b"):
                if t.get(key) not in scene_ids:
                    errors.append(f"{bundle.scenario_id}: proximity trigger references undeclared {t.get(key)}")
            d = float(t["distance_m"])
            if d <= 0 or d > 50:
                errors.append(f"{bundle.scenario_id}: proximity distance unreasonable: {d}")
        if t["type"] == "composite":
            for child in t.get("children", []):
                if child not in trigger_ids:
                    errors.append(f"{bundle.scenario_id}: composite child missing: {child}")
    has_weather = any(t["type"] == "weather_state" for t in compiled["triggers"])
    sid = bundle.scenario_id
    is_x_weather = sid.startswith("X1_") or sid.startswith("X4_")
    if sid.startswith(("L1", "L2", "L3", "L4", "L6")) and has_weather:
        errors.append(f"{sid}: non-weather layer contains weather_state trigger")
    if sid.startswith(("L5-1", "L5-2", "L5-3")) and not has_weather:
        errors.append(f"{sid}: weather scenario missing weather_state trigger")
    if sid.startswith(("L5-4", "L5-5")) and has_weather:
        errors.append(f"{sid}: light/temp scenario must not use weather_state trigger")
    if sid.startswith("X") and has_weather and not is_x_weather:
        errors.append(f"{sid}: cross-layer scenario has unexpected weather_state trigger")
    if sid.startswith(("X1_", "X4_")) and not has_weather:
        errors.append(f"{sid}: weather cross-layer scenario missing weather_state trigger")
    if len(bundle.validation_rules or []) < 2:
        errors.append(f"{sid}: scene_setup has fewer than two validation rules")
    if not bundle.cameras:
        errors.append(f"{sid}: scene_setup missing camera")
    if sid.startswith("L2-5"):
        if "traffic_signal_l2_5_v1" not in scene_ids or "police_unit_l2_5_v1" not in scene_ids:
            errors.append("L2-5_v1: missing traffic_signal primary or police secondary")
        vehicle_count = sum(1 for e in bundle.scene_entities if e["category"] == "vehicle")
        if vehicle_count < 3:
            errors.append("L2-5_v1: expected at least two cars plus police")
    if sid.startswith("L3-1"):
        counts = {"prop.roadwork.barrier.v1": 0, "prop.roadwork.construction_fence.v1": 0, "prop.roadwork.traffic_cone.v1": 0}
        for e in bundle.scene_entities:
            if e["logical_asset_id"] in counts:
                counts[e["logical_asset_id"]] += 1
                if e["placement_mode"] != "lane_anchor":
                    errors.append(f"{sid}: roadwork prop does not use lane_anchor: {e['entity_id']}")
        if counts["prop.roadwork.barrier.v1"] < 3 or counts["prop.roadwork.construction_fence.v1"] < 2 or counts["prop.roadwork.traffic_cone.v1"] < 5:
            errors.append(f"{sid}: roadwork prop counts invalid: {counts}")
    if sid.startswith("L4-10"):
        amb = [e for e in bundle.scene_entities if e["logical_asset_id"] == "vehicle.emergency.ambulance.v1"]
        if not amb or not amb[0].get("initial_state", {}).get("lights_on"):
            errors.append(f"{sid}: ambulance lights_on true missing")
    if sid.startswith("X") and len(compiled["events"]) < 5:
        errors.append(f"{sid}: cross-layer event chain shorter than 5")
    return errors


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_retry(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_text_retry(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(6):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2 * (attempt + 1))
    if last_error is not None:
        raise last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate semantic boundary scenario specs")
    parser.add_argument(
        "--scenario",
        action="append",
        nargs="+",
        default=[],
        help="Optional scenario ids to regenerate. May be repeated; X scenarios also accept their short id.",
    )
    args = parser.parse_args()
    selected_scenarios = {item for group in args.scenario for item in group}
    with open(ASSET_CATALOG, "r", encoding="utf-8") as f:
        catalog_ids = {a["logical_asset_id"] for a in json.load(f)["assets"]}
    bundles = collect_bundles(selected_scenarios or None)
    if selected_scenarios and not bundles:
        raise SystemExit(f"No matching scenarios for --scenario {sorted(selected_scenarios)}")
    errors: list[str] = []
    for bundle in bundles:
        compiled = compile_bundle(bundle)
        errors.extend(validate_bundle(bundle, compiled, catalog_ids))
        bundle.directory.mkdir(parents=True, exist_ok=True)
        write_json(bundle.directory / "event_script.json", compiled)
        write_json(bundle.directory / "scene_setup.json", scene_setup_from_bundle(bundle))
        write_text_retry(bundle.directory / "spec.py", write_spec_py(bundle))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Generated {len(bundles)} scenario specs, scene setups, and event scripts.")


if __name__ == "__main__":
    main()
