"""Regenerate semantic P0 scenario specs and compiled event scripts.

This generator replaces the old placeholder specs with concrete
SpecCompiler ScenarioSpec definitions.  It also emits scene_setup.json files
because the boundary document requires semantic placement, asset resolution,
and validation metadata that event_script_v1 does not carry.
"""

from __future__ import annotations

import argparse
import copy
import json
import csv
import heapq
import math
import os
import re
import sys
import time
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Sequence

from jsonschema import Draft202012Validator
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
from shapely.strtree import STRtree

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
from sumo_ground_flow.planner import (  # noqa: E402
    SumoGroundFlowPlanner,
    SumoRouteError,
)
from sumo_ground_flow.spatial_event_grid import (  # noqa: E402
    SpatialAssignment,
    SpatialEventGridPlanner,
    SpatialGridCell,
)
from runtime_state_contract import (  # noqa: E402
    RUNTIME_STATE_FIELDS,
    invalid_runtime_state_value_paths,
    unconsumed_runtime_state_paths,
)
from runtime_schedule_api_impact import require_runtime_schedule_api_impact  # noqa: E402
from roi_contract import (  # noqa: E402
    RoiContract,
    load_roi_contracts,
    TARGET_RECAPTURE_SCENARIO_IDS,
)
from inspect_observation_contract import (  # noqa: E402
    boundary_observation_samples,
    camera_footprint_half_extents_m,
    route_observes_boundary,
    route_observes_samples,
)

from Dataset.semantic_truth.facility_scope import (  # noqa: E402
    canonical_entity_kind_for_subtype,
    semantic_scope_for_source,
)
from Dataset.tools.runtime_state_source_guard import forbidden_runtime_state_paths  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "Dataset"
SCENARIOS_ROOT = DATASET_ROOT / "scenarios"
RUNTIME_STATE_SCHEDULE_PROFILE_DIR = (
    DATASET_ROOT / "semantic_rules" / "profiles" / "epi_runtime_state_schedules"
)
RUNTIME_STATE_SCHEDULE_SCHEMA_PATH = (
    DATASET_ROOT / "semantic_rules" / "schema" / "epi_runtime_state_schedule.schema.json"
)
OBJECTIVE_SEMANTIC_CONTRACT_PROFILE_PATH = (
    DATASET_ROOT / "semantic_rules" / "profiles" / "epi_objective_semantic_contract.json"
)
DOMAIN_STATE_SUPPLEMENT_PROFILE_PATH = (
    DATASET_ROOT / "semantic_rules" / "profiles" / "domain_state_supplement_profile.json"
)
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
LANE_CONNECTIONS_CSV = LANE_SAMPLES_CSV.with_name("lane_connections.csv")
SUMO_NET_XML = (
    ROOT
    / "Plugins"
    / "SumoImporter"
    / "Maps"
    / "donghu_road_topo"
    / "source"
    / "road_derived"
    / "road_derived.net.xml"
)
ROAD_GEOJSON = ROOT / "Content" / "Maps" / "donghu_road_topo" / "road" / "road.geojson"
BUILDING_GEOJSON = ROOT / "Content" / "Maps" / "donghu_road_topo" / "building" / "building.geojson"
UAV_SENSOR_PROFILE_PATH = (
    ROOT / "Config" / "LowAltitude" / "uav_sensor_profile.json"
)
WORLD_OFFSET_X_M = 7000.0
WORLD_OFFSET_Y_M = 6200.0
CURRENT_WORLD_OFFSET_X_M = WORLD_OFFSET_X_M
CURRENT_WORLD_OFFSET_Y_M = WORLD_OFFSET_Y_M
SPATIAL_PROVISIONAL_BUILD = False
ACTIVE_ROI_CONTRACT: RoiContract | None = None
# When regenerating only some scenarios, the location decision (grid cell, world origin,
# inspect loop) is read back from the checked-in artifacts instead of being re-derived
# from the global cursor.  Set AERO_REGEN_REUSE_FROZEN_LAYOUT=0 to force a full re-layout,
# which is what a deliberate geometry change requires.
REUSE_FROZEN_LAYOUT = os.environ.get("AERO_REGEN_REUSE_FROZEN_LAYOUT", "1") != "0"
# True while collecting a partial (--scenario) batch.  Frozen layout reuse applies only
# to partial runs; a full-batch run must reproduce the original global assignment.
PARTIAL_REGENERATION = False
ROI_ASSIGNMENT_MAX_OUTSIDE_M = 5.0
LANE_HALF_WIDTH_M = 1.9
PRIORITY_VEHICLE_MIN_CENTER_SEPARATION_M = 2.5
PRIORITY_YIELD_SPEED_MPS = 4.5
PRIORITY_AMBULANCE_SPEED_MPS = 10.0
PRIORITY_VEHICLE_EVIDENCE_DWELL_TICKS = 20
SUMO_DIRECTED_PHYSICAL_LANE_ROUTE_AUTHORITY = (
    "sumo_directed_physical_lane_geometry"
)
SIDEWALK_MIN_OFFSET_FROM_CURB_M = 1.2
GATHERING_MIN_OFFSET_FROM_CURB_M = 4.5
ROADWORK_MIN_OFFSET_FROM_EDGE_M = 1.5
LANDING_PAD_MIN_OFFSET_FROM_CURB_M = GATHERING_MIN_OFFSET_FROM_CURB_M
SERVICE_FACILITY_MIN_OFFSET_FROM_CURB_M = GATHERING_MIN_OFFSET_FROM_CURB_M
SERVICE_FACILITY_ASSETS = {
    "facility.charger.cityops.v1",
    "facility.landing_pad.visible.v1",
    "facility.radio.base_tower.v1",
}
SERVICE_MIN_CENTER_CLEARANCE_M = {
    "facility.landing_pad.visible.v1": LANE_HALF_WIDTH_M + 5.6,
    "facility.charger.cityops.v1": LANE_HALF_WIDTH_M + 3.6,
    "facility.radio.base_tower.v1": LANE_HALF_WIDTH_M + 4.5,
}
SERVICE_FACILITY_OFFSET_FALLBACK_M = (0.0, 1.5, 3.0, 5.0, 8.0, 12.0, 18.0, 26.0)
HOME_PAD_CLEARANCE_BY_KIND_M = {
    "pedestrian": 4.75,
    "vehicle": 7.0,
    "uav": 5.6,
    "landing_pad": 8.4,
    "charger": 6.0,
    "tower": 7.0,
    "facility": 7.0,
    "structure": 18.0,
}
ROAD_BLOCKING_ASSETS = {
    "facility.barrier.basic",
    "prop.roadwork.barrier.v1",
    "prop.roadwork.construction_fence.v1",
    "prop.roadwork.traffic_cone.v1",
}
GROUND_Z_M = 0.0
DEFAULT_DURATION_TICKS = 900
CAPTURE_TICK_STEP = 5
UAV_PAD_HOVER_Z_M = 3.0
UAV_TAKEOFF_ENTRY_TICK = 30
UAV_TAKEOFF_SPEED_MPS = 5.0
UAV_LANDING_AFTER_SCENARIO_DELAY_TICKS = 80
UAV_LANDING_ALTITUDE_MAX_M = 8.0
UAV_MISSION_ALTITUDE_MIN_M = 18.0
SCRIPT_TICK_HZ = 10.0
UAV_MAX_SPEED_MPS = 8.0
BOUNDARY_ROUTE_PLANNING_MARGIN_M = 0.05
UAV_TOUCHDOWN_ALTITUDE_TOLERANCE_M = 1.5
UAV_TOUCHDOWN_DWELL_TICKS = CAPTURE_TICK_STEP
UAV_FINAL_DESCENT_MIN_M = 1.0
MOVE_START_TOLERANCE_GROUND_M = 0.75
MOVE_START_TOLERANCE_UAV_M = 1.5
PEDESTRIAN_VEHICLE_COLLISION_EVENT_TYPE = "intentional_pedestrian_vehicle_collision"
PEDESTRIAN_VEHICLE_INTERACTION_EVENT_TYPE = "intentional_pedestrian_vehicle_interaction"
PEDESTRIAN_VEHICLE_ALLOWED_EVENT_TYPES = {
    PEDESTRIAN_VEHICLE_COLLISION_EVENT_TYPE,
    PEDESTRIAN_VEHICLE_INTERACTION_EVENT_TYPE,
}
PEDESTRIAN_VEHICLE_CLEARANCE_CHECK = "pedestrian_vehicle_dynamic_clearance"
MAX_WAYPOINT_SEGMENT_M = {
    "uav": 300.0,
    "vehicle": 80.0,
    "pedestrian": 20.0,
    "asset": 120.0,
}
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
BACKGROUND_PEDESTRIAN_OFFSET_PRIORITY_M = (4.8, 6.2, 8.0, 10.0, 12.0, 3.6, 2.4, SIDEWALK_MIN_OFFSET_FROM_CURB_M)
PEDESTRIAN_VEHICLE_DYNAMIC_CLEARANCE_MIN_M = 4.0
GROUND_FLOW_MIN_VISIBLE_MOTION_RATIO = 0.85
GROUND_FLOW_ROUTE_DURATION_TICKS = 900
BACKGROUND_PEDESTRIAN_TARGET_ROUTE_LENGTH_M = (
    BACKGROUND_PEDESTRIAN_FLOW_SPEED_MPS
    * (GROUND_FLOW_ROUTE_DURATION_TICKS / 10.0)
    * GROUND_FLOW_MIN_VISIBLE_MOTION_RATIO
)
BACKGROUND_PEDESTRIAN_ROUTE_STEP_M = 8.0
BACKGROUND_PEDESTRIAN_ROUTE_EDGE_MARGIN_M = 6.0
BACKGROUND_PEDESTRIAN_MAX_PACING_TRAVERSES = 8
BACKGROUND_VEHICLE_MIN_ROUTE_SPAN_M = 24.0
BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M = 9.0
CONTRACT_RUNTIME_VISIBLE_PADDING_M = 60.0
EVENT_ACTOR_ENTRY_BOUNDARY_CLEARANCE_M = 6.0
EVENT_ACTOR_ENTRY_ROUTE_STEP_M = 4.0
EVENT_ACTOR_ENTRY_MAX_REVERSE_EDGES = 16
EVENT_ACTOR_ENTRY_MAX_SEGMENT_M = 8.0
EVENT_ACTOR_ENTRY_SPEED_MIN_MPS = 1.0
EVENT_ACTOR_ENTRY_SPEED_MAX_MPS = 10.0
SUMO_TO_TRAFFIC_BUNDLE_MAX_SNAP_M = 15.0
FORCED_LANDING_DESCENT_SPEED_MPS = 8.0
# The longest forced-touchdown terminal leg is 2.2 m.  At 3.0 m/s it
# completes in ceil(2.2 / 3.0 * 10) = 8 ticks, so the stationary touchdown
# sample follows the airborne falling edge within the next two 5-tick samples.
FORCED_LANDING_TOUCHDOWN_SPEED_MPS = 3.0


def default_uav_sensor_fov_deg() -> float:
    profile = json.loads(UAV_SENSOR_PROFILE_PATH.read_text(encoding="utf-8-sig"))
    fov = float(profile.get("hfov_deg") or 0.0)
    if fov <= 0.0:
        raise RuntimeError(
            f"UAV sensor profile lacks positive hfov_deg: {UAV_SENSOR_PROFILE_PATH}"
        )
    return fov


def default_uav_sensor_profile() -> dict[str, Any]:
    profile = json.loads(UAV_SENSOR_PROFILE_PATH.read_text(encoding="utf-8-sig"))
    fov = float(profile.get("hfov_deg") or 0.0)
    width = int(profile.get("width") or 0)
    height = int(profile.get("height") or 0)
    rotation = dict(profile.get("fixed_rotation_offset_deg") or {})
    if fov <= 0.0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"UAV sensor profile lacks hfov/width/height: {UAV_SENSOR_PROFILE_PATH}"
        )
    if float(rotation.get("pitch_deg", 0.0)) > -60.0:
        raise RuntimeError(
            "UAV sensor profile must be downward looking for inspect coverage: "
            f"{UAV_SENSOR_PROFILE_PATH}"
        )
    return {
        "source": "Config/LowAltitude/uav_sensor_profile.json",
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

RUNTIME_STATE_SCHEDULE_SCHEMA_NAME = "epi_runtime_state_schedule"
RUNTIME_STATE_ALLOWED_FAMILIES = frozenset(RUNTIME_STATE_FIELDS)


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
class RuntimeStateSchedule:
    profile_id: str
    source_path: Path
    scenario_id: str
    initial_entity_states: dict[str, dict[str, dict[str, Any]]]
    event_state_transitions: dict[str, list[dict[str, Any]]]


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
class RoadLaneMetadata:
    lanes: int
    width_m: float


@dataclass(frozen=True)
class BackgroundVehiclePolicy:
    add_count: int
    min_uav_count: int = 0
    reason: str = ""


@dataclass(frozen=True)
class DeterministicTrafficTemplate:
    min_vehicle_count: int
    seed00_direction_weights: dict[str, float]
    seed01_direction_weights: dict[str, float]
    seed02_direction_weights: dict[str, float]
    semantic_vehicle_roles: tuple[str, ...]
    source_policy: str = "explicit_episode_plan_only_v1"


DETERMINISTIC_TRAFFIC_TEMPLATE_ORDINARY = DeterministicTrafficTemplate(
    min_vehicle_count=8,
    seed00_direction_weights={"inbound": 0.7, "outbound": 0.2, "cross": 0.1},
    seed01_direction_weights={"inbound": 0.34, "outbound": 0.33, "cross": 0.33},
    seed02_direction_weights={"inbound": 0.2, "outbound": 0.7, "cross": 0.1},
    semantic_vehicle_roles=("deterministic_background_vehicle",),
)
DETERMINISTIC_TRAFFIC_TEMPLATE_EVENT = DeterministicTrafficTemplate(
    min_vehicle_count=12,
    seed00_direction_weights={"inbound": 0.7, "outbound": 0.2, "cross": 0.1},
    seed01_direction_weights={"inbound": 0.34, "outbound": 0.33, "cross": 0.33},
    seed02_direction_weights={"inbound": 0.2, "outbound": 0.7, "cross": 0.1},
    semantic_vehicle_roles=("semantic_event_vehicle", "deterministic_background_vehicle"),
)
DETERMINISTIC_TRAFFIC_TEMPLATE_COMPLEX = DeterministicTrafficTemplate(
    min_vehicle_count=16,
    seed00_direction_weights={"inbound": 0.7, "outbound": 0.2, "cross": 0.1},
    seed01_direction_weights={"inbound": 0.34, "outbound": 0.33, "cross": 0.33},
    seed02_direction_weights={"inbound": 0.2, "outbound": 0.7, "cross": 0.1},
    semantic_vehicle_roles=(
        "emergency_vehicle",
        "police_vehicle",
        "yield_vehicle",
        "disabled_vehicle",
        "weather_slowdown_vehicle",
        "deterministic_background_vehicle",
    ),
)


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


def deterministic_traffic_template_for(scenario_id: str) -> DeterministicTrafficTemplate:
    if scenario_id.startswith(("X", "L5")):
        return DETERMINISTIC_TRAFFIC_TEMPLATE_COMPLEX
    if scenario_id.startswith(("L2", "L3", "L4")):
        return DETERMINISTIC_TRAFFIC_TEMPLATE_EVENT
    return DETERMINISTIC_TRAFFIC_TEMPLATE_ORDINARY


def deterministic_traffic_template_payload(template: DeterministicTrafficTemplate) -> dict[str, Any]:
    return {
        "source_policy": template.source_policy,
        "min_vehicle_count": template.min_vehicle_count,
        "seed_profiles": {
            "seed00": {
                "name": "morning_peak",
                "direction_bias": "inbound_to_roi",
                "direction_weights": template.seed00_direction_weights,
            },
            "seed01": {
                "name": "midday",
                "direction_bias": "balanced_cross_traffic",
                "direction_weights": template.seed01_direction_weights,
            },
            "seed02": {
                "name": "evening_peak",
                "direction_bias": "outbound_from_roi",
                "direction_weights": template.seed02_direction_weights,
            },
        },
        "semantic_vehicle_roles": list(template.semantic_vehicle_roles),
        "route_source": "sumo_explicit_vehicle_plan",
    }


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


def _load_road_lane_metadata(path: Path) -> dict[str, RoadLaneMetadata]:
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


def _road_lane_metadata_for_edge(edge_id: str) -> RoadLaneMetadata | None:
    raw_id = str(edge_id)
    if raw_id.startswith(":"):
        return None
    match = re.fullmatch(r"(cg_edge_\d+)(?:_[a-z0-9]+)*", raw_id)
    if match is None:
        raise ValueError(f"SUMO edge does not identify a canonical road: {edge_id}")
    return ROAD_LANE_METADATA.get(match.group(1))


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
ROAD_LANE_METADATA = _load_road_lane_metadata(ROAD_GEOJSON)
SUMO_GROUND_FLOW = SumoGroundFlowPlanner(
    SUMO_NET_XML,
    max_start_snap_m=SUMO_TO_TRAFFIC_BUNDLE_MAX_SNAP_M,
    lane_center_samples_csv=LANE_SAMPLES_CSV,
    lane_connections_csv=LANE_CONNECTIONS_CSV,
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


def dist3(a: list[float], b: list[float]) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def motion_class_from_asset(asset_id: str) -> str:
    if asset_id.startswith("uav."):
        return "uav"
    if asset_id.startswith("vehicle."):
        return "vehicle"
    if asset_id.startswith("pedestrian."):
        return "pedestrian"
    return "asset"


def compiled_event_allows_validation_skip(event: dict[str, Any], check_name: str, errors: list[str], scenario_id: str) -> bool:
    event_type = str(event.get("validation_event_type") or "")
    skip_checks = {str(item) for item in event.get("validation_skip_checks") or []}
    if not event_type and check_name not in skip_checks:
        return False
    if event_type != "intentional_motion_discontinuity":
        errors.append(
            f"{scenario_id}: event {event.get('event_id')} has unsupported validation_event_type={event_type!r}; "
            "only intentional_motion_discontinuity may skip motion continuity checks"
        )
        return False
    if check_name not in skip_checks:
        return False
    if not str(event.get("validation_reason") or "").strip():
        errors.append(f"{scenario_id}: event {event.get('event_id')} skips {check_name} without validation_reason")
        return False
    return True


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


@lru_cache(maxsize=None)
def _sumo_edge_length_m(edge_id: str) -> float:
    return SUMO_GROUND_FLOW.edge_shape_length_m(edge_id)


def _sumo_lane_sample_at_s(edge_id: str, s_m: float) -> LaneSample:
    x_m, y_m, yaw_deg = SUMO_GROUND_FLOW.point_at_edge_s(edge_id, s_m)
    return LaneSample(
        edge_id=edge_id,
        lane_index=0,
        s_m=max(0.0, min(float(s_m), _sumo_edge_length_m(edge_id))),
        x_m=float(x_m),
        y_m=float(y_m),
        z_m=GROUND_Z_M,
        yaw_deg=float(yaw_deg),
    )


def _sumo_lane_sample_near(
    pos_enu: list[float],
    *,
    edge_id_hint: str | None = None,
) -> LaneSample:
    if edge_id_hint and edge_id_hint in SUMO_GROUND_FLOW.edges:
        _distance_m, s_m, x_m, y_m, yaw_deg = SUMO_GROUND_FLOW.project_point_to_edge(
            edge_id_hint,
            pos_enu,
        )
        edge_id = edge_id_hint
    else:
        edge_id, _distance_m, s_m, x_m, y_m, yaw_deg = (
            SUMO_GROUND_FLOW.nearest_edge_projection(pos_enu, mode="vehicle")
        )
    return LaneSample(
        edge_id=edge_id,
        lane_index=0,
        s_m=float(s_m),
        x_m=float(x_m),
        y_m=float(y_m),
        z_m=GROUND_Z_M,
        yaw_deg=float(yaw_deg),
    )


def _sumo_vehicle_point_at_s(
    edge_id: str,
    s_m: float,
    lateral_m: float = 0.0,
    z_m: float = GROUND_Z_M,
) -> list[float]:
    return _offset_from_lane(
        _sumo_lane_sample_at_s(edge_id, s_m),
        lateral_m,
        z_m,
    )


def _physical_vehicle_lane_offsets(sample: LaneSample) -> list[float]:
    metadata = _road_lane_metadata_for_edge(sample.edge_id)
    if metadata is None or metadata.lanes <= 1:
        return [0.0]
    lane_spacing_m = metadata.width_m / float(metadata.lanes) if metadata.width_m > 0.0 else LANE_HALF_WIDTH_M * 2.0
    return [
        ((float(index) + 0.5) - 0.5 * float(metadata.lanes)) * lane_spacing_m
        for index in range(metadata.lanes)
    ]


def _sumo_edge_intersects_polygon(edge_id: str, polygon: Polygon) -> bool:
    edge = SUMO_GROUND_FLOW.edges.get(edge_id)
    if edge is None or len(edge.shape_xy) < 2:
        return False
    return LineString(edge.shape_xy).intersects(polygon)


def _active_roi_polygon() -> Polygon | None:
    contract = ACTIVE_ROI_CONTRACT
    if contract is None:
        return None
    return Polygon(contract.capture_polygon_enu_m)


def _priority_multi_lane_corridor(
    scenario_id: str,
    focus: list[float],
    roi_polygon: Polygon | None,
) -> tuple[str, float, list[float]]:
    edges_with_vehicle_predecessors = {
        target_edge_id
        for source_edge_id, target_edge_ids in SUMO_GROUND_FLOW.adjacency.items()
        if source_edge_id in SUMO_GROUND_FLOW.edges
        and SUMO_GROUND_FLOW._edge_allows(
            SUMO_GROUND_FLOW.edges[source_edge_id],
            "vehicle",
        )
        for target_edge_id in target_edge_ids
        if target_edge_id in SUMO_GROUND_FLOW.edges
        and SUMO_GROUND_FLOW._edge_allows(
            SUMO_GROUND_FLOW.edges[target_edge_id],
            "vehicle",
        )
    }
    candidates: list[tuple[float, str, float, list[float]]] = []
    for edge in SUMO_GROUND_FLOW.edges.values():
        if not SUMO_GROUND_FLOW._edge_allows(edge, "vehicle"):
            continue
        if edge.edge_id not in edges_with_vehicle_predecessors:
            continue
        length_m = SUMO_GROUND_FLOW.edge_shape_length_m(edge.edge_id)
        interior_length_m = length_m
        if roi_polygon is not None:
            if not _sumo_edge_intersects_polygon(edge.edge_id, roi_polygon):
                continue
            interior_length_m = float(
                LineString(SUMO_GROUND_FLOW.edges[edge.edge_id].shape_xy)
                .intersection(roi_polygon)
                .length
            )
            if interior_length_m < 70.0:
                continue
        elif length_m < 112.0:
            continue
        distance_m, focus_s_m, x_m, y_m, _yaw_deg = (
            SUMO_GROUND_FLOW.project_point_to_edge(edge.edge_id, focus)
        )
        if roi_polygon is not None and not roi_polygon.covers(Point(float(x_m), float(y_m))):
            continue
        lane_offsets = _physical_vehicle_lane_offsets(
            _sumo_lane_sample_at_s(edge.edge_id, focus_s_m)
        )
        if len(lane_offsets) < 2:
            continue
        candidates.append(
            (
                float(distance_m),
                edge.edge_id,
                float(focus_s_m),
                sorted(float(value) for value in lane_offsets),
            )
        )
    if not candidates:
        region = "contract ROI" if roi_polygon is not None else "focus"
        raise RuntimeError(
            f"{scenario_id}: no legal multi-lane SUMO priority corridor in {region}"
        )
    _distance_m, edge_id, focus_s_m, lane_offsets = min(candidates)
    return edge_id, focus_s_m, lane_offsets


def _background_vehicle_lateral(sample: LaneSample, index: int) -> float:
    offsets = [value for value in _physical_vehicle_lane_offsets(sample) if abs(value) > 1e-6]
    if not offsets:
        return 0.0
    offsets.sort(key=lambda value: (abs(value), value))
    return float(offsets[index % len(offsets)])


def _physical_vehicle_lane_audit(sample: LaneSample, lateral_m: float) -> dict[str, Any]:
    metadata = _road_lane_metadata_for_edge(sample.edge_id)
    offsets = _physical_vehicle_lane_offsets(sample)
    return {
        "placement_semantics": "physical_vehicle_lane_center",
        "road_lane_count": int(metadata.lanes if metadata else 1),
        "road_width_m": round(float(metadata.width_m if metadata else LANE_HALF_WIDTH_M * 2.0), 3),
        "physical_lane_center_offsets_m": [round(float(value), 3) for value in offsets],
        "selected_physical_lane_offset_m": round(float(lateral_m), 3),
    }


def _annotate_physical_vehicle_lane(scene: dict[str, Any], sample: LaneSample, lateral_m: float) -> None:
    placement = dict(scene.get("placement") or {})
    placement.update(_physical_vehicle_lane_audit(sample, lateral_m))
    scene["placement"] = placement


def _vehicle_physical_lateral_from_scene(scene: dict[str, Any], sample: LaneSample, index: int) -> float:
    placement = dict(scene.get("placement") or {})
    background = dict(scene.get("background_vehicle") or {})
    for key, source in (
        ("physical_lane_lateral_m", background),
        ("selected_physical_lane_offset_m", placement),
        ("resolved_lateral_from_center_m", placement),
        ("lateral_offset_m", placement),
    ):
        try:
            value = float(source.get(key))
        except (TypeError, ValueError):
            continue
        if abs(value) > 1e-6:
            return value
    return _background_vehicle_lateral(sample, index)


def _select_physical_vehicle_lateral(
    sample: LaneSample,
    requested_lateral_m: float,
    desired_pos_enu: list[float] | None,
    *,
    index: int = 0,
) -> float:
    offsets = _physical_vehicle_lane_offsets(sample)
    if not offsets:
        return float(requested_lateral_m)
    requested = float(requested_lateral_m)
    if min(abs(requested - value) for value in offsets) <= 0.25:
        return float(min(offsets, key=lambda value: abs(requested - value)))
    reference = requested
    if desired_pos_enu is not None:
        nx, ny = _lane_normal(sample)
        desired = (float(desired_pos_enu[0]) - sample.x_m) * nx + (float(desired_pos_enu[1]) - sample.y_m) * ny
        if abs(desired) > 1e-6:
            reference = desired
    if abs(reference) <= 1e-6:
        return _background_vehicle_lateral(sample, index)
    same_side = [value for value in offsets if value * reference > 0.0]
    candidates = same_side or offsets
    return float(min(candidates, key=lambda value: abs(value - reference)))


def _vehicle_lane_point_near(
    pos_enu: list[float],
    lateral_reference_m: float,
    *,
    index: int = 0,
    z_m: float | None = None,
    edge_id_hint: str | None = None,
) -> list[float]:
    sample = _sumo_lane_sample_near(pos_enu, edge_id_hint=edge_id_hint)
    lateral_m = _select_physical_vehicle_lateral(sample, lateral_reference_m, pos_enu, index=index)
    return _offset_from_lane(sample, lateral_m, float(z_m if z_m is not None else (pos_enu[2] if len(pos_enu) > 2 else GROUND_Z_M)))


def _project_vehicle_waypoints_to_physical_lanes(
    waypoints: list[list[float]],
    lateral_reference_m: float,
    *,
    index: int = 0,
    edge_id_hint: str | None = None,
) -> list[list[float]]:
    projected: list[list[float]] = []
    for waypoint in waypoints:
        if not isinstance(waypoint, list) or len(waypoint) < 2:
            continue
        z_m = float(waypoint[2] if len(waypoint) > 2 else GROUND_Z_M)
        waypoint_edge_hint = edge_id_hint
        if edge_id_hint and edge_id_hint in SUMO_GROUND_FLOW.edges:
            hinted_distance_m = SUMO_GROUND_FLOW.project_point_to_edge(
                edge_id_hint,
                waypoint,
            )[0]
            nearest_edge_id, nearest_distance_m, _s_m, _x_m, _y_m, _yaw_deg = (
                SUMO_GROUND_FLOW.nearest_edge_projection(waypoint, mode="vehicle")
            )
            if nearest_edge_id != edge_id_hint and nearest_distance_m + 0.25 < hinted_distance_m:
                waypoint_edge_hint = nearest_edge_id
        projected.append(
            _vehicle_lane_point_near(
                waypoint,
                lateral_reference_m,
                index=index,
                z_m=z_m,
                edge_id_hint=waypoint_edge_hint,
            )
        )
    return _dedupe_route_points(projected)


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
    min_clearance_m = SERVICE_MIN_CENTER_CLEARANCE_M["facility.landing_pad.visible.v1"]
    if road_clearance_m + 0.05 < min_clearance_m:
        raise RuntimeError(
            f"Illegal landing pad placement for {context}: lane clearance "
            f"{road_clearance_m:.3f}m < {min_clearance_m:.3f}m"
        )
    return round(road_clearance_m, 3)


def landing_pad_anchor(
    desired_pos_enu: list[float],
    *,
    context: str,
    offset_from_curb_m: float = LANDING_PAD_MIN_OFFSET_FROM_CURB_M,
    edge_id_hint: str | None = None,
    s_hint: float | None = None,
) -> PlannedSidewalkAnchor:
    anchor = SPATIAL.plan_sidewalk_anchor(
        desired_pos_enu,
        edge_id_hint=edge_id_hint,
        s_hint=s_hint,
        offset_from_curb_m=offset_from_curb_m,
        allow_green=True,
        placement_semantics="uav_pad",
    )
    _validate_landing_pad_position(anchor.position_enu_m, context=context)
    return anchor


def _service_anchor_search_seeds(
    desired_pos_enu: list[float],
    *,
    base_offset_m: float,
    edge_id_hint: str | None = None,
    s_hint: float | None = None,
) -> list[tuple[list[float], str | None, float | None]]:
    seeds: list[tuple[list[float], str | None, float | None]] = [
        (list(desired_pos_enu), edge_id_hint, s_hint)
    ]
    if edge_id_hint is not None:
        return seeds
    for sample in LANES.nearby_diverse_samples(
        float(desired_pos_enu[0]),
        float(desired_pos_enu[1]),
        max_radius_m=80.0,
        longitudinal_bucket_m=10.0,
        max_count=96,
    ):
        for side in (1.0, -1.0):
            seeds.append(
                (
                    _offset_from_lane(
                        sample,
                        side * (LANE_HALF_WIDTH_M + float(base_offset_m)),
                        GROUND_Z_M,
                    ),
                    sample.edge_id,
                    sample.s_m,
                )
            )
    return seeds


def landing_pad_pose(
    desired_pos_enu: list[float],
    *,
    context: str,
    offset_from_curb_m: float = LANDING_PAD_MIN_OFFSET_FROM_CURB_M,
    edge_id_hint: str | None = None,
    s_hint: float | None = None,
) -> tuple[list[float], float, PlannedSidewalkAnchor]:
    base_offset_m = max(float(offset_from_curb_m), LANDING_PAD_MIN_OFFSET_FROM_CURB_M)
    errors: list[str] = []
    tried: set[tuple[float, float, float, str | None, float | None]] = set()
    for desired_candidate, candidate_edge_id, candidate_s in _service_anchor_search_seeds(
        desired_pos_enu,
        base_offset_m=base_offset_m,
        edge_id_hint=edge_id_hint,
        s_hint=s_hint,
    ):
        for extra_m in SERVICE_FACILITY_OFFSET_FALLBACK_M:
            candidate_offset_m = round(base_offset_m + float(extra_m), 3)
            candidate_key = (
                round(float(desired_candidate[0]), 3),
                round(float(desired_candidate[1]), 3),
                candidate_offset_m,
                candidate_edge_id,
                round(float(candidate_s), 3) if candidate_s is not None else None,
            )
            if candidate_key in tried:
                continue
            tried.add(candidate_key)
            try:
                anchor = landing_pad_anchor(
                    desired_candidate,
                    context=context,
                    offset_from_curb_m=candidate_offset_m,
                    edge_id_hint=candidate_edge_id,
                    s_hint=candidate_s,
                )
            except (RuntimeError, SpatialValidationError) as exc:
                errors.append(str(exc))
                continue
            return anchor.position_enu_m, float(anchor.sample.yaw_deg), anchor
    raise RuntimeError(f"Unable to place landing pad for {context}: {errors[:6]}")


def _validate_service_facility_position(pos_enu: list[float], *, asset: str, context: str) -> float:
    errors = SPATIAL.validation_errors_for_point(
        pos_enu,
        context=context,
        allow_road=False,
        allow_green=True,
        road_buffer_m=PEDESTRIAN_ROAD_BUFFER_M,
    )
    road_clearance_m = SPATIAL.nearest_lane_clearance(pos_enu)
    if errors:
        raise RuntimeError(f"Illegal service facility placement for {context}: {errors[:4]}")
    min_clearance_m = SERVICE_MIN_CENTER_CLEARANCE_M.get(asset, LANE_HALF_WIDTH_M + SERVICE_FACILITY_MIN_OFFSET_FROM_CURB_M)
    if road_clearance_m + 0.05 < min_clearance_m:
        raise RuntimeError(
            f"Illegal service facility placement for {context}: lane clearance "
            f"{road_clearance_m:.3f}m < {min_clearance_m:.3f}m"
        )
    return round(road_clearance_m, 3)


def service_facility_pose(
    desired_pos_enu: list[float],
    *,
    asset: str,
    context: str,
    offset_from_curb_m: float = SERVICE_FACILITY_MIN_OFFSET_FROM_CURB_M,
) -> tuple[list[float], float, PlannedSidewalkAnchor, float]:
    if asset == "facility.landing_pad.visible.v1":
        pos, yaw, anchor = landing_pad_pose(
            desired_pos_enu,
            context=context,
            offset_from_curb_m=max(offset_from_curb_m, LANDING_PAD_MIN_OFFSET_FROM_CURB_M),
        )
        return pos, yaw, anchor, _validate_landing_pad_position(pos, context=context)
    base_offset_m = max(float(offset_from_curb_m), SERVICE_FACILITY_MIN_OFFSET_FROM_CURB_M)
    errors: list[str] = []
    desired_candidates = _service_anchor_search_seeds(
        desired_pos_enu,
        base_offset_m=base_offset_m,
    )
    tried: set[tuple[float, float, float, str | None, float | None]] = set()
    for desired_candidate, candidate_edge_id, candidate_s in desired_candidates:
        for extra_m in SERVICE_FACILITY_OFFSET_FALLBACK_M:
            candidate_offset_m = round(base_offset_m + float(extra_m), 3)
            candidate_key = (
                round(float(desired_candidate[0]), 3),
                round(float(desired_candidate[1]), 3),
                candidate_offset_m,
                candidate_edge_id,
                round(float(candidate_s), 3) if candidate_s is not None else None,
            )
            if candidate_key in tried:
                continue
            tried.add(candidate_key)
            try:
                anchor = SPATIAL.plan_sidewalk_anchor(
                    desired_candidate,
                    edge_id_hint=candidate_edge_id,
                    s_hint=candidate_s,
                    offset_from_curb_m=candidate_offset_m,
                    allow_green=True,
                    placement_semantics="service_facility",
                )
                clearance = _validate_service_facility_position(
                    anchor.position_enu_m,
                    asset=asset,
                    context=context,
                )
                if asset in {
                    "facility.charger.cityops.v1",
                    "facility.charging_pile.basic",
                }:
                    charging_contact = q(
                        float(anchor.position_enu_m[0]),
                        float(anchor.position_enu_m[1]),
                        float(anchor.position_enu_m[2]) + 1.2,
                    )
                    if not UAV_CORRIDORS.buildings.air_point_clear(
                        charging_contact
                    ):
                        raise RuntimeError(
                            f"Charging contact volume is obstructed for {context}: {charging_contact}"
                        )
            except (RuntimeError, SpatialValidationError) as exc:
                errors.append(str(exc))
                continue
            return (
                anchor.position_enu_m,
                float(anchor.sample.yaw_deg),
                anchor,
                clearance,
            )
    raise RuntimeError(f"Unable to place service facility {asset} for {context}: {errors[:6]}")


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
    else:
        # Non-pedestrian motion also needs an explicit activity transition when
        # the caller provides one.  Previously these arguments were silently
        # discarded, so a physically airborne UAV could remain labelled
        # ``preflight_on_pad`` throughout its mission.
        if activity_type is not None:
            params["activity_type"] = str(activity_type)
        if post_activity_type is not None:
            params["post_activity_type"] = str(post_activity_type)
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


def stage_event_actor(
    scene: dict[str, Any],
    *,
    policy: str,
    semantic_event_id: str,
) -> None:
    entity_id = str(scene.get("entity_id") or "")
    if not entity_id:
        raise RuntimeError("event actor scene lacks entity_id")
    if scene.get("spawn_policy") is not None:
        raise RuntimeError(f"{entity_id}: event actor already has a spawn policy")
    scene["activation_tick"] = DEFAULT_DURATION_TICKS + 1
    scene["enabled"] = False
    scene["spawn_policy"] = "event_script_only"
    scene["event_actor_motion_contract"] = {
        "schema": "event_actor_motion_contract_v1",
        "policy": policy,
        "semantic_event_id": semantic_event_id,
        "pre_activation_presence": "absent",
        "stationary_preplacement_allowed": False,
        "yaw_authority": "scene_lane_then_velocity",
    }


def event_actor_entry(
    *,
    scenario_id: str,
    scene: dict[str, Any],
    event_id: str,
    semantic_event_id: str,
    entry_tick: int,
    arrival_tick: int,
    route: list[list[float]],
    activity_type: str,
    requested_speed_mps: float | None = None,
) -> dict[str, Any]:
    if arrival_tick <= entry_tick:
        raise RuntimeError(
            f"{scenario_id}:{event_id}: arrival tick must follow entry tick"
        )
    start = scene_pos(scene)
    if start is None or len(route) < 2:
        raise RuntimeError(
            f"{scenario_id}:{event_id}: entry route must start at the staged scene pose"
        )
    if dist_xy(start, route[0]) > 0.05:
        route = [list(start), *[list(point) for point in route[1:]]]
    distance_m = path_length_m(route)
    speed_mps = (
        float(requested_speed_mps)
        if requested_speed_mps is not None
        else distance_m * SCRIPT_TICK_HZ / float(arrival_tick - entry_tick)
    )
    if not EVENT_ACTOR_ENTRY_SPEED_MIN_MPS <= speed_mps <= EVENT_ACTOR_ENTRY_SPEED_MAX_MPS:
        raise RuntimeError(
            f"{scenario_id}:{event_id}: modeled entry speed {speed_mps:.3f}m/s "
            "is outside the ground-actor operating range"
        )
    quantized_travel_ticks = quantized_path_duration_ticks(route, speed_mps)
    modeled_entry_tick = arrival_tick - quantized_travel_ticks
    if modeled_entry_tick < 0:
        raise RuntimeError(
            f"{scenario_id}:{event_id}: tick-quantized entry requires "
            f"{quantized_travel_ticks} ticks before arrival {arrival_tick}"
        )
    stage_event_actor(
        scene,
        policy="offstage_then_continuous_lane_entry",
        semantic_event_id=semantic_event_id,
    )
    contract = scene["event_actor_motion_contract"]
    contract.update(
        {
            "entry_event_id": event_id,
            "entry_tick": int(modeled_entry_tick),
            "arrival_tick": int(arrival_tick),
            "route_length_m": round(distance_m, 6),
            "entry_speed_mps": round(speed_mps, 6),
            "duration_model": "sum_segment_ceil_distance_over_speed_ticks",
            "minimum_visible_motion_ratio_before_event": 1.0,
        }
    )
    entry = context_event(
        event_id,
        tick(modeled_entry_tick),
        [
            spawn_from_scene(f"spawn_{scene['entity_id']}_entry", scene),
            move(
                f"move_{scene['entity_id']}_entry",
                str(scene["entity_id"]),
                route,
                speed_mps,
                activity_type=activity_type,
                post_activity_type=activity_type,
            ),
        ],
        0,
        scenario_id,
        "Event actor enters continuously on its legal lane before the semantic event",
        [str(scene["entity_id"])],
    )
    entry["capture_boundary_scope"] = "pre_roll_outside_event_stage"
    return entry


def spawn_event_actor_at_dispatch(
    scene: dict[str, Any],
    event_def: dict[str, Any],
) -> None:
    event_id = str(event_def.get("event_id") or "")
    stage_event_actor(
        scene,
        policy="offstage_until_semantic_dispatch",
        semantic_event_id=event_id,
    )
    event_def.setdefault("actions", []).insert(
        0,
        spawn_from_scene(f"spawn_{scene['entity_id']}_at_dispatch", scene),
    )


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


def _event_trigger_tick(event_def: dict[str, Any], resolved_ticks: dict[str, int]) -> int | None:
    trigger = dict(event_def.get("trigger") or {})
    trigger_type = str(trigger.get("type") or "")
    if trigger_type == "tick":
        try:
            raw_tick = int(trigger.get("tick"))
            return int(
                math.ceil(raw_tick / CAPTURE_TICK_STEP) * CAPTURE_TICK_STEP
            )
        except (TypeError, ValueError):
            return None
    if trigger_type in {"event_fired", "event_fired_after"}:
        event_ref = str(trigger.get("event_ref") or trigger.get("event_id") or "")
        if event_ref not in resolved_ticks:
            return None
        delay = 0
        if trigger_type == "event_fired_after":
            try:
                delay = int(trigger.get("delay_ticks") or 0)
            except (TypeError, ValueError):
                delay = 0
        raw_tick = resolved_ticks[event_ref] + max(1, delay)
        return int(math.ceil(raw_tick / CAPTURE_TICK_STEP) * CAPTURE_TICK_STEP)
    return None


def _event_trigger_ticks(events: list[dict[str, Any]]) -> dict[str, int]:
    resolved: dict[str, int] = {}
    pending = {str(item.get("event_id") or ""): item for item in events if item.get("event_id")}
    while pending:
        progressed = False
        for event_id, event_def in list(pending.items()):
            tick_value = _event_trigger_tick(event_def, resolved)
            if tick_value is None:
                continue
            resolved[event_id] = tick_value
            pending.pop(event_id)
            progressed = True
        if not progressed:
            break
    return resolved


def _event_dispatch_upper_bounds(events: list[dict[str, Any]]) -> dict[str, int]:
    """Resolve conservative dispatch upper bounds for authored event chains.

    A proximity trigger can fire during preceding motion, but the latest
    deterministic dispatch point is the completion of all preceding motion
    affecting its targets.  If that upper bound is not derivable, terminal
    feasibility cannot be proven from the authored graph.
    """

    resolved: dict[str, int] = {}
    pending = {
        str(item.get("event_id") or ""): item
        for item in events
        if item.get("event_id")
    }
    event_order = {
        str(item.get("event_id") or ""): index
        for index, item in enumerate(events)
        if item.get("event_id")
    }
    event_id_by_trigger_ref = {
        f"trig_{event_id}": event_id for event_id in event_order
    }
    while pending:
        progressed = False
        for event_id, event_def in list(pending.items()):
            trigger = dict(event_def.get("trigger") or {})
            trigger_type = str(trigger.get("type") or "")
            if trigger_type == "tick":
                try:
                    resolved[event_id] = int(
                        math.ceil(int(trigger["tick"]) / CAPTURE_TICK_STEP)
                        * CAPTURE_TICK_STEP
                    )
                except (KeyError, TypeError, ValueError):
                    raise RuntimeError(f"event {event_id}: tick trigger is not numeric")
            elif trigger_type in {"event_fired", "event_fired_after"}:
                predecessor = str(trigger.get("event_ref") or trigger.get("event_id") or "")
                if predecessor not in resolved:
                    continue
                delay = int(trigger.get("delay_ticks") or 0) if trigger_type == "event_fired_after" else 0
                raw_tick = resolved[predecessor] + max(1, delay)
                resolved[event_id] = int(
                    math.ceil(raw_tick / CAPTURE_TICK_STEP) * CAPTURE_TICK_STEP
                )
            elif trigger_type == "entity_proximity":
                index = event_order[event_id]
                entity_ids = {
                    str(trigger.get("entity_a") or ""),
                    str(trigger.get("entity_b") or ""),
                }
                candidates: list[int] = []
                for prior in events[:index]:
                    prior_id = str(prior.get("event_id") or "")
                    if prior_id not in resolved:
                        continue
                    prior_targets = {
                        str(value)
                        for value in prior.get("log_target_ids") or []
                        if value
                    }
                    prior_moves = [
                        action_def
                        for action_def in prior.get("actions") or []
                        if action_params(action_def).get("type") == "move_entity"
                        and str(action_params(action_def).get("entity_id") or "") in entity_ids
                    ]
                    if prior_targets.intersection(entity_ids) or prior_moves:
                        completion = resolved[prior_id] + max(
                            (move_duration_ticks(action_def) for action_def in prior_moves),
                            default=0,
                        )
                        candidates.append(completion)
                if not candidates:
                    raise RuntimeError(
                        f"event {event_id}: proximity activation has no deterministic predecessor window"
                    )
                resolved[event_id] = int(
                    math.ceil(max(candidates) / CAPTURE_TICK_STEP)
                    * CAPTURE_TICK_STEP
                )
            elif trigger_type == "composite":
                raw_children = trigger.get("composite_children") or trigger.get("children") or []
                child_events = [
                    event_id_by_trigger_ref.get(str(child), str(child).removeprefix("trig_"))
                    for child in raw_children
                ]
                if not child_events or any(child not in resolved for child in child_events):
                    continue
                operator = str(trigger.get("composite_operator") or trigger.get("operator") or "AND").upper()
                child_ticks = [resolved[child] for child in child_events]
                resolved[event_id] = max(child_ticks) if operator == "AND" else min(child_ticks)
            elif trigger_type == "weather_state":
                prior_ticks = [
                    resolved[str(prior.get("event_id") or "")]
                    for prior in events[: event_order[event_id]]
                    if str(prior.get("event_id") or "") in resolved
                ]
                if not prior_ticks:
                    raise RuntimeError(
                        f"event {event_id}: weather activation has no deterministic predecessor window"
                    )
                resolved[event_id] = max(prior_ticks)
            else:
                raise RuntimeError(f"event {event_id}: unsupported trigger type {trigger_type!r}")
            pending.pop(event_id)
            progressed = True
        if not progressed:
            unresolved = ", ".join(sorted(pending))
            raise RuntimeError(f"event dispatch upper bounds are cyclic or unresolved: {unresolved}")
    return resolved


AUTHORED_EVENT_CHAIN_POLICIES: dict[tuple[str, str], dict[str, Any]] = {
    ("L1-3_v1", "approach_boundary"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_l1_3_v1_primary",
            "lifecycle_takeoff_intruder_l1_3_v1",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L1-3_v2", "approach_boundary"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_l1_3_v2_primary",
            "lifecycle_takeoff_intruder_l1_3_v2",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L1-4_v1", "lifecycle_takeoff_uav_l1_4_v1_tertiary"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_l1_4_v1_primary",
            "lifecycle_takeoff_uav_l1_4_v1_secondary",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L1-4_v1", "corridor_capacity_exceeded"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_l1_4_v1_primary",
            "lifecycle_takeoff_uav_l1_4_v1_secondary",
            "lifecycle_takeoff_uav_l1_4_v1_tertiary",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L1-4_v2", "lifecycle_takeoff_uav_l1_4_v2_tertiary"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_l1_4_v2_primary",
            "lifecycle_takeoff_uav_l1_4_v2_secondary",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L1-4_v2", "corridor_capacity_exceeded"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_l1_4_v2_primary",
            "lifecycle_takeoff_uav_l1_4_v2_secondary",
            "lifecycle_takeoff_uav_l1_4_v2_tertiary",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L2-4_v1", "dual_pad_approach"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_priority_l2_4_v1",
            "lifecycle_takeoff_uav_divert_l2_4_v1",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L2-4_v2", "dual_pad_approach"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_priority_l2_4_v2",
            "lifecycle_takeoff_uav_divert_l2_4_v2",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L4-1_v1", "converging_approach"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_conflict_l4_1_v1_a",
            "lifecycle_takeoff_uav_conflict_l4_1_v1_b",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L4-1_v2", "converging_approach"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_conflict_l4_1_v2_a",
            "lifecycle_takeoff_uav_conflict_l4_1_v2_b",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L6-4_v1", "wideband_jamming"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_digital_l6_4_v1",
            "lifecycle_takeoff_uav_digital_l6_4_v1_secondary",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L6-4_v2", "wideband_jamming"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_digital_l6_4_v2",
            "lifecycle_takeoff_uav_digital_l6_4_v2_secondary",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("X4_fog_to_uav_conflict", "fog_onset"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_x4_a",
            "lifecycle_takeoff_uav_x4_b",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("X5_comm_failure_to_pad_contention", "station_failure"): {
        "causal_predecessor_event_ids": (
            "lifecycle_takeoff_uav_x5_priority",
            "lifecycle_takeoff_uav_x5_second",
        ),
        "causal_predecessor_mode": "all_of",
    },
    ("L4-10_v1", "civilian_vehicle_yields"): {
        "causal_temporal_policy": "same_tick_required",
        "temporal_source_event_id": "ambulance_priority_approach",
    },
    ("L4-10_v2", "civilian_vehicle_yields"): {
        "causal_temporal_policy": "same_tick_required",
        "temporal_source_event_id": "ambulance_priority_approach",
    },
    ("L5-1_v1", "rain_condition_met"): {
        "causal_temporal_policy": "same_tick_required",
        "temporal_source_event_id": "weather_onset",
    },
    ("L5-1_v2", "rain_condition_met"): {
        "causal_temporal_policy": "same_tick_required",
        "temporal_source_event_id": "weather_onset",
    },
    ("L5-1_v3", "rain_condition_met"): {
        "causal_temporal_policy": "same_tick_required",
        "temporal_source_event_id": "weather_onset",
    },
    ("L5-2_v1", "fog_condition_met"): {
        "causal_temporal_policy": "same_tick_required",
        "temporal_source_event_id": "weather_onset",
    },
    ("L5-2_v2", "fog_condition_met"): {
        "causal_temporal_policy": "same_tick_required",
        "temporal_source_event_id": "weather_onset",
    },
    ("L5-3_v1", "wind_condition_met"): {
        "causal_temporal_policy": "same_tick_required",
        "temporal_source_event_id": "weather_onset",
    },
    ("L5-3_v2", "wind_condition_met"): {
        "causal_temporal_policy": "same_tick_required",
        "temporal_source_event_id": "weather_onset",
    },
    ("X1_rain_to_c2loss_to_forced_landing", "rain_threshold"): {
        "causal_temporal_policy": "same_tick_required",
        "temporal_source_event_id": "rain_onset",
    },
    ("X4_fog_to_uav_conflict", "fog_threshold"): {
        "causal_temporal_policy": "same_tick_required",
        "temporal_source_event_id": "fog_onset",
    },
}


def apply_authored_event_chain_policies(
    events: list[dict[str, Any]], scenario_id: str, trigger_ticks: dict[str, int]
) -> None:
    """Apply only reviewed per-event policies and verify them against script facts."""

    event_by_id = {str(item.get("event_id") or ""): item for item in events}
    if len(event_by_id) != len(events) or "" in event_by_id:
        raise RuntimeError(f"{scenario_id}: event IDs must be unique non-empty strings")
    positions = {event_id: index for index, event_id in enumerate(event_by_id)}
    scenario_policies = {
        event_id: policy
        for (policy_scenario_id, event_id), policy in AUTHORED_EVENT_CHAIN_POLICIES.items()
        if policy_scenario_id == scenario_id
    }
    policy_fields = {
        "causal_predecessor_event_ids",
        "causal_predecessor_mode",
        "causal_temporal_policy",
    }
    unexpected = [
        event_id
        for event_id, item in event_by_id.items()
        if policy_fields & item.keys() and event_id not in scenario_policies
    ]
    if unexpected:
        raise RuntimeError(
            f"{scenario_id}: authored causal policy is absent from the reviewed table: "
            f"{sorted(unexpected)}"
        )

    for target_id, policy in scenario_policies.items():
        target = event_by_id.get(target_id)
        if target is None:
            raise RuntimeError(
                f"{scenario_id}: reviewed causal policy target is absent: {target_id}"
            )
        predecessor_intent = str(target.get("causal_predecessor_intent") or "")
        declared_ids = tuple(policy.get("causal_predecessor_event_ids") or ())
        if declared_ids:
            candidates = [
                item
                for item in events[: positions[target_id]]
                if str(item.get("intent") or "") == predecessor_intent
            ]
            if not candidates:
                raise RuntimeError(
                    f"{scenario_id}:{target_id}: reviewed predecessor intent has no candidates"
                )
            latest_stage_number = max(
                int(str(item["intent_stage"]).split(".", 1)[0])
                for item in candidates
            )
            latest_ids = tuple(
                str(item["event_id"])
                for item in candidates
                if int(str(item["intent_stage"]).split(".", 1)[0])
                == latest_stage_number
            )
            if set(declared_ids) != set(latest_ids):
                raise RuntimeError(
                    f"{scenario_id}:{target_id}: reviewed predecessor IDs "
                    f"{declared_ids} differ from latest-stage script facts {latest_ids}"
                )
            target["causal_predecessor_event_ids"] = list(declared_ids)
            target["causal_predecessor_mode"] = policy["causal_predecessor_mode"]

        temporal_policy = policy.get("causal_temporal_policy")
        if temporal_policy:
            source_id = str(policy["temporal_source_event_id"])
            source = event_by_id.get(source_id)
            if source is None or positions[source_id] >= positions[target_id]:
                raise RuntimeError(
                    f"{scenario_id}:{target_id}: reviewed temporal source is not earlier: "
                    f"{source_id}"
                )
            if str(source.get("intent") or "") != predecessor_intent:
                raise RuntimeError(
                    f"{scenario_id}:{target_id}: reviewed temporal source intent differs"
                )
            source_tick = trigger_ticks.get(source_id)
            target_tick = trigger_ticks.get(target_id)
            target_trigger = dict(target.get("trigger") or {})
            trigger_type = str(target_trigger.get("type") or "")
            if source_tick is None:
                raise RuntimeError(
                    f"{scenario_id}:{target_id}: reviewed temporal source tick is unresolved"
                )
            if target_tick is not None and source_tick != target_tick:
                raise RuntimeError(
                    f"{scenario_id}:{target_id}: same_tick_required differs from script "
                    f"facts ({source_tick} != {target_tick})"
                )
            if target_tick is None and trigger_type == "weather_state":
                parameter = str(target_trigger.get("weather_parameter") or "")
                threshold = float(target_trigger.get("weather_value"))
                operator = str(target_trigger.get("weather_operator") or "gte")
                authored_values = [
                    action_params(action_def).get("overrides", {}).get(parameter)
                    for action_def in source.get("actions") or []
                    if action_params(action_def).get("type") == "set_weather"
                ]
                comparisons = {
                    "gte": lambda value: float(value) >= threshold,
                    "gt": lambda value: float(value) > threshold,
                    "lte": lambda value: float(value) <= threshold,
                    "lt": lambda value: float(value) < threshold,
                    "eq": lambda value: float(value) == threshold,
                }
                if operator not in comparisons or not any(
                    value is not None and comparisons[operator](value)
                    for value in authored_values
                ):
                    raise RuntimeError(
                        f"{scenario_id}:{target_id}: source does not author the "
                        "same-tick weather threshold fact"
                    )
            elif target_tick is None and trigger_type == "entity_proximity":
                target_entities = {
                    str(target_trigger.get("entity_a") or ""),
                    str(target_trigger.get("entity_b") or ""),
                }
                moved_entities = {
                    str(action_params(action_def).get("entity_id") or "")
                    for action_def in source.get("actions") or []
                    if action_params(action_def).get("type") == "move_entity"
                }
                if not target_entities.intersection(moved_entities):
                    raise RuntimeError(
                        f"{scenario_id}:{target_id}: source does not author the "
                        "same-tick proximity activation fact"
                    )
            elif target_tick is None:
                raise RuntimeError(
                    f"{scenario_id}:{target_id}: same_tick_required has unsupported "
                    f"conditional trigger {trigger_type!r}"
                )
            target["causal_temporal_policy"] = temporal_policy


def finalize_event_semantics(events: list[dict[str, Any]], scenario_id: str) -> None:
    trigger_ticks = _event_trigger_ticks(events)
    events_by_id = {
        str(item.get("event_id") or ""): item
        for item in events
        if item.get("event_id")
    }
    predecessor = ""
    predecessor_tick: int | None = None
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
        event_tick = trigger_ticks.get(str(item.get("event_id") or ""))
        trigger = dict(item.get("trigger") or {})
        trigger_type = str(trigger.get("type") or "")
        event_ref = str(trigger.get("event_ref") or "")
        if trigger_type in {"event_fired", "event_fired_after"} and event_ref:
            item["causal_predecessor_intent"] = str(
                events_by_id[event_ref].get("intent") or ""
            )
        elif predecessor and predecessor_tick is not None and event_tick is not None and predecessor_tick > event_tick:
            item["causal_predecessor_intent"] = ""
        else:
            item["causal_predecessor_intent"] = predecessor
        item["target_roles"] = list(item.get("target_roles") or _target_roles(list(item.get("log_target_ids") or [])))
        predecessor = intent
        predecessor_tick = event_tick
    apply_authored_event_chain_policies(events, scenario_id, trigger_ticks)


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
    validation_event_type: str | None = None,
    validation_reason: str | None = None,
    validation_skip_checks: list[str] | None = None,
) -> dict[str, Any]:
    event_intent = intent or _event_intent(event_id, title, category)
    intent_stage = f"{priority:02d}.{event_intent}"
    payload = {
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
    if validation_event_type:
        payload["validation_event_type"] = validation_event_type
        payload["validation_reason"] = validation_reason or ""
        payload["validation_skip_checks"] = validation_skip_checks or []
    return payload


def context_event(
    event_id: str,
    trigger: dict[str, Any],
    actions: list[dict[str, Any]],
    priority: int,
    scenario_id: str,
    title: str,
    targets: list[str],
    *,
    overlay: str = "scene_context",
) -> dict[str, Any]:
    return event(
        event_id,
        trigger,
        actions,
        priority,
        scenario_id,
        title,
        "scene_context",
        targets,
        "info",
        overlay=overlay,
        intent="context",
    )


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
    placement_authority: str | None = None,
    facility_subtype: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    service_anchor: PlannedSidewalkAnchor | None = None
    service_clearance_m: float | None = None
    if (
        category == "facility"
        and asset in SERVICE_FACILITY_ASSETS
        and placement_authority != "validated_service_facility"
    ):
        pos_enu, yaw, service_anchor, service_clearance_m = service_facility_pose(
            pos_enu,
            asset=asset,
            context=f"{entity_id} auto service facility",
        )
        placement_authority = "auto_service_facility"
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
        elif asset.startswith("facility.") or category in {"facility", "traffic_light"}:
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
    if asset.startswith("uav.") or category == "uav":
        scene["ground_reference_z_m"] = float(GROUND_Z_M)
        scene["ground_reference_source"] = "map_ground_reference_v1"
    semantic_scope = semantic_scope_for_source(
        category=category,
        asset_id=asset,
        declared_subtype=facility_subtype,
    )
    if semantic_scope is not None:
        entity_kind = canonical_entity_kind_for_subtype(
            str(semantic_scope["scope_subtype"])
        )
        spec["semantic_scope"] = copy.deepcopy(semantic_scope)
        spec["entity_kind"] = entity_kind
        scene["semantic_scope"] = copy.deepcopy(semantic_scope)
        scene["entity_kind"] = entity_kind
        scene["category"] = "facility"
    if placement_authority in {"auto_service_facility", "validated_service_facility"}:
        scene["placement_mode"] = "service_facility_anchor"
        placement = dict(scene["placement"])
        placement.update(
            {
                "position_enu_m": pos_enu,
                "resolved_position_enu_m": pos_enu,
                "placement_semantics": "service_facility",
                "scene_occupancy_authority": "low_altitude_scene_occupancy_authority_v1",
                "road_clearance_m": service_clearance_m,
            }
        )
        if service_anchor is not None:
            placement.update(
                {
                    "anchor_edge_id": service_anchor.sample.edge_id,
                    "anchor_lane_s_m": round(float(service_anchor.sample.s_m), 3),
                    "offset_from_curb_m": round(float(service_anchor.offset_from_curb_m), 3),
                    "resolved_lateral_from_center_m": round(float(service_anchor.resolved_lateral_from_center_m), 3),
                }
            )
        scene["placement"] = placement
        scene["scene_occupancy"] = {
            "authority": "low_altitude_scene_occupancy_authority_v1",
            "role": "service_facility",
            "physical": True,
            "blocking": False,
            "roadway_allowed": False,
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
    route_authority: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if asset.startswith("vehicle."):
        sample = (
            _sumo_lane_sample_at_s(edge_id, s)
            if prefer_edge_hint and edge_id in SUMO_GROUND_FLOW.edges
            else _sumo_lane_sample_near(pos_enu)
        )
    else:
        sample = LANES.resolve_edge_s(edge_id, s) if prefer_edge_hint else LANES.nearest_to_xy(float(pos_enu[0]), float(pos_enu[1]))
    sign = _side_sign_for_desired(sample, pos_enu)
    requested_lateral = float(lateral)
    placement_semantics = "lane_center"
    resolved_route = route
    if asset.startswith("prop.roadwork."):
        sign = 1.0 if float(lateral) >= 0.0 else -1.0
        offset_from_edge = max(abs(requested_lateral), ROADWORK_MIN_OFFSET_FROM_EDGE_M)
        requested_lateral = sign * (LANE_HALF_WIDTH_M + offset_from_edge)
        placement_semantics = "roadwork_shoulder"
    elif asset.startswith("vehicle."):
        if route_authority == "sumo_directed_lane_geometry":
            requested_lateral = 0.0
        elif route_authority == SUMO_DIRECTED_PHYSICAL_LANE_ROUTE_AUTHORITY:
            physical_offsets = _physical_vehicle_lane_offsets(sample)
            if min(
                abs(requested_lateral - offset_m)
                for offset_m in physical_offsets
            ) > 0.25:
                raise RuntimeError(
                    f"{entity_id}: directed physical lane offset "
                    f"{requested_lateral:.3f} is not one of {physical_offsets}"
                )
            requested_lateral = min(
                physical_offsets,
                key=lambda offset_m: abs(requested_lateral - offset_m),
            )
        else:
            requested_lateral = _select_physical_vehicle_lateral(sample, requested_lateral, pos_enu)
        placement_semantics = "physical_vehicle_lane_center"
        if route:
            if route_authority in {
                "sumo_directed_lane_geometry",
                SUMO_DIRECTED_PHYSICAL_LANE_ROUTE_AUTHORITY,
            }:
                resolved_route = _dedupe_route_points(route)
            else:
                resolved_route = _project_vehicle_waypoints_to_physical_lanes(
                    route,
                    requested_lateral,
                    edge_id_hint=sample.edge_id if prefer_edge_hint else None,
                )
    pos_enu = _offset_from_lane(sample, requested_lateral, float(pos_enu[2] if len(pos_enu) > 2 else GROUND_Z_M))
    yaw = sample.yaw_deg
    spec, scene = world_entity(
        entity_id, asset, category, pos_enu, yaw, mode, activation_tick, resolved_route, visual_state
    )
    scene["placement_mode"] = "lane_anchor"
    scene["placement"] = {
        "edge_id": sample.edge_id,
        "lane_index": sample.lane_index,
        "longitudinal_s": round(sample.s_m, 3),
        "lateral_offset_m": round(requested_lateral if asset.startswith("vehicle.") else lateral, 3),
        "lane_half_width_m": LANE_HALF_WIDTH_M,
        "resolved_lateral_from_center_m": round(requested_lateral, 3),
        "placement_semantics": placement_semantics,
        "resolved_position_enu_m": pos_enu,
        "rotation_deg": rot(yaw),
        "source_edge_id_hint": edge_id,
        "source_longitudinal_s_hint": round(s, 3),
    }
    if asset.startswith("vehicle."):
        scene["placement"].update(_physical_vehicle_lane_audit(sample, requested_lateral))
        if route_authority is not None:
            scene["placement"]["route_authority"] = route_authority
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
    try:
        anchor = SPATIAL.plan_sidewalk_anchor(
            pos_enu,
            edge_id_hint=edge_id,
            s_hint=s,
            offset_from_curb_m=offset,
            allow_green=False,
            placement_semantics="sidewalk",
        )
    except SpatialValidationError:
        anchor = SPATIAL.plan_sidewalk_anchor(
            pos_enu,
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
    spec, scene = world_entity(
        entity_id,
        "facility.landing_pad.visible.v1",
        "facility",
        pos_enu,
        yaw,
        "available",
        placement_authority="validated_service_facility",
    )
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


def service_facility_entity(
    entity_id: str,
    asset: str,
    desired_pos_enu: list[float],
    *,
    context: str,
    yaw: float = 0.0,
    mode: str = "available",
    visual_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if asset == "facility.landing_pad.visible.v1":
        pad_pos, pad_yaw, pad_anchor = landing_pad_pose(desired_pos_enu, context=context)
        spec, scene = pad_entity(entity_id, pad_pos, safe_id(entity_id), "service_facility", pad_yaw, pad_anchor)
        if visual_state:
            spec["visual_state"] = {**(spec.get("visual_state") or {}), **visual_state}
            scene["initial_state"] = {**(scene.get("initial_state") or {}), **visual_state}
        return spec, scene

    pos, resolved_yaw, anchor, clearance_m = service_facility_pose(
        desired_pos_enu,
        asset=asset,
        context=context,
    )
    spec, scene = world_entity(
        entity_id,
        asset,
        "facility",
        pos,
        resolved_yaw if yaw == 0.0 else yaw,
        mode,
        visual_state=visual_state,
        placement_authority="validated_service_facility",
    )
    scene["placement_mode"] = "service_facility_anchor"
    scene["placement"] = {
        "position_enu_m": pos,
        "resolved_position_enu_m": pos,
        "rotation_deg": rot(resolved_yaw if yaw == 0.0 else yaw),
        "anchor_edge_id": anchor.sample.edge_id,
        "anchor_lane_s_m": round(float(anchor.sample.s_m), 3),
        "offset_from_curb_m": round(float(anchor.offset_from_curb_m), 3),
        "resolved_lateral_from_center_m": round(float(anchor.resolved_lateral_from_center_m), 3),
        "placement_semantics": "service_facility",
        "road_clearance_m": clearance_m,
        "scene_occupancy_authority": "low_altitude_scene_occupancy_authority_v1",
    }
    scene["scene_occupancy"] = {
        "authority": "low_altitude_scene_occupancy_authority_v1",
        "role": "nonroad_context_facility" if asset in ROAD_BLOCKING_ASSETS else "service_facility",
        "physical": True,
        "blocking": asset in ROAD_BLOCKING_ASSETS,
        "roadway_allowed": False,
        "intentional_road_blocking": False,
    }
    return spec, scene


def add_evacuation_ped_cohort(
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    sid: str,
    prefix: str,
    start_origin: list[float],
    safe_origin: list[float],
    count: int = 8,
    min_safe_distance_from_origin_m: float | None = None,
) -> list[tuple[str, list[float], list[list[float]]]]:
    def cohort_sidewalk_point(origin: list[float], index: int, s_jitter_m: float = 0.0, offset_jitter_m: float = 0.0) -> tuple[LaneSample, float, float, list[float]]:
        base = LANES.nearest_to_xy(float(origin[0]), float(origin[1]))
        col = index % 4
        row = index // 4
        s_offset = (col - 1.5) * 5.0 + s_jitter_m
        offset_from_curb = max(
            SIDEWALK_MIN_OFFSET_FROM_CURB_M,
            GATHERING_MIN_OFFSET_FROM_CURB_M + row * 5.0 + (0.8 if col % 2 else 0.0) + offset_jitter_m,
        )
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
        start_offset_from_curb: float,
        target_offset_from_curb: float,
        start: list[float],
        target: list[float],
    ) -> list[list[float]]:
        s0 = float(start_sample.s_m)
        s1 = float(target_sample.s_m)
        steps = max(1, int(math.ceil(abs(s1 - s0) / 18.0)))
        points: list[list[float]] = [start]
        for step in range(1, steps):
            alpha = step / float(steps)
            s_m = s0 + (s1 - s0) * alpha
            offset_from_curb = start_offset_from_curb + (target_offset_from_curb - start_offset_from_curb) * alpha
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

    def route_fraction_min_distance(route_a: list[list[float]], route_b: list[list[float]]) -> float:
        if not route_a or not route_b:
            return float("inf")

        def point_at_fraction(route: list[list[float]], fraction: float) -> list[float]:
            if len(route) == 1:
                return route[0]
            segment_lengths = [dist_xy(a, b) for a, b in zip(route, route[1:])]
            total = sum(segment_lengths)
            if total <= 1e-6:
                return route[0]
            remaining = max(0.0, min(1.0, fraction)) * total
            for index, segment_length in enumerate(segment_lengths):
                if remaining <= segment_length or index == len(segment_lengths) - 1:
                    alpha = 0.0 if segment_length <= 1e-6 else remaining / segment_length
                    a = route[index]
                    b = route[index + 1]
                    return q(a[0] + (b[0] - a[0]) * alpha, a[1] + (b[1] - a[1]) * alpha, GROUND_Z_M)
                remaining -= segment_length
            return route[-1]

        min_distance = float("inf")
        for step in range(21):
            fraction = step / 20.0
            min_distance = min(min_distance, dist_xy(point_at_fraction(route_a, fraction), point_at_fraction(route_b, fraction)))
        return min_distance

    def cohort_safe_route(start_sample: LaneSample, start_offset: float, start: list[float]) -> tuple[list[float], list[list[float]]]:
        side_sign = _side_sign_for_desired(start_sample, start)
        min_s, max_s = LANES.edge_s_bounds(start_sample.edge_id)
        errors: list[str] = []
        for dir_sign in (evacuation_direction, -evacuation_direction):
            for travel_m in (36.0, 30.0, 24.0, 21.0, 18.0, 15.0, 12.0, 9.0, 6.0):
                target_s = max(min_s, min(max_s, start_sample.s_m + dir_sign * travel_m))
                target_sample = LANES.resolve_edge_s(start_sample.edge_id, target_s)
                for extra_offset in (0.0, 0.8, 1.6, 3.0, 5.0, 8.0, 12.0, 16.0):
                    desired_target = _offset_from_lane(
                        target_sample,
                        side_sign * (LANE_HALF_WIDTH_M + start_offset + extra_offset),
                        GROUND_Z_M,
                    )
                    try:
                        target_anchor = SPATIAL.plan_sidewalk_anchor(
                            desired_target,
                            edge_id_hint=start_sample.edge_id,
                            s_hint=target_s,
                            offset_from_curb_m=max(SIDEWALK_MIN_OFFSET_FROM_CURB_M, start_offset + extra_offset),
                            allow_green=True,
                            placement_semantics="sidewalk_evacuation_safe_target",
                        )
                    except SpatialValidationError as exc:
                        errors.append(str(exc))
                        continue
                    if target_anchor.sample.edge_id != start_sample.edge_id:
                        errors.append(f"sidewalk_evacuation_safe_target changed edge: {target_anchor.sample.edge_id}")
                        continue
                    target_side_sign = _side_sign_for_desired(target_anchor.sample, target_anchor.position_enu_m)
                    if target_side_sign * side_sign <= 0.0:
                        errors.append(f"sidewalk_evacuation_safe_target crossed roadway side: {target_anchor.position_enu_m}")
                        continue
                    target = target_anchor.position_enu_m
                    point_errors = SPATIAL.validation_errors_for_point(target, context="sidewalk_evacuation_safe_target", allow_road=False, allow_green=True)
                    origin_clearance_ok = (
                        min_safe_distance_from_origin_m is None
                        or dist_xy(target, start_origin) >= float(min_safe_distance_from_origin_m)
                    )
                    if (
                        not point_errors
                        and math.hypot(start[0] - target[0], start[1] - target[1]) >= 4.0
                        and origin_clearance_ok
                    ):
                        try:
                            route = cohort_route(
                                start_sample,
                                target_anchor.sample,
                                side_sign,
                                start_offset,
                                target_anchor.offset_from_curb_m,
                                start,
                                target,
                            )
                            if any(route_fraction_min_distance(route, existing_route) < 0.9 for existing_route in occupied_routes):
                                continue
                            return target, route
                        except SpatialValidationError as exc:
                            errors.append(str(exc))
                            continue
                    errors.extend(point_errors[:2])
        raise RuntimeError(f"Unable to plan same-edge evacuation target from {start}: {errors[:6]}")

    cohort: list[tuple[str, list[float], list[list[float]]]] = []
    occupied_starts: list[list[float]] = []
    occupied_routes: list[list[list[float]]] = []
    for index in range(count):
        start_sample = None
        start_offset = 0.0
        start_lateral = 0.0
        start: list[float] | None = None
        for s_jitter_m in (0.0, -2.5, 2.5, -5.0, 5.0, -7.5, 7.5, -10.0, 10.0):
            for offset_jitter_m in (0.0, 0.8, -0.8, 1.6, -1.2, 2.4):
                try:
                    candidate_sample, candidate_offset, candidate_lateral, candidate_start = cohort_sidewalk_point(
                        start_origin,
                        index,
                        s_jitter_m=s_jitter_m,
                        offset_jitter_m=offset_jitter_m,
                    )
                except SpatialValidationError:
                    continue
                if all(dist_xy(candidate_start, other) >= 0.9 for other in occupied_starts):
                    start_sample = candidate_sample
                    start_offset = candidate_offset
                    start_lateral = candidate_lateral
                    start = candidate_start
                    break
            if start is not None:
                break
        if start_sample is None or start is None:
            raise RuntimeError(f"Unable to place separated evacuation pedestrian cohort member {prefix}_{sid}_{index:02d}")
        occupied_starts.append(start)
        _safe, route = cohort_safe_route(start_sample, start_offset, start)
        occupied_routes.append(route)
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
        return asset.startswith("facility.") or scene_category in {"facility", "traffic_light"}
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


_SIDEWALK_CONTEXT_DISTANCE_CACHE: dict[tuple[float, float], float] = {}


def _sidewalk_context_distance_m(origin: list[float]) -> float:
    key = (round(float(origin[0]), 3), round(float(origin[1]), 3))
    cached = _SIDEWALK_CONTEXT_DISTANCE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        anchor = SPATIAL.plan_sidewalk_anchor(
            origin,
            offset_from_curb_m=BACKGROUND_PEDESTRIAN_OFFSET_PRIORITY_M[0],
            allow_green=True,
            placement_semantics="semantic_event_capture_context_probe",
        )
        required_route_m = _background_pedestrian_target_length_m("walking", 3)
        if _sidewalk_edge_route_capacity_m(anchor.sample.edge_id) + 0.25 < required_route_m * 1.25:
            distance_m = float("inf")
        else:
            distance_m = dist_xy(anchor.position_enu_m, origin)
    except SpatialValidationError:
        distance_m = float("inf")
    _SIDEWALK_CONTEXT_DISTANCE_CACHE[key] = distance_m
    return distance_m


def _iter_free_sidewalk_anchors(
    origin: list[float],
    occupied: list[list[float]],
    *,
    semantic: str,
    allow_green: bool = True,
    offset_priority_m: Sequence[float] | None = None,
    min_route_length_m: float | None = None,
):
    min_required_route_m = float(min_route_length_m or 0.0)
    nearby_bases: list[LaneSample] = []
    seen_edges: set[str] = set()
    for sample in sorted(LANES.samples, key=lambda item: (item.x_m - float(origin[0])) ** 2 + (item.y_m - float(origin[1])) ** 2):
        if sample.edge_id in seen_edges:
            continue
        min_edge_s, max_edge_s = LANES.edge_s_bounds(sample.edge_id)
        if max_edge_s - min_edge_s < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M:
            continue
        if min_required_route_m > 0.0 and _sidewalk_edge_route_capacity_m(sample.edge_id) + 0.25 < min_required_route_m:
            continue
        seen_edges.add(sample.edge_id)
        nearby_bases.append(sample)
        if len(nearby_bases) >= 48:
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

    offsets_from_curb: list[float] = []
    for value in offset_priority_m or (SIDEWALK_MIN_OFFSET_FROM_CURB_M, 2.4, 3.6, 4.8, 6.2, 8.0, 10.0, 12.0):
        offset = round(max(float(value), SIDEWALK_MIN_OFFSET_FROM_CURB_M), 3)
        if offset not in offsets_from_curb:
            offsets_from_curb.append(offset)

    for base_sample in nearby_bases:
        min_s, max_s = LANES.edge_s_bounds(base_sample.edge_id)
        preferred_sign = _side_sign_for_desired(base_sample, origin)
        for s_offset in (0.0, 6.0, -6.0, 12.0, -12.0, 18.0, -18.0, 28.0, -28.0, 42.0, -42.0, 60.0, -60.0):
            sample = LANES.resolve_edge_s(base_sample.edge_id, max(min_s, min(max_s, base_sample.s_m + s_offset)))
            for sign in (preferred_sign, -preferred_sign):
                for offset_from_curb in offsets_from_curb:
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
    scenario_id: str,
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
            candidate_lateral_m = _background_vehicle_lateral(sample, index)
            candidate_pos = _offset_from_lane(sample, candidate_lateral_m, GROUND_Z_M)
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
            candidate_route = _semantic_vehicle_route(sample, mode, index, lateral_m=candidate_lateral_m)
            if not candidate_route:
                continue
            if not _vehicle_pedestrian_dynamic_clearance_ok(
                scenario_id,
                scenes,
                events,
                [candidate_pos, *candidate_route],
            ):
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


def _sumo_vehicle_route(
    plan_start: list[float],
    index: int,
    *,
    span_start: list[float] | None = None,
) -> list[list[float]]:
    min_span = BACKGROUND_VEHICLE_MIN_ROUTE_SPAN_M + index * 2.0
    try:
        route = SUMO_GROUND_FLOW.plan_vehicle_route_enu(
            plan_start,
            min_xy_span_m=min_span,
            min_path_length_m=max(32.0, min_span + 8.0),
            max_edges=24,
        )
    except SumoRouteError:
        return []
    directed_route = _dedupe_route_points(route)
    route_start = span_start if span_start is not None else plan_start
    if not directed_route or _route_xy_span(route_start, directed_route) < min_span:
        return []
    return directed_route


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


def _scene_route_points(scene: dict[str, Any]) -> list[list[float]]:
    start = scene_pos(scene)
    if not start:
        return []
    route = [
        q(float(item[0]), float(item[1]), float(item[2] if len(item) > 2 else GROUND_Z_M))
        for item in scene.get("route_waypoints_enu_m") or []
        if isinstance(item, list) and len(item) >= 2
    ]
    return _dedupe_route_points([start, *route])


def _route_duration_ticks(route: list[list[float]], speed_mps: float) -> int:
    if len(route) < 2:
        return 0
    return int(math.ceil(path_length_m(route) / max(0.1, float(speed_mps)) * 10.0))


def _route_position_at_tick(route: list[list[float]], speed_mps: float, tick_value: int) -> list[float]:
    if not route:
        return q(0.0, 0.0, GROUND_Z_M)
    if len(route) == 1:
        return list(route[0])
    remaining = max(0.0, float(speed_mps) * (float(tick_value) / 10.0))
    current = list(route[0])
    for target in route[1:]:
        segment_len = dist_xy(current, target)
        if segment_len <= 1e-6:
            current = list(target)
            continue
        if remaining <= segment_len:
            alpha = remaining / segment_len
            return q(
                current[0] + (target[0] - current[0]) * alpha,
                current[1] + (target[1] - current[1]) * alpha,
                current[2] + (target[2] - current[2]) * alpha,
            )
        remaining -= segment_len
        current = list(target)
    return list(route[-1])


def _sidewalk_point_for_sample(sample: LaneSample, side_sign: float, offset_from_curb_m: float) -> list[float]:
    return _offset_from_lane(
        sample,
        side_sign * (LANE_HALF_WIDTH_M + max(float(offset_from_curb_m), SIDEWALK_MIN_OFFSET_FROM_CURB_M)),
        GROUND_Z_M,
    )


def _append_sidewalk_leg(
    points: list[list[float]],
    *,
    edge_id: str,
    start_s: float,
    target_s: float,
    side_sign: float,
    offset_from_curb_m: float,
) -> None:
    distance = float(target_s) - float(start_s)
    if abs(distance) <= 1e-6:
        return
    steps = max(1, int(math.ceil(abs(distance) / BACKGROUND_PEDESTRIAN_ROUTE_STEP_M)))
    for step in range(1, steps + 1):
        s_m = float(start_s) + distance * (float(step) / float(steps))
        sample = LANES.resolve_edge_s(edge_id, s_m)
        points.append(_sidewalk_point_for_sample(sample, side_sign, offset_from_curb_m))


def _usable_sidewalk_edge_span_m(edge_id: str) -> float:
    min_s, max_s = LANES.edge_s_bounds(edge_id)
    edge_span_m = max(0.0, max_s - min_s)
    edge_margin_m = min(BACKGROUND_PEDESTRIAN_ROUTE_EDGE_MARGIN_M, edge_span_m / 4.0)
    return max(0.0, edge_span_m - 2.0 * edge_margin_m)


def _sidewalk_edge_route_capacity_m(edge_id: str) -> float:
    usable_span_m = _usable_sidewalk_edge_span_m(edge_id)
    if usable_span_m + 0.25 < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M:
        return 0.0
    return usable_span_m * float(BACKGROUND_PEDESTRIAN_MAX_PACING_TRAVERSES)


def _fast_surface_errors(point_enu_m: Sequence[float], *, context: str, allow_green: bool) -> list[str]:
    point = SPATIAL._point_geometry(point_enu_m)
    errors: list[str] = []
    if not SPATIAL.bounds_prepared.covers(point):
        errors.append(f"{context} outside bounds.geojson: {list(point_enu_m)}")
    if SPATIAL.water_prepared.covers(point):
        errors.append(f"{context} inside water.geojson: {list(point_enu_m)}")
    if SPATIAL.building_prepared.covers(point):
        errors.append(f"{context} inside building.geojson: {list(point_enu_m)}")
    if not allow_green and SPATIAL.green_prepared.covers(point):
        errors.append(f"{context} inside green.geojson without explicit green allowance: {list(point_enu_m)}")
    return errors


def _fast_surface_segment_errors(
    a_enu_m: Sequence[float],
    b_enu_m: Sequence[float],
    *,
    context: str,
    allow_green: bool,
) -> list[str]:
    line = LineString([(float(a_enu_m[0]), float(a_enu_m[1])), (float(b_enu_m[0]), float(b_enu_m[1]))])
    errors: list[str] = []
    if not SPATIAL.bounds_prepared.covers(line):
        errors.append(f"{context} segment leaves bounds.geojson")
    if SPATIAL.water_prepared.intersects(line):
        errors.append(f"{context} segment intersects water.geojson")
    if SPATIAL.building_prepared.intersects(line):
        errors.append(f"{context} segment intersects building.geojson")
    if not allow_green and SPATIAL.green_prepared.intersects(line):
        errors.append(f"{context} segment intersects green.geojson without explicit green allowance")
    return errors


def _validated_sidewalk_route(
    points: list[list[float]],
    *,
    allow_green: bool,
    context: str,
) -> list[list[float]]:
    route = _dedupe_route_points(points)
    if len(route) < 2:
        return []
    for index, point in enumerate(route):
        if _fast_surface_errors(point, context=f"{context} point {index}", allow_green=allow_green):
            return []
    for index, (a, b) in enumerate(zip(route, route[1:])):
        if _fast_surface_segment_errors(a, b, context=f"{context} segment {index}", allow_green=allow_green):
            return []
    return route


def _long_sidewalk_ground_flow_route(
    start: list[float],
    sample: LaneSample,
    *,
    side_sign: float,
    offset_from_curb_m: float,
    min_path_length_m: float,
    min_xy_span_m: float,
    allow_green: bool,
    context: str,
) -> list[list[float]]:
    min_s, max_s = LANES.edge_s_bounds(sample.edge_id)
    edge_span_m = max(0.0, max_s - min_s)
    edge_margin_m = min(BACKGROUND_PEDESTRIAN_ROUTE_EDGE_MARGIN_M, edge_span_m / 4.0)
    route_min_s = min_s + edge_margin_m
    route_max_s = max_s - edge_margin_m
    if route_max_s <= route_min_s:
        route_min_s, route_max_s = min_s, max_s
    route_span_m = max(0.0, route_max_s - route_min_s)
    if route_span_m + 0.25 < min_xy_span_m:
        return []
    start_s = max(route_min_s, min(route_max_s, float(sample.s_m)))
    forward_available = max(0.0, route_max_s - start_s)
    backward_available = max(0.0, start_s - route_min_s)
    direction_order = (1.0, -1.0) if forward_available >= backward_available else (-1.0, 1.0)

    def candidate_for_s_values(s_values: list[float]) -> list[list[float]]:
        points = [list(start)]
        current_s = start_s
        for target_s in s_values:
            clamped = max(route_min_s, min(route_max_s, float(target_s)))
            _append_sidewalk_leg(
                points,
                edge_id=sample.edge_id,
                start_s=current_s,
                target_s=clamped,
                side_sign=side_sign,
                offset_from_curb_m=offset_from_curb_m,
            )
            current_s = clamped
        validated = _validated_sidewalk_route(points, allow_green=allow_green, context=context)
        if not validated:
            return []
        route = validated[1:]
        if path_length_m(validated) + 0.25 < min_path_length_m:
            return []
        if _route_xy_span(start, route) + 0.25 < min_xy_span_m:
            return []
        return route

    for direction in direction_order:
        available = forward_available if direction > 0.0 else backward_available
        if available + 0.25 >= min_path_length_m:
            target_s = start_s + direction * min_path_length_m
            route = candidate_for_s_values([target_s])
            if route:
                return route

    for direction in direction_order:
        first_boundary = route_max_s if direction > 0.0 else route_min_s
        second_boundary = route_min_s if direction > 0.0 else route_max_s
        first_leg_m = abs(first_boundary - start_s)
        if first_leg_m <= 1.0:
            continue
        remaining_m = max(0.0, min_path_length_m - first_leg_m)
        second_leg_m = min(abs(first_boundary - second_boundary), remaining_m)
        second_target = first_boundary - direction * second_leg_m
        route = candidate_for_s_values([first_boundary, second_target])
        if route:
            return route

    if route_span_m * float(BACKGROUND_PEDESTRIAN_MAX_PACING_TRAVERSES) + 0.25 >= min_path_length_m:
        for direction in direction_order:
            boundary_a = route_max_s if direction > 0.0 else route_min_s
            boundary_b = route_min_s if direction > 0.0 else route_max_s
            current_s = start_s
            target_s = boundary_a
            accumulated_m = 0.0
            s_values: list[float] = []
            for _ in range(BACKGROUND_PEDESTRIAN_MAX_PACING_TRAVERSES):
                leg_m = abs(target_s - current_s)
                if leg_m <= 1.0:
                    target_s = boundary_b if target_s == boundary_a else boundary_a
                    leg_m = abs(target_s - current_s)
                    if leg_m <= 1.0:
                        break
                s_values.append(target_s)
                accumulated_m += leg_m
                current_s = target_s
                if accumulated_m + 0.25 >= min_path_length_m:
                    break
                target_s = boundary_b if target_s == boundary_a else boundary_a
            if accumulated_m + 0.25 < min_path_length_m:
                continue
            route = candidate_for_s_values(s_values)
            if route:
                return route

    return []


def _allows_pedestrian_vehicle_collision(scenario_id: str, events: list[dict[str, Any]]) -> bool:
    for event_def in events:
        event_type = str(event_def.get("validation_event_type") or "")
        skip_checks = {str(item) for item in event_def.get("validation_skip_checks") or []}
        if (
            event_type in PEDESTRIAN_VEHICLE_ALLOWED_EVENT_TYPES
            and PEDESTRIAN_VEHICLE_CLEARANCE_CHECK in skip_checks
            and str(event_def.get("validation_reason") or "").strip()
        ):
            return True
    return False


def _min_route_clearance_m(
    route_a: list[list[float]],
    speed_a_mps: float,
    route_b: list[list[float]],
    speed_b_mps: float,
) -> float:
    if not route_a or not route_b:
        return float("inf")
    max_tick = max(_route_duration_ticks(route_a, speed_a_mps), _route_duration_ticks(route_b, speed_b_mps))
    if max_tick <= 0:
        return dist_xy(route_a[0], route_b[0])
    min_distance = float("inf")
    for tick_value in range(0, max_tick + 1):
        pos_a = _route_position_at_tick(route_a, speed_a_mps, tick_value)
        pos_b = _route_position_at_tick(route_b, speed_b_mps, tick_value)
        min_distance = min(min_distance, dist_xy(pos_a, pos_b))
        if min_distance < PEDESTRIAN_VEHICLE_DYNAMIC_CLEARANCE_MIN_M:
            return min_distance
    return min_distance


def _pedestrian_vehicle_dynamic_clearance_ok(
    scenario_id: str,
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    pedestrian_route: list[list[float]],
) -> bool:
    if _allows_pedestrian_vehicle_collision(scenario_id, events):
        return True
    for scene in scenes:
        if not str(scene.get("logical_asset_id") or "").startswith("vehicle."):
            continue
        vehicle_route = _scene_route_points(scene)
        if not vehicle_route:
            continue
        speed_mps = float((scene.get("ground_flow_contract") or {}).get("speed_mps") or BACKGROUND_VEHICLE_FLOW_SPEED_MPS)
        clearance_m = _min_route_clearance_m(
            pedestrian_route,
            BACKGROUND_PEDESTRIAN_FLOW_SPEED_MPS,
            vehicle_route,
            speed_mps,
        )
        if clearance_m < PEDESTRIAN_VEHICLE_DYNAMIC_CLEARANCE_MIN_M:
            return False
    return True


def _vehicle_pedestrian_dynamic_clearance_ok(
    scenario_id: str,
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    vehicle_route: list[list[float]],
) -> bool:
    if _allows_pedestrian_vehicle_collision(scenario_id, events):
        return True
    for scene in scenes:
        if not str(scene.get("logical_asset_id") or "").startswith("pedestrian."):
            continue
        pedestrian_route = _scene_route_points(scene)
        if not pedestrian_route:
            continue
        speed_mps = float((scene.get("ground_flow_contract") or {}).get("speed_mps") or BACKGROUND_PEDESTRIAN_FLOW_SPEED_MPS)
        clearance_m = _min_route_clearance_m(
            pedestrian_route,
            speed_mps,
            vehicle_route,
            BACKGROUND_VEHICLE_FLOW_SPEED_MPS,
        )
        if clearance_m < PEDESTRIAN_VEHICLE_DYNAMIC_CLEARANCE_MIN_M:
            return False
    return True


def _semantic_vehicle_route(sample: LaneSample, _mode: str, index: int, *, lateral_m: float = 0.0) -> list[list[float]]:
    authoritative_sample = (
        _sumo_lane_sample_at_s(sample.edge_id, sample.s_m)
        if sample.edge_id in SUMO_GROUND_FLOW.edges
        else _sumo_lane_sample_near(_offset_from_lane(sample, lateral_m, GROUND_Z_M))
    )
    truth_start = _offset_from_lane(authoritative_sample, 0.0, GROUND_Z_M)
    return _sumo_vehicle_route(truth_start, index, span_start=truth_start)


def _background_pedestrian_target_length_m(mode: str, index: int) -> float:
    target_length_m = BACKGROUND_PEDESTRIAN_TARGET_ROUTE_LENGTH_M + float(index) * 3.0
    if mode in {"evacuating"}:
        target_length_m = max(target_length_m, BACKGROUND_PEDESTRIAN_TARGET_ROUTE_LENGTH_M + 10.0)
    return target_length_m


def _semantic_ped_route(anchor: Any, mode: str, index: int) -> list[list[float]]:
    side_sign = _side_sign_for_desired(anchor.sample, anchor.position_enu_m)
    target_length_m = _background_pedestrian_target_length_m(mode, index)
    return _long_sidewalk_ground_flow_route(
        list(anchor.position_enu_m),
        anchor.sample,
        side_sign=side_sign,
        offset_from_curb_m=float(anchor.offset_from_curb_m),
        min_path_length_m=target_length_m,
        min_xy_span_m=BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M,
        allow_green=True,
        context="semantic contract pedestrian route",
    )


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
    sample = _find_free_lane_sample(scenario_id, scenes, events, occupied, mode=moving_mode, index=index)
    authoritative_sample = _sumo_lane_sample_at_s(sample.edge_id, sample.s_m)
    entity_id = f"bg_vehicle_{safe_id(scenario_id)}_{index + 1:02d}"
    asset = BACKGROUND_VEHICLE_ASSETS[index % len(BACKGROUND_VEHICLE_ASSETS)]
    lateral_m = 0.0
    start = _offset_from_lane(authoritative_sample, lateral_m, GROUND_Z_M)
    route = _semantic_vehicle_route(authoritative_sample, moving_mode, index, lateral_m=lateral_m)
    if not route:
        raise RuntimeError(f"{scenario_id}: background vehicle {entity_id} lacks a valid continuous flow route")
    spec, scene = lane_entity(
        entity_id,
        asset,
        "vehicle",
        authoritative_sample.edge_id,
        authoritative_sample.s_m,
        lateral_m,
        start,
        authoritative_sample.yaw_deg,
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
        route_authority="sumo_directed_lane_geometry",
    )
    scene["background_vehicle"] = {
        "policy": "semantic_event_contract_v1",
        "semantic_role": contract.vehicle_role,
        "contract_scenario_id": scenario_id,
        "physical_lane_lateral_m": round(float(lateral_m), 3),
    }
    scene["ground_flow_contract"] = _ground_flow_contract(
        "vehicle",
        BACKGROUND_VEHICLE_FLOW_SPEED_MPS,
        [scene_pos(scene), *route],
        route_source="sumo_net_directed_lane_geometry",
    )
    scene["ground_flow_contract"]["physical_lane_lateral_m"] = round(float(lateral_m), 3)
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
    origin = _contract_center(scenes, events) if SPATIAL_PROVISIONAL_BUILD else _capture_boundary_focus_center(scenes, events)
    mode = _pedestrian_mode_from_role(contract.pedestrian_role)
    entity_id = f"bg_ped_{safe_id(scenario_id)}_{index + 1:02d}"
    target_length_m = _background_pedestrian_target_length_m(mode, index)
    anchor = None
    route: list[list[float]] = []
    checked = 0
    for candidate in _iter_free_sidewalk_anchors(
        origin,
        occupied,
        semantic="semantic_event_background_pedestrian",
        allow_green=True,
        offset_priority_m=BACKGROUND_PEDESTRIAN_OFFSET_PRIORITY_M,
        min_route_length_m=target_length_m,
    ):
        checked += 1
        candidate_route = _semantic_ped_route(candidate, mode, index)
        full_candidate_route = [candidate.position_enu_m, *candidate_route]
        if candidate_route and _pedestrian_vehicle_dynamic_clearance_ok(
            scenario_id,
            scenes,
            events,
            full_candidate_route,
        ):
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
    return category in {"uav", "vehicle", "pedestrian", "facility", "traffic_light", "airspace_constraint", "hazard_zone"}


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


def _semantic_ground_context_scene(scene: dict[str, Any]) -> bool:
    state = dict(scene.get("initial_state") or {})
    role = str(state.get("role") or "")
    if role in {"semantic_background_vehicle", "semantic_background_pedestrian"}:
        return True
    return bool(scene.get("background_vehicle") or scene.get("background_pedestrian"))


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


def _capture_boundary_focus_points(
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    include_ground_context: bool = True,
) -> list[list[float]]:
    boundary_scenes = [scene for scene in scenes if _capture_boundary_entity(scene)]
    points: list[list[float]] = []

    def add_scene_route_points(scene: dict[str, Any]) -> None:
        position = scene_pos(scene)
        if position:
            points.append(q(float(position[0]), float(position[1]), float(position[2] if len(position) > 2 else GROUND_Z_M)))
        for waypoint in scene.get("route_waypoints_enu_m") or []:
            if isinstance(waypoint, list) and len(waypoint) >= 2:
                points.append(q(float(waypoint[0]), float(waypoint[1]), float(waypoint[2] if len(waypoint) > 2 else GROUND_Z_M)))

    for scene in boundary_scenes:
        polygon = _scene_polygon_xy(scene)
        if polygon:
            z = float((scene_pos(scene) or [0.0, 0.0, GROUND_Z_M])[2])
            points.extend(q(point[0], point[1], z) for point in polygon)
        else:
            position = scene_pos(scene)
            if position:
                points.append(position)

    if include_ground_context:
        for scene in scenes:
            if _semantic_ground_context_scene(scene):
                add_scene_route_points(scene)

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
        if str(event_def.get("capture_boundary_scope") or "") == "pre_roll_outside_event_stage":
            continue
        if str(event_def.get("event_id") or "").startswith("lifecycle_"):
            continue
        for target in event_def.get("log_target_ids") or []:
            add_scene_point(str(target))
        for action_def in event_def.get("actions") or []:
            params = action_params(action_def)
            entity_id = str(params.get("entity_id") or params.get("ped_id") or "")
            add_scene_point(entity_id)
            scene = scenes_by_id.get(entity_id)
            if scene is not None and str((scene.get("initial_state") or {}).get("role") or "") == "U_inspect":
                continue
            for waypoint in params.get("waypoints_enu_m") or []:
                if isinstance(waypoint, list) and len(waypoint) >= 3:
                    points.append(q(float(waypoint[0]), float(waypoint[1]), float(waypoint[2])))
    return points or _contract_anchor_points(scenes, events)


def _capture_boundary_focus_center(scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[float]:
    if ACTIVE_ROI_CONTRACT is not None:
        return list(ACTIVE_ROI_CONTRACT.center_enu_m)
    points = _capture_boundary_focus_points(scenes, events)
    return [
        round(sum(point[0] for point in points) / len(points), 3),
        round(sum(point[1] for point in points) / len(points), 3),
        round(sum(point[2] for point in points) / len(points), 3),
    ]


def _sample_vehicle_edge_segment(
    edge_id: str,
    start_s_m: float,
    end_s_m: float,
    lateral_m: float,
) -> list[list[float]]:
    if end_s_m + 1e-6 < start_s_m:
        raise RuntimeError(
            f"directed vehicle segment reverses {edge_id}: {start_s_m:.3f}>{end_s_m:.3f}"
        )
    segment_count = max(
        1,
        int(
            math.ceil(
                (end_s_m - start_s_m) / EVENT_ACTOR_ENTRY_ROUTE_STEP_M
            )
        ),
    )
    return [
        _sumo_vehicle_point_at_s(
            edge_id,
            start_s_m + (end_s_m - start_s_m) * index / segment_count,
            lateral_m,
        )
        for index in range(segment_count + 1)
    ]


@lru_cache(maxsize=None)
def _vehicle_connection_shape(
    source_edge_id: str,
    target_edge_id: str,
) -> tuple[tuple[float, float], ...]:
    shape = SUMO_GROUND_FLOW.connection_shape_xy(source_edge_id, target_edge_id)
    source_end = SUMO_GROUND_FLOW.edges[source_edge_id].shape_xy[-1]
    if dist_xy(list(shape[-1]), list(source_end)) < dist_xy(
        list(shape[0]),
        list(source_end),
    ):
        return tuple(reversed(shape))
    return shape


@lru_cache(maxsize=None)
def _vehicle_connection_length_m(
    source_edge_id: str,
    target_edge_id: str,
) -> float:
    return float(
        LineString(_vehicle_connection_shape(source_edge_id, target_edge_id)).length
    )


def _sample_vehicle_connection(
    source_edge_id: str,
    target_edge_id: str,
) -> list[list[float]]:
    line = LineString(_vehicle_connection_shape(source_edge_id, target_edge_id))
    segment_count = max(
        1,
        int(math.ceil(float(line.length) / EVENT_ACTOR_ENTRY_ROUTE_STEP_M)),
    )
    return [
        q(*line.interpolate(float(line.length) * index / segment_count).coords[0], GROUND_Z_M)
        for index in range(segment_count + 1)
    ]


def _nearest_physical_vehicle_lateral(
    edge_id: str,
    s_m: float,
    desired_lateral_m: float,
) -> float:
    sample = _sumo_lane_sample_at_s(edge_id, s_m)
    return min(
        _physical_vehicle_lane_offsets(sample),
        key=lambda offset_m: abs(offset_m - desired_lateral_m),
    )


@lru_cache(maxsize=1)
def _vehicle_predecessors() -> dict[str, tuple[str, ...]]:
    predecessors: dict[str, list[str]] = {}
    for source_edge_id, target_edge_ids in SUMO_GROUND_FLOW.adjacency.items():
        source_edge = SUMO_GROUND_FLOW.edges.get(source_edge_id)
        if source_edge is None or not SUMO_GROUND_FLOW._edge_allows(
            source_edge, "vehicle"
        ):
            continue
        for target_edge_id in target_edge_ids:
            target_edge = SUMO_GROUND_FLOW.edges.get(target_edge_id)
            if target_edge is None or not SUMO_GROUND_FLOW._edge_allows(
                target_edge, "vehicle"
            ):
                continue
            predecessors.setdefault(target_edge_id, []).append(source_edge_id)
    return {
        edge_id: tuple(sorted(source_ids))
        for edge_id, source_ids in predecessors.items()
    }


def _vehicle_entry_route_outside_capture_boundary(
    scene: dict[str, Any],
    tactical_position: list[float],
    boundary_polygon: Polygon,
) -> tuple[str, float, float, list[list[float]]]:
    placement = dict(scene.get("placement") or {})
    target_edge_id = str(placement.get("edge_id") or "")
    if target_edge_id not in SUMO_GROUND_FLOW.edges:
        raise RuntimeError(
            f"{scene.get('entity_id')}: staged vehicle lacks a directed SUMO edge"
        )
    _distance_m, target_s_m, _x_m, _y_m, _yaw_deg = (
        SUMO_GROUND_FLOW.project_point_to_edge(target_edge_id, tactical_position)
    )
    lateral_m = float(placement.get("resolved_lateral_from_center_m") or 0.0)
    predecessors = _vehicle_predecessors()
    queue: list[tuple[float, str, tuple[str, ...]]] = [
        (0.0, target_edge_id, (target_edge_id,))
    ]
    best_after_edge_m = {target_edge_id: 0.0}
    best_entry: tuple[float, tuple[str, ...], float, float] | None = None
    while queue:
        after_edge_m, first_edge_id, path = heapq.heappop(queue)
        if after_edge_m != best_after_edge_m.get(first_edge_id):
            continue
        if best_entry is not None and after_edge_m >= best_entry[0]:
            break
        first_edge_end_s_m = (
            target_s_m if len(path) == 1 else _sumo_edge_length_m(first_edge_id)
        )
        sample_count = max(
            1,
            int(math.ceil(first_edge_end_s_m / EVENT_ACTOR_ENTRY_ROUTE_STEP_M)),
        )
        for sample_index in range(sample_count + 1):
            start_s_m = max(
                0.0,
                first_edge_end_s_m
                - sample_index * EVENT_ACTOR_ENTRY_ROUTE_STEP_M,
            )
            start_lateral_m = _nearest_physical_vehicle_lateral(
                first_edge_id,
                start_s_m,
                lateral_m,
            )
            start_position = _sumo_vehicle_point_at_s(
                first_edge_id,
                start_s_m,
                start_lateral_m,
            )
            start_point = Point(float(start_position[0]), float(start_position[1]))
            if boundary_polygon.covers(start_point):
                continue
            if boundary_polygon.distance(start_point) < EVENT_ACTOR_ENTRY_BOUNDARY_CLEARANCE_M:
                continue
            candidate_distance_m = (
                after_edge_m + first_edge_end_s_m - start_s_m
            )
            candidate = (
                candidate_distance_m,
                path,
                start_s_m,
                start_lateral_m,
            )
            if best_entry is None or candidate < best_entry:
                best_entry = candidate
            break
        if len(path) >= EVENT_ACTOR_ENTRY_MAX_REVERSE_EDGES:
            continue
        for predecessor_id in predecessors.get(first_edge_id, ()):
            if predecessor_id in path:
                continue
            cost_from_predecessor_end_m = (
                after_edge_m
                + first_edge_end_s_m
                + _vehicle_connection_length_m(predecessor_id, first_edge_id)
            )
            if cost_from_predecessor_end_m >= best_after_edge_m.get(
                predecessor_id,
                float("inf"),
            ):
                continue
            best_after_edge_m[predecessor_id] = cost_from_predecessor_end_m
            heapq.heappush(
                queue,
                (
                    cost_from_predecessor_end_m,
                    predecessor_id,
                    (predecessor_id, *path),
                ),
            )
    if best_entry is None:
        raise RuntimeError(
            f"{scene.get('entity_id')}: no directed lane entry begins outside the capture boundary"
        )
    _distance_m, path, start_s_m, start_lateral_m = best_entry
    start_edge_id = path[0]
    route: list[list[float]] = []
    for edge_index, edge_id in enumerate(path):
        segment_start_s_m = start_s_m if edge_index == 0 else 0.0
        segment_end_s_m = (
            target_s_m
            if edge_index == len(path) - 1
            else _sumo_edge_length_m(edge_id)
        )
        segment_lateral_m = _nearest_physical_vehicle_lateral(
            edge_id,
            segment_start_s_m,
            lateral_m,
        )
        segment = _sample_vehicle_edge_segment(
            edge_id,
            segment_start_s_m,
            segment_end_s_m,
            segment_lateral_m,
        )
        route.extend(segment if not route else segment[1:])
        if edge_index + 1 < len(path):
            connector = _sample_vehicle_connection(edge_id, path[edge_index + 1])
            route.extend(connector[1:])
    route = _dedupe_route_points(route)
    if dist_xy(route[-1], tactical_position) > 0.75:
        route.append(list(tactical_position))
    return start_edge_id, start_s_m, start_lateral_m, route


def _minimum_quantized_entry_speed_mps(
    route: list[list[float]],
    requested_speed_mps: float,
    arrival_tick: int,
) -> tuple[float, int]:
    if not (
        EVENT_ACTOR_ENTRY_SPEED_MIN_MPS
        <= requested_speed_mps
        <= EVENT_ACTOR_ENTRY_SPEED_MAX_MPS
    ):
        raise RuntimeError(
            f"requested event-actor entry speed {requested_speed_mps:.6f}m/s "
            f"is outside [{EVENT_ACTOR_ENTRY_SPEED_MIN_MPS:.1f}, "
            f"{EVENT_ACTOR_ENTRY_SPEED_MAX_MPS:.1f}]m/s"
        )
    requested_ticks = quantized_path_duration_ticks(route, requested_speed_mps)
    if requested_ticks <= arrival_tick:
        return requested_speed_mps, requested_ticks
    ceiling_ticks = quantized_path_duration_ticks(
        route,
        EVENT_ACTOR_ENTRY_SPEED_MAX_MPS,
    )
    if ceiling_ticks > arrival_tick:
        raise RuntimeError(
            f"entry route requires {ceiling_ticks} ticks before arrival "
            f"{arrival_tick} at the {EVENT_ACTOR_ENTRY_SPEED_MAX_MPS:.1f}m/s ceiling"
        )
    lower_mps = requested_speed_mps
    upper_mps = EVENT_ACTOR_ENTRY_SPEED_MAX_MPS
    for _ in range(64):
        midpoint_mps = 0.5 * (lower_mps + upper_mps)
        if quantized_path_duration_ticks(route, midpoint_mps) <= arrival_tick:
            upper_mps = midpoint_mps
        else:
            lower_mps = midpoint_mps
    speed_mps = math.ceil(upper_mps * 1_000_000.0) / 1_000_000.0
    travel_ticks = quantized_path_duration_ticks(route, speed_mps)
    if travel_ticks > arrival_tick or speed_mps > EVENT_ACTOR_ENTRY_SPEED_MAX_MPS:
        raise RuntimeError(
            "six-decimal event-actor speed cannot satisfy the quantized arrival contract"
        )
    return speed_mps, travel_ticks


def relocate_event_actor_entries_outside_capture_boundary(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    capture_boundary: dict[str, Any],
) -> None:
    polygon = Polygon(
        [
            (float(point[0]), float(point[1]))
            for point in capture_boundary.get("polygon_enu_m") or []
        ]
    )
    if not polygon.is_valid or polygon.is_empty:
        raise RuntimeError(f"{scenario_id}: capture boundary polygon is invalid")
    events_by_id = {
        str(event_def.get("event_id") or ""): event_def for event_def in events
    }
    specs_by_id = {str(spec.get("entity_id") or ""): spec for spec in specs}
    for scene in scenes:
        contract = dict(scene.get("event_actor_motion_contract") or {})
        if contract.get("policy") != "offstage_then_continuous_lane_entry":
            continue
        entity_id = str(scene["entity_id"])
        entry_event_id = str(contract["entry_event_id"])
        entry_event = events_by_id[entry_event_id]
        move_action = next(
            action_def
            for action_def in entry_event["actions"]
            if action_params(action_def).get("type") == "move_entity"
            and str(action_params(action_def).get("entity_id") or "") == entity_id
        )
        tactical_position = list(action_params(move_action)["waypoints_enu_m"][-1])
        start_edge_id, start_s_m, start_lateral_m, route = (
            _vehicle_entry_route_outside_capture_boundary(
                scene,
                tactical_position,
                polygon,
            )
        )
        route_length_m = path_length_m(route)
        max_segment_m = max(
            dist_xy(start, end)
            for start, end in zip(route, route[1:])
        )
        if max_segment_m > EVENT_ACTOR_ENTRY_MAX_SEGMENT_M:
            raise RuntimeError(
                f"{scenario_id}:{entity_id}: directed entry contains a "
                f"{max_segment_m:.3f}m waypoint gap"
            )
        if not polygon.covers(Point(route[-1][0], route[-1][1])):
            raise RuntimeError(
                f"{scenario_id}:{entity_id}: semantic arrival lies outside the ROI contract"
            )
        arrival_tick = int(contract["arrival_tick"])
        requested_speed_mps = float(contract["entry_speed_mps"])
        speed_mps, travel_ticks = _minimum_quantized_entry_speed_mps(
            route,
            requested_speed_mps,
            arrival_tick,
        )
        entry_tick = arrival_tick - travel_ticks
        if entry_tick < 0:
            raise RuntimeError(
                f"{scenario_id}:{entity_id}: outside-boundary entry route requires "
                f"{travel_ticks} ticks before arrival {arrival_tick} "
                f"at {speed_mps:.3f}m/s over {route_length_m:.3f}m"
            )
        start_sample = _sumo_lane_sample_at_s(start_edge_id, start_s_m)
        start_position = list(route[0])
        placement = dict(scene["placement"])
        placement.update(
            {
                "edge_id": start_edge_id,
                "longitudinal_s": round(start_s_m, 3),
                "source_edge_id_hint": start_edge_id,
                "source_longitudinal_s_hint": round(start_s_m, 3),
                "lateral_offset_m": round(start_lateral_m, 3),
                "resolved_lateral_from_center_m": round(start_lateral_m, 3),
                "resolved_position_enu_m": start_position,
                "rotation_deg": rot(start_sample.yaw_deg),
                **_physical_vehicle_lane_audit(
                    start_sample,
                    start_lateral_m,
                ),
            }
        )
        scene["placement"] = placement
        spec = specs_by_id[entity_id]
        spec["initial_pos_enu"] = start_position
        spec["initial_rotation_deg"] = [0.0, 0.0, start_sample.yaw_deg]
        contract.update(
            {
                "entry_tick": entry_tick,
                "route_length_m": round(route_length_m, 6),
                "entry_speed_mps": round(speed_mps, 6),
                "entry_capture_boundary_clearance_m": round(
                    polygon.distance(Point(start_position[0], start_position[1])),
                    6,
                ),
            }
        )
        scene["event_actor_motion_contract"] = contract
        entry_event["trigger"] = tick(entry_tick)
        write_action_params(
            move_action,
            {
                "waypoints_enu_m": route,
                "velocity_mps": speed_mps,
            },
        )
        spawn_action = next(
            action_def
            for action_def in entry_event["actions"]
            if action_params(action_def).get("type") == "spawn_entity"
            and str(action_params(action_def).get("entity_id") or "") == entity_id
        )
        write_action_params(
            spawn_action,
            {
                "position_enu_m": start_position,
                "rotation_deg": rot(start_sample.yaw_deg),
            },
        )


def _capture_boundary_assignment_center(scenes: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[float]:
    points = _capture_boundary_focus_points(scenes, events, include_ground_context=False)
    return [
        round(sum(point[0] for point in points) / len(points), 3),
        round(sum(point[1] for point in points) / len(points), 3),
        round(sum(point[2] for point in points) / len(points), 3),
    ]


def _capture_boundary_from_points(
    scenario_id: str,
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    apply_roi_contract: bool = True,
    include_ground_context: bool = True,
) -> dict[str, Any]:
    boundary_scenes = [scene for scene in scenes if _capture_boundary_entity(scene)]
    if boundary_scenes:
        primary = boundary_scenes[0]
        center = scene_pos(primary) or _contract_center(scenes, events)
        polygon = _scene_polygon_xy(primary)
        generated = {
            "boundary_id": str(primary.get("entity_id") or f"capture_boundary_{safe_id(scenario_id)}"),
            "source": "scene_entity",
            "source_entity_id": str(primary.get("entity_id") or ""),
            "center_enu_m": q(center[0], center[1], center[2] if len(center) > 2 else 0.0),
            "polygon_enu_m": polygon,
        }
    else:
        points = _capture_boundary_focus_points(
            scenes,
            events,
            include_ground_context=include_ground_context,
        )
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        margin = 16.0
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        center = q((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, GROUND_Z_M)
        source_entity_id = _capture_boundary_anchor_entity_id(scenes, events)
        generated = {
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
    roi_contract = ACTIVE_ROI_CONTRACT if apply_roi_contract else None
    if roi_contract is None:
        return generated
    if roi_contract.scenario_id != scenario_id:
        raise RuntimeError(
            f"{scenario_id}: active ROI contract belongs to {roi_contract.scenario_id}"
        )
    generated_boundary_id = str(generated.get("boundary_id") or "")
    if generated_boundary_id != roi_contract.capture_boundary_id:
        raise RuntimeError(
            f"{scenario_id}: generated boundary id {generated_boundary_id!r} "
            f"differs from ROI contract id {roi_contract.capture_boundary_id!r}"
        )
    return {
        **generated,
        "boundary_id": roi_contract.capture_boundary_id,
        "center_enu_m": list(roi_contract.center_enu_m),
        "polygon_enu_m": roi_contract.polygon_lists(),
        "geometry_authority": roi_contract.geometry_authority,
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


def _point_segment_distance_xy(point: list[float], a: list[float], b: list[float]) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def _distance_to_polygon_xy(point: list[float], polygon: list[list[float]]) -> float:
    if not polygon:
        return float("inf")
    if _point_in_polygon_xy(point, polygon):
        return 0.0
    return min(
        _point_segment_distance_xy(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _ray_polygon_exit_distance_xy(
    origin: Sequence[float], direction: Sequence[float], polygon: Sequence[Sequence[float]]
) -> float:
    if len(origin) < 2 or len(direction) < 2 or len(polygon) < 3:
        raise ValueError("origin, direction, and polygon are required")
    ox, oy = float(origin[0]), float(origin[1])
    ux, uy = float(direction[0]), float(direction[1])
    direction_norm = math.hypot(ux, uy)
    if direction_norm <= 1e-9:
        raise ValueError("ray direction must be non-zero")
    ux /= direction_norm
    uy /= direction_norm
    intersections: list[float] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        ax, ay = float(start[0]), float(start[1])
        vx = float(end[0]) - ax
        vy = float(end[1]) - ay
        determinant = ux * vy - uy * vx
        if abs(determinant) <= 1e-9:
            continue
        offset_x = ax - ox
        offset_y = ay - oy
        ray_distance = (offset_x * vy - offset_y * vx) / determinant
        segment_fraction = (offset_x * uy - offset_y * ux) / determinant
        if ray_distance > 1e-9 and -1e-9 <= segment_fraction <= 1.0 + 1e-9:
            intersections.append(ray_distance)
    if not intersections:
        raise ValueError("ray does not exit the capture-boundary polygon")
    return min(intersections)


def _route_distance_to_polygon_xy(
    route: Sequence[Sequence[float]], polygon: Sequence[Sequence[float]]
) -> float:
    if not route or len(polygon) < 3:
        raise ValueError("route and polygon are required for route-clearance validation")
    polygon_xy = [(float(point[0]), float(point[1])) for point in polygon]
    route_xy: list[tuple[float, float]] = []
    for point in route:
        xy = (float(point[0]), float(point[1]))
        if not route_xy or xy != route_xy[-1]:
            route_xy.append(xy)
    if not all(math.isfinite(value) for point in [*polygon_xy, *route_xy] for value in point):
        raise ValueError("route-clearance geometry contains a non-finite coordinate")
    polygon_geometry = Polygon(polygon_xy)
    if not polygon_geometry.is_valid or polygon_geometry.area <= 0.0:
        raise ValueError("route-clearance polygon must be valid and non-degenerate")
    clearance_m = (
        Point(route_xy[0]).distance(polygon_geometry)
        if len(route_xy) == 1
        else LineString(route_xy).distance(polygon_geometry)
    )
    if not math.isfinite(clearance_m):
        raise ValueError("route-clearance geometry produced a non-finite distance")
    return clearance_m


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


def _ensure_contract_pedestrians_reach_capture_context(
    scenario_id: str,
    scenes: list[dict[str, Any]],
    capture_boundary: dict[str, Any],
    contract: EpisodeContract,
) -> None:
    if contract.pedestrian <= 0:
        return
    polygon = [
        [float(point[0]), float(point[1])]
        for point in capture_boundary.get("polygon_enu_m") or []
        if isinstance(point, list) and len(point) >= 2
    ]
    if not polygon:
        raise RuntimeError(f"{scenario_id}: capture boundary polygon missing before pedestrian visibility check")
    far: list[str] = []
    for scene in scenes:
        asset = str(scene.get("logical_asset_id") or "")
        category = str(scene.get("category") or "")
        if not (asset.startswith("pedestrian.") or category == "pedestrian"):
            continue
        points = _scene_route_points(scene)
        if not points:
            position = scene_pos(scene)
            points = [position] if position else []
        if not points:
            far.append(f"{scene.get('entity_id')}: no motion points")
            continue
        min_distance_m = min(_distance_to_polygon_xy(point, polygon) for point in points)
        if min_distance_m > CONTRACT_RUNTIME_VISIBLE_PADDING_M:
            far.append(f"{scene.get('entity_id')}: {min_distance_m:.1f}m")
    if far:
        raise RuntimeError(
            f"{scenario_id}: contract pedestrians do not enter the expanded capture boundary "
            f"({CONTRACT_RUNTIME_VISIBLE_PADDING_M:.1f}m padding): {far[:8]}"
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
    for scene in scenes:
        if not str(scene.get("logical_asset_id") or "").startswith("uav."):
            continue
        initial_state = dict(scene.get("initial_state") or {})
        if str(initial_state.get("role") or "") == "U_inspect":
            continue
        corridor_role = str(
            scene.get("uav_corridor_role")
            or initial_state.get("uav_corridor_role")
            or ""
        )
        if corridor_role == "observer":
            current = scene_pos(scene) or crossing_waypoint
            dx = float(current[0]) - float(center[0])
            dy = float(current[1]) - float(center[1])
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                dx, dy, length = 1.0, 0.0, 1.0
            ux = dx / length
            uy = dy / length
            edge_scale = _ray_polygon_exit_distance_xy(center, [ux, uy], polygon)
            altitude_m = float(current[2])
            approach = q(
                float(center[0]) + ux * (edge_scale + 18.0),
                float(center[1]) + uy * (edge_scale + 18.0),
                altitude_m,
            )
            center_at_altitude = q(float(center[0]), float(center[1]), altitude_m)
            observer_route = [approach, center_at_altitude]
            entity_id = str(scene.get("entity_id") or "")
            update_world_entity_position(spec_by_id.get(entity_id, {}), scene, approach)
            scene["route_waypoints_enu_m"] = observer_route
            scene["observer_lifecycle"] = {
                "presence": "episode_full_duration",
                "terminal_behavior": "airborne_hold_inside_runtime_boundary",
                "maximum_terminal_boundary_distance_m": 18.0,
                "capture_boundary_id": capture_boundary.get("boundary_id"),
            }
            scene["boundary_crossing_injected"] = {
                "policy": "semantic_event_contract_v1",
                "capture_boundary_id": capture_boundary.get("boundary_id"),
                "reason": "observer_crosses_boundary_then_holds_inside_runtime_crop",
            }
            spec = spec_by_id.get(entity_id)
            if spec is not None:
                spec["movement_waypoints"] = observer_route
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


INSPECT_OBSERVATION_SEGMENT_HALF_LENGTH_M = 0.5
INSPECT_OBSERVATION_REACH_GUARD_M = 0.05


def _polygon_components(geometry: Any) -> list[Any]:
    components: list[Any] = []
    pending = [geometry]
    while pending:
        candidate = pending.pop()
        if candidate.is_empty:
            continue
        if candidate.geom_type == "Polygon":
            components.append(candidate)
            continue
        pending.extend(list(getattr(candidate, "geoms", ())))
    return sorted(components, key=lambda item: (-item.area, item.bounds))


def _inspect_observation_center_component(
    scenario_id: str,
    samples: list[list[float]],
    altitude_m: float,
) -> tuple[Any, float, float, float]:
    half_width_m, half_height_m = camera_footprint_half_extents_m(
        DEFAULT_UAV_SENSOR_PROFILE,
        altitude_m,
    )
    segment_half_length_m = INSPECT_OBSERVATION_SEGMENT_HALF_LENGTH_M
    observation_radius_m = math.hypot(
        half_width_m,
        half_height_m + segment_half_length_m,
    )
    maximum_center_distance_m = (
        observation_radius_m - INSPECT_OBSERVATION_REACH_GUARD_M
    )
    candidates: list[Any] = []
    for free_component in UAV_CORRIDORS.buildings.fixed_altitude_free_components(
        altitude_m
    ):
        for center_component in _polygon_components(free_component.buffer(-0.51)):
            if all(
                center_component.distance(
                    Point(float(sample[0]), float(sample[1]))
                )
                <= maximum_center_distance_m
                for sample in samples
            ):
                candidates.append(center_component)
    if not candidates:
        raise RuntimeError(
            f"{scenario_id}: ROI cannot be observed from one connected "
            f"UE-building-clear component at {altitude_m}m"
        )
    component = sorted(candidates, key=lambda item: (-item.area, item.bounds))[0]
    return component, half_width_m, half_height_m, maximum_center_distance_m


def _inspect_observation_segment(
    scenario_id: str,
    sample_index: int,
    sample: list[float],
    center_component: Any,
    altitude_m: float,
    half_width_m: float,
    half_height_m: float,
    maximum_center_distance_m: float,
) -> tuple[list[float], list[float]]:
    observation_region = center_component.intersection(
        Point(float(sample[0]), float(sample[1])).buffer(
            maximum_center_distance_m,
            resolution=64,
        )
    )
    if observation_region.is_empty:
        raise RuntimeError(
            f"{scenario_id}: no observation-center region for ROI sample {sample_index}"
        )
    sample_point = Point(float(sample[0]), float(sample[1]))
    center_candidates = [
        component.representative_point()
        for component in _polygon_components(observation_region)
    ]
    nearest_center = nearest_points(sample_point, center_component)[1]
    center_candidates.append(nearest_center)
    for reference in list(center_candidates):
        dx = reference.x - nearest_center.x
        dy = reference.y - nearest_center.y
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        for offset_m in (0.25, 0.5, 1.0, 2.0):
            fraction = min(1.0, offset_m / length)
            candidate = Point(
                nearest_center.x + dx * fraction,
                nearest_center.y + dy * fraction,
            )
            if center_component.covers(candidate):
                center_candidates.append(candidate)
    center_candidates.sort(
        key=lambda point: (
            -min(2.0, center_component.boundary.distance(point)),
            point.distance(sample_point),
            point.x,
            point.y,
        )
    )
    segment_half_length_m = INSPECT_OBSERVATION_SEGMENT_HALF_LENGTH_M
    def segment_for_center(center: Any) -> tuple[list[float], list[float]] | None:
        target_angle = math.atan2(
            float(sample[1]) - center.y,
            float(sample[0]) - center.x,
        )
        corner_angle = math.atan2(
            half_width_m,
            half_height_m + segment_half_length_m,
        )
        analytic_angles = (
            target_angle - math.pi * 0.5,
            target_angle + math.pi * 0.5,
            target_angle - corner_angle,
            target_angle + corner_angle,
            target_angle + math.pi - corner_angle,
            target_angle + math.pi + corner_angle,
        )
        scan_angles = tuple(math.tau * index / 360.0 for index in range(360))
        for angle in (*analytic_angles, *scan_angles):
            point_a = q(
                center.x - segment_half_length_m * math.cos(angle),
                center.y - segment_half_length_m * math.sin(angle),
                altitude_m,
            )
            point_b = q(
                center.x + segment_half_length_m * math.cos(angle),
                center.y + segment_half_length_m * math.sin(angle),
                altitude_m,
            )
            if not UAV_CORRIDORS.buildings.fixed_altitude_free_space(
                altitude_m
            ).covers(
                LineString(
                    [
                        (float(point_a[0]), float(point_a[1])),
                        (float(point_b[0]), float(point_b[1])),
                    ]
                )
            ):
                continue
            if not UAV_CORRIDORS.buildings.air_segment_clear(point_a, point_b):
                continue
            if route_observes_samples(
                [point_a, point_b],
                [sample],
                DEFAULT_UAV_SENSOR_PROFILE,
                altitude_m,
            ):
                return point_a, point_b
        return None

    for center in center_candidates:
        segment = segment_for_center(center)
        if segment is not None:
            return segment

    radius_step_m = 0.5
    radius_count = int(math.floor(maximum_center_distance_m / radius_step_m))
    for radius_index in range(radius_count + 1):
        radius_m = radius_index * radius_step_m
        for angle_index in range(64):
            angle = math.tau * angle_index / 64.0
            center = Point(
                float(sample[0]) + radius_m * math.cos(angle),
                float(sample[1]) + radius_m * math.sin(angle),
            )
            if not center_component.covers(center):
                continue
            segment = segment_for_center(center)
            if segment is not None:
                return segment
    raise RuntimeError(
        f"{scenario_id}: no exact oriented camera footprint covers ROI sample "
        f"{sample_index} at {altitude_m}m"
    )


def _frozen_inspect_route(scenario_id: str) -> list[list[float]] | None:
    """Return the inspect route already frozen for this scenario, if one exists."""
    for scene_path in sorted(SCENARIOS_ROOT.rglob("scene_setup.json")):
        try:
            scene_setup = json.loads(scene_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        existing_id = str(scene_setup.get("scenario_id") or scene_path.parent.name)
        if existing_id != scenario_id and scene_path.parent.name != scenario_id:
            continue
        for entity in scene_setup.get("entities") or []:
            contract = dict(entity.get("contract_inspect_uav") or {})
            route = contract.get("loop_route_enu_m") or []
            points = [
                [float(point[0]), float(point[1]), float(point[2])]
                for point in route
                if isinstance(point, list) and len(point) >= 3
            ]
            if len(points) >= 2:
                return points
        return None
    return None


def _frozen_inspect_route_is_current(
    route: list[list[float]],
    polygon: list[list[float]],
    altitude_m: float,
    inspect_contract: dict[str, Any],
) -> bool:
    """A frozen inspect route may be reused only while it still satisfies the live
    capture-boundary contract.  If the boundary moved, the route must be recomputed."""
    if any(abs(float(point[2]) - altitude_m) > 1e-6 for point in route):
        return False
    if not UAV_CORRIDORS.buildings.route_clear_at_altitude(route, altitude_m):
        return False
    return route_observes_boundary(route, polygon, inspect_contract)


def _find_inspect_route(
    scenario_id: str,
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    contract: EpisodeContract,
) -> tuple[list[float], list[list[float]]]:
    boundary = _capture_boundary_from_points(scenario_id, scenes, events)
    polygon = [
        [float(point[0]), float(point[1])]
        for point in boundary.get("polygon_enu_m") or []
        if isinstance(point, list) and len(point) >= 2
    ]
    altitude_m = float(contract.inspect_altitude_m)
    boundary_samples = boundary_observation_samples(polygon)
    if PARTIAL_REGENERATION and REUSE_FROZEN_LAYOUT and not SPATIAL_PROVISIONAL_BUILD:
        # A partial regeneration must not silently re-derive the inspect loop: the
        # search picks among equal-area components, so an unrelated code or data change
        # can flip it to a different loop.  Reuse the frozen route while it still
        # satisfies the current boundary contract, and recompute when it no longer does.
        frozen_route = _frozen_inspect_route(scenario_id)
        if frozen_route is not None:
            frozen_contract = {
                "inspect_altitude_m": altitude_m,
                "sensor_profile": DEFAULT_UAV_SENSOR_PROFILE,
            }
            if _frozen_inspect_route_is_current(
                frozen_route, polygon, altitude_m, frozen_contract
            ):
                return frozen_route[0], frozen_route[1:]
    if SPATIAL_PROVISIONAL_BUILD:
        # The provisional bundle is an in-memory translation probe only.  Its
        # inspect entity is excluded from the assignment-center calculation, so
        # choose a probe target from a physically valid free-space component.
        # The persisted bundle below always uses the complete boundary sample
        # set and the strict sensor-footprint contract.
        (
            center_component,
            half_width_m,
            half_height_m,
            maximum_center_distance_m,
        ) = _inspect_observation_center_component(
            scenario_id,
            [],
            altitude_m,
        )
        probe = center_component.representative_point()
        samples = [q(float(probe.x), float(probe.y), altitude_m)]
    else:
        samples = boundary_samples
        (
            center_component,
            half_width_m,
            half_height_m,
            maximum_center_distance_m,
        ) = _inspect_observation_center_component(
            scenario_id,
            samples,
            altitude_m,
        )
    observation_segments = [
        _inspect_observation_segment(
            scenario_id,
            sample_index,
            sample,
            center_component,
            altitude_m,
            half_width_m,
            half_height_m,
            maximum_center_distance_m,
        )
        for sample_index, sample in enumerate(samples)
    ]
    def observation_center(
        segment: tuple[list[float], list[float]],
    ) -> list[float]:
        return q(
            (segment[0][0] + segment[1][0]) * 0.5,
            (segment[0][1] + segment[1][1]) * 0.5,
            altitude_m,
        )

    first_a, first_b = observation_segments[0]
    first_center = observation_center(observation_segments[0])
    route = [first_center, first_a, first_b, first_center]
    for point_a, point_b in observation_segments[1:]:
        center = observation_center((point_a, point_b))
        leg = UAV_CORRIDORS.buildings.air_path_at_altitude(
            route[-1],
            center,
            altitude_m,
        )
        if not leg:
            raise RuntimeError(
                f"{scenario_id}: disconnected UE-building-clear inspect observations"
            )
        route.extend(leg)
        route.append(point_a)
        route.append(point_b)
        route.append(center)
    closing_leg = UAV_CORRIDORS.buildings.air_path_at_altitude(
        route[-1],
        route[0],
        altitude_m,
    )
    if not closing_leg:
        raise RuntimeError(
            f"{scenario_id}: inspect observation route cannot close in UE free space"
        )
    route.extend(closing_leg)
    if _route_length_m(route) <= 1e-6:
        raise RuntimeError(f"{scenario_id}: inspect route has zero length")
    while _route_length_m(route) < 80.0:
        route.extend([list(point) for point in route[1:]])
    inspect_contract = {
        "inspect_altitude_m": altitude_m,
        "sensor_profile": DEFAULT_UAV_SENSOR_PROFILE,
    }
    if not UAV_CORRIDORS.buildings.route_clear_at_altitude(route, altitude_m):
        raise RuntimeError(
            f"{scenario_id}: inspect route violates UE building clearance"
        )
    observes_required_geometry = (
        route_observes_samples(
            route,
            samples,
            DEFAULT_UAV_SENSOR_PROFILE,
            altitude_m,
        )
        if SPATIAL_PROVISIONAL_BUILD
        else route_observes_boundary(route, polygon, inspect_contract)
    )
    if not observes_required_geometry:
        raise RuntimeError(
            f"{scenario_id}: inspect route fails the exact sensor footprint contract"
        )
    return route[0], route[1:]


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
        "sensor_fov_source": "Config/LowAltitude/uav_sensor_profile.json#hfov_deg",
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
    if scenario_id.startswith("L4-10_") and index == 0:
        desired_pos = q(center[0] - 80.0, center[1] - 80.0, GROUND_Z_M)
    if scenario_id == "L6-1_v1" and index == 2:
        # The nominal third facility falls on a junction connector. Move its
        # road-side anchor just beyond the junction before applying curb offset.
        road_sample = min(
            (sample for sample in LANES.samples if not sample.edge_id.startswith(":")),
            key=lambda sample: (sample.x_m - desired_pos[0]) ** 2 + (sample.y_m - desired_pos[1]) ** 2,
        )
        min_s, max_s = LANES.edge_s_bounds(road_sample.edge_id)
        forward_s = min(max_s, road_sample.s_m + 2.0)
        backward_s = max(min_s, road_sample.s_m - 2.0)
        target_s = forward_s if forward_s - road_sample.s_m >= road_sample.s_m - backward_s else backward_s
        shifted_sample = LANES.resolve_edge_s(road_sample.edge_id, target_s)
        desired_pos = q(shifted_sample.x_m, shifted_sample.y_m, GROUND_Z_M)
    facility_cycle = (
        ("facility.landing_pad.visible.v1", "facility", "available"),
        ("facility.charger.cityops.v1", "facility", "available"),
        ("facility.radio.base_tower.v1", "facility", "online"),
        ("facility.barrier.basic", "facility", "available"),
    )
    asset, _category, mode = facility_cycle[index % len(facility_cycle)]
    entity_id = f"contract_facility_{safe_id(scenario_id)}_{index + 1:02d}"
    state = semantic_state(
        task_id=f"{scenario_id}.facility_context.{index + 1:02d}",
        role="semantic_facility",
        state_sequence=[mode],
        semantic_role="visible facility context required by low-altitude event-chain contract",
        mode=mode,
    )
    spec, scene = service_facility_entity(
        entity_id,
        asset,
        desired_pos,
        context=f"{scenario_id} contract facility {index + 1:02d}",
        yaw=float((index * 45) % 360),
        mode=mode,
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
    parameters["deterministic_sumo_traffic_template"] = deterministic_traffic_template_payload(
        deterministic_traffic_template_for(scenario_id)
    )
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
    ensure_background_ground_flow_routes(scenario_id, specs, scenes, events)
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
        payload = role_payload[kind]
        candidate_pool = sorted(non_event_candidates or candidates, key=lambda item: str(item.get("entity_id") or ""))
        selected = candidate_pool[0]
        selected_route: list[list[float]] = []
        selected_anchor: PlannedSidewalkAnchor | None = None
        if kind == "pedestrian":
            mode = str(payload["state_sequence"][0])
            for candidate in candidate_pool:
                candidate_index = scenes.index(candidate)
                route = _background_pedestrian_route_for_scene(
                    candidate,
                    candidate_index,
                    mode=mode,
                    scenario_id=scenario_id,
                    scenes=scenes,
                    events=events,
                )
                if route:
                    selected = candidate
                    selected_route = route
                    break
            if not selected_route:
                for candidate in candidate_pool:
                    candidate_index = scenes.index(candidate)
                    anchor, route = _relocate_background_pedestrian_route(
                        candidate,
                        candidate_index,
                        mode=mode,
                        scenario_id=scenario_id,
                        scenes=scenes,
                        events=events,
                    )
                    if anchor is not None and route:
                        selected = candidate
                        selected_anchor = anchor
                        selected_route = route
                        break
            if not selected_route:
                raise RuntimeError(
                    f"{scenario_id}: contract requires background pedestrian, but no candidate can host "
                    f"a continuous sidewalk flow route"
                )
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
        if kind == "pedestrian":
            if selected_anchor is not None:
                _apply_background_pedestrian_anchor(
                    selected,
                    spec,
                    selected_anchor,
                    selected_route,
                    scenario_id=scenario_id,
                )
            else:
                selected["route_waypoints_enu_m"] = selected_route
                if spec is not None:
                    spec["movement_waypoints"] = selected_route
        if spec is not None:
            spec["visual_state"] = {**(spec.get("visual_state") or {}), **state}


def _pedestrian_flow_seed_route(
    scene: dict[str, Any],
    index: int,
    *,
    scenario_id: str = "",
    scenes: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
    mode_override: str | None = None,
) -> list[list[float]]:
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
    mode = str(mode_override or (scene.get("initial_state") or {}).get("mode") or "walking")
    target_length_m = _background_pedestrian_target_length_m(mode, index)
    route = _long_sidewalk_ground_flow_route(
        start,
        sample,
        side_sign=side_sign,
        offset_from_curb_m=offset_from_curb,
        min_path_length_m=target_length_m,
        min_xy_span_m=BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M,
        allow_green=bool(placement.get("allow_green", True)),
        context="background pedestrian continuous flow route",
    )
    if route and scenes is not None and events is not None and not _pedestrian_vehicle_dynamic_clearance_ok(
        scenario_id,
        scenes,
        events,
        [start, *route],
    ):
        return []
    return route


def _fast_background_pedestrian_fallback_route(scene: dict[str, Any], index: int) -> list[list[float]]:
    start = scene_pos(scene)
    if not start:
        return []
    placement = dict(scene.get("placement") or {})
    edge_id = str(placement.get("lane_edge_id") or "")
    if edge_id and edge_id in LANES.by_edge:
        sample = LANES.resolve_edge_s(edge_id, float(placement.get("longitudinal_s") or 0.0))
    else:
        sample = LANES.nearest_to_xy(float(start[0]), float(start[1]))
    lateral = float(placement.get("resolved_lateral_from_center_m") or 0.0)
    side_sign = 1.0 if lateral >= 0.0 else -1.0
    offset_from_curb = max(float(placement.get("offset_from_curb_m") or SIDEWALK_MIN_OFFSET_FROM_CURB_M), SIDEWALK_MIN_OFFSET_FROM_CURB_M)
    min_s, max_s = LANES.edge_s_bounds(sample.edge_id)
    for s_delta in (16.0, -16.0, 24.0, -24.0, 10.0, -10.0, 32.0, -32.0):
        candidate_sample = LANES.resolve_edge_s(sample.edge_id, max(min_s, min(max_s, sample.s_m + s_delta + float(index % 3) * 1.5)))
        candidate = _sidewalk_point_for_sample(candidate_sample, side_sign, offset_from_curb)
        if _route_xy_span(start, [candidate]) < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M:
            continue
        if _fast_surface_errors(candidate, context="background pedestrian fallback route", allow_green=True):
            continue
        if _fast_surface_segment_errors(start, candidate, context="background pedestrian fallback route", allow_green=True):
            continue
        return [candidate]
    return []


def _background_pedestrian_route_meets_contract(
    start: list[float],
    route: list[list[float]],
    *,
    mode: str,
    index: int,
    scenario_id: str,
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> bool:
    if not route or _route_xy_span(start, route) < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M:
        return False
    full_route = [start, *route]
    target_length_m = _background_pedestrian_target_length_m(mode, index)
    if path_length_m(full_route) + 0.25 < target_length_m:
        return False
    return _pedestrian_vehicle_dynamic_clearance_ok(scenario_id, scenes, events, full_route)


def _background_pedestrian_route_for_scene(
    scene: dict[str, Any],
    index: int,
    *,
    mode: str,
    scenario_id: str,
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[list[float]]:
    start = scene_pos(scene)
    if not start:
        return []
    current_route = _scene_route_points(scene)[1:]
    if _background_pedestrian_route_meets_contract(
        start,
        current_route,
        mode=mode,
        index=index,
        scenario_id=scenario_id,
        scenes=scenes,
        events=events,
    ):
        return current_route
    seeded_route = _pedestrian_flow_seed_route(
        scene,
        index,
        scenario_id=scenario_id,
        scenes=scenes,
        events=events,
        mode_override=mode,
    )
    if _background_pedestrian_route_meets_contract(
        start,
        seeded_route,
        mode=mode,
        index=index,
        scenario_id=scenario_id,
        scenes=scenes,
        events=events,
    ):
        return seeded_route
    return []


def _apply_background_pedestrian_anchor(
    scene: dict[str, Any],
    spec: dict[str, Any] | None,
    anchor: PlannedSidewalkAnchor,
    route: list[list[float]],
    *,
    scenario_id: str,
) -> None:
    old_placement = dict(scene.get("placement") or {})
    placement = {
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
    if "cohort_id" in old_placement:
        placement["cohort_id"] = old_placement["cohort_id"]
    scene["placement_mode"] = "sidewalk_anchor"
    scene["placement"] = placement
    scene["route_waypoints_enu_m"] = route
    if spec is not None:
        spec["initial_pos_enu"] = anchor.position_enu_m
        spec["initial_rotation_deg"] = [0.0, 0.0, anchor.sample.yaw_deg]
        spec["movement_waypoints"] = route


def _relocate_background_pedestrian_route(
    scene: dict[str, Any],
    index: int,
    *,
    mode: str,
    scenario_id: str,
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[PlannedSidewalkAnchor | None, list[list[float]]]:
    origin = scene_pos(scene)
    if not origin:
        return None, []
    target_length_m = _background_pedestrian_target_length_m(mode, index)
    occupied = [
        point
        for other in scenes
        if other is not scene
        for point in [scene_pos(other)]
        if point
    ]
    checked = 0
    for anchor in _iter_free_sidewalk_anchors(
        origin,
        occupied,
        semantic="semantic_event_background_pedestrian",
        allow_green=True,
        offset_priority_m=BACKGROUND_PEDESTRIAN_OFFSET_PRIORITY_M,
        min_route_length_m=target_length_m,
    ):
        checked += 1
        route = _semantic_ped_route(anchor, mode, index)
        if _background_pedestrian_route_meets_contract(
            list(anchor.position_enu_m),
            route,
            mode=mode,
            index=index,
            scenario_id=scenario_id,
            scenes=scenes,
            events=events,
        ):
            return anchor, route
        if checked >= 512:
            break
    return None, []


def ensure_background_ground_flow_routes(
    scenario_id: str,
    specs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
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
                current_route = _sumo_vehicle_route(start, index, span_start=start)
            route = current_route
            if not route or _route_xy_span(start, route) < BACKGROUND_VEHICLE_MIN_ROUTE_SPAN_M:
                raise RuntimeError(f"{scenario_id}: background vehicle {entity_id} lacks continuous flow route")
            state.update({"mode": mode, "state_sequence": [mode, mode]})
            scene["ground_flow_contract"] = _ground_flow_contract(
                "vehicle",
                BACKGROUND_VEHICLE_FLOW_SPEED_MPS,
                [start, *route],
                route_source="sumo_net_directed_lane_geometry",
            )
            placement = dict(scene.get("placement") or {})
            scene["ground_flow_contract"]["physical_lane_lateral_m"] = round(
                float(placement.get("resolved_lateral_from_center_m") or 0.0),
                3,
            )
        else:
            mode = "walking"
            target_length_m = _background_pedestrian_target_length_m(mode, index)
            if (
                not current_route
                or _route_xy_span(start, current_route) < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M
                or path_length_m([start, *current_route]) + 0.25 < target_length_m
            ):
                current_route = _pedestrian_flow_seed_route(
                    scene,
                    index,
                    scenario_id=scenario_id,
                    scenes=scenes,
                    events=events,
                    mode_override=mode,
                )
            if (
                not current_route
                or _route_xy_span(start, current_route) < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M
            ):
                current_route = _fast_background_pedestrian_fallback_route(scene, index)
            route = current_route
            if (
                not route
                or _route_xy_span(start, route) < BACKGROUND_PEDESTRIAN_MIN_ROUTE_SPAN_M
            ):
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
    selected: list[tuple[LaneSample, float, list[list[float]]]] = []
    attempts = 0
    for sample in _background_vehicle_candidate_samples(scenes, events):
        attempts += 1
        if not _lane_has_route_span(sample, BACKGROUND_VEHICLE_MIN_ROUTE_SPAN_M):
            continue
        candidate_lateral_m = _background_vehicle_lateral(sample, len(selected))
        candidate_pos = _offset_from_lane(sample, candidate_lateral_m, GROUND_Z_M)
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
        if any(
            dist_xy(candidate_pos, _offset_from_lane(other, other_lateral, GROUND_Z_M)) < 8.0
            for other, other_lateral, _route in selected
        ):
            continue
        route = _semantic_vehicle_route(sample, "traffic_flow", len(selected), lateral_m=candidate_lateral_m)
        if not route:
            continue
        selected.append((sample, candidate_lateral_m, route))
        if len(selected) >= policy.add_count:
            break
    if len(selected) < policy.add_count:
        raise RuntimeError(
            f"{scenario_id}: unable to place {policy.add_count} background vehicles without overlay/collision conflicts "
            f"(placed {len(selected)}, checked {attempts} lane samples)"
        )

    added_ids: list[str] = []
    for index, (sample, lateral_m, route) in enumerate(selected):
        authoritative_sample = _sumo_lane_sample_at_s(sample.edge_id, sample.s_m)
        lateral_m = 0.0
        entity_id = f"bg_vehicle_{safe_id(scenario_id)}_{index + 1:02d}"
        asset = BACKGROUND_VEHICLE_ASSETS[index % len(BACKGROUND_VEHICLE_ASSETS)]
        spec, scene = lane_entity(
            entity_id,
            asset,
            "vehicle",
            authoritative_sample.edge_id,
            authoritative_sample.s_m,
            lateral_m,
            _offset_from_lane(authoritative_sample, lateral_m, GROUND_Z_M),
            authoritative_sample.yaw_deg,
            "traffic_flow",
            route=route,
            visual_state={
                "mode": "traffic_flow",
                "state_sequence": ["traffic_flow", "traffic_flow"],
                "background_role": "road_context",
            },
            prefer_edge_hint=True,
            route_authority="sumo_directed_lane_geometry",
        )
        scene["background_vehicle"] = {
            "policy": "road_context_non_event_actor_v1",
            "reason": policy.reason,
            "min_clearance_rule": "background_vehicle_clearance",
            "physical_lane_lateral_m": round(float(lateral_m), 3),
        }
        start = _offset_from_lane(sample, lateral_m, GROUND_Z_M)
        scene["ground_flow_contract"] = _ground_flow_contract(
            "vehicle",
            BACKGROUND_VEHICLE_FLOW_SPEED_MPS,
            [scene_pos(scene), *route],
            route_source="sumo_net_directed_lane_geometry",
        )
        scene["ground_flow_contract"]["physical_lane_lateral_m"] = round(float(lateral_m), 3)
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


def align_vehicle_lane_motion(
    spec_entities: list[dict[str, Any]],
    scene_entities: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    spec_by_id = {str(spec.get("entity_id") or ""): spec for spec in spec_entities}
    vehicle_laterals: dict[str, float] = {}
    directed_physical_entities: set[str] = set()
    for index, scene in enumerate(scene_entities):
        entity_id = str(scene.get("entity_id") or "")
        asset_id = str(scene.get("logical_asset_id") or "")
        if not asset_id.startswith("vehicle.") or scene.get("placement_mode") != "lane_anchor":
            continue
        placement = dict(scene.get("placement") or {})
        edge_id = str(placement.get("edge_id") or "")
        route_authority = str(placement.get("route_authority") or "")
        if route_authority == SUMO_DIRECTED_PHYSICAL_LANE_ROUTE_AUTHORITY:
            if edge_id not in SUMO_GROUND_FLOW.edges:
                raise RuntimeError(
                    f"{entity_id}: directed physical lane edge is not in SUMO: {edge_id}"
                )
            sample = _sumo_lane_sample_at_s(
                edge_id,
                float(placement["longitudinal_s"]),
            )
            lateral_m = float(placement["resolved_lateral_from_center_m"])
            physical_offsets = _physical_vehicle_lane_offsets(sample)
            if min(abs(lateral_m - offset_m) for offset_m in physical_offsets) > 0.25:
                raise RuntimeError(
                    f"{entity_id}: directed physical lane offset {lateral_m:.3f} "
                    f"is not one of {physical_offsets}"
                )
            lateral_m = min(
                physical_offsets,
                key=lambda offset_m: abs(lateral_m - offset_m),
            )
            resolved_position = _sumo_vehicle_point_at_s(
                edge_id,
                sample.s_m,
                lateral_m,
            )
            placement.update(
                {
                    "longitudinal_s": round(sample.s_m, 3),
                    "lateral_offset_m": round(lateral_m, 3),
                    "resolved_lateral_from_center_m": round(lateral_m, 3),
                    "placement_semantics": "physical_vehicle_lane_center",
                    "resolved_position_enu_m": resolved_position,
                    "rotation_deg": rot(sample.yaw_deg),
                    **_physical_vehicle_lane_audit(sample, lateral_m),
                }
            )
            scene["placement"] = placement
            spec = spec_by_id.get(entity_id)
            if spec is not None:
                spec["initial_pos_enu"] = resolved_position
                spec["initial_rotation_deg"] = [0.0, 0.0, sample.yaw_deg]
            directed_physical_entities.add(entity_id)
            continue
        if edge_id in LANES.by_edge:
            sample = LANES.resolve_edge_s(edge_id, float(placement.get("longitudinal_s") or 0.0))
        else:
            position = scene_pos(scene)
            sample = LANES.nearest_to_xy(float(position[0]), float(position[1]))
        position = scene_pos(scene)
        requested_lateral = float(placement.get("resolved_lateral_from_center_m", placement.get("lateral_offset_m", 0.0)))
        lateral_m = _select_physical_vehicle_lateral(sample, requested_lateral, position, index=index)
        resolved_position = _offset_from_lane(sample, lateral_m, float(position[2] if len(position) > 2 else GROUND_Z_M))
        route = _project_vehicle_waypoints_to_physical_lanes(
            [list(point) for point in scene.get("route_waypoints_enu_m") or []],
            lateral_m,
            index=index,
        )
        placement.update(
            {
                "edge_id": sample.edge_id,
                "longitudinal_s": round(sample.s_m, 3),
                "lateral_offset_m": round(lateral_m, 3),
                "resolved_lateral_from_center_m": round(lateral_m, 3),
                "placement_semantics": "physical_vehicle_lane_center",
                "resolved_position_enu_m": resolved_position,
                "rotation_deg": rot(sample.yaw_deg),
                **_physical_vehicle_lane_audit(sample, lateral_m),
            }
        )
        scene["placement"] = placement
        scene["route_waypoints_enu_m"] = route
        spec = spec_by_id.get(entity_id)
        if spec is not None:
            spec["initial_pos_enu"] = resolved_position
            spec["initial_rotation_deg"] = [0.0, 0.0, sample.yaw_deg]
            spec["movement_waypoints"] = route
        vehicle_laterals[entity_id] = lateral_m

    for event_def in events:
        for action_def in event_def.get("actions") or []:
            if str(action_def.get("type") or "") != "move_entity":
                continue
            params = action_params(action_def)
            entity_id = str(params.get("entity_id") or "")
            if entity_id in directed_physical_entities:
                continue
            if entity_id not in vehicle_laterals:
                continue
            waypoints = params.get("waypoints_enu_m") or []
            projected = _project_vehicle_waypoints_to_physical_lanes(
                [list(point) for point in waypoints if isinstance(point, list) and len(point) >= 2],
                vehicle_laterals[entity_id],
            )
            if projected:
                write_action_params(action_def, {"waypoints_enu_m": projected})


def path_length_m(waypoints: list[list[float]]) -> float:
    length = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        length += math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (float(a[2]) - float(b[2])) ** 2)
    return length


def quantized_path_duration_ticks(
    waypoints: list[list[float]],
    velocity_mps: float,
) -> int:
    if velocity_mps <= 0.0:
        raise ValueError(f"movement velocity must be positive, got {velocity_mps}")
    return sum(
        max(
            1,
            int(
                math.ceil(
                    math.dist(a, b) / velocity_mps * SCRIPT_TICK_HZ
                )
            ),
        )
        for a, b in zip(waypoints, waypoints[1:])
        if math.dist(a, b) > 1e-6
    )


def move_duration_ticks(action_def: dict[str, Any]) -> int:
    params = action_params(action_def)
    waypoints = params.get("waypoints_enu_m") or []
    if len(waypoints) < 2:
        return 0
    velocity = max(0.1, float(params.get("velocity_mps", 1.0)))
    return quantized_path_duration_ticks(waypoints, velocity)


def enforce_uav_terminal_feasibility(
    scenario_id: str,
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    """Prove every authored UAV terminal move fits the formal episode window."""

    scene_by_id = {str(scene.get("entity_id") or ""): scene for scene in scenes}
    preliminary_ticks = _event_dispatch_upper_bounds(events)
    event_by_id = {
        str(event_def.get("event_id") or ""): event_def for event_def in events
    }
    for event_def in events:
        trigger = event_def.get("trigger") or {}
        if trigger.get("type") != "event_fired_after":
            continue
        terminal_actions = [
            action_def
            for action_def in event_def.get("actions") or []
            if action_params(action_def).get("type") == "move_entity"
            and str(action_params(action_def).get("post_activity_type") or "").casefold()
            == "landed"
        ]
        if not terminal_actions:
            continue
        predecessor_id = str(trigger.get("event_ref") or "")
        predecessor = event_by_id[predecessor_id]
        predecessor_tick = preliminary_ticks[predecessor_id]
        minimum_physical_delay = max(
            (
                move_duration_ticks(action_def)
                for action_def in predecessor.get("actions") or []
                if action_params(action_def).get("type") == "move_entity"
            ),
            default=1,
        )
        route_reserve_ticks = max(
            int(
                math.ceil(
                    path_length_m(action_params(action_def).get("waypoints_enu_m") or [])
                    / UAV_MAX_SPEED_MPS
                    * SCRIPT_TICK_HZ
                )
            )
            for action_def in terminal_actions
        )
        latest_dispatch_tick = (
            DEFAULT_DURATION_TICKS
            - CAPTURE_TICK_STEP
            - UAV_TOUCHDOWN_DWELL_TICKS
            - route_reserve_ticks
        )
        latest_dispatch_tick = (
            latest_dispatch_tick // CAPTURE_TICK_STEP * CAPTURE_TICK_STEP
        )
        maximum_delay = latest_dispatch_tick - predecessor_tick
        if maximum_delay < minimum_physical_delay:
            raise RuntimeError(
                f"{scenario_id}:{event_def.get('event_id')}: no feasible terminal dispatch; "
                f"predecessor_tick={predecessor_tick} minimum_delay={minimum_physical_delay} "
                f"maximum_delay={maximum_delay} route_reserve={route_reserve_ticks}"
            )
        requested_delay = int(trigger.get("delay_ticks") or 0)
        applied_delay = min(max(requested_delay, minimum_physical_delay), maximum_delay)
        trigger["delay_ticks"] = applied_delay
        terminal_schedule = event_def.setdefault("terminal_schedule", {})
        terminal_schedule.update(
            {
                "final_predecessor_dispatch_upper_bound_tick": predecessor_tick,
                "minimum_physical_delay_ticks": minimum_physical_delay,
                "maximum_final_feasible_delay_ticks": maximum_delay,
                "applied_delay_ticks": applied_delay,
            }
        )
    trigger_ticks = _event_dispatch_upper_bounds(events)
    for event_def in events:
        event_id = str(event_def.get("event_id") or "")
        dispatch_tick = trigger_ticks.get(event_id)
        if dispatch_tick is None:
            raise RuntimeError(f"{scenario_id}:{event_id}: no deterministic dispatch upper bound")
        for action_def in event_def.get("actions") or []:
            params = action_params(action_def)
            if params.get("type") != "move_entity":
                continue
            entity_id = str(params.get("entity_id") or "")
            scene = scene_by_id.get(entity_id)
            if scene is None or not str(scene.get("logical_asset_id") or "").startswith("uav."):
                continue
            action_id = str(params.get("action_id") or action_def.get("action_id") or "")
            if str(params.get("post_activity_type") or "").casefold() != "landed":
                continue
            waypoints = [
                q(float(point[0]), float(point[1]), float(point[2]))
                for point in params.get("waypoints_enu_m") or []
                if isinstance(point, list) and len(point) >= 3
            ]
            if len(waypoints) < 2:
                raise RuntimeError(f"{scenario_id}:{action_id}: terminal move needs at least two waypoints")
            available_ticks = DEFAULT_DURATION_TICKS - int(dispatch_tick) - CAPTURE_TICK_STEP - UAV_TOUCHDOWN_DWELL_TICKS
            if available_ticks <= 0:
                raise RuntimeError(
                    f"{scenario_id}:{action_id}: no time remains after dispatch={dispatch_tick}"
                )
            route_length = path_length_m(waypoints)
            required_speed = route_length / (available_ticks / SCRIPT_TICK_HZ)
            if required_speed > UAV_MAX_SPEED_MPS + 1e-9:
                raise RuntimeError(
                    f"{scenario_id}:{action_id}: terminal route is unreachable; "
                    f"length={route_length:.3f}m dispatch={dispatch_tick} "
                    f"available_ticks={available_ticks} required_speed={required_speed:.3f} "
                    f"max_speed={UAV_MAX_SPEED_MPS:.3f}"
                )
            velocity = float(params.get("velocity_mps") or 0.0)
            if velocity <= 0.0 or velocity > UAV_MAX_SPEED_MPS + 1e-9:
                raise RuntimeError(
                    f"{scenario_id}:{action_id}: velocity_mps={velocity} violates UAV max-speed contract"
                )
            commanded_speed = min(UAV_MAX_SPEED_MPS, max(velocity, required_speed * 1.05))
            params["velocity_mps"] = round(commanded_speed, 3)
            params["terminal_feasibility"] = {
                "model": "constant_speed_path_with_explicit_landing_reference_v2",
                "dispatch_upper_bound_tick": int(dispatch_tick),
                "route_length_m": round(route_length, 6),
                "available_ticks": int(available_ticks),
                "required_speed_mps": round(required_speed, 6),
                "max_speed_mps": UAV_MAX_SPEED_MPS,
                "home_pose_origin_z_m": float(scene.get("ground_reference_z_m")),
                "landing_reference_enu_m": list(waypoints[-1]),
                "touchdown_altitude_tolerance_m": UAV_TOUCHDOWN_ALTITUDE_TOLERANCE_M,
                "touchdown_dwell_ticks": UAV_TOUCHDOWN_DWELL_TICKS,
            }
            write_action_params(action_def, params)


def validate_uav_conflict_causality(
    scenario_id: str,
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    """Require objective pair proximity to follow both aircraft activation paths."""

    scene_by_id = {str(scene.get("entity_id") or ""): scene for scene in scenes}
    for index, event_def in enumerate(events):
        trigger = dict(event_def.get("trigger") or {})
        if str(trigger.get("type") or "") != "entity_proximity":
            continue
        pair = (
            str(trigger.get("entity_a") or ""),
            str(trigger.get("entity_b") or ""),
        )
        if any(
            not str(scene_by_id.get(entity_id, {}).get("logical_asset_id") or "").startswith("uav.")
            for entity_id in pair
        ):
            continue
        prior_moves = {
            str(action_params(action_def).get("entity_id") or "")
            for prior in events[:index]
            for action_def in prior.get("actions") or []
            if action_params(action_def).get("type") == "move_entity"
        }
        missing_motion = sorted(entity_id for entity_id in pair if entity_id not in prior_moves)
        if missing_motion:
            raise RuntimeError(
                f"{scenario_id}:{event_def.get('event_id')}: proximity conflict has no prior motion for {missing_motion}"
            )
        for entity_id in pair:
            scene = scene_by_id[entity_id]
            position = scene_pos(scene)
            ground_z = scene.get("ground_reference_z_m")
            takeoff_seen = any(
                str(previous.get("intent") or "") == "uav_takes_off_from_visible_home_pad_and_enters_mission_airspace"
                and entity_id in {
                    str(value)
                    for value in previous.get("log_target_ids") or []
                    if value
                }
                for previous in events[:index]
            )
            initially_airborne = (
                position is not None
                and ground_z is not None
                and float(position[2]) - float(ground_z) > UAV_TOUCHDOWN_ALTITUDE_TOLERANCE_M
            )
            if not takeoff_seen and not initially_airborne:
                raise RuntimeError(
                    f"{scenario_id}:{event_def.get('event_id')}: {entity_id} is not airborne before conflict trigger"
                )


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


def enforce_move_start_continuity(
    scenario_id: str,
    scenes: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    scene_by_id = {str(scene.get("entity_id") or ""): scene for scene in scenes}
    current = {entity_id: scene_pos(scene) for entity_id, scene in scene_by_id.items()}
    repair_limits_m = {
        "pedestrian": 1.5,
        "vehicle": 5.0,
        "uav": 5.0,
        "asset": 1.5,
    }
    for event_def in events:
        for action_def in event_def.get("actions") or []:
            params = action_params(action_def)
            if params.get("type") != "move_entity":
                continue
            entity_id = str(params.get("entity_id") or "")
            if entity_id not in scene_by_id:
                continue
            waypoints = [
                q(float(point[0]), float(point[1]), float(point[2]))
                for point in params.get("waypoints_enu_m") or []
                if isinstance(point, list) and len(point) >= 3
            ]
            if not waypoints:
                continue
            expected_start = current.get(entity_id)
            asset_id = str(scene_by_id[entity_id].get("logical_asset_id") or "")
            motion_class = motion_class_from_asset(asset_id)
            if expected_start:
                gap_m = dist3(waypoints[0], expected_start)
                tolerance = MOVE_START_TOLERANCE_UAV_M if motion_class == "uav" else MOVE_START_TOLERANCE_GROUND_M
                if gap_m > tolerance:
                    if compiled_event_allows_validation_skip(event_def, "move_start_continuity", [], scenario_id):
                        current[entity_id] = waypoints[-1]
                        continue
                    if gap_m > repair_limits_m[motion_class]:
                        raise RuntimeError(
                            f"{scenario_id}: move_entity {params.get('action_id')} starts {gap_m:.1f}m from current "
                            f"{entity_id} position; this is too large for deterministic continuity repair"
                        )
                    original_start = list(waypoints[0])
                    waypoints[0] = q(expected_start[0], expected_start[1], expected_start[2])
                    params["waypoints_enu_m"] = waypoints
                    params["continuity_repair"] = {
                        "policy": "replace_first_waypoint_with_current_entity_position_v1",
                        "original_start_enu_m": original_start,
                        "start_gap_m": round(gap_m, 3),
                    }
                    write_action_params(action_def, params)
            current[entity_id] = waypoints[-1]


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
        if best[0] <= current_distance:
            # A governed threshold may intentionally exceed the generic 50 m
            # normalization ceiling.  If the authored physical trajectories
            # already reach that threshold, no normalization is needed.
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


def _uav_pad_airspace_clear(pad_position_enu: list[float]) -> bool:
    hover = q(pad_position_enu[0], pad_position_enu[1], UAV_PAD_HOVER_Z_M)
    approach = q(pad_position_enu[0], pad_position_enu[1], max(36.0, UAV_PAD_HOVER_Z_M))
    return UAV_CORRIDORS.buildings.air_point_clear(hover) and UAV_CORRIDORS.buildings.air_segment_clear(hover, approach)


def building_clear_landing_pad_pose(
    desired_pos_enu: list[float],
    *,
    context: str,
) -> tuple[list[float], float, PlannedSidewalkAnchor]:
    """Resolve a legal pad whose vertical UAV service volume is obstacle-clear."""

    candidate_points = [list(desired_pos_enu)]
    for radius_m in (12.0, 24.0, 36.0):
        for index in range(16):
            angle = math.tau * index / 16.0
            candidate_points.append(
                q(
                    float(desired_pos_enu[0]) + math.cos(angle) * radius_m,
                    float(desired_pos_enu[1]) + math.sin(angle) * radius_m,
                    GROUND_Z_M,
                )
            )
    seen_positions: set[tuple[float, float]] = set()
    errors: list[str] = []
    for candidate in candidate_points:
        try:
            position, yaw_deg, anchor = landing_pad_pose(
                candidate,
                context=context,
            )
        except (RuntimeError, SpatialValidationError) as exc:
            errors.append(str(exc))
            continue
        position_key = (round(float(position[0]), 3), round(float(position[1]), 3))
        if position_key in seen_positions:
            continue
        seen_positions.add(position_key)
        if _uav_pad_airspace_clear(position):
            return position, yaw_deg, anchor
        errors.append(f"{context}: vertical UAV service volume blocked at {position}")
    raise RuntimeError(
        f"Unable to place building-clear landing pad for {context}: {errors[:6]}"
    )


def _uav_terminal_landing_airspace_clear(ground_position_enu: list[float]) -> bool:
    touchdown = q(ground_position_enu[0], ground_position_enu[1], 0.8)
    land = q(ground_position_enu[0], ground_position_enu[1], 3.0)
    low = q(ground_position_enu[0], ground_position_enu[1], 8.0)
    return (
        UAV_CORRIDORS.buildings.air_point_clear(touchdown)
        and UAV_CORRIDORS.buildings.air_point_clear(land)
        and UAV_CORRIDORS.buildings.air_point_clear(low)
        and UAV_CORRIDORS.buildings.air_segment_clear(touchdown, land)
        and UAV_CORRIDORS.buildings.air_segment_clear(land, low)
    )


def _find_forced_landing_ground_anchor(
    desired_pos_enu: list[float],
    *,
    scenario_id: str,
) -> PlannedSidewalkAnchor:
    seed_points: list[list[float]] = []
    seen_seeds: set[tuple[float, float]] = set()

    def add_seed(point: list[float]) -> None:
        seed = q(point[0], point[1], GROUND_Z_M)
        key = (round(seed[0], 2), round(seed[1], 2))
        if key not in seen_seeds:
            seen_seeds.add(key)
            seed_points.append(seed)

    add_seed(desired_pos_enu)
    for altitude_m in (3.0, 8.0, 0.8):
        try:
            add_seed(UAV_CORRIDORS.buildings.repair_point(desired_pos_enu, altitude_m))
        except RuntimeError:
            pass
    repair_seed = seed_points[-1] if seed_points else desired_pos_enu
    for radius_m in (8.0, 16.0):
        for index in range(8):
            angle = (math.tau * index) / 8.0
            add_seed(
                q(
                    float(repair_seed[0]) + math.cos(angle) * radius_m,
                    float(repair_seed[1]) + math.sin(angle) * radius_m,
                    GROUND_Z_M,
                )
            )
    seed_points.sort(key=lambda point: dist_xy(point, desired_pos_enu))
    candidates: list[tuple[float, PlannedSidewalkAnchor]] = []
    errors: list[str] = []
    for seed in seed_points[:20]:
        for offset_from_curb in (4.8, 6.2, 8.0):
            try:
                anchor = SPATIAL.plan_sidewalk_anchor(
                    seed,
                    offset_from_curb_m=offset_from_curb,
                    allow_green=True,
                    placement_semantics="uav_forced_landing_zone",
                )
                _validate_landing_pad_position(anchor.position_enu_m, context=f"{scenario_id} forced landing zone")
            except (RuntimeError, SpatialValidationError) as exc:
                errors.append(str(exc))
                continue
            if not _uav_terminal_landing_airspace_clear(anchor.position_enu_m):
                errors.append(f"{scenario_id}: forced landing low/terminal airspace blocked at {anchor.position_enu_m}")
                continue
            candidates.append((dist_xy(anchor.position_enu_m, desired_pos_enu), anchor))
            if candidates and candidates[-1][0] <= 7.0:
                return anchor
    if not candidates:
        raise RuntimeError(f"{scenario_id}: unable to find building-clear forced landing ground anchor: {errors[:8]}")
    return min(candidates, key=lambda item: item[0])[1]


def _pedestrian_anchor_near_landing(
    landing_anchor: PlannedSidewalkAnchor,
    landing_position_enu: list[float],
    *,
    preferred_s_delta_m: float,
    occupied: list[list[float]],
) -> PlannedSidewalkAnchor:
    base = landing_anchor.sample
    min_s, max_s = LANES.edge_s_bounds(base.edge_id)
    side = _side_sign_for_desired(base, landing_position_enu)
    delta_candidates = (
        preferred_s_delta_m,
        preferred_s_delta_m * 0.7,
        preferred_s_delta_m * 1.3,
        -preferred_s_delta_m,
        0.0,
    )
    errors: list[str] = []
    for delta_s in delta_candidates:
        sample = LANES.resolve_edge_s(base.edge_id, max(min_s, min(max_s, base.s_m + delta_s)))
        for offset_from_curb in (4.8, 6.2, 8.0, 3.6, GATHERING_MIN_OFFSET_FROM_CURB_M, 10.0, 12.0):
            for side_sign in (side, -side):
                candidate_seed = _offset_from_lane(sample, side_sign * (LANE_HALF_WIDTH_M + offset_from_curb), GROUND_Z_M)
                try:
                    anchor = SPATIAL.plan_sidewalk_anchor(
                        candidate_seed,
                        edge_id_hint=sample.edge_id,
                        s_hint=sample.s_m,
                        offset_from_curb_m=offset_from_curb,
                        allow_green=True,
                        placement_semantics="sidewalk",
                    )
                except SpatialValidationError as exc:
                    errors.append(str(exc))
                    continue
                landing_distance_m = dist_xy(anchor.position_enu_m, landing_position_enu)
                if landing_distance_m < 2.0 or landing_distance_m > 7.0:
                    continue
                if any(dist_xy(anchor.position_enu_m, point) < 1.2 for point in occupied):
                    continue
                occupied.append(anchor.position_enu_m)
                return anchor
    raise RuntimeError(f"Unable to place crowd pedestrian near forced landing point {landing_position_enu}: {errors[:6]}")


def _apply_sidewalk_anchor_to_entity(
    spec: dict[str, Any],
    scene: dict[str, Any],
    anchor: PlannedSidewalkAnchor,
    *,
    placement_semantics: str = "sidewalk",
) -> None:
    update_world_entity_position(spec, scene, anchor.position_enu_m)
    spec["initial_rotation_deg"] = [0.0, 0.0, anchor.sample.yaw_deg]
    scene["placement_mode"] = "sidewalk_anchor"
    scene["placement"] = {
        **dict(scene.get("placement") or {}),
        "lane_edge_id": anchor.sample.edge_id,
        "longitudinal_s": round(anchor.sample.s_m, 3),
        "offset_from_curb_m": round(anchor.offset_from_curb_m, 3),
        "lane_half_width_m": LANE_HALF_WIDTH_M,
        "resolved_lateral_from_center_m": round(anchor.resolved_lateral_from_center_m, 3),
        "placement_semantics": placement_semantics,
        "resolved_position_enu_m": anchor.position_enu_m,
        "rotation_deg": rot(anchor.sample.yaw_deg),
    }


def _home_pad_clearance_kind(scene: dict[str, Any]) -> str | None:
    asset = str(scene.get("logical_asset_id") or "")
    category = str(scene.get("category") or "")
    if asset.startswith("pedestrian.") or category == "pedestrian":
        return "pedestrian"
    if asset.startswith("vehicle.") or category == "vehicle":
        return "vehicle"
    if asset.startswith("uav.") or category == "uav":
        return "uav"
    if category in {"facade_anchor", "building", "structure"}:
        return "structure"
    if asset == "facility.landing_pad.visible.v1":
        return "landing_pad"
    if asset == "facility.charger.cityops.v1":
        return "charger"
    if asset == "facility.radio.base_tower.v1":
        return "tower"
    if asset.startswith("facility.") or category == "facility":
        return "facility"
    return None


def _home_pad_occupied_points(
    scenes: list[dict[str, Any]],
    *,
    exclude_entity_ids: set[str] | None = None,
) -> list[tuple[list[float], float, str]]:
    excluded = exclude_entity_ids or set()
    occupied: list[tuple[list[float], float, str]] = []
    for scene in scenes:
        entity_id = str(scene.get("entity_id") or "")
        if entity_id in excluded:
            continue
        kind = _home_pad_clearance_kind(scene)
        if not kind:
            continue
        clearance_m = float(
            scene.get("uav_home_pad_clearance_m")
            or HOME_PAD_CLEARANCE_BY_KIND_M[kind]
        )
        points: list[list[float]] = []
        position = scene_pos(scene)
        if position:
            points.append(position)
        for waypoint in scene.get("route_waypoints_enu_m") or []:
            if isinstance(waypoint, list) and len(waypoint) >= 3:
                points.append(q(float(waypoint[0]), float(waypoint[1]), float(waypoint[2])))
        for point in points:
            occupied.append((point, clearance_m, f"{entity_id}:{kind}"))
    return occupied


def _home_pad_occupancy_conflict(
    pad_position: list[float],
    occupied: list[tuple[list[float], float, str]],
) -> str | None:
    for point, clearance_m, label in occupied:
        distance_m = dist_xy(pad_position, point)
        if distance_m + 0.05 < clearance_m:
            return f"{label} clearance {distance_m:.3f}m < {clearance_m:.3f}m"
    return None


def uav_home_pad_pose(
    mission_start_enu: list[float],
    slot_index: int,
    *,
    occupied: list[tuple[list[float], float, str]] | None = None,
    capture_boundary: dict[str, Any] | None = None,
    terminal_start_enu: list[float] | None = None,
) -> tuple[list[float], float]:
    base_sample = LANES.nearest_to_xy(float(mission_start_enu[0]), float(mission_start_enu[1]))
    preferred_side = 1.0 if slot_index % 2 == 0 else -1.0
    ring = slot_index // 2
    offset_base = GATHERING_MIN_OFFSET_FROM_CURB_M + ring * 4.0
    offset_candidates = [
        offset_base + extra for extra in (0.0, 1.5, 3.0, 5.0, 8.0, 12.0, 18.0)
    ]
    sample_candidates: list[LaneSample] = []
    seen_samples: set[tuple[str, float]] = set()

    def add_sample(sample: LaneSample) -> None:
        key = (sample.edge_id, round(sample.s_m, 3))
        if key not in seen_samples:
            seen_samples.add(key)
            sample_candidates.append(sample)

    min_s, max_s = LANES.edge_s_bounds(base_sample.edge_id)
    # Search longitudinally at metre-scale before moving to remote lane
    # samples.  This resolves narrow actor/building clearance intervals while
    # retaining the governed lateral pad clearance from the road.
    for delta_s in (
        0.0,
        -1.0,
        1.0,
        -2.0,
        2.0,
        -4.0,
        4.0,
        -6.0,
        6.0,
        -8.0,
        8.0,
        -16.0,
        16.0,
        -28.0,
        28.0,
        -45.0,
        45.0,
        -70.0,
        70.0,
    ):
        add_sample(
            LANES.interpolate_edge_s(
                base_sample.edge_id,
                base_sample.lane_id,
                max(min_s, min(max_s, base_sample.s_m + delta_s)),
            )
        )
    for sample in LANES.nearby_diverse_samples(
        float(mission_start_enu[0]),
        float(mission_start_enu[1]),
        max_radius_m=120.0,
        longitudinal_bucket_m=8.0,
        max_count=512,
    ):
        add_sample(sample)

    errors: list[str] = []
    for sample in sample_candidates:
        for side in (preferred_side, -preferred_side):
            for offset_from_curb in offset_candidates:
                desired_position = _offset_from_lane(
                    sample,
                    side * (LANE_HALF_WIDTH_M + offset_from_curb),
                    GROUND_Z_M,
                )
                try:
                    pad_position, yaw, _ = landing_pad_pose(
                    desired_position,
                    context=f"home UAV pad slot {slot_index}",
                    offset_from_curb_m=offset_from_curb,
                    edge_id_hint=sample.edge_id,
                    s_hint=sample.s_m,
                    )
                except (RuntimeError, SpatialValidationError) as exc:
                    errors.append(str(exc))
                    continue
                conflict = _home_pad_occupancy_conflict(pad_position, occupied or [])
                if conflict:
                    errors.append(f"home UAV pad slot {slot_index}: pad occupancy conflict at {pad_position}: {conflict}")
                    continue
                boundary_polygon = [
                    [float(point[0]), float(point[1])]
                    for point in (capture_boundary or {}).get("polygon_enu_m") or []
                    if isinstance(point, list) and len(point) >= 2
                ]
                if boundary_polygon:
                    distance_to_boundary_m = _distance_to_polygon_xy(pad_position, boundary_polygon)
                    required_restricted_clearance_m = float(
                        (capture_boundary or {}).get(
                            "pre_mission_takeoff_boundary_clearance_m", 0.0
                        )
                        or 0.0
                    )
                    if (
                        required_restricted_clearance_m > 0.0
                        and distance_to_boundary_m + 1e-6
                        < required_restricted_clearance_m
                    ):
                        errors.append(
                            f"home UAV pad slot {slot_index}: pad {pad_position} is "
                            f"{distance_to_boundary_m:.3f}m from restricted boundary, "
                            f"below required {required_restricted_clearance_m:.3f}m clearance"
                        )
                        continue
                    if required_restricted_clearance_m > 0.0:
                        pad_hover = q(
                            pad_position[0], pad_position[1], UAV_PAD_HOVER_Z_M
                        )
                        climb = q(
                            pad_position[0],
                            pad_position[1],
                            max(
                                UAV_MISSION_ALTITUDE_MIN_M,
                                min(float(mission_start_enu[2]) - 4.0, 24.0),
                            ),
                        )
                        try:
                            takeoff_route = _repair_uav_profile_waypoints(
                                "home_pad_selection",
                                f"takeoff_entry_slot_{slot_index}",
                                [pad_hover, climb, mission_start_enu],
                            )
                        except RuntimeError as exc:
                            errors.append(str(exc))
                            continue
                        route_clearance_m = _route_distance_to_polygon_xy(
                            takeoff_route, boundary_polygon
                        )
                        if (
                            route_clearance_m + 1e-6
                            < required_restricted_clearance_m
                        ):
                            errors.append(
                                f"home UAV pad slot {slot_index}: repaired takeoff route is "
                                f"{route_clearance_m:.3f}m from restricted boundary, below "
                                f"required {required_restricted_clearance_m:.3f}m clearance"
                            )
                            continue
                        maximum_takeoff_route_length_m = float(
                            (capture_boundary or {})[
                                "pre_mission_takeoff_max_route_length_m"
                            ]
                        )
                        takeoff_route_length_m = path_length_m(takeoff_route)
                        if (
                            takeoff_route_length_m
                            > maximum_takeoff_route_length_m + 1e-6
                        ):
                            errors.append(
                                f"home UAV pad slot {slot_index}: repaired takeoff route is "
                                f"{takeoff_route_length_m:.3f}m long, above the pre-mission "
                                f"limit {maximum_takeoff_route_length_m:.3f}m"
                            )
                            continue
                    required_landing_clearance_m = float(
                        (capture_boundary or {}).get(
                            "post_conflict_landing_boundary_clearance_m", 0.0
                        )
                        or 0.0
                    )
                    if required_landing_clearance_m > 0.0:
                        if terminal_start_enu is None:
                            raise RuntimeError(
                                f"home UAV pad slot {slot_index}: landing boundary "
                                "clearance requires an explicit terminal start"
                            )
                        try:
                            _boundary_clear_landing_route(
                                "home_pad_selection",
                                f"landing_return_slot_{slot_index}",
                                terminal_start_enu,
                                q(
                                    pad_position[0],
                                    pad_position[1],
                                    UAV_PAD_HOVER_Z_M,
                                ),
                                boundary_polygon,
                                required_landing_clearance_m,
                            )
                        except (RuntimeError, ValueError) as exc:
                            errors.append(
                                f"home UAV pad slot {slot_index}: {exc}"
                            )
                            continue
                    if distance_to_boundary_m > CONTRACT_RUNTIME_VISIBLE_PADDING_M:
                        errors.append(
                            f"home UAV pad slot {slot_index}: pad {pad_position} is "
                            f"{distance_to_boundary_m:.1f}m from capture boundary, outside "
                            f"{CONTRACT_RUNTIME_VISIBLE_PADDING_M:.1f}m runtime visibility"
                        )
                        continue
                if _uav_pad_airspace_clear(pad_position):
                    return pad_position, yaw
                errors.append(f"home UAV pad slot {slot_index}: pad hover/vertical airspace blocked at {pad_position}")
    raise RuntimeError(
        f"Unable to place building-clear home UAV pad for slot {slot_index} "
        f"after {len(sample_candidates)} lane samples: first={errors[:3]}, last={errors[-3:]}"
    )


def update_world_entity_position(spec: dict[str, Any], scene: dict[str, Any], position_enu: list[float]) -> None:
    spec["initial_pos_enu"] = position_enu
    placement = scene.setdefault("placement", {})
    placement["position_enu_m"] = position_enu
    placement["resolved_position_enu_m"] = position_enu


def preamble_sidewalk_start(
    event_position: list[float],
    dx_m: float,
    dy_m: float,
    offset_from_curb_m: float,
    *,
    min_span_m: float = 7.0,
    max_span_m: float = 18.0,
) -> list[float]:
    base_sample = LANES.nearest_to_xy(float(event_position[0]), float(event_position[1]))
    min_s, max_s = LANES.edge_s_bounds(base_sample.edge_id)
    yaw_rad = math.radians(base_sample.yaw_deg)
    preferred_s_delta = dx_m * math.cos(yaw_rad) + dy_m * math.sin(yaw_rad)
    desired_sign = _side_sign_for_desired(base_sample, event_position)
    errors: list[str] = []
    for sign in (desired_sign, -desired_sign):
        for s_delta in (
            preferred_s_delta,
            -preferred_s_delta,
            -22.0,
            22.0,
            -18.0,
            18.0,
            -14.0,
            14.0,
            -10.0,
            10.0,
            -7.0,
            7.0,
        ):
            sample = LANES.resolve_edge_s(base_sample.edge_id, max(min_s, min(max_s, base_sample.s_m + s_delta)))
            for extra_offset in (0.0, 1.5, 3.0, 5.0, 8.0):
                offset = max(abs(offset_from_curb_m), SIDEWALK_MIN_OFFSET_FROM_CURB_M) + extra_offset
                candidate = _offset_from_lane(sample, sign * (LANE_HALF_WIDTH_M + offset), GROUND_Z_M)
                span_m = dist_xy(candidate, event_position)
                if span_m < min_span_m or span_m > max_span_m:
                    continue
                point_errors = SPATIAL.validation_errors_for_point(
                    candidate,
                    context="preamble_sidewalk_start",
                    allow_road=False,
                    allow_green=True,
                )
                if point_errors:
                    errors.extend(point_errors[:2])
                    continue
                return q(candidate[0], candidate[1], candidate[2] if len(candidate) > 2 else GROUND_Z_M)
    raise RuntimeError(f"Unable to plan pre-event sidewalk start from {event_position}: {errors[:8]}")


def pedestrian_preamble_walk_event(
    spec: dict[str, Any],
    scene: dict[str, Any],
    scenario_id: str,
    event_id: str,
    action_id: str,
    *,
    tick_value: int = 120,
    dx_m: float = -12.0,
    dy_m: float = 8.0,
    offset_from_curb_m: float = GATHERING_MIN_OFFSET_FROM_CURB_M,
    activity_type: str = "walking",
) -> tuple[list[float], dict[str, Any]]:
    event_position = scene_pos(scene)
    entity_id = str(scene.get("entity_id") or spec.get("entity_id") or "")
    pre_event_start = preamble_sidewalk_start(
        event_position,
        dx_m,
        dy_m,
        offset_from_curb_m,
    )
    update_world_entity_position(spec, scene, pre_event_start)
    return event_position, context_event(
        event_id,
        tick(tick_value),
        [
            action(
                action_id,
                "move_entity",
                entity_id=entity_id,
                waypoints_enu_m=[pre_event_start, event_position],
                velocity_mps=1.25,
                activity_type=normalize_activity_type(activity_type, moving=True),
                post_activity_type=normalize_activity_type("waiting"),
            )
        ],
        0,
        scenario_id,
        "Context pedestrian walks into the event area before the primary event",
        [entity_id],
    )


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
    capture_boundary: dict[str, Any] | None = None,
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
    last_move_is_landed_terminal: dict[str, bool] = {}
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
            last_move_is_landed_terminal[entity_id] = (
                str(params.get("post_activity_type") or "").casefold() == "landed"
            )

    if not last_scenario_event_id:
        raise RuntimeError(f"{scenario_id}: UAV lifecycle requires a scenario event chain")

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
        occupied = _home_pad_occupied_points(scenes, exclude_entity_ids={entity_id})
        final_position = current_positions.get(entity_id, mission_start)
        pad_position, pad_yaw = uav_home_pad_pose(
            mission_start,
            slot_index,
            occupied=occupied,
            capture_boundary=capture_boundary,
            terminal_start_enu=final_position,
        )
        pad_hover = q(pad_position[0], pad_position[1], UAV_PAD_HOVER_Z_M)
        climb = q(pad_position[0], pad_position[1], max(18.0, min(float(mission_start[2]) - 4.0, 24.0)))
        takeoff_speed_mps = float(
            (capture_boundary or {}).get(
                "pre_mission_takeoff_speed_mps", UAV_TAKEOFF_SPEED_MPS
            )
        )

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
        scene["ground_reference_z_m"] = float(pad_hover[2])
        scene["ground_reference_source"] = "home_hover_pose_origin_v1"
        spec["movement_waypoints"] = [mission_start] + list(spec.get("movement_waypoints") or [])
        spec["visual_state"] = {**(spec.get("visual_state") or {}), "mode": "preflight_on_pad"}

        lifecycle_events.append(
            event(
                f"lifecycle_takeoff_{suffix}",
                tick(UAV_TAKEOFF_ENTRY_TICK + slot_index * 8),
                [
                    move(
                        f"move_{suffix}_takeoff_entry",
                        entity_id,
                        [pad_hover, climb, mission_start],
                        takeoff_speed_mps,
                        activity_type="takeoff",
                        post_activity_type="cruise",
                    )
                ],
                0,
                scenario_id,
                "UAV takes off from visible home pad and enters mission airspace",
                "uav_lifecycle",
                [entity_id, pad_id],
                "info",
            )
        )

        final_z = float(final_position[2])
        # A scene that declares its flight ends damaged and grounded gets no automatic
        # return-to-pad landing: the aircraft is not airworthy after the terminal event,
        # so scheduling a nominal landing would resurrect it as a normal flight.
        flight_terminated = bool(
            (scene.get("lifecycle") or {}).get("terminal_flight_state")
        )
        if not last_move_is_landed_terminal.get(entity_id, False) and not flight_terminated:
            landing_altitude = max(final_z, 36.0)
            approach = q(final_position[0], final_position[1], landing_altitude)
            pad_approach = q(pad_position[0], pad_position[1], landing_altitude)
            landing_after_event = last_scenario_event_id
            landing_route = [final_position, approach, pad_approach, pad_hover]
            landing_boundary_clearance_m = float(
                (capture_boundary or {}).get(
                    "post_conflict_landing_boundary_clearance_m", 0.0
                )
                or 0.0
            )
            if landing_boundary_clearance_m > 0.0:
                boundary_polygon = [
                    [float(point[0]), float(point[1])]
                    for point in (capture_boundary or {}).get("polygon_enu_m") or []
                    if isinstance(point, list) and len(point) >= 2
                ]
                if len(boundary_polygon) < 3:
                    raise RuntimeError(
                        f"{scenario_id}:{entity_id}: landing boundary clearance "
                        "requires an explicit polygon"
                    )
                landing_route = _boundary_clear_landing_route(
                    scenario_id,
                    f"move_{suffix}_landing_return",
                    final_position,
                    pad_hover,
                    boundary_polygon,
                    landing_boundary_clearance_m,
                )
            trigger_ticks = _event_dispatch_upper_bounds(events)
            base_dispatch_tick = int(
                trigger_ticks.get(
                    landing_after_event,
                    UAV_TAKEOFF_ENTRY_TICK + slot_index * 8,
                )
            )
            predecessor_event = next(
                event_def
                for event_def in events
                if str(event_def.get("event_id") or "") == landing_after_event
            )
            minimum_physical_delay_ticks = max(
                (
                    move_duration_ticks(action_def)
                    for action_def in predecessor_event.get("actions") or []
                    if action_params(action_def).get("type") == "move_entity"
                ),
                default=1,
            )
            configured_delay_ticks = max(
                UAV_LANDING_AFTER_SCENARIO_DELAY_TICKS + slot_index * 8,
                minimum_physical_delay_ticks,
            )
            max_delay_ticks = (
                DEFAULT_DURATION_TICKS
                - base_dispatch_tick
                - CAPTURE_TICK_STEP
                - UAV_TOUCHDOWN_DWELL_TICKS
                - int(math.ceil(path_length_m(landing_route) / UAV_MAX_SPEED_MPS * SCRIPT_TICK_HZ))
            )
            if SPATIAL_PROVISIONAL_BUILD:
                landing_delay_ticks = minimum_physical_delay_ticks
                landing_velocity_mps = UAV_MAX_SPEED_MPS
            else:
                if max_delay_ticks < minimum_physical_delay_ticks:
                    raise RuntimeError(
                        f"{scenario_id}:{entity_id}: terminal route is unreachable after the "
                        f"predecessor's {minimum_physical_delay_ticks}-tick physical completion; "
                        f"maximum feasible delay={max_delay_ticks}"
                    )
                landing_delay_ticks = min(configured_delay_ticks, max_delay_ticks)
                landing_dispatch_tick = base_dispatch_tick + landing_delay_ticks
                available_ticks = DEFAULT_DURATION_TICKS - landing_dispatch_tick - CAPTURE_TICK_STEP - UAV_TOUCHDOWN_DWELL_TICKS
                required_velocity_mps = path_length_m(landing_route) / (available_ticks / SCRIPT_TICK_HZ)
                landing_velocity_mps = round(
                    min(UAV_MAX_SPEED_MPS, max(4.0, required_velocity_mps * 1.05)),
                    3,
                )
            landing_event = event(
                f"lifecycle_landing_{suffix}",
                fired(landing_after_event, landing_delay_ticks),
                [
                    move(
                        f"move_{suffix}_landing_return",
                        entity_id,
                        landing_route,
                        landing_velocity_mps,
                        activity_type="return_to_pad",
                        post_activity_type="landed",
                    )
                ],
                9,
                scenario_id,
                "UAV returns to the visible home pad and lands after mission resolution",
                "uav_lifecycle",
                [entity_id, pad_id],
                "info",
                intent="landing_or_terminal_resolution",
            )
            landing_event["terminal_schedule"] = {
                "configured_delay_ticks": configured_delay_ticks,
                "applied_delay_ticks": landing_delay_ticks,
                "minimum_physical_delay_ticks": minimum_physical_delay_ticks,
                "base_dispatch_upper_bound_tick": base_dispatch_tick,
                "max_feasible_delay_ticks": max_delay_ticks,
                "route_length_m": round(path_length_m(landing_route), 6),
                "max_speed_mps": UAV_MAX_SPEED_MPS,
                "spatial_provisional_only": SPATIAL_PROVISIONAL_BUILD,
            }
            landing_events.append(landing_event)

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
    align_vehicle_lane_motion(spec_entities, scene_entities, events)
    ensure_uav_corridor_population(scenario_id, spec_entities, scene_entities, events, parameters)
    enforce_physical_delays(events)
    lifecycle_capture_boundary = _capture_boundary_from_points(scenario_id, scene_entities, events)
    takeoff_boundary_clearance_m = parameters.get(
        "pre_mission_takeoff_boundary_clearance_m"
    )
    landing_boundary_clearance_m = parameters.get(
        "post_conflict_landing_boundary_clearance_m"
    )
    if takeoff_boundary_clearance_m is not None:
        lifecycle_capture_boundary["pre_mission_takeoff_boundary_clearance_m"] = float(
            takeoff_boundary_clearance_m
        )
        lifecycle_capture_boundary["pre_mission_takeoff_max_route_length_m"] = float(
            parameters["pre_mission_takeoff_max_route_length_m"]
        )
        lifecycle_capture_boundary["pre_mission_takeoff_speed_mps"] = float(
            parameters["pre_mission_takeoff_speed_mps"]
        )
    if landing_boundary_clearance_m is not None:
        lifecycle_capture_boundary[
            "post_conflict_landing_boundary_clearance_m"
        ] = float(landing_boundary_clearance_m)
    add_uav_lifecycle(scenario_id, spec_entities, scene_entities, events, lifecycle_capture_boundary)
    provisional_capture_boundary = _capture_boundary_from_points(scenario_id, scene_entities, events)
    ensure_uav_routes_cross_capture_boundary(
        scenario_id,
        spec_entities,
        scene_entities,
        events,
        provisional_capture_boundary,
        contract,
    )
    apply_uav_corridor_contract(
        scenario_id,
        spec_entities,
        scene_entities,
        events,
        parameters,
        contract,
    )
    parameters["utm_service_plan"] = build_utm_service_plan(
        scenario_id,
        scene_entities,
    )
    enforce_move_start_continuity(scenario_id, scene_entities, events)
    validate_uav_conflict_causality(scenario_id, scene_entities, events)
    enforce_physical_delays(events)
    # The provisional bundle exists only to choose a spatial assignment center;
    # road projection can make its route lengths physically meaningless.
    if not SPATIAL_PROVISIONAL_BUILD:
        enforce_uav_terminal_feasibility(scenario_id, scene_entities, events)
    normalize_proximity_trigger_reachability(scenario_id, scene_entities, events, parameters)
    capture_boundary_policy = _scenario_pad_boundary_policy(scenario_id)
    capture_boundary = {
        **_capture_boundary_from_points(scenario_id, scene_entities, events),
        "uav_boundary_crossing_required": True,
        "pad_boundary_policy": capture_boundary_policy,
        "inspect_fov_coverage_required": True,
        "inspect_entity_role": "U_inspect",
    }
    if takeoff_boundary_clearance_m is not None:
        capture_boundary["pre_mission_takeoff_boundary_clearance_m"] = float(
            takeoff_boundary_clearance_m
        )
        capture_boundary["pre_mission_takeoff_max_route_length_m"] = float(
            parameters["pre_mission_takeoff_max_route_length_m"]
        )
        capture_boundary["pre_mission_takeoff_speed_mps"] = float(
            parameters["pre_mission_takeoff_speed_mps"]
        )
    if landing_boundary_clearance_m is not None:
        capture_boundary["post_conflict_landing_boundary_clearance_m"] = float(
            landing_boundary_clearance_m
        )
    if not SPATIAL_PROVISIONAL_BUILD:
        relocate_event_actor_entries_outside_capture_boundary(
            scenario_id,
            spec_entities,
            scene_entities,
            events,
            capture_boundary,
        )
        enforce_move_start_continuity(scenario_id, scene_entities, events)
        _ensure_contract_pedestrians_reach_capture_context(
            scenario_id,
            scene_entities,
            capture_boundary,
            contract,
        )
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
    finalize_event_semantics(events, scenario_id)
    apply_motion_contracts(scene_entities, events, contract)
    spec_by_id = {
        str(spec.get("entity_id") or ""): spec for spec in spec_entities
    }
    for scene in scene_entities:
        if not str(scene.get("logical_asset_id") or "").startswith("uav."):
            continue
        entity_id = str(scene.get("entity_id") or "")
        planned_route = scene.get("planned_route_waypoints_enu_m")
        if planned_route is None:
            planned_route = copy.deepcopy(scene.get("route_waypoints_enu_m") or [])
            scene["planned_route_waypoints_enu_m"] = planned_route
        if len(planned_route) < 2:
            raise RuntimeError(f"{scenario_id}:{entity_id}: UAV lacks an explicit planned route")
        spec_by_id[entity_id]["planned_route_waypoints_enu_m"] = copy.deepcopy(planned_route)
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
            "takeoff",
            "liftoff",
            "launch",
            "charger",
            "low_observation",
        )
    )


def build_utm_service_plan(
    scenario_id: str,
    scene_entities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Declare complete deterministic UTM state without consulting event labels."""

    uav_ids = sorted(
        str(entity["entity_id"])
        for entity in scene_entities
        if str(entity.get("category") or "") == "uav"
    )
    if not uav_ids:
        raise RuntimeError(f"{scenario_id}: UTM service plan requires at least one UAV")
    conflict_enabled = scenario_id.startswith("L2-4_") or scenario_id.startswith(
        "X5_"
    )
    shared_ids = set(uav_ids[:2]) if conflict_enabled and len(uav_ids) >= 2 else set()
    shared_corridor_id = f"utm_corridor:{scenario_id}:shared_service_volume"
    rows: list[dict[str, Any]] = []
    for rank, uav_id in enumerate(uav_ids):
        corridor_id = (
            shared_corridor_id
            if uav_id in shared_ids
            else f"utm_corridor:{scenario_id}:uav_{rank:02d}"
        )
        authorization_schedule = [
            {"start_tick": 0, "end_tick": 900, "status": "active"}
        ]
        if scenario_id.startswith("X6_") and rank == 0:
            authorization_schedule = [
                {"start_tick": 0, "end_tick": 510, "status": "active"},
                {"start_tick": 515, "end_tick": 675, "status": "revoked"},
                {"start_tick": 680, "end_tick": 900, "status": "active"},
            ]
        flight_plan_schedule = [
            {
                "start_tick": int(interval["start_tick"]),
                "end_tick": int(interval["end_tick"]),
                "approval_status": (
                    "approved" if interval["status"] == "active" else "rejected"
                ),
            }
            for interval in authorization_schedule
        ]
        rows.append(
            {
                "uav_id": uav_id,
                "corridor_id": corridor_id,
                "intent_start_tick": 260
                if uav_id in shared_ids and rank == 1
                else 0,
                "intent_end_tick": 520
                if uav_id in shared_ids and rank == 1
                else 900,
                "authorization_schedule": authorization_schedule,
                "flight_plan_schedule": flight_plan_schedule,
            }
        )
    return {
        "schema_name": "utm_service_plan",
        "schema_version": "1.0.0",
        "model_id": "aeroworld_deterministic_utm_service",
        "model_version": "1.0.0",
        "global_uav_flow_policy": {
            "source_contract": "uav_global_flow",
            "intent_start_tick": 0,
            "intent_end_tick": 900,
            "authorization_status": "active",
            "flight_plan_approval_status": "approved",
        },
        "uav_plans": rows,
    }


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


def _shortest_boundary_clear_path_xy(
    start: list[float],
    end: list[float],
    polygon: list[list[float]],
    minimum_clearance_m: float,
) -> list[tuple[float, float]]:
    polygon_geometry = Polygon(
        [(float(point[0]), float(point[1])) for point in polygon]
    )
    if not polygon_geometry.is_valid or polygon_geometry.area <= 0.0:
        raise ValueError("landing boundary polygon must be valid and non-degenerate")
    start_xy = (float(start[0]), float(start[1]))
    end_xy = (float(end[0]), float(end[1]))
    for label, point_xy in (("start", start_xy), ("end", end_xy)):
        endpoint_clearance_m = Point(point_xy).distance(polygon_geometry)
        if endpoint_clearance_m + 1e-6 < minimum_clearance_m:
            raise RuntimeError(
                f"landing {label} clearance {endpoint_clearance_m:.3f}m is below "
                f"required {minimum_clearance_m:.3f}m"
            )
    direct = LineString([start_xy, end_xy])
    if direct.distance(polygon_geometry) + 1e-6 >= minimum_clearance_m:
        return [start_xy, end_xy]

    routing_boundary = polygon_geometry.buffer(
        minimum_clearance_m + BOUNDARY_ROUTE_PLANNING_MARGIN_M,
        join_style=2,
    )
    if routing_boundary.geom_type != "Polygon":
        raise RuntimeError(
            f"landing boundary buffer produced unsupported geometry {routing_boundary.geom_type}"
        )
    boundary_vertices = [
        (float(x), float(y)) for x, y in list(routing_boundary.exterior.coords)[:-1]
    ]
    nodes = [start_xy, end_xy, *boundary_vertices]
    distances = [float("inf")] * len(nodes)
    predecessors: list[int | None] = [None] * len(nodes)
    unvisited = set(range(len(nodes)))
    distances[0] = 0.0
    while unvisited:
        current = min(unvisited, key=lambda index: (distances[index], index))
        unvisited.remove(current)
        if not math.isfinite(distances[current]) or current == 1:
            break
        for target in sorted(unvisited):
            segment = LineString([nodes[current], nodes[target]])
            if segment.distance(polygon_geometry) + 1e-6 < minimum_clearance_m:
                continue
            candidate_distance = distances[current] + segment.length
            if candidate_distance + 1e-9 < distances[target]:
                distances[target] = candidate_distance
                predecessors[target] = current
    if not math.isfinite(distances[1]):
        raise RuntimeError(
            f"no landing route satisfies {minimum_clearance_m:.3f}m boundary clearance"
        )
    route_indices = [1]
    while route_indices[-1] != 0:
        predecessor = predecessors[route_indices[-1]]
        if predecessor is None:
            raise RuntimeError("landing boundary route predecessor chain is incomplete")
        route_indices.append(predecessor)
    route_indices.reverse()
    return [nodes[index] for index in route_indices]


def _boundary_clear_landing_route(
    scenario_id: str,
    action_id: str,
    terminal_start: list[float],
    pad_hover: list[float],
    boundary_polygon: list[list[float]],
    minimum_clearance_m: float,
) -> list[list[float]]:
    landing_altitude = max(float(terminal_start[2]), 36.0)
    approach = q(terminal_start[0], terminal_start[1], landing_altitude)
    pad_approach = q(pad_hover[0], pad_hover[1], landing_altitude)
    horizontal_path = _shortest_boundary_clear_path_xy(
        approach,
        pad_approach,
        boundary_polygon,
        minimum_clearance_m,
    )
    planned_route = [terminal_start, approach]
    planned_route.extend(
        q(point[0], point[1], landing_altitude) for point in horizontal_path[1:]
    )
    planned_route.append(pad_hover)
    repaired_route = _repair_uav_profile_waypoints(
        scenario_id,
        action_id,
        planned_route,
    )
    actual_clearance_m = _route_distance_to_polygon_xy(
        repaired_route, boundary_polygon
    )
    if actual_clearance_m + 1e-6 < minimum_clearance_m:
        raise RuntimeError(
            f"building-repaired landing route clearance {actual_clearance_m:.3f}m is "
            f"below required {minimum_clearance_m:.3f}m"
        )
    return repaired_route


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
        home_hover = scene["lifecycle"].get("home_hover_enu_m")
        home_hover_pos = q(float(home_hover[0]), float(home_hover[1]), float(home_hover[2])) if isinstance(home_hover, list) and len(home_hover) >= 3 else None
        mission_start = next(
            (
                list(point)
                for point in repaired_route
                if len(point) >= 3
                and float(point[2]) >= UAV_MISSION_ALTITUDE_MIN_M
                and (home_hover_pos is None or dist_xy(point, home_hover_pos) > 2.0)
            ),
            None,
        )
        if mission_start is None:
            mission_start = next(
                (
                    list(point)
                    for point in repaired_route
                    if len(point) >= 3 and float(point[2]) >= UAV_MISSION_ALTITUDE_MIN_M
                ),
                None,
            )
        lifecycle_update = {
            **scene["lifecycle"],
            "assigned_altitude_m": altitude_m,
            "waypoint_repair_applied": True,
        }
        if mission_start is not None:
            lifecycle_update["mission_start_enu_m"] = mission_start
        scene["lifecycle"] = {
            **lifecycle_update,
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
    # The observer corridor is an intentionally recomputed search result; it is not
    # frozen.  Its endpoints are re-derived from the live capture boundary by
    # ensure_uav_route_crosses_capture_boundary, so pinning the slot would not pin
    # the observable position anyway.
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
    contract: EpisodeContract,
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
    fixed_altitudes = parameters.get("fixed_uav_assigned_altitudes_m") or {}
    if not isinstance(fixed_altitudes, Mapping):
        raise RuntimeError(f"{scenario_id}: fixed_uav_assigned_altitudes_m must be a mapping")
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
        elif entity_id in fixed_altitudes:
            altitude_m = float(fixed_altitudes[entity_id])
            needs_lateral_bypass = False
        else:
            altitude_m, needs_lateral_bypass = _choose_uav_altitude(entity_id, ordinal, route_points, used_altitudes)
        used_altitudes.add(altitude_m)
        assigned_altitudes[entity_id] = altitude_m
        used_lateral_bypass[entity_id] = needs_lateral_bypass

        base_start = scene_pos(scene)
        lifecycle = dict(scene.get("lifecycle") or {})
        preserve_low_lifecycle_start = bool(lifecycle.get("home_pad_entity_id")) and bool(
            base_start and float(base_start[2]) <= UAV_LANDING_ALTITUDE_MAX_M
        )
        if preserve_low_lifecycle_start:
            repaired_start = q(base_start[0], base_start[1], base_start[2])
            if not UAV_CORRIDORS.buildings.air_point_clear(repaired_start):
                raise RuntimeError(
                    f"{scenario_id}: UAV lifecycle pad hover is not building-clear for {entity_id}: {repaired_start}"
                )
        else:
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
            final_start, final_route = _find_inspect_route(
                scenario_id,
                scenes,
                events,
                contract,
            )
            planned_route = [final_start, *final_route]
            update_world_entity_position(spec, scene, final_start)
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
                "planned_route_enu_m": planned_route,
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


# L1-1 is the only L1 family whose primary UAV is expected to fly inside the capture
# polygon before the conflict response stops it.  Under the current engine a hover
# activity stops unfinished motion at dispatch, so the response radius alone decides
# whether the aircraft reaches its authored waypoint.  Keeping this separate from
# conflict_distance_m leaves the takeoff/landing clearance derivation untouched
# (clearance stays conflict_distance_m + one max-speed tick).
L1_1_BOUNDARY_RESPONSE_DISTANCE_M = 10.0


def l1_1_response_distance_m(scenario_id: str) -> float:
    """Response radius for boundary_conflict; only L1-1_v1/v2 uses the dedicated value."""
    if scenario_id in ("L1-1_v1", "L1-1_v2"):
        return L1_1_BOUNDARY_RESPONSE_DISTANCE_M
    return 14.0


def build_l1(scenario_id: str, idx: int) -> ScenarioBundle:
    approach_tick = 250 if scenario_id == "L1-1_v1" else 220
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
            tick(approach_tick),
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
            prox(uav, zone, l1_1_response_distance_m(scenario_id), 3, metric="xy"),
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
        intr_near = q(near[0] + 2, near[1] + 1, near[2] - 1)
        add(specs, scenes, world_entity(intruder, "uav.airsim.flying_pawn.v1", "uav", intr_start, 215, "noncooperative", route=[intr_near]))
        events[0]["actions"][0] = move(
            "move_boundary_near_miss_approach",
            uav,
            [start, approach_mid, near],
            8.0,
        )
        events[0]["actions"].append(
            move(
                "move_intruder_near_miss_converge",
                intruder,
                [intr_start, intr_near],
                10.0,
            )
        )
        events[1]["trigger"] = prox(intruder, zone, 14.0, 3, metric="xy")
        events[1]["log_title"] = "Noncooperative intruder enters constrained airspace"
        if scenario_id.startswith("L1-3_v2"):
            events[1]["log_title"] = "Fast intruder crossing constrained airspace"
            events[2]["trigger"] = fired("boundary_conflict", delay_ticks=130)
        else:
            events[2]["trigger"] = fired("boundary_conflict", delay_ticks=31)
        events[1]["log_target_ids"].append(intruder)
    events[1]["require_conditions"] = ["trig_approach_boundary"]
    parameters = {
        "approach_tick": approach_tick,
        "conflict_distance_m": 14.0,
        "resolution_tick": 420,
    }
    if scenario_id in ("L1-1_v1", "L1-1_v2"):
        # Response trigger radius, deliberately distinct from conflict_distance_m:
        # conflict_distance_m also feeds the takeoff/landing boundary clearance
        # (clearance = conflict_distance_m + max displacement per tick = 14.8 m),
        # so lowering it would silently relax the ground-clearance safety margin.
        parameters["boundary_response_distance_m"] = l1_1_response_distance_m(scenario_id)
        parameters["boundary_response_distance_basis"] = (
            "L1-1 only: places the boundary_conflict response late enough that the "
            "authored approach leg reaches its waypoint inside the capture polygon "
            "before the hover activity stops motion"
        )
        parameters["boundary_response_distance_scope"] = (
            "controls the boundary_conflict entity_proximity trigger only; does not "
            "affect conflict_distance_m, capture boundary geometry, or predicate "
            "thresholds"
        )
    if scenario_id.startswith("L1-4"):
        specs[:] = [item for item in specs if item.get("entity_id") != zone]
        scenes[:] = [item for item in scenes if item.get("entity_id") != zone]

        uav_b = f"uav_{scenario_id.lower().replace('-', '_')}_secondary"
        uav_c = f"uav_{scenario_id.lower().replace('-', '_')}_tertiary"
        congestion_altitude_m = 50.0
        cell_start = q(center[0] - 30.0, center[1], congestion_altitude_m)
        cell_end = q(center[0] + 60.0, center[1], congestion_altitude_m)
        primary_start = q(cell_start[0] - 35.0, cell_start[1] - 24.0, congestion_altitude_m)
        secondary_start = q(cell_start[0] - 55.0, cell_start[1] + 24.0, congestion_altitude_m)
        tertiary_start = q(cell_start[0] - 75.0, cell_start[1] - 24.0, congestion_altitude_m)
        primary_hold = q(cell_start[0] + 40.0, cell_start[1], congestion_altitude_m)
        secondary_hold = q(cell_start[0] + 20.0, cell_start[1], congestion_altitude_m)
        tertiary_hold = q(cell_start[0], cell_start[1], congestion_altitude_m)
        secondary_divert = q(secondary_hold[0], secondary_hold[1] + 24.0, congestion_altitude_m)
        tertiary_divert = q(tertiary_hold[0], tertiary_hold[1] - 24.0, congestion_altitude_m)
        primary_clear = q(cell_end[0] + 30.0, cell_end[1], congestion_altitude_m)

        primary_spec = next(item for item in specs if item["entity_id"] == uav)
        primary_scene = next(item for item in scenes if item["entity_id"] == uav)
        update_world_entity_position(primary_spec, primary_scene, primary_start)
        primary_spec["movement_waypoints"] = [primary_hold, primary_clear]
        primary_scene["route_waypoints_enu_m"] = [primary_hold, primary_clear]
        primary_scene["initial_state"] = {
            **(primary_scene.get("initial_state") or {}),
            "assigned_altitude_m": congestion_altitude_m,
            "uav_corridor_role": "mission_uav",
        }
        primary_spec["visual_state"] = {
            **(primary_spec.get("visual_state") or {}),
            "assigned_altitude_m": congestion_altitude_m,
        }

        secondary_spec, secondary_scene = add(
            specs,
            scenes,
            world_entity(
                uav_b,
                "uav.airsim.cv_pawn.v1",
                "uav",
                secondary_start,
                20,
                "patrol",
                route=[secondary_hold, secondary_divert],
            ),
        )
        tertiary_spec, tertiary_scene = add(
            specs,
            scenes,
            world_entity(
                uav_c,
                "uav.airsim.flying_pawn.v1",
                "uav",
                tertiary_start,
                45,
                "patrol",
                route=[tertiary_hold, tertiary_divert],
            ),
        )
        for spec, scene in (
            (secondary_spec, secondary_scene),
            (tertiary_spec, tertiary_scene),
        ):
            scene["initial_state"] = {
                **(scene.get("initial_state") or {}),
                "assigned_altitude_m": congestion_altitude_m,
                "uav_corridor_role": "mission_uav",
            }
            spec["visual_state"] = {
                **(spec.get("visual_state") or {}),
                "assigned_altitude_m": congestion_altitude_m,
            }

        shared_corridor_spec, shared_corridor_scene = _corridor_entity_from_segment(
            scenario_id,
            "shared_congestion_cell",
            0,
            cell_start,
            cell_end,
        )
        shared_corridor_scene["entity_id"] = f"corridor_{safe_id(scenario_id)}_shared_congestion_cell"
        shared_corridor_spec["entity_id"] = shared_corridor_scene["entity_id"]
        shared_corridor_scene["placement"]["source_uav_entity_id"] = "mission_uav_group"
        shared_corridor_scene["initial_state"] = {
            **(shared_corridor_scene.get("initial_state") or {}),
            "semantic_role": "shared capacity-one corridor cell",
        }
        add(specs, scenes, (shared_corridor_spec, shared_corridor_scene))

        parameters["fixed_uav_assigned_altitudes_m"] = {
            uav: congestion_altitude_m,
            uav_b: congestion_altitude_m,
            uav_c: congestion_altitude_m,
        }
        parameters["l1_4_corridor_congestion_contract"] = {
            "predicate_id": "airspace.airspace_corridor_over_capacity",
            "corridor_id": shared_corridor_scene["entity_id"],
            "capacity": 1,
            "minimum_center_separation_m": 15.0,
            "required_true_samples": 2,
            "mission_aircraft_ids": [uav, uav_b, uav_c],
            "inside_sample_positions_enu_m": {
                uav: primary_hold,
                uav_b: secondary_hold,
                uav_c: tertiary_hold,
            },
        }
        events = [
            event(
                "corridor_capacity_exceeded",
                tick(approach_tick),
                [
                    move("move_corridor_primary_enter_capacity_cell", uav, [primary_start, primary_hold], 6.0),
                    move("move_corridor_secondary_enter_capacity_cell", uav_b, [secondary_start, secondary_hold], 6.0),
                    move("move_corridor_tertiary_enter_capacity_cell", uav_c, [tertiary_start, tertiary_hold], 6.0),
                ],
                1,
                scenario_id,
                "Corridor capacity exceeded by mission UAVs",
                "airspace",
                [uav, uav_b, uav_c, shared_corridor_scene["entity_id"]],
                "warning",
                intent="corridor_congestion",
            ),
            event(
                "priority_resequence",
                fired("corridor_capacity_exceeded", delay_ticks=90),
                [
                    move("move_corridor_secondary_divert_out", uav_b, [secondary_hold, secondary_divert], 6.0),
                    move("move_corridor_tertiary_hold_out", uav_c, [tertiary_hold, tertiary_divert], 6.0),
                ],
                2,
                scenario_id,
                "Priority resequence diverts excess corridor traffic",
                "uav_mission",
                [uav, uav_b, uav_c, shared_corridor_scene["entity_id"]],
                "warning",
                intent="avoid_or_rth",
            ),
            event(
                "corridor_capacity_restored",
                fired("priority_resequence", delay_ticks=80),
                [
                    move("move_corridor_primary_clear_cell", uav, [primary_hold, primary_clear], 7.5),
                ],
                3,
                scenario_id,
                "Corridor capacity restored after resequence",
                "airspace",
                [uav, uav_b, uav_c, shared_corridor_scene["entity_id"]],
                "info",
                intent="corridor_congestion",
            ),
        ]
        events[1]["require_conditions"] = ["trig_corridor_capacity_exceeded"]
        events[2]["require_conditions"] = ["trig_priority_resequence"]
    if scenario_id.startswith("L1-2"):
        events[1]["actions"][0] = move("move_altitude_recover", uav, [near, q(near[0] - 3, near[1] + 4, 42), q(near[0] - 8, near[1] + 8, 34)], 4.0)
        events[1]["log_title"] = "Altitude deviation from assigned corridor"
    if scenario_id == "L1-1_v1":
        max_displacement_per_tick_m = UAV_MAX_SPEED_MPS / SCRIPT_TICK_HZ
        pre_mission_ticks = parameters["approach_tick"] - UAV_TAKEOFF_ENTRY_TICK
        parameters["pre_mission_takeoff_speed_mps"] = UAV_MAX_SPEED_MPS
        parameters["pre_mission_takeoff_boundary_clearance_m"] = round(
            parameters["conflict_distance_m"] + max_displacement_per_tick_m, 3
        )
        parameters["post_conflict_landing_boundary_clearance_m"] = parameters[
            "pre_mission_takeoff_boundary_clearance_m"
        ]
        parameters["pre_mission_takeoff_max_route_length_m"] = round(
            parameters["pre_mission_takeoff_speed_mps"]
            * pre_mission_ticks
            / SCRIPT_TICK_HZ,
            3,
        )
        parameters["pre_mission_takeoff_boundary_clearance_model"] = {
            "boundary_conflict_threshold_m": parameters["conflict_distance_m"],
            "uav_max_speed_mps": UAV_MAX_SPEED_MPS,
            "simulation_tick_hz": SCRIPT_TICK_HZ,
            "max_displacement_per_tick_m": max_displacement_per_tick_m,
            "takeoff_entry_tick": UAV_TAKEOFF_ENTRY_TICK,
            "first_mission_event_tick": parameters["approach_tick"],
            "pre_mission_ticks": pre_mission_ticks,
            "takeoff_speed_mps": parameters["pre_mission_takeoff_speed_mps"],
            "maximum_takeoff_route_length_m": parameters[
                "pre_mission_takeoff_max_route_length_m"
            ],
            "required_clearance_m": parameters[
                "pre_mission_takeoff_boundary_clearance_m"
            ],
        }
        parameters["post_conflict_landing_boundary_clearance_model"] = {
            "boundary_conflict_threshold_m": parameters["conflict_distance_m"],
            "uav_max_speed_mps": UAV_MAX_SPEED_MPS,
            "simulation_tick_hz": SCRIPT_TICK_HZ,
            "max_displacement_per_tick_m": max_displacement_per_tick_m,
            "routing_numerical_margin_m": BOUNDARY_ROUTE_PLANNING_MARGIN_M,
            "routing_algorithm": "shortest_boundary_visibility_graph_then_building_repair",
            "required_clearance_m": parameters[
                "post_conflict_landing_boundary_clearance_m"
            ],
        }
    validation_rules = [
        {
            "rule": "event_chain_min",
            "min_count": 3,
            "description": "Airspace scenario must include approach, conflict, and recovery",
        },
    ]
    if scenario_id.startswith("L1-4"):
        validation_rules.append(
            {
                "rule": "l1_4_corridor_congestion_contract",
                "description": "Mission UAV geometry must exceed one-aircraft corridor capacity before resequence and restore below capacity after diversion",
            }
        )
    else:
        validation_rules.insert(
            0,
            {
                "rule": "uav_start_outside_constraint",
                "entity_id": uav,
                "constraint_id": zone,
                "description": "UAV start is outside the no-fly or hazard volume",
            },
        )
    if scenario_id == "L1-1_v1":
        validation_rules.append(
            {
                "rule": "pre_mission_takeoff_boundary_clearance",
                "entity_id": uav,
                "boundary_id": zone,
                "minimum_distance_m": parameters[
                    "pre_mission_takeoff_boundary_clearance_m"
                ],
                "maximum_route_length_m": parameters[
                    "pre_mission_takeoff_max_route_length_m"
                ],
                "takeoff_speed_mps": parameters[
                    "pre_mission_takeoff_speed_mps"
                ],
                "description": "The complete building-repaired takeoff route must remain outside the restricted-airspace event threshold plus one maximum-speed simulation tick",
            }
        )
        validation_rules.append(
            {
                "rule": "post_conflict_landing_boundary_clearance",
                "entity_id": uav,
                "boundary_id": zone,
                "minimum_distance_m": parameters[
                    "post_conflict_landing_boundary_clearance_m"
                ],
                "description": "The complete building-repaired return-to-pad route must remain outside the restricted-airspace event threshold plus one maximum-speed simulation tick",
            }
        )
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
        parameters,
        events,
        validation_rules=validation_rules,
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
            event("patrol_resume_via_backup_link", fired("uav_link_response"), [visual("set_tower_backup", tower, "backup_link"), move("move_uav_resume_patrol", uav, [degraded_loiter, home], 7.0)], 3, scenario_id, "UAV resumes patrol; tower confirms backup-link visual state", "infrastructure", [uav, tower], "info"),
        ]
        desc = "Communication station failure with backup link recovery"
    elif scenario_id.startswith("L2-2"):
        tower = f"tower_{sid}"
        uav = f"uav_{sid}"
        facade = f"facade_{sid}"
        start = p(ox, oy, 30)
        planned = p(ox + 14, oy + 4, 28)
        drift = p(ox + 34, oy + 14, 24)
        corrected = p(ox + 18, oy + 20, 30)
        add(specs, scenes, world_entity(tower, "facility.radio.base_tower.v1", "facility", p(ox + 4, oy + 2, 0), 0, "online"))
        add(specs, scenes, facade_entity(facade, BUILDINGS.id_for(20 + idx), p(ox + 30, oy + 16, 18), [0.8, -0.6, 0.0]))
        uav_spec, uav_scene = add(
            specs,
            scenes,
            world_entity(
                uav,
                "uav.airsim.cv_pawn.v1",
                "uav",
                start,
                45,
                "corridor_follow",
                route=[planned, drift, corrected],
            ),
        )
        uav_scene["planned_route_waypoints_enu_m"] = [start, planned, corrected]
        uav_scene["path_deviation_contract"] = {
            "minimum_route_distance_m": 15.0,
            "predicate_id": "positioning.aircraft_deviates_from_planned_route",
        }
        uav_spec["planned_route_waypoints_enu_m"] = [start, planned, corrected]
        events = [
            event("gnss_anomaly", tick(250), [visual("set_tower_multipath", tower, "multipath_warning")], 1, scenario_id, "GNSS multipath anomaly in urban canyon", "digital_layer", [uav, tower], "warning"),
            event("route_drift", fired("gnss_anomaly"), [move("move_uav_multipath_drift", uav, [start, planned, drift], 6.0)], 2, scenario_id, "UAV drifts off planned route by more than 15m", "uav_mission", [uav, facade], "warning"),
            event("visual_relocalization", prox(uav, facade, 8.0, 3, metric="xy"), [visual("set_uav_visual_reloc", uav, "visual_relocalization"), screenshot("capture_gnss_drift")], 3, scenario_id, "Visual relocalization engages near facade", "uav_mission", [uav, facade], "warning"),
            event("route_corrected", fired("visual_relocalization"), [move("move_uav_corrected_route", uav, [drift, corrected], 5.0)], 4, scenario_id, "UAV corrects the route after visual relocalization", "uav_mission", [uav], "info"),
        ]
        params["path_deviation_min_m"] = 15.0
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
        backup_position = scene_pos(scenes[-1])
        backup_touchdown = q(backup_position[0], backup_position[1], 1.2)
        add(
            specs,
            scenes,
            world_entity(
                uav,
                "uav.inspect.quad.v1",
                "uav",
                start,
                110,
                "low_battery",
                route=[inspect, backup_app, backup_touchdown],
                visual_state={
                    "charging_service_assignment": {
                        "facility_id": backup,
                        "arrival_tick": 620,
                        "departure_tick": 810,
                        "physical_service_required": True,
                    }
                },
            ),
        )
        events = [
            event("charger_unavailable", tick(240), [visual("set_primary_charger_offline", charger, "unavailable")], 1, scenario_id, "Primary charger becomes unavailable", "infrastructure", [charger], "warning"),
            event("uav_inspects_charger", fired("charger_unavailable"), [move("move_uav_charger_inspection", uav, [start, p(ox, oy + 3, 18), inspect], 4.0), screenshot("capture_charger_fault")], 2, scenario_id, "UAV checks the unavailable charger", "uav_mission", [uav, charger], "warning"),
            event("reroute_backup_charger", fired("uav_inspects_charger"), [move("move_uav_backup_charger", uav, [inspect, p(ox + 14, oy + 6, 18), backup_app, backup_touchdown], 6.0, "return_to_pad", "landed"), visual("set_backup_charger_reserved", backup, "reserved")], 3, scenario_id, "UAV reroutes to backup charger and lands", "uav_mission", [uav, backup], "info", intent="avoid_or_rth"),
            event(
                "backup_charging_service",
                fired("reroute_backup_charger", delay_ticks=360),
                [visual("set_uav_charging", uav, "charging"), visual("set_backup_charger_serving", backup, "serving")],
                4,
                scenario_id,
                "UAV occupies the backup charger and begins deterministic charging service",
                "uav_mission",
                [uav, backup],
                "info",
                intent="charging_service",
            ),
        ]
        desc = "Charger unavailable with reroute to backup charger"
    elif scenario_id.startswith("L2-4"):
        pad = f"pad_shared_{sid}"
        uav_a = f"uav_priority_{sid}"
        uav_b = f"uav_divert_{sid}"
        start_a = p(ox - 12, oy + 12, 32)
        start_b = p(ox + 18, oy - 10, 30)
        hold_b = p(ox + 14, oy + 14, 36)
        pad_pos, pad_yaw, pad_anchor = building_clear_landing_pad_pose(
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
            event("second_uav_diverted", fired("priority_arbitration"), [move("move_second_uav_diversion", uav_b, [hold_b, p(ox + 22, oy, 34)], 8.0), move("move_priority_uav_landing", uav_a, [pad_hover, pad_touchdown], 1.5)], 3, scenario_id, "Second UAV diverts while priority UAV lands", "uav_mission", [uav_a, uav_b, pad], "info"),
        ]
        params["contention_distance_m"] = 7.0
        desc = "Landing pad emergency contention and priority diversion"
    else:
        signal = "traffic_signal_l2_5_v1"
        police = "police_unit_l2_5_v1"
        car_a = "civilian_car_l2_5_a"
        car_b = "civilian_car_l2_5_b"
        def signal_same_lane_corridor() -> tuple[str, dict[str, float]]:
            focus = p(ox, oy, GROUND_Z_M)
            offsets_m = {
                "police_start": -72.0,
                "car_b_entry": -58.0,
                "car_a_entry": -42.0,
                "car_b_start": -40.0,
                "car_a_start": -24.0,
                "car_b_stop": -12.0,
                "car_a_stop": -4.0,
                "police_manual_control": -48.0,
                "signal": 0.0,
                "car_b_release": 8.0,
                "car_a_release": 16.0,
            }
            candidates: list[tuple[float, float, str, dict[str, float]]] = []
            seen: set[tuple[str, int]] = set()
            for base_sample in LANES.samples:
                edge_id = str(base_sample.edge_id)
                if edge_id.startswith(":"):
                    continue
                sumo_edge = SUMO_GROUND_FLOW.edges.get(edge_id)
                if sumo_edge is None or not SUMO_GROUND_FLOW._edge_allows(sumo_edge, "vehicle"):
                    continue
                key = (edge_id, int(round(float(base_sample.s_m))))
                if key in seen:
                    continue
                seen.add(key)
                min_s, max_s = LANES.edge_s_bounds(edge_id)
                lane_laterals = [
                    _select_physical_vehicle_lateral(base_sample, 0.0, [base_sample.x_m, base_sample.y_m, GROUND_Z_M], index=index)
                    for index in (1, 2, 3)
                ]
                if max(abs(value) for value in lane_laterals) > 3.0:
                    continue
                route_s = {
                    name: float(base_sample.s_m) + offset
                    for name, offset in offsets_m.items()
                }
                if min(route_s.values()) < min_s - 1e-6 or max(route_s.values()) > max_s + 1e-6:
                    continue
                sumo_centerline = LineString(sumo_edge.shape_xy)
                planned_points = [
                    _offset_from_lane(LANES.resolve_edge_s(edge_id, s_m), 0.0, GROUND_Z_M)
                    for s_m in route_s.values()
                ]
                projection_error_m = max(
                    float(sumo_centerline.distance(Point(point[0], point[1])))
                    for point in planned_points
                )
                if projection_error_m > 1.0:
                    continue
                route_center = [
                    sum(point[0] for point in planned_points) / float(len(planned_points)),
                    sum(point[1] for point in planned_points) / float(len(planned_points)),
                    GROUND_Z_M,
                ]
                candidates.append(
                    (
                        dist_xy(route_center, focus),
                        dist_xy(_offset_from_lane(LANES.resolve_edge_s(edge_id, route_s["signal"]), 0.0, GROUND_Z_M), focus),
                        edge_id,
                        route_s,
                    )
                )
            if not candidates:
                raise RuntimeError(f"{scenario_id}: no legal same-lane SUMO corridor near assigned signal focus")
            _center_error, _signal_error, edge_id, route_s = min(candidates)
            return edge_id, route_s

        signal_edge, signal_s = signal_same_lane_corridor()

        def signal_lane_point(s_m: float, lateral_m: float = 0.0, z_m: float = GROUND_Z_M) -> list[float]:
            return _sumo_vehicle_point_at_s(signal_edge, s_m, lateral_m, z_m)

        def signal_lane_route(*s_values: float) -> list[list[float]]:
            return [signal_lane_point(s_m) for s_m in s_values]

        entry_a = signal_lane_point(signal_s["car_a_entry"])
        entry_b = signal_lane_point(signal_s["car_b_entry"])
        start_a = signal_lane_point(signal_s["car_a_start"])
        stop_a = signal_lane_point(signal_s["car_a_stop"])
        release_a = signal_lane_point(signal_s["car_a_release"])
        start_b = signal_lane_point(signal_s["car_b_start"])
        stop_b = signal_lane_point(signal_s["car_b_stop"])
        release_b = signal_lane_point(signal_s["car_b_release"])
        police_start = signal_lane_point(signal_s["police_start"])
        police_manual_control = signal_lane_point(signal_s["police_manual_control"])
        signal_pos = signal_lane_point(signal_s["signal"], -4.5, 4.0)
        add(specs, scenes, world_entity(signal, "prop.traffic_control.signal_light.v1", "traffic_light", signal_pos, 0, "green_cycle"))
        _, car_a_scene = add(specs, scenes, lane_entity(car_a, "vehicle.ground.boxcar.v1", "vehicle", signal_edge, signal_s["car_a_entry"], 0.0, entry_a, 80, "moving", prefer_edge_hint=True, route_authority="sumo_directed_lane_geometry"))
        _, car_b_scene = add(specs, scenes, lane_entity(car_b, "vehicle.ground.boxcar.v1", "vehicle", signal_edge, signal_s["car_b_entry"], 0.0, entry_b, 80, "moving", prefer_edge_hint=True, route_authority="sumo_directed_lane_geometry"))
        _, police_scene = add(specs, scenes, lane_entity(police, "vehicle.emergency.police_suv.v1", "vehicle", signal_edge, signal_s["police_start"], 0.0, police_start, 80, "response", visual_state={"mode": "response", "lights_on": True}, prefer_edge_hint=True, route_authority="sumo_directed_lane_geometry"))
        car_a_entry_event = event_actor_entry(
            scenario_id=scenario_id,
            scene=car_a_scene,
            event_id="civilian_car_a_fault_entry",
            semantic_event_id="traffic_confusion",
            entry_tick=210,
            arrival_tick=240,
            route=signal_lane_route(signal_s["car_a_entry"], signal_s["car_a_start"]),
            activity_type="moving",
        )
        car_b_entry_event = event_actor_entry(
            scenario_id=scenario_id,
            scene=car_b_scene,
            event_id="civilian_car_b_fault_entry",
            semantic_event_id="traffic_confusion",
            entry_tick=210,
            arrival_tick=240,
            route=signal_lane_route(signal_s["car_b_entry"], signal_s["car_b_start"]),
            activity_type="moving",
        )
        signal_repair = event(
            "signal_infrastructure_repair", fired("manual_flow_restore", delay_ticks=60),
            [visual("set_signal_nominal_after_repair", signal, "nominal")],
            2, scenario_id, "Traffic signal hardware repaired after manual control established",
            "infrastructure_layer", [signal], "info", intent="infrastructure_repair",
        )
        signal_repair["intent_stage"] = "07.infrastructure_repair"
        signal_repair["target_roles"] = ["facility"]
        events = [
            car_a_entry_event,
            car_b_entry_event,
            event("signal_all_red_fault", tick(210), [visual("set_signal_all_red", signal, "all_red_fault")], 1, scenario_id, "Traffic signal all-red fault", "infrastructure", [signal], "critical"),
            event("traffic_confusion", fired("signal_all_red_fault"), [move("move_car_a_confused_stop", car_a, signal_lane_route(signal_s["car_a_start"], signal_s["car_a_start"] + 4.0, signal_s["car_a_start"] + 8.0, signal_s["car_a_start"] + 12.0, signal_s["car_a_start"] + 16.0, signal_s["car_a_stop"]), 3.0), move("move_car_b_confused_stop", car_b, signal_lane_route(signal_s["car_b_start"], signal_s["car_b_start"] + 6.0, signal_s["car_b_start"] + 12.0, signal_s["car_b_start"] + 18.0, signal_s["car_b_start"] + 24.0, signal_s["car_b_stop"]), 3.0)], 2, scenario_id, "Traffic queues form under all-red fault", "vehicle", [car_a, car_b, signal], "warning"),
            event("police_arrival", fired("traffic_confusion", delay_ticks=96), [move("move_police_to_intersection", police, signal_lane_route(signal_s["police_start"], signal_s["police_start"] + 4.0, signal_s["police_start"] + 8.0, signal_s["police_start"] + 12.0, signal_s["police_start"] + 16.0, signal_s["police_manual_control"]), 12.0)], 3, scenario_id, "Police starts driving to the manual traffic-control point", "vehicle", [police, signal], "warning"),
            event("manual_flow_restore", fired("police_arrival"), [visual("set_police_manual_control", police, "manual_directing", lights_on=True), move("move_car_a_released", car_a, signal_lane_route(signal_s["car_a_stop"], signal_s["signal"], signal_s["signal"] + 4.0, signal_s["car_b_release"], signal_s["car_b_release"] + 4.0, signal_s["car_a_release"]), 6.0), move("move_car_b_released", car_b, signal_lane_route(signal_s["car_b_stop"], signal_s["car_b_stop"] + 4.0, signal_s["car_a_stop"], signal_s["signal"], signal_s["signal"] + 4.0, signal_s["car_b_release"]), 5.5)], 4, scenario_id, "Police restores manual vehicle flow", "vehicle", [police, car_a, car_b], "info"),
            signal_repair,
            context_event("signal_response_actors_leave", fired("manual_flow_restore", delay_ticks=100), [remove("remove_civilian_car_a_after_restore", car_a), remove("remove_civilian_car_b_after_restore", car_b), remove("remove_police_after_restore", police)], 0, scenario_id, "Traffic and police leave the event stage after restored flow is observed", [police, car_a, car_b]),
        ]
        spawn_event_actor_at_dispatch(police_scene, events[4])
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
    weather_profile: dict[str, Any] | None = None
    if scenario_id == "L3-1_v1":
        uav = "uav_roadwork_inspector_l3_1"
        car = "blocked_car_l3_1"
        prop_entities: list[dict[str, Any]] = []
        if ACTIVE_ROI_CONTRACT is None:
            work_base = LANES.nearest_to_xy(*p(ox, oy, 0)[:2])
            work_edge = work_base.edge_id
            work_s = work_base.s_m
        else:
            evidence = dict(ACTIVE_ROI_CONTRACT.minimal_change_evidence or {})
            if not ACTIVE_ROI_CONTRACT.full_chain_rebuild_required or not evidence:
                raise RuntimeError(
                    f"{scenario_id}: formal ROI is topologically infeasible and requires "
                    "an explicit minimal rebuild contract"
                )
            work_edge = str(evidence.get("chosen_ue_lane_edge_id") or "")
            work_s = float(evidence.get("roadwork_anchor_s_m"))
            if evidence.get("lane_geometry_authority") != SUMO_GROUND_FLOW.traffic_geometry_authority:
                raise RuntimeError(
                    f"{scenario_id}: ROI rebuild lane geometry authority diverges from UE traffic bundle"
                )
            if evidence.get("roadway_mesh_hash") != SUMO_GROUND_FLOW.roadway_mesh_hash:
                raise RuntimeError(
                    f"{scenario_id}: ROI rebuild roadway mesh hash is stale"
                )
            if work_edge not in SUMO_GROUND_FLOW.edges:
                raise RuntimeError(
                    f"{scenario_id}: rebuild contract references missing edge {work_edge!r}"
                )
            work_base = _sumo_lane_sample_at_s(work_edge, work_s)

        work_yaw_rad = math.radians(work_base.yaw_deg)
        work_tangent = (math.cos(work_yaw_rad), math.sin(work_yaw_rad))
        work_normal = (-work_tangent[1], work_tangent[0])
        work_center = _sumo_vehicle_point_at_s(work_edge, work_s)

        def roadwork_point(along_m: float, cross_m: float, z_m: float) -> list[float]:
            return q(
                work_center[0] + along_m * work_tangent[0] + cross_m * work_normal[0],
                work_center[1] + along_m * work_tangent[1] + cross_m * work_normal[1],
                z_m,
            )
        for i in range(3):
            ent = f"barrier_l3_1_{i + 1:02d}"
            pos = p(ox + i * 4.5, oy, 0)
            _, scene = add(specs, scenes, lane_entity(ent, "prop.roadwork.barrier.v1", "prop", work_edge, work_s + i * 5, 1.5, pos, 90, "staged", activation_tick=220, prefer_edge_hint=True))
            prop_entities.append(scene)
        for i, sx in enumerate([-3.0, 12.5]):
            ent = f"fence_l3_1_{i + 1:02d}"
            pos = p(ox + sx, oy + 1.6, 0)
            _, scene = add(specs, scenes, lane_entity(ent, "prop.roadwork.construction_fence.v1", "prop", work_edge, work_s - 2 + i * 18, 1.5, pos, 90, "staged", activation_tick=220, prefer_edge_hint=True))
            prop_entities.append(scene)
        for i in range(5):
            ent = f"cone_l3_1_{i + 1:02d}"
            pos = p(ox - 2 + i * 3.6, oy - 2.4, 0)
            _, scene = add(specs, scenes, lane_entity(ent, "prop.roadwork.traffic_cone.v1", "prop", work_edge, work_s - 4 + i * 4, 1.5, pos, 90, "staged", activation_tick=220, prefer_edge_hint=True))
            prop_entities.append(scene)
        car_tactical_s = work_s + 12.0
        car_entry_s = work_s
        car_edge_min_s, _car_edge_max_s = LANES.edge_s_bounds(work_edge)
        if car_entry_s < car_edge_min_s:
            raise RuntimeError(f"{scenario_id}: roadwork lane anchor is outside its SUMO edge")
        car_entry = _sumo_vehicle_point_at_s(work_edge, car_entry_s)
        car_start = _sumo_vehicle_point_at_s(work_edge, car_tactical_s)
        _, car_scene = add(specs, scenes, lane_entity(car, "vehicle.ground.boxcar.v1", "vehicle", work_edge, car_entry_s, 0.0, car_entry, 90, "moving", prefer_edge_hint=True))
        car_edge_id = str((car_scene.get("placement") or {}).get("edge_id") or "")
        car_s = car_tactical_s
        car_lateral = float((car_scene.get("placement") or {}).get("resolved_lateral_from_center_m") or 0.0)
        car_edge_max_s = SUMO_GROUND_FLOW.edge_shape_length_m(car_edge_id)
        car_mid = _sumo_vehicle_point_at_s(
            car_edge_id,
            min(car_edge_max_s, car_s + 12.0),
            car_lateral,
        )
        car_end = _sumo_vehicle_point_at_s(
            car_edge_id,
            min(car_edge_max_s, car_s + 28.0),
            car_lateral,
        )
        uav_start = roadwork_point(-18.0, -12.0, 32.0)
        uav_scan_mid = roadwork_point(4.0, -4.0, 28.0)
        uav_scan_end = roadwork_point(15.0, 3.0, 28.0)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 65, "inspection"))
        spawn_actions = [spawn_from_scene(f"spawn_{entity_scene['entity_id']}", entity_scene) for entity_scene in prop_entities]
        car_entry_event = event_actor_entry(
            scenario_id=scenario_id,
            scene=car_scene,
            event_id="roadwork_vehicle_entry",
            semantic_event_id="lane_closure_active",
            entry_tick=220,
            arrival_tick=250,
            route=[car_entry, car_start],
            activity_type="moving",
        )
        events = [
            car_entry_event,
            event("roadwork_barriers_spawn", tick(220), spawn_actions, 1, scenario_id, "Roadwork barriers and cones appear", "dynamic_constraint", [e["entity_id"] for e in prop_entities[:4]], "warning"),
            event("lane_closure_active", fired("roadwork_barriers_spawn"), [visual("set_blocked_car_waiting", car, "blocked"), screenshot("capture_lane_closure")], 2, scenario_id, "Lane closure becomes active", "vehicle", [car], "warning"),
            event("vehicle_detour", fired("lane_closure_active"), [move("move_blocked_car_detour", car, [car_start, car_mid, car_end], 6.0)], 3, scenario_id, "Blocked vehicle detours around construction", "vehicle", [car], "info"),
            event("uav_inspection_report", fired("vehicle_detour"), [move("move_uav_roadwork_scan", uav, [uav_start, uav_scan_mid, uav_scan_end], 5.0), screenshot("capture_roadwork_report")], 4, scenario_id, "UAV reports lane closure geometry", "uav_mission", [uav], "info"),
            context_event("roadwork_vehicle_leaves", fired("uav_inspection_report", delay_ticks=30), [remove("remove_blocked_car_after_detour", car)], 0, scenario_id, "Detoured vehicle leaves the event stage after the inspection evidence is captured", [car]),
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
        nfz_scene["deactivation_tick"] = 520
        events = [
            event("temporary_nfz_declared", tick(280), [spawn_from_scene("spawn_temporary_nfz", nfz_scene, visual_state={"mode": "active"})], 1, scenario_id, "Temporary no-fly zone declared mid-operation", "dynamic_constraint", [nfz], "warning"),
            event("uav_approaches_temporary_nfz", fired("temporary_nfz_declared"), [move("move_uav_to_nfz_edge", uav, [start, approach], 7.0)], 2, scenario_id, "UAV approaches newly declared NFZ", "uav_mission", [uav, nfz], "warning"),
            event("nfz_proximity_alert", prox(uav, nfz, 12.0, 3, metric="xy"), [move("move_uav_nfz_reroute", uav, [approach, p(ox - 2, oy + 18, 34), p(ox + 24, oy + 28, 34)], 8.5), screenshot("capture_temporary_nfz")], 3, scenario_id, "UAV reroutes around temporary NFZ", "uav_mission", [uav, nfz], "info"),
            event("temporary_nfz_cleared", tick(520), [visual("set_temporary_nfz_inactive", nfz, "inactive")], 4, scenario_id, "Temporary no-fly zone expires after the reroute window", "dynamic_constraint", [uav, nfz], "info", intent="recovery_or_restore"),
        ]
        desc = "Temporary no-fly zone activation and UAV reroute"
        rules = [{"rule": "dynamic_spawn_required", "entity_id": nfz, "description": "Temporary NFZ is spawned by event, not pre-activated statically"}]
    else:
        uav = f"uav_{sid}"
        hazard = f"hazmat_zone_{sid}"
        ped_a = f"ped_hazmat_{sid}_a"
        ped_b = f"ped_hazmat_{sid}_b"
        ambulance = f"ambulance_{sid}"
        safe_anchor_a = f"hazmat_safe_anchor_{sid}_a"
        safe_anchor_b = f"hazmat_safe_anchor_{sid}_b"
        arrival_anchor = f"hazmat_arrival_anchor_{sid}"
        center = p(ox + 8, oy + 6, 3)
        if scenario_id == "L3-3_v1":
            center = q(center[0] + 0.25, center[1] + 0.08, center[2])
        hazard_radius_m = 8.0 if scenario_id == "L3-3_v1" else 18.0
        hazard_safe_clearance_margin_m = 5.0
        safe_dwell_radius_m = 3.0
        safe_anchor_min_center_distance_m = (
            hazard_radius_m + hazard_safe_clearance_margin_m + safe_dwell_radius_m
        )
        hazard_arrival_tolerance_m = 6.0
        _, hazard_scene = add(specs, scenes, box_entity(hazard, "trigger.hazard.generic.box.v1", "hazard_zone", center, [13, 10, 4], activation_tick=901))
        ped_a_spec, ped_a_scene = add(specs, scenes, sidewalk_entity(ped_a, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_15", 48, 1.2, p(ox + 5, oy + 3, 0), 0, "walking"))
        ped_b_spec, ped_b_scene = add(specs, scenes, sidewalk_entity(ped_b, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_15", 54, 1.2, p(ox + 11, oy + 4, 0), 180, "walking"))

        exposure_anchors: list[PlannedSidewalkAnchor] = []
        for exposure_anchor in _iter_free_sidewalk_anchors(
            center,
            [],
            semantic="hazmat_exposure_sidewalk",
            allow_green=True,
            offset_priority_m=(SIDEWALK_MIN_OFFSET_FROM_CURB_M, 2.4, 3.6, 4.8),
        ):
            if dist_xy(exposure_anchor.position_enu_m, center) > hazard_radius_m:
                continue
            if any(
                dist_xy(exposure_anchor.position_enu_m, other.position_enu_m) < safe_dwell_radius_m
                for other in exposure_anchors
            ):
                continue
            exposure_anchors.append(exposure_anchor)
            if len(exposure_anchors) == 2:
                break
        if len(exposure_anchors) != 2:
            raise RuntimeError(
                f"{scenario_id}: unable to place two separated pedestrians inside the "
                f"{hazard_radius_m:.1f}m active hazmat radius"
            )
        _apply_sidewalk_anchor_to_entity(
            ped_a_spec,
            ped_a_scene,
            exposure_anchors[0],
            placement_semantics="hazmat_exposure_sidewalk",
        )
        _apply_sidewalk_anchor_to_entity(
            ped_b_spec,
            ped_b_scene,
            exposure_anchors[1],
            placement_semantics="hazmat_exposure_sidewalk",
        )
        ped_a_start = scene_pos(ped_a_scene)
        ped_b_start = scene_pos(ped_b_scene)
        ped_a_start, ped_a_context = pedestrian_preamble_walk_event(
            ped_a_spec,
            ped_a_scene,
            scenario_id,
            "hazmat_ped_a_approach",
            "move_hazmat_ped_a_approach",
            dx_m=-14.0,
            dy_m=8.0,
        )
        ped_b_start, ped_b_context = pedestrian_preamble_walk_event(
            ped_b_spec,
            ped_b_scene,
            scenario_id,
            "hazmat_ped_b_approach",
            "move_hazmat_ped_b_approach",
            dx_m=14.0,
            dy_m=8.0,
        )
        def hazmat_safe_sidewalk_point(
            start: list[float],
            pre_event_start: list[float],
            direction: float,
            excluded_points: Sequence[list[float]] = (),
            min_anchor_separation_m: float | None = None,
        ) -> tuple[list[float], str]:
            # Default separation between two distinct safe anchors follows the same
            # physical-chain bound as the hazard clearance. A caller may lower it
            # (never below two non-overlapping dwell balls) when the full bound cannot
            # be met on this street network.
            if min_anchor_separation_m is None:
                min_anchor_separation_m = safe_anchor_min_center_distance_m
            hard_min_separation_m = 2.0 * safe_dwell_radius_m + 2.0
            if min_anchor_separation_m < hard_min_separation_m:
                raise ValueError(
                    f"{scenario_id}: safe-anchor separation {min_anchor_separation_m:.3f}m "
                    f"would overlap the {safe_dwell_radius_m:.1f}m dwell balls"
                )
            errors: list[str] = []

            def reachable_route_mode(point: list[float], context: str) -> str | None:
                try:
                    SPATIAL.plan_sidewalk_route(
                        [start, point],
                        allow_green=True,
                        context=f"{context}_sidewalk",
                    )
                    return "sidewalk"
                except SpatialValidationError as exc:
                    errors.append(str(exc))
                try:
                    SPATIAL.plan_crossing_path(
                        [start, point],
                        context=f"{context}_crosswalk",
                    )
                    return "crosswalk"
                except SpatialValidationError as exc:
                    errors.append(str(exc))
                return None

            def clears_physical_bounds(point: list[float]) -> bool:
                center_distance_m = dist_xy(point, center)
                if center_distance_m + 1e-6 < safe_anchor_min_center_distance_m:
                    errors.append(
                        f"candidate {point} center distance {center_distance_m:.3f} below "
                        f"safe-anchor bound {safe_anchor_min_center_distance_m:.3f}"
                    )
                    return False
                approach_clearance_m = float(
                    LineString(
                        [
                            (float(pre_event_start[0]), float(pre_event_start[1])),
                            (float(start[0]), float(start[1])),
                        ]
                    ).distance(Point(float(point[0]), float(point[1])))
                )
                if approach_clearance_m <= safe_dwell_radius_m:
                    errors.append(
                        f"candidate {point} is only {approach_clearance_m:.3f}m from the "
                        "pre-event approach path"
                    )
                    return False
                for excluded_point in excluded_points:
                    anchor_separation_m = dist_xy(point, excluded_point)
                    if anchor_separation_m + 1e-6 < min_anchor_separation_m:
                        errors.append(
                            f"candidate {point} is only {anchor_separation_m:.3f}m from an "
                            f"already placed safe anchor {excluded_point}; safe anchors "
                            f"require at least {min_anchor_separation_m:.3f}m"
                        )
                        return False
                return True

            for dx_m, dy_m, offset_m in (
                (direction * 12.0, 12.0, 2.0),
                (direction * 18.0, 12.0, 2.0),
                (direction * 18.0, -12.0, 2.0),
                (direction * 14.0, 18.0, 4.0),
                (direction * 10.0, -18.0, 4.0),
                (0.0, 18.0, 5.0),
                (0.0, -18.0, 5.0),
            ):
                candidate_points: list[list[float]] = []
                try:
                    candidate_points.append(shifted_sidewalk_point(start, dx_m, dy_m, offset_m))
                except RuntimeError as exc:
                    errors.append(str(exc))
                try:
                    candidate_points.append(
                        preamble_sidewalk_start(
                            start,
                            dx_m,
                            dy_m,
                            offset_m,
                            min_span_m=hazard_radius_m + hazard_safe_clearance_margin_m,
                            max_span_m=hazard_radius_m + 24.0,
                        )
                    )
                except RuntimeError as exc:
                    errors.append(str(exc))
                if not candidate_points:
                    continue
                for point in candidate_points:
                    if not clears_physical_bounds(point):
                        continue
                    route_mode = reachable_route_mode(point, "hazmat_safe_target_reachability")
                    if route_mode is None:
                        continue
                    return point, route_mode
            base_sample = LANES.nearest_to_xy(float(start[0]), float(start[1]))
            desired_sign = _side_sign_for_desired(base_sample, start)
            min_s, max_s = LANES.edge_s_bounds(base_sample.edge_id)
            for sign in (desired_sign, -desired_sign):
                for s_delta in (
                    direction * 24.0,
                    direction * 32.0,
                    direction * 40.0,
                    direction * 55.0,
                    direction * 70.0,
                    -direction * 24.0,
                    -direction * 32.0,
                    -direction * 40.0,
                    -direction * 55.0,
                    -direction * 70.0,
                ):
                    sample = LANES.resolve_edge_s(
                        base_sample.edge_id,
                        max(min_s, min(max_s, base_sample.s_m + s_delta)),
                    )
                    for offset_m in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
                        point = _offset_from_lane(
                            sample,
                            sign * (LANE_HALF_WIDTH_M + offset_m),
                            GROUND_Z_M,
                        )
                        if not clears_physical_bounds(point):
                            continue
                        route_mode = reachable_route_mode(point, "hazmat_safe_target_lane_scan")
                        if route_mode is None:
                            continue
                        return point, route_mode
            raise RuntimeError(f"{scenario_id}: no hazmat pedestrian safe target satisfies physical hazard clearance: {errors[:4]}")

        ped_a_safe, ped_a_safe_route_mode = hazmat_safe_sidewalk_point(
            ped_a_start,
            scene_pos(ped_a_scene),
            -1.0,
        )
        # The two anchors name distinct refuges. Selecting pedestrian B's target with
        # pedestrian A's resolved anchor excluded prevents both evacuation terminals
        # from collapsing onto one shared coordinate. The full physical-chain
        # separation is tried first; if this street network cannot host two fully
        # separated refuges, a reduced (still non-overlapping dwell balls) separation
        # is the explicit fallback.
        try:
            ped_b_safe, ped_b_safe_route_mode = hazmat_safe_sidewalk_point(
                ped_b_start,
                scene_pos(ped_b_scene),
                1.0,
                excluded_points=[ped_a_safe],
            )
            ped_b_safe_separation_m = safe_anchor_min_center_distance_m
        except RuntimeError:
            ped_b_safe, ped_b_safe_route_mode = hazmat_safe_sidewalk_point(
                ped_b_start,
                scene_pos(ped_b_scene),
                1.0,
                excluded_points=[ped_a_safe],
                min_anchor_separation_m=2.0 * safe_dwell_radius_m + 2.0,
            )
            ped_b_safe_separation_m = 2.0 * safe_dwell_radius_m + 2.0
            safe_anchor_separation_relaxed_rule = {
                "rule": "hazmat_safe_anchor_separation_relaxed",
                "entity_id": safe_anchor_b,
                "min_center_separation_m": ped_b_safe_separation_m,
                "description": (
                    "Street network cannot host two anchors at the full "
                    f"{safe_anchor_min_center_distance_m:.1f}m separation; B's anchor "
                    "uses the minimum separation that keeps both dwell balls disjoint"
                ),
            }
        else:
            safe_anchor_separation_relaxed_rule = None
        ped_a_safe_action = "move_ped_a_safe_crosswalk" if ped_a_safe_route_mode == "crosswalk" else "move_ped_a_safe"
        ped_b_safe_action = "move_ped_b_safe_crosswalk" if ped_b_safe_route_mode == "crosswalk" else "move_ped_b_safe"
        add(specs, scenes, world_entity(safe_anchor_a, "semantic.spawn_zone", "crowd_anchor", ped_a_safe, 0, "safe_zone_reference"))
        add(specs, scenes, world_entity(safe_anchor_b, "semantic.spawn_zone", "crowd_anchor", ped_b_safe, 0, "safe_zone_reference"))
        def hazmat_same_lane_ambulance_corridor() -> tuple[str, float, float, list[list[float]]]:
            """Select an outward-to-perimeter response route on one lawful road lane."""

            candidates: list[tuple[float, float, str, float, float, list[list[float]]]] = []
            seen_arrivals: set[tuple[str, int]] = set()
            for arrival_sample in LANES.samples:
                edge_id = str(arrival_sample.edge_id)
                if edge_id.startswith(":"):
                    continue
                sumo_edge = SUMO_GROUND_FLOW.edges.get(edge_id)
                if sumo_edge is None or not SUMO_GROUND_FLOW._edge_allows(sumo_edge, "vehicle"):
                    continue
                arrival_point = _offset_from_lane(arrival_sample, 0.0, GROUND_Z_M)
                arrival_distance = dist_xy(arrival_point, center)
                if not (
                    hazard_radius_m
                    <= arrival_distance
                    <= hazard_radius_m + hazard_arrival_tolerance_m
                ):
                    continue
                arrival_key = (edge_id, int(round(float(arrival_sample.s_m))))
                if arrival_key in seen_arrivals:
                    continue
                seen_arrivals.add(arrival_key)
                min_s, max_s = LANES.edge_s_bounds(edge_id)
                sumo_centerline = LineString(sumo_edge.shape_xy)
                # SUMO lane position increases in the legal travel direction.
                # The responder must therefore start at a lower lane position
                # and approach the perimeter with increasing ``s``; reverse
                # source motion must never be repaired later by reordering data.
                for direction in (-1.0,):
                    available_m = (
                        float(arrival_sample.s_m) - float(min_s)
                        if direction < 0.0
                        else float(max_s) - float(arrival_sample.s_m)
                    )
                    for route_length_m in (35.0, 30.0, 25.0, 20.0, 15.0):
                        if available_m + 1e-6 < route_length_m:
                            continue
                        start_s = float(arrival_sample.s_m) + direction * route_length_m
                        step_count = max(2, int(math.ceil(route_length_m)))
                        route_samples = [
                            LANES.resolve_edge_s(
                                edge_id,
                                start_s
                                + (float(arrival_sample.s_m) - start_s)
                                * float(step_index)
                                / float(step_count),
                            )
                            for step_index in range(step_count + 1)
                        ]
                        route_points = [
                            _offset_from_lane(sample, 0.0, GROUND_Z_M)
                            for sample in route_samples
                        ]
                        route_distances = [dist_xy(point, center) for point in route_points]
                        if min(route_distances) + 1e-6 < hazard_radius_m:
                            continue
                        if route_distances[0] < hazard_radius_m + hazard_arrival_tolerance_m + 2.0:
                            continue
                        projection_errors = [
                            float(sumo_centerline.distance(Point(point[0], point[1])))
                            for point in route_points
                        ]
                        if max(projection_errors, default=float("inf")) > 1.0:
                            continue
                        candidates.append(
                            (
                                -route_length_m,
                                abs(arrival_distance - (hazard_radius_m + 2.0)),
                                edge_id,
                                float(arrival_sample.s_m),
                                start_s,
                                route_points,
                            )
                        )
                        break
            if not candidates:
                raise RuntimeError(
                    f"{scenario_id}: no same-lane ambulance route reaches the hazard outer annulus "
                    "without entering the active hazard radius"
                )
            _neg_length, _annulus_error, edge_id, arrival_s, start_s, route_points = min(candidates)
            return edge_id, start_s, arrival_s, _dedupe_route_points(route_points)

        ambulance_edge_id, ambulance_s_m, ambulance_arrival_s_m, ambulance_dispatch_route = (
            hazmat_same_lane_ambulance_corridor()
        )
        ambulance_hint = list(ambulance_dispatch_route[0])
        _, ambulance_scene = add(
            specs,
            scenes,
            lane_entity(
                ambulance,
                "vehicle.emergency.ambulance.v1",
                "vehicle",
                ambulance_edge_id,
                ambulance_s_m,
                0.0,
                ambulance_hint,
                70,
                "response",
                activation_tick=901,
                visual_state={"mode": "response", "lights_on": True},
                prefer_edge_hint=True,
            ),
        )
        ambulance_start = scene_pos(ambulance_scene)
        ambulance_arrival = _offset_from_lane(
            LANES.resolve_edge_s(ambulance_edge_id, ambulance_arrival_s_m),
            0.0,
            GROUND_Z_M,
        )
        ambulance_dispatch_route[0] = list(ambulance_start)
        ambulance_dispatch_route[-1] = list(ambulance_arrival)
        add(specs, scenes, world_entity(arrival_anchor, "semantic.spawn_zone", "vehicle_anchor", ambulance_arrival, 0, "hazmat_arrival_reference"))
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", p(ox - 8, oy - 15, 33), 45, "monitor"))
        if scenario_id == "L3-3_v1":
            weather_profile = {
                "initial": "clear",
                "transitions": [
                    {
                        "tick": 235,
                        "profile": "clear",
                        "overrides": {
                            "hazard_source_active": True,
                            "hazard_concentration_ppm": 120.0,
                            "hazard_radius_m": hazard_radius_m,
                        },
                    },
                    {
                        "tick": 520,
                        "profile": "clear",
                        "overrides": {
                            "hazard_source_active": False,
                            "hazard_concentration_ppm": 0.0,
                            "hazard_radius_m": 0.0,
                        },
                    },
                ],
            }
        else:
            weather_profile = {
                "initial": "clear",
                "transitions": [
                    {
                        "tick": 220,
                        "profile": "clear",
                        "overrides": {
                            "hazard_source_active": True,
                            "hazard_concentration_ppm": 85.0,
                            "hazard_radius_m": 7.0,
                        },
                    },
                    {
                        "tick": 235,
                        "profile": "clear",
                        "overrides": {
                            "hazard_source_active": True,
                            "hazard_concentration_ppm": 135.0,
                            "hazard_radius_m": hazard_radius_m,
                        },
                    },
                    {
                        "tick": 520,
                        "profile": "clear",
                        "overrides": {
                            "hazard_source_active": False,
                            "hazard_concentration_ppm": 0.0,
                            "hazard_radius_m": 0.0,
                        },
                    },
                ],
            }
        safe_condition_event_ids = ["hazmat_ped_a_safe_dwell", "hazmat_ped_b_safe_dwell"]
        events = [
            ped_a_context,
            ped_b_context,
            event("hazmat_leak", tick(240), [spawn_from_scene("spawn_hazmat_zone", hazard_scene, visual_state={"mode": "isolation_active"})], 1, scenario_id, "Hazmat leak declares isolation zone", "dynamic_constraint", [hazard], "critical"),
            event("isolation_setup", fired("hazmat_leak"), [screenshot("capture_hazmat_isolation")], 2, scenario_id, "Hazmat isolation perimeter is established", "dynamic_constraint", [hazard], "warning"),
            event("pedestrian_evacuation", fired("isolation_setup"), [move(ped_a_safe_action, ped_a, [ped_a_start, ped_a_safe], 1.6, activity_type="evacuating"), move(ped_b_safe_action, ped_b, [ped_b_start, ped_b_safe], 1.4, activity_type="evacuating")], 3, scenario_id, "Pedestrians evacuate from hazmat zone", "pedestrian", [ped_a, ped_b, hazard], "warning"),
            context_event("hazmat_ped_a_safe_dwell", prox(ped_a, safe_anchor_a, safe_dwell_radius_m, 10, metric="xy_plus_z", horizontal_distance_m=safe_dwell_radius_m, vertical_distance_m=5.0), [], 4, scenario_id, "First target pedestrian remains at the physical safe-zone anchor", [ped_a, safe_anchor_a]),
            context_event("hazmat_ped_b_safe_dwell", prox(ped_b, safe_anchor_b, safe_dwell_radius_m, 10, metric="xy_plus_z", horizontal_distance_m=safe_dwell_radius_m, vertical_distance_m=5.0), [], 4, scenario_id, "Second target pedestrian remains at the physical safe-zone anchor", [ped_b, safe_anchor_b]),
            event("pedestrian_safe_zone_reached", composite("AND", safe_condition_event_ids), [screenshot("capture_hazmat_safe_zone")], 4, scenario_id, "Both target pedestrians satisfy the physical safe-zone dwell threshold", "pedestrian", [ped_a, ped_b, hazard, safe_anchor_a, safe_anchor_b], "info"),
            event("ambulance_dispatch", fired("pedestrian_safe_zone_reached"), [spawn_from_scene(f"spawn_{ambulance}", ambulance_scene, visual_state={"mode": "response", "lights_on": True}), move("move_ambulance_hazmat_dispatch", ambulance, ambulance_dispatch_route, 11.0)], 5, scenario_id, "Ambulance is dispatched and departs toward isolation perimeter", "vehicle", [ambulance, hazard], "warning", validation_event_type=PEDESTRIAN_VEHICLE_INTERACTION_EVENT_TYPE, validation_reason="Hazmat response intentionally brings an emergency vehicle close to evacuating pedestrians at the isolation perimeter.", validation_skip_checks=[PEDESTRIAN_VEHICLE_CLEARANCE_CHECK]),
            event("ambulance_arrived", prox(ambulance, arrival_anchor, 3.0, 10, metric="xy"), [screenshot("capture_hazmat_ambulance_arrived")], 6, scenario_id, "Ambulance physically dwells at the isolation perimeter arrival anchor", "vehicle", [ambulance, hazard, arrival_anchor], "warning", validation_event_type=PEDESTRIAN_VEHICLE_INTERACTION_EVENT_TYPE, validation_reason="Hazmat arrival is checked after deterministic travel completion at the isolation perimeter.", validation_skip_checks=[PEDESTRIAN_VEHICLE_CLEARANCE_CHECK]),
            event("uav_external_monitor", fired("ambulance_arrived", delay_ticks=15), [move("move_uav_hazmat_orbit", uav, [p(ox - 8, oy - 15, 33), p(ox + 2, oy - 5, 30), p(ox + 18, oy + 18, 30)], 5.0), screenshot("capture_hazmat_monitor")], 7, scenario_id, "UAV monitors hazmat zone after responder handoff dwell", "uav_mission", [uav, hazard, ambulance], "info"),
            context_event("hazmat_clearance_observation_window", fired("uav_external_monitor", delay_ticks=5), [], 8, scenario_id, "Clearance observation opens after the weather profile clears and responder monitoring handoff completes", [hazard]),
            event("hazmat_resolved", composite("AND", ["hazmat_clearance_observation_window", "uav_external_monitor"]), [screenshot("capture_hazmat_resolved")], 8, scenario_id, "Hazmat clearance observation is documented after physical response gates", "dynamic_constraint", [hazard, ambulance, uav, safe_anchor_a, safe_anchor_b, arrival_anchor], "info"),
        ]
        for safe_event_id in safe_condition_event_ids:
            next(item for item in events if item["event_id"] == safe_event_id)[
                "require_conditions"
            ] = ["pedestrian_evacuation"]
        desc = "Hazmat leak isolation, evacuation, and UAV monitoring"
        rules = [
            {"rule": "hazmat_min_entities", "pedestrian_min": 2, "ambulance_min": 1, "uav_min": 1, "description": "Hazmat scenario includes pedestrians, ambulance, and UAV"},
            {"rule": "hazmat_physical_chain", "hazard_radius_m": hazard_radius_m, "safe_clearance_margin_m": hazard_safe_clearance_margin_m, "safe_dwell_radius_m": safe_dwell_radius_m, "safe_anchor_min_center_distance_m": safe_anchor_min_center_distance_m, "safe_start_max_center_distance_m": hazard_radius_m, "safe_dwell_ticks": 10, "ambulance_arrival_dwell_ticks": 10, "handoff_dwell_ticks": 15, "arrival_tolerance_m": hazard_arrival_tolerance_m, "description": "Hazmat response gates on physical evacuation from the active radius into a dwell ball whose full extent clears the safety perimeter"},
        ]
        if safe_anchor_separation_relaxed_rule is not None:
            rules.append(safe_anchor_separation_relaxed_rule)
    return make_bundle(
        scenario_id,
        l_path("L3_dynamic_constraints", "interaction", scenario_id),
        "dynamic_constraints",
        desc,
        specs,
        scenes,
        {"incident_tick": 240 if scenario_id.startswith("L3-3") else 220},
        events,
        weather_profile=weather_profile,
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
        events[1]["require_conditions"] = ["converging_approach"]
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
            event("facade_approach", tick(250), [move("move_uav_facade_near_miss_approach", uav, [start, p(ox - 12, oy - 2, 26), near], 7.0)], 1, scenario_id, "UAV approaches building facade on faulty route", "uav_mission", [uav, facade], "warning"),
            event("facade_proximity", prox(uav, facade, 8.0, 3, metric="xy"), [move("move_uav_facade_near_miss_pull_up", uav, [near, p(ox - 4, oy + 14, 34)], 10.0), screenshot("capture_facade_evasion")], 2, scenario_id, "Emergency evasion near facade", "uav_mission", [uav, facade], "critical"),
            event("facade_recovery", fired("facade_proximity"), [move("move_uav_facade_recovery", uav, [p(ox - 4, oy + 14, 34), p(ox - 22, oy + 18, 32)], 7.0)], 3, scenario_id, "UAV recovers after a facade near miss", "uav_mission", [uav], "info"),
        ]
        next(item for item in events if item["event_id"] == "facade_proximity")["require_conditions"] = [
            "facade_approach"
        ]
        desc = "UAV building facade near miss with evasive recovery"
        rules.append({"rule": "facade_approach_distance", "max_distance_m": 3.0, "description": "UAV trajectory nearest point is within 3m of facade anchor"})
    elif scenario_id.startswith("L4-3_"):
        uav = f"uav_forced_landing_{sid}"
        ped_a = f"ped_landing_{sid}_a"
        ped_b = f"ped_landing_{sid}_b"
        ped_a_spec, ped_a_scene = add(specs, scenes, sidewalk_entity(ped_a, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_19", 24, GATHERING_MIN_OFFSET_FROM_CURB_M, p(ox + 2, oy + 2, 0), 0, "chatting"))
        ped_b_spec, ped_b_scene = add(specs, scenes, sidewalk_entity(ped_b, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_19", 36, GATHERING_MIN_OFFSET_FROM_CURB_M, p(ox + 11, oy + 2, 0), 180, "chatting"))
        ped_a_start = scene_pos(ped_a_scene)
        ped_b_start = scene_pos(ped_b_scene)
        if dist_xy(ped_a_start, ped_b_start) < 1.0:
            base_placement = dict(ped_a_scene.get("placement") or {})
            anchor = SPATIAL.plan_sidewalk_anchor(
                ped_a_start,
                edge_id_hint=str(base_placement.get("lane_edge_id") or ""),
                s_hint=float(base_placement.get("longitudinal_s") or 0.0) + 5.0,
                offset_from_curb_m=GATHERING_MIN_OFFSET_FROM_CURB_M,
                allow_green=True,
                placement_semantics="sidewalk",
            )
            ped_b_start = anchor.position_enu_m
            update_world_entity_position(ped_b_spec, ped_b_scene, ped_b_start)
            ped_b_scene["placement"] = {
                **dict(ped_b_scene.get("placement") or {}),
                "lane_edge_id": anchor.sample.edge_id,
                "longitudinal_s": round(anchor.sample.s_m, 3),
                "offset_from_curb_m": round(anchor.offset_from_curb_m, 3),
                "lane_half_width_m": LANE_HALF_WIDTH_M,
                "resolved_lateral_from_center_m": round(anchor.resolved_lateral_from_center_m, 3),
                "placement_semantics": "sidewalk",
                "resolved_position_enu_m": anchor.position_enu_m,
                "rotation_deg": rot(anchor.sample.yaw_deg),
            }
        landing_seed = q((ped_a_start[0] + ped_b_start[0]) / 2.0, (ped_a_start[1] + ped_b_start[1]) / 2.0, GROUND_Z_M)
        landing_anchor = _find_forced_landing_ground_anchor(landing_seed, scenario_id=scenario_id)
        landing_point = landing_anchor.position_enu_m
        if dist_xy(landing_point, ped_a_start) > 7.0 or dist_xy(landing_point, ped_b_start) > 9.0:
            occupied_peds: list[list[float]] = []
            ped_a_anchor = _pedestrian_anchor_near_landing(
                landing_anchor,
                landing_point,
                preferred_s_delta_m=-4.5,
                occupied=occupied_peds,
            )
            ped_b_anchor = _pedestrian_anchor_near_landing(
                landing_anchor,
                landing_point,
                preferred_s_delta_m=4.5,
                occupied=occupied_peds,
            )
            _apply_sidewalk_anchor_to_entity(ped_a_spec, ped_a_scene, ped_a_anchor)
            _apply_sidewalk_anchor_to_entity(ped_b_spec, ped_b_scene, ped_b_anchor)
            ped_a_start = scene_pos(ped_a_scene)
            ped_b_start = scene_pos(ped_b_scene)
        ped_a_start, ped_a_context = pedestrian_preamble_walk_event(
            ped_a_spec,
            ped_a_scene,
            scenario_id,
            "landing_ped_a_approach",
            "move_landing_ped_a_approach",
            dx_m=-12.0,
            dy_m=9.0,
        )
        ped_b_start, ped_b_context = pedestrian_preamble_walk_event(
            ped_b_spec,
            ped_b_scene,
            scenario_id,
            "landing_ped_b_approach",
            "move_landing_ped_b_approach",
            dx_m=12.0,
            dy_m=9.0,
        )
        landing_x = landing_point[0]
        landing_y = landing_point[1]
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
            ped_a_context,
            ped_b_context,
            context_event("crowd_present", tick(150), [ped_activity(f"set_{ped_id}_crowd_chatting", ped_id, "chatting") for ped_id in crowd_ids], 0, scenario_id, "Ground crowd is visible before the UAV fault begins", crowd_ids),
            event("forced_landing_fault", tick(230), [visual("set_uav_forced_landing_fault", uav, "propulsion_fault")], 1, scenario_id, "UAV fault appears near a ground crowd", "uav_mission", [uav, ped_a, ped_b], "critical"),
            event("descent_to_low_altitude", fired("forced_landing_fault"), [move("move_uav_forced_descent", uav, [start, descent_mid, low, land], FORCED_LANDING_DESCENT_SPEED_MPS, "descending", "low_altitude_hold")], 2, scenario_id, "UAV descends from 30m to low height", "uav_mission", [uav], "critical"),
            event("crowd_proximity_response", prox(uav, ped_a, 7.5, 2, metric="xy_plus_z", horizontal_distance_m=7.5, vertical_distance_m=10.0), [move("move_crowd_ped_a_evade_landing", ped_a, [ped_a_start, ped_a_evade], 2.0, post_activity_type="quarrel"), move("move_crowd_ped_b_evade_landing", ped_b, [ped_b_start, ped_b_evade], 2.0, post_activity_type="quarrel"), screenshot("capture_forced_landing")], 3, scenario_id, "Crowd evades forced landing zone and yells at the forced landing site", "pedestrian", [ped_a, ped_b, uav], "warning"),
            event("safe_landing", fired("crowd_proximity_response"), [move("move_uav_safe_touchdown", uav, [land, q(land[0], land[1], 0.8)], FORCED_LANDING_TOUCHDOWN_SPEED_MPS, "descending", "landed")], 4, scenario_id, "UAV completes safe landing", "uav_mission", [uav], "info", intent="landing_or_terminal_resolution"),
        ]
        desc = "UAV forced landing near crowd"
        rules.append({"rule": "forced_landing_descent_profile", "z_sequence": [30, 8, 3], "description": "UAV descends through low-altitude forced landing profile"})
    elif scenario_id.startswith("L4-4_"):
        uav = f"uav_falls_vehicle_{sid}"
        car = f"car_under_uav_{sid}"
        u0 = p(ox - 18, oy - 5, 20)
        c0 = p(ox + 4, oy - 18, 0)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", u0, 70, "unstable"))
        uav_scene = next(s for s in scenes if s["entity_id"] == uav)
        _, car_scene = add(specs, scenes, lane_entity(car, "vehicle.ground.boxcar.v1", "vehicle", "cg_edge_20", 46, 0, c0, 0, "moving", prefer_edge_hint=True))
        car_start = scene_pos(car_scene)
        car_placement = dict(car_scene.get("placement") or {})
        car_edge_id = str(car_placement.get("edge_id") or "")
        car_s = float(car_placement.get("longitudinal_s") or 0.0)
        car_lateral = float(car_placement.get("resolved_lateral_from_center_m") or 0.0)
        car_min_s, car_max_s = 0.0, SUMO_GROUND_FLOW.edge_shape_length_m(car_edge_id)
        contact_s: float | None = None
        impact_c: list[float] | None = None
        impact_u: list[float] | None = None
        for delta_s in (18.0, 24.0, 30.0, 36.0, 44.0, 54.0, 66.0, 80.0, -18.0, -30.0, -42.0):
            candidate_s = max(car_min_s, min(car_max_s, car_s + delta_s))
            candidate_ground = _sumo_vehicle_point_at_s(car_edge_id, candidate_s, car_lateral)
            candidate_uav = q(candidate_ground[0], candidate_ground[1], 2.6)
            candidate_mid = q((u0[0] + candidate_uav[0]) / 2.0, (u0[1] + candidate_uav[1]) / 2.0, 12)
            try:
                repaired = _repair_uav_profile_waypoints(scenario_id, "move_uav_falling_cross", [u0, candidate_mid, candidate_uav])
            except RuntimeError:
                continue
            if repaired and dist_xy(repaired[-1], candidate_uav) <= 0.25 and abs(float(repaired[-1][2]) - candidate_uav[2]) <= 0.1:
                contact_s = candidate_s
                impact_c = candidate_ground
                impact_u = candidate_uav
                break
        if impact_c is None or impact_u is None or contact_s is None:
            raise RuntimeError(f"{scenario_id}: unable to find building-clear lane contact point for UAV-vehicle crossing")
        crossing_mid_s = max(car_min_s, min(car_max_s, (car_s + contact_s) / 2.0))
        crossing_mid = _sumo_vehicle_point_at_s(car_edge_id, crossing_mid_s, car_lateral)
        brake_end = _sumo_vehicle_point_at_s(
            car_edge_id,
            max(car_min_s, min(car_max_s, contact_s + 3.0)),
            car_lateral,
        )
        debris_ground = q(impact_c[0] + 2.0, impact_c[1] + 1.0, 0.4)
        events = [
            event("crossing_trajectories", tick(240), [move("move_uav_falling_cross", uav, [u0, q((u0[0] + impact_u[0]) / 2.0, (u0[1] + impact_u[1]) / 2.0, 12), impact_u], 5.5), move("move_car_crossing_path", car, [car_start, crossing_mid, impact_c], 7.0)], 1, scenario_id, "UAV and vehicle trajectories cross", "uav_mission", [uav, car], "critical"),
            event("vehicle_roof_contact", prox(uav, car, 3.0, 1), [move("move_vehicle_emergency_brake", car, [impact_c, brake_end], 0.5), move("move_uav_debris_ground", uav, [impact_u, debris_ground], 1.0), visual("set_uav_debris", uav, "debris"), screenshot("capture_uav_vehicle_contact")], 2, scenario_id, "UAV contacts vehicle roof, is damaged, and the vehicle brakes", "vehicle", [uav, car], "critical"),
        ]
        # The aircraft is damaged and grounded: it must not be re-scheduled as airworthy.
        # The scene carries that terminal state so the automatic lifecycle still emits the
        # take-off that starts the flight but emits no return-to-pad landing afterwards.
        uav_scene["lifecycle"] = {**(uav_scene.get("lifecycle") or {}),
                                  "terminal_flight_state": True}
        spawn_event_actor_at_dispatch(car_scene, events[0])
        desc = "UAV falls onto moving ground vehicle"
        rules.append({"rule": "trajectory_intersection_required", "description": "UAV and vehicle trajectories intersect rather than running parallel"})
        rules.append({"rule": "uav_contact_terminates_flight", "entity_id": uav, "description": "A roof contact damages and grounds the UAV; the post-contact process is vehicle emergency response, not a return-to-pad flight"})
    elif scenario_id.startswith("L4-5_"):
        uav = f"uav_ped_nearmiss_{sid}"
        ped = f"ped_nearmiss_{sid}"
        ped_spec, ped_scene = add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_21", 35, 1.2, p(ox + 2, oy + 1, 0), 15, "walking"))
        ped_start = scene_pos(ped_scene)
        ped_start, ped_context = pedestrian_preamble_walk_event(
            ped_spec,
            ped_scene,
            scenario_id,
            "nearmiss_ped_approach",
            "move_nearmiss_ped_approach",
            dx_m=-12.0,
            dy_m=7.0,
        )
        start = q(ped_start[0] - 1.394, ped_start[1] + 15.939, 30)
        descent_mid = q(ped_start[0] - 0.697, ped_start[1] + 7.97, 15)
        over = q(ped_start[0], ped_start[1], 5)
        # The UAV holds at the low point instead of climbing away: the risk must be
        # resolved by the pedestrian leaving, not by the aircraft retreating, or the
        # recovery would precede the evasive act and the causal order would be false.
        hold_low = q(ped_start[0], ped_start[1], 5)
        ped_clear = shifted_sidewalk_point(ped_start, 18, 18, 1.8)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 45, "descent"))
        events = [
            ped_context,
            event("uav_low_descent", tick(230), [ped_activity("set_ped_phone_distraction", ped, "phone_call"), move("move_uav_ped_descent", uav, [start, descent_mid, over], 4.5)], 1, scenario_id, "UAV descends toward pedestrian head height while pedestrian is distracted by a phone call", "uav_mission", [uav, ped], "warning"),
            event("pedestrian_near_miss", prox(uav, ped, 5.0, 2, metric="xy_plus_z", horizontal_distance_m=3.0, vertical_distance_m=6.0), [visual("set_uav_hover_nearmiss", uav, "hover"), move("move_uav_hold_low_after_nearmiss", uav, [over, hold_low], 1.0), screenshot("capture_ped_nearmiss")], 2, scenario_id, "UAV near-miss with pedestrian holds at low altitude pending clearance", "uav_mission", [uav, ped], "critical"),
            event("pedestrian_clears_area", fired("pedestrian_near_miss"), [move("move_ped_clear_nearmiss", ped, [ped_start, ped_clear], 1.5)], 3, scenario_id, "Pedestrian clears UAV operating area", "pedestrian", [ped], "info"),
        ]
        events[1]["require_conditions"] = ["uav_low_descent"]
        desc = "UAV low-altitude pedestrian near-miss"
    elif scenario_id.startswith("L4-6_"):
        ped = f"ped_jaywalk_{sid}"
        car = f"car_jaywalk_{sid}"

        def l4_6_same_edge_route(
            edge_id: str,
            start_s_m: float,
            end_s_m: float,
            *,
            spacing_m: float,
            lateral_m: float,
        ) -> list[list[float]]:
            if end_s_m + 1e-6 < start_s_m:
                raise RuntimeError(
                    f"{scenario_id}: L4-6 vehicle route must advance in SUMO edge direction, "
                    f"got {start_s_m:.3f}->{end_s_m:.3f}"
                )
            segment_count = max(1, int(math.ceil((end_s_m - start_s_m) / spacing_m)))
            return _dedupe_route_points(
                [
                    _sumo_vehicle_point_at_s(
                        edge_id,
                        start_s_m + (end_s_m - start_s_m) * index / segment_count,
                        lateral_m,
                    )
                    for index in range(segment_count + 1)
                ]
            )

        def l4_6_physical_conflict_corridor() -> tuple[str, Any, PlannedSidewalkAnchor]:
            desired_conflict = p(ox, oy, GROUND_Z_M)
            failures: list[str] = []
            for nearest_distance_m, edge_id, _shape_index, _projected_xy in SUMO_GROUND_FLOW._nearest_edges(
                desired_conflict,
                "vehicle",
            ):
                edge = SUMO_GROUND_FLOW.edges.get(edge_id)
                if edge is None or edge_id not in LANES.by_edge:
                    continue
                if not SUMO_GROUND_FLOW._edge_allows(edge, "vehicle"):
                    continue
                min_s_m, max_s_m = LANES.edge_s_bounds(edge_id)
                if max_s_m - min_s_m < 40.0:
                    continue
                sumo_line = LineString(edge.shape_xy)
                nearest_sample = min(
                    LANES.by_edge[edge_id],
                    key=lambda sample: (sample.x_m - desired_conflict[0]) ** 2
                    + (sample.y_m - desired_conflict[1]) ** 2,
                )
                for delta_s_m in (0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0):
                    conflict_s_m = max(
                        min_s_m + 32.0,
                        min(max_s_m - 5.0, nearest_sample.s_m + delta_s_m),
                    )
                    conflict_point = _offset_from_lane(
                        LANES.resolve_edge_s(edge_id, conflict_s_m),
                        0.0,
                        GROUND_Z_M,
                    )
                    if sumo_line.distance(Point(conflict_point[0], conflict_point[1])) > 3.0:
                        continue
                    try:
                        crossing = SPATIAL.plan_crossing_route(
                            conflict_point,
                            offset_from_curb_m=SIDEWALK_MIN_OFFSET_FROM_CURB_M,
                        )
                    except SpatialValidationError as exc:
                        failures.append(str(exc))
                        continue
                    if crossing.sample.edge_id != edge_id:
                        continue
                    conflict_s_m = float(crossing.sample.s_m)
                    if conflict_s_m - min_s_m < 32.0:
                        continue
                    if (
                        sumo_line.distance(
                            Point(
                                crossing.roadway_center_position_enu_m[0],
                                crossing.roadway_center_position_enu_m[1],
                            )
                        )
                        > 3.0
                    ):
                        continue
                    safe_s_m = max(min_s_m, conflict_s_m - 12.0)
                    safe_sample = LANES.resolve_edge_s(edge_id, safe_s_m)
                    safe_desired = _offset_from_lane(
                        safe_sample,
                        float(crossing.resolved_lateral_from_center_m),
                        GROUND_Z_M,
                    )
                    try:
                        safe_anchor = SPATIAL.plan_sidewalk_anchor(
                            safe_desired,
                            edge_id_hint=edge_id,
                            s_hint=safe_s_m,
                            offset_from_curb_m=float(crossing.offset_from_curb_m),
                            allow_green=False,
                            placement_semantics="sidewalk_safe_retreat",
                        )
                    except SpatialValidationError as exc:
                        failures.append(str(exc))
                        continue
                    if safe_anchor.sample.edge_id != edge_id:
                        continue
                    if (
                        float(safe_anchor.resolved_lateral_from_center_m)
                        * float(crossing.resolved_lateral_from_center_m)
                        <= 0.0
                    ):
                        continue
                    return edge_id, crossing, safe_anchor
            raise RuntimeError(
                f"{scenario_id}: no legal SUMO vehicle edge with a physical sidewalk crossing; "
                f"first_errors={failures[:3]}"
            )

        conflict_edge_id, crossing, safe_anchor = l4_6_physical_conflict_corridor()
        conflict_s_m = float(crossing.sample.s_m)
        physical_lane_offsets = _physical_vehicle_lane_offsets(crossing.sample)
        crossing_side = math.copysign(
            1.0,
            float(crossing.resolved_lateral_from_center_m),
        )
        same_side_offsets = [
            offset_m
            for offset_m in physical_lane_offsets
            if offset_m * crossing_side > 0.0
        ]
        vehicle_lateral_m = (
            min(
                same_side_offsets,
                key=lambda offset_m: abs(
                    float(crossing.resolved_lateral_from_center_m) - offset_m
                ),
            )
            if same_side_offsets
            else min(physical_lane_offsets, key=abs)
        )
        edge_min_s_m, _edge_max_s_m = LANES.edge_s_bounds(conflict_edge_id)
        car_brake_start_s_m = conflict_s_m - 4.0
        car_stop_s_m = conflict_s_m - 2.5
        car_start_s_m = car_brake_start_s_m - 25.0
        if car_start_s_m < edge_min_s_m:
            raise RuntimeError(
                f"{scenario_id}: selected L4-6 edge lacks the required 25 m directed approach"
            )
        car_approach_route = l4_6_same_edge_route(
            conflict_edge_id,
            car_start_s_m,
            car_brake_start_s_m,
            spacing_m=3.0,
            lateral_m=vehicle_lateral_m,
        )
        car_brake_route = l4_6_same_edge_route(
            conflict_edge_id,
            car_brake_start_s_m,
            car_stop_s_m,
            spacing_m=0.75,
            lateral_m=vehicle_lateral_m,
        )
        car_start = list(car_approach_route[0])
        road = _sumo_vehicle_point_at_s(
            conflict_edge_id,
            conflict_s_m,
            vehicle_lateral_m,
        )
        p0 = list(crossing.start_position_enu_m)
        retreat = list(safe_anchor.position_enu_m)
        ped_spec, ped_scene = add(
            specs,
            scenes,
            world_entity(
                ped,
                "pedestrian.cityops.basic.v1",
                "pedestrian",
                p0,
                crossing.sample.yaw_deg,
                "waiting",
            ),
        )
        ped_scene["placement_mode"] = "sidewalk_anchor"
        ped_scene["placement"] = {
            "lane_edge_id": conflict_edge_id,
            "longitudinal_s": round(conflict_s_m, 3),
            "lane_half_width_m": LANE_HALF_WIDTH_M,
            "offset_from_curb_m": round(float(crossing.offset_from_curb_m), 3),
            "resolved_lateral_from_center_m": round(
                float(crossing.resolved_lateral_from_center_m),
                3,
            ),
            "resolved_position_enu_m": p0,
            "roadway_conflict_position_enu_m": road,
            "safe_retreat_position_enu_m": retreat,
            "placement_semantics": "sidewalk_jaywalk_start",
            "rotation_deg": rot(crossing.sample.yaw_deg),
        }
        ped_spec["initial_pos_enu"] = p0
        _, car_scene = add(
            specs,
            scenes,
            lane_entity(
                car,
                "vehicle.ground.boxcar.v1",
                "vehicle",
                conflict_edge_id,
                car_start_s_m,
                vehicle_lateral_m,
                car_start,
                LANES.resolve_edge_s(conflict_edge_id, car_start_s_m).yaw_deg,
                "moving",
                prefer_edge_hint=True,
                route_authority=SUMO_DIRECTED_PHYSICAL_LANE_ROUTE_AUTHORITY,
            ),
        )
        if dist_xy(retreat, car_brake_route[-1]) < 8.0:
            raise RuntimeError(
                f"{scenario_id}: pedestrian retreat does not restore the required 8 m clearance"
            )
        car_resume_s_m = min(_edge_max_s_m, conflict_s_m + 24.0)
        if car_resume_s_m - car_stop_s_m < 12.0:
            raise RuntimeError(
                f"{scenario_id}: selected L4-6 edge lacks a physical post-conflict exit"
            )
        car_resume_route = l4_6_same_edge_route(
            conflict_edge_id,
            car_stop_s_m,
            car_resume_s_m,
            spacing_m=3.0,
            lateral_m=vehicle_lateral_m,
        )
        events = [
            event("ped_enters_roadway", tick(220), [move("move_ped_from_crosswalk_to_lane", ped, [p0, road], 1.6, activity_type="texting_walk"), move("move_car_toward_crosswalk", car, car_approach_route, 5.0, activity_type="driving", post_activity_type="approach_hold")], 1, scenario_id, "Texting pedestrian leaves a legal sidewalk and enters the vehicle conflict lane", "pedestrian", [ped, car], "warning", intent="jaywalk"),
            event("vehicle_ped_proximity", prox(ped, car, 4.1, 2, metric="3d"), [move("move_car_hard_brake_ped", car, car_brake_route, 0.6, activity_type="hard_braking", post_activity_type="stopped"), screenshot("capture_ped_vehicle_conflict")], 2, scenario_id, "Observed same-tick vehicle-pedestrian proximity triggers braking", "vehicle", [ped, car], "critical", intent="vehicle_brake", validation_event_type=PEDESTRIAN_VEHICLE_INTERACTION_EVENT_TYPE, validation_reason="Jaywalk scenario intentionally creates a close pedestrian-vehicle conflict before the vehicle stops short of the pedestrian.", validation_skip_checks=[PEDESTRIAN_VEHICLE_CLEARANCE_CHECK]),
            event("ped_retreats", fired("vehicle_ped_proximity"), [move("move_ped_retreat_sidewalk", ped, [road, p0, retreat], 1.8, activity_type="crossing", post_activity_type="waiting")], 3, scenario_id, "Pedestrian retreats from the conflict lane to a legal sidewalk safe point", "pedestrian", [ped], "info", intent="retreat"),
            event("conflict_cleared", fired("ped_retreats"), [move("move_car_resume_after_clearance", car, car_resume_route, 4.0, activity_type="driving", post_activity_type="stopped"), screenshot("capture_ped_vehicle_clearance")], 4, scenario_id, "Pedestrian safe clearance is restored and the vehicle resumes along its legal lane", "vehicle", [ped, car], "info"),
            context_event("jaywalk_vehicle_leaves_stage", fired("conflict_cleared", delay_ticks=20), [remove("remove_jaywalk_vehicle_after_clearance", car)], 0, scenario_id, "Recovered traffic leaves the event stage two seconds after the clearance evidence is complete", [car]),
        ]
        spawn_event_actor_at_dispatch(car_scene, events[0])
        params["l4_6_physical_chain"] = {
            "sumo_edge_id": conflict_edge_id,
            "sumo_lane_id": SUMO_GROUND_FLOW.edges[conflict_edge_id].lane_id,
            "vehicle_direction_policy": "strictly_increasing_edge_s",
            "vehicle_physical_lane_lateral_m": round(vehicle_lateral_m, 3),
            "vehicle_start_s_m": round(car_start_s_m, 3),
            "vehicle_brake_start_s_m": round(car_brake_start_s_m, 3),
            "vehicle_stop_s_m": round(car_stop_s_m, 3),
            "vehicle_approach_speed_mps": 5.0,
            "vehicle_braking_speed_mps": 0.6,
            "vehicle_stops_short_of_conflict_m": 2.5,
            "vehicle_max_sumo_projection_error_m": 3.0,
            "conflict_position_enu_m": road,
            "source_proximity_metric": "3d",
            "source_proximity_distance_m": 4.1,
            "source_proximity_required_ticks": 2,
            "right_of_way_holder": "vehicle",
            "pedestrian_start_semantics": "legal_sidewalk",
            "pedestrian_safe_retreat_enu_m": retreat,
            "pedestrian_safe_clearance_m": 8.0,
            "truth_authority": "same_tick_vehicle_and_pedestrian_pose_velocity_plus_sumo_lane_geometry",
            "authored_event_or_activity_labels_are_conflict_evidence": False,
        }
        desc = "Pedestrian jaywalk vehicle conflict"
        rules.append(
            {
                "rule": "vehicle_pedestrian_physical_conflict_chain",
                "vehicle_entity_id": car,
                "pedestrian_entity_id": ped,
                "sumo_edge_id": conflict_edge_id,
                "max_vehicle_projection_error_m": 3.0,
                "right_of_way_holder": "vehicle",
                "min_terminal_clearance_m": 8.0,
                "description": "Vehicle follows one legal directed SUMO edge, brakes short, and the pedestrian restores sidewalk clearance",
            }
        )
    elif scenario_id.startswith("L4-7_"):
        ped = f"ped_fall_{sid}"
        uav = f"uav_detect_{sid}"
        ambulance = f"ambulance_fall_{sid}"
        ped_spec, ped_scene = add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_23", 42, 1.2, p(ox + 3, oy + 2, 0), 0, "walking"))
        ped_start = scene_pos(ped_scene)
        ped_start, ped_context = pedestrian_preamble_walk_event(
            ped_spec,
            ped_scene,
            scenario_id,
            "fall_ped_approach",
            "move_fall_ped_approach",
            dx_m=-12.0,
            dy_m=8.0,
        )
        uav_start = q(ped_start[0] - 28.0, ped_start[1] - 20.0, 31)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 55, "patrol"))
        def l4_same_edge_vehicle_route(edge_id: str, start_s: float, arrival_s: float, lateral_m: float) -> list[list[float]]:
            route_s: list[float] = [float(start_s)]
            direction = 1.0 if float(arrival_s) >= float(start_s) else -1.0
            cursor = float(start_s)
            while abs(float(arrival_s) - cursor) > 6.0:
                cursor += direction * 6.0
                if (direction > 0.0 and cursor >= float(arrival_s)) or (direction < 0.0 and cursor <= float(arrival_s)):
                    break
                route_s.append(cursor)
            route_s.append(float(arrival_s))
            return _dedupe_route_points(
                [
                    _sumo_vehicle_point_at_s(
                        edge_id,
                        max(min(route_s), min(max(route_s), s_m)),
                        lateral_m,
                    )
                    for s_m in route_s
                ]
            )

        def l4_fall_ambulance_corridor() -> tuple[str, float, float, float, list[float], float]:
            ped_placement = dict(ped_scene.get("placement") or {})
            preferred_edge_id = str(ped_placement.get("lane_edge_id") or "")
            preferred_s = float(ped_placement.get("longitudinal_s") or 0.0)
            base_samples: list[LaneSample] = []
            if preferred_edge_id in LANES.by_edge:
                base_samples.append(LANES.resolve_edge_s(preferred_edge_id, preferred_s))
            nearby = sorted(
                LANES.samples,
                key=lambda sample: (sample.x_m - float(ped_start[0])) ** 2 + (sample.y_m - float(ped_start[1])) ** 2,
            )
            seen_edges = {sample.edge_id for sample in base_samples}
            for sample in nearby:
                if sample.edge_id in seen_edges:
                    continue
                seen_edges.add(sample.edge_id)
                base_samples.append(sample)
                if len(base_samples) >= 80:
                    break

            candidates: list[tuple[float, float, float, str, float, float, float, list[float], float]] = []
            for base_sample in base_samples:
                if str(base_sample.edge_id).startswith(":"):
                    continue
                min_s, max_s = LANES.edge_s_bounds(base_sample.edge_id)
                for arrival_delta_m in (0.0, -2.0, 2.0, -5.0, 5.0, -8.0, 8.0, -12.0, 12.0):
                    arrival_s = max(min_s, min(max_s, base_sample.s_m + arrival_delta_m))
                    arrival_sample = LANES.resolve_edge_s(base_sample.edge_id, arrival_s)
                    lateral_m = _select_physical_vehicle_lateral(arrival_sample, 0.0, ped_start)
                    arrival_point = _offset_from_lane(arrival_sample, lateral_m, GROUND_Z_M)
                    arrival_distance_m = dist_xy(arrival_point, ped_start)
                    if arrival_distance_m > 7.5:
                        continue
                    for start_delta_m in (-60.0, -42.0, -30.0):
                        start_s = max(min_s, min(max_s, arrival_s + start_delta_m))
                        if start_s + 0.05 >= arrival_s:
                            continue
                        route_distance_m = abs(start_s - arrival_s)
                        if route_distance_m < 30.0:
                            continue
                        route_points = l4_same_edge_vehicle_route(base_sample.edge_id, start_s, arrival_s, lateral_m)
                        if len(route_points) < 6:
                            continue
                        candidates.append(
                            (
                                arrival_distance_m,
                                abs(route_distance_m - 42.0),
                                0.0 if base_sample.edge_id == preferred_edge_id else 1.0,
                                base_sample.edge_id,
                                start_s,
                                arrival_s,
                                lateral_m,
                                route_points,
                                route_distance_m,
                            )
                        )
            if not candidates:
                raise RuntimeError(
                    f"{scenario_id}: unable to find a same-edge physical ambulance corridor within 7.5m of the fallen pedestrian"
                )
            (
                arrival_distance_m,
                _route_error_m,
                _preferred_penalty,
                edge_id,
                start_s,
                arrival_s,
                lateral_m,
                route_points,
                _route_distance_m,
            ) = min(candidates)
            return edge_id, start_s, arrival_s, lateral_m, route_points, arrival_distance_m

        (
            ambulance_edge_id,
            ambulance_s,
            ambulance_arrival_s,
            ambulance_lateral,
            ambulance_dispatch_route,
            ambulance_arrival_distance_m,
        ) = l4_fall_ambulance_corridor()
        ambulance_start_sample = LANES.resolve_edge_s(ambulance_edge_id, ambulance_s)
        ambulance_start = list(ambulance_dispatch_route[0])
        ambulance_arrival_threshold_m = round(max(8.0, min(14.0, ambulance_arrival_distance_m + 0.5)), 3)
        _, ambulance_scene = add(
            specs,
            scenes,
            lane_entity(
                ambulance,
                "vehicle.emergency.ambulance.v1",
                "vehicle",
                ambulance_edge_id,
                ambulance_s,
                ambulance_lateral,
                ambulance_start,
                ambulance_start_sample.yaw_deg,
                "response",
                visual_state={"mode": "response", "lights_on": True},
                prefer_edge_hint=True,
            ),
        )
        uav_observe = q(ped_start[0], ped_start[1], 16)
        events = [
            ped_context,
            event("pedestrian_fall", tick(230), [play("play_pedestrian_fall", ped)], 1, scenario_id, "Pedestrian fall animation starts", "pedestrian", [ped], "critical"),
            event("uav_observation_approach", fired("pedestrian_fall"), [move("move_uav_low_observation_fall", uav, [uav_start, uav_observe], 5.0)], 2, scenario_id, "UAV moves above the fallen pedestrian before detection is confirmed", "uav_mission", [uav, ped], "warning"),
            event("uav_detects_fall", prox(uav, ped, 35.0, 15, metric="3d"), [screenshot("capture_fall_detection")], 3, scenario_id, "UAV confirms the fallen pedestrian after physical 3-D overhead dwell", "uav_mission", [uav, ped], "warning", intent="uav_detect"),
            event("ambulance_response", fired("uav_detects_fall"), [move("move_ambulance_to_fall", ambulance, ambulance_dispatch_route, 12.0)], 4, scenario_id, "Ambulance responds to detected fall", "vehicle", [ambulance, ped], "warning", intent="ambulance", validation_event_type=PEDESTRIAN_VEHICLE_INTERACTION_EVENT_TYPE, validation_reason="Emergency response intentionally routes the ambulance close to the fallen pedestrian using a same-edge physical road-lane corridor.", validation_skip_checks=[PEDESTRIAN_VEHICLE_CLEARANCE_CHECK]),
            event("ambulance_arrived", prox(ambulance, ped, ambulance_arrival_threshold_m, 2), [screenshot("capture_fall_arrival")], 5, scenario_id, "Ambulance physically reaches curbside standoff distance from the fallen pedestrian", "vehicle", [ambulance, ped], "warning"),
            event("responder_handoff", fired("ambulance_arrived", delay_ticks=15), [screenshot("capture_fall_handoff")], 6, scenario_id, "Responder remains at the patient through the handoff dwell", "vehicle", [ambulance, ped], "info"),
            event("incident_documented", fired("responder_handoff"), [screenshot("capture_fall_response")], 7, scenario_id, "UAV documents the physically completed emergency handoff", "uav_mission", [uav, ambulance, ped], "info"),
        ]
        spawn_event_actor_at_dispatch(ambulance_scene, events[4])
        for scenario_event in events:
            if scenario_event.get("event_id") == "uav_detects_fall":
                scenario_event["require_conditions"] = ["uav_observation_approach"]
        desc = "Pedestrian fall detected by UAV and response chain"
        rules.append(
            {
                "rule": "ambulance_same_edge_physical_response",
                "entity_id": ambulance,
                "edge_id": ambulance_edge_id,
                "start_s_m": round(float(ambulance_s), 3),
                "arrival_s_m": round(float(ambulance_arrival_s), 3),
                "arrival_distance_m": round(float(ambulance_arrival_distance_m), 3),
                "arrival_threshold_m": ambulance_arrival_threshold_m,
                "max_authoritative_lane_projection_error_m": 3.0,
                "description": "Ambulance source waypoints stay on one physical lane and reach a curbside standoff distance from the fallen pedestrian",
            }
        )
    elif scenario_id.startswith("L4-8_"):
        uav = f"uav_crowd_monitor_{sid}"
        uav_start = p(ox + 6, oy + 4, 34)
        crowd_extent = [900, 600, 0]
        safe_extent = [1000, 500, 0]
        crowd_origin = gathering_point(p(ox + 3, oy + 3, 0), crowd_extent)
        safe_origin = gathering_point(p(ox + 24, oy + 19, 0), safe_extent)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 50, "monitor"))
        add(specs, scenes, world_entity(f"crowd_anchor_{sid}", "semantic.spawn_zone", "crowd_anchor", crowd_origin, 0))
        cohort = add_evacuation_ped_cohort(
            specs,
            scenes,
            sid,
            "ped_evac",
            crowd_origin,
            safe_origin,
            8,
            min_safe_distance_from_origin_m=30.0,
        )
        events = [
            context_event("crowd_generated", tick(120), [], 0, scenario_id, "Explicit pedestrian cohort is present before evacuation begins", [f"crowd_anchor_{sid}"]),
            event("evacuation_triggered", tick(220), evacuation_move_actions(cohort, "move_crowd_evacuation"), 2, scenario_id, "Visible pedestrian cohort walks to safe evacuation zone", "pedestrian", [f"crowd_anchor_{sid}"], "warning"),
            event("crowd_safe_zone_reached", fired("evacuation_triggered"), [screenshot("capture_crowd_safe_zone")], 3, scenario_id, "Pedestrian trajectories physically reach the safe evacuation zone", "pedestrian", [f"crowd_anchor_{sid}"], "warning"),
            event("uav_evacuation_monitor", fired("crowd_safe_zone_reached"), [move("move_uav_monitor_evacuation", uav, [uav_start, q(crowd_origin[0], crowd_origin[1], 30), q(safe_origin[0], safe_origin[1], 30)], 5.0), screenshot("capture_crowd_evacuation")], 4, scenario_id, "UAV monitors cleared evacuation geometry", "uav_mission", [uav], "info"),
            event("crowd_safe_hold", fired("uav_evacuation_monitor"), [screenshot("capture_crowd_safe_hold")], 5, scenario_id, "Crowd evacuation remains in safe hold without removing visible agents", "pedestrian", [uav], "info"),
        ]
        desc = "Crowd evacuation with UAV monitoring"
        rules.append({"rule": "explicit_pedestrian_count_min", "min_count": 8, "description": "Crowd evacuation scenario uses at least 8 individually controlled pedestrians"})
    elif scenario_id.startswith("L4-9_"):
        car_a = f"car_intersection_{sid}_a"
        car_b = f"car_intersection_{sid}_b"
        focus = p(ox, oy + 2, 0)

        def intersection_corridors() -> tuple[str, float, str, float]:
            focus_point = Point(float(focus[0]), float(focus[1]))
            nearby: list[tuple[float, str, LineString]] = []
            for edge in SUMO_GROUND_FLOW.edges.values():
                if not SUMO_GROUND_FLOW._edge_allows(edge, "vehicle"):
                    continue
                line = LineString(edge.shape_xy)
                distance_to_focus_m = float(line.distance(focus_point))
                if distance_to_focus_m <= 600.0:
                    nearby.append((distance_to_focus_m, edge.edge_id, line))
            nearby.sort(key=lambda item: (item[0], item[1]))
            nearby_tree = STRtree([item[2] for item in nearby])
            candidates: list[tuple[float, str, float, str, float]] = []
            for first_index, (_focus_distance_a, edge_a_id, line_a) in enumerate(nearby):
                road_a = edge_a_id.rsplit("_pl", 1)[0]
                adjacent_indices = sorted(
                    int(index)
                    for index in nearby_tree.query(line_a, predicate="dwithin", distance=5.8)
                    if int(index) > first_index
                )
                for second_index in adjacent_indices:
                    _focus_distance_b, edge_b_id, line_b = nearby[second_index]
                    if edge_b_id.rsplit("_pl", 1)[0] == road_a:
                        continue
                    distance_m = float(line_a.distance(line_b))
                    if distance_m > 5.8:
                        continue
                    point_a, point_b = nearest_points(line_a, line_b)
                    _da, s_a, _xa, _ya, yaw_a = SUMO_GROUND_FLOW.project_point_to_edge(
                        edge_a_id,
                        [point_a.x, point_a.y, GROUND_Z_M],
                    )
                    _db, s_b, _xb, _yb, yaw_b = SUMO_GROUND_FLOW.project_point_to_edge(
                        edge_b_id,
                        [point_b.x, point_b.y, GROUND_Z_M],
                    )
                    if min(s_a, s_b) < 28.0:
                        continue
                    angle_delta = abs((yaw_a - yaw_b + 180.0) % 360.0 - 180.0)
                    if angle_delta < 30.0 or angle_delta > 150.0:
                        continue
                    center_distance = dist_xy(
                        [0.5 * (point_a.x + point_b.x), 0.5 * (point_a.y + point_b.y), 0.0],
                        focus,
                    )
                    candidates.append((center_distance + distance_m * 4.0, edge_a_id, s_a, edge_b_id, s_b))
            if not candidates:
                raise RuntimeError(f"{scenario_id}: no physical crossing SUMO edge pair near assigned focus")
            _score, edge_a_id, s_a, edge_b_id, s_b = min(candidates)
            return edge_a_id, s_a, edge_b_id, s_b

        def directed_route(edge_id: str, start_s: float, end_s: float) -> list[list[float]]:
            segment_count = max(1, int(math.ceil((end_s - start_s) / 4.0)))
            return [
                _sumo_vehicle_point_at_s(
                    edge_id,
                    start_s + (end_s - start_s) * index / segment_count,
                )
                for index in range(segment_count + 1)
            ]

        edge_a, conflict_s_a, edge_b, conflict_s_b = intersection_corridors()
        start_s_a = conflict_s_a - 28.0
        start_s_b = conflict_s_b - 28.0
        approach_a = directed_route(edge_a, start_s_a, conflict_s_a)
        approach_b = directed_route(edge_b, start_s_b, conflict_s_b)
        a0 = approach_a[0]
        b0 = approach_b[0]
        _, car_a_scene = add(
            specs,
            scenes,
            lane_entity(
                car_a,
                "vehicle.ground.boxcar.v1",
                "vehicle",
                edge_a,
                start_s_a,
                0.0,
                a0,
                0.0,
                "moving",
                prefer_edge_hint=True,
            ),
        )
        _, car_b_scene = add(
            specs,
            scenes,
            lane_entity(
                car_b,
                "vehicle.emergency.suv.v1",
                "vehicle",
                edge_b,
                start_s_b,
                0.0,
                b0,
                0.0,
                "moving",
                prefer_edge_hint=True,
            ),
        )
        events = [
            event("vehicles_enter_intersection", tick(230), [move("move_car_a_intersection", car_a, approach_a, 8.0), move("move_car_b_intersection", car_b, approach_b, 7.5)], 1, scenario_id, "Two vehicles enter the same intersection from different legal SUMO edges", "vehicle", [car_a, car_b], "warning"),
            event("vehicle_collision_warning", prox(car_a, car_b, 6.0, 2), [visual("set_car_a_braking", car_a, "braking"), visual("set_car_b_braking", car_b, "braking"), screenshot("capture_vehicle_conflict")], 2, scenario_id, "Vehicle-vehicle proximity triggers simultaneous braking", "vehicle", [car_a, car_b], "critical"),
            context_event("intersection_conflict_stage_closes", fired("vehicle_collision_warning", delay_ticks=50), [remove("remove_car_a_after_conflict_evidence", car_a), remove("remove_car_b_after_conflict_evidence", car_b)], 0, scenario_id, "Conflict vehicles leave the event stage after the braking evidence dwell", [car_a, car_b]),
        ]
        spawn_event_actor_at_dispatch(car_a_scene, events[0])
        spawn_event_actor_at_dispatch(car_b_scene, events[0])
        desc = "Vehicle-vehicle intersection conflict"
    elif scenario_id.startswith("L4-10_"):
        ambulance = f"ambulance_priority_{sid}"
        car = f"civilian_yield_{sid}"
        uav = f"uav_priority_monitor_{sid}"
        contract_roi = _active_roi_polygon()
        if ACTIVE_ROI_CONTRACT is not None:
            focus = list(ACTIVE_ROI_CONTRACT.center_enu_m)
        else:
            focus = p(ox, oy, GROUND_Z_M)
        ambulance_edge, focus_s_m, priority_lane_offsets = _priority_multi_lane_corridor(
            scenario_id,
            focus,
            contract_roi,
        )
        car_edge = ambulance_edge
        through_lateral_m = priority_lane_offsets[0]
        yield_lateral_m = priority_lane_offsets[-1]
        lateral_separation_m = abs(yield_lateral_m - through_lateral_m)
        if lateral_separation_m < PRIORITY_VEHICLE_MIN_CENTER_SEPARATION_M:
            raise RuntimeError(
                f"{scenario_id}: priority corridor lane centers are only "
                f"{lateral_separation_m:.3f}m apart"
            )

        def l4_10_lane_route(
            edge_id: str,
            start_s: float,
            end_s: float,
            lateral_m: float,
            step_m: float = 6.0,
        ) -> list[list[float]]:
            if start_s >= end_s:
                raise RuntimeError(f"{scenario_id}: L4-10 route must move forward on {edge_id}: {start_s} -> {end_s}")
            route_s = [float(start_s)]
            cursor = float(start_s)
            while cursor + step_m < float(end_s):
                cursor += step_m
                route_s.append(cursor)
            route_s.append(float(end_s))
            return _dedupe_route_points(
                [
                    _sumo_vehicle_point_at_s(edge_id, s_m, lateral_m)
                    for s_m in route_s
                ]
            )

        def civilian_pull_over_route(
            edge_id: str,
            start_s: float,
            end_s: float,
        ) -> list[list[float]]:
            lane_change_end_s = start_s + 8.0
            route_s = [float(start_s)]
            cursor = float(start_s)
            while cursor + 2.0 < float(end_s):
                cursor += 2.0
                route_s.append(cursor)
            route_s.append(float(end_s))
            route: list[list[float]] = []
            for s_m in route_s:
                fraction = min(
                    1.0,
                    max(0.0, (s_m - start_s) / (lane_change_end_s - start_s)),
                )
                lateral_m = (
                    through_lateral_m
                    + (yield_lateral_m - through_lateral_m) * fraction
                )
                route.append(
                    _sumo_vehicle_point_at_s(edge_id, s_m, lateral_m)
                )
            return _dedupe_route_points(route)

        shared_length_m = SUMO_GROUND_FLOW.edge_shape_length_m(ambulance_edge)
        ambulance_span_m = min(65.0, max(32.0, shared_length_m - 20.0))
        ambulance_start_s = max(
            8.0,
            min(focus_s_m - 20.0, shared_length_m - ambulance_span_m - 8.0),
        )
        ambulance_clear_s = min(shared_length_m - 2.0, ambulance_start_s + ambulance_span_m)
        if ambulance_clear_s - ambulance_start_s < 30.0:
            raise RuntimeError(
                f"{scenario_id}: priority corridor is too short for ambulance passage "
                f"on {ambulance_edge}"
            )
        car_span_m = min(30.0, max(12.0, ambulance_clear_s - ambulance_start_s - 8.0))
        car_start_s = min(
            ambulance_start_s + min(32.0, (ambulance_clear_s - ambulance_start_s) * 0.45),
            ambulance_clear_s - car_span_m,
        )
        car_yield_s = min(shared_length_m - 2.0, car_start_s + car_span_m)
        ambulance_entry_s = max(0.0, ambulance_start_s - 12.0)
        car_entry_s = max(0.0, car_start_s - 12.0)
        if car_yield_s - car_start_s < 12.0:
            raise RuntimeError(
                f"{scenario_id}: priority corridor is too short for civilian yield "
                f"on {car_edge}"
            )
        ambulance_route = l4_10_lane_route(
            ambulance_edge,
            ambulance_start_s,
            ambulance_clear_s,
            through_lateral_m,
        )
        car_yield_route = civilian_pull_over_route(
            car_edge,
            car_start_s,
            car_yield_s,
        )
        ambulance_priority_motion_ticks = quantized_path_duration_ticks(
            ambulance_route,
            PRIORITY_AMBULANCE_SPEED_MPS,
        )
        civilian_yield_motion_ticks = quantized_path_duration_ticks(
            car_yield_route,
            PRIORITY_YIELD_SPEED_MPS,
        )
        ambulance_entry_sample = _sumo_lane_sample_at_s(ambulance_edge, ambulance_entry_s)
        car_entry_sample = _sumo_lane_sample_at_s(car_edge, car_entry_s)
        ambulance_entry = _sumo_vehicle_point_at_s(
            ambulance_edge,
            ambulance_entry_s,
            through_lateral_m,
        )
        car_entry = _sumo_vehicle_point_at_s(
            car_edge,
            car_entry_s,
            through_lateral_m,
        )
        ambulance_start = list(ambulance_route[0])
        car_start = list(car_yield_route[0])
        _, ambulance_scene = add(
            specs,
            scenes,
            lane_entity(
                ambulance,
                "vehicle.emergency.ambulance.v1",
                "vehicle",
                ambulance_edge,
                ambulance_entry_s,
                through_lateral_m,
                ambulance_entry,
                ambulance_entry_sample.yaw_deg,
                "response",
                visual_state={"mode": "response", "lights_on": True},
                prefer_edge_hint=True,
                route_authority=SUMO_DIRECTED_PHYSICAL_LANE_ROUTE_AUTHORITY,
            ),
        )
        _, car_scene = add(
            specs,
            scenes,
            lane_entity(
                car,
                "vehicle.ground.boxcar.v1",
                "vehicle",
                car_edge,
                car_entry_s,
                through_lateral_m,
                car_entry,
                car_entry_sample.yaw_deg,
                "moving",
                prefer_edge_hint=True,
                route_authority=SUMO_DIRECTED_PHYSICAL_LANE_ROUTE_AUTHORITY,
            ),
        )
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", q(ambulance_start[0] - 12.0, ambulance_start[1] + 12.0, 32), 60, "monitor"))
        ambulance_entry_event = event_actor_entry(
            scenario_id=scenario_id,
            scene=ambulance_scene,
            event_id="ambulance_priority_lane_entry",
            semantic_event_id="ambulance_priority_approach",
            entry_tick=160,
            arrival_tick=220,
            route=[ambulance_entry, ambulance_start],
            activity_type="response",
            requested_speed_mps=PRIORITY_AMBULANCE_SPEED_MPS,
        )
        civilian_entry_event = event_actor_entry(
            scenario_id=scenario_id,
            scene=car_scene,
            event_id="civilian_priority_lane_entry",
            semantic_event_id="civilian_vehicle_yields",
            entry_tick=160,
            arrival_tick=220,
            route=[car_entry, car_start],
            activity_type="moving",
        )
        events = [
            ambulance_entry_event,
            civilian_entry_event,
            event("ambulance_priority_approach", tick(220), [visual("set_ambulance_lights_on", ambulance, "response", lights_on=True), move("move_ambulance_priority", ambulance, ambulance_route, PRIORITY_AMBULANCE_SPEED_MPS, activity_type="response", post_activity_type="response"), move("move_civilian_priority_pull_over", car, car_yield_route, PRIORITY_YIELD_SPEED_MPS, activity_type="yielding", post_activity_type="yielding")], 1, scenario_id, "Ambulance advances in the through lane while the civilian continues at normal urban speed and pulls into the adjacent yield lane", "vehicle", [ambulance, car], "warning"),
            event("civilian_vehicle_yields", prox(ambulance, car, 35.0, 3, metric="xy"), [visual("set_civilian_yielding", car, "yielding")], 2, scenario_id, "Civilian vehicle establishes sustained reduced-speed motion in the adjacent yield lane before the ambulance passes", "vehicle", [ambulance, car], "info", intent="yield"),
            event("uav_priority_capture", fired("civilian_vehicle_yields", delay_ticks=25), [move("move_uav_priority_monitor", uav, [q(ambulance_start[0] - 12.0, ambulance_start[1] + 12.0, 32), q(ambulance_route[-1][0], ambulance_route[-1][1], 30)], 5.0), screenshot("capture_ambulance_priority")], 3, scenario_id, "UAV monitors ambulance priority passage after yielding is physically visible", "uav_mission", [uav, ambulance], "info"),
            context_event("ambulance_priority_leaves_stage", fired("ambulance_priority_approach", delay_ticks=ambulance_priority_motion_ticks + PRIORITY_VEHICLE_EVIDENCE_DWELL_TICKS), [remove("remove_ambulance_after_priority_passage", ambulance)], 0, scenario_id, "Ambulance leaves the event stage two seconds after completing its priority passage", [ambulance]),
            context_event("civilian_yield_leaves_stage", fired("ambulance_priority_approach", delay_ticks=civilian_yield_motion_ticks + PRIORITY_VEHICLE_EVIDENCE_DWELL_TICKS), [remove("remove_civilian_after_priority_passage", car)], 0, scenario_id, "Yielding civilian leaves the event stage two seconds after completing its lane change and forward passage", [car]),
        ]
        for scenario_event in events:
            if scenario_event.get("event_id") == "civilian_vehicle_yields":
                scenario_event["require_conditions"] = ["ambulance_priority_approach"]
        desc = "Ambulance priority passage with civilian yield"
        rules.append(
            {
                "rule": "ambulance_priority_physical_lane_chain",
                "ambulance_id": ambulance,
                "civilian_id": car,
                "ambulance_edge_id": ambulance_edge,
                "civilian_edge_id": car_edge,
                "ambulance_start_s_m": ambulance_start_s,
                "ambulance_clear_s_m": ambulance_clear_s,
                "civilian_start_s_m": car_start_s,
                "civilian_yield_s_m": car_yield_s,
                "civilian_yield_speed_mps": PRIORITY_YIELD_SPEED_MPS,
                "ambulance_motion_duration_ticks": ambulance_priority_motion_ticks,
                "civilian_motion_duration_ticks": civilian_yield_motion_ticks,
                "post_motion_evidence_dwell_ticks": PRIORITY_VEHICLE_EVIDENCE_DWELL_TICKS,
                "through_lane_lateral_m": through_lateral_m,
                "yield_lane_lateral_m": yield_lateral_m,
                "minimum_lane_center_separation_m": PRIORITY_VEHICLE_MIN_CENTER_SEPARATION_M,
                "min_initial_longitudinal_gap_m": 30.0,
                "max_authoritative_lane_projection_error_m": 3.0,
                "description": "Civilian pulls into an adjacent physical lane before the ambulance passes; priority/yield/clearance are derived from kinematics",
            }
        )
    else:
        av = f"av_fault_{sid}"
        uav = f"uav_av_report_{sid}"
        focus = p(ox, oy, GROUND_Z_M)
        lane_candidates: list[tuple[float, str, float, float]] = []
        for edge in SUMO_GROUND_FLOW.edges.values():
            if not SUMO_GROUND_FLOW._edge_allows(edge, "vehicle"):
                continue
            edge_length_m = SUMO_GROUND_FLOW.edge_shape_length_m(edge.edge_id)
            if edge_length_m < 40.0:
                continue
            distance_m, focus_s_m, _x_m, _y_m, _yaw_deg = (
                SUMO_GROUND_FLOW.project_point_to_edge(edge.edge_id, focus)
            )
            lane_candidates.append((float(distance_m), edge.edge_id, float(focus_s_m), edge_length_m))
        if not lane_candidates:
            raise RuntimeError(f"{scenario_id}: no legal directed SUMO lane for autonomous safe stop")
        _distance_m, av_edge_id, focus_s_m, av_edge_length_m = min(lane_candidates)
        av_s = max(20.0, min(focus_s_m - 7.5, av_edge_length_m - 18.0))
        nominal_start_s = av_s - 18.0
        safe_stop_mid_s = av_s + 7.5
        stop_s = av_s + 15.0
        av_sample = _sumo_lane_sample_at_s(av_edge_id, av_s)
        nominal_start = _sumo_vehicle_point_at_s(av_edge_id, nominal_start_s)
        start = _sumo_vehicle_point_at_s(av_edge_id, av_s)
        safe_stop_mid = _sumo_vehicle_point_at_s(av_edge_id, safe_stop_mid_s)
        stop = _sumo_vehicle_point_at_s(av_edge_id, stop_s)
        _, av_scene = add(
            specs,
            scenes,
            lane_entity(
                av,
                "vehicle.ground.boxcar.v1",
                "vehicle",
                av_edge_id,
                nominal_start_s,
                0.0,
                nominal_start,
                av_sample.yaw_deg,
                "autonomous",
                prefer_edge_hint=True,
            ),
        )
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", p(ox - 20, oy + 11, 33), 70, "patrol"))
        av_entry = event_actor_entry(
            scenario_id=scenario_id,
            scene=av_scene,
            event_id="av_nominal_lane_entry",
            semantic_event_id="av_sensor_fault",
            entry_tick=190,
            arrival_tick=230,
            route=[nominal_start, start],
            activity_type="autonomous",
        )
        events = [
            av_entry,
            event("av_sensor_fault", tick(230), [visual("set_av_sensor_fault", av, "sensor_fault")], 1, scenario_id, "Autonomous vehicle sensor fault detected", "vehicle", [av], "warning"),
            event("av_safe_stop", fired("av_sensor_fault"), [move("move_av_safe_stop", av, [start, safe_stop_mid, stop], 2.0), visual("set_av_safe_stop", av, "safe_stop")], 2, scenario_id, "AV performs safe stop in lane", "vehicle", [av], "critical"),
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
    uav_spec, uav_scene = add(
        specs,
        scenes,
        world_entity(uav, "uav.inspect.quad.v1", "uav", start, 35, "patrol"),
    )
    car = f"weather_car_{sid}"
    ped = f"weather_ped_{sid}"
    def add_default_weather_context() -> None:
        add(specs, scenes, lane_entity(car, "vehicle.ground.boxcar.v1", "vehicle", "cg_edge_29", 36, 0.0, p(ox + 8, oy - 8, 0), 90, "moving"))
        add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_29", 40, 8.0, p(ox + 2, oy + 5, 0), 0, "walking"))

    if scenario_id.startswith("L5-1"):
        def rain_vehicle_corridor() -> tuple[str, dict[str, float], float]:
            offsets_m = {
                "start": 0.0,
                "clear": 28.0,
                "wet": 52.0,
                "heavy": 72.0,
                "recovery": 96.0,
            }
            roi_polygon = _active_roi_polygon()
            focus = (
                list(ACTIVE_ROI_CONTRACT.center_enu_m)
                if ACTIVE_ROI_CONTRACT is not None
                else p(ox + 8, oy - 8, 0)
            )
            candidates: list[tuple[float, float, str, dict[str, float], float]] = []
            seen: set[tuple[str, int]] = set()
            for base_sample in LANES.samples:
                edge_id = str(base_sample.edge_id)
                if edge_id.startswith(":"):
                    continue
                sumo_edge = SUMO_GROUND_FLOW.edges.get(edge_id)
                if sumo_edge is None or not SUMO_GROUND_FLOW._edge_allows(sumo_edge, "vehicle"):
                    continue
                key = (edge_id, int(round(float(base_sample.s_m))))
                if key in seen:
                    continue
                seen.add(key)
                min_s, max_s = LANES.edge_s_bounds(edge_id)
                route_s = {name: float(base_sample.s_m) + offset for name, offset in offsets_m.items()}
                if min(route_s.values()) < min_s - 1e-6 or max(route_s.values()) > max_s + 1e-6:
                    continue
                physical_laterals = [
                    value
                    for value in _physical_vehicle_lane_offsets(base_sample)
                    if abs(value) <= 3.0
                ]
                if not physical_laterals:
                    continue
                lateral_m = min(
                    physical_laterals,
                    key=lambda value: (
                        dist_xy(
                            _sumo_vehicle_point_at_s(
                                edge_id,
                                float(base_sample.s_m) + offsets_m["clear"],
                                value,
                            ),
                            focus,
                        ),
                        value,
                    ),
                )
                sumo_centerline = LineString(sumo_edge.shape_xy)
                planned_points = [
                    _offset_from_lane(LANES.resolve_edge_s(edge_id, s_m), lateral_m, GROUND_Z_M)
                    for s_m in route_s.values()
                ]
                if roi_polygon is not None:
                    semantic_points = planned_points[1:]
                    pedestrian_point = _sumo_vehicle_point_at_s(
                        edge_id,
                        route_s["clear"],
                        8.0,
                    )
                    if not all(
                        roi_polygon.covers(Point(point[0], point[1]))
                        for point in [*semantic_points, pedestrian_point]
                    ):
                        continue
                projection_error_m = max(
                    float(sumo_centerline.distance(Point(point[0], point[1])))
                    for point in planned_points
                )
                if projection_error_m > 3.0:
                    continue
                focus_points = planned_points[1:] if roi_polygon is not None else planned_points
                route_center = [
                    sum(point[0] for point in focus_points) / float(len(focus_points)),
                    sum(point[1] for point in focus_points) / float(len(focus_points)),
                    GROUND_Z_M,
                ]
                candidates.append((dist_xy(route_center, focus), projection_error_m, edge_id, route_s, lateral_m))
            if not candidates:
                raise RuntimeError(f"{scenario_id}: no legal rain-impact SUMO vehicle corridor near assigned focus")
            candidates.sort()
            if roi_polygon is None:
                _center_error, _projection_error, edge_id, route_s, lateral_m = candidates[0]
                return edge_id, route_s, lateral_m
            same_edge_external_starts: dict[tuple[str, float], list[float]] = {}
            for _center_error, _projection_error, edge_id, _route_s, lateral_m in candidates:
                key = (edge_id, lateral_m)
                if key in same_edge_external_starts:
                    continue
                edge_length_m = SUMO_GROUND_FLOW.edge_shape_length_m(edge_id)
                sample_count = max(
                    1,
                    int(math.ceil(edge_length_m / EVENT_ACTOR_ENTRY_ROUTE_STEP_M)),
                )
                starts: list[float] = []
                for sample_index in range(sample_count + 1):
                    s_m = min(
                        edge_length_m,
                        sample_index * EVENT_ACTOR_ENTRY_ROUTE_STEP_M,
                    )
                    position = _sumo_vehicle_point_at_s(edge_id, s_m, lateral_m)
                    point = Point(position[0], position[1])
                    if (
                        not roi_polygon.covers(point)
                        and roi_polygon.distance(point)
                        >= EVENT_ACTOR_ENTRY_BOUNDARY_CLEARANCE_M
                    ):
                        starts.append(s_m)
                same_edge_external_starts[key] = starts
            entry_failures: list[dict[str, Any]] = []
            ranked_candidates: list[
                tuple[
                    int,
                    tuple[float, float, str, dict[str, float], float],
                ]
            ] = []
            for candidate in candidates:
                _center_error, _projection_error, edge_id, route_s, lateral_m = candidate
                target_s_m = route_s["clear"]
                starts = same_edge_external_starts[(edge_id, lateral_m)]
                start_index = bisect_right(starts, target_s_m) - 1
                if start_index >= 0:
                    directed_distance_lower_bound_m = target_s_m - starts[start_index]
                    if (
                        directed_distance_lower_bound_m
                        > EVENT_ACTOR_ENTRY_SPEED_MAX_MPS * 180.0 / SCRIPT_TICK_HZ
                    ):
                        continue
                    ranked_candidates.append((0, candidate))
                    continue
                ranked_candidates.append((1, candidate))
            ranked_candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1][0],
                    item[1][1],
                    item[1][2],
                    item[1][3]["clear"],
                    item[1][4],
                )
            )
            for _topology_priority, candidate in ranked_candidates:
                _center_error, _projection_error, edge_id, route_s, lateral_m = candidate
                target_s_m = route_s["clear"]
                tactical_position = _sumo_vehicle_point_at_s(
                    edge_id,
                    target_s_m,
                    lateral_m,
                )
                entry_scene = {
                    "entity_id": car,
                    "placement": {
                        "edge_id": edge_id,
                        "resolved_lateral_from_center_m": lateral_m,
                    },
                }
                try:
                    _start_edge, _start_s, _start_lateral, entry_route = (
                        _vehicle_entry_route_outside_capture_boundary(
                            entry_scene,
                            tactical_position,
                            roi_polygon,
                        )
                    )
                except RuntimeError:
                    continue
                travel_ticks = quantized_path_duration_ticks(
                    entry_route,
                    EVENT_ACTOR_ENTRY_SPEED_MAX_MPS,
                )
                if travel_ticks <= 180:
                    return edge_id, route_s, lateral_m
                entry_failures.append(
                    {
                        "edge_id": edge_id,
                        "clear_s_m": round(route_s["clear"], 3),
                        "route_length_m": round(path_length_m(entry_route), 3),
                        "ticks_at_speed_ceiling": travel_ticks,
                        "semantic_center_error_m": round(_center_error, 3),
                    }
                )
            entry_failures.sort(
                key=lambda item: (
                    item["ticks_at_speed_ceiling"],
                    item["semantic_center_error_m"],
                )
            )
            raise RuntimeError(
                f"{scenario_id}: ROI has no rain-impact corridor reachable before tick 180 "
                f"at {EVENT_ACTOR_ENTRY_SPEED_MAX_MPS:.1f}m/s; "
                f"best_entries={entry_failures[:5]}"
            )

        def rain_lane_point(edge_id: str, s_m: float, lateral_m: float) -> list[float]:
            return _sumo_vehicle_point_at_s(edge_id, s_m, lateral_m)

        def rain_lane_route(edge_id: str, start_s: float, end_s: float, lateral_m: float) -> list[list[float]]:
            route_s: list[float] = [float(start_s)]
            cursor = float(start_s)
            while float(end_s) - cursor > 6.0:
                cursor += 6.0
                if cursor >= float(end_s):
                    break
                route_s.append(cursor)
            route_s.append(float(end_s))
            return _dedupe_route_points(
                [rain_lane_point(edge_id, s_m, lateral_m) for s_m in route_s]
            )

        rain_edge, rain_s, car_lateral = rain_vehicle_corridor()
        car_start = rain_lane_point(rain_edge, rain_s["start"], car_lateral)
        car_clear = rain_lane_point(rain_edge, rain_s["clear"], car_lateral)
        car_wet = rain_lane_point(rain_edge, rain_s["wet"], car_lateral)
        car_heavy = rain_lane_point(rain_edge, rain_s["heavy"], car_lateral)
        car_recovery = rain_lane_point(rain_edge, rain_s["recovery"], car_lateral)
        _, car_scene = add(
            specs,
            scenes,
            lane_entity(
                car,
                "vehicle.ground.boxcar.v1",
                "vehicle",
                rain_edge,
                rain_s["start"],
                car_lateral,
                car_start,
                90,
                "moving",
                prefer_edge_hint=True,
            ),
        )
        car_start = scene_pos(car_scene)
        add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", rain_edge, rain_s["clear"], 8.0, car_clear, 0, "walking"))

        threshold = 0.5
        trigger = weather("rain", "gte", threshold, 5)
        profile = {
            "initial": "clear",
            "transitions": [
                {
                    "tick": 180,
                    "profile": "rain",
                    "overrides": {"rain": 0.55, "wetness": 0.65, "visibility_m": 2200.0},
                }
            ],
        }
        uav_clear = p(ox + 18, oy + 10, 33)
        uav_wet = p(ox + 26, oy + 18, 32)
        uav_heavy = p(ox + 18, oy + 28, 34)
        uav_recovery = p(ox - 4, oy + 20, 34)
        clear_motion_entry = event_actor_entry(
            scenario_id=scenario_id,
            scene=car_scene,
            event_id="clear_motion_baseline",
            semantic_event_id="weather_onset",
            entry_tick=120,
            arrival_tick=180,
            route=rain_lane_route(
                rain_edge,
                rain_s["start"],
                rain_s["clear"],
                car_lateral,
            ),
            activity_type="moving",
        )
        clear_motion_entry["actions"].append(
            move(
                "move_uav_clear_weather_baseline",
                uav,
                [start, uav_clear],
                6.0,
                "clear_weather_patrol",
                "clear_weather_patrol",
            )
        )
        clear_motion_entry["log_target_ids"].append(uav)
        clear_motion_entry["target_roles"] = _target_roles(
            clear_motion_entry["log_target_ids"]
        )
        clear_motion_entry["log_title"] = (
            "Vehicle enters on its legal lane while the UAV establishes the clear-weather motion baseline"
        )
        events = [
            clear_motion_entry,
            event("weather_onset", tick(180), [set_weather("set_rain_onset", "rain", rain=0.55, wetness=0.65, visibility_m=2200.0)], 1, scenario_id, "Rain onset is applied before threshold confirmation", "weather", [uav, car], "info"),
            event("rain_condition_met", trigger, [set_weather("set_rain_moderate", "rain", rain=0.55, wetness=0.72, visibility_m=2200.0)], 1, scenario_id, "Rain threshold reached", "weather", [uav, car], "warning"),
        ]
        if scenario_id.endswith("_v1"):
            events.extend(
                [
                    event("rain_speed_reduction", fired("rain_condition_met"), [move("move_uav_rain_slowdown", uav, [uav_clear, uav_wet], 2.8, "rain_caution_route", "rain_caution_route"), move("move_car_wet_slowdown", car, rain_lane_route(rain_edge, rain_s["clear"], rain_s["wet"], car_lateral), 3.4)], 2, scenario_id, "UAV follows the rain speed policy; vehicle follows its wet-road speed policy", "uav_mission", [uav, car], "warning"),
                    event("rain_intensifies", fired("rain_speed_reduction"), [set_weather("set_rain_heavy_visibility", "rain", rain=0.72, wetness=0.82, visibility_m=1100.0), move("move_car_heavy_rain_caution", car, rain_lane_route(rain_edge, rain_s["wet"], rain_s["heavy"], car_lateral), 3.0)], 3, scenario_id, "Rain intensifies and vehicle remains cautious", "weather", [uav, car], "critical"),
                    event("rain_recovery", fired("rain_intensifies"), [move("move_uav_rain_recover", uav, [uav_wet, uav_recovery], 5.5, "rain_recovery_route", "clear_weather_patrol"), move("move_car_rain_recover", car, rain_lane_route(rain_edge, rain_s["heavy"], rain_s["recovery"], car_lateral), 5.8), set_weather("set_rain_recovering", "clear", rain=0.02, wetness=0.2, visibility_m=7000.0)], 4, scenario_id, "Rain recovers and both actors resume higher-speed motion", "weather", [uav, car], "info"),
                ]
            )
            desc = "Rain slowdown with continuous recovery"
        elif scenario_id.endswith("_v2"):
            events.extend(
                [
                    event("rain_speed_reduction", fired("rain_condition_met"), [move("move_uav_rain_slowdown", uav, [uav_clear, uav_wet], 2.4, "rain_conservative_reroute", "rain_conservative_reroute"), move("move_car_wet_slowdown", car, rain_lane_route(rain_edge, rain_s["clear"], rain_s["wet"], car_lateral), 2.4)], 2, scenario_id, "UAV follows a conservative rain route; vehicle follows its wet-road speed policy", "uav_mission", [uav, car], "warning"),
                    event("rain_low_visibility_reroute", fired("rain_speed_reduction"), [set_weather("set_rain_low_visibility", "rain", rain=0.86, wetness=0.9, visibility_m=650.0), move("move_uav_low_visibility_reroute", uav, [uav_wet, uav_heavy], 1.8, "low_visibility_reroute", "low_visibility_reroute"), move("move_car_heavy_rain_caution", car, rain_lane_route(rain_edge, rain_s["wet"], rain_s["heavy"], car_lateral), 1.8)], 3, scenario_id, "Low visibility forces a slower reroute", "weather", [uav, car], "critical", intent="degraded_route"),
                    event("rain_partial_recovery", fired("rain_low_visibility_reroute"), [move("move_uav_partial_rain_recover", uav, [uav_heavy, uav_recovery], 4.0, "partial_recovery_reroute", "partial_recovery_reroute"), move("move_car_partial_rain_recover", car, rain_lane_route(rain_edge, rain_s["heavy"], rain_s["recovery"], car_lateral), 4.2), set_weather("set_rain_partial_recover", "rain", rain=0.25, wetness=0.55, visibility_m=3500.0)], 4, scenario_id, "Rain weakens and the reroute partially recovers", "weather", [uav, car], "info"),
                ]
            )
            desc = "Rain slowdown with low-visibility reroute and partial recovery"
        else:
            terminal_pad = f"pad_weather_abort_{sid}"
            terminal_pad_pos, terminal_pad_yaw, terminal_pad_anchor = building_clear_landing_pad_pose(
                p(ox - 4, oy + 20, 0),
                context=f"{scenario_id} heavy-rain terminal landing pad",
            )
            terminal_pad_hover = q(terminal_pad_pos[0], terminal_pad_pos[1], 5.0)
            terminal_pad_touchdown = q(terminal_pad_pos[0], terminal_pad_pos[1], 1.2)
            add(
                specs,
                scenes,
                pad_entity(
                    terminal_pad,
                    terminal_pad_pos,
                    f"weather_abort_{sid}",
                    "east",
                    terminal_pad_yaw,
                    terminal_pad_anchor,
                ),
            )
            events.extend(
                [
                    event("rain_speed_reduction", fired("rain_condition_met"), [move("move_uav_rain_slowdown", uav, [uav_clear, uav_wet], 2.0, "rain_abort_prepare", "rain_abort_prepare"), move("move_car_wet_slowdown", car, rain_lane_route(rain_edge, rain_s["clear"], rain_s["wet"], car_lateral), 1.4)], 2, scenario_id, "UAV resumes low-speed flight under heavy rain risk; vehicle follows the wet-road speed policy", "uav_mission", [uav, car], "warning"),
                    event("heavy_rain_termination", fired("rain_speed_reduction"), [set_weather("set_heavy_rain_termination", "rain", rain=0.93, wetness=0.95, visibility_m=550.0), move("move_uav_heavy_rain_hold", uav, [uav_wet, uav_heavy], 1.0, "heavy_rain_hold", "heavy_rain_hold"), move("move_car_heavy_rain_safe_stop", car, rain_lane_route(rain_edge, rain_s["wet"], rain_s["heavy"], car_lateral), 0.8)], 3, scenario_id, "Heavy rain prompts mission termination and a low-speed repositioning segment", "weather", [uav, car], "critical"),
                    event("rain_safe_termination", fired("heavy_rain_termination"), [move("move_uav_heavy_rain_terminal_landing", uav, [uav_heavy, terminal_pad_hover, terminal_pad_touchdown], 5.0, "return_to_pad", "landed"), visual("set_car_heavy_rain_safe_stop", car, "safe_stop"), set_weather("set_rain_terminal_heavy", "rain", rain=0.93, wetness=0.95, visibility_m=550.0)], 4, scenario_id, "Heavy rain remains active while the UAV follows its landing trajectory", "uav_mission", [uav, car, terminal_pad], "critical", intent="landing_or_terminal_resolution"),
                ]
            )
            desc = "Heavy rain slowdown with mission termination"
        if scenario_id.endswith("_v1"):
            events.append(
                context_event(
                    "weather_vehicle_leaves_after_recovery",
                    fired("rain_recovery", delay_ticks=80),
                    [remove("remove_weather_vehicle_after_recovery", car)],
                    0,
                    scenario_id,
                    "Recovered traffic leaves the weather event stage after the higher-speed segment",
                    [car],
                )
            )
        elif scenario_id.endswith("_v2"):
            events.append(
                context_event(
                    "weather_vehicle_leaves_after_partial_recovery",
                    fired("rain_partial_recovery", delay_ticks=80),
                    [remove("remove_weather_vehicle_after_partial_recovery", car)],
                    0,
                    scenario_id,
                    "Partially recovered traffic leaves the weather event stage after its reroute segment",
                    [car],
                )
            )
    elif scenario_id.startswith("L5-2"):
        add_default_weather_context()
        trigger = weather("fog", "gte", 0.5, 5)
        profile = {"initial": "clear", "transitions": [{"tick": 190, "profile": "fog", "overrides": {"fog": 0.6, "visibility_m": 180.0}}]}
        events = [
            event("weather_onset", tick(190), [set_weather("set_fog_onset", "fog", fog=0.6, visibility_m=180.0)], 1, scenario_id, "Fog onset is applied before threshold confirmation", "weather", [uav], "info"),
            event("fog_condition_met", trigger, [set_weather("set_fog_dense", "fog", fog=0.62, visibility_m=180.0)], 1, scenario_id, "Fog threshold reached", "weather", [uav], "warning"),
            event("visual_navigation_failure", fired("fog_condition_met"), [visual("set_uav_visual_nav_degraded", uav, "visual_nav_degraded"), move("move_uav_fog_slow_hold", uav, [start, p(ox + 6, oy + 4, 32)], 2.5)], 2, scenario_id, "Visual navigation degrades in fog", "uav_mission", [uav], "critical"),
            event("fog_worsens", fired("visual_navigation_failure"), [set_weather("set_fog_worse_visibility", "fog", fog=0.85, visibility_m=260.0), screenshot("capture_fog_failure")], 3, scenario_id, "Fog worsens and visibility collapses", "weather", [uav], "critical"),
            event("fog_mission_abort", fired("fog_worsens"), [move("move_uav_fog_abort", uav, [p(ox + 6, oy + 4, 32), p(ox - 8, oy + 12, 34)], 4.0)], 4, scenario_id, "UAV aborts mission under dense fog", "uav_mission", [uav], "warning"),
        ]
        desc = "Sudden fog visual navigation failure"
    elif scenario_id.startswith("L5-3"):
        add_default_weather_context()
        trigger = weather("wind_speed", "gte", 12.0, 4)
        profile = {"initial": "clear", "transitions": [{"tick": 210, "profile": "wind", "overrides": {"wind_speed": 12.5, "wind_direction_deg": 90.0}}]}
        bag = f"payload_bag_{sid}"
        add(specs, scenes, world_entity(bag, "prop.service.delivery_bag.v1", "prop", p(ox + 1, oy + 1, 28), 0, "attached"))
        events = [
            event("weather_onset", tick(210), [set_weather("set_wind_onset", "wind", wind_speed=12.5)], 1, scenario_id, "Wind onset is applied before threshold confirmation", "weather", [uav, bag], "info"),
            event("wind_condition_met", trigger, [set_weather("set_crosswind", "wind", wind_speed=12.5)], 1, scenario_id, "Crosswind threshold reached", "weather", [uav, bag], "warning"),
            event("payload_swing", fired("wind_condition_met"), [move("move_payload_swing_path", bag, [p(ox + 1, oy + 1, 28), p(ox + 4, oy - 3, 27), p(ox - 2, oy + 5, 29)], 2.0), visual("set_uav_wind_exposed", uav, "wind_exposed")], 2, scenario_id, "Attached payload swings while the UAV remains exposed to crosswind", "uav_mission", [uav, bag], "warning"),
            event("wind_gust_intensifies", fired("payload_swing"), [set_weather("set_wind_gust", "wind", wind_speed=18.0), move("move_uav_gust_reposition", uav, [start, p(ox + 3, oy + 5, 33)], 2.0), screenshot("capture_wind_payload")], 3, scenario_id, "Crosswind intensifies into gust", "weather", [uav, bag], "critical"),
            event("wind_recovery", fired("wind_gust_intensifies"), [move("move_uav_wind_recovery", uav, [p(ox + 3, oy + 5, 33), p(ox - 8, oy + 10, 34)], 5.0), set_weather("set_wind_recover", "clear", wind_speed=6.0)], 4, scenario_id, "Wind eases and UAV exits gust corridor", "uav_mission", [uav], "info"),
        ]
        desc = "Crosswind exposure with attached payload motion and UAV repositioning"
    elif scenario_id == "L5-4_v1":
        add_default_weather_context()
        profile = {
            "initial": "clear",
            "transitions": [
                {"tick": 250, "profile": "dusk", "overrides": {"illumination_lux": 80.0}},
                {"tick": 330, "profile": "clear", "overrides": {"illumination_lux": 12000.0}},
            ],
        }
        events = [
            event("dusk_light_shift", tick(260), [visual("set_camera_underexposed", uav, "camera_underexposed"), screenshot("capture_dusk_underexposure")], 1, scenario_id, "Dusk light shift underexposes camera", "environment", [uav], "warning"),
            event("infrared_switch", fired("dusk_light_shift"), [visual("set_uav_infrared_mode", uav, "infrared_navigation"), move("move_uav_ir_continue", uav, [start, p(ox + 12, oy + 8, 32)], 5.0)], 2, scenario_id, "UAV switches to infrared navigation", "uav_mission", [uav], "info"),
            event("low_light_termination", fired("infrared_switch"), [screenshot("capture_infrared_validation")], 3, scenario_id, "Low-light segment terminates after validation", "uav_mission", [uav], "info"),
        ]
        desc = "Dusk light shift and camera underexposure"
    else:
        add_default_weather_context()
        charger = f"charger_thermal_{sid}"
        _, charger_scene = add(
            specs,
            scenes,
            world_entity(
                charger,
                "facility.charger.cityops.v1",
                "facility",
                p(ox - 8, oy + 18, 0),
                0,
                "available",
            ),
        )
        charger_position = scene_pos(charger_scene)
        charger_hover = q(charger_position[0], charger_position[1], 8.0)
        charger_touchdown = q(charger_position[0], charger_position[1], 1.2)
        charging_assignment = {
            "facility_id": charger,
            "arrival_tick": 650,
            "departure_tick": 850,
            "physical_service_required": True,
        }
        uav_scene.setdefault("initial_state", {})[
            "charging_service_assignment"
        ] = copy.deepcopy(charging_assignment)
        uav_spec.setdefault("visual_state", {})[
            "charging_service_assignment"
        ] = copy.deepcopy(charging_assignment)
        uav_scene["route_waypoints_enu_m"] = [
            p(ox + 8, oy + 5, 32),
            charger_hover,
            charger_touchdown,
        ]
        uav_spec["movement_waypoints"] = copy.deepcopy(
            uav_scene["route_waypoints_enu_m"]
        )
        profile = {
            "initial": "clear",
            "transitions": [
                {"tick": 240, "profile": "heat", "overrides": {"temperature_c": 46.0}}
            ],
        }
        events = [
            event("high_temperature_process", tick(260), [visual("set_uav_battery_derating", uav, "battery_derating")], 1, scenario_id, "High temperature causes battery derating", "environment", [uav], "warning"),
            event("range_shortened", fired("high_temperature_process"), [move("move_uav_shortened_range", uav, [start, p(ox + 8, oy + 5, 32)], 4.0)], 2, scenario_id, "UAV range shortens under derating", "uav_mission", [uav], "warning"),
            event("charger_decision", fired("range_shortened"), [move("move_uav_to_thermal_charger", uav, [p(ox + 8, oy + 5, 32), charger_hover, charger_touchdown], 4.5), screenshot("capture_temperature_charger_approach")], 3, scenario_id, "UAV diverts to the declared charger and touches down", "uav_mission", [uav, charger], "warning"),
            event("thermal_abort", fired("charger_decision"), [visual("set_uav_thermal_abort_at_charger", uav, "charging_hold")], 4, scenario_id, "High-temperature mission abort completes at the charger", "uav_mission", [uav, charger], "warning"),
            event("thermal_charging_service", fired("thermal_abort", delay_ticks=300), [visual("set_uav_thermal_charging", uav, "charging"), visual("set_thermal_charger_serving", charger, "serving")], 5, scenario_id, "Charger service begins and battery state recovers", "uav_mission", [uav, charger], "info"),
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
            event("c2_recovered", fired("rth_engaged"), [visual("set_tower_link_restored", tower, "online"), visual("set_uav_link_restored_hold", uav, "hover")], 3, scenario_id, "C2 link restored; RTH cancelled and UAV holds its current position", "digital_layer", [uav, tower], "info"),
        ]
        desc = "C2 link loss with return-to-home"
    elif scenario_id.startswith("L6-2"):
        precheck = p(ox + 5, oy + 3, 32)
        delayed = p(ox + 12, oy + 6, 31)
        params["anomaly_tick"] = 400
        events = [
            context_event("uav_c2_precheck_patrol", tick(180), [move("move_uav_l6_2_precheck", uav, [start, precheck], 5.0)], 0, scenario_id, "UAV patrol is visible before intermittent C2 degradation", [uav]),
            event("c2_degradation", tick(400), [visual("set_tower_c2_degraded", tower, "degraded")], 1, scenario_id, "Intermittent C2 degradation starts", "digital_layer", [tower, uav], "warning"),
            event("uav_delayed_response", fired("c2_degradation"), [move("move_uav_delayed_slowdown", uav, [precheck, delayed], 3.0), visual("set_uav_command_delay", uav, "command_delay")], 2, scenario_id, "UAV resumes low-speed flight after a command-link delay", "uav_mission", [uav], "warning"),
            event("backup_link_lock", fired("uav_delayed_response"), [visual("set_tower_backup_lock", tower, "backup_link"), move("move_uav_backup_resume", uav, [delayed, p(ox + 2, oy + 18, 32)], 5.5)], 3, scenario_id, "Backup link reduces degradation", "digital_layer", [uav, tower], "info"),
            event("nominal_c2_restore", fired("backup_link_lock"), [visual("set_tower_nominal", tower, "online")], 4, scenario_id, "Nominal C2 restored", "digital_layer", [tower], "info"),
        ]
        desc = "Intermittent C2 degradation with slowdown and backup link"
    elif scenario_id.startswith("L6-3"):
        planned = p(ox + 12, oy + 8, 31)
        spoofed = p(ox + 27, oy + 18, 31)
        corrected = p(ox + 8, oy + 22, 32)
        uav_scene = next(scene for scene in scenes if scene.get("entity_id") == uav)
        uav_spec = next(spec for spec in specs if spec.get("entity_id") == uav)
        uav_scene["planned_route_waypoints_enu_m"] = [start, planned, corrected]
        uav_scene["path_deviation_contract"] = {
            "minimum_route_distance_m": 15.0,
            "predicate_id": "positioning.aircraft_deviates_from_planned_route",
        }
        uav_spec["planned_route_waypoints_enu_m"] = [start, planned, corrected]
        events = [
            event("spoofing_start", tick(240), [visual("set_gnss_spoofing_detected", tower, "gnss_spoofing")], 1, scenario_id, "GNSS spoofing starts", "digital_layer", [uav, tower], "critical"),
            event("spoofed_route_offset", fired("spoofing_start"), [move("move_uav_spoofed_offset", uav, [start, planned, spoofed], 6.0)], 2, scenario_id, "UAV deviates from planned route by more than 15m", "uav_mission", [uav], "critical"),
            event("safety_route_lock", fired("spoofed_route_offset"), [visual("set_uav_nav_lock", uav, "navigation_lock"), move("move_uav_spoof_corrected", uav, [spoofed, corrected], 4.0), screenshot("capture_spoof_correction")], 3, scenario_id, "Safety mechanism locks route and corrects spoofing", "digital_layer", [uav, tower], "warning"),
            event("spoof_recovery", fired("safety_route_lock"), [visual("set_tower_gnss_normal", tower, "online")], 4, scenario_id, "GNSS spoofing cleared", "digital_layer", [tower, uav], "info"),
        ]
        params["path_deviation_min_m"] = 15.0
        desc = "GNSS spoofing route hijack with actual offset"
    elif scenario_id.startswith("L6-4"):
        uav_b = f"uav_digital_{sid}_secondary"
        add(specs, scenes, world_entity(uav_b, "uav.airsim.flying_pawn.v1", "uav", p(ox - 12, oy + 7, 34), 50, "patrol"))
        events = [
            event("wideband_jamming", tick(240), [visual("set_tower_jamming", tower, "jammed"), visual("set_uav_a_jammed", uav, "jammed"), visual("set_uav_b_jammed", uav_b, "jammed")], 1, scenario_id, "Wideband jamming affects two UAVs", "digital_layer", [uav, uav_b, tower], "critical"),
            event("multi_uav_hold_entry", fired("wideband_jamming"), [move("move_uav_a_jam_hold", uav, [start, p(ox + 4, oy + 4, 35)], 2.0), move("move_uav_b_jam_hold", uav_b, [p(ox - 12, oy + 7, 34), p(ox - 7, oy + 12, 36)], 2.0)], 2, scenario_id, "Both UAVs maneuver to deterministic jamming hold points", "uav_mission", [uav, uav_b], "warning"),
            event("multi_uav_safe_hold", fired("multi_uav_hold_entry", delay_ticks=45), [], 3, scenario_id, "Both UAVs remain stationary in safe hold under jamming", "uav_mission", [uav, uav_b], "warning"),
            event("alternate_channel", fired("multi_uav_safe_hold", delay_ticks=60), [visual("set_tower_alternate_channel", tower, "alternate_channel"), move("move_uav_a_channel_recover", uav, [p(ox + 4, oy + 4, 35), p(ox + 15, oy + 12, 33)], 5.5), move("move_uav_b_channel_recover", uav_b, [p(ox - 7, oy + 12, 36), p(ox - 18, oy + 18, 34)], 5.5)], 4, scenario_id, "Alternate channel recovers both UAVs", "digital_layer", [uav, uav_b, tower], "info"),
        ]
        desc = "Wideband jamming affecting multiple UAVs"
    else:
        gcs = f"gcs_anchor_{sid}"
        add(
            specs,
            scenes,
            world_entity(
                gcs,
                "semantic.asset_anchor",
                "ground_station",
                p(ox + 5, oy - 5, 0),
                0,
                "gcs_online",
                facility_subtype="ground_control_station",
            ),
        )
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
        uav, tower, ped = "uav_x1_forced_landing", "tower_x1_c2", "ped_x1_crowd_x1_08"
        landing_anchor = _find_forced_landing_ground_anchor(
            p(ox + 16, oy + 9, GROUND_Z_M),
            scenario_id=scenario_id,
        )
        landing_point = landing_anchor.position_enu_m
        landing_x, landing_y = float(landing_point[0]), float(landing_point[1])
        start = q(landing_x - 28.0, landing_y - 22.0, 32.0)
        descent_entry = q(landing_x, landing_y, 26.0)
        descent_mid = q(landing_x, landing_y, 18.0)
        low = q(landing_x, landing_y, 8.0)
        # Keep the pre-touchdown hold outside the governed landing-zone altitude
        # tolerance (2.5 m above the 0.8 m endpoint).  A 3.0 m hold was close
        # enough to be latched as a numeric touchdown while the UAV was still
        # waiting for the crowd to clear.
        landing = q(landing_x, landing_y, 4.0)
        touchdown = q(landing_x, landing_y, 0.8)

        _, tower_scene = add(
            specs,
            scenes,
            world_entity(
                tower,
                "facility.radio.base_tower.v1",
                "facility",
                q(landing_x - 18.0, landing_y - 12.0, GROUND_Z_M),
                0,
                "online",
            ),
        )
        tower_scene["communication_state"] = {
            "availability": 1.0,
            "station_unavailable": False,
        }
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 30, "rain_patrol"))

        occupied_peds: list[list[float]] = []
        ped_anchor = _pedestrian_anchor_near_landing(
            landing_anchor,
            landing_point,
            preferred_s_delta_m=-4.5,
            occupied=occupied_peds,
        )
        ped_spec, ped_scene = world_entity(
            ped,
            "pedestrian.cityops.basic.v1",
            "pedestrian",
            ped_anchor.position_enu_m,
            ped_anchor.sample.yaw_deg,
            "observing",
        )
        _apply_sidewalk_anchor_to_entity(ped_spec, ped_scene, ped_anchor)
        add(specs, scenes, (ped_spec, ped_scene))
        ped_start = scene_pos(ped_scene)
        ped_start, ped_context = pedestrian_preamble_walk_event(
            ped_spec,
            ped_scene,
            scenario_id,
            "x1_ped_approach",
            "move_x1_ped_approach",
            dx_m=-12.0,
            dy_m=9.0,
            activity_type="walking",
        )
        crowd_origin = gathering_point(q(landing_x, landing_y, GROUND_Z_M), [800, 500, 0])
        extra_crowd_ids = add_static_ped_cohort(specs, scenes, "x1", "ped_x1_crowd", crowd_origin, 8, activity_type="observing")
        crowd_ids = [*extra_crowd_ids, ped]
        crowd_context_ids = list(crowd_ids)

        crowd_positions = {
            ped: ped_start,
            **{
                entity_id: scene_pos(next(scene for scene in scenes if scene.get("entity_id") == entity_id))
                for entity_id in extra_crowd_ids
            },
        }
        crowd_evacuation_actions: list[dict[str, Any]] = []
        crowd_clearance_targets: dict[str, list[float]] = {}
        selected_clearance_targets: list[list[float]] = []
        for crowd_index, crowd_id in enumerate(crowd_ids):
            crowd_start = crowd_positions[crowd_id]
            lane_sample = LANES.nearest_to_xy(float(crowd_start[0]), float(crowd_start[1]))
            yaw_rad = math.radians(lane_sample.yaw_deg)
            target_candidates: list[tuple[float, float, list[float], list[list[float]]]] = []
            for direction in (-1.0, 1.0):
                for travel_m in (18.0, 15.0, 12.0, 9.0):
                    for lateral_rank in range(9):
                        try:
                            target = shifted_sidewalk_point(
                                crowd_start,
                                direction * travel_m * math.cos(yaw_rad),
                                direction * travel_m * math.sin(yaw_rad),
                                GATHERING_MIN_OFFSET_FROM_CURB_M
                                + 1.35 * float((crowd_index + lateral_rank) % 9),
                            )
                        except RuntimeError:
                            continue
                        movement_m = dist_xy(crowd_start, target)
                        clearance_m = dist_xy(target, touchdown)
                        if movement_m < 8.0 or clearance_m <= 12.0:
                            continue
                        if any(
                            dist_xy(target, selected) < 1.5
                            for selected in selected_clearance_targets
                        ):
                            continue
                        route = SPATIAL.plan_sidewalk_route(
                            [crowd_start, target],
                            allow_green=True,
                            context=f"{scenario_id}:{crowd_id}:separated_crowd_clearance",
                        )
                        target_candidates.append(
                            (-clearance_m, -movement_m, target, route)
                        )
            if not target_candidates:
                raise RuntimeError(
                    f"{scenario_id}: no separated physical crowd-clearance target for {crowd_id}"
                )
            _clearance_score, _movement_score, crowd_target, crowd_route = min(
                target_candidates,
                key=lambda item: (item[0], item[1], item[2]),
            )
            selected_clearance_targets.append(crowd_target)
            crowd_clearance_targets[crowd_id] = crowd_target
            crowd_evacuation_actions.append(
                move(
                    f"move_x1_crowd_clear_{crowd_index:02d}",
                    crowd_id,
                    crowd_route,
                    2.0,
                    post_activity_type="observing",
                )
            )

        weather_profile = {"initial": "clear", "transitions": [{"tick": 180, "profile": "rain", "overrides": {"rain": 0.62, "visibility_m": 1600.0}}]}
        events = [
            ped_context,
            context_event("crowd_present", tick(150), [ped_activity(f"set_{ped_id}_crowd_observing", ped_id, "observing") for ped_id in crowd_context_ids], 0, scenario_id, "Crowd is visible near the future forced-landing area before the chain begins", crowd_context_ids),
            event("rain_onset", tick(180), [set_weather("set_x1_rain_onset", "rain", rain=0.62, visibility_m=1600.0)], 1, scenario_id, "Rain onset is applied before C2 coupling", "weather", [uav, tower], "info"),
            event(
                "rain_threshold",
                weather("rain", "gte", 0.5, 5),
                [
                    set_weather("set_x1_rain", "rain", rain=0.65, visibility_m=1500.0),
                    action(
                        "set_x1_tower_rain_degraded_state",
                        "set_runtime_state",
                        entity_id=tower,
                        delay_ticks=0,
                        state_patch={
                            "communication_state": {
                                "availability": 0.45,
                                "station_unavailable": False,
                            }
                        },
                    ),
                    visual("set_x1_tower_stressed", tower, "rain_stressed"),
                ],
                1,
                scenario_id,
                "Sustained rain attenuates the channel and degrades tower availability",
                "weather",
                [uav, tower],
                "warning",
            ),
            event(
                "c2_loss_tick",
                fired("rain_threshold", delay_ticks=75),
                [
                    action(
                        "set_x1_tower_unavailable_state",
                        "set_runtime_state",
                        entity_id=tower,
                        delay_ticks=0,
                        state_patch={
                            "communication_state": {
                                "availability": 0.0,
                                "station_unavailable": True,
                            }
                        },
                    ),
                    visual("set_x1_tower_c2_loss", tower, "link_lost"),
                ],
                2,
                scenario_id,
                "Persistent rain stress drives measured C2 availability to zero",
                "digital_layer",
                [uav, tower],
                "critical",
                intent="digital_anomaly",
            ),
            event(
                "c2_loss_after_rain",
                fired("c2_loss_tick", delay_ticks=10),
                [move("move_x1_uav_degraded", uav, [start, descent_entry], 6.0)],
                3,
                scenario_id,
                "Sustained link loss leaves the UAV without nominal C2 guidance",
                "digital_layer",
                [uav, tower],
                "critical",
                intent="digital_anomaly",
            ),
            event(
                "c2_safe_hold",
                fired("c2_loss_after_rain", delay_ticks=70),
                [],
                4,
                scenario_id,
                "The UAV remains stationary while the communication failsafe commits to landing",
                "uav_mission",
                [uav, tower],
                "warning",
                intent="safe_hold",
            ),
            event(
                "forced_landing_descent",
                fired("c2_safe_hold", delay_ticks=30),
                [move("move_x1_forced_landing", uav, [descent_entry, descent_mid, low, landing], FORCED_LANDING_DESCENT_SPEED_MPS, "descending", "low_altitude_hold")],
                5,
                scenario_id,
                "Communication failsafe produces a continuous forced descent toward a validated landing zone",
                "uav_mission",
                [uav, ped],
                "critical",
            ),
            event(
                "crowd_reacts_to_landing",
                prox(
                    uav,
                    ped,
                    7.5,
                    2,
                    metric="xy_plus_z",
                    horizontal_distance_m=7.5,
                    vertical_distance_m=10.0,
                ),
                [*crowd_evacuation_actions, screenshot("capture_x1_forced_landing")],
                6,
                scenario_id,
                "The full crowd clears the validated touchdown zone during the descent",
                "pedestrian",
                [uav, *crowd_ids],
                "warning",
            ),
            event(
                "x1_recovery",
                fired("crowd_reacts_to_landing"),
                [
                    move("move_x1_safe_touchdown", uav, [landing, touchdown], FORCED_LANDING_TOUCHDOWN_SPEED_MPS, "descending", "landed"),
                    action(
                        "set_x1_tower_restored_state",
                        "set_runtime_state",
                        entity_id=tower,
                        delay_ticks=0,
                        state_patch={
                            "communication_state": {
                                "availability": 1.0,
                                "station_unavailable": False,
                            }
                        },
                    ),
                    visual("set_x1_tower_restored", tower, "online"),
                ],
                6,
                scenario_id,
                "Cleared ground geometry permits numeric touchdown while C2 availability recovers",
                "digital_layer",
                [uav, tower, *crowd_ids],
                "info",
                intent="landing_or_terminal_resolution",
            ),
        ]
        params["x1_physical_chain"] = {
            "rain_threshold": 0.5,
            "tower_nominal_availability": 1.0,
            "tower_degraded_availability": 0.45,
            "tower_unavailable_availability": 0.0,
            "forced_descent_z_m": [30.0, 18.0, 8.0, 4.0, 0.8],
            "landing_point_enu_m": touchdown,
            "crowd_clearance_radius_m": 12.0,
            "crowd_clearance_actor_ids": crowd_ids,
            "crowd_clearance_targets_enu_m": crowd_clearance_targets,
            "semantic_authority": "weather_plus_structured_communication_state_plus_numeric_trajectory",
        }
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
        domain_state_profile = json.loads(
            DOMAIN_STATE_SUPPLEMENT_PROFILE_PATH.read_text(encoding="utf-8-sig")
        )
        medical_response_profile = dict(
            (domain_state_profile.get("domain_models") or {}).get("medical_response") or {}
        )
        uav_detection_radius_m = float(medical_response_profile["uav_detection_radius_m"])
        uav_detection_required_ticks = int(
            medical_response_profile["uav_detection_required_ticks"]
        )
        ped_spec, ped_scene = add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_32", 50, 1.2, p(ox + 4, oy + 3, 0), 0, "walking"))
        ped_start = scene_pos(ped_scene)
        # A preflight UAV must begin outside the same governed 3-D detection
        # radius used by the medical chain.  Reserve a horizontal pad-clearance
        # disc around the patient so automatic lifecycle placement cannot make
        # the proximity detector true before the fall.
        ped_scene["uav_home_pad_clearance_m"] = round(
            uav_detection_radius_m + 1.0,
            3,
        )
        ped_start, ped_context = pedestrian_preamble_walk_event(
            ped_spec,
            ped_scene,
            scenario_id,
            "x3_ped_approach",
            "move_x3_ped_approach",
            dx_m=-12.0,
            dy_m=8.0,
        )
        # Start clearly outside the governed 35 m 3-D medical-detection radius,
        # while keeping the emergency observation and return-to-pad mission
        # spatially compact enough to complete within the formal episode.
        uav_start = q(ped_start[0] - 48.0, ped_start[1] - 32.0, 32)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 60, "patrol"))
        arrival_center = road_center_point(ped_start)
        arrival_sample = LANES.nearest_to_xy(arrival_center[0], arrival_center[1])
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
        ambulance_lateral = float((ambulance_scene.get("placement") or {}).get("resolved_lateral_from_center_m") or 0.0)
        ambulance_arrival_s = float(arrival_sample.s_m)
        ambulance_route_spacing_m = 3.0
        ambulance_route_segments = max(
            1,
            int(math.ceil(abs(ambulance_arrival_s - ambulance_s) / ambulance_route_spacing_m)),
        )
        ambulance_route = [
            _offset_from_lane(
                LANES.resolve_edge_s(
                    arrival_sample.edge_id,
                    ambulance_s
                    + (ambulance_arrival_s - ambulance_s) * segment_index / ambulance_route_segments,
                ),
                ambulance_lateral,
                GROUND_Z_M,
            )
            for segment_index in range(ambulance_route_segments + 1)
        ]
        ambulance_route[0] = ambulance_start
        ambulance_arrival = ambulance_route[-1]
        # The governed medical-response detector uses full 3-D range.  A 30 m
        # AGL, roughly 12 m horizontal standoff gives a 32.3 m observation
        # range: inside the 35 m boundary, but only near the end of the approach.
        # This makes the independent 15-tick dwell occur after the UAV is holding
        # position instead of letting an activity label stand in for detection.
        uav_observe = q(ped_start[0] - 10.0, ped_start[1] - 6.5, 30)
        _, tape_scene = add(specs, scenes, world_entity(tape, "prop.incident.police_tape.v1", "prop", shifted_sidewalk_point(ped_start, 1.5, 1.0, 1.5), 0, "staged", activation_tick=901))
        tape_pos = scene_pos(tape_scene)
        events = [
            ped_context,
            event("pedestrian_fall", tick(220), [play("play_x3_ped_fall", ped)], 1, scenario_id, "Pedestrian falls", "pedestrian", [ped], "critical"),
            event("observation_approach", fired("pedestrian_fall"), [move("move_x3_uav_observation_approach", uav, [uav_start, uav_observe], 5.0, activity_type="cruise", post_activity_type="observation_hold")], 2, scenario_id, "UAV begins a physical observation approach after the pedestrian fall", "uav_mission", [uav, ped], "warning"),
            event("uav_detection", prox(uav, ped, uav_detection_radius_m, uav_detection_required_ticks, metric="3d"), [visual("set_x3_uav_detecting", uav, "detecting"), screenshot("capture_x3_detection")], 3, scenario_id, "UAV confirms the fallen pedestrian after governed 3D proximity dwell", "uav_mission", [uav, ped], "warning"),
            event("ambulance_dispatch", fired("uav_detection", delay_ticks=5), [move("move_x3_ambulance_dispatch", ambulance, ambulance_route, 12.0)], 4, scenario_id, "Ambulance is dispatched only after physical UAV detection", "vehicle", [ambulance, ped], "warning", validation_event_type=PEDESTRIAN_VEHICLE_INTERACTION_EVENT_TYPE, validation_reason="Cross-layer emergency response intentionally sends the ambulance close to the fallen pedestrian.", validation_skip_checks=[PEDESTRIAN_VEHICLE_CLEARANCE_CHECK]),
            event("ambulance_arrived", prox(ambulance, ped, 8.0, 2), [screenshot("capture_x3_arrival")], 5, scenario_id, "Ambulance physically reaches the fallen pedestrian", "vehicle", [ambulance, ped], "warning"),
            event("isolation_setup", fired("ambulance_arrived", delay_ticks=1), [spawn_from_scene("spawn_x3_police_tape", tape_scene)], 6, scenario_id, "Responder establishes the incident isolation perimeter after arrival", "dynamic_constraint", [tape, ped], "warning"),
            event("responder_handoff", fired("isolation_setup", delay_ticks=15), [screenshot("capture_x3_handoff")], 7, scenario_id, "Responder remains with the patient through the handoff dwell", "vehicle", [ambulance, ped], "info"),
            event("emergency_documented", fired("responder_handoff"), [visual("set_x3_police_tape_standdown", tape, "standdown"), screenshot("capture_x3_response")], 8, scenario_id, "Emergency response is documented after physical handoff", "uav_mission", [uav, ambulance, ped, tape], "info"),
        ]
        params["x3_physical_chain"] = {
            "ambulance_lane_edge_id": arrival_sample.edge_id,
            "ambulance_lane_start_s_m": round(float(ambulance_s), 3),
            "ambulance_lane_arrival_s_m": round(float(ambulance_arrival_s), 3),
            "ambulance_route_sample_spacing_m": ambulance_route_spacing_m,
            "uav_activity_sequence": [
                "preflight_on_pad",
                "takeoff",
                "cruise",
                "observation_hold",
                "detecting",
            ],
            "uav_detection_radius_m": uav_detection_radius_m,
            "uav_detection_required_ticks": uav_detection_required_ticks,
            "uav_detection_trigger_metric": "3d",
            "uav_detection_parameter_source": (
                "Dataset/semantic_rules/profiles/domain_state_supplement_profile.json"
            ),
            "medical_detection_truth_authority": (
                "confirmed_physical_fall_plus_nearest_uav_3d_geometry_plus_uninterrupted_dwell"
            ),
            "authored_activity_labels_are_detection_evidence": False,
        }
        desc = "Pedestrian fall to emergency response chain"
    elif short_id == "X4":
        scenario_id = dirname
        uav_a, uav_b = "uav_x4_a", "uav_x4_b"
        a0, b0 = p(ox - 18, oy, 29), p(ox + 20, oy + 10, 31)
        meet = p(ox + 1, oy + 5, 30)
        add(specs, scenes, world_entity(uav_a, "uav.inspect.quad.v1", "uav", a0, 70, "patrol"))
        add(specs, scenes, world_entity(uav_b, "uav.airsim.flying_pawn.v1", "uav", b0, 250, "patrol"))
        weather_profile = {"initial": "clear", "transitions": [{"tick": 220, "profile": "fog", "overrides": {"fog": 0.62, "visibility_m": 160.0}}]}
        events = [
            event("fog_onset", tick(220), [set_weather("set_x4_fog_onset", "fog", fog=0.62, visibility_m=160.0)], 1, scenario_id, "Fog onset is applied before conflict chain", "weather", [uav_a, uav_b], "info"),
            event("fog_threshold", weather("fog", "gte", 0.5, 5), [set_weather("set_x4_fog", "fog", fog=0.65, visibility_m=160.0)], 1, scenario_id, "Fog reaches operational threshold", "weather", [uav_a, uav_b], "warning"),
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
            event("backup_pad_unavailable", fired("station_failure"), [visual("set_x5_backup_pad_unavailable", pad_b, "unavailable")], 2, scenario_id, "Only one landing pad remains available", "infrastructure", [pad_a, pad_b], "warning", intent="pad_contention"),
            event("dual_uav_pad_contention", fired("backup_pad_unavailable"), [move("move_x5_uav_a_to_pad", uav_a, [p(ox - 18, oy + 12, 33), pad_a_hover], 5.0), move("move_x5_uav_b_to_pad_hold", uav_b, [p(ox + 18, oy - 12, 31), p(ox + 8, oy + 10, 34)], 5.0)], 3, scenario_id, "Two UAVs contend for one available pad", "uav_mission", [uav_a, uav_b, pad_a], "critical"),
            event("pad_priority_arbitration", prox(uav_a, pad_a, 7.0, 2), [visual("set_x5_pad_reserved", pad_a, "reserved"), visual("set_x5_uav_b_hold", uav_b, "hold")], 4, scenario_id, "Priority arbitration reserves pad for first UAV", "infrastructure", [uav_a, uav_b, pad_a], "warning"),
            event("second_uav_reroute", fired("pad_priority_arbitration"), [move("move_x5_uav_b_reroute", uav_b, [p(ox + 8, oy + 10, 34), p(ox + 34, oy + 24, 34)], 8.0)], 5, scenario_id, "Second UAV reroutes after contention", "uav_mission", [uav_b], "info"),
            event("station_recovered", fired("second_uav_reroute"), [visual("set_x5_tower_restored", tower, "online")], 6, scenario_id, "Communication station recovers", "infrastructure", [tower], "info"),
        ]
        desc = "Communication failure to pad contention chain"
    else:
        scenario_id = dirname
        uav, nfz, anchor = "uav_x6_monitor", "nfz_x6_lockdown", "crowd_anchor_x6"
        safe_anchor = "safe_zone_anchor_x6"
        reroute_exit_anchor = "reroute_exit_anchor_x6"
        # The monitor occupies the governed region while it is inactive.  Once
        # the evacuation gate activates the lockdown, the reroute action below
        # carries the affected aircraft through the southern exit corridor.
        uav_start = p(ox + 6, oy + 4, 34)
        crowd_extent = [900, 600, 0]
        safe_extent = [900, 500, 0]
        crowd_origin = gathering_point(p(ox + 4, oy + 3, 0), crowd_extent)
        safe_origin = gathering_point(p(ox + 28, oy + 18, 0), safe_extent)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 55, "monitor"))
        add(specs, scenes, world_entity(anchor, "semantic.spawn_zone", "crowd_anchor", crowd_origin, 0))
        cohort = add_evacuation_ped_cohort(
            specs,
            scenes,
            "x6",
            "ped_x6_evac",
            crowd_origin,
            safe_origin,
            8,
            min_safe_distance_from_origin_m=32.0,
        )
        scene_by_entity_id = {
            str(scene_entity["entity_id"]): scene_entity
            for scene_entity in scenes
        }
        cohort_placements = {
            ped_id: dict(scene_by_entity_id[ped_id]["placement"])
            for ped_id, _start, _route in cohort
        }
        cohort_edge_ids = {
            str(placement["lane_edge_id"])
            for placement in cohort_placements.values()
        }
        if len(cohort_edge_ids) != 1:
            raise RuntimeError(
                f"{scenario_id}: evacuation cohort spans multiple sidewalk components: "
                f"{sorted(cohort_edge_ids)}"
            )
        cohort_edge_id = next(iter(cohort_edge_ids))
        min_edge_s, max_edge_s = LANES.edge_s_bounds(cohort_edge_id)
        target_column_spacing_m = 1.8
        target_row_spacing_m = 2.0
        longitudinal_margin_m = 1.5 * target_column_spacing_m + 0.25
        center_s_candidates = [
            min_edge_s + longitudinal_margin_m + float(index)
            for index in range(
                max(
                    0,
                    int(
                        math.floor(
                            max_edge_s
                            - min_edge_s
                            - 2.0 * longitudinal_margin_m
                        )
                    )
                    + 1,
                )
            )
        ]
        center_s_candidates.sort(
            key=lambda candidate_s: (
                dist_xy(
                    _offset_from_lane(
                        LANES.resolve_edge_s(cohort_edge_id, candidate_s),
                        0.0,
                        GROUND_Z_M,
                    ),
                    safe_origin,
                ),
                candidate_s,
            )
        )
        compact_cohort = None
        compact_errors: list[str] = []
        for center_s in center_s_candidates:
            candidate_cohort: list[tuple[str, list[float], list[list[float]]]] = []
            candidate_targets: list[list[float]] = []
            try:
                for cohort_index, (ped_id, start, _route) in enumerate(cohort):
                    column = cohort_index % 4
                    row = cohort_index // 4
                    target_s = center_s + (column - 1.5) * target_column_spacing_m
                    target_sample = LANES.resolve_edge_s(cohort_edge_id, target_s)
                    start_placement = cohort_placements[ped_id]
                    start_sample = LANES.resolve_edge_s(
                        cohort_edge_id,
                        float(start_placement["longitudinal_s"]),
                    )
                    side_sign = _side_sign_for_desired(start_sample, start)
                    offset_from_curb_m = (
                        GATHERING_MIN_OFFSET_FROM_CURB_M
                        + row * target_row_spacing_m
                    )
                    desired_target = _offset_from_lane(
                        target_sample,
                        side_sign * (LANE_HALF_WIDTH_M + offset_from_curb_m),
                        GROUND_Z_M,
                    )
                    target_anchor = SPATIAL.plan_sidewalk_anchor(
                        desired_target,
                        edge_id_hint=cohort_edge_id,
                        s_hint=target_s,
                        offset_from_curb_m=offset_from_curb_m,
                        allow_green=True,
                        placement_semantics="x6_shared_safe_zone_target",
                    )
                    target = target_anchor.position_enu_m
                    if target_anchor.sample.edge_id != cohort_edge_id:
                        raise SpatialValidationError(
                            f"target changed edge to {target_anchor.sample.edge_id}"
                        )
                    if _side_sign_for_desired(target_anchor.sample, target) * side_sign <= 0.0:
                        raise SpatialValidationError("target crossed the authoritative sidewalk side")
                    if dist_xy(target, crowd_origin) < 32.0:
                        raise SpatialValidationError("target is inside the 32 m hazard clearance")
                    planned_route = SPATIAL.plan_sidewalk_route(
                        [start, target],
                        allow_green=True,
                        context=f"{scenario_id} shared safe-zone route {ped_id}",
                    )
                    candidate_cohort.append((ped_id, start, planned_route))
                    candidate_targets.append(target)
                candidate_center = [
                    sum(float(target[axis]) for target in candidate_targets)
                    / len(candidate_targets)
                    for axis in range(3)
                ]
                if max(dist_xy(target, candidate_center) for target in candidate_targets) > 8.0:
                    raise SpatialValidationError("shared safe-zone targets exceed the 8 m design radius")
                if any(
                    dist_xy(a, b) < 0.9
                    for target_index, a in enumerate(candidate_targets)
                    for b in candidate_targets[target_index + 1 :]
                ):
                    raise SpatialValidationError("shared safe-zone targets overlap")
                compact_cohort = candidate_cohort
                break
            except SpatialValidationError as exc:
                compact_errors.append(f"s={center_s:.3f}: {exc}")
        if compact_cohort is None:
            raise RuntimeError(
                f"{scenario_id}: no reachable shared safe-zone cluster on "
                f"{cohort_edge_id}: {compact_errors[:8]}"
            )
        cohort = compact_cohort
        cohort_ids = [ped_id for ped_id, _start, _route in cohort]
        safe_targets = {ped_id: route[-1] for ped_id, _start, route in cohort}
        safe_anchor_position = [
            round(
                sum(float(position[axis]) for position in safe_targets.values())
                / len(safe_targets),
                6,
            )
            for axis in range(3)
        ]
        # After the initial exit leg, the reroute remains south of the lockdown
        # prism.  Keeping the exit anchor on that corridor prevents the
        # building-clearance planner from introducing a return leg.
        reroute_exit_position = p(ox + 50, oy - 16, 36)
        add(
            specs,
            scenes,
            world_entity(
                safe_anchor,
                "semantic.spawn_zone",
                "crowd_anchor",
                safe_anchor_position,
                0,
                "safe_zone_reference",
            ),
        )
        add(
            specs,
            scenes,
            world_entity(
                reroute_exit_anchor,
                "semantic.spawn_zone",
                "crowd_anchor",
                reroute_exit_position,
                0,
                "reroute_exit_reference",
            ),
        )
        evacuation_actions = evacuation_move_actions(cohort, "move_x6_crowd_evacuation")
        _, nfz_scene = add(
            specs,
            scenes,
            box_entity(
                nfz,
                "trigger.no_fly.box.v1",
                "airspace_constraint",
                p(ox + 6, oy + 4, 28),
                [18, 13, 16],
                # The physical spawn action is the only activation authority.
                # A fixed activation tick would bypass the population gate when
                # a cohort member fails to reach the safe zone.
                activation_tick=901,
            ),
        )
        safe_condition_event_ids = [
            f"x6_safe_zone_member_{index:02d}"
            for index in range(len(cohort_ids))
        ]
        safe_condition_events = [
            context_event(
                condition_event_id,
                prox(
                    ped_id,
                    safe_anchor,
                    10.0,
                    10,
                    metric="xy_plus_z",
                    horizontal_distance_m=10.0,
                    vertical_distance_m=5.0,
                ),
                [],
                2,
                scenario_id,
                f"Explicit evacuation cohort member {index + 1} remains in the physical safe zone",
                [ped_id, safe_anchor],
            )
            for index, (condition_event_id, ped_id) in enumerate(
                zip(safe_condition_event_ids, cohort_ids)
            )
        ]
        reroute_exit_condition_event_id = "x6_uav_at_reroute_exit"
        events = [
            context_event("crowd_present", tick(120), [], 0, scenario_id, "The explicit pedestrian cohort occupies the hazard-side crowd region before evacuation", [anchor, *cohort_ids]),
            event("crowd_evacuation", tick(220), evacuation_actions, 1, scenario_id, "Evacuation activation starts continuous pedestrian trajectories away from the hazard region", "pedestrian", [anchor, *cohort_ids], "warning"),
            *safe_condition_events,
            event("crowd_safe_zone_move", composite("AND", safe_condition_event_ids), [screenshot("capture_x6_crowd_safe_zone")], 2, scenario_id, "All eight explicit cohort members satisfy the physical safe-zone dwell threshold", "pedestrian", [anchor, safe_anchor, *cohort_ids], "warning"),
            event("airspace_lockdown", fired("crowd_safe_zone_move", delay_ticks=40), [spawn_from_scene("spawn_x6_nfz", nfz_scene, visual_state={"mode": "active"})], 3, scenario_id, "The completed physical evacuation threshold activates a temporary no-fly region after the governed clearance interval", "dynamic_constraint", [nfz, safe_anchor, *cohort_ids], "critical"),
            event(
                "uav_reroute_lockdown",
                fired("airspace_lockdown", delay_ticks=0),
                [
                    move(
                        "move_x6_uav_avoid_nfz",
                        uav,
                        [
                            uav_start,
                            p(ox - 18, oy - 16, 36),
                            reroute_exit_position,
                        ],
                        7.0,
                    ),
                    screenshot("capture_x6_lockdown"),
                ],
                4,
                scenario_id,
                "The affected UAV follows a continuous route around the active no-fly region",
                "uav_mission",
                [uav, nfz],
                "warning",
            ),
            context_event(
                reroute_exit_condition_event_id,
                prox(
                    uav,
                    reroute_exit_anchor,
                    5.0,
                    10,
                    metric="xy_plus_z",
                    horizontal_distance_m=5.0,
                    vertical_distance_m=8.0,
                ),
                [],
                5,
                scenario_id,
                "The rerouted UAV remains at the observed exit anchor outside the no-fly volume",
                [uav, reroute_exit_anchor],
            ),
            event("lockdown_cleared", composite("AND", [*safe_condition_event_ids, reroute_exit_condition_event_id]), [visual("set_x6_nfz_standdown", nfz, "standdown")], 5, scenario_id, "Physical crowd clearance and completed UAV reroute permit lockdown release", "dynamic_constraint", [uav, nfz, safe_anchor, reroute_exit_anchor, *cohort_ids], "info"),
        ]
        params["x6_physical_chain"] = {
            "hazard_reference_enu_m": crowd_origin,
            "crowd_hazard_radius_m": 18.0,
            "safe_zone_distance_m": 28.0,
            "safe_zone_reached_fraction": 1.0,
            "minimum_route_target_distance_from_origin_m": 32.0,
            "evacuation_actor_ids": cohort_ids,
            "safe_targets_enu_m": safe_targets,
            "safe_zone_anchor_entity_id": safe_anchor,
            "safe_zone_anchor_enu_m": safe_anchor_position,
            "safe_zone_anchor_radius_m": 10.0,
            "safe_zone_required_dwell_ticks": 10,
            "lockdown_region_entity_id": nfz,
            "lockdown_activation_gate": "all_eight_explicit_cohort_members_at_safe_zone_anchor_for_required_dwell",
            "reroute_exit_anchor_entity_id": reroute_exit_anchor,
            "reroute_exit_anchor_enu_m": reroute_exit_position,
            "reroute_exit_radius_m": 5.0,
            "reroute_exit_required_dwell_ticks": 10,
            "lockdown_clearance_gate": "all_eight_safe_zone_conditions_and_uav_at_reroute_exit_anchor",
            "reroute_authority": "numeric_uav_trajectory_outside_active_region_geometry",
            "safe_zone_authority": "pedestrian_pose_plus_fixed_initial_hazard_reference_plus_population_threshold",
        }
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
        causal_predecessor_event_ids=data.get("causal_predecessor_event_ids", []),
        causal_predecessor_mode=data.get("causal_predecessor_mode", ""),
        causal_temporal_policy=data.get("causal_temporal_policy", ""),
        target_roles=data.get("target_roles", []),
        validation_event_type=data.get("validation_event_type", ""),
        validation_reason=data.get("validation_reason", ""),
        validation_skip_checks=data.get("validation_skip_checks", []),
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
        "simulation_plans": {
            "charging_service_plan": build_charging_service_plan(
                bundle.scenario_id,
                bundle.scene_entities,
            )
        },
        "spawn_sequencing": [
            {"entity_id": e["entity_id"], "tick": e.get("activation_tick", 0)}
            for e in bundle.scene_entities
        ],
        "weather_profile": bundle.weather_profile,
        "cameras": bundle.cameras,
        "validation_rules": bundle.validation_rules,
    }


def build_charging_service_plan(
    scenario_id: str,
    scene_entities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    uav_ids = sorted(
        str(entity["entity_id"])
        for entity in scene_entities
        if str(entity.get("category") or "") == "uav"
    )
    chargers = sorted(
        (
            entity
            for entity in scene_entities
            if str((entity.get("semantic_scope") or {}).get("scope_subtype"))
            == "charging_station"
        ),
        key=lambda entity: str(entity["entity_id"]),
    )
    if chargers and len(uav_ids) < 2:
        raise RuntimeError(
            f"{scenario_id}: charging service plan requires at least two UAVs"
        )
    facilities: list[dict[str, Any]] = []
    for index, charger in enumerate(chargers):
        charger_id = str(charger["entity_id"])
        explicit_requests = []
        for uav in scene_entities:
            if str(uav.get("category") or "") != "uav":
                continue
            assignment = (uav.get("initial_state") or {}).get(
                "charging_service_assignment"
            )
            if not isinstance(assignment, Mapping):
                continue
            if str(assignment.get("facility_id") or "") != charger_id:
                continue
            if assignment.get("physical_service_required") is not True:
                raise RuntimeError(
                    f"{scenario_id}:{uav['entity_id']} charging assignment must require physical service"
                )
            explicit_requests.append(
                {
                    "uav_id": str(uav["entity_id"]),
                    "arrival_tick": int(assignment["arrival_tick"]),
                    "departure_tick": int(assignment["departure_tick"]),
                }
            )
        if len(explicit_requests) > 1:
            raise RuntimeError(
                f"{scenario_id}:{charger_id} has multiple physical charging assignments"
            )
        if explicit_requests:
            first_request = explicit_requests[0]
            companion_id = next(
                uav_id
                for uav_id in uav_ids
                if uav_id != first_request["uav_id"]
            )
            second_request = {
                "uav_id": companion_id,
                "arrival_tick": min(850, int(first_request["arrival_tick"]) + 30),
                "departure_tick": min(900, int(first_request["departure_tick"]) + 30),
            }
        else:
            first_request = {
                "uav_id": uav_ids[index % len(uav_ids)],
                "arrival_tick": 620,
                "departure_tick": 780,
            }
            second_request = {
                "uav_id": uav_ids[(index + 1) % len(uav_ids)],
                "arrival_tick": 790,
                "departure_tick": 900,
            }
        availability_schedule = [
            {"start_tick": 0, "end_tick": 900, "status": "available"}
        ]
        if scenario_id.startswith("L2-3") and index == 0:
            availability_schedule = [
                {"start_tick": 0, "end_tick": 495, "status": "available"},
                {"start_tick": 500, "end_tick": 560, "status": "unavailable"},
                {"start_tick": 565, "end_tick": 900, "status": "available"},
            ]
        facilities.append(
            {
                "facility_id": charger_id,
                "capacity": int(
                    (charger.get("semantic_scope") or {})["service_capacity"]
                ),
                "availability_schedule": availability_schedule,
                "requests": [first_request, second_request],
            }
        )
    return {
        "schema_name": "charging_service_plan",
        "schema_version": "1.0.0",
        "model_id": "aeroworld_deterministic_charging_service",
        "model_version": "1.0.0",
        "facilities": facilities,
    }


def _runtime_state_context(source_path: Path, profile_id: str, scenario_id: str | None = None) -> str:
    source = str(source_path)
    if scenario_id:
        return f"{source} profile_id={profile_id} scenario_id={scenario_id}"
    return f"{source} profile_id={profile_id}"


def validate_runtime_state_patch_mapping(
    payload: Any,
    *,
    context: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{context}: runtime state patch must be a mapping")
    unknown = sorted(str(key) for key in payload if str(key) not in RUNTIME_STATE_ALLOWED_FAMILIES)
    if unknown:
        raise RuntimeError(f"{context}: unsupported runtime state families: {unknown}")
    forbidden = forbidden_runtime_state_paths(payload, policy="regenerate_split")
    if forbidden:
        raise RuntimeError(f"{context}: runtime state payload contains forbidden semantic fields or paths: {forbidden}")
    unconsumed = unconsumed_runtime_state_paths(payload)
    if unconsumed:
        raise RuntimeError(
            f"{context}: runtime state payload contains fields that no formal state-to-predicate rule consumes: "
            f"{unconsumed}"
        )
    invalid_values = invalid_runtime_state_value_paths(payload)
    if invalid_values:
        raise RuntimeError(
            f"{context}: runtime state payload contains values outside the governed field contract: "
            f"{invalid_values}"
        )
    patch: dict[str, dict[str, Any]] = {}
    for family, family_value in payload.items():
        family_name = str(family)
        if not isinstance(family_value, Mapping):
            raise RuntimeError(f"{context}: {family_name} must be a mapping")
        if not family_value:
            raise RuntimeError(f"{context}: {family_name} must not be an empty family patch")
        patch[family_name] = copy.deepcopy(dict(family_value))
    if not patch:
        raise RuntimeError(f"{context}: runtime state patch must include at least one allowed state family")
    return patch


def _deep_merge_mapping(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        key_text = str(key)
        if isinstance(value, Mapping) and isinstance(base.get(key_text), dict):
            _deep_merge_mapping(base[key_text], value)
        else:
            base[key_text] = copy.deepcopy(value)
    return base


def _boolean_leaf_paths(
    value: Any,
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            paths.update(_boolean_leaf_paths(item, (*prefix, str(key))))
    elif isinstance(value, bool):
        paths.add(prefix)
    return paths


def _require_boolean_transition_baselines(
    *,
    initial_entity_states: Mapping[str, Mapping[str, Any]],
    event_state_transitions: Mapping[str, Sequence[Mapping[str, Any]]],
    context: str,
) -> None:
    baseline_paths = {
        str(entity_id): _boolean_leaf_paths(state_mapping)
        for entity_id, state_mapping in initial_entity_states.items()
    }
    missing: list[str] = []
    for event_id, transitions in event_state_transitions.items():
        for transition_index, transition in enumerate(transitions):
            entity_id = str(transition.get("entity_id") or "")
            for path in sorted(_boolean_leaf_paths(transition.get("state_patch") or {})):
                if path not in baseline_paths.get(entity_id, set()):
                    missing.append(
                        f"{event_id}[{transition_index}] entity={entity_id} path={'.'.join(path)}"
                    )
    if missing:
        raise RuntimeError(
            f"{context}: every boolean runtime-state transition requires an explicit initial baseline; "
            f"missing {len(missing)}: {missing}"
        )


def _jsonschema_error_path(error: Any) -> str:
    parts = ["$"]
    for item in error.absolute_path:
        if isinstance(item, int):
            parts[-1] = f"{parts[-1]}[{item}]"
        else:
            parts.append(str(item))
    return ".".join(parts)


def _validate_runtime_state_schedule_profile_schema(path: Path, payload: Mapping[str, Any]) -> None:
    if not RUNTIME_STATE_SCHEDULE_SCHEMA_PATH.exists():
        raise RuntimeError(f"Missing runtime-state schedule schema: {RUNTIME_STATE_SCHEDULE_SCHEMA_PATH}")
    try:
        schema = json.loads(RUNTIME_STATE_SCHEDULE_SCHEMA_PATH.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{RUNTIME_STATE_SCHEDULE_SCHEMA_PATH}: invalid JSON schema: {exc}") from exc
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=_jsonschema_error_path)
    if errors:
        details = "; ".join(f"{_jsonschema_error_path(error)}: {error.message}" for error in errors[:5])
        raise RuntimeError(f"{path}: runtime-state schedule schema validation failed: {details}")


def _load_runtime_state_schedule_profile(
    path: Path,
    *,
    scenario_filter: set[str] | None = None,
) -> list[RuntimeStateSchedule]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path}: invalid runtime-state schedule JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{path}: runtime-state schedule profile must be a mapping")
    _validate_runtime_state_schedule_profile_schema(path, payload)
    if payload.get("schema_name") != RUNTIME_STATE_SCHEDULE_SCHEMA_NAME:
        raise RuntimeError(
            f"{path}: schema_name must be {RUNTIME_STATE_SCHEDULE_SCHEMA_NAME!r}, "
            f"got {payload.get('schema_name')!r}"
        )
    schema_version = str(payload.get("schema_version") or "")
    profile_id = str(payload.get("profile_id") or "")
    if not schema_version:
        raise RuntimeError(f"{path}: schema_version is required")
    if not profile_id:
        raise RuntimeError(f"{path}: profile_id is required")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise RuntimeError(f"{path}: scenarios must be a list")

    schedules: list[RuntimeStateSchedule] = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            raise RuntimeError(f"{path}: scenarios[{index}] must be a mapping")
        scenario_id = str(scenario.get("scenario_id") or "")
        if not scenario_id:
            raise RuntimeError(f"{path}: scenarios[{index}].scenario_id is required")
        if scenario_filter is not None and scenario_id not in scenario_filter:
            continue
        context = _runtime_state_context(path, profile_id, scenario_id)
        initial_raw = scenario.get("initial_entity_states")
        transitions_raw = scenario.get("event_state_transitions")
        if not isinstance(initial_raw, Mapping):
            raise RuntimeError(f"{context}: initial_entity_states must be a mapping")
        if not isinstance(transitions_raw, Mapping):
            raise RuntimeError(f"{context}: event_state_transitions must be a mapping")

        initial_entity_states: dict[str, dict[str, dict[str, Any]]] = {}
        for entity_id_raw, state_mapping in initial_raw.items():
            entity_id = str(entity_id_raw)
            if not entity_id:
                raise RuntimeError(f"{context}: initial_entity_states contains an empty entity_id")
            initial_entity_states[entity_id] = validate_runtime_state_patch_mapping(
                state_mapping,
                context=f"{context} initial_entity_states[{entity_id}]",
            )

        event_state_transitions: dict[str, list[dict[str, Any]]] = {}
        for event_id_raw, transitions in transitions_raw.items():
            event_id = str(event_id_raw)
            if not event_id:
                raise RuntimeError(f"{context}: event_state_transitions contains an empty event_id")
            if not isinstance(transitions, list):
                raise RuntimeError(f"{context}: event_state_transitions[{event_id}] must be a list")
            clean_transitions: list[dict[str, Any]] = []
            for transition_index, transition in enumerate(transitions):
                transition_context = f"{context} event_state_transitions[{event_id}][{transition_index}]"
                if not isinstance(transition, Mapping):
                    raise RuntimeError(f"{transition_context}: transition must be a mapping")
                entity_id = str(transition.get("entity_id") or "")
                if not entity_id:
                    raise RuntimeError(f"{transition_context}: entity_id is required")
                action_id = transition.get("action_id")
                if action_id is not None and not str(action_id):
                    raise RuntimeError(f"{transition_context}: action_id must be non-empty when provided")
                delay_ticks_raw = transition.get("delay_ticks", 0)
                if isinstance(delay_ticks_raw, bool) or not isinstance(delay_ticks_raw, int) or delay_ticks_raw < 0:
                    raise RuntimeError(f"{transition_context}: delay_ticks must be a non-negative integer")
                clean_transitions.append(
                    {
                        "entity_id": entity_id,
                        "action_id": str(action_id) if action_id is not None else "",
                        "delay_ticks": delay_ticks_raw,
                        "state_patch": validate_runtime_state_patch_mapping(
                            transition.get("state_patch"),
                            context=f"{transition_context}.state_patch",
                        ),
                    }
                )
            event_state_transitions[event_id] = clean_transitions

        _require_boolean_transition_baselines(
            initial_entity_states=initial_entity_states,
            event_state_transitions=event_state_transitions,
            context=context,
        )

        schedules.append(
            RuntimeStateSchedule(
                profile_id=profile_id,
                source_path=path,
                scenario_id=scenario_id,
                initial_entity_states=initial_entity_states,
                event_state_transitions=event_state_transitions,
            )
        )
    return schedules


def load_runtime_state_schedule_profiles(
    profile_dir: Path = RUNTIME_STATE_SCHEDULE_PROFILE_DIR,
    *,
    scenario_filter: set[str] | None = None,
    require_profiles: bool = False,
) -> dict[str, RuntimeStateSchedule]:
    profile_dir = Path(profile_dir)
    if not profile_dir.exists():
        if require_profiles:
            raise RuntimeError(f"{profile_dir}: runtime-state schedule profile directory is required")
        return {}
    if not profile_dir.is_dir():
        raise RuntimeError(f"{profile_dir}: runtime-state schedule profile path is not a directory")
    profile_paths = sorted(profile_dir.glob("*.json"))
    if require_profiles and not profile_paths:
        raise RuntimeError(f"{profile_dir}: runtime-state schedule profile directory contains no JSON profiles")
    schedules: dict[str, RuntimeStateSchedule] = {}
    for path in profile_paths:
        for schedule in _load_runtime_state_schedule_profile(path, scenario_filter=scenario_filter):
            existing = schedules.get(schedule.scenario_id)
            if existing is not None:
                raise RuntimeError(
                    f"{schedule.scenario_id}: duplicate runtime-state scenario in "
                    f"{existing.source_path} and {schedule.source_path}"
                )
            schedules[schedule.scenario_id] = schedule
    return schedules


def _action_id_from_event_action(event_action: dict[str, Any]) -> str:
    params = event_action.get("params")
    if isinstance(params, Mapping) and params.get("action_id"):
        return str(params.get("action_id"))
    if event_action.get("action_id"):
        return str(event_action.get("action_id"))
    return ""


def apply_runtime_state_schedules(
    bundles: list[ScenarioBundle],
    schedules: Mapping[str, RuntimeStateSchedule],
) -> None:
    if not schedules:
        return
    bundles_by_id = {bundle.scenario_id: bundle for bundle in bundles}
    unused_scenarios = sorted(set(schedules) - set(bundles_by_id))
    if unused_scenarios:
        raise RuntimeError(f"Unused runtime-state schedule scenarios: {unused_scenarios}")

    for scenario_id, schedule in sorted(schedules.items()):
        bundle = bundles_by_id[scenario_id]
        context = _runtime_state_context(schedule.source_path, schedule.profile_id, scenario_id)
        scene_by_id = {str(scene.get("entity_id") or ""): scene for scene in bundle.scene_entities}
        event_by_id = {str(event_def.get("event_id") or ""): event_def for event_def in bundle.events}
        used_action_ids: set[str] = set()
        for event_def in bundle.events:
            for event_action in event_def.get("actions") or []:
                action_id = _action_id_from_event_action(event_action)
                if action_id:
                    used_action_ids.add(action_id)

        for entity_id, state_mapping in schedule.initial_entity_states.items():
            scene = scene_by_id.get(entity_id)
            if scene is None:
                raise RuntimeError(f"{context}: unknown entity in initial_entity_states: {entity_id}")
            for family, family_patch in state_mapping.items():
                current = scene.get(family)
                if not isinstance(current, dict):
                    current = {}
                scene[family] = _deep_merge_mapping(current, family_patch)

        for event_id, transitions in schedule.event_state_transitions.items():
            event_def = event_by_id.get(event_id)
            if event_def is None:
                raise RuntimeError(f"{context}: unknown event in event_state_transitions: {event_id}")
            event_actions = event_def.setdefault("actions", [])
            for transition_index, transition in enumerate(transitions):
                entity_id = str(transition["entity_id"])
                if entity_id not in scene_by_id:
                    raise RuntimeError(
                        f"{context}: unknown entity in event_state_transitions[{event_id}][{transition_index}]: "
                        f"{entity_id}"
                    )
                action_id = str(transition.get("action_id") or "")
                if not action_id:
                    action_id = (
                        f"set_runtime_state_{safe_id(scenario_id)}_"
                        f"{safe_id(event_id)}_{transition_index:02d}"
                    )
                if action_id in used_action_ids:
                    raise RuntimeError(f"{context}: duplicate action_id for runtime-state transition: {action_id}")
                used_action_ids.add(action_id)
                event_actions.append(
                    action(
                        action_id,
                        "set_runtime_state",
                        entity_id=entity_id,
                        delay_ticks=int(transition.get("delay_ticks") or 0),
                        state_patch=copy.deepcopy(transition["state_patch"]),
                        runtime_state_schedule_profile_id=schedule.profile_id,
                        runtime_state_schedule_semantics="queued_after_event_actions_delay_ticks_0_next_tick",
                    )
                )


def require_runtime_state_schedule_coverage(
    bundles: Sequence[ScenarioBundle],
    schedules: Mapping[str, RuntimeStateSchedule],
) -> None:
    expected_schedule_ids = {bundle.scenario_id for bundle in bundles}
    missing_schedule_ids = sorted(expected_schedule_ids - set(schedules))
    if missing_schedule_ids:
        raise RuntimeError(
            "Formal regeneration requires one runtime-state schedule for every selected scenario; "
            f"missing {len(missing_schedule_ids)}: {missing_schedule_ids}"
        )


def require_l25_signal_repair_schedule(
    bundles: Sequence[ScenarioBundle],
    schedules: Mapping[str, RuntimeStateSchedule],
) -> None:
    target = next((bundle for bundle in bundles if bundle.scenario_id == "L2-5_v1"), None)
    if target is None:
        return
    events = target.events
    event_ids = [str(event_def.get("event_id") or "") for event_def in events]
    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError("L2-5_v1 contains duplicate event IDs")
    event_by_id = {event_def["event_id"]: event_def for event_def in events}
    for event_id, event_def in event_by_id.items():
        if event_def.get("on_fire_emit"):
            raise RuntimeError(f"L2-5_v1 {event_id} emits an extra event outside the declared chain")
    required_triggers = {
        "signal_all_red_fault": tick(210),
        "traffic_confusion": fired("signal_all_red_fault"),
        "police_arrival": fired("traffic_confusion", delay_ticks=96),
        "manual_flow_restore": fired("police_arrival"),
        "signal_infrastructure_repair": fired("manual_flow_restore", delay_ticks=60),
    }
    for event_id, trigger in required_triggers.items():
        event_def = event_by_id.get(event_id)
        if event_def is None or event_def.get("trigger") != trigger:
            raise RuntimeError(f"L2-5_v1 {event_id} trigger differs from the event chain")
        if (event_def.get("max_fire_count", 1) != 1
                or event_def.get("cooldown_ticks", 0) != 0
                or event_def.get("require_conditions")
                or event_def.get("on_fire_emit")):
            raise RuntimeError(f"L2-5_v1 {event_id} firing conditions differ from the event chain")
    signal_id = "traffic_signal_l2_5_v1"
    signal_actions = [
        (event_id, event_action)
        for event_id, event_def in event_by_id.items()
        for event_action in event_def.get("actions") or []
        if (event_action.get("params") or {}).get("entity_id") == signal_id
        and event_action.get("type") in {"set_visual_state", "set_runtime_state"}
    ]
    if (len(signal_actions) != 2
            or {(event_id, action_def.get("type"),
                 (action_def.get("params") or {}).get("visual_state", {}).get("mode"))
                for event_id, action_def in signal_actions}
            != {("signal_all_red_fault", "set_visual_state", "all_red_fault"),
                ("signal_infrastructure_repair", "set_visual_state", "nominal")}):
        raise RuntimeError("L2-5_v1 signal visual actions differ from the event chain")
    manual_event = event_by_id["manual_flow_restore"]
    manual_actions = manual_event.get("actions") or []
    manual_action_ids = {
        (action_def.get("type"), (action_def.get("params") or {}).get("action_id"),
         (action_def.get("params") or {}).get("entity_id"))
        for action_def in manual_actions
    }
    if len(manual_actions) != 3 or manual_action_ids != {
        ("set_visual_state", "set_police_manual_control", "police_unit_l2_5_v1"),
        ("move_entity", "move_car_a_released", "civilian_car_l2_5_a"),
        ("move_entity", "move_car_b_released", "civilian_car_l2_5_b"),
    }:
        raise RuntimeError("L2-5_v1 manual-flow actions differ from the event chain")
    manual_visual = next(action_def for action_def in manual_actions
                         if action_def.get("type") == "set_visual_state")
    if (manual_visual.get("params") or {}).get("visual_state") != {
        "mode": "manual_directing", "lights_on": True,
    }:
        raise RuntimeError("L2-5_v1 police manual visual state differs from the event chain")
    for event_id, event_def in event_by_id.items():
        if event_id == "manual_flow_restore":
            continue
        for action_def in event_def.get("actions") or []:
            params = action_def.get("params") or {}
            if (params.get("entity_id") == "police_unit_l2_5_v1"
                    and action_def.get("type") == "set_visual_state"
                    and (params.get("visual_state") or {}).get("mode") == "manual_directing"):
                raise RuntimeError(f"L2-5_v1 {event_id} starts manual control before restore")
    schedule = schedules.get("L2-5_v1")
    if schedule is None:
        raise RuntimeError("L2-5_v1 requires its signal fault and independent repair schedule")
    initial = schedule.initial_entity_states
    if (initial.get("traffic_signal_l2_5_v1", {}).get("facility_state", {}).get("fault") is not False
            or initial.get("police_unit_l2_5_v1", {}).get("incident_state", {}).get("manual_control_active") is not False):
        raise RuntimeError("L2-5_v1 initial signal and manual-control states differ from the event chain")
    expected = {
        "signal_all_red_fault": ("traffic_signal_l2_5_v1", {"facility_state": {"fault": True}}),
        "manual_flow_restore": ("police_unit_l2_5_v1", {"incident_state": {"manual_control_active": True}}),
        "signal_infrastructure_repair": ("traffic_signal_l2_5_v1", {"facility_state": {"fault": False}}),
    }
    for event_id, (entity_id, state_patch) in expected.items():
        transitions = schedule.event_state_transitions.get(event_id)
        if (not isinstance(transitions, list) or len(transitions) != 1
                or transitions[0].get("entity_id") != entity_id
                or transitions[0].get("delay_ticks") != 0
                or transitions[0].get("state_patch") != state_patch):
            raise RuntimeError(f"L2-5_v1 {event_id} runtime state differs from the event chain")
    for event_id, transitions in schedule.event_state_transitions.items():
        for transition in transitions:
            if (transition.get("entity_id") == signal_id
                    and "fault" in transition.get("state_patch", {}).get("facility_state", {})
                    and event_id not in {"signal_all_red_fault", "signal_infrastructure_repair"}):
                raise RuntimeError(f"L2-5_v1 {event_id} changes signal fault outside the repair chain")
            if (transition.get("entity_id") == "police_unit_l2_5_v1"
                    and "manual_control_active" in transition.get("state_patch", {}).get("incident_state", {})
                    and event_id != "manual_flow_restore"):
                raise RuntimeError(f"L2-5_v1 {event_id} changes manual control before restore")


def require_l25_retained_scene_layout(bundles: Sequence[ScenarioBundle]) -> None:
    target = next((bundle for bundle in bundles if bundle.scenario_id == "L2-5_v1"), None)
    if target is None:
        return
    retained_path = target.directory / "scene_setup.json"
    if not retained_path.exists():
        raise RuntimeError("L2-5_v1 retained scene_setup.json is required before regeneration")
    retained = json.loads(retained_path.read_text(encoding="utf-8"))
    if scene_setup_from_bundle(target) != retained:
        raise RuntimeError(
            "L2-5_v1 generated scene layout differs from the retained formal layout; "
            "resolve the spatial authority before replacing scene_setup.json"
        )


def require_l25_compiled_state_actions(compiled: Mapping[str, Any], schedule: RuntimeStateSchedule) -> None:
    """Check resolved actions, including parameter references, before replacing the script."""
    if compiled.get("scenario_id") != "L2-5_v1":
        return
    def unresolved_param(value: Any) -> bool:
        if isinstance(value, str):
            return value.startswith("$param.")
        if isinstance(value, Mapping):
            return any(unresolved_param(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(unresolved_param(item) for item in value)
        return False

    if unresolved_param(compiled["events"]):
        raise RuntimeError("L2-5_v1 compiled event chain has an unresolved parameter")
    expected_runtime = {
        "signal_all_red_fault": ("traffic_signal_l2_5_v1", {"facility_state": {"fault": True}}),
        "manual_flow_restore": ("police_unit_l2_5_v1", {"incident_state": {"manual_control_active": True}}),
        "signal_infrastructure_repair": ("traffic_signal_l2_5_v1", {"facility_state": {"fault": False}}),
    }
    observed_runtime: list[tuple[str, dict[str, Any]]] = []
    observed_signal_visual: list[tuple[str, dict[str, Any]]] = []
    observed_manual_visual: list[tuple[str, dict[str, Any]]] = []
    observed_target_actions: list[tuple[str, str, str, str]] = []
    for event_def in compiled["events"]:
        event_id = str(event_def["event_id"])
        for event_action in event_def.get("actions") or []:
            entity_id = event_action.get("entity_id")
            action_type = event_action.get("type")
            if entity_id in {"traffic_signal_l2_5_v1", "police_unit_l2_5_v1"}:
                observed_target_actions.append((event_id, str(action_type), str(event_action.get("action_id")), entity_id))
            if action_type == "set_runtime_state" and entity_id in {
                "traffic_signal_l2_5_v1", "police_unit_l2_5_v1",
            }:
                observed_runtime.append((event_id, event_action))
            if action_type == "set_visual_state" and entity_id == "traffic_signal_l2_5_v1":
                observed_signal_visual.append((event_id, event_action))
            if action_type == "set_visual_state" and entity_id == "police_unit_l2_5_v1":
                observed_manual_visual.append((event_id, event_action))
    if observed_target_actions != [
        ("signal_all_red_fault", "set_visual_state", "set_signal_all_red", "traffic_signal_l2_5_v1"),
        ("signal_all_red_fault", "set_runtime_state", "set_runtime_state_l2_5_v1_signal_all_red_fault_00", "traffic_signal_l2_5_v1"),
        ("police_arrival", "spawn_entity", "spawn_police_unit_l2_5_v1_at_dispatch", "police_unit_l2_5_v1"),
        ("police_arrival", "move_entity", "move_police_to_intersection", "police_unit_l2_5_v1"),
        ("manual_flow_restore", "set_visual_state", "set_police_manual_control", "police_unit_l2_5_v1"),
        ("manual_flow_restore", "set_runtime_state", "set_runtime_state_l2_5_v1_manual_flow_restore_00", "police_unit_l2_5_v1"),
        ("signal_infrastructure_repair", "set_visual_state", "set_signal_nominal_after_repair", "traffic_signal_l2_5_v1"),
        ("signal_infrastructure_repair", "set_runtime_state", "set_runtime_state_l2_5_v1_signal_infrastructure_repair_00", "traffic_signal_l2_5_v1"),
        ("signal_response_actors_leave", "remove_entity", "remove_police_after_restore", "police_unit_l2_5_v1"),
    ]:
        raise RuntimeError("L2-5_v1 signal/police actions differ from the event chain")
    required_runtime = [
        (
            event_id,
            {
                "type": "set_runtime_state",
                "action_id": f"set_runtime_state_l2_5_v1_{event_id}_00",
                "entity_id": entity_id,
                "delay_ticks": 0,
                "state_patch": state_patch,
                "runtime_state_schedule_profile_id": schedule.profile_id,
                "runtime_state_schedule_semantics": "queued_after_event_actions_delay_ticks_0_next_tick",
            },
        )
        for event_id, (entity_id, state_patch) in expected_runtime.items()
    ]
    if observed_runtime != required_runtime:
        raise RuntimeError("L2-5_v1 resolved signal/manual runtime actions differ from the event chain")
    if observed_signal_visual != [
        ("signal_all_red_fault", {
            "type": "set_visual_state", "action_id": "set_signal_all_red",
            "entity_id": "traffic_signal_l2_5_v1", "visual_state": {"mode": "all_red_fault"},
        }),
        ("signal_infrastructure_repair", {
            "type": "set_visual_state", "action_id": "set_signal_nominal_after_repair",
            "entity_id": "traffic_signal_l2_5_v1", "visual_state": {"mode": "nominal"},
        }),
    ]:
        raise RuntimeError("L2-5_v1 resolved signal visual actions differ from the event chain")
    if observed_manual_visual != [
        ("manual_flow_restore", {
            "type": "set_visual_state", "action_id": "set_police_manual_control",
            "entity_id": "police_unit_l2_5_v1",
            "visual_state": {"mode": "manual_directing", "lights_on": True},
        }),
    ]:
        raise RuntimeError("L2-5_v1 resolved police manual visual action differs from the event chain")
    police_move = next(
        action_def for event_def in compiled["events"]
        if event_def["event_id"] == "police_arrival"
        for action_def in event_def.get("actions") or []
        if action_def.get("action_id") == "move_police_to_intersection"
    )
    duration = quantized_path_duration_ticks(
        police_move["waypoints_enu_m"], float(police_move["velocity_mps"])
    )
    if duration > 26:
        raise RuntimeError("L2-5_v1 police arrival route extends into manual control")


def require_l25_retained_motion_actions(bundle: ScenarioBundle, compiled: Mapping[str, Any]) -> None:
    if bundle.scenario_id != "L2-5_v1":
        return
    retained_path = bundle.directory / "event_script.json"
    if not retained_path.exists():
        raise RuntimeError("L2-5_v1 retained event_script.json is required before regeneration")
    retained = json.loads(retained_path.read_text(encoding="utf-8"))

    def firing_controls(script: Mapping[str, Any]) -> tuple[Any, list[tuple[Any, ...]]]:
        return script["triggers"], [
            (
                event_def["event_id"], event_def["trigger_ref"],
                event_def.get("max_fire_count", 1), event_def.get("cooldown_ticks", 0),
                event_def.get("require_conditions", []), event_def.get("on_fire"),
                event_def.get("priority", 10),
            )
            for event_def in script["events"]
        ]

    if firing_controls(compiled) != firing_controls(retained):
        raise RuntimeError(
            "L2-5_v1 generated event timing differs from the retained formal script; "
            "resolve the event timing authority before replacing event_script.json"
        )

    def motion_actions(script: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        return [
            (str(event_def["event_id"]), action_def)
            for event_def in script["events"]
            for action_def in event_def.get("actions") or []
            if action_def.get("type") in {"spawn_entity", "move_entity", "remove_entity"}
        ]

    if motion_actions(compiled) != motion_actions(retained):
        raise RuntimeError(
            "L2-5_v1 generated movement actions differ from the retained formal script; "
            "resolve the trajectory authority before replacing event_script.json"
        )


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
    direct_write_body = (
        '    raise RuntimeError("L2-5_v1 direct spec writing bypasses the retained scene and event guards")'
        if bundle.scenario_id == "L2-5_v1" else
        '''    here = Path(__file__).resolve().parent
    spec = build_spec()
    compiled = SpecCompiler().compile(spec)
    with open(here / "event_script.json", "w", encoding="utf-8") as f:
        json.dump(compiled, f, indent=2, ensure_ascii=False)
        f.write("\\n")
    with open(here / "scene_setup.json", "w", encoding="utf-8") as f:
        json.dump(SCENE_SETUP, f, indent=2, ensure_ascii=False)
        f.write("\\n")'''
    )
    spec_note = (
        "Use the guarded scenario generator to replace event_script.json and scene_setup.json."
        if bundle.scenario_id == "L2-5_v1" else
        "This file is intentionally self-contained: running it recompiles\n"
        "event_script.json from the ScenarioSpec below."
    )
    writer_imports = "" if bundle.scenario_id == "L2-5_v1" else "import json\n"
    compiler_import = "" if bundle.scenario_id == "L2-5_v1" else "    SpecCompiler,\n"
    return f'''"""Concrete ScenarioSpec for {bundle.scenario_id}.

Generated from Dataset/tools/regenerate_boundary_scenarios.py.
{spec_note}
"""

from __future__ import annotations

{writer_imports}import sys
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
{compiler_import}    TriggerSpec,
    WaypointSpec,
)


SCENE_SETUP = {py_repr(scene_setup)}


SPEC_DATA = {py_repr(spec_data)}


def _trigger(data):
    allowed = set(TriggerSpec.__dataclass_fields__)
    cleaned = {{key: value for key, value in data.items() if key in allowed}}
    return TriggerSpec(**cleaned)


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
        causal_predecessor_event_ids=data.get("causal_predecessor_event_ids", []),
        causal_predecessor_mode=data.get("causal_predecessor_mode", ""),
        causal_temporal_policy=data.get("causal_temporal_policy", ""),
        target_roles=data.get("target_roles", []),
        validation_event_type=data.get("validation_event_type", ""),
        validation_reason=data.get("validation_reason", ""),
        validation_skip_checks=data.get("validation_skip_checks", []),
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
{direct_write_body}
'''


def _bundle_capture_center(bundle: ScenarioBundle) -> list[float]:
    boundary = dict(bundle.parameters.get("capture_boundary") or {})
    center = boundary.get("center_enu_m")
    if isinstance(center, list) and len(center) >= 2:
        return [float(center[0]), float(center[1]), float(center[2] if len(center) > 2 else GROUND_Z_M)]
    return _capture_boundary_focus_center(bundle.scene_entities, bundle.events)


def _bundle_assignment_center(bundle: ScenarioBundle) -> list[float]:
    return _capture_boundary_assignment_center(bundle.scene_entities, bundle.events)


def build_spatially_assigned_bundle(
    scenario_id: str,
    builder: Callable[[], ScenarioBundle],
    assignment: SpatialAssignment,
    provisional_center: list[float] | None = None,
    enforce_grid_target_error: bool = True,
) -> ScenarioBundle:
    if provisional_center is None:
        reset_world_event_origin()
        global SPATIAL_PROVISIONAL_BUILD
        previous_provisional_build = SPATIAL_PROVISIONAL_BUILD
        SPATIAL_PROVISIONAL_BUILD = True
        try:
            provisional = builder()
        finally:
            SPATIAL_PROVISIONAL_BUILD = previous_provisional_build
        provisional_center = _bundle_assignment_center(provisional)
    target = assignment.target_center_enu_m
    target_origin_x = WORLD_OFFSET_X_M + float(target[0]) - float(provisional_center[0])
    target_origin_y = WORLD_OFFSET_Y_M + float(target[1]) - float(provisional_center[1])
    set_world_event_origin(target_origin_x, target_origin_y)
    try:
        bundle = builder()
    finally:
        reset_world_event_origin()
    actual_center = _bundle_capture_center(bundle)
    actual_assignment_center = _bundle_assignment_center(bundle)
    target_error_m = dist_xy(actual_assignment_center, target)
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
                actual_assignment_center[0],
                actual_assignment_center[1],
                actual_assignment_center[2] if len(actual_assignment_center) > 2 else GROUND_Z_M,
            ),
            "expanded_capture_center_enu_m": q(
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
    if (
        enforce_grid_target_error
        and assignment.event_space_class == "traffic_incident_grid"
        and target_error_m > 80.0
    ):
        raise RuntimeError(
            f"{scenario_id}: traffic incident capture center is {target_error_m:.1f}m from SUMO anchor"
        )
    if (
        enforce_grid_target_error
        and assignment.event_space_class == "nontraffic_main_road_grid"
        and target_error_m > 140.0
    ):
        raise RuntimeError(
            f"{scenario_id}: nontraffic capture center is {target_error_m:.1f}m from assigned main-road grid"
        )
    return bundle


def _assignment_outside_roi_m(
    assignment_center: Sequence[float],
    roi_contract: RoiContract,
) -> float:
    roi = Polygon(roi_contract.capture_polygon_enu_m)
    point = Point(float(assignment_center[0]), float(assignment_center[1]))
    if roi.covers(point):
        return 0.0
    return float(roi.distance(point))


def build_roi_bound_bundle(
    scenario_id: str,
    builder: Callable[[], ScenarioBundle],
    primary_assignment: SpatialAssignment,
    spatial_grid: SpatialEventGridPlanner,
    roi_contract: RoiContract,
    provisional: ScenarioBundle,
) -> ScenarioBundle:
    provisional_assignment_center = _bundle_assignment_center(provisional)
    target_assignment_center = list(roi_contract.center_enu_m)
    attempts: list[dict[str, Any]] = []
    for attempt in range(3):
        assignment = SpatialAssignment(
            scenario_id=scenario_id,
            event_space_class=primary_assignment.event_space_class,
            target_center_enu_m=list(target_assignment_center),
            grid_cell=spatial_grid._cell_for_point(target_assignment_center),
            source=roi_contract.geometry_authority,
            traffic_incident_id=primary_assignment.traffic_incident_id,
            accident_class=primary_assignment.accident_class,
            exclusion_radius_m=primary_assignment.exclusion_radius_m,
        )
        bundle = build_spatially_assigned_bundle(
            scenario_id,
            builder,
            assignment,
            provisional_center=provisional_assignment_center,
            enforce_grid_target_error=False,
        )
        actual_assignment_center = _bundle_assignment_center(bundle)
        outside_m = _assignment_outside_roi_m(actual_assignment_center, roi_contract)
        boundary = dict(bundle.parameters.get("capture_boundary") or {})
        polygon = [
            [float(point[0]), float(point[1])]
            for point in boundary.get("polygon_enu_m") or []
        ]
        attempts.append(
            {
                "attempt": attempt + 1,
                "target_assignment_center_enu_m": q(*target_assignment_center),
                "actual_assignment_center_enu_m": q(*actual_assignment_center),
                "roi_contract_center_enu_m": list(roi_contract.center_enu_m),
                "assignment_outside_roi_m": round(outside_m, 6),
                "polygon_matches_roi_contract": polygon == roi_contract.polygon_lists(),
            }
        )
        if (
            polygon == roi_contract.polygon_lists()
            and outside_m <= ROI_ASSIGNMENT_MAX_OUTSIDE_M
        ):
            authority = {
                "policy": roi_contract.geometry_authority,
                "source_root": str(
                    roi_contract.authority_root.resolve().relative_to(ROOT.resolve())
                ).replace("\\", "/"),
                "roi_contract_center_enu_m": list(roi_contract.center_enu_m),
                "actual_assignment_center_enu_m": q(*actual_assignment_center),
                "assignment_outside_roi_m": round(outside_m, 6),
                "attempts": attempts,
            }
            bundle.parameters["roi_contract"] = roi_contract.contract_payload(
                authority_root=ROOT
            )
            bundle.parameters["release_spatial_authority"] = authority
            bundle.validation_rules = list(bundle.validation_rules or [])
            bundle.validation_rules.append(
                {
                    "rule": "release_spatial_authority",
                    **authority,
                    "description": "Targeted truth regeneration uses the explicit ROI contract and does not reassign a spatial grid cell",
                }
            )
            return bundle
        target_assignment_center[0] += (
            float(roi_contract.center_enu_m[0]) - float(actual_assignment_center[0])
        )
        target_assignment_center[1] += (
            float(roi_contract.center_enu_m[1]) - float(actual_assignment_center[1])
        )
    raise RuntimeError(
        f"{scenario_id}: cannot bind generated events to ROI contract "
        f"{roi_contract.center_enu_m}; attempts={attempts}"
    )


def _authoritative_assignment_payload(scenario_id: str) -> dict[str, Any] | None:
    """Return the spatial grid assignment already frozen for this scenario, if any.

    Where a scenario sits on the city grid is an authored location decision that was
    fixed when its bundle was first compiled, not something a later partial run may
    re-derive.  Re-reading it keeps a single-scenario regeneration anchored to the
    same world origin instead of letting the global cell cursor shift underneath it.
    """
    for script_path in sorted(SCENARIOS_ROOT.rglob("event_script.json")):
        try:
            script = json.loads(script_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        existing_id = str(script.get("scenario_id") or script_path.parent.name)
        if existing_id != scenario_id and script_path.parent.name != scenario_id:
            continue
        payload = dict((script.get("parameters") or {}).get("spatial_grid_assignment") or {})
        if (payload.get("grid_cell") or {}).get("cell_id"):
            return payload
        return None
    return None


def _grid_cell_from_payload(payload: Mapping[str, Any]) -> SpatialGridCell | None:
    cell = dict(payload.get("grid_cell") or {})
    cell_id = str(cell.get("cell_id") or "")
    representative = cell.get("representative_enu_m")
    if not cell_id or not isinstance(representative, list) or len(representative) < 2:
        return None
    center = cell.get("center_enu_m")
    if not isinstance(center, list) or len(center) < 2:
        center = list(representative)
    return SpatialGridCell(
        cell_id=cell_id,
        center_enu_m=[
            float(center[0]),
            float(center[1]),
            float(center[2]) if len(center) > 2 else 0.0,
        ],
        representative_enu_m=[
            float(representative[0]),
            float(representative[1]),
            float(representative[2]) if len(representative) > 2 else 0.0,
        ],
        edge_ids=tuple(str(item) for item in cell.get("edge_ids") or []),
        main_edge_count=int(cell.get("main_edge_count") or 0),
        total_main_edge_length_m=float(cell.get("total_main_edge_length_m") or 0.0),
        max_speed_mps=float(cell.get("max_speed_mps") or 0.0),
        incident_count=int(cell.get("incident_count") or 0),
    )


def _frozen_assignment_for(scenario_id: str) -> SpatialAssignment | None:
    payload = _authoritative_assignment_payload(scenario_id)
    if payload is None:
        return None
    cell = _grid_cell_from_payload(payload)
    if cell is None:
        return None
    target = payload.get("target_center_enu_m")
    if not isinstance(target, list) or len(target) < 2:
        return None
    return SpatialAssignment(
        scenario_id=scenario_id,
        event_space_class=str(payload.get("event_space_class") or ""),
        target_center_enu_m=[float(target[0]), float(target[1]), float(target[2]) if len(target) > 2 else 0.0],
        grid_cell=cell,
        source=str(payload.get("source") or ""),
        traffic_incident_id=str(payload.get("traffic_incident_id") or ""),
        accident_class=str(payload.get("accident_class") or ""),
        exclusion_radius_m=float(payload.get("exclusion_radius_m") or 0.0),
    )


def collect_bundles(
    selected_scenarios: set[str] | None = None,
    *,
    spatial_authority_root: Path | None = None,
) -> list[ScenarioBundle]:
    selected = set(selected_scenarios or set())
    global PARTIAL_REGENERATION
    PARTIAL_REGENERATION = bool(selected)

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
    roi_contracts: dict[str, RoiContract] = {}
    if spatial_authority_root is not None:
        resolved_authority_root = spatial_authority_root.resolve()
        resolved_authority_root.relative_to(ROOT.resolve())
        roi_contracts = load_roi_contracts(
            resolved_authority_root,
            [scenario_id for scenario_id in all_scenario_ids if include(scenario_id)],
        )
    spatial_grid = SpatialEventGridPlanner(planner=SUMO_GROUND_FLOW)
    spatial_assignments = spatial_grid.assign_scenarios(all_scenario_ids)
    nonincident_cells = spatial_grid.nontraffic_cells_for_assignments(spatial_assignments)
    used_nontraffic_cell_ids: set[str] = set()
    used_traffic_cell_ids: set[str] = set()
    if selected:
        for scenario_id in all_scenario_ids:
            if include(scenario_id):
                continue
            frozen = _frozen_assignment_for(scenario_id)
            if frozen is None:
                continue
            cell_id = frozen.grid_cell.cell_id
            if frozen.event_space_class == "nontraffic_main_road_grid":
                used_nontraffic_cell_ids.add(cell_id)
            elif frozen.event_space_class == "traffic_incident_grid":
                used_traffic_cell_ids.add(cell_id)
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

    def traffic_assignment_for_cell(scenario_id: str, cell: Any, primary: SpatialAssignment) -> SpatialAssignment:
        return SpatialAssignment(
            scenario_id=scenario_id,
            event_space_class="traffic_incident_grid",
            target_center_enu_m=list(cell.representative_enu_m),
            grid_cell=cell,
            source="main_road_grid.traffic_incident_cell.vehicle_presence_fallback",
            traffic_incident_id=primary.traffic_incident_id,
            accident_class=primary.accident_class,
            exclusion_radius_m=spatial_grid.accident_exclusion_radius_m,
        )

    def append_assigned(scenario_id: str, builder: Callable[[], ScenarioBundle]) -> None:
        primary = spatial_assignments[scenario_id]
        candidates = [primary]
        contract = get_contract(scenario_id)
        if selected and REUSE_FROZEN_LAYOUT:
            # A partial regeneration rebuilds only the requested scenarios.  For those,
            # the frozen on-disk cell is authoritative: re-deriving it from the global
            # cursor would relocate the whole scene to a different world origin.
            frozen = _frozen_assignment_for(scenario_id)
            if frozen is not None:
                primary = frozen
                candidates = [frozen]
        if primary.event_space_class == "traffic_incident_grid":
            original_order = {cell.cell_id: index for index, cell in enumerate(spatial_grid.cells)}

            def traffic_sort_key(cell: Any) -> tuple[Any, ...]:
                sidewalk_distance_m = _sidewalk_context_distance_m(cell.representative_enu_m)
                min_vehicle_count, total_vehicle_count, all_seed_count = spatial_grid.vehicle_presence_score(cell)
                return (
                    contract.vehicle > 0 and min_vehicle_count <= 0,
                    contract.pedestrian > 0 and sidewalk_distance_m > CONTRACT_RUNTIME_VISIBLE_PADDING_M,
                    cell.cell_id in used_traffic_cell_ids,
                    -all_seed_count,
                    -min_vehicle_count,
                    -total_vehicle_count,
                    sidewalk_distance_m if contract.pedestrian > 0 else 0.0,
                    original_order[cell.cell_id],
                )

            traffic_cells = sorted(spatial_grid.cells, key=traffic_sort_key)
            candidates.extend(
                traffic_assignment_for_cell(scenario_id, cell, primary)
                for cell in traffic_cells
                if cell.cell_id != primary.grid_cell.cell_id
            )
            if selected:
                candidates = candidates[:32]
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
            if selected:
                candidates = candidates[:16]
            if contract.pedestrian > 0 or contract.vehicle > 0:
                original_order = {candidate.grid_cell.cell_id: index for index, candidate in enumerate(candidates)}

                def candidate_sort_key(candidate: SpatialAssignment) -> tuple[Any, ...]:
                    sidewalk_distance_m = _sidewalk_context_distance_m(candidate.target_center_enu_m)
                    vehicle_counts = spatial_grid.vehicle_counts_by_seed(candidate.grid_cell)
                    min_vehicle_count = min(vehicle_counts) if vehicle_counts else 0
                    total_vehicle_count = sum(vehicle_counts)
                    return (
                        contract.vehicle > 0 and min_vehicle_count <= 0,
                        contract.pedestrian > 0 and sidewalk_distance_m > CONTRACT_RUNTIME_VISIBLE_PADDING_M,
                        candidate.grid_cell.cell_id in used_nontraffic_cell_ids,
                        -min_vehicle_count if contract.vehicle > 0 else 0,
                        -total_vehicle_count if contract.vehicle > 0 else 0,
                        sidewalk_distance_m if contract.pedestrian > 0 else 0.0,
                        original_order[candidate.grid_cell.cell_id],
                    )

                candidates.sort(key=candidate_sort_key)
        errors: list[str] = []
        reset_world_event_origin()
        global SPATIAL_PROVISIONAL_BUILD
        global ACTIVE_ROI_CONTRACT
        previous_provisional_build = SPATIAL_PROVISIONAL_BUILD
        previous_roi_contract = ACTIVE_ROI_CONTRACT
        SPATIAL_PROVISIONAL_BUILD = True
        ACTIVE_ROI_CONTRACT = None
        try:
            provisional = builder()
        finally:
            SPATIAL_PROVISIONAL_BUILD = previous_provisional_build
            ACTIVE_ROI_CONTRACT = previous_roi_contract
        provisional_center = _bundle_assignment_center(provisional)
        if scenario_id in roi_contracts:
            ACTIVE_ROI_CONTRACT = roi_contracts[scenario_id]
            try:
                bundle = build_roi_bound_bundle(
                    scenario_id,
                    builder,
                    primary,
                    spatial_grid,
                    roi_contracts[scenario_id],
                    provisional,
                )
            finally:
                ACTIVE_ROI_CONTRACT = None
            bundles.append(bundle)
            return
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
                elif assignment.event_space_class == "traffic_incident_grid":
                    used_traffic_cell_ids.add(assignment.grid_cell.cell_id)
                return
            except RuntimeError as exc:
                errors.append(str(exc))
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


def _runtime_state_value(
    mapping: Mapping[str, Any],
    family: str,
    field: str,
) -> bool | None:
    family_state = mapping.get(family)
    if not isinstance(family_state, Mapping) or field not in family_state:
        return None
    value = family_state[field]
    return value if isinstance(value, bool) else None


def _point_route_distance_xy(point: Sequence[float], route: Sequence[Sequence[float]]) -> float:
    if len(route) < 2:
        raise ValueError("route requires at least two waypoints")
    return float(
        LineString([(float(item[0]), float(item[1])) for item in route]).distance(
            Point(float(point[0]), float(point[1]))
        )
    )


def validate_v8_bundle_semantics(bundle: ScenarioBundle) -> list[str]:
    """Reject authored state/event contradictions before expensive simulation."""

    errors: list[str] = []
    sid = bundle.scenario_id
    scene_by_id = {
        str(scene.get("entity_id") or ""): scene for scene in bundle.scene_entities
    }
    uav_ids = {
        entity_id
        for entity_id, scene in scene_by_id.items()
        if str(scene.get("logical_asset_id") or "").startswith("uav.")
    }
    try:
        dispatch_ticks = _event_dispatch_upper_bounds(bundle.events)
    except RuntimeError as exc:
        return [f"{sid}: {exc}"]

    motion_windows: dict[str, list[tuple[int, int, str]]] = {}
    state_changes: dict[
        str, list[tuple[int, int, int, str, str, bool]]
    ] = {}
    landing_event_ids: set[str] = set()
    planned_deviation_points: dict[str, list[list[float]]] = {}

    for event_index, event_def in enumerate(bundle.events):
        event_id = str(event_def.get("event_id") or "")
        dispatch_tick = dispatch_ticks[event_id]
        event_intent = str(event_def.get("intent") or "")
        event_has_landed_move = False
        event_text = " ".join(
            (
                event_id,
                str(event_def.get("log_title") or ""),
                event_intent,
            )
        ).casefold()
        deviation_event = any(token in event_text for token in ("drift", "deviat", "spoofed_route_offset"))
        for action_index, action_def in enumerate(event_def.get("actions") or []):
            params = action_params(action_def)
            action_type = str(params.get("type") or "")
            entity_id = str(params.get("entity_id") or "")
            action_id = str(params.get("action_id") or "")
            if action_type == "move_entity":
                duration = move_duration_ticks(action_def)
                if duration > 0:
                    motion_windows.setdefault(entity_id, []).append(
                        (dispatch_tick + 1, dispatch_tick + duration, action_id)
                    )
                activity = str(params.get("activity_type") or "").casefold()
                post_activity = str(params.get("post_activity_type") or "").casefold()
                if activity == "landing":
                    errors.append(
                        f"{sid}:{action_id}: moving descent cannot use the ground-contact state 'landing'"
                    )
                if post_activity == "landed":
                    event_has_landed_move = True
                    landing_event_ids.add(event_id)
                    terminal = params.get("terminal_feasibility")
                    if not isinstance(terminal, Mapping):
                        errors.append(f"{sid}:{action_id}: landed move lacks terminal feasibility proof")
                    route = params.get("waypoints_enu_m") or []
                    landing_reference = (
                        terminal.get("landing_reference_enu_m")
                        if isinstance(terminal, Mapping)
                        else None
                    )
                    if not route or not isinstance(landing_reference, list) or len(landing_reference) < 3:
                        errors.append(f"{sid}:{action_id}: landed move lacks explicit landing reference")
                    elif math.dist(
                        [float(value) for value in route[-1][:3]],
                        [float(value) for value in landing_reference[:3]],
                    ) > 1e-6:
                        errors.append(f"{sid}:{action_id}: landed endpoint differs from landing reference")
                    approach = route[-2] if len(route) >= 2 else None
                    if approach is None or float(approach[2]) - float(route[-1][2]) <= UAV_FINAL_DESCENT_MIN_M:
                        errors.append(f"{sid}:{action_id}: landed move lacks a descending final leg")
                if deviation_event and entity_id in uav_ids:
                    planned_deviation_points.setdefault(entity_id, []).extend(
                        [
                            list(point)
                            for point in params.get("waypoints_enu_m") or []
                            if isinstance(point, list) and len(point) >= 2
                        ]
                    )
            elif action_type == "set_runtime_state":
                patch = params.get("state_patch")
                if not isinstance(patch, Mapping):
                    continue
                change_tick = dispatch_tick + max(1, int(params.get("delay_ticks") or 0))
                for family, raw_family_patch in patch.items():
                    if not isinstance(raw_family_patch, Mapping):
                        continue
                    for field, value in raw_family_patch.items():
                        if isinstance(value, bool):
                            state_changes.setdefault(entity_id, []).append(
                                (
                                    change_tick,
                                    event_index,
                                    action_index,
                                    str(family),
                                    str(field),
                                    value,
                                )
                            )
        if event_intent == "landing_or_terminal_resolution" and not event_has_landed_move:
            errors.append(f"{sid}:{event_id}: landing resolution has no landed terminal move")

    def state_at(entity_id: str, family: str, field: str, tick_value: int) -> bool | None:
        value = _runtime_state_value(scene_by_id.get(entity_id, {}), family, field)
        for change_tick, event_index, action_index, change_family, change_field, change_value in sorted(
            state_changes.get(entity_id, [])
        ):
            if change_tick > tick_value:
                break
            if change_family == family and change_field == field:
                value = change_value
        return value

    for entity_id, windows in motion_windows.items():
        if entity_id not in uav_ids and not str(
            scene_by_id.get(entity_id, {}).get("logical_asset_id") or ""
        ).startswith("vehicle."):
            continue
        governed_fields = [("control_state", "safe_hold_active")]
        if entity_id not in uav_ids:
            governed_fields.append(("vehicle_state", "emergency_stop"))
        for family, field in governed_fields:
            changes = [
                item
                for item in sorted(state_changes.get(entity_id, []))
                if item[3] == family and item[4] == field
            ]
            current = _runtime_state_value(scene_by_id.get(entity_id, {}), family, field)
            active_start = 0 if current is True else None
            active_intervals: list[tuple[int, int]] = []
            for change_tick, _, _, _, _, change_value in changes:
                if current is True and not change_value and active_start is not None:
                    active_intervals.append((active_start, change_tick))
                    active_start = None
                elif current is not True and change_value:
                    active_start = change_tick
                current = change_value
            if current is True and active_start is not None:
                active_intervals.append((active_start, bundle.duration_ticks + 1))
            for motion_start, motion_end, action_id in windows:
                if any(max(motion_start, start) < min(motion_end, end) for start, end in active_intervals):
                    errors.append(
                        f"{sid}:{action_id}: motion overlaps {family}.{field}=true for {entity_id}"
                    )

    for entity_id, changes in state_changes.items():
        for change_tick, _, _, family, field, value in changes:
            if family != "communication_state" or field != "alternate_link_active" or not value:
                continue
            if state_at(entity_id, "communication_state", "communication_unavailable", change_tick) is True:
                errors.append(
                    f"{sid}:{entity_id}: alternate link activates while communication remains unavailable"
                )
            unavailable_stations = [
                station_id
                for station_id in scene_by_id
                if state_at(station_id, "communication_state", "station_unavailable", change_tick) is True
            ]
            if unavailable_stations:
                errors.append(
                    f"{sid}:{entity_id}: alternate link activates while stations remain unavailable: "
                    f"{unavailable_stations}"
                )

    boundary = dict(bundle.parameters.get("capture_boundary") or {})
    boundary_polygon = [
        [float(point[0]), float(point[1])]
        for point in boundary.get("polygon_enu_m") or []
        if isinstance(point, list) and len(point) >= 2
    ]
    for entity_id, scene in scene_by_id.items():
        if str(scene.get("uav_corridor_role") or (scene.get("initial_state") or {}).get("uav_corridor_role") or "") == "observer":
            lifecycle = scene.get("observer_lifecycle")
            route = scene.get("route_waypoints_enu_m") or []
            if not isinstance(lifecycle, Mapping) or lifecycle.get("presence") != "episode_full_duration":
                errors.append(f"{sid}:{entity_id}: observer lacks full-duration lifecycle")
            elif len(route) < 2 or not boundary_polygon:
                errors.append(f"{sid}:{entity_id}: observer route/boundary is incomplete")
            else:
                limit = float(lifecycle.get("maximum_terminal_boundary_distance_m") or -1.0)
                distance = _distance_to_polygon_xy(list(route[-1]), boundary_polygon)
                if limit < 0.0 or distance > limit + 1e-6:
                    errors.append(
                        f"{sid}:{entity_id}: observer terminal boundary distance "
                        f"{distance:.3f}m exceeds {limit:.3f}m"
                    )

        activation_tick = scene.get("activation_tick")
        if (
            str(scene.get("entity_id") or "").startswith("temporary_nfz_")
            and isinstance(activation_tick, int)
            and activation_tick <= bundle.duration_ticks
        ):
            deactivation_tick = scene.get("deactivation_tick")
            if (
                not isinstance(deactivation_tick, int)
                or deactivation_tick <= activation_tick
                or deactivation_tick > bundle.duration_ticks
            ):
                errors.append(f"{sid}:{entity_id}: temporary NFZ lacks an in-window deactivation")
            clear_events = [
                event_def
                for event_def in bundle.events
                if str(event_def.get("event_id") or "").endswith(("cleared", "clear"))
                and entity_id in (event_def.get("log_target_ids") or [])
            ]
            if not clear_events:
                errors.append(f"{sid}:{entity_id}: temporary NFZ lacks an explicit clear event")

    minimum_deviation = bundle.parameters.get("path_deviation_min_m")
    if minimum_deviation is not None:
        threshold = float(minimum_deviation)
        candidates: list[tuple[str, float]] = []
        for entity_id, points in planned_deviation_points.items():
            planned_route = scene_by_id.get(entity_id, {}).get("planned_route_waypoints_enu_m") or []
            if len(planned_route) < 2:
                continue
            candidates.extend(
                (entity_id, _point_route_distance_xy(point, planned_route)) for point in points
            )
        maximum = max((distance for _, distance in candidates), default=-1.0)
        if maximum <= threshold:
            errors.append(
                f"{sid}: authored route deviation ceiling={maximum:.3f}m does not exceed "
                f"the governed {threshold:.3f}m threshold"
            )

    return errors


def validate_bundle(bundle: ScenarioBundle, compiled: dict[str, Any], catalog_ids: set[str]) -> list[str]:
    errors = validate_v8_bundle_semantics(bundle)
    scene_ids = {e["entity_id"] for e in bundle.scene_entities}
    scene_assets = {e["entity_id"]: e["logical_asset_id"] for e in bundle.scene_entities}
    current_positions = {str(e["entity_id"]): scene_pos(e) for e in bundle.scene_entities}
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
            if a["type"] == "set_runtime_state":
                try:
                    validate_runtime_state_patch_mapping(
                        a.get("state_patch"),
                        context=f"{bundle.scenario_id}: {a.get('action_id') or 'set_runtime_state'}",
                    )
                except RuntimeError as exc:
                    errors.append(str(exc))
            if a["type"] == "move_entity":
                asset = scene_assets.get(ent_id, "")
                if asset in {"prop.traffic_control.signal_light.v1", "facility.radio.base_tower.v1"}:
                    errors.append(f"{bundle.scenario_id}: static infrastructure moved: {ent_id}")
                waypoints = [
                    q(float(wp[0]), float(wp[1]), float(wp[2]))
                    for wp in a.get("waypoints_enu_m", [])
                    if isinstance(wp, list) and len(wp) >= 3
                ]
                expected_start = current_positions.get(str(ent_id))
                if expected_start and waypoints:
                    tolerance = MOVE_START_TOLERANCE_UAV_M if asset.startswith("uav.") else MOVE_START_TOLERANCE_GROUND_M
                    if dist3(waypoints[0], expected_start) > tolerance and not compiled_event_allows_validation_skip(
                        e,
                        "move_start_continuity",
                        errors,
                        bundle.scenario_id,
                    ):
                        errors.append(
                            f"{bundle.scenario_id}: move_entity {a.get('action_id')} starts "
                            f"{dist3(waypoints[0], expected_start):.1f}m from current {ent_id} position"
                        )
                motion_class = motion_class_from_asset(asset)
                max_segment_m = MAX_WAYPOINT_SEGMENT_M[motion_class]
                for segment_index, (start_wp, end_wp) in enumerate(zip(waypoints, waypoints[1:])):
                    segment_m = dist_xy(start_wp, end_wp)
                    if segment_m > max_segment_m and not compiled_event_allows_validation_skip(
                        e,
                        "move_segment_continuity",
                        errors,
                        bundle.scenario_id,
                    ):
                        errors.append(
                            f"{bundle.scenario_id}: move_entity {a.get('action_id')} segment "
                            f"{segment_index}->{segment_index + 1} is {segment_m:.1f}m, exceeds {motion_class} limit {max_segment_m:.1f}m"
                        )
                if waypoints:
                    current_positions[str(ent_id)] = waypoints[-1]
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
            governed_max_distance_m = 50.0
            x3_chain = dict(bundle.parameters.get("x3_physical_chain") or {})
            if (
                t.get("entity_a") == "uav_x3_detect"
                and t.get("entity_b") == "ped_x3_fall"
                and str(t.get("metric") or "").lower() == "3d"
            ):
                governed_max_distance_m = float(
                    x3_chain.get("uav_detection_radius_m") or governed_max_distance_m
                )
            if d <= 0 or d > governed_max_distance_m:
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
    scene_by_id = {str(scene["entity_id"]): scene for scene in bundle.scene_entities}
    for rule in bundle.validation_rules or []:
        if rule.get("rule") != "pre_mission_takeoff_boundary_clearance":
            continue
        boundary_id = str(rule.get("boundary_id") or "")
        entity_id = str(rule.get("entity_id") or "")
        boundary_scene = scene_by_id.get(boundary_id)
        polygon = [
            [float(point[0]), float(point[1])]
            for point in (boundary_scene or {}).get("placement", {}).get("polygon_enu_m", [])
            if isinstance(point, list) and len(point) >= 2
        ]
        minimum_distance_m = float(rule.get("minimum_distance_m") or 0.0)
        if not polygon or minimum_distance_m <= 0.0:
            errors.append(
                f"{sid}: takeoff clearance rule lacks a valid polygon or positive minimum"
            )
            continue
        takeoff_actions = [
            action_params(action_def)
            for event_def in bundle.events
            for action_def in event_def.get("actions") or []
            if action_params(action_def).get("type") == "move_entity"
            and str(action_params(action_def).get("entity_id") or "") == entity_id
            and "takeoff_entry"
            in str(action_params(action_def).get("action_id") or "").lower()
        ]
        if len(takeoff_actions) != 1:
            errors.append(
                f"{sid}: expected exactly one takeoff route for {entity_id}, got "
                f"{len(takeoff_actions)}"
            )
            continue
        expected_takeoff_speed_mps = float(rule.get("takeoff_speed_mps") or 0.0)
        actual_takeoff_speed_mps = float(
            takeoff_actions[0].get("velocity_mps") or 0.0
        )
        if expected_takeoff_speed_mps <= 0.0 or not math.isclose(
            actual_takeoff_speed_mps,
            expected_takeoff_speed_mps,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append(
                f"{sid}: takeoff speed for {entity_id} is "
                f"{actual_takeoff_speed_mps:.3f}m/s, expected "
                f"{expected_takeoff_speed_mps:.3f}m/s"
            )
        takeoff_route = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in takeoff_actions[0].get("waypoints_enu_m") or []
            if isinstance(point, list) and len(point) >= 3
        ]
        if not takeoff_route:
            errors.append(f"{sid}: takeoff route is empty for {entity_id}")
            continue
        distance_m = _route_distance_to_polygon_xy(takeoff_route, polygon)
        if distance_m + 1e-6 < minimum_distance_m:
            errors.append(
                f"{sid}: takeoff route for {entity_id} is {distance_m:.3f}m from "
                f"{boundary_id}, below required {minimum_distance_m:.3f}m"
            )
        maximum_route_length_m = float(rule.get("maximum_route_length_m") or 0.0)
        if maximum_route_length_m <= 0.0:
            errors.append(f"{sid}: takeoff clearance rule lacks a positive route limit")
            continue
        route_length_m = path_length_m(takeoff_route)
        if route_length_m > maximum_route_length_m + 1e-6:
            errors.append(
                f"{sid}: takeoff route for {entity_id} is {route_length_m:.3f}m long, "
                f"above the pre-mission limit {maximum_route_length_m:.3f}m"
            )
    for rule in bundle.validation_rules or []:
        if rule.get("rule") != "post_conflict_landing_boundary_clearance":
            continue
        boundary_id = str(rule.get("boundary_id") or "")
        entity_id = str(rule.get("entity_id") or "")
        boundary_scene = scene_by_id.get(boundary_id)
        polygon = [
            [float(point[0]), float(point[1])]
            for point in (boundary_scene or {}).get("placement", {}).get(
                "polygon_enu_m", []
            )
            if isinstance(point, list) and len(point) >= 2
        ]
        minimum_distance_m = float(rule.get("minimum_distance_m") or 0.0)
        if not polygon or minimum_distance_m <= 0.0:
            errors.append(
                f"{sid}: landing clearance rule lacks a valid polygon or positive minimum"
            )
            continue
        landing_actions = [
            action_params(action_def)
            for event_def in bundle.events
            for action_def in event_def.get("actions") or []
            if action_params(action_def).get("type") == "move_entity"
            and str(action_params(action_def).get("entity_id") or "") == entity_id
            and "landing_return"
            in str(action_params(action_def).get("action_id") or "").lower()
        ]
        if len(landing_actions) != 1:
            errors.append(
                f"{sid}: expected exactly one landing route for {entity_id}, got "
                f"{len(landing_actions)}"
            )
            continue
        landing_route = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in landing_actions[0].get("waypoints_enu_m") or []
            if isinstance(point, list) and len(point) >= 3
        ]
        if not landing_route:
            errors.append(f"{sid}: landing route is empty for {entity_id}")
            continue
        distance_m = _route_distance_to_polygon_xy(landing_route, polygon)
        if distance_m + 1e-6 < minimum_distance_m:
            errors.append(
                f"{sid}: landing route for {entity_id} is {distance_m:.3f}m from "
                f"{boundary_id}, below required {minimum_distance_m:.3f}m"
            )
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
    roi_contract_payload = dict(bundle.parameters.get("roi_contract") or {})
    contract_polygon = [
        [float(point[0]), float(point[1])]
        for point in roi_contract_payload.get("capture_polygon_enu_m") or []
    ]
    if roi_contract_payload:
        boundary = dict(bundle.parameters.get("capture_boundary") or {})
        polygon = [
            [float(point[0]), float(point[1])]
            for point in boundary.get("polygon_enu_m") or []
        ]
        if polygon != contract_polygon:
            errors.append(f"{sid}: capture polygon diverges from ROI contract")
        if str(boundary.get("boundary_id") or "") != str(
            roi_contract_payload.get("capture_boundary_id") or ""
        ):
            errors.append(f"{sid}: capture_boundary_id diverges from ROI contract")
        if contract_polygon:
            assignment_center = _bundle_assignment_center(bundle)
            roi = Polygon((float(point[0]), float(point[1])) for point in contract_polygon)
            assignment_point = Point(float(assignment_center[0]), float(assignment_center[1]))
            outside_m = (
                0.0 if roi.covers(assignment_point) else float(roi.distance(assignment_point))
            )
            if outside_m > ROI_ASSIGNMENT_MAX_OUTSIDE_M:
                errors.append(
                    f"{sid}: assignment center is {outside_m:.3f}m outside ROI contract"
                )
    if sid.startswith("L4-10"):
        amb = [e for e in bundle.scene_entities if e["logical_asset_id"] == "vehicle.emergency.ambulance.v1"]
        if not amb or not amb[0].get("initial_state", {}).get("lights_on"):
            errors.append(f"{sid}: ambulance lights_on true missing")
        if contract_polygon:
            roi = Polygon((float(point[0]), float(point[1])) for point in contract_polygon)
            ambulance_inside = False
            for event_def in bundle.events:
                if str(event_def.get("event_id") or "") != "ambulance_priority_approach":
                    continue
                for action_def in event_def.get("actions") or []:
                    params = action_params(action_def)
                    if str(params.get("type") or "") != "move_entity":
                        continue
                    entity_id = str(params.get("entity_id") or "")
                    if "ambulance" not in entity_id:
                        continue
                    for waypoint in params.get("waypoints_enu_m") or []:
                        if (
                            isinstance(waypoint, list)
                            and len(waypoint) >= 2
                            and roi.covers(Point(float(waypoint[0]), float(waypoint[1])))
                        ):
                            ambulance_inside = True
            if not ambulance_inside:
                errors.append(f"{sid}: ambulance priority route does not enter ROI contract")
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
    parser.add_argument(
        "--runtime-state-schedule-profile-dir",
        type=Path,
        default=RUNTIME_STATE_SCHEDULE_PROFILE_DIR,
        help="Directory containing epi runtime-state schedule profile JSON files.",
    )
    parser.add_argument(
        "--spatial-authority-root",
        type=Path,
        help="Read-only formal capture-filtered root whose three-seed ROI polygon is the spatial authority for targeted recapture regeneration.",
    )
    parser.add_argument(
        "--allow-missing-runtime-state-schedules",
        action="store_true",
        help="Development-only option: do not fail when the runtime-state schedule profile directory is missing or empty.",
    )
    args = parser.parse_args()
    selected_scenarios = {item for group in args.scenario for item in group}
    recapture_selected = selected_scenarios & set(TARGET_RECAPTURE_SCENARIO_IDS)
    if recapture_selected and args.spatial_authority_root is None:
        raise SystemExit(
            "targeted regeneration of recapture profiles requires "
            "--spatial-authority-root pointing at the formal capture-filtered corpus"
        )
    with open(ASSET_CATALOG, "r", encoding="utf-8") as f:
        catalog_ids = {a["logical_asset_id"] for a in json.load(f)["assets"]}
    bundles = collect_bundles(
        selected_scenarios or None,
        spatial_authority_root=args.spatial_authority_root,
    )
    if selected_scenarios and not bundles:
        raise SystemExit(f"No matching scenarios for --scenario {sorted(selected_scenarios)}")
    require_l25_retained_scene_layout(bundles)
    runtime_state_schedules = load_runtime_state_schedule_profiles(
        args.runtime_state_schedule_profile_dir,
        scenario_filter={bundle.scenario_id for bundle in bundles} if selected_scenarios else None,
        require_profiles=not args.allow_missing_runtime_state_schedules,
    )
    if not args.allow_missing_runtime_state_schedules:
        require_runtime_state_schedule_coverage(bundles, runtime_state_schedules)
        objective_contract = json.loads(
            OBJECTIVE_SEMANTIC_CONTRACT_PROFILE_PATH.read_text(encoding="utf-8-sig")
        )
        require_runtime_schedule_api_impact(runtime_state_schedules, objective_contract)
    require_l25_signal_repair_schedule(bundles, runtime_state_schedules)
    apply_runtime_state_schedules(bundles, runtime_state_schedules)
    errors: list[str] = []
    compiled_bundles: list[tuple[ScenarioBundle, dict[str, Any]]] = []
    for bundle in bundles:
        compiled = compile_bundle(bundle)
        if bundle.scenario_id == "L2-5_v1":
            require_l25_compiled_state_actions(compiled, runtime_state_schedules["L2-5_v1"])
            require_l25_retained_motion_actions(bundle, compiled)
        errors.extend(validate_bundle(bundle, compiled, catalog_ids))
        compiled_bundles.append((bundle, compiled))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    for bundle, compiled in compiled_bundles:
        bundle.directory.mkdir(parents=True, exist_ok=True)
        write_json(bundle.directory / "event_script.json", compiled)
        write_json(bundle.directory / "scene_setup.json", scene_setup_from_bundle(bundle))
        write_text_retry(bundle.directory / "spec.py", write_spec_py(bundle))
    print(f"Generated {len(bundles)} scenario specs, scene setups, and event scripts.")


if __name__ == "__main__":
    main()
