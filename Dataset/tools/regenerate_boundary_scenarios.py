"""Regenerate semantic P0 scenario specs and compiled event scripts.

This generator replaces the old placeholder specs with concrete
SpecCompiler ScenarioSpec definitions.  It also emits scene_setup.json files
because the boundary document requires semantic placement, asset resolution,
and validation metadata that event_script_v1 does not carry.
"""

from __future__ import annotations

import json
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

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
BUILDING_GEOJSON = ROOT / "Content" / "Maps" / "donghu_road_topo" / "building" / "building.geojson"
WORLD_OFFSET_X_M = 7000.0
WORLD_OFFSET_Y_M = 6200.0
LANE_HALF_WIDTH_M = 1.9
SIDEWALK_MIN_OFFSET_FROM_CURB_M = 1.2
GATHERING_MIN_OFFSET_FROM_CURB_M = 4.5
ROADWORK_MIN_OFFSET_FROM_EDGE_M = 1.5
GROUND_Z_M = 0.0

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


LANES = LaneSampleResolver(LANE_SAMPLES_CSV)
BUILDINGS = BuildingCatalog(BUILDING_GEOJSON)


def p(x: float, y: float, z: float = 0.0) -> list[float]:
    return [round(x + WORLD_OFFSET_X_M, 3), round(y + WORLD_OFFSET_Y_M, 3), round(z, 3)]


def q(x: float, y: float, z: float = 0.0) -> list[float]:
    return [round(x, 3), round(y, 3), round(z, 3)]


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
    sample = LANES.nearest_to_xy(float(pos_enu[0]), float(pos_enu[1]))
    sign = _side_sign_for_desired(sample, pos_enu)
    lateral = sign * (LANE_HALF_WIDTH_M + max(abs(offset_from_curb_m), SIDEWALK_MIN_OFFSET_FROM_CURB_M))
    return _offset_from_lane(sample, lateral, GROUND_Z_M)


def gathering_point(pos_enu: list[float], extent_cm: list[float] | None = None) -> list[float]:
    sample = LANES.nearest_to_xy(float(pos_enu[0]), float(pos_enu[1]))
    sign = _side_sign_for_desired(sample, pos_enu)
    half_extent_m = 0.0
    if extent_cm:
        half_extent_m = max(float(extent_cm[0]), float(extent_cm[1])) / 200.0
    offset_from_curb = max(GATHERING_MIN_OFFSET_FROM_CURB_M, half_extent_m + 1.0)
    return _offset_from_lane(sample, sign * (LANE_HALF_WIDTH_M + offset_from_curb), GROUND_Z_M)


def scene_pos(scene: dict[str, Any]) -> list[float]:
    placement = scene["placement"]
    return placement.get("resolved_position_enu_m") or placement.get("position_enu_m") or placement.get("center_enu_m")


def shifted_sidewalk_point(base_pos_enu: list[float], dx_m: float, dy_m: float, offset_from_curb_m: float = SIDEWALK_MIN_OFFSET_FROM_CURB_M) -> list[float]:
    return sidewalk_point([base_pos_enu[0] + dx_m, base_pos_enu[1] + dy_m, GROUND_Z_M], offset_from_curb_m)


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


def move(action_id: str, entity_id: str, waypoints: list[list[float]], velocity: float) -> dict[str, Any]:
    return action(
        action_id,
        "move_entity",
        entity_id=entity_id,
        waypoints_enu_m=waypoints,
        velocity_mps=velocity,
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


def remove(action_id: str, entity_id: str) -> dict[str, Any]:
    return action(action_id, "remove_entity", entity_id=entity_id)


def screenshot(action_id: str, camera_id: str = "demo_high_overview") -> dict[str, Any]:
    return action(action_id, "capture_screenshot", camera_id=camera_id)


def play(action_id: str, ped_id: str) -> dict[str, Any]:
    return action(
        action_id,
        "play_animation",
        ped_id=ped_id,
        animation_path="/AeroSimHost/Animations/AM_Crouch",
        start_section="",
        play_rate=1.0,
        loop_count=1,
    )


def crowd(action_id: str, group_id: str, count: int, origin: list[float], extent_cm: list[float], seed: int) -> dict[str, Any]:
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
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "trigger": trigger,
        "actions": actions,
        "priority": priority,
        "max_fire_count": 1,
        "on_fire_emit": emits or [],
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
    sample = LANES.nearest_to_xy(float(pos_enu[0]), float(pos_enu[1]))
    sign = _side_sign_for_desired(sample, pos_enu)
    offset_from_curb = max(abs(float(offset)), SIDEWALK_MIN_OFFSET_FROM_CURB_M)
    resolved_lateral = sign * (LANE_HALF_WIDTH_M + offset_from_curb)
    pos_enu = _offset_from_lane(sample, resolved_lateral, float(pos_enu[2] if len(pos_enu) > 2 else GROUND_Z_M))
    yaw = sample.yaw_deg
    spec, scene = world_entity(entity_id, asset, category, pos_enu, yaw, mode, route=route)
    scene["placement_mode"] = "sidewalk_anchor"
    scene["placement"] = {
        "lane_edge_id": sample.edge_id,
        "longitudinal_s": round(sample.s_m, 3),
        "offset_from_curb_m": round(offset_from_curb, 3),
        "lane_half_width_m": LANE_HALF_WIDTH_M,
        "resolved_lateral_from_center_m": round(resolved_lateral, 3),
        "placement_semantics": "sidewalk_or_plaza",
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
    sample = LANES.nearest_to_xy(float(pos_enu[0]), float(pos_enu[1]))
    sign = _side_sign_for_desired(sample, pos_enu)
    start_pos = _offset_from_lane(sample, sign * (LANE_HALF_WIDTH_M + SIDEWALK_MIN_OFFSET_FROM_CURB_M), GROUND_Z_M)
    road_pos = _offset_from_lane(sample, 0.0, GROUND_Z_M)
    opposite_pos = _offset_from_lane(sample, -sign * (LANE_HALF_WIDTH_M + SIDEWALK_MIN_OFFSET_FROM_CURB_M), GROUND_Z_M)
    spec, scene = world_entity(entity_id, asset, "pedestrian", start_pos, sample.yaw_deg, "walking", route=route)
    scene["placement_mode"] = "crosswalk_anchor"
    scene["placement"] = {
        "crosswalk_id": crosswalk_id,
        "side": side,
        "lane_edge_id": sample.edge_id,
        "longitudinal_s": round(sample.s_m, 3),
        "lane_half_width_m": LANE_HALF_WIDTH_M,
        "offset_from_curb_m": SIDEWALK_MIN_OFFSET_FROM_CURB_M,
        "resolved_lateral_from_center_m": round(sign * (LANE_HALF_WIDTH_M + SIDEWALK_MIN_OFFSET_FROM_CURB_M), 3),
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, scene = world_entity(entity_id, "facility.landing_pad.visible.v1", "facility", pos_enu, yaw, "available")
    scene["placement_mode"] = "pad_anchor"
    scene["placement"] = {
        "pad_instance_id": pad_instance_id,
        "approach_side": approach_side,
        "position_enu_m": pos_enu,
        "resolved_position_enu_m": pos_enu,
        "rotation_deg": rot(yaw),
    }
    return spec, scene


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
    ids = [e["entity_id"] for e in scene_entities]
    rules = base_rules(ids, scenario_id) + asset_rules(scene_entities)
    if validation_rules:
        rules.extend(validation_rules)
    if len(rules) < 2 and ids:
        rules.append({"rule": "entity_resolvable", "entity_id": ids[0], "description": "Entity must resolve"})
    cx = sum(e["initial_pos_enu"][0] for e in spec_entities) / max(1, len(spec_entities))
    cy = sum(e["initial_pos_enu"][1] for e in spec_entities) / max(1, len(spec_entities))
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
        cameras=cameras or camera_for(cx, cy),
        validation_rules=rules,
    )


def add(target_specs: list[dict[str, Any]], target_scenes: list[dict[str, Any]], pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, scene = pair
    target_specs.append(spec)
    target_scenes.append(scene)
    return pair


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
    asset = "trigger.no_fly.box.v1" if scenario_id.startswith(("L1-1", "L1-3")) else "trigger.hazard.generic.box.v1"
    add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 35, "patrol", route=[near, safe]))
    if scenario_id.startswith("L1-1"):
        polygon = [
            p(center[0] - 9, center[1] - 7, 22),
            p(center[0] + 11, center[1] - 6, 22),
            p(center[0] + 12, center[1] + 8, 22),
            p(center[0] - 8, center[1] + 9, 22),
        ]
        add(specs, scenes, polygon_entity(zone, asset, "airspace_constraint", polygon, 20.0, 26.0, center))
    else:
        add(specs, scenes, box_entity(zone, asset, "airspace_constraint", center, [14.0, 10.0, 14.0]))
    events = [
        event(
            "approach_boundary",
            tick(220),
            [move("move_boundary_approach", uav, [start, p(near[0] - 4, near[1] - 2, near[2]), near], 8.0)],
            1,
            scenario_id,
            "UAV approaches constrained airspace",
            "uav_mission",
            [uav, zone],
            "info",
        ),
        event(
            "boundary_conflict",
            prox(uav, zone, 14.0, 3),
            [
                move("move_boundary_avoidance", uav, [near, p(near[0] - 6, near[1] + 4, near[2] + 4)], 3.0),
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
            [move("move_boundary_return_safe", uav, [p(near[0] - 6, near[1] + 4, near[2] + 4), safe], 9.0)],
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
        intr_near = p(center[0] + 3, center[1] - 2, 29)
        add(specs, scenes, world_entity(intruder, "uav.airsim.flying_pawn.v1", "uav", intr_start, 215, "noncooperative", route=[intr_near]))
        events[0]["actions"].append(move("move_intruder_converge", intruder, [intr_start, intr_near], 10.0))
        events[1]["log_target_ids"].append(intruder)
    if scenario_id.startswith("L1-4"):
        uav_b = f"uav_{scenario_id.lower().replace('-', '_')}_secondary"
        uav_c = f"uav_{scenario_id.lower().replace('-', '_')}_tertiary"
        add(specs, scenes, world_entity(uav_b, "uav.airsim.cv_pawn.v1", "uav", p(ox + 6, oy - 18, 32), 20, "patrol"))
        add(specs, scenes, world_entity(uav_c, "uav.airsim.flying_pawn.v1", "uav", p(ox - 12, oy - 10, 34), 45, "patrol"))
        events[0]["actions"].extend(
            [
                move("move_corridor_secondary", uav_b, [p(ox + 6, oy - 18, 32), p(center[0] - 4, center[1] - 2, 32)], 7.0),
                move("move_corridor_tertiary", uav_c, [p(ox - 12, oy - 10, 34), p(center[0] - 2, center[1] + 1, 34)], 6.5),
            ]
        )
        events[1]["actions"].append(move("move_corridor_reroute_secondary", uav_b, [p(center[0] - 4, center[1] - 2, 32), p(center[0] + 16, center[1] + 16, 38)], 8.5))
        events[2]["actions"].append(move("move_corridor_resume_tertiary", uav_c, [p(center[0] - 2, center[1] + 1, 34), p(center[0] - 18, center[1] + 14, 34)], 7.5))
    if scenario_id.startswith("L1-2"):
        events[1]["actions"][0] = move("move_altitude_recover", uav, [near, p(near[0] - 3, near[1] + 4, 42), p(near[0] - 8, near[1] + 8, 34)], 4.0)
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
        events = [
            event("station_degraded", tick(260), [visual("set_tower_degraded", tower, "degraded")], 1, scenario_id, "Communication station degraded", "infrastructure", [tower], "warning"),
            event("uav_link_response", fired("station_degraded"), [move("move_uav_backup_loiter", uav, [start, degraded, p(degraded[0] + 3, degraded[1] + 5, 34)], 4.0)], 2, scenario_id, "UAV changes behavior under degraded C2", "uav_mission", [uav, tower], "warning"),
            event("backup_link_restore", fired("uav_link_response"), [visual("set_tower_backup", tower, "backup_link"), move("move_uav_resume_patrol", uav, [p(degraded[0] + 3, degraded[1] + 5, 34), home], 7.0)], 3, scenario_id, "Backup link restores service", "infrastructure", [uav, tower], "info"),
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
            event("visual_relocalization", prox(uav, facade, 8.0, 3), [visual("set_uav_visual_reloc", uav, "visual_relocalization"), screenshot("capture_gnss_drift")], 3, scenario_id, "Visual relocalization engages near facade", "uav_mission", [uav, facade], "warning"),
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
        add(specs, scenes, pad_entity(pad, p(ox + 2, oy + 2, 0), f"pad_{sid}", "east"))
        add(specs, scenes, world_entity(uav_a, "uav.inspect.quad.v1", "uav", start_a, 75, "emergency_landing", route=[p(ox + 2, oy + 2, 5)]))
        add(specs, scenes, world_entity(uav_b, "uav.airsim.flying_pawn.v1", "uav", start_b, 295, "landing_request", route=[hold_b]))
        events = [
            event("dual_pad_approach", tick(230), [move("move_priority_uav_to_pad", uav_a, [start_a, p(ox - 4, oy + 6, 22), p(ox + 2, oy + 2, 5)], 5.0), move("move_second_uav_to_pad_hold", uav_b, [start_b, p(ox + 6, oy - 2, 24), hold_b], 5.5)], 1, scenario_id, "Two UAVs request the same landing pad", "uav_mission", [uav_a, uav_b, pad], "warning"),
            event("priority_arbitration", prox(uav_a, pad, 7.0, 2), [visual("set_pad_reserved_priority", pad, "reserved"), visual("set_second_uav_hold", uav_b, "hold")], 2, scenario_id, "Pad priority arbitration reserves pad for emergency UAV", "infrastructure", [uav_a, uav_b, pad], "warning"),
            event("second_uav_diverted", fired("priority_arbitration"), [move("move_second_uav_diversion", uav_b, [hold_b, p(ox + 34, oy + 22, 34)], 8.0), move("move_priority_uav_landing", uav_a, [p(ox + 2, oy + 2, 5), p(ox + 2, oy + 2, 1.2)], 1.5)], 3, scenario_id, "Second UAV diverts while priority UAV lands", "uav_mission", [uav_a, uav_b, pad], "info"),
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
        prop_entities = []
        work_base = LANES.nearest_to_xy(*p(ox, oy, 0)[:2])
        work_edge = work_base.edge_id
        work_s = work_base.s_m
        for i in range(3):
            ent = f"barrier_l3_1_{i + 1:02d}"
            pos = p(ox + i * 4.5, oy, 0)
            _, scene = add(specs, scenes, lane_entity(ent, "prop.roadwork.barrier.v1", "roadwork_prop", work_edge, work_s + i * 5, 1.5, pos, 90, "staged", activation_tick=220, prefer_edge_hint=True))
            prop_entities.append((ent, "prop.roadwork.barrier.v1", scene_pos(scene)))
        for i, sx in enumerate([-3.0, 12.5]):
            ent = f"fence_l3_1_{i + 1:02d}"
            pos = p(ox + sx, oy + 1.6, 0)
            _, scene = add(specs, scenes, lane_entity(ent, "prop.roadwork.construction_fence.v1", "roadwork_prop", work_edge, work_s - 2 + i * 18, 1.5, pos, 90, "staged", activation_tick=220, prefer_edge_hint=True))
            prop_entities.append((ent, "prop.roadwork.construction_fence.v1", scene_pos(scene)))
        for i in range(5):
            ent = f"cone_l3_1_{i + 1:02d}"
            pos = p(ox - 2 + i * 3.6, oy - 2.4, 0)
            _, scene = add(specs, scenes, lane_entity(ent, "prop.roadwork.traffic_cone.v1", "roadwork_prop", work_edge, work_s - 4 + i * 4, 1.5, pos, 90, "staged", activation_tick=220, prefer_edge_hint=True))
            prop_entities.append((ent, "prop.roadwork.traffic_cone.v1", scene_pos(scene)))
        _, car_scene = add(specs, scenes, lane_entity(car, "vehicle.ground.boxcar.v1", "vehicle", work_edge, work_s + 12, 0.0, p(ox + 4, oy - 5, 0), 90, "blocked", prefer_edge_hint=True))
        car_start = scene_pos(car_scene)
        car_mid = road_center_point(p(ox + 2, oy - 14, 0))
        car_end = road_center_point(p(ox + 22, oy - 18, 0))
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", p(ox - 18, oy - 12, 32), 65, "inspection"))
        spawn_actions = [spawn(f"spawn_{ent}", ent, asset, pos, 90) for ent, asset, pos in prop_entities]
        events = [
            event("roadwork_barriers_spawn", tick(220), spawn_actions, 1, scenario_id, "Roadwork barriers and cones appear", "dynamic_constraint", [e[0] for e in prop_entities[:4]], "warning"),
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
        add(specs, scenes, box_entity(nfz, "trigger.no_fly.box.v1", "airspace_constraint", center, [12, 9, 15], activation_tick=280))
        events = [
            event("temporary_nfz_declared", tick(280), [spawn("spawn_temporary_nfz", nfz, "trigger.no_fly.box.v1", center, 0, visual_state={"mode": "active"})], 1, scenario_id, "Temporary no-fly zone declared mid-operation", "dynamic_constraint", [nfz], "warning"),
            event("uav_approaches_temporary_nfz", fired("temporary_nfz_declared"), [move("move_uav_to_nfz_edge", uav, [start, approach], 7.0)], 2, scenario_id, "UAV approaches newly declared NFZ", "uav_mission", [uav, nfz], "warning"),
            event("nfz_proximity_alert", prox(uav, nfz, 12.0, 3), [move("move_uav_nfz_reroute", uav, [approach, p(ox - 2, oy + 18, 34), p(ox + 24, oy + 28, 34)], 8.5), screenshot("capture_temporary_nfz")], 3, scenario_id, "UAV reroutes around temporary NFZ", "uav_mission", [uav, nfz], "info"),
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
        add(specs, scenes, box_entity(hazard, "trigger.hazard.generic.box.v1", "hazard_zone", center, [13, 10, 4], activation_tick=240))
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
            event("hazmat_leak", tick(240), [spawn("spawn_hazmat_zone", hazard, "trigger.hazard.generic.box.v1", center, 0, visual_state={"mode": "isolation_active"})], 1, scenario_id, "Hazmat leak declares isolation zone", "dynamic_constraint", [hazard], "critical"),
            event("pedestrian_evacuation", fired("hazmat_leak"), [move("move_ped_a_safe", ped_a, [ped_a_start, ped_a_safe], 1.6), move("move_ped_b_safe", ped_b, [ped_b_start, ped_b_safe], 1.4)], 2, scenario_id, "Pedestrians evacuate from hazmat zone", "pedestrian", [ped_a, ped_b, hazard], "warning"),
            event("ambulance_arrival", fired("pedestrian_evacuation"), [move("move_ambulance_hazmat", ambulance, [ambulance_start, ambulance_arrival], 11.0)], 3, scenario_id, "Ambulance arrives at isolation perimeter", "vehicle", [ambulance, hazard], "warning"),
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
            event("uav_conflict_resolution", prox(uav_a, uav_b, 8.0, 2), [visual("set_uav_a_hover", uav_a, "hover"), move("move_uav_b_altitude_reroute", uav_b, [meet, p(ox + 12, oy + 18, 36)], 9.0), screenshot("capture_uav_conflict")], 2, scenario_id, "UAV-UAV proximity conflict triggers separation", "uav_mission", [uav_a, uav_b], "critical"),
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
        start = p(ox - 12, oy - 12, 30)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 45, "fault"))
        _, ped_a_scene = add(specs, scenes, sidewalk_entity(ped_a, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_19", 24, GATHERING_MIN_OFFSET_FROM_CURB_M, p(ox + 2, oy + 2, 0), 0, "standing"))
        _, ped_b_scene = add(specs, scenes, sidewalk_entity(ped_b, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_19", 28, GATHERING_MIN_OFFSET_FROM_CURB_M, p(ox + 7, oy + 2, 0), 180, "standing"))
        ped_a_start = scene_pos(ped_a_scene)
        ped_b_start = scene_pos(ped_b_scene)
        crowd_origin = gathering_point(p(ox + 4, oy + 3, 0), [650, 420, 0])
        landing_x = (ped_a_start[0] + ped_b_start[0]) / 2.0
        landing_y = (ped_a_start[1] + ped_b_start[1]) / 2.0
        low = q(landing_x, landing_y, 8)
        land = q(landing_x, landing_y, 3)
        ped_a_evade = shifted_sidewalk_point(ped_a_start, -12, 12, GATHERING_MIN_OFFSET_FROM_CURB_M)
        ped_b_evade = shifted_sidewalk_point(ped_b_start, 12, 12, GATHERING_MIN_OFFSET_FROM_CURB_M)
        events = [
            event("forced_landing_fault", tick(230), [visual("set_uav_forced_landing_fault", uav, "propulsion_fault"), crowd("spawn_landing_crowd", f"crowd_{sid}", 8, crowd_origin, [650, 420, 0], 700 + idx)], 1, scenario_id, "UAV fault appears near a ground crowd", "uav_mission", [uav, ped_a, ped_b], "critical"),
            event("descent_to_low_altitude", fired("forced_landing_fault"), [move("move_uav_forced_descent", uav, [start, p(ox - 3, oy - 4, 18), low, land], 3.0)], 2, scenario_id, "UAV descends from 30m to low height", "uav_mission", [uav], "critical"),
            event("crowd_proximity_response", prox(uav, ped_a, 5.0, 2), [move("move_ped_a_evade_landing", ped_a, [ped_a_start, ped_a_evade], 2.0), move("move_ped_b_evade_landing", ped_b, [ped_b_start, ped_b_evade], 2.0), screenshot("capture_forced_landing")], 3, scenario_id, "Crowd evades forced landing zone", "pedestrian", [ped_a, ped_b, uav], "warning"),
            event("safe_landing", fired("crowd_proximity_response"), [move("move_uav_safe_touchdown", uav, [land, q(land[0] + 1, land[1] + 1, 0.8)], 1.0), clear_crowd("clear_landing_crowd", f"crowd_{sid}")], 4, scenario_id, "UAV completes safe landing", "uav_mission", [uav], "info"),
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
        start = p(ox - 9, oy - 8, 30)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", start, 45, "descent"))
        _, ped_scene = add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_21", 35, 1.2, p(ox + 2, oy + 1, 0), 15, "walking"))
        ped_start = scene_pos(ped_scene)
        over = q(ped_start[0], ped_start[1], 5)
        pull_up = q(ped_start[0] - 8, ped_start[1] + 8, 24)
        ped_clear = shifted_sidewalk_point(ped_start, 10, 10, 1.8)
        events = [
            event("uav_low_descent", tick(230), [move("move_uav_ped_descent", uav, [start, p(ox - 3, oy - 2, 15), over], 4.5)], 1, scenario_id, "UAV descends toward pedestrian head height", "uav_mission", [uav, ped], "warning"),
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
            event("ped_enters_roadway", tick(220), [move("move_ped_from_crosswalk_to_lane", ped, [p0, road], 1.6), move("move_car_toward_crosswalk", car, [car_start, car_conflict], 8.0)], 1, scenario_id, "Pedestrian moves from sidewalk into roadway", "pedestrian", [ped, car], "warning"),
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
        uav_start = p(ox - 14, oy - 8, 31)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 55, "patrol"))
        _, ambulance_scene = add(specs, scenes, lane_entity(ambulance, "vehicle.emergency.ambulance.v1", "vehicle", "cg_edge_24", 30, 0.0, p(ox - 32, oy - 18, 0), 65, "response", visual_state={"mode": "response", "lights_on": True}))
        ambulance_start = scene_pos(ambulance_scene)
        ambulance_arrival = road_center_point(ped_start)
        uav_observe = q(ped_start[0], ped_start[1], 16)
        events = [
            event("pedestrian_fall", tick(230), [play("play_pedestrian_fall", ped)], 1, scenario_id, "Pedestrian fall animation starts", "pedestrian", [ped], "critical"),
            event("uav_detects_fall", fired("pedestrian_fall"), [move("move_uav_detect_fall", uav, [uav_start, uav_observe], 5.0), screenshot("capture_fall_detection")], 2, scenario_id, "UAV detects fallen pedestrian", "uav_mission", [uav, ped], "warning"),
            event("ambulance_response", fired("uav_detects_fall"), [move("move_ambulance_to_fall", ambulance, [ambulance_start, ambulance_arrival], 12.0)], 3, scenario_id, "Ambulance responds to detected fall", "vehicle", [ambulance, ped], "warning"),
            event("incident_documented", fired("ambulance_response"), [screenshot("capture_fall_response")], 4, scenario_id, "UAV documents emergency response", "uav_mission", [uav, ambulance, ped], "info"),
        ]
        desc = "Pedestrian fall detected by UAV and response chain"
    elif scenario_id.startswith("L4-8_"):
        uav = f"uav_crowd_monitor_{sid}"
        group = f"crowd_{sid}"
        safe_group = f"crowd_safe_{sid}"
        uav_start = p(ox - 18, oy - 10, 34)
        crowd_extent = [900, 600, 0]
        safe_extent = [1000, 500, 0]
        crowd_origin = gathering_point(p(ox + 3, oy + 3, 0), crowd_extent)
        safe_origin = gathering_point(p(ox + 24, oy + 19, 0), safe_extent)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 50, "monitor"))
        add(specs, scenes, world_entity(f"crowd_anchor_{sid}", "semantic.spawn_zone", "crowd_anchor", crowd_origin, 0))
        events = [
            event("crowd_generated", tick(220), [crowd("spawn_incident_crowd", group, 12 if scenario_id.endswith("v1") else 16, crowd_origin, crowd_extent, 800 + idx)], 1, scenario_id, "Crowd generated in evacuation zone", "pedestrian", [f"crowd_anchor_{sid}"], "warning"),
            event("evacuation_triggered", fired("crowd_generated"), [clear_crowd("clear_incident_crowd_for_evacuation", group), crowd("spawn_crowd_at_safe_zone", safe_group, 12 if scenario_id.endswith("v1") else 16, safe_origin, safe_extent, 900 + idx)], 2, scenario_id, "Crowd moves to safe evacuation zone", "pedestrian", [f"crowd_anchor_{sid}"], "warning"),
            event("uav_evacuation_monitor", fired("evacuation_triggered"), [move("move_uav_monitor_evacuation", uav, [uav_start, q(crowd_origin[0], crowd_origin[1], 30), q(safe_origin[0], safe_origin[1], 30)], 5.0), screenshot("capture_crowd_evacuation")], 3, scenario_id, "UAV monitors evacuation movement", "uav_mission", [uav], "info"),
            event("crowd_clear", fired("uav_evacuation_monitor"), [clear_crowd("clear_safe_crowd", safe_group)], 4, scenario_id, "Crowd evacuation completes and group is cleared", "pedestrian", [uav], "info"),
        ]
        desc = "Crowd evacuation with UAV monitoring"
        rules.append({"rule": "crowd_spawn_count_min", "min_count": 8, "description": "Crowd evacuation scenario uses spawn_crowd with at least 8 people"})
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
        _, ped_scene = add(specs, scenes, sidewalk_entity(ped, "pedestrian.cityops.basic.v1", "pedestrian", "cg_edge_31", 42, GATHERING_MIN_OFFSET_FROM_CURB_M, p(ox + 16, oy + 9, 0), 0, "standing"))
        ped_start = scene_pos(ped_scene)
        crowd_origin = gathering_point(ped_start, [800, 500, 0])
        low = q(ped_start[0], ped_start[1], 8)
        landing = q(ped_start[0], ped_start[1], 3)
        ped_evade = shifted_sidewalk_point(ped_start, 12, 10, GATHERING_MIN_OFFSET_FROM_CURB_M)
        weather_profile = {"initial": "clear", "transitions": [{"tick": 180, "profile": "rain", "overrides": {"rain": 0.62, "visibility_m": 1600.0}}]}
        events = [
            event("rain_onset", tick(180), [set_weather("set_x1_rain_onset", "rain", rain=0.62, visibility_m=1600.0)], 1, scenario_id, "Rain onset is applied before C2 coupling", "weather", [uav, tower], "info"),
            event("rain_threshold", weather("rain", "gte", 0.5, 5), [set_weather("set_x1_rain", "rain", rain=0.65, visibility_m=1500.0)], 1, scenario_id, "Rain threshold triggers L5 degradation", "weather", [uav, tower], "warning"),
            event("c2_loss_tick", tick(260), [visual("set_x1_tower_stressed", tower, "rain_stressed")], 2, scenario_id, "C2 loss timing condition becomes true", "digital_layer", [tower], "warning"),
            event("c2_loss_after_rain", composite("AND", ["rain_threshold", "c2_loss_tick"]), [visual("set_x1_tower_c2_loss", tower, "link_lost"), move("move_x1_uav_degraded", uav, [start, p(ox + 9, oy + 4, 26)], 3.0)], 3, scenario_id, "Rain and C2 condition combine into C2 loss", "digital_layer", [uav, tower], "critical"),
            event("forced_landing_descent", fired("c2_loss_after_rain"), [move("move_x1_forced_landing", uav, [p(ox + 9, oy + 4, 26), low, landing], 2.5), crowd("spawn_x1_crowd", "crowd_x1", 10, crowd_origin, [800, 500, 0], 1101)], 4, scenario_id, "C2 loss causes forced landing near crowd", "uav_mission", [uav, ped], "critical"),
            event("crowd_reacts_to_landing", prox(uav, ped, 5.0, 2), [move("move_x1_ped_evade", ped, [ped_start, ped_evade], 2.0), clear_crowd("clear_x1_crowd", "crowd_x1"), screenshot("capture_x1_forced_landing")], 5, scenario_id, "Crowd reacts to forced landing", "pedestrian", [uav, ped], "warning"),
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
        uav_start = p(ox - 15, oy - 8, 32)
        add(specs, scenes, world_entity(uav, "uav.inspect.quad.v1", "uav", uav_start, 60, "patrol"))
        _, ambulance_scene = add(specs, scenes, lane_entity(ambulance, "vehicle.emergency.ambulance.v1", "vehicle", "cg_edge_33", 35, 0.0, p(ox - 32, oy - 15, 0), 70, "response", visual_state={"mode": "response", "lights_on": True}))
        ambulance_start = scene_pos(ambulance_scene)
        ambulance_arrival = road_center_point(ped_start)
        uav_observe = q(ped_start[0], ped_start[1], 16)
        _, tape_scene = add(specs, scenes, world_entity(tape, "prop.incident.police_tape.v1", "prop", shifted_sidewalk_point(ped_start, 1.5, 1.0, 1.5), 0, "staged", activation_tick=420))
        tape_pos = scene_pos(tape_scene)
        events = [
            event("pedestrian_fall", tick(220), [play("play_x3_ped_fall", ped)], 1, scenario_id, "Pedestrian falls", "pedestrian", [ped], "critical"),
            event("uav_detection", fired("pedestrian_fall"), [move("move_x3_uav_detect", uav, [uav_start, uav_observe], 5.0), screenshot("capture_x3_detection")], 2, scenario_id, "UAV detects pedestrian fall", "uav_mission", [uav, ped], "warning"),
            event("ambulance_dispatch", fired("uav_detection"), [move("move_x3_ambulance_dispatch", ambulance, [ambulance_start, ambulance_arrival], 12.0)], 3, scenario_id, "Ambulance is dispatched", "vehicle", [ambulance, ped], "warning"),
            event("isolation_setup", fired("ambulance_dispatch"), [spawn("spawn_x3_police_tape", tape, "prop.incident.police_tape.v1", tape_pos, 0)], 4, scenario_id, "Responder sets incident isolation tape", "dynamic_constraint", [tape, ped], "warning"),
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
            event("fog_uav_conflict", prox(uav_a, uav_b, 8.0, 2), [visual("set_x4_uav_a_hover", uav_a, "hover"), move("move_x4_uav_b_evasion", uav_b, [meet, p(ox + 14, oy + 18, 36)], 8.0), screenshot("capture_x4_conflict")], 4, scenario_id, "UAV proximity conflict under fog", "uav_mission", [uav_a, uav_b], "critical"),
            event("fog_conflict_recovered", fired("fog_uav_conflict"), [move("move_x4_uav_a_resume", uav_a, [meet, p(ox - 16, oy + 18, 30)], 5.5)], 5, scenario_id, "UAVs recover after fog conflict", "uav_mission", [uav_a, uav_b], "info"),
        ]
        desc = "Fog to multi-UAV conflict cross-layer chain"
    elif short_id == "X5":
        scenario_id = dirname
        tower, pad_a, pad_b = "tower_x5_comm", "pad_x5_primary", "pad_x5_backup"
        uav_a, uav_b = "uav_x5_priority", "uav_x5_second"
        add(specs, scenes, world_entity(tower, "facility.radio.base_tower.v1", "facility", p(ox + 5, oy + 1, 0), 0, "online"))
        add(specs, scenes, pad_entity(pad_a, p(ox + 2, oy + 4, 0), "pad_x5_a", "north"))
        add(specs, scenes, pad_entity(pad_b, p(ox + 26, oy + 8, 0), "pad_x5_b", "east"))
        add(specs, scenes, world_entity(uav_a, "uav.inspect.quad.v1", "uav", p(ox - 18, oy + 12, 33), 60, "landing_request"))
        add(specs, scenes, world_entity(uav_b, "uav.airsim.flying_pawn.v1", "uav", p(ox + 18, oy - 12, 31), 280, "landing_request"))
        events = [
            event("station_failure", tick(220), [visual("set_x5_tower_failed", tower, "failed")], 1, scenario_id, "Communication station failure", "infrastructure", [tower], "critical"),
            event("backup_pad_unavailable", fired("station_failure"), [visual("set_x5_backup_pad_unavailable", pad_b, "unavailable")], 2, scenario_id, "Only one landing pad remains available", "infrastructure", [pad_a, pad_b], "warning"),
            event("dual_uav_pad_contention", fired("backup_pad_unavailable"), [move("move_x5_uav_a_to_pad", uav_a, [p(ox - 18, oy + 12, 33), p(ox + 2, oy + 4, 6)], 5.0), move("move_x5_uav_b_to_pad_hold", uav_b, [p(ox + 18, oy - 12, 31), p(ox + 8, oy + 10, 34)], 5.0)], 3, scenario_id, "Two UAVs contend for one available pad", "uav_mission", [uav_a, uav_b, pad_a], "critical"),
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
        add(specs, scenes, box_entity(nfz, "trigger.no_fly.box.v1", "airspace_constraint", p(ox + 6, oy + 4, 28), [18, 13, 16], activation_tick=310))
        events = [
            event("crowd_evacuation", tick(220), [crowd("spawn_x6_crowd", "crowd_x6", 14, crowd_origin, crowd_extent, 1606)], 1, scenario_id, "Crowd evacuation begins", "pedestrian", [anchor], "warning"),
            event("crowd_safe_zone_move", fired("crowd_evacuation"), [clear_crowd("clear_x6_origin_crowd", "crowd_x6"), crowd("spawn_x6_safe_crowd", "crowd_x6_safe", 14, safe_origin, safe_extent, 1607)], 2, scenario_id, "Crowd moves to safe zone", "pedestrian", [anchor], "warning"),
            event("airspace_lockdown", fired("crowd_safe_zone_move"), [spawn("spawn_x6_nfz", nfz, "trigger.no_fly.box.v1", p(ox + 6, oy + 4, 28), 0, visual_state={"mode": "active"})], 3, scenario_id, "Airspace lockdown declares temporary NFZ", "dynamic_constraint", [nfz], "critical"),
            event("uav_reroute_lockdown", fired("airspace_lockdown"), [move("move_x6_uav_avoid_nfz", uav, [uav_start, p(ox - 5, oy + 14, 36), p(ox + 22, oy + 28, 36)], 7.0), screenshot("capture_x6_lockdown")], 4, scenario_id, "UAV reroutes around lockdown NFZ", "uav_mission", [uav, nfz], "warning"),
            event("lockdown_cleared", fired("uav_reroute_lockdown"), [clear_crowd("clear_x6_safe_crowd", "crowd_x6_safe"), visual("set_x6_nfz_standdown", nfz, "standdown")], 5, scenario_id, "Crowd evacuation and airspace lockdown clear", "dynamic_constraint", [uav, nfz], "info"),
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
    return TriggerSpec(**data)


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


def collect_bundles() -> list[ScenarioBundle]:
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
    bundles: list[ScenarioBundle] = []
    for i, scenario_id in enumerate(ids_l1):
        bundles.append(build_l1(scenario_id, i))
    for i, scenario_id in enumerate(ids_l2):
        bundles.append(build_l2(scenario_id, i))
    for i, scenario_id in enumerate(ids_l3):
        bundles.append(build_l3(scenario_id, i))
    for i, scenario_id in enumerate(ids_l4):
        bundles.append(build_l4(scenario_id, i))
    for i, scenario_id in enumerate(ids_l5):
        bundles.append(build_l5(scenario_id, i))
    for i, scenario_id in enumerate(ids_l6):
        bundles.append(build_l6(scenario_id, i))
    for i, (short_id, dirname) in enumerate(x_ids):
        bundles.append(build_x(short_id, dirname, i))
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    with open(ASSET_CATALOG, "r", encoding="utf-8") as f:
        catalog_ids = {a["logical_asset_id"] for a in json.load(f)["assets"]}
    bundles = collect_bundles()
    errors: list[str] = []
    for bundle in bundles:
        compiled = compile_bundle(bundle)
        errors.extend(validate_bundle(bundle, compiled, catalog_ids))
        bundle.directory.mkdir(parents=True, exist_ok=True)
        write_json(bundle.directory / "event_script.json", compiled)
        write_json(bundle.directory / "scene_setup.json", scene_setup_from_bundle(bundle))
        (bundle.directory / "spec.py").write_text(write_spec_py(bundle), encoding="utf-8")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Generated {len(bundles)} scenario specs, scene setups, and event scripts.")


if __name__ == "__main__":
    main()
