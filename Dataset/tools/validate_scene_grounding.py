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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "Dataset"
DEFAULT_ASSET_CATALOG = ROOT / "Config" / "LowAltitude" / "asset_catalog.json"
DEFAULT_TRAFFIC_BUNDLE = ROOT / "Config" / "LowAltitude" / "Maps" / "donghu_road_topo" / "traffic_bundle"
DEFAULT_BUILDING_GEOJSON = ROOT / "Content" / "Maps" / "donghu_road_topo" / "building" / "building.geojson"
LANE_HALF_WIDTH_M = 1.9
PEDESTRIAN_ROAD_BUFFER_M = 0.25
CROWD_ROAD_BUFFER_M = 0.5
POSITION_MATCH_TOLERANCE_M = 1.25
MOVE_START_TOLERANCE_GROUND_M = 8.0
MOVE_START_TOLERANCE_UAV_M = 18.0
ROI_MARGIN_M = 1000.0
MAX_WAYPOINT_SEGMENT_M = {
    "uav": 300.0,
    "vehicle": 80.0,
    "pedestrian": 20.0,
    "asset": 120.0,
}


@dataclass(frozen=True)
class LaneSample:
    edge_id: str
    lane_index: int
    s_m: float
    x_m: float
    y_m: float
    yaw_deg: float


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


def dist_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist3(a: list[float], b: list[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


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


def action_iter(script: dict[str, Any]):
    for event in script.get("events", []):
        for action in event.get("actions", []):
            yield event, action


def declared_ids(scene: dict[str, Any]) -> set[str]:
    return {entity["entity_id"] for entity in scene.get("entities", [])}


def is_ground_asset(asset_id: str) -> bool:
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


def check_assets(scene: dict[str, Any], known_assets: set[str], issues: list[str]) -> None:
    for entity in scene.get("entities", []):
        asset_id = entity.get("logical_asset_id")
        if asset_id not in known_assets:
            issues.append(f"{scene['scenario_id']}: unknown logical_asset_id {asset_id} on {entity.get('entity_id')}")


def check_placements(scene: dict[str, Any], lanes: LaneResolver, known_buildings: set[str], issues: list[str]) -> None:
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
        if asset_id.startswith("uav.") and position[2] < 20.0:
            issues.append(f"{scene['scenario_id']}: UAV {entity['entity_id']} starts below 20m: z={position[2]}")
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
            elif asset_id.startswith("vehicle.") and abs(lateral) > LANE_HALF_WIDTH_M:
                issues.append(f"{scene['scenario_id']}: vehicle {entity['entity_id']} lane_anchor is outside drivable lane")

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
            elif nearest_lane_clearance(lanes, start) <= LANE_HALF_WIDTH_M:
                issues.append(f"{scene['scenario_id']}: crosswalk start {entity['entity_id']} is in roadway instead of curb side")

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


def check_event_positions(scene: dict[str, Any], script: dict[str, Any], lanes: LaneResolver, issues: list[str]) -> None:
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    current = {entity_id: entity_pos(entity) for entity_id, entity in entities.items()}
    asset = {entity_id: entity.get("logical_asset_id", "") for entity_id, entity in entities.items()}

    for event, action in action_iter(script):
        action_type = action.get("type")
        entity_id = action.get("entity_id")
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
                if segment_m > max_segment:
                    issues.append(
                        f"{script['scenario_id']}: move_entity {action.get('action_id')} segment {index}->{index + 1} is {segment_m:.1f}m, exceeds {motion_class(asset[entity_id])} limit {max_segment:.1f}m"
                    )
            expected_start = current.get(entity_id)
            if expected_start:
                tolerance = MOVE_START_TOLERANCE_UAV_M if asset[entity_id].startswith("uav.") else MOVE_START_TOLERANCE_GROUND_M
                if dist3(waypoints[0], expected_start) > tolerance:
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
            if asset[entity_id].startswith("pedestrian."):
                crossing_context = "crosswalk" in action.get("action_id", "") or "roadway" in action.get("action_id", "") or "retreat" in action.get("action_id", "")
                for point in waypoints:
                    if abs(point[2]) > 0.75:
                        issues.append(f"{script['scenario_id']}: pedestrian waypoint z is not ground level in {action.get('action_id')}: {point}")
                    if not crossing_context and nearest_lane_clearance(lanes, point) < LANE_HALF_WIDTH_M + PEDESTRIAN_ROAD_BUFFER_M:
                        issues.append(f"{script['scenario_id']}: pedestrian non-crossing waypoint is inside roadway in {action.get('action_id')}: {point}")
            current[entity_id] = waypoints[-1]

        if action_type == "move_entity" and entity_id in entities:
            if entities[entity_id].get("category") in {"traffic_signal", "facility"} and asset[entity_id].startswith(("prop.traffic_control.signal_light", "facility.radio.base_tower")):
                issues.append(f"{script['scenario_id']}: non-movable infrastructure has move_entity action: {entity_id}")

        if action_type == "spawn_crowd":
            origin = pos3(action.get("spawn_origin_enu_m"))
            extent = action.get("spawn_box_extent_cm") or [0, 0, 0]
            check_roi_point(script["scenario_id"], f"spawn_crowd {action.get('group_id')} origin", origin, lanes, issues)
            if origin:
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
        cats = {category.get(a), category.get(b)}
        if "uav" in cats and ({"pedestrian", "vehicle", "uav"} & cats):
            metric = trigger.get("metric", "xy")
            if metric not in {"3d", "xy_plus_z"}:
                issues.append(f"{script['scenario_id']}: proximity trigger {trigger['trigger_id']} needs 3d/xy_plus_z metric")


def execute_validation_rules(scene: dict[str, Any], script: dict[str, Any], known_assets: set[str], issues: list[str]) -> None:
    entities = {entity["entity_id"]: entity for entity in scene.get("entities", [])}
    events = script.get("events", [])
    actions = [action for _, action in action_iter(script)]
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
            uavs = [entity for entity in entities.values() if entity.get("logical_asset_id", "").startswith("uav.")]
            for uav in uavs:
                uav_pos = entity_pos(uav)
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
            uav_positions = [entity_pos(entity) for entity in entities.values() if entity.get("logical_asset_id", "").startswith("uav.")]
            uav_positions = [item for item in uav_positions if item]
            ok = len(uav_positions) >= 2 and dist_xy(uav_positions[0], uav_positions[1]) >= float(rule.get("min_horizontal_distance_m", 0.0)) and abs(uav_positions[0][2] - uav_positions[1][2]) > 0.5
        elif name == "facade_approach_distance":
            facade_positions = [entity_pos(entity) for entity in entities.values() if entity.get("category") == "facade_anchor"]
            uav_waypoints = [point for action in actions if action.get("type") == "move_entity" for point in action.get("waypoints_enu_m", []) if action.get("entity_id", "").startswith("uav_")]
            ok = bool(facade_positions and uav_waypoints) and min(dist3(pos3(point), facade_positions[0]) for point in uav_waypoints if pos3(point)) <= float(rule.get("max_distance_m", 3.0)) + 2.0
        elif name == "forced_landing_descent_profile":
            uav_moves = [action for action in actions if action.get("type") == "move_entity" and "forced_descent" in action.get("action_id", "")]
            ok = any(min(point[2] for point in action.get("waypoints_enu_m", [])) <= 3.5 and max(point[2] for point in action.get("waypoints_enu_m", [])) >= 29.0 for action in uav_moves)
        elif name == "trajectory_intersection_required":
            uav_points = [pos3(point) for action in actions if action.get("type") == "move_entity" and str(action.get("entity_id", "")).startswith("uav_") for point in action.get("waypoints_enu_m", [])]
            vehicle_points = [pos3(point) for action in actions if action.get("type") == "move_entity" and str(action.get("entity_id", "")).startswith("car_") for point in action.get("waypoints_enu_m", [])]
            ok = any(up and vp and dist_xy(up, vp) <= 3.5 for up in uav_points for vp in vehicle_points)
        elif name == "crowd_spawn_count_min":
            ok = any(action.get("type") == "spawn_crowd" and int(action.get("count", 0)) >= int(rule.get("min_count", 0)) for action in actions)
        elif name == "ambulance_lights_on":
            target = entities.get(rule.get("entity_id"))
            ok = bool(target and target.get("initial_state", {}).get("lights_on") is True)
        elif name == "environment_trigger_kind":
            sid = script.get("scenario_id", "")
            has_weather = any(trigger.get("type") == "weather_state" for trigger in script.get("triggers", []))
            ok = has_weather if sid.startswith(("L5-1", "L5-2", "L5-3")) else not has_weather
        elif name == "digital_anomaly_chain":
            ok = any(action.get("type") == "move_entity" and str(action.get("entity_id", "")).startswith("uav_") for action in actions)
        else:
            ok = False
        if not ok:
            issues.append(f"{script['scenario_id']}: validation_rule failed or unsupported: {name}")


def validate_scenario(scene_path: Path, lanes: LaneResolver, known_assets: set[str], known_buildings: set[str]) -> list[str]:
    scene = load_json(scene_path)
    script_path = scene_path.with_name("event_script.json")
    if not script_path.exists():
        return [f"{scene_path.parent}: missing event_script.json"]
    script = load_json(script_path)
    issues: list[str] = []
    check_entity_references(scene, script, issues)
    check_assets(scene, known_assets, issues)
    check_placements(scene, lanes, known_buildings, issues)
    check_cameras(scene, lanes, issues)
    check_event_positions(scene, script, lanes, issues)
    check_weather_bootstrap(scene, script, issues)
    check_proximity_metrics(scene, script, issues)
    execute_validation_rules(scene, script, known_assets, issues)
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate grounded scene_setup/event_script pairs")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--asset-catalog", default=str(DEFAULT_ASSET_CATALOG))
    parser.add_argument("--traffic-bundle", default=str(DEFAULT_TRAFFIC_BUNDLE))
    parser.add_argument("--building-geojson", default=str(DEFAULT_BUILDING_GEOJSON))
    parser.add_argument("--limit", type=int, default=0, help="Stop after N scenario files for debugging")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    scenarios_root = dataset_root / "scenarios"
    traffic_bundle = Path(args.traffic_bundle).resolve()
    lanes = LaneResolver(traffic_bundle / "lane_center_samples.csv")
    known_assets = asset_ids(Path(args.asset_catalog).resolve())
    known_buildings = building_ids(Path(args.building_geojson).resolve())

    scene_paths = sorted(scenarios_root.rglob("scene_setup.json"))
    if args.limit:
        scene_paths = scene_paths[: args.limit]
    all_issues: list[str] = []
    for scene_path in scene_paths:
        all_issues.extend(validate_scenario(scene_path, lanes, known_assets, known_buildings))

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
