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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "Dataset"
DEFAULT_ASSET_CATALOG = ROOT / "Config" / "LowAltitude" / "asset_catalog.json"
DEFAULT_TRAFFIC_BUNDLE = ROOT / "Config" / "LowAltitude" / "Maps" / "donghu_road_topo" / "traffic_bundle"
DEFAULT_BUILDING_GEOJSON = ROOT / "Content" / "Maps" / "donghu_road_topo" / "building" / "building.geojson"
DEFAULT_RENDER_READY_ROOT = ROOT / "Dataset" / "render_ready_episodes"
LANE_HALF_WIDTH_M = 1.9
PEDESTRIAN_ROAD_BUFFER_M = 0.25
CROWD_ROAD_BUFFER_M = 0.5
POSITION_MATCH_TOLERANCE_M = 1.25
MOVE_START_TOLERANCE_GROUND_M = 8.0
MOVE_START_TOLERANCE_UAV_M = 18.0
UAV_INITIAL_ALTITUDE_MAX_M = 5.0
UAV_MISSION_ALTITUDE_MIN_M = 18.0
UAV_TERMINAL_ALTITUDE_MAX_M = 8.0
INITIAL_OVERLAP_MIN_M = {
    ("pedestrian", "pedestrian"): 0.75,
    ("vehicle", "vehicle"): 2.2,
    ("pedestrian", "vehicle"): 1.5,
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


def entity_mission_start_pos(entity: dict[str, Any]) -> list[float] | None:
    lifecycle = entity.get("lifecycle") or {}
    return pos3(lifecycle.get("mission_start_enu_m")) or entity_pos(entity)


def dist_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist3(a: list[float], b: list[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


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
        if str(asset_id).startswith("pedestrian."):
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
            lifecycle = entity.get("lifecycle") or {}
            mission_start = pos3(lifecycle.get("mission_start_enu_m"))
            if lifecycle:
                if position[2] > UAV_INITIAL_ALTITUDE_MAX_M:
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
        if required_ticks > delay_ticks:
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
    center = pos3(bounds.get("center_enu_m"))
    radius = float(bounds.get("radius_m", 0.0) or 0.0)
    if not center or radius <= 0.0:
        issues.append(f"{scene['scenario_id']}: scene_setup missing local_bounds.center_enu_m/radius_m")
        return
    for label, point in collect_scene_points(scene, script):
        distance = dist_xy(center, point)
        if distance > radius:
            issues.append(f"{scene['scenario_id']}: {label} is outside local_bounds radius {radius:.1f}m by {distance - radius:.1f}m")


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
        start = entity_pos(entity)
        lifecycle = entity.get("lifecycle") or {}
        pad_id = lifecycle.get("home_pad_entity_id")
        mission_start = pos3(lifecycle.get("mission_start_enu_m"))
        if not start:
            issues.append(f"{script['scenario_id']}: UAV {entity_id} lacks resolved initial position")
            continue
        if start[2] > UAV_INITIAL_ALTITUDE_MAX_M:
            issues.append(f"{script['scenario_id']}: UAV {entity_id} starts at {start[2]:.1f}m; expected visible low-altitude pad/preflight start")
        if not pad_id or pad_id not in entities:
            issues.append(f"{script['scenario_id']}: UAV {entity_id} lacks declared lifecycle home pad")
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
            if max(point[2] for point in first_waypoints) < UAV_MISSION_ALTITUDE_MIN_M:
                issues.append(f"{script['scenario_id']}: UAV {entity_id} takeoff never reaches mission altitude")

        last = moves[-1]
        last_waypoints = [pos3(point) for point in last.get("waypoints_enu_m", [])]
        last_waypoints = [point for point in last_waypoints if point]
        if last_waypoints:
            final_z = last_waypoints[-1][2]
            terminal_action = str(last.get("action_id", "")).lower()
            terminal_by_name = any(token in terminal_action for token in ("landing", "touchdown", "debris", "crash"))
            if final_z > UAV_TERMINAL_ALTITUDE_MAX_M and not terminal_by_name:
                issues.append(f"{script['scenario_id']}: UAV {entity_id} ends at {final_z:.1f}m without landing/crash terminal action")


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
            uav_positions = [entity_mission_start_pos(entity) for entity in entities.values() if entity.get("logical_asset_id", "").startswith("uav.")]
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
        else:
            ok = False
        if not ok:
            issues.append(f"{script['scenario_id']}: validation_rule failed or unsupported: {name}")


def check_render_ready_configs(render_ready_root: Path, issues: list[str]) -> None:
    if not render_ready_root.exists():
        return
    for config_path in sorted(render_ready_root.rglob("render_host_config.json")):
        config = load_json(config_path)
        projection = dict(config.get("pedestrian_roadside_projection") or {})
        if bool(projection.get("enabled", False)):
            issues.append(f"{config_path}: Dataset render config must disable pedestrian_roadside_projection.enabled")


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
    check_entity_references(scene, script, issues)
    check_assets(scene, known_assets, issues)
    check_dynamic_spawn_policy(scene, script, issues)
    check_spawn_schema(scene, script, known_assets, issues)
    check_placements(scene, lanes, spatial, known_buildings, issues)
    check_initial_mobile_overlaps(scene, issues)
    check_cameras(scene, lanes, issues)
    check_event_positions(scene, script, lanes, spatial, issues)
    check_event_delay_physics(script, issues)
    check_evacuation_pedestrian_spacing(scene, script, issues)
    check_local_bounds(scene, script, issues)
    check_weather_bootstrap(scene, script, issues)
    check_proximity_metrics(scene, script, issues)
    check_lifecycle_closure(scene, script, issues)
    execute_validation_rules(scene, script, known_assets, issues)
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate grounded scene_setup/event_script pairs")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--asset-catalog", default=str(DEFAULT_ASSET_CATALOG))
    parser.add_argument("--traffic-bundle", default=str(DEFAULT_TRAFFIC_BUNDLE))
    parser.add_argument("--building-geojson", default=str(DEFAULT_BUILDING_GEOJSON))
    parser.add_argument("--render-ready-root", default=str(DEFAULT_RENDER_READY_ROOT))
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
    for scene_path in scene_paths:
        all_issues.extend(validate_scenario(scene_path, lanes, spatial, known_assets, known_buildings))
    if not args.skip_render_ready_configs:
        check_render_ready_configs(Path(args.render_ready_root).resolve(), all_issues)

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
