"""Build SUMO-side incident anchors from existing episode event scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any, Iterable, Sequence

from .planner import SumoEdge, SumoGroundFlowPlanner


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIOS_ROOT = ROOT / "Dataset" / "scenarios"
DEFAULT_SUMO_NET_XML = ROOT / "Plugins" / "SumoImporter" / "Maps" / "donghu_road_topo" / "source" / "map.net.xml"
SCRIPT_TICK_HZ = 10.0


GROUND_TRAFFIC_EVENT_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("L2-5", "traffic_light_all_red_fault", ("signal_all_red_fault",)),
    ("L3-1", "lane_closure_roadwork", ("lane_closure_active",)),
    ("L3-3", "hazmat_isolation_zone", ("hazmat_leak",)),
    ("L4-4", "uav_vehicle_impact_brake", ("vehicle_roof_contact",)),
    ("L4-6", "ped_vehicle_brake", ("vehicle_ped_proximity",)),
    ("L4-9", "vehicle_intersection_conflict", ("vehicle_collision_warning",)),
    ("L4-10", "emergency_vehicle_priority", ("ambulance_priority_approach",)),
    ("L4-11", "av_sensor_fault_stop", ("av_safe_stop",)),
    ("L4-7", "medical_response_dispatch", ("ambulance_response",)),
    ("X3_pedestrian_fall_to_emergency_response", "medical_response_dispatch", ("ambulance_dispatch",)),
    ("L5-1", "weather_speed_degradation", ("rain_speed_reduction",)),
)


VEHICLE_ID_TOKENS = (
    "ambulance",
    "av_",
    "blocked_car",
    "car_",
    "civilian",
    "police",
    "vehicle",
)


@dataclass(frozen=True)
class LaneAnchor:
    truth_position_enu_m: list[float]
    requested_truth_position_enu_m: list[float]
    relocated_distance_m: float
    sumo_edge_id: str
    sumo_lane_id: str
    lane_index: int
    lane_position_m: float
    projected_truth_xy_m: list[float]
    projected_sumo_xy_m: list[float]
    edge_speed_mps: float
    vehicle_class: str
    geometry_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrafficLightAnchor:
    traffic_light_id: str
    truth_position_enu_m: list[float]
    sumo_xy_m: list[float]
    distance_to_incident_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SumoIncident:
    incident_id: str
    episode_scenario_id: str
    episode_event_id: str
    intent: str
    intent_stage: str
    causal_chain_id: str
    accident_class: str
    start_s: float
    end_s: float
    anchor: LaneAnchor
    injection_method: str
    affected_vehicle_ids: list[str] = field(default_factory=list)
    expected_observable_effect: str = ""
    source_event_title: str = ""
    source_event_category: str = ""
    traffic_light: TrafficLightAnchor | None = None
    episode_capture_boundary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["anchor"] = self.anchor.to_dict()
        payload["traffic_light"] = self.traffic_light.to_dict() if self.traffic_light else None
        return payload


def _norm_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value)).strip("_")


def _looks_like_vehicle_id(entity_id: str) -> bool:
    lowered = entity_id.lower()
    return any(token in lowered for token in VEHICLE_ID_TOKENS)


def vehicle_class_for_edge(edge: SumoEdge) -> str:
    if edge.allow:
        if "passenger" in edge.allow:
            return "passenger"
        if "delivery" in edge.allow:
            return "delivery"
        return ""
    blocked = {"passenger", "private"} & set(edge.disallow)
    forbidden_type = any(token in edge.edge_type for token in ("footway", "pedestrian", "path", "steps", "rail"))
    return "passenger" if not blocked and not forbidden_type else ""


def _distance_point_to_segment_xy(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float, tuple[float, float]]:
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay), 0.0, (ax, ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy), t, (qx, qy)


def _project_on_truth_edge(point_xy: Sequence[float], edge: SumoEdge) -> tuple[float, float, tuple[float, float]]:
    px = float(point_xy[0])
    py = float(point_xy[1])
    best_distance = float("inf")
    best_s = 0.0
    best_point = edge.shape_xy[0]
    cumulative = 0.0
    for a, b in zip(edge.shape_xy, edge.shape_xy[1:]):
        seg_len = math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        distance, t, projected = _distance_point_to_segment_xy(px, py, float(a[0]), float(a[1]), float(b[0]), float(b[1]))
        if distance < best_distance:
            best_distance = distance
            best_s = cumulative + t * seg_len
            best_point = projected
        cumulative += seg_len
    return best_distance, best_s, best_point


def _point_at_s(points_xy: Sequence[Sequence[float]], s_m: float) -> tuple[float, float]:
    if not points_xy:
        return 0.0, 0.0
    remaining = max(0.0, float(s_m))
    last = points_xy[0]
    for current in points_xy[1:]:
        seg_len = math.hypot(float(current[0]) - float(last[0]), float(current[1]) - float(last[1]))
        if seg_len > 1e-9 and remaining <= seg_len:
            t = remaining / seg_len
            return (
                float(last[0]) + (float(current[0]) - float(last[0])) * t,
                float(last[1]) + (float(current[1]) - float(last[1])) * t,
            )
        remaining -= seg_len
        last = current
    return float(points_xy[-1][0]), float(points_xy[-1][1])


def nearest_passenger_lane_anchor(
    planner: SumoGroundFlowPlanner,
    point_enu_m: Sequence[float],
    *,
    geometry_source: str,
) -> LaneAnchor:
    best: tuple[float, float, tuple[float, float], SumoEdge] | None = None
    for edge in planner.edges.values():
        vehicle_class = vehicle_class_for_edge(edge)
        if not vehicle_class or len(edge.shape_xy) < 2:
            continue
        distance, lane_s, projected_truth = _project_on_truth_edge(point_enu_m, edge)
        if best is None or distance < best[0]:
            best = (distance, lane_s, projected_truth, edge)
    if best is None:
        raise RuntimeError("No passenger-compatible SUMO lane found for incident anchor")
    distance, lane_s, projected_truth, edge = best
    sumo_edge = planner.coordinate_mapper.net.getEdge(edge.edge_id)
    lane_index = 0
    lane_shape = sumo_edge.getLanes()[lane_index].getShape()
    projected_sumo = _point_at_s(lane_shape, lane_s)
    requested_z = float(point_enu_m[2]) if len(point_enu_m) > 2 else 0.0
    return LaneAnchor(
        truth_position_enu_m=[round(float(projected_truth[0]), 6), round(float(projected_truth[1]), 6), requested_z],
        requested_truth_position_enu_m=[
            round(float(point_enu_m[0]), 6),
            round(float(point_enu_m[1]), 6),
            requested_z,
        ],
        relocated_distance_m=round(float(distance), 6),
        sumo_edge_id=edge.edge_id,
        sumo_lane_id=edge.lane_id,
        lane_index=lane_index,
        lane_position_m=round(float(lane_s), 6),
        projected_truth_xy_m=[round(float(projected_truth[0]), 6), round(float(projected_truth[1]), 6)],
        projected_sumo_xy_m=[round(float(projected_sumo[0]), 6), round(float(projected_sumo[1]), 6)],
        edge_speed_mps=round(float(edge.speed_mps), 6),
        vehicle_class=vehicle_class_for_edge(edge),
        geometry_source=geometry_source,
    )


def _load_traffic_light_anchors(net_xml: Path, planner: SumoGroundFlowPlanner) -> list[tuple[str, list[float], list[float]]]:
    anchors: list[tuple[str, list[float], list[float]]] = []
    for _event, elem in ET.iterparse(net_xml, events=("end",)):
        if elem.tag == "junction" and (elem.attrib.get("type") == "traffic_light" or elem.attrib.get("tl")):
            tls_id = str(elem.attrib.get("id") or elem.attrib.get("tl") or "")
            if tls_id:
                sx = float(elem.attrib.get("x") or 0.0)
                sy = float(elem.attrib.get("y") or 0.0)
                tx, ty = planner.coordinate_mapper.sumo_xy_to_truth_xy(sx, sy)
                anchors.append((tls_id, [round(tx, 6), round(ty, 6), 0.0], [round(sx, 6), round(sy, 6)]))
        elem.clear()
    return anchors


def nearest_traffic_light_anchor(
    traffic_lights: Sequence[tuple[str, list[float], list[float]]],
    point_enu_m: Sequence[float],
) -> TrafficLightAnchor | None:
    if not traffic_lights:
        return None
    px = float(point_enu_m[0])
    py = float(point_enu_m[1])
    tls_id, truth, sumo_xy = min(
        traffic_lights,
        key=lambda item: math.hypot(float(item[1][0]) - px, float(item[1][1]) - py),
    )
    return TrafficLightAnchor(
        traffic_light_id=tls_id,
        truth_position_enu_m=list(truth),
        sumo_xy_m=list(sumo_xy),
        distance_to_incident_m=round(math.hypot(float(truth[0]) - px, float(truth[1]) - py), 6),
    )


def _rule_for_event(scenario_id: str, event_id: str) -> tuple[str, str] | None:
    for scenario_prefix, accident_class, event_ids in GROUND_TRAFFIC_EVENT_RULES:
        if scenario_id.startswith(scenario_prefix) and event_id in event_ids:
            return accident_class, event_id
    return None


def _event_trigger_ticks(script: dict[str, Any]) -> dict[str, int]:
    triggers = {str(item.get("trigger_id") or ""): dict(item) for item in script.get("triggers") or []}
    events = list(script.get("events") or [])
    resolved: dict[str, int] = {}

    def resolve_event(event_def: dict[str, Any], previous_tick: int) -> int:
        trigger = triggers.get(str(event_def.get("trigger_ref") or ""), {})
        trigger_type = str(trigger.get("type") or "")
        if trigger_type == "tick":
            return int(trigger.get("tick", previous_tick))
        if trigger_type == "event_fired_after":
            prior = str(trigger.get("event_id") or trigger.get("event_ref") or "")
            return resolved.get(prior, previous_tick) + int(trigger.get("delay_ticks") or 0)
        if trigger_type == "event_fired":
            prior = str(trigger.get("event_id") or trigger.get("event_ref") or "")
            return resolved.get(prior, previous_tick)
        if trigger_type == "entity_proximity":
            return previous_tick + 25
        if trigger_type in {"weather_state", "composite"}:
            return previous_tick + 5
        return previous_tick

    previous = 0
    for event_def in events:
        tick = resolve_event(event_def, previous)
        event_id = str(event_def.get("event_id") or "")
        if event_id:
            resolved[event_id] = tick
        previous = tick
    return resolved


def _capture_boundary(script: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(script.get("parameters") or {})
    contract = dict(parameters.get("semantic_event_contract") or {})
    return dict(parameters.get("capture_boundary") or contract.get("capture_boundary") or {})


def _action_waypoints(event_def: dict[str, Any]) -> list[list[float]]:
    points: list[list[float]] = []
    for action in event_def.get("actions") or []:
        waypoints = action.get("waypoints_enu_m") or []
        for point in waypoints:
            if isinstance(point, list) and len(point) >= 2:
                points.append([float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else 0.0)])
    return points


def _affected_vehicle_ids(event_def: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    log_event = dict(event_def.get("log_event") or {})
    for target_id in list(log_event.get("target_ids") or []):
        if _looks_like_vehicle_id(str(target_id)):
            ids.append(_norm_id(str(target_id)))
    for action in event_def.get("actions") or []:
        entity_id = str(action.get("entity_id") or "")
        if entity_id and _looks_like_vehicle_id(entity_id):
            ids.append(_norm_id(entity_id))
    return list(dict.fromkeys(ids))


def _preferred_anchor_point(script: dict[str, Any], event_def: dict[str, Any]) -> tuple[list[float], str]:
    points = _action_waypoints(event_def)
    if points:
        return points[-1], f"event_action:{event_def.get('event_id')}"
    boundary = _capture_boundary(script)
    center = boundary.get("center_enu_m")
    if isinstance(center, list) and len(center) >= 2:
        return [float(center[0]), float(center[1]), float(center[2] if len(center) > 2 else 0.0)], "capture_boundary.center"
    return [0.0, 0.0, 0.0], "fallback.origin"


def _injection_method(accident_class: str) -> str:
    methods = {
        "traffic_light_all_red_fault": "trafficlight.setRedYellowGreenState",
        "lane_closure_roadwork": "lane.setMaxSpeed",
        "hazmat_isolation_zone": "edge.setMaxSpeed",
        "uav_vehicle_impact_brake": "vehicle.slowDown+vehicle.setStop+vehicle.setSignals",
        "ped_vehicle_brake": "vehicle.slowDown+vehicle.setStop+vehicle.setSignals",
        "vehicle_intersection_conflict": "vehicle.addFull+vehicle.slowDown+vehicle.setStop",
        "emergency_vehicle_priority": "vehicle.addFull+vehicle.setSignals+vehicle.slowDown",
        "av_sensor_fault_stop": "vehicle.slowDown+vehicle.setStop+vehicle.setSignals",
        "medical_response_dispatch": "vehicle.addFull+vehicle.setSignals",
        "weather_speed_degradation": "vehicle.slowDown+vehicle.setMaxSpeed",
    }
    return methods.get(accident_class, "traci")


def _expected_effect(accident_class: str) -> str:
    effects = {
        "traffic_light_all_red_fault": "traffic signal turns all-red and queues form",
        "lane_closure_roadwork": "lane speed is constrained and vehicles reroute or slow near roadwork",
        "hazmat_isolation_zone": "access around isolation zone is constrained",
        "uav_vehicle_impact_brake": "vehicle brakes hard at UAV impact point",
        "ped_vehicle_brake": "vehicle brakes for pedestrian conflict",
        "vehicle_intersection_conflict": "two vehicles enter conflict area and brake near same point",
        "emergency_vehicle_priority": "emergency vehicle passes while civilian traffic yields",
        "av_sensor_fault_stop": "autonomous vehicle enters safe stop and hazard hold",
        "medical_response_dispatch": "ambulance travels to medical incident anchor",
        "weather_speed_degradation": "vehicle speeds reduce under weather degradation",
    }
    return effects.get(accident_class, "SUMO traffic state changes at incident anchor")


def build_incident_plan(
    *,
    scenarios_root: Path = DEFAULT_SCENARIOS_ROOT,
    net_xml: Path = DEFAULT_SUMO_NET_XML,
    planner: SumoGroundFlowPlanner | None = None,
    duration_s: float = 270.0,
) -> dict[str, Any]:
    planner = planner or SumoGroundFlowPlanner(net_xml)
    traffic_lights = _load_traffic_light_anchors(net_xml, planner)
    incidents: list[SumoIncident] = []
    for script_path in sorted(Path(scenarios_root).rglob("event_script.json")):
        script = json.loads(script_path.read_text(encoding="utf-8-sig"))
        scenario_id = str(script.get("scenario_id") or script_path.parent.name)
        ticks_by_event = _event_trigger_ticks(script)
        boundary = _capture_boundary(script)
        for event_def in script.get("events") or []:
            event_id = str(event_def.get("event_id") or "")
            rule = _rule_for_event(scenario_id, event_id)
            if rule is None:
                continue
            accident_class, _ = rule
            point, geometry_source = _preferred_anchor_point(script, event_def)
            traffic_light = (
                nearest_traffic_light_anchor(traffic_lights, point)
                if accident_class == "traffic_light_all_red_fault"
                else None
            )
            if traffic_light is not None:
                point = list(traffic_light.truth_position_enu_m)
                geometry_source = f"sumo_traffic_light:{traffic_light.traffic_light_id}"
            anchor = nearest_passenger_lane_anchor(planner, point, geometry_source=geometry_source)
            tick = int(ticks_by_event.get(event_id, 0))
            start_s = round(max(0.0, tick / SCRIPT_TICK_HZ), 3)
            end_s = round(min(float(duration_s), start_s + 30.0), 3)
            log_event = dict(event_def.get("log_event") or {})
            incident_id = _norm_id(f"{scenario_id}.{event_id}.{accident_class}")
            incidents.append(
                SumoIncident(
                    incident_id=incident_id,
                    episode_scenario_id=scenario_id,
                    episode_event_id=event_id,
                    intent=str(event_def.get("intent") or ""),
                    intent_stage=str(event_def.get("intent_stage") or ""),
                    causal_chain_id=str(event_def.get("causal_chain_id") or log_event.get("causal_chain_id") or ""),
                    accident_class=accident_class,
                    start_s=start_s,
                    end_s=end_s,
                    anchor=anchor,
                    injection_method=_injection_method(accident_class),
                    affected_vehicle_ids=_affected_vehicle_ids(event_def),
                    expected_observable_effect=_expected_effect(accident_class),
                    source_event_title=str(log_event.get("title") or ""),
                    source_event_category=str(log_event.get("category") or ""),
                    traffic_light=traffic_light,
                    episode_capture_boundary=boundary,
                )
            )
    return {
        "schema_name": "sumo_incident_plan",
        "schema_version": "v1",
        "map_id": "donghu_road_topo",
        "duration_s": float(duration_s),
        "coordinate_route": "sumo_xy_to_lonlat_to_geojson_bundle_fit_to_ue_truth_xy",
        "coordinate_mapper": planner.coordinate_mapper.fit_summary,
        "incident_count": len(incidents),
        "incidents": [incident.to_dict() for incident in sorted(incidents, key=lambda item: (item.start_s, item.incident_id))],
    }


def write_incident_plan(plan: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_incident_plan(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def incident_records(plan: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for incident in plan.get("incidents") or []:
        if isinstance(incident, dict):
            yield incident
