"""Build per-episode SUMO semantic vehicle instruction plans.

This is a planning/QA bridge for the SUMO-only vehicle migration.  It does not
run SUMO and it does not create UE actors.  For every episode it translates
vehicle-related semantic events into explicit TraCI/SUMO-readable instructions
and writes reviewer artifacts:

* Dataset/episodes/<episode>/sumo_semantic_vehicle_plan.json
* G:/aw_cap/_meta/sumo_semantic_vehicle_review/<episode>/semantic_vehicle_actions.md
* G:/aw_cap/_meta/sumo_semantic_vehicle_review/<episode>/semantic_vehicle_actions.json
* optional ROI trajectory GIF for quick visual review
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EPISODES_ROOT = ROOT / "Dataset" / "episodes"
DEFAULT_RENDER_READY_ROOT = ROOT / "Dataset" / "render_ready_episodes_capture_filtered"
DEFAULT_REVIEW_ROOT = Path("G:/aw_cap/_meta/sumo_semantic_vehicle_review")
DEFAULT_SUMO_NET_XML = ROOT / "Plugins" / "SumoImporter" / "Maps" / "donghu_road_topo" / "source" / "road_derived" / "road_derived.net.xml"
DEFAULT_TICK_HZ = 10.0
BEFORE_AFTER_TICKS = 15
MAX_ROUTE_EDGES = 32
DEFAULT_VEHICLE_LENGTH_M = 5.0
FORCED_CONTROL_PROJECTION_TOLERANCE_M = 3.0
SEMANTIC_COLORS = (
    "#d62728",
    "#9467bd",
    "#ff7f0e",
    "#e377c2",
    "#8c564b",
    "#bcbd22",
)
BACKGROUND_COLOR = "#7f7f7f"
POLICE_COLOR = "#1f77b4"
MEDICAL_COLOR = "#2ca02c"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.sumo_ground_flow.planner import SumoEdge, SumoGroundFlowPlanner  # noqa: E402


@dataclass(frozen=True)
class ProjectedPoint:
    edge_id: str
    lane_id: str
    lane_index: int
    lane_position_m: float
    distance_m: float
    xy: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "lane_id": self.lane_id,
            "lane_index": self.lane_index,
            "lane_position_m": round(self.lane_position_m, 6),
            "distance_m": round(self.distance_m, 6),
            "xy_enu_m": [round(self.xy[0], 6), round(self.xy[1], 6)],
        }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def pos3(value: Any, default: Sequence[float] = (0.0, 0.0, 0.0)) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raw = list(default)
    return [
        float(raw[0] if len(raw) > 0 else default[0]),
        float(raw[1] if len(raw) > 1 else default[1]),
        float(raw[2] if len(raw) > 2 else default[2]),
    ]


def distance_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(distance_xy(a, b) for a, b in zip(points, points[1:]))


def compact_polyline(points: Sequence[Sequence[float]]) -> list[list[float]]:
    compact: list[list[float]] = []
    for point in points:
        current = pos3(point)
        if not compact or distance_xy(compact[-1], current) > 0.01 or abs(float(compact[-1][2]) - float(current[2])) > 0.01:
            compact.append(current)
    return compact


def point_at_distance(points: Sequence[Sequence[float]], distance_m: float) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0]
    remaining = max(0.0, float(distance_m))
    last = pos3(points[0])
    for current_raw in points[1:]:
        current = pos3(current_raw)
        segment_len = distance_xy(last, current)
        if segment_len > 1e-9 and remaining <= segment_len:
            ratio = remaining / segment_len
            return [
                last[0] + (current[0] - last[0]) * ratio,
                last[1] + (current[1] - last[1]) * ratio,
                last[2] + (current[2] - last[2]) * ratio,
            ]
        remaining -= segment_len
        last = current
    return pos3(points[-1])


def heading_deg_between(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def truth_yaw_to_sumo_angle(yaw_deg: float) -> float:
    angle = 90.0 - float(yaw_deg)
    while angle < 0.0:
        angle += 360.0
    while angle >= 360.0:
        angle -= 360.0
    return angle


def front_bumper_xy_for_center(center: Sequence[float], yaw_deg: float) -> tuple[float, float]:
    yaw_rad = math.radians(float(yaw_deg))
    half_length = 0.5 * DEFAULT_VEHICLE_LENGTH_M
    return (
        float(center[0]) + math.cos(yaw_rad) * half_length,
        float(center[1]) + math.sin(yaw_rad) * half_length,
    )


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def vehicle_category(entity: dict[str, Any]) -> str:
    return normalize_text(entity.get("category") or entity.get("entity_category") or entity.get("label_class"))


def is_vehicle_entity(entity: dict[str, Any]) -> bool:
    return vehicle_category(entity) == "vehicle"


def is_uav_entity(entity: dict[str, Any]) -> bool:
    category = normalize_text(entity.get("category") or entity.get("entity_category") or entity.get("label_class"))
    return category == "uav"


def event_topic(event: dict[str, Any], scenario_id: str) -> str:
    log_event = dict(event.get("log_event") or {})
    topic = str(log_event.get("topic") or event.get("topic") or event.get("event_id") or "")
    if topic.startswith("evt_"):
        return topic
    return f"evt_{scenario_id}_{topic}"


def event_targets(event: dict[str, Any], trigger: dict[str, Any] | None = None) -> list[str]:
    result: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)

    log_event = dict(event.get("log_event") or {})
    for value in log_event.get("target_ids") or []:
        add(value)
    for action in event.get("actions") or []:
        add(action.get("entity_id") or action.get("ped_id"))
    if trigger:
        add(trigger.get("entity_a"))
        add(trigger.get("entity_b"))
    return result


def action_entity_id(action: dict[str, Any]) -> str:
    return str(action.get("entity_id") or action.get("ped_id") or "").strip()


def event_title(event: dict[str, Any]) -> str:
    log_event = dict(event.get("log_event") or {})
    return str(log_event.get("title") or event.get("title") or event.get("event_id") or "")


def vehicle_role(entity_id: str, entity: dict[str, Any], event_modes: Sequence[str]) -> dict[str, str]:
    text = normalize_text(" ".join([entity_id, str(entity.get("logical_asset_id") or ""), " ".join(event_modes)]))
    if "police" in text:
        return {"type_id": "aero_police", "vehicle_class": "emergency", "role": "police"}
    if "ambulance" in text or "medical" in text or "emergency" in text:
        return {"type_id": "aero_ambulance", "vehicle_class": "emergency", "role": "medical_response"}
    if "delivery" in text:
        return {"type_id": "aero_delivery", "vehicle_class": "delivery", "role": "delivery"}
    return {"type_id": "aero_passenger", "vehicle_class": "passenger", "role": "civilian"}


def visual_mode(action: dict[str, Any]) -> str:
    return str((action.get("visual_state") or {}).get("mode") or action.get("mode") or "").strip()


def action_waypoints(action: dict[str, Any]) -> list[list[float]]:
    return [pos3(point) for point in action.get("waypoints_enu_m") or action.get("waypoints") or []]


def instruction(
    *,
    tick: int,
    instruction_type: str,
    traci_call: str,
    args: dict[str, Any],
    event_id: str = "",
    intent: str = "",
    target_role: str = "",
    reason: str = "",
    time_axis: str = "capture",
) -> dict[str, Any]:
    return {
        "tick": int(tick),
        "time_s": round(float(tick) / DEFAULT_TICK_HZ, 3),
        "time_axis": str(time_axis),
        "instruction_type": instruction_type,
        "traci_call": traci_call,
        "args": args,
        "event_id": event_id,
        "intent": intent,
        "target_role": target_role,
        "reason": reason,
    }


class RouteProjector:
    def __init__(self, planner: SumoGroundFlowPlanner) -> None:
        self.planner = planner

    def _edge_allows_vehicle_class(self, edge: SumoEdge, vehicle_class: str) -> bool:
        if edge.allow:
            return vehicle_class in edge.allow or ("emergency" in edge.allow and vehicle_class == "emergency")
        forbidden = bool(edge.disallow & {vehicle_class, "passenger", "delivery", "emergency"})
        return not forbidden and "footway" not in edge.edge_type and "steps" not in edge.edge_type

    @staticmethod
    def _project_to_shape(point: Sequence[float], shape: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
        px = float(point[0])
        py = float(point[1])
        best_distance = float("inf")
        best_xy = shape[0]
        best_s = 0.0
        accumulated = 0.0
        for a, b in zip(shape, shape[1:]):
            ax, ay = a
            bx, by = b
            dx = bx - ax
            dy = by - ay
            segment_len = math.hypot(dx, dy)
            if segment_len <= 1e-9:
                continue
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (segment_len * segment_len)))
            qx = ax + t * dx
            qy = ay + t * dy
            distance = math.hypot(px - qx, py - qy)
            if distance < best_distance:
                best_distance = distance
                best_xy = (qx, qy)
                best_s = accumulated + t * segment_len
            accumulated += segment_len
        return best_distance, best_xy[0], best_xy[1], best_s

    def project(self, point: Sequence[float], vehicle_class: str) -> ProjectedPoint | None:
        best: tuple[float, SumoEdge, float, float, float] | None = None
        for edge in self.planner.edges.values():
            if not self._edge_allows_vehicle_class(edge, vehicle_class):
                continue
            if len(edge.shape_xy) < 2:
                continue
            distance, qx, qy, lane_s = self._project_to_shape(point, edge.shape_xy)
            if best is None or distance < best[0]:
                best = (distance, edge, qx, qy, lane_s)
        if best is None:
            return None
        distance, edge, qx, qy, lane_s = best
        return ProjectedPoint(
            edge_id=edge.edge_id,
            lane_id=edge.lane_id,
            lane_index=0,
            lane_position_m=max(0.0, min(float(edge.length_m), float(lane_s))),
            distance_m=float(distance),
            xy=(float(qx), float(qy)),
        )

    def edge_path(self, start_edge: str, target_edge: str, vehicle_class: str) -> list[str]:
        if start_edge == target_edge:
            return [start_edge]
        queue: deque[list[str]] = deque([[start_edge]])
        visited = {start_edge}
        best = [start_edge]
        while queue:
            path = queue.popleft()
            if len(path) > len(best):
                best = path
            if len(path) >= MAX_ROUTE_EDGES:
                continue
            for next_edge_id in self.planner.adjacency.get(path[-1], []):
                if next_edge_id in visited:
                    continue
                edge = self.planner.edges.get(next_edge_id)
                if edge is None or not self._edge_allows_vehicle_class(edge, vehicle_class):
                    continue
                next_path = [*path, next_edge_id]
                if next_edge_id == target_edge:
                    return next_path
                visited.add(next_edge_id)
                queue.append(next_path)
        return best


def load_trace_by_topic(episode_dir: Path) -> dict[str, dict[str, Any]]:
    path = episode_dir / "event_trace.jsonl"
    if not path.exists():
        return {}
    return {str(row.get("topic") or row.get("source_event_id") or row.get("event_id") or ""): row for row in iter_jsonl(path)}


def load_samples_by_entity(episode_dir: Path, vehicle_ids: set[str]) -> dict[str, dict[int, dict[str, Any]]]:
    samples: dict[str, dict[int, dict[str, Any]]] = {entity_id: {} for entity_id in vehicle_ids}
    path = episode_dir / "trajectories.jsonl"
    if not path.exists():
        return samples
    for row in iter_jsonl(path):
        entity_id = str(row.get("entity_id") or "")
        if entity_id in samples:
            samples[entity_id][int(row.get("tick") or 0)] = row
    return samples


def sample_position(samples: dict[int, dict[str, Any]], tick: int, fallback: Sequence[float]) -> list[float]:
    if tick in samples:
        return pos3(samples[tick].get("pos_enu"), fallback)
    before = [value for value in samples if value <= tick]
    if before:
        return pos3(samples[max(before)].get("pos_enu"), fallback)
    after = [value for value in samples if value >= tick]
    if after:
        return pos3(samples[min(after)].get("pos_enu"), fallback)
    return pos3(fallback)


def event_references_entity(event: dict[str, Any], trigger: dict[str, Any] | None, entity_id: str) -> bool:
    if any(action_entity_id(action) == entity_id for action in event.get("actions") or []):
        return True
    return entity_id in set(event_targets(event, trigger))


def next_vehicle_event_tick(
    events: Sequence[dict[str, Any]],
    trace_by_event: dict[str, dict[str, Any]],
    triggers: dict[str, dict[str, Any]],
    entity_id: str,
    current_tick: int,
) -> int | None:
    candidates: list[int] = []
    for event in events:
        trigger = triggers.get(str(event.get("trigger_ref") or ""))
        if not event_references_entity(event, trigger, entity_id):
            continue
        row = trace_by_event.get(str(event.get("event_id") or ""))
        if not row:
            continue
        tick = int(row.get("tick") if row.get("tick") is not None else row.get("activated_tick", -1))
        if tick > current_tick:
            candidates.append(tick)
    return min(candidates) if candidates else None


def stop_required(event: dict[str, Any], action: dict[str, Any]) -> bool:
    text = normalize_text(" ".join([str(event.get("intent") or ""), event_title(event), visual_mode(action), str(action.get("post_activity_type") or "")]))
    return any(token in text for token in ("stop", "brake", "yield", "queue", "wait", "hold", "manual", "blocked", "conflict"))


def build_vehicle_base_instructions(
    *,
    episode_id: str,
    entity_id: str,
    entity: dict[str, Any],
    role: dict[str, str],
    projector: RouteProjector,
    referenced: bool,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    issues: list[str] = []
    placement = dict(entity.get("placement") or {})
    start = pos3(placement.get("resolved_position_enu_m") or placement.get("position_enu_m"))
    source_activation_tick = int(entity.get("activation_tick") or 0)
    projection = projector.project(start, role["vehicle_class"])
    if projection is None:
        issues.append(f"{entity_id}: no SUMO projection for depart position")
        projection_payload: dict[str, Any] = {}
        route_edges = []
    else:
        projection_payload = projection.as_dict()
        route_edges = [projection.edge_id]
    depart_lane = str(int(projection_payload.get("lane_index", 0) or 0)) if projection_payload else "best"
    route_id = f"{episode_id}.{entity_id}.depart_route"
    metadata = {
        "semantic_event_actor": bool(referenced),
        "scenario_vehicle_actor": True,
        "vehicle_role": role["role"],
        "source_entity_id": entity_id,
        "logical_asset_id": str(entity.get("logical_asset_id") or ""),
    }
    instructions = [
        instruction(
            tick=0,
            instruction_type="route.add",
            traci_call="traci.route.add",
            args={"route_id": route_id, "edges": route_edges},
            reason="initial SUMO route for the scenario vehicle; scripted event movement uses SUMO moveToXY_series when needed",
            time_axis="warmup",
        ),
        instruction(
            tick=0,
            instruction_type="vehicle.addFull",
            traci_call="traci.vehicle.addFull",
            args={
                "vehID": entity_id,
                "routeID": route_id,
                "typeID": role["type_id"],
                "depart": "0.000",
                "departLane": depart_lane,
                "departPos": projection_payload.get("lane_position_m", 0.0),
                "departSpeed": "0" if referenced else "max",
                "metadata": {**metadata, "source_activation_tick": source_activation_tick},
            },
            reason="all scenario vehicles enter SUMO at warm-up start; source activation is metadata, not insertion time",
            time_axis="warmup",
        ),
        instruction(
            tick=0,
            instruction_type="vehicle.setParameter",
            traci_call="traci.vehicle.setParameter",
            args={
                "vehID": entity_id,
                "key": "aero.semantic_metadata",
                "value": json.dumps({**metadata, "source_activation_tick": source_activation_tick}, ensure_ascii=False, sort_keys=True),
            },
            reason="metadata survives SUMO trajectory export and render-ready conversion",
            time_axis="warmup",
        ),
    ]
    return instructions, issues, {"depart_position_enu_m": start, "depart_projection": projection_payload}


def build_move_instructions(
    *,
    episode_id: str,
    scenario_id: str,
    event: dict[str, Any],
    event_tick: int,
    entity_id: str,
    action: dict[str, Any],
    role: dict[str, str],
    projector: RouteProjector,
    samples: dict[int, dict[str, Any]],
    next_tick: int | None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    issues: list[str] = []
    waypoints = action_waypoints(action)
    fallback_start = waypoints[0] if waypoints else [0.0, 0.0, 0.0]
    start = sample_position(samples, event_tick, fallback_start)
    target = waypoints[-1] if waypoints else start
    start_projection = projector.project(start, role["vehicle_class"])
    target_projection = projector.project(target, role["vehicle_class"])
    if start_projection is None or target_projection is None:
        issues.append(f"{entity_id}: move cannot project start/target to SUMO edges")
        route_edges: list[str] = []
    else:
        route_edges = projector.edge_path(start_projection.edge_id, target_projection.edge_id, role["vehicle_class"])
    control_polyline = compact_polyline([start, *waypoints]) if waypoints else compact_polyline([start])
    raw_length = path_length(control_polyline)
    requested_speed = float(action.get("velocity_mps") or 0.0)
    if requested_speed <= 0.0:
        duration_ticks = max(1, int((next_tick or event_tick + 30) - event_tick))
        requested_speed = raw_length / max(0.1, duration_ticks / DEFAULT_TICK_HZ) if raw_length > 0.0 else 0.0
    duration_ticks = max(1, int(math.ceil(raw_length / max(0.1, requested_speed) * DEFAULT_TICK_HZ))) if raw_length > 0.0 else 1
    if next_tick is not None:
        duration_ticks = min(duration_ticks, max(1, next_tick - event_tick))
    forced_control_reason = ""
    if start_projection is not None and target_projection is not None:
        if float(start_projection.distance_m) > FORCED_CONTROL_PROJECTION_TOLERANCE_M:
            forced_control_reason = f"start projection is {start_projection.distance_m:.2f}m from the scripted action point"
        elif float(target_projection.distance_m) > FORCED_CONTROL_PROJECTION_TOLERANCE_M:
            forced_control_reason = f"target projection is {target_projection.distance_m:.2f}m from the scripted action point"
        elif start_projection.edge_id == target_projection.edge_id and target_projection.lane_position_m + 0.5 < start_projection.lane_position_m:
            forced_control_reason = "scripted motion runs opposite the projected SUMO lane direction"
    else:
        forced_control_reason = "start or target projection is unavailable"
    if stop_required(event, action):
        forced_control_reason = forced_control_reason or "scripted stop/queue/yield/blocking event requires exact action-window endpoint hold"
    forced_control_reason = forced_control_reason or "explicit move_entity action is replayed as SUMO-layer forced_xy to preserve scripted event timing and spatial continuity"

    event_payload = {
        "scenario_id": scenario_id,
        "event_id": str(event.get("event_id") or ""),
        "intent": str(event.get("intent") or ""),
        "title": event_title(event),
    }
    instructions = [
        instruction(
            tick=event_tick,
            instruction_type="vehicle.setParameter",
            traci_call="traci.vehicle.setParameter",
            args={"vehID": entity_id, "key": "aero.active_semantic_event", "value": json.dumps(event_payload, ensure_ascii=False, sort_keys=True)},
            event_id=str(event.get("event_id") or ""),
            intent=str(event.get("intent") or ""),
            target_role=role["role"],
            reason="bind SUMO trajectory sample back to semantic event",
        )
    ]
    sample_end_tick = int(next_tick if next_tick is not None else event_tick + duration_ticks)
    if stop_required(event, action):
        sample_end_tick = max(sample_end_tick, int(next_tick if next_tick is not None else 900))
    samples_payload: list[dict[str, Any]] = []
    last_pos = control_polyline[0]
    start_anchor = start_projection or target_projection
    target_anchor = target_projection or start_projection
    for sample_tick in range(event_tick, max(event_tick, sample_end_tick) + 1):
        dt = max(0, sample_tick - event_tick)
        distance = min(raw_length, max(0.0, requested_speed) * dt / DEFAULT_TICK_HZ)
        current = point_at_distance(control_polyline, distance)
        next_point = point_at_distance(control_polyline, min(raw_length, distance + max(0.1, requested_speed) / DEFAULT_TICK_HZ))
        yaw = heading_deg_between(current, next_point) if distance < raw_length else heading_deg_between(last_pos, current)
        last_pos = current
        front_x, front_y = front_bumper_xy_for_center(current, yaw)
        projection = target_anchor if raw_length > 0.0 and distance >= raw_length * 0.5 else start_anchor
        samples_payload.append(
            {
                "tick": sample_tick,
                "time_s": round(sample_tick / DEFAULT_TICK_HZ, 3),
                "x": round(float(front_x), 6),
                "y": round(float(front_y), 6),
                "target_center_enu_m": [round(float(current[0]), 6), round(float(current[1]), 6), round(float(current[2]), 6)],
                "edgeID": projection.edge_id if projection else (start_projection.edge_id if start_projection else ""),
                "laneIndex": projection.lane_index if projection else 0,
                "angle": round(truth_yaw_to_sumo_angle(yaw), 6),
                "keepRoute": 2,
                "matchThreshold": 100.0,
            }
        )
    instructions.insert(
        0,
        instruction(
            tick=event_tick,
            instruction_type="vehicle.moveToXY_series",
            traci_call="traci.vehicle.moveToXY",
            args={
                "vehID": entity_id,
                "control_mode": "forced_xy",
                "reason": forced_control_reason,
                "samples": samples_payload,
            },
            event_id=str(event.get("event_id") or ""),
            intent=str(event.get("intent") or ""),
            target_role=role["role"],
            reason="scripted event geometry is replayed by SUMO TraCI moveToXY; no UE vehicle post-insertion or render-side override is used",
        ),
    )
    control_mode = "forced_xy"
    details = {
        "start_enu_m": [round(value, 6) for value in start],
        "target_enu_m": [round(value, 6) for value in target],
        "start_projection": start_projection.as_dict() if start_projection else {},
        "target_projection": target_projection.as_dict() if target_projection else {},
        "route_edges": route_edges,
        "control_mode": control_mode,
        "forced_control_reason": forced_control_reason,
        "path_length_m": round(raw_length, 6),
    }
    return instructions, issues, details


def build_visual_state_instructions(event: dict[str, Any], event_tick: int, entity_id: str, action: dict[str, Any], role: dict[str, str]) -> list[dict[str, Any]]:
    mode = visual_mode(action)
    payload = {
        "mode": mode,
        "visual_state": dict(action.get("visual_state") or {}),
        "event_id": str(event.get("event_id") or ""),
        "intent": str(event.get("intent") or ""),
    }
    instructions = [
        instruction(
            tick=event_tick,
            instruction_type="vehicle.setParameter",
            traci_call="traci.vehicle.setParameter",
            args={"vehID": entity_id, "key": "aero.semantic_vehicle_state", "value": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            event_id=str(event.get("event_id") or ""),
            intent=str(event.get("intent") or ""),
            target_role=role["role"],
            reason="vehicle visual state becomes SUMO/export metadata consumed by render-ready",
        )
    ]
    if bool((action.get("visual_state") or {}).get("lights_on")) or role["vehicle_class"] == "emergency":
        instructions.append(
            instruction(
                tick=event_tick,
                instruction_type="vehicle.setSignals",
                traci_call="traci.vehicle.setSignals",
                args={"vehID": entity_id, "signals": "emergency_or_warning"},
                event_id=str(event.get("event_id") or ""),
                intent=str(event.get("intent") or ""),
                target_role=role["role"],
                reason="police/medical/emergency visual cue represented in SUMO vehicle signal state",
            )
        )
    return instructions


def build_vehicle_hold_instructions(
    *,
    event: dict[str, Any],
    event_tick: int,
    entity_id: str,
    role: dict[str, str],
    projector: RouteProjector,
    samples: dict[int, dict[str, Any]],
    end_tick: int,
    reason: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    position = sample_position(samples, event_tick, [0.0, 0.0, 0.0])
    projection = projector.project(position, role["vehicle_class"])
    if projection is None:
        return [], [f"{entity_id}: hold cannot project event position to SUMO edge"]
    next_position = sample_position(samples, min(int(end_tick), int(event_tick) + 1), position)
    yaw = heading_deg_between(position, next_position)
    if path_length([position, next_position]) <= 1e-6:
        yaw = 0.0
    front_x, front_y = front_bumper_xy_for_center(position, yaw)
    samples_payload = [
        {
            "tick": tick,
            "time_s": round(tick / DEFAULT_TICK_HZ, 3),
            "x": round(float(front_x), 6),
            "y": round(float(front_y), 6),
            "target_center_enu_m": [round(float(position[0]), 6), round(float(position[1]), 6), round(float(position[2]), 6)],
            "edgeID": projection.edge_id,
            "laneIndex": int(projection.lane_index),
            "angle": round(truth_yaw_to_sumo_angle(yaw), 6),
            "keepRoute": 2,
            "matchThreshold": 100.0,
        }
        for tick in range(int(event_tick), max(int(event_tick), int(end_tick)) + 1)
    ]
    return [
        instruction(
            tick=event_tick,
            instruction_type="vehicle.moveToXY_series",
            traci_call="traci.vehicle.moveToXY",
            args={
                "vehID": entity_id,
                "control_mode": "forced_xy",
                "reason": reason,
                "samples": samples_payload,
            },
            event_id=str(event.get("event_id") or ""),
            intent=str(event.get("intent") or ""),
            target_role=role["role"],
            reason="hold SUMO semantic vehicle through the event evidence window so render truth can show the referenced vehicle",
        )
    ], []


def is_observation_only_vehicle_event(event: dict[str, Any], vehicle_targets: Sequence[str]) -> bool:
    """Events that document an existing vehicle state still need SUMO metadata."""
    actions = list(event.get("actions") or [])
    vehicle_target_set = {str(target) for target in vehicle_targets}
    if any(action_entity_id(action) in vehicle_target_set for action in actions):
        return False
    action_types = {str(action.get("type") or "").strip() for action in actions}
    text = normalize_text(" ".join([str(event.get("intent") or ""), event_title(event), str(event.get("event_id") or "")]))
    return bool(action_types & {"capture_screenshot", "log_event", "document_event", "set_weather"}) or any(
        token in text
        for token in (
            "documented",
            "observed",
            "confirmed",
            "handoff",
            "restore",
            "resolution",
            "report",
            "monitor",
            "weather",
            "rain",
            "visibility",
        )
    )


def build_observation_event_instructions(
    *,
    scenario_id: str,
    event: dict[str, Any],
    event_tick: int,
    vehicle_targets: Sequence[str],
    roles: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    payload = {
        "scenario_id": scenario_id,
        "event_id": str(event.get("event_id") or ""),
        "intent": str(event.get("intent") or ""),
        "title": event_title(event),
        "observation_only": True,
    }
    instructions: list[dict[str, Any]] = []
    for entity_id in vehicle_targets:
        role = roles.get(entity_id, {"role": ""})
        instructions.append(
            instruction(
                tick=event_tick,
                instruction_type="vehicle.setParameter",
                traci_call="traci.vehicle.setParameter",
                args={"vehID": entity_id, "key": "aero.active_semantic_event", "value": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
                event_id=str(event.get("event_id") or ""),
                intent=str(event.get("intent") or ""),
                target_role=str(role.get("role") or ""),
                reason="observation/documentation event binds an existing SUMO vehicle state; it does not require a new movement",
            )
        )
    return instructions


def resolve_manifest_path(value: Any, episode_dir: Path) -> Path:
    raw = str(value or "")
    path = Path(raw)
    if path.is_absolute():
        return path
    for candidate in ((ROOT / path).resolve(), (episode_dir / path).resolve()):
        if candidate.exists():
            return candidate
    return (ROOT / path).resolve()


def build_episode_plan(episode_dir: Path, projector: RouteProjector) -> dict[str, Any]:
    manifest = read_json(episode_dir / "episode_manifest.json")
    episode_id = str(manifest.get("episode_id") or episode_dir.name)
    scenario_id = str(manifest.get("scenario_id") or episode_id.split("__seed", 1)[0])
    script_path = resolve_manifest_path(manifest.get("source_event_script_path"), episode_dir)
    scene_path = resolve_manifest_path(manifest.get("source_scene_setup_path") or script_path.with_name("scene_setup.json"), episode_dir)
    script = read_json(script_path)
    scene = read_json(scene_path) if scene_path.exists() else {}
    triggers = {str(trigger.get("trigger_id") or ""): trigger for trigger in script.get("triggers") or []}
    events = list(script.get("events") or [])
    trace_by_topic = load_trace_by_topic(episode_dir)
    trace_by_event: dict[str, dict[str, Any]] = {}
    for event in events:
        topic = event_topic(event, scenario_id)
        row = trace_by_topic.get(topic) or trace_by_topic.get(str(event.get("event_id") or ""))
        if row:
            trace_by_event[str(event.get("event_id") or "")] = row

    scene_entities = {str(entity.get("entity_id") or ""): entity for entity in scene.get("entities") or [] if isinstance(entity, dict)}
    vehicle_entities = {entity_id: entity for entity_id, entity in scene_entities.items() if entity_id and is_vehicle_entity(entity)}
    uav_entities = {entity_id: entity for entity_id, entity in scene_entities.items() if entity_id and is_uav_entity(entity)}
    referenced_vehicle_ids: set[str] = set()
    event_modes_by_vehicle: dict[str, list[str]] = defaultdict(list)
    for event in events:
        trigger = triggers.get(str(event.get("trigger_ref") or ""))
        for target in event_targets(event, trigger):
            if target in vehicle_entities:
                referenced_vehicle_ids.add(target)
        for action in event.get("actions") or []:
            entity_id = action_entity_id(action)
            if entity_id in vehicle_entities:
                referenced_vehicle_ids.add(entity_id)
                mode = visual_mode(action)
                if mode:
                    event_modes_by_vehicle[entity_id].append(mode)

    samples_by_vehicle = load_samples_by_entity(episode_dir, set(vehicle_entities))
    roles = {entity_id: vehicle_role(entity_id, entity, event_modes_by_vehicle.get(entity_id, [])) for entity_id, entity in vehicle_entities.items()}
    vehicles: dict[str, dict[str, Any]] = {}
    instructions_by_vehicle: dict[str, list[dict[str, Any]]] = {}
    issues: list[str] = []
    for entity_id, entity in sorted(vehicle_entities.items()):
        base_instructions, base_issues, projection = build_vehicle_base_instructions(
            episode_id=episode_id,
            entity_id=entity_id,
            entity=entity,
            role=roles[entity_id],
            projector=projector,
            referenced=entity_id in referenced_vehicle_ids,
        )
        instructions_by_vehicle[entity_id] = list(base_instructions)
        issues.extend(base_issues)
        vehicles[entity_id] = {
            "entity_id": entity_id,
            "logical_asset_id": str(entity.get("logical_asset_id") or ""),
            "semantic_event_actor": entity_id in referenced_vehicle_ids,
            "sumo_vehicle": roles[entity_id],
            "projection": projection,
        }

    event_plans: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("event_id") or "")
        row = trace_by_event.get(event_id)
        event_tick = int(row.get("tick") if row and row.get("tick") is not None else row.get("activated_tick") if row else 0)
        trigger = triggers.get(str(event.get("trigger_ref") or ""))
        all_targets = event_targets(event, trigger)
        vehicle_targets = [target for target in all_targets if target in vehicle_entities]
        uav_targets = [target for target in all_targets if target in uav_entities or normalize_text(target).startswith("uav_")]
        event_instructions: list[dict[str, Any]] = []
        event_issues: list[str] = []
        action_details: list[dict[str, Any]] = []
        moved_vehicle_ids: set[str] = set()
        for action in event.get("actions") or []:
            entity_id = action_entity_id(action)
            action_type = str(action.get("type") or "")
            if entity_id not in vehicle_entities:
                continue
            if action_type == "move_entity":
                moved_vehicle_ids.add(entity_id)
                move_instructions, move_issues, details = build_move_instructions(
                    episode_id=episode_id,
                    scenario_id=scenario_id,
                    event=event,
                    event_tick=event_tick,
                    entity_id=entity_id,
                    action=action,
                    role=roles[entity_id],
                    projector=projector,
                    samples=samples_by_vehicle.get(entity_id, {}),
                    next_tick=next_vehicle_event_tick(events, trace_by_event, triggers, entity_id, event_tick),
                )
                event_instructions.extend(move_instructions)
                event_issues.extend(move_issues)
                action_details.append({"action_type": action_type, "entity_id": entity_id, **details})
            elif action_type == "set_visual_state":
                visual_instructions = build_visual_state_instructions(event, event_tick, entity_id, action, roles[entity_id])
                event_instructions.extend(visual_instructions)
                action_details.append({"action_type": action_type, "entity_id": entity_id, "mode": visual_mode(action)})
            elif action_type == "spawn_entity":
                event_instructions.append(
                    instruction(
                        tick=event_tick,
                        instruction_type="vehicle.addFull",
                        traci_call="traci.vehicle.addFull",
                        args={"vehID": entity_id, "depart": f"{event_tick / DEFAULT_TICK_HZ:.3f}", "typeID": roles[entity_id]["type_id"]},
                        event_id=event_id,
                        intent=str(event.get("intent") or ""),
                        target_role=roles[entity_id]["role"],
                        reason="vehicle spawn must be a SUMO vehicle insertion",
                    )
                )
            elif action_type == "remove_entity":
                event_instructions.append(
                    instruction(
                        tick=event_tick,
                        instruction_type="vehicle.remove",
                        traci_call="traci.vehicle.remove",
                        args={"vehID": entity_id},
                        event_id=event_id,
                        intent=str(event.get("intent") or ""),
                        target_role=roles[entity_id]["role"],
                        reason="vehicle removal must happen in SUMO, not UE actor deletion",
                    )
                )
        if vehicle_targets and not event_instructions and is_observation_only_vehicle_event(event, vehicle_targets):
            event_instructions.extend(
                build_observation_event_instructions(
                    scenario_id=scenario_id,
                    event=event,
                    event_tick=event_tick,
                    vehicle_targets=vehicle_targets,
                    roles=roles,
                )
            )
        for entity_id in vehicle_targets:
            if entity_id in moved_vehicle_ids:
                continue
            next_tick = next_vehicle_event_tick(events, trace_by_event, triggers, entity_id, event_tick)
            hold_end_tick = min(
                int(manifest.get("duration_ticks") or 900),
                max(event_tick + BEFORE_AFTER_TICKS, int(next_tick) if next_tick is not None else event_tick + BEFORE_AFTER_TICKS),
            )
            hold_instructions, hold_issues = build_vehicle_hold_instructions(
                event=event,
                event_tick=event_tick,
                entity_id=entity_id,
                role=roles[entity_id],
                projector=projector,
                samples=samples_by_vehicle.get(entity_id, {}),
                end_tick=hold_end_tick,
                reason="semantic vehicle is referenced by this event and must remain present through before/event/after evidence ticks",
            )
            event_instructions.extend(hold_instructions)
            event_issues.extend(hold_issues)
            if hold_instructions:
                action_details.append({"action_type": "hold_for_evidence", "entity_id": entity_id, "end_tick": hold_end_tick})
        for inst in event_instructions:
            veh_id = str(inst.get("args", {}).get("vehID") or "")
            if veh_id in instructions_by_vehicle:
                instructions_by_vehicle[veh_id].append(inst)
        status = "no_semantic_vehicle_actor"
        if vehicle_targets or event_instructions:
            status = "ok" if event_instructions else "vehicle_referenced_without_instruction"
        if event_issues:
            status = "needs_route_projection_fix"
        event_plans.append(
            {
                "event_id": event_id,
                "title": event_title(event),
                "intent": str(event.get("intent") or ""),
                "tick": event_tick,
                "review_ticks": {
                    "before": max(0, event_tick - BEFORE_AFTER_TICKS),
                    "event": event_tick,
                    "after": min(int(manifest.get("duration_ticks") or 900), event_tick + BEFORE_AFTER_TICKS),
                },
                "all_targets": all_targets,
                "vehicle_targets": vehicle_targets,
                "uav_targets": uav_targets,
                "status": status,
                "instructions": event_instructions,
                "action_details": action_details,
                "issues": event_issues,
            }
        )
        issues.extend(f"{event_id}: {issue}" for issue in event_issues)

    for entity_id in sorted(referenced_vehicle_ids):
        event_ticks_for_vehicle = [
            int(event.get("tick") or 0)
            for event in event_plans
            if entity_id in set(event.get("vehicle_targets") or [])
            or any(str(inst.get("args", {}).get("vehID") or "") == entity_id for inst in event.get("instructions") or [])
        ]
        if not event_ticks_for_vehicle:
            continue
        first_event_tick = min(event_ticks_for_vehicle)
        depart_projection = dict(vehicles.get(entity_id, {}).get("projection", {}).get("depart_projection") or {})
        edge_id = str(depart_projection.get("edge_id") or "")
        if first_event_tick > 0 and edge_id:
            start_position = list(vehicles.get(entity_id, {}).get("projection", {}).get("depart_position_enu_m") or [0.0, 0.0, 0.0])
            hold_front_x, hold_front_y = front_bumper_xy_for_center(start_position, 0.0)
            hold_samples = [
                {
                    "tick": tick,
                    "time_s": round(tick / DEFAULT_TICK_HZ, 3),
                    "x": round(float(hold_front_x), 6),
                    "y": round(float(hold_front_y), 6),
                    "target_center_enu_m": [round(float(start_position[0]), 6), round(float(start_position[1]), 6), round(float(start_position[2]), 6)],
                    "edgeID": edge_id,
                    "laneIndex": int(depart_projection.get("lane_index") or 0),
                    "angle": 90.0,
                    "keepRoute": 2,
                    "matchThreshold": 100.0,
                }
                for tick in range(0, first_event_tick)
            ]
            instructions_by_vehicle[entity_id].append(
                instruction(
                    tick=0,
                    instruction_type="vehicle.moveToXY_series",
                    traci_call="traci.vehicle.moveToXY",
                    args={
                        "vehID": entity_id,
                        "control_mode": "forced_xy",
                        "reason": "pre-event semantic hold at scripted action start",
                        "samples": hold_samples,
                    },
                    reason="hold semantic event vehicle at its scripted start until the first vehicle action begins",
                )
            )

    for entity_id, items in instructions_by_vehicle.items():
        items.sort(key=lambda item: (int(item.get("tick") or 0), str(item.get("instruction_type") or "")))

    return {
        "schema": "aero_sumo_semantic_vehicle_plan_v1",
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "source_event_script_path": str(script_path),
        "source_scene_setup_path": str(scene_path),
        "policy": {
            "all_vehicle_authority": "sumo",
            "render_ready_rule": "render-ready consumes SUMO vehicle trajectories only; no vehicle post-insertion or nearby SUMO deletion",
            "default_control": "vehicle.moveToXY_series forced_xy for scripted move_entity; vehicle.setParameter for observation/state bindings",
            "strong_control_allowed": True,
            "strong_control_note": "moveToXY is allowed only as an explicit SUMO-layer forced_xy control when route projection cannot realize event timing/geometry, such as contact/collision scripts on reverse or off-lane paths",
        },
        "vehicle_count": len(vehicle_entities),
        "uav_scene_count": len(uav_entities),
        "semantic_vehicle_count": len(referenced_vehicle_ids),
        "vehicles": vehicles,
        "instructions_by_vehicle": instructions_by_vehicle,
        "events": event_plans,
        "issues": issues,
        "acceptance": acceptance_payload(episode_id),
    }


def acceptance_payload(episode_id: str) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "must_have": [
            "sumo_semantic_vehicle_plan.json exists inside the episode directory",
            "all scene vehicles have vehicle.addFull instructions",
            "all semantic vehicle events have SUMO-readable forced_xy/state instructions or an explicit no_semantic_vehicle_actor status",
            "geometry-locked semantic vehicle events may use explicit traci.vehicle.moveToXY forced_xy samples; this remains SUMO-layer control, not UE post-insertion",
            "no vehicle event relies on UE post-insert rendering, semantic_vehicle_overlay, or local deletion of nearby SUMO vehicles",
            "ROI trajectory GIF exists in the review directory when --write-gif is requested",
            "semantic vehicles use distinct colors; police/medical/emergency vehicles are visually distinguishable",
            "UAV trajectories are visible in the same GIF when present in source or render-ready truth",
            "GIF metadata records track_source and counts for vehicles, SUMO vehicles, semantic vehicles, and UAVs",
            "review markdown lists event tick, before/after ticks, vehicle ids, instruction calls, and route projection issues",
        ],
        "failure_conditions": [
            "route_edges is empty for a move_entity instruction",
            "a forced_xy moveToXY instruction has no per-tick samples or samples outside the action review window",
            "a semantic vehicle target has status vehicle_referenced_without_instruction",
            "any instruction_type starts with ue. or render_ready.post_insert",
            "GIF/documentation missing for an episode assigned to a worker shard",
            "roi_vehicle_trajectories_review.json is missing or reports no UAV tracks for a UAV event episode",
        ],
    }


def write_markdown(path: Path, plan: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# SUMO Semantic Vehicle Actions - {plan['episode_id']}")
    lines.append("")
    lines.append(f"- Scenario: `{plan['scenario_id']}`")
    lines.append(f"- Vehicle authority: `{plan['policy']['all_vehicle_authority']}`")
    lines.append(f"- Scene vehicles: {plan['vehicle_count']}")
    lines.append(f"- Scene UAVs: {plan.get('uav_scene_count', 0)}")
    lines.append(f"- Semantic event vehicles: {plan['semantic_vehicle_count']}")
    lines.append("- Review GIF: `roi_vehicle_trajectories.gif` overlays semantic vehicles, SUMO/background vehicles, UAV trajectories, and event target pre/event/post tick markers.")
    lines.append("")
    lines.append("## Vehicle Legend")
    lines.append("")
    lines.append("| Vehicle | Semantic | Type | Role |")
    lines.append("|---|---:|---|---|")
    for vehicle_id, vehicle in sorted(plan["vehicles"].items()):
        sumo_vehicle = vehicle["sumo_vehicle"]
        lines.append(f"| `{vehicle_id}` | {vehicle['semantic_event_actor']} | `{sumo_vehicle['type_id']}` | `{sumo_vehicle['role']}` |")
    lines.append("")
    lines.append("## Event Instructions")
    lines.append("")
    for event in plan["events"]:
        lines.append(f"### {event['event_id']} @ tick {event['tick']}")
        lines.append("")
        lines.append(f"- Title: {event['title']}")
        lines.append(f"- Intent: `{event['intent']}`")
        ticks = event["review_ticks"]
        lines.append(f"- Review ticks: before `{ticks['before']}`, event `{ticks['event']}`, after `{ticks['after']}`")
        lines.append(f"- Status: `{event['status']}`")
        lines.append(f"- All targets: {', '.join(f'`{item}`' for item in event.get('all_targets', [])) if event.get('all_targets') else '<none>'}")
        lines.append(f"- Vehicle targets: {', '.join(f'`{item}`' for item in event['vehicle_targets']) if event['vehicle_targets'] else '<none>'}")
        lines.append(f"- UAV targets: {', '.join(f'`{item}`' for item in event.get('uav_targets', [])) if event.get('uav_targets') else '<none>'}")
        if event["issues"]:
            lines.append(f"- Issues: {'; '.join(event['issues'])}")
        lines.append("")
        if not event["instructions"]:
            lines.append("No vehicle SUMO instruction is required for this event.")
            lines.append("")
            continue
        lines.append("| Tick | Vehicle | TraCI call | Arguments | Reason |")
        lines.append("|---:|---|---|---|---|")
        for inst in event["instructions"]:
            args = dict(inst.get("args") or {})
            vehicle = str(args.get("vehID") or "")
            if inst.get("instruction_type") == "vehicle.moveToXY_series":
                samples = list(args.get("samples") or [])
                args = {
                    "vehID": vehicle,
                    "control_mode": args.get("control_mode"),
                    "reason": args.get("reason"),
                    "sample_count": len(samples),
                    "first_sample": samples[0] if samples else {},
                    "last_sample": samples[-1] if samples else {},
                }
            arg_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
            lines.append(f"| {inst['tick']} | `{vehicle}` | `{inst['traci_call']}` | `{arg_text}` | {inst['reason']} |")
        lines.append("")
    lines.append("## Acceptance")
    lines.append("")
    for item in plan["acceptance"]["must_have"]:
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def vehicle_color(vehicle_id: str, vehicle: dict[str, Any], semantic_index: int) -> str:
    role = str(dict(vehicle.get("sumo_vehicle") or {}).get("role") or "")
    if role == "police":
        return POLICE_COLOR
    if role == "medical_response":
        return MEDICAL_COLOR
    if bool(vehicle.get("semantic_event_actor")):
        return SEMANTIC_COLORS[semantic_index % len(SEMANTIC_COLORS)]
    return BACKGROUND_COLOR


def trajectory_points(samples: dict[int, dict[str, Any]], start_tick: int, end_tick: int) -> list[tuple[int, float, float]]:
    points: list[tuple[int, float, float]] = []
    for tick in sorted(samples):
        if tick < start_tick or tick > end_tick:
            continue
        pos = pos3(samples[tick].get("pos_enu"))
        points.append((tick, pos[0], pos[1]))
    return points


def truth_entity_position(entity: dict[str, Any]) -> list[float] | None:
    pose = dict(entity.get("truth_pose") or {})
    for key in ("position_enu_m", "position_m"):
        if key in pose:
            return pos3(pose.get(key))
    for key in ("pos_enu", "position_enu_m", "truth_position_enu_m"):
        if key in entity:
            return pos3(entity.get(key))
    return None


def entity_track_category(row: dict[str, Any]) -> str:
    return normalize_text(row.get("entity_category") or row.get("category") or row.get("label_class"))


def load_review_tracks(
    *,
    episode_dir: Path,
    render_ready_episode_dir: Path | None,
    plan: dict[str, Any],
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, dict[str, Any]], str]:
    tracks: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    metadata: dict[str, dict[str, Any]] = {}
    semantic_vehicle_ids = {
        str(vehicle_id)
        for vehicle_id, vehicle in dict(plan.get("vehicles") or {}).items()
        if bool(dict(vehicle).get("semantic_event_actor"))
    }
    plan_vehicle_ids = set(str(vehicle_id) for vehicle_id in dict(plan.get("vehicles") or {}))
    truth_path = (render_ready_episode_dir / "truth_frames.jsonl") if render_ready_episode_dir else None
    track_source = "render_ready_truth_frames" if truth_path is not None and truth_path.exists() else "missing_render_ready_truth_frames"
    if truth_path is not None and truth_path.exists():
        for frame in iter_jsonl(truth_path):
            tick = int(frame.get("tick") if frame.get("tick") is not None else frame.get("frame_seq", 0))
            for entity in frame.get("entities") or []:
                if not isinstance(entity, dict):
                    continue
                category = entity_track_category(entity)
                if category not in {"vehicle", "uav"}:
                    continue
                entity_id = str(entity.get("entity_id") or "")
                position = truth_entity_position(entity)
                if not entity_id or position is None:
                    continue
                row = {
                    "pos_enu": position,
                    "category": category,
                    "source": str(entity.get("source") or ""),
                    "logical_asset_id": str(entity.get("logical_asset_id") or entity.get("asset_id") or ""),
                    "role": str(entity.get("role") or entity.get("semantic_role") or entity.get("background_role") or ""),
                    "sumo_vehicle": dict(entity.get("sumo_vehicle") or {}),
                    "uav_global_flow": dict(entity.get("uav_global_flow") or {}),
                }
                tracks[entity_id][tick] = row
                meta = metadata.setdefault(entity_id, {})
                meta.update(
                    {
                        "category": category,
                        "source": row["source"],
                        "logical_asset_id": row["logical_asset_id"],
                        "role": row["role"],
                        "is_sumo_vehicle": category == "vehicle"
                        and (row["source"] == "sumo_traci" or bool(row["sumo_vehicle"])),
                        "is_semantic_vehicle": entity_id in semantic_vehicle_ids,
                        "is_plan_vehicle": entity_id in plan_vehicle_ids,
                    }
                )
    return dict(tracks), metadata, track_source


def nearest_track_point(samples: dict[int, dict[str, Any]], tick: int) -> tuple[int, float, float] | None:
    if not samples:
        return None
    nearest_tick = min(samples, key=lambda sample_tick: abs(int(sample_tick) - int(tick)))
    position = pos3(samples[nearest_tick].get("pos_enu"))
    return int(nearest_tick), position[0], position[1]


def write_gif(
    review_dir: Path,
    plan: dict[str, Any],
    episode_dir: Path,
    *,
    frame_step: int,
    render_ready_episode_dir: Path | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    samples, track_meta, track_source = load_review_tracks(
        episode_dir=episode_dir,
        render_ready_episode_dir=render_ready_episode_dir,
        plan=plan,
    )
    event_ticks = sorted(
        {
            int(event.get("tick") or 0)
            for event in plan.get("events") or []
            if int(event.get("tick") or 0) >= 0
        }
    )
    if event_ticks:
        start_tick = max(0, min(event_ticks) - 40)
        end_tick = min(900, max(event_ticks) + 80)
    else:
        start_tick = 0
        end_tick = 900
    all_points: list[tuple[float, float]] = []
    for entity_samples in samples.values():
        for _tick, x, y in trajectory_points(entity_samples, start_tick, end_tick):
            all_points.append((x, y))
    if not all_points:
        all_points = [(0.0, 0.0), (1.0, 1.0)]
    xs = [item[0] for item in all_points]
    ys = [item[1] for item in all_points]
    margin = max(10.0, 0.08 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0))
    xlim = (min(xs) - margin, max(xs) + margin)
    ylim = (min(ys) - margin, max(ys) + margin)
    semantic_order = [vid for vid, vehicle in sorted(plan["vehicles"].items()) if vehicle.get("semantic_event_actor")]
    semantic_index = {vid: idx for idx, vid in enumerate(semantic_order)}
    frame_step = max(5, int(frame_step or 15))
    frame_ticks = list(range(start_tick, end_tick + 1, frame_step))
    if end_tick not in frame_ticks:
        frame_ticks.append(end_tick)
    review_dir.mkdir(parents=True, exist_ok=True)
    gif_path = review_dir / "roi_vehicle_trajectories.gif"

    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)

    def draw(frame_tick: int) -> None:
        ax.clear()
        ax.set_title(f"{plan['episode_id']} ROI vehicle/UAV trajectories | tick {frame_tick}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#dddddd", linewidth=0.5)
        ax.set_xlabel("ENU X (m)")
        ax.set_ylabel("ENU Y (m)")
        for vehicle_id, vehicle in sorted(plan["vehicles"].items()):
            entity_samples = samples.get(vehicle_id, {})
            pts = trajectory_points(entity_samples, start_tick, frame_tick)
            if not pts:
                continue
            color = vehicle_color(vehicle_id, vehicle, semantic_index.get(vehicle_id, 0))
            linestyle = "-" if vehicle.get("semantic_event_actor") else "--"
            alpha = 0.95 if vehicle.get("semantic_event_actor") else 0.45
            ax.plot([p[1] for p in pts], [p[2] for p in pts], linestyle=linestyle, color=color, linewidth=2.2, alpha=alpha)
            marker = "P" if dict(vehicle.get("sumo_vehicle") or {}).get("role") == "police" else "X" if dict(vehicle.get("sumo_vehicle") or {}).get("role") == "medical_response" else "o"
            ax.scatter([pts[-1][1]], [pts[-1][2]], color=color, marker=marker, s=42 if vehicle.get("semantic_event_actor") else 18, alpha=alpha, zorder=4)
            if vehicle.get("semantic_event_actor"):
                ax.text(pts[-1][1], pts[-1][2], f" {vehicle_id}", fontsize=7, color=color)
        for entity_id, entity_samples in sorted(samples.items()):
            meta = track_meta.get(entity_id, {})
            if str(meta.get("category") or "") != "vehicle" or entity_id in plan["vehicles"]:
                continue
            pts = trajectory_points(entity_samples, start_tick, frame_tick)
            if not pts:
                continue
            is_sumo = bool(meta.get("is_sumo_vehicle"))
            color = "#9a9a9a" if is_sumo else "#555555"
            ax.plot([p[1] for p in pts], [p[2] for p in pts], linestyle="--", color=color, linewidth=1.0, alpha=0.32)
            ax.scatter([pts[-1][1]], [pts[-1][2]], color=color, marker=".", s=12, alpha=0.42, zorder=2)
        for entity_id, entity_samples in sorted(samples.items()):
            meta = track_meta.get(entity_id, {})
            if str(meta.get("category") or "") != "uav":
                continue
            pts = trajectory_points(entity_samples, start_tick, frame_tick)
            if not pts:
                continue
            ax.plot([p[1] for p in pts], [p[2] for p in pts], linestyle=":", color="#17becf", linewidth=2.0, alpha=0.88)
            ax.scatter([pts[-1][1]], [pts[-1][2]], color="#17becf", marker="^", s=46, alpha=0.95, zorder=5)
            ax.text(pts[-1][1], pts[-1][2], f" {entity_id}", fontsize=7, color="#0b7580")
        active_labels: list[str] = []
        for event in plan.get("events") or []:
            tick = int(event.get("tick") or 0)
            if start_tick <= tick <= end_tick:
                targets = list(event.get("all_targets") or event.get("vehicle_targets") or [])
                for target_id in targets:
                    target_samples = samples.get(str(target_id), {})
                    for marker_tick, label, marker, color in (
                        (max(0, tick - BEFORE_AFTER_TICKS), "pre", "<", "#777777"),
                        (tick, "event", "*", "#d62728"),
                        (min(900, tick + BEFORE_AFTER_TICKS), "post", ">", "#333333"),
                    ):
                        point = nearest_track_point(target_samples, marker_tick)
                        if point is None:
                            continue
                        _sample_tick, x, y = point
                        alpha = 0.95 if tick - BEFORE_AFTER_TICKS <= frame_tick <= tick + BEFORE_AFTER_TICKS else 0.28
                        ax.scatter([x], [y], marker=marker, color=color, s=70 if label == "event" else 34, alpha=alpha, zorder=6)
                        if label == "event" and alpha > 0.5:
                            ax.text(x, y, f" {event['event_id']}@{tick}", fontsize=7, color=color)
                if tick - BEFORE_AFTER_TICKS <= frame_tick <= tick + BEFORE_AFTER_TICKS:
                    active_labels.append(f"{event['event_id']}@{tick}: {', '.join(targets) or 'non-target'}")
        if active_labels:
            ax.text(
                0.01,
                0.99,
                "\n".join(active_labels[:5]),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "#333333", "alpha": 0.85},
            )
        legend_items = [
            "semantic vehicles: solid bright",
            "SUMO/background vehicles: dashed gray",
            "UAVs: cyan dotted ^ marker",
            "police: blue P marker",
            "medical/emergency: green X marker",
            "event target markers: pre <, event *, post >",
            f"track source: {track_source}",
            f"review window: ticks {start_tick}..{end_tick}",
        ]
        ax.text(
            0.99,
            0.01,
            "\n".join(legend_items),
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.8},
        )

    animation = matplotlib.animation.FuncAnimation(fig, draw, frames=frame_ticks, interval=120, repeat=True)
    animation.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig)
    metadata = {
        "schema": "aero_roi_vehicle_uav_gif_review_v1",
        "episode_id": plan["episode_id"],
        "track_source": track_source,
        "review_window_ticks": {"start": start_tick, "end": end_tick, "frame_step": frame_step},
        "track_counts": {
            "vehicle": sum(1 for meta in track_meta.values() if meta.get("category") == "vehicle"),
            "uav": sum(1 for meta in track_meta.values() if meta.get("category") == "uav"),
            "sumo_vehicle": sum(1 for meta in track_meta.values() if meta.get("is_sumo_vehicle")),
            "semantic_vehicle": sum(1 for meta in track_meta.values() if meta.get("is_semantic_vehicle")),
            "plan_vehicle": sum(1 for meta in track_meta.values() if meta.get("is_plan_vehicle")),
        },
        "event_count": len(plan.get("events") or []),
        "events": [
            {
                "event_id": event.get("event_id"),
                "tick": event.get("tick"),
                "all_targets": list(event.get("all_targets") or []),
                "vehicle_targets": list(event.get("vehicle_targets") or []),
                "uav_targets": list(event.get("uav_targets") or []),
            }
            for event in plan.get("events") or []
        ],
        "gif_path": str(gif_path),
    }
    write_json(review_dir / "roi_vehicle_trajectories_review.json", metadata)
    return gif_path


def parse_episode_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Episode file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def selected_episode_dirs(root: Path, names: list[str], limit: int) -> list[Path]:
    dirs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "episode_manifest.json").exists())
    if names:
        wanted = set(names)
        dirs = [path for path in dirs if path.name in wanted or path.name.split("__seed", 1)[0] in wanted]
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def write_subagent_manifest(review_root: Path, plans: Sequence[dict[str, Any]]) -> Path:
    groups: dict[str, list[str]] = defaultdict(list)
    for plan in plans:
        scenario = str(plan["scenario_id"])
        group = scenario.split("-", 1)[0] if "-" in scenario else scenario.split("_", 1)[0]
        groups[group].append(str(plan["episode_id"]))
    shards = []
    for index, (group, episodes) in enumerate(sorted(groups.items()), start=1):
        shards.append(
            {
                "shard_id": f"sumo_semantic_vehicle_{index:02d}_{group}",
                "group": group,
                "episodes": sorted(episodes),
                "write_scope": [
                    "Dataset/episodes/<episode>/sumo_semantic_vehicle_plan.json",
                    "G:/aw_cap/_meta/sumo_semantic_vehicle_review/<episode>/semantic_vehicle_actions.md",
                    "G:/aw_cap/_meta/sumo_semantic_vehicle_review/<episode>/semantic_vehicle_actions.json",
                    "G:/aw_cap/_meta/sumo_semantic_vehicle_review/<episode>/roi_vehicle_trajectories.gif",
                    "G:/aw_cap/_meta/sumo_semantic_vehicle_review/<episode>/roi_vehicle_trajectories_review.json",
                ],
                "acceptance": [
                    "Every episode has a non-empty SUMO semantic vehicle plan",
                    "Every semantic vehicle event is mapped to SUMO-readable instructions or explicitly marked no_semantic_vehicle_actor",
                    "GIF, GIF metadata, and markdown exist for every episode in this shard",
                    "GIF metadata proves UAV trajectories are present for UAV event episodes",
                    "No worker edits old UE/manual vehicle rendering paths; cleanup is owned by the parent after plan validation",
                ],
            }
        )
    payload = {
        "schema": "aero_sumo_semantic_vehicle_subagent_manifest_v1",
        "review_root": str(review_root),
        "shards": shards,
    }
    path = review_root / "subagent_work_manifest.json"
    write_json(path, payload)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_EPISODES_ROOT)
    parser.add_argument("--render-ready-root", type=Path, default=DEFAULT_RENDER_READY_ROOT)
    parser.add_argument("--sumo-net-xml", type=Path, default=DEFAULT_SUMO_NET_XML)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--episode-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-episode-files", action="store_true")
    parser.add_argument("--write-gif", action="store_true")
    parser.add_argument("--gif-frame-step", type=int, default=15)
    args = parser.parse_args()

    planner = SumoGroundFlowPlanner(args.sumo_net_xml)
    projector = RouteProjector(planner)
    episode_names = [*args.episode, *parse_episode_file(args.episode_file)]
    episode_dirs = selected_episode_dirs(args.episodes_root.resolve(), episode_names, int(args.limit or 0))
    if not episode_dirs:
        raise SystemExit("No episodes selected")
    plans: list[dict[str, Any]] = []
    for index, episode_dir in enumerate(episode_dirs, start=1):
        plan = build_episode_plan(episode_dir, projector)
        plans.append(plan)
        if not args.no_episode_files:
            write_json(episode_dir / "sumo_semantic_vehicle_plan.json", plan)
        review_dir = args.review_root / str(plan["episode_id"])
        write_markdown(review_dir / "semantic_vehicle_actions.md", plan)
        write_json(review_dir / "semantic_vehicle_actions.json", plan)
        if args.write_gif:
            write_gif(
                review_dir,
                plan,
                episode_dir,
                frame_step=int(args.gif_frame_step or 15),
                render_ready_episode_dir=Path(args.render_ready_root) / str(plan["episode_id"]),
            )
        print(f"[{index}/{len(episode_dirs)}] {plan['episode_id']}: vehicles={plan['vehicle_count']} semantic={plan['semantic_vehicle_count']} issues={len(plan['issues'])}")
    manifest_path = write_subagent_manifest(args.review_root, plans)
    summary = {
        "episodes": len(plans),
        "vehicles": sum(int(plan["vehicle_count"]) for plan in plans),
        "semantic_vehicles": sum(int(plan["semantic_vehicle_count"]) for plan in plans),
        "issues": sum(len(plan["issues"]) for plan in plans),
        "subagent_manifest": str(manifest_path),
    }
    write_json(args.review_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
