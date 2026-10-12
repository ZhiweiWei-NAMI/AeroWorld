"""Run Donghu SUMO traffic and export truth-frame compatible samples."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any, Sequence

from .coordinates import ensure_sumo_tools_path, sumo_executable_from_environment
from .incident_plan import (
    DEFAULT_SCENARIOS_ROOT,
    DEFAULT_SUMO_NET_XML,
    build_incident_plan,
    incident_records,
    load_incident_plan,
    write_incident_plan,
)
from .explicit_vehicle_plan import (
    VEHICLE_SOURCE_POLICY,
    instructions_from_explicit_plan,
    load_explicit_vehicle_plan,
    planned_vehicle_ids,
)
from .planner import SumoEdge, SumoGroundFlowPlanner
from .road_semantic_rules import resolve_episode_road_semantics
from .incident_execution import LaneIncidentExecution


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "aw_data" / "sumo_outputs" / "episode_vehicle_authority"
DEFAULT_SEMANTIC_PLAN_ROOT = ROOT / "Dataset" / "episodes"
DEFAULT_LANE_CENTER_SAMPLES = ROOT / "Config" / "LowAltitude" / "Maps" / "donghu_road_topo" / "traffic_bundle" / "lane_center_samples.csv"
DEFAULT_DURATION_S = 90.0
DEFAULT_WARMUP_S = 90.0
DEFAULT_STEP_LENGTH_S = 0.1
DEFAULT_SAMPLE_PERIOD_S = DEFAULT_STEP_LENGTH_S
DEFAULT_MAX_VEHICLES = 200
DEFAULT_VEHICLE_LENGTH_M = 5.0
DEFAULT_VEHICLE_WIDTH_M = 1.8
MIN_DEPART_FRONT_POS_M = DEFAULT_VEHICLE_LENGTH_M + 1.5
MIN_STOP_FRONT_POS_M = DEFAULT_VEHICLE_LENGTH_M + 7.0
STOP_END_MARGIN_M = 1.0
SIGNAL_HAZARD = 8
SIGNAL_EMERGENCY = 10
ROI_LANE_COVERAGE_PADDING_M = 60.0
ROI_LANE_COVERAGE_MIN_SAMPLE_POINTS = 3
ROI_LANE_COVERAGE_SPAWN_PERIOD = 2
DEMAND_RAMP_S = 15.0
MAX_FIND_ROUTE_ATTEMPTS_PER_VEHICLE = 768
MAX_FIND_ROUTE_VALID_CANDIDATES = 24
MAX_FIND_ROUTE_VALID_STARTS = 6
PENDING_STOP_RETRY_S = 2.0


@dataclass(frozen=True)
class RoiLaneCoverageAnchor:
    edge_id: str
    depart_pos_m: float
    source_road_id: str
    direction_role: str
    physical_lane_index: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_object_or_none(value: str) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _ordered_unique_strings(values: Sequence[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return ordered


def _apply_road_semantics_to_explicit_vehicle_plan(
    plan: dict[str, Any],
    *,
    episode_dir: Path,
    planner: SumoGroundFlowPlanner,
) -> dict[str, Any]:
    updated = copy.deepcopy(plan)
    semantics = resolve_episode_road_semantics(episode_dir, episode_id=str(updated.get("episode_id") or episode_dir.name))
    semantic_payload = semantics.as_dict()
    closed_edges = {edge_id for edge_id in semantics.closed_edges if edge_id in planner.edges}
    if not closed_edges:
        updated.setdefault("road_semantics", semantic_payload)
        return updated
    updated["road_semantics"] = semantic_payload
    for vehicle in updated.get("vehicles") or []:
        if not isinstance(vehicle, dict):
            continue
        if str(vehicle.get("traffic_role") or "") == "semantic_vehicle":
            continue
        forbidden_edges = {str(edge_id) for edge_id in vehicle.get("forbidden_edges") or [] if str(edge_id)}
        forbidden_edges.update(closed_edges)
        vehicle["forbidden_edges"] = sorted(forbidden_edges)
    return updated


def _bounded_endpoint_pairs(
    from_edges: Sequence[str],
    to_edges: Sequence[str],
    *,
    max_pairs: int,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(src: str, dst: str) -> None:
        if len(pairs) >= max_pairs:
            return
        pair = (str(src), str(dst))
        if not pair[0] or not pair[1] or pair in seen:
            return
        seen.add(pair)
        pairs.append(pair)

    if not from_edges or not to_edges or max_pairs <= 0:
        return pairs

    primary_from = str(from_edges[0])
    primary_to = str(to_edges[0])
    add(primary_from, primary_to)

    for index in range(max(len(from_edges), len(to_edges))):
        if index < len(to_edges):
            add(primary_from, str(to_edges[index]))
        if index < len(from_edges):
            add(str(from_edges[index]), primary_to)
        if index < len(from_edges) and index < len(to_edges):
            add(str(from_edges[index]), str(to_edges[index]))
        if len(pairs) >= max_pairs:
            return pairs

    for src in from_edges:
        for dst in to_edges:
            add(str(src), str(dst))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _vehicle_parameter(traci: Any, vehicle_id: str, key: str) -> str:
    try:
        return str(traci.vehicle.getParameter(vehicle_id, key) or "")
    except Exception:
        return ""


def _capture_boundary_bbox(event_script: dict[str, Any]) -> tuple[float, float, float, float] | None:
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

    visit(event_script)
    boundary = candidates[0] if candidates else None
    if boundary is None:
        return None
    points = boundary.get("polygon_enu_m") or []
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) >= 2:
            try:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
            except (TypeError, ValueError):
                pass
    if not xs or not ys:
        return None
    padding = float(boundary.get("expanded_boundary_padding_m") or ROI_LANE_COVERAGE_PADDING_M)
    return min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding


def _iter_scenario_event_scripts(scenarios_root: Path, scenario_ids: set[str]) -> list[Path]:
    scripts = sorted(Path(scenarios_root).glob("**/event_script.json"))
    if not scenario_ids:
        return []
    selected: list[Path] = []
    for path in scripts:
        scenario_id = path.parent.name
        if scenario_id in scenario_ids:
            selected.append(path)
    return selected


def build_roi_lane_coverage_anchors(
    *,
    scenarios_root: Path,
    lane_center_samples: Path = DEFAULT_LANE_CENTER_SAMPLES,
    scenario_ids: Sequence[str] = (),
) -> list[RoiLaneCoverageAnchor]:
    selected_ids = {str(item) for item in scenario_ids if str(item)}
    if not selected_ids or not Path(lane_center_samples).exists():
        return []
    bboxes: list[tuple[float, float, float, float]] = []
    for event_script_path in _iter_scenario_event_scripts(Path(scenarios_root), selected_ids):
        bbox = _capture_boundary_bbox(_read_json(event_script_path))
        if bbox is not None:
            bboxes.append(bbox)
    if not bboxes:
        return []

    import csv

    group_edges: dict[tuple[str, str], dict[int, dict[str, list[float]]]] = {}
    with Path(lane_center_samples).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            direction = str(row.get("direction_role") or "")
            if direction not in {"f", "r"}:
                continue
            try:
                x_m = float(row.get("x_m") or 0.0)
                y_m = float(row.get("y_m") or 0.0)
                s_m = float(row.get("s_m") or 0.0)
                physical_lane = int(float(row.get("physical_lane_index") or 0))
            except (TypeError, ValueError):
                continue
            if not any(min_x <= x_m <= max_x and min_y <= y_m <= max_y for min_x, min_y, max_x, max_y in bboxes):
                continue
            edge_id = str(row.get("edge_id") or "")
            if not edge_id.startswith("cg_edge_"):
                continue
            key = (str(row.get("source_road_id") or ""), direction)
            lane_counts = group_edges.setdefault(key, {}).setdefault(physical_lane, {})
            lane_counts.setdefault(edge_id, []).append(s_m)

    targets: list[RoiLaneCoverageAnchor] = []
    for _key, lanes in sorted(group_edges.items(), key=lambda item: (int(item[0][0] or 0), item[0][1])):
        if len(lanes) < 2:
            continue
        for lane, edges in sorted(lanes.items()):
            best_edge, s_values = max(edges.items(), key=lambda item: len(item[1]))
            sample_count = len(s_values)
            if sample_count >= ROI_LANE_COVERAGE_MIN_SAMPLE_POINTS:
                ordered_s = sorted(float(value) for value in s_values)
                depart_pos = ordered_s[len(ordered_s) // 2]
                targets.append(
                    RoiLaneCoverageAnchor(
                        edge_id=best_edge,
                        depart_pos_m=depart_pos,
                        source_road_id=_key[0],
                        direction_role=_key[1],
                        physical_lane_index=int(lane),
                    )
                )
    return targets


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _episode_scenario_id(episode_id: str) -> str:
    return re.sub(r"__seed\d+$", "", str(episode_id))


def _episode_output_dir(output_dir: Path, episode_id: str) -> Path:
    root = Path(output_dir)
    return root if root.name == episode_id else root / episode_id


def _filter_incident_plan_by_scenarios(incident_plan: dict[str, Any], scenario_ids: Sequence[str]) -> dict[str, Any]:
    scenario_scope = {str(scenario_id) for scenario_id in scenario_ids if str(scenario_id)}
    if not scenario_scope:
        return incident_plan
    filtered = copy.deepcopy(incident_plan)
    incidents = [
        item
        for item in (filtered.get("incidents") or [])
        if str(item.get("episode_scenario_id") or "") in scenario_scope
    ]
    filtered["incidents"] = incidents
    filtered["incident_count"] = len(incidents)
    filtered["scenario_scope"] = sorted(scenario_scope)
    filtered["policy"] = "episode_scoped_sumo_incidents_v1"
    return filtered


def _offset_incident_plan_times(incident_plan: dict[str, Any], *, warmup_s: float) -> dict[str, Any]:
    offset = float(warmup_s)
    if abs(offset) <= 1e-9:
        return incident_plan
    shifted = copy.deepcopy(incident_plan)
    for incident in shifted.get("incidents") or []:
        incident["episode_local_start_s"] = float(incident.get("start_s") or 0.0)
        incident["episode_local_end_s"] = float(incident.get("end_s") or 0.0)
        incident["start_s"] = round(offset + float(incident["episode_local_start_s"]), 6)
        incident["end_s"] = round(offset + float(incident["episode_local_end_s"]), 6)
    shifted["time_axis"] = {
        "policy": "episode_local_vehicle_sumo_after_warmup_v1",
        "warmup_s": round(offset, 6),
        "capture_start_s": round(offset, 6),
    }
    return shifted


def _truth_yaw_from_sumo_angle(angle_deg: float) -> float:
    yaw = 90.0 - float(angle_deg)
    while yaw > 180.0:
        yaw -= 360.0
    while yaw <= -180.0:
        yaw += 360.0
    return yaw


def _vehicle_center_from_front_bumper(
    *,
    front_truth_xy_m: tuple[float, float],
    yaw_deg: float,
    length_m: float,
) -> tuple[float, float]:
    yaw_rad = math.radians(float(yaw_deg))
    half_length = 0.5 * max(0.0, float(length_m))
    return (
        float(front_truth_xy_m[0]) - math.cos(yaw_rad) * half_length,
        float(front_truth_xy_m[1]) - math.sin(yaw_rad) * half_length,
    )


def _point_and_yaw_at_s(points_xy: Sequence[Sequence[float]], s_m: float) -> tuple[float, float, float] | None:
    if not points_xy:
        return None
    remaining = max(0.0, float(s_m))
    last = points_xy[0]
    for current in points_xy[1:]:
        seg_len = math.hypot(float(current[0]) - float(last[0]), float(current[1]) - float(last[1]))
        if seg_len > 1e-9 and remaining <= seg_len:
            t = remaining / seg_len
            return (
                float(last[0]) + (float(current[0]) - float(last[0])) * t,
                float(last[1]) + (float(current[1]) - float(last[1])) * t,
                math.degrees(math.atan2(float(current[1]) - float(last[1]), float(current[0]) - float(last[0]))),
            )
        remaining -= seg_len
        last = current
    if len(points_xy) >= 2:
        dx = float(points_xy[-1][0]) - float(points_xy[-2][0])
        dy = float(points_xy[-1][1]) - float(points_xy[-2][1])
        yaw_deg = math.degrees(math.atan2(dy, dx))
    else:
        yaw_deg = 0.0
    return float(points_xy[-1][0]), float(points_xy[-1][1]), yaw_deg


def _point_at_s(points_xy: Sequence[Sequence[float]], s_m: float) -> tuple[float, float] | None:
    pose = _point_and_yaw_at_s(points_xy, s_m)
    if pose is None:
        return None
    return pose[0], pose[1]


def _vehicle_body_center_pose(
    *,
    front_truth_xy_m: tuple[float, float],
    yaw_deg: float,
    length_m: float,
    lane_position_m: float,
    lane_shape: Sequence[Sequence[float]],
    coordinate_mapper: Any,
) -> tuple[float, float, float, float, str]:
    half_length_m = 0.5 * max(0.0, float(length_m))
    center_lane_position_m = float(lane_position_m) - half_length_m
    center_tx, center_ty = _vehicle_center_from_front_bumper(
        front_truth_xy_m=front_truth_xy_m,
        yaw_deg=yaw_deg,
        length_m=length_m,
    )
    return (
        float(center_tx),
        float(center_ty),
        float(yaw_deg),
        float(center_lane_position_m),
        "vehicle_body_center_from_front_bumper",
    )


def _round_list(values: Sequence[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def _count_net_traffic_lights(net_xml: Path) -> int:
    ids: set[str] = set()
    for _event, elem in ET.iterparse(net_xml, events=("end",)):
        if elem.tag == "tlLogic":
            ids.add(str(elem.attrib.get("id") or ""))
        elem.clear()
    return len(ids)


def _edge_allows_class(edge: SumoEdge, vehicle_class: str) -> bool:
    if edge.allow:
        return vehicle_class in edge.allow
    forbidden_type = any(token in edge.edge_type for token in ("footway", "pedestrian", "path", "steps", "rail"))
    return vehicle_class not in edge.disallow and not forbidden_type


class RouteFactory:
    EDGE_GROUP_RE = re.compile(r"^cg_edge_(?P<road>\d+)_(?P<direction>[fr])_pl(?P<lane>\d+)$")

    def __init__(self, planner: SumoGroundFlowPlanner, rng: random.Random) -> None:
        self.planner = planner
        self.rng = rng
        self.edges_by_class: dict[str, list[SumoEdge]] = {
            "passenger": [edge for edge in planner.edges.values() if _edge_allows_class(edge, "passenger")],
            "delivery": [edge for edge in planner.edges.values() if _edge_allows_class(edge, "delivery")],
        }
        self.edge_use_counts: dict[str, int] = {}
        self.sibling_physical_lanes: dict[tuple[str, str], list[str]] = {}
        for edge in planner.edges.values():
            key = self._edge_group_key(edge.edge_id)
            if key is not None:
                self.sibling_physical_lanes.setdefault(key, []).append(edge.edge_id)
        for key, values in list(self.sibling_physical_lanes.items()):
            values.sort(key=self._physical_lane_index)
            if len(values) < 2:
                self.sibling_physical_lanes.pop(key, None)

    def _edge_group_key(self, edge_id: str) -> tuple[str, str] | None:
        match = self.EDGE_GROUP_RE.match(edge_id)
        if not match:
            return None
        return match.group("road"), match.group("direction")

    def _physical_lane_index(self, edge_id: str) -> int:
        match = self.EDGE_GROUP_RE.match(edge_id)
        if not match:
            return 0
        return int(match.group("lane"))

    def _record_edge_use(self, edge_id: str) -> None:
        self.edge_use_counts[edge_id] = self.edge_use_counts.get(edge_id, 0) + 1

    def reset_balancer(self) -> None:
        self.edge_use_counts.clear()

    def _balanced_sibling(self, edge_id: str, vehicle_class: str) -> str:
        key = self._edge_group_key(edge_id)
        if key is None:
            return edge_id
        siblings = [
            sibling
            for sibling in self.sibling_physical_lanes.get(key, [])
            if sibling in self.planner.edges
            and self.planner.edges[sibling].length_m >= 20.0
            and _edge_allows_class(self.planner.edges[sibling], vehicle_class)
        ]
        if len(siblings) < 2:
            return edge_id
        min_count = min(self.edge_use_counts.get(sibling, 0) for sibling in siblings)
        least_used = [sibling for sibling in siblings if self.edge_use_counts.get(sibling, 0) == min_count]
        return self.rng.choice(least_used)

    def _choose_next_edge(self, candidates: list[str], vehicle_class: str, *, balance: bool) -> str:
        if not balance:
            return self.rng.choice(candidates)
        valid = [
            edge_id
            for edge_id in candidates
            if edge_id in self.planner.edges and _edge_allows_class(self.planner.edges[edge_id], vehicle_class)
        ]
        if not valid:
            valid = candidates
        min_count = min(self.edge_use_counts.get(edge_id, 0) for edge_id in valid)
        least_used = [edge_id for edge_id in valid if self.edge_use_counts.get(edge_id, 0) == min_count]
        return self.rng.choice(least_used)

    def path_from_edge(
        self,
        edge_id: str,
        vehicle_class: str,
        *,
        min_edges: int = 4,
        max_edges: int = 12,
        balance: bool = False,
    ) -> list[str]:
        if edge_id not in self.planner.edges:
            return self.random_path(vehicle_class, min_edges=min_edges, max_edges=max_edges)
        path = [edge_id]
        if balance:
            self._record_edge_use(edge_id)
        current = edge_id
        wanted = self.rng.randint(min_edges, max_edges)
        for _ in range(wanted - 1):
            candidates = [
                edge_id
                for edge_id in self.planner.adjacency.get(current, [])
                if edge_id in self.planner.edges and _edge_allows_class(self.planner.edges[edge_id], vehicle_class)
            ]
            if not candidates:
                break
            current = self._choose_next_edge(candidates, vehicle_class, balance=balance)
            if current in path[-3:]:
                break
            path.append(current)
            if balance:
                self._record_edge_use(current)
        return path

    def random_path(self, vehicle_class: str, *, min_edges: int = 4, max_edges: int = 14) -> list[str]:
        edges = self.edges_by_class.get(vehicle_class) or self.edges_by_class["passenger"]
        for _ in range(80):
            start = self.rng.choice(edges)
            if start.length_m < 20.0:
                continue
            start_edge_id = self._balanced_sibling(start.edge_id, vehicle_class)
            path = self.path_from_edge(
                start_edge_id,
                vehicle_class,
                min_edges=min_edges,
                max_edges=max_edges,
                balance=True,
            )
            if len(path) >= 2:
                return path
        return [self.rng.choice(edges).edge_id]


@dataclass
class RestoreAction:
    kind: str
    target_id: str
    end_s: float
    payload: Any


class TrafficRunner:
    def __init__(
        self,
        *,
        planner: SumoGroundFlowPlanner,
        incident_plan: dict[str, Any],
        output_dir: Path,
        sumo_bin: Path,
        duration_s: float,
        warmup_s: float,
        step_length_s: float,
        sample_period_s: float,
        max_vehicles: int,
        seed: int,
        roi_lane_coverage_anchors: Sequence[RoiLaneCoverageAnchor] = (),
        semantic_vehicle_instructions: Sequence[dict[str, Any]] = (),
        explicit_vehicle_plan: dict[str, Any] | None = None,
        explicit_vehicle_ids: set[str] | None = None,
    ) -> None:
        self.planner = planner
        self.incident_plan = incident_plan
        incident_step_s = float(step_length_s)
        if not math.isfinite(incident_step_s) or incident_step_s <= 0:
            raise RuntimeError(f"invalid SUMO incident step length: {step_length_s!r}")
        all_red_windows: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        for incident in incident_records(incident_plan):
            if incident["accident_class"] != "traffic_light_all_red_fault":
                continue
            target = (incident.get("traffic_light") or {}).get("traffic_light_id")
            if not isinstance(target, str) or not target:
                raise RuntimeError(f"all-red incident lacks a controller target: {incident['incident_id']}")
            start_s, end_s = float(incident["start_s"]), float(incident["end_s"])
            if (not math.isfinite(start_s) or not math.isfinite(end_s) or
                    start_s < 0 or start_s >= end_s):
                raise RuntimeError(f"all-red incident has invalid time bounds: {incident['incident_id']}")
            apply_step = math.ceil((start_s - 1e-9) / incident_step_s)
            restore_step = math.ceil((end_s - 1e-9) / incident_step_s)
            if apply_step >= restore_step:
                raise RuntimeError(
                    f"all-red incident has no observable SUMO step: {incident['incident_id']}")
            all_red_windows[target].append((start_s, end_s, incident["incident_id"]))
        for target, windows in all_red_windows.items():
            windows.sort()
            for previous, current in zip(windows, windows[1:]):
                previous_restore_step = math.ceil((previous[1] - 1e-9) / incident_step_s)
                current_apply_step = math.ceil((current[0] - 1e-9) / incident_step_s)
                if current[0] <= previous[1] or current_apply_step <= previous_restore_step:
                    raise RuntimeError(
                        f"all-red incidents overlap or share a SUMO step on controller {target}: "
                        f"{previous[2]}, {current[2]}"
                    )
        self.lane_incident_execution = LaneIncidentExecution()
        self.output_dir = output_dir
        self.sumo_bin = sumo_bin
        self.capture_duration_s = float(duration_s)
        self.warmup_s = float(warmup_s)
        self.duration_s = self.warmup_s + self.capture_duration_s
        self.step_length_s = float(step_length_s)
        self.sample_period_s = float(sample_period_s)
        self.max_vehicles = int(max_vehicles)
        self.rng = random.Random(seed)
        self.route_factory = RouteFactory(planner, self.rng)
        self.roi_lane_coverage_anchors = tuple(
            anchor
            for anchor in roi_lane_coverage_anchors
            if anchor.edge_id in self.planner.edges and _edge_allows_class(self.planner.edges[anchor.edge_id], "passenger")
        )
        self.explicit_vehicle_plan = dict(explicit_vehicle_plan or {})
        self.explicit_vehicle_ids = set(explicit_vehicle_ids or set())
        self.explicit_role_by_vehicle_id = {
            str(vehicle.get("vehicle_id") or ""): str(vehicle.get("role") or "")
            for vehicle in self.explicit_vehicle_plan.get("vehicles") or []
            if str(vehicle.get("vehicle_id") or "")
        }
        self.explicit_source_by_vehicle_id = {
            str(vehicle.get("vehicle_id") or ""): str(vehicle.get("source_entity_id") or "")
            for vehicle in self.explicit_vehicle_plan.get("vehicles") or []
            if str(vehicle.get("vehicle_id") or "") and str(vehicle.get("source_entity_id") or "")
        }
        self.explicit_vehicle_metadata_by_id = {
            str(vehicle.get("vehicle_id") or ""): dict(vehicle)
            for vehicle in self.explicit_vehicle_plan.get("vehicles") or []
            if str(vehicle.get("vehicle_id") or "")
        }
        self.explicit_semantic_vehicle_ids = {
            vehicle_id
            for vehicle_id, metadata in self.explicit_vehicle_metadata_by_id.items()
            if str(metadata.get("traffic_role") or "") == "semantic_vehicle"
        }
        self.explicit_source_presence_required_vehicle_ids = {
            vehicle_id
            for vehicle_id, metadata in self.explicit_vehicle_metadata_by_id.items()
            if metadata.get("source_presence_required") is True
        }
        self.resolved_route_edges_by_vehicle_id: dict[str, list[str]] = {}
        self.resolved_route_first_edge_by_vehicle_id: dict[str, str] = {}
        self.resolved_route_depart_pos_by_vehicle_id: dict[str, float] = {}
        all_semantic_vehicle_instructions = tuple(sorted(semantic_vehicle_instructions, key=lambda item: (float(item.get("time_s") or 0.0), int(item.get("sequence") or 0))))
        self.semantic_vehicle_instructions = tuple(
            item
            for item in all_semantic_vehicle_instructions
            if str(item.get("instruction_type") or "") != "vehicle.moveToXY"
        )
        self.post_step_move_to_xy_instructions = tuple(
            item
            for item in all_semantic_vehicle_instructions
            if str(item.get("instruction_type") or "") == "vehicle.moveToXY"
        )
        self.semantic_instruction_cursor = 0
        self.post_step_move_to_xy_cursor = 0
        instruction_vehicle_ids = {
            str(item.get("internal_vehicle_id") or "")
            for item in all_semantic_vehicle_instructions
            if str(item.get("internal_vehicle_id") or "")
        }
        self.semantic_vehicle_ids = set(self.explicit_semantic_vehicle_ids) if self.explicit_vehicle_ids else instruction_vehicle_ids
        self.semantic_source_by_vehicle_id = {
            str(item.get("internal_vehicle_id") or ""): str(item.get("source_entity_id") or "")
            for item in all_semantic_vehicle_instructions
            if str(item.get("internal_vehicle_id") or "") and str(item.get("source_entity_id") or "")
        }
        self.semantic_episode_by_vehicle_id = {
            str(item.get("internal_vehicle_id") or ""): str(item.get("episode_id") or "")
            for item in all_semantic_vehicle_instructions
            if str(item.get("internal_vehicle_id") or "") and str(item.get("episode_id") or "")
        }
        self.semantic_vehicle_id_by_source_entity = {
            str(item.get("source_entity_id") or ""): str(item.get("internal_vehicle_id") or "")
            for item in all_semantic_vehicle_instructions
            if str(item.get("source_entity_id") or "") and str(item.get("internal_vehicle_id") or "")
        }
        if self.explicit_vehicle_ids:
            self.semantic_vehicle_id_by_source_entity = {
                source_entity_id: vehicle_id
                for vehicle_id, source_entity_id in self.explicit_source_by_vehicle_id.items()
                if vehicle_id in self.explicit_semantic_vehicle_ids and source_entity_id
            }
        for vehicle_id, source_entity_id in self.explicit_source_by_vehicle_id.items():
            self.semantic_source_by_vehicle_id.setdefault(vehicle_id, source_entity_id)
            self.semantic_episode_by_vehicle_id.setdefault(vehicle_id, str(self.explicit_vehicle_plan.get("episode_id") or ""))
            if vehicle_id in self.explicit_semantic_vehicle_ids or not self.explicit_vehicle_ids:
                self.semantic_vehicle_id_by_source_entity.setdefault(source_entity_id, vehicle_id)
        self.semantic_metadata_by_vehicle_id: dict[str, dict[str, Any]] = {}
        for item in all_semantic_vehicle_instructions:
            args = dict(item.get("args") or {})
            vehicle_id = str(item.get("internal_vehicle_id") or args.get("vehID") or "")
            metadata = dict(args.get("metadata") or {})
            if vehicle_id and metadata:
                self.semantic_metadata_by_vehicle_id[vehicle_id] = metadata
        for vehicle in self.explicit_vehicle_plan.get("vehicles") or []:
            vehicle_id = str(vehicle.get("vehicle_id") or "")
            if not vehicle_id:
                continue
            self.semantic_metadata_by_vehicle_id.setdefault(
                vehicle_id,
                {
                    "explicit_vehicle_plan": True,
                    "vehicle_source_policy": VEHICLE_SOURCE_POLICY,
                    "role": vehicle.get("role"),
                    "traffic_role": vehicle.get("traffic_role"),
                    "direction_role": vehicle.get("direction_role"),
                    "traffic_slot_index": vehicle.get("traffic_slot_index"),
                    "expected_core_entry_tick": vehicle.get("expected_core_entry_tick"),
                    "expected_core_exit_tick": vehicle.get("expected_core_exit_tick"),
                    "route_source": "sumo_findRoute",
                    "protected_window_role": "allowed_semantic_vehicle" if str(vehicle.get("traffic_role") or "") == "semantic_vehicle" else "background_forbidden_from_protected_edges",
                },
            )
        self.semantic_instruction_failures: list[dict[str, Any]] = []
        self.filtered_background_vehicle_route_failures: list[dict[str, Any]] = []
        self.filtered_background_vehicle_ids: set[str] = set()
        self.accepted_background_vehicle_ids: set[str] = set()
        self.inactive_background_vehicle_ids: set[str] = set()
        self.explicit_vehicle_runtime_state: dict[str, dict[str, Any]] = {}
        self.pending_background_vehicle_controls: dict[tuple[str, str], dict[str, Any]] = {}
        self.background_vehicle_control_failures: list[dict[str, Any]] = []
        self.incident_control_skips: list[dict[str, Any]] = []
        self.background_ids: set[str] = set()
        self.controlled_ids: set[str] = set()
        self.seen_explicit_vehicle_ids: set[str] = set()
        self.departed_explicit_vehicle_ids: set[str] = set()
        self.arrived_explicit_vehicle_ids: set[str] = set()
        self.applied_incidents: set[str] = set()
        self.restores: list[RestoreAction] = []
        self.route_counter = 0
        self.resolved_route_endpoint_use_counts: dict[str, int] = {}
        self.active_incident_by_id = {str(item["incident_id"]): item for item in incident_records(incident_plan)}

    def _is_semantic_contract_vehicle(self, vehicle_id: str) -> bool:
        if self.explicit_vehicle_ids:
            return str(vehicle_id) in self.explicit_semantic_vehicle_ids
        return bool(vehicle_id)

    def _explicit_background_vehicle_ids(self) -> set[str]:
        return {
            vehicle_id
            for vehicle_id, metadata in self.explicit_vehicle_metadata_by_id.items()
            if str(metadata.get("traffic_role") or "") == "deterministic_background_vehicle"
        }

    def _accepted_background_vehicle_target(self) -> int:
        traffic_profile = dict(self.explicit_vehicle_plan.get("traffic_profile") or {})
        target = int(traffic_profile.get("accepted_background_vehicle_target") or 0)
        if target > 0:
            return target
        candidate_count = int(self.explicit_vehicle_plan.get("minimum_background_vehicle_count") or 0)
        return int(max(1, round(candidate_count * 0.82))) if candidate_count > 0 else 0

    def run(self) -> dict[str, Any]:
        ensure_sumo_tools_path()
        import traci  # type: ignore

        cmd = [
            str(self.sumo_bin),
            "-n",
            str(self.planner.net_xml),
            "--step-length",
            str(self.step_length_s),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--time-to-teleport",
            "-1",
            "--collision.action",
            "warn",
        ]
        traci.start(cmd)
        try:
            self._ensure_vehicle_types(traci)
            frames_path = self.output_dir / "sumo_traffic_frames.jsonl"
            frames_path.parent.mkdir(parents=True, exist_ok=True)
            sample_every_steps = max(1, int(round(self.sample_period_s / self.step_length_s)))
            total_steps = int(round(self.duration_s / self.step_length_s))
            sample_count = 0
            with frames_path.open("w", encoding="utf-8") as handle:
                self._apply_due_semantic_vehicle_instructions(traci, 0.0)
                self._apply_due_incidents(traci, float(traci.simulation.getTime()))
                self._restore_due_actions(traci, float(traci.simulation.getTime()))
                self._note_explicit_vehicle_states(traci)
                self._write_frame(handle, traci, tick=0)
                sample_count += 1
                for step in range(1, total_steps + 1):
                    sim_time_after_step = round(step * self.step_length_s, 9)
                    self._apply_due_semantic_vehicle_instructions(traci, sim_time_after_step)
                    self._assert_no_unplanned_active_vehicles(traci)
                    self._apply_due_post_step_move_to_xy_instructions(traci, sim_time_after_step)
                    traci.simulationStep(sim_time_after_step)
                    self._apply_due_incidents(traci, float(traci.simulation.getTime()))
                    self._note_explicit_vehicle_states(traci)
                    self._apply_pending_background_vehicle_controls(traci)
                    self._restore_due_actions(traci, float(traci.simulation.getTime()))
                    self._trim_vehicle_cap(traci)
                    if step % sample_every_steps == 0:
                        self._write_frame(handle, traci, tick=step)
                        sample_count += 1
            if self.semantic_instruction_failures:
                preview = json.dumps(self.semantic_instruction_failures[:8], ensure_ascii=False, sort_keys=True)
                raise RuntimeError(
                    f"{len(self.semantic_instruction_failures)} semantic SUMO vehicle instructions failed; "
                    f"first failures: {preview}"
                )
            if self.explicit_vehicle_ids:
                missing_required_source = sorted(
                    self.explicit_source_presence_required_vehicle_ids
                    - self.seen_explicit_vehicle_ids
                )
                if missing_required_source:
                    missing_runtime_states = {
                        vehicle_id: dict(self._runtime_state_for(vehicle_id))
                        for vehicle_id in missing_required_source
                    }
                    raise RuntimeError(
                        f"{VEHICLE_SOURCE_POLICY} source-required vehicles never became active in "
                        f"SUMO frames: {missing_runtime_states}"
                    )
                incomplete_required_source_lifecycles: list[dict[str, Any]] = []
                for vehicle_id in sorted(self.explicit_source_presence_required_vehicle_ids):
                    state = self._runtime_state_for(vehicle_id)
                    first_active_s = state.get("first_active_s")
                    last_active_s = state.get("last_active_s")
                    if (
                        first_active_s is None
                        or last_active_s is None
                        or float(first_active_s) > float(self.warmup_s) + 1e-9
                        or float(last_active_s) < float(self.duration_s) - 1e-9
                    ):
                        incomplete_required_source_lifecycles.append(
                            {
                                "vehicle_id": vehicle_id,
                                "first_active_s": first_active_s,
                                "last_active_s": last_active_s,
                                "required_active_interval_s": [
                                    round(float(self.warmup_s), 6),
                                    round(float(self.duration_s), 6),
                                ],
                            }
                        )
                if incomplete_required_source_lifecycles:
                    raise RuntimeError(
                        f"{VEHICLE_SOURCE_POLICY} source-required vehicles must remain active for the "
                        f"full capture interval: {incomplete_required_source_lifecycles[:20]}"
                    )
                missing_semantic = sorted(self.explicit_semantic_vehicle_ids - self.seen_explicit_vehicle_ids)
                if missing_semantic:
                    raise RuntimeError(
                        f"{VEHICLE_SOURCE_POLICY} semantic vehicles never became active in SUMO frames: {missing_semantic[:20]}"
                    )
                background_vehicle_ids = self._explicit_background_vehicle_ids()
                self.accepted_background_vehicle_ids = (
                    background_vehicle_ids & self.seen_explicit_vehicle_ids
                ) - self.filtered_background_vehicle_ids
                self.inactive_background_vehicle_ids = (
                    background_vehicle_ids - self.seen_explicit_vehicle_ids
                ) - self.filtered_background_vehicle_ids
            return self._manifest(sample_count=sample_count)
        finally:
            _write_json(self.output_dir / "lane_incident_execution.json", {
                "events": self.lane_incident_execution.events,
                "interpretation": "Measured lane speed controls; road-closure semantics require the governed typed incident contract.",
            })
            traci.close(False)

    def _ensure_vehicle_types(self, traci: Any) -> None:
        for source, type_id, vehicle_class, color in (
            ("DEFAULT_VEHTYPE", "aero_passenger", "passenger", (40, 160, 255, 255)),
            ("DEFAULT_VEHTYPE", "aero_delivery", "delivery", (255, 180, 40, 255)),
            ("DEFAULT_VEHTYPE", "aero_emergency", "emergency", (255, 40, 40, 255)),
            ("DEFAULT_VEHTYPE", "aero_police", "emergency", (30, 90, 255, 255)),
            ("DEFAULT_VEHTYPE", "aero_ambulance", "emergency", (40, 210, 80, 255)),
        ):
            try:
                traci.vehicletype.copy(source, type_id)
            except Exception:
                pass
            try:
                traci.vehicletype.setVehicleClass(type_id, vehicle_class)
                traci.vehicletype.setColor(type_id, color)
                traci.vehicletype.setLength(type_id, DEFAULT_VEHICLE_LENGTH_M)
                traci.vehicletype.setWidth(type_id, DEFAULT_VEHICLE_WIDTH_M)
                traci.vehicletype.setMinGap(type_id, 2.5)
            except Exception:
                pass

    def _new_route(self, traci: Any, edges: list[str]) -> str:
        route_id = f"sumo_route_{self.route_counter:06d}"
        self.route_counter += 1
        traci.route.add(route_id, edges)
        return route_id

    def _vehicle_type_for_class(self, vehicle_class: str, *, emergency: bool = False) -> str:
        if emergency:
            return "aero_emergency"
        if vehicle_class == "delivery":
            return "aero_delivery"
        return "aero_passenger"

    def _safe_depart_pos(self, route_edges: Sequence[str], requested_pos_m: float) -> float:
        edge = self.planner.edges.get(str(route_edges[0])) if route_edges else None
        lower = MIN_DEPART_FRONT_POS_M
        if edge is None:
            return max(lower, float(requested_pos_m))
        length_m = max(0.0, float(edge.length_m))
        if length_m <= lower + STOP_END_MARGIN_M:
            return max(0.0, length_m * 0.5)
        upper = max(lower, length_m - STOP_END_MARGIN_M)
        return max(lower, min(float(requested_pos_m), upper))

    def _spawn_vehicle(
        self,
        traci: Any,
        *,
        vehicle_id: str,
        route_edges: list[str],
        vehicle_class: str,
        depart_pos_m: float,
        depart_speed_mps: float | str,
        emergency: bool = False,
        controlled: bool = False,
    ) -> bool:
        if self.explicit_vehicle_ids and vehicle_id not in self.explicit_vehicle_ids:
            raise RuntimeError(f"{VEHICLE_SOURCE_POLICY} forbids runtime spawn of unplanned vehicle {vehicle_id!r}")
        try:
            route_id = self._new_route(traci, route_edges)
            depart_pos = self._safe_depart_pos(route_edges, float(depart_pos_m))
            depart_speed = depart_speed_mps if isinstance(depart_speed_mps, str) else f"{max(0.0, float(depart_speed_mps)):.3f}"
            traci.vehicle.addFull(
                vehicle_id,
                route_id,
                typeID=self._vehicle_type_for_class(vehicle_class, emergency=emergency),
                depart="now",
                departLane="best",
                departPos=f"{depart_pos:.3f}",
                departSpeed=str(depart_speed),
            )
            if controlled:
                self.controlled_ids.add(vehicle_id)
            else:
                self.background_ids.add(vehicle_id)
            return True
        except Exception:
            return False

    def _active_or_loaded(self, traci: Any, vehicle_id: str) -> bool:
        state = self._runtime_state_for(vehicle_id)
        return bool(state.get("addFull_succeeded")) or vehicle_id in self._active_vehicle_ids(traci)

    def _active_vehicle_ids(self, traci: Any) -> set[str]:
        try:
            return {str(vehicle_id) for vehicle_id in traci.vehicle.getIDList()}
        except Exception:
            return set()

    def _runtime_state_for(self, vehicle_id: str) -> dict[str, Any]:
        return self.explicit_vehicle_runtime_state.setdefault(
            vehicle_id,
            {
                "route_find_attempt_count": 0,
                "route_find_truncated": False,
                "route_added": False,
                "addFull_attempted": False,
                "addFull_succeeded": False,
                "addFull_error": "",
                "first_loaded_s": None,
                "last_loaded_s": None,
                "first_departed_s": None,
                "last_departed_s": None,
                "first_active_s": None,
                "last_active_s": None,
                "first_arrived_s": None,
                "last_arrived_s": None,
                "pending_control_count": 0,
                "pending_control_types": [],
            },
        )

    def _note_explicit_vehicle_states(self, traci: Any) -> None:
        if not self.explicit_vehicle_ids:
            return
        sim_time_s = round(float(traci.simulation.getTime()), 6)
        try:
            active = {str(vehicle_id) for vehicle_id in traci.vehicle.getIDList()}
        except Exception:
            active = set()
        try:
            departed = {str(vehicle_id) for vehicle_id in traci.simulation.getDepartedIDList()}
        except Exception:
            departed = set()
        try:
            arrived = {str(vehicle_id) for vehicle_id in traci.simulation.getArrivedIDList()}
        except Exception:
            arrived = set()
        self.departed_explicit_vehicle_ids.update(departed & self.explicit_vehicle_ids)
        self.arrived_explicit_vehicle_ids.update(arrived & self.explicit_vehicle_ids)
        for vehicle_id in sorted((active | departed | arrived) & self.explicit_vehicle_ids):
            state = self._runtime_state_for(vehicle_id)
            if vehicle_id in departed:
                state["first_departed_s"] = sim_time_s if state["first_departed_s"] is None else state["first_departed_s"]
                state["last_departed_s"] = sim_time_s
            if vehicle_id in active:
                state["first_active_s"] = sim_time_s if state["first_active_s"] is None else state["first_active_s"]
                state["last_active_s"] = sim_time_s
            if vehicle_id in arrived:
                state["first_arrived_s"] = sim_time_s if state["first_arrived_s"] is None else state["first_arrived_s"]
                state["last_arrived_s"] = sim_time_s

    def _queue_background_vehicle_control(self, item: dict[str, Any], *, sim_time_s: float) -> None:
        args = dict(item.get("args") or {})
        vehicle_id = str(args.get("vehID") or item.get("internal_vehicle_id") or "")
        instruction_type = str(item.get("instruction_type") or "")
        if not vehicle_id or not self._is_filterable_background_vehicle(vehicle_id):
            return
        if instruction_type not in {"vehicle.setSpeed", "vehicle.setMaxSpeed", "vehicle.setStop", "vehicle.resume"}:
            return
        expires_s = None
        if instruction_type == "vehicle.setStop":
            expires_s = float(sim_time_s) + PENDING_STOP_RETRY_S
        queued = copy.deepcopy(item)
        queued["_queued_s"] = float(sim_time_s)
        queued["_expires_s"] = expires_s
        self.pending_background_vehicle_controls[(vehicle_id, instruction_type)] = queued
        state = self._runtime_state_for(vehicle_id)
        state["pending_control_count"] = int(state.get("pending_control_count") or 0) + 1
        control_types = list(state.get("pending_control_types") or [])
        if instruction_type not in control_types:
            control_types.append(instruction_type)
        state["pending_control_types"] = control_types

    def _record_background_control_failure(self, item: dict[str, Any], message: str) -> None:
        args = dict(item.get("args") or {})
        vehicle_id = str(args.get("vehID") or item.get("internal_vehicle_id") or "")
        self.background_vehicle_control_failures.append(
            {
                "vehicle_id": vehicle_id,
                "instruction_type": item.get("instruction_type"),
                "time_s": item.get("time_s"),
                "tick": item.get("tick"),
                "episode_tick": item.get("episode_tick"),
                "message": message,
            }
        )

    def _background_stop_payload(self, vehicle_id: str, args: dict[str, Any]) -> tuple[str, float, int] | None:
        resolved_route = self.resolved_route_edges_by_vehicle_id.get(vehicle_id, [])
        requested_edge = str(args.get("edgeID") or "")
        edge_id = requested_edge
        pos = float(args.get("pos") or 0.0)
        lane_index = int(args.get("laneIndex") or 0)
        matched_stop_candidate = False
        for candidate in args.get("stop_candidates") or []:
            candidate_dict = dict(candidate)
            candidate_edge = str(candidate_dict.get("edgeID") or "")
            if resolved_route and candidate_edge in resolved_route:
                edge_id = candidate_edge
                pos = float(candidate_dict.get("pos") or pos)
                lane_index = int(candidate_dict.get("laneIndex") or lane_index)
                matched_stop_candidate = True
                break
        if (args.get("stop_candidates") or []) and not matched_stop_candidate:
            return None
        if resolved_route and not matched_stop_candidate and requested_edge not in resolved_route:
            edge_id = resolved_route[0]
            pos = self._safe_depart_pos(resolved_route, 0.0)
        return edge_id, pos, lane_index

    def _stop_edge_is_downstream(self, traci: Any, vehicle_id: str, edge_id: str) -> bool:
        try:
            route = [str(edge) for edge in traci.vehicle.getRoute(vehicle_id)]
            route_index = int(traci.vehicle.getRouteIndex(vehicle_id))
        except Exception:
            return True
        return any(index >= route_index for index, candidate in enumerate(route) if candidate == edge_id)

    def _apply_background_control_now(self, traci: Any, item: dict[str, Any]) -> bool:
        args = dict(item.get("args") or {})
        vehicle_id = str(args.get("vehID") or item.get("internal_vehicle_id") or "")
        instruction_type = str(item.get("instruction_type") or "")
        if not vehicle_id or vehicle_id not in traci.vehicle.getIDList():
            return False
        try:
            if instruction_type == "vehicle.setSpeed":
                traci.vehicle.setSpeed(vehicle_id, float(args["speed"] if args.get("speed") is not None else -1.0))
            elif instruction_type == "vehicle.setMaxSpeed":
                traci.vehicle.setMaxSpeed(vehicle_id, float(args.get("speed") or 0.0))
            elif instruction_type == "vehicle.resume":
                traci.vehicle.resume(vehicle_id)
            elif instruction_type == "vehicle.setStop":
                payload = self._background_stop_payload(vehicle_id, args)
                if payload is None:
                    self._record_background_control_failure(item, "vehicle.setStop skipped because no stop candidate is on the resolved SUMO route")
                    return True
                kwargs: dict[str, Any] = {}
                if args.get("until") not in (None, ""):
                    kwargs["until"] = float(args.get("until"))
                if args.get("duration") not in (None, ""):
                    kwargs["duration"] = float(args.get("duration"))
                edge_id, pos, lane_index = payload
                if not self._stop_edge_is_downstream(traci, vehicle_id, edge_id):
                    self._record_background_control_failure(item, "vehicle.setStop skipped because stop edge is no longer downstream")
                    return True
                traci.vehicle.setStop(vehicle_id, edge_id, pos, lane_index, **kwargs)
            else:
                return True
            return True
        except Exception as exc:
            self._record_background_control_failure(item, f"{instruction_type} retry raised {type(exc).__name__}: {exc}")
            return True

    def _apply_pending_background_vehicle_controls(self, traci: Any) -> None:
        if not self.pending_background_vehicle_controls:
            return
        sim_time_s = float(traci.simulation.getTime())
        active = self._active_vehicle_ids(traci)
        pending: dict[tuple[str, str], dict[str, Any]] = {}
        for key, item in self.pending_background_vehicle_controls.items():
            vehicle_id, _instruction_type = key
            expires_s = item.get("_expires_s")
            if expires_s is not None and sim_time_s > float(expires_s) + 1e-9:
                self._record_background_control_failure(item, "pending control expired before vehicle became active")
                continue
            if vehicle_id in active:
                self._apply_background_control_now(traci, item)
                continue
            pending[key] = item
        self.pending_background_vehicle_controls = pending

    def _record_semantic_instruction_failure(self, item: dict[str, Any], message: str) -> None:
        args = dict(item.get("args") or {})
        self.semantic_instruction_failures.append(
            {
                "time_s": item.get("time_s"),
                "tick": item.get("tick"),
                "time_axis": item.get("time_axis"),
                "episode_tick": item.get("episode_tick"),
                "episode_id": item.get("episode_id"),
                "source_entity_id": item.get("source_entity_id"),
                "internal_vehicle_id": item.get("internal_vehicle_id") or args.get("vehID"),
                "event_id": item.get("event_id"),
                "intent": item.get("intent"),
                "target_role": item.get("target_role"),
                "instruction_type": item.get("instruction_type"),
                "message": message,
            }
        )

    def _is_filterable_background_vehicle(self, vehicle_id: str) -> bool:
        if not vehicle_id or vehicle_id not in self.explicit_vehicle_ids:
            return False
        metadata = self.explicit_vehicle_metadata_by_id.get(vehicle_id, {})
        return (
            str(metadata.get("traffic_role") or "")
            == "deterministic_background_vehicle"
            and metadata.get("source_presence_required") is not True
        )

    def _semantic_corridor_edge(self, vehicle_id: str) -> str:
        metadata = self.explicit_vehicle_metadata_by_id.get(vehicle_id, {})
        if str(metadata.get("traffic_role") or "") != "semantic_vehicle":
            return ""
        corridor = dict(metadata.get("semantic_corridor") or {})
        edge_id = str(corridor.get("edge_id") or "")
        return edge_id if edge_id in self.planner.edges else ""

    def _record_background_route_filter(
        self,
        item: dict[str, Any],
        *,
        reason: str,
        details: Any,
    ) -> None:
        args = dict(item.get("args") or {})
        vehicle_id = str(args.get("vehID") or item.get("internal_vehicle_id") or args.get("vehicle_id") or "")
        if not vehicle_id:
            return
        self.filtered_background_vehicle_ids.add(vehicle_id)
        self.filtered_background_vehicle_route_failures.append(
            {
                "vehicle_id": vehicle_id,
                "reason": reason,
                "time_s": item.get("time_s"),
                "tick": item.get("tick"),
                "episode_tick": item.get("episode_tick"),
                "traffic_slot_index": self.explicit_vehicle_metadata_by_id.get(vehicle_id, {}).get("traffic_slot_index"),
                "from_edge": args.get("from_edge"),
                "to_edge": args.get("to_edge"),
                "details": details,
            }
        )

    def _inactive_background_vehicle_details(self) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        for vehicle_id in sorted(self.inactive_background_vehicle_ids):
            state = dict(self.explicit_vehicle_runtime_state.get(vehicle_id) or {})
            if not state.get("route_added"):
                reason = "route_not_added"
            elif not state.get("addFull_attempted"):
                reason = "addFull_not_attempted"
            elif state.get("addFull_error"):
                reason = "addFull_failed"
            elif state.get("first_loaded_s") is not None and state.get("first_active_s") is None:
                reason = "loaded_pending_never_departed"
            elif state.get("first_departed_s") is not None and state.get("first_active_s") is None:
                reason = "departed_but_never_active_in_sampled_frames"
            elif state.get("first_active_s") is not None:
                reason = "active_but_never_written_to_sampled_frame"
            else:
                reason = "addFull_succeeded_but_never_loaded_or_active"
            metadata = self.explicit_vehicle_metadata_by_id.get(vehicle_id, {})
            details.append(
                {
                    "vehicle_id": vehicle_id,
                    "reason": reason,
                    "traffic_slot_index": metadata.get("traffic_slot_index"),
                    "traffic_coverage_scope": metadata.get("traffic_coverage_scope"),
                    "expected_core_entry_tick": metadata.get("expected_core_entry_tick"),
                    "expected_core_exit_tick": metadata.get("expected_core_exit_tick"),
                    "route_find_attempt_count": state.get("route_find_attempt_count", 0),
                    "route_find_truncated": bool(state.get("route_find_truncated")),
                    "route_added": bool(state.get("route_added")),
                    "addFull_attempted": bool(state.get("addFull_attempted")),
                    "addFull_succeeded": bool(state.get("addFull_succeeded")),
                    "addFull_error": state.get("addFull_error", ""),
                    "first_loaded_s": state.get("first_loaded_s"),
                    "last_loaded_s": state.get("last_loaded_s"),
                    "first_departed_s": state.get("first_departed_s"),
                    "last_departed_s": state.get("last_departed_s"),
                    "first_active_s": state.get("first_active_s"),
                    "last_active_s": state.get("last_active_s"),
                    "first_arrived_s": state.get("first_arrived_s"),
                    "last_arrived_s": state.get("last_arrived_s"),
                    "pending_control_count": state.get("pending_control_count", 0),
                    "pending_control_types": state.get("pending_control_types", []),
                    "off_roi_speed_hold_skipped": bool(state.get("off_roi_speed_hold_skipped")),
                }
            )
        return details

    def _requires_active_semantic_vehicle(self, item: dict[str, Any]) -> bool:
        if item.get("active_required") is not None:
            return bool(item.get("active_required"))
        return str(item.get("instruction_type") or "") in {
            "vehicle.setRoute",
            "vehicle.slowDown",
            "vehicle.setStop",
            "vehicle.setSignals",
            "vehicle.setMaxSpeed",
            "vehicle.resume",
            "vehicle.remove",
            "vehicle.setParameter",
            "vehicle.moveToXY",
        } and str(item.get("time_axis") or "capture") == "capture"

    def _apply_due_semantic_vehicle_instructions(self, traci: Any, sim_time_s: float) -> None:
        while self.semantic_instruction_cursor < len(self.semantic_vehicle_instructions):
            item = self.semantic_vehicle_instructions[self.semantic_instruction_cursor]
            if float(item.get("time_s") or 0.0) > float(sim_time_s) + 1e-9:
                break
            self._apply_semantic_vehicle_instruction(traci, item)
            self.semantic_instruction_cursor += 1

    def _apply_due_post_step_move_to_xy_instructions(self, traci: Any, sim_time_s: float) -> None:
        while self.post_step_move_to_xy_cursor < len(self.post_step_move_to_xy_instructions):
            item = self.post_step_move_to_xy_instructions[self.post_step_move_to_xy_cursor]
            if float(item.get("time_s") or 0.0) > float(sim_time_s) + 1e-9:
                break
            self._apply_semantic_vehicle_instruction(traci, item)
            self.post_step_move_to_xy_cursor += 1

    def _apply_semantic_vehicle_instruction(self, traci: Any, item: dict[str, Any]) -> None:
        instruction_type = str(item.get("instruction_type") or "")
        args = dict(item.get("args") or {})
        vehicle_id = str(args.get("vehID") or item.get("internal_vehicle_id") or "")
        source_entity_id = str(item.get("source_entity_id") or self.semantic_source_by_vehicle_id.get(vehicle_id, ""))
        episode_id = str(item.get("episode_id") or self.semantic_episode_by_vehicle_id.get(vehicle_id, ""))
        try:
            if instruction_type == "route.findRoute":
                route_id = str(args.get("route_id") or args.get("routeID") or "")
                from_edge = str(args.get("from_edge") or "")
                to_edge = str(args.get("to_edge") or "")
                type_id = str(args.get("typeID") or args.get("type_id") or "aero_passenger")
                forbidden_edges = {str(edge_id) for edge_id in args.get("forbidden_edges") or [] if str(edge_id)}
                required_edge_order = _ordered_unique_strings(args.get("required_edges") or [])
                required_edges = set(required_edge_order)
                from_edge_depart_pos_by_edge: dict[str, float] = {}
                for edge_id, depart_pos in dict(args.get("from_edge_depart_pos_by_edge") or {}).items():
                    try:
                        from_edge_depart_pos_by_edge[str(edge_id)] = float(depart_pos)
                    except (TypeError, ValueError):
                        continue
                resolved_edges: list[str] = []
                rejected: list[dict[str, Any]] = []
                semantic_corridor_edge = self._semantic_corridor_edge(vehicle_id)
                if semantic_corridor_edge and route_id and args.get("semantic_bootstrap_edges"):
                    bootstrap_edges = [str(edge_id) for edge_id in args.get("semantic_bootstrap_edges") or [] if str(edge_id)]
                    route_edges = bootstrap_edges or [semantic_corridor_edge]
                    bootstrap_depart_pos = args.get("semantic_bootstrap_depart_pos_m")
                    try:
                        bootstrap_depart_pos_m = float(bootstrap_depart_pos) if bootstrap_depart_pos is not None else None
                    except (TypeError, ValueError):
                        bootstrap_depart_pos_m = None
                    if bootstrap_depart_pos_m is None and semantic_corridor_edge in from_edge_depart_pos_by_edge:
                        bootstrap_depart_pos_m = float(from_edge_depart_pos_by_edge[semantic_corridor_edge])
                    try:
                        traci.route.add(route_id, route_edges)
                    except Exception as exc:
                        self._record_semantic_instruction_failure(
                            item,
                            f"route.add for fixed semantic corridor failed: {type(exc).__name__}: {exc}",
                        )
                        return
                    self.resolved_route_edges_by_vehicle_id[vehicle_id] = route_edges
                    self.resolved_route_first_edge_by_vehicle_id[vehicle_id] = route_edges[0]
                    if bootstrap_depart_pos_m is not None:
                        self.resolved_route_depart_pos_by_vehicle_id[vehicle_id] = float(bootstrap_depart_pos_m)
                    self._runtime_state_for(vehicle_id)["route_added"] = True
                    return

                def resolved_sumo_route(src_edge: str, dst_edge: str) -> tuple[list[str], str]:
                    try:
                        try:
                            route_obj = traci.simulation.findRoute(src_edge, dst_edge, vType=type_id)
                        except TypeError:
                            route_obj = traci.simulation.findRoute(src_edge, dst_edge, type_id)
                        direct_edges = [
                            str(edge_id)
                            for edge_id in getattr(route_obj, "edges", [])
                            if str(edge_id)
                        ]
                    except Exception as exc:
                        return [], f"{type(exc).__name__}: {exc}"
                    if not required_edges or set(direct_edges) & required_edges:
                        return direct_edges, ""
                    waypoint_routes: list[list[str]] = []
                    for required_edge in required_edge_order:
                        try:
                            try:
                                prefix_obj = traci.simulation.findRoute(
                                    src_edge,
                                    required_edge,
                                    vType=type_id,
                                )
                                suffix_obj = traci.simulation.findRoute(
                                    required_edge,
                                    dst_edge,
                                    vType=type_id,
                                )
                            except TypeError:
                                prefix_obj = traci.simulation.findRoute(
                                    src_edge,
                                    required_edge,
                                    type_id,
                                )
                                suffix_obj = traci.simulation.findRoute(
                                    required_edge,
                                    dst_edge,
                                    type_id,
                                )
                        except Exception:
                            continue
                        prefix = [
                            str(edge_id)
                            for edge_id in getattr(prefix_obj, "edges", [])
                            if str(edge_id)
                        ]
                        suffix = [
                            str(edge_id)
                            for edge_id in getattr(suffix_obj, "edges", [])
                            if str(edge_id)
                        ]
                        if not prefix or not suffix or prefix[-1] != required_edge or suffix[0] != required_edge:
                            continue
                        joined = [*prefix, *suffix[1:]]
                        if joined[0] == src_edge and joined[-1] == dst_edge:
                            waypoint_routes.append(joined)
                    if waypoint_routes:
                        waypoint_routes.sort(key=lambda edges: (len(edges), tuple(edges)))
                        return waypoint_routes[0], ""
                    return direct_edges, ""

                from_edges = _ordered_unique_strings([from_edge, *(args.get("from_edge_candidates") or [])])
                to_edges = _ordered_unique_strings([to_edge, *(args.get("to_edge_candidates") or [])])
                rejection_counts: dict[str, int] = {}
                valid_candidates: list[tuple[int, int, str, str, list[str]]] = []
                attempted_pairs = 0
                state = self._runtime_state_for(vehicle_id) if vehicle_id else {}
                vehicle_metadata = self.explicit_vehicle_metadata_by_id.get(vehicle_id, {})
                local_component_source = (
                    str(vehicle_metadata.get("traffic_coverage_scope") or "")
                    == "roi_background_local_component_flow"
                )

                def reject(reason: str, payload: dict[str, Any]) -> None:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    if len(rejected) < 12:
                        rejected.append({"reason": reason, **payload})

                candidate_pairs = _bounded_endpoint_pairs(
                    from_edges,
                    to_edges,
                    max_pairs=MAX_FIND_ROUTE_ATTEMPTS_PER_VEHICLE,
                )
                for candidate_from, candidate_to in candidate_pairs:
                    if candidate_from == candidate_to and not local_component_source:
                        reject("invalid_endpoint_pair", {"from_edge": candidate_from, "to_edge": candidate_to})
                        continue
                    if candidate_from not in self.planner.edges or candidate_to not in self.planner.edges:
                        reject("unknown_endpoint_edge", {"from_edge": candidate_from, "to_edge": candidate_to})
                        continue
                    if candidate_from in forbidden_edges or candidate_to in forbidden_edges:
                        reject("endpoint_intersects_forbidden_edges", {"from_edge": candidate_from, "to_edge": candidate_to})
                        continue
                    attempted_pairs += 1
                    if candidate_from == candidate_to:
                        edges, error = [candidate_from], ""
                    else:
                        edges, error = resolved_sumo_route(candidate_from, candidate_to)
                    if error:
                        reject(
                            "sumo_findRoute_exception",
                            {"from_edge": candidate_from, "to_edge": candidate_to, "message": error},
                        )
                        continue
                    if len(edges) < 2 and not local_component_source:
                        reject("empty_or_single_edge_route", {"from_edge": candidate_from, "to_edge": candidate_to})
                        continue
                    hit_forbidden = sorted(set(edges) & forbidden_edges)
                    if hit_forbidden:
                        reject(
                            "route_intersects_forbidden_edges",
                            {"from_edge": candidate_from, "to_edge": candidate_to, "edges": hit_forbidden[:12]},
                        )
                        continue
                    if required_edges and not (set(edges) & required_edges):
                        reject("route_does_not_cross_required_edges", {"from_edge": candidate_from, "to_edge": candidate_to})
                        continue
                    endpoint_use_count = self.resolved_route_endpoint_use_counts.get(candidate_from, 0)
                    endpoint_use_count += self.resolved_route_endpoint_use_counts.get(candidate_to, 0)
                    valid_candidates.append((endpoint_use_count, len(edges), candidate_from, candidate_to, edges))
                    valid_start_count = len({candidate[2] for candidate in valid_candidates})
                    if len(valid_candidates) >= MAX_FIND_ROUTE_VALID_CANDIDATES and valid_start_count >= MAX_FIND_ROUTE_VALID_STARTS:
                        break
                if state is not None:
                    state["route_find_attempt_count"] = int(state.get("route_find_attempt_count") or 0) + attempted_pairs
                    total_pair_count = len(from_edges) * len(to_edges)
                    state["route_find_truncated"] = bool(
                        len(candidate_pairs) < total_pair_count
                        and not valid_candidates
                    )
                if len(candidate_pairs) < len(from_edges) * len(to_edges) and not valid_candidates:
                    reject(
                        "candidate_search_truncated",
                        {
                            "attempted_pair_count": attempted_pairs,
                            "total_pair_count": len(from_edges) * len(to_edges),
                            "max_attempts": MAX_FIND_ROUTE_ATTEMPTS_PER_VEHICLE,
                        },
                    )

                if not from_edges or not to_edges:
                    reject(
                        "missing_endpoint_candidates",
                        {
                            "from_edge": from_edge,
                            "to_edge": to_edge,
                            "from_candidate_count": len(from_edges),
                            "to_candidate_count": len(to_edges),
                        },
                    )
                if valid_candidates:
                    prefer_primary_endpoint = (
                        str(vehicle_metadata.get("traffic_coverage_scope") or "")
                        in {
                            "roi_background_visible_flow",
                            "roi_background_route_crossing_flow",
                            "roi_background_local_component_flow",
                        }
                        or str(vehicle_metadata.get("traffic_role") or "") == "semantic_vehicle"
                    )
                    valid_candidates.sort(
                        key=lambda candidate: (
                            (
                                0
                                if not prefer_primary_endpoint
                                else 0
                                if candidate[2] == from_edge and candidate[3] == to_edge
                                else 1
                                if candidate[2] == from_edge
                                else 2
                            ),
                            candidate[0],
                            self.resolved_route_endpoint_use_counts.get(candidate[2], 0),
                            candidate[1],
                            candidate[2],
                            candidate[3],
                        )
                    )
                    _, _, from_edge, to_edge, resolved_edges = valid_candidates[0]
                    self.resolved_route_endpoint_use_counts[from_edge] = self.resolved_route_endpoint_use_counts.get(from_edge, 0) + 1
                    self.resolved_route_endpoint_use_counts[to_edge] = self.resolved_route_endpoint_use_counts.get(to_edge, 0) + 1
                if rejection_counts:
                    rejected.append(
                        {
                            "reason": "candidate_route_rejection_summary",
                            "from_candidate_count": len(from_edges),
                            "to_candidate_count": len(to_edges),
                            "attempted_pair_count": attempted_pairs,
                            "max_attempts": MAX_FIND_ROUTE_ATTEMPTS_PER_VEHICLE,
                            "valid_candidate_count": len(valid_candidates),
                            "rejection_counts": rejection_counts,
                        }
                    )
                if not route_id or not resolved_edges:
                    if self._is_filterable_background_vehicle(vehicle_id):
                        dominant_reason = max(
                            rejection_counts.items(),
                            key=lambda item: item[1],
                        )[0] if rejection_counts else "sumo_findRoute_rejected_background_vehicle"
                        self._record_background_route_filter(
                            item,
                            reason=dominant_reason,
                            details=rejected[:13],
                        )
                        return
                    self._record_semantic_instruction_failure(
                        item,
                        f"route.findRoute failed for explicit endpoints from={args.get('from_edge')} to={args.get('to_edge')} rejected={rejected[:6]}",
                    )
                    return
                try:
                    traci.route.add(route_id, resolved_edges)
                except Exception as exc:
                    if self._is_filterable_background_vehicle(vehicle_id):
                        self._record_background_route_filter(
                            item,
                            reason="sumo_route_add_rejected_background_vehicle",
                            details=f"{type(exc).__name__}: {exc}",
                        )
                        return
                    self._record_semantic_instruction_failure(item, f"route.add for resolved findRoute failed: {type(exc).__name__}: {exc}")
                    return
                if vehicle_id:
                    self.resolved_route_edges_by_vehicle_id[vehicle_id] = resolved_edges
                    self.resolved_route_first_edge_by_vehicle_id[vehicle_id] = resolved_edges[0]
                    if from_edge in from_edge_depart_pos_by_edge:
                        self.resolved_route_depart_pos_by_vehicle_id[vehicle_id] = float(from_edge_depart_pos_by_edge[from_edge])
                    self._runtime_state_for(vehicle_id)["route_added"] = True
            elif instruction_type == "route.add":
                route_id = str(args.get("route_id") or args.get("routeID") or "")
                edges = [str(edge) for edge in args.get("edges") or [] if str(edge)]
                if route_id and edges:
                    try:
                        traci.route.add(route_id, edges)
                    except Exception:
                        pass
            elif instruction_type == "vehicle.addFull":
                if vehicle_id in self.filtered_background_vehicle_ids:
                    return
                if not vehicle_id or self._active_or_loaded(traci, vehicle_id):
                    return
                state = self._runtime_state_for(vehicle_id)
                state["addFull_attempted"] = True
                if self.explicit_vehicle_ids and vehicle_id not in self.explicit_vehicle_ids:
                    self._record_semantic_instruction_failure(
                        item,
                        f"vehicle.addFull skipped because {vehicle_id} is not in {VEHICLE_SOURCE_POLICY}",
                    )
                    return
                route_id = str(args.get("routeID") or "")
                depart_pos_raw = args.get("departPos")
                is_background_candidate = self._is_filterable_background_vehicle(vehicle_id)
                if isinstance(depart_pos_raw, str) and depart_pos_raw.strip() and not depart_pos_raw.replace(".", "", 1).isdigit():
                    depart_pos = depart_pos_raw.strip()
                else:
                    semantic_corridor_edge = self._semantic_corridor_edge(vehicle_id)
                    resolved_first_edge = self.resolved_route_first_edge_by_vehicle_id.get(vehicle_id)
                    resolved_route = self.resolved_route_edges_by_vehicle_id.get(vehicle_id, [])
                    resolved_depart_pos = self.resolved_route_depart_pos_by_vehicle_id.get(vehicle_id)
                    requested_depart_pos = (
                        float(resolved_depart_pos)
                        if resolved_depart_pos is not None
                        else float(depart_pos_raw or 0.0)
                    )
                    depart_pos = (
                        f"{self._safe_depart_pos(resolved_route, requested_depart_pos):.3f}"
                        if resolved_route and resolved_depart_pos is not None
                        else
                        f"{self._safe_depart_pos(resolved_route, requested_depart_pos):.3f}"
                        if is_background_candidate and resolved_route
                        else "0"
                        if resolved_first_edge
                        and resolved_first_edge != str(self.explicit_vehicle_metadata_by_id.get(vehicle_id, {}).get("from_edge") or "")
                        and resolved_first_edge != semantic_corridor_edge
                        else f"{requested_depart_pos:.3f}"
                    )
                depart_lane = "best" if is_background_candidate else str(args.get("departLane") or "best")
                try:
                    traci.vehicle.addFull(
                        vehicle_id,
                        route_id,
                        typeID=str(args.get("typeID") or "aero_passenger"),
                        depart="now",
                        departLane=depart_lane,
                        departPos=depart_pos,
                        departSpeed=str(args.get("departSpeed") or "0"),
                    )
                except Exception as exc:
                    state["addFull_error"] = f"{type(exc).__name__}: {exc}"
                    if is_background_candidate:
                        self._record_background_route_filter(
                            item,
                            reason="sumo_addFull_rejected_background_vehicle",
                            details={
                                "message": state["addFull_error"],
                                "route_id": route_id,
                                "departLane": depart_lane,
                                "departPos": depart_pos,
                            },
                        )
                        return
                    raise
                state["addFull_succeeded"] = True
                sim_time_s = round(float(traci.simulation.getTime()), 6)
                state["first_loaded_s"] = sim_time_s
                state["last_loaded_s"] = sim_time_s
                self._note_explicit_vehicle_states(traci)
                self.controlled_ids.add(vehicle_id)
                if self._is_semantic_contract_vehicle(vehicle_id):
                    self.semantic_vehicle_ids.add(vehicle_id)
            elif instruction_type == "vehicle.setRoute" and vehicle_id:
                if vehicle_id in self.filtered_background_vehicle_ids:
                    return
                edges = [str(edge) for edge in args.get("edges") or [] if str(edge)]
                if edges and vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.setRoute(vehicle_id, edges)
                elif self._requires_active_semantic_vehicle(item):
                    self._record_semantic_instruction_failure(item, "vehicle.setRoute skipped because semantic vehicle is not active")
            elif instruction_type == "vehicle.slowDown":
                if vehicle_id in self.filtered_background_vehicle_ids:
                    return
                if vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.slowDown(vehicle_id, float(args.get("speed") or 0.0), float(args.get("duration") or 0.1))
                elif self._requires_active_semantic_vehicle(item):
                    self._record_semantic_instruction_failure(item, "vehicle.slowDown skipped because semantic vehicle is not active")
            elif instruction_type == "vehicle.setSpeed":
                if vehicle_id in self.filtered_background_vehicle_ids:
                    return
                if vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.setSpeed(vehicle_id, float(args["speed"] if args.get("speed") is not None else -1.0))
                elif self._is_filterable_background_vehicle(vehicle_id):
                    self._queue_background_vehicle_control(item, sim_time_s=float(traci.simulation.getTime()))
                elif self._requires_active_semantic_vehicle(item):
                    self._record_semantic_instruction_failure(item, "vehicle.setSpeed skipped because semantic vehicle is not active")
            elif instruction_type == "vehicle.setStop":
                if vehicle_id in self.filtered_background_vehicle_ids:
                    return
                if vehicle_id in traci.vehicle.getIDList():
                    payload = self._background_stop_payload(vehicle_id, args)
                    if payload is None:
                        return
                    kwargs: dict[str, Any] = {}
                    if args.get("until") not in (None, ""):
                        kwargs["until"] = float(args.get("until"))
                    if args.get("duration") not in (None, ""):
                        kwargs["duration"] = float(args.get("duration"))
                    edge_id, pos, lane_index = payload
                    if self._is_filterable_background_vehicle(vehicle_id) and not self._stop_edge_is_downstream(traci, vehicle_id, edge_id):
                        self._record_background_control_failure(item, "vehicle.setStop skipped because stop edge is no longer downstream")
                        return
                    try:
                        traci.vehicle.setStop(
                            vehicle_id,
                            edge_id,
                            pos,
                            lane_index,
                            **kwargs,
                        )
                    except Exception as exc:
                        if self._is_filterable_background_vehicle(vehicle_id):
                            self._record_background_control_failure(item, f"vehicle.setStop raised {type(exc).__name__}: {exc}")
                            return
                        raise
                elif self._is_filterable_background_vehicle(vehicle_id):
                    self._queue_background_vehicle_control(item, sim_time_s=float(traci.simulation.getTime()))
                elif self._requires_active_semantic_vehicle(item):
                    self._record_semantic_instruction_failure(item, "vehicle.setStop skipped because semantic vehicle is not active")
            elif instruction_type == "vehicle.setSignals":
                if vehicle_id in self.filtered_background_vehicle_ids:
                    return
                if vehicle_id in traci.vehicle.getIDList():
                    raw = args.get("signals")
                    signals = SIGNAL_EMERGENCY if isinstance(raw, str) else int(raw or SIGNAL_HAZARD)
                    traci.vehicle.setSignals(vehicle_id, signals)
                elif self._requires_active_semantic_vehicle(item):
                    self._record_semantic_instruction_failure(item, "vehicle.setSignals skipped because semantic vehicle is not active")
            elif instruction_type == "vehicle.setMaxSpeed":
                if vehicle_id in self.filtered_background_vehicle_ids:
                    return
                if vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.setMaxSpeed(vehicle_id, float(args.get("speed") or 0.0))
                elif self._is_filterable_background_vehicle(vehicle_id):
                    self._queue_background_vehicle_control(item, sim_time_s=float(traci.simulation.getTime()))
                elif self._requires_active_semantic_vehicle(item):
                    self._record_semantic_instruction_failure(item, "vehicle.setMaxSpeed skipped because semantic vehicle is not active")
            elif instruction_type == "vehicle.resume":
                if vehicle_id in self.filtered_background_vehicle_ids:
                    return
                if vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.resume(vehicle_id)
                elif self._is_filterable_background_vehicle(vehicle_id):
                    self._queue_background_vehicle_control(item, sim_time_s=float(traci.simulation.getTime()))
                elif self._requires_active_semantic_vehicle(item):
                    self._record_semantic_instruction_failure(item, "vehicle.resume skipped because semantic vehicle is not active")
            elif instruction_type == "vehicle.remove":
                if vehicle_id in self.filtered_background_vehicle_ids:
                    return
                if vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.remove(vehicle_id)
                    self.controlled_ids.discard(vehicle_id)
                    self.background_ids.discard(vehicle_id)
                elif self._requires_active_semantic_vehicle(item):
                    self._record_semantic_instruction_failure(item, "vehicle.remove skipped because semantic vehicle is not active")
            elif instruction_type == "vehicle.setParameter":
                if vehicle_id in self.filtered_background_vehicle_ids:
                    return
                if vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.setParameter(vehicle_id, str(args.get("key") or ""), str(args.get("value") or ""))
                elif self._requires_active_semantic_vehicle(item):
                    self._record_semantic_instruction_failure(item, "vehicle.setParameter skipped because semantic vehicle is not active")
            elif instruction_type == "vehicle.moveToXY":
                if vehicle_id in self.filtered_background_vehicle_ids:
                    return
                active_vehicle_ids = set(traci.vehicle.getIDList())
                force_loaded_semantic = (
                    vehicle_id not in active_vehicle_ids
                    and self._requires_active_semantic_vehicle(item)
                    and self._is_semantic_contract_vehicle(vehicle_id)
                )
                if vehicle_id in active_vehicle_ids or force_loaded_semantic:
                    from traci import constants as tc  # type: ignore

                    edge_id = str(args.get("edgeID") or "")
                    lane_index = int(args.get("laneIndex") if args.get("laneIndex") is not None else args.get("lane") if args.get("lane") is not None else 0)
                    keep_route = int(args["keepRoute"] if args.get("keepRoute") is not None else 2)
                    # SUMO 1.8's TraCI server accepts the fixed-lane MOVE_TO_XY
                    # packet without the matchThreshold field added by newer clients.
                    traci.vehicle._setCmd(
                        tc.MOVE_TO_XY,
                        vehicle_id,
                        "tsidddb",
                        6,
                        edge_id,
                        lane_index,
                        float(args.get("x") or 0.0),
                        float(args.get("y") or 0.0),
                        float(args.get("angle") or -1073741824.0),
                        keep_route,
                    )
                    if force_loaded_semantic:
                        state = self._runtime_state_for(vehicle_id)
                        state["forced_insert_by_move_to_xy"] = True
                        state["forced_insert_by_move_to_xy_s"] = round(float(traci.simulation.getTime()), 6)
                        self._note_explicit_vehicle_states(traci)
                elif self._requires_active_semantic_vehicle(item):
                    self._record_semantic_instruction_failure(item, "vehicle.moveToXY skipped because semantic vehicle is not active")
            if vehicle_id:
                is_semantic_contract_vehicle = self._is_semantic_contract_vehicle(vehicle_id)
                if is_semantic_contract_vehicle:
                    self.semantic_vehicle_ids.add(vehicle_id)
                if source_entity_id:
                    self.semantic_source_by_vehicle_id[vehicle_id] = source_entity_id
                if episode_id:
                    self.semantic_episode_by_vehicle_id[vehicle_id] = episode_id
                if vehicle_id in traci.vehicle.getIDList():
                    if source_entity_id:
                        traci.vehicle.setParameter(vehicle_id, "aero.source_entity_id", source_entity_id)
                    if episode_id:
                        traci.vehicle.setParameter(vehicle_id, "aero.episode_id", episode_id)
                    metadata = self.semantic_metadata_by_vehicle_id.get(vehicle_id)
                    if metadata:
                        traci.vehicle.setParameter(
                            vehicle_id,
                            "aero.semantic_metadata",
                            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                        )
                    traci.vehicle.setParameter(
                        vehicle_id,
                        "aero.semantic_vehicle",
                        "true" if is_semantic_contract_vehicle else "false",
                    )
        except Exception as exc:
            if instruction_type in {"vehicle.addFull", "vehicle.moveToXY"} or self._requires_active_semantic_vehicle(item):
                self._record_semantic_instruction_failure(item, f"{instruction_type} raised {type(exc).__name__}: {exc}")
            return

    def _apply_due_incidents(self, traci: Any, sim_time_s: float) -> None:
        for incident in incident_records(self.incident_plan):
            incident_id = str(incident["incident_id"])
            if incident_id in self.applied_incidents:
                continue
            if sim_time_s + 1e-9 < float(incident["start_s"]):
                continue
            self._apply_incident(traci, incident)
            self.applied_incidents.add(incident_id)

    def _apply_incident(self, traci: Any, incident: dict[str, Any]) -> None:
        accident_class = str(incident.get("accident_class") or "")
        if accident_class not in {"lane_closure_roadwork", "hazmat_isolation_zone"} and self._incident_covered_by_semantic_vehicle_plan(incident):
            return
        if self.explicit_vehicle_ids and accident_class not in {
            "traffic_light_all_red_fault",
            "lane_closure_roadwork",
            "hazmat_isolation_zone",
            "weather_speed_degradation",
        }:
            return
        if accident_class == "traffic_light_all_red_fault":
            self._apply_all_red_fault(traci, incident)
        elif accident_class in {"lane_closure_roadwork", "hazmat_isolation_zone"}:
            self._apply_lane_or_edge_slowdown(traci, incident)
        elif accident_class == "weather_speed_degradation":
            self._apply_weather_slowdown(traci, incident)
        elif accident_class == "emergency_vehicle_priority":
            self._apply_emergency_priority(traci, incident)
        elif accident_class == "medical_response_dispatch":
            self._apply_medical_dispatch(traci, incident)
        else:
            self._apply_stop_incident(traci, incident)

    def _incident_covered_by_semantic_vehicle_plan(self, incident: dict[str, Any]) -> bool:
        return any(
            str(vehicle_id).replace(" ", "_") in self.semantic_vehicle_id_by_source_entity
            for vehicle_id in incident.get("affected_vehicle_ids") or []
        )

    def _anchor(self, incident: dict[str, Any]) -> dict[str, Any]:
        return dict(incident.get("anchor") or {})

    def _incident_vehicle_ids(self, incident: dict[str, Any], *, count: int = 1) -> list[str]:
        raw_ids = [str(item) for item in incident.get("affected_vehicle_ids") or [] if str(item)]
        if not raw_ids:
            raw_ids = [f"{incident['incident_id']}.vehicle_{index}" for index in range(count)]
        while len(raw_ids) < count:
            raw_ids.append(f"{incident['incident_id']}.vehicle_{len(raw_ids)}")
        resolved: list[str] = []
        for item in raw_ids[:count]:
            source_id = item.replace(" ", "_")
            resolved.append(self.semantic_vehicle_id_by_source_entity.get(source_id, source_id))
        return resolved

    def _spawn_controlled_near_anchor(
        self,
        traci: Any,
        incident: dict[str, Any],
        vehicle_id: str,
        *,
        approach_m: float = 45.0,
        speed_mps: float = 8.0,
        emergency: bool = False,
    ) -> bool:
        if vehicle_id in set(traci.vehicle.getIDList()):
            self.controlled_ids.add(vehicle_id)
            return True
        if self.explicit_vehicle_ids and vehicle_id not in self.explicit_vehicle_ids:
            self.incident_control_skips.append(
                {
                    "policy": VEHICLE_SOURCE_POLICY,
                    "reason": "skip_unplanned_incident_vehicle_spawn_in_explicit_mode",
                    "incident_id": str(incident.get("incident_id") or ""),
                    "accident_class": str(incident.get("accident_class") or ""),
                    "vehicle_id": vehicle_id,
                }
            )
            return False
        anchor = self._anchor(incident)
        edge_id = str(anchor["sumo_edge_id"])
        vehicle_class = str(anchor.get("vehicle_class") or "passenger")
        route_edges = self.route_factory.path_from_edge(edge_id, vehicle_class, min_edges=3, max_edges=10)
        anchor_s = float(anchor.get("lane_position_m") or 0.0)
        depart_pos = max(0.0, anchor_s - approach_m)
        available_stop_distance_m = max(0.0, anchor_s - depart_pos)
        resolved_speed_mps = (
            min(speed_mps, max(0.4, available_stop_distance_m / 8.0))
            if available_stop_distance_m < 20.0
            else speed_mps
        )
        return self._spawn_vehicle(
            traci,
            vehicle_id=vehicle_id,
            route_edges=route_edges,
            vehicle_class=vehicle_class,
            depart_pos_m=depart_pos,
            depart_speed_mps=resolved_speed_mps,
            emergency=emergency,
            controlled=True,
        )

    def _clamped_stop_pos(self, anchor: dict[str, Any], *, offset_m: float = 0.0) -> float:
        edge_id = str(anchor.get("sumo_edge_id") or "")
        raw = float(anchor.get("lane_position_m") or 1.0) + float(offset_m)
        edge = self.planner.edges.get(edge_id)
        if edge is None:
            return max(MIN_STOP_FRONT_POS_M, raw)
        length_m = max(0.0, float(edge.length_m))
        if length_m <= MIN_STOP_FRONT_POS_M + STOP_END_MARGIN_M:
            return max(0.0, length_m * 0.5)
        upper = max(MIN_STOP_FRONT_POS_M, length_m - STOP_END_MARGIN_M)
        return max(MIN_STOP_FRONT_POS_M, min(upper, raw))

    def _apply_all_red_fault(self, traci: Any, incident: dict[str, Any]) -> None:
        traffic_light = dict(incident.get("traffic_light") or {})
        tls_id = str(traffic_light.get("traffic_light_id") or "")
        if not tls_id:
            raise RuntimeError(f"all-red incident lacks a controller target: {incident.get('incident_id')}")
        try:
            original_state = traci.trafficlight.getRedYellowGreenState(tls_id)
            original_phase = traci.trafficlight.getPhase(tls_id)
            original_program = traci.trafficlight.getProgram(tls_id)
            traci.trafficlight.setRedYellowGreenState(tls_id, "r" * len(original_state))
            self.restores.append(
                RestoreAction(
                    kind="traffic_light",
                    target_id=tls_id,
                    end_s=float(incident["end_s"]),
                    payload={"state": original_state, "phase": original_phase, "program": original_program},
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"all-red incident could not set controller {tls_id}: {incident.get('incident_id')}"
            ) from exc

    def _apply_lane_or_edge_slowdown(self, traci: Any, incident: dict[str, Any]) -> None:
        anchor = self._anchor(incident)
        lane_id = str(anchor.get("sumo_lane_id") or "")
        edge_id = str(anchor.get("sumo_edge_id") or "")
        original = self.lane_incident_execution.apply(traci, incident, maximum_speed_mps=0.4)
        self.restores.append(RestoreAction("lane_speed", lane_id, float(incident["end_s"]), original))
        # Authored vehicle actions own their trajectory, while the lane control
        # above remains an independent effect of the incident.
        if self._incident_covered_by_semantic_vehicle_plan(incident):
            return
        for vehicle_id in self._incident_vehicle_ids(incident, count=1):
            if self._spawn_controlled_near_anchor(traci, incident, vehicle_id, approach_m=35.0, speed_mps=5.0):
                try:
                    traci.vehicle.slowDown(vehicle_id, 1.0, 10.0)
                    traci.vehicle.setStop(
                        vehicle_id,
                        edge_id,
                        self._clamped_stop_pos(anchor),
                        int(anchor.get("lane_index") or 0),
                        duration=8.0,
                    )
                except Exception:
                    pass

    def _apply_weather_slowdown(self, traci: Any, incident: dict[str, Any]) -> None:
        for vehicle_id in traci.vehicle.getIDList():
            if vehicle_id in self.controlled_ids:
                continue
            try:
                traci.vehicle.slowDown(vehicle_id, 3.5, 20.0)
                traci.vehicle.setMaxSpeed(vehicle_id, 4.5)
                self.restores.append(RestoreAction("vehicle_max_speed", vehicle_id, float(incident["end_s"]), 13.89))
            except Exception:
                pass

    def _apply_emergency_priority(self, traci: Any, incident: dict[str, Any]) -> None:
        ids = self._incident_vehicle_ids(incident, count=2)
        ambulance_id = ids[0]
        yield_id = ids[1]
        if self._spawn_controlled_near_anchor(traci, incident, ambulance_id, approach_m=70.0, speed_mps=12.0, emergency=True):
            try:
                traci.vehicle.setSignals(ambulance_id, SIGNAL_EMERGENCY)
            except Exception:
                pass
        if self._spawn_controlled_near_anchor(traci, incident, yield_id, approach_m=20.0, speed_mps=4.0):
            try:
                traci.vehicle.slowDown(yield_id, 1.2, 8.0)
                traci.vehicle.setSignals(yield_id, SIGNAL_HAZARD)
            except Exception:
                pass

    def _apply_medical_dispatch(self, traci: Any, incident: dict[str, Any]) -> None:
        vehicle_id = self._incident_vehicle_ids(incident, count=1)[0]
        if self._spawn_controlled_near_anchor(traci, incident, vehicle_id, approach_m=80.0, speed_mps=11.0, emergency=True):
            try:
                traci.vehicle.setSignals(vehicle_id, SIGNAL_EMERGENCY)
            except Exception:
                pass

    def _apply_stop_incident(self, traci: Any, incident: dict[str, Any]) -> None:
        anchor = self._anchor(incident)
        edge_id = str(anchor.get("sumo_edge_id") or "")
        lane_index = int(anchor.get("lane_index") or 0)
        ids = self._incident_vehicle_ids(incident, count=2 if incident.get("accident_class") == "vehicle_intersection_conflict" else 1)
        for offset, vehicle_id in enumerate(ids):
            approach = 55.0 + offset * 18.0
            if not self._spawn_controlled_near_anchor(traci, incident, vehicle_id, approach_m=approach, speed_mps=8.0 - offset):
                continue
            try:
                traci.vehicle.slowDown(vehicle_id, 0.5, 5.0)
                traci.vehicle.setSignals(vehicle_id, SIGNAL_HAZARD)
                traci.vehicle.setStop(
                    vehicle_id,
                    edge_id,
                    self._clamped_stop_pos(anchor, offset_m=offset * 2.0),
                    lane_index,
                    duration=12.0,
                )
            except Exception:
                pass

    def _restore_due_actions(self, traci: Any, sim_time_s: float) -> None:
        pending: list[RestoreAction] = []
        for restore in self.restores:
            if sim_time_s + 1e-9 < restore.end_s:
                pending.append(restore)
                continue
            if restore.kind == "lane_speed":
                traci.lane.setMaxSpeed(restore.target_id, float(restore.payload))
                self.lane_incident_execution.record_restoration(traci, restore.target_id, float(restore.payload))
                continue
            try:
                if restore.kind == "traffic_light":
                    program = str(restore.payload.get("program") or "")
                    if program:
                        traci.trafficlight.setProgram(restore.target_id, program)
                        traci.trafficlight.setPhase(restore.target_id, int(restore.payload["phase"]))
                    else:
                        traci.trafficlight.setRedYellowGreenState(restore.target_id, restore.payload["state"])
                elif restore.kind == "vehicle_max_speed" and restore.target_id in traci.vehicle.getIDList():
                    traci.vehicle.setMaxSpeed(restore.target_id, float(restore.payload))
            except Exception as exc:
                if restore.kind == "traffic_light":
                    raise RuntimeError(f"could not restore traffic light {restore.target_id}") from exc
        self.restores = pending

    def _trim_vehicle_cap(self, traci: Any) -> None:
        ids = set(traci.vehicle.getIDList())
        excess = len(ids) - self.max_vehicles
        if excess <= 0:
            return
        removable = sorted(vehicle_id for vehicle_id in ids if vehicle_id in self.background_ids)
        for vehicle_id in removable[:excess]:
            try:
                traci.vehicle.remove(vehicle_id)
                self.background_ids.discard(vehicle_id)
            except Exception:
                pass

    def _write_frame(self, handle: Any, traci: Any, *, tick: int) -> None:
        sim_time_s = round(float(traci.simulation.getTime()), 6)
        tick_10hz = int(round(sim_time_s / self.step_length_s))
        vehicles = self._vehicle_records(traci)
        if self.explicit_vehicle_ids:
            self.seen_explicit_vehicle_ids.update(
                str(vehicle.get("vehicle_id") or "")
                for vehicle in vehicles
                if str(vehicle.get("vehicle_id") or "") in self.explicit_vehicle_ids
            )
        if self.explicit_vehicle_ids:
            extra_vehicle_ids = sorted(
                str(vehicle.get("vehicle_id") or "")
                for vehicle in vehicles
                if str(vehicle.get("vehicle_id") or "") not in self.explicit_vehicle_ids
            )
            if extra_vehicle_ids:
                raise RuntimeError(
                    f"SUMO output contains vehicles not listed in explicit plan: {extra_vehicle_ids[:12]}"
                )
        frame = {
            "schema_name": "sumo_traffic_frame",
            "schema_version": "v1",
            "map_id": "donghu_road_topo",
            "tick": tick_10hz,
            "sample_tick": int(tick),
            "tick_hz": int(round(1.0 / self.step_length_s)),
            "sample_period_s": round(self.sample_period_s, 6),
            "sim_time_s": sim_time_s,
            "vehicles": vehicles,
            "traffic_lights": self._traffic_light_records(traci),
            "active_incidents": self._active_incidents(sim_time_s),
            "lane_incident_observations": self.lane_incident_execution.observe(traci),
        }
        handle.write(json.dumps(frame, ensure_ascii=False, default=_json_default, sort_keys=True) + "\n")

    def _vehicle_records(self, traci: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        episode_tick = int(round((float(traci.simulation.getTime()) - self.warmup_s) / self.step_length_s))
        for vehicle_id in sorted(traci.vehicle.getIDList()):
            try:
                sx, sy = traci.vehicle.getPosition(vehicle_id)
                speed = float(traci.vehicle.getSpeed(vehicle_id))
                accel = float(traci.vehicle.getAcceleration(vehicle_id))
                angle = float(traci.vehicle.getAngle(vehicle_id))
                length_m = float(traci.vehicle.getLength(vehicle_id))
                yaw_deg = _truth_yaw_from_sumo_angle(angle)
                front_tx, front_ty = self.planner.coordinate_mapper.sumo_xy_to_truth_xy(float(sx), float(sy))
                lane_position_m = float(traci.vehicle.getLanePosition(vehicle_id))
                lane_id = str(traci.vehicle.getLaneID(vehicle_id) or "")
                road_id = str(traci.vehicle.getRoadID(vehicle_id) or "")
                lane_shape: list[Any] = []
                source_entity_id = self.semantic_source_by_vehicle_id.get(vehicle_id, "")
                semantic_episode_id = self.semantic_episode_by_vehicle_id.get(vehicle_id, "")
                if not source_entity_id:
                    source_entity_id = self.explicit_source_by_vehicle_id.get(vehicle_id, "")
                if not semantic_episode_id and vehicle_id in self.explicit_vehicle_ids:
                    semantic_episode_id = str(self.explicit_vehicle_plan.get("episode_id") or "")
                if not source_entity_id:
                    source_entity_id = _vehicle_parameter(traci, vehicle_id, "aero.source_entity_id")
                if not semantic_episode_id:
                    semantic_episode_id = _vehicle_parameter(traci, vehicle_id, "aero.episode_id")
                semantic_metadata_raw = _vehicle_parameter(traci, vehicle_id, "aero.semantic_metadata")
                active_event_raw = _vehicle_parameter(traci, vehicle_id, "aero.active_semantic_event")
                semantic_state_raw = _vehicle_parameter(traci, vehicle_id, "aero.semantic_vehicle_state")
                semantic_metadata = _json_object_or_none(semantic_metadata_raw)
                active_event = _json_object_or_none(active_event_raw)
                semantic_state = _json_object_or_none(semantic_state_raw)
                explicit_metadata = self.explicit_vehicle_metadata_by_id.get(vehicle_id, {})
                is_semantic_contract_vehicle = self._is_semantic_contract_vehicle(vehicle_id)
                if is_semantic_contract_vehicle:
                    control_window = list(explicit_metadata.get("semantic_control_window_ticks") or [])
                    if len(control_window) < 2:
                        control_ticks = [
                            int(control.get("tick") or 0)
                            for control in explicit_metadata.get("semantic_controls") or []
                        ]
                        if control_ticks:
                            control_window = [min(control_ticks), max(control_ticks)]
                    if len(control_window) >= 2:
                        start_tick = int(control_window[0])
                        end_tick = int(control_window[1])
                        if episode_tick < start_tick or episode_tick > end_tick:
                            continue
                center_tx, center_ty, yaw_deg, center_lane_position_m, position_reference = _vehicle_body_center_pose(
                    front_truth_xy_m=(front_tx, front_ty),
                    yaw_deg=yaw_deg,
                    length_m=length_m,
                    lane_position_m=lane_position_m,
                    lane_shape=lane_shape,
                    coordinate_mapper=self.planner.coordinate_mapper,
                )
                records.append(
                    {
                        "vehicle_id": vehicle_id,
                        "vehicle_type": traci.vehicle.getTypeID(vehicle_id),
                        "route_id": traci.vehicle.getRouteID(vehicle_id),
                        "sumo_edge_id": road_id,
                        "sumo_lane_id": lane_id,
                        "lane_position_m": round(lane_position_m, 6),
                        "center_lane_position_m": round(center_lane_position_m, 6),
                        "sumo_xy_m": _round_list([sx, sy]),
                        "sumo_position_reference": "front_bumper_center",
                        "sumo_front_bumper_xy_m": _round_list([sx, sy]),
                        "truth_front_bumper_enu_m": _round_list([front_tx, front_ty, 0.0]),
                        "truth_position_enu_m": _round_list([center_tx, center_ty, 0.0]),
                        "truth_position_reference": position_reference,
                        "truth_yaw_deg": round(yaw_deg, 6),
                        "sumo_angle_deg": round(angle, 6),
                        "speed_mps": round(speed, 6),
                        "accel_mps2": round(accel, 6),
                        "signals": int(traci.vehicle.getSignals(vehicle_id)),
                        "dimensions_m": {
                            "length": round(length_m, 6),
                            "width": round(float(traci.vehicle.getWidth(vehicle_id)), 6),
                            "height": round(float(traci.vehicle.getHeight(vehicle_id)), 6),
                        },
                        "source": "sumo_traci",
                        "source_entity_id": source_entity_id,
                        "semantic_episode_id": semantic_episode_id,
                        "semantic_vehicle": is_semantic_contract_vehicle,
                        "semantic_metadata": semantic_metadata,
                        "active_semantic_event": active_event,
                        "semantic_vehicle_state": semantic_state,
                        "explicit_plan_vehicle": vehicle_id in self.explicit_vehicle_ids,
                        "vehicle_source_policy": VEHICLE_SOURCE_POLICY if vehicle_id in self.explicit_vehicle_ids else "",
                        "resolved_route_edges": self.resolved_route_edges_by_vehicle_id.get(vehicle_id, []),
                        "traffic_slot_index": explicit_metadata.get("traffic_slot_index"),
                        "expected_core_entry_tick": explicit_metadata.get("expected_core_entry_tick"),
                        "expected_core_exit_tick": explicit_metadata.get("expected_core_exit_tick"),
                        "protected_window_role": (
                            "allowed_semantic_vehicle"
                            if str(explicit_metadata.get("traffic_role") or "") == "semantic_vehicle"
                            else "background_forbidden_from_protected_edges"
                            if explicit_metadata
                            else ""
                        ),
                        "route_source": "sumo_findRoute" if vehicle_id in self.resolved_route_edges_by_vehicle_id else "",
                        "control_role": (
                            self.explicit_role_by_vehicle_id.get(vehicle_id)
                            or ("semantic_event" if source_entity_id else "incident_controlled" if vehicle_id in self.controlled_ids else "background")
                        ),
                    }
                )
            except Exception:
                continue
        return records

    def _traffic_light_records(self, traci: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for tls_id in sorted(traci.trafficlight.getIDList()):
            try:
                controlled_links = traci.trafficlight.getControlledLinks(tls_id)
                records.append(
                    {
                        "tls_id": tls_id,
                        "program_id": traci.trafficlight.getProgram(tls_id),
                        "phase_index": int(traci.trafficlight.getPhase(tls_id)),
                        "state": traci.trafficlight.getRedYellowGreenState(tls_id),
                        "next_switch_s": round(float(traci.trafficlight.getNextSwitch(tls_id)), 6),
                        "controlled_links": json.loads(json.dumps(controlled_links, default=_json_default)),
                    }
                )
            except Exception:
                continue
        return records

    def _active_incidents(self, sim_time_s: float) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for incident in incident_records(self.incident_plan):
            if incident["accident_class"] in {"lane_closure_roadwork", "hazmat_isolation_zone"}:
                control = self.lane_incident_execution.controls.get(incident["incident_id"])
                if control is None or control["restored"]:
                    continue
            if float(incident["start_s"]) <= sim_time_s and sim_time_s + 1e-9 < float(incident["end_s"]):
                active_incident = {
                    "incident_id": incident["incident_id"],
                    "episode_scenario_id": incident["episode_scenario_id"],
                    "episode_event_id": incident["episode_event_id"],
                    "intent": incident["intent"],
                    "accident_class": incident["accident_class"],
                    "start_s": incident["start_s"],
                    "end_s": incident["end_s"],
                    "anchor": incident["anchor"],
                    "injection_method": incident["injection_method"],
                    "affected_vehicle_ids": incident.get("affected_vehicle_ids") or [],
                }
                if incident["accident_class"] == "traffic_light_all_red_fault":
                    target = (incident.get("traffic_light") or {}).get("traffic_light_id")
                    if not isinstance(target, str) or not target:
                        raise RuntimeError(f"all-red incident lacks a controller target: {incident['incident_id']}")
                    active_incident["traffic_light_id"] = target
                active.append(active_incident)
        return active

    def _assert_no_unplanned_active_vehicles(self, traci: Any) -> None:
        active = {str(vehicle_id) for vehicle_id in traci.vehicle.getIDList()}
        unplanned = sorted(active - self.explicit_vehicle_ids)
        if unplanned:
            raise RuntimeError(f"{VEHICLE_SOURCE_POLICY} forbids unplanned active SUMO vehicles: {unplanned[:12]}")

    def _manifest(self, *, sample_count: int) -> dict[str, Any]:
        traffic_geometry = dict(self.planner.traffic_geometry_provenance)
        del traffic_geometry["roadway_mesh_hash"]
        if self.explicit_vehicle_ids:
            semantic_source_entity_ids = sorted(
                source_entity_id
                for vehicle_id in self.semantic_vehicle_ids
                if (source_entity_id := self.explicit_source_by_vehicle_id.get(vehicle_id, ""))
            )
        else:
            semantic_source_entity_ids = sorted(set(self.semantic_source_by_vehicle_id.values()))
        return {
            "schema_name": "sumo_traffic_manifest",
            "schema_version": "v1",
            "map_id": "donghu_road_topo",
            "net_xml": str(self.planner.net_xml),
            "road_derived_network_manifest": str(self.planner.net_xml.with_name("road_derived_network_manifest.json")),
            "coordinate_mapper": self.planner.coordinate_mapper.fit_summary,
            "traffic_geometry": traffic_geometry,
            "traffic_light_count": _count_net_traffic_lights(self.planner.net_xml),
            "duration_s": self.duration_s,
            "warmup_s": self.warmup_s,
            "capture_start_s": self.warmup_s,
            "capture_duration_s": self.capture_duration_s,
            "time_axis_policy": "episode_local_vehicle_sumo_after_warmup_v1",
            "step_length_s": self.step_length_s,
            "tick_hz": int(round(1.0 / self.step_length_s)),
            "sample_period_s": self.sample_period_s,
            "sample_every_ticks": int(round(self.sample_period_s / self.step_length_s)),
            "sample_count": int(sample_count),
            "max_vehicles": self.max_vehicles,
            "vehicle_source_policy": VEHICLE_SOURCE_POLICY,
            "explicit_vehicle_plan": {
                "enabled": bool(self.explicit_vehicle_ids),
                "schema": self.explicit_vehicle_plan.get("schema"),
                "episode_id": self.explicit_vehicle_plan.get("episode_id"),
                "vehicle_count": len(self.explicit_vehicle_ids),
                "vehicle_ids": sorted(self.explicit_vehicle_ids),
                "traffic_case": self.explicit_vehicle_plan.get("traffic_case"),
                "seed_profile": self.explicit_vehicle_plan.get("seed_profile"),
                "traffic_profile": self.explicit_vehicle_plan.get("traffic_profile"),
                "traffic_tuning": self.explicit_vehicle_plan.get("traffic_tuning"),
            },
            "explicit_vehicle_route_filter": {
                "policy": "sumo_findRoute_is_route_authority_filter_background_only_v1",
                "filtered_background_vehicle_count": len(self.filtered_background_vehicle_ids),
                "filtered_background_vehicle_ids": sorted(self.filtered_background_vehicle_ids),
                "failures": self.filtered_background_vehicle_route_failures,
            },
            "road_semantics": self.explicit_vehicle_plan.get("road_semantics") or {},
            "explicit_vehicle_activation": {
                "policy": "record_realized_background_activation_no_accepted_target_gate_v3",
                "accepted_background_vehicle_target": self._accepted_background_vehicle_target(),
                "accepted_background_vehicle_count": len(self.accepted_background_vehicle_ids),
                "accepted_background_vehicle_ids": sorted(self.accepted_background_vehicle_ids),
                "inactive_background_vehicle_count": len(self.inactive_background_vehicle_ids),
                "inactive_background_vehicle_ids": sorted(self.inactive_background_vehicle_ids),
                "inactive_background_vehicle_details": self._inactive_background_vehicle_details(),
                "pending_background_control_count": len(self.pending_background_vehicle_controls),
                "background_control_failures": self.background_vehicle_control_failures,
                "incident_control_skip_count": len(self.incident_control_skips),
                "incident_control_skips": self.incident_control_skips,
                "departed_explicit_vehicle_count": len(self.departed_explicit_vehicle_ids),
                "departed_explicit_vehicle_ids": sorted(self.departed_explicit_vehicle_ids),
                "arrived_explicit_vehicle_count": len(self.arrived_explicit_vehicle_ids),
                "arrived_explicit_vehicle_ids": sorted(self.arrived_explicit_vehicle_ids),
                "semantic_vehicle_count": len(self.semantic_vehicle_ids),
                "semantic_vehicle_ids": sorted(self.semantic_vehicle_ids),
                "source_presence_required_vehicle_count": len(
                    self.explicit_source_presence_required_vehicle_ids
                ),
                "source_presence_required_vehicle_ids": sorted(
                    self.explicit_source_presence_required_vehicle_ids
                ),
                "source_presence_capture_lifecycle_by_vehicle_id": {
                    vehicle_id: {
                        "first_active_s": self._runtime_state_for(vehicle_id).get("first_active_s"),
                        "last_active_s": self._runtime_state_for(vehicle_id).get("last_active_s"),
                        "required_active_interval_s": [
                            round(float(self.warmup_s), 6),
                            round(float(self.duration_s), 6),
                        ],
                    }
                    for vehicle_id in sorted(self.explicit_source_presence_required_vehicle_ids)
                },
            },
            "semantic_vehicle_plans": {
                "instruction_count": len(self.semantic_vehicle_instructions),
                "vehicle_count": len(self.semantic_vehicle_ids),
                "source_entity_count": len(semantic_source_entity_ids),
                "source_entity_ids": semantic_source_entity_ids,
            },
            "visual_overlap_guard": {
                "policy": "disabled_sumo_authoritative_vehicle_truth_v2",
                "removed_vehicle_count": 0,
                "removed_background_vehicle_count": 0,
                "removed_controlled_vehicle_count": 0,
            },
            "roi_lane_coverage": {
                "policy": "scenario_roi_multilane_anchor_round_robin_v2",
                "target_edge_count": len({anchor.edge_id for anchor in self.roi_lane_coverage_anchors}),
                "target_anchor_count": len(self.roi_lane_coverage_anchors),
                "spawn_period": ROI_LANE_COVERAGE_SPAWN_PERIOD,
                "target_edges": [anchor.edge_id for anchor in self.roi_lane_coverage_anchors[:120]],
                "target_anchors": [
                    {
                        "edge_id": anchor.edge_id,
                        "depart_pos_m": round(float(anchor.depart_pos_m), 3),
                        "source_road_id": anchor.source_road_id,
                        "direction_role": anchor.direction_role,
                        "physical_lane_index": anchor.physical_lane_index,
                    }
                    for anchor in self.roi_lane_coverage_anchors[:120]
                ],
            },
            "demand_profile": {
                "policy": VEHICLE_SOURCE_POLICY,
                "warmup": "no random background demand in explicit mode",
                "capture": "only vehicles from sumo_explicit_vehicle_plan.json may enter SUMO output",
                "semantic_time_axis": "explicit departures run at warmup_s + depart_tick / tick_hz; event controls run at warmup_s + episode_tick / tick_hz",
            },
            "incident_plan": {
                "incident_count": int(self.incident_plan.get("incident_count") or 0),
                "schema_name": self.incident_plan.get("schema_name"),
                "schema_version": self.incident_plan.get("schema_version"),
            },
            "outputs": {
                "manifest": str(self.output_dir / "sumo_traffic_manifest.json"),
                "incident_plan": str(self.output_dir / "sumo_incident_plan.json"),
                "frames": str(self.output_dir / "sumo_traffic_frames.jsonl"),
            },
        }


def run_traffic(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    incident_plan_path: Path | None = None,
    semantic_plan_root: Path = DEFAULT_SEMANTIC_PLAN_ROOT,
    scenarios_root: Path = DEFAULT_SCENARIOS_ROOT,
    net_xml: Path = DEFAULT_SUMO_NET_XML,
    sumo_bin: Path | None = None,
    duration_s: float = DEFAULT_DURATION_S,
    warmup_s: float = DEFAULT_WARMUP_S,
    step_length_s: float = DEFAULT_STEP_LENGTH_S,
    sample_period_s: float = DEFAULT_SAMPLE_PERIOD_S,
    max_vehicles: int = DEFAULT_MAX_VEHICLES,
    seed: int = 7,
    overwrite: bool = False,
    coverage_scenarios: Sequence[str] = (),
    semantic_episode_scope: Sequence[str] = (),
) -> dict[str, Any]:
    episode_scope = [str(episode_id) for episode_id in semantic_episode_scope if str(episode_id)]
    if len(episode_scope) != 1:
        raise ValueError("vehicle SUMO authority is episode-local; pass exactly one --episode")
    sumo_bin = Path(sumo_bin) if sumo_bin is not None else sumo_executable_from_environment("sumo")
    output_dir = Path(output_dir)
    output_dir = _episode_output_dir(output_dir, episode_scope[0])
    if overwrite:
        for name in ("sumo_traffic_manifest.json", "sumo_incident_plan.json", "sumo_traffic_frames.jsonl"):
            path = output_dir / name
            if path.exists():
                path.unlink()
    planner = SumoGroundFlowPlanner(net_xml)
    if incident_plan_path is not None:
        incident_plan = load_incident_plan(incident_plan_path)
    else:
        incident_plan = build_incident_plan(
            scenarios_root=scenarios_root,
            net_xml=net_xml,
            planner=planner,
            duration_s=duration_s,
        )
    scenario_scope = sorted({_episode_scenario_id(episode_id) for episode_id in episode_scope})
    if scenario_scope:
        incident_plan = _filter_incident_plan_by_scenarios(incident_plan, scenario_scope)
    incident_plan = _offset_incident_plan_times(incident_plan, warmup_s=warmup_s)
    write_incident_plan(incident_plan, output_dir / "sumo_incident_plan.json")
    explicit_vehicle_plan = load_explicit_vehicle_plan(Path(semantic_plan_root), episode_scope[0])
    if explicit_vehicle_plan.get("traffic_geometry") != planner.traffic_geometry_provenance:
        raise ValueError(
            f"{episode_scope[0]}: explicit SUMO vehicle plan traffic geometry provenance "
            "does not match the runtime UE traffic bundle"
        )
    explicit_vehicle_plan = _apply_road_semantics_to_explicit_vehicle_plan(
        explicit_vehicle_plan,
        episode_dir=Path(semantic_plan_root) / episode_scope[0],
        planner=planner,
    )
    explicit_vehicle_ids = planned_vehicle_ids(explicit_vehicle_plan)
    if not explicit_vehicle_ids:
        raise RuntimeError(f"{episode_scope[0]}: explicit SUMO vehicle plan has no vehicles")
    if len(explicit_vehicle_ids) > int(max_vehicles):
        raise RuntimeError(
            f"{episode_scope[0]}: explicit SUMO vehicle plan has {len(explicit_vehicle_ids)} vehicles, "
            f"above safety cap {max_vehicles}"
        )
    explicit_vehicle_instructions = instructions_from_explicit_plan(
        explicit_vehicle_plan,
        warmup_s=warmup_s,
        tick_hz=1.0 / step_length_s,
    )
    roi_lane_coverage_anchors = build_roi_lane_coverage_anchors(
        scenarios_root=scenarios_root,
        scenario_ids=coverage_scenarios,
    )
    runner = TrafficRunner(
        planner=planner,
        incident_plan=incident_plan,
        output_dir=output_dir,
        sumo_bin=sumo_bin,
        duration_s=duration_s,
        warmup_s=warmup_s,
        step_length_s=step_length_s,
        sample_period_s=sample_period_s,
        max_vehicles=max_vehicles,
        seed=seed,
        roi_lane_coverage_anchors=roi_lane_coverage_anchors,
        semantic_vehicle_instructions=explicit_vehicle_instructions,
        explicit_vehicle_plan=explicit_vehicle_plan,
        explicit_vehicle_ids=explicit_vehicle_ids,
    )
    manifest = runner.run()
    manifest.setdefault("semantic_vehicle_plans", {})["episode_scope"] = [
        str(episode_id) for episode_id in episode_scope
    ]
    manifest.setdefault("explicit_vehicle_plan", {})["path"] = str(
        Path(semantic_plan_root) / episode_scope[0] / "sumo_explicit_vehicle_plan.json"
    )
    manifest["incident_scenario_scope"] = scenario_scope
    _write_json(output_dir / "sumo_traffic_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Donghu SUMO traffic and export step-aligned samples.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--incident-plan", type=Path, default=None)
    parser.add_argument("--semantic-plan-root", type=Path, default=DEFAULT_SEMANTIC_PLAN_ROOT)
    parser.add_argument("--scenarios-root", type=Path, default=DEFAULT_SCENARIOS_ROOT)
    parser.add_argument("--net-xml", type=Path, default=DEFAULT_SUMO_NET_XML)
    parser.add_argument(
        "--sumo-bin",
        type=Path,
        default=None,
        help="SUMO executable; defaults to $SUMO_HOME/bin/sumo",
    )
    parser.add_argument("--capture-duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--warmup-s", type=float, default=DEFAULT_WARMUP_S)
    parser.add_argument("--step-length-s", type=float, default=DEFAULT_STEP_LENGTH_S)
    parser.add_argument("--sample-period-s", type=float, default=DEFAULT_SAMPLE_PERIOD_S)
    parser.add_argument("--max-vehicles", type=int, default=DEFAULT_MAX_VEHICLES)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--coverage-scenario", action="append", default=[])
    parser.add_argument("--episode", action="append", default=[], help="Episode id for the episode-local vehicle SUMO run. Exactly one is required.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_traffic(
        output_dir=args.output_dir,
        incident_plan_path=args.incident_plan,
        semantic_plan_root=args.semantic_plan_root,
        scenarios_root=args.scenarios_root,
        net_xml=args.net_xml,
        sumo_bin=args.sumo_bin,
        duration_s=args.capture_duration_s,
        warmup_s=args.warmup_s,
        step_length_s=args.step_length_s,
        sample_period_s=args.sample_period_s,
        max_vehicles=args.max_vehicles,
        seed=args.seed,
        overwrite=args.overwrite,
        coverage_scenarios=args.coverage_scenario,
        semantic_episode_scope=args.episode,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
