"""Integrate exported SUMO traffic samples into render-ready truth frames."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMO_OUTPUT_DIR = ROOT / "Dataset" / "sumo_outputs" / "donghu_traffic_270s"
SEGMENT_DURATION_S = 90.0
SEGMENT_COUNT = 3
DEFAULT_MIN_VISIBLE_VEHICLES = 8
DEFAULT_MAX_VISIBLE_VEHICLES = 36
DEFAULT_MIN_BACKGROUND_SELECTION_XY_SPAN_M = 2.0
DEFAULT_MIN_BACKGROUND_SELECTION_SPEED_MPS = 0.5


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return rows


def _round_vector(values: Sequence[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def _position2(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _position3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        z = float(value[2]) if len(value) >= 3 else 0.0
        return float(value[0]), float(value[1]), z
    except (TypeError, ValueError):
        return None


def _clean_entity_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def sumo_truth_entity_id(vehicle_id: str) -> str:
    return f"sumo_vehicle_{_clean_entity_token(vehicle_id)}"


def infer_seed_index(episode_id: str, manifest: dict[str, Any]) -> int:
    raw_seed = manifest.get("seed")
    if raw_seed not in (None, ""):
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{episode_id}: invalid manifest seed {raw_seed!r}") from exc
    else:
        match = re.search(r"__seed(\d+)$", episode_id)
        if not match:
            raise ValueError(f"{episode_id}: cannot infer seed index from episode id")
        seed = int(match.group(1))
    if seed < 0 or seed >= SEGMENT_COUNT:
        raise ValueError(f"{episode_id}: SUMO integration requires seed00, seed01, or seed02; found seed{seed:02d}")
    return seed


@dataclass(frozen=True)
class SumoSegment:
    seed_index: int
    segment_start_s: float
    segment_end_s: float
    duration_s: float
    phase_name: str

    @property
    def seed_label(self) -> str:
        return f"seed{self.seed_index:02d}"

    def absolute_time_s(self, episode_sim_time_s: float) -> float:
        local_time = max(0.0, min(self.duration_s, float(episode_sim_time_s)))
        return self.segment_start_s + local_time

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed_index": self.seed_index,
            "seed_label": self.seed_label,
            "segment_start_s": round(self.segment_start_s, 6),
            "segment_end_s": round(self.segment_end_s, 6),
            "duration_s": round(self.duration_s, 6),
            "phase_name": self.phase_name,
        }


def segment_for_seed(seed_index: int) -> SumoSegment:
    phase_names = {
        0: "vehicle_count_ramp_up",
        1: "vehicle_count_peak_hold",
        2: "vehicle_count_ramp_down",
    }
    start_s = float(seed_index) * SEGMENT_DURATION_S
    return SumoSegment(
        seed_index=seed_index,
        segment_start_s=start_s,
        segment_end_s=start_s + SEGMENT_DURATION_S,
        duration_s=SEGMENT_DURATION_S,
        phase_name=phase_names[seed_index],
    )


def point_in_polygon_xy(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
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
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def _distance_to_polyline_xy(point: Sequence[float], points: Sequence[Sequence[float]]) -> float:
    if not points:
        return float("inf")
    if len(points) == 1:
        return math.hypot(float(point[0]) - float(points[0][0]), float(point[1]) - float(points[0][1]))
    return min(_distance_point_to_segment_xy(point, a, b) for a, b in zip(points, points[1:]))


def _distance_to_polygon_xy(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> float:
    if not polygon:
        return float("inf")
    if point_in_polygon_xy(point, polygon):
        return 0.0
    return min(
        _distance_point_to_segment_xy(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _point_in_oriented_box_xy(
    point: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
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


@dataclass(frozen=True)
class VisibilityGeometry:
    capture_polygon_enu_m: tuple[tuple[float, float], ...]
    inspect_route_enu_m: tuple[tuple[float, float, float], ...]
    hfov_deg: float
    vfov_deg: float
    altitude_m: float
    padding_m: float = 25.0

    @classmethod
    def from_contract(cls, source_contract: dict[str, Any]) -> "VisibilityGeometry":
        polygon: list[tuple[float, float]] = []
        for point in source_contract.get("capture_boundary_polygon_enu_m") or []:
            pos = _position2(point)
            if pos is not None:
                polygon.append(pos)
        route: list[tuple[float, float, float]] = []
        for point in source_contract.get("inspect_route_enu_m") or []:
            pos3 = _position3(point)
            if pos3 is not None:
                route.append(pos3)
        inspect_contract = dict(source_contract.get("inspect_contract") or {})
        profile = dict(inspect_contract.get("sensor_profile") or {})
        hfov_deg = float(profile.get("hfov_deg") or profile.get("FOV_Degrees") or profile.get("fov_degrees") or 90.0)
        width = float(profile.get("width") or 1920.0)
        height = float(profile.get("height") or 1080.0)
        vfov_deg = float(profile.get("vfov_deg") or 0.0)
        if vfov_deg <= 0.0 and width > 0.0 and height > 0.0:
            vfov_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov_deg / 2.0)) * (height / width)))
        altitude_m = float(inspect_contract.get("inspect_altitude_m") or (route[0][2] if route else 36.0))
        return cls(
            capture_polygon_enu_m=tuple(polygon),
            inspect_route_enu_m=tuple(route),
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            altitude_m=altitude_m,
        )

    @property
    def footprint_half_width_m(self) -> float:
        return math.tan(math.radians(self.hfov_deg / 2.0)) * self.altitude_m

    @property
    def footprint_half_height_m(self) -> float:
        return math.tan(math.radians(self.vfov_deg / 2.0)) * self.altitude_m

    def observation_distance_m(self, point: Sequence[float]) -> float:
        if self.capture_polygon_enu_m and point_in_polygon_xy(point, self.capture_polygon_enu_m):
            return 0.0
        half_width = self.footprint_half_width_m
        half_height = self.footprint_half_height_m
        for a, b in zip(self.inspect_route_enu_m, self.inspect_route_enu_m[1:]):
            if _point_in_oriented_box_xy(point, a, b, half_width, half_height):
                return 0.0
        return min(
            _distance_to_polygon_xy(point, self.capture_polygon_enu_m),
            _distance_to_polyline_xy(point, self.inspect_route_enu_m),
        )

    def is_observable(self, point: Sequence[float]) -> bool:
        return self.observation_distance_m(point) <= self.padding_m

    def as_dict(self) -> dict[str, Any]:
        return {
            "capture_polygon_points": len(self.capture_polygon_enu_m),
            "inspect_route_points": len(self.inspect_route_enu_m),
            "hfov_deg": round(self.hfov_deg, 6),
            "vfov_deg": round(self.vfov_deg, 6),
            "altitude_m": round(self.altitude_m, 6),
            "padding_m": round(self.padding_m, 6),
            "footprint_half_width_m": round(self.footprint_half_width_m, 6),
            "footprint_half_height_m": round(self.footprint_half_height_m, 6),
        }


@dataclass(frozen=True)
class VehicleSelection:
    vehicle_ids: tuple[str, ...]
    entity_ids: dict[str, str]
    min_distance_m_by_vehicle_id: dict[str, float]
    frames_seen_by_vehicle_id: dict[str, int]
    motion_span_m_by_vehicle_id: dict[str, float]
    max_speed_mps_by_vehicle_id: dict[str, float]
    scenario_vehicle_ids: tuple[str, ...]
    candidate_count: int
    moving_candidate_count: int
    selected_count: int
    min_visible_vehicle_target: int
    max_visible_vehicle_target: int
    expanded_to_nearest: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "vehicle_ids": list(self.vehicle_ids),
            "entity_ids": dict(self.entity_ids),
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "scenario_vehicle_ids": list(self.scenario_vehicle_ids),
            "min_visible_vehicle_target": self.min_visible_vehicle_target,
            "max_visible_vehicle_target": self.max_visible_vehicle_target,
            "expanded_to_nearest": self.expanded_to_nearest,
            "min_distance_m_by_vehicle_id": {
                key: round(float(value), 6) for key, value in sorted(self.min_distance_m_by_vehicle_id.items())
            },
            "frames_seen_by_vehicle_id": dict(sorted(self.frames_seen_by_vehicle_id.items())),
            "motion_span_m_by_vehicle_id": {
                key: round(float(value), 6) for key, value in sorted(self.motion_span_m_by_vehicle_id.items())
            },
            "max_speed_mps_by_vehicle_id": {
                key: round(float(value), 6) for key, value in sorted(self.max_speed_mps_by_vehicle_id.items())
            },
        }


class SumoTrafficDataset:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.frames_path = self.output_dir / "sumo_traffic_frames.jsonl"
        self.manifest_path = self.output_dir / "sumo_traffic_manifest.json"
        self.incident_plan_path = self.output_dir / "sumo_incident_plan.json"
        missing = [path for path in (self.frames_path, self.manifest_path, self.incident_plan_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing SUMO output files: {missing}")
        self.manifest = _read_json(self.manifest_path)
        self.incident_plan = _read_json(self.incident_plan_path)
        self.frames = sorted(_read_jsonl(self.frames_path), key=lambda row: float(row.get("sim_time_s") or 0.0))
        if not self.frames:
            raise RuntimeError(f"No SUMO traffic frames in {self.frames_path}")
        self.times = [float(row.get("sim_time_s") or 0.0) for row in self.frames]
        self.incidents = list(self.incident_plan.get("incidents") or [])

    def segment_for_episode(self, episode_id: str, manifest: dict[str, Any]) -> SumoSegment:
        return segment_for_seed(infer_seed_index(episode_id, manifest))

    def scenario_incidents(self, scenario_id: str) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in self.incidents
            if str(item.get("episode_scenario_id") or "") == str(scenario_id)
        ]

    def active_incidents_at(self, absolute_time_s: float, scenario_id: str | None = None) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for item in self.incidents:
            if scenario_id is not None and str(item.get("episode_scenario_id") or "") != str(scenario_id):
                continue
            if float(item.get("start_s") or 0.0) <= float(absolute_time_s) <= float(item.get("end_s") or 0.0):
                active.append(_compact_incident(item))
        return active

    def _frame_pair(self, absolute_time_s: float) -> tuple[dict[str, Any], dict[str, Any], float]:
        time_s = max(self.times[0], min(self.times[-1], float(absolute_time_s)))
        index = bisect.bisect_right(self.times, time_s)
        if index <= 0:
            return self.frames[0], self.frames[0], 0.0
        if index >= len(self.frames):
            return self.frames[-1], self.frames[-1], 0.0
        prev_frame = self.frames[index - 1]
        next_frame = self.frames[index]
        prev_time = float(prev_frame.get("sim_time_s") or 0.0)
        next_time = float(next_frame.get("sim_time_s") or prev_time)
        if abs(time_s - prev_time) <= 1e-9:
            return prev_frame, prev_frame, 0.0
        span = max(1e-9, next_time - prev_time)
        alpha = max(0.0, min(1.0, (time_s - prev_time) / span))
        return prev_frame, next_frame, alpha

    def sample(
        self,
        *,
        segment: SumoSegment,
        episode_sim_time_s: float,
        selected_vehicle_ids: Iterable[str],
        scenario_id: str,
    ) -> dict[str, Any]:
        absolute_time_s = segment.absolute_time_s(episode_sim_time_s)
        prev_frame, next_frame, alpha = self._frame_pair(absolute_time_s)
        prev_time = float(prev_frame.get("sim_time_s") or absolute_time_s)
        next_time = float(next_frame.get("sim_time_s") or prev_time)
        selected = set(selected_vehicle_ids)
        prev_by_id = {str(item.get("vehicle_id") or ""): item for item in prev_frame.get("vehicles") or []}
        next_by_id = {str(item.get("vehicle_id") or ""): item for item in next_frame.get("vehicles") or []}
        vehicle_ids = sorted((set(prev_by_id) | set(next_by_id)) & selected)
        vehicles = [
            _interpolate_vehicle(prev_by_id.get(vehicle_id), next_by_id.get(vehicle_id), alpha, prev_time, next_time)
            for vehicle_id in vehicle_ids
        ]
        light_source = prev_frame if alpha < 0.5 else next_frame
        return {
            "absolute_time_s": round(absolute_time_s, 6),
            "source_prev_time_s": round(prev_time, 6),
            "source_next_time_s": round(next_time, 6),
            "source_alpha": round(alpha, 6),
            "source_vehicle_count": len(set(prev_by_id) | set(next_by_id)),
            "vehicles": [vehicle for vehicle in vehicles if vehicle is not None],
            "traffic_lights": _traffic_light_states(light_source),
            "active_incidents": self.active_incidents_at(absolute_time_s, scenario_id=scenario_id),
        }

    def first_vehicle_record_in_segment(self, segment: SumoSegment, vehicle_id: str) -> dict[str, Any] | None:
        for frame in self.frames:
            time_s = float(frame.get("sim_time_s") or 0.0)
            if time_s < segment.segment_start_s or time_s > segment.segment_end_s:
                continue
            for vehicle in frame.get("vehicles") or []:
                if str(vehicle.get("vehicle_id") or "") == vehicle_id:
                    return dict(vehicle)
        return None

    def select_visible_vehicles(
        self,
        *,
        segment: SumoSegment,
        visibility: VisibilityGeometry,
        scenario_id: str,
        min_visible: int = DEFAULT_MIN_VISIBLE_VEHICLES,
        max_visible: int = DEFAULT_MAX_VISIBLE_VEHICLES,
    ) -> VehicleSelection:
        min_distance_by_id: dict[str, float] = {}
        frames_seen_by_id: dict[str, int] = {}
        bounds_by_id: dict[str, list[float]] = {}
        max_speed_by_id: dict[str, float] = {}
        observable_ids: set[str] = set()
        scenario_ids: set[str] = set()
        scenario_incidents = self.scenario_incidents(scenario_id)
        affected_ids = {
            str(vehicle_id)
            for incident in scenario_incidents
            for vehicle_id in (incident.get("affected_vehicle_ids") or [])
            if str(vehicle_id)
        }
        for frame in self.frames:
            time_s = float(frame.get("sim_time_s") or 0.0)
            if time_s < segment.segment_start_s or time_s > segment.segment_end_s:
                continue
            for vehicle in frame.get("vehicles") or []:
                vehicle_id = str(vehicle.get("vehicle_id") or "")
                if not vehicle_id:
                    continue
                if str(vehicle.get("control_role") or "") == "incident_controlled" and vehicle_id not in affected_ids:
                    continue
                point = _position2(vehicle.get("truth_position_enu_m"))
                if point is None:
                    continue
                distance_m = visibility.observation_distance_m(point)
                min_distance_by_id[vehicle_id] = min(distance_m, min_distance_by_id.get(vehicle_id, float("inf")))
                frames_seen_by_id[vehicle_id] = frames_seen_by_id.get(vehicle_id, 0) + 1
                if vehicle_id not in bounds_by_id:
                    bounds_by_id[vehicle_id] = [point[0], point[0], point[1], point[1]]
                else:
                    bounds = bounds_by_id[vehicle_id]
                    bounds[0] = min(bounds[0], point[0])
                    bounds[1] = max(bounds[1], point[0])
                    bounds[2] = min(bounds[2], point[1])
                    bounds[3] = max(bounds[3], point[1])
                max_speed_by_id[vehicle_id] = max(
                    max_speed_by_id.get(vehicle_id, 0.0),
                    float(vehicle.get("speed_mps") or 0.0),
                )
                if distance_m <= visibility.padding_m:
                    observable_ids.add(vehicle_id)
                if vehicle_id in affected_ids:
                    scenario_ids.add(vehicle_id)

        motion_span_by_id = {
            vehicle_id: math.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2])
            for vehicle_id, bounds in bounds_by_id.items()
        }
        moving_ids = {
            vehicle_id
            for vehicle_id in min_distance_by_id
            if motion_span_by_id.get(vehicle_id, 0.0) >= DEFAULT_MIN_BACKGROUND_SELECTION_XY_SPAN_M
            or max_speed_by_id.get(vehicle_id, 0.0) >= DEFAULT_MIN_BACKGROUND_SELECTION_SPEED_MPS
        }

        selected: list[str] = []
        selected_set: set[str] = set()
        for vehicle_id in sorted(scenario_ids, key=lambda item: (min_distance_by_id.get(item, float("inf")), item)):
            selected.append(vehicle_id)
            selected_set.add(vehicle_id)

        primary = sorted(
            (observable_ids & moving_ids) - selected_set,
            key=lambda item: (
                min_distance_by_id.get(item, float("inf")),
                -frames_seen_by_id.get(item, 0),
                item,
            ),
        )
        for vehicle_id in primary:
            if len(selected) >= max_visible:
                break
            selected.append(vehicle_id)
            selected_set.add(vehicle_id)

        expanded = False
        if len(selected) < min_visible:
            expanded = True
            nearest = sorted(
                (set(min_distance_by_id) & moving_ids) - selected_set,
                key=lambda item: (
                    min_distance_by_id.get(item, float("inf")),
                    -frames_seen_by_id.get(item, 0),
                    item,
                ),
            )
            for vehicle_id in nearest:
                if len(selected) >= min_visible:
                    break
                selected.append(vehicle_id)
                selected_set.add(vehicle_id)

        entity_ids = {vehicle_id: sumo_truth_entity_id(vehicle_id) for vehicle_id in selected}
        return VehicleSelection(
            vehicle_ids=tuple(selected),
            entity_ids=entity_ids,
            min_distance_m_by_vehicle_id={vehicle_id: min_distance_by_id.get(vehicle_id, float("inf")) for vehicle_id in selected},
            frames_seen_by_vehicle_id={vehicle_id: frames_seen_by_id.get(vehicle_id, 0) for vehicle_id in selected},
            motion_span_m_by_vehicle_id={vehicle_id: motion_span_by_id.get(vehicle_id, 0.0) for vehicle_id in selected},
            max_speed_mps_by_vehicle_id={vehicle_id: max_speed_by_id.get(vehicle_id, 0.0) for vehicle_id in selected},
            scenario_vehicle_ids=tuple(sorted(scenario_ids & selected_set)),
            candidate_count=len(min_distance_by_id),
            moving_candidate_count=len(moving_ids),
            selected_count=len(selected),
            min_visible_vehicle_target=int(min_visible),
            max_visible_vehicle_target=int(max_visible),
            expanded_to_nearest=expanded,
        )

    def source_summary(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "frames": str(self.frames_path),
            "manifest": str(self.manifest_path),
            "incident_plan": str(self.incident_plan_path),
            "map_id": self.manifest.get("map_id"),
            "duration_s": self.manifest.get("duration_s"),
            "sample_period_s": self.manifest.get("sample_period_s"),
            "sample_count": self.manifest.get("sample_count"),
            "max_vehicles": self.manifest.get("max_vehicles"),
            "coordinate_mapper": self.manifest.get("coordinate_mapper"),
        }


def _traffic_light_states(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for item in frame.get("traffic_lights") or []:
        tls_id = str(item.get("tls_id") or "")
        if not tls_id:
            continue
        states[tls_id] = {
            "state": str(item.get("state") or ""),
            "phase_index": int(item.get("phase_index") or 0),
            "program_id": str(item.get("program_id") or ""),
            "next_switch_s": round(float(item.get("next_switch_s") or 0.0), 6),
        }
    return dict(sorted(states.items()))


def _compact_incident(item: dict[str, Any]) -> dict[str, Any]:
    anchor = dict(item.get("anchor") or {})
    return {
        "incident_id": item.get("incident_id"),
        "episode_scenario_id": item.get("episode_scenario_id"),
        "episode_event_id": item.get("episode_event_id"),
        "intent": item.get("intent"),
        "intent_stage": item.get("intent_stage"),
        "accident_class": item.get("accident_class"),
        "start_s": item.get("start_s"),
        "end_s": item.get("end_s"),
        "injection_method": item.get("injection_method"),
        "affected_vehicle_ids": list(item.get("affected_vehicle_ids") or []),
        "anchor": {
            "sumo_edge_id": anchor.get("sumo_edge_id"),
            "sumo_lane_id": anchor.get("sumo_lane_id"),
            "truth_position_enu_m": anchor.get("truth_position_enu_m") or anchor.get("projected_truth_xy_m"),
            "geometry_source": anchor.get("geometry_source"),
        },
    }


def _lerp_angle_deg(a: float, b: float, alpha: float) -> float:
    delta = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    value = float(a) + delta * float(alpha)
    while value > 180.0:
        value -= 360.0
    while value <= -180.0:
        value += 360.0
    return value


def _velocity_from_yaw(speed_mps: float, yaw_deg: float) -> list[float]:
    radians = math.radians(float(yaw_deg))
    return [float(speed_mps) * math.cos(radians), float(speed_mps) * math.sin(radians), 0.0]


def _interpolate_vehicle(
    prev_vehicle: dict[str, Any] | None,
    next_vehicle: dict[str, Any] | None,
    alpha: float,
    prev_time_s: float,
    next_time_s: float,
) -> dict[str, Any] | None:
    if prev_vehicle is None and next_vehicle is None:
        return None
    if prev_vehicle is None:
        source = dict(next_vehicle or {})
        position = _position3(source.get("truth_position_enu_m")) or (0.0, 0.0, 0.0)
        yaw = float(source.get("truth_yaw_deg") or 0.0)
        velocity = _velocity_from_yaw(float(source.get("speed_mps") or 0.0), yaw)
    elif next_vehicle is None or prev_vehicle is next_vehicle:
        source = dict(prev_vehicle)
        position = _position3(source.get("truth_position_enu_m")) or (0.0, 0.0, 0.0)
        yaw = float(source.get("truth_yaw_deg") or 0.0)
        velocity = _velocity_from_yaw(float(source.get("speed_mps") or 0.0), yaw)
    else:
        source = dict(prev_vehicle if alpha < 0.5 else next_vehicle)
        prev_pos = _position3(prev_vehicle.get("truth_position_enu_m")) or (0.0, 0.0, 0.0)
        next_pos = _position3(next_vehicle.get("truth_position_enu_m")) or prev_pos
        position = tuple(prev_pos[i] + (next_pos[i] - prev_pos[i]) * float(alpha) for i in range(3))
        dt = max(1e-9, float(next_time_s) - float(prev_time_s))
        velocity = [(next_pos[i] - prev_pos[i]) / dt for i in range(3)]
        if math.hypot(velocity[0], velocity[1]) > 1e-5:
            yaw = math.degrees(math.atan2(velocity[1], velocity[0]))
        else:
            yaw = _lerp_angle_deg(float(prev_vehicle.get("truth_yaw_deg") or 0.0), float(next_vehicle.get("truth_yaw_deg") or 0.0), alpha)
    source["truth_position_enu_m"] = _round_vector(position)
    source["truth_yaw_deg"] = round(float(yaw), 6)
    source["velocity_enu_mps"] = _round_vector(velocity)
    source["source_prev_time_s"] = round(float(prev_time_s), 6)
    source["source_next_time_s"] = round(float(next_time_s), 6)
    source["source_alpha"] = round(float(alpha), 6)
    return source


_DATASET_CACHE: dict[Path, SumoTrafficDataset] = {}


def load_sumo_traffic_dataset(output_dir: Path = DEFAULT_SUMO_OUTPUT_DIR) -> SumoTrafficDataset:
    resolved = output_dir.resolve()
    dataset = _DATASET_CACHE.get(resolved)
    if dataset is None:
        dataset = SumoTrafficDataset(resolved)
        _DATASET_CACHE[resolved] = dataset
    return dataset
