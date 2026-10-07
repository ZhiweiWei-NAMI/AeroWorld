"""Generate Donghu city-scale UAV traffic samples.

The UAV layer is deliberately generated before episode-local truth conversion.
It provides persistent city activity, pad/corridor patrols, logistics orders,
inspection tasks, relay tasks, and incident-response hooks at the same 10 Hz
tick convention used by the rest of the dataset.  The exported stream is sampled
every 0.5 s, i.e. every 5 ticks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import heapq
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ROOT / "Dataset" / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from Dataset.tools.map_spatial_index import MapSpatialIndex, offset_from_lane  # noqa: E402
from Dataset.tools.sumo_ground_flow.incident_plan import DEFAULT_SUMO_NET_XML  # noqa: E402
from Dataset.tools.sumo_ground_flow.spatial_event_grid import SpatialEventGridPlanner, SpatialGridCell  # noqa: E402
from Dataset.tools.uav_corridor_planner import BuildingObstacleIndex  # noqa: E402
from Dataset.semantic_truth.facility_scope import semantic_scope_for_source  # noqa: E402
from shapely.geometry import LineString, Point  # noqa: E402


MAP_ID = "donghu_road_topo"
SCHEMA_VERSION = "v1"
DEFAULT_OUTPUT_DIR = ROOT / "aw_data" / "uav_outputs" / "donghu_uav_flow_270s"
DEFAULT_SCENARIOS_ROOT = ROOT / "Dataset" / "scenarios"
DEFAULT_SUMO_OUTPUT_DIR = ROOT / "aw_data" / "sumo_outputs" / "episode_vehicle_authority"
DURATION_S = 270.0
TICK_HZ = 10
STEP_LENGTH_S = 0.1
SAMPLE_PERIOD_S = 0.5
SAMPLE_EVERY_TICKS = 5
PHASE_DURATION_S = 90.0
PAD_COUNT = 20
PAD_PATROL_COUNT = 20
INTERSECTION_INSPECT_COUNT = 18
RELAY_COUNT = 14
MIN_ACTIVE_UAVS = 50
DELIVERY_ORDERS_BY_PHASE = (45, 60, 45)
INFRA_INSPECTIONS_BY_PHASE = 6
GLOBAL_SEED = 424242
MIN_DYNAMIC_SEPARATION_M = 2.0
LANDING_PAD_ALTITUDE_M = 1.2
LANDING_APPROACH_ALTITUDE_M = 12.0
LANDING_VERTICAL_WAYPOINT_STEP_M = 3.0
DELIVERY_TERMINAL_HOLD_S = SAMPLE_PERIOD_S
GROUND_REFERENCE_Z_M = 0.0

ALLOWED_ALTITUDE_LAYERS_M = (18.0, 22.0, 28.0, 36.0, 50.0, 80.0, 110.0, 130.0)


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _project_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _round_point(point: Sequence[float], digits: int = 6) -> list[float]:
    z = float(point[2]) if len(point) >= 3 else 0.0
    return [_round(point[0], digits), _round(point[1], digits), _round(z, digits)]


def _dist_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _dist3(a: Sequence[float], b: Sequence[float]) -> float:
    az = float(a[2]) if len(a) >= 3 else 0.0
    bz = float(b[2]) if len(b) >= 3 else 0.0
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (az - bz) ** 2)


def _path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(_dist3(a, b) for a, b in zip(points, points[1:]))


def _xy_path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(_dist_xy(a, b) for a, b in zip(points, points[1:]))


def _dedupe(points: Iterable[Sequence[float]], *, min_gap_m: float = 0.05) -> list[list[float]]:
    result: list[list[float]] = []
    for raw in points:
        point = _round_point(raw, 3)
        if result and _dist3(result[-1], point) < min_gap_m:
            continue
        result.append(point)
    return result


def _vertical_waypoints(
    anchor: Sequence[float],
    start_z_m: float,
    end_z_m: float,
    *,
    max_step_m: float = LANDING_VERTICAL_WAYPOINT_STEP_M,
) -> list[list[float]]:
    span = abs(float(end_z_m) - float(start_z_m))
    if span <= 1e-6:
        return [[float(anchor[0]), float(anchor[1]), float(start_z_m)]]
    steps = max(1, int(math.ceil(span / max(0.1, float(max_step_m)))))
    return [
        [
            float(anchor[0]),
            float(anchor[1]),
            float(start_z_m) + (float(end_z_m) - float(start_z_m)) * (index / steps),
        ]
        for index in range(steps + 1)
    ]


def _stable_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _stable_int(value: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    return int(_stable_unit_interval(value) * modulo) % modulo


@dataclass(frozen=True)
class Pad:
    pad_id: str
    role: str
    position_enu_m: list[float]
    grid_cell_id: str
    lane_edge_id: str
    lane_s_m: float
    semantic_scope: dict[str, Any]
    phase_origin_weights: tuple[float, float, float]
    phase_destination_weights: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UavTask:
    task_id: str
    uav_id: str
    mission_type: str
    semantic_role: str
    corridor_family: str
    corridor_id: str
    altitude_layer_m: float
    route_waypoints_enu_m: list[list[float]]
    speed_mps: float
    start_s: float
    end_s: float
    looping: bool
    terminal_hold_s: float = 0.0
    phase_index: int | None = None
    origin_pad_id: str = ""
    target_pad_id: str = ""
    target_cell_id: str = ""
    active_event_ids: tuple[str, ...] = field(default_factory=tuple)
    source: str = "uav_global_flow"
    dynamic_avoidance_policy: str = "altitude_layer_and_corridor_phase_offset_v1"
    loop_phase_offset_override_m: float | None = None

    @property
    def route_length_m(self) -> float:
        return _path_length(self.route_waypoints_enu_m)

    @property
    def xy_route_length_m(self) -> float:
        return _xy_path_length(self.route_waypoints_enu_m)

    @property
    def phase_offset_m(self) -> float:
        if not self.looping:
            return 0.0
        if self.loop_phase_offset_override_m is not None:
            return float(self.loop_phase_offset_override_m)
        length = max(self.route_length_m, 1.0)
        return _stable_unit_interval(self.task_id) * length

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["route_length_m"] = _round(self.route_length_m)
        payload["xy_route_length_m"] = _round(self.xy_route_length_m)
        payload["expected_duration_s"] = _round(max(0.0, self.end_s - self.start_s))
        return payload


@dataclass(frozen=True)
class EventBinding:
    scenario_id: str
    center_enu_m: list[float]
    required_intents: tuple[str, ...]
    capture_boundary_id: str
    recommended_task_family: str
    phase_index: int
    source_event_script: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RouteSampler:
    def __init__(self, task: UavTask) -> None:
        self.task = task
        self.points = task.route_waypoints_enu_m
        if len(self.points) < 2:
            raise ValueError(f"{task.task_id} has fewer than 2 route points")
        self.segment_lengths = [_dist3(a, b) for a, b in zip(self.points, self.points[1:])]
        self.total_length = sum(self.segment_lengths)
        if self.total_length <= 1e-6:
            raise ValueError(f"{task.task_id} route has zero length")

    def active_at(self, sim_time_s: float) -> bool:
        if self.task.looping:
            return self.task.start_s <= sim_time_s <= self.task.end_s
        if self.task.terminal_hold_s > 0.0:
            return self.task.start_s <= sim_time_s <= self.task.end_s + self.task.terminal_hold_s + 1e-9
        return self.task.start_s <= sim_time_s < self.task.end_s

    def sample(self, sim_time_s: float) -> tuple[list[float], float, float]:
        if self.task.looping:
            distance = ((sim_time_s - self.task.start_s) * self.task.speed_mps + self.task.phase_offset_m) % self.total_length
            speed = self.task.speed_mps
        elif sim_time_s >= self.task.end_s:
            distance = self.total_length
            speed = 0.0
        else:
            distance = max(0.0, min((sim_time_s - self.task.start_s) * self.task.speed_mps, self.total_length))
            speed = self.task.speed_mps
        position, yaw = self._point_at_distance(distance)
        return position, yaw, speed

    def _point_at_distance(self, distance_m: float) -> tuple[list[float], float]:
        remaining = max(0.0, float(distance_m))
        for segment_length, a, b in zip(self.segment_lengths, self.points, self.points[1:]):
            if segment_length <= 1e-9:
                continue
            if remaining <= segment_length:
                t = remaining / segment_length
                point = [
                    float(a[0]) + (float(b[0]) - float(a[0])) * t,
                    float(a[1]) + (float(b[1]) - float(a[1])) * t,
                    float(a[2]) + (float(b[2]) - float(a[2])) * t,
                ]
                yaw = math.degrees(math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0])))
                return _round_point(point), _round(yaw, 3)
            remaining -= segment_length
        a, b = self.points[-2], self.points[-1]
        yaw = math.degrees(math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0])))
        return _round_point(self.points[-1]), _round(yaw, 3)


class DonghuUavFlowBuilder:
    def __init__(
        self,
        *,
        output_dir: Path,
        scenarios_root: Path,
        sumo_output_dir: Path,
        seed: int = GLOBAL_SEED,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.scenarios_root = Path(scenarios_root)
        self.sumo_output_dir = Path(sumo_output_dir)
        self.seed = int(seed)
        self.spatial = MapSpatialIndex.default(ROOT)
        self.buildings = BuildingObstacleIndex(self.spatial)
        self.grid = SpatialEventGridPlanner.default()
        self.pads: list[Pad] = []
        self.tasks: list[UavTask] = []
        self.event_bindings: list[EventBinding] = []
        self.delivery_overrun_dropped_task_ids: list[str] = []
        self._route_clear_cache: dict[tuple[tuple[float, float, float], ...], bool] = {}
        self._route_clear_strict_cache: dict[tuple[tuple[float, float, float], ...], bool] = {}
        self._transit_route_cache: dict[tuple[tuple[float, float, float], tuple[float, float, float]], list[list[float]]] = {}

    def build(self) -> dict[str, Any]:
        self.pads = self._select_pads()
        self.tasks.extend(self._build_pad_patrol_tasks())
        self.tasks.extend(self._build_intersection_inspect_tasks())
        self.tasks.extend(self._build_relay_tasks())
        self.tasks.extend(self._build_delivery_tasks())
        self.tasks.extend(self._build_infrastructure_tasks())
        self.tasks.extend(self._build_incident_response_tasks())
        self._resolve_dynamic_conflicts()
        self._drop_overrun_delivery_tasks()
        self.event_bindings = self._build_event_bindings()

        frames_path = self.output_dir / "uav_traffic_frames.jsonl"
        task_plan_path = self.output_dir / "uav_task_plan.json"
        manifest_path = self.output_dir / "uav_flow_manifest.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sample_count = self._write_frames(frames_path)
        task_plan = self._task_plan_payload()
        manifest = self._manifest_payload(sample_count=sample_count, frames_path=frames_path, task_plan_path=task_plan_path)
        _write_json(task_plan_path, task_plan)
        _write_json(manifest_path, manifest)
        return manifest

    def _phase_centers(self) -> tuple[list[float], list[float], list[float]]:
        cells = self.grid.cells
        min_x = min(cell.representative_enu_m[0] for cell in cells)
        max_x = max(cell.representative_enu_m[0] for cell in cells)
        min_y = min(cell.representative_enu_m[1] for cell in cells)
        max_y = max(cell.representative_enu_m[1] for cell in cells)
        return (
            [min_x + 0.25 * (max_x - min_x), min_y + 0.75 * (max_y - min_y), 0.0],
            [min_x + 0.62 * (max_x - min_x), min_y + 0.35 * (max_y - min_y), 0.0],
            [min_x + 0.32 * (max_x - min_x), min_y + 0.22 * (max_y - min_y), 0.0],
        )

    def _phase_weight(self, point: Sequence[float], center: Sequence[float], sigma_m: float) -> float:
        d = _dist_xy(point, center)
        return math.exp(-(d * d) / (2.0 * sigma_m * sigma_m)) + 0.08

    def _select_pads(self) -> list[Pad]:
        phase_centers = self._phase_centers()
        selected: list[Pad] = []
        errors: list[str] = []
        candidate_cells = sorted(
            self.grid.cells,
            key=lambda cell: (
                cell.cell_id in self.grid.incident_cell_ids,
                -cell.main_edge_count,
                -cell.total_main_edge_length_m,
                cell.cell_id,
            ),
        )
        candidate_cache: dict[tuple[str, int], Pad | None] = {}
        for min_spacing_m in (125.0, 70.0, 45.0, 25.0):
            for cell in candidate_cells:
                if len(selected) >= PAD_COUNT:
                    break
                if any(cell.cell_id == pad.grid_cell_id for pad in selected):
                    continue
                cache_key = (cell.cell_id, len(selected))
                if cache_key in candidate_cache:
                    pad = candidate_cache[cache_key]
                else:
                    pad = self._candidate_pad_for_cell(cell, len(selected), phase_centers, errors)
                    candidate_cache[cache_key] = pad
                if pad is not None:
                    if any(_dist_xy(pad.position_enu_m, existing.position_enu_m) < min_spacing_m for existing in selected):
                        continue
                    selected.append(pad)
            if len(selected) >= PAD_COUNT:
                break
        if len(selected) != PAD_COUNT:
            raise RuntimeError(f"Unable to select {PAD_COUNT} obstacle-clear pads; selected={len(selected)} errors={errors[:8]}")
        return selected

    def _candidate_pad_for_cell(
        self,
        cell: SpatialGridCell,
        index: int,
        phase_centers: tuple[list[float], list[float], list[float]],
        errors: list[str],
    ) -> Pad | None:
        for offset_m in (10.0, 14.0, 18.0, 24.0, 32.0):
            try:
                anchor = self.spatial.plan_sidewalk_anchor(
                    cell.representative_enu_m,
                    offset_from_curb_m=offset_m,
                    allow_green=True,
                    placement_semantics="uav_pad",
                )
            except Exception as exc:
                errors.append(f"{cell.cell_id}: {exc}")
                continue
            position = _round_point([anchor.position_enu_m[0], anchor.position_enu_m[1], 1.2], 3)
            if not (
                self.buildings.air_point_clear(position)
                and self.buildings.air_point_clear([position[0], position[1], 12.0])
                and self.buildings.air_point_clear([position[0], position[1], 18.0])
            ):
                continue
            patrol_altitude = 18.0 if index % 2 == 0 else 22.0
            if not self._find_lane_racetrack(
                position,
                altitude_m=patrol_altitude,
                min_length_m=100.0,
            ):
                continue
            origin_weights = tuple(
                _round(self._phase_weight(position, center, 470.0), 6)
                for center in phase_centers
            )
            dest_weights = tuple(
                _round(self._phase_weight(position, center, 620.0), 6)
                for center in reversed(phase_centers)
            )
            return Pad(
                pad_id=f"pad_{index:02d}",
                role="hub_charging_logistics" if index < 4 else "satellite_charging_logistics",
                position_enu_m=position,
                grid_cell_id=cell.cell_id,
                lane_edge_id=anchor.sample.edge_id,
                lane_s_m=_round(anchor.sample.s_m, 3),
                semantic_scope=semantic_scope_for_source(
                    category="facility",
                    asset_id="facility.landing_pad.visible.v1",
                    declared_subtype="landing_pad",
                ),
                phase_origin_weights=origin_weights,
                phase_destination_weights=dest_weights,
            )
        return None

    def _loop_points(
        self,
        center: Sequence[float],
        altitude_m: float,
        radius_x_m: float,
        radius_y_m: float,
        rotation_deg: float,
        point_count: int = 12,
    ) -> list[list[float]]:
        theta = math.radians(rotation_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        points: list[list[float]] = []
        for index in range(point_count):
            angle = math.tau * index / point_count
            lx = math.cos(angle) * radius_x_m
            ly = math.sin(angle) * radius_y_m
            x = float(center[0]) + lx * cos_t - ly * sin_t
            y = float(center[1]) + lx * sin_t + ly * cos_t
            points.append([_round(x, 3), _round(y, 3), _round(altitude_m, 3)])
        points.append(points[0])
        return points

    def _air_point_clear_fast(self, point: Sequence[float]) -> bool:
        geom = Point(float(point[0]), float(point[1]))
        if not self.spatial.bounds_prepared.covers(geom):
            return False
        z_m = float(point[2]) if len(point) >= 3 else 0.0
        if z_m <= 42.0 and self.spatial.building_prepared.covers(geom):
            return False
        return True

    def _air_segment_clear_fast(self, a: Sequence[float], b: Sequence[float]) -> bool:
        line = LineString([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])
        if not self.spatial.bounds_prepared.covers(line):
            return False
        z_min = min(float(a[2]) if len(a) >= 3 else 0.0, float(b[2]) if len(b) >= 3 else 0.0)
        if z_min <= 42.0 and self.spatial.building_prepared.intersects(line):
            return False
        return True

    def _route_clear(self, route: Sequence[Sequence[float]]) -> bool:
        if len(route) < 2:
            return False
        key = tuple((round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2] if len(p) >= 3 else 0.0), 2)) for p in route)
        cached = self._route_clear_cache.get(key)
        if cached is not None:
            return cached
        if any(not self._air_point_clear_fast(point) for point in route):
            self._route_clear_cache[key] = False
            return False
        clear = all(self._air_segment_clear_fast(a, b) for a, b in zip(route, route[1:]))
        self._route_clear_cache[key] = clear
        return clear

    def _route_clear_strict(self, route: Sequence[Sequence[float]]) -> bool:
        if len(route) < 2:
            return False
        key = tuple((round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2] if len(p) >= 3 else 0.0), 2)) for p in route)
        cached = self._route_clear_strict_cache.get(key)
        if cached is not None:
            return cached
        clear = all(self.buildings.air_point_clear(point) for point in route) and all(
            self.buildings.air_segment_clear(a, b) for a, b in zip(route, route[1:])
        )
        self._route_clear_strict_cache[key] = clear
        return clear

    def _find_loop(
        self,
        center: Sequence[float],
        *,
        altitude_m: float,
        min_length_m: float,
        point_count: int = 12,
    ) -> list[list[float]]:
        radii = (
            (26.0, 18.0),
            (32.0, 22.0),
            (40.0, 26.0),
            (52.0, 34.0),
            (68.0, 42.0),
            (90.0, 54.0),
            (120.0, 70.0),
        )
        rotations = tuple(range(0, 180, 5))
        for rx, ry in radii:
            for rotation in rotations:
                route = self._loop_points(center, altitude_m, rx, ry, rotation, point_count=point_count)
                if _xy_path_length(route) >= min_length_m and self._route_clear(route) and self._route_clear_strict(route):
                    return route
        return []

    def _find_lane_racetrack(
        self,
        center: Sequence[float],
        *,
        altitude_m: float,
        min_length_m: float,
    ) -> list[list[float]]:
        nearby_samples = sorted(
            self.spatial.lanes.samples,
            key=lambda sample: (sample.x_m - float(center[0])) ** 2 + (sample.y_m - float(center[1])) ** 2,
        )
        seen_edges: set[str] = set()
        ordered = []
        for sample in nearby_samples:
            if sample.edge_id in seen_edges:
                continue
            seen_edges.add(sample.edge_id)
            ordered.append(sample)
            if len(ordered) >= 36:
                break
        for sample in ordered:
            try:
                min_s, max_s = self.spatial.lanes.edge_s_bounds(sample.edge_id)
            except Exception:
                continue
            for span_m in (52.0, 70.0, 90.0, 115.0, 145.0):
                half_span = 0.5 * span_m
                s0 = max(min_s, min(max_s, sample.s_m - half_span))
                s1 = max(min_s, min(max_s, sample.s_m + half_span))
                if abs(s1 - s0) < 34.0:
                    continue
                start_sample = self.spatial.lanes.resolve_edge_s(sample.edge_id, s0)
                end_sample = self.spatial.lanes.resolve_edge_s(sample.edge_id, s1)
                for lateral_m in (10.0, 14.0, 20.0, 28.0, 36.0):
                    route = [
                        offset_from_lane(start_sample, lateral_m, altitude_m),
                        offset_from_lane(end_sample, lateral_m, altitude_m),
                        offset_from_lane(end_sample, -lateral_m, altitude_m),
                        offset_from_lane(start_sample, -lateral_m, altitude_m),
                        offset_from_lane(start_sample, lateral_m, altitude_m),
                    ]
                    if _xy_path_length(route) >= min_length_m and self._route_clear(route) and self._route_clear_strict(route):
                        return route
        return []

    def _build_pad_patrol_tasks(self) -> list[UavTask]:
        tasks: list[UavTask] = []
        for index, pad in enumerate(self.pads):
            altitude = 18.0 if index % 2 == 0 else 22.0
            route = self._find_lane_racetrack(
                pad.position_enu_m,
                altitude_m=altitude,
                min_length_m=100.0,
            )
            if not route:
                raise RuntimeError(f"Cannot build pad patrol loop for {pad.pad_id}")
            tasks.append(
                UavTask(
                    task_id=f"pad_patrol_{pad.pad_id}",
                    uav_id=f"uav_pad_patrol_{pad.pad_id}",
                    mission_type="pad_patrol",
                    semantic_role="pad_inspect_uav",
                    corridor_family="pad_patrol_local_loop",
                    corridor_id=f"corridor_pad_{pad.pad_id}",
                    altitude_layer_m=altitude,
                    route_waypoints_enu_m=route,
                    speed_mps=5.0 + 0.2 * (index % 4),
                    start_s=0.0,
                    end_s=DURATION_S,
                    looping=True,
                    origin_pad_id=pad.pad_id,
                    target_pad_id=pad.pad_id,
                    target_cell_id=pad.grid_cell_id,
                )
            )
        return tasks

    def _inspect_cells(self) -> list[SpatialGridCell]:
        reserved_pad_cells = {pad.grid_cell_id for pad in self.pads}
        candidates = [
            cell
            for cell in self.grid.cells
            if cell.cell_id not in reserved_pad_cells and cell.main_edge_count >= 5
        ]
        candidates.sort(key=lambda cell: (-cell.main_edge_count, -cell.total_main_edge_length_m, cell.cell_id))
        selected: list[SpatialGridCell] = []
        for cell in candidates:
            if len(selected) >= INTERSECTION_INSPECT_COUNT:
                break
            if any(_dist_xy(cell.representative_enu_m, chosen.representative_enu_m) < 120.0 for chosen in selected):
                continue
            selected.append(cell)
        for cell in candidates:
            if len(selected) >= INTERSECTION_INSPECT_COUNT:
                break
            if cell not in selected:
                selected.append(cell)
        if len(selected) < INTERSECTION_INSPECT_COUNT:
            raise RuntimeError(f"Need {INTERSECTION_INSPECT_COUNT} inspect cells, got {len(selected)}")
        return selected[:INTERSECTION_INSPECT_COUNT]

    def _build_intersection_inspect_tasks(self) -> list[UavTask]:
        tasks: list[UavTask] = []
        reserved_pad_cells = {pad.grid_cell_id for pad in self.pads}
        candidates = [
            cell
            for cell in self.grid.cells
            if cell.cell_id not in reserved_pad_cells and cell.main_edge_count >= 3
        ]
        candidates.sort(key=lambda cell: (-cell.main_edge_count, -cell.total_main_edge_length_m, cell.cell_id))
        selected_cells: list[SpatialGridCell] = []
        for cell in candidates:
            if len(tasks) >= INTERSECTION_INSPECT_COUNT:
                break
            if any(_dist_xy(cell.representative_enu_m, chosen.representative_enu_m) < 90.0 for chosen in selected_cells):
                continue
            index = len(tasks)
            altitude = 28.0 if index % 2 == 0 else 36.0
            route = self._find_loop(
                cell.representative_enu_m,
                altitude_m=altitude,
                min_length_m=130.0,
            )
            if not route:
                route = self._find_lane_racetrack(
                    cell.representative_enu_m,
                    altitude_m=altitude,
                    min_length_m=130.0,
                )
            if not route:
                continue
            selected_cells.append(cell)
            tasks.append(
                UavTask(
                    task_id=f"intersection_inspect_{index:02d}",
                    uav_id=f"uav_intersection_inspect_{index:02d}",
                    mission_type="intersection_inspect",
                    semantic_role="corridor_inspect_uav",
                    corridor_family="road_corridor_inspection_loop",
                    corridor_id=f"corridor_intersection_{cell.cell_id}",
                    altitude_layer_m=altitude,
                    route_waypoints_enu_m=route,
                    speed_mps=5.7 + 0.15 * (index % 5),
                    start_s=0.0,
                    end_s=DURATION_S,
                    looping=True,
                    target_cell_id=cell.cell_id,
                )
            )
        if len(tasks) != INTERSECTION_INSPECT_COUNT:
            raise RuntimeError(f"Cannot build {INTERSECTION_INSPECT_COUNT} intersection inspect loops; built={len(tasks)}")
        return tasks

    def _relay_cells(self) -> list[SpatialGridCell]:
        cells = list(self.grid.cells)
        cells.sort(key=lambda cell: (cell.representative_enu_m[0], cell.representative_enu_m[1]))
        selected: list[SpatialGridCell] = []
        buckets = [
            sorted(cells, key=lambda cell: (cell.representative_enu_m[0], cell.representative_enu_m[1])),
            sorted(cells, key=lambda cell: (-cell.representative_enu_m[0], cell.representative_enu_m[1])),
            sorted(cells, key=lambda cell: (cell.representative_enu_m[0], -cell.representative_enu_m[1])),
            sorted(cells, key=lambda cell: (-cell.representative_enu_m[0], -cell.representative_enu_m[1])),
            sorted(cells, key=lambda cell: (-cell.main_edge_count, cell.cell_id)),
        ]
        for bucket in buckets:
            for cell in bucket:
                if len(selected) >= RELAY_COUNT:
                    return selected
                if any(_dist_xy(cell.representative_enu_m, chosen.representative_enu_m) < 180.0 for chosen in selected):
                    continue
                selected.append(cell)
        for cell in sorted(cells, key=lambda item: item.cell_id):
            if len(selected) >= RELAY_COUNT:
                break
            if cell not in selected:
                selected.append(cell)
        return selected[:RELAY_COUNT]

    def _build_relay_tasks(self) -> list[UavTask]:
        tasks: list[UavTask] = []
        altitudes = (80.0, 110.0, 130.0)
        selected_cells: list[SpatialGridCell] = []
        relay_candidates = self._relay_cells()
        relay_candidates.extend(cell for cell in self.grid.cells if cell not in relay_candidates)
        for cell in relay_candidates:
            if len(tasks) >= RELAY_COUNT:
                break
            if any(_dist_xy(cell.representative_enu_m, chosen.representative_enu_m) < 140.0 for chosen in selected_cells):
                continue
            index = len(tasks)
            altitude = altitudes[index % len(altitudes)]
            route = self._find_loop(
                cell.representative_enu_m,
                altitude_m=altitude,
                min_length_m=260.0,
                point_count=14,
            )
            if not route:
                route = self._find_lane_racetrack(
                    cell.representative_enu_m,
                    altitude_m=altitude,
                    min_length_m=240.0,
                )
            if not route:
                continue
            selected_cells.append(cell)
            tasks.append(
                UavTask(
                    task_id=f"edge_relay_{index:02d}",
                    uav_id=f"uav_edge_relay_{index:02d}",
                    mission_type="edge_compute_relay",
                    semantic_role="communication_relay_uav",
                    corridor_family="high_altitude_sector_relay_loop",
                    corridor_id=f"corridor_relay_{cell.cell_id}",
                    altitude_layer_m=altitude,
                    route_waypoints_enu_m=route,
                    speed_mps=8.0 + 0.35 * (index % 4),
                    start_s=0.0,
                    end_s=DURATION_S,
                    looping=True,
                    target_cell_id=cell.cell_id,
                )
            )
        if len(tasks) != RELAY_COUNT:
            raise RuntimeError(f"Cannot build {RELAY_COUNT} relay loops; built={len(tasks)}")
        return tasks

    def _transit_route_via_grid(
        self,
        origin_air: Sequence[float],
        target_air: Sequence[float],
        *,
        altitude_m: float,
    ) -> list[list[float]]:
        nodes: list[list[float]] = [
            [_round(origin_air[0], 3), _round(origin_air[1], 3), _round(altitude_m, 3)],
            [_round(target_air[0], 3), _round(target_air[1], 3), _round(altitude_m, 3)],
        ]
        seen_xy = {(nodes[0][0], nodes[0][1]), (nodes[1][0], nodes[1][1])}
        for cell in sorted(self.grid.cells, key=lambda item: _dist_xy(item.representative_enu_m, origin_air) + _dist_xy(item.representative_enu_m, target_air)):
            node = [_round(cell.representative_enu_m[0], 3), _round(cell.representative_enu_m[1], 3), _round(altitude_m, 3)]
            key = (node[0], node[1])
            if key in seen_xy or not self._air_point_clear_fast(node) or not self.buildings.air_point_clear(node):
                continue
            seen_xy.add(key)
            nodes.append(node)

        def segment_clear(left: int, right: int) -> bool:
            return self._route_clear([nodes[left], nodes[right]]) and self._route_clear_strict([nodes[left], nodes[right]])

        def reconstruct(parent: dict[int, int | None], node_index: int) -> list[list[float]]:
            indices = []
            current: int | None = node_index
            while current is not None:
                indices.append(current)
                current = parent[current]
            indices.reverse()
            return _dedupe([nodes[index] for index in indices])

        for max_segment_m, neighbor_limit in ((420.0, 14), (680.0, 24), (float("inf"), len(nodes))):
            best: dict[int, float] = {0: 0.0}
            parent: dict[int, int | None] = {0: None}
            frontier: list[tuple[float, float, int]] = [(_dist_xy(nodes[0], nodes[1]), 0.0, 0)]
            while frontier:
                _score, cost, current = heapq.heappop(frontier)
                if cost > best.get(current, float("inf")) + 1e-6:
                    continue
                if current == 1:
                    route = reconstruct(parent, current)
                    if self._route_clear(route) and self._route_clear_strict(route):
                        return route
                    break
                yielded_target = False
                yielded = 0
                for neighbor in sorted(range(len(nodes)), key=lambda index: _dist_xy(nodes[current], nodes[index])):
                    if neighbor == current:
                        continue
                    distance_m = _dist_xy(nodes[current], nodes[neighbor])
                    if neighbor == 1:
                        yielded_target = True
                    elif distance_m > max_segment_m:
                        continue
                    elif yielded >= neighbor_limit:
                        continue
                    if not segment_clear(current, neighbor):
                        continue
                    next_cost = cost + distance_m
                    if next_cost + 1e-6 >= best.get(neighbor, float("inf")):
                        continue
                    best[neighbor] = next_cost
                    parent[neighbor] = current
                    priority = next_cost + _dist_xy(nodes[neighbor], nodes[1])
                    heapq.heappush(frontier, (priority, next_cost, neighbor))
                    yielded += 1
                if not yielded_target and segment_clear(current, 1):
                    next_cost = cost + _dist_xy(nodes[current], nodes[1])
                    if next_cost + 1e-6 < best.get(1, float("inf")):
                        best[1] = next_cost
                        parent[1] = current
                        heapq.heappush(frontier, (next_cost, next_cost, 1))
        return []

    def _transit_route_between(
        self,
        origin: Sequence[float],
        target: Sequence[float],
        *,
        altitude_m: float,
        route_key: str,
    ) -> list[list[float]]:
        origin_air = [float(origin[0]), float(origin[1]), altitude_m]
        target_air = [float(target[0]), float(target[1]), altitude_m]
        cache_key = (
            (round(origin_air[0], 2), round(origin_air[1], 2), round(origin_air[2], 2)),
            (round(target_air[0], 2), round(target_air[1], 2), round(target_air[2], 2)),
        )
        cached = self._transit_route_cache.get(cache_key)
        if cached is not None:
            return cached
        mid_candidates: list[list[float]] = []
        route_units = sorted(self.grid.cells, key=lambda cell: _dist_xy(cell.representative_enu_m, origin) + _dist_xy(cell.representative_enu_m, target))
        for cell in route_units[:14]:
            mid_candidates.append([cell.representative_enu_m[0], cell.representative_enu_m[1], altitude_m])
        mid_candidates.extend(
            [
                [0.5 * (float(origin[0]) + float(target[0])), 0.5 * (float(origin[1]) + float(target[1])), altitude_m],
                [origin_air[0], target_air[1], altitude_m],
                [target_air[0], origin_air[1], altitude_m],
            ]
        )
        raw_routes = [[origin_air, target_air]]
        raw_routes.extend([[origin_air, mid, target_air] for mid in mid_candidates])
        for raw in raw_routes:
            if self._route_clear(raw) and self._route_clear_strict(raw):
                route = _dedupe(raw)
                self._transit_route_cache[cache_key] = route
                return route
        for raw in raw_routes:
            try:
                horizontal = self.buildings.repair_route_at_altitude(raw, altitude_m)
            except Exception:
                continue
            if self._route_clear(horizontal) and self._route_clear_strict(horizontal):
                route = _dedupe(horizontal)
                self._transit_route_cache[cache_key] = route
                return route
        grid_route = self._transit_route_via_grid(origin_air, target_air, altitude_m=altitude_m)
        if grid_route:
            self._transit_route_cache[cache_key] = grid_route
            return grid_route
        raise RuntimeError(f"Cannot build obstacle-clear UAV transit route {route_key} at {altitude_m}m")

    def _full_mission_route(
        self,
        origin: Sequence[float],
        target: Sequence[float],
        *,
        altitude_m: float,
        route_key: str,
        include_landing: bool,
        target_loop: list[list[float]] | None = None,
        transit_altitude_m: float | None = None,
    ) -> list[list[float]]:
        transit_altitude = float(transit_altitude_m if transit_altitude_m is not None else altitude_m)
        transit = self._transit_route_between(origin, target, altitude_m=transit_altitude, route_key=route_key)
        route: list[list[float]] = _vertical_waypoints(
            origin,
            LANDING_PAD_ALTITUDE_M,
            LANDING_APPROACH_ALTITUDE_M,
        )
        route.extend(
            _vertical_waypoints(
                origin,
                LANDING_APPROACH_ALTITUDE_M,
                transit_altitude,
            )[1:]
        )
        route.extend(transit)
        if target_loop:
            loop_start = target_loop[0]
            if abs(float(loop_start[2]) - transit_altitude) > 0.05:
                route.append([float(loop_start[0]), float(loop_start[1]), transit_altitude])
                route.append([float(loop_start[0]), float(loop_start[1]), float(loop_start[2])])
            route.extend(target_loop)
        if include_landing:
            route.extend(
                _vertical_waypoints(
                    target,
                    transit_altitude,
                    LANDING_APPROACH_ALTITUDE_M,
                )
            )
            route.extend(
                _vertical_waypoints(
                    target,
                    LANDING_APPROACH_ALTITUDE_M,
                    LANDING_PAD_ALTITUDE_M,
                )[1:]
            )
        return _dedupe(route)

    def _build_delivery_tasks(self) -> list[UavTask]:
        tasks: list[UavTask] = []
        last_departure_s_by_pad: dict[str, float] = {}
        last_arrival_s_by_pad: dict[str, float] = {}

        def ranked_pad_pairs(phase_index: int) -> list[tuple[Pad, Pad]]:
            pairs: list[tuple[float, float, str, Pad, Pad]] = []
            for origin_pad in self.pads:
                for target_pad in self.pads:
                    if origin_pad.pad_id == target_pad.pad_id:
                        continue
                    phase_score = (
                        origin_pad.phase_origin_weights[phase_index]
                        * target_pad.phase_destination_weights[phase_index]
                    )
                    distance_m = _dist_xy(origin_pad.position_enu_m, target_pad.position_enu_m)
                    pairs.append((phase_score, distance_m, f"{origin_pad.pad_id}->{target_pad.pad_id}", origin_pad, target_pad))
            pairs.sort(key=lambda item: (-item[0], -item[1], item[2]))
            return [(origin_pad, target_pad) for _score, _distance, _key, origin_pad, target_pad in pairs]

        for phase_index, count in enumerate(DELIVERY_ORDERS_BY_PHASE):
            phase_start = phase_index * PHASE_DURATION_S
            deterministic_pairs = ranked_pad_pairs(phase_index)
            for order_index in range(count):
                desired_start_s = phase_start + 1.5 + (PHASE_DURATION_S - 8.0) * (order_index / max(1, count - 1))
                timing_unit = _stable_unit_interval(
                    f"{self.seed}:delivery_start:{phase_index}:{order_index}"
                )
                desired_start_s += (2.0 * timing_unit - 1.0) * 0.7
                preferred_altitude = 50.0 if (order_index + phase_index) % 3 == 0 else 80.0
                altitude_candidates = (
                    preferred_altitude,
                    80.0 if preferred_altitude < 80.0 else 50.0,
                    110.0,
                )
                task_id = f"delivery_p{phase_index}_{order_index:03d}"
                route: list[list[float]] | None = None
                altitude = preferred_altitude
                speed = 14.0 + 0.5 * ((order_index + phase_index) % 5)

                def try_pair(origin_pad: Pad, target_pad: Pad, attempt_label: str) -> tuple[list[list[float]], float] | None:
                    for candidate_altitude in altitude_candidates:
                        try:
                            candidate_route = self._full_mission_route(
                                origin_pad.position_enu_m,
                                target_pad.position_enu_m,
                                altitude_m=candidate_altitude,
                                route_key=f"{task_id}_{attempt_label}_{int(candidate_altitude)}m",
                                include_landing=True,
                            )
                        except RuntimeError:
                            continue
                        if _xy_path_length(candidate_route) >= 120.0:
                            return candidate_route, candidate_altitude
                    return None

                def schedule(candidate_origin: Pad, candidate_target: Pad, candidate_route: list[list[float]]) -> tuple[float, float]:
                    duration_s = _path_length(candidate_route) / speed
                    start_s = max(
                        0.0,
                        desired_start_s,
                        last_departure_s_by_pad.get(candidate_origin.pad_id, -999.0) + 3.0,
                    )
                    arrival_s = start_s + duration_s
                    target_ready_s = last_arrival_s_by_pad.get(candidate_target.pad_id, -999.0) + 3.0
                    if arrival_s < target_ready_s:
                        start_s += target_ready_s - arrival_s
                        arrival_s = start_s + duration_s
                    return start_s, arrival_s

                pair_offset = _stable_int(
                    f"{self.seed}:delivery_pair:{phase_index}:{order_index}",
                    len(deterministic_pairs),
                )
                ordered_pairs = deterministic_pairs[pair_offset:] + deterministic_pairs[:pair_offset]
                origin: Pad | None = None
                target: Pad | None = None
                for pair_index, (origin_candidate, target_candidate) in enumerate(ordered_pairs):
                    result = try_pair(origin_candidate, target_candidate, f"pair{pair_index:03d}")
                    if result is None:
                        continue
                    origin = origin_candidate
                    target = target_candidate
                    route, altitude = result
                    break
                if route is None:
                    raise RuntimeError(
                        f"Unable to build route for {task_id}; candidate_pad_pairs={len(ordered_pairs)}"
                    )
                if origin is None or target is None:
                    raise RuntimeError(f"{task_id}: route resolved without an origin/target pad pair")
                start_s, arrival_s = schedule(origin, target, route)
                if arrival_s > DURATION_S + 1e-6:
                    continue
                last_departure_s_by_pad[origin.pad_id] = start_s
                last_arrival_s_by_pad[target.pad_id] = arrival_s
                tasks.append(
                    UavTask(
                        task_id=task_id,
                        uav_id=f"uav_{task_id}",
                        mission_type="logistics_delivery",
                        semantic_role="delivery_uav",
                        corridor_family="pad_to_pad_logistics_corridor",
                        corridor_id=f"corridor_delivery_{origin.pad_id}_{target.pad_id}_{int(altitude)}m",
                        altitude_layer_m=altitude,
                        route_waypoints_enu_m=route,
                        speed_mps=speed,
                        start_s=_round(start_s, 3),
                        end_s=_round(arrival_s, 3),
                        looping=False,
                        terminal_hold_s=DELIVERY_TERMINAL_HOLD_S,
                        phase_index=phase_index,
                        origin_pad_id=origin.pad_id,
                        target_pad_id=target.pad_id,
                        target_cell_id=target.grid_cell_id,
                    )
                )
        return tasks

    def _nearest_pad(self, point: Sequence[float]) -> Pad:
        return min(self.pads, key=lambda pad: _dist_xy(pad.position_enu_m, point))

    def _build_infrastructure_tasks(self) -> list[UavTask]:
        tasks: list[UavTask] = []
        candidates = [cell for cell in self.grid.nonincident_cells() if cell.main_edge_count >= 4]
        candidates.sort(key=lambda cell: (-cell.total_main_edge_length_m, cell.cell_id))
        for phase_index in range(3):
            phase_start = phase_index * PHASE_DURATION_S
            for local_index in range(INFRA_INSPECTIONS_BY_PHASE):
                cell = candidates[(phase_index * INFRA_INSPECTIONS_BY_PHASE + local_index) % len(candidates)]
                altitude = 28.0 if local_index % 2 == 0 else 36.0
                loop = self._find_loop(
                    cell.representative_enu_m,
                    altitude_m=altitude,
                    min_length_m=120.0,
                )
                if not loop:
                    loop = self._find_lane_racetrack(
                        cell.representative_enu_m,
                        altitude_m=altitude,
                        min_length_m=120.0,
                    )
                if not loop:
                    continue
                origin = self._nearest_pad(cell.representative_enu_m)
                task_id = f"infra_inspect_p{phase_index}_{local_index:02d}"
                route = self._full_mission_route(
                    origin.position_enu_m,
                    loop[0],
                    altitude_m=altitude,
                    route_key=task_id,
                    include_landing=False,
                    target_loop=loop + loop[1:],
                    transit_altitude_m=50.0,
                )
                speed = 8.0
                start_s = phase_start + 8.0 + local_index * 12.0
                duration = min(42.0, _path_length(route) / speed)
                tasks.append(
                    UavTask(
                        task_id=task_id,
                        uav_id=f"uav_{task_id}",
                        mission_type="infrastructure_inspection",
                        semantic_role="temporary_infra_inspect_uav",
                        corridor_family="infrastructure_inspection_corridor",
                        corridor_id=f"corridor_infra_{cell.cell_id}_{int(altitude)}m",
                        altitude_layer_m=altitude,
                        route_waypoints_enu_m=route,
                        speed_mps=speed,
                        start_s=_round(start_s, 3),
                        end_s=_round(min(DURATION_S, start_s + duration), 3),
                        looping=False,
                        phase_index=phase_index,
                        origin_pad_id=origin.pad_id,
                        target_cell_id=cell.cell_id,
                    )
                )
        return tasks

    def _load_incidents(self) -> list[dict[str, Any]]:
        incident_paths = sorted(self.sumo_output_dir.glob("*/sumo_incident_plan.json"))
        if not incident_paths:
            raise RuntimeError(
                f"No episode SUMO incident plans found under {self.sumo_output_dir}"
            )
        incidents_by_id: dict[str, dict[str, Any]] = {}
        for incident_path in incident_paths:
            payload = json.loads(incident_path.read_text(encoding="utf-8-sig"))
            for incident in payload.get("incidents") or []:
                record = dict(incident)
                incident_id = str(record.get("incident_id") or record.get("episode_event_id") or "")
                if not incident_id:
                    raise RuntimeError(f"{incident_path}: incident is missing an authoritative id")
                incidents_by_id.setdefault(incident_id, record)
        return [
            incidents_by_id[key]
            for key in sorted(
                incidents_by_id,
                key=lambda item: (
                    float(incidents_by_id[item].get("start_s") or 0.0),
                    item,
                ),
            )
        ]

    def _build_incident_response_tasks(self) -> list[UavTask]:
        tasks: list[UavTask] = []
        for index, incident in enumerate(self._load_incidents()):
            anchor = dict(incident.get("anchor") or {})
            point = anchor.get("truth_position_enu_m") or anchor.get("projected_truth_xy_m")
            if not isinstance(point, list) or len(point) < 2:
                continue
            center = [float(point[0]), float(point[1]), 0.0]
            altitude = 28.0 if str(incident.get("accident_class") or "").startswith("medical") else 36.0
            loop = self._find_loop(
                center,
                altitude_m=altitude,
                min_length_m=110.0,
            )
            if not loop:
                loop = self._find_lane_racetrack(
                    center,
                    altitude_m=altitude,
                    min_length_m=110.0,
                )
            if not loop:
                continue
            origin = self._nearest_pad(center)
            task_id = f"incident_response_{index:02d}"
            route = self._full_mission_route(
                origin.position_enu_m,
                loop[0],
                altitude_m=altitude,
                route_key=task_id,
                include_landing=False,
                target_loop=loop + loop[1:],
                transit_altitude_m=50.0,
            )
            event_id = str(incident.get("incident_id") or "")
            start_s = max(0.0, float(incident.get("start_s") or 0.0) - 5.0)
            duration = min(55.0, _path_length(route) / 8.5)
            tasks.append(
                UavTask(
                    task_id=task_id,
                    uav_id=f"uav_{task_id}",
                    mission_type="incident_response_inspection",
                    semantic_role="incident_response_uav",
                    corridor_family="priority_incident_response_corridor",
                    corridor_id=f"corridor_incident_{index:02d}_{int(altitude)}m",
                    altitude_layer_m=altitude,
                    route_waypoints_enu_m=route,
                    speed_mps=8.5,
                    start_s=_round(start_s, 3),
                    end_s=_round(min(DURATION_S, start_s + duration), 3),
                    looping=False,
                    phase_index=min(2, int(start_s // PHASE_DURATION_S)),
                    origin_pad_id=origin.pad_id,
                    target_cell_id=self.grid._cell_for_point(center).cell_id,  # noqa: SLF001 - deterministic grid lookup.
                    active_event_ids=(event_id,) if event_id else tuple(),
                )
            )
        return tasks

    def _first_dynamic_conflict(self) -> tuple[str, str, float, float] | None:
        samplers = [RouteSampler(task) for task in self.tasks]
        total_ticks = int(round(DURATION_S * TICK_HZ))
        for tick in range(0, total_ticks + 1, SAMPLE_EVERY_TICKS):
            sim_time_s = tick / float(TICK_HZ)
            by_layer: dict[float, list[tuple[UavTask, list[float]]]] = {}
            for sampler in samplers:
                if not sampler.active_at(sim_time_s):
                    continue
                task = sampler.task
                position, _yaw, _speed = sampler.sample(sim_time_s)
                by_layer.setdefault(round(float(task.altitude_layer_m), 3), []).append((task, position))
            for _layer, items in by_layer.items():
                for left_index, (left_task, left_pos) in enumerate(items):
                    for right_task, right_pos in items[left_index + 1 :]:
                        distance = _dist3(left_pos, right_pos)
                        if distance < MIN_DYNAMIC_SEPARATION_M:
                            return left_task.task_id, right_task.task_id, sim_time_s, distance
        return None

    def _loop_altitude_candidates(self, task: UavTask) -> tuple[float, ...]:
        if task.mission_type == "pad_patrol":
            return (18.0, 22.0, 28.0)
        if task.mission_type == "intersection_inspect":
            return (22.0, 28.0, 36.0, 50.0, 80.0)
        if task.mission_type == "edge_compute_relay":
            return (80.0, 110.0, 130.0)
        return tuple(float(value) for value in ALLOWED_ALTITUDE_LAYERS_M)

    def _route_at_altitude(self, route: Sequence[Sequence[float]], altitude_m: float) -> list[list[float]]:
        return [[_round(point[0], 3), _round(point[1], 3), _round(altitude_m, 3)] for point in route]

    def _try_move_loop_altitude(self, victim: UavTask) -> bool:
        for altitude in self._loop_altitude_candidates(victim):
            if abs(altitude - victim.altitude_layer_m) < 0.05:
                continue
            route = self._route_at_altitude(victim.route_waypoints_enu_m, altitude)
            if not (self._route_clear(route) and self._route_clear_strict(route)):
                continue
            self.tasks = [
                replace(task, altitude_layer_m=altitude, route_waypoints_enu_m=route)
                if task.task_id == victim.task_id
                else task
                for task in self.tasks
            ]
            return True
        return False

    def _resolve_dynamic_conflicts(self) -> None:
        movable_types = {
            "logistics_delivery",
            "infrastructure_inspection",
            "incident_response_inspection",
        }
        for _iteration in range(800):
            conflict = self._first_dynamic_conflict()
            if conflict is None:
                return
            left_id, right_id, sim_time_s, _distance = conflict
            tasks_by_id = {task.task_id: task for task in self.tasks}
            left = tasks_by_id[left_id]
            right = tasks_by_id[right_id]
            candidates = [
                task
                for task in (left, right)
                if (not task.looping) and task.mission_type in movable_types
            ]
            if not candidates:
                loop_candidates = [task for task in (left, right) if task.looping]
                if not loop_candidates:
                    raise RuntimeError(f"Unresolvable UAV conflict {left_id}/{right_id} at {sim_time_s:.1f}s")
                loop_candidates.sort(key=lambda task: (task.mission_type, task.task_id))
                victim = loop_candidates[-1]
                if self._try_move_loop_altitude(victim):
                    continue
                length = max(_path_length(victim.route_waypoints_enu_m), 1.0)
                new_offset = (victim.phase_offset_m + 9.0 + (_iteration % 9) * 1.7) % length
                new_speed = victim.speed_mps + 0.03 * ((_iteration % 5) + 1)
                self.tasks = [
                    replace(
                        task,
                        speed_mps=_round(new_speed, 3),
                        loop_phase_offset_override_m=_round(new_offset, 3),
                    )
                    if task.task_id == victim.task_id
                    else task
                    for task in self.tasks
                ]
            else:
                candidates.sort(key=lambda task: (task.mission_type != "logistics_delivery", task.start_s, task.task_id))
                victim = candidates[-1]
                shift_s = 2.5
                self.tasks = [
                    replace(task, start_s=_round(task.start_s + shift_s, 3), end_s=_round(task.end_s + shift_s, 3))
                    if task.task_id == victim.task_id
                    else task
                    for task in self.tasks
                ]
        conflict = self._first_dynamic_conflict()
        raise RuntimeError(f"Unable to resolve UAV dynamic conflicts after 800 shifts; last_conflict={conflict}")

    def _drop_overrun_delivery_tasks(self) -> None:
        kept: list[UavTask] = []
        dropped: list[str] = []
        for task in self.tasks:
            if task.mission_type == "logistics_delivery" and task.end_s > DURATION_S + 1e-6:
                dropped.append(task.task_id)
                continue
            kept.append(task)
        self.tasks = kept
        self.delivery_overrun_dropped_task_ids = sorted(dropped)

    def _capture_boundary_from_script(self, script: dict[str, Any]) -> dict[str, Any]:
        parameters = dict(script.get("parameters") or {})
        contract = dict(parameters.get("semantic_event_contract") or {})
        return dict(parameters.get("capture_boundary") or contract.get("capture_boundary") or {})

    def _build_event_bindings(self) -> list[EventBinding]:
        bindings: list[EventBinding] = []
        for path in sorted(self.scenarios_root.rglob("event_script.json")):
            try:
                script = json.loads(path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                continue
            scenario_id = str(script.get("scenario_id") or path.parent.name)
            boundary = self._capture_boundary_from_script(script)
            center = boundary.get("center_enu_m")
            if not isinstance(center, list) or len(center) < 2:
                continue
            parameters = dict(script.get("parameters") or {})
            contract = dict(parameters.get("semantic_event_contract") or {})
            intents = tuple(str(item) for item in contract.get("required_intents") or [])
            mission_family = self._mission_family_for_intents(scenario_id, intents)
            phase_index = _stable_int(scenario_id, 3)
            bindings.append(
                EventBinding(
                    scenario_id=scenario_id,
                    center_enu_m=_round_point(center),
                    required_intents=intents,
                    capture_boundary_id=str(boundary.get("boundary_id") or ""),
                    recommended_task_family=mission_family,
                    phase_index=phase_index,
                    source_event_script=_project_path(path),
                )
            )
        return bindings

    def _mission_family_for_intents(self, scenario_id: str, intents: Sequence[str]) -> str:
        text = " ".join([scenario_id, *intents]).lower()
        if "pad_contention" in text or "landing" in text:
            return "pad_contention_or_landing_uav"
        if "collision" in text or "convergence" in text or "near_miss" in text:
            return "conflict_or_avoidance_uav"
        if "geofence" in text or "airspace" in text or "nfz" in text:
            return "airspace_boundary_uav"
        if "digital" in text or "gnss" in text or "c2" in text or "comm" in text:
            return "digital_fault_or_relay_uav"
        if "weather" in text:
            return "weather_adaptation_uav"
        return "event_inspection_uav"

    def _delivery_orders_realized_by_phase(self) -> list[int]:
        counts = [0 for _item in DELIVERY_ORDERS_BY_PHASE]
        for task in self.tasks:
            if task.mission_type != "logistics_delivery":
                continue
            if task.phase_index is None or int(task.phase_index) < 0 or int(task.phase_index) >= len(counts):
                continue
            counts[int(task.phase_index)] += 1
        return counts

    def _task_plan_payload(self) -> dict[str, Any]:
        realized_delivery_orders = self._delivery_orders_realized_by_phase()
        return {
            "schema_name": "donghu_uav_task_plan",
            "schema_version": SCHEMA_VERSION,
            "map_id": MAP_ID,
            "ground_reference_z_m": GROUND_REFERENCE_Z_M,
            "ground_reference_source": "map_ground_reference_v1",
            "duration_s": DURATION_S,
            "tick_hz": TICK_HZ,
            "sample_period_s": SAMPLE_PERIOD_S,
            "minimum_active_uavs_required": MIN_ACTIVE_UAVS,
            "pad_count": len(self.pads),
            "pads": [pad.to_dict() for pad in self.pads],
            "baseline_policy": {
                "pad_patrol_uavs": PAD_PATROL_COUNT,
                "intersection_inspect_uavs": INTERSECTION_INSPECT_COUNT,
                "edge_compute_relay_uavs": RELAY_COUNT,
                "baseline_active_uavs": PAD_PATROL_COUNT + INTERSECTION_INSPECT_COUNT + RELAY_COUNT,
            },
            "delivery_orders_by_phase": list(DELIVERY_ORDERS_BY_PHASE),
            "delivery_orders_requested_by_phase": list(DELIVERY_ORDERS_BY_PHASE),
            "delivery_orders_realized_by_phase": realized_delivery_orders,
            "delivery_orders_skipped_by_phase": [
                requested - realized
                for requested, realized in zip(DELIVERY_ORDERS_BY_PHASE, realized_delivery_orders)
            ],
            "delivery_terminal_hold_s": DELIVERY_TERMINAL_HOLD_S,
            "landing_vertical_waypoint_step_m": LANDING_VERTICAL_WAYPOINT_STEP_M,
            "altitude_layers_m": list(ALLOWED_ALTITUDE_LAYERS_M),
            "tasks": [task.to_dict() for task in sorted(self.tasks, key=lambda item: item.task_id)],
            "event_bindings": [binding.to_dict() for binding in self.event_bindings],
        }

    def _manifest_payload(self, *, sample_count: int, frames_path: Path, task_plan_path: Path) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for task in self.tasks:
            by_type[task.mission_type] = by_type.get(task.mission_type, 0) + 1
        realized_delivery_orders = self._delivery_orders_realized_by_phase()
        return {
            "schema_name": "donghu_uav_flow_manifest",
            "schema_version": SCHEMA_VERSION,
            "map_id": MAP_ID,
            "generator": "Dataset.tools.uav_global_flow.generate_uav_flow",
            "seed": self.seed,
            "duration_s": DURATION_S,
            "tick_hz": TICK_HZ,
            "step_length_s": STEP_LENGTH_S,
            "sample_period_s": SAMPLE_PERIOD_S,
            "sample_every_ticks": SAMPLE_EVERY_TICKS,
            "sample_count": sample_count,
            "minimum_active_uavs_required": MIN_ACTIVE_UAVS,
            "baseline_active_uavs": PAD_PATROL_COUNT + INTERSECTION_INSPECT_COUNT + RELAY_COUNT,
            "pad_count": len(self.pads),
            "task_count": len(self.tasks),
            "task_count_by_type": dict(sorted(by_type.items())),
            "delivery_orders_by_phase": list(DELIVERY_ORDERS_BY_PHASE),
            "delivery_orders_requested_by_phase": list(DELIVERY_ORDERS_BY_PHASE),
            "delivery_orders_realized_by_phase": realized_delivery_orders,
            "delivery_orders_skipped_by_phase": [
                requested - realized
                for requested, realized in zip(DELIVERY_ORDERS_BY_PHASE, realized_delivery_orders)
            ],
            "delivery_overrun_dropped_task_ids": list(self.delivery_overrun_dropped_task_ids),
            "delivery_terminal_hold_s": DELIVERY_TERMINAL_HOLD_S,
            "landing_vertical_waypoint_step_m": LANDING_VERTICAL_WAYPOINT_STEP_M,
            "altitude_layers_m": list(ALLOWED_ALTITUDE_LAYERS_M),
            "coordinate_frame": "ue_truth_enu_m",
            "avoidance_contract": {
                "static_obstacle_clearance": "BuildingObstacleIndex.air_point_clear/air_segment_clear",
                "dynamic_deconfliction": "independent corridor ids, separated altitude layers, deterministic phase offsets",
                "route_search_policy": "deterministic_candidate_enumeration_without_compatibility_paths",
            },
            "outputs": {
                "manifest": _project_path(self.output_dir / "uav_flow_manifest.json"),
                "task_plan": _project_path(task_plan_path),
                "frames": _project_path(frames_path),
            },
        }

    def _write_frames(self, frames_path: Path) -> int:
        samplers = [RouteSampler(task) for task in sorted(self.tasks, key=lambda item: item.task_id)]
        total_ticks = int(round(DURATION_S * TICK_HZ))
        sample_ticks = list(range(0, total_ticks + 1, SAMPLE_EVERY_TICKS))
        frames_path.parent.mkdir(parents=True, exist_ok=True)
        with frames_path.open("w", encoding="utf-8") as handle:
            for tick in sample_ticks:
                sim_time_s = tick / float(TICK_HZ)
                uavs = []
                for sampler in samplers:
                    if not sampler.active_at(sim_time_s):
                        continue
                    task = sampler.task
                    position, yaw, speed = sampler.sample(sim_time_s)
                    uavs.append(
                        {
                            "uav_id": task.uav_id,
                            "task_id": task.task_id,
                            "mission_type": task.mission_type,
                            "semantic_role": task.semantic_role,
                            "corridor_family": task.corridor_family,
                            "corridor_id": task.corridor_id,
                            "altitude_layer_m": task.altitude_layer_m,
                            "position_enu_m": position,
                            "yaw_deg": yaw,
                            "speed_mps": _round(speed, 3),
                            "origin_pad_id": task.origin_pad_id,
                            "target_pad_id": task.target_pad_id,
                            "target_cell_id": task.target_cell_id,
                            "active_event_ids": list(task.active_event_ids),
                            "source": task.source,
                        }
                    )
                frame = {
                    "schema_name": "donghu_uav_traffic_frame",
                    "schema_version": SCHEMA_VERSION,
                    "map_id": MAP_ID,
                    "tick": tick,
                    "sim_time_s": _round(sim_time_s, 3),
                    "uavs": sorted(uavs, key=lambda item: item["uav_id"]),
                    "active_uav_count": len(uavs),
                    "minimum_active_uavs_required": MIN_ACTIVE_UAVS,
                }
                handle.write(json.dumps(frame, ensure_ascii=False, sort_keys=True) + "\n")
        return len(sample_ticks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Donghu global UAV flow samples at 0.5s cadence.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios-root", type=Path, default=DEFAULT_SCENARIOS_ROOT)
    parser.add_argument("--sumo-output-dir", type=Path, default=DEFAULT_SUMO_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not DEFAULT_SUMO_NET_XML.exists():
        raise SystemExit(f"SUMO net.xml not found: {DEFAULT_SUMO_NET_XML}")
    builder = DonghuUavFlowBuilder(
        output_dir=args.output_dir,
        scenarios_root=args.scenarios_root,
        sumo_output_dir=args.sumo_output_dir,
        seed=args.seed,
    )
    manifest = builder.build()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
