"""City-scale spatial grid allocation for low-altitude event episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .incident_plan import GROUND_TRAFFIC_EVENT_RULES
from .planner import SumoEdge, SumoGroundFlowPlanner


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMO_OUTPUT_DIR = ROOT / "Dataset" / "sumo_outputs" / "donghu_traffic_270s"
DEFAULT_INCIDENT_PLAN = DEFAULT_SUMO_OUTPUT_DIR / "sumo_incident_plan.json"
DEFAULT_NET_XML = ROOT / "Plugins" / "SumoImporter" / "Maps" / "donghu_road_topo" / "source" / "map.net.xml"
DEFAULT_CELL_SIZE_M = 220.0
DEFAULT_ACCIDENT_EXCLUSION_RADIUS_M = 180.0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _vehicle_edge(edge: SumoEdge) -> bool:
    if "footway" in edge.edge_type or "pedestrian" in edge.edge_type or "steps" in edge.edge_type:
        return False
    if edge.allow:
        return bool(set(edge.allow) & {"passenger", "delivery", "truck", "bus", "taxi", "emergency"})
    return not bool(set(edge.disallow) & {"passenger", "delivery", "truck", "bus", "taxi", "emergency"})


def _distance_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(_distance_xy(a, b) for a, b in zip(points, points[1:]))


def _stable_index(value: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def _cell_id(point: Sequence[float], *, min_x: float, min_y: float, cell_size_m: float) -> str:
    ix = int(math.floor((float(point[0]) - min_x) / cell_size_m))
    iy = int(math.floor((float(point[1]) - min_y) / cell_size_m))
    return f"grid_{ix:03d}_{iy:03d}"


def _midpoint(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return (0.5 * (float(a[0]) + float(b[0])), 0.5 * (float(a[1]) + float(b[1])))


@dataclass(frozen=True)
class SpatialGridCell:
    cell_id: str
    center_enu_m: list[float]
    representative_enu_m: list[float]
    edge_ids: tuple[str, ...]
    main_edge_count: int
    total_main_edge_length_m: float
    max_speed_mps: float
    incident_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["edge_ids"] = list(self.edge_ids)
        payload["total_main_edge_length_m"] = round(float(self.total_main_edge_length_m), 6)
        payload["max_speed_mps"] = round(float(self.max_speed_mps), 6)
        return payload


@dataclass(frozen=True)
class SpatialAssignment:
    scenario_id: str
    event_space_class: str
    target_center_enu_m: list[float]
    grid_cell: SpatialGridCell
    source: str
    traffic_incident_id: str = ""
    accident_class: str = ""
    exclusion_radius_m: float = DEFAULT_ACCIDENT_EXCLUSION_RADIUS_M

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "event_space_class": self.event_space_class,
            "target_center_enu_m": [round(float(v), 6) for v in self.target_center_enu_m],
            "grid_cell": self.grid_cell.to_dict(),
            "source": self.source,
            "traffic_incident_id": self.traffic_incident_id,
            "accident_class": self.accident_class,
            "exclusion_radius_m": round(float(self.exclusion_radius_m), 6),
            "policy": "city_grid_main_road_event_coverage_v1",
        }


class SpatialEventGridPlanner:
    def __init__(
        self,
        *,
        planner: SumoGroundFlowPlanner,
        incident_plan_path: Path | None = DEFAULT_INCIDENT_PLAN,
        cell_size_m: float = DEFAULT_CELL_SIZE_M,
        accident_exclusion_radius_m: float = DEFAULT_ACCIDENT_EXCLUSION_RADIUS_M,
    ) -> None:
        self.planner = planner
        self.incident_plan_path = Path(incident_plan_path) if incident_plan_path else None
        self.cell_size_m = float(cell_size_m)
        self.accident_exclusion_radius_m = float(accident_exclusion_radius_m)
        self.incident_plan = _read_json(self.incident_plan_path) if self.incident_plan_path and self.incident_plan_path.exists() else {"incidents": []}
        self.incidents = [dict(item) for item in self.incident_plan.get("incidents") or []]
        self.traffic_scenario_ids = {str(item.get("episode_scenario_id") or "") for item in self.incidents}
        self.traffic_prefixes = tuple(rule[0] for rule in GROUND_TRAFFIC_EVENT_RULES)
        self.cells = self._build_grid_cells()
        self.incident_points = self._incident_points()
        self.cells_by_id = {cell.cell_id: cell for cell in self.cells}
        self.incident_cell_ids = {
            self._cell_for_point(point).cell_id
            for point in self.incident_points
            if self._cell_for_point(point) is not None
        }
        self.incidents_by_scenario = {
            str(item.get("episode_scenario_id") or ""): dict(item)
            for item in self.incidents
            if str(item.get("episode_scenario_id") or "")
        }

    @classmethod
    def default(
        cls,
        *,
        net_xml: Path = DEFAULT_NET_XML,
        incident_plan_path: Path | None = DEFAULT_INCIDENT_PLAN,
    ) -> "SpatialEventGridPlanner":
        return cls(planner=SumoGroundFlowPlanner(net_xml), incident_plan_path=incident_plan_path)

    def _main_edges(self) -> list[SumoEdge]:
        edges = [
            edge
            for edge in self.planner.edges.values()
            if _vehicle_edge(edge)
            and edge.length_m >= 35.0
            and edge.speed_mps >= 8.0
            and len(edge.shape_xy) >= 2
        ]
        if not edges:
            edges = [
                edge
                for edge in self.planner.edges.values()
                if _vehicle_edge(edge) and edge.length_m >= 20.0 and len(edge.shape_xy) >= 2
            ]
        return edges

    def _build_grid_cells(self) -> list[SpatialGridCell]:
        edges = self._main_edges()
        points = [point for edge in edges for point in edge.shape_xy]
        if not points:
            raise RuntimeError("Cannot build spatial event grid: no vehicle-capable SUMO road geometry")
        min_x = min(float(point[0]) for point in points)
        min_y = min(float(point[1]) for point in points)
        buckets: dict[str, dict[str, Any]] = {}
        for edge in edges:
            shape = list(edge.shape_xy)
            length = _path_length(shape)
            for a, b in zip(shape, shape[1:]):
                mid = _midpoint(a, b)
                cell_id = _cell_id(mid, min_x=min_x, min_y=min_y, cell_size_m=self.cell_size_m)
                bucket = buckets.setdefault(
                    cell_id,
                    {
                        "points": [],
                        "edge_ids": set(),
                        "length": 0.0,
                        "max_speed": 0.0,
                    },
                )
                bucket["points"].append(mid)
                bucket["edge_ids"].add(edge.edge_id)
                bucket["length"] += length / max(1, len(shape) - 1)
                bucket["max_speed"] = max(float(bucket["max_speed"]), edge.speed_mps)
        cells: list[SpatialGridCell] = []
        for cell_id, bucket in buckets.items():
            bucket_points = bucket["points"]
            if not bucket_points:
                continue
            cx = sum(point[0] for point in bucket_points) / len(bucket_points)
            cy = sum(point[1] for point in bucket_points) / len(bucket_points)
            representative = min(bucket_points, key=lambda point: _distance_xy(point, (cx, cy)))
            edge_ids = tuple(sorted(bucket["edge_ids"]))
            cells.append(
                SpatialGridCell(
                    cell_id=cell_id,
                    center_enu_m=[round(cx, 6), round(cy, 6), 0.0],
                    representative_enu_m=[round(representative[0], 6), round(representative[1], 6), 0.0],
                    edge_ids=edge_ids,
                    main_edge_count=len(edge_ids),
                    total_main_edge_length_m=float(bucket["length"]),
                    max_speed_mps=float(bucket["max_speed"]),
                )
            )
        cells.sort(
            key=lambda cell: (
                -cell.main_edge_count,
                -cell.total_main_edge_length_m,
                -cell.max_speed_mps,
                cell.cell_id,
            )
        )
        return cells

    def _incident_points(self) -> list[list[float]]:
        points: list[list[float]] = []
        for incident in self.incidents:
            anchor = dict(incident.get("anchor") or {})
            point = anchor.get("truth_position_enu_m") or anchor.get("projected_truth_xy_m")
            if isinstance(point, list) and len(point) >= 2:
                points.append([float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else 0.0)])
        return points

    def _cell_for_point(self, point: Sequence[float]) -> SpatialGridCell:
        return min(self.cells, key=lambda cell: _distance_xy(cell.representative_enu_m, point))

    def _incident_for_scenario(self, scenario_id: str) -> dict[str, Any] | None:
        matches = [
            item
            for item in self.incidents
            if str(item.get("episode_scenario_id") or "") == str(scenario_id)
        ]
        if not matches:
            return None
        matches.sort(key=lambda item: (float(item.get("start_s") or 0.0), str(item.get("incident_id") or "")))
        return matches[0]

    def is_traffic_incident_scenario(self, scenario_id: str) -> bool:
        if scenario_id in self.traffic_scenario_ids:
            return True
        return any(scenario_id.startswith(prefix) for prefix in self.traffic_prefixes)

    def nonincident_cells(self) -> list[SpatialGridCell]:
        return self._nonincident_cells(reserved_points=self.incident_points, reserved_cell_ids=self.incident_cell_ids)

    def _nonincident_cells(
        self,
        *,
        reserved_points: Sequence[Sequence[float]],
        reserved_cell_ids: set[str],
    ) -> list[SpatialGridCell]:
        result: list[SpatialGridCell] = []
        for cell in self.cells:
            if cell.cell_id in reserved_cell_ids:
                continue
            if any(_distance_xy(cell.representative_enu_m, point) < self.accident_exclusion_radius_m for point in reserved_points):
                continue
            result.append(cell)
        return result or list(self.cells)

    def nontraffic_cells_for_assignments(self, assignments: dict[str, SpatialAssignment]) -> list[SpatialGridCell]:
        reserved_points = [
            assignment.target_center_enu_m
            for assignment in assignments.values()
            if assignment.event_space_class == "traffic_incident_grid"
        ]
        reserved_cell_ids = {
            assignment.grid_cell.cell_id
            for assignment in assignments.values()
            if assignment.event_space_class == "traffic_incident_grid"
        }
        return self._nonincident_cells(reserved_points=reserved_points, reserved_cell_ids=reserved_cell_ids)

    def _traffic_light_assignment(self, scenario_id: str) -> SpatialAssignment | None:
        incident = self._incident_for_scenario(scenario_id)
        if incident is None or str(incident.get("accident_class") or "") != "traffic_light_all_red_fault":
            return None
        traffic_light = dict(incident.get("traffic_light") or {})
        point = traffic_light.get("truth_position_enu_m")
        if not isinstance(point, list) or len(point) < 2:
            anchor = dict(incident.get("anchor") or {})
            point = anchor.get("truth_position_enu_m") or anchor.get("projected_truth_xy_m")
        if not isinstance(point, list) or len(point) < 2:
            return None
        target = [float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else 0.0)]
        return SpatialAssignment(
            scenario_id=scenario_id,
            event_space_class="traffic_incident_grid",
            target_center_enu_m=[round(target[0], 6), round(target[1], 6), 0.0],
            grid_cell=self._cell_for_point(target),
            source="sumo_traffic_light.anchor",
            traffic_incident_id=str(incident.get("incident_id") or ""),
            accident_class=str(incident.get("accident_class") or ""),
            exclusion_radius_m=self.accident_exclusion_radius_m,
        )

    def assign_scenarios(self, scenario_ids: Iterable[str]) -> dict[str, SpatialAssignment]:
        scenario_list = list(scenario_ids)
        assignments: dict[str, SpatialAssignment] = {}
        traffic_scenarios = [scenario_id for scenario_id in scenario_list if self.is_traffic_incident_scenario(scenario_id)]
        traffic_cells = list(self.cells)
        reserved_traffic_cell_ids: set[str] = set()
        traffic_counter = 0
        for scenario_id in traffic_scenarios:
            traffic_light_assignment = self._traffic_light_assignment(scenario_id)
            if traffic_light_assignment is not None:
                assignments[scenario_id] = traffic_light_assignment
                reserved_traffic_cell_ids.add(traffic_light_assignment.grid_cell.cell_id)
                continue
            base_index = (_stable_index(f"traffic:{scenario_id}", len(traffic_cells)) + traffic_counter * 3) % len(traffic_cells)
            cell = traffic_cells[base_index]
            for offset in range(len(traffic_cells)):
                candidate = traffic_cells[(base_index + offset) % len(traffic_cells)]
                if candidate.cell_id not in reserved_traffic_cell_ids:
                    cell = candidate
                    break
            reserved_traffic_cell_ids.add(cell.cell_id)
            traffic_counter += 1
            incident = self._incident_for_scenario(scenario_id) or {}
            assignments[scenario_id] = SpatialAssignment(
                scenario_id=scenario_id,
                event_space_class="traffic_incident_grid",
                target_center_enu_m=list(cell.representative_enu_m),
                grid_cell=cell,
                source="main_road_grid.traffic_incident_cell",
                traffic_incident_id=str(incident.get("incident_id") or ""),
                accident_class=str(incident.get("accident_class") or ""),
                exclusion_radius_m=self.accident_exclusion_radius_m,
            )

        nonincident = self.nontraffic_cells_for_assignments(assignments)
        nontraffic_counter = 0
        cycle_stride = 7
        while nonincident and math.gcd(cycle_stride, len(nonincident)) != 1:
            cycle_stride += 2
        cycle_start = _stable_index("nontraffic:grid-cycle:v1", len(nonincident)) if nonincident else 0
        for scenario_id in scenario_list:
            if scenario_id in assignments:
                continue
            index = (cycle_start + nontraffic_counter * cycle_stride) % len(nonincident)
            cell = nonincident[index]
            nontraffic_counter += 1
            assignments[scenario_id] = SpatialAssignment(
                scenario_id=scenario_id,
                event_space_class="nontraffic_main_road_grid",
                target_center_enu_m=list(cell.representative_enu_m),
                grid_cell=cell,
                source="main_road_grid.nonincident_cell",
                exclusion_radius_m=self.accident_exclusion_radius_m,
            )
        return assignments

    def summary(self) -> dict[str, Any]:
        return {
            "schema_name": "spatial_event_grid",
            "schema_version": "v1",
            "cell_size_m": round(self.cell_size_m, 6),
            "accident_exclusion_radius_m": round(self.accident_exclusion_radius_m, 6),
            "cell_count": len(self.cells),
            "incident_count": len(self.incidents),
            "incident_cell_count": len(self.incident_cell_ids),
            "nonincident_cell_count": len(self.nonincident_cells()),
            "source_incident_plan": str(self.incident_plan_path) if self.incident_plan_path else "",
            "policy": "traffic incidents use SUMO anchors; other EPI scenarios use non-incident main-road grid cells",
        }
