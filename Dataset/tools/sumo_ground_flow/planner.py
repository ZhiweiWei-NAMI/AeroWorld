"""SUMO `.net.xml` topology adapter for deterministic ground-flow routes.

SUMO lane shapes and FCD samples are not in the UE truth-frame coordinates. This
adapter maps SUMO XY through the net projection and the GeoJSON-to-traffic-bundle
fit before using route points.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .coordinates import SumoTruthCoordinateMapper


class SumoRouteError(RuntimeError):
    """Raised when the SUMO network cannot produce a compliant route."""


@dataclass(frozen=True)
class SumoEdge:
    edge_id: str
    edge_type: str
    lane_id: str
    speed_mps: float
    length_m: float
    allow: frozenset[str]
    disallow: frozenset[str]
    shape_xy: tuple[tuple[float, float], ...]


def _parse_tokens(value: str | None) -> frozenset[str]:
    return frozenset(token for token in str(value or "").split() if token)


def _shape_points(value: str | None) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for raw in str(value or "").split():
        parts = raw.split(",")
        if len(parts) < 2:
            continue
        points.append((float(parts[0]), float(parts[1])))
    return tuple(points)


def _path_length_xy(points: list[list[float]]) -> float:
    return sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(points, points[1:]))


def _xy_span(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _dedupe(points: list[list[float]], *, min_gap_m: float = 0.35) -> list[list[float]]:
    result: list[list[float]] = []
    for point in points:
        if result and math.hypot(result[-1][0] - point[0], result[-1][1] - point[1]) < min_gap_m:
            continue
        result.append([round(float(point[0]), 3), round(float(point[1]), 3), round(float(point[2]), 3)])
    return result


def _distance_point_to_segment_xy(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, tuple[float, float], int]:
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay), (ax, ay), 0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy), (qx, qy), 1 if t > 0.5 else 0


class SumoGroundFlowPlanner:
    def __init__(
        self,
        net_xml: Path,
        *,
        coordinate_mapper: SumoTruthCoordinateMapper | None = None,
        max_nearest_edges: int = 36,
        max_start_snap_m: float = 15.0,
    ) -> None:
        self.net_xml = Path(net_xml)
        if not self.net_xml.exists():
            raise FileNotFoundError(f"SUMO net.xml not found: {self.net_xml}")
        self.coordinate_mapper = coordinate_mapper or SumoTruthCoordinateMapper.default(self.net_xml)
        self.max_nearest_edges = max_nearest_edges
        self.max_start_snap_m = max_start_snap_m
        self.edges: dict[str, SumoEdge] = {}
        self.adjacency: dict[str, list[str]] = {}
        self.location: dict[str, str] = {}
        self._load()

    def plan_vehicle_route_enu(
        self,
        start_enu_m: list[float],
        *,
        min_xy_span_m: float,
        min_path_length_m: float | None = None,
        max_edges: int = 18,
        max_start_snap_m: float | None = None,
    ) -> list[list[float]]:
        return self._plan_route(
            start_enu_m,
            mode="vehicle",
            min_xy_span_m=min_xy_span_m,
            min_path_length_m=min_path_length_m,
            max_edges=max_edges,
            max_start_snap_m=max_start_snap_m,
        )

    def plan_pedestrian_route_enu(
        self,
        start_enu_m: list[float],
        *,
        min_xy_span_m: float,
        min_path_length_m: float | None = None,
        max_edges: int = 18,
        max_start_snap_m: float | None = None,
    ) -> list[list[float]]:
        return self._plan_route(
            start_enu_m,
            mode="pedestrian",
            min_xy_span_m=min_xy_span_m,
            min_path_length_m=min_path_length_m,
            max_edges=max_edges,
            max_start_snap_m=max_start_snap_m,
        )

    def _load(self) -> None:
        edge_id: str | None = None
        edge_type = ""
        edge_function = ""
        for _event, elem in ET.iterparse(self.net_xml, events=("end",)):
            if elem.tag == "location":
                self.location = dict(elem.attrib)
            elif elem.tag == "edge":
                edge_id = elem.attrib.get("id")
                edge_type = elem.attrib.get("type", "")
                edge_function = elem.attrib.get("function", "")
                if edge_id and edge_function != "internal":
                    lane = elem.find("lane")
                    if lane is not None:
                        raw_shape = _shape_points(lane.attrib.get("shape"))
                        shape = self.coordinate_mapper.sumo_shape_to_truth_xy(raw_shape)
                        if len(shape) >= 2:
                            self.edges[edge_id] = SumoEdge(
                                edge_id=edge_id,
                                edge_type=edge_type,
                                lane_id=str(lane.attrib.get("id") or ""),
                                speed_mps=float(lane.attrib.get("speed") or 0.0),
                                length_m=float(lane.attrib.get("length") or 0.0),
                                allow=_parse_tokens(lane.attrib.get("allow")),
                                disallow=_parse_tokens(lane.attrib.get("disallow")),
                                shape_xy=shape,
                            )
                elem.clear()
            elif elem.tag == "connection":
                src = elem.attrib.get("from")
                dst = elem.attrib.get("to")
                if src and dst and src != dst:
                    self.adjacency.setdefault(src, []).append(dst)
                elem.clear()

    def _edge_allows(self, edge: SumoEdge, mode: str) -> bool:
        if mode == "pedestrian":
            if edge.allow:
                return "pedestrian" in edge.allow
            return "pedestrian" not in edge.disallow and any(
                token in edge.edge_type
                for token in ("footway", "pedestrian", "path", "steps", "step", "service", "residential")
            )
        vehicle_tokens = {
            "passenger",
            "delivery",
            "truck",
            "bus",
            "taxi",
            "motorcycle",
            "moped",
        }
        if edge.allow:
            return bool(edge.allow & vehicle_tokens)
        return not bool(edge.disallow & vehicle_tokens) and "footway" not in edge.edge_type and "steps" not in edge.edge_type

    def _nearest_edges(self, start: list[float], mode: str) -> list[tuple[float, str, int, tuple[float, float]]]:
        px = float(start[0])
        py = float(start[1])
        ranked: list[tuple[float, str, int, tuple[float, float]]] = []
        for edge in self.edges.values():
            if not self._edge_allows(edge, mode):
                continue
            best_distance = float("inf")
            best_index = 0
            best_projected = edge.shape_xy[0]
            shape = edge.shape_xy
            for index, (a, b) in enumerate(zip(shape, shape[1:])):
                distance, _projected, offset = _distance_point_to_segment_xy(px, py, a[0], a[1], b[0], b[1])
                if distance < best_distance:
                    best_distance = distance
                    best_index = min(index + offset, len(shape) - 1)
                    best_projected = _projected
            ranked.append((best_distance, edge.edge_id, best_index, best_projected))
        ranked.sort(key=lambda item: item[0])
        return ranked[: self.max_nearest_edges]

    def _plan_route(
        self,
        start_enu_m: list[float],
        *,
        mode: str,
        min_xy_span_m: float,
        min_path_length_m: float | None,
        max_edges: int,
        max_start_snap_m: float | None,
    ) -> list[list[float]]:
        min_length = float(min_path_length_m if min_path_length_m is not None else min_xy_span_m)
        snap_limit = float(self.max_start_snap_m if max_start_snap_m is None else max_start_snap_m)
        errors: list[str] = []
        for nearest_distance, start_edge_id, start_shape_index, projected_xy in self._nearest_edges(start_enu_m, mode):
            if nearest_distance > snap_limit:
                errors.append(f"nearest {mode} edge {start_edge_id} is {nearest_distance:.1f}m away")
                continue
            candidate = self._search_from_edge(
                start_enu_m,
                start_edge_id,
                start_shape_index,
                projected_xy,
                mode=mode,
                min_xy_span_m=min_xy_span_m,
                min_path_length_m=min_length,
                max_edges=max_edges,
            )
            if candidate:
                return candidate
        raise SumoRouteError(f"No SUMO {mode} route from {start_enu_m}; {errors[:4]}")

    def _search_from_edge(
        self,
        start_enu_m: list[float],
        start_edge_id: str,
        start_shape_index: int,
        projected_xy: tuple[float, float],
        *,
        mode: str,
        min_xy_span_m: float,
        min_path_length_m: float,
        max_edges: int,
    ) -> list[list[float]]:
        queue: deque[tuple[str, list[str]]] = deque([(start_edge_id, [start_edge_id])])
        visited: set[tuple[str, int]] = set()
        best: list[list[float]] = []
        best_span = 0.0
        while queue:
            edge_id, path = queue.popleft()
            key = (edge_id, len(path))
            if key in visited:
                continue
            visited.add(key)
            route = self._points_for_edge_path(start_enu_m, path, start_shape_index, projected_xy)
            span = _xy_span(route)
            length = _path_length_xy(route)
            if span > best_span:
                best = route
                best_span = span
            if span >= min_xy_span_m and length >= min_path_length_m:
                return route[1:]
            if len(path) >= max_edges:
                continue
            for dst in self.adjacency.get(edge_id, []):
                edge = self.edges.get(dst)
                if edge is None or not self._edge_allows(edge, mode):
                    continue
                if dst in path[-3:]:
                    continue
                queue.append((dst, [*path, dst]))
        if best and best_span >= min_xy_span_m:
            return best[1:]
        return []

    def _points_for_edge_path(
        self,
        start_enu_m: list[float],
        path: list[str],
        start_shape_index: int,
        projected_xy: tuple[float, float],
    ) -> list[list[float]]:
        points: list[list[float]] = [[float(start_enu_m[0]), float(start_enu_m[1]), float(start_enu_m[2] if len(start_enu_m) > 2 else 0.0)]]
        points.append([float(projected_xy[0]), float(projected_xy[1]), float(start_enu_m[2] if len(start_enu_m) > 2 else 0.0)])
        for path_index, edge_id in enumerate(path):
            edge = self.edges[edge_id]
            shape = list(edge.shape_xy)
            if path_index == 0:
                shape = shape[max(0, min(start_shape_index, len(shape) - 1)) :]
                if len(shape) < 2:
                    shape = list(edge.shape_xy)
            for x, y in shape:
                points.append([x, y, 0.0])
        return _dedupe(points)
