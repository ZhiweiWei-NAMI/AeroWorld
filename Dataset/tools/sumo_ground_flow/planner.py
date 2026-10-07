"""SUMO `.net.xml` topology adapter for deterministic ground-flow routes.

SUMO lane shapes and FCD samples are not in the UE truth-frame coordinates. This
adapter maps SUMO XY through the net projection and the GeoJSON-to-traffic-bundle
fit before using route points.
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .coordinates import SumoTruthCoordinateMapper


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UE_TRAFFIC_BUNDLE_DIR = (
    ROOT / "Config" / "LowAltitude" / "Maps" / "donghu_road_topo" / "traffic_bundle"
)
DEFAULT_UE_LANE_CENTER_SAMPLES_CSV = (
    DEFAULT_UE_TRAFFIC_BUNDLE_DIR / "lane_center_samples.csv"
)
DEFAULT_UE_LANE_CONNECTIONS_CSV = (
    DEFAULT_UE_TRAFFIC_BUNDLE_DIR / "lane_connections.csv"
)


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


def _dedupe(points: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for point in points:
        rounded = [
            round(float(point[0]), 3),
            round(float(point[1]), 3),
            round(float(point[2]), 3),
        ]
        if result and result[-1] == rounded:
            continue
        result.append(rounded)
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
        lane_center_samples_csv: Path = DEFAULT_UE_LANE_CENTER_SAMPLES_CSV,
        lane_connections_csv: Path = DEFAULT_UE_LANE_CONNECTIONS_CSV,
    ) -> None:
        self.net_xml = Path(net_xml)
        if not self.net_xml.exists():
            raise FileNotFoundError(f"SUMO net.xml not found: {self.net_xml}")
        self.coordinate_mapper = coordinate_mapper or SumoTruthCoordinateMapper.default(self.net_xml)
        self.max_nearest_edges = max_nearest_edges
        self.max_start_snap_m = max_start_snap_m
        self.edges: dict[str, SumoEdge] = {}
        self.adjacency: dict[str, list[str]] = {}
        self.connection_shapes: dict[
            tuple[str, str],
            tuple[tuple[float, float], ...],
        ] = {}
        self.lane_center_samples_path = Path(lane_center_samples_csv).resolve()
        self.lane_connections_path = Path(lane_connections_csv).resolve()
        self.traffic_geometry_authority = ""
        self.lane_geometry_source = ""
        self.roadway_mesh_hash = ""
        self.lane_sample_count = 0
        self.internal_lane_count = 0
        self.lane_connection_row_count = 0
        self.normal_connection_count = 0
        self.multi_stage_normal_connection_count = 0
        self.max_internal_chain_segments = 0
        self.location: dict[str, str] = {}
        self._load()
        self._load_ue_traffic_bundle(
            self.lane_center_samples_path,
            self.lane_connections_path,
        )
        self._edge_ids = tuple(self.edges)
        self._edge_lines = tuple(LineString(self.edges[edge_id].shape_xy) for edge_id in self._edge_ids)
        self._edge_tree = STRtree(self._edge_lines)
        self._allowed_edge_indices = {
            mode: tuple(
                index
                for index, edge_id in enumerate(self._edge_ids)
                if self._edge_allows(self.edges[edge_id], mode)
            )
            for mode in ("vehicle", "pedestrian")
        }

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
                        lane_id = str(lane.attrib.get("id") or "")
                        raw_shape = _shape_points(lane.attrib.get("shape"))
                        shape = self.coordinate_mapper.sumo_shape_to_truth_xy(raw_shape)
                        if len(shape) >= 2:
                            self.edges[edge_id] = SumoEdge(
                                edge_id=edge_id,
                                edge_type=edge_type,
                                lane_id=lane_id,
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

    def _load_ue_traffic_bundle(
        self,
        lane_center_samples_csv: Path,
        lane_connections_csv: Path,
    ) -> None:
        if not lane_center_samples_csv.is_file():
            raise FileNotFoundError(lane_center_samples_csv)
        if not lane_connections_csv.is_file():
            raise FileNotFoundError(lane_connections_csv)
        samples_by_edge: dict[str, list[tuple[float, float, float]]] = {}
        lane_ids_by_edge: dict[str, str] = {}
        authority_values: set[str] = set()
        source_values: set[str] = set()
        mesh_hashes: set[str] = set()
        with lane_center_samples_csv.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                edge_id = str(row.get("edge_id") or "")
                lane_id = str(row.get("lane_id") or "")
                if not edge_id or not lane_id:
                    raise ValueError(
                        f"{lane_center_samples_csv}: lane sample lacks edge_id/lane_id"
                    )
                samples_by_edge.setdefault(edge_id, []).append(
                    (
                        float(row["s_m"]),
                        float(row["x_m"]),
                        float(row["y_m"]),
                    )
                )
                existing_lane_id = lane_ids_by_edge.setdefault(edge_id, lane_id)
                if existing_lane_id != lane_id:
                    raise ValueError(
                        f"{lane_center_samples_csv}: {edge_id} maps to multiple UE lanes"
                    )
                authority_values.add(str(row.get("geometry_authority") or ""))
                source_values.add(str(row.get("lane_geometry_source") or ""))
                mesh_hashes.add(str(row.get("roadway_mesh_hash") or ""))
        if authority_values != {"road_geojson_topology_plus_ue_roadway_mesh"}:
            raise ValueError(
                f"{lane_center_samples_csv}: unexpected geometry authorities "
                f"{sorted(authority_values)}"
            )
        if source_values != {"ue_roadway_mesh"}:
            raise ValueError(
                f"{lane_center_samples_csv}: lane geometry is not exclusively UE roadway mesh"
            )
        if len(mesh_hashes) != 1 or "" in mesh_hashes:
            raise ValueError(
                f"{lane_center_samples_csv}: roadway mesh hash is missing or inconsistent"
            )
        missing_normal_edges = sorted(set(self.edges) - set(samples_by_edge))
        if missing_normal_edges:
            raise ValueError(
                f"{lane_center_samples_csv}: missing UE geometry for normal edges "
                f"{missing_normal_edges[:8]}"
            )
        for edge_id, edge in list(self.edges.items()):
            samples = sorted(samples_by_edge[edge_id])
            shape = tuple((x_m, y_m) for _s_m, x_m, y_m in samples)
            if len(shape) < 2:
                raise ValueError(
                    f"{lane_center_samples_csv}: {edge_id} has fewer than two UE samples"
                )
            self.edges[edge_id] = replace(
                edge,
                lane_id=lane_ids_by_edge[edge_id],
                shape_xy=shape,
            )
        internal_shapes = {
            lane_ids_by_edge[edge_id]: tuple(
                (x_m, y_m)
                for _s_m, x_m, y_m in sorted(samples)
            )
            for edge_id, samples in samples_by_edge.items()
            if edge_id.startswith(":")
        }
        internal_edge_by_lane_id = {
            lane_ids_by_edge[edge_id]: edge_id
            for edge_id in samples_by_edge
            if edge_id.startswith(":")
        }
        connection_rows: list[dict[str, str]] = []
        with lane_connections_csv.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            connection_rows = [
                {key: str(value or "") for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        rows_by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
        internal_chain_lengths: list[int] = []
        for row in connection_rows:
            rows_by_pair.setdefault(
                (row["from_edge"], row["to_edge"]),
                [],
            ).append(row)
        for row in connection_rows:
            source_edge_id = row["from_edge"]
            target_edge_id = row["to_edge"]
            if source_edge_id not in self.edges or target_edge_id not in self.edges:
                continue
            identity = (source_edge_id, target_edge_id)
            if identity in self.connection_shapes:
                raise ValueError(
                    f"{lane_connections_csv}: duplicate normal-edge connection {identity}"
                )
            connector_points: list[tuple[float, float]] = []
            via_lane_id = row["via_lane"]
            current_internal_edge = internal_edge_by_lane_id.get(via_lane_id)
            visited_internal_edges: set[str] = set()
            while current_internal_edge is not None:
                if current_internal_edge in visited_internal_edges:
                    raise ValueError(
                        f"{lane_connections_csv}: internal connector cycle for {identity}"
                    )
                visited_internal_edges.add(current_internal_edge)
                internal_lane_id = lane_ids_by_edge[current_internal_edge]
                internal_shape = internal_shapes[internal_lane_id]
                if connector_points:
                    join_gap_m = math.dist(connector_points[-1], internal_shape[0])
                    if join_gap_m > 0.01:
                        raise ValueError(
                            f"{lane_connections_csv}: {identity} UE internal chain has a "
                            f"{join_gap_m:.3f}m discontinuity"
                        )
                    if join_gap_m <= 0.0021:
                        connector_points.extend(internal_shape[1:])
                    else:
                        connector_points.extend(internal_shape)
                else:
                    connector_points.extend(internal_shape)
                continuation = rows_by_pair.get(
                    (current_internal_edge, target_edge_id),
                    [],
                )
                if len(continuation) != 1:
                    raise ValueError(
                        f"{lane_connections_csv}: {identity} internal edge "
                        f"{current_internal_edge} has {len(continuation)} continuations"
                    )
                next_via_lane_id = continuation[0]["via_lane"]
                current_internal_edge = internal_edge_by_lane_id.get(
                    next_via_lane_id
                )
            if not connector_points:
                raise ValueError(
                    f"{lane_connections_csv}: {identity} lacks UE connector geometry"
                )
            source_join_gap_m = math.dist(
                self.edges[source_edge_id].shape_xy[-1],
                connector_points[0],
            )
            target_join_gap_m = math.dist(
                connector_points[-1],
                self.edges[target_edge_id].shape_xy[0],
            )
            if source_join_gap_m > 0.01 or target_join_gap_m > 0.01:
                raise ValueError(
                    f"{lane_connections_csv}: {identity} UE connector endpoint gaps are "
                    f"{source_join_gap_m:.3f}m/{target_join_gap_m:.3f}m"
                )
            self.connection_shapes[identity] = tuple(connector_points)
            internal_chain_lengths.append(len(visited_internal_edges))
        self.traffic_geometry_authority = next(iter(authority_values))
        self.lane_geometry_source = next(iter(source_values))
        self.roadway_mesh_hash = next(iter(mesh_hashes))
        self.lane_sample_count = sum(len(samples) for samples in samples_by_edge.values())
        self.internal_lane_count = len(internal_shapes)
        self.lane_connection_row_count = len(connection_rows)
        self.normal_connection_count = len(self.connection_shapes)
        self.multi_stage_normal_connection_count = sum(
            length > 1 for length in internal_chain_lengths
        )
        self.max_internal_chain_segments = max(internal_chain_lengths, default=0)

    @property
    def traffic_geometry_provenance(self) -> dict[str, str | int]:
        return {
            "authority": self.traffic_geometry_authority,
            "lane_geometry_source": self.lane_geometry_source,
            "roadway_mesh_hash": self.roadway_mesh_hash,
            "lane_center_samples": str(self.lane_center_samples_path),
            "lane_connections": str(self.lane_connections_path),
            "normal_edge_count": len(self.edges),
            "lane_sample_count": self.lane_sample_count,
            "internal_lane_count": self.internal_lane_count,
            "lane_connection_row_count": self.lane_connection_row_count,
            "normal_connection_count": self.normal_connection_count,
            "multi_stage_normal_connection_count": self.multi_stage_normal_connection_count,
            "max_internal_chain_segments": self.max_internal_chain_segments,
        }

    def connection_shape_xy(
        self,
        source_edge_id: str,
        target_edge_id: str,
    ) -> tuple[tuple[float, float], ...]:
        connector = self.connection_shapes.get((source_edge_id, target_edge_id))
        if connector is not None:
            return connector
        raise SumoRouteError(
            f"{source_edge_id}->{target_edge_id}: UE traffic bundle lacks connector geometry"
        )

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
        point = Point(px, py)
        candidate_indices: set[int] = set()
        target_count = min(self.max_nearest_edges, len(self._allowed_edge_indices[mode]))
        search_radius_m = max(25.0, float(self.max_start_snap_m))
        while len(candidate_indices) < target_count:
            candidate_indices.update(
                int(index)
                for index in self._edge_tree.query(
                    point,
                    predicate="dwithin",
                    distance=search_radius_m,
                )
                if self._edge_allows(self.edges[self._edge_ids[int(index)]], mode)
            )
            if len(candidate_indices) >= target_count:
                break
            search_radius_m *= 2.0
        ranked: list[tuple[float, str, int, tuple[float, float]]] = []
        for edge_index in sorted(candidate_indices):
            edge = self.edges[self._edge_ids[edge_index]]
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

    def edge_shape_length_m(self, edge_id: str) -> float:
        edge = self.edges[edge_id]
        return sum(
            math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
            for a, b in zip(edge.shape_xy, edge.shape_xy[1:])
        )

    def point_at_edge_s(self, edge_id: str, s_m: float) -> tuple[float, float, float]:
        """Return truth-frame x/y/yaw at directed distance along a SUMO edge."""

        edge = self.edges[edge_id]
        remaining = max(0.0, float(s_m))
        previous = edge.shape_xy[0]
        for current in edge.shape_xy[1:]:
            dx = float(current[0]) - float(previous[0])
            dy = float(current[1]) - float(previous[1])
            length = math.hypot(dx, dy)
            if length > 1e-9 and remaining <= length:
                ratio = remaining / length
                return (
                    float(previous[0]) + dx * ratio,
                    float(previous[1]) + dy * ratio,
                    math.degrees(math.atan2(dy, dx)),
                )
            remaining -= length
            previous = current
        last = edge.shape_xy[-1]
        before_last = edge.shape_xy[-2]
        return (
            float(last[0]),
            float(last[1]),
            math.degrees(
                math.atan2(
                    float(last[1]) - float(before_last[1]),
                    float(last[0]) - float(before_last[0]),
                )
            ),
        )

    def project_point_to_edge(
        self,
        edge_id: str,
        point_enu_m: list[float],
    ) -> tuple[float, float, float, float, float]:
        """Project a truth-frame point to one directed SUMO edge.

        Returns ``(distance_m, s_m, x_m, y_m, yaw_deg)``.
        """

        edge = self.edges[edge_id]
        px = float(point_enu_m[0])
        py = float(point_enu_m[1])
        best: tuple[float, float, float, float, float] | None = None
        cumulative = 0.0
        for a, b in zip(edge.shape_xy, edge.shape_xy[1:]):
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            dx = bx - ax
            dy = by - ay
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (length * length)))
            qx = ax + ratio * dx
            qy = ay + ratio * dy
            candidate = (
                math.hypot(px - qx, py - qy),
                cumulative + ratio * length,
                qx,
                qy,
                math.degrees(math.atan2(dy, dx)),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
            cumulative += length
        if best is None:
            raise SumoRouteError(f"SUMO edge {edge_id} has no non-degenerate segment")
        return best

    def nearest_edge_projection(
        self,
        point_enu_m: list[float],
        *,
        mode: str,
    ) -> tuple[str, float, float, float, float, float]:
        ranked = self._nearest_edges(point_enu_m, mode)
        if not ranked:
            raise SumoRouteError(f"No legal SUMO {mode} edge near {point_enu_m}")
        edge_id = ranked[0][1]
        distance_m, s_m, x_m, y_m, yaw_deg = self.project_point_to_edge(
            edge_id,
            point_enu_m,
        )
        return edge_id, distance_m, s_m, x_m, y_m, yaw_deg

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
            if path_index + 1 < len(path):
                connector = self.connection_shape_xy(
                    edge_id,
                    path[path_index + 1],
                )
                for x, y in connector:
                    points.append([x, y, 0.0])
        return _dedupe(points)
