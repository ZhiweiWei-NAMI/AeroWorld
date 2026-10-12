"""Align road.geojson topology to the UE DCC RoadWay mesh."""

from __future__ import annotations

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from shapely import set_precision, union_all
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep


REQUIRED_MESH_GEOMETRY_AUTHORITY = "ue_roadway_mesh"
FORMAL_GEOMETRY_AUTHORITY = "road_geojson_topology_plus_ue_roadway_mesh"
FORMAL_LANE_GEOMETRY_SOURCE = "ue_roadway_mesh"
FORMAL_UE_LEVEL = "donghu"
ROADWAY_SURFACE_MATERIAL_ROLES = frozenset(("RoadWay",))
ROAD_MARKING_MATERIAL_ROLES = frozenset(("YellowLine",))
ROAD_MATERIAL_ROLES = ROADWAY_SURFACE_MATERIAL_ROLES | ROAD_MARKING_MATERIAL_ROLES
DRIVABLE_MATERIAL_ROLES = ROADWAY_SURFACE_MATERIAL_ROLES
FORBIDDEN_LANE_TOP_SURFACE_MATERIAL_ROLES = frozenset(("SideWalk",))
LANE_SEPARATOR_MATERIAL_ROLES = frozenset(("YellowLine",))
DRIVABLE_COMPONENT_NAMES = frozenset(("Road",))
ALIGNMENT_SAMPLE_INTERVAL_M = 2.0
ROADWAY_GRID_CELL_M = 20.0
LANE_TOP_SURFACE_GRID_CELL_M = 2.0
LANE_TOP_SURFACE_FOOTPRINT_WIDTH_M = 2.0
LANE_TOP_SURFACE_FOOTPRINT_LATERAL_OFFSETS_M = (-1.0, -0.5, 0.0, 0.5, 1.0)
LANE_TOP_SURFACE_FORBIDDEN_STATION_MIN_FRACTION = 0.50
LANE_TOP_SURFACE_REJECT_MIN_STATIONS = 2
LANE_TOP_SURFACE_REJECT_MIN_STATION_RATIO = 0.05
ROADWAY_NUMERIC_EPSILON_M = 0.05
MIN_MARKING_SPLIT_SIDE_WIDTH_M = 1.2
VALID_SEPARATOR_MIN_MAIN_BODY_COVERAGE = 0.60
YELLOW_SEPARATOR_END_BUFFER_M = 8.0
YELLOW_SEPARATOR_LATERAL_TOLERANCE_M = 0.35
YELLOW_SEPARATOR_AMBIGUOUS_COVERAGE_RATIO = 0.85
YELLOW_SEPARATOR_MIN_SECONDARY_COVERAGE = 0.40
EXTERNAL_LANE_SNAP_DISCARD_THRESHOLD_M = 0.25
EXTERNAL_LANE_FRAGMENT_SIMPLIFY_TOLERANCE_M = 1.0
EXTERNAL_LANE_FRAGMENT_HEADING_DELTA_DEG = 110.0
EXTERNAL_LANE_FRAGMENT_RAW_HEADING_DELTA_DEG = 65.0


class RoadwayMeshAlignmentError(RuntimeError):
    """Raised when UE RoadWay mesh alignment cannot be made formal."""


@dataclass(frozen=True)
class RoadwayLaneShape:
    physical_lane_index: int
    shape_xyz: tuple[tuple[float, float, float], ...]
    lane_width_m: float
    corridor_polygon: Any | None = None


@dataclass(frozen=True)
class RoadwayRoadAlignment:
    road_id: int
    visual_centerline_xyz: tuple[tuple[float, float, float], ...]
    lane_shapes: dict[int, RoadwayLaneShape]
    direction_lane_indices: dict[str, tuple[int, ...]]
    mean_roadway_width_m: float
    sample_count: int
    fallback_sample_count: int
    min_roadway_width_m: float
    max_roadway_width_m: float
    yellow_separator_sample_count: int = 0
    lane_layout_policy: str = "equal_width"
    separator_required: bool = False
    separator_resolution_status: str = "not_required"
    yellow_separator_intervals: tuple[tuple[float, float], ...] = ()
    yellow_separator_coverage_main_body: float = 0.0
    yellow_separator_lateral_offset_median: float = 0.0
    yellow_separator_lateral_offset_p95_error: float = 0.0
    yellow_separator_lateral_offset_span_m: float = 0.0
    yellow_separator_source_mesh_ids: tuple[str, ...] = ()

    @property
    def fallback_ratio(self) -> float:
        if self.sample_count <= 0:
            return 1.0
        return float(self.fallback_sample_count) / float(self.sample_count)


@dataclass(frozen=True)
class RoadwayMesh:
    manifest_path: Path
    vertices_csv: Path
    triangles_csv: Path
    mesh_hash: str
    drivable_component_names: tuple[str, ...]
    rejected_drivable_role_counts: dict[str, int]
    roadway_union: Any
    prepared_roadway: Any
    yellowline_union: Any
    roadway_triangles: tuple[tuple[float, float, float, float, float, float, float, float, float], ...]
    z_grid: dict[tuple[int, int], tuple[int, ...]]
    surface_triangles: tuple[tuple[float, float, float, float, float, float, float, float, float], ...]
    surface_triangle_roles: tuple[str, ...]
    surface_grid: dict[tuple[int, int], tuple[int, ...]]
    yellow_separator_source_mesh_ids: tuple[str, ...] = ()
    yellowline_by_source: tuple[tuple[str, Any], ...] = ()

    def covers_xy(self, x_m: float, y_m: float) -> bool:
        point = Point(float(x_m), float(y_m))
        return bool(self.prepared_roadway.covers(point)) or self.roadway_union.distance(point) <= ROADWAY_NUMERIC_EPSILON_M

    def snap_xy(self, x_m: float, y_m: float) -> tuple[float, float, float]:
        point = Point(float(x_m), float(y_m))
        if self.prepared_roadway.covers(point):
            return float(x_m), float(y_m), 0.0
        nearest = nearest_points(point, self.roadway_union)[1]
        return float(nearest.x), float(nearest.y), float(point.distance(nearest))

    def z_at_xy(self, x_m: float, y_m: float, fallback_z_m: float) -> float:
        cell_x, cell_y = _grid_key(x_m, y_m)
        candidate_indices: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidate_indices.extend(self.z_grid.get((cell_x + dx, cell_y + dy), ()))
        for index in candidate_indices:
            z_value = _triangle_z_at_xy(self.roadway_triangles[index], x_m, y_m)
            if z_value is not None:
                return z_value
        return float(fallback_z_m)

    def top_surface_at_xy(self, x_m: float, y_m: float) -> tuple[str, float] | None:
        cell_x, cell_y = _surface_grid_key(x_m, y_m)
        candidate_indices: set[int] = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidate_indices.update(self.surface_grid.get((cell_x + dx, cell_y + dy), ()))
        top_role: str | None = None
        top_z: float | None = None
        for index in candidate_indices:
            z_value = _triangle_z_at_xy(self.surface_triangles[index], x_m, y_m)
            if z_value is None:
                continue
            if top_z is None or z_value > top_z:
                top_z = float(z_value)
                top_role = self.surface_triangle_roles[index]
        if top_role is None or top_z is None:
            return None
        return top_role, top_z

    def top_surface_audit(
        self,
        points_xyz: Sequence[Sequence[float]],
        *,
        forbidden_roles: frozenset[str] = FORBIDDEN_LANE_TOP_SURFACE_MATERIAL_ROLES,
        footprint_width_m: float = LANE_TOP_SURFACE_FOOTPRINT_WIDTH_M,
        max_examples: int = 6,
    ) -> dict[str, Any]:
        sample_count = 0
        footprint_sample_count = 0
        forbidden_footprint_hit_count = 0
        invalid_station_count = 0
        role_counts: dict[str, int] = {}
        invalid_role_counts: dict[str, int] = {}
        examples: list[dict[str, Any]] = []
        points = [point for point in points_xyz if len(point) >= 2]
        lateral_offsets = _lane_top_surface_lateral_offsets(float(footprint_width_m))
        station_min_hits = max(
            1,
            int(math.ceil(float(len(lateral_offsets)) * LANE_TOP_SURFACE_FORBIDDEN_STATION_MIN_FRACTION)),
        )
        for sample_index, point in enumerate(points):
            if len(point) < 2:
                continue
            sample_count += 1
            x_m = float(point[0])
            y_m = float(point[1])
            lane_z_m = float(point[2]) if len(point) >= 3 else 0.0
            normal_xy = _normal_xy_for_points(points, sample_index)
            station_forbidden_hits = 0
            station_roles: dict[str, int] = {}
            station_forbidden_roles: dict[str, int] = {}
            station_examples: list[dict[str, Any]] = []
            for offset_m in lateral_offsets:
                sample_x = x_m + normal_xy[0] * offset_m
                sample_y = y_m + normal_xy[1] * offset_m
                top_surface = self.top_surface_at_xy(sample_x, sample_y)
                if top_surface is None:
                    top_role = "NO_SURFACE"
                    top_z_m = None
                else:
                    top_role, top_z_m = top_surface
                footprint_sample_count += 1
                role_counts[top_role] = role_counts.get(top_role, 0) + 1
                station_roles[top_role] = station_roles.get(top_role, 0) + 1
                if top_role not in forbidden_roles:
                    continue
                forbidden_footprint_hit_count += 1
                station_forbidden_hits += 1
                invalid_role_counts[top_role] = invalid_role_counts.get(top_role, 0) + 1
                station_forbidden_roles[top_role] = station_forbidden_roles.get(top_role, 0) + 1
                if len(station_examples) < 2:
                    station_examples.append(
                        {
                            "lateral_offset_m": round(float(offset_m), 3),
                            "x_m": round(sample_x, 3),
                            "y_m": round(sample_y, 3),
                            "top_material_role": top_role,
                            "top_z_m": round(float(top_z_m), 3) if top_z_m is not None else None,
                            "top_minus_lane_z_m": (
                                round(float(top_z_m) - lane_z_m, 3) if top_z_m is not None else None
                            ),
                        }
                    )
            if station_forbidden_hits < station_min_hits:
                continue
            invalid_station_count += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "sample_index": sample_index,
                        "center_x_m": round(x_m, 3),
                        "center_y_m": round(y_m, 3),
                        "lane_z_m": round(lane_z_m, 3),
                        "forbidden_lateral_hit_count": station_forbidden_hits,
                        "footprint_lateral_sample_count": len(lateral_offsets),
                        "station_top_material_role_counts": dict(sorted(station_roles.items())),
                        "station_forbidden_role_counts": dict(sorted(station_forbidden_roles.items())),
                        "invalid_lateral_examples": station_examples,
                    }
                )
        invalid_station_ratio = (
            float(invalid_station_count) / float(sample_count) if sample_count > 0 else 0.0
        )
        lane_rejected = (
            invalid_station_count >= LANE_TOP_SURFACE_REJECT_MIN_STATIONS
            and invalid_station_ratio >= LANE_TOP_SURFACE_REJECT_MIN_STATION_RATIO
        )
        return {
            "sample_count": sample_count,
            "footprint_width_m": float(footprint_width_m),
            "footprint_lateral_offsets_m": list(lateral_offsets),
            "footprint_sample_count": footprint_sample_count,
            "forbidden_footprint_hit_count": forbidden_footprint_hit_count,
            "forbidden_station_min_fraction": LANE_TOP_SURFACE_FORBIDDEN_STATION_MIN_FRACTION,
            "invalid_top_surface_count": invalid_station_count,
            "invalid_top_surface_station_ratio": round(invalid_station_ratio, 6),
            "lane_top_surface_rejected": bool(lane_rejected),
            "lane_top_surface_reject_min_stations": LANE_TOP_SURFACE_REJECT_MIN_STATIONS,
            "lane_top_surface_reject_min_station_ratio": LANE_TOP_SURFACE_REJECT_MIN_STATION_RATIO,
            "forbidden_top_material_roles": sorted(forbidden_roles),
            "top_material_role_counts": dict(sorted(role_counts.items())),
            "invalid_top_material_role_counts": dict(sorted(invalid_role_counts.items())),
            "invalid_examples": examples,
        }

    def best_cross_section_interval(
        self,
        *,
        origin_xyz: Sequence[float],
        normal_xy: tuple[float, float],
        fallback_width_m: float,
    ) -> tuple[float, float] | None:
        half_span = max(12.0, float(fallback_width_m) * 2.5)
        x0, y0 = float(origin_xyz[0]), float(origin_xyz[1])
        nx, ny = normal_xy
        section = LineString(
            [
                (x0 - nx * half_span, y0 - ny * half_span),
                (x0 + nx * half_span, y0 + ny * half_span),
            ]
        )
        intersection = self.roadway_union.intersection(section)
        intervals = _intervals_from_cross_section(intersection, x0, y0, nx, ny)
        usable = [
            (min_t, max_t)
            for min_t, max_t in intervals
            if (max_t - min_t) >= max(1.5, min(float(fallback_width_m) * 0.5, float(fallback_width_m) - 0.25))
        ]
        if not usable:
            return None
        selected_min, selected_max = min(usable, key=lambda item: abs((item[0] + item[1]) * 0.5))
        selected_width = selected_max - selected_min
        target_width = max(1.5, float(fallback_width_m))
        if selected_width > target_width * 1.5:
            if selected_min <= 0.0 <= selected_max:
                center_t = 0.0
            else:
                center_t = min(max(0.0, selected_min), selected_max)
            half_target = target_width * 0.5
            center_t = min(max(center_t, selected_min + half_target), selected_max - half_target)
            return center_t - half_target, center_t + half_target
        return selected_min, selected_max

    def yellow_separator_interval(
        self,
        *,
        origin_xyz: Sequence[float],
        normal_xy: tuple[float, float],
        fallback_width_m: float,
        full_interval: tuple[float, float],
    ) -> tuple[float, float] | None:
        intervals = self.yellow_separator_intervals(
            origin_xyz=origin_xyz,
            normal_xy=normal_xy,
            fallback_width_m=fallback_width_m,
            full_interval=full_interval,
        )
        return intervals[0] if intervals else None

    def yellow_separator_intervals(
        self,
        *,
        origin_xyz: Sequence[float],
        normal_xy: tuple[float, float],
        fallback_width_m: float,
        full_interval: tuple[float, float],
    ) -> list[tuple[float, float]]:
        if self.yellowline_union.is_empty:
            return []
        half_span = max(12.0, float(fallback_width_m) * 2.5)
        x0, y0 = float(origin_xyz[0]), float(origin_xyz[1])
        nx, ny = normal_xy
        section = LineString(
            [
                (x0 - nx * half_span, y0 - ny * half_span),
                (x0 + nx * half_span, y0 + ny * half_span),
            ]
        )
        intersection = self.yellowline_union.intersection(section)
        raw_intervals = _intervals_from_cross_section(intersection, x0, y0, nx, ny)
        full_min, full_max = full_interval
        intervals: list[tuple[float, float]] = []
        for min_t, max_t in raw_intervals:
            clipped_min = max(float(full_min), float(min_t))
            clipped_max = min(float(full_max), float(max_t))
            if clipped_max - clipped_min >= 0.03:
                intervals.append((clipped_min, clipped_max))
        if not intervals:
            return []
        intervals.sort(key=lambda item: (item[0] + item[1]) * 0.5)
        grouped_intervals: list[tuple[float, float]] = []
        current_group: list[tuple[float, float]] = []
        current_center: float | None = None
        for interval in intervals:
            center = (interval[0] + interval[1]) * 0.5
            if current_group and current_center is not None and abs(center - current_center) > 1.5:
                grouped_intervals.append(
                    (min(item[0] for item in current_group), max(item[1] for item in current_group))
                )
                current_group = []
            current_group.append(interval)
            current_center = (
                sum((item[0] + item[1]) * 0.5 for item in current_group) / float(len(current_group))
            )
        if current_group:
            grouped_intervals.append(
                (min(item[0] for item in current_group), max(item[1] for item in current_group))
            )
        grouped_intervals.sort(key=lambda item: abs((item[0] + item[1]) * 0.5))
        return grouped_intervals

    def yellow_separator_source_ids_for_interval(
        self,
        *,
        origin_xyz: Sequence[float],
        normal_xy: tuple[float, float],
        fallback_width_m: float,
        full_interval: tuple[float, float],
        target_interval: tuple[float, float],
    ) -> tuple[str, ...]:
        if not self.yellowline_by_source:
            return ()
        half_span = max(12.0, float(fallback_width_m) * 2.5)
        x0, y0 = float(origin_xyz[0]), float(origin_xyz[1])
        nx, ny = normal_xy
        section = LineString(
            [
                (x0 - nx * half_span, y0 - ny * half_span),
                (x0 + nx * half_span, y0 + ny * half_span),
            ]
        )
        full_min, full_max = full_interval
        target_min, target_max = target_interval
        source_ids: list[str] = []
        for source_id, geometry in self.yellowline_by_source:
            intersection = geometry.intersection(section)
            intervals = _intervals_from_cross_section(intersection, x0, y0, nx, ny)
            for min_t, max_t in intervals:
                clipped_min = max(float(full_min), float(min_t))
                clipped_max = min(float(full_max), float(max_t))
                if clipped_max < float(target_min) or clipped_min > float(target_max):
                    continue
                source_ids.append(source_id)
                break
        return tuple(sorted(set(source_ids)))


@dataclass(frozen=True)
class YellowSeparatorResolution:
    required: bool
    valid: bool
    status: str
    split_index: int | None
    side_a_count: int
    side_b_count: int
    sample_count: int
    main_body_sample_count: int
    valid_sample_count: int
    coverage_main_body: float
    lateral_offset_median: float
    lateral_offset_p95_error: float
    lateral_offset_span_m: float
    intervals: tuple[tuple[float, float], ...]
    source_mesh_ids: tuple[str, ...] = ()


def _grid_key(x_m: float, y_m: float) -> tuple[int, int]:
    return int(math.floor(float(x_m) / ROADWAY_GRID_CELL_M)), int(math.floor(float(y_m) / ROADWAY_GRID_CELL_M))


def _surface_grid_key(x_m: float, y_m: float) -> tuple[int, int]:
    return int(math.floor(float(x_m) / LANE_TOP_SURFACE_GRID_CELL_M)), int(
        math.floor(float(y_m) / LANE_TOP_SURFACE_GRID_CELL_M)
    )


def _lane_top_surface_lateral_offsets(footprint_width_m: float) -> tuple[float, ...]:
    half_width = max(0.0, float(footprint_width_m) * 0.5)
    offsets = [
        float(value)
        for value in LANE_TOP_SURFACE_FOOTPRINT_LATERAL_OFFSETS_M
        if abs(float(value)) <= half_width + 1e-6
    ]
    if 0.0 not in offsets:
        offsets.append(0.0)
    return tuple(sorted(set(round(value, 6) for value in offsets)))


def _normal_xy_for_points(points_xyz: Sequence[Sequence[float]], sample_index: int) -> tuple[float, float]:
    if not points_xyz:
        return 0.0, 1.0
    index = min(max(int(sample_index), 0), len(points_xyz) - 1)
    if len(points_xyz) == 1:
        return 0.0, 1.0
    if index <= 0:
        prev_point = points_xyz[0]
        next_point = points_xyz[1]
    elif index >= len(points_xyz) - 1:
        prev_point = points_xyz[-2]
        next_point = points_xyz[-1]
    else:
        prev_point = points_xyz[index - 1]
        next_point = points_xyz[index + 1]
    dx = float(next_point[0]) - float(prev_point[0])
    dy = float(next_point[1]) - float(prev_point[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return 0.0, 1.0
    return -dy / norm, dx / norm


def _component_name_from_row(row: dict[str, str]) -> str:
    component_name = str(row.get("component_name") or "").strip()
    if component_name:
        return component_name
    component_id = str(row.get("component_id") or "").strip()
    if "/" in component_id:
        return component_id.rsplit("/", 1)[-1].strip()
    return component_id


def _source_road_id_from_edge_id(edge_id: str) -> int | None:
    match = re.match(r"^cg_edge_(\d+)_", str(edge_id))
    if match is None:
        return None
    return int(match.group(1))


def _triangle_z_at_xy(
    triangle: tuple[float, float, float, float, float, float, float, float, float],
    x_m: float,
    y_m: float,
) -> float | None:
    x0, y0, z0, x1, y1, z1, x2, y2, z2 = triangle
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(den) <= 1e-9:
        return None
    a = ((y1 - y2) * (x_m - x2) + (x2 - x1) * (y_m - y2)) / den
    b = ((y2 - y0) * (x_m - x2) + (x0 - x2) * (y_m - y2)) / den
    c = 1.0 - a - b
    if a < -1e-5 or b < -1e-5 or c < -1e-5:
        return None
    return a * z0 + b * z1 + c * z2


def _intervals_from_cross_section(geometry: Any, x0: float, y0: float, nx: float, ny: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    if geometry.is_empty:
        return intervals
    geom_type = geometry.geom_type
    if geom_type == "LineString":
        coords = list(geometry.coords)
        if len(coords) >= 2:
            ts = [((float(x) - x0) * nx + (float(y) - y0) * ny) for x, y, *_rest in coords]
            intervals.append((min(ts), max(ts)))
    elif geom_type == "MultiLineString" or geom_type == "GeometryCollection":
        for part in geometry.geoms:
            intervals.extend(_intervals_from_cross_section(part, x0, y0, nx, ny))
    return intervals


def _path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(
        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        for a, b in zip(points, points[1:])
    )


def _point_at_s(points: Sequence[Sequence[float]], s_m: float) -> tuple[float, float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    remaining = max(0.0, float(s_m))
    last = points[0]
    for current in points[1:]:
        dx = float(current[0]) - float(last[0])
        dy = float(current[1]) - float(last[1])
        dz = float(current[2]) - float(last[2]) if len(current) >= 3 and len(last) >= 3 else 0.0
        seg_len = math.hypot(dx, dy)
        if seg_len > 1e-9 and remaining <= seg_len:
            t = remaining / seg_len
            return (
                float(last[0]) + dx * t,
                float(last[1]) + dy * t,
                float(last[2]) + dz * t if len(last) >= 3 else 0.0,
                math.degrees(math.atan2(dy, dx)),
            )
        remaining -= seg_len
        last = current
    if len(points) >= 2:
        dx = float(points[-1][0]) - float(points[-2][0])
        dy = float(points[-1][1]) - float(points[-2][1])
        yaw_deg = math.degrees(math.atan2(dy, dx))
    else:
        yaw_deg = 0.0
    return float(points[-1][0]), float(points[-1][1]), float(points[-1][2] if len(points[-1]) >= 3 else 0.0), yaw_deg


def _s_values_for_path(points: Sequence[Sequence[float]], interval_m: float) -> list[float]:
    total = _path_length(points)
    if total <= 1e-9:
        return [0.0]
    count = max(1, int(math.floor(total / max(0.5, interval_m))))
    values = [float(index) * interval_m for index in range(count + 1)]
    if values[-1] < total:
        values.append(total)
    return values


def _normal_from_yaw(yaw_deg: float) -> tuple[float, float]:
    radians = math.radians(float(yaw_deg))
    dx = math.cos(radians)
    dy = math.sin(radians)
    # UE road XY uses +Y as the right side for a +X forward vector, so this is
    # the UE right normal of the source-road tangent.
    return -dy, dx


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _yellow_interval_has_usable_sides(
    full_interval: tuple[float, float],
    yellow_interval: tuple[float, float] | None,
    *,
    min_side_width_m: float = MIN_MARKING_SPLIT_SIDE_WIDTH_M,
) -> bool:
    if yellow_interval is None:
        return False
    min_t, max_t = full_interval
    yellow_min, yellow_max = yellow_interval
    side_a_width = max(0.0, float(yellow_min) - float(min_t))
    side_b_width = max(0.0, float(max_t) - float(yellow_max))
    return side_a_width >= min_side_width_m and side_b_width >= min_side_width_m


def _lane_center_offsets(width_m: float, lanes: int) -> list[float]:
    lanes = max(1, int(lanes))
    lane_width_m = max(0.5, float(width_m) / float(lanes))
    start = -0.5 * float(width_m) + 0.5 * lane_width_m
    return [start + float(index) * lane_width_m for index in range(lanes)]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(100.0, float(percentile))) / 100.0 * float(len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    t = rank - float(lower)
    return ordered[lower] * (1.0 - t) + ordered[upper] * t


def _separator_required_for_road(road: Any) -> bool:
    try:
        isolation = int(getattr(road, "isolation", 0) or 0)
    except (TypeError, ValueError):
        isolation = 0
    try:
        direction = int(getattr(road, "direction", 1) or 1)
    except (TypeError, ValueError):
        direction = 1
    return isolation == 2 and int(getattr(road, "lanes")) >= 2 and direction != 2


def _separator_interval_summary(candidates: Sequence[dict[str, Any]], sample_interval_m: float) -> tuple[tuple[float, float], ...]:
    s_values = sorted({round(float(candidate["s_m"]), 6) for candidate in candidates})
    if not s_values:
        return ()
    intervals: list[tuple[float, float]] = []
    start_s = s_values[0]
    last_s = s_values[0]
    max_gap = max(0.5, float(sample_interval_m) * 1.75)
    for s_m in s_values[1:]:
        if s_m - last_s > max_gap:
            intervals.append((round(start_s, 3), round(last_s, 3)))
            start_s = s_m
        last_s = s_m
    intervals.append((round(start_s, 3), round(last_s, 3)))
    return tuple(intervals)


def _resolve_valid_yellow_separator(
    road: Any,
    sections: Sequence[dict[str, Any]],
    *,
    sample_interval_m: float,
    roadway_mesh: RoadwayMesh,
) -> YellowSeparatorResolution:
    lanes = max(1, int(getattr(road, "lanes")))
    required = _separator_required_for_road(road)
    if lanes < 2:
        return YellowSeparatorResolution(
            required=required,
            valid=False,
            status="not_applicable_single_lane",
            split_index=None,
            side_a_count=lanes,
            side_b_count=0,
            sample_count=len(sections),
            main_body_sample_count=0,
            valid_sample_count=0,
            coverage_main_body=0.0,
            lateral_offset_median=0.0,
            lateral_offset_p95_error=0.0,
            lateral_offset_span_m=0.0,
            intervals=(),
        )

    width_m = max(0.5, float(getattr(road, "width_m")))
    lane_width_m = width_m / float(lanes)
    lane_centers = _lane_center_offsets(width_m, lanes)
    total_length_m = _path_length(getattr(road, "centerline_xyz"))
    end_buffer_m = max(YELLOW_SEPARATOR_END_BUFFER_M, width_m)
    if total_length_m > end_buffer_m * 2.0 + max(1.0, sample_interval_m):
        main_start_m = end_buffer_m
        main_end_m = total_length_m - end_buffer_m
    else:
        main_start_m = 0.0
        main_end_m = total_length_m
    main_sections = [
        section
        for section in sections
        if main_start_m - 1e-6 <= float(section["s_m"]) <= main_end_m + 1e-6
    ]
    if not main_sections:
        main_sections = list(sections)
    main_body_sample_count = max(1, len(main_sections))
    main_s_values = {round(float(section["s_m"]), 6) for section in main_sections}

    candidates_by_split: dict[int, list[dict[str, Any]]] = {}
    raw_candidate_count = 0
    for section in sections:
        yellow_intervals = list(section.get("yellow_intervals") or ())
        for yellow_interval in yellow_intervals:
            if not _yellow_interval_has_usable_sides(section["interval"], yellow_interval):
                continue
            raw_candidate_count += 1
            yellow_center = 0.5 * (float(yellow_interval[0]) + float(yellow_interval[1]))
            if yellow_center < -0.5 * width_m - YELLOW_SEPARATOR_LATERAL_TOLERANCE_M:
                continue
            if yellow_center > 0.5 * width_m + YELLOW_SEPARATOR_LATERAL_TOLERANCE_M:
                continue
            split_index = sum(1 for center in lane_centers if center < yellow_center)
            if split_index <= 0 or split_index >= lanes:
                continue
            candidates_by_split.setdefault(split_index, []).append(
                {
                    "s_m": float(section["s_m"]),
                    "offset_m": yellow_center,
                    "yellow_interval": yellow_interval,
                    "origin_xyz": (float(section["x_m"]), float(section["y_m"]), float(section["z_m"])),
                    "normal_xy": section["normal_xy"],
                    "full_interval": section["interval"],
                    "is_main_body": round(float(section["s_m"]), 6) in main_s_values,
                }
            )

    if not candidates_by_split:
        return YellowSeparatorResolution(
            required=required,
            valid=False,
            status="no_yellow_separator_candidates_in_road_corridor"
            if raw_candidate_count <= 0
            else "no_yellow_separator_candidates_between_lane_centers",
            split_index=None,
            side_a_count=lanes,
            side_b_count=0,
            sample_count=len(sections),
            main_body_sample_count=main_body_sample_count,
            valid_sample_count=0,
            coverage_main_body=0.0,
            lateral_offset_median=0.0,
            lateral_offset_p95_error=0.0,
            lateral_offset_span_m=0.0,
            intervals=(),
        )

    ranked: list[dict[str, Any]] = []
    for split_index, candidates in candidates_by_split.items():
        main_s_for_split = {round(float(candidate["s_m"]), 6) for candidate in candidates if candidate["is_main_body"]}
        coverage = float(len(main_s_for_split)) / float(main_body_sample_count)
        offsets = [float(candidate["offset_m"]) for candidate in candidates if candidate["is_main_body"]]
        if not offsets:
            offsets = [float(candidate["offset_m"]) for candidate in candidates]
        median_offset = _median(offsets)
        abs_errors = [abs(value - median_offset) for value in offsets]
        ranked.append(
            {
                "split_index": split_index,
                "coverage": coverage,
                "candidates": candidates,
                "median_offset": median_offset,
                "p95_error": _percentile(abs_errors, 95.0),
                "span": max(offsets) - min(offsets) if offsets else 0.0,
            }
        )
    ranked.sort(key=lambda item: (float(item["coverage"]), -float(item["p95_error"])), reverse=True)
    best = ranked[0]
    if len(ranked) > 1:
        second = ranked[1]
        if float(second["coverage"]) >= YELLOW_SEPARATOR_MIN_SECONDARY_COVERAGE and float(second["coverage"]) >= (
            float(best["coverage"]) * YELLOW_SEPARATOR_AMBIGUOUS_COVERAGE_RATIO
        ):
            source_mesh_ids = sorted(
                {
                    source_id
                    for candidate in best["candidates"]
                    for source_id in roadway_mesh.yellow_separator_source_ids_for_interval(
                        origin_xyz=candidate["origin_xyz"],
                        normal_xy=candidate["normal_xy"],
                        fallback_width_m=width_m,
                        full_interval=candidate["full_interval"],
                        target_interval=candidate["yellow_interval"],
                    )
                }
            )
            return YellowSeparatorResolution(
                required=required,
                valid=False,
                status="ambiguous_multiple_yellow_separator_candidates",
                split_index=None,
                side_a_count=lanes,
                side_b_count=0,
                sample_count=len(sections),
                main_body_sample_count=main_body_sample_count,
                valid_sample_count=len(best["candidates"]),
                coverage_main_body=float(best["coverage"]),
                lateral_offset_median=float(best["median_offset"]),
                lateral_offset_p95_error=float(best["p95_error"]),
                lateral_offset_span_m=float(best["span"]),
                intervals=_separator_interval_summary(best["candidates"], sample_interval_m),
                source_mesh_ids=tuple(source_mesh_ids),
            )

    p95_threshold_m = max(0.35, min(0.50, lane_width_m * 0.20))
    span_threshold_m = max(0.75, lane_width_m * 1.10)
    source_mesh_ids = sorted(
        {
            source_id
            for candidate in best["candidates"]
            for source_id in roadway_mesh.yellow_separator_source_ids_for_interval(
                origin_xyz=candidate["origin_xyz"],
                normal_xy=candidate["normal_xy"],
                fallback_width_m=width_m,
                full_interval=candidate["full_interval"],
                target_interval=candidate["yellow_interval"],
            )
        }
    )
    if float(best["coverage"]) < VALID_SEPARATOR_MIN_MAIN_BODY_COVERAGE:
        status = "insufficient_yellow_separator_main_body_coverage"
    elif float(best["p95_error"]) > p95_threshold_m or float(best["span"]) > span_threshold_m:
        status = "unstable_yellow_separator_lateral_offset"
    else:
        split_index = int(best["split_index"])
        return YellowSeparatorResolution(
            required=required,
            valid=True,
            status="valid_yellow_separator",
            split_index=split_index,
            side_a_count=split_index,
            side_b_count=lanes - split_index,
            sample_count=len(sections),
            main_body_sample_count=main_body_sample_count,
            valid_sample_count=len(best["candidates"]),
            coverage_main_body=float(best["coverage"]),
            lateral_offset_median=float(best["median_offset"]),
            lateral_offset_p95_error=float(best["p95_error"]),
            lateral_offset_span_m=float(best["span"]),
            intervals=_separator_interval_summary(best["candidates"], sample_interval_m),
            source_mesh_ids=tuple(source_mesh_ids),
        )

    return YellowSeparatorResolution(
        required=required,
        valid=False,
        status=status,
        split_index=None,
        side_a_count=lanes,
        side_b_count=0,
        sample_count=len(sections),
        main_body_sample_count=main_body_sample_count,
        valid_sample_count=len(best["candidates"]),
        coverage_main_body=float(best["coverage"]),
        lateral_offset_median=float(best["median_offset"]),
        lateral_offset_p95_error=float(best["p95_error"]),
        lateral_offset_span_m=float(best["span"]),
        intervals=_separator_interval_summary(best["candidates"], sample_interval_m),
        source_mesh_ids=tuple(source_mesh_ids),
    )


def _legacy_direction_lane_indices(road: Any) -> dict[str, tuple[int, ...]]:
    lanes = max(1, int(getattr(road, "lanes")))
    direction = int(getattr(road, "direction"))
    if direction == 4:
        return {"r": tuple(range(lanes))}
    if direction == 2:
        forward_count = (lanes + 1) // 2
        result: dict[str, tuple[int, ...]] = {}
        if forward_count > 0:
            result["f"] = tuple(range(forward_count))
        if lanes - forward_count > 0:
            result["r"] = tuple(range(forward_count, lanes))
        return result
    return {"f": tuple(range(lanes))}


def _yellow_split_direction_lane_indices(
    *,
    side_a_count: int,
    side_b_count: int,
) -> dict[str, tuple[int, ...]]:
    side_a = tuple(range(side_a_count))
    side_b = tuple(range(side_a_count, side_a_count + side_b_count))
    # Lane indices are allocated along the UE right normal of the source-road
    # tangent: side_a is the lower/left-of-source side, side_b is the
    # higher/right-of-source side.  China/right-hand traffic requires the
    # YellowLine to be left of vehicle motion, so side_b follows source order
    # and side_a runs reverse.
    return {"f": side_b, "r": side_a}


def _shape_points(value: str | None) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for raw in str(value or "").split():
        parts = raw.split(",")
        if len(parts) >= 2:
            points.append((float(parts[0]), float(parts[1])))
    return tuple(points)


def _format_shape(points: Sequence[Sequence[float]]) -> str:
    return " ".join(f"{float(point[0]):.3f},{float(point[1]):.3f}" for point in points)


def _heading_delta_deg(a_deg: float, b_deg: float) -> float:
    return abs((float(b_deg) - float(a_deg) + 180.0) % 360.0 - 180.0)


def _max_simplified_heading_delta(points: Sequence[tuple[float, float]]) -> tuple[float, list[tuple[float, float]]]:
    if len(points) < 3:
        return 0.0, list(points)
    try:
        simplified = list(
            LineString(points)
            .simplify(EXTERNAL_LANE_FRAGMENT_SIMPLIFY_TOLERANCE_M, preserve_topology=False)
            .coords
        )
    except Exception:
        simplified = list(points)
    if len(simplified) < 3:
        return 0.0, simplified
    headings: list[float] = []
    for start, end in zip(simplified, simplified[1:]):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if math.hypot(dx, dy) <= 1e-6:
            continue
        headings.append(math.degrees(math.atan2(dy, dx)))
    if len(headings) < 2:
        return 0.0, simplified
    return max(_heading_delta_deg(a, b) for a, b in zip(headings, headings[1:])), simplified


def _max_adjacent_heading_delta(points: Sequence[tuple[float, float]]) -> float:
    headings: list[float] = []
    for start, end in zip(points, points[1:]):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if math.hypot(dx, dy) <= 1e-6:
            continue
        headings.append(math.degrees(math.atan2(dy, dx)))
    if len(headings) < 2:
        return 0.0
    return max(_heading_delta_deg(a, b) for a, b in zip(headings, headings[1:]))


def _densify_xy(points: Sequence[tuple[float, float]], interval_m: float = 1.0) -> list[tuple[float, float]]:
    if len(points) < 2:
        return list(points)
    dense: list[tuple[float, float]] = [points[0]]
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(length / max(0.25, interval_m))))
        for step in range(1, steps + 1):
            t = float(step) / float(steps)
            dense.append((start[0] + dx * t, start[1] + dy * t))
    return dense


def load_roadway_mesh(manifest_path: Path) -> RoadwayMesh:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise RoadwayMeshAlignmentError(f"UE RoadWay mesh manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if str(manifest.get("geometry_authority") or "") != REQUIRED_MESH_GEOMETRY_AUTHORITY:
        raise RoadwayMeshAlignmentError(
            f"RoadWay mesh manifest geometry_authority must be {REQUIRED_MESH_GEOMETRY_AUTHORITY}: {manifest_path}"
        )
    if str(manifest.get("ue_level") or "") != FORMAL_UE_LEVEL:
        raise RoadwayMeshAlignmentError(f"RoadWay mesh manifest ue_level must be {FORMAL_UE_LEVEL}: {manifest_path}")
    mesh_hash = str(manifest.get("mesh_hash") or "").strip()
    if not mesh_hash:
        raise RoadwayMeshAlignmentError(f"RoadWay mesh manifest is missing mesh_hash: {manifest_path}")
    vertices_csv = Path(str(manifest.get("vertices_csv") or ""))
    triangles_csv = Path(str(manifest.get("triangles_csv") or ""))
    if not vertices_csv.exists() or not triangles_csv.exists():
        raise RoadwayMeshAlignmentError(
            f"RoadWay mesh CSVs are missing: vertices={vertices_csv} triangles={triangles_csv}"
        )

    vertices: dict[int, tuple[float, float, float]] = {}
    with vertices_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["vertex_index"])
            vertices[index] = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))

    polygons: list[Polygon] = []
    polygons_by_role: dict[str, list[Polygon]] = {role: [] for role in ROAD_MATERIAL_ROLES}
    roadway_triangles: list[tuple[float, float, float, float, float, float, float, float, float]] = []
    z_grid_lists: dict[tuple[int, int], list[int]] = {}
    surface_triangles: list[tuple[float, float, float, float, float, float, float, float, float]] = []
    surface_triangle_roles: list[str] = []
    surface_grid_lists: dict[tuple[int, int], list[int]] = {}
    rejected_drivable_role_counts: dict[str, int] = {}
    separator_source_mesh_ids: set[str] = set()
    separator_polygons_by_source: dict[str, list[Polygon]] = {}
    with triangles_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            material_role = str(row.get("material_role") or "")
            component_name = _component_name_from_row(row)
            v0 = vertices.get(int(row["v0"]))
            v1 = vertices.get(int(row["v1"]))
            v2 = vertices.get(int(row["v2"]))
            if v0 is None or v1 is None or v2 is None:
                continue
            triangle_xyz = (v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2])
            area2 = abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
            if area2 <= 1e-10:
                continue
            if component_name in DRIVABLE_COMPONENT_NAMES:
                surface_index = len(surface_triangles)
                surface_triangles.append(triangle_xyz)
                surface_triangle_roles.append(material_role or "UNKNOWN")
                minx = min(v0[0], v1[0], v2[0])
                miny = min(v0[1], v1[1], v2[1])
                maxx = max(v0[0], v1[0], v2[0])
                maxy = max(v0[1], v1[1], v2[1])
                min_cell = _surface_grid_key(minx, miny)
                max_cell = _surface_grid_key(maxx, maxy)
                for cell_x in range(min_cell[0], max_cell[0] + 1):
                    for cell_y in range(min_cell[1], max_cell[1] + 1):
                        surface_grid_lists.setdefault((cell_x, cell_y), []).append(surface_index)
            if material_role not in ROAD_MATERIAL_ROLES:
                continue
            if component_name not in DRIVABLE_COMPONENT_NAMES:
                key = f"{component_name}:{material_role}:{row.get('material_name') or ''}"
                rejected_drivable_role_counts[key] = rejected_drivable_role_counts.get(key, 0) + 1
                continue
            polygon = Polygon([(v0[0], v0[1]), (v1[0], v1[1]), (v2[0], v2[1])])
            polygon = set_precision(polygon, 0.001)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= 1e-8:
                continue
            polygons_by_role.setdefault(material_role, []).append(polygon)
            if material_role in LANE_SEPARATOR_MATERIAL_ROLES:
                component_id = str(row.get("component_id") or component_name or "").strip()
                material_name = str(row.get("material_name") or "").strip()
                source_id = ":".join(part for part in (component_id, material_role, material_name) if part)
                separator_source_mesh_ids.add(source_id)
                separator_polygons_by_source.setdefault(source_id, []).append(polygon)
            if material_role in DRIVABLE_MATERIAL_ROLES:
                triangle_index = len(roadway_triangles)
                roadway_triangles.append(triangle_xyz)
                polygons.append(polygon)
                minx, miny, maxx, maxy = polygon.bounds
                min_cell = _grid_key(minx, miny)
                max_cell = _grid_key(maxx, maxy)
                for cell_x in range(min_cell[0], max_cell[0] + 1):
                    for cell_y in range(min_cell[1], max_cell[1] + 1):
                        z_grid_lists.setdefault((cell_x, cell_y), []).append(triangle_index)

    if not polygons:
        raise RoadwayMeshAlignmentError(f"RoadWay mesh contains zero usable RoadWay triangles: {manifest_path}")
    try:
        roadway_union = union_all(polygons, grid_size=0.001)
    except Exception:
        roadway_union = unary_union([polygon.buffer(0) for polygon in polygons])
    if roadway_union.is_empty:
        raise RoadwayMeshAlignmentError(f"RoadWay mesh union is empty: {manifest_path}")
    separator_polygons: list[Polygon] = []
    for role in LANE_SEPARATOR_MATERIAL_ROLES:
        separator_polygons.extend(polygons_by_role.get(role, []))
    if separator_polygons:
        try:
            yellowline_union = union_all(separator_polygons, grid_size=0.001)
        except Exception:
            yellowline_union = unary_union([polygon.buffer(0) for polygon in separator_polygons])
    else:
        yellowline_union = GeometryCollection()
    yellowline_by_source: list[tuple[str, Any]] = []
    for source_id, source_polygons in sorted(separator_polygons_by_source.items()):
        try:
            source_geometry = union_all(source_polygons, grid_size=0.001)
        except Exception:
            source_geometry = unary_union([polygon.buffer(0) for polygon in source_polygons])
        if not source_geometry.is_empty:
            yellowline_by_source.append((source_id, source_geometry))
    return RoadwayMesh(
        manifest_path=manifest_path,
        vertices_csv=vertices_csv,
        triangles_csv=triangles_csv,
        mesh_hash=mesh_hash,
        drivable_component_names=tuple(sorted(DRIVABLE_COMPONENT_NAMES)),
        rejected_drivable_role_counts=rejected_drivable_role_counts,
        roadway_union=roadway_union,
        prepared_roadway=prep(roadway_union),
        yellowline_union=yellowline_union,
        roadway_triangles=tuple(roadway_triangles),
        z_grid={key: tuple(value) for key, value in z_grid_lists.items()},
        surface_triangles=tuple(surface_triangles),
        surface_triangle_roles=tuple(surface_triangle_roles),
        surface_grid={key: tuple(value) for key, value in surface_grid_lists.items()},
        yellow_separator_source_mesh_ids=tuple(sorted(separator_source_mesh_ids)),
        yellowline_by_source=tuple(yellowline_by_source),
    )


def align_roads_to_roadway_mesh(
    roads: Sequence[Any],
    roadway_mesh: RoadwayMesh,
    *,
    sample_interval_m: float = ALIGNMENT_SAMPLE_INTERVAL_M,
    max_fallback_ratio: float = 0.10,
) -> tuple[dict[int, RoadwayRoadAlignment], dict[str, Any]]:
    alignments: dict[int, RoadwayRoadAlignment] = {}
    road_reports: list[dict[str, Any]] = []
    failing_roads: list[dict[str, Any]] = []
    separator_unresolved_roads: list[dict[str, Any]] = []
    total_samples = 0
    total_fallbacks = 0

    for road in roads:
        lane_points: dict[int, list[tuple[float, float, float]]] = {index: [] for index in range(int(road.lanes))}
        lane_widths: dict[int, list[float]] = {index: [] for index in range(int(road.lanes))}
        lane_left_boundary_points: dict[int, list[tuple[float, float]]] = {index: [] for index in range(int(road.lanes))}
        lane_right_boundary_points: dict[int, list[tuple[float, float]]] = {index: [] for index in range(int(road.lanes))}
        centerline: list[tuple[float, float, float]] = []
        fallback_count = 0
        widths: list[float] = []
        s_values = _s_values_for_path(road.centerline_xyz, sample_interval_m)
        sections: list[dict[str, Any]] = []
        for s_m in s_values:
            x_m, y_m, z_m, yaw_deg = _point_at_s(road.centerline_xyz, s_m)
            normal_xy = _normal_from_yaw(yaw_deg)
            interval = roadway_mesh.best_cross_section_interval(
                origin_xyz=(x_m, y_m, z_m),
                normal_xy=normal_xy,
                fallback_width_m=float(road.width_m),
            )
            if interval is None:
                fallback_count += 1
                interval = (-0.5 * float(road.width_m), 0.5 * float(road.width_m))
            yellow_intervals: list[tuple[float, float]] = []
            if int(road.lanes) >= 2:
                yellow_intervals = roadway_mesh.yellow_separator_intervals(
                    origin_xyz=(x_m, y_m, z_m),
                    normal_xy=normal_xy,
                    fallback_width_m=float(road.width_m),
                    full_interval=interval,
                )
            sections.append(
                {
                    "s_m": s_m,
                    "x_m": x_m,
                    "y_m": y_m,
                    "z_m": z_m,
                    "yaw_deg": yaw_deg,
                    "normal_xy": normal_xy,
                    "interval": interval,
                    "yellow_intervals": tuple(yellow_intervals),
                }
            )

        separator_resolution = _resolve_valid_yellow_separator(
            road,
            sections,
            sample_interval_m=sample_interval_m,
            roadway_mesh=roadway_mesh,
        )
        yellow_separator_sample_count = separator_resolution.valid_sample_count
        use_yellow_split = separator_resolution.valid
        if use_yellow_split:
            side_a_count = separator_resolution.side_a_count
            side_b_count = separator_resolution.side_b_count
            direction_lane_indices = _yellow_split_direction_lane_indices(
                side_a_count=side_a_count,
                side_b_count=side_b_count,
            )
            lane_layout_policy = "valid_yellow_separator_canonical_lane_split"
        else:
            side_a_count = int(road.lanes)
            side_b_count = 0
            direction_lane_indices = _legacy_direction_lane_indices(road)
            lane_layout_policy = "equal_width"

        for section in sections:
            x_m = float(section["x_m"])
            y_m = float(section["y_m"])
            z_m = float(section["z_m"])
            normal_xy = section["normal_xy"]
            interval = section["interval"]
            min_t, max_t = interval
            roadway_width_m = max(0.5, max_t - min_t)
            mid_t = 0.0 if use_yellow_split else 0.5 * (min_t + max_t)
            center_x = x_m + normal_xy[0] * mid_t
            center_y = y_m + normal_xy[1] * mid_t
            center_z = roadway_mesh.z_at_xy(center_x, center_y, z_m)
            centerline.append((center_x, center_y, center_z))
            widths.append(float(road.width_m) if use_yellow_split else roadway_width_m)

            lane_intervals: list[tuple[int, float, float]] = []
            if use_yellow_split:
                lane_width_m = float(road.width_m) / float(road.lanes)
                for lane_index in range(int(road.lanes)):
                    lane_min = -0.5 * float(road.width_m) + float(lane_index) * lane_width_m
                    lane_max = lane_min + lane_width_m
                    lane_intervals.append((lane_index, lane_min, lane_max))

            if not lane_intervals:
                lane_width_m = roadway_width_m / float(road.lanes)
                for lane_index in range(int(road.lanes)):
                    lane_min = min_t + float(lane_index) * lane_width_m
                    lane_max = lane_min + lane_width_m
                    lane_intervals.append((lane_index, lane_min, lane_max))

            for lane_index, lane_min, lane_max in lane_intervals:
                lane_width_m = max(0.5, float(lane_max) - float(lane_min))
                lateral_t = 0.5 * (float(lane_min) + float(lane_max))
                lane_x = x_m + normal_xy[0] * lateral_t
                lane_y = y_m + normal_xy[1] * lateral_t
                snapped_x, snapped_y, _snap_distance = roadway_mesh.snap_xy(lane_x, lane_y)
                lane_z = roadway_mesh.z_at_xy(snapped_x, snapped_y, z_m)
                lane_points[lane_index].append((snapped_x, snapped_y, lane_z))
                lane_widths[lane_index].append(lane_width_m)
                lane_left_boundary_points[lane_index].append(
                    (
                        x_m + normal_xy[0] * float(lane_min),
                        y_m + normal_xy[1] * float(lane_min),
                    )
                )
                lane_right_boundary_points[lane_index].append(
                    (
                        x_m + normal_xy[0] * float(lane_max),
                        y_m + normal_xy[1] * float(lane_max),
                    )
                )

        sample_count = len(s_values)
        total_samples += sample_count
        total_fallbacks += fallback_count
        mean_width = sum(widths) / float(len(widths)) if widths else float(road.width_m)
        lane_shapes: dict[int, RoadwayLaneShape] = {}
        for lane_index, points in lane_points.items():
            if len(points) < 2:
                continue
            left_points = lane_left_boundary_points.get(lane_index, [])
            right_points = lane_right_boundary_points.get(lane_index, [])
            corridor_polygon = None
            if len(left_points) >= 2 and len(right_points) >= 2:
                try:
                    polygon = Polygon(left_points + list(reversed(right_points)))
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    if not polygon.is_empty and polygon.area > 1e-6:
                        corridor_polygon = polygon
                except Exception:
                    corridor_polygon = None
            lane_shapes[lane_index] = RoadwayLaneShape(
                physical_lane_index=lane_index,
                shape_xyz=tuple(points),
                lane_width_m=(
                    sum(lane_widths.get(lane_index, [])) / float(len(lane_widths.get(lane_index, [])))
                    if lane_widths.get(lane_index)
                    else float(road.width_m) / float(road.lanes)
                ),
                corridor_polygon=corridor_polygon,
            )
        alignment = RoadwayRoadAlignment(
            road_id=int(road.road_id),
            visual_centerline_xyz=tuple(centerline),
            lane_shapes=lane_shapes,
            direction_lane_indices=direction_lane_indices,
            mean_roadway_width_m=mean_width,
            sample_count=sample_count,
            fallback_sample_count=fallback_count,
            min_roadway_width_m=min(widths) if widths else 0.0,
            max_roadway_width_m=max(widths) if widths else 0.0,
            yellow_separator_sample_count=yellow_separator_sample_count,
            lane_layout_policy=lane_layout_policy,
            separator_required=separator_resolution.required,
            separator_resolution_status=separator_resolution.status,
            yellow_separator_intervals=separator_resolution.intervals,
            yellow_separator_coverage_main_body=separator_resolution.coverage_main_body,
            yellow_separator_lateral_offset_median=separator_resolution.lateral_offset_median,
            yellow_separator_lateral_offset_p95_error=separator_resolution.lateral_offset_p95_error,
            yellow_separator_lateral_offset_span_m=separator_resolution.lateral_offset_span_m,
            yellow_separator_source_mesh_ids=separator_resolution.source_mesh_ids,
        )
        alignments[int(road.road_id)] = alignment
        report_row = {
            "road_id": int(road.road_id),
            "direction": int(getattr(road, "direction", 1)),
            "isolation": int(getattr(road, "isolation", 0) or 0),
            "samples": sample_count,
            "fallback_samples": fallback_count,
            "fallback_ratio": round(alignment.fallback_ratio, 6),
            "expected_lane_count": int(road.lanes),
            "mean_roadway_width_m": round(alignment.mean_roadway_width_m, 6),
            "min_roadway_width_m": round(alignment.min_roadway_width_m, 6),
            "max_roadway_width_m": round(alignment.max_roadway_width_m, 6),
            "lane_shape_count": len(lane_shapes),
            "yellow_separator_samples": yellow_separator_sample_count,
            "yellow_separator_ratio": round(float(yellow_separator_sample_count) / float(sample_count), 6) if sample_count else 0.0,
            "separator_required": separator_resolution.required,
            "separator_resolution_status": separator_resolution.status,
            "yellow_separator_intervals": [
                [round(float(start_s), 3), round(float(end_s), 3)]
                for start_s, end_s in separator_resolution.intervals
            ],
            "yellow_separator_coverage_main_body": round(separator_resolution.coverage_main_body, 6),
            "yellow_separator_lateral_offset_median": round(separator_resolution.lateral_offset_median, 6),
            "yellow_separator_lateral_offset_p95_error": round(separator_resolution.lateral_offset_p95_error, 6),
            "yellow_separator_lateral_offset_span_m": round(separator_resolution.lateral_offset_span_m, 6),
            "yellow_separator_source_mesh_ids": list(separator_resolution.source_mesh_ids),
            "lane_group_counts": {
                "side_a": side_a_count,
                "side_b": side_b_count,
            },
            "lane_layout_policy": lane_layout_policy,
            "direction_authority": (
                "valid_yellow_separator_lane_level"
                if use_yellow_split
                else "road_level_default_flow"
            ),
            "direction_lane_indices": {
                role: list(indices) for role, indices in sorted(direction_lane_indices.items())
            },
        }
        road_reports.append(report_row)
        if separator_resolution.required and not separator_resolution.valid:
            separator_unresolved_roads.append(report_row)
        if alignment.fallback_ratio > max_fallback_ratio or len(lane_shapes) != int(road.lanes):
            failing_roads.append(report_row)

    report = {
        "schema_name": "roadway_alignment_report",
        "schema_version": "v1",
        "geometry_authority": FORMAL_GEOMETRY_AUTHORITY,
        "lane_geometry_source": FORMAL_LANE_GEOMETRY_SOURCE,
        "drivable_material_roles": sorted(DRIVABLE_MATERIAL_ROLES),
        "lane_separator_material_roles": sorted(LANE_SEPARATOR_MATERIAL_ROLES),
        "drivable_component_names": sorted(DRIVABLE_COMPONENT_NAMES),
        "rejected_drivable_role_counts": roadway_mesh.rejected_drivable_role_counts,
        "ue_level": FORMAL_UE_LEVEL,
        "roadway_mesh_manifest": str(roadway_mesh.manifest_path),
        "roadway_mesh_hash": roadway_mesh.mesh_hash,
        "road_count": len(alignments),
        "sample_count": total_samples,
        "fallback_sample_count": total_fallbacks,
        "fallback_ratio": round(float(total_fallbacks) / float(total_samples), 6) if total_samples else 1.0,
        "failing_roads": failing_roads,
        "separator_unresolved_roads": separator_unresolved_roads,
        "roads": road_reports,
    }
    return alignments, report


def write_alignment_report(path: Path, report: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def postprocess_net_lane_shapes_to_roadway(net_xml: Path, roadway_mesh: RoadwayMesh) -> dict[str, Any]:
    tree = ET.parse(net_xml)
    root = tree.getroot()
    external_lane_count = 0
    internal_lane_count = 0
    snapped_point_count = 0
    snapped_internal_point_count = 0
    snapped_external_point_count = 0
    max_snap_distance_m = 0.0
    max_internal_snap_distance_m = 0.0
    max_external_snap_distance_m = 0.0
    outside_after_count = 0
    snap_examples: list[dict[str, Any]] = []
    external_snap_discard_source_road_ids: set[int] = set()
    external_fragment_discard_source_road_ids: set[int] = set()
    fragment_examples: list[dict[str, Any]] = []

    for edge in root.findall("edge"):
        is_internal = edge.attrib.get("function") == "internal"
        edge_id = str(edge.attrib.get("id") or "")
        for lane in edge.findall("lane"):
            lane_id = str(lane.attrib.get("id") or "")
            original = _shape_points(lane.attrib.get("shape"))
            if len(original) < 2:
                continue
            if is_internal:
                internal_lane_count += 1
            else:
                external_lane_count += 1
            adjusted: list[tuple[float, float]] = []
            densify_interval_m = 0.05 if is_internal else 0.25
            for x_m, y_m in _densify_xy(original, densify_interval_m):
                if roadway_mesh.covers_xy(x_m, y_m):
                    snapped = (x_m, y_m)
                    snap_distance = 0.0
                else:
                    sx, sy, snap_distance = roadway_mesh.snap_xy(x_m, y_m)
                    snapped = (sx, sy)
                    snapped_point_count += 1
                    max_snap_distance_m = max(max_snap_distance_m, snap_distance)
                    if is_internal:
                        snapped_internal_point_count += 1
                        max_internal_snap_distance_m = max(max_internal_snap_distance_m, snap_distance)
                    else:
                        snapped_external_point_count += 1
                        max_external_snap_distance_m = max(max_external_snap_distance_m, snap_distance)
                        if snap_distance > EXTERNAL_LANE_SNAP_DISCARD_THRESHOLD_M:
                            source_road_id = _source_road_id_from_edge_id(edge_id)
                            if source_road_id is not None:
                                external_snap_discard_source_road_ids.add(source_road_id)
                    snap_examples.append(
                        {
                            "edge_id": edge_id,
                            "lane_id": lane_id,
                            "edge_function": "internal" if is_internal else "normal",
                            "x_m": round(float(x_m), 3),
                            "y_m": round(float(y_m), 3),
                            "snapped_x_m": round(float(sx), 3),
                            "snapped_y_m": round(float(sy), 3),
                            "distance_m": round(float(snap_distance), 6),
                        }
                    )
                    snap_examples.sort(key=lambda item: float(item["distance_m"]), reverse=True)
                    del snap_examples[12:]
                adjusted.append(snapped)
            if len(adjusted) < 2:
                adjusted = list(original)
            for x_m, y_m in adjusted:
                if not roadway_mesh.covers_xy(x_m, y_m):
                    outside_after_count += 1
            if not is_internal:
                max_fragment_heading_delta_deg, simplified = _max_simplified_heading_delta(adjusted)
                max_raw_heading_delta_deg = _max_adjacent_heading_delta(adjusted)
                if (
                    max_fragment_heading_delta_deg > EXTERNAL_LANE_FRAGMENT_HEADING_DELTA_DEG
                    or max_raw_heading_delta_deg > EXTERNAL_LANE_FRAGMENT_RAW_HEADING_DELTA_DEG
                ):
                    source_road_id = _source_road_id_from_edge_id(edge_id)
                    if source_road_id is not None:
                        external_fragment_discard_source_road_ids.add(source_road_id)
                    fragment_examples.append(
                        {
                            "edge_id": edge_id,
                            "lane_id": lane_id,
                            "source_road_id": source_road_id,
                            "max_heading_delta_deg": round(float(max_fragment_heading_delta_deg), 6),
                            "max_raw_heading_delta_deg": round(float(max_raw_heading_delta_deg), 6),
                            "simplified_point_count": len(simplified),
                        }
                    )
                    fragment_examples.sort(
                        key=lambda item: max(
                            float(item["max_heading_delta_deg"]),
                            float(item["max_raw_heading_delta_deg"]),
                        ),
                        reverse=True,
                    )
                    del fragment_examples[12:]
            lane.attrib["shape"] = _format_shape(adjusted)
            lane.attrib["length"] = f"{_path_length(adjusted):.3f}"

    ET.indent(root)
    tree.write(net_xml, encoding="utf-8", xml_declaration=True)
    return {
        "external_lane_count": external_lane_count,
        "internal_lane_count": internal_lane_count,
        "snapped_point_count": snapped_point_count,
        "snapped_internal_point_count": snapped_internal_point_count,
        "snapped_external_point_count": snapped_external_point_count,
        "external_snap_discard_threshold_m": EXTERNAL_LANE_SNAP_DISCARD_THRESHOLD_M,
        "external_snap_discard_source_road_ids": sorted(external_snap_discard_source_road_ids),
        "external_fragment_heading_delta_threshold_deg": EXTERNAL_LANE_FRAGMENT_HEADING_DELTA_DEG,
        "external_fragment_raw_heading_delta_threshold_deg": EXTERNAL_LANE_FRAGMENT_RAW_HEADING_DELTA_DEG,
        "external_fragment_simplify_tolerance_m": EXTERNAL_LANE_FRAGMENT_SIMPLIFY_TOLERANCE_M,
        "external_fragment_discard_source_road_ids": sorted(external_fragment_discard_source_road_ids),
        "max_snap_distance_m": round(max_snap_distance_m, 6),
        "max_internal_snap_distance_m": round(max_internal_snap_distance_m, 6),
        "max_external_snap_distance_m": round(max_external_snap_distance_m, 6),
        "snap_examples": snap_examples,
        "fragment_examples": fragment_examples,
        "outside_after_count": outside_after_count,
    }


def audit_lane_samples_against_roadway(
    rows: Iterable[dict[str, Any]],
    roadway_mesh: RoadwayMesh,
) -> dict[str, Any]:
    total_count = 0
    outside_count = 0
    max_distance_m = 0.0
    examples: list[dict[str, Any]] = []
    outside_source_road_ids: set[int] = set()
    for row in rows:
        total_count += 1
        x_m = float(row["x_m"])
        y_m = float(row["y_m"])
        if roadway_mesh.covers_xy(x_m, y_m):
            continue
        _sx, _sy, distance_m = roadway_mesh.snap_xy(x_m, y_m)
        outside_count += 1
        max_distance_m = max(max_distance_m, distance_m)
        try:
            outside_source_road_ids.add(int(float(row.get("source_road_id") or -1)))
        except (TypeError, ValueError):
            pass
        if len(examples) < 10:
            examples.append(
                {
                    "edge_id": row.get("edge_id"),
                    "lane_id": row.get("lane_id"),
                    "source_road_id": row.get("source_road_id"),
                    "x_m": round(x_m, 3),
                    "y_m": round(y_m, 3),
                    "distance_m": round(distance_m, 6),
                }
            )
    return {
        "sample_count": total_count,
        "outside_roadway_count": outside_count,
        "max_outside_distance_m": round(max_distance_m, 6),
        "outside_source_road_ids": sorted(road_id for road_id in outside_source_road_ids if road_id >= 0),
        "outside_examples": examples,
    }
