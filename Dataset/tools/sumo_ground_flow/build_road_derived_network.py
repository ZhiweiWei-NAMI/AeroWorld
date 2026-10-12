"""Build a SUMO net and traffic bundle from road.geojson topology plus UE RoadWay mesh.

``road.geojson`` is the topology, point-order, corridor width, and physical-lane
count authority.  Its ``direction`` field is only a road-level default flow.
Valid UE YellowLine separators are the lane-level direction authority.  SUMO,
the runtime traffic bundle, and the audit manifest are emitted from the same
RoadWay-aligned road-lane graph so vehicle coordinates are already in the UE
truth XY frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_TOOLS_DIR = _REPO_ROOT / "Dataset" / "tools"
for _ImportPath in (str(_TOOLS_DIR), str(_REPO_ROOT)):
    if _ImportPath not in sys.path:
        sys.path.insert(0, _ImportPath)

try:
    from roadway_mesh_alignment import (  # type: ignore
        ALIGNMENT_SAMPLE_INTERVAL_M,
        DRIVABLE_COMPONENT_NAMES,
        DRIVABLE_MATERIAL_ROLES,
        FORBIDDEN_LANE_TOP_SURFACE_MATERIAL_ROLES,
        FORMAL_GEOMETRY_AUTHORITY,
        FORMAL_LANE_GEOMETRY_SOURCE,
        FORMAL_UE_LEVEL,
        LANE_SEPARATOR_MATERIAL_ROLES,
        LANE_TOP_SURFACE_FOOTPRINT_WIDTH_M,
        RoadwayMesh,
        RoadwayMeshAlignmentError,
        RoadwayRoadAlignment,
        align_roads_to_roadway_mesh,
        audit_lane_samples_against_roadway,
        load_roadway_mesh,
        postprocess_net_lane_shapes_to_roadway,
        write_alignment_report,
    )
    from map_spatial_index import (  # type: ignore
        DEFAULT_MAP_PACKAGE,
        ROOT,
        GeoJsonBundleFit,
        SpatialValidationError,
        _bounds_center_mercator,
        _geojson_coord_to_local_xy,
    )
    from sumo_ground_flow.coordinates import sumo_executable_from_environment  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - supports package imports from repo root.
    from Dataset.tools.sumo_ground_flow.roadway_mesh_alignment import (  # type: ignore
        ALIGNMENT_SAMPLE_INTERVAL_M,
        DRIVABLE_COMPONENT_NAMES,
        DRIVABLE_MATERIAL_ROLES,
        FORBIDDEN_LANE_TOP_SURFACE_MATERIAL_ROLES,
        FORMAL_GEOMETRY_AUTHORITY,
        FORMAL_LANE_GEOMETRY_SOURCE,
        FORMAL_UE_LEVEL,
        LANE_SEPARATOR_MATERIAL_ROLES,
        LANE_TOP_SURFACE_FOOTPRINT_WIDTH_M,
        RoadwayMesh,
        RoadwayMeshAlignmentError,
        RoadwayRoadAlignment,
        align_roads_to_roadway_mesh,
        audit_lane_samples_against_roadway,
        load_roadway_mesh,
        postprocess_net_lane_shapes_to_roadway,
        write_alignment_report,
    )
    from Dataset.tools.map_spatial_index import (  # type: ignore
        DEFAULT_MAP_PACKAGE,
        ROOT,
        GeoJsonBundleFit,
        SpatialValidationError,
        _bounds_center_mercator,
        _geojson_coord_to_local_xy,
    )
    from Dataset.tools.sumo_ground_flow.coordinates import sumo_executable_from_environment  # type: ignore


DEFAULT_ROAD_DERIVED_SOURCE_DIR = (
    ROOT / "Plugins" / "SumoImporter" / "Maps" / "donghu_road_topo" / "source" / "road_derived"
)
DEFAULT_ROAD_DERIVED_NET_XML = DEFAULT_ROAD_DERIVED_SOURCE_DIR / "road_derived.net.xml"
DEFAULT_TRAFFIC_BUNDLE_DIR = ROOT / "Config" / "LowAltitude" / "Maps" / "donghu_road_topo" / "traffic_bundle"
DEFAULT_ROADWAY_MESH_MANIFEST = Path("G:/aw_cap/_meta/road_alignment/roadway_mesh_manifest.json")
LANE_SAMPLE_INTERVAL_M = 1.0
MAX_EXTERNAL_LANE_RAW_HEADING_DELTA_DEG = 65.0
MAX_EXTERNAL_LANE_RAW_Z_DELTA_M = 1.0
SELF_LOOP_SPLIT_MIN_POINTS = 2
NODE_QUANTIZATION_M = 0.01
ROAD_DERIVED_COORDINATE_ROUTE = "sumo_xy_identity_truth_enu_m"


class RoadDerivedNetworkError(RuntimeError):
    """Raised when road-derived SUMO generation or validation fails."""


@dataclass(frozen=True)
class RoadFeature:
    road_id: int
    source_index: int
    properties: dict[str, Any]
    centerline_xyz: tuple[tuple[float, float, float], ...]
    lanes: int
    width_m: float
    direction: int
    isolation: int
    func_level: int
    manage_level: int


@dataclass(frozen=True)
class PlannedEdge:
    edge_id: str
    source_edge_id: str
    source_road_id: int
    direction_role: str
    physical_lane_index: int
    lane_width_m: float
    speed_mps: float
    priority: int
    from_node: str
    to_node: str
    shape_xyz: tuple[tuple[float, float, float], ...]
    split_part: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _map_sources(map_package: Path) -> tuple[Path, Path, Path, Path]:
    package = _load_json(map_package)
    source_geojson = dict(package.get("source_geojson") or {})
    road_geojson = _project_path(str(source_geojson.get("road") or ""))
    bounds_geojson = _project_path(str(source_geojson.get("bounds") or ""))
    traffic_bundle_dir = _project_path(str(package.get("traffic_bundle_dir") or ""))
    plugin_source_dir = _project_path(str(package.get("plugin_owned_source_dir") or ""))
    return road_geojson, bounds_geojson, traffic_bundle_dir, plugin_source_dir


def _fit_from_transform_json(path: Path) -> GeoJsonBundleFit | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    matrix = payload.get("matrix")
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise RoadDerivedNetworkError(f"Invalid road_geojson_transform matrix: {path}")
    return GeoJsonBundleFit(
        matrix=(
            (float(matrix[0][0]), float(matrix[0][1])),
            (float(matrix[1][0]), float(matrix[1][1])),
            (float(matrix[2][0]), float(matrix[2][1])),
        ),
        mean_error_m=float(payload.get("mean_error_m") or 0.0),
        max_error_m=float(payload.get("max_error_m") or 0.0),
        pair_count=int(payload.get("pair_count") or 0),
    )


def _load_required_fit(
    *,
    traffic_bundle_dir: Path,
) -> GeoJsonBundleFit:
    transform_json = traffic_bundle_dir / "road_geojson_transform.json"
    existing = _fit_from_transform_json(transform_json)
    if existing is not None:
        return existing
    raise RoadDerivedNetworkError(
        f"{transform_json} is missing. Formal RoadWay-aligned generation does not bootstrap from legacy lane_center_samples.csv."
    )


def _transform_geojson_coord(
    coord: Sequence[Any],
    *,
    center_xy_m: tuple[float, float],
    fit: GeoJsonBundleFit,
) -> tuple[float, float, float]:
    local_x, local_y = _geojson_coord_to_local_xy(coord, center_xy_m)
    x_m, y_m = fit.transform_local_xy(local_x, local_y)
    z_m = float(coord[2]) if len(coord) >= 3 else 0.0
    return x_m, y_m, z_m


def _load_roads(road_geojson_path: Path, bounds_geojson_path: Path, fit: GeoJsonBundleFit) -> list[RoadFeature]:
    payload = _load_json(road_geojson_path)
    center_xy = _bounds_center_mercator(bounds_geojson_path)
    roads: list[RoadFeature] = []
    for source_index, feature in enumerate(payload.get("features") or []):
        geometry = dict(feature.get("geometry") or {})
        if str(geometry.get("type") or "") != "LineString":
            continue
        coords = geometry.get("coordinates") or []
        if len(coords) < 2:
            continue
        props = dict(feature.get("properties") or {})
        road_id = int(float(props.get("id", source_index)))
        lanes = max(1, int(float(props.get("lanes") or 1)))
        width_m = float(props.get("width") or lanes * 3.0)
        if not math.isfinite(width_m) or width_m <= 0.0:
            width_m = lanes * 3.0
        direction = int(float(props.get("direction") or 1))
        isolation = int(float(props.get("isolation") or 0))
        roads.append(
            RoadFeature(
                road_id=road_id,
                source_index=source_index,
                properties=props,
                centerline_xyz=tuple(
                    _transform_geojson_coord(coord, center_xy_m=center_xy, fit=fit) for coord in coords
                ),
                lanes=lanes,
                width_m=width_m,
                direction=direction,
                isolation=isolation,
                func_level=int(float(props.get("func_level") or 9)),
                manage_level=int(float(props.get("manage_level") or 10)),
            )
        )
    if not roads:
        raise RoadDerivedNetworkError(f"No usable road LineString features found in {road_geojson_path}")
    return roads


def _path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(
        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        for a, b in zip(points, points[1:])
    )


def _heading_delta_deg(a_deg: float, b_deg: float) -> float:
    return abs((float(b_deg) - float(a_deg) + 180.0) % 360.0 - 180.0)


def _max_adjacent_heading_delta(points: Sequence[Sequence[float]]) -> float:
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


def _max_adjacent_z_delta(points: Sequence[Sequence[float]]) -> float:
    max_delta = 0.0
    for start, end in zip(points, points[1:]):
        if len(start) < 3 or len(end) < 3:
            continue
        max_delta = max(max_delta, abs(float(end[2]) - float(start[2])))
    return max_delta


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
            yaw_deg = math.degrees(math.atan2(dy, dx))
            return (
                float(last[0]) + dx * t,
                float(last[1]) + dy * t,
                float(last[2]) + dz * t if len(last) >= 3 else 0.0,
                yaw_deg,
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


def _split_polyline_half(
    points: Sequence[tuple[float, float, float]],
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]]:
    total = _path_length(points)
    if len(points) < SELF_LOOP_SPLIT_MIN_POINTS or total <= 1e-9:
        return tuple(points), tuple(points)
    half = total * 0.5
    first: list[tuple[float, float, float]] = [points[0]]
    remaining = half
    last = points[0]
    for index, current in enumerate(points[1:], start=1):
        seg_len = math.hypot(current[0] - last[0], current[1] - last[1])
        if seg_len > 1e-9 and remaining <= seg_len:
            t = remaining / seg_len
            mid = (
                last[0] + (current[0] - last[0]) * t,
                last[1] + (current[1] - last[1]) * t,
                last[2] + (current[2] - last[2]) * t,
            )
            first.append(mid)
            second = [mid, *points[index:]]
            return tuple(first), tuple(second)
        first.append(current)
        remaining -= seg_len
        last = current
    return tuple(points[:-1]), tuple(points[-2:])


def _node_key(point: Sequence[float]) -> tuple[int, int]:
    return (
        int(round(float(point[0]) / NODE_QUANTIZATION_M)),
        int(round(float(point[1]) / NODE_QUANTIZATION_M)),
    )


def _format_shape(points: Sequence[Sequence[float]]) -> str:
    return " ".join(f"{float(point[0]):.3f},{float(point[1]):.3f}" for point in points)


def _speed_for_road(road: RoadFeature) -> float:
    if road.func_level <= 2:
        return 13.89
    if road.func_level <= 4:
        return 11.11
    if road.func_level <= 6:
        return 8.33
    return 6.94


def _priority_for_road(road: RoadFeature) -> int:
    return max(1, min(9, 10 - int(road.func_level)))


def _direction_roles(road: RoadFeature, alignment: RoadwayRoadAlignment | None = None) -> list[tuple[str, list[int]]]:
    if alignment is not None and alignment.direction_lane_indices:
        roles: list[tuple[str, list[int]]] = []
        for role in ("f", "r"):
            indices = list(alignment.direction_lane_indices.get(role) or ())
            if indices:
                roles.append((role, indices))
        if roles:
            return roles
    if road.direction == 4:
        return [("r", list(range(road.lanes)))]
    if road.direction == 2:
        forward_count = (road.lanes + 1) // 2
        roles: list[tuple[str, list[int]]] = []
        if forward_count > 0:
            roles.append(("f", list(range(forward_count))))
        if road.lanes - forward_count > 0:
            roles.append(("r", list(range(forward_count, road.lanes))))
        return roles
    return [("f", list(range(road.lanes)))]


def _build_nodes_and_edges(
    roads: Sequence[RoadFeature],
    road_alignments: dict[int, RoadwayRoadAlignment],
    roadway_mesh: RoadwayMesh,
) -> tuple[dict[str, tuple[float, float]], list[PlannedEdge], dict[str, Any]]:
    node_ids_by_key: dict[tuple[int, int], str] = {}
    node_accumulator: dict[str, list[float]] = {}

    def node_for(topology_point: Sequence[float], visual_point: Sequence[float]) -> str:
        nonlocal node_accumulator
        key = _node_key(topology_point)
        node_id = node_ids_by_key.get(key)
        if node_id is None:
            node_id = f"n_{len(node_ids_by_key):05d}"
            node_ids_by_key[key] = node_id
            node_accumulator[node_id] = [0.0, 0.0, 0.0]
        node_accumulator[node_id][0] += float(visual_point[0])
        node_accumulator[node_id][1] += float(visual_point[1])
        node_accumulator[node_id][2] += 1.0
        return node_id

    base_edges: list[PlannedEdge] = []
    self_loop_source_edges: list[str] = []
    discarded_abrupt_lane_edges: list[dict[str, Any]] = []
    discarded_z_jump_lane_edges: list[dict[str, Any]] = []
    discarded_top_surface_lane_edges: list[dict[str, Any]] = []
    for road in roads:
        if len(road.centerline_xyz) < 2:
            continue
        alignment = road_alignments.get(road.road_id)
        if alignment is None or len(alignment.visual_centerline_xyz) < 2:
            raise RoadDerivedNetworkError(f"Missing UE RoadWay alignment for road {road.road_id}")
        start_node = node_for(road.centerline_xyz[0], alignment.visual_centerline_xyz[0])
        end_node = node_for(road.centerline_xyz[-1], alignment.visual_centerline_xyz[-1])
        speed_mps = _speed_for_road(road)
        priority = _priority_for_road(road)
        for role, lane_indices in _direction_roles(road, alignment):
            is_reverse = role == "r"
            for physical_lane_index in lane_indices:
                lane_shape = alignment.lane_shapes.get(physical_lane_index)
                if lane_shape is None:
                    raise RoadDerivedNetworkError(
                        f"Missing UE RoadWay lane shape for road={road.road_id} physical_lane_index={physical_lane_index}"
                    )
                shape = lane_shape.shape_xyz
                if is_reverse:
                    shape = tuple(reversed(shape))
                edge_id = f"cg_edge_{road.road_id}_{role}_pl{physical_lane_index}"
                top_surface_audit = roadway_mesh.top_surface_audit(shape)
                if bool(top_surface_audit.get("lane_top_surface_rejected")):
                    discarded_top_surface_lane_edges.append(
                        {
                            "edge_id": edge_id,
                            "source_road_id": road.road_id,
                            "direction_role": role,
                            "physical_lane_index": physical_lane_index,
                            "discard_reason": "generated_lane_geometry_top_surface_sidewalk",
                            "top_surface_footprint_width_m": top_surface_audit.get("footprint_width_m"),
                            "footprint_lateral_offsets_m": top_surface_audit.get("footprint_lateral_offsets_m"),
                            "forbidden_top_material_roles": sorted(FORBIDDEN_LANE_TOP_SURFACE_MATERIAL_ROLES),
                            "sample_count": top_surface_audit.get("sample_count"),
                            "footprint_sample_count": top_surface_audit.get("footprint_sample_count"),
                            "forbidden_footprint_hit_count": top_surface_audit.get("forbidden_footprint_hit_count"),
                            "invalid_top_surface_count": top_surface_audit.get("invalid_top_surface_count"),
                            "invalid_top_surface_station_ratio": top_surface_audit.get(
                                "invalid_top_surface_station_ratio"
                            ),
                            "lane_top_surface_reject_min_stations": top_surface_audit.get(
                                "lane_top_surface_reject_min_stations"
                            ),
                            "lane_top_surface_reject_min_station_ratio": top_surface_audit.get(
                                "lane_top_surface_reject_min_station_ratio"
                            ),
                            "top_material_role_counts": top_surface_audit.get("top_material_role_counts"),
                            "invalid_top_material_role_counts": top_surface_audit.get(
                                "invalid_top_material_role_counts"
                            ),
                            "invalid_examples": top_surface_audit.get("invalid_examples"),
                        }
                    )
                    continue
                max_raw_z_delta_m = _max_adjacent_z_delta(shape)
                if max_raw_z_delta_m > MAX_EXTERNAL_LANE_RAW_Z_DELTA_M:
                    discarded_z_jump_lane_edges.append(
                        {
                            "edge_id": edge_id,
                            "source_road_id": road.road_id,
                            "direction_role": role,
                            "physical_lane_index": physical_lane_index,
                            "max_raw_z_delta_m": round(float(max_raw_z_delta_m), 6),
                            "discard_threshold_m": MAX_EXTERNAL_LANE_RAW_Z_DELTA_M,
                            "discard_reason": "generated_lane_geometry_abrupt_z_layer_jump",
                        }
                    )
                    continue
                max_raw_heading_delta_deg = _max_adjacent_heading_delta(shape)
                if max_raw_heading_delta_deg > MAX_EXTERNAL_LANE_RAW_HEADING_DELTA_DEG:
                    discarded_abrupt_lane_edges.append(
                        {
                            "edge_id": edge_id,
                            "source_road_id": road.road_id,
                            "direction_role": role,
                            "physical_lane_index": physical_lane_index,
                            "max_raw_heading_delta_deg": round(float(max_raw_heading_delta_deg), 6),
                            "discard_reason": "generated_lane_geometry_abrupt_heading_change",
                        }
                    )
                    continue
                from_node = end_node if is_reverse else start_node
                to_node = start_node if is_reverse else end_node
                base_edges.append(
                    PlannedEdge(
                        edge_id=edge_id,
                        source_edge_id=edge_id,
                        source_road_id=road.road_id,
                        direction_role=role,
                        physical_lane_index=physical_lane_index,
                        lane_width_m=lane_shape.lane_width_m,
                        speed_mps=speed_mps,
                        priority=priority,
                        from_node=from_node,
                        to_node=to_node,
                        shape_xyz=shape,
                        split_part="",
                    )
                )

    edges: list[PlannedEdge] = []
    node_xy: dict[str, tuple[float, float]] = {
        node_id: (values[0] / max(1.0, values[2]), values[1] / max(1.0, values[2]))
        for node_id, values in node_accumulator.items()
    }
    for edge in base_edges:
        if edge.from_node != edge.to_node:
            edges.append(edge)
            continue
        first_half, second_half = _split_polyline_half(edge.shape_xyz)
        if not first_half or not second_half:
            raise RoadDerivedNetworkError(f"Cannot split self-loop edge with empty shape: {edge.edge_id}")
        virtual_node = f"v_{edge.edge_id}_mid"
        mid = first_half[-1]
        node_xy[virtual_node] = (mid[0], mid[1])
        self_loop_source_edges.append(edge.edge_id)
        edges.append(
            PlannedEdge(
                edge_id=f"{edge.edge_id}_a",
                source_edge_id=edge.edge_id,
                source_road_id=edge.source_road_id,
                direction_role=edge.direction_role,
                physical_lane_index=edge.physical_lane_index,
                lane_width_m=edge.lane_width_m,
                speed_mps=edge.speed_mps,
                priority=edge.priority,
                from_node=edge.from_node,
                to_node=virtual_node,
                shape_xyz=first_half,
                split_part="a",
            )
        )
        edges.append(
            PlannedEdge(
                edge_id=f"{edge.edge_id}_b",
                source_edge_id=edge.edge_id,
                source_road_id=edge.source_road_id,
                direction_role=edge.direction_role,
                physical_lane_index=edge.physical_lane_index,
                lane_width_m=edge.lane_width_m,
                speed_mps=edge.speed_mps,
                priority=edge.priority,
                from_node=virtual_node,
                to_node=edge.to_node,
                shape_xyz=second_half,
                split_part="b",
            )
        )

    stats = {
        "base_directed_physical_edge_count": len(base_edges),
        "self_loop_source_edge_count": len(self_loop_source_edges),
        "self_loop_source_edges": self_loop_source_edges,
        "expected_output_edge_count": len(edges),
        "node_count": len(node_xy),
        "discarded_abrupt_lane_edge_count": len(discarded_abrupt_lane_edges),
        "discarded_abrupt_lane_heading_delta_threshold_deg": MAX_EXTERNAL_LANE_RAW_HEADING_DELTA_DEG,
        "discarded_abrupt_lane_edges": discarded_abrupt_lane_edges,
        "discarded_z_jump_lane_edge_count": len(discarded_z_jump_lane_edges),
        "discarded_z_jump_lane_delta_threshold_m": MAX_EXTERNAL_LANE_RAW_Z_DELTA_M,
        "discarded_z_jump_lane_edges": discarded_z_jump_lane_edges,
        "discarded_top_surface_lane_edge_count": len(discarded_top_surface_lane_edges),
        "discarded_top_surface_footprint_width_m": LANE_TOP_SURFACE_FOOTPRINT_WIDTH_M,
        "discarded_top_surface_forbidden_material_roles": sorted(FORBIDDEN_LANE_TOP_SURFACE_MATERIAL_ROLES),
        "discarded_top_surface_lane_edges": discarded_top_surface_lane_edges,
    }
    return node_xy, edges, stats


def _traffic_light_nodes(
    roads: Sequence[RoadFeature],
    edges: Sequence[PlannedEdge],
) -> set[str]:
    road_by_id = {road.road_id: road for road in roads}
    incoming_predecessors: dict[str, set[str]] = {}
    lane_degree_counts: dict[str, int] = {}
    neighbors: dict[str, set[str]] = {}
    roads_by_node: dict[str, set[int]] = {}
    for edge in edges:
        incoming_predecessors.setdefault(edge.to_node, set()).add(edge.from_node)
        lane_degree_counts[edge.from_node] = lane_degree_counts.get(edge.from_node, 0) + 1
        lane_degree_counts[edge.to_node] = lane_degree_counts.get(edge.to_node, 0) + 1
        neighbors.setdefault(edge.from_node, set()).add(edge.to_node)
        neighbors.setdefault(edge.to_node, set()).add(edge.from_node)
        roads_by_node.setdefault(edge.from_node, set()).add(edge.source_road_id)
        roads_by_node.setdefault(edge.to_node, set()).add(edge.source_road_id)
    tls_nodes: set[str] = set()
    for node_id, node_neighbors in neighbors.items():
        incident_roads = [road_by_id[road_id] for road_id in roads_by_node.get(node_id, set()) if road_id in road_by_id]
        has_major_or_managed = any(road.func_level <= 2 or road.manage_level <= 3 for road in incident_roads)
        if len(incoming_predecessors.get(node_id, set())) >= 2 and lane_degree_counts.get(node_id, 0) >= 3 and has_major_or_managed:
            tls_nodes.add(node_id)
    return tls_nodes


def _write_plain_xml(
    *,
    node_xml: Path,
    edge_xml: Path,
    nodes: dict[str, tuple[float, float]],
    edges: Sequence[PlannedEdge],
    tls_nodes: set[str],
    overwrite: bool,
) -> None:
    for path in (node_xml, edge_xml):
        if path.exists() and not overwrite:
            raise RoadDerivedNetworkError(f"Refusing to overwrite existing generated file without --overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    node_root = ET.Element("nodes")
    for node_id in sorted(nodes):
        x_m, y_m = nodes[node_id]
        attrib = {"id": node_id, "x": f"{x_m:.3f}", "y": f"{y_m:.3f}"}
        if node_id in tls_nodes:
            attrib["type"] = "traffic_light"
        ET.SubElement(node_root, "node", attrib)
    ET.indent(node_root)
    ET.ElementTree(node_root).write(node_xml, encoding="utf-8", xml_declaration=True)

    edge_root = ET.Element("edges")
    for edge in sorted(edges, key=lambda item: item.edge_id):
        ET.SubElement(
            edge_root,
            "edge",
            {
                "id": edge.edge_id,
                "from": edge.from_node,
                "to": edge.to_node,
                "priority": str(edge.priority),
                "numLanes": "1",
                "speed": f"{edge.speed_mps:.3f}",
                "width": f"{edge.lane_width_m:.3f}",
                "spreadType": "center",
                "shape": _format_shape(edge.shape_xyz),
            },
        )
    ET.indent(edge_root)
    ET.ElementTree(edge_root).write(edge_xml, encoding="utf-8", xml_declaration=True)


def _run_netconvert(*, netconvert: Path, node_xml: Path, edge_xml: Path, net_xml: Path, overwrite: bool) -> list[str]:
    if net_xml.exists() and not overwrite:
        raise RoadDerivedNetworkError(f"Refusing to overwrite existing SUMO net without --overwrite: {net_xml}")
    net_xml.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(netconvert),
        "-n",
        str(node_xml),
        "-e",
        str(edge_xml),
        "-o",
        str(net_xml),
        "--offset.disable-normalization",
        "true",
        "--no-turnarounds",
        "true",
        "--no-turnarounds.tls",
        "true",
        "--no-warnings",
        "true",
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RoadDerivedNetworkError(
            "netconvert failed with exit code "
            f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return cmd


def _audit_net(net_xml: Path, expected_edges: Iterable[str]) -> dict[str, Any]:
    expected = set(expected_edges)
    normal_edges: set[str] = set()
    lane_count = 0
    connection_count = 0
    crossing_count = 0
    walkingarea_count = 0
    pedestrian_allow_lane_count = 0
    tl_logic_ids: set[str] = set()
    junction_tls: set[str] = set()
    for _event, elem in ET.iterparse(net_xml, events=("end",)):
        if elem.tag == "edge":
            function = str(elem.attrib.get("function") or "")
            if function == "walkingarea":
                walkingarea_count += 1
            if function != "internal":
                edge_id = str(elem.attrib.get("id") or "")
                normal_edges.add(edge_id)
                lanes = list(elem.findall("lane"))
                lane_count += len(lanes)
                for lane in lanes:
                    allow = str(lane.attrib.get("allow") or "")
                    if "pedestrian" in allow.split():
                        pedestrian_allow_lane_count += 1
            elem.clear()
        elif elem.tag == "connection":
            connection_count += 1
            elem.clear()
        elif elem.tag == "crossing":
            crossing_count += 1
            elem.clear()
        elif elem.tag == "tlLogic":
            tl_logic_ids.add(str(elem.attrib.get("id") or ""))
            elem.clear()
        elif elem.tag == "junction":
            if elem.attrib.get("type") == "traffic_light" or elem.attrib.get("tl"):
                junction_tls.add(str(elem.attrib.get("id") or elem.attrib.get("tl") or ""))
            elem.clear()
    missing = sorted(expected - normal_edges)
    unexpected = sorted(edge_id for edge_id in normal_edges - expected if not edge_id.startswith(":"))
    audit = {
        "normal_edge_count": len(normal_edges),
        "lane_count": lane_count,
        "connection_count": connection_count,
        "tl_logic_count": len(tl_logic_ids),
        "junction_traffic_light_count": len(junction_tls),
        "traffic_light_ids": sorted(tl_logic_ids),
        "crossing_count": crossing_count,
        "walkingarea_count": walkingarea_count,
        "pedestrian_allow_lane_count": pedestrian_allow_lane_count,
        "missing_expected_edges": missing,
        "unexpected_normal_edges": unexpected,
    }
    failures = []
    if missing:
        failures.append(f"missing expected SUMO edges: {missing[:10]}")
    if crossing_count:
        failures.append(f"crossing_count must be 0, got {crossing_count}")
    if walkingarea_count:
        failures.append(f"walkingarea_count must be 0, got {walkingarea_count}")
    if pedestrian_allow_lane_count:
        failures.append(f"pedestrian allow lanes must be 0, got {pedestrian_allow_lane_count}")
    if failures:
        raise RoadDerivedNetworkError("; ".join(failures))
    return audit


def _shape_points(value: str | None) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for raw in str(value or "").split():
        parts = raw.split(",")
        if len(parts) >= 2:
            points.append((float(parts[0]), float(parts[1])))
    return tuple(points)


def _iter_net_lanes(
    net_xml: Path,
    *,
    include_internal: bool = False,
) -> Iterable[tuple[str, dict[str, str], dict[str, str]]]:
    for _event, elem in ET.iterparse(net_xml, events=("end",)):
        if elem.tag == "lane":
            continue
        if elem.tag != "edge":
            elem.clear()
            continue
        if elem.attrib.get("function") == "internal" and not include_internal:
            elem.clear()
            continue
        edge_attrib = dict(elem.attrib)
        for lane in elem.findall("lane"):
            yield str(edge_attrib.get("id") or ""), edge_attrib, dict(lane.attrib)
        elem.clear()


def _sample_s_values(length_m: float) -> list[float]:
    sample_count = max(1, int(math.floor(length_m / LANE_SAMPLE_INTERVAL_M)))
    values = [float(index) * LANE_SAMPLE_INTERVAL_M for index in range(sample_count + 1)]
    if not values or abs(values[-1] - length_m) > 1e-6:
        values.append(length_m)
    return values


def _edge_id_from_lane_id(lane_id: str) -> str:
    if "_" not in lane_id:
        return lane_id
    return lane_id.rsplit("_", 1)[0]


def _resolve_planned_edge_id(
    *,
    edge_id: str,
    side: str,
    planned_by_edge: dict[str, PlannedEdge],
    internal_edge_to_connection: dict[str, dict[str, Any]],
    stack: frozenset[str] = frozenset(),
) -> str | None:
    if edge_id in planned_by_edge:
        return edge_id
    if not edge_id.startswith(":") or edge_id in stack:
        return None
    connection = internal_edge_to_connection.get(edge_id)
    if connection is None:
        return None
    next_edge_id = str((connection.get("from_edge") if side == "from" else connection.get("to_edge")) or "")
    return _resolve_planned_edge_id(
        edge_id=next_edge_id,
        side=side,
        planned_by_edge=planned_by_edge,
        internal_edge_to_connection=internal_edge_to_connection,
        stack=stack | frozenset((edge_id,)),
    )


def _build_internal_lane_metrics(
    *,
    net_lanes: Sequence[tuple[str, dict[str, str], dict[str, str]]],
    planned_by_edge: dict[str, PlannedEdge],
    connections_by_via: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    internal_lane_lengths: dict[str, float] = {}
    internal_edge_to_lane: dict[str, str] = {}
    for edge_id, edge_attrib, lane_attrib in net_lanes:
        if edge_attrib.get("function") != "internal":
            continue
        lane_id = str(lane_attrib.get("id") or f"{edge_id}_0")
        lane_shape_xy = _shape_points(lane_attrib.get("shape"))
        length_m = float(lane_attrib.get("length") or _path_length(lane_shape_xy))
        internal_lane_lengths[lane_id] = length_m
        internal_edge_to_lane[edge_id] = lane_id

    internal_edge_to_connection = {
        _edge_id_from_lane_id(via_lane): row for via_lane, row in connections_by_via.items()
    }
    successor_via_by_from_edge: dict[str, list[str]] = {}
    for via_lane, row in connections_by_via.items():
        from_edge_id = str(row.get("from_edge") or "")
        if from_edge_id.startswith(":"):
            successor_via_by_from_edge.setdefault(from_edge_id, []).append(via_lane)

    start_cache: dict[str, float] = {}
    suffix_cache: dict[str, float] = {}
    origin_cache: dict[str, str | None] = {}
    dest_cache: dict[str, str | None] = {}

    def resolve_origin(lane_id: str) -> str | None:
        if lane_id not in origin_cache:
            row = connections_by_via[lane_id]
            origin_cache[lane_id] = _resolve_planned_edge_id(
                edge_id=str(row.get("from_edge") or ""),
                side="from",
                planned_by_edge=planned_by_edge,
                internal_edge_to_connection=internal_edge_to_connection,
            )
        return origin_cache[lane_id]

    def resolve_dest(lane_id: str) -> str | None:
        if lane_id not in dest_cache:
            row = connections_by_via[lane_id]
            dest_cache[lane_id] = _resolve_planned_edge_id(
                edge_id=str(row.get("to_edge") or ""),
                side="to",
                planned_by_edge=planned_by_edge,
                internal_edge_to_connection=internal_edge_to_connection,
            )
        return dest_cache[lane_id]

    def start_s(lane_id: str, stack: frozenset[str] = frozenset()) -> float:
        if lane_id in start_cache:
            return start_cache[lane_id]
        if lane_id in stack:
            raise RoadDerivedNetworkError(f"Internal lane chain has a cycle at {lane_id}")
        row = connections_by_via[lane_id]
        predecessor_edge_id = str(row.get("from_edge") or "")
        predecessor_lane_id = internal_edge_to_lane.get(predecessor_edge_id)
        if predecessor_lane_id is None:
            value = 0.0
        else:
            value = start_s(predecessor_lane_id, stack | frozenset((lane_id,))) + internal_lane_lengths.get(
                predecessor_lane_id,
                0.0,
            )
        start_cache[lane_id] = value
        return value

    def suffix_length(lane_id: str, stack: frozenset[str] = frozenset()) -> float:
        if lane_id in suffix_cache:
            return suffix_cache[lane_id]
        if lane_id in stack:
            raise RoadDerivedNetworkError(f"Internal lane chain has a cycle at {lane_id}")
        edge_id = _edge_id_from_lane_id(lane_id)
        origin_id = resolve_origin(lane_id)
        dest_id = resolve_dest(lane_id)
        best_successor: str | None = None
        for successor_lane_id in successor_via_by_from_edge.get(edge_id, []):
            if resolve_origin(successor_lane_id) == origin_id and resolve_dest(successor_lane_id) == dest_id:
                best_successor = successor_lane_id
                break
        if best_successor is None:
            value = 0.0
        else:
            value = internal_lane_lengths.get(best_successor, 0.0) + suffix_length(
                best_successor,
                stack | frozenset((lane_id,)),
            )
        suffix_cache[lane_id] = value
        return value

    metrics: dict[str, dict[str, Any]] = {}
    for lane_id in sorted(internal_lane_lengths):
        if lane_id not in connections_by_via:
            raise RoadDerivedNetworkError(f"Internal lane has no matching connection via lane: {lane_id}")
        origin_id = resolve_origin(lane_id)
        dest_id = resolve_dest(lane_id)
        if origin_id is None or dest_id is None:
            row = connections_by_via.get(lane_id, {})
            raise RoadDerivedNetworkError(
                "Internal lane references unknown planned edge chain: "
                f"lane={lane_id} from={row.get('from_edge')} to={row.get('to_edge')}"
            )
        lane_length = internal_lane_lengths[lane_id]
        lane_start_s = start_s(lane_id)
        metrics[lane_id] = {
            "from_edge_id": origin_id,
            "to_edge_id": dest_id,
            "from_planned": planned_by_edge[origin_id],
            "to_planned": planned_by_edge[dest_id],
            "chain_start_s": lane_start_s,
            "chain_total_length_m": max(lane_start_s + lane_length + suffix_length(lane_id), lane_length),
        }
    return metrics


def _internal_lane_z_at_s(
    *,
    from_edge: PlannedEdge,
    to_edge: PlannedEdge,
    s_m: float,
    length_m: float,
) -> float:
    z0 = from_edge.shape_xyz[-1][2] if from_edge.shape_xyz else 0.0
    z1 = to_edge.shape_xyz[0][2] if to_edge.shape_xyz else z0
    if length_m <= 1e-9:
        return z0
    t = max(0.0, min(1.0, s_m / length_m))
    return z0 + (z1 - z0) * t


def _sample_net_lane_rows(
    *,
    net_xml: Path,
    planned_by_edge: dict[str, PlannedEdge],
    connection_rows: Sequence[dict[str, Any]],
    roadway_mesh_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    center_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    stats = {
        "normal_lane_count": 0,
        "internal_lane_count": 0,
        "internal_lane_sample_count": 0,
    }
    connections_by_via = {
        str(row.get("via_lane") or ""): row for row in connection_rows if str(row.get("via_lane") or "")
    }
    net_lanes = list(_iter_net_lanes(net_xml, include_internal=True))
    internal_lane_metrics = _build_internal_lane_metrics(
        net_lanes=net_lanes,
        planned_by_edge=planned_by_edge,
        connections_by_via=connections_by_via,
    )
    sampled_lane_ids: set[str] = set()
    for edge_id, edge_attrib, lane_attrib in net_lanes:
        lane_id = str(lane_attrib.get("id") or f"{edge_id}_0")
        lane_index = int(lane_attrib.get("index") or 0)
        lane_shape_xy = _shape_points(lane_attrib.get("shape"))
        if len(lane_shape_xy) < 2:
            continue
        length_m = float(lane_attrib.get("length") or _path_length(lane_shape_xy))
        s_values = _sample_s_values(length_m)
        is_internal = edge_attrib.get("function") == "internal"

        if is_internal:
            metrics = internal_lane_metrics.get(lane_id)
            if metrics is None:
                raise RoadDerivedNetworkError(f"Internal lane has no matching connection via lane: {lane_id}")
            from_edge_id = str(metrics["from_edge_id"])
            to_edge_id = str(metrics["to_edge_id"])
            from_planned = metrics["from_planned"]
            to_planned = metrics["to_planned"]
            planned = from_planned
            edge_type = "road_geojson_internal_lane"
            direction_role = "internal"
            source_edge_id = f"{from_edge_id}->{to_edge_id}"
            split_part = "internal"
            stats["internal_lane_count"] += 1
        else:
            planned = planned_by_edge.get(edge_id)
            if planned is None:
                raise RoadDerivedNetworkError(f"SUMO normal edge has no planned road-derived edge: {edge_id}")
            from_planned = planned
            to_planned = planned
            edge_type = "road_geojson_physical_lane"
            direction_role = planned.direction_role
            source_edge_id = planned.source_edge_id
            split_part = planned.split_part
            stats["normal_lane_count"] += 1
        lane_width_m = float(lane_attrib.get("width") or planned.lane_width_m)

        for s_m in s_values:
            x_m, y_m, _unused_z, yaw_deg = _point_at_s(tuple((x, y, 0.0) for x, y in lane_shape_xy), s_m)
            if is_internal:
                chain_s_m = float(metrics["chain_start_s"]) + s_m
                z_m = _internal_lane_z_at_s(
                    from_edge=from_planned,
                    to_edge=to_planned,
                    s_m=chain_s_m,
                    length_m=float(metrics["chain_total_length_m"]),
                )
            else:
                _px, _py, z_m, _unused_plan_yaw = _point_at_s(planned.shape_xyz, s_m)
            center_rows.append(
                {
                    "edge_id": edge_id,
                    "lane_id": lane_id,
                    "lane_index": lane_index,
                    "s_m": f"{s_m:.3f}",
                    "x_m": f"{x_m:.3f}",
                    "y_m": f"{y_m:.3f}",
                    "z_m": f"{z_m:.3f}",
                    "yaw_deg": f"{yaw_deg:.3f}",
                    "source_road_id": planned.source_road_id,
                    "direction_role": direction_role,
                    "physical_lane_index": planned.physical_lane_index,
                    "lane_width_m": f"{lane_width_m:.3f}",
                    "coordinate_route": ROAD_DERIVED_COORDINATE_ROUTE,
                    "geometry_authority": FORMAL_GEOMETRY_AUTHORITY,
                    "lane_geometry_source": FORMAL_LANE_GEOMETRY_SOURCE,
                    "roadway_mesh_hash": roadway_mesh_hash,
                }
            )
            if is_internal:
                stats["internal_lane_sample_count"] += 1
        sampled_lane_ids.add(lane_id)
        meta_rows.append(
            {
                "edge_id": edge_id,
                "lane_id": lane_id,
                "lane_index": lane_index,
                "length_m": f"{length_m:.3f}",
                "speed_mps": f"{float(lane_attrib.get('speed') or planned.speed_mps):.3f}",
                "edge_type": edge_type,
                "road_weight": f"{planned.priority / 10.0:.3f}",
                "source_road_id": planned.source_road_id,
                "direction_role": direction_role,
                "physical_lane_index": planned.physical_lane_index,
                "lane_width_m": f"{lane_width_m:.3f}",
                "source_edge_id": source_edge_id,
                "split_part": split_part,
                "coordinate_route": ROAD_DERIVED_COORDINATE_ROUTE,
                "geometry_authority": FORMAL_GEOMETRY_AUTHORITY,
                "lane_geometry_source": FORMAL_LANE_GEOMETRY_SOURCE,
                "roadway_mesh_hash": roadway_mesh_hash,
            }
        )
    via_lanes = set(connections_by_via)
    missing_via_lanes = sorted(via_lanes - sampled_lane_ids)
    if missing_via_lanes:
        raise RoadDerivedNetworkError(
            f"Internal via lanes are missing from lane_center_samples: {missing_via_lanes[:10]}"
        )
    stats["via_lanes_in_connections"] = len(via_lanes)
    stats["via_lanes_missing_from_samples"] = 0
    return center_rows, meta_rows, stats


def _connection_rows(net_xml: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _event, elem in ET.iterparse(net_xml, events=("end",)):
        if elem.tag == "connection":
            rows.append(
                {
                    "from_edge": str(elem.attrib.get("from") or ""),
                    "from_lane": int(elem.attrib.get("fromLane") or 0),
                    "to_edge": str(elem.attrib.get("to") or ""),
                    "to_lane": int(elem.attrib.get("toLane") or 0),
                    "via_lane": str(elem.attrib.get("via") or ""),
                }
            )
        elem.clear()
    return rows


def _road_center_rows(
    roads: Sequence[RoadFeature],
    road_alignments: dict[int, RoadwayRoadAlignment],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for road in roads:
        alignment = road_alignments.get(road.road_id)
        centerline_xyz = alignment.visual_centerline_xyz if alignment is not None else road.centerline_xyz
        length_m = _path_length(centerline_xyz)
        sample_count = max(1, int(math.floor(length_m / LANE_SAMPLE_INTERVAL_M)))
        s_values = [float(index) * LANE_SAMPLE_INTERVAL_M for index in range(sample_count + 1)]
        if not s_values or abs(s_values[-1] - length_m) > 1e-6:
            s_values.append(length_m)
        for s_m in s_values:
            x_m, y_m, z_m, yaw_deg = _point_at_s(centerline_xyz, s_m)
            rows.append(
                {
                    "road_id": road.road_id,
                    "edge_id": f"cg_edge_{road.road_id}",
                    "s_m": f"{s_m:.3f}",
                    "x_m": f"{x_m:.3f}",
                    "y_m": f"{y_m:.3f}",
                    "z_m": f"{z_m:.3f}",
                    "yaw_deg": f"{yaw_deg:.3f}",
                    "width_m": f"{road.width_m:.3f}",
                    "lanes": road.lanes,
                    "direction": road.direction,
                    "isolation": road.isolation,
                    "func_level": road.func_level,
                    "manage_level": road.manage_level,
                    "coordinate_route": ROAD_DERIVED_COORDINATE_ROUTE,
                    "geometry_authority": FORMAL_GEOMETRY_AUTHORITY,
                    "lane_geometry_source": FORMAL_LANE_GEOMETRY_SOURCE,
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RoadDerivedNetworkError(f"Refusing to overwrite existing generated CSV without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_traffic_bundle(
    *,
    traffic_bundle_dir: Path,
    net_xml: Path,
    roads: Sequence[RoadFeature],
    road_alignments: dict[int, RoadwayRoadAlignment],
    planned_edges: Sequence[PlannedEdge],
    roadway_mesh: RoadwayMesh,
    overwrite: bool,
) -> dict[str, Any]:
    planned_by_edge = {edge.edge_id: edge for edge in planned_edges}
    connection_rows = _connection_rows(net_xml)
    center_rows, meta_rows, lane_stats = _sample_net_lane_rows(
        net_xml=net_xml,
        planned_by_edge=planned_by_edge,
        connection_rows=connection_rows,
        roadway_mesh_hash=roadway_mesh.mesh_hash,
    )
    roadway_sample_audit = audit_lane_samples_against_roadway(center_rows, roadway_mesh)
    road_rows = _road_center_rows(roads, road_alignments)
    _write_csv(
        traffic_bundle_dir / "lane_center_samples.csv",
        center_rows,
        (
            "edge_id",
            "lane_id",
            "lane_index",
            "s_m",
            "x_m",
            "y_m",
            "z_m",
            "yaw_deg",
            "source_road_id",
            "direction_role",
            "physical_lane_index",
            "lane_width_m",
            "coordinate_route",
            "geometry_authority",
            "lane_geometry_source",
            "roadway_mesh_hash",
        ),
        overwrite=overwrite,
    )
    _write_csv(
        traffic_bundle_dir / "lane_meta.csv",
        meta_rows,
        (
            "edge_id",
            "lane_id",
            "lane_index",
            "length_m",
            "speed_mps",
            "edge_type",
            "road_weight",
            "source_road_id",
            "direction_role",
            "physical_lane_index",
            "lane_width_m",
            "source_edge_id",
            "split_part",
            "coordinate_route",
            "geometry_authority",
            "lane_geometry_source",
            "roadway_mesh_hash",
        ),
        overwrite=overwrite,
    )
    _write_csv(
        traffic_bundle_dir / "lane_connections.csv",
        connection_rows,
        ("from_edge", "from_lane", "to_edge", "to_lane", "via_lane"),
        overwrite=overwrite,
    )
    _write_csv(
        traffic_bundle_dir / "road_center_samples.csv",
        road_rows,
        (
            "road_id",
            "edge_id",
            "s_m",
            "x_m",
            "y_m",
            "z_m",
            "yaw_deg",
            "width_m",
            "lanes",
            "direction",
            "isolation",
            "func_level",
            "manage_level",
            "coordinate_route",
            "geometry_authority",
            "lane_geometry_source",
        ),
        overwrite=overwrite,
    )
    return {
        "lane_center_sample_count": len(center_rows),
        "lane_meta_count": len(meta_rows),
        "lane_connection_count": len(connection_rows),
        "road_center_sample_count": len(road_rows),
        "roadway_sample_audit": roadway_sample_audit,
        **lane_stats,
    }


def _write_transform_json(
    *,
    traffic_bundle_dir: Path,
    road_geojson_path: Path,
    bounds_geojson_path: Path,
    fit: GeoJsonBundleFit,
    overwrite: bool,
) -> Path:
    transform_json = traffic_bundle_dir / "road_geojson_transform.json"
    if transform_json.exists() and not overwrite:
        raise RoadDerivedNetworkError(f"Refusing to overwrite existing transform without --overwrite: {transform_json}")
    _write_json(
        transform_json,
        {
            "schema_name": "road_geojson_transform",
            "schema_version": "v1",
            "coordinate_route": "geojson_web_mercator_local_to_ue_truth_xy",
            "road_geojson": str(road_geojson_path),
            "bounds_geojson": str(bounds_geojson_path),
            "matrix": fit.matrix,
            "mean_error_m": round(fit.mean_error_m, 9),
            "max_error_m": round(fit.max_error_m, 9),
            "pair_count": int(fit.pair_count),
            "max_allowed_fit_error_m": 0.05,
        },
    )
    return transform_json


def _write_bundle_manifest(
    *,
    traffic_bundle_dir: Path,
    network_manifest: dict[str, Any],
    overwrite: bool,
) -> None:
    path = traffic_bundle_dir / "traffic_bundle_manifest.json"
    if path.exists() and not overwrite:
        raise RoadDerivedNetworkError(f"Refusing to overwrite existing traffic bundle manifest without --overwrite: {path}")
    _write_json(
        path,
        {
            "schema_name": "traffic_bundle_manifest",
            "schema_version": "v2",
            "map_id": "donghu_road_topo",
            "coordinate_route": ROAD_DERIVED_COORDINATE_ROUTE,
            "geometry_authority": FORMAL_GEOMETRY_AUTHORITY,
            "lane_geometry_source": FORMAL_LANE_GEOMETRY_SOURCE,
            "roadway_mesh_hash": network_manifest["roadway_mesh"]["mesh_hash"],
            "drivable_component_policy": network_manifest["roadway_mesh"]["drivable_component_policy"],
            "drivable_component_names": network_manifest["roadway_mesh"]["drivable_component_names"],
            "drivable_material_roles": network_manifest["roadway_mesh"]["drivable_material_roles"],
            "lane_separator_material_roles": network_manifest["roadway_mesh"].get("lane_separator_material_roles", []),
            "lane_separator_source_mesh_ids": network_manifest["roadway_mesh"].get("lane_separator_source_mesh_ids", []),
            "rejected_drivable_role_counts": network_manifest["roadway_mesh"]["rejected_drivable_role_counts"],
            "ue_level": FORMAL_UE_LEVEL,
            "alignment_report": network_manifest["roadway_alignment"]["alignment_report"],
            "sumo_net_xml": network_manifest["outputs"]["net_xml"],
            "road_derived_network_manifest": network_manifest["outputs"]["manifest"],
            "lane_center_samples": str(traffic_bundle_dir / "lane_center_samples.csv"),
            "lane_meta": str(traffic_bundle_dir / "lane_meta.csv"),
            "lane_connections": str(traffic_bundle_dir / "lane_connections.csv"),
            "road_center_samples": str(traffic_bundle_dir / "road_center_samples.csv"),
            "road_geojson_transform": str(traffic_bundle_dir / "road_geojson_transform.json"),
            "stats": network_manifest["traffic_bundle"],
        },
    )


def _disconnected_lane_group_discards(traffic_bundle_dir: Path) -> tuple[list[int], list[dict[str, Any]]]:
    lane_meta_path = traffic_bundle_dir / "lane_meta.csv"
    lane_connections_path = traffic_bundle_dir / "lane_connections.csv"
    if not lane_meta_path.exists() or not lane_connections_path.exists():
        return [], []

    connected_edges: set[str] = set()
    with lane_connections_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            for key in ("from_edge", "to_edge"):
                edge_id = str(row.get(key) or "")
                if edge_id and not edge_id.startswith(":"):
                    connected_edges.add(edge_id)

    rows_by_road: dict[int, list[dict[str, str]]] = {}
    with lane_meta_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("edge_type") or "") != "road_geojson_physical_lane":
                continue
            try:
                road_id = int(float(row.get("source_road_id") or -1))
            except (TypeError, ValueError):
                continue
            if road_id < 0:
                continue
            rows_by_road.setdefault(road_id, []).append(row)

    discarded_road_ids: list[int] = []
    examples: list[dict[str, Any]] = []
    for road_id, rows in sorted(rows_by_road.items()):
        role_to_edges: dict[str, set[str]] = {}
        role_to_lanes: dict[str, list[int]] = {}
        for row in rows:
            role = str(row.get("direction_role") or "")
            if role not in ("f", "r"):
                continue
            edge_id = str(row.get("edge_id") or "")
            if not edge_id:
                continue
            role_to_edges.setdefault(role, set()).add(edge_id)
            try:
                lane_index = int(float(row.get("physical_lane_index") or -1))
            except (TypeError, ValueError):
                lane_index = -1
            if lane_index >= 0:
                role_to_lanes.setdefault(role, []).append(lane_index)
        if set(role_to_edges) != {"f", "r"}:
            continue
        disconnected_roles = [
            role for role, edges in sorted(role_to_edges.items()) if edges and not (edges & connected_edges)
        ]
        if not disconnected_roles:
            continue
        discarded_road_ids.append(road_id)
        examples.append(
            {
                "road_id": road_id,
                "disconnected_roles": disconnected_roles,
                "direction_lane_indices": {
                    role: sorted(set(lanes)) for role, lanes in sorted(role_to_lanes.items())
                },
                "edges": {role: sorted(edges) for role, edges in sorted(role_to_edges.items())},
            }
        )
    return discarded_road_ids, examples


def _filter_alignment_report(report: dict[str, Any], active_road_ids: set[int]) -> dict[str, Any]:
    filtered = dict(report)
    rows = [dict(row) for row in report.get("roads", []) if int(row.get("road_id") or -1) in active_road_ids]
    sample_count = sum(int(row.get("samples") or 0) for row in rows)
    fallback_sample_count = sum(int(row.get("fallback_samples") or 0) for row in rows)
    filtered["roads"] = rows
    filtered["road_count"] = len(rows)
    filtered["sample_count"] = sample_count
    filtered["fallback_sample_count"] = fallback_sample_count
    filtered["fallback_ratio"] = round(float(fallback_sample_count) / float(sample_count), 6) if sample_count else 1.0
    filtered["failing_roads"] = [
        dict(row)
        for row in rows
        if int(row.get("fallback_samples") or 0) > 0
        or int(row.get("lane_shape_count") or 0) != int(row.get("expected_lane_count") or 0)
    ]
    filtered["separator_unresolved_roads"] = [
        dict(row)
        for row in rows
        if bool(row.get("separator_required")) and str(row.get("separator_resolution_status") or "") != "valid_yellow_separator"
    ]
    return filtered


def build_road_derived_network(
    *,
    map_package: Path = DEFAULT_MAP_PACKAGE,
    netconvert: Path | None = None,
    roadway_mesh_manifest: Path = DEFAULT_ROADWAY_MESH_MANIFEST,
    source_dir: Path = DEFAULT_ROAD_DERIVED_SOURCE_DIR,
    net_xml: Path = DEFAULT_ROAD_DERIVED_NET_XML,
    traffic_bundle_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    road_geojson_path, bounds_geojson_path, package_bundle_dir, _plugin_source_dir = _map_sources(map_package)
    traffic_bundle_dir = traffic_bundle_dir or package_bundle_dir
    netconvert = Path(netconvert) if netconvert is not None else sumo_executable_from_environment("netconvert")
    for path in (road_geojson_path, bounds_geojson_path, netconvert, roadway_mesh_manifest):
        if not path.exists():
            raise RoadDerivedNetworkError(f"Required input does not exist: {path}")
    fit = _load_required_fit(
        traffic_bundle_dir=traffic_bundle_dir,
    )
    if fit.max_error_m > 0.05:
        raise SpatialValidationError(
            f"Road GeoJSON transform is too imprecise: max_error={fit.max_error_m:.6f}m > 0.050m"
        )
    all_roads = _load_roads(road_geojson_path, bounds_geojson_path, fit)
    roadway_mesh = load_roadway_mesh(roadway_mesh_manifest)
    node_xml = source_dir / "road_derived.nod.xml"
    edge_xml = source_dir / "road_derived.edg.xml"
    discarded_fragmented_road_ids: set[int] = set()
    discard_history: list[dict[str, Any]] = []
    final_iteration = 0
    roads: list[RoadFeature] = []
    road_alignments: dict[int, RoadwayRoadAlignment] = {}
    alignment_report: dict[str, Any] = {}
    nodes: dict[str, tuple[float, float]] = {}
    planned_edges: list[PlannedEdge] = []
    graph_stats: dict[str, Any] = {}
    tls_nodes: set[str] = set()
    netconvert_cmd: list[str] = []
    lane_shape_postprocess: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    bundle_stats: dict[str, Any] = {}

    all_road_alignments, base_alignment_report = align_roads_to_roadway_mesh(
        all_roads,
        roadway_mesh,
        sample_interval_m=ALIGNMENT_SAMPLE_INTERVAL_M,
    )
    alignment_fragmented_ids = sorted(
        int(row["road_id"])
        for row in base_alignment_report.get("roads", [])
        if int(row.get("fallback_samples") or 0) > 0
        or int(row.get("lane_shape_count") or 0) != int(row.get("expected_lane_count") or 0)
    )
    if alignment_fragmented_ids:
        discarded_fragmented_road_ids.update(alignment_fragmented_ids)
        discard_history.append(
            {
                "iteration": 0,
                "reason": "roadway_cross_section_missing",
                "source_road_ids": alignment_fragmented_ids,
            }
        )

    for iteration in range(1, 17):
        final_iteration = iteration
        roads = [road for road in all_roads if road.road_id not in discarded_fragmented_road_ids]
        if not roads:
            raise RoadDerivedNetworkError("All road.geojson roads were discarded while removing fragmented RoadWay material segments.")

        active_road_ids = {int(road.road_id) for road in roads}
        road_alignments = {
            road_id: alignment
            for road_id, alignment in all_road_alignments.items()
            if road_id in active_road_ids
        }
        alignment_report = _filter_alignment_report(base_alignment_report, active_road_ids)
        unresolved_separator_rows = list(alignment_report.get("separator_unresolved_roads") or [])
        if unresolved_separator_rows:
            unresolved_separator_road_ids = sorted(
                {
                    int(row.get("road_id"))
                    for row in unresolved_separator_rows
                    if row.get("road_id") is not None
                }
            )
            examples = [
                {
                    "road_id": row.get("road_id"),
                    "direction": row.get("direction"),
                    "isolation": row.get("isolation"),
                    "lanes": row.get("expected_lane_count"),
                    "status": row.get("separator_resolution_status"),
                    "coverage": row.get("yellow_separator_coverage_main_body"),
                    "offset_median": row.get("yellow_separator_lateral_offset_median"),
                    "offset_p95_error": row.get("yellow_separator_lateral_offset_p95_error"),
                }
                for row in unresolved_separator_rows[:12]
            ]
            discard_history.append(
                {
                    "iteration": iteration,
                    "reason": "required_lane_separator_unresolved",
                    "source_road_ids": unresolved_separator_road_ids,
                    "policy": "do not fall back to all-forward direction for isolation=2 lanes>=2 direction!=2 roads",
                    "examples": examples,
                }
            )
            discarded_fragmented_road_ids.update(unresolved_separator_road_ids)
            continue

        nodes, planned_edges, graph_stats = _build_nodes_and_edges(roads, road_alignments, roadway_mesh)
        tls_nodes = _traffic_light_nodes(roads, planned_edges)
        _write_plain_xml(
            node_xml=node_xml,
            edge_xml=edge_xml,
            nodes=nodes,
            edges=planned_edges,
            tls_nodes=tls_nodes,
            overwrite=True,
        )
        netconvert_cmd = _run_netconvert(
            netconvert=netconvert,
            node_xml=node_xml,
            edge_xml=edge_xml,
            net_xml=net_xml,
            overwrite=True,
        )
        lane_shape_postprocess = postprocess_net_lane_shapes_to_roadway(net_xml, roadway_mesh)
        external_snap_fragmented_ids = sorted(
            int(road_id)
            for road_id in lane_shape_postprocess.get("external_snap_discard_source_road_ids", [])
        )
        new_external_snap_fragmented_ids = [
            road_id
            for road_id in external_snap_fragmented_ids
            if road_id not in discarded_fragmented_road_ids
        ]
        if new_external_snap_fragmented_ids:
            discard_history.append(
                {
                    "iteration": iteration,
                    "reason": "external_lane_shape_outside_drivable_material_after_netconvert",
                    "source_road_ids": new_external_snap_fragmented_ids,
                    "external_snap_discard_threshold_m": lane_shape_postprocess.get("external_snap_discard_threshold_m"),
                    "max_external_snap_distance_m": lane_shape_postprocess.get("max_external_snap_distance_m"),
                    "snap_examples": lane_shape_postprocess.get("snap_examples", [])[:6],
                }
            )
            discarded_fragmented_road_ids.update(new_external_snap_fragmented_ids)
            continue
        external_heading_fragmented_ids = sorted(
            int(road_id)
            for road_id in lane_shape_postprocess.get("external_fragment_discard_source_road_ids", [])
        )
        new_external_heading_fragmented_ids = [
            road_id
            for road_id in external_heading_fragmented_ids
            if road_id not in discarded_fragmented_road_ids
        ]
        if new_external_heading_fragmented_ids:
            discard_history.append(
                {
                    "iteration": iteration,
                    "reason": "external_lane_shape_abrupt_heading_change_after_netconvert",
                    "source_road_ids": new_external_heading_fragmented_ids,
                    "heading_delta_threshold_deg": lane_shape_postprocess.get(
                        "external_fragment_heading_delta_threshold_deg"
                    ),
                    "raw_heading_delta_threshold_deg": lane_shape_postprocess.get(
                        "external_fragment_raw_heading_delta_threshold_deg"
                    ),
                    "simplify_tolerance_m": lane_shape_postprocess.get(
                        "external_fragment_simplify_tolerance_m"
                    ),
                    "fragment_examples": lane_shape_postprocess.get("fragment_examples", [])[:6],
                }
            )
            discarded_fragmented_road_ids.update(new_external_heading_fragmented_ids)
            continue
        audit = _audit_net(net_xml, (edge.edge_id for edge in planned_edges))
        bundle_stats = _write_traffic_bundle(
            traffic_bundle_dir=traffic_bundle_dir,
            net_xml=net_xml,
            roads=roads,
            road_alignments=road_alignments,
            planned_edges=planned_edges,
            roadway_mesh=roadway_mesh,
            overwrite=True,
        )
        disconnected_lane_group_ids, disconnected_lane_group_examples = _disconnected_lane_group_discards(
            traffic_bundle_dir
        )
        new_disconnected_lane_group_ids = [
            road_id
            for road_id in disconnected_lane_group_ids
            if road_id not in discarded_fragmented_road_ids
        ]
        if new_disconnected_lane_group_ids:
            discard_history.append(
                {
                    "iteration": iteration,
                    "reason": "lane_group_not_connected_after_netconvert",
                    "source_road_ids": new_disconnected_lane_group_ids,
                    "policy": "f/r lane groups must participate in lane-to-lane connections; disconnected groups are audited/discarded instead of exporting unusable directed lanes",
                    "examples": disconnected_lane_group_examples[:12],
                }
            )
            discarded_fragmented_road_ids.update(new_disconnected_lane_group_ids)
            continue
        roadway_sample_audit = dict(bundle_stats.get("roadway_sample_audit") or {})
        outside_count = int(roadway_sample_audit.get("outside_roadway_count") or 0)
        if outside_count <= 0:
            break
        outside_road_ids = [
            int(road_id)
            for road_id in roadway_sample_audit.get("outside_source_road_ids") or []
            if int(road_id) not in discarded_fragmented_road_ids
        ]
        if not outside_road_ids:
            raise RoadDerivedNetworkError(
                "RoadWay material fragmentation remains but no source_road_id could be identified: "
                f"{roadway_sample_audit.get('outside_examples')}"
            )
        discarded_fragmented_road_ids.update(outside_road_ids)
        discard_history.append(
            {
                "iteration": iteration,
                "reason": "lane_sample_outside_drivable_material",
                "source_road_ids": sorted(outside_road_ids),
                "outside_examples": roadway_sample_audit.get("outside_examples"),
            }
        )
    else:
        raise RoadDerivedNetworkError(
            f"RoadWay material-fragmented road discard did not converge after {final_iteration} iterations. "
            f"discarded={sorted(discarded_fragmented_road_ids)}"
        )

    alignment_report["discard_policy"] = {
        "reason": "source roads are removed when UE drivable material is fragmented, required lane separators are unresolved, generated lane samples leave the drivable RoadWay/road-marking union, or f/r lane groups cannot connect in SUMO",
        "discarded_source_road_ids": sorted(discarded_fragmented_road_ids),
        "discard_history": discard_history,
        "iteration_count": final_iteration,
    }
    alignment_report_path = write_alignment_report(roadway_mesh_manifest.with_name("road_alignment_report.json"), alignment_report)
    transform_json = _write_transform_json(
        traffic_bundle_dir=traffic_bundle_dir,
        road_geojson_path=road_geojson_path,
        bounds_geojson_path=bounds_geojson_path,
        fit=fit,
        overwrite=overwrite,
    )
    manifest_path = source_dir / "road_derived_network_manifest.json"
    manifest = {
        "schema_name": "road_derived_sumo_network_manifest",
        "schema_version": "v1",
        "map_id": "donghu_road_topo",
        "geometry_authority": FORMAL_GEOMETRY_AUTHORITY,
        "lane_geometry_source": FORMAL_LANE_GEOMETRY_SOURCE,
        "coordinate_route": ROAD_DERIVED_COORDINATE_ROUTE,
        "ue_level": FORMAL_UE_LEVEL,
        "roadway_mesh": {
            "manifest": str(roadway_mesh_manifest),
            "mesh_hash": roadway_mesh.mesh_hash,
            "vertices_csv": str(roadway_mesh.vertices_csv),
            "triangles_csv": str(roadway_mesh.triangles_csv),
            "drivable_component_policy": "only BP_CityBaseGenerator Road component contributes vehicle drivable geometry; Ground/Terrain/Water are excluded even if old exports label them RoadWay",
            "lane_top_surface_policy": "generated lanes are audited against the top Road-component UE mesh material inside a 2.0m lane-center footprint; lanes are discarded before SUMO edge export only when SideWalk occupies enough footprint stations to indicate the lane itself is on sidewalk, while adjacent sidewalk and painted road markings such as WhiteLine/YellowLine are accepted",
            "drivable_component_names": list(roadway_mesh.drivable_component_names),
            "drivable_material_roles": sorted(DRIVABLE_MATERIAL_ROLES),
            "lane_separator_material_roles": sorted(LANE_SEPARATOR_MATERIAL_ROLES),
            "lane_separator_source_mesh_ids": list(roadway_mesh.yellow_separator_source_mesh_ids),
            "rejected_drivable_role_counts": roadway_mesh.rejected_drivable_role_counts,
        },
        "roadway_alignment": {
            "alignment_report": str(alignment_report_path),
            "sample_interval_m": ALIGNMENT_SAMPLE_INTERVAL_M,
            "fallback_sample_count": alignment_report["fallback_sample_count"],
            "fallback_ratio": alignment_report["fallback_ratio"],
            "lane_shape_postprocess": lane_shape_postprocess,
            "discard_policy": alignment_report["discard_policy"],
        },
        "direction_policy": {
            "direction_1": "road-level default_flow is GeoJSON point order; only used when no stronger valid lane-level separator is available",
            "direction_2": "road-level default_flow is bidirectional; valid YellowLine separators refine lane groups when present but are not required for bidirectional default",
            "direction_4": "road-level default_flow is reverse GeoJSON point order; only used when no stronger valid lane-level separator is available",
            "valid_yellow_separator": "lane-level direction authority only after corridor projection, lateral offset, road main-body coverage, lateral stability, road-end buffer, and ambiguity checks pass",
            "separator_required": "isolation=2 with lanes>=2 and direction!=2 requires valid separator resolution; unresolved roads are explicitly audited/discarded instead of silently exporting all lanes as direction default",
        },
        "lane_policy": "one physical lane centerline per directed SUMO edge unless generated geometry is explicitly discarded by a documented geometry guard; abrupt Z-layer jumps and SideWalk top-surface samples are lane-level discards before SUMO edge export; painted road markings are allowed as top surface samples; valid YellowLine splits use road.width/road.lanes canonical lane offsets rather than guessing lane groups from RoadWay cross-section widths",
        "traffic_light_policy": {
            "source": "deterministic_synthetic_from_road_geojson",
            "rule": "node has at least two incoming motorized approaches, directed physical-lane graph degree >= 3, and an incident road has func_level <= 2 or manage_level <= 3",
            "candidate_node_count": int(audit.get("junction_traffic_light_count", len(tls_nodes))),
            "requested_candidate_node_count": len(tls_nodes),
            "netconvert_removed_candidate_count": max(
                0,
                len(tls_nodes) - int(audit.get("junction_traffic_light_count", len(tls_nodes))),
            ),
        },
        "pedestrian_policy": {
            "crossings_guess": False,
            "sidewalks_guess": False,
            "walkingareas": False,
            "pedestrian_routes_source": "existing pedestrian/semantic generator, not SUMO road net",
        },
        "connection_policy": {
            "source": "SUMO netconvert automatic connections/right-of-way/TLS phases",
            "turnarounds": False,
            "internal_links": True,
            "reason": "internal lane geometry is postprocessed against the UE RoadWay mesh so SUMO turns stay on the visual drivable surface",
        },
        "transform": {
            "road_geojson_transform": str(transform_json),
            "mean_error_m": round(fit.mean_error_m, 9),
            "max_error_m": round(fit.max_error_m, 9),
            "pair_count": int(fit.pair_count),
        },
        "graph": graph_stats,
        "net_audit": audit,
        "traffic_bundle": bundle_stats,
        "netconvert": {
            "binary": str(netconvert),
            "argv": netconvert_cmd,
        },
        "outputs": {
            "node_xml": str(node_xml),
            "edge_xml": str(edge_xml),
            "net_xml": str(net_xml),
            "manifest": str(manifest_path),
            "traffic_bundle_dir": str(traffic_bundle_dir),
        },
        "invalidated_legacy_outputs": [
            "legacy SUMO output generated from the old net",
            "Dataset/render_ready_episodes generated from the old SUMO net",
            "Dataset/render_ready_episodes_capture_filtered generated from the old SUMO net",
        ],
    }
    _write_json(manifest_path, manifest)
    _write_bundle_manifest(traffic_bundle_dir=traffic_bundle_dir, network_manifest=manifest, overwrite=overwrite)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build road.geojson-derived SUMO net and traffic bundle.")
    parser.add_argument("--map-package", type=Path, default=DEFAULT_MAP_PACKAGE)
    parser.add_argument(
        "--netconvert",
        type=Path,
        default=None,
        help="netconvert executable; defaults to $SUMO_HOME/bin/netconvert",
    )
    parser.add_argument("--roadway-mesh-manifest", type=Path, default=DEFAULT_ROADWAY_MESH_MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_ROAD_DERIVED_SOURCE_DIR)
    parser.add_argument("--net-xml", type=Path, default=DEFAULT_ROAD_DERIVED_NET_XML)
    parser.add_argument("--traffic-bundle-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_road_derived_network(
        map_package=args.map_package,
        netconvert=args.netconvert,
        roadway_mesh_manifest=args.roadway_mesh_manifest,
        source_dir=args.source_dir,
        net_xml=args.net_xml,
        traffic_bundle_dir=args.traffic_bundle_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
