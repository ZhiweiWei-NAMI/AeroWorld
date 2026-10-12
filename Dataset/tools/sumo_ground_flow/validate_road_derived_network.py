"""Audit the road.geojson-derived SUMO net, bundle, and optional traffic truth."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep

try:
    from roadway_mesh_alignment import (  # type: ignore
        FORMAL_GEOMETRY_AUTHORITY,
        FORMAL_LANE_GEOMETRY_SOURCE,
        FORMAL_UE_LEVEL,
        load_roadway_mesh,
    )
except ModuleNotFoundError:  # pragma: no cover
    from Dataset.tools.sumo_ground_flow.roadway_mesh_alignment import (  # type: ignore
        FORMAL_GEOMETRY_AUTHORITY,
        FORMAL_LANE_GEOMETRY_SOURCE,
        FORMAL_UE_LEVEL,
        load_roadway_mesh,
    )


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    ROOT
    / "Plugins"
    / "SumoImporter"
    / "Maps"
    / "donghu_road_topo"
    / "source"
    / "road_derived"
    / "road_derived_network_manifest.json"
)
DEFAULT_TRAFFIC_BUNDLE_DIR = ROOT / "Config" / "LowAltitude" / "Maps" / "donghu_road_topo" / "traffic_bundle"
DEFAULT_ROAD_SURFACE_TOLERANCE_M = 0.75
DEFAULT_LANE_DISTANCE_TOLERANCE_M = 1.25
ROAD_SURFACE_NUMERIC_EPSILON_M = 0.05
FORBIDDEN_DIRECTION_AUTHORITY_TOKENS = (
    "white_arrow",
    "white-arrow",
    "white arrow",
    "whitelinearrow",
    "arrow_direction_override",
    "manual_direction",
)
YELLOW_SPLIT_POLICIES = frozenset(
    (
        "yellow_separator_visual_width_split",
        "valid_yellow_separator_canonical_lane_split",
    )
)
YELLOW_DIRECTION_AUTHORITIES = frozenset(
    (
        "yellow_separator_visual_width_split",
        "valid_yellow_separator_lane_level",
    )
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _find_forbidden_direction_tokens(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if any(token in lowered for token in FORBIDDEN_DIRECTION_AUTHORITY_TOKENS):
                findings.append(f"{path}.{key_text}")
            findings.extend(_find_forbidden_direction_tokens(child, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden_direction_tokens(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in FORBIDDEN_DIRECTION_AUTHORITY_TOKENS):
            findings.append(path)
    return findings


def _validate_yellow_split_alignment_report(path: Path) -> list[str]:
    errors: list[str] = []
    report = _read_json(path)
    forbidden = _find_forbidden_direction_tokens(report)
    if forbidden:
        errors.append(f"alignment report contains forbidden white-arrow/manual direction authority fields: {forbidden[:10]}")
    for row in report.get("roads") or []:
        if bool(row.get("separator_required")) and str(row.get("separator_resolution_status") or "") != "valid_yellow_separator":
            errors.append(
                f"road {row.get('road_id')}: separator is required but unresolved "
                f"(status={row.get('separator_resolution_status')}, coverage={row.get('yellow_separator_coverage_main_body')})"
            )
        if str(row.get("lane_layout_policy") or "") not in YELLOW_SPLIT_POLICIES:
            continue
        road_id = row.get("road_id")
        indices = dict(row.get("direction_lane_indices") or {})
        try:
            expected_lane_count = int(row.get("expected_lane_count") or 0)
            f_indices = sorted(int(value) for value in indices.get("f") or [])
            r_indices = sorted(int(value) for value in indices.get("r") or [])
        except (TypeError, ValueError):
            errors.append(f"road {road_id}: invalid yellow split direction_lane_indices={indices}")
            continue
        all_indices = f_indices + r_indices
        if not f_indices or not r_indices:
            errors.append(f"road {road_id}: yellow split must assign both f and r lanes, got {indices}")
            continue
        if sorted(all_indices) != list(range(expected_lane_count)):
            errors.append(
                f"road {road_id}: yellow split lane indices must cover 0..{expected_lane_count - 1}, got {indices}"
            )
            continue
        if max(r_indices) >= min(f_indices):
            errors.append(
                f"road {road_id}: UE YellowLine right-hand rule requires lower/left-of-source lane indices as r "
                f"and higher/right-of-source lane indices as f, got {indices}"
            )
    return errors


def _load_net_lane_shapes(net_xml: Path) -> dict[str, list[tuple[float, float]]]:
    import xml.etree.ElementTree as ET

    lane_shapes: dict[str, list[tuple[float, float]]] = {}
    for _event, elem in ET.iterparse(net_xml, events=("end",)):
        if elem.tag == "lane":
            lane_id = str(elem.attrib.get("id") or "")
            points = _shape_points(elem.attrib.get("shape"))
            if lane_id and points:
                lane_shapes[lane_id] = points
            elem.clear()
    return lane_shapes


def _load_net_edge_shapes(net_xml: Path) -> dict[str, list[tuple[float, float]]]:
    import xml.etree.ElementTree as ET

    edge_shapes: dict[str, list[tuple[float, float]]] = {}
    for _event, elem in ET.iterparse(net_xml, events=("end",)):
        if elem.tag != "edge" or elem.attrib.get("function") == "internal":
            elem.clear()
            continue
        edge_id = str(elem.attrib.get("id") or "")
        lane = elem.find("lane")
        if edge_id and lane is not None:
            points = _shape_points(lane.attrib.get("shape"))
            if len(points) >= 2:
                edge_shapes[edge_id] = points
        elem.clear()
    return edge_shapes


def _load_road_requirements(traffic_bundle_dir: Path) -> dict[int, dict[str, Any]]:
    roads: dict[int, dict[str, Any]] = {}
    with (traffic_bundle_dir / "road_center_samples.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                road_id = int(float(row["road_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            if road_id < 0:
                continue
            entry = roads.setdefault(
                road_id,
                {
                    "road_id": road_id,
                    "lanes": int(float(row.get("lanes") or 1)),
                    "direction": int(float(row.get("direction") or 1)),
                    "isolation": int(float(row.get("isolation") or 0)),
                    "points": [],
                },
            )
            entry["points"].append((float(row["x_m"]), float(row["y_m"])))
    return roads


def _load_normal_lane_meta(traffic_bundle_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (traffic_bundle_dir / "lane_meta.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("edge_type") or "") != "road_geojson_physical_lane":
                continue
            try:
                row["_source_road_id"] = int(float(row["source_road_id"]))
                row["_physical_lane_index"] = int(float(row["physical_lane_index"]))
            except (KeyError, TypeError, ValueError):
                row["_source_road_id"] = -1
                row["_physical_lane_index"] = -1
            rows.append(row)
    return rows


def _allowed_geometry_discards(manifest: dict[str, Any]) -> dict[int, set[int]]:
    allowed: dict[int, set[int]] = defaultdict(set)
    graph = dict(manifest.get("graph") or {})
    for key in (
        "discarded_abrupt_lane_edges",
        "skipped_abrupt_lane_edges",
        "rejected_abrupt_lane_edges",
        "discarded_z_jump_lane_edges",
        "discarded_top_surface_lane_edges",
    ):
        for row in graph.get(key) or []:
            try:
                road_id = int(float(row["source_road_id"]))
                lane_index = int(float(row["physical_lane_index"]))
            except (KeyError, TypeError, ValueError):
                continue
            if road_id >= 0 and lane_index >= 0:
                allowed[road_id].add(lane_index)
    return allowed


def _unit_direction(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    start = points[0]
    end = points[-1]
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return None
    return dx / norm, dy / norm


def _validate_physical_lane_accounting_and_connectivity(
    *,
    manifest: dict[str, Any],
    traffic_bundle_dir: Path,
    net_xml: Path | None,
) -> list[str]:
    errors: list[str] = []
    roads = _load_road_requirements(traffic_bundle_dir)
    normal_rows = _load_normal_lane_meta(traffic_bundle_dir)
    rows_by_road: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in normal_rows:
        rows_by_road[int(row.get("_source_road_id", -1))].append(row)
    allowed_discards = _allowed_geometry_discards(manifest)

    edge_shapes = _load_net_edge_shapes(net_xml) if net_xml is not None and net_xml.exists() else {}
    connection_rows: list[dict[str, Any]] = []
    connections_path = traffic_bundle_dir / "lane_connections.csv"
    if connections_path.exists():
        with connections_path.open(newline="", encoding="utf-8-sig") as handle:
            connection_rows = list(csv.DictReader(handle))
    connected_edges = {
        str(row.get("from_edge") or "")
        for row in connection_rows
        if str(row.get("from_edge") or "") and not str(row.get("from_edge") or "").startswith(":")
    } | {
        str(row.get("to_edge") or "")
        for row in connection_rows
        if str(row.get("to_edge") or "") and not str(row.get("to_edge") or "").startswith(":")
    }

    for road_id, road in sorted(roads.items()):
        expected = set(range(max(1, int(road.get("lanes") or 1))))
        exported = {
            int(row.get("_physical_lane_index", -1))
            for row in rows_by_road.get(road_id, [])
            if int(row.get("_physical_lane_index", -1)) >= 0
        }
        discarded = allowed_discards.get(road_id, set())
        if sorted(exported | discarded) != sorted(expected):
            errors.append(
                f"road {road_id}: physical_lane_index coverage must equal source lanes after explicit geometry discards; "
                f"expected={sorted(expected)} exported={sorted(exported)} discarded={sorted(discarded)}"
            )
        if exported & discarded:
            errors.append(
                f"road {road_id}: physical lanes cannot be both exported and geometry-discarded: {sorted(exported & discarded)}"
            )

        road_dir = _unit_direction(road.get("points") or [])
        if road_dir is not None:
            for row in rows_by_road.get(road_id, []):
                edge_id = str(row.get("edge_id") or "")
                role = str(row.get("direction_role") or "")
                edge_dir = _unit_direction(edge_shapes.get(edge_id) or [])
                if edge_dir is None or role not in ("f", "r"):
                    continue
                dot = edge_dir[0] * road_dir[0] + edge_dir[1] * road_dir[1]
                if role == "f" and dot < 0.20:
                    errors.append(f"road {road_id} edge {edge_id}: forward shape is not aligned to GeoJSON point order (dot={dot:.3f})")
                if role == "r" and dot > -0.20:
                    errors.append(f"road {road_id} edge {edge_id}: reverse shape is not opposite GeoJSON point order (dot={dot:.3f})")

        roles = {
            str(row.get("direction_role") or "")
            for row in rows_by_road.get(road_id, [])
            if str(row.get("direction_role") or "") in ("f", "r")
        }
        if roles == {"f", "r"}:
            for role in ("f", "r"):
                role_edges = {
                    str(row.get("edge_id") or "")
                    for row in rows_by_road.get(road_id, [])
                    if str(row.get("direction_role") or "") == role
                }
                if role_edges and not (role_edges & connected_edges):
                    errors.append(
                        f"road {road_id}: {role} lane group exists but no edge participates in lane-to-lane connections"
                    )
    return errors


def _validate_yellow_split_lane_directions(
    *,
    manifest: dict[str, Any],
    traffic_bundle_dir: Path,
    net_xml: Path,
    margin_m: float = 0.20,
) -> list[str]:
    errors: list[str] = []
    roadway_alignment = dict(manifest.get("roadway_alignment") or {})
    alignment_report_path = Path(str(roadway_alignment.get("alignment_report") or ""))
    if not alignment_report_path.exists() or not net_xml.exists():
        return errors
    alignment_report = _read_json(alignment_report_path)
    yellow_road_ids = {
        int(row.get("road_id"))
        for row in alignment_report.get("roads") or []
        if str(row.get("lane_layout_policy") or "") in YELLOW_SPLIT_POLICIES
        or str(row.get("direction_authority") or "") in YELLOW_DIRECTION_AUTHORITIES
    }
    if not yellow_road_ids:
        return errors
    roadway_mesh_payload = dict(manifest.get("roadway_mesh") or {})
    roadway_manifest = Path(str(roadway_mesh_payload.get("manifest") or ""))
    if not roadway_manifest.exists():
        errors.append(f"cannot validate UE YellowLine lane directions without roadway mesh manifest: {roadway_manifest}")
        return errors
    mesh = load_roadway_mesh(roadway_manifest)
    if mesh.yellowline_union.is_empty:
        errors.append("cannot validate UE YellowLine lane directions because mesh.yellowline_union is empty")
        return errors

    lane_shapes = _load_net_lane_shapes(net_xml)
    checked = 0
    lane_meta_path = traffic_bundle_dir / "lane_meta.csv"
    with lane_meta_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("edge_type") or "") != "road_geojson_physical_lane":
                continue
            try:
                source_road_id = int(float(row["source_road_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            if source_road_id not in yellow_road_ids:
                continue
            lane_id = str(row.get("lane_id") or "")
            points = lane_shapes.get(lane_id)
            if not points or len(points) < 2:
                errors.append(f"yellow-split lane {lane_id}: missing SUMO lane shape")
                continue
            try:
                lane_width_m = float(row.get("lane_width_m") or 3.0)
            except (TypeError, ValueError):
                lane_width_m = 3.0
            max_yellow_distance_m = max(8.0, lane_width_m * 2.5)
            line = LineString(points)
            length_m = float(line.length)
            close_scores: list[float] = []
            for fraction in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                s_m = length_m * fraction
                start = line.interpolate(max(0.0, s_m - 0.5))
                end = line.interpolate(min(length_m, s_m + 0.5))
                dx = float(end.x) - float(start.x)
                dy = float(end.y) - float(start.y)
                norm = math.hypot(dx, dy)
                if norm <= 1e-9:
                    continue
                lane_point = line.interpolate(s_m)
                yellow_point = nearest_points(lane_point, mesh.yellowline_union)[1]
                delta_x = float(yellow_point.x) - float(lane_point.x)
                delta_y = float(yellow_point.y) - float(lane_point.y)
                yellow_distance_m = math.hypot(delta_x, delta_y)
                if yellow_distance_m > max_yellow_distance_m:
                    continue
                ux = dx / norm
                uy = dy / norm
                close_scores.append(delta_x * (-uy) + delta_y * ux)
            if not close_scores:
                continue
            close_scores.sort()
            median_score_m = close_scores[len(close_scores) // 2]
            bad_count = sum(1 for value in close_scores if value >= -abs(float(margin_m)))
            bad_ratio = float(bad_count) / float(len(close_scores))
            if median_score_m >= -abs(float(margin_m)) or bad_ratio > 0.40:
                errors.append(
                    f"yellow-split lane {lane_id}: YellowLine is not on the UE-left side of fragment motion "
                    f"(median_right_score={median_score_m:.3f}m, bad_ratio={bad_ratio:.3f}, "
                    f"close_samples={len(close_scores)}, road={source_road_id}, edge={row.get('edge_id')})"
                )
                continue
            checked += 1
    if checked <= 0:
        errors.append(f"no yellow-split physical lanes were validated from {lane_meta_path}")
    return errors


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc


def _load_lane_samples(traffic_bundle_dir: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    samples: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    with (traffic_bundle_dir / "lane_center_samples.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            edge_id = str(row.get("edge_id") or "")
            if not edge_id:
                continue
            lane_width = float(row.get("lane_width_m") or 0.0)
            samples[edge_id].append((float(row["s_m"]), float(row["x_m"]), float(row["y_m"]), lane_width))
    for rows in samples.values():
        rows.sort(key=lambda item: item[0])
    return samples


def _shape_points(value: str | None) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for raw in str(value or "").split():
        parts = raw.split(",")
        if len(parts) >= 2:
            points.append((float(parts[0]), float(parts[1])))
    return points


def _junction_envelopes(net_xml: Path | None, tolerance_m: float) -> list[Any]:
    if net_xml is None or not net_xml.exists():
        return []
    envelopes: list[Any] = []
    import xml.etree.ElementTree as ET

    for _event, elem in ET.iterparse(net_xml, events=("end",)):
        if elem.tag == "lane":
            continue
        if elem.tag == "edge" and elem.attrib.get("function") == "internal":
            for lane in elem.findall("lane"):
                points = _shape_points(lane.attrib.get("shape"))
                if len(points) >= 2:
                    lane_width = float(lane.attrib.get("width") or 3.0)
                    envelopes.append(
                        LineString(points).buffer(0.5 * lane_width + max(0.0, tolerance_m), cap_style=1, join_style=2)
                    )
        elif elem.tag == "junction" and elem.attrib.get("type") != "internal":
            points = _shape_points(elem.attrib.get("shape"))
            if len(points) >= 3:
                polygon = Polygon(points)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if not polygon.is_empty:
                    envelopes.append(polygon.buffer(max(0.0, tolerance_m), cap_style=1, join_style=2))
            elif points:
                envelopes.append(LineString(points).buffer(max(0.0, tolerance_m), cap_style=1, join_style=2))
        elem.clear()
    return envelopes


def _load_road_surface(
    traffic_bundle_dir: Path,
    tolerance_m: float,
    net_xml: Path | None,
    manifest: dict[str, Any],
):
    roadway_mesh_payload = dict(manifest.get("roadway_mesh") or {})
    roadway_manifest = Path(str(roadway_mesh_payload.get("manifest") or ""))
    if roadway_manifest.exists():
        mesh = load_roadway_mesh(roadway_manifest)
        return mesh.roadway_union.buffer(max(0.0, tolerance_m), cap_style=1, join_style=2), prep(
            mesh.roadway_union.buffer(max(0.0, tolerance_m), cap_style=1, join_style=2)
        )

    roads: dict[str, list[tuple[float, float]]] = defaultdict(list)
    widths: dict[str, float] = {}
    with (traffic_bundle_dir / "road_center_samples.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            road_id = str(row.get("road_id") or row.get("edge_id") or "")
            if not road_id:
                continue
            roads[road_id].append((float(row["x_m"]), float(row["y_m"])))
            widths[road_id] = float(row.get("width_m") or 3.0)
    buffers = []
    for road_id, points in roads.items():
        if len(points) < 2:
            continue
        half_width = 0.5 * max(0.0, widths.get(road_id, 3.0)) + tolerance_m
        buffers.append(LineString(points).buffer(half_width, cap_style=1, join_style=2))
    buffers.extend(_junction_envelopes(net_xml, tolerance_m))
    if not buffers:
        raise RuntimeError(f"No road surface buffers could be built from {traffic_bundle_dir / 'road_center_samples.csv'}")
    surface = unary_union(buffers)
    return surface, prep(surface)


def _nearest_lane_sample(
    rows: Sequence[tuple[float, float, float, float]],
    lane_position_m: float,
) -> tuple[float, float, float, float]:
    values = [item[0] for item in rows]
    index = bisect.bisect_left(values, lane_position_m)
    candidates = []
    if index < len(rows):
        candidates.append(rows[index])
    if index > 0:
        candidates.append(rows[index - 1])
    if not candidates:
        return rows[0]
    return min(candidates, key=lambda item: abs(item[0] - lane_position_m))


def _validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    forbidden_manifest = _find_forbidden_direction_tokens(manifest)
    if forbidden_manifest:
        errors.append(f"manifest contains forbidden white-arrow/manual direction authority fields: {forbidden_manifest[:10]}")
    direction_policy = dict(manifest.get("direction_policy") or {})
    direction_1 = str(direction_policy.get("direction_1") or "")
    direction_2 = str(direction_policy.get("direction_2") or "")
    direction_4 = str(direction_policy.get("direction_4") or "")
    valid_yellow = str(direction_policy.get("valid_yellow_separator") or "")
    separator_required = str(direction_policy.get("separator_required") or "")
    if "default_flow" not in direction_1 or "GeoJSON point order" not in direction_1:
        errors.append(f"direction_policy.direction_1 must describe road-level default_flow only, got: {direction_1}")
    if "bidirectional" not in direction_2 or "not required" not in direction_2:
        errors.append(f"direction_policy.direction_2 must be bidirectional default and must not require YellowLine, got: {direction_2}")
    if "default_flow" not in direction_4 or "reverse GeoJSON point order" not in direction_4:
        errors.append(f"direction_policy.direction_4 must describe reverse road-level default_flow only, got: {direction_4}")
    if "lane-level direction authority" not in valid_yellow or "coverage" not in valid_yellow or "stability" not in valid_yellow:
        errors.append(f"direction_policy.valid_yellow_separator must define valid separator authority, got: {valid_yellow}")
    if "isolation=2" not in separator_required or "direction!=2" not in separator_required:
        errors.append(f"direction_policy.separator_required must target isolation=2 lanes>=2 direction!=2 roads, got: {separator_required}")
    if manifest.get("coordinate_route") != "sumo_xy_identity_truth_enu_m":
        errors.append(f"coordinate_route must be sumo_xy_identity_truth_enu_m, got {manifest.get('coordinate_route')}")
    if manifest.get("geometry_authority") != FORMAL_GEOMETRY_AUTHORITY:
        errors.append(f"geometry_authority must be {FORMAL_GEOMETRY_AUTHORITY}, got {manifest.get('geometry_authority')}")
    if manifest.get("lane_geometry_source") != FORMAL_LANE_GEOMETRY_SOURCE:
        errors.append(f"lane_geometry_source must be {FORMAL_LANE_GEOMETRY_SOURCE}, got {manifest.get('lane_geometry_source')}")
    if manifest.get("ue_level") != FORMAL_UE_LEVEL:
        errors.append(f"ue_level must be {FORMAL_UE_LEVEL}, got {manifest.get('ue_level')}")
    roadway_mesh = dict(manifest.get("roadway_mesh") or {})
    if not str(roadway_mesh.get("mesh_hash") or ""):
        errors.append("roadway_mesh.mesh_hash is required")
    roadway_alignment = dict(manifest.get("roadway_alignment") or {})
    alignment_report = Path(str(roadway_alignment.get("alignment_report") or ""))
    if not alignment_report.exists():
        errors.append(f"roadway_alignment.alignment_report does not exist: {alignment_report}")
    else:
        errors.extend(_validate_yellow_split_alignment_report(alignment_report))
    audit = dict(manifest.get("net_audit") or {})
    if audit.get("missing_expected_edges"):
        errors.append(f"net audit has missing expected edges: {audit.get('missing_expected_edges')[:10]}")
    for key in ("crossing_count", "walkingarea_count", "pedestrian_allow_lane_count"):
        if int(audit.get(key) or 0) != 0:
            errors.append(f"{key} must be 0, got {audit.get(key)}")
    expected_tls = int(dict(manifest.get("traffic_light_policy") or {}).get("candidate_node_count") or 0)
    actual_tls = int(audit.get("tl_logic_count") or 0)
    if expected_tls and actual_tls != expected_tls:
        errors.append(f"TLS count mismatch: candidate_node_count={expected_tls}, tl_logic_count={actual_tls}")
    graph = dict(manifest.get("graph") or {})
    expected_edges = int(graph.get("expected_output_edge_count") or 0)
    normal_edges = int(audit.get("normal_edge_count") or 0)
    if expected_edges and normal_edges != expected_edges:
        errors.append(f"normal edge count mismatch: expected={expected_edges}, net={normal_edges}")
    bundle = dict(manifest.get("traffic_bundle") or {})
    via_lanes = int(bundle.get("via_lanes_in_connections") or 0)
    internal_lanes = int(bundle.get("internal_lane_count") or 0)
    missing_via = int(bundle.get("via_lanes_missing_from_samples") or 0)
    if via_lanes <= 0:
        errors.append("traffic bundle must include SUMO via/internal lanes, got 0")
    if missing_via != 0:
        errors.append(f"traffic bundle has missing via lane samples: {missing_via}")
    if via_lanes and internal_lanes != via_lanes:
        errors.append(f"internal lane count mismatch: via_lanes={via_lanes}, sampled_internal_lanes={internal_lanes}")
    return errors


def _validate_traffic_truth(
    *,
    output_dir: Path,
    traffic_bundle_dir: Path,
    net_xml: Path | None,
    manifest: dict[str, Any],
    road_surface_tolerance_m: float,
    lane_distance_tolerance_m: float,
) -> dict[str, Any]:
    frames_path = output_dir / "sumo_traffic_frames.jsonl"
    if not frames_path.exists():
        return {"ok": False, "errors": [f"missing {frames_path}"], "vehicle_count": 0}
    road_surface, prepared_road_surface = _load_road_surface(
        traffic_bundle_dir,
        road_surface_tolerance_m,
        net_xml,
        manifest,
    )
    lane_samples = _load_lane_samples(traffic_bundle_dir)
    errors: list[str] = []
    vehicle_count = 0
    internal_edge_vehicle_count = 0
    max_lane_distance_m = 0.0
    for frame in _iter_jsonl(frames_path):
        for vehicle in frame.get("vehicles") or []:
            vehicle_count += 1
            position = vehicle.get("truth_position_enu_m") or []
            if len(position) < 2:
                errors.append(f"vehicle {vehicle.get('vehicle_id')} missing truth_position_enu_m")
                continue
            point = Point(float(position[0]), float(position[1]))
            if not prepared_road_surface.covers(point) and road_surface.distance(point) > ROAD_SURFACE_NUMERIC_EPSILON_M:
                errors.append(
                    f"vehicle {vehicle.get('vehicle_id')} at t={frame.get('sim_time_s')} is outside UE RoadWay surface"
                )
                if len(errors) >= 20:
                    break
            edge_id = str(vehicle.get("sumo_edge_id") or "")
            if edge_id.startswith(":"):
                internal_edge_vehicle_count += 1
            rows = lane_samples.get(edge_id)
            if not rows:
                errors.append(f"vehicle {vehicle.get('vehicle_id')} references unknown lane edge {edge_id}")
                if len(errors) >= 20:
                    break
                continue
            front_position = vehicle.get("truth_front_bumper_enu_m") or position
            lane_s = float(vehicle.get("lane_position_m") or 0.0)
            _sample_s, sx, sy, lane_width = _nearest_lane_sample(rows, lane_s)
            distance_m = math.hypot(float(front_position[0]) - sx, float(front_position[1]) - sy)
            max_lane_distance_m = max(max_lane_distance_m, distance_m)
            tolerance = max(lane_distance_tolerance_m, 0.5 * lane_width + road_surface_tolerance_m)
            if distance_m > tolerance:
                errors.append(
                    f"vehicle {vehicle.get('vehicle_id')} lane distance {distance_m:.3f}m exceeds {tolerance:.3f}m"
                )
                if len(errors) >= 20:
                    break
        if len(errors) >= 20:
            break
    return {
        "ok": not errors,
        "errors": errors,
        "vehicle_count": vehicle_count,
        "internal_edge_vehicle_count": internal_edge_vehicle_count,
        "max_lane_distance_m": round(max_lane_distance_m, 6),
    }


def validate(
    *,
    manifest_path: Path,
    traffic_bundle_dir: Path,
    sumo_output_dir: Path | None = None,
    road_surface_tolerance_m: float = DEFAULT_ROAD_SURFACE_TOLERANCE_M,
    lane_distance_tolerance_m: float = DEFAULT_LANE_DISTANCE_TOLERANCE_M,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    errors = _validate_manifest(manifest)
    net_xml_value = dict(manifest.get("outputs") or {}).get("net_xml")
    net_xml = Path(net_xml_value) if net_xml_value else None
    if net_xml is not None:
        errors.extend(
            _validate_yellow_split_lane_directions(
                manifest=manifest,
                traffic_bundle_dir=traffic_bundle_dir,
                net_xml=net_xml,
            )
        )
        errors.extend(
            _validate_physical_lane_accounting_and_connectivity(
                manifest=manifest,
                traffic_bundle_dir=traffic_bundle_dir,
                net_xml=net_xml,
            )
        )
    result: dict[str, Any] = {
        "ok": not errors,
        "errors": errors,
        "network": {
            "manifest": str(manifest_path),
            "net_xml": net_xml_value,
            "traffic_light_count": dict(manifest.get("net_audit") or {}).get("tl_logic_count"),
            "normal_edge_count": dict(manifest.get("net_audit") or {}).get("normal_edge_count"),
            "crossing_count": dict(manifest.get("net_audit") or {}).get("crossing_count"),
            "walkingarea_count": dict(manifest.get("net_audit") or {}).get("walkingarea_count"),
            "pedestrian_allow_lane_count": dict(manifest.get("net_audit") or {}).get("pedestrian_allow_lane_count"),
        },
    }
    if sumo_output_dir is not None:
        truth = _validate_traffic_truth(
            output_dir=sumo_output_dir,
            traffic_bundle_dir=traffic_bundle_dir,
            net_xml=net_xml,
            manifest=manifest,
            road_surface_tolerance_m=road_surface_tolerance_m,
            lane_distance_tolerance_m=lane_distance_tolerance_m,
        )
        result["traffic_truth"] = truth
        if not truth.get("ok"):
            result["errors"].extend(str(error) for error in truth.get("errors") or [])
            result["ok"] = False
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate road.geojson-derived SUMO network and optional truth output.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--traffic-bundle-dir", type=Path, default=DEFAULT_TRAFFIC_BUNDLE_DIR)
    parser.add_argument("--sumo-output-dir", type=Path, default=None)
    parser.add_argument("--road-surface-tolerance-m", type=float, default=DEFAULT_ROAD_SURFACE_TOLERANCE_M)
    parser.add_argument("--lane-distance-tolerance-m", type=float, default=DEFAULT_LANE_DISTANCE_TOLERANCE_M)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate(
        manifest_path=args.manifest,
        traffic_bundle_dir=args.traffic_bundle_dir,
        sumo_output_dir=args.sumo_output_dir,
        road_surface_tolerance_m=args.road_surface_tolerance_m,
        lane_distance_tolerance_m=args.lane_distance_tolerance_m,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
