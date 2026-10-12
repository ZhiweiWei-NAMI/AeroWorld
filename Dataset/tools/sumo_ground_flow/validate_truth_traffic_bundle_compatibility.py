"""Validate capture truth vehicle lanes against the current SUMO traffic bundle.

This validator checks the render-time contract used by
``episode_render_host.py``: vehicle truth records carry SUMO lane IDs, and the
host resolves vehicle Z from the current ``lane_center_samples.csv``.  The check
is intentionally scoped to that compatibility boundary; it does not validate
SUMO traffic quality, events, UAVs, pedestrians, or UE capture output.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_ROOT = ROOT / "Dataset" / "render_ready_episodes_capture_filtered"
DEFAULT_TRAFFIC_BUNDLE_DIR = ROOT / "Config" / "LowAltitude" / "Maps" / "donghu_road_topo" / "traffic_bundle"
DEFAULT_NET_XML = ROOT / "Plugins" / "SumoImporter" / "Maps" / "donghu_road_topo" / "source" / "road_derived" / "road_derived.net.xml"
DEFAULT_TICK_STRIDE = 5
DEFAULT_MAX_LANE_SNAP_DISTANCE_M = 35.0
DEFAULT_MAX_EXAMPLES_PER_CATEGORY = 5


LANE_SAMPLES: dict[str, list[tuple[float, float, float, float]]] = {}
EDGE_HAS_SAMPLES: set[str] = set()
INTERNAL_CONNECTION_BY_LANE: dict[str, dict[str, Any]] = {}
INTERNAL_CONNECTION_BY_EDGE: dict[str, dict[str, Any]] = {}
NON_INTERNAL_SUCCESSORS: dict[str, set[str]] = {}
MAX_SKIPPED_INTERNAL_CONNECTION_HOPS = 4


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_lane_samples(path: Path) -> tuple[dict[str, list[tuple[float, float, float, float]]], set[str]]:
    lanes: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    edge_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lane_id = str(row.get("lane_id") or "").strip()
            edge_id = str(row.get("edge_id") or "").strip()
            if not lane_id:
                continue
            try:
                sample = (
                    float(row.get("s_m") or 0.0),
                    float(row.get("x_m") or 0.0),
                    float(row.get("y_m") or 0.0),
                    float(row.get("z_m") or 0.0),
                )
            except (TypeError, ValueError):
                continue
            lanes[lane_id].append(sample)
            if edge_id:
                edge_ids.add(edge_id)
    result = {lane_id: sorted(samples, key=lambda item: item[0]) for lane_id, samples in lanes.items()}
    return result, edge_ids


def _load_internal_connections(
    net_xml: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, set[str]]]:
    connections: list[dict[str, str]] = []
    for _event, elem in ET.iterparse(net_xml, events=("end",)):
        if elem.tag != "connection":
            elem.clear()
            continue
        via_lane = str(elem.attrib.get("via") or "").strip()
        via_edge = via_lane.rsplit("_", 1)[0]
        payload = {
            "via_lane": via_lane,
            "via_edge": via_edge,
            "from": str(elem.attrib.get("from") or ""),
            "to": str(elem.attrib.get("to") or ""),
            "fromLane": str(elem.attrib.get("fromLane") or ""),
            "toLane": str(elem.attrib.get("toLane") or ""),
            "dir": str(elem.attrib.get("dir") or ""),
        }
        connections.append(payload)
        elem.clear()

    incoming_by_to: dict[str, list[dict[str, str]]] = defaultdict(list)
    outgoing_by_from: dict[str, list[dict[str, str]]] = defaultdict(list)
    via_by_edge: dict[str, list[dict[str, str]]] = defaultdict(list)
    for connection in connections:
        if connection["to"]:
            incoming_by_to[connection["to"]].append(connection)
        if connection["from"]:
            outgoing_by_from[connection["from"]].append(connection)
        if connection["via_edge"]:
            via_by_edge[connection["via_edge"]].append(connection)

    def upstream_non_internal(edge_id: str, seen: set[str] | None = None) -> set[str]:
        if not _is_internal_edge(edge_id):
            return {edge_id} if edge_id else set()
        seen = set(seen or set())
        if edge_id in seen:
            return set()
        seen.add(edge_id)
        result: set[str] = set()
        for connection in incoming_by_to.get(edge_id, []):
            result.update(upstream_non_internal(connection["from"], seen))
        for connection in via_by_edge.get(edge_id, []):
            result.update(upstream_non_internal(connection["from"], seen))
        return result

    def downstream_non_internal(edge_id: str, seen: set[str] | None = None) -> set[str]:
        if not _is_internal_edge(edge_id):
            return {edge_id} if edge_id else set()
        seen = set(seen or set())
        if edge_id in seen:
            return set()
        seen.add(edge_id)
        result: set[str] = set()
        for connection in outgoing_by_from.get(edge_id, []):
            result.update(downstream_non_internal(connection["to"], seen))
        for connection in via_by_edge.get(edge_id, []):
            result.update(downstream_non_internal(connection["to"], seen))
        return result

    by_lane: dict[str, dict[str, Any]] = {}
    by_edge: dict[str, dict[str, Any]] = {}
    non_internal_successors: dict[str, set[str]] = defaultdict(set)
    for connection in connections:
        via_lane = connection["via_lane"]
        if not via_lane.startswith(":"):
            continue
        via_edge = connection["via_edge"]
        payload: dict[str, Any] = dict(connection)
        payload["terminal_from_edges"] = sorted(upstream_non_internal(connection["from"]))
        payload["terminal_to_edges"] = sorted(downstream_non_internal(connection["to"]))
        for from_edge in payload["terminal_from_edges"]:
            for to_edge in payload["terminal_to_edges"]:
                if from_edge and to_edge and from_edge != to_edge:
                    non_internal_successors[str(from_edge)].add(str(to_edge))
        by_lane[via_lane] = payload
        by_edge.setdefault(via_edge, payload)
    return by_lane, by_edge, {edge: set(values) for edge, values in non_internal_successors.items()}


def _init_worker(lane_center_samples: str, net_xml: str) -> None:
    global LANE_SAMPLES, EDGE_HAS_SAMPLES, INTERNAL_CONNECTION_BY_LANE, INTERNAL_CONNECTION_BY_EDGE, NON_INTERNAL_SUCCESSORS
    LANE_SAMPLES, EDGE_HAS_SAMPLES = _load_lane_samples(Path(lane_center_samples))
    INTERNAL_CONNECTION_BY_LANE, INTERNAL_CONNECTION_BY_EDGE, NON_INTERNAL_SUCCESSORS = _load_internal_connections(
        Path(net_xml)
    )


def _non_internal_reachable(start_edge: str, target_edges: set[str], max_hops: int) -> bool:
    start = str(start_edge or "")
    targets = {str(edge) for edge in target_edges if str(edge)}
    if not start or not targets:
        return False
    if start in targets:
        return True
    frontier = {start}
    seen = {start}
    for _hop in range(max(0, int(max_hops))):
        next_frontier: set[str] = set()
        for edge in frontier:
            for successor in NON_INTERNAL_SUCCESSORS.get(edge, set()):
                if successor in targets:
                    return True
                if successor not in seen:
                    seen.add(successor)
                    next_frontier.add(successor)
        if not next_frontier:
            return False
        frontier = next_frontier
    return False


def _nearest_sample(samples: Sequence[tuple[float, float, float, float]], s_m: float) -> tuple[float, float, float, float] | None:
    if not samples or not math.isfinite(float(s_m)):
        return None
    values = [sample[0] for sample in samples]
    index = bisect.bisect_left(values, float(s_m))
    candidates: list[tuple[float, float, float, float]] = []
    if index < len(samples):
        candidates.append(samples[index])
    if index > 0:
        candidates.append(samples[index - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item[0] - float(s_m)))


def _coerce_position(entity: dict[str, Any]) -> list[float] | None:
    pose = dict(entity.get("truth_pose") or {})
    position = pose.get("position_enu_m") or pose.get("position_m")
    if not isinstance(position, list) or len(position) < 2:
        return None
    try:
        return [
            float(position[0]),
            float(position[1]),
            float(position[2] if len(position) > 2 else 0.0),
        ]
    except (TypeError, ValueError):
        return None


def _read_vector(config: dict[str, Any], field_name: str, default: Sequence[float]) -> tuple[float, float, float]:
    raw = config.get(field_name, default)
    values = list(raw if isinstance(raw, Sequence) and not isinstance(raw, str) else default)
    return (
        float(values[0] if len(values) > 0 else default[0]),
        float(values[1] if len(values) > 1 else default[1]),
        float(values[2] if len(values) > 2 else default[2]),
    )


def _apply_axis_mapping(axis_mapping: str, x: float, y: float, z: float) -> tuple[float, float, float]:
    mapping = axis_mapping.strip() or "XY_To_XY"
    if mapping == "XY_To_XNegY":
        return x, -y, z
    if mapping == "XY_To_YX":
        return y, x, z
    if mapping == "XY_To_YNegX":
        return y, -x, z
    return x, y, z


def _transform_position_for_render(position: Sequence[float], render_config: dict[str, Any]) -> list[float]:
    coordinate_space = str(render_config.get("truth_frame_coordinate_space", "local_enu") or "local_enu").strip().lower()
    if coordinate_space in {"map_enu", "world_enu", "transformed_enu"}:
        return [
            float(position[0] if len(position) > 0 else 0.0),
            float(position[1] if len(position) > 1 else 0.0),
            float(position[2] if len(position) > 2 else 0.0),
        ]

    transform = dict(render_config.get("coordinate_transform") or {})
    enabled = bool(transform.get("enabled", False))
    scale = _read_vector(transform, "scale_enu", (1.0, 1.0, 1.0))
    translation = _read_vector(transform, "translation_enu_m", (0.0, 0.0, 0.0))
    axis_mapping = str(transform.get("axis_mapping", "XY_To_XY") or "XY_To_XY")
    yaw_deg = float(transform.get("yaw_deg", 0.0))

    x = float(position[0] if len(position) > 0 else 0.0) * scale[0]
    y = float(position[1] if len(position) > 1 else 0.0) * scale[1]
    z = float(position[2] if len(position) > 2 else 0.0) * scale[2]
    if enabled:
        x, y, z = _apply_axis_mapping(axis_mapping, x, y, z)
        if abs(yaw_deg) > 1e-6:
            yaw_rad = math.radians(yaw_deg)
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            x, y = x * cos_yaw - y * sin_yaw, x * sin_yaw + y * cos_yaw
        x += translation[0]
        y += translation[1]
        z += translation[2]
    return [x, y, z]


def _render_snap_s(sumo_vehicle: dict[str, Any]) -> float | None:
    raw_s = sumo_vehicle.get("center_lane_position_m")
    if raw_s is None:
        raw_s = sumo_vehicle.get("lane_position_m")
    try:
        s_m = float(raw_s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(s_m) or s_m < 0.0:
        return None
    return s_m


def _is_vehicle(entity: dict[str, Any]) -> bool:
    return str(entity.get("entity_category") or entity.get("category") or "").strip().lower() == "vehicle"


def _is_internal_edge(edge_id: str) -> bool:
    return str(edge_id or "").startswith(":")


def _selected_episode_dirs(input_root: Path, episode_names: list[str], limit: int) -> list[Path]:
    all_dirs = sorted(path for path in input_root.iterdir() if path.is_dir() and (path / "truth_frames.jsonl").exists())
    if episode_names:
        by_name = {path.name: path for path in all_dirs}
        missing = sorted(name for name in episode_names if name not in by_name)
        if missing:
            raise SystemExit(f"Missing formal capture episodes: {missing}")
        all_dirs = [by_name[name] for name in episode_names]
    if limit > 0:
        all_dirs = all_dirs[:limit]
    return all_dirs


def _add_error(
    errors: dict[str, dict[str, Any]],
    *,
    category: str,
    key: str,
    message: str,
    example: dict[str, Any],
    max_examples_per_category: int,
) -> None:
    payload = errors.setdefault(category, {"count": 0, "examples": []})
    seen_keys = payload.setdefault("_seen_keys", set())
    if key in seen_keys:
        payload["count"] += 1
        return
    seen_keys.add(key)
    payload["count"] += 1
    if len(payload["examples"]) < max_examples_per_category:
        payload["examples"].append({"message": message, **example})


def _finalize_errors(errors: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for category, payload in sorted(errors.items()):
        finalized[category] = {
            "count": int(payload.get("count") or 0),
            "examples": list(payload.get("examples") or []),
        }
    return finalized


def _validate_episode_impl(
    episode_dir: Path,
    *,
    tick_stride: int,
    all_ticks: bool,
    max_lane_snap_distance_m: float,
    max_examples_per_category: int,
) -> dict[str, Any]:
    render_config_path = episode_dir / "render_host_config.json"
    truth_path = episode_dir / "truth_frames.jsonl"
    render_config = _load_json(render_config_path) if render_config_path.exists() else {}
    errors: dict[str, dict[str, Any]] = {}
    records_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats: dict[str, Any] = {
        "frames_scanned": 0,
        "vehicle_records": 0,
        "lane_snap_records": 0,
        "internal_lane_records": 0,
        "internal_connection_skip_tolerated_records": 0,
        "max_lane_sample_distance_m": 0.0,
        "max_lane_sample_distance_record": None,
    }

    with truth_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            frame = json.loads(line)
            tick = int(frame.get("tick") or 0)
            if not all_ticks and tick_stride > 1 and tick % tick_stride != 0:
                continue
            stats["frames_scanned"] += 1
            entities = frame.get("entities") or []
            if not isinstance(entities, list):
                continue
            for entity in entities:
                if not isinstance(entity, dict) or not _is_vehicle(entity):
                    continue
                sumo_vehicle = dict(entity.get("sumo_vehicle") or {})
                if not sumo_vehicle:
                    continue
                entity_id = str(entity.get("entity_id") or sumo_vehicle.get("vehicle_id") or "")
                edge_id = str(sumo_vehicle.get("sumo_edge_id") or "").strip()
                lane_id = str(sumo_vehicle.get("sumo_lane_id") or "").strip()
                position = _coerce_position(entity)
                if position is None:
                    continue
                render_position = _transform_position_for_render(position, render_config)
                s_m = _render_snap_s(sumo_vehicle)
                stats["vehicle_records"] += 1
                if _is_internal_edge(edge_id) or _is_internal_edge(lane_id):
                    stats["internal_lane_records"] += 1
                record = {
                    "tick": tick,
                    "line_number": line_number,
                    "entity_id": entity_id,
                    "edge_id": edge_id,
                    "lane_id": lane_id,
                    "s_m": s_m,
                    "position_enu_m": render_position,
                }
                records_by_entity[entity_id].append(record)

                if not lane_id:
                    _add_error(
                        errors,
                        category="missing_sumo_lane_id",
                        key=f"{entity_id}:{tick}",
                        message="vehicle record has no sumo_lane_id",
                        example={"tick": tick, "entity_id": entity_id, "edge_id": edge_id},
                        max_examples_per_category=max_examples_per_category,
                    )
                    continue

                samples = LANE_SAMPLES.get(lane_id)
                if samples is None:
                    _add_error(
                        errors,
                        category="missing_lane_sample",
                        key=f"{lane_id}",
                        message=f"current lane_center_samples.csv has no lane_id={lane_id}",
                        example={"tick": tick, "entity_id": entity_id, "edge_id": edge_id, "lane_id": lane_id},
                        max_examples_per_category=max_examples_per_category,
                    )
                    if edge_id and edge_id not in EDGE_HAS_SAMPLES:
                        _add_error(
                            errors,
                            category="missing_edge_sample",
                            key=f"{edge_id}",
                            message=f"current lane_center_samples.csv has no samples for edge_id={edge_id}",
                            example={"tick": tick, "entity_id": entity_id, "edge_id": edge_id, "lane_id": lane_id},
                            max_examples_per_category=max_examples_per_category,
                        )
                    continue

                if s_m is None:
                    continue
                sample = _nearest_sample(samples, s_m)
                if sample is None:
                    continue
                stats["lane_snap_records"] += 1
                distance_m = math.hypot(float(sample[1]) - render_position[0], float(sample[2]) - render_position[1])
                if distance_m > float(stats["max_lane_sample_distance_m"]):
                    stats["max_lane_sample_distance_m"] = round(distance_m, 6)
                    stats["max_lane_sample_distance_record"] = {
                        "tick": tick,
                        "entity_id": entity_id,
                        "edge_id": edge_id,
                        "lane_id": lane_id,
                        "s_m": s_m,
                        "truth_position_enu_m": [round(value, 6) for value in render_position],
                        "sample_position_enu_m": [round(sample[1], 6), round(sample[2], 6), round(sample[3], 6)],
                        "distance_m": round(distance_m, 6),
                    }
                if distance_m > max_lane_snap_distance_m:
                    _add_error(
                        errors,
                        category="lane_sample_xy_distance",
                        key=f"{entity_id}:{tick}:{lane_id}:{round(distance_m, 3)}",
                        message=(
                            f"lane_id={lane_id} sample at s={s_m:.3f} is {distance_m:.3f}m from truth XY "
                            f"(limit {max_lane_snap_distance_m:.3f}m)"
                        ),
                        example={
                            "tick": tick,
                            "entity_id": entity_id,
                            "edge_id": edge_id,
                            "lane_id": lane_id,
                            "s_m": round(s_m, 6),
                            "truth_position_enu_m": [round(value, 6) for value in render_position],
                            "sample_position_enu_m": [round(sample[1], 6), round(sample[2], 6), round(sample[3], 6)],
                            "distance_m": round(distance_m, 6),
                        },
                        max_examples_per_category=max_examples_per_category,
                    )

    for entity_id, records in records_by_entity.items():
        records.sort(key=lambda item: int(item["tick"]))
        previous_non_internal: list[str | None] = []
        last_non_internal: str | None = None
        for record in records:
            previous_non_internal.append(last_non_internal)
            edge_id = str(record["edge_id"])
            if edge_id and not _is_internal_edge(edge_id):
                last_non_internal = edge_id
        next_non_internal: list[str | None] = [None] * len(records)
        next_edge: str | None = None
        for index in range(len(records) - 1, -1, -1):
            next_non_internal[index] = next_edge
            edge_id = str(records[index]["edge_id"])
            if edge_id and not _is_internal_edge(edge_id):
                next_edge = edge_id

        for index, record in enumerate(records):
            edge_id = str(record["edge_id"])
            lane_id = str(record["lane_id"])
            if not (_is_internal_edge(edge_id) or _is_internal_edge(lane_id)):
                continue
            prev_edge = previous_non_internal[index]
            next_edge = next_non_internal[index]
            if not prev_edge or not next_edge:
                continue
            connection = INTERNAL_CONNECTION_BY_LANE.get(lane_id) or INTERNAL_CONNECTION_BY_EDGE.get(edge_id)
            if connection is None:
                _add_error(
                    errors,
                    category="missing_internal_connection",
                    key=f"{edge_id}:{lane_id}:{prev_edge}:{next_edge}",
                    message=f"current net.xml has no connection for internal lane_id={lane_id} edge_id={edge_id}",
                    example={
                        "tick": int(record["tick"]),
                        "entity_id": entity_id,
                        "edge_id": edge_id,
                        "lane_id": lane_id,
                        "prev_non_internal_edge": prev_edge,
                        "next_non_internal_edge": next_edge,
                    },
                    max_examples_per_category=max_examples_per_category,
                )
                continue
            current_from = str(connection.get("from") or "")
            current_to = str(connection.get("to") or "")
            terminal_from_edges = set(str(value) for value in (connection.get("terminal_from_edges") or []))
            terminal_to_edges = set(str(value) for value in (connection.get("terminal_to_edges") or []))
            from_ok = _non_internal_reachable(
                prev_edge,
                terminal_from_edges,
                MAX_SKIPPED_INTERNAL_CONNECTION_HOPS,
            )
            to_ok = any(
                _non_internal_reachable(
                    terminal_to_edge,
                    {next_edge},
                    MAX_SKIPPED_INTERNAL_CONNECTION_HOPS,
                )
                for terminal_to_edge in terminal_to_edges
            )
            if from_ok and to_ok:
                if prev_edge not in terminal_from_edges or next_edge not in terminal_to_edges:
                    stats["internal_connection_skip_tolerated_records"] += 1
                continue
            if not from_ok or not to_ok:
                _add_error(
                    errors,
                    category="internal_connection_mismatch",
                    key=f"{edge_id}:{lane_id}:{prev_edge}:{next_edge}:{current_from}:{current_to}",
                    message=(
                        f"internal lane_id={lane_id} is used between {prev_edge}->{next_edge} in truth, but current net "
                        f"terminal mapping is {sorted(terminal_from_edges)}->{sorted(terminal_to_edges)}"
                    ),
                    example={
                        "tick": int(record["tick"]),
                        "entity_id": entity_id,
                        "edge_id": edge_id,
                        "lane_id": lane_id,
                        "truth_prev_non_internal_edge": prev_edge,
                        "truth_next_non_internal_edge": next_edge,
                        "current_net_from": current_from,
                        "current_net_to": current_to,
                        "current_net_terminal_from_edges": sorted(terminal_from_edges),
                        "current_net_terminal_to_edges": sorted(terminal_to_edges),
                        "current_net_via_lane": str(connection.get("via_lane") or ""),
                    },
                    max_examples_per_category=max_examples_per_category,
                )

    finalized_errors = _finalize_errors(errors)
    return {
        "episode_id": episode_dir.name,
        "status": "pass" if not finalized_errors else "fail",
        "stats": stats,
        "errors": finalized_errors,
    }


def _validate_episode_worker(args: tuple[str, int, bool, float, int]) -> dict[str, Any]:
    episode_dir, tick_stride, all_ticks, max_lane_snap_distance_m, max_examples_per_category = args
    return _validate_episode_impl(
        Path(episode_dir),
        tick_stride=tick_stride,
        all_ticks=all_ticks,
        max_lane_snap_distance_m=max_lane_snap_distance_m,
        max_examples_per_category=max_examples_per_category,
    )


def _parse_episode_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                names.append(item)
    return names


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--traffic-bundle-dir", type=Path, default=DEFAULT_TRAFFIC_BUNDLE_DIR)
    parser.add_argument("--net-xml", type=Path, default=DEFAULT_NET_XML)
    parser.add_argument("--episode", action="append", help="Episode directory name; may be repeated or comma-separated.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tick-stride", type=int, default=DEFAULT_TICK_STRIDE)
    parser.add_argument("--all-ticks", action="store_true", help="Scan every truth tick instead of the formal capture tick stride.")
    parser.add_argument("--max-lane-snap-distance-m", type=float, default=DEFAULT_MAX_LANE_SNAP_DISTANCE_M)
    parser.add_argument("--max-examples-per-category", type=int, default=DEFAULT_MAX_EXAMPLES_PER_CATEGORY)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_root = args.input_root.resolve()
    traffic_bundle_dir = args.traffic_bundle_dir.resolve()
    lane_center_samples = traffic_bundle_dir / "lane_center_samples.csv"
    net_xml = args.net_xml.resolve()
    if not lane_center_samples.exists():
        raise SystemExit(f"missing lane_center_samples.csv: {lane_center_samples}")
    if not net_xml.exists():
        raise SystemExit(f"missing net.xml: {net_xml}")

    episode_dirs = _selected_episode_dirs(input_root, _parse_episode_names(args.episode), int(args.limit))
    if not episode_dirs:
        raise SystemExit(f"no truth episodes found under {input_root}")

    if int(args.workers) <= 1:
        _init_worker(str(lane_center_samples), str(net_xml))
        results = [
            _validate_episode_impl(
                episode_dir,
                tick_stride=max(1, int(args.tick_stride)),
                all_ticks=bool(args.all_ticks),
                max_lane_snap_distance_m=float(args.max_lane_snap_distance_m),
                max_examples_per_category=max(1, int(args.max_examples_per_category)),
            )
            for episode_dir in episode_dirs
        ]
    else:
        results = []
        worker_args = [
            (
                str(episode_dir),
                max(1, int(args.tick_stride)),
                bool(args.all_ticks),
                float(args.max_lane_snap_distance_m),
                max(1, int(args.max_examples_per_category)),
            )
            for episode_dir in episode_dirs
        ]
        with ProcessPoolExecutor(
            max_workers=max(1, int(args.workers)),
            initializer=_init_worker,
            initargs=(str(lane_center_samples), str(net_xml)),
        ) as executor:
            futures = [executor.submit(_validate_episode_worker, item) for item in worker_args]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: str(item.get("episode_id") or ""))

    failed = [result for result in results if result.get("status") != "pass"]
    error_counts: Counter[str] = Counter()
    for result in failed:
        for category, payload in (result.get("errors") or {}).items():
            error_counts[str(category)] += int(payload.get("count") or 0)
    output = {
        "schema_name": "truth_traffic_bundle_compatibility_report",
        "schema_version": "v1",
        "input_root": str(input_root),
        "lane_center_samples": str(lane_center_samples),
        "net_xml": str(net_xml),
        "tick_policy": "all_ticks" if args.all_ticks else f"tick % {max(1, int(args.tick_stride))} == 0",
        "max_lane_snap_distance_m": float(args.max_lane_snap_distance_m),
        "summary": {
            "episode_count": len(results),
            "pass_count": len(results) - len(failed),
            "fail_count": len(failed),
            "failed_episodes": [str(result.get("episode_id") or "") for result in failed],
            "error_counts": dict(sorted(error_counts.items())),
        },
        "episodes": results,
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.quiet:
        summary = output["summary"]
        print(
            f"[truth-bundle-compat] episodes={summary['episode_count']} "
            f"pass={summary['pass_count']} fail={summary['fail_count']} "
            f"tick_policy={output['tick_policy']} max_lane_snap_distance_m={output['max_lane_snap_distance_m']}"
        )
        if error_counts:
            print("[truth-bundle-compat] error_counts=" + json.dumps(dict(sorted(error_counts.items())), ensure_ascii=False))
        if failed:
            print("[truth-bundle-compat] failed_episodes=" + ", ".join(summary["failed_episodes"]))
            for result in failed[:20]:
                categories = {
                    category: int(payload.get("count") or 0)
                    for category, payload in (result.get("errors") or {}).items()
                }
                print(f"[truth-bundle-compat] {result['episode_id']}: {json.dumps(categories, ensure_ascii=False)}")
                for category, payload in (result.get("errors") or {}).items():
                    examples = payload.get("examples") or []
                    if examples:
                        print(
                            f"  - {category}: "
                            + json.dumps(examples[0], ensure_ascii=False)
                        )
            if len(failed) > 20:
                print(f"[truth-bundle-compat] ... {len(failed) - 20} more failed episodes omitted from console")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
