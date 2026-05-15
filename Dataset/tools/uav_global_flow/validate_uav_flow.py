"""Validate Donghu global UAV flow exports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ROOT / "Dataset" / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from Dataset.tools.map_spatial_index import (  # noqa: E402
    PEDESTRIAN_ROAD_BUFFER_M,
    MapSpatialIndex,
)
from Dataset.tools.uav_corridor_planner import BuildingObstacleIndex  # noqa: E402
from Dataset.tools.uav_global_flow.generate_uav_flow import (  # noqa: E402
    ALLOWED_ALTITUDE_LAYERS_M,
    DEFAULT_OUTPUT_DIR,
    DURATION_S,
    MIN_ACTIVE_UAVS,
    SAMPLE_EVERY_TICKS,
    SAMPLE_PERIOD_S,
    TICK_HZ,
)


REQUIRED_UAV_FIELDS = {
    "uav_id",
    "task_id",
    "mission_type",
    "semantic_role",
    "corridor_family",
    "corridor_id",
    "altitude_layer_m",
    "position_enu_m",
    "yaw_deg",
    "speed_mps",
    "source",
}
REQUIRED_PAD_FIELDS = {
    "pad_id",
    "role",
    "position_enu_m",
    "grid_cell_id",
    "lane_edge_id",
    "lane_s_m",
}
PAD_ENDPOINT_TOLERANCE_M = 3.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    return rows


def _dist_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _dist3(a: Sequence[float], b: Sequence[float]) -> float:
    az = float(a[2]) if len(a) >= 3 else 0.0
    bz = float(b[2]) if len(b) >= 3 else 0.0
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (az - bz) ** 2)


def _path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(_dist3(a, b) for a, b in zip(points, points[1:]))


def _position(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError):
        return None


def _route_clear(route: Sequence[Sequence[float]], buildings: BuildingObstacleIndex) -> bool:
    if len(route) < 2:
        return False
    if any(not buildings.air_point_clear(point) for point in route):
        return False
    return all(buildings.air_segment_clear(a, b) for a, b in zip(route, route[1:]))


def validate_output(output_dir: Path, *, check_obstacles: bool = True) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "uav_flow_manifest.json"
    task_plan_path = output_dir / "uav_task_plan.json"
    frames_path = output_dir / "uav_traffic_frames.jsonl"
    errors: list[str] = []
    warnings: list[str] = []
    for path in (manifest_path, task_plan_path, frames_path):
        if not path.exists():
            errors.append(f"missing required output: {path}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}

    manifest = _load_json(manifest_path)
    task_plan = _load_json(task_plan_path)
    frames = _load_jsonl(frames_path)
    expected_frame_count = int(round(DURATION_S / SAMPLE_PERIOD_S)) + 1

    if manifest.get("sample_period_s") != SAMPLE_PERIOD_S:
        errors.append(f"manifest sample_period_s must be {SAMPLE_PERIOD_S}, got {manifest.get('sample_period_s')}")
    if manifest.get("sample_every_ticks") != SAMPLE_EVERY_TICKS:
        errors.append(f"manifest sample_every_ticks must be {SAMPLE_EVERY_TICKS}, got {manifest.get('sample_every_ticks')}")
    if len(frames) != expected_frame_count:
        errors.append(f"expected {expected_frame_count} sampled frames, got {len(frames)}")
    if frames and abs(float(frames[0].get("sim_time_s") or 0.0)) > 1e-9:
        errors.append("first frame must be sim_time_s=0.0")
    if frames and abs(float(frames[-1].get("sim_time_s") or 0.0) - DURATION_S) > 1e-6:
        errors.append(f"last frame must be sim_time_s={DURATION_S}, got {frames[-1].get('sim_time_s')}")

    for previous, current in zip(frames, frames[1:]):
        dt = round(float(current["sim_time_s"]) - float(previous["sim_time_s"]), 6)
        tick_delta = int(current["tick"]) - int(previous["tick"])
        if dt != SAMPLE_PERIOD_S:
            errors.append(f"non-{SAMPLE_PERIOD_S}s frame delta at {previous.get('sim_time_s')} -> {current.get('sim_time_s')}: {dt}")
            break
        if tick_delta != SAMPLE_EVERY_TICKS:
            errors.append(f"non-{SAMPLE_EVERY_TICKS}tick frame delta at {previous.get('tick')} -> {current.get('tick')}: {tick_delta}")
            break

    counts = [int(frame.get("active_uav_count") or len(frame.get("uavs") or [])) for frame in frames]
    if counts and min(counts) < MIN_ACTIVE_UAVS:
        errors.append(f"active UAV floor violated: min={min(counts)} required={MIN_ACTIVE_UAVS}")

    pads = list(task_plan.get("pads") or [])
    if int(manifest.get("pad_count") or 0) != len(pads):
        errors.append(f"manifest pad_count {manifest.get('pad_count')} != task_plan pads {len(pads)}")
    pad_by_id: dict[str, dict[str, Any]] = {}
    pad_positions: dict[str, tuple[float, float, float]] = {}
    pad_road_clearance_min_m = float("inf")
    if len(pads) < int(manifest.get("pad_count") or 0):
        errors.append(f"task_plan pads count {len(pads)} is below manifest pad_count")
    spatial: MapSpatialIndex | None = None
    if pads:
        try:
            spatial = MapSpatialIndex.default(ROOT)
        except Exception as exc:
            errors.append(f"unable to load map spatial index for pad validation: {exc}")
    for pad in pads:
        if not isinstance(pad, dict):
            errors.append(f"invalid pad record: {pad!r}")
            break
        missing = REQUIRED_PAD_FIELDS - set(pad)
        if missing:
            errors.append(f"pad record missing fields: {sorted(missing)}")
            break
        pad_id = str(pad.get("pad_id") or "")
        if not pad_id:
            errors.append("pad record has empty pad_id")
            break
        if pad_id in pad_by_id:
            errors.append(f"duplicate pad_id: {pad_id}")
            break
        pos = _position(pad.get("position_enu_m"))
        if pos is None:
            errors.append(f"pad {pad_id} invalid position_enu_m")
            break
        pad_by_id[pad_id] = pad
        pad_positions[pad_id] = pos
        if spatial is not None:
            pad_errors = spatial.validation_errors_for_point(
                pos,
                context=f"pad {pad_id}",
                allow_road=False,
                allow_green=True,
                road_buffer_m=PEDESTRIAN_ROAD_BUFFER_M,
            )
            if pad_errors:
                errors.extend(pad_errors)
                break
            clearance = spatial.nearest_lane_clearance(pos)
            pad_road_clearance_min_m = min(pad_road_clearance_min_m, clearance)

    allowed_altitudes = {float(value) for value in ALLOWED_ALTITUDE_LAYERS_M}
    per_uav_positions: dict[str, list[tuple[float, tuple[float, float, float]]]] = {}
    source_values: set[str] = set()
    same_layer_min_distance_m = float("inf")
    for frame in frames:
        uavs = list(frame.get("uavs") or [])
        if len(uavs) != int(frame.get("active_uav_count") or 0):
            errors.append(f"active_uav_count mismatch at tick={frame.get('tick')}")
            break
        by_layer: dict[float, list[tuple[str, tuple[float, float, float]]]] = {}
        for uav in uavs:
            missing = REQUIRED_UAV_FIELDS - set(uav)
            if missing:
                errors.append(f"UAV record missing fields at tick={frame.get('tick')}: {sorted(missing)}")
                break
            if uav.get("source") != "uav_global_flow":
                errors.append(f"UAV {uav.get('uav_id')} has invalid source={uav.get('source')}")
                break
            source_values.add(str(uav.get("source")))
            for ref_key in ("origin_pad_id", "target_pad_id"):
                pad_id = str(uav.get(ref_key) or "")
                if pad_id and pad_id not in pad_by_id:
                    errors.append(f"UAV {uav.get('uav_id')} references unknown {ref_key}={pad_id} at tick={frame.get('tick')}")
                    break
            if errors:
                break
            pos = _position(uav.get("position_enu_m"))
            if pos is None:
                errors.append(f"UAV {uav.get('uav_id')} invalid position at tick={frame.get('tick')}")
                break
            altitude_layer = float(uav.get("altitude_layer_m"))
            if altitude_layer not in allowed_altitudes:
                errors.append(f"UAV {uav.get('uav_id')} uses disallowed altitude layer {altitude_layer}")
                break
            mission_type = str(uav.get("mission_type") or "")
            fixed_altitude_missions = {"pad_patrol", "intersection_inspect", "edge_compute_relay"}
            if mission_type in fixed_altitude_missions and abs(pos[2] - altitude_layer) > 0.75:
                warnings.append(
                    f"UAV {uav.get('uav_id')} position z={pos[2]:.2f} differs from layer={altitude_layer:.2f} at tick={frame.get('tick')}"
                )
            elif mission_type not in fixed_altitude_missions and pos[2] > max(ALLOWED_ALTITUDE_LAYERS_M) + 1.0:
                warnings.append(
                    f"UAV {uav.get('uav_id')} climbed above allowed layers z={pos[2]:.2f} at tick={frame.get('tick')}"
                )
            per_uav_positions.setdefault(str(uav.get("uav_id")), []).append((float(frame["sim_time_s"]), pos))
            by_layer.setdefault(round(altitude_layer, 3), []).append((str(uav.get("uav_id")), pos))
        if errors:
            break
        for layer, items in by_layer.items():
            for idx, (left_id, left) in enumerate(items):
                for right_id, right in items[idx + 1 :]:
                    distance = _dist3(left, right)
                    if distance < same_layer_min_distance_m:
                        same_layer_min_distance_m = distance
                    if distance < 2.0:
                        errors.append(
                            f"dynamic separation violated at tick={frame.get('tick')} layer={layer}: {left_id}/{right_id} {distance:.2f}m"
                        )
                        break
                if errors:
                    break
            if errors:
                break
        if errors:
            break

    static_uavs: list[dict[str, Any]] = []
    delayed_or_stuck_uavs: list[dict[str, Any]] = []
    max_stationary_frames = 0
    for uav_id, samples in per_uav_positions.items():
        if len(samples) < 4:
            continue
        xs = [pos[0] for _time, pos in samples]
        ys = [pos[1] for _time, pos in samples]
        zs = [pos[2] for _time, pos in samples]
        span = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2)
        if span < 1.5:
            static_uavs.append({"uav_id": uav_id, "xy_span_m": round(span, 3), "sample_count": len(samples)})
        stationary_run = 0
        longest = 0
        for (_prev_t, prev_pos), (_t, pos) in zip(samples, samples[1:]):
            if _dist3(prev_pos, pos) < 0.15:
                stationary_run += 1
                longest = max(longest, stationary_run)
            else:
                stationary_run = 0
        max_stationary_frames = max(max_stationary_frames, longest)
        if longest >= 4:
            delayed_or_stuck_uavs.append({"uav_id": uav_id, "longest_stationary_sample_run": longest})
    if static_uavs:
        errors.append(f"static UAV tracks detected: {static_uavs[:10]}")
    if delayed_or_stuck_uavs:
        errors.append(f"UAVs had >=4 consecutive 0.5s near-stationary samples: {delayed_or_stuck_uavs[:10]}")

    tasks = list(task_plan.get("tasks") or [])
    baseline_tasks = [
        task
        for task in tasks
        if task.get("mission_type") in {"pad_patrol", "intersection_inspect", "edge_compute_relay"}
    ]
    if len(baseline_tasks) < 52:
        errors.append(f"baseline persistent UAV count must be >=52, got {len(baseline_tasks)}")
    task_types = {str(task.get("mission_type")) for task in tasks}
    required_types = {
        "pad_patrol",
        "intersection_inspect",
        "edge_compute_relay",
        "logistics_delivery",
        "infrastructure_inspection",
        "incident_response_inspection",
    }
    missing_types = sorted(required_types - task_types)
    if missing_types:
        errors.append(f"missing required UAV task types: {missing_types}")
    delivery_reference_checked_task_count = 0
    for task in tasks:
        route = task.get("route_waypoints_enu_m") or []
        if len(route) < 2:
            errors.append(f"task {task.get('task_id')} has fewer than 2 route points")
            break
        if float(task.get("speed_mps") or 0.0) <= 0.0:
            errors.append(f"task {task.get('task_id')} has nonpositive speed")
            break
        if _path_length(route) < 15.0:
            errors.append(f"task {task.get('task_id')} route too short")
            break
        origin_pad_id = str(task.get("origin_pad_id") or "")
        target_pad_id = str(task.get("target_pad_id") or "")
        target_cell_id = str(task.get("target_cell_id") or "")
        if origin_pad_id and origin_pad_id not in pad_by_id:
            errors.append(f"task {task.get('task_id')} references unknown origin_pad_id={origin_pad_id}")
            break
        if target_pad_id and target_pad_id not in pad_by_id:
            errors.append(f"task {task.get('task_id')} references unknown target_pad_id={target_pad_id}")
            break
        if target_pad_id and target_cell_id:
            target_cell = str((pad_by_id.get(target_pad_id) or {}).get("grid_cell_id") or "")
            if target_cell and target_cell_id != target_cell:
                errors.append(
                    f"task {task.get('task_id')} target_cell_id={target_cell_id} does not match {target_pad_id}.grid_cell_id={target_cell}"
                )
                break
        if str(task.get("mission_type") or "") == "logistics_delivery":
            delivery_reference_checked_task_count += 1
            if not origin_pad_id or not target_pad_id:
                errors.append(f"delivery task {task.get('task_id')} must declare origin_pad_id and target_pad_id")
                break
            if origin_pad_id == target_pad_id:
                errors.append(f"delivery task {task.get('task_id')} origin and target pads must differ")
                break
            origin_pos = pad_positions.get(origin_pad_id)
            target_pos = pad_positions.get(target_pad_id)
            route_start = _position(route[0])
            route_end = _position(route[-1])
            if origin_pos and route_start and _dist_xy(route_start, origin_pos) > PAD_ENDPOINT_TOLERANCE_M:
                errors.append(f"delivery task {task.get('task_id')} route start is not at origin pad {origin_pad_id}")
                break
            if target_pos and route_end and _dist_xy(route_end, target_pos) > PAD_ENDPOINT_TOLERANCE_M:
                errors.append(f"delivery task {task.get('task_id')} route end is not at target pad {target_pad_id}")
                break

    obstacle_checked_tasks = 0
    if check_obstacles and not errors:
        spatial = spatial or MapSpatialIndex.default(ROOT)
        buildings = BuildingObstacleIndex(spatial)
        for task in tasks:
            route = task.get("route_waypoints_enu_m") or []
            if not _route_clear(route, buildings):
                errors.append(f"task {task.get('task_id')} route is not obstacle/bounds clear")
                break
            obstacle_checked_tasks += 1

    event_bindings = list(task_plan.get("event_bindings") or [])
    if len(event_bindings) < 60:
        errors.append(f"expected broad scenario event bindings, got {len(event_bindings)}")

    summary = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings[:40],
        "frame_count": len(frames),
        "sample_period_s": SAMPLE_PERIOD_S,
        "sample_every_ticks": SAMPLE_EVERY_TICKS,
        "active_uav_count_min": min(counts) if counts else 0,
        "active_uav_count_max": max(counts) if counts else 0,
        "active_uav_count_mean": round(sum(counts) / len(counts), 3) if counts else 0.0,
        "unique_uav_count": len(per_uav_positions),
        "pad_count": len(pads),
        "pad_road_clearance_min_m": round(pad_road_clearance_min_m, 3) if math.isfinite(pad_road_clearance_min_m) else None,
        "task_count": len(tasks),
        "baseline_task_count": len(baseline_tasks),
        "delivery_reference_checked_task_count": delivery_reference_checked_task_count,
        "event_binding_count": len(event_bindings),
        "source_values": sorted(source_values),
        "same_layer_min_distance_m": round(same_layer_min_distance_m, 3) if math.isfinite(same_layer_min_distance_m) else None,
        "max_stationary_sample_run": max_stationary_frames,
        "obstacle_checked_tasks": obstacle_checked_tasks,
    }
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Donghu UAV global flow outputs.")
    parser.add_argument("output_dir", type=Path, nargs="?", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-obstacle-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_output(args.output_dir, check_obstacles=not args.skip_obstacle_check)
    (Path(args.output_dir) / "uav_flow_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
