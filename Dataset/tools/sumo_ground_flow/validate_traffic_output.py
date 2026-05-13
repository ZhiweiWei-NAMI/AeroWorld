"""Validate SUMO traffic exports against the capture-oriented contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


REQUIRED_VEHICLE_FIELDS = {
    "vehicle_id",
    "vehicle_type",
    "route_id",
    "sumo_edge_id",
    "sumo_lane_id",
    "lane_position_m",
    "sumo_xy_m",
    "truth_position_enu_m",
    "truth_yaw_deg",
    "speed_mps",
    "accel_mps2",
    "signals",
    "dimensions_m",
    "source",
}

REQUIRED_TRAFFIC_LIGHT_FIELDS = {
    "tls_id",
    "program_id",
    "phase_index",
    "state",
    "next_switch_s",
    "controlled_links",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_output(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "sumo_traffic_manifest.json"
    incident_plan_path = output_dir / "sumo_incident_plan.json"
    frames_path = output_dir / "sumo_traffic_frames.jsonl"
    errors: list[str] = []
    warnings: list[str] = []
    for path in (manifest_path, incident_plan_path, frames_path):
        if not path.exists():
            errors.append(f"missing required output: {path}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}

    manifest = _load_json(manifest_path)
    plan = _load_json(incident_plan_path)
    frames = _load_jsonl(frames_path)
    duration_s = float(manifest.get("duration_s") or 0.0)
    sample_period_s = float(manifest.get("sample_period_s") or 0.0)
    sample_every_ticks = int(manifest.get("sample_every_ticks") or 0)
    expected_frame_count = int(round(duration_s / sample_period_s)) + 1 if sample_period_s > 0 else 0

    if sample_period_s != 0.5:
        errors.append(f"sample_period_s must be 0.5, got {sample_period_s}")
    if sample_every_ticks != 5:
        errors.append(f"sample_every_ticks must be 5, got {sample_every_ticks}")
    if len(frames) != expected_frame_count:
        errors.append(f"expected {expected_frame_count} frames, got {len(frames)}")
    if frames and abs(float(frames[0].get("sim_time_s") or 0.0) - 0.0) > 1e-9:
        errors.append("first frame must be sim_time_s=0.0")
    if frames and abs(float(frames[-1].get("sim_time_s") or 0.0) - duration_s) > 1e-6:
        errors.append(f"last frame must be sim_time_s={duration_s}, got {frames[-1].get('sim_time_s')}")

    for prev, current in zip(frames, frames[1:]):
        dt = round(float(current["sim_time_s"]) - float(prev["sim_time_s"]), 6)
        if dt != 0.5:
            errors.append(f"non-0.5s frame delta at {prev.get('sim_time_s')} -> {current.get('sim_time_s')}: {dt}")
            break
        tick_delta = int(current["tick"]) - int(prev["tick"])
        if tick_delta != 5:
            errors.append(f"non-5tick frame delta at {prev.get('tick')} -> {current.get('tick')}: {tick_delta}")
            break

    traffic_light_counts = [len(frame.get("traffic_lights") or []) for frame in frames]
    if not traffic_light_counts or min(traffic_light_counts) <= 0:
        errors.append("every sampled frame must include traffic light records")
    for frame in frames[: min(5, len(frames))]:
        for traffic_light in frame.get("traffic_lights") or []:
            missing = REQUIRED_TRAFFIC_LIGHT_FIELDS - set(traffic_light)
            if missing:
                errors.append(f"traffic light record missing fields: {sorted(missing)}")
                break

    vehicle_counts = [len(frame.get("vehicles") or []) for frame in frames]
    for frame in frames:
        for vehicle in frame.get("vehicles") or []:
            missing = REQUIRED_VEHICLE_FIELDS - set(vehicle)
            if missing:
                errors.append(f"vehicle record missing fields: {sorted(missing)}")
                break
            if vehicle.get("source") != "sumo_traci":
                errors.append(f"vehicle {vehicle.get('vehicle_id')} has non-SUMO source {vehicle.get('source')}")
                break
            if len(vehicle.get("sumo_xy_m") or []) != 2 or len(vehicle.get("truth_position_enu_m") or []) != 3:
                errors.append(f"vehicle {vehicle.get('vehicle_id')} has invalid coordinate payload")
                break
        if errors:
            break

    max_vehicles = int(manifest.get("max_vehicles") or 0)
    if max_vehicles > 0 and vehicle_counts and max(vehicle_counts) > max_vehicles:
        errors.append(f"vehicle count exceeded max_vehicles={max_vehicles}: observed={max(vehicle_counts)}")
    if duration_s >= 270.0 and max_vehicles >= 200:
        phase_a = [len(frame.get("vehicles") or []) for frame in frames if 0.0 <= float(frame["sim_time_s"]) <= 90.0]
        phase_b = [len(frame.get("vehicles") or []) for frame in frames if 90.0 <= float(frame["sim_time_s"]) <= 180.0]
        phase_c = [len(frame.get("vehicles") or []) for frame in frames if 180.0 <= float(frame["sim_time_s"]) <= 270.0]
        if phase_a and max(phase_a) < 120:
            errors.append(f"ramp-up phase did not build enough traffic: max={max(phase_a)}")
        if phase_b and max(phase_b) < 180:
            errors.append(f"steady phase did not reach near cap: max={max(phase_b)}")
        if phase_c and phase_c[-1] > 25:
            errors.append(f"ramp-down phase left too many active vehicles: final={phase_c[-1]}")

    incidents = list(plan.get("incidents") or [])
    incident_ids = {str(item.get("incident_id")) for item in incidents}
    active_ids = {str(item.get("incident_id")) for frame in frames for item in frame.get("active_incidents") or []}
    missing_incidents = sorted(incident_ids - active_ids)
    if missing_incidents:
        errors.append(f"incidents never became active in sampled frames: {missing_incidents[:10]}")

    classes = {str(item.get("accident_class")) for item in incidents}
    active_classes = {str(item.get("accident_class")) for frame in frames for item in frame.get("active_incidents") or []}
    missing_classes = sorted(classes - active_classes)
    if missing_classes:
        errors.append(f"accident classes never became active: {missing_classes}")

    all_red_incidents = [item for item in incidents if item.get("accident_class") == "traffic_light_all_red_fault"]
    for incident in all_red_incidents:
        traffic_light = dict(incident.get("traffic_light") or {})
        tls_id = str(traffic_light.get("traffic_light_id") or "")
        if not tls_id:
            errors.append(f"all-red incident lacks traffic light id: {incident.get('incident_id')}")
            continue
        observed = False
        for frame in frames:
            sim_time_s = float(frame.get("sim_time_s") or 0.0)
            if not (float(incident["start_s"]) <= sim_time_s <= float(incident["end_s"])):
                continue
            for traffic_light_record in frame.get("traffic_lights") or []:
                if traffic_light_record.get("tls_id") == tls_id and set(str(traffic_light_record.get("state") or "")) == {"r"}:
                    observed = True
                    break
            if observed:
                break
        if not observed:
            errors.append(f"all-red traffic light state was not observed for {incident.get('incident_id')}")

    summary = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "frame_count": len(frames),
        "duration_s": duration_s,
        "sample_period_s": sample_period_s,
        "traffic_light_count_min": min(traffic_light_counts) if traffic_light_counts else 0,
        "traffic_light_count_max": max(traffic_light_counts) if traffic_light_counts else 0,
        "vehicle_count_min": min(vehicle_counts) if vehicle_counts else 0,
        "vehicle_count_max": max(vehicle_counts) if vehicle_counts else 0,
        "vehicle_count_final": vehicle_counts[-1] if vehicle_counts else 0,
        "incident_count": len(incidents),
        "active_incident_count": len(active_ids),
        "accident_classes": sorted(classes),
    }
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SUMO traffic output files.")
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_output(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
