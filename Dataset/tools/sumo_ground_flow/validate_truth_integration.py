"""Validate SUMO-backed traffic semantics inside render-ready truth frames."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from .truth_integration import SEGMENT_DURATION_S, segment_for_seed
except ImportError:  # pragma: no cover - supports direct script execution.
    from truth_integration import SEGMENT_DURATION_S, segment_for_seed  # type: ignore


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RENDER_READY_ROOT = ROOT / "Dataset" / "render_ready_episodes"
DEFAULT_SCENARIOS_ROOT = ROOT / "Dataset" / "scenarios"
DEFAULT_TICK_HZ = 10
DEFAULT_DURATION_TICKS = 900
DEFAULT_TRAFFIC_LIGHT_COUNT = 29
MAX_ABS_COORD_M = 20000.0
MAX_SUMO_OBSERVATION_DISTANCE_M = 120.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc


def scenario_ids(scenarios_root: Path) -> list[str]:
    return sorted(path.parent.name for path in scenarios_root.rglob("event_script.json"))


def _finite_position(position: Any) -> bool:
    if not isinstance(position, list) or len(position) < 3:
        return False
    try:
        values = [float(position[0]), float(position[1]), float(position[2])]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) and abs(value) <= MAX_ABS_COORD_M for value in values)


def _scenario_incident_overlaps_seed(manifest: dict[str, Any], seed_index: int) -> bool:
    sumo = dict(manifest.get("sumo_traffic") or {})
    segment = segment_for_seed(seed_index)
    for incident in sumo.get("scenario_incidents") or []:
        start_s = float(incident.get("start_s") or 0.0)
        end_s = float(incident.get("end_s") or 0.0)
        if start_s <= segment.segment_end_s and end_s >= segment.segment_start_s:
            return True
    return False


def _observable_sumo_selection_summary(sumo_manifest: dict[str, Any]) -> dict[str, Any]:
    selection = dict(sumo_manifest.get("selection") or {})
    visibility = dict(sumo_manifest.get("visibility_geometry") or {})
    try:
        padding_m = float(visibility.get("padding_m") or 0.0)
    except (TypeError, ValueError):
        padding_m = 0.0
    distances: list[float] = []
    for value in dict(selection.get("min_distance_m_by_vehicle_id") or {}).values():
        try:
            distance_m = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(distance_m):
            distances.append(distance_m)
    observable_count = sum(1 for distance_m in distances if distance_m <= padding_m)
    return {
        "selected_count": int(selection.get("selected_count") or 0),
        "padding_m": padding_m,
        "nearest_distance_m": min(distances) if distances else float("inf"),
        "observable_selected_count": observable_count,
        "has_observable_selection": observable_count > 0,
    }


def validate_episode(episode_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = episode_dir / "episode_manifest.json"
    truth_path = episode_dir / "truth_frames.jsonl"
    roster_path = episode_dir / "global_entity_roster.json"
    for path in (manifest_path, truth_path, roster_path):
        if not path.exists():
            return {"episode_id": episode_dir.name, "ok": False, "errors": [f"missing {path.name}"], "warnings": []}

    manifest = read_json(manifest_path)
    episode_id = str(manifest.get("episode_id") or episode_dir.name)
    scenario_id = str(manifest.get("scenario_id") or episode_id.split("__seed")[0])
    seed = int(manifest.get("seed", -1))
    if seed not in (0, 1, 2):
        errors.append(f"seed must be 0, 1, or 2; found {seed}")
        seed = max(0, min(2, seed))
    expected_segment = segment_for_seed(seed)
    sumo_manifest = dict(manifest.get("sumo_traffic") or {})
    if sumo_manifest.get("enabled") is not True:
        errors.append("manifest.sumo_traffic.enabled is not true")
    manifest_segment = dict(sumo_manifest.get("segment") or {})
    if float(manifest_segment.get("segment_start_s", -1.0)) != expected_segment.segment_start_s:
        errors.append("manifest SUMO segment_start_s does not match seed")
    if float(manifest_segment.get("segment_end_s", -1.0)) != expected_segment.segment_end_s:
        errors.append("manifest SUMO segment_end_s does not match seed")
    observable_selection = _observable_sumo_selection_summary(sumo_manifest)
    has_observable_sumo_selection = bool(observable_selection["has_observable_selection"])

    roster = read_json(roster_path)
    roster_entities = list(roster.get("entities") or [])
    roster_sumo_count = sum(1 for entity in roster_entities if str(entity.get("source") or "") == "sumo_traci")
    if roster_sumo_count <= 0 and has_observable_sumo_selection:
        errors.append("global roster has no SUMO traffic vehicles")

    frame_count = 0
    last_tick = -1
    traffic_light_count_min = 10**9
    traffic_light_count_max = 0
    source_vehicle_counts: list[int] = []
    active_selected_counts: list[int] = []
    active_incident_frame_count = 0
    category_presence: defaultdict[str, int] = defaultdict(int)
    sumo_entity_frame_count = 0
    sumo_motion: dict[str, dict[str, Any]] = {}
    max_observation_distance_m = 0.0

    for frame in iter_jsonl(truth_path):
        frame_count += 1
        tick = int(frame.get("tick", -1))
        last_tick = tick
        if tick != frame_count - 1:
            errors.append(f"non-contiguous tick at frame {frame_count}: {tick}")
            break
        if int(frame.get("tick_hz") or 0) != DEFAULT_TICK_HZ:
            errors.append(f"tick_hz must be {DEFAULT_TICK_HZ}; found {frame.get('tick_hz')}")
        if abs(float(frame.get("sim_time_s") or 0.0) - tick / float(DEFAULT_TICK_HZ)) > 1e-6:
            errors.append(f"sim_time_s does not match tick at tick {tick}")
        segment = dict(frame.get("sumo_segment") or {})
        if float(segment.get("segment_start_s", -1.0)) != expected_segment.segment_start_s:
            errors.append(f"frame {tick} SUMO segment_start_s does not match seed")
        sumo_semantics = dict(frame.get("sumo_semantics") or {})
        if sumo_semantics.get("enabled") is not True:
            errors.append(f"frame {tick} missing enabled sumo_semantics")
        expected_absolute = expected_segment.segment_start_s + tick / float(DEFAULT_TICK_HZ)
        absolute_value = sumo_semantics.get("absolute_time_s")
        if absolute_value is None or abs(float(absolute_value) - expected_absolute) > 1e-6:
            errors.append(f"frame {tick} SUMO absolute_time_s mismatch")
        traffic_lights = dict(frame.get("sumo_traffic_light_states") or {})
        traffic_light_count_min = min(traffic_light_count_min, len(traffic_lights))
        traffic_light_count_max = max(traffic_light_count_max, len(traffic_lights))
        source_vehicle_counts.append(int(sumo_semantics.get("source_vehicle_count") or 0))
        active_selected_counts.append(int(sumo_semantics.get("active_selected_vehicle_count") or 0))
        if frame.get("sumo_active_incidents"):
            active_incident_frame_count += 1
            for incident in frame.get("sumo_active_incidents") or []:
                if str(incident.get("episode_scenario_id") or "") != scenario_id:
                    errors.append(f"frame {tick} has SUMO incident for another scenario")

        for entity in frame.get("entities") or []:
            category = str(entity.get("entity_category") or "")
            if category:
                category_presence[category] += 1
            pose = dict(entity.get("truth_pose") or {})
            position = pose.get("position_enu_m")
            if not _finite_position(position):
                errors.append(f"frame {tick} entity {entity.get('entity_id')} has invalid truth position")
            if str(entity.get("source") or "") != "sumo_traci":
                continue
            sumo_entity_frame_count += 1
            entity_id = str(entity.get("entity_id") or "")
            visibility = dict(entity.get("sumo_visibility") or {})
            distance_m = float(visibility.get("inspect_observation_distance_m") or 0.0)
            max_observation_distance_m = max(max_observation_distance_m, distance_m)
            if distance_m > MAX_SUMO_OBSERVATION_DISTANCE_M:
                errors.append(
                    f"frame {tick} SUMO vehicle {entity_id} observation distance {distance_m:.3f}m "
                    f"exceeds {MAX_SUMO_OBSERVATION_DISTANCE_M:.1f}m"
                )
            motion = sumo_motion.setdefault(
                entity_id,
                {
                    "first": position,
                    "last": position,
                    "max_speed": 0.0,
                    "frames": 0,
                    "control_role": str((entity.get("sumo_vehicle") or {}).get("control_role") or ""),
                },
            )
            motion["last"] = position
            motion["frames"] += 1
            velocity = (pose.get("velocity_enu_mps") or [0.0, 0.0, 0.0])
            if isinstance(velocity, list) and len(velocity) >= 2:
                try:
                    motion["max_speed"] = max(motion["max_speed"], math.hypot(float(velocity[0]), float(velocity[1])))
                except (TypeError, ValueError):
                    pass

    if frame_count != DEFAULT_DURATION_TICKS + 1:
        errors.append(f"truth frame count must be {DEFAULT_DURATION_TICKS + 1}; found {frame_count}")
    if last_tick != DEFAULT_DURATION_TICKS:
        errors.append(f"last tick must be {DEFAULT_DURATION_TICKS}; found {last_tick}")
    if traffic_light_count_min != DEFAULT_TRAFFIC_LIGHT_COUNT or traffic_light_count_max != DEFAULT_TRAFFIC_LIGHT_COUNT:
        errors.append(
            f"traffic light count must stay {DEFAULT_TRAFFIC_LIGHT_COUNT}; "
            f"min={traffic_light_count_min}, max={traffic_light_count_max}"
        )
    for required_category in ("uav", "pedestrian", "vehicle"):
        if category_presence[required_category] <= 0:
            errors.append(f"truth has no {required_category} entities")
    if sumo_entity_frame_count <= 0 and has_observable_sumo_selection:
        errors.append("truth has no active SUMO vehicles")
    moving_background = 0
    for motion in sumo_motion.values():
        first = motion.get("first") or [0.0, 0.0, 0.0]
        last = motion.get("last") or first
        span = math.hypot(float(last[0]) - float(first[0]), float(last[1]) - float(first[1]))
        if str(motion.get("control_role") or "") == "background" and (span >= 2.0 or float(motion.get("max_speed") or 0.0) >= 0.5):
            moving_background += 1
    if moving_background <= 0 and has_observable_sumo_selection and active_incident_frame_count <= 0:
        errors.append("truth has no moving SUMO background vehicle")
    if not has_observable_sumo_selection:
        nearest = float(observable_selection["nearest_distance_m"])
        nearest_label = f"{nearest:.3f}m" if math.isfinite(nearest) else "none"
        warnings.append(
            "no selected SUMO vehicle enters the episode visibility geometry "
            f"(padding={float(observable_selection['padding_m']):.1f}m, nearest={nearest_label})"
        )
    elif moving_background <= 0 and active_incident_frame_count > 0:
        warnings.append("SUMO traffic appears only as incident-controlled vehicles inside visibility geometry")

    source_min = min(source_vehicle_counts) if source_vehicle_counts else 0
    source_max = max(source_vehicle_counts) if source_vehicle_counts else 0
    source_first = source_vehicle_counts[0] if source_vehicle_counts else 0
    source_last = source_vehicle_counts[-1] if source_vehicle_counts else 0
    if seed == 0 and not (source_first <= 2 and source_max >= 180):
        errors.append(f"seed00 must cover ramp-up traffic; first={source_first}, max={source_max}")
    if seed == 1 and not (source_min >= 180 and source_max >= 190):
        errors.append(f"seed01 must cover peak traffic; min={source_min}, max={source_max}")
    if seed == 2 and not (source_first >= 180 and source_last <= 5):
        errors.append(f"seed02 must cover ramp-down traffic; first={source_first}, last={source_last}")

    overlaps_incident = _scenario_incident_overlaps_seed(manifest, seed)
    if overlaps_incident and active_incident_frame_count <= 0:
        errors.append("scenario SUMO incident overlaps this seed but no active incident appears in truth")
    if not overlaps_incident and active_incident_frame_count > 0:
        errors.append("truth has active SUMO incident outside this scenario/seed contract")

    if roster_sumo_count > 0 and sumo_entity_frame_count == 0:
        warnings.append("SUMO roster vehicles exist but never appear in truth frames")

    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "frame_count": frame_count,
            "traffic_light_count_min": traffic_light_count_min,
            "traffic_light_count_max": traffic_light_count_max,
            "source_vehicle_count_first": source_first,
            "source_vehicle_count_max": source_max,
            "source_vehicle_count_last": source_last,
            "active_selected_vehicle_count_max": max(active_selected_counts) if active_selected_counts else 0,
            "sumo_roster_vehicle_count": roster_sumo_count,
            "sumo_entity_frame_count": sumo_entity_frame_count,
            "moving_background_sumo_vehicles": moving_background,
            "active_incident_frame_count": active_incident_frame_count,
            "max_observation_distance_m": round(max_observation_distance_m, 6),
            "sumo_observable_selected_vehicle_count": int(observable_selection["observable_selected_count"]),
            "sumo_nearest_selected_distance_m": (
                round(float(observable_selection["nearest_distance_m"]), 6)
                if math.isfinite(float(observable_selection["nearest_distance_m"]))
                else None
            ),
        },
    }


def validate_all(render_ready_root: Path, scenarios_root: Path) -> dict[str, Any]:
    scenarios = scenario_ids(scenarios_root)
    errors: list[str] = []
    episode_results: list[dict[str, Any]] = []
    for scenario_id in scenarios:
        for seed in range(3):
            episode_dir = render_ready_root / f"{scenario_id}__seed{seed:02d}"
            if not episode_dir.exists():
                errors.append(f"missing render-ready episode: {episode_dir.name}")
                continue
            result = validate_episode(episode_dir)
            episode_results.append(result)
            if not result["ok"]:
                for error in result["errors"]:
                    errors.append(f"{episode_dir.name}: {error}")
    return {
        "ok": not errors,
        "scenario_count": len(scenarios),
        "expected_episode_count": len(scenarios) * 3,
        "validated_episode_count": len(episode_results),
        "errors": errors,
        "episodes": episode_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SUMO truth integration in render-ready episodes.")
    parser.add_argument("--render-ready-root", type=Path, default=DEFAULT_RENDER_READY_ROOT)
    parser.add_argument("--scenarios-root", type=Path, default=DEFAULT_SCENARIOS_ROOT)
    parser.add_argument("--episode", type=Path, help="Validate one render-ready episode directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episode:
        result = validate_episode(args.episode.resolve())
    else:
        result = validate_all(args.render_ready_root.resolve(), args.scenarios_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
