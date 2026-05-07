"""Validate that event_script chains are reachable with mocked runtime motion.

This is intentionally stricter than schema validation and lighter than UE PIE:
move_entity advances the interpreter entity state to the final waypoint,
set_weather mutates the weather state, and the script is stepped until all
events fire or the horizon is exhausted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "Dataset"
SUMO_SCRIPTS_DIR = ROOT / "Plugins" / "SumoImporter" / "Scripts"
if str(SUMO_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SUMO_SCRIPTS_DIR))

from donghu_core.event_script_interpreter import EventScriptInterpreter  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def pos3(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return None


def entity_pos(entity: dict[str, Any]) -> list[float] | None:
    placement = entity.get("placement") or {}
    for key in ("resolved_position_enu_m", "position_enu_m", "center_enu_m"):
        found = pos3(placement.get(key))
        if found:
            return found
    polygon = placement.get("polygon_enu_m") or []
    if polygon:
        return [
            sum(float(point[0]) for point in polygon) / len(polygon),
            sum(float(point[1]) for point in polygon) / len(polygon),
            float(placement.get("base_z_m", 0.0)),
        ]
    return None


def weather_payload_from_profile(profile: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if profile == "rain":
        payload["rain"] = 0.6
    elif profile == "fog":
        payload["fog"] = 0.6
        payload["fog_density"] = 0.6
    elif profile == "wind":
        payload["wind_speed"] = 12.5
    payload["profile"] = profile
    for key, value in (overrides or {}).items():
        payload[key] = value
        if key == "fog":
            payload["fog_density"] = value
        elif key == "visibility_m":
            payload["visibility"] = value
    return payload


def horizon_ticks(script: dict[str, Any]) -> int:
    max_tick = 0
    total_delay = 0
    for trigger in script.get("triggers", []):
        if trigger.get("type") == "tick":
            max_tick = max(max_tick, int(trigger.get("tick", 0) or 0))
        if trigger.get("type") == "event_fired_after":
            total_delay += int(trigger.get("delay_ticks", 0) or 0)
    return max(900, max_tick + total_delay + 300)


def validate_scene(scene_path: Path) -> list[str]:
    scene = load_json(scene_path)
    script_path = scene_path.with_name("event_script.json")
    script = load_json(script_path)
    positions = {
        str(entity["entity_id"]): entity_pos(entity)
        for entity in scene.get("entities", [])
        if entity.get("entity_id") and entity_pos(entity)
    }
    weather = weather_payload_from_profile(str(scene.get("weather_profile", {}).get("initial") or "clear"))
    transitions_by_tick: dict[int, list[dict[str, Any]]] = {}
    for transition in scene.get("weather_profile", {}).get("transitions", []) or []:
        transitions_by_tick.setdefault(int(transition.get("tick", 0) or 0), []).append(transition)

    interpreter = EventScriptInterpreter(script_path, episode_id=f"reachability_{script.get('scenario_id', scene_path.parent.name)}")
    skipped_actions: list[dict[str, Any]] = []

    def sync_entity(entity_id: str) -> None:
        position = positions.get(entity_id)
        if position:
            interpreter.update_entity_state(entity_id, position, {}, (0.0, 0.0, 0.0))

    def handler(action: dict[str, Any]) -> dict[str, Any]:
        action_type = str(action.get("type") or "")
        entity_id = str(action.get("entity_id") or action.get("ped_id") or "")
        if action_type == "move_entity" and entity_id:
            waypoints = [pos3(point) for point in action.get("waypoints_enu_m", [])]
            waypoints = [point for point in waypoints if point]
            if waypoints:
                positions[entity_id] = waypoints[-1]
                sync_entity(entity_id)
        elif action_type == "spawn_entity" and entity_id:
            positions[entity_id] = pos3(action.get("position_enu_m")) or positions.get(entity_id, [0.0, 0.0, 0.0])
            sync_entity(entity_id)
        elif action_type == "remove_entity" and entity_id:
            positions.pop(entity_id, None)
        elif action_type == "set_weather":
            weather.update(weather_payload_from_profile(str(action.get("profile") or "clear"), dict(action.get("overrides") or {})))
            interpreter.update_weather_state(weather)
        elif action_type == "set_pedestrian_activity" and entity_id:
            interpreter.update_entity_activity(entity_id, str(action.get("activity_type") or ""))
        return {"status": "ok"}

    for action_type in (
        "spawn_entity",
        "move_entity",
        "remove_entity",
        "set_visual_state",
        "set_pedestrian_activity",
        "play_animation",
        "spawn_crowd",
        "clear_crowd",
        "set_weather",
        "capture_screenshot",
    ):
        interpreter.register_handler(action_type, handler)

    for tick in range(horizon_ticks(script) + 1):
        for transition in transitions_by_tick.get(tick, []):
            weather.update(weather_payload_from_profile(str(transition.get("profile") or "clear"), dict(transition.get("overrides") or {})))
        for entity_id in list(positions):
            sync_entity(entity_id)
        interpreter.update_weather_state(weather)
        for entry in interpreter.tick(tick):
            if (entry.get("result") or {}).get("status") == "skipped":
                skipped_actions.append(entry)

    fired_events = {
        event_id
        for event_id, state in interpreter.event_states.items()
        if state.fired
    }
    fired_events.update(
        str(row.get("event_id"))
        for row in interpreter.get_event_log()
        if row.get("event_id")
    )
    expected_events = {str(event["event_id"]) for event in script.get("events", [])}
    missing = sorted(expected_events - fired_events)
    issues: list[str] = []
    if missing:
        issues.append(f"{script['scenario_id']}: unreachable events: {', '.join(missing)}")
    if skipped_actions:
        issues.append(f"{script['scenario_id']}: skipped actions: {len(skipped_actions)}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate event_script reachability with mocked runtime handlers")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    scene_paths = sorted((Path(args.dataset_root).resolve() / "scenarios").rglob("scene_setup.json"))
    if args.limit:
        scene_paths = scene_paths[: args.limit]

    issues: list[str] = []
    for scene_path in scene_paths:
        issues.extend(validate_scene(scene_path))

    print("=" * 72)
    print("Event Reachability Validation")
    print("=" * 72)
    print(f"Scenarios checked: {len(scene_paths)}")
    print(f"Issues: {len(issues)}")
    if issues:
        for issue in issues[:200]:
            print(f"  - {issue}")
        if len(issues) > 200:
            print(f"  ... {len(issues) - 200} additional issues")
    else:
        print("All event chains are reachable under mocked runtime motion.")
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
