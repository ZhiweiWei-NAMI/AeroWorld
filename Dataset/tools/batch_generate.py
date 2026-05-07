"""Generate Dataset/episodes from grounded scene_setup/event_script pairs.

The render pipeline consumes Dataset/episodes, so this generator must keep those
episodes aligned with the scenario files that validation checks. It uses
scene_setup.json as the authoritative entity/asset/initial-pose source and
event_script.json as the event/motion source.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from pedestrian_activity_catalog import get_activity, normalize_activity_type


TICK_HZ = 10
DEFAULT_DURATION_TICKS = 900


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def project_relative(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(dataset_root.parent.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolve_param(value: Any, params: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$param."):
        return params.get(value[len("$param.") :], value)
    if isinstance(value, list):
        return [resolve_param(item, params) for item in value]
    if isinstance(value, dict):
        return {key: resolve_param(item, params) for key, item in value.items()}
    return value


def vector3(value: Any, default: Sequence[float] = (0.0, 0.0, 0.0)) -> list[float]:
    values = list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []
    return [
        float(values[0] if len(values) > 0 else default[0]),
        float(values[1] if len(values) > 1 else default[1]),
        float(values[2] if len(values) > 2 else default[2]),
    ]


def distance3(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def label_class_for(entity_id: str, logical_asset_id: str, category: str) -> str:
    category_text = str(category).lower()
    asset_text = str(logical_asset_id).lower()
    text = f"{entity_id} {asset_text} {category_text}".lower()
    if category_text == "uav" or asset_text.startswith("uav."):
        return "uav"
    if category_text == "vehicle" or asset_text.startswith("vehicle."):
        return "vehicle"
    if category_text == "pedestrian" or asset_text.startswith("pedestrian."):
        return "pedestrian"
    if "signal_light" in asset_text:
        return "traffic_light"
    if "base_tower" in asset_text or "tower" in text or "station" in text:
        return "radio_tower"
    if "charger" in asset_text:
        return "charging_pile"
    if asset_text.startswith("trigger.") or "no_fly" in text or "hazard" in text or "zone" in text:
        return "no_fly_zone"
    if asset_text.startswith("facility."):
        return "facility"
    if asset_text.startswith("prop."):
        return "prop"
    return "other"


def position_from_placement(entity: dict[str, Any]) -> list[float]:
    placement = dict(entity.get("placement") or {})
    mode = str(entity.get("placement_mode") or "").lower()
    for key in ("resolved_position_enu_m", "position_enu_m", "center_enu_m"):
        if isinstance(placement.get(key), list):
            return vector3(placement[key])
    if mode == "polygon_prism" and isinstance(placement.get("polygon_enu_m"), list):
        points = [vector3(point) for point in placement["polygon_enu_m"] if isinstance(point, list)]
        if points:
            return [
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
                float(placement.get("base_z_m", points[0][2])),
            ]
    logical_asset_id = str(entity.get("logical_asset_id") or "")
    category = str(entity.get("category") or "")
    if logical_asset_id.startswith("pedestrian.") or category == "pedestrian":
        raise RuntimeError(f"Pedestrian entity lacks resolved placement: {entity.get('entity_id')}")
    return [50.0, 20.0, 0.0]


def event_ticks(script: dict[str, Any]) -> dict[str, int]:
    params = dict(script.get("parameters") or {})
    triggers = {str(trigger.get("trigger_id")): dict(trigger) for trigger in script.get("triggers", [])}
    events = list(script.get("events") or [])
    resolved: dict[str, int] = {}

    def trigger_tick(trigger_id: str, stack: set[str] | None = None) -> int:
        stack = stack or set()
        if trigger_id in stack:
            return 0
        trigger = triggers.get(trigger_id) or {}
        ttype = str(trigger.get("type") or "")
        if ttype == "tick":
            value = resolve_param(trigger.get("tick", 0), params)
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0
        if ttype == "event_fired_after":
            return resolved.get(str(trigger.get("event_id") or ""), 0) + int(trigger.get("delay_ticks") or 0)
        if ttype == "event_fired":
            return resolved.get(str(trigger.get("event_id") or ""), 0) + 1
        if ttype == "composite":
            child_ticks = [trigger_tick(str(child), stack | {trigger_id}) for child in trigger.get("children", [])]
            return max(child_ticks) if child_ticks else 0
        if ttype == "weather_state":
            return 300
        if ttype == "entity_proximity":
            return 300
        return 0

    for _ in range(max(1, len(events))):
        changed = False
        for event in events:
            event_id = str(event.get("event_id") or "")
            if not event_id:
                continue
            tick = trigger_tick(str(event.get("trigger_ref") or ""))
            if resolved.get(event_id) != tick:
                resolved[event_id] = tick
                changed = True
        if not changed:
            break
    return resolved


def _append_frame(frames: list[tuple[int, list[float], str]], tick: int, pos: list[float], state: str) -> None:
    state = normalize_activity_type(state)
    if frames and frames[-1][0] == tick:
        frames[-1] = (tick, list(pos), state)
        return
    frames.append((tick, list(pos), state))


def keyframes_for(
    initial_pos: list[float],
    schedules: list[dict[str, Any]],
    initial_state: str,
) -> list[tuple[int, list[float], str]]:
    current_activity = normalize_activity_type(initial_state)
    frames = [(0, list(initial_pos), current_activity)]
    current_pos = list(initial_pos)
    current_tick = 0
    for schedule in sorted(schedules, key=lambda item: int(item.get("tick", 0))):
        schedule_type = str(schedule.get("type") or "move")
        start_tick = max(current_tick, int(schedule.get("tick", 0)))
        if start_tick > current_tick:
            _append_frame(frames, start_tick, current_pos, current_activity)
            current_tick = start_tick
        if schedule_type == "activity":
            current_activity = normalize_activity_type(str(schedule.get("activity_type") or current_activity))
            _append_frame(frames, start_tick, current_pos, current_activity)
            continue
        velocity = max(0.1, float(schedule.get("velocity_mps") or 1.0))
        waypoints = list(schedule.get("waypoints_enu_m", []) or [])
        if not waypoints:
            continue
        moving_activity = normalize_activity_type(str(schedule.get("activity_type") or current_activity), moving=True)
        post_activity = normalize_activity_type(str(schedule.get("post_activity_type") or "waiting"))
        _append_frame(frames, start_tick, current_pos, moving_activity)
        for waypoint_index, waypoint in enumerate(waypoints):
            target = vector3(waypoint, current_pos)
            distance = distance3(current_pos, target)
            if distance <= 1e-6:
                continue
            current_tick += max(1, int(math.ceil(distance / velocity * TICK_HZ)))
            current_pos = target
            frame_activity = post_activity if waypoint_index == len(waypoints) - 1 else moving_activity
            _append_frame(frames, current_tick, current_pos, frame_activity)
            current_activity = frame_activity
    return frames


def sample_keyframes(frames: list[tuple[int, list[float], str]], tick: int) -> tuple[list[float], list[float], str]:
    if not frames:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], "idle"
    frames = sorted(frames, key=lambda item: item[0])
    if tick <= frames[0][0]:
        return list(frames[0][1]), [0.0, 0.0, 0.0], frames[0][2]
    for previous, current in zip(frames, frames[1:]):
        if previous[0] <= tick <= current[0]:
            span = max(1, current[0] - previous[0])
            alpha = (tick - previous[0]) / float(span)
            pos = [previous[1][i] + (current[1][i] - previous[1][i]) * alpha for i in range(3)]
            dt_s = span / float(TICK_HZ)
            vel = [(current[1][i] - previous[1][i]) / dt_s for i in range(3)]
            if tick == current[0]:
                state = current[2]
            else:
                state = previous[2] if math.sqrt(sum(value * value for value in vel)) > 0.05 else current[2]
            return pos, vel, state
    return list(frames[-1][1]), [0.0, 0.0, 0.0], frames[-1][2]


def build_entities(scene_setup: dict[str, Any], script: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    for scene_entity in scene_setup.get("entities") or []:
        entity_id = str(scene_entity.get("entity_id") or scene_entity.get("instance_id") or "")
        if not entity_id:
            continue
        logical_asset_id = str(scene_entity.get("logical_asset_id") or "")
        category = str(scene_entity.get("category") or "")
        initial_state = dict(scene_entity.get("initial_state") or {})
        activation_tick = int(scene_entity.get("activation_tick") or 0)
        spawn_policy = str(scene_entity.get("spawn_policy") or "").lower()
        initial_activity = (
            initial_state.get("activity_type")
            or (initial_state.get("state_facets") or {}).get("activity", {}).get("activity_type")
            or initial_state.get("mode")
            or "idle"
        )
        if logical_asset_id.startswith("pedestrian.") or category == "pedestrian":
            initial_activity = normalize_activity_type(str(initial_activity or "waiting"))
        entities[entity_id] = {
            "entity_id": entity_id,
            "pos": position_from_placement(scene_entity),
            "label_class": label_class_for(entity_id, logical_asset_id, category),
            "asset_id": logical_asset_id,
            "state": str(initial_activity or "idle"),
            "active_from": activation_tick if activation_tick > 0 or spawn_policy == "event_script_only" else 0,
            "schedules": [],
        }

    ticks = event_ticks(script)
    params = dict(script.get("parameters") or {})
    for event in script.get("events") or []:
        tick = int(ticks.get(str(event.get("event_id") or ""), 0))
        for action in event.get("actions") or []:
            atype = str(action.get("type") or "")
            entity_id = str(resolve_param(action.get("entity_id", action.get("ped_id", "")), params) or "")
            if not entity_id:
                continue
            if atype == "spawn_entity":
                asset_id = str(resolve_param(action.get("asset_id", ""), params) or "")
                position = vector3(resolve_param(action.get("position_enu_m", [50.0, 20.0, 0.0]), params))
                if entity_id not in entities:
                    raise RuntimeError(f"spawn_entity references undeclared entity {entity_id}")
                entities.setdefault(
                    entity_id,
                    {
                        "entity_id": entity_id,
                        "pos": position,
                        "label_class": label_class_for(entity_id, asset_id, ""),
                        "asset_id": asset_id,
                        "state": "idle",
                        "active_from": tick,
                        "schedules": [],
                    },
                )
            if atype == "move_entity":
                waypoints = resolve_param(action.get("waypoints_enu_m", []), params)
                if entity_id not in entities:
                    raise RuntimeError(f"move_entity references undeclared entity {entity_id}")
                label_class = str(entities[entity_id].get("label_class") or "")
                activity_type = action.get("activity_type")
                post_activity_type = action.get("post_activity_type")
                if label_class == "pedestrian":
                    activity_type = normalize_activity_type(str(activity_type or "walking"), moving=True)
                    post_activity_type = normalize_activity_type(str(post_activity_type or "waiting"))
                entities[entity_id]["schedules"].append(
                    {
                        "type": "move",
                        "tick": tick,
                        "waypoints_enu_m": waypoints if isinstance(waypoints, list) else [],
                        "velocity_mps": action.get("velocity_mps", 1.0),
                        "activity_type": activity_type,
                        "post_activity_type": post_activity_type,
                    }
                )
            if atype == "set_pedestrian_activity":
                if entity_id not in entities:
                    raise RuntimeError(f"set_pedestrian_activity references undeclared entity {entity_id}")
                activity_type = normalize_activity_type(str(action.get("activity_type") or "waiting"))
                entities[entity_id]["schedules"].append(
                    {
                        "type": "activity",
                        "tick": tick,
                        "activity_type": activity_type,
                    }
                )
            if atype == "play_animation" and entity_id not in entities:
                raise RuntimeError(f"play_animation references undeclared entity {entity_id}")
    return entities


def generate_trajectories(scene_setup: dict[str, Any], script: dict[str, Any], duration_ticks: int) -> list[dict[str, Any]]:
    entities = build_entities(scene_setup, script)
    keyframes = {
        entity_id: keyframes_for(vector3(entity["pos"]), list(entity.get("schedules") or []), str(entity.get("state") or "idle"))
        for entity_id, entity in entities.items()
    }
    rows: list[dict[str, Any]] = []
    for tick in range(0, duration_ticks + 1, 10):
        for entity_id, entity in sorted(entities.items()):
            if tick < int(entity.get("active_from") or 0):
                pos = vector3(entity["pos"])
                vel = [0.0, 0.0, 0.0]
                state = "offstage"
            else:
                pos, vel, state = sample_keyframes(keyframes[entity_id], tick)
            rows.append(
                {
                    "tick": tick,
                    "entity_id": entity_id,
                    "label_class": entity["label_class"],
                    "asset_id": entity["asset_id"],
                    "pos_enu": [round(float(value), 6) for value in pos],
                    "vel_mps": [round(float(value), 6) for value in vel],
                    "state": state,
                    "activity_type": get_activity(state).activity_type if entity["label_class"] == "pedestrian" else state,
                    "animation_hint": get_activity(state).animation_hint if entity["label_class"] == "pedestrian" else state,
                    "posture": get_activity(state).posture if entity["label_class"] == "pedestrian" else "standing",
                    "social_state": get_activity(state).social_state if entity["label_class"] == "pedestrian" else "solo",
                }
            )
    return rows


def generate_weather(script: dict[str, Any], duration_ticks: int) -> list[dict[str, Any]]:
    current = {"rain": 0.0, "fog": 0.0, "wind_speed": 2.0, "visibility_m": 20000.0}
    params = dict(script.get("parameters") or {})
    changes: dict[int, dict[str, Any]] = {}
    trigger_by_id = {str(t.get("trigger_id")): dict(t) for t in script.get("triggers") or []}
    for event in script.get("events") or []:
        trigger = trigger_by_id.get(str(event.get("trigger_ref") or "")) or {}
        tick = 0
        if str(trigger.get("type") or "") == "tick":
            try:
                tick = int(resolve_param(trigger.get("tick", 0), params))
            except (TypeError, ValueError):
                tick = 0
        for action in event.get("actions") or []:
            if action.get("type") == "set_weather":
                changes.setdefault(tick, {}).update(dict(action.get("overrides") or {}))
    for trigger in script.get("triggers") or []:
        if trigger.get("type") != "weather_state":
            continue
        parameter = str(trigger.get("parameter") or "")
        if parameter:
            changes.setdefault(300, {})[parameter] = resolve_param(trigger.get("value", 0.0), params)

    rows: list[dict[str, Any]] = []
    for tick in range(0, duration_ticks + 1, 10):
        if tick in changes:
            current.update(changes[tick])
        rows.append({"tick": tick, **current})
    return rows


def generate_episode(script_path: Path, episode_dir: Path, dataset_root: Path, seed: int = 0, skip_graphs: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)

    script = read_json(script_path)
    scene_setup = read_json(script_path.with_name("scene_setup.json")) if script_path.with_name("scene_setup.json").exists() else {}
    scenario_id = str(script.get("scenario_id") or script_path.parent.name)
    duration = int(script.get("parameters", {}).get("duration_ticks") or DEFAULT_DURATION_TICKS)

    trajectories = generate_trajectories(scene_setup, script, duration)
    weather = generate_weather(script, duration)
    write_jsonl(episode_dir / "trajectories.jsonl", trajectories)
    write_jsonl(episode_dir / "weather_meta.jsonl", weather)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Plugins" / "SumoImporter" / "Scripts"))
    from donghu_core.event_script_interpreter import EventScriptInterpreter

    by_tick: dict[int, list[dict[str, Any]]] = {}
    for row in trajectories:
        by_tick.setdefault(int(row["tick"]), []).append(row)
    weather_by_tick = {int(row["tick"]): row for row in weather}
    interpreter = EventScriptInterpreter(script_path)
    for tick in range(duration + 1):
        for row in by_tick.get(tick, []):
            interpreter.update_entity_state(row["entity_id"], row["pos_enu"], {}, row["vel_mps"])
        if tick in weather_by_tick:
            interpreter.update_weather_state(weather_by_tick[tick])
        interpreter.tick(tick)
    event_log = interpreter.get_event_log()
    write_jsonl(episode_dir / "event_trace.jsonl", event_log)

    roster: dict[str, dict[str, Any]] = {}
    for row in trajectories:
        roster.setdefault(
            row["entity_id"],
            {
                "entity_id": row["entity_id"],
                "label_class": row["label_class"],
                "asset_id": row["asset_id"],
            },
        )
    write_json(episode_dir / "global_entity_roster.json", roster)

    manifest = {
        "episode_id": episode_dir.name,
        "scenario_id": scenario_id,
        "duration_ticks": duration,
        "seed": seed,
        "n_events": len(event_log),
        "n_entities": len(roster),
        "source_event_script_path": project_relative(script_path, dataset_root),
        "source_scene_setup_path": project_relative(script_path.with_name("scene_setup.json"), dataset_root),
        "generator": "Dataset/tools/batch_generate.py",
    }
    write_json(episode_dir / "episode_manifest.json", manifest)

    summary: dict[str, Any] = {"n_frames": len({row["tick"] for row in trajectories}), "n_unique_entities": len(roster)}
    if not skip_graphs:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from generate_graph_labels import generate_frame_graphs

        frame_graphs, summary = generate_frame_graphs(episode_dir)
        graphs_dir = episode_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(graphs_dir / "frame_graphs.jsonl", frame_graphs)
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate grounded episodes")
    parser.add_argument("--dataset-root", default="Dataset")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--episode", action="append", default=[], help="Scenario id / episode name to generate. Repeatable.")
    parser.add_argument("--skip-graphs", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    scenario_scripts = sorted((dataset_root / "scenarios").rglob("event_script.json"))
    if args.episode:
        wanted = {str(value).replace("__seed00", "") for value in args.episode}
        scenario_scripts = [path for path in scenario_scripts if path.parent.name in wanted or str(read_json(path).get("scenario_id")) in wanted]
    print(f"Found {len(scenario_scripts)} scenarios")

    total_episodes = 0
    total_events = 0
    for script_path in scenario_scripts:
        scenario_name = script_path.parent.name
        for seed in range(max(1, int(args.seeds))):
            episode_dir = dataset_root / "episodes" / f"{scenario_name}__seed{seed:02d}"
            manifest, summary = generate_episode(script_path, episode_dir, dataset_root, seed, bool(args.skip_graphs))
            total_episodes += 1
            total_events += int(manifest["n_events"])
            print(
                f"  [OK] {episode_dir.name}: {manifest['n_events']} events, "
                f"{summary.get('n_frames', 0)} frames, {summary.get('n_unique_entities', 0)} entities"
            )
    print(f"\nGenerated {total_episodes} episodes with {total_events} total events")


if __name__ == "__main__":
    main()
