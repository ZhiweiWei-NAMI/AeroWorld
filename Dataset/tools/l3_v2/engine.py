"""Run one explicitly declared L3 mechanism on the shipped engine.

The engine is deterministic given (scene, script, in-memory mechanism).  A
mechanism is exactly one of: retime a trigger, omit authored actions, inject
explicit actions at a tick, or override a runtime-state patch already authored in
the script.  Nothing is inferred from an arm id or an intent name.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from Dataset.tools.l3_v2.common import ROOT, load, project_path  # noqa: F401 (sys.path)
from Dataset.tools.batch_generate import EpisodeStateEngine
from donghu_core.event_script_interpreter import EventScriptInterpreter


def handlers(engine: EpisodeStateEngine) -> dict[str, Any]:
    return {
        "move_entity": engine._handle_move_entity,
        "set_visual_state": engine._handle_set_visual_state,
        "set_runtime_state": engine._handle_set_runtime_state,
        "spawn_entity": engine._handle_spawn_entity,
        "set_weather": engine._handle_set_weather,
        "set_pedestrian_activity": engine._handle_set_pedestrian_activity,
        "remove_entity": engine._handle_remove_entity,
        "capture_screenshot": engine._handle_capture_screenshot,
    }


def _rewrite_trigger(interpreter: EventScriptInterpreter, retime: Mapping[str, Any]) -> dict:
    """Move one declared trigger threshold.  The field must exist on that trigger kind."""
    trigger_id = str(retime["trigger_id"])
    field = str(retime["field"])
    delta = int(retime["delta"])
    for trigger in interpreter.script.get("triggers") or []:
        if str(trigger.get("trigger_id")) != trigger_id:
            continue
        if field not in trigger:
            raise ValueError(f"retime field {field!r} absent on trigger {trigger_id!r}")
        trigger[field] = int(trigger[field]) + delta
        return {"trigger_id": trigger_id, "field": field, "delta": delta,
                "new_value": trigger[field]}
    raise ValueError(f"retime target not found: {trigger_id}")


def _override_state_patch(interpreter: EventScriptInterpreter, override: Mapping[str, Any]) -> dict:
    """Replace one authored set_runtime_state value in the loaded script.

    Used where the layer's own predicate reads a typed runtime family that the
    author writes declaratively; the override is explicit and recorded.
    """
    action_id = str(override["action_id"])
    family = str(override["family"])
    field = str(override["field"])
    value = override["value"]
    for event in interpreter.script.get("events") or []:
        for action in event.get("actions") or []:
            if str(action.get("action_id")) != action_id:
                continue
            patch = action.get("state_patch")
            if not isinstance(patch, Mapping) or family not in patch or field not in patch[family]:
                raise ValueError(f"override target {action_id}.{family}.{field} not authored")
            action["state_patch"][family][field] = value
            return {"action_id": action_id, "family": family, "field": field, "value": value}
    raise ValueError(f"override target action not found: {action_id}")


def execute(scene_ref: str, script_ref: str, episode_id: str, *, end_tick: int = 900,
            retime: Mapping[str, Any] | None = None,
            state_overrides: Sequence[Mapping[str, Any]] = (),
            omit_action_ids: Sequence[str] = (),
            injections: Sequence[Mapping[str, Any]] = (),
            hold_at_fork: Mapping[str, Any] | None = None,
            snapshot_ticks: Sequence[int] = ()) -> dict[str, Any]:
    scene = load(project_path(scene_ref))
    script = load(project_path(script_ref))
    engine = EpisodeStateEngine(scene, script, project_path(script_ref), end_tick)
    interpreter = EventScriptInterpreter(project_path(script_ref), episode_id=episode_id)

    retimed = _rewrite_trigger(interpreter, retime) if retime else None
    overrides = [_override_state_patch(interpreter, item) for item in state_overrides]
    if retime or overrides:
        # The engine simulates the same in-memory script the interpreter evaluates.
        engine.script = copy.deepcopy(interpreter.script)

    omit = set(omit_action_ids)
    handler_map = handlers(engine)
    clock = [0]
    audit: list[dict[str, Any]] = []
    snapshots: dict[int, dict] = {}

    def dispatch_source(action: Mapping[str, Any]) -> dict[str, Any]:
        resolved = copy.deepcopy(dict(action))
        if resolved.get("action_id") in omit:
            audit.append({"tick": clock[0], "origin": "source_script", "action": resolved,
                          "result": {"status": "omitted_by_configuration"}})
            return {"status": "ok"}
        result = handler_map[resolved["type"]](resolved, clock[0])
        audit.append({"tick": clock[0], "origin": "source_script",
                      "action": copy.deepcopy(resolved), "result": copy.deepcopy(result)})
        if result.get("status") != "ok":
            raise RuntimeError(result)
        return result

    for action_type in handler_map:
        interpreter.register_handler(action_type, dispatch_source)

    for tick in range(end_tick + 1):
        clock[0] = tick
        engine._apply_pending_runtime_state_patches(tick)
        for weather in engine.weather_transitions.get(tick, []):
            engine.weather = weather
        current = engine._record_tick_rows(tick)
        # Region activation is part of proximity participation. An inactive
        # restriction remains in render/semantic truth, but is not an active
        # obstacle capable of triggering an avoidance alert.
        trigger_rows = [row for row in current
                        if row.get("incident_state", {}).get("temporary_lockdown_active") is not False]
        present = {row["entity_id"] for row in trigger_rows}
        # Proximity must use this tick's live entities. Retaining the last pose
        # of a removed region otherwise permits alerts against a vanished region.
        for entity_id in interpreter.entity_states.keys() - present:
            del interpreter.entity_states[entity_id]
            interpreter.entity_activity_states.pop(entity_id, None)
        for row in trigger_rows:
            interpreter.update_entity_state(row["entity_id"], row["pos_enu"], {}, row["vel_mps"])
            if row.get("label_class") == "pedestrian":
                interpreter.update_entity_activity(row["entity_id"], row["activity_type"])
        interpreter.update_weather_state(engine.weather_rows[-1])
        for row in current:
            if row.get("label_class") == "pedestrian":
                row["pedestrian_state"] = {**row.get("pedestrian_state", {}),
                    "evacuation_active": row.get("activity_type") == "evacuating"}
        if tick in snapshot_ticks:
            snapshots[tick] = snapshot(engine, interpreter, tick, current)
        for fired in interpreter.tick(tick):
            if fired["result"].get("status") != "ok":
                raise RuntimeError(fired)
        for injection in injections:
            if int(injection["tick"]) != tick:
                continue
            action = copy.deepcopy({k: v for k, v in injection.items() if k != "tick"})
            result = handler_map[action["type"]](action, tick)
            audit.append({"tick": tick, "origin": "intervention", "action": copy.deepcopy(action),
                          "result": copy.deepcopy(result)})
            if result.get("status") != "ok":
                raise RuntimeError(result)
    return {"engine": engine, "interpreter": interpreter, "audit": audit,
            "retimed_trigger": retimed, "state_overrides": overrides,
            "omitted_action_ids": sorted(omit), "snapshots": snapshots}


def snapshot(engine: EpisodeStateEngine, interpreter: EventScriptInterpreter, tick: int,
             sampled_states: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tick": tick,
        "phase": "after_state_sample_before_actions",
        "sampled_states": copy.deepcopy(sampled_states),
        "entity_definitions": copy.deepcopy(engine.entities),
        "keyframes": copy.deepcopy(engine.keyframes),
        "weather": copy.deepcopy(engine.weather),
        "pending_runtime_state_patches": copy.deepcopy(engine.pending_runtime_state_patches),
        "last_yaw_by_entity": copy.deepcopy(engine.last_yaw_by_entity),
        "interpreter": {
            "trigger_states": {k: asdict(v) for k, v in interpreter.trigger_states.items()},
            "event_states": {k: asdict(v) for k, v in interpreter.event_states.items()},
            "emitted_events": sorted(interpreter.emitted_events),
            "entity_activity_states": copy.deepcopy(interpreter.entity_activity_states),
            "weather_state": copy.deepcopy(interpreter.weather_state),
        },
    }
