"""Shared measured-state snapshot and semantic-chain helpers used by L5."""
from __future__ import annotations
import copy
from dataclasses import asdict
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "Dataset/tools", ROOT / "Plugins/SumoImporter/Scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from batch_generate import EpisodeStateEngine, TICK_HZ
from donghu_core.event_script_interpreter import EventScriptInterpreter
from Dataset.semantic_truth.outcome_projection import recovery_tick_for_chain

def snapshot(engine, interpreter, tick, sampled_states):
    """Actual sampled states and execution internals, not a standalone restore API."""
    return {
        "tick": tick, "phase": "after_state_sample_before_actions",
        "sampled_states": copy.deepcopy(sampled_states),
        "entity_definitions": copy.deepcopy(engine.entities),
        "keyframes": copy.deepcopy(engine.keyframes),
        "weather": copy.deepcopy(engine.weather),
        "pending_runtime_state_patches": copy.deepcopy(engine.pending_runtime_state_patches),
        "last_yaw_by_entity": copy.deepcopy(engine.last_yaw_by_entity),
        "interpreter": {
            "trigger_states": {k: asdict(v) for k, v in interpreter.trigger_states.items()},
            "event_states": {k: asdict(v) for k, v in interpreter.event_states.items()},
            "entity_states": {k: asdict(v) for k, v in interpreter.entity_states.items()},
            "emitted_events": sorted(interpreter.emitted_events),
            "entity_activity_states": copy.deepcopy(interpreter.entity_activity_states),
            "weather_state": copy.deepcopy(interpreter.weather_state),
        },
    }

def build_chain(semantic, *, window_start_tick, window_end_tick):
    nodes, relations = [], []
    for transition in semantic["predicate_transitions"]:
        nodes.append({"node_id": transition["transition_id"], "kind": "predicate_transition",
                      "tick": transition["to_tick"], "evidence_ref":
                      "objective/predicate_transitions.jsonl#" + transition["transition_id"]})
    for event in semantic["event_occurrences"]:
        nodes.append({"node_id": event["event_id"], "kind": "objective_event",
                      "tick": event["trigger_tick"], "evidence_ref":
                      "objective/event_occurrences.jsonl#" + event["event_id"]})
        for phase in event["event_phases"]:
            relations.append({"source": phase["transition_id"], "target": event["event_id"],
                              "relation_type": "predicate_" + phase["phase_kind"] + "_evidence",
                              "tick": phase["tick"], "causal_claim": False,
                              "rule_id": event["rule_id"]})
    for outcome in semantic["event_outcomes"]:
        recovery_tick = recovery_tick_for_chain(
            outcome, window_start_tick=window_start_tick,
            window_end_tick=window_end_tick)
        if recovery_tick is not None:
            nodes.append({"node_id": outcome["outcome_id"], "kind": "event_recovery_confirmed",
                          "tick": recovery_tick, "evidence_ref":
                          "objective/event_outcomes.jsonl#" + outcome["outcome_id"]})
            relations.append({"source": outcome["event_id"], "target": outcome["outcome_id"],
                              "relation_type": "has_observed_outcome", "causal_claim": False,
                              "tick": recovery_tick})
    return sorted(nodes, key=lambda r: (r["tick"], r["node_id"])), relations
