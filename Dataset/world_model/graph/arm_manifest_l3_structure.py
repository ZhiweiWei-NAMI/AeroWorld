"""Closed root grammar for L3 ARM manifests.

The L3 runtime writes one whole JSON root.  The frozen implementation list
and the current keyed implementation inventory remain separate branches.
The mechanism shapes come from the declared L3 ARM entries.
"""

from __future__ import annotations

from copy import deepcopy

__all__ = ["arm_manifest_schema"]

_PROFILES = ("L3-1_v1", "L3-2_v1", "L3-2_v2", "L3-3_v1", "L3-3_v2")
_IMPLEMENTATION_FILES = (
    "Dataset/tools/l3_v2/engine.py", "Dataset/tools/l3_v2/semantics.py",
    "Dataset/tools/l3_v2/arm_runtime.py", "Dataset/tools/batch_generate.py",
    "Plugins/SumoImporter/Scripts/donghu_core/event_script_interpreter.py",
    "Dataset/semantic_truth/objective_pipeline.py",
    "Dataset/semantic_simulation/predicate_state_computers.py",
    "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json",
)


def _obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _string() -> dict:
    return {"type": "string", "minLength": 1}


def _strings() -> dict:
    return {"type": "array", "items": _string()}


def _tick() -> dict:
    return {"type": "integer", "minimum": 0}


def _vector3() -> dict:
    return {"type": "array", "prefixItems": [
        {"type": "number"} for _ in range(3)], "items": False,
        "minItems": 3, "maxItems": 3}


def _window() -> dict:
    return _obj({
        "start_tick": _tick(), "fork_tick": _tick(), "end_tick": _tick(),
        "phase": {"const": "after_state_sample_before_actions"},
        "intervention_order": {"const": "source_actions_then_intervention"},
    })


def _action() -> dict:
    common = {"action_id": _string(), "entity_id": _string(), "tick": _tick()}
    move = _obj({**common, "type": {"const": "move_entity"},
                 "activity_type": _string(), "post_activity_type": _string(),
                 "velocity_mps": {"type": "number"},
                 "waypoints_enu_m": {"type": "array", "items": _vector3()}})
    patch = _obj({**common, "type": {"const": "set_runtime_state"},
                  "delay_ticks": _tick(),
                  "state_patch": {
                      "oneOf": [
                          _obj({"incident_state": _obj({
                              "temporary_lockdown_active": {"type": "boolean"}})}),
                          _obj({
                              "control_state": {"oneOf": [
                                  _obj({"reroute_active": {"type": "boolean"}}),
                                  _obj({"alternate_reroute_active": {"type": "boolean"}}),
                              ]},
                              "incident_state": _obj({
                                  "requires_reroute": {"type": "boolean"}}),
                          }),
                      ]}})
    visual = _obj({**common, "type": {"const": "set_visual_state"},
                   "visual_state": _obj({"mode": _string()})})
    return {"oneOf": [move, patch, visual]}


def _mechanism() -> dict:
    actions = {"type": "array", "items": _action()}
    return {"oneOf": [
        _obj({}),
        _obj({"kind": _string(), "actions": actions}),
        _obj({"kind": _string(), "actions": actions,
              "consequence": _string(), "retained_motion": _string()}),
        _obj({"kind": _string(), "omit_action_ids": _strings()}),
    ]}


def _window_status() -> dict:
    counts = {"type": "object", "patternProperties": {
        r"^[a-z][a-z0-9_]*$": _tick()}, "additionalProperties": False}
    target = _obj({
        "event_id": _string(), "onset_tick": _tick(),
        "terminal_tick": {"oneOf": [_tick(), {"type": "null"}]},
        "status": _string(),
    })
    return _obj({
        "window": _window(), "status": _string(),
        "absence_is_only_within_window": {"type": "boolean"},
        "event_count": _tick(), "inherited_event_ids": _strings(),
        "new_event_ids": _strings(), "event_family_counts": counts,
        "target_family": _string(),
        "target_events": {"type": "array", "items": target},
        "pending_event_ids": _strings(),
    })


def _path(*, digest: bool = False, role: bool = False) -> dict:
    fields = {"path": _string()}
    if digest:
        fields["digest"] = _string()
    if role:
        fields["role"] = _string()
    return _obj(fields)


def _base() -> dict:
    properties = {
        "schema": {"const": "aeroworld_l3_v2_arm"}, "version": {"const": 1},
        "arm_id": _string(), "group_key": _string(),
        "episode_id": {"type": "string", "pattern": r"^L3-[1-3]_v[12]__seed[0-9]{2}$"},
        "profile": {"enum": list(_PROFILES)},
        "seed": {"type": "string", "pattern": r"^seed[0-9]{2}$"},
        "mode": {"enum": ["factual_observation", "intervention"]},
        "event_family": _string(), "target_event_id": _string(),
        "fork_event_id": _string(), "window": _window(),
        "window_basis": _string(), "primary_predicate": _string(),
        "entity_ids": _strings(), "role": _string(),
        "mechanism": _mechanism(), "retime": {"type": "null"},
        "state_overrides": {"type": "array", "items": False, "maxItems": 0},
        "omit_action_ids": _strings(), "injected_action_ids": _strings(),
        "hypothesis": _string(), "falsified_if": _string(),
        "generation_method": _string(),
        "factual_baseline": {"const": "current_engine_factual_execution"},
        "historical_capture_role": {"const": "comparison_only_not_the_baseline"},
        "semantic_authority": _string(), "window_status": _window_status(),
        "event_occurrences": _tick(), "event_outcomes": _tick(),
        "predicate_transitions": _tick(),
        "sources": _obj({
            "scene": _path(), "script": _path(),
            "capture": _path(role=True),
            "historical_semantic": _path(role=True),
        }),
        "implementation": {"type": "array", "items": _string()},
        "environment": _obj({
            "python": _string(), "platform": _string(),
            "started_utc": _string(), "elapsed_s": {"type": "number"},
        }),
        "limitations": _strings(), "materialized_source": _string(),
        "semantic_sampling": {"const": "authoritative_capture_grid"},
        "sumo_intervention": {"oneOf": [
            {"type": "null"},
            _obj({"incident_id": _string(),
                  "release_tick": {"oneOf": [_tick(), {"type": "null"}]}}),
        ]},
        "artifacts": _strings(), "configuration": _path(),
        "entrypoint": _path(),
        "ue_delivery": _obj({
            "capture_executed": {"type": "boolean"},
            "scenario_package": _string(), "validation": _string(),
        }),
    }
    return _obj(properties)


def arm_manifest_schema() -> dict:
    """Return the closed L3 path-list and digest-map root branches."""
    frozen = _base()
    current = deepcopy(frozen)
    current["properties"]["implementation"] = _obj({
        key: _string() for key in _IMPLEMENTATION_FILES})
    source = current["properties"]["sources"]["properties"]
    source["scene"] = _path(digest=True)
    source["script"] = _path(digest=True)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/arm_manifest_l3.record.schema.json",
        "title": "AERO_WORLD arm_manifest L3 record schema",
        "oneOf": [frozen, current],
    }
