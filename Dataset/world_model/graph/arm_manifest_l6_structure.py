"""Closed whole-root grammar for L6 ARM manifests.

``l6_v2.arm_runtime._manifest`` writes the main record; the frozen UE
delivery subset adds two root fields.  A nonempty retime writes the same
three-key edit to ``declared_mechanism`` and its delay provenance; the
current L6 manifests have no instance of this producer-supported branch.
Action and window-node member shapes follow the dedicated ARM declarations.
Import reads no data.
"""

from __future__ import annotations

from copy import deepcopy

__all__ = ["arm_manifest_schema"]

_PROFILES = (
    "L6-1_v1", "L6-1_v2", "L6-2_v1", "L6-2_v2", "L6-3_v1",
    "L6-3_v2", "L6-4_v1", "L6-4_v2", "L6-5_v1", "L6-5_v2",
)
_COUNT_KEYS = (
    "predicate_truth_ticks", "predicate_truth", "predicate_transitions",
    "event_occurrences", "event_outcomes",
)
_NODE_KEYSETS = (
    ("event_id", "role", "tick"),
    ("action_id", "role", "tick"),
    ("predicate_id", "role", "tick"),
    ("motion_id", "role", "tick"),
    ("entity_id", "event_id", "kind", "role", "tick"),
    ("entity_id", "event_id", "role", "tick"),
    ("entity_id", "event_id", "role", "terminal_state_tick", "tick"),
    ("effective_ticks", "event_id", "role", "tick"),
    ("effective_tick", "event_id", "role", "tick"),
    ("effective_tick", "entity_id", "event_id", "role", "tick"),
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


def _path() -> dict:
    return _obj({"path": _string()})


def _vector3() -> dict:
    return {"type": "array", "prefixItems": [
        {"type": "number"} for _ in range(3)], "items": False,
        "minItems": 3, "maxItems": 3}


def _window() -> dict:
    return _obj({
        "start_tick": _tick(), "fork_tick": _tick(), "end_tick": _tick(),
        "phase": {"const": "after_state_sample_before_actions"},
    })


def _state_patch() -> dict:
    families = {
        "communication_state": {
            "mode": _string(), "handover_active": {"type": "boolean"},
            "backup_link_active": {"type": "boolean"},
            "communication_unavailable": {"type": "boolean"},
        },
        "control_state": {
            "safe_hold_active": {"type": "boolean"},
            "slowdown_active": {"type": "boolean"},
        },
        "navigation_state": {
            "gnss_spoofed": {"type": "boolean"},
            "visual_relocalization": {"type": "boolean"},
            "mission_recovered": {"type": "boolean"},
            "route_uncertain": {"type": "boolean"},
            "route_recovered": {"type": "boolean"},
            "operational_state": _string(),
        },
        "security_state": {
            "alternate_channel_active": {"type": "boolean"},
            "jamming_active": {"type": "boolean"},
            "lockout_active": {"type": "boolean"},
            "command_integrity_violation": {"type": "boolean"},
            "command_lockout": {"type": "boolean"},
            "unauthorized_command": {"type": "boolean"},
        },
    }
    return {
        "type": "object", "properties": {
            family: {"type": "object", "properties": fields,
                     "minProperties": 1, "additionalProperties": False}
            for family, fields in families.items()},
        "minProperties": 1, "additionalProperties": False,
    }


def _injected_action() -> dict:
    common = {"tick": _tick(), "action_id": _string(), "entity_id": _string()}
    motion = {**common, "type": {"const": "move_entity"},
              "velocity_mps": {"type": "number"},
              "waypoints_enu_m": {"type": "array", "items": _vector3()}}
    return {"oneOf": [
        _obj(motion),
        _obj({**motion, "activity_type": _string(),
              "post_activity_type": _string()}),
        _obj({**common, "type": {"const": "set_runtime_state"},
              "delay_ticks": _tick(), "state_patch": _state_patch()}),
        _obj({**common, "type": {"const": "set_visual_state"},
              "visual_state": _obj({"mode": _string()})}),
    ]}


def _needed_node() -> dict:
    def member(keys: tuple[str, ...]) -> dict:
        return _obj({key: (
            _tick() if key in ("tick", "terminal_state_tick", "effective_tick")
            else {"type": "array", "items": _tick()} if key == "effective_ticks"
            else _string()) for key in keys})
    return {"oneOf": [member(keys) for keys in _NODE_KEYSETS]}


def _base() -> dict:
    observed_counts = {"oneOf": [
        _obj({"false": _tick()}),
        _obj({"false": _tick(), "true": _tick()}),
    ]}
    properties = {
        "schema": {"const": "aero_l6_v2_arm_manifest"}, "version": {"const": 1},
        "episode_id": {"type": "string", "pattern": r"^L6-[1-5]_v[12]__seed[0-9]{2}$"},
        "profile": {"enum": list(_PROFILES)},
        "seed": {"type": "string", "pattern": r"^seed[0-9]{2}$"},
        "group_key": _string(), "arm_id": _string(),
        "mode": {"enum": ["factual_observation", "intervention"]},
        "role": _string(), "anchor_event_id": _string(),
        "declared_anchor_family": _string(), "anchor_dispatch_tick": _tick(),
        "window": _window(), "tick_hz": {"const": 10},
        "formal_step_ticks": {"const": 5}, "window_basis": _string(),
        "window_needed_nodes": {"type": "array", "items": _needed_node()},
        "window_censored": {"type": "boolean"},
        "completion_requirements": {"type": "array", "items": _obj({
            "bindings": _obj({"actor": _string()}),
            "predicate_id": _string(), "required_value": _string(),
        })},
        "scope": _obj({key: _strings() for key in (
            "entity_ids", "aircraft_ids", "station_ids", "gcs_ids",
            "home_pad_ids")}),
        "predicate_ids": _strings(), "event_families": _strings(),
        "primary_predicate": _string(), "generation_method": _string(),
        "ue_script_role": _string(),
        "declared_mechanism": _obj({
            "kind": _string(), "retime": {"type": "null"},
            "omit_action_ids": _strings(),
            "inject_actions": {"type": "array", "items": _injected_action()},
            "cancel_event_ids": _strings(),
        }),
        "hypothesis": _string(), "falsified_if": _string(),
        "provenance": _obj({
            "parameter_sources": _obj({
                "scene": _string(), "script": _string(),
                "delay_ticks": {"type": "null"},
                "delay_source": {"type": "null"},
                "factual_baseline": _string(),
            }),
            "implementation_sources": {"oneOf": [
                _strings(), _obj({}),
            ]},
        }),
        "source_refs": _obj({
            "scene": _path(), "script": _path(),
            "semantic_history": _obj({
                "path": _string(), "format": _string(), "scope": _string()}),
            "configuration": _path(), "entry": _path(),
            "experiment": _path(),
        }),
        "artifacts": _strings(),
        "record_counts": _obj({key: _tick() for key in _COUNT_KEYS}),
        "observed": _obj({
            "window_status": _string(),
            "primary_formal_counts": observed_counts,
            "event_count": _tick(),
        }),
        "run_receipt": _obj({
            "started_utc": _string(), "finished_utc": _string(),
            "elapsed_s": {"type": "number"}, "python": _string(),
            "executable": _string(), "platform": _string(),
            "exit_code": _tick(),
        }),
        "review": _obj({
            "review_type": _string(), "release_certification": _string(),
        }),
    }
    return _obj(properties)


def arm_manifest_schema() -> dict:
    """Return closed L6 roots across UE delivery and retime state."""
    base = _base()
    delivered = deepcopy(base)
    delivered["properties"]["ue_package_ref"] = _string()
    delivered["properties"]["ue_capture_status"] = _string()
    delivered["required"].extend(("ue_package_ref", "ue_capture_status"))

    retimed = deepcopy(base)
    retimed["properties"]["mode"] = {"const": "intervention"}
    retimed["properties"]["declared_mechanism"]["properties"]["retime"] = _obj({
        "trigger_id": _string(), "field": _string(),
        "delta": {"type": "integer"},
    })
    retime_sources = retimed["properties"]["provenance"]["properties"]["parameter_sources"]["properties"]
    retime_sources["delay_ticks"] = {"type": "integer"}
    retime_sources["delay_source"] = {"const": "explicit configuration"}
    retimed_delivered = deepcopy(retimed)
    retimed_delivered["properties"]["ue_package_ref"] = _string()
    retimed_delivered["properties"]["ue_capture_status"] = _string()
    retimed_delivered["required"].extend(("ue_package_ref", "ue_capture_status"))
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/arm_manifest_l6.record.schema.json",
        "title": "AERO_WORLD arm_manifest L6 record schema",
        "oneOf": [base, delivered, retimed, retimed_delivered],
    }
