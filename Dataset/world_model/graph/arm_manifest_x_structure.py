"""Closed whole-root grammar for X-layer ARM manifests.

``x_arm_pipeline`` writes scoped factual and explicit branch manifests.
Frozen roots may additionally carry the source-reference integrity map.
Construction reads no data and preserves both complete root shapes.
"""

from __future__ import annotations

from copy import deepcopy

__all__ = ["arm_manifest_schema"]

_PROFILES = (
    "X1_rain_to_c2loss_to_forced_landing",
    "X2_gnss_spoof_to_geofence_violation",
    "X3_pedestrian_fall_to_emergency_response",
    "X4_fog_to_uav_conflict",
    "X5_comm_failure_to_pad_contention",
    "X6_crowd_evacuation_to_airspace_lockdown",
)
_SOURCE_KEYS = ("scene", "original_scene", "script", "canonical",
                "predicate_script")
_CHANGE_KEYS = (
    "removed_event_ids", "added_event_ids", "removed_action_ids",
    "added_action_ids", "replaced_action_ids", "changed_event_ids",
    "changed_trigger_ids",
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


def _root() -> dict:
    return _obj({
        "episode_id": {"type": "string", "pattern":
                       r"^X[1-6]_[a-z0-9_]+__seed[0-9]{2}$"},
        "arm_id": _string(),
        "mode": {"enum": ["factual_observation", "intervention"]},
        "window": _obj({
            "start_tick": _tick(), "fork_tick": _tick(),
            "end_tick": _tick(),
            "phase": {"const": "after_state_sample_before_actions"},
        }),
        "sources": _obj({key: _string() for key in _SOURCE_KEYS}),
        "entity_ids": _strings(), "removed_event_ids": _strings(),
        "scope": _obj({
            "simulated_entity_ids": _strings(),
            "canonical_only_scene_entity_ids": _strings(),
            "background_policy": _string(), "interaction_limit": _string(),
        }),
        "branch_changes": _obj({key: _strings() for key in _CHANGE_KEYS}),
        "baseline": _string(), "semantics": _string(),
    })


def arm_manifest_schema() -> dict:
    """Return both closed X ARM manifest root shapes."""
    base = _root()
    referenced = deepcopy(base)
    referenced["properties"]["source_refs"] = {
        "type": "object", "patternProperties": {
            r"^(?:interventions/X[1-6]_[a-z0-9_]+/seed[0-9]{2}/grp_[a-z0-9_]+/|Dataset/scenarios/X_cross_layer/X[1-6]_[a-z0-9_]+/)[A-Za-z0-9_./-]+\.(?:yaml|json)$": _string(),
        }, "additionalProperties": False,
    }
    referenced["required"].append("source_refs")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/arm_manifest_x.record.schema.json",
        "title": "AERO_WORLD arm_manifest X record schema",
        "oneOf": [base, referenced],
    }
