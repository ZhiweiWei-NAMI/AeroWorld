"""Closed root grammar for L2 ARM manifests.

``l2_window.execute_config`` writes the current path/list form.  Frozen
transfer records also retain a complete digest-map form.  The two forms
have separate closed branches; construction reads no source files.
"""

from __future__ import annotations

from copy import deepcopy

__all__ = ["arm_manifest_schema"]

_PROFILES = (
    "L2-1_v1", "L2-1_v2", "L2-2_v1", "L2-2_v2", "L2-3_v1",
    "L2-3_v2", "L2-4_v1", "L2-4_v2", "L2-5_v1",
)
_SOURCE_KEYS = ("roster", "scene", "script", "trajectories", "weather")
_CODE_KEYS = (
    "Dataset/tools/batch_generate.py", "Dataset/tools/l2_arm_semantics.py",
    "Dataset/tools/l2_window.py", "rules/l2_window.schema.json",
)


def _obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _string() -> dict:
    return {"type": "string", "minLength": 1}


def _strings() -> dict:
    return {"type": "array", "items": _string()}


def _integer() -> dict:
    return {"type": "integer", "minimum": 0}


def _path(*, digest: bool) -> dict:
    fields = {"path": _string()}
    if digest:
        fields["digest"] = _string()
    return _obj(fields)


def _window() -> dict:
    return _obj({
        "start_tick": _integer(), "fork_tick": _integer(),
        "end_tick": _integer(),
        "phase": {"const": "after_state_sample_before_actions"},
        "intervention_order": {"const": "source_actions_then_intervention"},
    })


def _current() -> dict:
    properties = {
        "schema_name": {"const": "l2_event_chain_window_arm"},
        "schema_version": {"const": "1.0.0"},
        "episode_id": {"type": "string", "pattern": r"^L2-[1-5]_v[12]__seed[0-9]{2}$"},
        "profile": {"enum": list(_PROFILES)},
        "seed": {"type": "string", "pattern": r"^seed[0-9]{2}$"},
        "group_key": {"type": "string", "pattern": r"^L2-[1-5]_v[12]/seed[0-9]{2}/grp_[a-z0-9_]+$"},
        "arm_id": _string(),
        "mode": {"enum": ["factual_observation", "intervention"]},
        "anchor_event": _string(), "neighbor_event": _string(),
        "chain_nodes": {"type": "array", "items": {
            "type": "array", "prefixItems": [_string(), _integer()],
            "items": False, "minItems": 2, "maxItems": 2,
        }},
        "target_families": _strings(), "entity_ids": _strings(),
        "semantic_entity_ids": _strings(), "predicate_ids": _strings(),
        "window": _window(), "tick_hz": {"const": 10},
        "formal_step_ticks": {"const": 5},
        "sources": _obj({key: _path(digest=False) for key in _SOURCE_KEYS}),
        "config": _path(digest=False), "arm_script": _path(digest=False),
        "code": {"type": "array", "items": _string(),
                 "minItems": 4, "maxItems": 4},
        "execution": _obj({"omit_action_ids": _strings(), "actions": _strings()}),
        "semantic_model": _obj({
            "gnss_formal_step_ticks": _integer(),
            "gnss_nonformal_policy": _string(), "model_version": _string(),
            "initial_gnss_error_vector_m": {
                "type": "array", "items": {"type": "number"},
                "minItems": 3, "maxItems": 3,
            },
            "initial_gnss_authority": _string(),
            "full_producer_certification": _string(),
            "frozen_background_feedback": _string(),
        }),
        "files": _strings(), "window_status": _string(),
        "event_count": _integer(),
        "environment": _obj({"python": _string(), "platform": _string()}),
        "data_provenance": _string(), "historical_reference": _string(),
        "snapshot_contract": _string(), "generation_method": _string(),
        "limitations": _strings(),
    }
    return _obj(properties)


def arm_manifest_schema() -> dict:
    """Return the closed current/frozen L2 manifest root grammar."""
    current = _current()
    frozen = deepcopy(current)
    fields = frozen["properties"]
    fields["sources"] = _obj({key: _path(digest=True) for key in _SOURCE_KEYS})
    fields["config"] = _path(digest=True)
    fields["arm_script"] = _path(digest=True)
    fields["code"] = _obj({key: _string() for key in _CODE_KEYS})
    fields["files"] = {
        "type": "object", "patternProperties": {
            r"^(?:raw|objective|chain)/[A-Za-z0-9_./-]+\.(?:json|jsonl)$": _string(),
        }, "additionalProperties": False,
    }
    model = fields["semantic_model"]
    model["properties"]["seed_digest"] = _string()
    model["required"].append("seed_digest")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/arm_manifest_l2.record.schema.json",
        "title": "AERO_WORLD arm_manifest L2 record schema",
        "oneOf": [current, frozen],
    }
