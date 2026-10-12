"""Complete L4 local business observation row, selected by producer profile.

``l4_window.execute_config`` writes this JSONL only when
``l4_incident_observations.supported`` accepts the profile.  That producer's
``evaluate`` function has four explicit mechanism branches.  Other L4 profiles
and the other ARM levels have no current local-business writer.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from Dataset.tools.l4_incident_observations import PROFILES


_DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _tick() -> dict[str, Any]:
    return {"type": "integer", "minimum": 0}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"oneOf": [schema, {"type": "null"}]}


def _vector3() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "prefixItems": [{"type": "number"} for _ in range(3)],
        "items": False,
    }


def _subject_map(value: dict[str, Any]) -> dict[str, Any]:
    # The L4 config contract accepts any nonempty entity_id.  The map keys
    # are precisely the configured subject IDs, not fixed business fields.
    return {
        "type": "object",
        "properties": {},
        "patternProperties": {"^.+$": value},
        "additionalProperties": False,
    }


def _values_and_metrics(mechanism: str) -> tuple[dict[str, Any], dict[str, Any]]:
    boolean = {"type": "boolean"}
    number = {"type": "number"}
    if mechanism == "crowd_evacuation":
        values = _object({
            "command_active": _nullable(boolean),
            "actual_motion": _nullable(boolean),
            "all_at_destination": _nullable(boolean),
            "arrival_acknowledged": _nullable(boolean),
        })
        metrics = {
            "oneOf": [
                _object({}),
                _object({
                    "destination_distance_m": _subject_map(number),
                    "destinations_enu_m": _subject_map(_vector3()),
                }),
            ]
        }
    elif mechanism == "crowd_proximity_response":
        values = _object({
            "proximity_volume_active": boolean,
            "proximity_dwell_ticks": _tick(),
            "proximity_recovered": boolean,
            "crowd_egress_motion": boolean,
        })
        metrics = _object({
            "horizontal_distance_m": number,
            "vertical_distance_m": number,
            "horizontal_limit_m": number,
            "vertical_limit_m": number,
            "recovery_dwell_ticks": _tick(),
        })
    elif mechanism == "medical_response":
        values = _object({
            "injury_context": _nullable(boolean),
            "dispatch_declared": _nullable(boolean),
            "physical_dispatch": boolean,
            "arrival_gate": _nullable(boolean),
            "near_presence_held": _nullable(boolean),
            "near_presence_handoff": _nullable(boolean),
        })
        metrics = _object({
            "distance_m": _nullable(number),
            "distance_limit_m": number,
            "near_dwell_ticks": _tick(),
            "required_dwell_ticks": _tick(),
            "arrival_dispatch_tick": _nullable(_tick()),
            "handoff_dispatch_tick": _nullable(_tick()),
            "dispatch_action_tick": _nullable(_tick()),
            "handoff_delay_ticks": _tick(),
        })
    elif mechanism == "fault_conditioned_landing":
        values = _object({
            "fault_active": boolean,
            "fault_descent_motion": boolean,
            "crowd_proximity_sustained": boolean,
            "crowd_motion": boolean,
            "landing_descent_motion": boolean,
            "touchdown_in_zone": _nullable(boolean),
        })
        metrics = _object({
            "horizontal_distance_m": number,
            "vertical_distance_m": number,
            "horizontal_limit_m": number,
            "vertical_limit_m": number,
            "landing_dispatch_tick": _nullable(_tick()),
        })
    else:
        raise ValueError(f"no L4 local business producer for {mechanism!r}")
    return values, metrics


def arm_local_business_truth_schema(profile: str) -> dict[str, Any]:
    """Return one closed JSONL-row schema for an explicitly supported L4 profile."""
    if profile not in PROFILES:
        raise ValueError(f"no L4 local business producer for {profile!r}")
    mechanism = PROFILES[profile]
    values, metrics = _values_and_metrics(mechanism)
    source_ref = {
        "type": "string",
        "oneOf": [
            {"pattern": r"^raw/states\.jsonl#tick=[0-9]+&entity=.+$"},
            {"pattern": r"^raw/actions\.jsonl#tick=[0-9]+&action_id=.+$"},
            {"pattern": r"^objective/predicate_truth\.jsonl#.+$"},
        ],
    }
    schema = {
        "$schema": _DRAFT,
        "$id": f"https://aeroworld.local/schema/arm_local_business_truth/{profile}",
        "type": "object",
        "additionalProperties": False,
        "required": ["tick", "values", "metrics", "source_refs"],
        "properties": {
            "tick": _tick(),
            "values": values,
            "metrics": metrics,
            "source_refs": {"type": "array", "items": source_ref},
        },
    }
    if mechanism == "crowd_evacuation":
        # A missing configured subject produces four null values and no
        # metrics; full subject coverage produces four booleans and both
        # subject maps.  The producer never mixes these two states.
        value_names = list(values["properties"])
        schema["properties"]["values"] = {"type": "object"}
        schema["properties"]["metrics"] = {"type": "object"}
        schema["oneOf"] = [
            {"properties": {
                "values": _object({name: {"type": "null"} for name in value_names}),
                "metrics": _object({}),
            }},
            {"properties": {
                "values": _object({name: {"type": "boolean"} for name in value_names}),
                "metrics": metrics["oneOf"][1],
            }},
        ]
    return deepcopy(schema)


__all__ = ["arm_local_business_truth_schema"]
