"""Whole-record grammar for L4 ARM ``objective/event_occurrences.jsonl``.

``Dataset/tools/l4_window.py::execute_config`` is the writer for all declared
L4 ARM windows.  It persists one of two producer-owned record envelopes:

* Profiles without an L4 business rule use
  ``minimal_semantics_adapter._adapt_occurrences`` and then crop phases and
  evidence references at the window end.  The existing formal occurrence
  schema already admits every cropped phase sequence because an in-window
  event always retains its onset.  Frozen L4 rows may omit the integrity field
  ``rule_digest``; the current adapter emits it, so this L4 branch keeps the
  field closed and pattern-checked but optional.
* The profiles registered in
  ``Dataset/semantic_rules/profiles/l4_arm_business_semantics.json`` use
  ``l4_business_semantics.evaluate``.  Their records have a distinct 12-key
  envelope, a profile-sized ``bindings.entities`` list, and an exact phase
  sequence consisting of onset plus an optional terminal transition.  The
  frozen L4-9_v1/v2 occurrence records instead have an 11-key envelope
  without ``rule_digest`` and ``bindings.vehicles``; a separate closed arm
  preserves those existing records without changing the current writer arm.

The profile split is checked against the current L4 authorities when the
schema is constructed.  This module reads no corpus data and does not register
the grammar globally.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from Dataset.tools.l4_business_semantics import contract as business_contract
from Dataset.tools.l4_chain_truth import L4_INCIDENT_FAMILY
from Dataset.world_model.graph.event_structure import (
    formal_event_occurrence_schema,
)


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_DIGEST_PATTERN = "^sha256:[0-9a-f]{64}$"
_BUSINESS_SCHEMA_NAME = "l4_business_event_occurrence"
_BUSINESS_SCHEMA_VERSION = "1.0.0"

_DECLARED_PROFILES = frozenset({
    "L4-1_v1", "L4-1_v2", "L4-2_v1", "L4-2_v2", "L4-3_v1",
    "L4-3_v2", "L4-3_v3", "L4-4_v1", "L4-4_v2", "L4-5_v1",
    "L4-5_v2", "L4-5_v3", "L4-6_v1", "L4-6_v2", "L4-7_v1",
    "L4-7_v2", "L4-8_v1", "L4-8_v2", "L4-9_v1", "L4-9_v2",
    "L4-10_v1", "L4-10_v2", "L4-11_v1", "L4-11_v2",
})
_BUSINESS_PROFILES = frozenset({
    "L4-4_v1", "L4-4_v2", "L4-5_v1", "L4-5_v2", "L4-5_v3",
    "L4-9_v1", "L4-9_v2", "L4-10_v1", "L4-10_v2",
    "L4-11_v1", "L4-11_v2",
})
_BINDING_COUNTS = {
    "uav_vehicle": 2,
    "uav_pedestrian": 2,
    "ambulance": 2,
    "intersection": 2,
    "av": 1,
}


def _string() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _object(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    required: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties) if required is None else list(required),
        "properties": {str(key): dict(value) for key, value in properties.items()},
    }


def _capture_tick() -> dict[str, Any]:
    return {
        "type": "integer",
        "minimum": 0,
        "maximum": 900,
        "multipleOf": 5,
    }


def _episode_pattern(profiles: set[str] | frozenset[str]) -> str:
    if not profiles:
        raise ValueError("L4 occurrence branch has no profiles")
    alternatives = "|".join(re.escape(profile) for profile in sorted(profiles))
    return rf"^(?:{alternatives})__seed(?:00|01|02)$"


def _formal_occurrence_schema() -> dict[str, Any]:
    """Adapt the shared formal schema only at the two L4 source boundaries."""

    schema = formal_event_occurrence_schema()
    for annotation in ("$schema", "$id", "title"):
        schema.pop(annotation)
    schema["properties"]["episode_id"] = {
        "type": "string",
        "pattern": _episode_pattern(_DECLARED_PROFILES - _BUSINESS_PROFILES),
    }
    # Frozen L4 records predate this current-adapter integrity field.  Keep
    # its exact raw value rule while allowing those whole records unchanged.
    schema["required"].remove("rule_digest")
    return schema


def _business_phase(kind: str, predicate_id: str) -> dict[str, Any]:
    return _object({
        "phase_kind": {"const": kind},
        "tick": _capture_tick(),
        "transition_id": {
            "type": "string",
            "pattern": "^l4_transition:[0-9a-f]{24}$",
        },
        "predicate_id": {"const": predicate_id},
    })


def _business_phases(onset: str, terminal: str) -> dict[str, Any]:
    onset_phase = _business_phase("onset", onset)
    terminal_phase = _business_phase("terminal", terminal)
    return {
        "oneOf": [
            {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "prefixItems": [onset_phase],
                "items": False,
            },
            {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "prefixItems": [onset_phase, terminal_phase],
                "items": False,
            },
        ],
    }


def _business_source_refs() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "string",
            "pattern": (
                r"^raw/(?:states\.jsonl#tick=(?:0|[1-9][0-9]*)&entity=[\s\S]+"
                r"|actions\.jsonl#tick=(?:0|[1-9][0-9]*)&action_id=[\s\S]+)$"
            ),
        },
        "minItems": 1,
        "uniqueItems": True,
    }


def _business_occurrence_schema(
    profile: str,
    rule: Mapping[str, Any],
    authority_scope: str,
) -> dict[str, Any]:
    mechanism = rule.get("mechanism")
    if mechanism not in _BINDING_COUNTS:
        raise ValueError(f"{profile}: unknown L4 business mechanism {mechanism!r}")
    event_family = rule.get("event_family")
    onset = rule.get("onset_predicate")
    terminal = rule.get("terminal_predicate")
    predicate_ids = rule.get("predicate_ids")
    if (
        not isinstance(event_family, str)
        or not event_family
        or not isinstance(onset, str)
        or not onset
        or not isinstance(terminal, str)
        or not terminal
        or not isinstance(predicate_ids, list)
        or onset not in predicate_ids
        or terminal not in predicate_ids
    ):
        raise ValueError(f"{profile}: invalid L4 business occurrence rule")
    binding_count = _BINDING_COUNTS[mechanism]
    entities = {
        "type": "array",
        "items": _string(),
        "minItems": binding_count,
        "maxItems": binding_count,
        "uniqueItems": True,
    }
    return _object({
        "schema_name": {"const": _BUSINESS_SCHEMA_NAME},
        "schema_version": {"const": _BUSINESS_SCHEMA_VERSION},
        "authority_scope": {"const": authority_scope},
        "episode_id": {
            "type": "string",
            "pattern": _episode_pattern({profile}),
        },
        "event_id": {
            "type": "string",
            "pattern": "^l4_event:[0-9a-f]{24}$",
        },
        "event_family_id": {"const": event_family},
        "trigger_tick": _capture_tick(),
        "rule_id": {"const": "l4_arm_business." + mechanism},
        "rule_digest": {
            "type": "string",
            "pattern": _DIGEST_PATTERN,
        },
        "bindings": _object({"entities": entities}),
        "event_phases": _business_phases(onset, terminal),
        "source_refs": _business_source_refs(),
    })


def _frozen_intersection_occurrence_schema(
    profile: str,
    rule: Mapping[str, Any],
    authority_scope: str,
) -> dict[str, Any]:
    if profile not in {"L4-9_v1", "L4-9_v2"} or rule.get("mechanism") != "intersection":
        raise ValueError(f"{profile}: frozen occurrence branch requires L4-9 intersection")
    schema = _business_occurrence_schema(profile, rule, authority_scope)
    schema["properties"].pop("rule_digest")
    schema["required"].remove("rule_digest")
    entities = schema["properties"]["bindings"]["properties"]["entities"]
    schema["properties"]["bindings"] = _object({"vehicles": entities})
    return schema


def arm_event_occurrences_schema() -> dict[str, Any]:
    """Return the closed schema for one L4 ARM event-occurrence JSONL row."""

    if set(L4_INCIDENT_FAMILY) != _DECLARED_PROFILES:
        raise ValueError("L4 profile declaration changed; review occurrence grammar")
    authority = business_contract()
    rules = authority.get("profiles")
    if not isinstance(rules, Mapping) or set(rules) != _BUSINESS_PROFILES:
        raise ValueError("L4 business profile rules changed; review occurrence grammar")
    if authority.get("capture_step_ticks") != 5:
        raise ValueError("L4 business capture step changed; review occurrence grammar")
    authority_scope = authority.get("authority_scope")
    if not isinstance(authority_scope, str) or not authority_scope:
        raise ValueError("L4 business authority scope is absent")

    return {
        "$schema": _DRAFT,
        "$id": (
            "https://aeroworld.local/schema/"
            "arm_event_occurrence.l4.record.schema.json"
        ),
        "title": "L4 ARM event occurrence JSONL record",
        "oneOf": [
            _formal_occurrence_schema(),
            *[
                _business_occurrence_schema(profile, rules[profile], authority_scope)
                for profile in sorted(rules)
            ],
            *[
                _frozen_intersection_occurrence_schema(
                    profile, rules[profile], authority_scope,
                )
                for profile in ("L4-9_v1", "L4-9_v2")
            ],
        ],
    }


__all__ = ["arm_event_occurrences_schema"]
