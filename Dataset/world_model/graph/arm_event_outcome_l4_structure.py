"""Whole-row grammar for L4 ARM ``objective/event_outcomes.jsonl``.

``Dataset.tools.l4_window.execute_config`` selects one of two producers.  The
thirteen profiles without a local business rule pass formal outcomes through
``window_outcome.arm_window_outcome``; the other eleven profiles retain the
distinct ``l4_business_event_outcome`` envelope written by
``l4_business_semantics.evaluate``.

Current formal rows use ``arm_window_event_outcome`` version ``1.0.0``.  A
terminal beyond the window becomes a closed ``right_censored`` row with
``window_censor_tick``.  Current business rows use the native business schema,
with their family, terminal predicate, hold length, and interpretation bound to
the executable profile registry.

Frozen L4 rows predate those current envelopes.  Formal rows retain
``event_outcome`` version ``3.0.0`` and both frozen producers add the paired
``terminal_within_window`` and ``censored_by_window`` fields.  Frozen L4-9
business rows also retain their earlier interpretation sentence.  These
recorded shapes stay separate from the current writer shapes; every root is
closed and every nullable terminal is selected by its status branch.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping

from Dataset.tools.l4_business_semantics import contract as business_contract
from Dataset.tools.l4_chain_truth import L4_INCIDENT_FAMILY
from Dataset.world_model.graph.event_structure import formal_event_outcome_schema


__all__ = ["arm_event_outcomes_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
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
_FROZEN_WINDOW_FIELDS = (
    "terminal_within_window",
    "censored_by_window",
)
_CURRENT_INTERPRETATION = (
    "Confirms the stated local response only; no claim of complete hazard "
    "recovery, legal right of way, or clinical recovery."
)
_FROZEN_L4_9_INTERPRETATION = (
    "Confirms the stated local response only; no claim of complete hazard "
    "recovery or legal passage."
)


def _episode_pattern(profiles: set[str] | frozenset[str]) -> str:
    if not profiles:
        raise ValueError("L4 outcome branch has no profiles")
    alternatives = "|".join(re.escape(profile) for profile in sorted(profiles))
    return rf"^(?:{alternatives})__seed(?:00|01|02)$"


def _formal_envelope(
    *, schema_name: str, schema_version: str
) -> dict[str, Any]:
    """Detach the formal schema and bind the non-business L4 profiles."""

    schema = formal_event_outcome_schema()
    for annotation in ("$schema", "$id", "title"):
        schema.pop(annotation)
    properties = schema.get("properties")
    required = schema.get("required")
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or type(properties) is not dict
        or type(required) is not list
        or set(required) != set(properties)
    ):
        raise ValueError("formal event-outcome schema has no closed root envelope")
    properties["schema_name"] = {"const": schema_name}
    properties["schema_version"] = {"const": schema_version}
    properties["episode_id"] = {
        "type": "string",
        "pattern": _episode_pattern(_DECLARED_PROFILES - _BUSINESS_PROFILES),
    }
    return schema


def _frozen_formal_schema() -> dict[str, Any]:
    schema = _formal_envelope(
        schema_name="event_outcome",
        schema_version="3.0.0",
    )
    properties = schema["properties"]
    properties.update({
        field: {"type": "boolean"} for field in _FROZEN_WINDOW_FIELDS
    })
    schema["required"].extend(_FROZEN_WINDOW_FIELDS)
    schema["allOf"].append({
        "oneOf": [
            {
                "properties": {
                    "status": {"const": "pending"},
                    "terminal_within_window": {"const": False},
                    "censored_by_window": {"const": True},
                },
                "required": ["status", *_FROZEN_WINDOW_FIELDS],
            },
            {
                "properties": {
                    "status": {"const": "succeeded"},
                    "terminal_within_window": {"const": True},
                    "censored_by_window": {"const": False},
                },
                "required": ["status", *_FROZEN_WINDOW_FIELDS],
            },
            {
                "properties": {
                    "status": {"const": "succeeded"},
                    "terminal_within_window": {"const": False},
                    "censored_by_window": {"const": True},
                },
                "required": ["status", *_FROZEN_WINDOW_FIELDS],
            },
        ]
    })
    return schema


def _current_formal_schema() -> dict[str, Any]:
    return _formal_envelope(
        schema_name="arm_window_event_outcome",
        schema_version="1.0.0",
    )


def _current_right_censored_schema() -> dict[str, Any]:
    """Bind the exact branch produced for a terminal beyond the window."""

    schema = _formal_envelope(
        schema_name="arm_window_event_outcome",
        schema_version="1.0.0",
    )
    properties = schema["properties"]
    properties.update({
        "status": {"const": "right_censored"},
        "terminal_tick": {"const": None},
        "lifecycle_status": {"const": "observed_until_window_end"},
        "lifecycle_phase_evidence": {
            "type": "array",
            "minItems": 0,
            "maxItems": 0,
            "items": False,
        },
        "window_censor_tick": {
            "type": "integer",
            "minimum": 0,
        },
    })
    schema["required"] = list(properties)

    selector = schema.get("allOf")
    if (
        type(selector) is not list
        or len(selector) != 1
        or type(selector[0]) is not dict
        or type(selector[0].get("oneOf")) is not list
    ):
        raise ValueError("formal event-outcome schema has no family selector")
    family_branches = []
    for variant in selector[0]["oneOf"]:
        if type(variant) is not dict or type(variant.get("oneOf")) is not list:
            raise ValueError("formal event-outcome family has no status selector")
        statuses = {
            branch.get("properties", {}).get("status", {}).get("const")
            for branch in variant["oneOf"]
            if type(branch) is dict
        }
        if statuses != {"pending", "succeeded"}:
            raise ValueError("formal event-outcome status contract changed")
        family = deepcopy(variant)
        family.pop("oneOf")
        family_branches.append(family)
    schema["allOf"] = [{"oneOf": family_branches}]
    return schema


def _string() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _ids() -> dict[str, Any]:
    return {
        "type": "array",
        "items": _string(),
        "minItems": 0,
        "uniqueItems": True,
    }


def _object(properties: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": deepcopy(dict(properties)),
    }


def _business_schema(
    profile: str,
    rule: Mapping[str, Any],
    terminal_hold_samples: int,
    *,
    frozen: bool,
) -> dict[str, Any]:
    interpretation = (
        _FROZEN_L4_9_INTERPRETATION
        if frozen and profile.startswith("L4-9_")
        else _CURRENT_INTERPRETATION
    )
    properties: dict[str, Any] = {
        "schema_name": {"const": "l4_business_event_outcome"},
        "schema_version": {"const": "1.0.0"},
        "episode_id": {
            "type": "string",
            "pattern": _episode_pattern({profile}),
        },
        "event_id": {
            "type": "string",
            "pattern": "^l4_event:[0-9a-f]{24}$",
        },
        "event_family_id": {"const": rule["event_family"]},
        "outcome_id": {
            "type": "string",
            "pattern": "^l4_outcome:[0-9a-f]{24}$",
        },
        # The compiler resolves this property through the required status
        # selector below.  Both alternatives give it an exact type/value.
        "terminal_tick": {},
        "status": {"enum": ["pending", "succeeded"]},
        "lifecycle_status": {
            "enum": [
                "response_not_confirmed_in_window",
                "observed_response_confirmed",
            ]
        },
        "terminal_predicate_id": {"const": rule["terminal_predicate"]},
        "supporting_truth_ids": _ids(),
        "terminal_hold_samples": {"const": terminal_hold_samples},
        "interpretation": {"const": interpretation},
    }
    if frozen:
        properties.update({
            field: {"type": "boolean"} for field in _FROZEN_WINDOW_FIELDS
        })

    pending: dict[str, Any] = {
        "status": {"const": "pending"},
        "terminal_tick": {"const": None},
        "lifecycle_status": {"const": "response_not_confirmed_in_window"},
        "supporting_truth_ids": {"maxItems": 0},
    }
    succeeded: dict[str, Any] = {
        "status": {"const": "succeeded"},
        "terminal_tick": {"type": "integer", "minimum": 0},
        "lifecycle_status": {"const": "observed_response_confirmed"},
        "supporting_truth_ids": {"minItems": 1},
    }
    if frozen:
        pending.update({
            "terminal_within_window": {"const": False},
            "censored_by_window": {"const": True},
        })
        succeeded.update({
            "terminal_within_window": {"const": True},
            "censored_by_window": {"const": False},
        })

    schema = _object(properties)
    schema["allOf"] = [{
        "oneOf": [
            {
                "properties": pending,
                "required": list(pending),
            },
            {
                "properties": succeeded,
                "required": list(succeeded),
            },
        ]
    }]
    return schema


def arm_event_outcomes_schema() -> dict[str, Any]:
    """Return the closed schema for one L4 ARM event-outcome JSONL row."""

    if set(L4_INCIDENT_FAMILY) != _DECLARED_PROFILES:
        raise ValueError("L4 profile declaration changed; review outcome grammar")
    authority = business_contract()
    rules = authority.get("profiles")
    if not isinstance(rules, Mapping) or set(rules) != _BUSINESS_PROFILES:
        raise ValueError("L4 business profile rules changed; review outcome grammar")
    terminal_hold_samples = authority.get("terminal_hold_samples")
    if type(terminal_hold_samples) is not int or terminal_hold_samples < 1:
        raise ValueError("L4 business terminal hold contract is invalid")
    for profile, rule in rules.items():
        predicate_ids = rule.get("predicate_ids")
        terminal_predicate = rule.get("terminal_predicate")
        if (
            not isinstance(rule.get("event_family"), str)
            or not rule["event_family"]
            or not isinstance(terminal_predicate, str)
            or not terminal_predicate
            or not isinstance(predicate_ids, list)
            or terminal_predicate not in predicate_ids
        ):
            raise ValueError(f"{profile}: invalid L4 business outcome rule")

    frozen_business = {
        "oneOf": [
            _business_schema(
                profile,
                rules[profile],
                terminal_hold_samples,
                frozen=True,
            )
            for profile in sorted(rules)
        ]
    }
    current_business = {
        "oneOf": [
            _business_schema(
                profile,
                rules[profile],
                terminal_hold_samples,
                frozen=False,
            )
            for profile in sorted(rules)
        ]
    }
    return {
        "$schema": _DRAFT,
        "$id": (
            "https://aeroworld.local/schema/"
            "arm_event_outcome.l4.record.schema.json"
        ),
        "title": "L4 ARM event outcome JSONL record",
        "oneOf": [
            _frozen_formal_schema(),
            _current_formal_schema(),
            _current_right_censored_schema(),
            frozen_business,
            current_business,
        ],
    }
