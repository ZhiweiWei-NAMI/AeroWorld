"""Closed whole-row grammar for L3 ARM event-outcome records.

Every bound L3 entrypoint runs through ``l3_ue_delivery`` and
``l3_v2.arm_runtime``.  The runtime selects formal outcomes from the objective
pipeline, applies the bounded-window censor rule, and validates the result
through ``window_outcome.arm_window_outcome`` before writing it.

The current writer changes the formal header to
``arm_window_event_outcome`` version ``1.0.0``.  Frozen L3 rows retain the
pre-wrapper ``event_outcome`` version ``3.0.0`` header.  These are disjoint
closed branches.  Both keep the formal producer's registry-bound event rules;
the window-only ``right_censored`` branch has its own closed terminal shape.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from Dataset.world_model.graph.event_structure import formal_event_outcome_schema


__all__ = ["arm_event_outcomes_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_EPISODE_ID_PATTERN = r"^(?:L3-1_v1|L3-2_v[12]|L3-3_v[12])__seed(?:00|01|02)$"
_HEADERS = (
    ("event_outcome", "3.0.0"),
    ("arm_window_event_outcome", "1.0.0"),
)


def _formal_envelope(
    formal: dict[str, Any], *, schema_name: str, schema_version: str
) -> dict[str, Any]:
    """Detach the formal schema and bind one L3 window header."""

    schema = deepcopy(formal)
    for annotation in ("$schema", "$id", "title"):
        schema.pop(annotation, None)

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
        "pattern": _EPISODE_ID_PATTERN,
    }
    return schema


def _right_censored_envelope(
    formal: dict[str, Any], *, schema_name: str, schema_version: str
) -> dict[str, Any]:
    """Declare the exact row produced when a terminal lies past the window."""

    schema = _formal_envelope(
        formal, schema_name=schema_name, schema_version=schema_version
    )
    properties = schema["properties"]
    properties["status"] = {"const": "right_censored"}
    properties["terminal_tick"] = {"const": None}
    properties["lifecycle_status"] = {"const": "observed_until_window_end"}
    properties["lifecycle_phase_evidence"] = {
        "type": "array",
        "minItems": 0,
        "maxItems": 0,
        "items": False,
    }
    properties["window_censor_tick"] = {
        "type": "integer",
        "minimum": 0,
    }
    schema["required"] = list(properties)

    all_of = schema.get("allOf")
    if (
        type(all_of) is not list
        or len(all_of) != 1
        or type(all_of[0]) is not dict
        or type(all_of[0].get("oneOf")) is not list
        or not all_of[0]["oneOf"]
    ):
        raise ValueError("formal event-outcome schema has no event-rule variants")

    event_rules = []
    for rule_variant in all_of[0]["oneOf"]:
        if type(rule_variant) is not dict or type(rule_variant.get("oneOf")) is not list:
            raise ValueError("formal event-outcome rule has no status variants")
        statuses = {
            branch.get("properties", {}).get("status", {}).get("const")
            for branch in rule_variant["oneOf"]
            if type(branch) is dict
        }
        if statuses != {"pending", "succeeded"}:
            raise ValueError("formal event-outcome status contract has changed")
        event_rule = deepcopy(rule_variant)
        event_rule.pop("oneOf")
        event_rules.append(event_rule)
    schema["allOf"] = [{"oneOf": event_rules}]
    return schema


def arm_event_outcomes_schema() -> dict[str, Any]:
    """Return the closed grammar for one L3 ARM event-outcome JSONL row."""

    formal = formal_event_outcome_schema()
    variants = []
    for schema_name, schema_version in _HEADERS:
        variants.extend(
            (
                _formal_envelope(
                    formal,
                    schema_name=schema_name,
                    schema_version=schema_version,
                ),
                _right_censored_envelope(
                    formal,
                    schema_name=schema_name,
                    schema_version=schema_version,
                ),
            )
        )
    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/arm_event_outcome.l3.record.schema.json",
        "title": "L3 ARM event outcome record",
        "description": (
            "One unchanged objective/event_outcomes.jsonl row from a bound "
            "L3 ARM branch."
        ),
        "oneOf": variants,
    }
