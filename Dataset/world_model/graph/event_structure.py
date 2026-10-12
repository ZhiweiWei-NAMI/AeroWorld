"""Closed producer grammar for formal L2 event occurrences and outcomes."""

from __future__ import annotations

from copy import deepcopy
from functools import cache
from itertools import product
from typing import Any, Mapping

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types,
    get_core_predicate_templates,
)

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_VERSION = "3.0.0"
_RESTRICTED_PARAMETER_REF = (
    "aeroworld.world_truth_predicates.v3#parameter=boundary_margin_m"
)


def _string() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _tick() -> dict[str, Any]:
    return {"type": "integer", "minimum": 0}


def _object(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties) if required is None else required,
        "properties": deepcopy(dict(properties)),
    }


def _ids(*, min_items: int = 0) -> dict[str, Any]:
    return {
        "type": "array",
        "items": _string(),
        "minItems": min_items,
        "uniqueItems": True,
    }


def _fixed_array(values: list[str]) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": len(values),
        "maxItems": len(values),
        **({"prefixItems": [{"const": value} for value in values]} if values else {}),
        "items": False,
    }


@cache
def _authorities() -> tuple[tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
    rules = tuple(get_core_event_occurrence_types())
    templates = {
        template["id"]: template
        for template in get_core_predicate_templates()
        if template.get("implementation_status") == "executable_l1"
    }
    rule_ids: set[str] = set()
    event_types: set[str] = set()
    for rule in rules:
        rule_id = rule["rule_id"]
        event_type = rule["event_type"]
        if rule_id in rule_ids or event_type in event_types:
            raise ValueError(f"formal event rule or type repeats: {rule_id}")
        rule_ids.add(rule_id)
        event_types.add(event_type)
        for predicate_id in (
            rule["trigger_predicate_id"],
            rule["terminal"]["predicate_id"],
            *(item["predicate_id"] for item in rule["support_conditions"]),
        ):
            if predicate_id not in templates:
                raise ValueError(f"{rule_id}: predicate is not executable: {predicate_id}")
        if rule["trigger_direction"] not in ("rising", "falling"):
            raise ValueError(f"{rule_id}: invalid trigger direction")
        if rule["terminal"]["value"] not in ("true", "false"):
            raise ValueError(f"{rule_id}: invalid terminal truth")
        for condition in rule["support_conditions"]:
            if condition["value"] not in ("true", "false"):
                raise ValueError(f"{rule_id}: invalid support truth")
    return rules, templates


def _predicate_bindings(predicate_id: str, templates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    roles = templates[predicate_id]["argument_roles"]
    keys = [role["key"] for role in roles]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError(f"{predicate_id}: invalid predicate roles")
    return _object({key: _string() for key in keys})


def _event_bindings(rule: Mapping[str, Any]) -> dict[str, Any]:
    roles = [role["role"] for role in rule["participant_roles"]]
    if not roles or len(roles) != len(set(roles)):
        raise ValueError(f"{rule['rule_id']}: invalid event roles")
    if list(rule["event_role_sources"]) != roles:
        raise ValueError(f"{rule['rule_id']}: event role source order differs")
    return _object({role: _string() for role in roles})


def _phase(
    kind: str,
    predicate_id: str,
    direction: str,
    templates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return _object(
        {
            "phase_kind": {"const": kind},
            "predicate_id": {"const": predicate_id},
            "predicate_tuple_id": _string(),
            "predicate_bindings": _predicate_bindings(predicate_id, templates),
            "direction": {"const": direction},
            "tick": _tick(),
            "transition_id": _string(),
        }
    )


def _occurrence_phases(
    rule: Mapping[str, Any], templates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    onset = _phase(
        "onset", rule["trigger_predicate_id"], rule["trigger_direction"], templates
    )
    supports = [
        _phase(
            "escalation_support",
            condition["predicate_id"],
            "rising" if condition["value"] == "true" else "falling",
            templates,
        )
        for condition in rule["support_conditions"]
    ]
    terminal = rule["terminal"]
    terminal_phase = _phase(
        "terminal",
        terminal["predicate_id"],
        "rising" if terminal["value"] == "true" else "falling",
        templates,
    )
    alternatives = []
    for included in product((False, True), repeat=len(supports)):
        selected = [phase for phase, use in zip(supports, included) if use]
        for include_terminal in (False, True):
            sequence = [onset, *selected]
            if include_terminal:
                sequence.append(terminal_phase)
            alternatives.append(
                {
                    "type": "array",
                    "minItems": len(sequence),
                    "maxItems": len(sequence),
                    "prefixItems": sequence,
                    "items": False,
                }
            )
    return {"oneOf": alternatives}


def _occurrence_variant(
    rule: Mapping[str, Any], templates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    roles = [role["role"] for role in rule["participant_roles"]]
    parameter_refs = (
        [_RESTRICTED_PARAMETER_REF]
        if rule["event_family_id"] == "restricted_region_intrusion"
        else []
    )
    properties = {
        "event_family_id": {"const": rule["event_family_id"]},
        "event_type_id": {"const": rule["event_type"]},
        "rule_id": {"const": rule["rule_id"]},
        "trigger_predicate_id": {"const": rule["trigger_predicate_id"]},
        "bindings": _event_bindings(rule),
        "participant_roles": _fixed_array(roles),
        "supporting_parameter_refs": _fixed_array(parameter_refs),
        "event_phases": _occurrence_phases(rule, templates),
    }
    return {"properties": properties, "required": list(properties)}


@cache
def _occurrence_schema() -> dict[str, Any]:
    rules, templates = _authorities()
    properties = {
        "schema_name": {"const": "event_occurrence"},
        "schema_version": {"const": _SCHEMA_VERSION},
        "annotation_layer": {"const": "L2"},
        "source_layer": {"const": "L1"},
        "event_id": _string(),
        "episode_id": _string(),
        "event_family_id": {"type": "string"},
        "event_type_id": {"type": "string"},
        "event_subtype": {"type": "string"},
        "rule_id": {"type": "string"},
        "rule_digest": {
            "type": "string",
            "pattern": "^sha256:[0-9a-f]{64}$",
        },
        "trigger_tick": _tick(),
        "detection_tick": _tick(),
        "start_tick": _tick(),
        "end_tick": _tick(),
        "bindings": {"type": "object"},
        "participant_roles": {"type": "array"},
        "supporting_truth_ids": _ids(min_items=1),
        "supporting_transition_ids": _ids(min_items=1),
        "source_refs": _ids(min_items=1),
        "source_class": {"const": "deterministic_derived"},
        "trigger_predicate_id": {"type": "string"},
        "supporting_parameter_refs": {"type": "array"},
        "continuity_status": {"const": "strict_adjacent_transition"},
        "event_level": {"const": 2},
        "event_phases": {"type": "array", "uniqueItems": True},
        "mechanism_id": {
            "oneOf": [{"type": "string"}, {"type": "null"}],
        },
        "template_parameters": _object({}),
    }
    schema = _object(
        properties,
        required=[key for key in properties if key != "event_subtype"],
    )
    schema["$schema"] = _DRAFT
    schema["$id"] = "https://aeroworld.local/schema/event_occurrence.formal.schema.json"
    schema["title"] = "Formal event occurrence record"
    schema["allOf"] = [
        {"oneOf": [_occurrence_variant(rule, templates) for rule in rules]}
    ]
    return schema


def formal_event_occurrence_schema() -> dict[str, Any]:
    """Return the closed grammar for the current formal occurrence writer."""
    return deepcopy(_occurrence_schema())


def _terminal_hold_phase(
    rule: Mapping[str, Any], templates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    terminal = rule["terminal"]
    return _object(
        {
            "phase_kind": {"const": "terminal_hold"},
            "predicate_id": {"const": terminal["predicate_id"]},
            "predicate_tuple_id": _string(),
            "predicate_bindings": _predicate_bindings(
                terminal["predicate_id"], templates
            ),
            "value": {"const": terminal["value"]},
            "start_tick": _tick(),
            "end_tick": _tick(),
            "entry_transition_id": _string(),
            "supporting_truth_ids": _ids(min_items=1),
        }
    )


def _outcome_variant(
    rule: Mapping[str, Any], templates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    terminal = rule["terminal"]
    lifecycle = (
        "terminal_predicate_hold"
        if terminal["value"] == "true"
        else "recovery_predicate_hold"
    )
    pending = {
        "properties": {
            "status": {"const": "pending"},
            "terminal_tick": {"const": None},
            "lifecycle_status": {
                "enum": [
                    "terminal_transition_not_observed",
                    "terminal_hold_not_observed",
                ]
            },
            "supporting_truth_ids": {"maxItems": 0},
            "lifecycle_phase_evidence": {
                "type": "array",
                "minItems": 0,
                "maxItems": 0,
                "items": False,
            },
        },
        "required": [
            "status",
            "terminal_tick",
            "lifecycle_status",
            "supporting_truth_ids",
            "lifecycle_phase_evidence",
        ],
    }
    succeeded = {
        "properties": {
            "status": {"const": "succeeded"},
            "terminal_tick": _tick(),
            "lifecycle_status": {"const": lifecycle},
            "supporting_truth_ids": {"minItems": 1},
            "lifecycle_phase_evidence": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "prefixItems": [_terminal_hold_phase(rule, templates)],
                "items": False,
            },
        },
        "required": [
            "status",
            "terminal_tick",
            "lifecycle_status",
            "supporting_truth_ids",
            "lifecycle_phase_evidence",
        ],
    }
    properties = {
        "event_family_id": {"const": rule["event_family_id"]},
        "event_type_id": {"const": rule["event_type"]},
    }
    return {
        "properties": properties,
        "required": list(properties),
        "oneOf": [pending, succeeded],
    }


@cache
def _outcome_schema() -> dict[str, Any]:
    rules, templates = _authorities()
    properties = {
        "schema_name": {"const": "event_outcome"},
        "schema_version": {"const": _SCHEMA_VERSION},
        "annotation_layer": {"const": "L2"},
        "source_layer": {"const": "L1"},
        "outcome_id": _string(),
        "event_id": _string(),
        "episode_id": _string(),
        "event_family_id": {"type": "string"},
        "event_type_id": {"type": "string"},
        "status": {"enum": ["pending", "succeeded"]},
        "terminal_tick": {},
        "lifecycle_status": {
            "enum": [
                "terminal_predicate_hold",
                "recovery_predicate_hold",
                "terminal_transition_not_observed",
                "terminal_hold_not_observed",
            ]
        },
        "source_class": {"const": "deterministic_derived"},
        "supporting_truth_ids": _ids(),
        "lifecycle_phase_evidence": {"type": "array"},
    }
    schema = _object(properties)
    schema["$schema"] = _DRAFT
    schema["$id"] = "https://aeroworld.local/schema/event_outcome.formal.schema.json"
    schema["title"] = "Formal event outcome record"
    schema["allOf"] = [
        {"oneOf": [_outcome_variant(rule, templates) for rule in rules]}
    ]
    return schema


def formal_event_outcome_schema() -> dict[str, Any]:
    """Return the closed grammar for the current formal outcome writer."""
    return deepcopy(_outcome_schema())


__all__ = ["formal_event_occurrence_schema", "formal_event_outcome_schema"]
