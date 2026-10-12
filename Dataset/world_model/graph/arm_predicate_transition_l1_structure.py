"""Closed record grammar for L1 ARM predicate-transition rows.

``l1_semantics.project`` constructs eight predicate IDs and their bindings
from exact producer branches. The adapter writes transitions only for
adjacent false-to-true or true-to-false changes; L1 projection raises when
the adapter reports a continuity break. Each predicate ID is paired with
its core registry role keys.
L1 rows carry the top-level ``rule_digest`` emitted by the adapter.

The bounded real-record sample used for this repair covers four predicate
IDs. The four lifecycle predicate IDs are declared from the producer but have
no instance in that sample (尚无实例验证).
"""

from __future__ import annotations

from typing import Any, Mapping

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates,
)


_SCHEMA_VERSION = "3.0.0"
_L1_PREDICATE_IDS = (
    "airspace.aircraft_inside_restricted_airspace",
    "airspace.aircraft_outside_assigned_altitude_band",
    "safety.aircraft_pair_below_separation",
    "airspace.airspace_corridor_over_capacity",
    "airspace.aircraft_airborne",
    "airspace.aircraft_at_home_pad",
    "airspace.aircraft_touchdown",
    "airspace.aircraft_in_landing_zone",
)


def _string(*, non_empty: bool = False) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if non_empty:
        schema["minLength"] = 1
    return schema


def _object(properties: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": {str(key): dict(value) for key, value in properties.items()},
    }


def _predicate_role_keys() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The L1 projector's eight predicate IDs paired with registry roles."""

    templates: dict[str, Mapping[str, Any]] = {}
    for template in get_core_predicate_templates():
        predicate_id = template.get("id")
        if not isinstance(predicate_id, str) or not predicate_id:
            raise ValueError("core predicate template lacks an identifier")
        if predicate_id in templates:
            raise ValueError(f"duplicate core predicate template: {predicate_id}")
        templates[predicate_id] = template
    result: list[tuple[str, tuple[str, ...]]] = []
    for predicate_id in _L1_PREDICATE_IDS:
        template = templates.get(predicate_id)
        if template is None:
            raise ValueError(f"L1 producer predicate is absent from registry: {predicate_id}")
        if template.get("implementation_status") != "executable_l1":
            raise ValueError(f"L1 producer predicate is not executable: {predicate_id}")
        roles = template.get("argument_roles")
        if not isinstance(roles, list) or not roles:
            raise ValueError(f"{predicate_id}: predicate has no argument roles")
        keys: list[str] = []
        for role in roles:
            if not isinstance(role, Mapping):
                raise ValueError(f"{predicate_id}: predicate role is not an object")
            key = role.get("key")
            if not isinstance(key, str) or not key or key in keys:
                raise ValueError(
                    f"{predicate_id}: predicate role key is invalid or duplicated"
                )
            keys.append(key)
        result.append((predicate_id, tuple(keys)))
    return tuple(result)


def _bindings_variant(predicate_id: str, role_keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        "properties": {
            "predicate_id": {"const": predicate_id},
            "bindings": _object(
                {role: _string(non_empty=True) for role in role_keys}
            ),
        },
        "required": ["predicate_id", "bindings"],
    }


def arm_predicate_transitions_schema() -> dict[str, Any]:
    """Return the closed Draft 2020-12 schema for one L1 ARM transition row."""

    predicate_roles = _predicate_role_keys()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/arm_predicate_transition.l1.record.schema.json",
        "title": "L1 ARM predicate transition record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_name",
            "schema_version",
            "annotation_layer",
            "source_layer",
            "materialization_policy",
            "transition_id",
            "episode_id",
            "predicate_id",
            "tuple_id",
            "bindings",
            "from_tick",
            "to_tick",
            "from_value",
            "to_value",
            "direction",
            "evidence_truth_ids",
            "source_refs",
            "rule_digest",
            "template_parameters",
        ],
        "properties": {
            "schema_name": {"const": "predicate_transition"},
            "schema_version": {"const": _SCHEMA_VERSION},
            "annotation_layer": {"const": "L1"},
            "source_layer": {"const": "L0"},
            "materialization_policy": {"const": "predicate_delta"},
            "transition_id": _string(non_empty=True),
            "episode_id": _string(non_empty=True),
            "predicate_id": {"enum": [predicate_id for predicate_id, _ in predicate_roles]},
            "tuple_id": _string(non_empty=True),
            # The branch below binds each predicate to its exact role keys.
            "bindings": {"type": "object"},
            "from_tick": {"type": "integer", "minimum": 0},
            "to_tick": {"type": "integer", "minimum": 0},
            "from_value": {"enum": ["false", "true"]},
            "to_value": {"enum": ["false", "true"]},
            "direction": {"enum": ["rising", "falling"]},
            # One evidence row per endpoint of the adjacent sample pair.
            "evidence_truth_ids": {
                "type": "array",
                "items": _string(non_empty=True),
                "minItems": 2,
                "maxItems": 2,
            },
            # The L1 projector always includes the raw trajectory ref plus its
            # own composed assertion ref (l1_semantics.py:96,116).
            "source_refs": {
                "type": "array",
                "items": _string(non_empty=True),
                "minItems": 1,
            },
            # Required integrity key: the adapter always emits it and every
            # frozen L1 row carries it (digest_object -> "sha256:<64 hex>").
            "rule_digest": {
                "type": "string",
                "pattern": "^sha256:[0-9a-f]{64}$",
            },
            # The adapter always emits the empty object; the compiler forbids a
            # container const, so emptiness is declared structurally.
            "template_parameters": _object({}),
        },
        "allOf": [
            {
                "oneOf": [
                    _bindings_variant(predicate_id, role_keys)
                    for predicate_id, role_keys in predicate_roles
                ]
            },
            {
                "oneOf": [
                    {
                        "properties": {
                            "direction": {"const": "rising"},
                            "from_value": {"const": "false"},
                            "to_value": {"const": "true"},
                        },
                        "required": ["direction", "from_value", "to_value"],
                    },
                    {
                        "properties": {
                            "direction": {"const": "falling"},
                            "from_value": {"const": "true"},
                            "to_value": {"const": "false"},
                        },
                        "required": ["direction", "from_value", "to_value"],
                    },
                ]
            },
        ],
    }


__all__ = ["arm_predicate_transitions_schema"]
