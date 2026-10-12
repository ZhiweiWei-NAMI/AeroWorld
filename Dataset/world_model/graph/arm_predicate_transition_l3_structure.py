"""Closed record grammar for L3 ARM predicate-transition rows.

The L3 window writer stores the transition adapter's rows. The adapter builds
transitions from world-event assertions, whose grounded roles come from each
predicate's executable registry template. Only adjacent boolean flips reach
this file; continuity breaks are separate records.

Stored 3.0.0 rows have two exact envelopes: 18 keys without ``rule_digest``
and 19 keys with it. The current transition adapter also writes 3.0.0 with
the digest, so that field cannot identify a current-only predicate subset.
Both envelopes retain the executable registry vocabulary and bind every
predicate ID to its own role keys. The current producer's narrower predicate
set is an upper bound on current output, not a discriminator for frozen rows.
"""

from __future__ import annotations

from typing import Any, Mapping

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates,
)


_SCHEMA_VERSION = "3.0.0"
_EPISODE_ID_PATTERN = r"^(?:L3-1_v1|L3-2_v[12]|L3-3_v[12])__seed(?:00|01|02)$"


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
    """Pair each executable predicate with its grounded adapter role keys."""

    predicate_roles: dict[str, tuple[str, ...]] = {}
    for template in get_core_predicate_templates():
        predicate_id = template.get("id")
        if not isinstance(predicate_id, str) or not predicate_id:
            raise ValueError("core predicate template lacks an identifier")
        if template.get("implementation_status") != "executable_l1":
            continue
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
        if predicate_id in predicate_roles:
            raise ValueError(f"duplicate executable predicate: {predicate_id}")
        predicate_roles[predicate_id] = tuple(sorted(keys))
    if not predicate_roles:
        raise ValueError("core registry declares no executable predicates")
    return tuple(sorted(predicate_roles.items()))


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


def _transition_branch(
    predicate_roles: tuple[tuple[str, tuple[str, ...]], ...], *,
    with_digest: bool,
) -> dict[str, Any]:
    """Close one stored 3.0.0 key set while retaining exact PID roles."""

    tick = {"type": "integer", "minimum": 0, "maximum": 900, "multipleOf": 5}
    properties = {
        "schema_name": {"const": "predicate_transition"},
        "schema_version": {"const": _SCHEMA_VERSION},
        "annotation_layer": {"const": "L1"},
        "source_layer": {"const": "L0"},
        "materialization_policy": {"const": "predicate_delta"},
        "transition_id": {"type": "string", "pattern": r"^transition:[0-9a-f]{24}$"},
        "episode_id": {"type": "string", "pattern": _EPISODE_ID_PATTERN},
        "predicate_id": {"enum": [predicate_id for predicate_id, _ in predicate_roles]},
        "tuple_id": {"type": "string", "pattern": r"^world_predicate_tuple:[0-9a-f]{24}$"},
        "bindings": {"type": "object"},
        "from_tick": dict(tick),
        "to_tick": dict(tick),
        "from_value": {"enum": ["false", "true"]},
        "to_value": {"enum": ["false", "true"]},
        "direction": {"enum": ["rising", "falling"]},
        "evidence_truth_ids": {
            "type": "array",
            "items": {"type": "string", "pattern": r"^truth:[0-9a-f]{24}$"},
            "minItems": 2,
            "maxItems": 2,
            "uniqueItems": True,
        },
        # The adapter sorts a set union of endpoint refs. It may be empty.
        "source_refs": {
            "type": "array", "items": _string(non_empty=True),
            "uniqueItems": True,
        },
        "template_parameters": _object({}),
    }
    if with_digest:
        properties["rule_digest"] = {
            "type": "string", "pattern": "^sha256:[0-9a-f]{64}$",
        }
    schema = _object(properties)
    schema["allOf"] = [
        {
            "oneOf": [
                _bindings_variant(predicate_id, role_keys)
                for predicate_id, role_keys in predicate_roles
            ],
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
            ],
        },
    ]
    return schema


def arm_predicate_transitions_schema() -> dict[str, Any]:
    """Return the two disjoint stored envelopes and producer role bindings."""

    predicate_roles = _predicate_role_keys()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/arm_predicate_transition.l3.record.schema.json",
        "title": "L3 ARM predicate transition record",
        "oneOf": [
            _transition_branch(predicate_roles, with_digest=False),
            _transition_branch(predicate_roles, with_digest=True),
        ],
    }


__all__ = ["arm_predicate_transitions_schema"]
