"""Whole-record grammar for L4 ARM predicate-transition rows.

The formal branch follows the executable core predicate registry and adapter
role bindings. The business branch follows the current per-profile L4 business
contract, including L4-4 and L4-5. Frozen L4-9 records retain the earlier
``vehicles`` binding and omitted ``rule_digest``; current rows use ``entities``.
Both producer branches require a value change; the business branch can begin
at an unknown previous value.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates,
)
from Dataset.tools.l4_business_semantics import contract as business_contract


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_FORMAL_SCHEMA_NAME = "predicate_transition"
_FORMAL_SCHEMA_VERSION = "3.0.0"
_BUSINESS_SCHEMA_NAME = "l4_business_predicate_transition"
_BUSINESS_SCHEMA_VERSION = "1.0.0"
_DIGEST_PATTERN = "^sha256:[0-9a-f]{64}$"


def _string(*, non_empty: bool = False) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if non_empty:
        schema["minLength"] = 1
    return schema


def _object(properties: Mapping[str, Mapping[str, Any]], *,
            required: tuple[str, ...] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties) if required is None else list(required),
        "properties": {str(key): dict(value) for key, value in properties.items()},
    }


def _empty_object() -> dict[str, Any]:
    return _object({})


def _sha256_digest() -> dict[str, Any]:
    return {"type": "string", "pattern": _DIGEST_PATTERN}


def _evidence_truth_ids() -> dict[str, Any]:
    return {
        "type": "array",
        "items": _string(non_empty=True),
        "minItems": 2,
        "maxItems": 2,
    }


def _capture_tick() -> dict[str, Any]:
    return {"type": "integer", "minimum": 0, "maximum": 900, "multipleOf": 5}


def _predicate_role_keys() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Bind each executable formal predicate to its adapter role keys."""

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
        raise ValueError("core predicate registry has no executable predicates")
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


def _formal_transition_schema() -> dict[str, Any]:
    """The 19-key adapter record of ``_adapt_transitions`` lines 890-922."""

    predicate_roles = _predicate_role_keys()
    properties = {
        "schema_name": {"const": _FORMAL_SCHEMA_NAME},
        "schema_version": {"const": _FORMAL_SCHEMA_VERSION},
        "annotation_layer": {"const": "L1"},
        "source_layer": {"const": "L0"},
        "materialization_policy": {"const": "predicate_delta"},
        "transition_id": _string(non_empty=True),
        "episode_id": _string(non_empty=True),
        "predicate_id": {"enum": [predicate_id for predicate_id, _ in predicate_roles]},
        "tuple_id": _string(non_empty=True),
        # Each oneOf branch below binds this predicate to its exact role keys.
        "bindings": {"type": "object"},
        "from_tick": _capture_tick(),
        "to_tick": _capture_tick(),
        # build_transitions keeps only the boolean flips; unknown/out_of_scope
        # endpoints and same-value pairs never reach this file.
        "from_value": {"enum": ["true", "false"]},
        "to_value": {"enum": ["true", "false"]},
        "direction": {"enum": ["rising", "falling"]},
        "evidence_truth_ids": _evidence_truth_ids(),
        # The adapter unions the two endpoint evidence ref lists without a
        # non-empty check for transitions.
        "source_refs": {
            "type": "array",
            "items": _string(non_empty=True),
        },
        # Optional integrity key: emitted by the current adapter, absent in
        # frozen corpora written by the earlier revision (L3 precedent).
        "rule_digest": _sha256_digest(),
        # The adapter always emits the empty object; the compiler forbids a
        # container const, so emptiness is declared structurally.
        "template_parameters": _empty_object(),
    }
    schema = _object(properties, required=(
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
        "template_parameters",
    ))
    schema["allOf"] = [
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
    ]
    return schema


def _business_profile_variants() -> tuple[list[dict[str, Any]], list[str]]:
    """Use the producer's profile rules for predicates and binding sizes."""

    rules = business_contract()["profiles"]
    if not isinstance(rules, Mapping) or not rules:
        raise ValueError("L4 business contract has no profile rules")
    variants: list[dict[str, Any]] = []
    predicate_ids: set[str] = set()
    counts = {
        "uav_vehicle": 2,
        "uav_pedestrian": 2,
        "ambulance": 2,
        "intersection": 2,
        "av": 1,
    }
    for profile, rule in sorted(rules.items()):
        if not isinstance(profile, str) or not profile or not isinstance(rule, Mapping):
            raise ValueError("invalid L4 business profile rule")
        mechanism = rule["mechanism"]
        if mechanism not in counts:
            raise ValueError(f"{profile}: unknown L4 business mechanism {mechanism!r}")
        ids = rule["predicate_ids"]
        if (not isinstance(ids, list) or not ids or
                any(not isinstance(pid, str) or not pid for pid in ids) or
                len(ids) != len(set(ids))):
            raise ValueError(f"{profile}: invalid business predicate IDs")
        predicate_ids.update(ids)
        entity_list = {
            "type": "array",
            "items": _string(non_empty=True),
            "minItems": counts[mechanism],
            "maxItems": counts[mechanism],
            "uniqueItems": True,
        }
        binding_options = [_object({"entities": entity_list})]
        if profile in ("L4-9_v1", "L4-9_v2"):
            binding_options.append(_object({"vehicles": entity_list}))
        bindings = (binding_options[0] if len(binding_options) == 1 else
                    {"oneOf": binding_options})
        variants.append({
            "properties": {
                "episode_id": {"type": "string", "pattern": f"^{re.escape(profile)}__[^/]+$"},
                "predicate_id": {"enum": ids},
                "bindings": bindings,
            },
            "required": ["episode_id", "predicate_id", "bindings"],
        })
    return variants, sorted(predicate_ids)


def _business_transition_schema() -> dict[str, Any]:
    """The current 15-key business row and frozen L4-9 14-key row."""

    profile_variants, predicate_ids = _business_profile_variants()
    properties = {
        "schema_name": {"const": _BUSINESS_SCHEMA_NAME},
        "schema_version": {"const": _BUSINESS_SCHEMA_VERSION},
        "transition_id": _string(non_empty=True),
        "episode_id": _string(non_empty=True),
        "predicate_id": {"enum": predicate_ids},
        "from_tick": _capture_tick(),
        "to_tick": _capture_tick(),
        # The previous value may be the unknown pre-window seed; out_of_scope
        # is impossible (dense values are "unknown" or str(bool).lower()).
        "from_value": {"enum": ["true", "false", "unknown"]},
        # A tick resolving unknown produces no transition row (line 349).
        "to_value": {"enum": ["true", "false"]},
        "direction": {"enum": ["rising", "falling"]},
        "observation_boundary": {"type": "boolean"},
        "bindings": {"type": "object"},
        # Optional integrity key: the current writer always emits it
        # (lines 164, 356); frozen L4-9 rows predate it.
        "rule_digest": _sha256_digest(),
        # The writer references measured states and optionally a dispatched
        # action; observed transitions have at least one state reference.
        "source_refs": {
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
        },
        "evidence_truth_ids": _evidence_truth_ids(),
    }
    schema = _object(properties, required=(
        "schema_name",
        "schema_version",
        "transition_id",
        "episode_id",
        "predicate_id",
        "from_tick",
        "to_tick",
        "from_value",
        "to_value",
        "direction",
        "observation_boundary",
        "bindings",
        "source_refs",
        "evidence_truth_ids",
    ))
    schema["allOf"] = [
        {
            "oneOf": profile_variants,
        },
        {
            # Rows require changed values. Unknown is allowed only at the
            # observed boundary; direction is the new boolean value.
            "oneOf": [
                {
                    "properties": {
                        "from_value": {"const": "false"},
                        "to_value": {"const": "true"},
                        "direction": {"const": "rising"},
                        "observation_boundary": {"const": False},
                    },
                    "required": ["from_value", "to_value", "direction", "observation_boundary"],
                },
                {
                    "properties": {
                        "from_value": {"const": "true"},
                        "to_value": {"const": "false"},
                        "direction": {"const": "falling"},
                        "observation_boundary": {"const": False},
                    },
                    "required": ["from_value", "to_value", "direction", "observation_boundary"],
                },
                {
                    "properties": {
                        "from_value": {"const": "unknown"},
                        "to_value": {"const": "true"},
                        "direction": {"const": "rising"},
                        "observation_boundary": {"const": True},
                    },
                    "required": ["from_value", "to_value", "direction", "observation_boundary"],
                },
                {
                    "properties": {
                        "from_value": {"const": "unknown"},
                        "to_value": {"const": "false"},
                        "direction": {"const": "falling"},
                        "observation_boundary": {"const": True},
                    },
                    "required": ["from_value", "to_value", "direction", "observation_boundary"],
                },
            ],
        },
    ]
    return schema


def arm_predicate_transitions_schema() -> dict[str, Any]:
    """Return the closed schema for one L4 ``arm_predicate_transitions`` row.

    The root is a ``oneOf`` over the two producer branches, discriminated by
    the ``schema_name`` fixed in each branch. Business profile rules and
    formal predicate roles are read from the current producer authorities.
    """

    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/arm_predicate_transition.l4.record.schema.json",
        "title": "L4 arm predicate transition record",
        "oneOf": [_formal_transition_schema(), _business_transition_schema()],
    }


__all__ = ["arm_predicate_transitions_schema"]
