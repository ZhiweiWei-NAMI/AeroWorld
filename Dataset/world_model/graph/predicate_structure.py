"""Closed producer grammar for formal predicate records.

The semantic registry owns predicate identifiers, argument roles, ontology
classes, and the value units used by formal evidence.  This module turns that
registry into exact Draft 2020-12 record schemas.  The schemas describe the
current formal producer only; historical ARM producer variants are separate
contracts and are intentionally not accepted here.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import cache
import json
from pathlib import Path
from typing import Any, Mapping

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates,
)


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_TRUTH_VALUES = ("true", "false", "unknown", "out_of_scope")
_BOOLEAN_TRUTH_VALUES = ("true", "false")
_FORMAL_SCHEMA_VERSION = "3.1.0"
_DELTA_SCHEMA_VERSION = "3.0.0"
_ROLE_CLASS_PREFIX = "world:"


def _executable_templates() -> tuple[Mapping[str, Any], ...]:
    return tuple(
        template for template in get_core_predicate_templates()
        if template.get("implementation_status") == "executable_l1"
    )


def _object(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    required: tuple[str, ...] | None = None,
    additional_properties: bool = False,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": additional_properties,
        "required": list(properties) if required is None else list(required),
        "properties": {str(key): deepcopy(dict(value)) for key, value in properties.items()},
    }


def _empty_object() -> dict[str, Any]:
    return _object({})


def _string(*, non_empty: bool = False) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if non_empty:
        schema["minLength"] = 1
    return schema


def _string_list(*, min_items: int = 0) -> dict[str, Any]:
    return {"type": "array", "items": _string(), "minItems": min_items}


def _truth(*, boolean_only: bool = False) -> dict[str, Any]:
    return {"enum": list(_BOOLEAN_TRUTH_VALUES if boolean_only else _TRUTH_VALUES)}


def _unit_value_schema(unit: str) -> dict[str, Any]:
    """Map a registry evidence unit to one explicit JSON scalar grammar."""

    if unit == "bool":
        return {"type": "boolean"}
    if unit == "four-value":
        return _truth()
    if unit in {"identifier", "state-code"}:
        return _string(non_empty=True)
    if unit in {
        "aircraft",
        "vehicle",
        "m",
        "m/s",
        "degC",
        "lux",
        "ratio",
    }:
        # Producers preserve integer counts in some families and emit numeric
        # projections in others; JSON number covers both without opening the
        # field to strings or containers.
        return {"type": "number"}
    raise ValueError(f"unsupported formal predicate evidence unit: {unit!r}")


def _role_bindings(template: Mapping[str, Any]) -> dict[str, Any]:
    roles = template.get("argument_roles")
    if not isinstance(roles, list) or not roles:
        raise ValueError(f"{template.get('id')!r}: predicate has no argument roles")
    properties: dict[str, dict[str, Any]] = {}
    for role in roles:
        if not isinstance(role, Mapping):
            raise ValueError(f"{template.get('id')!r}: predicate role is not an object")
        key = role.get("key")
        ontology_class = role.get("class")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{template.get('id')!r}: predicate role key is invalid")
        if not isinstance(ontology_class, str) or not ontology_class:
            raise ValueError(f"{template.get('id')!r}: predicate role class is invalid")
        if key in properties:
            raise ValueError(f"{template.get('id')!r}: duplicate predicate role {key!r}")
        properties[key] = _string(non_empty=True)
    return _object(properties)


def _role_classes(template: Mapping[str, Any]) -> dict[str, Any]:
    roles = template.get("argument_roles")
    if not isinstance(roles, list):
        raise ValueError(f"{template.get('id')!r}: predicate roles are not an array")
    properties: dict[str, dict[str, Any]] = {}
    for role in roles:
        if not isinstance(role, Mapping):
            raise ValueError(f"{template.get('id')!r}: predicate role is not an object")
        key = role.get("key")
        ontology_class = role.get("class")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{template.get('id')!r}: predicate role key is invalid")
        if not isinstance(ontology_class, str) or not ontology_class:
            raise ValueError(f"{template.get('id')!r}: predicate role class is invalid")
        properties[key] = {"const": _ROLE_CLASS_PREFIX + ontology_class}
    return _object(properties)


def _field_observation(template: Mapping[str, Any], field: Mapping[str, Any]) -> dict[str, Any]:
    path = field.get("field")
    unit = field.get("unit")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{template.get('id')!r}: evidence field path is invalid")
    if not isinstance(unit, str) or not unit:
        raise ValueError(f"{template.get('id')!r}: evidence field unit is invalid")
    return _object(
        {
            "path": {"const": path},
            "value": _unit_value_schema(unit),
            "source_ref": _string(non_empty=True),
        },
        required=("path", "value"),
    )


def _unavailable_observation(predicate_id: str) -> dict[str, Any]:
    unavailable = _object(
        {
            "contract_id": _string(non_empty=True),
            "predicate_id": {"const": predicate_id},
            "scope_type": {
                "enum": [
                    "uav",
                    "vehicle",
                    "pedestrian",
                    "facility",
                    "compute_node",
                    "scene",
                ]
            },
            "scope_entity_id": _string(non_empty=True),
            "executable": {"const": False},
            "non_executable_reason": _string(non_empty=True),
            "closure_statement": _string(non_empty=True),
            "source_ref": _string(non_empty=True),
        }
    )
    return _object(
        {
            "path": {"const": "l0.predicate_source_unavailability"},
            "value": unavailable,
            "source_ref": _string(non_empty=True),
        },
        required=("path", "value"),
    )


def _scope_inactive_observation() -> dict[str, Any]:
    return _object(
        {
            "path": {"const": "l0.scope_active"},
            "value": {"const": False},
            "source_ref": _string(non_empty=True),
        },
        required=("path", "value"),
    )


def _world_observation_items(template: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = template.get("state_contract")
    if not isinstance(contract, Mapping):
        # Non-executable predicates do not have formal evidence rows.  Keep an
        # explicit empty branch so the top-level registry mapping stays exact.
        fields: list[Any] = []
    else:
        fields = contract.get("required_fields") or []
    if not isinstance(fields, list):
        raise ValueError(f"{template.get('id')!r}: required_fields is not an array")
    items = [_field_observation(template, field) for field in fields]
    items.extend(
        (
            _unavailable_observation(str(template["id"])),
            _scope_inactive_observation(),
        )
    )
    return items


def _evidence_schema(template: Mapping[str, Any]) -> dict[str, Any]:
    predicate_id = template.get("id")
    if not isinstance(predicate_id, str) or not predicate_id:
        raise ValueError("predicate template lacks an identifier")
    roles = tuple(
        str(role["key"])
        for role in template.get("argument_roles", ())
        if isinstance(role, Mapping) and isinstance(role.get("key"), str)
    )
    if not roles:
        raise ValueError(f"{predicate_id}: predicate has no argument roles")
    world_observations = {
        "type": "array",
        "items": {
            "oneOf": _world_observation_items(template),
        },
        "minItems": 0,
    }
    value = _object(
        {
            "world_truth_value": _truth(),
            "world_observations": world_observations,
            "truth_state_update_tick": {"type": "integer", "minimum": 0},
            "evidence_update_tick": {"type": "integer", "minimum": 0},
            "missing_source_record": _string_list(),
            "bindings": _role_bindings(template),
            "binding_ontology_classes": _role_classes(template),
        }
    )
    return _object(
        {
            "source_refs": _string_list(),
            "observations": {
                "type": "array",
                "items": {
                    "oneOf": [
                        _object(
                            {
                                "path": {
                                    "const": "normalized.grounded_roles__"
                                    + "__".join(roles)
                                },
                                "value": value,
                            },
                            required=("path", "value"),
                        )
                    ]
                },
                "minItems": 1,
            },
            "missing_requirements": _string_list(),
        }
    )


def _base_properties(
    *,
    schema_name: str,
    schema_version: str,
    materialization_policy: str,
    identity: str,
) -> dict[str, dict[str, Any]]:
    properties: dict[str, dict[str, Any]] = {
        "schema_name": {"const": schema_name},
        "schema_version": {"const": schema_version},
        "annotation_layer": {"const": "L1"},
        "source_layer": {"const": "L0"},
        "materialization_policy": {"const": materialization_policy},
        identity: _string(non_empty=True),
        "episode_id": _string(non_empty=True),
        "predicate_id": _string(non_empty=True),
        "tuple_id": _string(non_empty=True),
        "bindings": {"type": "object"},
        "template_parameters": _empty_object(),
    }
    return properties


def _predicate_truth_base() -> dict[str, Any]:
    properties = _base_properties(
        schema_name="predicate_truth",
        schema_version=_FORMAL_SCHEMA_VERSION,
        materialization_policy="on_demand_epi_predicate",
        identity="truth_id",
    )
    properties.update(
        {
            "tick": {"type": "integer", "minimum": 0},
            "truth_state_update_tick": {"type": "integer", "minimum": 0},
            "evidence_update_tick": {"type": "integer", "minimum": 0},
            "authoritative_tick": {"const": True},
            "value": _truth(),
            "evidence": {"type": "object"},
            "source_class": {
                "enum": ["deterministic_derived", "simulated_derived"]
            },
            # The closed grammar compiler requires mixed scalar types to be
            # expressed as explicit branches rather than a JSON Schema type
            # array.
            "mechanism_id": {
                "oneOf": [{"type": "string"}, {"type": "null"}],
            },
            "rule_digest": {
                "type": "string",
                "pattern": "^sha256:[0-9a-f]{64}$",
            },
            "parameter_digest": {
                "type": "string",
                "pattern": "^sha256:[0-9a-f]{64}$",
            },
            "input_digest": {
                "type": "string",
                "pattern": "^sha256:[0-9a-f]{64}$",
            },
        }
    )
    return _object(properties)


def _predicate_truth_variant(template: Mapping[str, Any]) -> dict[str, Any]:
    predicate_id = str(template["id"])
    return {
        "properties": {
            "predicate_id": {"const": predicate_id},
            "bindings": _role_bindings(template),
            "evidence": _evidence_schema(template),
        },
        "required": ["predicate_id", "bindings", "evidence"],
    }


@cache
def _formal_predicate_truth_schema() -> dict[str, Any]:
    schema = _predicate_truth_base()
    schema["$schema"] = _DRAFT
    schema["$id"] = "https://aeroworld.local/schema/predicate_truth.formal.schema.json"
    schema["title"] = "Formal predicate truth record"
    schema["required"] = [
        "schema_name",
        "schema_version",
        "annotation_layer",
        "source_layer",
        "materialization_policy",
        "truth_id",
        "episode_id",
        "tick",
        "truth_state_update_tick",
        "evidence_update_tick",
        "authoritative_tick",
        "predicate_id",
        "tuple_id",
        "bindings",
        "value",
        "evidence",
        "source_class",
        "template_parameters",
        "rule_digest",
        "parameter_digest",
        "input_digest",
    ]
    schema["allOf"] = [
        {"oneOf": [_predicate_truth_variant(template) for template in _executable_templates()]}
    ]
    return schema


def _delta_base(schema_name: str) -> dict[str, Any]:
    properties = _base_properties(
        schema_name=schema_name,
        schema_version=_DELTA_SCHEMA_VERSION,
        materialization_policy="predicate_delta",
        identity="transition_id",
    )
    properties.update(
        {
            "from_tick": {"type": "integer", "minimum": 0},
            "to_tick": {"type": "integer", "minimum": 0},
            "from_value": _truth(boolean_only=schema_name == "predicate_transition"),
            "to_value": _truth(boolean_only=schema_name == "predicate_transition"),
            "direction": {"type": "string"},
            "evidence_truth_ids": {
                "type": "array",
                "items": _string(non_empty=True),
                "minItems": 2,
                "maxItems": 2,
            },
            "source_refs": _string_list(),
            "rule_digest": {
                "type": "string",
                "pattern": "^sha256:[0-9a-f]{64}$",
            },
        }
    )
    return _object(properties)


def _delta_variants(*, continuity_break: bool) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for template in _executable_templates():
        predicate_id = str(template["id"])
        variants.append(
            {
                "properties": {
                    "predicate_id": {"const": predicate_id},
                    "bindings": _role_bindings(template),
                },
                "required": ["predicate_id", "bindings"],
            }
        )
    return variants


@cache
def _predicate_transition_schema() -> dict[str, Any]:
    schema = _delta_base("predicate_transition")
    schema["$schema"] = _DRAFT
    schema["$id"] = "https://aeroworld.local/schema/predicate_transition.formal.schema.json"
    schema["title"] = "Formal predicate transition record"
    schema["required"] = [
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
        "rule_digest",
    ]
    schema["allOf"] = [
        {"oneOf": _delta_variants(continuity_break=False)},
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


@cache
def _predicate_continuity_break_schema() -> dict[str, Any]:
    schema = _delta_base("predicate_continuity_break")
    schema["$schema"] = _DRAFT
    schema["$id"] = "https://aeroworld.local/schema/predicate_continuity_break.formal.schema.json"
    schema["title"] = "Formal predicate continuity break record"
    schema["properties"].update(
        {
            "break_id": _string(non_empty=True),
            "direction": {"const": "continuity_break"},
            "reason": {"const": "non_adjacent_formal_samples"},
        }
    )
    schema["required"] = [
        "schema_name",
        "schema_version",
        "annotation_layer",
        "source_layer",
        "materialization_policy",
        "transition_id",
        "break_id",
        "episode_id",
        "predicate_id",
        "tuple_id",
        "bindings",
        "from_tick",
        "to_tick",
        "from_value",
        "to_value",
        "direction",
        "reason",
        "evidence_truth_ids",
        "source_refs",
        "template_parameters",
        "rule_digest",
    ]
    schema["properties"]["from_value"] = _truth()
    schema["properties"]["to_value"] = _truth()
    schema["allOf"] = [{"oneOf": _delta_variants(continuity_break=True)}]
    return schema


def formal_predicate_truth_schema() -> dict[str, Any]:
    """Return the closed schema for current formal 3.1.0 predicate truth."""

    return deepcopy(_formal_predicate_truth_schema())


def native_predicate_truth_schema() -> dict[str, Any]:
    """Render the native schema from the graph's closed producer grammar."""

    schema = formal_predicate_truth_schema()
    schema["$id"] = "https://aeroworld.local/schema/predicate_truth.schema.json"
    schema["title"] = "Predicate truth record"
    return schema


def predicate_transition_schema() -> dict[str, Any]:
    """Return the current formal predicate transition schema."""

    return deepcopy(_predicate_transition_schema())


def predicate_continuity_break_schema() -> dict[str, Any]:
    """Return the dedicated non-adjacent formal sample schema."""

    return deepcopy(_predicate_continuity_break_schema())


__all__ = [
    "formal_predicate_truth_schema",
    "native_predicate_truth_schema",
    "predicate_transition_schema",
    "predicate_continuity_break_schema",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-native-truth", action="store_true")
    action.add_argument("--check-native-truth", action="store_true")
    args = parser.parse_args(argv)
    path = (Path(__file__).resolve().parents[2]
            / "semantic_rules/schema/predicate_truth.schema.json")
    expected = native_predicate_truth_schema()
    if args.write_native_truth:
        path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    elif json.loads(path.read_text(encoding="utf-8")) != expected:
        raise ValueError(f"native predicate truth schema differs from producer grammar: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
