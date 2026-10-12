"""Closed raw producer grammars for world-truth base and delta records.

Publication filtering is separate; raw integrity fields remain required here.
"""

from __future__ import annotations

from copy import deepcopy
from functools import cache
from typing import Any, Mapping

from Dataset.semantic_truth.core_semantic_registry import load_core_semantic_registry
from Dataset.semantic_truth.world_truth import (
    BASE_SCHEMA_NAME, DELTA_SCHEMA_NAME, SCHEMA_VERSION, SCOPE_TYPES,
)
from Dataset.world_model.graph.predicate_structure import (
    _role_bindings, _world_observation_items,
)

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_DIGEST = {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}
_STRING = {"type": "string", "minLength": 1}
_TRUTH = {"enum": ["true", "false", "unknown", "out_of_scope"]}


def _object(properties: Mapping[str, Any], *, required: tuple[str, ...] | None = None) -> dict:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties) if required is None else list(required),
        "additionalProperties": False,
    }


def _strings(*, min_items: int = 0) -> dict:
    return {"type": "array", "items": _STRING, "minItems": min_items}


def _nullable_string() -> dict:
    return {"oneOf": [_STRING, {"type": "null"}]}


def _registry_parts() -> tuple[dict, list[dict], list[dict]]:
    registry = load_core_semantic_registry()
    templates = registry["templates"]
    executable = [
        template for template in templates
        if template["implementation_status"] == "executable_l1"
    ]
    if (len(templates) != 79 or len(executable) != 72
            or set(registry["implementation"]["executable_predicate_ids"])
            != {template["id"] for template in executable}):
        raise ValueError("world-truth predicate registry differs from producer contract")
    return registry, templates, executable


def _scope_node_schema() -> dict:
    variants = []
    for scope_type in SCOPE_TYPES:
        source_id = ({"type": "null"} if scope_type == "scene"
                     else _nullable_string() if scope_type == "compute_node"
                     else _STRING)
        subtype = _STRING if scope_type == "facility" else {"type": "null"}
        ontology_class = ({"type": "null"} if scope_type == "scene" else _STRING)
        variants.append(_object({
            "node_id": _STRING,
            "kind": {"const": "world_scope_entity"},
            "scope_type": {"const": scope_type},
            "entity_id": _STRING,
            "source_entity_id": source_id,
            "scope_subtype": subtype,
            "ontology_class_id": ontology_class,
        }))
    return {"oneOf": variants}


def _non_executable_schema(contracts: Mapping[str, Any]) -> dict:
    authority = {"oneOf": [
        _object({"kind": {"const": "ontology_role_individual"},
                 "ontology_class": _STRING}),
        _object({"kind": {"const": "supporting_individual"},
                 "ontology_class": _STRING}),
        _object({"kind": {"const": "runtime_state_field"},
                 "field": _STRING}),
    ]}
    contract = _object({
        "reason_code": _STRING,
        "missing_authorities": {"type": "array", "items": authority, "minItems": 1},
        "closure_statement": _STRING,
    })
    return _object({predicate_id: contract for predicate_id in contracts})


def _governed_parameters_schema(defaults: Mapping[str, Any]) -> dict:
    properties = {}
    for key, value in defaults.items():
        if type(value) is int:
            properties[key] = {"type": "integer"}
        elif type(value) is float:
            properties[key] = {
                "type": "number", "x-source-value-types": ["float"]
            }
        else:
            raise TypeError(f"unsupported governed parameter source type: {key}")
    return _object(properties)


def _role_properties(template: Mapping[str, Any]) -> dict:
    grounding = template["grounding_spec"]
    authority = grounding["candidate_authority"]
    provenance = {}
    classes = {}
    for role in template["argument_roles"]:
        key = role["key"]
        source = grounding["role_sources"][key]
        binding_source = ("scope_entity" if source["kind"] == "scope_entity"
                          else source["path"])
        classes[key] = _STRING
        provenance[key] = _object({
            "entity_id": _STRING,
            "ontology_class_id": _STRING,
            "candidate_authority_kind": {"const": authority["kind"]},
            "candidate_authority_source": {"const": authority["source"]},
            "binding_source": {"const": binding_source},
        })
    return {
        "predicate_id": {"const": template["id"]},
        "bindings": _role_bindings(template),
        "binding_ontology_classes": _object(classes),
        "binding_provenance": _object(provenance),
        "candidate_authority": _object({
            "kind": {"const": authority["kind"]},
            "source": {"const": authority["source"]},
        }),
    }


def _observations_schema(template: Mapping[str, Any]) -> dict:
    return {
        "type": "array",
        "items": {"oneOf": _world_observation_items(template)},
    }


def _assertion_schema(template: Mapping[str, Any]) -> dict:
    roles = [role["key"] for role in template["argument_roles"]]
    role_properties = _role_properties(template)
    return _object({
        "assertion_id": _STRING,
        "predicate_id": role_properties["predicate_id"],
        "ontology_argument_roles": {
            "type": "array",
            "prefixItems": [{"const": role} for role in roles],
            "items": False,
            "minItems": len(roles),
            "maxItems": len(roles),
        },
        "tuple_id": _STRING,
        "bindings": role_properties["bindings"],
        "binding_ontology_classes": role_properties["binding_ontology_classes"],
        "binding_provenance": role_properties["binding_provenance"],
        "value": _TRUTH,
        "missing_source_record": _strings(),
        "observations": _observations_schema(template),
        "source_refs": _strings(),
        "candidate_authority": role_properties["candidate_authority"],
        "rule_digest": _DIGEST,
        "parameter_digest": _DIGEST,
        "input_digest": _DIGEST,
        "truth_state_update_tick": {"type": "integer", "minimum": 0},
        "evidence_update_tick": {"type": "integer", "minimum": 0},
    })


def _identity_properties(template: Mapping[str, Any]) -> dict:
    role_properties = _role_properties(template)
    return {
        "assertion_id": _STRING,
        "predicate_id": role_properties["predicate_id"],
        "tuple_id": _STRING,
        "bindings": role_properties["bindings"],
        "binding_ontology_classes": role_properties["binding_ontology_classes"],
    }


def _operation_schemas(executable: list[dict]) -> list[dict]:
    add = _object({
        "operation": {"const": "add_predicate_assertion"},
        "assertion": {"oneOf": [_assertion_schema(t) for t in executable]},
    })
    variants = [add]
    for template in executable:
        identity = _identity_properties(template)
        variants.extend((
            _object({
                "operation": {"const": "remove_predicate_assertion"},
                **identity,
                "from_value": _TRUTH,
            }),
            _object({
                "operation": {"const": "set_predicate_value"},
                **identity,
                "from_value": _TRUTH,
                "to_value": _TRUTH,
                "missing_source_record": _strings(),
                "observations": _observations_schema(template),
                "source_refs": _strings(),
                "rule_digest": _DIGEST,
            }),
            _object({
                "operation": {"const": "refresh_predicate_evidence"},
                **identity,
                "value": _TRUTH,
                "missing_source_record": _strings(),
                "observations": _observations_schema(template),
                "source_refs": _strings(),
            }),
        ))
    return variants


@cache
def _world_truth_base_schema() -> dict:
    registry, templates, executable = _registry_parts()
    implementation = registry["implementation"]
    predicate_ids = [t["id"] for t in templates]
    executable_ids = [t["id"] for t in executable]
    candidate_counts = {
        "type": "object",
        "propertyNames": {"enum": executable_ids},
        "additionalProperties": {"type": "integer", "minimum": 0},
        "minProperties": len(executable_ids),
        "maxProperties": len(executable_ids),
    }
    schema = _object({
        "schema_name": {"const": BASE_SCHEMA_NAME},
        "schema_version": {"const": SCHEMA_VERSION},
        "annotation_layer": {"const": "L1"},
        "source_layer": {"const": "L0"},
        "episode_id": _STRING,
        "tick": {"const": 0},
        "representation": {"const": "grounded_predicate_candidate_base"},
        "candidate_set_complete": {"const": True},
        "candidate_set_scope": {"const": "executable_l1_predicates_only"},
        "predicate_vocabulary": {
            "type": "array", "items": {"enum": predicate_ids},
            "minItems": len(predicate_ids), "maxItems": len(predicate_ids),
            "uniqueItems": True,
        },
        "executable_predicate_ids": {
            "type": "array", "items": {"enum": executable_ids},
            "minItems": len(executable_ids), "maxItems": len(executable_ids),
            "uniqueItems": True,
        },
        "non_executable_predicates": _non_executable_schema(
            implementation["non_executable_predicates"]
        ),
        "scope_entities": {
            "type": "array", "items": _scope_node_schema(), "minItems": 1,
        },
        "initial_assertions": {
            "type": "array",
            "items": {"oneOf": [_assertion_schema(t) for t in executable]},
        },
        "summary": _object({
            "predicate_count": {"const": len(predicate_ids)},
            "executable_predicate_count": {"const": len(executable_ids)},
            "non_executable_predicate_count": {
                "const": len(implementation["non_executable_predicates"])
            },
            "initial_assertion_count": {"type": "integer", "minimum": 0},
            "initial_candidate_counts_by_predicate": candidate_counts,
        }),
        "input_digest": _DIGEST,
        "parameter_digest": _DIGEST,
        "governed_parameters": _governed_parameters_schema(
            registry["parameter_governance"]["governed_defaults"]
        ),
        "base_graph_digest": _DIGEST,
    })
    schema["$schema"] = _DRAFT
    return schema


@cache
def _world_truth_delta_schema() -> dict:
    _registry, _templates, executable = _registry_parts()
    schema = _object({
        "schema_name": {"const": DELTA_SCHEMA_NAME},
        "schema_version": {"const": SCHEMA_VERSION},
        "annotation_layer": {"const": "L1"},
        "source_layer": {"const": "L0"},
        "episode_id": _STRING,
        "tick": {
            "type": "integer", "minimum": 5, "maximum": 900, "multipleOf": 5,
        },
        "delta_id": _STRING,
        "previous_delta_id": _nullable_string(),
        "operations": {
            "type": "array", "items": {"oneOf": _operation_schemas(executable)},
            "minItems": 1,
        },
        "operation_count": {"type": "integer", "minimum": 1},
        "base_graph_digest": _DIGEST,
        "delta_digest": _DIGEST,
    })
    schema["$schema"] = _DRAFT
    return schema


def world_truth_base_schema() -> dict:
    return deepcopy(_world_truth_base_schema())


def world_truth_delta_schema() -> dict:
    return deepcopy(_world_truth_delta_schema())


__all__ = ["world_truth_base_schema", "world_truth_delta_schema"]
