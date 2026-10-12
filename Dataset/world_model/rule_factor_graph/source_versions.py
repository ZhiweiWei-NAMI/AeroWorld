"""Resolve declared producer semantics against the current YAML source contract.

Producer versions and semantic binding versions are separate authority. A stored
3.0.0 record without a semantic declaration cannot identify which historical
field mapping or cooperation policy produced its published label.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.parse import parse_qsl

from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates
from Dataset.semantic_truth.world_truth import _world_class
from .unified_contracts import ContractError, canonical_class_is_a


class SourceVersionError(ContractError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


class UndeclaredSourceSemantics(SourceVersionError):
    """Recorded producer versions do not declare the source field semantics."""


@dataclass(frozen=True)
class ResolvedSourceVersion:
    producer_identity: tuple[str, ...]
    semantic_contract: Mapping[str, Any]
    bindings: Mapping[str, str]
    context: Mapping[str, Any]
    source_ref: str
    status: str = "current_declared_source"


PRODUCER_IDENTITY_FIELDS = (
    "schema_name", "schema_version", "model_id", "model_version",
    "rule_id", "rule_version", "observation_family",
)

CURRENT_SOURCE_DECLARATIONS = {
    ("geometric_computer", "geometric_computer.restricted_region_distance",
     "predicate_contract_aircraft_region_geometry"): {
        "contract_id": "restricted_prism_signed_distance", "version": "1.0.0",
        "field_sources": {
            "geometry.restricted_region_active": "record:values.restricted_region_active",
            "geometry.minimum_restricted_boundary_distance_m": "record:values.signed_distance_to_boundary_m",
        },
        "yaml_field_sources": {
            "geometry.restricted_region_active": "record:values.restricted_region_active",
            "geometry.minimum_restricted_boundary_distance_m": "record:values.signed_distance_to_boundary_m",
        },
        "field_units": {"geometry.restricted_region_active": "bool",
                        "geometry.minimum_restricted_boundary_distance_m": "m"},
        "role_sources": {"actor": "values.aircraft_id", "restricted_region": "values.restricted_region_id"},
    },
    ("plan_window_computer", "plan_window_computer.aircraft_control_and_compliance",
     "predicate_contract_aircraft_plan"): {
        "contract_id": "observed_aircraft_cooperation", "version": "1.0.0",
        "field_sources": {"plan.cooperative_flag": "record:values.cooperative_flag"},
        "yaml_field_sources": {"plan.cooperative_flag": "context:plan.cooperative_flag"},
        "field_units": {"plan.cooperative_flag": "bool"},
        "role_sources": {"aircraft": "subject_id"},
        "observed_input_sources": {
            "plan.cooperative_flag": "truth_frames.entities[].control_state.cooperative_flag",
        },
        "authored_value_contract": "strict_bool_when_present",
        "observation_missing_policy": "absent_or_null_is_unknown",
    },
}


def _required(mapping: Mapping, key: str):
    if key not in mapping:
        raise SourceVersionError("missing_source_authority", key)
    return mapping[key]


def _path(record: Mapping, path: str):
    value = record
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise SourceVersionError("missing_declared_source_field", path)
        value = value[key]
    return value


def _set_path(context: dict, path: str, value):
    parts = path.split(".")
    target = context
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    target[parts[-1]] = deepcopy(value)


def resolve_source_version(record: Mapping[str, Any], *, predicate_id: str,
                           episode_id: str, tick: int, bindings: Mapping[str, Mapping[str, str]],
                           source_ref: str) -> ResolvedSourceVersion:
    """Validate exact producer/binding authority before reading field values.

    The predicate ID selects the normative YAML contract supplied by the caller;
    producer dispatch uses its complete declared identity and semantic contract.
    This function rejects missing or incompatible authority and never guesses a
    version from a value, sign, actor name, role or published label.
    """
    if not source_ref or not isinstance(source_ref, str):
        raise SourceVersionError("missing_source_reference", "exact source record reference required")
    identity = tuple(_required(record, field) for field in PRODUCER_IDENTITY_FIELDS)
    if any(not isinstance(value, str) or not value for value in identity):
        raise SourceVersionError("invalid_producer_identity", str(identity))
    key = (record["model_id"], record["rule_id"], record["observation_family"])
    if key not in CURRENT_SOURCE_DECLARATIONS:
        raise SourceVersionError("unsupported_declared_producer", str(key))
    version_tuple = (record["schema_name"], record["schema_version"], record["model_version"], record["rule_version"])
    if version_tuple not in {
            ("domain_state_observation", "3.0.0", "3.0.0", "3.0.0"),
            ("domain_state_observation", "3.0.0", "4.0.0", "4.0.0")}:
        raise SourceVersionError("unsupported_declared_producer_version", str(identity))
    if (record.get("episode_id") != episode_id or record.get("tick") != tick
            or type(tick) is not int or type(record.get("tick")) is not int):
        raise SourceVersionError("source_episode_tick_mismatch", f"expected {episode_id}@{tick}")
    if "source_semantic_contract" not in record:
        code = ("undeclared_legacy_semantic_source_binding" if record["model_version"] == "3.0.0"
                else "undeclared_current_semantic_source_binding")
        raise UndeclaredSourceSemantics(code,
            "schema/model/rule versions alone do not identify the source mapping")
    expected = CURRENT_SOURCE_DECLARATIONS[key]
    declaration = record["source_semantic_contract"]
    if declaration != expected:
        raise SourceVersionError("incompatible_semantic_source_declaration",
            "declared contract/version/fields/roles/units do not match current source authority")
    if (record["schema_name"], record["schema_version"], record["model_version"], record["rule_version"]) != (
            "domain_state_observation", "3.0.0", "4.0.0", "4.0.0"):
        raise SourceVersionError("unsupported_declared_producer_version", str(identity))
    templates = {item["id"]: item for item in get_core_predicate_templates()}
    if predicate_id not in templates or templates[predicate_id]["implementation_status"] != "executable_l1":
        raise SourceVersionError("invalid_normative_contract", predicate_id)
    template = templates[predicate_id]
    fields = {item["field"]: item["unit"] for item in template["state_contract"]["required_fields"]}
    if fields != declaration["field_units"] or template["grounding_spec"]["field_sources"] != declaration["yaml_field_sources"]:
        raise SourceVersionError("current_yaml_source_binding_mismatch", predicate_id)
    declared_roles = {role["key"]: role["class"] for role in template["argument_roles"]}
    if set(bindings) != set(declared_roles) or set(bindings) != set(declaration["role_sources"]):
        raise SourceVersionError("source_role_set_mismatch", str(tuple(bindings)))
    resolved_bindings = {}
    for role, source in declaration["role_sources"].items():
        entity = bindings[role]
        entity_id = _required(entity, "entity_id")
        ontology_class = _required(entity, "ontology_class")
        if not isinstance(entity_id, str) or not entity_id:
            raise SourceVersionError("invalid_source_role_identifier", role)
        required_class = _world_class(declared_roles[role])
        if not canonical_class_is_a(ontology_class, required_class):
            raise SourceVersionError("source_role_class_mismatch", role)
        if _path(record, source) != entity_id:
            raise SourceVersionError("source_role_binding_mismatch", role)
        resolved_bindings[role] = entity_id
        if source.startswith("values."):
            class_source = source[:-3] + "_ontology_class_id"
            if not canonical_class_is_a(_path(record, class_source), required_class):
                raise SourceVersionError("record_role_class_mismatch", role)
    subject = ("|".join(resolved_bindings[role] for role in declaration["role_sources"])
               if len(resolved_bindings) > 1 else next(iter(resolved_bindings.values())))
    if record.get("subject_id") != subject:
        raise SourceVersionError("source_subject_mismatch", subject)
    context = {}
    for field, source in declaration["field_sources"].items():
        if not source.startswith("record:"):
            raise SourceVersionError("invalid_declared_record_source", source)
        value = _path(record, source[len("record:"):])
        unit = fields[field]
        if value is not None and ((unit == "bool" and type(value) is not bool)
                                 or (unit == "m" and (type(value) not in (int, float) or not math.isfinite(value)))):
            raise SourceVersionError("source_field_unit_type_mismatch", field)
        _set_path(context, field, value)
    if "observed_input_sources" in declaration:
        observations = _required(record, "input_observations")
        if not isinstance(observations, list) or len(observations) != 1:
            raise SourceVersionError("invalid_observed_input_authority", "one exact cooperation observation required")
        observation = observations[0]
        expected_observation = {
            "episode_id": episode_id,
            "input_selector": declaration["observed_input_sources"]["plan.cooperative_flag"],
            "tick": tick, "entity_id": resolved_bindings["aircraft"],
            "field": "control_state.cooperative_flag", "value": record["values"]["cooperative_flag"],
        }
        if any(_required(observation, key) != value for key, value in expected_observation.items()):
            raise SourceVersionError("observed_input_binding_mismatch", "cooperation observation does not identify this exact input")
        if type(observation["tick"]) is not int:
            raise SourceVersionError("invalid_observed_input_tick_type", "tick must be an integer")
        if not isinstance(_required(observation, "source_ref"), str) or not observation["source_ref"]:
            raise SourceVersionError("missing_observed_input_reference", "source_ref")
        path = _required(observation, "path")
        if not isinstance(path, str) or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or PurePosixPath(path).name != "truth_frames.jsonl":
            raise SourceVersionError("invalid_observed_input_path", str(path))
        if observation["source_ref"] not in _required(record, "source_refs"):
            raise SourceVersionError("observed_input_reference_not_declared", observation["source_ref"])
        ref_path, separator, fragment = observation["source_ref"].partition("#")
        pairs = parse_qsl(fragment, keep_blank_values=True)
        ref_fields = dict(pairs)
        if (not separator or ref_path != path or len(pairs) != len(ref_fields)
                or set(ref_fields) != {"row", "tick", "entity", "field"}
                or ref_fields["tick"] != str(tick) or ref_fields["entity"] != resolved_bindings["aircraft"]
                or ref_fields["field"] != observation["field"] or not ref_fields["row"].isdigit()
                or int(ref_fields["row"]) < 1):
            raise SourceVersionError("observed_input_reference_binding_mismatch", observation["source_ref"])
        status = _required(observation, "status")
        value = observation["value"]
        if (value is not None and (type(value) is not bool or status != "observed")) or (
                value is None and status not in {"null", "missing_entity", "missing_field"}):
            raise SourceVersionError("observed_input_status_value_mismatch", str(status))
    return ResolvedSourceVersion(identity, deepcopy(declaration), resolved_bindings, context, source_ref)
