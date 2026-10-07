"""Load the ontology-derived V3 predicate registry and fail closed on drift."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping


DATASET_ROOT = Path(__file__).resolve().parents[1]
CORE_REGISTRY_PATH = (
    DATASET_ROOT
    / "semantic_rules"
    / "predicates"
    / "core_semantic_predicate_templates.json"
)
DOMAIN_ONTOLOGY_ROOT = DATASET_ROOT / "knowledge_graph" / "domain"
EVENT_ONTOLOGY_ROOT = DATASET_ROOT / "knowledge_graph" / "events"
EVENT_FAMILY_MAPPING_PATH = (
    DATASET_ROOT / "semantic_rules" / "profiles" / "event_family_authority_mapping.json"
)
OBJECTIVE_CONTRACT_PATH = (
    DATASET_ROOT
    / "semantic_rules"
    / "profiles"
    / "epi_objective_semantic_contract.json"
)
EXPECTED_ENGINE_MODULE = "Dataset.semantic_truth.world_truth"
EXPECTED_ENGINE_ENTRYPOINT = "evaluate_world_truth"
EXPECTED_EVENT_ENGINE_MODULE = "Dataset.semantic_truth.minimal_semantics"
EXPECTED_EVENT_ENGINE_ENTRYPOINT = "build_events"
EXPECTED_EVENT_ADAPTER_MODULE = "Dataset.semantic_truth.minimal_semantics_adapter"
EXPECTED_EVENT_ADAPTER_ENTRYPOINT = "run_minimal_semantic_engine"
EXPECTED_DECLARED_PREDICATE_COUNT = 79
EXPECTED_EXECUTABLE_PREDICATE_COUNT = 72
EXPECTED_NON_EXECUTABLE_PREDICATE_COUNT = 7
_PREDICATE_IDENTIFIER = re.compile(r'aw:predicateIdentifier\s+"([^"]+)"')
_EVENT_TYPE_IDENTIFIER = re.compile(r'aw:eventTypeIdentifier\s+"([^"]+)"')
GROUNDING_AUTHORITY_KINDS = frozenset(
    {
        "scope_entity",
        "scope_relation",
        "domain_observation",
        "domain_nested_relation",
        "direct_predicate_row",
        "utm_record",
        "episode_operational_region",
    }
)
GROUNDING_ROLE_SOURCE_KINDS = frozenset(
    {"scope_entity", "context", "record", "record_binding", "manifest"}
)
GROUNDING_ROLE_KINDS_BY_AUTHORITY = {
    "scope_entity": frozenset({"scope_entity"}),
    "scope_relation": frozenset({"scope_entity", "context"}),
    "domain_observation": frozenset({"record"}),
    "domain_nested_relation": frozenset({"record"}),
    "direct_predicate_row": frozenset({"record_binding"}),
    "utm_record": frozenset({"record"}),
    "episode_operational_region": frozenset({"manifest"}),
}


class CoreSemanticRegistryError(ValueError):
    """Raised when the ontology vocabulary and executable registry diverge."""


def _domain_predicate_ids() -> tuple[str, ...]:
    identifiers: list[str] = []
    for path in sorted(DOMAIN_ONTOLOGY_ROOT.glob("*.ttl")):
        if path.name.endswith(".shacl.ttl"):
            continue
        identifiers.extend(
            _PREDICATE_IDENTIFIER.findall(path.read_text(encoding="utf-8"))
        )
    if not identifiers:
        raise CoreSemanticRegistryError(
            f"no predicate identifiers found in {DOMAIN_ONTOLOGY_ROOT}"
        )
    duplicates = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count != 1
    )
    if duplicates:
        raise CoreSemanticRegistryError(
            f"duplicate predicate identifiers in domain ontologies: {duplicates}"
        )
    return tuple(identifiers)


def _event_ontology_ids() -> tuple[str, ...]:
    identifiers = [
        identifier
        for path in sorted(EVENT_ONTOLOGY_ROOT.glob("*_events.ttl"))
        for identifier in _EVENT_TYPE_IDENTIFIER.findall(
            path.read_text(encoding="utf-8")
        )
    ]
    if not identifiers:
        raise CoreSemanticRegistryError(
            f"no event identifiers found in {EVENT_ONTOLOGY_ROOT}"
        )
    duplicates = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count != 1
    )
    if duplicates:
        raise CoreSemanticRegistryError(
            f"duplicate event identifiers in event ontologies: {duplicates}"
        )
    return tuple(identifiers)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise CoreSemanticRegistryError(f"{label} must be an object")
    return value


def _named_records(
    value: Any,
    *,
    id_field: str,
    label: str,
) -> tuple[list[Mapping[str, Any]], tuple[str, ...]]:
    if not isinstance(value, list) or not all(
        isinstance(record, Mapping) for record in value
    ):
        raise CoreSemanticRegistryError(f"{label} must be an array of objects")
    records = list(value)
    identifiers = tuple(record.get(id_field) for record in records)
    if any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ):
        raise CoreSemanticRegistryError(
            f"every {label} record must have a non-empty {id_field}"
        )
    duplicates = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count != 1
    )
    if duplicates:
        raise CoreSemanticRegistryError(f"duplicate {label} identifiers: {duplicates}")
    return records, identifiers


def _validate_state_contract(
    template: Mapping[str, Any],
    governed_defaults: Mapping[str, Any],
) -> None:
    predicate_id = str(template["id"])
    if not isinstance(template.get("tuple_type"), str):
        raise CoreSemanticRegistryError(f"{predicate_id}: tuple_type is required")
    expression = template.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise CoreSemanticRegistryError(f"{predicate_id}: expression is required")
    scope_types = template.get("scope_types")
    if not isinstance(scope_types, list) or not scope_types:
        raise CoreSemanticRegistryError(
            f"{predicate_id}: scope_types must be a non-empty array"
        )
    if not set(scope_types) <= {
        "uav",
        "vehicle",
        "pedestrian",
        "facility",
        "compute_node",
        "scene",
    }:
        raise CoreSemanticRegistryError(
            f"{predicate_id}: scope_types contain an unknown world scope"
        )
    status = template.get("implementation_status")
    if status == "non_executable":
        forbidden = {
            "state_contract",
            "grounding_spec",
            "evaluation_spec",
            "simulation_log_projection",
        } & set(template)
        if forbidden:
            raise CoreSemanticRegistryError(
                f"{predicate_id}: non-executable template declares runtime fields "
                f"{sorted(forbidden)}"
            )
        _validate_non_executable_contract(template)
        return
    if status != "executable_l1":
        raise CoreSemanticRegistryError(
            f"{predicate_id}: implementation_status must be executable_l1 or "
            "non_executable"
        )
    if "non_executable_contract" in template:
        raise CoreSemanticRegistryError(
            f"{predicate_id}: executable template has a non-executable contract"
        )
    contract = template.get("state_contract")
    if not isinstance(contract, Mapping):
        raise CoreSemanticRegistryError(
            f"{predicate_id}: state_contract must be an object"
        )
    required_fields = contract.get("required_fields")
    unknown_fields = contract.get("unknown_if_missing")
    thresholds = contract.get("threshold_parameters")
    if not isinstance(required_fields, list) or not isinstance(unknown_fields, list):
        raise CoreSemanticRegistryError(
            f"{predicate_id}: required_fields and unknown_if_missing must be arrays"
        )
    if not isinstance(thresholds, list):
        raise CoreSemanticRegistryError(
            f"{predicate_id}: threshold_parameters must be an array"
        )
    if not required_fields or not unknown_fields:
        raise CoreSemanticRegistryError(
            f"{predicate_id}: implemented predicate requires fields and unknown rules"
        )
    field_names: set[str] = set()
    for field in required_fields:
        if not isinstance(field, Mapping) or any(
            not isinstance(field.get(key), str) or not field.get(key)
            for key in ("field", "source", "unit")
        ):
            raise CoreSemanticRegistryError(
                f"{predicate_id}: each required field needs field/source/unit"
            )
        field_names.add(str(field["field"]))
    missing_unknown_fields = sorted(set(unknown_fields) - field_names)
    if missing_unknown_fields:
        raise CoreSemanticRegistryError(
            f"{predicate_id}: unknown fields are not declared: {missing_unknown_fields}"
        )
    unknown_thresholds = sorted(set(thresholds) - set(governed_defaults))
    if unknown_thresholds:
        raise CoreSemanticRegistryError(
            f"{predicate_id}: unknown threshold parameters: {unknown_thresholds}"
        )
    if contract.get("missing_record_field") != "missing_source_record":
        raise CoreSemanticRegistryError(
            f"{predicate_id}: missing_record_field must be missing_source_record"
        )
    evaluation = template.get("evaluation_spec")
    if not isinstance(evaluation, Mapping) or not isinstance(evaluation.get("op"), str):
        raise CoreSemanticRegistryError(
            f"{predicate_id}: evaluation_spec must declare an operator"
        )
    _validate_grounding_spec(template)


def _validate_non_executable_contract(template: Mapping[str, Any]) -> None:
    predicate_id = str(template["id"])
    contract = template.get("non_executable_contract")
    if not isinstance(contract, Mapping):
        raise CoreSemanticRegistryError(
            f"{predicate_id}: non_executable_contract is required"
        )
    required = {"reason_code", "missing_authorities", "closure_statement"}
    if set(contract) != required:
        raise CoreSemanticRegistryError(
            f"{predicate_id}: invalid non-executable contract fields"
        )
    reason_code = contract.get("reason_code")
    missing_authorities = contract.get("missing_authorities")
    closure = contract.get("closure_statement")
    if not isinstance(reason_code, str) or not re.fullmatch(
        r"[a-z][a-z0-9_]*", reason_code
    ):
        raise CoreSemanticRegistryError(
            f"{predicate_id}: invalid non-executable reason_code"
        )
    if (
        not isinstance(missing_authorities, list)
        or not missing_authorities
        or not all(isinstance(item, Mapping) for item in missing_authorities)
    ):
        raise CoreSemanticRegistryError(
            f"{predicate_id}: missing_authorities must be a non-empty object array"
        )
    declared_classes = {
        str(role.get("class") or "")
        for role in template.get("argument_roles", ())
        if isinstance(role, Mapping)
    }
    authority_keys: set[tuple[str, str]] = set()
    for authority in missing_authorities:
        kind = authority.get("kind")
        if kind in {"ontology_role_individual", "supporting_individual"}:
            if set(authority) != {"kind", "ontology_class"}:
                raise CoreSemanticRegistryError(
                    f"{predicate_id}: invalid {kind} authority fields"
                )
            value = authority.get("ontology_class")
            if not isinstance(value, str) or not value:
                raise CoreSemanticRegistryError(
                    f"{predicate_id}: invalid missing authority ontology_class"
                )
            if kind == "ontology_role_individual" and value not in declared_classes:
                raise CoreSemanticRegistryError(
                    f"{predicate_id}: missing role class is not an ontology argument"
                )
        elif kind == "runtime_state_field":
            if set(authority) != {"kind", "field"}:
                raise CoreSemanticRegistryError(
                    f"{predicate_id}: invalid runtime_state_field authority fields"
                )
            value = authority.get("field")
            if not isinstance(value, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]*_state\.[a-z][a-z0-9_]*", value
            ):
                raise CoreSemanticRegistryError(
                    f"{predicate_id}: invalid missing runtime-state authority field"
                )
        else:
            raise CoreSemanticRegistryError(
                f"{predicate_id}: unsupported missing authority kind {kind!r}"
            )
        authority_key = (str(kind), str(value))
        if authority_key in authority_keys:
            raise CoreSemanticRegistryError(
                f"{predicate_id}: duplicate missing authority"
            )
        authority_keys.add(authority_key)
    if not isinstance(closure, str) or not closure.strip():
        raise CoreSemanticRegistryError(
            f"{predicate_id}: closure_statement is required"
        )


def _world_class(value: str) -> str:
    return value if ":" in value else f"world:{value}"


def _ontology_class_path(identifier_path: str) -> str:
    prefix, separator, leaf = identifier_path.rpartition(".")
    if not leaf.endswith("_id"):
        raise CoreSemanticRegistryError(
            f"grounding identifier path must end in _id: {identifier_path}"
        )
    class_leaf = f"{leaf[:-3]}_ontology_class_id"
    return f"{prefix}.{class_leaf}" if separator else class_leaf


def _validate_grounding_spec(template: Mapping[str, Any]) -> None:
    predicate_id = str(template["id"])
    spec = template.get("grounding_spec")
    if not isinstance(spec, Mapping):
        raise CoreSemanticRegistryError(f"{predicate_id}: grounding_spec is required")
    authority = spec.get("candidate_authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("kind") not in GROUNDING_AUTHORITY_KINDS
        or not isinstance(authority.get("source"), str)
        or not authority["source"]
    ):
        raise CoreSemanticRegistryError(
            f"{predicate_id}: invalid grounding candidate authority"
        )
    roles = template.get("argument_roles")
    if not isinstance(roles, list) or not roles:
        raise CoreSemanticRegistryError(f"{predicate_id}: argument_roles are required")
    expected_roles = [str(role["key"]) for role in roles]
    role_sources = spec.get("role_sources")
    if not isinstance(role_sources, Mapping) or set(role_sources) != set(
        expected_roles
    ):
        raise CoreSemanticRegistryError(
            f"{predicate_id}: grounding roles differ from ontology roles"
        )
    role_classes = {str(role["key"]): str(role["class"]) for role in roles}
    authority_kind = str(authority["kind"])
    allowed_role_kinds = GROUNDING_ROLE_KINDS_BY_AUTHORITY[authority_kind]
    for role, source in role_sources.items():
        if (
            not isinstance(source, Mapping)
            or source.get("kind") not in GROUNDING_ROLE_SOURCE_KINDS
            or (
                source.get("kind") != "scope_entity"
                and (not isinstance(source.get("path"), str) or not source["path"])
            )
        ):
            raise CoreSemanticRegistryError(
                f"{predicate_id}:{role}: invalid grounding role source"
            )
        source_kind = str(source["kind"])
        if source_kind not in allowed_role_kinds:
            raise CoreSemanticRegistryError(
                f"{predicate_id}:{role}: role source is incompatible with "
                f"candidate authority {authority_kind}"
            )
        path = source.get("path")
        expected_class = _world_class(role_classes[str(role)])
        if source_kind == "scope_entity":
            identifier_source = "typed_scope_entity.entity_id"
            class_sources = ["typed_scope_entity.ontology_class_id"]
        elif source_kind == "record_binding":
            assert isinstance(path, str)
            identifier_source = f"bindings.{path}"
            class_sources = [f"binding_ontology_classes.{path}"]
        elif source_kind == "manifest":
            assert isinstance(path, str)
            if expected_class != "world:OperationalRegion":
                raise CoreSemanticRegistryError(
                    f"{predicate_id}:{role}: manifest role must be OperationalRegion"
                )
            identifier_source = path
            class_sources = ["constant:world:OperationalRegion"]
        else:
            assert isinstance(path, str)
            identifier_source = path
            class_sources = [
                _ontology_class_path(path),
                "typed_instance_catalog[identifier]",
            ]
        expected_provenance = {
            "identifier_source": identifier_source,
            "ontology_class_sources": class_sources,
            "required_ontology_class_id": expected_class,
            "validation": "runtime_exact_is_a_fail_closed",
        }
        if source.get("identity_provenance") != expected_provenance:
            raise CoreSemanticRegistryError(
                f"{predicate_id}:{role}: identity provenance differs from runtime "
                "resolution contract"
            )
    if (
        authority_kind in {"scope_entity", "scope_relation"}
        and sum(source["kind"] == "scope_entity" for source in role_sources.values())
        != 1
    ):
        raise CoreSemanticRegistryError(
            f"{predicate_id}: scope authority requires exactly one scope role"
        )
    required_fields = template["state_contract"]["required_fields"]
    expected_fields = {str(field["field"]) for field in required_fields}
    field_sources = spec.get("field_sources")
    if not isinstance(field_sources, Mapping) or set(field_sources) != expected_fields:
        raise CoreSemanticRegistryError(
            f"{predicate_id}: grounding field sources are incomplete"
        )
    if spec.get("role_class_validation") != "exact_ontology_class_per_binding":
        raise CoreSemanticRegistryError(
            f"{predicate_id}: binding classes must be validated exactly"
        )
    for label in ("distinct_role_sets", "unordered_role_sets"):
        constraints = spec.get(label)
        if not isinstance(constraints, list):
            raise CoreSemanticRegistryError(f"{predicate_id}: {label} must be an array")
        for pair in constraints:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or pair[0] == pair[1]
                or any(role not in role_classes for role in pair)
            ):
                raise CoreSemanticRegistryError(
                    f"{predicate_id}: invalid role constraint {pair!r}"
                )
            if (
                label == "unordered_role_sets"
                and role_classes[pair[0]] != role_classes[pair[1]]
            ):
                raise CoreSemanticRegistryError(
                    f"{predicate_id}: unordered role classes differ"
                )


def validate_core_semantic_registry(registry: Mapping[str, Any]) -> None:
    """Validate ontology equality, state contracts, and engine metadata."""

    if registry.get("schema_name") != "aeroworld_core_semantic_predicate_registry":
        raise CoreSemanticRegistryError("unexpected registry schema_name")
    if registry.get("schema_version") != "3.0.0":
        raise CoreSemanticRegistryError("the executable predicate registry must be V3")
    if registry.get("authority") != "Dataset/knowledge_graph/domain/*.ttl":
        raise CoreSemanticRegistryError(
            "domain TTL files must be the vocabulary authority"
        )
    policy = registry.get("runtime_policy")
    if not isinstance(policy, Mapping):
        raise CoreSemanticRegistryError("runtime_policy must be an object")
    if policy.get("official_predicate_vocabulary") != "domain_ontology":
        raise CoreSemanticRegistryError(
            "domain ontology must be the official vocabulary"
        )
    if policy.get("authored_event_outcomes_as_truth_inputs") != "forbidden":
        raise CoreSemanticRegistryError("authored event outcomes must remain forbidden")
    if policy.get("l1_materialization") != (
        "tick0_grounded_candidate_base_plus_add_set_remove_delta"
    ):
        raise CoreSemanticRegistryError(
            "L1 must use a grounded tick-0 base plus add/set/remove deltas"
        )
    if policy.get("l2_materialization") != "roi_or_event_entities_only":
        raise CoreSemanticRegistryError(
            "L2 must be restricted to ROI or event entities"
        )
    if policy.get("non_executable_predicate_policy") != (
        "declared_vocabulary_only; forbidden_from_l1_candidates_and_deltas"
    ):
        raise CoreSemanticRegistryError(
            "non-executable predicates must be forbidden from L1 materialization"
        )
    if registry.get("world_scope_types") != [
        "uav",
        "vehicle",
        "pedestrian",
        "facility",
        "compute_node",
        "scene",
    ]:
        raise CoreSemanticRegistryError("V3 must declare the six ordered world scopes")

    templates, template_ids = _named_records(
        registry.get("templates"),
        id_field="id",
        label="predicate templates",
    )
    ontology_ids = _domain_predicate_ids()
    if set(template_ids) != set(ontology_ids):
        raise CoreSemanticRegistryError(
            "domain TTL predicate ids and registry template ids differ: "
            f"ontology_only={sorted(set(ontology_ids) - set(template_ids))}, "
            f"registry_only={sorted(set(template_ids) - set(ontology_ids))}"
        )
    if registry.get("template_count") != len(templates):
        raise CoreSemanticRegistryError("template_count does not match templates")
    role_keys_by_predicate = {
        str(template["id"]): [
            str(role["key"])
            for role in template.get("argument_roles", ())
            if isinstance(role, Mapping)
        ]
        for template in templates
    }
    role_classes_by_predicate = {
        str(template["id"]): {
            str(role["key"]): str(role["class"])
            for role in template.get("argument_roles", ())
            if isinstance(role, Mapping)
            and isinstance(role.get("key"), str)
            and isinstance(role.get("class"), str)
        }
        for template in templates
    }

    governance = registry.get("parameter_governance")
    if not isinstance(governance, Mapping):
        raise CoreSemanticRegistryError("parameter_governance must be an object")
    if governance.get("status") != "reviewed_for_controlled_simulation":
        raise CoreSemanticRegistryError("unexpected parameter governance status")
    defaults = governance.get("governed_defaults")
    if not isinstance(defaults, Mapping):
        raise CoreSemanticRegistryError("governed_defaults must be an object")
    bounds = governance.get("theoretical_performance_bounds")
    if not isinstance(bounds, Mapping) or set(bounds) != set(defaults):
        raise CoreSemanticRegistryError(
            "every governed parameter requires a theoretical performance bound"
        )
    if defaults.get("boundary_margin_m") != 14.0:
        raise CoreSemanticRegistryError(
            "the governed L1-1 boundary-margin default must be 14.0 m"
        )
    override_policy = governance.get("scenario_override_policy")
    boundary_override = (
        override_policy.get("boundary_margin_m")
        if isinstance(override_policy, Mapping)
        else None
    )
    if (
        not isinstance(boundary_override, Mapping)
        or boundary_override.get("source")
        != "parameter_governance.governed_defaults.boundary_margin_m"
    ):
        raise CoreSemanticRegistryError(
            "boundary_margin_m requires the single governed-default authority"
        )
    response_override = (
        override_policy.get("boundary_response_distance_m")
        if isinstance(override_policy, Mapping)
        else None
    )
    if (
        not isinstance(response_override, Mapping)
        or response_override.get("source")
        != "parameter_governance.governed_defaults.boundary_response_distance_m"
    ):
        raise CoreSemanticRegistryError(
            "boundary_response_distance_m requires the single governed-default authority"
        )
    response_distance_m = defaults.get("boundary_response_distance_m")
    if (
        not isinstance(response_distance_m, (int, float))
        or isinstance(response_distance_m, bool)
        or not 0.0 < float(response_distance_m) <= float(defaults["boundary_margin_m"])
    ):
        raise CoreSemanticRegistryError(
            "governed boundary_response_distance_m must lie in (0, boundary_margin_m]"
        )
    for template in templates:
        _validate_state_contract(template, defaults)

    implementation = registry.get("implementation")
    if not isinstance(implementation, Mapping):
        raise CoreSemanticRegistryError("implementation must be an object")
    expected = {
        "engine_module": EXPECTED_ENGINE_MODULE,
        "engine_entrypoint": EXPECTED_ENGINE_ENTRYPOINT,
        "event_engine_module": EXPECTED_EVENT_ENGINE_MODULE,
        "event_engine_entrypoint": EXPECTED_EVENT_ENGINE_ENTRYPOINT,
        "event_adapter_module": EXPECTED_EVENT_ADAPTER_MODULE,
        "event_adapter_entrypoint": EXPECTED_EVENT_ADAPTER_ENTRYPOINT,
    }
    for field, value in expected.items():
        if implementation.get(field) != value:
            raise CoreSemanticRegistryError(f"implementation.{field} must be {value!r}")
    declared_ids = implementation.get("declared_predicate_vocabulary_ids")
    executable_ids = implementation.get("executable_predicate_ids")
    non_executable_ids = implementation.get("non_executable_predicate_ids")
    non_executable_contracts = implementation.get("non_executable_predicates")
    for field, values in (
        ("declared_predicate_vocabulary_ids", declared_ids),
        ("executable_predicate_ids", executable_ids),
        ("non_executable_predicate_ids", non_executable_ids),
    ):
        if (
            not isinstance(values, list)
            or not all(isinstance(identifier, str) for identifier in values)
            or len(values) != len(set(values))
        ):
            raise CoreSemanticRegistryError(f"{field} must contain unique IDs")
    if not isinstance(non_executable_contracts, Mapping):
        raise CoreSemanticRegistryError("non_executable_predicates must be an object")
    executable_status_ids = {
        str(template["id"])
        for template in templates
        if template.get("implementation_status") == "executable_l1"
    }
    non_executable_status_ids = {
        str(template["id"])
        for template in templates
        if template.get("implementation_status") == "non_executable"
    }
    if list(declared_ids) != list(template_ids):
        raise CoreSemanticRegistryError(
            "declared_predicate_vocabulary_ids differ from templates"
        )
    if set(executable_ids) != executable_status_ids:
        raise CoreSemanticRegistryError(
            "executable_predicate_ids differ from executable_l1 templates"
        )
    if (
        set(non_executable_ids) != non_executable_status_ids
        or set(non_executable_contracts) != non_executable_status_ids
    ):
        raise CoreSemanticRegistryError(
            "non-executable implementation metadata differs from templates"
        )
    template_by_id = {str(template["id"]): template for template in templates}
    for predicate_id in non_executable_status_ids:
        if non_executable_contracts[predicate_id] != template_by_id[predicate_id].get(
            "non_executable_contract"
        ):
            raise CoreSemanticRegistryError(
                f"{predicate_id}: non-executable metadata contract differs"
            )
    if executable_status_ids | non_executable_status_ids != set(template_ids):
        raise CoreSemanticRegistryError(
            "predicate execution statuses do not partition the vocabulary"
        )
    if (
        len(template_ids) != EXPECTED_DECLARED_PREDICATE_COUNT
        or len(executable_status_ids) != EXPECTED_EXECUTABLE_PREDICATE_COUNT
        or len(non_executable_status_ids) != EXPECTED_NON_EXECUTABLE_PREDICATE_COUNT
    ):
        raise CoreSemanticRegistryError(
            "predicate execution partition must contain exactly 79 declared, "
            "72 executable, and 7 non-executable predicates"
        )
    if implementation.get("declared_predicate_vocabulary_count") != len(templates):
        raise CoreSemanticRegistryError(
            "declared_predicate_vocabulary_count does not match ontology"
        )
    if implementation.get("executable_predicate_count") != len(executable_status_ids):
        raise CoreSemanticRegistryError("executable_predicate_count is stale")
    if implementation.get("non_executable_predicate_count") != len(
        non_executable_status_ids
    ):
        raise CoreSemanticRegistryError("non_executable_predicate_count is stale")
    event_rules, rule_ids = _named_records(
        registry.get("event_occurrence_types"),
        id_field="rule_id",
        label="event occurrence rules",
    )
    if registry.get("event_occurrence_type_count") != len(event_rules):
        raise CoreSemanticRegistryError(
            "event_occurrence_type_count does not match event occurrence rules"
        )
    event_ontology_ids = _event_ontology_ids()
    mapping = _read_object(EVENT_FAMILY_MAPPING_PATH, "event-family mapping")
    objective_contract = _read_object(OBJECTIVE_CONTRACT_PATH, "objective contract")
    mapped_families = mapping.get("families")
    contract_families = objective_contract.get("event_families")
    if not isinstance(mapped_families, Mapping) or not isinstance(
        contract_families, Mapping
    ):
        raise CoreSemanticRegistryError(
            "event family authority artifacts lack families"
        )
    if set(mapped_families) != set(contract_families):
        raise CoreSemanticRegistryError(
            "mapping and objective-contract event family sets differ"
        )
    executable_families = {
        str(family_id)
        for family_id, family in mapped_families.items()
        if isinstance(family, Mapping) and family.get("executable") is True
    }
    registered_families: set[str] = set()
    registered_event_types: set[str] = set()
    for rule in event_rules:
        required = {
            "rule_id",
            "event_family_id",
            "event_type",
            "mapping_class",
            "trigger_predicate_id",
            "trigger_direction",
            "event_tick",
            "support_conditions",
            "terminal",
            "event_role_sources",
            "participant_roles",
        }
        missing = sorted(required - set(rule))
        if missing:
            raise CoreSemanticRegistryError(
                f"{rule.get('rule_id')}: event rule lacks {missing}"
            )
        family_id = str(rule["event_family_id"])
        event_type = str(rule["event_type"])
        trigger_id = str(rule["trigger_predicate_id"])
        if trigger_id not in executable_status_ids:
            raise CoreSemanticRegistryError(
                f"{rule['rule_id']}: trigger predicate is not executable"
            )
        if event_type not in event_ontology_ids:
            raise CoreSemanticRegistryError(
                f"{rule['rule_id']}: event type is not ontology-authoritative"
            )
        if rule["trigger_direction"] not in {"rising", "falling"}:
            raise CoreSemanticRegistryError(
                f"{rule['rule_id']}: invalid trigger direction"
            )
        if rule["event_tick"] not in {"before", "after"}:
            raise CoreSemanticRegistryError(f"{rule['rule_id']}: invalid event tick")
        terminal = rule["terminal"]
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("predicate_id") not in executable_status_ids
        ):
            raise CoreSemanticRegistryError(
                f"{rule['rule_id']}: terminal predicate is not executable"
            )
        trigger_roles = set(role_keys_by_predicate[trigger_id])
        terminal_join_roles = terminal.get("join_roles")
        expected_terminal_join_roles = sorted(
            trigger_roles
            & set(role_keys_by_predicate[str(terminal.get("predicate_id"))])
        )
        if not expected_terminal_join_roles:
            raise CoreSemanticRegistryError(
                f"{rule['rule_id']}: terminal predicate has no exact trigger-role join"
            )
        if terminal_join_roles != expected_terminal_join_roles:
            raise CoreSemanticRegistryError(
                f"{rule['rule_id']}: terminal join_roles must be "
                f"{expected_terminal_join_roles}"
            )
        for support in rule["support_conditions"]:
            required_support_keys = {"predicate_id", "value", "at", "join_roles"}
            if (
                not isinstance(support, Mapping)
                or not required_support_keys <= set(support)
                or not set(support) - required_support_keys <= {"hold_samples"}
            ):
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}: support condition must contain exactly "
                    "predicate_id, value, at, join_roles, and optional hold_samples"
                )
            if support["predicate_id"] not in executable_status_ids:
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}: support predicate is not executable"
                )
            expected_support_join_roles = sorted(
                trigger_roles
                & set(role_keys_by_predicate[str(support["predicate_id"])])
            )
            if support.get("join_roles") != expected_support_join_roles:
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}: support condition join_roles must be "
                    f"{expected_support_join_roles}"
                )
            if support["value"] not in {"true", "false"}:
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}: support condition has invalid value"
                )
            if support["at"] not in {"before", "after"}:
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}: support condition has invalid temporal anchor"
                )
            hold_samples = support.get("hold_samples", 1)
            if (
                not isinstance(hold_samples, int)
                or isinstance(hold_samples, bool)
                or hold_samples < 1
                or (hold_samples > 1 and support["at"] != "after")
            ):
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}: support condition has invalid hold_samples"
                )
        roles = rule["participant_roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or any(
                not isinstance(role, Mapping)
                or role.get("background_allowed") is not False
                for role in roles
            )
        ):
            raise CoreSemanticRegistryError(
                f"{rule['rule_id']}: participant roles must forbid background binding"
            )
        role_by_name: dict[str, Mapping[str, Any]] = {}
        for role in roles:
            role_name = role.get("role")
            if (
                not isinstance(role_name, str)
                or not role_name
                or role_name in role_by_name
            ):
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}: participant role names must be unique strings"
                )
            if set(role) != {
                "role",
                "ontology_class",
                "source_kind",
                "background_allowed",
            }:
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}: participant role has unexpected fields"
                )
            role_by_name[role_name] = role
        event_role_sources = rule["event_role_sources"]
        if not isinstance(event_role_sources, Mapping) or set(
            event_role_sources
        ) != set(role_by_name):
            raise CoreSemanticRegistryError(
                f"{rule['rule_id']}: event_role_sources must exactly cover participant roles"
            )
        support_predicate_ids = {
            str(condition["predicate_id"]) for condition in rule["support_conditions"]
        }
        for event_role, raw_source in event_role_sources.items():
            if not isinstance(raw_source, Mapping) or set(raw_source) != {
                "event_role",
                "event_ontology_class",
                "kind",
                "predicate_id",
                "predicate_role",
                "source_ontology_class",
                "background_allowed",
            }:
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}:{event_role}: invalid exact event role source"
                )
            kind = raw_source.get("kind")
            predicate_id = raw_source.get("predicate_id")
            predicate_role = raw_source.get("predicate_role")
            if kind not in {"trigger_binding", "support_binding"}:
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}:{event_role}: unsupported event role source kind"
                )
            if (
                kind == "trigger_binding"
                and predicate_id != rule["trigger_predicate_id"]
            ) or (
                kind == "support_binding" and predicate_id not in support_predicate_ids
            ):
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}:{event_role}: event role source is outside the rule"
                )
            predicate_roles = role_classes_by_predicate.get(str(predicate_id), {})
            if predicate_roles.get(str(predicate_role)) != raw_source.get(
                "source_ontology_class"
            ):
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}:{event_role}: source class differs from predicate role"
                )
            participant_role = role_by_name[str(event_role)]
            if (
                raw_source.get("event_role") != event_role
                or raw_source.get("event_ontology_class")
                != participant_role.get("ontology_class")
                or raw_source.get("kind") != participant_role.get("source_kind")
                or raw_source.get("background_allowed") is not False
            ):
                raise CoreSemanticRegistryError(
                    f"{rule['rule_id']}:{event_role}: participant/source declarations differ"
                )
        registered_families.add(family_id)
        registered_event_types.add(event_type)
    if registered_families != executable_families:
        raise CoreSemanticRegistryError(
            "registry event-rule families differ from executable mapping families"
        )
    if implementation.get("implemented_event_types") != sorted(registered_event_types):
        raise CoreSemanticRegistryError(
            "implemented_event_types must match the authoritative executable subset"
        )
    counts = registry.get("event_counts")
    expected_counts = {
        "ontology_event_type_count": len(event_ontology_ids),
        "registered_event_rule_count": len(event_rules),
        "executable_event_type_count": len(registered_event_types),
        "event_family_count": len(mapped_families),
        "executable_event_family_count": len(executable_families),
    }
    if counts != expected_counts:
        raise CoreSemanticRegistryError(
            f"event count seam mismatch: expected {expected_counts}, found {counts}"
        )
    if registry.get("event_ontology_type_ids") != list(event_ontology_ids):
        raise CoreSemanticRegistryError(
            "registry event ontology inventory differs from the 60-type authority"
        )
    expected_family_count = objective_contract.get("validation", {}).get(
        "expected_event_family_count"
    )
    if expected_family_count != len(mapped_families):
        raise CoreSemanticRegistryError(
            "objective-contract expected event-family count is stale"
        )

def load_core_semantic_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the ontology-derived core registry."""

    registry_path = Path(path) if path is not None else CORE_REGISTRY_PATH
    value = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CoreSemanticRegistryError("registry root must be an object")
    validate_core_semantic_registry(value)
    return value


def get_core_predicate_templates() -> list[dict[str, Any]]:
    return deepcopy(load_core_semantic_registry()["templates"])


def get_core_predicate_ids() -> tuple[str, ...]:
    return tuple(template["id"] for template in get_core_predicate_templates())


def get_executable_predicate_ids() -> tuple[str, ...]:
    implementation = load_core_semantic_registry()["implementation"]
    return tuple(implementation["executable_predicate_ids"])


def get_core_event_occurrence_types() -> list[dict[str, Any]]:
    return deepcopy(load_core_semantic_registry()["event_occurrence_types"])


def get_governed_parameter_defaults() -> dict[str, Any]:
    registry = load_core_semantic_registry()
    return deepcopy(registry["parameter_governance"]["governed_defaults"])


def get_implementation_metadata() -> dict[str, Any]:
    return deepcopy(load_core_semantic_registry()["implementation"])


__all__ = [
    "CORE_REGISTRY_PATH",
    "CoreSemanticRegistryError",
    "get_core_event_occurrence_types",
    "get_core_predicate_ids",
    "get_core_predicate_templates",
    "get_executable_predicate_ids",
    "get_governed_parameter_defaults",
    "get_implementation_metadata",
    "load_core_semantic_registry",
    "validate_core_semantic_registry",
]
