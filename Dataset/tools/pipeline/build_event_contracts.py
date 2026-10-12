#!/usr/bin/env python3
"""Build the current authoritative L2 family and stage contracts.

The authored source records stable EPI and family prose.  The event-family and
stage authority mappings record executable evidence decisions.  This builder
projects those sources into generated contracts without reading generated
outputs back as input.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = PROJECT_ROOT / "Dataset" / "semantic_rules" / "profiles"
SOURCE_PATH = PROFILE_ROOT / "epi_event_contract_authoring_source.json"
MAPPING_PATH = PROFILE_ROOT / "event_family_authority_mapping.json"
STAGE_MAPPING_PATH = PROFILE_ROOT / "epi_stage_authority_mapping.json"
OBJECTIVE_PATH = PROFILE_ROOT / "epi_objective_semantic_contract.json"
STAGE_PATH = PROFILE_ROOT / "epi_stage_acceptance_contract.json"
EVENT_CATALOG_PATH = (
    PROJECT_ROOT / "Dataset" / "knowledge_graph" / "events" / "catalog.yaml"
)
SOURCE_CLASSES = [
    "engine_truth",
    "deterministic_derived",
    "simulated_derived",
    "visual_derived",
]
EVENT_OCCURRENCE_PHASES = {
    "onset",
    "escalation_support",
    "terminal",
}
FORBIDDEN_EVIDENTIARY_FIELDS = [
    "event_trace.jsonl",
    "event_trace",
    "event_realization.jsonl",
    "event_realization",
    "active_event_ids",
    "dynamic_labels",
    "authored_event_label",
    "expected_event",
    "expected_event_id",
    "semantic_role",
    "confirmedAtTick",
    "scenario_plan",
    "scenario_plan_as_truth",
    "scenario_title",
    "epi_id",
    "level",
    "state_facets",
    "task_id",
    "variant",
]
OCCURRENCE_EVIDENCE = [
    "predicate_truth_before",
    "predicate_truth_after",
    "strict_adjacent_transition",
    "event_occurrence_record",
    "participant_identity",
    "source_refs",
    "tuple_binding_continuity",
]
TRANSITION_EVIDENCE = [
    "predicate_truth_before",
    "predicate_truth_after",
    "strict_adjacent_transition",
    "source_refs",
    "tuple_binding_continuity",
]
AUTHORED_FAMILY_KEYS = {
    "display_name",
    "required_state_fields",
    "required_response",
    "observable_dimensions",
    "unknown_policy",
    "participant_roles",
}
BUILDER_AUTHORED_FAMILY_METADATA = {
    "charging_queue_over_capacity": {
        "display_name": "charging_queue_over_capacity",
        "required_state_fields": [
            "facility_kind",
            "requester_count",
            "capacity",
        ],
        "required_response": (
            "the charging facility queue exceeds its capacity and the planner "
            "must wait or reroute"
        ),
        "observable_dimensions": [
            "facility_status",
            "queue_length",
            "uav_energy_state",
            "planner_assignment",
        ],
        "unknown_policy": (
            "unknown if facility requester counts or capacity are absent"
        ),
    }
}
BUILDER_OBJECTIVE_DECISION_FIELDS: dict[str, dict[str, str]] = {}
BUILDER_EVENT_OCCURRENCE_BINDINGS = [
    {
        "event_family_id": "charging_queue_over_capacity",
        "event_type_id": "facility.charging_queue_overload_event",
        "trigger_predicate_id": "facility.charging_queue_over_capacity",
        "direction": "rising",
        "event_phase": "onset",
        "publish_event_phase": True,
        "stage_keys": ["L2-3_v1::1", "L2-3_v2::1"],
    }
]
BUILDER_NOT_APPLICABLE_STAGES: dict[str, dict[str, str]] = {}
GENERATED_FAMILY_KEYS = {
    "required_predicates",
    "start_condition",
    "terminal_condition",
    "permitted_source_classes",
    "mapping_class",
    "executable",
    "ontology_event_type_ids",
    "semantic_rationale",
    "trigger_expression",
    "terminal_expression",
    "non_executable_reason",
    "closure_statement",
}
CLASS_SCOPE_TYPES = {
    "UnmannedAircraft": "uav",
    "GroundVehicle": "vehicle",
    "EmergencyVehicle": "vehicle",
    "Pedestrian": "pedestrian",
    "LandingPad": "facility",
    "LandingArea": "facility",
    "ChargingStation": "facility",
    "TrafficSignal": "facility",
    "CommunicationStation": "facility",
    "GroundControlStation": "facility",
    "BuildingStructure": "facility",
    "Barrier": "facility",
    "ComputeNode": "compute_node",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _render(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _event_catalog_roles(catalog: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    modules = catalog.get("modules")
    if not isinstance(modules, Mapping):
        raise ValueError("event catalog lacks modules")
    roles_by_event_type: dict[str, list[dict[str, str]]] = {}
    for module_id, module in modules.items():
        events = module.get("events") if isinstance(module, Mapping) else None
        if not isinstance(events, list):
            raise ValueError(f"event catalog module {module_id} lacks events")
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError(
                    f"event catalog module {module_id} has a non-object event"
                )
            identifier = event.get("identifier")
            roles = event.get("roles")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(
                    f"event catalog module {module_id} has an event without identifier"
                )
            if not isinstance(roles, list) or not roles:
                raise ValueError(f"event catalog event {identifier} lacks roles")
            parsed_roles: list[dict[str, str]] = []
            for role in roles:
                if not isinstance(role, Mapping):
                    raise ValueError(
                        f"event catalog event {identifier} has a non-object role"
                    )
                key = role.get("key")
                class_id = role.get("class")
                if not isinstance(key, str) or not isinstance(class_id, str):
                    raise ValueError(
                        f"event catalog event {identifier} has an invalid role"
                    )
                parsed_roles.append({"role": key, "class": class_id})
            roles_by_event_type[identifier] = parsed_roles
    return roles_by_event_type


def _scope_type(class_id: str) -> str:
    return CLASS_SCOPE_TYPES.get(class_id, "scene")


def _participant_role(
    role: str,
    *,
    class_id: str,
    binding: str,
    background_allowed: bool,
) -> dict[str, Any]:
    if background_allowed:
        raise ValueError(f"event role {role} allows background binding")
    return {
        "role": role,
        "scope_type": _scope_type(class_id),
        "binding": binding,
        "background_allowed": False,
    }


def _trigger_role_sources(decision: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trigger = decision.get("trigger")
    if not isinstance(trigger, Mapping):
        return []
    options = trigger.get("any_of")
    triggers = options if isinstance(options, list) else [trigger]
    sources: list[Mapping[str, Any]] = []
    for option in triggers:
        if not isinstance(option, Mapping):
            continue
        role_sources = option.get("event_role_sources")
        if isinstance(role_sources, Mapping):
            sources.extend(
                source
                for source in role_sources.values()
                if isinstance(source, Mapping)
            )
    return sources


def _participant_roles(
    family_id: str,
    source_family: Mapping[str, Any],
    decision: Mapping[str, Any],
    event_roles: Mapping[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    explicit_roles = source_family.get("participant_roles")
    if isinstance(explicit_roles, list) and explicit_roles:
        return deepcopy(explicit_roles)

    roles: dict[str, dict[str, Any]] = {}
    for role_source in _trigger_role_sources(decision):
        role = role_source.get("predicate_role")
        class_id = role_source.get("source_ontology_class")
        kind = role_source.get("kind")
        predicate_id = role_source.get("predicate_id")
        if not all(
            isinstance(item, str) and item
            for item in (role, class_id, kind, predicate_id)
        ):
            raise ValueError(f"{family_id} has an incomplete event_role_sources entry")
        binding = f"{kind}:{predicate_id}:{role}"
        roles[str(role)] = _participant_role(
            str(role),
            class_id=str(class_id),
            binding=binding,
            background_allowed=bool(role_source.get("background_allowed")),
        )
    if roles:
        return [roles[key] for key in sorted(roles)]

    ontology_event_type_ids = decision.get("ontology_event_type_ids")
    if isinstance(ontology_event_type_ids, list) and ontology_event_type_ids:
        for event_type_id in ontology_event_type_ids:
            if not isinstance(event_type_id, str) or event_type_id not in event_roles:
                raise ValueError(
                    f"{family_id} references unknown event type {event_type_id!r}"
                )
            for role in event_roles[event_type_id]:
                role_name = role["role"]
                roles.setdefault(
                    role_name,
                    _participant_role(
                        role_name,
                        class_id=role["class"],
                        binding=f"ontology_role:{event_type_id}:{role_name}",
                        background_allowed=False,
                    ),
                )
    if roles:
        return [roles[key] for key in sorted(roles)]
    raise ValueError(f"{family_id} lacks explicit participant roles")


def _trigger_options(family: Mapping[str, Any]) -> list[dict[str, str]]:
    trigger = family.get("trigger")
    if not isinstance(trigger, Mapping):
        return []
    raw_options = trigger.get("any_of")
    if isinstance(raw_options, list):
        return [dict(option) for option in raw_options if isinstance(option, Mapping)]
    if isinstance(trigger.get("predicate_id"), str):
        option = dict(trigger)
        option.setdefault("event_type_id", family["ontology_event_type_ids"][0])
        return [option]
    return []


def _terminal_options(family: Mapping[str, Any]) -> list[dict[str, str]]:
    terminal = family.get("terminal")
    if not isinstance(terminal, Mapping):
        return []
    raw_options = terminal.get("any_of")
    if isinstance(raw_options, list):
        return [dict(option) for option in raw_options if isinstance(option, Mapping)]
    return [dict(terminal)] if isinstance(terminal.get("predicate_id"), str) else []


def _validate_terminal_bindings(family_id: str, family: Mapping[str, Any]) -> None:
    if not family.get("executable"):
        return
    options = _terminal_options(family)
    if not options:
        raise ValueError(f"{family_id} executable family lacks a terminal predicate")
    for index, option in enumerate(options):
        join_roles = option.get("join_roles")
        if not isinstance(join_roles, list) or not join_roles:
            raise ValueError(f"{family_id} terminal[{index}] has unbound join_roles")
        if not all(isinstance(role, str) and role for role in join_roles):
            raise ValueError(f"{family_id} terminal[{index}] has invalid join_roles")


def _build_objective(
    source: Mapping[str, Any],
    mapping: Mapping[str, Any],
    event_roles: Mapping[str, list[dict[str, str]]],
) -> dict[str, Any]:
    if source.get("schema_name") != "epi_event_contract_authoring_source":
        raise ValueError("current source has an invalid schema_name")
    source_classes = mapping.get("source_classes")
    if source_classes != SOURCE_CLASSES:
        raise ValueError("event-family mapping source_classes drift")

    result: dict[str, Any] = {
        "schema_name": "epi_objective_semantic_contract",
        "schema_version": "5.0.0",
        "profile_id": "aeroworld_low_altitude_epi_objective_contract_v5",
        "source_policy": deepcopy(source["source_policy"]),
        "event_families": {},
        "epi_contracts": {},
        "validation": deepcopy(source["validation"]),
        "event_family_mapping": {
            "path": str(MAPPING_PATH.relative_to(PROJECT_ROOT)),
            "digest": _digest(mapping),
        },
    }
    result["source_policy"]["forbidden_evidentiary_fields"] = list(
        FORBIDDEN_EVIDENTIARY_FIELDS
    )
    result["source_policy"]["permitted_source_classes"] = list(SOURCE_CLASSES)

    source_families = source.get("event_family_metadata")
    mapped = mapping.get("families")
    if not isinstance(source_families, Mapping) or not isinstance(mapped, Mapping):
        raise ValueError("objective contract and mapping must contain event families")
    duplicate_builder_families = sorted(
        set(source_families) & set(BUILDER_AUTHORED_FAMILY_METADATA)
    )
    if duplicate_builder_families:
        raise ValueError(
            "builder-authored event families duplicate the authoring source: "
            f"{duplicate_builder_families}"
        )
    families = {
        **source_families,
        **BUILDER_AUTHORED_FAMILY_METADATA,
    }
    if set(families) != set(mapped):
        raise ValueError(
            "event-family mapping drift: "
            f"contract_only={sorted(set(families) - set(mapped))}, "
            f"mapping_only={sorted(set(mapped) - set(families))}"
        )
    rebuilt: dict[str, dict[str, Any]] = {}
    for family_id, source_family in families.items():
        decision = deepcopy(mapped[family_id])
        decision.update(BUILDER_OBJECTIVE_DECISION_FIELDS.get(family_id, {}))
        if not isinstance(source_family, Mapping):
            raise ValueError(f"family source {family_id} must be an object")
        unexpected = set(source_family) - AUTHORED_FAMILY_KEYS
        generated = set(source_family) & GENERATED_FAMILY_KEYS
        if unexpected or generated:
            raise ValueError(
                f"family source {family_id} contains non-authored fields: "
                f"{sorted(unexpected | generated)}"
            )
        _validate_terminal_bindings(str(family_id), decision)
        family = deepcopy(source_family)
        family["required_predicates"] = list(decision["authoritative_predicates"])
        family["permitted_source_classes"] = list(SOURCE_CLASSES)
        family["mapping_class"] = str(decision["mapping_class"])
        family["executable"] = bool(decision["executable"])
        family["ontology_event_type_ids"] = list(decision["ontology_event_type_ids"])
        family["participant_roles"] = _participant_roles(
            str(family_id), source_family, decision, event_roles
        )
        family["semantic_rationale"] = str(decision.get("semantic_rationale") or "")
        if decision["executable"]:
            family["trigger_expression"] = deepcopy(decision["trigger"])
            family["terminal_expression"] = deepcopy(decision["terminal"])
            sort_expression_keys = family_id not in BUILDER_AUTHORED_FAMILY_METADATA
            family["start_condition"] = json.dumps(
                decision["trigger"],
                ensure_ascii=False,
                sort_keys=sort_expression_keys,
            )
            family["terminal_condition"] = json.dumps(
                decision["terminal"],
                ensure_ascii=False,
                sort_keys=sort_expression_keys,
            )
            family.pop("non_executable_reason", None)
            family.pop("closure_statement", None)
        else:
            family["non_executable_reason"] = str(decision["non_executable_reason"])
            family["closure_statement"] = str(decision["closure_statement"])
            family["start_condition"] = "not_applicable: no authoritative trigger chain"
            family["terminal_condition"] = "not_applicable: event is not executed"
            family.pop("trigger_expression", None)
            family.pop("terminal_expression", None)
        rebuilt[str(family_id)] = family
    result["event_families"] = rebuilt

    epi_contracts = source.get("epi_contracts")
    if not isinstance(epi_contracts, Mapping):
        raise ValueError("current source must contain epi_contracts")
    generated_epis: dict[str, dict[str, Any]] = {}
    for epi_id, source_epi in epi_contracts.items():
        if not isinstance(source_epi, Mapping):
            raise ValueError(f"EPI source {epi_id} must be an object")
        epi = deepcopy(source_epi)
        for stage_index, stage in enumerate(epi.get("required_chain", ())):
            if not isinstance(stage, Mapping):
                raise ValueError(f"{epi_id} stage {stage_index} must be an object")
            family_id = str(stage.get("event_family_id") or "")
            if family_id not in mapped:
                raise ValueError(
                    f"{epi_id} stage {stage_index} uses unknown family {family_id}"
                )
            alternatives = stage.get("alternative_event_family_ids")
            if alternatives is not None:
                if not isinstance(alternatives, list):
                    raise ValueError(
                        f"{epi_id} stage {stage_index} alternatives must be an array"
                    )
                unknown = sorted(
                    str(item) for item in alternatives if str(item) not in mapped
                )
                if unknown:
                    raise ValueError(
                        f"{epi_id} stage {stage_index} has unknown alternatives {unknown}"
                    )
        epi["permitted_source_classes"] = list(SOURCE_CLASSES)
        generated_epis[str(epi_id)] = epi
    result["epi_contracts"] = generated_epis

    validation = result["validation"]
    validation["expected_epi_count"] = len(generated_epis)
    validation["expected_event_family_count"] = len(rebuilt)
    return result


def _source_stages_from_objective(
    objective: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    stages: dict[str, dict[str, Any]] = {}
    epi_contracts = objective.get("epi_contracts")
    if not isinstance(epi_contracts, Mapping):
        raise ValueError("objective contract must contain epi_contracts")
    for epi_id, epi in epi_contracts.items():
        chain = epi.get("required_chain") if isinstance(epi, Mapping) else None
        if not isinstance(chain, list):
            raise ValueError(f"{epi_id} lacks required_chain")
        for stage_index, source_stage in enumerate(chain):
            if not isinstance(source_stage, Mapping):
                raise ValueError(f"{epi_id} stage {stage_index} must be an object")
            key = f"{epi_id}::{stage_index}"
            stages[key] = {
                "key": key,
                "epi_id": str(epi_id),
                "stage_index": stage_index,
                "stage_goal": str(source_stage["stage_goal"]),
                "event_family_id": str(source_stage["event_family_id"]),
                "temporal_order": {
                    "mode": (
                        "chain_root"
                        if stage_index == 0
                        else "strictly_after_previous_stage"
                    ),
                    "previous_stage_key": (
                        None if stage_index == 0 else f"{epi_id}::{stage_index - 1}"
                    ),
                    "same_episode": True,
                    "strictly_later": stage_index != 0,
                },
            }
    return stages


def _stage_occurrence_clause(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "acceptance_kind": "event_occurrence",
        "event_family_id": str(binding["event_family_id"]),
        "event_type_id": str(binding["event_type_id"]),
        "occurrence_status": "occurred",
        "trigger_predicate_id": str(binding["trigger_predicate_id"]),
        "direction": str(binding["direction"]),
        "api_availability": "available",
    }


def _stage_transition_clause(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "acceptance_kind": "predicate_transition",
        "predicate_id": str(binding["predicate_id"]),
        "direction": str(binding["direction"]),
        "api_availability": "available",
    }


def _not_applicable_stage(
    stage: Mapping[str, Any],
    *,
    reason: str,
    closure_statement: str,
) -> dict[str, Any]:
    return {
        "key": str(stage["key"]),
        "epi_id": str(stage["epi_id"]),
        "stage_index": int(stage["stage_index"]),
        "stage_goal": str(stage["stage_goal"]),
        "event_family_id": str(stage["event_family_id"]),
        "implementation_status": "not_applicable",
        "clause_policy": "not_applicable",
        "acceptance_clauses": [],
        "required_evidence_types": ["declared_not_applicable"],
        "not_applicable_reason": reason,
        "closure_statement": closure_statement,
        "temporal_order": deepcopy(stage["temporal_order"]),
    }


def _build_stage_contract(
    source: Mapping[str, Any],
    objective: Mapping[str, Any],
    mapping: Mapping[str, Any],
    stage_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    stage_source = source.get("stage_acceptance")
    if not isinstance(stage_source, Mapping):
        raise ValueError("current source must contain stage_acceptance")
    result: dict[str, Any] = {
        "schema_name": "epi_stage_acceptance_contract",
        "schema_version": "5.0.0",
        "profile_id": "aeroworld_low_altitude_epi_stage_acceptance_v5",
        "source_contract": str(OBJECTIVE_PATH.relative_to(PROJECT_ROOT)),
        "stage_key_format": str(stage_source["stage_key_format"]),
        "evidence_type_catalog": deepcopy(stage_source["evidence_type_catalog"]),
        "stages": {},
        "validation": deepcopy(stage_source["validation"]),
    }
    result["stage_authority_mapping"] = {
        "path": str(STAGE_MAPPING_PATH.relative_to(PROJECT_ROOT)),
        "digest": _digest(stage_mapping),
    }
    mapped = mapping["families"]
    if stage_mapping.get("schema_name") != "epi_stage_authority_mapping":
        raise ValueError("stage authority mapping has an invalid schema_name")
    mapped_not_applicable_stages = stage_mapping.get("not_applicable_stages")
    if not isinstance(mapped_not_applicable_stages, Mapping):
        raise ValueError("stage authority mapping lacks not_applicable_stages")
    builder_executable_stage_keys = {
        str(stage_key)
        for binding in BUILDER_EVENT_OCCURRENCE_BINDINGS
        for stage_key in binding["stage_keys"]
    }
    missing_replaced_stage_keys = sorted(
        builder_executable_stage_keys - set(mapped_not_applicable_stages)
    )
    if missing_replaced_stage_keys:
        raise ValueError(
            "builder event-occurrence bindings no longer replace mapped "
            f"not-applicable stages: {missing_replaced_stage_keys}"
        )
    duplicate_builder_not_applicable_stages = sorted(
        set(mapped_not_applicable_stages) & set(BUILDER_NOT_APPLICABLE_STAGES)
    )
    if duplicate_builder_not_applicable_stages:
        raise ValueError(
            "builder not-applicable stages duplicate the stage authority mapping: "
            f"{duplicate_builder_not_applicable_stages}"
        )
    not_applicable_stages = {
        str(key): value
        for key, value in mapped_not_applicable_stages.items()
        if str(key) not in builder_executable_stage_keys
    }
    not_applicable_stages.update(BUILDER_NOT_APPLICABLE_STAGES)
    explicit: dict[str, dict[str, Any]] = {}
    for kind, collection_name in (
        ("event_occurrence", "event_occurrence_bindings"),
        ("predicate_transition", "predicate_transition_bindings"),
    ):
        bindings = stage_mapping.get(collection_name)
        if not isinstance(bindings, list):
            raise ValueError(
                f"stage authority mapping {collection_name} must be an array"
            )
        effective_bindings = [
            *bindings,
            *(BUILDER_EVENT_OCCURRENCE_BINDINGS if kind == "event_occurrence" else ()),
        ]
        for binding_index, raw_binding in enumerate(effective_bindings):
            if not isinstance(raw_binding, Mapping):
                raise ValueError(
                    f"stage authority mapping {collection_name}[{binding_index}] must be an object"
                )
            binding = dict(raw_binding)
            family_id = str(binding.get("event_family_id") or "")
            if family_id not in mapped or not mapped[family_id]["executable"]:
                raise ValueError(
                    f"stage authority binding references a non-executable family: {family_id}"
                )
            stage_keys = binding.get("stage_keys")
            if not isinstance(stage_keys, list) or not stage_keys:
                raise ValueError(
                    f"stage authority binding {family_id} must select at least one stage"
                )
            if kind == "event_occurrence":
                if "binding_projection_from_previous" in binding:
                    raise ValueError(
                        "event-occurrence stage authority must use its exact event "
                        f"bindings: {family_id}"
                    )
                event_phase = binding.get("event_phase")
                if event_phase not in EVENT_OCCURRENCE_PHASES:
                    raise ValueError(
                        "stage authority occurrence must declare a canonical "
                        f"event phase: {family_id} {event_phase!r}"
                    )
                selected = {
                    "predicate_id": str(binding.get("trigger_predicate_id") or ""),
                    "direction": str(binding.get("direction") or ""),
                    "event_type_id": str(binding.get("event_type_id") or ""),
                }
                if not any(
                    all(option.get(key) == value for key, value in selected.items())
                    for option in _trigger_options(mapped[family_id])
                ):
                    raise ValueError(
                        f"stage authority occurrence is not a registered family trigger: "
                        f"{family_id} {selected}"
                    )
            else:
                if "event_phase" in binding:
                    raise ValueError(
                        "predicate-transition stage authority must not claim an "
                        f"event lifecycle phase: {family_id}"
                    )
                projection = binding.get("binding_projection_from_previous")
                if projection is not None:
                    if not isinstance(projection, Mapping) or not projection:
                        raise ValueError(
                            "binding_projection_from_previous must be a non-empty object: "
                            f"{family_id}"
                        )
                    current_roles = list(projection)
                    previous_roles = list(projection.values())
                    if (
                        any(
                            not isinstance(role, str) or not role
                            for role in (*current_roles, *previous_roles)
                        )
                        or len(current_roles) != len(set(current_roles))
                        or len(previous_roles) != len(set(previous_roles))
                    ):
                        raise ValueError(
                            "binding_projection_from_previous must be a one-to-one "
                            f"role map: {family_id}"
                        )
                predicate_id = str(binding.get("predicate_id") or "")
                direction = str(binding.get("direction") or "")
                if predicate_id not in mapped[family_id][
                    "authoritative_predicates"
                ] or direction not in {"rising", "falling"}:
                    raise ValueError(
                        f"stage authority transition is outside the family vocabulary: "
                        f"{family_id} {predicate_id}/{direction}"
                    )
            for stage_key in stage_keys:
                key = str(stage_key)
                if key in explicit:
                    raise ValueError(f"stage authority mapping duplicates {key}")
                explicit[key] = {"kind": kind, **binding}

    source_stages = _source_stages_from_objective(objective)
    extra_stage_keys = sorted(set(explicit) - set(source_stages))
    if extra_stage_keys:
        raise ValueError(
            f"stage authority mapping references unknown stages: {extra_stage_keys}"
        )
    extra_not_applicable_keys = sorted(set(not_applicable_stages) - set(source_stages))
    if extra_not_applicable_keys:
        raise ValueError(
            "stage authority mapping declares unknown not-applicable stages: "
            f"{extra_not_applicable_keys}"
        )
    duplicate_decisions = sorted(set(explicit) & set(not_applicable_stages))
    if duplicate_decisions:
        raise ValueError(
            "stage authority mapping gives executable and not-applicable decisions: "
            f"{duplicate_decisions}"
        )
    for key, binding in explicit.items():
        projection = binding.get("binding_projection_from_previous")
        if projection is None:
            continue
        stage_index = int(source_stages[key]["stage_index"])
        if stage_index == 0:
            raise ValueError(
                f"root stage cannot declare a prior binding projection: {key}"
            )
        previous_key = f"{source_stages[key]['epi_id']}::{stage_index - 1}"
        if (
            source_stages[previous_key]["event_family_id"]
            != source_stages[key]["event_family_id"]
        ):
            raise ValueError(
                "binding projection requires consecutive stages from the same "
                f"event family: {key}"
            )
    for key, decision in not_applicable_stages.items():
        if not isinstance(decision, Mapping):
            raise ValueError(f"not-applicable stage {key} must be an object")
        if not str(decision.get("reason") or "").strip():
            raise ValueError(f"not-applicable stage {key} lacks a reason")
        if not str(decision.get("closure_statement") or "").strip():
            raise ValueError(f"not-applicable stage {key} lacks a future path")
    validation_policy = stage_mapping.get("validation")
    if not isinstance(validation_policy, Mapping):
        raise ValueError("stage authority mapping lacks validation")
    if int(validation_policy.get("expected_stage_count", -1)) != len(source_stages):
        raise ValueError("stage authority mapping stage-count drift")
    if int(validation_policy.get("expected_epi_count", -1)) != len(
        objective["epi_contracts"]
    ):
        raise ValueError("stage authority mapping EPI-count drift")

    rebuilt: dict[str, dict[str, Any]] = {}
    for key, old_stage in source_stages.items():
        epi_id = str(old_stage["epi_id"])
        family_id = str(old_stage["event_family_id"])
        if family_id not in mapped:
            raise ValueError(
                f"stage source {key} references unknown family {family_id}"
            )
        family = objective["event_families"][family_id]
        binding = explicit.get(str(key))
        if binding is not None and str(binding["event_family_id"]) != family_id:
            raise ValueError(
                f"stage authority family drift at {key}: "
                f"{binding['event_family_id']} != {family_id}"
            )
        if binding is None:
            if str(key) in not_applicable_stages:
                decision = not_applicable_stages[str(key)]
                reason = str(decision["reason"])
                closure_statement = str(decision["closure_statement"])
            elif not family["executable"]:
                reason = str(family["non_executable_reason"])
                closure_statement = str(family["closure_statement"])
            else:
                raise ValueError(
                    "executable-family stage lacks explicit authority decision: "
                    f"{key}"
                )
            stage = _not_applicable_stage(
                old_stage,
                reason=reason,
                closure_statement=closure_statement,
            )
        elif binding["kind"] == "event_occurrence":
            stage = {
                "key": str(key),
                "epi_id": epi_id,
                "stage_index": int(old_stage["stage_index"]),
                "stage_goal": str(old_stage["stage_goal"]),
                "event_family_id": family_id,
                "implementation_status": "executable",
                "clause_policy": "single",
                "acceptance_clauses": [_stage_occurrence_clause(binding)],
                "required_evidence_types": list(OCCURRENCE_EVIDENCE),
                "temporal_order": deepcopy(old_stage["temporal_order"]),
            }
            if binding.get("publish_event_phase", True):
                stage["event_phase"] = str(binding["event_phase"])
        elif binding["kind"] == "predicate_transition":
            stage = {
                "key": str(key),
                "epi_id": epi_id,
                "stage_index": int(old_stage["stage_index"]),
                "stage_goal": str(old_stage["stage_goal"]),
                "event_family_id": family_id,
                "implementation_status": "executable",
                "clause_policy": "single",
                "acceptance_clauses": [_stage_transition_clause(binding)],
                "required_evidence_types": list(TRANSITION_EVIDENCE),
                "temporal_order": deepcopy(old_stage["temporal_order"]),
            }
            if "binding_projection_from_previous" in binding:
                stage["binding_projection_from_previous"] = deepcopy(
                    binding["binding_projection_from_previous"]
                )
        else:
            raise ValueError(f"unsupported stage authority binding kind at {key}")
        source_stage = objective["epi_contracts"][epi_id]["required_chain"][
            int(old_stage["stage_index"])
        ]
        alternatives = source_stage.get("alternative_event_family_ids")
        if isinstance(alternatives, list) and alternatives:
            stage["alternative_event_family_ids"] = deepcopy(alternatives)
        rebuilt[str(key)] = stage
    result["stages"] = rebuilt

    catalog = set(result["evidence_type_catalog"])
    used_evidence_types = {
        str(evidence_type)
        for stage in rebuilt.values()
        for evidence_type in stage["required_evidence_types"]
    }
    missing_evidence_types = sorted(used_evidence_types - catalog)
    if missing_evidence_types:
        raise ValueError(
            "stage evidence catalog omits required evidence types: "
            f"{missing_evidence_types}"
        )

    status_counts = {
        status: sum(
            stage["implementation_status"] == status for stage in rebuilt.values()
        )
        for status in ("executable", "not_applicable")
    }
    validation = result["validation"]
    validation["expected_epi_count"] = len(objective["epi_contracts"])
    validation["expected_stage_count"] = len(rebuilt)
    validation["expected_executable_stage_count"] = status_counts["executable"]
    validation["expected_not_applicable_stage_count"] = status_counts["not_applicable"]
    return result


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    source = _load(SOURCE_PATH)
    mapping = _load(MAPPING_PATH)
    stage_mapping = _load(STAGE_MAPPING_PATH)
    if mapping.get("family_count") != len(mapping.get("families", {})):
        raise ValueError("mapping family_count drift")
    event_roles = _event_catalog_roles(_load_yaml(EVENT_CATALOG_PATH))
    objective = _build_objective(source, mapping, event_roles)
    stage = _build_stage_contract(source, objective, mapping, stage_mapping)
    return objective, stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    objective, stage = build()
    expected = {
        OBJECTIVE_PATH: _render(objective),
        STAGE_PATH: _render(stage),
    }
    stale = [
        path
        for path, value in ((OBJECTIVE_PATH, objective), (STAGE_PATH, stage))
        if not path.is_file() or _load(path) != value
    ]
    if args.check:
        if stale:
            raise SystemExit(
                "current event contracts are stale: "
                + ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in stale)
            )
        print("current event contracts are current")
        return 0
    for path, text in expected.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "event_family_count": len(objective["event_families"]),
                "epi_count": len(objective["epi_contracts"]),
                "stage_count": len(stage["stages"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
