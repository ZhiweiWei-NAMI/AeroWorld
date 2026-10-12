"""Evidence-based acceptance for the fixed executable predicate inventory.

Registration supplies the rows, never their acceptance status.  Each required
check needs explicit execution evidence; unavailable data remains unavailable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from Dataset.semantic_truth.core_semantic_registry import load_core_semantic_registry


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = REPO_ROOT / "Dataset/world_model/runs/rule_factor_graph"
REQUIRED_CHECKS = (
    "contract_structure", "typed_grounding", "binding_output_completeness", "hard_support_boundaries",
    "shared_state", "encoder_scorer_forward", "complete_dependency_closure",
    "joint_oracle", "event_joint_oracle", "censoring", "real_replay",
    "real_prediction_window", "full_known_zero_gradient",
    "latent_gradient_fd_update", "generic_invariances",
)
CHECK_STATUSES = frozenset({"pending", "pass", "fail", "blocked", "not_available"})
CLOSURE_CHECKS = ("current_contract_deterministic", "alias_binding", "raw_replay",
                  "controlled_replay", "version_compatibility")
CONTROLLED_GATE_PREDICATES = frozenset({
    "airspace.aircraft_inside_protected_airspace", "ground.pedestrian_in_crosswalk",
    "regulation.pedestrian_has_crosswalk_priority",
})
VERSION_GATE_PREDICATES = frozenset({
    "airspace.aircraft_inside_restricted_airspace", "airspace.aircraft_is_non_cooperative",
})


def _set_path(context: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    target = context
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def _leaf_nodes(spec: Mapping[str, Any]):
    if spec["op"] == "all":
        for argument in spec["args"]:
            yield from _leaf_nodes(argument)
    else:
        yield spec


def boundary_domains(template: Mapping[str, Any], parameters: Mapping[str, Any]) -> dict[str, tuple]:
    """Operator-defined finite fixtures, not claimed physical state domains.

    These fixtures cover truth tables and threshold equality.  Prediction
    domains are separately declared by each runtime request.
    """
    choices = {}
    for field in template["state_contract"]["required_fields"]:
        unit = field["unit"]
        choices[field["field"]] = [True, False] if unit == "bool" else (
            ["fixture_present", "none"] if unit == "identifier" else
            ["true", "false"] if unit == "four-value" else
            ["fixture_category"] if unit == "state-code" else [0.0, 1.0])
    for node in _leaf_nodes(template["evaluation_spec"]):
        op = node["op"]
        field = node.get("field")
        if op in {"gt", "gte", "lt", "lte"}:
            threshold = float(parameters[node["threshold"]])
            delta = max(abs(threshold), 1.0) * 0.25
            choices[field] = [threshold - delta, threshold, threshold + delta]
        elif op == "gt_constant":
            constant = float(node["value"])
            choices[field] = [constant - 1.0, constant, constant + 1.0]
        elif op in {"truthy", "falsy"}:
            choices[field] = [False, True]
        elif op == "direct_truth":
            choices[field] = ["false", "true"]
        elif op == "equals":
            choices[field] = [node["value"], "fixture_other_category"]
        elif op == "in":
            choices[field] = list(node["values"]) + ["fixture_other_category"]
        elif op == "present":
            choices[field] = ["fixture_present", "none"]
        elif op == "gt_fields":
            choices[node["left"]] = [0.0, 1.0, 2.0]
            choices[node["right"]] = [0.0, 1.0, 2.0]
        elif op == "absolute_difference_gt":
            threshold = float(parameters[node["threshold"]])
            choices[node["left"]] = [-threshold - 1.0, -threshold, 0.0, threshold, threshold + 1.0]
            choices[node["right"]] = [0.0]
        elif op == "outside_range":
            minimum, maximum = float(parameters[node["minimum"]]), float(parameters[node["maximum"]])
            choices[field] = [minimum - 1.0, minimum, (minimum + maximum) / 2.0, maximum, maximum + 1.0]
        elif op == "changed":
            choices[field] = [0.0, 1.0]
        else:
            raise ValueError(f"No boundary fixture for actual operator {op}")
    return {field: tuple(values) for field, values in choices.items()}


def validate_rule_boundaries(contract, compiled_rule, parameters: Mapping[str, Any]) -> dict:
    """Compare compiled support with the current evaluator and missing policy."""
    from itertools import product
    from Dataset.semantic_truth import world_truth as authority
    template = contract.template
    domains = boundary_domains(template, parameters)
    fields = tuple(domains)
    records = []
    statuses = Counter()
    for values in product(*(domains[field] for field in fields)):
        context = {}
        for field, value in zip(fields, values):
            _set_path(context, field, value)
        expected = authority._expression_truth_value(authority._evaluate_expression(
            template["evaluation_spec"], context, {}, parameters))
        actual = compiled_rule.evaluate(context, {}, parameters)
        if actual != expected:
            raise AssertionError(f"{contract.predicate_id}: compiled={actual} authority={expected} context={context}")
        statuses[actual] += 1
        records.append({"context": context, "expected": expected, "actual": actual})
    if not statuses["true"] or not statuses["false"]:
        raise AssertionError(f"{contract.predicate_id}: boundary domains did not exercise true and false")
    valid_context = deepcopy(records[0]["context"])
    missing_records = []
    for field in fields:
        context = deepcopy(valid_context)
        _set_path(context, field, None)
        actual = compiled_rule.evaluate(context, {}, parameters)
        if actual != "unknown":
            raise AssertionError(f"{contract.predicate_id}: required field {field} missing must be unknown")
        missing_records.append({"missing_field": field, "actual": actual})
    outside = compiled_rule.evaluate(valid_context, {}, parameters, applicable=False)
    if outside != "out_of_scope":
        raise AssertionError(f"{contract.predicate_id}: inapplicable grounding did not retain out_of_scope")
    if contract.semantic_mode == "direct_observation":
        direct_context = {"direct_predicate_value": "out_of_scope"}
        if compiled_rule.evaluate(direct_context, {}, parameters) != "out_of_scope":
            raise AssertionError("direct_observation lost source applicability")
    return {"source": "synthetic_operator_boundary_fixture", "truth_table": records,
            "truth_counts": dict(statuses), "missing": missing_records,
            "out_of_scope": outside, "domains": domains,
            "scope": "finite_fixture_truth_tables_and_equality_not_continuous_exactness"}


def operator_counts(spec: Mapping[str, Any]) -> Counter:
    result = Counter({str(spec["op"]): 1})
    for argument in spec.get("args", ()):
        result.update(operator_counts(argument))
    return result


def initial_matrix() -> dict[str, Any]:
    registry = load_core_semantic_registry()
    executable = tuple(registry["implementation"]["executable_predicate_ids"])
    templates = {template["id"]: template for template in registry["templates"]}
    if len(executable) != 72 or len(templates) != 79:
        raise ValueError("The unified stage requires the current fixed 79/72 inventory")
    rows = []
    total_operators = Counter()
    for index, predicate_id in enumerate(executable):
        template = templates[predicate_id]
        counts = operator_counts(template["evaluation_spec"])
        total_operators.update(counts)
        direct = template["evaluation_spec"]["op"] == "direct_truth"
        rows.append({
            "inventory_index": index,
            "predicate_id": predicate_id,
            "semantic_mode": "direct_observation" if direct else "declared_formula",
            "physical_formula": "not_declared" if direct else "declared",
            "physical_semantic_status": "not_declared" if direct else "pending",
            "semantic_ambiguity": [],
            "source_refs": [
                "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json",
                "Dataset/semantic_rules/predicates/world_truth_predicate_contracts.yaml",
                f"Dataset/knowledge_graph/domain/{template['ontology_module']}.ttl",
            ],
            "argument_roles": template["argument_roles"],
            "required_fields": template["state_contract"]["required_fields"],
            "threshold_parameters": template["state_contract"]["threshold_parameters"],
            "evaluation_spec": template["evaluation_spec"],
            "operators": dict(sorted(counts.items())),
            "missing_operators": [],
            "typed_bindings": [],
            "real_replay_cases": 0,
            "real_prediction_windows": 0,
            "natural_missing_cases": 0,
            "synthetic_hidden_cases": 0,
            "graph_reports": [],
            "checks": {name: {"status": "pending", "evidence": [], "reason": "not_run"}
                       for name in REQUIRED_CHECKS},
            "structure_support": False,
            "binding_pass": False,
            "contract_semantics": False,
            "real_data_support": False,
            "joint_event_support": False,
            "overall_pass": False,
        })
    non_executable = registry["implementation"]["non_executable_predicates"]
    if len(non_executable) != 7:
        raise ValueError("The seven declared non-executable predicates must stay excluded")
    matrix = {
        "schema": "aeroworld.unified72.functional-acceptance.v1",
        "inventory_source": "Dataset/semantic_truth/core_semantic_registry.py",
        "inventory_count": 79, "required_count": 72,
        "direct_observation_count": sum(row["semantic_mode"] == "direct_observation" for row in rows),
        "operator_node_counts": dict(sorted(total_operators.items())),
        "required_checks": list(REQUIRED_CHECKS),
        "excluded_non_executable": non_executable,
        "rows": rows,
        "generic_checks": {},
        "acceptance_scope": "finite_candidate_engine_and_explicit_contract_semantics",
        "physical_semantics_claim": "direct_observation rows have no declared physical formula",
    }
    refresh_matrix(matrix)
    return matrix


def record_check(row: dict, name: str, status: str, evidence=(), reason: str = "") -> None:
    if name not in REQUIRED_CHECKS or status not in CHECK_STATUSES:
        raise ValueError(f"Invalid acceptance check: {name}={status}")
    evidence = list(evidence)
    if status == "pass" and not evidence:
        raise ValueError(f"Passing check {name} requires execution evidence")
    if status != "pass" and not reason:
        raise ValueError(f"Nonpassing check {name} requires a reason")
    row["checks"][name] = {"status": status, "evidence": evidence, "reason": reason}


def refresh_matrix(matrix: dict) -> dict:
    rows = matrix["rows"]
    expected = tuple(load_core_semantic_registry()["implementation"]["executable_predicate_ids"])
    if tuple(row["predicate_id"] for row in rows) != expected:
        raise ValueError("Coverage rows differ from the fixed executable inventory")
    def passed(row, names):
        return all(row["checks"][name]["status"] == "pass" for name in names)
    for row in rows:
        if tuple(row["checks"]) != REQUIRED_CHECKS:
            raise ValueError(f"Required checks changed for {row['predicate_id']}")
        for name, check in row["checks"].items():
            if check["status"] not in CHECK_STATUSES:
                raise ValueError(f"Invalid status for {row['predicate_id']}:{name}")
            if check["status"] == "pass" and not check["evidence"]:
                raise ValueError(f"Missing evidence for {row['predicate_id']}:{name}")
        if row["checks"]["real_replay"]["status"] == "pass" and row["real_replay_cases"] <= 0:
            raise ValueError(f"No real replay cases for {row['predicate_id']}")
        if row["checks"]["real_prediction_window"]["status"] == "pass" and row["real_prediction_windows"] <= 0:
            raise ValueError(f"No real prediction window for {row['predicate_id']}")
        row["structure_support"] = passed(row, (
            "contract_structure", "typed_grounding", "shared_state", "encoder_scorer_forward",
            "complete_dependency_closure", "generic_invariances"))
        row["binding_pass"] = passed(row, ("binding_output_completeness",))
        row["contract_semantics"] = passed(row, ("hard_support_boundaries", "real_replay"))
        row["real_data_support"] = passed(row, ("real_replay", "real_prediction_window"))
        row["joint_event_support"] = passed(row, ("joint_oracle", "event_joint_oracle", "censoring"))
        row["overall_pass"] = passed(row, REQUIRED_CHECKS)
        if row["semantic_mode"] == "declared_formula":
            row["physical_semantic_status"] = (
                "semantic_ambiguity" if row["semantic_ambiguity"] else
                "contract_verified" if row["contract_semantics"] else "pending")
    matrix["counts"] = {
        key: sum(bool(row[key]) for row in rows)
        for key in ("structure_support", "binding_pass", "contract_semantics", "real_data_support", "joint_event_support", "overall_pass")
    }
    matrix["complete"] = matrix["counts"]["overall_pass"] == matrix["required_count"]
    matrix["updated_at"] = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
    return matrix


def write_matrix(matrix: dict, output: Path = RUN_ROOT) -> None:
    refresh_matrix(matrix)
    output.mkdir(parents=True, exist_ok=True)
    (output / "coverage_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    blockers = [{"predicate_id": row["predicate_id"], "checks": {
        name: check["reason"] for name, check in row["checks"].items() if check["status"] != "pass"
    }} for row in matrix["rows"] if not row["overall_pass"]]
    summary = {key: matrix[key] for key in (
        "schema", "inventory_count", "required_count", "direct_observation_count",
        "operator_node_counts", "counts", "complete", "updated_at", "acceptance_scope",
        "physical_semantics_claim",
    )}
    summary["blockers"] = blockers
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def initial_dba_matrix() -> dict[str, Any]:
    """The controlling deterministic/binding/alias gate precedes neural work."""
    registry = load_core_semantic_registry()
    templates = {template["id"]: template for template in registry["templates"]}
    executable = tuple(registry["implementation"]["executable_predicate_ids"])
    rows = []
    for index, predicate_id in enumerate(executable):
        template = templates[predicate_id]
        direct = template["evaluation_spec"]["op"] == "direct_truth"
        rows.append({
            "inventory_index": index, "predicate_id": predicate_id,
            "semantic_mode": "direct_observation" if direct else "declared_formula",
            "physical_formula": "not_declared" if direct else "declared",
            "authored_expression": template["expression"],
            "evaluation_spec": template["evaluation_spec"],
            "argument_roles": template["argument_roles"],
            "declared_output_count": 1,
            "consequent_semantics": "pending_source_review",
            "source_refs": [
                f"Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json#predicate={predicate_id}",
                f"Dataset/semantic_rules/predicates/world_truth_predicate_contracts.yaml#predicate={predicate_id}",
                f"Dataset/knowledge_graph/domain/{template['ontology_module']}.ttl",
            ],
            "D": {"status": "pending", "proof_status": "pending", "natural_validation_status": "pending",
                  "source_case_count": 0, "proof_evidence": [], "natural_evidence": [], "evidence": [],
                  "proof_counterexamples": [], "natural_counterexamples": [], "counterexamples": [], "reason": "not_run"},
            "B": {"status": "pending", "proof_status": "pending", "natural_validation_status": "pending",
                  "source_case_count": 0, "output_instance_count": 0,
                  "proof_evidence": [], "natural_evidence": [], "proof_counterexamples": [], "natural_counterexamples": [],
                  "negative_checks": [], "evidence": [], "counterexamples": [], "reason": "not_run"},
            "A": {"status": "pending", "proof_status": "pending", "natural_validation_status": "pending",
                  "source_case_count": 0, "declared_uris": [], "proof_evidence": [], "natural_evidence": [],
                  "proof_counterexamples": [], "natural_counterexamples": [],
                  "runtime_aliases": [], "field_owners": [], "evidence": [],
                  "counterexamples": [], "reason": "not_run"},
            "semantic_ambiguity": [],
            "independent_review": {"owner": "/root/unified_science" if index < 34 else "/root/unified_review", "status": "pending",
                                   "conclusion": "not_reviewed", "evidence": []},
            "overall_pass": False,
        })
    result = {"schema": "aeroworld.unified72.deterministic-binding-alias.v1",
              "required_count": 72, "inventory_count": 79,
              "excluded_non_executable": registry["implementation"]["non_executable_predicates"],
              "rows": rows, "neural_inference_work": "paused_until_this_gate_is_independently_complete",
              "acceptance_policy": "formal_D_B_A_proof_and_required_natural_validation_and_independent_review",
              "proof_policy": "formal proof can pass without natural instances; missing natural validation remains separate",
              "D_natural_scope": "current-source four-value replay, including unknown; binary supervision and prediction windows are separate downstream requirements",
              "direct_observation_policy": "readvalue projection preserves four values; no physical formula claim"}
    refresh_dba_matrix(result)
    return result


def refresh_dba_matrix(matrix: dict) -> dict:
    expected = tuple(load_core_semantic_registry()["implementation"]["executable_predicate_ids"])
    if len(expected) != 72 or tuple(row["predicate_id"] for row in matrix["rows"]) != expected:
        raise ValueError("D/B/A rows differ from the fixed executable inventory")
    for row in matrix["rows"]:
        for gate in ("D", "B", "A"):
            check = row[gate]
            for field in ("proof_status", "natural_validation_status"):
                if check[field] not in CHECK_STATUSES:
                    raise ValueError(f"Invalid D/B/A status: {row['predicate_id']}:{gate}:{field}")
            if check["proof_status"] == "pass" and (not check["proof_evidence"] or check["proof_counterexamples"]):
                raise ValueError(f"Passing {gate} proof requires evidence and no unresolved proof counterexamples")
            if check["natural_validation_status"] == "pass" and (not check["natural_evidence"]
                    or check["source_case_count"] <= 0 or check["natural_counterexamples"]):
                raise ValueError(f"Passing {gate} natural validation requires actual records and no unresolved counterexamples")
            statuses = (check["proof_status"], check["natural_validation_status"])
            check["status"] = ("pass" if statuses == ("pass", "pass") else
                               "fail" if "fail" in statuses else
                               "blocked" if "blocked" in statuses else
                               "not_available" if "not_available" in statuses else "pending")
            check["evidence"] = check["proof_evidence"] + check["natural_evidence"]
            check["counterexamples"] = check["proof_counterexamples"] + check["natural_counterexamples"]
        if row["B"]["status"] == "pass" and row["B"]["output_instance_count"] <= 0:
            raise ValueError("Output binding pass requires actual output instances")
        review = row["independent_review"]
        if review["status"] not in CHECK_STATUSES:
            raise ValueError("Invalid independent review status")
        if review["status"] == "pass" and (not review["owner"] or not review["evidence"]
                                             or review["conclusion"] == "not_reviewed"):
            raise ValueError("Independent pass requires an owner, conclusion and execution evidence")
        row["overall_pass"] = all(row[gate]["status"] == "pass" for gate in ("D", "B", "A")) and review["status"] == "pass"
    matrix["counts"] = {gate: sum(row[gate]["status"] == "pass" for row in matrix["rows"])
                        for gate in ("D", "B", "A")}
    for gate in ("D", "B", "A"):
        matrix["counts"][gate + "_proof"] = sum(row[gate]["proof_status"] == "pass" for row in matrix["rows"])
        matrix["counts"][gate + "_natural"] = sum(row[gate]["natural_validation_status"] == "pass" for row in matrix["rows"])
    matrix["counts"]["four_value_replay_pass"] = sum(row.get("real_replay", {}).get("four_value_replay_status") == "pass"
                                                     for row in matrix["rows"])
    matrix["counts"]["binary_backed_replay_pass"] = sum(
        row.get("real_replay", {}).get("four_value_replay_status") == "pass"
        and row.get("real_replay", {}).get("binary_selected_case_count", 0) > 0 for row in matrix["rows"])
    matrix["counts"]["independently_reviewed"] = sum(row["independent_review"]["status"] == "pass" for row in matrix["rows"])
    matrix["counts"]["overall_pass"] = sum(row["overall_pass"] for row in matrix["rows"])
    matrix["formal_proof_complete"] = all(
        row[gate]["proof_status"] == "pass" for row in matrix["rows"] for gate in ("D", "B", "A")) and all(
        row["independent_review"]["status"] == "pass" for row in matrix["rows"])
    matrix["complete"] = matrix["counts"]["overall_pass"] == 72
    matrix["updated_at"] = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
    if "closure_gate" in matrix:
        refresh_closure_gate(matrix)
    return matrix


def write_dba_matrix(matrix: dict, output: Path = RUN_ROOT) -> None:
    refresh_dba_matrix(matrix)
    output.mkdir(parents=True, exist_ok=True)
    (output / "deterministic_binding_alias_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    if "closure_gate" in matrix:
        write_closure_summary(matrix, output)


def attach_dba_contract_sources(matrix: dict, metadata: Mapping[str, Any]) -> None:
    """Attach authority references without turning metadata into acceptance."""
    by_id = {item["predicate_id"]: item for item in metadata["rows"]}
    if tuple(by_id) != tuple(row["predicate_id"] for row in matrix["rows"]):
        raise ValueError("Contract evidence does not cover the fixed ordered inventory")
    for row in matrix["rows"]:
        item = by_id[row["predicate_id"]]
        mapping = item["runtime_mapping"]
        ontology = item["ontology"]
        row["consequent_semantics"] = mapping["engine_logical_direction"]
        row["declared_output_count"] = mapping["output_count"]
        row["ontology_formal_implication_direction"] = mapping["ontology_formal_implication_direction"]
        row["ontology_definition"] = ontology["definition_en"]
        row["D"]["source_divergence"] = item["source_divergence"]
        row["D"]["contract_metadata_ref"] = "sources/deterministic_contract_sources.json#predicate=" + row["predicate_id"]
        row["A"]["declared_uris"] = [ontology["predicate_uri"], *(
            role["declared_class_uri"] for role in ontology["argument_roles"])]
        row["A"]["runtime_aliases"] = ontology["argument_roles"]
        row["A"]["field_owners"] = mapping["state_ownership"]
        row["provenance_precision_limitations"] = [
            {"field": field["input_path"], "note": field["semantic_ambiguity"]}
            for field in mapping["state_ownership"] if field.get("semantic_ambiguity")]
        if item["source_divergence"]:
            row["provenance_precision_limitations"].append({
                "note": "Current source projection differs from committed contract; this is a provenance difference, not an alias",
                "source_divergence": item["source_divergence"]})


def audit_dba_source_cases(contracts: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the deterministic and output gates on recorded source examples.

    This result deliberately proposes no normative or alias acceptance. Those
    columns also require authored-source review and precise alias authority.
    """
    from dataclasses import replace
    from Dataset.semantic_truth import world_truth as authority
    from .ast_compiler import compile_rule
    from .generic_grounding import (EntityClassCatalog, OutputBindingAuthority, ParameterProfile,
                                    TypedEntity, bind_roles, ground_contract)
    from .unified_contracts import ContractError
    records = {}
    raw_catalogs = bundle["class_catalogs_by_episode"]
    catalogs = {episode: EntityClassCatalog(episode,
        {entity_id: tuple(classes) for entity_id, classes in raw["classes"].items()},
        {entity_id: tuple(refs) for entity_id, refs in raw["source_refs"].items()})
        for episode, raw in raw_catalogs.items()}
    for predicate_id, contract in contracts.items():
        cases = bundle["cases_by_predicate"][predicate_id]
        result = {"predicate_id": predicate_id, "source_case_count": len(cases),
                  "binary_source_case_count": 0, "replay": [], "output_bindings": [],
                  "counterexamples": [], "binding_errors": [], "negative_checks": [], "positive_symmetry_checks": []}
        compiled = compile_rule(contract)
        for case in cases:
            context = case["context"]
            inactive = authority._value_at_path(context, "l0.scope_active") is False
            source_unavailable = authority._value_at_path(context, "l0.predicate_source_unavailability") is not authority.MISSING
            actual = compiled.evaluate(context, case["previous_context"], case["parameters"],
                                       applicable=not inactive, source_out_of_scope=source_unavailable)
            result["binary_source_case_count"] += case["saved_truth"] in {"false", "true"}
            replay = {"source_ref": case["graph_node_ref"], "saved_truth": case["saved_truth"],
                      "recorded_context_replay": actual, "source_current_replay": case["current_truth"],
                      "missing_fields": case["missing_fields"], "source_refs": case["source_refs"]}
            result["replay"].append(replay)
            if actual != case["current_truth"] or actual != case["saved_truth"]:
                result["counterexamples"].append(replay)
            profile_raw = case["parameter_profile"]
            profile = ParameterProfile(profile_raw["profile_id"], profile_raw["parameters"], profile_raw["source_ref"])
            independent = case["output_binding_authority"]
            independent_bindings = bind_roles(contract, independent["bindings"])
            output_authority = OutputBindingAuthority(independent["predicate_id"], independent_bindings,
                independent["tick"], independent["episode_id"], independent["branch_id"],
                independent["parameter_profile_id"], independent["parameters"],
                tuple(independent.get("source_refs", (independent["source_ref"],))))
            # Missing observations stay a declared None candidate, not invented
            # physics or a binary observation. This is only a binding check.
            domains = {field.field: (None,) for field in contract.required_fields
                       if field.field in case["missing_fields"]}
            variables = {}
            try:
                rule = ground_contract(contract, case["bindings"], context, case["parameters"],
                    tick=case["tick"], episode_id=case["episode_id"], branch_id=case["branch_id"],
                    domain_by_field=domains, observed_fields=case["observed_fields"],
                    previous_context=case["previous_context"], previous_tick=case["tick"] - 5 if case["tick"] >= 5 else None,
                    predicate_evidence=case["saved_truth"], variable_registry=variables,
                    source_refs=tuple(case["source_refs"]), parameter_profile=profile,
                    output_authority=output_authority, applicable=not inactive,
                    source_out_of_scope=source_unavailable, class_catalog=catalogs[case["episode_id"]])
            except (ContractError, KeyError, ValueError) as exc:
                result["binding_errors"].append({"source_ref": case["graph_node_ref"], "error": str(exc)})
                continue
            output = rule.output
            result["output_bindings"].append({"source_ref": case["graph_node_ref"],
                "predicate_id": contract.predicate_id, "tick": output.tick, "episode_id": output.episode_id,
                "branch_id": output.branch_id, "parameter_profile_id": profile.profile_id,
                "parameters": dict(rule.parameters),
                "owner": [{"entity_id": entity.entity_id, "ontology_class": entity.ontology_class,
                           "ontology_classes": entity.ontology_classes} for entity in output.owner],
                "roles": [{"role": binding.role, "entity_id": binding.entity_id,
                           "ontology_class": binding.ontology_class,
                           "ontology_classes": binding.entity.ontology_classes} for binding in rule.bindings],
                "input_scope": [{"field": field, "key_field": key.field, "tick": key.tick,
                                 "owner": [{"entity_id": entity.entity_id, "ontology_class": entity.ontology_class,
                                            "ontology_classes": entity.ontology_classes} for entity in key.owner]}
                                for field, key in rule.inputs],
                "output_count": 1, "declared_consequents": [contract.predicate_id]})
            first = output.owner[0]
            wrong_owner = (TypedEntity(first.entity_id + "__wrong_binding", first.ontology_class), *output.owner[1:])
            wrong_role = replace(rule.bindings[0], role=rule.bindings[0].role + "__wrong_role")
            mutations = {
                "wrong_predicate_output": lambda: replace(rule, output=replace(output, field="predicate:invalid_output")),
                "wrong_typed_entity_owner": lambda: replace(rule, output=replace(output, owner=wrong_owner)),
                "wrong_argument_role": lambda: replace(rule, bindings=(wrong_role, *rule.bindings[1:])),
                "wrong_output_tick": lambda: replace(rule, output=replace(output, tick=output.tick + 5)),
                "wrong_branch_profile": lambda: replace(rule, output=replace(output, branch_id=output.branch_id + "__wrong")),
                "wrong_parameter_profile_id": lambda: replace(rule, parameter_profile=replace(profile, profile_id=profile.profile_id + "__wrong")),
            }
            # Every source profile has numeric parameters, even when this
            # particular rule uses no threshold. Alter its independent snapshot.
            snapshot = dict(profile.parameters)
            if snapshot:
                name = next(iter(snapshot))
                snapshot[name] += 1
                mutations["wrong_parameter_profile_snapshot"] = lambda: replace(rule, parameter_profile=replace(profile, parameters=snapshot))
            if rule.parameters:
                wrong_parameters = dict(rule.parameters)
                name = next(iter(wrong_parameters))
                wrong_parameters[name] += 1
                mutations["wrong_rule_threshold_value"] = lambda: replace(rule, parameters=wrong_parameters)
            if rule.inputs:
                mutations["omitted_required_input"] = lambda: replace(rule, inputs=rule.inputs[1:])
            for name, mutation in mutations.items():
                rejected = False
                error = ""
                try:
                    mutation()
                except ContractError as exc:
                    rejected, error = True, str(exc)
                check = {"source_ref": case["graph_node_ref"], "mutation": name,
                         "rejected": rejected, "error": error,
                         "source_kind": "negative_mutation_of_actual_record_binding"}
                result["negative_checks"].append(check)
                if not rejected:
                    result["binding_errors"].append(check)
            for group in contract.grounding_spec["unordered_role_sets"]:
                swapped = deepcopy(case["bindings"])
                first_role, second_role = group
                swapped[first_role], swapped[second_role] = swapped[second_role], swapped[first_role]
                try:
                    equivalent = ground_contract(contract, swapped, context, case["parameters"],
                        tick=case["tick"], episode_id=case["episode_id"], branch_id=case["branch_id"],
                        domain_by_field=domains, observed_fields=case["observed_fields"],
                        previous_context=case["previous_context"],
                        previous_tick=case["tick"] - 5 if case["tick"] >= 5 else None,
                        predicate_evidence=case["saved_truth"], variable_registry={},
                        source_refs=tuple(case["source_refs"]), parameter_profile=profile,
                        output_authority=output_authority, applicable=not inactive,
                        source_out_of_scope=source_unavailable, class_catalog=catalogs[case["episode_id"]])
                    symmetry_ok, error = equivalent.output == output, ""
                except ContractError as exc:
                    symmetry_ok, error = False, str(exc)
                check = {"source_ref": case["graph_node_ref"], "unordered_group": group,
                         "same_output": symmetry_ok, "error": error,
                         "source_kind": "legal_unordered_permutation_of_actual_record_binding"}
                result["positive_symmetry_checks"].append(check)
                if not symmetry_ok:
                    result["binding_errors"].append(check)
        records[predicate_id] = result
    return {"schema": "aeroworld.unified72.dba-record-execution.v1", "rows": records,
            "scope": "recorded_source_replay_and_binding_negatives; normative_and_alias_review_required"}


def attach_dba_natural_execution(matrix: dict, bundle: Mapping[str, Any], execution: Mapping[str, Any]) -> None:
    """Keep natural source reconstruction and all-instance enumeration distinct."""
    for row in matrix["rows"]:
        predicate_id = row["predicate_id"]
        result = execution["rows"][predicate_id]
        cases = bundle["cases_by_predicate"][predicate_id]
        row["real_replay"] = {"status": "available" if cases else "not_available",
                              "selected_case_count": len(cases),
                              "read_case_count": bundle["coverage"][predicate_id]["real_replay_case_count"],
                              "raw_verified_case_count": sum(bool(case.get("raw_source_verified")) for case in cases),
                              "binary_selected_case_count": result["binary_source_case_count"],
                              "four_value_replay_status": "pending",
                              "binary_supervision_status": "available" if result["binary_source_case_count"] else "not_available",
                              "prediction_window_status": "not_evaluated_in_DBA_gate"}
        ref = {"artifact": "evidence_dba_source_execution.json#predicate=" + predicate_id,
               "source_bundle": "sources/real_source_cases.json#predicate=" + predicate_id,
               "source_refs": [case["graph_node_ref"] for case in cases]}
        for gate in ("D", "B", "A"):
            row[gate]["source_case_count"] = len(cases)
            row[gate]["natural_evidence"] = [ref] if cases else []
        row["B"]["output_instance_count"] = len(result["output_bindings"])
        row["B"]["negative_checks"] = result["negative_checks"]
        row["B"]["positive_symmetry_checks"] = result["positive_symmetry_checks"]
        row["B"]["natural_counterexamples"] = result["binding_errors"]
        row["D"]["natural_counterexamples"] = result["counterexamples"]
        row["D"]["source_field_differences"] = [
            {"source_ref": case["graph_node_ref"], "differences": case.get("raw_field_differences", [])}
            for case in cases if case.get("raw_field_differences")]
        row["semantic_ambiguity"] = [
            {"source_ref": case["graph_node_ref"], "ambiguities": case["semantic_ambiguities"]}
            for case in cases if case["semantic_ambiguities"]]
        if not cases:
            row["real_replay"]["four_value_replay_status"] = "not_available"
            for gate in ("D", "B", "A"):
                row[gate]["natural_validation_status"] = "not_available"
                row[gate]["reason"] = "no_actual_natural_instances; formal_proof_status_is_separate"
            continue
        raw_incomplete = any(not case.get("raw_source_verified") for case in cases)
        ambiguity = bool(row["semantic_ambiguity"])
        if not raw_incomplete and not ambiguity and not result["counterexamples"]:
            row["real_replay"]["four_value_replay_status"] = "pass"
        if result["binding_errors"]:
            row["B"]["natural_validation_status"] = "fail"
            row["B"]["reason"] = "actual_record_binding_or_permutation_counterexample"
        else:
            row["B"]["natural_validation_status"] = "pending"
            row["B"]["reason"] = "selected_bindings_and_negatives_pass; all_instance_output_enumeration_pending"
        row["A"]["natural_validation_status"] = "pending"
        row["A"]["reason"] = "complete_actual_catalog_and_all_instance_merge_split_validation_pending"
        if raw_incomplete:
            row["D"]["natural_validation_status"] = "pending"
            row["D"]["reason"] = "strict_L0_source_projection_reconstruction_pending; no_rule_semantics_failure_claim"
        elif ambiguity or result["counterexamples"]:
            row["D"]["natural_validation_status"] = "pending"
            row["D"]["reason"] = "raw_vs_published_field_projection_difference_under_investigation; no_relabel"
        else:
            row["D"]["natural_validation_status"] = "pass"
            row["D"]["reason"] = ""
        if row["D"].get("current_producer_version_limitation"):
            row["real_replay"]["stored_raw_four_value_replay_status"] = row["real_replay"]["four_value_replay_status"]
            row["real_replay"]["four_value_replay_status"] = "pending_current_producer_version_review"
            row["D"]["natural_validation_status"] = "pending"
            row["D"]["reason"] = "stored_source_replay_and_current_producer_semantics_require_separate_version_review"


def attach_dba_population_execution(matrix: dict, bundle: Mapping[str, Any], execution: Mapping[str, Any]) -> None:
    """Accept only documented complete actual output populations and catalogs."""
    catalogs = bundle["class_catalogs_by_episode"]
    for row in matrix["rows"]:
        predicate_id = row["predicate_id"]
        coverage = bundle["coverage"][predicate_id]
        cases = bundle["cases_by_predicate"][predicate_id]
        result = execution["rows"][predicate_id]
        audits = coverage["population_audits"]
        row["B"]["all_instance_population_audits"] = audits
        source_errors = [audit for audit in audits if audit.get("source_population_error") or audit.get("error")]
        row["B"]["source_population_output_count"] = (
            None if source_errors else sum(audit["expected_candidate_count"] for audit in audits))
        row["A"]["actual_catalog_refs"] = [
            {"episode_id": episode, "catalog_id": catalogs[episode]["catalog_id"],
             "source_ref": "sources/real_source_cases.json#class_catalogs_by_episode/" + episode}
            for episode in sorted({case["episode_id"] for case in cases})]
        if not cases:
            continue  # Empty output sets never supply missing natural instances.
        ref = {"artifact": "sources/real_source_cases.json#coverage/" + predicate_id + "/population_audits",
               "selected_binding_execution": "evidence_dba_source_execution.json#predicate=" + predicate_id,
               "scope": "every actual candidate of the audited episode/tick populations, with exact typed role tuples"}
        complete = coverage["real_candidate_population_completeness_verified"]
        natural = any(audit["has_real_candidates"] for audit in audits)
        errors = source_errors + [audit for audit in audits if audit.get("has_real_candidates")
                                  and not audit["completeness_pass"]]
        if not complete or not natural or errors:
            for gate in ("B", "A"):
                row[gate]["natural_validation_status"] = "pending"
                row[gate]["reason"] = "actual_candidate_population_or_catalog_completeness_not_established"
            row["B"]["population_counterexamples"] = errors
            continue
        if result["binding_errors"]:
            continue
        if any(not case.get("raw_source_verified") for case in cases):
            continue
        for gate in ("B", "A"):
            row[gate]["natural_validation_status"] = "pass"
            row[gate]["natural_evidence"].append(ref)
            row[gate]["reason"] = ""


def initialize_closure_gate(matrix: dict, bundle: Mapping[str, Any], execution: Mapping[str, Any],
                            component_evidence: Mapping[str, Any]) -> None:
    """Add authorized controlled/version closure without changing natural counts."""
    refresh_dba_matrix(matrix)
    if "closure_gate" in matrix:
        raise ValueError("Controlled/version closure is already initialized")
    baseline = {name: matrix["counts"][name] for name in
                ("D_natural", "B_natural", "A_natural", "four_value_replay_pass",
                 "binary_backed_replay_pass", "overall_pass")}
    source_metrics = {
        "raw_source_predicates": bundle["summary"]["raw_verified_predicates"],
        "binary_source_predicates": bundle["summary"]["real_binary_predicates"],
        "selected_source_cases": sum(item["source_case_count"] for item in execution["rows"].values()),
        "selected_typed_outputs": sum(len(item["output_bindings"]) for item in execution["rows"].values()),
        "binding_negative_checks": sum(len(item["negative_checks"]) for item in execution["rows"].values()),
        "unordered_positive_checks": sum(len(item["positive_symmetry_checks"]) for item in execution["rows"].values()),
        "component_host_relations": component_evidence["relation_count"],
    }
    if not component_evidence["all_pass"]:
        raise ValueError("Actual component/host identity verification failed")
    for row in matrix["rows"]:
        predicate_id = row["predicate_id"]
        route = ("controlled_pipeline" if predicate_id in CONTROLLED_GATE_PREDICATES else
                 "source_version_adjudication" if predicate_id in VERSION_GATE_PREDICATES else "natural")
        if route == "natural" and not row["overall_pass"]:
            raise ValueError(f"Unexpected unresolved natural predicate: {predicate_id}")
        result = execution["rows"][predicate_id]
        cases = bundle["cases_by_predicate"][predicate_id]
        replay_pass = (bool(cases) and all(case["raw_source_verified"] for case in cases)
                       and not result["counterexamples"] and not result["binding_errors"])
        row["closure_route"] = route
        row["closure_provenance"] = {"raw_records": "preserved", "original_labels": "preserved",
                                       "current_contract_version_compatibility": "separate_adjudication"}
        row["closure_validation"] = {
            "current_contract_deterministic": {
                "status": row["D"]["proof_status"], "source_proof": deepcopy(row["D"]["proof_evidence"]),
                "source_kind": "formal_current_contract_proof", "case_count": 0,
                "scope": "Exact current expression and four-value semantics; producer replay remains separate."},
            "alias_binding": {
                "status": "pass" if row["B"]["status"] == row["A"]["status"] == "pass" else "pending",
                "source_proof": deepcopy(row["B"]["evidence"] + row["A"]["evidence"]),
                "source_kind": "actual_natural_binding_population" if cases else "formal_proof_without_natural_instances",
                "output_instance_count": row["B"]["output_instance_count"],
                "negative_checks": deepcopy(result["negative_checks"]),
                "positive_symmetry_checks": deepcopy(result["positive_symmetry_checks"])},
            "raw_replay": {
                "status": "pass" if replay_pass else "not_available" if not cases else "fail",
                "source_proof": [{"artifact": "evidence_dba_source_execution.json#predicate=" + predicate_id,
                                  "source_bundle": "sources/real_source_cases.json#predicate=" + predicate_id}] if cases else [],
                "source_kind": "stored_raw_replay_with_strict_L0_scope",
                "case_count": len(cases),
                "scope": "Stored records and saved evidence replay; this is historical evidence, never current-contract version compatibility.",
                "current_contract_compatibility_claim": "none",
                "training_interpretation": "historical_evidence_only_until_explicit_version_adjudication" if route == "source_version_adjudication"
                                           else "raw_replay_does_not_decide_source_version_compatibility"},
            "controlled_replay": {"status": "pending" if route != "natural" else "not_available",
                                  "source_proof": [], "case_count": 0,
                                  "reason": "actual_current_pipeline_execution_not_received" if route != "natural"
                                            else "natural_route_uses_actual_raw_replay"},
            "version_compatibility": {"status": "pending" if route == "source_version_adjudication" else "not_available",
                                      "compatibility_status": "unresolved", "source_proof": [],
                                      "reason": "explicit_source_version_adjudication_not_received" if route == "source_version_adjudication"
                                                else "no_version_adjudication_in_this_route"},
        }
        row["closure_independent_review"] = (deepcopy(row["independent_review"]) if route == "natural" else
            {"owner": "", "status": "pending", "conclusion": "actual_closure_not_reviewed", "evidence": []})
        row["gate_ready"] = False
    matrix["closure_gate"] = {
        "schema": "aeroworld.unified72.controlled-version-closure.v1",
        "scope": "current deterministic contract and typed binding closure; neural/inference work stays paused",
        "policy": "67 natural routes plus three actual controlled pipeline routes plus two explicit source-version/current-producer routes, each independently reviewed",
        "frozen_natural_counts": baseline, "frozen_source_metrics": source_metrics,
        "controlled_predicates": sorted(CONTROLLED_GATE_PREDICATES),
        "version_predicates": sorted(VERSION_GATE_PREDICATES),
        "natural_coverage_claim": "Natural coverage is unchanged; controlled fixture and version closure are separate evidence.",
    }
    refresh_closure_gate(matrix)


def record_closure_validation(row: dict, check: str, status: str, *, source_proof,
                              source_kind: str, **details) -> None:
    """Record executed proof; empty counts or unspecified compatibility cannot pass."""
    if check not in CLOSURE_CHECKS or status not in CHECK_STATUSES:
        raise ValueError(f"Invalid closure validation: {check}={status}")
    proof = list(source_proof)
    if status == "pass":
        if not proof or not source_kind:
            raise ValueError("Passing closure validation requires explicit source proof")
        if check in {"current_contract_deterministic", "raw_replay", "controlled_replay"}:
            if type(details["case_count"]) is not int or details["case_count"] <= 0:
                raise ValueError("Passing executed replay requires actual cases")
        if check == "alias_binding" and (type(details["output_instance_count"]) is not int
                                           or details["output_instance_count"] <= 0):
            raise ValueError("Passing binding validation requires actual outputs")
        if check == "alias_binding" and (not details["negative_checks"]
                or any(not mutation["rejected"] for mutation in details["negative_checks"])):
            raise ValueError("Passing binding validation requires rejected mutations of actual output bindings")
        if check == "controlled_replay" and (not details["pipeline_entrypoints"] or not details["producer_artifacts"]):
            raise ValueError("Controlled replay must identify its executed pipeline and producer artifacts")
        if check == "controlled_replay" and not details["executed_by"]:
            raise ValueError("Controlled replay must identify its execution owner")
        if check == "version_compatibility":
            if details["compatibility_status"] not in {"compatible", "incompatible", "version_specific"}:
                raise ValueError("Version compatibility must be explicitly adjudicated")
            if not details["current_contract_training_policy"]:
                raise ValueError("Version adjudication must state current-contract training use")
        if details["counterexamples"]:
            raise ValueError("Passing closure validation has unresolved counterexamples")
    row["closure_validation"][check] = {"status": status, "source_proof": proof,
                                         "source_kind": source_kind, **details}


def record_closure_review(row: dict, *, owner: str, status: str, conclusion: str, evidence) -> None:
    evidence = list(evidence)
    if status not in CHECK_STATUSES or (status == "pass" and (not owner or not conclusion or not evidence)):
        raise ValueError("Passing closure review needs an independent owner, conclusion and evidence")
    row["closure_independent_review"] = {"owner": owner, "status": status,
                                          "conclusion": conclusion, "evidence": evidence}


def refresh_closure_gate(matrix: dict) -> None:
    gate = matrix["closure_gate"]
    actual = {name: matrix["counts"][name] for name in gate["frozen_natural_counts"]}
    if actual != gate["frozen_natural_counts"]:
        raise ValueError("Controlled/version closure must not change the frozen natural coverage")
    requirements = {
        "natural": ("current_contract_deterministic", "alias_binding", "raw_replay"),
        "controlled_pipeline": ("current_contract_deterministic", "alias_binding", "controlled_replay"),
        "source_version_adjudication": CLOSURE_CHECKS,
    }
    for row in matrix["rows"]:
        route = row["closure_route"]
        expected_route = ("controlled_pipeline" if row["predicate_id"] in CONTROLLED_GATE_PREDICATES else
                          "source_version_adjudication" if row["predicate_id"] in VERSION_GATE_PREDICATES else "natural")
        if route != expected_route or set(row["closure_validation"]) != set(CLOSURE_CHECKS):
            raise ValueError("Closure route or declared checks differ from the authorized scope")
        for check in row["closure_validation"].values():
            if check["status"] not in CHECK_STATUSES or (check["status"] == "pass" and not check["source_proof"]):
                raise ValueError("Closure pass requires actual proof and a declared status")
        bindings = row["closure_validation"]["alias_binding"]
        if bindings["status"] == "pass" and (bindings["output_instance_count"] <= 0
                or not bindings["negative_checks"] or any(not item["rejected"] for item in bindings["negative_checks"])):
            raise ValueError("Closure binding pass requires actual outputs and successful binding mutations")
        controlled = row["closure_validation"]["controlled_replay"]
        if controlled["status"] == "pass" and (controlled["case_count"] <= 0
                or not controlled["pipeline_entrypoints"] or not controlled["producer_artifacts"] or not controlled["executed_by"]):
            raise ValueError("Controlled closure pass requires an executed pipeline with actual producer artifacts")
        version = row["closure_validation"]["version_compatibility"]
        if version["status"] == "pass" and (version["compatibility_status"] not in {"compatible", "incompatible", "version_specific"}
                or not version["current_contract_training_policy"]):
            raise ValueError("Version closure pass requires explicit compatibility and training adjudication")
        review = row["closure_independent_review"]
        if review["status"] not in CHECK_STATUSES:
            raise ValueError("Closure review has an undeclared status")
        if review["status"] == "pass" and (not review["owner"] or not review["conclusion"] or not review["evidence"]):
            raise ValueError("Closure review pass needs its independent reviewer, conclusion and proof")
        if route != "natural" and review["status"] == controlled["status"] == "pass" and review["owner"] == controlled["executed_by"]:
            raise ValueError("Controlled fixture executor cannot be its independent closure reviewer")
        reviewed = (review["status"] == "pass" and bool(review["owner"]) and bool(review["evidence"]))
        row["gate_ready"] = reviewed and all(row["closure_validation"][name]["status"] == "pass"
                                            for name in requirements[route])
        row["gate_required_checks"] = list(requirements[route])
    gate["counts"] = {name: sum(row["closure_validation"][name]["status"] == "pass" for row in matrix["rows"])
                       for name in CLOSURE_CHECKS}
    gate["counts"]["independently_reviewed"] = sum(row["closure_independent_review"]["status"] == "pass"
                                                    for row in matrix["rows"])
    gate["counts"]["gate_ready"] = sum(row["gate_ready"] for row in matrix["rows"])
    gate["counts"]["controlled_binding_negative_checks"] = sum(
        len(row["closure_validation"]["alias_binding"]["negative_checks"])
        for row in matrix["rows"] if row["closure_route"] != "natural"
        and row["closure_validation"]["alias_binding"]["source_kind"] == "actual_controlled_pipeline_binding")
    gate["ready_by_route"] = {route: sum(row["gate_ready"] and row["closure_route"] == route for row in matrix["rows"])
                              for route in requirements}
    gate["ready"] = gate["counts"]["gate_ready"] == matrix["required_count"]
    matrix["gate_ready"] = gate["ready"]


def write_closure_summary(matrix: dict, output: Path = RUN_ROOT) -> None:
    """Keep the controlled gate report separate from full functional acceptance."""
    refresh_closure_gate(matrix)
    gate = matrix["closure_gate"]
    summary = {
        "schema": "aeroworld.unified72.controlled-version-gate-summary.v1",
        "scope": gate["scope"], "gate_ready": gate["ready"], "counts": gate["counts"],
        "ready_by_route": gate["ready_by_route"],
        "natural_counts_unchanged": gate["frozen_natural_counts"],
        "source_metrics_unchanged": gate["frozen_source_metrics"],
        "matrix_ref": "deterministic_binding_alias_matrix.json",
        "closure_rows": [{
            "predicate_id": row["predicate_id"], "route": row["closure_route"],
            "gate_ready": row["gate_ready"],
            "checks": {name: check["status"] for name, check in row["closure_validation"].items()},
            "version_adjudication": row["closure_validation"]["version_compatibility"],
            "provenance": row["closure_provenance"],
            "independent_review": row["closure_independent_review"],
            "source_proof": {name: check["source_proof"] for name, check in row["closure_validation"].items()},
        } for row in matrix["rows"] if row["closure_route"] != "natural"],
        "future_work": "Neural, inference, performance, cleanup and commit remain paused; this gate does not execute or accept those stages.",
    }
    (output / "controlled_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true", help="Write the fixed pending matrix")
    parser.add_argument("--output", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    if not args.initialize:
        parser.error("Select --initialize or run the functional acceptance API")
    matrix = initial_matrix()
    write_matrix(matrix, args.output)
    print(json.dumps({"counts": matrix["counts"], "operators": matrix["operator_node_counts"]}))


if __name__ == "__main__":
    main()
