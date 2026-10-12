"""Read immutable predicate evidence and construct prefix-only prediction windows.

Published graph assertions provide recorded field observations and typed bindings.
They are replayed against the current executable contract without relabelling the
published future labels. A current/source disagreement remains explicit evidence.
"""
from __future__ import annotations

from Dataset.semantic_truth.provenance import open_artifact_text
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl

import pyarrow.dataset as ds

from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates
from Dataset.semantic_truth import world_truth as wt

REPO_ROOT = Path(__file__).resolve().parents[3]
GRAPH_ROOT = Path("Dataset/world_model/runs/belief_label_graph/formal")
TRUTH_VALUES = frozenset(("true", "false", "unknown", "out_of_scope"))
BINARY_VALUES = frozenset(("true", "false"))
TRUTH_SEMANTICS = {
    "case_current_truth": "Current executable evaluator on the case context. Raw-verified cases use stored raw source records with strict L0 scope reconstruction; this does not regenerate every physical producer.",
    "case_saved_truth": "Original published label, preserved without relabelling.",
    "current_published_disagreement_predicates": "Predicates whose recorded case context evaluates differently from the saved published label. A zero count does not prove that every current producer matches its stored version.",
    "raw_current_vs_saved_ambiguity_predicates": "Predicates with field-mapping or producer-version evidence differences, including differences that leave the selected stored-raw truth unchanged.",
    "fresh_current_producer_probes": "Separately executed current PlanWindowComputer probes retain their own contexts, actual scope activity and truths; they do not replace case.context or saved_truth.",
}


def _set_path(target: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        if not isinstance(current[part], dict):
            raise ValueError(f"Recorded observation paths collide: {path}")
        current = current[part]
    leaf = parts[-1]
    if leaf in current and current[leaf] != value:
        raise ValueError(f"Recorded observations disagree: {path}")
    current[leaf] = deepcopy(value)


def _get_path(context: Mapping[str, Any], path: str) -> Any:
    current: Any = context
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return wt.MISSING
        current = current[part]
    return current


def _binding_key(bindings: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((role, value["entity_id"]) for role, value in bindings.items()))


def _evaluate(template: dict, context: dict, previous: dict, parameters: dict) -> str:
    # These recorded applicability markers precede field availability in L1.
    if _get_path(context, "l0.scope_active") is False:
        return "out_of_scope"
    unavailable = _get_path(context, "l0.predicate_source_unavailability")
    if unavailable is not wt.MISSING:
        return "out_of_scope"
    for field in template["state_contract"]["required_fields"]:
        if wt._missing_value(_get_path(context, field["field"])):
            return "unknown"
    return wt._expression_truth_value(wt._evaluate_expression(
        template["evaluation_spec"], context, previous, parameters,
    ))


def evaluate_versioned_source_case(case: Mapping[str, Any], record: Mapping[str, Any], *,
                                  source_ref: str, scope_active: bool,
                                  scope_source_ref: str) -> dict:
    """Resolve producer semantics before indexing or evaluating a source record.

    Current YAML supplies field, unit, binding and expression authority. This
    isolated entry point preserves the original context and label and attaches a
    separate versioned context/truth only after declared source authority passes.
    Historical records lacking semantic binding authority remain ineligible for
    current training; their stored values are never converted or relabelled.
    """
    from .source_versions import (
        SourceVersionError, UndeclaredSourceSemantics, resolve_source_version,
    )
    result = deepcopy(dict(case))
    result["versioned_source_ref"] = source_ref
    result["eligible_for_current_training"] = False
    result["source_authority_eligible"] = False
    result["saved_label_eligible_for_current_training"] = False
    result["saved_label_current_compatibility"] = "not_established_across_source_versions; preserved_evidence_only"
    result["current_binary_supervision_available"] = False
    result["current_training_target_field"] = "versioned_current_truth"
    try:
        if type(scope_active) is not bool or not isinstance(scope_source_ref, str) or not scope_source_ref:
            raise SourceVersionError("missing_exact_scope_authority", "scope activity and its source reference required")
        scope_path, separator, fragment = scope_source_ref.partition("#")
        scope_file = Path(scope_path)
        scope_pairs = parse_qsl(fragment, keep_blank_values=True)
        scope_selector = dict(scope_pairs)
        normative = next(item for item in get_core_predicate_templates() if item["id"] == case["predicate_id"])
        aircraft_roles = [role["key"] for role in normative["argument_roles"] if role["class"] == "UnmannedAircraft"]
        if (not separator or scope_file.is_absolute() or ".." in scope_file.parts
                or scope_file.name != "truth_frames.jsonl" or scope_file.parent.name != case["episode_id"]
                or len(scope_pairs) != len(scope_selector) or set(scope_selector) != {"tick", "entity"}
                or len(aircraft_roles) != 1 or scope_selector["tick"] != str(case["tick"])
                or scope_selector["entity"] != case["bindings"][aircraft_roles[0]]["entity_id"]):
            raise SourceVersionError("scope_source_binding_mismatch", scope_source_ref)
        resolved = resolve_source_version(record, predicate_id=case["predicate_id"],
            episode_id=case["episode_id"], tick=case["tick"], bindings=case["bindings"], source_ref=source_ref)
        template = next(item for item in get_core_predicate_templates() if item["id"] == case["predicate_id"])
        if not all(key in case["parameters"] for key in template["state_contract"]["threshold_parameters"]):
            raise SourceVersionError("missing_normative_threshold", case["predicate_id"])
        context = deepcopy(dict(resolved.context))
        context["l0"] = {"scope_active": scope_active}
        current_truth = _evaluate(template, context, {}, case["parameters"])
        result.update({"source_version_status": resolved.status,
            "producer_identity": dict(zip(("schema_name", "schema_version", "model_id", "model_version", "rule_id", "rule_version", "observation_family"), resolved.producer_identity)),
            "resolved_source_semantic_contract": deepcopy(resolved.semantic_contract),
            "versioned_context": context,
            "versioned_current_truth": current_truth,
            "versioned_scope_source_ref": scope_source_ref,
            "eligible_for_current_training": True,
            "source_authority_eligible": True,
            "current_binary_supervision_available": current_truth in BINARY_VALUES,
            "label_training_policy": "only_versioned_current_truth; saved_truth_remains_historical_evidence",
            "normative_authority": "Dataset/semantic_rules/predicates/world_truth_predicate_contracts.yaml",
            "source_version_error": None})
    except SourceVersionError as error:
        status = ("legacy_semantics_unresolved" if error.code == "undeclared_legacy_semantic_source_binding"
                  else "rejected_for_current_training")
        result.update({"source_version_status": status,
            "source_version_error": {"code": error.code, "detail": str(error)},
            "label_policy": "preserve_original_label_and_raw_context"})
    return result


def ground_versioned_source_case(case: Mapping[str, Any], record: Mapping[str, Any], *,
                                source_ref: str, scope_active: bool,
                                scope_source_ref: str, grounding_options: Mapping[str, Any]):
    """Enforce source authority before creating any typed graph variable."""
    from .generic_grounding import ground_contract
    from .source_versions import SourceVersionError
    from .unified_contracts import load_unified_contracts

    adjudicated = evaluate_versioned_source_case(case, record, source_ref=source_ref,
        scope_active=scope_active, scope_source_ref=scope_source_ref)
    if adjudicated["source_authority_eligible"] is not True:
        error = adjudicated["source_version_error"]
        raise SourceVersionError(error["code"], error["detail"])
    if "predicate_evidence" in grounding_options:
        raise SourceVersionError("external_predicate_label_not_allowed",
            "versioned physical-source grounding cannot consume an externally supplied predicate label")
    if "branch_id" in grounding_options:
        raise SourceVersionError("duplicate_source_branch_authority", "branch is declared by the source case")
    contract = load_unified_contracts()[case["predicate_id"]]
    rule = ground_contract(contract, case["bindings"], adjudicated["versioned_context"],
        case["parameters"], tick=case["tick"], episode_id=case["episode_id"], branch_id=case["branch_id"],
        source_refs=(source_ref, scope_source_ref), applicable=scope_active, **dict(grounding_options))
    return adjudicated, rule


def read_episode_cases(
    episode_id: str,
    *,
    repo_root: Path = REPO_ROOT,
    start_tick: int = 0,
    end_tick: int = 100,
    predicate_ids: Iterable[str] | None = None,
) -> list[dict]:
    """Read original graph assertions as recorded evidence without relabelling.

    This reader does not establish current training eligibility. Versioned source
    gates must use ground_versioned_source_case before admitting a typed graph.
    """
    repo_root = Path(repo_root)
    templates = {item["id"]: item for item in get_core_predicate_templates()
                 if item["implementation_status"] == "executable_l1"}
    selected = set(templates) if predicate_ids is None else set(predicate_ids)
    unknown_ids = selected - set(templates)
    if unknown_ids:
        raise ValueError(f"Non-executable or unknown predicate IDs: {sorted(unknown_ids)}")
    if start_tick < 0 or end_tick < start_tick or start_tick % 5 or end_tick % 5:
        raise ValueError("Replay bounds must be nonnegative formal five-tick samples")
    directory = repo_root / GRAPH_ROOT / episode_id
    scope_rows = ds.dataset(directory / "scope.parquet").to_table().to_pylist()
    if len(scope_rows) != 1:
        raise ValueError(f"Expected one graph scope row: {directory}")
    scope = scope_rows[0]
    if scope["episode_id"] != episode_id or scope["source_kind"] != "formal":
        raise ValueError(f"Formal graph scope differs: {directory}")
    summary_path = Path("aw_data/objective_semantic_truth") / episode_id / "world_truth_summary.json"
    summary = json.loads((repo_root / summary_path).read_text())
    if summary["episode_id"] != episode_id:
        raise ValueError(f"World-truth summary episode differs: {summary_path}")
    parameters = summary["governed_parameters"]
    if not isinstance(parameters, dict):
        raise ValueError(f"Missing recorded parameter profile: {summary_path}")
    required_parameters = {parameter for predicate in selected
                           for parameter in templates[predicate]["state_contract"]["threshold_parameters"]}
    if not required_parameters <= set(parameters):
        raise ValueError(f"Recorded profile misses governed thresholds: {sorted(required_parameters - set(parameters))}")
    parameter_profile = {"profile_id": f"formal:{episode_id}:governed_parameters",
                         "source_ref": str(summary_path) + "#governed_parameters",
                         "parameters": deepcopy(parameters)}
    expression = ((ds.field("node_type") == "PredicateAssertion")
                  & (ds.field("tick") >= start_tick) & (ds.field("tick") <= end_tick))
    table = ds.dataset(directory / "nodes.parquet").to_table(
        columns=["node_id", "tick", "attrs_json", "source_file"], filter=expression,
    )
    cases = []
    identities = set()
    for row in table.to_pylist():
        attrs = json.loads(row["attrs_json"])
        if attrs.get("assertion_layer") is not None:
            continue
        predicate = attrs["predicate_id"]
        if predicate not in selected:
            continue
        template = templates[predicate]
        bindings = attrs["bindings"]
        classes = attrs["binding_ontology_classes"]
        roles = {role["key"] for role in template["argument_roles"]}
        if set(bindings) != roles or set(classes) != roles:
            raise ValueError(f"Saved typed roles differ: {row['node_id']}")
        typed = {role: {"entity_id": bindings[role], "ontology_class": classes[role]}
                 for role in bindings}
        for role in template["argument_roles"]:
            expected = wt._world_class(role["class"])
            if not wt.ontology_class_is_a(classes[role["key"]], expected):
                raise ValueError(f"Saved role class differs: {row['node_id']}:{role['key']}")
        wt._validate_binding_constraints(predicate, bindings, template["grounding_spec"])
        truth = attrs["truth_status"]
        if truth not in TRUTH_VALUES or attrs["candidate_present"] is not True:
            raise ValueError(f"Invalid saved assertion: {row['node_id']}")
        identity = (predicate, int(row["tick"]), _binding_key(typed))
        if identity in identities:
            raise ValueError(f"Duplicate original graph assertion: {row['node_id']}")
        identities.add(identity)
        context, field_refs = {}, {}
        for evidence in attrs["evidence"]:
            _set_path(context, evidence["path"], evidence["value"])
            if "source_ref" in evidence:
                field_refs[evidence["path"]] = evidence["source_ref"]
        required = [item["field"] for item in template["state_contract"]["required_fields"]]
        missing = [field for field in required if wt._missing_value(_get_path(context, field))]
        graph_ref = str(directory.relative_to(repo_root) / "nodes.parquet") + "#node_id=" + row["node_id"]
        cases.append({
            "predicate_id": predicate, "episode_id": episode_id, "branch_id": "factual",
            "tick": int(row["tick"]), "bindings": typed, "context": context,
            "previous_context": {}, "parameters": deepcopy(parameters),
            "parameter_profile": deepcopy(parameter_profile),
            "parameter_profile_id": parameter_profile["profile_id"],
            "output_binding_authority": {
                "predicate_id": predicate, "bindings": deepcopy(typed), "tick": int(row["tick"]),
                "episode_id": episode_id, "branch_id": "factual",
                "parameter_profile_id": parameter_profile["profile_id"],
                "parameters": deepcopy(parameters), "source_ref": graph_ref, "source_refs": (graph_ref,),
            },
            "saved_truth": truth, "current_truth": None,
            "observed_fields": [field for field in required if field not in missing],
            "missing_fields": missing, "field_source_refs": field_refs,
            "source_refs": [graph_ref, *attrs["source_refs"]], "graph_node_ref": graph_ref,
            "graph_source_file": row["source_file"],
            "candidate_authority": deepcopy(attrs["candidate_authority"]),
            "semantic_mode": "direct_observation" if template["evaluation_spec"]["op"] == "direct_truth" else "physical_rule",
            "physical_formula": "not_declared" if template["evaluation_spec"]["op"] == "direct_truth" else "declared",
            "semantic_ambiguities": [],
            "source_kind": "published_recorded_field_evidence",
            "source_window_end_tick": end_tick,
        })
    cases.sort(key=lambda case: (case["predicate_id"], _binding_key(case["bindings"]), case["tick"]))
    prior = {}
    for case in cases:
        identity = (case["predicate_id"], _binding_key(case["bindings"]))
        previous = prior.get(identity)
        if previous is not None and previous["tick"] + 5 == case["tick"]:
            case["previous_context"] = deepcopy(previous["context"])
        case["current_truth"] = _evaluate(templates[case["predicate_id"]], case["context"], case["previous_context"], parameters)
        if case["current_truth"] != case["saved_truth"]:
            case["semantic_ambiguities"].append({
                "kind": "current_contract_vs_published_truth", "saved_truth": case["saved_truth"],
                "current_truth": case["current_truth"], "label_policy": "preserve_published_label",
            })
        prior[identity] = case
    return cases


def prediction_windows(cases: Iterable[dict], *, horizon_steps: int = 10, step_ticks: int = 5) -> list[dict]:
    """Labels and censoring come from future rows; model inputs contain prefix only."""
    if horizon_steps != 10 or step_ticks != 5:
        raise ValueError("The executable event contract is ten half-second steps")
    groups = defaultdict(dict)
    for case in cases:
        identity = (case["predicate_id"], case["episode_id"], case["branch_id"], _binding_key(case["bindings"]))
        if case["tick"] in groups[identity]:
            raise ValueError(f"Duplicate window assertion: {identity}:{case['tick']}")
        groups[identity][case["tick"]] = case
    windows = []
    for identity, by_tick in groups.items():
        ticks = sorted(by_tick)
        for anchor_tick in ticks:
            anchor = by_tick[anchor_tick]
            if anchor["saved_truth"] not in BINARY_VALUES:
                continue  # Unknown anchors do not define a first-opposite target.
            # The terminal read bound and a candidate's last observation differ.
            # A short lived tuple is censored, never extended by interpolation.
            if anchor_tick >= anchor["source_window_end_tick"]:
                continue
            prefix = tuple(by_tick[tick] for tick in ticks if tick <= anchor_tick)
            future_ticks = tuple(anchor_tick + index * step_ticks for index in range(1, horizon_steps + 1))
            labels = tuple(by_tick[tick]["saved_truth"] if tick in by_tick else None for tick in future_ticks)
            if not labels or labels[0] not in BINARY_VALUES:
                continue  # No observed future risk interval is a zero-risk row.
            first_opposite, censor_step = None, None
            for index, label in enumerate(labels, 1):
                if label not in BINARY_VALUES:
                    censor_step = index
                    break
                if label != anchor["saved_truth"]:
                    first_opposite = index
                    break
            if first_opposite is not None:
                label_status, event_bin = "observed_event", first_opposite - 1
            elif censor_step is not None:
                label_status, event_bin = "right_censored", None
            else:
                label_status, event_bin = "observed_no_event", horizon_steps
            future_cases = [by_tick[tick] for tick in future_ticks if tick in by_tick]
            ambiguities = [issue for case in [anchor, *future_cases] for issue in case["semantic_ambiguities"]]
            windows.append({
                "predicate_id": identity[0], "episode_id": identity[1], "branch_id": identity[2],
                "bindings": deepcopy(anchor["bindings"]), "anchor_tick": anchor_tick,
                "anchor_case": anchor, "prefix_cases": prefix,
                "model_inputs": {"prefix_cases": prefix, "anchor_tick": anchor_tick},
                "future_ticks": future_ticks, "future_labels": labels,
                "future_label_source_refs": tuple(by_tick[tick]["graph_node_ref"] if tick in by_tick else None for tick in future_ticks),
                "real_future_row_count": len(future_cases), "horizon_steps": horizon_steps,
                "step_ticks": step_ticks, "label_status": label_status, "event_bin": event_bin,
                "censor_step": censor_step, "risk_steps_observed": first_opposite if first_opposite is not None else (horizon_steps if censor_step is None else censor_step - 1),
                "censor_reason": (None if censor_step is None else
                    "source_read_bound" if future_ticks[censor_step - 1] > anchor["source_window_end_tick"] else
                    "candidate_absent" if labels[censor_step - 1] is None else labels[censor_step - 1]),
                "semantic_ambiguities": ambiguities, "label_policy": "published_evaluation_only",
            })
    return windows


def _representative_episodes(repo_root: Path) -> list[str]:
    directories = sorted(path.name for path in (repo_root / GRAPH_ROOT).iterdir() if path.is_dir())
    first = "L1-1_v1__seed00"
    if first not in directories:
        raise FileNotFoundError(repo_root / GRAPH_ROOT / first)
    # These are source selection priorities, never graph ownership or model inputs.
    priority = [first, "L1-4_v1__seed00", "L2-1_v1__seed00", "L3-2_v1__seed00", "L4-2_v2__seed01",
                "L6-3_v1__seed00", "X6_crowd_evacuation_to_airspace_lockdown__seed00",
                "L2-3_v1__seed00", "L4-10_v1__seed00", "L6-5_v1__seed00"]
    present = [episode for episode in priority if episode in directories]
    remaining = [episode for episode in directories if episode not in present]
    return present + sorted(remaining, key=lambda episode: (not episode.endswith("seed00"), episode))


def load_real_source_bundle(
    repo_root: Path = REPO_ROOT,
    *,
    episodes: Iterable[str] | None = None,
    max_episodes: int = 12,
    start_tick: int = 0,
    end_tick: int = 100,
    cases_per_status: int = 1,
    windows_per_predicate: int = 2,
    deterministic_only: bool = True,
) -> dict:
    """Return a fixed 72-row availability ledger with compact real examples.

    Counts describe rows/windows actually read, not estimates for the corpus.
    Uncovered predicates remain explicit and never receive synthetic rows.
    """
    repo_root = Path(repo_root)
    templates = {item["id"]: item for item in get_core_predicate_templates()
                 if item["implementation_status"] == "executable_l1"}
    if len(templates) != 72:
        raise ValueError("Executable source inventory must contain exactly 72 predicates")
    if max_episodes < 1 or cases_per_status < 1 or windows_per_predicate < 1:
        raise ValueError("Source budgets must be positive")
    sequence = _representative_episodes(repo_root) if episodes is None else list(episodes)
    cases_by_predicate = {predicate: [] for predicate in templates}
    windows_by_predicate = {predicate: [] for predicate in templates}
    counts = {predicate: Counter() for predicate in templates}
    status_samples = {predicate: Counter() for predicate in templates}
    read_episodes, coupled = [], []
    for episode_id in sequence[:max_episodes]:
        missing = {predicate for predicate in templates
                   if (not cases_by_predicate[predicate] if deterministic_only else not windows_by_predicate[predicate])}
        selected = set(templates) if not read_episodes else missing
        if not selected:
            break
        cases = read_episode_cases(episode_id, repo_root=repo_root, start_tick=start_tick, end_tick=end_tick, predicate_ids=selected)
        read_episodes.append({"episode_id": episode_id, "start_tick": start_tick, "end_tick": end_tick, "real_rows_read": len(cases)})
        for case in cases:
            predicate = case["predicate_id"]
            counts[predicate]["real_replay_case_count"] += 1
            counts[predicate]["real_binary_case_count"] += case["saved_truth"] in BINARY_VALUES
            counts[predicate]["saved_current_disagreement_count"] += bool(case["semantic_ambiguities"])
            counts[predicate]["missing_required_field_case_count"] += bool(case["missing_fields"])
            counts[predicate]["status_" + case["saved_truth"]] += 1
            if status_samples[predicate][case["saved_truth"]] < cases_per_status:
                cases_by_predicate[predicate].append(case)
                status_samples[predicate][case["saved_truth"]] += 1
        windows = [] if deterministic_only else prediction_windows(cases)
        for window in windows:
            predicate = window["predicate_id"]
            counts[predicate]["real_prediction_window_count"] += 1
            counts[predicate]["uncensored_prediction_window_count"] += window["label_status"] != "right_censored"
            counts[predicate]["real_event_window_count"] += window["label_status"] == "observed_event"
            counts[predicate]["semantic_ambiguity_window_count"] += bool(window["semantic_ambiguities"])
        for predicate in selected:
            candidates = [window for window in windows if window["predicate_id"] == predicate]
            # One observed history before the anchor is preferred; one-step
            # prefixes remain executable and are explicitly counted as such.
            candidates.sort(key=lambda window: (len(window["prefix_cases"]) < 2,
                            window["label_status"] == "right_censored", window["anchor_tick"], _binding_key(window["bindings"])))
            windows_by_predicate[predicate].extend(candidates[:windows_per_predicate])
        if not coupled:
            by_tick = defaultdict(list)
            for case in cases:
                if case["saved_truth"] in BINARY_VALUES and len(case["bindings"]) > 1:
                    by_tick[case["tick"]].append(case)
            for tick, rows in sorted(by_tick.items()):
                corridor = [case for case in rows if case["candidate_authority"]["source"] == "predicate_contract_aircraft_corridor_geometry"]
                owners = Counter(entity["entity_id"] for case in corridor for entity in case["bindings"].values())
                shared = {owner for owner, count in owners.items() if count >= 2}
                if shared:
                    initial = [case for case in corridor if any(value["entity_id"] in shared for value in case["bindings"].values())]
                    entity_ids = {value["entity_id"] for case in initial for value in case["bindings"].values()}
                    other = [case for case in rows if case not in initial and any(value["entity_id"] in entity_ids for value in case["bindings"].values())]
                    coupled = initial[:9] + other[:3]
                    break
    coverage = {}
    for predicate, template in templates.items():
        row = dict(counts[predicate])
        for key in ("real_replay_case_count", "real_binary_case_count", "real_prediction_window_count", "uncensored_prediction_window_count", "real_event_window_count", "saved_current_disagreement_count", "missing_required_field_case_count", "semantic_ambiguity_window_count"):
            if key not in row:
                row[key] = 0
        row["availability"] = "available" if row["real_replay_case_count"] else "not_available"
        row["prediction_availability"] = "not_evaluated" if deterministic_only else ("available" if row["real_prediction_window_count"] else "not_available")
        row["source_kind"] = "published_recorded_field_evidence"
        row["semantic_mode"] = "direct_observation" if template["evaluation_spec"]["op"] == "direct_truth" else "physical_rule"
        row["physical_formula"] = "not_declared" if row["semantic_mode"] == "direct_observation" else "declared"
        row["label_policy"] = "preserve_published_label"
        row["gaps"] = []
        if not row["real_replay_case_count"]:
            row["gaps"].append("no_real_candidate_in_read_episodes")
        elif not deterministic_only and not row["real_prediction_window_count"]:
            row["gaps"].append("no_binary_anchor_with_recorded_five_second_future")
        if row["saved_current_disagreement_count"]:
            row["gaps"].append("current_contract_differs_from_published_semantics")
        coverage[predicate] = row
    return {"schema": "aeroworld.unified.real_sources.v1", "truth_semantics": deepcopy(TRUTH_SEMANTICS),
            "cases_by_predicate": cases_by_predicate,
            "windows_by_predicate": windows_by_predicate, "coverage": coverage,
            "episodes": read_episodes, "coupled_cases": coupled,
            "summary": {"executable_predicates": 72, "real_replay_predicates": sum(bool(row["real_replay_case_count"]) for row in coverage.values()),
                        "real_prediction_predicates": sum(bool(row["real_prediction_window_count"]) for row in coverage.values()),
                        "current_published_disagreement_predicates": sum(bool(row["saved_current_disagreement_count"]) for row in coverage.values()),
                        "future_inputs": "prefix_only; future labels are evaluation-only"}}


def formal_source_availability(repo_root: Path = REPO_ROOT) -> dict:
    """Aggregate existing formal summaries; this is data availability evidence."""
    templates = {item["id"]: item for item in get_core_predicate_templates()
                 if item["implementation_status"] == "executable_l1"}
    truth_counts = {predicate: Counter() for predicate in templates}
    episodes = {predicate: [] for predicate in templates}
    paths = sorted((Path(repo_root) / "aw_data/objective_semantic_truth").glob("*/world_truth_summary.json"))
    for path in paths:
        summary = json.loads(path.read_text())
        if summary["episode_id"] != path.parent.name:
            raise ValueError(f"Summary identity differs: {path}")
        for predicate in templates:
            counts = summary["value_counts_by_predicate"].get(predicate, {})
            truth_counts[predicate].update(counts)
            if counts:
                episodes[predicate].append({"episode_id": path.parent.name, "truth_counts": counts,
                    "source_ref": str(path.relative_to(repo_root)) + "#value_counts_by_predicate/" + predicate})
    return {"formal_episode_summary_count": len(paths), "predicates": {
        predicate: {"truth_counts": dict(truth_counts[predicate]), "episodes_with_rows": len(episodes[predicate]),
                    "real_binary_count": truth_counts[predicate]["true"] + truth_counts[predicate]["false"],
                    "episodes": episodes[predicate]}
        for predicate in templates}}


_TICK_TOKEN = re.compile(r'"tick"\s*:\s*([0-9]+)(?=\s*[,}])')


def _selected_jsonl(path: Path, ticks: set[int], repo_root: Path) -> list[dict]:
    rows = []
    with open_artifact_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not any(int(match.group(1)) in ticks for match in _TICK_TOKEN.finditer(line)):
                continue
            row = json.loads(line)
            if row.get("tick") in ticks:
                row = deepcopy(row)
                row["_unified_source_ref"] = str(path.relative_to(repo_root)) + f":{line_number}"
                rows.append(row)
    return rows


def _canonical_instance_catalog(*, scopes, frames, domain_rows, direct_rows,
                                utm_records, compute_rows, actual) -> tuple[dict, dict, list]:
    """Collect actual type assertions without treating a component host as identity.

    world_truth's separate lookup index remains the input to its official source
    adapters. Its source_entity_id associations do not enter this identity catalog;
    complete output signatures use this catalog to validate every role class.
    """
    classes, refs = defaultdict(set), defaultdict(set)
    for scope in scopes:
        ontology_class = wt._scope_ontology_class(scope)
        if ontology_class:
            classes[scope.entity_id].add(ontology_class)
            refs[scope.entity_id].update(actual["input_source_refs"])
            refs[scope.entity_id].add(actual["replay_ref"])
    for tick, entities in frames.items():
        discovered = defaultdict(set)
        for entity in entities.values():
            wt._collect_explicit_identifier_classes(entity, discovered)
        for entity_id, tags in discovered.items():
            classes[entity_id].update(tags)
            refs[entity_id].add(actual["frame_refs"][tick])
            refs[entity_id].update(actual["input_source_refs"])
    for record in [*domain_rows, *direct_rows, *(row for rows in utm_records.values() for row in rows)]:
        discovered = defaultdict(set)
        wt._collect_explicit_identifier_classes(record, discovered)
        for entity_id, tags in discovered.items():
            classes[entity_id].update(tags)
            refs[entity_id].add(record["_unified_source_ref"])
    component_relations = {}
    scope_classes = {scope.entity_id: wt._scope_ontology_class(scope) for scope in scopes}
    for record in compute_rows:
        node, host = record["node_id"], record["entity_id"]
        refs[node].add(record["_unified_source_ref"])
        refs[node].add("Dataset/semantic_truth/world_truth.py#_scope_ontology_class")
        if node != host:
            key = (node, host)
            relation = component_relations.setdefault(key, {
                "component_entity_id": node, "source_entity_id": host,
                "component_class": scope_classes[node], "source_refs": [],
                "class_source_ref": "Dataset/semantic_truth/world_truth.py#_scope_ontology_class",
                "source_fields": {"component_entity_id": "node_id", "source_entity_id": "entity_id"},
                "relation_kind": "component_host", "identity_assertion": False,
            })
            relation["source_refs"].append(record["_unified_source_ref"])
    return classes, refs, [component_relations[key] for key in sorted(component_relations)]


def _actual_objective_scope_inputs(episode: str, ticks: set[int], repo_root: Path,
                                   plan_probe_entities: set[str] | None = None) -> dict:
    """Replay the declared strict L0 adapter in its isolated preparation tree."""
    from Dataset.semantic_truth.objective_pipeline import _objective_strict_episode_root
    from Dataset.semantic_truth.episode_sources import source_episode_root, source_sumo_frames

    capture = repo_root / "aw_data/render_ready_episodes_capture_filtered" / episode
    refs = [str((capture / name).relative_to(repo_root))
            for name in ("episode_manifest.json", "global_entity_roster.json", "truth_frames.jsonl", "trajectories.jsonl", "weather_meta.jsonl")]
    scenario = source_episode_root(capture)
    if scenario is not None:
        refs += [str((scenario / name).relative_to(repo_root)) for name in ("global_entity_roster.json", "trajectories.jsonl")]
    refs.append(str(source_sumo_frames(capture).relative_to(repo_root)))
    replay_ref = "Dataset/semantic_truth/objective_pipeline.py#_objective_strict_episode_root;episode=" + episode
    plan_probes = []
    with _objective_strict_episode_root(capture) as strict:
        manifest = json.loads((strict / "episode_manifest.json").read_text())
        roster = json.loads((strict / "global_entity_roster.json").read_text())
        frame_rows = _selected_jsonl(strict / "truth_frames.jsonl", ticks, repo_root)
        weather_rows = _selected_jsonl(strict / "weather_meta.jsonl", ticks, repo_root)
        frame_refs = {row["tick"]: replay_ref + f"#tick={row['tick']}" for row in frame_rows}
        frames = {row["tick"]: {entity["entity_id"]: entity for entity in row["entities"]} for row in frame_rows}
        weather = {row["tick"]: {name: value for name, value in row.items() if name != "_unified_source_ref"} for row in weather_rows}
        summary = json.loads((strict / "l0_state_materialization_summary.json").read_text())
        measured = {name: summary[name] for name in ("active_uav_tick_count", "active_vehicle_tick_count", "road_crossing_count", "regulation_field_count", "trajectory_entity_tick_count")}
        if plan_probe_entities:
            from Dataset.semantic_simulation.predicate_state_computers import GeometricStateComputer, PlanWindowComputer
            geometry = GeometricStateComputer(strict).compute()
            for record in PlanWindowComputer(strict, geometry).compute():
                if (record["observation_family"] == "predicate_contract_aircraft_plan"
                        and record["subject_id"] in plan_probe_entities and record["tick"] in {0, 20}):
                    plan_probes.append({"subject_id": record["subject_id"], "tick": record["tick"],
                        "observation_family": record["observation_family"],
                        "model_id": record["model_id"], "model_version": record["model_version"],
                        "rule_id": record["rule_id"], "rule_version": record["rule_version"],
                        "context": {"plan": {"cooperative_flag": record["values"]["cooperative_flag"]},
                                    "l0": {"scope_active": record["subject_id"] in frames[record["tick"]]}},
                        "scope_source_ref": frame_refs[record["tick"]] + "&entity=" + record["subject_id"],
                        "source_ref": replay_ref + ";producer=Dataset.semantic_simulation.predicate_state_computers.PlanWindowComputer.compute",
                        "input_source_refs": refs})
    return {"manifest": manifest, "roster": roster, "frames": frames, "weather": weather,
            "input_source_refs": refs, "replay_ref": replay_ref, "frame_refs": frame_refs,
            "selected_ticks": sorted(ticks), "l0_materialization_measured": measured,
            "current_plan_producer_probes": plan_probes}


def replay_raw_source_cases(cases: Iterable[dict], *, repo_root: Path = REPO_ROOT,
                            class_catalogs: dict | None = None,
                            population_audits: dict | None = None) -> list[dict]:
    """Reground selected recorded cases using the current source-field contract.

    This reads immutable raw records and reuses the authoritative candidate and
    context adapters. Recorded labels stay unchanged. Errors are attached to the
    affected cases so independent rows can still be audited; no error is a pass.
    """
    repo_root = Path(repo_root)
    templates = {item["id"]: item for item in get_core_predicate_templates()}
    grouped = defaultdict(list)
    for original in cases:
        replay = deepcopy(original)
        if "published_context" in replay:
            replay["context"] = deepcopy(replay["published_context"])
        grouped[original["episode_id"]].append(replay)
    result = []
    for episode, selected in grouped.items():
        ticks = {case["tick"] for case in selected}
        plan_probe_entities = set()
        if episode == "L1-1_v1__seed00":
            plan_probe_entities = {case["bindings"]["aircraft"]["entity_id"] for case in selected
                if case["predicate_id"] == "airspace.aircraft_is_non_cooperative"}
            plan_probe_entities = set(sorted(plan_probe_entities)[:2])
            if plan_probe_entities:
                ticks.update((0, 20))
        source = repo_root / "aw_data/objective_semantic_truth" / episode
        capture = repo_root / "aw_data/render_ready_episodes_capture_filtered" / episode
        roster_path = capture / "global_entity_roster.json"
        domain_rows = _selected_jsonl(source / "domain_state_observations.jsonl", ticks, repo_root)
        domain_rows += _selected_jsonl(source / "l0_predicate_state.jsonl", ticks, repo_root)
        compute_rows = _selected_jsonl(source / "compute_state.jsonl", ticks, repo_root)
        # The objective producer's full-world input snapshot includes all compute
        # nodes; the focused supplement contains only its own scenario subset.
        direct_path = source / "compute_predicate_truth.jsonl"
        direct_rows = _selected_jsonl(direct_path, ticks, repo_root)
        # A JSON object has named roles; canonical publication sorts its keys.
        # Reconstruct the contract's role order without changing any role value.
        for record in direct_rows:
            declared_roles = tuple(role["key"] for role in templates[record["predicate_id"]]["argument_roles"])
            if set(record["bindings"]) != set(declared_roles) or set(record["binding_ontology_classes"]) != set(declared_roles):
                raise ValueError(f"Direct source roles differ: {record['_unified_source_ref']}")
            record["bindings"] = {role: record["bindings"][role] for role in declared_roles}
            record["binding_ontology_classes"] = {role: record["binding_ontology_classes"][role] for role in declared_roles}
        required_utm_files = {template["grounding_spec"]["candidate_authority"]["source"]
                            for template in templates.values() if template["implementation_status"] == "executable_l1"
                            and template["grounding_spec"]["candidate_authority"]["kind"] == "utm_record"}
        producer_manifest = json.loads((source / "manifest.json").read_text())
        utm_files = sorted(producer_manifest["utm_supplement_summary"]["record_counts"])
        if not required_utm_files <= set(utm_files):
            raise ValueError("Recorded objective producer misses declared UTM input families")
        # objective_pipeline supplies these exact producer states to world_truth;
        # simulation_logs.py persists each state without replacing its meaning.
        utm_records = {name: [] for name in utm_files}
        for record in _selected_jsonl(source / "simulation_logs/utm_log.jsonl", ticks, repo_root):
            name = record["source_record_kind"] + ".jsonl"
            if name not in utm_records:
                raise ValueError(f"Undeclared recorded UTM source family: {name}")
            state = deepcopy(record["state"])
            state["_unified_source_ref"] = record["_unified_source_ref"] + "#state"
            utm_records[name].append(state)
        # Validate recorded schemas before the strict L0 reconstruction.
        direct_by_tick = wt._grounded_direct_rows(direct_rows, episode, templates)
        utm_by_tick = wt._grounded_utm_rows(utm_records, episode)
        compute_index = wt._index_compute_rows(compute_rows, episode)
        domain_refs = defaultdict(list)
        domain_payloads = []
        for record in domain_rows:
            key = (record["tick"], record["observation_family"], record["subject_id"])
            domain_refs[key].append(record["_unified_source_ref"])
            payload = {name: value for name, value in record.items() if name != "_unified_source_ref"}
            domain_payloads.append(payload)
        domain_index, domain_by_family = wt._index_domain_rows(domain_payloads, episode)
        for key, record in domain_index.items():
            record["_unified_source_ref"] = domain_refs[key][0]
        utm_index = wt._index_utm_records(utm_records, episode)
        actual = _actual_objective_scope_inputs(episode, ticks, repo_root, plan_probe_entities)
        manifest, roster = actual["manifest"], actual["roster"]
        roster_entities = wt._index_roster(roster["entities"], roster_path)
        frames, weather = actual["frames"], actual["weather"]
        scopes = wt._build_scope_inventory(episode, roster_entities, compute_rows)
        scope_by_id = {scope.entity_id: scope for scope in scopes}
        classes = wt._build_grounding_class_catalog(scopes=scopes, frames=frames, domain_rows=domain_rows,
            direct_rows=direct_rows, utm_records=utm_records)
        canonical_classes, class_refs, component_relations = _canonical_instance_catalog(
            scopes=scopes, frames=frames, domain_rows=domain_rows, direct_rows=direct_rows,
            utm_records=utm_records, compute_rows=compute_rows, actual=actual)
        binding_entity_ids = {entity["entity_id"] for case in selected for entity in case["bindings"].values()}
        candidate_populations, candidate_population_errors = {}, {}
        audit_tick = min(ticks)
        if class_catalogs is not None or population_audits is not None:
            raw_contexts = wt._build_tick_contexts(episode_id=episode, tick=audit_tick,
                scopes=scopes, roster_entities=roster_entities, entities_at_tick=frames[audit_tick],
                weather_row=weather[audit_tick], domain_index=domain_index,
                domain_by_tick_family=domain_by_family, compute_by_node=compute_index, utm_index=utm_index)
            for predicate, template in templates.items():
                if template["implementation_status"] != "executable_l1":
                    continue
                try:
                    population = wt._ground_candidates_for_template(episode_id=episode, tick=audit_tick,
                        manifest=manifest, template=template, scopes=scopes, contexts=raw_contexts,
                        domain_by_tick_family=domain_by_family, direct_by_tick_predicate=direct_by_tick,
                        utm_by_file_tick=utm_by_tick, entity_classes=classes)
                    candidate_populations[predicate] = population
                    binding_entity_ids.update(entity for candidate in population for entity in candidate["bindings"].values())
                except (ValueError, KeyError, TypeError) as error:
                    candidate_population_errors[predicate] = f"{type(error).__name__}: {error}"
        binding_classes = {entity_id: tags for entity_id, tags in canonical_classes.items() if entity_id in binding_entity_ids}
        for probe in actual["current_plan_producer_probes"]:
            key = (probe["tick"], probe["observation_family"], probe["subject_id"])
            stored_record = domain_index.get(key)
            if stored_record is not None:
                probe["stored_source_context"] = {"plan": {"cooperative_flag": stored_record["values"]["cooperative_flag"]},
                                                  "l0": deepcopy(probe["context"]["l0"])}
                probe["stored_source_ref"] = stored_record["_unified_source_ref"]
                probe["stored_source_truth"] = _evaluate(templates["airspace.aircraft_is_non_cooperative"],
                    probe["stored_source_context"], {}, selected[0]["parameters"])
                probe["current_producer_truth"] = _evaluate(templates["airspace.aircraft_is_non_cooperative"],
                    probe["context"], {}, selected[0]["parameters"])
                probe["source_version_disagreement"] = probe["stored_source_context"] != probe["context"]
            else:
                probe["source_version_disagreement"] = False
                probe["stored_source_record_status"] = "not_available_at_probe_tick"
        if class_catalogs is not None:
            class_catalogs[episode] = {"catalog_id": f"formal:{episode}:instance_classes", "episode_id": episode,
                "classes": {entity_id: sorted(tags) for entity_id, tags in binding_classes.items()},
                "source_refs": {entity_id: sorted(class_refs[entity_id]) for entity_id in binding_classes},
                "nonbinding_class_metadata": {entity_id: {"classes": sorted(tags), "source_refs": sorted(class_refs[entity_id])}
                                              for entity_id, tags in canonical_classes.items() if entity_id not in binding_entity_ids},
                "component_host_relations": component_relations,
                "candidate_resolver_class_index_policy": "official_world_truth_lookup; source_entity_id associations are lookup hints, not identity assertions",
                "identity_catalog_policy": "all_explicit_same_id_class_assertions; component_host_relation_is_not_identity",
                "current_plan_producer_probes": actual["current_plan_producer_probes"],
                "l0_replay": {"source_refs": actual["input_source_refs"], "replay_ref": actual["replay_ref"],
                              "selected_ticks": actual["selected_ticks"], "measured": actual["l0_materialization_measured"]}}
        if population_audits is not None:
            from .generic_grounding import EntityClassCatalog, TypedEntity

            catalog = EntityClassCatalog(episode,
                {entity_id: tuple(tags) for entity_id, tags in binding_classes.items()},
                {entity_id: tuple(sorted(class_refs[entity_id])) for entity_id in binding_classes})
            saved_population = read_episode_cases(episode, repo_root=repo_root,
                start_tick=audit_tick, end_tick=audit_tick)
            saved_by_predicate = defaultdict(list)
            for record in saved_population:
                saved_by_predicate[record["predicate_id"]].append(record)

            def signature(bindings, declared_classes):
                return tuple((role, bindings[role],
                    catalog.resolve(TypedEntity(bindings[role], declared_classes[role])).ontology_classes)
                    for role in sorted(bindings))

            rows = {}
            for predicate, template in templates.items():
                if template["implementation_status"] != "executable_l1":
                    continue
                try:
                    if predicate in candidate_population_errors:
                        raise ValueError(candidate_population_errors[predicate])
                    candidates = candidate_populations[predicate]
                    expected = {signature(candidate["bindings"], candidate["binding_ontology_classes"])
                                for candidate in candidates}
                    saved = {signature({role: entity["entity_id"] for role, entity in record["bindings"].items()},
                                {role: entity["ontology_class"] for role, entity in record["bindings"].items()})
                             for record in saved_by_predicate[predicate]}
                    rows[predicate] = {"predicate_id": predicate, "episode_id": episode,
                        "branch_id": "factual", "tick": audit_tick,
                        "parameter_profile_id": f"formal:{episode}:governed_parameters",
                        "parameter_source_ref": f"aw_data/objective_semantic_truth/{episode}/world_truth_summary.json#governed_parameters",
                        "expected_candidate_count": len(expected), "saved_candidate_count": len(saved),
                        "expected_output_signatures": sorted(expected), "saved_output_signatures": sorted(saved),
                        "missing_outputs": sorted(expected - saved), "unexpected_outputs": sorted(saved - expected),
                        "output_set_equal": expected == saved, "has_real_candidates": bool(expected),
                        "completeness_pass": bool(expected) and expected == saved,
                        "source_refs": actual["input_source_refs"] + [actual["replay_ref"],
                            str(GRAPH_ROOT / episode / "nodes.parquet") + f"#tick={audit_tick}"]}
                except (ValueError, KeyError, TypeError) as error:
                    rows[predicate] = {"predicate_id": predicate, "episode_id": episode, "tick": audit_tick,
                        "has_real_candidates": False, "completeness_pass": False,
                        "source_population_error": f"{type(error).__name__}: {error}"}
            population_audits[episode] = {"episode_id": episode, "tick": audit_tick,
                "rows": rows, "summary": {"fixed_executable_rows": len(rows),
                    "predicates_with_real_candidates": sum(row["has_real_candidates"] for row in rows.values()),
                    "complete_source_output_sets": sum(row["completeness_pass"] for row in rows.values()),
                    "source_population_error_predicates": sum("source_population_error" in row for row in rows.values())}}
        contexts_by_tick = {}
        for case in selected:
            predicate = case["predicate_id"]
            template = templates[predicate]
            grounding = template["grounding_spec"]
            kind = grounding["candidate_authority"]["kind"]
            authority_source = grounding["candidate_authority"]["source"]
            tick = case["tick"]
            expected_ids = {role: entity["entity_id"] for role, entity in case["bindings"].items()}
            case["raw_source_verified"] = False
            case["raw_source_replay_error"] = None
            case["class_catalog_id"] = f"formal:{episode}:instance_classes"
            case["l0_replay_ref"] = actual["replay_ref"]
            try:
                scope = None
                if kind in {"scope_entity", "scope_relation"}:
                    scope_role = next(role for role, spec in grounding["role_sources"].items() if spec["kind"] == "scope_entity")
                    scope = scope_by_id[expected_ids[scope_role]]
                    context_key = (tick, scope.scope_type, scope.entity_id)
                    if context_key not in contexts_by_tick:
                        scene_scope = next(item for item in scopes if item.scope_type == "scene")
                        context_scopes = [scope] if scope.scope_type == "scene" else [scope, scene_scope]
                        built = wt._build_tick_contexts(episode_id=episode, tick=tick, scopes=context_scopes,
                            roster_entities=roster_entities, entities_at_tick=frames[tick], weather_row=weather[tick],
                            domain_index=domain_index, domain_by_tick_family=domain_by_family,
                            compute_by_node=compute_index, utm_index=utm_index)
                        contexts_by_tick[context_key] = built[(scope.scope_type, scope.entity_id)]
                    records = [contexts_by_tick[context_key]]
                elif kind == "domain_observation":
                    records = list(domain_by_family.get((tick, authority_source), ()))
                elif kind == "domain_nested_relation":
                    family, separator, nested_path = authority_source.partition(":")
                    if not separator:
                        raise ValueError(f"Invalid declared nested authority: {authority_source}")
                    records = []
                    for parent in domain_by_family.get((tick, family), ()):
                        nested = _get_path(parent, nested_path)
                        if nested is wt.MISSING or nested is None:
                            continue
                        if not isinstance(nested, list):
                            raise ValueError(f"Declared nested authority is not a list: {authority_source}")
                        for index, item in enumerate(nested):
                            child = deepcopy(item)
                            child["_unified_source_ref"] = parent["_unified_source_ref"] + f"#{nested_path}[{index}]"
                            records.append(child)
                elif kind == "direct_predicate_row":
                    records = [row for row in direct_rows if row["tick"] == tick and row["predicate_id"] == predicate]
                elif kind == "utm_record":
                    records = [row for row in utm_records[authority_source] if row["tick"] == tick]
                elif kind == "episode_operational_region":
                    record = deepcopy(manifest)
                    record["weather"] = deepcopy(weather[tick])
                    record["_unified_source_ref"] = str((capture / "episode_manifest.json").relative_to(repo_root))
                    records = [record]
                else:
                    raise ValueError(f"Unsupported declared source authority: {kind}")
                matches = []
                for record in records:
                    bound = {}
                    for role, spec in grounding["role_sources"].items():
                        role_kind = spec["kind"]
                        if role_kind == "scope_entity":
                            bound[role] = scope.entity_id
                        elif role_kind == "record_binding":
                            bound[role] = _get_path(record, "bindings." + spec["path"])
                        else:
                            bound[role] = _get_path(record, spec["path"])
                    if bound == expected_ids:
                        matches.append(record)
                if not matches:
                    raise ValueError(f"No actual source record with the saved exact typed binding: {predicate}@{tick}:{expected_ids}")
                candidates = [wt._candidate_from_authority_record(episode_id=episode, tick=tick,
                    template=template, authority_kind=kind, authority_source=authority_source,
                    record=record, scope=scope, entity_classes=classes) for record in matches]
                if any(candidate is None for candidate in candidates):
                    raise ValueError("A matched actual record did not form a declared candidate")
                first = candidates[0]
                if any(candidate["bindings"] != first["bindings"] or candidate["binding_ontology_classes"] != first["binding_ontology_classes"] for candidate in candidates[1:]):
                    raise ValueError("Actual authority records have inconsistent typed bindings")
                if first["binding_ontology_classes"] != {role: value["ontology_class"] for role, value in case["bindings"].items()}:
                    raise ValueError("Actual source classes differ from published assertion classes")
                first["duplicate_records"] = matches
                raw_context = wt._candidate_evaluation_context(tick=tick, candidate=first,
                    template=template, domain_by_tick_family=domain_by_family)
                if scope is not None:
                    active = _get_path(matches[0], "l0.scope_active")
                    if active is not wt.MISSING:
                        raw_context["l0"] = {"scope_active": active}
                # Unavailability is an explicit immutable source contract.
                unavailable = _get_path(case["context"], "l0.predicate_source_unavailability")
                if unavailable is not wt.MISSING:
                    availability_path = source / "l0_predicate_source_availability.json"
                    availability = json.loads(availability_path.read_text())
                    if unavailable not in availability["entries"]:
                        raise ValueError("Published source-unavailability marker is not present in actual availability authority")
                    if "l0" not in raw_context:
                        raw_context["l0"] = {}
                    raw_context["l0"]["predicate_source_unavailability"] = deepcopy(unavailable)
                clean = {}
                for field in template["state_contract"]["required_fields"]:
                    value = _get_path(raw_context, field["field"])
                    if value is not wt.MISSING:
                        _set_path(clean, field["field"], value)
                if "l0" in raw_context:
                    clean["l0"] = deepcopy(raw_context["l0"])
                saved_context = deepcopy(case["context"])
                raw_truth = _evaluate(template, clean, case["previous_context"], case["parameters"])
                field_differences = []
                for field in template["state_contract"]["required_fields"]:
                    name = field["field"]
                    saved, current = _get_path(saved_context, name), _get_path(clean, name)
                    if saved is wt.MISSING and current is wt.MISSING:
                        continue
                    if saved is wt.MISSING or current is wt.MISSING or saved != current:
                        field_differences.append({"field": name,
                            "published": None if saved is wt.MISSING else saved,
                            "current_source": None if current is wt.MISSING else current,
                            "published_missing": saved is wt.MISSING, "current_source_missing": current is wt.MISSING})
                refs = [record["_unified_source_ref"] for record in matches if "_unified_source_ref" in record]
                if scope is not None:
                    refs += [str(roster_path.relative_to(repo_root)) + "#entity=" + scope.entity_id,
                             actual["frame_refs"][tick] + f"&entity={scope.source_entity_id}"]
                    refs += actual["input_source_refs"]
                    refs += list(raw_context.get("_observation_sources", {}).values())
                required = [item["field"] for item in template["state_contract"]["required_fields"]]
                case.update({"published_context": saved_context, "context": clean,
                    "current_truth": raw_truth, "raw_source_truth": raw_truth,
                    "raw_source_verified": True, "raw_source_refs": list(dict.fromkeys(refs)),
                    "raw_binding_provenance": deepcopy(first["binding_provenance"]),
                    "raw_field_differences": field_differences,
                    "observed_fields": [field for field in required if not wt._missing_value(_get_path(clean, field))],
                    "missing_fields": [field for field in required if wt._missing_value(_get_path(clean, field))],
                    "source_kind": "raw_authoritative_record_replay"})
                case["semantic_ambiguities"] = []
                semantic_field_differences = [difference for difference in field_differences
                    if not (wt._missing_value(_get_path(saved_context, difference["field"]))
                            and wt._missing_value(_get_path(clean, difference["field"])))]
                if semantic_field_differences:
                    case["semantic_ambiguities"].append({"kind": "current_field_mapping_vs_published_evidence", "field_differences": semantic_field_differences,
                        "label_policy": "preserve_published_label"})
                if raw_truth != case["saved_truth"]:
                    case["semantic_ambiguities"].append({"kind": "current_source_contract_vs_published_truth", "saved_truth": case["saved_truth"],
                        "current_truth": raw_truth, "label_policy": "preserve_published_label"})
                if predicate == "airspace.aircraft_is_non_cooperative":
                    probes = [probe for probe in actual["current_plan_producer_probes"]
                              if probe.get("source_version_disagreement")]
                    case["current_producer_probes"] = probes
                    exact = [probe for probe in probes if probe["subject_id"] == expected_ids["aircraft"] and probe["tick"] == tick]
                    if exact:
                        case["current_producer_context"] = deepcopy(exact[0]["context"])
                        case["current_producer_truth"] = exact[0]["current_producer_truth"]
                    if probes:
                        case["semantic_ambiguities"].append({"kind": "stored_raw_plan_vs_current_producer_version",
                            "actual_probes": probes, "label_policy": "preserve_published_label",
                            "raw_context_policy": "retain_stored_source_context_separately_from_current_producer_context"})
            except (ValueError, KeyError, TypeError) as error:
                case["raw_source_replay_error"] = f"{type(error).__name__}: {error}"
            result.append(case)
    return result
