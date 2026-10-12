"""Executable event tasks, evidence checks and an explicit sampling population."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types, get_core_predicate_templates,
)
from Dataset.world_model.graph.contract import REPO, CONTRACT_VERSION, declared_samples, iter_jsonl, read_json
from Dataset.world_model.graph.records import DEFAULT_ROOT, load_scope
from Dataset.world_model.graph.contract import canonical_predicate_id, _l4_roles, compile_catalog
from Dataset.world_model.graph.validator import GraphValidator
from Dataset.world_model.graph.entity_identity import BUSINESS_SOURCE_OBJECTS

HORIZON_TICKS = 50
SAMPLING_STEP = 5
ANCHOR_STEP = 25
TASK_VERSION = "event-belief-3"
CATALOG = REPO / "design/belief/event_tasks.json"
OUTPUT = REPO / "Dataset/world_model/runs/belief_event_tasks"


@lru_cache(maxsize=1)
def task_definitions() -> list[dict[str, Any]]:
    templates = {r["id"]: r for r in get_core_predicate_templates()}
    result = []
    for rule in get_core_event_occurrence_types():
        trigger = templates[rule["trigger_predicate_id"]]
        if trigger["implementation_status"] != "executable_l1":
            raise ValueError(f"event trigger is not executable: {rule['rule_id']}")
        result.append({
            "id": rule["event_type"], "target_kind": "ontology_event", "rule_id": rule["rule_id"],
            "event_family": rule["event_family_id"],
            "trigger_predicate": rule["trigger_predicate_id"],
            "onset_direction": rule["trigger_direction"],
            "onset_tick": rule["event_tick"],
            "trigger_roles": [r["key"] for r in trigger["argument_roles"]],
            "participants": rule["participant_roles"],
            "support_conditions": rule["support_conditions"],
            "terminal_condition": rule["terminal"],
            "prediction_horizon_ticks": HORIZON_TICKS,
            "sampling_step_ticks": SAMPLING_STEP,
            "anchor_step_ticks": ANCHOR_STEP,
            "queries": ["future_onset", "current_hidden_predicate", "future_terminal"],
            "definition_source": "Dataset/semantic_rules/profiles/event_family_authority_mapping.json",
            "predicate_source": "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json",
        })
    local_source = "Dataset/semantic_rules/profiles/l4_arm_business_semantics.json"
    local = read_json(REPO / local_source)
    families = defaultdict(list)
    for profile, rule in local["profiles"].items():
        families[rule["event_family"]].append((profile, rule))
    for family, profiles in sorted(families.items()):
        rule = profiles[0][1]
        trigger = canonical_predicate_id(rule["onset_predicate"])
        terminal = canonical_predicate_id(rule["terminal_predicate"])
        trigger_roles = _l4_roles(trigger.removeprefix("l4."))
        terminal_roles = _l4_roles(terminal.removeprefix("l4."))
        result.append({"id": "l4." + family.removeprefix("l4_"), "target_kind": "local_business_event",
            "rule_id": "l4_arm_business." + rule["mechanism"], "event_family": family,
            "authority_scope": local["authority_scope"],
            "profiles": [profile for profile, _ in profiles],
            "trigger_predicate": trigger, "onset_direction": "rising",
            "onset_tick": "after", "unknown_before_onset": "left_censored",
            "trigger_roles": [r["key"] for r in trigger_roles], "participants": trigger_roles,
            "support_conditions": [], "terminal_condition": {"predicate_id": terminal,
                "value": "true", "hold_samples": local["terminal_hold_samples"],
                "join_roles": sorted({r["key"] for r in trigger_roles} & {r["key"] for r in terminal_roles})},
            "prediction_horizon_ticks": HORIZON_TICKS, "sampling_step_ticks": SAMPLING_STEP,
            "anchor_step_ticks": ANCHOR_STEP,
            "queries": ["future_onset", "current_hidden_predicate", "future_terminal"],
            "definition_source": local_source, "predicate_source": local_source})
    _, arm_sources = declared_samples()
    x_profiles = sorted({p.relative_to(REPO / "arms").parts[0] for p in arm_sources
                         if p.relative_to(REPO / "arms").parts[0].startswith("X")})
    for predicate in compile_catalog():
        if predicate.get("kind") != "predicate" or predicate.get("source_system") != "X":
            continue
        if predicate["execution_status"] != "source_executable":
            raise ValueError(f"X predicate has no executable definition: {predicate['id']}")
        result.append({"id": "predicate:" + predicate["id"], "target_kind": "predicate_only",
            "rule_id": predicate["id"], "event_family": None, "profiles": x_profiles,
            "trigger_predicate": predicate["id"], "onset_direction": "rising",
            "onset_tick": "after", "trigger_roles": [r["key"] for r in predicate["roles"]],
            "participants": predicate["roles"], "support_conditions": [], "terminal_condition": None,
            "prediction_horizon_ticks": HORIZON_TICKS, "sampling_step_ticks": 1,
            "anchor_step_ticks": ANCHOR_STEP, "queries": ["current_hidden_predicate", "future_onset"],
            "definition_source": "Dataset/tools/x_arm_pipeline.py:calculate",
            "predicate_source": predicate["source"], "raw_predicate_ids": predicate["raw_ids"]})
    business_source = "Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json"
    business = read_json(REPO / business_source)
    event_roles = {r["id"]: r["roles"] for r in compile_catalog() if r["kind"] == "event"}
    predicate_roles = {r["id"]: r["roles"] for r in compile_catalog() if r["kind"] == "predicate"}
    existing = {r["id"] for r in result}
    for predicate, rule in business["event_rules"].items():
        identifier = rule["event_type_id"]
        if identifier in existing:
            continue
        result.append({"id": identifier, "target_kind": "business_ontology_event",
            "rule_id": rule["rule_id"], "event_family": identifier.split(".")[0],
            "trigger_predicate": predicate, "onset_direction": "rising", "onset_tick": "after",
            "trigger_roles": [r["key"] for r in predicate_roles[predicate]],
            "participants": event_roles[identifier], "support_conditions": [], "terminal_condition": None,
            "recovery_scope": "not declared by the current executable rule",
            "prediction_horizon_ticks": HORIZON_TICKS, "sampling_step_ticks": SAMPLING_STEP,
            "anchor_step_ticks": ANCHOR_STEP, "queries": ["current_hidden_predicate", "future_onset"],
            "definition_source": business_source, "predicate_source": business_source})
    return result


def _binding_key(bindings: dict) -> tuple:
    return tuple(sorted(bindings.items()))


def _matches(event: dict, bindings: dict) -> bool:
    return all(event["bindings"].get(role) == entity for role, entity in bindings.items())


def load_evidence(partition: Path, episode_id: str) -> dict[str, Any]:
    """Use typed graph evidence, including validated L6 inheritance and X onsets."""
    scope = load_scope(partition)
    if scope["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"event tasks require the current graph: {partition}")
    nodes, truths, transitions, events, outcomes = {}, {}, {}, [], {}
    streams, by_predicate, by_predicate_tick = defaultdict(dict), defaultdict(list), defaultdict(list)
    identities, observed_ticks = {}, defaultdict(set)
    for batch in pq.ParquetFile(partition / "nodes.parquet").iter_batches(batch_size=8192):
        for stored in batch.to_pylist():
            if stored["node_type"] not in {"PredicateAssertion", "PredicateTransition", "EventOccurrence", "EventOutcome", "EntityIdentity", "EntityState", "CommandDispatch", "ContextAnchor"}:
                continue
            a = json.loads(stored["attrs_json"])
            if stored["node_type"] == "ContextAnchor" and not (a.get("ontology_class") in BUSINESS_SOURCE_OBJECTS
                                                              and a.get("source_object_field") is not None):
                continue
            nodes[stored["node_id"]] = {"kind": stored["node_type"], "tick": stored["tick"],
                "identity_status": a.get("identity_status"), "bindings": a.get("bindings"),
                "ontology_class": a.get("ontology_class"),
                "raw": a.get("raw") if stored["node_type"] == "PredicateTransition" else None,
                "source": stored["source_file"], "line": stored["source_line"]}
            location = {"source": stored["source_file"], "line": stored["source_line"], "graph_node": stored["node_id"]}
            if stored["node_type"] == "EntityIdentity" and a["identity_namespace"] in {"sample", "sumo_tls_controller", "compute_node"}:
                identities[a["entity_id"]] = a
            if (stored["node_type"] == "ContextAnchor" and a.get("ontology_class") in BUSINESS_SOURCE_OBJECTS
                    and a.get("source_object_field") is not None):
                identities[a["entity_id"]] = {"identity_status": "resolved_business_object",
                    "ontology_class": a["ontology_class"], "event_non_actor_allowed": True}
            if stored["node_type"] == "EntityState" and stored["tick"] % SAMPLING_STEP == 0:
                observed_ticks[a["entity_id"]].add(stored["tick"])
            if stored["node_type"] == "CommandDispatch" and isinstance(a.get("target_entity_id"), str):
                observed_ticks[a["target_entity_id"]].add(stored["tick"])
            if stored["node_type"] == "PredicateAssertion":
                row = {"truth_id": stored["node_id"], "tick": stored["tick"], "predicate_id": stored["category_id"],
                       "bindings": a["bindings"], "value": a["truth_status"], "record_location": location}
                truths[stored["node_id"]] = (stored["source_line"], row)
                # Formal world truth and ARM's own sampled truth are the declared
                # authorities. Supplement/projection copies are not extra trials.
                selected = (scope["source_kind"] == "formal" and a.get("assertion_layer") is None
                            or scope["source_kind"] == "arm" and a.get("assertion_layer") in {"sampled", "dense", "history"})
                if not selected:
                    continue
                key = row["predicate_id"], _binding_key(row["bindings"])
                tick = row["tick"]
                if tick in streams[key]:
                    raise ValueError(f"ambiguous target authority: {partition}:{key}@{tick}")
                streams[key][tick] = row
                by_predicate[row["predicate_id"]].append(row)
                by_predicate_tick[row["predicate_id"], tick].append(row)
            elif stored["node_type"] == "PredicateTransition":
                transitions[stored["node_id"]] = (stored["source_line"], stored)
            elif stored["node_type"] == "EventOccurrence":
                events.append({"event_id": stored["node_id"], "event_type_id": stored["category_id"],
                    "bindings": a["bindings"], "trigger_tick": a.get("trigger_tick", (a.get("raw") or {}).get("trigger_tick", stored["tick"])),
                    "detection_tick": stored["tick"],
                    "onset_evidence_status": a.get("onset_evidence_status", "within_saved_window"),
                    "record_location": location, "supporting_truth_ids": [], "supporting_transition_ids": []})
            elif stored["node_type"] == "EventOutcome":
                outcomes[stored["node_id"]] = {"outcome_id": stored["node_id"], "status": a["status"],
                    "censored": a.get("censored", False),
                    "terminal_tick": stored["tick"], "target_missing": a.get("target_missing", False), "record_location": location}
    events_by_id = {e["event_id"]: e for e in events}
    matched_outcomes = {}
    evidence_errors = []
    for batch in pq.ParquetFile(partition / "edges.parquet").iter_batches(batch_size=16384):
        for edge in batch.to_pylist():
            kind = edge["edge_type"]
            if kind == "event_outcome" and edge["source"] in events_by_id:
                if edge["target"] not in outcomes or edge["source"] in matched_outcomes:
                    raise ValueError(f"event outcome is ambiguous: {partition}:{edge['edge_id']}")
                matched_outcomes[edge["source"]] = outcomes[edge["target"]]
            if edge["target"] in events_by_id and kind in {"assertion_supports_event", "transition_supports_event"}:
                name = "supporting_truth_ids" if kind == "assertion_supports_event" else "supporting_transition_ids"
                events_by_id[edge["target"]][name].append(edge["source"])
            if kind == "event_participant" and edge["source"] in events_by_id:
                target = nodes[edge["target"]]
                if (target["identity_status"] != "resolved"
                        and not (target["kind"] == "ContextAnchor" and target["ontology_class"] in BUSINESS_SOURCE_OBJECTS)):
                    evidence_errors.append({"event": edge["source"], "participant": edge["target"], "reason": "unresolved participant"})
    executable = {r["id"]: r for r in task_definitions()}
    for event in events:
        if event["event_type_id"] in executable:
            if "future_terminal" in executable[event["event_type_id"]]["queries"] and event["event_id"] not in matched_outcomes:
                evidence_errors.append({"event": event["event_id"], "reason": "outcome/censor record missing"})
            if (not event["supporting_truth_ids"] and not event["supporting_transition_ids"]
                    and event["onset_evidence_status"] != "left_censored"):
                evidence_errors.append({"event": event["event_id"], "reason": "onset evidence missing"})
    if evidence_errors:
        raise ValueError(f"event evidence is incomplete: {partition}:{evidence_errors}")
    return {"nodes": nodes, "truths": truths, "transitions": transitions, "streams": streams,
            "by_predicate": by_predicate, "by_predicate_tick": by_predicate_tick,
            "identities": identities, "observed_ticks": observed_ticks,
            "events": events, "outcomes": matched_outcomes}


def _support_known(definition: dict, evidence: dict, bindings: dict, tick: int) -> bool:
    for condition in definition["support_conditions"]:
        at = tick - SAMPLING_STEP if condition["at"] == "before" else tick
        for offset in range(condition.get("hold_samples", 1)):
            joined = [r for r in evidence["by_predicate_tick"][condition["predicate_id"], at + offset * SAMPLING_STEP]
                      if all(r["bindings"].get(role) == bindings.get(role)
                                                for role in condition["join_roles"])]
            if not joined or any(r["value"] not in {"true", "false"} for r in joined):
                return False
    return True


def target_rows(sample_key: str, evidence: dict, start: int, end: int) -> Iterable[dict]:
    """Regular supervised queries; applicability never selects the model input."""
    from Dataset.semantic_truth.entity_scope import entity_scope_contract
    role_policy = entity_scope_contract()["role_role_kinds"]
    events_by_type = defaultdict(list)
    for event in evidence["events"]:
        events_by_type[event["event_type_id"]].append(event)
    for definition in task_definitions():
        profile = sample_key.split(":", 1)[1].split("/")[0].split("__")[0]
        if definition.get("profiles") is not None and profile not in definition["profiles"]:
            continue
        predicate = definition["trigger_predicate"]
        for (pred, binding_key), stream in sorted(evidence["streams"].items()):
            if pred != predicate:
                continue
            bindings = dict(binding_key)
            matching_events = [e for e in events_by_type[definition["id"]] if _matches(e, bindings)]
            first_anchor = max(start, min(stream))
            first_anchor += (-first_anchor) % ANCHOR_STEP
            for anchor in range(first_anchor, end + 1, ANCHOR_STEP):
                current = stream.get(anchor)
                if current is None:
                    continue  # No candidate observed at this query time.
                allowed = True
                for role, entity in bindings.items():
                    identity = evidence["identities"].get(entity)
                    is_actor = role in role_policy["actor_roles"] or definition["id"].startswith("l4.")
                    permission = "event_actor_allowed" if is_actor else "event_non_actor_allowed"
                    allowed &= identity is not None and identity.get(permission) is True
                applicable = current["value"] != "out_of_scope" and allowed
                current_known = current["value"] in {"true", "false"}
                observation_missing = any(not any(max(start, anchor - 100) <= t <= anchor for t in evidence["observed_ticks"][entity])
                                          for entity in bindings.values())
                base = {"sample_key": sample_key, "task_id": definition["id"],
                        "predicate_id": predicate, "bindings": bindings, "anchor_tick": anchor,
                        "horizon_ticks": HORIZON_TICKS, "applicable": applicable,
                        "observation_missing": observation_missing}
                yield {**base, "query": "current_hidden_predicate", "censored": False, "observed_until_tick": anchor,
                       "target_valid": current_known and applicable, "target_value": int(current["value"] == "true") if current_known and applicable else None,
                       "target_missing": not current_known and applicable, "evidence_ids": [current["truth_id"]]}
                # A horizon past the saved window is truncated (censored). An onset or
                # terminal actually observed before the window end is still a known
                # positive; only "nothing happened" stays unknown under truncation.
                observed_until = min(end, anchor + HORIZON_TICKS)
                future_ticks = range(anchor, observed_until + 1, definition["sampling_step_ticks"])
                values = [stream.get(t) for t in future_ticks]
                gaps = any(r is None or r["value"] == "unknown" for r in values)
                outside = any(r is not None and r["value"] == "out_of_scope" for r in values)
                support_known = True
                for previous, after in zip(values, values[1:]):
                    if previous is None or after is None:
                        continue
                    direction = ("rising" if (previous["value"], after["value"]) == ("false", "true")
                                 else "falling" if (previous["value"], after["value"]) == ("true", "false") else None)
                    if direction == definition["onset_direction"]:
                        support_known &= _support_known(definition, evidence, bindings, after["tick"])
                positives = [e for e in matching_events if anchor < e["trigger_tick"] <= observed_until]
                predicate_onsets = [after for previous, after in zip(values, values[1:])
                    if previous is not None and after is not None
                    and (previous["value"], after["value"]) == ("false", "true")]
                left_censored = (definition.get("unknown_before_onset") == "left_censored"
                    and any(evidence["nodes"][identifier]["raw"].get("from_value") == "unknown"
                            for event in positives for identifier in event["supporting_transition_ids"]))
                censored = anchor + HORIZON_TICKS > end
                observed_positive = bool(predicate_onsets if definition["target_kind"] == "predicate_only" else positives)
                valid = (applicable and not outside and not gaps and support_known and not left_censored
                         and (not censored or observed_positive))
                yield {**base, "query": "future_onset", "censored": censored, "observed_until_tick": observed_until,
                       "left_censored": left_censored,
                       "target_valid": valid, "target_value": int(observed_positive) if valid else None,
                       "target_missing": gaps or not support_known, "applicable": applicable and not outside,
                       "evidence_ids": [r["truth_id"] for r in predicate_onsets] if definition["target_kind"] == "predicate_only" else [e["event_id"] for e in positives],
                       "positive_ticks": [r["tick"] for r in predicate_onsets] if definition["target_kind"] == "predicate_only" else [e["trigger_tick"] for e in positives],
                       "positive_evidence_ids": [r["truth_id"] for r in predicate_onsets] if definition["target_kind"] == "predicate_only" else [e["event_id"] for e in positives],
                       "evidence_sources": [r["record_location"] for r in predicate_onsets] if definition["target_kind"] == "predicate_only" else [e["record_location"] for e in positives]}
                if definition["terminal_condition"] is None:
                    continue
                active = [event for event in matching_events if event["detection_tick"] <= anchor
                          and (evidence["outcomes"][event["event_id"]]["terminal_tick"] is None
                               or evidence["outcomes"][event["event_id"]]["terminal_tick"] > anchor)]
                terminal = definition["terminal_condition"]
                terminal_rows = [r for t in future_ticks
                    for r in evidence["by_predicate_tick"][terminal["predicate_id"], t]
                    if all(r["bindings"].get(role) == bindings.get(role) for role in terminal["join_roles"])]
                covered = {r["tick"] for r in terminal_rows if r["value"] in {"true", "false"}}
                terminal_missing = any(t not in covered for t in future_ticks)
                terminal_outside = any(r["value"] == "out_of_scope" for r in terminal_rows)
                matched = [evidence["outcomes"][event["event_id"]] for event in active]
                completed = [r for r in matched if r["status"] in {"success", "succeeded"}
                             and r["terminal_tick"] is not None and anchor < r["terminal_tick"] <= observed_until]
                terminal_valid = (applicable and bool(active) and not terminal_missing and not terminal_outside
                                  and (not censored or bool(completed)))
                yield {**base, "query": "future_terminal", "censored": censored, "observed_until_tick": observed_until,
                       "applicable": applicable and bool(active) and not terminal_outside,
                       "target_valid": terminal_valid, "target_value": int(bool(completed)) if terminal_valid else None,
                       "target_missing": terminal_missing if active else False,
                       "onset_event_ids": [event["event_id"] for event in active],
                       "evidence_ids": [r["outcome_id"] for r in matched],
                       "positive_ticks": [r["terminal_tick"] for r in completed],
                       "positive_evidence_ids": [r["outcome_id"] for r in completed],
                       "evidence_sources": [r["record_location"] for r in matched]}


def _target_inputs(root: Path) -> dict:
    paths = [Path(__file__), REPO / "Dataset/semantic_rules/profiles/event_family_authority_mapping.json",
              REPO / "Dataset/semantic_rules/profiles/l4_arm_business_semantics.json",
              REPO / "Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json"]
    return {"graph_version": CONTRACT_VERSION, "task_version": TASK_VERSION,
        "graph_files": {name: {"bytes": (root / name).stat().st_size,
                             "mtime_ns": (root / name).stat().st_mtime_ns}
                        for name in ("nodes.parquet", "edges.parquet", "scope.parquet")},
        "definition_inputs": {str(path.relative_to(REPO)): {"bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns} for path in paths}}


def _prepare_targets(partition_text: str, destination_text: str) -> dict:
    root, destination = Path(partition_text), Path(destination_text)
    scope = load_scope(root)
    evidence = load_evidence(root, scope["episode_id"])
    counts, local = defaultdict(Counter), Counter()
    with destination.open("w", encoding="utf-8") as stream:
        for row in target_rows(scope["sample_key"], evidence, scope["start_tick"], scope["end_tick"]):
            status = ("positive" if row["target_value"] == 1 else "valid_negative") if row["target_valid"] else "invalid"
            key = row["task_id"], row["query"]
            counts[key][status] += 1
            for flag in ("censored", "target_missing", "observation_missing"):
                counts[key][flag] += int(row[flag])
            counts[key]["inapplicable"] += int(not row["applicable"])
            counts[key]["left_censored"] += int(row.get("left_censored", False))
            if row["censored"]:
                counts[key]["censored_observed_positive" if row["target_valid"] else "censored_invalid"] += 1
            local[status] += 1
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    event_counts = Counter(e["event_type_id"] for e in evidence["events"])
    return {"sample_key": scope["sample_key"], "target_counts": dict(local),
        "event_counts": dict(event_counts), "source_status": "present",
        "counts_by_task": [{"task_id": task, "query": query, "counts": dict(c)}
                           for (task, query), c in sorted(counts.items())], "artifact_inputs": _target_inputs(root)}


def build_task_products(output: Path = OUTPUT, *, graph_root: Path = DEFAULT_ROOT,
                        samples: tuple[str, ...] = (), workers: int = 8) -> dict:
    tasks = task_definitions()
    output, graph_root = output.resolve(), graph_root.resolve()
    partitions = sorted(p.parent for p in graph_root.rglob("scope.parquet") if ".preparation" not in p.relative_to(graph_root).parts)
    if samples:
        partitions = [p for p in partitions if load_scope(p)["sample_key"] in samples]
    counts, event_counts = defaultdict(Counter), Counter()
    coverage, unregistered, failures = [], Counter(), []
    task_ids = {r["id"] for r in tasks}
    def collect(result):
        for row in result["counts_by_task"]:
            counts[row["task_id"], row["query"]].update(row["counts"])
        event_counts.update(result["event_counts"])
        unregistered.update({k: v for k, v in result["event_counts"].items() if k not in task_ids})
        coverage.append(result)
    output.mkdir(parents=True, exist_ok=True)
    preparation = output / ".preparation"
    preparation.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=preparation) as temporary, ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {}
        for index, partition in enumerate(partitions):
            existing = output / partition.relative_to(graph_root)
            if (existing / "metadata.json").is_file() and (existing / "targets.jsonl").is_file():
                result = read_json(existing / "metadata.json")
                if result["artifact_inputs"] == _target_inputs(partition):
                    collect(result)
                    continue
            destination = Path(temporary) / f"{index}.jsonl"
            future = pool.submit(_prepare_targets, str(partition), str(destination))
            jobs[future] = partition, destination
        for future in as_completed(jobs):
            partition, prepared = jobs[future]
            try:
                result = future.result()
            except Exception as error:
                failures.append({"partition": str(partition), "error": f"{type(error).__name__}: {error}"})
                print(json.dumps({"target_failure": failures[-1]}), flush=True)
                continue
            target_path = output / partition.relative_to(graph_root) / "targets.jsonl"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            prepared.replace(target_path)
            result["targets"] = target_path.relative_to(REPO).as_posix()
            (target_path.parent / "metadata.json").write_text(json.dumps(result, ensure_ascii=False) + "\n")
            collect(result)
            print(json.dumps({"targets_completed": len(coverage), "requested": len(partitions),
                              "sample_key": result["sample_key"], "counts": result["target_counts"]}), flush=True)
    if failures:
        raise ValueError(f"event evidence/targets failed: {failures}")
    catalog = {"version": TASK_VERSION, "status": "constructed_requires_graph_and_input_acceptance",
               "graph_version": CONTRACT_VERSION,
               "population": "declared trigger tuples at regular 25-tick anchors; model queries require prefix-known subjects and never depend on future outcomes",
               "tasks": [{**r, "actual_event_records": event_counts[r["id"]],
                          "counts": {q: dict(counts[r['id'], q]) for q in r["queries"]}} for r in tasks],
               "samples": coverage, "unregistered_event_types": dict(unregistered)}
    catalog_path = CATALOG if output == OUTPUT and not samples else output / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = build_task_products(args.output, graph_root=args.graph_root, samples=tuple(args.sample), workers=args.workers)
    print(json.dumps({"samples": len(result["samples"]), "tasks": len(result["tasks"])}))
