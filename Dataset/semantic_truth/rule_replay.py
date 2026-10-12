"""Replay repaired predicate rules against published formal graph assertions.

Sources remain immutable. This produces a change ledger and measures how those
changes affect the frozen transition pilot's sample index. It does not regenerate
event chains, ARM branches, graph products, model inputs or checkpoints.
"""
from __future__ import annotations

from Dataset.semantic_truth.provenance import open_artifact_text
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.compute as ac
import pyarrow.parquet as pq
import torch

from Dataset.semantic_simulation import compute_comm as cc
from Dataset.semantic_simulation import domain_state as ds
from Dataset.semantic_truth import world_truth as wt
from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates, get_governed_parameter_defaults,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "aw_data/objective_semantic_truth"
GRAPHS = ROOT / "Dataset/world_model/runs/belief_label_graph/formal"
OUT = ROOT / "Dataset/world_model/runs/rule_consistency"
PILOT = ROOT / "Dataset/world_model/runs/predicate_transition_pilot"
RESTRICTED = "airspace.aircraft_inside_restricted_airspace"
COOPERATIVE = "airspace.aircraft_is_non_cooperative"
DEGRADED = "communication.link_degraded"
CAPACITY = "compute.compute_resource_over_allocated"
FAILED = "compute.compute_task_execution_failed"
COMPROMISED = "security.ground_control_station_compromised"
TARGETS = (RESTRICTED, COOPERATIVE, DEGRADED, CAPACITY, FAILED, COMPROMISED)
ISSUES = {
    "restricted_interior": "Signed restricted-prism containment and clearance",
    "inactive_region_cooperation": "Cooperation requires observed evidence; active regions alone do not establish it",
    "link_thresholds": "Link degradation follows latency and loss requirements",
    "resource_dimensions": "Requested CPU, GPU and memory are compared with capacity",
    "station_compromise": "Station compromise is independent of command authorization",
    "migration_failure": "Task failure follows execution state",
    "direct_applicability": "Direct truth preserves out-of-scope applicability",
    "global_threshold": "Episode values cannot override governed thresholds",
}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def binding_key(bindings: dict) -> str:
    return json.dumps(bindings, sort_keys=True, separators=(",", ":"))


def key(tick: int, predicate: str, bindings: dict) -> tuple:
    return int(tick), predicate, binding_key(bindings)


def evaluate_context(template: dict, context: dict, parameters: dict) -> str:
    for field in template["state_contract"]["required_fields"]:
        if wt._missing_value(wt._value_at_path(context, field["field"])):
            return "unknown"
    return wt._expression_truth_value(wt._evaluate_expression(
        template["evaluation_spec"], context, {}, parameters))


def rows(path: Path, tokens: tuple[str, ...] = ()) -> Iterator[dict]:
    with open_artifact_text(path) as stream:
        for line in stream:
            if not tokens or any(token in line for token in tokens):
                yield json.loads(line)


def frozen_assertions(episode: str) -> dict:
    result = {}
    path = GRAPHS / episode / "nodes.parquet"
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["node_id", "node_type", "tick", "attrs_json"], batch_size=65536
    ):
        batch = batch.filter(ac.equal(batch.column("node_type"), "PredicateAssertion"))
        attrs = batch.column("attrs_json")
        selected = ac.match_substring(attrs, TARGETS[0])
        for predicate in TARGETS[1:]:
            selected = ac.or_(selected, ac.match_substring(attrs, predicate))
        selected = ac.or_(selected, ac.and_(
            ac.match_substring(attrs, "direct_predicate_value"),
            ac.match_substring(attrs, "out_of_scope"),
        ))
        for row in batch.filter(selected).to_pylist():
            value = json.loads(row["attrs_json"])
            if value.get("assertion_layer") is not None:
                continue
            identity = key(row["tick"], value["predicate_id"], value["bindings"])
            if identity in result:
                raise ValueError(f"Duplicate formal assertion: {episode}:{identity}")
            result[identity] = {
                "value": value["truth_status"], "node_id": row["node_id"],
                "evidence": value["evidence"], "predicate": value["predicate_id"],
                "bindings": value["bindings"],
            }
    return result


def replay_episode(episode: str, templates: dict, parameters: dict) -> tuple[dict, list]:
    source = SOURCES / episode
    frozen = frozen_assertions(episode)
    compared = Counter()
    seen = set()
    changes = {}

    def compare(tick, predicate, bindings, expected, issue, source_ref):
        identity = key(tick, predicate, bindings)
        if identity not in frozen:
            return  # Source rows outside the published ontology/candidate population.
        actual = frozen[identity]
        seen.add(identity)
        compared[issue] += 1
        # These repairs leave upstream entity applicability unchanged. The direct
        # truth repair explicitly changes its own projected applicability value.
        if actual["value"] == "out_of_scope" and issue != "direct_applicability":
            return
        if actual["value"] == expected:
            return
        if expected not in wt.TRUTH_VALUES:
            raise ValueError(f"Invalid replay result: {episode}:{identity}:{expected}")
        if identity in changes:
            if changes[identity]["after"] != expected:
                raise ValueError(f"Conflicting rule replays: {episode}:{identity}")
            changes[identity]["issues"].append(issue)
            return
        changes[identity] = dict(
            episode=episode, tick=int(tick), predicate=predicate,
            bindings_json=binding_key(bindings), before=actual["value"], after=expected,
            issues=[issue], graph_node_id=actual["node_id"], source_ref=source_ref,
        )

    families = (
        "predicate_contract_aircraft_region_geometry",
        "predicate_contract_aircraft_plan",
        "predicate_contract_region_runtime_state",
    )
    l0_path = source / "l0_predicate_state.jsonl"
    geometry = []
    plans = []
    runtime = defaultdict(list)
    for row in rows(l0_path, families):
        family = row["observation_family"]
        if family not in families:
            continue
        if family == families[0]:
            geometry.append(row)
        elif family == families[1]:
            plans.append(row)
        else:
            runtime[(row["tick"], family)].append(row)
    wt._resolve_episode_governed_parameters(parameters, runtime)
    global_checks = sum(len(value) for value in runtime.values())

    for row in geometry:
        values = row["values"]
        bindings = {"actor": values["aircraft_id"], "restricted_region": values["restricted_region_id"]}
        context = wt._candidate_evaluation_context(
            tick=row["tick"], template=templates[RESTRICTED],
            candidate={"record": row, "tuple_id": row["observation_id"], "bindings": bindings},
            domain_by_tick_family={},
        )
        expected = evaluate_context(templates[RESTRICTED], context, parameters)
        compare(row["tick"], RESTRICTED, bindings, expected, "restricted_interior",
                relative(l0_path) + "#observation_id=" + row["observation_id"])
    for row in plans:
        actor = row["subject_id"]
        # These plan rows used name/role inference, not observed cooperation.
        expected = evaluate_context(templates[COOPERATIVE], {"plan": {"cooperative_flag": None}}, parameters)
        compare(row["tick"], COOPERATIVE, {"aircraft": actor}, expected, "inactive_region_cooperation",
                relative(l0_path) + "#observation_id=" + row["observation_id"])

    compute_path = source / "compute_state.jsonl"
    for row in rows(compute_path):
        ref = relative(compute_path) + f"#tick={row['tick']}&node={row['node_id']}"
        compare(row["tick"], CAPACITY, {"node": row["node_id"]}, cc._truth_from_compute_capacity(row),
                "resource_dimensions", ref)
        compare(row["tick"], FAILED, {"task": row["task_id"]}, cc._truth_from_failure(row),
                "migration_failure", ref)
    communication_path = source / "communication_state.jsonl"
    for row in rows(communication_path):
        compare(row["tick"], DEGRADED, {"actor": row["entity_id"], "station": row["station_id"]},
                cc._truth_from_link_degraded(row), "link_thresholds",
                relative(communication_path) + f"#tick={row['tick']}&flow={row['flow_id']}")

    for (tick, predicate, _), item in frozen.items():
        direct = [entry["value"] for entry in item["evidence"] if entry["path"] == "direct_predicate_value"]
        if direct and direct[0] == "out_of_scope":
            expected = wt._expression_truth_value(wt._evaluate_expression(
                templates[predicate]["evaluation_spec"], {"direct_predicate_value": direct[0]}, {}, parameters))
            compare(tick, predicate, item["bindings"], expected, "direct_applicability",
                    relative(GRAPHS / episode / "nodes.parquet") + "#node_id=" + item["node_id"])

    if any(item["predicate"] == COMPROMISED for item in frozen.values()):
        from Dataset.semantic_truth.objective_pipeline import (
            DEFAULT_DOMAIN_PROFILE_PATH, _objective_strict_episode_root,
        )
        capture = ROOT / "aw_data/render_ready_episodes_capture_filtered" / episode
        profile = ds.load_domain_state_profile(DEFAULT_DOMAIN_PROFILE_PATH)
        common = {"model_id": profile["model"]["model_id"], "model_version": profile["model"]["model_version"]}
        with _objective_strict_episode_root(capture) as strict:
            inputs = ds.load_episode_inputs(strict, profile, charging_service_plan_path=strict / "charging_service_plan.json")
            durations = defaultdict(int)
            for frame in inputs.frames:
                tick = frame["tick"]
                if tick not in wt.FORMAL_TICKS:
                    continue
                facilities = ds._facility_entities(ds._current_entities(frame), {"ground_control_station"})
                for row in ds._security_rows(inputs=inputs, profile=profile, common=common,
                        tick=tick, uavs=[], facilities=facilities, communication_by_tick_entity={}, lockout_duration=durations):
                    expected = evaluate_context(templates[COMPROMISED], {"domain": {"security_command": row["values"]}}, parameters)
                    compare(tick, COMPROMISED, {"station": row["subject_id"]}, expected,
                            "station_compromise", relative(capture / "truth_frames.jsonl") + f"#tick={tick}&entity={row['subject_id']};runtime_state_materialization")

    unmatched = Counter()
    for identity, item in frozen.items():
        if item["predicate"] not in TARGETS or identity in seen:
            continue
        if item["value"] in {"true", "false"}:
            raise ValueError(f"Binary published assertion lacks replay authority: {episode}:{identity}")
        unmatched[item["value"]] += 1
    changed = Counter(issue for row in changes.values() for issue in row["issues"])
    return dict(episode=episode, published_selected_assertions=len(frozen), compared_by_issue=dict(compared),
                changed_by_issue=dict(changed), changed_assertions=len(changes),
                governed_parameter_declarations_checked=global_checks,
                unmatched_nonbinary_assertions=dict(unmatched)), list(changes.values())


def pilot_impact(differences: pd.DataFrame, completed: set[str]) -> dict | None:
    config = json.loads((PILOT / "config.json").read_text())
    pilot_episodes = {Path(path).name for path in config["source_paths"]}
    if not pilot_episodes <= completed:
        return None
    episodes = torch.load(PILOT / "episodes.pt", weights_only=False)
    samples = pd.read_parquet(PILOT / "samples.parquet")
    counts = defaultdict(Counter)
    codes = {"false": 1, "true": 2, "unknown": 3, "out_of_scope": 4}
    for profile, episode in episodes.items():
        before = episode["truth"]
        after = before.copy()
        keys = {(predicate, binding_key(dict(bindings))): i for i, (predicate, bindings) in enumerate(episode["keys"])}
        changed = differences[differences.episode == profile + "__seed00"]
        for row in changed.itertuples():
            query = keys[(row.predicate, row.bindings_json)]
            if before[row.tick // 5, query] != codes[row.before]:
                raise ValueError("Pilot projection and frozen assertion disagree")
            after[row.tick // 5, query] = codes[row.after]
        frame_changes = np.any(before != after, axis=1)
        for (split, anchor), group in samples[samples.profile == profile].groupby(["split", "anchor"]):
            index = int(anchor) // 5
            counter = counts[split]
            counter["anchors"] += 1
            counter["existing_queries"] += len(group)
            graph_changed = bool(frame_changes[index-20:index+1].any())
            counter["anchors_with_assertion_changes_in_history"] += int(graph_changed)
            counter["queries_at_those_anchors"] += len(group) if graph_changed else 0
            for row in group.itertuples():
                q = int(row.query)
                current = int(after[index, q])
                future = after[index+1:index+11, q]
                eligible = current in (1, 2)
                valid = eligible and bool(np.isin(future, [1, 2]).all())
                opposite = np.flatnonzero(future == 3-current) if eligible else np.array([], dtype=int)
                label = int(opposite[0]) if valid and len(opposite) else 10 if valid else -1
                counter["queries_with_changed_current_truth"] += int(current != before[index, q])
                counter["queries_losing_current_binary_eligibility"] += int(not eligible)
                counter["queries_with_changed_future_validity"] += int(valid != bool(row.valid))
                counter["queries_with_changed_first_switch_label"] += int(label != int(row.label))
                counter["queries_with_changed_own_history"] += int(np.any(before[index-20:index+1, q] != after[index-20:index+1, q]))
    return dict(scope="Frozen sample index only; new eligible queries and changed numerical inputs are not enumerated.",
                counts={split: dict(value) for split, value in counts.items()},
                model_metrics_status="Historical scores against pre-repair labels; no retraining or corrected-model evaluation performed.")


def write_report(summary: dict) -> None:
    lines = ["# Semantic rule replay", "", "## Scope", "",
             f"Replayed {summary['episodes']} published formal graph partitions using the current production rule functions and preserved source records. Found {summary['changed_assertions']} distinct predicate-instance/tick truth differences across {summary['episodes_with_changes']} partitions.", "",
             "This is a read-only comparison with the published graph population. It does not publish corrected corpus labels or regenerate downstream transitions, events, ARM products or models. Security inputs are reconstructed through the existing strict L0 materialization path. The other rules use the preserved numerical/geometry input records. Source paths and graph node identifiers locate each difference.", "",
             "## Rule effects", "", "| Rule | Compared published assertions | Changed assertions |", "|---|---:|---:|"]
    for issue, title in ISSUES.items():
        if issue == "global_threshold":
            continue
        lines.append(f"| {title} | {summary['compared_by_issue'].get(issue, 0)} | {summary['changed_by_issue'].get(issue, 0)} |")
    lines.extend(["", f"Verified {summary['governed_parameter_declarations_checked']} episode declarations against the unchanged global boundary margin. A conflicting or missing value raises an error and stops replay. Direct-applicability counts cover stored direct `out_of_scope` evidence; zero observed cases do not establish that the old operator was correct.", "",
                  "Unmatched nonbinary assertions retain their published missing/applicability status and are counted explicitly in `impact.json`. A binary assertion without a replay source is an error. Source records outside the published candidate population do not enter the changed-assertion counts. These are sampled truth differences, not independent episodes, training examples, events or causal consequences.", ""])
    if summary["pilot_impact"] is not None:
        lines.extend(["## Effect on the frozen transition pilot", "",
                      "The following counts apply the replayed truth changes to the existing sample index. They do not create corrected training data. New query eligibility and numerical-state changes are outside this count.", "",
                      "| Split | Existing queries | Current truth changed | First-switch label changed | Future validity changed | Queries at anchors with changed graph history |",
                      "|---|---:|---:|---:|---:|---:|"])
        for split, c in summary["pilot_impact"]["counts"].items():
            lines.append(f"| {split} | {c['existing_queries']} | {c['queries_with_changed_current_truth']} | {c['queries_with_changed_first_switch_label']} | {c['queries_with_changed_future_validity']} | {c['queries_at_those_anchors']} |")
        lines.extend(["", "The saved model scores remain historical results against pre-repair labels. Rebuild and verify the label/graph lineage, prepare a new experiment directory, and train/evaluate again before using the scores to choose the next architecture or expand data.", ""])
    lines.extend(["## Reproduce", "", "Run from the repository root:", "", "```bash",
                  "/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python -B -m Dataset.semantic_truth.rule_replay --all",
                  "```", "", "`--episode EPISODE_ID` can select a narrower source population. The output directory contains one current report, `impact.json` with coverage and per-episode counts, and `differences.parquet` with every measured truth difference. The command replaces these derived replay products.", ""])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("--all", action="store_true")
    select.add_argument("--episode", action="append")
    args = parser.parse_args()
    episodes = sorted(path.parent.name for path in SOURCES.glob("*/world_truth_summary.json")) if args.all else sorted(args.episode)
    templates = {row["id"]: row for row in get_core_predicate_templates()}
    parameters = get_governed_parameter_defaults()
    reports = []
    differences = []
    for index, episode in enumerate(episodes, 1):
        report, changed = replay_episode(episode, templates, parameters)
        reports.append(report)
        differences.extend(changed)
        print(json.dumps(dict(completed=index, total=len(episodes), episode=episode,
                              changed_assertions=len(changed)), sort_keys=True), flush=True)
    columns = ["episode", "tick", "predicate", "bindings_json", "before", "after", "issues", "graph_node_id", "source_ref"]
    frame = pd.DataFrame(differences, columns=columns)
    compared = Counter()
    changed = Counter()
    for report in reports:
        compared.update(report["compared_by_issue"])
        changed.update(report["changed_by_issue"])
    summary = dict(scope="Published formal graph partitions; ARM partitions are not included.",
                   episodes=len(reports), episode_ids=episodes,
                   changed_assertions=len(frame), episodes_with_changes=sum(bool(r["changed_assertions"]) for r in reports),
                   compared_by_issue=dict(compared), changed_by_issue=dict(changed),
                   governed_parameter_declarations_checked=sum(r["governed_parameter_declarations_checked"] for r in reports),
                   per_episode=reports, pilot_impact=pilot_impact(frame, set(episodes)))
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT / "differences.parquet", index=False)
    (OUT / "impact.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary)
    print(json.dumps({k: summary[k] for k in ["episodes", "changed_assertions", "episodes_with_changes", "changed_by_issue", "pilot_impact"]}), flush=True)


if __name__ == "__main__":
    main()
