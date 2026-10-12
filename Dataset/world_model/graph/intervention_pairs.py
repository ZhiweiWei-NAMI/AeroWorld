"""Measured factual/intervention comparisons, with explicit admission evidence."""

from __future__ import annotations

import argparse

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import pyarrow.dataset as ds
from torch_geometric.data import HeteroData
from functools import lru_cache

from Dataset.world_model.graph.contract import REPO, declared_arm_samples, iter_jsonl, read_json
from Dataset.world_model.graph.records import DEFAULT_ROOT, load_scope
from Dataset.semantic_simulation.domain_state import OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES
from Dataset.semantic_truth.provenance import without_integrity_metadata
from Dataset.world_model.graph.contract import CONTRACT_VERSION

OUTPUT = REPO / "Dataset/world_model/runs/belief_pairs"
BOOKKEEPING = frozenset({"frame_id", "episode_id", "arm_key", "parent_episode_id"})
PHYSICAL_STATE_FIELDS = frozenset({"tick", "entity_id", "label_class", "pos_enu", "state", "activity_type"}) | frozenset(OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES)


def _comparison_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _comparison_value(v) for k, v in value.items() if k not in BOOKKEEPING}
    if isinstance(value, list):
        return [_comparison_value(v) for v in value]
    return value


def _changes(a: Any, b: Any, path: str = "") -> list[dict]:
    if type(a) is not type(b):
        return [{"path": path, "factual": a, "intervention": b}]
    if isinstance(a, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            child = f"{path}.{key}" if path else key
            if key not in a or key not in b:
                result.append({"path": child, "factual_present": key in a,
                               "intervention_present": key in b,
                               "factual": a.get(key), "intervention": b.get(key)})
            else:
                result.extend(_changes(a[key], b[key], child))
        return result
    if isinstance(a, list):
        if len(a) != len(b):
            return [{"path": path, "factual": a, "intervention": b}]
        return [change for i, (x, y) in enumerate(zip(a, b)) for change in _changes(x, y, f"{path}[{i}]")]
    return [] if a == b else [{"path": path, "factual": a, "intervention": b}]


def _states(root: Path) -> dict[tuple, tuple[int, dict]]:
    result = {}
    for line, row in iter_jsonl(root / "raw/states.jsonl"):
        key = row["tick"], row["entity_id"]
        if key in result:
            raise ValueError(f"duplicate measured state: {root}:{key}")
        result[key] = line, _comparison_value(row)
    return result


def _location(root: Path, name: str, line: int | None = None) -> dict:
    return {"source": (root / name).relative_to(REPO).as_posix(), "line": line}


def _executed_action(rows: list[dict]) -> list[dict]:
    """Compare commands and omissions independently of generated identifiers."""
    result = []
    for row in rows:
        action = {k: v for k, v in row["action"].items() if k not in {
            "action_id", "runtime_state_schedule_profile_id", "runtime_state_schedule_semantics",
            "uav_corridor_validation", "terminal_feasibility"}}
        result.append({"tick": row["tick"], "action": action,
                       "applied": row["result"]["status"] == "ok"})
    return result


def _declared_x_queue_change(factual: Path, intervention: Path, left: dict, right: dict) -> dict:
    """Check changed future definitions against the scripts actually executed."""
    from copy import deepcopy
    from dataclasses import asdict
    from Dataset.tools.l6_v2.engine import EventScriptInterpreter
    from Dataset.tools.x_arm_pipeline import branch_changes

    scripts, interpreters, sources = [], [], []
    for root in (factual, intervention):
        relative = root.relative_to(REPO / "arms")
        script = REPO / "interventions" / relative.parent / (root.name + "_event_script.json")
        config = read_json(script.with_name(root.name + ".yaml"))
        actual = read_json(script)
        original = read_json(REPO / config["sources"]["script"])
        manifest = read_json(root / "manifest.json")
        if branch_changes(original, actual) != manifest["branch_changes"]:
            raise ValueError(f"saved X branch changes differ from the executed script: {script}")
        scripts.append(actual)
        interpreters.append(EventScriptInterpreter(script, episode_id=manifest["episode_id"]))
        sources.append(script.relative_to(REPO).as_posix())
    reduced = [deepcopy(left), deepcopy(right)]
    allowed = []
    for field, identifier in (("event_states", "event_id"), ("trigger_states", "trigger_id")):
        state_maps = [reduced[index]["interpreter"][field] for index in (0, 1)]
        for key in sorted(state_maps[0].keys() ^ state_maps[1].keys()):
            side = 0 if key in state_maps[0] else 1
            initial = asdict(getattr(interpreters[side], field)[key])
            if state_maps[side][key] != initial:
                return {"verified": False, "reason": "changed future definition already has runtime history", "id": key}
            allowed.append({"kind": field, "id": key, "side": ("factual", "intervention")[side],
                            "state": state_maps[side].pop(key), "script": sources[side]})
    remaining = _changes(reduced[0], reduced[1])
    return {"verified": not remaining, "allowed_unfired_definition_changes": allowed,
            "remaining_execution_state_differences": remaining, "executed_script_sources": sources,
            "basis": "x_arm_pipeline.run executes the individually declared branch script; initialized future definitions have no prior firing"}


def compare_pair(factual: Path, intervention: Path) -> dict:
    fm, im = read_json(factual / "manifest.json"), read_json(intervention / "manifest.json")
    if fm["mode"] != "factual_observation" or im["mode"] != "intervention":
        raise ValueError("pair modes are not factual/intervention")
    if fm["episode_id"] != im["episode_id"] or factual.parent != intervention.parent:
        raise ValueError("pair source episode/group differs")
    fw, iw = fm["window"], im["window"]
    if fw["fork_tick"] != iw["fork_tick"] or fw["phase"] != iw["phase"]:
        raise ValueError("pair fork tick/phase differs")
    fork = fw["fork_tick"]
    start, end = max(fw["start_tick"], iw["start_tick"]), min(fw["end_tick"], iw["end_tick"])
    fs, ins = _states(factual), _states(intervention)
    distinct_motion_serialization = factual.parts[-4].startswith("L1-")
    history = []
    representation_differences = []
    state_changes = []
    for key in sorted(fs.keys() | ins.keys()):
        tick, entity = key
        if not start <= tick <= end:
            continue
        f, i = fs.get(key), ins.get(key)
        changes = _changes(f[1] if f else None, i[1] if i else None)
        if changes:
            row = {"tick": tick, "entity_id": entity, "changes": changes,
                   "factual": _location(factual, "raw/states.jsonl", f[0] if f else None),
                   "intervention": _location(intervention, "raw/states.jsonl", i[0] if i else None)}
            physical_changes = changes
            if distinct_motion_serialization:
                projected_f = {k: v for k, v in f[1].items() if k in PHYSICAL_STATE_FIELDS} if f else None
                projected_i = {k: v for k, v in i[1].items() if k in PHYSICAL_STATE_FIELDS} if i else None
                physical_changes = _changes(projected_f, projected_i)
                representation_differences.append({**row, "phase": "pre_fork" if tick <= fork else "post_fork"})
            if physical_changes:
                (history if tick <= fork else state_changes).append({**row, "changes": physical_changes})
    initial_f = read_json(factual / "raw/initial_state.json")
    initial_i = read_json(intervention / "raw/initial_state.json")
    fork_f, fork_i = read_json(factual / "raw/fork_state.json"), read_json(intervention / "raw/fork_state.json")
    if fork_f["tick"] != fork or fork_i["tick"] != fork:
        raise ValueError("saved fork state has a different tick")
    initial_differences = _changes(_comparison_value(initial_f.get("sampled_states", [])),
                                   _comparison_value(initial_i.get("sampled_states", [])))
    fork_differences = _changes(_comparison_value(fork_f.get("sampled_states", [])),
                                _comparison_value(fork_i.get("sampled_states", [])))
    # Full interpreter/queue differences stay visible and restrict admission.
    fork_context = _changes({k: _comparison_value(v) for k, v in fork_f.items() if k != "sampled_states"},
                            {k: _comparison_value(v) for k, v in fork_i.items() if k != "sampled_states"})
    actions = {}
    for label, root in (("factual", factual), ("intervention", intervention)):
        actions[label] = [{"line": line, "row": _comparison_value(row)}
                          for line, row in iter_jsonl(root / "raw/actions.jsonl")]
    prior_f = [r["row"] for r in actions["factual"] if r["row"]["tick"] < fork]
    prior_i = [r["row"] for r in actions["intervention"] if r["row"]["tick"] < fork]
    pre_action_changes = _changes(prior_f, prior_i)
    future_f = [r for r in actions["factual"] if fork <= r["row"]["tick"] <= end]
    future_i = [r for r in actions["intervention"] if fork <= r["row"]["tick"] <= end]
    raw_action_changes = _changes([r["row"] for r in future_f], [r["row"] for r in future_i])
    action_changes = _changes(_executed_action([r["row"] for r in future_f]),
                             _executed_action([r["row"] for r in future_i]))
    predicate_streams = {}
    for label, root in (("factual", factual), ("intervention", intervention)):
        rows = {}
        for line, row in iter_jsonl(root / "objective/predicate_truth.jsonl"):
            if not fork < row["tick"] <= end:
                continue
            key = row["tick"], row["predicate_id"], json.dumps(row["bindings"], sort_keys=True)
            if key in rows:
                raise ValueError(f"duplicate predicate in pair: {root}:{key}")
            rows[key] = line, row["value"]
        predicate_streams[label] = rows
    predicate_changes = []
    for key in sorted(predicate_streams["factual"].keys() | predicate_streams["intervention"].keys()):
        f, i = predicate_streams["factual"].get(key), predicate_streams["intervention"].get(key)
        if f is None or i is None or f[1] != i[1]:
            predicate_changes.append({"tick": key[0], "predicate_id": key[1], "bindings": json.loads(key[2]),
                "factual_value": f[1] if f else None, "intervention_value": i[1] if i else None,
                "factual": _location(factual, "objective/predicate_truth.jsonl", f[0] if f else None),
                "intervention": _location(intervention, "objective/predicate_truth.jsonl", i[0] if i else None),
                "comparison_status": "measured_truth_difference" if f and i and f[1] in {"true", "false"} and i[1] in {"true", "false"}
                                     else "observation_status_or_scope_difference"})
    outcomes = {}
    for label, root in (("factual", factual), ("intervention", intervention)):
        outcomes[label] = [{"record": row, "source": _location(root, "objective/event_outcomes.jsonl", line)}
                           for line, row in iter_jsonl(root / "objective/event_outcomes.jsonl")]
    prehistory_equal = not (history or initial_differences or fork_differences or pre_action_changes)
    queue_equal = not fork_context
    queue_review = {"verified": queue_equal}
    if not queue_equal and factual.parts[-4].startswith("X"):
        queue_review = _declared_x_queue_change(factual, intervention,
            {k: _comparison_value(v) for k, v in fork_f.items() if k != "sampled_states"},
            {k: _comparison_value(v) for k, v in fork_i.items() if k != "sampled_states"})
    feedback = [r["row"].get("result", {}).get("status") for r in future_f + future_i]
    feedback_complete = bool(feedback) and set(feedback) <= {"ok", "omitted_by_explicit_intervention", "omitted_by_configuration"}
    eligibility = ("admitted_measured_window" if prehistory_equal and queue_review["verified"] and action_changes and feedback_complete
                   else "excluded_pre_history_difference" if not prehistory_equal
                   else "excluded_execution_state_difference" if not queue_review["verified"]
                   else "excluded_missing_action_feedback" if not feedback_complete
                   else "no_executed_action_difference")
    return without_integrity_metadata({"version": "intervention-evidence-2", "graph_version": CONTRACT_VERSION, "episode_id": fm["episode_id"],
        "factual_sample": "arm:" + factual.relative_to(REPO / "arms").as_posix(),
        "intervention_sample": "arm:" + intervention.relative_to(REPO / "arms").as_posix(),
        "factual_graph": (DEFAULT_ROOT / "arm" / factual.relative_to(REPO / "arms")).relative_to(REPO).as_posix(),
        "intervention_graph": (DEFAULT_ROOT / "arm" / intervention.relative_to(REPO / "arms")).relative_to(REPO).as_posix(),
        "window": {"start_tick": start, "fork_tick": fork, "end_tick": end, "phase": fw["phase"]},
        "branch_end_ticks": {"factual": fw["end_tick"], "intervention": iw["end_tick"]},
        "fork_sources": [_location(factual, "raw/fork_state.json"), _location(intervention, "raw/fork_state.json")],
        "initial_sources": [_location(factual, "raw/initial_state.json"), _location(intervention, "raw/initial_state.json")],
        "prehistory_equal": prehistory_equal, "fork_queue_equal": queue_equal,
        "queue_change_review": queue_review, "action_feedback_complete": feedback_complete,
        "action_feedback_counts": dict(Counter(feedback)),
        "eligibility": eligibility, "prehistory_differences": history,
        "raw_representation_differences": representation_differences,
        "prehistory_comparison": {
            "exact_fields": sorted(PHYSICAL_STATE_FIELDS) if distinct_motion_serialization else "all source state fields",
            "full_initial_and_fork_engine_states_compared": True,
            "velocity_and_heading_basis": "formal motion is recomputed from rounded positions; engine raw motion is serialized directly"
                                          if distinct_motion_serialization else "same recorded fields on both actual branches",
            "producer_sources": ["Dataset/tools/convert_to_render_ready.py:align_dynamic_entity_motion_to_final_positions",
                                 "Dataset/tools/l1_window.py:run_engine"] if distinct_motion_serialization else ["same scoped engine serialization on both branches"],
            "numeric_tolerance_used": False},
        "initial_state_differences": initial_differences, "fork_state_differences": fork_differences,
        "pre_fork_action_differences": pre_action_changes, "fork_context_differences": fork_context,
        "executed_actions": {"factual": future_f, "intervention": future_i},
        "executed_action_differences": action_changes, "raw_action_record_differences": raw_action_changes,
        "state_changes": state_changes,
        "predicate_changes": predicate_changes, "outcomes": outcomes,
        "known_predicate_difference_count": sum(r["comparison_status"] == "measured_truth_difference" for r in predicate_changes),
        "observed_effect": "changed" if any(r.get("factual") and r.get("intervention") for r in state_changes)
                          or any(r["comparison_status"] == "measured_truth_difference" for r in predicate_changes)
                          else "observation_coverage_differs" if state_changes or predicate_changes else "same_measured_state_and_predicates",
        "effect_scope": "actual common window and recorded entities/predicates only",
        "background_policy": (im.get("scope") or {}).get("background_policy"),
        "limitations": ["background feedback outside the actual branch is not simulated",
                        "scheduled roots express design intent; measured differences remain paired evidence",
                        "unequal branch endpoints are separate censoring boundaries"]})


def pair_pyg(pair: dict) -> HeteroData:
    graph = HeteroData()
    graph["SharedFork"].num_nodes = 1
    graph["SharedFork"].tick = torch.tensor([pair["window"]["fork_tick"]])
    for side, kind in (("factual", "FactualAction"), ("intervention", "InterventionAction")):
        rows = pair["executed_actions"][side]
        graph[kind].num_nodes = len(rows)
        graph[kind].tick = torch.tensor([r["row"]["tick"] for r in rows], dtype=torch.long)
        graph[kind].applied = torch.tensor([r["row"]["result"]["status"] == "ok" for r in rows], dtype=torch.bool)
        graph["SharedFork", "followed_by", kind].edge_index = torch.tensor([(0, i) for i in range(len(rows))], dtype=torch.long).reshape(-1, 2).T.contiguous()
    for key, kind in (("state_changes", "StateDifference"), ("predicate_changes", "PredicateDifference")):
        rows = pair[key]
        graph[kind].num_nodes = len(rows)
        graph[kind].tick = torch.tensor([r["tick"] for r in rows], dtype=torch.long)
        graph["SharedFork", "measured_difference", kind].edge_index = torch.tensor([(0, i) for i in range(len(rows))], dtype=torch.long).reshape(-1, 2).T.contiguous()
    for side, kind in (("factual", "FactualOutcome"), ("intervention", "InterventionOutcome")):
        graph[kind].num_nodes = len(pair["outcomes"][side])
        graph["SharedFork", "has_outcome", kind].edge_index = torch.tensor([(0, i) for i in range(graph[kind].num_nodes)], dtype=torch.long).reshape(-1, 2).T.contiguous()
    if "task_comparisons" in pair:
        from Dataset.world_model.graph.event_tasks import task_definitions
        codes = {r["id"]: i for i, r in enumerate(task_definitions())}
        queries = {"current_hidden_predicate": 0, "future_onset": 1, "future_terminal": 2}
        rows = pair["task_comparisons"]
        store = graph["PairedQuery"]
        store.num_nodes = len(rows)
        store.tick = torch.tensor([r["anchor_tick"] for r in rows], dtype=torch.long)
        store.task_id = torch.tensor([codes[r["task_id"]] for r in rows], dtype=torch.long)
        store.query_id = torch.tensor([queries[r["query"]] for r in rows], dtype=torch.long)
        store.measured_common_window = torch.tensor([r["measured_common_window"] for r in rows])
        store.paired_effect_valid = torch.tensor([r["status"] in {"paired_valid_equal", "paired_valid_difference"}
                                                 for r in rows])
        for side in ("factual", "intervention"):
            store[side + "_valid"] = torch.tensor([bool(r[side] and r[side]["target_valid"]) for r in rows])
            store[side + "_y"] = torch.tensor([float(r[side]["target_value"]) if r[side] and r[side]["target_valid"] else 0. for r in rows])
            for flag in ("applicable", "target_missing", "observation_missing", "censored", "left_censored"):
                store[side + "_" + flag] = torch.tensor([bool(r[side] and r[side][flag]) for r in rows])
        graph["SharedFork", "aligned_query", "PairedQuery"].edge_index = torch.tensor(
            [(0, i) for i in range(len(rows))], dtype=torch.long).reshape(-1, 2).T.contiguous()
    graph.evidence = pair
    return graph


@lru_cache(maxsize=8)
def _paired_targets(path_text: str) -> dict:
    path = Path(path_text)
    rows = {}
    for line, row in iter_jsonl(path):
        key = row["task_id"], row["query"], row["anchor_tick"], json.dumps(row["bindings"], sort_keys=True)
        if key in rows:
            raise ValueError(f"duplicate paired target: {path}:{key}")
        rows[key] = {"target_valid": row["target_valid"], "target_value": row["target_value"],
            "target_missing": row["target_missing"], "observation_missing": row["observation_missing"],
            "applicable": row["applicable"], "censored": row["censored"],
            "observed_until_tick": row["observed_until_tick"],
            "left_censored": row.get("left_censored", False),
            "positive_ticks": row.get("positive_ticks", []), "evidence_ids": row["evidence_ids"],
            "horizon_ticks": row["horizon_ticks"],
            "target_source": str(path.relative_to(REPO)), "target_line": line}
    return rows


def _paired_event_records(partition: Path) -> list[dict]:
    records = []
    expression = ds.field("node_type").isin(["EventOccurrence", "EventOutcome"])
    for batch in ds.dataset(partition / "nodes.parquet", format="parquet").scanner(
            filter=expression, batch_size=8192).to_batches():
        for stored in batch.to_pylist():
            a = json.loads(stored["attrs_json"])
            records.append({"graph_node": stored["node_id"], "node_type": stored["node_type"],
                "event_type": stored["category_id"], "tick": stored["tick"],
                "bindings": a.get("bindings", {}), "status": a.get("status"),
                "censored": a.get("censored", False), "target_missing": a.get("target_missing", False),
                "onset_evidence_status": a.get("onset_evidence_status", "within_saved_window"),
                "source_file": stored["source_file"], "source_line": stored["source_line"]})
    return records


def attach_task_comparisons(*, graph_root: Path = DEFAULT_ROOT,
                            target_root: Path = REPO / "Dataset/world_model/runs/belief_event_tasks",
                            output: Path = OUTPUT) -> dict:
    """Extend already reviewed fork proofs with aligned, masked task outcomes.

    Equal valid labels are retained. Missing, inapplicable and censored labels
    do not become an event difference or a zero outcome. A target whose horizon
    passes the common window end stays outside the comparison even when one side
    observed a positive before that end: the other side's truncated horizon has
    no executed feedback to compare against.
    """
    graph_root, target_root, output = graph_root.resolve(), target_root.resolve(), output.resolve()
    totals = Counter()
    for path in sorted(output.rglob("pair.json")):
        pair = read_json(path)
        if pair["version"] != "intervention-evidence-2" or pair["graph_version"] != CONTRACT_VERSION:
            raise ValueError(f"paired fork evidence version differs: {path}")
        by_side, events = {}, {}
        for side in ("factual", "intervention"):
            sample = pair[side + "_sample"]
            relative = Path("arm") / sample.split(":", 1)[1]
            partition = graph_root / relative
            scope = load_scope(partition)
            if scope["sample_key"] != sample or scope["contract_version"] != CONTRACT_VERSION:
                raise ValueError(f"pair graph differs: {sample}")
            by_side[side] = _paired_targets(str(target_root / relative / "targets.jsonl"))
            events[side] = _paired_event_records(partition)
        comparisons, counts = [], Counter()
        keys = by_side["factual"].keys() | by_side["intervention"].keys()
        for task, query, tick, bindings in sorted(keys):
            if not pair["window"]["fork_tick"] <= tick <= pair["window"]["end_tick"]:
                continue
            key = task, query, tick, bindings
            f, i = by_side["factual"].get(key), by_side["intervention"].get(key)
            prediction_end = tick if query == "current_hidden_predicate" else tick + max(
                r["horizon_ticks"] for r in (f, i) if r is not None)
            measured = prediction_end <= pair["window"]["end_tick"]
            status = ("pair_not_admitted" if pair["eligibility"] != "admitted_measured_window" else
                      "outside_measured_common_window" if not measured else
                      "paired_valid_equal" if f and i and f["target_valid"] and i["target_valid"]
                      and f["target_value"] == i["target_value"] else
                      "paired_valid_difference" if f and i and f["target_valid"] and i["target_valid"] else
                      "one_side_not_in_candidate_scope" if not f or not i else "masked_target_comparison")
            counts[status] += 1
            comparisons.append({"task_id": task, "query": query, "anchor_tick": tick,
                "prediction_end_tick": prediction_end, "measured_common_window": measured,
                "bindings": json.loads(bindings), "status": status, "factual": f, "intervention": i})
        pair.update(task_comparison_population="matching task, query, tick and participant bindings after the shared fork within the measured common window",
                    task_comparison_counts=dict(counts), task_comparisons=comparisons,
                    canonical_event_records=events,
                    task_effect_limit="A valid aligned label difference belongs to this admitted run pair and observation window; it is not a universal causal claim")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(path)
        torch.save(pair_pyg(pair), path.parent / "pair_evidence.pt")
        totals.update(counts)
        totals["pairs"] += 1
    return dict(totals)


def build_pairs(output: Path = OUTPUT) -> dict:
    groups = defaultdict(list)
    for branch in declared_arm_samples():
        groups[branch.parent].append(branch)
    entries, statuses, effects = [], Counter(), Counter()
    for group, branches in sorted(groups.items()):
        factual = [p for p in branches if read_json(p / "manifest.json")["mode"] == "factual_observation"]
        if len(factual) != 1:
            raise ValueError(f"group must have one actual factual: {group}")
        for branch in branches:
            if branch == factual[0]:
                continue
            pair = compare_pair(factual[0], branch)
            relative = branch.relative_to(REPO / "arms")
            destination = output / relative
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "pair.json").write_text(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
            torch.save(pair_pyg(pair), destination / "pair_evidence.pt")
            statuses[pair["eligibility"]] += 1
            effects[pair["observed_effect"]] += 1
            entries.append({"sample_key": pair["intervention_sample"], "factual_sample": pair["factual_sample"],
                            "eligibility": pair["eligibility"], "observed_effect": pair["observed_effect"],
                            "pair": (destination / "pair.json").relative_to(REPO).as_posix()})
            print(f"pair {relative}: {pair['eligibility']}", flush=True)
    result = {"version": "intervention-evidence-2", "graph_version": CONTRACT_VERSION, "pair_count": len(entries),
              "eligibility_counts": dict(statuses), "measured_difference_counts": dict(effects), "pairs": entries}
    (REPO / "design/belief/intervention_pairs.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fork", "targets", "all"), default="all")
    args = parser.parse_args()
    if args.phase in {"fork", "all"}:
        result = build_pairs()
        print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, ensure_ascii=False))
    if args.phase in {"targets", "all"}:
        print(json.dumps(attach_task_comparisons(), ensure_ascii=False))
