"""Time-aligned predicate labels and observed event sequences (no model inputs).

Run from the repository root using the runtime specified in CONTRACT.md.
World graph deltas are a persistent state stream; supplement predicates are
sampled rows and MUST NOT be carried across missing samples or entity changes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

from Dataset.semantic_truth.provenance import open_artifact_text, stable_identifier
from Dataset.world_model.data.provenance import SOURCE_SPLITS, resolve_source_split

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "Dataset/world_model/configs/state_distribution_5s.json"
HORIZONS = [5, 10, 20, 30, 50]
DEV_PROFILES = {"L2-2_v2__seed00", "L4-5_v1__seed00", "L5-3_v1__seed00"}
PREDICATES = {
    "safety.aircraft_pair_below_separation": "separation",
    "airspace.airspace_corridor_over_capacity": "corridor",
    "airspace.aircraft_occupies_airspace_corridor": "corridor",
    "communication.handover_in_progress": "communication",
    "communication.link_degraded": "communication",
    "communication.link_unavailable": "communication",
    "communication.data_flow_latency_above_threshold": "communication",
}
COMM = {p for p, family in PREDICATES.items() if family == "communication"}
ROLES = {
    "communication.handover_in_progress": ("handover",),
    "communication.link_degraded": ("actor", "station"),
    "communication.link_unavailable": ("actor", "station"),
    "communication.data_flow_latency_above_threshold": ("flow",),
}
KNOWN = {"true", "false"}
VALUES = KNOWN | {"unknown", "out_of_scope"}


def read_json(path):
    with open_artifact_text(path) as handle:
        return json.load(handle)


def rows(path, needles=()):
    """Filter before parsing large archives; still read via the zstd-aware API."""
    with open_artifact_text(path) as handle:
        for line in handle:
            if line.strip() and (not needles or any(n in line for n in needles)):
                yield json.loads(line)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dump_line(handle, value):
    handle.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def issue(audit, name, detail):
    entry = audit.setdefault("predicate_issues", {}).setdefault(name, {"count": 0, "examples": []})
    entry["count"] += 1
    if len(entry["examples"]) < 10:
        entry["examples"].append(detail)


def comm_tuple(pid, bindings):
    # Exactly the verified world_truth._candidate_from_record identifier rule.
    return stable_identifier("world_predicate_tuple", pid,
                             [(role, bindings[role]) for role in ROLES[pid]])


def load_world(directory, audit):
    base = read_json(directory / "world_truth_graph_base.json")
    summary = read_json(directory / "world_truth_summary.json")
    step = base["governed_parameters"]["formal_step_ticks"]
    start = base["tick"]
    end = start + (summary["formal_tick_count"] - 1) * step
    if step != 5 or start != 0 or not base["candidate_set_complete"]:
        raise ValueError("Expected complete world graph at tick 0 and five-tick formal cadence")
    timelines, bindings, current = {}, {}, {}

    def add(body, tick):
        key = (body["predicate_id"], body["tuple_id"])
        value = body["value"]
        if value not in VALUES:
            raise ValueError(f"Undeclared truth value: {value}")
        if key in bindings and bindings[key] != body["bindings"]:
            raise ValueError(f"Tuple changed bindings: {key}")
        bindings[key] = body["bindings"]
        timelines.setdefault(key, {})[tick] = value
        current[key] = value

    for body in base["initial_assertions"]:
        if body["predicate_id"] in PREDICATES:
            add(body, start)
    delta_count, op_counts = 0, Counter()
    for delta in rows(directory / "world_truth_graph_deltas.jsonl"):
        tick = delta["tick"]
        if tick % step or tick > end:
            raise ValueError(f"Non-formal graph tick {tick}")
        delta_count += 1
        for op in delta["operations"]:
            body = op["assertion"] if op["operation"] == "add_predicate_assertion" else op
            if body["predicate_id"] not in PREDICATES:
                continue
            kind = op["operation"]
            key = (body["predicate_id"], body["tuple_id"])
            op_counts[kind] += 1
            if kind == "add_predicate_assertion":
                if key in current:
                    raise ValueError(f"Adding existing tuple {key}")
                add(body, tick)
            elif kind == "remove_predicate_assertion":
                if current.pop(key) != op["from_value"]:
                    raise ValueError(f"Remove from_value disagrees: {key}")
                timelines[key][tick] = None  # Absence, not a false predicate.
            elif kind == "set_predicate_value":
                if current[key] != op["from_value"]:
                    raise ValueError(f"Delta from_value disagrees: {key}")
                add({**body, "value": body["to_value"]}, tick)
            elif kind == "refresh_predicate_evidence":
                if current[key] != op["value"]:
                    raise ValueError(f"Evidence refresh disagrees: {key}")
            else:
                raise ValueError(f"Unsupported graph operation: {kind}")
    # Dense on the source cadence only. Absent intervals stay absent.
    dense = {}
    for key, changes in timelines.items():
        state, samples = None, {}
        for tick in range(start, end + 1, step):
            if tick in changes:
                state = changes[tick]
            if state is not None:
                samples[tick] = state
        dense[key] = samples
    audit["world"] = {"start_tick": start, "end_tick": end, "cadence_ticks": step,
                      "cadence_seconds": 0.5, "delta_batches": delta_count,
                      "selected_operation_counts": dict(op_counts)}
    return dense, bindings, start, end


def load_comm(directory, audit):
    from Dataset.semantic_simulation import compute_comm as producer

    values, bindings = {}, {}
    all_ticks, counts, versions = set(), Counter(), Counter()
    for row in rows(directory / "predicate_truth.jsonl", tuple(COMM)):
        pid = row["predicate_id"]
        if pid not in COMM:
            continue
        tick, value = row["tick"], row["value"]
        if tick % 5 or value not in VALUES:
            raise ValueError(f"Invalid communication sample: {pid}@{tick}: {value}")
        key = (pid, comm_tuple(pid, row["bindings"]))
        samples = values.setdefault(key, {})
        if tick in samples:
            issue(audit, "duplicate_comm_sample", {"predicate_id": pid, "tick": tick, "bindings": row["bindings"]})
            if samples[tick] != value:
                raise ValueError(f"Conflicting communication rows: {key}@{tick}")
        samples[tick] = value
        bindings[key] = row["bindings"]
        counts[pid] += 1
        versions[row["model_version"]] += 1
        all_ticks.add(tick)
    # Recompute from the actual simulation state with the existing producer API.
    state_index, joined = {}, set()
    for state in rows(directory / "communication_state.jsonl"):
        tick = state["tick"]
        key = (tick, state["entity_id"])
        if key in state_index:
            raise ValueError(f"Duplicate communication state {key}")
        state_index[key] = state
        derived = producer._communication_predicate_values(state)
        state_bindings = {
            "communication.handover_in_progress": {"handover": state["handover_id"]},
            "communication.data_flow_latency_above_threshold": {"flow": state["flow_id"]},
            "communication.link_degraded": {"actor": state["entity_id"], "station": state["station_id"]},
            "communication.link_unavailable": {"actor": state["entity_id"], "station": state["station_id"]},
        }
        for pid, bound in state_bindings.items():
            k = (pid, comm_tuple(pid, bound))
            previous = values.get(k, {}).get(tick)
            joined.add((k, tick))
            if previous != derived[pid]:
                issue(audit, "comm_predicate_vs_state", {"predicate_id": pid, "tick": tick,
                      "bindings": bound, "predicate_value": previous, "computed_value": derived[pid]})
                # Existing rows belong to the recorded producer run. Today's
                # implementation can differ; never silently relabel that run.
                # A missing row can be restored from an actual state calculation.
                if previous is None:
                    values.setdefault(k, {})[tick] = derived[pid]
                    bindings[k] = bound
    unjoined = sum((key, tick) not in joined for key, samples in values.items() for tick in samples)
    audit["communication"] = {"cadence_ticks": 5, "cadence_seconds": 0.5,
        "sample_counts": dict(counts), "state_rows": len(state_index),
        "source_model_versions": dict(versions),
        "samples_without_state_row": unjoined, "start_tick": min(all_ticks), "end_tick": max(all_ticks),
        "observed_tick_gcd": math.gcd(*(b-a for a,b in zip(sorted(all_ticks), sorted(all_ticks)[1:]))) }
    return values, bindings, state_index


def compare_predicates(directory, world, labels, audit):
    counts = Counter()
    for row in rows(directory / "predicate_truth.jsonl", tuple(PREDICATES)):
        pid = row["predicate_id"]
        if pid not in PREDICATES:
            continue
        key, tick = (pid, row["tuple_id"]), row["tick"]
        counts[pid] += 1
        graph_value = world.get(key, {}).get(tick)
        if graph_value != row["value"]:
            issue(audit, "projection_vs_world", {"predicate_id": pid, "tuple_id": key[1],
                "tick": tick, "projection": row["value"], "world": graph_value})
    for key, samples in world.items():
        if key[0] in COMM:
            for tick, value in samples.items():
                other = labels.get(key, {}).get(tick)
                if other != value:
                    category = "world_comm_sample_absent_from_supplement" if other is None else "world_vs_comm_value_difference"
                    issue(audit, category, {"predicate_id": key[0], "tuple_id": key[1],
                        "tick": tick, "world": value, "supplement": other})
    for key, samples in labels.items():
        if key[0] in COMM:
            for tick in samples:
                if tick not in world.get(key, {}):
                    issue(audit, "supplement_comm_sample_absent_from_world", {
                        "predicate_id": key[0], "tuple_id": key[1], "tick": tick})
    transitions = Counter()
    for row in rows(directory / "predicate_transitions.jsonl"):
        if row["predicate_id"] not in PREDICATES:
            continue
        key = (row["predicate_id"], row["tuple_id"])
        transitions[key[0]] += 1
        samples = labels.get(key, {})
        if samples.get(row["from_tick"]) != row["from_value"] or samples.get(row["to_tick"]) != row["to_value"]:
            issue(audit, "projection_transition_vs_authority", {"transition_id": row["transition_id"],
                "predicate_id": key[0], "from_tick": row["from_tick"], "to_tick": row["to_tick"],
                "source_values": [row["from_value"], row["to_value"]],
                "authority_values": [samples.get(row["from_tick"]), samples.get(row["to_tick"])]})
    audit["projection"] = {"selected_truth_rows": dict(counts), "selected_transition_rows": dict(transitions)}


def event_family(event):
    typ = event["event_type_id"]
    if typ == "safety.aircraft_pair_conflict_event":
        return "separation"
    if "corridor" in typ:
        return "corridor"
    if typ.startswith("communication."):
        return "communication"
    return "other"


def normalized_event(row, source):
    return {"event_type_id": row["event_type_id"], "event_id": row["event_id"],
            "tick": row["detection_tick"] if source == "objective" else row["trigger_tick"],
            "bindings": row["bindings"], "source": source}


def reconcile_events(episode, objective, supplement, canonical, audit):
    """Audit all event views; select primary records, never duplicate projections."""
    notes = defaultdict(list)
    raw_objective = list(rows(objective / "event_occurrences.jsonl"))
    raw_supplement = list(rows(supplement / "events.jsonl"))
    primary, selected, signatures = {}, {}, {}
    for source, records in [("objective", raw_objective), ("communication_supplement", raw_supplement)]:
        for row in records:
            ev = normalized_event(row, source)
            eid = ev["event_id"]
            if source == "objective":
                primary[eid] = row
                if row["trigger_tick"] != row["detection_tick"]:
                    notes["trigger_detection_tick_difference"].append({"event_id": eid,
                        "trigger_tick": row["trigger_tick"], "detection_tick": row["detection_tick"]})
            if eid in selected:
                notes["duplicate_event_id"].append({"event_id": eid, "source": source,
                    "identical": selected[eid] == ev})
                if selected[eid] != ev:
                    raise ValueError(f"Conflicting authoritative event id {eid}")
                continue
            sig = (ev["event_type_id"], ev["tick"], tuple(sorted(ev["bindings"].items())))
            if sig in signatures:
                notes["same_event_different_id"].append({"retained": signatures[sig], "discarded": eid,
                    "event_type_id": ev["event_type_id"], "tick": ev["tick"], "bindings": ev["bindings"]})
                continue
            signatures[sig] = eid
            selected[eid] = ev
    archive, archive_count = {}, 0
    for row in rows(canonical, ('"event"',)):
        if row.get("record_kind") != "event":
            continue
        archive_count += 1
        raw = row["fields"]["archive.event_occurrences"]
        eid = raw["event_id"]
        if eid in archive:
            notes["duplicate_canonical_id"].append({"event_id": eid, "tick": row["tick"]})
        archive[eid] = raw
        if row["tick"] != raw["detection_tick"] or row["id"] != eid:
            notes["canonical_wrapper_difference"].append({"event_id": eid, "canonical_id": row["id"],
                "canonical_tick": row["tick"], "detection_tick": raw["detection_tick"]})
        if eid in primary:
            changed = {k: {"objective": primary[eid].get(k), "canonical": raw.get(k)}
                       for k in set(primary[eid]) | set(raw) if primary[eid].get(k) != raw.get(k)}
            if changed:
                notes["canonical_payload_difference"].append({"event_id": eid, "fields": changed})
        else:
            candidates = [r for r in raw_objective if r["event_type_id"] == raw["event_type_id"]
                          and r["bindings"] == raw["bindings"]]
            notes["canonical_only_event"].append({"event_id": eid, "tick": row["tick"],
                "event_type_id": raw["event_type_id"], "bindings": raw["bindings"],
                "current_candidates": [{"event_id": r["event_id"], "tick": r["detection_tick"]} for r in candidates]})
    notes["objective_missing_from_canonical"] = sorted(set(primary) - set(archive))
    outcomes, outcome_ids = list(rows(objective / "event_outcomes.jsonl")), set()
    for row in outcomes:
        eid = row["event_id"]
        if eid in outcome_ids:
            notes["duplicate_outcome_event_id"].append(eid)
        outcome_ids.add(eid)
        if eid not in primary:
            notes["outcome_without_occurrence"].append(eid)
        elif row["event_type_id"] != primary[eid]["event_type_id"]:
            notes["outcome_type_difference"].append(eid)
    notes["occurrence_without_outcome"] = sorted(set(primary) - outcome_ids)
    graph_ids, graph_count = set(), 0
    for graph in rows(objective / "semantic_truth_graphs.jsonl"):
        for node in graph["nodes"]:
            if node["kind"] != "event_occurrence":
                continue
            graph_count += 1
            eid = node["source_record_id"]
            graph_ids.add(eid)
            if eid not in primary:
                notes["graph_only_event"].append({"event_id": eid, "window": graph["window"]})
                continue
            changed = {k: {"objective": primary[eid].get(k), "projection": node[k]}
                       for k in ("event_type_id", "bindings", "start_tick", "end_tick", "trigger_tick", "event_phases")
                       if k in node and primary[eid].get(k) != node[k]}
            if changed:
                notes["graph_event_projection_difference"].append({"event_id": eid,
                    "window": graph["window"], "fields": changed})
            leaked = [p["tick"] for p in node.get("event_phases", []) if p["tick"] > graph["window"]["end_tick"]]
            if leaked:
                notes["graph_future_phase_metadata"].append({"event_id": eid,
                    "window_end_tick": graph["window"]["end_tick"], "future_phase_ticks": leaked})
    notes["occurrence_missing_from_graph_projection"] = sorted(set(primary) - graph_ids)
    # Script relations have a separate id vocabulary and are not inferred causal labels.
    chain_dir = ROOT / "aw_data/event_chains" / episode
    relation_path = chain_dir / "event_chain_relations.jsonl"
    manifest_path = chain_dir / "event_chain_manifest.json"
    chain_ids, relations = set(), []
    if relation_path.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        for root in manifest["roots"]:
            chain_ids.add(root["event_id"])
        relations = list(rows(relation_path))
        for rel in relations:
            for end in ("source_endpoint", "target_endpoint"):
                endpoint = rel[end]
                chain_ids.update(endpoint.get("candidate_event_ids", []))
                if endpoint.get("event_id"):
                    chain_ids.add(endpoint["event_id"])
        audit["script_chain"] = {"relation_rows": len(relations), "declared_relation_count": manifest["relation_count"],
            "runtime_validation_status": manifest["runtime_validation_status"],
            "strict_edge_count": sum(bool(r["runtime_strict_edge_eligible"]) for r in relations),
            "script_event_ids": sorted(chain_ids), "objective_id_overlap": sorted(chain_ids & set(primary)),
            "runtime_audit": manifest["runtime_audit"]}
        if len(relations) != manifest["relation_count"]:
            notes["chain_relation_count_difference"].append({"actual": len(relations), "declared": manifest["relation_count"]})
    else:
        notes["missing_chain_files"] = [str(p.relative_to(ROOT)) for p in (relation_path, manifest_path) if not p.exists()]
        audit["script_chain"] = {"relation_rows": 0, "script_event_ids": []}
    audit["events"] = {"objective_rows": len(raw_objective), "supplement_rows": len(raw_supplement),
        "canonical_rows": archive_count, "canonical_same_id_copies": len(set(primary) & set(archive)),
        "outcome_rows": len(outcomes), "graph_event_projections": graph_count,
        "repeated_graph_projections": graph_count - len(graph_ids),
        "graph_unique_event_ids": len(graph_ids), "retained_events": len(selected),
        "discrepancies": dict(notes), "discrepancy_counts": {k: len(v) for k, v in notes.items()},
        "detected_events_without_script_id_join": sorted(set(selected) - chain_ids)}
    return sorted(selected.values(), key=lambda e: (e["tick"], e["event_type_id"], e["event_id"]))


def counts_template():
    return {"queries": 0, "state_at_cutoff": Counter(), "positives_per_horizon": Counter(),
            "known_per_horizon": Counter(), "unknown_per_horizon": Counter(),
            "transition_queries": 0, "fully_observed_transition_windows": 0,
            "unique_transitions": 0, "event_target_occurrences": 0, "unique_bound_events": 0}


def process_episode(args):
    episode, config, output, shard_dir = args
    started = time.perf_counter()
    split = resolve_source_split(episode)
    if split not in {"TRAIN", "VALID"}:
        raise ValueError(f"Refusing to read TEST labels: {episode}")
    objective = ROOT / "aw_data/objective_semantic_truth" / episode
    supplement = ROOT / "aw_data/compute_comm_supplement" / episode
    audit = {"episode": episode, "split": split, "dev": episode in DEV_PROFILES,
             "sources": {"world": str(objective.relative_to(ROOT)),
                         "communication": str(supplement.relative_to(ROOT)), "canonical": config["canonical"]}}
    world, world_bindings, start, end = load_world(objective, audit)
    comm, comm_bindings, state_index = load_comm(supplement, audit)
    labels = {k: v for k, v in world.items() if k[0] not in COMM}
    labels.update(comm)
    bindings = {**world_bindings, **comm_bindings}
    compare_predicates(objective, world, labels, audit)
    events = reconcile_events(episode, objective, supplement, ROOT / config["canonical"], audit)
    # Index observed event entities using only bindings declared by the event.
    # Communication ontology individuals are mapped to their real state at event time.
    aliases = defaultdict(dict)
    for (tick, actor), state in state_index.items():
        for field in ("flow_id", "handover_id", "session_id", "message_id", "retransmission_id", "transmission_attempt_id", "channel_allocation_id"):
            aliases[tick][state[field]] = state
    for ev in events:
        entities = set(ev["bindings"].values())
        if ev["source"] == "communication_supplement":
            for entity in list(entities):
                state = aliases.get(ev["tick"], {}).get(entity)
                if state is not None:
                    entities.update((state["entity_id"], state["station_id"], state["flow_id"], state["handover_id"]))
        ev["entities"] = entities
    stats = {p: counts_template() for p in PREDICATES}
    transition_sets, bound_event_sets = defaultdict(set), defaultdict(set)
    queries_by_tick = defaultdict(list)
    excluded = Counter()
    for key, samples in labels.items():
        pid, _ = key
        for tick, value in samples.items():
            if value in KNOWN and samples.get(tick-5) in KNOWN and samples[tick-5] != value:
                transition_sets[pid].add((key[1], tick))
            if tick < start + 50 or tick > end:
                continue
            if value == "out_of_scope":
                excluded[pid] += 1
            else:
                queries_by_tick[tick].append(key)
    query_path = Path(shard_dir) / f"{episode}.jsonl"
    chain_path = Path(shard_dir) / f"chains_{episode}.jsonl"
    chain_rows, chain_positive = 0, 0
    with query_path.open("w") as qhandle, chain_path.open("w") as chandle:
        for cutoff in range(start+50, end+1, 5):
            future_events = [e for e in events if cutoff < e["tick"] <= cutoff+50]
            history_events = [event_target(e) for e in events if e["tick"] <= cutoff]
            chain = {"episode": episode, "split": split, "dev": episode in DEV_PROFILES, "cutoff": cutoff,
                     "history_events": history_events, "next_events": [event_target(e) for e in future_events],
                     "censored_at": min(cutoff+50, end)}
            dump_line(chandle, chain)
            chain_rows += 1
            chain_positive += bool(future_events)
            for key in sorted(queries_by_tick[cutoff]):
                pid, tuple_id = key
                samples = labels[key]
                state = samples[cutoff]
                future = {str(h): contract_value(samples.get(cutoff+h)) if cutoff+h <= end else "unknown" for h in HORIZONS}
                first_transition, censored = None, cutoff
                if state in KNOWN:
                    previous = state
                    for tick in range(cutoff+5, min(cutoff+50, end)+1, 5):
                        value = samples.get(tick)
                        if value not in KNOWN:
                            break
                        censored = tick
                        if first_transition is None and value != previous:
                            first_transition = tick
                        previous = value
                entities = set(bindings[key].values())
                targets = [event_target(e) for e in future_events if entities <= e["entities"]]
                row = {"episode": episode, "split": split, "dev": episode in DEV_PROFILES,
                    "family": PREDICATES[pid], "cutoff": cutoff, "predicate_id": pid,
                    "tuple_id": tuple_id, "bindings": bindings[key], "state_at_cutoff": state,
                    "horizons": HORIZONS, "future_state": future,
                    "first_transition_tick": first_transition, "transition_censored_at": censored,
                    "event_targets": targets, "source": {
                        "authority": "communication_supplement" if pid in COMM else "world_truth_graph",
                        "cadence_ticks": 5, "cadence_seconds": 0.5, "tick_seconds": 0.1,
                        "cutoff_source_tick": cutoff, "label_window_end_tick": min(cutoff+50, end),
                        "event_tick_policy": "detection_tick_for_objective_trigger_tick_for_supplement"}}
                dump_line(qhandle, row)
                count = stats[pid]
                count["queries"] += 1
                count["state_at_cutoff"][state] += 1
                count["transition_queries"] += first_transition is not None
                count["fully_observed_transition_windows"] += censored == cutoff+50
                count["event_target_occurrences"] += len(targets)
                bound_event_sets[pid].update(t["event_id"] for t in targets)
                for h, value in future.items():
                    count["positives_per_horizon"][h] += value == "true"
                    count["known_per_horizon"][h] += value in KNOWN
                    count["unknown_per_horizon"][h] += value == "unknown"
    for pid in PREDICATES:
        stats[pid]["unique_transitions"] = len(transition_sets[pid])
        stats[pid]["unique_bound_events"] = len(bound_event_sets[pid])
    audit["excluded_out_of_scope_cutoff_samples"] = dict(excluded)
    audit["counts_by_predicate"] = stats
    family_events = Counter(event_family(e) for e in events)
    audit["events_by_family"] = {f: family_events[f] for f in ("separation", "corridor", "communication", "other")}
    family_counts = {f: {} for f in ("separation", "corridor", "communication")}
    for pid, count in stats.items():
        merge_counts(family_counts[PREDICATES[pid]], count)
    for family, count in family_counts.items():
        count["unique_bound_events"] = len(set().union(*(bound_event_sets[p] for p in PREDICATES if PREDICATES[p] == family)))
        count["unique_events"] = family_events[family]
    family_counts["other"] = {"unique_events": family_events["other"]}
    audit["counts_by_family"] = family_counts
    audit["events_by_type"] = dict(Counter(e["event_type_id"] for e in events))
    audit["chain_queries"] = chain_rows
    audit["chain_queries_with_next_event"] = chain_positive
    audit["elapsed_seconds"] = round(time.perf_counter()-started, 3)
    write_json(Path(output) / "reconciliation" / f"{episode}.json", audit)
    return audit


def contract_value(value):
    return value if value in KNOWN else "unknown"


def event_target(event):
    # Do not copy outcomes, end ticks or future phase metadata into history.
    return {k: event[k] for k in ("event_type_id", "tick", "event_id")}


def merge_counts(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            merge_counts(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", help="Configured TRAIN/VALID episode ids; default all 24")
    parser.add_argument("--workers", type=int, default=8, help="Episode processes, 1..16 (default 8)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    config = read_json(CONFIG)["episodes"]
    episodes = sorted(set(args.episodes or config))
    # Enforce the split boundary BEFORE opening any episode labels.
    for episode in episodes:
        if resolve_source_split(episode) not in {"TRAIN", "VALID"}:
            parser.error(f"TEST labels are excluded: {episode}")
        if episode not in config:
            parser.error(f"Episode not in configured first-loop cohort: {episode}")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    audits = []
    with tempfile.TemporaryDirectory(prefix=".episode_shards_", dir=args.out) as shard_dir:
        jobs = [(ep, config[ep], str(args.out), shard_dir) for ep in episodes]
        if args.workers == 1:
            results = map(process_episode, jobs)
            for audit in results:
                audits.append(audit)
                print(json.dumps({"episode": audit["episode"], "seconds": audit["elapsed_seconds"],
                    "queries": sum(c["queries"] for c in audit["counts_by_predicate"].values())}), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
                for audit in pool.map(process_episode, jobs):
                    audits.append(audit)
                    print(json.dumps({"episode": audit["episode"], "seconds": audit["elapsed_seconds"],
                        "queries": sum(c["queries"] for c in audit["counts_by_predicate"].values())}), flush=True)
        for split in ("TRAIN", "VALID"):
            for prefix in ("", "chains_"):
                with (args.out / f"{prefix}{split}.jsonl").open("wb") as destination:
                    for audit in audits:
                        if audit["split"] == split:
                            with (Path(shard_dir) / f"{prefix}{audit['episode']}.jsonl").open("rb") as source:
                                shutil.copyfileobj(source, destination)
    grouped, issues, predicates_by_split, event_source_counts = {}, Counter(), {}, Counter()
    for audit in audits:
        # TRAIN includes dev rows; FIT and DEV expose the training/model-selection separation.
        groups = [audit["split"]]
        if audit["split"] == "TRAIN":
            groups.append("DEV" if audit["dev"] else "FIT")
        for group in groups:
            for family, count in audit["counts_by_family"].items():
                merge_counts(grouped.setdefault(group, {}).setdefault(family, {}), count)
            for pid, count in audit["counts_by_predicate"].items():
                merge_counts(predicates_by_split.setdefault(group, {}).setdefault(pid, {}), count)
        event_source_counts.update({k: v for k,v in audit["events"].items() if type(v) is int})
        issues.update(audit["events"]["discrepancy_counts"])
        issues.update({k: v["count"] for k,v in audit.get("predicate_issues", {}).items()})
    support = {}
    for group, predicates in predicates_by_split.items():
        support[group] = {}
        for pid, c in predicates.items():
            positive = c["positives_per_horizon"].get("50", 0)
            negative = c["known_per_horizon"].get("50", 0) - positive
            support[group][pid] = {"positive_at_50": positive, "negative_at_50": negative,
                "unique_transitions": c["unique_transitions"],
                "state_support": "single_class_or_empty" if min(positive, negative) == 0 else
                                 "sparse_minority_under_20" if min(positive, negative) < 20 else "both_classes",
                "transition_support": "none" if c["unique_transitions"] == 0 else
                                      "sparse_under_20" if c["unique_transitions"] < 20 else "at_least_20"}
    manifest = {"schema_version": "1.0.0", "episodes": episodes, "episode_count": len(episodes),
        "dev_profiles": sorted(DEV_PROFILES), "source_splits": {ep: SOURCE_SPLITS[ep] for ep in episodes},
        "split_authority": "Dataset/world_model/data/provenance.py:SOURCE_SPLITS/resolve_source_split",
        "horizons": HORIZONS, "tick_seconds": 0.1, "cutoff_stride_ticks": 5, "history_ticks": 50,
        "workers": min(args.workers, len(episodes)), "elapsed_seconds": round(time.perf_counter()-started, 3),
        "counts_by_split_family": grouped, "counts_by_split_predicate": predicates_by_split,
        "support_assessment": support, "reconciliation_counts": dict(issues),
        "event_source_counts": dict(event_source_counts),
        "episode_audits": [{"episode": a["episode"], "split": a["split"], "dev": a["dev"],
            "path": f"reconciliation/{a['episode']}.json", "seconds": a["elapsed_seconds"],
            "chain_queries": a["chain_queries"], "chain_queries_with_next_event": a["chain_queries_with_next_event"]} for a in audits],
        "policies": {
            "predicate_authority": "Complete world graph base+delta for separation/corridor; persisted sampled communication supplement predicates for communication. Current producer calculations from communication_state are audited against persisted labels; disagreements retain the recorded-run labels. Semantic predicate projections are audited, not used to fill candidate sets.",
            "alignment": "Source ticks already denote 0.1 seconds: both authorities sample every 5 ticks (0.5 seconds). Queries/horizons fall on that grid; no rescaling or invented 0.1-second measurements. World values persist until delta/removal; communication values require an exact sampled row.",
            "query_selection": "All tuples present at each cutoff 50..episode_end, every 5 ticks; out_of_scope cutoffs excluded. Candidate membership depends only on state at cutoff, never future positivity. History requirement is episode history, not 50 ticks of tuple existence.",
            "missing": "Absent/out_of_scope future samples and post-episode horizons map to contract unknown, never false. Explicit unknown cutoff rows retained. Missing communication predicate samples with actual state rows are computed with the existing producer and audited; existing recorded-run values are preserved on calculation disagreements.",
            "transitions": "First adjacent known true/false change in (cutoff,cutoff+50]. Stop at first missing/unknown/out_of_scope sample; censor at preceding known sample. Unknown cutoff censors at cutoff. Unique transition counts cover whole episode, query counts cover eligible cutoffs.",
            "events": "Objective occurrences authoritative over canonical and graph copies. All supplement events additionally retained, including compute events for chain context. Exact type/tick/bindings duplicates deduplicated, objective first. Different ticks are not merged by proximity. Event tick is objective detection_tick (canonical importer convention), supplement trigger_tick. No outcome or future phase metadata enters history/targets.",
            "event_binding": "All query entity ids must be present in event bindings or communication-state identity relations at that event tick. No join using a shared station alone. Unique family events count event types; bound-event counts can span event types.",
            "chains": "One chronological observed sequence per episode/cutoff, all past detected events and next 50 ticks. Script chain relations are separately audited: script IDs are not detected-event IDs and no inferred causal edge is fabricated. Sequences are not causal-chain annotations.",
            "reconciliation": "Canonical-only/stale payloads discarded in favor of current objective source; outcomes joined by event_id only for audit; graph projections deduplicated, window-clipped end ticks and future phase metadata excluded. Every event discrepancy is recorded in episode audits; predicate discrepancies have counts and ten examples per category.",
            "dev": "TRAIN file includes dev=true profiles; exclude them when fitting. FIT/DEV counts provided separately. VALID is held-out evaluation; TEST files never opened.",
            "support": "Support flags use descriptive 20-example/transition thresholds, not a validation-selected inclusion rule; retain all seven predicates to make sparse and single-class supervision explicit.",
            "bounds": "Per-query state/event/history selection is bounded by cutoff+50; source files are scanned once per episode for indexing and source reconciliation. Those whole-episode audits are not predictor inputs."}}
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps({"out": str(args.out), "episodes": len(episodes), "seconds": manifest["elapsed_seconds"],
        "queries": sum(c["queries"] for a in audits for c in a["counts_by_predicate"].values())}), flush=True)


if __name__ == "__main__":
    main()
