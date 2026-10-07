"""Derive business predicates and events from the current saved execution ledgers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from Dataset.semantic_simulation.compute_comm import (
    _artifact_rows, _build_event_rows, _predicate_matrix_tick_row,
    _compute_predicate_values, _communication_predicate_values,
    _build_predicate_rows, _simulation_common, load_compute_comm_profile,
    load_episode_inputs,
)
from Dataset.semantic_truth.provenance import canonical_json, without_integrity_metadata

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json"
FILES = {"compute_predicate_truth.jsonl": "predicate_truth",
         "compute_predicate_matrix.jsonl": "predicate_truth_matrix",
         "compute_events.jsonl": "events"}


def manifest_entry(records: dict[str, list[dict]]) -> dict:
    return {"producer": "Dataset/semantic_truth/compute_event_truth.py:derive",
        "state_sources": ["compute_state.jsonl", "communication_state.jsonl"],
        "source_path_base": "objective episode directory",
        "profile_source": PROFILE.relative_to(REPO).as_posix(),
        "record_counts": {name.removesuffix(".jsonl"): len(rows) for name, rows in records.items()},
        "basis": "current saved execution states; predicates and events recomputed by the current business rules",
        "matrix_population": "only scopes with a same-tick execution record; absent scopes are not converted to out_of_scope",
        "recovery_scope": "event rules declare rising onsets; no recovery is inferred without a declared terminal rule"}


def derive(episode_root: Path, compute_rows: list[dict], communication_rows: list[dict],
           profile_path: Path = PROFILE) -> dict[str, list[dict]]:
    profile = load_compute_comm_profile(profile_path)
    inputs = load_episode_inputs(episode_root, profile)
    common, _ = _simulation_common(inputs, profile, profile_path)
    predicates = _build_predicate_rows(compute_rows, communication_rows, profile, common)
    scopes = defaultdict(list)
    for rows, kind, field, values in ((compute_rows, "compute_node", "node_id", _compute_predicate_values),
                                     (communication_rows, "uav", "entity_id", _communication_predicate_values)):
        seen = set()
        for row in rows:
            key = row["tick"], row[field]
            # _build_compute_rows emits a row only while its entity is an active
            # simulation candidate. Communication also writes that condition.
            active = True if kind == "compute_node" else row["scope_active"]
            if key in seen or type(active) is not bool:
                raise ValueError(f"execution matrix scope is ambiguous: {kind}:{key}")
            seen.add(key)
            scopes[row["tick"]].append({"scope_type": kind, "scope_entity_id": row[field],
                "scope_active": active, "workload_active": active,
                "predicate_values": values(row)})
    matrix = [_predicate_matrix_tick_row(common, tick, sorted(scopes[tick],
                key=lambda r: (r["scope_type"], r["scope_entity_id"]))) for tick in sorted(scopes)]
    for row in matrix:
        row["source_refs"] = [f"compute_state.jsonl#tick={row['tick']}",
                              f"communication_state.jsonl#tick={row['tick']}"]
    events = _build_event_rows(predicates, profile, common)
    return {"compute_predicate_truth.jsonl": list(_artifact_rows(predicates)),
            "compute_predicate_matrix.jsonl": list(_artifact_rows(matrix)),
            "compute_events.jsonl": list(_artifact_rows(events))}


def prepare(episode: str) -> dict:
    from Dataset.world_model.graph.contract import iter_jsonl
    root = REPO / "aw_data/objective_semantic_truth" / episode
    capture = REPO / "aw_data/render_ready_episodes_capture_filtered" / episode
    records = derive(capture, [r for _, r in iter_jsonl(root / "compute_state.jsonl")],
                     [r for _, r in iter_jsonl(root / "communication_state.jsonl")])
    return {"episode": episode, "files": {name: "".join(canonical_json(without_integrity_metadata(row)) + "\n" for row in rows)
                                         for name, rows in records.items()},
            "manifest_entry": manifest_entry(records)}


def publish(prepared: dict) -> dict:
    from Dataset.world_model.graph.contract import read_json
    episode = prepared["episode"]
    root = REPO / "aw_data/objective_semantic_truth" / episode
    for name, text in prepared["files"].items():
        temporary = root / (name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(root / name)
    manifest = read_json(root / "manifest.json")
    manifest["business_event_truth"] = prepared["manifest_entry"]
    manifest_path = root / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return {"episode": episode, "counts": prepared["manifest_entry"]["record_counts"]}


if __name__ == "__main__":
    from Dataset.world_model.graph.contract import declared_samples
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", action="append")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    episodes = args.episode if args.episode else declared_samples()[0]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending, remaining = {}, iter(episodes)
        for episode in list(episodes)[:args.workers]:
            pending[pool.submit(prepare, next(remaining))] = episode
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                print(json.dumps(publish(future.result())), flush=True)
                episode = next(remaining, None)
                if episode is not None:
                    pending[pool.submit(prepare, episode)] = episode
