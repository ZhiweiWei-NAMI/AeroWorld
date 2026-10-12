"""Read isolated PyG observations and supervision as separate training objects."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from Dataset.world_model.graph.contract import CONTRACT_VERSION, REPO, read_json
from Dataset.world_model.graph.records import DEFAULT_ROOT, load_scope
from Dataset.world_model.graph.prediction import INPUT_VERSION, OBSERVATIONS, DEFAULT_OBSERVATION, ObservationCandidates, isolate_records, observation_nodes, to_pyg
from Dataset.world_model.graph.observation_logs import OUTPUT as LOG_ROOT
from Dataset.world_model.graph.records import partition_for

OUTPUT = REPO / "Dataset/world_model/runs/belief_training"
DEFAULT_HOLDOUT = ("L4-3_v2", "L4-4_v2")
SPLIT_PROTOCOL = "profile-development-1"
SPLITS = ("development", "profile_test")


def source_split(scope: dict, heldout_profiles: tuple[str, ...] = DEFAULT_HOLDOUT) -> str:
    """Seeds are not independent units: seed00/01/02 replay identical observed
    inputs. Evaluation folds are built from profiles inside "development"."""
    return "profile_test" if scope["profile"] in heldout_profiles else "development"


def _index_inputs(partition: Path, target: Path, parameters: dict) -> dict:
    paths = [partition / "scope.parquet", partition / "nodes.parquet", target,
             Path(__file__), Path(__file__).with_name("prediction.py"),
             Path(__file__).with_name("observation_logs.py")]
    if parameters["observation"].endswith("_logs"):
        logs = partition_for(load_scope(partition), LOG_ROOT)
        paths.extend([logs / "metadata.json", logs / "observations.parquet"])
    return {**parameters, "graph_version": CONTRACT_VERSION, "input_version": INPUT_VERSION,
        "split_protocol": SPLIT_PROTOCOL,
        "artifacts": [{"path": str(p.relative_to(REPO)), "bytes": p.stat().st_size,
                       "modified_ns": p.stat().st_mtime_ns} for p in paths]}


def build_index(graph_root: Path = DEFAULT_ROOT, *,
                target_root: Path = REPO / "Dataset/world_model/runs/belief_event_tasks",
                output: Path = OUTPUT, task_ids: tuple[str, ...] = (),
                observation: str = DEFAULT_OBSERVATION, history_ticks: int = 100,
                heldout_profiles: tuple[str, ...] = DEFAULT_HOLDOUT) -> dict:
    """Index prefix-known queries, including invalid targets with separate masks.

    Future label validity never selects the input population. Missing or
    truncated supervision remains an indexed query with target_valid false.
    """
    if observation not in OBSERVATIONS or history_ticks < 0:
        raise ValueError("undeclared observation condition")
    graph_root, target_root, output = graph_root.resolve(), target_root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    groups, coverage, totals = {}, [], Counter()
    for target_path in sorted(target_root.rglob("targets.jsonl")):
        partition = graph_root / target_path.parent.relative_to(target_root)
        destination = output / target_path.parent.relative_to(target_root)
        inputs = _index_inputs(partition, target_path, {"task_ids": list(task_ids),
            "observation": observation, "history_ticks": history_ticks,
            "heldout_profiles": list(heldout_profiles)})
        marker = destination / "metadata.json"
        if marker.is_file():
            metadata = read_json(marker)
            if (metadata.get("artifact_inputs") == inputs and
                    (not metadata["counts"].get("indexed_queries") or (destination / "index.parquet").is_file())):
                group, split = metadata["source_group"], metadata["split"]
                if group in groups and groups[group] != split:
                    raise ValueError(f"same-source branches cross training splits: {group}")
                groups[group] = split
                coverage.append(metadata)
                totals.update(metadata["counts"])
                continue
        scope = load_scope(partition)
        if scope["contract_version"] != CONTRACT_VERSION:
            raise ValueError(f"training partition version differs: {partition}")
        split, group = source_split(scope, heldout_profiles), scope["episode_id"]
        if group in groups and groups[group] != split:
            raise ValueError(f"same-source branches cross training splits: {group}")
        groups[group] = split
        nodes = observation_nodes(partition, observation=observation)
        candidates = ObservationCandidates(nodes, observation)
        views, entries, counts = {}, [], Counter()
        with target_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                text = stream.readline()
                if not text:
                    break
                target = json.loads(text)
                if task_ids and target["task_id"] not in task_ids:
                    continue
                tick = target["anchor_tick"]
                if tick not in views:
                    views[tick] = candidates.entities(max(scope["start_tick"], tick - history_ticks), tick)
                if not set(target["bindings"].values()) <= views[tick]:
                    counts["candidate_not_known_under_observation_condition"] += 1
                    continue
                entries.append({"target_offset": offset, "anchor_tick": tick,
                                "task_id": target["task_id"], "query": target["query"]})
                counts["indexed_queries"] += 1
                counts["valid_targets"] += int(target["target_valid"])
                counts["missing_targets"] += int(target["target_missing"])
                counts["censored_targets"] += int(target["censored"])
                counts["censored_observed_positive"] += int(target["censored"] and target["target_valid"])
                counts["source_graph_observation_missing"] += int(target["observation_missing"])
                counts["left_censored"] += int(target.get("left_censored", False))
                counts["inapplicable"] += int(not target["applicable"])
        destination.mkdir(parents=True, exist_ok=True)
        if entries:
            pq.write_table(pa.Table.from_pylist(entries), destination / "index.parquet", compression="zstd")
        elif (destination / "index.parquet").exists():
            (destination / "index.parquet").unlink()
        metadata = {"sample_key": scope["sample_key"], "source_group": group, "split": split,
            "profile": scope["profile"], "graph_partition": str(partition.relative_to(REPO)),
            "targets": str(target_path.relative_to(REPO)), "counts": dict(counts),
            "index": str((destination / "index.parquet").relative_to(output)),
            "graph_version": CONTRACT_VERSION, "input_version": INPUT_VERSION,
            "artifact_inputs": inputs}
        (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False) + "\n")
        coverage.append(metadata)
        totals.update(counts)
        print(json.dumps({"training_index_completed": len(coverage), "sample": scope["sample_key"],
                          "counts": dict(counts)}), flush=True)
    result = {"graph_version": CONTRACT_VERSION, "input_version": INPUT_VERSION,
        "split_protocol": SPLIT_PROTOCOL, "observation": observation, "history_ticks": history_ticks, "task_ids": list(task_ids),
        "heldout_profiles": list(heldout_profiles), "source_group_splits": groups,
        "population": "queries whose subjects are known from the observed prefix; invalid, missing and censored targets remain masked entries",
        "counts": dict(totals), "partitions": coverage}
    (output / "scope.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


class EventBeliefDataset(Dataset):
    """Return (input_graph, target, trace); train only with target_valid.

    Use a collator that batches input_graph with PyG Batch.from_data_list.
    Neither target nor trace is attached to input_graph.
    """

    def __init__(self, root: Path = OUTPUT, *, split: str = "development",
                 profiles: tuple[str, ...] = (), exclude_profiles: tuple[str, ...] = ()):
        """profiles keeps only those profiles; exclude_profiles drops them.

        Pass at most one of them, e.g. exclude a held-out scenario's profiles
        for a training fold and select the same profiles for its evaluation.
        """
        self.root = root.resolve()
        self.scope = read_json(self.root / "scope.json")
        if (self.scope["graph_version"] != CONTRACT_VERSION or self.scope["input_version"] != INPUT_VERSION
                or self.scope.get("split_protocol") != SPLIT_PROTOCOL):
            raise ValueError("training index requires rebuilding for the current contracts")
        if split not in SPLITS:
            raise ValueError("undeclared training split")
        if profiles and exclude_profiles:
            raise ValueError("select profiles or exclude profiles, not both")
        known = {metadata["profile"] for metadata in self.scope["partitions"]}
        unknown = (set(profiles) | set(exclude_profiles)) - known
        if unknown:
            raise ValueError(f"unknown profiles: {sorted(unknown)}")
        self.entries = []
        for metadata in self.scope["partitions"]:
            if metadata["split"] != split or not metadata["counts"].get("indexed_queries", 0):
                continue
            if (profiles and metadata["profile"] not in profiles) or metadata["profile"] in exclude_profiles:
                continue
            for row in pq.read_table(self.root / metadata["index"]).to_pylist():
                self.entries.append((metadata, row))

    def __len__(self):
        return len(self.entries)

    @lru_cache(maxsize=2)
    def _observations(self, partition: str):
        return observation_nodes(REPO / partition, observation=self.scope["observation"])

    def __getitem__(self, index):
        metadata, entry = self.entries[index]
        with (REPO / metadata["targets"]).open("rb") as stream:
            stream.seek(entry["target_offset"])
            target = json.loads(stream.readline())
        if (target["sample_key"] != metadata["sample_key"] or target["anchor_tick"] != entry["anchor_tick"]
                or target["task_id"] != entry["task_id"] or target["query"] != entry["query"]):
            raise ValueError("target/index changed; rebuild the affected training index")
        target = {**target, "source_graph_observation_missing": target["observation_missing"],
                  "observation_missing": False, "observation_condition": self.scope["observation"]}
        partition = REPO / metadata["graph_partition"]
        scope = load_scope(partition)
        query = {key: target[key] for key in ("task_id", "query", "bindings", "horizon_ticks")}
        view = isolate_records(self._observations(metadata["graph_partition"]),
            cutoff_tick=target["anchor_tick"],
            start_tick=max(scope["start_tick"], target["anchor_tick"] - self.scope["history_ticks"]),
            observation=self.scope["observation"], query=query)
        return to_pyg(view), target, {**view["trace"], "sample_key": metadata["sample_key"],
                                     "source_group": metadata["source_group"], "split": metadata["split"]}


def paired_observations(pair: dict, comparison: dict, *, graph_root: Path = DEFAULT_ROOT,
                        observation: str = DEFAULT_OBSERVATION, history_ticks: int = 100):
    """Return two isolated inputs and keep measured paired supervision outside.

    Native identities and fork evidence stay in traces. A comparison beyond
    the common feedback window cannot supervise an intervention effect.
    """
    if pair["graph_version"] != CONTRACT_VERSION or pair["eligibility"] != "admitted_measured_window":
        raise ValueError("paired observations require an admitted current fork pair")
    tick = comparison["anchor_tick"]
    if not pair["window"]["fork_tick"] <= tick <= pair["window"]["end_tick"]:
        raise ValueError("paired query lies outside the measured common window")
    inputs, traces = {}, {}
    for side in ("factual", "intervention"):
        if comparison[side] is None:
            raise ValueError(f"paired query is absent from the {side} candidate scope")
        sample = pair[side + "_sample"]
        partition = graph_root / "arm" / sample.split(":", 1)[1]
        scope = load_scope(partition)
        query = {"task_id": comparison["task_id"], "query": comparison["query"],
                 "bindings": comparison["bindings"], "horizon_ticks": comparison[side]["horizon_ticks"]}
        view = isolate_records(observation_nodes(partition, start=max(scope["start_tick"], tick-history_ticks), end=tick, observation=observation),
            cutoff_tick=tick, start_tick=max(scope["start_tick"], tick-history_ticks),
            observation=observation, query=query)
        inputs[side], traces[side] = to_pyg(view), view["trace"]
    return inputs, comparison, {"sides": traces, "fork_tick": pair["window"]["fork_tick"],
                                "effect_scope": pair["effect_scope"], "limitations": pair["limitations"]}


def collate_queries(items):
    """Keep supervision and source references outside the batched input graph."""
    from torch_geometric.data import Batch
    inputs, targets, traces = zip(*items)
    valid = torch.tensor([r["target_valid"] for r in targets], dtype=torch.bool)
    values = torch.tensor([float(r["target_value"]) if r["target_valid"] else 0. for r in targets])
    batch = Batch.from_data_list(inputs, exclude_keys=["input_metadata", "query_roles"])
    batch.query_metadata = [r.input_metadata["query"] for r in inputs]
    return batch, {"value": values, "valid": valid, "records": list(targets)}, list(traces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target-root", type=Path, default=REPO / "Dataset/world_model/runs/belief_event_tasks")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--observation", choices=OBSERVATIONS, default=DEFAULT_OBSERVATION)
    args = parser.parse_args()
    report = build_index(args.graph_root, target_root=args.target_root, output=args.output,
                         task_ids=tuple(args.task), observation=args.observation)
    print(json.dumps({"partitions": len(report["partitions"]), "counts": report["counts"]}))
