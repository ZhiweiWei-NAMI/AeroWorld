"""Fast TRAIN/dev/VALID loop for the two Task B predicate predictors.

Example (from repository root):
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 \
    Dataset/world_model/runtimes/qwen35/bin/python -m \
    Dataset.world_model.predicate_stage.train_predicate --model temporal \
    --output /mnt/data2/weizhiwei/AERO_WORLD/Dataset/world_model/runs/predicate_stage/runs/task_b_temporal

Limits define reproducible role-specific cohorts. --full-valid exports every
eligible VALID row (the two recorded-version-mismatched link predicates are
excluded). --eval-only --checkpoint runs no fitting or model selection.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from Dataset.semantic_truth.core_semantic_registry import load_core_semantic_registry
from Dataset.semantic_truth.provenance import stable_identifier
from Dataset.semantic_simulation.compute_comm import _communication_predicate_values
from Dataset.world_model.data.provenance import resolve_source_split
from Dataset.world_model.model.causal_input import cell, fit_observation_scales, load_index
from Dataset.world_model.model.query_head import fit_input_scale
from Dataset.world_model.model.train_state import source_provenance
from .qwen_predicate import (
    DEFAULT_QWEN, EXCLUDED_PREDICATES, HORIZONS, PREDICATES, TRUTH, QwenPredicateModel, probability_loss, probabilities,
)
from .temporal_baseline import TemporalBaseline, TemporalEpisodeSource, history_features

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = Path("/mnt/data2/weizhiwei/AERO_WORLD/Dataset/world_model/runs/predicate_stage")
DEV_PROFILES = {"L2-2_v2__seed00", "L4-5_v1__seed00", "L5-3_v1__seed00"}
COMM_PREDICATES = {p for p, family in PREDICATES.items() if family == "communication"}


def log(**record):
    print(json.dumps(record, allow_nan=False), flush=True)


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role_of(row):
    return "dev" if row["dev"] else "train" if row["split"] == "TRAIN" else "valid"


def row_key(row):
    return tuple(row[key] for key in ("episode", "cutoff", "predicate_id", "tuple_id"))


def read_supervision(directory, splits=("TRAIN", "VALID")):
    rows = []
    seen = set()
    for split in splits:
        with (directory / f"{split}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["predicate_id"] in EXCLUDED_PREDICATES:
                    continue
                if resolve_source_split(row["episode"]) != split or row["split"] != split:
                    raise ValueError("supervision split disagrees with source authority")
                if row["dev"] != (row["episode"] in DEV_PROFILES):
                    raise ValueError("supervision dev flag disagrees with contract")
                if tuple(row["horizons"]) != HORIZONS or row["state_at_cutoff"] not in TRUTH:
                    raise ValueError("supervision horizon/truth contract mismatch")
                if row["cutoff"] < 50 or row["cutoff"] % 5:
                    raise ValueError("supervision cutoff lacks the required history/cadence")
                if row["family"] != PREDICATES[row["predicate_id"]]:
                    raise ValueError("supervision predicate family mismatch")
                key = row_key(row)
                if key in seen:
                    raise ValueError(f"duplicate supervision key: {key}")
                seen.add(key)
                rows.append(row)
    return rows


def selected_rows(rows, args):
    """Profile round-robin windows and predicate round-robin queries; no label sampling."""
    selected = []
    for role in ("train", "dev", "valid"):
        # Changing one role's budget cannot advance another role's RNG stream.
        rng = random.Random(f"predicate-cohort-v2:{args.seed}:{role}")
        by_episode = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if role_of(row) == role:
                if role == "train" and not has_training_target(row, getattr(args, "event_head", False)):
                    continue
                by_episode[row["episode"]][row["cutoff"]].append(row)
        episodes = sorted(by_episode)
        rng.shuffle(episodes)
        cutoffs = {ep: list(sorted(by_episode[ep])) for ep in episodes}
        for values in cutoffs.values():
            rng.shuffle(values)
        windows = []
        while any(cutoffs.values()):
            for episode in episodes:
                if cutoffs[episode]:
                    windows.append(by_episode[episode][cutoffs[episode].pop()])
        full_valid = role == "valid" and getattr(args, "full_valid", False)
        limit = 0 if full_valid else getattr(args, role + "_windows")
        query_limit = getattr(args, role + "_queries_per_window", None)
        query_limit = args.queries_per_window if query_limit is None else query_limit
        if full_valid:
            query_limit = 0
        for group in windows[:limit] if limit else windows:
            predicates = defaultdict(list)
            for row in group:
                predicates[row["predicate_id"]].append(row)
            for values in predicates.values():
                rng.shuffle(values)
            picked = []
            while any(predicates.values()) and (not query_limit or len(picked) < query_limit):
                for predicate in sorted(predicates):
                    if predicates[predicate] and (not query_limit or len(picked) < query_limit):
                        picked.append(predicates[predicate].pop())
            selected.extend(picked)
    return selected


def has_training_target(row, event_head=False):
    """End-of-episode/all-unknown rows supply no loss, but remain in VALID export."""
    return (any(value in ("true", "false") for value in row["future_state"].values())
            or (row["state_at_cutoff"] != "unknown" and
                (row["first_transition_tick"] is not None or row["transition_censored_at"] > row["cutoff"]))
            or (event_head and (bool(row["event_targets"]) or
                               row["source"]["label_window_end_tick"] >= row["cutoff"] + 50)))


def reconcile_causal_communication(index, communication):
    """Use recorded communication truths; calculate only genuinely absent rows.

    The objective graph's stale communication predicates are replaced with
    recorded sample-cadence assertions, with the actual producer state used
    only for missing recorded samples. Disappearing
    tuples are removed, not carried through gaps. SUMO and other graph relations
    remain the existing load_index replay.
    """
    templates = {t["id"]: t for t in load_core_semantic_registry()["templates"]}
    recorded = {}
    with communication.with_name("predicate_truth.jsonl").open() as stream:
        for line in stream:
            if not any(pid in line for pid in COMM_PREDICATES):
                continue
            row = json.loads(line)
            if row["predicate_id"] in COMM_PREDICATES and index.window_start <= row["tick"] <= index.cutoff:
                key = row["tick"], row["predicate_id"], tuple(sorted(row["bindings"].items()))
                recorded[key] = row["value"]
    by_tick = defaultdict(list)
    with communication.open() as stream:
        for line in stream:
            row = json.loads(line)
            if index.window_start <= row["tick"] <= index.cutoff:
                by_tick[row["tick"]].append(row)
    index.semantic_records = [r for r in index.semantic_records
                              if "relation" not in r or r["relation"]["predicate_id"] not in COMM_PREDICATES | EXCLUDED_PREDICATES]
    index.bound_communication_states = {}
    active = {}
    for tick in range(index.window_start, index.cutoff + 1, 5):
        current = {}
        for row in by_tick[tick]:
            values = _communication_predicate_values(row)
            bindings = {
                "communication.handover_in_progress": {"handover": row["handover_id"]},
                "communication.data_flow_latency_above_threshold": {"flow": row["flow_id"]},
                "communication.link_degraded": {"actor": row["entity_id"], "station": row["station_id"]},
                "communication.link_unavailable": {"actor": row["entity_id"], "station": row["station_id"]},
            }
            for pid in sorted(COMM_PREDICATES):
                roles = templates[pid]["argument_roles"]
                bound = bindings[pid]
                tuple_id = stable_identifier("world_predicate_tuple", pid, [(r["key"], bound[r["key"]]) for r in roles])
                body = {"predicate_id": pid, "tuple_id": tuple_id,
                        "assertion_id": stable_identifier("world_predicate_assertion", pid, tuple_id),
                        "bindings": bound,
                        "binding_ontology_classes": {r["key"]: "world:" + r["class"].removeprefix("world:") for r in roles},
                        "value": (recorded[(tick, pid, tuple(sorted(bound.items())))]
                                  if (tick, pid, tuple(sorted(bound.items()))) in recorded else values[pid]),
                        "observations": [],
                        "truth_state_update_tick": tick, "evidence_update_tick": tick}
                key = pid, tuple_id
                if key in current and current[key]["value"] != body["value"]:
                    raise ValueError("conflicting producer states for one bound communication tuple")
                current[key] = body
                for role, identity in bound.items():
                    if identity not in index.local_identities:
                        index.local_identities[identity] = len(index.local_identities)
                    if identity not in index.graph_nodes:
                        index.graph_nodes[identity] = {"identity": identity,
                            "ontology_classes": [body["binding_ontology_classes"][role]], "role_bindings": []}
            # The state of a flow/handover is the communication measurement of
            # its explicitly bound actor at this same tick. No actor pose is copied.
            actor_state = index.states[tick, row["entity_id"]]
            comm_state = {k: v for k, v in actor_state.items() if k.startswith("communication.")}
            if len(comm_state) != 4:
                raise ValueError("load_index did not join all four authentic communication fields")
            for identity in (row["flow_id"], row["handover_id"]):
                index.bound_communication_states[tick, identity] = comm_state
                index.semantic_records.append({"kind": "communication", "tick": tick, "entity": identity,
                    "id": f"producer_communication:{identity}:{tick}", "fields": comm_state})
        for key in sorted(set(active) | set(current)):
            if key not in current:
                body = {**active[key], "from_value": active[key]["value"]}
                kind, operation = "update", "remove_predicate_assertion"
            elif key not in active:
                body = current[key]
                kind, operation = ("relation", None) if tick == index.window_start else ("update", "add_predicate_assertion")
            else:
                body = current[key]
                if body["value"] == active[key]["value"]:
                    kind, operation = "update", "refresh_predicate_evidence"
                    body["truth_state_update_tick"] = active[key]["truth_state_update_tick"]
                else:
                    body = {**body, "from_value": active[key]["value"], "to_value": body["value"]}
                    kind, operation = "update", "set_predicate_value"
            record = {"kind": kind, "tick": tick, "entity": None, "id": body["assertion_id"], "fields": {}, "relation": body}
            if operation:
                record["operation"] = operation
            index.semantic_records.append(record)
        active = current
    index.identity_aliases = {f"local:{i}": identity for identity, i in index.local_identities.items()}


@dataclass
class Window:
    episode: str
    cutoff: int
    role: str
    rows: list
    index: object
    load_ms: float


@lru_cache(maxsize=1)
def causal_reader_digest():
    """Source-projection code, independent of neural graph-encoder edits."""
    path = ROOT / "Dataset/world_model/model/causal_input.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ClassDef) and node.name in ("TypedGraphEncoder", "_IncidenceAttention"))]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def causal_window_cache_path(key, config, cache_dir):
    episode, cutoff = key
    spec = config["episodes"][episode]
    sources = [ROOT / spec[k] for k in ("canonical", "communication", "graph_base")]
    sources += [sources[2].with_name("world_truth_graph_deltas.jsonl"),
                sources[1].with_name("predicate_truth.jsonl"),
                ROOT / "Dataset/world_model/model/sumo_temporal.py"]
    if spec.get("sumo_net_xml"):
        sources.append(ROOT / spec["sumo_net_xml"])
    fingerprint = json.dumps([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources])
    loader_code = (inspect.getsource(load_causal_window) + inspect.getsource(reconcile_causal_communication)
                   + inspect.getsource(causal_window_cache_path) + inspect.getsource(causal_reader_digest))
    signature = hashlib.sha256(f"v2:{episode}:{cutoff}:{fingerprint}:{causal_reader_digest()}:{loader_code}".encode()).hexdigest()
    return Path(cache_dir) / f"{signature}.pt", sources


def load_causal_window(key, rows, config, args):
    episode, cutoff = key
    if resolve_source_split(episode) not in ("TRAIN", "VALID"):
        raise ValueError("TEST episodes are prohibited")
    spec = config["episodes"][episode]
    cache, sources = causal_window_cache_path(key, config, args.cache_dir)
    started = time.perf_counter()
    if cache.exists():
        index = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        index = load_index(sources[0], sources[1], episode, cutoff, [cutoff + 50], sources[2],
                           history_s=5., keyframe_s=1., observer_ids=(spec["observer"],),
                           sumo_net_xml=ROOT / spec["sumo_net_xml"] if spec.get("sumo_net_xml") else None)
        # The existing API also returns requested label cells. Remove them at
        # the loader boundary; all later preprocessing sees only the cutoff.
        index.states = {k: v for k, v in index.states.items() if k[0] <= cutoff}
        index.target_ticks = ()
        index.source_bindings = []
        reconcile_causal_communication(index, sources[1])
        temporary = cache.with_suffix(f".{os.getpid()}.tmp")
        torch.save(index, temporary)
        os.replace(temporary, cache)
    return Window(episode, cutoff, role_of(rows[0]), rows, index, 1000 * (time.perf_counter() - started))


def prepare_windows(rows, config, args):
    """Metadata only: full cohorts must not eagerly replay thousands of windows."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode"], row["cutoff"]].append(row)
    return [Window(episode, cutoff, role_of(group[0]), group, None, 0.)
            for (episode, cutoff), group in sorted(grouped.items())]


class WindowStore:
    """Bounded window cache; temporal inputs are indexed once per episode."""
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.windows, self.prepared, self.episodes = OrderedDict(), OrderedDict(), {}

    def load(self, window):
        start = time.perf_counter()
        key = window.episode, window.cutoff
        if key in self.windows:
            self.windows.move_to_end(key)
            cached = self.windows[key]
            return Window(*key, cached.role, cached.rows, cached.index, 1000 * (time.perf_counter()-start))
        if self.args.model == "qwen":
            result = load_causal_window(key, window.rows, self.config, self.args)
        else:
            if window.episode not in self.episodes:
                spec = self.config["episodes"][window.episode]
                paths = [ROOT / spec[k] for k in ("canonical", "communication", "graph_base")]
                paths += [paths[1].with_name("predicate_truth.jsonl"), paths[2].with_name("world_truth_graph_deltas.jsonl")]
                stamp = str([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths])
                signature = hashlib.sha256((stamp + inspect.getsource(TemporalEpisodeSource)).encode()).hexdigest()
                cache = self.args.cache_dir / f"temporal_{signature}.pt"
                if cache.exists():
                    source = torch.load(cache, map_location="cpu", weights_only=False)
                else:
                    source = TemporalEpisodeSource(window.episode, spec, ROOT)
                    temporary = cache.with_suffix(f".{os.getpid()}.tmp")
                    torch.save(source, temporary)
                    os.replace(temporary, cache)
                self.episodes[window.episode] = source
                log(phase="episode_indexed", episode=window.episode, seconds=time.perf_counter()-start)
            index = self.episodes[window.episode].window(window.cutoff, window.rows)
            result = Window(*key, window.role, window.rows, index, 1000 * (time.perf_counter()-start))
        self.windows[key] = result
        if len(self.windows) > self.args.window_cache_size:
            self.windows.popitem(last=False)
        return result

    def features(self, model, window):
        key = window.episode, window.cutoff
        if key in self.prepared:
            self.prepared.move_to_end(key)
            return self.prepared[key]
        if self.args.model == "temporal":
            result = history_features(window.index, window.rows, model.query)
        elif self.args.freeze_graph:
            with torch.no_grad():
                result = model.encode_prefix(window.index)
        else:
            return None
        self.prepared[key] = result
        if len(self.prepared) > self.args.window_cache_size:
            self.prepared.popitem(last=False)
        return result


def input_scales(windows):
    """Reuse field-scale implementation, fitting strictly on causal TRAIN inputs."""
    values = defaultdict(list)
    for w in windows:
        if w.role != "train":
            continue
        for field in w.index.fields:
            if field.logical_dtype != "real":
                continue
            for state in w.index.states.values():
                if field.family in state:
                    kind, value = cell(state[field.family], field.component)
                    if kind == "present":
                        values[field.family, field.component].append(value)
    return {key: fit_input_scale(*key, samples) for key, samples in values.items()}


def evaluate_metrics(rows, predictions):
    result = {"queries": len(rows), "state": {}}
    for h in HORIZONS:
        pairs = [(r["future_state"][str(h)] == "true", p["p_future_true"][str(h)])
                 for r, p in zip(rows, predictions) if r["future_state"][str(h)] in ("true", "false")]
        if not pairs:
            result["state"][str(h)] = {"n": 0, "log_loss": None, "brier": None}
            continue
        y, p = np.array(pairs, dtype=np.float64).T
        safe = np.clip(p, 1e-7, 1 - 1e-7)
        result["state"][str(h)] = {"n": len(y), "positives": int(y.sum()),
            "log_loss": float(-(y * np.log(safe) + (1-y) * np.log1p(-safe)).mean()),
            "brier": float(((p-y)**2).mean())}
    nll, correct, observed_correct, observed_joint_correct = [], [], [], []
    event_nll, censored_nll, partial_censored = [], [], 0
    for row, prediction in zip(rows, predictions):
        if row["state_at_cutoff"] == "unknown":
            continue
        masses = prediction["transition_hazard"] + [1 - prediction["p_transition"]]
        transition = row["first_transition_tick"]
        if transition is not None:
            k = next(i for i, h in enumerate(HORIZONS) if transition - row["cutoff"] <= h)
            likelihood = masses[k]
            correct.append(int(np.argmax(masses) == k))
            observed_correct.append(int(np.argmax(masses[:5]) == k))
            observed_joint_correct.append(int(np.argmax(masses) == k))
            event_nll.append(-math.log(max(likelihood, 1e-12)))
        else:
            elapsed = min(50, row["transition_censored_at"] - row["cutoff"])
            if elapsed <= 0:
                continue
            likelihood = sum(p for h, p in zip(HORIZONS, masses[:5]) if h > elapsed) + masses[5]
            censored_nll.append(-math.log(max(likelihood, 1e-12)))
            partial_censored += int(elapsed < 50)
            if elapsed == 50:
                correct.append(int(np.argmax(masses) == 5))
        nll.append(-math.log(max(likelihood, 1e-12)))
    result["transition"] = {"n": len(nll), "events": len(observed_correct),
        "nll": float(np.mean(nll)) if nll else None,
        "censor_aware_nll": float(np.mean(nll)) if nll else None,
        "observed_transition_nll": float(np.mean(event_nll)) if event_nll else None,
        "censored_survival_nll": float(np.mean(censored_nll)) if censored_nll else None,
        "censored_windows": len(censored_nll), "partially_censored_windows": partial_censored,
        "observed_transition_timing_correct": sum(observed_correct),
        "observed_transition_timing_accuracy": float(np.mean(observed_correct)) if observed_correct else None,
        "observed_transition_joint_accuracy": float(np.mean(observed_joint_correct)) if observed_joint_correct else None,
        "bin_accuracy": float(np.mean(correct)) if correct else None,
        "bin_accuracy_includes_no_transition": True}
    scores = [v["log_loss"] for v in result["state"].values() if v["n"]]
    if nll:
        scores.append(float(np.mean(nll)))
    result["selection_score"] = float(sum(scores) / len(scores)) if scores else None
    return result


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, windows, args, store, scales):
    model.eval()
    all_rows, output_rows, timed = [], [], []
    with torch.no_grad():
        for window_number, window in enumerate(windows, 1):
            w = store.load(window)
            w.index.scales = scales
            sync(args.device)
            started = time.perf_counter()
            extra = store.features(model, w)
            if args.model == "qwen":
                # Exactly one Qwen prefill per window, shared by every bounded
                # continuation microbatch. Each continuation forks all states.
                extra = model.prefill(w.index, extra)
            state, mass = [], []
            # Qwen stores the optional event head in its readout.
            event_types = model.heads.event_types if args.model == "temporal" else model.readout.heads.event_types
            event = [] if event_types else None
            for start in range(0, len(w.rows), args.eval_batch_size):
                group = w.rows[start:start+args.eval_batch_size]
                inputs = extra[start:start+len(group)] if args.model == "temporal" else extra
                output = model(w.index, group, inputs)
                ps, pm, pe = probabilities(output)
                state.extend(ps.cpu().double().tolist())
                mass.extend(pm.cpu().double().tolist())
                if event is not None:
                    event.extend(pe.cpu().double().tolist())
            sync(args.device)
            elapsed = 1000 * (time.perf_counter() - started)
            timed.append({"episode": w.episode, "cutoff": w.cutoff, "queries": len(w.rows), "forward_ms": elapsed,
                          "source_load_ms": w.load_ms})
            for i, row in enumerate(w.rows):
                prediction = {key: row[key] for key in ("episode", "cutoff", "predicate_id", "tuple_id")}
                # Normalize in float64 before serialization; the sum is the
                # contract's P(transition), including at extreme softmax values.
                bins = np.array(mass[i], dtype=np.float64)
                bins /= bins.sum()
                prediction.update(p_future_true={str(h): state[i][j] for j, h in enumerate(HORIZONS)},
                                  p_transition=float(bins[:5].sum()), transition_hazard=bins[:5].tolist(),
                                  runtime_ms_per_query=elapsed / len(w.rows), model=args.model)
                if event is not None:
                    prediction["p_event"] = dict(zip(event_types, event[i]))
                output_rows.append(prediction)
            all_rows.extend(w.rows)
            if window_number % 50 == 0:
                log(phase="evaluation_progress", role=w.role, windows=window_number, total_windows=len(windows), queries=len(all_rows))
            del extra, output
    metrics = evaluate_metrics(all_rows, output_rows)
    metrics["runtime_ms_per_query"] = sum(x["forward_ms"] for x in timed) / len(all_rows)
    metrics["timing_scope"] = "input features/graph, one Qwen prefill per window, continuation batches and CPU export; source load separate"
    metrics["timing_windows"] = timed
    return metrics, all_rows, output_rows


def save_checkpoint(path, model, args, scales, observation_scales, event_types, step, score):
    # Frozen Qwen weights are referenced by path, never copied to run artifacts.
    parameters = {name: value.detach().cpu() for name, value in model.state_dict().items()
                  if not name.startswith("qwen.") or "lora_" in name}
    torch.save({"parameters": parameters, "state_scales": scales, "observation_scales": observation_scales,
                "predicate_ids": sorted(PREDICATES),
                "event_types": event_types, "step": step, "dev_score": score,
                "arguments": {k: str(v) if isinstance(v, (Path, torch.device)) else v for k, v in vars(args).items()}}, path)


def load_parameters(model, checkpoint):
    missing, unexpected = model.load_state_dict(checkpoint["parameters"], strict=False)
    if unexpected or any(not k.startswith("qwen.") or "lora_" in k for k in missing):
        raise ValueError(f"checkpoint parameter mismatch: missing={missing}, unexpected={unexpected}")


def training_weight(row, args):
    """TRAIN-only oversampling; never evaluated on dev/VALID outcomes."""
    return (1. + args.positive_weight * int(any(v == "true" for v in row["future_state"].values()))
            + args.transition_weight * int(row["first_transition_tick"] is not None))


def run(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if checkpoint is not None:
        if checkpoint["arguments"]["model"] != args.model:
            raise ValueError("checkpoint model family mismatch")
        if checkpoint.get("predicate_ids") != sorted(PREDICATES):
            raise ValueError("checkpoint predates the five-predicate B2 scope; retrain with B2")
        for name in ("width", "max_prefix_tokens", "graph_mode", "freeze_graph"):
            setattr(args, name, checkpoint["arguments"][name])
        if args.model == "qwen":
            args.qwen_path = Path(checkpoint["arguments"]["qwen_path"])
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.isdigit() or int(visible) == 0:
            raise ValueError("CUDA_VISIBLE_DEVICES must name one physical nonzero GPU assigned by the coordinator")
        if not torch.cuda.is_available():
            raise RuntimeError("Assigned GPU unavailable; select --device cpu explicitly for CPU execution")
        args.device = torch.device("cuda:0")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., 23 * 1024**3 / total), 0)
        torch.cuda.reset_peak_memory_stats()
    else:
        args.device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = json.loads(args.config.read_text())
    needed = [args.supervision / f"{split}.jsonl" for split in (("VALID",) if args.eval_only else ("TRAIN", "VALID"))]
    rows = read_supervision(args.supervision, ("VALID",) if args.eval_only else ("TRAIN", "VALID"))
    selected = selected_rows(rows, args)
    counts = Counter(role_of(row) for row in selected)
    required = ("valid",) if args.eval_only else ("train", "dev", "valid")
    if any(not counts[role] for role in required):
        raise ValueError(f"missing required supervision role: {dict(counts)}")
    for role in ("train", "dev", "valid"):
        write_rows(args.output / f"supervision_{role.upper()}.jsonl", [r for r in selected if role_of(r) == role])
    log(phase="supervision", selected=dict(counts), available=dict(Counter(role_of(r) for r in rows)),
        train_rows_without_observed_targets=sum(role_of(r) == "train" and not has_training_target(r, args.event_head) for r in rows),
        excluded_predicates=sorted(EXCLUDED_PREDICATES))
    windows = prepare_windows(selected, config, args)
    train = [w for w in windows if w.role == "train"]
    dev = [w for w in windows if w.role == "dev"]
    valid = [w for w in windows if w.role == "valid"]
    store = WindowStore(config, args)
    scale_windows = []
    if checkpoint is not None:
        scales, observation_scales = checkpoint["state_scales"], checkpoint["observation_scales"]
    elif args.model == "temporal":
        # Physical signed-log baseline features have no fitted normalization.
        scales, observation_scales = {}, {}
    else:
        # Independent, TRAIN-only normalization cohort, taken in the original
        # profile round-robin cohort order. Never derive scales from VALID.
        keys = list(dict.fromkeys((r["episode"], r["cutoff"]) for r in selected if role_of(r) == "train"))
        by_key = {(w.episode, w.cutoff): w for w in train}
        for key in keys[:args.scale_windows]:
            w = store.load(by_key[key])
            scale_windows.append(w)
            log(phase="normalization_window", episode=w.episode, cutoff=w.cutoff, load_ms=w.load_ms)
        scales = input_scales(scale_windows)
        observation_scales = fit_observation_scales([w.index for w in scale_windows])
    event_types = (tuple(checkpoint["event_types"]) if checkpoint else
                   tuple(sorted({e["event_type_id"] for w in train for row in w.rows for e in row["event_targets"]}))
                   if args.event_head else ())
    model = (TemporalBaseline(args.width, event_types).to(args.device) if args.model == "temporal" else
             QwenPredicateModel(observation_scales, model_path=args.qwen_path, device=args.device,
                                width=args.width, seed=args.seed, lora_lr=args.lora_lr,
                                max_prefix_tokens=args.max_prefix_tokens, graph_mode=args.graph_mode, event_types=event_types))
    if checkpoint is not None:
        load_parameters(model, checkpoint)
    if args.model == "qwen" and args.freeze_graph:
        model.graph.requires_grad_(False).eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    provenance = source_provenance(config, scale_windows, args)
    provenance.update(
        predicate_source_sha256={p.name: digest(p) for p in (
            Path(__file__), Path(__file__).with_name("qwen_predicate.py"), Path(__file__).with_name("temporal_baseline.py"))},
        supervision={str(p): {"sha256": digest(p), "bytes": p.stat().st_size} for p in needed},
        input_authority=("indexed native canonical frames + graph deltas + recorded communication; GRU feature fields only"
                         if args.model == "temporal" else "load_index causal graph/SUMO + recorded communication, absent rows calculated from state"),
        splits={w.episode: resolve_source_split(w.episode) for w in windows},
        excluded_predicates=sorted(EXCLUDED_PREDICATES), predicate_ids=sorted(PREDICATES),
        capture_tokens=False, compact_future_distribution_inputs=False,
        trainable_parameters=trainable, event_types_train_only=event_types,
        arguments={k: str(v) if isinstance(v, (Path, torch.device)) else v for k, v in vars(args).items()},
        role_rng="Random(predicate-cohort-v2:seed:role); independent training-order stream",
        training_sampling={"mode": args.train_sampling,
            "row_weight": "1 + positive_weight * any known future true + transition_weight * observed transition",
            "window_weight": "mean row weight", "positive_weight": args.positive_weight,
            "transition_weight": args.transition_weight,
            "objective": "unweighted loss on oversampled TRAIN; no dev/VALID label-dependent sampling or correction"},
        normalization_windows=[(w.episode, w.cutoff) for w in scale_windows],
        device=str(args.device))
    if args.model == "qwen":
        provenance.update(qwen_path=str(args.qwen_path), lora=model.lora_config,
                          checkpointed_layers=model.checkpointed_layers, max_prefix_tokens=args.max_prefix_tokens,
                          fixed_decode_batch_size=model.query_batch_size,
                          graph_mode=args.graph_mode, freeze_graph=args.freeze_graph,
                          query_isolation="one sequence per query; copied K/V, convolution and recurrent states; identical position IDs")
    write_json(args.output / "provenance.json", provenance)
    log(phase="model_ready", trainable=trainable, elapsed_seconds=time.perf_counter()-started)
    best_path = args.output / "best.pt"
    curve, step_times, visited = [], [], set()
    initial = None
    if args.eval_only:
        best_score, best_step = checkpoint["dev_score"], checkpoint["step"]
    else:
        initial, _, _ = evaluate(model, dev, args, store, scales)
        best_score = initial["selection_score"]
        if best_score is None:
            raise ValueError("dev has no observed supervision")
        best_step = 0
        save_checkpoint(best_path, model, args, scales, observation_scales, event_types, 0, best_score)
        curve.append({"step": 0, "dev": initial})
        parameters = [{"params": [p for name, p in model.named_parameters() if p.requires_grad and "lora_" not in name], "lr": args.lr}]
        adapters = [p for name, p in model.named_parameters() if p.requires_grad and "lora_" in name]
        if adapters:
            parameters.append({"params": adapters, "lr": args.lora_lr})
        optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay)
        rng = random.Random(f"predicate-training-v2:{args.seed}")
        order = list(range(len(train)))
        weights = [sum(training_weight(r, args) for r in w.rows) / len(w.rows) for w in train]
        fitting_started = time.perf_counter()
        for step in range(1, args.steps + 1):
            if args.time_budget_seconds and time.perf_counter()-fitting_started >= args.time_budget_seconds:
                break
            if args.train_sampling == "weighted":
                wi = rng.choices(order, weights=weights, k=1)[0]
            else:
                if (step - 1) % len(order) == 0:
                    rng.shuffle(order)
                wi = order[(step - 1) % len(order)]
            begin = time.perf_counter()
            w = store.load(train[wi])
            w.index.scales = scales
            visited.add((w.episode, w.cutoff))
            if args.train_sampling == "weighted":
                # Weighted draws with replacement have documented exact row
                # probabilities. Repeated sampled keys are fitting draws only;
                # each supervision key is still exported exactly once.
                ids = rng.choices(range(len(w.rows)), weights=[training_weight(r, args) for r in w.rows],
                                  k=min(args.batch_size, len(w.rows)))
            else:
                ids = rng.sample(range(len(w.rows)), min(args.batch_size, len(w.rows)))
            group = [w.rows[i] for i in ids]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            extra = store.features(model, w)
            if args.model == "temporal":
                extra = extra[ids]
            output = model(w.index, group, extra)
            loss, components = probability_loss(output, group)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            sync(args.device)
            step_times.append(time.perf_counter() - begin)
            log(phase="train", step=step, episode=w.episode, cutoff=w.cutoff, queries=len(group),
                loss=float(loss.detach()), grad_norm=float(norm), seconds=step_times[-1],
                components={k: float(v.detach()) for k, v in components.items()})
            del output, loss, components, extra
            expired = args.time_budget_seconds and time.perf_counter()-fitting_started >= args.time_budget_seconds
            if step % args.eval_every == 0 or step == args.steps or expired:
                metrics, _, _ = evaluate(model, dev, args, store, scales)
                curve.append({"step": step, "dev": metrics})
                score = metrics["selection_score"]
                log(phase="dev", step=step, selection_score=score)
                if score < best_score:
                    best_score, best_step = score, step
                    save_checkpoint(best_path, model, args, scales, observation_scales, event_types, step, score)
            if expired:
                break
        load_parameters(model, torch.load(best_path, map_location="cpu", weights_only=False))
    # No VALID likelihood, label-dependent selection, or forward before selection ends.
    valid_metrics = None
    if not args.skip_valid:
        valid_metrics, valid_rows, predictions = evaluate(model, valid, args, store, scales)
        expected = {row_key(r) for r in selected if role_of(r) == "valid"}
        actual = [row_key(p) for p in predictions]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("prediction identity/cardinality mismatch")
        write_rows(args.output / "predictions_VALID.jsonl", predictions)
        write_rows(args.output / "supervision_VALID.jsonl", valid_rows)
    results = {"model": args.model, "device": str(args.device), "best_step": best_step, "best_dev_score": best_score,
               "initial_dev": initial, "curve": curve, "valid": valid_metrics,
               "selected_queries": dict(counts), "total_seconds": time.perf_counter()-started,
               "training_step_seconds": step_times, "train_windows_visited": len(visited),
               "train_windows_available": len(train), "trainable_parameters": trainable,
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if args.device.type == "cuda" else None,
               "valid_complete": not args.skip_valid and counts["valid"] == sum(role_of(r) == "valid" for r in rows),
               "valid_completeness_scope": "all supervision rows after explicit exclusion of link_degraded/link_unavailable"}
    write_json(args.output / "metrics.json", results)
    log(phase="complete", output=str(args.output), best_step=best_step, seconds=results["total_seconds"],
        valid_queries=None if valid_metrics is None else valid_metrics["queries"], valid_complete=results["valid_complete"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("temporal", "qwen"), default="temporal")
    parser.add_argument("--config", type=Path, default=ROOT / "Dataset/world_model/configs/state_distribution_5s.json")
    parser.add_argument("--supervision", type=Path, default=WORKSPACE / "supervision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=WORKSPACE / "runs/task_b_cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--train-windows", type=int, default=6)
    parser.add_argument("--dev-windows", type=int, default=3)
    parser.add_argument("--valid-windows", type=int, default=3)
    parser.add_argument("--queries-per-window", type=int, default=32)
    for role in ("train", "dev", "valid"):
        parser.add_argument(f"--{role}-queries-per-window", type=int, default=None,
                            help="override this role's cohort query cap; 0 keeps all")
    parser.add_argument("--full-valid", action="store_true", help="all VALID windows/rows after the two explicit predicate exclusions")
    parser.add_argument("--eval-only", action="store_true", help="load checkpoint and VALID only; no TRAIN/dev read or selection")
    parser.add_argument("--skip-valid", action="store_true", help="train/dev only; export VALID in a separate --eval-only invocation")
    parser.add_argument("--batch-size", type=int, default=8, help="TRAIN queries per update, independent of cohort size")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="independent query continuations per shared prefix")
    parser.add_argument("--time-budget-seconds", type=float, default=0,
                        help="training wall budget including source loads and dev checks, after initial dev; 0 is unlimited")
    parser.add_argument("--train-sampling", choices=("uniform", "weighted"), default="uniform")
    parser.add_argument("--positive-weight", type=float, default=4.)
    parser.add_argument("--transition-weight", type=float, default=8.)
    parser.add_argument("--scale-windows", type=int, default=18, help="TRAIN windows used for Qwen normalization")
    parser.add_argument("--window-cache-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--qwen-path", type=Path, default=Path(DEFAULT_QWEN))
    parser.add_argument("--max-prefix-tokens", type=int, default=192)
    parser.add_argument("--graph-mode", choices=("graph_full", "graph_summary"), default="graph_summary")
    parser.add_argument("--freeze-graph", action="store_true", help="explicit fast ablation: cache graph encodings; graph parameters receive no updates")
    parser.add_argument("--event-head", action="store_true", help="optional BCE head; vocabulary from training rows only")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("threads must be in [1,16]")
    if args.steps < 0 or args.eval_every < 1 or any(getattr(args, k) < 0 for k in ("train_windows", "dev_windows", "valid_windows", "queries_per_window")):
        parser.error("steps/limits must be nonnegative and eval-every positive")
    if any(getattr(args, f"{role}_queries_per_window") is not None and getattr(args, f"{role}_queries_per_window") < 0
           for role in ("train", "dev", "valid")):
        parser.error("role query caps must be nonnegative")
    if min(args.batch_size, args.eval_batch_size, args.scale_windows, args.window_cache_size) < 1 or min(
            args.time_budget_seconds, args.positive_weight, args.transition_weight) < 0:
        parser.error("batch/cache/scale sizes must be positive, weights and time budget nonnegative")
    if args.eval_only and (not args.checkpoint or args.skip_valid):
        parser.error("--eval-only requires --checkpoint and cannot use --skip-valid")
    if not args.output.resolve().is_relative_to((WORKSPACE / "runs").resolve()) or not args.cache_dir.resolve().is_relative_to((WORKSPACE / "runs").resolve()):
        parser.error("Task B artifacts must stay under the contract runs directory")
    run(args)


if __name__ == "__main__":
    main()
