"""Train or export compact-future/Qwen mixtures on the predicate contract.

Examples and measured CPU checks: runs/predicate_stage/reports/task_d.md.
Use --prepare-subset-only to write the common outcome-independent VALID cohort.
Use --eval-only --checkpoint ... --full-valid for complete eligible VALID export.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch

from Dataset.world_model.model.causal_input import fit_observation_scales
from .qwen_predicate import DEFAULT_QWEN, HORIZONS, PREDICATES, EXCLUDED_PREDICATES, probability_loss, probabilities
from .train_predicate import (ROOT, WORKSPACE, WindowStore, read_supervision, selected_rows, prepare_windows,
                             input_scales, role_of, row_key, load_parameters, save_checkpoint, evaluate_metrics,
                             sync, log, write_json, write_rows, digest, training_weight)
from .two_stage import (COMPACT, CompactFutureAdapter, TwoStageModel, future_feedback,
                        window_seed, assert_causal_index, causal_arrays, rule_features, RULE_WIDTH)
from .frozen_history import FrozenHistoryCache
from .temporal_baseline import RuleTemporalBaseline, tuple_history_features, history_features, TemporalEpisodeSource
from .rule_predicates import evaluate as evaluate_rule


def atomic_json(path, document):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    write_json(temporary, document)
    os.replace(temporary, path)


class TaskEFeatures:
    """Source-keyed real input cache, shared by all Task E methods.

    The Monte Carlo draw is fixed per window. No future labels, transition
    labels, supervision cutoff truth or dev/VALID scores enter this cache.
    """
    def __init__(self, config, args, adapter):
        self.args, self.adapter = args, adapter
        temporal_args = copy.copy(args)
        temporal_args.model = "temporal"
        self.store = WindowStore(config, temporal_args)
        self.directory = args.feature_cache_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        signature = hashlib.sha256()
        for name in ("sampling.py", "rule_predicates.py"):
            signature.update(Path(__file__).with_name(name).read_bytes())
        for function in (rule_features, tuple_history_features, history_features, TemporalEpisodeSource,
                         CompactFutureAdapter.causal_source, CompactFutureAdapter.sample, CompactFutureAdapter.observed):
            signature.update(inspect.getsource(function).encode())
        paths = [*adapter.pose_paths, *adapter.latency_paths, *adapter.handover_paths, args.pose_pool]
        paths += [ROOT / spec[key] for spec in config["episodes"].values() for key in ("canonical", "communication", "graph_base")]
        paths += [ROOT / spec["communication"] for spec in config["episodes"].values()]
        paths += [p.with_name("predicate_truth.jsonl") for p in paths if p.name == "communication_state.jsonl"]
        paths += [p.with_name("world_truth_graph_deltas.jsonl") for p in paths if p.name == "world_truth_graph_base.json"]
        for path in paths:
            path = Path(path)
            stat = path.stat()
            signature.update(str((path, stat.st_size, stat.st_mtime_ns)).encode())
        self.signature = signature.hexdigest()
        self.query = None
        self.hits = self.misses = 0

    def load(self, window, k, mode):
        requested_k, requested_mode = k, mode
        # A canonical nested draw makes runtime K comparisons use identical
        # first samples and shares authentic history across all methods.
        k, mode = max(8, k), "sampled"
        metadata = [(row_key(r), sorted(r["bindings"].items())) for r in window.rows]
        seed = window_seed(self.args.seed, window.episode, window.cutoff)
        signature = hashlib.sha256(repr((self.signature, metadata, seed, k, mode)).encode()).hexdigest()
        path = self.directory / f"{signature}.pt"
        if path.exists():
            self.hits += 1
            result = torch.load(path, map_location="cpu", weights_only=False)
            if result["keys"] != [row_key(r) for r in window.rows] or result.get("source_signature") != self.signature:
                raise ValueError("Task E feature cache/source metadata mismatch")
            result["cache_hit"] = True
            return self.select(result, requested_k, requested_mode)
        begin = time.perf_counter()
        w = self.store.load(window)
        direct = tuple_history_features(w.index, w.rows)
        if self.query is None:
            from .qwen_predicate import PredicateQueryEncoder
            self.query = PredicateQueryEncoder()
        history = history_features(w.index, w.rows, self.query)
        _, context, _ = self.adapter.source(w.episode)
        if mode == "history":
            future = self.adapter.observed(w.episode, w.cutoff)
            cutoff = []
            for row in w.rows:
                record = context.communication_row(w.cutoff, row["bindings"]) if row["family"] == "communication" else None
                truth = evaluate_rule(future, context, row["predicate_id"], row["bindings"],
                                      owner=None if record is None else record["entity_id"], fixed_scope=True)[0, 0]
                cutoff.append([float(truth == 0), float(truth == 1), float(not np.isfinite(truth))])
            result = {"tuple": direct, "history": history, "cutoff_rule": torch.tensor(cutoff)}
        else:
            future = self.adapter.sample(w.episode, w.cutoff, k, seed)
            result = {"tuple": direct, "history": history, **rule_features(future, context, w.rows, direct), "future": future}
        result.update(keys=[row_key(r) for r in w.rows], input_ms=(time.perf_counter()-begin)*1000, cache_hit=False,
                      source_signature=self.signature)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(result, temporary)
        os.replace(temporary, path)
        self.misses += 1
        return self.select(result, requested_k, requested_mode)

    def select(self, result, k, mode):
        if mode == "history":
            return {name: value for name, value in result.items() if name not in ("future", "rule", "state_prior", "mass_prior")}
        result = dict(result)
        rule = result["rule"][:k]
        # Fixed per-sample shrinkage is a declared input prior, independent of
        # runtime K. Its strength is NOT an asserted Monte Carlo sample count.
        strength = self.args.rule_prior_strength
        state = rule[:, :, [12, 13, 15, 17, 21]]
        mass = rule[:, :, 22:28]
        result.update(rule=rule, state_prior=(strength*state+.5)/(strength+1.), mass_prior=(strength*mass+.5)/(strength+3.))
        future = result["future"]
        from dataclasses import replace
        result["future"] = replace(future, position=future.position[:k], velocity=future.velocity[:k],
            communication={name: value[:k] for name, value in future.communication.items()})
        return result


def slice_features(features, ids):
    return {key: value[:, ids] if key in ("rule", "state_prior", "mass_prior") else value[ids]
            for key, value in features.items() if key in ("rule", "state_prior", "mass_prior", "tuple", "history", "cutoff_rule")}


def without_cross_relations(index):
    """Fixed-checkpoint ablation of explicit multi-entity incidence records.

    Numerical states, unary relations, direct tuple history and sampled rule
    features are retained. A separate cache namespace prevents an ablated
    prefix from replacing the authentic full-context graph prefix.
    """
    result = copy.copy(index)
    result.semantic_records = [record for record in index.semantic_records
        if not any(len(set(record[key]["bindings"].values())) > 1
                   for key in ("relation", "context_relation") if key in record)]
    return result


def load_e_parameters(model, checkpoint, *, legacy=False):
    missing, unexpected = model.load_state_dict(checkpoint["parameters"], strict=False)
    permitted = ("direct_context.",) if legacy else ()
    errors = [key for key in missing if (not key.startswith("qwen.") or "lora_" in key) and not key.startswith(permitted)]
    if errors or unexpected:
        raise ValueError(f"Task E parameter mismatch: {errors}, {unexpected}")


def run_task_e(args):
    """Meaningful-scale fit, TRAIN-dev selection and strict-cohort export."""
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
            raise ValueError("Task E is authorized on physical GPU 3 only")
        args.device = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(23_000_000_000/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
    else:
        args.device = torch.device("cpu")
    if args.future_mode == "oracle" or args.positive_weight or args.transition_weight:
        raise ValueError("Task E uses uniform TRAIN sampling and no oracle inputs")
    if args.history_cache_dir and not args.freeze_graph and args.model == "qwen":
        raise ValueError("frozen graph caches require --freeze-graph")
    if args.graph_device == "cpu" and not args.freeze_graph and args.model == "qwen":
        raise ValueError("CPU graph preparation requires --freeze-graph")
    config = json.loads(args.config.read_text())
    valid_rows = read_supervision(args.supervision, ("VALID",))
    selected_valid, _ = valid_cohort(valid_rows, args)
    selected_train = [] if args.eval_only else selected_rows(read_supervision(args.supervision, ("TRAIN",)), args)
    windows = prepare_windows(selected_train+selected_valid, config, args)
    train, dev, valid = ([w for w in windows if w.role == role] for role in ("train", "dev", "valid"))
    adapter = CompactFutureAdapter(arrays=args.arrays, members=args.members, pose_pool=args.pose_pool)
    features = TaskEFeatures(config, args, adapter)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    legacy = checkpoint is not None and not checkpoint["arguments"].get("context_features", False)
    if checkpoint and not legacy:
        if checkpoint["arguments"]["model"] != args.model or checkpoint["arguments"]["future_mode"] != args.future_mode:
            raise ValueError("checkpoint architecture/input mode mismatch")
        for name in ("width", "graph_mode", "history_tokens", "max_prefix_tokens", "graph_device", "rule_prior_strength"):
            setattr(args, name, checkpoint["arguments"][name])
        args.qwen_path = Path(checkpoint["arguments"]["qwen_path"])
        args.freeze_graph = checkpoint["arguments"]["freeze_graph"]
        args.temporal_per_sample = checkpoint["arguments"].get("temporal_per_sample", False)
    if args.prepare_features_only:
        for number, w in enumerate(windows, 1):
            features.load(w, args.samples, args.future_mode)
            if number % 50 == 0:
                log(phase="features", windows=number, total=len(windows), seconds=time.perf_counter()-started)
        write_json(args.output / "feature_preparation.json", {"windows": len(windows), "hits": features.hits, "misses": features.misses,
                   "seconds": time.perf_counter()-started, "signature": features.signature})
        return
    scales, observation_scales = ({}, {}) if checkpoint is None else (checkpoint["state_scales"], checkpoint["observation_scales"])
    if args.model == "qwen":
        if checkpoint is None:
            raise ValueError("Task E Qwen needs the existing B2 or Task E checkpoint")
        model = TwoStageModel(observation_scales, model_path=args.qwen_path, device=args.device, width=args.width,
                              seed=args.seed, lora_lr=args.lora_lr, max_prefix_tokens=args.max_prefix_tokens,
                              graph_mode=args.graph_mode, history_tokens=args.history_tokens,
                              sample_batch_size=args.sample_batch_size, query_batch_size=args.query_batch_size,
                              context_features=True)
        load_e_parameters(model, checkpoint, legacy=legacy)
        if legacy and args.future_mode == "sampled":
            for head in (model.readout.heads.state, model.readout.heads.transition):
                torch.nn.init.zeros_(head.weight)
                torch.nn.init.zeros_(head.bias)
        if args.freeze_graph:
            model.graph.requires_grad_(False)
        if args.graph_device == "cpu":
            model.graph.cpu()
        if args.history_cache_dir:
            model.history_cache = FrozenHistoryCache(model.graph, config, args.cache_dir, args.history_cache_dir, pooled_only=True)
        graph_store = WindowStore(config, args)
    elif args.model == "temporal":
        model = RuleTemporalBaseline(RULE_WIDTH, args.width, per_sample=args.temporal_per_sample).to(args.device)
        if checkpoint:
            load_e_parameters(model, checkpoint)
        graph_store = features.store
    else:
        model = None
        graph_store = features.store
    provenance = {"arguments": {key: str(value) if isinstance(value, (Path, torch.device)) else value for key, value in vars(args).items()},
        "cohorts": {role: [[w.episode, w.cutoff] for w in windows if w.role == role] for role in ("train", "dev", "valid")},
        "feature_signature": features.signature, "compact": adapter.describe(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_memory_budget_bytes": 23_000_000_000 if args.device.type == "cuda" else None,
        "sampling": "shuffled fitting windows; uniform queries without replacement; fixed per-window Monte Carlo draw; no oversampling or prior correction needed",
        "inputs": "eleven observed tuple truths, full causal current-spell age and left-censor flag, cutoff rule truth; per-sample raw/aligned relations, first-change bin and masked operands",
        "rule_prior": {"fixed_strength": args.rule_prior_strength,
            "policy": "fixed per-sample binary/six-category shrinkage; not an MC sample-count claim; Qwen learns log-odds/log-mass corrections then averages probabilities"},
        "qwen_kernels": None if args.model != "qwen" else model.kernel_config,
        "trainable_parameters": 0 if model is None else sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(args.output / "provenance.json", provenance)

    cross_relation_cache = None

    def infer(window, ids=None, training=False, k=None, use_cache=True, ablation=None):
        nonlocal cross_relation_cache
        k = args.samples if k is None else k
        begin = time.perf_counter()
        bundle = features.load(window, k, args.future_mode)
        input_end = time.perf_counter()
        ids = list(range(len(window.rows))) if ids is None else ids
        rows = [window.rows[i] for i in ids]
        sliced = slice_features(bundle, ids)
        if ablation == "tuple":
            # Retain observed cutoff truth; remove the ten preceding direct
            # truth slots and explicitly mask the age channels for this ablation.
            sliced["tuple"] = sliced["tuple"].clone()
            sliced["tuple"][:, :30] = 0
            sliced["tuple"][:, 2:30:3] = 1
            sliced["tuple"][:, 33:] = 0
        if model is None:
            # Rules-alone estimator uses actual K counts and Jeffreys smoothing.
            raw_state = sliced["rule"][:, :, [12, 13, 15, 17, 21]].mean(0)
            raw_mass = sliced["rule"][:, :, 22:28].mean(0)
            state = sliced["state_prior"].mean(0) if args.rule_only_prior else (k*raw_state+.5)/(k+1.)
            mass = sliced["mass_prior"].mean(0) if args.rule_only_prior else (k*raw_mass+.5)/(k+3.)
            output = {"state_logits": torch.logit(state), "transition_logits": mass.log(), "event_logits": None, "event_types": ()}
        else:
            w = graph_store.load(window)
            w.index.scales = scales
            if args.model == "temporal":
                output = model(w.index, rows, sliced)
            else:
                index = without_cross_relations(w.index) if ablation == "cross_relations" else w.index
                feedbacks = None if args.future_mode == "history" else [future_feedback(index, bundle["future"], i) for i in range(k)]
                original_cache = model.history_cache
                if ablation == "cross_relations" and original_cache is not None:
                    if cross_relation_cache is None:
                        cross_relation_cache = FrozenHistoryCache(model.graph, config, args.cache_dir,
                            args.history_cache_dir / "without_cross_relations", pooled_only=True)
                    model.history_cache = cross_relation_cache
                with torch.no_grad() if args.freeze_graph else nullcontext():
                    try:
                        structures = model.structures(index, feedbacks)
                    finally:
                        model.history_cache = original_cache
                if ablation == "graph":
                    # Retain numerical inputs/query features; remove contextual prefix embeddings.
                    if structures[0] is None:
                        structures[1]["embeddings"] = torch.zeros_like(structures[1]["embeddings"])
                    else:
                        structures[0]["embeddings"] = torch.zeros_like(structures[0]["embeddings"])
                        for prefix in structures[1]:
                            prefix["embeddings"] = torch.zeros_like(prefix["embeddings"])
                output = model.score_samples(index, rows, model.prefill_samples(index, structures), sliced)
        sync(args.device)
        return output, {"elapsed_ms": (time.perf_counter()-begin)*1000, "feature_read_ms": (input_end-begin)*1000,
                        "original_input_ms": bundle["input_ms"], "input_cache_hit": bundle["cache_hit"]}

    def evaluate_e(cohort, k, ablation=None):
        if model is not None:
            model.eval()
        predictions, timings = [], []
        effective_k = k if args.future_mode == "sampled" else 1
        with torch.no_grad():
            for number, w in enumerate(cohort, 1):
                output, timing = infer(w, k=k, ablation=ablation)
                state, mass, _ = probabilities(output)
                state, mass = state.cpu().double().tolist(), mass.cpu().double().tolist()
                runtime = timing["elapsed_ms"]/len(w.rows)
                for i, row in enumerate(w.rows):
                    predictions.append({**{key: row[key] for key in ("episode", "cutoff", "predicate_id", "tuple_id")},
                        "p_future_true": dict(zip(map(str, HORIZONS), state[i])), "transition_hazard": mass[i][:5],
                        "p_transition": sum(mass[i][:5]), "runtime_ms_per_query": runtime,
                        "model": f"task_e_{args.model}_{args.future_mode}_K{effective_k}"})
                timings.append({"episode": w.episode, "cutoff": w.cutoff, "queries": len(w.rows), **timing})
                if number % 10 == 0 or number == len(cohort):
                    log(phase="eval_e", number=number, windows=len(cohort), k=k, ms_per_query=runtime)
        from .evaluate_predictions import metric_report
        rows = {row_key(r): r for w in cohort for r in w.rows}
        metrics, _, _ = metric_report(rows, {row_key(r): r for r in predictions}, sorted({w.episode for w in cohort}))
        metrics["selection_score"] = (metrics["all"]["mean_horizon_log_loss"]+metrics["all"]["transition"]["nll"])/2
        metrics["runtime_ms_per_query"] = sum(t["elapsed_ms"] for t in timings)/len(predictions)
        metrics["samples"] = effective_k
        metrics["timings"] = timings
        return metrics, predictions

    curve, updates, visited = [], [], set()
    best_step, best_score = (checkpoint["step"], checkpoint["dev_score"]) if args.eval_only and checkpoint else (0, None)
    if not args.eval_only and model is not None:
        if not train or not dev:
            raise ValueError("Task E fit requires TRAIN and TRAIN-dev")
        initial, _ = evaluate_e(dev, args.dev_samples)
        best_score = initial["selection_score"]
        best = args.output / "best.pt"
        save_checkpoint(best, model, args, scales, observation_scales, (), 0, best_score)
        curve.append({"step": 0, "dev": initial})
        groups = [{"params": [p for name, p in model.named_parameters() if p.requires_grad and "lora_" not in name], "lr": args.lr},
                  {"params": [p for name, p in model.named_parameters() if p.requires_grad and "lora_" in name], "lr": args.lora_lr}]
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        rng = random.Random(f"task-e-uniform:{args.seed}")
        order = list(range(len(train)))
        fit_start = time.perf_counter()
        for step in range(1, args.steps+1):
            if (step-1) % len(order) == 0:
                rng.shuffle(order)
            w = train[order[(step-1) % len(order)]]
            ids = rng.sample(range(len(w.rows)), min(args.batch_size, len(w.rows)))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output, timing = infer(w, ids, True, args.train_samples)
            loss, components = probability_loss(output, [w.rows[i] for i in ids])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite Task E training loss")
            loss.backward()
            gradient_groups = {}
            for label, term in (("lora", "lora_"), ("direct_context", "direct_context."), ("query", "query_token."), ("graph", "graph.")):
                gradients = [p.grad.float().norm().square() for name, p in model.named_parameters() if term in name and p.grad is not None]
                gradient_groups[label] = float(torch.stack(gradients).sum().sqrt()) if gradients else None
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            visited.add((w.episode, w.cutoff))
            record = {"step": step, "episode": w.episode, "cutoff": w.cutoff, "queries": len(ids),
                      "samples": args.train_samples if args.future_mode == "sampled" else 1,
                      "loss": float(loss.detach()), "grad_norm": float(norm), "gradient_groups": gradient_groups, **timing,
                      "components": {key: float(value.detach()) for key, value in components.items()}}
            updates.append(record)
            if step % 10 == 0:
                log(phase="train_e", **record)
            del output, loss
            expired = args.time_budget_seconds and time.perf_counter()-fit_start >= args.time_budget_seconds
            if step % args.eval_every == 0 or step == args.steps or expired:
                measured, _ = evaluate_e(dev, args.dev_samples)
                score = measured["selection_score"]
                curve.append({"step": step, "dev": measured})
                if score < best_score:
                    best_step, best_score = step, score
                    save_checkpoint(best, model, args, scales, observation_scales, (), step, score)
                atomic_json(args.output / "training_progress.json", {"best_step": best_step, "best_dev_score": best_score,
                           "curve": curve, "updates": updates, "visited": len(visited)})
                log(phase="dev_e", step=step, score=score, best_step=best_step, visited=len(visited))
            if expired:
                break
        load_e_parameters(model, torch.load(best, map_location="cpu", weights_only=False))
        if args.dev_k_sweep:
            sweep = []
            for k in args.dev_k_sweep if args.future_mode == "sampled" else [1]:
                measured, _ = evaluate_e(dev, k)
                sweep.append({"k": k, "metrics": measured})
            atomic_json(args.output / "dev_k_sweep.json", {"checkpoint_step": best_step, "selection_role": "TRAIN-dev", "sweep": sweep})
    valid_metrics = None
    if not args.skip_valid:
        valid_metrics, predictions = evaluate_e(valid, args.samples)
        expected = {row_key(r) for r in selected_valid}
        if len(predictions) != len(expected) or {row_key(r) for r in predictions} != expected:
            raise ValueError("Task E export coverage mismatch")
        write_rows(args.output / "predictions_VALID.jsonl", predictions)
        if args.context_ablation:
            for ablation in ("tuple", "graph", "cross_relations"):
                measured, ablated = evaluate_e(valid, args.samples, ablation)
                write_rows(args.output / f"predictions_VALID_without_{ablation}.jsonl", ablated)
                write_json(args.output / f"metrics_without_{ablation}.json", measured)
    atomic_json(args.output / "metrics.json", {"best_step": best_step, "best_dev_score": best_score,
        "curve": curve, "training_steps": updates, "train_windows_visited": len(visited), "valid": valid_metrics,
        "seconds": time.perf_counter()-started, "memory": memory(args.device),
        "feature_cache": {"hits": features.hits, "misses": features.misses},
        "graph_cache": None if args.model != "qwen" or model.history_cache is None else
            {"hits": model.history_cache.hits, "misses": model.history_cache.misses,
             "future_hits": model.history_cache.future_hits, "future_misses": model.history_cache.future_misses,
             "pooled_hits": model.history_cache.pooled_hits, "pooled_misses": model.history_cache.pooled_misses}})
    log(phase="complete_e", seconds=time.perf_counter()-started, best_step=best_step, **memory(args.device))


def subset_windows(rows, size, seed):
    """Assign equal family strata using only episode/cutoff/family metadata.

    A window can contain several families. Assign one stratum without replacement,
    then retain ALL eligible queries in each selected window for every method.
    """
    metadata = defaultdict(set)
    for row in rows:
        metadata[row["episode"], row["cutoff"]].add(row["family"])
    rng = random.Random(f"two-stage-valid-subset-v1:{seed}")
    queues = {}
    for family in sorted(set(PREDICATES.values())):
        by_episode = defaultdict(list)
        for key, families in sorted(metadata.items()):
            if family in families:
                by_episode[key[0]].append(key)
        episodes = sorted(by_episode)
        rng.shuffle(episodes)
        for keys in by_episode.values():
            rng.shuffle(keys)
        queue = []
        while any(by_episode.values()):
            for episode in episodes:
                if by_episode[episode]:
                    queue.append(by_episode[episode].pop())
        queues[family] = iter(queue)
    picked, used = [], set()
    while len(picked) < min(size, len(metadata)):
        progress = False
        for family, queue in queues.items():
            key = next((key for key in queue if key not in used), None)
            if key is not None:
                used.add(key)
                picked.append({"episode": key[0], "cutoff": key[1], "stratum": family,
                               "families": sorted(metadata[key])})
                progress = True
                if len(picked) == min(size, len(metadata)):
                    break
        if not progress:
            raise ValueError("family strata exhausted before covering the requested subset")
    return {"version": 1, "seed": seed, "requested_windows": size,
            "selection": "family round robin, profile round robin within family, no truth/outcome fields",
            "query_policy": "all eligible queries in selected windows; same keys for every method",
            "windows": picked}


def valid_cohort(rows, args):
    expected = subset_windows(rows, args.subset_size, args.seed)
    if args.subset_file.exists():
        chosen = json.loads(args.subset_file.read_text())
        if chosen != expected:
            raise ValueError("existing subset differs from seed/metadata; use its original settings or a new --subset-file")
    else:
        args.subset_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.subset_file, expected)
        chosen = expected
    lookup = {(r["episode"], r["cutoff"]) for r in chosen["windows"]}
    subset = [r for r in rows if (r["episode"], r["cutoff"]) in lookup]
    # A ready-to-use supervision directory lets B2 and C2 export identical keys.
    shared = args.subset_file.parent / args.subset_file.stem / "supervision"
    shared.mkdir(parents=True, exist_ok=True)
    path = shared / "VALID.jsonl"
    if not path.exists():
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        write_rows(temporary, subset)
        os.replace(temporary, path)
    result = rows if args.full_valid else subset
    windows = sorted({(r["episode"], r["cutoff"]) for r in result})
    if args.valid_windows and not args.full_valid:
        allowed = set(windows[:args.valid_windows])
        result = [r for r in result if (r["episode"], r["cutoff"]) in allowed]
    if args.eval_queries_per_window and not args.full_valid:
        groups = defaultdict(list)
        for row in result:
            groups[row["episode"], row["cutoff"]].append(row)
        # Metadata-only round robin prevents small CPU checks losing a family.
        reduced = []
        for key, group in sorted(groups.items()):
            by_pred = defaultdict(list)
            for row in sorted(group, key=row_key):
                by_pred[row["predicate_id"]].append(row)
            n = 0
            while any(by_pred.values()) and n < args.eval_queries_per_window:
                for pid in sorted(by_pred):
                    if by_pred[pid] and n < args.eval_queries_per_window:
                        reduced.append(by_pred[pid].pop(0))
                        n += 1
        result = reduced
    return result, chosen


def memory(device):
    return {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None}


def feedback_for(adapter, index, mode, k, seed, training=False):
    if mode == "history":
        return None
    future = (adapter.oracle(index.episode, index.cutoff) if mode == "oracle" else
              adapter.sample(index.episode, index.cutoff, k, seed, training=training))
    return [future_feedback(index, future, i) for i in range(future.position.shape[0])]


def prepare_forward(model, adapter, index, args, k, seed, *, training=False):
    sync(args.device)
    start = time.perf_counter()
    feedbacks = feedback_for(adapter, index, args.future_mode, k, seed, training)
    sample_ms = (time.perf_counter()-start)*1000
    with torch.no_grad() if args.freeze_graph else nullcontext():
        structures = model.structures(index, feedbacks)
    sync(args.device)
    graph_end = time.perf_counter()
    prepared = model.prefill_samples(index, structures)
    sync(args.device)
    return prepared, {"sampling_ms": sample_ms, "graph_ms": (graph_end-start)*1000-sample_ms,
                      "prefill_ms": (time.perf_counter()-graph_end)*1000,
                      "prefixes": prepared[1]}


def prediction_rows(rows, output, runtime, mode, k):
    state, mass, event = probabilities(output)
    state, mass = state.detach().cpu().double().tolist(), mass.detach().cpu().double().numpy()
    state_std = output["state_std"].detach().cpu().double().tolist()
    mass_std = output["transition_std"].detach().cpu().double().tolist()
    event = None if event is None else event.detach().cpu().double().tolist()
    event_std = None if output["event_std"] is None else output["event_std"].detach().cpu().double().tolist()
    predictions, spread = [], []
    for i, row in enumerate(rows):
        bins = mass[i]/mass[i].sum()
        identity = {key: row[key] for key in ("episode", "cutoff", "predicate_id", "tuple_id")}
        prediction = {**identity, "p_future_true": {str(h): state[i][j] for j, h in enumerate(HORIZONS)},
                      "p_transition": float(sum(bins[:5])), "transition_hazard": bins[:5].tolist(),
                      "runtime_ms_per_query": runtime, "model": f"two_stage_{mode}_K{k}"}
        variance = {**identity, "samples": k, "p_future_true_std": dict(zip(map(str, HORIZONS), state_std[i])),
                    "transition_bin_std": mass_std[i], "p_transition_std": mass_std[i][-1],
                    "std_convention": "population, ddof=0"}
        if event is not None:
            prediction["p_event"] = dict(zip(output["event_types"], event[i]))
            variance["p_event_std"] = dict(zip(output["event_types"], event_std[i]))
        predictions.append(prediction)
        spread.append(variance)
    return predictions, spread


def evaluate(model, adapter, windows, args, store, scales, k=None):
    model.eval()
    k = args.samples if k is None else k
    effective_k = k if args.future_mode == "sampled" else 1
    all_rows, predictions, spreads, timings = [], [], [], []
    with torch.no_grad():
        for number, window in enumerate(windows, 1):
            w = store.load(window)
            w.index.scales = scales
            # Episode source indexing is separately timed; per-cutoff slicing,
            # neural law inference and sampling remain inside the query timer.
            begin = time.perf_counter()
            if args.future_mode != "history":
                adapter.source(w.episode)
            source_ms = w.load_ms+(time.perf_counter()-begin)*1000
            sync(args.device)
            start = time.perf_counter()
            prepared, timing = prepare_forward(model, adapter, w.index, args, k,
                                                window_seed(args.seed, w.episode, w.cutoff))
            collected, std_rows = [], []
            for offset in range(0, len(w.rows), args.eval_batch_size):
                group = w.rows[offset:offset+args.eval_batch_size]
                output = model.score_samples(w.index, group, prepared)
                p, s = prediction_rows(group, output, 0., args.future_mode, effective_k)
                collected.extend(p)
                std_rows.extend(s)
            sync(args.device)
            elapsed = (time.perf_counter()-start)*1000
            for row in collected:
                row["runtime_ms_per_query"] = elapsed/len(w.rows)
            timings.append({"episode": w.episode, "cutoff": w.cutoff, "queries": len(w.rows),
                            "inference_ms": elapsed, "source_ms": source_ms, **timing})
            all_rows.extend(w.rows)
            predictions.extend(collected)
            spreads.extend(std_rows)
            log(phase="eval_window", role=w.role, number=number, windows=len(windows),
                episode=w.episode, cutoff=w.cutoff, queries=len(w.rows), samples=effective_k,
                ms_per_query=elapsed/len(w.rows), source_ms=source_ms)
            del prepared, output
    metrics = evaluate_metrics(all_rows, predictions)
    metrics.update(samples=effective_k, windows=len(windows),
                   runtime_ms_per_query=sum(t["inference_ms"] for t in timings)/len(all_rows),
                   runtime_with_source_ms_per_query=sum(t["inference_ms"]+t["source_ms"] for t in timings)/len(all_rows),
                   timing_scope="causal slicing, compact inference/sampling, feedback, graph, Qwen, heads, aggregation, CPU export; source load separate",
                   timings=timings, memory=memory(args.device))
    return metrics, all_rows, predictions, spreads


def validate_real(model, adapter, window, args, store, scales):
    """Execution audit on an actual window; writes evidence, no test files."""
    model.eval()
    w = store.load(window)
    parameter_shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    index = w.index
    index.scales = scales
    assert_causal_index(index)
    seed = window_seed(args.seed, w.episode, w.cutoff)
    first = adapter.sample(w.episode, w.cutoff, 2, seed)
    original = adapter.sources[w.episode]
    raw, context, poses = original
    truncated = copy.copy(context)
    truncated.comm = {t: r for t, r in context.comm.items() if t <= w.cutoff}
    adapter.sources[w.episode] = (causal_arrays(raw, w.cutoff), truncated,
                                   {t: r for t, r in poses.items() if t <= w.cutoff})
    try:
        second = adapter.sample(w.episode, w.cutoff, 2, seed)
    finally:
        adapter.sources[w.episode] = original
    equal = (np.array_equal(first.position, second.position) and np.array_equal(first.velocity, second.velocity)
             and all(np.array_equal(first.communication[n], second.communication[n], equal_nan=True) for n in first.communication))
    if not equal:
        raise AssertionError("post-cutoff source removal changed samples")
    feedbacks = [future_feedback(index, first, i) for i in range(2)]
    queries = w.rows[:min(3, len(w.rows))]
    with torch.no_grad():
        structures = model.structures(index, feedbacks)
        together = model.prefill_samples(index, structures)
        output = model.score_samples(index, queries, together)
        repeat = model.score_samples(index, queries, together)
        reverse = model.score_samples(index, list(reversed(queries)), together)
        single_prepared = model.prefill_samples(index, (structures[0], structures[1][:1], structures[2][:1]))
        alone = model.score_samples(index, queries, single_prepared)
        reversed_samples = model.prefill_samples(index, (structures[0], list(reversed(structures[1])), structures[2]))
        swapped = model.score_samples(index, queries, reversed_samples)
        singleton_query = model.score_samples(index, queries[:1], together)
        differences = {
            "repeat": float((output["sample_state"]-repeat["sample_state"]).abs().max()),
            "query_permutation": float((output["sample_state"]-reverse["sample_state"].flip(1)).abs().max()),
            "single_query": float((output["sample_state"][:, :1]-singleton_query["sample_state"]).abs().max()),
            "sample_alone_vs_together": float((output["sample_state"][:1]-alone["sample_state"]).abs().max()),
            "sample_permutation": float((output["sample_state"]-swapped["sample_state"].flip(0)).abs().max())}
        if max(differences.values()) > 1e-5:
            raise AssertionError(f"sample/query isolation failed: {differences}")
        # Same weights, same history, genuine future source values through the
        # exact same predicted-record path, never future predicate/event labels.
        oracle = adapter.oracle(w.episode, w.cutoff)
        oracle_output = model(index, queries, [future_feedback(index, oracle, 0)])
        oracle_difference = float((probabilities(output)[0]-probabilities(oracle_output)[0]).abs().max())
        if oracle_difference == 0:
            raise AssertionError("oracle and sampled probabilities are identical on this real window")
        aggregate_error = float((probabilities(output)[0]-output["sample_state"].mean(0)).abs().max())
        mass_error = float((probabilities(output)[1]-output["sample_transition"].mean(0)).abs().max())
        if max(aggregate_error, mass_error) > 1e-6:
            raise AssertionError("aggregation is not the mean of per-sample probabilities")
        # Poison supervision outcomes: only query template/bindings/horizons may
        # enter the model. Keep contract identity fields unchanged.
        poisoned = [{**r, "future_state": {str(h): "unknown" for h in HORIZONS},
                     "first_transition_tick": None, "event_targets": [], "state_at_cutoff": "unknown"} for r in queries]
        poisoned_output = model.score_samples(index, poisoned, together)
        label_error = float((output["sample_state"]-poisoned_output["sample_state"]).abs().max())
        if label_error:
            raise AssertionError("supervision outcomes entered Qwen inputs")
    changed_shapes = parameter_shapes != {name: tuple(p.shape) for name, p in model.named_parameters()}
    if changed_shapes:
        raise AssertionError("changing runtime sample count changed learned parameter shapes")
    result = {"episode": w.episode, "cutoff": w.cutoff, "queries": len(queries), "samples": 2,
              "postcutoff_source_removal_bitwise_equal": equal, "max_observed_graph_tick": max(r["tick"] for r in index.semantic_records),
              "future_ticks": [t for t, _ in feedbacks[0]], "isolation_max_absolute_errors": differences,
              "oracle_vs_sampled_max_probability_difference": oracle_difference,
              "probability_mean_max_error": aggregate_error, "transition_mean_max_error": mass_error,
              "poisoned_supervision_max_error": label_error,
              "parameter_shapes_depend_on_k": changed_shapes, "prefixes": structures[2], "memory": memory(args.device)}
    write_json(args.output / "real_data_validation.json", result)
    log(phase="real_data_validation", **result)
    return result


def run(args):
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    valid_rows = read_supervision(args.supervision, ("VALID",))
    selected_valid, subset = valid_cohort(valid_rows, args)
    if args.prepare_subset_only:
        log(phase="subset_ready", windows=len(subset["windows"]), queries=len(selected_valid), path=str(args.subset_file),
            strata=dict(Counter(w["stratum"] for w in subset["windows"])))
        return
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.isdigit() or int(visible) == 0:
            raise ValueError("CUDA_VISIBLE_DEVICES must be the single nonzero GPU assigned by the coordinator")
        if not torch.cuda.is_available():
            raise RuntimeError("assigned GPU unavailable; explicitly select --device cpu")
        args.device = torch.device("cuda:0")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., 23*1024**3/total), 0)
        torch.cuda.reset_peak_memory_stats()
    else:
        args.device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if checkpoint:
        if checkpoint["predicate_ids"] != sorted(PREDICATES) or checkpoint["arguments"]["model"] != "qwen":
            raise ValueError("checkpoint must use the five-predicate B2 Qwen schema")
        args.width = int(checkpoint["arguments"]["width"])
        args.graph_mode = checkpoint["arguments"]["graph_mode"]
        args.qwen_path = Path(checkpoint["arguments"]["qwen_path"])
    config = json.loads(args.config.read_text())
    selected_train = [] if args.eval_only else selected_rows(read_supervision(args.supervision, ("TRAIN",)), args)
    windows = prepare_windows(selected_train+selected_valid, config, args)
    train = [w for w in windows if w.role == "train"]
    dev = [w for w in windows if w.role == "dev"]
    valid = [w for w in windows if w.role == "valid"]
    if not valid or (not args.eval_only and (not train or not dev)):
        raise ValueError("empty required TRAIN/dev/VALID cohort")
    store = WindowStore(config, args)
    if checkpoint:
        scales, observation_scales = checkpoint["state_scales"], checkpoint["observation_scales"]
    else:
        normalization = [store.load(w) for w in train[:args.scale_windows]]
        scales = input_scales(normalization)
        observation_scales = fit_observation_scales([w.index for w in normalization])
    event_types = (tuple(checkpoint["event_types"]) if checkpoint else
                   tuple(sorted({e["event_type_id"] for w in train for r in w.rows for e in r["event_targets"]}))
                   if args.event_head else ())
    model = TwoStageModel(observation_scales, model_path=args.qwen_path, device=args.device, width=args.width,
                          seed=args.seed, lora_lr=args.lora_lr, max_prefix_tokens=args.max_prefix_tokens,
                          graph_mode=args.graph_mode, event_types=event_types, history_tokens=args.history_tokens,
                          sample_batch_size=args.sample_batch_size, query_batch_size=args.query_batch_size)
    if checkpoint:
        load_parameters(model, checkpoint)
    if args.freeze_graph:
        model.graph.requires_grad_(False)
    if args.history_cache_dir:
        if not args.freeze_graph:
            raise ValueError("--history-cache-dir requires --freeze-graph")
        model.history_cache = FrozenHistoryCache(model.graph, config, args.cache_dir, args.history_cache_dir)
    adapter = CompactFutureAdapter(arrays=args.arrays, members=args.members, pose_pool=args.pose_pool,
                                    pose_checkpoints=args.pose_checkpoints, latency_checkpoints=args.latency_checkpoints,
                                    handover_checkpoints=args.handover_checkpoints, shared_rho=args.shared_rho,
                                    train_noise_scale=args.train_noise_scale) if args.future_mode != "history" or args.validate_real else None
    provenance = {"arguments": {k: str(v) if isinstance(v, (Path, torch.device)) else v for k, v in vars(args).items()},
                  "sources": {str(p): digest(p) for p in (Path(__file__), Path(__file__).with_name("two_stage.py"),
                      Path(__file__).with_name("qwen_predicate.py"), Path(__file__).with_name("sampling.py"),
                      Path(__file__).with_name("frozen_history.py"))},
                  "compact": None if adapter is None else adapter.describe(), "qwen_kernels": model.kernel_config,
                  "subset_file": str(args.subset_file), "subset_sha256": digest(args.subset_file),
                  "cohorts": {role: [[w.episode, w.cutoff] for w in windows if w.role == role] for role in ("train", "dev", "valid")},
                  "excluded_predicates": sorted(EXCLUDED_PREDICATES), "checkpoint_initialization": str(args.checkpoint),
                  "training_objective": "known-state BCE + censor-aware timing NLL (+ optional event BCE) of mean probabilities",
                  "training_sampling": {"positive_weight": args.positive_weight, "transition_weight": args.transition_weight,
                      "policy": "B2 mean-row-weight window draws, weighted query draws with replacement" if
                                args.positive_weight or args.transition_weight else "shuffled windows, uniform queries without replacement",
                      "prior_correction": False},
                  "oracle_exception": args.future_mode == "oracle",
                  "in_sample_compact_train": True,
                  "limits": ["pose model was fitted on fitting TRAIN cutoffs; its TRAIN samples can be sharper than held-out samples",
                             "train-noise-scale is optional continuous inflation, not an estimated DEV error profile",
                             "after 1 s: C2 physical extension, not learned five-second distributions",
                             "no future predicate/event truth, capture tokens, or future SUMO observations",
                             "causal graph history, source indexing and Qwen history prefill shared; future branches isolated"],
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(args.output / "provenance.json", provenance)
    log(phase="ready", elapsed_seconds=time.perf_counter()-start, device=str(args.device),
        train_windows=len(train), dev_windows=len(dev), valid_windows=len(valid), **memory(args.device))
    validation = validate_real(model, adapter, valid[0], args, store, scales) if args.validate_real else None
    curve, steps, visited = [], [], set()
    best_step, best_score = (checkpoint["step"], checkpoint["dev_score"]) if args.eval_only else (0, None)
    if not args.eval_only:
        initial, _, _, _ = evaluate(model, adapter, dev, args, store, scales, args.dev_samples)
        best_score = initial["selection_score"]
        if best_score is None:
            raise ValueError("no usable dev selection score")
        best_path = args.output / "best.pt"
        save_checkpoint(best_path, model, args, scales, observation_scales, event_types, 0, best_score)
        curve.append({"step": 0, "dev": initial})
        parameters = [{"params": [p for name, p in model.named_parameters() if p.requires_grad and "lora_" not in name], "lr": args.lr},
                      {"params": [p for name, p in model.named_parameters() if p.requires_grad and "lora_" in name], "lr": args.lora_lr}]
        optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay)
        rng = random.Random(f"two-stage-training:{args.seed}")
        order = list(range(len(train)))
        weighted = args.positive_weight or args.transition_weight
        weights = [sum(training_weight(row, args) for row in w.rows)/len(w.rows) for w in train] if weighted else None
        fitting_start = time.perf_counter()
        for step in range(1, args.steps+1):
            if args.time_budget_seconds and time.perf_counter()-fitting_start >= args.time_budget_seconds:
                break
            if weighted:
                chosen = rng.choices(train, weights=weights, k=1)[0]
            else:
                if (step-1) % len(order) == 0:
                    rng.shuffle(order)
                chosen = train[order[(step-1) % len(order)]]
            w = store.load(chosen)
            w.index.scales = scales
            visited.add((w.episode, w.cutoff))
            queries = (rng.choices(w.rows, weights=[training_weight(row, args) for row in w.rows], k=args.batch_size)
                       if weighted else rng.sample(w.rows, min(args.batch_size, len(w.rows))))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            begin = time.perf_counter()
            prepared, timing = prepare_forward(model, adapter, w.index, args, args.train_samples,
                                                window_seed(args.seed, w.episode, w.cutoff, step), training=True)
            output = model.score_samples(w.index, queries, prepared)
            loss, components = probability_loss(output, queries)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite aggregate training loss")
            loss.backward()
            gradient = {}
            for label, term in (("lora", "lora_"), ("query", "query_token."), ("graph", "graph.")):
                grads = [p.grad.float().norm().square() for name, p in model.named_parameters() if term in name and p.grad is not None]
                gradient[label] = float(torch.stack(grads).sum().sqrt()) if grads else None
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            sync(args.device)
            record = {"step": step, "loss": float(loss.detach()), "grad_norm": float(norm), "gradient_groups": gradient,
                      "seconds": time.perf_counter()-begin, "components": {k: float(v.detach()) for k, v in components.items()},
                      "timing": timing}
            steps.append(record)
            log(phase="train", **record)
            del prepared, output, loss, components
            expired = args.time_budget_seconds and time.perf_counter()-fitting_start >= args.time_budget_seconds
            if step % args.eval_every == 0 or step == args.steps or expired:
                metrics, _, _, _ = evaluate(model, adapter, dev, args, store, scales, args.dev_samples)
                score = metrics["selection_score"]
                curve.append({"step": step, "dev": metrics})
                if score < best_score:
                    best_step, best_score = step, score
                    save_checkpoint(best_path, model, args, scales, observation_scales, event_types, step, score)
                log(phase="dev", step=step, score=score, best_step=best_step)
            if expired:
                break
        load_parameters(model, torch.load(best_path, map_location="cpu", weights_only=False))
    valid_metrics = None
    if not args.skip_valid:
        valid_metrics, target_rows, predictions, spread = evaluate(model, adapter, valid, args, store, scales)
        actual = [row_key(r) for r in predictions]
        expected = {row_key(r) for r in selected_valid}
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("VALID prediction cardinality/identity mismatch")
        write_rows(args.output / "predictions_VALID.jsonl", predictions)
        write_rows(args.output / "sample_spread_VALID.jsonl", spread)
        write_rows(args.output / "supervision_VALID.jsonl", target_rows)
    benchmarks = []
    if args.benchmark_k:
        if args.future_mode != "sampled":
            raise ValueError("--benchmark-k requires sampled futures")
        parameter_shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
        for k in args.benchmark_k:
            trials = []
            for _ in range(args.benchmark_repeats):
                measured, _, _, _ = evaluate(model, adapter, valid[:args.benchmark_windows], args, store, scales, k)
                trials.append(measured)
            if parameter_shapes != {name: tuple(p.shape) for name, p in model.named_parameters()}:
                raise AssertionError("K changed learned parameter shapes")
            benchmarks.append({"k": k, "median_ms_per_query": float(np.median([m["runtime_ms_per_query"] for m in trials])),
                               "parameter_shapes_unchanged": True, "trials": trials})
            write_json(args.output / "benchmark.json", benchmarks)
    result = {"best_step": best_step, "best_dev_score": best_score, "training_steps": steps, "curve": curve,
              "valid": valid_metrics, "validation": validation, "benchmark": benchmarks,
              "total_seconds": time.perf_counter()-start, "train_windows_visited": len(visited),
              "valid_complete": not args.skip_valid and len(selected_valid) == len(valid_rows),
              "memory": memory(args.device)}
    if model.history_cache is not None:
        result["frozen_history_cache"] = {"hits": model.history_cache.hits, "misses": model.history_cache.misses,
                                          "directory": str(args.history_cache_dir)}
    write_json(args.output / "metrics.json", result)
    log(phase="complete", output=str(args.output), seconds=result["total_seconds"], **memory(args.device))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=("qwen", "temporal", "rules"), default="qwen")
    p.add_argument("--temporal-per-sample", action="store_true", help="extra GRU control: conditional correction per sample, then mean probabilities")
    p.add_argument("--context-features", action="store_true", help="Task E tuple histories and coherent per-sample rule refinement")
    p.add_argument("--prepare-features-only", action="store_true")
    p.add_argument("--feature-cache-dir", type=Path, default=WORKSPACE / "runs/task_e_features")
    p.add_argument("--context-ablation", action="store_true", help="export declared no-tuple and no-graph context ablations")
    p.add_argument("--rule-prior-strength", type=float, default=128., help="fixed per-sample shrinkage strength, independent of runtime K; not an MC sample count")
    p.add_argument("--rule-only-prior", action="store_true", help="rules-only control using exactly the learned models' fixed input prior")
    p.add_argument("--dev-k-sweep", type=lambda s: [int(k) for k in s.split(",")], help="evaluate selected checkpoint on TRAIN-dev at these runtime K values")
    p.add_argument("--graph-device", choices=("cpu", "cuda"), default="cuda", help="CPU permits parallel frozen-prefix preparation; Qwen remains on the assigned GPU")
    p.add_argument("--config", type=Path, default=ROOT / "Dataset/world_model/configs/state_distribution_5s.json")
    p.add_argument("--supervision", type=Path, default=WORKSPACE / "supervision")
    p.add_argument("--cache-dir", type=Path, default=WORKSPACE / "runs/task_b_cache")
    p.add_argument("--subset-file", type=Path, default=WORKSPACE / "runs/two_stage_subset_windows.json")
    p.add_argument("--subset-size", type=int, default=150)
    p.add_argument("--prepare-subset-only", action="store_true")
    choice = p.add_mutually_exclusive_group()
    choice.add_argument("--train", action="store_true", help="default operation: fit TRAIN and select on dev")
    choice.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--full-valid", action="store_true")
    p.add_argument("--skip-valid", action="store_true")
    p.add_argument("--future-mode", choices=("sampled", "oracle", "history"), default="sampled")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--train-samples", type=int, default=2)
    p.add_argument("--dev-samples", type=int, default=2)
    p.add_argument("--sample-batch-size", type=int, default=2, help="fixed physical future batch size; pad/discard last block")
    p.add_argument("--query-batch-size", type=int, default=8, help="fixed physical Qwen query batch per sample; pad/discard last block")
    p.add_argument("--train-noise-scale", type=float, default=1., help="TRAIN-only continuous-law inflation sensitivity, not fitted DEV noise")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--qwen-path", type=Path, default=Path(DEFAULT_QWEN))
    p.add_argument("--max-prefix-tokens", type=int, default=256, help="history + future token budget; runtime, not checkpoint shape")
    p.add_argument("--history-tokens", type=int, default=128)
    p.add_argument("--graph-mode", choices=("graph_summary", "graph_full"), default="graph_summary")
    p.add_argument("--freeze-graph", action="store_true", help="explicit cheaper ablation; retains checkpoint graph weights")
    p.add_argument("--history-cache-dir", type=Path, help="optional source/weight-keyed frozen graph history cache; timings report hits")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--event-head", action="store_true")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--train-windows", type=int, default=6)
    p.add_argument("--dev-windows", type=int, default=3)
    p.add_argument("--valid-windows", type=int, default=0, help="debug cap within the fixed subset; full-valid overrides")
    p.add_argument("--train-queries-per-window", type=int, default=32)
    p.add_argument("--dev-queries-per-window", type=int, default=16)
    p.add_argument("--eval-queries-per-window", type=int, default=0, help="debug cap; zero/full-valid retains all")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--positive-weight", type=float, default=0., help="TRAIN-only B2 oversampling of queries with any positive future state")
    p.add_argument("--transition-weight", type=float, default=0., help="TRAIN-only B2 oversampling of observed transitions")
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--scale-windows", type=int, default=3)
    p.add_argument("--window-cache-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--time-budget-seconds", type=float, default=0)
    p.add_argument("--arrays", type=Path, default=COMPACT / "arrays")
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--pose-pool", type=Path, default=COMPACT / "logs/pool_lambda_micro.json")
    p.add_argument("--pose-checkpoints", type=Path, nargs="+")
    p.add_argument("--latency-checkpoints", type=Path, nargs="+")
    p.add_argument("--handover-checkpoints", type=Path, nargs="+")
    p.add_argument("--shared-rho", type=float, default=0.)
    p.add_argument("--validate-real", action="store_true")
    p.add_argument("--benchmark-k", type=lambda s: [int(k) for k in s.split(",")], help="e.g. 1,4,8,16; same weights and windows")
    p.add_argument("--benchmark-windows", type=int, default=1)
    p.add_argument("--benchmark-repeats", type=int, default=1)
    args = p.parse_args()
    # B2 checkpoint serialization supports scalar Paths, not nested Paths.
    for name in ("pose_checkpoints", "latency_checkpoints", "handover_checkpoints"):
        if getattr(args, name):
            setattr(args, name, list(map(str, getattr(args, name))))
    args.queries_per_window = 0
    args.valid_queries_per_window = 0
    positive = ("samples", "train_samples", "dev_samples", "sample_batch_size", "query_batch_size", "subset_size", "batch_size", "eval_batch_size",
                "scale_windows", "window_cache_size", "eval_every", "benchmark_windows", "benchmark_repeats")
    nonnegative = ("steps", "train_windows", "dev_windows", "valid_windows", "train_queries_per_window", "dev_queries_per_window",
                   "eval_queries_per_window", "time_budget_seconds", "positive_weight", "transition_weight")
    if any(getattr(args, k) < 1 for k in positive) or any(getattr(args, k) < 0 for k in nonnegative):
        p.error("invalid size, limit or time budget")
    if not 1 <= args.threads <= 16 or not 1 <= args.members <= 5 or not 0 <= args.shared_rho <= 1 or args.train_noise_scale <= 0:
        p.error("invalid threads/members/rho/noise scale")
    if args.rule_prior_strength <= 0:
        p.error("rule prior strength must be positive")
    if not 0 < args.history_tokens <= args.max_prefix_tokens-10:
        p.error("prefix budget must leave at least ten future tokens")
    if args.eval_only and not args.checkpoint and args.model != "rules":
        p.error("--eval-only requires --checkpoint")
    if args.benchmark_k and any(k < 1 for k in args.benchmark_k):
        p.error("benchmark K must be positive")
    for path in (args.output, args.cache_dir, args.subset_file, args.feature_cache_dir, *([args.history_cache_dir] if args.history_cache_dir else [])):
        if not path.resolve().is_relative_to((WORKSPACE / "runs").resolve()):
            p.error("all artifacts must stay under Dataset/world_model/runs/predicate_stage/runs")
    if args.context_features:
        run_task_e(args)
    else:
        if args.model != "qwen" or args.prepare_features_only or args.context_ablation:
            p.error("Task E options require --context-features")
        run(args)


if __name__ == "__main__":
    main()
