"""Run Task C on real source states or Task A supervision; write contract JSONL.

Example (fast CPU development loop, no TEST labels):
  python -m Dataset.world_model.predicate_stage.run_sampling --split TRAIN \
    --dev-only --cutoff-stride 50 --max-cutoffs 4 --samples 64 \
    --output /mnt/data2/weizhiwei/AERO_WORLD/Dataset/world_model/runs/predicate_stage/runs/task_c_dev

Without --supervision, only predicate/transition labels are reconstructed.
Bootstrap mode cannot provide event targets and explicitly omits event output.
Oracle truth never enters model features.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from Dataset.world_model.data.provenance import resolve_source_split
from Dataset.world_model.model.compact_data import load_episode
from .rule_predicates import (RuleContext, EXPORTABLE, HORIZONS, OFFSETS, SEPARATION,
                              CAPACITY, OCCUPANCY, binding_key, evaluate, probabilities, rows)
from .sampling import CompactSampler, causal_pose_arrays, communication_features

COMPACT = Path("/mnt/data2/weizhiwei/AERO_WORLD/Dataset/world_model/runs/compact")
DEV = {"L2-2_v2__seed00", "L4-5_v1__seed00", "L5-3_v1__seed00"}


def family(pid):
    return "separation" if pid == SEPARATION else "corridor" if pid in (CAPACITY, OCCUPANCY) else "communication"


def bootstrap_queries(context, cutoffs):
    """Development supervision from formal truth; no labels become model inputs."""
    needed = {c+int(h) for c in cutoffs for h in OFFSETS}
    formal = defaultdict(dict)
    for r in context.formal_rows(needed):
        if r["predicate_id"] not in EXPORTABLE:
            continue
        key = (r["predicate_id"], r["tuple_id"])
        formal[key][r["tick"]] = r
    queries = []
    for (pid, tid), series in sorted(formal.items()):
        for c in cutoffs:
            if c not in series or series[c]["value"] not in ("true", "false"):
                continue
            anchor = series[c]
            first, censored, previous = None, c, anchor["value"]
            for h in OFFSETS[1:]:
                value = series.get(c+int(h), {}).get("value")
                if value not in ("true", "false"):
                    break
                censored = c+int(h)
                if first is None and value != previous:
                    first = c+int(h)
                previous = value
            queries.append({"episode": context.episode, "split": resolve_source_split(context.episode),
                            "dev": context.episode in DEV, "family": family(pid), "cutoff": c,
                            "predicate_id": pid, "tuple_id": tid, "bindings": anchor["bindings"],
                            "state_at_cutoff": anchor["value"], "horizons": list(HORIZONS),
                            "future_state": {str(h): series[c+h]["value"] if c+h in series else "unknown" for h in HORIZONS},
                            "first_transition_tick": first, "transition_censored_at": censored,
                            "event_targets_available": False,
                            "source": {"development": "formal graph and recorded communication truth",
                                       "event_targets": "unavailable; bootstrap does not emit event predictions"}})
    return queries


def oracle_check(context, queries, poses):
    """Rule accuracy on TRUE future states, deduplicated by predicate/tuple/tick."""
    from Dataset.semantic_simulation.compute_comm import _communication_predicate_values
    ticks = sorted({q["cutoff"]+int(h) for q in queries for h in (0, *HORIZONS)})
    states = context.true_states(ticks, poses)
    lookup = {t: i for i, t in enumerate(ticks)}
    stats = defaultdict(lambda: defaultdict(int))
    examples, by_binding = [], {}
    references = {}
    for q in queries:
        for h in (0, *HORIZONS):
            value = q["state_at_cutoff"] if h == 0 else q["future_state"][str(h)]
            key = (q["predicate_id"], q["tuple_id"], q["cutoff"]+h)
            row = {"tick": q["cutoff"]+h, "predicate_id": q["predicate_id"], "bindings": q["bindings"], "value": value}
            if key in references and references[key] != row:
                raise ValueError(f"contradictory supervision: {key}")
            references[key] = row
    for r in references.values():
        pid = r["predicate_id"]
        key = (pid, binding_key(r["bindings"]))
        if key not in by_binding:
            by_binding[key] = evaluate(states, context, pid, r["bindings"])[0]
        actual = by_binding[key][lookup[r["tick"]]]
        expected = r["value"]
        s = stats[pid]
        s["formal_rows"] += 1
        if pid.startswith("communication."):
            source = context.communication_row(r["tick"], r["bindings"])
            if source is not None:
                current = _communication_predicate_values(source)[pid]
                if current in ("true", "false") and np.isfinite(actual):
                    s["current_producer_compared"] += 1
                    s["current_producer_matches"] += int(bool(actual) == (current == "true"))
        if expected not in ("true", "false"):
            s["formal_unknown"] += 1
        elif not np.isfinite(actual):
            s["rule_unknown_on_known"] += 1
        else:
            s["known_compared"] += 1
            s["formal_positive"] += int(expected == "true")
            s["matches"] += int(bool(actual) == (expected == "true"))
            if bool(actual) != (expected == "true") and len(examples) < 20:
                examples.append({"episode": context.episode, "tick": r["tick"], "predicate_id": pid,
                                 "bindings": r["bindings"], "formal": expected, "computed": bool(actual)})
    result = {}
    for pid, s in stats.items():
        result[pid] = dict(s)
        result[pid]["agreement"] = s["matches"]/s["known_compared"] if s["known_compared"] else None
    return {"by_predicate": result, "mismatch_examples": examples}


def score(predictions, queries):
    from sklearn.metrics import roc_auc_score
    stats = defaultdict(lambda: [[], []])
    transition, censored_nll, time_bin_correct, time_bin_n = [], [], 0, 0
    censored_positive, censored_negative, coarsened = 0, 0, 0
    for p, q in zip(predictions, queries):
        for h in HORIZONS:
            label = q["future_state"][str(h)]
            if label not in ("true", "false"):
                continue
            for group in ("all", q["predicate_id"], q["family"]):
                stats[(group, str(h))][0].append(label == "true")
                stats[(group, str(h))][1].append(p["p_future_true"][str(h)])
        first = q["first_transition_tick"]
        observed = first is not None and first <= q["cutoff"]+50
        censor = min(50, q["transition_censored_at"]-q["cutoff"])
        masses = p["transition_hazard"] + [max(0., 1-p["p_transition"])]
        if observed:
            if first > q["transition_censored_at"]:
                raise ValueError("observed transition lies after censor time")
            bin_id = int(np.searchsorted(HORIZONS, first-q["cutoff"], side="left"))
            likelihood = masses[bin_id]
            censored_positive += int(censor < 50)
        else:
            # Five-bin output cannot resolve a censor inside a bin. Score the
            # observed survival through the last fully observed bin boundary.
            finished = int(np.searchsorted(HORIZONS, censor, side="right"))
            likelihood = sum(masses[finished:])
            censored_negative += int(censor < 50)
            coarsened += int(censor not in (0, *HORIZONS))
        censored_nll.append(-np.log(max(likelihood, 1e-6)))
        # Binary full-window outcomes and accuracy require complete follow-up,
        # regardless of whether an early transition happened to be observed.
        if censor == 50:
            delta = first-q["cutoff"] if observed else 51
            bin_id = int(np.searchsorted(HORIZONS, delta, side="left"))
            transition.append((observed, p["p_transition"], -np.log(max(masses[bin_id], 1e-6))))
            time_bin_correct += int(np.argmax(masses) == bin_id)
            time_bin_n += 1
    result = defaultdict(dict)
    for (group, h), (yy, pp) in stats.items():
        y, p = np.asarray(yy, float), np.asarray(pp)
        clipped = np.clip(p, 1e-6, 1-1e-6)
        result[group][h] = {"n": len(y), "positive": int(y.sum()), "brier": float(np.mean((y-p)**2)),
                            "log_loss_clip_1e-6": float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))),
                            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}
    if transition:
        t = np.asarray(transition)
        result["transition"] = {"n": len(t), "positive": int(t[:, 0].sum()),
                                 "brier": float(np.mean((t[:, 0]-t[:, 1])**2)),
                                 "time_nll_clip_1e-6": float(t[:, 2].mean()),
                                 "time_bin_accuracy_including_no_event": time_bin_correct/time_bin_n}
    result["transition_censor_aware"] = {"n": len(censored_nll),
        "time_nll_clip_1e-6": float(np.mean(censored_nll)), "censored_with_transition": censored_positive,
        "censored_without_transition": censored_negative, "censors_coarsened_to_bin_boundary": coarsened}
    return dict(result)


def predict_cutoff(sampler, a, context, cutoff, queries, samples, rho, seed, inferred=None, *, comm_arrays=None):
    start = time.perf_counter()
    states = sampler.sample(a, cutoff, context, samples=samples, seed=seed, shared_rho=rho, inferred=inferred,
                            comm_arrays=comm_arrays)
    predictions = []
    for q in queries:
        owner = None
        if q["family"] == "communication":
            row = context.communication_row(cutoff, q["bindings"])
            if row is None:
                raise ValueError(f"cutoff communication identity has no source row: {q}")
            owner = row["entity_id"]
            if sampler.comm == []:
                raise ValueError("communication queries require a real communication checkpoint")
        truth = evaluate(states, context, q["predicate_id"], q["bindings"], owner=owner, fixed_scope=True)
        p = {k: q[k] for k in ("episode", "cutoff", "predicate_id", "tuple_id")}
        event_rules = context.events[q["predicate_id"]] if "event_targets" in q else ()
        p.update(probabilities(truth, states.ticks, cutoff, q["state_at_cutoff"], event_rules))
        p["model"] = f"compact_micro_shrink_link_physics_rules_S{samples}_rho{rho:g}_M{len(sampler.pose)}"
        predictions.append(p)
    elapsed = (time.perf_counter()-start)*1000
    for p in predictions:
        p["runtime_ms_per_query"] = elapsed/len(predictions)
    return predictions, elapsed


def verify_comm_features(prepared, cache_path):
    """Compare actual inference inputs with cached training-builder columns.

    Reads keys, features and causal physics anchors only; no cached targets.
    """
    with np.load(cache_path, allow_pickle=True) as cache:
        names = list(cache["names"])
        X, physics = cache["X"], cache["physics"]
        keys = zip(cache["episode"], cache["cutoff"], cache["entity"], cache["horizon"])
        lookup = {tuple(key): i for i, key in enumerate(keys)}
    compared, max_error, anchor_error = 0, 0., 0.
    episodes, examples = set(), []
    for _, context, grouped, a in prepared:
        index = {e: i for i, e in enumerate(a.entities)}
        for cutoff in sorted(grouped):
            ents = np.array([index[e] for e in context.comm.get(cutoff, {})
                             if e in index and a.present[cutoff, index[e]] and
                             any((a.episode, cutoff, e, hi) in lookup for hi in range(2))], dtype=int)
            if not len(ents):
                continue
            features, actual_names, anchors = communication_features(a, cutoff, ents, context, return_physics=True)
            if actual_names != names:
                raise ValueError("communication feature names/order differ from cache")
            for hi in range(2):
                for j, e in enumerate(ents):
                    key = a.episode, cutoff, a.entities[e], hi
                    if key not in lookup:
                        continue  # cache requires a known future target; inference does not
                    k = lookup[key]
                    diff = np.abs(features[hi, j]-X[k])
                    max_error = max(max_error, float(diff.max()))
                    anchor_error = max(anchor_error, abs(float(anchors[hi, j]-physics[k])))
                    compared += 1
                    episodes.add(a.episode)
                    if np.any(diff) and len(examples) < 5:
                        examples.append({"key": list(key), "columns": [names[i] for i in np.flatnonzero(diff)]})
    if not compared:
        raise ValueError("no selected rows overlap communication cache")
    result = {"cache": str(cache_path), "rows": compared, "columns": len(names),
              "episodes": sorted(episodes), "max_absolute_feature_error": max_error,
              "max_absolute_physics_anchor_error": anchor_error, "mismatches": examples}
    if max_error or anchor_error:
        raise ValueError(f"feature parity failed: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("Dataset/world_model/configs/state_distribution_5s.json"))
    parser.add_argument("--arrays", type=Path, default=COMPACT/"arrays")
    parser.add_argument("--supervision", type=Path)
    parser.add_argument("--split", choices=("TRAIN", "VALID"), default="VALID")
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--episodes", help="comma-separated episode IDs; never TEST")
    parser.add_argument("--cutoff-stride", type=int, default=5)
    parser.add_argument("--max-cutoffs", type=int, default=0, help="per episode, evenly spaced; 0 means all")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--shared-rho", type=float, default=0.)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--pose-checkpoints", nargs="+", type=Path)
    parser.add_argument("--comm-checkpoints", nargs="+", type=Path, help="physics-anchored latency ensemble")
    parser.add_argument("--handover-checkpoints", nargs="+", type=Path)
    parser.add_argument("--pose-pool", type=Path, default=COMPACT/"logs/pool_lambda_micro.json")
    parser.add_argument("--verify-comm-features", type=Path, help="compare selected real rows against a compact_comm cache")
    parser.add_argument("--device", choices=("cpu", "cuda:4"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--benchmark-samples", help="comma-separated S values; repeats inference for honest end-to-end timing")
    parser.add_argument("--benchmark-rhos", default="0,0.5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16 or args.cutoff_stride < 5 or args.cutoff_stride % 5:
        parser.error("threads must be 1..16; cutoff-stride must be a positive multiple of 5")
    if not 1 <= args.members <= 5:
        parser.error("members must be 1..5")
    if args.dev_only and args.split != "TRAIN":
        parser.error("--dev-only requires --split TRAIN")
    args.output.resolve().relative_to(Path("/mnt/data2/weizhiwei/AERO_WORLD/Dataset/world_model/runs/predicate_stage/runs"))
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    config = json.loads(args.config.read_text())
    selected = set(args.episodes.split(",")) if args.episodes else set(config["episodes"])
    if args.episodes and any(resolve_source_split(e) == "TEST" for e in selected):
        parser.error("TEST is outside the Task C development/evaluation scope")
    selected = {e for e in selected if resolve_source_split(e) == args.split and (not args.dev_only or e in DEV)}
    supplied, excluded = defaultdict(list), defaultdict(int)
    if args.supervision:
        for q in rows(args.supervision):
            if q["episode"] in selected:
                if resolve_source_split(q["episode"]) != q["split"]:
                    raise ValueError("supervision split disagrees with split authority")
                if q["predicate_id"] not in EXPORTABLE:
                    excluded[q["predicate_id"]] += 1
                    continue
                if q["state_at_cutoff"] not in ("true", "false"):
                    raise ValueError("eligible supervision requires known cutoff truth")
                supplied[q["episode"]].append(q)
        selected &= supplied.keys()
    if not selected:
        raise ValueError("no episodes selected")
    if not args.supervision:
        print(json.dumps({"phase": "bootstrap", "event_targets": "unavailable",
                          "event_predictions": "disabled; provide --supervision for event evaluation"}), flush=True)
    prepared, all_queries, oracle = [], [], {}
    for episode in sorted(selected):
        context = RuleContext(episode)
        a = load_episode(args.arrays/f"{episode}.npz")
        cutoffs = (sorted({q["cutoff"] for q in supplied[episode]}) if args.supervision else
                   list(range(50, a.present.shape[0]-50, args.cutoff_stride)))
        cutoffs = [c for c in cutoffs if (c-50) % args.cutoff_stride == 0]
        if args.max_cutoffs and len(cutoffs) > args.max_cutoffs:
            cutoffs = [cutoffs[i] for i in np.linspace(0, len(cutoffs)-1, args.max_cutoffs, dtype=int)]
        if not cutoffs:
            raise ValueError(f"no selected cutoff in {episode}")
        queries = ([q for q in supplied[episode] if q["cutoff"] in cutoffs] if args.supervision else
                   bootstrap_queries(context, cutoffs))
        queries.sort(key=lambda q: (q["cutoff"], q["predicate_id"], q["tuple_id"]))
        if not queries:
            raise ValueError(f"no scoped queries in {episode}")
        poses = context.pose_sources(max(cutoffs)+50)
        oracle[episode] = oracle_check(context, queries, poses)
        oracle[episode]["source"] = context.source_descriptor
        # Predictor consumes only the <= cutoff prefix through cell_features.
        comm_arrays = a
        a, supplement_count = causal_pose_arrays(a, poses, max(cutoffs))
        grouped = defaultdict(list)
        for q in queries:
            grouped[q["cutoff"]].append(q)
        prepared.append((a, context, grouped, comm_arrays))
        all_queries.extend(queries)
        print(json.dumps({"phase": "oracle", "episode": episode, "queries": len(queries),
                          "source_pose_rows_connected": supplement_count, **oracle[episode]}), flush=True)
    (args.output/"oracle.json").write_text(json.dumps(oracle, indent=2, allow_nan=False)+"\n")
    if args.oracle_only:
        return
    if args.verify_comm_features:
        parity = verify_comm_features(prepared, args.verify_comm_features)
        (args.output/"feature_parity.json").write_text(json.dumps(parity, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"phase": "feature_parity", **parity}), flush=True)
    pose_paths = args.pose_checkpoints or [COMPACT/f"runs/f{i}_mix3_aug_micro/model.pt" for i in range(1, args.members+1)]
    comm_paths = args.comm_checkpoints or [COMPACT/f"runs/comm4_e{i}/model.pt" for i in range(1, args.members+1)]
    handover_paths = args.handover_checkpoints or [COMPACT/f"runs/comm2_e{i}/model.pt" for i in range(1, args.members+1)]
    sampler = CompactSampler(pose_paths, comm_paths, args.device, handover_checkpoints=handover_paths,
                             pose_pool=args.pose_pool)
    initialization_s = time.perf_counter()-start
    # One real warmup is excluded from inference timing and recorded explicitly.
    a, ctx, groups, comm_arrays = prepared[0]
    warm_start = time.perf_counter()
    sampler.sample(a, next(iter(groups)), ctx, samples=min(16, args.samples), seed=args.seed, comm_arrays=comm_arrays)
    warmup_ms = (time.perf_counter()-warm_start)*1000
    configs = [(args.samples, args.shared_rho)]
    if args.benchmark_samples:
        configs.extend((int(s), float(rho)) for s in args.benchmark_samples.split(",") for rho in args.benchmark_rhos.split(","))
    configs = list(dict.fromkeys(configs))
    summary = {"initialization_and_oracle_s": initialization_s, "warmup_ms": warmup_ms,
               "device": args.device, "threads": args.threads, "query_count": len(all_queries),
               "supervision": str(args.supervision) if args.supervision else "development formal-source queries",
               "selected_episodes": sorted(selected), "pose_checkpoints": [str(p) for p in pose_paths],
               "comm_checkpoints": [str(p) for p in comm_paths],
               "handover_checkpoints": [str(p) for p in handover_paths],
               "pose_pool": str(args.pose_pool), "latency_shrinkage": sampler.latency_shrinkage,
               "excluded_predicate_queries": dict(excluded), "exportable_predicates": list(EXPORTABLE),
               "event_predictions_enabled": bool(args.supervision), "runs": []}
    for run_index, (samples, rho) in enumerate(configs):
        predictions, timings = [], []
        for a, context, grouped, comm_arrays in prepared:
            for cutoff, queries in sorted(grouped.items()):
                seed = args.seed+int.from_bytes(hashlib.blake2b(f"{a.episode}:{cutoff}".encode(), digest_size=4).digest(), "little")
                pred, elapsed = predict_cutoff(sampler, a, context, cutoff, queries, samples, rho, seed,
                                              comm_arrays=comm_arrays)
                predictions.extend(pred)
                timings.append({"episode": a.episode, "cutoff": cutoff, "queries": len(queries), "ms": elapsed})
        directory = args.output if run_index == 0 else args.output/f"S{samples}_rho{rho:g}"
        directory.mkdir(exist_ok=True)
        with (directory/f"predictions_{args.split}.jsonl").open("w") as stream:
            for p in predictions:
                stream.write(json.dumps(p, allow_nan=False)+"\n")
        metrics = score(predictions, all_queries)
        result = {"samples": samples, "shared_rho": rho, "prediction_file": str(directory/f"predictions_{args.split}.jsonl"),
                  "ms_per_query": sum(t["ms"] for t in timings)/len(predictions),
                  "ms_per_cutoff_mean": float(np.mean([t["ms"] for t in timings])),
                  "cutoff_timings": timings, "metrics": metrics}
        summary["runs"].append(result)
        print(json.dumps({"phase": "sampling", "S": samples, "rho": rho,
                          "ms_per_query": result["ms_per_query"], "all_metrics": metrics.get("all"),
                          "transition": metrics.get("transition")}), flush=True)
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    summary["wall_s"] = time.perf_counter()-start
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
