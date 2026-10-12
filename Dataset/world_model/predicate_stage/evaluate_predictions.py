"""Strict matched predicate evaluation on a declared VALID cohort.

Run with --supervision <VALID.jsonl> --run name=<predictions.jsonl> ...
Full-VALID predictions may cover a subset cohort: extra keys are counted and
must belong to --universe. Missing/duplicate queries and invalid probabilities
are errors. --mc name=S explicitly applies Jeffreys smoothing to MC counts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .qwen_predicate import EXCLUDED_PREDICATES, HORIZONS, PREDICATES
from .train_predicate import WORKSPACE, row_key, write_json


def read_unique(path, *, supervision=False):
    result, excluded = {}, 0
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            key = row_key(row)
            if supervision and row["predicate_id"] in EXCLUDED_PREDICATES:
                excluded += 1
                continue
            if key in result:
                raise ValueError(f"duplicate query in {path}:{number}: {key}")
            if row["predicate_id"] not in PREDICATES:
                raise ValueError(f"unsupported predicate in {path}:{number}")
            result[key] = row
    if not result:
        raise ValueError(f"empty cohort in {path}")
    return result, excluded


def validate_prediction(row):
    expected = set(map(str, HORIZONS))
    if set(row["p_future_true"]) != expected:
        raise ValueError(f"invalid state horizons: {row_key(row)}")
    bins = row["transition_hazard"]
    if len(bins) != len(HORIZONS):
        raise ValueError(f"invalid transition bins: {row_key(row)}")
    numbers = list(row["p_future_true"].values()) + bins + [row["p_transition"]]
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 1 for x in numbers):
        raise ValueError(f"invalid probability: {row_key(row)}")
    if abs(sum(bins)-row["p_transition"]) > 1e-7:
        raise ValueError(f"transition masses disagree with P(transition): {row_key(row)}")
    runtime = row.get("runtime_ms_per_query")
    if runtime is None or isinstance(runtime, bool) or not math.isfinite(runtime) or runtime < 0:
        raise ValueError(f"missing/invalid measured runtime: {row_key(row)}")


def smooth_mc(row, samples):
    bins = np.array(row["transition_hazard"] + [1-row["p_transition"]], dtype=float)
    bins = (bins*samples+.5)/(samples+3.)
    return {**row, "p_future_true": {h: (p*samples+.5)/(samples+1.) for h, p in row["p_future_true"].items()},
            "transition_hazard": bins[:5].tolist(), "p_transition": float(bins[:5].sum())}


def fitted_law(train, cohort):
    states = defaultdict(lambda: [0, 0])
    # Binomial discrete hazards: events and completed-bin survivals both
    # contribute. A bin crossed by censoring remains unresolved.
    transitions = defaultdict(lambda: np.zeros((5, 2)))
    fitting = 0
    for row in train.values():
        if row["split"] != "TRAIN":
            raise ValueError("baseline fitting input must be TRAIN")
        if row["dev"]:
            continue
        fitting += 1
        identity = row["predicate_id"], row["state_at_cutoff"]
        for h in HORIZONS:
            truth = row["future_state"][str(h)]
            if truth in ("true", "false"):
                counts = states[(*identity, h)]
                counts[0] += truth == "true"
                counts[1] += 1
        if row["state_at_cutoff"] == "unknown":
            continue
        tick = row["first_transition_tick"]
        if tick is not None:
            elapsed = tick-row["cutoff"]
            if not 0 < elapsed <= 50:
                raise ValueError("transition outside supervision window")
            event_bin = next(i for i, h in enumerate(HORIZONS) if elapsed <= h)
            transitions[identity][:event_bin, 1] += 1
            transitions[identity][event_bin, 0] += 1
        else:
            elapsed = row["transition_censored_at"]-row["cutoff"]
            for i, h in enumerate(HORIZONS):
                if elapsed >= h:
                    transitions[identity][i, 1] += 1
    if not fitting:
        raise ValueError("no non-dev TRAIN observations for baseline")
    output = {}
    for key, row in cohort.items():
        identity = row["predicate_id"], row["state_at_cutoff"]
        event, survived = transitions[identity].T
        hazards = (event+.5)/(event+survived+1.)
        survival = np.concatenate(([1.], np.cumprod(1-hazards)))
        probabilities = np.concatenate((survival[:-1]*hazards, survival[-1:]))
        state = {}
        for h in HORIZONS:
            positive, count = states[(*identity, h)]
            if count == 0:
                raise ValueError(f"baseline lacks TRAIN observations for {identity}, horizon {h}")
            state[str(h)] = (positive+.5)/(count+1.)
        output[key] = {"p_future_true": state, "transition_hazard": probabilities[:5].tolist(),
                       "p_transition": float(probabilities[:5].sum()), "runtime_ms_per_query": None}
    return output, fitting


def binary_metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if not len(y):
        return {"n": 0, "positives": 0, "log_loss": None, "brier": None, "auc": None, "ece": None}
    clipped = np.clip(p, 1e-7, 1-1e-7)
    losses = -(y*np.log(clipped)+(1-y)*np.log1p(-clipped))
    buckets = np.minimum((p*10).astype(int), 9)
    calibration = []
    for index in range(10):
        selected = buckets == index
        if selected.any():
            calibration.append({"bin": index, "n": int(selected.sum()),
                                "mean_probability": float(p[selected].mean()), "frequency": float(y[selected].mean())})
    return {"n": len(y), "positives": int(y.sum()), "log_loss": float(losses.mean()),
            "brier": float(np.square(p-y).mean()),
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "ece": sum(b["n"]*abs(b["mean_probability"]-b["frequency"]) for b in calibration)/len(y),
            "calibration_bins": calibration}


def metric_report(cohort, predictions, episodes):
    ep_id = {name: i for i, name in enumerate(episodes)}
    sums = {kind: np.zeros(len(episodes)) for kind in ("state", "transition", *(
            f"{family}/{kind}" for family in ("separation", "corridor", "communication") for kind in ("state", "transition")))}
    counts = {kind: np.zeros(len(episodes), dtype=int) for kind in sums}
    report = {}
    for family in ("all", "separation", "corridor", "communication"):
        keys = [key for key, row in cohort.items() if family == "all" or row["family"] == family]
        if not keys:
            continue
        horizons, pooled_y, pooled_p = {}, [], []
        for h in HORIZONS:
            truth, probability = [], []
            for key in keys:
                label = cohort[key]["future_state"][str(h)]
                if label not in ("true", "false"):
                    continue
                y, p = int(label == "true"), predictions[key]["p_future_true"][str(h)]
                truth.append(y)
                probability.append(p)
                kind = "state" if family == "all" else f"{family}/state"
                safe = min(max(p, 1e-7), 1-1e-7)
                sums[kind][ep_id[key[0]]] += -(y*math.log(safe)+(1-y)*math.log1p(-safe))
                counts[kind][ep_id[key[0]]] += 1
            horizons[str(h)] = binary_metrics(truth, probability)
            pooled_y.extend(truth)
            pooled_p.extend(probability)
        nll, event_nll, survival_nll, det_y, det_p, timing, joint, excess = [], [], [], [], [], [], [], []
        partial, unknown_cutoff, uninformative, before_first = 0, 0, 0, 0
        for key in keys:
            row, pred = cohort[key], predictions[key]
            if row["state_at_cutoff"] == "unknown":
                unknown_cutoff += 1
                continue
            bins = np.array(pred["transition_hazard"] + [1-pred["p_transition"]])
            cumulative = bins[:5].cumsum()
            for h, cdf in zip(HORIZONS, cumulative):
                p = pred["p_future_true"][str(h)]
                change = p if row["state_at_cutoff"] == "false" else 1-p
                excess.append(max(0., change-cdf))
            tick, elapsed = row["first_transition_tick"], row["transition_censored_at"]-row["cutoff"]
            if tick is not None:
                duration = tick-row["cutoff"]
                if not 0 < duration <= 50:
                    raise ValueError("transition outside declared future window")
                index = next(i for i, h in enumerate(HORIZONS) if duration <= h)
                loss = -math.log(max(float(bins[index]), 1e-12))
                event_nll.append(loss)
                timing.append(int(bins[:5].argmax() == index))
                joint.append(int(bins.argmax() == index))
                det_y.append(1)
                det_p.append(pred["p_transition"])
            elif elapsed >= HORIZONS[0]:
                survival = bins[-1]+sum(p for h, p in zip(HORIZONS, bins[:5]) if h > min(elapsed, 50))
                loss = -math.log(max(float(survival), 1e-12))
                survival_nll.append(loss)
                if elapsed >= 50:
                    det_y.append(0)
                    det_p.append(pred["p_transition"])
                else:
                    partial += 1
            else:
                uninformative += 1
                before_first += elapsed > 0
                continue
            nll.append(loss)
            kind = "transition" if family == "all" else f"{family}/transition"
            sums[kind][ep_id[key[0]]] += loss
            counts[kind][ep_id[key[0]]] += 1
        mean = lambda values: float(np.mean(values)) if values else None
        report[family] = {"queries": len(keys), "horizons": horizons, "state_pooled": binary_metrics(pooled_y, pooled_p),
                          "mean_horizon_log_loss": mean([v["log_loss"] for v in horizons.values() if v["n"]]),
                          "transition": {"n": len(nll), "nll": mean(nll), "events": len(event_nll),
                              "observed_nll": mean(event_nll), "survival_nll": mean(survival_nll),
                              "partially_censored": partial, "unknown_cutoff_excluded": unknown_cutoff,
                              "no_completed_bin_excluded": uninformative,
                              "positive_censor_before_first_bin_excluded": before_first,
                              "conditional_timing_accuracy": mean(timing), "joint_timing_accuracy": mean(joint),
                              "detection": binary_metrics(det_y, det_p)},
                          "sequence_consistency": {"n": len(excess), "mean_excess": mean(excess),
                              "violation_rate": mean([x > 1e-3 for x in excess])}}
    runtime = [row["runtime_ms_per_query"] for row in predictions.values() if row["runtime_ms_per_query"] is not None]
    report["runtime_ms_per_query"] = mean(runtime)
    return report, sums, counts


def paired_report(sums, counts, name, reference, bootstrap):
    paired = {}
    for kind in sums[name]:
        if not np.array_equal(counts[name][kind], counts[reference][kind]):
            raise ValueError("paired loss counts differ")
        denominator = counts[name][kind][bootstrap].sum(1)
        informative = denominator > 0
        if not informative.any():
            continue
        delta = sums[name][kind]-sums[reference][kind]
        values = delta[bootstrap][informative].sum(1)/denominator[informative]
        paired[kind] = {"mean_difference": float(delta.sum()/counts[name][kind].sum()),
                        "ci95": np.quantile(values, [.025, .975]).tolist(),
                        "informative_bootstrap_draws": int(informative.sum()),
                        "zero_label_draws": int((~informative).sum())}
    return paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--universe", type=Path, default=WORKSPACE / "supervision/VALID.jsonl")
    parser.add_argument("--train-supervision", type=Path, default=WORKSPACE / "supervision/TRAIN.jsonl")
    parser.add_argument("--run", action="append", required=True, help="name=prediction_file")
    parser.add_argument("--mc", action="append", default=[], help="name=MC_sample_count; explicit smoothing")
    parser.add_argument("--reference", default="persistence_law")
    parser.add_argument("--pair", action="append", default=[], help="additional paired contrast name:reference")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort, excluded = read_unique(args.supervision, supervision=True)
    universe, _ = read_unique(args.universe, supervision=True)
    if not set(cohort) <= set(universe):
        raise ValueError("declared cohort is not a subset of VALID")
    for key, row in cohort.items():
        if row != universe[key] or row["split"] != "VALID" or row["dev"]:
            raise ValueError("cohort supervision differs from authentic VALID")
    train, _ = read_unique(args.train_supervision, supervision=True)
    baseline, fitting = fitted_law(train, cohort)
    runs = {"persistence_law": baseline}
    sources, smoothing = {}, {}
    for spec in args.mc:
        name, count = spec.split("=", 1)
        if name in smoothing or int(count) < 1:
            raise ValueError("duplicate/invalid MC smoothing specification")
        smoothing[name] = int(count)
    for spec in args.run:
        name, filename = spec.split("=", 1)
        if name in runs:
            raise ValueError(f"duplicate run name {name}")
        path = Path(filename)
        loaded, _ = read_unique(path)
        missing, outside = set(cohort)-set(loaded), set(loaded)-set(universe)
        if missing or outside:
            raise ValueError(f"{name}: missing {len(missing)} expected queries, {len(outside)} non-VALID queries")
        for row in loaded.values():
            validate_prediction(row)
        runs[name] = {key: smooth_mc(loaded[key], smoothing[name]) if name in smoothing else loaded[key] for key in cohort}
        sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "supplied_queries": len(loaded), "cohort_queries": len(cohort),
                         "extra_valid_queries": len(loaded)-len(cohort), "mc_samples": smoothing.get(name)}
    if set(smoothing)-set(runs) or args.reference not in runs or args.bootstrap < 1:
        raise ValueError("invalid smoothing name/reference/bootstrap count")
    pairs = [tuple(spec.split(":", 1)) for spec in args.pair]
    if len(set(pairs)) != len(pairs) or any(len(pair) != 2 or any(name not in runs for name in pair) for pair in pairs):
        raise ValueError("duplicate/invalid additional paired contrast")
    episodes = sorted({key[0] for key in cohort})
    bootstrap = np.random.default_rng(args.seed).integers(0, len(episodes), (args.bootstrap, len(episodes)))
    reports, sums, counts = {}, {}, {}
    for name, predictions in runs.items():
        reports[name], sums[name], counts[name] = metric_report(cohort, predictions, episodes)
    for name in runs:
        paired = paired_report(sums, counts, name, args.reference, bootstrap)
        reports[name]["paired_difference"] = paired
        print(json.dumps({"run": name, "queries": len(cohort),
                          "state_log_loss": reports[name]["all"]["state_pooled"]["log_loss"],
                          "transition_nll": reports[name]["all"]["transition"]["nll"],
                          "paired": paired}, ensure_ascii=False), flush=True)
    contrasts = [{"run": name, "reference": reference,
                  "losses": paired_report(sums, counts, name, reference, bootstrap)} for name, reference in pairs]
    write_json(args.output, {"cohort": {"path": str(args.supervision), "sha256": hashlib.sha256(args.supervision.read_bytes()).hexdigest(),
                   "queries": len(cohort), "windows": len({key[:2] for key in cohort}), "episodes": episodes,
                   "excluded_rows": excluded, "baseline_fitting_queries": fitting},
                   "probability_policy": "validate raw values before log-only clipping; no mass repair",
                   "transition_policy": "unknown cutoff and non-events before first bin boundary excluded; partial censoring uses survival beyond completed bins; fitted baseline uses binomial hazards with completed-bin survivals",
                   "bootstrap": {"unit": "episode", "samples": args.bootstrap, "seed": args.seed,
                                 "reference": args.reference, "denominator": "observed labels in each resampled episode multiset"},
                   "sources": sources, "runs": reports, "paired_contrasts": contrasts})


if __name__ == "__main__":
    main()
