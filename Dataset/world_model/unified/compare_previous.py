"""Compare previous VALID predictors with unified dev predictions on matched outcomes (§10)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp

from Dataset.world_model.unified import assemble as A, codes as K, data as D, schema as S, windows as W

EPISODES = ("L2-3_v2__seed00", "L4-3_v1__seed00", "L6-5_v1__seed00")
OUT = S.WORKSPACE/"comparison"
NUMERIC = ("position", "latency_ms")
METRICS = ("nll", "brier", "coarse_rps", "artifact_mass_nll")


class Audit:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.examples = defaultdict(list)

    def add(self, episode, kind, **example):
        self.counts[episode][kind] += 1
        if example and len(self.examples[kind]) < 10:
            self.examples[kind].append({"episode": episode, **example})

    def report(self):
        total = Counter()
        for counts in self.counts.values():
            total.update(counts)
        return {"total": dict(total), "by_episode": dict(self.counts), "examples": dict(self.examples)}


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid probability: {value!r}")
    return float(value)


def hazard_classes(hazards):
    """Conditional hazards -> five first-change bins and survival beyond tick 50."""
    if len(hazards) != len(S.FLIP_BINS):
        raise ValueError("expected five transition hazards")
    h = np.array([probability(v) for v in hazards])
    survival = np.concatenate(([1.], np.cumprod(1-h)))
    return np.concatenate((h*survival[:-1], survival[-1:]))


def hazard_log_classes(hazards):
    h = np.array([probability(v) for v in hazards])
    if len(h) != len(S.FLIP_BINS):
        raise ValueError("expected five transition hazards")
    with np.errstate(divide="ignore"):
        log_survival = np.concatenate(([0.], np.cumsum(np.log1p(-h))))
        return np.concatenate((np.log(h)+log_survival[:-1], log_survival[-1:]))


def validate_law(row):
    p = row["parameters"]
    if row["law"] != "anchor_inflated_normal" or not math.isfinite(p["anchor"]) or not math.isfinite(row["truth"]):
        raise ValueError(f"invalid compact law: {row}")
    probability(p["anchor_probability"])
    components = p["components"]
    if not components or any(len(c) != 3 or not all(math.isfinite(v) for v in c)
                             or c[0] < 0 or c[2] <= 0 for c in components):
        raise ValueError(f"invalid Gaussian components: {components}")
    if not math.isclose(sum(c[0] for c in components), 1., abs_tol=1e-6):
        raise ValueError("compact component weights do not sum to one")


def load_previous():
    """Read only the three named dev/VALID episodes, preferring original artifacts."""
    dev = set(json.loads(D.SPLITS.read_text())["dev"])
    if not set(EPISODES) <= dev:
        raise ValueError(f"named VALID episodes absent from dev: {sorted(set(EPISODES)-dev)}")
    runs = S.ROOT/"Dataset/world_model/runs"
    specs = {f"pose_f{i}": (f"compact/runs/f{i}_mix3_aug_micro/predictions_valid_windows.jsonl",
                             f"baselines/compact/pose_f{i}/predictions_valid_windows.jsonl") for i in range(1, 6)}
    specs.update(link=("compact/runs/link_final/latency_valid_windows.jsonl",
                       "baselines/compact/link_final/latency_valid_windows.jsonl"),
                 gru=("predicate_stage/runs/task_e_gru/predictions_VALID.jsonl",
                      "baselines/predicate/gru_rule_features/predictions_VALID.jsonl"),
                 c2=("predicate_stage/runs/task_c2_valid/predictions_VALID.jsonl",
                     "baselines/predicate/sampling_rules_c2/predictions_VALID.jsonl"))
    previous, files = {}, {}
    origins = set(D.ORIGINS)
    for name, candidates in specs.items():
        paths = [runs/p for p in candidates]
        path = next((p for p in paths if p.is_file()), None)
        info = files[name] = {"available": path is not None, "searched": [str(p) for p in paths]}
        if path is None:
            previous[name] = {}
            continue
        info.update(path=str(path), fallback=path != paths[0])
        numeric = name.startswith("pose_") or name == "link"
        by_window = defaultdict(dict)
        counts = Counter()
        for line in path.open():
            row = json.loads(line)
            if row["episode"] not in EPISODES:
                continue
            cutoff = row["window_cutoff"] if numeric else row["cutoff"]
            if cutoff not in origins:
                counts["outside_unified_origins"] += 1
                continue
            if numeric:
                expected = "communication.latency_ms" if name == "link" else "pose.position_enu_m"
                if row["field"] != expected:
                    counts["other_numeric_fields"] += 1
                    continue
                if row["horizon_ticks"] not in S.HORIZONS:
                    counts["other_horizons"] += 1
                    continue
                if row["component"] not in (range(1) if name == "link" else range(3)):
                    raise ValueError(f"unexpected component in {path}")
                validate_law(row)
                key = row["entity"], row["horizon_ticks"], row["component"]
            else:
                if set(row["p_future_true"]) != set(map(str, S.HORIZONS)):
                    raise ValueError(f"unexpected predicate horizons in {path}")
                for value in row["p_future_true"].values():
                    probability(value)
                classes = hazard_classes(row["transition_hazard"])
                p_transition = probability(row["p_transition"])
                counts["stored_bins_equal_p_transition"] += math.isclose(
                    sum(row["transition_hazard"]), p_transition, abs_tol=1e-7)
                counts["hazard_conversion_disagrees_with_p_transition"] += not math.isclose(
                    1-classes[-1], p_transition, abs_tol=1e-7)
                key = row["predicate_id"], row["tuple_id"]
            window = row["episode"], cutoff
            if key in by_window[window]:
                raise ValueError(f"duplicate prediction in {path}: {window}, {key}")
            by_window[window][key] = row
            counts["matched_rows"] += 1
        previous[name] = dict(by_window)
        info.update(counts=dict(counts), windows=len(by_window))
    files["pose_ensemble"] = {"available": all(files[f"pose_f{i}"]["available"] for i in range(1, 6)),
                              "members": 5, "weights": "equal; never substitute a partial ensemble"}
    return previous, files


def common_windows(previous):
    return sorted({window for predictions in previous.values() for window in predictions})


def relation_aliases(src, origin):
    fticks = W.formal_ticks(origin)
    rels = [r for r in src.relations.values() if any(t in r.samples for t in fticks)]
    return {r.tuple_id: f"r{i+1}" for i, r in enumerate(sorted(rels, key=lambda r: (r.predicate, r.tuple_id)))}


def matched_queries(src, window, previous, audit):
    """Explicit query indices and the previous predictors attached to each query."""
    episode, origin = window["episode"], window["origin"]
    key = episode, origin
    aliases = window["aliases"]
    rel_alias = relation_aliases(src, origin)
    lookup = {}
    for i, q in enumerate(window["queries"]):
        if q["group"] in NUMERIC:
            lookup[q["group"], q["entity"], q["horizon"]] = i
        elif q["group"] in ("predicate", "first_change"):
            lookup[q["group"], q["tuple"], q.get("horizon")] = i
    matches = defaultdict(list)
    members = [previous[f"pose_f{i}"].get(key, {}) for i in range(1, 6)]
    targets = sorted({(e, h) for member in members for e, h, _ in member})
    for entity, horizon in targets:
        if not all((entity, horizon, c) in member for member in members for c in range(3)):
            audit.add(episode, "incomplete_pose_ensemble_target", origin=origin, entity=entity, horizon=horizon)
            continue
        i = lookup.get(("position", aliases.get(entity), horizon))
        if i is None:
            category = src.category.get(entity)
            reason = "outside_unified_moving_categories" if category not in S.MOVING else "absent_origin_position"
            audit.add(episode, "position_query_unavailable", origin=origin, entity=entity, horizon=horizon,
                      category=category, reason=reason)
            continue
        rows = [[member[entity, horizon, c] for member in members] for c in range(3)]
        matches[i].append(("compact_pose_ensemble", entity, rows))
    for (entity, horizon, _), row in previous["link"].get(key, {}).items():
        i = lookup.get(("latency_ms", aliases.get(entity), horizon))
        if i is None:
            audit.add(episode, "latency_query_unavailable", origin=origin, entity=entity, horizon=horizon,
                      reason="absent_entity_or_unknown_origin_latency")
            continue
        matches[i].append(("compact_link", entity, [[row]]))
    for name in ("gru", "c2"):
        for (predicate, tid), row in previous[name].get(key, {}).items():
            rel = src.relations.get(tid)
            if rel is None or rel.predicate != predicate:
                audit.add(episode, f"{name}_tuple_mismatch", origin=origin, tuple_id=tid, predicate=predicate)
                continue
            alias = rel_alias.get(tid)
            indices = [lookup.get(("predicate", alias, h)) for h in S.HORIZONS]
            indices.append(lookup.get(("first_change", alias, None)))
            if any(i is None for i in indices):
                reason = "masked_origin" if (tid, origin) in src.masked_labels else "unknown_or_absent_origin"
                audit.add(episode, f"{name}_{reason}", origin=origin, tuple_id=tid)
                continue
            for i in indices:
                matches[i].append((name, tid, row))
    qids = sorted(matches)
    for i in qids:
        audit.add(episode, "queries_"+window["queries"][i]["group"])
    audit.add(episode, "common_windows")
    if not qids:
        audit.add(episode, "windows_without_matched_queries", origin=origin)
    return qids, dict(matches)


def normal_log_interval(lo, hi, mean, sd):
    """Log Gaussian interval mass without upper-tail CDF subtraction cancellation."""
    a, b = (lo-mean)/sd, (hi-mean)/sd
    upper, lower = (log_ndtr(-a), log_ndtr(-b)) if a >= 0 else (log_ndtr(b), log_ndtr(a))
    if lower == -math.inf:
        return float(upper)
    if upper == lower:
        return -math.inf
    return float(upper + math.log(-math.expm1(float(lower-upper))))


def law_log_interval(row, lo, hi):
    """Mass of (lo, hi], including the anchor atom and all off-atom normals."""
    if not lo < hi:
        raise ValueError(f"empty interval: {lo}, {hi}")
    params = row["parameters"]
    atom = params["anchor_probability"]
    terms = [math.log(atom)] if atom > 0 and lo < params["anchor"] <= hi else []
    if atom < 1:
        weights = sum(w for w, _, _ in params["components"])
        for w, mean, sd in params["components"]:
            if w > 0:
                terms.append(math.log1p(-atom)+math.log(w/weights)+normal_log_interval(lo, hi, mean, sd))
    return float(logsumexp(terms)) if terms else -math.inf


def mixture_log_interval(rows, lo, hi):
    return float(logsumexp([law_log_interval(row, lo, hi) for row in rows])-math.log(len(rows)))


def coarse_intervals(ic):
    bounds = {}
    for code, (lo, hi) in zip(ic.codes, ic.real_bounds()):
        if code[0] in bounds:
            a, b = bounds[code[0]]
            bounds[code[0]] = min(a, lo), max(b, hi)
        else:
            bounds[code[0]] = lo, hi
    return sorted(bounds.items(), key=lambda entry: entry[1][0])


def previous_numeric(q, rows, x0, answers):
    ic = answers.interval(q["group"], q["category"], q["horizon"], q["eps"])
    bounds = ic.real_bounds()
    indices = [ic.index(r) for r in q["residual"]]
    if any(i is None for i in indices):
        raise ValueError("numeric label outside declared support")
    logps = [mixture_log_interval(members, x+a, x+b)
             for members, x, (a, b) in zip(rows, x0, [bounds[i] for i in indices])]
    coarse = coarse_intervals(ic)
    probs = []
    for j, (_, (a, b)) in enumerate(coarse):
        lo = -math.inf if j == 0 else x0[0]+a
        hi = math.inf if j == len(coarse)-1 else x0[0]+b
        probs.append(math.exp(mixture_log_interval(rows[0], lo, hi)))
    target = next(j for j, (symbol, _) in enumerate(coarse) if symbol == ic.codes[indices[0]][0])
    tails = [math.exp(mixture_log_interval(members, -math.inf, x+min(a for a, _ in bounds)))
             + math.exp(mixture_log_interval(members, x+max(b for _, b in bounds), math.inf))
             for members, x in zip(rows, x0)]
    return {"nll": -sum(logps), "coarse_rps": K.ranked_probability_score(probs, target),
            "component_log_probability": logps, "outside_support_probability": tails}


def previous_predicate(q, row):
    if q["group"] == "predicate":
        p = probability(row["p_future_true"][str(q["horizon"])])
        y = q["label"]
        nll = (-math.log(p) if p else math.inf) if y else (-math.log1p(-p) if p < 1 else math.inf)
        return {"nll": nll, "brier": 2*(p-y)**2, "p_true": p}
    compatible = [q["label"]] if q["label"] is not None else q["compatible"]
    logp = float(logsumexp(hazard_log_classes(row["transition_hazard"])[compatible]))
    result = {"nll": -logp, "class_probability": hazard_classes(row["transition_hazard"]).tolist()}
    if math.isclose(sum(row["transition_hazard"]), row["p_transition"], abs_tol=1e-7):
        masses = np.array([*row["transition_hazard"], 1-row["p_transition"]])
        mass = float(masses[compatible].sum())
        result.update(artifact_mass_nll=-math.log(mass) if mass else math.inf,
                      artifact_class_probability=masses.tolist())
    return result


def previous_scores(src, window, qids, matches, answers, audit):
    """Score the same unified labels, auditing old truths before any model inference."""
    episode, origin = window["episode"], window["origin"]
    scores = {}
    for i in qids:
        q = window["queries"][i]
        _, target, compatible = answers.query_domain(q)
        if target is None and compatible is None:
            audit.add(episode, "undefined_"+q["group"])
            continue
        if target == "out_of_support":
            audit.add(episode, "out_of_support_"+q["group"], origin=origin, query=i, residual=q["residual"])
            continue
        audit.add(episode, "scored_"+q["group"])
        if q["group"] == "first_change" and q["label"] is None:
            audit.add(episode, "censored_first_change")
        for name, identity, prediction in matches[i]:
            if q["group"] in NUMERIC:
                if q["group"] == "position":
                    raw = src.position_raw[identity]
                    x0, truth = raw[origin], raw[origin+q["horizon"]]
                else:
                    source, field, _ = S.NUMERIC_TARGETS[q["group"]]
                    raw = src.target_raw[identity, source, field]
                    x0, truth = [raw[origin]], [raw[origin+q["horizon"]]]
                for c, members in enumerate(prediction):
                    for row in members:
                        audit.add(episode, "truth_checks_"+q["group"])
                        old_r = int(round((row["truth"]-x0[c])/q["eps"]))
                        if abs(row["truth"]-truth[c]) > q["eps"]/2 or old_r != q["residual"][c]:
                            audit.add(episode, "truth_mismatch_"+q["group"], origin=origin, entity=identity,
                                      horizon=q["horizon"], component=c, model=row["model"], old_truth=row["truth"],
                                      unified_truth=truth[c], old_residual=old_r, unified_residual=q["residual"][c])
                score = previous_numeric(q, prediction, x0, answers)
            else:
                score = previous_predicate(q, prediction)
            if not math.isfinite(score["nll"]):
                audit.add(episode, f"{name}_zero_probability_"+q["group"])
            scores[i, name] = score
    return scores


def paired_summary(rows, draws=2000, seed=0):
    table = defaultdict(list)
    for row in rows:
        base = row["previous_model"], row["group"]
        keys = {(*base, "all", "all"), (*base, "all", row["horizon"]),
                (*base, row["category"], row["horizon"])}
        for key in keys:
            table[key].append(row)
    result = []
    for (model, group, category, horizon), records in sorted(table.items(), key=lambda entry: str(entry[0])):
        entry = {"previous_model": model, "group": group, "category": category, "horizon": horizon,
                 "n": len(records), "episodes": sorted({r["episode"] for r in records})}
        metrics = {}
        for metric in METRICS:
            pairs = [r for r in records if metric in r["previous"] and metric in r["unified"]]
            if not pairs:
                continue
            unified = np.array([r["unified"][metric] for r in pairs])
            previous = np.array([r["previous"][metric] for r in pairs])
            diffs = unified-previous
            if np.isnan(diffs).any():
                raise ValueError("undefined paired difference (both scores infinite)")
            episodes = sorted({r["episode"] for r in pairs})
            clusters = [diffs[[r["episode"] == ep for r in pairs]] for ep in episodes]
            sums = np.array([v.sum() for v in clusters])
            sizes = np.array([len(v) for v in clusters])
            pick = np.random.default_rng(seed).integers(len(episodes), size=(draws, len(episodes)))
            means = sums[pick].sum(1)/sizes[pick].sum(1)
            ci = np.quantile(means, [.025, .975], method="inverted_cdf").tolist()
            metrics[metric] = {"n": len(pairs), "unified": float(unified.mean()), "previous": float(previous.mean()),
                               "difference": float(diffs.mean()), "difference_ci95": ci,
                               "previous_infinite_n": int(np.isinf(previous).sum())}
        entry["metrics"] = metrics
        result.append(entry)
    return result


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("NaN cannot be reported")
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--budget", type=int, default=A.TOTAL_BUDGET, help="total budget")
    parser.add_argument("--policy", choices=A.POLICIES, default="relevance")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.budget < 0 or (args.limit is not None and args.limit < 1):
        parser.error("budget must be nonnegative and limit must be positive")
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be a single directory name")
    previous, files = load_previous()
    windows = common_windows(previous)
    if not windows:
        raise ValueError(f"no common dev/VALID windows; availability: {files}")
    selected = windows[:args.limit] if args.limit is not None else windows
    print(json.dumps({"common_windows": len(windows), "selected_windows": len(selected),
                      "unavailable": [k for k, v in files.items() if not v["available"]]}), flush=True)
    from Dataset.world_model.unified.evaluate import score_item
    from Dataset.world_model.unified.predictor import Predictor
    from Dataset.world_model.unified.train_predictor import load_trainable

    model = Predictor(device="cuda", query_chunk=8)
    load_trainable(model, args.checkpoint)
    model.eval()
    enc, audit, rows = D.Encoder(), Audit(), []
    out = OUT/args.name
    out.mkdir(parents=True, exist_ok=True)
    with (out/"rows.jsonl").open("w") as fh:
        for episode, origin in selected:
            prepared = enc.window(episode, origin)
            window, pw = prepared
            src = D.cached_source(episode)
            qids, matches = matched_queries(src, window, previous, audit)
            scores = previous_scores(src, window, qids, matches, enc.answers, audit)
            if not scores:
                continue
            enc.set_relevance(window, pw, qids)
            item = enc.item(episode, origin, query_ids=qids, budget=args.budget, policy=args.policy, prepared=prepared)
            used = item["counts"]["optional_used"]
            expected = sorted({i for i, _ in scores})
            if item["query_index"] != expected:
                raise ValueError(f"encoded queries disagree with matched labels: {episode}@{origin}")
            res = score_item(model, item)
            if len(res["rows"]) != len(expected):
                raise ValueError("unified scores disagree with query indices")
            for i, unified in zip(item["query_index"], res["rows"]):
                q = window["queries"][i]
                for name, identity, _ in matches[i]:
                    old = scores[i, name]
                    uscore = {k: unified[k] for k in METRICS if k in unified}
                    if "artifact_mass_nll" in old:
                        uscore["artifact_mass_nll"] = unified["nll"]
                    if any(not math.isfinite(v) for v in uscore.values()):
                        raise ValueError(f"non-finite unified score: {episode}@{origin}, query {i}")
                    row = {"episode": episode, "cutoff": origin, "query_index": i, "query": q,
                           "group": q["group"], "category": q.get("category", q.get("predicate")),
                           "horizon": q.get("horizon", 50), "target_id": identity, "previous_model": name,
                           "budget": args.budget, "optional_tokens": used, "counts": item["counts"],
                           "unified": uscore, "previous": old,
                           "difference": {k: uscore[k]-old[k] for k in METRICS if k in uscore and k in old}}
                    rows.append(row)
                    fh.write(json.dumps(json_safe(row), allow_nan=False)+"\n")
            fh.flush()
            print(json.dumps({"episode": episode, "cutoff": origin, "queries": len(expected)}), flush=True)
    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "budget": args.budget, "policy": args.policy,
              "episodes": list(EPISODES), "common_windows": len(windows), "selected_windows": len(selected),
              "paired_rows": len(rows), "files": files, "audit": audit.report(), "scores": paired_summary(rows),
              "methodology": {
                  "means": "query means; differences paired on episode/cutoff/query and bootstrapped by episode",
                  "bootstrap": {"draws": 2000, "seed": 0, "interval": "95% percentile; inverted empirical CDF"},
                  "numeric_nll": "bounded quantum-centred (lo,hi] residual intervals shifted by raw origin x0; "
                                 "equal member mixture per component, product across components; no support conditioning",
                  "compact_weights": "component weights normalized only for serialization roundoff (sum tolerance 1e-6)",
                  "coarse_rps": "component 0; normalized by coarse thresholds as in score_item; outside-support "
                                "tails included in extreme coarse classes for both predictors; fine NLL stays bounded",
                  "brier": "sum of squared errors over false/true, exactly the scale in evaluate.score_item",
                  "first_change": "task-specified conditional hazard conversion, with unified censoring compatibility sets",
                  "artifact_mass_nll": "secondary first-change comparison: old producers actually stored unconditional "
                                       "bin masses under transition_hazard (train_predicate.py and rule_predicates.py); "
                                       "use those masses plus 1-p_transition to preserve the previous predictor",
                  "zeros": "no probability floor or smoothing; true zero likelihood gives Infinity/-Infinity strings",
                  "predicate_truth": "old prediction rows contain no truth; known/masked labels come from unified sources"}}
    (out/"report.json").write_text(json.dumps(json_safe(report), indent=1, allow_nan=False)+"\n")
    print(json.dumps(json_safe({"paired_rows": len(rows), "audit": audit.report(), "output": str(out)})), flush=True)


if __name__ == "__main__":
    main()
