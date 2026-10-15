"""Deterministic selector evaluation.

The selector and the recency/relevance anchors are scored on the same jobs, queries
and frozen predictor, so their differences are paired.
"""
from __future__ import annotations

import random
import time

import torch

from Dataset.world_model.unified import assemble as A
from Dataset.world_model.unified.train_predictor import quick_eval
from Dataset.world_model.unified.supervise import scoring_window

def selector_keep(selector, enc, sc):
    kept, info = selector.select(sc.pw, enc.tok, sc.qids, sc.budget, sc.actions_ids)
    return kept, info["positions"]


@torch.no_grad()
def score_jobs(selector, predictor, enc, jobs, policies=("selector",), sink=None):
    """One record per (job, policy); the policies of a job share one SetScorer (queries, labels, budget)."""
    if selector is not None:
        selector.eval()
    records = []
    for job in jobs:
        sc = scoring_window(predictor, enc, job)
        for policy in policies:
            t = time.perf_counter()
            if policy == "selector":
                kept, read_positions = selector_keep(selector, enc, sc)
            else:
                kept, read_positions = A.select(sc.pw, sc.qids, sc.budget, policy, random.Random(job["id"]))[0], 0
            selection_seconds = time.perf_counter() - t
            calls = sc.calls
            result = sc(kept)
            record = _record(job, policy, kept, read_positions, selection_seconds, sc.calls - calls, result, sc)
            records.append(record)
            if sink is not None:
                sink(record)
    return records


def _record(job, policy, kept, read_positions, selection_seconds, calls, result, sc):
    return {"job_id": job["id"], "policy": policy, "episode": job["episode"], "origin": job["origin"],
            "arm_key": job["arm_key"], "which": job["which"], "kept": kept,
            "query_ids": list(sc.qids), "inclusion": sc.pw.inclusion, "optional_budget": sc.budget,
            "selector_positions": read_positions, "selection_seconds": selection_seconds,
            "predictor_calls": calls, **result}


def _row(r, other=None):
    """quick_eval row; with other, the per-window NLL totals are the paired difference r - other."""
    sums = {g: (v * r["query_counts"][g], r["query_counts"][g]) for g, v in r["groups"].items()}
    if other is not None:
        sums = {g: (t - other["groups"][g] * n, n) for g, (t, n) in sums.items()}
    return (r["episode"], r["origin"], r["arm_key"]), sums, r["counts"]["N_captures"] > 0


def _score(rows):
    parent = [x for x in rows if x[0][2] is None]
    arm = [x for x in rows if x[0][2] is not None]
    pn, _, ps, p, _ = quick_eval(parent)
    if not arm:
        return p, p, None, pn, {}, ps, {}
    an, _, ase, a, _ = quick_eval(arm)
    return (5 * p + a) / 6, p, a, pn, an, ps, ase


def summarize(records, draws=2000, seed=0):
    """Per-policy scores and paired differences (selector minus each anchor, relevance minus recency).

    Queries and label counts are identical across the policies of a job, so the score is linear in
    the per-window NLL totals. A window-cluster bootstrap (ARM members share a cluster) gives paired
    95% intervals; negative differences favour the first policy.
    """
    by_policy = {}
    for r in records:
        by_policy.setdefault(r["policy"], {})[r["job_id"]] = r
    metrics = {}
    for policy, rs in by_policy.items():
        score, p, a, pn, an, ps, ase = _score([_row(r) for r in rs.values()])
        metrics[policy] = {"score": score, "parent_score": p, "arm_score": a, "parent_group_nll": pn,
                           "arm_group_nll": an, "parent_se": ps, "arm_se": ase, "windows": len(rs),
                           "predictor_calls": sum(r["predictor_calls"] for r in rs.values()),
                           "predictor_positions": sum(r["counts"]["N_total"] for r in rs.values()),
                           "selector_positions": sum(r["selector_positions"] for r in rs.values()),
                           "windows_with_captures": sum(r["counts"]["N_captures"] > 0 for r in rs.values()),
                           "selection_seconds": sum(r["selection_seconds"] for r in rs.values())}
    paired = {}
    for a, b in (("selector", "recency"), ("selector", "relevance"), ("relevance", "recency")):
        if a not in by_policy or b not in by_policy:
            continue
        if set(by_policy[a]) != set(by_policy[b]):
            raise ValueError(f"paired comparison {a}-{b} needs the same jobs")
        clusters = {}
        for i in sorted(by_policy[a]):
            row = _row(by_policy[a][i], other=by_policy[b][i])
            clusters.setdefault(row[0][:2], []).append(row)
        keys = sorted(clusters)
        rng = random.Random(seed)
        # A cluster drawn twice merges into one window in quick_eval; its totals and counts add, which
        # leaves the resampled mean exact.
        samples = sorted(_score([x for k in rng.choices(keys, k=len(keys)) for x in clusters[k]])[0]
                         for _ in range(draws))
        score, p, arm, *_ = _score([x for k in keys for x in clusters[k]])
        paired[f"{a}-{b}"] = {"score": score, "parent_score": p, "arm_score": arm,
                              "ci95": [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]],
                              "clusters": len(keys)}
    return {"policies": metrics, "paired": paired}
