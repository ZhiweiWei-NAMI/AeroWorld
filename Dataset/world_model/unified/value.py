"""Measured-value context selection.

collect: for each TRAIN window, score the recency/relevance anchors and M random full contexts with the frozen 2B.
Random contexts are greedy fills under the exact optional budget on a random priority over record metadata plus
Gumbel noise; a quarter fill only 50-100% of the budget. The measured losses are the only training signal for the
0.8B value head (value_head.py); future labels never reach the selector's inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from Dataset.world_model.unified import assemble as A, data as D, windows as W
from Dataset.world_model.unified.supervise import frozen_predictor, make_jobs, scoring_window

KINDS = ("origin", "knot", "frame_change", "comm_change", "compute_change", "rel_change",
         "evidence_change", "event", "scheduled", "leave", "capture", "other")
ROLES = ("position", "latency_ms", "cpu_utilization_pct", "queried_relation", "bound_entity")
META = ("age", "head_rank", "log_cost", "scoped", "capture", "relevance", "head_size", "tick_crowd") + KINDS + ROLES \
    + ("role_queries",)


def head_roles(pw, qids):
    """Label-free query roles of each head: entity queried for position / latency / cpu, queried relation, entity
    bound by a queried relation; and the number of queries addressing the head."""
    roles, count = {}, {}
    for i in qids:
        q = pw.w["queries"][i]
        if "tuple" in q:
            heads = [("rel|" + q["tuple"], "queried_relation")] + \
                [("entity|" + a, "bound_entity") for a in pw.w["rel_entities"].get(q["tuple"], ())]
        elif "entity" in q and q["group"] in ROLES:
            heads = [("entity|" + q["entity"], q["group"])]
        else:
            continue
        for h, role in heads:
            roles.setdefault(h, set()).add(role)
            count[h] = count.get(h, 0) + 1
    return roles, count


def candidate_table(pw, qids):
    """Candidate keys (fixed order) and their label-free metadata [n, len(META)]."""
    units = sorted(pw.candidates(qids), key=lambda u: (u.head, -u.tick, u.key))
    scoped = {"entity|" + a for a in pw.addressed_entities(qids)} | {"rel|" + t for t in pw.queried_tuples(qids)}
    roles, count = head_roles(pw, qids)
    rank, size, crowd = {}, {}, {}
    for u in units:
        size[u.head] = size.get(u.head, 0) + 1
        crowd[u.tick] = crowd.get(u.tick, 0) + 1
    rows = []
    for u in units:
        r = rank.get(u.head, -1) + 1
        rank[u.head] = r
        kind = u.kind if u.kind in KINDS else "other"
        rows.append([(pw.w["origin"] - u.tick) / 50., math.log1p(r) / 5., math.log1p(len(u.ids) + u.extra) / 10.,
                     float(u.head in scoped), float(u.kind == "capture"), math.log1p(max(0., u.relevance)),
                     math.log1p(size[u.head]) / 8., math.log1p(crowd[u.tick]) / 6.]
                    + [float(kind == k) for k in KINDS] + [float(r in roles.get(u.head, ())) for r in ROLES]
                    + [math.log1p(count.get(u.head, 0)) / 4.])
    return [u.key for u in units], np.asarray(rows, dtype=np.float32)


def fill(pw, qids, budget, keys, priority, cost_aware=False, fraction=1.):
    """Greedy fill in descending priority (or priority per position) up to fraction x budget."""
    if cost_aware:
        cost = np.array([len(pw.units[k].ids) + pw.units[k].extra for k in keys], dtype=np.float64)
        priority = priority / cost
    order = np.argsort(-priority, kind="stable")
    return pw.greedy_fill([pw.units[keys[i]] for i in order], int(fraction * budget), qids)[0]


def random_priority(meta, rng):
    """One random member of the exploration family; returns (priority, description)."""
    m = {name: meta[:, i] for i, name in enumerate(META)}
    w = {"age": -rng.uniform(0, 3), "head_rank": -rng.uniform(0, 3), "scoped": rng.uniform(-1, 2),
         "capture": rng.uniform(-3, 3), "relevance": rng.uniform(0, 2), "head_size": rng.uniform(-1, 1)}
    kinds = {k: rng.gauss(0, .5) for k in KINDS if k != "capture"}
    tau = rng.choice((0., .3, 1., 3.))
    g = np.random.default_rng(rng.randrange(2**32)).gumbel(size=len(meta))
    priority = sum(v * m[k] for k, v in w.items()) + sum(v * m[k] for k, v in kinds.items()) + tau * g
    return priority, {"w": w, "kinds": kinds, "tau": tau}


def random_contexts(pw, qids, budget, keys, meta, rng, n):
    """n label-free complete contexts from the exploration family (a quarter fill only 50-100% of the budget)."""
    out = []
    for m in range(n):
        fraction = 1. if rng.random() < .75 else rng.uniform(.5, 1.)
        priority, info = random_priority(meta, rng)
        info["fraction"] = fraction
        out.append((f"sample{m}", fill(pw, qids, budget, keys, priority, fraction=fraction), info))
    return out


def collect(args):
    jobs = json.loads(Path(args.jobs).read_text())[args.shard::args.shards]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"samples.{args.shard}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["job"]["id"] for l in path.open()}
    sink = path.open("a")
    predictor = frozen_predictor(args.predictor, {"predictor_lora_rank": 16, "predictor_query_chunk": 8})
    enc = D.Encoder()
    selector = None
    if args.backbone:
        from Dataset.world_model.unified.selector import Selector
        selector = Selector.load(args.backbone)
    for job in jobs:
        if job["id"] in done:
            continue
        started = time.perf_counter()
        sc = scoring_window(predictor, enc, job)
        keys, meta = candidate_table(sc.pw, sc.qids)
        if selector is not None:
            from Dataset.world_model.unified.value_head import record_states, save_features
            states = record_states(selector, sc.pw, enc.tok, sc.qids, sc.budget, sc.actions_ids, keys)
            save_features(out / "feat" / f"{job['id']}.pt", sc.pw, sc.qids, sc.budget, keys, meta, states)
            save_features(out / "meta" / f"{job['id']}.pt", sc.pw, sc.qids, sc.budget, keys, meta, None)
        rng = random.Random(W.stable_hash("value-collect", args.seed, job["id"]))
        contexts = [("recency", A.select(sc.pw, sc.qids, sc.budget, "recency")[0], {}),
                    ("relevance", A.select(sc.pw, sc.qids, sc.budget, "relevance")[0], {})]
        contexts += random_contexts(sc.pw, sc.qids, sc.budget, keys, meta, rng, args.samples)
        rows = []
        for name, kept, info in contexts:
            if sc.cost(kept) > sc.budget:
                raise RuntimeError("context exceeds the optional budget")
            r = sc(kept)
            rows.append({"name": name, "kept": sorted(kept), "loss": r["loss"], "groups": r["groups"],
                         "query_counts": r["query_counts"], "counts": r["counts"], "info": info})
        sink.write(json.dumps({"job": job, "query_ids": list(sc.qids), "inclusion": sc.pw.inclusion,
                               "budget": sc.budget, "n_candidates": len(keys), "contexts": rows,
                               "seconds": time.perf_counter() - started}) + "\n")
        sink.flush()
        print(json.dumps({"job": job["id"], "seconds": round(time.perf_counter() - started, 1),
                          "losses": [round(r["loss"], 4) for r in rows[:3]]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--split", default="train"); j.add_argument("--windows", type=int, required=True)
    j.add_argument("--seed", type=int, required=True); j.add_argument("--output", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--jobs", required=True); c.add_argument("--out", required=True)
    c.add_argument("--predictor", required=True)
    c.add_argument("--backbone", help="0.8B checkpoint whose record states are cached during collection")
    c.add_argument("--samples", type=int, default=12); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--shard", type=int, default=0); c.add_argument("--shards", type=int, default=1)
    args = p.parse_args()
    if args.command == "jobs":
        jobs = make_jobs(args.split, args.windows, args.seed)
        Path(args.output).write_text(json.dumps(jobs, indent=1))
    else:
        collect(args)


if __name__ == "__main__":
    main()
