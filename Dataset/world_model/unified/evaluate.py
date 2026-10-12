"""Loss-budget evaluation on the development holdout (RESEARCH_PLAN §10).

For each window (deterministic query set), each total budget and each selection
method over text records and captures (recency, uniform time, change point,
random, geometric relevance, the structured scorer and the learned selector),
the same predictor scores the realized future:

  * per-group marginal NLL (locally normalized prefix tree),
  * Brier score for single-level domains (predicate state, first change, events),
  * ranked probability score on the coarse interval level of numeric targets
    (common intervals, so continuous predictors can be compared on the same outcome space),
  * joint NLL per window (time-major autoregressive mode) with --joint,
  * input positions (background, fixed, text, frame headers, sensor tokens, queries, total), the
    numbers of captures kept and available, and wall time.

Aggregates are means over windows with paired, parent-clustered bootstrap
intervals of each method's difference to recency.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
import time

import numpy as np
import torch

from Dataset.world_model.unified import assemble as A, codes as K, data as D, schema as S
from Dataset.world_model.unified.predictor import Predictor
from Dataset.world_model.unified.train_predictor import load_trainable

OUT = S.WORKSPACE/"evaluation"
SINGLE = ("predicate", "first_change", "event", "global_event", "scheduled")


def dev_windows(per_episode=4, episodes=None):
    eps = episodes or D.splits()["dev"]
    origins = D.ORIGINS[::max(1, len(D.ORIGINS) // per_episode)][:per_episode]
    return [(e, o) for e in eps for o in origins]


@torch.no_grad()
def score_item(model, item, joint=False):
    loader = D.visual_loader(item)
    encs = item["encs"]
    ctx = model.build_context(item["segments"], loader)
    t = time.time()
    lps = model.marginal(ctx, [(e.ids, e.steps) for e in encs])
    rows = []
    for e, lp in zip(encs, lps):
        nll = float(K.query_loss(lp, e))
        r = {"group": e.group, "nll": nll}
        if e.targets is not None and e.group in SINGLE and len(lp) == 1:
            p = lp[0].exp().cpu().numpy()
            y = np.zeros_like(p)
            y[e.targets[0]] = 1.
            r["brier"] = float(((p - y) ** 2).sum())
        if e.targets is not None and e.group in ("position", "latency_ms", "cpu_utilization_pct"):
            # Coarse level of the first component: leaves ordered "z", "-0".."-k", "+0".."+k" -> sort by value.
            tree = e.trees[0].tree
            kids = tree.children[()]
            order = sorted(range(len(kids)), key=lambda i: _coarse_rank(kids[i]))
            p = lp[0].exp().cpu().numpy()[order]
            k = order.index(e.targets[0])
            r["coarse_rps"] = K.ranked_probability_score(p, k)
        rows.append(r)
    out = {"rows": rows, "seconds_marginal": time.time() - t}
    if joint and item["joint"]:
        order = item["joint"]
        ctx = model.build_context(item["segments"], loader)
        jl = model.joint(ctx, [(encs[j].full_ids, encs[j].steps) for j in order])
        out["joint_nll"] = float(sum(float(K.query_loss(lp, encs[j])) for lp, j in zip(jl, order)))
        out["joint_queries"] = len(order)
        out["marginal_nll_same_queries"] = float(sum(rows[j]["nll"] for j in order))
    return out


def sample_value(code, ic, rng):
    """A residual value (field units) drawn uniformly inside the sampled interval leaf."""
    lo, hi = ic.real_bounds()[ic.codes.index(code)]
    return rng.uniform(lo, hi)


@torch.no_grad()
def joint_sample_scores(model, item, vocab, k=8, seed=0):
    """Energy score of sampled position futures and the rule-consistency rate of sampled predicate futures."""
    encs, order = item["encs"], item["joint"]
    queries = [(encs[j].full_ids[:len(encs[j].ids) - len(encs[j].steps) + 1], encs[j].trees) for j in order]
    loader = D.visual_loader(item)
    ctx = model.build_context(item["segments"], loader)
    gen = torch.Generator(device=model.device).manual_seed(seed)
    trees = [[t.tree if isinstance(t, K.IntervalCode) else t for t in encs[j].trees] for j in order]
    samples = model.sample_joint(ctx, [(q, tr) for (q, _), tr in zip(queries, trees)], k, vocab, generator=gen)
    rng = np.random.default_rng(seed)
    energy, consistent, checked = [], 0, 0
    qmeta = [item["query_meta"][j] for j in order]
    for pos_j, j in enumerate(order):
        e = encs[j]
        if e.group != "position":
            continue
        ic = e.trees[0]
        draws = np.array([[sample_value(code, ic, rng) for code in s[pos_j]] for s in samples])
        truth = np.array([(r) * ic.eps for r in qmeta[pos_j]["residual"]])
        es = np.mean(np.linalg.norm(draws - truth, axis=1)) - 0.5 * np.mean(
            np.linalg.norm(draws[:, None] - draws[None], axis=2))
        energy.append(float(es))
    # Rule consistency: a sampled first change in (a, b] requires the sampled states at horizons <= a to equal
    # the origin value; ">5 s" requires every sampled horizon state to equal it.
    by_tuple = {}
    for pos_j, j in enumerate(order):
        m = qmeta[pos_j]
        if "tuple" in m:
            by_tuple.setdefault(m["tuple"], []).append((pos_j, m))
    for t, entries in by_tuple.items():
        flip = [p for p, m in entries if m["group"] == "first_change"]
        states = [(p, m) for p, m in entries if m["group"] == "predicate"]
        if not flip or not states:
            continue
        v0 = ("false", "true").index(entries[0][1]["state"])
        for s in samples:
            cls = int(s[flip[0]][0][0])
            a = S.FLIP_BINS[cls][0] if cls < 5 else 50
            ok = all(("false", "true").index(s[p][0][0]) == v0 for p, m in states if m["horizon"] <= a)
            consistent += ok
            checked += 1
    return {"energy_position": float(np.mean(energy)) if energy else None, "energy_n": len(energy),
            "rule_consistency": consistent / checked if checked else None, "rule_checked": checked}


def _coarse_rank(symbol):
    if symbol == "z":
        return 0.
    sign = -1 if symbol[0] == "-" else 1
    return sign * (int(symbol[1:]) + 1)


def summarize(rows):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("nll", "brier", "coarse_rps"):
            if k in r:
                g[r["group"]][k].append(r[k])
    return {grp: {k: float(np.mean(v)) for k, v in d.items()} | {"n": len(d["nll"])} for grp, d in g.items()}


def weighted_nll(summary):
    groups = [g for g in summary if summary[g]["n"]]
    return float(np.mean([summary[g]["nll"] for g in groups])) if groups else None


def bootstrap(diffs_by_episode, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    eps = list(diffs_by_episode)
    means = []
    for _ in range(draws):
        pick = rng.choice(len(eps), len(eps))
        vals = [v for i in pick for v in diffs_by_episode[eps[i]]]
        means.append(np.mean(vals))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--selector", default=None)
    p.add_argument("--scorer", default=None, help="structured scorer state (selector-capacity comparison)")
    p.add_argument("--name", required=True)
    p.add_argument("--per-episode", type=int, default=4)
    p.add_argument("--budgets", type=int, nargs="+", default=[10000, A.TOTAL_BUDGET], help="total budgets")
    p.add_argument("--methods", nargs="*", default=list(A.POLICIES))
    p.add_argument("--visual", choices=("learned", "fixed"), default="learned",
                   help="selector/scorer: select captures too, or keep the fixed capture packet (text-only stage)")
    p.add_argument("--query-chunk", type=int, default=8)
    p.add_argument("--joint", action="store_true")
    p.add_argument("--joint-samples", type=int, default=0, help="K ancestral joint samples (energy score, rule consistency)")
    p.add_argument("--sample-budget", type=int, default=A.TOTAL_BUDGET)
    p.add_argument("--sample-every", type=int, default=4, help="joint samples on every n-th window only")
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--report", action="store_true", help="only aggregate existing rows")
    args = p.parse_args()
    if args.report:
        print(json.dumps(report(args.name), indent=1))
        return
    out = OUT/args.name
    out.mkdir(parents=True, exist_ok=True)
    model = Predictor(device="cuda", query_chunk=args.query_chunk)
    if args.checkpoint != "none":
        load_trainable(model, args.checkpoint)
    model.eval()
    selector = None
    if args.selector:
        from Dataset.world_model.unified.selector import Selector
        selector = Selector()
        state = torch.load(args.selector, map_location="cpu")
        selector.load_state_dict(state, strict=False)
        selector.eval()
    scorer = None
    if args.scorer:
        from Dataset.world_model.unified.scorer import Scorer
        scorer = Scorer()
        scorer.load_state_dict(torch.load(args.scorer, map_location="cpu"))
        scorer.eval()
    enc = D.Encoder()
    methods = args.methods + (["selector"] if selector else []) + (["scorer"] if scorer else [])
    check_signature(out, {"checkpoint": args.checkpoint, "no_images": args.no_images}, "rows")
    path = out/f"rows_{args.shard:02d}.jsonl"
    done = {(r["episode"], r["origin"], r["budget"], r["method"]) for p in out.glob("rows_*.jsonl")
            for r in map(json.loads, p.open())}
    with path.open("a") as fh:
        for i, (episode, origin) in enumerate(dev_windows(args.per_episode)):
            if i % args.shards != args.shard:
                continue
            prepared = enc.window(episode, origin, images=not args.no_images)
            w, pw = prepared
            qids = pw.select_queries(None)
            enc.set_relevance(w, pw, qids)
            features = {}
            for budget in args.budgets:
                try:
                    optional = enc.optional_budget(w, pw, qids, budget)
                except ValueError as exc:          # mandatory inputs alone exceed this budget
                    print(json.dumps({"skipped": str(exc)}), flush=True)
                    continue
                for method in methods:
                    if (episode, origin, budget, method) in done:
                        continue
                    t = time.time()
                    if method in ("selector", "scorer"):
                        check_signature(out, {method: args.selector if method == "selector" else args.scorer,
                                              "visual": args.visual}, method)
                    if method == "selector":
                        kept, _ = selector.select(pw, enc.tok, qids, optional, visual=args.visual)
                    elif method == "scorer":
                        kept, _ = scorer.select(pw, qids, optional, visual=args.visual)
                    else:
                        kept, _ = A.select(pw, qids, optional, method, rng=random.Random(origin))
                    t_sel = time.time() - t
                    item = enc.item(episode, origin, kept=kept, query_ids=qids, budget=budget, images=not args.no_images,
                                    prepared=prepared, features=features)
                    res = score_item(model, item, joint=args.joint)
                    if args.joint_samples and budget == args.sample_budget and method in ("recency", "selector") \
                            and i % args.sample_every == 0:
                        res.update(joint_sample_scores(model, item, enc.vocab, args.joint_samples, seed=origin))
                    rec = {"episode": episode, "origin": origin, "budget": budget, "method": method,
                           "counts": item["counts"], "selection_seconds": t_sel, **{k: v for k, v in res.items() if k != "rows"},
                           "summary": summarize(res["rows"])}
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()


def check_signature(out, signature, part):
    """Resumed rows must come from the same models and modes: rows of an output name are bound to the
    signature of their first run (per part: the predictor rows, the selector, the scorer)."""
    path = out/f"signature_{part}.json"
    try:
        with open(path, "x") as fh:           # first writer wins; parallel shards never see a truncated file
            fh.write(json.dumps(signature))
    except FileExistsError:
        for _ in range(50):
            text = path.read_text()
            if text:
                break
            time.sleep(0.1)
        prior = json.loads(text)
        if prior != signature:
            raise SystemExit(f"{out} was written with {prior}, not {signature}; use a new --name")


def report(name):
    """Aggregate rows of an evaluation into loss-budget curves with paired bootstrap intervals."""
    rows = [json.loads(l) for path in sorted((OUT/name).glob("rows_*.jsonl")) for l in path.open()]
    table = defaultdict(lambda: defaultdict(list))
    for r in rows:
        table[(r["budget"], r["method"])]["weighted_nll"].append((r["episode"], r["origin"], weighted_nll(r["summary"])))
        for g, s in r["summary"].items():
            table[(r["budget"], r["method"])][g].append((r["episode"], r["origin"], s["nll"]))
        c = r["counts"]
        for k in ("N_text", "N_visual", "N_frame_text", "N_captures", "N_total"):
            table[(r["budget"], r["method"])][k].append((r["episode"], r["origin"], c[k]))
        if c.get("N_text_full"):
            table[(r["budget"], r["method"])]["text_compression_ratio"].append(
                (r["episode"], r["origin"], c["N_text"] / c["N_text_full"]))
        if c.get("N_captures_full"):
            table[(r["budget"], r["method"])]["captures_kept_fraction"].append(
                (r["episode"], r["origin"], c["N_captures"] / c["N_captures_full"]))
    result = {}
    for (budget, method), cols in sorted(table.items()):
        entry = {k: float(np.mean([v for _, _, v in vals if v is not None])) for k, vals in cols.items()}
        entry["windows"] = len(cols["weighted_nll"])
        if method != "recency" and (budget, "recency") in table:
            base = {(e, o): v for e, o, v in table[(budget, "recency")]["weighted_nll"]}
            diffs = defaultdict(list)
            for e, o, v in cols["weighted_nll"]:
                if (e, o) in base:
                    diffs[e].append(v - base[(e, o)])
            if diffs:
                entry["diff_vs_recency"] = float(np.mean([d for v in diffs.values() for d in v]))
                entry["diff_vs_recency_ci95"] = bootstrap(diffs)
        result[f"{budget}/{method}"] = entry
    (OUT/name/"report.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
