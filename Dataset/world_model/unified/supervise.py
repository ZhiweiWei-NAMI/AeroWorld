"""Selection supervision with the predictor frozen (RESEARCH_PLAN §6, §9 step 2).

For each sampled window, query set and total budget, candidate evidence sets are scored by the predictor's
group-weighted marginal NLL on the realized future (position, link, compute, predicate, first-change and event
groups), with queries and labels held fixed.  Units are text records and captures.

visual = "learned" (default): text and captures are perturbed together and every set is the complete selected
text-visual evidence.  visual = "fixed": a text-only stage; the capture packet (assemble.fixed_packet) is held
fixed in every set and only text varies within the remaining capacity.

The pool P (the stand-in for the full set, which exceeds the context) is the union of the simple policies'
selections at the optional budget b and a random mixed reduction up to 2b.  Sets:

  * the simple policies (recency, uniform time, change point, relevance) and random mixed subsets of P at b,
  * block replacement inside the budget (a block of the best set is replaced by a random refill from the rest
    of P; repeated while it improves), so every scored set respects the 20K total,
  * complementary text-visual groups around the best set: for a kept capture, the kept records of the entities
    it shows within +-1 s; the capture alone, the records alone and both are removed, so evidence useful only in
    combination is measured (interaction = L(-both) - L(-capture) - L(-records) + L(S*)),
  * swaps around the best set (a capture or a record block exchanged for outside units of similar cost),
  * removal effects u(S*) = L(S* minus block) - L(S*) of kept captures and kept per-entity record blocks.

Records keep every evaluated set (indices into P) with its loss, per-group losses and cost; soft targets and
ranking pairs are formed at selector-training time.

  CUDA_VISIBLE_DEVICES=5 python -m Dataset.world_model.unified.supervise --checkpoint .../step_001000.pt --out sup_r1
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time

import torch

from Dataset.world_model.unified import assemble as A, data as D, schema as S
from Dataset.world_model.unified.predictor import Predictor
from Dataset.world_model.unified.train_predictor import item_losses, load_trainable

OUT = S.WORKSPACE/"supervision"
GROUP_WINDOW = 10            # ticks: records within +-1 s of a capture belong to its complementary group


class SetScorer:
    """L_theta(S; I) for kept sets of one window with fixed queries; frozen features reused across sets."""
    def __init__(self, model, enc, episode, origin, rng, total):
        self.model, self.enc = model, enc
        self.episode, self.origin = episode, origin
        self.w, self.pw = enc.window(episode, origin)
        self.qids = self.pw.select_queries(rng)
        self.total = total
        self.budget = enc.optional_budget(self.w, self.pw, self.qids, total)
        enc.set_relevance(self.w, self.pw, self.qids)
        self.features, self.cache = {}, {}

    @torch.no_grad()
    def __call__(self, kept):
        key = tuple(sorted(kept))
        if key not in self.cache:
            item = self.enc.item(self.episode, self.origin, kept=list(key), query_ids=self.qids, budget=self.total,
                                 prepared=(self.w, self.pw), features=self.features)
            total, groups, _ = item_losses(self.model, item, 1, train=False, modes=("marginal",))
            self.cache[key] = {"loss": total, "groups": groups, "counts": item["counts"]}
        return self.cache[key]

    def cost(self, keys):
        return self.pw.set_cost(keys, self.qids)


def fit(sc, keys, budget, rng):
    """Drop random units until the set fits the budget."""
    keys = list(keys)
    while keys and sc.cost(keys) > budget:
        keys.remove(rng.choice(keys))
    return keys


def complementary_group(pw, cap, keep):
    """Kept text records of the entities a capture shows (directly, or through a relation binding them or objects
    they own; window rel_entities), within +-GROUP_WINDOW ticks of the capture."""
    shown = {a for a, *_ in cap.capture.info["seen"]}
    out = []
    for k in keep:
        u = pw.units[k]
        if u.kind == "capture" or abs(u.tick - cap.tick) > GROUP_WINDOW:
            continue
        kind, _, name = u.head.partition("|")
        if (kind == "entity" and name in shown) or (kind == "rel" and set(pw.w["rel_entities"].get(name, ())) & shown):
            out.append(k)
    return out


def supervise_window(model, enc, episode, origin, total, rng, visual="learned", n_random=6, rounds=2, blocks=4,
                     groups=2, swaps=3, deletions=6):
    sc = SetScorer(model, enc, episode, origin, rng, total)
    pw, qids = sc.pw, sc.qids
    kinds = "text" if visual == "fixed" else "all"
    packet, packet_cost = A.fixed_packet(pw, qids, sc.budget) if visual == "fixed" else ([], 0)
    budget = sc.budget - packet_cost
    sets = {}

    def evaluate(keys, source):
        keys = sorted(set(keys) | set(packet))
        r = sc(keys)
        sets.setdefault(tuple(keys), {"keep": keys, "loss": r["loss"], "groups": r["groups"], "cost": sc.cost(keys),
                                      "counts": r["counts"], "source": source})
        return r["loss"]

    def text_only(keys):
        return [k for k in keys if k not in set(packet)]

    def policy_set(pol):
        if visual == "fixed":
            order = A.policy_order(pw, qids, "recency" if pol == "relevance" else pol, kinds="text")
            return pw.greedy_fill(order, budget, qids)[0]
        return A.select(pw, qids, budget, pol, random.Random(rng.random()))[0]

    policy_sets = {}
    for pol in ("recency", "uniform_time", "change_point", "relevance"):
        policy_sets[pol] = policy_set(pol)
        evaluate(policy_sets[pol], pol)
    pool = set().union(*map(set, policy_sets.values()))
    if visual == "fixed":
        pool |= set(pw.greedy_fill(A.policy_order(pw, qids, "random", random.Random(rng.random()), "text"), 2 * budget,
                                   qids)[0])
    else:
        pool |= set(A.mixed_fill(pw, qids, 2 * budget, "random", "random", rng.uniform(0., .8),
                                 random.Random(rng.random()))[0])
    pool = sorted(pool)
    pool_units = [pw.units[k] for k in pool]

    class Sub:              # the pool as a candidate set for mixed_fill
        def __init__(self):
            self.units = pw.units
            self.unit_cost, self.base_heads, self.greedy_fill = pw.unit_cost, pw.base_heads, pw.greedy_fill

        def candidates(self, _qids, kinds="all"):
            return [u for u in pool_units if kinds == "all" or (kinds == "visual") == (u.kind == "capture")]
    sub = Sub()
    for _ in range(n_random):
        kept, _ = A.mixed_fill(sub, qids, budget, "random", "random", rng.uniform(0., .8), random.Random(rng.random()))
        evaluate(kept, "random")
    # Block replacement search inside the budget (every scored set fits the 20K total): split the best set into
    # blocks, replace one block by a random refill from the rest of the pool, keep the best replacement if it helps.
    feasible = lambda: [s for s in sets.values() if s["cost"] <= sc.budget]
    for _ in range(rounds):
        best = min(feasible(), key=lambda s: s["loss"])
        keep = text_only(best["keep"])
        rng.shuffle(keep)
        trials = []
        for b in range(blocks):
            drop = set(keep[b::blocks])
            rest = [k for k in keep if k not in drop]
            outside = [pw.units[k] for k in pool if k not in set(keep)]
            rng.shuffle(outside)
            heads = set(pw.base_heads(qids)) | set(pw.units[k].head for k in rest) | \
                set(pw.units[k].head for k in packet)
            used = sc.cost(rest + list(packet))
            add = []
            for u in outside:
                c = pw.unit_cost(u, heads)
                if used + c <= sc.budget:
                    add.append(u.key)
                    used += c
                    heads.add(u.head)
            trials.append(evaluate(rest + add, "block_replace"))
        if min(trials) >= best["loss"]:
            break
    best = min(feasible(), key=lambda s: s["loss"])
    keep = text_only(best["keep"])
    effects = []
    # Complementary text-visual groups (learned visual selection only).
    caps = [k for k in keep if pw.units[k].kind == "capture"]
    for ck in rng.sample(caps, min(groups, len(caps))):
        recs = complementary_group(pw, pw.units[ck], keep)
        if not recs:
            continue
        l_cap = evaluate([k for k in keep if k != ck], "group_capture")
        l_rec = evaluate([k for k in keep if k not in set(recs)], "group_records")
        l_both = evaluate([k for k in keep if k != ck and k not in set(recs)], "group_both")
        base = best["loss"]
        effects += [{"kind": "capture", "members": [ck], "delta": l_cap - base},
                    {"kind": "records", "members": recs, "delta": l_rec - base},
                    {"kind": "group", "members": [ck] + recs, "delta": l_both - base,
                     "interaction": l_both - l_cap - l_rec + base}]
    # Swaps around the best feasible set (sets over the budget are never "best").
    for _ in range(swaps):
        keep = text_only(best["keep"])
        outside = [k for k in pool if k not in set(keep)]
        if not keep or not outside:
            break
        out_k = rng.choice(keep)
        need = pw.units[out_k].extra + len(pw.units[out_k].ids)
        add, got = [], 0
        for k in rng.sample(outside, len(outside)):
            if got >= need:
                break
            add.append(k)
            got += pw.units[k].extra + len(pw.units[k].ids)
        evaluate(fit(sc, [k for k in keep if k != out_k] + add, budget, rng), "swap")
        best = min(feasible(), key=lambda s: s["loss"])
    # Removal effects of kept captures and kept per-entity record blocks.
    keep = text_only(best["keep"])
    by_head = {}
    for k in keep:
        if pw.units[k].kind != "capture":
            by_head.setdefault(pw.units[k].head, []).append(k)
    blocks_ = [("capture", [k]) for k in keep if pw.units[k].kind == "capture"] + \
        [("record_block", v) for v in by_head.values()]
    for kind, members in rng.sample(blocks_, min(deletions, len(blocks_))):
        loss = evaluate([k for k in keep if k not in set(members)], "removal_effect")
        effects.append({"kind": kind, "members": members, "delta": loss - best["loss"]})
    index = {k: i for i, k in enumerate(pool)}
    for s in sets.values():
        s["keep"] = [index[k] for k in s["keep"] if k in index]
    return {"episode": episode, "origin": origin, "total_budget": total, "budget": sc.budget, "visual": visual,
            "query_ids": qids, "pool": pool, "packet": packet,
            "unit_costs": {k: pw.units[k].extra + len(pw.units[k].ids) for k in pool},
            "kinds": {k: pw.units[k].kind for k in pool}, "heads": {k: pw.units[k].head for k in pool},
            "best": best["keep"], "best_loss": best["loss"], "effects": effects,
            "evaluated_sets": list(sets.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--windows", type=int, default=400)
    p.add_argument("--budgets", type=int, nargs="+", default=[A.TOTAL_BUDGET], help="total budgets")
    p.add_argument("--visual", choices=("learned", "fixed"), default="learned")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--all-parents", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--query-chunk", type=int, default=8)
    args = p.parse_args()
    out = OUT/args.out
    out.mkdir(parents=True, exist_ok=True)
    model = Predictor(device="cuda", query_chunk=args.query_chunk)
    load_trainable(model, args.checkpoint)
    model.eval()
    enc = D.Encoder()
    episodes = D.parents() if args.all_parents else D.splits()["fit"]
    rng = random.Random(args.seed)
    # Every parent gets a window before any parent gets a second one (with --windows >= the number of parents).
    jobs = [(e, rng.choice(D.ORIGINS), rng.choice(args.budgets), rng.random())
            for e in D.cover_episodes(episodes, args.windows, rng)]
    # One signature per shard (no shared file to race on); a resume must use the same models, settings and
    # number of shards.
    signature = {"checkpoint": args.checkpoint, "visual": args.visual, "budgets": args.budgets, "seed": args.seed,
                 "all_parents": args.all_parents, "windows": args.windows, "shards": args.shards,
                 "commit": S.code_version()["commit"]}
    for other in out.glob("signature_*.json"):
        prior = json.loads(other.read_text())
        if prior != signature:
            raise SystemExit(f"{out} was written with {prior}, not {signature}; use a new --out")
    (out/f"signature_{args.shard:02d}.json").write_text(json.dumps(signature))
    code = S.code_version()
    path = out/f"shard_{args.shard:02d}.jsonl"
    done = {(r["episode"], r["origin"], r["total_budget"]) for p in out.glob("shard_*.jsonl") for r in map(json.loads, p.open())}
    with path.open("a") as fh:
        for i, (episode, origin, total, seed) in enumerate(jobs):
            if i % args.shards != args.shard or (episode, origin, total) in done:
                continue
            t = time.time()
            try:
                rec = supervise_window(model, enc, episode, origin, total, random.Random(seed), visual=args.visual)
            except ValueError as exc:          # mandatory inputs alone exceed this budget
                print(json.dumps({"skipped": str(exc)}), flush=True)
                continue
            rec["seconds"] = round(time.time() - t, 1)
            rec["checkpoint"] = args.checkpoint
            rec["code"] = code
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
