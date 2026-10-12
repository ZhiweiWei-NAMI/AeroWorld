"""Small structured scorer: the selector-capacity comparison of RESEARCH_PLAN §10.

Trained on the same soft inclusion targets as the language-model selector, from
hand-declared unit features only: record kind (captures included), age, position
cost, owner type, whether the owner is addressed by the query set (directly or
through a queried tuple's bindings), a capture's geometric relevance and number
of entities shown, and the budget.  Selection is the selector's: decreasing
p_keep within the optional budget.

  python -m Dataset.world_model.unified.scorer --supervision sup_r1 --name scorer_r1
"""
from __future__ import annotations

import argparse
import json
import math
import random

import torch
from torch import nn
import torch.nn.functional as F

from Dataset.world_model.unified import assemble as A, data as D, schema as S
from Dataset.world_model.unified.selector import soft_targets
from Dataset.world_model.unified.train_selector import load_records

KINDS = ("knot", "leave", "frame_change", "comm_change", "compute_change", "rel_change", "evidence_change", "event",
         "scheduled", "origin", "capture")
HEADS = ("entity", "rel", "events", "scheduled", "images")
DIM = len(KINDS) + len(HEADS) + 8


def query_scope(pw, query_ids):
    """Aliases addressed by the query set: entities (incl. owners of objects bound in queried tuples) and tuples."""
    return pw.addressed_entities(query_ids), pw.queried_tuples(query_ids)


def features(pw, query_ids, budget, units):
    ents, tuples = query_scope(pw, query_ids)
    rows = []
    for u in units:
        kind, _, name = u.head.partition("|")
        bound = set(pw.w["rel_entities"].get(name, ())) if kind == "rel" else set()
        shown = {a for a, *_ in u.capture.info["seen"]} if u.kind == "capture" else set()
        f = [float(u.kind == k) for k in KINDS] + [float(kind == h) for h in HEADS] + [
            (pw.w["origin"] - u.tick) / S.HISTORY_TICKS, math.log1p(len(u.ids) + u.extra) / 6.,
            float(kind == "entity" and name in ents), float(kind == "rel" and name in tuples),
            float(bool(bound & ents) or bool(shown & ents)), math.log2(max(2, budget)) / 15.,
            math.log1p(u.relevance), math.log1p(len(shown)) / 3.]
        rows.append(f)
    return torch.tensor(rows, dtype=torch.float32)


class Scorer(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def select(self, pw, query_ids, budget, visual="learned"):
        """Same decision rule as the selector (relevance must be set on the window's captures)."""
        if visual == "fixed":
            kept, used = A.fixed_packet(pw, query_ids, budget)
            units = pw.candidates(query_ids, "text")
        else:
            kept, used = [], 0
            units = pw.candidates(query_ids)
        if not units:
            return kept, {}
        p = torch.sigmoid(self(features(pw, query_ids, budget, units))).tolist()
        heads = set(pw.base_heads(query_ids)) | {pw.units[k].head for k in kept}
        for u, _ in sorted(zip(units, p), key=lambda up: (-up[1], up[0].key)):
            c = pw.unit_cost(u, heads)
            if used + c <= budget:
                kept.append(u.key)
                used += c
                heads.add(u.head)
        return kept, {u.key: pi for u, pi in zip(units, p)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--supervision", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--tau", type=float, default=0.02)
    args = p.parse_args()
    out = S.WORKSPACE/"selector"/args.name
    out.mkdir(parents=True, exist_ok=True)
    enc = D.Encoder()
    data = []
    for rec in load_records(args.supervision):
        w, pw = enc.window(rec["episode"], rec["origin"])
        enc.set_relevance(w, pw, rec["query_ids"])
        targets, _ = soft_targets(rec, args.tau)
        units = [pw.units[k] for k in rec["pool"] if k in pw.units]
        if not units:
            continue
        data.append((features(pw, rec["query_ids"], rec["budget"], units),
                     torch.tensor([targets[u.key] for u in units]), rec["episode"]))
    rng = random.Random(0)
    eps = sorted({d[2] for d in data})
    val_eps = set(rng.sample(eps, max(1, len(eps) // 10)))
    model = Scorer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    log = open(out/"log.jsonl", "a")
    for epoch in range(args.epochs):
        rng.shuffle(data)
        for x, y, e in data:
            if e in val_eps:
                continue
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = [float(F.binary_cross_entropy_with_logits(model(x), y)) for x, y, e in data if e in val_eps]
        log.write(json.dumps({"epoch": epoch, "val_bce": sum(vl)/max(1, len(vl))}) + "\n")
        log.flush()
    torch.save(model.state_dict(), out/"scorer.pt")


if __name__ == "__main__":
    main()
