"""Selector training on selection supervision (RESEARCH_PLAN §9 step 3; predictor frozen).

Each example reads the window's pool units (text records and captures, labelled by soft inclusion targets)
together with unlabelled candidate units drawn from the rest of the window (as many as the pool, within a token
cap), in the canonical reading order, so the selector learns to score labelled units among realistic
distractors.  Loss = binary cross-entropy against the soft inclusion targets + RANK_WEIGHT x pairwise ranking of
the measured removal effects (captures, record blocks and complementary text-visual groups).

  CUDA_VISIBLE_DEVICES=5 python -m Dataset.world_model.unified.train_selector --supervision sup_r1 --name sel_r1
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from Dataset.world_model.unified import data as D, schema as S
from Dataset.world_model.unified.selector import Selector, build_chunks, ranking_loss, selector_loss, soft_targets

RUNS = S.WORKSPACE/"selector"
RANK_WEIGHT = 0.5


def load_records(names):
    out = []
    for name in names:
        for path in sorted((S.WORKSPACE/"supervision"/name).glob("shard_*.jsonl")):
            out += [json.loads(line) for line in path.open()]
    return out


def example(enc, rec, rng, tau=0.02, max_tokens=48000):
    w, pw = enc.window(rec["episode"], rec["origin"])
    qids = rec["query_ids"]
    targets, info = soft_targets(rec, tau)
    pool = [k for k in rec["pool"] if k in pw.units]
    kinds = "text" if rec.get("visual") == "fixed" else "all"
    others = [u.key for u in pw.candidates(qids, kinds) if u.key not in set(pool)]
    rng.shuffle(others)
    units, used = list(pool), sum(len(pw.selector_ids(pw.units[k])) for k in pool)
    for k in others[:len(pool)]:
        n = len(pw.selector_ids(pw.units[k]))
        if used + n > max_tokens:
            break
        units.append(k)
        used += n
    chunks = build_chunks(pw, enc.tok, qids, rec["budget"], units)
    return pw, chunks, targets, rec.get("effects", []), info


def loss_of(model, enc, rec, rng, tau):
    pw, chunks, targets, effects, _ = example(enc, rec, rng, tau)
    if not chunks:
        return None, None, None
    logits, keys = model(chunks, rec["budget"], pw)
    bce = selector_loss(logits, keys, targets)
    rank = ranking_loss(logits, keys, effects)
    if bce is None:
        return None, None, None
    total = bce + (RANK_WEIGHT * rank if rank is not None else 0.)
    return total, float(bce), (float(rank) if rank is not None else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--supervision", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--holdout", type=float, default=0.1, help="fraction of supervision windows for selector validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None, help="selector state to continue from (later rounds of the alternation)")
    args = p.parse_args()
    out = RUNS/args.name
    out.mkdir(parents=True, exist_ok=True)
    (out/"args.json").write_text(json.dumps({**vars(args), "code": S.code_version()}, indent=1))
    recs = load_records(args.supervision)
    rng = random.Random(args.seed)
    rng.shuffle(recs)
    # Windows of one parent stay on one side of the selector-validation split.
    eps = sorted({r["episode"] for r in recs})
    val_eps = set(rng.sample(eps, int(round(len(eps) * args.holdout))))     # --holdout 0: final training on all
    train = [r for r in recs if r["episode"] not in val_eps]
    val = [r for r in recs if r["episode"] in val_eps]
    model = Selector()
    if args.init:
        missing = model.load_state_dict(torch.load(args.init, map_location="cpu"), strict=False)
        if missing.unexpected_keys:
            raise ValueError(f"unexpected keys in {args.init}: {missing.unexpected_keys[:5]}")
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    enc = D.Encoder()
    log = open(out/"log.jsonl", "a")
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        rng.shuffle(train)
        opt.zero_grad(set_to_none=True)
        losses, bces, ranks = [], [], []
        for i, rec in enumerate(train):
            loss, bce, rank = loss_of(model, enc, rec, rng, args.tau)
            if loss is None:
                continue
            (loss / args.accum).backward()
            losses.append(float(loss))
            bces.append(bce)
            if rank is not None:
                ranks.append(rank)
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                log.write(json.dumps({"epoch": epoch, "step": step, "loss": sum(losses)/len(losses),
                                      "bce": sum(bces)/len(bces), "rank": sum(ranks)/len(ranks) if ranks else None,
                                      "elapsed_s": round(time.time()-t0, 1)}) + "\n")
                log.flush()
                losses, bces, ranks = [], [], []
        model.eval()
        with torch.no_grad():
            vb, vr = [], []
            for r in val:
                _, bce, rank = loss_of(model, enc, r, random.Random(0), args.tau)
                if bce is not None:
                    vb.append(bce)
                if rank is not None:
                    vr.append(rank)
        model.train()
        log.write(json.dumps({"epoch": epoch, "val_bce": sum(vb)/max(1, len(vb)),
                              "val_rank": sum(vr)/len(vr) if vr else None, "val_windows": len(vb)}) + "\n")
        log.flush()
        torch.save(trainable_state(model), out/f"epoch_{epoch}.pt")


def trainable_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.startswith(("budget.", "head.", "out.", "visual", "kind")) or ("lora_" in k and k.startswith("lm."))}


if __name__ == "__main__":
    main()
