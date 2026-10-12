"""Predictor training (RESEARCH_PLAN §9 step 1, and step 4 with --selector-items).

Every item is trained in marginal and joint mode (equal weight), on a
group-weighted NLL (§6).  Items are whole windows under the 20K total budget
with mixed text/capture reductions (data.TrainStream).  Launch with torchrun for
several GPUs; gradients of the LoRA and sensor-adapter parameters are
all-reduced manually after accumulation.

  torchrun --nproc_per_node 3 -m Dataset.world_model.unified.train_predictor --name p1
"""
from __future__ import annotations

import argparse
import glob
from collections import defaultdict
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from Dataset.world_model.unified import assemble as A, codes as K, data as D, schema as S
from Dataset.world_model.unified.predictor import Predictor

RUNS = S.WORKSPACE/"predictor"


def item_losses(model, item, accum, train=True, modes=("marginal", "joint")):
    """Weighted marginal + joint loss of one item (backward when training); per-group mean marginal NLL."""
    loader = D.visual_loader(item)
    encs = item["encs"]
    per_query = [None] * len(encs)
    total = 0.
    scale = 1. / (len(modes) * accum)
    if "marginal" in modes:
        w = D.query_weights([e.group for e in encs], [e.inclusion for e in encs])
        ctx = model.build_context(item["segments"], loader)

        def loss_fn(i, lp):
            loss = K.query_loss(lp, encs[i])
            per_query[i] = float(loss)
            return scale * w[i] * loss
        if train:
            total += model.marginal_backward(ctx, [(e.ids, e.steps) for e in encs], loss_fn)
        else:
            lps = model.marginal(ctx, [(e.ids, e.steps) for e in encs])
            total += float(sum(loss_fn(i, lp) for i, lp in enumerate(lps)))
    joint_nll = None
    if "joint" in modes and item["joint"]:
        order = item["joint"]
        w = D.query_weights([encs[j].group for j in order], [encs[j].inclusion for j in order])
        ctx = model.build_context(item["segments"], loader)
        lps = model.joint(ctx, [(encs[j].full_ids, encs[j].steps) for j in order])
        losses = [K.query_loss(lp, encs[j]) for lp, j in zip(lps, order)]
        loss = scale * sum(wi * l for wi, l in zip(w, losses))
        if train:
            loss.backward()
        total += float(loss)
        joint_nll = float(sum(float(l) for l in losses))
    groups = defaultdict(list)
    for e, l in zip(encs, per_query):
        if l is not None:
            groups[e.group].append(l)
    return total, {g: sum(v)/len(v) for g, v in groups.items()}, joint_nll


def dev_windows(n, seed=7):
    rng = random.Random(seed)
    dev = D.splits()["dev"]
    return sorted({(rng.choice(dev), rng.choice(D.ORIGINS[::10])) for _ in range(n)})


DEV_POLICY = "relevance"


@torch.no_grad()
def eval_rows(model, enc, windows, budget=A.TOTAL_BUDGET, policy=DEV_POLICY, images=True):
    """Per-window marginal NLL sums of the fixed DEV set: [(window, {group: (sum, n)}, has captures)]."""
    model.eval()
    rows = []
    for episode, origin in windows:
        item = enc.item(episode, origin, None, budget, policy, images=images)
        _, groups, _ = item_losses(model, item, 1, train=False, modes=("marginal",))
        sums = {g: (v * sum(1 for e in item["encs"] if e.group == g), sum(1 for e in item["encs"] if e.group == g))
                for g, v in groups.items()}
        rows.append(((episode, origin), sums, item["counts"]["N_captures"] > 0))
    model.train()
    return rows


def quick_eval(rows):
    """Fixed DEV set (deterministic windows, query sets and reductions: captures by geometric relevance in half of
    the optional capacity, text by recency, at the 20K total budget): per-group mean marginal NLL with
    window-clustered standard errors, and the group-averaged NLL used for convergence assessment and checkpoint
    selection."""
    per_window = defaultdict(list)          # group -> [(sum, n)] per window
    for _, sums, _ in rows:
        for g, r in sums.items():
            per_window[g].append(r)
    out, se = {}, {}
    for g, rs in per_window.items():
        tot, n = sum(r[0] for r in rs), sum(r[1] for r in rs)
        out[g] = tot / n
        # Ratio-estimator SE clustered by window.
        k = len(rs)
        resid = [(r[0] - out[g] * r[1]) for r in rs]
        se[g] = (k / max(1, k - 1) * sum(x * x for x in resid)) ** 0.5 / n if k > 1 else None
    weighted = sum(out.values()) / len(out)
    return out, {g: sum(r[1] for r in rs) for g, rs in per_window.items()}, se, weighted, \
        sum(r[2] for r in rows) / len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-windows", type=int, default=96)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--patience", type=int, default=0, help="stop when the group-averaged DEV NLL improved by less "
                                                           "than --tolerance (relative) over this many evaluations")
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--query-chunk", type=int, default=4, help="marginal-mode query microbatch (2: ~10 GiB, 4: ~13 GiB at 20K)")
    p.add_argument("--all-parents", action="store_true", help="final training on all 210 parents")
    p.add_argument("--init", default=None, help="checkpoint to continue from (step 4 adaptation)")
    p.add_argument("--init-text-only", action="store_true", help="load only the LoRA weights from --init")
    p.add_argument("--selector-items", default=None, help="jsonl of selector choices mixed into training (step 4), "
                                                       "written by selector.py --emit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    # Identical model construction on every rank; rank 0's trainable parameters are then broadcast, so every rank
    # starts from the same weights (gradients alone are all-reduced below).  Data sampling stays rank-specific.
    torch.manual_seed(args.seed)
    out = RUNS/args.name
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out/"args.json").write_text(json.dumps({**vars(args), "code": S.code_version(), "world_size": world}, indent=1))
    model = Predictor(device=f"cuda:{local}", lora_rank=args.lora_rank, query_chunk=args.query_chunk)
    if args.init:
        load_trainable(model, args.init, skip_adapter=args.init_text_only)
    params = model.trainable_parameters()
    init_checksum = synchronize_parameters(params, world)
    torch.manual_seed(args.seed * 1000 + rank)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., (s+1)/args.warmup) *
                                              (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1., s/args.steps)))))
    episodes = D.parents() if args.all_parents else D.splits()["fit"]
    selector_items = None
    if args.selector_items:
        paths = sorted(glob.glob(glob.escape(args.selector_items) + ".[0-9][0-9]")) or [args.selector_items]
        selector_items = [json.loads(l) for path in paths for l in open(path)]
        selector_items = [r for r in selector_items if r["episode"] in set(episodes)]
    stream = D.TrainStream(episodes, seed=args.seed, rank=rank, images=not args.no_images, selector_items=selector_items)
    loader = iter(DataLoader(stream, batch_size=1, num_workers=args.workers, collate_fn=D.collate, prefetch_factor=4))
    enc = D.Encoder()
    dev = dev_windows(args.eval_windows) if not args.all_parents else []
    log = open(out/"log.jsonl", "a") if rank == 0 else None
    if rank == 0:
        log.write(json.dumps({"step": 0, "init_checksum_per_rank": init_checksum, "trainable_parameters":
                              sum(p.numel() for p in params), "code": S.code_version()}) + "\n")
        log.flush()
    t0 = time.time()
    history, stop = [], False
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        stats, totals = defaultdict(list), []
        for _ in range(args.accum):
            item = next(loader)[0]
            total, groups, joint = item_losses(model, item, args.accum)
            totals.append(total)
            for g, v in groups.items():
                stats[g].append(v)
            stats["joint_nll_per_query"].append(joint / max(1, len(item["joint"])) if joint is not None else 0.)
            c = item["counts"]
            stats["total_positions"].append(c["N_total"])
            stats["captures"].append(c["N_captures"])
            stats["visual_fraction"].append(c["N_visual"] / max(1, c["N_total"]))
        if world > 1:
            for prm in params:
                if prm.grad is None:
                    prm.grad = torch.zeros_like(prm)
                dist.all_reduce(prm.grad)
                prm.grad /= world
        # Separate clipping: the sensor adapter's gradients must not starve the LoRA update under one global norm.
        lora = [prm for n, prm in model.named_parameters() if prm.requires_grad and not n.startswith("adapter.")]
        adapter = [prm for n, prm in model.named_parameters() if prm.requires_grad and n.startswith("adapter.")]
        norm = torch.nn.utils.clip_grad_norm_(lora, 1.0)
        torch.nn.utils.clip_grad_norm_(adapter, 1.0)
        opt.step()
        sched.step()
        if rank == 0:
            rec = {"step": step+1, "loss": sum(totals), "grad_norm": float(norm), "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time()-t0, 1), **{g: sum(v)/len(v) for g, v in stats.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                save_trainable(model, out/f"step_{step+1:06d}.pt")
        if dev and ((step + 1) % args.eval_every == 0 or step + 1 == args.steps):
            # Every rank scores its share of the DEV windows; rank 0 aggregates.
            rows = eval_rows(model, enc, dev[rank::world], images=not args.no_images)
            if world > 1:
                gathered = [None] * world
                dist.all_gather_object(gathered, rows)
                rows = [r for g in gathered for r in g]
            if rank == 0:
                nll, counts, se, weighted, vis = quick_eval(rows)
                log.write(json.dumps({"step": step+1, "dev_group_nll": nll, "dev_policy": f"{DEV_POLICY}@{A.TOTAL_BUDGET}",
                                      "dev_se": se, "dev_weighted_nll": weighted, "dev_counts": counts,
                                      "dev_windows_with_images": vis}) + "\n")
                log.flush()
                history.append(weighted)
                if args.patience and len(history) > args.patience:
                    span = history[-(args.patience + 1):]
                    if (span[0] - min(span[1:])) / span[0] < args.tolerance:
                        stop = True
                        save_trainable(model, out/f"step_{step+1:06d}.pt")
                        log.write(json.dumps({"step": step+1, "early_stop": "DEV converged", "span": span}) + "\n")
                        log.flush()
        if world > 1:
            flag = torch.tensor([float(stop)], device=f"cuda:{local}")
            dist.broadcast(flag, 0)
            stop = bool(flag.item())
            dist.barrier()
        if stop:
            break
    if world > 1:
        dist.destroy_process_group()


def synchronize_parameters(params, world):
    """Broadcast rank 0's trainable parameters to every rank; return every rank's checksum after the broadcast
    (they must all agree, else training stops)."""
    with torch.no_grad():
        if world > 1:
            for prm in params:
                dist.broadcast(prm.data, 0)
        checksum = sum(float(prm.double().sum()) + float(prm.double().abs().sum()) for prm in params)
    sums = [checksum]
    if world > 1:
        sums = [None] * world
        dist.all_gather_object(sums, checksum)
    if any(abs(c - sums[0]) > 1e-6 * max(1., abs(sums[0])) for c in sums):
        raise RuntimeError(f"ranks start from different parameters: {sums}")
    return sums


def save_trainable(model, path):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if ("lora_" in k and k.startswith("lm.")) or k.startswith("adapter.")}
    torch.save(state, path)


def load_trainable(model, path, skip_adapter=False):
    """Load LoRA/adapter weights; skip_adapter keeps the freshly initialized sensor adapter (warm start of the
    language side after the sensor inputs changed)."""
    state = torch.load(path, map_location="cpu")
    if skip_adapter:
        state = {k: v for k, v in state.items() if not k.startswith("adapter.")}
    missing = model.load_state_dict(state, strict=False)
    if missing.unexpected_keys:
        raise ValueError(f"unexpected keys in {path}: {missing.unexpected_keys[:5]}")


if __name__ == "__main__":
    main()
