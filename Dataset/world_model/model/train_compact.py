"""Train/evaluate the compact numerical state predictor on all TRAIN cutoffs.

Usage:
  python -m Dataset.world_model.model.train_compact --arrays DIR --cache DIR --output DIR [--steps N] ...
Cells: every tick of every TRAIN episode is a cutoff (history 5 s, horizons
0.5/1.0 s).  VALID is scored on the 48 configured windows (the cells of the
existing Qwen evaluation) and on every VALID cutoff.  TEST is never read.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from Dataset.world_model.data.provenance import resolve_source_split
from Dataset.world_model.model.compact_model import (CompactStatePredictor, activity_vocabulary, batch_tensors, build_cells,
                                                     cell_units, compact_loss, fit_units, law_terms, prediction_rows,
                                                     symmetric_scaling, symmetric_xy)


def cached_cells(cache, name, build):
    path = Path(cache)/f"{name}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files if k != "names"}, list(data["names"])
    cells, names = build()
    np.savez(path, names=np.array(names), **cells)
    return cells, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("Dataset/world_model/configs/state_distribution_5s.json"))
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--mixture", type=int, default=1, help="Gaussian components off the atom")
    parser.add_argument("--augment", action="store_true", help="random planar D4 symmetry per training row")
    parser.add_argument("--micro-scale", type=float, default=0., help="std (physical units) of an extra zero-offset component; 0 disables")
    parser.add_argument("--dev-episodes", default="L2-2_v2__seed00,L4-5_v1__seed00,L5-3_v1__seed00",
                        help="TRAIN episodes held out for model selection (whole profiles)")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--crps-weight", type=float, default=1.)
    parser.add_argument("--scale-floor", type=float, default=.05, help="minimum Gaussian std in units")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--drop-features", default="", help="comma-separated feature-name prefixes to zero (ablation)")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    episodes = sorted(config["episodes"])
    dev_eps = [e for e in args.dev_episodes.split(",") if e]
    if any(resolve_source_split(e) != "TRAIN" for e in dev_eps):
        raise ValueError("dev episodes must be TRAIN episodes")
    train_eps = [e for e in episodes if resolve_source_split(e) == "TRAIN" and e not in dev_eps]
    valid_eps = [e for e in episodes if resolve_source_split(e) == "VALID"]
    if any(resolve_source_split(e) == "TEST" for e in episodes):
        raise ValueError("TEST episodes must not be configured")
    # Activity names of all TRAIN episodes (the cached cells' index table).
    vocab = activity_vocabulary(args.arrays, sorted(train_eps+dev_eps))
    every = lambda e: range(0, 901)
    windows = {g["episode"]: g["cutoffs"] for g in config["eval_windows"]}
    all_train, names = cached_cells(args.cache, "train_all", lambda: build_cells(args.arrays, train_eps+dev_eps, every, vocab, args.workers))
    is_dev = np.isin(all_train["episode"], dev_eps)
    train = {k: v[~is_dev] for k, v in all_train.items()}
    dev = {k: v[is_dev] for k, v in all_train.items()}
    if dev_eps:  # dev selection uses a fixed stride-5 cutoff subsample
        keep = dev["cutoff"] % 5 == 0
        dev = {k: v[keep] for k, v in dev.items()}
    valid_win, _ = cached_cells(args.cache, "valid_windows", lambda: build_cells(args.arrays, valid_eps, lambda e: windows[e], vocab, args.workers))
    valid_all, _ = cached_cells(args.cache, "valid_all", lambda: build_cells(args.arrays, valid_eps, every, vocab, args.workers))
    rates = symmetric_xy(fit_units(train)) if args.augment else fit_units(train)
    units = {name: cell_units(c, rates) for name, c in (("train", train), ("dev", dev), ("valid_windows", valid_win), ("valid_all", valid_all))}
    mean, std = symmetric_scaling(train["X"], names) if args.augment else (train["X"].mean(0), train["X"].std(0)+1e-6)
    dropped = [i for i, n in enumerate(names) if any(n.startswith(p) for p in args.drop_features.split(",") if p)]
    if dropped:
        for c in (train, dev, valid_win, valid_all):
            c["X"] = c["X"].copy(); c["X"][:, dropped] = mean[dropped]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactStatePredictor(len(names), len(vocab), args.hidden, args.depth, args.dropout, args.mixture,
                                  args.micro_scale > 0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=.05)
    log = open(args.output/"log.jsonl", "w")
    def emit(row):
        log.write(json.dumps(row)+"\n"); log.flush(); print(json.dumps(row), flush=True)
    emit({"phase": "data", "train_rows": len(train["horizon"]), "dev_rows": len(dev["horizon"]), "dev_episodes": dev_eps, "valid_window_rows": len(valid_win["horizon"]),
          "valid_all_rows": len(valid_all["horizon"]), "features": len(names), "dropped": [names[i] for i in dropped],
          "parameters": sum(p.numel() for p in model.parameters()), "unit_rates": rates.tolist()})

    def evaluate(cells, name):
        model.eval(); totals = {"nll_physical": 0., "nll_units": 0., "gate_nll": 0., "crps_units": 0., "n": 0}
        outputs = []
        with torch.no_grad():
            for start in range(0, len(cells["horizon"]), 65536):
                rows = np.arange(start, min(start+65536, len(cells["horizon"])))
                batch = batch_tensors(cells, rows, units[name], mean, std, device)
                raw = model(batch["x"], batch["category"], batch["activity"], batch["horizon"])
                _, terms = compact_loss(raw, batch, crps_weight=1., floor=args.scale_floor, micro=args.micro_scale)
                n = len(rows)*6
                totals["nll_units"] += float(terms["nll_units"])*n; totals["crps_units"] += float(terms["crps_units"])*n
                totals["gate_nll"] += float(terms["gate_nll"])*n
                # physical NLL adds log(unit) on off-atom cells
                totals["nll_physical"] += float(terms["nll_units"])*n+float((torch.log(batch["unit"])*(~batch["atom"])).sum())
                totals["n"] += n
                outputs.append([t.cpu() for t in law_terms(raw, args.scale_floor, batch["unit"], args.micro_scale)])
        model.train()
        return {k: v/totals["n"] for k, v in totals.items() if k != "n"}, outputs

    started, step = time.time(), 0
    n_train = len(train["horizon"])
    best = None
    while step < args.steps:
        rows = np.random.randint(0, n_train, args.batch)
        batch = batch_tensors(train, rows, units["train"], mean, std, device,
                              np.random.randint(0, 8, len(rows)) if args.augment else None, names)
        raw = model(batch["x"], batch["category"], batch["activity"], batch["horizon"])
        loss, terms = compact_loss(raw, batch, args.crps_weight, args.scale_floor, args.micro_scale)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); schedule.step(); step += 1
        if step % 100 == 0:
            emit({"phase": "train", "step": step, "loss": float(loss), **{k: float(v) for k, v in terms.items()},
                  "lr": schedule.get_last_lr()[0], "elapsed": round(time.time()-started, 1)})
        if step % args.eval_every == 0 or step == args.steps:
            dev_scores, _ = evaluate(dev, "dev")
            scores, _ = evaluate(valid_win, "valid_windows")
            emit({"phase": "evaluation", "step": step, "dev": dev_scores, "valid_windows": scores})
            # Model selection on the held-out TRAIN profiles only (VALID is reported, not used).
            if best is None or dev_scores["nll_physical"] < best[0]:
                best = (dev_scores["nll_physical"], step, {k: v.detach().clone() for k, v in model.state_dict().items()})
    emit({"phase": "selected", "step": best[1], "dev_nll_physical": best[0]})
    model.load_state_dict(best[2])
    torch.save({"model": model.state_dict(), "selected_step": best[1], "mean": mean, "std": std, "rates": rates, "vocab": vocab, "names": names,
                "args": {k: str(v) for k, v in vars(args).items()}}, args.output/"model.pt")
    for name, cells in (("valid_windows", valid_win), ("valid_all", valid_all)):
        scores, outputs = evaluate(cells, name)
        emit({"phase": "final_"+name, **scores})
        terms = [torch.cat([o[i] for o in outputs]) for i in range(5)]
        with open(args.output/f"predictions_{name}.jsonl", "w") as f:
            for start in range(0, len(terms[0]), 20000):
                rows = np.arange(start, min(start+20000, len(terms[0])))
                for row in prediction_rows(cells, rows, [t[rows] for t in terms], units[name], args.output.name):
                    f.write(json.dumps(row)+"\n")
    emit({"phase": "complete", "elapsed": round(time.time()-started, 1)})


if __name__ == "__main__":
    main()
