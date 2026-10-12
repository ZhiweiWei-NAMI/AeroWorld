"""Train/evaluate the compact link predictor (latency distribution + handover probability) and its baselines.

Cells: every communication tick (multiple of 5) of the TRAIN episodes; model
selection on the held-out TRAIN profiles; VALID scored on the 48 configured
windows (the cells of the Qwen evaluation) and on every VALID communication
tick.  Writes prediction rows for the model and the baselines.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from Dataset.world_model.data.provenance import resolve_source_split
from Dataset.world_model.model.compact_comm import CompactCommPredictor, build_comm_cells
from Dataset.world_model.model.compact_data import HORIZONS
from Dataset.world_model.model.compact_model import FEATURE_CLIP, compact_loss, law_terms, mixture_median

FIELD = "communication.latency_ms"
PHYSICS_TOLERANCE = 1e-5


def cached(cache, name, build):
    path = Path(cache)/f"{name}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files if k != "names"}, list(data["names"])
    cells, names = build()
    np.savez(path, names=np.array(names), **cells)
    return cells, names


def latency_rows(cells, p, comps, name, band=0.):
    """comps: [N, K, 3] physical (weight, location, scale)."""
    rows = []
    anchor = cells["anchor"]
    median = mixture_median(torch.tensor(anchor), torch.tensor(p), torch.tensor(comps[:, :, 0]), torch.tensor(comps[:, :, 1]),
                            torch.tensor(comps[:, :, 2])).numpy()
    mean = p*anchor+(1-p)*(comps[:, :, 0]*comps[:, :, 1]).sum(1)
    for i in range(len(anchor)):
        rows.append({"episode": str(cells["episode"][i]), "entity": str(cells["entity"][i]), "field": FIELD, "component": 0,
                     "window_cutoff": int(cells["cutoff"][i]), "horizon_ticks": HORIZONS[int(cells["horizon"][i])],
                     "law": "anchor_inflated_normal", "model": name,
                     "parameters": {"anchor": float(anchor[i]), "anchor_probability": float(p[i]), "atom_band": float(band),
                                    "components": comps[i].tolist(), "location": float(comps[i, 0, 1]), "scale": float(comps[i, 0, 2]),
                                    "mixture_median": float(median[i]), "mixture_mean": float(mean[i])},
                     "truth": float(cells["truth"][i]), "truth_atom": bool(cells["atom"][i])})
    return rows


def handover_rows(cells, prob, name):
    return [{"episode": str(cells["episode"][i]), "entity": str(cells["entity"][i]), "field": "communication.handover_active",
             "window_cutoff": int(cells["cutoff"][i]), "horizon_ticks": HORIZONS[int(cells["horizon"][i])], "model": name,
             "probability": float(prob[i]), "truth": bool(cells["handover"][i])} for i in range(len(prob))]


def write(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("Dataset/world_model/configs/state_distribution_5s.json"))
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--mixture", type=int, default=3)
    parser.add_argument("--scale-floor", type=float, default=.005)
    parser.add_argument("--crps-weight", type=float, default=1.)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--dev-episodes", default="L2-2_v2__seed00,L4-5_v1__seed00,L5-3_v1__seed00")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--micro-scale", type=float, default=0., help="std (ms) of an extra zero-offset latency component")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--physics-anchor", action="store_true",
                        help="anchor the law at the link-model forecast; atom = forecast exact within PHYSICS_TOLERANCE")
    parser.add_argument("--physics-center", action="store_true",
                        help="centre the Gaussian components on the link-model latency forecast (atom stays at persistence)")
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    episodes = sorted(config["episodes"])
    dev_eps = args.dev_episodes.split(",")
    train_eps = [e for e in episodes if resolve_source_split(e) == "TRAIN"]
    valid_eps = [e for e in episodes if resolve_source_split(e) == "VALID"]
    windows = {g["episode"]: g["cutoffs"] for g in config["eval_windows"]}
    every = lambda e: range(0, 901, 5)
    all_train, names = cached(args.cache, "comm_train_all", lambda: build_comm_cells(config, args.arrays, train_eps, every))
    valid_win, _ = cached(args.cache, "comm_valid_windows", lambda: build_comm_cells(config, args.arrays, valid_eps, lambda e: windows[e]))
    valid_all, _ = cached(args.cache, "comm_valid_all", lambda: build_comm_cells(config, args.arrays, valid_eps, every))
    if args.physics_anchor:
        # The link-model forecast is exact (<= 1e-5 ms) for ~87% of TRAIN cells: it becomes the anchor and its atom.
        for cells in (all_train, valid_win, valid_all):
            cells["persistence"] = cells["anchor"].copy()
            cells["anchor"] = cells["physics"]
            cells["atom"] = np.abs(cells["truth"]-cells["physics"]) <= PHYSICS_TOLERANCE
    is_dev = np.isin(all_train["episode"], dev_eps)
    train = {k: v[~is_dev] for k, v in all_train.items()}; dev = {k: v[is_dev] for k, v in all_train.items()}
    sets = {"dev": dev, "valid_windows": valid_win, "valid_all": valid_all}
    # Unit: TRAIN off-atom latency residual rate per horizon-independent second.
    residual = (train["truth"]-train["anchor"])/train["elapsed"]
    rate = float(np.sqrt(np.mean(residual[~train["atom"]]**2)))
    mean, std = train["X"].mean(0), train["X"].std(0)+1e-6
    phys_col = names.index("latency_phys_target[0]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = open(args.output/"log.jsonl", "w")
    def emit(row):
        log.write(json.dumps(row)+"\n"); log.flush(); print(json.dumps(row), flush=True)
    emit({"phase": "data", "train": len(train["horizon"]), "dev": len(dev["horizon"]), "valid_windows": len(valid_win["horizon"]),
          "valid_all": len(valid_all["horizon"]), "features": len(names), "rate": rate,
          "train_handover_rate": float(train["handover"].mean()), "train_atom_rate": float(train["atom"].mean())})

    def tensors(cells, rows):
        unit = rate*cells["elapsed"][rows]
        return {"x": torch.tensor(np.clip((cells["X"][rows]-mean)/std, -FEATURE_CLIP, FEATURE_CLIP), dtype=torch.float32, device=device),
                "horizon": torch.tensor(cells["horizon"][rows], device=device),
                "unit": torch.tensor(unit[:, None], dtype=torch.float32, device=device),
                "residual_units": torch.tensor(((cells["truth"][rows]-cells["anchor"][rows])/unit)[:, None], dtype=torch.float32, device=device),
                "band_units": torch.tensor(((PHYSICS_TOLERANCE if args.physics_anchor else 0.)/unit)[:, None], dtype=torch.float32, device=device),
                "atom": torch.tensor(cells["atom"][rows][:, None], device=device),
                "handover": torch.tensor(cells["handover"][rows], device=device),
                "center": torch.tensor(((cells["X"][rows, phys_col]-cells["anchor"][rows])/unit if args.physics_center
                                        else np.zeros(len(rows))), dtype=torch.float32, device=device)}

    def shifted(raw, b):
        # Gaussian offsets are relative to the link-model forecast (in units); the atom is unchanged.
        raw = raw.clone()
        raw[:, 0, 1:1+3*args.mixture:3] = raw[:, 0, 1:1+3*args.mixture:3]+b["center"][:, None]
        return raw

    def model_predict(model, cells):
        model.eval(); outs = []
        with torch.no_grad():
            for s in range(0, len(cells["horizon"]), 65536):
                rows = np.arange(s, min(s+65536, len(cells["horizon"])))
                b = tensors(cells, rows)
                raw, logit = model(b["x"], b["horizon"]); raw = shifted(raw, b)
                _, terms = compact_loss(raw, b, 1., args.scale_floor, args.micro_scale)
                lp, _, lw, off, sc = law_terms(raw, args.scale_floor, b["unit"], args.micro_scale)
                outs.append((float(terms["nll_units"])*len(rows), float(terms["crps_units"])*len(rows),
                             float(F.binary_cross_entropy_with_logits(logit, b["handover"], reduction="sum")),
                             lp[:, 0].exp().cpu().numpy(), lw[:, 0].exp().cpu().numpy(), off[:, 0].cpu().numpy(), sc[:, 0].cpu().numpy(),
                             torch.sigmoid(logit).cpu().numpy(), (torch.log(b["unit"][:, 0])*(~b["atom"][:, 0])).sum().item()))
        model.train()
        n = len(cells["horizon"])
        unit = rate*cells["elapsed"]
        p = np.concatenate([o[3] for o in outs]); w = np.concatenate([o[4] for o in outs])
        mu = cells["anchor"][:, None]+np.concatenate([o[5] for o in outs])*unit[:, None]; sig = np.concatenate([o[6] for o in outs])*unit[:, None]
        scores = {"nll_physical": (sum(o[0] for o in outs)+sum(o[8] for o in outs))/n, "crps_units": sum(o[1] for o in outs)/n,
                  "handover_bce": sum(o[2] for o in outs)/n}
        return scores, p, np.stack([w, mu, sig], -1), np.concatenate([o[7] for o in outs])

    if not args.baselines:
        model = CompactCommPredictor(len(names), args.hidden, 2, args.dropout, args.mixture, args.micro_scale > 0).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=.05)
        best = None
        for step in range(1, args.steps+1):
            rows = np.random.randint(0, len(train["horizon"]), args.batch)
            b = tensors(train, rows)
            raw, logit = model(b["x"], b["horizon"]); raw = shifted(raw, b)
            loss, terms = compact_loss(raw, b, args.crps_weight, args.scale_floor, args.micro_scale)
            bce = F.binary_cross_entropy_with_logits(logit, b["handover"])
            total = loss+bce
            optimizer.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); schedule.step()
            if step % args.eval_every == 0 or step == args.steps:
                dev_scores = model_predict(model, dev)[0]; val_scores = model_predict(model, valid_win)[0]
                emit({"phase": "evaluation", "step": step, "train_loss": float(total), "dev": dev_scores, "valid_windows": val_scores})
                key = dev_scores["nll_physical"]+dev_scores["handover_bce"]
                if best is None or key < best[0]:
                    best = (key, step, {k: v.detach().clone() for k, v in model.state_dict().items()})
        model.load_state_dict(best[2])
        emit({"phase": "selected", "step": best[1]})
        torch.save({"model": model.state_dict(), "mean": mean, "std": std, "rate": rate, "names": names,
                    "args": {k: str(v) for k, v in vars(args).items()}}, args.output/"model.pt")
        for name, cells in sets.items():
            scores, p, comps, prob = model_predict(model, cells)
            emit({"phase": "final_"+name, **scores})
            write(args.output/f"latency_{name}.jsonl", latency_rows(cells, p, comps, args.output.name,
                                                                   PHYSICS_TOLERANCE if args.physics_anchor else 0.))
            write(args.output/f"handover_{name}.jsonl", handover_rows(cells, prob, args.output.name))
        return

    # ---------------- baselines (fitted on the training cells; dev kept out for parity)
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    point = lambda cells, values: (np.zeros(len(values)), np.stack([np.ones(len(values)), values, np.full(len(values), 1e-9)], -1)[:, None])
    atom_rate = {h: float(np.clip(train["atom"][train["horizon"] == h].mean(), 1e-4, 1-1e-4)) for h in range(len(HORIZONS))}
    off_scale = {h: float(np.sqrt(np.mean((train["truth"]-train["anchor"])[(train["horizon"] == h) & ~train["atom"]]**2))) for h in range(len(HORIZONS))}
    gbm = []
    Xh = lambda c: np.column_stack([c["X"], c["horizon"]])
    reg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=.1, max_leaf_nodes=31, random_state=0)
    reg.fit(Xh(train), train["truth"]-train["anchor"])
    cls = HistGradientBoostingClassifier(max_iter=200, learning_rate=.1, max_leaf_nodes=15, random_state=0)
    cls.fit(Xh(train), train["handover"].astype(int))
    logistic = LogisticRegression(max_iter=2000, C=1.)
    logistic.fit(np.clip((train["X"]-mean)/std, -FEATURE_CLIP, FEATURE_CLIP), train["handover"].astype(int))
    base_rate = float(np.clip(train["handover"].mean(), 1e-5, 1))
    trend_col = names.index("latency_rate_lag5[0]")
    adv_col = names.index("distance_advantage_cv_target[0]"); since_col = names.index("ticks_since_station_change[0]")
    for name, cells in sets.items():
        a = cells["anchor"]; h = cells["horizon"]
        write(args.output/f"latency_persistence_{name}.jsonl", latency_rows(cells, *point(cells, a), "persistence"))
        phys = cells["X"][:, names.index("latency_phys_target[0]")]
        write(args.output/f"latency_link_physics_{name}.jsonl", latency_rows(cells, *point(cells, phys), "link_physics"))
        trend = a+cells["X"][:, trend_col]*cells["elapsed"]
        write(args.output/f"latency_linear_trend_{name}.jsonl", latency_rows(cells, *point(cells, trend), "linear_trend"))
        p = np.array([atom_rate[int(x)] for x in h]); sig = np.array([off_scale[int(x)] for x in h])
        write(args.output/f"latency_fitted_persistence_law_{name}.jsonl",
              latency_rows(cells, p, np.stack([np.ones(len(a)), a, sig], -1)[:, None], "fitted_persistence_law"))
        write(args.output/f"latency_gbm_residual_{name}.jsonl", latency_rows(cells, *point(cells, a+reg.predict(Xh(cells))), "gbm_residual"))
        write(args.output/f"handover_base_rate_{name}.jsonl", handover_rows(cells, np.full(len(a), base_rate), "base_rate"))
        write(args.output/f"handover_gbm_{name}.jsonl", handover_rows(cells, cls.predict_proba(Xh(cells))[:, 1], "gbm"))
        write(args.output/f"handover_logistic_{name}.jsonl",
              handover_rows(cells, logistic.predict_proba(np.clip((cells["X"]-mean)/std, -FEATURE_CLIP, FEATURE_CLIP))[:, 1], "logistic"))
        # Explicit geometric rule of the simulator's policy (15 m margin, 30-tick dwell), smoothed.
        rule = (cells["X"][:, adv_col] > 15.) & (cells["X"][:, since_col]*60 >= 30)
        write(args.output/f"handover_rule_{name}.jsonl", handover_rows(cells, np.where(rule, .9, base_rate), "geometric_rule"))
    emit({"phase": "baselines_complete"})


if __name__ == "__main__":
    main()
