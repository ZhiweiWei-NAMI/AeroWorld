"""Forecasting baselines on the compact cells, exported in the paired-evaluation row format.

* fitted_anchor_law: per (category, component, horizon) TRAIN atom rate and
  zero-mean off-atom Gaussian scale around the constant-velocity anchor.
* cv_gaussian: constant-velocity Gaussian (Kalman CV predictive form) with the
  TRAIN residual RMS per (category, component, horizon); no atom.
* constant_acceleration: point forecast with the last-0.5 s acceleration.
* gbm_residual: gradient-boosted median of the anchor residual on the compact
  features (point forecast).
Point forecasts are written as a Gaussian of negligible width (CRPS = abs error).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Dataset.world_model.model.compact_data import CATEGORIES, HORIZONS
from Dataset.world_model.model.compact_model import COMPONENTS

POINT_SCALE = 1e-9


def load(cache, name):
    data = np.load(Path(cache)/f"{name}.npz", allow_pickle=True)
    return {k: data[k] for k in data.files if k != "names"}, list(data["names"])


def group_index(cells):
    return cells["category"]*len(HORIZONS)+cells["horizon"]


def write(path, cells, p, mu, sigma, name):
    with open(path, "w") as f:
        for r in range(len(cells["horizon"])):
            for j, (field, component) in enumerate(COMPONENTS):
                a = float(cells["anchor"][r, j])
                pr = float(p[r, j])
                point = pr*a+(1-pr)*float(mu[r, j])
                median = a if pr >= .5 else float(mu[r, j])
                f.write(json.dumps({"episode": str(cells["episode"][r]), "entity": str(cells["entity"][r]), "field": field,
                                    "component": component, "window_cutoff": int(cells["cutoff"][r]),
                                    "horizon_ticks": HORIZONS[int(cells["horizon"][r])], "law": "anchor_inflated_normal",
                                    "model": name, "parameters": {"anchor": a, "anchor_probability": pr, "location": float(mu[r, j]),
                                                                  "scale": float(sigma[r, j]), "mixture_median": median,
                                                                  "mixture_mean": point, "atom_band": float(cells["band"][r, j])},
                                    "truth": float(cells["truth"][r, j]), "truth_atom": bool(cells["atom"][r, j])})+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval", default="valid_windows")
    parser.add_argument("--gbm-rows", type=int, default=400000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train, names = load(args.cache, "train_all")
    cells, _ = load(args.cache, args.eval)
    groups = len(CATEGORIES)*len(HORIZONS)
    residual = train["truth"]-train["anchor"]
    g_train, g_eval = group_index(train), group_index(cells)
    atom_rate = np.full((groups, 6), .5); off_scale = np.ones((groups, 6)); all_scale = np.ones((groups, 6))
    pooled_off = np.array([np.sqrt(np.mean(residual[~train["atom"][:, j], j]**2)) for j in range(6)])
    for g in range(groups):
        rows = g_train == g
        if not rows.any():
            off_scale[g] = all_scale[g] = pooled_off
            continue
        atom = train["atom"][rows]
        atom_rate[g] = np.clip(atom.mean(0), 1e-4, 1-1e-4)
        for j in range(6):
            off = residual[rows][~atom[:, j], j]
            off_scale[g, j] = np.sqrt(np.mean(off**2)) if len(off) >= 20 else pooled_off[j]
            all_scale[g, j] = max(np.sqrt(np.mean(residual[rows][:, j]**2)), 1e-6)
    anchor = cells["anchor"]
    write(args.output/f"fitted_anchor_law_{args.eval}.jsonl", cells, atom_rate[g_eval], anchor, off_scale[g_eval], "fitted_anchor_law")
    write(args.output/f"cv_gaussian_{args.eval}.jsonl", cells, np.full_like(anchor, 1e-9), anchor, all_scale[g_eval], "cv_gaussian")
    # Constant acceleration from the last 0.5 s velocity change (feature accel_lag5).
    accel = cells["X"][:, [names.index(f"accel_lag5[{i}]") for i in range(3)]].astype(np.float64)
    h = cells["elapsed"][:, None]
    ca = anchor.copy()
    ca[:, :3] += .5*accel*h**2
    ca[:, 3:] += accel*h
    write(args.output/f"constant_acceleration_{args.eval}.jsonl", cells, np.zeros_like(anchor), ca, np.full_like(anchor, POINT_SCALE), "constant_acceleration")
    # Gradient-boosted median residual per component, on a TRAIN subsample.
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(0)
    rows = rng.choice(len(train["horizon"]), min(args.gbm_rows, len(train["horizon"])), replace=False)
    def design(c, r=None):
        r = np.arange(len(c["horizon"])) if r is None else r
        return np.column_stack([c["X"][r], c["category"][r], c["activity"][r], c["horizon"][r]])
    Xt, Xe = design(train, rows), design(cells)
    categorical = np.zeros(Xt.shape[1], bool); categorical[-3:] = True
    gbm = anchor.copy()
    for j in range(6):
        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=.1, max_leaf_nodes=63,
                                              categorical_features=categorical, random_state=0)
        model.fit(Xt, residual[rows, j])
        gbm[:, j] = anchor[:, j]+model.predict(Xe)
        print(json.dumps({"gbm_component": j, "train_rows": len(rows)}), flush=True)
    write(args.output/f"gbm_residual_{args.eval}.jsonl", cells, np.zeros_like(anchor), gbm, np.full_like(anchor, POINT_SCALE), "gbm_residual")


if __name__ == "__main__":
    main()
