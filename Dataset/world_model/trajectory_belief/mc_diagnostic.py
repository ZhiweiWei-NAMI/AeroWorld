"""Validation-only Monte Carlo convergence diagnostic for the frozen empirical q."""
from __future__ import annotations

import argparse
from .experiment import OUTPUT_ROOT
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .contracts import WindowDataset
from .experiment import scores, write_json
from .rules import evaluate_pair, first_opposite_distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=OUTPUT_ROOT / "model")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "evaluation/monte_carlo")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((args.run_dir / "config.json").read_text())
    data = WindowDataset.load(config["data_dir"])
    validation = data.subset(data.rows.split.eq("validation").to_numpy())
    sampler = joblib.load(args.run_dir / "sampler.joblib")
    prior = np.asarray(json.loads((args.run_dir / "prior.json").read_text())["probabilities"])
    prior = prior[validation.rows.current.to_numpy(dtype=int)]
    ks = (1, 8, 16, 64)
    args.output.mkdir(parents=True)
    records, timings = [], []
    for mode in ("joint", "independent"):
        start = time.perf_counter()
        draws = sampler.sample_dataset(validation, k=max(ks), seed=config["q_seed"], mode=mode)
        timings.append(dict(mode=mode, k=max(ks), seconds=time.perf_counter()-start,
                            array_bytes=draws.nbytes, operation="sample_max_K_once"))
        for k in ks:
            start = time.perf_counter()
            p = first_opposite_distribution(validation.rows.current.to_numpy(),
                evaluate_pair(draws[:, :k], validation.rule_parameters))
            timings.append(dict(mode=mode, k=k, seconds=time.perf_counter()-start,
                                operation="rule_and_first_flip_only"))
            for alpha in (0., .1, 1., 10.):
                probability = (k*p+alpha*prior)/(k+alpha)
                measured, _ = scores(validation.rows, probability)
                records.append(dict(mode=mode, k=k, alpha=alpha, **measured))
    pd.DataFrame(records).to_csv(args.output / "validation_mc_metrics.csv", index=False)
    write_json(args.output / "monte_carlo_summary.json", dict(
        purpose="Validation-only MC integration diagnostic; no head fitting or test evaluation",
        run_dir=str(args.run_dir.resolve()), seed=config["q_seed"], ks=ks,
        primary_alpha=1., diagnostic_alphas=[0., .1, 1., 10.], timings=timings,
        draw_policy="One max-K draw per mode; all smaller K use its prefix. These draws differ from the max-K16 training cache.",
        smoothing="(K*p_MC + alpha*p_train_prior(current))/(K+alpha); alpha0 is raw, including infinite NLL",
        theoretical_binwise_mc_rmse_upper_bound={str(k): .5/np.sqrt(k) for k in ks},
        theoretical_scope="IID draw integration error relative to fixed q, not predictive calibration or model accuracy",
        max_draw_bytes=len(validation)*max(ks)*10*2*6*8,
        model_or_alpha_selected=False, test_opened=False))
    print(pd.DataFrame(records)[["mode", "k", "alpha", "nll", "brier", "auprc", "zero_likelihood_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
