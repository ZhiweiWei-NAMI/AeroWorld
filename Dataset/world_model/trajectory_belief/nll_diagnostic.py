"""Decompose saved validation likelihood without fitting or selecting anything."""
from __future__ import annotations

import argparse
from .experiment import OUTPUT_ROOT
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import scores, write_json


def decompose(rows, probability):
    measured, _ = scores(rows, probability)
    p = np.asarray(probability, dtype=np.float64)
    event = rows.event_observed.to_numpy(dtype=bool)
    known = rows.horizon_outcome_known.to_numpy(dtype=bool)
    valid = rows.likelihood_valid.to_numpy(dtype=bool)
    partial = valid & ~known
    event_mass = p[:, :10].sum(axis=1)
    selected_bin = p[np.arange(len(rows)), rows.event_bin.to_numpy(dtype=int).clip(0)]
    with np.errstate(divide="ignore", invalid="ignore"):
        binary = -np.log(np.where(event, event_mass, p[:, 10]))
        conditional = -np.log(selected_bin[event] / event_mass[event])
        known_nll = -np.log(np.where(event, selected_bin, p[:, 10])[known]).mean()
        tail = (p * (np.arange(11)[None] >= rows.observed_bins.to_numpy()[:, None])).sum(axis=1)
        partial_nll = -np.log(tail[partial]).mean() if partial.any() else 0.
    prevalence = event[known].mean()
    time_nll = conditional.mean() if event.any() else 0.
    weighted_time = prevalence * time_nll
    reconstruction = known.sum()/valid.sum()*(binary[known].mean()+weighted_time)
    reconstruction += partial.sum()/valid.sum()*partial_nll
    finite = np.isfinite(measured["nll"])
    if finite and not np.isclose(measured["nll"], reconstruction, rtol=0, atol=1e-10):
        raise ValueError("binary/time/censor decomposition does not reproduce NLL")
    return dict(likelihood_windows=int(valid.sum()), known_windows=int(known.sum()),
                event_windows=int(event.sum()), partial_windows=int(partial.sum()),
                event_prevalence_known=float(prevalence), binary_nll=float(binary[known].mean()),
                conditional_time_nll=float(time_nll), conditional_time_weighted=float(weighted_time),
                known_11class_nll=float(known_nll), partial_tail_nll=float(partial_nll),
                overall_censor_nll=measured["nll"], reconstructed_nll=float(reconstruction),
                decomposition_status="exact_finite" if finite else "raw_zero_mass_infinite_or_undefined")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=OUTPUT_ROOT / "model")
    args = parser.parse_args()
    output = args.run_dir / "validation_nll_components.csv"
    receipt = args.run_dir / "validation_nll_components.json"
    if output.exists() or receipt.exists():
        raise FileExistsError("validation NLL diagnostic already exists")
    rows = pd.read_parquet(args.run_dir / "validation_predictions.parquet")
    if not rows.split.eq("validation").all():
        raise ValueError("diagnostic accepts validation predictions only")
    records = [dict(model=name, **decompose(group, group[[f"p{i}" for i in range(11)]].to_numpy()))
               for name, group in rows.groupby("model", sort=False)]
    pd.DataFrame(records).to_csv(output, index=False)
    write_json(receipt, dict(
        identity="overall = n_known/n_lik*(binary_NLL + event_prevalence_known*conditional_time_NLL) "
                 "+ n_partial/n_lik*partial_tail_NLL",
        fit_performed=False, model_selected=False, test_opened=False, records=records,
        zero_mass_policy="Raw rule infinities and undefined conditional terms remain explicit; no clipping"))
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
