"""Fixed-seed validation replication; no new hyperparameter or test selection."""
from __future__ import annotations

import argparse
from .experiment import OUTPUT_ROOT
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .contracts import WindowDataset
from .experiment import (
    _output_directory, _prediction_rows, fixed_fpr_threshold, infer, sample_cache,
    scores, sensitivity_scores, split_dataset, train_one, training_summary, write_json,
)
from .nll_diagnostic import decompose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=OUTPUT_ROOT / "model")
    parser.add_argument("--protocol", type=Path, default=OUTPUT_ROOT / "validation_replication/protocol.json")
    parser.add_argument("--run-dir", type=Path, default=OUTPUT_ROOT / "validation_replication")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source = args.source_run_dir
    protocol = json.loads(args.protocol.read_text())
    config = json.loads((source / "config.json").read_text())
    selection = json.loads((source / "selection.json").read_text())
    if (OUTPUT_ROOT / "evaluation/test_complete.json").exists() or not selection["validation_only"]:
        raise ValueError("replication requires an unopened-test validation run")
    for key in ("width", "batch_size", "epochs", "patience", "weight_decay", "gradient_clip"):
        if protocol[key] != config[key]:
            raise ValueError(f"replication protocol differs from selected configuration: {key}")
    if protocol["sampler_seed"] != config["q_seed"] or protocol["training_seeds"] != [0, 1, 2]:
        raise ValueError("fixed replication requires seeds 0/1/2 and unchanged sampler seed")
    output = _output_directory(args.run_dir, new=True)
    write_json(output / "protocol.json", protocol)
    data = WindowDataset.load(config["data_dir"])
    train, validation = split_dataset(data, "train"), split_dataset(data, "validation")
    sampler = joblib.load(source / "sampler.joblib")
    train_q, _ = sample_cache(sampler, train, config)
    validation_q, _ = sample_cache(sampler, validation, config)
    prior = json.loads((source / "prior.json").read_text())
    prior_logits = np.asarray(prior["logits"])
    saved_predictions = pd.read_parquet(source / "validation_predictions.parquet")
    saved_curves = pd.read_csv(source / "learning_curves.csv")
    records, predictions, curve_records, checkpoint_records, decomposition = [], [], [], [], []
    sensitivity = {}
    for spec in protocol["selected_specs"]:
        name, mode, k = spec["name"], spec["mode"], spec["k"]
        saved = torch.load(source / (name + ".pt"), map_location="cpu", weights_only=False)
        if saved["seed"] != 0 or saved["learning_rate"] != protocol["learning_rate"] or saved["spec"] != spec:
            raise ValueError(f"seed-zero checkpoint differs from the frozen replication protocol: {name}")
        for seed in protocol["training_seeds"]:
            local_config = dict(config, seed=seed)
            if seed == 0:
                prior_rows = saved_predictions.loc[saved_predictions.model.eq(name)].set_index("row_id")
                prior_rows = prior_rows.loc[validation.rows.row_id]
                probability = prior_rows[[f"p{i}" for i in range(11)]].to_numpy()
                curves = saved_curves.loc[saved_curves.model.eq(name) &
                    saved_curves.learning_rate.eq(protocol["learning_rate"])].to_dict("records")
                checkpoint_path = source / (name + ".pt")
            else:
                print(json.dumps(dict(replication_model=name, seed=seed, sampler_seed=config["q_seed"])), flush=True)
                model, curves, value = train_one(train, validation, train_q[mode][:, :k],
                    validation_q[mode][:, :k], prior_logits, local_config, protocol["learning_rate"], args.device)
                checkpoint_path = output / f"{name}_seed{seed}.pt"
                torch.save(dict(state=model.state_dict(), width=config["width"], spec=spec,
                    seed=seed, learning_rate=protocol["learning_rate"], validation_nll=value,
                    **training_summary(curves)), checkpoint_path)
                probability, _ = infer(model, validation, validation_q[mode][:, :k], config, args.device)
            operating = fixed_fpr_threshold(validation.rows, probability, config["target_fpr"])
            measured, _ = scores(validation.rows, probability, operating["threshold"])
            records.append(dict(model=name, seed=seed, **measured))
            predictions.append(_prediction_rows(validation, probability, name).assign(seed=seed))
            curve_records.extend(dict(r, model=name, seed=seed) for r in curves)
            checkpoint_records.append(dict(model=name, seed=seed, path=str(checkpoint_path.resolve()),
                source="reused_seed_zero" if seed == 0 else "replication_fit", operating_point=operating,
                **training_summary(curves)))
            sensitivity[f"{name}_seed{seed}"] = sensitivity_scores(validation.rows, probability, operating["threshold"])
            decomposition.append(dict(model=name, seed=seed, **decompose(validation.rows, probability)))
    measured = pd.DataFrame(records)
    metric_names = ["nll", "brier", "auprc", "recall_at_fixed_fpr", "realized_fpr", "rps"]
    summary = measured.groupby("model")[metric_names].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "validation_summary.csv", index=False)
    measured.to_csv(output / "validation_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(output / "validation_predictions.parquet", index=False)
    pd.DataFrame(curve_records).to_csv(output / "learning_curves.csv", index=False)
    pd.DataFrame(decomposition).to_csv(output / "validation_nll_components.csv", index=False)
    write_json(output / "validation_sensitivity.json", sensitivity)
    pairs = []
    for seed in protocol["training_seeds"]:
        indexed = measured.loc[measured.seed.eq(seed)].set_index("model")
        for baseline in ("center_K1", "independent_K16"):
            pairs.append(dict(seed=seed, comparison="joint_K8-minus-"+baseline,
                **{key: float(indexed.loc["joint_K8", key]-indexed.loc[baseline, key]) for key in metric_names},
                relative_nll_improvement=float(1-indexed.loc["joint_K8", "nll"]/indexed.loc[baseline, "nll"])))
    pd.DataFrame(pairs).to_csv(output / "validation_paired_differences.csv", index=False)
    selected = str(summary.nll_mean.idxmin())
    write_json(output / "replication_complete.json", dict(
        protocol_source=str(args.protocol.resolve()), source_run=str(source.resolve()),
        checkpoints=checkpoint_records, selected_fixed_configuration=selected,
        selection_rule="minimum mean validation window NLL over all three fixed training seeds",
        seed_selected=False, sample_standard_deviation_ddof=1, test_opened=False,
        new_neural_fits=6, reused_seed_zero_models=3, sampling_seed_fixed=config["q_seed"],
        calibration="none", hyperparameters_changed=False,
        limitation="Training-seed dispersion conditional on one fixed sampler realization and five validation components"))
    print(summary.to_string(), flush=True)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
