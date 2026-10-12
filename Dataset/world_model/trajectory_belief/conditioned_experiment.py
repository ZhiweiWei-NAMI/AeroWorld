"""Two prespecified nearest-prefix-q configurations, replicated on validation only."""
from __future__ import annotations

import argparse
from .experiment import OUTPUT_ROOT
from copy import deepcopy
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch

from Dataset.world_model.transition_prediction.data import LABEL_VIEW, TARGET_COLUMNS
from .conditioned_sampling import KinematicResidualSampler, NEIGHBOR_COUNT
from .contracts import WindowDataset
from .experiment import (
    _output_directory, _prediction_rows, fixed_fpr_threshold, infer, make_prior,
    scores, sensitivity_scores, smooth_rule_probabilities, split_dataset,
    supervision_weight_audit, train_one, training_summary, validate_probabilities, write_json,
)
from .model import analytical_ceiling
from .nll_diagnostic import decompose
from .rules import evaluate_pair, first_opposite_distribution


FIXED_SPECS = [dict(name="conditioned_joint_K8", mode="joint", k=8),
               dict(name="conditioned_independent_K16", mode="independent", k=16)]
FIXED_TRAINING = dict(specs=FIXED_SPECS, seeds=[0, 1, 2], q_seed=0, width=24,
                      learning_rate=.003, epochs=30, patience=5, batch_size=256,
                      weight_decay=1e-4, gradient_clip=5., max_cache_k=16,
                      min_support=2, expected_new_fits=6, parameter_count=10033)
BASELINE_MODELS = ("center_K1", "independent_K16", "joint_K8")
METRIC_NAMES = ("nll", "brier", "auprc", "recall_at_fixed_fpr", "realized_fpr", "rps",
                "zero_likelihood_rate")
PAIRED_COMPARISONS = (("conditioned_joint_K8", "joint_K8"),
                      ("conditioned_independent_K16", "independent_K16"),
                      ("conditioned_joint_K8", "center_K1"),
                      ("conditioned_joint_K8", "conditioned_independent_K16"),
                      ("conditioned_joint_K8", "GBT"),
                      ("conditioned_independent_K16", "GBT"))


def validate_protocol(protocol, *, expected_training=FIXED_TRAINING):
    """Require the preexisting nested protocol without additional search knobs."""
    if protocol["test_opened"] is not False or protocol["calibration"] != "none for all methods":
        raise ValueError("conditioned experiment requires closed test and no calibration")
    if protocol["experiment_limit"] != 2 or protocol["conditioning"]["neighbors"] != NEIGHBOR_COUNT:
        raise ValueError("conditioned experiment requires exactly two specifications and fixed nearest32")
    if (protocol["smoothing"]["primary_alpha"] != 1.
            or protocol["smoothing"]["further_alpha_search"] is not False):
        raise ValueError("conditioned experiment requires fixed alpha=1 and no alpha search")
    training = protocol["training"]
    for key, expected in expected_training.items():
        if training[key] != expected:
            raise ValueError(f"fixed conditioned training protocol differs: {key}")
    if training["specs"] != FIXED_SPECS or training["seeds"] != [0, 1, 2]:
        raise ValueError("exactly the prespecified two configurations and all three seeds are required")
    if protocol["additional_compute_budget_minutes"] <= 0 or protocol["additional_compute_budget_minutes"] > 30:
        raise ValueError("additional compute budget must be positive and at most thirty minutes")
    ceiling = analytical_ceiling(training["width"], training["batch_size"], training["max_cache_k"])
    if ceiling["parameters"] != training["parameter_count"]:
        raise ValueError("shared conditional head parameter count differs from protocol")
    return deepcopy(training)


def _require_unopened(source, replication):
    source_selection = json.loads((source / "selection.json").read_text())
    replication_receipt = json.loads((replication / "replication_complete.json").read_text())
    if (OUTPUT_ROOT / "evaluation/test_complete.json").exists():
        raise ValueError("conditioned comparison requires unopened-test source runs")
    if not source_selection["validation_only"] or source_selection["test_opened"]:
        raise ValueError("source must be a completed validation-only run")
    if replication_receipt["test_opened"] or replication_receipt["seed_selected"]:
        raise ValueError("baseline replication must preserve all seeds and closed test")
    return source_selection, replication_receipt


def _aligned_probability(saved, validation, *, model, seed=None):
    """Reuse a saved comparator only when every validation row and target agrees."""
    if not saved.split.eq("validation").all():
        raise ValueError("conditioned experiment accepts saved validation predictions only")
    selected = saved.loc[saved.model.eq(model)]
    if seed is not None:
        selected = selected.loc[selected.seed.eq(seed)]
    if (len(selected) != len(validation) or selected.row_id.duplicated().any()
            or set(selected.row_id) != set(validation.rows.row_id)):
        raise ValueError(f"saved comparator row population differs: {model}, seed={seed}")
    selected = selected.set_index("row_id").loc[validation.rows.row_id]
    for column in ("current", "split_group", "family_component", "full_copy_group",
                   "full_copy_representative", *TARGET_COLUMNS):
        if not np.array_equal(selected[column].to_numpy(), validation.rows[column].to_numpy()):
            raise ValueError(f"saved comparator metadata differs: {model}, {column}")
    probability = selected[[f"p{i}" for i in range(11)]].to_numpy(dtype=np.float64)
    validate_probabilities(probability)
    return probability


def _flatten_ids(value):
    if isinstance(value, str):
        return [value]
    return [row_id for child in value for row_id in _flatten_ids(child)]


def conditioned_cache(sampler, dataset, *, k=16, seed=0):
    """Draw maximum K once per mode; all fits use the same nested draw prefixes."""
    if k != 16 or seed != 0 or not dataset.rows.split.isin(["train", "validation"]).all():
        raise ValueError("conditioned cache requires maxK16/seed0 and development queries only")
    if dataset.rows.split.nunique() != 1:
        raise ValueError("conditioned cache requires one declared query split at a time")
    result, metadata = {}, []
    bank_components = {donor.family_component for donor in sampler._donors}
    bank_rows = {donor.row_id for donor in sampler._donors}
    for mode in ("joint", "independent"):
        begin = time.perf_counter()
        q = sampler.sample_dataset(dataset, k=k, seed=seed, mode=mode)
        if q.shape != (len(dataset), k, 10, 2, 6) or not np.isfinite(q).all():
            raise ValueError("conditioned sampler must return finite [N,16,10,2,6] draws")
        support = sampler.last_sample_metadata.copy()
        if len(support) != len(dataset) or support.row_id.tolist() != dataset.rows.row_id.tolist():
            raise ValueError("conditioned sampler support rows differ from query rows")
        for index, record in enumerate(support.to_dict("records")):
            component = int(dataset.rows.family_component.iloc[index])
            coarse, nearest = set(record["coarse_pool_components"]), set(record["nearest_pool_components"])
            if not nearest.issubset(coarse) or not coarse.issubset(bank_components):
                raise ValueError("conditioned candidate pool contains non-TRAIN components")
            if dataset.rows.split.iloc[index] == "train" and component in coarse:
                raise ValueError("conditioned training query includes its own component")
            if not set(record["nearest_donor_row_ids"]).issubset(bank_rows):
                raise ValueError("conditioned nearest pool contains non-TRAIN donor IDs")
            if not set(_flatten_ids(record["donor_row_ids"])).issubset(set(record["nearest_donor_row_ids"])):
                raise ValueError("conditioned draw contains a donor outside the nearest pool")
            if record["nearest_pool_count"] > NEIGHBOR_COUNT or record["nearest_pool_count"] < sampler.min_support:
                raise ValueError("conditioned nearest pool violates declared support bounds")
        support["split"] = dataset.rows.split.to_numpy()
        support["q_seed"] = seed
        support["cache_k"] = k
        support["sample_seconds"] = time.perf_counter() - begin
        metadata.append(support)
        result[mode] = q
    return result, pd.concat(metadata, ignore_index=True)


def metric_summary(measured):
    result = measured.groupby("model", sort=False)[list(METRIC_NAMES)].agg(["mean", "std"])
    result.columns = ["_".join(column) for column in result.columns]
    result["reported_replicates"] = measured.groupby("model", sort=False).size()
    return result.reset_index()


def paired_differences(measured):
    rows = []
    gbt = measured.loc[measured.model.eq("GBT")]
    if len(gbt) != 1 or not gbt.seed.isna().all() or gbt.origin.iloc[0] != "unchanged_saved_tuned_GBT":
        raise ValueError("GBT paired differences require one unchanged, non-replicated GBT score")
    for seed in (0, 1, 2):
        indexed = measured.loc[measured.seed.eq(seed)].set_index("model")
        if indexed.index.duplicated().any():
            raise ValueError("paired comparisons require one model record per training seed")
        for model, baseline in PAIRED_COMPARISONS:
            baseline_values = gbt.iloc[0] if baseline == "GBT" else indexed.loc[baseline]
            record = dict(seed=seed, comparison=f"{model}-minus-{baseline}", model=model, baseline=baseline,
                          baseline_training_randomness=baseline != "GBT")
            for metric in METRIC_NAMES:
                left, right = indexed.loc[model, metric], baseline_values[metric]
                record[metric] = float(left-right) if pd.notna(left) and pd.notna(right) else None
            record["relative_nll_improvement"] = float(1-indexed.loc[model, "nll"]/baseline_values["nll"])
            rows.append(record)
    return pd.DataFrame(rows)


def run_fixed_experiment(*, source_run_dir, baseline_replication_dir, data_dir, run_dir,
                         protocol_path, device="cpu", threads=4, expected_training=FIXED_TRAINING):
    """Six new fits only. expected_training is overrideable solely by synthetic tests."""
    if not 1 <= threads <= 4:
        raise ValueError("use one to four CPU threads")
    torch.set_num_threads(threads)
    source = Path(source_run_dir).resolve()
    replication = Path(baseline_replication_dir).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    training = validate_protocol(protocol, expected_training=expected_training)
    for key, path in (("source_run", source), ("baseline_replication", replication), ("data_dir", Path(data_dir).resolve())):
        if Path(protocol[key]).resolve() != path:
            raise ValueError(f"explicit run source differs from predeclared protocol: {key}")
    source_selection, baseline_receipt = _require_unopened(source, replication)
    source_config = json.loads((source / "config.json").read_text())
    if Path(source_config["data_dir"]).resolve() != Path(data_dir).resolve() or source_config["label_view"] != LABEL_VIEW:
        raise ValueError("source and conditioned experiment must use the identical v2 dataset")
    for key in ("width", "batch_size", "epochs", "patience", "weight_decay", "gradient_clip"):
        if source_config[key] != training[key]:
            raise ValueError(f"conditioned fit budget differs from fixed baseline: {key}")
    if source_config["q_seed"] != training["q_seed"] or source_config["target_fpr"] != .05:
        raise ValueError("conditioned comparisons require unchanged q seed and 5% negative-window FPR budget")
    if baseline_receipt["sampling_seed_fixed"] != training["q_seed"]:
        raise ValueError("baseline replication sampler seed differs")
    dataset = WindowDataset.load(data_dir)
    if dataset.report["support_split_revision"]["version"] != "support-stratified-development-v2":
        raise ValueError("conditioned experiment requires the fixed support-adjusted input split")
    train, validation = split_dataset(dataset, "train"), split_dataset(dataset, "validation")
    if set(train.rows.split_group) & set(validation.rows.split_group):
        raise ValueError("train and validation independent components overlap")
    prior, prior_logits, prior_counts = make_prior(train.rows)
    source_prior = json.loads((source / "prior.json").read_text())
    if not np.array_equal(np.asarray(source_prior["probabilities"]), prior) or not np.array_equal(
            np.asarray(source_prior["logits"]), prior_logits):
        raise ValueError("source prior differs from unchanged TRAIN-only hazard prior")
    baseline_predictions = pd.read_parquet(replication / "validation_predictions.parquet")
    source_predictions = pd.read_parquet(source / "validation_predictions.parquet")
    baseline_probability = {(name, seed): _aligned_probability(baseline_predictions, validation, model=name, seed=seed)
                            for name in BASELINE_MODELS for seed in (0, 1, 2)}
    gbt_probability = _aligned_probability(source_predictions, validation, model="GBT")
    output = _output_directory(run_dir, new=True)
    write_json(output / "protocol.json", protocol)
    config = dict(training, data_dir=str(Path(data_dir).resolve()), source_run=str(source),
                  baseline_replication=str(replication), target_fpr=source_config["target_fpr"],
                  label_view=LABEL_VIEW, test_locked=True, device=str(device),
                  hypothesis="nearest prefix kinematics may reduce q misspecification; narrower pools may lose modes",
                  model_inputs="unchanged ConditionalTrajectoryHead features and capacity; q replaced only",
                  learning_rates_searched=False, k_searched=False, neighbors_searched=False,
                  calibration="none", primary_alpha=1.)
    write_json(output / "config.json", config)
    write_json(output / "analytical_ceiling.json", dict(
        estimates=[analytical_ceiling(training["width"], training["batch_size"], spec["k"])
                   for spec in training["specs"]], measured=False, new_neural_fits=6,
        max_epochs_per_fit=training["epochs"], max_cache_k=16,
        max_cache_bytes_per_mode=(len(train)+len(validation))*16*10*2*6*8,
        conditioning_features=15, nearest_pool=NEIGHBOR_COUNT,
        conditioning_added_storage_bytes_per_donor_at_most=240,
        conditioning_query_work="O(15*eligible_donors+eligible_donors*log(eligible_donors)+120*K)",
        unchanged_shared_head_parameters=training["parameter_count"],
        additional_compute_budget_minutes=protocol["additional_compute_budget_minutes"]))
    write_json(output / "prior.json", dict(probabilities=prior.tolist(), logits=prior_logits.tolist(), **prior_counts))
    write_json(output / "training_supervision_weights.json", supervision_weight_audit(train.rows))
    start = time.perf_counter()
    sampler = KinematicResidualSampler(min_support=training["min_support"]).fit(train)
    if sampler.diagnostics["normalization_split"] != "train" or sampler.diagnostics["neighbor_count"] != NEIGHBOR_COUNT:
        raise ValueError("conditioned sampler normalization or neighbor contract differs")
    train_q, train_support = conditioned_cache(sampler, train)
    validation_q, validation_support = conditioned_cache(sampler, validation)
    pd.concat([train_support, validation_support], ignore_index=True).to_json(
        output / "sampler_support.jsonl", orient="records", lines=True)
    joblib.dump(sampler, output / "sampler.joblib")
    write_json(output / "sampler_diagnostics.json", sampler.diagnostics)
    records, predictions, curves, decompositions, checkpoints, latency = [], [], [], [], [], []
    sensitivity, operating_points = {}, {}

    def record(name, probability, *, seed=None, origin, directory=None):
        operating = fixed_fpr_threshold(validation.rows, probability, source_config["target_fpr"])
        measured, _ = scores(validation.rows, probability, operating["threshold"])
        metrics = dict(model=name, seed=seed, split="validation", origin=origin, **measured)
        prediction = _prediction_rows(validation, probability, name).assign(seed=seed, origin=origin)
        component = dict(model=name, seed=seed, **decompose(validation.rows, probability))
        report = sensitivity_scores(validation.rows, probability, operating["threshold"])
        key = name if seed is None else f"{name}_seed{seed}"
        records.append(metrics)
        predictions.append(prediction)
        decompositions.append(component)
        sensitivity[key] = report
        operating_points[key] = operating
        if directory is not None:
            write_json(directory / "validation_metrics.json", metrics)
            prediction.to_parquet(directory / "validation_predictions.parquet", index=False)
            write_json(directory / "validation_nll_components.json", component)
            write_json(directory / "validation_sensitivity.json", report)
            write_json(directory / "operating_point.json", operating)
        return operating, measured

    for name in BASELINE_MODELS:
        for seed in (0, 1, 2):
            record(name, baseline_probability[name, seed], seed=seed, origin="unchanged_saved_three_seed_replication")
    record("GBT", gbt_probability, origin="unchanged_saved_tuned_GBT")
    for spec in training["specs"]:
        name, mode, k = spec["name"], spec["mode"], spec["k"]
        raw = first_opposite_distribution(validation.rows.current.to_numpy(),
                                          evaluate_pair(validation_q[mode][:, :k], validation.rule_parameters))
        record("Rule_"+name, raw, origin="conditioned_q_raw_rule")
        smooth = smooth_rule_probabilities(raw, prior[validation.rows.current.to_numpy(dtype=int)], k, 1.)
        record("SmoothRule_"+name, smooth, origin="conditioned_q_fixed_alpha1_train_prior")
        for seed in training["seeds"]:
            if time.perf_counter()-start >= protocol["additional_compute_budget_minutes"]*60:
                raise TimeoutError("declared conditioned additional compute budget exhausted before next fit")
            local = dict(config, seed=seed)
            print(json.dumps(dict(training=name, seed=seed, k=k, learning_rate=training["learning_rate"],
                                  q_seed=training["q_seed"])), flush=True)
            model, learning_curve, value = train_one(train, validation, train_q[mode][:, :k],
                validation_q[mode][:, :k], prior_logits, local, training["learning_rate"], device)
            seed_dir = output / f"{name}_seed{seed}"
            seed_dir.mkdir()
            checkpoint_path = seed_dir / "checkpoint.pt"
            checkpoint = dict(state=model.state_dict(), width=training["width"], spec=spec, seed=seed,
                              learning_rate=training["learning_rate"], q_seed=training["q_seed"],
                              validation_nll=value, sampler_source=str(output / "sampler.joblib"),
                              protocol_source=str(protocol_path), **training_summary(learning_curve))
            torch.save(checkpoint, checkpoint_path)
            pd.DataFrame(learning_curve).to_csv(seed_dir / "learning_curves.csv", index=False)
            curves.extend(dict(row, model=name, seed=seed, learning_rate=training["learning_rate"])
                          for row in learning_curve)
            probability, timing = infer(model, validation, validation_q[mode][:, :k], local, device)
            latency.append(dict(model=name, seed=seed, split="validation", **timing))
            operating, measured = record(name, probability, seed=seed, origin="fixed_conditioned_fit", directory=seed_dir)
            checkpoints.append(dict(model=name, seed=seed, path=str(checkpoint_path), spec=spec,
                                    operating_point=operating, validation_nll=measured["nll"],
                                    **training_summary(learning_curve)))
    measured = pd.DataFrame(records)
    summary = metric_summary(measured)
    pairs = paired_differences(measured)
    paired_summary = pairs.groupby("comparison")[list(METRIC_NAMES)+["relative_nll_improvement"]].agg(["mean", "std"])
    paired_summary.columns = ["_".join(column) for column in paired_summary.columns]
    measured.to_csv(output / "validation_metrics.csv", index=False)
    summary.to_csv(output / "validation_summary.csv", index=False)
    pairs.to_csv(output / "validation_paired_differences.csv", index=False)
    paired_summary.reset_index().to_csv(output / "validation_paired_summary.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(output / "validation_predictions.parquet", index=False)
    pd.DataFrame(curves).to_csv(output / "learning_curves.csv", index=False)
    pd.DataFrame(decompositions).to_csv(output / "validation_nll_components.csv", index=False)
    pd.DataFrame(latency).to_csv(output / "latency.csv", index=False)
    write_json(output / "validation_sensitivity.json", sensitivity)
    pd.DataFrame([dict(model=name, **view) for name, report in sensitivity.items() for view in report["views"]]).to_csv(
        output / "validation_sensitivity_metrics.csv", index=False)
    write_json(output / "conditioned_complete.json", dict(
        protocol_source=str(protocol_path), source_run=str(source), baseline_replication=str(replication),
        data_dir=str(Path(data_dir).resolve()), checkpoints=checkpoints, new_neural_fits=len(checkpoints),
        specs=training["specs"], training_seeds=training["seeds"], sampler_seed=training["q_seed"],
        max_cache_k=16, nested_cache_used=True, reused_baseline_models=list(BASELINE_MODELS), reused_tuned_GBT=True,
        seed_selected=False, configuration_selected=False, comparison_only=True,
        sample_standard_deviation_ddof=1, calibration="none", primary_alpha=1.,
        alpha_search=False, learning_rate_search=False, k_search=False, neighbor_search=False,
        test_opened=False, split_changed=False, source_artifacts_changed=False,
        operating_points=operating_points, elapsed_seconds=time.perf_counter()-start,
        limitation="Training-seed dispersion uses one fixed conditioned sampler realization; "
                   "five validation components; nearest pools may lose modes"))
    print(summary.to_string(index=False), flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=OUTPUT_ROOT / "model")
    parser.add_argument("--baseline-replication-dir", type=Path, default=OUTPUT_ROOT / "validation_replication")
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_ROOT / "inputs")
    parser.add_argument("--run-dir", type=Path, default=OUTPUT_ROOT / "conditioned_replication")
    parser.add_argument("--protocol", type=Path, default=OUTPUT_ROOT / "conditioned_replication/protocol.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    run_fixed_experiment(source_run_dir=args.source_run_dir, baseline_replication_dir=args.baseline_replication_dir,
                         data_dir=args.data_dir, run_dir=args.run_dir, protocol_path=args.protocol,
                         device=args.device, threads=args.threads)


if __name__ == "__main__":
    main()
