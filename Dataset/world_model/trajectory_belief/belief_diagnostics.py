"""Frozen-q/head TRAIN/validation diagnostics; no fitting or test evaluation.

Run ``--phase state`` for physical support/coverage and streamed available-donor
oracles, or ``--phase integration`` for five sampler seeds and nested K=8/16/64.
All output is diagnostic. Model, K, sampler seed and thresholds are not selected.
"""
from __future__ import annotations

import argparse
from .experiment import OUTPUT_ROOT
import json
from pathlib import Path
import shlex
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch

from Dataset.world_model.transition_prediction.data import LABEL_VIEW
from .contracts import (FUTURE_STEPS, HISTORY_STEPS, RULE_PARAMETER_NAMES,
                        STEP_SECONDS, STEP_TICKS, TICK_SECONDS, WindowDataset)
from .experiment import (_output_directory, infer, scores, sensitivity_scores,
                         smooth_rule_probabilities, split_dataset, validate_probabilities, write_json)
from .model import ConditionalTrajectoryHead, analytical_ceiling
from .rules import evaluate_pair, first_opposite_distribution
from .sampling import _conditions, constant_velocity_center


COORDINATES = ("x", "y", "z", "vx", "vy", "vz")
INTEGRATION_METRICS = ("nll", "brier", "auprc", "recall_at_fixed_fpr", "realized_fpr", "rps")


def require_development_rows(rows):
    if not rows.split.isin(["train", "validation"]).all():
        raise ValueError("belief diagnostics accept TRAIN/validation rows only")


def load_development_data(directory):
    """Read split metadata and filtered development rows, never test row fields.

    NPZ compression requires decoding each stored array. Only development slots
    are retained or exposed to a dataset/consumer. Test labels/IDs are not read.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = dict(array_contract="trajectory-window-v1", label_view=LABEL_VIEW,
                    history_steps=HISTORY_STEPS, future_steps=FUTURE_STEPS,
                    step_ticks=STEP_TICKS, tick_seconds=TICK_SECONDS,
                    state_fields=list(COORDINATES), rule_parameter_names=list(RULE_PARAMETER_NAMES))
    for key, value in expected.items():
        if manifest[key] != value:
            raise ValueError(f"development diagnostic dataset contract differs: {key}")
    split = pd.read_parquet(directory / "rows.parquet", columns=["split"])
    indices = np.flatnonzero(split.split.isin(["train", "validation"]).to_numpy())
    rows = pd.read_parquet(directory / "rows.parquet", filters=[("split", "in", ["train", "validation"])])
    if len(rows) != len(indices):
        raise ValueError("filtered development row order/count differs from array slots")
    require_development_rows(rows)
    with np.load(directory / "windows.npz", allow_pickle=False) as arrays:
        values = {key: arrays[key][indices] for key in ("history", "future", "future_present", "rule_parameters")}
    return WindowDataset(rows=rows, **values)


def baseline_pool(sampler, dataset, index):
    """The unchanged coarse supported q pool, with complete train exclusion."""
    row = dataset.rows.iloc[index]
    return sampler._pool(_conditions(dataset.history[index:index+1])[0],
                         int(row.family_component), row.split, row.row_id)


def restricted_pool_oracle(dataset, sampler, *, pool_provider=baseline_pool, chunk_size=128):
    """Select one full donor block by xyz MSE; velocity uses that same block.

    Actual future is used only to score candidates after the admissible pool is
    fixed from history/metadata. The position error bounds joint sampled best
    position error; the selected velocity error is not a separate minimum.
    """
    require_development_rows(dataset.rows)
    if chunk_size < 1:
        raise ValueError("oracle chunk size must be positive")
    center = constant_velocity_center(dataset.history)
    complete = dataset.future_present.all(axis=(1, 2))
    records = []
    for index, row in dataset.rows.iterrows():
        pool, level, excluded = pool_provider(sampler, dataset, index)
        if not len(pool):
            raise ValueError("available-donor oracle has no admissible pool")
        if row.split == "train" and any(sampler._donors[int(d)].family_component == int(row.family_component) for d in pool):
            raise ValueError("TRAIN oracle pool includes the query's family component")
        record = dict(row_id=row.row_id, level=level, support_count=len(pool), excluded_component_donor_count=excluded,
                      support_component_count=len({sampler._donors[int(d)].family_component for d in pool}),
                      support_components=sorted({sampler._donors[int(d)].family_component for d in pool}),
                      oracle_donor_row_id=None, oracle_xyz_mse=np.nan, oracle_same_block_vxyz_mse=np.nan,
                      physical_complete=bool(complete[index]))
        if complete[index]:
            target_residual = dataset.future[index] - center[index]
            best, best_velocity, chosen = float("inf"), None, None
            for start in range(0, len(pool), chunk_size):
                donor_indices = pool[start:start+chunk_size]
                error = sampler._residuals[donor_indices] - target_residual[None]
                xyz_mse = np.square(error[..., :3]).mean(axis=(1, 2, 3))
                local = int(xyz_mse.argmin())
                if xyz_mse[local] < best:
                    best = float(xyz_mse[local])
                    best_velocity = float(np.square(error[local, ..., 3:]).mean())
                    chosen = int(donor_indices[local])
            record.update(oracle_donor_row_id=sampler._donors[chosen].row_id,
                          oracle_xyz_mse=best, oracle_same_block_vxyz_mse=best_velocity)
        records.append(record)
    return pd.DataFrame(records)


def marginal_intervals(draws, actual):
    """Empirical coordinate/time intervals; returned coverage is not joint coverage."""
    draws = np.asarray(draws, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if draws.ndim != 5 or actual.shape != (draws.shape[0], 10, 2, 6) or draws.shape[2:] != (10, 2, 6):
        raise ValueError("marginal intervals require [N,K,10,2,6] and complete [N,10,2,6]")
    if not np.isfinite(draws).all() or not np.isfinite(actual).all():
        raise ValueError("marginal interval population must have complete finite futures")
    lower, upper = np.quantile(draws, [.05, .95], axis=1)
    width = upper-lower
    covered = (actual >= lower) & (actual <= upper)
    return dict(coverage_by_coordinate=covered.mean(axis=1), width_by_coordinate=width.mean(axis=1),
                zero_width_by_coordinate=(width == 0).mean(axis=1),
                coverage=covered.mean(axis=(1, 2, 3)), width_xyz=width[..., :3].mean(axis=(1, 2, 3)),
                width_vxyz=width[..., 3:].mean(axis=(1, 2, 3)), zero_width_rate=(width == 0).mean(axis=(1, 2, 3)))


def _event_status(rows):
    return np.where(rows.event_observed, "event", np.where(rows.horizon_outcome_known, "non_event", "censored"))


def state_diagnostics(dataset, sampler, *, k=64, seed=0, pool_provider=baseline_pool, oracle_chunk=128):
    """Reusable state diagnostics; explicit pool_provider supports a new q class."""
    require_development_rows(dataset.rows)
    if k != 64 or seed != 0:
        raise ValueError("state diagnostics freeze q_seed=0 and K=64")
    oracle = restricted_pool_oracle(dataset, sampler, pool_provider=pool_provider, chunk_size=oracle_chunk)
    coarse_oracle = oracle if pool_provider is baseline_pool else restricted_pool_oracle(
        dataset, sampler, pool_provider=baseline_pool, chunk_size=oracle_chunk)
    center = constant_velocity_center(dataset.history)
    complete = dataset.future_present.all(axis=(1, 2))
    valid_indices = np.flatnonzero(complete)
    truth = dataset.future[complete]
    conditions = _conditions(dataset.history)
    acceleration = np.diff(dataset.history[..., 3:], axis=1) / STEP_SECONDS
    acceleration_rms = np.sqrt(np.square(acceleration).mean(axis=(1, 2, 3)))
    row_records, coordinate_records = [], []
    for mode in ("center", "joint", "independent"):
        draws = sampler.sample_dataset(dataset, k=1 if mode == "center" else k, seed=seed, mode=mode)
        if not np.isfinite(draws).all():
            raise ValueError("state diagnostic draws must be finite")
        sampled = draws[complete]
        error = sampled-truth[:, None]
        xyz_mse = np.square(error[..., :3]).mean(axis=(2, 3, 4))
        best_draw = xyz_mse.argmin(axis=1)
        chosen_error = error[np.arange(len(error)), best_draw]
        intervals = marginal_intervals(sampled, truth)
        mean_error = sampled.mean(axis=1)-truth
        cv_error = center[complete]-truth
        binary = evaluate_pair(draws, dataset.rule_parameters)
        pmf = first_opposite_distribution(dataset.rows.current.to_numpy(), binary)
        _, likelihood_detail = scores(dataset.rows, pmf)
        frame = dataset.rows[["row_id", "split", "family_component", "full_copy_group", "full_copy_representative",
                              "event_observed", "event_bin", "observed_bins", "likelihood_valid", "horizon_outcome_known"]].copy()
        frame["mode"], frame["q_seed"], frame["k"] = mode, seed, draws.shape[1]
        frame["physical_complete"] = complete
        frame["event_status"] = _event_status(dataset.rows)
        frame["prefix_acceleration_rms_mps2"] = acceleration_rms
        frame["first_moving"], frame["second_moving"], frame["radial_state"] = conditions.T
        frame["coarse_condition"] = [",".join(map(str, condition)) for condition in conditions]
        frame["pool_level"] = oracle.level.to_numpy()
        frame["pool_support_count"] = oracle.support_count.to_numpy()
        frame["pool_support_component_count"] = oracle.support_component_count.to_numpy()
        frame["excluded_component_donor_count"] = oracle.excluded_component_donor_count.to_numpy()
        frame["sampled_event_probability"] = pmf[:, :10].sum(axis=1)
        frame["raw_rule_nll"] = likelihood_detail["nll"]
        frame["raw_rule_zero_likelihood"] = likelihood_detail["likelihood"] == 0
        labels = np.where(dataset.rows.event_observed, dataset.rows.event_bin, 10).astype(int)
        support = pmf[np.arange(len(dataset)), labels]
        frame["observed_bin_support_mass"] = np.where(dataset.rows.horizon_outcome_known, support, np.nan)
        frame["observed_bin_supported"] = np.where(dataset.rows.horizon_outcome_known, support > 0, np.nan)
        for j in range(11):
            frame[f"rule_p{j}"] = pmf[:, j]
        values = dict(cv_xyz_mse=np.square(cv_error[..., :3]).mean(axis=(1, 2, 3)),
                      cv_vxyz_mse=np.square(cv_error[..., 3:]).mean(axis=(1, 2, 3)),
                      mean_xyz_mse=np.square(mean_error[..., :3]).mean(axis=(1, 2, 3)),
                      mean_vxyz_mse=np.square(mean_error[..., 3:]).mean(axis=(1, 2, 3)),
                      sampled_best_xyz_mse=xyz_mse.min(axis=1),
                      sampled_best_same_draw_vxyz_mse=np.square(chosen_error[..., 3:]).mean(axis=(1, 2, 3)),
                      marginal_coverage90=intervals["coverage"], interval_width_xyz_m=intervals["width_xyz"],
                      interval_width_vxyz_mps=intervals["width_vxyz"], interval_zero_width_rate=intervals["zero_width_rate"],
                      q_xyz_spread_variance=sampled[..., :3].var(axis=1).mean(axis=(1, 2, 3)),
                      q_vxyz_spread_variance=sampled[..., 3:].var(axis=1).mean(axis=(1, 2, 3)))
        for key, value in values.items():
            frame[key] = np.nan
            frame.loc[valid_indices, key] = value
        frame["oracle_xyz_mse"] = oracle.oracle_xyz_mse.to_numpy()
        frame["oracle_same_block_vxyz_mse"] = oracle.oracle_same_block_vxyz_mse.to_numpy()
        frame["coarse_oracle_xyz_mse"] = coarse_oracle.oracle_xyz_mse.to_numpy()
        frame["coarse_oracle_same_block_vxyz_mse"] = coarse_oracle.oracle_same_block_vxyz_mse.to_numpy()
        frame["coarse_pool_support_count"] = coarse_oracle.support_count.to_numpy()
        frame["coarse_pool_support_component_count"] = coarse_oracle.support_component_count.to_numpy()
        frame["oracle_position_bound_applies"] = mode == "joint"
        if mode == "joint" and not (frame.loc[complete, "oracle_xyz_mse"].to_numpy() <= xyz_mse.min(axis=1)+1e-9).all():
            raise ArithmeticError("allowed full-block oracle position error exceeds a joint sampled draw")
        if not (frame.loc[complete, "coarse_oracle_xyz_mse"].to_numpy() <= frame.loc[complete, "oracle_xyz_mse"].to_numpy()+1e-9).all():
            raise ArithmeticError("coarse available-pool position bound exceeds its restricted-pool bound")
        row_records.append(frame)
        for entity in range(2):
            for coordinate, name in enumerate(COORDINATES):
                coordinate_records.append(pd.DataFrame(dict(
                    row_id=dataset.rows.loc[complete, "row_id"].to_numpy(), split=dataset.rows.loc[complete, "split"].to_numpy(),
                    family_component=dataset.rows.loc[complete, "family_component"].to_numpy(),
                    event_status=_event_status(dataset.rows)[complete], coarse_condition=frame.loc[complete, "coarse_condition"].to_numpy(),
                    mode=mode, entity_role=entity, coordinate=name,
                    marginal_coverage90=intervals["coverage_by_coordinate"][:, entity, coordinate],
                    interval_width=intervals["width_by_coordinate"][:, entity, coordinate],
                    zero_width_rate=intervals["zero_width_by_coordinate"][:, entity, coordinate])))
    return pd.concat(row_records, ignore_index=True), pd.concat(coordinate_records, ignore_index=True), oracle


def _mean(value):
    value = np.asarray(value, dtype=float)
    value = value[~np.isnan(value)]
    return float(value.mean()) if len(value) else None


def state_group_summary(frame, columns):
    records = []
    for key, group in frame.groupby(columns, sort=True, dropna=False):
        key = (key,) if len(columns) == 1 else key
        record = dict(zip(columns, key))
        physical = group[group.physical_complete]
        record.update(windows=len(group), physical_windows=len(physical), omitted_physical_windows=int((~group.physical_complete).sum()),
                      event_windows=int(group.event_observed.sum()), partial_censored_windows=int((~group.horizon_outcome_known).sum()),
                      event_physical_windows=int(physical.event_observed.sum()), independent_components=int(group.family_component.nunique()))
        for field in ("cv_xyz_mse", "cv_vxyz_mse", "mean_xyz_mse", "mean_vxyz_mse", "sampled_best_xyz_mse",
                      "sampled_best_same_draw_vxyz_mse", "oracle_xyz_mse", "oracle_same_block_vxyz_mse"):
            mean = _mean(physical[field])
            record[field.replace("mse", "rmse")] = np.sqrt(mean) if mean is not None else None
        for field in ("coarse_oracle_xyz_mse", "coarse_oracle_same_block_vxyz_mse"):
            mean = _mean(physical[field])
            record[field.replace("mse", "rmse")] = np.sqrt(mean) if mean is not None else None
        for field in ("pool_support_count", "pool_support_component_count", "coarse_pool_support_count", "coarse_pool_support_component_count"):
            record[field+"_mean"] = _mean(group[field])
        for field in ("marginal_coverage90", "interval_width_xyz_m", "interval_width_vxyz_mps", "interval_zero_width_rate",
                      "q_xyz_spread_variance", "q_vxyz_spread_variance", "observed_bin_support_mass", "observed_bin_supported"):
            record[field] = _mean(group[field])
        supported = group[group.likelihood_valid]
        record["raw_rule_nll"] = _mean(supported.raw_rule_nll)
        record["raw_rule_zero_likelihood_rate"] = _mean(supported.raw_rule_zero_likelihood)
        record["known_bin_support_windows"] = int(group.horizon_outcome_known.sum())
        records.append(record)
    return pd.DataFrame(records)


def acceleration_correlations(frame):
    records = []
    for (split, mode, condition), group in frame.loc[frame.physical_complete].groupby(["split", "mode", "coarse_condition"]):
        for metric in ("cv_xyz_mse", "mean_xyz_mse", "q_xyz_spread_variance", "oracle_xyz_mse"):
            a, b = group.prefix_acceleration_rms_mps2.to_numpy(), group[metric].to_numpy()
            identifiable = len(a) > 1 and a.std() > 0 and b.std() > 0
            records.append(dict(split=split, mode=mode, coarse_condition=condition, metric=metric, windows=len(group),
                                correlation=float(np.corrcoef(a, b)[0, 1]) if identifiable else None,
                                status="measured_association_only" if identifiable else "constant_or_insufficient_population"))
    return pd.DataFrame(records)


def nested_prefixes(draws, ks):
    if list(ks) != [8, 16, 64] or draws.ndim != 5 or draws.shape[1] != 64:
        raise ValueError("integration requires one max64 array and nested K=8/16/64")
    return {k: draws[:, :k] for k in ks}


def probability_mc_variance(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[0] != 5 or probabilities.shape[-1] != 11:
        raise ValueError("MC variance requires five actually measured [N,11] distributions")
    for probability in probabilities:
        validate_probabilities(probability)
    mass_variance = probabilities.var(axis=0, ddof=1)
    event_variance = probabilities[..., :10].sum(axis=-1).var(axis=0, ddof=1)
    return dict(mass_variance=mass_variance, event_variance=event_variance,
                mass_variance_mean=float(mass_variance.mean()), event_variance_mean=float(event_variance.mean()),
                mass_sd_mean=float(np.sqrt(mass_variance).mean()), event_sd_mean=float(np.sqrt(event_variance).mean()),
                sampling_seeds=5, sample_variance_ddof=1)


def frozen_scores(rows, probability, operating_point):
    """No call to threshold fitting: apply the stored operating point verbatim."""
    if operating_point["false_positive_unit"] != "negative_window":
        raise ValueError("diagnostic operating point must use negative-window false positives")
    return scores(rows, probability, operating_point["threshold"])[0]


def _require_protocol(protocol, config, receipt, replication):
    if protocol["test_opened"] or receipt["test_opened"] or replication["test_opened"] or not receipt["validation_only"]:
        raise ValueError("belief diagnostics require a closed test and validation-only source")
    if Path(protocol["data_dir"]).resolve() != Path(config["data_dir"]).resolve():
        raise ValueError("phase-two source dataset differs from frozen protocol")
    if Path(replication["source_run"]).resolve() != Path(protocol["source_run"]).resolve():
        raise ValueError("replication source differs from frozen phase-two source")
    integration = protocol["integration"]
    if integration["training_seeds"] != [0, 1, 2] or integration["sampler_seeds"] != [0, 1, 2, 3, 4] or integration["ks"] != [8, 16, 64]:
        raise ValueError("integration seeds and K differ from the predeclared contract")
    if integration["fit_performed"] or integration["model_or_sampler_seed_selection"] or protocol["smoothing"]["primary_alpha"] != 1.:
        raise ValueError("phase-two diagnostics forbid fitting, selection and alpha changes")
    if protocol["belief_diagnostics"]["splits"] != ["train", "validation"]:
        raise ValueError("physical diagnostics must use train/validation only")


def run_state(args, output, dataset, sampler, protocol, *, pool_provider=baseline_pool):
    begin = time.perf_counter()
    all_rows, all_coordinates, all_oracles = [], [], []
    for split in ("train", "validation"):
        local = split_dataset(dataset, split)
        rows, coordinate, oracle = state_diagnostics(local, sampler, pool_provider=pool_provider, oracle_chunk=args.oracle_chunk)
        rows["sampler_kind"], coordinate["sampler_kind"], oracle["sampler_kind"] = args.sampler_kind, args.sampler_kind, args.sampler_kind
        all_rows.append(rows)
        all_coordinates.append(coordinate)
        all_oracles.append(oracle.assign(split=split))
        print(json.dumps(dict(state_diagnostic_split=split, windows=len(local), complete=int(local.future_present.all(axis=(1, 2)).sum()),
                              elapsed_seconds=time.perf_counter()-begin)), flush=True)
    rows = pd.concat(all_rows, ignore_index=True)
    coordinates = pd.concat(all_coordinates, ignore_index=True)
    rows.to_parquet(output / "state_window_diagnostics.parquet", index=False)
    pd.concat(all_oracles, ignore_index=True).to_csv(output / "available_donor_oracle.csv", index=False)
    state_group_summary(rows, ["split", "mode"]).to_csv(output / "state_summary.csv", index=False)
    state_group_summary(rows, ["split", "mode", "event_status", "coarse_condition"]).to_csv(output / "state_conditions.csv", index=False)
    state_group_summary(rows, ["split", "mode", "family_component", "event_status"]).to_csv(output / "state_components.csv", index=False)
    state_group_summary(rows, ["split", "mode", "event_bin"]).to_csv(output / "first_flip_bin_support.csv", index=False)
    for filename, groups in (("marginal_coordinate_summary.csv", ["split", "mode", "entity_role", "coordinate"]),
                             ("marginal_coordinate_conditions.csv", ["split", "mode", "event_status", "coarse_condition", "entity_role", "coordinate"]),
                             ("marginal_coordinate_components.csv", ["split", "mode", "family_component", "entity_role", "coordinate"])):
        coordinates.groupby(groups, sort=True).agg(windows=("row_id", "size"),
            marginal_coverage90=("marginal_coverage90", "mean"), interval_width=("interval_width", "mean"),
            zero_width_rate=("zero_width_rate", "mean")).reset_index().to_csv(output / filename, index=False)
    acceleration_correlations(rows).to_csv(output / "prefix_acceleration_associations.csv", index=False)


def run_integration(args, output, dataset, sampler, protocol, config, receipt, replication):
    validation = split_dataset(dataset, "validation")
    source = Path(protocol["source_run"])
    replication_dir = Path(protocol["baseline_replication"])
    checkpoints = {(r["model"], r["seed"]): r for r in replication["checkpoints"]}
    required = {(name, seed) for name in ("center_K1", "joint_K8", "independent_K16") for seed in (0, 1, 2)}
    if set(checkpoints) != required:
        raise ValueError("integration requires exactly nine frozen three-seed checkpoints")
    prior = np.asarray(json.loads((source / "prior.json").read_text())["probabilities"])
    current_prior = prior[validation.rows.current.to_numpy(dtype=int)]
    saved_probabilities = pd.read_parquet(replication_dir / "validation_predictions.parquet")
    records, sensitivity_records, centers, timing, measured_probability = [], [], [], [], {}
    for seed in (0, 1, 2):
        saved = saved_probabilities.loc[saved_probabilities.model.eq("center_K1") & saved_probabilities.seed.eq(seed)].set_index("row_id")
        probability = saved.loc[validation.rows.row_id, [f"p{i}" for i in range(11)]].to_numpy()
        centers.append(dict(model="center_K1", training_seed=seed, source="unchanged_saved_validation_predictions",
                            **frozen_scores(validation.rows, probability, checkpoints[("center_K1", seed)]["operating_point"])))
    for mode, name in (("joint", "joint_K8"), ("independent", "independent_K16")):
        models = {}
        for seed in (0, 1, 2):
            saved = torch.load(checkpoints[(name, seed)]["path"], map_location=args.device, weights_only=False)
            if saved["seed"] != seed or saved["width"] != config["width"] or saved["spec"]["name"] != name:
                raise ValueError("frozen checkpoint does not match model/seed/capacity")
            model = ConditionalTrajectoryHead(saved["width"]).to(args.device)
            model.load_state_dict(saved["state"])
            model.eval()
            models[seed] = model
        for q_seed in protocol["integration"]["sampler_seeds"]:
            draws = sampler.sample_dataset(validation, k=64, seed=q_seed, mode=mode)
            if draws.nbytes > protocol["integration"]["max_validation_draw_bytes"]:
                raise MemoryError("validation draw array exceeds declared resource ceiling")
            for k, sampled in nested_prefixes(draws, protocol["integration"]["ks"]).items():
                raw = first_opposite_distribution(validation.rows.current.to_numpy(), evaluate_pair(sampled, validation.rule_parameters))
                smooth = smooth_rule_probabilities(raw, current_prior, k, 1.)
                for baseline_name, probability in (("Rule_"+name, raw), ("SmoothRule_"+name, smooth)):
                    operating = receipt["selection"][baseline_name]["operating_point"]
                    record = dict(model=baseline_name, mode=mode, training_seed=-1, q_seed=q_seed, k=k,
                                  **frozen_scores(validation.rows, probability, operating))
                    records.append(record)
                    report = sensitivity_scores(validation.rows, probability, operating["threshold"])
                    sensitivity_records.extend(dict(model=baseline_name, mode=mode, training_seed=-1, q_seed=q_seed, k=k, **view)
                                               for view in report["views"])
                    measured_probability.setdefault((baseline_name, -1, k), []).append(probability)
                for seed, model in models.items():
                    probability, elapsed = infer(model, validation, sampled, config, args.device)
                    operating = checkpoints[(name, seed)]["operating_point"]
                    records.append(dict(model=name, mode=mode, training_seed=seed, q_seed=q_seed, k=k,
                                        **frozen_scores(validation.rows, probability, operating)))
                    report = sensitivity_scores(validation.rows, probability, operating["threshold"])
                    sensitivity_records.extend(dict(model=name, mode=mode, training_seed=seed, q_seed=q_seed, k=k, **view)
                                               for view in report["views"])
                    timing.append(dict(model=name, training_seed=seed, q_seed=q_seed, **elapsed))
                    measured_probability.setdefault((name, seed, k), []).append(probability)
            print(json.dumps(dict(integration_mode=mode, q_seed=q_seed, completed_nested_ks=[8, 16, 64])), flush=True)
        del models
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / "integration_metrics.csv", index=False)
    pd.DataFrame(sensitivity_records).to_csv(output / "integration_sensitivity_metrics.csv", index=False)
    pd.DataFrame(centers).to_csv(output / "center_reference_metrics.csv", index=False)
    pd.DataFrame(timing).to_csv(output / "integration_latency.csv", index=False)
    summaries, variances, row_variances = [], [], []
    for (name, seed, k), values in measured_probability.items():
        value = probability_mc_variance(values)
        variances.append(dict(model=name, training_seed=seed, k=k,
                              **{key: v for key, v in value.items() if key not in ("mass_variance", "event_variance")},
                              theoretical_iid_bin_mc_rmse_bound=1/(2*np.sqrt(k))))
        frame = pd.DataFrame(dict(row_id=validation.rows.row_id, family_component=validation.rows.family_component,
                                  model=name, training_seed=seed, k=k, event_probability_mc_variance=value["event_variance"]))
        for bin_index in range(11):
            frame[f"p{bin_index}_mc_variance"] = value["mass_variance"][:, bin_index]
        row_variances.append(frame)
        group = metrics.loc[metrics.model.eq(name) & metrics.training_seed.eq(seed) & metrics.k.eq(k)]
        summary = dict(model=name, training_seed=seed, k=k, q_seeds=5)
        for field in INTEGRATION_METRICS:
            v = group[field].to_numpy(dtype=float)
            summary[field+"_qseed_mean"] = float(v.mean()) if not np.isnan(v).all() else None
            summary[field+"_qseed_std"] = float(v.std(ddof=1)) if np.isfinite(v).all() else None
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(output / "integration_qseed_summary.csv", index=False)
    pd.DataFrame(variances).to_csv(output / "probability_mc_variance.csv", index=False)
    pd.concat(row_variances, ignore_index=True).to_parquet(output / "probability_mc_variance_by_window.parquet", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("state", "integration"), required=True)
    parser.add_argument("--protocol", type=Path, default=OUTPUT_ROOT / "frozen_integration/protocol.json")
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--oracle-chunk", type=int, default=128)
    parser.add_argument("--sampler-kind", choices=("coarse", "conditioned32"), default="coarse")
    args = parser.parse_args(argv)
    if args.run_dir is None:
        directory = "frozen_integration" if args.phase == "integration" else "evaluation/belief_conditioned" if args.sampler_kind == "conditioned32" else "evaluation/belief_coarse"
        args.run_dir = str(OUTPUT_ROOT / directory)
    if args.threads < 1 or args.threads > 4 or args.oracle_chunk < 1:
        raise ValueError("diagnostics require one to four CPU threads and a positive oracle chunk")
    torch.set_num_threads(args.threads)
    protocol = json.loads(args.protocol.read_text())
    source, replication_dir = Path(protocol["source_run"]), Path(protocol["baseline_replication"])
    config = json.loads((source / "config.json").read_text())
    receipt = json.loads((source / "selection.json").read_text())
    replication = json.loads((replication_dir / "replication_complete.json").read_text())
    if (OUTPUT_ROOT / "evaluation/test_complete.json").exists():
        raise ValueError("phase-two diagnostics require the source test to remain unopened")
    _require_protocol(protocol, config, receipt, replication)
    if args.phase == "integration" and (args.sampler_kind != "coarse" or args.device != config["device"]):
        raise ValueError("frozen-head integration uses the original coarse sampler and original training device")
    output = _output_directory(args.run_dir, new=True)
    write_json(output / "protocol.json", protocol)
    dataset = load_development_data(protocol["data_dir"])
    sampler = joblib.load(source / "sampler.joblib")
    pool_provider = baseline_pool
    if args.sampler_kind == "conditioned32":
        from .conditioned_sampling import KinematicResidualSampler
        if protocol["conditioning"]["neighbors"] != 32:
            raise ValueError("conditioned state diagnostics freeze exactly 32 neighbors")
        conditioned = KinematicResidualSampler(min_support=config["min_support"]).fit(split_dataset(dataset, "train"))
        if [d.row_id for d in conditioned._donors] != [d.row_id for d in sampler._donors] or not np.array_equal(conditioned._residuals, sampler._residuals):
            raise ValueError("conditioned diagnostic residual bank differs from frozen TRAIN coarse bank")
        sampler = conditioned
        pool_provider = lambda sampler, dataset, index: sampler.query_pool(dataset, index)
        joblib.dump(sampler, output / "conditioned_sampler.joblib")
    write_json(output / "sampler_diagnostics.json", sampler.diagnostics)
    write_json(output / "resource_estimates.json", dict(measured=False,
        max_validation_draw_bytes=protocol["integration"]["max_validation_draw_bytes"],
        max_training_draw_bytes=protocol["integration"]["max_training_draw_bytes"],
        oracle_chunk_difference_bytes=args.oracle_chunk*10*2*6*8,
        oracle_upper_scalar_differences=int(dataset.future_present.all(axis=(1, 2)).sum())*len(sampler._donors)*10*2*6,
        neural_batch_estimate=analytical_ceiling(config["width"], config["batch_size"], 64),
        assumptions="draw arrays float64; oracle scans finite admissible full donor pools in chunks; "
                    "GPU framework workspace and Python donor metadata excluded"))
    if args.phase == "state":
        run_state(args, output, dataset, sampler, protocol, pool_provider=pool_provider)
    else:
        run_integration(args, output, dataset, sampler, protocol, config, receipt, replication)


if __name__ == "__main__":
    main()
