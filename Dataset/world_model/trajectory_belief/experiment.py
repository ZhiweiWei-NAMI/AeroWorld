"""Validation-only development and one separately invoked locked test evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import shlex
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from Dataset.world_model.transition_prediction.data import (
    LABEL_VIEW, TARGET_COLUMNS, person_period_targets, validate_targets,
)
from .contracts import WindowDataset
from .data import load_window_dataset
from .model import ConditionalTrajectoryHead, analytical_ceiling, censored_nll
from .rules import evaluate_pair, first_opposite_distribution
from .sampling import EmpiricalResidualSampler


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "Dataset/world_model/runs/trajectory_belief"
PREDICATE = "safety.aircraft_pair_below_separation"
SMOOTH_RULE_PRIMARY_ALPHA = 1.0
SMOOTH_RULE_ALPHAS = (0.1, 1.0, 10.0)


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _output_directory(path, *, new: bool) -> Path:
    output = Path(path).resolve()
    if output == OUTPUT_ROOT or OUTPUT_ROOT not in output.parents:
        raise ValueError(f"run directory must be a new child of {OUTPUT_ROOT}")
    if new and output.exists():
        raise FileExistsError(f"refusing to overwrite an existing experiment: {output}")
    if new:
        output.mkdir(parents=True)
    elif not output.is_dir():
        raise FileNotFoundError(output)
    return output


def split_dataset(dataset: WindowDataset, split: str) -> WindowDataset:
    indices = np.flatnonzero(dataset.rows.split.eq(split).to_numpy())
    if not len(indices):
        raise ValueError(f"empty {split} split")
    result = dataset.subset(indices)
    validate_targets(result.rows)
    return result


def make_prior(rows: pd.DataFrame, pseudo_count: float = 0.5):
    """Train-only empirical person-period hazards, with declared fixed regularization."""
    validate_targets(rows)
    events = np.full((2, 10), pseudo_count, dtype=np.float64)
    risk = np.full((2, 10), 2 * pseudo_count, dtype=np.float64)
    person, period, target = person_period_targets(rows)
    current = rows.current.to_numpy(dtype=np.int64)
    np.add.at(events, (current[person], period), target)
    np.add.at(risk, (current[person], period), 1)
    hazard = events / risk
    return hazard_probabilities(hazard), np.log(hazard) - np.log1p(-hazard), dict(
        pseudo_event_count=pseudo_count, pseudo_survival_count=pseudo_count,
        observed_events=(events - pseudo_count).tolist(), observed_risk=(risk - 2 * pseudo_count).tolist())


def hazard_probabilities(hazard: np.ndarray) -> np.ndarray:
    hazard = np.asarray(hazard, dtype=np.float64)
    if hazard.ndim != 2 or hazard.shape[1] != 10 or not np.isfinite(hazard).all() or ((hazard < 0) | (hazard > 1)).any():
        raise ValueError("hazards must be finite [N,10] values in [0,1]")
    survival = np.cumprod(1 - hazard, axis=1)
    before = np.c_[np.ones(len(hazard)), survival[:, :-1]]
    return np.c_[hazard * before, survival[:, -1]]


def validate_probabilities(probability: np.ndarray):
    p = np.asarray(probability)
    if p.ndim != 2 or p.shape[1] != 11 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probability must be finite, nonnegative [N,11]")
    if not np.allclose(p.sum(-1), 1, atol=3e-6, rtol=0):
        raise ValueError("first-event distribution must sum to one")
    if (np.diff(p[:, :10].cumsum(-1), axis=-1) < -1e-8).any():
        raise ValueError("event CDF must be monotone")


def smooth_rule_probabilities(raw_probability, prior_probability, k: int, alpha: float):
    """A declared train-prior pseudo-trajectory, with no likelihood clipping."""
    raw = np.asarray(raw_probability, dtype=np.float64)
    prior = np.asarray(prior_probability, dtype=np.float64)
    validate_probabilities(raw)
    validate_probabilities(prior)
    if raw.shape != prior.shape or not (prior > 0).all():
        raise ValueError("rule smoothing requires a matching train prior with full eleven-class support")
    if not isinstance(k, (int, np.integer)) or k < 1 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("rule smoothing requires K>=1 and finite alpha>0")
    probability = (k * raw + alpha * prior) / (k + alpha)
    validate_probabilities(probability)
    if not (probability > 0).all():
        raise FloatingPointError("smoothed rule probabilities must have full support")
    return probability


def baseline_candidates(selection):
    """All rule baselines compete; only learned joint heads are excluded."""
    names = []
    for name, entry in selection.items():
        kind = entry["kind"]
        if kind in ("prior", "gbt", "rule_mc"):
            eligible = True
        elif kind == "smoothed_rule_mc":
            eligible = entry["alpha"] == SMOOTH_RULE_PRIMARY_ALPHA
        elif kind == "conditional_head":
            eligible = entry["mode"] in ("center", "independent")
        else:
            raise ValueError(f"unknown baseline classification: {kind}")
        if eligible:
            names.append(name)
    return names


def fixed_fpr_threshold(rows: pd.DataFrame, probability: np.ndarray, target_fpr: float):
    """Choose the lowest validation threshold satisfying the empirical FPR budget."""
    validate_targets(rows)
    validate_probabilities(probability)
    known = rows.horizon_outcome_known.to_numpy(dtype=bool)
    event = rows.event_observed.to_numpy(dtype=bool)
    scores = probability[:, :10].sum(-1)
    negatives = scores[known & ~event]
    if not len(negatives):
        return dict(threshold=None, status="insufficient_validation_negatives", target_fpr=target_fpr,
                    false_positive_unit="negative_window")
    candidates = np.r_[np.unique(scores[known]), np.nextafter(float(scores[known].max()), np.inf)]
    allowed = candidates[np.array([(negatives >= t).mean() <= target_fpr for t in candidates])]
    threshold = float(allowed.min())
    return dict(threshold=threshold, status="frozen_from_validation", target_fpr=target_fpr,
                validation_fpr=float((negatives >= threshold).mean()),
                validation_negatives=len(negatives), false_positive_unit="negative_window")


def scores(rows: pd.DataFrame, probability: np.ndarray, threshold: float | None = None):
    """Do not clip zero likelihood: empirical rule-only NLL can be infinite."""
    validate_targets(rows)
    validate_probabilities(probability)
    p = np.asarray(probability, dtype=np.float64)
    event = rows.event_observed.to_numpy(dtype=bool)
    bins = rows.event_bin.to_numpy(dtype=np.int64)
    observed = rows.observed_bins.to_numpy(dtype=np.int64)
    likelihood_valid = rows.likelihood_valid.to_numpy(dtype=bool)
    known = rows.horizon_outcome_known.to_numpy(dtype=bool)
    tail = (p * (np.arange(11)[None] >= observed[:, None])).sum(-1)
    likelihood = np.where(event, p[np.arange(len(rows)), bins.clip(0)], tail)
    with np.errstate(divide="ignore"):
        nll = -np.log(likelihood)
    event_probability = p[:, :10].sum(-1)
    horizon_y = event[known]
    horizon_p = event_probability[known]
    labels = rows.label.to_numpy(dtype=np.int64)
    target_cdf = np.arange(10)[None] >= labels[:, None]
    rps = ((p[:, :10].cumsum(-1) - target_cdf) ** 2).mean(-1)
    positives = int(horizon_y.sum())
    negatives = int((~horizon_y).sum())
    predicted = horizon_p >= threshold if threshold is not None else None
    result = dict(predicate=PREDICATE, n=len(rows), likelihood_n=int(likelihood_valid.sum()),
                  horizon_n=int(known.sum()), events=int(event.sum()), horizon_events=positives,
                  partial_censored=int((~known).sum()), zero_bin_censored=int((~likelihood_valid).sum()),
                  nll=float(nll[likelihood_valid].mean()) if likelihood_valid.any() else None,
                  zero_likelihood_rate=float((likelihood[likelihood_valid] == 0).mean()) if likelihood_valid.any() else None,
                  brier=float(((horizon_p - horizon_y) ** 2).mean()) if len(horizon_y) else None,
                  auprc=float(average_precision_score(horizon_y, horizon_p)) if positives else None,
                  rps=float(rps[known].mean()) if known.any() else None,
                  recall_at_fixed_fpr=float(predicted[horizon_y].mean()) if predicted is not None and positives else None,
                  realized_fpr=float(predicted[~horizon_y].mean()) if predicted is not None and negatives else None,
                  negative_windows=negatives, false_positive_unit="negative_window",
                  threshold=threshold, label_view=LABEL_VIEW,
                  metric_population="NLL: event or >=1 whole observed bin; horizon: observed event or all ten event-free bins")
    return result, dict(nll=nll, likelihood=likelihood, event_probability=event_probability,
                        brier=np.where(known, (event_probability-event) ** 2, np.nan),
                        rps=np.where(known, rps, np.nan))


SENSITIVITY_METRICS = ("nll", "zero_likelihood_rate", "brier", "auprc", "rps",
                       "recall_at_fixed_fpr", "realized_fpr")


def _independence_counts(rows):
    """Event counts are observed event windows, not independent physical events."""
    required = {"family_component", "full_copy_group", "full_copy_representative"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"independence metadata missing: {sorted(missing)}")
    if rows[list(required)].isna().any().any():
        raise ValueError("independence metadata must be present for every window")
    return dict(independent_components=int(rows.family_component.nunique()),
                event_components=int(rows.loc[rows.event_observed, "family_component"].nunique()),
                full_copy_groups=int(rows.full_copy_group.nunique()),
                event_full_copy_groups=int(rows.loc[rows.event_observed, "full_copy_group"].nunique()),
                event_windows=int(rows.event_observed.sum()),
                duplicate_windows=int(len(rows)-rows.full_copy_group.nunique()))


def _full_copy_representatives(rows):
    counts = rows.groupby("full_copy_group", sort=False).full_copy_representative.sum()
    if not counts.eq(1).all():
        raise ValueError("each evaluation full_copy_group requires exactly one declared representative")
    if rows.groupby("full_copy_group").family_component.nunique().gt(1).any():
        raise ValueError("evaluation full copies cannot cross independent components")
    # Truth/censor targets must agree. The data layer established exact physical
    # equality; this layer explicitly verifies equivalent supervision.
    for column in ("current", *TARGET_COLUMNS):
        if rows.groupby("full_copy_group")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"full-copy supervision differs: {column}")
    return np.flatnonzero(rows.full_copy_representative.to_numpy(dtype=bool))


def _component_macro(rows, probability, threshold):
    records = []
    for component in rows.family_component.unique():
        indices = np.flatnonzero(rows.family_component.eq(component).to_numpy())
        measured, _ = scores(rows.iloc[indices], probability[indices], threshold)
        records.append(dict(family_component=int(component), **measured))
    measured = scores(rows, probability, threshold)[0]
    support = {}
    for key in SENSITIVITY_METRICS:
        values = [record[key] for record in records if record[key] is not None]
        measured[key] = float(np.mean(values)) if values else None
        support[key] = len(values)
    return measured, support, records


def sensitivity_scores(rows, probability, threshold=None):
    """Report duplicate and independent-component sensitivity without refitting.

    Every view uses the same all-window validation threshold. Component means
    weight supported lineage components equally; undefined per-component AUPRC
    or recall contributes no value and its contributing count is explicit.
    """
    validate_targets(rows)
    validate_probabilities(probability)
    counts = _independence_counts(rows)
    representative = _full_copy_representatives(rows)
    dedup_rows, dedup_probability = rows.iloc[representative], probability[representative]
    all_window = scores(rows, probability, threshold)[0]
    dedup = scores(dedup_rows, dedup_probability, threshold)[0]
    component, component_support, component_records = _component_macro(rows, probability, threshold)
    component_dedup, dedup_support, dedup_component_records = _component_macro(dedup_rows, dedup_probability, threshold)
    views = []
    for method, measured, population in (("all_window", all_window, rows),
                                          ("full_copy_dedup", dedup, dedup_rows),
                                          ("component_macro", component, rows),
                                          ("component_macro_dedup", component_dedup, dedup_rows)):
        views.append(dict(weighting=method, **measured, **_independence_counts(population)))
    deltas = {}
    for method, measured in (("full_copy_dedup", dedup), ("component_macro", component),
                              ("component_macro_dedup", component_dedup)):
        deltas[method] = {}
        for key in SENSITIVITY_METRICS:
            a, b = measured[key], all_window[key]
            deltas[method][key] = float(a-b) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) else None
    return dict(counts=counts, views=views, delta_from_all_window=deltas,
                component_metric_support=component_support, dedup_component_metric_support=dedup_support,
                per_component=component_records, dedup_per_component=dedup_component_records,
                threshold_method="same frozen all-window validation operating point for every weighting view",
                selection_method="all-window validation NLL; sensitivity measures do not reselect models",
                event_count_definition="observed event windows and event-bearing full-copy groups/components; "
                                       "overlapping windows are not claimed to be independent physical events")


def supervision_weight_audit(rows):
    counts = _independence_counts(rows)
    _full_copy_representatives(rows)
    supported = rows.loc[rows.likelihood_valid]
    if not len(supported):
        raise ValueError("training objective has no supported windows")
    components = supported.groupby("family_component").agg(
        supported_windows=("row_id", "size"), observed_event_windows=("event_observed", "sum"),
        full_copy_groups=("full_copy_group", "nunique"))
    components["all_window_objective_mass"] = components.supported_windows / len(supported)
    components["equal_component_reference_mass"] = 1 / len(components)
    copy_sizes = supported.groupby("full_copy_group").size()
    return dict(counts=counts, likelihood_windows=len(supported),
                objective="mean censored NLL over supported training windows; zero-bin censoring excluded",
                donor_deduplication="sampler deduplication affects q donors and does not change supervision weights",
                supported_full_copy_groups=int(len(copy_sizes)),
                repeated_supported_windows=int(len(supported)-len(copy_sizes)),
                largest_full_copy_multiplicity=int(copy_sizes.max()),
                largest_component_objective_mass=float(components.all_window_objective_mass.max()),
                component_weights=components.reset_index().to_dict("records"), training_weight_changed=False)


def cluster_bootstrap(rows, probability, threshold, *, repetitions=200, seed=0):
    """Family-lineage uncertainty; too few independent groups is explicit."""
    column = "split_group"
    groups = rows[column].unique()
    if len(groups) < 5:
        return dict(status="insufficient_independent_groups", group_column=column, groups=len(groups),
                    minimum_groups=5, repetitions=0)
    generator = np.random.default_rng(seed)
    records = []
    members = {g: np.flatnonzero(rows[column].eq(g).to_numpy()) for g in groups}
    for _ in range(repetitions):
        index = np.concatenate([members[g] for g in generator.choice(groups, len(groups), replace=True)])
        records.append(scores(rows.iloc[index], probability[index], threshold)[0])
    intervals = {}
    for key in ("nll", "brier", "auprc", "recall_at_fixed_fpr", "rps"):
        values = np.array([r[key] for r in records if r[key] is not None], dtype=float)
        if not len(values):
            intervals[key] = dict(status="unmeasured")
        elif not np.isfinite(values).all():
            intervals[key] = dict(status="contains_infinite_replicates", infinite_replicates=int(np.isinf(values).sum()))
        else:
            intervals[key] = dict(lower=float(np.quantile(values, .025)), upper=float(np.quantile(values, .975)))
    return dict(status="measured", group_column=column, groups=len(groups), repetitions=repetitions, intervals=intervals)


def build_inputs(dataset, sampled_future, indices, device):
    h = dataset.history[indices]
    q = sampled_future[indices]
    parameters = dataset.rule_parameters[indices]
    values = dict(history=h, sampled_future=q, rule_parameters=parameters,
                  current=dataset.rows.current.to_numpy(dtype=np.int64)[indices],
                  history_rule_truth=evaluate_pair(h, parameters), sampled_rule_truth=evaluate_pair(q, parameters))
    return {key: torch.as_tensor(value, device=device, dtype=torch.long if key == "current" else torch.float32)
            for key, value in values.items()}


def infer(model, dataset, sampled_future, config, device):
    model.eval()
    predictions = []
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(dataset), config["batch_size"]):
            indices = np.arange(start, min(start + config["batch_size"], len(dataset)))
            logp = model(build_inputs(dataset, sampled_future, indices, device))
            predictions.append(logp.double().exp().cpu().numpy())
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - begin
    result = np.concatenate(predictions)
    validate_probabilities(result)
    return result, dict(seconds=elapsed, rows=len(dataset), k=sampled_future.shape[1],
                        seconds_per_query=elapsed / len(dataset), includes_rule_evaluation_and_transfer=True)


def training_summary(curves):
    selected = next(record for record in reversed(curves) if record["checkpoint_selected"])
    return dict(best_epoch=selected["epoch"], trained_epochs=curves[-1]["epoch"],
                total_optimizer_steps=sum(record["steps"] for record in curves),
                gradient_nonzero_steps=sum(record["gradient_nonzero_steps"] for record in curves))


def train_one(train, validation, train_q, validation_q, prior_logits, config, learning_rate, device):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = ConditionalTrajectoryHead(config["width"]).to(device)
    model.prior_logits.copy_(torch.tensor(prior_logits, device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config["weight_decay"])
    best, best_state, curves, stale = float("inf"), None, [], 0
    begin = time.perf_counter()
    supported = train.rows.likelihood_valid.to_numpy(dtype=bool)
    if not supported.any():
        raise ValueError("no likelihood-supported training rows")
    for epoch in range(config["epochs"] + 1):
        train_objective, gradient_max, gradient_nonzero_steps, steps = 0.0, 0.0, 0, 0
        if epoch:
            model.train()
            order = np.random.permutation(np.flatnonzero(supported))
            for start in range(0, len(order), config["batch_size"]):
                indices = order[start:start+config["batch_size"]]
                rows = train.rows.iloc[indices]
                optimizer.zero_grad(set_to_none=True)
                logp = model(build_inputs(train, train_q, indices, device))
                loss = censored_nll(logp, rows.event_observed.to_numpy(), rows.event_bin.to_numpy(),
                                    rows.observed_bins.to_numpy()).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                gradient_max = max(gradient_max, float(norm))
                gradient_nonzero_steps += int(float(norm) > 0)
                optimizer.step()
                train_objective += float(loss.detach()) * len(indices)
                steps += 1
        train_probability, _ = infer(model, train, train_q, config, device)
        validation_probability, _ = infer(model, validation, validation_q, config, device)
        tm, _ = scores(train.rows, train_probability)
        vm, _ = scores(validation.rows, validation_probability)
        record = dict(epoch=epoch, train_objective=train_objective/int(supported.sum()) if epoch else None,
                      train_nll=tm["nll"], validation_nll=vm["nll"], train_brier=tm["brier"],
                      validation_brier=vm["brier"], train_auprc=tm["auprc"], validation_auprc=vm["auprc"],
                      gradient_norm_max=gradient_max, gradient_nonzero_steps=gradient_nonzero_steps, steps=steps,
                      correction_weight_l2=float(model.hazard.weight.detach().norm()),
                      correction_bias=float(model.hazard.bias.detach()), elapsed_seconds=time.perf_counter()-begin)
        record["checkpoint_selected"] = vm["nll"] < best - 1e-5
        curves.append(record)
        if record["checkpoint_selected"]:
            best, best_state, stale = vm["nll"], deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch % 5 == 0 or stale >= config["patience"] or epoch == config["epochs"]:
            print(json.dumps(_json_value(dict(learning_rate=learning_rate, k=train_q.shape[1], **record))), flush=True)
        if stale >= config["patience"]:
            break
    if best_state is None:
        raise FloatingPointError("no finite validation checkpoint")
    model.load_state_dict(best_state)
    print(json.dumps(_json_value(dict(kind="training_selection", learning_rate=learning_rate,
                                     k=train_q.shape[1], seed=seed, validation_nll=best,
                                     **training_summary(curves)))), flush=True)
    return model, curves, best


def engineered(dataset):
    """History/rule summaries for the strongest existing person-period GBT API."""
    h, p = dataset.history, dataset.rule_parameters
    delta = h[:, :, 1] - h[:, :, 0]
    scales = np.c_[p[:, 1], p[:, 1], p[:, 2], p[:, 1], p[:, 1], p[:, 2]]
    relative = delta / scales[:, None]
    pair_velocity = h[:, :, :, 3:].mean(axis=2) / scales[:, None, :3]
    truth = evaluate_pair(h, p)
    flips = truth[:, 1:] != truth[:, :-1]
    return np.c_[relative[:, -1], relative.mean(1), relative.std(1), relative[:, -1]-relative[:, 0],
                 pair_velocity[:, -1], pair_velocity.mean(1), truth.mean(1), flips.sum(1)/20,
                 p[:, 0], np.log(p[:, 1:])/5, dataset.rows.current.to_numpy(dtype=float)]


def tree_predict(model, features):
    hazards = np.stack([model.predict_proba(np.c_[features, np.full(len(features), k/9)])[:, 1]
                        for k in range(10)], axis=1)
    return hazard_probabilities(hazards)


def train_trees(train, validation):
    train_x, validation_x = engineered(train), engineered(validation)
    person, period, target = person_period_targets(train.rows)
    if len(np.unique(target)) != 2:
        raise ValueError("person-period GBT requires observed events and survival decisions")
    x = np.c_[train_x[person], period/9]
    best, winner, records = float("inf"), None, []
    for leaves in (7, 15):
        tree = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=leaves, learning_rate=.06,
                                             l2_regularization=1.0, early_stopping=False, random_state=0)
        begin = time.perf_counter()
        tree.fit(x, target)
        validation_probability = tree_predict(tree, validation_x)
        measured, _ = scores(validation.rows, validation_probability)
        records.append(dict(leaves=leaves, seconds=time.perf_counter()-begin, validation=measured))
        if measured["nll"] < best:
            best, winner = measured["nll"], tree
    return winner, records


def specs(ks):
    return [dict(name="center_K1", mode="center", k=1)] + [
        dict(name=f"{mode}_K{k}", mode=mode, k=k) for mode in ("independent", "joint") for k in ks]


def sample_cache(sampler, dataset, config):
    result, metadata = {}, []
    for mode in ("center", "independent", "joint"):
        k = 1 if mode == "center" else max(config["ks"])
        begin = time.perf_counter()
        result[mode] = sampler.sample_dataset(dataset, k=k, seed=config["q_seed"], mode=mode)
        if result[mode].shape != (len(dataset), k, 10, 2, 6) or not np.isfinite(result[mode]).all():
            raise ValueError("sampler must return finite [N,K,10,2,6] draws")
        rows = sampler.last_sample_metadata.copy()
        rows["row_id"] = dataset.rows.row_id.to_numpy()
        rows["mode"] = mode
        rows["sample_seconds"] = time.perf_counter()-begin
        metadata.append(rows)
    return result, pd.concat(metadata, ignore_index=True)


def _prediction_rows(dataset, probability, name):
    fields = ["row_id", "family", "episode", "split", "current", "split_group", "family_component",
              "full_copy_group", "full_copy_representative", *TARGET_COLUMNS]
    rows = dataset.rows[fields].copy()
    rows["model"] = name
    for i in range(11):
        rows[f"p{i}"] = probability[:, i]
    return rows


def develop(args):
    output = _output_directory(args.run_dir, new=True)
    config = dict(data_dir=str(Path(args.data_dir).resolve()), label_view=LABEL_VIEW,
                  width=args.width, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                  learning_rates=args.learning_rates, ks=args.ks, seed=args.seed, q_seed=args.q_seed,
                  weight_decay=args.weight_decay, gradient_clip=5., target_fpr=args.fixed_fpr,
                  min_support=args.min_support, device=args.device, command=shlex.join(sys.argv),
                  neural_specs=specs(args.ks), selection_metric="validation_window_nll", test_locked=True)
    config["rule_mc_smoothing"] = dict(
        formula="p_smooth=(K*p_MC+alpha*p_train_prior(current))/(K+alpha)",
        primary_alpha=SMOOTH_RULE_PRIMARY_ALPHA, prespecified_alphas=list(SMOOTH_RULE_ALPHAS),
        diagnostic_only_alphas=[a for a in SMOOTH_RULE_ALPHAS if a != SMOOTH_RULE_PRIMARY_ALPHA],
        prior="train-only empirical hazards with fixed Beta(0.5,0.5) pseudo counts",
        primary_alpha_selection="fixed before validation, never tuned", likelihood_clipping=False)
    ceilings = [analytical_ceiling(args.width, args.batch_size, k) for k in args.ks]
    write_json(output / "config.json", config)
    write_json(output / "analytical_ceiling.json", dict(estimates=ceilings, measured=False,
                                                      neural_trials=len(config["neural_specs"])*len(args.learning_rates),
                                                      max_epochs_per_trial=args.epochs, gbt_trials=2))
    dataset = load_window_dataset(args.data_dir)
    train, validation = split_dataset(dataset, "train"), split_dataset(dataset, "validation")
    group_column = "split_group"
    if set(train.rows[group_column]) & set(validation.rows[group_column]):
        raise ValueError("train and validation lineage groups overlap")
    prior, prior_logits, prior_counts = make_prior(train.rows)
    write_json(output / "training_supervision_weights.json", supervision_weight_audit(train.rows))
    write_json(output / "prior.json", dict(probabilities=prior.tolist(), logits=prior_logits.tolist(), **prior_counts))
    sampler = EmpiricalResidualSampler(min_support=config["min_support"]).fit(train)
    train_q, train_support = sample_cache(sampler, train, config)
    validation_q, validation_support = sample_cache(sampler, validation, config)
    pd.concat([train_support.assign(split="train"), validation_support.assign(split="validation")]).to_csv(
        output / "sampler_support.csv", index=False)
    joblib.dump(sampler, output / "sampler.joblib")
    write_json(output / "sampler_diagnostics.json", sampler.diagnostics)
    selection, metrics, per_predicate, prediction_rows, curves, latency, sensitivity = {}, [], [], [], [], [], {}
    smooth_sensitivity, smooth_sensitivity_views = [], []

    def record(name, probability, parameters=0):
        threshold = fixed_fpr_threshold(validation.rows, probability, config["target_fpr"])
        measured, _ = scores(validation.rows, probability, threshold["threshold"])
        metrics.append(dict(model=name, split="validation", parameters=parameters, **measured))
        per_predicate.append(dict(model=name, split="validation", **measured))
        prediction_rows.append(_prediction_rows(validation, probability, name))
        sensitivity[name] = sensitivity_scores(validation.rows, probability, threshold["threshold"])
        return measured, threshold

    probability = prior[validation.rows.current.to_numpy(dtype=np.int64)]
    measured, threshold = record("Prior", probability)
    selection["Prior"] = dict(validation=measured, operating_point=threshold, kind="prior")
    tree, tree_trials = train_trees(train, validation)
    joblib.dump(tree, output / "GBT.joblib")
    measured, threshold = record("GBT", tree_predict(tree, engineered(validation)))
    selection["GBT"] = dict(validation=measured, operating_point=threshold, trials=tree_trials, kind="gbt")
    for spec in config["neural_specs"]:
        name, mode, k = spec["name"], spec["mode"], spec["k"]
        rule_probability = first_opposite_distribution(validation.rows.current.to_numpy(),
                                                       evaluate_pair(validation_q[mode][:, :k], validation.rule_parameters))
        measured, threshold = record("Rule_"+name, rule_probability)
        selection["Rule_"+name] = dict(validation=measured, operating_point=threshold, kind="rule_mc", **spec)
        validation_prior = prior[validation.rows.current.to_numpy(dtype=np.int64)]
        smooth_probability = smooth_rule_probabilities(rule_probability, validation_prior, k, SMOOTH_RULE_PRIMARY_ALPHA)
        measured, threshold = record("SmoothRule_"+name, smooth_probability)
        selection["SmoothRule_"+name] = dict(validation=measured, operating_point=threshold,
                                             kind="smoothed_rule_mc", alpha=SMOOTH_RULE_PRIMARY_ALPHA, **spec)
        for alpha in SMOOTH_RULE_ALPHAS:
            probability = smooth_rule_probabilities(rule_probability, validation_prior, k, alpha)
            operating_point = fixed_fpr_threshold(validation.rows, probability, config["target_fpr"])
            measured, _ = scores(validation.rows, probability, operating_point["threshold"])
            diagnostic_name = f"SmoothRule_{name}_alpha{alpha:g}"
            smooth_sensitivity.append(dict(name=diagnostic_name, spec=spec, alpha=alpha,
                                           diagnostic_only=alpha != SMOOTH_RULE_PRIMARY_ALPHA,
                                           operating_point=operating_point, validation=measured))
            report = sensitivity_scores(validation.rows, probability, operating_point["threshold"])
            smooth_sensitivity_views.extend(dict(model=diagnostic_name, split="validation", alpha=alpha,
                                                  diagnostic_only=alpha != SMOOTH_RULE_PRIMARY_ALPHA, **view)
                                            for view in report["views"])
        best, winner, trials = float("inf"), None, []
        for learning_rate in config["learning_rates"]:
            print(json.dumps(dict(training=name, learning_rate=learning_rate)), flush=True)
            model, curve, value = train_one(train, validation, train_q[mode][:, :k], validation_q[mode][:, :k],
                                           prior_logits, config, learning_rate, args.device)
            curves.extend(dict(model=name, seed=config["seed"], learning_rate=learning_rate, **r) for r in curve)
            summary = training_summary(curve)
            trials.append(dict(learning_rate=learning_rate, validation_nll=value, epochs=len(curve)-1, **summary))
            pd.DataFrame(curves).to_csv(output / "learning_curves.csv", index=False)
            if value < best:
                best, winner = value, model
                checkpoint = dict(state=model.state_dict(), width=config["width"], spec=spec,
                                  seed=config["seed"], learning_rate=learning_rate, validation_nll=value, **summary)
                torch.save(checkpoint, output / (name+".pt"))
        if winner is None:
            raise FloatingPointError(f"no validation-selected checkpoint for {name}")
        probability, timing = infer(winner, validation, validation_q[mode][:, :k], config, args.device)
        latency.append(dict(model=name, split="validation", **timing))
        parameters = sum(p.numel() for p in winner.parameters())
        measured, threshold = record(name, probability, parameters)
        selection[name] = dict(validation=measured, operating_point=threshold, trials=trials,
                               kind="conditional_head", **spec)
    candidates = config["neural_specs"]
    selected = min(candidates, key=lambda spec: selection[spec["name"]]["validation"]["nll"])["name"]
    baseline_names = baseline_candidates(selection)
    selected_baseline = min(baseline_names, key=lambda name: selection[name]["validation"]["nll"])
    joint_names = [spec["name"] for spec in candidates if spec["mode"] == "joint"]
    selected_joint = min(joint_names, key=lambda name: selection[name]["validation"]["nll"])
    receipt = dict(selection=selection, selected_neural_model=selected, selected_baseline_model=selected_baseline,
                   selected_joint_model=selected_joint,
                   validation_only=True, test_opened=False,
                   prespecified_test_models=list(selection), seed=config["seed"], q_seed=config["q_seed"],
                   target_fpr=config["target_fpr"], label_view=LABEL_VIEW,
                   selection_metric=config["selection_metric"],
                   sensitivity_methods=["all_window", "full_copy_dedup", "component_macro", "component_macro_dedup"],
                   sensitivity_used_for_selection=False, baseline_candidates=baseline_names, required_gbt_model="GBT",
                   baseline_scope="Prior, GBT, center/independent conditional heads, all raw and alpha=1 smoothed "
                                  "rule-only MC methods including joint q; learned joint heads excluded",
                   rule_mc_smoothing=config["rule_mc_smoothing"], smooth_rule_sensitivity=smooth_sensitivity)
    # This receipt is the final development write and is never modified by evaluation.
    pd.DataFrame(metrics).to_csv(output / "validation_metrics.csv", index=False)
    pd.DataFrame(per_predicate).to_csv(output / "validation_per_predicate_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(output / "validation_predictions.parquet", index=False)
    pd.DataFrame(latency).to_csv(output / "latency.csv", index=False)
    write_json(output / "validation_sensitivity.json", sensitivity)
    pd.DataFrame([dict(model=name, split="validation", **view) for name, report in sensitivity.items()
                  for view in report["views"]]).to_csv(output / "validation_sensitivity_metrics.csv", index=False)
    pd.DataFrame(smooth_sensitivity_views).to_csv(output / "validation_smooth_rule_sensitivity_metrics.csv", index=False)
    write_json(output / "selection.json", receipt)
    print(json.dumps(dict(development_complete=str(output), selected_neural_model=selected)), flush=True)


def gbt_refinement_grid():
    return [dict(leaves=leaves, learning_rate=learning_rate, iterations=iterations)
            for leaves in (7, 15) for learning_rate in (.03, .06) for iterations in (100, 200)]


def train_tree_refinement(train, validation, target_fpr):
    train_x, validation_x = engineered(train), engineered(validation)
    person, period, target = person_period_targets(train.rows)
    if len(np.unique(target)) != 2:
        raise ValueError("person-period GBT refinement requires events and survival decisions")
    x = np.c_[train_x[person], period/9]
    best, winner, winner_probability, records = float("inf"), None, None, []
    for candidate in gbt_refinement_grid():
        tree = HistGradientBoostingClassifier(max_iter=candidate["iterations"], max_leaf_nodes=candidate["leaves"],
                                             learning_rate=candidate["learning_rate"], l2_regularization=1.0,
                                             early_stopping=False, random_state=0)
        begin = time.perf_counter()
        tree.fit(x, target)
        probability = tree_predict(tree, validation_x)
        operating_point = fixed_fpr_threshold(validation.rows, probability, target_fpr)
        measured, _ = scores(validation.rows, probability, operating_point["threshold"])
        record = dict(**candidate, seconds=time.perf_counter()-begin, validation=measured,
                      operating_point=operating_point)
        records.append(record)
        print(json.dumps(_json_value(dict(model="GBT_refinement", **record))), flush=True)
        if measured["nll"] < best:
            best, winner, winner_probability = measured["nll"], tree, probability
    if winner is None:
        raise FloatingPointError("no finite validation GBT refinement candidate")
    return winner, winner_probability, records


def refine_gbt(args):
    """Replace GBT in a new artifact while preserving the completed source."""
    source = _output_directory(args.source_run_dir, new=False)
    destination = Path(args.run_dir).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("GBT refinement output must be outside the immutable source directory")
    for name in ("test_metrics.csv", "test_predictions.parquet", "test_complete.json"):
        if (source / name).exists():
            raise ValueError("GBT refinement requires an unopened source test split")
    config = json.loads((source / "config.json").read_text())
    receipt = json.loads((source / "selection.json").read_text())
    if not receipt["validation_only"] or receipt["test_opened"] or config["label_view"] != LABEL_VIEW:
        raise ValueError("GBT refinement requires a completed validation-only selection receipt")
    if "gbt_refinement" in receipt:
        raise ValueError("this fixed grid can refine only the initial development artifact")
    initial_records = receipt["selection"]["GBT"]["trials"]
    if len(initial_records) != 2 or {r["leaves"] for r in initial_records} != {7, 15}:
        raise ValueError("initial GBT contract requires exactly the executed 7/15-leaf candidates")
    canonical = ["config.json", "analytical_ceiling.json", "prior.json", "sampler.joblib",
                 "sampler_diagnostics.json", "sampler_support.csv", "training_supervision_weights.json",
                 "learning_curves.csv", "GBT.joblib", "validation_metrics.csv", "validation_per_predicate_metrics.csv",
                 "validation_predictions.parquet", "latency.csv", "validation_sensitivity.json",
                 "validation_sensitivity_metrics.csv", "validation_smooth_rule_sensitivity_metrics.csv", "selection.json"]
    canonical.extend(spec["name"]+".pt" for spec in config["neural_specs"])
    original = {name: (source / name).read_bytes() for name in canonical}
    output = _output_directory(args.run_dir, new=True)
    for name in canonical:
        if name != "selection.json":
            shutil.copy2(source / name, output / name)
    changed = {"config.json", "selection.json", "GBT.joblib", "validation_metrics.csv",
               "validation_per_predicate_metrics.csv", "validation_predictions.parquet",
               "validation_sensitivity.json", "validation_sensitivity_metrics.csv"}
    frozen = [name for name in canonical if name not in changed]
    dataset = load_window_dataset(config["data_dir"])
    train, validation = split_dataset(dataset, "train"), split_dataset(dataset, "validation")
    if set(train.rows.split_group) & set(validation.rows.split_group):
        raise ValueError("GBT refinement train/validation lineage groups overlap")
    grid = gbt_refinement_grid()
    person, _, _ = person_period_targets(train.rows)
    features = engineered(train).shape[1] + 1
    write_json(output / "gbt_analytical_ceiling.json", dict(
        kind="analytical_upper_bound", measured=False, candidates=len(grid), person_period_rows=len(person),
        features=features, maximum_iterations=200, maximum_leaves=15,
        maximum_tree_nodes=200*(2*15-1), maximum_scalar_split_comparisons_per_query=10*200*(15-1),
        histogram_accumulation_upper_bound=len(person)*features*200*(15-1),
        assumptions="binary trees with at most 15 leaves/14 depth, 200 boosting rounds; ten person-period "
                    "predictions per query; histogram upper bound scans all train rows/features per depth; "
                    "excludes binning, split search and framework overhead; not measured runtime"))
    tree, probability, records = train_tree_refinement(train, validation, config["target_fpr"])
    joblib.dump(tree, output / "GBT.joblib")
    operating_point = fixed_fpr_threshold(validation.rows, probability, config["target_fpr"])
    measured, _ = scores(validation.rows, probability, operating_point["threshold"])
    report = sensitivity_scores(validation.rows, probability, operating_point["threshold"])
    selection = receipt["selection"]
    selection["GBT"] = dict(kind="gbt", validation=measured, operating_point=operating_point, trials=records)
    for filename in ("validation_metrics.csv", "validation_per_predicate_metrics.csv"):
        table = pd.read_csv(source / filename)
        row = dict(model="GBT", split="validation", **measured)
        if filename == "validation_metrics.csv":
            row["parameters"] = 0
        pd.concat([table.loc[table.model.ne("GBT")], pd.DataFrame([row])], ignore_index=True).to_csv(output / filename, index=False)
    predictions = pd.read_parquet(source / "validation_predictions.parquet")
    pd.concat([predictions.loc[predictions.model.ne("GBT")], _prediction_rows(validation, probability, "GBT")],
              ignore_index=True).to_parquet(output / "validation_predictions.parquet", index=False)
    sensitivity = json.loads((source / "validation_sensitivity.json").read_text())
    sensitivity["GBT"] = report
    write_json(output / "validation_sensitivity.json", sensitivity)
    table = pd.read_csv(source / "validation_sensitivity_metrics.csv")
    replacements = pd.DataFrame([dict(model="GBT", split="validation", **view) for view in report["views"]])
    pd.concat([table.loc[table.model.ne("GBT")], replacements], ignore_index=True).to_csv(
        output / "validation_sensitivity_metrics.csv", index=False)
    formal_neural = [(name, trial["learning_rate"]) for name, entry in selection.items()
                     if entry["kind"] == "conditional_head" for trial in entry["trials"]]
    initial_candidates = {(r["leaves"], .06, 100) for r in initial_records}
    refined_candidates = {(r["leaves"], r["learning_rate"], r["iterations"]) for r in records}
    counts = dict(neural_unique_formal_candidates=len(set(formal_neural)), neural_formal_trial_executions=len(formal_neural),
                  exploratory_neural_trial_executions=args.exploratory_neural_trials,
                  neural_total_trial_executions=len(formal_neural)+args.exploratory_neural_trials,
                  neural_refinement_trial_executions=0, gbt_initial_trial_executions=len(initial_records),
                  gbt_refinement_trial_executions=len(records),
                  gbt_total_trial_executions=len(initial_records)+len(records),
                  gbt_unique_candidates=len(initial_candidates | refined_candidates),
                  gbt_repeated_candidates=len(initial_candidates & refined_candidates),
                  exploratory_count_provenance="explicit CLI declaration; exploratory fits are disclosed separately")
    receipt["baseline_candidates"] = baseline_candidates(selection)
    receipt["selected_baseline_model"] = min(receipt["baseline_candidates"],
                                              key=lambda name: float(selection[name]["validation"]["nll"]))
    receipt["gbt_refinement"] = dict(source_run_dir=str(source), grid=grid, initial_trials=initial_records,
                                      candidate_counts=counts, neural_q_prior_unchanged=True,
                                      source_unchanged=True, frozen_files=frozen, test_opened=False)
    config["development_command"] = config["command"]
    config["command"] = shlex.join(sys.argv)
    config["source_run_dir"] = str(source)
    config["gbt_refinement_grid"] = grid
    write_json(output / "config.json", config)
    write_json(output / "gbt_refinement.json", receipt["gbt_refinement"])
    for name, content in original.items():
        if (source / name).read_bytes() != content:
            raise ArithmeticError(f"source experiment changed during refinement: {name}")
    for name in frozen:
        if (output / name).read_bytes() != original[name]:
            raise ArithmeticError(f"frozen experiment artifact changed during refinement: {name}")
    write_json(output / "selection.json", receipt)
    print(json.dumps(dict(gbt_refinement_complete=str(output), selected_baseline_model=receipt["selected_baseline_model"],
                          validation_gbt_nll=measured["nll"], candidate_counts=counts)), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("phase", choices=("develop", "refine-gbt"))
    result.add_argument("--run-dir", default=str(OUTPUT_ROOT / "model"))
    result.add_argument("--data-dir", default=str(OUTPUT_ROOT / "inputs"))
    result.add_argument("--source-run-dir")
    result.add_argument("--exploratory-neural-trials", type=int, default=5)
    result.add_argument("--device", default="cpu")
    result.add_argument("--width", type=int, default=24)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--epochs", type=int, default=30)
    result.add_argument("--patience", type=int, default=5)
    result.add_argument("--learning-rates", type=float, nargs="+", default=[.001, .003])
    result.add_argument("--ks", type=int, nargs="+", default=[1, 8, 16])
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--q-seed", type=int, default=0)
    result.add_argument("--min-support", type=int, default=2)
    result.add_argument("--fixed-fpr", type=float, default=.05)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--threads", type=int, default=4)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.phase == "develop" and args.data_dir is None:
        raise ValueError("develop requires --data-dir")
    if args.phase == "refine-gbt" and args.source_run_dir is None:
        raise ValueError("refine-gbt requires --source-run-dir")
    if args.exploratory_neural_trials < 0:
        raise ValueError("exploratory trial count must be nonnegative")
    if not args.ks or min(args.ks) < 1 or len(set(args.ks)) != len(args.ks) or args.ks != sorted(args.ks):
        raise ValueError("ks must be unique increasing positive integers")
    if args.width < 1 or args.batch_size < 1 or args.epochs < 0 or args.patience < 1 or args.threads < 1:
        raise ValueError("invalid model/training budget")
    if not 0 <= args.fixed_fpr < 1 or not args.learning_rates or min(args.learning_rates) <= 0:
        raise ValueError("invalid operating point or learning rate")
    torch.set_num_threads(args.threads)
    if args.phase == "develop":
        develop(args)
    elif args.phase == "refine-gbt":
        refine_gbt(args)


if __name__ == "__main__":
    main()
