"""Calibration, missing-label bounds, kinematic baseline and graph deletion analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import GraphDataset, OUT, ROOT, write_json, TARGET_COLUMNS, METRIC_SELECTION, person_period_targets, attach_label_view, validate_targets, LABEL_VIEW, require_v2_configuration
from .experiment import build_batches, hazard_probabilities, scores, infer, tree_features, tree_predict, train_trees
from .model import SwitchModel, build_model, censored_nll

PAIR_TASK = "safety.aircraft_pair_below_separation"
PROB_COLUMNS = [f"p{k}" for k in range(11)]


def cpa_features(dataset, batches):
    std = np.array(dataset.vocab["normalization"]["std"])
    tick_seconds = yaml.safe_load((ROOT / "rules/physics.yaml").read_text())["platform_envelope"]["tick_seconds"]
    assert tick_seconds == 0.1
    result = {}
    for split, batch in batches.items():
        features, frames = [], []
        for x, _, rows in batch:
            select = np.flatnonzero(rows.predicate.to_numpy() == PAIR_TASK)
            if not len(select):
                continue
            q = x["query"][select]
            pointer = x["pointers"][q, :2]
            present = x["physical_present"][-1, pointer].all(-1).cpu().numpy()
            select, pointer = select[present], pointer[present]
            if not len(select):
                continue
            states = x["physical"][-1, pointer].cpu().numpy()*std
            relative = states[:, 1, :3]-states[:, 0, :3]
            velocity = states[:, 1, 3:6]-states[:, 0, 3:6]
            speed2 = (velocity**2).sum(-1)
            rv = (relative*velocity).sum(-1)
            tau = np.divide(-rv, speed2, out=np.zeros_like(rv), where=speed2 > 1e-12).clip(0, 50*tick_seconds)
            distance = np.linalg.norm(relative, axis=-1)
            closest = np.linalg.norm(relative+tau[:, None]*velocity, axis=-1)
            features.append(np.c_[distance, closest, tau, np.sqrt(speed2), rv/distance.clip(1e-6), rows.iloc[select].current])
            frames.append(rows.iloc[select])
        result[split] = np.concatenate(features), pd.concat(frames)
    return result


def cpa_predict(model, x):
    hazards = np.stack([model.predict_proba(np.c_[x, np.full(len(x), k/9)])[:, 1] for k in range(10)], -1)
    return hazard_probabilities(hazards)


def supplement_development():
    assert not (OUT / "metrics.csv").exists(), "Freeze supplemental baselines before opening held-out metrics"
    dataset = GraphDataset()
    config = json.loads((OUT / "config.json").read_text())
    require_v2_configuration(config)
    batches = build_batches(dataset, "cpu")
    feature = cpa_features(dataset, batches)
    xx, rows = feature["train"]
    selected = rows.likelihood_valid.to_numpy()
    xx, rows = xx[selected], rows.loc[selected]
    person, period, Y = person_period_targets(rows)
    X = np.c_[xx[person], period/9]
    records, best = [], float("inf")
    for c in (0.1, 1.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=0))
        model.fit(X, Y)
        metric, _ = scores(feature["validation"][1], cpa_predict(model, feature["validation"][0]), [PAIR_TASK])
        records.append(dict(C=c, validation=metric))
        if metric["nll"] < best:
            best = metric["nll"]
            joblib.dump(model, OUT / "models/CPA.joblib")
    write_json(OUT / "cpa_selection.json", dict(task=PAIR_TASK, trials=records,
        features=["distance_m", "closest_distance_m", "closest_time_s_clipped_0_5", "relative_speed_mps", "radial_speed_mps", "current_truth", "relative_future_bin"],
        physical_authority="rules/physics.yaml:platform_envelope.tick_seconds", person_period_rows=len(Y)))
    # A regularized absolute-phase diagnostic. Empty training phases retain the explicitly declared prior.
    prior = np.array(json.loads((OUT / "prior.json").read_text())["probabilities"])
    events = np.broadcast_to(prior[:, :, None, :10]*10, (len(prior), 2, 10, 10)).copy()
    tail = np.stack([prior[:, :, k:].sum(-1) for k in range(10)], -1)
    risk = np.broadcast_to(tail[:, :, None, :]*10, events.shape).copy()
    train = dataset.index[(dataset.index.split == "train") & dataset.index.likelihood_valid]
    person, period, target = person_period_targets(train)
    types = np.array([dataset.vocab["predicates"].index(p) for p in train.predicate])
    current, phase_bin = train.current.to_numpy(), train.anchor.to_numpy()//100
    np.add.at(events, (types[person], current[person], phase_bin[person], period), target)
    np.add.at(risk, (types[person], current[person], phase_bin[person], period), 1)
    phase = hazard_probabilities((events/risk).reshape(-1, 10)).reshape(len(prior), 2, 10, 11)
    write_json(OUT / "phase_prior.json", dict(probabilities=phase.tolist(), phase_width_ticks=100,
        prior_strength=10, train_phase_bins=sorted((train.anchor//100).unique().tolist()),
        warning="Validation anchors occupy phases with no training support. This diagnostic cannot rule out unobserved script-phase confounding."))
    config["supplemental_baselines"] = dict(CPA=dict(task=PAIR_TASK, C=[0.1, 1.0]),
        phase=dict(width_ticks=100, prior_strength=10), deletion=dict(models=["G", "G_primary"], seed=0,
        maximum_queries_per_primary_task=8, random_seed=4242, selection="uniform over likelihood-valid test queries per task; no outcome stratification"))
    write_json(OUT / "config.json", config)
    tree, tree_trials = train_trees(tree_features(batches, dataset.vocab), dict(config, objective_scope="primary"))
    joblib.dump(tree, OUT / "models/GBT_primary.joblib")
    write_json(OUT / "tree_primary_selection.json", tree_trials)
    print(json.dumps(dict(cpa=records, phase_bins=sorted((train.anchor//100).unique().tolist()))), flush=True)


def freeze_evaluation(device):
    assert not (OUT / "metrics.csv").exists(), "Cannot select a candidate after held-out metrics"
    assert (OUT / "refinement_summary.json").exists()
    dataset = GraphDataset()
    config = json.loads((OUT / "config.json").read_text())
    require_v2_configuration(config)
    batches = build_batches(dataset, device)
    candidates, runtime = [], []
    import time
    for name, mode in config["run_modes"].items():
        probabilities = []
        for seed in config["optimization_seeds"]:
            model = build_model(dataset.vocab, mode, config["width"]).to(device)
            model.load_state_dict(torch.load(OUT / "models" / f"{name}_seed{seed}.pt", map_location=device, weights_only=False)["state"])
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            frame, p = infer(model, batches, "validation")
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            runtime.append(dict(model=name, seed=seed, validation_inference_seconds=time.perf_counter()-begin,
                queries=len(frame), parameters=sum(p.numel() for p in model.parameters())))
            probabilities.append(p)
        metric, _ = scores(frame, np.mean(probabilities, axis=0), config["primary_predicates"])
        candidates.append(dict(model=name+"_ensemble", **metric))
    feature = tree_features({"validation": batches["validation"]}, dataset.vocab)["validation"]
    for name in ("GBT", "GBT_primary"):
        tree = joblib.load(OUT / "models" / (name+".joblib"))
        metric, _ = scores(feature[1], tree_predict(tree, feature[0]), config["primary_predicates"])
        candidates.append(dict(model=name, **metric))
    winner = min(candidates, key=lambda row: row["nll"])["model"]
    write_json(OUT / "evaluation_protocol.json", dict(label_view=LABEL_VIEW, candidate=winner, candidates=candidates,
        selection="lowest primary-task validation joint NLL; neural candidates average three fixed optimization seeds",
        test_unopened=True, primary_predicates=config["primary_predicates"],
        minimum_relative_nll_gain=config["minimum_relative_nll_gain"], reliability_tolerance=config["reliability_tolerance"],
        no_further_training="All subsequent work is inference, declared ablations, diagnostics and reporting."))
    pd.DataFrame(runtime).to_csv(OUT / "model_resources.csv", index=False)
    print(json.dumps(dict(candidate=winner, validation_nll={r["model"]:r["nll"] for r in candidates})), flush=True)


def tree_curves():
    """Evaluate stored boosting stages; no refit or checkpoint reselection."""
    dataset = GraphDataset()
    config = json.loads((OUT / "config.json").read_text())
    require_v2_configuration(config)
    features = tree_features(build_batches(dataset, "cpu"), dataset.vocab)
    curves = []
    for name in ("GBT", "GBT_primary"):
        model = joblib.load(OUT / "models" / (name+".joblib"))
        for split, (x, frame) in features.items():
            augmented = np.c_[np.repeat(x, 10, axis=0), np.tile(np.arange(10)/9, len(x))]
            for stage, pp in enumerate(model.staged_predict_proba(augmented), 1):
                if stage % 5:
                    continue
                prob = hazard_probabilities(pp[:, 1].reshape(len(x), 10))
                m, _ = scores(frame, prob, config["primary_predicates"])
                curves.append(dict(model=name, split=split, stage=stage, **m))
    pd.DataFrame(curves).to_csv(OUT / "tree_learning_curves.csv", index=False)
    print(json.dumps(dict(staged_curve_rows=len(curves), refitted=False, checkpoints_reselected=False)), flush=True)


def delete_nonquery(x, query_position):
    """Keep query entities, their states and predicates wholly contained in this entity set."""
    query = int(x["query"][query_position])
    entities = x["pointers"][query][x["role_present"][query]].unique()
    retained = torch.isin(x["pointers"], entities) | ~x["role_present"]
    pred = torch.where(retained.all(-1))[0]
    assert (pred == query).any()
    remap = torch.full((x["physical"].shape[1],), -1, dtype=torch.long, device=entities.device)
    remap[entities] = torch.arange(len(entities), device=entities.device)
    pointers = remap[x["pointers"][pred]]
    pointers[~x["role_present"][pred]] = 0
    return dict(physical=x["physical"][:, entities], physical_present=x["physical_present"][:, entities],
        entity_visible=x["entity_visible"][:, entities], entity_type=x["entity_type"][entities],
        truth=x["truth"][:, pred], predicate_type=x["predicate_type"][pred], pointers=pointers,
        role_type=x["role_type"][pred], role_present=x["role_present"][pred],
        query=torch.where(pred == query)[0], relative_age=x["relative_age"])


def reliability(frame, probability, predicates):
    validate_targets(frame)
    selected = frame.horizon_outcome_known.to_numpy() & frame.predicate.isin(predicates).to_numpy()
    frame = frame.loc[selected]
    probability = probability[selected].astype(float)
    probability /= probability.sum(-1, keepdims=True)
    values = []
    for task in predicates:
        use = frame.predicate.to_numpy() == task
        if not use.any():
            continue
        y = frame.label.to_numpy()[use]
        p = probability[use]
        for k in range(11):
            predicted = p[:, k] if k < 10 else p[:, :10].sum(-1)
            observed = y == k if k < 10 else y != 10
            bucket = np.minimum((predicted*10).astype(int), 9)
            for b in range(10):
                mask = bucket == b
                if mask.any():
                    values.append(dict(label_view=LABEL_VIEW, predicate=task, quantity=f"bin_{k+1}" if k < 10 else "any_switch",
                        bucket=b, n=int(mask.sum()), positives=int(observed[mask].sum()),
                        mean_probability=float(predicted[mask].mean()), frequency=float(observed[mask].mean())))
    return values


def classification_counts(rows, threshold):
    """Horizon detection counts, excluding outcomes censored before the horizon."""
    validate_targets(rows)
    rows = rows[rows.horizon_outcome_known]
    true = rows.event_observed.to_numpy(dtype=bool)
    predicted = rows.switch_probability.to_numpy() >= threshold
    tp, fp = int((true & predicted).sum()), int((~true & predicted).sum())
    fn, tn = int((true & ~predicted).sum()), int((~true & ~predicted).sum())
    def ratio(num, den):
        return num/den if den else None
    return dict(n=len(rows), positive=int(true.sum()), predicted_positive=tp+fp,
        tp=tp, fp=fp, fn=fn, tn=tn, precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        f1_denominator_positive=bool(2*tp+fp+fn),
        fpr=ratio(fp, fp+tn), fdr=ratio(fp, fp+tp),
        prevalence=float(true.mean()) if len(true) else None)


def classification():
    """Fixed-threshold detection diagnostics from frozen probabilities; no threshold fitting."""
    config = json.loads((OUT / "config.json").read_text())
    require_v2_configuration(config)
    names = ["Prior", "GBT", "GBT_primary"] + [name+"_ensemble" for name in config["run_modes"]]
    thresholds = [0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.9]
    columns = ["row_id", "model", "split", "predicate"] + PROB_COLUMNS
    frame = pd.read_parquet(OUT / "predictions.parquet", columns=columns, filters=[
        ("model", "in", names), ("split", "in", ["validation", "test"]),
        ("predicate", "in", config["primary_predicates"])])
    frame = attach_label_view(frame, GraphDataset().index)
    frame = frame[frame.horizon_outcome_known]
    masses = frame[PROB_COLUMNS].to_numpy(dtype=np.float64)
    frame["switch_probability"] = masses[:, :10].sum(-1)/masses.sum(-1)

    overall, per_task = [], []
    for (model, split), group in frame.groupby(["model", "split"], sort=True):
        for threshold in thresholds:
            parts = []
            for predicate, rows in group.groupby("predicate"):
                record = dict(label_view=LABEL_VIEW, model=model, split=split, threshold=threshold, predicate=predicate,
                              **classification_counts(rows, threshold))
                per_task.append(record)
                parts.append(record)
            tasks = pd.DataFrame(parts)
            overall.append(dict(label_view=LABEL_VIEW, model=model, split=split, threshold=threshold,
                **classification_counts(group, threshold), macro_f1=float(tasks.f1.mean()),
                macro_fpr=float(tasks.fpr.mean()), predicates=len(tasks),
                predicates_with_positives=int((tasks.positive > 0).sum())))
    pd.DataFrame(overall).to_csv(OUT / "classification_metrics.csv", index=False)
    pd.DataFrame(per_task).to_csv(OUT / "classification_per_predicate.csv", index=False)
    write_json(OUT / "classification_protocol.json", dict(
        label_view=LABEL_VIEW, positive_class="First opposite binary observed on the five-tick grid within the next 50 ticks",
        score="Sum of the ten first-switch event masses, normalized with no-switch mass",
        decision="score >= threshold", thresholds=thresholds, reference_threshold=0.5,
        threshold_selection="None; descriptive fixed operating points added after the original held-out evaluation",
        scope="Known horizon outcomes in the frozen primary predicate set", metric_selection=METRIC_SELECTION,
        unit="Predicate-instance query window; overlapping windows are not deduplicated event alarms",
        f1="2 TP / (2 TP + FP + FN); zero when the denominator is zero, flagged separately",
        fpr="FP / (FP + TN): fraction of negative queries triggering an alarm",
        fdr="FP / (TP + FP): fraction of alarms that are false; one minus precision",
        aggregation="Pooled confusion counts plus equal-predicate macro F1 and FPR; undefined precision/recall remain null",
        model_or_calibration_changed=False, operational_alarm_policy_selected=False))
    result = pd.DataFrame(overall)
    print(result[(result.split == "test") & (result.threshold == 0.5)][
        ["model", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "macro_f1", "fpr", "macro_fpr", "fdr"]].to_string(index=False), flush=True)


def analyze(device):
    config = json.loads((OUT / "config.json").read_text())
    require_v2_configuration(config)
    dataset = GraphDataset()
    predictions = attach_label_view(pd.read_parquet(OUT / "predictions.parquet"), dataset.index)
    base_models = {"Prior", "GBT", "GBT_primary", *config["run_modes"]}
    predictions = predictions[predictions.model.isin(base_models) & predictions.seed.isin(config["optimization_seeds"])].copy()
    extra = []
    phase = np.array(json.loads((OUT / "phase_prior.json").read_text())["probabilities"])
    for split, frame in dataset.index.groupby("split"):
        types = np.array([dataset.vocab["predicates"].index(p) if p in dataset.vocab["predicates"] else 0 for p in frame.predicate])
        p = phase[types, frame.current.to_numpy(), frame.anchor.to_numpy()//100]
        slim = frame[["row_id", "profile", "split", "predicate", "anchor", "current", "observed_opposite", *TARGET_COLUMNS]].copy()
        slim["model"], slim["seed"] = "Phase", 0
        for k in range(11):
            slim[f"p{k}"] = p[:, k]
        extra.append(slim)
    batch_cpu = build_batches(dataset, "cpu", include_test=True)
    cpa = joblib.load(OUT / "models/CPA.joblib")
    for split, (x, frame) in cpa_features(dataset, batch_cpu).items():
        p = cpa_predict(cpa, x)
        slim = frame[["row_id", "profile", "split", "predicate", "anchor", "current", "observed_opposite", *TARGET_COLUMNS]].copy()
        slim["model"], slim["seed"] = "CPA", 0
        for k in range(11):
            slim[f"p{k}"] = p[:, k]
        extra.append(slim)
    # Ensembles diagnose optimization variability; three seeds are not three scenario families.
    ensembles = []
    for mode in config["run_modes"]:
        base = predictions[predictions.model == mode]
        mean = base.groupby("row_id", sort=True)[PROB_COLUMNS].mean()
        spread = base.assign(ps=base[PROB_COLUMNS[:10]].sum(axis=1)).groupby("row_id").ps.std()
        slim = base[base.seed == 0].set_index("row_id").sort_index().copy()
        slim[PROB_COLUMNS] = mean
        slim["switch_probability_seed_sd"] = spread
        slim["seed"], slim["model"] = -1, mode+"_ensemble"
        ensembles.append(slim.reset_index())
    combined = pd.concat([predictions, *extra, *ensembles], ignore_index=True)
    combined.to_parquet(OUT / "predictions.parquet", index=False)
    metric_rows, per_task, calibration, strata, bounds = [], [], [], [], []
    for (model, seed, split), frame in combined.groupby(["model", "seed", "split"], sort=True):
        p = frame[PROB_COLUMNS].to_numpy()
        for scope, predicates in (("primary", config["primary_predicates"]), ("all", config["all_train_predicates"])):
            metric, table = scores(frame, p, predicates)
            metric_rows.append(dict(model=model, seed=seed, split=split, scope=scope, **metric))
            per_task.extend(dict(model=model, seed=seed, split=split, scope=scope, **r) for r in table)
        if seed in (0, -1):
            calibration.extend(dict(model=model, split=split, **r) for r in reliability(frame, p, config["primary_predicates"]))
        for (profile, current), group in frame.groupby(["profile", "current"]):
            m, _ = scores(group, group[PROB_COLUMNS].to_numpy(), config["primary_predicates"])
            strata.append(dict(model=model, seed=seed, split=split, profile=profile, current=current, **m))
        # Binary completion bounds retain only events observed before first censoring.
        for pred, group in frame[frame.predicate.isin(config["primary_predicates"])].groupby("predicate"):
            masses = group[PROB_COLUMNS].to_numpy().astype(np.float64)
            masses /= masses.sum(-1, keepdims=True)
            prob = masses[:, :10].sum(-1)
            tiny = np.finfo(np.float64).tiny
            l0, l1 = -np.log(masses[:, 10].clip(tiny)), -np.log(prob.clip(tiny))
            known = group.horizon_outcome_known.to_numpy()
            outcome = group.event_observed.to_numpy()
            loss = np.where(outcome, l1, l0)
            lower = np.where(known, loss, np.minimum(l0, l1))
            upper = np.where(known, loss, np.maximum(l0, l1))
            bounds.append(dict(model=model, seed=seed, split=split, predicate=pred, n=len(group),
                horizon_unknown=int((~group.horizon_outcome_known).sum()), metric_selection=METRIC_SELECTION, binary_nll_lower=float(lower.mean()), binary_nll_upper=float(upper.mean())))
    pd.DataFrame(metric_rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(per_task).to_csv(OUT / "per_predicate_metrics.csv", index=False)
    pd.DataFrame(calibration).to_csv(OUT / "reliability.csv", index=False)
    pd.DataFrame(strata).to_csv(OUT / "stratified_metrics.csv", index=False)
    pd.DataFrame(bounds).to_csv(OUT / "missing_label_bounds.csv", index=False)
    selection = dataset.index[(dataset.index.split == "test") & dataset.index.likelihood_valid & dataset.index.predicate.isin(config["primary_predicates"])]
    chosen = pd.concat([g.sample(min(8, len(g)), random_state=4242) for _, g in selection.groupby("predicate")])
    deletion = []
    for name in config["supplemental_baselines"]["deletion"]["models"]:
        model = SwitchModel(dataset.vocab, "G", config["width"]).to(device)
        saved = torch.load(OUT / "models" / (name+"_seed0.pt"), map_location=device, weights_only=False)
        model.load_state_dict(saved["state"])
        model.eval()
        with torch.no_grad():
            for (profile, anchor), rows in chosen.groupby(["profile", "anchor"]):
                x, _, all_rows = dataset.get((profile, int(anchor)), device)
                full = model(x)
                full_loss = censored_nll(full, all_rows.event_observed.to_numpy(),
                    all_rows.event_bin.to_numpy(), all_rows.observed_bins.to_numpy())
                positions = {row_id: i for i, row_id in enumerate(all_rows.row_id)}
                for row in rows.itertuples():
                    j = positions[row.row_id]
                    kept = delete_nonquery(x, j)
                    deleted_loss = censored_nll(model(kept), [row.event_observed], [row.event_bin], [row.observed_bins])
                    deletion.append(dict(label_view=LABEL_VIEW, model=name, row_id=row.row_id, predicate=row.predicate, label=row.label,
                        full_nll=float(full_loss[j]), deleted_nll=float(deleted_loss[0]),
                        full_entities=x["physical"].shape[1], retained_entities=kept["physical"].shape[1]))
    pd.DataFrame(deletion).to_csv(OUT / "nonquery_deletion.csv", index=False)
    validity = dataset.index.assign(positive_evidence=dataset.index.observed_opposite).groupby(
        ["split", "predicate", "positive_evidence"]).likelihood_valid.agg(["size", "sum", "mean"])
    validity.reset_index().to_csv(OUT / "validity_by_evidence.csv", index=False)
    print(json.dumps(dict(deletion_queries=len(deletion), calibration_rows=len(calibration),
        warning="Overlapping queries and one held-out family: no independent-family confidence interval or guaranteed calibration.")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["develop", "freeze", "analyze", "curves", "classification"])
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase == "develop":
        supplement_development()
    elif args.phase == "freeze":
        freeze_evaluation(args.device)
    elif args.phase == "curves":
        tree_curves()
    elif args.phase == "classification":
        classification()
    else:
        analyze(args.device)
