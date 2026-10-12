"""Reproducible development, frozen held-out evaluation and probability diagnostics."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib

from .data import GraphDataset, OUT, write_json, validate_targets, person_period_targets, METRIC_SELECTION, TARGET_COLUMNS, LABEL_VIEW, require_v2_configuration
from .model import build_model, censored_nll


def make_prior(dataset):
    train = dataset.index[(dataset.index.split == "train") & dataset.index.likelihood_valid]
    names = dataset.vocab["predicates"]
    events = np.full((len(names), 2, 10), 0.5, np.float64)
    risk = np.broadcast_to(0.5 * np.arange(11, 1, -1), events.shape).copy()
    person, period, target = person_period_targets(train)
    types = np.array([names.index(p) for p in train.predicate], dtype=np.int64)
    current = train.current.to_numpy(dtype=np.int64)
    np.add.at(events, (types[person], current[person], period), target)
    np.add.at(risk, (types[person], current[person], period), 1)
    hazard = events / risk
    prior = hazard_probabilities(hazard.reshape(-1, 10)).reshape(len(names), 2, 11)
    return prior, np.log(hazard) - np.log1p(-hazard)


def scores(frame, probability, predicates):
    validate_targets(frame)
    selected = frame.likelihood_valid.to_numpy() & frame.predicate.isin(predicates).to_numpy()
    f = frame.loc[selected].copy()
    p = probability[selected].astype(np.float64)
    if not len(f):
        return {}, []
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(-1) <= 0).any():
        raise ValueError("invalid predictive probabilities")
    p /= p.sum(-1, keepdims=True)
    event = f.event_observed.to_numpy(dtype=bool)
    bins = f.event_bin.to_numpy(dtype=np.int64)
    observed = f.observed_bins.to_numpy(dtype=np.int64)
    known = f.horizon_outcome_known.to_numpy(dtype=bool)
    ps = p[:, :10].sum(-1)
    tiny = np.finfo(np.float64).tiny
    tail_mass = np.array([mass[m:].sum() for mass, m in zip(p, observed)])
    likelihood = np.where(event, p[np.arange(len(p)), bins.clip(0)], tail_mass)
    f["nll"] = -np.log(likelihood.clip(tiny))
    f["binary_nll"] = np.where(known, -np.log(np.where(event, ps, p[:, 10]).clip(tiny)), np.nan)
    f["brier"] = np.where(known, (ps-event)**2, np.nan)
    f["probability"] = ps
    conditional_mean = np.divide(p[:, :10] @ (np.arange(1, 11)*5), ps,
                                 out=np.full_like(ps, np.nan), where=ps > 0)
    f["time_mae"] = np.where(event, abs(conditional_mean-(bins+1)*5), np.nan)
    y = f.label.to_numpy(dtype=np.int64)
    target_cdf = np.arange(10)[None, :] >= y[:, None]
    f["rps"] = np.where(known, ((p[:, :10].cumsum(-1)-target_cdf)**2).mean(-1), np.nan)

    def mean_or_none(value):
        valid = pd.Series(value).dropna()
        return float(valid.mean()) if len(valid) else None

    per_task = []
    for pred, group in f.groupby("predicate", sort=True):
        horizon = group[group.horizon_outcome_known]
        binary = horizon.event_observed.to_numpy(dtype=bool)
        predicted = horizon.probability.to_numpy()
        ece = 0.0 if len(horizon) else None
        bucket = np.minimum((predicted*10).astype(int), 9)
        for b in range(10):
            mask = bucket == b
            if mask.any():
                ece += mask.mean()*abs(binary[mask].mean()-predicted[mask].mean())
        per_task.append(dict(predicate=pred, n=len(group), horizon_n=len(horizon),
            positive=int(group.event_observed.sum()), partial_censored=int((~group.horizon_outcome_known).sum()),
            nll=float(group.nll.mean()), binary_nll=mean_or_none(group.binary_nll),
            brier=mean_or_none(group.brier), rps=mean_or_none(group.rps), ece=ece,
            probability=mean_or_none(predicted), rate=mean_or_none(binary),
            average_precision=float(average_precision_score(binary, predicted)) if binary.any() else None,
            auroc=float(roc_auc_score(binary, predicted)) if len(np.unique(binary)) == 2 else None,
            time_mae=mean_or_none(group.time_mae), label_view=LABEL_VIEW, metric_selection=METRIC_SELECTION))
    table = pd.DataFrame(per_task)
    aggregate = {key: mean_or_none(table[key]) for key in
                 ("nll", "binary_nll", "brier", "rps", "ece", "probability", "rate", "average_precision", "time_mae")}
    aggregate.update(label_view=LABEL_VIEW, n=len(f), horizon_n=int(known.sum()), positive=int(event.sum()),
        partial_censored=int((~known).sum()), predicates=len(table), metric_selection=METRIC_SELECTION)
    return aggregate, per_task


def infer(model, batches, split):
    model.eval()
    frames, predictions = [], []
    with torch.no_grad():
        for x, _, rows in batches[split]:
            predictions.append(model(x).double().exp().cpu().numpy())
            frames.append(rows)
    return pd.concat(frames), np.concatenate(predictions)


def task_macro_weights(training, rows):
    """Give each supported task equal weight, with zero weight for zero-bin censoring."""
    validate_targets(training)
    validate_targets(rows)
    counts = training.loc[training.likelihood_valid, "predicate"].value_counts()
    if not len(counts):
        raise ValueError("training objective has no likelihood-supported tasks")
    return np.array([1/counts[p]/len(counts) if valid and p in counts else 0
        for p, valid in zip(rows.predicate, rows.likelihood_valid)], dtype=np.float32)


def train_one(mode, lr, seed, dataset, batches, config, prior_logits):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = batches["train"][0][0]["physical"].device
    model = build_model(dataset.vocab, mode, config["width"]).to(device)
    model.prior_logits.copy_(torch.tensor(prior_logits, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    training = dataset.index[(dataset.index.split == "train") & dataset.index.likelihood_valid]
    if config.get("objective_scope") == "primary":
        training = training[training.predicate.isin(config["primary_predicates"])]
    weights = [torch.as_tensor(task_macro_weights(training, rows), device=device)
               for _, _, rows in batches["train"]]
    best, best_state, curves, stale = float("inf"), None, [], 0
    begin = time.perf_counter()
    for epoch in range(config["epochs"] + 1):
        train_loss, grad_max = 0.0, 0.0
        if epoch:
            model.train()
            for j in np.random.permutation(len(batches["train"])):
                if not weights[j].any():
                    continue
                x, _, rows = batches["train"][j]
                optimizer.zero_grad(set_to_none=True)
                logp = model(x)
                loss_values = censored_nll(logp, rows.event_observed.to_numpy(),
                    rows.event_bin.to_numpy(), rows.observed_bins.to_numpy())
                loss = (loss_values * weights[j]).sum() * len(batches["train"])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                grad_max = max(grad_max, float(norm))
                optimizer.step()
                train_loss += float(loss.detach()) / len(batches["train"])
        train_frame, train_probability = infer(model, batches, "train")
        validation_frame, validation_probability = infer(model, batches, "validation")
        tm, _ = scores(train_frame, train_probability, config["primary_predicates"])
        vm, _ = scores(validation_frame, validation_probability, config["primary_predicates"])
        curves.append(dict(label_view=LABEL_VIEW, epoch=epoch, train_objective=train_loss if epoch else None,
            train_nll=tm["nll"], validation_nll=vm["nll"], train_brier=tm["brier"],
            validation_brier=vm["brier"], gradient_norm_max=grad_max,
            elapsed_seconds=time.perf_counter()-begin))
        if vm["nll"] < best - 1e-5:
            best, best_state, stale = vm["nll"], deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch % 5 == 0 or stale >= config["patience"]:
            print(json.dumps(dict(model=mode, lr=lr, seed=seed, **curves[-1])), flush=True)
        if stale >= config["patience"]:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, curves, best


def engineered(x):
    """Graph-derived, ID-free summaries for the person-period boosted baseline."""
    x = {k: v.cpu().numpy() for k, v in x.items()}
    query = x["query"]
    qtruth = x["truth"][:, query]
    n = len(query)
    predicate_count = int(x["predicate_vocab_size"])
    # Predicate IDs below are contract type codes, not entity/episode identifiers.
    typ = x["predicate_type"][query]
    truth_onehot = np.stack([(qtruth == i).mean(0) for i in range(5)], -1)
    binary_pair = np.isin(qtruth[:-1], [1, 2]) & np.isin(qtruth[1:], [1, 2])
    flips = (qtruth[:-1] != qtruth[1:]) & binary_pair
    last_age = np.where(flips, np.arange(20, 0, -1)[:, None]/20, 1.05).min(0)
    phys = x["physical"][:, x["pointers"][query]]
    masks = x["physical_present"][:, x["pointers"][query]] & x["role_present"][query][None]
    recent = np.concatenate([phys[-1], masks[-1, :, :, None]], -1).reshape(n, -1)
    mean = (phys*masks[..., None]).sum(0) / masks.sum(0).clip(1)[..., None]
    delta = ((phys[-1]-phys[0]) * (masks[-1]&masks[0])[..., None]).reshape(n, -1)
    active = np.isin(x["truth"], [1, 2, 3])
    hist = np.zeros(predicate_count*3, np.float32)
    for k in range(predicate_count):
        select = x["predicate_type"] == k
        denom = max(int(active[:, select].sum()), 1)
        for value in (1, 2, 3):
            hist[3*k+value-1] = (x["truth"][:, select] == value).sum()/denom
    global_physical = x["physical"][x["physical_present"]]
    context = np.r_[hist, global_physical.mean(0), global_physical.std(0)]
    return np.concatenate([np.eye(predicate_count, dtype=np.float32)[typ], qtruth[-1, :, None]-1,
        truth_onehot, flips.sum(0)[:, None]/20, last_age[:, None], recent, mean.reshape(n, -1), delta,
        np.broadcast_to(context, (n, len(context)))], -1).astype(np.float32)


def hazard_probabilities(hazard):
    survive = np.cumprod(1-hazard, axis=-1)
    event = hazard * np.concatenate([np.ones((len(hazard), 1)), survive[:, :-1]], -1)
    return np.concatenate([event, survive[:, -1:]], -1)


def tree_features(batches, vocab):
    result = {}
    for split, bs in batches.items():
        ff, rr = [], []
        for x, _, rows in bs:
            ff.append(engineered({**x, "predicate_vocab_size": torch.tensor(len(vocab["predicates"]))}))
            rr.append(rows)
        result[split] = (np.concatenate(ff), pd.concat(rr))
    return result


def train_trees(features, config):
    xx, frame = features["train"]
    select = frame.likelihood_valid.to_numpy()
    if config.get("objective_scope") == "primary":
        select &= frame.predicate.isin(config["primary_predicates"]).to_numpy()
    xx, frame = xx[select], frame.loc[select]
    counts = frame.predicate.value_counts()
    weights = np.array([1/counts[p] for p in frame.predicate])
    person, period, Y = person_period_targets(frame)
    X = np.c_[xx[person], period/9]
    W = weights[person]
    best, records, winner = float("inf"), [], None
    for leaves in (7, 15):
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=leaves, learning_rate=0.06,
            l2_regularization=1.0, early_stopping=False, random_state=0)
        begin = time.perf_counter()
        model.fit(X, Y, sample_weight=W / W.mean())
        prob = tree_predict(model, features["validation"][0])
        score, _ = scores(features["validation"][1], prob, config["primary_predicates"])
        record = dict(leaves=leaves, seconds=time.perf_counter()-begin, validation=score)
        records.append(record)
        print(json.dumps(dict(model="GBT", **record)), flush=True)
        if score["nll"] < best:
            best, winner = score["nll"], model
    return winner, records


def tree_predict(model, x):
    hazards = np.stack([model.predict_proba(np.c_[x, np.full(len(x), k/9)])[:, 1] for k in range(10)], -1)
    return hazard_probabilities(hazards)


def build_batches(dataset, device, include_test=False):
    batches = {"train": [], "validation": []}
    if include_test:
        batches["test"] = []
    for sample in dataset.samples:
        rows = dataset.index.iloc[dataset.rows[sample]]
        split = rows.split.iloc[0]
        if split in batches:
            batches[split].append(dataset.get(sample, device))
    return batches


def develop(device):
    config = json.loads((OUT / "config.json").read_text())
    require_v2_configuration(config)
    dataset = GraphDataset()
    prior, prior_logits = make_prior(dataset)
    write_json(OUT / "prior.json", dict(label_view=LABEL_VIEW, probabilities=prior.tolist(), hazard_logits=prior_logits.tolist(), pseudo_count_per_bin=0.5))
    batches = build_batches(dataset, device)
    run_dir = OUT / "models"
    run_dir.mkdir(exist_ok=True)
    curve_rows, selection = [], {}
    for mode in config["models"]:
        trials, best = [], float("inf")
        for lr in config["learning_rates"]:
            model, curves, score = train_one(mode, lr, 0, dataset, batches, config, prior_logits)
            curve_rows.extend(dict(model=mode, lr=lr, seed=0, **row) for row in curves)
            trials.append(dict(lr=lr, validation_nll=score, epochs=len(curves)-1))
            if score < best:
                best = score
                best_lr = lr
                torch.save(dict(state=model.state_dict(), mode=mode, lr=lr, seed=0, width=config["width"]), run_dir / (mode+"_seed0.pt"))
            del model
            pd.DataFrame(curve_rows).to_csv(OUT / "learning_curves.csv", index=False)
        selection[mode] = dict(learning_rate=best_lr, validation_nll=best, trials=trials)
        write_json(OUT / "selection.json", selection)
        for seed in config["optimization_seeds"][1:]:
            model, curves, score = train_one(mode, best_lr, seed, dataset, batches, config, prior_logits)
            curve_rows.extend(dict(model=mode, lr=best_lr, seed=seed, **row) for row in curves)
            torch.save(dict(state=model.state_dict(), mode=mode, lr=best_lr, seed=seed, width=config["width"]), run_dir / f"{mode}_seed{seed}.pt")
            del model
            pd.DataFrame(curve_rows).to_csv(OUT / "learning_curves.csv", index=False)
    features = tree_features(batches, dataset.vocab)
    tree, trials = train_trees(features, config)
    joblib.dump(tree, run_dir / "GBT.joblib")
    write_json(OUT / "tree_selection.json", trials)
    write_json(OUT / "development_complete.json", dict(models=config["models"], seeds=config["optimization_seeds"],
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0,
        label_view=LABEL_VIEW, selection_source="validation only; no held-out test predictions computed"))


def refine(device):
    assert not (OUT / "metrics.csv").exists(), "Development refinement must precede held-out evaluation"
    config = json.loads((OUT / "config.json").read_text())
    require_v2_configuration(config)
    dataset = GraphDataset()
    _, prior_logits = make_prior(dataset)
    batches = build_batches(dataset, device)
    curve_rows = pd.read_csv(OUT / "learning_curves.csv").to_dict("records")
    selection = json.loads((OUT / "selection.json").read_text())
    base_modes = config["models"]
    runs = [("Offset", "Offset", "all")] + [(m+"_primary", m, "primary") for m in base_modes]
    config["refinement"] = dict(reason="Validation curves showed weak primary-task fitting and G matched M2; isolate prior correction and equalize optimization with the frozen primary estimand.",
        before_test=True, modes=[r[0] for r in runs], changed="loss task population only; split, labels, input, width, trials, epochs and early stopping fixed",
        objective_weights="15 primary tasks: 1/15 each; initial 60-task objective: 1/60 each")
    config["run_modes"] = {m: m for m in base_modes} | {name: mode for name, mode, _ in runs}
    write_json(OUT / "config.json", config)
    for name, mode, scope in runs:
        local = dict(config, objective_scope=scope)
        trials, best = [], float("inf")
        for lr in config["learning_rates"]:
            model, curves, score = train_one(mode, lr, 0, dataset, batches, local, prior_logits)
            curve_rows.extend(dict(model=name, lr=lr, seed=0, **row) for row in curves)
            trials.append(dict(lr=lr, validation_nll=score, epochs=len(curves)-1))
            if score < best:
                best, best_lr = score, lr
                torch.save(dict(state=model.state_dict(), mode=mode, lr=lr, seed=0, width=config["width"], objective_scope=scope), OUT / "models" / (name+"_seed0.pt"))
            del model
            pd.DataFrame(curve_rows).to_csv(OUT / "learning_curves.csv", index=False)
        selection[name] = dict(learning_rate=best_lr, validation_nll=best, trials=trials, objective_scope=scope)
        write_json(OUT / "selection.json", selection)
        for seed in config["optimization_seeds"][1:]:
            model, curves, _ = train_one(mode, best_lr, seed, dataset, batches, local, prior_logits)
            curve_rows.extend(dict(model=name, lr=best_lr, seed=seed, **row) for row in curves)
            torch.save(dict(state=model.state_dict(), mode=mode, lr=best_lr, seed=seed, width=config["width"], objective_scope=scope), OUT / "models" / f"{name}_seed{seed}.pt")
            del model
            pd.DataFrame(curve_rows).to_csv(OUT / "learning_curves.csv", index=False)
    write_json(OUT / "refinement_summary.json", dict(validation_only=True, runs=[r[0] for r in runs],
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0))


def evaluate(device):
    config = json.loads((OUT / "config.json").read_text())
    require_v2_configuration(config)
    assert (OUT / "development_complete.json").exists()
    assert (OUT / "evaluation_protocol.json").exists()
    dataset = GraphDataset()
    batches = build_batches(dataset, device, include_test=True)
    prior = np.array(json.loads((OUT / "prior.json").read_text())["probabilities"])
    records, task_records, prediction_frames = [], [], []

    def record(name, seed, split, frame, prob, parameters=0):
        assert np.isfinite(prob).all() and np.allclose(prob.sum(-1), 1, atol=3e-6)
        assert np.all(np.diff(prob[:, :10].cumsum(-1), axis=-1) >= -1e-8)
        for scope, predicates in [("primary", config["primary_predicates"]), ("all", config["all_train_predicates"])]:
            aggregate, per_task = scores(frame, prob, predicates)
            records.append(dict(model=name, seed=seed, split=split, scope=scope, parameters=parameters, **aggregate))
            task_records.extend(dict(model=name, seed=seed, split=split, scope=scope, **r) for r in per_task)
        slim = frame[["row_id", "profile", "split", "predicate", "anchor", "current", "observed_opposite", *TARGET_COLUMNS]].copy()
        slim["model"], slim["seed"] = name, seed
        for k in range(11):
            slim[f"p{k}"] = prob[:, k]
        prediction_frames.append(slim)

    for split in batches:
        frame = pd.concat([r for _, _, r in batches[split]])
        types = np.array([dataset.vocab["predicates"].index(p) if p in dataset.vocab["predicates"] else 0 for p in frame.predicate])
        prob = prior[types, frame.current.to_numpy()]
        record("Prior", 0, split, frame, prob)
    for name, mode in config["run_modes"].items():
        for seed in config["optimization_seeds"]:
            saved = torch.load(OUT / "models" / f"{name}_seed{seed}.pt", map_location=device, weights_only=False)
            model = build_model(dataset.vocab, mode, config["width"]).to(device)
            model.load_state_dict(saved["state"])
            count = sum(p.numel() for p in model.parameters())
            for split in batches:
                frame, prob = infer(model, batches, split)
                record(name, seed, split, frame, prob, count)
            del model
            print(json.dumps(dict(evaluated=name, seed=seed)), flush=True)
    features = tree_features(batches, dataset.vocab)
    for name in ("GBT", "GBT_primary"):
        tree = joblib.load(OUT / "models" / (name+".joblib"))
        for split, (x, frame) in features.items():
            record(name, 0, split, frame, tree_predict(tree, x))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(task_records).to_csv(OUT / "per_predicate_metrics.csv", index=False)
    pd.concat(prediction_frames).to_parquet(OUT / "predictions.parquet", index=False)
    write_json(OUT / "evaluation_summary.json", dict(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0,
        label_view=LABEL_VIEW, distribution_checks="finite, normalized within 3e-6, cumulative event mass monotone for every saved model prediction"))
    print(metrics[(metrics.scope == "primary") & (metrics.split == "test")].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["develop", "refine", "evaluate"])
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False)
    if args.phase == "develop":
        develop(args.device)
    elif args.phase == "refine":
        refine(args.device)
    else:
        evaluate(args.device)
