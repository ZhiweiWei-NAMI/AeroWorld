"""Prior-enabled Ours main method and controlled no-prior sampling ablations.

Full TRAIN/VALID cohorts, sealed TEST, fixedF10>=.5, no threshold selection.
Run the authorized implementation directly; assess results after execution.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace

import numpy as np
import torch

from .events import EventConstraint
from .communication_cv_inference import CumulativeBranchLogs
from .communication_cv_models import (NumericHistoryContext,
    HistoryTrendTrajectoryHead, ControlledSamplingTrajectoryHead,
    IndependentEventHeads, linear_forecast_bins, parameter_manifest)

ROOT = Path(__file__).resolve().parents[3]
RUN_REL = Path("Dataset/world_model/runs/communication_component_cv")
NUMERIC_CACHE_PATH = ROOT / RUN_REL / "ordinary_K8_quadratic_numeric_cache.npz"
RESULTS = ROOT / RUN_REL / "current_comparison/results"
CONTROLLED_SAMPLING_METHODS = ("OursSamplingHybrid", "OursNoSamplingHybrid")
PRIOR_OURS_METHODS = ("OursHybrid", "OursRPS", "OursCDFBCE")
HYBRID_SNAPSHOT_METHODS = ("OursHybrid", *CONTROLLED_SAMPLING_METHODS)
METHODS = (*PRIOR_OURS_METHODS, *CONTROLLED_SAMPLING_METHODS, "Bmean", "CatBoost", "TabM", "RuleMC", "PrefixLinear")
LOSSES = {"OursHybrid": "hybrid", "OursRPS": "rps", "OursCDFBCE": "cumulative_bce",
          "OursSamplingHybrid": "hybrid", "OursNoSamplingHybrid": "hybrid"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def seed_rng(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)


def cumulative_logs(log_pmf):
    return CumulativeBranchLogs(torch.logcumsumexp(log_pmf, -1)[..., :10],
        torch.logcumsumexp(log_pmf.flip(-1), -1).flip(-1)[..., 1:])


def load_numeric_cache(path=NUMERIC_CACHE_PATH, *, device):
    """Read only the accepted immutable cache; precompute split tensors once."""
    splits = {}
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        context_model = NumericHistoryContext().to(device)
        for split, expected in (("train", 2935), ("validation", 1928)):
            records = metadata["splits"][split]
            if len(records["queries"]) != expected:
                raise ValueError(f"{split} must contain the complete {expected}-window cohort")
            data = {}
            for name in ("numeric_history", "numeric_branches", "raw_history", "raw_branches", "governed_thresholds", "targets", "known", "predicate_trajectories"):
                dtype = torch.bool if name in ("known", "predicate_trajectories") else torch.float64
                data[name] = torch.as_tensor(np.array(archive[f"{split}_{name}"], copy=True), dtype=dtype, device=device)
            events = [dict(zip(("L", "P"), (EventConstraint(kind, step) for kind, step in row))) for row in records["constraints"]]
            compatible = torch.zeros((expected, 2, 11), dtype=torch.bool, device=device)
            for i, row in enumerate(events):
                for column, predicate in enumerate(("L", "P")):
                    event = row[predicate]
                    if event.kind == "first_flip":
                        compatible[i, column, event.step-1] = True
                    elif event.kind == "right_censored":
                        compatible[i, column, event.step:] = True
                    elif event.kind == "none":
                        compatible[i, column, 10] = True
                    else:
                        raise ValueError(f"unknown cached event kind {event.kind}")
            h, raw, thresholds = data["numeric_history"], data["raw_history"], data["governed_thresholds"]
            with torch.no_grad():
                context = context_model(h, raw, thresholds)
            mean = data["numeric_branches"].mean(1, keepdim=True)
            mean_truth = data["raw_branches"].mean(1, keepdim=True) > thresholds[:, None, None]
            tabular = torch.cat((context.flatten(1), mean[:, 0].flatten(1), mean_truth[:, 0].to(h.dtype).flatten(1)), -1)
            components = [query["component"] for query in records["queries"]]
            eligible = {}
            for component in sorted(set(components)):
                eligible[component] = np.asarray([i for i, group in enumerate(components) if group == component and any(e.kind != "right_censored" or e.step > 0 for e in events[i].values())], dtype=np.int64)
                if not len(eligible[component]):
                    raise ValueError(f"component {component} has no informative windows")
            splits[split] = SimpleNamespace(**data, context=context, mean=mean, mean_truth=mean_truth,
                tabular=tabular, tabular_cpu=tabular.cpu().numpy(), events=events, compatible=compatible,
                components=components, eligible=eligible, window_ids=np.asarray([q["window_id"] for q in records["queries"]]), size=expected)
    return SimpleNamespace(splits=splits, metadata=metadata, path=str(path))


def batch_sequence(split, seed, updates=2000):
    rng = np.random.default_rng(seed)
    groups = tuple(split.eligible.values())
    choices = rng.integers(len(groups), size=(updates, 64))
    batches = np.empty_like(choices)
    for group, indices in enumerate(groups):
        mask = choices == group
        batches[mask] = rng.choice(indices, size=int(mask.sum()), replace=True)
    return batches


def marginal_log_pmf(model, method, split, indices):
    h = split.numeric_history[indices]
    raw, thresholds = split.raw_history[indices], split.governed_thresholds[indices]
    if method == "OursNoSamplingHybrid":
        branch = model(h, raw_history=raw, thresholds=thresholds)
        return torch.logsumexp(branch, 2) - math.log(8)
    if method in (*PRIOR_OURS_METHODS, "OursSamplingHybrid"):
        branch = model(h, split.numeric_branches[indices], split.predicate_trajectories[indices], raw_history=raw, thresholds=thresholds)
        return torch.logsumexp(branch, 2) - math.log(8)
    if method == "Bmean":
        b = split.mean[indices]
        truth = split.mean_truth[indices]
        branch = torch.log_softmax(model(h, b, truth, split.context[indices]), -1)
        return torch.logsumexp(branch, 2) - math.log(branch.shape[2])
    if method == "TabM":
        from .communication_cv_tabm import tabm_log_pmf
        return tabm_log_pmf(model, split.tabular[indices])
    if method == "RuleMC":
        truth = split.predicate_trajectories[indices]
        opposite = truth[:, :, 1:] != truth[:, :1, :1]
        first = opposite.long().argmax(2)
        bins = torch.where(opposite.any(2), first, torch.full_like(first, 10))
        return torch.nn.functional.one_hot(bins.permute(0, 2, 1), 11).double().mean(2).log()
    if method == "PrefixLinear":
        bins = linear_forecast_bins(raw, thresholds)
        return torch.nn.functional.one_hot(bins, 11).double().log()
    raise ValueError(f"unknown numeric method {method}")


def censored_nll(log_pmf, compatible):
    return -torch.logsumexp(torch.where(compatible, log_pmf, -torch.inf), -1).mean()


def numeric_cumulative_loss(log_pmf, targets, known, mode):
    """Same masked Hybrid/RPS/CDF-BCE law using deterministic log scans."""
    logs = cumulative_logs(log_pmf)
    normalizer = torch.logaddexp(logs.log_cdf, logs.log_survival)
    cdf = (logs.log_cdf-normalizer).exp()
    squared_mask, bce_mask = known.clone(), known.clone()
    if mode == "hybrid":
        squared_mask[..., 9] = False
        bce_mask[..., :9] = False
    elif mode == "rps":
        bce_mask[:] = False
    elif mode == "cumulative_bce":
        squared_mask[:] = False
    else:
        raise ValueError(mode)
    zero = log_pmf.exp().sum()*0
    squared = (cdf[squared_mask]-targets[squared_mask]).square().mean() if bool(squared_mask.any()) else zero
    bce = zero
    if bool(bce_mask.any()):
        chosen = torch.where(targets[bce_mask].bool(), logs.log_cdf[bce_mask], logs.log_survival[bce_mask])
        if bool(torch.isneginf(chosen).any()):
            raise FloatingPointError("known cumulative BCE target has zero predicted probability")
        bce = -chosen.mean()
    return squared + .5*bce if mode == "hybrid" else squared+bce


def whole_cohort_loss_snapshot(cache, model, method, step, device):
    """Fixed full-cohort known-cell objectives; no optimization or selection."""
    training = model.training
    model.eval()
    snapshot = dict(step=step)
    with torch.no_grad():
        for name, split in cache.splits.items():
            log_pmf = torch.cat([marginal_log_pmf(model, method, split,
                torch.arange(start, min(start+64, split.size), device=device))
                for start in range(0, split.size, 64)])
            snapshot[name] = float(numeric_cumulative_loss(log_pmf, split.targets, split.known, LOSSES[method]))
    model.train(training)
    print(json.dumps(dict(phase="whole_cohort_loss", method=method, pid=os.getpid(),
                          physical_GPU=os.environ["CUDA_VISIBLE_DEVICES"], **snapshot)), flush=True)
    return snapshot


def train_method(cache, method, seed, device, *, checkpoint_path=None):
    from .communication_cv_catboost import CatBoostHazardModel
    from .communication_cv_tabm import NumericTabM, make_tabm_optimizer, tabm_member_censored_nll, tabm_parameter_manifest
    seed_rng(seed)
    split = cache.splits["train"]
    started = time.perf_counter()
    print(json.dumps(dict(phase="train_start", method=method, seed=seed, pid=os.getpid(), physical_GPU=os.environ["CUDA_VISIBLE_DEVICES"], step=0)), flush=True)
    if method in ("RuleMC", "PrefixLinear"):
        return None, dict(optimizer_updates=0, trainable_parameters=0, training_seconds=0)
    if method == "CatBoost":
        model = CatBoostHazardModel.fit(split.tabular_cpu, split.events, split.components, seed=seed, iterations=2000)
        return model, dict(optimizer_updates=None, trees_per_head=2000, native_metadata=model.metadata,
                           training_seconds=time.perf_counter()-started)
    if method in PRIOR_OURS_METHODS:
        model = HistoryTrendTrajectoryHead(tuple(cache.metadata["numeric_keys"])).to(device=device, dtype=torch.float64)
    elif method in CONTROLLED_SAMPLING_METHODS:
        source = HistoryTrendTrajectoryHead(tuple(cache.metadata["numeric_keys"]))
        model = ControlledSamplingTrajectoryHead(source,
            sampling_enabled=method == "OursSamplingHybrid").to(device=device, dtype=torch.float64)
        del source
    elif method == "Bmean":
        model = IndependentEventHeads().to(device=device, dtype=torch.float64)
    elif method == "TabM":
        model = NumericTabM().to(device)
    else:
        raise ValueError(method)
    print(json.dumps(dict(phase="model_initialized", method=method, seed=seed, pid=os.getpid(),
        physical_GPU=os.environ["CUDA_VISIBLE_DEVICES"],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()))), flush=True)
    optimizer = (make_tabm_optimizer(model) if method == "TabM" else
                 torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.003, weight_decay=.0001))
    model.train()
    losses = []
    snapshots = []
    if method in HYBRID_SNAPSHOT_METHODS:
        snapshots.append(whole_cohort_loss_snapshot(cache, model, method, 0, device))
    for step, rows in enumerate(batch_sequence(split, seed), 1):
        indices = torch.as_tensor(rows, device=device)
        optimizer.zero_grad(set_to_none=True)
        if method == "TabM":
            loss = tabm_member_censored_nll(model(split.tabular[indices]), split.compatible[indices])
        else:
            log_pmf = marginal_log_pmf(model, method, split, indices)
            if method in LOSSES:
                loss = numeric_cumulative_loss(log_pmf, split.targets[indices], split.known[indices], LOSSES[method])
            else:
                loss = censored_nll(log_pmf, split.compatible[indices])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"{method} seed{seed} step{step} nonfinite loss")
        loss.backward()
        if method != "TabM":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        elif any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError("nonfinite TabM gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 100 == 0:
            print(json.dumps(dict(phase="train_update", method=method, seed=seed, pid=os.getpid(), physical_GPU=os.environ["CUDA_VISIBLE_DEVICES"], step=step, loss=losses[-1])), flush=True)
        if method in HYBRID_SNAPSHOT_METHODS and step % 500 == 0:
            snapshots.append(whole_cohort_loss_snapshot(cache, model, method, step, device))
    torch.cuda.synchronize(device)
    metadata = tabm_parameter_manifest(model) if method == "TabM" else parameter_manifest(model)
    final_checkpoint = None
    if checkpoint_path is not None:
        config = dict(method=method, seed=seed, numeric_keys=list(cache.metadata["numeric_keys"]),
            hidden_dim=32, cache=cache.path, q_seed=0, optimizer="AdamW", optimizer_defaults=optimizer.defaults,
            optimizer_updates=2000, batch_size=64, objective="Hybridlambda.5", learning_rate=.003,
            weight_decay=.0001, gradient_clip_norm=1., model_dtype="float64", K=8)
        if method == "OursHybrid":
            config.update(sampling_enabled=True, prefix_features=32,
                fixed_linear_logit_prior_8I=True,
                initialization="fresh seeded HistoryTrendTrajectoryHead",
                comparison_role="prior_enabled_main_method",
                prior_scope="observed-prefix linear8I skip, crossing time and quadratic category; no future labels")
        else:
            config.update(sampling_enabled=model.sampling_enabled, observed_prefix_features=20,
                initialization="fresh seeded current Ours; copy full common modules, slice only prefix[0:18,30,31]; identical tensors for both arms",
                ablation_scope="cachedK8 numeric/rule encoding plus trainable4I sampled-rule outcome skip; prefix forecast priors absent from both arms",
                inactive_capacity=model.inactive_capacity())
        with Path(checkpoint_path).open("xb") as stream:
            torch.save(dict(state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                model_training_config=config, whole_cohort_loss_snapshots=snapshots,
                source_files=["Dataset/world_model/rule_factor_graph/communication_cv_models.py",
                              "Dataset/world_model/rule_factor_graph/communication_cv_experiment.py"]), stream)
        final_checkpoint = str(Path(checkpoint_path).relative_to(ROOT))
    return model, dict(optimizer_updates=2000, batch_size=64, learning_rate=.002 if method == "TabM" else .003,
        weight_decay=.0003 if method == "TabM" else .0001, gradient_clip_norm=None if method == "TabM" else 1.,
        loss_first=losses[0], loss_last=losses[-1], loss_minimum=min(losses),
        whole_cohort_loss_snapshots=snapshots,
        final_checkpoint=final_checkpoint,
        snapshot_definition="natural full TRAIN/VALID cohort known-cell means; fixed steps0/500/1000/1500/2000; no selection or extra optimizer steps",
        training_seconds=time.perf_counter()-started, model_metadata=metadata)


def ranking(scores, labels):
    """Tie-grouped AP and pair-concordance AUC for the supplied scores."""
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not len(scores):
        return dict(average_precision=None, AP_status="no_known_targets", AUC=None, positives=0, negatives=0)
    order = np.argsort(-scores, kind="mergesort")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(ordered)-1]
    hits = np.cumsum(labels[order])[ends]
    increments = np.diff(np.r_[0, hits])
    ap = float(np.sum(hits/(ends+1)*increments)/positives) if positives else None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts)-(counts-1)/2)[inverse]
    auc = float((ranks[labels].sum()-positives*(positives+1)/2)/(positives*negatives)) if positives and negatives else None
    return dict(average_precision=ap, AP_status="defined" if positives else "no_positives", AUC=auc,
                AUC_status="defined" if positives and negatives else "requires_both_classes", positives=positives, negatives=negatives)


def probability_metrics(cdf, log_cdf, log_survival, target, known, *, ranking_scores=None):
    """Use stable logs for BCE; rank optional exact scores or the log-CDF."""
    y = target[known].astype(bool)
    probability = cdf[known]
    chosen = np.where(y, log_cdf[known], log_survival[known])
    failures = int(np.isneginf(chosen).sum())
    return dict(count=len(y), brier=float(np.mean((probability-y)**2)) if len(y) else None,
        bce=float(-chosen.mean()) if len(y) and not failures else None,
        bce_status="infinite_from_zero_probability" if failures else "finite" if len(y) else "no_known_targets",
        zero_probability_targets=failures,
        **ranking(log_cdf[known] if ranking_scores is None else ranking_scores[known], y))


def evaluate_split(model, method, split, device):
    if model is not None and method != "CatBoost":
        model.eval()
    with torch.no_grad():
        if method == "CatBoost":
            log_pmf = model.predict_log_pmf(split.tabular_cpu, device=device)
        else:
            log_pmf = torch.cat([marginal_log_pmf(model, method, split, torch.arange(start, min(start+64, split.size), device=device)) for start in range(0, split.size, 64)])
        logs = cumulative_logs(log_pmf)
    pmf = log_pmf.exp().cpu().numpy()
    log_cdf, log_survival = logs.log_cdf.cpu().numpy(), logs.log_survival.cpu().numpy()
    cdf = np.exp(log_cdf)
    targets, known = split.targets.cpu().numpy(), split.known.cpu().numpy()
    ranking_scores = None
    if method == "RuleMC":
        # Integer rule counts preserve exact mathematical ties at multiples of1/8.
        # Stable logCDF/survival remains unchanged for BCE and every other metric.
        truth = split.predicate_trajectories.cpu().numpy()
        opposite = truth[:, :, 1:] != truth[:, :1, :1]
        cumulative = np.logical_or.accumulate(opposite, axis=2)
        ranking_scores = cumulative.sum(axis=1).transpose(0, 2, 1)/8.0
    metrics = {}
    for column, predicate in enumerate(("L", "P")):
        horizons = [dict(horizon_step=h+1, horizon_seconds=(h+1)*.5,
            **probability_metrics(cdf[:, column, h], log_cdf[:, column, h], log_survival[:, column, h], targets[:, column, h], known[:, column, h],
                ranking_scores=None if ranking_scores is None else ranking_scores[:, column, h])) for h in range(10)]
        mask = known[:, column, 9]
        labels = targets[mask, column, 9].astype(bool)
        predictions = cdf[mask, column, 9] >= .5
        tp = int((labels & predictions).sum()); fp = int((~labels & predictions).sum())
        fn = int((labels & ~predictions).sum()); tn = int((~labels & ~predictions).sum())
        all_errors, matched_errors, misses, undefined, detected = [], [], 0, 0, 0
        flips = [i for i, event in enumerate(split.events) if event[predicate].kind == "first_flip"]
        for i in flips:
            mass = pmf[i, column, :10]
            hit = cdf[i, column, 9] >= .5
            detected += int(hit); misses += int(not hit)
            if mass.sum() == 0:
                undefined += 1
                continue
            predicted = float(np.dot(mass, np.arange(1, 11)*.5)/mass.sum())
            error = abs(predicted-split.events[i][predicate].step*.5)
            all_errors.append(error)
            if hit:
                matched_errors.append(error)
        metrics[predicate] = dict(horizons=horizons,
            common_full10=probability_metrics(cdf[:, column], log_cdf[:, column], log_survival[:, column], targets[:, column], known[:, column],
                ranking_scores=None if ranking_scores is None else ranking_scores[:, column]),
            occurrence=dict(threshold=.5, count=int(mask.sum()), TP=tp, FP=fp, FN=fn, TN=tn,
                precision=tp/(tp+fp) if tp+fp else None, precision_denominator=tp+fp,
                recall=tp/(tp+fn) if tp+fn else None, recall_denominator=tp+fn,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None, f1_denominator=2*tp+fp+fn,
                undefined_precision=not bool(tp+fp), undefined_recall=not bool(tp+fn), undefined_f1=not bool(2*tp+fp+fn)),
            timing=dict(all_known_flips=len(flips), detected_known_flips=detected, missed_flips=misses,
                missed_event_rate=misses/len(flips) if flips else None, missed_event_rate_denominator=len(flips),
                all_flip_MAE_seconds=float(np.mean(all_errors)) if all_errors else None, all_flip_error_denominator=len(all_errors),
                matched_time_MAE_seconds=float(np.mean(matched_errors)) if matched_errors else None, matched_time_error_denominator=len(matched_errors),
                undefined_finite_time_count=undefined, tail_timestamp=None,
                definition="conditional mean over finite bins; all-known-flip and detected-flip errors separate; no tail timestamp"))
    metrics["aggregate_full10"] = probability_metrics(cdf, log_cdf, log_survival, targets, known,
        ranking_scores=ranking_scores)
    return metrics, pmf


def run_methods(methods, *, device, results=RESULTS):
    cache = load_numeric_cache(device=device)
    print(json.dumps(dict(phase="controller_start", pid=os.getpid(), physical_GPU=os.environ["CUDA_VISIBLE_DEVICES"], methods=methods,
                          train_windows=cache.splits["train"].size, valid_windows=cache.splits["validation"].size, cache=cache.path)), flush=True)
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    for method in methods:
        for seed in (0, 1):
            started = time.perf_counter()
            path = results / f"{method}.seed{seed}.json"
            if path.exists():
                raise FileExistsError(f"fresh rerun refuses existing result: {path}")
            try:
                checkpoint_path = (results/f"{method}.seed{seed}.final.pt"
                                   if method in HYBRID_SNAPSHOT_METHODS else None)
                model, training = train_method(cache, method, seed, device, checkpoint_path=checkpoint_path)
                metrics, predictions = {}, {}
                for split_name, split in cache.splits.items():
                    metrics[split_name], predictions[split_name] = evaluate_split(model, method, split, device)
                product = results / f"{method}.seed{seed}.pmf.npz"
                with product.open("xb") as stream:
                    np.savez_compressed(stream, train_pmf=predictions["train"], validation_pmf=predictions["validation"],
                        train_window_ids=cache.splits["train"].window_ids, validation_window_ids=cache.splits["validation"].window_ids)
                row = dict(status="completed", method=method, seed=seed, pid=os.getpid(), physical_GPU=os.environ["CUDA_VISIBLE_DEVICES"],
                    completed_at=datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(), cache=cache.path,
                    classification_threshold=.5, threshold_selection=False, TEST_accessed=False, training=training,
                    ranking_score_convention=("exact cumulative fixed-rule K8 counts / 8" if method == "RuleMC" else "stable log-CDF"),
                    metrics=metrics, perquery_pmfs=str(product.relative_to(ROOT)), seconds=time.perf_counter()-started)
                write_json(path, row)
                print(json.dumps(dict(phase="fit_complete", method=method, seed=seed, pid=os.getpid(), seconds=row["seconds"])), flush=True)
                del model
                torch.cuda.empty_cache()
            except Exception as error:
                write_json(path, dict(status="failed", method=method, seed=seed, pid=os.getpid(),
                                      error_type=type(error).__name__, error=str(error)))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each assigned process requires one visible GPU, logical CUDA0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run_methods(tuple(args.methods), device=torch.device("cuda:0"), results=args.results)


if __name__ == "__main__":
    main()
