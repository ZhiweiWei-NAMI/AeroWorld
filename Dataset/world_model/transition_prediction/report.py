"""Render the current experimental results without refitting or selecting on test."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import OUT, ROOT, write_json, GraphDataset, LEGACY_LABEL_VIEW


def table(frame, columns, digits=4):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for c in columns:
            value = row[c]
            cells.append(f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render():
    config = json.loads((OUT / "config.json").read_text())
    if "label_view" in config and config["label_view"] != LEGACY_LABEL_VIEW:
        raise ValueError("this report renderer describes only the historical complete-window-v1 pilot")
    data = json.loads((OUT / "data_summary.json").read_text())
    curves = pd.read_csv(OUT / "learning_curves.csv")
    metrics = pd.read_csv(OUT / "metrics.csv")
    task = pd.read_csv(OUT / "per_predicate_metrics.csv")
    reliability = pd.read_csv(OUT / "reliability.csv")
    deletion = pd.read_csv(OUT / "nonquery_deletion.csv")
    label_index = GraphDataset(label_view=LEGACY_LABEL_VIEW).index
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    # This renderer describes only the frozen complete-window pilot.
    tree_curves = pd.read_csv(OUT / "tree_learning_curves.csv")
    for name, frame in (("metrics", metrics), ("per-predicate metrics", task), ("predictions", predictions),
                        ("learning curves", curves), ("tree learning curves", tree_curves),
                        ("reliability", reliability), ("deletion", deletion)):
        if "label_view" in frame and not frame.label_view.eq(LEGACY_LABEL_VIEW).all():
            raise ValueError(f"historical report cannot consume {name} from a different label view")
    if "label_view" in data and data["label_view"] != LEGACY_LABEL_VIEW:
        raise ValueError("historical report cannot consume a v2 data summary")
    classification_protocol = json.loads((OUT / "classification_protocol.json").read_text())
    if "label_view" in classification_protocol and classification_protocol["label_view"] != LEGACY_LABEL_VIEW:
        raise ValueError("historical report cannot consume v2 classification diagnostics")
    selection = json.loads((OUT / "selection.json").read_text())
    lock = json.loads((OUT / "evaluation_protocol.json").read_text())
    rule_replay = json.loads((ROOT / "Dataset/world_model/runs/rule_consistency/impact.json").read_text())
    label_impact = pd.DataFrame([
        dict(split=split, existing_queries=count["existing_queries"],
             current_truth_changed=count["queries_with_changed_current_truth"],
             first_switch_label_changed=count["queries_with_changed_first_switch_label"],
             queries_with_changed_graph_history=count["queries_at_those_anchors"])
        for split, count in rule_replay["pilot_impact"]["counts"].items()
    ])
    detection = pd.read_csv(OUT / "classification_metrics.csv")
    if "label_view" in detection and not detection.label_view.eq(LEGACY_LABEL_VIEW).all():
        raise ValueError("historical report cannot consume v2 classification metrics")
    detection_names = ["Prior", "Offset_ensemble", "M2_ensemble", "T_ensemble", "G_ensemble", "GBT"]
    detection_table = detection[(detection.split == "test") & (detection.threshold == 0.5) &
        detection.model.isin(detection_names)].set_index("model").loc[detection_names].reset_index()
    detection_columns = ["model", "precision", "recall", "f1", "macro_f1", "fpr", "macro_fpr", "fdr"]
    gbt_detection = detection_table.set_index("model").loc["GBT"]
    threshold_table = detection[(detection.split == "test") & (detection.model == "GBT")]
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 170, "figure.dpi": 130})
    colors = {"train": "#0072B2", "validation": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, name in zip(axes, ["GBT", "GBT_primary"]):
        for split, rows in tree_curves[tree_curves.model == name].groupby("split"):
            ax.plot(rows.stage, rows.nll, label=split, color=colors[split])
        ax.set(title=name, xlabel="Boosting iteration", ylabel="Primary-task macro joint NLL")
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False)
    fig.suptitle("Stored boosting stages; no refitting or post-test checkpoint selection")
    fig.savefig(figures / "tree_learning_curves.png")
    fig.savefig(figures / "tree_learning_curves.svg")
    plt.close(fig)
    for suffix, filename in [("", "learning_curves_all_tasks"), ("_primary", "learning_curves_primary_tasks")]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)
        for ax, mode in zip(axes.flat, config["models"]):
            name = mode+suffix
            lr = selection[name]["learning_rate"]
            current = curves[(curves.model == name) & (curves.lr == lr)]
            for split in ("train", "validation"):
                for seed, rows in current.groupby("seed"):
                    ax.plot(rows.epoch, rows[split+"_nll"], color=colors[split], alpha=0.75,
                            lw=1.3, label=split if seed == 0 else None)
            ax.set(title=name, xlabel="Epoch", ylabel="Primary-task macro joint NLL")
            ax.grid(alpha=0.18)
        axes.flat[0].legend(frameon=False)
        fig.suptitle("Selected learning rate; three optimization seeds; validation selects checkpoints")
        fig.savefig(figures / (filename+".png"))
        fig.savefig(figures / (filename+".svg"))
        plt.close(fig)
    primary = metrics[metrics.scope == "primary"]
    wanted = ["Prior", "Offset_ensemble", "M0_ensemble", "M1_ensemble", "M2_ensemble", "T_ensemble", "G_ensemble",
              "G_no_history_ensemble", "M0_primary_ensemble", "M1_primary_ensemble", "M2_primary_ensemble",
              "T_primary_ensemble", "G_primary_ensemble", "G_no_history_primary_ensemble", "GBT", "GBT_primary"]
    summary = primary[(primary.split == "test") & primary.model.isin(wanted)].copy()
    summary = summary.set_index("model").loc[wanted].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.8), constrained_layout=True)
    for ax, metric in zip(axes, ("nll", "brier", "ece")):
        ax.barh(np.arange(len(summary)), summary[metric], color="#0072B2")
        ax.set_yticks(np.arange(len(summary)), summary.model.str.replace("_ensemble", "", regex=False))
        ax.invert_yaxis()
        ax.set_xlabel(metric+" (lower is better)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Held-out family L1-4; primary-task macro scores; ensembles are optimization repeats")
    fig.savefig(figures / "model_comparison.png")
    fig.savefig(figures / "model_comparison.svg")
    plt.close(fig)
    selected_models = ["Prior", "GBT", "G_ensemble", "G_primary_ensemble", "M2_primary_ensemble", "T_primary_ensemble"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax, name in zip(axes.flat, selected_models):
        r = reliability[(reliability.model == name) & (reliability.split == "test") & (reliability.quantity == "any_switch")]
        ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", lw=1)
        ax.scatter(r.mean_probability, r.frequency, s=8+np.sqrt(r.n), c=r.bucket, cmap="viridis", alpha=0.7)
        ax.set(title=name.replace("_ensemble", ""), xlabel="Mean predicted switch probability",
               ylabel="Observed switch fraction", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.grid(alpha=0.18)
    fig.suptitle("Each point is a predicate × probability bucket; no independent-family uncertainty claim")
    fig.savefig(figures / "reliability.png")
    fig.savefig(figures / "reliability.svg")
    plt.close(fig)
    # Time-bin diagnostics: average calibration-in-the-large and actual event support.
    calibration_summary = []
    for (model, split, predicate, quantity), r in reliability.groupby(["model", "split", "predicate", "quantity"]):
        calibration_summary.append(dict(model=model, split=split, predicate=predicate, quantity=quantity,
            n=int(r.n.sum()), positive=int(r.positives.sum()),
            probability=float(np.average(r.mean_probability, weights=r.n)),
            rate=float(r.positives.sum()/r.n.sum()),
            ece=float(np.average(abs(r.mean_probability-r.frequency), weights=r.n))))
    calibration_summary = pd.DataFrame(calibration_summary)
    calibration_summary.to_csv(OUT / "calibration_summary.csv", index=False)
    validation_support = calibration_summary[(calibration_summary.model == "Prior") &
        (calibration_summary.split == "validation") & (calibration_summary.quantity != "any_switch")]
    task_ece = task[(task.scope == "primary") & (task.split == "test") & task.model.isin(selected_models)]
    failures = task_ece.assign(fail=task_ece.ece > config["reliability_tolerance"]).groupby("model").fail.agg(["sum", "count"])
    failures.reset_index().to_csv(OUT / "calibration_tolerance.csv", index=False)
    # Confidence/accuracy relation from a single probability vector, without treating seeds as calibrated uncertainty.
    selective = []
    for name in selected_models:
        f = predictions[(predictions.model == name) & (predictions.split == "test") & predictions.valid &
                        predictions.predicate.isin(config["primary_predicates"])].copy()
        p = f[[f"p{k}" for k in range(11)]].to_numpy().astype(float)
        p /= p.sum(-1, keepdims=True)
        confidence = p.max(-1)
        correct = p.argmax(-1) == f.label.to_numpy()
        for threshold in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
            keep = confidence >= threshold
            if keep.any():
                selective.append(dict(model=name, threshold=threshold, queries=int(keep.sum()), coverage=float(keep.mean()),
                    mean_confidence=float(confidence[keep].mean()), observed_accuracy=float(correct[keep].mean()),
                    true_switches=int((f.label.to_numpy()[keep] != 10).sum())))
    pd.DataFrame(selective).to_csv(OUT / "confidence_accuracy.csv", index=False)
    timing = []
    for name in ["Prior", "GBT", "GBT_primary", "G_ensemble", "G_primary_ensemble", "T_primary_ensemble"]:
        f = predictions[(predictions.model == name) & (predictions.split == "test") & predictions.valid &
                        predictions.predicate.isin(config["primary_predicates"])]
        for pred, g in f.groupby("predicate"):
            positive = g.label.to_numpy() != 10
            if not positive.any():
                continue
            p = g[[f"p{k}" for k in range(10)]].to_numpy()[positive].astype(float)
            p /= p.sum(-1, keepdims=True)
            y = g.label.to_numpy()[positive]
            timing.append(dict(model=name, predicate=pred, positive=len(y),
                conditional_time_nll=float(-np.log(p[np.arange(len(y)), y].clip(np.finfo(float).tiny)).mean()),
                uniform_time_nll=float(np.log(10)),
                conditional_time_mae=float(abs(p @ (np.arange(1, 11)*5) - (y+1)*5).mean()),
                uniform_time_mae=float(abs(27.5-(y+1)*5).mean())))
    pd.DataFrame(timing).to_csv(OUT / "conditional_timing.csv", index=False)
    # Paired task differences are descriptive; no permutation p-value from one family.
    tasks = task[(task.scope == "primary") & (task.split == "test")]
    pivot = tasks.groupby(["predicate", "model"])[["nll", "brier"]].mean().unstack("model")
    paired = []
    for g in ("G_ensemble", "G_primary_ensemble"):
        for other in ("M1_ensemble", "M2_ensemble", "T_ensemble", "GBT", "GBT_primary", "Offset_ensemble",
                      "M1_primary_ensemble", "M2_primary_ensemble", "T_primary_ensemble", "G_no_history_primary_ensemble"):
            both = pivot["nll"][[g, other]].dropna()
            for pred, row in both.iterrows():
                paired.append(dict(graph=g, comparator=other, predicate=pred, nll_difference=float(row[g]-row[other])))
    pd.DataFrame(paired).to_csv(OUT / "paired_task_differences.csv", index=False)
    deletion["difference"] = deletion.deleted_nll-deletion.full_nll
    deletion_summary = deletion.groupby("model").agg(n=("row_id", "size"), full_nll=("full_nll", "mean"),
        deleted_nll=("deleted_nll", "mean"), delta=("difference", "mean"))
    deletion_summary.reset_index().to_csv(OUT / "deletion_summary.csv", index=False)
    # Data rows and duplicated labels describe dependence, not independent sample sizes.
    sample = label_index
    first = sample[sample.predicate == "ground.vehicle_moving_in_lane"]
    second = sample[sample.predicate == "ground.vehicle_stopped_in_lane"]
    pairs = first.merge(second, on=["profile", "anchor", "bindings_json"], suffixes=("_moving", "_stopped"))
    both = pairs.valid_moving & pairs.valid_stopped
    dependence = dict(moving_stopped_matched_valid_windows=int(both.sum()),
        identical_first_switch_labels=int((pairs.loc[both, "label_moving"] == pairs.loc[both, "label_stopped"]).sum()),
        note="Correlated semantic tasks remain in the frozen macro estimand; they are not independent experimental units.")
    write_json(OUT / "dependence.json", dependence)
    stats = dict(validation_time_bins=len(validation_support), validation_bins_with_fewer_than_5_switches=int((validation_support.positive < 5).sum()),
        tolerance_failures=failures.reset_index().to_dict("records"), deletion=deletion_summary.reset_index().to_dict("records"),
        dependence=dependence, deployment_candidate_locked_before_test=lock["candidate"],
        verdict="This historical pilot uses pre-repair source labels and cannot establish calibrated graph-propagation superiority or per-instance probability accuracy.")
    by_model = summary.set_index("model")
    graph_decisions = []
    for graph, peers in [("G_ensemble", ["M1_ensemble", "M2_ensemble", "T_ensemble", "GBT"]),
                         ("G_primary_ensemble", ["M1_primary_ensemble", "M2_primary_ensemble", "T_primary_ensemble", "GBT", "GBT_primary"])]:
        for peer in peers:
            relative = (by_model.loc[peer, "nll"]-by_model.loc[graph, "nll"])/by_model.loc[peer, "nll"]
            graph_decisions.append(dict(graph=graph, comparator=peer, relative_nll_gain=float(relative),
                brier_change=float(by_model.loc[graph, "brier"]-by_model.loc[peer, "brier"]),
                passes_declared_numeric_rule=bool(relative >= config["minimum_relative_nll_gain"] and by_model.loc[graph, "brier"] <= by_model.loc[peer, "brier"])))
    stats["graph_comparisons"] = graph_decisions
    stats["graph_numeric_rule_by_model"] = {g: all(r["passes_declared_numeric_rule"] for r in graph_decisions if r["graph"] == g)
        for g in ("G_ensemble", "G_primary_ensemble")}
    stats["graph_numeric_rule_passed"] = any(stats["graph_numeric_rule_by_model"].values())
    stats["test_primary_predicates_present"] = int(summary.predicates.min())
    stats["test_primary_valid_queries"] = int(summary.n.min())
    stats["test_primary_positive_queries"] = int(summary.positive.min())
    stats["validation_candidate_test_ece"] = float(by_model.loc[lock["candidate"], "ece"])
    candidate_time = pd.DataFrame(timing).groupby("model").mean(numeric_only=True).loc[lock["candidate"]]
    stats["candidate_conditional_time_nll"] = float(candidate_time.conditional_time_nll)
    stats["uniform_conditional_time_nll"] = float(candidate_time.uniform_time_nll)
    stats["candidate_vs_prior_nll_reduction"] = float(1-by_model.loc[lock["candidate"], "nll"]/by_model.loc["Prior", "nll"])
    stats["binary_share_of_nll_reduction"] = float((by_model.loc["Prior", "binary_nll"]-by_model.loc[lock["candidate"], "binary_nll"])/(by_model.loc["Prior", "nll"]-by_model.loc[lock["candidate"], "nll"]))
    stats["gbt_detection_at_fixed_threshold_0_5"] = {k: float(gbt_detection[k]) for k in ["precision", "recall", "f1", "macro_f1", "fpr", "macro_fpr", "fdr"]}
    stats["label_integrity"] = dict(
        status="historical_pre_repair_source_labels",
        report="Dataset/world_model/runs/rule_consistency/README.md",
        published_assertion_changes=rule_replay["changed_assertions"],
        published_partitions_replayed=rule_replay["episodes"],
        frozen_pilot_impact=rule_replay["pilot_impact"],
        corrected_model_evaluation_performed=False,
    )
    write_json(OUT / "result_summary.json", stats)
    displayed = summary[["model", "nll", "binary_nll", "brier", "ece", "time_mae"]].copy()
    text = f'''# Predicate-switch pilot: historical results before rule repair

## Source-label integrity

Eight upstream semantic rule errors have been repaired in code. Replaying {rule_replay['episodes']} published formal graph partitions found {rule_replay['changed_assertions']} changed sampled assertions. The source graphs, cached pilot data, checkpoints and predictions used by this report still contain the pre-repair labels. No corrected-data training or model evaluation has been performed. See the [rule replay report](../rule_consistency/README.md) for scope, per-rule effects and source references.

Applying those truth differences to the frozen pilot sample index gives:

{table(label_impact, list(label_impact.columns))}

These are overlapping query windows. The count does not enumerate newly eligible queries or regenerate any model input. The scores below describe the historical target; reassess architecture rankings and calibration after rebuilding the affected label/graph lineage and training in a new experiment directory. Source correction and verification precede rule-factor compilation or targeted expansion.

## Scope and decision

The experiment predicts the first binary predicate switch from an ideal semantic graph prefix. It uses L1-3 v1/v2 seed00 for chronological development and the selected L1-4 v1/v2 seed00 episodes for held-out evaluation; every selected L1-4 source stays out of development. It does not evaluate observation-to-graph estimation or support a claim about arbitrary unseen families.

The validation-selected candidate locked before held-out evaluation was **{lock['candidate']}**. The graph models did not pass the declared numeric superiority rule against the full comparator set on the original labels. GBT remains the historical comparison reference; the corrected target requires a new comparison. This graph head has no demonstrated superiority or guaranteed probability accuracy. GBT's original held-out macro ECE is {stats['validation_candidate_test_ece']:.4f}, against the descriptive tolerance of 0.05.

The table below reports every predeclared model, including the original all-task objective and the primary-task refinement. The suffix `_primary` means the loss optimizes the frozen primary task set; `_ensemble` is the arithmetic mean of three optimization-seed distributions. It provides no independent-family confidence interval.

{table(displayed, list(displayed.columns))}

The held-out set contains {stats['test_primary_predicates_present']} of the 15 declared primary predicates, with {stats['test_primary_valid_queries']} valid primary queries and {stats['test_primary_positive_queries']} switches. All scores are macro averages over these supported predicates. `time_mae` is measured in ticks, conditional on observed switches. Lower is better for the displayed scores. Compare per-predicate results before attributing aggregate improvements to an architecture. CPA is task-specific and appears in `per_predicate_metrics.csv`, not the full-task table. Phase is a diagnostic; its validation phases have no training support.

GBT reduces joint NLL relative to the fixed prior by {stats['candidate_vs_prior_nll_reduction']:.1%}; {stats['binary_share_of_nll_reduction']:.1%} of that reduction comes from the binary switch/no-switch term. Its conditional time-bin NLL is {stats['candidate_conditional_time_nll']:.4f}, compared with {stats['uniform_conditional_time_nll']:.4f} for uniform timing. The evidence for whether a switch occurs is stronger than the evidence for its precise time.

The decision threshold was a 2% relative joint-NLL gain for G over M1, M2, T and GBT, without binary-Brier regression. Scientific confirmation additionally needs independent-family evidence; this pilot has only one held-out family. Accurate joint probabilities across predicates and per-instance calibration remain unestablished.

## F1 and false alarms

The following diagnostics were added to the frozen predictions after the original held-out evaluation. No model was refitted, no calibration was changed and no alarm threshold was selected on the test labels. A positive means at least one predicate switch in the next 50 ticks; its direction can be false→true or true→false. It does not mean that the predicate itself is true.

At the fixed reporting threshold `P(any switch) >= 0.5`:

{table(detection_table, detection_columns)}

`f1` uses pooled query counts: `2 TP / (2 TP + FP + FN)`. `macro_f1` gives each present predicate equal weight, with a declared zero when neither actual nor predicted positives exist. Undefined precision and recall remain explicit in the per-predicate CSV. `fpr = FP/(FP+TN)` measures negative queries incorrectly flagged; `fdr = FP/(TP+FP)` measures false alarms among all alarms. The positive fraction is only {gbt_detection['prevalence']:.2%}, so a small pooled FPR can coexist with poor alarm precision.

GBT produces {int(gbt_detection['tp'])} true alarms and {int(gbt_detection['fp'])} false alarms, misses {int(gbt_detection['fn'])} positive windows and correctly rejects {int(gbt_detection['tn'])} negatives. Its pooled F1 is {gbt_detection['f1']:.4f}, while equal-predicate macro F1 is {gbt_detection['macro_f1']:.4f}. These are overlapping predicate-instance windows, not deduplicated event alarms or a false-alarm rate per hour. Complete-window exclusion also limits the population to which these rates apply.

Fixed-threshold tradeoffs for GBT, without choosing a new operating point:

{table(threshold_table, ['threshold', 'precision', 'recall', 'f1', 'fpr', 'fdr'])}

Per-predicate failures matter. At threshold 0.5, GBT flags every retransmission query, so that task's FPR is 100% despite its high positive base rate. For altitude-band excursion, 24 of 25 alarms are false. The corresponding counts, denominators and all other task results are in `classification_per_predicate.csv`.

The existing joint NLL penalizes a false alarm through the no-switch term `-log(p_none)`: assigning large switch mass to a true no-switch window increases loss. Task weighting, smoothing and early stopping do not impose an explicit false-positive budget. There is currently no task-specific alarm threshold, asymmetric false-alarm cost, fitted probability calibration or false-positive-rate constraint in the deployment interface. Probability normalization itself does not control false alarms. F1 also ignores true negatives and does not test probability calibration.

A future alarm policy should retain a proper probability objective and specify the acceptable false-alarm/recall tradeoff per predicate. Thresholds or constrained decision rules must be fitted on development/calibration groups and assessed on fresh independent groups; this test set cannot be reused to certify a new policy. Upweighting negatives or replacing NLL with a classification-focused loss changes the probability estimand unless corrected, so it is not an automatic calibration-preserving remedy.

## Data and coverage

| Split | Queries | Complete valid windows | First-switch positives |
|---|---:|---:|---:|
| Train | {data['split_counts']['train']} | {data['valid_split_counts']['train']} | {data['positive_split_counts']['train']} |
| Validation | {data['split_counts']['validation']} | {data['valid_split_counts']['validation']} | {data['positive_split_counts']['validation']} |
| Test | {data['split_counts']['test']} | {data['valid_split_counts']['test']} | {data['positive_split_counts']['test']} |

There are {data['total_train_predicates']} training-supported predicate types and {len(config['primary_predicates'])} primary types, selected from training support alone. Across the three splits, {data['multiple_switch_windows']} valid windows contain more than one sampled switch; {data['same_endpoint_positive_windows']} positive windows end at their initial truth. These are first-switch positives, not negatives. The table counts overlapping windows, not independent trials.

Matched moving/stopped vehicle predicates have {dependence['moving_stopped_matched_valid_windows']} valid paired windows; {dependence['identical_first_switch_labels']} have identical first-switch labels. These semantic tasks remain in the fixed macro estimand and are not independent repetitions.

## Learning curves and loss refinement

![Boosted baseline learning curves](figures/tree_learning_curves.png)

![Original objective learning curves](figures/learning_curves_all_tasks.png)

![Primary objective learning curves](figures/learning_curves_primary_tasks.png)

The curves plot primary-task NLL for training and validation. The original running training objective averages 60 task types and therefore differs from the plotted 15-task aggregate. Its primary task weight totals 25%. The second iteration changes the task population of the loss for every architecture, without changing data, labels or inputs. Optimizer gradients are checked for finite values, checkpoints are selected on validation NLL, and early stopping uses the same policy across neural models. A falling training curve alone is not evidence of generalization.

The original curves showed nearly identical G and M2 validation performance and little primary-task fitting. This motivated the ten-parameter Offset baseline and the primary-loss refinement before held-out predictions were opened. `selection.json` and `learning_curves.csv` retain all learning-rate trials and optimization seeds. The metric implementation was corrected to sum small event masses in float64; this is a numerical correction, not a calibration fit.

The focused objective improves training fit and validation selection for several architectures but worsens held-out scores against the original labels. This shows limited transfer and overfitting relative to that target; the source-label errors are an additional confounder. Correcting the labels does not authorize retuning on the held-out family. The small Offset model is an important control: graph features must improve on learned prior corrections as well as on the fixed prior.

## Probability calibration and accuracy

![Probability reliability](figures/reliability.png)

The head guarantees total probability one, nonnegative masses, a monotone cumulative first-switch probability and agreement between the binary marginal and time-bin masses. NLL is a proper distribution score; Brier/log loss assess its binary marginal. These properties align the optimization target with the reported probabilities, but they do not guarantee a correct learned distribution.

Of {stats['validation_time_bins']} primary-task time-bin cells in validation, {stats['validation_bins_with_fewer_than_5_switches']} contain fewer than five observed first switches. Fine timing calibration is therefore weakly supported even when a pooled calibration error looks small. `calibration_summary.csv` and `reliability.csv` report counts and discrepancies for every task/bin; `confidence_accuracy.csv` compares predicted categorical confidence with observed accuracy at explicit thresholds. High no-switch accuracy can conceal poor switch detection, so that file also reports actual switch counts.

The 0.05 reliability tolerance is descriptive. No confidence guarantee is attached to ECE, optimization-seed standard deviation or a high maximum predicted probability. The experiment fits no temperature or post-hoc calibration on the test family. Independent query outputs do not enforce logical constraints across a future graph.

## Ablations and missingness

Non-query deletion uses fixed-seed sampling of at most eight valid held-out queries per primary task, without selecting on outcomes. It deletes non-query entities and predicates requiring them while preserving the query's actual remaining features.

{table(deletion_summary.reset_index(), list(deletion_summary.reset_index().columns))}

These are descriptive paired losses for this small query subset. `G_no_history` is a retrained temporal ablation. `paired_task_differences.csv` retains per-task G differences against the baselines. No family permutation test is valid with one held-out family.

Incomplete future windows are excluded from the joint likelihood. `validity_by_evidence.csv` reports validity conditional on observed opposite truth; this is not validity conditional on the unobserved true outcome. `missing_label_bounds.csv` scores uncertain binary outcomes both ways under a hypothetical binary completion, fixing positive outcomes only where a consecutive pair of binary samples actually switches. An out-of-scope predicate need not have been defined, so the completion assumption is substantive. The bounds expose sensitivity to missingness; complete-window scores do not certify performance on the excluded population.

## Reproduce and inspect

See [implementation commands](../../transition_prediction/README.md) and [model/input/output design](../../../../design/belief/predicate_transition_model.md). `config.json` fixes the protocol; `evaluation_protocol.json` records selection before test; `resource_budget.json` records algebraic resource assumptions. `models/` contains learned checkpoints and baselines. `episodes.pt` is a compact projection of source graphs; `samples.parquet` retains label and source references outside the model tensors. Predictions and metrics regenerate from those objects without modifying the frozen source graphs.

![Held-out model comparison](figures/model_comparison.png)
'''
    (OUT / "README.md").write_text(text)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    render()
