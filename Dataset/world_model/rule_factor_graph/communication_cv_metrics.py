"""Streaming, censor-aware metrics for the fixed communication component CV."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import torch

EVENTS = ("L", "P")
HORIZON = 10
OCCURRENCE_TARGETS = ("L", "P", "L_AND_P")
OCCURRENCE_METRIC_SCHEMA = "communication_first_flip_occurrence_v1"
CLASSIFICATION_THRESHOLD = 0.5
DESCRIPTIVE_THRESHOLD_GRID = tuple(index / 10 for index in range(11))
RELIABILITY_BIN_COUNT = 10


def occurrence_metric_contract() -> dict:
    """Fixed definitions to record before opening held-out model scores."""
    return {
        "schema": OCCURRENCE_METRIC_SCHEMA,
        "horizon_seconds": 5.0,
        "targets": {
            "L": "First opposite latency-predicate truth occurs within the next ten half-second bins.",
            "P": "First opposite packet-loss-predicate truth occurs within the next ten half-second bins.",
            "L_AND_P": "Both first opposite truths occur within five seconds; their first-flip bins may differ.",
        },
        "input_distribution": "The final postprojection 121-cell joint PMF, after all q branches and model ensemble members have been mixed.",
        "score_definitions": {"L": "Sum of the first ten L-event bins, over all P outcomes.",
                              "P": "Sum of the first ten P-event bins, over all L outcomes.",
                              "L_AND_P": "Sum of joint cells whose two event outcomes are both first-flip bins."},
        "binary_nll": "Use logsumexp of positive or negative joint-cell sets for identified labels; do not compute a complement by 1-p.",
        "marginal_eligibility": "first_flip is positive; none is negative; right_censored is unknown and excluded.",
        "joint_AND_eligibility": "Both first_flip is positive; either known none is negative even if the other event is censored; all remaining cases are unknown and excluded. This is separate from both-events-identified eligibility for 121-way Brier.",
        "complete_future_sensitivity": "Require all ten future L/P observation bins known, then apply the same target definitions.",
        "anchor_truth_strata": "L/P tables separately report anchor truth 0 (first flip 0->1) and 1 (first flip 1->0); both directions remain occurrence targets.",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "threshold_rule": "Probability >= threshold predicts positive, including equality.",
        "descriptive_threshold_grid": list(DESCRIPTIVE_THRESHOLD_GRID),
        "threshold_selection": "Fixed before the run; no threshold is selected from held-out scores.",
        "rates": {"precision": "TP/(TP+FP)", "recall": "TP/(TP+FN)",
                  "false_negative_rate": "FN/(TP+FN)", "false_positive_rate": "FP/(FP+TN)",
                  "f1": "2TP/(2TP+FP+FN)", "specificity": "TN/(TN+FP)"},
        "undefined_rates": "Null when the stated denominator is zero; denominator and support are reported.",
        "average_precision": "Rank monotonic log occurrence probabilities to preserve tiny-score ordering; group equal scores at one threshold; null with no positives.",
        "roc_auc": "Exact positive/negative pair concordance with tied scores receiving half credit; null unless both classes occur.",
        "ranking_score": "Both AP and ROC AUC rank log-positive probabilities to preserve probability order through exponentiation underflow; fixed thresholds and reliability bins use probabilities.",
        "crossfold_aggregation": "Micro confusion counts are summed over held-out folds before computing rates, separately per seed. Macro AP gives equal weight to positive-supported folds and reports their count out of six; ROC AUC macro analogously requires both classes. Scores are never pooled across folds for AP/AUC.",
        "reliability_bins": {"count": RELIABILITY_BIN_COUNT,
                             "edges": [index / RELIABILITY_BIN_COUNT for index in range(RELIABILITY_BIN_COUNT + 1)],
                             "rule": "Left-closed, right-open equal-width bins; probability 1 belongs to the final bin.",
                             "sufficient_statistics": ["count", "sum_probability", "sum_label"]},
        "interpretation": "Rates describe overlapping prediction windows. They are not physical event-level false-alarm or missed-alarm rates; no alarm matching or event deduplication is performed.",
        "persistence": "Only final scalar aggregates, fixed-grid confusion counts and reliability-bin sufficient statistics. Scores and labels exist in fit-local RAM only; no per-window predictions or PMFs are written.",
    }


def occurrence_targets(observations: Sequence[dict]) -> np.ndarray:
    """Return [L, P, AND] labels, with explicit -1 for unidentifiable targets."""
    result = np.full((len(observations), len(OCCURRENCE_TARGETS)), -1, dtype=np.int8)
    for index, row in enumerate(observations):
        for column, name in enumerate(EVENTS):
            kind = row[name].kind
            if kind == "first_flip":
                result[index, column] = 1
            elif kind == "none":
                result[index, column] = 0
            elif kind != "right_censored":
                raise ValueError(f"Unknown event observation kind {kind}")
        if (result[index, :2] == 1).all():
            result[index, 2] = 1
        elif (result[index, :2] == 0).any():
            result[index, 2] = 0
    return result


def _ratio(numerator: int | float, denominator: int) -> dict:
    return {"value": float(numerator / denominator) if denominator else None,
            "numerator": numerator, "denominator": denominator}


def _confusion(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    positive = labels == 1
    return {"TP": int((predicted & positive).sum()), "TN": int((~predicted & ~positive).sum()),
            "FP": int((predicted & ~positive).sum()), "FN": int((~predicted & positive).sum())}


def _ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    if not positives:
        return {"average_precision": _ratio(0, 0),
                "roc_auc": _ratio(0, positives * negatives)}
    order = np.argsort(scores, kind="stable")[::-1]
    ordered_scores, ordered_labels = scores[order], labels[order]
    # One endpoint per distinct score makes both rankings invariant to tie order.
    endpoints = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), len(scores) - 1]
    tp = np.cumsum(ordered_labels, dtype=np.int64)[endpoints]
    fp = endpoints + 1 - tp
    group_positive = np.diff(np.r_[0, tp])
    group_negative = np.diff(np.r_[0, fp])
    ap_numerator = float((group_positive * tp / (tp + fp)).sum())
    concordant_pairs = float((group_positive * (negatives - fp + 0.5 * group_negative)).sum())
    return {"average_precision": _ratio(ap_numerator, positives),
            "roc_auc": _ratio(concordant_pairs, positives * negatives)}


@dataclass
class BinaryOccurrenceMetrics:
    """Keep only eligible binary scores and labels in fit-local memory."""
    total_windows: int = 0
    _scores: list = field(default_factory=list, repr=False)
    _ranking_scores: list = field(default_factory=list, repr=False)
    _labels: list = field(default_factory=list, repr=False)
    binary_nll_sum: float = 0.0
    binary_nll_count: int = 0

    def update(self, probabilities: np.ndarray, labels: np.ndarray,
               *, log_positive: np.ndarray, log_negative: np.ndarray) -> None:
        scores, targets = np.asarray(probabilities, dtype=np.float64), np.asarray(labels)
        if scores.ndim != 1 or targets.shape != scores.shape or not np.isin(targets, (-1, 0, 1)).all():
            raise ValueError("Occurrence scores/labels must be aligned vectors with labels in -1,0,1")
        if not np.isfinite(scores).all() or (scores < -1e-12).any() or (scores > 1 + 1e-12).any():
            raise ValueError("Occurrence scores must be finite probabilities")
        self.total_windows += len(targets)
        eligible = targets >= 0
        positive_logs, negative_logs = np.asarray(log_positive), np.asarray(log_negative)
        if positive_logs.shape != scores.shape or negative_logs.shape != scores.shape:
            raise ValueError("Binary occurrence log probabilities must align with scores")
        if (np.isnan(positive_logs).any() or np.isposinf(positive_logs).any()
                or np.isnan(negative_logs).any() or np.isposinf(negative_logs).any()):
            raise ValueError("Binary occurrence logs must be finite or explicit negative infinity")
        nll = -np.where(targets[eligible] == 1, positive_logs[eligible], negative_logs[eligible])
        if not np.isfinite(nll).all():
            raise FloatingPointError("Identified binary occurrence target has zero predicted support")
        self.binary_nll_sum += float(nll.sum())
        self.binary_nll_count += int(eligible.sum())
        if eligible.any():
            # Clip only the admitted floating-point endpoint roundoff.
            self._scores.append(np.clip(scores[eligible], 0, 1))
            self._ranking_scores.append(np.minimum(positive_logs[eligible], 0).astype(np.float64, copy=True))
            self._labels.append(targets[eligible].astype(np.int8, copy=True))

    def result(self) -> dict:
        scores = np.concatenate(self._scores) if self._scores else np.empty(0, dtype=np.float64)
        labels = np.concatenate(self._labels) if self._labels else np.empty(0, dtype=np.int8)
        ranking_scores = (np.concatenate(self._ranking_scores) if self._ranking_scores
                          else np.empty(0, dtype=np.float64))
        confusion = _confusion(scores, labels, CLASSIFICATION_THRESHOLD)
        tp, tn, fp, fn = (confusion[key] for key in ("TP", "TN", "FP", "FN"))
        bins = np.minimum((scores * RELIABILITY_BIN_COUNT).astype(np.int64), RELIABILITY_BIN_COUNT - 1)
        counts = np.bincount(bins, minlength=RELIABILITY_BIN_COUNT)
        probability_sums = np.bincount(bins, weights=scores, minlength=RELIABILITY_BIN_COUNT)
        label_sums = np.bincount(bins, weights=labels, minlength=RELIABILITY_BIN_COUNT)
        return {
            "total_windows": self.total_windows, "eligible_windows": len(labels),
            "excluded_unknown_windows": self.total_windows - len(labels),
            "positive_support": tp + fn, "negative_support": tn + fp,
            "threshold": CLASSIFICATION_THRESHOLD, "confusion": confusion,
            "binary_nll": {"sum": self.binary_nll_sum, "count": self.binary_nll_count,
                           "mean": self.binary_nll_sum / self.binary_nll_count if self.binary_nll_count else None},
            "rates": {"precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                      "false_negative_rate": _ratio(fn, tp + fn), "false_positive_rate": _ratio(fp, fp + tn),
                      "f1": _ratio(2 * tp, 2 * tp + fp + fn), "specificity": _ratio(tn, tn + fp)},
            **_ranking_metrics(ranking_scores, labels),
            "fixed_threshold_grid": [{"threshold": threshold, **_confusion(scores, labels, threshold)}
                                     for threshold in DESCRIPTIVE_THRESHOLD_GRID],
            "reliability_bins": [{"bin": index, "lower": index / RELIABILITY_BIN_COUNT,
                                  "upper": (index + 1) / RELIABILITY_BIN_COUNT,
                                  "upper_inclusive": index == RELIABILITY_BIN_COUNT - 1,
                                  "count": int(counts[index]), "sum_probability": float(probability_sums[index]),
                                  "sum_label": int(label_sums[index])}
                                 for index in range(RELIABILITY_BIN_COUNT)],
        }


def _occurrence_states() -> dict:
    return {cohort: {target: BinaryOccurrenceMetrics() for target in OCCURRENCE_TARGETS}
            for cohort in ("raw", "complete_future")}


def _anchor_occurrence_states() -> dict:
    return {cohort: {target: {str(anchor): BinaryOccurrenceMetrics() for anchor in (0, 1)}
                    for target in EVENTS} for cohort in ("raw", "complete_future")}


def event_compatible_vector(event, horizon: int = HORIZON) -> np.ndarray:
    result = np.zeros(horizon + 1, dtype=bool)
    if event.kind == "first_flip":
        if not 1 <= event.step <= horizon:
            raise ValueError("First flip lies outside fixed horizon")
        result[event.step - 1] = True
    elif event.kind == "none":
        result[-1] = True
    elif event.kind == "right_censored":
        if not 0 <= event.step < horizon:
            raise ValueError("Censor endpoint lies outside fixed horizon")
        result[event.step:] = True
    else:
        raise ValueError(f"Unknown event observation kind {event.kind}")
    return result


def compatible_event_masks(observations: Sequence[dict]) -> np.ndarray:
    return np.stack([event_compatible_vector(row["L"])[:, None] &
                     event_compatible_vector(row["P"])[None, :] for row in observations])


def informative_event(event) -> bool:
    return event.kind != "right_censored" or event.step > 0


def _add(store: dict, name: str, values) -> None:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"Nonfinite metric {name}")
    if name not in store:
        store[name] = {"sum": 0.0, "count": 0}
    store[name]["sum"] += float(array.sum())
    store[name]["count"] += int(array.size)


@dataclass
class StreamingMetrics:
    """Every reported mean carries its eligible denominator."""
    sums: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=lambda: {
        "windows": 0, "informative_windows": 0, "zero_information_windows": 0,
        "fully_observed_future_windows": 0,
        "events": {name: {"first_flip": 0, "none": 0, "right_censored": 0,
                           "zero_observed_future_bin": 0} for name in EVENTS}})
    normalization_error_max: float = 0.0
    occurrence: dict = field(default_factory=_occurrence_states, repr=False)
    anchor_occurrence: dict = field(default_factory=_anchor_occurrence_states, repr=False)

    def update(self, log_pmf: torch.Tensor, observations: Sequence[dict], *,
               complete_future: Sequence[bool], raw_illegal_mass: torch.Tensor,
               projected_illegal_mass: torch.Tensor,
               initial_predicates: Sequence[tuple[bool, bool]] | torch.Tensor) -> None:
        logs = log_pmf.detach().cpu().numpy()
        if logs.shape != (len(observations), 11, 11) or np.isnan(logs).any() or np.isposinf(logs).any():
            raise ValueError("Joint event log PMF must have valid shape [B,11,11]")
        pmf = np.exp(logs)
        norms = pmf.sum((1, 2))
        error = float(np.max(np.abs(norms - 1)))
        self.normalization_error_max = max(self.normalization_error_max, error)
        if error > 1e-8:
            raise ValueError(f"Unnormalized joint event distribution: {error}")
        masks = compatible_event_masks(observations)
        # Log-space summation retains small, nonzero compatible probabilities.
        masked_logs = log_pmf.detach().masked_fill(
            ~torch.as_tensor(masks, dtype=torch.bool, device=log_pmf.device), -torch.inf)
        nll = -torch.logsumexp(masked_logs.flatten(1), 1).cpu().numpy()
        informative = np.array([any(informative_event(row[name]) for name in EVENTS)
                                for row in observations], dtype=bool)
        complete = np.asarray(complete_future, dtype=bool)
        if complete.shape != informative.shape or (complete & ~informative).any():
            raise ValueError("Complete future-window flag violates observation contract")
        if any(complete[index] and any(row[name].kind == "right_censored" for name in EVENTS)
               for index, row in enumerate(observations)):
            raise ValueError("A completely observed future cannot contain a censored first-flip target")
        anchors = (initial_predicates.detach().cpu().numpy() if isinstance(initial_predicates, torch.Tensor)
                   else np.asarray(initial_predicates))
        if anchors.shape != (len(observations), 2) or anchors.dtype != np.dtype(bool):
            raise ValueError("Anchor predicate truths must be a known Boolean matrix [B,2]")
        _add(self.sums, "joint_censored_nll", nll[informative])
        _add(self.sums, "complete_future_joint_censored_nll", nll[complete])
        for name, value in (("raw_illegal_mass",raw_illegal_mass),("projected_illegal_mass",projected_illegal_mass)):
            values = value.detach().cpu().numpy()
            if values.shape != (len(observations),) or (values < -1e-12).any() or (values > 1+1e-12).any():
                raise ValueError("Illegal event mass must be a probability per window")
        _add(self.sums, "raw_illegal_mass", raw_illegal_mass.detach().cpu().numpy())
        _add(self.sums, "projected_illegal_mass", projected_illegal_mass.detach().cpu().numpy())
        self.coverage["windows"] += len(observations)
        self.coverage["informative_windows"] += int(informative.sum())
        self.coverage["zero_information_windows"] += int((~informative).sum())
        self.coverage["fully_observed_future_windows"] += int(complete.sum())
        exact_joint = np.ones(len(observations), dtype=bool)
        targets = np.zeros_like(pmf)
        for column, name in enumerate(EVENTS):
            marginal = pmf.sum(2 if column == 0 else 1)
            marginal_log = torch.logsumexp(log_pmf.detach(), 2 if column == 0 else 1)
            event_masks = np.stack([event_compatible_vector(row[name]) for row in observations])
            event_nll = -torch.logsumexp(marginal_log.masked_fill(
                ~torch.as_tensor(event_masks, device=log_pmf.device), -torch.inf), 1).cpu().numpy()
            eligible = np.array([informative_event(row[name]) for row in observations], dtype=bool)
            _add(self.sums, name + "_marginal_event_nll", event_nll[eligible])
            _add(self.sums, "complete_future_" + name + "_marginal_event_nll", event_nll[complete])
            identified = np.array([row[name].kind != "right_censored" for row in observations], dtype=bool)
            exact_joint &= identified
            onehot = np.zeros_like(marginal)
            occurrence = []
            for i, row in enumerate(observations):
                event = row[name]
                self.coverage["events"][name][event.kind] += 1
                self.coverage["events"][name]["zero_observed_future_bin"] += int(
                    event.kind == "right_censored" and event.step == 0)
                if event.kind == "first_flip":
                    onehot[i, event.step - 1] = 1
                    occurrence.append(1)
                elif event.kind == "none":
                    onehot[i, -1] = 1
                    occurrence.append(0)
            _add(self.sums, name + "_marginal_event_brier", ((marginal-onehot)**2).sum(1)[identified])
            occurrence_error = ((1-marginal[identified, -1])-np.asarray(occurrence))**2
            _add(self.sums, name + "_event_within_horizon_brier", occurrence_error)
            full_occurrence_error = ((1-marginal[:, -1])-onehot[:, :-1].sum(1))**2
            _add(self.sums, "complete_future_" + name + "_event_within_horizon_brier",full_occurrence_error[complete])
            _add(self.sums, "complete_future_" + name + "_marginal_event_brier",
                 ((marginal-onehot)**2).sum(1)[complete])
        for i, row in enumerate(observations):
            if exact_joint[i]:
                indices = [row[name].step-1 if row[name].kind == "first_flip" else HORIZON
                           for name in EVENTS]
                targets[i, indices[0], indices[1]] = 1
        _add(self.sums, "joint_event_brier", ((pmf-targets)**2).sum((1, 2))[exact_joint])
        _add(self.sums, "complete_future_joint_event_brier", ((pmf-targets)**2).sum((1, 2))[complete])
        binary_targets = occurrence_targets(observations)
        joint_logs = log_pmf.detach()
        log_positive = torch.stack((torch.logsumexp(joint_logs[:, :HORIZON, :].flatten(1), 1),
                                    torch.logsumexp(joint_logs[:, :, :HORIZON].flatten(1), 1),
                                    torch.logsumexp(joint_logs[:, :HORIZON, :HORIZON].flatten(1), 1)), 1)
        joint_negative = torch.ones((11, 11), dtype=torch.bool, device=joint_logs.device)
        joint_negative[:HORIZON, :HORIZON] = False
        log_negative = torch.stack((torch.logsumexp(joint_logs[:, HORIZON, :], 1),
                                    torch.logsumexp(joint_logs[:, :, HORIZON], 1),
                                    torch.logsumexp(joint_logs[:, joint_negative], 1)), 1)
        occurrence_probabilities = torch.exp(log_positive).cpu().numpy()
        positive_logs, negative_logs = log_positive.cpu().numpy(), log_negative.cpu().numpy()
        for column, target in enumerate(OCCURRENCE_TARGETS):
            for cohort, eligible_cohort in (("raw", np.ones(len(complete), dtype=bool)),
                                            ("complete_future", complete)):
                self.occurrence[cohort][target].update(occurrence_probabilities[eligible_cohort, column],
                    binary_targets[eligible_cohort, column], log_positive=positive_logs[eligible_cohort, column],
                    log_negative=negative_logs[eligible_cohort, column])
                if target in EVENTS:
                    for anchor in (0, 1):
                        selected = eligible_cohort & (anchors[:, column] == bool(anchor))
                        self.anchor_occurrence[cohort][target][str(anchor)].update(
                            occurrence_probabilities[selected, column], binary_targets[selected, column],
                            log_positive=positive_logs[selected, column], log_negative=negative_logs[selected, column])

    def result(self) -> dict:
        return {"coverage": self.coverage, "metrics": {
            name: {**item, "mean": item["sum"]/item["count"] if item["count"] else None}
            for name, item in self.sums.items()},
            "joint_pmf_normalization_error_max": self.normalization_error_max,
            "brier_eligibility": "11-way and occurrence Brier use identified first-flip/none targets; censor-only targets are excluded; complete-future sensitivity requires all future L/P observations",
            "occurrence_classification": {"contract": occurrence_metric_contract(),
                **{cohort: {target: {**self.occurrence[cohort][target].result(),
                            **({"anchor_truth_strata": {anchor: state.result()
                                for anchor, state in self.anchor_occurrence[cohort][target].items()}}
                               if target in EVENTS else {})} for target in OCCURRENCE_TARGETS}
                   for cohort in ("raw", "complete_future")}}}


def _classification_table(row: dict, cohort: str, target: str, anchor: str | None = None) -> dict:
    table = row["occurrence_classification"][cohort][target]
    return table if anchor is None else table["anchor_truth_strata"][anchor]


def _classification_scalar(table: dict, name: str) -> tuple[float | None, int]:
    if name == "binary_nll":
        return table[name]["mean"], table[name]["count"]
    value = table[name] if name in ("average_precision", "roc_auc") else table["rates"][name]
    return value["value"], value["denominator"]


CLASSIFICATION_SCALARS = ("precision", "recall", "false_negative_rate", "false_positive_rate",
                          "f1", "specificity", "average_precision", "roc_auc", "binary_nll")


def _seed_scalar_summary(values: dict, denominators: dict) -> dict:
    complete = set(values) == set(range(5)) and all(value is not None for value in values.values())
    return {"seed_values": values, "denominators_by_seed": denominators,
            "mean": float(np.mean(list(values.values()))) if complete else None,
            "range": [min(values.values()), max(values.values())] if complete else None}


def _group_classification_table(selected: Sequence[dict], cohort: str,
                                target: str, anchor: str | None = None) -> dict:
    support_keys = ("total_windows", "eligible_windows", "excluded_unknown_windows", "positive_support", "negative_support")
    tables = {row["training_seed"]: _classification_table(row, cohort, target, anchor) for row in selected}
    supports = [{key: table[key] for key in support_keys} for table in tables.values()]
    if supports and any(support != supports[0] for support in supports[1:]):
        raise ValueError("Occurrence target support changed across seeds in one held-out group")
    return {"completed_seeds": sorted(tables), "all_five_seeds_completed": len(tables) == 5,
            "support": supports[0] if supports else None,
            "confusion_counts_by_seed": {seed: table["confusion"] for seed, table in tables.items()},
            "scalar_seed_summaries": {name: _seed_scalar_summary(
                {seed: _classification_scalar(table, name)[0] for seed, table in tables.items()},
                {seed: _classification_scalar(table, name)[1] for seed, table in tables.items()})
                for name in CLASSIFICATION_SCALARS}}


def _group_classification(selected: Sequence[dict]) -> dict:
    return {cohort: {target: {**_group_classification_table(selected, cohort, target),
                    **({"anchor_truth_strata": {anchor: _group_classification_table(selected, cohort, target, anchor)
                                                for anchor in ("0", "1")}}
                       if target in EVENTS else {})} for target in OCCURRENCE_TARGETS}
            for cohort in ("raw", "complete_future")}


def _rates_from_counts(confusion: dict) -> dict:
    tp, tn, fp, fn = (confusion[key] for key in ("TP", "TN", "FP", "FN"))
    return {"precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "false_negative_rate": _ratio(fn, tp + fn), "false_positive_rate": _ratio(fp, fp + tn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn), "specificity": _ratio(tn, tn + fp)}


def _crossfold_classification_table(lookup: dict, groups: Sequence[str], model: str,
                                    cohort: str, target: str, anchor: str | None = None) -> dict:
    """Sum confusion per seed; rank metrics average supported folds, never scores."""
    complete_seeds, micro = [], {}
    macro = {name: {"seed_values": {}, "eligible_fold_ids_by_seed": {}}
             for name in ("average_precision", "roc_auc")}
    for seed in range(5):
        if not all((group, seed, model) in lookup for group in groups):
            continue
        complete_seeds.append(seed)
        tables = {group: _classification_table(lookup[group, seed, model], cohort, target, anchor)
                  for group in groups}
        confusion = {key: sum(table["confusion"][key] for table in tables.values())
                     for key in ("TP", "TN", "FP", "FN")}
        denominator = sum(table["binary_nll"]["count"] for table in tables.values())
        nll_sum = sum(table["binary_nll"]["sum"] for table in tables.values())
        micro[seed] = {"confusion": confusion, "rates": _rates_from_counts(confusion),
            "total_windows": sum(table["total_windows"] for table in tables.values()),
            "eligible_windows": sum(table["eligible_windows"] for table in tables.values()),
            "excluded_unknown_windows": sum(table["excluded_unknown_windows"] for table in tables.values()),
            "positive_support": confusion["TP"] + confusion["FN"],
            "negative_support": confusion["TN"] + confusion["FP"],
            "binary_nll": {"sum": nll_sum, "count": denominator,
                           "mean": nll_sum / denominator if denominator else None},
            "fixed_threshold_grid": [{"threshold": threshold,
                **{key: sum(table["fixed_threshold_grid"][index][key] for table in tables.values())
                   for key in ("TP", "TN", "FP", "FN")}}
                for index, threshold in enumerate(DESCRIPTIVE_THRESHOLD_GRID)],
            "reliability_bins": [{"bin": index, "lower": index / RELIABILITY_BIN_COUNT,
                "upper": (index + 1) / RELIABILITY_BIN_COUNT,
                "upper_inclusive": index == RELIABILITY_BIN_COUNT - 1,
                **{key: sum(table["reliability_bins"][index][key] for table in tables.values())
                   for key in ("count", "sum_probability", "sum_label")}}
                for index in range(RELIABILITY_BIN_COUNT)]}
        for name in macro:
            eligible_groups = [group for group, table in tables.items() if table[name]["value"] is not None]
            macro[name]["eligible_fold_ids_by_seed"][seed] = eligible_groups
            macro[name]["seed_values"][seed] = (float(np.mean([tables[group][name]["value"] for group in eligible_groups]))
                                               if eligible_groups else None)
    rates = ("precision", "recall", "false_negative_rate", "false_positive_rate", "f1", "specificity")
    scalar_summaries = {name: _seed_scalar_summary(
        {seed: row["rates"][name]["value"] for seed, row in micro.items()},
        {seed: row["rates"][name]["denominator"] for seed, row in micro.items()}) for name in rates}
    scalar_summaries["binary_nll"] = _seed_scalar_summary(
        {seed: row["binary_nll"]["mean"] for seed, row in micro.items()},
        {seed: row["binary_nll"]["count"] for seed, row in micro.items()})
    for name, item in macro.items():
        denominators = {seed: len(ids) for seed, ids in item["eligible_fold_ids_by_seed"].items()}
        item.update(_seed_scalar_summary(item["seed_values"], denominators))
        item["fold_coverage_by_seed"] = {seed: {"eligible": count, "total": len(groups)}
                                         for seed, count in denominators.items()}
        item["weighting"] = "Equal weight for each eligible held-out fold; seed averaging follows; no pooled cross-fit score ranking."
    return {"completed_six_fold_seeds": complete_seeds,
            "all_five_seeds_and_six_folds_completed": len(complete_seeds) == 5,
            "micro_by_seed": micro, "micro_scalar_seed_summaries": scalar_summaries,
            "equal_eligible_fold_macro_ranking": macro}


def _crossfold_classification(lookup: dict, groups: Sequence[str], models: Sequence[str]) -> dict:
    return {model: {cohort: {target: {**_crossfold_classification_table(lookup, groups, model, cohort, target),
                    **({"anchor_truth_strata": {anchor: _crossfold_classification_table(lookup, groups, model, cohort, target, anchor)
                                                for anchor in ("0", "1")}}
                       if target in EVENTS else {})} for target in OCCURRENCE_TARGETS}
                    for cohort in ("raw", "complete_future")} for model in models}


def aggregate_fits(records: Sequence[dict], failures: Sequence[dict] = (), *,
                   model_names: Sequence[str], comparisons: Sequence[tuple[str, str]]) -> dict:
    """Seed averaging precedes equal-component macro; failed fits stay visible."""
    groups = [f"G{i:02d}" for i in range(6)]
    models = tuple(model_names)
    comparisons = tuple(tuple(pair) for pair in comparisons)
    if not models or len(set(models)) != len(models) or any(not isinstance(model, str) or not model for model in models):
        raise ValueError("Fit aggregation requires explicitly declared unique model names")
    if any(len(pair) != 2 or pair[0] not in models or pair[1] not in models or pair[0] == pair[1] for pair in comparisons):
        raise ValueError("Fit comparisons must name distinct models from the declared roster")
    if len(set(comparisons)) != len(comparisons):
        raise ValueError("Duplicate declared model comparison")
    for row in records:
        if row["heldout_group"] not in groups or type(row["training_seed"]) is not int or row["training_seed"] not in range(5) or row["model"] not in models:
            raise ValueError("Fit result lies outside the declared group/seed/model roster")
        if row["occurrence_classification"]["contract"] != occurrence_metric_contract():
            raise ValueError("Fit occurrence metric contract differs from the frozen definitions")
    lookup = {(row["heldout_group"], row["training_seed"], row["model"]): row for row in records}
    if len(lookup) != len(records):
        raise ValueError("Duplicate fit result key")
    per_group = {}
    for group in groups:
        per_group[group] = {}
        for model in models:
            selected = [lookup[(group, seed, model)] for seed in range(5) if (group, seed, model) in lookup]
            values = [row["metrics"]["joint_censored_nll"]["mean"] for row in selected]
            if any(value is None for value in values):
                raise ValueError("Held-out component has no informative NLL windows")
            per_group[group][model] = {"completed_seeds": [row["training_seed"] for row in selected],
                "all_five_seeds_completed": len(selected)==5,
                "joint_censored_nll_mean": float(np.mean(values)) if len(selected)==5 else None,
                "joint_censored_nll_range": [min(values),max(values)] if len(selected)==5 else None,
                "metric_seed_summaries":{metric:{
                    "seed_values":{row["training_seed"]:row["metrics"][metric]["mean"] for row in selected},
                    "eligible_windows_per_seed":{row["training_seed"]:row["metrics"][metric]["count"] for row in selected},
                    "mean":float(np.mean([row["metrics"][metric]["mean"] for row in selected]))
                        if len(selected)==5 and all(row["metrics"][metric]["mean"] is not None for row in selected) else None,
                    "range":[min(row["metrics"][metric]["mean"] for row in selected),max(row["metrics"][metric]["mean"] for row in selected)]
                        if len(selected)==5 and all(row["metrics"][metric]["mean"] is not None for row in selected) else None}
                    for metric in sorted({metric for row in selected for metric in row["metrics"]})},
                "training_seconds_by_seed":{row["training_seed"]:row["training_seconds"] for row in selected},
                "inference_seconds_by_seed":{row["training_seed"]:row["inference_seconds"] for row in selected},
                "structural_trainable_parameters":sorted({row["structural_trainable_parameters"] for row in selected}),
                "likelihood_active_parameters":sorted({row["likelihood_active_parameters"] for row in selected})}
            per_group[group][model]["occurrence_classification_seed_summaries"] = _group_classification(selected)
        for left,right in comparisons:
            pairs = [(seed, lookup[(group,seed,left)]["metrics"]["joint_censored_nll"]["mean"] -
                           lookup[(group,seed,right)]["metrics"]["joint_censored_nll"]["mean"])
                     for seed in range(5) if (group,seed,left) in lookup and (group,seed,right) in lookup]
            per_group[group][left+"-"+right] = {"paired_seed_differences":dict(pairs),
                "mean":float(np.mean([value for _,value in pairs])) if len(pairs)==5 else None,
                "range":[min(value for _,value in pairs),max(value for _,value in pairs)] if len(pairs)==5 else None}
        per_group[group]["support_note"] = ("heldout fold trains on only4314windows" if group=="G00" else
            "no flip-positive targets; event occurrence benefit is unsupported here" if group in ("G01","G04","G05") else "")
    macro = {}
    for key in (*models, *(left + "-" + right for left, right in comparisons)):
        values = [per_group[group][key]["joint_censored_nll_mean" if key in models else "mean"] for group in groups]
        macro[key] = float(np.mean(values)) if all(value is not None for value in values) else None
    weighted = {}
    for model in models:
        selected = [row for row in records if row["model"]==model]
        weighted[model] = (sum(row["metrics"]["joint_censored_nll"]["sum"] for row in selected) /
            sum(row["metrics"]["joint_censored_nll"]["count"] for row in selected)) if len(selected)==30 else None
    return {"completed_fits":len(records), "expected_fits":len(groups)*5*len(models),"failures":list(failures),
        "model_names":list(models),"comparisons":[list(pair) for pair in comparisons],
        "per_group":per_group,"equal_group_macro_after_seed_average":macro,
        "secondary_equal_group_macro":{model:{metric:float(np.mean([
            per_group[group][model]["metric_seed_summaries"][metric]["mean"] for group in groups]))
            if all(metric in per_group[group][model]["metric_seed_summaries"] and
                   per_group[group][model]["metric_seed_summaries"][metric]["mean"] is not None for group in groups)
            else None for metric in sorted({metric for group in groups
                   for metric in per_group[group][model]["metric_seed_summaries"]})} for model in models},
        "secondary_equal_group_macro_seed_ranges":{model:{metric:[
            min(float(np.mean([per_group[group][model]["metric_seed_summaries"][metric]["seed_values"][seed]
                for group in groups])) for seed in range(5)),
            max(float(np.mean([per_group[group][model]["metric_seed_summaries"][metric]["seed_values"][seed]
                for group in groups])) for seed in range(5))]
            if all(metric in per_group[group][model]["metric_seed_summaries"] and
                   per_group[group][model]["metric_seed_summaries"][metric]["mean"] is not None for group in groups)
            else None for metric in sorted({metric for group in groups
                   for metric in per_group[group][model]["metric_seed_summaries"]})} for model in models},
        "window_weighted_supplement":weighted,
        "occurrence_classification_contract":occurrence_metric_contract(),
        "crossfold_occurrence_classification":_crossfold_classification(lookup,groups,models),
        "interpretation":"Retrospective exploratory CV. Seeds are optimization repeats; six fold scores share training sets. No independent-scene significance inference."}
