"""Retained-T baseline adapter: CatBoost, TabM and RuleMC on prepared caches.

Loads the retained T-arm numerical caches (1640 TRAIN, 820 VALID per seed,
q_arm T, exact 2000x64 schedule) through the unchanged training_pilot loader
and never touches TEST, raw logs or the source producer. Every reported metric
uses the identical cached known-cumulative labels and the fixed F10>=.5
decision. The complete-support common-mask cohort is the windows whose T-arm
K8-mean rule PMF has complete support; it is reported separately with its
coverage count, next to the full natural scores that carry an explicit status.

Feature contract for the learned baselines (exact order in FEATURE_ORDER):
normalized 8x2 history, its known mask, K8 known-weighted mean 11x2 candidate
states with known fractions, K8 known-rule-truth mean 11x2 with known
fractions, the declared linear8I prior logits 11x2 and the governed thresholds
(144 columns, one shared order for CatBoost and TabM). No query future operand,
label or sampled future rule enters a feature; the prior logits are visible
inputs only, never hidden forecasts. RuleMC consumes the SAME T sampled-rule
traces as the T-arm Q diagnostic, with the declared gap-aware first-flip
semantics (an observed opposite after preceding gaps establishes the flip,
which the retained diagnostic's drop-at-first-gap helper does not).
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .training_pilot import (GOVERNED_THRESHOLDS, PILOT_VERSION, cumulative_loss,
                             load_prepared_cache, metrics as cumulative_metrics)
from Dataset.world_model.rule_factor_graph.events import EventConstraint
from Dataset.world_model.rule_factor_graph.predicate_trajectory_data import TRUTH_CODES
from Dataset.world_model.rule_factor_graph.communication_cv_tabm import (NumericTabM, TabMConfig,
    make_tabm_optimizer, tabm_log_pmf, tabm_member_censored_nll, tabm_parameter_manifest)

BASELINE_VERSION = PILOT_VERSION + ':baselines/v1'
CATBOOST_VERSION = "1.2.10"
DECISION_F10 = .5
CATBOOST_ITERATIONS = 2000
CATBOOST_DEPTH = 6
CATBOOST_LEARNING_RATE = .03
TABM_UPDATES = 2000
TABM_BATCH_SIZE = 64
HEADS = ('L', 'P')
NUMERIC_FIELD_NAMES = ('latency_mean_ms', 'packet_loss_ratio')

# Exact column order of the 144 learned-baseline features (one shared order for
# CatBoost and TabM). Field axis f0 is latency_mean_ms (governed threshold
# 120 ms) and f1 is packet_loss_ratio (governed threshold .05), matching RULES
# and GOVERNED_THRESHOLDS order; rule blocks follow the same L then P order.
FEATURE_ORDER = (
    *[f'normalized_history_t{t}_{NUMERIC_FIELD_NAMES[f]}' for t in range(8) for f in (0, 1)],
    *[f'history_known_t{t}_{NUMERIC_FIELD_NAMES[f]}' for t in range(8) for f in (0, 1)],
    *[f'k8_known_weighted_mean_state_t{t}_{NUMERIC_FIELD_NAMES[f]}' for t in range(11) for f in (0, 1)],
    *[f'state_known_fraction_t{t}_{NUMERIC_FIELD_NAMES[f]}' for t in range(11) for f in (0, 1)],
    *[f'k8_known_rule_truth_mean_{HEADS[p]}_t{t}' for p in (0, 1) for t in range(11)],
    *[f'rule_known_fraction_{HEADS[p]}_t{t}' for p in (0, 1) for t in range(11)],
    *[f'linear_prior_logit_{HEADS[p]}_bin{b}' for p in (0, 1) for b in range(11)],
    'governed_latency_threshold_ms', 'governed_loss_threshold',
)


def _truth_codes(prepared):
    """[B,K,11,F] four-valued sampled rule codes rebuilt from the saved cache.

    assemble_trajectory_batch defined sampled_truth = code == true,
    truth_known = code in {true,false} and applicable = code != out_of_scope;
    that triple reconstructs the codes exactly.
    """
    truth = prepared.tensors['sampled_truth'].cpu().numpy()
    known = prepared.tensors['truth_known'].cpu().numpy()
    applicable = prepared.tensors['applicable'].cpu().numpy()
    codes = np.full(known.shape, TRUTH_CODES['unknown'], np.int8)
    codes[known] = np.where(truth[known], TRUTH_CODES['true'], TRUTH_CODES['false'])
    codes[~applicable] = TRUTH_CODES['out_of_scope']
    return codes


def baseline_features(prepared, *, split):
    """Declared feature blocks plus the RuleMC trace inputs for one split.

    Candidate states and rule truth are the T-arm sampled donor operands and
    fixed-rule traces already inside the cache; no query future enters.
    """
    if prepared.split != split:
        raise ValueError(f'prepared cache split {prepared.split} differs from requested {split}')
    history = prepared.tensors['history'].cpu().numpy()
    history_known = prepared.tensors['history_known'].cpu().numpy()
    states = prepared.tensors['sampled_states'].cpu().numpy()
    state_known = prepared.tensors['state_known'].cpu().numpy()
    prior = prepared.tensors['prior_logits'].cpu().numpy()
    known = prepared.tensors['target_known'].cpu().numpy()
    targets = prepared.tensors['targets'].cpu().numpy()
    codes = _truth_codes(prepared)
    known_codes = (codes == TRUTH_CODES['true']) | (codes == TRUTH_CODES['false'])
    rule_mean = np.divide(np.where(known_codes, codes, 0).sum(1), np.maximum(known_codes.sum(1), 1))
    state_weighted = np.divide(np.where(state_known, states, 0.).sum(1), np.maximum(state_known.sum(1), 1))
    blocks = (
        np.where(history_known, history, 0.).reshape(len(history), -1),
        history_known.astype(np.float64).reshape(len(history), -1),
        state_weighted.reshape(len(history), -1),
        state_known.astype(np.float64).mean(1).reshape(len(history), -1),
        rule_mean.transpose(0, 2, 1).reshape(len(history), -1),
        known_codes.astype(np.float64).mean(1).transpose(0, 2, 1).reshape(len(history), -1),
        prior.reshape(len(history), -1),
        np.broadcast_to(GOVERNED_THRESHOLDS.astype(np.float64), (len(history), 2)),
    )
    features = np.concatenate(blocks, 1)
    if features.shape != (len(history), 144) or not np.isfinite(features).all():
        raise ValueError('learned baseline features must be finite with the declared 144 columns')
    if prior.shape != (len(history), 2, 11) or known.shape != (len(history), 2, 10):
        raise ValueError('prepared supervision tensors have unexpected shapes')
    return dict(features=features, codes=codes,
                targets=targets, known=known,
                compatible=prepared.tensors['compatible'].cpu().numpy())


def rulemc_first_flip_pmf(codes):
    """[...,P,11] K8-mean first-flip PMF from the SAME T sampled-rule traces.

    Accepts [K,11,F] for one window or [B,K,11,F] for a cohort and maps over
    the leading axes. Declared RuleMC semantics: per donor, the anchor trace
    code must be observed; an observed opposite at any later instant
    establishes the flip at that bin even after preceding UNKNOWN/OOS gaps,
    and the no-flip tail bin requires the complete observed prefix. (The Q
    diagnostic helper candidate_first_flip_pmf instead abandons a donor at its
    first gap; RuleMC keeps the established opposite.) Donors that establish
    neither contribute no mass, so the K-mean total mass is the resolved
    support and the remainder stays unresolved UNKNOWN branch mass, never
    false and never renormalized.
    """
    k, steps, fields = codes.shape[-3:]
    leading = codes.shape[:-3]
    batch = int(np.prod(leading, dtype=np.int64)) if leading else 1
    codes = codes.reshape((batch, k, steps, fields))
    observed = (codes == TRUTH_CODES['true']) | (codes == TRUTH_CODES['false'])
    mass = np.zeros((batch, k, fields, steps), dtype=np.float64)
    for index in range(len(codes)):
        for f in range(fields):
            for d in range(k):
                if not observed[index, d, 0, f]:
                    continue
                anchor = codes[index, d, 0, f]
                placed, complete = False, True
                for t in range(1, steps):
                    if observed[index, d, t, f]:
                        if codes[index, d, t, f] != anchor:
                            mass[index, d, f, t - 1] = 1.
                            placed = True
                            break
                    else:
                        complete = False
                if not placed and complete:
                    mass[index, d, f, 10] = 1.
    return (mass.mean(1).reshape(leading + (fields, steps)),
            (~observed).sum((1, 2)).reshape(leading + (fields,)))


def natural_rulemc_pmf(pmf):
    """[B,P,11] mass to a -inf-padded log PMF plus the per-predicate support.

    A predicate has support exactly when its K-mean event mass sums to one
    (all eight donors resolved); otherwise its branch mass is unresolved, the
    log PMF is all -inf, and the window is excluded from natural scores and
    counted, with no renormalization or invented tail.
    """
    resolved = np.isclose(pmf.sum(-1), 1., rtol=0., atol=1e-12)
    with np.errstate(divide='ignore'):
        log_pmf = torch.as_tensor(np.where(resolved[..., None], np.log(np.where(
            resolved[..., None], pmf, 1.)), -np.inf), dtype=torch.float64)
    return log_pmf, resolved


def window_events(targets, known, compatible):
    """Per-window EventConstraint rows from the cached cumulative supervision.

    The supervision's known pattern is an initial observed prefix (target 0),
    optionally followed by a gap and, after the first observed opposite, known
    cells that all carry target 1. The event is the first positive known
    horizon (a first flip) or, with no positive, right censoring after the
    complete observed prefix.
    """
    events = []
    for i in range(len(targets)):
        row = {}
        for p, name in enumerate(HEADS):
            positions = np.flatnonzero(known[i, p].astype(bool))
            if len(positions) == 0:
                row[name] = EventConstraint('right_censored', 0)
                continue
            positives = positions[targets[i, p, positions] == 1]
            if len(positives):
                event = EventConstraint('first_flip', int(positives[0]) + 1)
                if int(positives[0]) + 1 != event.step or targets[i, p, :event.step - 1][
                        known[i, p, :event.step - 1]].astype(bool).any():
                    raise ValueError('a known negative horizon after the first flip is impossible')
            else:
                event = EventConstraint('right_censored', int(positions[-1]) + 1)
                if not known[i, p, :event.step].all():
                    raise ValueError('a zero-target censoring requires a complete known prefix')
            row[name] = event
        events.append(row)
    support = compatible.astype(bool)
    for i, row in enumerate(events):
        for p, name in enumerate(HEADS):
            columns = np.flatnonzero(support[i, p])
            event = row[name]
            # Cached event-likelihood support (supervision()): a first flip at
            # zero-indexed position F admits the gap-aware interval
            # [last_known_negative+1, F]; right censoring after the last known
            # negative position s admits bins s+1..10; no known horizon admits
            # all eleven.
            negatives = np.flatnonzero(known[i, p].astype(bool) & (targets[i, p] == 0))
            if event.kind == 'first_flip':
                last_negative = int(negatives[-1]) if len(negatives) else -1
                expected = np.arange(last_negative + 1, event.step)
            elif event.step == 0:
                expected = np.arange(11)
            else:
                expected = np.arange(event.step, 11)
            if len(columns) != len(expected) or (columns != expected).any():
                raise ValueError(f'cached compatible support disagrees with the {event.kind} event of {name}')
    return events


def hazard_rows(targets, known):
    """Per-window risk rows from exactly observed horizons.

    An observed negative horizon contributes (bin j+1, label 0); the first
    observed positive horizon contributes (bin j+1, label 1). Gap horizons and
    the later known positives of an already flipped window contribute nothing:
    unknown timing is never interval-censored as false and the process is dead
    after the flip. window_events separately cross-checks the cached
    event-likelihood support used by the TabM censored NLL.
    """
    rows = []
    for i in range(len(targets)):
        item = {}
        for p, name in enumerate(HEADS):
            positions = np.flatnonzero(known[i, p].astype(bool))
            positives = positions[targets[i, p, positions] == 1] if len(positions) else positions
            if len(positives):
                first = int(positives[0])
                negatives = np.flatnonzero(known[i, p] & (targets[i, p] == 0))
                if targets[i, p, first + 1:][known[i, p, first + 1:]].sum() != (
                        known[i, p, first + 1:].sum()):
                    raise ValueError('post-flip known horizons must all carry the flip label')
                if negatives.size and int(negatives[-1]) > first:
                    raise ValueError('a known negative horizon after the first positive is impossible')
                bins = np.append(negatives, first).astype(np.int64) + 1
                labels = np.append(np.zeros(len(negatives), np.int64), 1)
            elif len(positions):
                if not known[i, p, :int(positions[-1]) + 1].all():
                    raise ValueError('a zero-target censoring requires a complete known prefix')
                bins = positions.astype(np.int64) + 1
                labels = np.zeros(len(positions), dtype=np.int64)
            else:
                bins = np.empty(0, dtype=np.int64)
                labels = np.empty(0, dtype=np.int64)
            item[name] = (bins, labels)
        rows.append(item)
    return rows


def _risk_rows(event):
    """Observed risk-bin ids and binary hazard labels; censor zero has no rows."""
    if event.kind == 'first_flip':
        if not 1 <= event.step <= 10:
            raise ValueError('first-flip risk time must lie in 1..10')
        target = np.zeros(event.step, dtype=np.int64)
        target[-1] = 1
    elif event.kind == 'right_censored':
        if not 0 <= event.step <= 10:
            raise ValueError('observed censor time must lie in 0..10')
        target = np.zeros(event.step, dtype=np.int64)
    else:
        raise ValueError('hazard rows need a first_flip or right_censored event')
    return np.arange(1, len(target) + 1, dtype=np.int64), target


def hazard_log_pmf(probability):
    """P(survive),P(first flip at bin) columns to an eleven-bin log PMF.

    Both native Probability columns are used directly with no subtraction from
    one and no epsilon clipping; an exact 0/1 endpoint is an error.
    """
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (10, 2) or not np.isfinite(values).all():
        raise ValueError('native hazard probabilities must be finite [B,10,2]')
    if ((values <= 0) | (values >= 1)).any():
        raise FloatingPointError('CatBoost Probability produced an exact 0/1 endpoint; no clipping')
    logs = np.log(values)
    survived = np.zeros(len(values))
    columns = []
    for b in range(10):
        columns.append(survived + logs[:, b, 1])
        survived = survived + logs[:, b, 0]
    columns.append(survived)
    return torch.as_tensor(np.stack(columns, 1))


def pava_project(cdf):
    """Explicit monotone projection of the ten CDF prefixes onto [0,1].

    Isotonic regression (pool adjacent violators, unit weights) replaces the
    non-monotone native CDF values; the projected CDF is then first-differenced
    into the declared eleven-bin PMF, whose tail bin keeps the surviving mass.
    """
    values = np.asarray(cdf, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 10 or not np.isfinite(values).all():
        raise ValueError('PAVA projection needs finite [B,10] CDF prefixes')
    values = np.clip(values, 0., 1.)
    projected = np.empty(values.shape, dtype=np.float64)
    for i in range(len(values)):
        sums, counts = [float(values[i, 0])], [1.0]
        for value in values[i, 1:]:
            sums.append(float(value))
            counts.append(1.0)
            while len(sums) > 1 and sums[-2] / counts[-2] > sums[-1] / counts[-1]:
                value, count = sums.pop(), counts.pop()
                sums[-1] += value
                counts[-1] += count
        position = 0
        for total, count in zip(sums, counts):
            projected[i, position:position + int(count)] = total / count
            position += int(count)
    projected = np.clip(projected, 0., 1.)
    bins = np.empty((len(values), 11), dtype=np.float64)
    bins[:, 0] = projected[:, 0]
    bins[:, 1:10] = np.diff(projected, axis=1)
    bins[:, 10] = 1. - projected[:, -1]
    if (bins < -1e-12).any():
        raise FloatingPointError('PAVA projection produced negative PMF mass')
    if not np.allclose(bins.sum(-1), 1., rtol=0., atol=1e-9):
        raise FloatingPointError('PAVA-projected PMF does not telescope to one')
    return bins


def fit_catboost(features, targets, known, compatible, *, seed, iterations=CATBOOST_ITERATIONS):
    """Shared horizon-conditioned native GPU Logloss classifier per predicate.

    One classifier per head; the horizon bin is an explicit input column, so a
    single tree ensemble scores all ten hazard bins. Every observed known
    horizon contributes its own risk row; gap horizons contribute nothing and
    interval-censored labels are never recorded as false. Prepared caches
    declare no component grouping, so each window is its own group and the
    declared window weights are uniform.
    """
    from catboost import CatBoostClassifier, Pool, __version__
    if __version__ != CATBOOST_VERSION:
        raise RuntimeError(f'CatBoost version mismatch: {__version__}')
    config = dict(loss_function='Logloss', task_type='GPU', devices='0', depth=CATBOOST_DEPTH,
                  iterations=iterations, learning_rate=CATBOOST_LEARNING_RATE, l2_leaf_reg=3.0,
                  random_seed=seed, thread_count=2, gpu_ram_part=.35, allow_writing_files=False,
                  verbose=False)
    groups = len(features)
    rows = hazard_rows(targets, known)
    heads, diagnostics = {}, {}
    for name in HEADS:
        indices, bins, labels, weights = [], [], [], []
        for i, item in enumerate(rows):
            window_bins, window_labels = item[name]
            if len(window_bins) == 0:
                continue  # uninformative window: no observed horizon label
            indices.extend([i] * len(window_bins))
            bins.extend(window_bins)
            labels.extend(window_labels)
            weights.extend([1 / groups] * len(window_bins))
        if not indices:
            raise ValueError(f'CatBoost {name} requires at least one observed hazard label')
        x = np.concatenate((features[np.asarray(indices)], np.asarray(bins)[:, None]), -1).astype(np.float32)
        labels, weights = np.asarray(labels), np.asarray(weights)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f'CatBoost {name} requires both hazard classes')
        weights *= len(weights) / weights.sum()
        model = CatBoostClassifier(**config)
        started = time.perf_counter()
        model.fit(Pool(x, labels, weight=weights))
        losses = model.get_evals_result()['learn']['Logloss']
        diagnostics[name] = dict(risk_rows=len(labels), positives=int(labels.sum()),
                                 trees=int(model.tree_count_), seconds=time.perf_counter() - started,
                                 loss_curve=losses, actual_native_params=model.get_all_params())
        heads[name] = model
        print(f'CatBoost {name} seed={seed} trees={model.tree_count_}', flush=True)
    return heads, dict(configuration=config, feature_count=features.shape[1] + 1, heads=diagnostics,
                       version=__version__,
                       objective='per-window uniform weighted censored risk terms SUM; '
                                 'prepared caches declare no component grouping',
                       shared_horizon_conditioned_classifier_per_predicate=True,
                       interval_censor_policy='every exactly observed horizon contributes its risk '
                                              'row; gap horizons contribute nothing; '
                                              'censor-as-false never applied',
                       projection='explicit monotone PAVA of the hazard CDF, then first-difference '
                                  '11-bin PMF', epsilon_clipping=False)


def catboost_split_pmf(heads, features, *, split_name):
    """Native hazard CDF for one split, then the declared PAVA-projected PMF."""
    rows = np.concatenate((np.repeat(features, 10, axis=0),
                           np.tile(np.arange(1, 11), len(features))[:, None]), -1).astype(np.float32)
    native_cdfs = []
    for name in HEADS:
        probability = heads[name].predict(rows, prediction_type='Probability', task_type='GPU', thread_count=2)
        native = hazard_log_pmf(np.asarray(probability).reshape(len(features), 10, 2))
        cdf = torch.exp(torch.logcumsumexp(native, -1)[..., :10])
        native_cdfs.append(cdf)
    native_cdf = torch.stack(native_cdfs, 1).numpy()
    pmf = pava_project(native_cdf[:, 0]), pava_project(native_cdf[:, 1])
    with np.errstate(divide='ignore'):
        log_pmf = torch.as_tensor(np.stack([np.log(pmf[0]), np.log(pmf[1])], 1), dtype=torch.float64)
    diagnostics = dict(split=split_name,
                       native_cdf_nonmonotone_cells=int(sum((np.diff(item, axis=1) < -1e-12).sum()
                                                            for item in native_cdfs)),
                       native_cdf_out_of_range_cells=int(sum(((item < 0) | (item > 1)).sum()
                                                             for item in native_cdfs)),
                       native_cdf_out_of_unit_interval=bool(any(((item < 0) | (item > 1)).any()
                                                                for item in native_cdfs)))
    return log_pmf, diagnostics


def fit_tabm(train, valid, *, seed, device, schedule, input_features, updates=TABM_UPDATES,
             batch_size=TABM_BATCH_SIZE):
    """Official TabM numeric adaptation on the declared feature width.

    Member-wise censored NLL on TRAIN, mean member probabilities for inference;
    the exact retained 2000x64 schedule drives every update. Evaluation rows
    report the same hybrid cumulative loss as the neural pilot, full cohort.
    """
    if schedule.shape != (updates, batch_size):
        raise ValueError('TabM requires the retained exact 2000x64 schedule')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = NumericTabM(TabMConfig(), input_features=input_features).to(device)
    optimizer = make_tabm_optimizer(model)
    x = torch.as_tensor(train['features'], dtype=torch.float32, device=device)
    xv = torch.as_tensor(valid['features'], dtype=torch.float32, device=device)
    cohort = {}
    for split, features_x, item in (('train', x, train), ('validation', xv, valid)):
        cohort[split] = (features_x,
                         torch.as_tensor(item['targets'], dtype=torch.float64, device=device),
                         torch.as_tensor(item['known'], dtype=torch.bool, device=device))
    losses, evaluations = [], []

    def evaluate(step):
        model.eval()
        row = dict(step=step)
        with torch.no_grad():
            for split, (features_x, targets, known) in cohort.items():
                log_pmf = torch.cat([tabm_log_pmf(model, features_x[i:i + 256])
                                     for i in range(0, len(features_x), 256)])
                row[f'full_{split.upper()}_hybrid'] = float(cumulative_loss(log_pmf, targets, known, 'hybrid'))
        evaluations.append(row)
        print(json.dumps(row), flush=True)

    evaluate(0)
    for step, indices in enumerate(schedule, 1):
        rows = torch.as_tensor(indices, dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = tabm_member_censored_nll(model(x[rows]),
                                        torch.as_tensor(train['compatible'], dtype=torch.bool, device=device)[rows])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite TabM member censored NLL')
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if step % 200 == 0 or step == updates:
            evaluate(step)
    return model, optimizer, losses, evaluations, dict(optimizer_updates=updates, batch_size=batch_size,
                                            learning_rate=model.config.learning_rate,
                                            weight_decay=model.config.weight_decay,
                                            member_ensemble=model.config.ensemble_members,
                                            input_features=input_features,
                                            loss_first=losses[0], loss_last=losses[-1],
                                            loss_minimum=min(losses),
                                            parameter_manifest=tabm_parameter_manifest(model))


def evaluate_tabm(model, tabm_features, *, device):
    model.eval()
    x = torch.as_tensor(tabm_features, dtype=torch.float32, device=device)
    with torch.no_grad():
        return torch.cat([tabm_log_pmf(model, x[i:i + 256]) for i in range(0, len(x), 256)]).cpu()


def report(log_pmf, prepared, *, resolved=None):
    """Full natural scores plus complete-support common-mask metrics and coverage.

    Natural scores use only labels whose support is actually defined (the
    per-predicate resolved support); unresolved UNKNOWN branch mass is N/A and
    counted, never renormalized, turned into false, or scored as a zero-CDF
    non-event. The common-mask block restricts to windows whose whole K8 rule
    support is complete.
    """
    pmf = log_pmf.exp().cpu().numpy()
    log_numpy = log_pmf.cpu().numpy()
    targets = prepared.tensors['targets'].cpu().numpy()
    known = prepared.tensors['target_known'].cpu().numpy()
    compatible = prepared.tensors['compatible'].cpu().numpy()
    support = np.ones(pmf.shape[:2], bool) if resolved is None else np.asarray(resolved, bool)
    if support.shape != pmf.shape[:2]:
        raise ValueError('resolved support must align with [B,P]')
    scored = known & support[:, :, None]
    natural = cumulative_metrics(pmf, targets, scored, compatible, log_pmf=log_numpy)
    natural['score_status'] = ('complete_support' if bool(support.all()) else
                               'per-predicate resolved support only; unresolved UNKNOWN branch mass is '
                               'N/A, never renormalized or turned into false')
    for p, name in enumerate(HEADS):
        natural[name]['excluded_unresolved_known_cdf_cells'] = int((known[:, p] & ~support[:, p, None]).sum())
    complete = support.all(1)
    common = cumulative_metrics(pmf[complete], targets[complete], known[complete], compatible[complete],
                                log_pmf=log_numpy[complete])
    coverage = dict(windows=len(pmf), complete_support_windows=int(complete.sum()),
                    unresolved_windows=int((~complete).sum()),
                    resolved_by_predicate=support.sum(0).astype(int).tolist(),
                    known_cdf_cells=int(known.sum()),
                    common_mask_known_cdf_cells=int(known[complete].sum()))
    return dict(natural=natural, complete_support_common_mask=common, coverage=coverage,
                pmf_sum_min=float(pmf.sum(-1).min()), pmf_sum_max=float(pmf.sum(-1).max()),
                supports_complete_probability=bool((pmf.sum(-1) == 1).all()))


def run_baselines(args):
    if 'CatBoost' in args.methods and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError('CatBoost GPU fit requires the launcher to set CUDA_VISIBLE_DEVICES')
    root = Path(args.root)
    source = Path(args.prepared_root) if Path(args.prepared_root).is_absolute() else root / args.prepared_root
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    roots = args.prepared_roots or [str(source)] * len(args.seeds)
    if len(roots) != len(args.seeds):
        raise ValueError('one prepared root per seed is required when several are given')
    roots = [item if Path(item).is_absolute() else root / item for item in roots]
    roots = [Path(item) / f'T_seed{seed}' for item, seed in zip(roots, args.seeds)]
    device = torch.device(args.device)
    protocol = dict(version=BASELINE_VERSION, prepared_root=str(source.resolve()),
                    prepared_root_by_seed={str(s): str(Path(r).resolve())
                                           for s, r in zip(args.seeds, roots) if Path(r).resolve() != source.resolve()},
                    seeds=args.seeds, device=args.device, decision_F10=DECISION_F10,
                    catboost_iterations=CATBOOST_ITERATIONS, catboost_depth=CATBOOST_DEPTH,
                    catboost_learning_rate=CATBOOST_LEARNING_RATE,
                    tabm_updates=TABM_UPDATES, tabm_batch_size=TABM_BATCH_SIZE,
                    tabm_learning_rate=TabMConfig().learning_rate,
                    methods=list(args.methods),
                    feature_order=list(FEATURE_ORDER), feature_count=len(FEATURE_ORDER),
                    tabm_feature_order=list(FEATURE_ORDER), tabm_feature_count=len(FEATURE_ORDER),
                    tabm_shared_feature_contract='CatBoost and TabM consume the same 144-column '
                                                 'feature matrix; the member ensemble and objective differ',
                    feature_sources=dict(
                        normalized_history='prepared cache history, TRAIN-bank mean/scale already applied',
                        history_known='prepared cache history_known',
                        k8_known_weighted_mean_state='K8-mean of sampled_states under state_known, per instant',
                        state_known_fraction='state_known fraction over K8, per instant',
                        k8_known_rule_truth_mean='K8-mean of fixed-rule codes where true/false under truth_known, per instant',
                        rule_known_fraction='truth_known fraction over K8, per instant',
                        linear_prior_logits='declared linear8I prior_logits saved in the prepared cache',
                        governed_thresholds='[120 ms, .05] governed parameters'),
                    tabm_prior_policy='prior logits are visible input features; no hidden forecast path',
                    rulemc_traces='SAME retained T sampled_rule traces as the T-arm Q diagnostic',
                    unknown_branch_policy='unresolved mass is reported, never renormalized or turned into false',
                    metric_contract=dict(labels='identical cached known cumulative labels',
                                         decision='fixed F10>=.5',
                                         common_mask='complete K8-rule-support windows only'))
    combined = {}
    for seed, cache_root in zip(args.seeds, roots):
        prepared, features, schedule = {}, {}, None
        for split in ('train', 'validation'):
            prepared[split], metadata = load_prepared_cache(
                cache_root / f'prepared_{split}__T__seed{seed}.npz', device=device)
            if str(metadata['q_arm']) != 'T':
                raise ValueError(f'retained cache belongs to arm {metadata["q_arm"]}, not T')
            features[split] = baseline_features(prepared[split], split=split)
            if schedule is None:
                schedule = metadata['schedule']
        if schedule is None or schedule.shape != (TABM_UPDATES, TABM_BATCH_SIZE):
            raise ValueError('requested update/batch budget differs from the retained exact schedule')
        protocol.setdefault('counts', {})[str(seed)] = {
            split: dict(windows=prepared[split].size,
                        known_cdf_cells=int(prepared[split].tensors['target_known'].sum()))
            for split in ('train', 'validation')}
        combined[str(seed)] = {}
        for method in args.methods:
            started = time.perf_counter()
            if method == 'CatBoost':
                heads, native = fit_catboost(features['train']['features'], features['train']['targets'],
                                             features['train']['known'], features['train']['compatible'],
                                             seed=seed)
                destination = out / f'catboost__seed{seed}'
                destination.mkdir()
                for name, model in heads.items():
                    model.save_model(destination / f'{name}.cbm', format='cbm')
                for split in ('train', 'validation'):
                    log_pmf, projection = catboost_split_pmf(heads, features[split]['features'],
                                                             split_name=split)
                    summary = report(log_pmf, prepared[split])
                    summary['projection_diagnostics'] = projection
                    np.savez_compressed(destination / f'{split}_pmf.npz', pmf=log_pmf.exp().numpy(),
                                        log_pmf=log_pmf.numpy(),
                                        targets=features[split]['targets'],
                                        target_known=features[split]['known'],
                                        compatible=features[split]['compatible'])
                    combined[str(seed)].setdefault(method, {})[split] = summary
                (destination / 'native_metadata.json').write_text(json.dumps(native, indent=2, default=float))
            elif method == 'TabM':
                model, optimizer, losses, evaluations, native = fit_tabm(
                    features['train'], features['validation'], seed=seed, device=device,
                    schedule=schedule, input_features=features['train']['features'].shape[1])
                destination = out / f'tabm__seed{seed}'
                destination.mkdir()
                torch.save(dict(step=TABM_UPDATES, model=model.state_dict(),
                                optimizer=optimizer.state_dict(), losses=losses,
                                evaluations=evaluations, seed=seed,
                                input_features=features['train']['features'].shape[1],
                                config=asdict(model.config)), destination / 'final_state.pt')
                for split in ('train', 'validation'):
                    log_pmf = evaluate_tabm(model, features[split]['features'], device=device)
                    summary = report(log_pmf, prepared[split])
                    np.savez_compressed(destination / f'{split}_pmf.npz', pmf=log_pmf.exp().numpy(),
                                        log_pmf=log_pmf.numpy(), targets=features[split]['targets'],
                                        target_known=features[split]['known'],
                                        compatible=features[split]['compatible'])
                    combined[str(seed)].setdefault(method, {})[split] = summary
                (destination / 'native_metadata.json').write_text(json.dumps(native, indent=2))
            elif method == 'RuleMC':
                destination = out / f'rulemc__seed{seed}'
                destination.mkdir()
                for split in ('train', 'validation'):
                    item = features[split]
                    pmf, missing = rulemc_first_flip_pmf(item['codes'])
                    log_pmf, resolved = natural_rulemc_pmf(pmf)
                    summary = report(log_pmf, prepared[split], resolved=resolved)
                    summary['coverage']['missing_donor_cells_by_predicate'] = (
                        missing.reshape(len(item['targets']), -1).sum(0).astype(int).tolist())
                    np.savez_compressed(destination / f'{split}_pmf.npz', pmf=pmf,
                                        log_pmf=log_pmf.numpy(), rule_resolved=resolved,
                                        missing_donor_cells=missing,
                                        targets=item['targets'], target_known=item['known'],
                                        compatible=item['compatible'])
                    combined[str(seed)].setdefault(method, {})[split] = summary
            else:
                raise ValueError(f'undeclared baseline method: {method}')
            print(json.dumps(dict(method=method, seed=seed, seconds=time.perf_counter() - started)), flush=True)
    (out / 'combined_report.json').write_text(json.dumps(
        dict(version=BASELINE_VERSION, per_seed=combined), indent=2, allow_nan=False))
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False))
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/mnt/data2/weizhiwei/AERO_WORLD')
    parser.add_argument('--prepared-root',
                        default='Dataset/world_model/rule_factor_graph/runs/p01_ns3_transported_q')
    parser.add_argument('--prepared-roots', nargs='*')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--methods', nargs='+', default=['CatBoost', 'TabM', 'RuleMC'],
                        choices=['CatBoost', 'TabM', 'RuleMC'])
    run_baselines(parser.parse_args())


if __name__ == '__main__':
    main()
