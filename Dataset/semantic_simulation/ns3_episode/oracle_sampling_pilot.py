"""Oracle noisy-future information diagnostic, K1 versus K8.

This pilot intentionally supplies the actual query future numeric states,
perturbed once by a declared Gaussian, so its outputs are an upper-information
ORACLE/NOISY-FUTURE INFORMATION DIAGNOSTIC. They are never causal forecasts or
deployable accuracy: at every query the sampler reads that query's own future
operands. The learned Q law, nearest donors, prefix gate and auxiliary losses
are all absent here; only the sample count K differs between the two arms.

Inputs are the retained transported-Q caches: frozen 1640 TRAIN / 820 VALID
windows in their exact retained order, the per-seed fixed batch schedules,
frozen linear8I prior, cumulative labels, compatibility and the
compiled layout/topology metadata. History reads actual R1 source rows on an
explicit inclusive grid, defaulting to 3.5 seconds at 0.5-second intervals; a
longer duration requires existing earlier source rows and retains their masks.
From the same current R1 source logs the
pilot reads only latency_mean_ms and packet_loss_ratio at anchor+offset for
offsets 0,5,...,50, preserving each query's UNKNOWN / OUT_OF_SCOPE cells.

Sampler: one Gaussian per query with mean equal to that query's actual future
vector and sigma = 0.1 * retained TRAIN normalization_scale, frozen from the
retained protocol before any VALID use (no forecast model is fitted). K8 noise
is drawn once per seed/query with numpy.random.default_rng(seed*1000003+query)
for future points only; K1 is exactly the first draw of that same stream. The
anchor stays the unperturbed observed anchor. Known cells only are sampled,
and the physical projection latency to [0,(query packet_ttl_ns-1)/1e6] ms and
loss to [0,1] is declared, with pre-projection violation and endpoint counts
and no epsilon.

K1 and K8 differ only in sample count: same parameters, initialization,
schedule, labels, thresholds and losses. Both arms run through the shared
dense forward with the model's own equal logmeanexp across the actual K. The
PyG assembly metadata of the L/P slice remains an assembly artifact, not the
neural backbone.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import numpy as np
import torch

from .training_adapter import (load_r1_logs, RunSelection, R1TrainingLogs, RULES,
                               TICK_NS, GRID_NS, _metric, WindowKey, FlowKey, PredictionPrefix)
from .training_pilot import (metrics, initialize_outcome_skip,
                             evaluate_full)
from Dataset.world_model.rule_factor_graph.predicate_trajectory_models import SharedPredicateTrajectoryModel
from Dataset.world_model.rule_factor_graph.predicate_trajectory_data import STATE_OFFSETS, TRUTH_CODES, EpisodeContract, assemble_trajectory_batch
from Dataset.world_model.rule_factor_graph.predicate_trajectory_oracle import (
    StateFieldNoise, FieldTransform, OracleStateSampler, OracleRuleEvaluator,
    build_candidate_record, apply_field_transforms, compute_first_flip_supervision)
from Dataset.world_model.rule_factor_graph.predicate_trajectory_training import (
    PreparedTrajectorySplit, train_step)

ORACLE_VERSION = 'p01.ns3.oracle-noisy-future-diagnostic/generic-v3'
# Donor-Q fields of the retained protocol that do not describe this diagnostic.
ORACLE_DROPPED_PROTOCOL_KEYS = frozenset((
    'version', 'q_arms', 'q_transport_law', 'q_diagnostic', 'q_common_protocol',
    'candidate_selection', 'sampling_off', 'sampling_on', 'reliance_mode',
    'reliance_gate_parameters', 'prepared_cache_reuse', 'model_capacity',
    'component_supervision_weight', 'component_supervision_formula'))
ORACLE_LAW = ('oracleGaussian: mean = actual query future operands read from the current '
              'R1 source logs at anchor+offset 0,5,...,50 (latency_mean_ms, packet_loss_ratio); '
              'sigma = 0.1 * retained TRAIN normalization_scale, frozen from the retained '
              'transported-Q protocol before VALID use; no forecast model is fitted; K8 noise '
              'drawn once per seed/query from numpy.random.default_rng(seed*1000003+query), '
              'future points only; K1 is exactly the first draw; the anchor stays the '
              'unperturbed observed anchor; known cells only')
ORACLE_LABEL = ('ORACLE/NOISY-FUTURE INFORMATION DIAGNOSTIC; intentionally reads the query '
                'future states; never a causal forecast or deployable accuracy')
# Declared field order of the read source metrics, joined to RULES symbol order.
ORACLE_FIELDS = tuple((symbol, field, unit, dimension, threshold)
                      for symbol, _, field, unit, dimension, _, threshold in RULES)
SIGMA_FRACTION = 0.1
FUTURE_OFFSETS = STATE_OFFSETS[1:]
FUTURE_STEPS = len(FUTURE_OFFSETS)
HISTORY_STEPS = 8
ANCHOR_STEPS = 11
ORACLE_CACHE_K = 8
ORACLE_ARMS = (1, 8)
WINDOW_KEY_FIELDS = ('episode_id', 'profile', 'run', 'link_id', 'flow_id', 'src_node_id',
                     'owner', 'life_epoch', 'dst_node_id', 'receiver_owner', 'receiver_life_epoch')
L_P_SLICE_NOTE = ('L/P numeric slice only: latency_mean_ms (ms, time) and packet_loss_ratio '
                  '(1, dimensionless) are read per exact (flow, anchor+offset) source row. RULES '
                  'thresholds are evaluated strictly through the declared (symbol, field, unit, '
                  'dimension, threshold) order, never by predicate-string heuristics. The dense '
                  'SharedPredicateTrajectoryModel forward is the neural backbone; the PyG '
                  'trajectory graph of this slice is assembly metadata only and is neither '
                  'rebuilt nor used for training here.')
# Static tensors reused verbatim from the retained T caches; the per-query
# sampled_*/label arrays are rebuilt by this diagnostic.
ORACLE_REUSED_CACHE_ARRAYS = frozenset((
    'history', 'history_known', 'prior_logits', 'targets', 'target_known', 'compatible',
    'linear_bins', 'linear_available', 'predicate_field_incidence', 'owner_field_incidence',
    'field_type_ids', 'window_keys'))


def oracle_seed(seed, query):
    """Declared exact per-query draw stream."""
    return int(seed) * 1_000_003 + int(query)


def retained_normalization_scale(protocol):
    scale = np.asarray(protocol['normalization_scale'], dtype=float)
    if scale.shape != (len(ORACLE_FIELDS),) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('retained protocol lacks the declared per-field normalization scale')
    return scale


def frozen_sigma(protocol):
    """sigma = 0.1 * retained TRAIN scale, physical units, frozen before VALID."""
    return SIGMA_FRACTION * retained_normalization_scale(protocol)


def frozen_sigma_scale(protocol):
    """Frozen per-seed sigma = 0.1 * the retained normalization scale."""
    return frozen_sigma(protocol)


def source_future_table(logs, windows):
    """One pass over the selected logs: actual numeric future states per query.

    Returns physical values [N,11,2] (nan where UNKNOWN), known flags [N,11,2]
    and in-scope flags [N,11,2]. Query-specific UNKNOWN and OUT_OF_SCOPE are
    preserved; nothing is filled, imputed or averaged. The offset-0 row is the
    observed prefix anchor, read only as the declared unperturbed anchor of
    the sampler and of the strict anchor comparison.
    """
    values = np.empty((len(windows), ANCHOR_STEPS, len(ORACLE_FIELDS)))
    known = np.empty((len(windows), ANCHOR_STEPS, len(ORACLE_FIELDS)), dtype=bool)
    scope = np.empty((len(windows), ANCHOR_STEPS, len(ORACLE_FIELDS)), dtype=bool)
    for i, window in enumerate(windows):
        for t, offset in enumerate(STATE_OFFSETS):
            row = logs._states[(window.flow, window.anchor_ns + offset * TICK_NS)]
            active = row['status'] == 'IN_SCOPE'
            for f, (_, field, unit, dimension, _) in enumerate(ORACLE_FIELDS):
                value, observed = _metric(row, field, unit, dimension)
                values[i, t, f] = value if active else np.nan
                known[i, t, f] = observed and active
                scope[i, t, f] = active
    return values, known, scope


def load_retained_protocol(args):
    """Retained protocol plus one-time agreement with the declared manifest."""
    manifest = json.loads(Path(args.manifest).read_text())
    protocol = json.loads((Path(args.prepared_root)/'T_seed0'/'protocol.json').read_text())
    return protocol


def load_source_logs(args, protocol):
    """Reopen the current R1 source logs named by the retained manifest."""
    root = Path(args.root)
    manifest = protocol['manifest']
    parts = []
    for selection in manifest['runs']:
        parts.append(load_r1_logs(root, (RunSelection(selection['episode_id'],
            selection['profile'], selection['run']),),
            scenario_by_episode=manifest['scenario_by_episode'],
            split_by_scenario=manifest['split_by_scenario'], r1_root=selection['r1_root']))
    merged = {name: {} for name in ('_states', '_predicates', '_flows', 'configs')}
    splits = {}
    for part in parts:
        splits.update(part.split_by_episode)
        for name, index in merged.items():
            source = getattr(part, name)
            if index.keys() & source.keys():
                raise ValueError('duplicate explicitly selected run or flow index')
            index.update(source)
    return R1TrainingLogs(split_by_episode=splits, scenario_by_episode=manifest['scenario_by_episode'],
        states=merged['_states'], predicates=merged['_predicates'], flows=merged['_flows'],
        configs=merged['configs'])


def load_retained_caches(prepared_root, seeds):
    """Open every retained T cache once; their tensors become the fixed inputs.

    The retained transported-Q layout is one directory per seed, each holding
    that seed's prepared TRAIN and VALID caches and its protocol copy.
    """
    root = Path(prepared_root)
    caches = {}
    for seed in seeds:
        for split in ('train', 'validation'):
            path = root/f'T_seed{seed}'/f'prepared_{split}__T__seed{seed}.npz'
            with np.load(path, allow_pickle=False) as archive:
                if str(archive['q_arm']) != 'T':
                    raise ValueError(f'retained cache {path.name} is not the T-arm cohort')
                caches[(seed, split)] = {key: np.array(archive[key], copy=True)
                                         for key in archive.files}
    return caches


def static_cache_tensors(cache):
    """Reuse the retained history/prior/labels/layout tensors untouched."""
    tensors = {}
    for name in ('history', 'history_known', 'prior_logits', 'targets', 'target_known', 'compatible'):
        tensors[name] = torch.as_tensor(np.array(cache[name], copy=True))
    tensors['field_type_ids'] = torch.as_tensor(np.array(cache['field_type_ids'], copy=True),
                                                dtype=torch.long)
    tensors['predicate_field_incidence'] = torch.as_tensor(
        np.array(cache['predicate_field_incidence'], copy=True), dtype=torch.bool)
    tensors['owner_field_incidence'] = torch.as_tensor(
        np.array(cache['owner_field_incidence'], copy=True), dtype=torch.bool)
    layout = TrajectoryLayout_from_cache(cache)
    return tensors, layout


def TrajectoryLayout_from_cache(cache):
    from Dataset.world_model.rule_factor_graph.predicate_trajectory_models import TrajectoryLayout
    return TrajectoryLayout(**{key: tuple(value)
                               for key, value in json.loads(str(cache['layout'])).items()})


def lp_noise_declarations(sigma):
    """Mature-datagram slice declares physical numeric fields and domains."""
    return tuple(StateFieldNoise(field, 'numeric', unit, sigma=(float(sigma[f]),),
        projection=((0., 1.),) if symbol == 'P' else (),
        source_ref='retained TRAIN normalization_scale; mature datagram R1')
        for f, (symbol, field, unit, _, _) in enumerate(ORACLE_FIELDS))


def sample_oracle_futures(future_table, windows, sigma, upper_by_run, k, seed):
    """Delegate declared Gaussian draws and physical projection to the oracle."""
    upper = np.stack([upper_by_run[(w.flow.episode_id, w.flow.profile, w.flow.run)]
                      for w in windows])
    return OracleStateSampler(lp_noise_declarations(sigma), seed_formula=oracle_seed).sample(
        future_table, k=k, seed=seed, upper_by_row=upper)


@dataclass(frozen=True)
class HistoryConfiguration:
    """Inclusive history grid ending at the observed anchor, in exact ns."""
    history_seconds: float = 3.5
    sampling_interval_seconds: float = .5

    @property
    def duration_ns(self):
        return self._nanoseconds(self.history_seconds)

    @property
    def interval_ns(self):
        return self._nanoseconds(self.sampling_interval_seconds)

    @staticmethod
    def _nanoseconds(value):
        number = Decimal(str(value)) * 1_000_000_000
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError('History settings must resolve to exact integer nanoseconds')
        return int(number)

    def __post_init__(self):
        if self.duration_ns <= 0 or self.interval_ns <= 0:
            raise ValueError('History duration and sampling interval must be positive')
        if self.duration_ns % self.interval_ns:
            raise ValueError('History duration must be an integer number of sampling intervals')
        if self.interval_ns % GRID_NS:
            raise ValueError('R1 source rows use 0.5-second sampling; requested interval must be a multiple of 0.5 seconds')

    @property
    def steps(self):
        return self.duration_ns // self.interval_ns + 1

    def times_ns(self, anchor_ns):
        return tuple(anchor_ns-self.duration_ns+i*self.interval_ns for i in range(self.steps))


def source_history_prefix(logs, window, configuration):
    """Read exact existing source rows; missing rows are coverage errors."""
    times = configuration.times_ns(window.anchor_ns)
    values, known, applicable = [], [], []
    for time in times:
        try:
            row = logs._states[(window.flow, time)]
        except KeyError as error:
            raise ValueError(f'Unsupported source history: {configuration.history_seconds}s/'
                f'{configuration.sampling_interval_seconds}s requires existing row at '
                f'time_ns={time}, anchor_ns={window.anchor_ns}, flow={window.flow}; '
                'the retained 8-point cache cannot supply earlier observations') from error
        active = row['status'] == 'IN_SCOPE'
        metrics = [_metric(row, field, unit, dimension)
                   for _, field, unit, dimension, _ in ORACLE_FIELDS]
        values.append([value if active else np.nan for value, _ in metrics])
        known.append([observed and active for _, observed in metrics])
        applicable.append([active]*len(ORACLE_FIELDS))
    authority = logs._flows[window.flow]
    episode = EpisodeContract(window.flow.episode_id,
        f'{window.flow.profile}/run{window.flow.run}/{window.flow.flow_id}',
        logs.split_by_episode[window.flow.episode_id], tuple(time//TICK_NS for time in times),
        window.anchor_ns//TICK_NS, authority['fields'], authority['queries'])
    return PredictionPrefix(window, episode, np.asarray(values), np.asarray(known, bool),
                            np.asarray(applicable, bool), authority['state_ref'])


def lp_contract_records(logs, windows, *, history_seconds=3.5, sampling_interval_seconds=.5):
    """Compile fixed plans once per exact flow; retain each window's clock."""
    configuration = HistoryConfiguration(history_seconds, sampling_interval_seconds)
    prefixes = tuple(source_history_prefix(logs, window, configuration) for window in windows)
    evaluators = {}
    for window, prefix in zip(windows, prefixes):
        if window.flow not in evaluators:
            evaluators[window.flow] = OracleRuleEvaluator(prefix.episode)
    codes = np.asarray([[[TRUTH_CODES[logs._predicates[
        (window.flow, window.anchor_ns + offset*TICK_NS, symbol)]['value']]
        for offset in STATE_OFFSETS] for symbol, *_ in RULES] for window in windows],
        dtype=np.int8)
    supervision = compute_first_flip_supervision(codes)
    return prefixes, evaluators, supervision


def prepare_split(windows, cache, future_table, upper_by_run, *, seed, k, device,
                  split, sigma, mean, scale, contracts):
    """Physical CandidateTrajectory assembly, fixed rules, then TRAIN transform.

    Actual source history uses the configured inclusive grid; the retained 8I
    prior/layout and schedule remain fixed.
    Source predicate labels use the generic eleven-point censoring function;
    candidate truth uses compiled rules on physical operands before scaling.
    """
    tensors, layout = static_cache_tensors(cache)
    sampled, sampled_known, counts = sample_oracle_futures(future_table, windows, sigma,
                                                           upper_by_run, k, seed)
    prefixes, fixed_evaluators, supervision = contracts
    records, evaluators = [], []
    for query, (window, prefix) in enumerate(zip(windows, prefixes)):
        record = build_candidate_record(prefix.episode, prefix.history, prefix.history_known,
                                        sampled[query], sampled_known[query])
        scope = np.broadcast_to(future_table[2][query],
                                (k,) + future_table[2][query].shape)
        records.append(record)
        evaluators.append(fixed_evaluators[window.flow].scoped(scope, episode=prefix.episode))
    batch = assemble_trajectory_batch(records, evaluators=evaluators)
    arrays = batch.as_mapping()
    transforms = tuple(FieldTransform(field, float(mean[f]), float(scale[f]))
                       for f, (_, field, _, _, _) in enumerate(ORACLE_FIELDS))
    tensors['history'] = torch.as_tensor(apply_field_transforms(
        arrays['history'], arrays['history_known'], transforms), dtype=torch.float64, device=device)
    tensors['history_known'] = torch.as_tensor(np.array(arrays['history_known'], copy=True), device=device)
    normalized = apply_field_transforms(arrays['sampled_states'], arrays['state_known'], transforms)
    tensors['sampled_states'] = torch.as_tensor(np.ascontiguousarray(normalized),
                                                dtype=torch.float64, device=device)
    for name in ('state_known', 'sampled_truth', 'truth_known', 'applicable'):
        tensors[name] = torch.as_tensor(np.array(arrays[name], copy=True), device=device)
    targets, target_known, compatible = supervision
    for name, value in (('targets', targets), ('target_known', target_known),
                        ('compatible', compatible)):
        tensors[name] = torch.as_tensor(np.array(value, copy=True), device=device)
    tensors = {name: value.to(device) for name, value in tensors.items()}
    return PreparedTrajectorySplit(tensors, layout, split, batch), counts


def per_seed_metrics(summary):
    """Readable per-seed record: common-mask headline metrics beside the full report."""
    readable = {'label': ORACLE_LABEL,
                'oracle_counts': {split: summary[split]['oracle_counts']
                                  for split in ('train', 'validation')},
                'sigma_physical_units': summary['train']['sigma_physical_units'],
                'noise_variance_physical_units': summary['train']['noise_variance_physical_units'],
                'common_masks': {}}
    for split in ('train', 'validation'):
        record = summary[split]
        readable['common_masks'][split] = {
            model: {predicate: {key: cell[key] for key in
                                ('known_cdf_cells', 'known_F10', 'positive_F10', 'F10_brier',
                                 'cdf_brier', 'f1', 'precision', 'recall', 'known_exact_flips',
                                 'missed_known_exact_flips', 'timing_mae_all_known_flips_s')}
                    for predicate, cell in (('L', record[model]['L']), ('P', record[model]['P']))}
            for model in ('ours', 'linear')}
    return {'readable': readable, 'full': summary}


def fit_seed(args, out, protocol, seed, arms_data):
    """Fit the two oracle arms for one seed; common torch init and cached schedule.

    K1 and K8 share the identical initialization and the identical retained
    batch schedule; only the prepared sample count differs. Final state, the
    fresh hybrid loss curve, reusable oracle arrays and common-mask metrics
    are saved per arm under the declared names Ours_K1/Ours_K8.
    """
    summary = {}
    for k in ORACLE_ARMS:
        name = f'Ours_K{k}'
        data = arms_data[k]
        train = data['train'][0]
        device = train.tensors['history'].device
        dest = out/f'{name}__seed{seed}'
        dest.mkdir()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SharedPredicateTrajectoryModel(train.layout).to(device=device, dtype=torch.float64)
        initialize_outcome_skip(model, args.outcome_skip_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        losses, evaluations = [], []
        with (dest/'train_evaluations.jsonl').open('w') as checkpoints:
            def evaluate(step):
                _, hybrid, *_ = evaluate_full(model, train)
                row = {'arm': name, 'K': k, 'seed': seed, 'step': step,
                       'full_TRAIN_hybrid': hybrid, 'label': 'ORACLE/NOISY-FUTURE INFORMATION DIAGNOSTIC'}
                evaluations.append(row)
                checkpoints.write(json.dumps(row) + '\n')
                checkpoints.flush()
                print(json.dumps(row), flush=True)
            evaluate(0)
            for step, indices in enumerate(protocol['schedule'], 1):
                loss = train_step(model, optimizer,
                                  train.batch(torch.as_tensor(indices, device=device)), mode='hybrid')
                losses.append(float(loss))
                if step == 1:
                    gradient_l2 = float(torch.sqrt(sum(
                        p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
                    print(json.dumps({'event': 'optimizer_update', 'arm': name, 'seed': seed,
                                      'step': 1, 'minibatch_hybrid': float(loss),
                                      'gradient_l2': gradient_l2, 'device': str(device)}), flush=True)
                if step % 200 == 0 or step == args.steps:
                    evaluate(step)
        torch.save({'step': args.steps, 'K': k, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'losses': losses,
                    'evaluations': evaluations, 'seed': seed, 'protocol': protocol,
                    'oracle': ORACLE_LAW, 'label': ORACLE_LABEL}, dest/'final_state.pt')
        summary[name] = {}
        arrays = {'losses': np.array(losses)}
        for split in ('train', 'validation'):
            prepared, counts = data[split]
            cache = data['cache_' + split]
            logs_pmf, hybrid, *_ = evaluate_full(model, prepared)
            pmf = np.exp(logs_pmf)
            targets = prepared.tensors['targets'].cpu().numpy()
            known = prepared.tensors['target_known'].cpu().numpy()
            compatible = prepared.tensors['compatible'].cpu().numpy()
            linear_pmf = np.eye(11)[cache['linear_bins']]
            linear_mask = known & cache['linear_available'][:, :, None]
            summary[f'Ours_K{k}'][split] = {
                'ours': metrics(pmf, targets, known, compatible, log_pmf=logs_pmf),
                'linear': metrics(linear_pmf, targets, linear_mask, compatible),
                'hybrid': hybrid,
                'linear_unavailable_predicates': int((~cache['linear_available']).sum()),
                'oracle_counts': counts,
                'sigma_physical_units': data['sigma'].tolist(),
                'noise_variance_physical_units': (data['sigma'] ** 2).tolist()}
            arrays.update({f'{split}_pmf': pmf, f'{split}_log_pmf': logs_pmf,
                           f'{split}_targets': targets, f'{split}_known': known,
                           f'{split}_compatible': compatible, f'{split}_linear_pmf': linear_pmf,
                           f'{split}_linear_available': cache['linear_available'],
                           f'{split}_sampled_states': prepared.tensors['sampled_states'].cpu().numpy(),
                           f'{split}_state_known': prepared.tensors['state_known'].cpu().numpy(),
                           f'{split}_truth_known': prepared.tensors['truth_known'].cpu().numpy(),
                           f'{split}_applicable': prepared.tensors['applicable'].cpu().numpy(),
                           f'{split}_window_keys': cache['window_keys']})
        np.savez_compressed(dest/'final_predictions.npz', **arrays)
        (dest/'metrics.json').write_text(json.dumps(per_seed_metrics(summary[name]),
                                                    indent=2, allow_nan=False))
    return summary


def run_diagnostic(args):
    """Prepare both oracle arms per seed from retained caches and fit them."""
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU execution required for the oracle diagnostic')
    protocol = load_retained_protocol(args)
    seeds = list(args.seeds)
    out = Path(args.output)
    caches = load_retained_caches(args.prepared_root, seeds)
    logs = load_source_logs(args, protocol)
    # Exact persisted keys construct the cohort; no independent re-selection,
    # mission-catalog reconstruction or cache/provenance verification scans.
    windows = {}
    for split in ('train', 'validation'):
        rows = (json.loads(str(value)) for value in caches[(seeds[0], split)]['window_keys'])
        windows[split] = tuple(WindowKey(FlowKey(**row['flow']), row['anchor_ns']) for row in rows)
    sigma = frozen_sigma(protocol)
    mean = np.asarray(protocol['normalization_mean'], dtype=float)
    scale = retained_normalization_scale(protocol)
    protocol = {key: value for key, value in protocol.items()
                if key not in ORACLE_DROPPED_PROTOCOL_KEYS}
    history_config = HistoryConfiguration(args.history_seconds, args.sampling_interval_seconds)
    protocol.update(history_seconds=args.history_seconds,
                    sampling_interval_seconds=args.sampling_interval_seconds,
                    history_steps=history_config.steps, steps=args.steps, batch_size=args.batch_size, lr=args.lr, device=args.device,
                    seeds=seeds, arms=[f'Ours_K{k}' for k in ORACLE_ARMS],
                    oracle=ORACLE_LAW, oracle_label=ORACLE_LABEL, oracle_version=ORACLE_VERSION,
                    oracle_seed_formula='seed*1000003+query',
                    sigma_physical_units=sigma.tolist(),
                    noise_variance_physical_units=(sigma ** 2).tolist(),
                    sigma_fraction_of_retained_scale=SIGMA_FRACTION,
                    retained_normalization_scale=retained_normalization_scale(protocol).tolist(),
                    thresholds_by_symbol={symbol: float(threshold)
                                          for symbol, _, _, _, threshold in ORACLE_FIELDS},
                    declared_field_order=[{'symbol': symbol, 'field': field, 'unit': unit,
                                           'dimension': dimension, 'threshold': threshold}
                                          for symbol, field, unit, dimension, threshold in ORACLE_FIELDS],
                    l_p_slice_and_backbone=L_P_SLICE_NOTE,
                    projection=('latency_ms to [0,(query packet_ttl_ns-1)/1e6]; packet_loss_ratio to '
                                '[0,1]; exact endpoints; no epsilon'),
                    neural_input_transform='(physical_state-TRAIN_mean)/TRAIN_scale for known cells; unknown tensor0 masked; rules use physical operands',
                    outcome_skip_init=args.outcome_skip_init,
                    oracle_model_capacity=oracle_model_capacity(),
                    oracle_window_counts={split: len(windows[split]) for split in windows},
                    event_blind_split=True, K_by_arm={f'Ours_K{k}': k for k in ORACLE_ARMS})
    upper_by_run = {key: np.array([(config['network']['packet_ttl_ns'] - 1) / 1_000_000, 1.])
                    for key, config in logs.configs.items()}
    future_table = source_future_table(logs, windows['train'] + windows['validation'])
    n_train = len(windows['train'])
    split_tables = {'train': tuple(item[:n_train] for item in future_table),
                    'validation': tuple(item[n_train:] for item in future_table)}
    contracts = {split: lp_contract_records(logs, windows[split],
        history_seconds=args.history_seconds, sampling_interval_seconds=args.sampling_interval_seconds)
        for split in windows}
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    device = torch.device(args.device)
    for seed in seeds:
        schedule = caches[(seed, 'train')]['schedule']
        if schedule.shape != (args.steps, args.batch_size):
            raise ValueError('requested update/batch budget differs from the retained exact schedule')
        fit_protocol = dict(protocol, schedule=np.array(schedule), seed=seed)
        arms_data = {}
        for k in ORACLE_ARMS:
            data = {'sigma': sigma}
            for split in ('train', 'validation'):
                prepared, counts = prepare_split(windows[split], caches[(seed, split)],
                    split_tables[split], upper_by_run, seed=seed, k=k, device=device,
                    split=split, sigma=sigma, mean=mean, scale=scale, contracts=contracts[split])
                data[split] = (prepared, counts)
                data['cache_' + split] = caches[(seed, split)]
            arms_data[k] = data
        fit_seed(args, out, fit_protocol, seed, arms_data)
    return protocol


def oracle_model_capacity():
    """Setup-time capacity of the shared dense model, no source reads."""
    from Dataset.world_model.rule_factor_graph.predicate_trajectory_models import TrajectoryLayout
    keys = tuple((f'field_{index}', 'numeric', 'unit', 'oracle', (), (('uav',),))
                 for index in range(len(ORACLE_FIELDS)))
    layout = TrajectoryLayout(keys, tuple(range(len(keys))),
                              tuple(tuple(True for _ in keys) for _ in keys))
    with torch.random.fork_rng(devices=[]):
        model = SharedPredicateTrajectoryModel(layout)
    total = sum(parameter.numel() for parameter in model.parameters())
    head = model.trajectory_head
    return {'hidden_dim': head.history_encoder.hidden_size,
            'field_dim': model.field_embedding.embedding_dim,
            'instantiated_parameters': total,
            'shared_by_K': 'same parameters for K1 and K8; equal logmeanexp across the actual K',
            'reliance_gate_parameters': 0, 'prefix_gate': 'absent', 'auxiliary_loss': 'absent'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/mnt/data2/weizhiwei/AERO_WORLD')
    parser.add_argument('--manifest', default='design/p01_ns3/transported_q/manifest.json')
    parser.add_argument('--prepared-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--history-seconds', type=float, default=3.5)
    parser.add_argument('--sampling-interval-seconds', type=float, default=.5)
    parser.add_argument('--outcome-skip-init', type=float, default=4.0)
    run_diagnostic(parser.parse_args())


if __name__ == '__main__':
    main()
