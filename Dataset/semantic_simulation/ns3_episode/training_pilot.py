"""Fresh scenario-held-out, mission-only mature-datagram trajectory pilot.

Q is an empirical TRAIN-only conditional law over whole joint future operands.
Nearest-prefix selection never reads query future operands or labels. Sampling
on draws eight nearby donor trajectories; off never reads sampled future-state
or future-rule inputs and uses fixed zero branch hidden/outcome. Donor values
remain original, with exact query anchor replacement. Arm T transports the same
selected donor futures anchor-relative: query_reference + (donor_future -
donor_reference) per field, retaining the donor field/time draws and UNKNOWN
masks; latency is projected to [0, (query packet_ttl_ns-1)/1e6] ms and loss ratio is
projected to [0, 1] as a declared part of the law, with pre-projection violations,
fractions and endpoint mass recorded. Arm U is the transported uniform-donor placebo. References are the
last known prefix operand of each side (anchor when known) and never read the
query future or labels.
All arms share dense generic topology, linear8I prior, seeds and update schedule.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch

from .training_adapter import (load_r1_logs, RunSelection, CandidateOperands,
    CandidateLawDeclaration, RULES, STATE_OFFSETS, TICK_NS, _metric, R1TrainingLogs)
from Dataset.world_model.rule_factor_graph.predicate_trajectory_models import SharedPredicateTrajectoryModel, TrajectoryLayout
from Dataset.world_model.rule_factor_graph.predicate_trajectory_data import (
    CandidateTrajectory, FuturePredicateEvaluator, assemble_trajectory_batch, TRUTH_CODES)
from Dataset.world_model.rule_factor_graph.predicate_trajectory_training import (
    prepare_assembled_batch, PreparedTrajectorySplit, predict_batch, train_step, cumulative_loss,
    FORWARD_KEYS)

PILOT_VERSION = 'p01.ns3.mission-empirical-pilot/v1'
COMPONENT_SUPERVISION_FORMULA = ('H(mixture) + w*(H(history)+H(donor))/2; '
                                 'H = hybrid knownCDF-square1..9 + .5 knownBCE10, same known mask')

# Authorized Q arms. A is the existing absolute nearest64 law; T transports the
# same selected donor indices; U draws uniform eligible TRAIN donors (query
# scenario excluded) and transports them; Off is the existing absent branch.
Q_ARMS = {'A': ('sampling_on_A', True, 'absolute'), 'T': ('sampling_on_T', True, 'transport'),
          'U': ('sampling_on_U', True, 'uniform'), 'Off': ('sampling_off', False, 'absolute')}
# Declared physical bounds of transported operands, per RULES field order.
TRANSPORT_KEYS = ('instances','reference_pairs','applied','moved','no_reference',
                  'no_reference_donors','both_missing_reference','cells','below','above','endpoint_low','endpoint_high')
TRANSPORT_FIELD_KEYS = ('applied','moved','no_reference','both_missing_reference','cells','below','above','endpoint_low','endpoint_high')
GOVERNED_THRESHOLDS = np.array([120., .05])


def mission_ids(root, episode, scenario):
    path = root/'aw_data/render_ready_episodes'/episode/'scenario_plan.json'
    plan = json.loads(path.read_text())
    if (plan['episode_id'], plan['scenario_id']) != (episode, scenario):
        raise ValueError('authored scenario/episode authority differs')
    ids = frozenset(row['entity_id'] for row in plan['global_entity_roster']
                    if row.get('role') == 'mission_uav')
    if not ids:
        raise ValueError('authored mission_uav role has no entities')
    return ids


class EmpiricalJointLaw:
    """Setup-time donor bank; joint temporal/field coupling is never shuffled."""
    def __init__(self, logs, train_windows, *, neighbours=64):
        self.logs, self.windows, self.neighbours = logs, tuple(train_windows), neighbours
        if not self.windows:
            raise ValueError('no mission TRAIN trajectories')
        self._prefix_cache = {w:logs.prediction_prefix(w) for w in self.windows}
        self._evaluators = {}
        self.prefixes = tuple(self._prefix_cache[w] for w in self.windows)
        h = np.stack([p.history for p in self.prefixes])
        m = np.stack([p.history_known for p in self.prefixes])
        counts = m.sum((0, 1))
        if (counts == 0).any():
            raise ValueError('TRAIN empirical law lacks observed data for a declared field')
        self.mean = np.where(m, h, 0).sum((0, 1))/counts
        variance = np.where(m, (h-self.mean)**2, 0).sum((0, 1))/counts
        # Numerical scale floor in physical units, declared before VALID.
        self.scale = np.maximum(np.sqrt(variance), np.array([1., .01]))
        self.history = np.where(m, (h-self.mean)/self.scale, 0)
        self.known = m
        self.scenarios = np.array([logs.scenario_by_episode[w.flow.episode_id] for w in self.windows])
        futures, masks, scopes = [], [], []
        for w in self.windows:
            rows = [logs._states[(w.flow,w.anchor_ns+offset*TICK_NS)] for offset in STATE_OFFSETS]
            values, known, scope = [], [], []
            for row in rows:
                active = row['status'] == 'IN_SCOPE'
                metrics = [_metric(row,f,u,d) for _,_,f,u,d,_,_ in RULES]
                values.append([value if active else np.nan for value,_ in metrics])
                known.append([flag and active for _,flag in metrics])
                scope.append([active]*2)
            futures.append(values);masks.append(known);scopes.append(scope)
        self.values, self.masks, self.scopes = np.array(futures), np.array(masks), np.array(scopes)
        # Last known TRAIN prefix operand per field: the donor transport
        # reference (the anchor where that field is known). Computed once.
        bank = np.stack([p.history for p in self.prefixes])
        last = (bank.shape[1]-1) - self.known[:, ::-1].argmax(1)
        self.donor_reference = np.squeeze(np.take_along_axis(bank, last[:, None, :], axis=1), 1)
        # A window with no known sample of a field lacks a reference entirely.
        self.donor_reference_known = self.known.any(1)
        self.upper_by_run={key:np.array([(config['network']['packet_ttl_ns']-1)/1_000_000,1.])
                           for key,config in logs.configs.items()}
        upper=np.stack([self.upper_by_run[(w.flow.episode_id,w.flow.profile,w.flow.run)] for w in self.windows])
        self.donor_below=((self.values[:,1:]<0)&self.masks[:,1:]).sum((0,1))
        self.donor_above=((self.values[:,1:]>upper[:,None])&self.masks[:,1:]).sum((0,1))
        self.declaration = logs.declare_candidate_law(CandidateLawDeclaration(
            'TRAIN-prefix-nearest-whole-joint-trajectories/v1',
            tuple(sorted({w.flow.episode_id for w in self.windows})), PILOT_VERSION))

    def prefix(self, window):
        if window not in self._prefix_cache:
            self._prefix_cache[window]=self.logs.prediction_prefix(window)
        return self._prefix_cache[window]

    def assemble(self, prefix, candidates):
        # Exact flow field/owner/rule plans are compiled once; anchors reuse them.
        key=prefix.key.flow
        if key not in self._evaluators:
            self._evaluators[key]=FuturePredicateEvaluator(prefix.episode)
        evaluator=ReboundScopedEvaluator(prefix.episode,self._evaluators[key],candidates.applicable)
        record=CandidateTrajectory(prefix.episode,prefix.history,prefix.history_known,candidates.values,candidates.known)
        return assemble_trajectory_batch((record,),evaluators=(evaluator,))

    def candidates(self, prefix, *, seed, k=8, selection_mode='nearest', anchor_replacement=True):
        """Sample K donors, preserving the joint field/time draws.

        selection_mode 'nearest' is the existing prefix-only nearest64 law;
        'uniform' draws from all eligible TRAIN windows with the same RNG so
        every per-window draw stream matches the A arm for a given seed.
        anchor_replacement keeps the existing exact query anchor substitution;
        transport arms pass False and substitute after transforming operands.
        """
        scenario = self.logs.scenario_by_episode[prefix.key.flow.episode_id]
        eligible = np.flatnonzero(self.scenarios != scenario)
        if len(eligible) < k:
            raise ValueError(f'cross-scenario donor insufficiency: {scenario}: {len(eligible)} < {k}')
        generator = np.random.default_rng(seed)
        if selection_mode == 'uniform':
            selected = generator.choice(eligible,k,replace=False)
        elif selection_mode == 'nearest':
            query = np.where(prefix.history_known,(prefix.history-self.mean)/self.scale,0)
            overlap = self.known[eligible] & prefix.history_known
            n = overlap.sum((1,2))
            # No common observation cannot support prefix-nearest selection.
            distance = np.where(overlap,(self.history[eligible]-query)**2,0).sum((1,2))/np.maximum(n,1)
            distance += (self.known[eligible] != prefix.history_known).mean((1,2))
            distance[n == 0] = np.inf
            order = eligible[np.argsort(distance,kind='stable')]
            order = order[np.isfinite(distance[np.argsort(distance,kind='stable')])]
            if len(order) < k:
                raise ValueError('insufficient cross-scenario donors with shared prefix observations')
            nearby = order[:self.neighbours]
            selected = generator.choice(nearby,k,replace=False)
        else:
            raise ValueError(f'undeclared donor selection mode: {selection_mode}')
        values=self.values[selected].copy();known=self.masks[selected].copy();scope=self.scopes[selected].copy()
        if anchor_replacement:
            values[:,0]=prefix.history[-1];known[:,0]=prefix.history_known[-1];scope[:,0]=prefix.history_applicable[-1]
        return CandidateOperands(values,known,scope,PILOT_VERSION+':TRAIN-donor-bank','predicted_future_operands'), selected


def linear_prior(prefix):
    """Last-two observed slope, strict governed thresholds, fixed linear8I.

    Missing last-two operands contribute neutral logits and unavailable reference
    masks; they are not filled or reported as observed negative predictions.
    """
    thresholds=GOVERNED_THRESHOLDS;available=prefix.history_known[-2:].all(0)
    anchor=prefix.history[-1];slope=anchor-prefix.history[-2]
    projected=anchor[None]+np.arange(1,11)[:,None]*slope[None]
    opposite=(projected>thresholds)!=(anchor>thresholds)[None]
    bins=np.where(opposite.any(0),opposite.argmax(0),10)
    logits=np.zeros((2,11));logits[np.flatnonzero(available),bins[available]]=8.
    return logits,available,bins


def empty_transport_counts(values,known):
    counts={key:0 for key in TRANSPORT_KEYS}
    counts.update(instances=len(values),reference_pairs=len(values)*values.shape[-1],cells=int(known[:,1:].sum()))
    counts.update({key+'_by_field':[0]*values.shape[-1] for key in TRANSPORT_FIELD_KEYS})
    counts['cells_by_field']=known[:,1:].sum((0,1)).astype(int).tolist()
    return counts


def transport_operands(law,prefix,selected,values,known):
    """Joint donor deltas from last-known PREFIX; absent references stay absolute.

    Future-only physical projection uses query TTL's strict integer-ns bound,
    and [0,1] loss bounds, with no epsilon. Masks are unchanged; query anchor
    is restored exactly. No query future truth enters this operation.
    """
    positions=(prefix.history.shape[0]-1)-prefix.history_known[::-1].argmax(0)
    query_reference=np.take_along_axis(prefix.history,positions[None],axis=0)[0]
    query_known=prefix.history_known.any(0)
    donor_reference=law.donor_reference[selected];donor_known=law.donor_reference_known[selected]
    both=donor_known&query_known
    with np.errstate(invalid='ignore'):
        shifted=values[:,1:]-donor_reference[:,None,:]+query_reference
    future_known=known[:,1:];applied=future_known&both[:,None,:]
    future=np.where(applied,shifted,values[:,1:])
    key=prefix.key.flow;upper=law.upper_by_run[(key.episode_id,key.profile,key.run)]
    below=(future<0)&future_known;above=(future>upper)&future_known
    projected=np.where(future_known,np.clip(future,0,upper),future)
    moved=(projected!=values[:,1:])&future_known
    low=(projected==0)&future_known;high=(projected==upper)&future_known
    result=values.copy();result[:,1:]=projected;result[:,0]=prefix.history[-1]
    result_known=known.copy();result_known[:,0]=prefix.history_known[-1]
    absent=~both;both_missing=(~donor_known)&(~query_known)
    counts=empty_transport_counts(values,known)
    counts.update(applied=int(applied.sum()),moved=int(moved.sum()),no_reference=int(absent.sum()),
        no_reference_donors=int(absent.any(-1).sum()),both_missing_reference=int(both_missing.sum()),
        below=int(below.sum()),above=int(above.sum()),endpoint_low=int(low.sum()),endpoint_high=int(high.sum()))
    matrices={'applied':applied,'moved':moved,'no_reference':absent,'both_missing_reference':both_missing,
              'cells':future_known,'below':below,'above':above,'endpoint_low':low,'endpoint_high':high}
    counts.update({key+'_by_field':value.sum(tuple(range(value.ndim-1))).astype(int).tolist()
                   for key,value in matrices.items()})
    return result,result_known,counts


def q_transport_summary(law,transport,*,arm):
    sums={key:sum(item[key] for item in transport.values()) for key in TRANSPORT_KEYS}
    sums.update({key+'_by_field':np.sum([item[key+'_by_field'] for item in transport.values()],axis=0).astype(int).tolist()
                 for key in TRANSPORT_FIELD_KEYS})
    known=sums['cells'];pairs=sums['reference_pairs'];instances=sums['instances']
    return {'arm':arm,'windows':len(transport),'donor_instances':instances,'counts':sums,
        'applied_fraction_of_known':sums['applied']/known if known else None,
        'projection_fraction_of_known':(sums['below']+sums['above'])/known if known else None,
        'endpoint_low_fraction_of_known':sums['endpoint_low']/known if known else None,
        'endpoint_high_fraction_of_known':sums['endpoint_high']/known if known else None,
        'no_reference_fraction_of_pairs':sums['no_reference']/pairs if pairs else None,
        'no_reference_fraction_of_donors':sums['no_reference_donors']/instances if instances else None,
        'future_only_counts':True,
        'query_bounds_by_run':{'/'.join(map(str,key)):{'latency_ms':[0.,float(upper[0])],'loss_ratio':[0.,1.]}
                               for key,upper in law.upper_by_run.items()},
        'donor_bank':{'train_windows':len(law.windows),'no_reference_windows_by_field':(~law.donor_reference_known).sum(0).astype(int).tolist(),
            'known_prefix_operands_by_field':law.known.sum((0,1)).astype(int).tolist(),
            'below_bounds':law.donor_below.tolist(),'above_bounds':law.donor_above.tolist()}}


def candidate_first_flip_pmf(codes):
    """Unnormalized K-mean fixed-rule event mass; UNKNOWN/OOS stays unresolved."""
    k,steps,fields=codes.shape;mass=np.zeros((k,fields,11))
    observed=np.isin(codes,(TRUTH_CODES['true'],TRUTH_CODES['false']))
    for f in range(fields):
        for d in range(k):
            if not observed[d,0,f]:continue
            for t in range(1,steps):
                if not observed[d,t,f]:break
                if codes[d,t,f]!=codes[d,0,f]:
                    mass[d,f,t-1]=1;break
            else:mass[d,f,10]=1
    return mass.mean(0),(~observed).sum((0,1))


def q_diagnostic_summary(law,windows,donors,diagnostic,*,arm):
    mismatch=0;eligible=0;pair_counts=[];scores=np.zeros(2);score_counts=np.zeros(2,int)
    pmfs=np.stack([item[0] for item in diagnostic]);missing=np.stack([item[1] for item in diagnostic])
    for i,window in enumerate(windows):
        prefix=law.prefix(window);anchor=prefix.history[-1];donor_anchor=law.values[donors[i],0]
        side=prefix.history_known[-1]&law.masks[donors[i],0]
        different=side&((anchor>GOVERNED_THRESHOLDS)!=(donor_anchor>GOVERNED_THRESHOLDS))
        pair_counts.append(different.sum(0));mismatch+=int(different.sum());eligible+=int(side.sum())
        labels=law.logs.supervision(window);complete=pmfs[i].sum(-1)==1
        mask=labels.known&complete[:,None];cdf=np.cumsum(pmfs[i],-1)[:,:10]
        scores+=np.where(mask,(cdf-labels.targets)**2,0).sum(-1);score_counts+=mask.sum(-1)
    summary={'arm':arm,'queries':len(windows),'role':'offline Q-only analysis; never a training gate',
        'pmf_definition':'actual arm transformed fixed-rule traces, K-mean; unresolved mass not renormalized',
        'raw_donor_anchor_side_mismatch':mismatch,'anchor_mismatch_eligible_pairs':eligible,
        'anchor_side_mismatch_rate':mismatch/eligible if eligible else None,
        'missing_rule_cells_by_field':missing.sum(0).astype(int).tolist(),
        'unresolved_mass_by_window_field':(1-pmfs.sum(-1)).tolist(),
        'cdf_brier_fully_resolved_by_field':[float(scores[f]/score_counts[f]) if score_counts[f] else None for f in range(2)],
        'cdf_scored_cells_by_field':score_counts.tolist()}
    return summary,{'q_pmf':pmfs,'q_missing_cells':missing,'q_unresolved_mass':1-pmfs.sum(-1),
                    'raw_donor_anchor_side_mismatch_by_field':np.array(pair_counts)}


class ReboundScopedEvaluator(FuturePredicateEvaluator):
    """Reuse the same immutable exact-flow rule plans at different anchors."""
    def __init__(self, episode, template, applicable):
        self.episode=episode;self.plans=template.plans
        self.predicate_field_incidence=template.predicate_field_incidence
        self.applicable=applicable

    def evaluate(self, record):
        codes=super().evaluate(record)
        return np.where(self.applicable,codes,TRUTH_CODES['out_of_scope']).astype(np.int8)


class ControlledGenericTrajectoryModel(SharedPredicateTrajectoryModel):
    """Shared donor/history predictions with optional prefix-only probability reliance."""
    def __init__(self,layout,*,sampling_enabled,reliance_mode='none'):
        super().__init__(layout)
        self.sampling_enabled=sampling_enabled
        self.reliance_mode=reliance_mode if sampling_enabled else 'none'
        if self.reliance_mode!='none':
            self.reliance_gate=torch.nn.Linear(self.trajectory_head.history_encoder.hidden_size,1)
            torch.nn.init.zeros_(self.reliance_gate.weight)
            torch.nn.init.zeros_(self.reliance_gate.bias)

    def _history_prediction(self,history,kwargs):
        # Same absent-branch representation/head; no sampled operands read.
        incidence=kwargs['predicate_field_incidence'].to(history.dtype)
        if incidence.ndim==2:incidence=incidence[None].expand(history.shape[0],-1,-1)
        pool=incidence/incidence.sum(-1,keepdim=True)
        features=self._fields(history,kwargs['history_known'],kwargs.get('owner_field_incidence'),kwargs.get('field_type_ids'))
        prefix=torch.einsum('bpf,bhfd->bphd',pool,features)
        b,p,h,d=prefix.shape;head=self.trajectory_head
        _,hidden=head.history_encoder(prefix.reshape(b*p,h,d))
        prefix_hidden=hidden[-1].reshape(b,p,-1)
        hidden=prefix_hidden[:,:,None].expand(-1,-1,8,-1)
        context=head.context_encoder(torch.cat((hidden,torch.zeros_like(hidden)),-1))
        outcome=context.new_zeros((*context.shape[:-1],11))
        logits=head.residual(torch.cat((context,outcome),-1))
        if kwargs.get('prior_logits') is not None:logits=logits+kwargs['prior_logits'][:,:,None]
        return torch.log_softmax(logits,-1),prefix_hidden

    def _reliance_logits(self,prefix_hidden):
        if self.reliance_mode=='constant':
            return prefix_hidden.new_zeros((*prefix_hidden.shape[:-1],1))
        return self.reliance_gate(prefix_hidden)

    def reliance_alpha(self,history,**kwargs):
        _,hidden=self._history_prediction(history,kwargs)
        return torch.sigmoid(self._reliance_logits(hidden)).squeeze(-1)

    def forward(self,history,**kwargs):
        if self.sampling_enabled and self.reliance_mode=='none':
            return super().forward(history,**kwargs)
        history_logs,hidden=self._history_prediction(history,kwargs)
        if self.sampling_enabled:
            branch_kwargs=dict(kwargs,return_branch_logs=True)
            donor_logs=super().forward(history,**branch_kwargs)
            gate=self._reliance_logits(hidden)[:,:,None]
            logs=torch.logaddexp(torch.nn.functional.logsigmoid(gate)+donor_logs,
                                torch.nn.functional.logsigmoid(-gate)+history_logs)
        else:
            logs=history_logs;donor_logs=None
        if kwargs.get('return_branch_logs',False):
            return logs,donor_logs,history_logs
        return torch.logsumexp(logs,2)-np.log(8)



def prepare_cohort(logs, windows, law, *, seed, device, arm='A'):
    """Fixed numerical cache for one Q arm; donor arrays stay per-arm data.

    Arms T and U transport their operands; arm A keeps donor values absolute.
    Selection for U is uniform over eligible TRAIN windows; selection for A and
    T is the existing nearest64 law. The transport summary and the Q-only
    diagnostic of every prepared window are returned with the cohort.
    """
    if arm not in Q_ARMS:raise ValueError(f'undeclared Q arm: {arm}')
    mode, sampling, transform = Q_ARMS[arm]
    if not sampling:raise ValueError('arm Off has no prepared operand cohort')
    prepared=[];donors=[];linear=[];linear_known=[];transport={};diagnostic=[]
    for i,window in enumerate(windows):
        prefix=law.prefix(window)
        selection='uniform' if transform=='uniform' else 'nearest'
        candidates, selected=law.candidates(prefix,seed=seed*1_000_003+i,selection_mode=selection,
            anchor_replacement=arm not in ('T','U'))
        if arm in ('T','U'):
            # Transform the original donor operands, then apply the exact query
            # anchor replacement, so the observed anchor stays query-derived.
            values,known,counts=transport_operands(law,prefix,selected,candidates.values,candidates.known)
            values[:,0]=prefix.history[-1];known[:,0]=prefix.history_known[-1]
            scope=candidates.applicable.copy();scope[:,0]=prefix.history_applicable[-1]
            candidates=CandidateOperands(values,known,scope,
                candidates.source_ref+':anchor-relative/v1','predicted_future_operands')
        else:counts=empty_transport_counts(candidates.values,candidates.known)
        transport[i]=counts
        assembled=law.assemble(prefix,candidates)
        diagnostic.append(candidate_first_flip_pmf(assembled.as_mapping()['truth_codes'][0]))
        labels=logs.supervision(window);prior,available,bins=linear_prior(prefix)
        item=prepare_assembled_batch(assembled,split=prefix.episode.split,device=device,
            targets=labels.targets[None],target_known=labels.known[None],compatible=labels.compatible[None],
            prior_logits=prior[None])
        for name,mask_name in (('history','history_known'),('sampled_states','state_known')):
            value=item.tensors[name];mask=item.tensors[mask_name]
            mean=torch.as_tensor(law.mean,device=device,dtype=value.dtype)
            scale=torch.as_tensor(law.scale,device=device,dtype=value.dtype)
            item.tensors[name]=torch.where(mask,(value-mean)/scale,torch.zeros_like(value))
        prepared.append(item);donors.append(selected);linear.append(bins);linear_known.append(available)
    if not prepared:
        raise ValueError('fixed cohort contains no eligible mission windows')
    first=prepared[0];tensors={}
    for key in first.tensors:
        if key in ('field_type_ids','predicate_field_incidence'):
            tensors[key]=first.tensors[key]
        else:
            tensors[key]=torch.cat([p.tensors[key] for p in prepared],0)
    return (PreparedTrajectorySplit(tensors,first.layout,first.split,None),np.array(donors),
        np.array(linear),np.array(linear_known),transport,diagnostic)


def metrics(pmf, targets, known, compatible, *, log_pmf=None):
    """Cumulative scores and timing with explicit availability denominators."""
    with np.errstate(divide='ignore'):
        logs=np.log(pmf) if log_pmf is None else log_pmf
    logcdf=np.logaddexp.accumulate(logs,axis=-1)[...,:10]
    logsurvival=np.logaddexp.accumulate(logs[...,::-1],axis=-1)[...,::-1][...,1:]
    cdf=np.exp(logcdf);result={}
    for p,name in enumerate(('L','P')):
        m=known[:,p];end=m[:,-1];y=targets[:,p,-1][end].astype(bool);pred=cdf[:,p,-1][end]>=.5
        tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum());tn=int((~pred&~y).sum())
        bce=-np.where(targets[:,p][m].astype(bool),logcdf[:,p][m],logsurvival[:,p][m])
        infinite=int(np.isinf(bce).sum())
        exact=(compatible[:,p].sum(-1)==1)&(compatible[:,p,:10].sum(-1)==1)&end&(targets[:,p,-1]==1)
        observed=(compatible[:,p].argmax(-1)+1)*.5
        flipmass=pmf[:,p,:10].sum(-1);available=flipmass>0
        predicted=np.divide((pmf[:,p,:10]*np.arange(.5,5.1,.5)).sum(-1),flipmass,
                            out=np.full(len(pmf),np.nan),where=available)
        detected=exact&(cdf[:,p,-1]>=.5);usable=exact&available
        detected_usable=detected&available
        all_mae=float(np.abs(predicted[exact]-observed[exact]).mean()) if exact.any() and available[exact].all() else None
        result[name]={'known_cdf_cells':int(m.sum()),'known_F10':int(end.sum()),'positive_F10':int(y.sum()),
            'cdf_brier':float(((cdf[:,p][m]-targets[:,p][m])**2).mean()) if m.any() else None,
            'cdf_bce':float(bce.mean()) if len(bce) and not infinite else None,
            'cdf_bce_infinite_terms':infinite,'cdf_bce_null_reason':'zero_probability_observed_label' if infinite else 'no_known_labels' if not len(bce) else None,
            'F10_brier':float(((cdf[:,p,-1][end]-y)**2).mean()) if end.any() else None,
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'timing_estimator':'conditional mean E[T|T<=5s]; no tail timestamp',
            'known_exact_flips':int(exact.sum()),'interval_censored_flips':int(((targets[:,p,-1]==1)&end&~exact).sum()),
            'missed_known_exact_flips':int((exact&~detected).sum()),'timing_unavailable_flips':int((exact&~available).sum()),
            'timing_mae_all_known_flips_s':all_mae,'timing_all_denominator':int(exact.sum()),
            'timing_available_denominator':int(usable.sum()),'timing_detected_denominator':int(detected_usable.sum()),
            'timing_mae_detected_subset_s':float(np.abs(predicted[detected_usable]-observed[detected_usable]).mean()) if detected_usable.any() else None}
    return result


def predict_component_logs(model, batch):
    """One numerical forward returning mixture/donor/history log PMFs.

    Mirrors predict_batch's exact kwarg contract and adds the model's
    return_branch_logs request; no extra batch keys are read. Declared only
    for the component-supervision arms: a sampled-donor model in prefix or
    constant reliance mode, whose forward returns the (mixture, donor,
    history) branch tuple; every other configuration has no donor PMF and is
    rejected here instead of mis-read.
    """
    kwargs={key:batch[key] for key in FORWARD_KEYS}
    for key in ("prior_logits","owner_field_incidence","field_type_ids"):
        if key in batch:
            kwargs[key]=batch[key]
    logs=model(**kwargs,return_branch_logs=True)
    if not (isinstance(logs,tuple) and len(logs)==3
            and all(torch.is_tensor(item) and item.ndim==4 and item.shape[-1]==11 for item in logs)
            and logs[1] is not None):
        raise ValueError('component supervision needs a sampled-donor arm in prefix or '
                         f'constant reliance mode; this model returned {type(logs).__name__}')
    return logs


def evaluate_full(model, prepared, *, component_supervision_weight=0.):
    """Full-cohort log PMF and losses from one numerical forward per batch.

    hybrid is the unchanged primary cumulative_loss over the returned mixture
    log PMF. With component_supervision_weight>0 the K8-average history/donor
    log PMFs come from that same forward (no duplicate donor encoding) and
    total is the declared H(mixture)+w*(H(history)+H(donor))/2 objective;
    otherwise total is hybrid and both component losses are None.
    """
    model.eval()
    device=prepared.tensors['history'].device
    mixture=[];history=[];donor=[]
    with torch.no_grad():
        for i in range(0,prepared.size,128):
            batch=prepared.batch(torch.arange(i,min(i+128,prepared.size),device=device))
            if component_supervision_weight:
                mixture_logs,donor_logs,history_logs=predict_component_logs(model,batch)
                history.append(history_logs);donor.append(donor_logs)
            else:
                mixture_logs=predict_batch(model,batch)
            mixture.append(mixture_logs)
        targets=prepared.tensors['targets'];known=prepared.tensors['target_known']
        total=hybrid=None;history_loss=None;donor_loss=None
        if component_supervision_weight:
            # Branch logs are [B,P,K,11]; the K8-average log PMF is the
            # declared component/mixture PMF for every cumulative loss here.
            logs=torch.logsumexp(torch.cat(mixture),2)-np.log(8)
            hybrid=float(cumulative_loss(logs,targets,known,'hybrid'))
            history_loss=float(cumulative_loss(torch.logsumexp(torch.cat(history),2)-np.log(8),targets,known,'hybrid'))
            donor_loss=float(cumulative_loss(torch.logsumexp(torch.cat(donor),2)-np.log(8),targets,known,'hybrid'))
            total=hybrid+component_supervision_weight*(history_loss+donor_loss)/2
        else:
            logs=torch.cat(mixture)
            hybrid=float(cumulative_loss(logs,targets,known,'hybrid'))
            total=hybrid
    return logs.cpu().numpy(),hybrid,total,history_loss,donor_loss


def save_prepared(path, prepared, donors, linear, linear_known, windows, schedule, *,
                  q_arm='A', transport_summary=None, diagnostic=None):
    """Persist the reusable final numerical cache before optimization starts."""
    arrays={name:tensor.detach().cpu().numpy() for name,tensor in prepared.tensors.items()}
    arrays.update(donor_indices=donors,linear_bins=linear,linear_available=linear_known,schedule=schedule,
        window_keys=np.array([json.dumps(asdict(w)) for w in windows]),
        layout=np.array(json.dumps(asdict(prepared.layout))),split=np.array(prepared.split),cache_version=np.array(PILOT_VERSION),
        q_arm=np.array(q_arm),
        transport_summary=np.array(json.dumps(transport_summary or {})),
        q_diagnostic_pmf=np.stack([item[0] for item in diagnostic]) if diagnostic else np.zeros((0,2,11)),
        q_diagnostic_missing=np.stack([item[1] for item in diagnostic]) if diagnostic else np.zeros((0,2),np.int64))
    np.savez_compressed(path,**arrays)


def load_prepared_cache(path, *, device):
    """Restore saved numerical inputs without reopening source logs."""
    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value,list) else value
    metadata_keys={'donor_indices','linear_bins','linear_available','schedule','window_keys','layout','split',
                   'cache_version','q_arm','transport_summary','q_diagnostic_pmf','q_diagnostic_missing'}
    with np.load(path,allow_pickle=False) as archive:
        if str(archive['cache_version'])!=PILOT_VERSION:raise ValueError('prepared cache contract differs')
        missing=[key for key in ('q_arm','transport_summary','q_diagnostic_pmf','q_diagnostic_missing')
                 if key not in archive.files]
        if missing:raise ValueError(f'prepared cache predates the Q-arm contract: absent {missing}')
        if str(archive['q_arm']) not in Q_ARMS:
            raise ValueError(f'prepared cache names an undeclared Q arm: {str(archive["q_arm"])}')
        layout=TrajectoryLayout(**{k:tuples(v) for k,v in json.loads(str(archive['layout'])).items()})
        tensors={k:torch.as_tensor(np.array(archive[k],copy=True),device=device) for k in archive.files if k not in metadata_keys}
        metadata={k:np.array(archive[k],copy=True) for k in metadata_keys}
    return PreparedTrajectorySplit(tensors,layout,str(metadata['split']),None),metadata


def observed_transition_counts(logs, windows):
    """Post-load mathematical description, not event-script causal attribution."""
    result={}
    for key in dict.fromkeys(w.flow for w in windows):
        group=f'{key.episode_id}/{key.profile}/run{key.run}'
        if group not in result:
            result[group]={symbol:{'known_adjacent_flips':0,'false_to_true':0,'true_to_false':0,
                                   'unknown_or_out_of_scope_pairs':0} for symbol,*_ in RULES}
        times=range(0,logs._flows[key]['duration_ns']+1,500_000_000)
        for symbol,*_ in RULES:
            values=[logs._predicates[(key,t,symbol)]['value'] for t in times]
            counts=result[group][symbol]
            for previous,current in zip(values,values[1:]):
                if previous not in ('true','false') or current not in ('true','false'):
                    counts['unknown_or_out_of_scope_pairs']+=1
                elif previous!=current:
                    counts['known_adjacent_flips']+=1
                    counts['false_to_true' if current=='true' else 'true_to_false']+=1
    return result


def initialize_outcome_skip(model, value):
    with torch.no_grad():
        weight=model.trajectory_head.outcome_skip.weight
        weight.copy_(value*torch.eye(11,device=weight.device,dtype=weight.dtype))


def model_capacity(layout,*,reliance_mode='none'):
    """Setup-time dimensions and absent-branch active capacity, no source reads."""
    with torch.random.fork_rng(devices=[]):
        model=ControlledGenericTrajectoryModel(layout,sampling_enabled=True,reliance_mode=reliance_mode)
    head=model.trajectory_head
    total=sum(p.numel() for p in model.parameters())
    inactive=sum(p.numel() for name in ('branch_encoder','outcome_skip') for p in getattr(head,name).parameters())
    hidden=head.history_encoder.hidden_size
    inactive+=head.context_encoder[0].weight.shape[0]*hidden
    inactive+=head.residual[0].weight.shape[0]*11
    return {'hidden_dim':hidden,'field_dim':model.field_embedding.embedding_dim,
            'reliance_mode':reliance_mode,'instantiated_parameters':total,
            'reliance_gate_parameters':33 if reliance_mode!='none' else 0,
            'reliance_gate_data_active_parameters':33 if reliance_mode=='prefix' else 0,
            'sampling_on_active_parameters':total-(33 if reliance_mode=='constant' else 0),
            'sampling_off_instantiated_parameters':total-(33 if reliance_mode!='none' else 0),
            'sampling_off_active_parameters':total-inactive-(33 if reliance_mode!='none' else 0),
            'sampling_off_inactive_parameters':inactive,
            'active_definition':'parameters/input columns with nonzero data paths; absent branch modules remain instantiated'}


def fit_seed(args,out,protocol,seed,arm_cohorts,schedule,*,arms=('A','Off')):
    """Fit the declared Q arms for one seed; common init and schedule shared.

    arm_cohorts maps each arm to its {split: cohort} record. Every arm starts
    from the identical torch-seeded initialization and uses the exact same batch
    schedule; only the prepared operand cohort (and the absent branch for Off)
    differs. Final state, native loss curve and common metrics are saved per
    arm; no intermediate checkpoints are written.

    component_supervision_weight>0 optimizes the declared
    H(mixture)+w*(H(history)+H(donor))/2 objective on every TRAIN update and
    records its full-TRAIN decomposition at the existing evaluations; the
    saved losses[] stay the actual optimized values. weight 0 keeps the
    ordinary hybrid updates and curve untouched.
    """
    weight=float(getattr(args,'component_supervision_weight',0.) or 0.)
    device=next(iter(arm_cohorts.values()))['train'][0].tensors['history'].device
    for arm in arms:
        name,sampling,transform=Q_ARMS[arm];dest=out/f'{name}__seed{seed}';dest.mkdir()
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        cohorts=arm_cohorts[arm]
        model=ControlledGenericTrajectoryModel(cohorts['train'][0].layout,sampling_enabled=sampling,reliance_mode=args.reliance_mode).to(device=device,dtype=torch.float64)
        initialize_outcome_skip(model,args.outcome_skip_init)
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr);losses=[];evaluations=[]
        train=cohorts['train'][0]

        def evaluate_train(step):
            _,hybrid,total,history_loss,donor_loss=evaluate_full(model,train,component_supervision_weight=weight)
            row={'arm':name,'q_arm':arm,'seed':seed,'step':step,'full_TRAIN_hybrid':hybrid,
                 'full_TRAIN_total_objective':total,'full_TRAIN_history_loss':history_loss,
                 'full_TRAIN_donor_loss':donor_loss}
            evaluations.append(row);checkpoints.write(json.dumps(row)+'\n');checkpoints.flush()
            print(json.dumps(row),flush=True)

        def branch_hybrid(item,batch):
            # K8-average log PMF of one branch from its [B,P,K,11] branch logs.
            return cumulative_loss(torch.logsumexp(item,2)-np.log(8),
                                   batch['targets'],batch['target_known'],'hybrid')

        with (dest/'train_evaluations.jsonl').open('w') as checkpoints:
            evaluate_train(0)
            for step,indices in enumerate(schedule,1):
                batch=train.batch(torch.as_tensor(indices,device=device))
                if weight:
                    model.train();optimizer.zero_grad(set_to_none=True)
                    mixture_logs,donor_logs,history_logs=predict_component_logs(model,batch)
                    loss=(branch_hybrid(mixture_logs,batch)
                          +weight*(branch_hybrid(history_logs,batch)+branch_hybrid(donor_logs,batch))/2)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite cumulative objective')
                    loss.backward();optimizer.step();loss=loss.detach()
                else:
                    loss=train_step(model,optimizer,batch,mode='hybrid')
                losses.append(float(loss))
                if step%200==0 or step==args.steps:evaluate_train(step)
        torch.save({'step':args.steps,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'losses':losses,'evaluations':evaluations,'seed':seed,'q_arm':arm,'protocol':protocol,
                    'component_supervision_weight':weight,
                    'component_supervision_formula':COMPONENT_SUPERVISION_FORMULA},dest/'final_state.pt')
        summary={};arrays={'losses':np.array(losses)}
        for split,(prepared,donors,linear,linear_known,_,_) in cohorts.items():
            logs_pmf,hybrid,total,history_loss,donor_loss=evaluate_full(model,prepared,
                component_supervision_weight=weight if split=='train' else 0.)
            pmf=np.exp(logs_pmf)
            targets=prepared.tensors['targets'].cpu().numpy();known=prepared.tensors['target_known'].cpu().numpy()
            compatible=prepared.tensors['compatible'].cpu().numpy()
            linear_pmf=np.eye(11)[linear];linear_mask=known&linear_known[:,:,None]
            summary[split]={'ours':metrics(pmf,targets,known,compatible,log_pmf=logs_pmf),
                'linear':metrics(linear_pmf,targets,linear_mask,compatible),'hybrid':hybrid,
                'linear_unavailable_predicates':int((~linear_known).sum())}
            if split=='train':
                # Final full-TRAIN objective decomposition for this arm/seed.
                summary[split].update(full_TRAIN_hybrid=hybrid,full_TRAIN_total_objective=total,
                    full_TRAIN_history_loss=history_loss,full_TRAIN_donor_loss=donor_loss,
                    component_supervision_weight=weight,
                    component_supervision_formula=COMPONENT_SUPERVISION_FORMULA)
            if model.reliance_mode!='none':
                with torch.no_grad():
                    alpha=torch.cat([model.reliance_alpha(**{key:value for key,value in prepared.batch(
                        torch.arange(i,min(i+128,prepared.size),device=device)).items()
                        if key in ('history','history_known','predicate_field_incidence','owner_field_incidence','field_type_ids')})
                        for i in range(0,prepared.size,128)]).cpu().numpy()
                arrays[split+'_alpha']=alpha
                summary[split]['donor_reliance_alpha']={
                    'mean_by_predicate':alpha.mean(0).tolist(),'std_by_predicate':alpha.std(0).tolist(),
                    'quantiles_10_50_90_by_predicate':np.quantile(alpha,[.1,.5,.9],axis=0).tolist(),
                    'windows':len(alpha),'shared_across_K':True}
            arrays.update({f'{split}_'+name:value for name,value in dict(pmf=pmf,log_pmf=logs_pmf,targets=targets,
                known=known,compatible=compatible,linear_pmf=linear_pmf,linear_available=linear_known).items()})
        np.savez_compressed(dest/'final_predictions.npz',**arrays)
        (dest/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


def prepared_cache_paths(root, arm, seed):
    """Per-arm cache files; Off has no operands and reuses the A-arm cohort."""
    source='A' if arm=='Off' else arm
    return {split:Path(root)/f'prepared_{split}__{source}__seed{seed}.npz' for split in ('train','validation')}


def component_supervision_weight_of(args):
    """Validated --component-supervision-weight w>=0; experiment value .1.

    Declared validity: sampled-donor arms in prefix or constant reliance
    mode. Weight 0 keeps the ordinary hybrid objective anywhere.
    """
    weight=float(getattr(args,'component_supervision_weight',0.) or 0.)
    if weight<0 or not np.isfinite(weight):
        raise ValueError(f'component_supervision_weight must be finite >=0: {weight}')
    if weight:
        if getattr(args,'reliance_mode','none') not in ('prefix','constant'):
            raise ValueError('component supervision requires reliance mode prefix or constant')
        if 'Off' in getattr(args,'q_arms',()):
            raise ValueError('component supervision needs a donor PMF; arm Off samples none')
    return weight

def run_prepared_pilot(args):
    """Retrain declared arms from retained prepared caches.

    q_arms filters the trained arms; A/T/U load their per-arm cohort caches,
    Off reuses the A-arm cohort because the absent branch reads no operands.
    Each seed may be directed to its own retained cache root.
    """
    weight=component_supervision_weight_of(args)
    source=Path(args.prepared_root)
    protocol=json.loads((source/'protocol.json').read_text())
    protocol.update(outcome_skip_init=args.outcome_skip_init,lr=args.lr,batch_size=args.batch_size,
                    device=args.device,steps=args.steps,seeds=args.seeds,
                    prepared_root=str(source.resolve()),prepared_cache_reuse=True,q_arms=list(args.q_arms),
                    reliance_mode=args.reliance_mode,reliance_mixture='alpha*donorPMF+(1-alpha)*shared-historyPMF; shared prefix sigmoid32-to1; meanK8',
                    reliance_gate_parameters=33 if args.reliance_mode!='none' else 0,
                    component_supervision_weight=weight,component_supervision_formula=COMPONENT_SUPERVISION_FORMULA)
    unknown=[arm for arm in args.q_arms if arm not in Q_ARMS]
    if unknown:raise ValueError(f'undeclared Q arms: {unknown}')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    roots=args.prepared_roots or [str(source)]*len(args.seeds)
    if len(roots)!=len(args.seeds):
        raise ValueError('one prepared root per seed is required when several are given')
    per_seed={s:r for s,r in zip(args.seeds,roots) if r!=str(source)}
    if per_seed:protocol['prepared_root_by_seed']=per_seed
    for seed,root in zip(args.seeds,roots):
        cache_root=Path(root)
        cohorts={};schedule=None
        for arm in args.q_arms:
            for split,path in prepared_cache_paths(cache_root,arm,seed).items():
                prepared,metadata=load_prepared_cache(path,device=args.device)
                if str(metadata['q_arm'])!=('A' if arm=='Off' else arm):
                    raise ValueError(f'retained cache {path.name} belongs to arm {metadata["q_arm"]}')
                cohorts.setdefault(arm,{})[split]=(prepared,metadata['donor_indices'],metadata['linear_bins'],
                                           metadata['linear_available'],None,None)
                if schedule is None:schedule=metadata['schedule']
        if schedule is None or schedule.shape!=(args.steps,args.batch_size):
            raise ValueError('requested update/batch budget differs from the retained exact schedule')
        if 'model_capacity' not in protocol or protocol['model_capacity'].get('reliance_mode','none')!=args.reliance_mode:
            protocol['model_capacity']=model_capacity(next(iter(cohorts.values()))['train'][0].layout,reliance_mode=args.reliance_mode)
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        fit_seed(args,out,protocol,seed,cohorts,schedule,arms=tuple(args.q_arms))
    return protocol


def q_only_diagnostics(law, windows, out, diagnostic_cohorts):
    """Persist the Q-only offline diagnostic per sampling arm; not a launch gate."""
    summary={}
    for arm,(donors,diagnostic) in diagnostic_cohorts.items():
        result,arrays=q_diagnostic_summary(law,windows,donors,diagnostic,arm=arm)
        summary[arm]=result
        np.savez_compressed(out/f'q_only_diagnostic__{arm}.npz',**arrays)
    (out/'q_only_diagnostics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


def run_pilot(args):
    weight=component_supervision_weight_of(args)
    if args.prepared_root is not None:return run_prepared_pilot(args)
    if args.manifest is None and args.source_protocol is None:raise ValueError('source-log mode needs explicit manifest authority')
    root=Path(args.root)
    manifest=json.loads(Path(args.source_protocol).read_text())['manifest'] if args.source_protocol else json.loads(Path(args.manifest).read_text())
    parts=[]
    for selection in manifest['runs']:
        parts.append(load_r1_logs(root,(RunSelection(selection['episode_id'],selection['profile'],selection['run']),),
            scenario_by_episode=manifest['scenario_by_episode'],split_by_scenario=manifest['split_by_scenario'],
            r1_root=selection['r1_root']))
    merged={name:{} for name in ('_states','_predicates','_flows','configs')}
    splits={}
    for part in parts:
        splits.update(part.split_by_episode)
        for name,index in merged.items():
            source=getattr(part,name)
            if index.keys() & source.keys():raise ValueError('duplicate explicitly selected run or flow index')
            index.update(source)
    logs=R1TrainingLogs(split_by_episode=splits,scenario_by_episode=manifest['scenario_by_episode'],
        states=merged['_states'],predicates=merged['_predicates'],flows=merged['_flows'],configs=merged['configs'])
    mission={e:mission_ids(root,e,scenario) for e,scenario in manifest['scenario_by_episode'].items()}
    windows={split:tuple(w for w in logs.eligible_windows(split=split) if w.flow.owner in mission[w.flow.episode_id])
             for split in ('train','validation')}
    if not windows['validation']:
        raise ValueError('independent scenario VALID cohort is absent')
    if set(manifest['split_by_scenario'].values()) != {'train','validation'}:
        raise ValueError('pilot requires explicitly distinct TRAIN and VALID scenarios')
    if not torch.cuda.is_available():raise RuntimeError('GPU execution required for this pilot')
    device=torch.device(args.device);law=EmpiricalJointLaw(logs,windows['train'])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    protocol={'version':PILOT_VERSION,'manifest':manifest,'steps':args.steps,'seeds':args.seeds,
        'outcome_skip_init':args.outcome_skip_init,'lr':args.lr,'batch_size':args.batch_size,'device':args.device,
        'reliance_mode':args.reliance_mode,'reliance_gate_parameters':33 if args.reliance_mode!='none' else 0,
        'component_supervision_weight':weight,'component_supervision_formula':COMPONENT_SUPERVISION_FORMULA,
        'prepared_cache_reuse':False,'K':8,'history_steps':8,'future_tick_offsets':list(STATE_OFFSETS),'tick_ns':TICK_NS,
        'loss':'knownCDF_square1..9 + 0.5knownBCE10','decision_F10':.5,'prior':'last-two-observed-linear8I',
        'candidate_selection':'TRAIN donors, query scenario excluded; per-arm donor draws recorded in every prepared cache',
        'sampling_off':'future operands/rules never read; fixed zero branch hidden/outcome',
        'sampling_on':'8 without-replacement donors per window; A/T nearest64, U uniform eligible TRAIN',
        'q_arms':{arm:{'directory':name,'sampling':sampling,'transform':transform}
                  for arm,(name,sampling,transform) in Q_ARMS.items()},
        'q_transport_law':'T/U: query_reference + (donor_future - donor_reference) per field, joint donor draws '
                          'retained; reference = anchor when known, else last known PREFIX operand of that field; '
                          'absolute donor field when either side lacks a reference (counted, never query truth); '
                          'latency projected to [0,(query packet_ttl_ns-1)/1e6] ms, loss to [0,1] with recorded pre-projection violations, '
                          'projection fractions and endpoint mass; exact endpoints preserved, no epsilon',
        'q_diagnostic':'Q-only offline report (q_only_diagnostics.json): anchor-side mismatch, mature donor '
                       'fixed-rule first-flip PMF/CDF scoring, support/missing counts; not a launch gate',
        'q_common_protocol':'per-seed common torch initialization and identical batch schedule across arms',
        'normalization_mean':law.mean.tolist(),'normalization_scale':law.scale.tolist(),
        'counts':{split:logs.label_counts(ws) for split,ws in windows.items()},
        'event_blind_split':True,'observed_grid_transitions':{split:observed_transition_counts(logs,ws) for split,ws in windows.items()},
        'mission_ids':{e:sorted(ids) for e,ids in mission.items()},'neural_backbone':'dense shared generic trajectory; PyG assembly artifact'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for seed in args.seeds:
        schedule=np.random.default_rng(seed).integers(0,len(windows['train']),size=(args.steps,args.batch_size))
        arm_cohorts={};diagnostic_cohorts={}
        for arm in args.q_arms:
            if not Q_ARMS[arm][1]:continue
            cohorts={}
            for split in ('train','validation'):
                cohorts[split]=prepare_cohort(logs,windows[split],law,seed=seed,device=device,arm=arm)
                prepared,donors,linear,linear_known,transport,diagnostic=cohorts[split]
                save_prepared(out/f'prepared_{split}__{arm}__seed{seed}.npz',prepared,donors,linear,linear_known,
                    windows[split],schedule,q_arm=arm,
                    transport_summary=q_transport_summary(law,transport,arm=f'{arm}.{split}'),
                    diagnostic=diagnostic)
            arm_cohorts[arm]=cohorts
            diagnostic_cohorts[arm]=(cohorts['validation'][1],cohorts['validation'][5])
        if not arm_cohorts:
            raise ValueError('declared arms contain no sampling cohort')
        if 'Off' in args.q_arms:
            # The absent branch reads no operands; it shares the first cohort.
            arm_cohorts['Off']=arm_cohorts[next(iter(arm_cohorts))]
        if 'model_capacity' not in protocol:
            protocol['model_capacity']=model_capacity(next(iter(arm_cohorts.values()))['train'][0].layout,reliance_mode=args.reliance_mode)
            (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        fit_seed(args,out,protocol,seed,arm_cohorts,schedule,arms=tuple(args.q_arms))
        q_only_diagnostics(law,windows['validation'],out,diagnostic_cohorts)
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='/mnt/data2/weizhiwei/AERO_WORLD');parser.add_argument('--manifest');parser.add_argument('--source-protocol');parser.add_argument('--prepared-root')
    parser.add_argument('--prepared-roots',nargs='*')
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--seeds',type=int,nargs='+',default=[0,1])
    parser.add_argument('--q-arms',nargs='+',default=['A','Off'],choices=tuple(Q_ARMS))
    parser.add_argument('--reliance-mode',choices=('none','prefix','constant'),default='none')
    parser.add_argument('--component-supervision-weight',type=float,default=0.,
        help='w>=0; optimizes H(mixture)+w*(H(history)+H(donor))/2 with the same known mask (experiment: 0.1); '
             'valid for prefix/constant reliance modes; 0 keeps the ordinary hybrid objective')
    parser.add_argument('--outcome-skip-init',type=float,default=4.0)
    parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--lr',type=float,default=.003)
    run_pilot(parser.parse_args())


if __name__=='__main__':main()
