"""Single-arm execution runtime for the L6 v2 workflow.

One arm = one dedicated Python entry under scripts/arms/. The entry declares its
own window, its own mechanism (trigger retiming, exact action omissions, exact
injected actions) and its own hypothesis. This module only executes what the
entry declares; it never derives a mechanism from an intent name or an arm id.

Two modes:
  factual_observation  factual process within the declared window
  intervention         execution with the declared mechanism, compared with factual
"""
from __future__ import annotations

import copy
import datetime
import json
import math
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[3]
for _candidate in (str(_ROOT), str(_ROOT / 'Dataset/tools'),
                   str(_ROOT / 'Plugins/SumoImporter/Scripts')):
    if _candidate not in _sys.path:
        _sys.path.insert(0, _candidate)

from Dataset.tools.l6_v2 import semantics
from Dataset.tools.l6_v2.common import (
    ROOT, RUNTIME_FAMILIES, COMPARE_FIELDS, TICK_HZ, digest, file_digest, grid_up, load, project_path,
    rows, write, write_rows,
)
from Dataset.tools.l6_v2.engine import execute, snapshot
from Dataset.tools.l6_v2.scope import aircraft_geometry, select_scope
from Dataset.semantic_truth.outcome_projection import source_index, validated_recovery_tick

BASE_PREDICATES = (
    'airspace.aircraft_airborne', 'airspace.aircraft_touchdown', 'airspace.aircraft_at_home_pad',
    'airspace.aircraft_in_landing_zone', 'airspace.aircraft_outside_assigned_altitude_band',
    'communication.link_degraded', 'communication.link_unavailable',
    'emergency.aircraft_in_return_to_home', 'emergency.aircraft_in_safe_hold', 'positioning.aircraft_deviates_from_planned_route',
)
PREDICATE_ADDITIONS = {
    'positioning.': ('positioning.gnss_quality_degraded', 'positioning.gnss_spoofing_active',
                     'positioning.navigation_error_exceeds_tolerance'),
}
BASE_FAMILIES = ('uav_takeoff', 'uav_landing', 'altitude_excursion', 'c2_degradation_loss')
FAMILY_ADDITIONS = {
    'positioning.': ('gnss_signal_degradation', 'gnss_spoofing_effect'),
    'security.communication_jamming_impact': ('jamming_gcs_intrusion',),
    'security.ground_control_station': ('jamming_gcs_intrusion',),
}


@dataclass(frozen=True)
class ArmSpec:
    """Everything one dedicated ARM entry must declare about itself."""
    profile: str
    seed: str
    anchor_event_id: str
    arm_id: str
    mode: str
    role: str
    window_start_tick: int
    window_fork_tick: int
    window_end_tick: int
    primary_predicate: str
    subject_entity_ids: tuple[str, ...]
    mechanism: Mapping[str, Any] = field(default_factory=dict)
    retime: Mapping[str, Any] | None = None
    omit_action_ids: tuple[str, ...] = ()
    inject_actions: tuple[Mapping[str, Any], ...] = ()
    cancel_event_ids: tuple[str, ...] = ()
    hypothesis: str = ''
    falsified_if: str = ''
    window_basis: str = ''
    window_needed_nodes: tuple[Mapping[str, Any], ...] = ()
    window_censored: bool = False
    completion_requirements: tuple[Mapping[str, Any], ...] = ()
    group_suffix: str = 'v2'

    @property
    def episode_id(self) -> str:
        return f'{self.profile}__{self.seed}'

    @property
    def group_key(self) -> str:
        stem = f'grp_{self.anchor_event_id}' + (f'_{self.group_suffix}' if self.group_suffix else '')
        return f'{self.profile}/{self.seed}/{stem}'

    @property
    def output(self) -> str:
        return f'arms/{self.group_key}/{self.arm_id}'

    @property
    def scene(self) -> str:
        return f'Dataset/scenarios/L6_digital_layer/failure/{self.profile}/scene_setup.json'

    @property
    def script(self) -> str:
        return f'Dataset/scenarios/L6_digital_layer/failure/{self.profile}/event_script.json'

    def predicate_ids(self) -> list[str]:
        identifiers = set(BASE_PREDICATES)
        identifiers.add(self.primary_predicate)
        for prefix, extra in PREDICATE_ADDITIONS.items():
            if self.primary_predicate.startswith(prefix):
                identifiers.update(extra)
        return sorted(identifiers)

    def event_families(self) -> list[str]:
        families = set(BASE_FAMILIES)
        for prefix, extra in FAMILY_ADDITIONS.items():
            if self.primary_predicate.startswith(prefix):
                families.update(extra)
        return sorted(families)

    def window(self) -> dict:
        return {'start_tick': self.window_start_tick, 'fork_tick': self.window_fork_tick,
                'end_tick': self.window_end_tick,
                'phase': 'after_state_sample_before_actions'}


def declared_family(profile: str, anchor_event_id: str) -> str:
    table = load(ROOT / 'design/L6/anchor_families.json')['families']
    return table[f'{profile}|{anchor_event_id}']


def make_group_dir(spec: ArmSpec) -> Path:
    directory = ROOT / 'arms' / spec.group_key
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def roster_for_scope(scope: Mapping[str, Any], engine: Any) -> dict:
    """Build the governed roster records the entity-scope contract reads.

    The engine entity map uses ``category``; the governed contract classifies on
    ``entity_category`` plus the authored role fields. This projection supplies the
    contract field and copies the authored roles verbatim, so event-role binding is
    decided by the same declarations the scene author wrote.
    """
    roster = {}
    for entity_id in scope['entity_ids']:
        entity = engine.entities[entity_id]
        roster[entity_id] = {
            'entity_id': entity_id,
            'entity_category': entity.get('entity_category') or entity.get('category'),
            'role': entity.get('role'),
            'semantic_role': entity.get('semantic_role'),
            'uav_corridor_role': entity.get('uav_corridor_role'),
            'background_role': entity.get('background_role'),
            'label_class': entity.get('label_class'),
            'initial_state': {'role': entity.get('role'),
                              'semantic_role': entity.get('semantic_role')},
        }
    return roster


def run_arm(spec: ArmSpec) -> dict:
    out = project_path(spec.output)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    created_history_paths: list[Path] = []
    try:
        return _run_arm(spec, created_history_paths)
    except Exception:
        if out.exists():
            shutil.rmtree(out)
        for history_path in created_history_paths:
            history_path.unlink(missing_ok=True)
            try:
                history_path.parent.rmdir()
            except OSError:
                pass
        raise


def _run_arm(spec: ArmSpec, created_history_paths: list[Path]) -> dict:
    """Execute one arm and write its window artifacts."""
    if spec.mode not in ('factual_observation', 'intervention'):
        raise ValueError(f'Unsupported ARM mode: {spec.mode}')
    if spec.mode == 'factual_observation' and (spec.retime or spec.omit_action_ids or spec.inject_actions or spec.cancel_event_ids):
        raise ValueError('factual observation cannot declare intervention edits')
    from dataclasses import asdict
    executable = {'retime': spec.retime, 'omit_action_ids': list(spec.omit_action_ids),
                  'cancel_event_ids': list(spec.cancel_event_ids), 'injections': list(spec.inject_actions),
                  'inject_actions': list(spec.inject_actions)}
    for key, value in executable.items():
        if key in spec.mechanism and json.loads(json.dumps(spec.mechanism[key])) != json.loads(json.dumps(value)):
            raise ValueError('mechanism metadata differs from executable field: ' + key)
    for requirement in spec.completion_requirements:
        if requirement['required_value'] not in ('true', 'false'):
            raise ValueError('completion required_value must be true or false')
        if requirement['predicate_id'] not in spec.predicate_ids():
            raise ValueError('completion predicate is outside the declared scope')
        roles = {r['key'] for r in semantics.TEMPLATES[requirement['predicate_id']]['argument_roles']}
        if set(requirement['bindings']) != roles:
            raise ValueError('completion binding roles differ from predicate authority')
    out = project_path(spec.output)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clock = time.perf_counter()
    engine_run = execute(spec.scene, spec.script, spec.episode_id,
                         end_tick=spec.window_end_tick, retime=spec.retime,
                         omit_action_ids=spec.omit_action_ids, injections=spec.inject_actions,
                         cancel_event_ids=spec.cancel_event_ids,
                         snapshot_ticks=(spec.window_start_tick, spec.window_fork_tick),
                         fork_tick=spec.window_fork_tick)
    base_engine = engine_run['engine']
    scope = select_scope(base_engine, load(project_path(spec.script)))
    geometry = aircraft_geometry(base_engine, scope)
    roster = roster_for_scope(scope, base_engine)
    config = {
        'group_key': spec.group_key, 'episode_id': spec.episode_id, 'arm_id': spec.arm_id,
        'scope': scope, 'aircraft_geometry': geometry,
        'predicate_ids': spec.predicate_ids(), 'event_families': spec.event_families(),
        'producer_seed_digest': digest({'episode_id': spec.episode_id}),
        'output': spec.output, 'sources': {'scene': spec.scene, 'script': spec.script},
    }

    trajectories = [r for r in base_engine.trajectory_rows
                    if r['entity_id'] in scope['entity_ids']]
    weather = list(base_engine.weather_rows)
    audit = engine_run['audit']
    method = ('current_engine_with_declared_mechanism' if spec.mode == 'intervention'
              else 'current_engine_no_intervention_factual_baseline')

    start, end = spec.window_start_tick, spec.window_end_tick
    computed = semantics.evaluate(config, trajectories, weather, roster)
    from Dataset.tools.l6_v2.observations import window_semantics, physical_observations
    objective, history_ref, inherited, chain_rows = window_semantics(
        spec, computed, trajectories, weather,
        validate_objective=lambda projected: _chain(
            projected, source_truth=computed['objective']['predicate_truth'],
            source_transitions=computed['objective']['predicate_transitions'],
            window_start_tick=start, window_end_tick=end),
        created_history_paths=created_history_paths)
    nodes, relations = chain_rows
    out.mkdir(parents=True)
    full_trajectories = trajectories
    trajectories = [r for r in trajectories if start <= r['tick'] <= end]
    weather = [r for r in weather if start <= r['tick'] <= end]
    audit = [r for r in audit if start <= r['tick'] <= end]
    write(out / 'objective/initial_events.json', inherited)
    write(out / 'objective/physical_observations.json',
          physical_observations(spec, engine_run, scope, full_trajectories, geometry))
    write(out / 'raw/event_dispositions.json', engine_run['event_dispositions'])
    write(out / 'raw/event_dispatches.json', engine_run['event_dispatches'])
    write(out / 'raw/executed_event_script.json', engine_run['interpreter'].script)

    snapshots = engine_run['snapshots']
    write(out / 'raw/snapshots.json', snapshots)
    write(out / 'raw/initial_state.json', snapshots[str(start)])
    write(out / 'raw/fork_state.json', snapshots[str(spec.window_fork_tick)])
    write_rows(out / 'raw/trajectories.jsonl', trajectories)
    write_rows(out / 'raw/states.jsonl', (
        {'tick': r['tick'], 'entity_id': r['entity_id'], 'present': True,
         'values': {f: r[f] for f in COMPARE_FIELDS if f in r}, 'missing_reason': None}
        for r in trajectories))
    write_rows(out / 'raw/weather.jsonl', weather)
    write_rows(out / 'raw/actions.jsonl', audit)
    for name, data in computed['raw'].items():
        write_rows(out / 'raw' / f'{name}.jsonl',
                   [r for r in data if start <= int(r.get('tick', start)) <= end])
    write(out / 'raw/final_schedules.json',
          {e: engine_run['engine'].entities[e]['schedules'] for e in scope['entity_ids']})
    if spec.retime:
        write(out / 'raw/retimed_trigger.json', engine_run['retimed_trigger'])
    status = _window_status(spec, objective)
    write(out / 'objective/window_status.json', status)
    for name, data in objective.items():
        write_rows(out / 'objective' / f'{name}.jsonl', data)
    known = {n["node_id"] for n in nodes}
    for row in inherited["transitions"]:
        if row["transition_id"] not in known:
            nodes.append({"node_id": row["transition_id"], "kind": "inherited_predicate_transition",
                          "tick": row["to_tick"], "evidence_ref": history_ref["path"] + "#" + row["transition_id"]})
    write_rows(out / 'chain/nodes.jsonl', nodes)
    write_rows(out / 'chain/relations.jsonl', relations)

    from Dataset.tools.arm_ue_export import write_engine_inputs
    write_engine_inputs(out, base_engine, engine_run['interpreter'], spec.window(),
                        script=engine_run['interpreter'].script)
    manifest = _manifest(spec, scope, objective, status, out, method, started, clock)
    manifest['source_refs']['semantic_history'] = history_ref
    config_path = ROOT / 'interventions' / spec.group_key / (spec.arm_id + '.json')
    entry_path = ROOT / 'scripts/arms' / spec.group_key / (spec.arm_id + '.py')
    from dataclasses import asdict
    if json.loads(json.dumps(asdict(spec))) != json.loads(json.dumps(asdict(ArmSpec(**load(config_path)['spec'])))):
        raise ValueError('entry ArmSpec differs from bound config')
    manifest['source_refs']['configuration'] = {'path': config_path.relative_to(ROOT).as_posix()}
    manifest['source_refs']['entry'] = {'path': entry_path.relative_to(ROOT).as_posix()}
    experiment_path = ROOT / 'design' / spec.group_key / 'experiment.json'
    manifest['source_refs']['experiment'] = {'path': experiment_path.relative_to(ROOT).as_posix()}
    write(out / 'manifest.json', manifest)
    return {'arm_id': spec.arm_id, 'group': spec.group_key, 'mode': spec.mode,
            'window': spec.window(), 'status': status['status'],
            'primary_formal_counts': status['primary_formal_counts'],
            'events': status['event_family_counts'],
            'elapsed_s': round(time.perf_counter() - clock, 2)}


def _window_status(spec: ArmSpec, objective: Mapping[str, Any]) -> dict:
    occurrences = [r for r in objective['event_occurrences']
                   if r['event_family_id'] == declared_family(spec.profile, spec.anchor_event_id)]
    outcomes = {r['event_id']: r for r in objective['event_outcomes']}
    start, end = spec.window_start_tick, spec.window_end_tick
    rows_primary = [r for r in objective['predicate_truth_ticks']
                    if r['predicate_id'] == spec.primary_predicate
                    and r['sampling_scope'] == 'formal'
                    and start <= r['tick'] <= end
                    and r['value'] != 'out_of_scope']
    counts = Counter(r['value'] for r in rows_primary)
    pending = [o for o in occurrences
               if outcomes[o['event_id']]['terminal_tick'] is None]
    if pending:
        status = 'right_censored'
    elif occurrences:
        status = 'events_recovered'
    else:
        status = 'target_absent_in_observed_window'
    formal_status = status
    completion = []
    for requirement in spec.completion_requirements:
        sampled = [r for r in objective['predicate_truth_ticks']
                   if r['sampling_scope'] == 'formal'
                   and r['predicate_id'] == requirement['predicate_id']
                   and r['bindings'] == requirement['bindings']]
        count = int(semantics.PARAMS['recovery_hold_samples'])
        last_tick = end - end % 5
        needed = list(range(last_tick - (count - 1) * 5, last_tick + 1, 5))
        samples = [r for r in sampled if r['tick'] in needed]
        if [r['tick'] for r in samples] != needed:
            raise ValueError('missing completion observation samples')
        if any(r['value'] in ('unknown', 'out_of_scope') for r in samples):
            raise ValueError('completion requirement lacks measured truth')
        satisfied = all(r['value'] == requirement['required_value'] for r in samples)
        completion.append({**dict(requirement), 'satisfied': satisfied,
                           'hold_samples': count, 'samples': [{'tick': r['tick'], 'value': r['value'],
                                                             'source_refs': r['source_refs']} for r in samples]})
    if any(not r['satisfied'] for r in completion):
        status = 'right_censored'
    return {'window': spec.window(), 'status': status,
            'formal_lifecycle_status': formal_status, 'completion_requirements': completion,
            'absence_is_only_within_window': True,
            'status_scope': 'declared_anchor_family_and_explicit_completion_requirements',
            'ancillary_event_count': len(objective['event_occurrences']) - len(occurrences),
            'primary_predicate': spec.primary_predicate,
            'primary_formal_counts': dict(counts),
            'event_family_counts': dict(Counter(o['event_family_id'] for o in occurrences)),
            'events': [{'event_id': o['event_id'], 'family': o['event_family_id'],
                        'onset_tick': o['trigger_tick'],
                        'terminal_tick': outcomes[o['event_id']]['terminal_tick'],
                        'status': outcomes[o['event_id']]['status']}
                       for o in occurrences]}


def _chain(objective: Mapping[str, Any], *, window_start_tick: int,
           window_end_tick: int, source_truth: Sequence[Mapping[str, Any]],
           source_transitions: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    nodes, relations = [], []
    occurrence_by_id = source_index(objective['event_occurrences'], 'event_id')
    truth_by_id = source_index(source_truth, 'truth_id')
    transition_by_id = source_index(source_transitions, 'transition_id')
    for transition in objective['predicate_transitions']:
        nodes.append({'node_id': transition['transition_id'], 'kind': 'predicate_transition',
                      'tick': transition['to_tick'],
                      'evidence_ref': 'objective/predicate_transitions.jsonl#'
                                      + transition['transition_id']})
    for occurrence in objective['event_occurrences']:
        nodes.append({'node_id': occurrence['event_id'], 'kind': 'objective_event',
                      'tick': occurrence['trigger_tick'],
                      'evidence_ref': 'objective/event_occurrences.jsonl#'
                                      + occurrence['event_id']})
        for phase in occurrence['event_phases']:
            relations.append({'source': phase['transition_id'], 'target': occurrence['event_id'],
                              'relation_type': 'predicate_' + phase['phase_kind'] + '_evidence',
                              'tick': phase['tick'], 'causal_claim': False,
                              'rule_id': occurrence['rule_id']})
    for outcome in objective['event_outcomes']:
        recovery_tick = validated_recovery_tick(
            outcome, occurrence=occurrence_by_id.get(outcome.get('event_id')),
            truth_by_id=truth_by_id, transition_by_id=transition_by_id,
            window_start_tick=window_start_tick, window_end_tick=window_end_tick)
        if recovery_tick is None:
            continue
        nodes.append({'node_id': outcome['outcome_id'], 'kind': 'event_recovery_confirmed',
                      'tick': recovery_tick,
                      'evidence_ref': 'objective/event_outcomes.jsonl#' + outcome['outcome_id']})
        relations.append({'source': outcome['event_id'], 'target': outcome['outcome_id'],
                          'relation_type': 'has_observed_outcome', 'causal_claim': False,
                          'tick': recovery_tick})
    return (sorted(nodes, key=lambda r: (r['tick'], r['node_id'])),
            sorted(relations, key=lambda r: (r['tick'], r['source'], r['target'])))


def _implementation_sources() -> list:
    paths = [ROOT / 'Dataset/tools/batch_generate.py',
             ROOT / 'Plugins/SumoImporter/Scripts/donghu_core/event_script_interpreter.py',
             *(ROOT / 'Dataset/tools/l6_v2' / name for name in
               ('arm_runtime.py', 'engine.py', 'semantics.py', 'scope.py', 'common.py', 'observations.py')),
             ROOT / 'Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json',
             ROOT / 'Dataset/semantic_rules/profiles/entity_scope_classes.json',
             ROOT / 'Dataset/semantic_truth/entity_scope.py',
             ROOT / 'Dataset/semantic_simulation/compute_comm.py',
             ROOT / 'Dataset/semantic_simulation/domain_state.py',
             ROOT / 'Dataset/semantic_simulation/control_response_state.py']
    paths.extend((ROOT / 'Dataset/semantic_truth').glob('*.py'))
    paths.extend((ROOT / 'Dataset/semantic_rules').rglob('*.json'))
    paths.append(ROOT / 'Dataset/tools/arm_ue_export.py')
    return sorted({p.relative_to(ROOT).as_posix() for p in paths})


def _manifest(spec: ArmSpec, scope: dict, objective: Mapping[str, Any], status: dict,
              out: Path, method: str, started: str, clock: float) -> dict:
    source_refs = {
        'scene': {'path': spec.scene},
        'script': {'path': spec.script},
    }
    roster_path = f'{scope["entity_ids"][0]}'
    return {
        'schema': 'aero_l6_v2_arm_manifest', 'version': 1,
        'episode_id': spec.episode_id, 'profile': spec.profile, 'seed': spec.seed,
        'group_key': spec.group_key, 'arm_id': spec.arm_id, 'mode': spec.mode,
        'role': spec.role, 'anchor_event_id': spec.anchor_event_id,
        'declared_anchor_family': declared_family(spec.profile, spec.anchor_event_id),
        'anchor_dispatch_tick': spec.window_fork_tick,
        'window': spec.window(), 'tick_hz': TICK_HZ, 'formal_step_ticks': 5,
        'window_basis': spec.window_basis,
        'window_needed_nodes': [dict(n) for n in spec.window_needed_nodes],
        'window_censored': spec.window_censored,
        'completion_requirements': [dict(r) for r in spec.completion_requirements],
        'scope': scope, 'predicate_ids': spec.predicate_ids(),
        'event_families': spec.event_families(), 'primary_predicate': spec.primary_predicate,
        'generation_method': method,
        'ue_script_role': 'Executed source context after explicit cancellation and retiming; action omissions and injections remain in raw/actions.jsonl and declared_mechanism. The UE package consumes measured truth, not independent script replay.',
        'declared_mechanism': {
            'kind': spec.mechanism.get('kind'),
            'retime': dict(spec.retime) if spec.retime else None,
            'omit_action_ids': list(spec.omit_action_ids),
            'inject_actions': [dict(a) for a in spec.inject_actions],
            'cancel_event_ids': list(spec.cancel_event_ids),
        },
        'hypothesis': spec.hypothesis, 'falsified_if': spec.falsified_if,
        'provenance': {
            'parameter_sources': {
                'scene': spec.scene, 'script': spec.script,
                'delay_ticks': (spec.retime or {}).get('delta'),
                'delay_source': 'explicit configuration' if spec.retime else None,
                'factual_baseline': 'current engine, no intervention, same window',
            },
            'implementation_sources': _implementation_sources(),
        },
        'source_refs': source_refs,
        'artifacts': sorted(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()),
        'record_counts': {k: len(v) for k, v in objective.items()},
        'observed': {'window_status': status['status'],
                     'primary_formal_counts': status['primary_formal_counts'],
                     'event_count': len(objective['event_occurrences'])},
        'run_receipt': {'started_utc': started,
                        'finished_utc': datetime.datetime.now(
                            datetime.timezone.utc).isoformat(),
                        'elapsed_s': time.perf_counter() - clock,
                        'python': sys.version, 'executable': sys.executable,
                        'platform': platform.platform(), 'exit_code': 0},
        'review': {'review_type': 'generator_self_review_not_independent',
                   'release_certification': 'not_claimed'},
    }
