"""Semantic layer for L6 v2 arms.

Predicate values and objective events are computed by the shipped registry
expressions and producers. The roster is passed into the event adapter so the
governed entity-scope contract can bind event roles; without it the adapter
cannot tell a mission aircraft from background presence.
"""
from __future__ import annotations

import copy
import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from Dataset.tools.l6_v2.common import (
    ROOT, RUNTIME_FAMILIES, load, digest, rows,
)

from Dataset.semantic_simulation import compute_comm as comm
from Dataset.semantic_simulation import control_response_state as control
from Dataset.semantic_simulation import domain_state as domain
from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types, get_core_predicate_templates,
    get_governed_parameter_defaults,
)
from Dataset.semantic_truth.minimal_semantics import (
    Assertion, ObservationRow, Truth, build_events, build_outcomes, build_transitions,
)
from Dataset.semantic_truth.minimal_semantics_adapter import (
    _adapt_assertions, _adapt_occurrences, _adapt_outcomes, _adapt_transitions,
)
from Dataset.semantic_truth.world_truth import _evaluate_expression, _missing_value, _value_at_path

PARAMS = get_governed_parameter_defaults()
TEMPLATES = {t['id']: t for t in get_core_predicate_templates()}
COMM_PATH = ROOT / 'Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json'
DOMAIN_PATH = ROOT / 'Dataset/semantic_rules/profiles/domain_state_supplement_profile.json'


class FocusedDomainIdentity:
    """Only input consumed by the focused security observation primitive."""

    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id


def local_frames(config: Mapping[str, Any], trajectories: list[dict]) -> list[dict]:
    frames: dict[int, list[dict]] = defaultdict(list)
    scope = config['scope']
    for row in trajectories:
        entity = {k: copy.deepcopy(v) for k, v in row.items()
                  if k in ('entity_id', 'entity_kind', 'entity_type', 'assigned_altitude_m',
                           'planned_route_waypoints_enu_m', *RUNTIME_FAMILIES)}
        entity_id = row['entity_id']
        if entity_id in scope['aircraft_ids']:
            category = 'uav'
        elif entity_id in scope['station_ids']:
            category = 'communication_station'
        elif entity_id in scope['gcs_ids']:
            category = 'ground_station'
        elif entity_id in scope['home_pad_ids']:
            category = 'facility'
        else:
            raise ValueError('unclassified declared entity ' + entity_id)
        entity['entity_category'] = category
        entity['truth_pose'] = {'position_enu_m': row['pos_enu'],
                                'velocity_enu_mps': row['vel_mps']}
        frames[row['tick']].append(entity)
    return [{'episode_id': config['episode_id'], 'tick': tick, 'sim_time_s': tick / 10.0,
             'entities': frames[tick], 'scope': 'current_source_script_selected_entities_only'}
            for tick in sorted(frames)]


def communication(config: Mapping[str, Any], frames: list[dict], weather: list[dict],
                  formal: bool) -> list[dict]:
    profile = comm.load_compute_comm_profile(COMM_PATH)
    actual = [f for f in frames if not formal or f['tick'] % PARAMS['formal_step_ticks'] == 0]
    first = next((f for f in frames if f['entities']), None)
    if first is None:
        raise ValueError('no entity frames supplied to the communication producer')
    roster = {e['entity_id']: e for e in first['entities']}
    stations = [e for e in roster.values() if comm._is_station_entity(e)]
    if not stations:
        # No authored communication station exists in this scene. The governed
        # communication predicates cannot be evaluated; return no rows and let the
        # predicate layer report unknown rather than fabricating a station result.
        return []
    identity = config['episode_id'] + '::' + config['group_key'] + '/' + config['arm_id']
    inputs = comm.EpisodeInputs(
        episode_id=identity, episode_root=ROOT / config['output'],
        manifest={'episode_id': identity, 'tick_hz': 10}, tick_hz=10.0,
        tick_hz_source_ref='Dataset/tools/batch_generate.py#TICK_HZ',
        roster_entities=roster, frames=actual,
        weather_by_tick={r['tick']: r for r in weather},
        input_files={'truth_frames': Path('raw/local_frames.jsonl'),
                     'weather': Path('raw/weather.jsonl')},
        input_digest=digest([actual, weather]))
    common, _ = comm._simulation_common(inputs, profile, COMM_PATH)
    result = comm._build_communication_rows(inputs, profile, common)
    for row in result:
        if row['station_source'] != 'observed_simulator_truth':
            raise ValueError('non-authored communication station')
        row['producer_source_refs'] = row['source_refs']
        row['source_refs'] = [
            f"raw/local_frames.jsonl#tick={row['tick']}&entity={row['entity_id']}",
            f"raw/local_frames.jsonl#tick={row['tick']}&entity={row['station_id']}",
            f"raw/weather.jsonl#tick={row['tick']}"]
        row['sampling_scope'] = 'formal' if formal else 'off_grid_diagnostic_sequence'
    return result


def producer_observations(config: Mapping[str, Any], frames: list[dict], weather: list[dict]):
    formal_comm = communication(config, frames, weather, True)
    dense_comm = communication(config, frames, weather, False)
    comm_index = {(r['tick'], r['entity_id']): r for r in dense_comm}
    comm_index.update({(r['tick'], r['entity_id']): r for r in formal_comm})
    profile = domain.load_domain_state_profile(DOMAIN_PATH)
    gnss_params = profile['domain_models']['gnss']
    scope = config['scope']
    episode = config['episode_id'] + '::' + config['group_key'] + '/' + config['arm_id']
    seed = config['producer_seed_digest']
    common = {'model_id': profile['model']['model_id'],
              'model_version': profile['model']['model_version'],
              'input_digest': digest(frames),
              'parameter_digest': digest(profile['domain_models']),
              'seed_digest': seed}
    previous_error = {e: (0.0, 0.0) for e in scope['aircraft_ids']}
    lockout: dict[str, int] = defaultdict(int)
    ctrl_state = {e: {} for e in scope['aircraft_ids'] + scope['gcs_ids']}
    prev_sample = {}
    observations: list[dict] = []
    controls: list[dict] = []
    for frame in frames:
        tick = frame['tick']
        formal = tick % PARAMS['formal_step_ticks'] == 0
        entities = {e['entity_id']: e for e in frame['entities']}
        aircraft = [entities[e] for e in scope['aircraft_ids'] if e in entities]
        facilities = [entities[e] for e in scope['gcs_ids'] if e in entities]
        security_rows = domain._security_rows(
            inputs=FocusedDomainIdentity(episode), profile=profile, common=common, tick=tick,
            uavs=aircraft, facilities=facilities,
            communication_by_tick_entity=comm_index,
            lockout_duration=lockout if formal else copy.deepcopy(lockout))
        for row in security_rows:
            row['producer_source_refs'] = row['source_refs']
            row['source_refs'] = [f"raw/local_frames.jsonl#tick={tick}&entity={row['subject_id']}"]
            if row['subject_id'] in scope['aircraft_ids']:
                row['source_refs'].append(
                    f"raw/communication_state.jsonl#tick={tick}&entity={row['subject_id']}"
                    if formal else
                    f"raw/communication_ticks.jsonl#tick={tick}&entity={row['subject_id']}")
            row['sampling_scope'] = 'formal' if formal else 'probe_without_advancing_formal_memory'
        observations.extend(security_rows)
        for entity in aircraft:
            entity_id = entity['entity_id']
            values, quality, source = domain._gnss_values(
                uav=entity, uav_id=entity_id,
                position=entity['truth_pose']['position_enu_m'],
                previous_error_range=previous_error[entity_id],
                params=gnss_params, tick=tick, seed_digest=seed)
            next_error = values.pop('_next_error_range_m')
            if formal:
                previous_error[entity_id] = next_error
            values.update({'actor_id': entity_id,
                           'actor_ontology_class_id': 'world:UnmannedAircraft',
                           'gnss_service_id': gnss_params['service_instance']['service_id'],
                           'spoofing_signal_id': gnss_params['spoofing_signal_instance']['service_id']})
            observations.append({
                'tick': tick, 'subject_id': entity_id,
                'observation_family': 'gnss_navigation', 'values': values,
                'quality': quality, 'source_class': source,
                'model_id': profile['model']['model_id'],
                'model_version': profile['model']['model_version'],
                'seed_digest': seed,
                'sampling_scope': 'formal' if formal else 'probe_without_advancing_formal_memory',
                'source_refs': [f'raw/local_frames.jsonl#tick={tick}&entity={entity_id}']})
        security_by_subject = {r['subject_id']: r for r in security_rows}
        for entity_id in scope['aircraft_ids'] + scope['gcs_ids']:
            if entity_id not in entities:
                continue
            sample = control._sample_entity(tick, entities[entity_id])
            memory = ctrl_state[entity_id] if formal else copy.deepcopy(ctrl_state[entity_id])
            values = control._derive_values(
                sample, prev_sample.get(entity_id), memory,
                [comm_index[(tick, entity_id)]] if (tick, entity_id) in comm_index else [],
                [security_by_subject[entity_id]] if entity_id in security_by_subject else [])
            controls.append({
                'tick': tick, 'entity_id': entity_id, 'values': values,
                'model_id': 'control_response_state.'
                            'structured_truth_route_comm_kinematic_rth_v5',
                'sampling_scope': 'formal' if formal else 'probe_without_advancing_formal_memory',
                'source_refs': [f'raw/local_frames.jsonl#tick={tick}&entity={entity_id}']})
            if formal:
                prev_sample[entity_id] = sample
    return formal_comm, dense_comm, observations, controls, comm_index


def binding_universe(config: Mapping[str, Any]) -> list[dict]:
    """Exact typed binding universe from the governed predicate contracts."""
    scope = config['scope']
    result: list[dict] = []
    for predicate_id in config['predicate_ids']:
        template = TEMPLATES[predicate_id]
        roles = [r['key'] for r in template['argument_roles']]
        if roles in (['actor'], ['aircraft']):
            bindings = [{roles[0]: e} for e in scope['aircraft_ids']]
        elif set(roles) == {'actor', 'station'}:
            bindings = [{'actor': a, 'station': s}
                        for a in scope['aircraft_ids'] for s in scope['station_ids']]
        elif roles == ['station']:
            bindings = [{'station': e} for e in scope['gcs_ids']]
        elif roles == ['actor', 'service']:
            service_id = load(DOMAIN_PATH)['domain_models']['gnss']['service_instance']['service_id']
            bindings = [{'actor': e, 'service': service_id} for e in scope['aircraft_ids']]
        elif roles == ['actor', 'signal']:
            signal_id = load(DOMAIN_PATH)['domain_models']['gnss']['spoofing_signal_instance']['service_id']
            bindings = [{'actor': e, 'signal': signal_id} for e in scope['aircraft_ids']]
        elif roles == ['aircraft', 'home_pad']:
            bindings = [{'aircraft': e, 'home_pad': config['aircraft_geometry'][e]['home_pad_id']}
                        for e in scope['aircraft_ids']]
        elif roles == ['aircraft', 'landing_zone']:
            bindings = [{'aircraft': e, 'landing_zone': config['aircraft_geometry'][e]['home_pad_id']}
                        for e in scope['aircraft_ids']]
        elif roles == ['first_aircraft', 'second_aircraft']:
            bindings = [dict(zip(roles, pair))
                        for pair in itertools.combinations(scope['aircraft_ids'], 2)]
        else:
            raise ValueError(f'unsupported binding authority, not synthesized: {predicate_id}/{roles}')
        for binding in bindings:
            result.append({'predicate_id': predicate_id, 'bindings': binding,
                           'classes': {r['key']: 'world:' + r['class']
                                       for r in template['argument_roles']}})
    return result


def evaluate(config: Mapping[str, Any], trajectories: list[dict], weather: list[dict],
             roster_by_id: Mapping[str, Any]) -> dict:
    frames = local_frames(config, trajectories)
    formal_comm, dense_comm, observations, controls, comm_index = producer_observations(
        config, frames, weather)
    observation_index = {(r['tick'], r['subject_id'], r['observation_family']): r
                         for r in observations}
    control_index = {(r['tick'], r['entity_id']): r for r in controls}
    traj_index = {(r['tick'], r['entity_id']): r for r in trajectories}
    dense: list[dict] = []
    assertions: list[Assertion] = []
    observation_rows: list[ObservationRow] = []
    lineage: dict = {}
    semantic_episode = config['episode_id'] + '::' + config['group_key'] + '/' + config['arm_id']
    for tick in sorted({r['tick'] for r in trajectories}):
        for spec in binding_universe(config):
            predicate_id = spec['predicate_id']
            binding = spec['bindings']
            template = TEMPLATES[predicate_id]
            context: dict = {}
            in_scope = True
            extra_refs: list[str] = []
            entity_id = (binding.get('actor') or binding.get('aircraft')
                         or binding.get('station') or binding.get('first_aircraft'))
            row = traj_index.get((tick, entity_id))
            missing: list[str] = []
            if row is None:
                missing = ['raw/trajectories.jsonl#missing_tick_entity']
            if predicate_id.startswith('communication.'):
                if not comm_index:
                    missing.append('communication_state.producer_absent_no_authored_station')
                station = binding['station']
                link = comm_index.get((tick, entity_id))
                if link is None:
                    missing.append('communication_state.actor')
                elif station != link['station_id']:
                    in_scope = False
                else:
                    value = (comm._truth_from_link_degraded(link)
                             if predicate_id == 'communication.link_degraded'
                             else comm._truth_from_link_unavailable(link))
                    context = {'direct_predicate_value': value}
                    extra_refs = [f"raw/communication_state.jsonl#tick={tick}&entity={entity_id}"
                                  if tick % 5 == 0 else
                                  f"raw/communication_ticks.jsonl#tick={tick}&entity={entity_id}"]
            elif predicate_id.startswith('positioning.gnss') or \
                    predicate_id == 'positioning.navigation_error_exceeds_tolerance':
                gnss = observation_index.get((tick, entity_id, 'gnss_navigation'))
                if gnss is not None:
                    context = {'domain': {'gnss_navigation': gnss['values']}}
                extra_refs = [f'raw/domain_observations.jsonl#tick={tick}'
                              f'&subject={entity_id}&family=gnss_navigation']
            elif predicate_id.startswith('security.'):
                security = observation_index.get((tick, entity_id, 'security_command'))
                if security is not None:
                    context = {'domain': {'security_command': security['values']}}
                if 'actor' in binding and (tick, entity_id) in comm_index and \
                        binding['station'] != comm_index[(tick, entity_id)]['station_id']:
                    in_scope = False
                extra_refs = [f'raw/domain_observations.jsonl#tick={tick}'
                              f'&subject={entity_id}&family=security_command']
            elif predicate_id == 'positioning.aircraft_deviates_from_planned_route':
                if (tick, entity_id) in control_index:
                    context = {'navigation': {
                        'path_deviation': control_index[(tick, entity_id)]['values']['path_deviation']}}
                extra_refs = [f'raw/control_observations.jsonl#tick={tick}&entity={entity_id}']
            elif predicate_id.startswith('emergency.'):
                if (tick, entity_id) in control_index:
                    values = control_index[(tick, entity_id)]['values']
                    context = {'control': values,
                               'plan': {'rth_mode_active': values['rth_active']}}
                extra_refs = [f'raw/control_observations.jsonl#tick={tick}&entity={entity_id}']
            elif predicate_id == 'safety.aircraft_pair_below_separation':
                other = traj_index.get((tick, binding['second_aircraft']))
                if row and other:
                    context = {'derived': {
                        'nearest_aircraft_distance_m': math.dist(row['pos_enu'], other['pos_enu'])}}
                extra_refs = [f"raw/trajectories.jsonl#tick={tick}"
                              f"&entity={binding['second_aircraft']}"]
            elif predicate_id.startswith('airspace.') and row:
                geometry_spec = config['aircraft_geometry'][entity_id]
                pad = traj_index.get((tick, geometry_spec['home_pad_id']))
                geometry = {
                    'position_z_m': row['pos_enu'][2],
                    'assigned_altitude_m': geometry_spec['assigned_altitude_m'],
                    'local_ground_reference_z_m': geometry_spec['ground_reference_z_m'],
                    'z_agl_m': row['pos_enu'][2] - geometry_spec['ground_reference_z_m'],
                    'speed_mps': math.dist([0.0, 0.0, 0.0], row['vel_mps']),
                    'home_pad_id': geometry_spec['home_pad_id'],
                    'assigned_landing_zone_id': geometry_spec['home_pad_id']}
                if pad:
                    extra_refs.append(f'raw/trajectories.jsonl#tick={tick}&entity={geometry_spec["home_pad_id"]}')
                    distance = math.dist(row['pos_enu'][:2], pad['pos_enu'][:2])
                    geometry.update({'xy_distance_to_home_pad_m': distance,
                                     'xy_distance_to_assigned_landing_zone_m': distance})
                context = {'geometry': geometry}
            for field in template['state_contract']['required_fields']:
                if _missing_value(_value_at_path(context, field['field'])):
                    missing.append(field['field'])
            value = ('out_of_scope' if not in_scope
                     else 'unknown' if missing
                     else _evaluate_expression(template['evaluation_spec'], context, {}, PARAMS))
            if isinstance(value, bool):
                value = 'true' if value else 'false'
            if value is None:
                value = 'unknown'
            refs = [f'raw/trajectories.jsonl#tick={tick}&entity={entity_id}', *extra_refs]
            dense.append({'tick': tick, 'predicate_id': predicate_id, 'bindings': binding,
                          'value': value, 'context': context, 'missing_inputs': missing,
                          'source_refs': refs,
                          'sampling_scope': 'formal' if tick % 5 == 0 else 'diagnostic_off_grid'})
            if tick % PARAMS['formal_step_ticks']:
                continue
            # Participant tuple order follows the predicate template's declared
            # argument_roles order. The event rule layer compares this tuple
            # positionally against its own participant_roles, so the two must agree;
            # both come from the same governed registry.
            roles = [r['key'] for r in template['argument_roles']]
            tuple_type = 'grounded_roles__' + '__'.join(roles)
            tuple_id = 'arm_tuple:' + digest([semantic_episode, predicate_id, binding]).split(':')[1][:24]
            full_context = {**context, 'bindings': binding,
                            'binding_ontology_classes': spec['classes'],
                            'world_truth_value': value}
            observation_rows.append(ObservationRow(
                semantic_episode, tick, tuple_id, tuple_type,
                tuple(binding[r] for r in roles), in_scope, 'simulated_derived',
                tuple(refs), None, full_context))
            assertions.append(Assertion(
                semantic_episode, tick, tuple_id, tuple_type,
                tuple(binding[r] for r in roles), predicate_id, Truth(value),
                'simulated_derived', tuple(refs), None))
            lineage[(semantic_episode, tick, tuple_id, predicate_id)] = (digest(dense[-1]),
                                                                        digest(PARAMS))
    rules = [r for r in get_core_event_occurrence_types()
             if r['event_family_id'] in config['event_families']]
    transitions = build_transitions(assertions, PARAMS)
    events = build_events(observation_rows, assertions, transitions, rules,
                          roster_by_id=roster_by_id)
    outcomes = build_outcomes(events, assertions, PARAMS, rules, transitions=transitions)
    truth = _adapt_assertions(assertions, observation_rows, lineage)
    truth_map = {(r['episode_id'], r['tick'], r['tuple_id'], r['predicate_id']): r
                 for r in truth}
    converted, breaks = _adapt_transitions(transitions, truth_map)
    if breaks:
        raise ValueError('non-adjacent formal predicate samples')
    transition_map = {(r['episode_id'], r['to_tick'], r['tuple_id'], r['predicate_id'],
                       r['direction']): r for r in converted}
    occurrences = _adapt_occurrences(events, observation_rows, assertions, transition_map,
                                     rules, outcomes, roster_by_id=roster_by_id)
    adapted_outcomes = _adapt_outcomes(outcomes, occurrences, truth_map,
                                       formal_step=PARAMS['formal_step_ticks'])
    for collection in (truth, converted, occurrences):
        for record in collection:
            record['parent_episode_id'] = config['episode_id']
            record['arm_key'] = config['group_key'] + '/' + config['arm_id']
    return {
        'raw': {'local_frames': frames, 'communication_state': formal_comm,
                'communication_ticks': dense_comm, 'domain_observations': observations,
                'control_observations': controls},
        'objective': {'predicate_truth_ticks': dense, 'predicate_truth': truth,
                      'predicate_transitions': converted,
                      'event_occurrences': occurrences,
                      'event_outcomes': adapted_outcomes},
    }
