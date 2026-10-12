"""Window projection with inherited lifecycle evidence and measured motion results."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict

from Dataset.tools.l6_v2.common import ROOT, RUNTIME_FAMILIES, file_digest


def window_semantics(spec, computed, trajectories, weather, *, validate_objective,
                     created_history_paths):
    start, end = spec.window_start_tick, spec.window_end_tick
    full = computed['objective']
    outcomes = {r['event_id']: r for r in full['event_outcomes']}
    events = [copy.deepcopy(r) for r in full['event_occurrences']
              if r['trigger_tick'] <= end and (outcomes[r['event_id']]['terminal_tick'] is None
                                              or outcomes[r['event_id']]['terminal_tick'] >= start)]
    event_ids = {r['event_id'] for r in events}
    history = {
        'schema': 'l6_measured_pre_window_evidence',
        'episode_id': spec.episode_id, 'arm_key': spec.group_key + '/' + spec.arm_id,
        'end_exclusive_tick': start,
        'computation': 'continuous execution and producer memory from tick zero',
        'raw': {k: [r for r in v if r['tick'] < start] for k, v in computed['raw'].items()},
        'objective': {k: [r for r in full[k] if r['to_tick' if k == 'predicate_transitions' else 'tick'] < start]
                      for k in ('predicate_truth', 'predicate_transitions')},
    }
    history['raw']['trajectories'] = [r for r in trajectories if r['tick'] < start]
    history['raw']['weather'] = [r for r in weather if r['tick'] < start]
    payload = json.dumps(history, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    compressed = gzip.compress(payload, mtime=0)
    sha = hashlib.sha256(compressed).hexdigest()
    path = ROOT / 'design' / spec.group_key / 'history' / (spec.arm_id + '_' + sha[:16] + '.json.gz')
    ref = {'path': path.relative_to(ROOT).as_posix(),
           'format': 'gzip_json', 'scope': 'pre_window_measured_evidence'}
    objective = {k: [copy.deepcopy(r) for r in full[k] if start <= r['tick'] <= end]
                 for k in ('predicate_truth_ticks', 'predicate_truth')}
    objective['predicate_transitions'] = [copy.deepcopy(r) for r in full['predicate_transitions']
                                        if start <= r['to_tick'] <= end]
    objective['event_occurrences'] = events
    objective['event_outcomes'] = [copy.deepcopy(r) for r in full['event_outcomes'] if r['event_id'] in event_ids]
    from Dataset.semantic_truth.window_outcome import arm_window_outcome
    objective['event_outcomes'] = [
        arm_window_outcome(row, start_tick=start, end_tick=end)
        for row in objective['event_outcomes']]
    needed = {p['transition_id'] for e in events for p in e['event_phases']}
    inherited_transitions = [r for r in history['objective']['predicate_transitions'] if r['transition_id'] in needed]
    initial = {'tick': start, 'policy': 'inherit_original_onset_and_producer_memory', 'source': ref,
               'events': [{'event_id': r['event_id'], 'onset_tick': r['trigger_tick'],
                           'bindings': r['bindings'], 'event_family_id': r['event_family_id']}
                          for r in events if r['trigger_tick'] < start],
               'transitions': inherited_transitions}

    def rewrite(value):
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if isinstance(value, str) and value.startswith('raw/'):
            match = re.search(r'#tick=(\d+)', value)
            if match and int(match.group(1)) < start:
                return ref['path'] + '#' + value
        return value
    projected_objective = rewrite(objective)
    projected_initial = rewrite(initial)
    validation_result = validate_objective(projected_objective)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != compressed:
        raise ValueError('history content address collision')
    if not path.exists():
        created_history_paths.append(path)
        path.write_bytes(compressed)
    return projected_objective, ref, projected_initial, validation_result


def _intervals(rows, getter):
    result = []
    for row in rows:
        value = getter(row)
        if result and result[-1]['value'] == value and result[-1]['end_tick'] + 1 == row['tick']:
            result[-1]['end_tick'] = row['tick']
        else:
            result.append({'start_tick': row['tick'], 'end_tick': row['tick'], 'value': value})
    return result


def physical_observations(spec, run, scope, trajectories, geometry):
    start, end = spec.window_start_tick, spec.window_end_tick
    by_entity = defaultdict(list)
    for row in trajectories:
        by_entity[row['entity_id']].append(row)
    entities = []
    for entity_id in scope['aircraft_ids'] + scope['station_ids'] + scope['gcs_ids']:
        observed = [r for r in by_entity[entity_id] if start <= r['tick'] <= end]
        if len(observed) != end - start + 1:
            raise ValueError('incomplete physical observation ' + entity_id)
        record = {'entity_id': entity_id, 'initial_position': observed[0]['pos_enu'],
                  'final_position': observed[-1]['pos_enu'],
                  'path_length_m': sum(math.dist(a['pos_enu'], b['pos_enu']) for a,b in zip(observed, observed[1:])),
                  'max_speed_mps': max(math.dist([0,0,0], r['vel_mps']) for r in observed),
                  'motion_intervals': _intervals(observed, lambda r: math.dist([0,0,0], r['vel_mps']) > 1e-9),
                  'activity_intervals': _intervals(observed, lambda r: r['activity_type']),
                  'runtime_intervals': _intervals(observed, lambda r: {k:r[k] for k in RUNTIME_FAMILIES if k in r})}
        if entity_id in geometry:
            pad = by_entity[geometry[entity_id]['home_pad_id']][-1]
            record['final_home_horizontal_distance_m'] = math.dist(observed[-1]['pos_enu'][:2], pad['pos_enu'][:2])
        entities.append(record)
    movement = []
    actions = run['audit']
    for index, item in enumerate(actions):
        action, tick = item['action'], item['tick']
        if action['type'] != 'move_entity' or item['result']['status'] != 'ok' or tick > end:
            continue
        entity_id = item['result']['entity_id']
        later = [r for r in actions[index+1:] if r['result']['status'] == 'ok'
                 and r['result'].get('entity_id') == entity_id
                 and (r['action']['type'] == 'move_entity' or
                      (r['action']['type'] == 'set_visual_state' and r['result']['mode'] in ('hold','hover')))]
        next_tick = later[0]['tick'] if later else end + 1
        if next_tick < start:
            continue
        effect = item['motion_schedule']
        terminal = effect['terminal_tick']
        if terminal < start:
            continue
        endpoint = effect['endpoint_enu_m']
        samples = [r for r in by_entity[entity_id] if tick < r['tick'] <= min(next_tick, end)]
        reached = [r['tick'] for r in samples if math.dist(r['pos_enu'], endpoint) < 1e-6 and r['tick'] >= terminal]
        status = ('completed' if reached else 'preempted' if next_tick <= end else 'right_censored')
        movement.append({'action_id': action['action_id'], 'entity_id': entity_id, 'dispatch_tick': tick,
                         'origin': item['origin'], 'status': status,
                         'scheduled_terminal_tick': terminal, 'endpoint_enu_m': endpoint,
                         'observed_terminal_tick': reached[0] if reached else None,
                         'next_motion_override_tick': next_tick if later else None,
                         'first_motion_tick': next((r['tick'] for r in samples if math.dist([0,0,0],r['vel_mps']) > 1e-9), None),
                         'max_observed_speed_mps': max((math.dist([0,0,0],r['vel_mps']) for r in samples), default=0.0)})
    return {'window': spec.window(), 'source': 'executed_actions_and_sampled_trajectory',
            'entities': entities, 'movement_actions': movement,
            'limits': ['Piecewise linear kinematics; no acceleration model.',
                       'State labels and physical location are reported separately.']}
