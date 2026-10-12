"""Script-editing and measured-state primitives for individually authored X arms.

This module does not select experiments, infer descendants, or choose outcomes.
Every edit is supplied by the dedicated arm implementation.
"""
from __future__ import annotations

import copy
import math


def edit_script(source, *, remove_events=(), append_actions=None,
                replace_actions=None, add_events=(), add_triggers=(),
                replace_triggers=None):
    """Apply exact, explicit edits and remove unreachable trigger definitions."""
    script = copy.deepcopy(source)
    removed = set(remove_events)
    events = {event['event_id']: event for event in script['events']}
    if len(events) != len(script['events']) or not removed <= events.keys():
        raise ValueError('Duplicate source events or an unknown explicit removal')
    script['events'] = [event for event in script['events']
                        if event['event_id'] not in removed]
    retained = {event['event_id']: event for event in script['events']}
    for event_id, actions in ({} if append_actions is None else append_actions).items():
        retained[event_id]['actions'].extend(copy.deepcopy(actions))
    replacements = {} if replace_actions is None else replace_actions
    replaced = set()
    for event in script['events']:
        for index, action in enumerate(event['actions']):
            action_id = action['action_id']
            if action_id in replacements:
                replacement = copy.deepcopy(replacements[action_id])
                if replacement['action_id'] != action_id:
                    raise ValueError('Replacement must preserve the declared action identity')
                event['actions'][index] = replacement
                replaced.add(action_id)
    if replaced != replacements.keys():
        raise ValueError('A replacement action is absent from the retained script')
    script['events'].extend(copy.deepcopy(add_events))
    script['triggers'].extend(copy.deepcopy(add_triggers))
    triggers = {trigger['trigger_id']: trigger for trigger in script['triggers']}
    if len(triggers) != len(script['triggers']):
        raise ValueError('Duplicate trigger identity')
    for trigger_id, replacement in ({} if replace_triggers is None else replace_triggers).items():
        if trigger_id not in triggers or replacement['trigger_id'] != trigger_id:
            raise ValueError('Unknown or renamed replacement trigger')
        triggers[trigger_id] = copy.deepcopy(replacement)
    event_ids = [event['event_id'] for event in script['events']]
    action_ids = [action['action_id'] for event in script['events'] for action in event['actions']]
    if len(set(event_ids)) != len(event_ids) or len(set(action_ids)) != len(action_ids):
        raise ValueError('Duplicate branch event/action identity')
    needed = set()

    def visit(trigger_id, ancestors):
        if trigger_id in ancestors:
            raise ValueError('Cyclic trigger definition')
        if trigger_id in needed:
            return
        trigger = triggers[trigger_id]
        if trigger['type'] in ('event_fired', 'event_fired_after'):
            if trigger['event_id'] not in event_ids:
                raise ValueError('A retained trigger references an explicitly removed event')
        for child in trigger.get('children', []):
            visit(child, ancestors | {trigger_id})
        needed.add(trigger_id)

    for event in script['events']:
        visit(event['trigger_ref'], set())
    script['triggers'] = [triggers[trigger['trigger_id']] for trigger in script['triggers']
                          if trigger['trigger_id'] in needed]
    return script


def require(condition, message):
    if not condition:
        raise ValueError(message)


def samples(run, entity_id, start_tick, end_tick):
    values = [row for row in run['engine'].trajectory_rows
              if row['entity_id'] == entity_id and start_tick <= row['tick'] <= end_tick]
    require([row['tick'] for row in values] == list(range(start_tick, end_tick + 1)),
            f'Incomplete measured interval for {entity_id}: {start_tick}..{end_tick}')
    return values


def event_tick(run, event_id):
    state = run['interpreter'].event_states[event_id]
    require(state.fired, f'Required event did not fire: {event_id}')
    return state.last_fired_tick


def absent_event(run, event_id):
    states = run['interpreter'].event_states
    require(event_id not in states or not states[event_id].fired,
            f'Forbidden event fired: {event_id}')


def field_values(run, entity_id, family, field, start_tick, end_tick):
    return [row[family][field] for row in samples(run, entity_id, start_tick, end_tick)]


def path_length(run, entity_id, start_tick, end_tick):
    values = samples(run, entity_id, start_tick, end_tick)
    return sum(math.dist(before['pos_enu'], after['pos_enu'])
               for before, after in zip(values, values[1:]))


def require_fields(run, entity_id, expected, start_tick, end_tick):
    """Check explicitly named nested runtime fields over a complete interval."""
    values = samples(run, entity_id, start_tick, end_tick)
    for row in values:
        for path, wanted in expected.items():
            value = row
            for key in path.split('.'):
                value = value[key]
            require(value == wanted,
                    f'{entity_id} {path} at {row["tick"]}: {value!r} != {wanted!r}')
    return {'entity_id': entity_id, 'start_tick': start_tick, 'end_tick': end_tick,
            'samples': len(values), 'observed_fields': copy.deepcopy(expected)}


def completed_move(run, action_id, end_tick):
    dispatched = [item for item in run['audit'] if item['action']['action_id'] == action_id]
    require(len(dispatched) == 1, f'Expected one dispatched move: {action_id}')
    item = dispatched[0]
    action = item['action']
    require(action['type'] == 'move_entity' and item['result']['status'] == 'ok',
            f'Move was not accepted: {action_id}')
    entity_id = action['entity_id']
    preemptions = [other['tick'] for other in run['audit']
                   if other['tick'] > item['tick'] and
                   other['action'].get('entity_id') == entity_id and
                   other['action']['type'] in ('move_entity', 'set_pedestrian_activity')]
    last = min(end_tick, min(preemptions) - 1) if preemptions else end_tick
    target = action['waypoints_enu_m'][-1]
    hits = [row['tick'] for row in samples(run, entity_id, item['tick'] + 1, last)
            if math.dist(row['pos_enu'], target) < 1e-5]
    require(bool(hits), f'No measured target arrival before end/preemption: {action_id}')
    return {'action_id': action_id, 'dispatch_tick': item['tick'],
            'target_tick': min(hits), 'target_enu_m': target}


def measured_pair(run, first, second, start_tick, end_tick):
    a = samples(run, first, start_tick, end_tick)
    b = samples(run, second, start_tick, end_tick)
    distances = [math.dist(x['pos_enu'], y['pos_enu']) for x, y in zip(a, b)]
    return {'minimum_distance_m': min(distances), 'maximum_distance_m': max(distances),
            'minimum_tick': start_tick + distances.index(min(distances))}


def run_entry(entry_path):
    from Dataset.tools.x_arm_pipeline import run_arm, ROOT
    from pathlib import Path
    relative = Path(entry_path).resolve().relative_to(ROOT / 'scripts/arms')
    return run_arm(ROOT / 'interventions' / relative.with_suffix('.yaml'))
