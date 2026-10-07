"""Consume indexed P09 energy sources into the existing frozen-baseline graph.

This producer does not run physics, change P01, or combine selected executions
with frozen semantic labels. The selected execution regime remains separate.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import shlex
import os
import sys
from collections import Counter
from pathlib import Path

import orjson

PREDICATE = 'facility.aircraft_battery_low'
FIELD = 'domain.payload_energy.state_of_charge_ratio'
TICKS = tuple(range(0, 901, 5))
RESOLVED_GAPS = frozenset({'global_entity_roster.entities[].activation_tick',
    'prior_energy_state_unresolved', 'pre_first_visible_energy_history_unavailable'})
ENERGY_FIELDS = ('state_of_charge_ratio', 'energy_consumed_ratio', 'energy_charged_ratio',
    'power_derating_ratio', 'predicted_range_m', 'temperature_energy_factor',
    'range_insufficient', 'planned_route_distance_m')


def read(path):
    return orjson.loads(path.read_bytes())


def json_rows(path):
    with path.open('rb') as handle:
        for line in handle:
            if line.strip():
                yield orjson.loads(line)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2) + b'\n')


def numeric(value):
    import math
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def merge_typed_energy(source, target, ledger, bindings):
    payload = {}; counts = Counter(); target.parent.mkdir(parents=True, exist_ok=True)
    pending_path = target.with_suffix(target.suffix + '.pending')
    with source.open('rb') as src, pending_path.open('wb') as dst:
        for line in src:
            if b'"payload_energy"' not in line:
                dst.write(line); continue
            row = orjson.loads(line)
            if row['observation_family'] != 'payload_energy':
                dst.write(line); continue
            key = (row['tick'], row['subject_id']); counts['payload_rows'] += 1
            old_soc = row['values']['state_of_charge_ratio']; indexed = ledger.get(key)
            if numeric(old_soc):
                counts['old_known_soc_preserved'] += 1
            elif indexed is not None and numeric(indexed['values'].get('state_of_charge_ratio')):
                values = row['values']; original = copy.deepcopy(values); actual = indexed['values']
                for field in ENERGY_FIELDS:
                    if field in actual:
                        value = actual[field]
                        values[field] = round(float(value), 6) if numeric(value) else value
                values['world_birth_episode_s'] = actual['exact_birth_episode_s']
                values['world_history_consumption_since_birth_ratio'] = round(actual['consumption_since_world_birth_ratio'], 6)
                values['energy_history_source'] = 'global_task_plan_episode_weather_and_complete_charging_plan'
                values['energy_history_source_gap'] = None
                values['battery_telemetry_measured'] = False
                unresolved = [gap for gap in values.get('missing_inputs', []) if gap not in RESOLVED_GAPS]
                values['missing_inputs'] = unresolved
                row['quality']['status'] = 'unknown' if unresolved else 'complete'
                if 'missing_inputs' in row['quality']:
                    row['quality']['missing_inputs'] = unresolved
                row['source_refs'] = list(dict.fromkeys(row['source_refs'] + indexed['source_refs']))
                if original['state_of_charge_ratio'] != values['state_of_charge_ratio']:
                    counts['new_known_soc_rows'] += 1
                bindings.append({'episode_id': row['episode_id'], 'regime': 'frozen_baseline',
                    'entity_id': row['subject_id'], 'task_id': indexed['task_id'], 'tick': row['tick'],
                    'time_s': row['tick'] / 10, 'state_of_charge_ratio': values['state_of_charge_ratio'],
                    'unit': 'ratio', 'status': 'computed_source_consumed_by_existing_typed_payload_energy',
                    'observation_id': row['observation_id'], 'source_refs': indexed['source_refs'],
                    'physical_source_kind': 'declared_model_computation_not_battery_telemetry'})
                line = orjson.dumps(row) + b'\n'
            else:
                counts['unresolved_soc_rows_preserved'] += 1
            payload[key] = row
            dst.write(line)
    missing = set(ledger) - set(payload)
    if missing:
        raise ValueError(f'{source}: {len(missing)} indexed energy rows lack an existing typed owner/time binding')
    pending_path.replace(target)
    return payload, dict(counts)


def apply_original_operation(states, operation, tick):
    kind = operation['operation']; body = operation.get('assertion', operation); key = body['tuple_id']
    if kind == 'add_predicate_assertion':
        states[key] = copy.deepcopy(body)
    elif kind == 'remove_predicate_assertion':
        del states[key]
    elif kind in ('set_predicate_value', 'refresh_predicate_evidence'):
        state = states[key]
        if kind == 'set_predicate_value':
            state['value'] = body['to_value']; state['truth_state_update_tick'] = tick
        for field in ('missing_source_record', 'observations', 'source_refs'):
            state[field] = copy.deepcopy(body[field])
        state['evidence_update_tick'] = tick
    else:
        raise ValueError('Unsupported existing world predicate operation: ' + kind)


def scoped_state(original, payload, tick, spec, defaults, world):
    state = copy.deepcopy(original); actor = state['bindings']['aircraft']
    row = payload.get((tick, actor))
    if row is None or not numeric(row['values'].get('state_of_charge_ratio')) or state['value'] == 'out_of_scope':
        return state
    soc = row['values']['state_of_charge_ratio']
    context = {'domain': {'payload_energy': {'state_of_charge_ratio': soc}}}
    state['value'] = world._expression_truth_value(world._evaluate_expression(spec, context, {}, defaults))
    source_ref = 'domain_state_observations.jsonl#observation=' + row['observation_id']
    state['observations'] = [{'path': FIELD, 'value': soc, 'source_ref': source_ref}]
    state['missing_source_record'] = []
    state['source_refs'] = sorted(set(original['source_refs']) | {source_ref} | set(row['source_refs']))
    state['truth_state_update_tick'] = tick; state['evidence_update_tick'] = tick
    return state


def state_operations(previous, current, tick):
    operations = []
    for key in sorted(previous.keys() - current.keys()):
        state = previous[key]
        operations.append({'operation': 'remove_predicate_assertion',
            **{field: state[field] for field in ('assertion_id', 'predicate_id', 'tuple_id', 'bindings', 'binding_ontology_classes')},
            'from_value': state['value']})
    for key in sorted(current.keys() - previous.keys()):
        operations.append({'operation': 'add_predicate_assertion', 'assertion': current[key]})
    for key in sorted(current.keys() & previous.keys()):
        old, new = previous[key], current[key]
        if old['bindings'] != new['bindings'] or old['binding_ontology_classes'] != new['binding_ontology_classes']:
            raise ValueError('Existing grounded battery tuple changed its ontology binding')
        common = {field: new[field] for field in ('assertion_id', 'predicate_id', 'tuple_id', 'bindings', 'binding_ontology_classes',
            'missing_source_record', 'observations', 'source_refs')}
        if old['value'] != new['value']:
            operations.append({'operation': 'set_predicate_value', **common, 'from_value': old['value'], 'to_value': new['value']})
        else:
            new['truth_state_update_tick'] = old['truth_state_update_tick']
            if any(old[field] != new[field] for field in ('missing_source_record', 'observations', 'source_refs')):
                operations.append({'operation': 'refresh_predicate_evidence', **common, 'value': new['value']})
            else:
                new['evidence_update_tick'] = old['evidence_update_tick']
    return operations


def update_world_graph(source, target, payload, template, world, stable_identifier):
    base = read(source / 'world_truth_graph_base.json'); defaults = base['governed_parameters']
    original_states = {state['tuple_id']: copy.deepcopy(state) for state in base['initial_assertions'] if state['predicate_id'] == PREDICATE}
    if not original_states:
        raise ValueError('Existing graph has no battery-low grounded candidates: ' + str(source))
    current = {key: scoped_state(state, payload, 0, template['evaluation_spec'], defaults, world) for key, state in original_states.items()}
    changed_initial = sum(current[key] != original_states[key] for key in current)
    base['initial_assertions'] = [current[state['tuple_id']] if state['predicate_id'] == PREDICATE else state for state in base['initial_assertions']]
    counts = Counter(state['value'] for state in base['initial_assertions'])
    if 'value_counts' in base['summary']:
        base['summary']['value_counts'] = dict(counts)
    base_target = target / 'world_truth_graph_base.json'
    base_pending = base_target.with_suffix(base_target.suffix + '.pending')
    write_json(base_pending, base)
    original_deltas = iter(json_rows(source / 'world_truth_graph_deltas.jsonl')); pending = next(original_deltas, None)
    previous = current; previous_delta_id = None; graph_counts = Counter()
    delta_target = target / 'world_truth_graph_deltas.jsonl'
    delta_pending = delta_target.with_suffix(delta_target.suffix + '.pending')
    with delta_pending.open('wb') as destination:
        for tick in TICKS[1:]:
            delta = None; other = []
            if pending is not None and pending['tick'] < tick:
                raise ValueError('Existing graph delta is outside the formal source clock')
            if pending is not None and pending['tick'] == tick:
                delta = pending
                for operation in delta['operations']:
                    body = operation.get('assertion', operation)
                    if body['predicate_id'] == PREDICATE:
                        apply_original_operation(original_states, operation, tick)
                    else:
                        other.append(operation)
                pending = next(original_deltas, None)
            current = {key: scoped_state(state, payload, tick, template['evaluation_spec'], defaults, world) for key, state in original_states.items()}
            scoped = state_operations(previous, current, tick); operations = other + scoped
            for operation in scoped:
                graph_counts[operation['operation']] += 1
            if operations:
                if delta is None:
                    delta = {'schema_name': world.DELTA_SCHEMA_NAME, 'schema_version': world.SCHEMA_VERSION,
                        'annotation_layer': 'L1', 'source_layer': 'L0', 'episode_id': base['episode_id'], 'tick': tick}
                delta.update(operations=operations, operation_count=len(operations), previous_delta_id=previous_delta_id,
                    delta_id=stable_identifier('world_truth_graph_delta', base['episode_id'], tick, operations))
                destination.write(orjson.dumps(delta) + b'\n'); previous_delta_id = delta['delta_id']; graph_counts['delta_rows'] += 1
            previous = current
    if pending is not None:
        raise ValueError('Existing graph extends outside the declared episode ticks')
    delta_pending.replace(delta_target)
    base_pending.replace(base_target)
    return {'initial_battery_assertions': len(current), 'initial_battery_evidence_updates': changed_initial,
        'scoped_operations': dict(graph_counts), 'other_predicate_operations_preserved': True,
        'predicate_id': PREDICATE, 'evaluation_spec': template['evaluation_spec'], 'battery_low_ratio': defaults['battery_low_ratio']}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--core-source-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--episode')
    args = parser.parse_args()
    project = args.project_root.resolve()
    core = (args.core_source_root or project / 'aw_data/domain_state_supplement').resolve()
    output = (args.output_dir or project / 'aw_data/objective_semantic_truth').resolve()
    sys.path.insert(0, str(project))
    from Dataset.semantic_truth import world_truth as world
    from Dataset.semantic_truth.provenance import stable_identifier
    registry = read(project / 'Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json')
    templates = [template for template in registry['templates'] if 'payload_energy' in orjson.dumps(template).decode()]
    if len(templates) != 1 or templates[0]['id'] != PREDICATE:
        raise ValueError('Current scoped energy predicate contract changed')
    dependencies = [rule for rule in registry['event_occurrence_types'] if PREDICATE in orjson.dumps(rule).decode()]
    if dependencies:
        raise ValueError('Current event families require native event regeneration: ' + ','.join(rule['rule_id'] for rule in dependencies))
    index_path = core / 'source_index.json'
    index = read(index_path)
    eligible = {entry['episode_id'] for entry in index['episodes']
                if entry['execution_regime'] == 'frozen_baseline'
                and entry['objective_semantic_truth']['matches_current_execution'] is True}
    if args.episode:
        if args.episode not in eligible:
            raise ValueError('Indexed baseline energy does not belong to the selected current execution: ' + args.episode)
        eligible = {args.episode}
    ledger = {}
    for row in json_rows(core / 'global_uav_energy.jsonl'):
        if row['episode_id'] not in eligible or row['origin'] != 'actual_current_episode_source_history_computation':
            continue
        key = (row['tick'], row['entity_id'])
        per_episode = ledger.setdefault(row['episode_id'], {})
        if key in per_episode:
            raise ValueError('Duplicate indexed energy owner/time')
        per_episode[key] = row
    entries = []; totals = Counter()
    for episode in sorted(eligible):
        source = project / 'aw_data/objective_semantic_truth' / episode
        target = output / episode
        target.mkdir(parents=True, exist_ok=True)
        bindings = []
        payload, typed_counts = merge_typed_energy(source / 'domain_state_observations.jsonl',
            target / 'domain_state_observations.jsonl', ledger.get(episode, {}), bindings)
        graph = update_world_graph(source, target, payload, templates[0], world, stable_identifier)
        events = source / 'event_occurrences.jsonl'
        if target != source:
            import shutil
            shutil.copyfile(events, target / events.name)
        entries.append({'episode_id': episode, 'regime': 'frozen_baseline',
            'typed_current_ref': str(target / 'domain_state_observations.jsonl'),
            'world_graph_base_ref': str(target / 'world_truth_graph_base.json'),
            'world_graph_deltas_ref': str(target / 'world_truth_graph_deltas.jsonl'),
            'events_ref': str(target / events.name), 'event_dependency_count': 0,
            'typed_counts': typed_counts, 'graph': graph})
        totals.update(typed_counts)
        print('Scoped energy graph updated', episode, typed_counts.get('new_known_soc_rows', 0), flush=True)
    index['energy_semantic_current'] = dict(semantic_truth_root=str(output),
        baseline_episodes=entries, current_selected_execution_graphs_updated=False,
        reproduction_command=shlex.join([sys.executable, '-B', '-m', 'Dataset.semantic_truth.p09_energy_graph_update',
            '--project-root', str(project), '--core-source-root', str(core), '--output-dir', str(output)]
            + (['--episode', args.episode] if args.episode else [])))
    index['typed_truth_producer'].update(
        requires_regeneration_before_P01_receives_repaired_typed_truth=False,
        repaired_scope='indexed_matching_frozen_baseline_payload_energy_and_battery_low_predicate',
        repaired_episode_ids=[entry['episode_id'] for entry in entries],
        current_selected_execution_requires_separate_objective_generation=True,
        all_core_labels_repaired=False)
    write_json(index_path, index)
    print(orjson.dumps({'baseline_episode_count': len(entries), 'typed_counts': dict(totals)}).decode(), flush=True)


if __name__ == '__main__':
    main()
