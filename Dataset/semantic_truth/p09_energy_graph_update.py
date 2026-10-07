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
    if target.is_symlink():
        target.unlink()
    with source.open('rb') as src, target.open('wb') as dst:
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
    write_json(target / 'world_truth_graph_base.json', base)
    original_deltas = iter(json_rows(source / 'world_truth_graph_deltas.jsonl')); pending = next(original_deltas, None)
    previous = current; previous_delta_id = None; graph_counts = Counter()
    with (target / 'world_truth_graph_deltas.jsonl').open('wb') as destination:
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
    return {'initial_battery_assertions': len(current), 'initial_battery_evidence_updates': changed_initial,
        'scoped_operations': dict(graph_counts), 'other_predicate_operations_preserved': True,
        'predicate_id': PREDICATE, 'evaluation_spec': template['evaluation_spec'], 'battery_low_ratio': defaults['battery_low_ratio']}


def verify_P01_consumption(args, output, bindings):
    """Run the actual P01 entry and verify every newly computed tick-zero owner."""
    sys.path.insert(0, str(args.p01_code_root.resolve()))
    specification = importlib.util.spec_from_file_location("p09_current_P01_import", args.p01_import_source.resolve())
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    episode, report = module.import_episode(args.project_root / "aw_data/render_ready_episodes_capture_filtered",
        output / "frozen_baseline", None, "L1-1_v1__seed00", None, None)
    if report.errors:
        raise ValueError(report.errors)
    battery_records = [record for record in episode.records if record["record_kind"] == "edge"
                       and record["fields"]["relation.predicate_id"] == PREDICATE]
    with (output / "p01_battery_edges.jsonl").open("wb") as handle:
        for record in battery_records:
            handle.write(orjson.dumps(record) + b"\n")
    groups = {}
    for binding in bindings:
        if binding["tick"] == 0:
            groups.setdefault(binding["episode_id"], []).append(binding)
    consumed = []
    for episode_id, requested in sorted(groups.items()):
        canonical = module.CanonicalEpisode(episode_id)
        native_report = module.ImportReport(episode_id=episode_id)
        base_path = output / "frozen_baseline" / episode_id / "world_truth_graph_base.json"
        module.import_truth_base(canonical, base_path, native_report)
        records = {record["fields"]["relation.bindings"]["aircraft"]: record
                   for record in canonical.records if record["fields"]["relation.predicate_id"] == PREDICATE}
        for binding in requested:
            record = records[binding["entity_id"]]
            if record["tick"] != 0 or record["source_family"] != "world_truth_graph_base":
                raise ValueError("P01 initial state lost its native owner/time/source binding")
            observations = record["fields"]["relation.evidence"]["observations"]
            soc = [observation["value"] for observation in observations if observation["path"] == FIELD]
            if soc != [binding["state_of_charge_ratio"]] or record["fields"]["relation.value"] == "unknown":
                raise ValueError("P01 initial battery state differs from its computed typed SOC")
            consumed.append({**binding, "tuple_id": record["fields"]["relation.tuple_id"],
                "label_value": record["fields"]["relation.value"], "P01_initial_record_id": record["id"],
                "native_source": record["source"], "status": "native_initial_assertion_consumed_by_existing_P01_edge_fields"})
    with (output / "p01_initial_energy_bindings.jsonl").open("wb") as handle:
        for record in consumed:
            handle.write(orjson.dumps(record) + b"\n")
    initial_battery = [record for record in battery_records if record["tick"] == 0]
    result = {"episode_id": "L1-1_v1__seed00", "existing_P01_import_actually_executed": True,
        "P01_import_source_ref": str(args.p01_import_source), "source_files": report.sources,
        "canonical_record_counts": dict(report.canonical_records), "battery_edge_records": len(battery_records),
        "initial_assertions_imported": True, "tick0_battery_edges": len(initial_battery),
        "numeric_SOC_evidence_tick0_battery_edges": sum(any(item["path"] == FIELD for item in record["fields"]["relation.evidence"].get("observations", [])) for record in initial_battery),
        "new_SOC_tick0_bindings_consumed": len(consumed), "new_SOC_tick0_episode_count": len(groups),
        "new_SOC_tick0_affected_episode_ids": sorted(groups),
        "record_ref": str(output / "p01_battery_edges.jsonl"),
        "new_initial_source_bindings_ref": str(output / "p01_initial_energy_bindings.jsonl"),
        "P01_field_schema_changed": False, "cohort_split_sampling_loss_changed": False}
    write_json(output / "p01_consumer_current.json", result)
    return result


def finalize_P01_consumer_index(args, output, consumption):
    summary = read(output / "source_index.json")
    summary["P01_existing_consumer_entry"].update(
        fields=["truth_frames", "roster", "world_truth_graph_base.initial_assertions", "world_truth_graph_deltas", "event_occurrences"],
        P01_code_changed=True, P01_field_schema_changed=False,
        native_initial_assertions_imported=True, P01_import_actually_executed=True,
        P01_import_source_ref=str(args.p01_import_source),
        actual_consumer_result_ref=str(output / "p01_consumer_current.json"),
        new_energy_side_table_directly_imported=False)
    summary["P01_consumer_verification"] = {key: consumption[key] for key in
        ("new_SOC_tick0_bindings_consumed", "new_SOC_tick0_episode_count", "canonical_record_counts", "tick0_battery_edges")}
    summary["reproduction_command"] = shlex.join([sys.executable, "-B", "-m", "Dataset.semantic_truth.p09_energy_graph_update",
        "--project-root", str(args.project_root), "--release-code-root", str(args.release_code_root),
        "--core-source-root", str(args.core_source_root), "--ue-input-root", str(args.ue_input_root), "--output-dir", str(output),
        "--p01-code-root", str(args.p01_code_root), "--p01-import-source", str(args.p01_import_source)])
    write_json(output / "source_index.json", summary)
    core_index = read(args.core_source_root / "source_index.json")
    core_index["energy_semantic_current"] = {"source_index_ref": str(output / "source_index.json"),
        "regime": "frozen_baseline", "typed_and_existing_battery_predicate_graphs_written": True,
        "P01_existing_import_fields_compatible": True, "P01_import_actually_executed": True,
        "native_initial_assertions_imported": True, "actual_consumer_result_ref": str(output / "p01_consumer_current.json"),
        "new_tick0_SOC_owner_bindings_consumed": consumption["new_SOC_tick0_bindings_consumed"],
        "current_selected_execution_graphs_updated": False}
    core_index["p01_import_contract_changed"] = True
    core_index["current_p01_import"].update(
        reads=["truth_frames.jsonl", "global_entity_roster.json", "world_truth_graph_base.json.initial_assertions", "world_truth_graph_deltas.jsonl", "event_occurrences.jsonl"],
        native_initial_assertions_consumed=True,
        actual_consumer_result_ref=str(output / "p01_consumer_current.json"),
        computed_energy_ledger_consumed_by_scoped_baseline_producer=True,
        new_battery_labels_in_actual_imported_derivative=True,
        new_core_side_tables_consumed=False, new_core_labels_in_current_p01_head=False)
    core_index["typed_truth_producer"].update(
        requires_regeneration_before_P01_receives_repaired_typed_truth=False,
        repaired_scope="computed_frozen_baseline_payload_energy_and_existing_battery_low_predicate_only",
        current_selected_execution_requires_separate_objective_generation=True,
        all_core_labels_repaired=False)
    write_json(args.core_source_root / "source_index.json", core_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--release-code-root', type=Path, required=True)
    parser.add_argument('--core-source-root', type=Path, required=True)
    parser.add_argument('--ue-input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--p01-code-root', type=Path, required=True)
    parser.add_argument('--p01-import-source', type=Path, required=True)
    args = parser.parse_args(); project = args.project_root.resolve(); output = args.output_dir.resolve()
    sys.path.insert(0, str(args.release_code_root.resolve()))
    from Dataset.semantic_truth import world_truth as world
    from Dataset.semantic_truth.provenance import stable_identifier
    print('Scoped energy graph producer PID', os.getpid(), flush=True)
    registry = read(project / 'Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json')
    templates = [template for template in registry['templates'] if 'payload_energy' in orjson.dumps(template).decode()]
    if len(templates) != 1 or templates[0]['id'] != PREDICATE:
        raise ValueError('Current scoped energy predicate contract changed')
    dependencies = [rule for rule in registry['event_occurrence_types'] if PREDICATE in orjson.dumps(rule).decode()]
    if dependencies:
        raise ValueError('Current event families require native event regeneration: ' + ','.join(rule['rule_id'] for rule in dependencies))
    ledger = {}; ledger_rows = 0
    for row in json_rows(args.core_source_root / 'global_uav_energy.jsonl'):
        if row['origin'] != 'actual_current_episode_source_history_computation':
            continue
        key = (row['tick'], row['entity_id']); per_episode = ledger.setdefault(row['episode_id'], {})
        if key in per_episode:
            raise ValueError('Duplicate indexed energy owner/time')
        per_episode[key] = row; ledger_rows += 1
    baseline = project / 'aw_data/objective_semantic_truth'
    render = project / 'aw_data/render_ready_episodes_capture_filtered'
    for immutable in (baseline, render):
        if output == immutable or output in immutable.parents or immutable in output.parents:
            raise ValueError('Output overlaps immutable episode source: ' + str(immutable))
    episodes = sorted(path.name for path in render.iterdir() if (path / 'episode_manifest.json').is_file())
    if len(episodes) != 210:
        raise ValueError('Frozen baseline episode coverage differs from 210')
    entries = []; bindings = []; totals = Counter()
    for episode in episodes:
        source = baseline / episode; target = output / 'frozen_baseline' / episode
        target.mkdir(parents=True, exist_ok=True)
        payload, typed_counts = merge_typed_energy(source / 'domain_state_observations.jsonl', target / 'domain_state_observations.jsonl', ledger.get(episode, {}), bindings)
        graph = update_world_graph(source, target, payload, templates[0], world, stable_identifier)
        if typed_counts.get('new_known_soc_rows', 0) == 0:
            (target / 'domain_state_observations.jsonl').unlink()
            (target / 'domain_state_observations.jsonl').symlink_to(source / 'domain_state_observations.jsonl')
        event_path = target / 'event_occurrences.jsonl'
        if event_path.exists() or event_path.is_symlink():
            event_path.unlink()
        event_path.symlink_to(source / 'event_occurrences.jsonl')
        entry = {'episode_id': episode, 'regime': 'frozen_baseline', 'status': 'scoped_typed_energy_and_battery_predicate_updated',
            'render_ready_source_ref': str(render / episode), 'semantic_baseline_source_ref': str(source),
            'semantic_truth_root': str(output / 'frozen_baseline'),
            'typed_current_ref': str(target / 'domain_state_observations.jsonl'),
            'world_graph_base_ref': str(target / 'world_truth_graph_base.json'),
            'world_graph_deltas_ref': str(target / 'world_truth_graph_deltas.jsonl'),
            'events_ref': str(target / 'event_occurrences.jsonl'), 'event_dependency_count': 0,
            'events_preserved_byte_for_byte': True, 'typed_counts': typed_counts, 'graph': graph}
        entries.append(entry); totals.update(typed_counts)
        print('Scoped energy graph updated', episode, typed_counts.get('new_known_soc_rows', 0), flush=True)
    ue_manifest = read(args.ue_input_root / 'source_status_manifest.json')
    selected = []
    for entry in ue_manifest['episodes']:
        if 'historical_v14_source' in entry:
            selected.append({'episode_id': entry['episode_id'], 'regime': 'current_selected_v14_execution',
                'ue_input_ref': str(args.ue_input_root / entry['input_dir']), 'semantic_pose_binding_status': 'unverified',
                'frozen_baseline_graph_not_claimed_current_execution': True})
        elif 'actual_l6_receipt_story' in entry:
            selected.append({'episode_id': entry['episode_id'], 'regime': 'current_actual_L6_R2_execution',
                'ue_input_ref': str(args.ue_input_root / entry['input_dir']),
                'semantic_pose_binding_status': 'requires_separate_actual_objective_result',
                'frozen_baseline_graph_not_claimed_current_execution': True})
    consumption = verify_P01_consumption(args, output, bindings)
    summary = {'schema_version': 'p09.energy-semantic-current/v1', 'producer': 'Dataset/semantic_truth/p09_energy_graph_update.py',
        'baseline_episode_count': len(entries), 'indexed_computed_energy_rows': ledger_rows, 'typed_counts': dict(totals),
        'current_selected_execution_regimes': selected, 'baseline_episodes': entries,
        'P01_existing_consumer_entry': {'render_ready_root': str(render), 'semantic_truth_root': str(output / 'frozen_baseline'),
            'fields': ['truth_frames', 'roster', 'world_truth_graph_base.initial_assertions', 'world_truth_graph_deltas', 'event_occurrences'],
            'P01_code_changed': True, 'P01_field_schema_changed': False, 'native_initial_assertions_imported': True,
            'actual_consumer_result_ref': str(output / 'p01_consumer_current.json'),
            'new_energy_side_table_directly_imported': False},
        'pre_episode_source_gaps_preserved_ref': str(args.core_source_root / 'core_field_status.jsonl'),
        'computed_source_bindings_ref': str(output / 'computed_energy_source_bindings.jsonl'),
        'current_selected_execution_graphs_updated_by_this_operation': False,
        'reproduction_command': ' '.join([sys.executable, '-B', '-m', 'Dataset.semantic_truth.p09_energy_graph_update',
            '--project-root', str(project), '--release-code-root', str(args.release_code_root),
            '--core-source-root', str(args.core_source_root), '--ue-input-root', str(args.ue_input_root), '--output-dir', str(output),
            '--p01-code-root', str(args.p01_code_root), '--p01-import-source', str(args.p01_import_source)])}
    with (output / 'computed_energy_source_bindings.jsonl').open('wb') as handle:
        for binding in bindings:
            handle.write(orjson.dumps(binding) + b'\n')
    write_json(output / 'source_index.json', summary)
    finalize_P01_consumer_index(args, output, consumption)
    print(orjson.dumps({'baseline_episode_count': len(entries), 'indexed_computed_energy_rows': ledger_rows, 'typed_counts': dict(totals)}).decode(), flush=True)


if __name__ == '__main__':
    main()
