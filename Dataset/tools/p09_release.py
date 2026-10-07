"""Assemble current P09 UE inputs and source-grounded weather replacements.

This entrypoint serializes recorded inputs. It runs no simulator, model,
objective builder, or UE capture and leaves frozen aw_data and arms unchanged.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import csv
import math
import os
import re
import shlex
import shutil
import sys
import time
import types
from collections import Counter
from pathlib import Path

import orjson

_render_source_arg = sys.argv.index('--render-source-root') if '--render-source-root' in sys.argv else None
_module_source_root = Path(sys.argv[_render_source_arg + 1]) if _render_source_arg is not None else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_module_source_root / 'Dataset/tools'))
from Dataset.tools import convert_to_render_ready as conv
from Dataset.tools.filter_render_ready_truth_for_capture import update_frame_summaries
from Dataset.tools.runtime_state_contract import RUNTIME_STATE_FIELDS

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path('/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09')
BASE = ROOT / 'aw_data/render_ready_episodes_capture_filtered'
L6_ID = 'L6-5_v1__seed00'
TARGETS = {'uav_digital_l6_5_v1', 'gcs_anchor_l6_5_v1'}
DENSE_TICKS = list(range(901))
CAPTURE_TICKS = list(range(0, 901, 5))
INPUT_FILES = ('truth_frames.jsonl', 'trajectories.jsonl', 'event_trace.jsonl',
    'event_realization.jsonl', 'dynamic_labels.jsonl', 'weather_meta.jsonl',
    'scenario_plan.json', 'scenario_package.json', 'episode_manifest.json',
    'global_entity_roster.json', 'scene_occupancy_manifest.json', 'render_host_config.json',
    'scene_setup.json', 'event_script.json', 'capture_window.json', 'multimodal_window_mask.jsonl')
CLOSURE_DIR = ROOT / 'design/p09/mechanism_completion_v1/numeric_mapping_checkpoint/builder_short_session_v1/author_formal_objective_checkpoint_v8'


def read(path):
    return orjson.loads(Path(path).read_bytes())


def rows(path):
    with Path(path).open('rb') as handle:
        for line in handle:
            if line.strip():
                yield orjson.loads(line)


def clean(value):
    """Remove obsolete identity checks and event-firing-as-science claims."""
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()
                if not any(word in key.lower() for word in ('sha256', 'digest', 'hash'))
                and key != 'scientific_story_passed'}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(clean(value), option=orjson.OPT_INDENT_2) + b'\n')


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        for value in values:
            handle.write(orjson.dumps(clean(value)) + b'\n')


def relocate(value, source_dir, relative_dir, v14):
    if isinstance(value, dict):
        return {key: relocate(item, source_dir, relative_dir, v14) for key, item in clean(value).items()}
    if isinstance(value, list):
        return [relocate(item, source_dir, relative_dir, v14) for item in value]
    if isinstance(value, str):
        prefixes = [str(source_dir), 'Dataset/render_ready_episodes_capture_filtered/' + source_dir.name,
                    'aw_data/render_ready_episodes_capture_filtered/' + source_dir.name]
        if source_dir.is_relative_to(ROOT):
            prefixes.append(str(source_dir.relative_to(ROOT)))
        for prefix in prefixes:
            if value == prefix or value.startswith(prefix + '/'):
                return relative_dir + value[len(prefix):]
        if value.startswith('sources/'):
            return str(v14 / value)
    return value


def copy_inputs(source, target, v14):
    """Keep recorded numeric streams; rewrite only metadata/source references."""
    target.mkdir(parents=True, exist_ok=True)
    relative = 'capture_filtered_updates/' + target.name
    for name in INPUT_FILES:
        src, dst = source / name, target / name
        if not src.is_file():
            raise FileNotFoundError(src)
        if src.suffix == '.json':
            write_json(dst, relocate(read(src), source, relative, v14))
        elif name in ('truth_frames.jsonl', 'trajectories.jsonl'):
            shutil.copyfile(src, dst)
        else:
            write_rows(dst, (relocate(row, source, relative, v14) for row in rows(src)))


def weather_service(render_root):
    path = render_root / 'Plugins/SumoImporter/Scripts/donghu_core/weather_service.py'
    package = types.ModuleType('_p09_release_weather')
    package.__path__ = [str(path.parent)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + '.weather_service', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    profiles = render_root / 'Config/LowAltitude/weather_render_profiles.json'
    return module.WeatherService.from_profiles_path(profiles), {
        'service_source_ref': str(path), 'profile_source_ref': str(profiles),
        'ue_consumer_source_ref': str(render_root / 'Plugins/AeroSimHost/Source/AeroWeatherRender/Private/AeroWeatherRenderSubsystem.cpp'),
        'semantics': 'Shared-main renderer payload evaluation only; profile dust is a render setting, not a scientific measurement',
        'local_ue_plugin_inspected': False, 'actual_ue_capture_performed': False}


def normalized_weather(source, *, full_episode):
    raw = list(rows(source))
    if full_episode:
        raw = [row for row in raw if row['tick'] <= 900]
    normalized = [conv.normalize_weather_row(row, row['tick']) for row in raw]
    ticks = [row['tick'] for row in normalized]
    if full_episode and ticks != DENSE_TICKS:
        raise ValueError('Episode weather must cover 0..900 exactly: ' + str(source))
    if not full_episode and (not ticks or ticks != list(range(ticks[0], ticks[-1] + 1))):
        raise ValueError('ARM raw weather must preserve its actual contiguous window: ' + str(source))
    return raw, normalized


def weather_change(before, after, service):
    if [row['tick'] for row in before] != [row['tick'] for row in after]:
        raise ValueError('Weather source and target tick windows diverge')
    changed, render_changed, removed = [], [], Counter()
    for old, new in zip(before, after):
        if old != new:
            changed.append(new['tick'])
        for key in old.keys() - new.keys():
            removed[key] += 1
        if service.payload_for_row(old) != service.payload_for_row(new):
            render_changed.append(new['tick'])
    return {'source_value_changed_ticks': changed, 'render_payload_changed_ticks': render_changed,
            'removed_source_fields': dict(removed), 'render_payload_rows_compared': len(after)}


def main_weather(ep, full, selected, v14, receipt, service):
    original = ROOT / 'Dataset/episodes' / ep / 'weather_meta.jsonl'
    baseline_path = BASE / ep / 'weather_meta.jsonl'
    baseline = list(rows(baseline_path))
    raw_ref = None
    if ep == L6_ID:
        raw_ref = receipt / 'weather.jsonl'
    elif ep in selected:
        recorded = v14 / 'sources' / ep / 'weather.jsonl'
        if recorded.is_file():
            raw_ref = recorded
        elif original.is_file():
            raw_ref = original
    elif original.is_file():
        raw_ref = original
    if raw_ref is not None:
        raw, current = normalized_weather(raw_ref, full_episode=True)
        field_keys = sorted(set().union(*(row.keys() for row in current)))
        field_sources = {key: {'status': 'recorded_simulator_state', 'source_ref': str(raw_ref)}
                        for key in field_keys if key != 'tick'}
        if not any('dust' in row for row in raw):
            field_sources['dust'] = {'status': 'absent_from_selected_raw_source',
                                    'scientific_measurement_available': False}
        status = 'DERIVED_FROM_RECORDED_RAW_WEATHER'
    else:
        # The original L2 raw authority is absent. Retain its published values
        # with an explicit source limitation; never call its old zero measured.
        current = [{key: value for key, value in row.items() if key != 'dust'} for row in baseline]
        if [row['tick'] for row in current] != DENSE_TICKS:
            raise ValueError('Published weather does not cover 0..900: ' + ep)
        field_sources = {key: {'status': 'published_value_raw_authority_unavailable',
                              'source_ref': str(baseline_path)}
                        for key in current[0] if key != 'tick'}
        if 'dust' in baseline[0]:
            field_sources['dust'] = {'status': 'unsupported_legacy_default_removed',
                                    'scientific_measurement_available': False,
                                    'source_ref': str(baseline_path)}
        status = 'PUBLISHED_VALUES_RETAINED_WITH_RAW_SOURCE_LIMITATION'
    if ep in selected and ep != L6_ID:
        previous = list(rows(full / 'weather_meta.jsonl'))
        if any({k: v for k, v in old.items() if k != 'dust'} !=
               {k: v for k, v in new.items() if k != 'dust'} for old, new in zip(previous, current)):
            raise ValueError('Selected v14 numeric weather changed outside absent dust: ' + ep)
    return current, {'status': status, 'original_raw_source_ref': str(original),
        'original_raw_source_available': original.is_file(),
        'selected_raw_source_ref': None if raw_ref is None else str(raw_ref),
        'published_source_ref': str(baseline_path), 'field_sources': field_sources,
        'rows': len(current), **weather_change(baseline, current, service)}


def l6_overlay(output, receipt, v14):
    base, target = BASE / L6_ID, output / 'capture_filtered_updates' / L6_ID
    target.mkdir(parents=True, exist_ok=True)
    relative = 'capture_filtered_updates/' + L6_ID
    source_rows = {(row['tick'], row['entity_id']): row for row in rows(receipt / 'trajectories.jsonl')
                   if row['entity_id'] in TARGETS}
    script = read(ROOT / 'Dataset/scenarios/L6_digital_layer/failure/L6-5_v1/event_script.json')
    scene = read(ROOT / 'Dataset/scenarios/L6_digital_layer/failure/L6-5_v1/scene_setup.json')
    scene_entities = {entry['entity_id']: entry for entry in scene['entities']}
    equality_count, backgrounds, trajectory_count = 0, 0, 0
    target_counts = Counter()
    with (target / 'truth_frames.jsonl').open('wb') as truth, (target / 'trajectories.jsonl').open('wb') as trajectories:
        for frame in rows(base / 'truth_frames.jsonl'):
            original_ids = [e['entity_id'] for e in frame['entities']]
            for entity in frame['entities']:
                eid = entity['entity_id']
                if eid not in TARGETS:
                    backgrounds += 1
                    continue
                recorded = source_rows[frame['tick'], eid]
                old_visibility = {key: copy.deepcopy(entity[key]) for key in
                    ('render_presence', 'runtime_visibility', 'uav_visibility') if key in entity}
                entity['truth_pose'] = conv.truth_pose(recorded['pos_enu'], recorded['yaw_deg'], recorded['vel_mps'])
                entity['truth_pose']['position_enu_m'] = recorded['pos_enu']
                entity['truth_pose']['velocity_enu_mps'] = recorded['vel_mps']
                entity['truth_pose']['rotation_deg']['yaw_deg'] = recorded['yaw_deg']
                entity['truth_pose']['authority_owner'] = 'actual_l6_r2_engine_projection'
                entity['annotations'] = conv.build_annotations(recorded['activity_type'], recorded, entity['entity_category'])
                entity['annotations']['state_facets'].pop('network', None)
                entity.update(conv.preserved_fields_from(recorded, scene_entities[eid]))
                entity['state'] = recorded['state']
                for key in RUNTIME_STATE_FIELDS:
                    if key in recorded:
                        entity[key] = copy.deepcopy(recorded[key])
                    else:
                        entity.pop(key, None)
                entity.update(old_visibility)
                pose = entity['truth_pose']
                if (pose['position_enu_m'], pose['velocity_enu_mps'], pose['rotation_deg']['yaw_deg']) != (
                        recorded['pos_enu'], recorded['vel_mps'], recorded['yaw_deg']):
                    raise ValueError('L6 actual pose projection differs from recorded source')
                equality_count += 1
                target_counts[eid] += 1
            if original_ids != [e['entity_id'] for e in frame['entities']]:
                raise ValueError('L6 projection changed captured entity activation/visibility')
            update_frame_summaries(frame, frame['entities'])
            frame['entity_motion_state'] = conv.truth_boundary_summary(frame['entities'],
                capture_boundary_id=frame['capture_boundary_id'], uav_crosses_boundary=frame['uav_crosses_boundary'],
                inspect_observes_boundary=frame['inspect_observes_boundary'], pad_boundary_policy=frame['pad_boundary_policy'])['entity_motion_state']
            frame['p09_input_revision'] = 'current-l6-r2-receipt-projection'
            truth.write(orjson.dumps(clean(frame)) + b'\n')
            for entity in frame['entities']:
                trajectories.write(orjson.dumps(clean(conv.truth_entity_to_trajectory_row(frame, entity))) + b'\n')
                trajectory_count += 1
    for name in ('global_entity_roster.json', 'scenario_plan.json', 'scenario_package.json',
                 'episode_manifest.json', 'scene_occupancy_manifest.json', 'render_host_config.json'):
        write_json(target / name, relocate(read(base / name), base, relative, v14))
    write_json(target / 'event_script.json', script)
    write_json(target / 'scene_setup.json', scene)
    write_rows(target / 'event_trace.jsonl', rows(receipt / 'raw/ue_event_trace.jsonl'))
    write_rows(target / 'event_realization.jsonl', rows(receipt / 'raw/ue_event_realization.jsonl'))
    realizations = list(rows(target / 'event_realization.jsonl'))
    events = list(rows(target / 'event_trace.jsonl'))
    labels = conv.build_dynamic_labels(events, L6_ID, scenario_id='L6-5_v1', event_realization_rows=realizations)
    write_rows(target / 'dynamic_labels.jsonl', labels)
    manifest = read(target / 'episode_manifest.json')
    manifest.update(source_event_script_path=relative + '/event_script.json', source_scene_setup_path=relative + '/scene_setup.json',
        p09_actual_receipt_source='sources/' + L6_ID, event_realization_status='ACTUAL_ENGINE_DISPATCH_AND_TERMINAL_ROWS',
        n_events=len(events), n_event_realizations=len(realizations))
    manifest['record_counts'].update(truth_frames=901, trajectories=trajectory_count, event_trace=len(events),
                                    event_realization=len(realizations), dynamic_labels=len(labels), weather_meta=901)
    manifest['canonical_record_counts'] = copy.deepcopy(manifest['record_counts'])
    write_json(target / 'episode_manifest.json', manifest)
    config = read(target / 'render_host_config.json')
    config.update(episode_dir=relative, event_script_path=relative + '/event_script.json')
    write_json(target / 'render_host_config.json', config)
    package = read(target / 'scenario_package.json')
    package.update(root_dir=relative, scene_setup=relative + '/scene_setup.json', event_script=relative + '/event_script.json')
    write_json(target / 'scenario_package.json', package)
    plan = read(target / 'scenario_plan.json')
    plan['p09_current_event_script'] = 'event_script.json'
    plan['p09_current_scene'] = 'scene_setup.json'
    plan['p09_compilation_status'] = 'Original background compilation retained; target execution is actual R2 source'
    write_json(target / 'scenario_plan.json', plan)
    window = {'episode_id': L6_ID, 'tick_start': 0, 'tick_end_inclusive': 900,
        'capture_step_ticks': 5, 'planned_capture_ticks': CAPTURE_TICKS, 'planned_capture_frames': len(CAPTURE_TICKS),
        'actual_sensor_frames_collected_by_this_operation': 0,
        'source_ref': str(receipt), 'story_acceptance_is_separate_from_capture_window_completeness': True}
    write_json(target / 'capture_window.json', window)
    write_rows(target / 'multimodal_window_mask.jsonl', ({'episode_id': L6_ID, 'tick': tick,
        'multimodal_window_valid': True, 'observed_sensor_availability': 'NOT_CAPTURED_BY_THIS_ASSEMBLY'} for tick in DENSE_TICKS))
    evidence = output / 'sources' / L6_ID
    for name in ('command_receipts.json', 'command_input.json', 'physical_evidence.json', 'recovery_application.json',
                 'audit.jsonl', 'executed_actions.jsonl', 'trajectories.jsonl', 'weather.jsonl'):
        source, destination = receipt / name, evidence / name
        if source.suffix == '.json':
            write_json(destination, read(source))
        else:
            write_rows(destination, rows(source))
    return {'source_ref': str(receipt), 'target_pose_equalities': equality_count,
            'target_rows': dict(target_counts), 'unchanged_background_rows': backgrounds,
            'background_authority': 'Exact frozen captured SUMO/global/background rows; no restored or synthesized actors',
            'actor_activation_and_visibility': 'Preserved from frozen captured entities per tick'}


def visual_changes(base, current, weather):
    changes = {key: [] for key in ('position', 'rotation', 'entities', 'visual_state')}
    dense, capture = [], []
    for before, after in zip(rows(base / 'truth_frames.jsonl'), rows(current / 'truth_frames.jsonl'), strict=True):
        tick = after['tick']
        if before['tick'] != tick:
            raise ValueError('Base/current frame ticks diverge: ' + current.name)
        dense.append(tick)
        if tick % 5:
            continue
        capture.append(tick)
        left = {e['entity_id']: e for e in before['entities']}
        right = {e['entity_id']: e for e in after['entities']}
        if left.keys() != right.keys():
            changes['entities'].append(tick)
        position, rotation, visual = False, False, False
        for eid in left.keys() & right.keys():
            a, b = left[eid], right[eid]
            position |= a['truth_pose']['position_enu_m'] != b['truth_pose']['position_enu_m']
            rotation |= a['truth_pose']['rotation_deg'] != b['truth_pose']['rotation_deg']
            visual |= any(a.get(key) != b.get(key) for key in ('state', 'logical_asset_id', 'render_presence'))
            visual |= a['annotations']['activity_type'] != b['annotations']['activity_type']
        for key, changed in (('position', position), ('rotation', rotation), ('visual_state', visual)):
            if changed:
                changes[key].append(tick)
    if dense != DENSE_TICKS or capture != CAPTURE_TICKS:
        raise ValueError('Full UE input must have dense 0..900 and capture step5: ' + current.name)
    changes['weather_render_payload'] = [tick for tick in weather['render_payload_changed_ticks'] if tick % 5 == 0]
    requested = sorted(set().union(*changes.values()))
    return {'dense_ticks_verified': len(dense), 'capture_ticks_verified': len(capture),
        'capture_step_ticks': 5, 'tick_start': dense[0], 'tick_end': dense[-1],
        'first_position_changed_capture_tick': None if not changes['position'] else changes['position'][0],
        'changed_capture_ticks_by_field': changes, 'recapture_ticks': requested,
        'recapture_frame_count': len(requested),
        'comparison_semantics': 'Position/rotation/entity/visual-state/render-weather values; velocity-only rounding is excluded from position onset'}


def formal_l2_status():
    closure_path = CLOSURE_DIR / 'objective_r2/L2-1_v2__seed00/epi_closure.json'
    closure = read(closure_path)
    return {'status': closure['status'], 'source_ref': str(closure_path),
        'receipt_source_ref': str(CLOSURE_DIR / 'formal_objective_receipt_r2.json'),
        'diagnosis_source_ref': str(CLOSURE_DIR / 'link_loss_diagnosis.json'),
        'stages': [{key: stage[key] for key in ('stage_index', 'stage_goal', 'stage_goal_status', 'supporting_tick', 'status_reason')}
                   for stage in closure['stages']],
        'historical_v14_event_firing_is_not_scientific_closure': True}


def global_yaw_source_index(source_dir):
    """Sample the fixed city source once; index candidates before file IO."""
    from Dataset.tools.uav_global_flow.truth_integration import UavGlobalFlowDataset, segment_for_seed
    dataset = UavGlobalFlowDataset(source_dir)
    for frame in dataset.frames:
        for entity in frame['uavs']:
            if conv.source_yaw_degrees(entity, 'yaw_deg', context=str(dataset.frames_path)) is None:
                raise ValueError('Native global flow yaw is unrecorded: ' + entity['uav_id'])
    index = {}
    for seed in range(3):
        segment = segment_for_seed(seed)
        previous = {}
        for tick in DENSE_TICKS:
            records = dataset.sample(segment=segment, episode_sim_time_s=tick / 10)['uavs']
            for entity in records:
                uid, position = entity['uav_id'], entity['position_enu_m']
                old_position = previous.get(uid)
                still = math.hypot(*entity['velocity_enu_mps'][:2]) <= 1e-5
                if old_position is not None:
                    still |= math.hypot(position[0] - old_position[0], position[1] - old_position[1]) <= 1e-6
                if still:
                    index.setdefault((seed, uid), {})[tick] = {
                        'yaw_deg': entity['yaw_deg'], 'position_enu_m': position,
                        'source_prev_time_s': entity['source_prev_time_s'],
                        'source_next_time_s': entity['source_next_time_s'],
                        'source_alpha': entity['source_alpha']}
                previous[uid] = position
    return dataset, index


def global_yaw_candidates(roster, index, start, end, bbox=None):
    result = {}
    for entry in roster.values():
        if entry.get('source') != 'uav_global_flow' or entry.get('entity_category') != 'uav':
            continue
        uid = entry['uav_id']
        segment = entry['uav_segment']
        seed = segment['seed_index']
        for tick, source in index.get((seed, uid), {}).items():
            if not start <= tick <= end:
                continue
            x, y = source['position_enu_m'][:2]
            if bbox is not None and not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
                continue
            result.setdefault(tick, {})[entry['entity_id']] = {'uav_id': uid, 'seed_index': seed, **source}
    return result


_TOP_LEVEL_TICK_PREFIX = re.compile(rb'^\s*\{\s*(?:"[^"\\]*"\s*:\s*(?:"(?:[^"\\]|\\.)*"|-?[0-9.eE+]+|true|false|null)\s*,\s*)*"tick"\s*:\s*(\d+)(?=\s*[,}])')


def explicit_row_tick(line, path):
    # The fast path accepts scalar top-level keys only, never nested entity ticks.
    match = _TOP_LEVEL_TICK_PREFIX.match(line[:2048])
    if match is not None:
        return int(match[1]), None
    row = orjson.loads(line)
    tick = row['tick']
    if not isinstance(tick, int) or isinstance(tick, bool):
        raise ValueError('Row tick is not an explicit integer: ' + str(path))
    return tick, row


def pending_global_yaw_changes(path, changes):
    """Recover actual completed changes from their immutable original rows."""
    applicable, pending, rejected = {}, {}, Counter()
    wanted_ticks = {tick for tick, _ in changes}
    with path.open('rb') as handle:
        for line in handle:
            tick, frame = explicit_row_tick(line, path)
            if tick not in wanted_ticks:
                continue
            if frame is None:
                frame = orjson.loads(line)
            for entity in frame['entities']:
                key = tick, entity['entity_id']
                change = changes.get(key)
                if change is None or entity.get('source') != 'uav_global_flow':
                    continue
                pose = entity['truth_pose']
                if pose['position_enu_m'] != change['position_enu_m'] or math.hypot(*pose['velocity_enu_mps'][:2]) > 1e-5:
                    rejected['source_position_or_horizontal_motion_differs_current_world'] += 1
                    continue
                yaw = conv.source_yaw_degrees(pose['rotation_deg'], 'yaw_deg', context=entity['entity_id'])
                if yaw not in (change['old_yaw_deg'], change['new_yaw_deg']):
                    raise ValueError('Current global yaw differs from both source and frozen binding: ' + str(key))
                applicable[key] = change
                if yaw != change['new_yaw_deg']:
                    pending[key] = change
    rejected['source_actor_not_in_current_scope'] = len(changes) - len(applicable) - sum(rejected.values())
    return applicable, pending, dict(rejected)


def global_yaw_changes(path, candidates):
    """Parse only indexed ticks; keep intervened or displaced actors untouched."""
    changes, frame_ticks, rejected = {}, [], Counter()
    with path.open('rb') as handle:
        for line in handle:
            tick, frame = explicit_row_tick(line, path)
            frame_ticks.append(tick)
            if tick not in candidates:
                continue
            if frame is None:
                frame = orjson.loads(line)
            for entity in frame['entities']:
                source = candidates[tick].get(entity['entity_id'])
                if source is None or entity.get('source') != 'uav_global_flow':
                    continue
                pose = entity['truth_pose']
                if pose['position_enu_m'] != source['position_enu_m']:
                    rejected['source_position_differs_current_world'] += 1
                    continue
                if math.hypot(*pose['velocity_enu_mps'][:2]) > 1e-5:
                    continue
                before = conv.source_yaw_degrees(pose['rotation_deg'], 'yaw_deg', context=entity['entity_id'])
                after = conv.source_yaw_degrees(source, 'yaw_deg', context=source['uav_id'])
                if before != after:
                    changes[tick, entity['entity_id']] = {**source, 'tick': tick,
                        'entity_id': entity['entity_id'], 'old_yaw_deg': before, 'new_yaw_deg': after}
    return changes, frame_ticks, dict(rejected)


def patch_global_yaw_pose_files(target, changes):
    """Rewrite changed records only; copy every other numeric row byte for byte."""
    ticks = {key[0] for key in changes}; counts = Counter()
    entity_tokens = {key[1].encode() for key in changes}
    for name in ('truth_frames.jsonl', 'trajectories.jsonl'):
        path = target / name; temporary = path.with_suffix('.p09_yaw_pending')
        with path.open('rb') as source, temporary.open('wb') as destination:
            for line in source:
                if not any(token in line for token in entity_tokens):
                    destination.write(line); continue
                tick, row = explicit_row_tick(line, path)
                if tick not in ticks:
                    destination.write(line); continue
                if row is None:
                    row = orjson.loads(line)
                changed = False
                if name == 'truth_frames.jsonl':
                    for entity in row['entities']:
                        change = changes.get((row['tick'], entity['entity_id']))
                        if change is not None:
                            entity['truth_pose']['rotation_deg']['yaw_deg'] = change['new_yaw_deg']
                            counts[name] += 1; changed = True
                else:
                    change = changes.get((row['tick'], row['entity_id']))
                    if change is not None:
                        if 'yaw_deg' not in row:
                            raise ValueError('Trajectory yaw is unrecorded: ' + str(path))
                        row['yaw_deg'] = change['new_yaw_deg']; counts[name] += 1; changed = True
                destination.write(orjson.dumps(row) + b'\n' if changed else line)
        temporary.replace(path)
    if any(counts[name] != len(changes) for name in ('truth_frames.jsonl', 'trajectories.jsonl')):
        raise ValueError('Truth and trajectory source bindings differ: ' + str(target))


def copy_frozen_capture_inputs(source, target, relative, weather_path, *, start, end, v14):
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name not in INPUT_FILES and path.name != 'capture_plan.json':
            continue
        if path.suffix == '.json':
            write_json(target / path.name, relocate(read(path), source, relative, v14))
        else:
            shutil.copyfile(path, target / path.name)
    manifest = read(target / 'episode_manifest.json')
    for field, name in (('source_scene_setup_path', 'scene_setup.json'), ('source_event_script_path', 'event_script.json')):
        source_path = Path(manifest[field])
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if not (target / name).is_file():
            write_json(target / name, read(source_path))
        manifest[field] = relative + '/' + name
    manifest['p09_global_yaw_source_policy'] = 'recorded_global_yaw_when_horizontal_motion_norm_le_1e-5'
    write_json(target / 'episode_manifest.json', manifest)
    config = read(target / 'render_host_config.json')
    config.update(episode_dir=relative, event_script_path=relative + '/event_script.json')
    write_json(target / 'render_host_config.json', config)
    package = read(target / 'scenario_package.json')
    package.update(root_dir=relative, scene_setup=relative + '/scene_setup.json', event_script=relative + '/event_script.json')
    # ARM metadata uses its own arms/... reference prefix.
    old_relative = str(source.relative_to(ROOT))
    package = relocate(package, ROOT / old_relative, relative, v14)
    for field, value in list(package.items()):
        if isinstance(value, str) and value.startswith(old_relative + '/'):
            package[field] = relative + value[len(old_relative):]
    write_json(target / 'scenario_package.json', package)
    if weather_path.resolve() != (target / 'weather_meta.jsonl').resolve():
        shutil.copyfile(weather_path, target / 'weather_meta.jsonl')
    capture_ticks = [tick for tick in range(start, end + 1) if tick % 5 == 0]
    write_json(target / 'capture_window.json', {'episode_id': manifest['episode_id'],
        'tick_start': start, 'tick_end_inclusive': end, 'capture_step_ticks': 5,
        'planned_capture_ticks': capture_ticks, 'planned_capture_frames': len(capture_ticks),
        'actual_sensor_frames_collected_by_this_operation': 0, 'source_ref': str(source)})
    write_rows(target / 'multimodal_window_mask.jsonl', ({'episode_id': manifest['episode_id'],
        'tick': tick, 'multimodal_window_valid': True,
        'observed_sensor_availability': 'NOT_CAPTURED_BY_THIS_ASSEMBLY'} for tick in range(start, end + 1)))


def apply_core_source_updates(output, v14, render_source_root):
    """Increment existing release without rerunning simulator/weather/ontology."""
    started = time.perf_counter(); manifest_path = output / 'source_status_manifest.json'; manifest = read(manifest_path)
    if manifest.get('global_yaw_source_updates', {}).get('status') == 'complete':
        return manifest['global_yaw_source_updates']
    print('Incremental core-source publisher PID', os.getpid(), flush=True)
    service, _ = weather_service(render_source_root)
    weather_removed = []
    for entry in manifest['episodes']:
        weather = entry['weather']
        if weather['selected_raw_source_ref'] is not None:
            if not weather['original_raw_source_available']:
                weather['selected_source_origin_limitation'] = 'Original simulator source absent; selected runtime weather is recorded, not a measured dust authority'
            continue
        path = output / entry['weather_update_path']; before = list(rows(path))
        if any(row.get('dust', 0) != 0 for row in before):
            raise ValueError('Published nonzero dust requires its actual source: ' + str(path))
        after = [{key: value for key, value in row.items() if key != 'dust'} for row in before]
        difference = weather_change(before, after, service)
        if difference['render_payload_changed_ticks']:
            raise ValueError('Removing unsupported dust unexpectedly changed renderer payload: ' + str(path))
        write_rows(path, after)
        weather.update(weather_change(list(rows(BASE / entry['episode_id'] / 'weather_meta.jsonl')), after, service))
        weather['field_sources']['dust'] = {'status': 'unsupported_legacy_default_removed',
            'scientific_measurement_available': False, 'source_ref': weather['published_source_ref']}
        weather_removed.append({'episode_id': entry['episode_id'], 'removed_rows': len(before), 'renderer_payload_equal_rows': len(before)})
    origin_manifest = read(BASE / manifest['episodes'][0]['episode_id'] / 'episode_manifest.json')
    dataset, index = global_yaw_source_index(Path(origin_manifest['uav_global_flow']['source']['output_dir']))
    print('Global source indexed', len(dataset.frames), 'frames', sum(map(len, index.values())), 'stationary dense actor rows', flush=True)
    evidence = []; main_updates = []; arm_updates = []; arm_checked = 0
    for entry in manifest['episodes']:
        ep = entry['episode_id']; source = BASE / ep; full = output / 'capture_filtered_updates' / ep
        current = full if (full / 'truth_frames.jsonl').is_file() else source
        native_manifest = read(source / 'episode_manifest.json')
        if Path(native_manifest['uav_global_flow']['source']['output_dir']).resolve() != dataset.output_dir:
            raise ValueError('Episode declares a different global source: ' + ep)
        bbox = native_manifest['uav_global_flow']['runtime_spatial_crop']['bbox_enu_m']
        candidates = global_yaw_candidates(conv.read_source_roster(current / 'global_entity_roster.json'), index, 0, 900, bbox)
        if not candidates:
            continue
        changes, ticks, rejected = global_yaw_changes(source / 'truth_frames.jsonl', candidates)
        pending = changes
        if current != source:
            changes, pending, recovery_rejected = pending_global_yaw_changes(current / 'truth_frames.jsonl', changes)
            rejected.update(recovery_rejected)
        if ticks != DENSE_TICKS:
            raise ValueError('Actual episode dense grid differs: ' + ep)
        if not changes:
            continue
        if current == source:
            weather_path = output / entry['weather_update_path']
            copy_frozen_capture_inputs(source, full, 'capture_filtered_updates/' + ep, weather_path, start=0, end=900, v14=v14)
        if entry['update_scope'] != 'full_capture_inputs':
            entry.update(input_dir='capture_filtered_updates/' + ep, update_scope='full_capture_inputs', weather_update_path='capture_filtered_updates/' + ep + '/weather_meta.jsonl')
            entry['capture_contract_and_changes'] = {'dense_ticks_verified': len(ticks),
                'capture_ticks_verified': len([tick for tick in ticks if tick % 5 == 0]),
                'capture_step_ticks': 5, 'tick_start': 0, 'tick_end': 900,
                'first_position_changed_capture_tick': None,
                'changed_capture_ticks_by_field': {'position': [], 'rotation': [], 'entities': [], 'visual_state': [],
                    'weather_render_payload': [tick for tick in entry['weather']['render_payload_changed_ticks'] if tick % 5 == 0]},
                'comparison_semantics': 'Existing frozen actors and positions preserved; source yaw corrected only without horizontal motion'}
        if pending:
            patch_global_yaw_pose_files(full, pending)
        capture_ticks = sorted({tick for tick, _ in changes if tick % 5 == 0})
        detail = {'episode_id': ep, 'dense_actor_rows_changed': len(changes),
            'capture_actor_rows_changed': sum(tick % 5 == 0 for tick, _ in changes),
            'changed_dense_ticks': sorted({tick for tick, _ in changes}), 'changed_capture_ticks': capture_ticks,
            'source_position_rejections': rejected}
        main_updates.append(detail); entry['global_uav_yaw_update'] = detail
        rotation = entry['capture_contract_and_changes']['changed_capture_ticks_by_field']['rotation']
        rotation[:] = sorted(set(rotation) | set(capture_ticks))
        evidence.extend({'episode_id': ep, 'input_dir': entry['input_dir'], 'source_ref': str(dataset.frames_path), **change} for change in changes.values())
        print('Global yaw episode updated', ep, len(changes), flush=True)
    for arm_record in manifest['arm_weather_updates']:
        arm_relative = arm_record['arm_path']; source = ROOT / 'arms' / arm_relative / 'ue'
        roster = conv.read_source_roster(source / 'global_entity_roster.json')
        if not any(entity.get('source') == 'uav_global_flow' and entity.get('entity_category') == 'uav' for entity in roster.values()):
            continue
        arm_checked += 1; native = read(source / 'episode_manifest.json')
        start, end = native['tick_start'], native['tick_end']
        candidates = global_yaw_candidates(roster, index, start, end)
        if not candidates:
            continue
        changes, ticks, rejected = global_yaw_changes(source / 'truth_frames.jsonl', candidates)
        if not changes:
            continue
        if ticks != list(range(start, end + 1)):
            raise ValueError('Actual ARM window differs: ' + arm_relative)
        relative = 'arm_capture_updates/' + arm_relative + '/ue'; target = output / relative
        pending = changes
        if (target / 'truth_frames.jsonl').is_file():
            changes, pending, recovery_rejected = pending_global_yaw_changes(target / 'truth_frames.jsonl', changes)
            rejected.update(recovery_rejected)
        else:
            copy_frozen_capture_inputs(source, target, relative, output / arm_record['weather_update_path'], start=start, end=end, v14=v14)
        if pending:
            patch_global_yaw_pose_files(target, pending)
        capture_ticks = sorted({tick for tick, _ in changes if tick % 5 == 0})
        detail = {'arm_path': arm_relative, 'input_dir': relative, 'ue_entry': relative + '/render_host_config.json',
            'dense_actor_rows_changed': len(changes), 'capture_actor_rows_changed': sum(tick % 5 == 0 for tick, _ in changes),
            'changed_dense_ticks': sorted({tick for tick, _ in changes}), 'recapture_ticks': capture_ticks,
            'recapture_frame_count': len(capture_ticks), 'source_position_rejections': rejected,
            'target_intervention_state_modified': False}
        arm_updates.append(detail); arm_record['global_uav_yaw_update'] = detail
        evidence.extend({'arm_path': arm_relative, 'episode_id': native['episode_id'], 'input_dir': relative,
            'source_ref': str(dataset.frames_path), **change} for change in changes.values())
        print('Global yaw ARM background updated', arm_relative, len(changes), flush=True)
    recapture = []
    for entry in manifest['episodes']:
        if entry['update_scope'] == 'full_capture_inputs':
            detail = entry['capture_contract_and_changes']; detail['recapture_ticks'] = sorted(set().union(*detail['changed_capture_ticks_by_field'].values()))
            detail['recapture_frame_count'] = len(detail['recapture_ticks'])
            if detail['recapture_ticks']:
                recapture.append({'episode_id': entry['episode_id'], 'ue_entry': entry['input_dir'] + '/render_host_config.json', **detail})
        else:
            ticks = [tick for tick in entry['weather']['render_payload_changed_ticks'] if tick % 5 == 0]
            if ticks:
                recapture.append({'episode_id': entry['episode_id'], 'update_scope': 'weather_only',
                    'recapture_ticks': ticks, 'recapture_frame_count': len(ticks), 'weather_update_path': entry['weather_update_path']})
    full_entries = [entry for entry in manifest['episodes'] if entry['update_scope'] == 'full_capture_inputs']
    manifest.update(full_capture_input_count=len(full_entries), weather_only_episode_count=len(manifest['episodes']) - len(full_entries),
        full_input_dense_frames=sum(e['capture_contract_and_changes']['dense_ticks_verified'] for e in full_entries),
        full_input_capture_frames=sum(e['capture_contract_and_changes']['capture_ticks_verified'] for e in full_entries),
        recapture=recapture, recapture_episode_count=len(recapture), recapture_planned_frames=sum(e['recapture_frame_count'] for e in recapture),
        arm_pose_recapture=arm_updates, arm_pose_recapture_planned_frames=sum(e['recapture_frame_count'] for e in arm_updates))
    summary = {'status': 'complete', 'policy': 'recorded_global_yaw_when_horizontal_motion_norm_le_1e-5',
        'source_frames_ref': str(dataset.frames_path), 'source_task_plan_ref': str(dataset.task_plan_path),
        'main_episode_updates': main_updates, 'arm_background_updates': arm_updates,
        'arm_global_background_rosters_checked': arm_checked, 'unsupported_legacy_dust_removed': weather_removed,
        'dense_actor_rows_changed': sum(e['dense_actor_rows_changed'] for e in main_updates),
        'capture_actor_rows_changed': sum(e['capture_actor_rows_changed'] for e in main_updates),
        'arm_dense_actor_rows_changed': sum(e['dense_actor_rows_changed'] for e in arm_updates),
        'arm_capture_actor_rows_changed': sum(e['capture_actor_rows_changed'] for e in arm_updates),
        'actual_ue_execution_performed': False, 'frozen_inputs_modified': False, 'elapsed_s': time.perf_counter() - started}
    manifest['global_yaw_source_updates'] = summary
    manifest['incremental_reproduction_command'] = shlex.join([sys.executable, '-B', '-m', 'Dataset.tools.p09_release', '--project-root', str(ROOT),
        '--output-dir', str(output), '--v14-dir', str(v14), '--render-source-root', str(render_source_root), '--incremental-core-sources'])
    write_rows(output / 'global_uav_yaw_updates.jsonl', evidence)
    write_json(manifest_path, manifest)
    for entry in full_entries:
        obsolete = output / 'weather_updates' / entry['episode_id'] / 'weather_meta.jsonl'
        if obsolete.exists():
            obsolete.unlink(); obsolete.parent.rmdir()
    with (output / 'recapture_episodes.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['episode_id','full_episode_capture_frames','changed_capture_frames','first_position_changed_capture_tick','changed_fields','changed_capture_ticks','ue_entry'])
        for entry in recapture:
            fields = entry.get('changed_capture_ticks_by_field', {})
            writer.writerow([entry['episode_id'], entry.get('capture_ticks_verified', ''), entry['recapture_frame_count'],
                entry.get('first_position_changed_capture_tick'), ';'.join(key for key, value in fields.items() if value),
                orjson.dumps(entry['recapture_ticks']).decode(), entry.get('ue_entry', '')])
    with (output / 'recapture_arms.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['arm_path','changed_capture_frames','changed_capture_ticks','ue_entry'])
        for entry in arm_updates:
            writer.writerow([entry['arm_path'], entry['recapture_frame_count'], orjson.dumps(entry['recapture_ticks']).decode(), entry['ue_entry']])
    print(orjson.dumps({key: value for key, value in summary.items() if key not in ('main_episode_updates', 'arm_background_updates')}).decode(), flush=True)
    return summary


def main():
    global ROOT, BASE, CLOSURE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=RUNTIME / 'ue_input_current')
    parser.add_argument('--v14-dir', type=Path, default=RUNTIME / 'ue_input_overlay_v14')
    parser.add_argument('--receipt-dir', type=Path, default=RUNTIME / 'l6_receipt_r2')
    parser.add_argument('--render-source-root', type=Path, default=ROOT)
    parser.add_argument('--incremental-core-sources', action='store_true')
    args = parser.parse_args()
    ROOT = args.project_root.resolve()
    BASE = ROOT / 'aw_data/render_ready_episodes_capture_filtered'
    CLOSURE_DIR = ROOT / 'design/p09/mechanism_completion_v1/numeric_mapping_checkpoint/builder_short_session_v1/author_formal_objective_checkpoint_v8'
    output, v14, receipt = args.output_dir.resolve(), args.v14_dir.resolve(), args.receipt_dir.resolve()
    if args.incremental_core_sources:
        apply_core_source_updates(output, v14, args.render_source_root.resolve())
        return
    for source in (v14, receipt, BASE, ROOT / 'arms'):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('Output directory overlaps an immutable input: ' + str(source))
    if (output / 'source_status_manifest.json').exists():
        raise FileExistsError('Refusing completed current release: ' + str(output))
    started = time.perf_counter()
    service, consumer = weather_service(args.render_source_root.resolve())
    source_manifest = read(v14 / 'assembly_manifest.json')
    selected = {record['episode_id']: record for record in source_manifest['episodes']}
    episode_dirs = sorted(path for path in BASE.iterdir() if path.is_dir() and (path / 'episode_manifest.json').is_file())
    arm_sources = sorted((ROOT / 'arms').glob('*/*/*/*/ue/weather_meta.jsonl'))
    if len(selected) != 36 or len(episode_dirs) != 210 or len(arm_sources) != 690:
        raise ValueError('Current release inputs diverge from declared 36/210/690 scope')
    output.mkdir(parents=True, exist_ok=True)
    for ep in selected:
        copy_inputs(v14 / 'capture_filtered_updates' / ep, output / 'capture_filtered_updates' / ep, v14)
        print('Copied recorded full input', ep, flush=True)
    l6_projection = l6_overlay(output, receipt, v14)
    full_ids = set(selected) | {L6_ID}
    episodes, recapture = [], []
    for base in episode_dirs:
        ep = base.name
        full = output / 'capture_filtered_updates' / ep
        weather, status = main_weather(ep, full, selected, v14, receipt, service)
        destination = full / 'weather_meta.jsonl' if ep in full_ids else output / 'weather_updates' / ep / 'weather_meta.jsonl'
        write_rows(destination, weather)
        entry = {'episode_id': ep, 'base_capture_ref': str(base), 'weather': status,
                 'weather_update_path': str(destination.relative_to(output)),
                 'update_scope': 'full_capture_inputs' if ep in full_ids else 'weather_only',
                 'scientific_closure_status': 'NOT_ASSESSED_BY_THIS_SERIALIZATION'}
        if ep in full_ids:
            entry['input_dir'] = str(full.relative_to(output))
            entry['capture_contract_and_changes'] = visual_changes(base, full, status)
            changed = entry['capture_contract_and_changes']
            if changed['recapture_ticks']:
                recapture.append({'episode_id': ep, 'ue_entry': str((full / 'render_host_config.json').relative_to(output)), **changed})
            if ep in selected:
                entry['historical_v14_source'] = {'recorded_trajectory_source_ref': selected[ep]['current_trajectory'],
                    'recorded_source_refs': [record['source'] for record in selected[ep]['files']],
                    'recorded_event_firing_status': selected[ep]['story_status'],
                    'unfired_events': selected[ep]['unfired_events'],
                    'event_firing_does_not_certify_scientific_closure': True}
            else:
                entry['actual_l6_receipt_story'] = read(receipt / 'summary.json')['fired_ticks']
                entry['target_projection'] = l6_projection
        elif status['render_payload_changed_ticks']:
            ticks = [tick for tick in status['render_payload_changed_ticks'] if tick % 5 == 0]
            if ticks:
                recapture.append({'episode_id': ep, 'update_scope': 'weather_only', 'recapture_ticks': ticks,
                                  'recapture_frame_count': len(ticks), 'weather_update_path': entry['weather_update_path']})
        if ep == 'L2-1_v2__seed00':
            entry['scientific_closure'] = formal_l2_status()
            entry['scientific_closure_status'] = entry['scientific_closure']['status']
        episodes.append(entry)
        print('Episode inputs assessed', ep, entry['update_scope'], flush=True)
    arm_records = []
    for old_path in arm_sources:
        arm = old_path.parent.parent
        source = arm / 'raw/ue_weather.jsonl'
        raw, normalized = normalized_weather(source, full_episode=False)
        old = list(rows(old_path))
        difference = weather_change(old, normalized, service)
        if [dict((key, value) for key, value in row.items() if key != 'dust') for row in old] != normalized:
            raise ValueError('ARM weather changed outside raw-grounded removal of absent dust: ' + str(arm))
        relative = arm.relative_to(ROOT / 'arms')
        target = output / 'arm_weather_updates' / relative / 'weather_meta.jsonl'
        write_rows(target, normalized)
        arm_records.append({'arm_path': str(relative), 'raw_source_ref': str(source), 'published_source_ref': str(old_path),
            'weather_update_path': str(target.relative_to(output)), 'start_tick': normalized[0]['tick'],
            'end_tick': normalized[-1]['tick'], 'weather_rows': len(normalized),
            'dust_status': 'absent from raw simulator source; omitted from derivative, not measured zero', **difference})
    manifest = {'schema_version': 'p09.current-ue-source-status/v1',
        'scope': 'Recorded UE inputs and weather serialization; no new scientific acceptance or sensor capture',
        'producer': 'Dataset/tools/p09_release.py', 'frozen_inputs_modified': False,
        'reproduction_command': shlex.join([sys.executable, '-B', '-m', 'Dataset.tools.p09_release',
            '--project-root', str(ROOT), '--output-dir', str(output), '--v14-dir', str(v14),
            '--receipt-dir', str(receipt), '--render-source-root', str(args.render_source_root.resolve())]),
        'raw_sensor_captures': 0, 'actual_ue_execution_performed': False,
        'full_capture_input_count': len(full_ids), 'weather_only_episode_count': len(episodes) - len(full_ids),
        'episode_weather_count': len(episodes), 'arm_weather_update_count': len(arm_records),
        'original_episode_raw_weather_available': sum(e['weather']['original_raw_source_available'] for e in episodes),
        'original_episode_raw_weather_unavailable': [e['episode_id'] for e in episodes if not e['weather']['original_raw_source_available']],
        'weather_consumer_evidence': consumer, 'episodes': episodes, 'arm_weather_updates': arm_records,
        'recapture': recapture, 'recapture_episode_count': len(recapture),
        'recapture_planned_frames': sum(e['recapture_frame_count'] for e in recapture),
        'full_input_dense_frames': sum(e['capture_contract_and_changes']['dense_ticks_verified'] for e in episodes if e['episode_id'] in full_ids),
        'full_input_capture_frames': sum(e['capture_contract_and_changes']['capture_ticks_verified'] for e in episodes if e['episode_id'] in full_ids),
        'arm_weather_render_payload_equal_rows': sum(a['render_payload_rows_compared'] for a in arm_records if not a['render_payload_changed_ticks']),
        'elapsed_s': time.perf_counter() - started}
    write_json(output / 'source_status_manifest.json', manifest)
    apply_core_source_updates(output, v14, args.render_source_root.resolve())
    print(orjson.dumps({key: manifest[key] for key in ('full_capture_input_count', 'weather_only_episode_count',
        'episode_weather_count', 'arm_weather_update_count', 'original_episode_raw_weather_available',
        'recapture_episode_count', 'full_input_dense_frames', 'full_input_capture_frames',
        'arm_weather_render_payload_equal_rows', 'raw_sensor_captures', 'elapsed_s')}).decode(), flush=True)


if __name__ == '__main__':
    main()
