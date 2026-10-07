"""Apply source-bound global UAV yaw corrections to adopted formal inputs.

Each invocation selects one episode in the existing domain source index.
It performs no simulation or sensor capture and creates no staging dataset.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Dataset/tools'))
from Dataset.tools import convert_to_render_ready as conv

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'aw_data/render_ready_episodes_capture_filtered'
DENSE_TICKS = list(range(901))


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


def main():
    """Apply source corrections to one explicitly selected, adopted input."""
    global ROOT, BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--episode', required=True)
    args = parser.parse_args()
    ROOT = args.project_root.resolve()
    BASE = ROOT / 'aw_data/render_ready_episodes_capture_filtered'
    index_path = ROOT / 'aw_data/domain_state_supplement/source_index.json'
    manifest = read(index_path)
    entries = [entry for entry in manifest['episodes'] if entry['episode_id'] == args.episode]
    if len(entries) != 1:
        raise ValueError('Current source index must uniquely bind episode: ' + args.episode)
    target = BASE / args.episode
    native = read(target / 'episode_manifest.json')
    source_dir = Path(native['uav_global_flow']['source']['output_dir'])
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    dataset, index = global_yaw_source_index(source_dir)
    candidates = global_yaw_candidates(conv.read_source_roster(target / 'global_entity_roster.json'),
        index, 0, 900, native['uav_global_flow']['runtime_spatial_crop']['bbox_enu_m'])
    changes, ticks, rejected = global_yaw_changes(target / 'truth_frames.jsonl', candidates)
    if ticks != DENSE_TICKS:
        raise ValueError('Current episode dense clock differs: ' + args.episode)
    if changes:
        patch_global_yaw_pose_files(target, changes)
    entries[0]['global_uav_yaw_source'] = {
        'policy': 'recorded_global_yaw_when_horizontal_motion_norm_le_1e-5',
        'source_frames_ref': str(dataset.frames_path), 'source_task_plan_ref': str(dataset.task_plan_path),
        'source_position_rejections': rejected,
        'capture_inputs_path': str(target), 'position_modified': False,
        'UE_capture_performed': False}
    write_json(index_path, manifest)
    print(orjson.dumps({'episode_id': args.episode, 'dense_actor_rows_changed': len(changes),
        'position_modified': False, 'UE_capture_performed': False}).decode(), flush=True)


if __name__ == '__main__':
    main()
