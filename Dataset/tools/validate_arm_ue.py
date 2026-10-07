"""Validate exported ARM packages against their measured engine input files."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from Dataset.tools.arm_ue_export import ROOT, bounds, cv, read_json, rows



def validate_label_frame_references(labels, frames_by_tick: dict[int, dict]) -> None:
    for label in labels:
        for field, tick_field in (("frame_id", "tick"), ("source_frame_id", "dispatch_tick")):
            tick = int(label[tick_field])
            if tick not in frames_by_tick:
                raise ValueError(f"Dynamic label {tick_field} has no branch truth frame: {tick}")
            frame = frames_by_tick[tick]
            if label.get(field) != frame["frame_id"] or label["episode_id"] != frame["episode_id"]:
                raise ValueError(f"Dynamic label {field} does not reference its branch frame at tick {tick}")


def validate_arm(arm_dir: Path) -> dict:
    arm_dir = Path(arm_dir).resolve()
    out, raw = arm_dir / 'ue', arm_dir / 'raw'
    start, end = bounds(read_json(arm_dir / 'manifest.json')['window'])
    package = read_json(out / 'scenario_package.json')
    for key, value in package.items():
        if key in {'scenario_id', 'episode_id'}:
            continue
        path = (ROOT / value).resolve()
        if not path.is_relative_to(out) or not path.exists():
            raise ValueError(f'Package reference is absent or outside branch: {key}={value}')
    config = read_json(out / 'render_host_config.json')
    for key in ('episode_dir', 'event_script_path'):
        path = (ROOT / config[key]).resolve()
        if not path.is_relative_to(out) or not path.exists():
            raise ValueError(f'Render host reference outside branch: {key}')
    frames = list(rows(out / 'truth_frames.jsonl'))
    ticks = list(range(start, end + 1))
    if [f['tick'] for f in frames] != ticks:
        raise ValueError('Frame ticks do not match window')
    capture = read_json(out / 'capture_plan.json')
    if capture['capture_ticks'] != [t for t in ticks if t % 5 == 0]:
        raise ValueError('Capture ticks do not match formal grid within window')
    roster = {r['entity_id']: r for r in read_json(raw / 'ue_roster.json')['entities']}
    exported_roster = {r['entity_id']: r for r in read_json(out / 'global_entity_roster.json')['entities']}
    measured = {(r['tick'], r['entity_id']): r for r in rows(raw / 'ue_trajectories.jsonl')}
    exported = {}
    for frame in frames:
        entities = {e['entity_id']: e for e in frame['entities']}
        if len(entities) != len(frame['entities']):
            raise ValueError(f'Duplicate frame entities: {frame["tick"]}')
        for eid in roster:
            row = measured.get((frame['tick'], eid))
            entity = entities.get(eid)
            if (row is None) != (entity is None):
                raise ValueError(f'Authored presence mismatch {frame["tick"]}/{eid}')
            if row is None:
                continue
            pose = entity['truth_pose']
            for field, source in [('position_enu_m', 'pos_enu'), ('velocity_enu_mps', 'vel_mps')]:
                if pose[field] != [round(float(x), 6) for x in row[source]]:
                    raise ValueError(f'Authored {source} mismatch {frame["tick"]}/{eid}')
            if pose['rotation_deg']['yaw_deg'] != round(float(row['yaw_deg']), 6):
                raise ValueError(f'Authored yaw mismatch {frame["tick"]}/{eid}')
            if entity['logical_asset_id'] != row['asset_id']:
                raise ValueError(f'Authored asset mismatch {frame["tick"]}/{eid}')
            if entity['state'] != row['state']:
                raise ValueError(f'Authored state mismatch {frame["tick"]}/{eid}')
            for field in cv.RUNTIME_STATE_FIELDS:
                if entity.get(field) != row.get(field):
                    raise ValueError(f'Authored runtime mismatch {frame["tick"]}/{eid}/{field}')
        for eid, entity in entities.items():
            if eid not in exported_roster:
                raise ValueError(f'Truth actor absent from exported roster: {eid}')
            exported[(frame['tick'], eid)] = cv.truth_entity_to_trajectory_row(frame, entity)
        expected_summary = {'total': len(entities), 'by_category': dict(Counter(e['entity_category'] for e in entities.values()))}
        if frame['roster_summary'] != expected_summary:
            raise ValueError('Frame roster summary mismatch')
    traj = {(r['tick'], r['entity_id']): r for r in rows(out / 'trajectories.jsonl')}
    if traj != exported:
        raise ValueError('Standard trajectories differ from embedded truth poses/state')
    expected_weather = [cv.normalize_weather_row(r, int(r['tick'])) for r in rows(raw / 'ue_weather.jsonl')]
    if list(rows(out / 'weather_meta.jsonl')) != expected_weather:
        raise ValueError('Exported weather differs from measured branch weather')
    for label in rows(out / 'dynamic_labels.jsonl'):
        if 'action_statuses' not in label or 'result_observed' not in label:
            raise ValueError('Dynamic label lacks measured dispatch/result status')
    validate_label_frame_references(list(rows(out / 'dynamic_labels.jsonl')), {int(f['tick']): f for f in frames})
    for filename, source in [('event_trace.jsonl','ue_event_trace.jsonl'), ('event_realization.jsonl','ue_event_realization.jsonl')]:
        if list(rows(out / filename)) != list(rows(raw / source)):
            raise ValueError(f'{filename} differs from branch evidence')
    return {'ok': True, 'window': {'start_tick': start, 'end_tick': end},
            'frame_count': len(frames), 'capture_frame_count': len(capture['capture_ticks']),
            'authored_sample_count': len(measured), 'truth_entity_sample_count': len(exported),
            'roster_entity_count': len(exported_roster), 'capture_executed': False,
            'checks': ['branch_local_package_paths', 'closed_window_coverage', 'formal_capture_grid',
                       'all_authored_presence_pose_state_runtime', 'roster_and_trajectory_consistency',
                       'measured_weather', 'measured_event_trace_and_realization', 'dynamic_label_frame_references']}


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('arm_dir', type=Path)
    print(json.dumps(validate_arm(parser.parse_args().arm_dir), ensure_ascii=False, indent=2))
