"""Build current L6 objective labels from actual R2 world and UE observer inputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

EPISODE = 'L6-5_v1__seed00'
TARGETS = {'uav_digital_l6_5_v1', 'gcs_anchor_l6_5_v1'}
PROJECT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    from Dataset.semantic_truth.provenance import without_integrity_metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(without_integrity_metadata(value), ensure_ascii=False, indent=2) + '\n')


def write_rows(path, records):
    from Dataset.semantic_truth.provenance import without_integrity_metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        for row in records:
            handle.write(json.dumps(without_integrity_metadata(row), ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--receipt-dir', type=Path)
    parser.add_argument('--capture-dir', type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = (args.output_dir or project / 'aw_data/objective_semantic_truth' / EPISODE).resolve()
    receipt = (args.receipt_dir or project / 'Dataset/episodes' / EPISODE).resolve()
    capture = (args.capture_dir or project / 'aw_data/render_ready_episodes_capture_filtered' / EPISODE).resolve()
    for source in (receipt, capture):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('Objective output overlaps source: ' + str(source))
    print('Objective projection PID', os.getpid(), flush=True)
    started = time.perf_counter()
    from Dataset.semantic_truth import objective_pipeline as pipeline, episode_sources, charging_supplement, l0_supplement, input_adapter
    from Dataset.semantic_simulation import domain_state, predicate_state_computers
    inputs = project / 'Dataset/episodes' / EPISODE
    inputs.mkdir(parents=True, exist_ok=True)
    # Bind every existing resolver to the same adopted episode source root.
    episode_sources.PROJECT_ROOT = project
    pipeline.PROJECT_ROOT = project
    charging_supplement.PROJECT_ROOT = project
    domain_state.PROJECT_ROOT = project
    l0_supplement.PROJECT_ROOT = project
    input_adapter.PROJECT_ROOT = project
    predicate_state_computers.PROJECT_ROOT = project
    predicate_state_computers.GLOBAL_UAV_TASK_PLAN_PATH = project / 'aw_data/uav_outputs/donghu_uav_flow_270s/uav_task_plan.json'
    for name in ('scene_occupancy_manifest.json', 'scene_setup.json', 'truth_frames.jsonl', 'weather_meta.jsonl'):
        shutil.copyfile(capture / name, inputs / name)
    geometry = input_adapter.scene_setup_geometry(read(inputs / 'scene_setup.json'), str(inputs / 'scene_setup.json'))
    write(inputs / 'semantic_static_geometry.json', {'entities': list(geometry.values())})
    source_targets = [row for row in rows(receipt / 'actual_execution_trajectories.jsonl') if row['entity_id'] in TARGETS]
    for eid in TARGETS:
        if [row['tick'] for row in source_targets if row['entity_id'] == eid] != list(range(901)):
            raise ValueError('Actual R2 target must cover 0..900: ' + eid)
    background = [row for row in rows(capture / 'trajectories.jsonl') if row['entity_id'] not in TARGETS]
    world = sorted(background + source_targets, key=lambda row: (row['tick'], row['entity_id']))
    write_rows(inputs / 'trajectories.jsonl', world)
    roster = read(capture / 'global_entity_roster.json')
    for entry in roster['entities']:
        if entry['entity_id'] in TARGETS:
            initial = next(row for row in source_targets if row['entity_id'] == entry['entity_id'] and row['tick'] == 0)
            entry['initial_yaw_deg'] = initial['yaw_deg']
            entry['initial_position_enu_m'] = initial['pos_enu']
    write(inputs / 'global_entity_roster.json', roster)
    manifest = read(capture / 'episode_manifest.json')
    manifest['generation']['source_episode_dir'] = str(inputs.relative_to(project))
    manifest['generation']['trajectory_source'] = 'actual_R2_full_target_scope_and_frozen_capture_background'
    manifest['generation']['trajectory_contract'] = 'target_901_ticks_observer_eligibility_preserved_separately'
    manifest['source_scene_setup_path'] = str(inputs / 'scene_setup.json')
    authority = inputs
    shutil.copyfile(capture / 'event_script.json', authority / 'event_script.json')
    manifest['source_event_script_path'] = str(authority / 'event_script.json')
    sumo = Path(manifest['sumo_traffic']['source']['frames'])
    if not sumo.is_absolute():
        sumo = project / sumo
    if sumo.resolve() != (inputs / 'sumo_traffic_frames.jsonl').resolve():
        shutil.copyfile(sumo, inputs / 'sumo_traffic_frames.jsonl')
    manifest['sumo_traffic']['source']['frames'] = str((inputs / 'sumo_traffic_frames.jsonl').relative_to(project))
    manifest['current_capture_input_ref'] = str(capture)
    manifest['actual_R2_execution_source_ref'] = str(receipt)
    write(inputs / 'episode_manifest.json', manifest)
    sources = inputs
    for name in ('executed_actions.jsonl', 'event_trace.jsonl', 'audit.jsonl', 'command_receipts.json',
                 'physical_evidence.json', 'recovery_application.json'):
        if (sources / name).resolve() == (receipt / name).resolve():
            continue
        if name.endswith('.jsonl'):
            write_rows(sources / name, rows(receipt / name))
        else:
            write(sources / name, read(receipt / name))
    print('Actual assembled world rows', len(world), 'target rows', len(source_targets),
          'observer source', capture / 'truth_frames.jsonl', flush=True)
    destination = output
    print('Starting existing build_episode_objective_artifacts', inputs, flush=True)
    artifacts = pipeline.build_episode_objective_artifacts(inputs, destination)
    pipeline.write_episode_outputs(artifacts)
    binding = {'episode_id': EPISODE, 'objective_dir': str(destination), 'world_input_dir': str(inputs),
        'capture_input_ref': str(capture), 'actual_execution_source_ref': str(receipt),
        'execution_source_dir': str(sources), 'target_full_scope_rows': len(source_targets),
        'frozen_background_rows': len(background), 'world_trajectory_rows': len(world),
        'observer_frame_source_ref': str(capture / 'truth_frames.jsonl'),
        'event_input_policy': 'Actual actions/events are execution evidence; typed objective events are derived from current states, never copied from authored event labels',
        'closure': artifacts.closure, 'artifact_files': sorted(artifacts.files),
        'elapsed_s': time.perf_counter() - started,
        'native_reexecuted': False, 'UE_capture_performed': False, 'capture_inputs_modified': False}
    write(output / 'current_binding.json', binding)
    print(json.dumps({'closure': artifacts.closure, 'elapsed_s': binding['elapsed_s'],
                      'artifact_file_count': len(artifacts.files)}), flush=True)


if __name__ == '__main__':
    main()
