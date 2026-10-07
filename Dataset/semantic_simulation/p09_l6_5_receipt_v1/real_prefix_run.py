"""Run the real L6 R2 abnormal/recovery/landing receipt-coupled story.

Native ns-3 receipts and actual engine state are iterated together. Timing is
read from fired events and applied runtime patches; the original source script,
seed, radio parameters and action-specific grants stay authoritative.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import traceback
from pathlib import Path

from Dataset.semantic_simulation.ns3_episode.linked_engine import execute_linked
from Dataset.semantic_simulation.ns3_episode.linked_transport import episode_from_engine
from Dataset.semantic_simulation.ns3_episode.provider_adapter import ProviderAdapter
from Dataset.semantic_simulation.p09_l6_5_receipt_v1 import adapter as A
from Dataset.semantic_simulation.p09_l6_5_receipt_v1 import single_action_prefix as S
from Dataset.tools.arm_ue_export import write_engine_inputs

ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / 'Dataset/scenarios/L6_digital_layer/failure/L6-5_v1/scene_setup.json'
SCRIPT = ROOT / 'Dataset/scenarios/L6_digital_layer/failure/L6-5_v1/event_script.json'
RADIO_REF = ROOT / 'design/p01_ns3/radio_reference_R1.json'
R1_DIR = ROOT / 'design/p09/mechanism_completion_v1/l6_5_command_receipt_revision/real_prefix_run_r1'
R1_TRAJECTORY = R1_DIR / 'trajectories.jsonl'
PUBLISHED = ROOT / 'aw_data/render_ready_episodes_capture_filtered/L6-5_v1__seed00/trajectories.jsonl'
OUTPUT_DIR = Path('/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/l6_receipt_r2')
PROGRESS = OUTPUT_DIR.parent / 'current_dispatch/l6_progress.json'
RUN_COMMAND = shlex.join(sys.orig_argv)
MAX_ITERATIONS = 6
PROFILE = {'episode_id': A.EPISODE_ID, 'source_script': str(SCRIPT),
           'actor_ids': [A.UAV], 'primary_station': A.GCS}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_safe(value):
    if isinstance(value, dict):
        return {(key if isinstance(key, (str, int, float, bool, type(None))) else repr(key)):
                json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def write_json(path, payload):
    Path(path).write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), separators=(',', ':'), allow_nan=False) + '\n')


def progress(status, **fields):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    row = {'task_id': 'P09-M03-L6R2', 'status': status, 'pid': os.getpid(),
           'command': RUN_COMMAND, 'output_dir': str(OUTPUT_DIR), **fields}
    write_json(PROGRESS, row)
    print(json.dumps(row), flush=True)


def fired_tick_map(run):
    return {key: state.last_fired_tick for key, state in run['interpreter'].event_states.items() if state.fired}


def build_bindings():
    bindings = {}
    for action, role in ((A.MOVEMENT_ACTION, A.MOVEMENT_ROLE),
                         (A.NOTIFICATION_ACTION, A.NOTIFICATION_ROLE),
                         (A.LANDING_ACTION, A.MOVEMENT_ROLE)):
        binding = A.bind_command(f'{A.EPISODE_ID}:command:{action}', action, role,
            {'claimed_roles': [role]}, source_owner=A.GCS, source_epoch=0,
            receiver_owner=A.UAV, receiver_epoch=0, control_epoch=0)
        if action == A.NOTIFICATION_ACTION:
            binding['incident_movement_epoch'] = A.revoked_movement_epoch()
        bindings[action] = binding
    return bindings


def schedule_from_engine(run, script):
    fires = fired_tick_map(run)
    actions = {a['action_id']: a for a in run['engine'].executed_actions}
    events = {e['event_id']: e for e in script['events']}
    triggers = {t['trigger_id']: t for t in script['triggers']}
    secure = fires[S.RECOVERY_EVENT_ID]
    gcs_applied = actions[S.GCS_SECURE_ACTION]['result']['effective_tick']
    landing_due = secure + triggers[events[S.LANDING_EVENT_ID]['trigger_ref']]['delay_ticks']
    return {A.MOVEMENT_ACTION: {'send_tick': A.STORY['abnormal_command_event_tick'],
                               'evidence_tick': A.STORY['gcs_intrusion_event_tick']},
            A.NOTIFICATION_ACTION: {'send_tick': gcs_applied, 'evidence_tick': gcs_applied},
            A.LANDING_ACTION: {'send_tick': landing_due, 'evidence_tick': gcs_applied}}


def messages_and_config(episode, bindings, schedule):
    messages = {action: A.build_message(episode, binding, schedule[action]['send_tick'],
                     schedule[action]['evidence_tick'] * A.STEP_NS)
                for action, binding in bindings.items()}
    fields = ('flow_id', 'link_id', 'src_owner', 'life_epoch', 'dst_owner',
              'dst_life_epoch', 'payload_bytes', 'period_ns', 'start_ns', 'end_ns')
    config = {'network_profile': 'timed_flow', 'network': {'traffic_windows': [
        {key: message['flow'][key] for key in fields} for message in messages.values()]},
        'seed': 1, 'run': 1, 'radio': load_json(RADIO_REF), 'radio_actions': [],
        'radio_config_source_ref': str(RADIO_REF)}
    return messages, config


def converge(scene, script, bindings, history):
    ungated = execute_linked(scene, script, SCRIPT, A.EPISODE_ID, seed=0)
    source_run = ungated
    previous = None
    # Provider compilation and radio contract resolution stay out of per-tick gates.
    with ProviderAdapter() as provider:
        for iteration in range(1, MAX_ITERATIONS + 1):
            episode = episode_from_engine(PROFILE, source_run['engine'])
            schedule = schedule_from_engine(source_run, script)
            messages, config = messages_and_config(episode, bindings, schedule)
            progress('native_running', iteration=iteration, schedule=schedule)
            started = time.perf_counter()
            network = provider.run_episode(episode, config, OUTPUT_DIR)
            native_s = time.perf_counter() - started
            receipts = {action: A.extract_receipts(episode, message, network)
                        for action, message in messages.items()}
            run = S.wire_receipt_story(scene, script, SCRIPT, A.EPISODE_ID, 0,
                bindings=bindings, receipts=receipts, messages=messages)
            state = run['story_state']
            signature = {'schedule': schedule, 'rx_ns': {action:
                None if record['accepted'] is None else record['accepted']['accepted_ns']
                for action, record in receipts.items()}, 'fired_ticks': fired_tick_map(run),
                'recovery_applied_tick': state['uav_recovery_applied_tick'],
                'native_packet_timing': [{key: p[key] for key in
                    ('flow_id', 'first_tx_ns', 'rx_ns')} for p in network['network_packets']]}
            history.append({'iteration': iteration, **signature, 'native_wall_time_s': native_s})
            write_json(OUTPUT_DIR / 'iteration_history.json', history)
            progress('iteration_completed', iteration=iteration, rx_ns=signature['rx_ns'],
                     fired_ticks=signature['fired_ticks'], recovery_applied_tick=state['uav_recovery_applied_tick'])
            if previous == signature:
                return {'run': run, 'ungated': ungated, 'network': network, 'receipts': receipts,
                        'messages': messages, 'config': config, 'episode': episode,
                        'iteration': iteration, 'native_wall_time_s': native_s}
            previous, source_run = signature, run
    raise RuntimeError(f'actual receipt/mobility coupling did not converge in {MAX_ITERATIONS} iterations')


def pose_differences(rows, source_path, *, capture_only=False):
    actual = {(r['entity_id'], r['tick']): (r['pos_enu'], r['vel_mps'])
              for r in rows if r['entity_id'] in (A.UAV, A.GCS)
              and (not capture_only or r['tick'] % 5 == 0)}
    first, counts = {}, {A.UAV: 0, A.GCS: 0}
    first_position, first_velocity = {}, {}
    positions, velocity_only = {A.UAV: 0, A.GCS: 0}, {A.UAV: 0, A.GCS: 0}
    velocity_only_max_abs_delta = {A.UAV: 0.0, A.GCS: 0.0}
    velocity_only_ticks = {A.UAV: [], A.GCS: []}
    matched = {A.UAV: 0, A.GCS: 0}
    with Path(source_path).open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            key = row['entity_id'], row['tick']
            if key in actual:
                eid = row['entity_id']
                matched[eid] += 1
                position_changed = actual[key][0] != row['pos_enu']
                velocity_changed = actual[key][1] != row['vel_mps']
                if position_changed:
                    positions[eid] += 1
                elif velocity_changed:
                    velocity_only[eid] += 1
                    velocity_only_ticks[eid].append(row['tick'])
                    velocity_only_max_abs_delta[eid] = max(velocity_only_max_abs_delta[eid],
                        max(abs(a - b) for a, b in zip(actual[key][1], row['vel_mps'])))
                if actual[key][0] != row['pos_enu'] and row['entity_id'] not in first_position:
                    first_position[row['entity_id']] = row['tick']
                if actual[key][1] != row['vel_mps'] and row['entity_id'] not in first_velocity:
                    first_velocity[row['entity_id']] = row['tick']
            if key in actual and actual[key] != (row['pos_enu'], row['vel_mps']):
                counts[row['entity_id']] += 1
                if row['entity_id'] not in first:
                    first[row['entity_id']] = row['tick']
    return {'source_ref': str(source_path), 'pose_fields': ['pos_enu', 'vel_mps'],
            'capture_only': capture_only, 'first_changed_tick': first, 'changed_rows': counts,
            'first_position_changed_tick': first_position,
            'first_velocity_changed_tick': first_velocity,
            'matched_source_rows': matched, 'position_changed_rows': positions,
            'velocity_only_changed_rows': velocity_only,
            'velocity_only_changed_ticks': velocity_only_ticks,
            'velocity_only_max_abs_component_delta_mps': velocity_only_max_abs_delta,
            'comparison_semantics': 'Exact source values; velocity-only differences do not establish a position change'}


def produce(outcome, bindings, script, history, started):
    run, network, receipts = outcome['run'], outcome['network'], outcome['receipts']
    engine, state = run['engine'], run['story_state']
    for name, payload in (('command_input', {'bindings': bindings, 'messages': outcome['messages'],
            'native_config': outcome['config'], 'source_scene': str(SCENE), 'source_script': str(SCRIPT)}),
            ('command_receipts', receipts), ('network_backend', network['backend']),
            ('gate_evidence', run['gate_evidence']), ('prefix_refusals', run['prefix_refusals']),
            ('recovery_application', {'state': state, 'observations': run['recovery_observations']}),
            ('native_episode', outcome['episode'])):
        write_json(OUTPUT_DIR / (name + '.json'), payload)
    for name, rows in (('network_packets', network['network_packets']),
            ('network_events', network['network_events']), ('audit', run['audit']),
            ('trajectories', engine.trajectory_rows), ('weather', engine.weather_rows),
            ('executed_actions', engine.executed_actions), ('event_trace', run['interpreter'].get_event_log())):
        write_jsonl(OUTPUT_DIR / (name + '.jsonl'), rows)
    write_jsonl(OUTPUT_DIR / 'raw' / 'actions.jsonl', run['audit'])
    ue_inputs = write_engine_inputs(OUTPUT_DIR, engine, run['interpreter'],
        {'start_tick': 0, 'end_tick': engine.duration_ticks}, script=script)
    successes = {r['action']['action_id']: r for r in run['audit'] if r['result']['status'] == 'ok'}
    physical = {}
    for action in (A.MOVEMENT_ACTION, A.LANDING_ACTION):
        if action not in successes:
            raise RuntimeError(f'actual handler execution absent: {action}')
        row = successes[action]
        accepted = receipts[action]['accepted']
        if accepted is None or row['tick'] * A.STEP_NS <= accepted['accepted_ns']:
            raise RuntimeError(f'action was not executed after its actual RX: {action}')
        physical[action] = {'actual_dispatch_tick': row['tick'], 'accepted_rx_ns': accepted['accepted_ns'],
            'executed_action': row['action'], 'handler_result': row['result'],
            'actual_motion_schedule': row['motion_schedule'],
            'motion': A.summarize_motion_rows(engine.trajectory_rows,
                      row['tick'] * A.STEP_NS, (row['tick'] + 10) * A.STEP_NS)}
    if state['uav_recovery_applied_tick'] is None or state['movement_epoch0_active']:
        raise RuntimeError('recovery did not apply or abnormal movement authority was rearmed')
    uav_rows = [r for r in engine.trajectory_rows if r['entity_id'] == A.UAV]
    landing_tick = successes[A.LANDING_ACTION]['tick']
    landed = [r for r in uav_rows if r['tick'] > landing_tick and r['state'] == 'landed']
    stopped = [r for r in landed if r['vel_mps'] == [0.0, 0.0, 0.0]]
    if not landed or not stopped:
        raise RuntimeError('actual landing and zero-velocity terminal rows were not observed')
    terminal = {'first_landed_row': landed[0], 'first_landed_zero_velocity_row': stopped[0],
                'final_uav_row': uav_rows[-1]}
    write_json(OUTPUT_DIR / 'physical_evidence.json', {'movement': physical, 'terminal': terminal})
    differences = {'versus_r1': pose_differences(engine.trajectory_rows, R1_TRAJECTORY),
                   'versus_published_dense': pose_differences(engine.trajectory_rows, PUBLISHED),
                   'versus_published_capture': pose_differences(engine.trajectory_rows, PUBLISHED, capture_only=True)}
    write_json(OUTPUT_DIR / 'pose_comparison.json', differences)
    native_missing = [p for p in network['network_packets'] if p['rx_ns'] is None]
    return {'task_id': 'P09-M03-L6R2', 'episode_id': A.EPISODE_ID, 'stop_reason': 'converged',
        'runtime_identity': A.runtime_identity(), 'converged_iteration': outcome['iteration'],
        'total_elapsed_s': time.perf_counter() - started, 'iteration_history': history,
        'fired_ticks': fired_tick_map(run), 'recovery_application': state,
        'physical_evidence': physical, 'terminal': terminal, 'pose_comparison': differences,
        'ue_engine_inputs': ue_inputs, 'native_missing_rx_packets': len(native_missing),
        'native_loss_negative_observed': bool(native_missing),
        'workflow': 'authorized implementation executed directly; result assessment follows actual run'}


def main():
    global OUTPUT_DIR, PROGRESS, R1_TRAJECTORY, PUBLISHED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                        help='R2 output directory; completed outputs cannot be overwritten')
    parser.add_argument('--r1-trajectory', type=Path, default=R1_TRAJECTORY,
                        help='Required immutable R1 trajectories.jsonl for comparison')
    parser.add_argument('--published-trajectory', type=Path, default=PUBLISHED,
                        help='Required published trajectories.jsonl for dense/capture comparison')
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    R1_TRAJECTORY = args.r1_trajectory.resolve()
    PUBLISHED = args.published_trajectory.resolve()
    PROGRESS = OUTPUT_DIR.parent / 'current_dispatch/l6_progress.json'
    required_inputs = (SCENE, SCRIPT, RADIO_REF, R1_TRAJECTORY, PUBLISHED)
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required input files missing: ' + ', '.join(missing))
    if (OUTPUT_DIR / 'summary.json').exists():
        raise FileExistsError(f'refusing completed output: {OUTPUT_DIR}')
    if (OUTPUT_DIR / 'trajectories.jsonl').resolve() in (R1_TRAJECTORY, PUBLISHED):
        raise ValueError('Output trajectories.jsonl collides with a required comparison input')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'raw').mkdir(exist_ok=True)
    started, history = time.perf_counter(), []
    progress('launched')
    try:
        scene, script, bindings = load_json(SCENE), load_json(SCRIPT), build_bindings()
        outcome = converge(scene, script, bindings, history)
        summary = produce(outcome, bindings, script, history, started)
        write_json(OUTPUT_DIR / 'summary.json', summary)
        progress('completed', summary_ref=str(OUTPUT_DIR / 'summary.json'),
                 fired_ticks=summary['fired_ticks'],
                 uav_recovery_applied_tick=summary['recovery_application']['uav_recovery_applied_tick'],
                 first_landed_tick=summary['terminal']['first_landed_row']['tick'],
                 first_zero_velocity_tick=summary['terminal']['first_landed_zero_velocity_row']['tick'])
    except BaseException:
        write_json(OUTPUT_DIR / 'failure.json', {'command': RUN_COMMAND,
            'elapsed_s': time.perf_counter() - started, 'iteration_history': history,
            'traceback': traceback.format_exc()})
        progress('failed', failure_ref=str(OUTPUT_DIR / 'failure.json'))
        raise


if __name__ == '__main__':
    main()
