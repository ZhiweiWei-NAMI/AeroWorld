"""基站故障与配额失效真实发生；本地预留规则提前确定优先级，双机不再同时请求同一停机坪。

上游与 factual 完全一致（station_unavailable、allocation_failed、allocation_stale）。
分叉点在中间机制：factual 里两机在配额未知下同时请求同一 pad，引发争用、仲裁与
改航；本臂在双机出发前由本地缓存按确定性优先级完成预留（winner = priority 机），
second 机改往东南等待点保持。预留期只有 priority 请求，争用、仲裁和改航链不发生。
服务按声明时刻恢复；second 在窗口内返航着陆，priority 保持原进场终点，源返航在窗口外。
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, completed_move, run_entry)

SPEC = {'profile': 'X5_comm_failure_to_pad_contention', 'seed': 'seed00', 'arm_id': 'fallback_pad_reservation',
        'window': {'start_tick': 210, 'fork_tick': 220, 'end_tick': 670, 'phase': 'after_state_sample_before_actions'},
        'research_question': '配额服务失效时，本地预留规则能否在双机进场前消除停机坪争用？',
        'hypothesis': '基站故障与配额失效真实发生；本地预留于双机出发前确定性授予 priority 机，second 机在等待点保持，请求计数恒为 1，争用/仲裁/改航链不发生。',
        'window_basis': 'factual 双请求于 285 正式样本可见，仲裁命令 560、改航 590、基站恢复 655。本地预留取 275，双机 280 出发；second 改到独立等待点，555 开始返航并在窗口内着陆；priority 保持原进场终点，源返航位于窗口外。',
        'parameter_model': {'parameter_changes': False, 'motion_model': '沿用源动作分段恒速与 ceil(distance/speed*10) 整 tick 计时；second 机改道速度与源出发一致 5.0 m/s。', 'state_visibility': '先采样后动作；runtime 补丁按声明 delay_ticks 生效。', 'recovery_model': '正式步长 5，recovery_hold_samples=3。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

WAIT_WP = [[6634.525, 6505.436, 31.0], [6650.0, 6470.0, 45.0]]
SECOND_LAND_WP = [[6650.0, 6470.0, 45.0], [6650.819, 6498.378, 27.5], [6659.262, 6513.999, 24.0], [6659.262, 6513.999, 3.0], [6659.262, 6513.999, 0.8]]


def build_script(source):
    import copy as _copy
    pruned = _copy.deepcopy(source)
    contention = next(e for e in pruned['events'] if e['event_id'] == 'dual_uav_pad_contention')
    keep = {'move_x5_uav_a_to_pad', 'move_x5_uav_b_to_pad_hold'}
    contention['actions'] = [a for a in contention['actions'] if a['action_id'] in keep]
    contention['log_event'].update(
        title='The priority UAV approaches its reserved pad while the second UAV moves to the waiting point',
        intent='local_pad_reservation_execution', intent_stage='03.local_pad_reservation_execution',
        causal_predecessor_intent='local_pad_reservation')
    script = edit_script(pruned,
        remove_events=('pad_priority_arbitration', 'second_uav_reroute'),
        replace_triggers={'trig_after_second_uav_reroute_delay_63':
            {'trigger_id': 'trig_after_second_uav_reroute_delay_63', 'type': 'tick', 'tick': 655}})
    script = edit_script(script,
        replace_actions={'move_x5_uav_b_to_pad_hold': {
            'type': 'move_entity', 'action_id': 'move_x5_uav_b_to_pad_hold',
            'entity_id': 'uav_x5_second', 'velocity_mps': 5.0,
            'activity_type': 'cruise', 'post_activity_type': 'holding',
            'waypoints_enu_m': WAIT_WP}})
    restored = next(e for e in script['events'] if e['event_id'] == 'station_recovered')
    del restored['log_event']['causal_predecessor_intent']
    return edit_script(script,
        add_triggers=[{'trigger_id': 'x_local_pad_reservation', 'type': 'tick', 'tick': 275},
                      {'trigger_id': 'x_second_return', 'type': 'tick', 'tick': 555}],
        add_events=[
            {'event_id': 'x_local_pad_reservation', 'trigger_ref': 'x_local_pad_reservation',
             'priority': 100, 'max_fire_count': 1,
             'actions': [{'type': 'set_runtime_state', 'action_id': 'x5_local_reservation',
                'entity_id': 'pad_x5_primary', 'delay_ticks': 0,
                'state_patch': {'facility_state': {'arbitration_winner_id': 'uav_x5_priority',
                    'priority_granted': True, 'priority_granted_to': 'uav_x5_priority',
                    'request_count': 1, 'requester_ids': ['uav_x5_priority']}}}],
             'log_event': {'topic': 'x_local_pad_reservation', 'category': 'facility',
                 'title': 'Local reservation cache grants priority before departure',
                 'intent': 'local_pad_reservation', 'intent_stage': '03.local_pad_reservation',
                 'severity': 'info', 'target_ids': ['uav_x5_priority']}},
            {'event_id': 'x_second_return', 'trigger_ref': 'x_second_return',
             'priority': 100, 'max_fire_count': 1,
             'actions': [{'type': 'move_entity', 'action_id': 'x5_second_return_from_wait',
                'entity_id': 'uav_x5_second', 'velocity_mps': 7.579,
                'activity_type': 'descending', 'post_activity_type': 'landed',
                'waypoints_enu_m': SECOND_LAND_WP,
                'terminal_feasibility': {'model': 'constant_speed_path_with_explicit_landing_reference_v2',
                    'landing_reference_enu_m': [6659.262, 6513.999, 0.8],
                    'touchdown_altitude_tolerance_m': 1.5,
                    'touchdown_dwell_ticks': 5}}],
             'log_event': {'topic': 'x_second_return', 'category': 'lifecycle',
                 'title': 'Second UAV returns from the waiting point',
                 'severity': 'info', 'target_ids': []}}])


def verify_run(config, run):
    event_tick(run, 'station_failure')
    event_tick(run, 'backup_pad_unavailable')
    require_fields(run, 'uav_x5_priority',
                   {'communication_state.station_unavailable': True}, 260, 270)
    require_fields(run, 'pad_x5_backup',
                   {'facility_state.allocation_failed': True}, 260, 270)
    require_fields(run, 'pad_x5_primary',
                   {'facility_state.allocation_stale': True}, 260, 270)
    event_tick(run, 'x_local_pad_reservation')
    require_fields(run, 'pad_x5_primary',
                   {'facility_state.request_count': 1}, 280, 655)
    pad_rows = samples(run, 'pad_x5_primary', 280, 655)
    require(all(row['facility_state'].get('request_count') == 1 for row in pad_rows),
            '争用期出现双请求计数')
    for event in ('pad_priority_arbitration', 'second_uav_reroute'):
        absent_event(run, event)
    wait = completed_move(run, 'move_x5_uav_b_to_pad_hold', 670)
    require(wait['target_enu_m'] == WAIT_WP[-1], 'second 机未抵达等待点')
    back = completed_move(run, 'x5_second_return_from_wait', 670)
    arrival = completed_move(run, 'move_x5_uav_a_to_pad', 670)
    event_tick(run, 'station_recovered')
    return {'upstream': {'station_failure': 220, 'allocation_failed': 250},
            'block': {'local_reservation_tick': 275, 'request_count_window': [280, 655],
                      'second_wait_point': WAIT_WP[-1]},
            'arrivals': {'priority_pad': arrival, 'second_wait': wait,
                         'second_return': back},
            'absent_chain': ['pad_priority_arbitration', 'second_uav_reroute']}


if __name__ == '__main__':
    run_entry(Path(__file__))
