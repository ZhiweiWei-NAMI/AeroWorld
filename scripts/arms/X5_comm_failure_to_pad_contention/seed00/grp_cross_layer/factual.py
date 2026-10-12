"""双机请求后仲裁和改航实际完成；通信恢复和主停机坪请求关闭，备用停机坪仍故障。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import copy
import math
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, path_length, completed_move, measured_pair, run_entry)
from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults

SPEC = {'profile': 'X5_comm_failure_to_pad_contention', 'seed': 'seed00', 'arm_id': 'factual', 'window': {'start_tick': 210, 'fork_tick': 220, 'end_tick': 670, 'phase': 'after_state_sample_before_actions'}, 'research_question': '通信失效与备用停机坪故障后，双机请求、仲裁和改航是否发生？', 'hypothesis': '双机请求后仲裁和改航实际完成；通信恢复和主停机坪请求关闭，备用停机坪仍故障。', 'window_basis': '优先机570到达，第二架647完成改航；655恢复通信，最后状态补丁660生效，恢复保持660/665/670。备用停机坪故障不随通信恢复自动清除；685以后返航不属于本窗口。', 'parameter_model': {'parameter_changes': False, 'motion_model': '沿用源动作分段恒速与ceil(distance/speed*10)整tick计时；不改变速度、航点或语义阈值。', 'state_visibility': '先采样后动作；runtime补丁按声明delay_ticks生效。', 'recovery_model': '正式步长5，当前recovery_hold_samples=3；first_false_capture+(3-1)*5。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

def build_script(source):
    return edit_script(source)

def verify_run(config, run):
    contention, arbitration, reroute = [event_tick(run, e) for e in
        ('dual_uav_pad_contention', 'pad_priority_arbitration', 'second_uav_reroute')]
    require(contention < arbitration < reroute, '请求、仲裁、改航顺序不一致')
    requests = require_fields(run, 'pad_x5_primary', {'facility_state.request_count': 2}, 285, 594)
    priority = completed_move(run, 'move_x5_uav_a_to_pad', 670)
    diverted = completed_move(run, 'move_x5_uav_b_reroute', 670)
    link = [require_fields(run, eid, {'communication_state.station_unavailable': False}, 660, 670)
            for eid in ('uav_x5_priority', 'uav_x5_second')]
    primary = require_fields(run, 'pad_x5_primary', {'facility_state.request_count': 0,
        'facility_state.arbitration_active': False, 'facility_state.allocation_stale': False}, 660, 670)
    backup = require_fields(run, 'pad_x5_backup', {'facility_state.allocation_failed': True}, 255, 670)
    return {'contention_tick': contention, 'arbitration_tick': arbitration, 'dual_requests': requests,
            'priority_arrival': priority, 'reroute_arrival': diverted, 'restored_communication': link,
            'closed_primary_allocation': primary, 'unresolved_backup_failure': backup,
            'scope': 'station_recovered只关闭通信故障；备用停机坪仍故障，不能报告全设施恢复'}

if __name__ == '__main__':
    run_entry(Path(__file__))
