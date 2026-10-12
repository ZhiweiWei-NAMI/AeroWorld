"""疏散及实际抵达继续，取消封锁与绕飞，区域不存在且没有解除声明。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import copy
import math
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, path_length, completed_move, measured_pair, run_entry)
from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults

SPEC = {'profile': 'X6_crowd_evacuation_to_airspace_lockdown', 'seed': 'seed02', 'arm_id': 'suppress_direct_consequence', 'window': {'start_tick': 210, 'fork_tick': 220, 'end_tick': 690, 'phase': 'after_state_sample_before_actions'}, 'research_question': '人群到达安全区后是否启用临时限制、执行绕飞并实际停用限制？', 'hypothesis': '疏散及实际抵达继续，取消封锁与绕飞，区域不存在且没有解除声明。', 'window_basis': '人群最后517到达路径终点；绕飞669到达，675宣布停用；实体standdown及控制状态在680正式样本均成立，恢复保持至690。735以后的返航不属于窗口。', 'parameter_model': {'parameter_changes': False, 'motion_model': '沿用源动作分段恒速与ceil(distance/speed*10)整tick计时；不改变速度、航点或语义阈值。', 'state_visibility': '先采样后动作；runtime补丁按声明delay_ticks生效。', 'recovery_model': '正式步长5，当前recovery_hold_samples=3；first_false_capture+(3-1)*5。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

def build_script(source):
    return edit_script(source, remove_events=('airspace_lockdown', 'uav_reroute_lockdown',
        'x6_uav_at_reroute_exit', 'lockdown_cleared', 'lifecycle_landing_uav_x6_monitor'))

def verify_run(config, run):
    safe = event_tick(run, 'crowd_safe_zone_move')
    arrivals = [completed_move(run, 'move_x6_crowd_evacuation_%02d' % i, 690) for i in range(8)]
    for event in ('airspace_lockdown', 'uav_reroute_lockdown', 'lockdown_cleared'):
        absent_event(run, event)
    require(not any(r['entity_id'] == 'nfz_x6_lockdown' for r in run['engine'].trajectory_rows), '取消封锁后禁飞区仍出现')
    control = require_fields(run, 'uav_x6_monitor', {'incident_state.requires_reroute': False,
        'control_state.reroute_active': False}, 515, 690)
    flight = path_length(run, 'uav_x6_monitor', 515, 690)
    require(flight == 0.0, '未封锁却执行了绕飞')
    return {'all_safe_event_tick': safe, 'crowd_arrivals': arrivals, 'region_spawned': False,
            'control': control, 'post_lockdown_time_uav_path_m': flight, 'clearance_claimed': False}

if __name__ == '__main__':
    run_entry(Path(__file__))
