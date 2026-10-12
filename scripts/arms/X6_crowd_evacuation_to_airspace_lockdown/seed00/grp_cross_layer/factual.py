"""安全区抵达先于封锁；绕飞完成后限制转standdown且控制标志关闭。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import copy
import math
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, path_length, completed_move, measured_pair, run_entry)
from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults

SPEC = {'profile': 'X6_crowd_evacuation_to_airspace_lockdown', 'seed': 'seed00', 'arm_id': 'factual', 'window': {'start_tick': 210, 'fork_tick': 220, 'end_tick': 690, 'phase': 'after_state_sample_before_actions'}, 'research_question': '人群到达安全区后是否启用临时限制、执行绕飞并实际停用限制？', 'hypothesis': '安全区抵达先于封锁；绕飞完成后限制转standdown且控制标志关闭。', 'window_basis': '人群最后517到达路径终点；绕飞669到达，675宣布停用；实体standdown及控制状态在680正式样本均成立，恢复保持至690。735以后的返航不属于窗口。', 'parameter_model': {'parameter_changes': False, 'motion_model': '沿用源动作分段恒速与ceil(distance/speed*10)整tick计时；不改变速度、航点或语义阈值。', 'state_visibility': '先采样后动作；runtime补丁按声明delay_ticks生效。', 'recovery_model': '正式步长5，当前recovery_hold_samples=3；first_false_capture+(3-1)*5。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

def build_script(source):
    script = edit_script(source)
    event = next(e for e in script['events'] if e['event_id'] == 'uav_reroute_lockdown')
    event['log_event']['title'] = 'UAV adjusts altitude and follows an exit route from the newly activated restricted region'
    return script

def verify_run(config, run):
    safe = event_tick(run, 'crowd_safe_zone_move')
    lockdown, reroute, cleared = [event_tick(run, e) for e in
        ('airspace_lockdown', 'uav_reroute_lockdown', 'lockdown_cleared')]
    require(safe < lockdown < reroute < cleared, '疏散、封锁、绕飞、解除顺序不一致')
    arrivals = [completed_move(run, 'move_x6_crowd_evacuation_%02d' % i, 690) for i in range(8)]
    flight = completed_move(run, 'move_x6_uav_avoid_nfz', 690)
    require(flight['target_tick'] <= cleared, '绕飞未完成却宣告解除')
    active = require_fields(run, 'nfz_x6_lockdown', {'state': 'active', 'activity_type': 'active'}, lockdown + 1, cleared)
    inactive = require_fields(run, 'nfz_x6_lockdown', {'state': 'standdown', 'activity_type': 'standdown'}, 680, 690)
    control = require_fields(run, 'uav_x6_monitor', {'incident_state.requires_reroute': False,
        'control_state.reroute_active': False, 'mission_state.hazard_managed': True}, 680, 690)
    return {'all_safe_event_tick': safe, 'lockdown_tick': lockdown, 'crowd_arrivals': arrivals,
            'reroute_arrival': flight, 'cleared_tick': cleared, 'active_region_interval': active,
            'standdown_region': inactive, 'closed_control': control,
            'region_authority': 'restricted_region_runtime_active: standdown表示停用，实体仍可存在'}

if __name__ == '__main__':
    run_entry(Path(__file__))
