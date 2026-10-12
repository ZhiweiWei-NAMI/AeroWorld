"""正式水平与垂直安全间隔违例触发避让；避让完成后雾和导航降级仍持续。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import copy
import math
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, path_length, completed_move, measured_pair, run_entry)
from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults

SPEC = {'profile': 'X4_fog_to_uav_conflict', 'seed': 'seed02', 'arm_id': 'factual', 'window': {'start_tick': 210, 'fork_tick': 220, 'end_tick': 415, 'phase': 'after_state_sample_before_actions'}, 'research_question': '雾中汇聚是否进入正式安全间隔范围，避让能否解除间隔违例？', 'hypothesis': '正式水平与垂直安全间隔违例触发避让；避让完成后雾和导航降级仍持续。', 'window_basis': '最后相关续航移动作404到达，正式停止样本405/410/415；避让控制已退出，但雾和视觉导航降级持续。445以后普通返航不属于直接后果。', 'parameter_model': {'parameter_changes': False, 'motion_model': '沿用源动作分段恒速与ceil(distance/speed*10)整tick计时；不改变速度、航点或语义阈值。', 'state_visibility': '先采样后动作；runtime补丁按声明delay_ticks生效。', 'recovery_model': '正式步长5，当前recovery_hold_samples=3；first_false_capture+(3-1)*5。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

def build_script(source):
    event = next(e for e in source['events'] if e['event_id'] == 'fog_conflict_recovered')
    replacements = {}
    for action in event['actions']:
        if action['type'] == 'set_runtime_state':
            repaired = copy.deepcopy(action)
            require(repaired['state_patch']['navigation_state']['visual_navigation_degraded'] is False,
                    '源导航恢复动作契约变化')
            del repaired['state_patch']['navigation_state']
            replacements[action['action_id']] = repaired
    require(len(replacements) == 2, '缺少双机恢复动作')
    script = edit_script(source, replace_actions=replacements)
    events = {event['event_id']: event for event in script['events']}
    events['fog_uav_conflict']['log_event']['title'] = 'Aircraft enter the configured horizontal and vertical separation limits and begin evasion'
    events['fog_conflict_recovered']['log_event']['title'] = 'Spacing is restored and avoidance control ends while fog and visual navigation degradation persist'
    return script

def verify_run(config, run):
    convergence = event_tick(run, 'fog_converging_routes')
    close_approach = event_tick(run, 'fog_uav_conflict')
    require(convergence < close_approach, '近距触发先于汇聚动作')
    evasion = completed_move(run, 'move_x4_uav_b_evasion', 415)
    resume = completed_move(run, 'move_x4_uav_a_resume', 415)
    state = [require_fields(run, eid, {'navigation_state.visual_navigation_degraded': True,
        'navigation_state.route_uncertain': True, 'navigation_state.visual_navigation_recovered': False,
        'navigation_state.route_recovered': False, 'control_state.evasion_active': False,
        'control_state.deconfliction_active': False}, 380, 415) for eid in ('uav_x4_a', 'uav_x4_b')]
    degraded = [require_fields(run, eid, {'navigation_state.visual_navigation_degraded': True,
        'navigation_state.route_uncertain': True}, 255, 415) for eid in ('uav_x4_a', 'uav_x4_b')]
    fog = [row['fog_density'] for row in run['engine'].weather_rows if 255 <= row['tick'] <= 415]
    require(len(fog) == 161 and min(fog) > get_governed_parameter_defaults()['fog_density_threshold'],
            '雾已变化，应重新检查视觉导航恢复依据')
    pair = measured_pair(run, 'uav_x4_a', 'uav_x4_b', 210, 415)
    margin = get_governed_parameter_defaults()['separation_margin_m']
    trigger = next(t for t in run['interpreter'].script['triggers']
        if t['type'] == 'entity_proximity' and t.get('metric') == 'xy_plus_z'
        and {t['entity_a'], t['entity_b']} == {'uav_x4_a', 'uav_x4_b'})
    first = samples(run, 'uav_x4_a', 210, 415)
    second = samples(run, 'uav_x4_b', 210, 415)
    formal_true_ticks = [a['tick'] for a, b in zip(first, second)
        if a['tick'] % 5 == 0
        and math.dist(a['pos_enu'][:2], b['pos_enu'][:2]) < trigger['horizontal_distance_m']
        and abs(a['pos_enu'][2] - b['pos_enu'][2]) < trigger['vertical_distance_m']]
    return {'close_approach_tick': close_approach, 'evasion': evasion, 'resume': resume,
            'restored_control': state, 'unresolved_navigation': degraded, 'minimum_fog_density': min(fog), 'mission_pair': pair, 'separation_margin_m': margin,
            'actual_separation_violation': bool(formal_true_ticks),
            'formal_separation_true_ticks': formal_true_ticks,
            'separation_limits_m': {'horizontal': trigger['horizontal_distance_m'], 'vertical': trigger['vertical_distance_m']},
            'interpretation': '正式水平与垂直间隔违例发生后解除；避让控制结束，雾与导航降级持续'}

if __name__ == '__main__':
    run_entry(Path(__file__))
