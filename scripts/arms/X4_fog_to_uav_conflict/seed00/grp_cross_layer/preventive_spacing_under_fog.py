"""雾与视觉降级保留；A 的出发时刻和路径不变，只让 B 提前 20 tick 出发、终点向东外移 40 m。实际接近条件决定是否响应，不注入恢复。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, completed_move, measured_pair, run_entry)

SPEC = {'profile': 'X4_fog_to_uav_conflict', 'seed': 'seed00', 'arm_id': 'preventive_spacing_under_fog', 'window': {'start_tick': 210, 'fork_tick': 220, 'end_tick': 415, 'phase': 'after_state_sample_before_actions'}, 'research_question': '雾中视觉降级时，B 机提前错峰能否阻断接近冲突？', 'hypothesis': '雾与视觉降级保留；A 的出发时刻和路径不变，只让 B 提前 20 tick 出发、终点向东外移 40 m。实际接近条件决定是否响应，不注入恢复。', 'window_basis': '沿用同 seed factual 共同窗口、速度与接近条件。B 独立调度，A 原调度不变；依赖冲突响应的后继未触发不解释为任务完成。', 'parameter_model': {'parameter_changes': False, 'motion_model': '不改源速度或谓词阈值；分段运动沿用当前引擎。', 'state_visibility': '先采样后动作；runtime 补丁按声明 delay_ticks 生效。', 'recovery_model': '正式步长 5，recovery_hold_samples=3。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

def factual_script(source):
    import importlib.util
    path = Path(__file__).with_name("factual.py")
    spec = importlib.util.spec_from_file_location("_same_seed_factual", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_script(source)


def build_script(source):
    import copy
    script = factual_script(source)
    converge = next(e for e in script['events'] if e['event_id'] == 'fog_converging_routes')
    trigger = next(t for t in script['triggers'] if t['trigger_id'] == converge['trigger_ref'])
    move = next(a for a in converge['actions'] if a['action_id'] == 'move_x4_uav_b_converge')
    converge['actions'].remove(move)
    move['waypoints_enu_m'][-1][0] += 40.0
    move['activity_type'] = 'cruise'
    move['post_activity_type'] = 'holding'
    b_trigger = copy.deepcopy(trigger)
    b_trigger['trigger_id'] = 'trig_x4_b_preventive_spacing'
    b_trigger['delay_ticks'] -= 20
    b_event = copy.deepcopy(converge)
    b_event.update(event_id='x4_b_preventive_spacing', trigger_ref=b_trigger['trigger_id'], actions=[move])
    b_event['log_event'].update(
        topic='evt_X4_fog_to_uav_conflict_x4_b_preventive_spacing',
        title='Only UAV B departs early toward the exterior waiting point',
        target_ids=[move['entity_id']], intent='preventive_spacing', intent_stage='03.preventive_spacing')
    converge['log_event'].update(
        title='UAV A continues its original schedule while UAV B follows the separate route',
        target_ids=['uav_x4_a'], intent='mission_route_continued', intent_stage='03.mission_route_continued')
    return edit_script(script, add_triggers=[b_trigger], add_events=[b_event])


def verify_run(config, run):
    upstream = require_fields(run, 'uav_x4_a',
        {'navigation_state.route_uncertain': True,
         'navigation_state.visual_navigation_degraded': True}, 255, 415)
    a_move = completed_move(run, 'move_x4_uav_a_converge', 415)
    b_move = completed_move(run, 'move_x4_uav_b_converge', 415)
    for event in ('fog_uav_conflict', 'fog_conflict_recovered'):
        absent_event(run, event)
    return {'upstream_persists': upstream, 'a_move': a_move, 'b_move': b_move,
            'pair_motion': measured_pair(run, 'uav_x4_a', 'uav_x4_b', 220, 415),
            'interpretation': 'A keeps its factual schedule; only B is advanced and routed outward. Fog and visual degradation remain; no synthetic recovery.'}


if __name__ == '__main__':
    run_entry(Path(__file__))
