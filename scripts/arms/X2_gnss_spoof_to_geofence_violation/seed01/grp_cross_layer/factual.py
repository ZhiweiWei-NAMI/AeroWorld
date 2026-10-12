"""纠偏后GNSS与路线恢复，受限空域边界条件仍成立，因此围栏告警持续且不宣告任务恢复。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import copy
import math
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, path_length, completed_move, measured_pair, run_entry)
from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults

SPEC = {'profile': 'X2_gnss_spoof_to_geofence_violation', 'seed': 'seed01', 'arm_id': 'factual', 'window': {'start_tick': 220, 'fork_tick': 230, 'end_tick': 415, 'phase': 'after_state_sample_before_actions'}, 'research_question': 'GNSS欺骗、航迹偏移与围栏接近告警如何衔接；导航恢复时受限空域条件是否仍成立？', 'hypothesis': '纠偏后GNSS与路线恢复，受限空域边界条件仍成立，因此围栏告警持续且不宣告任务恢复。', 'window_basis': '纠偏移动作390到达，route_recovered在400派发，按5 tick延迟恢复导航；正式样本405/410/415确认。空域边界条件仍成立，围栏告警与任务恢复保持未解决；480开始的普通返航不属于此窗口。', 'parameter_model': {'parameter_changes': False, 'motion_model': '沿用源动作分段恒速与ceil(distance/speed*10)整tick计时；不改变速度、航点或语义阈值。', 'state_visibility': '先采样后动作；runtime补丁按声明delay_ticks生效。', 'recovery_model': '正式步长5，当前recovery_hold_samples=3；first_false_capture+(3-1)*5。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

def build_script(source):
    actions = {a['action_id']: a for e in source['events'] for a in e['actions']}
    correction_id = 'set_runtime_state_x2_gnss_spoof_to_geofence_violation_safety_correction_00'
    recovery_id = 'set_runtime_state_x2_gnss_spoof_to_geofence_violation_route_recovered_00'
    correction, recovery = copy.deepcopy(actions[correction_id]), copy.deepcopy(actions[recovery_id])
    require(correction['state_patch']['navigation_state']['geofence_violation'] is False, '纠偏源动作契约变化')
    del correction['state_patch']['navigation_state']['geofence_violation']
    recovery['state_patch']['navigation_state']['mission_recovered'] = False
    recovery['state_patch']['navigation_state']['operational_state'] = 'geofence_alert'
    visual = copy.deepcopy(actions['set_x2_nav_restored'])
    visual['visual_state']['mode'] = 'geofence_alert'
    return edit_script(source, replace_actions={correction_id: correction, recovery_id: recovery,
        'set_x2_nav_restored': visual})

def verify_run(config, run):
    alert = event_tick(run, 'geofence_violation')
    correction = completed_move(run, 'move_x2_corrected_route', 415)
    recovery = event_tick(run, 'route_recovered')
    require(correction['target_tick'] < recovery, '导航尚未完成纠偏就宣告恢复')
    held = require_fields(run, 'uav_x2_spoofed', {'navigation_state.geofence_violation': True}, alert + 1, 415)
    restored = require_fields(run, 'uav_x2_spoofed', {'navigation_state.geofence_violation': True,
        'navigation_state.mission_recovered': False, 'navigation_state.operational_state': 'geofence_alert',
        'navigation_state.gnss_spoofed': False, 'navigation_state.route_uncertain': False,
        'navigation_state.route_recovered': True}, 405, 415)
    pair = measured_pair(run, 'uav_x2_spoofed', 'nfz_x2_geofence', 405, 415)
    require(pair['minimum_distance_m'] > 10.0, '源10米接近触发尚未退出')
    from Dataset.semantic_simulation.predicate_state_computers import _restricted_regions, _distance_to_restricted_prism
    import json
    scene = json.loads((ROOT/config['sources']['scene']).read_text())
    region = _restricted_regions({e['entity_id']: e for e in scene['entities']})['nfz_x2_geofence']
    margin = get_governed_parameter_defaults()['boundary_margin_m']
    distances = [_distance_to_restricted_prism(row['pos_enu'], region)
                 for row in samples(run, 'uav_x2_spoofed', 405, 415)]
    require(all(d <= margin for d in distances), '空域边界条件已变化，应重新解释持续告警')
    return {'alert_tick': alert, 'correction': correction, 'recovery_tick': recovery,
            'latched_alert': held, 'restored_navigation': restored, 'source_trigger_distance': pair,
            'restricted_region': {'maximum_distance_m': max(distances), 'boundary_margin_m': margin},
            'interpretation': '导航恢复，源10米接近触发退出；受限空域边界条件持续，围栏告警与任务状态未恢复'}

if __name__ == '__main__':
    run_entry(Path(__file__))
