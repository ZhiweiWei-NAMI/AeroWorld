"""保留跌倒与发现，取消派车及响应后继，伤情持续而救护车不移动。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import copy
import math
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, path_length, completed_move, measured_pair, run_entry)
from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults

SPEC = {'profile': 'X3_pedestrian_fall_to_emergency_response', 'seed': 'seed02', 'arm_id': 'suppress_direct_consequence', 'window': {'start_tick': 210, 'fork_tick': 220, 'end_tick': 460, 'phase': 'after_state_sample_before_actions'}, 'research_question': '跌倒后是否发生发现、派车、抵达和响应记录，派车状态是否正确关闭？', 'hypothesis': '保留跌倒与发现，取消派车及响应后继，伤情持续而救护车不移动。', 'window_basis': '救护车397到达；445完成响应记录，446关闭派车状态；正式样本450/455/460确认关闭。伤情未恢复，仍保留为截断状态。525开始的无人机返航不属于响应结果。', 'parameter_model': {'parameter_changes': False, 'motion_model': '沿用源动作分段恒速与ceil(distance/speed*10)整tick计时；不改变速度、航点或语义阈值。', 'state_visibility': '先采样后动作；runtime补丁按声明delay_ticks生效。', 'recovery_model': '正式步长5，当前recovery_hold_samples=3；first_false_capture+(3-1)*5。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

def build_script(source):
    return edit_script(source, remove_events=('ambulance_dispatch', 'ambulance_arrived', 'isolation_setup',
        'responder_handoff', 'emergency_documented', 'lifecycle_landing_uav_x3_detect'))

def verify_run(config, run):
    fall, detection = event_tick(run, 'pedestrian_fall'), event_tick(run, 'uav_detection')
    require(fall < detection, '发现事件先于跌倒')
    for event in ('ambulance_dispatch', 'ambulance_arrived', 'isolation_setup', 'responder_handoff', 'emergency_documented'):
        absent_event(run, event)
    injury = require_fields(run, 'ped_x3_fall', {'pedestrian_state.posture': 'fallen',
        'pedestrian_state.health': 'injured', 'incident_state.dispatch_active': False}, 225, 460)
    distance = path_length(run, 'ambulance_x3', 220, 460)
    require(distance == 0.0, '派车被取消但救护车仍在行驶')
    return {'fall_tick': fall, 'detected_tick': detection, 'unresolved_injury': injury,
            'ambulance_path_m': distance, 'response_completed': False}

if __name__ == '__main__':
    run_entry(Path(__file__))
