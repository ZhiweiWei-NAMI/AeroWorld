"""实际派车、抵达和响应记录完成后关闭派车状态；伤情仍保留，不能宣称治愈。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import copy
import math
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, path_length, completed_move, measured_pair, run_entry)
from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults

SPEC = {'profile': 'X3_pedestrian_fall_to_emergency_response', 'seed': 'seed02', 'arm_id': 'factual', 'window': {'start_tick': 210, 'fork_tick': 220, 'end_tick': 460, 'phase': 'after_state_sample_before_actions'}, 'research_question': '跌倒后是否发生发现、派车、抵达和响应记录，派车状态是否正确关闭？', 'hypothesis': '实际派车、抵达和响应记录完成后关闭派车状态；伤情仍保留，不能宣称治愈。', 'window_basis': '救护车397到达；445完成响应记录，446关闭派车状态；正式样本450/455/460确认关闭。伤情未恢复，仍保留为截断状态。525开始的无人机返航不属于响应结果。', 'parameter_model': {'parameter_changes': False, 'motion_model': '沿用源动作分段恒速与ceil(distance/speed*10)整tick计时；不改变速度、航点或语义阈值。', 'state_visibility': '先采样后动作；runtime补丁按声明delay_ticks生效。', 'recovery_model': '正式步长5，当前recovery_hold_samples=3；first_false_capture+(3-1)*5。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

def build_script(source):
    return edit_script(source, append_actions={'emergency_documented': [{
        'type': 'set_runtime_state', 'action_id': 'x3_close_dispatch_after_documentation',
        'entity_id': 'ped_x3_fall', 'delay_ticks': 1,
        'state_patch': {'incident_state': {'dispatch_active': False}}}]})

def verify_run(config, run):
    detected, dispatched = event_tick(run, 'uav_detection'), event_tick(run, 'ambulance_dispatch')
    arrived, documented = event_tick(run, 'ambulance_arrived'), event_tick(run, 'emergency_documented')
    motion = completed_move(run, 'move_x3_ambulance_dispatch', 460)
    require(detected < dispatched < arrived <= motion['target_tick'] < documented, '救援事件与救护车实际到达顺序不一致')
    active = require_fields(run, 'ped_x3_fall', {'incident_state.dispatch_active': True}, dispatched + 1, documented)
    closed = require_fields(run, 'ped_x3_fall', {'incident_state.dispatch_active': False}, 450, 460)
    injury = require_fields(run, 'ped_x3_fall', {'pedestrian_state.posture': 'fallen', 'pedestrian_state.health': 'injured'}, 225, 460)
    return {'detection_tick': detected, 'dispatch_tick': dispatched, 'arrival_event_tick': arrived,
            'ambulance_target': motion, 'documented_tick': documented, 'dispatch_active_interval': active,
            'closed_dispatch': closed, 'unresolved_injury': injury}

if __name__ == '__main__':
    run_entry(Path(__file__))
