"""降雨后失联、迫降与人群避让实际发生；恢复链路不等于天气恢复。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import copy
import math
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, path_length, completed_move, measured_pair, run_entry)
from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults

SPEC = {'profile': 'X1_rain_to_c2loss_to_forced_landing', 'seed': 'seed01', 'arm_id': 'factual', 'window': {'start_tick': 170, 'fork_tick': 180, 'end_tick': 635, 'phase': 'after_state_sample_before_actions'}, 'research_question': '降雨及通信失效是否引起安全保持、迫降和人群避让？', 'hypothesis': '降雨后失联、迫降与人群避让实际发生；恢复链路不等于天气恢复。', 'window_basis': '迫降及人群避让最后到达分别为621和608；最后运动停止的正式样本625，连续三样本至635。保留迫降本身，不扩大到普通飞行生命周期。', 'parameter_model': {'parameter_changes': False, 'motion_model': '沿用源动作分段恒速与ceil(distance/speed*10)整tick计时；不改变速度、航点或语义阈值。', 'state_visibility': '先采样后动作；runtime补丁按声明delay_ticks生效。', 'recovery_model': '正式步长5，当前recovery_hold_samples=3；first_false_capture+(3-1)*5。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

def build_script(source):
    return edit_script(source)

def verify_run(config, run):
    event_tick(run, 'c2_loss_tick')
    event_tick(run, 'forced_landing_descent')
    reaction = event_tick(run, 'crowd_reacts_to_landing')
    landing = completed_move(run, 'move_x1_safe_touchdown', 635)
    people = [completed_move(run, 'move_x1_crowd_clear_%02d' % i, 635) for i in range(9)]
    restored = require_fields(run, 'uav_x1_forced_landing', {
        'communication_state.communication_unavailable': False,
        'control_state.safe_hold_active': False, 'mission_state.unsafe': False}, 615, 635)
    tower = require_fields(run, 'tower_x1_c2', {'communication_state.availability': 1.0,
        'communication_state.station_unavailable': False}, 615, 635)
    require(all(row['vel_mps'] == [0.0, 0.0, 0.0] for row in samples(run, 'uav_x1_forced_landing', 625, 635)), '迫降最终状态仍运动')
    return {'reaction_tick': reaction, 'touchdown': landing, 'crowd_arrivals': people,
            'restored_link': restored, 'restored_tower': tower}

if __name__ == '__main__':
    run_entry(Path(__file__))
