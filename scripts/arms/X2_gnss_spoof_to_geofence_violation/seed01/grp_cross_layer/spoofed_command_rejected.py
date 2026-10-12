"""欺骗与定位可信度下降保留；只拒绝污染移动，原任务航段继续，是否越界由正式几何谓词判断。不要求窗口内导航恢复或返航。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, completed_move, run_entry)

SPEC = {'profile': 'X2_gnss_spoof_to_geofence_violation', 'seed': 'seed01', 'arm_id': 'spoofed_command_rejected', 'window': {'start_tick': 220, 'fork_tick': 230, 'end_tick': 415, 'phase': 'after_state_sample_before_actions'}, 'research_question': '欺骗持续时，执行端拒绝污染指令能否阻断越界？', 'hypothesis': '欺骗与定位可信度下降保留；只拒绝污染移动，原任务航段继续，是否越界由正式几何谓词判断。不要求窗口内导航恢复或返航。', 'window_basis': '沿用同 seed factual 的共同窗口与任务航段；不注入定时恢复或提前着陆。', 'parameter_model': {'parameter_changes': False, 'motion_model': '不改源速度或谓词阈值；分段运动沿用当前引擎。', 'state_visibility': '先采样后动作；runtime 补丁按声明 delay_ticks 生效。', 'recovery_model': '正式步长 5，recovery_hold_samples=3。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

def factual_script(source):
    import importlib.util
    path = Path(__file__).with_name("factual.py")
    spec = importlib.util.spec_from_file_location("_same_seed_factual", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_script(source)


def build_script(source):
    script = factual_script(source)
    event = next(e for e in script['events'] if e['event_id'] == 'trajectory_offset')
    action = next(a for a in event['actions'] if a['action_id'] == 'move_x2_spoof_offset')
    event['actions'].remove(action)
    event['log_event'].update(
        title='Spoofed movement command rejected; previously scheduled mission motion continues',
        intent='command_rejected', intent_stage='02.command_rejected')
    return script


def verify_run(config, run):
    onset = event_tick(run, 'spoofing_start')
    rejected = event_tick(run, 'trajectory_offset')
    upstream = require_fields(run, 'uav_x2_spoofed',
        {'navigation_state.gnss_spoofed': True, 'navigation_state.route_uncertain': True}, 265, 415)
    for event in ('geofence_violation', 'safety_correction', 'route_recovered'):
        absent_event(run, event)
    return {'spoofing_tick': onset, 'command_rejection_tick': rejected,
            'upstream_persists': upstream,
            'removed_movement': 'move_x2_spoof_offset',
            'interpretation': 'Original mission motion continues; no navigation recovery or return-home completion is claimed.'}


if __name__ == '__main__':
    run_entry(Path(__file__))
