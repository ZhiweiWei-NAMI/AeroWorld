"""雨与 C2 失联真实发生，UAV 进入安全保持；备用链路在保持期内提前恢复，迫降链不发生。

上游与 factual 完全一致（降雨、塔台失效、UAV 通信不可用、safe_hold 激活），
分叉点在 hold 期间：本地备份链路于 tick 390 恢复，hold 解除后 UAV 沿计划返航
航线返回 home pad。完整返航由一条移动命令执行，含平飞中间点与下降着陆；
显式取消迫降链。
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import math
from Dataset.tools.x_branch_support import (edit_script, require, samples, event_tick,
    absent_event, require_fields, completed_move, run_entry)

SPEC = {'profile': 'X1_rain_to_c2loss_to_forced_landing', 'seed': 'seed01', 'arm_id': 'early_link_recovery',
        'window': {'start_tick': 170, 'fork_tick': 180, 'end_tick': 635, 'phase': 'after_state_sample_before_actions'},
        'research_question': '雨与 C2 失联都发生时，备用链路提前恢复能否阻断迫降？',
        'hypothesis': '雨与 C2 失联真实发生，UAV 先 hold；备用链路在保持期内恢复，hold 解除后沿计划返航，不进入迫降。',
        'window_basis': 'factual hold 激活 365（首个 true 样本 370）、迫降下降 395 出发。恢复动作取 390（hold 期内、迫降出发前），返航按源动作恒速模型于窗口末端前完成。',
        'parameter_model': {'parameter_changes': False, 'motion_model': '沿用源动作分段恒速与 ceil(distance/speed*10) 整 tick 计时；返航巡航 5.0 m/s、下降 5.0 m/s 为显式分支参数。', 'state_visibility': '先采样后动作；runtime 补丁按声明 delay_ticks 生效。', 'recovery_model': '正式步长 5，recovery_hold_samples=3。', 'scope': '窗口推导来自源动作实际完成及状态保持；分支运行后重新验证。'}}

RETURN_WP_DESCEND = [[6378.5, 6556.0, 50.0], [6351.654, 6527.313, 3.0], [6351.654, 6527.313, 0.8]]


def build_script(source):
    return edit_script(source,
        remove_events=('forced_landing_descent', 'crowd_reacts_to_landing', 'x1_recovery'),
        add_triggers=[{'trigger_id': 'x_backup_link_restore', 'type': 'tick', 'tick': 390},
                      {'trigger_id': 'x_resume_after_restore', 'type': 'event_fired_after',
                       'event_id': 'x_backup_link_restore', 'delay_ticks': 5}],
        add_events=[
            {'event_id': 'x_backup_link_restore', 'trigger_ref': 'x_backup_link_restore',
             'priority': 100, 'max_fire_count': 1,
             'actions': [
                 {'type': 'set_runtime_state', 'action_id': 'x1_backup_tower_restore',
                  'entity_id': 'tower_x1_c2', 'delay_ticks': 0,
                  'state_patch': {'communication_state': {'availability': 1.0, 'station_unavailable': False}}},
                 {'type': 'set_visual_state', 'action_id': 'x1_backup_tower_online',
                  'entity_id': 'tower_x1_c2', 'visual_state': {'mode': 'online'}},
                 {'type': 'set_runtime_state', 'action_id': 'x1_backup_uav_link_restore',
                  'entity_id': 'uav_x1_forced_landing', 'delay_ticks': 0,
                  'state_patch': {'communication_state': {'communication_unavailable': False}}}],
             'log_event': {'topic': 'x_backup_link_restore', 'category': 'communication',
                 'title': 'Backup link restored during safe hold', 'severity': 'info', 'target_ids': []}},
            {'event_id': 'x_resume_after_restore', 'trigger_ref': 'x_resume_after_restore',
             'priority': 100, 'max_fire_count': 1,
             'actions': [
                 {'type': 'set_runtime_state', 'action_id': 'x1_release_hold_on_restore',
                  'entity_id': 'uav_x1_forced_landing', 'delay_ticks': 0,
                  'state_patch': {'control_state': {'safe_hold_active': False}, 'mission_state': {'unsafe': False}}},
                 {'type': 'move_entity', 'action_id': 'x1_descend_on_backup_link',
                  'entity_id': 'uav_x1_forced_landing', 'velocity_mps': 5.0,
                  'activity_type': 'descending', 'post_activity_type': 'landed',
                  'waypoints_enu_m': RETURN_WP_DESCEND,
                  'terminal_feasibility': {'model': 'constant_speed_path_with_explicit_landing_reference_v2',
                      'landing_reference_enu_m': [6351.654, 6527.313, 0.8],
                      'touchdown_altitude_tolerance_m': 1.5,
                      'touchdown_dwell_ticks': 5}}],
             'log_event': {'topic': 'x_resume_after_restore', 'category': 'mission',
                 'title': 'Resume mission on restored backup link', 'severity': 'info', 'target_ids': []}}])


def verify_run(config, run):
    event_tick(run, 'rain_threshold')
    event_tick(run, 'c2_loss_tick')
    event_tick(run, 'c2_safe_hold')
    for event in ('forced_landing_descent', 'crowd_reacts_to_landing', 'x1_recovery'):
        absent_event(run, event)
    require_fields(run, 'tower_x1_c2',
                   {'communication_state.station_unavailable': True}, 265, 385)
    require_fields(run, 'uav_x1_forced_landing',
                   {'communication_state.communication_unavailable': True}, 260, 385)
    require_fields(run, 'uav_x1_forced_landing',
                   {'control_state.safe_hold_active': True}, 370, 385)
    event_tick(run, 'x_backup_link_restore')
    event_tick(run, 'x_resume_after_restore')
    require_fields(run, 'tower_x1_c2',
                   {'communication_state.availability': 1.0,
                    'communication_state.station_unavailable': False}, 395, 635)
    require_fields(run, 'uav_x1_forced_landing',
                   {'communication_state.communication_unavailable': False,
                    'control_state.safe_hold_active': False}, 400, 635)
    midpoint = RETURN_WP_DESCEND[0]
    midpoint_ticks = [row['tick'] for row in samples(run, 'uav_x1_forced_landing', 396, 635)
                      if math.dist(row['pos_enu'], midpoint) < 1e-5]
    require(bool(midpoint_ticks), '完整返航没有经过平飞中间点')
    waypoint = {'target_tick': min(midpoint_ticks), 'target_enu_m': midpoint,
                'attribution': 'intermediate waypoint of x1_descend_on_backup_link'}
    landing = completed_move(run, 'x1_descend_on_backup_link', 635)
    require(landing['target_tick'] <= 635, '返航未在窗口内完成')
    return {'upstream': {'rain': event_tick(run, 'rain_threshold'),
                         'c2_loss': event_tick(run, 'c2_loss_tick'),
                         'safe_hold': event_tick(run, 'c2_safe_hold')},
            'block': {'backup_restore_tick': 390, 'hold_released': True},
            'return_waypoint': waypoint, 'return_landing': landing,
            'absent_chain': ['forced_landing_descent', 'crowd_reacts_to_landing', 'x1_recovery']}


if __name__ == '__main__':
    run_entry(Path(__file__))
