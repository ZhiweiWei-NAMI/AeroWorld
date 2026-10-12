"""Dedicated ARM entry for L6-4_v2 / seed01 / grp_multi_uav_hold_entry / arm_hold_b_only.

Source scene: Dataset/scenarios/L6_digital_layer/failure/L6-4_v2/scene_setup.json
Source script: Dataset/scenarios/L6_digital_layer/failure/L6-4_v2/event_script.json
Window: [265, 520], shared checkpoint 270, phase after_state_sample_before_actions.
Mechanism: secondary_hold_primary_mission_continuation.
Ordinary landing at dispatch ticks 540/550 is outside this experiment window.
Theory and scope: design/L6-4_v2/seed01/grp_multi_uav_hold_entry/experiment.json.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-4_v2"
SEED = "seed01"
ANCHOR_EVENT_ID = "multi_uav_hold_entry"
ARM_ID = "arm_hold_b_only"
MODE = "intervention"
ROLE = "narrative_branch"
UAV_A = "uav_digital_l6_4_v2"
UAV_B = "uav_digital_l6_4_v2_secondary"

# Explicit per-arm mechanism. Every omitted action and cancelled source event is listed here.
CANCEL_EVENT_IDS = ()

OMIT_ACTION_IDS = ('move_uav_a_jam_hold',
 'set_runtime_state_l6_4_v2_multi_uav_safe_hold_00',
 'move_uav_a_channel_recover')

INJECTIONS: tuple[dict, ...] = ({'tick': 270,
  'type': 'move_entity',
  'action_id': 'arm_uav_a_mission_continuation',
  'entity_id': 'uav_digital_l6_4_v2',
  'activity_type': 'cruise',
  'post_activity_type': 'cruise',
  'velocity_mps': 5.5,
  'waypoints_enu_m': [[6636.321, 6526.731, 30.0],
                      [6640.321, 6530.731, 30.0],
                      [6651.321, 6538.731, 30.0]]},)

RETIME = None

MECHANISM = {'kind': 'secondary_hold_primary_mission_continuation',
 'cancel_event_ids': [],
 'omit_action_ids': ['move_uav_a_jam_hold',
                     'set_runtime_state_l6_4_v2_multi_uav_safe_hold_00',
                     'move_uav_a_channel_recover'],
 'injections': [{'tick': 270,
                 'type': 'move_entity',
                 'action_id': 'arm_uav_a_mission_continuation',
                 'entity_id': 'uav_digital_l6_4_v2',
                 'activity_type': 'cruise',
                 'post_activity_type': 'cruise',
                 'velocity_mps': 5.5,
                 'waypoints_enu_m': [[6636.321, 6526.731, 30.0],
                                     [6640.321, 6530.731, 30.0],
                                     [6651.321, 6538.731, 30.0]]}]}

SPEC = ArmSpec(
    profile=PROFILE,
    seed=SEED,
    anchor_event_id=ANCHOR_EVENT_ID,
    arm_id=ARM_ID,
    mode=MODE,
    role=ROLE,
    window_start_tick=265,
    window_fork_tick=270,
    window_end_tick=520,
    primary_predicate="security.communication_jamming_impact_active",
    subject_entity_ids=(UAV_A, UAV_B),
    mechanism=MECHANISM,
    retime=RETIME,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis=(
        '干扰链照常运行。270 仅 uav_digital_l6_4_v2_secondary 执行 authored hold；uav_digital_l6_4_v2 的 hold move、safe_hold 补丁和 460 recovery move 被逐项省略，同时从 270 以 5.5 m/s 沿 scene 计划点继续任务，在 311 起静止于 [6651.321, 6538.731, 30.0]。B 在 405–460 保持 safe_hold_active=true，A 全窗保持 false；461 起两机的干扰标志清零并启用备用抗干扰模式。'
    ),
    falsified_if=(
        '若被省略的 A 动作仍执行，A 未在 271 开始运动或未在 310 到达计划终点，A 任一时刻 safe_hold_active=true，B 未完成 hold，或 461 后干扰仍为 true，则假设被证伪。'
    ),
    window_basis=(
        '闭区间 [265,520]；270 是 after_state_sample_before_actions 共享检查点。窗口覆盖双机 hold 完成、400 的 safe_hold 声明、460/461 的备用模式恢复和最晚 486 的恢复移动终止，并保留至 520 的后置观测。普通 landing 的 540/550 派发位于窗口之外。'
    ),
    window_needed_nodes=({'event_id': 'multi_uav_hold_entry', 'tick': 270, 'role': 'anchor_kept'},
 {'event_id': 'move_uav_b_jam_hold',
  'tick': 270,
  'entity_id': 'uav_digital_l6_4_v2_secondary',
  'terminal_state_tick': 387,
  'role': 'held_aircraft_move'},
 {'event_id': 'move_uav_a_jam_hold',
  'tick': 270,
  'entity_id': 'uav_digital_l6_4_v2',
  'role': 'omitted_unheld_move'},
 {'event_id': 'arm_uav_a_mission_continuation',
  'tick': 270,
  'entity_id': 'uav_digital_l6_4_v2',
  'terminal_state_tick': 311,
  'role': 'injected_mission_continuation'},
 {'event_id': 'set_runtime_state_l6_4_v2_multi_uav_safe_hold_01',
  'tick': 400,
  'entity_id': 'uav_digital_l6_4_v2_secondary',
  'effective_tick': 405,
  'role': 'held_aircraft_safe_hold'},
 {'event_id': 'set_runtime_state_l6_4_v2_multi_uav_safe_hold_00',
  'tick': 400,
  'entity_id': 'uav_digital_l6_4_v2',
  'role': 'omitted_unheld_safe_hold'},
 {'event_id': 'alternate_channel', 'tick': 460, 'effective_tick': 461, 'role': 'backup_mode_kept'},
 {'event_id': 'move_uav_a_channel_recover',
  'tick': 460,
  'entity_id': 'uav_digital_l6_4_v2',
  'role': 'omitted_duplicate_recovery_move'},
 {'event_id': 'move_uav_b_channel_recover',
  'tick': 460,
  'entity_id': 'uav_digital_l6_4_v2_secondary',
  'terminal_state_tick': 484,
  'role': 'held_aircraft_recovery_move'}),
    window_censored=False,
    # 工作单 group_key 为 grp_multi_uav_hold_entry，按 runtime 的 group_key 拼接规则给出 v3。
    group_suffix="",
)

if __name__ == "__main__":
    run_arm(SPEC)
