"""Dedicated ARM entry for L6-4_v2 / seed00 / grp_wideband_jamming / arm_jamming_never_happens.

Source scene: Dataset/scenarios/L6_digital_layer/failure/L6-4_v2/scene_setup.json
Source script: Dataset/scenarios/L6_digital_layer/failure/L6-4_v2/event_script.json
Window: [235, 520], shared checkpoint 240, phase after_state_sample_before_actions.
Mechanism: explicit_fault_and_response_cancellation_with_mission_continuation.
Ordinary landing at dispatch ticks 540/550 is outside this experiment window.
Theory and scope: design/L6-4_v2/seed00/grp_wideband_jamming/experiment.json.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-4_v2"
SEED = "seed00"
ANCHOR_EVENT_ID = "wideband_jamming"
ARM_ID = "arm_jamming_never_happens"
MODE = "intervention"
ROLE = "narrative_branch"
UAV_A = "uav_digital_l6_4_v2"
UAV_B = "uav_digital_l6_4_v2_secondary"

# Explicit per-arm mechanism. Every omitted action and cancelled source event is listed here.
CANCEL_EVENT_IDS = ('wideband_jamming', 'multi_uav_hold_entry', 'multi_uav_safe_hold', 'alternate_channel')

OMIT_ACTION_IDS = ()

INJECTIONS = ({'tick': 241,
  'type': 'move_entity',
  'action_id': 'arm_dual_cruise_continuation',
  'entity_id': 'uav_digital_l6_4_v2',
  'activity_type': 'cruise',
  'post_activity_type': 'cruise',
  'velocity_mps': 5.0,
  'waypoints_enu_m': [[6636.321, 6526.731, 30.0],
                      [6640.321, 6530.731, 30.0],
                      [6651.321, 6538.731, 30.0]]},
 {'tick': 241,
  'type': 'move_entity',
  'action_id': 'arm_dual_cruise_continuation_b',
  'entity_id': 'uav_digital_l6_4_v2_secondary',
  'activity_type': 'cruise',
  'post_activity_type': 'cruise',
  'velocity_mps': 5.0,
  'waypoints_enu_m': [[6624.321, 6533.731, 50.0],
                      [6629.321, 6538.731, 50.0],
                      [6618.321, 6544.731, 50.0]]})

MECHANISM = {'kind': 'explicit_fault_and_response_cancellation_with_mission_continuation',
 'cancel_event_ids': ['wideband_jamming',
                      'multi_uav_hold_entry',
                      'multi_uav_safe_hold',
                      'alternate_channel'],
 'omit_action_ids': [],
 'injections': [{'tick': 241,
                 'type': 'move_entity',
                 'action_id': 'arm_dual_cruise_continuation',
                 'entity_id': 'uav_digital_l6_4_v2',
                 'activity_type': 'cruise',
                 'post_activity_type': 'cruise',
                 'velocity_mps': 5.0,
                 'waypoints_enu_m': [[6636.321, 6526.731, 30.0],
                                     [6640.321, 6530.731, 30.0],
                                     [6651.321, 6538.731, 30.0]]},
                {'tick': 241,
                 'type': 'move_entity',
                 'action_id': 'arm_dual_cruise_continuation_b',
                 'entity_id': 'uav_digital_l6_4_v2_secondary',
                 'activity_type': 'cruise',
                 'post_activity_type': 'cruise',
                 'velocity_mps': 5.0,
                 'waypoints_enu_m': [[6624.321, 6533.731, 50.0],
                                     [6629.321, 6538.731, 50.0],
                                     [6618.321, 6544.731, 50.0]]}]}

SPEC = ArmSpec(
    profile=PROFILE,
    seed=SEED,
    anchor_event_id=ANCHOR_EVENT_ID,
    arm_id=ARM_ID,
    mode=MODE,
    role=ROLE,
    window_start_tick=235,
    window_fork_tick=240,
    window_end_tick=520,
    primary_predicate="security.communication_jamming_impact_active",
    subject_entity_ids=(UAV_A, UAV_B),
    mechanism=MECHANISM,
    # 工作单没有 retime 字段：本臂不改写任何触发时刻，分歧来自显式列出的事件取消与航迹注入。
    retime=None,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis=(
        'wideband_jamming、hold_entry、safe_hold 和 alternate_channel 被逐项显式取消。241 注入双机 5.0 m/s 任务继续动作，A/B 分别在 286/315 起静止于计划终点。全窗 jamming_active、safe_hold_active 和 alternate_channel_active 均应为 false；普通 landing 在窗口之外且不属于目标。'
    ),
    falsified_if=(
        '若任一被取消事件仍触发，任一相关状态变为 true，双机未从 242 开始运动，或未分别在 285/314 到达计划终点，则假设被证伪。'
    ),
    window_basis=(
        '闭区间 [235,520]；240 是 after_state_sample_before_actions 共享检查点。窗口从干扰上升前的 正式采样开始，覆盖 241 的干扰状态、所有干预、460/461 的恢复边界和最晚 486 的 factual 恢复移动终止，并保留至 520 的后置观测。普通 landing 的 540/550 派发位于窗口之外。'
    ),
    window_needed_nodes=({'event_id': 'wideband_jamming', 'tick': 240, 'role': 'explicitly_cancelled_anchor'},
 {'event_id': 'arm_dual_cruise_continuation',
  'tick': 241,
  'entity_id': 'uav_digital_l6_4_v2',
  'terminal_state_tick': 286,
  'role': 'injected_mission_continuation'},
 {'event_id': 'arm_dual_cruise_continuation_b',
  'tick': 241,
  'entity_id': 'uav_digital_l6_4_v2_secondary',
  'terminal_state_tick': 315,
  'role': 'injected_mission_continuation'},
 {'event_id': 'multi_uav_hold_entry', 'tick': 270, 'role': 'explicitly_cancelled'},
 {'event_id': 'multi_uav_safe_hold', 'tick': 400, 'role': 'explicitly_cancelled'},
 {'event_id': 'alternate_channel', 'tick': 460, 'role': 'explicitly_cancelled'}),
    window_censored=False,
    group_suffix="",
)

if __name__ == "__main__":
    run_arm(SPEC)
