"""Dedicated ARM entry for L6-4_v1 / seed01 / grp_wideband_jamming / arm_early_alternate_channel.

Source scene: Dataset/scenarios/L6_digital_layer/failure/L6-4_v1/scene_setup.json
Source script: Dataset/scenarios/L6_digital_layer/failure/L6-4_v1/event_script.json
Window: [235, 520], shared checkpoint 240, phase after_state_sample_before_actions.
Mechanism: early_backup_mode_with_mission_continuation.
Ordinary landing at dispatch ticks 540/550 is outside this experiment window.
Theory and scope: design/L6-4_v1/seed01/grp_wideband_jamming/experiment.json.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-4_v1"
SEED = "seed01"
ANCHOR_EVENT_ID = "wideband_jamming"
ARM_ID = "arm_early_alternate_channel"
MODE = "intervention"
ROLE = "narrative_branch"

# Explicit per-arm mechanism. Every omitted action and cancelled source event is listed here.
CANCEL_EVENT_IDS = ('multi_uav_hold_entry', 'multi_uav_safe_hold', 'alternate_channel')

OMIT_ACTION_IDS = ()

INJECTIONS = ({'tick': 250,
  'type': 'set_visual_state',
  'action_id': 'arm_early_alternate_tower',
  'entity_id': 'tower_l6_4_v1',
  'visual_state': {'mode': 'alternate_channel'}},
 {'tick': 250,
  'type': 'move_entity',
  'action_id': 'arm_early_mission_continuation_a',
  'entity_id': 'uav_digital_l6_4_v1',
  'activity_type': 'cruise',
  'post_activity_type': 'cruise',
  'velocity_mps': 5.5,
  'waypoints_enu_m': [[6634.035, 6525.817, 30.0],
                      [6638.035, 6529.817, 30.0],
                      [6649.035, 6537.817, 30.0]]},
 {'tick': 250,
  'type': 'move_entity',
  'action_id': 'arm_early_mission_continuation_b',
  'entity_id': 'uav_digital_l6_4_v1_secondary',
  'activity_type': 'cruise',
  'post_activity_type': 'cruise',
  'velocity_mps': 5.5,
  'waypoints_enu_m': [[6622.035, 6532.817, 50.0],
                      [6627.035, 6537.817, 50.0],
                      [6616.035, 6543.817, 50.0]]},
 {'tick': 250,
  'type': 'set_runtime_state',
  'action_id': 'arm_early_alternate_a',
  'entity_id': 'uav_digital_l6_4_v1',
  'delay_ticks': 1,
  'state_patch': {'control_state': {'safe_hold_active': False},
                  'security_state': {'alternate_channel_active': True, 'jamming_active': False}}},
 {'tick': 250,
  'type': 'set_runtime_state',
  'action_id': 'arm_early_alternate_b',
  'entity_id': 'uav_digital_l6_4_v1_secondary',
  'delay_ticks': 1,
  'state_patch': {'control_state': {'safe_hold_active': False},
                  'security_state': {'alternate_channel_active': True, 'jamming_active': False}}})

MECHANISM = {'kind': 'early_backup_mode_with_mission_continuation',
 'cancel_event_ids': ['multi_uav_hold_entry', 'multi_uav_safe_hold', 'alternate_channel'],
 'omit_action_ids': [],
 'injections': [{'tick': 250,
                 'type': 'set_visual_state',
                 'action_id': 'arm_early_alternate_tower',
                 'entity_id': 'tower_l6_4_v1',
                 'visual_state': {'mode': 'alternate_channel'}},
                {'tick': 250,
                 'type': 'move_entity',
                 'action_id': 'arm_early_mission_continuation_a',
                 'entity_id': 'uav_digital_l6_4_v1',
                 'activity_type': 'cruise',
                 'post_activity_type': 'cruise',
                 'velocity_mps': 5.5,
                 'waypoints_enu_m': [[6634.035, 6525.817, 30.0],
                                     [6638.035, 6529.817, 30.0],
                                     [6649.035, 6537.817, 30.0]]},
                {'tick': 250,
                 'type': 'move_entity',
                 'action_id': 'arm_early_mission_continuation_b',
                 'entity_id': 'uav_digital_l6_4_v1_secondary',
                 'activity_type': 'cruise',
                 'post_activity_type': 'cruise',
                 'velocity_mps': 5.5,
                 'waypoints_enu_m': [[6622.035, 6532.817, 50.0],
                                     [6627.035, 6537.817, 50.0],
                                     [6616.035, 6543.817, 50.0]]},
                {'tick': 250,
                 'type': 'set_runtime_state',
                 'action_id': 'arm_early_alternate_a',
                 'entity_id': 'uav_digital_l6_4_v1',
                 'delay_ticks': 1,
                 'state_patch': {'control_state': {'safe_hold_active': False},
                                 'security_state': {'alternate_channel_active': True,
                                                    'jamming_active': False}}},
                {'tick': 250,
                 'type': 'set_runtime_state',
                 'action_id': 'arm_early_alternate_b',
                 'entity_id': 'uav_digital_l6_4_v1_secondary',
                 'delay_ticks': 1,
                 'state_patch': {'control_state': {'safe_hold_active': False},
                                 'security_state': {'alternate_channel_active': True,
                                                    'jamming_active': False}}}]}

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
    subject_entity_ids=("uav_digital_l6_4_v1", "uav_digital_l6_4_v1_secondary"),
    mechanism=MECHANISM,
    # 工作单没有 retime 字段：本臂不改写任何触发时刻，分歧来自显式列出的事件取消、运行时状态注入与航迹注入。
    retime=None,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis=(
        'wideband_jamming@240 保留。250 注入塔的 alternate_channel 视觉态、双机任务继续移动和双机 运行时补丁；251 起 jamming_active=false、alternate_channel_active=true，A/B 分别在 291/317 起静止于计划终点。hold_entry、safe_hold 和原 alternate_channel 被显式取消。该臂只证明备用抗干扰模式已启用并清除当前链路退化，不声明物理频段、信道 ID 或 handover 发生变化。'
    ),
    falsified_if=(
        '若 251 后任一 UAV 的 jamming_active=true 或 alternate_channel_active=false，塔未进入 alternate_channel 视觉态，任一 hold/safe_hold/原 alternate 事件触发，任一任务继续动作未达到计划终点，或 safe_hold_active=true，则假设被证伪。'
    ),
    window_basis=(
        '闭区间 [235,520]；240 是 after_state_sample_before_actions 共享检查点。窗口从干扰上升前的 正式采样开始，覆盖 241 的干扰状态、所有干预、460/461 的恢复边界和最晚 486 的 factual 恢复移动终止，并保留至 520 的后置观测。普通 landing 的 540/550 派发位于窗口之外。'
    ),
    window_needed_nodes=({'event_id': 'wideband_jamming', 'tick': 240, 'effective_tick': 241, 'role': 'anchor_root_kept'},
 {'event_id': 'arm_early_alternate_tower', 'tick': 250, 'role': 'injected_backup_mode_visual'},
 {'event_id': 'arm_early_alternate_a',
  'tick': 250,
  'entity_id': 'uav_digital_l6_4_v1',
  'effective_tick': 251,
  'role': 'injected_backup_mode_state'},
 {'event_id': 'arm_early_alternate_b',
  'tick': 250,
  'entity_id': 'uav_digital_l6_4_v1_secondary',
  'effective_tick': 251,
  'role': 'injected_backup_mode_state'},
 {'event_id': 'arm_early_mission_continuation_a',
  'tick': 250,
  'entity_id': 'uav_digital_l6_4_v1',
  'terminal_state_tick': 291,
  'role': 'injected_mission_continuation'},
 {'event_id': 'arm_early_mission_continuation_b',
  'tick': 250,
  'entity_id': 'uav_digital_l6_4_v1_secondary',
  'terminal_state_tick': 317,
  'role': 'injected_mission_continuation'},
 {'event_id': 'multi_uav_hold_entry', 'tick': 270, 'role': 'explicitly_cancelled'},
 {'event_id': 'multi_uav_safe_hold', 'tick': 400, 'role': 'explicitly_cancelled'},
 {'event_id': 'alternate_channel', 'tick': 460, 'role': 'explicitly_cancelled_source_recovery'}),
    window_censored=False,
    group_suffix="",
)

if __name__ == "__main__":
    run_arm(SPEC)
