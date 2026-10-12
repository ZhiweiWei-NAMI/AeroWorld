"""Dedicated ARM entry for L6-4_v2 / seed00 / grp_wideband_jamming / arm_factual.

Source scene: Dataset/scenarios/L6_digital_layer/failure/L6-4_v2/scene_setup.json
Source script: Dataset/scenarios/L6_digital_layer/failure/L6-4_v2/event_script.json
Window: [235, 520], shared checkpoint 240, phase after_state_sample_before_actions.
Mechanism: factual_no_intervention_current_engine.
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
ARM_ID = "arm_factual"
MODE = "factual_observation"
ROLE = "factual"
UAV_A = "uav_digital_l6_4_v2"
UAV_B = "uav_digital_l6_4_v2_secondary"

# Explicit per-arm mechanism. Every omitted action and cancelled source event is listed here.
CANCEL_EVENT_IDS = ()

OMIT_ACTION_IDS = ()

INJECTIONS: tuple[dict, ...] = ()

RETIME = None

MECHANISM = {'kind': 'factual_no_intervention_current_engine',
 'cancel_event_ids': [],
 'omit_action_ids': [],
 'injections': []}

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
    retime=RETIME,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis=(
        '当前引擎无干预运行是本组事实基线：wideband_jamming 于 240 派发，241 起双机 jamming_active=true；hold_entry@270、safe_hold@400、alternate_channel@460 依次发生；461 起双机 jamming_active=false、alternate_channel_active=true，恢复移动最晚在 486 起静止。普通 landing 的 540/550 派发在窗口之外，不属于本组目标。'
    ),
    falsified_if=(
        '不适用：该臂为 factual_observation，不声明干预假设；动作、状态和事件以实跑产物为准。'
    ),
    window_basis=(
        '闭区间 [235,520]；240 是 after_state_sample_before_actions 共享检查点。窗口从干扰上升前的 正式采样开始，覆盖 241 的干扰状态、所有干预、460/461 的恢复边界和最晚 486 的 factual 恢复移动终止，并保留至 520 的后置观测。普通 landing 的 540/550 派发位于窗口之外。'
    ),
    window_needed_nodes=({'event_id': 'wideband_jamming', 'tick': 240, 'effective_tick': 241, 'role': 'factual_anchor_root'},
 {'event_id': 'multi_uav_hold_entry', 'tick': 270, 'role': 'factual_hold_dispatch'},
 {'event_id': 'move_uav_a_jam_hold',
  'tick': 270,
  'entity_id': 'uav_digital_l6_4_v2',
  'terminal_state_tick': 310,
  'role': 'factual_primary_hold_move'},
 {'event_id': 'move_uav_b_jam_hold',
  'tick': 270,
  'entity_id': 'uav_digital_l6_4_v2_secondary',
  'terminal_state_tick': 387,
  'role': 'factual_secondary_hold_move'},
 {'event_id': 'multi_uav_safe_hold',
  'tick': 400,
  'effective_ticks': [401, 405],
  'role': 'factual_safe_hold'},
 {'event_id': 'alternate_channel', 'tick': 460, 'effective_tick': 461, 'role': 'factual_backup_mode'},
 {'event_id': 'move_uav_a_channel_recover',
  'tick': 460,
  'entity_id': 'uav_digital_l6_4_v2',
  'terminal_state_tick': 486,
  'role': 'factual_primary_recovery_move'},
 {'event_id': 'move_uav_b_channel_recover',
  'tick': 460,
  'entity_id': 'uav_digital_l6_4_v2_secondary',
  'terminal_state_tick': 484,
  'role': 'factual_secondary_recovery_move'}),
    window_censored=False,
    group_suffix="",
)

if __name__ == "__main__":
    run_arm(SPEC)
