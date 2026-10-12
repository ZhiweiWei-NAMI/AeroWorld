"""Dedicated ARM entry for L6-3_v2 / seed00 / safety_route_lock / arm_correction_skipped.

保留物理偏航，跳过纠偏，信号恢复后仍明确保留未恢复航线状态。
窗口 [330, 390]，fork=335，采样相位为
after_state_sample_before_actions。

业务假设：UAV 已在偏置端点 [6643.475, 6539.959, 30.0] 完成偏航并于 331 首次静止。safety_route_lock@335 保留 navigation_lock 视觉动作；该动作不会截断运动，静止来自此前偏置航段已经完成。纠偏移动、纠偏截图与本版本的 safety 状态动作被省略。spoof_recovery@375 仍将塔台恢复 online，但源成功补丁被省略；替代补丁在 380 清除 gnss_spoofed，同时保持 route_uncertain=true、mission_recovered=false、route_recovered=false。偏差持续到窗口末。

窗口依据：窗口 [330, 390] 使用 start=330、fork=335，覆盖纠偏省略、spoof_recovery@375 与替代补丁@380。窗口末保留两个正式采样点 385/390 证明信号已清除而物理偏差与 route_uncertain 仍持续。该负结果在窗口右边界仍未恢复，因此 window_censored=true；landing@455 不参与本问题。

证伪条件：若 move_uav_spoof_corrected 或 capture_spoof_correction 执行，UAV 在 335 后向 C 点移动，380 后 route_uncertain 不为 true、mission_recovered/route_recovered 任一为 true，或航线偏差在 390 前消失，则该分支被证伪。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-3_v2"
SEED = "seed00"
ANCHOR_EVENT_ID = "safety_route_lock"
ARM_ID = "arm_correction_skipped"
MODE = "intervention"
ROLE = "narrative_branch"
UAV = "uav_digital_l6_3_v2"
# This dedicated entry declares its own exact branch rewrite.
CANCEL_EVENT_IDS = ()
OMIT_ACTION_IDS = tuple(['move_uav_spoof_corrected',
 'set_runtime_state_l6_3_v2_safety_route_lock_00',
 'capture_spoof_correction',
 'set_runtime_state_l6_3_v2_spoof_recovery_00'])
INJECTIONS = tuple([{'tick': 375,
  'type': 'set_runtime_state',
  'action_id': 'arm_route_recovery_unresolved',
  'entity_id': 'uav_digital_l6_3_v2',
  'delay_ticks': 5,
  'state_patch': {'navigation_state': {'gnss_spoofed': False,
                                       'visual_relocalization': False,
                                       'mission_recovered': False,
                                       'route_uncertain': True,
                                       'route_recovered': False,
                                       'operational_state': 'relocalizing'}}}])
MECHANISM = {'kind': 'physical_correction_skipped_with_unresolved_route_recovery_state',
 'cancel_event_ids': [],
 'omit_action_ids': ['move_uav_spoof_corrected',
                     'set_runtime_state_l6_3_v2_safety_route_lock_00',
                     'capture_spoof_correction',
                     'set_runtime_state_l6_3_v2_spoof_recovery_00'],
 'injections': [{'tick': 375,
                 'type': 'set_runtime_state',
                 'action_id': 'arm_route_recovery_unresolved',
                 'entity_id': 'uav_digital_l6_3_v2',
                 'delay_ticks': 5,
                 'state_patch': {'navigation_state': {'gnss_spoofed': False,
                                                      'visual_relocalization': False,
                                                      'mission_recovered': False,
                                                      'route_uncertain': True,
                                                      'route_recovered': False,
                                                      'operational_state': 'relocalizing'}}}]}

SPEC = ArmSpec(
    profile=PROFILE,
    seed=SEED,
    anchor_event_id=ANCHOR_EVENT_ID,
    arm_id=ARM_ID,
    mode=MODE,
    role=ROLE,
    window_start_tick=330,
    window_fork_tick=335,
    window_end_tick=390,
    primary_predicate='positioning.gnss_spoofing_active',
    subject_entity_ids=(UAV,),
    mechanism=MECHANISM,
    retime=None,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis='UAV 已在偏置端点 [6643.475, 6539.959, 30.0] 完成偏航并于 331 首次静止。safety_route_lock@335 保留 navigation_lock 视觉动作；该动作不会截断运动，静止来自此前偏置航段已经完成。纠偏移动、纠偏截图与本版本的 safety 状态动作被省略。spoof_recovery@375 仍将塔台恢复 online，但源成功补丁被省略；替代补丁在 380 清除 gnss_spoofed，同时保持 route_uncertain=true、mission_recovered=false、route_recovered=false。偏差持续到窗口末。',
    falsified_if='若 move_uav_spoof_corrected 或 capture_spoof_correction 执行，UAV 在 335 后向 C 点移动，380 后 route_uncertain 不为 true、mission_recovered/route_recovered 任一为 true，或航线偏差在 390 前消失，则该分支被证伪。',
    window_basis='窗口 [330, 390] 使用 start=330、fork=335，覆盖纠偏省略、spoof_recovery@375 与替代补丁@380。窗口末保留两个正式采样点 385/390 证明信号已清除而物理偏差与 route_uncertain 仍持续。该负结果在窗口右边界仍未恢复，因此 window_censored=true；landing@455 不参与本问题。',
    window_needed_nodes=tuple([{'event_id': 'safety_route_lock', 'tick': 335, 'role': 'anchor_event_shell_retained'},
 {'event_id': 'set_uav_nav_lock',
  'tick': 335,
  'entity_id': 'uav_digital_l6_3_v2',
  'role': 'retained_visual_navigation_lock'},
 {'event_id': 'move_uav_spoof_corrected',
  'tick': 335,
  'entity_id': 'uav_digital_l6_3_v2',
  'role': 'omitted_physical_correction'},
 {'event_id': 'capture_spoof_correction', 'tick': 335, 'role': 'omitted_false_correction_capture'},
 {'event_id': 'spoof_recovery', 'tick': 375, 'role': 'retained_signal_recovery'},
 {'event_id': 'set_runtime_state_l6_3_v2_spoof_recovery_00',
  'tick': 380,
  'entity_id': 'uav_digital_l6_3_v2',
  'role': 'omitted_false_route_success_patch'},
 {'event_id': 'arm_route_recovery_unresolved',
  'tick': 380,
  'entity_id': 'uav_digital_l6_3_v2',
  'role': 'injected_unresolved_route_state'}]),
    window_censored=True,
    completion_requirements=tuple([{'predicate_id': 'positioning.aircraft_deviates_from_planned_route',
  'bindings': {'actor': 'uav_digital_l6_3_v2'},
  'required_value': 'false'}]),
    group_suffix='',
)

if __name__ == "__main__":
    run_arm(SPEC)
