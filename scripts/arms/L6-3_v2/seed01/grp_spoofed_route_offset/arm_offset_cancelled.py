"""Dedicated ARM entry for L6-3_v2 / seed01 / spoofed_route_offset / arm_offset_cancelled.

保留欺骗检测，取消物理偏置并移除不存在的纠偏动作。
窗口 [265, 390]，fork=270，采样相位为
after_state_sample_before_actions。

业务假设：spoofing_start 与响应事件的 authored 调度保留。tick 270 省略偏置移动和 route_uncertain 补丁，改为沿计划 B=[6628.475, 6529.959, 31.0]、C=[6624.475, 6543.959, 32.0] 以 5.0 m/s 续航，329 到达、330 首次静止。safety_route_lock@335 仍可执行导航保护与 GNSS 重定位状态，但省略不存在物理纠偏对应的 move_uav_spoof_corrected 和 capture_spoof_correction。航线偏差不达到 15 m。

窗口依据：窗口 [265, 390] 使用 start=265、fork=270，覆盖锚点改写、329 到达、330 保持、safety_route_lock@335、spoof_recovery@375 和恢复补丁@380；landing@455 位于窗口外。

证伪条件：若 route_uncertain 由 offset 事件置 true，UAV 向偏置端点移动，纠偏移动或纠偏截图执行，名义续航未在 329 到达 C，或航线偏差达到 15 m，则该分支被证伪。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-3_v2"
SEED = "seed01"
ANCHOR_EVENT_ID = "spoofed_route_offset"
ARM_ID = "arm_offset_cancelled"
MODE = "intervention"
ROLE = "narrative_branch"
UAV = "uav_digital_l6_3_v2"
# This dedicated entry declares its own exact branch rewrite.
CANCEL_EVENT_IDS = ()
OMIT_ACTION_IDS = tuple(['move_uav_spoofed_offset',
 'set_runtime_state_l6_3_v2_spoofed_route_offset_00',
 'move_uav_spoof_corrected',
 'capture_spoof_correction'])
INJECTIONS = tuple([{'tick': 270,
  'type': 'move_entity',
  'action_id': 'arm_nominal_route_continuation',
  'entity_id': 'uav_digital_l6_3_v2',
  'velocity_mps': 5.0,
  'waypoints_enu_m': [[6628.475, 6529.959, 31.0], [6624.475, 6543.959, 32.0]]}])
MECHANISM = {'kind': 'offset_and_false_correction_actions_omitted_with_nominal_route',
 'cancel_event_ids': [],
 'omit_action_ids': ['move_uav_spoofed_offset',
                     'set_runtime_state_l6_3_v2_spoofed_route_offset_00',
                     'move_uav_spoof_corrected',
                     'capture_spoof_correction'],
 'injections': [{'tick': 270,
                 'type': 'move_entity',
                 'action_id': 'arm_nominal_route_continuation',
                 'entity_id': 'uav_digital_l6_3_v2',
                 'velocity_mps': 5.0,
                 'waypoints_enu_m': [[6628.475, 6529.959, 31.0], [6624.475, 6543.959, 32.0]]}]}

SPEC = ArmSpec(
    profile=PROFILE,
    seed=SEED,
    anchor_event_id=ANCHOR_EVENT_ID,
    arm_id=ARM_ID,
    mode=MODE,
    role=ROLE,
    window_start_tick=265,
    window_fork_tick=270,
    window_end_tick=390,
    primary_predicate='positioning.gnss_spoofing_active',
    subject_entity_ids=(UAV,),
    mechanism=MECHANISM,
    retime=None,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis='spoofing_start 与响应事件的 authored 调度保留。tick 270 省略偏置移动和 route_uncertain 补丁，改为沿计划 B=[6628.475, 6529.959, 31.0]、C=[6624.475, 6543.959, 32.0] 以 5.0 m/s 续航，329 到达、330 首次静止。safety_route_lock@335 仍可执行导航保护与 GNSS 重定位状态，但省略不存在物理纠偏对应的 move_uav_spoof_corrected 和 capture_spoof_correction。航线偏差不达到 15 m。',
    falsified_if='若 route_uncertain 由 offset 事件置 true，UAV 向偏置端点移动，纠偏移动或纠偏截图执行，名义续航未在 329 到达 C，或航线偏差达到 15 m，则该分支被证伪。',
    window_basis='窗口 [265, 390] 使用 start=265、fork=270，覆盖锚点改写、329 到达、330 保持、safety_route_lock@335、spoof_recovery@375 和恢复补丁@380；landing@455 位于窗口外。',
    window_needed_nodes=tuple([{'event_id': 'spoofed_route_offset', 'tick': 270, 'role': 'anchor_event_shell_retained'},
 {'event_id': 'move_uav_spoofed_offset',
  'tick': 270,
  'entity_id': 'uav_digital_l6_3_v2',
  'role': 'omitted_offset_move'},
 {'event_id': 'arm_nominal_route_continuation',
  'tick': 270,
  'entity_id': 'uav_digital_l6_3_v2',
  'role': 'injected_nominal_route'},
 {'event_id': 'safety_route_lock', 'tick': 335, 'role': 'retained_protective_response'},
 {'event_id': 'move_uav_spoof_corrected',
  'tick': 335,
  'entity_id': 'uav_digital_l6_3_v2',
  'role': 'omitted_nonexistent_correction'},
 {'event_id': 'capture_spoof_correction',
  'tick': 335,
  'role': 'omitted_nonexistent_correction_capture'},
 {'event_id': 'spoof_recovery', 'tick': 375, 'role': 'retained_signal_recovery'}]),
    window_censored=False,
    group_suffix='',
)

if __name__ == "__main__":
    run_arm(SPEC)
