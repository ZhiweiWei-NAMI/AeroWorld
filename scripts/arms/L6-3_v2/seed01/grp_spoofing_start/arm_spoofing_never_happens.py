"""Dedicated ARM entry for L6-3_v2 / seed01 / spoofing_start / arm_spoofing_never_happens.

取消 GNSS 欺骗根事件，并沿场景计划航线继续。
窗口 [235, 390]，fork=240，采样相位为
after_state_sample_before_actions。

业务假设：spoofing_start 被精确取消；依赖它的事件因父事件未发生而自然不触发。tick 241 从当前 A 点沿场景 planned_route_waypoints 的 B=[6628.475, 6529.959, 31.0]、C=[6624.475, 6543.959, 32.0] 以 5.0 m/s 继续。A→B 长 14.456832 m、需 29 ticks；B→C 长 14.594520 m、需 30 ticks；tick 300 到达 C，301 首次静止。GNSS 欺骗、偏航、纠偏与恢复动作均不发生。

窗口依据：窗口 [235, 390] 与同组 factual 共用 start=235、fork=240。它覆盖取消锚点、241 的名义续航、300 到达、301 保持，以及原响应链的 270/335/375 检查点；landing@455 位于窗口外。

证伪条件：若 spoofing_start 或任一依赖响应事件发生，gnss_spoofed 取 true，名义续航未在 300 到达 C，或 planned-route deviation 达到 15 m，则该分支被证伪。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-3_v2"
SEED = "seed01"
ANCHOR_EVENT_ID = "spoofing_start"
ARM_ID = "arm_spoofing_never_happens"
MODE = "intervention"
ROLE = "narrative_branch"
UAV = "uav_digital_l6_3_v2"
# This dedicated entry declares its own exact branch rewrite.
CANCEL_EVENT_IDS = tuple(['spoofing_start'])
OMIT_ACTION_IDS = ()
INJECTIONS = tuple([{'tick': 241,
  'type': 'move_entity',
  'action_id': 'arm_route_continuation',
  'entity_id': 'uav_digital_l6_3_v2',
  'velocity_mps': 5.0,
  'waypoints_enu_m': [[6628.475, 6529.959, 31.0], [6624.475, 6543.959, 32.0]]}])
MECHANISM = {'kind': 'root_event_cancelled_with_nominal_route_continuation',
 'cancel_event_ids': ['spoofing_start'],
 'omit_action_ids': [],
 'injections': [{'tick': 241,
                 'type': 'move_entity',
                 'action_id': 'arm_route_continuation',
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
    window_start_tick=235,
    window_fork_tick=240,
    window_end_tick=390,
    primary_predicate='positioning.gnss_spoofing_active',
    subject_entity_ids=(UAV,),
    mechanism=MECHANISM,
    retime=None,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis='spoofing_start 被精确取消；依赖它的事件因父事件未发生而自然不触发。tick 241 从当前 A 点沿场景 planned_route_waypoints 的 B=[6628.475, 6529.959, 31.0]、C=[6624.475, 6543.959, 32.0] 以 5.0 m/s 继续。A→B 长 14.456832 m、需 29 ticks；B→C 长 14.594520 m、需 30 ticks；tick 300 到达 C，301 首次静止。GNSS 欺骗、偏航、纠偏与恢复动作均不发生。',
    falsified_if='若 spoofing_start 或任一依赖响应事件发生，gnss_spoofed 取 true，名义续航未在 300 到达 C，或 planned-route deviation 达到 15 m，则该分支被证伪。',
    window_basis='窗口 [235, 390] 与同组 factual 共用 start=235、fork=240。它覆盖取消锚点、241 的名义续航、300 到达、301 保持，以及原响应链的 270/335/375 检查点；landing@455 位于窗口外。',
    window_needed_nodes=tuple([{'event_id': 'spoofing_start', 'tick': 240, 'role': 'explicitly_cancelled_anchor'},
 {'event_id': 'arm_route_continuation',
  'tick': 241,
  'entity_id': 'uav_digital_l6_3_v2',
  'role': 'injected_nominal_route'},
 {'event_id': 'spoofed_route_offset', 'tick': 270, 'role': 'naturally_absent_without_parent'},
 {'event_id': 'safety_route_lock', 'tick': 335, 'role': 'naturally_absent_without_parent'},
 {'event_id': 'spoof_recovery', 'tick': 375, 'role': 'naturally_absent_without_parent'}]),
    window_censored=False,
    group_suffix='',
)

if __name__ == "__main__":
    run_arm(SPEC)
