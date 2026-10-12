"""Dedicated ARM entry for L6-1_v1 / seed00 / grp_c2_loss / arm_c2_loss_never_happens.

Window: [235, 340], shared fork 240, after_state_sample_before_actions.
Mechanism: cancel_c2_loss_and_continue_existing_mission_segment.
Expected process: 在共享 fork tick 240 显式取消 c2_loss，并从该状态继续原任务段至既有终点。tower 保持 online，communication_unavailable 和 rth_active 全窗为 false；rth_engaged 与 c2_recovered 因父事件未发生而自然不触发。
Window basis: 闭区间 [235,340] 与 factual 共享 tick 240 动作前快照，覆盖根事件缺失的直接后果。续航段直线距离 16.248077 m，6 m/s、10 Hz 下 ceil(16.248077/6×10)=28 tick，从 240 至 268 完成，随后在终点保持；不把终点保持写成全窗持续巡航。landing@405 在范围外。
Falsification: 若 c2_loss 或其依赖事件仍触发，tower 离开 online，通信或 RTH 状态变为 true，或注入路径未在 268 完成，则分支被证伪。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-1_v1"
SEED = "seed00"
ANCHOR_EVENT_ID = "c2_loss"
ARM_ID = "arm_c2_loss_never_happens"
MODE = "intervention"
ROLE = "narrative_branch"
UAV = "uav_digital_l6_1_v1"

# Cancel only the authored c2_loss root; its event_fired_after dependents then remain unsatisfied.
CANCEL_EVENT_IDS = ('c2_loss',)

# Continue the existing mission segment from the shared fork; the UAV holds after reaching its endpoint.
INJECTIONS = ({'tick': 240,
               'type': 'move_entity',
               'action_id': 'arm_cruise_continuation',
               'entity_id': 'uav_digital_l6_1_v1',
               'velocity_mps': 6.0,
               'waypoints_enu_m': [[6667.742, 6549.248, 30.0]]},)

MECHANISM = {'kind': 'cancel_c2_loss_and_continue_existing_mission_segment',
             'cancel_event_ids': ['c2_loss'],
             'omit_action_ids': [],
             'injections': [{'tick': 240,
                             'type': 'move_entity',
                             'action_id': 'arm_cruise_continuation',
                             'entity_id': 'uav_digital_l6_1_v1',
                             'velocity_mps': 6.0,
                             'waypoints_enu_m': [[6667.742, 6549.248, 30.0]]}]}

SPEC = ArmSpec(
    profile=PROFILE,
    seed=SEED,
    anchor_event_id=ANCHOR_EVENT_ID,
    arm_id=ARM_ID,
    mode=MODE,
    role=ROLE,
    window_start_tick=235,
    window_fork_tick=240,
    window_end_tick=340,
    primary_predicate="communication.link_unavailable",
    subject_entity_ids=('tower_l6_1_v1', 'uav_digital_l6_1_v1'),
    mechanism=MECHANISM,
    retime=None,
    omit_action_ids=(),
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis=(
        ('在共享 fork tick 240 显式取消 c2_loss，并从该状态继续原任务段至既有终点。tower 保持 online，communication_unavailable 和 '
         'rth_active 全窗为 false；rth_engaged 与 c2_recovered 因父事件未发生而自然不触发。')
    ),
    falsified_if=(
        '若 c2_loss 或其依赖事件仍触发，tower 离开 online，通信或 RTH 状态变为 true，或注入路径未在 268 完成，则分支被证伪。'
    ),
    window_basis=(
        ('闭区间 [235,340] 与 factual 共享 tick 240 动作前快照，覆盖根事件缺失的直接后果。续航段直线距离 16.248077 m，6 m/s、10 Hz 下 '
         'ceil(16.248077/6×10)=28 tick，从 240 至 268 完成，随后在终点保持；不把终点保持写成全窗持续巡航。landing@405 在范围外。')
    ),
    window_needed_nodes=({'event_id': 'c2_loss', 'tick': 240, 'role': 'explicitly_cancelled_anchor'},
                         {'action_id': 'arm_cruise_continuation',
                          'tick': 240,
                          'role': 'injected_existing_segment_continuation'},
                         {'event_id': 'rth_engaged', 'tick': 275, 'role': 'expected_absent_dependent'},
                         {'event_id': 'c2_recovered', 'tick': 325, 'role': 'expected_absent_dependent'}),
    window_censored=False,
    group_suffix="",
)

if __name__ == "__main__":
    run_arm(SPEC)
