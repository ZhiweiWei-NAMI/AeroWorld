"""Dedicated ARM entry for L6-1_v1 / seed02 / grp_rth_engaged / arm_rth_cancelled.

Window: [270, 340], shared fork 275, after_state_sample_before_actions.
Mechanism: omit_rth_response_and_hold_at_fork.
Expected process: rth_engaged 事件仍在 275 派发，但其三项响应动作被省略并注入 hold，因此 UAV 从 fork 位置保持、rth_active 始终为 false。c2_recovered 仍在 325 派发；保留 tower 恢复及通信恢复，只省略会覆盖本分支显式 hold 状态的 set_uav_link_restored_hold。
Window basis: 闭区间 [270,340] 与 factual 共享 tick 275 动作前快照，覆盖被省略的 RTH 响应和仍保留的链路恢复。省略 set_uav_link_restored_hold 保留本分支显式 hold 状态；当前位置悬停不代表到家；landing@405 在范围外，不评价着陆。
Falsification: 若 UAV 在 275 后移动或进入 return_to_home，rth_active 变为 true，c2_recovered 未在 325 派发，或通信未在 330 恢复并稳定至 340，则分支被证伪。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-1_v1"
SEED = "seed02"
ANCHOR_EVENT_ID = "rth_engaged"
ARM_ID = "arm_rth_cancelled"
MODE = "intervention"
ROLE = "narrative_branch"
UAV = "uav_digital_l6_1_v1"

# The single exact action: at the fork the UAV is held in place, superseding the
# authored return-to-home motion already dispatched at 274. uav-mode hold sets
# stop_motion=true, which truncates the in-flight move_uav_rth_home trajectory.
INJECTIONS = ({'tick': 275,
               'type': 'set_visual_state',
               'action_id': 'arm_rth_hold',
               'entity_id': 'uav_digital_l6_1_v1',
               'visual_state': {'mode': 'hold'}},)

MECHANISM = {'kind': 'omit_rth_response_and_hold_at_fork',
             'cancel_event_ids': [],
             'omit_action_ids': ['set_uav_rth',
                                 'move_uav_rth_home',
                                 'set_runtime_state_l6_1_v1_rth_engaged_00',
                                 'set_uav_link_restored_hold'],
             'injections': [{'tick': 275,
                             'type': 'set_visual_state',
                             'action_id': 'arm_rth_hold',
                             'entity_id': 'uav_digital_l6_1_v1',
                             'visual_state': {'mode': 'hold'}}]}

HYPOTHESIS = (
    ('rth_engaged 事件仍在 275 派发，但其三项响应动作被省略并注入 hold，因此 UAV 从 fork 位置保持、rth_active 始终为 '
     'false。c2_recovered 仍在 325 派发；保留 tower 恢复及通信恢复，只省略会覆盖本分支显式 hold 状态的 '
     'set_uav_link_restored_hold。')
)
FALSIFIED_IF = (
    ('若 UAV 在 275 后移动或进入 return_to_home，rth_active 变为 true，c2_recovered 未在 325 派发，或通信未在 '
     '330 恢复并稳定至 340，则分支被证伪。')
)
WINDOW_BASIS = (
    ('闭区间 [270,340] 与 factual 共享 tick 275 动作前快照，覆盖被省略的 RTH 响应和仍保留的链路恢复。省略 set_uav_link_restored_hold '
     '保留本分支显式 hold 状态；当前位置悬停不代表到家；landing@405 在范围外，不评价着陆。')
)
WINDOW_NEEDED_NODES = ({'event_id': 'rth_engaged', 'tick': 275, 'role': 'authored_event_actions_omitted'},
                       {'action_id': 'arm_rth_hold', 'tick': 275, 'role': 'injected_stationary_hold'},
                       {'event_id': 'c2_recovered', 'tick': 325, 'role': 'authored_recovery_retained'},
                       {'predicate_id': 'communication.link_unavailable',
                        'tick': 340,
                        'role': 'recovery_hold_terminal'})

SPEC = ArmSpec(
    profile=PROFILE,
    seed=SEED,
    anchor_event_id=ANCHOR_EVENT_ID,
    arm_id=ARM_ID,
    mode=MODE,
    role=ROLE,
    window_start_tick=270,
    window_fork_tick=275,
    window_end_tick=340,
    primary_predicate="communication.link_unavailable",
    subject_entity_ids=('tower_l6_1_v1', 'uav_digital_l6_1_v1'),
    mechanism=MECHANISM,
    retime=None,
    omit_action_ids=('set_uav_rth',
                     'move_uav_rth_home',
                     'set_runtime_state_l6_1_v1_rth_engaged_00',
                     'set_uav_link_restored_hold'),
    inject_actions=INJECTIONS,
    cancel_event_ids=(),
    hypothesis=HYPOTHESIS,
    falsified_if=FALSIFIED_IF,
    window_basis=WINDOW_BASIS,
    window_needed_nodes=WINDOW_NEEDED_NODES,
    window_censored=False,
    group_suffix="",
)

if __name__ == "__main__":
    run_arm(SPEC)
