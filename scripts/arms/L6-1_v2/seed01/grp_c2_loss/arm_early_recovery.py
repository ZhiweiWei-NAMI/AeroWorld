"""Dedicated ARM entry for L6-1_v2 / seed01 / grp_c2_loss / arm_early_recovery.

Window: [235, 340], shared fork 240, after_state_sample_before_actions.
Mechanism: restore_link_before_rth_and_cancel_rth_event.
Expected process: c2_loss 仍在 240 发生；tick 250 显式恢复 tower 可视状态并排队恢复 UAV 通信，二者自 251 的采样一致为 online/available。仅显式取消 rth_engaged，c2_recovered 及 landing 因依赖事件未发生而自然不触发。
Window basis: 闭区间 [235,340] 保留 c2_loss 前共享样本、提前恢复和至 340 的稳定结果。源 c2_loss 任务段在 271 完成后保持；本分支不声称继续巡航。只取消 rth_engaged，后继依赖按真实非触发传播；landing@405 在范围外。
Falsification: 若 251 起 tower 仍为 link_lost 或通信仍不可用，rth_engaged/c2_recovered 仍触发，UAV 进入 RTH，或源 c2_loss 任务段未在 271 完成后保持，则分支被证伪。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-1_v2"
SEED = "seed01"
ANCHOR_EVENT_ID = "c2_loss"
ARM_ID = "arm_early_recovery"
MODE = "intervention"
ROLE = "narrative_branch"
UAV = "uav_digital_l6_1_v2"

# Cancel only rth_engaged; its dependent successors then remain unsatisfied.
CANCEL_EVENT_IDS = ('rth_engaged',)

# Restore the tower visual state and UAV communication state at the declared early-recovery tick.
INJECTIONS = ({'tick': 250,
               'type': 'set_visual_state',
               'action_id': 'arm_early_recovery_tower_online',
               'entity_id': 'tower_l6_1_v2',
               'visual_state': {'mode': 'online'}},
              {'tick': 250,
               'type': 'set_runtime_state',
               'action_id': 'arm_early_recovery_comm_available',
               'entity_id': 'uav_digital_l6_1_v2',
               'delay_ticks': 1,
               'state_patch': {'communication_state': {'communication_unavailable': False}}})

MECHANISM = {'kind': 'restore_link_before_rth_and_cancel_rth_event',
             'cancel_event_ids': ['rth_engaged'],
             'omit_action_ids': [],
             'injections': [{'tick': 250,
                             'type': 'set_visual_state',
                             'action_id': 'arm_early_recovery_tower_online',
                             'entity_id': 'tower_l6_1_v2',
                             'visual_state': {'mode': 'online'}},
                            {'tick': 250,
                             'type': 'set_runtime_state',
                             'action_id': 'arm_early_recovery_comm_available',
                             'entity_id': 'uav_digital_l6_1_v2',
                             'delay_ticks': 1,
                             'state_patch': {'communication_state': {'communication_unavailable': False}}}]}

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
    subject_entity_ids=('tower_l6_1_v2', 'uav_digital_l6_1_v2'),
    mechanism=MECHANISM,
    retime=None,
    omit_action_ids=(),
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis=(
        ('c2_loss 仍在 240 发生；tick 250 显式恢复 tower 可视状态并排队恢复 UAV 通信，二者自 251 的采样一致为 online/available。仅显式取消 '
         'rth_engaged，c2_recovered 及 landing 因依赖事件未发生而自然不触发。')
    ),
    falsified_if=(
        ('若 251 起 tower 仍为 link_lost 或通信仍不可用，rth_engaged/c2_recovered 仍触发，UAV 进入 RTH，或源 c2_loss 任务段未在 '
         '271 完成后保持，则分支被证伪。')
    ),
    window_basis=(
        ('闭区间 [235,340] 保留 c2_loss 前共享样本、提前恢复和至 340 的稳定结果。源 c2_loss 任务段在 271 完成后保持；本分支不声称继续巡航。只取消 '
         'rth_engaged，后继依赖按真实非触发传播；landing@405 在范围外。')
    ),
    window_needed_nodes=({'event_id': 'c2_loss', 'tick': 240, 'role': 'authored_anchor'},
                         {'action_id': 'arm_early_recovery_tower_online',
                          'tick': 250,
                          'role': 'injected_visual_recovery'},
                         {'action_id': 'arm_early_recovery_comm_available',
                          'tick': 250,
                          'role': 'injected_runtime_recovery'},
                         {'event_id': 'rth_engaged', 'tick': 275, 'role': 'explicitly_cancelled_event'},
                         {'event_id': 'c2_recovered', 'tick': 325, 'role': 'expected_absent_dependent'}),
    window_censored=False,
    group_suffix="",
)

if __name__ == "__main__":
    run_arm(SPEC)
