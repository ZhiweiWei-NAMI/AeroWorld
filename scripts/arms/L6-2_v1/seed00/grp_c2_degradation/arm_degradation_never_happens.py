"""Dedicated intervention ARM entry for L6-2_v1 / seed00 / grp_c2_degradation.

Only c2_degradation is explicitly cancelled at the tick-400 fork. Its dependent
triggers remain authored and do not dispatch because their conditions are unsatisfied.
The UAV had already completed precheck motion before this window, so the measured
consequence in [395, 510] is a stationary airborne task hold with online tower state and
no C2 degradation occurrence. Landing@575 is out of scope.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-2_v1"
SEED = "seed00"
ANCHOR_EVENT_ID = "c2_degradation"
ARM_ID = "arm_degradation_never_happens"
MODE = "intervention"
ROLE = "narrative_branch"
UAV = "uav_digital_l6_2_v1"
TOWER = "tower_l6_2_v1"

# Executable values for this dedicated ARM; mirrored in MECHANISM and ArmSpec.
CANCEL_EVENT_IDS = (
    'c2_degradation',
)
OMIT_ACTION_IDS = ()
INJECTIONS = ()

MECHANISM = {'kind': 'explicit_root_cancel_without_implicit_successor_closure',
 'cancel_event_ids': ['c2_degradation'],
 'omit_action_ids': [],
 'injections': []}

SPEC = ArmSpec(
    profile=PROFILE,
    seed=SEED,
    anchor_event_id=ANCHOR_EVENT_ID,
    arm_id=ARM_ID,
    mode=MODE,
    role=ROLE,
    window_start_tick=395,
    window_fork_tick=400,
    window_end_tick=510,
    primary_predicate="communication.link_degraded",
    subject_entity_ids=(UAV,),
    mechanism=MECHANISM,
    retime=None,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis=(
        '只显式取消 c2_degradation。400 以后其 event_fired_after 后继因触发条件未满足而不派发；它们不是隐式 cancel 列表。在 [395,510] 内 tower_l6_2_v1 的 visual/typed mode 保持 online，uav_digital_l6_2_v1 保持 airborne task hold、速度 0，handover_active=false、backup_link_active=false、slowdown_active=false，communication.link_degraded 全窗 false且无 c2 occurrence。普通 landing@575 在窗外，不作发生或完成承诺。'
    ),
    falsified_if=(
        '若 c2_degradation 的任一源动作执行，或 tower_l6_2_v1 离开 online，或 uav_digital_l6_2_v1 在 fork 后出现非零速度、slowdown/handover/backup 为 true，或 communication.link_degraded 出现 true，或 430/460/495 任一依赖事件被派发，则该分支被证伪。'
    ),
    window_basis=(
        '闭区间 [395,510]；395 是 fork 前一个正式样本，400 是被取消根事件的动作前共享状态。同组 factual 需要观察到 500 的 typed 恢复和 510 的三样本 hold，故两臂共用 end=510；575 landing 在范围外。'
    ),
    window_needed_nodes=(
        {'event_id': 'c2_degradation',
         'kind': 'cancelled_authored_event',
         'tick': 400,
         'entity_id': 'tower_l6_2_v1',
         'role': 'explicitly_cancelled_anchor'},
        {'event_id': 'uav_delayed_response',
         'kind': 'authored_event',
         'tick': 430,
         'entity_id': 'uav_digital_l6_2_v1',
         'role': 'expected_not_dispatched_trigger_unsatisfied'},
        {'event_id': 'backup_link_lock',
         'kind': 'authored_event',
         'tick': 460,
         'entity_id': 'tower_l6_2_v1',
         'role': 'expected_not_dispatched_trigger_unsatisfied'},
        {'event_id': 'nominal_c2_restore',
         'kind': 'authored_event',
         'tick': 495,
         'entity_id': 'tower_l6_2_v1',
         'role': 'expected_not_dispatched_trigger_unsatisfied'},
    ),
    window_censored=False,
    group_suffix="",
)

if __name__ == "__main__":
    run_arm(SPEC)
