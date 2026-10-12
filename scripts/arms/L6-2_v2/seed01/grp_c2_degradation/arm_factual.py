"""Dedicated factual ARM entry for L6-2_v2 / seed01 / grp_c2_degradation.

The current source script runs without cancellation, omission, retiming, or injection.
The closed window is [395, 510]; the fork snapshot is sampled before actions at tick
400. Actual dispatches are c2_degradation@400,
uav_delayed_response@430, backup_link_lock@460, and nominal_c2_restore@495.
Typed nominal recovery is visible at 500. The formal C2 lifecycle recovers at tick 475 after false samples 465/470/475.
The ordinary landing dispatch at 575 is outside this experiment and is not claimed here.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-2_v2"
SEED = "seed01"
ANCHOR_EVENT_ID = "c2_degradation"
ARM_ID = "arm_factual"
MODE = "factual_observation"
ROLE = "factual"
UAV = "uav_digital_l6_2_v2"
TOWER = "tower_l6_2_v2"

# Executable values for this dedicated ARM; mirrored in MECHANISM and ArmSpec.
CANCEL_EVENT_IDS = ()
OMIT_ACTION_IDS = ()
INJECTIONS = ()
RETIME = None

MECHANISM = {'kind': 'factual_no_intervention', 'cancel_event_ids': [], 'omit_action_ids': [], 'injections': []}

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
    retime=RETIME,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis=(
        '当前引擎无干预执行窗口 [395, 510]：c2_degradation@400 使 tower_l6_2_v2 的 visual 与 typed mode 从 401 起为 degraded，communication.link_degraded 于正式样本 405 onset；uav_delayed_response@430 产生 431–456 的 2.929144 m/s 插值慢段，457 停止；backup_link_lock@460 产生 461–489 的 5.386379 m/s 插值恢复段，490 停止；nominal_c2_restore 实际派发于 495，tower visual 从 496 起 online，typed mode 与 uav_digital_l6_2_v2 的 handover/backup/slowdown 字段从 500 起恢复。正式谓词从 465 起为 false，并在 465/470/475 三个正式样本后于 475 完成恢复。普通 landing 实际派发于 575，位于本窗口之外。'
    ),
    falsified_if=(
        '若空机制下 fork 前状态与同组干预不一致，或 c2_degradation 未在 400 派发、正式 onset 不在 405，或 430/460/495 的源动作序列和上述运动区间不符，或 500 的 typed 恢复状态不符，则 factual 不成立。'
    ),
    window_basis=(
        '闭区间 [395,510]；start=fork−5，fork=400 为动作前共享状态。end=510 覆盖 nominal typed 恢复首次可见的 500 以及 500/505/510 三个正式恢复样本，同时排除 575 普通 landing 和其后 590 altitude excursion。'
    ),
    window_needed_nodes=(
        {'event_id': 'c2_degradation',
         'kind': 'authored_event',
         'tick': 400,
         'entity_id': 'tower_l6_2_v2',
         'role': 'factual_anchor'},
        {'event_id': 'uav_delayed_response',
         'kind': 'authored_event',
         'tick': 430,
         'entity_id': 'uav_digital_l6_2_v2',
         'role': 'factual_successor'},
        {'event_id': 'backup_link_lock',
         'kind': 'authored_event',
         'tick': 460,
         'entity_id': 'tower_l6_2_v2',
         'role': 'factual_successor'},
        {'event_id': 'nominal_c2_restore',
         'kind': 'authored_event',
         'tick': 495,
         'entity_id': 'tower_l6_2_v2',
         'role': 'factual_successor'},
    ),
    window_censored=False,
    group_suffix="",
)

if __name__ == "__main__":
    run_arm(SPEC)
