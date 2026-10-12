'Dedicated ARM entry for L6-1_v2 / seed02 / grp_rth_engaged / arm_factual.\n\nWindow: [270, 340], shared fork 275, after_state_sample_before_actions.\nMechanism: factual_baseline_no_mechanism.\nExpected process: 无干预运行中，rth_engaged 在 275 派发。返航单段朝实际 home pad 飞行，但 c2_recovered@325 将运动截断为原地悬停；未抵达 home，更未着陆。c2_recovered 在 325 派发，通信不可用和 rth_active 于 330 清除，C2 事件于 340 完成恢复保持。\nWindow basis: 闭区间 [270,340] 覆盖失联响应、实际返航与链路恢复保持，不进入 landing@405。返航段长 50.281000 m，7 m/s、10 Hz 下完整航段需 72 tick，理论到达 347；实际 325 被恢复动作截断，326 首次静止，距 home 水平 15.363639 m。恢复运行状态的 delay_ticks=5，330 生效；动作说明与 delay_ticks=5 一致，状态在派发后 5 tick 生效。后续着陆保留原速度和 home36→home3，从当时实际位置接入，不把窗外着陆算作本窗结果。\nFalsification: 本臂是当前脚本和当前引擎的 factual 观测；实跑不能复现该时序时，本组对照失效。\n'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-1_v2"
SEED = "seed02"
ANCHOR_EVENT_ID = "rth_engaged"
ARM_ID = "arm_factual"
MODE = "factual_observation"
ROLE = "factual"
UAV = "uav_digital_l6_1_v2"

# Factual: no trigger cascade is cancelled, no action is omitted or injected, no
# trigger is retimed. The engine runs the authored script unchanged from the
# shared prefix; every downstream successor fires at its authored tick.
CANCEL_EVENT_IDS = ()
OMIT_ACTION_IDS = ()
INJECTIONS = ()

MECHANISM = {'kind': 'factual_baseline_no_mechanism',
             'cancel_event_ids': [],
             'omit_action_ids': [],
             'injections': []}

HYPOTHESIS = (
    ('无干预运行中，rth_engaged 在 275 派发。返航单段朝实际 home pad 飞行，但 c2_recovered@325 将运动截断为原地悬停；未抵达 home，更未着陆。c2_recovered 在 325 派发，通信不可用和 rth_active 于 330 清除，C2 事件于 340 完成恢复保持。')
)
FALSIFIED_IF = (
    '本臂是当前脚本和当前引擎的 factual 观测；实跑不能复现该时序时，本组对照失效。'
)
WINDOW_BASIS = (
    ('闭区间 [270,340] 覆盖失联响应、实际返航与链路恢复保持，不进入 landing@405。返航段长 50.281000 m，7 m/s、10 Hz 下完整航段需 72 tick，理论到达 347；实际 325 被恢复动作截断，326 首次静止，距 home 水平 15.363639 m。恢复运行状态的 delay_ticks=5，330 生效；动作说明与 delay_ticks=5 一致，状态在派发后 5 tick 生效。后续着陆保留原速度和 home36→home3，从当时实际位置接入，不把窗外着陆算作本窗结果。')
)
WINDOW_NEEDED_NODES = ({'event_id': 'rth_engaged', 'tick': 275, 'role': 'authored_anchor'}, {'motion_id': 'move_uav_rth_home', 'tick': 325, 'role': 'interrupted_by_link_restored_hold_before_home'}, {'event_id': 'c2_recovered', 'tick': 325, 'role': 'authored_recovery'}, {'predicate_id': 'communication.link_unavailable', 'tick': 340, 'role': 'recovery_hold_terminal'})

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
    subject_entity_ids=('tower_l6_1_v2', 'uav_digital_l6_1_v2'),
    mechanism=MECHANISM,
    retime=None,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis=HYPOTHESIS,
    falsified_if=FALSIFIED_IF,
    window_basis=WINDOW_BASIS,
    window_needed_nodes=WINDOW_NEEDED_NODES,
    window_censored=False,
    group_suffix="",
)

if __name__ == "__main__":
    run_arm(SPEC)
