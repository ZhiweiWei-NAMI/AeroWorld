'Dedicated ARM entry for L6-1_v1 / seed00 / grp_c2_loss / arm_factual.\n\nWindow: [235, 340], shared fork 240, after_state_sample_before_actions.\nMechanism: factual_baseline_no_mechanism.\nExpected process: 无干预运行中，c2_loss 在 240 派发，链路不可用自 241 生效；rth_engaged 在 275 派发。返航单段朝实际 home pad 飞行，在 283 到达 home 正上方 30 m，随后悬停；该窗口未着陆。c2_recovered 在 325 派发，通信不可用和 rth_active 于 330 清除，C2 事件于 340 完成恢复保持。\nWindow basis: 闭区间 [235,340] 覆盖失联响应、实际返航与链路恢复保持，不进入 landing@405。返航段长 5.048194 m，7 m/s、10 Hz 下 ceil(distance/7×10)=8 tick，275→283；at_home_pad 只判水平邻近，不等于 touchdown。恢复运行状态的 delay_ticks=5，330 生效；动作说明与 delay_ticks=5 一致，状态在派发后 5 tick 生效。后续着陆保留原速度和 home36→home3，从当时实际位置接入，不把窗外着陆算作本窗结果。\nFalsification: 本臂是当前脚本和当前引擎的 factual 观测。若上述时序或状态未由实跑复现，组内对照失效。\n'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-1_v1"
SEED = "seed00"
ANCHOR_EVENT_ID = "c2_loss"
ARM_ID = "arm_factual"
MODE = "factual_observation"
ROLE = "factual"
UAV = "uav_digital_l6_1_v1"

# Factual control: no trigger cascade is cancelled.
CANCEL_EVENT_IDS = ()

# Factual control: no authored action is omitted.
OMIT_ACTION_IDS = ()

# Factual control: no action is injected anywhere in the window.
INJECT_ACTIONS = ()

# Factual control: no trigger is retimed.
RETIME = None

# The mechanism dict is exactly the workorder mechanism field: the factual
# baseline declares every mechanism lever empty.
MECHANISM = {'kind': 'factual_baseline_no_mechanism',
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
    window_end_tick=340,
    primary_predicate="communication.link_unavailable",
    subject_entity_ids=('tower_l6_1_v1', 'uav_digital_l6_1_v1'),
    mechanism=MECHANISM,
    retime=None,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECT_ACTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis=(
        ('无干预运行中，c2_loss 在 240 派发，链路不可用自 241 生效；rth_engaged 在 275 派发。返航单段朝实际 home pad 飞行，在 283 到达 home 正上方 30 m，随后悬停；该窗口未着陆。c2_recovered 在 325 派发，通信不可用和 rth_active 于 330 清除，C2 事件于 340 完成恢复保持。')
    ),
    falsified_if=(
        '本臂是当前脚本和当前引擎的 factual 观测。若上述时序或状态未由实跑复现，组内对照失效。'
    ),
    window_basis=(
        ('闭区间 [235,340] 覆盖失联响应、实际返航与链路恢复保持，不进入 landing@405。返航段长 5.048194 m，7 m/s、10 Hz 下 ceil(distance/7×10)=8 tick，275→283；at_home_pad 只判水平邻近，不等于 touchdown。恢复运行状态的 delay_ticks=5，330 生效；动作说明与 delay_ticks=5 一致，状态在派发后 5 tick 生效。后续着陆保留原速度和 home36→home3，从当时实际位置接入，不把窗外着陆算作本窗结果。')
    ),
    window_needed_nodes=({'event_id': 'c2_loss', 'tick': 240, 'role': 'authored_anchor'},
                         {'event_id': 'rth_engaged', 'tick': 275, 'role': 'authored_direct_response'},
                         {'event_id': 'c2_recovered', 'tick': 325, 'role': 'authored_recovery'},
                         {'predicate_id': 'communication.link_unavailable',
                          'tick': 340,
                          'role': 'recovery_hold_terminal'}),
    window_censored=False,
    group_suffix="",
)

if __name__ == "__main__":
    run_arm(SPEC)
