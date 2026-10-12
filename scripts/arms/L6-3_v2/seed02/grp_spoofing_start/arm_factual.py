"""Dedicated ARM entry for L6-3_v2 / seed02 / spoofing_start / arm_factual.

无干预事实基线。
窗口 [235, 390]，fork=240，采样相位为
after_state_sample_before_actions。

业务假设：无干预运行按 authored 脚本保留spoofing_start@240、偏航@270、安全纠偏@335 与恢复@375/380。纠偏航段从偏置端点回到计划航线，理论长度 19.416488 m；按 4.0 m/s、10 Hz 和逐段向上取整，需要 49 ticks，tick 384 到达、385 首次静止。恢复状态补丁在 380 生效，390 提供恢复后的正式保持采样。

窗口依据：窗口 [235, 390] 使用同组 fork=240 和一个正式采样点的共享前缀。end=390 覆盖 spoof_recovery@375、恢复补丁@380、纠偏终点@384、首次静止@385 与恢复后正式保持采样@390；普通 lifecycle_landing@455 位于窗口外。

证伪条件：若 source action 未按 240/270/335/375 派发，运行时补丁未按 241/271/336或340/380 生效，或纠偏未在 384 到达并于 385 静止，则该 factual 运行不符合 authored 脚本。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-3_v2"
SEED = "seed02"
ANCHOR_EVENT_ID = "spoofing_start"
ARM_ID = "arm_factual"
MODE = "factual_observation"
ROLE = "factual"
UAV = "uav_digital_l6_3_v2"
# This dedicated entry declares its own exact branch rewrite.
CANCEL_EVENT_IDS = ()
OMIT_ACTION_IDS = ()
INJECTIONS = ()
MECHANISM = {'kind': 'no_intervention_factual_baseline',
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
    window_end_tick=390,
    primary_predicate='positioning.gnss_spoofing_active',
    subject_entity_ids=(UAV,),
    mechanism=MECHANISM,
    retime=None,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis='无干预运行按 authored 脚本保留spoofing_start@240、偏航@270、安全纠偏@335 与恢复@375/380。纠偏航段从偏置端点回到计划航线，理论长度 19.416488 m；按 4.0 m/s、10 Hz 和逐段向上取整，需要 49 ticks，tick 384 到达、385 首次静止。恢复状态补丁在 380 生效，390 提供恢复后的正式保持采样。',
    falsified_if='若 source action 未按 240/270/335/375 派发，运行时补丁未按 241/271/336或340/380 生效，或纠偏未在 384 到达并于 385 静止，则该 factual 运行不符合 authored 脚本。',
    window_basis='窗口 [235, 390] 使用同组 fork=240 和一个正式采样点的共享前缀。end=390 覆盖 spoof_recovery@375、恢复补丁@380、纠偏终点@384、首次静止@385 与恢复后正式保持采样@390；普通 lifecycle_landing@455 位于窗口外。',
    window_needed_nodes=tuple([{'event_id': 'spoofing_start', 'tick': 240, 'role': 'factual_anchor'},
 {'event_id': 'spoofed_route_offset', 'tick': 270, 'role': 'factual_offset'},
 {'event_id': 'safety_route_lock', 'tick': 335, 'role': 'factual_correction'},
 {'event_id': 'spoof_recovery', 'tick': 375, 'role': 'factual_signal_and_route_recovery'},
 {'event_id': 'set_runtime_state_l6_3_v2_spoof_recovery_00',
  'tick': 380,
  'entity_id': 'uav_digital_l6_3_v2',
  'role': 'factual_recovery_state_patch'}]),
    window_censored=False,
    group_suffix='',
)

if __name__ == "__main__":
    run_arm(SPEC)
