"""Dedicated ARM entry for L6-5_v1 / seed00 / grp_gcs_intrusion / arm_intrusion_never_happens.

窗口为 [235,395]，fork=240，采用 after_state_sample_before_actions。
机制：root_event_cancelled_with_dependent_events_unscheduled。

实验假设：仅取消 gcs_intrusion@240；其依赖事件因触发条件不满足而不派发。窗口 [235,395] 内 gcs_anchor_l6_5_v1 保持未受侵入，uav_digital_l6_5_v1 不进入异常指令、lockout 或恢复过程。普通 landing 在窗外，本实验不评价窗外任务终止。

时间与运动模型：仿真为 10 Hz 分段线性运动，事件依赖在 5 tick 网格派发。abnormal_uav_behavior@270 的两段位移时长为 ceil(2/9×10)=3 tick 和 ceil(23.3238076/9×10)=26 tick，故异常航段在 299 完成；command_lockout 的依赖候选为 270+31=301，向上对齐到 305；25.4950976 m 的 factual 纠偏按 4 m/s 需 ceil(25.4950976/4×10)=64 tick，在 369 完成；secure_recovery 的候选为 305+71=376，向上对齐到 380。GCS/UAV 恢复补丁分别在 381/385 生效；正式谓词首个 false 样本为 385，三样本恢复保持为 385/390/395，所以共同闭窗 tick 为 395。

证伪条件：若 gcs_intrusion 或任一依赖事件仍派发，正式主谓词出现 true，GCS 出现 intrusion_detected/locked_out，或 UAV 出现异常位移、unauthorized_command、command_lockout，则分支被证伪。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.l6_v2.arm_runtime import ArmSpec, run_arm

PROFILE = "L6-5_v1"
SEED = "seed00"
ANCHOR_EVENT_ID = "gcs_intrusion"
ARM_ID = "arm_intrusion_never_happens"
MODE = "intervention"
ROLE = "narrative_branch"
UAV = "uav_digital_l6_5_v1"
GCS = "gcs_anchor_l6_5_v1"

# 四个机制位来自本臂配置；动作 ID 均已逐项核对当前 source script。
CANCEL_EVENT_IDS = ('gcs_intrusion',)

OMIT_ACTION_IDS = ()

INJECTIONS = ()

RETIME = None

MECHANISM = {'kind': 'root_event_cancelled_with_dependent_events_unscheduled',
 'cancel_event_ids': ['gcs_intrusion'],
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
    window_end_tick=395,
    primary_predicate='security.ground_control_station_compromised',
    subject_entity_ids=(UAV,),
    mechanism=MECHANISM,
    retime=RETIME,
    omit_action_ids=OMIT_ACTION_IDS,
    inject_actions=INJECTIONS,
    cancel_event_ids=CANCEL_EVENT_IDS,
    hypothesis='仅取消 gcs_intrusion@240；其依赖事件因触发条件不满足而不派发。窗口 [235,395] 内 gcs_anchor_l6_5_v1 保持未受侵入，uav_digital_l6_5_v1 不进入异常指令、lockout 或恢复过程。普通 landing 在窗外，本实验不评价窗外任务终止。',
    falsified_if='若 gcs_intrusion 或任一依赖事件仍派发，正式主谓词出现 true，GCS 出现 intrusion_detected/locked_out，或 UAV 出现异常位移、unauthorized_command、command_lockout，则分支被证伪。',
    window_basis='闭区间 [235,395]，fork=240，phase=after_state_sample_before_actions。窗口从根事件前一正式采样点开始，覆盖根事件、全部安全后继和正式恢复保持。仿真为 10 Hz 分段线性运动，事件依赖在 5 tick 网格派发。abnormal_uav_behavior@270 的两段位移时长为 ceil(2/9×10)=3 tick 和 ceil(23.3238076/9×10)=26 tick，故异常航段在 299 完成；command_lockout 的依赖候选为 270+31=301，向上对齐到 305；25.4950976 m 的 factual 纠偏按 4 m/s 需 ceil(25.4950976/4×10)=64 tick，在 369 完成；secure_recovery 的候选为 305+71=376，向上对齐到 380。GCS/UAV 恢复补丁分别在 381/385 生效；正式谓词首个 false 样本为 385，三样本恢复保持为 385/390/395，所以共同闭窗 tick 为 395。普通 landing@460 不属于 needed nodes，也不参与成功或证伪判断。',
    window_needed_nodes=({'event_id': 'gcs_intrusion',
  'tick': 240,
  'entity_id': 'gcs_anchor_l6_5_v1',
  'role': 'cancelled_anchor_root'},
 {'event_id': 'abnormal_uav_behavior',
  'tick': 270,
  'entity_id': 'uav_digital_l6_5_v1',
  'role': 'dependent_event_expected_absent'},
 {'event_id': 'command_lockout',
  'tick': 305,
  'entity_id': 'gcs_anchor_l6_5_v1',
  'role': 'dependent_event_expected_absent'},
 {'event_id': 'secure_recovery',
  'tick': 380,
  'entity_id': 'gcs_anchor_l6_5_v1',
  'role': 'dependent_event_expected_absent'}),
    window_censored=False,
    group_suffix='',
)

if __name__ == "__main__":
    run_arm(SPEC)
