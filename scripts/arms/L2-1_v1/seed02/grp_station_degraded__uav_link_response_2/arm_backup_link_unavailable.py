"""ARM arm_backup_link_unavailable —— L2-1_v1 seed02 事件链片段 station_degraded → uav_link_response

机制：基站故障后链路一直不可用（省略链路响应补丁），UAV 仅执行故障 loiter；省略后续 patrol resume 与基站恢复动作，t=350 注入 hold 使 UAV 原地保持，任务中断。
窗口依据：闭区间 [255,360]；260 为动作前共享检查点。观察基站故障 → 链路中断 → UAV 响应的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：occurs
证伪：窗口内 link_unavailable 恢复为 false，或 UAV 恢复任务航路飞行。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l2_window import run_arm

SPEC = {
    "profile": "L2-1_v1",
    "seed": "seed02",
    "episode_id": "L2-1_v1__seed02",
    "anchor_event": "station_degraded",
    "neighbor_event": "uav_link_response",
    "chain_nodes": [
        ["station_degraded", 260],
        ["uav_link_response", 290],
    ],
    "window": {
        "start_tick": 255,
        "fork_tick": 260,
        "end_tick": 360,
    },
    "window_basis": "闭区间 [255,360]；260 为动作前共享检查点。观察基站故障 → 链路中断 → UAV 响应的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "uav_l2_1_v1",
    ],
    "predicate_ids": [
        "communication.link_unavailable",
        "communication.link_degraded",
    ],
    "target_families": [
        "c2_degradation_loss",
    ],
    "factual_root": "aw_data",
    "sources": {
        "scene": "Dataset/scenarios/L2_infrastructure/failure/L2-1_v1/scene_setup.json",
        "script": "Dataset/scenarios/L2_infrastructure/failure/L2-1_v1/event_script.json",
        "trajectories": "aw_data/render_ready_episodes_capture_filtered/L2-1_v1__seed02/trajectories.jsonl",
        "weather": "aw_data/render_ready_episodes_capture_filtered/L2-1_v1__seed02/weather_meta.jsonl",
        "roster": "aw_data/render_ready_episodes_capture_filtered/L2-1_v1__seed02/global_entity_roster.json",
    },
    "script_path": "scripts/arms/L2-1_v1/seed02/{grp}/{arm_id}.py",
    "output_path": "arms/L2-1_v1/seed02/{grp}/{arm_id}",
    "group_key": "L2-1_v1/seed02/{grp}",
    "arm_id": "arm_backup_link_unavailable",
    "mode": "intervention",
    "actions": [
        {
            "tick": 350,
            "action": {
                "type": "set_visual_state",
                "action_id": "l2_arm_backup_link_unavailable_hold",
                "entity_id": "uav_l2_1_v1",
                "visual_state": {
                    "mode": "hold"
                }
            }
        }
    ],
    "omit_action_ids": [
        "set_runtime_state_l2_1_v1_uav_link_response_00",
        "set_runtime_state_l2_1_v1_uav_link_response_01",
        "move_uav_resume_patrol",
        "set_tower_backup"
    ],
    "expected": "occurs",
    "rationale": "备用链路不可用：链路中断持续整个窗口，UAV loiter 后持续 hold，任务无法恢复，观察链路持续中断下响应失败分支。",
    "falsified_if": "窗口内 link_unavailable 恢复为 false，或 UAV 恢复任务航路飞行。",
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
