"""ARM arm_early_backup_link_recovery —— L2-1_v1 seed02 事件链片段 station_degraded → uav_link_response

机制：基站故障照常发生（link_unavailable 于 t=261 起 True，UAV 进入 safe_hold）；t=270 注入备用链路提前恢复并解除 safe_hold，t=275 注入 nominal 续航航路（省略原漂移 loiter 与链路响应补丁），UAV 提前恢复任务飞行。
窗口依据：闭区间 [255,310]；260 为动作前共享检查点。观察基站故障 → 链路中断 → UAV 响应的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：occurs
证伪：窗口内 link_unavailable 未恢复为 False，或 UAV 未沿注入航路恢复飞行。
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
        "end_tick": 310,
    },
    "window_basis": "闭区间 [255,310]；260 为动作前共享检查点。观察基站故障 → 链路中断 → UAV 响应的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "tower_l2_1_v1",
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
    "arm_id": "arm_early_backup_link_recovery",
    "mode": "intervention",
    "actions": [

        {
            "tick": 270,
            "action": {
                "type": "set_runtime_state",
                "action_id": "l2_arm_early_backup_link_recovery",
                "entity_id": "uav_l2_1_v1",
                "delay_ticks": 0,
                "state_patch": {
                    "communication_state": {
                        "station_unavailable": False,
                        "backup_link_active": True,
                        "alternate_link_active": True,
                        "communication_unavailable": False
                    },
                    "control_state": {
                        "safe_hold_active": False
                    }
                }
            }
        },
        {
            "tick": 270,
            "action": {
                "type": "set_runtime_state",
                "action_id": "l2_arm_early_backup_station_recovered",
                "entity_id": "tower_l2_1_v1",
                "delay_ticks": 0,
                "state_patch": {
                    "communication_state": {
                        "station_unavailable": False
                    }
                }
            }
        },
        {
            "tick": 275,
            "action": {
                "type": "move_entity",
                "action_id": "l2_arm_early_backup_resume_route",
                "entity_id": "uav_l2_1_v1",
                "waypoints_enu_m": [
                    [7257.213, 6374.569, 30.0],
                    [7246.213, 6376.569, 30.0]
                ],
                "velocity_mps": 7.0
            }
        }
    ],
    "omit_action_ids": [
        "move_uav_backup_loiter",
        "set_runtime_state_l2_1_v1_uav_link_response_00",
        "set_runtime_state_l2_1_v1_uav_link_response_01"
    ],
    "expected": "occurs",
    "rationale": "备用链路提前恢复：链路中断仅短暂成立（约 261..270），UAV 短暂 safe_hold 后经备用链路继续任务航路，观察链路恢复与任务恢复的联动。",
    "falsified_if": "窗口内 link_unavailable 未恢复为 False，或 UAV 未沿注入航路恢复飞行。",
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
