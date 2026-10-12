"""ARM arm_early_signal_repair —— L2-5_v1 seed01 事件链片段 police_arrival → manual_flow_restore

机制：信号灯故障与车辆滞留照常发生；在警车出发时刻 t=340 由基础设施提前完成维修
（fault=false + 信号恢复 nominal），省略警察出动与人工指挥整组动作，t=342 注入两车
放行 —— 设备修复在先，人工指挥不再需要，交通恢复流动。源脚本的人工放行与设备修复已拆分。
窗口依据：闭区间 [335,380]；340 为动作前共享检查点。观察 police_arrival → manual_flow_restore 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：occurs；窗口前已发生的设备故障经维修恢复，人工指挥链不执行。
证伪：窗口内 ground.traffic_signal_malfunction_active 仍为 true，或车辆未恢复移动。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l2_window import run_arm

SPEC = {
    "profile": "L2-5_v1",
    "seed": "seed01",
    "episode_id": "L2-5_v1__seed01",
    "anchor_event": "police_arrival",
    "neighbor_event": "manual_flow_restore",
    "chain_nodes": [
        ["police_arrival", 340],
        ["manual_flow_restore", 370]
    ],
    "window": {
        "start_tick": 335,
        "fork_tick": 340,
        "end_tick": 380
    },
    "window_basis": "闭区间 [335,380]；340 为动作前共享检查点。观察 police_arrival → manual_flow_restore 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "civilian_car_l2_5_a",
        "civilian_car_l2_5_b",
        "traffic_signal_l2_5_v1"
    ],
    "predicate_ids": [
        "ground.traffic_signal_malfunction_active"
    ],
    "target_families": [
        "signal_malfunction"
    ],
    "factual_root": "aw_data",
    "sources": {
        "scene": "Dataset/scenarios/L2_infrastructure/interaction/L2-5_v1/scene_setup.json",
        "script": "Dataset/scenarios/L2_infrastructure/interaction/L2-5_v1/event_script.json",
        "trajectories": "aw_data/render_ready_episodes_capture_filtered/L2-5_v1__seed01/trajectories.jsonl",
        "weather": "aw_data/render_ready_episodes_capture_filtered/L2-5_v1__seed01/weather_meta.jsonl",
        "roster": "aw_data/render_ready_episodes_capture_filtered/L2-5_v1__seed01/global_entity_roster.json"
    },
    "script_path": "scripts/arms/L2-5_v1/seed01/grp_police_arrival__manual_flow_restore/{arm_id}.py",
    "output_path": "arms/L2-5_v1/seed01/grp_police_arrival__manual_flow_restore/{arm_id}",
    "group_key": "L2-5_v1/seed01/grp_police_arrival__manual_flow_restore",
    "arm_id": "arm_early_signal_repair",
    "mode": "intervention",
    "actions": [
        {
            "tick": 340,
            "action": {
                "type": "set_runtime_state",
                "action_id": "l2_arm_early_signal_repair_fault_clear",
                "entity_id": "traffic_signal_l2_5_v1",
                "delay_ticks": 0,
                "state_patch": {
                    "facility_state": {
                        "fault": False
                    }
                }
            }
        },
        {
            "tick": 340,
            "action": {
                "type": "set_visual_state",
                "action_id": "l2_arm_early_signal_repair_nominal",
                "entity_id": "traffic_signal_l2_5_v1",
                "visual_state": {
                    "mode": "nominal"
                }
            }
        },
        {
            "tick": 342,
            "action": {
                "type": "move_entity",
                "action_id": "l2_arm_early_signal_repair_release_a",
                "entity_id": "civilian_car_l2_5_a",
                "waypoints_enu_m": [[6273.005, 7906.527, 0.0], [6271.736, 7910.312, 0.0], [6270.462, 7914.11, 0.0]],
                "velocity_mps": 6.0
            }
        },
        {
            "tick": 342,
            "action": {
                "type": "move_entity",
                "action_id": "l2_arm_early_signal_repair_release_b",
                "entity_id": "civilian_car_l2_5_b",
                "waypoints_enu_m": [[6278.397, 7899.892, 0.0], [6277.119, 7903.689, 0.0], [6275.85, 7907.477, 0.0]],
                "velocity_mps": 5.5
            }
        }
    ],
    "omit_action_ids": [
        "spawn_police_unit_l2_5_v1_at_dispatch",
        "move_police_to_intersection",
        "set_police_manual_control",
        "set_runtime_state_l2_5_v1_police_arrival_00",
        "move_car_a_released",
        "move_car_b_released"
    ],
    "expected": 'occurs',
    "rationale": "设备提前维修：信号在警车出发时刻由基础设施修复恢复 nominal，车辆放行由维修直接触发；人工指挥整组动作不再发生，设备恢复与交通流恢复由同一维修动作联动。",
    "falsified_if": '维修后故障谓词未恢复为 false，或车辆未恢复移动。'
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
