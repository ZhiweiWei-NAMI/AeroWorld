"""ARM arm_manual_control_absent —— L2-5_v1 seed01 事件链片段 police_arrival → manual_flow_restore

机制：信号灯故障后，省略警车出动、人工指挥与人工放行动作；源过程不包含设备维修。
车辆完成此前进队后停留，信号故障持续整个窗口。
窗口依据：闭区间 [335,380]；340 为动作前共享检查点。观察 police_arrival → manual_flow_restore 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：occurs —— 信号故障持续成立且车辆保持滞留。
证伪：窗口内车辆恢复移动，或信号故障谓词转为 false。
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
    "script_path": "scripts/arms/L2-5_v1/seed01/grp_police_arrival__manual_flow_restore_2/{arm_id}.py",
    "output_path": "arms/L2-5_v1/seed01/grp_police_arrival__manual_flow_restore_2/{arm_id}",
    "group_key": "L2-5_v1/seed01/grp_police_arrival__manual_flow_restore_2",
    "arm_id": "arm_manual_control_absent",
    "mode": "intervention",
    "actions": [],
    "omit_action_ids": [
        "spawn_police_unit_l2_5_v1_at_dispatch",
        "move_police_to_intersection",
        "set_police_manual_control",
        "set_runtime_state_l2_5_v1_police_arrival_00",
        "move_car_a_released",
        "move_car_b_released"
    ],
    "expected": "occurs",
    "rationale": "人工指挥缺席：警察不出动、不放行、不修信号，车辆持续滞留，故障持续成立；用于研究响应缺失的直接后果。",
    "falsified_if": "窗口内车辆恢复移动，或信号故障谓词转为 false。"
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
