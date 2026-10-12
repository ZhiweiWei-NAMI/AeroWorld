"""ARM arm_factual —— L2-5_v1 seed00 事件链片段 police_arrival → manual_flow_restore

机制、窗口、动作在本文件里逐条写死。
窗口依据：闭区间 [335,380]；340 为动作前共享检查点，可早于实际干预。观察 police_arrival → manual_flow_restore 的局部过程和分支直接后果；终点保留现有实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：对照臂：不加任何干预
证伪：不适用：对照臂
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l2_window import run_arm

SPEC = {
    "profile": "L2-5_v1",
    "seed": "seed00",
    "episode_id": "L2-5_v1__seed00",
    "anchor_event": "police_arrival",
    "neighbor_event": "manual_flow_restore",
    "chain_nodes": [
        ["police_arrival", 340],
        ["manual_flow_restore", 370],
    ],
    "window": {
        "start_tick": 335,
        "fork_tick": 340,
        "end_tick": 380,
    },
    "window_basis": "闭区间 [335,380]；340 为动作前共享检查点，可早于实际干预。观察 police_arrival → manual_flow_restore 的局部过程和分支直接后果；终点保留现有实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "traffic_signal_l2_5_v1",
        "police_unit_l2_5_v1",
        "civilian_car_l2_5_a",
        "civilian_car_l2_5_b",
    ],
    "predicate_ids": [
        "ground.traffic_signal_malfunction_active",
    ],
    "target_families": [
        "signal_malfunction",
    ],
    "factual_root": "aw_data",
    "sources": {
        "scene": "Dataset/scenarios/L2_infrastructure/interaction/L2-5_v1/scene_setup.json",
        "script": "Dataset/scenarios/L2_infrastructure/interaction/L2-5_v1/event_script.json",
        "trajectories": "aw_data/render_ready_episodes_capture_filtered/L2-5_v1__seed00/trajectories.jsonl",
        "weather": "aw_data/render_ready_episodes_capture_filtered/L2-5_v1__seed00/weather_meta.jsonl",
        "roster": "aw_data/render_ready_episodes_capture_filtered/L2-5_v1__seed00/global_entity_roster.json",
    },
    "script_path": "scripts/arms/L2-5_v1/seed00/grp_police_arrival__manual_flow_restore/{arm_id}.py",
    "output_path": "arms/L2-5_v1/seed00/grp_police_arrival__manual_flow_restore/{arm_id}",
    "group_key": "L2-5_v1/seed00/grp_police_arrival__manual_flow_restore",
    "arm_id": "arm_factual",
    "mode": "factual_observation",
    "actions": [],
    "omit_action_ids": [],
    "expected": "observe",
    "rationale": "对照臂：不加任何干预",
    "falsified_if": "不适用：对照臂",
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
