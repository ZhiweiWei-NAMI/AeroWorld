"""ARM arm_factual —— L2-3_v2 seed02 事件链片段 charger_unavailable → uav_inspects_charger

机制、窗口、动作在本文件里逐条写死。
窗口依据：闭区间 [235,495]；240 为动作前共享检查点，可早于实际干预。观察 charger_unavailable → uav_inspects_charger 的局部过程和分支直接后果；终点保留现有实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：observe —— 对照臂：不加任何干预。
证伪：不适用：对照臂
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l2_window import run_arm

SPEC = {
    "profile": "L2-3_v2",
    "seed": "seed02",
    "episode_id": "L2-3_v2__seed02",
    "anchor_event": "charger_unavailable",
    "neighbor_event": "uav_inspects_charger",
    "chain_nodes": [
        ["charger_unavailable", 240],
        ["uav_inspects_charger", 270],
    ],
    "window": {
        "start_tick": 235,
        "fork_tick": 240,
        "end_tick": 495,
    },
    "window_basis": "闭区间 [235,495]；240 为动作前共享检查点，可早于实际干预。观察 charger_unavailable → uav_inspects_charger 的局部过程和分支直接后果；终点保留现有实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "uav_l2_3_v2",
        "charger_primary_l2_3_v2",
        "charger_backup_l2_3_v2",
    ],
    "predicate_ids": [
        "facility.charging_station_unavailable",
        "facility.charging_queue_over_capacity",
    ],
    "target_families": [
        "charger_unavailable",
        "charging_queue_over_capacity",
    ],
    "factual_root": "aw_data",
    "sources": {
        "scene": "Dataset/scenarios/L2_infrastructure/failure/L2-3_v2/scene_setup.json",
        "script": "Dataset/scenarios/L2_infrastructure/failure/L2-3_v2/event_script.json",
        "trajectories": "aw_data/render_ready_episodes_capture_filtered/L2-3_v2__seed02/trajectories.jsonl",
        "weather": "aw_data/render_ready_episodes_capture_filtered/L2-3_v2__seed02/weather_meta.jsonl",
        "roster": "aw_data/render_ready_episodes_capture_filtered/L2-3_v2__seed02/global_entity_roster.json",
    },
    "script_path": "scripts/arms/L2-3_v2/seed02/grp_charger_unavailable__uav_inspects_charger/{arm_id}.py",
    "output_path": "arms/L2-3_v2/seed02/grp_charger_unavailable__uav_inspects_charger/{arm_id}",
    "group_key": "L2-3_v2/seed02/grp_charger_unavailable__uav_inspects_charger",
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
