"""ARM arm_factual —— L2-2_v2 seed02 事件链片段 gnss_anomaly → route_drift

机制、窗口、动作在本文件里逐条写死。
窗口依据：窗口取事件链的相邻两节点 gnss_anomaly@250 → route_drift@280（含分叉 tick 250 的那一段），闭区间：start=230（第一节点前 20 tick，4 个正式样本，对齐 5 网格）；fork=250（本片段干预生效 tick，位于两节点之间）；end=290（覆盖决定性谓词差异 [255] 后的 2 个正式样本，并至少到第二节点 +10 tick）。同组各臂共用该窗口。
预期：对照臂：不加任何干预
证伪：不适用：对照臂
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l2_window import run_arm

SPEC = {
    "profile": "L2-2_v2",
    "seed": "seed02",
    "episode_id": "L2-2_v2__seed02",
    "anchor_event": "gnss_anomaly",
    "neighbor_event": "route_drift",
    "chain_nodes": [
        [
            "gnss_anomaly",
            250,
        ],
        [
            "route_drift",
            280,
        ],
    ],
    "window": {
        "start_tick": 245,
        "fork_tick": 280,
        "end_tick": 365,
    },
    "window_basis": "闭区间 [245,365]；280 为漂移派发前共享检查点，可早于实际干预。观察 gnss_anomaly → route_drift 的局部过程和分支直接后果；终点保留现有实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "uav_l2_2_v2",
    ],
    "predicate_ids": [
        "positioning.gnss_quality_degraded",
        "safety.aircraft_near_building_structure",
    ],
    "target_families": [
        "gnss_signal_degradation",
        "structure_near_miss",
    ],
    "factual_root": "aw_data",
    "sources": {
        "scene": "Dataset/scenarios/L2_infrastructure/failure/L2-2_v2/scene_setup.json",
        "script": "Dataset/scenarios/L2_infrastructure/failure/L2-2_v2/event_script.json",
        "trajectories": "aw_data/render_ready_episodes_capture_filtered/L2-2_v2__seed02/trajectories.jsonl",
        "weather": "aw_data/render_ready_episodes_capture_filtered/L2-2_v2__seed02/weather_meta.jsonl",
        "roster": "aw_data/render_ready_episodes_capture_filtered/L2-2_v2__seed02/global_entity_roster.json",
    },
    "script_path": "scripts/arms/L2-2_v2/seed02/grp_gnss_anomaly__route_drift/{arm_id}.py",
    "output_path": "arms/L2-2_v2/seed02/grp_gnss_anomaly__route_drift/{arm_id}",
    "group_key": "L2-2_v2/seed02/grp_gnss_anomaly__route_drift",
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
