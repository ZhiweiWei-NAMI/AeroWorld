"""ARM arm_factual —— L2-4_v2 seed01 事件链片段 second_uav_diverted → lifecycle_landing_uav_priority_l2_4_v2

机制、窗口、动作在本文件里逐条写死。
窗口依据：闭区间 [330,560]；335 为动作前共享检查点，可早于实际干预。观察 second_uav_diverted → lifecycle_landing_uav_priority_l2_4_v2 的局部过程和分支直接后果；终点保留现有实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：对照臂：不加任何干预
证伪：不适用：对照臂
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l2_window import run_arm

SPEC = {
    "profile": "L2-4_v2",
    "seed": "seed01",
    "episode_id": "L2-4_v2__seed01",
    "anchor_event": "second_uav_diverted",
    "neighbor_event": "lifecycle_landing_uav_priority_l2_4_v2",
    "chain_nodes": [
        [
            "second_uav_diverted",
            335
        ],
        [
            "lifecycle_landing_uav_priority_l2_4_v2",
            415
        ]
    ],
    "window": {
        "start_tick": 330,
        "fork_tick": 335,
        "end_tick": 560
    },
    "window_basis": "闭区间 [330,560]；335 为动作前共享检查点，可早于实际干预。观察 second_uav_diverted → lifecycle_landing_uav_priority_l2_4_v2 的局部过程和分支直接后果；终点保留现有实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "uav_priority_l2_4_v2",
        "uav_divert_l2_4_v2",
        "pad_shared_l2_4_v2"
    ],
    "predicate_ids": [
        "facility.landing_pad_contention",
        "safety.aircraft_pair_below_separation"
    ],
    "target_families": [
        "pad_contention",
        "uav_separation_conflict"
    ],
    "factual_root": "aw_data",
    "sources": {
        "scene": "Dataset/scenarios/L2_infrastructure/failure/L2-4_v2/scene_setup.json",
        "script": "Dataset/scenarios/L2_infrastructure/failure/L2-4_v2/event_script.json",
        "trajectories": "aw_data/render_ready_episodes_capture_filtered/L2-4_v2__seed01/trajectories.jsonl",
        "weather": "aw_data/render_ready_episodes_capture_filtered/L2-4_v2__seed01/weather_meta.jsonl",
        "roster": "aw_data/render_ready_episodes_capture_filtered/L2-4_v2__seed01/global_entity_roster.json"
    },
    "script_path": "scripts/arms/L2-4_v2/seed01/grp_second_uav_diverted__lifecycle_landing_uav_priority_l2_4_v2/{arm_id}.py",
    "output_path": "arms/L2-4_v2/seed01/grp_second_uav_diverted__lifecycle_landing_uav_priority_l2_4_v2/{arm_id}",
    "group_key": "L2-4_v2/seed01/grp_second_uav_diverted__lifecycle_landing_uav_priority_l2_4_v2",
    "arm_id": "arm_factual",
    "mode": "factual_observation",
    "actions": [],
    "omit_action_ids": [],
    "expected": "observe",
    "rationale": "对照臂：不加任何干预",
    "falsified_if": "不适用：对照臂"
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
