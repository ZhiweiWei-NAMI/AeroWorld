"""ARM arm_arbitration_delayed —— L2-4_v1 seed01 事件链片段 priority_arbitration → second_uav_diverted

机制：仲裁照常发生，但其决议被推迟 —— t=580 省略第二架 UAV 的改航飞行与仲裁收尾补丁
（arbitration_active 保持 true，requester 保持 2）：争用持续成立，第二架 UAV 持续 hold，
直至窗口结束仍未解除（右删失）。
窗口依据：闭区间 [545,600]；550 为动作前共享检查点。观察 priority_arbitration → second_uav_diverted 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：occurs —— 争用事件持续成立且窗口内不解除。
证伪：第二架 UAV 实际执行改航，或争用在窗口内解除。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l2_window import run_arm

SPEC = {
    "profile": "L2-4_v1",
    "seed": "seed01",
    "episode_id": "L2-4_v1__seed01",
    "anchor_event": "priority_arbitration",
    "neighbor_event": "second_uav_diverted",
    "chain_nodes": [
        ["priority_arbitration", 550],
        ["second_uav_diverted", 580]
    ],
    "window": {
        "start_tick": 545,
        "fork_tick": 550,
        "end_tick": 600
    },
    "window_basis": "闭区间 [545,600]；550 为动作前共享检查点。观察 priority_arbitration → second_uav_diverted 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "pad_shared_l2_4_v1",
        "uav_divert_l2_4_v1",
        "uav_priority_l2_4_v1"
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
        "scene": "Dataset/scenarios/L2_infrastructure/failure/L2-4_v1/scene_setup.json",
        "script": "Dataset/scenarios/L2_infrastructure/failure/L2-4_v1/event_script.json",
        "trajectories": "aw_data/render_ready_episodes_capture_filtered/L2-4_v1__seed01/trajectories.jsonl",
        "weather": "aw_data/render_ready_episodes_capture_filtered/L2-4_v1__seed01/weather_meta.jsonl",
        "roster": "aw_data/render_ready_episodes_capture_filtered/L2-4_v1__seed01/global_entity_roster.json"
    },
    "script_path": "scripts/arms/L2-4_v1/seed01/grp_priority_arbitration__second_uav_diverted/{arm_id}.py",
    "output_path": "arms/L2-4_v1/seed01/grp_priority_arbitration__second_uav_diverted/{arm_id}",
    "group_key": "L2-4_v1/seed01/grp_priority_arbitration__second_uav_diverted",
    "arm_id": "arm_arbitration_delayed",
    "mode": "intervention",
    "actions": [],
    "omit_action_ids": [
        "move_second_uav_diversion",
        "set_runtime_state_l2_4_v1_second_uav_diverted_00",
        "set_runtime_state_l2_4_v1_second_uav_diverted_01"
    ],
    "expected": "occurs",
    "rationale": "仲裁推迟：决议补丁与改航飞行一并省略，第二架 UAV 持续 hold，争用持续成立（右删失），用于研究恢复机制的延迟后果。",
    "falsified_if": "第二架 UAV 实际执行改航，或争用在窗口内解除。"
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
