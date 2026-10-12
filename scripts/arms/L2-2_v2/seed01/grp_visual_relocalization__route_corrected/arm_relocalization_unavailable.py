"""ARM arm_relocalization_unavailable —— L2-2_v2 seed01 事件链片段 visual_relocalization → route_corrected

机制：视觉重定位动作照常尝试（set_uav_visual_reloc 照发），但省略两条恢复补丁
（重定位解未登记、纠偏状态未确认），并省略 move_uav_corrected_route；在原纠偏时刻
注入 set_visual_state hold —— 重定位不可用，不信任当前位置，原地保持，结构接近未恢复。
轨迹（361 起静止、活动标签 hold）、导航状态（multipath_warning 持续 true）、
谓词（positioning.gnss_quality_degraded 持续 true）三者一致。
窗口依据：闭区间 [325,370]；330 为动作前共享检查点。观察 visual_relocalization → route_corrected 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：occurs —— GNSS 降级在窗口内持续存在（起点在窗外，继承进入）。
证伪：UAV 仍执行 corrected route，或 hold 未生效，或窗口内 positioning.gnss_quality_degraded 恢复为 false。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l2_window import run_arm

SPEC = {
    "profile": "L2-2_v2",
    "seed": "seed01",
    "episode_id": "L2-2_v2__seed01",
    "anchor_event": "visual_relocalization",
    "neighbor_event": "route_corrected",
    "chain_nodes": [
        ["visual_relocalization", 330],
        ["route_corrected", 360],
    ],
    "window": {
        "start_tick": 325,
        "fork_tick": 330,
        "end_tick": 370,
    },
    "window_basis": "闭区间 [325,370]；330 为动作前共享检查点。观察 visual_relocalization → route_corrected 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
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
        "trajectories": "aw_data/render_ready_episodes_capture_filtered/L2-2_v2__seed01/trajectories.jsonl",
        "weather": "aw_data/render_ready_episodes_capture_filtered/L2-2_v2__seed01/weather_meta.jsonl",
        "roster": "aw_data/render_ready_episodes_capture_filtered/L2-2_v2__seed01/global_entity_roster.json",
    },
    "script_path": "scripts/arms/L2-2_v2/seed01/grp_visual_relocalization__route_corrected/{arm_id}.py",
    "output_path": "arms/L2-2_v2/seed01/grp_visual_relocalization__route_corrected/{arm_id}",
    "group_key": "L2-2_v2/seed01/grp_visual_relocalization__route_corrected",
    "arm_id": "arm_relocalization_unavailable",
    "mode": "intervention",
    "actions": [
        {
            "tick": 360,
            "action": {
                "type": "set_visual_state",
                "action_id": "l2_arm_relocalization_unavailable_hold",
                "entity_id": "uav_l2_2_v2",
                "visual_state": {
                    "mode": "hold"
                }
            }
        }
    ],
    "omit_action_ids": [
        "move_uav_corrected_route",
        "set_runtime_state_l2_2_v2_visual_relocalization_00",
        "set_runtime_state_l2_2_v2_route_corrected_00"
    ],
    "expected": "occurs",
    "rationale": "重定位不可用：省略纠偏航路与两条恢复补丁，UAV 以 hold 在漂移终点原地保持，GNSS 降级（多径）与结构接近均未恢复。",
    "falsified_if": "UAV 仍执行 corrected route，或 hold 未生效，或窗口内 positioning.gnss_quality_degraded 恢复为 false。",
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
