"""ARM arm_staggered_pad_admission —— L2-4_v2 seed02 事件链片段 dual_pad_approach → priority_arbitration

机制：错峰进场 —— t=230 省略第二架 UAV 的进场飞行，并注入共享停机坪请求缩减
（request_count=1，仅优先 UAV），t=250 注入第二架 UAV hold 在外等待；优先 UAV 正常
进场与仲裁；优先 UAV 释放后第二架 UAV 于 t=480 开始进场（错峰后段）。
窗口依据：闭区间 [225,560]；230 为动作前共享检查点。观察 dual_pad_approach → priority_arbitration 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：absent_in_window —— 停机坪争用不发生。
证伪：窗口内出现 landing_pad_contention，或两架 UAV 再次同时接近共享停机坪。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l2_window import run_arm

SPEC = {
    "profile": "L2-4_v2",
    "seed": "seed02",
    "episode_id": "L2-4_v2__seed02",
    "anchor_event": "dual_pad_approach",
    "neighbor_event": "priority_arbitration",
    "chain_nodes": [
        ["dual_pad_approach", 230],
        ["priority_arbitration", 305]
    ],
    "window": {
        "start_tick": 225,
        "fork_tick": 230,
        "end_tick": 560
    },
    "window_basis": "闭区间 [225,560]；230 为动作前共享检查点。观察 dual_pad_approach → priority_arbitration 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "pad_shared_l2_4_v2",
        "uav_divert_l2_4_v2",
        "uav_priority_l2_4_v2"
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
        "trajectories": "aw_data/render_ready_episodes_capture_filtered/L2-4_v2__seed02/trajectories.jsonl",
        "weather": "aw_data/render_ready_episodes_capture_filtered/L2-4_v2__seed02/weather_meta.jsonl",
        "roster": "aw_data/render_ready_episodes_capture_filtered/L2-4_v2__seed02/global_entity_roster.json"
    },
    "script_path": "scripts/arms/L2-4_v2/seed02/grp_dual_pad_approach__priority_arbitration/{arm_id}.py",
    "output_path": "arms/L2-4_v2/seed02/grp_dual_pad_approach__priority_arbitration/{arm_id}",
    "group_key": "L2-4_v2/seed02/grp_dual_pad_approach__priority_arbitration",
    "arm_id": "arm_staggered_pad_admission",
    "mode": "intervention",
    "actions": [
        {
            "tick": 230,
            "action": {
                "type": "set_runtime_state",
                "action_id": "l2_arm_staggered_single_request",
                "entity_id": "pad_shared_l2_4_v2",
                "delay_ticks": 0,
                "state_patch": {
                    "facility_state": {
                        "request_count": 1,
                        "requester_ids": ["uav_priority_l2_4_v2"],
                        "arbitration_active": False
                    }
                }
            }
        },
        {
            "tick": 250,
            "action": {
                "type": "set_visual_state",
                "action_id": "l2_arm_staggered_second_hold",
                "entity_id": "uav_divert_l2_4_v2",
                "visual_state": {
                    "mode": "hold"
                }
            }
        },
        {
            "tick": 450,
            "action": {
                "type": "set_runtime_state",
                "action_id": "l2_arm_staggered_resource_handover",
                "entity_id": "pad_shared_l2_4_v2",
                "delay_ticks": 0,
                "state_patch": {
                    "facility_state": {
                        "request_count": 1,
                        "requester_ids": ["uav_divert_l2_4_v2"],
                        "arbitration_active": False,
                        "priority_granted": False,
                        "priority_granted_to": "none"
                    }
                }
            }
        },
        {
            "tick": 450,
            "action": {
                "type": "set_runtime_state",
                "action_id": "l2_arm_staggered_second_pad_assigned",
                "entity_id": "uav_divert_l2_4_v2",
                "delay_ticks": 0,
                "state_patch": {
                    "navigation_state": {
                        "landing_pad_assigned": True
                    }
                }
            }
        },
        {
            "tick": 455,
            "action": {
                "type": "move_entity",
                "action_id": "l2_arm_staggered_second_approach",
                "entity_id": "uav_divert_l2_4_v2",
                "waypoints_enu_m": [[7711.828, 7476.768, 30.0], [7702.918, 7482.097, 10.0], [7702.918, 7502.097, 0.0]],
                "velocity_mps": 5.0,
                "activity_type": "return_to_pad",
                "post_activity_type": "landed",
                "terminal_feasibility": {
                    "landing_reference_enu_m": [7702.918, 7502.097, 0.0]
                }
            }
        }
    ],
    "omit_action_ids": [
        "move_second_uav_to_pad_hold",
        "move_second_uav_diversion",
        "move_uav_divert_l2_4_v2_landing_return",
        "set_runtime_state_l2_4_v2_second_uav_diverted_00"
    ],
    "expected": "absent_in_window",
    "rationale": "错峰进场与资源移交：第二架 UAV 请求推迟并外围 hold；优先 UAV 起飞返航释放后，停机坪请求权移交第二架，其再进场触地；争用与二次接近全程不发生。",
    "falsified_if": "窗口内出现 landing_pad_contention，或两架 UAV 再次同时接近共享停机坪。"
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
