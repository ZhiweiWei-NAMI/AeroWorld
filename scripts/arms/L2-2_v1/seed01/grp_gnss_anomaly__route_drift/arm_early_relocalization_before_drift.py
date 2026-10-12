"""ARM arm_early_relocalization_before_drift —— L2-2_v1 seed01 事件链片段 gnss_anomaly → route_drift

机制：多径异常照常发生（multipath_warning=true，GNSS 降级），但在漂移机动派发时刻 t=280 省略漂移航路，
并注入远离立面的航路；重定位于280启动，复用factual的30 tick处理过程于310派发完成。多径环境保留，定位质量按实际状态重算。
窗口依据：闭区间 [245,365]；280 为漂移派发前共享检查点。观察 gnss_anomaly → route_drift 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：occurs；已有 GNSS 降级在重定位完成后恢复，实际航迹不接近建筑。
证伪：窗口内出现 structure_near_miss，或 UAV 实际执行了漂移航路。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l2_window import run_arm

SPEC = {
    "profile": "L2-2_v1",
    "seed": "seed01",
    "episode_id": "L2-2_v1__seed01",
    "anchor_event": "gnss_anomaly",
    "neighbor_event": "route_drift",
    "chain_nodes": [
        ["gnss_anomaly", 250],
        ["route_drift", 280],
    ],
    "window": {
        "start_tick": 245,
        "fork_tick": 280,
        "end_tick": 365,
    },
    "window_basis": "闭区间 [245,365]；280 为漂移派发前共享检查点。观察 gnss_anomaly → route_drift 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "uav_l2_2_v1",
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
        "scene": "Dataset/scenarios/L2_infrastructure/failure/L2-2_v1/scene_setup.json",
        "script": "Dataset/scenarios/L2_infrastructure/failure/L2-2_v1/event_script.json",
        "trajectories": "aw_data/render_ready_episodes_capture_filtered/L2-2_v1__seed01/trajectories.jsonl",
        "weather": "aw_data/render_ready_episodes_capture_filtered/L2-2_v1__seed01/weather_meta.jsonl",
        "roster": "aw_data/render_ready_episodes_capture_filtered/L2-2_v1__seed01/global_entity_roster.json",
    },
    "script_path": "scripts/arms/L2-2_v1/seed01/grp_gnss_anomaly__route_drift/{arm_id}.py",
    "output_path": "arms/L2-2_v1/seed01/grp_gnss_anomaly__route_drift/{arm_id}",
    "group_key": "L2-2_v1/seed01/grp_gnss_anomaly__route_drift",
    "arm_id": "arm_early_relocalization_before_drift",
    "mode": "intervention",
    "actions": [{'tick': 280,
      'action': {'type': 'set_runtime_state',
                 'action_id': 'l2_arm_early_relocalization_started',
                 'entity_id': 'uav_l2_2_v1',
                 'delay_ticks': 0,
                 'state_patch': {'navigation_state': {'visual_relocalization_active': True,
                                                      'relocalization_complete': False}}}},
     {'tick': 280,
      'action': {'type': 'set_visual_state',
                 'action_id': 'l2_arm_early_relocalization',
                 'entity_id': 'uav_l2_2_v1',
                 'visual_state': {'mode': 'visual_relocalization'}}},
     {'tick': 280,
      'action': {'type': 'move_entity',
                 'action_id': 'l2_arm_early_relocalization_correct_route',
                 'entity_id': 'uav_l2_2_v1',
                 'waypoints_enu_m': [[6341.378, 7694.749, 30.0]],
                 'velocity_mps': 5.0}},
     {'tick': 310,
      'action': {'type': 'set_runtime_state',
                 'action_id': 'l2_arm_early_relocalization_completed',
                 'entity_id': 'uav_l2_2_v1',
                 'delay_ticks': 0,
                 'state_patch': {'navigation_state': {'visual_relocalization_active': False,
                                                      'relocalization_complete': True}}}}],
    "omit_action_ids": [
        "move_uav_multipath_drift"
    ],
    "expected": "occurs",
    "rationale": '漂移前提前视觉重定位：保留多径环境，省略漂移航路并执行远离立面的航路；在280派发重定位启动，沿factual既有30 tick处理过程于310派发完成，GNSS质量由当前误差状态机计算。',
    "falsified_if": '窗口内出现 structure_near_miss，或实际执行漂移航路，或重定位未按既有30 tick过程完成。',
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
