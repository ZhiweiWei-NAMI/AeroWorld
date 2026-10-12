"""ARM arm_primary_charger_repaired_before_reroute —— L2-3_v1 seed02 事件链片段 charger_unavailable → uav_inspects_charger

机制：主充电站故障与 UAV 检查照常发生；检查完成后 t=310 注入主站快速修复
（availability=available, fault=false），省略改航备用站整组动作；t=315 注入 UAV
飞往主充电站并触地，t=370 注入 charging 状态 —— UAV 在主站充电，不再改航。
窗口依据：闭区间 [235,420]；240 为动作前共享检查点。观察 charger_unavailable → uav_inspects_charger 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：occurs —— 主站故障事件真实发生并在修复后终止；备份站全程可用且未被使用。
证伪：UAV 仍飞往备用充电站，或主站故障未在窗口内终止。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l2_window import run_arm

SPEC = {
    "profile": "L2-3_v1",
    "seed": "seed02",
    "episode_id": "L2-3_v1__seed02",
    "anchor_event": "charger_unavailable",
    "neighbor_event": "uav_inspects_charger",
    "chain_nodes": [
        ["charger_unavailable", 240],
        ["uav_inspects_charger", 270],
    ],
    "window": {
        "start_tick": 235,
        "fork_tick": 240,
        "end_tick": 420,
    },
    "window_basis": "闭区间 [235,420]；240 为动作前共享检查点。观察 charger_unavailable → uav_inspects_charger 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "charger_primary_l2_3_v1",
        "uav_l2_3_v1"
    ],
    "predicate_ids": [
        "facility.charging_station_unavailable",
        "facility.charging_queue_over_capacity"
    ],
    "target_families": [
        "charger_unavailable",
        "charging_queue_over_capacity"
    ],
    "factual_root": "aw_data",
    "sources": {
        "scene": "Dataset/scenarios/L2_infrastructure/failure/L2-3_v1/scene_setup.json",
        "script": "Dataset/scenarios/L2_infrastructure/failure/L2-3_v1/event_script.json",
        "trajectories": "aw_data/render_ready_episodes_capture_filtered/L2-3_v1__seed02/trajectories.jsonl",
        "weather": "aw_data/render_ready_episodes_capture_filtered/L2-3_v1__seed02/weather_meta.jsonl",
        "roster": "aw_data/render_ready_episodes_capture_filtered/L2-3_v1__seed02/global_entity_roster.json"
    },
    "script_path": "scripts/arms/L2-3_v1/seed02/grp_charger_unavailable__uav_inspects_charger/{arm_id}.py",
    "output_path": "arms/L2-3_v1/seed02/grp_charger_unavailable__uav_inspects_charger/{arm_id}",
    "group_key": "L2-3_v1/seed02/grp_charger_unavailable__uav_inspects_charger",
    "arm_id": "arm_primary_charger_repaired_before_reroute",
    "mode": "intervention",
    "actions": [{'tick': 310,
      'action': {'type': 'set_runtime_state',
                 'action_id': 'l2_arm_primary_charger_repair',
                 'entity_id': 'charger_primary_l2_3_v1',
                 'delay_ticks': 0,
                 'state_patch': {'facility_state': {'availability': 'available', 'fault': False}}}},
     {'tick': 310,
      'action': {'type': 'set_runtime_state',
                 'action_id': 'l2_arm_primary_charger_selected',
                 'entity_id': 'uav_l2_3_v1',
                 'delay_ticks': 0,
                 'state_patch': {'navigation_state': {'primary_charger_selected': True,
                                                      'replan_active': False}}}},
     {'tick': 310,
      'action': {'type': 'set_visual_state',
                 'action_id': 'l2_arm_primary_charger_available',
                 'entity_id': 'charger_primary_l2_3_v1',
                 'visual_state': {'mode': 'available'}}},
     {'tick': 315,
      'action': {'type': 'move_entity',
                 'action_id': 'l2_arm_primary_charger_return',
                 'entity_id': 'uav_l2_3_v1',
                 'waypoints_enu_m': [[6070.968, 6401.408, 10.0], [6071.117, 6417.794, 0.0]],
                 'velocity_mps': 4.0,
                 'activity_type': 'return_to_pad',
                 'post_activity_type': 'landed',
                 'terminal_feasibility': {'landing_reference_enu_m': [6071.117, 6417.794, 0.0]}}},
     {'tick': 370,
      'action': {'type': 'set_visual_state',
                 'action_id': 'l2_arm_primary_charger_charging',
                 'entity_id': 'uav_l2_3_v1',
                 'visual_state': {'mode': 'charging'}}},
     {'tick': 370,
      'action': {'type': 'set_visual_state',
                 'action_id': 'l2_arm_primary_charger_serving',
                 'entity_id': 'charger_primary_l2_3_v1',
                 'visual_state': {'mode': 'serving'}}}],
    "omit_action_ids": [
        "move_uav_backup_charger",
        "set_backup_charger_reserved",
        "set_runtime_state_l2_3_v1_reroute_backup_charger_00",
        "set_runtime_state_l2_3_v1_reroute_backup_charger_01"
    ],
    "expected": "occurs",
    "rationale": "检查后快速修复主站：故障先真实发生并被 UAV 检查确认，修复后 UAV 返回主站充电，不再改航备用站；设施状态决定 UAV 去哪个站。",
    "falsified_if": "UAV 仍飞往备用充电站，或主站故障未在窗口内终止。"
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
