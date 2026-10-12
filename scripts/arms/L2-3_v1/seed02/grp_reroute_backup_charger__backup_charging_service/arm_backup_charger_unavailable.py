"""ARM arm_backup_charger_unavailable —— L2-3_v1 seed02 事件链片段 reroute_backup_charger → backup_charging_service

机制：主站故障后 UAV 准备改航备用站；t=320 注入备用站不可用（availability=unavailable, fault=true），
省略改航飞行、备用站预约与接受补丁；t=330 注入 hold —— 两个充电站均不可用，UAV 原地安全保持，
不进入任何充电流程。
窗口依据：闭区间 [320,470]；320 为动作前共享检查点。观察 reroute_backup_charger → backup_charging_service 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。
预期：occurs —— 主站与备用站的不可用事件均持续成立，UAV 保持等待。
证伪：UAV 进入备用站充电流程，或任一充电站不可用谓词在窗口内恢复 false。
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
    "anchor_event": "reroute_backup_charger",
    "neighbor_event": "backup_charging_service",
    "chain_nodes": [
        ["reroute_backup_charger", 324],
        ["backup_charging_service", 684],
    ],
    "window": {
        "start_tick": 320,
        "fork_tick": 320,
        "end_tick": 470,
    },
    "window_basis": "闭区间 [320,470]；320 为动作前共享检查点。观察 reroute_backup_charger → backup_charging_service 的局部过程和分支直接后果；终点保留实跑观察范围，恢复、无事件或截断按实际谓词与事件判定，不声称覆盖完整生命周期。factual 由当前引擎无干预执行，冻结 episode 仅作来源及背景引用。",
    "entity_ids": [
        "charger_backup_l2_3_v1",
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
    "script_path": "scripts/arms/L2-3_v1/seed02/grp_reroute_backup_charger__backup_charging_service/{arm_id}.py",
    "output_path": "arms/L2-3_v1/seed02/grp_reroute_backup_charger__backup_charging_service/{arm_id}",
    "group_key": "L2-3_v1/seed02/grp_reroute_backup_charger__backup_charging_service",
    "arm_id": "arm_backup_charger_unavailable",
    "mode": "intervention",
    "actions": [{'tick': 320,
      'action': {'type': 'set_runtime_state',
                 'action_id': 'l2_arm_backup_charger_fault',
                 'entity_id': 'charger_backup_l2_3_v1',
                 'delay_ticks': 0,
                 'state_patch': {'facility_state': {'availability': 'unavailable', 'fault': True}}}},
     {'tick': 320,
      'action': {'type': 'set_visual_state',
                 'action_id': 'l2_arm_backup_charger_unavailable_visual',
                 'entity_id': 'charger_backup_l2_3_v1',
                 'visual_state': {'mode': 'unavailable'}}},
     {'tick': 330,
      'action': {'type': 'set_visual_state',
                 'action_id': 'l2_arm_backup_charger_hold',
                 'entity_id': 'uav_l2_3_v1',
                 'visual_state': {'mode': 'hold'}}}],
    "omit_action_ids": [
        "move_uav_backup_charger",
        "set_backup_charger_reserved",
        "set_runtime_state_l2_3_v1_reroute_backup_charger_00",
        "set_runtime_state_l2_3_v1_reroute_backup_charger_01"
    ],
    "expected": "occurs",
    "rationale": "备用站不可用：改航被抑制，UAV 在两站均不可用下原地保持，不进入充电；设施状态决定服务结果。",
    "falsified_if": "UAV 进入备用站充电流程，或任一充电站不可用谓词在窗口内恢复 false。"
}

if __name__ == "__main__":
    result = run_arm(SPEC)
    print(result["arm_id"], result["window_status"])
