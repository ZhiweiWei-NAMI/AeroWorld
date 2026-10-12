"""Dedicated implementation for L4-5_v2/seed01/grp_uav_low_descent/arm_hold_until_area_clear.

Mechanism (from design/L4-5_v2/seed01/grp_uav_low_descent/experiment.json): arm_hold_until_area_clear.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-5_v2/seed01/grp_uav_low_descent/arm_hold_until_area_clear.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = 'L4-5_v2__seed01'
GROUP_KEY = 'L4-5_v2/seed01/grp_uav_low_descent'
FORK_TICK = 230
ANCHOR = 'uav_low_descent'
WINDOW = {'end_tick': 780, 'fork_tick': 230, 'intervention_order': 'source_actions_then_intervention', 'phase': 'after_state_sample_before_actions', 'start_tick': 225}

PLAN = {'omit_action_ids': ['move_uav_ped_descent'], 'actions': [{'tick': 230, 'action': {'type': 'move_entity', 'action_id': 'arm_hold_until_clear_descent', 'entity_id': 'uav_ped_nearmiss_l4_5_v2', 'velocity_mps': 4.5, 'waypoints_enu_m': [[6655.113, 6518.441, 30.0], [6655.81, 6510.472, 15.0], [6656.507, 6502.502, 10.0]]}}, {'tick': 232, 'action': {'type': 'set_runtime_state', 'action_id': 'arm_hold_until_clear_declare_hold', 'entity_id': 'uav_ped_nearmiss_l4_5_v2', 'state_patch': {'control_state': {'hold_active': True}}}}, {'tick': 300, 'action': {'type': 'move_entity', 'action_id': 'arm_hold_until_clear_pedestrian_move', 'entity_id': 'ped_nearmiss_l4_5_v2', 'velocity_mps': 1.5, 'waypoints_enu_m': [[6656.507, 6502.502, 0.0], [6642.124, 6508.178, 0.0]]}}, {'tick': 560, 'action': {'type': 'set_runtime_state', 'action_id': 'arm_hold_until_clear_release_hold', 'entity_id': 'uav_ped_nearmiss_l4_5_v2', 'state_patch': {'control_state': {'hold_active': False}}}}, {'tick': 560, 'action': {'type': 'move_entity', 'action_id': 'arm_hold_until_clear_land', 'entity_id': 'uav_ped_nearmiss_l4_5_v2', 'velocity_mps': 2.0, 'waypoints_enu_m': [[6656.507, 6502.502, 10.0], [6656.507, 6502.502, 5.0], [6656.507, 6502.502, 3.0]], 'terminal_feasibility': {'model': 'constant_speed_path_with_explicit_landing_reference_v2', 'landing_reference_enu_m': [6656.507, 6502.502, 3.0], 'home_pose_origin_z_m': 3.0, 'touchdown_altitude_tolerance_m': 1.5, 'touchdown_dwell_ticks': 5, 'max_speed_mps': 8.0}}}], 'trigger_rewrites': []}

if __name__ == "__main__":
    config = load_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"))
    assert config["episode_id"] == EPISODE_ID
    assert config["window"] == WINDOW, (config["window"], WINDOW)
    assert config["execution"] == PLAN, (config["execution"], PLAN)
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem,
                      "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
