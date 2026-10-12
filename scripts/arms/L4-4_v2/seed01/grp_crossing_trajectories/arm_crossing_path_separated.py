"""Dedicated implementation for L4-4_v2/seed01/grp_crossing_trajectories/arm_crossing_path_separated.

Mechanism (from design/L4-4_v2/seed01/grp_crossing_trajectories/experiment.json): arm_crossing_path_separated.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-4_v2/seed01/grp_crossing_trajectories/arm_crossing_path_separated.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = 'L4-4_v2__seed01'
GROUP_KEY = 'L4-4_v2/seed01/grp_crossing_trajectories'
FORK_TICK = 240
ANCHOR = 'crossing_trajectories'
WINDOW = {'end_tick': 700, 'fork_tick': 240, 'intervention_order': 'source_actions_then_intervention', 'phase': 'after_state_sample_before_actions', 'start_tick': 235}

PLAN = {'omit_action_ids': ['move_uav_falling_cross'], 'actions': [{'tick': 240, 'action': {'type': 'move_entity', 'action_id': 'arm_crossing_path_separated_move_uav', 'entity_id': 'uav_falls_vehicle_l4_4_v2', 'velocity_mps': 6.0, 'waypoints_enu_m': [[6097.872, 7530.636, 20.0], [6116.332, 7529.505, 20.0], [6134.791, 7528.374, 20.0], [6134.791, 7528.374, 36.0], [6099.261, 7527.709, 36.0], [6099.261, 7527.709, 3.0]], 'terminal_feasibility': {'model': 'constant_speed_path_with_explicit_landing_reference_v2', 'landing_reference_enu_m': [6099.261, 7527.709, 3.0], 'home_pose_origin_z_m': 3.0, 'touchdown_altitude_tolerance_m': 1.5, 'touchdown_dwell_ticks': 5, 'max_speed_mps': 8.0}}}], 'trigger_rewrites': []}

if __name__ == "__main__":
    config = load_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"))
    assert config["episode_id"] == EPISODE_ID
    assert config["window"] == WINDOW, (config["window"], WINDOW)
    assert config["execution"] == PLAN, (config["execution"], PLAN)
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem,
                      "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
