"""Dedicated implementation for L4-4_v2/seed00/grp_crossing_trajectories/arm_emergency_stop_withheld.

Mechanism (from design/L4-4_v2/seed00/grp_crossing_trajectories/experiment.json): arm_emergency_stop_withheld.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-4_v2/seed00/grp_crossing_trajectories/arm_emergency_stop_withheld.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = 'L4-4_v2__seed00'
GROUP_KEY = 'L4-4_v2/seed00/grp_crossing_trajectories'
FORK_TICK = 240
ANCHOR = 'crossing_trajectories'
WINDOW = {'end_tick': 700, 'fork_tick': 240, 'intervention_order': 'source_actions_then_intervention', 'phase': 'after_state_sample_before_actions', 'start_tick': 235}

PLAN = {'omit_action_ids': ['move_vehicle_emergency_brake', 'set_runtime_state_l4_4_v2_vehicle_roof_contact_01'], 'actions': [{'tick': 315, 'action': {'type': 'move_entity', 'action_id': 'arm_emergency_stop_withheld_move_vehicle', 'entity_id': 'car_under_uav_l4_4_v2', 'velocity_mps': 0.3, 'waypoints_enu_m': [[6134.791, 7528.374, 0.0], [6152.958456327348, 7536.737224891985, 0.0]]}}], 'trigger_rewrites': []}

if __name__ == "__main__":
    config = load_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"))
    assert config["episode_id"] == EPISODE_ID
    assert config["window"] == WINDOW, (config["window"], WINDOW)
    assert config["execution"] == PLAN, (config["execution"], PLAN)
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem,
                      "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
