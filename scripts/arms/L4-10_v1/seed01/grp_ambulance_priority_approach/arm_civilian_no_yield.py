"""Dedicated implementation for L4-10_v1/seed01/grp_ambulance_priority_approach/arm_civilian_no_yield.

Mechanism (from design/L4-10_v1/seed01/grp_ambulance_priority_approach/experiment.json): arm_civilian_no_yield.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-10_v1/seed01/grp_ambulance_priority_approach/arm_civilian_no_yield.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = 'L4-10_v1__seed01'
GROUP_KEY = 'L4-10_v1/seed01/grp_ambulance_priority_approach'
FORK_TICK = 220
ANCHOR = 'ambulance_priority_approach'
WINDOW = {'end_tick': 560, 'fork_tick': 220, 'intervention_order': 'source_actions_then_intervention', 'phase': 'after_state_sample_before_actions', 'start_tick': 210}

PLAN = {'omit_action_ids': ['move_civilian_priority_pull_over', 'set_runtime_state_l4_10_v1_civilian_vehicle_yields_00', 'set_civilian_yielding', 'remove_ambulance_after_priority_passage', 'remove_civilian_after_priority_passage'], 'actions': [{'tick': 220, 'action': {'type': 'move_entity', 'action_id': 'arm_civilian_no_yield_move_civilian', 'entity_id': 'civilian_yield_l4_10_v1', 'velocity_mps': 4.5, 'waypoints_enu_m': [[6232.633, 7758.157, 0.0], [6237.805966705222, 7761.196805169195, 0.0], [6259.359994643647, 7773.862660040841, 0.0]]}}, {'tick': 240, 'action': {'type': 'move_entity', 'action_id': 'arm_civilian_no_yield_hold_ambulance', 'entity_id': 'ambulance_priority_l4_10_v1', 'velocity_mps': 0.5, 'waypoints_enu_m': [[6219.385333, 7750.833333, 0.0], [6221.109655235075, 7751.846601389731, 0.0]]}}], 'trigger_rewrites': []}

if __name__ == "__main__":
    config = load_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"))
    assert config["episode_id"] == EPISODE_ID
    assert config["window"] == WINDOW, (config["window"], WINDOW)
    assert config["execution"] == PLAN, (config["execution"], PLAN)
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem,
                      "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
