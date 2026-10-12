"""Dedicated implementation for L4-10_v1/seed02/grp_ambulance_priority_approach/arm_civilian_yield_delayed.

Mechanism (from design/L4-10_v1/seed02/grp_ambulance_priority_approach/experiment.json): arm_civilian_yield_delayed.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-10_v1/seed02/grp_ambulance_priority_approach/arm_civilian_yield_delayed.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = 'L4-10_v1__seed02'
GROUP_KEY = 'L4-10_v1/seed02/grp_ambulance_priority_approach'
FORK_TICK = 220
ANCHOR = 'ambulance_priority_approach'
WINDOW = {'end_tick': 560, 'fork_tick': 220, 'intervention_order': 'source_actions_then_intervention', 'phase': 'after_state_sample_before_actions', 'start_tick': 210}

PLAN = {'omit_action_ids': ['move_civilian_priority_pull_over', 'set_runtime_state_l4_10_v1_civilian_vehicle_yields_00', 'remove_ambulance_after_priority_passage', 'remove_civilian_after_priority_passage', 'set_civilian_yielding'], 'actions': [{'tick': 240, 'action': {'type': 'move_entity', 'action_id': 'arm_civilian_yield_delayed_ambulance_wait', 'entity_id': 'ambulance_priority_l4_10_v1', 'velocity_mps': 0.5, 'waypoints_enu_m': [[6219.385333, 7750.833333, 0.0], [6221.971816352611, 7752.353235584597, 0.0]]}}, {'tick': 280, 'action': {'type': 'move_entity', 'action_id': 'arm_civilian_yield_delayed_pull_over', 'entity_id': 'civilian_yield_l4_10_v1', 'velocity_mps': 4.5, 'waypoints_enu_m': [[6232.633, 7758.157, 0.0], [6234.091, 7759.719, 0.0], [6235.547, 7761.281, 0.0], [6237.003, 7762.843, 0.0], [6238.461, 7764.405, 0.0], [6240.252, 7765.296, 0.0], [6242.041, 7766.185, 0.0], [6243.833, 7767.077, 0.0], [6245.616, 7767.967, 0.0], [6247.411, 7768.865, 0.0], [6249.201, 7769.756, 0.0], [6250.993, 7770.646, 0.0], [6252.802, 7771.535, 0.0], [6254.649, 7772.403, 0.0], [6256.485, 7773.225, 0.0], [6258.312, 7774.033, 0.0]]}}, {'tick': 280, 'action': {'type': 'set_visual_state', 'action_id': 'arm_civilian_yield_delayed_visual', 'entity_id': 'civilian_yield_l4_10_v1', 'visual_state': {'mode': 'yielding'}}}, {'tick': 280, 'action': {'type': 'set_runtime_state', 'action_id': 'arm_civilian_yield_delayed_patch', 'entity_id': 'civilian_yield_l4_10_v1', 'state_patch': {'vehicle_state': {'yielding': True}}}}, {'tick': 330, 'action': {'type': 'move_entity', 'action_id': 'arm_civilian_yield_delayed_ambulance_resume', 'entity_id': 'ambulance_priority_l4_10_v1', 'velocity_mps': 10.0, 'waypoints_enu_m': [[6221.971816352611, 7752.353235584597, 0.0], [6264.785, 7773.635, 0.0], [6286.339027938425, 7786.300854871645, 0.0]]}}], 'trigger_rewrites': []}

if __name__ == "__main__":
    config = load_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"))
    assert config["episode_id"] == EPISODE_ID
    assert config["window"] == WINDOW, (config["window"], WINDOW)
    assert config["execution"] == PLAN, (config["execution"], PLAN)
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem,
                      "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
