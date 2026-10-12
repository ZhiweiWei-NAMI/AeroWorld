"""Dedicated implementation for L4-10_v2/seed01/grp_ambulance_priority_approach/arm_civilian_yield_delayed.

Mechanism (from design/L4-10_v2/seed01/grp_ambulance_priority_approach/experiment.json): arm_civilian_yield_delayed.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-10_v2/seed01/grp_ambulance_priority_approach/arm_civilian_yield_delayed.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = 'L4-10_v2__seed01'
GROUP_KEY = 'L4-10_v2/seed01/grp_ambulance_priority_approach'
FORK_TICK = 220
ANCHOR = 'ambulance_priority_approach'
WINDOW = {'end_tick': 560, 'fork_tick': 220, 'intervention_order': 'source_actions_then_intervention', 'phase': 'after_state_sample_before_actions', 'start_tick': 210}

PLAN = {'omit_action_ids': ['move_civilian_priority_pull_over', 'set_runtime_state_l4_10_v2_civilian_vehicle_yields_00', 'remove_ambulance_after_priority_passage', 'remove_civilian_after_priority_passage', 'set_civilian_yielding'], 'actions': [{'tick': 240, 'action': {'type': 'move_entity', 'action_id': 'arm_civilian_yield_delayed_ambulance_wait', 'entity_id': 'ambulance_priority_l4_10_v2', 'velocity_mps': 0.5, 'waypoints_enu_m': [[7975.783333, 7350.193167, 0.0], [7972.872146515181, 7349.4686028202295, 0.0]]}}, {'tick': 280, 'action': {'type': 'move_entity', 'action_id': 'arm_civilian_yield_delayed_pull_over', 'entity_id': 'civilian_yield_l4_10_v2', 'velocity_mps': 4.5, 'waypoints_enu_m': [[7963.903, 7347.221, 0.0], [7962.492, 7344.546, 0.0], [7961.103, 7341.882, 0.0], [7959.714, 7339.214, 0.0], [7958.327, 7336.542, 0.0], [7956.363, 7336.054, 0.0], [7954.445, 7335.572, 0.0], [7952.487, 7335.085, 0.0], [7950.56, 7334.6, 0.0], [7948.602, 7334.113, 0.0], [7946.685, 7333.631, 0.0], [7944.716, 7333.143, 0.0], [7942.779, 7332.665, 0.0], [7940.818, 7332.187, 0.0], [7938.894, 7331.714, 0.0], [7936.958, 7331.235, 0.0]]}}, {'tick': 280, 'action': {'type': 'set_visual_state', 'action_id': 'arm_civilian_yield_delayed_visual', 'entity_id': 'civilian_yield_l4_10_v2', 'visual_state': {'mode': 'yielding'}}}, {'tick': 280, 'action': {'type': 'set_runtime_state', 'action_id': 'arm_civilian_yield_delayed_patch', 'entity_id': 'civilian_yield_l4_10_v2', 'state_patch': {'vehicle_state': {'yielding': True}}}}, {'tick': 330, 'action': {'type': 'move_entity', 'action_id': 'arm_civilian_yield_delayed_ambulance_resume', 'entity_id': 'ambulance_priority_l4_10_v2', 'velocity_mps': 10.0, 'waypoints_enu_m': [[7972.872146515181, 7349.4686028202295, 0.0], [7929.208, 7338.601, 0.0], [7904.948112626508, 7332.562965168575, 0.0]]}}], 'trigger_rewrites': []}

if __name__ == "__main__":
    config = load_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"))
    assert config["episode_id"] == EPISODE_ID
    assert config["window"] == WINDOW, (config["window"], WINDOW)
    assert config["execution"] == PLAN, (config["execution"], PLAN)
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem,
                      "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
