"""Dedicated implementation for L4-9_v1/seed01/grp_vehicles_enter_intersection/arm_brake_declared_unstopped.

Mechanism (from design/L4-9_v1/seed01/grp_vehicles_enter_intersection/experiment.json): arm_brake_declared_unstopped.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-9_v1/seed01/grp_vehicles_enter_intersection/arm_brake_declared_unstopped.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = "L4-9_v1__seed01"
GROUP_KEY = "L4-9_v1/seed01/grp_vehicles_enter_intersection"
FORK_TICK = 230
ANCHOR = "vehicles_enter_intersection"
WINDOW = {"end_tick": 320, "fork_tick": 230, "intervention_order": "source_actions_then_intervention", "phase": "after_state_sample_before_actions", "start_tick": 220}

PLAN = {
    "omit_action_ids": [
    ],
    "actions": [
        {"tick": 276,
         "action": {
             "type": "move_entity",
             "action_id": "arm_brake_declared_unstopped_move_car_b",
             "entity_id": "car_intersection_l4_9_v1_b",
             "velocity_mps": 1.0,
             "waypoints_enu_m": [
                 [7051.5, 6207.94, 0.0],
                 [7049.54, 6213.61, 0.0],
             ],
         }},
    ],
    "trigger_rewrites": [
    ],
}

if __name__ == "__main__":
    config = load_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"))
    assert config["episode_id"] == EPISODE_ID
    assert config["window"] == WINDOW, (config["window"], WINDOW)
    assert config["execution"] == PLAN, (config["execution"], PLAN)
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem,
                      "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
