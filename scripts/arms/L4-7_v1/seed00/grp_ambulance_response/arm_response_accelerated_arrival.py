"""Dedicated implementation for L4-7_v1/seed00/grp_ambulance_response/arm_response_accelerated_arrival.

Mechanism (from design/L4-7_v1/seed00/grp_ambulance_response/experiment.json): arm_response_accelerated_arrival.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-7_v1/seed00/grp_ambulance_response/arm_response_accelerated_arrival.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = "L4-7_v1__seed00"
GROUP_KEY = "L4-7_v1/seed00/grp_ambulance_response"
FORK_TICK = 385
ANCHOR = "ambulance_response"
WINDOW = {"end_tick": 480, "fork_tick": 385, "intervention_order": "source_actions_then_intervention", "phase": "after_state_sample_before_actions", "start_tick": 365}

PLAN = {
    "omit_action_ids": [
    ],
    "actions": [
        {"tick": 385, "action": {"type": "move_entity", "action_id": "arm_ambulance_approach_18mps_seed00", "entity_id": "ambulance_fall_l4_7_v1", "waypoints_enu_m": [[6895.737, 7149.788, 0.0], [6889.739, 7149.648, 0.0], [6883.745, 7149.507, 0.0], [6877.733, 7149.368, 0.0], [6871.523, 7149.4, 0.0], [6865.525, 7149.55, 0.0], [6859.527, 7149.691, 0.0], [6853.529, 7149.841, 0.0]], "velocity_mps": 18.0}},
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
