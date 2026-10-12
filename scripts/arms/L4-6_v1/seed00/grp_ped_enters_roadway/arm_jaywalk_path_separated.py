"""Dedicated implementation for L4-6_v1/seed00/grp_ped_enters_roadway/arm_jaywalk_path_separated.

Mechanism (from design/L4-6_v1/seed00/grp_ped_enters_roadway/experiment.json): arm_jaywalk_path_separated.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-6_v1/seed00/grp_ped_enters_roadway/arm_jaywalk_path_separated.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = "L4-6_v1__seed00"
GROUP_KEY = "L4-6_v1/seed00/grp_ped_enters_roadway"
FORK_TICK = 220
ANCHOR = "ped_enters_roadway"
WINDOW = {"end_tick": 410, "fork_tick": 220, "intervention_order": "source_actions_then_intervention", "phase": "after_state_sample_before_actions", "start_tick": 210}

PLAN = {
    "omit_action_ids": [
    ],
    "actions": [
        {"tick": 220, "action": {"type": "move_entity", "action_id": "arm_ped_hold_lateral_v1_seed00", "entity_id": "ped_jaywalk_l4_6_v1", "waypoints_enu_m": [[6237.747, 7958.445, 0.0], [6237.747, 7962.445, 0.0]], "velocity_mps": 1.6}},
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
