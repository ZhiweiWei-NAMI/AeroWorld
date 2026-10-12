"""Dedicated implementation for L4-1_v2/seed01/grp_converging_approach/arm_separation_forced.

Mechanism (from design/L4-1_v2/seed01/grp_converging_approach/experiment.json): arm_separation_forced.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-1_v2/seed01/grp_converging_approach/arm_separation_forced.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = "L4-1_v2__seed01"
GROUP_KEY = "L4-1_v2/seed01/grp_converging_approach"
FORK_TICK = 240
ANCHOR = "converging_approach"
WINDOW = {
    "end_tick": 540,
    "fork_tick": 240,
    "intervention_order": "source_actions_then_intervention",
    "phase": "after_state_sample_before_actions",
    "start_tick": 230,
}

PLAN = {
    "omit_action_ids": [
        "move_uav_b_converge",
    ],
    "actions": [
        {
            "tick": 240,
            "action": {
                "type": "move_entity",
                "action_id": "arm_move_uav_conflict_l4_1_v2_b_separation_hold",
                "entity_id": "uav_conflict_l4_1_v2_b",
                "velocity_mps": 7.5,
                "waypoints_enu_m": [
                    [
                        6205.444,
                        6872.292,
                        30.0,
                    ],
                    [
                        6225.444,
                        6885.292,
                        60.0,
                    ],
                    [
                        6240.444,
                        6902.292,
                        80.0,
                    ],
                ],
            },
        },
    ],
    "trigger_rewrites": [],
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
