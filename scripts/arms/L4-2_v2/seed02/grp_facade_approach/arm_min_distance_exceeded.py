"""Dedicated implementation for L4-2_v2/seed02/grp_facade_approach/arm_min_distance_exceeded.

Mechanism (from design/L4-2_v2/seed02/grp_facade_approach/experiment.json): arm_min_distance_exceeded.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-2_v2/seed02/grp_facade_approach/arm_min_distance_exceeded.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = "L4-2_v2__seed02"
GROUP_KEY = "L4-2_v2/seed02/grp_facade_approach"
FORK_TICK = 250
ANCHOR = "facade_approach"
WINDOW = {
    "end_tick": 500,
    "fork_tick": 250,
    "intervention_order": "source_actions_then_intervention",
    "phase": "after_state_sample_before_actions",
    "start_tick": 230,
}

PLAN = {
    "omit_action_ids": [
        "move_uav_facade_near_miss_approach",
    ],
    "actions": [
        {
            "tick": 250,
            "action": {
                "type": "move_entity",
                "action_id": "arm_move_uav_facade_outer_approach",
                "entity_id": "uav_facade_l4_2_v2",
                "velocity_mps": 7.0,
                "waypoints_enu_m": [
                    [
                        6601.935,
                        6508.06,
                        31.0,
                    ],
                    [
                        6650.338,
                        6458.054,
                        28.5,
                    ],
                    [
                        6615.022,
                        6514.904,
                        26.0,
                    ],
                    [
                        6628.525,
                        6523.84,
                        18.0,
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
