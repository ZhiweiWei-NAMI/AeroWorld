"""Dedicated implementation for L4-7_v2/seed02/grp_ambulance_response/arm_response_accelerated_arrival.

Mechanism (from design/L4-7_v2/seed02/grp_ambulance_response/experiment.json): arm_response_accelerated_arrival.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-7_v2/seed02/grp_ambulance_response/arm_response_accelerated_arrival.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = "L4-7_v2__seed02"
GROUP_KEY = "L4-7_v2/seed02/grp_ambulance_response"
FORK_TICK = 330
ANCHOR = "ambulance_response"
WINDOW = {"end_tick": 420, "fork_tick": 330, "intervention_order": "source_actions_then_intervention", "phase": "after_state_sample_before_actions", "start_tick": 310}

PLAN = {
    "omit_action_ids": [],
    "actions": [
        {
            "tick": 335,
            "action": {
                "type": "move_entity",
                "action_id": "arm_preempt_move_ambulance_to_fall_accelerated",
                "entity_id": "ambulance_fall_l4_7_v2",
                "velocity_mps": 18.0,
                "waypoints_enu_m": [
                    [
                        6709.283,
                        7097.885,
                        0.0
                    ],
                    [
                        6703.582,
                        7096.177,
                        0.0
                    ],
                    [
                        6697.825,
                        7094.63,
                        0.0
                    ],
                    [
                        6692.024,
                        7093.217,
                        0.0
                    ],
                    [
                        6686.197,
                        7091.886,
                        0.0
                    ],
                    [
                        6680.346,
                        7090.629,
                        0.0
                    ],
                    [
                        6674.475,
                        7089.391,
                        0.0
                    ]
                ]
            }
        }
    ],
    "trigger_rewrites": []
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
