"""Dedicated implementation for L4-11_v2/seed02/grp_av_safe_stop/arm_mrm_decelerated_not_stopped.

Mechanism (from design/L4-11_v2/seed02/grp_av_safe_stop/experiment.json): arm_mrm_decelerated_not_stopped.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-11_v2/seed02/grp_av_safe_stop/arm_mrm_decelerated_not_stopped.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = "L4-11_v2__seed02"
GROUP_KEY = "L4-11_v2/seed02/grp_av_safe_stop"
FORK_TICK = 260
ANCHOR = "av_safe_stop"
WINDOW = {"end_tick": 380, "fork_tick": 260, "intervention_order": "source_actions_then_intervention", "phase": "after_state_sample_before_actions", "start_tick": 230}

PLAN = {
    "omit_action_ids": [],
    "actions": [
        {
            "tick": 260,
            "action": {
                "type": "move_entity",
                "action_id": "arm_mrm_decelerated_not_stopped_creep",
                "entity_id": "av_fault_l4_11_v2",
                "velocity_mps": 0.5,
                "waypoints_enu_m": [
                    [
                        8045.103,
                        6750.785,
                        0.0
                    ],
                    [
                        8048.873,
                        6731.144,
                        0.0
                    ],
                    [
                        8052.643,
                        6711.502,
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
