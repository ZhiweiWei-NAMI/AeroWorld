"""Dedicated implementation for L4-2_v2/seed01/grp_facade_approach/arm_recovery_precondition_removed.

Mechanism (from design/L4-2_v2/seed01/grp_facade_approach/experiment.json): arm_recovery_precondition_removed.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-2_v2/seed01/grp_facade_approach/arm_recovery_precondition_removed.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = "L4-2_v2__seed01"
GROUP_KEY = "L4-2_v2/seed01/grp_facade_approach"
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
        "set_runtime_state_l4_2_v2_facade_recovery_00",
    ],
    "actions": [],
    "trigger_rewrites": [
        {
            "trigger_id": "trig_facade_proximity",
            "mechanism": "min_true_ticks_shift",
            "delay_ticks": 15,
        },
        {
            "trigger_id": "trig_after_facade_proximity_delay_30",
            "mechanism": "fired_after_delay_shift",
            "delay_ticks": 15,
        },
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
