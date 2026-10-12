"""Dedicated implementation for L4-8_v1/seed00/grp_evacuation_triggered/arm_command_state_withheld.

Mechanism (from design/L4-8_v1/seed00/grp_evacuation_triggered/experiment.json): arm_command_state_withheld.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-8_v1/seed00/grp_evacuation_triggered/arm_command_state_withheld.yaml and the assertion at the bottom proves they still agree."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l4_window import load_config
from Dataset.tools.l4_ue_delivery import execute_config

EPISODE_ID = "L4-8_v1__seed00"
GROUP_KEY = "L4-8_v1/seed00/grp_evacuation_triggered"
FORK_TICK = 220
ANCHOR = "evacuation_triggered"
WINDOW = {"end_tick": 640, "fork_tick": 220, "intervention_order": "source_actions_then_intervention", "phase": "after_state_sample_before_actions", "start_tick": 210}

PLAN = {
    "omit_action_ids": [
        "move_crowd_evacuation_00",
        "move_crowd_evacuation_01",
        "move_crowd_evacuation_02",
        "move_crowd_evacuation_03",
        "move_crowd_evacuation_04",
        "move_crowd_evacuation_05",
        "move_crowd_evacuation_06",
        "move_crowd_evacuation_07",
        "set_runtime_state_l4_8_v1_evacuation_triggered_00",
        "set_runtime_state_l4_8_v1_evacuation_triggered_01",
        "set_runtime_state_l4_8_v1_evacuation_triggered_02",
        "set_runtime_state_l4_8_v1_evacuation_triggered_03",
        "set_runtime_state_l4_8_v1_evacuation_triggered_04",
        "set_runtime_state_l4_8_v1_evacuation_triggered_05",
        "set_runtime_state_l4_8_v1_evacuation_triggered_06",
        "set_runtime_state_l4_8_v1_evacuation_triggered_07",
        "capture_crowd_safe_zone",
        "set_runtime_state_l4_8_v1_crowd_safe_zone_reached_00",
        "set_runtime_state_l4_8_v1_crowd_safe_zone_reached_01",
        "set_runtime_state_l4_8_v1_crowd_safe_zone_reached_02",
        "set_runtime_state_l4_8_v1_crowd_safe_zone_reached_03",
        "set_runtime_state_l4_8_v1_crowd_safe_zone_reached_04",
        "set_runtime_state_l4_8_v1_crowd_safe_zone_reached_05",
        "set_runtime_state_l4_8_v1_crowd_safe_zone_reached_06",
        "set_runtime_state_l4_8_v1_crowd_safe_zone_reached_07",
        "capture_crowd_safe_hold",
        "set_runtime_state_l4_8_v1_crowd_safe_hold_00",
    ],
    "actions": [],
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
