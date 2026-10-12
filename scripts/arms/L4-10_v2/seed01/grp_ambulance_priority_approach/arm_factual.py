"""Dedicated implementation for L4-10_v2/seed01/grp_ambulance_priority_approach/arm_factual.

Mechanism (from design/L4-10_v2/seed01/grp_ambulance_priority_approach/experiment.json): arm_factual.
The plan below is written for this arm alone: the literals mirror the configuration in
interventions/L4-10_v2/seed01/grp_ambulance_priority_approach/arm_factual.yaml and the assertion at the bottom proves they still agree."""
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

PLAN = {'omit_action_ids': [], 'actions': [], 'trigger_rewrites': []}

if __name__ == "__main__":
    config = load_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"))
    assert config["episode_id"] == EPISODE_ID
    assert config["window"] == WINDOW, (config["window"], WINDOW)
    assert config["execution"] == PLAN, (config["execution"], PLAN)
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem,
                      "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
