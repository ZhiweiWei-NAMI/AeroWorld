"""L1-1_v1/seed00/grp_reroute_before_restricted_entry/arm_reroute_before_restricted_entry。

在接近段之前改为直飞源脚本自身的安全航点，使本机始终位于禁限区多边形之外。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l1_window import execute_config

EPISODE_ID = "L1-1_v1__seed00"
GROUP_KEY = "L1-1_v1/seed00/grp_reroute_before_restricted_entry"
FORK_TICK = 250
SUBJECT = "uav_l1_1_v1_primary"
MODE = "intervention"
FACTUAL_ROOT = "aw_data"
WINDOW = {"start_tick": 245, "fork_tick": 250, "end_tick": 350}

PLAN = {
    # Mechanism of this branch, written literally: the source actions it replaces
    # and the trajectory it injects instead.
    "omit_action_ids": ["move_boundary_approach"],
    "actions":     [
        {
            "tick": 250,
            "action": {
                "type": "move_entity",
                "action_id": "arm_reroute_before_restricted_entry",
                "entity_id": "uav_l1_1_v1_primary",
                "velocity_mps": 8.0,
                "waypoints_enu_m": [
                    [
                        6622.401,
                        6522.145,
                        30.0
                    ],
                    [
                        6617.401,
                        6547.145,
                        30.0
                    ]
                ]
            }
        }
    ],
    "trigger_rewrites": [],}


if __name__ == "__main__":
    from Dataset.tools.l1_window import load_config
    config = load_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"))
    assert config["episode_id"] == EPISODE_ID
    assert config["mode"] == MODE
    assert {k: config["window"][k] for k in WINDOW} == WINDOW, (config["window"], WINDOW)
    assert config["entity_ids"] == ["uav_l1_1_v1_primary"]
    assert FACTUAL_ROOT == config["provenance"]["parameter_sources"]["factual_root"]
    assert config["execution"]["omit_action_ids"] == PLAN["omit_action_ids"], (
        config["execution"]["omit_action_ids"], PLAN["omit_action_ids"])
    assert config["execution"]["actions"] == PLAN["actions"], (
        config["execution"]["actions"], PLAN["actions"])
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem, "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
