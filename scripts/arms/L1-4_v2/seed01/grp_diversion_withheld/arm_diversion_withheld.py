"""L1-4_v2/seed01/grp_diversion_withheld/arm_diversion_withheld。

提前替换 secondary 的分流计划，保留进入容量格的航段；超限形成后继续前行，使共同占用持续更久。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l1_window import execute_config

EPISODE_ID = "L1-4_v2__seed01"
GROUP_KEY = "L1-4_v2/seed01/grp_diversion_withheld"
FORK_TICK = 355
SUBJECT = "uav_l1_4_v2_primary"
MODE = "intervention"
FACTUAL_ROOT = "aw_data"
WINDOW = {"start_tick": 350, "fork_tick": 355, "end_tick": 530}

PLAN = {
    # Mechanism of this branch, written literally: the source actions it replaces
    # and the trajectory it injects instead.
    "omit_action_ids": ["move_corridor_secondary_divert_out"],
    "actions":     [
        {
            "tick": 360,
            "action": {
                "type": "move_entity",
                "action_id": "arm_diversion_withheld_secondary_continue",
                "entity_id": "uav_l1_4_v2_secondary",
                "activity_type": "airborne",
                "post_activity_type": "landed",
                "terminal_feasibility": {"landing_reference_enu_m": [6547.625, 6561.326, 3.0]},
                "velocity_mps": 6.0,
                "waypoints_enu_m": [
                    [
                        6644.535,
                        6535.945,
                        50.0
                    ],
                    [
                        6744.535,
                        6535.945,
                        50.0
                    ],
                    [
                        6547.625,
                        6561.326,
                        50.0
                    ],
                    [
                        6547.625,
                        6561.326,
                        3.0
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
    assert config["entity_ids"] == ["uav_l1_4_v2_primary", "uav_l1_4_v2_secondary", "uav_l1_4_v2_tertiary"]
    assert FACTUAL_ROOT == config["provenance"]["parameter_sources"]["factual_root"]
    assert config["execution"]["omit_action_ids"] == PLAN["omit_action_ids"], (
        config["execution"]["omit_action_ids"], PLAN["omit_action_ids"])
    assert config["execution"]["actions"] == PLAN["actions"], (
        config["execution"]["actions"], PLAN["actions"])
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem, "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
