"""L1-3_v2/seed02/grp_prevent_target_pair_convergence/arm_prevent_target_pair_convergence。

在冲突形成前改写入侵机航段，使其从主机接近线外侧通过，指定机对不形成汇聚。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l1_window import execute_config

EPISODE_ID = "L1-3_v2__seed02"
GROUP_KEY = "L1-3_v2/seed02/grp_prevent_target_pair_convergence"
FORK_TICK = 220
SUBJECT = "intruder_l1_3_v2"
MODE = "intervention"
FACTUAL_ROOT = "aw_data"
WINDOW = {"start_tick": 215, "fork_tick": 220, "end_tick": 430}

PLAN = {
    # Mechanism of this branch, written literally: the source actions it replaces
    # and the trajectory it injects instead.
    "omit_action_ids": ["move_intruder_near_miss_converge"],
    "actions":     [
        {
            "tick": 220,
            "action": {
                "type": "move_entity",
                "action_id": "arm_prevent_target_pair_convergence",
                "entity_id": "intruder_l1_3_v2",
                "velocity_mps": 10.0,
                "waypoints_enu_m": [
                    [
                        6379.822,
                        7016.332,
                        29.0
                    ],
                    [
                        6403.822,
                        7046.332,
                        29.0
                    ],
                    [
                        6403.131,
                        7028.31,
                        29.0
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
    assert config["entity_ids"] == ["intruder_l1_3_v2", "uav_l1_3_v2_primary"]
    assert FACTUAL_ROOT == config["provenance"]["parameter_sources"]["factual_root"]
    assert config["execution"]["omit_action_ids"] == PLAN["omit_action_ids"], (
        config["execution"]["omit_action_ids"], PLAN["omit_action_ids"])
    assert config["execution"]["actions"] == PLAN["actions"], (
        config["execution"]["actions"], PLAN["actions"])
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem, "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
