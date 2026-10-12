"""L1-3_v2/seed01/grp_avoidance_withheld/arm_avoidance_withheld。

保留冲突形成，取消本机规避与双方返回段，使指定机对沿名义航迹继续。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l1_window import execute_config

EPISODE_ID = "L1-3_v2__seed01"
GROUP_KEY = "L1-3_v2/seed01/grp_avoidance_withheld"
FORK_TICK = 240
SUBJECT = "intruder_l1_3_v2"
MODE = "intervention"
FACTUAL_ROOT = "aw_data"
WINDOW = {"start_tick": 235, "fork_tick": 240, "end_tick": 520}

PLAN = {
    # Mechanism of this branch, written literally: the source actions it replaces
    # and the trajectory it injects instead.
    "omit_action_ids": ["move_boundary_avoidance", "move_boundary_return_safe", "move_intruder_l1_3_v2_return_safe"],
    "actions":     [],
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
