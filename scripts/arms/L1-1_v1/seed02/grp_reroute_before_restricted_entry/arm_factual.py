"""L1-1_v1/seed02/grp_reroute_before_restricted_entry/arm_factual。

记录声明事实源中的所选事件及恢复，不预设干预效果。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l1_window import execute_config

EPISODE_ID = "L1-1_v1__seed02"
GROUP_KEY = "L1-1_v1/seed02/grp_reroute_before_restricted_entry"
FORK_TICK = 250
SUBJECT = "uav_l1_1_v1_primary"
MODE = "factual_observation"
FACTUAL_ROOT = "aw_data"
WINDOW = {"start_tick": 245, "fork_tick": 250, "end_tick": 350}

PLAN = {
    "omit_action_ids": [],
    # Factual branch: the frozen factual is the observation; nothing is changed.
    "actions": [],
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
