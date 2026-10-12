"""L1-3_v2/seed00/grp_prevent_target_pair_convergence/arm_factual。

记录声明事实源中的所选事件及恢复，不预设干预效果。
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l1_window import execute_config

EPISODE_ID = "L1-3_v2__seed00"
GROUP_KEY = "L1-3_v2/seed00/grp_prevent_target_pair_convergence"
FORK_TICK = 220
SUBJECT = "intruder_l1_3_v2"
MODE = "factual_observation"
FACTUAL_ROOT = "aw_data"
WINDOW = {"start_tick": 215, "fork_tick": 220, "end_tick": 430}

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
    assert config["entity_ids"] == ["intruder_l1_3_v2", "uav_l1_3_v2_primary"]
    assert FACTUAL_ROOT == config["provenance"]["parameter_sources"]["factual_root"]
    assert config["execution"]["omit_action_ids"] == PLAN["omit_action_ids"], (
        config["execution"]["omit_action_ids"], PLAN["omit_action_ids"])
    assert config["execution"]["actions"] == PLAN["actions"], (
        config["execution"]["actions"], PLAN["actions"])
    result = execute_config(ROOT / "interventions" / GROUP_KEY / (Path(__file__).stem + ".yaml"), PLAN)
    print(json.dumps({"arm": GROUP_KEY + "/" + Path(__file__).stem, "fork_tick": FORK_TICK,
                      "window_status": result["window_status"]}, ensure_ascii=False))
