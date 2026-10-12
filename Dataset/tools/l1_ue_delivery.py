"""Execute a fixed L1 definition through the measured UE export boundary.

`l1_window.execute_config` already writes the arm's own raw/ue_* engine inputs;
this module binds the branch-local UE package from those measured rows plus the
frozen factual episode that supplies the real background.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from Dataset.tools import l1_window
from Dataset.tools.arm_ue_export import export_arm


def execute_config(config_path, plan):
    config_path = Path(config_path)
    config = l1_window.load_config(config_path)
    l1_window.execute_config(config_path, plan)
    return _bind(config)


def export_l1_arm(config_path):
    """Rebind the UE package of an already-executed arm from its measured rows."""
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _bind(config)


def _bind(config: dict) -> dict:
    arm = l1_window.ROOT / config["output"]
    canonical = l1_window.ROOT / Path(config["sources"]["trajectories"]).parent
    export_arm(arm, canonical,
               l1_window.ROOT / config["sources"]["scene"],
               l1_window.ROOT / config["sources"]["script"])
    manifest = l1_window.read_json(arm / "manifest.json")
    manifest["ue_entry_module"] = {"path": "Dataset/tools/l1_ue_delivery.py"}
    l1_window.write_json(arm / "manifest.json", manifest)
    return manifest
