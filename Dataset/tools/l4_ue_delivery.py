"""Execute a fixed L4 definition through the measured UE export boundary."""
from __future__ import annotations

from Dataset.tools import l4_window as window
from Dataset.tools.arm_ue_export import export_arm
from Dataset.tools.l4_repair_delivery import write_manifest_inputs


def execute_config(config_path, plan):
    config = window.load_config(config_path)
    window.execute_config(config_path, plan)
    arm = window.ROOT / config['output']
    write_manifest_inputs(arm, config)
    export_arm(arm, window.ROOT / config['sources']['trajectories'].rsplit('/', 1)[0],
               window.ROOT / config['sources']['scene'], window.ROOT / config['sources']['script'])
    manifest = window.read_json(arm / 'manifest.json')
    manifest['ue_entry_module'] = {'path': 'Dataset/tools/l4_ue_delivery.py'}
    window.write_json(arm / 'manifest.json', manifest)
    return manifest
