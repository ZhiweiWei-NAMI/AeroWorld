"""当前源脚本实跑；任务巡航独立调度，干预只延迟天气到达。"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l5_arm_common import load_config
from Dataset.tools.l5_arm_branch import (
    write_branch, write_experiment, set_trigger, bind_branch_sources, measured_window)
from Dataset.tools.l5_arm_pipeline import execute_config

CONFIG = ROOT / 'interventions/L5-2_v1/seed00/grp_l5_env_branch_v1/arm_factual.yaml'
AUTHORED_SCRIPT = ROOT / 'Dataset/scenarios/L5_environment/interaction/L5-2_v1/event_script.json'
AUTHORED_SCENE = ROOT / 'Dataset/scenarios/L5_environment/interaction/L5-2_v1/scene_setup.json'
OFFSET_TICKS = 0  # 保留该 seed 原分支的延迟，不按结果调参。

if __name__ == '__main__':
    config = load_config(CONFIG)
    script = json.loads(AUTHORED_SCRIPT.read_text())
    scene = json.loads(AUTHORED_SCENE.read_text())
    onset = scene['weather_profile']['transitions'][0]['tick']
    if OFFSET_TICKS:
        set_trigger(script, 'weather_onset', onset + OFFSET_TICKS)
        for transition in scene['weather_profile']['transitions']:
            transition['tick'] += OFFSET_TICKS
    # 保留天气条件与事件依赖。后继是否触发、从何处开始运动由实际运行决定；
    # 不把响应和着陆改成整套固定 tick，不移动独立任务航段。
    design = ROOT / 'design' / config['group_key']
    paths = write_branch(design, script, scene, name=config['arm_id'])
    window, evidence = measured_window(paths['script'], paths['scene'],
        config['primary_entity_id'], config['window']['fork_tick'], config['episode_id'])
    write_experiment(design, arm_id=config['arm_id'], group_key=config['group_key'],
        episode_id=config['episode_id'], window=window,
        hazard={'onset_tick': onset + OFFSET_TICKS, 'offset_ticks': OFFSET_TICKS,
                'overrides': scene['weather_profile']['transitions'][0]['overrides']},
        kept=[e['event_id'] for e in script['events']], dropped=[], designed=[],
        target_event={'families': config['event_families'],
                      'basis': '预期：天气作用于独立任务进度；实际谓词和事件以本次 objective 为准。'},
        basis=evidence['basis'])
    bind_branch_sources(CONFIG, paths['script'], paths['scene'], window=window)
    execute_config(CONFIG)
