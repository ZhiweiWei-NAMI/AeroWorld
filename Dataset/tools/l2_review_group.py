"""组级审查记录：把机制核验与假设判定写成每个组的 review.json。"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify(group):
    base = ROOT / 'arms' / group
    configs = {Path(p).stem: yaml.safe_load(Path(p).read_text())
               for p in sorted((ROOT / 'interventions' / group).glob('*.yaml'))}
    arms = {}
    for arm_id, config in configs.items():
        out = base / arm_id
        if not (out / 'manifest.json').is_file():
            arms[arm_id] = {'present': False}
            continue
        status = json.loads((out / 'objective/window_status.json').read_text())
        acts = rows(out / 'raw/actions.jsonl')
        arms[arm_id] = {
            'present': True, 'mode': config['mode'],
            'window_status': status['status'],
            'event_families': status['event_families'],
            'outcomes': [(o['family'], o['status'], o['terminal_tick']) for o in status['outcomes']],
            'injected': sorted({a['action']['action_id'] for a in acts if a['origin'] == 'intervention'}),
            'omitted': sorted({a['action']['action_id'] for a in acts
                               if a['result']['status'].startswith('omitted')}),
            'declared_injected': sorted(s['action']['action_id'] for s in config['execution']['actions']),
            'declared_omitted': sorted(config['execution']['omit_action_ids']),
        }
    checks = {}
    for arm_id, rec in arms.items():
        if not rec.get('present') or rec['mode'] != 'intervention':
            continue
        checks[arm_id] = {
            'A18_injected_actions_match_config': rec['injected'] == rec['declared_injected'],
            'A18_omitted_actions_match_config': rec['omitted'] == rec['declared_omitted'],
        }
    offset = {}
    for arm_id, config in configs.items():
        if 'offset_path' not in arm_id:
            continue
        inj = config['execution']['actions'][0]['action']
        ent = inj['entity_id']
        uavs = [e for e in config['entity_ids'] if e.startswith('uav_')]
        other = [e for e in uavs if e != ent][0]
        fork, end = config['window']['fork_tick'], config['window']['end_tick']

        def series(arm):
            out = {}
            for r in rows(base / arm / 'raw/trajectories.jsonl'):
                out.setdefault(r['tick'], {})[r['entity_id']] = r['pos_enu']
            return out

        ctl, arm = series('arm_factual'), series(arm_id)
        xy = max(math.dist(ctl[t][ent][:2], arm[t][ent][:2]) for t in range(fork, end + 1)
                 if t in ctl and t in arm)
        dz = max(abs(ctl[t][ent][2] - arm[t][ent][2]) for t in range(fork, end + 1)
                 if t in ctl and t in arm)
        sep_arm = any(o['family'] == 'uav_separation_conflict'
                      for o in json.loads((base / arm_id / 'objective/window_status.json')
                                          .read_text())['outcomes'])
        sep_ctl = any(o['family'] == 'uav_separation_conflict'
                      for o in json.loads((base / 'arm_factual/objective/window_status.json')
                                          .read_text())['outcomes'])
        # 检查项的值一律“True = 符合设计期望”；原始观测另列在 observations。
        checks[arm_id].update({
            'A19_displacement_is_real': xy >= 25.0 or dz >= 8.0,
            'A19_target_event_removed_as_designed': (sep_ctl is True and sep_arm is False),
        })
        observations = {
            'A19_max_xy_offset_vs_factual_m': round(xy, 3),
            'A19_max_z_offset_vs_factual_m': round(dz, 3),
            'A19_separation_event_in_factual': sep_ctl,
            'A19_separation_event_in_arm': sep_arm,
        }
        offset[arm_id] = observations
    return {'group_key': group,
            'window': configs['arm_factual']['window'],
            'anchor_event': configs['arm_factual']['anchor_event'],
            'target_families': configs['arm_factual']['target_families'],
            'controls': {'factual_observation': 'current engine, no intervention, original process window',
                         },
            'arms': arms, 'mechanism_checks': checks,
            'mechanism_observations': offset,
            'A20_suppression': ('省略动作与注入动作按当前配置逐项对照；动作记录不等于机制成功，'
                                '结果以本次轨迹、运行时状态、正式谓词和事件 outcome 为准。'
                                'pending 保留为窗口右删失，不按预期补成恢复。'),
            'validation_scope': ('现有组级工具读取当前窗口产物，记录动作及正式事件实测；'
                                 '这是生成后的内容自查，不是独立审查或全族重跑。'
                                 'UE 输入包已生成，尚未执行 UE 图像采集。')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True)
    args = parser.parse_args()
    report = verify(args.group)
    path = ROOT / 'arms' / args.group / 'review.json'
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'group': args.group, 'arms': len(report['arms']),
                      'checks': len(report['mechanism_checks'])}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
