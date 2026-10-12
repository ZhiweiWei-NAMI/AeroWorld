"""组级对比：factual 原始过程片段 / 各干预臂的实测差异与假设判定。

只读窗口产物；结论由真实运行的事件、谓词与轨迹算出，expected 只作假设对照。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _directory in (ROOT, ROOT / 'Dataset/tools'):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import yaml  # noqa: E402
from Dataset.tools.l2_window import load, stream_rows, write_json  # noqa: E402


def arm_rows(group, arm_id):
    base = ROOT / 'arms' / group / arm_id
    if not (base / 'manifest.json').is_file():
        return None
    status = load(base / 'objective/window_status.json')
    occurrences = list(stream_rows(base / 'objective/event_occurrences.jsonl'))
    outcomes = list(stream_rows(base / 'objective/event_outcomes.jsonl'))
    config = yaml.safe_load((ROOT / 'interventions' / group / f'{arm_id}.yaml').read_text())
    tracks = list(stream_rows(base / 'raw/trajectories.jsonl'))
    return {'arm_id': arm_id, 'mode': config['mode'],
            'expected': config['expected'], 'window': config['window'],
            'status': status, 'occurrences': occurrences, 'outcomes': outcomes,
            'trajectories': tracks, 'omit': config['execution']['omit_action_ids'],
            'actions': [s['action']['action_id'] for s in config['execution']['actions']]}


def compare(rows, fork):
    out = {}
    for name, arm in rows.items():
        out[name] = sorted((o['event_family_id'], o['trigger_tick'],
                            next((x['status'] for x in arm['outcomes']
                                  if x['event_id'] == o['event_id']), None),
                            next((x['terminal_tick'] for x in arm['outcomes']
                                  if x['event_id'] == o['event_id']), None))
                           for o in arm['occurrences'])
    return out


def fork_digest(arm):
    return json.dumps(arm['trajectories'], sort_keys=True)


def build_group(group):
    arm_ids = sorted(p.stem for p in (ROOT / 'interventions' / group).glob('*.yaml'))
    rows = {}
    for arm_id in arm_ids:
        rec = arm_rows(group, arm_id)
        if rec:
            rows[arm_id] = rec
    if 'arm_factual' not in rows:
        raise ValueError('group lacks factual baseline')
    intervention_ids = [a for a in rows if rows[a]['mode'] == 'intervention']
    if not intervention_ids:
        raise ValueError('group has no intervention arms')

    deltas = {}
    for arm_id in intervention_ids:
        arm = rows[arm_id]
        deltas[arm_id] = {
            'expected_target_event': arm['expected']['target_event'],
            'evidence': compare({'factual': rows['arm_factual'], 'arm': arm}, 0),
            'omitted_actions': arm['omit'],
            'injected_actions': arm['actions'],
        }
    report = {'group_key': group, 'window': rows['arm_factual']['window'],
              'baseline_arm_id': 'arm_factual',
              'arm_deltas': deltas,
              'arm_status': {a: rows[a]['status']['status'] for a in rows}}
    write_json(ROOT / 'arms' / group / 'comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()
    report = build_group(args.group)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
