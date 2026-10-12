"""Compare each L5 environment intervention with the saved factual window."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUP = 'grp_l5_env_branch_v1'

def rows(path):
    p = Path(path)
    with p.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def arm_summary(arm_dir):
    occurrences = rows(arm_dir / 'objective/event_occurrences.jsonl')
    outcomes = {r['event_id']: r.get('terminal_tick') for r in rows(arm_dir / 'objective/event_outcomes.jsonl')}
    status = json.loads((arm_dir / 'objective/window_status.json').read_text())
    trajectories = {(r['tick'], r['entity_id']): r for r in rows(arm_dir / 'raw/trajectories.jsonl')}
    return {'events': [{'family': r['event_family_id'], 'onset': r['detection_tick'],
                        'terminal': outcomes.get(r['event_id'])} for r in occurrences],
            'event_count': len(occurrences), 'window': status['window'],
            'pending_event_ids': status['pending_event_ids'],
            'right_censored_event_ids': status['right_censored_event_ids'],
            'trajectories': trajectories,
            'transitions': [{'predicate_id':r['predicate_id'],'bindings':r['bindings'],
                'tick':r['to_tick'],'value':r['to_value']}
                for r in rows(arm_dir / 'objective/predicate_transitions.jsonl')]}


def compare_group(group_dir):
    control = arm_summary(group_dir / 'arm_factual')
    report = {'group': str(group_dir.relative_to(ROOT / 'arms')), 'window': control['window'],
              'arms': {}, 'baseline_arm_id': 'arm_factual'}
    for d in sorted(p.parent for p in group_dir.glob('*/manifest.json')):
        arm = d.name
        s = arm_summary(d)
        common = sorted(set(s['trajectories']) & set(control['trajectories']))
        distances = [math.dist(s['trajectories'][k]['pos_enu'], control['trajectories'][k]['pos_enu'])
                     for k in common]
        report['arms'][arm] = {
            'status': 'ok', 'event_count': s['event_count'], 'events': s['events'],
            'pending_event_ids': s['pending_event_ids'],
            'right_censored_event_ids': s['right_censored_event_ids'],
            'trajectory_ticks_common': len(common),
            'max_position_difference_m_vs_factual': round(max(distances), 6) if distances else None,
            'predicate_transitions': s['transitions'],
            'event_signature_vs_factual': 'identical' if s['events'] == control['events'] else 'differs'}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', action='append', default=[])
    args = parser.parse_args()
    groups = args.group or [str(p.relative_to(ROOT / 'arms'))
                            for p in sorted((ROOT / 'arms').glob(f'L5-*/*/{GROUP}'))]
    reports = [compare_group(ROOT / 'arms' / g) for g in groups]
    print(json.dumps({'groups': len(reports), 'reports': reports}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
