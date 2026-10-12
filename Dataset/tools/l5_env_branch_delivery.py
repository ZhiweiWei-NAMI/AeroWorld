"""Generate group-level comparison.json and review.json for grp_l5_env_branch_v1.

Uses saved formal predicates, events and trajectories against the current factual.
Self-review only; this does not certify UE image capture or full background feedback.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
GROUP = 'grp_l5_env_branch_v1'


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def arm_evidence(arm_dir):
    occurrences = list(rows(arm_dir / 'objective/event_occurrences.jsonl'))
    outcomes = {r['event_id']: r.get('terminal_tick') for r in rows(arm_dir / 'objective/event_outcomes.jsonl')}
    status = json.loads((arm_dir / 'objective/window_status.json').read_text())
    window = status['window']
    trajectories = {(r['tick'], r['entity_id']): r for r in rows(arm_dir / 'raw/trajectories.jsonl')}
    manifest = json.loads((arm_dir / 'manifest.json').read_text())
    return {'events': [{'family': r['event_family_id'], 'onset': r['detection_tick'],
                        'terminal': outcomes.get(r['event_id'])} for r in occurrences],
            'window': window, 'right_censored': status['right_censored'],
            'trajectories': trajectories, 'mode': manifest['mode'],
            'certification': manifest.get('certification', {})}


def build_group(group_dir):
    group_rel = str(group_dir.relative_to(ROOT / 'arms'))
    config_dir = ROOT / 'interventions' / group_rel
    arm_names = sorted(p.stem for p in config_dir.glob('*.yaml'))
    factual = arm_evidence(group_dir / 'arm_factual')

    arm_deltas = {}
    arm_status = {}
    for arm_name in arm_names:
        arm_dir = group_dir / arm_name
        if not arm_dir.is_dir():
            arm_status[arm_name] = 'not_generated'
            continue
        ev = arm_evidence(arm_dir)
        common_t = sorted(set(ev['trajectories']) & set(factual['trajectories']))
        max_diff = round(max(math.dist(ev['trajectories'][k]['pos_enu'], factual['trajectories'][k]['pos_enu'])
                             for k in common_t), 6) if common_t else None
        env_events = sum(1 for e in ev['events'] if e['family'] == 'rain_fog_environment')
        arm_status[arm_name] = 'generated'
        arm_deltas[arm_name] = {
            'mode': ev['mode'],
            'event_count': len(ev['events']),
            'environment_event_count': env_events,
            'max_position_diff_m_vs_factual': max_diff,
            'event_signature_vs_factual': 'identical' if ev['events'] == factual['events'] else 'differs',
            'predicate_transitions': [{'predicate_id':r['predicate_id'], 'bindings':r['bindings'],
                'tick':r['to_tick'], 'value':r['to_value']}
                for r in rows(arm_dir / 'objective/predicate_transitions.jsonl')],
            'right_censored': ev['right_censored'],
        }

    comparison = {
        'group_key': group_rel,
        'window': factual['window'],
        'arm_status': arm_status,
        'arm_deltas': arm_deltas,
        'baseline_arm_id': 'arm_factual',
    }

    # Current measured outcomes; no inherited verdicts for deleted branches.
    arms_review = {}
    for arm_name in arm_names:
        arm_dir = group_dir / arm_name
        if not arm_dir.is_dir():
            continue
        ev = arm_evidence(arm_dir)
        m = json.loads((arm_dir / 'manifest.json').read_text())
        arms_review[arm_name] = {
            'mode': ev['mode'],
            'declared_injected': m.get('provenance', {}).get('parameter_sources', {}).get(
                'branch_environment', ''),
            'event_families': {f: sum(1 for e in ev['events'] if e['family'] == f)
                               for f in sorted({e['family'] for e in ev['events']})},
            'outcomes': [[e['family'], e['onset'], e['terminal']] for e in ev['events']],
        }

    review = {
        'group_key': group_rel,
        'review_type': 'self_review',
        'independent_reviewer': False,
        'arms': arms_review,
        'controls': {
            'factual_observation': 'saved factual window',
        },
        'target_families': ['rain_fog_environment', 'altitude_excursion', 'uav_landing', 'uav_takeoff'],
        'anchor_event': 'weather_onset',
        'formal_certification': 'UE capture and full background feedback are outside this local execution',
        'claims': {
            'scope': 'same current-engine factual source; nominal task schedule independent of weather delay',
            'predicate_authority': 'objective/predicate_truth.jsonl and objective/predicate_transitions.jsonl',
            'physical_scope': 'kinematic trajectories and explicit runtime states; no aerodynamic or camera-visibility measurement',
            'factual_only': len(arm_names) == 1,
        },
        'outside_current_delivery': [
            'full producer feedback (SUMO, global UAV flow)',
            'UE RGB/depth/segmentation capture',
        ],
        'release': 'local_ue_inputs_exported_capture_not_executed',
    }
    return comparison, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    groups = sorted((ROOT / 'arms').glob(f'L5-*/*/{GROUP}'))
    made = 0
    for g in groups:
        group_key = g.relative_to(ROOT / 'arms').as_posix()
        definition_path = ROOT / 'design' / group_key / 'experiment.json'
        definition = json.loads(definition_path.read_text())
        current_arms = {}
        for config_path in sorted((ROOT / 'interventions' / group_key).glob('*.yaml')):
            config = yaml.safe_load(config_path.read_text())
            ref = config['provenance']['experiment_ref']
            experiment = json.loads((ROOT / ref).read_text())
            current_arms[config['arm_id']] = {'experiment_ref': ref, 'window': experiment['window']}
        definition['arms'] = current_arms
        definition['common_observation_window'] = {
            'start_tick': max(arm['window']['start_tick'] for arm in current_arms.values()),
            'end_tick': min(arm['window']['end_tick'] for arm in current_arms.values()),
        }
        definition['shared_checkpoint_tick'] = current_arms['arm_factual']['window']['fork_tick']
        comparison, review = build_group(g)
        cp = g / 'comparison.json'
        rp = g / 'review.json'
        if not args.dry_run:
            definition_path.write_text(json.dumps(definition, ensure_ascii=False, indent=2) + '\n')
            cp.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + '\n')
            rp.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
        made += 1
    print(f'generated comparison.json + review.json for {made} groups')


if __name__ == '__main__':
    main()
