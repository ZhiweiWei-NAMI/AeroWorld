"""Group-level comparison for an L4 incident-fragment group.

Reads the group's own arm products and states, in the group's own terms:
  - each intervention arm is compared against factual on the incident evidence
  - the target is reported as observed / absent / blocked_non_executable_family, never invented
Writes arms/<group>/comparison.json.  Read-only over the arm products.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Dataset.tools import l4_chain_truth as chain  # noqa: E402


def _rows(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def arm_evidence(arm_dir: Path) -> dict[str, Any]:
    manifest = arm_dir / 'manifest.json'
    doc = json.loads(manifest.read_text())
    status = json.loads((arm_dir / 'objective/window_status.json').read_text())
    occurrences = list(_rows(arm_dir / 'objective/event_occurrences.jsonl'))
    outcomes = list(_rows(arm_dir / 'objective/event_outcomes.jsonl'))
    trajectories = list(_rows(arm_dir / 'raw/trajectories.jsonl'))
    actions = list(_rows(arm_dir / 'raw/actions.jsonl'))
    family = doc['event_family']
    window = doc['window']
    return {
        'arm_id': doc['arm_id'], 'mode': doc['mode'], 'event_family': family,
        'window': window, 'status': status['status'],
        'target_assessment': status.get('target_assessment'),
        'local_business_observation': status.get('local_business_observation'),
        'non_executable_family_reason': status.get('non_executable_family_reason'),
        'target': sorted((r['trigger_tick'], r['event_id'])
                         for r in occurrences if r['event_family_id'] == family),
        'terminal': sorted((r['terminal_tick'], r['event_id']) for r in outcomes
                           if r.get('terminal_tick') is not None),
        'injected_actions': [r for r in actions if r.get('origin') == 'intervention'],
        'omitted_actions': [r for r in actions
                            if r['result'].get('status') == 'omitted_by_explicit_intervention'],
        'trajectories': trajectories,
    }


def motion(trajectories, entity_ids, fork) -> dict[str, Any]:
    """Per-subject motion fingerprint after the fork: displacement, peak speed, motion span."""
    by: dict[str, dict[int, dict]] = {}
    for r in trajectories:
        if r['entity_id'] in entity_ids:
            by.setdefault(r['entity_id'], {})[r['tick']] = r
    out = {}
    for e, ticks in by.items():
        marks = sorted(ticks)
        base = ticks.get(fork) or (ticks[marks[0]] if marks else None)
        if base is None:
            continue
        anchor = fork if fork in ticks else marks[0]
        later = [t for t in marks if t > anchor]
        if not later:
            continue
        last = ticks[later[-1]]
        moving = [t for t in later if any(abs(v) > 1e-9 for v in ticks[t]['vel_mps'])]
        out[e] = {
            'spawned_at_or_after_fork': fork not in ticks,
            'dxy_m': round(((last['pos_enu'][0]-base['pos_enu'][0])**2
                            + (last['pos_enu'][1]-base['pos_enu'][1])**2) ** 0.5, 4),
            'dz_m': round(last['pos_enu'][2]-base['pos_enu'][2], 4),
            'vmax_after_mps': round(max((sum(v*v for v in ticks[t]['vel_mps'])) ** 0.5
                                        for t in later), 4),
            'first_motion_tick': moving[0] if moving else None,
            'last_motion_tick': moving[-1] if moving else None,
            'stopped_after_fork': not moving,
        }
    return out


def typed_states(trajectories: list[dict[str, Any]], entity_ids: list[str]) -> dict[str, set]:
    """Per-subject runtime-state fingerprint over the window: the observed typed values.

    A state rewrite leaves motion untouched while changing what the engine recorded for
    the declared runtime-state families; that difference is part of the measured result.
    """
    families = ('control_state', 'mission_state', 'navigation_state', 'security_state',
                'sensor_state', 'facility_state', 'communication_state', 'incident_state',
                'vehicle_state', 'pedestrian_state')
    out: dict[str, set] = {}
    for r in trajectories:
        e = r['entity_id']
        if e not in entity_ids:
            continue
        for fam in families:
            fam_val = r.get(fam)
            if isinstance(fam_val, dict):
                for key, value in sorted(fam_val.items()):
                    out.setdefault(e, set()).add(f'{fam}.{key}={value}')
    return {e: sorted(v) for e, v in out.items()}


def build(group_dir: Path) -> dict[str, Any]:
    profile, seed, _ = group_dir.relative_to(ROOT / 'arms').parts
    arms = {}
    for arm_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        ev = arm_evidence(arm_dir)
        arms[ev['arm_id']] = ev
    factual = arms['arm_factual']
    if factual['mode'] != 'factual_observation':
        raise ValueError('arm_factual must contain factual observations')
    problems: list[str] = []
    entity_ids = sorted({r['entity_id'] for r in factual['trajectories']})
    fork = factual['window']['fork_tick']
    base_motion = motion(factual['trajectories'], entity_ids, fork)
    base_states = typed_states(factual['trajectories'], entity_ids)
    per_arm = {}
    for arm_id, ev in arms.items():
        factual_prefix = [r for r in factual['trajectories'] if r['tick'] <= fork]
        arm_prefix = [r for r in ev['trajectories'] if r['tick'] <= fork]
        prefix_equal = factual_prefix == arm_prefix
        factual_fork = json.loads((group_dir / 'arm_factual/raw/fork_state.json').read_text())
        arm_fork = json.loads((group_dir / arm_id / 'raw/fork_state.json').read_text())
        fork_equal = factual_fork == arm_fork
        if not prefix_equal or not fork_equal:
            problems.append(f'{arm_id}: pre-intervention states or fork differ from factual')
        m = motion(ev['trajectories'], entity_ids, fork)
        changed = {}
        for e, b in base_motion.items():
            a = m.get(e)
            if a is None:
                continue
            if (a['dxy_m'], a['dz_m'], a['first_motion_tick'], a['last_motion_tick'],
                    a['vmax_after_mps']) != (b['dxy_m'], b['dz_m'], b['first_motion_tick'],
                                             b['last_motion_tick'], b['vmax_after_mps']):
                changed[e] = {'factual': b, 'arm': a}
        arm_states = typed_states(ev['trajectories'], entity_ids)
        state_changed = {e: {'factual': base_states.get(e), 'arm': arm_states.get(e)}
                         for e in entity_ids
                         if base_states.get(e) != arm_states.get(e)}
        per_arm[arm_id] = {
            'mode': ev['mode'], 'window_status': ev['status'],
            'target_assessment': ev['target_assessment'],
            'local_business_observation': ev['local_business_observation'],
            'prefix_equal_to_factual': prefix_equal,
            'fork_equal_to_factual': fork_equal,
            'target_ticks': ev['target'], 'terminal_ticks': ev['terminal'],
            'injected_action_count': len(ev['injected_actions']),
            'omitted_action_ids': sorted(r['action']['action_id'] for r in ev['omitted_actions']),
            'motion_vs_factual': changed,
            'typed_state_vs_factual': state_changed,
            # A state rewrite can leave motion and the trigger tick untouched while changing
            # whether/when the event terminates (e.g. an arrival declaration removed): the
            # measured window status and terminal record are part of the difference.
            'differentiated': bool(changed) or bool(state_changed)
                              or ev['target'] != factual['target']
                              or ev['status'] != factual['status']
                              or ev['terminal'] != factual['terminal'],
        }
    report = {
        'group_key': str(group_dir.relative_to(ROOT / 'arms')),
        'episode_id': f'{profile}__{seed}',
        'incident_family': factual['event_family'],
        'window': factual['window'],
        'entity_ids': entity_ids,
        'baseline_arm_id': 'arm_factual',
        'arms': per_arm,
        'problems': problems,
        'interpretation': ('per-arm motion and target evidence against the group factual; '
                           'a mechanism is only claimed when the measured fields differ'),
        'certification': {'historical_counterfactual': 'not_claimed',
                          'release': 'blocked_no_ue_no_full_producer'},
    }
    (group_dir / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', action='append', default=[])
    args = parser.parse_args()
    groups = [Path(g) if Path(g).is_absolute() else ROOT / g for g in args.group]
    if not groups:
        groups = sorted({ROOT / 'arms' / c.parent.relative_to(ROOT / 'interventions')
                         for c in (ROOT / 'interventions').glob('L4-*/*/grp_*/*.yaml')})
    bad = 0
    for g in groups:
        r = build(g)
        if r['problems']:
            bad += 1
            print(f"{r['group_key']}: {r['problems']}", file=sys.stderr)
    print(json.dumps({'groups': len(groups), 'with_problems': bad}, ensure_ascii=False))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
