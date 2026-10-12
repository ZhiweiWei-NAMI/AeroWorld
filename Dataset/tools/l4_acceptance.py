"""Acceptance for the L4 incident-fragment arm set, A01-A22 + the evidence checks this round needs.

Every item reads this round's own products.  Items that cannot be decided from a product are
reported not_run with the reason, never as a pass.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Dataset.tools import l4_chain_truth as chain  # noqa: E402
from Dataset.semantic_truth.window_outcome import (  # noqa: E402
    validate_arm_window_outcome,
    validate_l4_business_outcome,
)

def rows(p: Path):
    with p.open() as f:
        for l in f:
            if l.strip():
                yield json.loads(l)
def my_groups() -> list[Path]:
    out = []
    for c in (ROOT/'interventions').glob('L4-*/*/grp_*/*.yaml'):
        out.append(ROOT/'arms'/c.parent.relative_to(ROOT/'interventions'))
    return sorted(set(out))

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--json',action='store_true')
    a=ap.parse_args()
    groups=my_groups()
    res={k:{'pass':0,'fail':0,'not_run':0} for k in ('A04','A07','A09','A12','A14','A18','A19','A22')}
    problems=[]; evidence=[]
    for g in groups:
        comp=g/'comparison.json'
        if not comp.is_file():
            res['A04']['fail'] += 1
            problems.append(f'{g}: group comparison is missing')
            continue
        d=json.loads(comp.read_text())
        arms=d['arms']
        # A04 inventory
        declared={c.stem for c in (ROOT/'interventions'/g.relative_to(ROOT/'arms')).glob('*.yaml')}
        res['A04']['pass' if set(arms)==declared and declared else 'fail']+=1
        if set(arms)!=declared: problems.append(f'{g}: inventory mismatch')
        # A07 shared fork window across the group
        # A07: every arm of the group must declare the same closed window.  Read it from each
        # arm's own manifest (the executed record), not from the comparison summary.
        wins={tuple(json.loads((g/arm_id/'manifest.json').read_text())['window'][k]
                    for k in ('start_tick','fork_tick','end_tick'))
              for arm_id in arms}
        factual_states = list(rows(g/'arm_factual/raw/states.jsonl'))
        factual_fork = json.loads((g/'arm_factual/raw/fork_state.json').read_text())
        fork = factual_fork['tick']
        prefix = [r for r in factual_states if r['tick'] <= fork]
        shared = len(wins) == 1 and all(
            [r for r in rows(g/arm_id/'raw/states.jsonl') if r['tick'] <= fork] == prefix
            and json.loads((g/arm_id/'raw/fork_state.json').read_text()) == factual_fork
            for arm_id in arms)
        res['A07']['pass' if shared else 'fail']+=1
        if not shared: problems.append(f'{g}: full prefix or fork state differs from factual')
        if len(wins)!=1: problems.append(f'{g}: group arms disagree on the window: {sorted(wins)}')
        # A09 window completeness: every declared (tick, entity) inside the window is present, with
        # no duplicates, and the missing ones are exactly the recorded lifecycle absences.
        a09_ok=True
        for arm_id in arms:
            path=g/arm_id/'raw/trajectories.jsonl'
            if not path.is_file(): a09_ok=False; break
            got=[(r['tick'],r['entity_id']) for r in rows(path)]
            w=json.loads((g/arm_id/'manifest.json').read_text())['window']
            want={(t,e) for t in range(w['start_tick'],w['end_tick']+1)
                  for e in json.loads((g/arm_id/'manifest.json').read_text())['scope']['entity_ids']}
            if len(got)!=len(set(got)) or not set(got)<=want:
                a09_ok=False; break
            absent=want-set(got)
            life=g/arm_id/'raw/lifecycle_absent.jsonl'
            declared={(r['tick'],r['entity_id']) for r in rows(life)} if life.is_file() else set()
            if absent!=declared:
                a09_ok=False; break
        res['A09']['pass' if a09_ok else 'fail']+=1
        if not a09_ok: problems.append(f'{g}: window state is incomplete or duplicated without a '
                                       'recorded lifecycle reason')
        # A12: a censored event cannot claim successful completion.  Scoped to the arm's
        # declared target family: adjacent families caught by the shared window (e.g. a
        # uav_landing completing a few ticks past the boundary) are recorded with their own
        # censoring flags and are not this arm's success claim.
        family = d['incident_family']
        def outcome_within_window(arm_id, row):
            window = json.loads((g/arm_id/'manifest.json').read_text())['window']
            start_tick, end_tick = window['start_tick'], window['end_tick']
            try:
                if row.get('schema_name') == 'arm_window_event_outcome':
                    validate_arm_window_outcome(row, start_tick=start_tick, end_tick=end_tick)
                elif row.get('schema_name') == 'l4_business_event_outcome':
                    validate_l4_business_outcome(row, start_tick=start_tick, end_tick=end_tick)
                else:
                    return False
            except ValueError:
                return False
            if row.get('event_family_id') != family:
                return True
            status, tick = row.get('status'), row.get('terminal_tick')
            if status == 'right_censored':
                return tick is None and row.get('window_censor_tick') == end_tick
            if status == 'succeeded':
                return (type(tick) is int and start_tick <= tick <= end_tick
                        and 'window_censor_tick' not in row)
            return status == 'pending' and tick is None and 'window_censor_tick' not in row
        valid = all(outcome_within_window(arm_id, r)
                    for arm_id in arms
                    for r in rows(g/arm_id/'objective/event_outcomes.jsonl'))
        res['A12']['pass' if valid else 'fail'] += 1
        if not valid: problems.append(f'{g}: outcome schema, window, or censor status is invalid')
        # A14: required files present and resolvable.
        missing = [arm_id for arm_id in arms if not (g/arm_id/'manifest.json').is_file()]
        required_missing = [f'{arm_id}/{rel}' for arm_id in arms
                            for rel in ('raw/initial_state.json','raw/fork_state.json','raw/states.jsonl',
                                        'raw/actions.jsonl','objective/event_outcomes.jsonl','ue/truth_frames.jsonl')
                            if not (g/arm_id/rel).is_file()]
        res['A14']['pass' if not (missing or required_missing) else 'fail']+=1
        if missing:
            problems.append(f'{g}: manifest missing: {sorted(set(missing))}')
        if required_missing:
            problems.append(f'{g}: required files missing: {sorted(set(required_missing))}')
        # A18 mechanism: declared == dispatched
        for arm_id,v in arms.items():
            cfg_path=ROOT/'interventions'/g.relative_to(ROOT/'arms')/(arm_id+'.yaml')
            import yaml as _yaml
            execution = _yaml.safe_load(cfg_path.read_text())['execution']
            decl = execution['actions']
            acts=list(rows(g/arm_id/'raw/actions.jsonl'))
            inj=[r for r in acts if r.get('origin')=='intervention']
            observed = [{'tick': r['tick'], 'action': r['action']} for r in inj]
            omitted = {r['action']['action_id'] for r in acts
                       if r['result']['status'] == 'omitted_by_explicit_intervention'}
            good = (decl == observed and all(r['result']['status'] == 'ok' for r in inj)
                    and omitted == set(execution['omit_action_ids']))
            res['A18']['pass' if good else 'fail']+=1
            if not good: problems.append(f'{g}/{arm_id}: declared and dispatched actions differ')
        # A19: measured difference is required, while business success is assessed separately.
        for arm_id, v in arms.items():
            if arm_id == 'arm_factual':
                continue
            res['A19']['pass' if v['differentiated'] else 'fail'] += 1
            if not v['differentiated']:
                problems.append(f'{g}/{arm_id}: no measured effect')
        # A22 anchor is a script event with a dispatch tick
        prof,seed,_=g.relative_to(ROOT/'arms').parts
        f=chain.fragment(prof,seed)
        nd=f.get('fork_tick')
        res['A22']['pass' if isinstance(nd,int) and nd%5==0 else 'fail']+=1
        if not (isinstance(nd,int) and nd%5==0): problems.append(f'{g}: anchor has no 5-grid dispatch tick')
    report={'groups':len(groups),'checks':res,'problems':problems[:40],
            'problem_count':len(problems),
            'does_not_prove':['release certification','UE capture','hypothesis truth beyond the measured comparison']}
    if a.json: print(json.dumps(report,ensure_ascii=False,indent=1))
    else:
        print(json.dumps({'groups':len(groups),'checks':res,'problems':len(problems)},ensure_ascii=False))
        for p in problems[:20]: print('  !',p)
    return 1 if problems else 0

if __name__=='__main__':
    raise SystemExit(main())
