"""Compare measured L6 branches against their shared factual state."""
from __future__ import annotations

from Dataset.tools.l6_v2.common import load, rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_group(group):
    paths=sorted(p.parent for p in group.glob('*/manifest.json'))
    require(bool(paths), 'empty group')
    factual=group/'arm_factual'; fm=load(factual/'manifest.json');fw=fm['window']
    ft={(r['tick'],r['entity_id']):r for r in rows(factual/'raw/trajectories.jsonl')}
    fs=load(factual/'raw/fork_state.json'); summaries={}
    for p in paths:
        m=load(p/'manifest.json');require(m['window']==fw,'group windows differ')
        tr={(r['tick'],r['entity_id']):r for r in rows(p/'raw/trajectories.jsonl')}
        require(tr.keys()==ft.keys(), 'group scope/coverage differ')
        prefix=[k for k in ft if k[0]<=fw['fork_tick']]
        require(all(ft[k]==tr[k] for k in prefix), 'pre-fork trajectory differs')
        require(load(p/'raw/fork_state.json')==fs, 'complete fork state differs')
        summaries[p.name]={'shared_prefix_rows':len(prefix),'complete_fork_equal':True,
                           'changed_rows':sum(ft[k]!=tr[k] for k in ft),
                           'status':m['observed']['window_status'],
                           'physical':load(p/'objective/physical_observations.json')}
    return {'group_key':fm['group_key'],'factual':'arm_factual','window':fw,'arms':summaries,
            'causal_scope':'explicit combined interventions; script dependencies are scheduling evidence'}
