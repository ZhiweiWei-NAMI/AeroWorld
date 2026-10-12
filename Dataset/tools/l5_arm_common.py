"""L5 candidate ARM I/O, identity and evidence utilities. No release inputs are written."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = 'aeroworld_l5_candidate_arm_config'
RUNTIME_FAMILIES = ('control_state', 'mission_state', 'navigation_state', 'sensor_state',
                    'energy_state', 'battery_state', 'payload_state', 'communication_state',
                    'positioning_state')


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path: Path | str) -> list[dict]:
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def dump_rows(path: Path, values: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')


def object_hash(value: Any) -> str:
    return 'sha256:'+hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source(ref: str) -> Path:
    path = ROOT / ref
    if Path(ref).is_absolute() or '..' in Path(ref).parts:
        raise ValueError('source must be a repository-relative path: '+ref)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def states_from_rows(trajectories: list[dict], entities: list[str], end: int,
                     start: int = 0) -> list[dict]:
    index = {(r['tick'],r['entity_id']):r for r in trajectories}
    result=[]
    for tick in range(start, end+1):
        for entity in entities:
            if (tick, entity) not in index:
                result.append({'tick':tick,'entity_id':entity,'observation_status':
                    'engine_inactive',
                    'execution_active':False,
                    'missing_source_record':['raw/trajectories.jsonl'],
                    'state':None,'activity_type':None,'runtime_state':{}})
                continue
            row=index[(tick,entity)]
            result.append({'tick':tick,'entity_id':entity,'observation_status':'observed',
                'execution_active':True,
                'state':row['state'],'activity_type':row['activity_type'],
                'pos_enu':row['pos_enu'],'vel_mps':row['vel_mps'],
                'runtime_state':{k:copy.deepcopy(row[k]) for k in RUNTIME_FAMILIES if k in row},
                'unavailable_runtime_families':[k for k in RUNTIME_FAMILIES if k not in row],
                'source_ref':f'raw/trajectories.jsonl#tick={tick}&entity={entity}'})
    return result


def load_config(path: Path) -> dict:
    import yaml
    from jsonschema import Draft202012Validator
    c=yaml.safe_load(path.read_text())
    Draft202012Validator(read_json(source('rules/l5_arm_config.schema.json'))).validate(c)
    profile,seed,group=c['group_key'].split('/')
    if not re.fullmatch('grp_[a-z0-9_]+',group): raise ValueError('invalid group')
    if c['episode_id']!=profile+'__'+seed: raise ValueError('seed/profile identity mismatch')
    if path.resolve()!=ROOT/'interventions'/c['group_key']/(c['arm_id']+'.yaml'):
        raise ValueError('config path mismatch')
    if c['output']!='arms/'+c['group_key']+'/'+c['arm_id']:
        raise ValueError('output identity mismatch')
    if c['script']!='scripts/arms/'+c['group_key']+'/'+c['arm_id']+'.py':
        raise ValueError('entry identity mismatch')
    w=c['window']
    if not 0<=w['start_tick']<w['fork_tick']<w['end_tick']<=900 or any(
            w[k]%5 for k in ('start_tick','fork_tick','end_tick')):
        raise ValueError('invalid closed formal-grid window: [start,fork,end] must be on the 5-tick grid, '
                         '0 <= start < fork < end <= 900')
    if c['mode']!='intervention' and (c['execution']['actions'] or c['execution']['omit_action_ids']):
        raise ValueError('control contains intervention')
    ids=[s['action']['action_id'] for s in c['execution']['actions']]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate action IDs')
    for step in c['execution']['actions']:
        if not w['fork_tick']<=step['tick']<w['end_tick']: raise ValueError('action outside window')
    return c
