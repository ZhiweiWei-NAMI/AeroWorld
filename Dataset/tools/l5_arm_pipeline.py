"""Generate immutable L5 window assets and measured group contrasts."""
from __future__ import annotations
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from Dataset.tools.l5_arm_common import (
    ROOT, RUNTIME_FAMILIES, read_json, rows, dump_json, dump_rows,
    source, relative, load_config,
    states_from_rows)
from Dataset.tools.l5_arm_engine import execute
from Dataset.tools.l5_arm_semantics import calculate


def action_effects(config, audit, trajectories, owners, weather):
    scene=read_json(source(config['sources']['scene']))
    attached={e['entity_id'] for e in scene['entities'] if 'attachment' in e.get('motion_contract',{})}
    # The declared window bounds the experiment: an action dispatched before the
    # window start is pre-window history and must not be re-attributed inside it.
    start=config['window']['start_tick']; end=config['window']['end_tick']
    audit=[a for a in audit if start<=a['tick']<=end]
    owners=[r for r in owners if start<=r['tick']<=end]
    tr={(r['tick'],r['entity_id']):r for r in trajectories}; wx={r['tick']:r for r in weather}
    motion=[]; weather_checks=[]; runtime_checks=[]
    for a in audit:
        action=a['action']; kind=action['type']; aid=action['action_id']; tick=a['tick']
        if kind=='set_weather' and a['origin']=='intervention':
            if tick+1 not in wx: raise ValueError('weather evidence sample outside window')
            actual=wx[tick+1]
            requested={**action['overrides'],'condition':action['profile']}
            weather_checks.append({'action_id':aid,'dispatch_tick':tick,'first_sample_tick':tick+1,
                'matches_at_next_sample':all(actual[k]==v for k,v in requested.items()),
                'requested':requested,'observed':{k:actual[k] for k in requested},
                'evidence_refs':[f'raw/actions.jsonl#sequence={a["sequence"]}',f'raw/weather.jsonl#tick={tick+1}']})
        if kind=='set_runtime_state' and action['entity_id'] in config['entity_ids'] and a['result']['status']=='ok':
            effective=a['result']['effective_tick']; entity=action['entity_id']; checks=[]
            for family,patch in action['state_patch'].items():
                for key,value in patch.items():
                    record=tr[(effective,entity)] if (effective,entity) in tr else {}
                    observed=record[family][key] if family in record and key in record[family] else None
                    checks.append({'field':family+'.'+key,'requested':value,'observed':observed,'matches':observed==value})
            runtime_checks.append({'action_id':aid,'dispatch_tick':tick,'effective_tick':effective,'checks':checks,
                'evidence_ref':f'raw/trajectories.jsonl#tick={effective}&entity={entity}'})
        if kind!='move_entity' or action['entity_id'] not in config['entity_ids']: continue
        entity=action['entity_id']
        owned=[r for r in owners if r['entity_id']==entity and r['motion_owner'] is not None
               and r['motion_owner']['action_id']==aid and r['tick']>tick]
        if a['result']['status']=='omitted_by_explicit_intervention':
            if owned or a['effect']['keyframes_changed']: raise ValueError('omitted action owns motion')
            motion.append({'action_id':aid,'entity_id':entity,'dispatch_tick':tick,
                'status':'explicitly_omitted_no_owned_motion','first_owned_motion_tick':None,
                'evidence_ref':f'raw/actions.jsonl#sequence={a["sequence"]}'})
            continue
        model=a['effect']['motion_model']; nominal=model['nominal_completion_tick']
        if entity in attached:
            motion.append({'action_id':aid,'entity_id':entity,'dispatch_tick':tick,
                'status':'relative_swing_profile_completed' if nominal<=end else 'right_censored',
                'nominal_completion_tick':nominal,'model':model,
                'interpretation':'Attached payload follows its parent; source waypoints prescribe tether direction, not world arrival.',
                'evidence_refs':['raw/trajectories.jsonl',config['sources']['scene']]})
            continue
        terminal=a['effect']['added_schedule']['waypoints_enu_m'][-1]
        moving=[]; completed=[]
        for r in owned:
            t=r['tick']; current=tr[(t,entity)]; previous=tr[(t-1,entity)] if (t-1,entity) in tr else None
            if r['speed_mps']>1e-6 and previous and math.dist(current['pos_enu'],previous['pos_enu'])>1e-6:
                moving.append(t)
            if t>=nominal+1 and r['speed_mps']<=1e-6 and math.dist(current['pos_enu'],terminal)<=2e-6:
                completed.append(t)
        following=[r for r in owners if r['entity_id']==entity and r['tick']>tick and r['motion_owner'] is not None
                   and r['motion_owner']['action_id']!=aid and r['motion_owner']['dispatch_tick']>=tick]
        next_owner=following[0]['motion_owner'] if following else None
        status='observed_completed' if completed else 'preempted' if next_owner else 'right_censored'
        motion.append({'action_id':aid,'entity_id':entity,'dispatch_tick':tick,
            'scheduled_tick':a['result']['scheduled_tick'],'nominal_completion_tick':nominal,
            'first_owned_motion_tick':moving[0] if moving else None,
            'last_owned_motion_tick':moving[-1] if moving else None,
            'completion_stationary_sample_tick':completed[0] if completed else None,
            'status':status,'next_motion_owner':next_owner,'model':model,
            'accepted_without_owned_movement':not moving,
            'evidence_refs':[f'raw/actions.jsonl#sequence={a["sequence"]}',
                            'raw/motion_attribution.jsonl', 'raw/trajectories.jsonl']})
    return {'motion_commands':motion,'weather_interventions':weather_checks,'runtime_state_patches':runtime_checks,
        'configured_omissions':config['execution']['omit_action_ids'],
        'observed_omissions':[a['action']['action_id'] for a in audit if a['result']['status']=='omitted_by_explicit_intervention'],
        'right_censored_motion_ids':[m['action_id'] for m in motion if m['status']=='right_censored'],
        'interpretation':'accepted is dispatch only; observed movement is assigned through actual schedule ownership, not the next action in time'}


def execute_config(config_path: Path):
    """Run one ARM.

    The ARM's entry script is responsible for its own branch: it builds the
    script and scene that define this branch and passes their paths through the
    configuration's ``sources``.  This function only executes what the
    configuration declares and writes the resulting artifacts; it makes no
    branch decisions of its own.
    """
    config_path = Path(config_path).absolute()
    c=load_config(config_path); out=ROOT/c['output']
    if out.exists(): raise FileExistsError('refusing to overwrite '+str(out))
    started=datetime.now(timezone.utc).isoformat(); random.seed(c['seed'])
    end=c['window']['end_tick']; fork=c['window']['fork_tick']
    start=c['window']['start_tick']
    engine,interpreter,audit,snaps,owners=execute(c)
    trajectories=[r for r in engine.trajectory_rows if r['entity_id'] in c['entity_ids'] and start<=r['tick']<=end]
    weather=[r for r in engine.weather_rows if start<=r['tick']<=end]
    effects=action_effects(c,audit,trajectories,owners,weather)
    for tick in (start,fork,end):
        sampled={r['entity_id']:r for r in snaps[tick]['sampled_states']}
        for r in trajectories:
            if r['tick']==tick and sampled[r['entity_id']]!=r: raise ValueError('snapshot/trajectory disagreement')
    semantic,geometry,catalog,mechanisms,unresolved,status,nodes,relations=calculate(c,trajectories,weather)
    status['right_censored_motion_ids']=effects['right_censored_motion_ids']
    status['right_censored']=status['right_censored'] or bool(effects['right_censored_motion_ids'])
    state_rows=states_from_rows(trajectories,c['entity_ids'],end,start=start)
    if not out.exists(): out.mkdir(parents=True,exist_ok=False)
    dump_json(out/'raw/initial_state.json',snaps[start]);dump_json(out/'raw/fork_state.json',snaps[fork])
    dump_json(out/'raw/end_state.json',snaps[end]);dump_rows(out/'raw/actions.jsonl',[r for r in audit if start<=r['tick']<=end])
    dump_rows(out/'raw/states.jsonl',state_rows)
    dump_rows(out/'raw/trajectories.jsonl',trajectories)
    dump_rows(out/'raw/weather.jsonl',weather);dump_rows(out/'raw/motion_attribution.jsonl',[r for r in owners if start<=r['tick']<=end])
    dump_rows(out/'raw/geometry.jsonl',geometry);dump_json(out/'raw/binding_catalog.json',catalog)
    for key,values in semantic.items(): dump_rows(out/'objective'/(key+'.jsonl'),values)
    dump_json(out/'objective/window_status.json',status);dump_json(out/'objective/action_effects.json',effects)
    dump_rows(out/'objective/mechanism_observables.jsonl',mechanisms)
    dump_rows(out/'objective/unresolved_predicates.jsonl',unresolved)
    dump_rows(out/'chain/events.jsonl',nodes);dump_rows(out/'chain/relations.jsonl',relations)
    manifest={'schema':'aeroworld_l5_candidate_window','version':1,'arm_id':c['arm_id'],'arm_key':c['group_key']+'/'+c['arm_id'],
        'episode_id':c['episode_id'],'seed':c['seed'],'mode':c['mode'],'window':c['window'],
        'source_version': c['group_key'].split('/')[-1]+':current_authored_scene_script_execution',
        'source_refs':c['sources'],
        'configuration_ref':relative(config_path),
        'experiment_ref':c['provenance']['experiment_ref'],
        'provenance':provenance_block(c,config_path,start,fork,end),
        'entity_ids':c['entity_ids'],'primary_entity_id':c['primary_entity_id'],'predicate_ids':c['predicate_ids'],
        'blocked_event_families':c['blocked_event_families'],
        'record_counts':{'trajectories':len(trajectories),'states':len(state_rows),'weather':len(weather),
            **{k:len(v) for k,v in semantic.items()},'unresolved_predicate_records':len(unresolved)},
        'declaration':c.get('declaration'),
        'runtime':{'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform(),
            'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
            'random_seed':c['seed'],'primary_engine_rng':'none: seed labels identify independent executions, not varied UAV parameters'},
        'certification':{'local_candidate_asset':'generated','historical_counterfactual':'not_claimed',
                         'full_L0_producer':'outside_declared_local_scope','UE_capture':'not_generated',
                         'release':'local_window_generated'},
        'limitations':['authored-scene actors only; no complete SUMO/global-UAV feedback',
            'local authored-pad geometry only; missing global pad/contact producer',
            'weather control is a simulator-state intervention, not a physical weather-control system',
            'no measured camera pixels, battery current/capacity/temperature, payload swing angle or UAV roll/pitch']}
    dump_json(out/'manifest.json',manifest)
    print(json.dumps({'arm_key':manifest['arm_key'],'mode':c['mode'],'trajectory_rows':len(trajectories),
        'events':status['event_counts'],'right_censored':status['right_censored'],
        'unknown_dense_rows':status['unknown_dense_rows']},ensure_ascii=False),flush=True)
    from Dataset.tools.l5_family_delivery import deliver_config
    deliver_config(config_path, export=True)
    return read_json(out/'manifest.json')


def provenance_block(config, config_path, start, fork, end):
    """Record where the numbers in this ARM came from.

    ``delay_ticks`` is a *deterministic digest*, not a physical derivation: the
    tick offset is sha256(episode_id|group|arm_id) mapped onto the 5..60 grid.
    Recording the digest inputs lets a reader recompute the same offset.
    """
    import hashlib
    block = dict(config.get('provenance') or {})
    sources = dict(block.get('parameter_sources') or {})
    branch = config.get('branch') or {}
    if branch.get('environment'):
        env = branch['environment']
        onset = env.get('onset_tick')
        authored_fork = config['window']['fork_tick']
        offset = None
        if isinstance(onset, int) and onset != authored_fork:
            offset = onset - authored_fork
        sources['branch_environment'] = (
            f"built by {config['script']}: field={env.get('field')} value={env.get('value')} "
            f"onset_tick={onset} hold_hazard_at_value={env.get('hold_hazard_at_value', False)}")
        if offset is not None:
            digest = hashlib.sha256(
                f"{config['episode_id']}|{config['group_key']}|{config['arm_id']}".encode()).digest()
            sources['delay_ticks'] = {
                'value': offset,
                'source': 'deterministic digest, not a physical derivation',
                'rule': 'sha256(episode_id|group|arm_id) -> 5 * (first 8 bytes mod 12 + 1), range 5..60',
                'digest_prefix': digest.hex()[:16],
                'grid': 'multiple of 5, within 5..60',
                'recompute': 'sha256(f"{episode_id}|{group_key}|{arm_id}").digest()[:8] interpreted big-endian',
            }
    block['parameter_sources'] = sources
    block['window'] = {'start_tick': start, 'fork_tick': fork, 'end_tick': end,
                       'basis': ('start = fork - one formal sample; fork is the common checkpoint before '
                                 'the original environment transition, which occurs before state sampling; '
                                 'each branch retains its own measured observation end')}
    block['implementation'] = str(config_path.relative_to(ROOT))
    return block
