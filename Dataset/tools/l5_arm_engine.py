"""Instrumented calls to the supplied engine, retaining its action order and handlers."""
from __future__ import annotations
import copy
import math
from Dataset.tools.l5_arm_common import ROOT, read_json, source, object_hash
from Dataset.tools.l1_2_window import EpisodeStateEngine, EventScriptInterpreter, snapshot, TICK_HZ


def execute(config: dict):
    # sources.script / sources.scene are the authority for this ARM.  An ARM that
    # rewrites its branch points them at its own branch files; an ARM that uses
    # the authored chain points them at the authored scene/script.  Either way the
    # entry script decides, and this executor only runs what is declared.
    script_path=source(config['sources']['script'])
    script=read_json(script_path)
    scene=read_json(source(config['sources']['scene']))
    attachments={e['entity_id']:e['motion_contract']['attachment'] for e in scene['entities']
                 if 'attachment' in e.get('motion_contract',{})}
    previous_attached={}
    end=config['window']['end_tick']
    start=config['window']['start_tick']
    engine=EpisodeStateEngine(scene,script,script_path,end)
    interpreter=EventScriptInterpreter(script_path,episode_id=config['episode_id'])
    source_actions={a['action_id']:a for event in script['events'] for a in event['actions']}
    omissions=set(config['execution']['omit_action_ids'])
    if not omissions<=source_actions.keys(): raise ValueError('unknown source omission')
    handlers={k:getattr(engine,'_handle_'+k) for k in (
        'move_entity','set_visual_state','set_runtime_state','spawn_entity',
        'set_weather','set_pedestrian_activity','remove_entity','capture_screenshot')}
    audit=[]; snaps={}; tick_now=[0]; ownership={}; motion_rows=[]
    def dispatch(action,origin):
        tick=tick_now[0]; kind=action['type']
        if kind not in handlers: raise ValueError('unsupported action '+kind)
        entity=action['entity_id'] if 'entity_id' in action else action['ped_id'] if 'ped_id' in action else None
        before=copy.deepcopy(engine.keyframes[entity]) if entity in engine.entities else None
        weather_before=copy.deepcopy(engine.weather)
        if origin=='source_script' and tick>=config['window']['fork_tick'] and action['action_id'] in omissions:
            result={'status':'omitted_by_explicit_intervention'}
        else:
            result=handlers[kind](action,tick)
            if result['status']!='ok': raise RuntimeError(result)
        effect={'keyframes_changed': before != (engine.keyframes[entity] if entity in engine.entities else None),
                'weather_before':weather_before if kind=='set_weather' else None,
                'weather_after':copy.deepcopy(engine.weather) if kind=='set_weather' else None}
        if entity in engine.entities and kind in ('move_entity','set_visual_state','set_pedestrian_activity') and result['status']=='ok':
            schedule=copy.deepcopy(engine.entities[entity]['schedules'][-1])
            effect['added_schedule']=schedule
            effect['keyframes_after']=copy.deepcopy(engine.keyframes[entity])
            effect['previous_motion_owner']=copy.deepcopy(ownership[entity]) if entity in ownership else None
            if schedule['type']=='move':
                positions=[schedule['start_pos_enu'],*schedule['waypoints_enu_m']]
                speed=schedule['velocity_mps']; duration=0; segments=[]
                for p,q in zip(positions,positions[1:]):
                    distance=math.dist(p,q)
                    if distance<=1e-6: continue
                    ticks=max(1,math.ceil(distance/speed*TICK_HZ)); duration+=ticks
                    segments.append({'distance_m':distance,'ticks':ticks,
                        'velocity_mps':[(b-a)*TICK_HZ/ticks for a,b in zip(p,q)]})
                effect['motion_model']={'configured_speed_mps':speed,'segment_rounding':'ceil_per_nonzero_segment',
                    'nominal_duration_ticks':duration,'nominal_completion_tick':schedule['tick']+duration,
                    'continuous_lower_bound_ticks':sum(s['distance_m'] for s in segments)/speed*TICK_HZ,
                    'segments':segments}
                ownership[entity]={'action_id':action['action_id'],'origin':origin,
                    'dispatch_tick':tick,'scheduled_tick':schedule['tick'],
                    'nominal_completion_tick':schedule['tick']+duration}
            elif schedule['type']=='activity' and schedule['stop_motion'] is True:
                ownership[entity]={'action_id':action['action_id'],'origin':origin,
                    'dispatch_tick':tick,'scheduled_tick':tick,'nominal_completion_tick':tick,
                    'stopping_activity':True}
        audit.append({'tick':tick,'sequence':len(audit),'origin':origin,'action':copy.deepcopy(action),
                      'result':copy.deepcopy(result),'effect':effect})
        return result
    for kind in handlers:
        interpreter.register_handler(kind,lambda action:dispatch(action,'source_script'))
    for tick in range(end+1):
        tick_now[0]=tick
        engine._apply_pending_runtime_state_patches(tick)
        for weather in engine.weather_transitions.get(tick,[]): engine.weather=copy.deepcopy(weather)
        current=engine._record_tick_rows(tick)
        sampled={r['entity_id']:r for r in current}
        for entity, attachment in attachments.items():
            row=sampled[entity]; parent=sampled[attachment['parent_entity_id']]
            # Authored payload motion supplies a relative swing direction, not an
            # independent world trajectory. A taut tether bounds that direction;
            # contact with the authored pad surface permits slack on landing.
            direction=[p-q for p,q in zip(row['pos_enu'],attachment['reference_enu_m'])]
            length=math.sqrt(sum(v*v for v in direction))
            position=[p+v*attachment['tether_length_m']/length
                      for p,v in zip(parent['pos_enu'],direction)]
            position[2]=max(attachment['ground_surface_z_m'],position[2])
            previous=previous_attached.get(entity,position)
            row['pos_enu']=position
            row['vel_mps']=[(p-q)*TICK_HZ for p,q in zip(position,previous)]
            previous_attached[entity]=position
        for row in current:
            interpreter.update_entity_state(row['entity_id'],row['pos_enu'],{},row['vel_mps'])
            if row['label_class']=='pedestrian': interpreter.update_entity_activity(row['entity_id'],row['activity_type'])
            if row['entity_id'] in config['entity_ids']:
                owner=copy.deepcopy(ownership[row['entity_id']]) if row['entity_id'] in ownership else None
                motion_rows.append({'tick':tick,'entity_id':row['entity_id'],
                    'motion_owner':owner,'speed_mps':math.dist([0,0,0],row['vel_mps']),
                    'source_ref':f"raw/trajectories.jsonl#tick={tick}&entity={row['entity_id']}",
                    'attribution_scope':'motion command, not later non-stopping activity labels'})
        interpreter.update_weather_state(engine.weather_rows[-1])
        if tick in (config['window']['start_tick'],config['window']['fork_tick'],end):
            snaps[tick]=snapshot(engine,interpreter,tick,current)
            snaps[tick]['motion_ownership']=copy.deepcopy(ownership)
            snaps[tick]['remaining_configured_interventions']=[copy.deepcopy(s) for s in config['execution']['actions'] if s['tick']>=tick]
        for row in interpreter.tick(tick):
            if row['result']['status'] not in ('ok','omitted_by_explicit_intervention'): raise RuntimeError(row)
        for step in config['execution']['actions']:
            if step['tick']==tick: dispatch(step['action'],'intervention')
    observed={r['action']['action_id'] for r in audit if r['result']['status']=='omitted_by_explicit_intervention'}
    if observed!=omissions: raise ValueError('configured omissions were not dispatched')
    return engine,interpreter,audit,snaps,motion_rows
