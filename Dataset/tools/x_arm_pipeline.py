"""Measured cross-layer windows; branch definitions explicitly select source events.

Business observations and script emissions are separate from registry predicates.
No ontology event is inferred from a script event name.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, math, sys, time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'Dataset/tools'))
from Dataset.tools.l6_v2.engine import execute, snapshot
from Dataset.tools.arm_ue_export import write_engine_inputs,export_arm
from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates,get_governed_parameter_defaults
from Dataset.semantic_truth.world_truth import _evaluate_expression, _expression_truth_value
from Dataset.semantic_simulation.predicate_state_computers import (
 _restricted_regions, _signed_distance_to_restricted_prism, restricted_region_runtime_active,
 _aircraft_pair_separation_models, _equivalent_aircraft_separation_distance)

PYTHON='/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python'
PARAMS=get_governed_parameter_defaults()
TEMPLATES={x['id']:x for x in get_core_predicate_templates()}
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def lines(p,rs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def relative(p):return str(Path(p).relative_to(ROOT))

def entry_module(path):
 path=Path(path).resolve()
 spec=importlib.util.spec_from_file_location('x_arm_'+hashlib.sha256(str(path).encode()).hexdigest(),path)
 if spec is None or spec.loader is None:raise ValueError('Cannot load dedicated arm implementation '+str(path))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 required={'profile','seed','arm_id','window','research_question','hypothesis','window_basis','parameter_model'}
 if set(module.SPEC)!=required:raise ValueError('Dedicated SPEC fields differ from current contract '+str(path))
 if not callable(module.build_script) or not callable(module.verify_run):raise ValueError('Missing branch construction or measured verification '+str(path))
 return module


def branch_changes(source,branch):
 old_events={e['event_id']:e for e in source['events']};new_events={e['event_id']:e for e in branch['events']}
 old_actions={a['action_id']:a for e in source['events'] for a in e['actions']};new_actions={a['action_id']:a for e in branch['events'] for a in e['actions']}
 old_triggers={t['trigger_id']:t for t in source['triggers']};new_triggers={t['trigger_id']:t for t in branch['triggers']}
 return {'removed_event_ids':sorted(old_events.keys()-new_events.keys()),'added_event_ids':sorted(new_events.keys()-old_events.keys()),
         'removed_action_ids':sorted(old_actions.keys()-new_actions.keys()),'added_action_ids':sorted(new_actions.keys()-old_actions.keys()),
         'replaced_action_ids':sorted(k for k in new_actions.keys()&old_actions.keys() if new_actions[k]!=old_actions[k]),
         'changed_event_ids':sorted(k for k in new_events.keys()&old_events.keys() if new_events[k]!=old_events[k]),
         'changed_trigger_ids':sorted(k for k in new_triggers if k not in old_triggers or new_triggers[k]!=old_triggers[k])}


def prepare():
 """Compile individually written branches; never generate their implementations."""
 entries=load(ROOT/'design/X_ue_delivery/entries.json')
 if not entries or len(entries)!=len(set(entries)):raise ValueError('Empty or duplicate X implementation inventory')
 configs=[];group_definitions={}
 for name in entries:
  entry=ROOT/name;module=entry_module(entry);spec=copy.deepcopy(module.SPEC)
  profile,seed,group_id,filename=entry.relative_to(ROOT/'scripts/arms').parts
  arm=entry.stem;group='/'.join((profile,seed,group_id))
  if (spec['profile'],spec['seed'],spec['arm_id'])!=(profile,seed,arm):raise ValueError('Dedicated branch identity differs from its path')
  win=spec['window'];start,fork,end=(win[k] for k in ('start_tick','fork_tick','end_tick'))
  if not 0<=start<fork<end<=900 or any(t%5 for t in (start,fork,end)):raise ValueError('Invalid formal window')
  if win['phase']!='after_state_sample_before_actions':raise ValueError('Invalid fork phase')
  source_scene=ROOT/'Dataset/scenarios/X_cross_layer'/profile/'scene_setup.json';source_script=source_scene.with_name('event_script.json')
  canonical=ROOT/'aw_data/render_ready_episodes_capture_filtered'/f'{profile}__{seed}'
  if not canonical.is_dir():raise FileNotFoundError(canonical)
  original=load(source_script);scene=load(source_scene);branch=module.build_script(copy.deepcopy(original))
  excluded=[e['entity_id'] for e in scene['entities'] if e.get('initial_state',{}).get('role')=='semantic_background_vehicle']
  referenced={a[k] for e in original['events'] for a in e['actions'] for k in ('entity_id','ped_id') if k in a}
  referenced.update(t[k] for t in original['triggers'] for k in ('entity_a','entity_b') if k in t)
  if referenced.intersection(excluded):raise ValueError('A scripted participant cannot be relegated to immutable background')
  scene['entities']=[e for e in scene['entities'] if e['entity_id'] not in excluded]
  changes=branch_changes(original,branch)
  modified_actions=set(changes['added_action_ids']+changes['replaced_action_ids'])
  intervention_actions=[{'event_id':e['event_id'],'trigger_ref':e['trigger_ref'],'action':a} for e in branch['events'] for a in e['actions'] if a['action_id'] in modified_actions]
  config_dir=ROOT/'interventions'/group;config_path=config_dir/(arm+'.yaml');scene_path=config_dir/'scene_setup.json';script_path=config_dir/(arm+'_event_script.json')
  definition={'entrypoint':name,'research_question':spec['research_question'],'window_basis':spec['window_basis'],'parameter_model':spec['parameter_model'],'branch_changes':changes}
  sources={'scene':relative(scene_path),'original_scene':relative(source_scene),'script':relative(source_script),'canonical':relative(canonical),'predicate_script':relative(config_dir/'factual_event_script.json')}
  scope={'simulated_entity_ids':[e['entity_id'] for e in scene['entities']],'canonical_only_scene_entity_ids':excluded,'background_policy':'Every canonical truth-frame entity outside the engine roster is retained unchanged. Background vehicles are not simulated or replaced.','interaction_limit':'Scripted participants, geometry triggers and runtime state are recomputed; no feedback from immutable background traffic is claimed.'}
  config={'profile':profile,'seed':seed,'episode_id':f'{profile}__{seed}','group':group,'arm_id':arm,'mode':'factual_observation' if arm=='factual' else 'intervention','window':win,'sources':sources,'removed_event_ids':changes['removed_event_ids'],'intervention_actions':intervention_actions,'definition':definition,'gates':{t['trigger_id']:t for t in branch['triggers'] if t['type']=='composite'},'scope':scope,'hypothesis':spec['hypothesis']}
  if config_path.exists() or script_path.exists():
   # Re-prepare must be idempotent: an existing config/event-script pair is
   # accepted only when identical to the current declaration.
   existing=load(config_path)
   existing.get('definition',{}).pop('entrypoint_sha256',None)
   if not (config_path.exists() and script_path.exists() and existing==config and load(script_path)==branch):
    if script_path.exists() and load(script_path)!=branch:
     write(script_path,branch)
     write(config_path,config)
    elif not (config_path.exists() and script_path.exists()):
     raise FileExistsError(config_path)
    else:
     raise FileExistsError(config_path)
   if group not in group_definitions:
    group_definitions[group]={'group':group,'window':win,'scope':scope,'baseline':'Explicitly versioned source-script factual, executed with the current engine; canonical background retained unchanged. Branch-local repairs are enumerated per arm.','arms':{}}
   if group_definitions[group]['window']!=win:raise ValueError('Group observation windows differ')
   group_definitions[group]['arms'][arm]=definition
   configs.append(relative(config_path));continue
  if scene_path.exists():
   if load(scene_path)!=scene:raise ValueError('Group arms do not share their scoped scene')
  else:write(scene_path,scene)
  write(config_path,config);write(script_path,branch);configs.append(relative(config_path))
  if group not in group_definitions:
   group_definitions[group]={'group':group,'window':win,'scope':scope,'baseline':'Explicitly versioned source-script factual, executed with the current engine; canonical background retained unchanged. Branch-local repairs are enumerated per arm.','arms':{}}
  if group_definitions[group]['window']!=win:raise ValueError('Group observation windows differ')
  group_definitions[group]['arms'][arm]=definition
 for group,experiment in group_definitions.items():
  factual={'factual'}
  declared=set(experiment['arms'])
  if not factual < declared:raise ValueError('Comparison group lacks factual '+group)
  write(ROOT/'design'/group/'experiment.json',experiment)
 write(ROOT/'design/X_ue_delivery/configs.json',configs)
 return configs

def trajectory_index(rows):
 out=defaultdict(dict)
 for r in rows:out[r['tick']][r['entity_id']]=r
 return out

def check_canonical_background(out,config):
 """Compare every retained frame entity with its immutable canonical record."""
 canonical=ROOT/config['sources']['canonical']/'truth_frames.jsonl'
 with (out/'ue/truth_frames.jsonl').open() as handle:actual={r['tick']:r for line in handle if (r:=json.loads(line))}
 engine_ids=set(config['scope']['simulated_entity_ids']);start=config['window']['start_tick'];end=config['window']['end_tick']
 checked=0;entity_rows=0;seen=set();background_ids=set()
 with canonical.open() as handle:
  for line in handle:
   frame=json.loads(line);tick=frame['tick']
   if not start<=tick<=end:continue
   expected=[entity for entity in frame['entities'] if entity['entity_id'] not in engine_ids]
   measured=[entity for entity in actual[tick]['entities'] if entity['entity_id'] not in engine_ids]
   if measured!=expected:raise ValueError('Canonical background entity record changed at '+str(tick))
   seen.add(tick);checked+=1;entity_rows+=len(expected);background_ids.update(e['entity_id'] for e in expected)
 if seen!=actual.keys() or not seen:raise ValueError('Canonical/UE window frame coverage differs')
 report={'status':'pass','canonical_source':relative(canonical),
         'compared_frames':checked,'compared_background_entity_records':entity_rows,'background_entity_ids':sorted(background_ids),
         'comparison':'Exact decoded entity records, including identity, pose, velocity, annotations and runtime fields. Frame episode/frame identifiers are intentionally ARM-local.'}
 write(out/'raw/canonical_background_reference.json',report)
 return report

def calculate(config,run):
 engine=run['engine'];start=config['window']['start_tick'];end=config['window']['end_tick'];dense=[];tr=trajectory_index(engine.trajectory_rows);weather={r['tick']:r for r in engine.weather_rows}
 def add(t,pid,bindings,value,measure,authority='X business observation'):
  dense.append({'tick':t,'predicate_id':pid,'bindings':bindings,'value':_expression_truth_value(value),'measurements':measure,'authority':authority,'source_refs':[f'raw/trajectories.jsonl#tick={t}&entity={e}' for e in bindings.values() if e in tr[t]] or [f'raw/weather.jsonl#tick={t}' if pid.startswith('environment.') else f'raw/trajectories.jsonl#tick={t}']})
 def reg(t,pid,b,ctx):add(t,pid,b,_evaluate_expression(TEMPLATES[pid]['evaluation_spec'],ctx,{},PARAMS),ctx,'Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json#'+pid)
 scene=load(ROOT/config['sources']['scene']);auth={e['entity_id']:e for e in scene['entities']}
 main=[i for i,e in auth.items() if e.get('initial_state',{}).get('role')=='mission_uav']
 regions=_restricted_regions(auth)
 original=load(ROOT/config['sources']['predicate_script'])
 separation_models=_aircraft_pair_separation_models(original,auth,PARAMS['separation_margin_m'])
 families=('communication_state','navigation_state','control_state','mission_state','incident_state','facility_state')
 runtime_fields={eid:{family:set() for family in families} for eid in auth}
 request_entities={a['entity_id'] for event in original['events'] for a in event['actions'] if a['type']=='set_runtime_state' and 'request_count' in a['state_patch'].get('facility_state',{})}
 for eid,entry in auth.items():
  for family in families:
   runtime_fields[eid][family].update(k for k,v in entry.get(family,{}).items() if type(v) is bool)
 for event in original['events']:
  for action in event['actions']:
   if action['type']=='set_runtime_state' and action['entity_id'] in auth:
    for family in families:
     runtime_fields[action['entity_id']][family].update(k for k,v in action['state_patch'].get(family,{}).items() if type(v) is bool)
 for tick in range(start,end+1):
  frame=tr[tick];w=weather[tick]
  for pid,key in [('environment.rain_active','rain'),('environment.fog_active','fog_density')]:
   if key in w:reg(tick,pid,{'region':config['profile']+'.authored_weather_domain'},{'weather':{key:w[key]}})
  for eid,row in frame.items():
   if eid not in auth:continue
   # Source engine runtime families are physical/control observations, not event
   # labels. Missing booleans are unknown unless the authored source declares them.
   for family in families:
    base=auth[eid].get(family,{})
    values=copy.deepcopy(base);values.update(row.get(family,{}))
    for field in sorted(runtime_fields[eid][family]):
     value=values.get(field)
     add(tick,'x.runtime.'+family+'.'+field,{'entity':eid},value,{family:{field:value}})
   if row['label_class'] in ('uav','pedestrian','vehicle'):
    speed=math.dist([0,0,0],row['vel_mps']);add(tick,'x.motion.moving',{'entity':eid},speed>1e-6,{'speed_mps':speed})
    if row['label_class']=='uav':add(tick,'x.motion.descending',{'entity':eid},row['vel_mps'][2]<-1e-6,{'vz_mps':row['vel_mps'][2]})
    if row['label_class']=='pedestrian':
     posture=row.get('pedestrian_state',{}).get('posture',row['posture'])
     add(tick,'x.pedestrian.fallen',{'entity':eid},posture in ('fallen','lying','lying_down') or row['activity_type'] in ('fall','fallen','falling','lying','lying_down'),{'activity_type':row['activity_type'],'posture':posture,'pedestrian_state':row.get('pedestrian_state')})
   if eid in request_entities:
    facility=row.get('facility_state',{});count=facility.get('request_count');add(tick,'x.facility.multiple_requests',{'entity':eid},None if count is None else count>1,{'request_count':count,'requester_ids':facility.get('requester_ids')})
  for i,a in enumerate(main):
   for b in main[i+1:]:
    if a in frame and b in frame:
     pa,pb=frame[a]['pos_enu'],frame[b]['pos_enu']
     dist=_equivalent_aircraft_separation_distance(euclidean_distance=math.dist(pa,pb),horizontal_distance=math.dist(pa[:2],pb[:2]),vertical_distance=abs(pa[2]-pb[2]),model=separation_models.get(tuple(sorted((a,b)))),separation_margin_m=PARAMS['separation_margin_m'])
     reg(tick,'safety.aircraft_pair_below_separation',{'first_aircraft':a,'second_aircraft':b},{'derived':{'nearest_aircraft_distance_m':dist}})
  # Use the current region geometry and activation contract, independently of
  # the source script's spherical proximity alert. A restored navigation flag
  # cannot manufacture clearance of the governed restricted-airspace predicate.
  for region_id,region in regions.items():
   if region['spawn_policy']=='event_script_only':
    active=False if region_id not in frame else restricted_region_runtime_active({**frame[region_id],'annotations':{'activity_type':frame[region_id]['activity_type']}})
   else:
    active=tick>=region['activation_tick'] and (region['deactivation_tick'] is None or tick<region['deactivation_tick'])
   reg(tick,'emergency.temporary_no_fly_zone_active',{'region':region_id},{'scene':{'restricted_region_active':active}})
   dense[-1]['source_refs'].append(config['sources']['scene']+'#entity='+region_id)
   for aircraft in main:
    if aircraft not in frame:continue
    distance=_signed_distance_to_restricted_prism(frame[aircraft]['pos_enu'],region)
    reg(tick,'airspace.aircraft_inside_restricted_airspace',{'actor':aircraft,'restricted_region':region_id},{'geometry':{'restricted_region_active':active,'minimum_restricted_boundary_distance_m':distance}})
    dense[-1]['source_refs'].append(config['sources']['scene']+'#entity='+region_id)
  # Every authored geometry trigger is evaluated from actual branch samples;
  # hold counters/event gates belong to script emission evidence below.
  for trigger in original['triggers']:
   if trigger['type']!='entity_proximity':continue
   a,b=trigger['entity_a'],trigger['entity_b'];value=None;measure={}
   if a in frame and b in frame:
    pa,pb=frame[a]['pos_enu'],frame[b]['pos_enu'];d=math.dist(pa,pb);xy=math.dist(pa[:2],pb[:2]);z=abs(pa[2]-pb[2]);measure={'distance_3d_m':d,'distance_xy_m':xy,'distance_z_m':z,'trigger':trigger}
    value=(xy<=trigger['horizontal_distance_m'] and z<=trigger['vertical_distance_m']) if trigger.get('metric')=='xy_plus_z' else d<=trigger['distance_m']
   add(tick,'x.geometry.'+trigger['trigger_id'],{'first':a,'second':b},value,measure)
 # Formal truth transitions and complete business lifecycles are derived from
 # measured dense series; no expected field or event firing fills truth values.
 grouped=defaultdict(list)
 for r in dense:
  if r['tick']%5==0:grouped[(r['predicate_id'],digest(r['bindings']))].append(r)
 transitions=[];occurrences=[];outcomes=[]
 for series in grouped.values():
  active=None;false_count=0
  for before,after in zip(series,series[1:]):
   if before['value']!=after['value']:
    transition={'transition_id':'xt:'+digest([after['predicate_id'],after['bindings'],after['tick']]),'predicate_id':after['predicate_id'],'bindings':after['bindings'],'from_tick':before['tick'],'to_tick':after['tick'],'before':before['value'],'after':after['value'],'source_refs':after['source_refs']};transitions.append(transition)
   if before['value']=='false' and after['value']=='true' and active is None:
    active={'event_id':'xb:'+digest([config['episode_id'],config['arm_id'],after['predicate_id'],after['bindings'],after['tick']]),'tick':after['tick'],'predicate_id':after['predicate_id'],'bindings':after['bindings'],'kind':'measured_business_predicate_onset','ontology_event':False,'source_refs':after['source_refs']};occurrences.append(active);false_count=0
   if active:
    false_count=false_count+1 if after['value']=='false' else 0
    if false_count>=PARAMS['recovery_hold_samples']:
     outcomes.append({'event_id':active['event_id'],'terminal_tick':after['tick'],'status':'recovered','hold_samples':false_count});active=None
  if active:outcomes.append({'event_id':active['event_id'],'terminal_tick':None,'status':'right_censored'})
 return dense,transitions,occurrences,outcomes

def run_arm(config_path):
 t=time.monotonic();config_path=Path(config_path);config=load(config_path);group=config['group'];arm=config['arm_id'];out=ROOT/'arms'/group/arm
 if config['mode'] not in ('factual_observation','intervention'):raise ValueError('Unsupported ARM mode '+str(config['mode']))
 if out.exists():raise FileExistsError(out)
 entry=ROOT/config['definition']['entrypoint'];module=entry_module(entry)
 sp=config_path.with_name(arm+'_event_script.json');window=config['window'];script=load(sp)
 if module.build_script(load(ROOT/config['sources']['script']))!=script:raise ValueError('Compiled branch differs from dedicated implementation')
 start,fork,end=(window[k] for k in ('start_tick','fork_tick','end_tick'))
 run=execute(config['sources']['scene'],relative(sp),config['episode_id'],end_tick=end,snapshot_ticks=(start,fork))
 engine=run['engine'];interp=run['interpreter'];observed=[r for r in engine.trajectory_rows if start<=r['tick']<=end]
 branch_evidence=module.verify_run(config,run)
 if not isinstance(branch_evidence,dict) or not branch_evidence:raise ValueError('Dedicated branch returned no measured semantic evidence')
 lines(out/'raw/trajectories.jsonl',observed);lines(out/'raw/states.jsonl',observed);lines(out/'raw/weather.jsonl',[r for r in engine.weather_rows if start<=r['tick']<=end]);lines(out/'raw/actions.jsonl',[r for r in run['audit'] if start<=r['tick']<=end])
 write(out/'objective/branch_semantic_checks.json',{'status':'pass','implementation':relative(entry),'measurements':branch_evidence,'review_type':'execution_self_check'})
 for tick,name in ((start,'initial_state'),(fork,'fork_state')):write(out/'raw'/f'{name}.json',run['snapshots'][str(tick)])
 dense,transitions,occurrences,outcomes=calculate(config,run)
 # Independent numerical checks cover every reported executable registry value.
 for row in dense:
  pid=row['predicate_id'];ctx=row['measurements']
  if pid=='environment.rain_active': expected=ctx['weather']['rain']>PARAMS['rain_active_threshold']
  elif pid=='environment.fog_active': expected=ctx['weather']['fog_density']>PARAMS['fog_density_threshold']
  elif pid=='safety.aircraft_pair_below_separation': expected=ctx['derived']['nearest_aircraft_distance_m']<PARAMS['separation_margin_m']
  else:continue
  if (row['value']=='true')!=expected:raise ValueError('independent predicate formula mismatch')
 effects=[];all_rows=trajectory_index(engine.trajectory_rows)
 for item in run['audit']:
  action=item['action'];tick=item['tick']
  if not start<=tick<=end:continue
  eid=action.get('entity_id',action.get('ped_id'));later=[a['tick'] for a in run['audit'] if a['tick']>tick and a['action'].get('entity_id',a['action'].get('ped_id'))==eid and a['action']['type'] in ('move_entity','set_pedestrian_activity')]
  until=min(later) if later else end+1
  samples=[(t,rs[eid]) for t,rs in all_rows.items() if tick<t<until and eid in rs]
  moving=[t for t,r in samples if math.dist([0,0,0],r['vel_mps'])>1e-6]
  effect={'action_id':action['action_id'],'dispatch_tick':tick,'entity_id':eid,'result':item['result'],'evidence_interval_end_exclusive':until,'first_observed_moving_tick':min(moving) if moving else None,'attribution':'motion observed within this command interval; runtime effects remain in predicate/state samples'}
  if action['type']=='move_entity':
   target=action['waypoints_enu_m'][-1];hit=[t for t,r in samples if math.dist(r['pos_enu'],target)<1e-5]
   effect.update({'target_enu_m':target,'first_target_tick':min(hit) if hit else None,'right_censored':not hit and until==end+1,'preempted_before_target':not hit and until<=end})
  effects.append(effect)
 lines(out/'raw/action_effects.jsonl',effects)
 for name,data in [('predicate_truth_ticks',dense),('predicate_truth',[r for r in dense if r['tick']%5==0]),('predicate_transitions',transitions),('event_occurrences',occurrences),('event_outcomes',outcomes)]:lines(out/'objective'/(name+'.jsonl'),data)
 emissions=[{'node_id':'script:'+eid,'event_id':eid,'tick':state.last_fired_tick,'kind':'script_emission','ontology_event':False} for eid,state in interp.event_states.items() if state.fired and start<=state.last_fired_tick<=end]
 lines(out/'raw/script_emissions.jsonl',emissions)
 nodes=[dict(n) for n in emissions]+[{'node_id':e['event_id'],**e} for e in occurrences]
 emitted={n['event_id']:n for n in emissions};by_trigger={x['trigger_id']:x for x in script['triggers']};edges=[]
 def predecessors(ref):
  tr=by_trigger[ref]
  if tr['type'] in ('event_fired','event_fired_after'):return [tr['event_id']]
  return [p for c in tr.get('children',[]) for p in predecessors(c)]
 for event in script['events']:
  if event['event_id'] not in emitted:continue
  for pred in predecessors(event['trigger_ref']):
   if pred in emitted:edges.append({'source':'script:'+pred,'target':'script:'+event['event_id'],'type':'script_dependency','causal_claim':False,'source_ref':relative(sp)+'#'+event['trigger_ref']})
 lines(out/'chain/nodes.jsonl',nodes);lines(out/'chain/relations.jsonl',edges)
 status={'window':window,'business_event_count':len(occurrences),'script_emission_count':len(emissions),'right_censored_event_ids':[e['event_id'] for e in outcomes if e['terminal_tick'] is None],'unknown_rows':sum(r['value']=='unknown' for r in dense),'ontology_certification':False,'registry_predicates':[p for p in sorted({r['predicate_id'] for r in dense}) if not p.startswith('x.')],'left_boundary_policy':'initial true is observed, never fabricated as a rising event','left_active_observations':[{'predicate_id':r['predicate_id'],'bindings':r['bindings'],'tick':r['tick']} for r in dense if r['tick']==start and r['value']=='true'],'unknown_runtime_policy':'missing undeclared runtime booleans remain unknown at every tick','unfinished_action_ids':[e['action_id'] for e in effects if e.get('right_censored')]}
 write(out/'objective/window_status.json',status)
 write_engine_inputs(out,engine,interp,window,script=script)
 sources={relative(config_path):relative(config_path),relative(sp):relative(sp),**{config['sources'][key]:config['sources'][key] for key in ('scene','original_scene','script','predicate_script')}}
 manifest={'episode_id':config['episode_id'],'arm_id':arm,'mode':config['mode'],'window':window,'sources':config['sources'],'source_refs':{k:v for k,v in sources.items()},'entity_ids':sorted(engine.entities),'removed_event_ids':config['removed_event_ids'],'scope':config['scope'],'branch_changes':config['definition']['branch_changes'],'baseline':'Versioned scoped factual; branch-local script repairs explicitly recorded. Canonical background entities outside the engine roster are retained unchanged.','semantics':'business observations plus executable registry expressions; script nodes retain script_dependency type'}
 write(out/'manifest.json',manifest)
 ue=export_arm(out,ROOT/config['sources']['canonical'],ROOT/config['sources']['scene'],sp)
 background=check_canonical_background(out,config)
 rows=trajectory_index(observed)
 expected=set(range(start,end+1))
 if set(rows)!=expected:raise ValueError('missing state ticks')
 # Samples precede action dispatch. Reconstruct presence from the initial
 # scene and successful lifecycle actions, without leaking final spawn state.
 lifecycle={e['entity_id']:{'active':e.get('spawn_policy','').lower()!='event_script_only',
             'from_tick':int(e.get('activation_tick') or 0)} for e in load(ROOT/config['sources']['scene'])['entities']}
 changes={}
 for action in engine.executed_actions:
  if action['type'] in ('spawn_entity','remove_entity') and action['result']['status']=='ok':
   changes.setdefault(action['tick']+1,[]).append(action)
 for tick in range(end+1):
  for action in changes.get(tick,[]):
   lifecycle[action['entity_id']]={'active':action['type']=='spawn_entity','from_tick':tick}
  if tick<start:continue
  want={eid for eid,e in lifecycle.items() if e['active'] and e['from_tick']<=tick}
  if set(rows[tick])!=want:raise ValueError(('entity lifecycle coverage',tick,sorted(want-set(rows[tick])),sorted(set(rows[tick])-want)))
 for sname,tick in [('initial_state',start),('fork_state',fork)]:
  if load(out/'raw'/f'{sname}.json')['sampled_states']!=[r for r in observed if r['tick']==tick]:raise ValueError('checkpoint mismatch')
 write(out/'manifest.json',manifest)
 result={'arm':relative(out),'seconds':round(time.monotonic()-t,3),'window':window,'ue':ue,'background':background,'source':config['sources'],'data_validation':'pass','script_emissions':emissions,'predicate_rows':len(dense),'business_events':len(occurrences)}
 write(out/'validation.json',result);print(json.dumps({'arm':result['arm'],'seconds':result['seconds'],'data_validation':'pass'}),flush=True)
 return result

def compare(group):
 root=ROOT/'arms'/group
 arms=sorted(p.name for p in root.iterdir() if p.is_dir() and (p/'manifest.json').is_file())
 if 'factual' not in arms:raise ValueError('Comparison group lacks factual '+group)
 paths={a:root/a for a in arms}
 def readrows(p):return [json.loads(l) for l in p.read_text().splitlines()]
 base=readrows(paths['factual']/'raw/trajectories.jsonl');win=load(paths['factual']/'manifest.json')['window'];results={}
 for arm,path in paths.items():
  actual=readrows(path/'raw/trajectories.jsonl');left=[r for r in base if r['tick']<=win['fork_tick']];right=[r for r in actual if r['tick']<=win['fork_tick']]
  if left!=right:raise ValueError('shared prefix differs '+str(path))
  snap=load(path/'raw/fork_state.json');ref=load(paths['factual']/'raw/fork_state.json')
  # Static future schedule definitions intentionally differ. Compare the actual
  # sampled state and physical pending/keyframe state at the declared fork.
  for key in ('sampled_states','keyframes','weather','pending_runtime_state_patches','last_yaw_by_entity'):
   if snap[key]!=ref[key]:raise ValueError('fork execution state mismatch '+key)
  events=readrows(path/'raw/script_emissions.jsonl');manifest=load(path/'manifest.json');actual_ids={e['event_id'] for e in events}
  if actual_ids.intersection(manifest['removed_event_ids']):raise ValueError('removed branch event fired')
  results[arm]={'shared_prefix':'pass','shared_physical_execution_state':'pass','sample_count':len(actual),'script_events':sorted(actual_ids),'absent_declared_events':manifest['removed_event_ids'],'trajectory_matches_factual':actual==base}
 report={'group':group,'status':'pass','baseline':'saved factual window','arms':results,'causal_scope':'declared script subchain ablation; geometry and runtime measured independently','review_type':'self_check'}
 write(root/'comparison.json',report);return report

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',type=int);parser.add_argument('--all',action='store_true');args=parser.parse_args()
 configs=prepare() if args.prepare else load(ROOT/'design/X_ue_delivery/configs.json')
 if args.run is not None:run_arm(ROOT/configs[args.run])
 if args.all:
  for c in configs:
   conf=load(ROOT/c);out=ROOT/'arms'/conf['group']/conf['arm_id']
   if out.exists():
    load(out/'validation.json');continue
   run_arm(ROOT/c)
  groups=sorted({load(ROOT/c)['group'] for c in configs})
  for group in groups:compare(group)
if __name__=='__main__':main()
