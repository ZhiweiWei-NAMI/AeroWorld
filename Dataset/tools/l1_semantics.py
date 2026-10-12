"""L1 scoped projection using the current geometry, predicate and event authorities.

No event is authored here. This module supplies measured operands to registered
expressions, then uses the shared transition, occurrence and recovery engines.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path
from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates, get_core_event_occurrence_types, get_governed_parameter_defaults
from Dataset.semantic_truth.world_truth import _evaluate_expression, _expression_truth_value
from Dataset.semantic_truth.minimal_semantics import ObservationRow, Assertion, Truth, build_transitions, build_events, build_outcomes
from Dataset.semantic_truth.minimal_semantics_adapter import _adapt_assertions, _adapt_transitions, _adapt_occurrences, _adapt_outcomes
from Dataset.semantic_simulation.predicate_state_computers import (
    _restricted_regions, _signed_distance_to_restricted_prism, _airspace_corridors,
    _corridor_active_at_tick, _point_in_oriented_corridor_box,
    _aircraft_pair_separation_models, _equivalent_aircraft_separation_distance,
    resolve_aircraft_ground_references,
)
ROOT = Path(__file__).resolve().parents[2]
PARAMS = get_governed_parameter_defaults()
TEMPLATES = {x['id']:x for x in get_core_predicate_templates()}
RULES = {x['event_family_id']:x for x in get_core_event_occurrence_types()}
RESTRICTED='airspace.aircraft_inside_restricted_airspace'
ALTITUDE='airspace.aircraft_outside_assigned_altitude_band'
PAIR='safety.aircraft_pair_below_separation'
CAPACITY='airspace.airspace_corridor_over_capacity'
AIRBORNE='airspace.aircraft_airborne'
HOME='airspace.aircraft_at_home_pad'
TOUCHDOWN='airspace.aircraft_touchdown'
LANDING='airspace.aircraft_in_landing_zone'

def digest(x):
    return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

@lru_cache(maxsize=3)
def source_uavs(path):
    result={}
    with (ROOT/path).open() as f:
        for line in f:
            r=json.loads(line)
            if r['label_class']=='uav':result[(r['tick'],r['entity_id'])]=r
    return result

def project(config, trajectories, engine_entities):
    """Compute the target family's full-prefix predicates and lifecycle evidence."""
    scene=json.loads((ROOT/config['sources']['scene']).read_text())
    script=json.loads((ROOT/config['sources']['script']).read_text())
    roster={e['entity_id']:e for e in json.loads((ROOT/config['sources']['roster']).read_text())['entities']}
    scene_by={e['entity_id']:e for e in scene['entities']}
    # Observed background remains source-owned; all simulated actor rows override it.
    observed=dict(source_uavs(config['sources']['trajectories']))
    for r in trajectories:
        if r['label_class']=='uav':observed[(r['tick'],r['entity_id'])]=r
    positions={};velocities={};activities={}
    for (tick,eid),r in observed.items():
        positions.setdefault(tick,{})[eid]=tuple(r['pos_enu'])
        velocities.setdefault(tick,{})[eid]=tuple(r['vel_mps'])
        activities.setdefault(tick,{})[eid]=r['activity_type']
    ground=resolve_aircraft_ground_references(roster,positions,velocities,activities,scene_by_id=scene_by,episode_id=config['episode_id'])
    regions=_restricted_regions(scene_by)
    if any(x['spawn_policy']=='event_script_only' for x in regions.values()):
        raise ValueError('L1 scoped projector requires explicit static region activity')
    corridors=_airspace_corridors(scene_by,PARAMS['separation_margin_m'])
    models=_aircraft_pair_separation_models(script,{**scene_by,**roster},PARAMS['separation_margin_m'])
    subject=config['entity_ids'][0]
    rule=RULES[config['event_family']]
    primary=rule['trigger_predicate_id']
    bindings=[]
    if primary==RESTRICTED:
        bindings=[(primary,{'actor':subject,'restricted_region':rid}) for rid in regions]
    elif primary==ALTITUDE:bindings=[(primary,{'actor':subject})]
    elif primary==PAIR:
        actors=sorted(e for e in engine_entities if engine_entities[e]['label_class']=='uav')
        bindings=[(primary,{'first_aircraft':a,'second_aircraft':b}) for a,b in itertools.combinations(actors,2) if subject in (a,b)]
    elif primary==CAPACITY:
        bindings=[(primary,{'corridor':cid}) for cid in corridors]
    elif primary in (AIRBORNE,TOUCHDOWN):
        home,_,_,zone,_=ground.home_by_uav[subject]
        if home is None or zone is None:raise ValueError(f'{subject}: lifecycle geometry absent')
        bindings=[(AIRBORNE,{'aircraft':subject}),(HOME,{'aircraft':subject,'home_pad':home}),
                  (TOUCHDOWN,{'aircraft':subject}),(LANDING,{'aircraft':subject,'landing_zone':zone})]
    else:raise ValueError(f'Unsupported L1 primary predicate {primary}')
    observations=[];assertions=[];lineage={};dense=[]
    max_tick=max(r['tick'] for r in trajectories)
    for tick in range(max_tick+1):
        pos=positions.get(tick,{})
        for pid,b in bindings:
            tpl=TEMPLATES[pid];ctx={};extra={};needed=[]
            if pid in (RESTRICTED,ALTITUDE,AIRBORNE,HOME,TOUCHDOWN,LANDING):needed=[subject]
            elif pid==PAIR:needed=list(b.values())
            missing=[e for e in needed if e not in pos]
            refs=[f'raw/ue_trajectories.jsonl#tick={tick}&entity={e}' if e in engine_entities else config['sources']['trajectories']+f'#tick={tick}&entity={e}' for e in needed]
            if not missing:
                if pid==RESTRICTED:
                    reg=regions[b['restricted_region']];active=tick>=reg['activation_tick'] and (reg['deactivation_tick'] is None or tick<reg['deactivation_tick'])
                    ctx={'geometry':{'restricted_region_active':active,'minimum_restricted_boundary_distance_m':_signed_distance_to_restricted_prism(pos[subject],reg)}}
                elif pid==ALTITUDE:
                    z=roster[subject].get('assigned_altitude_m')
                    if z is None:raise ValueError('Assigned altitude absent: '+subject)
                    ctx={'geometry':{'position_z_m':pos[subject][2],'assigned_altitude_m':z}}
                elif pid==PAIR:
                    a,c=pos[b['first_aircraft']],pos[b['second_aircraft']]
                    model=models.get(tuple(sorted(b.values())))
                    distance=_equivalent_aircraft_separation_distance(euclidean_distance=math.dist(a,c),horizontal_distance=math.dist(a[:2],c[:2]),vertical_distance=abs(a[2]-c[2]),model=model,separation_margin_m=PARAMS['separation_margin_m'])
                    ctx={'derived':{'nearest_aircraft_distance_m':distance}}
                    extra={'pair_positions_enu_m':[list(a),list(c)],'separation_model':model}
                elif pid==CAPACITY:
                    cell=corridors[b['corridor']];active=_corridor_active_at_tick(cell,tick)
                    occupants=sorted(e for e,p in pos.items() if active and _point_in_oriented_corridor_box(p,cell))
                    ctx={'scene':{'maximum_corridor_occupancy_count':len(occupants),'maximum_corridor_capacity':cell['capacity']}}
                    extra={'occupied_aircraft_ids':occupants,'corridor_active':active}
                    refs=[config['sources']['scene']+'#entity='+b['corridor']]+[f'raw/ue_trajectories.jsonl#tick={tick}&entity={e}' if e in engine_entities else config['sources']['trajectories']+f'#tick={tick}&entity={e}' for e in occupants]
                else:
                    home,pad,_,zone,landing=ground.home_by_uav[subject]
                    local=ground.local_ground_reference_by_tick[tick][subject]
                    ctx={'geometry':{'z_agl_m':pos[subject][2]-local,'local_ground_reference_z_m':local,'speed_mps':math.dist((0,0,0),velocities[tick][subject]),'xy_distance_to_home_pad_m':math.dist(pos[subject][:2],pad[:2]),'home_pad_id':home,'xy_distance_to_assigned_landing_zone_m':math.dist(pos[subject][:2],landing[:2]),'assigned_landing_zone_id':zone}}
                for f in tpl['state_contract']['required_fields']:
                    parent,key=f['field'].split('.',1)
                    if ctx.get(parent,{}).get(key) is None:raise ValueError(f'Missing operand {pid} {f}')
                value=_evaluate_expression(tpl['evaluation_spec'],ctx,{},PARAMS)
                if value is None:raise ValueError('Unknown despite complete geometry: '+pid)
                value=_expression_truth_value(value)
            else:value='out_of_scope'
            operands={f'{a}.{k}':v for a,items in ctx.items() for k,v in items.items()}
            row={'tick':tick,'predicate_id':pid,'bindings':b,'value':value,'operands':operands,'source_refs':refs,**extra}
            dense.append(row)
            if tick%PARAMS['formal_step_ticks']:continue
            roleorder=[x['key'] for x in tpl['argument_roles']];participants=tuple(b[k] for k in roleorder)
            tupletype='grounded_roles__'+'__'.join(roleorder);tupleid='l1_tuple:'+digest([pid,b]).split(':')[1][:24]
            context={'bindings':b,'binding_ontology_classes':{x['key']:'world:'+x['class'] for x in tpl['argument_roles']},'world_truth_value':value,**ctx}
            observations.append(ObservationRow(config['episode_id'],tick,tupleid,tupletype,participants,not missing,'deterministic_derived',tuple(refs),None,context))
            assertions.append(Assertion(config['episode_id'],tick,tupleid,tupletype,participants,pid,Truth(value),'deterministic_derived',tuple(refs),None))
            lineage[(config['episode_id'],tick,tupleid,pid)]=(digest(row),digest(PARAMS))
    transitions=build_transitions(assertions,PARAMS)
    events=build_events(observations,assertions,transitions,[rule],roster_by_id=roster)
    outcomes=build_outcomes(events,assertions,PARAMS,[rule],transitions=transitions)
    truth=_adapt_assertions(assertions,observations,lineage)
    truthmap={(r['episode_id'],r['tick'],r['tuple_id'],r['predicate_id']):r for r in truth}
    tr,breaks=_adapt_transitions(transitions,truthmap)
    if breaks:raise ValueError('Unexpected semantic sampling discontinuity')
    trmap={(r['episode_id'],r['to_tick'],r['tuple_id'],r['predicate_id'],r['direction']):r for r in tr}
    ev=_adapt_occurrences(events,observations,assertions,trmap,[rule],outcomes,roster_by_id=roster)
    oc=_adapt_outcomes(outcomes,ev,truthmap,formal_step=PARAMS['formal_step_ticks'])
    return {'predicate_truth':truth,'predicate_transitions':tr,'event_occurrences':ev,'event_outcomes':oc,'l0_predicate_state':dense,'predicate_continuity_breaks':[]}
