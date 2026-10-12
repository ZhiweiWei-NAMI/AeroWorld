"""Current registry semantics over an explicitly bounded, authored-scene candidate domain.

Missing global background, C2/GNSS producers and unresolved roles are not replaced
with nominal values. Historical facts are retained separately from this recomputation.
"""
from __future__ import annotations
import copy
import math
from collections import Counter
from Dataset.tools.l5_arm_common import read_json, source, object_hash
from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates, get_core_event_occurrence_types, get_governed_parameter_defaults)
from Dataset.semantic_truth.world_truth import _evaluate_expression, _expression_truth_value
from Dataset.semantic_truth.minimal_semantics import (
    ObservationRow, Assertion, Truth, build_transitions, build_events, build_outcomes)
from Dataset.semantic_truth.entity_scope import undeclared_event_roles
from Dataset.semantic_truth.minimal_semantics_adapter import (
    _adapt_assertions, _adapt_transitions, _adapt_occurrences, _adapt_outcomes)
from Dataset.tools.l1_2_window import build_chain
from Dataset.semantic_simulation.predicate_state_computers import (
    _assigned_landing_zone, _landing_pad_contact_references_by_tick,
    _local_aircraft_ground_reference_z, _entity_position)

ENV_PREDICATES=(
    'environment.rain_active','environment.fog_active',
    'environment.visibility_below_threshold','environment.wind_speed_above_threshold',
    'environment.low_illumination_active','environment.temperature_outside_operating_range')
MAIN_PREDICATES=(
    'airspace.aircraft_outside_assigned_altitude_band','airspace.aircraft_airborne',
    'airspace.aircraft_at_home_pad','airspace.aircraft_touchdown',
    'airspace.aircraft_in_landing_zone')
NEAR='safety.aircraft_near_pedestrian'
COMPUTED_FAMILIES=('altitude_excursion','rain_fog_environment','high_temp_battery_derating',
                   'uav_takeoff','uav_landing','uav_pedestrian_near_miss')


def get_path(context, dotted):
    node=context
    for part in dotted.split('.'):
        if not isinstance(node,dict) or part not in node: return None
        node=node[part]
    return node


def local_geometry(config, trajectories):
    """Use existing local-contact primitives; deliberately no global-pad substitution.

    This is a new local research domain, not the full public geometry producer.
    Every pad is an authored scene pad resolved through the supplied typed roster.
    Outside those pads the existing per-aircraft ground-surface branch takes the
    home pad surface z, which is an explicit source coordinate, not a guessed plane.
    Global pads/cross-aircraft contact effects remain an external certification block.
    """
    scene=read_json(source(config['sources']['scene']))
    scene_by_id={r['entity_id']:r for r in scene['entities']}
    roster={r['entity_id']:r for r in read_json(source(config['sources']['roster']))['entities']}
    main=config['primary_entity_id']; vehicle=scene_by_id[main]
    if vehicle['category']!='uav' or roster[main]['label_class']!='uav': raise ValueError('primary actor class')
    lifecycle=vehicle['lifecycle']; home=lifecycle['home_pad_entity_id']
    pads={i:r for i,r in roster.items() if i in scene_by_id and 'semantic_scope' in r
          and r['semantic_scope']['scope_type']=='facility'
          and r['semantic_scope']['scope_subtype']=='landing_pad'}
    if home not in pads: raise ValueError('home pad missing from typed authored pad catalog')
    home_pose=_entity_position(pads[home])
    if home_pose is None: raise ValueError('home pad position missing')
    home_hover=tuple(lifecycle['home_hover_enu_m'])
    landing_id,landing_pose=_assigned_landing_zone(main,vehicle,home,home_pose,home_hover)
    if landing_id is None or landing_pose is None: raise ValueError('assigned landing zone unresolved')
    assigned=vehicle['initial_state']['assigned_altitude_m']
    if assigned!=roster[main]['assigned_altitude_m']: raise ValueError('assigned altitude source disagreement')
    positions={t:{} for t in range(config['window']['start_tick'],config['window']['end_tick']+1)}
    velocities={t:{} for t in positions}; activities={t:{} for t in positions}
    for r in trajectories:
        positions[r['tick']][r['entity_id']]=tuple(r['pos_enu'])
        velocities[r['tick']][r['entity_id']]=tuple(r['vel_mps'])
        activities[r['tick']][r['entity_id']]=r['activity_type']
    params=get_governed_parameter_defaults(); radius=params['landing_zone_radius_m']
    contacts=_landing_pad_contact_references_by_tick(pads,positions,velocities,activities,[main],radius)
    geometry=[]
    for tick,points in positions.items():
        row={'tick':tick,'entity_id':main,'geometry':{},'missing_source_record':[]}
        if main not in points:
            row['missing_source_record']=['raw/trajectories.jsonl#primary_actor_not_observed']
        else:
            pos=points[main]; vel=velocities[tick][main]
            ground=_local_aircraft_ground_reference_z(pos,landing_pose,home_pose,
                vehicle['ground_reference_z_m'],pads,contacts[tick],radius,home_pose[2])
            row['geometry']={'position_z_m':pos[2],'assigned_altitude_m':assigned,
                'z_agl_m':pos[2]-ground,'local_ground_reference_z_m':ground,
                'home_pad_id':home,'home_pad_ontology_class_id':'world:LandingPad',
                'assigned_landing_zone_id':landing_id,
                'assigned_landing_zone_ontology_class_id':'world:LandingArea' if landing_id.startswith('assigned_landing_zone:') else pads[landing_id]['semantic_scope']['ontology_class_id'],
                'speed_mps':math.dist((0,0,0),vel),
                'xy_distance_to_home_pad_m':math.dist(pos[:2],home_pose[:2]),
                'xy_distance_to_assigned_landing_zone_m':math.dist(pos[:2],landing_pose[:2])}
            row['local_ground_reference_z_m']=ground
            row['pad_contact_references']=contacts[tick]
        geometry.append(row)
    catalog={'primary_entity_id':main,'primary_ontology_class':'world:UnmannedAircraft',
        'home_pad_id':home,'home_pad_pose_enu_m':list(home_pose),
        'home_contact_reference_z_m':vehicle['ground_reference_z_m'],
        'assigned_altitude_m':assigned,'landing_zone_id':landing_id,
        'landing_zone_pose_enu_m':list(landing_pose),
        'landing_zone_ontology_class':'world:LandingArea' if landing_id.startswith('assigned_landing_zone:')
            else pads[landing_id]['semantic_scope']['ontology_class_id'],
        'pads':pads,'local_ground_surface_z_m':home_pose[2],
        'operational_region_id':config['operational_region_id'],
        'ground_domain':'authored_scene_pads_and_primary_contacts_only',
        'full_global_pad_certification':'blocked: aw_data/uav_outputs/donghu_uav_flow_270s/uav_task_plan.json',
        'source_refs':[config['sources']['scene'],config['sources']['roster']],
        'pedestrian_ids':config['semantic_pedestrian_ids']}
    return geometry,catalog


def calculate(config, trajectories, weather):
    templates={r['id']:r for r in get_core_predicate_templates()}
    params=get_governed_parameter_defaults(); step=params['formal_step_ticks']
    main=config['primary_entity_id']; region=config['operational_region_id']
    geometry,catalog=local_geometry(config,trajectories)
    tr={(r['tick'],r['entity_id']):r for r in trajectories}
    wx={r['tick']:r for r in weather}
    start=config['window']['start_tick']; end=config['window']['end_tick']
    if set(wx)!=set(range(start,end+1)): raise ValueError('weather grid incomplete')
    semantic_episode=config['episode_id']+'::'+config['group_key']+'/'+config['arm_id']
    arm_key=config['group_key']+'/'+config['arm_id']
    observations=[]; assertions=[]; lineage={}; dense=[]; mechanisms=[]
    def add(tick,pid,bindings,classes,context,refs):
        template=templates[pid]
        if {r['key'] for r in template['argument_roles']}!=set(bindings): raise ValueError('predicate binding mismatch')
        required=[r['field'] for r in template['state_contract']['required_fields']]
        missing=[p for p in required if get_path(context,p) is None]
        value=None if missing else _evaluate_expression(template['evaluation_spec'],context,{},params)
        truth=_expression_truth_value(value)
        tuple_id='l5_tuple:'+object_hash([arm_key,pid,bindings]).split(':')[1][:24]
        roles=list(bindings); tuple_type='grounded_roles__'+'__'.join(roles)
        row={'tick':tick,'tuple_id':tuple_id,'predicate_id':pid,'bindings':bindings,
             'value':truth,'context':copy.deepcopy(context),'missing_source_record':missing,
             'source_refs':refs,'authority_ref':'Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json#'+pid}
        dense.append(row)
        if tick%step: return
        ob_context=copy.deepcopy(context)
        ob_context.update({'bindings':bindings,'binding_ontology_classes':classes,'world_truth_value':truth})
        observations.append(ObservationRow(semantic_episode,tick,tuple_id,tuple_type,tuple(bindings.values()),
            True,'deterministic_derived',tuple(refs),None,ob_context))
        assertions.append(Assertion(semantic_episode,tick,tuple_id,tuple_type,tuple(bindings.values()),pid,
            Truth(truth),'deterministic_derived',tuple(refs),None,tuple(missing)))
        lineage[(semantic_episode,tick,tuple_id,pid)]=(object_hash(row),object_hash(params))
    for geo in geometry:
        tick=geo['tick']; main_context={'geometry':geo['geometry']}
        mainref=f'raw/trajectories.jsonl#tick={tick}&entity={main}'
        if 'emergency.mission_aborted' in config['predicate_ids']:
            state=tr[(tick,main)]['mission_state']
            add(tick,'emergency.mission_aborted',{'actor':main},
                {'actor':'world:UnmannedAircraft'},
                {'control':{'mission_abort_active':state['mission_abort_active']}},[mainref])
        for pid in ENV_PREDICATES:
            add(tick,pid,{'region':region},{'region':'world:OperationalRegion'},
                {'weather':wx[tick]},[f'raw/weather.jsonl#tick={tick}'])
        for pid in MAIN_PREDICATES:
            if pid.endswith('outside_assigned_altitude_band'):
                bindings={'actor':main}; classes={'actor':'world:UnmannedAircraft'}
            else:
                bindings={'aircraft':main};classes={'aircraft':'world:UnmannedAircraft'}
                if pid.endswith('at_home_pad'):
                    bindings['home_pad']=catalog['home_pad_id'];classes['home_pad']='world:LandingPad'
                elif pid.endswith('in_landing_zone'):
                    bindings['landing_zone']=catalog['landing_zone_id'];classes['landing_zone']=catalog['landing_zone_ontology_class']
            add(tick,pid,bindings,classes,main_context,[mainref,f'raw/geometry.jsonl#tick={tick}',
                    'raw/binding_catalog.json'])
        for ped in config['semantic_pedestrian_ids']:
            distance=None
            if (tick,main) in tr and (tick,ped) in tr:
                distance=math.dist(tr[(tick,main)]['pos_enu'],tr[(tick,ped)]['pos_enu'])
            add(tick,NEAR,{'aircraft':main,'pedestrian':ped},
                {'aircraft':'world:UnmannedAircraft','pedestrian':'world:Pedestrian'},
                {'derived':{'nearest_pedestrian_distance_m':distance}},
                [mainref,f'raw/trajectories.jsonl#tick={tick}&entity={ped}'])
        mr=tr[(tick,main)] if (tick,main) in tr else None
        mechanisms.append({'tick':tick,'entity_id':main,
            'speed_mps':math.dist([0,0,0],mr['vel_mps']) if mr else None,
            'position_enu_m':mr['pos_enu'] if mr else None,
            'runtime_state':{k:copy.deepcopy(mr[k]) for k in config['runtime_state_families'] if mr and k in mr},
            'missing_runtime_families':[k for k in config['runtime_state_families'] if not mr or k not in mr],
            'weather':{k:wx[tick][k] for k in ('rain','fog_density','visibility_m','wind_speed','temperature_c','illumination_lux')},
            'evidence_refs':[mainref,f'raw/weather.jsonl#tick={tick}'],
            'camera_and_physical_battery_measurements':'not_generated'})
    rules=[r for r in get_core_event_occurrence_types() if r['event_family_id'] in COMPUTED_FAMILIES
           and (r['event_family_id']!='uav_pedestrian_near_miss' or config['semantic_pedestrian_ids'])]
    offenders=undeclared_event_roles()
    if offenders: raise ValueError('event rules bind roles that entity_scope_classes does not declare: '+repr(offenders))
    transitions=build_transitions(assertions,params)
    roster_by_id={r['entity_id']:r for r in read_json(source(config['sources']['roster']))['entities']}
    events=build_events(observations,assertions,transitions,rules,roster_by_id=roster_by_id)
    outcomes=build_outcomes(events,assertions,params,rules,transitions=transitions)
    truth=_adapt_assertions(assertions,observations,lineage)
    truth_map={(r['episode_id'],r['tick'],r['tuple_id'],r['predicate_id']):r for r in truth}
    converted,breaks=_adapt_transitions(transitions,truth_map)
    if breaks: raise ValueError('unexpected formal grid discontinuity')
    transition_map={(r['episode_id'],r['to_tick'],r['tuple_id'],r['predicate_id'],r['direction']):r for r in converted}
    occurrences=_adapt_occurrences(events,observations,assertions,transition_map,rules,outcomes,roster_by_id=roster_by_id)
    adapted_outcomes=_adapt_outcomes(outcomes,occurrences,truth_map,formal_step=step)
    from Dataset.semantic_truth.window_outcome import arm_window_outcome
    adapted_outcomes=[arm_window_outcome(row,start_tick=start,end_tick=end)
                      for row in adapted_outcomes]
    semantic={'predicate_truth':truth,'predicate_truth_ticks':dense,'predicate_transitions':converted,
              'event_occurrences':occurrences,'event_outcomes':adapted_outcomes}
    nodes,relations=build_chain(
        semantic, window_start_tick=start, window_end_tick=config['window']['end_tick'])
    family_counts=Counter(r['event_family_id'] for r in occurrences)
    right_censored=[r['event_id'] for r in adapted_outcomes if r['status']=='right_censored']
    pending_events=[r['event_id'] for r in adapted_outcomes if r['status']=='pending']
    last=config['window']['end_tick']
    end_true=[{'predicate_id':r['predicate_id'],'bindings':r['bindings']} for r in dense if r['tick']==last and r['value']=='true']
    initial_true=[{'predicate_id':r['predicate_id'],'bindings':r['bindings']} for r in dense if r['tick']==start and r['value']=='true']
    unresolved=[]
    rule_catalog=get_core_event_occurrence_types()
    for entry in config['blocked_event_families']:
        pids=sorted({r['trigger_predicate_id'] for r in rule_catalog if r['event_family_id']==entry['event_family_id']})
        for tick in range(start,last+1):
            unresolved.append({'tick':tick,'event_family_id':entry['event_family_id'],
                'predicate_ids':pids,'value':'unknown','bindings':None,
                'candidate_actor':main,'missing_source_record':entry['missing'],
                'reason':entry['reason'],'status':'blocked_not_absent'})
    status={'window':config['window'],'event_counts':dict(family_counts),
        'event_count':len(occurrences),'pending_event_ids':pending_events,
        'right_censored_event_ids':right_censored,
        'right_censored':bool(right_censored),'active_predicates_at_left_boundary':initial_true,
        'active_predicates_at_right_boundary':end_true,
        'unknown_dense_rows':sum(r['value']=='unknown' for r in dense),
        'unresolved_family_count':len(config['blocked_event_families']),
        'empty_event_scope':'only grounded and observed predicate series; blocked families have no absence claim',
        'left_boundary_policy':'no manufactured rising transition from unknown or initial true',
        'ground_geometry_domain':catalog['ground_domain'],
        'formal_step_ticks':step,'recovery_hold_samples':params['recovery_hold_samples'],
        'terminal_hold_samples':params['terminal_hold_samples']}
    return semantic,geometry,catalog,mechanisms,unresolved,status,nodes,relations
