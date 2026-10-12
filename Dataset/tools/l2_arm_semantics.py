"""L2 候选：当前语义表达式 + 原有状态原语，声明局部输入边界。"""
from __future__ import annotations
import copy
import itertools
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
for _directory in (ROOT, ROOT / 'Dataset/tools', ROOT / 'Plugins/SumoImporter/Scripts'):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

# 运行时状态族：当前引擎每 tick 真实写出的 typed runtime 字段。
RUNTIME_FAMILIES = ('communication_state', 'navigation_state', 'facility_state',
                    'control_state', 'incident_state', 'battery_state', 'mission_state',
                    'security_state', 'sensor_state', 'constraint_state')


def load(path: str | Path) -> Any:
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return 'sha256:' + h.hexdigest()


def digest(value: Any) -> str:
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                  allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _source(ref: str) -> Path:
    path = (ROOT / ref).resolve()
    path.relative_to(ROOT)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def identities(episode: dict) -> dict:
    """Author-declared actor set: every entity named by the event script plus scene UAVs."""
    scene = load(_source(episode['scene']))
    script = load(_source(episode['script']))
    scene_entities = {e['entity_id']: e for e in scene['entities']}
    uavs = sorted(e['entity_id'] for e in scene['entities'] if e['category'] == 'uav')
    actors = sorted({a['entity_id'] for e in script['events'] for a in e['actions']
                     if 'entity_id' in a and a['type'] != 'capture_screenshot'} | set(uavs))
    return {'scene_entities': scene_entities, 'uavs': uavs, 'actors': actors}
from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates, get_core_event_occurrence_types,
    get_governed_parameter_defaults,
)
from Dataset.semantic_truth.world_truth import _evaluate_expression, _expression_truth_value
from Dataset.semantic_truth.minimal_semantics import (
    Assertion, ObservationRow, Truth, build_transitions, build_events, build_outcomes,
)
from Dataset.semantic_truth.minimal_semantics_adapter import (
    _adapt_assertions, _adapt_transitions, _adapt_occurrences, _adapt_outcomes,
)
from Dataset.semantic_simulation import predicate_state_computers as geo
from Dataset.semantic_simulation import domain_state as domain
from Dataset.semantic_simulation import compute_comm as comm
from Dataset.semantic_truth.facility_scope import validate_roster_facility_scope

REGISTRY = 'Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json'
DOMAIN = 'Dataset/semantic_rules/profiles/domain_state_supplement_profile.json'
COMM = 'Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json'
PARAMS = get_governed_parameter_defaults()
TEMPLATES = {t['id']: t for t in get_core_predicate_templates()}
RULES = get_core_event_occurrence_types()
ALT = 'airspace.aircraft_outside_assigned_altitude_band'
AIR = 'airspace.aircraft_airborne'
TOUCH = 'airspace.aircraft_touchdown'
HOME = 'airspace.aircraft_at_home_pad'
LAND = 'airspace.aircraft_in_landing_zone'
PAIR = 'safety.aircraft_pair_below_separation'
STRUCTURE = 'safety.aircraft_near_building_structure'
CORRIDOR = 'airspace.airspace_corridor_over_capacity'
C2_LOSS = 'communication.link_unavailable'
C2_DEG = 'communication.link_degraded'
GNSS = 'positioning.gnss_quality_degraded'
CHARGER = 'facility.charging_station_unavailable'
QUEUE = 'facility.charging_queue_over_capacity'
PAD = 'facility.landing_pad_contention'
SIGNAL = 'ground.traffic_signal_malfunction_active'


def _path(value: dict, path: str) -> Any:
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _fields(spec: dict) -> list[str]:
    if spec['op'] == 'all':
        return [p for a in spec['args'] for p in _fields(a)]
    return [spec[k] for k in ('field', 'left', 'right') if k in spec]


def evaluate_known(predicate_id: str, context: dict) -> tuple[str, list[str]]:
    spec = TEMPLATES[predicate_id]['evaluation_spec']
    missing = [f for f in _fields(spec) if _path(context, f) is None or _path(context, f) == 'unknown']
    if missing:
        return 'unknown', missing
    result = _evaluate_expression(spec, context, {}, PARAMS)
    return _expression_truth_value(result), missing


def metadata_for(episode: dict, initial_snapshot: dict, *, historical: bool) -> dict:
    """roster 的终态字段不进入动态状态。当前静态运动元数据来自本次初始引擎。"""
    roster = {e['entity_id']: e for e in load(ROOT / episode['capture_core_path'] /
                                           'global_entity_roster.json')['entities']}
    scene = load(ROOT / episode['scene'])
    scenes = {e['entity_id']: e for e in scene['entities']}
    metadata = {}
    static_keys = ('entity_id', 'label_class', 'entity_category', 'entity_kind',
                   'semantic_scope', 'lifecycle', 'ground_reference_z_m', 'assigned_altitude_m',
                   'route_waypoints_enu_m', 'initial_position_enu_m', 'ontology_class_id')
    for entity_id, authored in scenes.items():
        if entity_id not in roster:
            # 静态走廊不一定进入 capture roster；它的几何权威就是作者源。
            if authored['category'] == 'airspace_corridor':
                metadata[entity_id] = copy.deepcopy(authored)
                continue
            if authored['category'] == 'vehicle' and entity_id not in identities(episode)['actors']:
                # capture 外背景车辆不进入本地声明集合，不构造类型或动态状态。
                continue
            raise ValueError(f'声明实体缺少类型库存: {entity_id}')
        record = {k: copy.deepcopy(roster[entity_id][k]) for k in static_keys if k in roster[entity_id]}
        if not historical:
            if entity_id in initial_snapshot['entity_definitions']:
                actual = initial_snapshot['entity_definitions'][entity_id]
                record['initial_position_enu_m'] = copy.deepcopy(actual['pos'])
                record['label_class'] = actual['label_class']
                for key in ('lifecycle', 'ground_reference_z_m', 'assigned_altitude_m'):
                    if key in actual:
                        record[key] = copy.deepcopy(actual[key])
            if 'route_waypoints_enu_m' in authored:
                record['route_waypoints_enu_m'] = copy.deepcopy(authored['route_waypoints_enu_m'])
        metadata[entity_id] = record
    return metadata


def _view(row: dict, metadata: dict) -> dict:
    entity = metadata[row['entity_id']]
    result = {k: copy.deepcopy(entity[k]) for k in ('entity_id', 'entity_category', 'entity_kind',
              'semantic_scope', 'lifecycle') if k in entity}
    result['truth_pose'] = {'position_enu_m': row['pos_enu'], 'velocity_enu_mps': row['vel_mps']}
    for family in RUNTIME_FAMILIES:
        if family in row:
            result[family] = copy.deepcopy(row[family])
    return result


def scope_for(episode: dict, metadata: dict, dynamic_ids: list[str]) -> dict:
    """绑定按类型、作者状态字段与计划枚举；不按历史是否发生删掉候选。"""
    scene = load(ROOT / episode['scene'])
    sources = {e['entity_id']: e for e in scene['entities']}
    uavs = sorted(e for e in dynamic_ids if metadata[e]['entity_category'] == 'uav')
    mission = sorted(e for e in uavs if 'lifecycle' in metadata[e]
                     and metadata[e]['lifecycle'].get('requires_takeoff') is True)
    families = sorted({r['event_family_id'] for r in episode['required_chain']} |
                      set(episode['observed_event_families']))
    if mission:
        # 即使历史 inventory 中未出现 altitude，也保留其完整候选计算。
        for family in ('altitude_excursion', 'uav_takeoff', 'uav_landing'):
            if family not in families:
                families.append(family)
    structures = sorted(e for e,v in sources.items() if geo._is_building_structure(e, v))
    corridors = geo._airspace_corridors(sources, PARAMS['separation_margin_m']) if 'corridor_congestion' in families else {}
    stations = sorted(e for e in dynamic_ids if comm._is_station_entity(metadata[e]))
    gnss_actors = sorted(e for e in uavs if 'navigation_state' in sources[e]
                        and 'multipath_warning' in sources[e]['navigation_state'])
    c2_actors = sorted(e for e in uavs if 'communication_state' in sources[e]
                      and 'communication_unavailable' in sources[e]['communication_state'])
    chargers = sorted(f['facility_id'] for f in scene['simulation_plans']['charging_service_plan']['facilities']) if 'charger_unavailable' in families else []
    pads = sorted(e for e in dynamic_ids if 'facility_state' in sources[e]
                  and 'requester_ids' in sources[e]['facility_state']) if 'pad_contention' in families else []
    signals = sorted(e for e in dynamic_ids if sources[e]['category'] == 'traffic_light') if 'signal_malfunction' in families else []
    specs = []
    def add(pid, bindings, kind, **extra):
        classes = {r['key']: 'world:' + r['class'] for r in TEMPLATES[pid]['argument_roles']}
        if set(bindings) != set(classes):
            raise ValueError(f'契约绑定不匹配 {pid}: {bindings} != {classes}')
        specs.append({'predicate_id': pid, 'bindings': bindings, 'binding_ontology_classes': classes,
                      'kind': kind, **extra})
    for uav in uavs:
        if 'altitude_excursion' in families:
            if 'assigned_altitude_m' not in metadata[uav]:
                raise ValueError('缺少 assigned_altitude_m: ' + uav)
            add(ALT, {'actor':uav}, 'altitude')
    for uav in mission:
        lifecycle = metadata[uav]['lifecycle']
        pad = lifecycle['home_pad_entity_id']
        if pad not in metadata:
            raise ValueError('缺少 home pad: ' + pad)
        home_pose = geo._entity_position(metadata[pad])
        hover = tuple(lifecycle['home_hover_enu_m'])
        landing_id, landing_pose = geo._assigned_landing_zone(uav, metadata[uav], pad, home_pose, hover)
        if landing_id is None or landing_pose is None:
            raise ValueError('缺少着陆区域权威: ' + uav)
        if chargers or pads:
            # 任务可在本场景设施间改选目标；逻辑身份在 factual 和干预中一致。
            landing_id = f'assigned_landing_zone:{uav}'
        for pid in (AIR, TOUCH): add(pid, {'aircraft':uav}, 'aircraft_ground')
        add(HOME, {'aircraft':uav,'home_pad':pad}, 'home')
        add(LAND, {'aircraft':uav,'landing_zone':landing_id}, 'landing', landing_position=list(landing_pose))
    if 'uav_separation_conflict' in families:
        for first, second in itertools.combinations(uavs, 2):
            add(PAIR, {'first_aircraft':first,'second_aircraft':second}, 'pair')
    if 'structure_near_miss' in families:
        for uav in mission:
            for structure in structures:
                add(STRUCTURE, {'aircraft':uav,'structure':structure}, 'structure')
    for corridor, values in corridors.items():
        add(CORRIDOR, {'corridor':corridor}, 'corridor', corridor=values)
    if 'c2_degradation_loss' in families:
        if len(stations) != 1:
            raise ValueError('本地链路模型要求一个已声明基站，拒绝合成或猜选基站')
        for uav in c2_actors:
            for pid in (C2_LOSS,C2_DEG): add(pid, {'actor':uav,'station':stations[0]}, 'communication')
    if 'gnss_signal_degradation' in families:
        service = load(ROOT / DOMAIN)['domain_models']['gnss']['service_instance']['service_id']
        for uav in gnss_actors: add(GNSS, {'actor':uav,'service':service}, 'gnss')
    for charger in chargers:
        validate_roster_facility_scope(metadata[charger])
        for pid in (CHARGER,QUEUE): add(pid, {'station':charger}, 'charger')
    for pad in pads:
        validate_roster_facility_scope(metadata[pad])
        for first,second in itertools.combinations(uavs,2):
            add(PAD, {'first_aircraft':first,'pad':pad,'second_aircraft':second}, 'pad')
    for signal in signals: add(SIGNAL, {'signal':signal}, 'signal')
    ids = [digest([s['predicate_id'], s['bindings']]) for s in specs]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('语义库存为空或重复')
    return {'families':sorted(families), 'bindings':specs, 'uav_ids':uavs,
            'mission_uav_ids':mission, 'station_ids':stations, 'gnss_actor_ids':gnss_actors,
            'charger_ids':chargers, 'pad_ids':pads, 'signal_ids':signals,
            'structure_ids':structures, 'corridor_ids':sorted(corridors),
            'scope_policy':'仅当前作者源实体；全量背景流不回灌、不声称全世界无事件'}


def _ground_contexts(trajectories: list[dict], metadata: dict, scope: dict,
                     action_audit: list[dict]) -> dict:
    positions, velocities, activities = {}, {}, {}
    for row in trajectories:
        tick, entity = row['tick'], row['entity_id']
        if tick not in positions:
            positions[tick], velocities[tick], activities[tick] = {}, {}, {}
        positions[tick][entity] = tuple(row['pos_enu'])
        velocities[tick][entity] = tuple(row['vel_mps'])
        activities[tick][entity] = row['activity_type']
    landing_changes = {}
    for record in action_audit:
        action = record['action']
        if record['result']['status'] != 'ok' or action['type'] != 'move_entity':
            continue
        if 'terminal_feasibility' not in action:
            continue
        reference = action['terminal_feasibility']['landing_reference_enu_m']
        # 派发 tick 已完成采样；以实际运动调度后的首个可见 tick 更新目标。
        tick = max(record['tick'], record['result']['scheduled_tick']) + 1
        landing_changes[(tick, action['entity_id'])] = tuple(reference)
    pads = {e:v for e,v in metadata.items() if v.get('semantic_scope',{}).get('scope_subtype') == 'landing_pad'}
    contacts = geo._landing_pad_contact_references_by_tick(pads,positions,velocities,activities,
                    scope['mission_uav_ids'], PARAMS['landing_zone_radius_m'])
    result = {}
    for uav in scope['mission_uav_ids']:
        meta = metadata[uav]
        life = meta['lifecycle']
        home_id = life['home_pad_entity_id']
        home = geo._entity_position(metadata[home_id])
        if home is None or 'ground_reference_z_m' not in meta:
            raise ValueError('本地地面参考缺失: '+uav)
        _, landing = geo._assigned_landing_zone(uav,meta,home_id,home,tuple(life['home_hover_enu_m']))
        for tick in sorted(positions):
            present = positions[tick]
            if (tick, uav) in landing_changes:
                landing = landing_changes[(tick, uav)]
            if uav not in present:
                continue
            pos = present[uav]
            z = geo._local_aircraft_ground_reference_z(pos,landing,home,meta['ground_reference_z_m'],
                        pads,contacts[tick],PARAMS['landing_zone_radius_m'],home[2])
            result[(tick,uav)] = {'z_agl_m':pos[2]-z,'local_ground_reference_z_m':z,
                                 'xy_distance_to_assigned_landing_zone_m':math.dist(pos[:2],landing[:2]),
                                 'speed_mps':math.dist([0,0,0],velocities[tick][uav])}
    return result


def recompute(episode: dict, config: dict, trajectories: list[dict], weather: list[dict],
              metadata: dict, scope: dict, action_audit: list[dict]) -> tuple[dict, list[dict], dict]:
    """只从本次输入算 observed；不读取 expected/event_realization。"""
    rows_by_tick = {}
    for row in trajectories:
        if row['tick'] not in rows_by_tick: rows_by_tick[row['tick']] = {}
        rows_by_tick[row['tick']][row['entity_id']] = row
    weather_by_tick = {r['tick']:r for r in weather}
    # 事件参与实体合法性由 roster 判定（背景标记、任务角色、类别）；缺失即报错，
    # 不把名册外实体当作基础设施静默放过。
    roster_path = ROOT / episode['capture_core_path'] / 'global_entity_roster.json'
    if not roster_path.is_file():
        raise FileNotFoundError(roster_path)
    roster_by_id = {r['entity_id']: r for r in load(roster_path)['entities']}
    dp, cp = load(ROOT/DOMAIN), load(ROOT/COMM)
    scene = load(ROOT/episode['scene'])
    charge_plan = scene['simulation_plans']['charging_service_plan']
    domain._validate_charging_service_plan(charge_plan, episode['episode_id'])
    inputs = SimpleNamespace(episode_id=episode['episode_id'], charging_service_plan=charge_plan,
                            roster_entities={e:metadata[e] for e in scope['uav_ids']})
    shared_seed = digest({'namespace':'l2_arm_sources_seed_v1','episode_id':episode['episode_id'],
                          'scene':sha(ROOT/episode['scene']),'script':sha(ROOT/episode['script']),
                          'domain_parameters':sha(ROOT/DOMAIN)})
    previous_gnss = {e:[0.0,0.0,0.0] for e in scope['gnss_actor_ids']}
    gnss_history_known = {e:True for e in scope['gnss_actor_ids']}
    ground = _ground_contexts(trajectories,metadata,scope,action_audit)
    dense, mechanisms = [], []
    observations, assertions, lineage = [], [], {}
    arm_key = config['group_key']+'/'+config['arm_id']
    semantic_episode = episode['episode_id']+'::'+arm_key
    # 语义重算从 tick 0 起：窗口起点之前已触发的事件必须以真实前置状态进入，
    # 否则窗内会被误判为「无事件」。产物仍由调用方按窗口裁切。
    for tick in range(0, config['window']['end_tick']+1):
        present = rows_by_tick.get(tick,{})
        views = {e:_view(r,metadata) for e,r in present.items()}
        gnss_values, charge_values, comm_values = {}, {}, {}
        for uav in scope['gnss_actor_ids']:
            if tick % PARAMS['formal_step_ticks']:
                continue
            if uav not in views:
                gnss_history_known[uav] = False
                gnss_values[uav] = {'quality_level':'unknown','reason':'历史正式样本缺测，递归误差状态无法认证'}
                continue
            # 历史缺测后不重置递归状态，不以默认值修补它。
            if not gnss_history_known[uav]:
                gnss_values[uav] = {'quality_level':'unknown','reason':'前序 GNSS 递归状态缺测'}
                continue
            values, quality, source_class = domain._gnss_values(uav=views[uav], uav_id=uav,
                       position=present[uav]['pos_enu'], previous_error=previous_gnss[uav],
                       facilities=[], params=dp['domain_models']['gnss'],tick=tick,seed_digest=shared_seed)
            previous_gnss[uav] = values.pop('_next_error_vector_m')
            gnss_values[uav] = values
            mechanisms.append({'tick':tick,'kind':'gnss','entity_id':uav,'values':values,
                               'quality':quality,'source_class':source_class})
        for station in scope['charger_ids']:
            matches = [f for f in charge_plan['facilities'] if f['facility_id']==station]
            if len(matches)!=1: raise ValueError('充电计划身份不唯一')
            plan = matches[0]
            capacity = validate_roster_facility_scope(metadata[station])['service_capacity']
            if plan['capacity'] != capacity: raise ValueError('充电容量不符合资产契约')
            availability = [w['status'] for w in plan['availability_schedule'] if w['start_tick']<=tick<=w['end_tick']]
            requesters = [r['uav_id'] for r in plan['requests'] if r['arrival_tick']<=tick<=r['departure_tick']]
            missing = []
            if len(availability)!=1: missing.append('availability_schedule_dense_gap')
            if station not in views: missing.append('facility_capture_sample_missing')
            # 请求计划自身可计算请求数；不等同于到站或获得充电。
            fault = views.get(station,{}).get('facility_state',{}).get('fault')
            known_fault = isinstance(fault,bool)
            if not known_fault: missing.append('facility_state.fault_missing')
            available = None
            if len(availability)==1 and station in views:
                # 上游当前模型以明确 fault=True 覆盖计划；否则采用计划。
                # 对没有 fault 字段的备选设施，reserved/accepted 不是故障证据。
                state = views[station].get('facility_state')
                if state is not None:
                    available = 'unavailable' if fault is True else availability[0]
            values = {'availability':available,'requester_count':len(requesters),'capacity':capacity,
                      'requester_ids':requesters,'fault_active':fault,'missing':missing}
            charge_values[station]=values
            mechanisms.append({'tick':tick,'kind':'charging_service','entity_id':station,'values':values,
                               'source_refs':[episode['scene']+'#simulation_plans.charging_service_plan']})
        for spec in scope['bindings']:
            pid, bindings, kind = spec['predicate_id'], spec['bindings'], spec['kind']
            ctx, missing = {}, []
            dynamic = sorted(set(bindings.values()) & set(config['entity_ids']))
            refs = [f'raw/states.jsonl#tick={tick}&entity={e}' for e in dynamic]
            if not refs: refs=['raw/metadata.json']
            if kind=='altitude':
                uav=bindings['actor']; row=present.get(uav)
                ctx={'geometry':{'position_z_m':None if row is None else row['pos_enu'][2],
                                 'assigned_altitude_m':metadata[uav]['assigned_altitude_m']}}
            elif kind=='aircraft_ground':
                ctx={'geometry':ground.get((tick,bindings['aircraft']),{})}
            elif kind=='home':
                row=present.get(bindings['aircraft']); pad=geo._entity_position(metadata[bindings['home_pad']])
                ctx={'geometry':{'xy_distance_to_home_pad_m':None if row is None or pad is None else math.dist(row['pos_enu'][:2],pad[:2])}}
            elif kind=='landing':
                ctx={'geometry':ground.get((tick,bindings['aircraft']),{})}
            elif kind=='pair':
                a,b=present.get(bindings['first_aircraft']),present.get(bindings['second_aircraft'])
                ctx={'derived':{'nearest_aircraft_distance_m':None if a is None or b is None else math.dist(a['pos_enu'],b['pos_enu'])}}
            elif kind=='structure':
                a=present.get(bindings['aircraft']); b=geo._entity_position(metadata[bindings['structure']])
                ctx={'derived':{'nearest_building_distance_m':None if a is None or b is None else math.dist(a['pos_enu'],b)}}
            elif kind=='corridor':
                c=spec['corridor']
                occupancy=sum(geo._point_in_oriented_corridor_box(present[u]['pos_enu'],c)
                              for u in scope['uav_ids'] if u in present)
                unknown_uavs=[u for u in scope['uav_ids'] if u not in present]
                ctx={'scene':{'maximum_corridor_occupancy_count':None if unknown_uavs else occupancy,
                              'maximum_corridor_capacity':c['capacity']}}
                refs=[f'raw/states.jsonl#tick={tick}&entity={u}' for u in scope['uav_ids']]+['raw/metadata.json']
            elif kind=='communication':
                a,b=bindings['actor'],bindings['station']; key=(a,b)
                if key not in comm_values:
                    if a not in views or b not in views or tick not in weather_by_tick:
                        comm_values[key]={'link_quality':{'quality_level':'unknown'},'requirement':{'status':'source_required'}}
                    else:
                        station_profile=cp['communication']['station_profiles']['base_station']
                        operational=comm._station_operational_state(views[b],views[a],station_profile)
                        distance=math.dist(present[a]['pos_enu'],present[b]['pos_enu'])
                        quality=comm._link_quality(distance,weather_by_tick[tick],cp['communication']['channel'],station_profile,operational)
                        comm_values[key]={'link_quality':quality,'operational_state':operational,
                              'requirement':{'status':'source_required' if operational['missing_parameters'] else 'not_required'}}
                        mechanisms.append({'tick':tick,'kind':'communication','entity_id':a,'station_id':b,
                                           'values':comm_values[key]})
                f=comm._truth_from_link_unavailable if pid==C2_LOSS else comm._truth_from_link_degraded
                ctx={'direct_predicate_value':f(comm_values[key])}
            elif kind=='gnss':
                ctx={'domain':{'gnss_navigation':{'quality_level':gnss_values.get(bindings['actor'],{}).get('quality_level')}}}
                if tick%PARAMS['formal_step_ticks']: missing.append('非正式 tick：原 GNSS 递归只定义 step=5，不猜测插值')
            elif kind=='charger':
                ctx={'domain':{'pad_facility':charge_values[bindings['station']]}}
            elif kind=='pad':
                pad=bindings['pad']; state=views.get(pad,{}).get('facility_state',{})
                requesters=state.get('requester_ids'); capacity=validate_roster_facility_scope(metadata[pad])['service_capacity']
                if 'capacity' in state and state['capacity'] != capacity: raise ValueError('停机坪容量漂移')
                if isinstance(requesters,list):
                    if len(requesters)!=len(set(requesters)): raise ValueError('requester_ids 重复')
                    if 'request_count' in state and state['request_count'] != len(requesters): raise ValueError('request_count 不符')
                    value=(len(requesters)>capacity and bindings['first_aircraft'] in requesters and bindings['second_aircraft'] in requesters)
                else: value=None
                ctx={'domain':{'pad_facility':{'pair_contention':value,'requester_ids':requesters,'capacity':capacity}}}
            elif kind=='signal':
                fault=views.get(bindings['signal'],{}).get('facility_state',{}).get('fault')
                ctx={'domain':{'signal_queue':{'controller_health':'all_red_fault' if fault is True else 'nominal' if fault is False else None}}}
            else: raise ValueError('不支持的显式语义模型 '+kind)
            value, required_missing=evaluate_known(pid,ctx)
            missing.extend(required_missing)
            row={'tick':tick,'predicate_id':pid,'bindings':bindings,'binding_ontology_classes':spec['binding_ontology_classes'],
                 'value':value,'context':ctx,'source_refs':refs,'missing_source_record':missing}
            dense.append(row)
            if tick%PARAMS['formal_step_ticks']: continue
            tuple_id='arm_tuple:'+digest([arm_key,pid,bindings]).split(':')[1][:24]
            participants=tuple(bindings[k] for k in sorted(bindings))
            tuple_type='grounded_roles__'+'__'.join(sorted(bindings))
            source_class='simulated_derived' if kind in ('communication','gnss','charger','pad','signal') else 'deterministic_derived'
            context={**ctx,'bindings':bindings,'binding_ontology_classes':spec['binding_ontology_classes'],
                     'world_truth_value':value,'truth_state_update_tick':tick,'evidence_update_tick':tick}
            observations.append(ObservationRow(semantic_episode,tick,tuple_id,tuple_type,participants,True,source_class,tuple(refs),None,context))
            assertions.append(Assertion(semantic_episode,tick,tuple_id,tuple_type,participants,pid,Truth(value),source_class,tuple(refs),None,tuple(missing)))
            lineage[(semantic_episode,tick,tuple_id,pid)]=(digest(ctx),digest(PARAMS))
    rules=[r for r in RULES if r['event_family_id'] in scope['families']]
    transitions=build_transitions(assertions,PARAMS)
    events=build_events(observations,assertions,transitions,rules,roster_by_id=roster_by_id)
    outcomes=build_outcomes(events,assertions,PARAMS,rules,transitions=transitions)
    truth=_adapt_assertions(assertions,observations,lineage)
    truth_map={(r['episode_id'],r['tick'],r['tuple_id'],r['predicate_id']):r for r in truth}
    converted,breaks=_adapt_transitions(transitions,truth_map)
    transition_map={(r['episode_id'],r['to_tick'],r['tuple_id'],r['predicate_id'],r['direction']):r for r in converted}
    occurrences=_adapt_occurrences(events,observations,assertions,transition_map,rules,outcomes,roster_by_id=roster_by_id)
    adapted=_adapt_outcomes(outcomes,occurrences,truth_map,formal_step=PARAMS['formal_step_ticks'])
    for collection in (truth,converted,occurrences):
        for row in collection:
            row['parent_episode_id']=episode['episode_id'];row['arm_key']=arm_key
    semantic={'predicate_truth':truth,'predicate_truth_ticks':dense,'predicate_transitions':converted,
              'event_occurrences':occurrences,'event_outcomes':adapted,'discontinuities':breaks}
    info={'gnss_formal_step_ticks':PARAMS['formal_step_ticks'],
          'gnss_nonformal_policy':'unknown_no_interpolation','model_version':'l2_local_state_v1',
          'initial_gnss_error_vector_m':[0,0,0], 'initial_gnss_authority':'当前 domain_state._build_observation_rows 的 episode 起始状态',
          'full_producer_certification':'blocked','frozen_background_feedback':'not_computed'}
    return semantic,mechanisms,info
