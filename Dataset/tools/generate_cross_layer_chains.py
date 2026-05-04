"""Generate cross-layer event chains."""
from __future__ import annotations

import json, sys
sys.path.insert(0, 'Plugins/SumoImporter/Scripts')
from pathlib import Path
from donghu_core.event_script_interpreter import EventScriptInterpreter

chains = {
    'X1_rain_to_c2loss_to_forced_landing': {
        'desc': 'L5-1(rain) -> L6-1(C2 loss) -> L4-3(forced landing)',
        'events': [
            ('rain_onset', 'tick', 300, 1, ['c2_degradation'],
             [{'action_id':'a1','type':'set_weather','profile':'rain','overrides':{'rain':0.8,'visibility_m':1500.0,'wind_speed':8.0}}],
             'weather', '降雨开始', 'warning', 'weather', ['weather_global']),
            ('c2_degradation', 'event_fired', None, 2, ['forced_landing'],
             [{'action_id':'a2','type':'set_visual_state','entity_id':'tower_01','visual_state':{'mode':'disabled'}},
              {'action_id':'a3','type':'capture_screenshot','camera_id':'demo_overview'}],
             'infrastructure', 'C2链路退化', 'critical', 'infrastructure', ['tower_01']),
            ('forced_landing', 'event_fired', None, 3, [],
             [{'action_id':'a4','type':'move_entity','entity_id':'uav_01','waypoints_enu_m':[[50,20,35],[45,15,0.5]],'velocity_mps':3.0}],
             'uav_mission', '紧急迫降', 'critical', 'uav_mission', ['uav_01']),
        ]
    },
    'X2_gnss_spoof_to_geofence_violation': {
        'desc': 'L6-3(GPS spoof) -> L1-1(geofence violation)',
        'events': [
            ('spoofing_start', 'tick', 350, 1, ['geofence_breach'],
             [{'action_id':'a1','type':'move_entity','entity_id':'uav_del_01','waypoints_enu_m':[[58,22,25],[50,20,20]],'velocity_mps':6.0}],
             'uav_mission', 'GPS欺骗开始', 'critical', 'uav_mission', ['uav_del_01']),
            ('geofence_breach', 'event_fired', None, 2, [],
             [{'action_id':'a2','type':'move_entity','entity_id':'uav_del_01','waypoints_enu_m':[[50,20,20],[60,28,30]],'velocity_mps':8.0},
              {'action_id':'a3','type':'capture_screenshot','camera_id':'demo_overview'}],
             'uav_mission', '地理围栏违规', 'critical', 'uav_mission', ['uav_del_01']),
        ]
    },
    'X4_fog_to_uav_conflict': {
        'desc': 'L5-2(fog) -> L4-1(UAV-UAV near-miss)',
        'events': [
            ('fog_onset', 'tick', 300, 1, ['uav_approach'],
             [{'action_id':'a1','type':'set_weather','profile':'fog','overrides':{'fog':0.9,'visibility_m':300.0}}],
             'weather', '团雾出现', 'warning', 'weather', ['weather_global']),
            ('uav_approach', 'event_fired', None, 2, ['conflict_detected'],
             [{'action_id':'a2','type':'move_entity','entity_id':'uav_a','waypoints_enu_m':[[55,18,30],[50,20,25]],'velocity_mps':4.0},
              {'action_id':'a3','type':'move_entity','entity_id':'uav_b','waypoints_enu_m':[[48,22,30],[50,20,25]],'velocity_mps':4.0}],
             'uav_mission', '两UAV在雾中接近', 'warning', 'uav_mission', ['uav_a','uav_b']),
            ('conflict_detected', 'event_fired', None, 3, [],
             [{'action_id':'a4','type':'move_entity','entity_id':'uav_a','waypoints_enu_m':[[50,20,25],[58,18,35]],'velocity_mps':8.0},
              {'action_id':'a5','type':'move_entity','entity_id':'uav_b','waypoints_enu_m':[[50,20,25],[45,24,35]],'velocity_mps':8.0},
              {'action_id':'a6','type':'capture_screenshot','camera_id':'demo_overview'}],
             'uav_mission', '冲突检测-紧急避让', 'critical', 'uav_mission', ['uav_a','uav_b']),
        ]
    },
    'X5_comm_failure_to_pad_contention': {
        'desc': 'L2-1(station failure) -> L6-1(C2 loss) -> L2-4(pad contention)',
        'events': [
            ('station_failure', 'tick', 350, 1, ['uav_lost_link'],
             [{'action_id':'a1','type':'set_visual_state','entity_id':'tower_01','visual_state':{'mode':'disabled'}}],
             'infrastructure', '基站故障', 'warning', 'infrastructure', ['tower_01']),
            ('uav_lost_link', 'event_fired', None, 2, ['pad_conflict'],
             [{'action_id':'a2','type':'move_entity','entity_id':'uav_a','waypoints_enu_m':[[50,20,30],[45,15,20]],'velocity_mps':5.0},
              {'action_id':'a3','type':'move_entity','entity_id':'uav_b','waypoints_enu_m':[[55,25,30],[45,15,25]],'velocity_mps':5.0}],
             'uav_mission', 'C2链路丢失-双机自主返航', 'critical', 'uav_mission', ['uav_a','uav_b']),
            ('pad_conflict', 'event_fired', None, 3, [],
             [{'action_id':'a4','type':'move_entity','entity_id':'uav_a','waypoints_enu_m':[[45,15,20],[45,15,0.5]],'velocity_mps':2.0},
              {'action_id':'a5','type':'move_entity','entity_id':'uav_b','waypoints_enu_m':[[45,15,25],[40,12,0.5]],'velocity_mps':2.0},
              {'action_id':'a6','type':'capture_screenshot','camera_id':'demo_overview'}],
             'uav_mission', '起降平台冲突-优先级仲裁', 'warning', 'uav_mission', ['uav_a','uav_b']),
        ]
    },
    'X6_crowd_evacuation_to_airspace_lockdown': {
        'desc': 'L4-8(crowd evacuation) -> L3-2(temp NFZ activation)',
        'events': [
            ('crowd_evacuation', 'tick', 400, 1, ['nfz_activation'],
             [{'action_id':'a1','type':'spawn_crowd','group_id':'crowd_evac','count':30,'spawn_origin_enu_m':[52,20,0],'spawn_box_extent_cm':[1000,1000,0],'seed':123},
              {'action_id':'a2','type':'capture_screenshot','camera_id':'demo_overview'}],
             'pedestrian', '人群紧急疏散', 'critical', 'pedestrian', ['crowd_evac']),
            ('nfz_activation', 'event_fired', None, 2, [],
             [{'action_id':'a3','type':'spawn_entity','asset_id':'trigger.no_fly.box.v1','entity_id':'nfz_emergency','position_enu_m':[52,20,15]},
              {'action_id':'a4','type':'clear_crowd','group_id':'crowd_evac'}],
             'infrastructure', '紧急禁飞区激活-空域封锁', 'critical', 'infrastructure', ['nfz_emergency']),
        ]
    },
}

def build_chain(chain_data: dict) -> tuple[list[dict], list[dict]]:
    triggers = []
    events = []
    for i, (evt_id, trig_type, tick_val, priority, emit_list, actions, cat, title, sev, overlay, targets) in enumerate(chain_data['events']):
        if trig_type == 'tick':
            tid = f'trig_{evt_id}'
            triggers.append({'trigger_id': tid, 'type': 'tick', 'tick': tick_val})
        else:
            prev_evt = chain_data['events'][i-1][0]
            tid = f'trig_after_{prev_evt}'
            if not any(t['trigger_id'] == tid for t in triggers):
                triggers.append({'trigger_id': tid, 'type': 'event_fired', 'event_id': prev_evt})
        event = {
            'event_id': evt_id, 'trigger_ref': tid,
            'priority': priority, 'max_fire_count': 1,
            'actions': actions,
            'log_event': {
                'topic': f'evt_{chain_id}_{evt_id}',
                'category': cat, 'title': title, 'severity': sev,
                'overlay': overlay, 'target_ids': targets,
            },
        }
        if emit_list:
            event['on_fire'] = {'emit_events': emit_list}
        events.append(event)
    return triggers, events

base = Path('Dataset/scenarios/X_cross_layer')
for chain_id, chain_data in chains.items():
    triggers, events = build_chain(chain_data)
    script = {
        '$schema': 'event_script_v1',
        'scenario_id': chain_id,
        'description': chain_data['desc'],
        'parameters': {},
        'triggers': triggers,
        'events': events,
    }
    d = base / chain_id
    d.mkdir(parents=True, exist_ok=True)
    with open(d / 'event_script.json', 'w', encoding='utf-8') as f:
        json.dump(script, f, indent=2, ensure_ascii=False)

    interp = EventScriptInterpreter(d / 'event_script.json')
    for t in range(900):
        interp.tick(t)
    log = interp.get_event_log()
    exp = len(chain_data['events'])
    status = 'OK' if len(log) == exp else f'FAIL ({len(log)}/{exp})'
    print(f'{status}: {chain_id} — {chain_data["desc"]}')
