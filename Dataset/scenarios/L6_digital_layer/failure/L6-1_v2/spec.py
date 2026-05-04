"""Concrete ScenarioSpec for L6-1_v2.

Generated from Dataset/tools/regenerate_boundary_scenarios.py.
This file is intentionally self-contained: running it recompiles
event_script.json from the ScenarioSpec below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve()
while _TOOLS.name != "Dataset" and _TOOLS.parent != _TOOLS:
    _TOOLS = _TOOLS.parent
_TOOLS = _TOOLS / "tools"
sys.path.insert(0, str(_TOOLS))

from spec_compiler import (
    ActionSpec,
    EntitySpec,
    EventStepSpec,
    ScenarioSpec,
    SpecCompiler,
    TriggerSpec,
    WaypointSpec,
)


SCENE_SETUP = {'$schema': 'scene_setup_v1',
 'cameras': [{'camera_id': 'demo_high_overview',
              'fov_deg': 90.0,
              'placement': {'position_enu_m': [7046.0, 6401.2, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'C2 link loss with return-to-home',
 'entities': [{'activation_tick': 0,
               'category': 'facility',
               'entity_id': 'tower_l6_1_v2',
               'initial_state': {'mode': 'online'},
               'logical_asset_id': 'facility.radio.base_tower.v1',
               'placement': {'position_enu_m': [7050.0, 6403.2, 0],
                             'resolved_position_enu_m': [7050.0, 6403.2, 0],
                             'rotation_deg': {'yaw_deg': 0}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_digital_l6_1_v2',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7042.0, 6399.2, 32],
                             'resolved_position_enu_m': [7042.0, 6399.2, 32],
                             'rotation_deg': {'yaw_deg': 35}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L6-1_v2',
 'spawn_sequencing': [{'entity_id': 'tower_l6_1_v2', 'tick': 0}, {'entity_id': 'uav_digital_l6_1_v2', 'tick': 0}],
 'validation_rules': [{'description': 'tower_l6_1_v2 is declared before event_script references it in L6-1_v2',
                       'entity_id': 'tower_l6_1_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'uav_digital_l6_1_v2 is declared before event_script references it in L6-1_v2',
                       'entity_id': 'uav_digital_l6_1_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'tower_l6_1_v2',
                       'logical_asset_id': 'facility.radio.base_tower.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_digital_l6_1_v2',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Digital layer anomaly produces actual UAV movement and recovery action',
                       'rule': 'digital_anomaly_chain'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'digital_layer',
 'description': 'C2 link loss with return-to-home',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'facility.radio.base_tower.v1',
               'entity_id': 'tower_l6_1_v2',
               'initial_pos_enu': [7050.0, 6403.2, 0],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': {'mode': 'online'}},
              {'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_digital_l6_1_v2',
               'initial_pos_enu': [7042.0, 6399.2, 32],
               'initial_rotation_deg': [0.0, 0.0, 35],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_tower_c2_loss',
                                          'entity_id': 'tower_l6_1_v2',
                                          'visual_state': {'mode': 'link_lost'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_uav_until_c2_loss',
                                          'entity_id': 'uav_digital_l6_1_v2',
                                          'velocity_mps': 6.0,
                                          'waypoints_enu_m': [[7042.0, 6399.2, 32], [7056.0, 6407.2, 31]]},
                               'type': 'move_entity'}],
                  'event_id': 'c2_loss',
                  'log_category': 'digital_layer',
                  'log_overlay': 'digital_layer',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_digital_l6_1_v2', 'tower_l6_1_v2'],
                  'log_title': 'C2 link loss occurs',
                  'log_topic': 'evt_L6-1_v2_c2_loss',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 240, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'set_uav_rth',
                                          'entity_id': 'uav_digital_l6_1_v2',
                                          'visual_state': {'mode': 'return_to_home'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_uav_rth_home',
                                          'entity_id': 'uav_digital_l6_1_v2',
                                          'velocity_mps': 7.0,
                                          'waypoints_enu_m': [[7056.0, 6407.2, 31],
                                                              [7044.0, 6400.2, 34],
                                                              [7032.0, 6393.2, 30]]},
                               'type': 'move_entity'}],
                  'event_id': 'rth_engaged',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_digital_l6_1_v2', 'tower_l6_1_v2'],
                  'log_title': 'UAV return-to-home safety action',
                  'log_topic': 'evt_L6-1_v2_rth_engaged',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'c2_loss', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'set_tower_link_restored',
                                          'entity_id': 'tower_l6_1_v2',
                                          'visual_state': {'mode': 'online'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'set_uav_rth_complete',
                                          'entity_id': 'uav_digital_l6_1_v2',
                                          'visual_state': {'mode': 'home_hold'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'c2_recovered',
                  'log_category': 'digital_layer',
                  'log_overlay': 'digital_layer',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_digital_l6_1_v2', 'tower_l6_1_v2'],
                  'log_title': 'C2 link recovered after RTH',
                  'log_topic': 'evt_L6-1_v2_c2_recovered',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'rth_engaged', 'type': 'event_fired_after'}}],
 'parameters': {'anomaly_tick': 240, 'recovery_tick': 520},
 'scenario_id': 'L6-1_v2'}


def _trigger(data):
    return TriggerSpec(**data)


def _action(data):
    return ActionSpec(data["type"], data.get("params", {}))


def _event(data):
    return EventStepSpec(
        event_id=data["event_id"],
        trigger=_trigger(data["trigger"]),
        actions=[_action(a) for a in data.get("actions", [])],
        on_fire_emit=data.get("on_fire_emit", []),
        priority=data.get("priority", 10),
        max_fire_count=data.get("max_fire_count", 1),
        cooldown_ticks=data.get("cooldown_ticks", 0),
        require_conditions=data.get("require_conditions", []),
        log_topic=data.get("log_topic", ""),
        log_category=data.get("log_category", ""),
        log_title=data.get("log_title", ""),
        log_severity=data.get("log_severity", "info"),
        log_overlay=data.get("log_overlay", ""),
        log_target_ids=data.get("log_target_ids", []),
    )


def build_spec():
    return ScenarioSpec(
        scenario_id=SPEC_DATA["scenario_id"],
        category=SPEC_DATA["category"],
        description=SPEC_DATA["description"],
        duration_ticks=SPEC_DATA["duration_ticks"],
        parameters=SPEC_DATA["parameters"],
        entities=[
            EntitySpec(
                entity_id=e["entity_id"],
                asset_id=e["asset_id"],
                initial_pos_enu=e["initial_pos_enu"],
                initial_rotation_deg=e.get("initial_rotation_deg", [0.0, 0.0, 0.0]),
                movement_waypoints=[WaypointSpec(w) for w in e.get("movement_waypoints", [])],
                visual_state=e.get("visual_state"),
            )
            for e in SPEC_DATA["entities"]
        ],
        event_chain=[_event(e) for e in SPEC_DATA["event_chain"]],
    )


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    spec = build_spec()
    compiled = SpecCompiler().compile(spec)
    with open(here / "event_script.json", "w", encoding="utf-8") as f:
        json.dump(compiled, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(here / "scene_setup.json", "w", encoding="utf-8") as f:
        json.dump(SCENE_SETUP, f, indent=2, ensure_ascii=False)
        f.write("\n")
