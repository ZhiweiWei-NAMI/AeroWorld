"""Concrete ScenarioSpec for L5-1_v1.

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
              'placement': {'position_enu_m': [7025.87, 6339.903, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Rain cascade with UAV and ground traffic slowdown',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_weather_l5_1_v1',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7030.0, 6346.0, 32],
                             'resolved_position_enu_m': [7030.0, 6346.0, 32],
                             'rotation_deg': {'yaw_deg': 35}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'weather_car_l5_1_v1',
               'initial_state': {'mode': 'moving'},
               'logical_asset_id': 'vehicle.ground.boxcar.v1',
               'placement': {'edge_id': 'cg_edge_122',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 156.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7044.328, 6308.499, 0.0],
                             'rotation_deg': {'yaw_deg': 12.871},
                             'source_edge_id_hint': 'cg_edge_29',
                             'source_longitudinal_s_hint': 36},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'pedestrian',
               'entity_id': 'weather_ped_l5_1_v1',
               'initial_state': {'mode': 'walking'},
               'logical_asset_id': 'pedestrian.cityops.basic.v1',
               'placement': {'lane_edge_id': 'cg_edge_784',
                             'lane_half_width_m': 1.9,
                             'longitudinal_s': 15.0,
                             'offset_from_curb_m': 1.2,
                             'placement_semantics': 'sidewalk_or_plaza',
                             'resolved_lateral_from_center_m': 3.1,
                             'resolved_position_enu_m': [7003.282, 6365.21, 0.0],
                             'rotation_deg': {'yaw_deg': -111.942},
                             'source_lane_edge_id_hint': 'cg_edge_29',
                             'source_longitudinal_s_hint': 40},
               'placement_mode': 'sidewalk_anchor',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7025.87, 6339.903, 0.0],
                  'radius_m': 196.427,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L5-1_v1',
 'spawn_sequencing': [{'entity_id': 'uav_weather_l5_1_v1', 'tick': 0},
                      {'entity_id': 'weather_car_l5_1_v1', 'tick': 0},
                      {'entity_id': 'weather_ped_l5_1_v1', 'tick': 0}],
 'validation_rules': [{'description': 'uav_weather_l5_1_v1 is declared before event_script references it in L5-1_v1',
                       'entity_id': 'uav_weather_l5_1_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'weather_car_l5_1_v1 is declared before event_script references it in L5-1_v1',
                       'entity_id': 'weather_car_l5_1_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_weather_l5_1_v1',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'weather_car_l5_1_v1',
                       'logical_asset_id': 'vehicle.ground.boxcar.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Rain/fog/wind use weather_state; light and temperature use tick simulation',
                       'rule': 'environment_trigger_kind'}],
 'weather_profile': {'initial': 'clear',
                     'transitions': [{'overrides': {'rain': 0.55, 'visibility_m': 2200.0},
                                      'profile': 'rain',
                                      'tick': 180}]}}


SPEC_DATA = {'category': 'environment',
 'description': 'Rain cascade with UAV and ground traffic slowdown',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_weather_l5_1_v1',
               'initial_pos_enu': [7030.0, 6346.0, 32],
               'initial_rotation_deg': [0.0, 0.0, 35],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'vehicle.ground.boxcar.v1',
               'entity_id': 'weather_car_l5_1_v1',
               'initial_pos_enu': [7044.328, 6308.499, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 12.871],
               'movement_waypoints': [],
               'visual_state': {'mode': 'moving'}},
              {'asset_id': 'pedestrian.cityops.basic.v1',
               'entity_id': 'weather_ped_l5_1_v1',
               'initial_pos_enu': [7003.282, 6365.21, 0.0],
               'initial_rotation_deg': [0.0, 0.0, -111.942],
               'movement_waypoints': [],
               'visual_state': {'mode': 'walking'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_rain_onset',
                                          'overrides': {'rain': 0.55, 'visibility_m': 2200.0},
                                          'profile': 'rain'},
                               'type': 'set_weather'}],
                  'event_id': 'weather_onset',
                  'log_category': 'weather',
                  'log_overlay': 'weather',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_weather_l5_1_v1', 'weather_car_l5_1_v1'],
                  'log_title': 'Rain onset is applied before threshold confirmation',
                  'log_topic': 'evt_L5-1_v1_weather_onset',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 180, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'set_rain_moderate',
                                          'overrides': {'rain': 0.55, 'visibility_m': 2200.0},
                                          'profile': 'rain'},
                               'type': 'set_weather'}],
                  'event_id': 'rain_condition_met',
                  'log_category': 'weather',
                  'log_overlay': 'weather',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_weather_l5_1_v1', 'weather_car_l5_1_v1'],
                  'log_title': 'Rain threshold reached',
                  'log_topic': 'evt_L5-1_v1_rain_condition_met',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'sustain_ticks': 5,
                              'type': 'weather_state',
                              'weather_operator': 'gte',
                              'weather_parameter': 'rain',
                              'weather_value': 0.5}},
                 {'actions': [{'params': {'action_id': 'move_uav_rain_slowdown',
                                          'entity_id': 'uav_weather_l5_1_v1',
                                          'velocity_mps': 3.0,
                                          'waypoints_enu_m': [[7030.0, 6346.0, 32], [7040.0, 6354.0, 32]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_car_wet_slowdown',
                                          'entity_id': 'weather_car_l5_1_v1',
                                          'velocity_mps': 3.0,
                                          'waypoints_enu_m': [[7044.328, 6308.499, 0.0], [7051.988, 6310.346, 0.0]]},
                               'type': 'move_entity'}],
                  'event_id': 'rain_speed_reduction',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_weather_l5_1_v1', 'weather_car_l5_1_v1'],
                  'log_title': 'UAV and vehicle slow under rain',
                  'log_topic': 'evt_L5-1_v1_rain_speed_reduction',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'rain_condition_met', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'set_rain_heavy_visibility',
                                          'overrides': {'rain': 0.82, 'visibility_m': 900.0},
                                          'profile': 'rain'},
                               'type': 'set_weather'}],
                  'event_id': 'rain_intensifies',
                  'log_category': 'weather',
                  'log_overlay': 'weather',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_weather_l5_1_v1', 'weather_car_l5_1_v1'],
                  'log_title': 'Rain intensifies and visibility drops',
                  'log_topic': 'evt_L5-1_v1_rain_intensifies',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 47, 'event_ref': 'rain_speed_reduction', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_uav_rain_recover',
                                          'entity_id': 'uav_weather_l5_1_v1',
                                          'velocity_mps': 5.0,
                                          'waypoints_enu_m': [[7040.0, 6354.0, 32], [7024.0, 6362.0, 34]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'set_rain_recovering',
                                          'overrides': {'rain': 0.2, 'visibility_m': 5000.0},
                                          'profile': 'clear'},
                               'type': 'set_weather'}],
                  'event_id': 'rain_recovery',
                  'log_category': 'weather',
                  'log_overlay': 'weather',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_weather_l5_1_v1'],
                  'log_title': 'Rain recovers and UAV exits degraded area',
                  'log_topic': 'evt_L5-1_v1_rain_recovery',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'rain_intensifies', 'type': 'event_fired_after'}}],
 'parameters': {'weather_threshold_tick': 180},
 'scenario_id': 'L5-1_v1'}


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
