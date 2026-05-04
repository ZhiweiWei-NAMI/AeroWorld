"""Concrete ScenarioSpec for L5-2_v1.

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
              'placement': {'position_enu_m': [57.333, 155.5, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Sudden fog visual navigation failure',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_weather_l5_2_v1',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [54.0, 156.5, 32], 'rotation_deg': {'yaw_deg': 35}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'weather_car_l5_2_v1',
               'initial_state': {'mode': 'moving'},
               'logical_asset_id': 'vehicle.ground.boxcar.v1',
               'placement': {'edge_id': 'cg_edge_29', 'lane_index': 0, 'lateral_offset_m': 0.0, 'longitudinal_s': 36},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'pedestrian',
               'entity_id': 'weather_ped_l5_2_v1',
               'initial_state': {'mode': 'walking'},
               'logical_asset_id': 'pedestrian.cityops.basic.v1',
               'placement': {'lane_edge_id': 'cg_edge_29', 'longitudinal_s': 40, 'offset_from_curb_m': 1.2},
               'placement_mode': 'sidewalk_anchor',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L5-2_v1',
 'spawn_sequencing': [{'entity_id': 'uav_weather_l5_2_v1', 'tick': 0},
                      {'entity_id': 'weather_car_l5_2_v1', 'tick': 0},
                      {'entity_id': 'weather_ped_l5_2_v1', 'tick': 0}],
 'validation_rules': [{'description': 'uav_weather_l5_2_v1 is declared before event_script references it in L5-2_v1',
                       'entity_id': 'uav_weather_l5_2_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'weather_car_l5_2_v1 is declared before event_script references it in L5-2_v1',
                       'entity_id': 'weather_car_l5_2_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_weather_l5_2_v1',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'weather_car_l5_2_v1',
                       'logical_asset_id': 'vehicle.ground.boxcar.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Rain/fog/wind use weather_state; light and temperature use tick simulation',
                       'rule': 'environment_trigger_kind'}],
 'weather_profile': {'initial': 'clear',
                     'transitions': [{'overrides': {'fog': 0.6, 'visibility_m': 650.0},
                                      'profile': 'fog',
                                      'tick': 190}]}}


SPEC_DATA = {'category': 'environment',
 'description': 'Sudden fog visual navigation failure',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_weather_l5_2_v1',
               'initial_pos_enu': [54.0, 156.5, 32],
               'initial_rotation_deg': [0.0, 0.0, 35],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'vehicle.ground.boxcar.v1',
               'entity_id': 'weather_car_l5_2_v1',
               'initial_pos_enu': [62.0, 148.5, 0],
               'initial_rotation_deg': [0.0, 0.0, 90],
               'movement_waypoints': [],
               'visual_state': {'mode': 'moving'}},
              {'asset_id': 'pedestrian.cityops.basic.v1',
               'entity_id': 'weather_ped_l5_2_v1',
               'initial_pos_enu': [56.0, 161.5, 0],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': {'mode': 'walking'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_fog_dense',
                                          'overrides': {'fog': 0.62, 'visibility_m': 650.0},
                                          'profile': 'fog'},
                               'type': 'set_weather'}],
                  'event_id': 'fog_condition_met',
                  'log_category': 'weather',
                  'log_overlay': 'weather',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_weather_l5_2_v1'],
                  'log_title': 'Fog threshold reached',
                  'log_topic': 'evt_L5-2_v1_fog_condition_met',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'sustain_ticks': 5,
                              'type': 'weather_state',
                              'weather_operator': 'gte',
                              'weather_parameter': 'fog',
                              'weather_value': 0.5}},
                 {'actions': [{'params': {'action_id': 'set_uav_visual_nav_degraded',
                                          'entity_id': 'uav_weather_l5_2_v1',
                                          'visual_state': {'mode': 'visual_nav_degraded'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_uav_fog_slow_hold',
                                          'entity_id': 'uav_weather_l5_2_v1',
                                          'velocity_mps': 2.5,
                                          'waypoints_enu_m': [[54.0, 156.5, 32], [60.0, 160.5, 32]]},
                               'type': 'move_entity'}],
                  'event_id': 'visual_navigation_failure',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_weather_l5_2_v1'],
                  'log_title': 'Visual navigation degrades in fog',
                  'log_topic': 'evt_L5-2_v1_visual_navigation_failure',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'event_ref': 'fog_condition_met', 'type': 'event_fired'}},
                 {'actions': [{'params': {'action_id': 'set_fog_worse_visibility',
                                          'overrides': {'fog': 0.85, 'visibility_m': 260.0},
                                          'profile': 'fog'},
                               'type': 'set_weather'},
                              {'params': {'action_id': 'capture_fog_failure', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'fog_worsens',
                  'log_category': 'weather',
                  'log_overlay': 'weather',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_weather_l5_2_v1'],
                  'log_title': 'Fog worsens and visibility collapses',
                  'log_topic': 'evt_L5-2_v1_fog_worsens',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'event_ref': 'visual_navigation_failure', 'type': 'event_fired'}},
                 {'actions': [{'params': {'action_id': 'move_uav_fog_abort',
                                          'entity_id': 'uav_weather_l5_2_v1',
                                          'velocity_mps': 4.0,
                                          'waypoints_enu_m': [[60.0, 160.5, 32], [46.0, 168.5, 34]]},
                               'type': 'move_entity'}],
                  'event_id': 'fog_mission_abort',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_weather_l5_2_v1'],
                  'log_title': 'UAV aborts mission under dense fog',
                  'log_topic': 'evt_L5-2_v1_fog_mission_abort',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'event_ref': 'fog_worsens', 'type': 'event_fired'}}],
 'parameters': {'weather_threshold_tick': 180},
 'scenario_id': 'L5-2_v1'}


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
