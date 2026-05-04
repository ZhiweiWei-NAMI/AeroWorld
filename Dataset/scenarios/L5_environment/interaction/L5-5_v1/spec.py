"""Concrete ScenarioSpec for L5-5_v1.

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
              'placement': {'position_enu_m': [7111.104, 6371.289, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'High temperature battery derating and shortened range',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_weather_l5_5_v1',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7094.0, 6374.0, 32],
                             'resolved_position_enu_m': [7094.0, 6374.0, 32],
                             'rotation_deg': {'yaw_deg': 35}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'weather_car_l5_5_v1',
               'initial_state': {'mode': 'moving'},
               'logical_asset_id': 'vehicle.ground.boxcar.v1',
               'placement': {'edge_id': 'cg_edge_122',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 232.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7115.257, 6334.099, 0.0],
                             'rotation_deg': {'yaw_deg': 23.489},
                             'source_edge_id_hint': 'cg_edge_29',
                             'source_longitudinal_s_hint': 36},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'pedestrian',
               'entity_id': 'weather_ped_l5_5_v1',
               'initial_state': {'mode': 'walking'},
               'logical_asset_id': 'pedestrian.cityops.basic.v1',
               'placement': {'lane_edge_id': 'cg_edge_718',
                             'lane_half_width_m': 1.9,
                             'longitudinal_s': 50.0,
                             'offset_from_curb_m': 1.2,
                             'placement_semantics': 'sidewalk_or_plaza',
                             'resolved_lateral_from_center_m': 3.1,
                             'resolved_position_enu_m': [7124.056, 6405.767, 0.0],
                             'rotation_deg': {'yaw_deg': 134.16},
                             'source_lane_edge_id_hint': 'cg_edge_29',
                             'source_longitudinal_s_hint': 40},
               'placement_mode': 'sidewalk_anchor',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7111.104, 6371.289, 0.0],
                  'radius_m': 197.421,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L5-5_v1',
 'spawn_sequencing': [{'entity_id': 'uav_weather_l5_5_v1', 'tick': 0},
                      {'entity_id': 'weather_car_l5_5_v1', 'tick': 0},
                      {'entity_id': 'weather_ped_l5_5_v1', 'tick': 0}],
 'validation_rules': [{'description': 'uav_weather_l5_5_v1 is declared before event_script references it in L5-5_v1',
                       'entity_id': 'uav_weather_l5_5_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'weather_car_l5_5_v1 is declared before event_script references it in L5-5_v1',
                       'entity_id': 'weather_car_l5_5_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_weather_l5_5_v1',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'weather_car_l5_5_v1',
                       'logical_asset_id': 'vehicle.ground.boxcar.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Rain/fog/wind use weather_state; light and temperature use tick simulation',
                       'rule': 'environment_trigger_kind'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'environment',
 'description': 'High temperature battery derating and shortened range',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_weather_l5_5_v1',
               'initial_pos_enu': [7094.0, 6374.0, 32],
               'initial_rotation_deg': [0.0, 0.0, 35],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'vehicle.ground.boxcar.v1',
               'entity_id': 'weather_car_l5_5_v1',
               'initial_pos_enu': [7115.257, 6334.099, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 23.489],
               'movement_waypoints': [],
               'visual_state': {'mode': 'moving'}},
              {'asset_id': 'pedestrian.cityops.basic.v1',
               'entity_id': 'weather_ped_l5_5_v1',
               'initial_pos_enu': [7124.056, 6405.767, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 134.16],
               'movement_waypoints': [],
               'visual_state': {'mode': 'walking'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_uav_battery_derating',
                                          'entity_id': 'uav_weather_l5_5_v1',
                                          'visual_state': {'mode': 'battery_derating'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'high_temperature_process',
                  'log_category': 'environment',
                  'log_overlay': 'environment',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_weather_l5_5_v1'],
                  'log_title': 'High temperature causes battery derating',
                  'log_topic': 'evt_L5-5_v1_high_temperature_process',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 260, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_uav_shortened_range',
                                          'entity_id': 'uav_weather_l5_5_v1',
                                          'velocity_mps': 4.0,
                                          'waypoints_enu_m': [[7094.0, 6374.0, 32], [7102.0, 6379.0, 32]]},
                               'type': 'move_entity'}],
                  'event_id': 'range_shortened',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_weather_l5_5_v1'],
                  'log_title': 'UAV range shortens under derating',
                  'log_topic': 'evt_L5-5_v1_range_shortened',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30,
                              'event_ref': 'high_temperature_process',
                              'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_uav_thermal_abort',
                                          'entity_id': 'uav_weather_l5_5_v1',
                                          'velocity_mps': 4.5,
                                          'waypoints_enu_m': [[7102.0, 6379.0, 32], [7089.0, 6383.0, 31]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_temperature_abort',
                                          'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'thermal_abort',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_weather_l5_5_v1'],
                  'log_title': 'UAV terminates high-temperature mission',
                  'log_topic': 'evt_L5-5_v1_thermal_abort',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'range_shortened', 'type': 'event_fired_after'}}],
 'parameters': {'weather_threshold_tick': 260},
 'scenario_id': 'L5-5_v1'}


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
