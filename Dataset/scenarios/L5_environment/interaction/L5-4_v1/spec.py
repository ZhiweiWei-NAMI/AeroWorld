"""Concrete ScenarioSpec for L5-4_v1.

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
              'placement': {'position_enu_m': [89.333, 169.5, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Dusk light shift and camera underexposure',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_weather_l5_4_v1',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [86.0, 170.5, 32], 'rotation_deg': {'yaw_deg': 35}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'weather_car_l5_4_v1',
               'initial_state': {'mode': 'moving'},
               'logical_asset_id': 'vehicle.ground.boxcar.v1',
               'placement': {'edge_id': 'cg_edge_29', 'lane_index': 0, 'lateral_offset_m': 0.0, 'longitudinal_s': 36},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'pedestrian',
               'entity_id': 'weather_ped_l5_4_v1',
               'initial_state': {'mode': 'walking'},
               'logical_asset_id': 'pedestrian.cityops.basic.v1',
               'placement': {'lane_edge_id': 'cg_edge_29', 'longitudinal_s': 40, 'offset_from_curb_m': 1.2},
               'placement_mode': 'sidewalk_anchor',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L5-4_v1',
 'spawn_sequencing': [{'entity_id': 'uav_weather_l5_4_v1', 'tick': 0},
                      {'entity_id': 'weather_car_l5_4_v1', 'tick': 0},
                      {'entity_id': 'weather_ped_l5_4_v1', 'tick': 0}],
 'validation_rules': [{'description': 'uav_weather_l5_4_v1 is declared before event_script references it in L5-4_v1',
                       'entity_id': 'uav_weather_l5_4_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'weather_car_l5_4_v1 is declared before event_script references it in L5-4_v1',
                       'entity_id': 'weather_car_l5_4_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_weather_l5_4_v1',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'weather_car_l5_4_v1',
                       'logical_asset_id': 'vehicle.ground.boxcar.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Rain/fog/wind use weather_state; light and temperature use tick simulation',
                       'rule': 'environment_trigger_kind'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'environment',
 'description': 'Dusk light shift and camera underexposure',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_weather_l5_4_v1',
               'initial_pos_enu': [86.0, 170.5, 32],
               'initial_rotation_deg': [0.0, 0.0, 35],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'vehicle.ground.boxcar.v1',
               'entity_id': 'weather_car_l5_4_v1',
               'initial_pos_enu': [94.0, 162.5, 0],
               'initial_rotation_deg': [0.0, 0.0, 90],
               'movement_waypoints': [],
               'visual_state': {'mode': 'moving'}},
              {'asset_id': 'pedestrian.cityops.basic.v1',
               'entity_id': 'weather_ped_l5_4_v1',
               'initial_pos_enu': [88.0, 175.5, 0],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': {'mode': 'walking'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_camera_underexposed',
                                          'entity_id': 'uav_weather_l5_4_v1',
                                          'visual_state': {'mode': 'camera_underexposed'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'capture_dusk_underexposure',
                                          'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'dusk_light_shift',
                  'log_category': 'environment',
                  'log_overlay': 'environment',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_weather_l5_4_v1'],
                  'log_title': 'Dusk light shift underexposes camera',
                  'log_topic': 'evt_L5-4_v1_dusk_light_shift',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 260, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'set_uav_infrared_mode',
                                          'entity_id': 'uav_weather_l5_4_v1',
                                          'visual_state': {'mode': 'infrared_navigation'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_uav_ir_continue',
                                          'entity_id': 'uav_weather_l5_4_v1',
                                          'velocity_mps': 5.0,
                                          'waypoints_enu_m': [[86.0, 170.5, 32], [98.0, 178.5, 32]]},
                               'type': 'move_entity'}],
                  'event_id': 'infrared_switch',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_weather_l5_4_v1'],
                  'log_title': 'UAV switches to infrared navigation',
                  'log_topic': 'evt_L5-4_v1_infrared_switch',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'event_ref': 'dusk_light_shift', 'type': 'event_fired'}},
                 {'actions': [{'params': {'action_id': 'capture_infrared_validation',
                                          'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'low_light_termination',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_weather_l5_4_v1'],
                  'log_title': 'Low-light segment terminates after validation',
                  'log_topic': 'evt_L5-4_v1_low_light_termination',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'event_ref': 'infrared_switch', 'type': 'event_fired'}}],
 'parameters': {'weather_threshold_tick': 260},
 'scenario_id': 'L5-4_v1'}


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
