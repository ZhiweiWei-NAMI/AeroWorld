"""Concrete ScenarioSpec for X3_pedestrian_fall_to_emergency_response.

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
              'placement': {'position_enu_m': [7111.265, 6431.571, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Pedestrian fall to emergency response chain',
 'entities': [{'activation_tick': 0,
               'category': 'pedestrian',
               'entity_id': 'ped_x3_fall',
               'initial_state': {'mode': 'walking'},
               'logical_asset_id': 'pedestrian.cityops.basic.v1',
               'placement': {'lane_edge_id': 'cg_edge_718',
                             'lane_half_width_m': 1.9,
                             'longitudinal_s': 72.149,
                             'offset_from_curb_m': 1.2,
                             'placement_semantics': 'sidewalk_or_plaza',
                             'resolved_lateral_from_center_m': 3.1,
                             'resolved_position_enu_m': [7114.373, 6426.431, 0.0],
                             'rotation_deg': {'yaw_deg': 117.638},
                             'source_lane_edge_id_hint': 'cg_edge_32',
                             'source_longitudinal_s_hint': 50},
               'placement_mode': 'sidewalk_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_x3_detect',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7059.0, 6450.0, 32],
                             'resolved_position_enu_m': [7059.0, 6450.0, 32],
                             'rotation_deg': {'yaw_deg': 60}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'ambulance_x3',
               'initial_state': {'lights_on': True, 'mode': 'response'},
               'logical_asset_id': 'vehicle.emergency.ambulance.v1',
               'placement': {'edge_id': 'cg_edge_718',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 12.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7157.581, 6423.559, 0.0],
                             'rotation_deg': {'yaw_deg': -140.031},
                             'source_edge_id_hint': 'cg_edge_718',
                             'source_longitudinal_s_hint': 12.149},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 420,
               'category': 'prop',
               'entity_id': 'police_tape_x3',
               'initial_state': {'mode': 'staged'},
               'logical_asset_id': 'prop.incident.police_tape.v1',
               'placement': {'position_enu_m': [7114.107, 6426.292, 0.0],
                             'resolved_position_enu_m': [7114.107, 6426.292, 0.0],
                             'rotation_deg': {'yaw_deg': 0}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'X3_pedestrian_fall_to_emergency_response',
 'spawn_sequencing': [{'entity_id': 'ped_x3_fall', 'tick': 0},
                      {'entity_id': 'uav_x3_detect', 'tick': 0},
                      {'entity_id': 'ambulance_x3', 'tick': 0},
                      {'entity_id': 'police_tape_x3', 'tick': 420}],
 'validation_rules': [{'description': 'ped_x3_fall is declared before event_script references it in '
                                      'X3_pedestrian_fall_to_emergency_response',
                       'entity_id': 'ped_x3_fall',
                       'rule': 'entity_resolvable'},
                      {'description': 'uav_x3_detect is declared before event_script references it in '
                                      'X3_pedestrian_fall_to_emergency_response',
                       'entity_id': 'uav_x3_detect',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'ped_x3_fall',
                       'logical_asset_id': 'pedestrian.cityops.basic.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_x3_detect',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Cross-layer scenario has at least five causal steps',
                       'min_count': 5,
                       'rule': 'cross_layer_event_chain_min'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'cross_layer',
 'description': 'Pedestrian fall to emergency response chain',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'pedestrian.cityops.basic.v1',
               'entity_id': 'ped_x3_fall',
               'initial_pos_enu': [7114.373, 6426.431, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 117.638],
               'movement_waypoints': [],
               'visual_state': {'mode': 'walking'}},
              {'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_x3_detect',
               'initial_pos_enu': [7059.0, 6450.0, 32],
               'initial_rotation_deg': [0.0, 0.0, 60],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'vehicle.emergency.ambulance.v1',
               'entity_id': 'ambulance_x3',
               'initial_pos_enu': [7157.581, 6423.559, 0.0],
               'initial_rotation_deg': [0.0, 0.0, -140.031],
               'movement_waypoints': [],
               'visual_state': {'lights_on': True, 'mode': 'response'}},
              {'asset_id': 'prop.incident.police_tape.v1',
               'entity_id': 'police_tape_x3',
               'initial_pos_enu': [7114.107, 6426.292, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': {'mode': 'staged'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'play_x3_ped_fall',
                                          'animation_path': '/AeroSimHost/Animations/AM_Crouch',
                                          'loop_count': 1,
                                          'ped_id': 'ped_x3_fall',
                                          'play_rate': 1.0,
                                          'start_section': ''},
                               'type': 'play_animation'}],
                  'event_id': 'pedestrian_fall',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'critical',
                  'log_target_ids': ['ped_x3_fall'],
                  'log_title': 'Pedestrian falls',
                  'log_topic': 'evt_X3_pedestrian_fall_to_emergency_response_pedestrian_fall',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 220, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_x3_uav_detect',
                                          'entity_id': 'uav_x3_detect',
                                          'velocity_mps': 5.0,
                                          'waypoints_enu_m': [[7059.0, 6450.0, 32], [7114.373, 6426.431, 16]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_x3_detection', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'uav_detection',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_x3_detect', 'ped_x3_fall'],
                  'log_title': 'UAV detects pedestrian fall',
                  'log_topic': 'evt_X3_pedestrian_fall_to_emergency_response_uav_detection',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'pedestrian_fall', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_x3_ambulance_dispatch',
                                          'entity_id': 'ambulance_x3',
                                          'velocity_mps': 12.0,
                                          'waypoints_enu_m': [[7157.581, 6423.559, 0.0], [7117.119, 6427.869, 0.0]]},
                               'type': 'move_entity'}],
                  'event_id': 'ambulance_dispatch',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'warning',
                  'log_target_ids': ['ambulance_x3', 'ped_x3_fall'],
                  'log_title': 'Ambulance is dispatched',
                  'log_topic': 'evt_X3_pedestrian_fall_to_emergency_response_ambulance_dispatch',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'uav_detection', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'spawn_x3_police_tape',
                                          'asset_id': 'prop.incident.police_tape.v1',
                                          'entity_id': 'police_tape_x3',
                                          'position_enu_m': [7114.107, 6426.292, 0.0],
                                          'rotation_deg': {'yaw_deg': 0}},
                               'type': 'spawn_entity'}],
                  'event_id': 'isolation_setup',
                  'log_category': 'dynamic_constraint',
                  'log_overlay': 'dynamic_constraint',
                  'log_severity': 'warning',
                  'log_target_ids': ['police_tape_x3', 'ped_x3_fall'],
                  'log_title': 'Responder sets incident isolation tape',
                  'log_topic': 'evt_X3_pedestrian_fall_to_emergency_response_isolation_setup',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'ambulance_dispatch', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'capture_x3_response', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'emergency_documented',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_x3_detect', 'ambulance_x3', 'ped_x3_fall'],
                  'log_title': 'Emergency response documented',
                  'log_topic': 'evt_X3_pedestrian_fall_to_emergency_response_emergency_documented',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 5,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'isolation_setup', 'type': 'event_fired_after'}}],
 'parameters': {'cross_layer': True},
 'scenario_id': 'X3_pedestrian_fall_to_emergency_response'}


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
