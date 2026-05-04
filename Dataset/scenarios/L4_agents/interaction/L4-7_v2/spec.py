"""Concrete ScenarioSpec for L4-7_v2.

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
              'placement': {'position_enu_m': [7109.887, 6319.54, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Pedestrian fall detected by UAV and response chain',
 'entities': [{'activation_tick': 0,
               'category': 'pedestrian',
               'entity_id': 'ped_fall_l4_7_v2',
               'initial_state': {'mode': 'walking'},
               'logical_asset_id': 'pedestrian.cityops.basic.v1',
               'placement': {'lane_edge_id': 'cg_edge_122',
                             'lane_half_width_m': 1.9,
                             'longitudinal_s': 241.0,
                             'offset_from_curb_m': 1.2,
                             'placement_semantics': 'sidewalk_or_plaza',
                             'resolved_lateral_from_center_m': -3.1,
                             'resolved_position_enu_m': [7124.743, 6334.783, 0.0],
                             'rotation_deg': {'yaw_deg': 23.381},
                             'source_lane_edge_id_hint': 'cg_edge_23',
                             'source_longitudinal_s_hint': 42},
               'placement_mode': 'sidewalk_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_detect_l4_7_v2',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7110.5, 6319.0, 31],
                             'resolved_position_enu_m': [7110.5, 6319.0, 31],
                             'rotation_deg': {'yaw_deg': 55}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'ambulance_fall_l4_7_v2',
               'initial_state': {'lights_on': True, 'mode': 'response'},
               'logical_asset_id': 'vehicle.emergency.ambulance.v1',
               'placement': {'edge_id': 'cg_edge_204',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 33.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7094.417, 6304.837, 0.0],
                             'rotation_deg': {'yaw_deg': 21.536},
                             'source_edge_id_hint': 'cg_edge_24',
                             'source_longitudinal_s_hint': 30},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-7_v2',
 'spawn_sequencing': [{'entity_id': 'ped_fall_l4_7_v2', 'tick': 0},
                      {'entity_id': 'uav_detect_l4_7_v2', 'tick': 0},
                      {'entity_id': 'ambulance_fall_l4_7_v2', 'tick': 0}],
 'validation_rules': [{'description': 'ped_fall_l4_7_v2 is declared before event_script references it in L4-7_v2',
                       'entity_id': 'ped_fall_l4_7_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'uav_detect_l4_7_v2 is declared before event_script references it in L4-7_v2',
                       'entity_id': 'uav_detect_l4_7_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'ped_fall_l4_7_v2',
                       'logical_asset_id': 'pedestrian.cityops.basic.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_detect_l4_7_v2',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'Pedestrian fall detected by UAV and response chain',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'pedestrian.cityops.basic.v1',
               'entity_id': 'ped_fall_l4_7_v2',
               'initial_pos_enu': [7124.743, 6334.783, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 23.381],
               'movement_waypoints': [],
               'visual_state': {'mode': 'walking'}},
              {'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_detect_l4_7_v2',
               'initial_pos_enu': [7110.5, 6319.0, 31],
               'initial_rotation_deg': [0.0, 0.0, 55],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'vehicle.emergency.ambulance.v1',
               'entity_id': 'ambulance_fall_l4_7_v2',
               'initial_pos_enu': [7094.417, 6304.837, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 21.536],
               'movement_waypoints': [],
               'visual_state': {'lights_on': True, 'mode': 'response'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'play_pedestrian_fall',
                                          'animation_path': '/AeroSimHost/Animations/AM_Crouch',
                                          'loop_count': 1,
                                          'ped_id': 'ped_fall_l4_7_v2',
                                          'play_rate': 1.0,
                                          'start_section': ''},
                               'type': 'play_animation'}],
                  'event_id': 'pedestrian_fall',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'critical',
                  'log_target_ids': ['ped_fall_l4_7_v2'],
                  'log_title': 'Pedestrian fall animation starts',
                  'log_topic': 'evt_L4-7_v2_pedestrian_fall',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 230, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_uav_detect_fall',
                                          'entity_id': 'uav_detect_l4_7_v2',
                                          'velocity_mps': 5.0,
                                          'waypoints_enu_m': [[7110.5, 6319.0, 31], [7124.743, 6334.783, 16]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_fall_detection', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'uav_detects_fall',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_detect_l4_7_v2', 'ped_fall_l4_7_v2'],
                  'log_title': 'UAV detects fallen pedestrian',
                  'log_topic': 'evt_L4-7_v2_uav_detects_fall',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'pedestrian_fall', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_ambulance_to_fall',
                                          'entity_id': 'ambulance_fall_l4_7_v2',
                                          'velocity_mps': 12.0,
                                          'waypoints_enu_m': [[7094.417, 6304.837, 0.0], [7123.513, 6337.628, 0.0]]},
                               'type': 'move_entity'}],
                  'event_id': 'ambulance_response',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'warning',
                  'log_target_ids': ['ambulance_fall_l4_7_v2', 'ped_fall_l4_7_v2'],
                  'log_title': 'Ambulance responds to detected fall',
                  'log_topic': 'evt_L4-7_v2_ambulance_response',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'uav_detects_fall', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'capture_fall_response', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'incident_documented',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_detect_l4_7_v2', 'ambulance_fall_l4_7_v2', 'ped_fall_l4_7_v2'],
                  'log_title': 'UAV documents emergency response',
                  'log_topic': 'evt_L4-7_v2_incident_documented',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'ambulance_response', 'type': 'event_fired_after'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-7_v2'}


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
