"""Concrete ScenarioSpec for L4-5_v2.

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
              'placement': {'position_enu_m': [6991.798, 6239.602, 180.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'UAV low-altitude pedestrian near-miss',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_ped_nearmiss_l4_5_v2',
               'initial_state': {'mode': 'preflight_on_pad'},
               'lifecycle': {'auto_lifecycle_applied': True,
                             'home_hover_enu_m': [7085.829, 6310.459, 3.0],
                             'home_pad_entity_id': 'pad_home_uav_ped_nearmiss_l4_5_v2',
                             'mission_start_enu_m': [7088.0, 6304.0, 30],
                             'requires_landing_or_terminal_resolution': True,
                             'requires_takeoff': True},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7085.829, 6310.459, 3.0],
                             'resolved_position_enu_m': [7085.829, 6310.459, 3.0],
                             'rotation_deg': {'yaw_deg': 45}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': [[7088.0, 6304.0, 30]]},
              {'activation_tick': 0,
               'category': 'pedestrian',
               'entity_id': 'ped_nearmiss_l4_5_v2',
               'initial_state': {'activity_type': 'waiting',
                                 'animation_hint': 'pedestrian_idle',
                                 'mode': 'waiting',
                                 'posture': 'standing',
                                 'social_state': 'solo'},
               'logical_asset_id': 'pedestrian.cityops.basic.v1',
               'placement': {'lane_edge_id': 'cg_edge_21',
                             'lane_half_width_m': 1.9,
                             'longitudinal_s': 35.0,
                             'offset_from_curb_m': 1.2,
                             'placement_semantics': 'sidewalk',
                             'resolved_lateral_from_center_m': -3.1,
                             'resolved_position_enu_m': [6857.804, 6138.514, 0.0],
                             'rotation_deg': {'yaw_deg': 121.281},
                             'source_lane_edge_id_hint': 'cg_edge_21',
                             'source_longitudinal_s_hint': 35},
               'placement_mode': 'sidewalk_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'facility',
               'entity_id': 'pad_home_uav_ped_nearmiss_l4_5_v2',
               'initial_state': {'mode': 'available'},
               'logical_asset_id': 'facility.landing_pad.visible.v1',
               'placement': {'approach_side': 'departure',
                             'pad_instance_id': 'home_uav_ped_nearmiss_l4_5_v2',
                             'position_enu_m': [7085.829, 6310.459, 0.0],
                             'resolved_position_enu_m': [7085.829, 6310.459, 0.0],
                             'rotation_deg': {'yaw_deg': 21.914}},
               'placement_mode': 'pad_anchor',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [6991.798, 6239.602, 0.0],
                  'radius_m': 329.786,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-5_v2',
 'spawn_sequencing': [{'entity_id': 'uav_ped_nearmiss_l4_5_v2', 'tick': 0},
                      {'entity_id': 'ped_nearmiss_l4_5_v2', 'tick': 0},
                      {'entity_id': 'pad_home_uav_ped_nearmiss_l4_5_v2', 'tick': 0}],
 'validation_rules': [{'description': 'uav_ped_nearmiss_l4_5_v2 is declared before event_script references it in '
                                      'L4-5_v2',
                       'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'ped_nearmiss_l4_5_v2 is declared before event_script references it in L4-5_v2',
                       'entity_id': 'ped_nearmiss_l4_5_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'ped_nearmiss_l4_5_v2',
                       'logical_asset_id': 'pedestrian.cityops.basic.v1',
                       'rule': 'asset_in_catalog'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'UAV low-altitude pedestrian near-miss',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_ped_nearmiss_l4_5_v2',
               'initial_pos_enu': [7085.829, 6310.459, 3.0],
               'initial_rotation_deg': [0.0, 0.0, 45],
               'movement_waypoints': [[7088.0, 6304.0, 30]],
               'visual_state': {'mode': 'preflight_on_pad'}},
              {'asset_id': 'pedestrian.cityops.basic.v1',
               'entity_id': 'ped_nearmiss_l4_5_v2',
               'initial_pos_enu': [6857.804, 6138.514, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 121.281],
               'movement_waypoints': [],
               'visual_state': {'activity_type': 'waiting',
                                'animation_hint': 'pedestrian_idle',
                                'mode': 'waiting',
                                'posture': 'standing',
                                'social_state': 'solo'}},
              {'asset_id': 'facility.landing_pad.visible.v1',
               'entity_id': 'pad_home_uav_ped_nearmiss_l4_5_v2',
               'initial_pos_enu': [7085.829, 6310.459, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 21.914],
               'movement_waypoints': [],
               'visual_state': {'mode': 'available'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'move_uav_ped_nearmiss_l4_5_v2_takeoff_entry',
                                          'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                                          'velocity_mps': 5.0,
                                          'waypoints_enu_m': [[7085.829, 6310.459, 3.0],
                                                              [7085.829, 6310.459, 24.0],
                                                              [7088.0, 6304.0, 30]]},
                               'type': 'move_entity'}],
                  'event_id': 'lifecycle_takeoff_uav_ped_nearmiss_l4_5_v2',
                  'log_category': 'uav_lifecycle',
                  'log_overlay': 'uav_lifecycle',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_ped_nearmiss_l4_5_v2', 'pad_home_uav_ped_nearmiss_l4_5_v2'],
                  'log_title': 'UAV takes off from visible home pad and enters mission airspace',
                  'log_topic': 'evt_L4-5_v2_lifecycle_takeoff_uav_ped_nearmiss_l4_5_v2',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 0,
                  'trigger': {'tick': 30, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'set_ped_phone_distraction',
                                          'activity_type': 'phone_call',
                                          'entity_id': 'ped_nearmiss_l4_5_v2'},
                               'type': 'set_pedestrian_activity'},
                              {'params': {'action_id': 'move_uav_ped_descent',
                                          'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                                          'velocity_mps': 4.5,
                                          'waypoints_enu_m': [[7088.0, 6304.0, 30],
                                                              [7094.0, 6310.0, 15],
                                                              [6857.804, 6138.514, 5]]},
                               'type': 'move_entity'}],
                  'event_id': 'uav_low_descent',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_ped_nearmiss_l4_5_v2', 'ped_nearmiss_l4_5_v2'],
                  'log_title': 'UAV descends toward pedestrian head height while pedestrian is distracted by a phone '
                               'call',
                  'log_topic': 'evt_L4-5_v2_uav_low_descent',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 230, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'set_uav_hover_nearmiss',
                                          'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                                          'visual_state': {'mode': 'hover'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_uav_pull_up_after_nearmiss',
                                          'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                                          'velocity_mps': 6.0,
                                          'waypoints_enu_m': [[6857.804, 6138.514, 5], [6849.804, 6146.514, 24]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_ped_nearmiss', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'pedestrian_near_miss',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_ped_nearmiss_l4_5_v2', 'ped_nearmiss_l4_5_v2'],
                  'log_title': 'UAV near-miss with pedestrian triggers pull-up',
                  'log_topic': 'evt_L4-5_v2_pedestrian_near_miss',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'distance_m': 5.0,
                              'entity_a': 'uav_ped_nearmiss_l4_5_v2',
                              'entity_b': 'ped_nearmiss_l4_5_v2',
                              'horizontal_distance_m': 3.0,
                              'metric': 'xy_plus_z',
                              'min_true_ticks': 2,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity',
                              'vertical_distance_m': 6.0}},
                 {'actions': [{'params': {'action_id': 'move_ped_clear_nearmiss',
                                          'activity_type': 'walking',
                                          'entity_id': 'ped_nearmiss_l4_5_v2',
                                          'post_activity_type': 'waiting',
                                          'velocity_mps': 1.5,
                                          'waypoints_enu_m': [[6857.804, 6138.514, 0.0], [6856.76, 6141.388, 0.0]]},
                               'type': 'move_entity'}],
                  'event_id': 'pedestrian_clears_area',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'info',
                  'log_target_ids': ['ped_nearmiss_l4_5_v2'],
                  'log_title': 'Pedestrian clears UAV operating area',
                  'log_topic': 'evt_L4-5_v2_pedestrian_clears_area',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 41, 'event_ref': 'pedestrian_near_miss', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_uav_ped_nearmiss_l4_5_v2_landing_return',
                                          'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                                          'velocity_mps': 4.0,
                                          'waypoints_enu_m': [[6849.804, 6146.514, 24.0],
                                                              [7085.829, 6310.459, 22.0],
                                                              [7085.829, 6310.459, 3.0]]},
                               'type': 'move_entity'}],
                  'event_id': 'lifecycle_landing_uav_ped_nearmiss_l4_5_v2',
                  'log_category': 'uav_lifecycle',
                  'log_overlay': 'uav_lifecycle',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_ped_nearmiss_l4_5_v2', 'pad_home_uav_ped_nearmiss_l4_5_v2'],
                  'log_title': 'UAV returns to the visible home pad and lands after mission resolution',
                  'log_topic': 'evt_L4-5_v2_lifecycle_landing_uav_ped_nearmiss_l4_5_v2',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 9,
                  'trigger': {'delay_ticks': 80,
                              'event_ref': 'pedestrian_clears_area',
                              'type': 'event_fired_after'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-5_v2'}


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
