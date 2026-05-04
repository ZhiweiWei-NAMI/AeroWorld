"""Concrete ScenarioSpec for L4-3_v3.

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
              'placement': {'position_enu_m': [7073.115, 6299.412, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'UAV forced landing near crowd',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_forced_landing_l4_3_v3',
               'initial_state': {'mode': 'fault'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7063.0, 6288.0, 30],
                             'resolved_position_enu_m': [7063.0, 6288.0, 30],
                             'rotation_deg': {'yaw_deg': 45}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'pedestrian',
               'entity_id': 'ped_landing_l4_3_v3_a',
               'initial_state': {'mode': 'standing'},
               'logical_asset_id': 'pedestrian.cityops.basic.v1',
               'placement': {'lane_edge_id': 'cg_edge_204',
                             'lane_half_width_m': 1.9,
                             'longitudinal_s': 16.0,
                             'offset_from_curb_m': 4.5,
                             'placement_semantics': 'sidewalk_or_plaza',
                             'resolved_lateral_from_center_m': 6.4,
                             'resolved_position_enu_m': [7076.294, 6304.305, 0.0],
                             'rotation_deg': {'yaw_deg': 24.029},
                             'source_lane_edge_id_hint': 'cg_edge_19',
                             'source_longitudinal_s_hint': 24},
               'placement_mode': 'sidewalk_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'pedestrian',
               'entity_id': 'ped_landing_l4_3_v3_b',
               'initial_state': {'mode': 'standing'},
               'logical_asset_id': 'pedestrian.cityops.basic.v1',
               'placement': {'lane_edge_id': 'cg_edge_204',
                             'lane_half_width_m': 1.9,
                             'longitudinal_s': 20.0,
                             'offset_from_curb_m': 4.5,
                             'placement_semantics': 'sidewalk_or_plaza',
                             'resolved_lateral_from_center_m': 6.4,
                             'resolved_position_enu_m': [7080.052, 6305.93, 0.0],
                             'rotation_deg': {'yaw_deg': 22.837},
                             'source_lane_edge_id_hint': 'cg_edge_19',
                             'source_longitudinal_s_hint': 28},
               'placement_mode': 'sidewalk_anchor',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-3_v3',
 'spawn_sequencing': [{'entity_id': 'uav_forced_landing_l4_3_v3', 'tick': 0},
                      {'entity_id': 'ped_landing_l4_3_v3_a', 'tick': 0},
                      {'entity_id': 'ped_landing_l4_3_v3_b', 'tick': 0}],
 'validation_rules': [{'description': 'uav_forced_landing_l4_3_v3 is declared before event_script references it in '
                                      'L4-3_v3',
                       'entity_id': 'uav_forced_landing_l4_3_v3',
                       'rule': 'entity_resolvable'},
                      {'description': 'ped_landing_l4_3_v3_a is declared before event_script references it in '
                                      'L4-3_v3',
                       'entity_id': 'ped_landing_l4_3_v3_a',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_forced_landing_l4_3_v3',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'ped_landing_l4_3_v3_a',
                       'logical_asset_id': 'pedestrian.cityops.basic.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'UAV descends through low-altitude forced landing profile',
                       'rule': 'forced_landing_descent_profile',
                       'z_sequence': [30, 8, 3]}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'UAV forced landing near crowd',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_forced_landing_l4_3_v3',
               'initial_pos_enu': [7063.0, 6288.0, 30],
               'initial_rotation_deg': [0.0, 0.0, 45],
               'movement_waypoints': [],
               'visual_state': {'mode': 'fault'}},
              {'asset_id': 'pedestrian.cityops.basic.v1',
               'entity_id': 'ped_landing_l4_3_v3_a',
               'initial_pos_enu': [7076.294, 6304.305, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 24.029],
               'movement_waypoints': [],
               'visual_state': {'mode': 'standing'}},
              {'asset_id': 'pedestrian.cityops.basic.v1',
               'entity_id': 'ped_landing_l4_3_v3_b',
               'initial_pos_enu': [7080.052, 6305.93, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 22.837],
               'movement_waypoints': [],
               'visual_state': {'mode': 'standing'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_uav_forced_landing_fault',
                                          'entity_id': 'uav_forced_landing_l4_3_v3',
                                          'visual_state': {'mode': 'propulsion_fault'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'spawn_landing_crowd',
                                          'count': 8,
                                          'group_id': 'crowd_l4_3_v3',
                                          'seed': 706,
                                          'spawn_box_extent_cm': [650, 420, 0],
                                          'spawn_origin_enu_m': [7078.228, 6305.152, 0.0]},
                               'type': 'spawn_crowd'}],
                  'event_id': 'forced_landing_fault',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_forced_landing_l4_3_v3', 'ped_landing_l4_3_v3_a', 'ped_landing_l4_3_v3_b'],
                  'log_title': 'UAV fault appears near a ground crowd',
                  'log_topic': 'evt_L4-3_v3_forced_landing_fault',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 230, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_uav_forced_descent',
                                          'entity_id': 'uav_forced_landing_l4_3_v3',
                                          'velocity_mps': 3.0,
                                          'waypoints_enu_m': [[7063.0, 6288.0, 30],
                                                              [7072.0, 6296.0, 18],
                                                              [7078.173, 6305.118, 8],
                                                              [7078.173, 6305.118, 3]]},
                               'type': 'move_entity'}],
                  'event_id': 'descent_to_low_altitude',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_forced_landing_l4_3_v3'],
                  'log_title': 'UAV descends from 30m to low height',
                  'log_topic': 'evt_L4-3_v3_descent_to_low_altitude',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'forced_landing_fault', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_ped_a_evade_landing',
                                          'entity_id': 'ped_landing_l4_3_v3_a',
                                          'velocity_mps': 2.0,
                                          'waypoints_enu_m': [[7076.294, 6304.305, 0.0], [7069.151, 6308.486, 0.0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_ped_b_evade_landing',
                                          'entity_id': 'ped_landing_l4_3_v3_b',
                                          'velocity_mps': 2.0,
                                          'waypoints_enu_m': [[7080.052, 6305.93, 0.0], [7087.549, 6314.958, 0.0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_forced_landing', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'crowd_proximity_response',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'warning',
                  'log_target_ids': ['ped_landing_l4_3_v3_a', 'ped_landing_l4_3_v3_b', 'uav_forced_landing_l4_3_v3'],
                  'log_title': 'Crowd evades forced landing zone',
                  'log_topic': 'evt_L4-3_v3_crowd_proximity_response',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'distance_m': 5.0,
                              'entity_a': 'uav_forced_landing_l4_3_v3',
                              'entity_b': 'ped_landing_l4_3_v3_a',
                              'metric': '3d',
                              'min_true_ticks': 2,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity'}},
                 {'actions': [{'params': {'action_id': 'move_uav_safe_touchdown',
                                          'entity_id': 'uav_forced_landing_l4_3_v3',
                                          'velocity_mps': 1.0,
                                          'waypoints_enu_m': [[7078.173, 6305.118, 3], [7079.173, 6306.118, 0.8]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'clear_landing_crowd', 'group_id': 'crowd_l4_3_v3'},
                               'type': 'clear_crowd'}],
                  'event_id': 'safe_landing',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_forced_landing_l4_3_v3'],
                  'log_title': 'UAV completes safe landing',
                  'log_topic': 'evt_L4-3_v3_safe_landing',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'delay_ticks': 30,
                              'event_ref': 'crowd_proximity_response',
                              'type': 'event_fired_after'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-3_v3'}


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
