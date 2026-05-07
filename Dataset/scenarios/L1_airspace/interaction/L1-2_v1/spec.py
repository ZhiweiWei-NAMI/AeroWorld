"""Concrete ScenarioSpec for L1-2_v1.

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
              'placement': {'position_enu_m': [7062.692, 6224.281, 80.231],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Altitude deviation from assigned corridor',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_l1_2_v1_primary',
               'initial_state': {'mode': 'preflight_on_pad'},
               'lifecycle': {'auto_lifecycle_applied': True,
                             'home_hover_enu_m': [7060.141, 6205.438, 3.0],
                             'home_pad_entity_id': 'pad_home_uav_l1_2_v1_primary',
                             'mission_start_enu_m': [7055.0, 6220.0, 34],
                             'requires_landing_or_terminal_resolution': True,
                             'requires_takeoff': True},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7060.141, 6205.438, 3.0],
                             'resolved_position_enu_m': [7060.141, 6205.438, 3.0],
                             'rotation_deg': {'yaw_deg': 35}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': [[7055.0, 6220.0, 34], [7074.0, 6230.0, 34], [7050.0, 6245.0, 34]]},
              {'activation_tick': 0,
               'category': 'airspace_constraint',
               'entity_id': 'nfz_l1_2_v1',
               'initial_state': {},
               'logical_asset_id': 'trigger.hazard.generic.box.v1',
               'placement': {'center_enu_m': [7081.0, 6234.0, 28],
                             'extent_m': [14.0, 10.0, 14.0],
                             'resolved_position_enu_m': [7081.0, 6234.0, 28]},
               'placement_mode': 'box_volume',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'facility',
               'entity_id': 'pad_home_uav_l1_2_v1_primary',
               'initial_state': {'mode': 'available'},
               'logical_asset_id': 'facility.landing_pad.visible.v1',
               'placement': {'approach_side': 'departure',
                             'pad_instance_id': 'home_uav_l1_2_v1_primary',
                             'position_enu_m': [7060.141, 6205.438, 0.0],
                             'resolved_position_enu_m': [7060.141, 6205.438, 0.0],
                             'rotation_deg': {'yaw_deg': -161.35}},
               'placement_mode': 'pad_anchor',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7062.692, 6224.281, 0.0],
                  'radius_m': 184.297,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L1-2_v1',
 'spawn_sequencing': [{'entity_id': 'uav_l1_2_v1_primary', 'tick': 0},
                      {'entity_id': 'nfz_l1_2_v1', 'tick': 0},
                      {'entity_id': 'pad_home_uav_l1_2_v1_primary', 'tick': 0}],
 'validation_rules': [{'description': 'uav_l1_2_v1_primary is declared before event_script references it in L1-2_v1',
                       'entity_id': 'uav_l1_2_v1_primary',
                       'rule': 'entity_resolvable'},
                      {'description': 'nfz_l1_2_v1 is declared before event_script references it in L1-2_v1',
                       'entity_id': 'nfz_l1_2_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_l1_2_v1_primary',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'nfz_l1_2_v1',
                       'logical_asset_id': 'trigger.hazard.generic.box.v1',
                       'rule': 'asset_in_catalog'},
                      {'constraint_id': 'nfz_l1_2_v1',
                       'description': 'UAV start is outside the no-fly or hazard volume',
                       'entity_id': 'uav_l1_2_v1_primary',
                       'rule': 'uav_start_outside_constraint'},
                      {'description': 'Airspace scenario must include approach, conflict, and recovery',
                       'min_count': 3,
                       'rule': 'event_chain_min'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'airspace',
 'description': 'Altitude deviation from assigned corridor',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_l1_2_v1_primary',
               'initial_pos_enu': [7060.141, 6205.438, 3.0],
               'initial_rotation_deg': [0.0, 0.0, 35],
               'movement_waypoints': [[7055.0, 6220.0, 34], [7074.0, 6230.0, 34], [7050.0, 6245.0, 34]],
               'visual_state': {'mode': 'preflight_on_pad'}},
              {'asset_id': 'trigger.hazard.generic.box.v1',
               'entity_id': 'nfz_l1_2_v1',
               'initial_pos_enu': [7081.0, 6234.0, 28],
               'initial_rotation_deg': [0.0, 0.0, 0.0],
               'movement_waypoints': [],
               'visual_state': None},
              {'asset_id': 'facility.landing_pad.visible.v1',
               'entity_id': 'pad_home_uav_l1_2_v1_primary',
               'initial_pos_enu': [7060.141, 6205.438, 0.0],
               'initial_rotation_deg': [0.0, 0.0, -161.35],
               'movement_waypoints': [],
               'visual_state': {'mode': 'available'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'move_uav_l1_2_v1_primary_takeoff_entry',
                                          'entity_id': 'uav_l1_2_v1_primary',
                                          'velocity_mps': 5.0,
                                          'waypoints_enu_m': [[7060.141, 6205.438, 3.0],
                                                              [7060.141, 6205.438, 24.0],
                                                              [7055.0, 6220.0, 34]]},
                               'type': 'move_entity'}],
                  'event_id': 'lifecycle_takeoff_uav_l1_2_v1_primary',
                  'log_category': 'uav_lifecycle',
                  'log_overlay': 'uav_lifecycle',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l1_2_v1_primary', 'pad_home_uav_l1_2_v1_primary'],
                  'log_title': 'UAV takes off from visible home pad and enters mission airspace',
                  'log_topic': 'evt_L1-2_v1_lifecycle_takeoff_uav_l1_2_v1_primary',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 0,
                  'trigger': {'tick': 30, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_boundary_approach',
                                          'entity_id': 'uav_l1_2_v1_primary',
                                          'velocity_mps': 8.0,
                                          'waypoints_enu_m': [[7055.0, 6220.0, 34],
                                                              [7070.0, 6228.0, 34],
                                                              [7074.0, 6230.0, 34]]},
                               'type': 'move_entity'}],
                  'event_id': 'approach_boundary',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l1_2_v1_primary', 'nfz_l1_2_v1'],
                  'log_title': 'UAV approaches constrained airspace',
                  'log_topic': 'evt_L1-2_v1_approach_boundary',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 220, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_altitude_recover',
                                          'entity_id': 'uav_l1_2_v1_primary',
                                          'velocity_mps': 4.0,
                                          'waypoints_enu_m': [[7074.0, 6230.0, 34],
                                                              [7071.0, 6234.0, 42],
                                                              [7066.0, 6238.0, 34]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'hover_boundary_hold',
                                          'entity_id': 'uav_l1_2_v1_primary',
                                          'visual_state': {'mode': 'hover'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'boundary_conflict',
                  'log_category': 'airspace',
                  'log_overlay': 'airspace',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_l1_2_v1_primary', 'nfz_l1_2_v1'],
                  'log_title': 'Altitude deviation from assigned corridor',
                  'log_topic': 'evt_L1-2_v1_boundary_conflict',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'distance_m': 14.0,
                              'entity_a': 'uav_l1_2_v1_primary',
                              'entity_b': 'nfz_l1_2_v1',
                              'metric': '3d',
                              'min_true_ticks': 3,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity'}},
                 {'actions': [{'params': {'action_id': 'move_boundary_return_safe',
                                          'entity_id': 'uav_l1_2_v1_primary',
                                          'velocity_mps': 9.0,
                                          'waypoints_enu_m': [[7068.0, 6234.0, 38], [7050.0, 6245.0, 34]]},
                               'type': 'move_entity'}],
                  'event_id': 'return_safe_airspace',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l1_2_v1_primary', 'nfz_l1_2_v1'],
                  'log_title': 'UAV returns to safe airspace',
                  'log_topic': 'evt_L1-2_v1_return_safe_airspace',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 55, 'event_ref': 'boundary_conflict', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_uav_l1_2_v1_primary_landing_return',
                                          'entity_id': 'uav_l1_2_v1_primary',
                                          'velocity_mps': 4.0,
                                          'waypoints_enu_m': [[7050.0, 6245.0, 34.0],
                                                              [7060.141, 6205.438, 22.0],
                                                              [7060.141, 6205.438, 3.0]]},
                               'type': 'move_entity'}],
                  'event_id': 'lifecycle_landing_uav_l1_2_v1_primary',
                  'log_category': 'uav_lifecycle',
                  'log_overlay': 'uav_lifecycle',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l1_2_v1_primary', 'pad_home_uav_l1_2_v1_primary'],
                  'log_title': 'UAV returns to the visible home pad and lands after mission resolution',
                  'log_topic': 'evt_L1-2_v1_lifecycle_landing_uav_l1_2_v1_primary',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 9,
                  'trigger': {'delay_ticks': 80, 'event_ref': 'return_safe_airspace', 'type': 'event_fired_after'}}],
 'parameters': {'approach_tick': 220, 'conflict_distance_m': 14.0, 'resolution_tick': 420},
 'scenario_id': 'L1-2_v1'}


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
