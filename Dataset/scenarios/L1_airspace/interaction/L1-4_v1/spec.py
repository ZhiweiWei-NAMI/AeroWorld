"""Concrete ScenarioSpec for L1-4_v1.

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
              'placement': {'position_enu_m': [85.5, 28.5, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Corridor congestion with multi-UAV separation',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_l1_4_v1_primary',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [80.5, 32.0, 34], 'rotation_deg': {'yaw_deg': 35}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': [[99.5, 42.0, 34], [75.5, 57.0, 34]]},
              {'activation_tick': 0,
               'category': 'airspace_constraint',
               'entity_id': 'nfz_l1_4_v1',
               'initial_state': {},
               'logical_asset_id': 'trigger.hazard.generic.box.v1',
               'placement': {'center_enu_m': [106.5, 46.0, 28], 'extent_m': [14.0, 10.0, 14.0]},
               'placement_mode': 'box_volume',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_l1_4_v1_secondary',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.airsim.cv_pawn.v1',
               'placement': {'position_enu_m': [86.5, 14.0, 32], 'rotation_deg': {'yaw_deg': 20}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_l1_4_v1_tertiary',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.airsim.flying_pawn.v1',
               'placement': {'position_enu_m': [68.5, 22.0, 34], 'rotation_deg': {'yaw_deg': 45}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L1-4_v1',
 'spawn_sequencing': [{'entity_id': 'uav_l1_4_v1_primary', 'tick': 0},
                      {'entity_id': 'nfz_l1_4_v1', 'tick': 0},
                      {'entity_id': 'uav_l1_4_v1_secondary', 'tick': 0},
                      {'entity_id': 'uav_l1_4_v1_tertiary', 'tick': 0}],
 'validation_rules': [{'description': 'uav_l1_4_v1_primary is declared before event_script references it in L1-4_v1',
                       'entity_id': 'uav_l1_4_v1_primary',
                       'rule': 'entity_resolvable'},
                      {'description': 'nfz_l1_4_v1 is declared before event_script references it in L1-4_v1',
                       'entity_id': 'nfz_l1_4_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_l1_4_v1_primary',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'nfz_l1_4_v1',
                       'logical_asset_id': 'trigger.hazard.generic.box.v1',
                       'rule': 'asset_in_catalog'},
                      {'constraint_id': 'nfz_l1_4_v1',
                       'description': 'UAV start is outside the no-fly or hazard volume',
                       'entity_id': 'uav_l1_4_v1_primary',
                       'rule': 'uav_start_outside_constraint'},
                      {'description': 'Airspace scenario must include approach, conflict, and recovery',
                       'min_count': 3,
                       'rule': 'event_chain_min'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'airspace',
 'description': 'Corridor congestion with multi-UAV separation',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_l1_4_v1_primary',
               'initial_pos_enu': [80.5, 32.0, 34],
               'initial_rotation_deg': [0.0, 0.0, 35],
               'movement_waypoints': [[99.5, 42.0, 34], [75.5, 57.0, 34]],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'trigger.hazard.generic.box.v1',
               'entity_id': 'nfz_l1_4_v1',
               'initial_pos_enu': [106.5, 46.0, 28],
               'initial_rotation_deg': [0.0, 0.0, 0.0],
               'movement_waypoints': [],
               'visual_state': None},
              {'asset_id': 'uav.airsim.cv_pawn.v1',
               'entity_id': 'uav_l1_4_v1_secondary',
               'initial_pos_enu': [86.5, 14.0, 32],
               'initial_rotation_deg': [0.0, 0.0, 20],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'uav.airsim.flying_pawn.v1',
               'entity_id': 'uav_l1_4_v1_tertiary',
               'initial_pos_enu': [68.5, 22.0, 34],
               'initial_rotation_deg': [0.0, 0.0, 45],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'move_boundary_approach',
                                          'entity_id': 'uav_l1_4_v1_primary',
                                          'velocity_mps': 8.0,
                                          'waypoints_enu_m': [[80.5, 32.0, 34], [95.5, 40.0, 34], [99.5, 42.0, 34]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_corridor_secondary',
                                          'entity_id': 'uav_l1_4_v1_secondary',
                                          'velocity_mps': 7.0,
                                          'waypoints_enu_m': [[86.5, 14.0, 32], [102.5, 44.0, 32]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_corridor_tertiary',
                                          'entity_id': 'uav_l1_4_v1_tertiary',
                                          'velocity_mps': 6.5,
                                          'waypoints_enu_m': [[68.5, 22.0, 34], [104.5, 47.0, 34]]},
                               'type': 'move_entity'}],
                  'event_id': 'approach_boundary',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l1_4_v1_primary', 'nfz_l1_4_v1'],
                  'log_title': 'UAV approaches constrained airspace',
                  'log_topic': 'evt_L1-4_v1_approach_boundary',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 220, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_boundary_avoidance',
                                          'entity_id': 'uav_l1_4_v1_primary',
                                          'velocity_mps': 3.0,
                                          'waypoints_enu_m': [[99.5, 42.0, 34], [93.5, 46.0, 38]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'hover_boundary_hold',
                                          'entity_id': 'uav_l1_4_v1_primary',
                                          'visual_state': {'mode': 'hover'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_corridor_reroute_secondary',
                                          'entity_id': 'uav_l1_4_v1_secondary',
                                          'velocity_mps': 8.5,
                                          'waypoints_enu_m': [[102.5, 44.0, 32], [122.5, 62.0, 38]]},
                               'type': 'move_entity'}],
                  'event_id': 'boundary_conflict',
                  'log_category': 'airspace',
                  'log_overlay': 'airspace',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_l1_4_v1_primary', 'nfz_l1_4_v1'],
                  'log_title': 'Airspace boundary conflict detected',
                  'log_topic': 'evt_L1-4_v1_boundary_conflict',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'distance_m': 14.0,
                              'entity_a': 'uav_l1_4_v1_primary',
                              'entity_b': 'nfz_l1_4_v1',
                              'min_true_ticks': 3,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity'}},
                 {'actions': [{'params': {'action_id': 'move_boundary_return_safe',
                                          'entity_id': 'uav_l1_4_v1_primary',
                                          'velocity_mps': 9.0,
                                          'waypoints_enu_m': [[93.5, 46.0, 38], [75.5, 57.0, 34]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_corridor_resume_tertiary',
                                          'entity_id': 'uav_l1_4_v1_tertiary',
                                          'velocity_mps': 7.5,
                                          'waypoints_enu_m': [[104.5, 47.0, 34], [88.5, 60.0, 34]]},
                               'type': 'move_entity'}],
                  'event_id': 'return_safe_airspace',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l1_4_v1_primary', 'nfz_l1_4_v1'],
                  'log_title': 'UAV returns to safe airspace',
                  'log_topic': 'evt_L1-4_v1_return_safe_airspace',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'event_ref': 'boundary_conflict', 'type': 'event_fired'}}],
 'parameters': {'approach_tick': 220, 'conflict_distance_m': 14.0, 'resolution_tick': 420},
 'scenario_id': 'L1-4_v1'}


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
