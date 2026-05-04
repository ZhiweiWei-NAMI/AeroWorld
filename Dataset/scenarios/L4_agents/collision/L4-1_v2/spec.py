"""Concrete ScenarioSpec for L4-1_v2.

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
              'placement': {'position_enu_m': [46.5, 91.0, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'UAV-UAV converging conflict and separation',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_conflict_l4_1_v2_a',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [27.5, 85.0, 25], 'rotation_deg': {'yaw_deg': 70}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_conflict_l4_1_v2_b',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.airsim.flying_pawn.v1',
               'placement': {'position_enu_m': [65.5, 97.0, 30], 'rotation_deg': {'yaw_deg': 250}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-1_v2',
 'spawn_sequencing': [{'entity_id': 'uav_conflict_l4_1_v2_a', 'tick': 0},
                      {'entity_id': 'uav_conflict_l4_1_v2_b', 'tick': 0}],
 'validation_rules': [{'description': 'uav_conflict_l4_1_v2_a is declared before event_script references it in '
                                      'L4-1_v2',
                       'entity_id': 'uav_conflict_l4_1_v2_a',
                       'rule': 'entity_resolvable'},
                      {'description': 'uav_conflict_l4_1_v2_b is declared before event_script references it in '
                                      'L4-1_v2',
                       'entity_id': 'uav_conflict_l4_1_v2_b',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_conflict_l4_1_v2_a',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_conflict_l4_1_v2_b',
                       'logical_asset_id': 'uav.airsim.flying_pawn.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'UAV starts are separated and altitudes differ',
                       'min_horizontal_distance_m': 15.0,
                       'rule': 'uav_start_separation'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'UAV-UAV converging conflict and separation',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_conflict_l4_1_v2_a',
               'initial_pos_enu': [27.5, 85.0, 25],
               'initial_rotation_deg': [0.0, 0.0, 70],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'uav.airsim.flying_pawn.v1',
               'entity_id': 'uav_conflict_l4_1_v2_b',
               'initial_pos_enu': [65.5, 97.0, 30],
               'initial_rotation_deg': [0.0, 0.0, 250],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'move_uav_a_converge',
                                          'entity_id': 'uav_conflict_l4_1_v2_a',
                                          'velocity_mps': 8.0,
                                          'waypoints_enu_m': [[27.5, 85.0, 25], [42.5, 89.0, 25], [47.5, 91.0, 28]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_uav_b_converge',
                                          'entity_id': 'uav_conflict_l4_1_v2_b',
                                          'velocity_mps': 7.5,
                                          'waypoints_enu_m': [[65.5, 97.0, 30], [53.5, 93.0, 30], [47.5, 91.0, 28]]},
                               'type': 'move_entity'}],
                  'event_id': 'converging_approach',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_conflict_l4_1_v2_a', 'uav_conflict_l4_1_v2_b'],
                  'log_title': 'Two UAVs converge from separated starts',
                  'log_topic': 'evt_L4-1_v2_converging_approach',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 240, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'set_uav_a_hover',
                                          'entity_id': 'uav_conflict_l4_1_v2_a',
                                          'visual_state': {'mode': 'hover'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_uav_b_altitude_reroute',
                                          'entity_id': 'uav_conflict_l4_1_v2_b',
                                          'velocity_mps': 9.0,
                                          'waypoints_enu_m': [[47.5, 91.0, 28], [59.5, 103.0, 36]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_uav_conflict', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'uav_conflict_resolution',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_conflict_l4_1_v2_a', 'uav_conflict_l4_1_v2_b'],
                  'log_title': 'UAV-UAV proximity conflict triggers separation',
                  'log_topic': 'evt_L4-1_v2_uav_conflict_resolution',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'distance_m': 8.0,
                              'entity_a': 'uav_conflict_l4_1_v2_a',
                              'entity_b': 'uav_conflict_l4_1_v2_b',
                              'min_true_ticks': 2,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity'}},
                 {'actions': [{'params': {'action_id': 'move_uav_a_resume',
                                          'entity_id': 'uav_conflict_l4_1_v2_a',
                                          'velocity_mps': 6.0,
                                          'waypoints_enu_m': [[47.5, 91.0, 28], [29.5, 105.0, 27]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_uav_b_resume',
                                          'entity_id': 'uav_conflict_l4_1_v2_b',
                                          'velocity_mps': 7.0,
                                          'waypoints_enu_m': [[59.5, 103.0, 36], [71.5, 107.0, 32]]},
                               'type': 'move_entity'}],
                  'event_id': 'resume_patrols',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_conflict_l4_1_v2_a', 'uav_conflict_l4_1_v2_b'],
                  'log_title': 'UAVs separate and resume patrol',
                  'log_topic': 'evt_L4-1_v2_resume_patrols',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'event_ref': 'uav_conflict_resolution', 'type': 'event_fired'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-1_v2'}


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
