"""Concrete ScenarioSpec for L4-2_v2.

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
              'placement': {'position_enu_m': [48.0, 90.0, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'UAV building facade near strike',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_facade_l4_2_v2',
               'initial_state': {'mode': 'navigation_fault'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [33.5, 82.0, 31], 'rotation_deg': {'yaw_deg': 55}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'facade_anchor',
               'entity_id': 'facade_anchor_l4_2_v2',
               'initial_state': {},
               'logical_asset_id': 'semantic.asset_anchor',
               'placement': {'building_id': 'building_east_block_3',
                             'outward_normal_enu': [1.0, -0.4, 0.0],
                             'position_enu_m': [62.5, 98.0, 16],
                             'stand_off_m': 1.0},
               'placement_mode': 'facade_anchor',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-2_v2',
 'spawn_sequencing': [{'entity_id': 'uav_facade_l4_2_v2', 'tick': 0},
                      {'entity_id': 'facade_anchor_l4_2_v2', 'tick': 0}],
 'validation_rules': [{'description': 'uav_facade_l4_2_v2 is declared before event_script references it in L4-2_v2',
                       'entity_id': 'uav_facade_l4_2_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'facade_anchor_l4_2_v2 is declared before event_script references it in '
                                      'L4-2_v2',
                       'entity_id': 'facade_anchor_l4_2_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_facade_l4_2_v2',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'facade_anchor_l4_2_v2',
                       'logical_asset_id': 'semantic.asset_anchor',
                       'rule': 'asset_in_catalog'},
                      {'description': 'UAV trajectory nearest point is within 3m of facade anchor',
                       'max_distance_m': 3.0,
                       'rule': 'facade_approach_distance'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'UAV building facade near strike',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_facade_l4_2_v2',
               'initial_pos_enu': [33.5, 82.0, 31],
               'initial_rotation_deg': [0.0, 0.0, 55],
               'movement_waypoints': [],
               'visual_state': {'mode': 'navigation_fault'}},
              {'asset_id': 'semantic.asset_anchor',
               'entity_id': 'facade_anchor_l4_2_v2',
               'initial_pos_enu': [62.5, 98.0, 16],
               'initial_rotation_deg': [0.0, 0.0, 0.0],
               'movement_waypoints': [],
               'visual_state': None}],
 'event_chain': [{'actions': [{'params': {'action_id': 'move_uav_facade_approach',
                                          'entity_id': 'uav_facade_l4_2_v2',
                                          'velocity_mps': 7.0,
                                          'waypoints_enu_m': [[33.5, 82.0, 31], [46.5, 89.0, 26], [60.5, 97.0, 18]]},
                               'type': 'move_entity'}],
                  'event_id': 'facade_approach',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_facade_l4_2_v2', 'facade_anchor_l4_2_v2'],
                  'log_title': 'UAV approaches building facade on faulty route',
                  'log_topic': 'evt_L4-2_v2_facade_approach',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 250, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_uav_facade_evasion',
                                          'entity_id': 'uav_facade_l4_2_v2',
                                          'velocity_mps': 10.0,
                                          'waypoints_enu_m': [[60.5, 97.0, 18], [54.5, 105.0, 34]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_facade_evasion', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'facade_proximity',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_facade_l4_2_v2', 'facade_anchor_l4_2_v2'],
                  'log_title': 'Emergency evasion near facade',
                  'log_topic': 'evt_L4-2_v2_facade_proximity',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'distance_m': 8.0,
                              'entity_a': 'uav_facade_l4_2_v2',
                              'entity_b': 'facade_anchor_l4_2_v2',
                              'min_true_ticks': 3,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity'}},
                 {'actions': [{'params': {'action_id': 'move_uav_facade_recover',
                                          'entity_id': 'uav_facade_l4_2_v2',
                                          'velocity_mps': 7.0,
                                          'waypoints_enu_m': [[54.5, 105.0, 34], [36.5, 109.0, 32]]},
                               'type': 'move_entity'}],
                  'event_id': 'facade_recovery',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_facade_l4_2_v2'],
                  'log_title': 'UAV recovers after facade near strike',
                  'log_topic': 'evt_L4-2_v2_facade_recovery',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'event_ref': 'facade_proximity', 'type': 'event_fired'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-2_v2'}


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
