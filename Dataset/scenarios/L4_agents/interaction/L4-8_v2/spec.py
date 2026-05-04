"""Concrete ScenarioSpec for L4-8_v2.

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
              'placement': {'position_enu_m': [7128.02, 6329.612, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Crowd evacuation with UAV monitoring',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_crowd_monitor_l4_8_v2',
               'initial_state': {'mode': 'monitor'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7117.5, 6323.0, 34],
                             'resolved_position_enu_m': [7117.5, 6323.0, 34],
                             'rotation_deg': {'yaw_deg': 50}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'crowd_anchor',
               'entity_id': 'crowd_anchor_l4_8_v2',
               'initial_state': {},
               'logical_asset_id': 'semantic.spawn_zone',
               'placement': {'position_enu_m': [7138.539, 6336.224, 0.0],
                             'resolved_position_enu_m': [7138.539, 6336.224, 0.0],
                             'rotation_deg': {'yaw_deg': 0}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7128.02, 6329.612, 0.0],
                  'radius_m': 172.425,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-8_v2',
 'spawn_sequencing': [{'entity_id': 'uav_crowd_monitor_l4_8_v2', 'tick': 0},
                      {'entity_id': 'crowd_anchor_l4_8_v2', 'tick': 0}],
 'validation_rules': [{'description': 'uav_crowd_monitor_l4_8_v2 is declared before event_script references it in '
                                      'L4-8_v2',
                       'entity_id': 'uav_crowd_monitor_l4_8_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'crowd_anchor_l4_8_v2 is declared before event_script references it in L4-8_v2',
                       'entity_id': 'crowd_anchor_l4_8_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_crowd_monitor_l4_8_v2',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'crowd_anchor_l4_8_v2',
                       'logical_asset_id': 'semantic.spawn_zone',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Crowd evacuation scenario uses spawn_crowd with at least 8 people',
                       'min_count': 8,
                       'rule': 'crowd_spawn_count_min'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'Crowd evacuation with UAV monitoring',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_crowd_monitor_l4_8_v2',
               'initial_pos_enu': [7117.5, 6323.0, 34],
               'initial_rotation_deg': [0.0, 0.0, 50],
               'movement_waypoints': [],
               'visual_state': {'mode': 'monitor'}},
              {'asset_id': 'semantic.spawn_zone',
               'entity_id': 'crowd_anchor_l4_8_v2',
               'initial_pos_enu': [7138.539, 6336.224, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': None}],
 'event_chain': [{'actions': [{'params': {'action_id': 'spawn_incident_crowd',
                                          'count': 16,
                                          'group_id': 'crowd_l4_8_v2',
                                          'seed': 817,
                                          'spawn_box_extent_cm': [900, 600, 0],
                                          'spawn_origin_enu_m': [7138.539, 6336.224, 0.0]},
                               'type': 'spawn_crowd'}],
                  'event_id': 'crowd_generated',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'warning',
                  'log_target_ids': ['crowd_anchor_l4_8_v2'],
                  'log_title': 'Crowd generated in evacuation zone',
                  'log_topic': 'evt_L4-8_v2_crowd_generated',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 220, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'clear_incident_crowd_for_evacuation',
                                          'group_id': 'crowd_l4_8_v2'},
                               'type': 'clear_crowd'},
                              {'params': {'action_id': 'spawn_crowd_at_safe_zone',
                                          'count': 16,
                                          'group_id': 'crowd_safe_l4_8_v2',
                                          'seed': 917,
                                          'spawn_box_extent_cm': [1000, 500, 0],
                                          'spawn_origin_enu_m': [7162.402, 6346.883, 0.0]},
                               'type': 'spawn_crowd'}],
                  'event_id': 'evacuation_triggered',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'warning',
                  'log_target_ids': ['crowd_anchor_l4_8_v2'],
                  'log_title': 'Crowd moves to safe evacuation zone',
                  'log_topic': 'evt_L4-8_v2_evacuation_triggered',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'crowd_generated', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_uav_monitor_evacuation',
                                          'entity_id': 'uav_crowd_monitor_l4_8_v2',
                                          'velocity_mps': 5.0,
                                          'waypoints_enu_m': [[7117.5, 6323.0, 34],
                                                              [7138.539, 6336.224, 30],
                                                              [7162.402, 6346.883, 30]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_crowd_evacuation', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'uav_evacuation_monitor',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_crowd_monitor_l4_8_v2'],
                  'log_title': 'UAV monitors evacuation movement',
                  'log_topic': 'evt_L4-8_v2_uav_evacuation_monitor',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'evacuation_triggered', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'clear_safe_crowd', 'group_id': 'crowd_safe_l4_8_v2'},
                               'type': 'clear_crowd'}],
                  'event_id': 'crowd_clear',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_crowd_monitor_l4_8_v2'],
                  'log_title': 'Crowd evacuation completes and group is cleared',
                  'log_topic': 'evt_L4-8_v2_crowd_clear',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'delay_ticks': 113,
                              'event_ref': 'uav_evacuation_monitor',
                              'type': 'event_fired_after'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-8_v2'}


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
