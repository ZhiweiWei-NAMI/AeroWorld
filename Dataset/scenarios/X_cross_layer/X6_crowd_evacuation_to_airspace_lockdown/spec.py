"""Concrete ScenarioSpec for X6_crowd_evacuation_to_airspace_lockdown.

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
              'placement': {'position_enu_m': [119.333, 278.0, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Crowd evacuation to airspace lockdown chain',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_x6_monitor',
               'initial_state': {'mode': 'monitor'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [104.0, 269.0, 34], 'rotation_deg': {'yaw_deg': 55}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'crowd_anchor',
               'entity_id': 'crowd_anchor_x6',
               'initial_state': {},
               'logical_asset_id': 'semantic.spawn_zone',
               'placement': {'position_enu_m': [126.0, 282.0, 0], 'rotation_deg': {'yaw_deg': 0}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 310,
               'category': 'airspace_constraint',
               'entity_id': 'nfz_x6_lockdown',
               'initial_state': {},
               'logical_asset_id': 'trigger.no_fly.box.v1',
               'placement': {'center_enu_m': [128.0, 283.0, 28], 'extent_m': [18, 13, 16]},
               'placement_mode': 'box_volume',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'X6_crowd_evacuation_to_airspace_lockdown',
 'spawn_sequencing': [{'entity_id': 'uav_x6_monitor', 'tick': 0},
                      {'entity_id': 'crowd_anchor_x6', 'tick': 0},
                      {'entity_id': 'nfz_x6_lockdown', 'tick': 310}],
 'validation_rules': [{'description': 'uav_x6_monitor is declared before event_script references it in '
                                      'X6_crowd_evacuation_to_airspace_lockdown',
                       'entity_id': 'uav_x6_monitor',
                       'rule': 'entity_resolvable'},
                      {'description': 'crowd_anchor_x6 is declared before event_script references it in '
                                      'X6_crowd_evacuation_to_airspace_lockdown',
                       'entity_id': 'crowd_anchor_x6',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_x6_monitor',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'crowd_anchor_x6',
                       'logical_asset_id': 'semantic.spawn_zone',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Cross-layer scenario has at least five causal steps',
                       'min_count': 5,
                       'rule': 'cross_layer_event_chain_min'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'cross_layer',
 'description': 'Crowd evacuation to airspace lockdown chain',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_x6_monitor',
               'initial_pos_enu': [104.0, 269.0, 34],
               'initial_rotation_deg': [0.0, 0.0, 55],
               'movement_waypoints': [],
               'visual_state': {'mode': 'monitor'}},
              {'asset_id': 'semantic.spawn_zone',
               'entity_id': 'crowd_anchor_x6',
               'initial_pos_enu': [126.0, 282.0, 0],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': None},
              {'asset_id': 'trigger.no_fly.box.v1',
               'entity_id': 'nfz_x6_lockdown',
               'initial_pos_enu': [128.0, 283.0, 28],
               'initial_rotation_deg': [0.0, 0.0, 0.0],
               'movement_waypoints': [],
               'visual_state': None}],
 'event_chain': [{'actions': [{'params': {'action_id': 'spawn_x6_crowd',
                                          'count': 14,
                                          'group_id': 'crowd_x6',
                                          'seed': 1606,
                                          'spawn_box_extent_cm': [900, 600, 0],
                                          'spawn_origin_enu_m': [126.0, 282.0, 0]},
                               'type': 'spawn_crowd'}],
                  'event_id': 'crowd_evacuation',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'warning',
                  'log_target_ids': ['crowd_anchor_x6'],
                  'log_title': 'Crowd evacuation begins',
                  'log_topic': 'evt_X6_crowd_evacuation_to_airspace_lockdown_crowd_evacuation',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 220, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'clear_x6_origin_crowd', 'group_id': 'crowd_x6'},
                               'type': 'clear_crowd'},
                              {'params': {'action_id': 'spawn_x6_safe_crowd',
                                          'count': 14,
                                          'group_id': 'crowd_x6_safe',
                                          'seed': 1607,
                                          'spawn_box_extent_cm': [900, 500, 0],
                                          'spawn_origin_enu_m': [150.0, 297.0, 0]},
                               'type': 'spawn_crowd'}],
                  'event_id': 'crowd_safe_zone_move',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'warning',
                  'log_target_ids': ['crowd_anchor_x6'],
                  'log_title': 'Crowd moves to safe zone',
                  'log_topic': 'evt_X6_crowd_evacuation_to_airspace_lockdown_crowd_safe_zone_move',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'event_ref': 'crowd_evacuation', 'type': 'event_fired'}},
                 {'actions': [{'params': {'action_id': 'spawn_x6_nfz',
                                          'asset_id': 'trigger.no_fly.box.v1',
                                          'entity_id': 'nfz_x6_lockdown',
                                          'position_enu_m': [128.0, 283.0, 28],
                                          'rotation_deg': {'yaw_deg': 0},
                                          'visual_state': {'mode': 'active'}},
                               'type': 'spawn_entity'}],
                  'event_id': 'airspace_lockdown',
                  'log_category': 'dynamic_constraint',
                  'log_overlay': 'dynamic_constraint',
                  'log_severity': 'critical',
                  'log_target_ids': ['nfz_x6_lockdown'],
                  'log_title': 'Airspace lockdown declares temporary NFZ',
                  'log_topic': 'evt_X6_crowd_evacuation_to_airspace_lockdown_airspace_lockdown',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'event_ref': 'crowd_safe_zone_move', 'type': 'event_fired'}},
                 {'actions': [{'params': {'action_id': 'move_x6_uav_avoid_nfz',
                                          'entity_id': 'uav_x6_monitor',
                                          'velocity_mps': 7.0,
                                          'waypoints_enu_m': [[104.0, 269.0, 34],
                                                              [117.0, 293.0, 36],
                                                              [144.0, 307.0, 36]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_x6_lockdown', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'uav_reroute_lockdown',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_x6_monitor', 'nfz_x6_lockdown'],
                  'log_title': 'UAV reroutes around lockdown NFZ',
                  'log_topic': 'evt_X6_crowd_evacuation_to_airspace_lockdown_uav_reroute_lockdown',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'event_ref': 'airspace_lockdown', 'type': 'event_fired'}},
                 {'actions': [{'params': {'action_id': 'clear_x6_safe_crowd', 'group_id': 'crowd_x6_safe'},
                               'type': 'clear_crowd'},
                              {'params': {'action_id': 'set_x6_nfz_standdown',
                                          'entity_id': 'nfz_x6_lockdown',
                                          'visual_state': {'mode': 'standdown'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'lockdown_cleared',
                  'log_category': 'dynamic_constraint',
                  'log_overlay': 'dynamic_constraint',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_x6_monitor', 'nfz_x6_lockdown'],
                  'log_title': 'Crowd evacuation and airspace lockdown clear',
                  'log_topic': 'evt_X6_crowd_evacuation_to_airspace_lockdown_lockdown_cleared',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 5,
                  'trigger': {'event_ref': 'uav_reroute_lockdown', 'type': 'event_fired'}}],
 'parameters': {'cross_layer': True},
 'scenario_id': 'X6_crowd_evacuation_to_airspace_lockdown'}


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
