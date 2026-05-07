"""Concrete ScenarioSpec for L2-3_v1.

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
              'placement': {'position_enu_m': [7080.02, 6209.67, 79.562],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Charger unavailable with reroute to backup charger',
 'entities': [{'activation_tick': 0,
               'category': 'facility',
               'entity_id': 'charger_primary_l2_3_v1',
               'initial_state': {'mode': 'available'},
               'logical_asset_id': 'facility.charger.cityops.v1',
               'placement': {'position_enu_m': [7078.0, 6204.0, 0],
                             'resolved_position_enu_m': [7078.0, 6204.0, 0],
                             'rotation_deg': {'yaw_deg': 90}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'facility',
               'entity_id': 'charger_backup_l2_3_v1',
               'initial_state': {'mode': 'available'},
               'logical_asset_id': 'facility.charger.cityops.v1',
               'placement': {'position_enu_m': [7103.0, 6216.0, 0],
                             'resolved_position_enu_m': [7103.0, 6216.0, 0],
                             'rotation_deg': {'yaw_deg': 90}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_l2_3_v1',
               'initial_state': {'mode': 'preflight_on_pad'},
               'lifecycle': {'auto_lifecycle_applied': True,
                             'home_hover_enu_m': [7070.563, 6208.956, 3.0],
                             'home_pad_entity_id': 'pad_home_uav_l2_3_v1',
                             'mission_start_enu_m': [7070.0, 6212.0, 28],
                             'requires_landing_or_terminal_resolution': True,
                             'requires_takeoff': True},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7070.563, 6208.956, 3.0],
                             'resolved_position_enu_m': [7070.563, 6208.956, 3.0],
                             'rotation_deg': {'yaw_deg': 110}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': [[7070.0, 6212.0, 28], [7079.0, 6205.0, 10], [7101.0, 6214.0, 10]]},
              {'activation_tick': 0,
               'category': 'facility',
               'entity_id': 'pad_home_uav_l2_3_v1',
               'initial_state': {'mode': 'available'},
               'logical_asset_id': 'facility.landing_pad.visible.v1',
               'placement': {'approach_side': 'departure',
                             'pad_instance_id': 'home_uav_l2_3_v1',
                             'position_enu_m': [7070.563, 6208.956, 0.0],
                             'resolved_position_enu_m': [7070.563, 6208.956, 0.0],
                             'rotation_deg': {'yaw_deg': -161.35}},
               'placement_mode': 'pad_anchor',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7080.02, 6209.67, 0.0],
                  'radius_m': 183.836,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L2-3_v1',
 'spawn_sequencing': [{'entity_id': 'charger_primary_l2_3_v1', 'tick': 0},
                      {'entity_id': 'charger_backup_l2_3_v1', 'tick': 0},
                      {'entity_id': 'uav_l2_3_v1', 'tick': 0},
                      {'entity_id': 'pad_home_uav_l2_3_v1', 'tick': 0}],
 'validation_rules': [{'description': 'charger_primary_l2_3_v1 is declared before event_script references it in '
                                      'L2-3_v1',
                       'entity_id': 'charger_primary_l2_3_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'charger_backup_l2_3_v1 is declared before event_script references it in '
                                      'L2-3_v1',
                       'entity_id': 'charger_backup_l2_3_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'charger_primary_l2_3_v1',
                       'logical_asset_id': 'facility.charger.cityops.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'charger_backup_l2_3_v1',
                       'logical_asset_id': 'facility.charger.cityops.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Infrastructure scenarios include fault, agent response, and recovery or '
                                      'control',
                       'min_count': 3,
                       'rule': 'event_chain_min'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'infrastructure',
 'description': 'Charger unavailable with reroute to backup charger',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'facility.charger.cityops.v1',
               'entity_id': 'charger_primary_l2_3_v1',
               'initial_pos_enu': [7078.0, 6204.0, 0],
               'initial_rotation_deg': [0.0, 0.0, 90],
               'movement_waypoints': [],
               'visual_state': {'mode': 'available'}},
              {'asset_id': 'facility.charger.cityops.v1',
               'entity_id': 'charger_backup_l2_3_v1',
               'initial_pos_enu': [7103.0, 6216.0, 0],
               'initial_rotation_deg': [0.0, 0.0, 90],
               'movement_waypoints': [],
               'visual_state': {'mode': 'available'}},
              {'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_l2_3_v1',
               'initial_pos_enu': [7070.563, 6208.956, 3.0],
               'initial_rotation_deg': [0.0, 0.0, 110],
               'movement_waypoints': [[7070.0, 6212.0, 28], [7079.0, 6205.0, 10], [7101.0, 6214.0, 10]],
               'visual_state': {'mode': 'preflight_on_pad'}},
              {'asset_id': 'facility.landing_pad.visible.v1',
               'entity_id': 'pad_home_uav_l2_3_v1',
               'initial_pos_enu': [7070.563, 6208.956, 0.0],
               'initial_rotation_deg': [0.0, 0.0, -161.35],
               'movement_waypoints': [],
               'visual_state': {'mode': 'available'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'move_uav_l2_3_v1_takeoff_entry',
                                          'entity_id': 'uav_l2_3_v1',
                                          'velocity_mps': 5.0,
                                          'waypoints_enu_m': [[7070.563, 6208.956, 3.0],
                                                              [7070.563, 6208.956, 24.0],
                                                              [7070.0, 6212.0, 28]]},
                               'type': 'move_entity'}],
                  'event_id': 'lifecycle_takeoff_uav_l2_3_v1',
                  'log_category': 'uav_lifecycle',
                  'log_overlay': 'uav_lifecycle',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l2_3_v1', 'pad_home_uav_l2_3_v1'],
                  'log_title': 'UAV takes off from visible home pad and enters mission airspace',
                  'log_topic': 'evt_L2-3_v1_lifecycle_takeoff_uav_l2_3_v1',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 0,
                  'trigger': {'tick': 30, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'set_primary_charger_offline',
                                          'entity_id': 'charger_primary_l2_3_v1',
                                          'visual_state': {'mode': 'unavailable'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'charger_unavailable',
                  'log_category': 'infrastructure',
                  'log_overlay': 'infrastructure',
                  'log_severity': 'warning',
                  'log_target_ids': ['charger_primary_l2_3_v1'],
                  'log_title': 'Primary charger becomes unavailable',
                  'log_topic': 'evt_L2-3_v1_charger_unavailable',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 240, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_uav_charger_inspection',
                                          'entity_id': 'uav_l2_3_v1',
                                          'velocity_mps': 4.0,
                                          'waypoints_enu_m': [[7070.0, 6212.0, 28],
                                                              [7076.0, 6207.0, 18],
                                                              [7079.0, 6205.0, 10]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_charger_fault', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'uav_inspects_charger',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_l2_3_v1', 'charger_primary_l2_3_v1'],
                  'log_title': 'UAV checks the unavailable charger',
                  'log_topic': 'evt_L2-3_v1_uav_inspects_charger',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'charger_unavailable', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_uav_backup_charger',
                                          'entity_id': 'uav_l2_3_v1',
                                          'velocity_mps': 6.0,
                                          'waypoints_enu_m': [[7079.0, 6205.0, 10],
                                                              [7090.0, 6210.0, 18],
                                                              [7101.0, 6214.0, 10]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'set_backup_charger_reserved',
                                          'entity_id': 'charger_backup_l2_3_v1',
                                          'visual_state': {'mode': 'reserved'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'reroute_backup_charger',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l2_3_v1', 'charger_backup_l2_3_v1'],
                  'log_title': 'UAV reroutes to backup charger',
                  'log_topic': 'evt_L2-3_v1_reroute_backup_charger',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 60, 'event_ref': 'uav_inspects_charger', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_uav_l2_3_v1_landing_return',
                                          'entity_id': 'uav_l2_3_v1',
                                          'velocity_mps': 4.0,
                                          'waypoints_enu_m': [[7101.0, 6214.0, 10.0],
                                                              [7070.563, 6208.956, 12.0],
                                                              [7070.563, 6208.956, 3.0]]},
                               'type': 'move_entity'}],
                  'event_id': 'lifecycle_landing_uav_l2_3_v1',
                  'log_category': 'uav_lifecycle',
                  'log_overlay': 'uav_lifecycle',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l2_3_v1', 'pad_home_uav_l2_3_v1'],
                  'log_title': 'UAV returns to the visible home pad and lands after mission resolution',
                  'log_topic': 'evt_L2-3_v1_lifecycle_landing_uav_l2_3_v1',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 9,
                  'trigger': {'delay_ticks': 80,
                              'event_ref': 'reroute_backup_charger',
                              'type': 'event_fired_after'}}],
 'parameters': {'failure_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L2-3_v1'}


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
