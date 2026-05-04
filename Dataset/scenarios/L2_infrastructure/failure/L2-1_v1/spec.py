"""Concrete ScenarioSpec for L2-1_v1.

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
              'placement': {'position_enu_m': [7051.0, 6184.0, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Communication station failure with backup link recovery',
 'entities': [{'activation_tick': 0,
               'category': 'facility',
               'entity_id': 'tower_l2_1_v1',
               'initial_state': {'mode': 'online'},
               'logical_asset_id': 'facility.radio.base_tower.v1',
               'placement': {'position_enu_m': [7054.0, 6186.0, 0],
                             'resolved_position_enu_m': [7054.0, 6186.0, 0],
                             'rotation_deg': {'yaw_deg': 0}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_l2_1_v1',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7048.0, 6182.0, 31],
                             'resolved_position_enu_m': [7048.0, 6182.0, 31],
                             'rotation_deg': {'yaw_deg': 30}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': [[7058.0, 6192.0, 30], [7037.0, 6184.0, 31]]}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L2-1_v1',
 'spawn_sequencing': [{'entity_id': 'tower_l2_1_v1', 'tick': 0}, {'entity_id': 'uav_l2_1_v1', 'tick': 0}],
 'validation_rules': [{'description': 'tower_l2_1_v1 is declared before event_script references it in L2-1_v1',
                       'entity_id': 'tower_l2_1_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'uav_l2_1_v1 is declared before event_script references it in L2-1_v1',
                       'entity_id': 'uav_l2_1_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'tower_l2_1_v1',
                       'logical_asset_id': 'facility.radio.base_tower.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_l2_1_v1',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Infrastructure scenarios include fault, agent response, and recovery or '
                                      'control',
                       'min_count': 3,
                       'rule': 'event_chain_min'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'infrastructure',
 'description': 'Communication station failure with backup link recovery',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'facility.radio.base_tower.v1',
               'entity_id': 'tower_l2_1_v1',
               'initial_pos_enu': [7054.0, 6186.0, 0],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': {'mode': 'online'}},
              {'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_l2_1_v1',
               'initial_pos_enu': [7048.0, 6182.0, 31],
               'initial_rotation_deg': [0.0, 0.0, 30],
               'movement_waypoints': [[7058.0, 6192.0, 30], [7037.0, 6184.0, 31]],
               'visual_state': {'mode': 'patrol'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_tower_degraded',
                                          'entity_id': 'tower_l2_1_v1',
                                          'visual_state': {'mode': 'degraded'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'station_degraded',
                  'log_category': 'infrastructure',
                  'log_overlay': 'infrastructure',
                  'log_severity': 'warning',
                  'log_target_ids': ['tower_l2_1_v1'],
                  'log_title': 'Communication station degraded',
                  'log_topic': 'evt_L2-1_v1_station_degraded',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 260, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_uav_backup_loiter',
                                          'entity_id': 'uav_l2_1_v1',
                                          'velocity_mps': 4.0,
                                          'waypoints_enu_m': [[7048.0, 6182.0, 31],
                                                              [7058.0, 6192.0, 30],
                                                              [7061.0, 6197.0, 34]]},
                               'type': 'move_entity'}],
                  'event_id': 'uav_link_response',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_l2_1_v1', 'tower_l2_1_v1'],
                  'log_title': 'UAV changes behavior under degraded C2',
                  'log_topic': 'evt_L2-1_v1_uav_link_response',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'station_degraded', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'set_tower_backup',
                                          'entity_id': 'tower_l2_1_v1',
                                          'visual_state': {'mode': 'backup_link'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_uav_resume_patrol',
                                          'entity_id': 'uav_l2_1_v1',
                                          'velocity_mps': 7.0,
                                          'waypoints_enu_m': [[7061.0, 6197.0, 34], [7037.0, 6184.0, 31]]},
                               'type': 'move_entity'}],
                  'event_id': 'backup_link_restore',
                  'log_category': 'infrastructure',
                  'log_overlay': 'infrastructure',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l2_1_v1', 'tower_l2_1_v1'],
                  'log_title': 'Backup link restores service',
                  'log_topic': 'evt_L2-1_v1_backup_link_restore',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'uav_link_response', 'type': 'event_fired_after'}}],
 'parameters': {'failure_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L2-1_v1'}


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
