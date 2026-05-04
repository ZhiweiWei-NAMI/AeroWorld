"""Concrete ScenarioSpec for L3-2_v1.

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
              'placement': {'position_enu_m': [7050.0, 6245.5, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Temporary no-fly zone activation and UAV reroute',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_l3_2_v1',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7038.0, 6237.0, 31],
                             'resolved_position_enu_m': [7038.0, 6237.0, 31],
                             'rotation_deg': {'yaw_deg': 45}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': [[7056.0, 6251.0, 31]]},
              {'activation_tick': 280,
               'category': 'airspace_constraint',
               'enabled': False,
               'entity_id': 'temporary_nfz_l3_2_v1',
               'initial_state': {},
               'logical_asset_id': 'trigger.no_fly.box.v1',
               'placement': {'center_enu_m': [7062.0, 6254.0, 28],
                             'extent_m': [12, 9, 15],
                             'resolved_position_enu_m': [7062.0, 6254.0, 28]},
               'placement_mode': 'box_volume',
               'route_waypoints_enu_m': [],
               'spawn_policy': 'event_script_only'}],
 'local_bounds': {'center_enu_m': [7052.0, 6247.333, 0.0],
                  'radius_m': 177.401,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L3-2_v1',
 'spawn_sequencing': [{'entity_id': 'uav_l3_2_v1', 'tick': 0}, {'entity_id': 'temporary_nfz_l3_2_v1', 'tick': 280}],
 'validation_rules': [{'description': 'uav_l3_2_v1 is declared before event_script references it in L3-2_v1',
                       'entity_id': 'uav_l3_2_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'temporary_nfz_l3_2_v1 is declared before event_script references it in '
                                      'L3-2_v1',
                       'entity_id': 'temporary_nfz_l3_2_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_l3_2_v1',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'temporary_nfz_l3_2_v1',
                       'logical_asset_id': 'trigger.no_fly.box.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Temporary NFZ is spawned by event, not pre-activated statically',
                       'entity_id': 'temporary_nfz_l3_2_v1',
                       'rule': 'dynamic_spawn_required'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'dynamic_constraints',
 'description': 'Temporary no-fly zone activation and UAV reroute',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_l3_2_v1',
               'initial_pos_enu': [7038.0, 6237.0, 31],
               'initial_rotation_deg': [0.0, 0.0, 45],
               'movement_waypoints': [[7056.0, 6251.0, 31]],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'trigger.no_fly.box.v1',
               'entity_id': 'temporary_nfz_l3_2_v1',
               'initial_pos_enu': [7062.0, 6254.0, 28],
               'initial_rotation_deg': [0.0, 0.0, 0.0],
               'movement_waypoints': [],
               'visual_state': None}],
 'event_chain': [{'actions': [{'params': {'action_id': 'spawn_temporary_nfz',
                                          'asset_id': 'trigger.no_fly.box.v1',
                                          'entity_id': 'temporary_nfz_l3_2_v1',
                                          'position_enu_m': [7062.0, 6254.0, 28],
                                          'rotation_deg': {'yaw_deg': 0.0},
                                          'visual_state': {'mode': 'active'}},
                               'type': 'spawn_entity'}],
                  'event_id': 'temporary_nfz_declared',
                  'log_category': 'dynamic_constraint',
                  'log_overlay': 'dynamic_constraint',
                  'log_severity': 'warning',
                  'log_target_ids': ['temporary_nfz_l3_2_v1'],
                  'log_title': 'Temporary no-fly zone declared mid-operation',
                  'log_topic': 'evt_L3-2_v1_temporary_nfz_declared',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 280, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_uav_to_nfz_edge',
                                          'entity_id': 'uav_l3_2_v1',
                                          'velocity_mps': 7.0,
                                          'waypoints_enu_m': [[7038.0, 6237.0, 31], [7056.0, 6251.0, 31]]},
                               'type': 'move_entity'}],
                  'event_id': 'uav_approaches_temporary_nfz',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_l3_2_v1', 'temporary_nfz_l3_2_v1'],
                  'log_title': 'UAV approaches newly declared NFZ',
                  'log_topic': 'evt_L3-2_v1_uav_approaches_temporary_nfz',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'temporary_nfz_declared', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_uav_nfz_reroute',
                                          'entity_id': 'uav_l3_2_v1',
                                          'velocity_mps': 8.5,
                                          'waypoints_enu_m': [[7056.0, 6251.0, 31],
                                                              [7046.0, 6267.0, 34],
                                                              [7072.0, 6277.0, 34]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_temporary_nfz', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'nfz_proximity_alert',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l3_2_v1', 'temporary_nfz_l3_2_v1'],
                  'log_title': 'UAV reroutes around temporary NFZ',
                  'log_topic': 'evt_L3-2_v1_nfz_proximity_alert',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'distance_m': 12.0,
                              'entity_a': 'uav_l3_2_v1',
                              'entity_b': 'temporary_nfz_l3_2_v1',
                              'metric': '3d',
                              'min_true_ticks': 3,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity'}}],
 'parameters': {'incident_tick': 220},
 'scenario_id': 'L3-2_v1'}


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
