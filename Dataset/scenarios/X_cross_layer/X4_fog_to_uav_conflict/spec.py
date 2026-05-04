"""Concrete ScenarioSpec for X4_fog_to_uav_conflict.

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
              'placement': {'position_enu_m': [7091.0, 6470.0, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Fog to multi-UAV conflict cross-layer chain',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_x4_a',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7072.0, 6465.0, 29],
                             'resolved_position_enu_m': [7072.0, 6465.0, 29],
                             'rotation_deg': {'yaw_deg': 70}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_x4_b',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.airsim.flying_pawn.v1',
               'placement': {'position_enu_m': [7110.0, 6475.0, 31],
                             'resolved_position_enu_m': [7110.0, 6475.0, 31],
                             'rotation_deg': {'yaw_deg': 250}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7091.0, 6470.0, 0.0], 'radius_m': 179.647, 'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'X4_fog_to_uav_conflict',
 'spawn_sequencing': [{'entity_id': 'uav_x4_a', 'tick': 0}, {'entity_id': 'uav_x4_b', 'tick': 0}],
 'validation_rules': [{'description': 'uav_x4_a is declared before event_script references it in '
                                      'X4_fog_to_uav_conflict',
                       'entity_id': 'uav_x4_a',
                       'rule': 'entity_resolvable'},
                      {'description': 'uav_x4_b is declared before event_script references it in '
                                      'X4_fog_to_uav_conflict',
                       'entity_id': 'uav_x4_b',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_x4_a',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_x4_b',
                       'logical_asset_id': 'uav.airsim.flying_pawn.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Cross-layer scenario has at least five causal steps',
                       'min_count': 5,
                       'rule': 'cross_layer_event_chain_min'}],
 'weather_profile': {'initial': 'clear',
                     'transitions': [{'overrides': {'fog': 0.62, 'visibility_m': 700.0},
                                      'profile': 'fog',
                                      'tick': 180}]}}


SPEC_DATA = {'category': 'cross_layer',
 'description': 'Fog to multi-UAV conflict cross-layer chain',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_x4_a',
               'initial_pos_enu': [7072.0, 6465.0, 29],
               'initial_rotation_deg': [0.0, 0.0, 70],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'uav.airsim.flying_pawn.v1',
               'entity_id': 'uav_x4_b',
               'initial_pos_enu': [7110.0, 6475.0, 31],
               'initial_rotation_deg': [0.0, 0.0, 250],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_x4_fog_onset',
                                          'overrides': {'fog': 0.62, 'visibility_m': 700.0},
                                          'profile': 'fog'},
                               'type': 'set_weather'}],
                  'event_id': 'fog_onset',
                  'log_category': 'weather',
                  'log_overlay': 'weather',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_x4_a', 'uav_x4_b'],
                  'log_title': 'Fog onset is applied before conflict chain',
                  'log_topic': 'evt_X4_fog_to_uav_conflict_fog_onset',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 180, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'set_x4_fog',
                                          'overrides': {'fog': 0.65, 'visibility_m': 700.0},
                                          'profile': 'fog'},
                               'type': 'set_weather'}],
                  'event_id': 'fog_threshold',
                  'log_category': 'weather',
                  'log_overlay': 'weather',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_x4_a', 'uav_x4_b'],
                  'log_title': 'Fog reaches operational threshold',
                  'log_topic': 'evt_X4_fog_to_uav_conflict_fog_threshold',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'sustain_ticks': 5,
                              'type': 'weather_state',
                              'weather_operator': 'gte',
                              'weather_parameter': 'fog',
                              'weather_value': 0.5}},
                 {'actions': [{'params': {'action_id': 'set_x4_uav_a_visual_degraded',
                                          'entity_id': 'uav_x4_a',
                                          'visual_state': {'mode': 'visual_degraded'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'set_x4_uav_b_visual_degraded',
                                          'entity_id': 'uav_x4_b',
                                          'visual_state': {'mode': 'visual_degraded'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'visibility_drop',
                  'log_category': 'weather',
                  'log_overlay': 'weather',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_x4_a', 'uav_x4_b'],
                  'log_title': 'Visibility drops for both UAVs',
                  'log_topic': 'evt_X4_fog_to_uav_conflict_visibility_drop',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'fog_threshold', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_x4_uav_a_converge',
                                          'entity_id': 'uav_x4_a',
                                          'velocity_mps': 6.0,
                                          'waypoints_enu_m': [[7072.0, 6465.0, 29], [7091.0, 6470.0, 30]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_x4_uav_b_converge',
                                          'entity_id': 'uav_x4_b',
                                          'velocity_mps': 6.0,
                                          'waypoints_enu_m': [[7110.0, 6475.0, 31], [7091.0, 6470.0, 30]]},
                               'type': 'move_entity'}],
                  'event_id': 'fog_converging_routes',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_x4_a', 'uav_x4_b'],
                  'log_title': 'Fog causes converging route conflict',
                  'log_topic': 'evt_X4_fog_to_uav_conflict_fog_converging_routes',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'visibility_drop', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'set_x4_uav_a_hover',
                                          'entity_id': 'uav_x4_a',
                                          'visual_state': {'mode': 'hover'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_x4_uav_b_evasion',
                                          'entity_id': 'uav_x4_b',
                                          'velocity_mps': 8.0,
                                          'waypoints_enu_m': [[7091.0, 6470.0, 30], [7104.0, 6483.0, 36]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_x4_conflict', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'fog_uav_conflict',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_x4_a', 'uav_x4_b'],
                  'log_title': 'UAV proximity conflict under fog',
                  'log_topic': 'evt_X4_fog_to_uav_conflict_fog_uav_conflict',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'distance_m': 8.0,
                              'entity_a': 'uav_x4_a',
                              'entity_b': 'uav_x4_b',
                              'metric': '3d',
                              'min_true_ticks': 2,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity'}},
                 {'actions': [{'params': {'action_id': 'move_x4_uav_a_resume',
                                          'entity_id': 'uav_x4_a',
                                          'velocity_mps': 5.5,
                                          'waypoints_enu_m': [[7091.0, 6470.0, 30], [7074.0, 6483.0, 30]]},
                               'type': 'move_entity'}],
                  'event_id': 'fog_conflict_recovered',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_x4_a', 'uav_x4_b'],
                  'log_title': 'UAVs recover after fog conflict',
                  'log_topic': 'evt_X4_fog_to_uav_conflict_fog_conflict_recovered',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 5,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'fog_uav_conflict', 'type': 'event_fired_after'}}],
 'parameters': {'cross_layer': True},
 'scenario_id': 'X4_fog_to_uav_conflict'}


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
