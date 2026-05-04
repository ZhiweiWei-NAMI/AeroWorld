"""Concrete ScenarioSpec for L6-2_v1.

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
              'placement': {'position_enu_m': [7054.0, 6404.4, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Intermittent C2 degradation with slowdown and backup link',
 'entities': [{'activation_tick': 0,
               'category': 'facility',
               'entity_id': 'tower_l6_2_v1',
               'initial_state': {'mode': 'online'},
               'logical_asset_id': 'facility.radio.base_tower.v1',
               'placement': {'position_enu_m': [7058.0, 6406.4, 0],
                             'resolved_position_enu_m': [7058.0, 6406.4, 0],
                             'rotation_deg': {'yaw_deg': 0}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_digital_l6_2_v1',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7050.0, 6402.4, 32],
                             'resolved_position_enu_m': [7050.0, 6402.4, 32],
                             'rotation_deg': {'yaw_deg': 35}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L6-2_v1',
 'spawn_sequencing': [{'entity_id': 'tower_l6_2_v1', 'tick': 0}, {'entity_id': 'uav_digital_l6_2_v1', 'tick': 0}],
 'validation_rules': [{'description': 'tower_l6_2_v1 is declared before event_script references it in L6-2_v1',
                       'entity_id': 'tower_l6_2_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'uav_digital_l6_2_v1 is declared before event_script references it in L6-2_v1',
                       'entity_id': 'uav_digital_l6_2_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'tower_l6_2_v1',
                       'logical_asset_id': 'facility.radio.base_tower.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_digital_l6_2_v1',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Digital layer anomaly produces actual UAV movement and recovery action',
                       'rule': 'digital_anomaly_chain'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'digital_layer',
 'description': 'Intermittent C2 degradation with slowdown and backup link',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'facility.radio.base_tower.v1',
               'entity_id': 'tower_l6_2_v1',
               'initial_pos_enu': [7058.0, 6406.4, 0],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': {'mode': 'online'}},
              {'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_digital_l6_2_v1',
               'initial_pos_enu': [7050.0, 6402.4, 32],
               'initial_rotation_deg': [0.0, 0.0, 35],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_tower_c2_degraded',
                                          'entity_id': 'tower_l6_2_v1',
                                          'visual_state': {'mode': 'degraded'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'c2_degradation',
                  'log_category': 'digital_layer',
                  'log_overlay': 'digital_layer',
                  'log_severity': 'warning',
                  'log_target_ids': ['tower_l6_2_v1', 'uav_digital_l6_2_v1'],
                  'log_title': 'Intermittent C2 degradation starts',
                  'log_topic': 'evt_L6-2_v1_c2_degradation',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 240, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_uav_delayed_slowdown',
                                          'entity_id': 'uav_digital_l6_2_v1',
                                          'velocity_mps': 3.0,
                                          'waypoints_enu_m': [[7050.0, 6402.4, 32], [7062.0, 6408.4, 31]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'set_uav_command_delay',
                                          'entity_id': 'uav_digital_l6_2_v1',
                                          'visual_state': {'mode': 'command_delay'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'uav_delayed_response',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_digital_l6_2_v1'],
                  'log_title': 'UAV slows due to delayed command link',
                  'log_topic': 'evt_L6-2_v1_uav_delayed_response',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'c2_degradation', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'set_tower_backup_lock',
                                          'entity_id': 'tower_l6_2_v1',
                                          'visual_state': {'mode': 'backup_link'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_uav_backup_resume',
                                          'entity_id': 'uav_digital_l6_2_v1',
                                          'velocity_mps': 5.5,
                                          'waypoints_enu_m': [[7062.0, 6408.4, 31], [7052.0, 6420.4, 32]]},
                               'type': 'move_entity'}],
                  'event_id': 'backup_link_lock',
                  'log_category': 'digital_layer',
                  'log_overlay': 'digital_layer',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_digital_l6_2_v1', 'tower_l6_2_v1'],
                  'log_title': 'Backup link reduces degradation',
                  'log_topic': 'evt_L6-2_v1_backup_link_lock',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'uav_delayed_response', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'set_tower_nominal',
                                          'entity_id': 'tower_l6_2_v1',
                                          'visual_state': {'mode': 'online'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'nominal_c2_restore',
                  'log_category': 'digital_layer',
                  'log_overlay': 'digital_layer',
                  'log_severity': 'info',
                  'log_target_ids': ['tower_l6_2_v1'],
                  'log_title': 'Nominal C2 restored',
                  'log_topic': 'evt_L6-2_v1_nominal_c2_restore',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'backup_link_lock', 'type': 'event_fired_after'}}],
 'parameters': {'anomaly_tick': 240, 'recovery_tick': 520},
 'scenario_id': 'L6-2_v1'}


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
