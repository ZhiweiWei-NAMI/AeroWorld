"""Concrete ScenarioSpec for L6-5_v1.

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
              'placement': {'position_enu_m': [7102.333, 6421.267, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Ground control station intrusion and abnormal UAV behavior',
 'entities': [{'activation_tick': 0,
               'category': 'facility',
               'entity_id': 'tower_l6_5_v1',
               'initial_state': {'mode': 'online'},
               'logical_asset_id': 'facility.radio.base_tower.v1',
               'placement': {'position_enu_m': [7106.0, 6425.6, 0],
                             'resolved_position_enu_m': [7106.0, 6425.6, 0],
                             'rotation_deg': {'yaw_deg': 0}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_digital_l6_5_v1',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7098.0, 6421.6, 32],
                             'resolved_position_enu_m': [7098.0, 6421.6, 32],
                             'rotation_deg': {'yaw_deg': 35}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'ground_station',
               'entity_id': 'gcs_anchor_l6_5_v1',
               'initial_state': {'mode': 'gcs_online'},
               'logical_asset_id': 'semantic.asset_anchor',
               'placement': {'position_enu_m': [7103.0, 6416.6, 0],
                             'resolved_position_enu_m': [7103.0, 6416.6, 0],
                             'rotation_deg': {'yaw_deg': 0}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L6-5_v1',
 'spawn_sequencing': [{'entity_id': 'tower_l6_5_v1', 'tick': 0},
                      {'entity_id': 'uav_digital_l6_5_v1', 'tick': 0},
                      {'entity_id': 'gcs_anchor_l6_5_v1', 'tick': 0}],
 'validation_rules': [{'description': 'tower_l6_5_v1 is declared before event_script references it in L6-5_v1',
                       'entity_id': 'tower_l6_5_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'uav_digital_l6_5_v1 is declared before event_script references it in L6-5_v1',
                       'entity_id': 'uav_digital_l6_5_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'tower_l6_5_v1',
                       'logical_asset_id': 'facility.radio.base_tower.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_digital_l6_5_v1',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Digital layer anomaly produces actual UAV movement and recovery action',
                       'rule': 'digital_anomaly_chain'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'digital_layer',
 'description': 'Ground control station intrusion and abnormal UAV behavior',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'facility.radio.base_tower.v1',
               'entity_id': 'tower_l6_5_v1',
               'initial_pos_enu': [7106.0, 6425.6, 0],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': {'mode': 'online'}},
              {'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_digital_l6_5_v1',
               'initial_pos_enu': [7098.0, 6421.6, 32],
               'initial_rotation_deg': [0.0, 0.0, 35],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}},
              {'asset_id': 'semantic.asset_anchor',
               'entity_id': 'gcs_anchor_l6_5_v1',
               'initial_pos_enu': [7103.0, 6416.6, 0],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': {'mode': 'gcs_online'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_gcs_intrusion',
                                          'entity_id': 'gcs_anchor_l6_5_v1',
                                          'visual_state': {'mode': 'intrusion_detected'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'gcs_intrusion',
                  'log_category': 'digital_layer',
                  'log_overlay': 'digital_layer',
                  'log_severity': 'critical',
                  'log_target_ids': ['gcs_anchor_l6_5_v1', 'uav_digital_l6_5_v1'],
                  'log_title': 'Ground control station intrusion detected',
                  'log_topic': 'evt_L6-5_v1_gcs_intrusion',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 240, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_uav_intrusion_abnormal',
                                          'entity_id': 'uav_digital_l6_5_v1',
                                          'velocity_mps': 9.0,
                                          'waypoints_enu_m': [[7098.0, 6421.6, 32], [7118.0, 6409.6, 32]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'set_uav_abnormal_commanded',
                                          'entity_id': 'uav_digital_l6_5_v1',
                                          'visual_state': {'mode': 'abnormal_command'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'abnormal_uav_behavior',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_digital_l6_5_v1', 'gcs_anchor_l6_5_v1'],
                  'log_title': 'UAV follows abnormal command path',
                  'log_topic': 'evt_L6-5_v1_abnormal_uav_behavior',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'gcs_intrusion', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'set_gcs_locked',
                                          'entity_id': 'gcs_anchor_l6_5_v1',
                                          'visual_state': {'mode': 'locked_out'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_uav_command_lock',
                                          'entity_id': 'uav_digital_l6_5_v1',
                                          'velocity_mps': 4.0,
                                          'waypoints_enu_m': [[7118.0, 6409.6, 32], [7101.0, 6428.6, 34]]},
                               'type': 'move_entity'}],
                  'event_id': 'command_lockout',
                  'log_category': 'digital_layer',
                  'log_overlay': 'digital_layer',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_digital_l6_5_v1', 'gcs_anchor_l6_5_v1'],
                  'log_title': 'Command lockout and safe route intervention',
                  'log_topic': 'evt_L6-5_v1_command_lockout',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'abnormal_uav_behavior', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'set_gcs_secure',
                                          'entity_id': 'gcs_anchor_l6_5_v1',
                                          'visual_state': {'mode': 'secure'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'capture_gcs_recovery', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'secure_recovery',
                  'log_category': 'digital_layer',
                  'log_overlay': 'digital_layer',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_digital_l6_5_v1', 'gcs_anchor_l6_5_v1'],
                  'log_title': 'Ground control station secured',
                  'log_topic': 'evt_L6-5_v1_secure_recovery',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'command_lockout', 'type': 'event_fired_after'}}],
 'parameters': {'anomaly_tick': 240, 'recovery_tick': 520},
 'scenario_id': 'L6-5_v1'}


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
