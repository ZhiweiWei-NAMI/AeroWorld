"""Concrete ScenarioSpec for X2_gnss_spoof_to_geofence_violation.

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
              'placement': {'position_enu_m': [7073.5, 6458.0, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'GNSS spoofing to geofence violation chain',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_x2_spoofed',
               'initial_state': {'mode': 'mission'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7058.0, 6451.0, 32],
                             'resolved_position_enu_m': [7058.0, 6451.0, 32],
                             'rotation_deg': {'yaw_deg': 45}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'airspace_constraint',
               'entity_id': 'nfz_x2_geofence',
               'initial_state': {},
               'logical_asset_id': 'trigger.no_fly.box.v1',
               'placement': {'center_enu_m': [7089.0, 6465.0, 28],
                             'extent_m': [12, 10, 15],
                             'resolved_position_enu_m': [7089.0, 6465.0, 28]},
               'placement_mode': 'box_volume',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'X2_gnss_spoof_to_geofence_violation',
 'spawn_sequencing': [{'entity_id': 'uav_x2_spoofed', 'tick': 0}, {'entity_id': 'nfz_x2_geofence', 'tick': 0}],
 'validation_rules': [{'description': 'uav_x2_spoofed is declared before event_script references it in '
                                      'X2_gnss_spoof_to_geofence_violation',
                       'entity_id': 'uav_x2_spoofed',
                       'rule': 'entity_resolvable'},
                      {'description': 'nfz_x2_geofence is declared before event_script references it in '
                                      'X2_gnss_spoof_to_geofence_violation',
                       'entity_id': 'nfz_x2_geofence',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_x2_spoofed',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'nfz_x2_geofence',
                       'logical_asset_id': 'trigger.no_fly.box.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Cross-layer scenario has at least five causal steps',
                       'min_count': 5,
                       'rule': 'cross_layer_event_chain_min'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'cross_layer',
 'description': 'GNSS spoofing to geofence violation chain',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_x2_spoofed',
               'initial_pos_enu': [7058.0, 6451.0, 32],
               'initial_rotation_deg': [0.0, 0.0, 45],
               'movement_waypoints': [],
               'visual_state': {'mode': 'mission'}},
              {'asset_id': 'trigger.no_fly.box.v1',
               'entity_id': 'nfz_x2_geofence',
               'initial_pos_enu': [7089.0, 6465.0, 28],
               'initial_rotation_deg': [0.0, 0.0, 0.0],
               'movement_waypoints': [],
               'visual_state': None}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_x2_spoof_flag',
                                          'entity_id': 'uav_x2_spoofed',
                                          'visual_state': {'mode': 'gnss_spoofed'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'spoofing_start',
                  'log_category': 'digital_layer',
                  'log_overlay': 'digital_layer',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_x2_spoofed'],
                  'log_title': 'GNSS spoofing starts',
                  'log_topic': 'evt_X2_gnss_spoof_to_geofence_violation_spoofing_start',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 230, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_x2_spoof_offset',
                                          'entity_id': 'uav_x2_spoofed',
                                          'velocity_mps': 7.0,
                                          'waypoints_enu_m': [[7058.0, 6451.0, 32],
                                                              [7071.0, 6456.0, 32],
                                                              [7086.0, 6463.0, 32]]},
                               'type': 'move_entity'}],
                  'event_id': 'trajectory_offset',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_x2_spoofed'],
                  'log_title': 'Spoofing offsets UAV trajectory',
                  'log_topic': 'evt_X2_gnss_spoof_to_geofence_violation_trajectory_offset',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'spoofing_start', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'set_x2_geofence_alert',
                                          'entity_id': 'uav_x2_spoofed',
                                          'visual_state': {'mode': 'geofence_alert'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'capture_x2_geofence_violation',
                                          'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'geofence_violation',
                  'log_category': 'airspace',
                  'log_overlay': 'airspace',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_x2_spoofed', 'nfz_x2_geofence'],
                  'log_title': 'Spoofed route approaches NFZ',
                  'log_topic': 'evt_X2_gnss_spoof_to_geofence_violation_geofence_violation',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'distance_m': 10.0,
                              'entity_a': 'uav_x2_spoofed',
                              'entity_b': 'nfz_x2_geofence',
                              'metric': '3d',
                              'min_true_ticks': 2,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity'}},
                 {'actions': [{'params': {'action_id': 'move_x2_corrected_route',
                                          'entity_id': 'uav_x2_spoofed',
                                          'velocity_mps': 8.0,
                                          'waypoints_enu_m': [[7086.0, 6463.0, 32], [7066.0, 6475.0, 34]]},
                               'type': 'move_entity'}],
                  'event_id': 'safety_correction',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_x2_spoofed', 'nfz_x2_geofence'],
                  'log_title': 'Safety correction pulls UAV out of NFZ approach',
                  'log_topic': 'evt_X2_gnss_spoof_to_geofence_violation_safety_correction',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'geofence_violation', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'set_x2_nav_restored',
                                          'entity_id': 'uav_x2_spoofed',
                                          'visual_state': {'mode': 'mission_recovered'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'route_recovered',
                  'log_category': 'digital_layer',
                  'log_overlay': 'digital_layer',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_x2_spoofed'],
                  'log_title': 'Navigation recovers',
                  'log_topic': 'evt_X2_gnss_spoof_to_geofence_violation_route_recovered',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 5,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'safety_correction', 'type': 'event_fired_after'}}],
 'parameters': {'cross_layer': True},
 'scenario_id': 'X2_gnss_spoof_to_geofence_violation'}


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
