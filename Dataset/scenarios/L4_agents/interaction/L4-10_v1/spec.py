"""Concrete ScenarioSpec for L4-10_v1.

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
              'placement': {'position_enu_m': [7135.715, 6346.109, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Ambulance priority passage with civilian yield',
 'entities': [{'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'ambulance_priority_l4_10_v1',
               'initial_state': {'lights_on': True, 'mode': 'response'},
               'logical_asset_id': 'vehicle.emergency.ambulance.v1',
               'placement': {'edge_id': 'cg_edge_122',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 242.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7124.436, 6338.028, 0.0],
                             'rotation_deg': {'yaw_deg': 23.5},
                             'source_edge_id_hint': 'cg_edge_27',
                             'source_longitudinal_s_hint': 42},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'civilian_yield_l4_10_v1',
               'initial_state': {'mode': 'moving'},
               'logical_asset_id': 'vehicle.ground.boxcar.v1',
               'placement': {'edge_id': 'cg_edge_122',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 262.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7142.71, 6346.299, 0.0],
                             'rotation_deg': {'yaw_deg': 24.992},
                             'source_edge_id_hint': 'cg_edge_27',
                             'source_longitudinal_s_hint': 58},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_priority_monitor_l4_10_v1',
               'initial_state': {'mode': 'monitor'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7140.0, 6354.0, 32],
                             'resolved_position_enu_m': [7140.0, 6354.0, 32],
                             'rotation_deg': {'yaw_deg': 60}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7135.715, 6346.109, 0.0],
                  'radius_m': 173.875,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-10_v1',
 'spawn_sequencing': [{'entity_id': 'ambulance_priority_l4_10_v1', 'tick': 0},
                      {'entity_id': 'civilian_yield_l4_10_v1', 'tick': 0},
                      {'entity_id': 'uav_priority_monitor_l4_10_v1', 'tick': 0}],
 'validation_rules': [{'description': 'ambulance_priority_l4_10_v1 is declared before event_script references it in '
                                      'L4-10_v1',
                       'entity_id': 'ambulance_priority_l4_10_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'civilian_yield_l4_10_v1 is declared before event_script references it in '
                                      'L4-10_v1',
                       'entity_id': 'civilian_yield_l4_10_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'ambulance_priority_l4_10_v1',
                       'logical_asset_id': 'vehicle.emergency.ambulance.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'civilian_yield_l4_10_v1',
                       'logical_asset_id': 'vehicle.ground.boxcar.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Ambulance starts and moves with lights_on true',
                       'entity_id': 'ambulance_priority_l4_10_v1',
                       'rule': 'ambulance_lights_on'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'Ambulance priority passage with civilian yield',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'vehicle.emergency.ambulance.v1',
               'entity_id': 'ambulance_priority_l4_10_v1',
               'initial_pos_enu': [7124.436, 6338.028, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 23.5],
               'movement_waypoints': [],
               'visual_state': {'lights_on': True, 'mode': 'response'}},
              {'asset_id': 'vehicle.ground.boxcar.v1',
               'entity_id': 'civilian_yield_l4_10_v1',
               'initial_pos_enu': [7142.71, 6346.299, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 24.992],
               'movement_waypoints': [],
               'visual_state': {'mode': 'moving'}},
              {'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_priority_monitor_l4_10_v1',
               'initial_pos_enu': [7140.0, 6354.0, 32],
               'initial_rotation_deg': [0.0, 0.0, 60],
               'movement_waypoints': [],
               'visual_state': {'mode': 'monitor'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_ambulance_lights_on',
                                          'entity_id': 'ambulance_priority_l4_10_v1',
                                          'visual_state': {'lights_on': True, 'mode': 'response'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_ambulance_priority',
                                          'entity_id': 'ambulance_priority_l4_10_v1',
                                          'velocity_mps': 13.0,
                                          'waypoints_enu_m': [[7124.0, 6338.0, 0],
                                                              [7140.0, 6339.0, 0],
                                                              [7164.0, 6341.0, 0]]},
                               'type': 'move_entity'}],
                  'event_id': 'ambulance_priority_approach',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'warning',
                  'log_target_ids': ['ambulance_priority_l4_10_v1', 'civilian_yield_l4_10_v1'],
                  'log_title': 'Ambulance approaches with lights active',
                  'log_topic': 'evt_L4-10_v1_ambulance_priority_approach',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 220, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_civilian_yield',
                                          'entity_id': 'civilian_yield_l4_10_v1',
                                          'velocity_mps': 3.0,
                                          'waypoints_enu_m': [[7146.0, 6340.0, 0],
                                                              [7149.0, 6346.0, 0],
                                                              [7151.0, 6350.0, 0]]},
                               'type': 'move_entity'}],
                  'event_id': 'civilian_vehicle_yields',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'info',
                  'log_target_ids': ['ambulance_priority_l4_10_v1', 'civilian_yield_l4_10_v1'],
                  'log_title': 'Civilian vehicle yields to ambulance',
                  'log_topic': 'evt_L4-10_v1_civilian_vehicle_yields',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 34,
                              'event_ref': 'ambulance_priority_approach',
                              'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_uav_priority_monitor',
                                          'entity_id': 'uav_priority_monitor_l4_10_v1',
                                          'velocity_mps': 5.0,
                                          'waypoints_enu_m': [[7140.0, 6354.0, 32], [7158.0, 6352.0, 28]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_ambulance_priority',
                                          'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'uav_priority_capture',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_priority_monitor_l4_10_v1', 'ambulance_priority_l4_10_v1'],
                  'log_title': 'UAV monitors priority passage',
                  'log_topic': 'evt_L4-10_v1_uav_priority_capture',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 41,
                              'event_ref': 'civilian_vehicle_yields',
                              'type': 'event_fired_after'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-10_v1'}


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
