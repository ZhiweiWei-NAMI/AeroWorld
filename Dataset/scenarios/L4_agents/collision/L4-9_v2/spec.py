"""Concrete ScenarioSpec for L4-9_v2.

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
              'placement': {'position_enu_m': [7143.348, 6342.758, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Vehicle-vehicle intersection conflict',
 'entities': [{'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'car_intersection_l4_9_v2_a',
               'initial_state': {'mode': 'moving'},
               'logical_asset_id': 'vehicle.ground.boxcar.v1',
               'placement': {'edge_id': 'cg_edge_122',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 241.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7123.513, 6337.628, 0.0],
                             'rotation_deg': {'yaw_deg': 23.381},
                             'source_edge_id_hint': 'cg_edge_25',
                             'source_longitudinal_s_hint': 65},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'car_intersection_l4_9_v2_b',
               'initial_state': {'mode': 'moving'},
               'logical_asset_id': 'vehicle.emergency.suv.v1',
               'placement': {'edge_id': 'cg_edge_122',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 274.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7153.573, 6351.421, 0.0],
                             'rotation_deg': {'yaw_deg': 25.455},
                             'source_edge_id_hint': 'cg_edge_26',
                             'source_longitudinal_s_hint': 40},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7143.348, 6342.758, 0.0],
                  'radius_m': 180.487,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-9_v2',
 'spawn_sequencing': [{'entity_id': 'car_intersection_l4_9_v2_a', 'tick': 0},
                      {'entity_id': 'car_intersection_l4_9_v2_b', 'tick': 0}],
 'validation_rules': [{'description': 'car_intersection_l4_9_v2_a is declared before event_script references it in '
                                      'L4-9_v2',
                       'entity_id': 'car_intersection_l4_9_v2_a',
                       'rule': 'entity_resolvable'},
                      {'description': 'car_intersection_l4_9_v2_b is declared before event_script references it in '
                                      'L4-9_v2',
                       'entity_id': 'car_intersection_l4_9_v2_b',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'car_intersection_l4_9_v2_a',
                       'logical_asset_id': 'vehicle.ground.boxcar.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'car_intersection_l4_9_v2_b',
                       'logical_asset_id': 'vehicle.emergency.suv.v1',
                       'rule': 'asset_in_catalog'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'Vehicle-vehicle intersection conflict',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'vehicle.ground.boxcar.v1',
               'entity_id': 'car_intersection_l4_9_v2_a',
               'initial_pos_enu': [7123.513, 6337.628, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 23.381],
               'movement_waypoints': [],
               'visual_state': {'mode': 'moving'}},
              {'asset_id': 'vehicle.emergency.suv.v1',
               'entity_id': 'car_intersection_l4_9_v2_b',
               'initial_pos_enu': [7153.573, 6351.421, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 25.455],
               'movement_waypoints': [],
               'visual_state': {'mode': 'moving'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'move_car_a_intersection',
                                          'entity_id': 'car_intersection_l4_9_v2_a',
                                          'velocity_mps': 8.0,
                                          'waypoints_enu_m': [[7123.513, 6337.628, 0.0],
                                                              [7139.5, 6340.0, 0],
                                                              [7146.5, 6341.0, 0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_car_b_intersection',
                                          'entity_id': 'car_intersection_l4_9_v2_b',
                                          'velocity_mps': 7.5,
                                          'waypoints_enu_m': [[7153.573, 6351.421, 0.0],
                                                              [7147.5, 6348.0, 0],
                                                              [7146.5, 6341.0, 0]]},
                               'type': 'move_entity'}],
                  'event_id': 'vehicles_enter_intersection',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'warning',
                  'log_target_ids': ['car_intersection_l4_9_v2_a', 'car_intersection_l4_9_v2_b'],
                  'log_title': 'Two vehicles enter the same intersection from different lanes',
                  'log_topic': 'evt_L4-9_v2_vehicles_enter_intersection',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 230, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_car_a_brake',
                                          'entity_id': 'car_intersection_l4_9_v2_a',
                                          'velocity_mps': 0.5,
                                          'waypoints_enu_m': [[7146.5, 6341.0, 0], [7145.5, 6341.0, 0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_car_b_brake',
                                          'entity_id': 'car_intersection_l4_9_v2_b',
                                          'velocity_mps': 0.5,
                                          'waypoints_enu_m': [[7146.5, 6341.0, 0], [7147.5, 6342.0, 0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_vehicle_conflict', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'vehicle_collision_warning',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'critical',
                  'log_target_ids': ['car_intersection_l4_9_v2_a', 'car_intersection_l4_9_v2_b'],
                  'log_title': 'Vehicle-vehicle proximity triggers simultaneous braking',
                  'log_topic': 'evt_L4-9_v2_vehicle_collision_warning',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'distance_m': 6.0,
                              'entity_a': 'car_intersection_l4_9_v2_a',
                              'entity_b': 'car_intersection_l4_9_v2_b',
                              'metric': '3d',
                              'min_true_ticks': 2,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-9_v2'}


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
