"""Concrete ScenarioSpec for L4-4_v2.

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
              'placement': {'position_enu_m': [7076.206, 6300.907, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'UAV falls onto moving ground vehicle',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_falls_vehicle_l4_4_v2',
               'initial_state': {'mode': 'unstable'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7068.0, 6301.0, 20],
                             'resolved_position_enu_m': [7068.0, 6301.0, 20],
                             'rotation_deg': {'yaw_deg': 70}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'car_under_uav_l4_4_v2',
               'initial_state': {'mode': 'moving'},
               'logical_asset_id': 'vehicle.ground.boxcar.v1',
               'placement': {'edge_id': 'cg_edge_204',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0,
                             'longitudinal_s': 22.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7084.412, 6300.813, 0.0],
                             'rotation_deg': {'yaw_deg': 22.434},
                             'source_edge_id_hint': 'cg_edge_20',
                             'source_longitudinal_s_hint': 46},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7076.206, 6300.907, 0.0],
                  'radius_m': 168.207,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-4_v2',
 'spawn_sequencing': [{'entity_id': 'uav_falls_vehicle_l4_4_v2', 'tick': 0},
                      {'entity_id': 'car_under_uav_l4_4_v2', 'tick': 0}],
 'validation_rules': [{'description': 'uav_falls_vehicle_l4_4_v2 is declared before event_script references it in '
                                      'L4-4_v2',
                       'entity_id': 'uav_falls_vehicle_l4_4_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'car_under_uav_l4_4_v2 is declared before event_script references it in '
                                      'L4-4_v2',
                       'entity_id': 'car_under_uav_l4_4_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_falls_vehicle_l4_4_v2',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'car_under_uav_l4_4_v2',
                       'logical_asset_id': 'vehicle.ground.boxcar.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'UAV and vehicle trajectories intersect rather than running parallel',
                       'rule': 'trajectory_intersection_required'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'UAV falls onto moving ground vehicle',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_falls_vehicle_l4_4_v2',
               'initial_pos_enu': [7068.0, 6301.0, 20],
               'initial_rotation_deg': [0.0, 0.0, 70],
               'movement_waypoints': [],
               'visual_state': {'mode': 'unstable'}},
              {'asset_id': 'vehicle.ground.boxcar.v1',
               'entity_id': 'car_under_uav_l4_4_v2',
               'initial_pos_enu': [7084.412, 6300.813, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 22.434],
               'movement_waypoints': [],
               'visual_state': {'mode': 'moving'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'move_uav_falling_cross',
                                          'entity_id': 'uav_falls_vehicle_l4_4_v2',
                                          'velocity_mps': 5.5,
                                          'waypoints_enu_m': [[7068.0, 6301.0, 20],
                                                              [7079.0, 6305.0, 12],
                                                              [7088.0, 6308.0, 2.6]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_car_crossing_path',
                                          'entity_id': 'car_under_uav_l4_4_v2',
                                          'velocity_mps': 7.0,
                                          'waypoints_enu_m': [[7084.412, 6300.813, 0.0],
                                                              [7089.0, 6299.0, 0],
                                                              [7088.0, 6308.0, 0]]},
                               'type': 'move_entity'}],
                  'event_id': 'crossing_trajectories',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_falls_vehicle_l4_4_v2', 'car_under_uav_l4_4_v2'],
                  'log_title': 'UAV and vehicle trajectories cross',
                  'log_topic': 'evt_L4-4_v2_crossing_trajectories',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 240, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_vehicle_emergency_brake',
                                          'entity_id': 'car_under_uav_l4_4_v2',
                                          'velocity_mps': 0.5,
                                          'waypoints_enu_m': [[7088.0, 6308.0, 0], [7088.5, 6308.5, 0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_uav_debris_ground',
                                          'entity_id': 'uav_falls_vehicle_l4_4_v2',
                                          'velocity_mps': 1.0,
                                          'waypoints_enu_m': [[7088.0, 6308.0, 2.6], [7090.0, 6309.0, 0.4]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'set_uav_debris',
                                          'entity_id': 'uav_falls_vehicle_l4_4_v2',
                                          'visual_state': {'mode': 'debris'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'capture_uav_vehicle_contact',
                                          'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'vehicle_roof_contact',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_falls_vehicle_l4_4_v2', 'car_under_uav_l4_4_v2'],
                  'log_title': 'UAV contacts vehicle roof and vehicle brakes',
                  'log_topic': 'evt_L4-4_v2_vehicle_roof_contact',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'distance_m': 3.0,
                              'entity_a': 'uav_falls_vehicle_l4_4_v2',
                              'entity_b': 'car_under_uav_l4_4_v2',
                              'metric': '3d',
                              'min_true_ticks': 1,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-4_v2'}


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
