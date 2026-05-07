"""Concrete ScenarioSpec for L4-6_v2.

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
              'placement': {'position_enu_m': [7126.952, 6334.358, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Pedestrian jaywalk vehicle conflict',
 'entities': [{'activation_tick': 0,
               'category': 'pedestrian',
               'entity_id': 'ped_jaywalk_l4_6_v2',
               'initial_state': {'mode': 'walking'},
               'logical_asset_id': 'pedestrian.cityops.basic.v1',
               'placement': {'crosswalk_id': 'crosswalk_l4_6_v2',
                             'lane_edge_id': 'cg_edge_122',
                             'lane_half_width_m': 1.9,
                             'longitudinal_s': 243.0,
                             'offset_from_curb_m': 1.2,
                             'opposite_curb_position_enu_m': [7124.126, 6341.274, 0.0],
                             'placement_semantics': 'crosswalk_curb_start',
                             'resolved_lateral_from_center_m': -3.1,
                             'resolved_position_enu_m': [7126.61, 6335.594, 0.0],
                             'roadway_center_position_enu_m': [7125.368, 6338.434, 0.0],
                             'rotation_deg': {'yaw_deg': 23.613},
                             'side': 'west'},
               'placement_mode': 'crosswalk_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'car_jaywalk_l4_6_v2',
               'initial_state': {'mode': 'moving'},
               'logical_asset_id': 'vehicle.ground.boxcar.v1',
               'placement': {'edge_id': 'cg_edge_204',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0,
                             'longitudinal_s': 74.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7132.167, 6319.323, 0.0],
                             'rotation_deg': {'yaw_deg': 20.709},
                             'source_edge_id_hint': 'cg_edge_22',
                             'source_longitudinal_s_hint': 58},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7126.952, 6334.358, 0.0],
                  'radius_m': 175.914,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-6_v2',
 'spawn_sequencing': [{'entity_id': 'ped_jaywalk_l4_6_v2', 'tick': 0},
                      {'entity_id': 'car_jaywalk_l4_6_v2', 'tick': 0}],
 'validation_rules': [{'description': 'ped_jaywalk_l4_6_v2 is declared before event_script references it in L4-6_v2',
                       'entity_id': 'ped_jaywalk_l4_6_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'car_jaywalk_l4_6_v2 is declared before event_script references it in L4-6_v2',
                       'entity_id': 'car_jaywalk_l4_6_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'ped_jaywalk_l4_6_v2',
                       'logical_asset_id': 'pedestrian.cityops.basic.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'car_jaywalk_l4_6_v2',
                       'logical_asset_id': 'vehicle.ground.boxcar.v1',
                       'rule': 'asset_in_catalog'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'Pedestrian jaywalk vehicle conflict',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'pedestrian.cityops.basic.v1',
               'entity_id': 'ped_jaywalk_l4_6_v2',
               'initial_pos_enu': [7126.61, 6335.594, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 23.613],
               'movement_waypoints': [],
               'visual_state': {'mode': 'walking'}},
              {'asset_id': 'vehicle.ground.boxcar.v1',
               'entity_id': 'car_jaywalk_l4_6_v2',
               'initial_pos_enu': [7132.167, 6319.323, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 20.709],
               'movement_waypoints': [],
               'visual_state': {'mode': 'moving'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'move_ped_from_crosswalk_to_lane',
                                          'entity_id': 'ped_jaywalk_l4_6_v2',
                                          'velocity_mps': 1.6,
                                          'waypoints_enu_m': [[7126.61, 6335.594, 0.0], [7125.368, 6338.434, 0.0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_car_toward_crosswalk',
                                          'entity_id': 'car_jaywalk_l4_6_v2',
                                          'velocity_mps': 8.0,
                                          'waypoints_enu_m': [[7132.167, 6319.323, 0.0], [7125.368, 6338.434, 0.0]]},
                               'type': 'move_entity'}],
                  'event_id': 'ped_enters_roadway',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'warning',
                  'log_target_ids': ['ped_jaywalk_l4_6_v2', 'car_jaywalk_l4_6_v2'],
                  'log_title': 'Pedestrian moves from sidewalk into roadway',
                  'log_topic': 'evt_L4-6_v2_ped_enters_roadway',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 220, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_car_hard_brake_ped',
                                          'entity_id': 'car_jaywalk_l4_6_v2',
                                          'velocity_mps': 0.8,
                                          'waypoints_enu_m': [[7125.368, 6338.434, 0.0], [7126.368, 6338.734, 0.0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_ped_vehicle_conflict',
                                          'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'vehicle_ped_proximity',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'critical',
                  'log_target_ids': ['ped_jaywalk_l4_6_v2', 'car_jaywalk_l4_6_v2'],
                  'log_title': 'Vehicle brakes for pedestrian conflict',
                  'log_topic': 'evt_L4-6_v2_vehicle_ped_proximity',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'distance_m': 4.0,
                              'entity_a': 'ped_jaywalk_l4_6_v2',
                              'entity_b': 'car_jaywalk_l4_6_v2',
                              'horizontal_distance_m': 4.0,
                              'metric': 'xy_plus_z',
                              'min_true_ticks': 2,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity',
                              'vertical_distance_m': 1.5}},
                 {'actions': [{'params': {'action_id': 'move_ped_retreat_sidewalk',
                                          'entity_id': 'ped_jaywalk_l4_6_v2',
                                          'velocity_mps': 1.8,
                                          'waypoints_enu_m': [[7125.368, 6338.434, 0.0], [7124.126, 6341.274, 0.0]]},
                               'type': 'move_entity'}],
                  'event_id': 'ped_retreats',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'info',
                  'log_target_ids': ['ped_jaywalk_l4_6_v2'],
                  'log_title': 'Pedestrian retreats from roadway',
                  'log_topic': 'evt_L4-6_v2_ped_retreats',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'vehicle_ped_proximity', 'type': 'event_fired_after'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-6_v2'}


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
