"""Concrete ScenarioSpec for L4-5_v2.

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
              'placement': {'position_enu_m': [7093.835, 6307.112, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'UAV low-altitude pedestrian near-miss',
 'entities': [{'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_ped_nearmiss_l4_5_v2',
               'initial_state': {'mode': 'descent'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7088.0, 6304.0, 30],
                             'resolved_position_enu_m': [7088.0, 6304.0, 30],
                             'rotation_deg': {'yaw_deg': 45}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'pedestrian',
               'entity_id': 'ped_nearmiss_l4_5_v2',
               'initial_state': {'mode': 'walking'},
               'logical_asset_id': 'pedestrian.cityops.basic.v1',
               'placement': {'lane_edge_id': 'cg_edge_204',
                             'lane_half_width_m': 1.9,
                             'longitudinal_s': 40.0,
                             'offset_from_curb_m': 1.2,
                             'placement_semantics': 'sidewalk_or_plaza',
                             'resolved_lateral_from_center_m': 3.1,
                             'resolved_position_enu_m': [7099.67, 6310.224, 0.0],
                             'rotation_deg': {'yaw_deg': 21.255},
                             'source_lane_edge_id_hint': 'cg_edge_21',
                             'source_longitudinal_s_hint': 35},
               'placement_mode': 'sidewalk_anchor',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7093.835, 6307.112, 0.0],
                  'radius_m': 166.613,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-5_v2',
 'spawn_sequencing': [{'entity_id': 'uav_ped_nearmiss_l4_5_v2', 'tick': 0},
                      {'entity_id': 'ped_nearmiss_l4_5_v2', 'tick': 0}],
 'validation_rules': [{'description': 'uav_ped_nearmiss_l4_5_v2 is declared before event_script references it in '
                                      'L4-5_v2',
                       'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'ped_nearmiss_l4_5_v2 is declared before event_script references it in L4-5_v2',
                       'entity_id': 'ped_nearmiss_l4_5_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'ped_nearmiss_l4_5_v2',
                       'logical_asset_id': 'pedestrian.cityops.basic.v1',
                       'rule': 'asset_in_catalog'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'UAV low-altitude pedestrian near-miss',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_ped_nearmiss_l4_5_v2',
               'initial_pos_enu': [7088.0, 6304.0, 30],
               'initial_rotation_deg': [0.0, 0.0, 45],
               'movement_waypoints': [],
               'visual_state': {'mode': 'descent'}},
              {'asset_id': 'pedestrian.cityops.basic.v1',
               'entity_id': 'ped_nearmiss_l4_5_v2',
               'initial_pos_enu': [7099.67, 6310.224, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 21.255],
               'movement_waypoints': [],
               'visual_state': {'mode': 'walking'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'move_uav_ped_descent',
                                          'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                                          'velocity_mps': 4.5,
                                          'waypoints_enu_m': [[7088.0, 6304.0, 30],
                                                              [7094.0, 6310.0, 15],
                                                              [7099.67, 6310.224, 5]]},
                               'type': 'move_entity'}],
                  'event_id': 'uav_low_descent',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_ped_nearmiss_l4_5_v2', 'ped_nearmiss_l4_5_v2'],
                  'log_title': 'UAV descends toward pedestrian head height',
                  'log_topic': 'evt_L4-5_v2_uav_low_descent',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 230, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'set_uav_hover_nearmiss',
                                          'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                                          'visual_state': {'mode': 'hover'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'move_uav_pull_up_after_nearmiss',
                                          'entity_id': 'uav_ped_nearmiss_l4_5_v2',
                                          'velocity_mps': 6.0,
                                          'waypoints_enu_m': [[7099.67, 6310.224, 5], [7091.67, 6318.224, 24]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_ped_nearmiss', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'pedestrian_near_miss',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'critical',
                  'log_target_ids': ['uav_ped_nearmiss_l4_5_v2', 'ped_nearmiss_l4_5_v2'],
                  'log_title': 'UAV near-miss with pedestrian triggers pull-up',
                  'log_topic': 'evt_L4-5_v2_pedestrian_near_miss',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'distance_m': 5.0,
                              'entity_a': 'uav_ped_nearmiss_l4_5_v2',
                              'entity_b': 'ped_nearmiss_l4_5_v2',
                              'horizontal_distance_m': 3.0,
                              'metric': 'xy_plus_z',
                              'min_true_ticks': 2,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity',
                              'vertical_distance_m': 6.0}},
                 {'actions': [{'params': {'action_id': 'move_ped_clear_nearmiss',
                                          'entity_id': 'ped_nearmiss_l4_5_v2',
                                          'velocity_mps': 1.5,
                                          'waypoints_enu_m': [[7099.67, 6310.224, 0.0], [7111.369, 6315.378, 0.0]]},
                               'type': 'move_entity'}],
                  'event_id': 'pedestrian_clears_area',
                  'log_category': 'pedestrian',
                  'log_overlay': 'pedestrian',
                  'log_severity': 'info',
                  'log_target_ids': ['ped_nearmiss_l4_5_v2'],
                  'log_title': 'Pedestrian clears UAV operating area',
                  'log_topic': 'evt_L4-5_v2_pedestrian_clears_area',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 41, 'event_ref': 'pedestrian_near_miss', 'type': 'event_fired_after'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-5_v2'}


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
