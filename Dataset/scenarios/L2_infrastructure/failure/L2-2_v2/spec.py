"""Concrete ScenarioSpec for L2-2_v2.

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
              'placement': {'position_enu_m': [80.333, 4.5, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'GNSS urban canyon multipath drift and visual correction',
 'entities': [{'activation_tick': 0,
               'category': 'facility',
               'entity_id': 'tower_l2_2_v2',
               'initial_state': {'mode': 'online'},
               'logical_asset_id': 'facility.radio.base_tower.v1',
               'placement': {'position_enu_m': [73.0, 0.5, 0], 'rotation_deg': {'yaw_deg': 0}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'facade_anchor',
               'entity_id': 'facade_l2_2_v2',
               'initial_state': {},
               'logical_asset_id': 'semantic.asset_anchor',
               'placement': {'building_id': 'urban_canyon_block_3',
                             'outward_normal_enu': [0.8, -0.6, 0.0],
                             'position_enu_m': [99.0, 14.5, 18],
                             'stand_off_m': 1.0},
               'placement_mode': 'facade_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_l2_2_v2',
               'initial_state': {'mode': 'corridor_follow'},
               'logical_asset_id': 'uav.airsim.cv_pawn.v1',
               'placement': {'position_enu_m': [69.0, -1.5, 30], 'rotation_deg': {'yaw_deg': 45}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': [[83.0, 2.5, 28], [97.0, 12.5, 24], [87.0, 18.5, 30]]}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L2-2_v2',
 'spawn_sequencing': [{'entity_id': 'tower_l2_2_v2', 'tick': 0},
                      {'entity_id': 'facade_l2_2_v2', 'tick': 0},
                      {'entity_id': 'uav_l2_2_v2', 'tick': 0}],
 'validation_rules': [{'description': 'tower_l2_2_v2 is declared before event_script references it in L2-2_v2',
                       'entity_id': 'tower_l2_2_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'facade_l2_2_v2 is declared before event_script references it in L2-2_v2',
                       'entity_id': 'facade_l2_2_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'tower_l2_2_v2',
                       'logical_asset_id': 'facility.radio.base_tower.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'facade_l2_2_v2',
                       'logical_asset_id': 'semantic.asset_anchor',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Infrastructure scenarios include fault, agent response, and recovery or '
                                      'control',
                       'min_count': 3,
                       'rule': 'event_chain_min'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'infrastructure',
 'description': 'GNSS urban canyon multipath drift and visual correction',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'facility.radio.base_tower.v1',
               'entity_id': 'tower_l2_2_v2',
               'initial_pos_enu': [73.0, 0.5, 0],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': {'mode': 'online'}},
              {'asset_id': 'semantic.asset_anchor',
               'entity_id': 'facade_l2_2_v2',
               'initial_pos_enu': [99.0, 14.5, 18],
               'initial_rotation_deg': [0.0, 0.0, 0.0],
               'movement_waypoints': [],
               'visual_state': None},
              {'asset_id': 'uav.airsim.cv_pawn.v1',
               'entity_id': 'uav_l2_2_v2',
               'initial_pos_enu': [69.0, -1.5, 30],
               'initial_rotation_deg': [0.0, 0.0, 45],
               'movement_waypoints': [[83.0, 2.5, 28], [97.0, 12.5, 24], [87.0, 18.5, 30]],
               'visual_state': {'mode': 'corridor_follow'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_tower_multipath',
                                          'entity_id': 'tower_l2_2_v2',
                                          'visual_state': {'mode': 'multipath_warning'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'gnss_anomaly',
                  'log_category': 'digital_layer',
                  'log_overlay': 'digital_layer',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_l2_2_v2', 'tower_l2_2_v2'],
                  'log_title': 'GNSS multipath anomaly in urban canyon',
                  'log_topic': 'evt_L2-2_v2_gnss_anomaly',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 250, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_uav_multipath_drift',
                                          'entity_id': 'uav_l2_2_v2',
                                          'velocity_mps': 6.0,
                                          'waypoints_enu_m': [[69.0, -1.5, 30], [83.0, 2.5, 28], [97.0, 12.5, 24]]},
                               'type': 'move_entity'}],
                  'event_id': 'route_drift',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_l2_2_v2', 'facade_l2_2_v2'],
                  'log_title': 'UAV drifts off planned route by more than 10m',
                  'log_topic': 'evt_L2-2_v2_route_drift',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'event_ref': 'gnss_anomaly', 'type': 'event_fired'}},
                 {'actions': [{'params': {'action_id': 'set_uav_visual_reloc',
                                          'entity_id': 'uav_l2_2_v2',
                                          'visual_state': {'mode': 'visual_relocalization'}},
                               'type': 'set_visual_state'},
                              {'params': {'action_id': 'capture_gnss_drift', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'visual_relocalization',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_l2_2_v2', 'facade_l2_2_v2'],
                  'log_title': 'Visual relocalization engages near facade',
                  'log_topic': 'evt_L2-2_v2_visual_relocalization',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'distance_m': 8.0,
                              'entity_a': 'uav_l2_2_v2',
                              'entity_b': 'facade_l2_2_v2',
                              'min_true_ticks': 3,
                              'proximity_operator': 'lte',
                              'type': 'entity_proximity'}},
                 {'actions': [{'params': {'action_id': 'move_uav_corrected_route',
                                          'entity_id': 'uav_l2_2_v2',
                                          'velocity_mps': 5.0,
                                          'waypoints_enu_m': [[97.0, 12.5, 24], [87.0, 18.5, 30]]},
                               'type': 'move_entity'}],
                  'event_id': 'route_corrected',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'info',
                  'log_target_ids': ['uav_l2_2_v2'],
                  'log_title': 'UAV corrects the route after visual relocalization',
                  'log_topic': 'evt_L2-2_v2_route_corrected',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'event_ref': 'visual_relocalization', 'type': 'event_fired'}}],
 'parameters': {'failure_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L2-2_v2'}


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
