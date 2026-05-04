"""Concrete ScenarioSpec for L2-5_v1.

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
              'placement': {'position_enu_m': [7090.097, 6216.691, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'Traffic signal all-red fault with police intervention',
 'entities': [{'activation_tick': 0,
               'category': 'traffic_signal',
               'entity_id': 'traffic_signal_l2_5_v1',
               'initial_state': {'mode': 'green_cycle'},
               'logical_asset_id': 'prop.traffic_control.signal_light.v1',
               'placement': {'position_enu_m': [7104.0, 6226.0, 4],
                             'resolved_position_enu_m': [7104.0, 6226.0, 4],
                             'rotation_deg': {'yaw_deg': 0}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'civilian_car_l2_5_a',
               'initial_state': {'mode': 'moving'},
               'logical_asset_id': 'vehicle.ground.boxcar.v1',
               'placement': {'edge_id': 'cg_edge_147',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 17.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7080.193, 6221.072, 0.0],
                             'rotation_deg': {'yaw_deg': -161.35},
                             'source_edge_id_hint': 'cg_edge_7',
                             'source_longitudinal_s_hint': 82},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': [[7098.0, 6226.0, 0], [7112.0, 6227.0, 0]]},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'civilian_car_l2_5_b',
               'initial_state': {'mode': 'moving'},
               'logical_asset_id': 'vehicle.ground.boxcar.v1',
               'placement': {'edge_id': 'cg_edge_686',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 27.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7110.214, 6203.415, 0.0],
                             'rotation_deg': {'yaw_deg': -58.929},
                             'source_edge_id_hint': 'cg_edge_8',
                             'source_longitudinal_s_hint': 54},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': [[7106.0, 6221.0, 0], [7106.0, 6235.0, 0]]},
              {'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'police_unit_l2_5_v1',
               'initial_state': {'lights_on': True, 'mode': 'response'},
               'logical_asset_id': 'vehicle.emergency.police_suv.v1',
               'placement': {'edge_id': 'cg_edge_147',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 32.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7065.982, 6216.275, 0.0],
                             'rotation_deg': {'yaw_deg': -161.35},
                             'source_edge_id_hint': 'cg_edge_12',
                             'source_longitudinal_s_hint': 38},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []}],
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L2-5_v1',
 'spawn_sequencing': [{'entity_id': 'traffic_signal_l2_5_v1', 'tick': 0},
                      {'entity_id': 'civilian_car_l2_5_a', 'tick': 0},
                      {'entity_id': 'civilian_car_l2_5_b', 'tick': 0},
                      {'entity_id': 'police_unit_l2_5_v1', 'tick': 0}],
 'validation_rules': [{'description': 'traffic_signal_l2_5_v1 is declared before event_script references it in '
                                      'L2-5_v1',
                       'entity_id': 'traffic_signal_l2_5_v1',
                       'rule': 'entity_resolvable'},
                      {'description': 'civilian_car_l2_5_a is declared before event_script references it in L2-5_v1',
                       'entity_id': 'civilian_car_l2_5_a',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'traffic_signal_l2_5_v1',
                       'logical_asset_id': 'prop.traffic_control.signal_light.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'civilian_car_l2_5_a',
                       'logical_asset_id': 'vehicle.ground.boxcar.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Infrastructure scenarios include fault, agent response, and recovery or '
                                      'control',
                       'min_count': 3,
                       'rule': 'event_chain_min'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'infrastructure',
 'description': 'Traffic signal all-red fault with police intervention',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'prop.traffic_control.signal_light.v1',
               'entity_id': 'traffic_signal_l2_5_v1',
               'initial_pos_enu': [7104.0, 6226.0, 4],
               'initial_rotation_deg': [0.0, 0.0, 0],
               'movement_waypoints': [],
               'visual_state': {'mode': 'green_cycle'}},
              {'asset_id': 'vehicle.ground.boxcar.v1',
               'entity_id': 'civilian_car_l2_5_a',
               'initial_pos_enu': [7080.193, 6221.072, 0.0],
               'initial_rotation_deg': [0.0, 0.0, -161.35],
               'movement_waypoints': [[7098.0, 6226.0, 0], [7112.0, 6227.0, 0]],
               'visual_state': {'mode': 'moving'}},
              {'asset_id': 'vehicle.ground.boxcar.v1',
               'entity_id': 'civilian_car_l2_5_b',
               'initial_pos_enu': [7110.214, 6203.415, 0.0],
               'initial_rotation_deg': [0.0, 0.0, -58.929],
               'movement_waypoints': [[7106.0, 6221.0, 0], [7106.0, 6235.0, 0]],
               'visual_state': {'mode': 'moving'}},
              {'asset_id': 'vehicle.emergency.police_suv.v1',
               'entity_id': 'police_unit_l2_5_v1',
               'initial_pos_enu': [7065.982, 6216.275, 0.0],
               'initial_rotation_deg': [0.0, 0.0, -161.35],
               'movement_waypoints': [],
               'visual_state': {'lights_on': True, 'mode': 'response'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_signal_all_red',
                                          'entity_id': 'traffic_signal_l2_5_v1',
                                          'visual_state': {'mode': 'all_red_fault'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'signal_all_red_fault',
                  'log_category': 'infrastructure',
                  'log_overlay': 'infrastructure',
                  'log_severity': 'critical',
                  'log_target_ids': ['traffic_signal_l2_5_v1'],
                  'log_title': 'Traffic signal all-red fault',
                  'log_topic': 'evt_L2-5_v1_signal_all_red_fault',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 210, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_car_a_confused_stop',
                                          'entity_id': 'civilian_car_l2_5_a',
                                          'velocity_mps': 3.0,
                                          'waypoints_enu_m': [[7079.0, 6226.0, 0], [7099.0, 6226.0, 0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_car_b_confused_stop',
                                          'entity_id': 'civilian_car_l2_5_b',
                                          'velocity_mps': 3.0,
                                          'waypoints_enu_m': [[7108.0, 6202.0, 0], [7106.0, 6222.0, 0]]},
                               'type': 'move_entity'}],
                  'event_id': 'traffic_confusion',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'warning',
                  'log_target_ids': ['civilian_car_l2_5_a', 'civilian_car_l2_5_b', 'traffic_signal_l2_5_v1'],
                  'log_title': 'Traffic queues form under all-red fault',
                  'log_topic': 'evt_L2-5_v1_traffic_confusion',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'signal_all_red_fault', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_police_to_intersection',
                                          'entity_id': 'police_unit_l2_5_v1',
                                          'velocity_mps': 12.0,
                                          'waypoints_enu_m': [[7068.0, 6210.0, 0], [7097.0, 6222.0, 0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'set_police_manual_control',
                                          'entity_id': 'police_unit_l2_5_v1',
                                          'visual_state': {'lights_on': True, 'mode': 'manual_directing'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'police_arrival',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'warning',
                  'log_target_ids': ['police_unit_l2_5_v1', 'traffic_signal_l2_5_v1'],
                  'log_title': 'Police arrives for manual traffic control',
                  'log_topic': 'evt_L2-5_v1_police_arrival',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'traffic_confusion', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_car_a_released',
                                          'entity_id': 'civilian_car_l2_5_a',
                                          'velocity_mps': 6.0,
                                          'waypoints_enu_m': [[7099.0, 6226.0, 0], [7120.0, 6227.0, 0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'move_car_b_released',
                                          'entity_id': 'civilian_car_l2_5_b',
                                          'velocity_mps': 5.5,
                                          'waypoints_enu_m': [[7106.0, 6222.0, 0], [7106.0, 6241.0, 0]]},
                               'type': 'move_entity'}],
                  'event_id': 'manual_flow_restore',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'info',
                  'log_target_ids': ['police_unit_l2_5_v1', 'civilian_car_l2_5_a', 'civilian_car_l2_5_b'],
                  'log_title': 'Police restores manual vehicle flow',
                  'log_topic': 'evt_L2-5_v1_manual_flow_restore',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'police_arrival', 'type': 'event_fired_after'}}],
 'parameters': {'failure_tick': 260,
                'primary_id': 'traffic_signal_l2_5_v1',
                'resolution_tick': 520,
                'secondary_id': 'police_unit_l2_5_v1'},
 'scenario_id': 'L2-5_v1'}


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
