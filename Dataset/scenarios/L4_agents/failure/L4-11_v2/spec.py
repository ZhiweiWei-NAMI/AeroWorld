"""Concrete ScenarioSpec for L4-11_v2.

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
              'placement': {'position_enu_m': [7150.13, 6356.28, 75.0],
                            'rotation_deg': {'pitch_deg': -70.0, 'yaw_deg': 0.0}},
              'placement_mode': 'world_pose'}],
 'description': 'AV sensor failure, safe stop, and UAV report',
 'entities': [{'activation_tick': 0,
               'category': 'vehicle',
               'entity_id': 'av_fault_l4_11_v2',
               'initial_state': {'mode': 'autonomous'},
               'logical_asset_id': 'vehicle.ground.boxcar.v1',
               'placement': {'edge_id': 'cg_edge_122',
                             'lane_half_width_m': 1.9,
                             'lane_index': 0,
                             'lateral_offset_m': 0.0,
                             'longitudinal_s': 272.0,
                             'placement_semantics': 'lane_center',
                             'resolved_lateral_from_center_m': 0.0,
                             'resolved_position_enu_m': [7151.76, 6350.559, 0.0],
                             'rotation_deg': {'yaw_deg': 25.392},
                             'source_edge_id_hint': 'cg_edge_28',
                             'source_longitudinal_s_hint': 47},
               'placement_mode': 'lane_anchor',
               'route_waypoints_enu_m': []},
              {'activation_tick': 0,
               'category': 'uav',
               'entity_id': 'uav_av_report_l4_11_v2',
               'initial_state': {'mode': 'patrol'},
               'logical_asset_id': 'uav.inspect.quad.v1',
               'placement': {'position_enu_m': [7148.5, 6362.0, 33],
                             'resolved_position_enu_m': [7148.5, 6362.0, 33],
                             'rotation_deg': {'yaw_deg': 70}},
               'placement_mode': 'world_pose',
               'route_waypoints_enu_m': []}],
 'local_bounds': {'center_enu_m': [7150.13, 6356.28, 0.0],
                  'radius_m': 165.948,
                  'source': 'scene_entities_and_routes'},
 'map_ref': {'coordinate_frame': 'ENU',
             'geo_reference': {'alt': 24.0, 'lat': 30.5609, 'lon': 114.3627},
             'map_id': 'donghu_road_topo'},
 'scenario_id': 'L4-11_v2',
 'spawn_sequencing': [{'entity_id': 'av_fault_l4_11_v2', 'tick': 0},
                      {'entity_id': 'uav_av_report_l4_11_v2', 'tick': 0}],
 'validation_rules': [{'description': 'av_fault_l4_11_v2 is declared before event_script references it in L4-11_v2',
                       'entity_id': 'av_fault_l4_11_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'uav_av_report_l4_11_v2 is declared before event_script references it in '
                                      'L4-11_v2',
                       'entity_id': 'uav_av_report_l4_11_v2',
                       'rule': 'entity_resolvable'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'av_fault_l4_11_v2',
                       'logical_asset_id': 'vehicle.ground.boxcar.v1',
                       'rule': 'asset_in_catalog'},
                      {'description': 'Asset ID must match Config/LowAltitude/asset_catalog.json',
                       'entity_id': 'uav_av_report_l4_11_v2',
                       'logical_asset_id': 'uav.inspect.quad.v1',
                       'rule': 'asset_in_catalog'}],
 'weather_profile': {'initial': 'clear', 'transitions': []}}


SPEC_DATA = {'category': 'agents',
 'description': 'AV sensor failure, safe stop, and UAV report',
 'duration_ticks': 900,
 'entities': [{'asset_id': 'vehicle.ground.boxcar.v1',
               'entity_id': 'av_fault_l4_11_v2',
               'initial_pos_enu': [7151.76, 6350.559, 0.0],
               'initial_rotation_deg': [0.0, 0.0, 25.392],
               'movement_waypoints': [],
               'visual_state': {'mode': 'autonomous'}},
              {'asset_id': 'uav.inspect.quad.v1',
               'entity_id': 'uav_av_report_l4_11_v2',
               'initial_pos_enu': [7148.5, 6362.0, 33],
               'initial_rotation_deg': [0.0, 0.0, 70],
               'movement_waypoints': [],
               'visual_state': {'mode': 'patrol'}}],
 'event_chain': [{'actions': [{'params': {'action_id': 'set_av_sensor_fault',
                                          'entity_id': 'av_fault_l4_11_v2',
                                          'visual_state': {'mode': 'sensor_fault'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'av_sensor_fault',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'warning',
                  'log_target_ids': ['av_fault_l4_11_v2'],
                  'log_title': 'Autonomous vehicle sensor fault detected',
                  'log_topic': 'evt_L4-11_v2_av_sensor_fault',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 1,
                  'trigger': {'tick': 230, 'type': 'tick'}},
                 {'actions': [{'params': {'action_id': 'move_av_safe_stop',
                                          'entity_id': 'av_fault_l4_11_v2',
                                          'velocity_mps': 2.0,
                                          'waypoints_enu_m': [[7152.5, 6350.0, 0],
                                                              [7163.5, 6350.6, 0],
                                                              [7170.5, 6351.0, 0]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'set_av_safe_stop',
                                          'entity_id': 'av_fault_l4_11_v2',
                                          'visual_state': {'mode': 'safe_stop'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'av_safe_stop',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'critical',
                  'log_target_ids': ['av_fault_l4_11_v2'],
                  'log_title': 'AV performs safe stop in lane',
                  'log_topic': 'evt_L4-11_v2_av_safe_stop',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 2,
                  'trigger': {'delay_ticks': 30, 'event_ref': 'av_sensor_fault', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'move_uav_blockage_report',
                                          'entity_id': 'uav_av_report_l4_11_v2',
                                          'velocity_mps': 5.0,
                                          'waypoints_enu_m': [[7148.5, 6362.0, 33], [7168.5, 6357.0, 25]]},
                               'type': 'move_entity'},
                              {'params': {'action_id': 'capture_av_lane_blockage', 'camera_id': 'demo_high_overview'},
                               'type': 'capture_screenshot'}],
                  'event_id': 'lane_blockage_report',
                  'log_category': 'uav_mission',
                  'log_overlay': 'uav_mission',
                  'log_severity': 'warning',
                  'log_target_ids': ['uav_av_report_l4_11_v2', 'av_fault_l4_11_v2'],
                  'log_title': 'UAV reports lane blockage',
                  'log_topic': 'evt_L4-11_v2_lane_blockage_report',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 3,
                  'trigger': {'delay_ticks': 100, 'event_ref': 'av_safe_stop', 'type': 'event_fired_after'}},
                 {'actions': [{'params': {'action_id': 'set_av_hazard_lights',
                                          'entity_id': 'av_fault_l4_11_v2',
                                          'visual_state': {'lights_on': True, 'mode': 'blocked_lane'}},
                               'type': 'set_visual_state'}],
                  'event_id': 'av_recovery_hold',
                  'log_category': 'vehicle',
                  'log_overlay': 'vehicle',
                  'log_severity': 'info',
                  'log_target_ids': ['av_fault_l4_11_v2'],
                  'log_title': 'AV remains in safe stop with hazard state',
                  'log_topic': 'evt_L4-11_v2_av_recovery_hold',
                  'max_fire_count': 1,
                  'on_fire_emit': [],
                  'priority': 4,
                  'trigger': {'delay_ticks': 49, 'event_ref': 'lane_blockage_report', 'type': 'event_fired_after'}}],
 'parameters': {'incident_tick': 260, 'resolution_tick': 520},
 'scenario_id': 'L4-11_v2'}


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
