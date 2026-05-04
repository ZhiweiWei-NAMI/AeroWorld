"""Handcrafted scenario generator.

Each builder returns a complete event_script_v1 dictionary. The scripts are
kept intentionally declarative so they can be generated and verified without UE.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Plugins/SumoImporter/Scripts"))
from donghu_core.event_script_interpreter import EventScriptInterpreter

OUTPUT_BASE = Path(__file__).resolve().parent.parent / "scenarios_handcrafted"


def _log(event_id: str, title: str, category: str, severity: str, targets: list[str]) -> dict:
    return {
        "topic": event_id,
        "category": category,
        "title": title,
        "severity": severity,
        "overlay": category,
        "target_ids": targets,
    }


def _move(entity_id: str, start: list[float], end: list[float], speed: float = 5.0) -> dict:
    return {
        "action_id": f"move_{entity_id}",
        "type": "move_entity",
        "entity_id": entity_id,
        "waypoints_enu_m": [start, end],
        "velocity_mps": speed,
    }


def _make_script(
    scenario_id: str,
    description: str,
    titles: list[str],
    targets: list[str],
    category: str = "uav_mission",
    start_tick: int = 300,
) -> dict:
    primary = targets[0]
    secondary = targets[1] if len(targets) > 1 else "support_entity"
    event_ids = ["evt_stage1", "evt_stage2", "evt_stage3", "evt_stage4"]
    return {
        "$schema": "event_script_v1",
        "scenario_id": f"{scenario_id}_handcrafted",
        "description": description,
        "parameters": {
            "primary_id": primary,
            "secondary_id": secondary,
            "start_tick": start_tick,
            "duration_ticks": 900,
        },
        "triggers": [
            {"trigger_id": "trig_stage1_start", "type": "tick", "tick": "$param.start_tick"},
            {
                "trigger_id": "trig_weather_ready",
                "type": "weather_state",
                "parameter": "rain",
                "operator": "gte",
                "value": 0.2,
                "sustain_ticks": 5,
            },
            {"trigger_id": "trig_after_stage1", "type": "event_fired", "event_id": "evt_stage1"},
            {
                "trigger_id": "trig_stage2_gate",
                "type": "composite",
                "operator": "AND",
                "children": ["trig_after_stage1", "trig_weather_ready"],
            },
            {"trigger_id": "trig_after_stage2", "type": "event_fired", "event_id": "evt_stage2"},
            {"trigger_id": "trig_after_stage3", "type": "event_fired", "event_id": "evt_stage3"},
        ],
        "events": [
            {
                "event_id": event_ids[0],
                "trigger_ref": "trig_stage1_start",
                "priority": 1,
                "max_fire_count": 1,
                "actions": [_move("$param.primary_id", [58, 24, 32], [54, 22, 30], 5.0)],
                "log_event": _log(event_ids[0], titles[0], category, "warning", ["$param.primary_id"]),
            },
            {
                "event_id": event_ids[1],
                "trigger_ref": "trig_stage2_gate",
                "priority": 2,
                "max_fire_count": 1,
                "require_conditions": ["trig_weather_ready"],
                "actions": [
                    _move("$param.primary_id", [54, 22, 30], [50, 20, 24], 6.0),
                    {
                        "action_id": "capture_stage2",
                        "type": "capture_screenshot",
                        "camera_id": "demo_high_overview",
                    },
                ],
                "log_event": _log(event_ids[1], titles[1], category, "critical", ["$param.primary_id", "$param.secondary_id"]),
            },
            {
                "event_id": event_ids[2],
                "trigger_ref": "trig_after_stage2",
                "priority": 3,
                "max_fire_count": 1,
                "actions": [
                    {
                        "action_id": "secondary_state_change",
                        "type": "set_visual_state",
                        "entity_id": "$param.secondary_id",
                        "visual_state": {"mode": "response"},
                    }
                ],
                "log_event": _log(event_ids[2], titles[2], category, "warning", ["$param.primary_id", "$param.secondary_id"]),
            },
            {
                "event_id": event_ids[3],
                "trigger_ref": "trig_after_stage3",
                "priority": 4,
                "max_fire_count": 1,
                "actions": [_move("$param.primary_id", [50, 20, 24], [56, 26, 32], 4.0)],
                "log_event": _log(event_ids[3], titles[3], category, "info", ["$param.primary_id", "$param.secondary_id"]),
            },
        ],
    }


def verify_and_save(scenario_id: str, script: dict, subdir: str) -> bool:
    scenario_dir = OUTPUT_BASE / subdir / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    out_path = scenario_dir / "event_script.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(script, f, indent=2, ensure_ascii=False)

    script_str = json.dumps(script)
    entity_ids = set()
    for pattern in [r'"entity_[ab]"\s*:\s*"([^"]+)"', r'"entity_id"\s*:\s*"([^"]+)"', r'"ped_id"\s*:\s*"([^"]+)"']:
        entity_ids.update(re.findall(pattern, script_str))

    params = script.get("parameters", {})
    resolved_entities = {
        params.get(eid[7:], eid) if eid.startswith("$param.") else eid
        for eid in entity_ids
    }

    interp = EventScriptInterpreter(out_path)
    for tick in range(script.get("parameters", {}).get("duration_ticks", 900)):
        for i, eid in enumerate(sorted(resolved_entities)):
            z = 30.0 if "uav" in eid.lower() or "drone" in eid.lower() else 0.0
            interp.update_entity_state(eid, [50.0 + i, 20.0 + i, z], {}, [0.0, 0.0, 0.0])
        weather_rain = 0.0 if tick < 300 else min(0.95, (tick - 300) / 200.0)
        interp.update_weather_state({
            "rain": weather_rain,
            "fog": 0.0,
            "wind_speed": 3.0,
            "visibility_m": 20000.0 * (1 - weather_rain),
        })
        interp.tick(tick)

    log = interp.get_event_log()
    expected = len(script["events"])
    if len(log) == expected:
        print(f"  [OK] {scenario_id}: {len(log)}/{expected} events")
        return True

    fired_ids = {e.get("source_event_id", e.get("instance_id", "")) for e in log}
    expected_ids = {e["event_id"] for e in script["events"]}
    print(f"  [WARN] {scenario_id}: {len(log)}/{expected} events, missing: {expected_ids - fired_ids}")
    return False


def build_L4_5_uav_pedestrian_near_miss() -> dict:
    return _make_script("L4-5_v1", "UAV low-altitude pedestrian near miss.", ["UAV descends near walkway", "Proximity warning becomes active", "Critical avoidance maneuver starts", "UAV returns to patrol"], ["drone_inspect_05", "ped_elder_03"])


def build_L4_3_uav_forced_landing_crowd() -> dict:
    return _make_script("L4-3_v1", "UAV propulsion failure forces landing near a crowd.", ["Motor failure detected", "Crowd-zone descent risk rises", "Emergency response is activated", "Landing site is secured"], ["drone_patrol_12", "crowd_bystanders"])


def build_L4_1_uav_uav_conflict() -> dict:
    return _make_script("L4-1_v1", "Two UAVs converge in shared airspace.", ["Converging routes detected", "Conflict gate opens", "Priority rule is applied", "Separation restored"], ["drone_delivery_07", "drone_inspect_03"])


def build_L6_1_c2_link_loss_rth() -> dict:
    return _make_script("L6-1_v1", "C2 loss causes autonomous return-to-home.", ["C2 link quality drops", "Failsafe gate is satisfied", "Return-to-home is commanded", "Link-loss recovery confirmed"], ["drone_c2_loss_01", "tower_ops_01"])


def build_L6_3_gnss_spoofing_hijack() -> dict:
    return _make_script("L6-3_v1", "GNSS spoofing pulls a UAV toward an unsafe route.", ["Spoofing onset detected", "Navigation divergence confirmed", "Geofence protection engages", "Trusted navigation restored"], ["drone_spoof_01", "gnss_station_01"])


def build_L5_1_rain_cascade_multi_agent() -> dict:
    return _make_script("L5-1_v1", "Heavy rain cascades through several agents.", ["Rain front approaches", "Weather gate activates", "Agents reduce speed", "Operations stabilize"], ["drone_rain_01", "car_rain_01"], "weather")


def build_L4_7_pedestrian_fall_full_chain() -> dict:
    return _make_script("L4-7_v1", "Pedestrian fall triggers UAV detection and response.", ["Pedestrian falls", "UAV detects incident", "Responder dispatch starts", "Scene is documented"], ["drone_med_01", "ped_victim_01"], "pedestrian")


def build_L4_6_pedestrian_jaywalk_conflict() -> dict:
    return _make_script("L4-6_v1", "Pedestrian jaywalking creates a vehicle conflict.", ["Pedestrian steps into road", "Conflict condition becomes active", "Vehicle brakes", "Crossing clears"], ["car_jaywalk_01", "ped_jaywalk_01"], "vehicle")


def build_L4_9_vehicle_collision_intersection() -> dict:
    return _make_script("L4-9_v1", "Two vehicles conflict at an intersection.", ["Vehicles approach intersection", "Collision risk gate opens", "Emergency braking begins", "Intersection is cleared"], ["car_a", "car_b"], "vehicle")


def build_L4_10_ambulance_priority_passage() -> dict:
    return _make_script("L4-10_v1", "Ambulance receives priority passage.", ["Ambulance enters corridor", "Priority condition activates", "Traffic yields", "Ambulance passes safely"], ["ambulance_01", "traffic_signal_01"], "vehicle")


def build_L4_8_crowd_evacuation_uav_monitor() -> dict:
    return _make_script("L4-8_v1", "Crowd evacuation is monitored by UAV.", ["Crowd movement begins", "Evacuation gate activates", "UAV monitors density", "Area clears"], ["drone_crowd_01", "crowd_group_01"], "pedestrian")


def build_L1_1_geofence_violation_rth() -> dict:
    return _make_script("L1-1_v1", "Geofence violation triggers return-to-home.", ["UAV approaches boundary", "Boundary gate activates", "Return command starts", "Geofence margin restored"], ["drone_geo_01", "nfz_zone_01"])


def build_L1_3_noncooperative_intrusion() -> dict:
    return _make_script("L1-3_v1", "Noncooperative UAV intrudes into operational airspace.", ["Intruder appears", "Airspace alert gate activates", "Operational UAV diverts", "Intruder tracked"], ["drone_intruder_01", "drone_ops_01"])


def build_L2_1_comm_station_failure_cascade() -> dict:
    return _make_script("L2-1_v1", "Communication station failure degrades service.", ["Station fault appears", "Service degradation gate activates", "UAV adjusts mission", "Backup link stabilizes"], ["drone_station_01", "tower_comm_01"], "infrastructure")


def build_L2_4_landing_pad_emergency_contention() -> dict:
    return _make_script("L2-4_v1", "Two UAVs contend for an emergency landing pad.", ["Emergency pad requested", "Contention gate activates", "Priority landing assigned", "Second UAV diverts"], ["drone_pad_a", "pad_emergency_01"])


def build_L5_2_sudden_fog_visual_failure() -> dict:
    return _make_script("L5-2_v1", "Sudden fog degrades visual navigation.", ["Fog forms", "Visibility gate activates", "Visual navigation degrades", "Alternate sensor mode stabilizes"], ["drone_fog_01", "weather_global"], "weather")


def build_L5_3_crosswind_payload_swing() -> dict:
    return _make_script("L5-3_v1", "Crosswind creates payload swing.", ["Crosswind rises", "Wind gate activates", "Payload swing detected", "Speed reduction damps swing"], ["drone_payload_01", "payload_box_01"], "weather")


def build_L6_4_comm_jamming_multiple_uavs() -> dict:
    return _make_script("L6-4_v1", "Wideband jamming affects multiple UAVs.", ["Jamming source activates", "Degradation gate activates", "UAVs change channel", "Link quality improves"], ["drone_jam_near_01", "drone_jam_far_01"])


def build_L3_2_temporary_nfz_mid_operation() -> dict:
    return _make_script("L3-2_v1", "Temporary no-fly zone activates during operation.", ["Temporary restriction issued", "Restriction gate activates", "UAVs divert", "Airspace compliance restored"], ["drone_tfr_01", "nfz_temporary_01"])


def build_L4_4_uav_falls_on_vehicle() -> dict:
    return _make_script("L4-4_v1", "UAV falls onto a ground vehicle.", ["UAV descent anomaly begins", "Impact risk gate activates", "Vehicle stops", "Debris response begins"], ["drone_fall_01", "car_victim_01"], "vehicle")


def build_X1_rain_c2loss_conflict_forced_landing() -> dict:
    return _make_script("X1", "Rain causes C2 loss, airspace conflict, and forced landing.", ["Rain degrades link", "Cross-layer gate activates", "Airspace conflict forms", "Forced landing completes"], ["drone_x1_a", "drone_x1_b"])


def build_L4_2_uav_building_impact() -> dict:
    return _make_script("L4-2_v1", "Navigation fault drives a UAV into a building facade with falling debris.", ["Navigation fault begins", "UAV drifts toward facade", "Facade impact occurs", "Debris falls to street"], ["drone_facade_fault_01", "tower_facade_east"], "infrastructure")


def build_L2_2_gnss_urban_canyon_degradation() -> dict:
    return _make_script("L2-2_v1", "Urban canyon multipath causes GNSS drift and visual-aided correction.", ["GPS multipath rises", "Position drift pulls UAV off route", "Visual landmarks reacquired", "Route correction completes"], ["drone_delivery_canyon_01", "urban_canyon_gnss"])


def build_L4_11_av_sensor_failure_stall() -> dict:
    return _make_script("L4-11_v1", "Autonomous vehicle sensor failure causes safe stop and lane blockage.", ["Sensor confidence collapses", "Vehicle performs safe stop", "Lane blockage is detected", "UAV reports stalled vehicle"], ["av_sensor_fault_01", "drone_traffic_watch_01"], "vehicle")


def build_L6_5_ground_station_intrusion() -> dict:
    return _make_script("L6-5_v1", "Ground control station intrusion causes abnormal UAV behavior and recovery.", ["Anomalous command stream detected", "UAV deviates from command", "Safety lockout engages", "Operator restores command path"], ["drone_ops_intrusion_01", "gcs_primary_01"])


def build_L5_4_dusk_light_shift() -> dict:
    return _make_script("L5-4_v1", "Dusk light shift causes camera underexposure and infrared recovery.", ["Light level drops rapidly", "Camera feed underexposes", "Infrared mode is selected", "Exposure stabilizes"], ["drone_dusk_camera_01", "weather_global"], "weather")


def build_L3_3_emergency_isolation_zone() -> dict:
    return _make_script("L3-3_v1", "Hazmat leak creates an isolation zone with evacuation and aerial monitoring.", ["Hazmat leak is reported", "Isolation perimeter is declared", "Personnel evacuate", "UAV monitors from outside zone"], ["drone_monitor_hazmat_01", "hazmat_isolation_zone_01"], "infrastructure")


def build_L2_5_all_red_signal_failure() -> dict:
    return _make_script("L2-5_v1", "Traffic signal all-red failure causes congestion and police intervention.", ["Signal enters all-red fault", "Four-way stop forms", "Queue buildup spreads", "Police begins manual control"], ["traffic_signal_allred_01", "police_unit_signal_01"], "vehicle")


def build_L3_1_roadwork_detour() -> dict:
    return _make_script("L3-1_v1", "Road construction closes a lane and forces detour reporting.", ["Construction barrier appears", "Lane closure slows traffic", "Vehicles detour", "UAV reports work zone"], ["drone_roadwork_report_01", "roadwork_barrier_01"], "infrastructure")


def build_L5_5_high_temperature_battery_derating() -> dict:
    return _make_script("L5-5_v1", "High temperature derates UAV battery and triggers early return.", ["Ambient temperature exceeds threshold", "Battery resistance increases", "Endurance reserve drops", "UAV returns early"], ["drone_heat_derate_01", "weather_global"], "weather")


def build_L6_2_intermittent_c2_degradation() -> dict:
    return _make_script("L6-2_v1", "Intermittent C2 degradation creates delay, slowdown, and recovery.", ["C2 signal becomes unstable", "Control delay appears", "Operator commands slowdown", "Link quality recovers"], ["drone_c2_intermittent_01", "gcs_c2_monitor_01"])


BUILDERS = [
    ("L4-5_v1", build_L4_5_uav_pedestrian_near_miss, "L4_agents/collision"),
    ("L4-3_v1", build_L4_3_uav_forced_landing_crowd, "L4_agents/failure"),
    ("L4-1_v1", build_L4_1_uav_uav_conflict, "L4_agents/collision"),
    ("L6-1_v1", build_L6_1_c2_link_loss_rth, "L6_digital_layer"),
    ("L6-3_v1", build_L6_3_gnss_spoofing_hijack, "L6_digital_layer"),
    ("L5-1_v1", build_L5_1_rain_cascade_multi_agent, "L5_environment"),
    ("L4-7_v1", build_L4_7_pedestrian_fall_full_chain, "L4_agents/interaction"),
    ("L4-6_v1", build_L4_6_pedestrian_jaywalk_conflict, "L4_agents/collision"),
    ("L4-9_v1", build_L4_9_vehicle_collision_intersection, "L4_agents/collision"),
    ("L4-10_v1", build_L4_10_ambulance_priority_passage, "L4_agents/interaction"),
    ("L4-8_v1", build_L4_8_crowd_evacuation_uav_monitor, "L4_agents/interaction"),
    ("L1-1_v1", build_L1_1_geofence_violation_rth, "L1_airspace"),
    ("L1-3_v1", build_L1_3_noncooperative_intrusion, "L1_airspace"),
    ("L2-1_v1", build_L2_1_comm_station_failure_cascade, "L2_infrastructure"),
    ("L2-4_v1", build_L2_4_landing_pad_emergency_contention, "L2_infrastructure"),
    ("L5-2_v1", build_L5_2_sudden_fog_visual_failure, "L5_environment"),
    ("L5-3_v1", build_L5_3_crosswind_payload_swing, "L5_environment"),
    ("L6-4_v1", build_L6_4_comm_jamming_multiple_uavs, "L6_digital_layer"),
    ("L3-2_v1", build_L3_2_temporary_nfz_mid_operation, "L3_dynamic_constraints"),
    ("L4-4_v1", build_L4_4_uav_falls_on_vehicle, "L4_agents/collision"),
    ("X1", build_X1_rain_c2loss_conflict_forced_landing, "X_cross_layer"),
    ("L4-2_v1", build_L4_2_uav_building_impact, "L4_agents/collision"),
    ("L2-2_v1", build_L2_2_gnss_urban_canyon_degradation, "L2_infrastructure/failure"),
    ("L4-11_v1", build_L4_11_av_sensor_failure_stall, "L4_agents/failure"),
    ("L6-5_v1", build_L6_5_ground_station_intrusion, "L6_digital_layer"),
    ("L5-4_v1", build_L5_4_dusk_light_shift, "L5_environment"),
    ("L3-3_v1", build_L3_3_emergency_isolation_zone, "L3_dynamic_constraints"),
    ("L2-5_v1", build_L2_5_all_red_signal_failure, "L2_infrastructure/interaction"),
    ("L3-1_v1", build_L3_1_roadwork_detour, "L3_dynamic_constraints/interaction"),
    ("L5-5_v1", build_L5_5_high_temperature_battery_derating, "L5_environment"),
    ("L6-2_v1", build_L6_2_intermittent_c2_degradation, "L6_digital_layer"),
]


def main() -> None:
    print("Generating handcrafted scenarios...\n")
    ok = 0
    fail = 0
    for sid, builder, subdir in BUILDERS:
        if verify_and_save(sid, builder(), subdir):
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} OK, {fail} failed")


if __name__ == "__main__":
    main()
