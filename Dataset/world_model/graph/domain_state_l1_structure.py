"""Closed domain-state observation rows from the shared supplement writer.

``Dataset.semantic_simulation.domain_state._observation_row`` fixes the root.
Its family-specific producers fix ``values``; facility values also branch on
the validated facility subtype.  The same writer serves all formal levels.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from Dataset.semantic_simulation.domain_state import (
    RULE_VERSION,
    SCHEMA_VERSION,
    SOURCE_CLASSES,
)


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_LEVELS = frozenset({"L1", "L2", "L3", "L4", "L5", "L6", "X"})
_OTHER_FACILITY_SUBTYPES = (
    "communication_base_station", "ground_control_station", "radio_tower",
    "no_fly_zone", "barrier",
)


def _object(properties: dict[str, Any], *, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def _unknown_or(schema: dict[str, Any]) -> dict[str, Any]:
    return {"oneOf": [schema, {"const": "unknown"}]}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"oneOf": [schema, {"type": "null"}]}


def _string_list() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _vector3() -> dict[str, Any]:
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "prefixItems": [{"type": "number"} for _ in range(3)],
        "items": False,
    }


def _dynamic_map(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object", "properties": {},
        "patternProperties": {"^.+$": value},
        "additionalProperties": False,
    }


_TYPES = {
    "b": {"type": "boolean"},
    "i": {"type": "integer"},
    "f": {"type": "number"},
    "s": {"type": "string"},
    "l": _string_list(),
    "v": _vector3(),
    "B": _unknown_or({"type": "boolean"}),
    "I": _unknown_or({"type": "integer"}),
    "F": _unknown_or({"type": "number"}),
    "V": _unknown_or(_vector3()),
    "S": _nullable({"type": "string"}),
}


def _values(*, b: str = "", i: str = "", f: str = "", s: str = "",
            l: str = "", v: str = "", B: str = "", I: str = "",
            F: str = "", V: str = "", S: str = "",
            extra: dict[str, Any] | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for code, names in (("b", b), ("i", i), ("f", f), ("s", s),
                        ("l", l), ("v", v), ("B", B), ("I", I),
                        ("F", F), ("V", V), ("S", S)):
        for name in names.split():
            if name in properties:
                raise ValueError(f"duplicate domain-state value field: {name}")
            properties[name] = deepcopy(_TYPES[code])
    for name, schema in (extra or {}).items():
        if name in properties:
            raise ValueError(f"duplicate domain-state value field: {name}")
        properties[name] = deepcopy(schema)
    return _object(properties)


def _facility_values() -> dict[str, Any]:
    common = {
        "facility_kind": _TYPES["s"],
        "ontology_class_id": _TYPES["s"],
        "availability": _TYPES["s"],
        "fault_active": _TYPES["B"],
        "reserved": _TYPES["B"],
        "reservation_state": _TYPES["s"],
        "position_enu_m": _TYPES["V"],
    }
    landing = {
        **common,
        "facility_subtype": {"const": "landing_pad"},
        "requester_ids": _string_list(),
        "occupier_ids": _unknown_or(_string_list()),
        "occupier_ids_complete": _TYPES["b"],
        "occupied": _TYPES["B"],
        "capacity": _nullable(_TYPES["i"]),
        "contention": _nullable(_TYPES["b"]),
        "contention_pairs": {
            "type": "array",
            "items": _object({
                "pad_id": _TYPES["s"],
                "pad_ontology_class_id": {"const": "world:LandingPad"},
                "first_aircraft_id": _TYPES["s"],
                "first_aircraft_ontology_class_id": {"const": "world:UnmannedAircraft"},
                "second_aircraft_id": _TYPES["s"],
                "second_aircraft_ontology_class_id": {"const": "world:UnmannedAircraft"},
                "pair_contention": _TYPES["b"],
            }),
        },
        "known_occupier_ids": _string_list(),
    }
    charging = {
        **common,
        "facility_subtype": {"const": "charging_station"},
        "requester_ids": _string_list(),
        "capacity": _nullable(_TYPES["i"]),
        "contention": _nullable(_TYPES["b"]),
        "service_aircraft_id": _TYPES["s"],
        "queue_depth": _TYPES["i"],
        "service_plan_id": _TYPES["s"],
        "service_contact_active": _TYPES["B"],
        "service_contact_distance_m": _TYPES["F"],
    }
    other = {**common, "facility_subtype": {"enum": list(_OTHER_FACILITY_SUBTYPES)}}
    return {"oneOf": [
        _object(landing, required=[key for key in landing if key != "known_occupier_ids"]),
        _object(charging),
        _object(other),
    ]}


def _quality(family: str) -> dict[str, Any]:
    status = {"enum": ["complete", "unknown"]}
    if family == "gnss_navigation":
        return _object({
            "status": status,
            "numeric_fsm_applied": _TYPES["b"],
            "rain_visibility_not_used_as_event_driver": {"const": True},
        })
    if family in ("pad_facility", "payload_energy"):
        return _object({"status": status, "missing_inputs": _string_list()}, required=["status"])
    return _object({"status": status})


def _family_values() -> dict[str, dict[str, Any]]:
    return {
        "uav_motion": _values(
            B="airborne at_pad",
            F="speed_mps altitude_m vertical_speed_mps nearest_pad_xy_distance_m nearest_pad_altitude_delta_m nearest_pad_distance_m",
            V="position_enu_m velocity_enu_mps nearest_pad_delta_enu_m",
            s="nearest_pad_id",
        ),
        "gnss_navigation": _values(
            s="quality_level operational_state fsm_state gnss_service_id gnss_service_ontology_class_id gnss_service_source_ref spoofing_signal_id spoofing_signal_ontology_class_id spoofing_signal_source_ref",
            B="multipath_warning visual_relocalization gnss_spoofed geofence_alert mission_recovered",
            F="position_error_m recovery_decay",
            V="truth_position_enu_m reported_position_enu_m drift_offset_enu_m",
            l="missing_inputs",
        ),
        "pad_facility": _facility_values(),
        "signal_queue": _values(
            s="traffic_signal_system_id controller_health queue_lane_id queue_lane_ontology_class_id",
            l="traffic_signal_ids queued_vehicle_ids",
            i="queue_vehicle_count",
            I="controller_count max_red_duration_ticks",
            B="malfunction",
            F="queue_mean_speed_mps network_mean_speed_mps",
            extra={
                "red_duration_ticks_by_controller": _dynamic_map(_TYPES["I"]),
                "queue_sampling_window": _unknown_or(_object({
                    "queue_vehicle_count": _TYPES["i"],
                    "queued_vehicle_ids": _string_list(),
                    "queue_lane_id": _TYPES["s"],
                    "stopped_vehicle_count_by_lane": _dynamic_map(_TYPES["i"]),
                    "peak_source_tick": _TYPES["i"],
                    "window_start_tick": _TYPES["i"],
                    "window_end_tick": _TYPES["i"],
                })),
            },
        ),
        "signal_queue_lane_state": _values(
            s="queue_lane_id queue_lane_ontology_class_id",
            i="queue_vehicle_count lane_vehicle_count",
            l="queued_vehicle_ids",
        ),
        "traffic_signal_state": _values(
            s="signal_id signal_ontology_class_id controller_state controller_health",
        ),
        "road_closure": _values(
            B="road_closed barrier_active",
            I="active_closure_count affected_vehicle_count route_deviation_vehicle_count detour_candidate_vehicle_count",
            l="barrier_prop_ids closed_segment_refs closed_road_segment_ids affected_vehicle_ids",
        ),
        "road_segment_state": _values(
            s="road_segment_id road_segment_ontology_class_id",
            B="road_closed",
        ),
        # ``_medical_rows`` explicitly assigns UNKNOWN to confirmed_fall,
        # dispatch_active, responder_arrived, handoff_complete and medical_resolved
        # when the fall/dispatch/responder evidence cannot decide them.
        # The five fields in the b group remain strictly boolean.
        "medical_response": _values(
            s="posture health_state nearest_uav_id nearest_responder_id",
            b="responder_geometry_available first_responder_deployment responder_approaching physical_dispatch_trigger physical_dispatch_latched",
            B="fallen_or_injured confirmed_fall detection_active responder_deployed dispatch_active responder_arrived handoff_complete medical_resolved",
            i="fall_persistence_ticks static_duration_ticks detection_dwell_ticks responder_arrival_dwell_ticks handoff_dwell_ticks",
            F="speed_mps nearest_uav_distance_m nearest_responder_distance_m responder_eta_s",
            extra={"response_rule_parameters": _object({
                "fall_confirmation_ticks": _TYPES["i"],
                "uav_detection_radius_m": _TYPES["f"],
                "uav_detection_required_ticks": _TYPES["i"],
                "responder_arrival_radius_m": _TYPES["f"],
                "responder_approach_min_delta_m": _TYPES["f"],
                "handoff_required_dwell_ticks": _TYPES["i"],
            })},
        ),
        "crowd_evacuation": _values(
            s="crowd_ontology_class_id cohort_definition_source safe_zone_geometry_source",
            S="crowd_id",
            l="cohort_member_ids missing_cohort_member_ids",
            i="pedestrian_count observed_pedestrian_count_total cohort_member_count observed_cohort_member_count moving_count evacuating_state_count structured_evacuation_state_count continuous_movement_count safe_zone_required_count safe_zone_condition_dwell_ticks safe_zone_required_dwell_ticks",
            I="hazard_zone_count safe_zone_count",
            f="safe_zone_reached_fraction safe_zone_distance_m",
            B="evacuation_active safe_zone_reached structured_safe_zone_reached",
            V="crowd_centroid_enu_m hazard_reference_enu_m",
            extra={"pedestrian_displacement_m": _dynamic_map(_TYPES["F"])},
        ),
        "security_command": _values(
            S="actor_id actor_ontology_class_id ground_control_station_id ground_control_station_ontology_class_id communication_station_id communication_station_ontology_class_id",
            s="operational_state auth_state",
            F="auth_score command_integrity_score spectrum_interference_ratio",
            B="command_integrity command_source_valid jamming_indicator lockout_active",
            I="lockout_duration_ticks",
            l="missing_inputs",
        ),
        "payload_energy": _values(
            s="operational_state payload_swing_state range_state",
            l="missing_inputs",
            B="charging_active range_insufficient",
            F="wind_speed_mps speed_mps payload_swing_angle_deg payload_swing_rate_deg_per_tick temperature_c state_of_charge_ratio energy_consumed_ratio energy_charged_ratio temperature_energy_factor power_derating_ratio predicted_range_m planned_route_distance_m",
        ),
        "av_safe_stop": _values(
            B="sensor_fault_active mrm_active warning_effective stopped_safe safe_stop_failed",
            I="fault_start_tick",
            i="affected_vehicle_count stopped_vehicle_count stop_dwell_ticks",
            F="mean_affected_speed_mps mean_affected_accel_mps2",
            l="affected_vehicle_ids stopped_vehicle_ids",
        ),
        "ambulance_priority": _values(
            b="priority_active incident_priority_active response_priority_active ambulance_in_priority_lane yield_observed_latched clearance_reached",
            B="ambulance_passage_completed",
            s="priority_basis medical_subject_id primary_ambulance_id priority_edge_id priority_lane_id",
            l="ambulance_vehicle_ids nearby_civilian_vehicle_ids yielding_vehicle_ids",
            i="yielding_vehicle_count",
            I="priority_activation_tick",
            F="ambulance_mean_speed_mps civilian_mean_speed_mps clearance_gap_m ambulance_travel_since_priority_m",
        ),
        "forced_landing_state": _values(
            s="communication_station_id descent_driver forced_descent_initiating_driver causal_communication_entity_id causal_communication_station_id landing_zone_id",
            b="link_unavailable failsafe_active descent_seen emergency_landing_zone_valid ground_zone_cleared in_landing_zone touchdown_candidate touchdown landed",
            B="physical_fault_active fault_active forced_descent airborne_override",
            i="link_unavailable_duration_ticks touchdown_dwell_ticks",
            I="link_unavailable_since_tick forced_descent_start_tick",
            F="altitude_m speed_mps heartbeat_age_ms vertical_speed_mps xy_distance_to_landing_zone_m altitude_delta_to_landing_zone_m",
            V="position_enu_m landing_zone_center_enu_m landing_zone_approach_enu_m",
            l="ground_zone_occupant_ids",
        ),
    }


_FAMILY_RULES = {
    "uav_motion": ("uav", "domain_state.uav.motion_kinematics"),
    "gnss_navigation": ("uav", "domain_state.gnss.reported_pose_error"),
    "pad_facility": ("facility", "domain_state.facility.typed_service_state"),
    "signal_queue": ("traffic_control", "domain_state.traffic.signal_health_and_queue"),
    "signal_queue_lane_state": ("road_lane", "domain_state.traffic.sumo_lane_queue"),
    "traffic_signal_state": ("traffic_signal", "domain_state.traffic.signal_controller_state"),
    "road_closure": ("road_network", "domain_state.traffic.road_barrier_closure"),
    "road_segment_state": ("road_segment", "domain_state.traffic.sumo_edge_closure"),
    "medical_response": ("pedestrian", "domain_state.medical.fall_static_responder_distance"),
    "crowd_evacuation": ("crowd", "domain_state.crowd.hazard_and_safe_zone_counts"),
    "security_command": ("uav", "domain_state.security.auth_command_integrity"),
    "payload_energy": ("uav", "domain_state.payload.wind_swing_derating_range"),
    "av_safe_stop": ("vehicle_safety", "domain_state.vehicle.av_sensor_fault_mrm_safe_stop"),
    "ambulance_priority": ("traffic_priority", "domain_state.traffic.ambulance_priority_yield_clearance"),
    "forced_landing_state": ("uav", "domain_state.uav.fault_failsafe_forced_descent_touchdown"),
}


def _row(family: str, category: str, rule_id: str, values: dict[str, Any]) -> dict[str, Any]:
    return _object({
        "schema_name": {"const": "domain_state_observation"},
        "schema_version": {"const": SCHEMA_VERSION},
        "episode_id": _TYPES["s"],
        "tick": {"type": "integer", "minimum": 0},
        "observation_family": {"const": family},
        "subject_id": _TYPES["s"],
        "subject_category": {"const": category},
        "source_class": {"enum": sorted(SOURCE_CLASSES)},
        "rule_id": {"const": rule_id},
        "rule_version": {"const": RULE_VERSION},
        "model_id": _TYPES["s"],
        "model_version": _TYPES["s"],
        "source_refs": _string_list(),
        "values": values,
        "quality": _quality(family),
        "observation_id": _TYPES["s"],
    })


def domain_state_schema(level: str) -> dict[str, Any]:
    """Return the complete JSONL-row schema for an explicit formal level."""
    if level not in _LEVELS:
        raise ValueError(f"unsupported domain-state level: {level!r}")
    values = _family_values()
    if set(values) != set(_FAMILY_RULES):
        raise ValueError("domain-state family definitions differ")
    variants = [
        _row(family, *_FAMILY_RULES[family], family_values)
        for family, family_values in values.items()
        if family != "security_command"
    ]
    security = values["security_command"]
    variants.extend([
        _row("security_command", "uav", "domain_state.security.auth_command_integrity", security),
        _row("security_command", "ground_station", "domain_state.security.gcs_auth_command_integrity", security),
    ])
    from Dataset.world_model.graph.current_domain_structure import current_domain_variants
    variants.extend(current_domain_variants())
    return deepcopy({
        "$schema": _DRAFT,
        "$id": f"https://aeroworld.local/schema/domain_state/{level}",
        "oneOf": variants,
    })


__all__ = ["domain_state_schema"]
