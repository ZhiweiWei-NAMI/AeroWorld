"""Executable contract for runtime-state fields that can affect truth.

The formal schedules may only write fields that are consumed by the
structured observable, control-response, or domain-state pipelines.  Keeping
this list explicit prevents an authored but unread status label from looking
like semantic evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RUNTIME_STATE_FIELDS = (
    "control_state",
    "mission_state",
    "navigation_state",
    "security_state",
    "sensor_state",
    "facility_state",
    "communication_state",
    "incident_state",
    "vehicle_state",
    "pedestrian_state",
)

# ``rule_id`` and ``mechanism_id`` are provenance metadata.  They are allowed
# beside consumed fields, but no truth rule reads them as evidence.
RUNTIME_STATE_METADATA_FIELDS = frozenset({"rule_id", "mechanism_id"})

RUNTIME_STATE_CONSUMED_FIELDS: dict[str, frozenset[str]] = {
    "control_state": frozenset(
        {
            "active_command",
            "active_commands",
            "active_mode",
            "alternate_reroute_active",
            "alternate_rth_active",
            "altitude_corrective_maneuver",
            "altitude_corrective_maneuver_active",
            "command_mode",
            "commands",
            "cooperative_flag",
            "deconfliction_active",
            "diversion_active",
            "divert_active",
            "evasion_active",
            "hold_active",
            "mode",
            "pull_up_active",
            "reroute_active",
            "resequence_active",
            "rth_active",
            "safe_hold",
            "safe_hold_active",
            "slowdown_active",
            "structure_evasion",
            "structure_recovery",
        }
    ),
    "mission_state": frozenset(
        {
            "active",
            "active_command",
            "active_commands",
            "commands",
            "hazard_managed",
            "inspection_complete",
            "medical_resolved",
            "mission_abort_active",
            "mode",
            "phase",
            "safe_altitude_reached",
            "status",
            "termination_active",
            "unsafe",
        }
    ),
    "navigation_state": frozenset(
        {
            "alternate_charger_selected",
            "primary_charger_selected",
            "landing_pad_assigned",
            "correction_active",
            "degraded",
            "geofence_alert",
            "geofence_violation",
            "gnss_mode",
            "gnss_spoofed",
            "mission_recovered",
            "multipath_warning",
            "navigation_mode",
            "operational_state",
            "recovery_state",
            "relocalization_complete",
            "replan_active",
            "route_degraded",
            "route_recovered",
            "route_uncertain",
            "spoofing_active",
            "visual_navigation_degraded",
            "visual_navigation_failed",
            "visual_navigation_recovered",
            "visual_relocalization",
            "visual_relocalization_active",
        }
    ),
    "security_state": frozenset(
        {
            "active_command",
            "active_commands",
            "alternate_channel_active",
            "command_integrity_violation",
            "command_lockout",
            "commands",
            "condition",
            "gcs_compromised",
            "jamming_active",
            "lockout_active",
            "mode",
            "secure_recovery_active",
            "status",
            "threat",
            "unauthorized_command",
        }
    ),
    "sensor_state": frozenset(
        {
            "exposure_recovered",
            "exposure_status",
            "image_underexposed",
            "imaging_mode",
            "infrared",
            "infrared_mode_active",
            "intruder_detected",
            "ir_active",
            "sensor_fault",
            "underexposed",
        }
    ),
    "facility_state": frozenset(
        {
            "allocation_failed",
            "allocation_stale",
            "arbitration_active",
            "arbitration_winner_id",
            "availability",
            "backup_charger_accepted",
            "capacity",
            "contention",
            "fault",
            "multipath_warning",
            "priority_granted",
            "priority_granted_to",
            "request_count",
            "requester_ids",
            "reserved",
        }
    ),
    "communication_state": frozenset(
        {
            "alternate_link_active",
            "availability",
            "availability_state",
            "available",
            "backup_active",
            "backup_link_active",
            "channel_mode",
            "channel_recovered",
            "communication_unavailable",
            "handover_active",
            "handover_mode",
            "link_availability",
            "link_available",
            "link_mode",
            "link_restored",
            "link_state",
            "link_status",
            "mode",
            "station_available",
            "station_status",
            "station_unavailable",
            "status",
        }
    ),
    "incident_state": frozenset(
        {
            "detection_active",
            "dispatch_active",
            "emergency_fault",
            "failure_type",
            "fault_active",
            "fault_type",
            "handoff_complete",
            "hazard_concentration_ppm",
            "hazard_leak_active",
            "hazard_radius_m",
            "hazard_source_active",
            "hazard_spread_active",
            "hazmat_resolved",
            "incident_type",
            "isolation_active",
            "landing_failed",
            "lockdown_clear",
            "manual_control_active",
            "motor_fault",
            "physical_fault_active",
            "physical_state",
            "propulsion_fault",
            "requires_reroute",
            "responder_arrived",
            "temporary_lockdown_active",
            "uav_fault",
        }
    ),
    "vehicle_state": frozenset(
        {
            "priority_request_acknowledged",
            "brake_active",
            "braking",
            "collision_contact",
            "emergency_braking_active",
            "emergency_stop",
            "emergency_stop_active",
            "minimal_risk_maneuver_active",
            "minimal_risk_stop_active",
            "near_miss_classified",
            "safe_stop_failed",
            "sensor_fault",
            "stopped_safe",
            "warning_active",
            "yielding",
        }
    ),
    "pedestrian_state": frozenset(
        {
            "evacuation_active",
            "fallen",
            "health",
            "injured",
            "jaywalking",
            "posture",
            "retreating",
            "safe_zone_reached",
        }
    ),
}

RUNTIME_STATE_BOOLEAN_FIELDS: dict[str, frozenset[str]] = {
    "control_state": frozenset(
        RUNTIME_STATE_CONSUMED_FIELDS["control_state"]
        - {
            "active_command",
            "active_commands",
            "active_mode",
            "command_mode",
            "commands",
            "mode",
        }
    ),
    "mission_state": frozenset(
        RUNTIME_STATE_CONSUMED_FIELDS["mission_state"]
        - {"active_command", "active_commands", "commands", "mode", "phase", "status"}
    ),
    "navigation_state": frozenset(
        RUNTIME_STATE_CONSUMED_FIELDS["navigation_state"]
        - {"gnss_mode", "navigation_mode", "operational_state", "recovery_state"}
    ),
    "security_state": frozenset(
        RUNTIME_STATE_CONSUMED_FIELDS["security_state"]
        - {
            "active_command",
            "active_commands",
            "commands",
            "condition",
            "mode",
            "status",
            "threat",
        }
    ),
    "sensor_state": frozenset(
        RUNTIME_STATE_CONSUMED_FIELDS["sensor_state"]
        - {"exposure_status", "imaging_mode"}
    ),
    "facility_state": frozenset(
        {
            "allocation_failed",
            "allocation_stale",
            "arbitration_active",
            "backup_charger_accepted",
            "contention",
            "fault",
            "multipath_warning",
            "priority_granted",
            "reserved",
        }
    ),
    "communication_state": frozenset(
        {
            "alternate_link_active",
            "available",
            "backup_active",
            "backup_link_active",
            "channel_recovered",
            "communication_unavailable",
            "handover_active",
            "link_available",
            "link_restored",
            "station_available",
            "station_unavailable",
        }
    ),
    "incident_state": frozenset(
        RUNTIME_STATE_CONSUMED_FIELDS["incident_state"]
        - {
            "failure_type",
            "fault_type",
            "hazard_concentration_ppm",
            "hazard_radius_m",
            "incident_type",
            "physical_state",
        }
    ),
    "vehicle_state": frozenset(RUNTIME_STATE_CONSUMED_FIELDS["vehicle_state"]),
    "pedestrian_state": frozenset(
        RUNTIME_STATE_CONSUMED_FIELDS["pedestrian_state"] - {"health", "posture"}
    ),
}

_CONTROL_ENUMS = frozenset(
    {
        "nominal",
        "normal",
        "none",
        "idle",
        "rth",
        "return_to_home",
        "alternate_rth",
        "alternate_return_to_home",
        "reroute",
        "alternate_reroute",
        "alternate_route",
        "altitude_correction",
        "altitude_corrective_maneuver",
        "deconfliction",
        "evasion",
        "pull_up",
        "safe_hold",
        "hold",
        "resequence",
        "divert",
        "diversion",
        "slowdown",
    }
)
_MISSION_ENUMS = frozenset(
    {"nominal", "normal", "active", "none", "abort", "mission_abort"}
)
_SECURITY_ENUMS = frozenset(
    {
        "nominal",
        "normal",
        "clear",
        "none",
        "gcs_compromised",
        "jamming",
        "unauthorized_command",
        "command_integrity_violation",
        "command_lockout",
    }
)
_COMMUNICATION_STATUS_ENUMS = frozenset(
    {
        "available",
        "online",
        "up",
        "nominal",
        "normal",
        "unavailable",
        "offline",
        "down",
        "failed",
        "lost",
    }
)
_COMMUNICATION_MODE_ENUMS = frozenset(
    {
        "primary",
        "available",
        "online",
        "nominal",
        "normal",
        "main",
        "backup",
        "backup_link",
        "alternate_link",
        "handover",
        "degraded",
        "rain_stressed",
        "link_lost",
    }
)
_GNSS_ENUMS = frozenset(
    {
        "nominal",
        "multipath_degraded",
        "spoofed",
        "geofence_alert",
        "relocalizing",
        "recovered",
    }
)
_INCIDENT_FAULT_ENUMS = frozenset(
    {
        "uav_fault",
        "propulsion_fault",
        "motor_fault",
        "flight_fault",
        "emergency_fault",
    }
)

RUNTIME_STATE_ENUM_FIELDS: dict[tuple[str, str], frozenset[str]] = {
    **{
        ("control_state", field): _CONTROL_ENUMS
        for field in ("mode", "active_mode", "command_mode")
    },
    **{
        ("mission_state", field): _MISSION_ENUMS
        for field in ("mode", "status", "phase")
    },
    **{
        ("navigation_state", field): _GNSS_ENUMS
        for field in (
            "operational_state",
            "gnss_mode",
            "navigation_mode",
            "recovery_state",
        )
    },
    **{
        ("security_state", field): _SECURITY_ENUMS
        for field in ("mode", "status", "threat", "condition")
    },
    ("sensor_state", "exposure_status"): frozenset(
        {"nominal", "underexposed", "recovered"}
    ),
    ("sensor_state", "imaging_mode"): frozenset(
        {"visible", "rgb", "infrared", "thermal", "night_vision"}
    ),
    ("facility_state", "availability"): frozenset(
        {"available", "unavailable", "reserved", "fault", "failed"}
    ),
    **{
        ("communication_state", field): _COMMUNICATION_STATUS_ENUMS
        for field in (
            "availability_state",
            "link_state",
            "link_status",
            "station_status",
            "status",
        )
    },
    **{
        ("communication_state", field): _COMMUNICATION_MODE_ENUMS
        for field in ("mode", "link_mode", "channel_mode", "handover_mode")
    },
    **{
        ("incident_state", field): _INCIDENT_FAULT_ENUMS
        for field in ("incident_type", "fault_type", "failure_type", "physical_state")
    },
    ("pedestrian_state", "health"): frozenset({"nominal", "injured"}),
    ("pedestrian_state", "posture"): frozenset(
        {"standing", "fallen", "lying", "prone"}
    ),
}

RUNTIME_STATE_COMMAND_FIELDS = frozenset(
    {
        ("control_state", "active_command"),
        ("control_state", "active_commands"),
        ("control_state", "commands"),
        ("mission_state", "active_command"),
        ("mission_state", "active_commands"),
        ("mission_state", "commands"),
        ("security_state", "active_command"),
        ("security_state", "active_commands"),
        ("security_state", "commands"),
    }
)


def unconsumed_runtime_state_paths(payload: Any) -> list[str]:
    """Return state paths that no formal runtime-state consumer reads."""

    if not isinstance(payload, Mapping):
        return []
    unconsumed: list[str] = []
    for family, family_value in payload.items():
        family_name = str(family)
        if family_name not in RUNTIME_STATE_CONSUMED_FIELDS:
            if isinstance(family_value, Mapping) and family_value:
                unconsumed.extend(f"{family_name}.{field}" for field in family_value)
            else:
                unconsumed.append(family_name)
            continue
        if not isinstance(family_value, Mapping):
            continue
        allowed = RUNTIME_STATE_CONSUMED_FIELDS[family_name]
        for field, value in family_value.items():
            field_name = str(field)
            path = f"{family_name}.{field_name}"
            if (
                field_name not in allowed
                and field_name not in RUNTIME_STATE_METADATA_FIELDS
            ):
                unconsumed.append(path)
                continue
            # All declared consumers read a scalar or a flat sequence at the
            # field boundary.  A nested mapping would therefore hide unread
            # authored state below an otherwise valid field name.
            if isinstance(value, Mapping):
                unconsumed.append(path)
    return sorted(set(unconsumed))


def invalid_runtime_state_value_paths(payload: Any) -> list[str]:
    """Return consumed fields whose values cannot produce a governed state."""

    if not isinstance(payload, Mapping):
        return []
    invalid: list[str] = []
    for family, family_value in payload.items():
        family_name = str(family)
        if not isinstance(family_value, Mapping):
            continue
        boolean_fields = RUNTIME_STATE_BOOLEAN_FIELDS.get(family_name, frozenset())
        for field, value in family_value.items():
            field_name = str(field)
            path = f"{family_name}.{field_name}"
            if field_name in RUNTIME_STATE_METADATA_FIELDS:
                if not isinstance(value, str) or not value.strip():
                    invalid.append(path)
                continue
            if field_name in boolean_fields:
                if not isinstance(value, bool):
                    invalid.append(path)
                continue
            allowed_enum = RUNTIME_STATE_ENUM_FIELDS.get((family_name, field_name))
            if allowed_enum is not None:
                if (
                    not isinstance(value, str)
                    or value.strip().lower() not in allowed_enum
                ):
                    invalid.append(path)
                continue
            if (family_name, field_name) in RUNTIME_STATE_COMMAND_FIELDS:
                members = [value] if isinstance(value, str) else value
                if (
                    not isinstance(members, (list, tuple))
                    or not members
                    or any(
                        not isinstance(item, str) or not item.strip()
                        for item in members
                    )
                ):
                    invalid.append(path)
                    continue
                allowed = (
                    _CONTROL_ENUMS
                    if family_name == "control_state"
                    else _MISSION_ENUMS
                    if family_name == "mission_state"
                    else _SECURITY_ENUMS
                )
                if any(item.strip().lower() not in allowed for item in members):
                    invalid.append(path)
                continue
            if family_name == "facility_state" and field_name in {
                "capacity",
                "request_count",
            }:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0
                ):
                    invalid.append(path)
                continue
            if family_name == "facility_state" and field_name == "requester_ids":
                if not isinstance(value, list) or any(
                    not isinstance(item, str) or not item for item in value
                ):
                    invalid.append(path)
                continue
            if family_name == "facility_state" and field_name in {
                "priority_granted_to",
                "arbitration_winner_id",
            }:
                if not isinstance(value, str) or not value.strip():
                    invalid.append(path)
                continue
            if family_name == "communication_state" and field_name in {
                "availability",
                "link_availability",
            }:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    invalid.append(path)
                continue
            if family_name == "incident_state" and field_name in {
                "hazard_concentration_ppm",
                "hazard_radius_m",
            }:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0
                ):
                    invalid.append(path)
                continue
    return sorted(set(invalid))


__all__ = [
    "RUNTIME_STATE_CONSUMED_FIELDS",
    "RUNTIME_STATE_BOOLEAN_FIELDS",
    "RUNTIME_STATE_ENUM_FIELDS",
    "RUNTIME_STATE_FIELDS",
    "RUNTIME_STATE_METADATA_FIELDS",
    "invalid_runtime_state_value_paths",
    "unconsumed_runtime_state_paths",
]
