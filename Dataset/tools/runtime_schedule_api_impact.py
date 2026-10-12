"""Scenario runtime-state reachability gate for the single core engine.

Each authored runtime-state field must reach a domain-state producer consumed
by the objective-state adapter.  The same EPI contract is also checked against
the sole 79-predicate registry.  This gate contains no legacy API catalog.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.semantic_simulation.control_response_state import (  # noqa: E402
    SUPPORTED_API_VALUE_KEYS as CONTROL_RESPONSE_API_VALUE_KEYS,
)
from Dataset.semantic_simulation.observable_state_completion import (  # noqa: E402
    _COMMUNICATION_BOOL_FIELDS,
    _CONTROL_BOOL_FIELDS,
    _INCIDENT_BOOL_FIELDS,
    _MISSION_BOOL_FIELDS,
    _NAVIGATION_BOOL_FIELDS,
    _SECURITY_BOOL_FIELDS,
)
from Dataset.semantic_truth.core_semantic_registry import (  # noqa: E402
    get_core_predicate_ids,
)
from runtime_state_contract import RUNTIME_STATE_METADATA_FIELDS  # noqa: E402


@dataclass(frozen=True)
class UpstreamValuePath:
    observation_family: str
    value_field: str
    producer: str

    @property
    def source_path(self) -> str:
        return f"domain_state_observation:{self.observation_family}.values.{self.value_field}"


@dataclass(frozen=True)
class RuntimeFieldImpact:
    field_path: str
    upstream_paths: tuple[str, ...]
    producer_reachable: bool
    metadata_only: bool = False

    @property
    def has_core_impact(self) -> bool:
        return self.metadata_only or self.producer_reachable


@dataclass(frozen=True)
class ScenarioRuntimeApiImpact:
    scenario_id: str
    required_core_predicate_ids: tuple[str, ...]
    field_impacts: tuple[RuntimeFieldImpact, ...]

    @property
    def unreachable_fields(self) -> tuple[RuntimeFieldImpact, ...]:
        return tuple(
            impact for impact in self.field_impacts if not impact.has_core_impact
        )


def _add(
    mapping: dict[str, set[UpstreamValuePath]],
    runtime_field: str,
    observation_family: str,
    value_field: str,
    producer: str,
) -> None:
    mapping[runtime_field].add(
        UpstreamValuePath(
            observation_family=observation_family,
            value_field=value_field,
            producer=producer,
        )
    )


def _build_runtime_upstream_index() -> dict[str, set[UpstreamValuePath]]:
    mapping: dict[str, set[UpstreamValuePath]] = defaultdict(set)

    for field in _MISSION_BOOL_FIELDS:
        _add(
            mapping,
            f"mission_state.{field}",
            "observable_mission_state",
            field,
            "observable_state_completion",
        )
    for field in _NAVIGATION_BOOL_FIELDS:
        _add(
            mapping,
            f"navigation_state.{field}",
            "observable_navigation_state",
            field,
            "observable_state_completion",
        )
    for field in _COMMUNICATION_BOOL_FIELDS:
        _add(
            mapping,
            f"communication_state.{field}",
            "observable_communication_state",
            field,
            "observable_state_completion",
        )
    for field in _INCIDENT_BOOL_FIELDS:
        _add(
            mapping,
            f"incident_state.{field}",
            "observable_incident_state",
            field,
            "observable_state_completion",
        )
    for field in _CONTROL_BOOL_FIELDS:
        _add(
            mapping,
            f"control_state.{field}",
            "observable_control_state",
            field,
            "observable_state_completion",
        )
    for field in _SECURITY_BOOL_FIELDS:
        _add(
            mapping,
            f"security_state.{field}",
            "observable_security_state",
            field,
            "observable_state_completion",
        )

    communication_values = {"backup_link_active", "station_unavailable"}
    mission_values = {"mission_abort_active"}
    navigation_values = {"path_deviation", "route_recovered", "sustained_drift"}
    security_values = {
        "command_integrity_violation",
        "command_lockout",
        "gcs_compromised",
        "jamming_active",
        "unauthorized_command",
    }
    for value_key in sorted(set(CONTROL_RESPONSE_API_VALUE_KEYS.values())):
        if value_key in communication_values:
            runtime_family = "communication_state"
        elif value_key in mission_values:
            runtime_family = "mission_state"
        elif value_key in navigation_values:
            runtime_family = "navigation_state"
        elif value_key in security_values:
            runtime_family = "security_state"
        else:
            runtime_family = "control_state"
        _add(
            mapping,
            f"{runtime_family}.{value_key}",
            "control_response_state",
            value_key,
            "control_response_state",
        )

    for field in (
        "safe_hold",
        "mode",
        "active_mode",
        "command_mode",
        "active_command",
        "active_commands",
        "commands",
    ):
        for value_key in (
            "alternate_reroute_active",
            "alternate_rth_active",
            "altitude_corrective_maneuver_active",
            "deconfliction_active",
            "diversion_active",
            "divert_active",
            "evasion_active",
            "hold_active",
            "pull_up_active",
            "reroute_active",
            "resequence_active",
            "rth_active",
            "safe_hold_active",
            "slowdown_active",
        ):
            _add(
                mapping,
                f"control_state.{field}",
                "control_response_state",
                value_key,
                "control_response_state",
            )
    for field in (
        "mode",
        "status",
        "phase",
        "active_command",
        "active_commands",
        "commands",
        "mission_abort_active",
    ):
        _add(
            mapping,
            f"mission_state.{field}",
            "control_response_state",
            "mission_abort_active",
            "control_response_state",
        )
    for field in (
        "mode",
        "status",
        "threat",
        "condition",
        "active_command",
        "active_commands",
        "commands",
    ):
        for value_key in (
            "command_integrity_violation",
            "command_lockout",
            "gcs_compromised",
            "jamming_active",
            "unauthorized_command",
        ):
            _add(
                mapping,
                f"security_state.{field}",
                "control_response_state",
                value_key,
                "control_response_state",
            )
    for field in (
        "alternate_link_active",
        "backup_active",
        "backup_link_active",
        "handover_active",
    ):
        _add(
            mapping,
            f"communication_state.{field}",
            "control_response_state",
            "backup_link_active",
            "control_response_state",
        )
    for field in ("mode", "link_mode", "channel_mode", "handover_mode"):
        _add(
            mapping,
            f"communication_state.{field}",
            "communication_state",
            "station_operational_state",
            "compute_comm_supplement",
        )
    for field in (
        "availability",
        "availability_state",
        "available",
        "communication_unavailable",
        "link_availability",
        "link_available",
        "link_state",
        "link_status",
        "station_available",
        "station_status",
        "station_unavailable",
        "status",
    ):
        _add(
            mapping,
            f"communication_state.{field}",
            "control_response_state",
            "station_unavailable",
            "control_response_state",
        )

    sensor_paths = {
        "exposure_recovered": ("exposure_recovered",),
        "image_underexposed": ("image_underexposed", "underexposed"),
        "underexposed": ("image_underexposed", "underexposed"),
        "exposure_status": ("image_underexposed", "exposure_status"),
        "infrared_mode_active": ("infrared_mode_active", "infrared"),
        "ir_active": ("infrared_mode_active",),
        "infrared": ("infrared_mode_active", "infrared"),
        "imaging_mode": ("infrared_mode_active", "imaging_mode"),
        "intruder_detected": ("intruder_detected",),
        "sensor_fault": ("sensor_fault",),
    }
    for field, value_fields in sensor_paths.items():
        for value_field in value_fields:
            _add(
                mapping,
                f"sensor_state.{field}",
                "observable_sensor_state",
                value_field,
                "observable_state_completion",
            )

    facility_paths = {
        "allocation_failed": (("observable_pad_facility", "allocation_failed"),),
        "allocation_stale": (("observable_pad_facility", "allocation_stale"),),
        "arbitration_active": (("observable_pad_facility", "arbitration_active"),),
        "arbitration_winner_id": (("observable_pad_facility", "pad_priority_granted"),),
        "availability": (
            ("pad_facility", "availability"),
            ("observable_pad_facility", "availability"),
        ),
        "backup_charger_accepted": (
            ("observable_pad_facility", "backup_charger_accepted"),
        ),
        "capacity": (
            ("pad_facility", "capacity"),
            ("observable_pad_facility", "capacity"),
        ),
        "contention": (
            ("pad_facility", "contention"),
            ("observable_pad_facility", "contention"),
        ),
        "fault": (
            ("pad_facility", "fault_active"),
            ("observable_pad_facility", "fault"),
            ("signal_queue", "controller_health"),
            ("signal_queue", "malfunction"),
        ),
        "priority_granted": (("observable_pad_facility", "pad_priority_granted"),),
        "priority_granted_to": (("observable_pad_facility", "pad_priority_granted"),),
        "request_count": (
            ("observable_pad_facility", "simultaneous_pad_requests"),
            ("observable_pad_facility", "request_count"),
        ),
        "requester_ids": (
            ("pad_facility", "requester_ids"),
            ("observable_pad_facility", "requester_ids"),
        ),
        "reserved": (
            ("pad_facility", "reserved"),
            ("observable_pad_facility", "reserved"),
        ),
    }
    for field, upstreams in facility_paths.items():
        for family, value_field in upstreams:
            _add(
                mapping,
                f"facility_state.{field}",
                family,
                value_field,
                "domain_state/observable_state_completion",
            )

    incident_paths = {
        "detection_active": (
            ("medical_response", "detection_active"),
            ("observable_incident_state", "detection_active"),
        ),
        "dispatch_active": (
            ("medical_response", "dispatch_active"),
            ("observable_incident_state", "dispatch_active"),
        ),
        "handoff_complete": (
            ("medical_response", "handoff_complete"),
            ("hazmat_response_state", "handoff_complete"),
            ("observable_incident_state", "handoff_complete"),
        ),
        "hazard_concentration_ppm": (
            ("observable_environment", "hazard_concentration_ppm"),
        ),
        "hazard_leak_active": (
            ("observable_environment", "hazard_source_active"),
            ("observable_incident_state", "hazard_leak_active"),
        ),
        "hazard_radius_m": (("observable_environment", "hazard_radius_m"),),
        "hazard_source_active": (("observable_environment", "hazard_source_active"),),
        "hazard_spread_active": (
            ("observable_incident_state", "hazard_spread_active"),
        ),
        "hazmat_resolved": (
            ("hazmat_response_state", "hazmat_resolved"),
            ("observable_incident_state", "hazmat_resolved"),
        ),
        "isolation_active": (
            ("hazmat_response_state", "isolation_active"),
            ("observable_incident_state", "isolation_active"),
        ),
        "landing_failed": (("observable_incident_state", "landing_failed"),),
        "lockdown_clear": (
            ("lockdown_control", "clear"),
            ("observable_incident_state", "lockdown_clear"),
        ),
        "manual_control_active": (
            ("observable_incident_state", "manual_control_active"),
        ),
        "physical_fault_active": (("forced_landing_state", "physical_fault_active"),),
        "requires_reroute": (
            ("lockdown_control", "requires_reroute"),
            ("observable_incident_state", "requires_reroute"),
        ),
        "responder_arrived": (
            ("medical_response", "responder_arrived"),
            ("hazmat_response_state", "responder_arrived"),
            ("observable_incident_state", "responder_arrived"),
        ),
        "temporary_lockdown_active": (
            ("observable_incident_state", "temporary_lockdown_active"),
        ),
    }
    for field, upstreams in incident_paths.items():
        for family, value_field in upstreams:
            _add(
                mapping,
                f"incident_state.{field}",
                family,
                value_field,
                "domain_state/observable_state_completion",
            )

    pedestrian_paths = {
        "evacuation_active": (
            ("observable_pedestrian_response", "evacuation_active"),
            ("crowd_evacuation", "evacuation_active"),
        ),
        "fallen": (
            ("medical_response", "confirmed_fall"),
            ("observable_pedestrian_response", "fallen"),
        ),
        "health": (
            ("medical_response", "health_state"),
            ("observable_pedestrian_response", "health"),
        ),
        "injured": (
            ("medical_response", "health_state"),
            ("observable_pedestrian_response", "injured"),
        ),
        "jaywalking": (("observable_pedestrian_response", "jaywalking"),),
        "posture": (
            ("medical_response", "posture"),
            ("observable_pedestrian_response", "posture"),
        ),
        "retreating": (
            ("observable_pedestrian_response", "pedestrian_retreating"),
            ("observable_pedestrian_response", "retreating"),
        ),
        "safe_zone_reached": (
            ("observable_pedestrian_response", "safe_zone_reached"),
            ("crowd_evacuation", "safe_zone_reached"),
        ),
    }
    for field, upstreams in pedestrian_paths.items():
        for family, value_field in upstreams:
            _add(
                mapping,
                f"pedestrian_state.{field}",
                family,
                value_field,
                "domain_state/observable_state_completion",
            )

    vehicle_paths = {
        "ambulance_passage_completed": (
            ("observable_vehicle_response", "ambulance_passage_completed"),
            ("ambulance_priority", "clearance_reached"),
        ),
        "brake_active": (("observable_vehicle_response", "vehicle_braking"),),
        "braking": (
            ("observable_vehicle_response", "vehicle_braking"),
            ("observable_vehicle_response", "braking"),
        ),
        "collision_contact": (("observable_vehicle_response", "collision_contact"),),
        "emergency_braking_active": (
            ("observable_vehicle_response", "vehicle_braking"),
        ),
        "emergency_stop": (
            ("observable_vehicle_response", "vehicle_emergency_stop_active"),
            ("observable_vehicle_response", "emergency_stop"),
        ),
        "emergency_stop_active": (
            ("observable_vehicle_response", "vehicle_emergency_stop_active"),
        ),
        "minimal_risk_maneuver_active": (
            ("av_safe_stop", "mrm_active"),
            ("observable_vehicle_response", "minimal_risk_maneuver_active"),
        ),
        "minimal_risk_stop_active": (
            ("observable_vehicle_response", "vehicle_emergency_stop_active"),
        ),
        "near_miss_classified": (
            ("observable_vehicle_response", "near_miss_classified"),
        ),
        "safe_stop_failed": (
            ("av_safe_stop", "safe_stop_failed"),
            ("observable_vehicle_response", "safe_stop_failed"),
        ),
        "sensor_fault": (
            ("av_safe_stop", "sensor_fault_active"),
            ("observable_vehicle_response", "sensor_fault"),
        ),
        "stopped_safe": (
            ("av_safe_stop", "stopped_safe"),
            ("observable_vehicle_response", "stopped_safe"),
        ),
        "warning_active": (
            ("av_safe_stop", "warning_effective"),
            ("observable_vehicle_response", "warning_active"),
        ),
        "yielding": (
            ("observable_vehicle_response", "yielding"),
            ("ambulance_priority", "yielding_vehicle_count"),
        ),
    }
    for field, upstreams in vehicle_paths.items():
        for family, value_field in upstreams:
            _add(
                mapping,
                f"vehicle_state.{field}",
                family,
                value_field,
                "domain_state/observable_state_completion",
            )

    security_domain_paths = {
        "command_integrity_violation": (
            ("security_command", "command_integrity_score"),
        ),
        "command_lockout": (
            ("security_command", "lockout_active"),
            ("security_command", "lockout_duration_ticks"),
        ),
        "gcs_compromised": (("security_command", "auth_score"),),
        "jamming_active": (
            ("security_command", "jamming_indicator"),
            ("security_command", "spectrum_interference_ratio"),
        ),
        "lockout_active": (
            ("security_command", "lockout_active"),
            ("security_command", "lockout_duration_ticks"),
        ),
        "unauthorized_command": (
            ("security_command", "auth_score"),
            ("security_command", "command_source_valid"),
        ),
    }
    for field, upstreams in security_domain_paths.items():
        for family, value_field in upstreams:
            _add(
                mapping, f"security_state.{field}", family, value_field, "domain_state"
            )

    navigation_domain_fields = (
        "geofence_violation",
        "gnss_spoofed",
        "mission_recovered",
        "multipath_warning",
        "operational_state",
        "relocalization_complete",
        "visual_relocalization",
        "visual_relocalization_active",
    )
    for field in navigation_domain_fields:
        for value_field in (
            "quality_level",
            "position_error_m",
            "visual_relocalization",
            "mission_recovered",
            "operational_state",
        ):
            _add(
                mapping,
                f"navigation_state.{field}",
                "gnss_navigation",
                value_field,
                "domain_state",
            )

    return mapping


def collect_schedule_runtime_field_paths(schedule: Any) -> tuple[str, ...]:
    fields: set[str] = set()
    for state_by_entity in _mapping_values(
        getattr(schedule, "initial_entity_states", {})
    ):
        _collect_payload_fields(state_by_entity, fields)
    for transitions in _mapping_values(
        getattr(schedule, "event_state_transitions", {})
    ):
        if not isinstance(transitions, Sequence) or isinstance(
            transitions, (str, bytes, bytearray)
        ):
            continue
        for transition in transitions:
            if isinstance(transition, Mapping):
                _collect_payload_fields(transition.get("state_patch"), fields)
    return tuple(sorted(fields))


def _required_core_predicate_ids_for_epi(
    contract: Mapping[str, Any],
    scenario_id: str,
) -> tuple[str, ...]:
    epi_contracts = contract.get("epi_contracts")
    event_families = contract.get("event_families")
    if not isinstance(epi_contracts, Mapping) or not isinstance(
        event_families, Mapping
    ):
        raise ValueError("objective contract lacks epi_contracts or event_families")
    base_epi_id = scenario_id.rsplit("__seed", 1)[0]
    epi = epi_contracts.get(base_epi_id)
    if not isinstance(epi, Mapping):
        raise KeyError(base_epi_id)
    required: set[str] = set()
    for stage in epi.get("required_chain", ()):
        if not isinstance(stage, Mapping):
            continue
        family_ids = [stage.get("event_family_id")]
        alternatives = stage.get("alternative_event_family_ids")
        if isinstance(alternatives, Sequence) and not isinstance(
            alternatives, (str, bytes, bytearray)
        ):
            family_ids.extend(alternatives)
        for family_id in family_ids:
            family = event_families.get(family_id)
            if not isinstance(family, Mapping):
                continue
            required.update(
                str(predicate_id)
                for predicate_id in family.get("required_predicates", ())
                if isinstance(predicate_id, str)
            )
    core_ids = set(get_core_predicate_ids())
    unknown = sorted(required - core_ids)
    if unknown:
        raise ValueError(
            f"EPI {base_epi_id} requires predicates outside the 79-predicate registry: {unknown}"
        )
    return tuple(sorted(required))


def scenario_runtime_api_impact(
    schedule: Any,
    contract: Mapping[str, Any],
) -> ScenarioRuntimeApiImpact:
    scenario_id = str(getattr(schedule, "scenario_id", "") or "")
    required_core_predicate_ids = _required_core_predicate_ids_for_epi(
        contract,
        scenario_id,
    )
    upstream_index = _build_runtime_upstream_index()
    impacts: list[RuntimeFieldImpact] = []
    for field_path in collect_schedule_runtime_field_paths(schedule):
        family, _, field = field_path.partition(".")
        if field in RUNTIME_STATE_METADATA_FIELDS:
            impacts.append(
                RuntimeFieldImpact(
                    field_path=field_path,
                    upstream_paths=(),
                    producer_reachable=False,
                    metadata_only=True,
                )
            )
            continue
        upstreams = tuple(
            sorted(
                upstream_index.get(field_path, ()), key=lambda item: item.source_path
            )
        )
        impacts.append(
            RuntimeFieldImpact(
                field_path=f"{family}.{field}",
                upstream_paths=tuple(upstream.source_path for upstream in upstreams),
                producer_reachable=bool(upstreams),
            )
        )
    return ScenarioRuntimeApiImpact(
        scenario_id=scenario_id,
        required_core_predicate_ids=required_core_predicate_ids,
        field_impacts=tuple(impacts),
    )


def validate_runtime_schedule_api_impact(
    schedules: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[ScenarioRuntimeApiImpact, ...]:
    return tuple(
        scenario_runtime_api_impact(schedule, contract)
        for _, schedule in sorted(schedules.items())
    )


def require_runtime_schedule_api_impact(
    schedules: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[ScenarioRuntimeApiImpact, ...]:
    reports = validate_runtime_schedule_api_impact(schedules, contract)
    failures = [report for report in reports if report.unreachable_fields]
    if failures:
        raise RuntimeError(format_runtime_schedule_api_impact_failures(failures))
    return reports


def format_runtime_schedule_api_impact_failures(
    failures: Sequence[ScenarioRuntimeApiImpact],
) -> str:
    lines = [
        "Runtime-state schedule reachability gate failed: every non-metadata field must reach a producer consumed by the single objective-state adapter."
    ]
    for report in failures:
        lines.append(
            f"{report.scenario_id}: required_core_predicate_ids={list(report.required_core_predicate_ids)}; unreachable_fields:"
        )
        for impact in report.unreachable_fields:
            lines.append(
                "  "
                + impact.field_path
                + f" upstream_paths={list(impact.upstream_paths)}"
            )
    return "\n".join(lines)


def _collect_payload_fields(payload: Any, fields: set[str]) -> None:
    if not isinstance(payload, Mapping):
        return
    for family, family_value in payload.items():
        if not isinstance(family, str) or not isinstance(family_value, Mapping):
            continue
        for field in family_value:
            fields.add(f"{family}.{field}")


def _mapping_values(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(value.values())


__all__ = [
    "RuntimeFieldImpact",
    "ScenarioRuntimeApiImpact",
    "collect_schedule_runtime_field_paths",
    "format_runtime_schedule_api_impact_failures",
    "require_runtime_schedule_api_impact",
    "scenario_runtime_api_impact",
    "validate_runtime_schedule_api_impact",
]
