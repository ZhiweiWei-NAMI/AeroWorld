"""Closed record structures for published compute/communication supplement output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Dataset.semantic_simulation.compute_comm import (
    ARTIFACT_SCHEMA_VERSION,
    ONTOLOGY_EVENT_ROLES,
    load_compute_comm_profile,
)


_PROFILE = (
    Path(__file__).resolve().parents[3]
    / "Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json"
)
_DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _object(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _array(item: dict[str, Any], *, minimum: int = 0,
           maximum: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": item, "minItems": minimum}
    if maximum is not None:
        schema["maxItems"] = maximum
    return schema


def _string() -> dict[str, Any]:
    return {"type": "string"}


def _tick() -> dict[str, Any]:
    return {"type": "integer", "minimum": 0}


def _float() -> dict[str, Any]:
    return {"type": "number", "x-source-value-types": ["float"]}


def _rounded() -> dict[str, Any]:
    return {"oneOf": [_float(), {"const": "unknown"}]}


def _handover_policy(*, selected: bool) -> dict[str, Any]:
    common = {
        "handover_margin_m": _float(),
        "minimum_dwell_ticks": {"type": "integer"},
        "last_switch_tick": {"oneOf": [_tick(), {"const": "none"}]},
        "candidate_station_id": _string(),
        "candidate_distance_m": _float() if selected else {"const": "unknown"},
    }
    variants = []
    for decision, switch_allowed, current_distance_known in (
        ("initial_station_or_unavailable", False, False),
        ("current_station_unavailable", True, False),
        ("switch", True, True),
        ("retain_current_station", False, True),
    ):
        variants.append(_object({
            **common,
            "current_station_distance_m": (
                _float() if current_distance_known else {"const": "unknown"}
            ),
            "distance_advantage_m": (
                _float() if current_distance_known else {"const": "unknown"}
            ),
            "dwell_elapsed_ticks": (
                _tick() if current_distance_known else {"const": "unknown"}
            ),
            "switch_allowed": {"const": switch_allowed},
            "decision": {"const": decision},
        }))
    return {"oneOf": variants}


def _station_operational_state(*, selected: bool) -> dict[str, Any]:
    source = (
        {"enum": ["observed_simulator_truth", "simulated_from_facility_anchor"]}
        if selected else {"const": "unavailable"}
    )
    common = {
        "state_reasons": _array(_string()),
        "source_class": source,
    }
    factors = ("operational_factor", "capacity_factor", "range_factor", "availability")
    unknown = _object({
        **common,
        "activity_type": {"const": "unknown"},
        "state_mode": {"const": "unknown"},
        "state_basis": {"const": "structured_communication_security_state"},
        **{name: _rounded() for name in factors},
    })
    if not selected:
        return unknown
    known = _object({
        **common,
        "activity_type": {"type": "string", "pattern": "^(?!unknown$).+$"},
        "state_mode": {"type": "string", "pattern": "^(?!unknown$).+$"},
        "state_basis": {
            "enum": ["structured_communication_security_state",
                     "governed_profile_no_runtime_override"]
        },
        **{name: _float() for name in factors},
    })
    return {"oneOf": [unknown, known]}


def _link_quality(*, selected: bool) -> dict[str, Any]:
    numeric = (
        "quality_score", "latency_ms", "packet_loss_ratio", "distance_m",
        "effective_range_m", "weather_attenuation", "operational_factor",
        "capacity_factor", "range_factor", "availability",
    )
    preserved_when_unknown = {
        "distance_m", "operational_factor", "capacity_factor", "range_factor",
        "availability",
    }
    unknown = _object({
        "quality_level": {"const": "unknown"},
        **{
            name: (
                (_float() if selected else {"const": "unknown"})
                if name == "distance_m" else _rounded()
            ) if name in preserved_when_unknown else {"const": "unknown"}
            for name in numeric
        },
    })
    if not selected:
        return unknown
    known = _object({
        "quality_level": {"enum": ["excellent", "good", "fair", "poor", "down"]},
        **{name: _float() for name in numeric},
    })
    return {"oneOf": [unknown, known]}


def _retransmission(*, source_required: bool) -> dict[str, Any]:
    if source_required:
        return _object({"count": {"const": "unknown"},
                        "reason": {"const": "unknown"}})
    return {"oneOf": [
        _object({"count": {"const": 0}, "reason": {"const": "none"}}),
        _object({"count": {"type": "integer", "minimum": 1},
                 "reason": {"const": "link_quality"}}),
    ]}


def _handover(*, source_required: bool) -> dict[str, Any]:
    common = {"previous_station_id": _string(), "current_station_id": _string()}
    active = {"const": "unknown"} if source_required else {"const": True}
    inactive = {"const": "unknown"} if source_required else {"const": False}
    return {"oneOf": [
        _object({**common, "active": inactive, "basis": {"const": "none"},
                 "association_source": {"const": "none"}}),
        _object({**common, "active": active,
                 "basis": {"const": "geometric_station_switch"},
                 "association_source": {"const": "none"}}),
        _object({**common, "active": active,
                 "basis": {"const": "governed_station_backup_transition"},
                 "association_source": {
                     "const": "communication.station_profiles.base_station.backup_station_id"
                 }}),
    ]}


def _communication_row(*, source_required: bool, station_case: str) -> dict[str, Any]:
    selected = station_case != "absent"
    literal_unknown_id = station_case == "selected_unknown_id"
    route_requires_source = not selected or literal_unknown_id
    station_id = (
        {"const": "unknown"} if route_requires_source
        else {"type": "string", "pattern": "^(?!unknown$).+$"}
    )
    station_source = (
        {"enum": ["observed_simulator_truth", "simulated_from_facility_anchor"]}
        if selected else {"const": "unavailable"}
    )
    props = {
        "schema_name": {"const": "communication_state"},
        "schema_version": {"const": ARTIFACT_SCHEMA_VERSION},
        "episode_id": _string(),
        "source_class": {"const": "simulated_derived"},
        "rule_id": {"const": "compute_comm_supplement.communication_state"},
        "rule_version": _string(),
        "model_id": _string(),
        "model_version": _string(),
        "source_refs": _array(_string(), minimum=4),
        "tick": _tick(),
        "station_id": station_id,
        "station_source": station_source,
        "session_id": _string(),
        "flow_id": _string(),
        "message_id": _string(),
        "channel_allocation_id": _string(),
        "transmission_attempt_id": _string(),
        "handover_id": _string(),
        "retransmission_id": _string(),
        "entity_id": _string(),
        "scope_active": {"const": True},
        "scope_activity_authority": {"const": "truth_frames.entities"},
        "heartbeat_age_ms": _rounded(),
        "channel": _object({
            "kind": _string(),
            "capacity_mbps": _rounded(),
            "nominal_capacity_mbps": _rounded(),
            "allocated_bandwidth_mbps": _rounded(),
            "queued_bandwidth_mbps": _rounded(),
        }),
        "bandwidth": _object({
            "requested_mbps": _rounded(),
            "allocated_mbps": _rounded(),
            "dropped_mbps": _rounded(),
        }),
        "route": _object({
            "hops": _array(_string(), minimum=2, maximum=2),
            "distance_m": _float() if selected else {"const": "unknown"},
            "requirement_status": {
                "const": "source_required" if route_requires_source else "not_required"
            },
        }),
        "retransmission": _retransmission(source_required=source_required),
        "handover": _handover(source_required=source_required),
        "handover_policy": (
            _handover_policy(selected=selected) if selected
            else _handover_policy(selected=False)["oneOf"][0]
        ),
        "station_operational_state": _station_operational_state(selected=selected),
        "link_quality": _link_quality(selected=selected),
        "quality_thresholds": _object({
            "latency_ms": _rounded(), "packet_loss_ratio": _rounded(),
        }),
        "requirement": _object({
            "status": {"const": "source_required" if source_required else "not_required"},
            "missing_parameters": _array(
                _string(), minimum=1 if source_required else 0,
                maximum=None if source_required else 0,
            ),
        }),
    }
    return _object(props)


def communication_state_record_schema() -> dict[str, Any]:
    """Return every published communication state branch, including source loss."""
    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/communication_state.record.schema.json",
        "title": "Published communication state record",
        "oneOf": [
            _communication_row(source_required=required, station_case=station_case)
            for station_case, required in (
                ("selected", False),
                ("selected", True),
                ("selected_unknown_id", False),
                ("selected_unknown_id", True),
                ("absent", True),
            )
        ],
    }


def compute_comm_event_record_schema() -> dict[str, Any]:
    """Return supplement event variants from its validated current profile."""
    profile = load_compute_comm_profile(_PROFILE)
    variants = []
    for predicate_id, rule in sorted(profile["event_rules"].items()):
        event_type = rule["event_type_id"]
        roles = set(rule["participant_bindings"])
        if roles != ONTOLOGY_EVENT_ROLES[event_type]:
            raise ValueError(f"event roles differ from producer ontology: {event_type}")
        variants.append(_object({
            "schema_name": {"const": "compute_comm_event"},
            "schema_version": {"const": ARTIFACT_SCHEMA_VERSION},
            "episode_id": _string(),
            "source_class": {"const": "simulated_derived"},
            "rule_id": {"const": rule["rule_id"]},
            "rule_version": {"const": rule["rule_version"]},
            "model_id": _string(),
            "model_version": _string(),
            "source_refs": _array(_string(), minimum=2, maximum=2),
            "event_id": _string(),
            "event_type_id": {"const": event_type},
            "derivation_kind": {"const": "predicate_rising"},
            "trigger_tick": _tick(),
            "source_predicate_id": {"const": predicate_id},
            "transition": _object({
                "from_tick": _tick(),
                "to_tick": _tick(),
                "from_value": {"const": "false"},
                "to_value": {"const": "true"},
            }),
            "bindings": _object({role: _string() for role in sorted(roles)}),
        }))
    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/compute_comm_event.record.schema.json",
        "title": "Published compute communication supplement event record",
        "oneOf": variants,
    }
