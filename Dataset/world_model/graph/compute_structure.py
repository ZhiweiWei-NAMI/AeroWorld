"""Closed record structure for published compute-state supplement rows."""

from __future__ import annotations


def _object(properties: dict, *, required: tuple[str, ...] | None = None) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties) if required is None else list(required),
        "properties": properties,
    }


def _rounded() -> dict:
    return {
        "oneOf": [
            {"type": "number", "x-source-value-types": ["float"]},
            {"const": "unknown"},
        ],
    }


def _numeric_object(*names: str) -> dict:
    return _object({name: _rounded() for name in names})


def _deadline(*, missing_source: bool) -> dict:
    unknown = _object({
        "queue_started_tick": {"const": "unknown"},
        "deadline_tick": {"const": "unknown"},
        "slack_ticks": {"const": "unknown"},
        "status": {"const": "unknown"},
        "requirement_status": {"const": "missing_source_record"},
    })
    if missing_source:
        return unknown
    inactive = _object({
        "queue_started_tick": {"type": "null"},
        "deadline_tick": {"type": "null"},
        "slack_ticks": {"type": "integer", "minimum": 0},
        "status": {"const": "inactive"},
        "requirement_status": {"const": "not_applicable"},
    })
    active = _object({
        "queue_started_tick": {"type": "integer", "minimum": 0},
        "deadline_tick": {"type": "integer", "minimum": 0},
        "slack_ticks": {"type": "integer", "minimum": 0},
        "status": {"enum": ["within_budget", "overdue"]},
        "requirement_status": {"const": "not_required"},
    })
    return {"oneOf": [inactive, active, unknown]}


def _compute_row(*, missing_source: bool) -> dict:
    failure = _object({
        "mode": {"const": "missing_source_record"} if missing_source else {
            "enum": ["none", "capacity_pressure", "capacity_queue_overflow",
                     "node_unavailable"]},
        "requirement_status": {
            "const": "missing_source_record" if missing_source else "not_required"
        },
        "missing_parameters": {
            "type": "array", "items": {"type": "string"},
            "minItems": 1 if missing_source else 0,
            **({} if missing_source else {"maxItems": 0}),
        },
    })
    properties = {
        "schema_name": {"const": "compute_state"},
        "schema_version": {"const": "2.0.0"},
        "episode_id": {"type": "string"},
        "tick": {"type": "integer", "minimum": 0},
        "entity_id": {"type": "string"},
        "node_id": {"type": "string"},
        "node_kind": {"enum": ["uav_edge_node", "vehicle_edge_node"]},
        "queue_id": {"type": "string"},
        "execution_id": {"type": "string"},
        "allocation_id": {"type": "string"},
        "resource_id": {"type": "string"},
        "failure_state_id": {"type": "string"},
        "offload_id": {"type": "string"},
        "migration_id": {"type": "string"},
        "task_id": {"type": "string"},
        "task_kind": {"type": "string"},
        "execution_state": {"const": "unknown"} if missing_source else {
            "enum": ["running", "running_offloaded", "queued", "failed", "migrating"]},
        "cpu_utilization_pct": _rounded(),
        "gpu_utilization_pct": _rounded(),
        "memory_used_mb": _rounded(),
        "queue_depth": {"const": "unknown"} if missing_source else {
            "type": "integer", "minimum": 0},
        "declared_capacity": _numeric_object(
            "cpu_cores", "gpu_units", "memory_mb", "max_queue_depth"),
        "capacity": _numeric_object(
            "cpu_cores", "gpu_units", "memory_mb", "max_queue_depth"),
        "resource_request": _numeric_object("cpu_cores", "gpu_units", "memory_mb"),
        "allocation": _numeric_object("cpu_cores", "gpu_units", "memory_mb"),
        "resource_balance": _numeric_object(
            "cpu_requested", "cpu_allocated", "cpu_queued", "cpu_offloaded",
            "cpu_utilization_pct", "gpu_requested", "gpu_allocated", "gpu_queued",
            "gpu_utilization_pct", "memory_requested", "memory_allocated",
            "memory_queued", "memory_utilization_pct", "queue_depth"),
        "inbound_allocation": _numeric_object("cpu_cores"),
        "node_availability": _object({
            "available": {"type": "boolean"},
            "source": {"const": "deterministic_maintenance_schedule"},
        }),
        "offload": _object({
            "enabled": {"type": "boolean"},
            "source_node_id": {"type": "string"},
            "target_node_id": {"type": "string"},
            "requested_cpu_cores": _rounded(),
            "accepted_cpu_cores": _rounded(),
            "reason": {"enum": ["local_capacity_pressure", "not_required"]},
        }),
        "migration": _object({
            "active": {"type": "boolean"},
            "source_node_id": {"type": "string"},
            "target_node_id": {"type": "string"},
            "transferred_cpu_cores": _rounded(),
        }),
        "failure": failure,
        "deadline": _deadline(missing_source=missing_source),
        "source_class": {"const": "simulated_derived"},
        "rule_id": {"const": "compute_comm_supplement.compute_state"},
        "rule_version": {"type": "string"},
        "model_id": {"type": "string"},
        "model_version": {"type": "string"},
        "source_refs": {"type": "array", "items": {"type": "string"},
                        "minItems": 2, "maxItems": 2},
    }
    return _object(properties)


def compute_state_record_schema() -> dict:
    """Return the published compute-state structure, including source absence."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/compute_state.record.schema.json",
        "title": "Published compute state record",
        "oneOf": [
            _compute_row(missing_source=False),
            _compute_row(missing_source=True),
        ],
    }
