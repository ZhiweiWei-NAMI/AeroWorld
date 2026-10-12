"""Closed record structures for compute/communication predicate supplements."""

from __future__ import annotations

from typing import Any, Mapping

from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates


_TRUTH_VALUES = ["true", "false", "unknown", "out_of_scope"]
_COMPUTE_PREDICATE_IDS = (
    "compute.compute_node_unavailable",
    "compute.compute_resource_over_allocated",
    "compute.compute_task_deadline_missed",
    "compute.compute_task_execution_failed",
    "compute.compute_task_migrating",
    "compute.compute_task_offloaded",
    "compute.compute_task_queued",
)
_COMMUNICATION_PREDICATE_IDS = (
    "communication.channel_bandwidth_over_allocated",
    "communication.communication_session_interrupted",
    "communication.data_flow_latency_above_threshold",
    "communication.handover_in_progress",
    "communication.link_degraded",
    "communication.link_unavailable",
    "communication.message_transmission_failed",
    "communication.packet_loss_above_threshold",
    "communication.retransmission_in_progress",
)


def _object(properties: Mapping[str, dict], *, required: tuple[str, ...] | None = None) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties) if required is None else list(required),
        "properties": dict(properties),
    }


def _string_list(*, min_items: int, max_items: int | None = None) -> dict:
    result: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": min_items,
    }
    if max_items is not None:
        result["maxItems"] = max_items
    return result


def _role_class(value: Any) -> str:
    value = str(value)
    return value if ":" in value else f"world:{value}"


def _registry_roles() -> dict[str, tuple[tuple[str, str], ...]]:
    roles: dict[str, tuple[tuple[str, str], ...]] = {}
    for template in get_core_predicate_templates():
        predicate_id = str(template["id"])
        if not predicate_id.startswith(("compute.", "communication.")):
            continue
        roles[predicate_id] = tuple(
            (str(role["key"]), _role_class(role["class"]))
            for role in template["argument_roles"]
        )
    expected = set(_COMPUTE_PREDICATE_IDS) | set(_COMMUNICATION_PREDICATE_IDS)
    if set(roles) != expected:
        raise ValueError(
            "compute/communication predicate registry differs from supplement producer: "
            f"expected={sorted(expected)}, actual={sorted(roles)}"
        )
    return roles


def _participant_object(names: tuple[str, ...]) -> dict:
    return _object({name: {"type": "string"} for name in names})


def _predicate_variant(predicate_id: str, roles: tuple[tuple[str, str], ...], participants: dict) -> dict:
    return _object({
        "schema_name": {"const": "compute_comm_predicate_truth"},
        "schema_version": {"const": "2.0.0"},
        "episode_id": {"type": "string"},
        "source_class": {"const": "simulated_derived"},
        "rule_id": {"type": "string"},
        "rule_version": {"type": "string"},
        "model_id": {"type": "string"},
        "model_version": {"type": "string"},
        "source_refs": _string_list(min_items=1, max_items=1),
        "predicate_truth_id": {"type": "string"},
        "tick": {"type": "integer", "minimum": 0},
        "predicate_id": {"const": predicate_id},
        "bindings": _object({role: {"type": "string"} for role, _ in roles}),
        "binding_ontology_classes": _object({
            role: {"const": ontology_class} for role, ontology_class in roles
        }),
        "event_participants": participants,
        "binding_status": {"enum": ["complete", "incomplete_binding"]},
        "value": {"enum": _TRUTH_VALUES},
        "requirement_status": {
            "enum": ["not_required", "source_required", "missing_source_record"]
        },
    })


def predicate_truth_record_schema() -> dict:
    """Return the registry-aligned closed schema for one predicate truth row."""
    roles = _registry_roles()
    compute_participants = _participant_object(
        ("node", "resource", "execution", "task", "failure_state")
    )
    communication_participants = _participant_object(
        (
            "actor", "station", "session", "flow", "handover", "source_station",
            "target_station", "retransmission", "original_attempt", "message",
            "attempt", "allocation",
        )
    )
    variants = []
    for predicate_id in _COMPUTE_PREDICATE_IDS:
        variants.append(_predicate_variant(predicate_id, roles[predicate_id], compute_participants))
    for predicate_id in _COMMUNICATION_PREDICATE_IDS:
        variants.append(_predicate_variant(
            predicate_id, roles[predicate_id], communication_participants
        ))
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/compute_comm_predicate_truth.record.schema.json",
        "title": "Published compute communication predicate truth record",
        "oneOf": variants,
    }


def _matrix_scope_variant(scope_type: str, predicate_ids: tuple[str, ...]) -> dict:
    return _object({
        "scope_type": {"const": scope_type},
        "scope_entity_id": {"type": "string"},
        "scope_active": {"type": "boolean"},
        "workload_active": {"type": "boolean"},
        "predicate_values": _object({
            predicate_id: {"enum": _TRUTH_VALUES}
            for predicate_id in predicate_ids
        }),
    })


def predicate_truth_matrix_record_schema() -> dict:
    """Return the closed scope-variant schema for one matrix tick row."""
    scope_item = {
        "oneOf": [
            _matrix_scope_variant("compute_node", _COMPUTE_PREDICATE_IDS),
            _matrix_scope_variant("uav", _COMMUNICATION_PREDICATE_IDS),
        ]
    }
    properties = {
        "schema_name": {"const": "compute_comm_predicate_truth_matrix"},
        "schema_version": {"const": "2.0.0"},
        "episode_id": {"type": "string"},
        "source_class": {"const": "simulated_derived"},
        "rule_id": {"const": "compute_comm_supplement.full_predicate_matrix"},
        "rule_version": {"type": "string"},
        "model_id": {"type": "string"},
        "model_version": {"type": "string"},
        "source_refs": _string_list(min_items=2, max_items=2),
        "matrix_row_id": {"type": "string"},
        "tick": {"type": "integer", "minimum": 0},
        "scope_values": {"type": "array", "items": scope_item, "minItems": 1},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/compute_comm_predicate_truth_matrix.record.schema.json",
        "title": "Published compute communication predicate truth matrix row",
        **_object(properties),
    }
