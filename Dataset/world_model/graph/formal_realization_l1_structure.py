"""Closed formal event-realization row shared by the seven producer levels.

``batch_generate.build_event_realizations`` writes the source row;
``convert_to_render_ready.reconcile_event_realizations_to_render_truth`` adds
render snapshots and conditional reconciliation fields.  Capture filtering
copies this JSONL unchanged.  The producer branches on action type, not level.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from Dataset.tools.runtime_state_contract import RUNTIME_STATE_FIELDS


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_LEVELS = frozenset({"L1", "L2", "L3", "L4", "L5", "L6", "X"})
_ALIGN_POLICY = "sumo_replaced_source_vehicle_source_snapshot_alignment_v1"
_TRIGGER_POLICY = "render_truth_proximity_trigger_replaces_source_dispatch_v1"
_CASCADE_POLICY = "render_truth_event_fired_after_cascade_v1"


def _object(properties: dict[str, Any], *, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def _tick() -> dict[str, Any]:
    return {"type": "integer", "minimum": 0}


def _string() -> dict[str, Any]:
    return {"type": "string"}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"oneOf": [schema, {"type": "null"}]}


def _vector3() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "prefixItems": [{"type": "number"} for _ in range(3)],
        "items": False,
    }


def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": _string()}


def _action_schema() -> dict[str, Any]:
    common = {
        "action_id": _string(),
        "entity_id": _string(),
        "dispatch_tick": _tick(),
        "scheduled_tick": _tick(),
        "status": _string(),
        "result_tick": _tick(),
        "evidence_tick": _tick(),
    }
    variants = []
    variants.append(_object({
        **common,
        "action_type": {"const": "move_entity"},
        "path_length_m": _nullable({"type": "number"}),
        "first_motion_tick": _nullable(_tick()),
        "first_capture_motion_tick": _nullable(_tick()),
        "terminal_tick": _nullable(_tick()),
        "terminal_required": {"type": "boolean"},
        "terminal_enu_m": _nullable(_vector3()),
    }))
    for action_type in ("set_visual_state", "set_pedestrian_activity", "spawn_entity"):
        variants.append(_object({
            **common,
            "action_type": {"const": action_type},
            "expected_state": _nullable(_string()),
        }))
    variants.append(_object({
        **common,
        "action_type": {"const": "set_runtime_state"},
        "state_families": {
            "type": "array",
            "items": {"enum": list(RUNTIME_STATE_FIELDS)},
            "uniqueItems": True,
        },
    }))
    variants.append(_object({
        **common,
        "action_type": {"enum": [
            "capture_screenshot", "sequence", "remove_entity",
            "play_animation", "spawn_crowd", "set_weather",
        ]},
        "metadata_only": {"const": True},
    }))
    return {"oneOf": variants}


def _sample_properties(*, render: bool) -> dict[str, Any]:
    properties = {
        "present": {"const": True},
        "tick": _tick(),
        "position_enu_m": _nullable(_vector3()),
        "velocity_enu_mps": _nullable(_vector3()),
        "state": _nullable(_string()),
        "activity_type": _nullable(_string()),
        "label_class": _nullable(_string()),
        "entity_category": _nullable(_string()),
    }
    if render:
        properties["source"] = _nullable(_string())
    return properties


def _source_snapshot() -> dict[str, Any]:
    base = _sample_properties(render=False)
    with_source = {**base, "source": _nullable(_string())}
    aligned = {
        **with_source,
        "sumo_authority_replaced": {"const": True},
        "snapshot_policy": {"const": _ALIGN_POLICY},
    }
    absent_aligned = {
        "present": {"const": False},
        "position_enu_m": {"type": "null"},
        "velocity_enu_mps": {"type": "null"},
        "state": {"type": "null"},
        "activity_type": {"type": "null"},
        "source": {"const": "sumo_traci"},
        "sumo_authority_replaced": {"const": True},
        "snapshot_policy": {"const": _ALIGN_POLICY},
    }
    old_present_aligned = {
        **absent_aligned,
        "tick": _tick(),
        "label_class": _nullable(_string()),
        "entity_category": _nullable(_string()),
    }
    return {"oneOf": [
        _object({"present": {"const": False}}),
        _object(base),
        _object(with_source),
        _object(aligned),
        _object(absent_aligned),
        _object(old_present_aligned),
    ]}


def _render_snapshot() -> dict[str, Any]:
    return {"oneOf": [
        _object({"present": {"const": False}}),
        _object(_sample_properties(render=True)),
    ]}


def _snapshot_map(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "patternProperties": {"^[0-9]+$": {
            "type": "object",
            "properties": {},
            "patternProperties": {"^.+$": item},
            "additionalProperties": False,
        }},
        "additionalProperties": False,
    }


def _trigger_reconciliation() -> dict[str, Any]:
    common = {
        "event_id": _string(),
        "trigger_id": _string(),
        "source_dispatch_tick": _tick(),
        "render_dispatch_tick": _tick(),
    }
    return {"oneOf": [
        _object({**common, "policy": {"const": _TRIGGER_POLICY}}),
        _object({
            **common,
            "policy": {"const": _CASCADE_POLICY},
            "parent_event_id": _string(),
        }),
    ]}


def _render_reconciliation() -> dict[str, Any]:
    change = _object({
        "action_id": _string(),
        "entity_id": _string(),
        "old_result_tick": _nullable(_tick()),
        "old_terminal_tick": _nullable(_tick()),
        "render_result_tick": _tick(),
        "render_terminal_tick": _tick(),
    })
    return _object({
        "policy": {"const": "render_authoritative_event_realization_after_sumo_uav_replacement_v1"},
        "changes": {"type": "array", "items": change, "minItems": 1},
    })


_BASIS_MEMBERS = (
    "Dataset/episodes trajectories generated after event dispatch",
    "render_ready truth_frames after SUMO/UAV authority reconciliation",
    "render_ready truth_frames after SUMO authority trigger reconciliation",
)

# Dataset/tools/repair_l25_signal_truth.py freezes these three L2-5 provenance
# labels in the row it rewrites (retained_basis :263-266, the manual_flow_restore
# label :294-297, the signal_infrastructure_repair label :302-305).  They name
# checked-in L2-5 artifacts only, so they join the enum at the L2 level alone;
# every other level keeps the three producer-generic members above.
_BASIS_MEMBERS_L2 = (
    "Retained formal event_realization source snapshots and formal truth_frames "
    "render samples; checked-in L2-5 event_script.json for the event chain",
    "Retained source snapshots with police state corrected by the checked-in "
    "event script; render snapshots from retained formal truth_frames",
    "Checked-in L2-5 event_script.json and scene_setup.json state replay; "
    "render snapshots from retained formal truth_frames",
)


def _basis_members(level: str) -> list[str]:
    if level == "L2":
        return [*_BASIS_MEMBERS, *_BASIS_MEMBERS_L2]
    return list(_BASIS_MEMBERS)


def formal_event_realization_schema(level: str) -> dict[str, Any]:
    """Return a producer-shared whole-row schema for an explicit source level."""
    if level not in _LEVELS:
        raise ValueError(f"unsupported formal event realization level: {level!r}")
    required = [
        "schema_name", "schema_version", "scenario_id", "sequence_no",
        "event_id", "topic", "title", "intent", "dispatch_tick",
        "result_tick", "evidence_tick", "before_tick", "after_tick",
        "capture_grid_step", "target_ids", "terminal_result_required",
        "action_realizations", "source_truth_snapshots_by_tick",
        "render_truth_snapshots_by_tick", "basis",
    ]
    properties = {
        "schema_name": {"const": "event_realization"},
        "schema_version": {"const": "v1"},
        "scenario_id": _string(),
        "sequence_no": {"type": "integer", "minimum": 1},
        "event_id": _string(),
        "topic": _string(),
        "title": _string(),
        "intent": _nullable(_string()),
        "dispatch_tick": _tick(),
        "result_tick": _tick(),
        "evidence_tick": _tick(),
        "before_tick": _tick(),
        "after_tick": _tick(),
        "capture_grid_step": {"const": 5},
        "target_ids": _string_array(),
        "terminal_result_required": {"type": "boolean"},
        "action_realizations": {"type": "array", "items": _action_schema()},
        "source_truth_snapshots_by_tick": _snapshot_map(_source_snapshot()),
        "render_truth_snapshots_by_tick": _snapshot_map(_render_snapshot()),
        "basis": {"enum": _basis_members(level)},
        "render_truth_reconciliation": _render_reconciliation(),
        "render_truth_trigger_reconciliation": _trigger_reconciliation(),
    }
    return deepcopy({
        "$schema": _DRAFT,
        "$id": f"https://aeroworld.local/schema/formal_event_realization/{level}",
        **_object(properties, required=required),
    })


__all__ = ["formal_event_realization_schema"]
