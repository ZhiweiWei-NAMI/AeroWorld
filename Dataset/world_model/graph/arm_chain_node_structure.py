"""Closed JSONL row schemas for ARM chain/nodes.

L3, L6 and X write this file.  The X frozen source also contains measured
geometry and observed dispatch nodes that its current writer no longer emits;
their existing rows remain part of the source structure.
"""

from __future__ import annotations

from copy import deepcopy

from Dataset.tools.runtime_state_contract import RUNTIME_STATE_BOOLEAN_FIELDS
from Dataset.world_model.graph.arm_actions_x_structure import arm_actions_schema


__all__ = ["arm_chain_nodes_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_KINDS = {
    "L3": frozenset({"predicate_transition", "objective_event", "event_recovery_confirmed"}),
    "L6": frozenset({"predicate_transition", "objective_event", "event_recovery_confirmed",
                     "inherited_predicate_transition"}),
    "X": frozenset({"script_emission", "measured_business_predicate_onset",
                    "observed_action_dispatch", "measured_geometry_support"}),
}


def _row(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def _str() -> dict:
    return {"type": "string", "minLength": 1}


def _tick() -> dict:
    return {"type": "integer", "minimum": 0}


def _refs() -> dict:
    return {"type": "array", "minItems": 1, "items": _str()}


def _bindings(*roles: str) -> dict:
    return _row({role: _str() for role in roles})


def _formal_node(level: str, kind: str) -> dict:
    properties = {"node_id": _str(), "kind": {"const": kind}, "tick": _tick()}
    if kind == "objective_event" and level == "L3":
        properties["family"] = _str()
    if kind == "inherited_predicate_transition":
        properties["evidence_ref"] = {"type": "string", "pattern": "^design/.+\\.json\\.gz#.+$"}
    else:
        file = ("predicate_transitions.jsonl" if kind == "predicate_transition" else
                "event_occurrences.jsonl" if kind == "objective_event" else
                "event_outcomes.jsonl")
        properties["evidence_ref"] = {"type": "string", "pattern": f"^objective/{file}#.+$"}
    return _row(properties)


def _x_script() -> dict:
    return _row({"node_id": {"type": "string", "pattern": "^script:.+$"},
                 "event_id": _str(), "tick": _tick(),
                 "kind": {"const": "script_emission"}, "ontology_event": {"const": False}})


def _x_business() -> dict:
    runtime = sorted(f"x.runtime.{family}.{field}"
                     for family in ("communication_state", "navigation_state", "control_state",
                                    "mission_state", "incident_state", "facility_state")
                     for field in RUNTIME_STATE_BOOLEAN_FIELDS[family])
    predicate_bindings = (
        ({"enum": ["environment.rain_active", "environment.fog_active",
                   "emergency.temporary_no_fly_zone_active"]}, _bindings("region")),
        ({"enum": [*runtime, "x.motion.moving", "x.motion.descending",
                   "x.pedestrian.fallen", "x.facility.multiple_requests"]}, _bindings("entity")),
        ({"const": "safety.aircraft_pair_below_separation"},
         _bindings("first_aircraft", "second_aircraft")),
        ({"const": "airspace.aircraft_inside_restricted_airspace"},
         _bindings("actor", "restricted_region")),
        ({"type": "string", "pattern": "^x\\.geometry\\.[A-Za-z0-9_]+$"},
         _bindings("first", "second")),
    )
    rows = []
    for predicate, bindings in predicate_bindings:
        rows.append(_row({
            "node_id": {"type": "string", "pattern": "^xb:[0-9a-f]{64}$"},
            "event_id": {"type": "string", "pattern": "^xb:[0-9a-f]{64}$"},
            "tick": _tick(), "predicate_id": predicate, "bindings": bindings,
            "kind": {"const": "measured_business_predicate_onset"},
            "ontology_event": {"const": False}, "source_refs": _refs(),
        }))
    return {"oneOf": rows}


def _x_dispatch() -> dict:
    # The frozen node copies an actual action and its handler result.  Reuse
    # the X raw-action branches, but not raw/actions.jsonl's motion_schedule
    # sibling: it is absent from the saved chain node.
    rows = []
    for action_row in arm_actions_schema()["oneOf"]:
        action_properties = action_row["properties"]
        rows.append(_row({
            "node_id": {"type": "string", "pattern": "^action:.+$"},
            "kind": {"const": "observed_action_dispatch"}, "tick": _tick(),
            "action": deepcopy(action_properties["action"]),
            "result": deepcopy(action_properties["result"]),
            "source_ref": {"type": "string", "pattern": "^raw/actions\\.jsonl#.+$"},
        }))
    return {"oneOf": rows}


def _x_geometry() -> dict:
    common_trigger = {
        "trigger_id": _str(), "type": {"const": "entity_proximity"},
        "entity_a": _str(), "entity_b": _str(),
        "distance_m": {"type": "number", "minimum": 0},
        "operator": {"const": "lte"}, "min_true_ticks": {"type": "integer", "minimum": 1},
    }
    trigger = {"oneOf": [
        _row({**common_trigger, "metric": {"const": "3d"}}),
        _row({**common_trigger, "metric": {"const": "xy_plus_z"},
              "horizontal_distance_m": {"type": "number", "minimum": 0},
              "vertical_distance_m": {"type": "number", "minimum": 0}}),
    ]}
    measurements = _row({
        "distance_3d_m": {"type": "number", "minimum": 0},
        "distance_xy_m": {"type": "number", "minimum": 0},
        "distance_z_m": {"type": "number", "minimum": 0},
        "trigger": trigger,
    })
    return _row({
        "node_id": {"type": "string", "pattern": "^geometry_evidence:.+$"},
        "kind": {"const": "measured_geometry_support"}, "tick": _tick(),
        "predicate_id": {"type": "string", "pattern": "^x\\.geometry\\.[A-Za-z0-9_]+$"},
        "bindings": _bindings("first", "second"), "value": {"const": "true"},
        "measurements": measurements, "authority": {"const": "X business observation"},
        "source_refs": _refs(),
    })


_X_BUILDERS = {
    "script_emission": _x_script,
    "measured_business_predicate_onset": _x_business,
    "observed_action_dispatch": _x_dispatch,
    "measured_geometry_support": _x_geometry,
}


def arm_chain_nodes_schema(level: str, effective_family: str) -> dict:
    """Return one closed chain/nodes row schema for a producer and row type."""
    if level not in _KINDS:
        raise ValueError(f"{level!r} has no chain/nodes producer")
    kind = effective_family.removeprefix("arm_chain_nodes:")
    if effective_family != "arm_chain_nodes:" + kind or kind not in _KINDS[level]:
        raise ValueError(f"{level} chain/nodes has no row type {effective_family!r}")
    body = _X_BUILDERS[kind]() if level == "X" else _formal_node(level, kind)
    return {"$schema": _DRAFT,
            "$id": f"https://aeroworld.local/schema/arm_chain_nodes_{level.lower()}_"
                   f"{kind}.record.schema.json",
            "title": f"ARM {level} {effective_family} JSONL row", **body}
