"""Closed JSONL row grammar for L2 ARM ``objective/predicate_truth.jsonl``.

``l2_window.py`` writes the window-cropped ``predicate_truth`` collection from
``l2_arm_semantics.recompute``.  That producer creates one assertion and one
observation per predicate at each formal tick; ``_adapt_assertions`` builds the
truth row, and L2 adds ``parent_episode_id`` and ``arm_key``.  The separate
``predicate_truth_ticks.jsonl`` is the dense, eight-key ``dense`` collection.

Frozen L2 rows carry schema version 3.0.0, without either update-tick field.
The current adapter writes version 3.1.0 and includes both update ticks at the
root and in ``evidence.observations[0].value``.  Both shapes are declared
explicitly.  The 3.1.0 branch has producer evidence but no L2 instance checked
with this module.  Frozen 3.0.0 ground geometry includes partial field sets;
the current producer writes either all four fields or an empty object.

The predicate branches and role keys come from ``scope_for`` in
``Dataset/tools/l2_arm_semantics.py``.  Role classes below match the current
``core_semantic_predicate_templates.json`` registry; ``scope_for.add`` checks
their role-key sets against that registry before emitting rows.  Entity IDs
remain values of fixed role keys rather than schema property names.
"""

from __future__ import annotations

from typing import Any, Mapping


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_PROFILE = r"(?:L2-[1-4]_v[12]|L2-5_v1)"
_SEED = r"(?:00|01|02)"
_ARM_SUFFIX = r"/[^/]+/arm_[a-z0-9_]+"
_TRUTH_VALUES = ("true", "false", "unknown")
_SIMULATED_KINDS = frozenset(("communication", "gnss", "charger", "pad", "signal"))

# Predicate → (context kind, role → ontology class).  Each entry is an
# ``scope_for.add`` call; classes are the corresponding registry argument_roles.
_PREDICATES: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "airspace.aircraft_outside_assigned_altitude_band":
        ("altitude", (("actor", "UnmannedAircraft"),)),
    "airspace.aircraft_airborne":
        ("ground", (("aircraft", "UnmannedAircraft"),)),
    "airspace.aircraft_touchdown":
        ("ground", (("aircraft", "UnmannedAircraft"),)),
    "airspace.aircraft_at_home_pad":
        ("home", (("aircraft", "UnmannedAircraft"), ("home_pad", "LandingPad"))),
    "airspace.aircraft_in_landing_zone":
        ("landing", (("aircraft", "UnmannedAircraft"),
                     ("landing_zone", "LandingArea"))),
    "safety.aircraft_pair_below_separation":
        ("pair", (("first_aircraft", "UnmannedAircraft"),
                  ("second_aircraft", "UnmannedAircraft"))),
    "safety.aircraft_near_building_structure":
        ("structure", (("aircraft", "UnmannedAircraft"),
                       ("structure", "BuildingStructure"))),
    "airspace.airspace_corridor_over_capacity":
        ("corridor", (("corridor", "AirspaceCorridor"),)),
    "communication.link_unavailable":
        ("communication", (("actor", "UnmannedAircraft"),
                           ("station", "CommunicationStation"))),
    "communication.link_degraded":
        ("communication", (("actor", "UnmannedAircraft"),
                           ("station", "CommunicationStation"))),
    "positioning.gnss_quality_degraded":
        ("gnss", (("actor", "UnmannedAircraft"), ("service", "GnssService"))),
    "facility.charging_station_unavailable":
        ("charger", (("station", "ChargingStation"),)),
    "facility.charging_queue_over_capacity":
        ("charger", (("station", "ChargingStation"),)),
    "facility.landing_pad_contention":
        ("pad", (("first_aircraft", "UnmannedAircraft"),
                 ("pad", "LandingPad"), ("second_aircraft", "UnmannedAircraft"))),
    "ground.traffic_signal_malfunction_active":
        ("signal", (("signal", "TrafficSignal"),)),
}


def _object(properties: Mapping[str, dict[str, Any]],
            *, required: tuple[str, ...] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties) if required is None else list(required),
        "additionalProperties": False,
    }


def _string() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _strings(*, min_items: int = 0) -> dict[str, Any]:
    return {"type": "array", "items": _string(), "minItems": min_items}


def _nullable(value: dict[str, Any]) -> dict[str, Any]:
    return {"oneOf": [{"type": "null"}, value]}


def _tick() -> dict[str, Any]:
    # recompute emits assertions only on the governed formal_step_ticks=5 grid.
    return {"type": "integer", "minimum": 0, "maximum": 900, "multipleOf": 5}


def _roles(roles: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    return _object({role: _string() for role, _class in roles})


def _role_classes(roles: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    return _object({role: {"const": f"world:{ontology_class}"}
                    for role, ontology_class in roles})


def _ground_geometry(version: str) -> dict[str, Any]:
    fields = {
        "z_agl_m": {"type": "number"},
        "local_ground_reference_z_m": {"type": "number"},
        "xy_distance_to_assigned_landing_zone_m": {"type": "number"},
        "speed_mps": {"type": "number"},
    }
    if version == "3.0.0":
        # Frozen records include one-field, three-field and four-field ground
        # geometry.  Each known field may be absent, and no other key may occur.
        return _object(fields, required=())
    # Current _ground_contexts returns a complete row or ground.get(..., {}).
    return {"oneOf": [_object({}), _object(fields)]}


def _context(kind: str, version: str) -> dict[str, Any]:
    number = {"type": "number"}
    integer = {"type": "integer"}
    if kind == "altitude":
        return {"geometry": _object({
            "position_z_m": _nullable(number),
            "assigned_altitude_m": number,
        })}
    if kind in ("ground", "landing"):
        return {"geometry": _ground_geometry(version)}
    if kind == "home":
        return {"geometry": _object({
            "xy_distance_to_home_pad_m": _nullable(number),
        })}
    if kind == "pair":
        return {"derived": _object({
            "nearest_aircraft_distance_m": _nullable(number),
        })}
    if kind == "structure":
        return {"derived": _object({
            "nearest_building_distance_m": _nullable(number),
        })}
    if kind == "corridor":
        return {"scene": _object({
            "maximum_corridor_occupancy_count": _nullable(integer),
            "maximum_corridor_capacity": integer,
        })}
    if kind == "communication":
        return {"direct_predicate_value": {"enum": list(_TRUTH_VALUES)}}
    if kind == "gnss":
        return {"domain": _object({"gnss_navigation": _object({
            "quality_level": {"enum": [
                "nominal", "degraded", "spoofing_suspect", "unknown",
            ]},
        })})}
    if kind == "charger":
        return {"domain": _object({"pad_facility": _object({
            "availability": _nullable({"enum": ["available", "unavailable"]}),
            "requester_count": integer,
            "capacity": integer,
            "requester_ids": _strings(),
            "fault_active": {"oneOf": [
                {"type": "null"}, {"type": "boolean"}, {"const": "unknown"},
            ]},
            "missing": _strings(),
        })})}
    if kind == "pad":
        return {"domain": _object({"pad_facility": _object({
            "pair_contention": _nullable({"type": "boolean"}),
            "requester_ids": _nullable(_strings()),
            "capacity": integer,
        })})}
    if kind == "signal":
        return {"domain": _object({"signal_queue": _object({
            "controller_health": _nullable({"enum": ["all_red_fault", "nominal"]}),
        })})}
    raise ValueError(f"unrecognized L2 predicate context kind: {kind}")


def _observation(kind: str, roles: tuple[tuple[str, str], ...],
                 version: str) -> dict[str, Any]:
    names = tuple(sorted(role for role, _class in roles))
    values: dict[str, dict[str, Any]] = {
        "bindings": _roles(roles),
        "binding_ontology_classes": _role_classes(roles),
        "world_truth_value": {"enum": list(_TRUTH_VALUES)},
        **_context(kind, version),
    }
    if version == "3.1.0":
        values["truth_state_update_tick"] = _tick()
        values["evidence_update_tick"] = _tick()
    return _object({
        "path": {"const": "normalized.grounded_roles__" + "__".join(names)},
        "value": _object(values),
    })


def _branch(predicate_id: str, kind: str,
            roles: tuple[tuple[str, str], ...], version: str) -> dict[str, Any]:
    digest = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
    properties: dict[str, dict[str, Any]] = {
        "schema_name": {"const": "predicate_truth"},
        "schema_version": {"const": version},
        "annotation_layer": {"const": "L1"},
        "source_layer": {"const": "L0"},
        "materialization_policy": {"const": "on_demand_epi_predicate"},
        "truth_id": {"type": "string", "pattern": r"^truth:[0-9a-f]{24}$"},
        "episode_id": {"type": "string", "pattern": (
            rf"^{_PROFILE}__seed{_SEED}::{_PROFILE}/seed{_SEED}"
            rf"{_ARM_SUFFIX}$")},
        "parent_episode_id": {"type": "string", "pattern":
                              rf"^{_PROFILE}__seed{_SEED}$"},
        "arm_key": {"type": "string", "pattern":
                    rf"^{_PROFILE}/seed{_SEED}{_ARM_SUFFIX}$"},
        "tick": _tick(),
        "authoritative_tick": {"const": True},
        "predicate_id": {"const": predicate_id},
        "tuple_id": {"type": "string", "pattern": r"^arm_tuple:[0-9a-f]{24}$"},
        "bindings": _roles(roles),
        "value": {"enum": list(_TRUTH_VALUES)},
        "evidence": _object({
            "source_refs": _strings(min_items=1),
            "observations": {
                "type": "array", "minItems": 1, "maxItems": 1,
                "items": _observation(kind, roles, version),
            },
            "missing_requirements": _strings(),
        }),
        "source_class": {"const": (
            "simulated_derived" if kind in _SIMULATED_KINDS
            else "deterministic_derived")},
        "mechanism_id": {"const": None},
        "template_parameters": _object({}),
        "rule_digest": digest,
        "parameter_digest": digest.copy(),
        "input_digest": digest.copy(),
    }
    if version == "3.1.0":
        properties["truth_state_update_tick"] = _tick()
        properties["evidence_update_tick"] = _tick()
    return _object(properties)


def arm_predicate_truth_schema() -> dict[str, Any]:
    """Return a detached, closed L2 ARM predicate-truth row schema.

    The two version branches reflect frozen 3.0.0 rows and the current 3.1.0
    adapter.  Each predicate/version alternative covers the complete record.
    """
    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/arm_predicate_truth_l2.record.schema.json",
        "title": "L2 ARM predicate truth JSONL record",
        "oneOf": [
            _branch(predicate_id, kind, roles, version)
            for version in ("3.0.0", "3.1.0")
            for predicate_id, (kind, roles) in _PREDICATES.items()
        ],
    }
