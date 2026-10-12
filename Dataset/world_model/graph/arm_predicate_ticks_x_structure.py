"""Closed whole-record grammar for X dense predicate tick rows.

Source family ``arm_predicate_truth_ticks``, layer X only; profiles
X1_rain_to_c2loss_to_forced_landing, X2_gnss_spoof_to_geofence_violation,
X3_pedestrian_fall_to_emergency_response, X4_fog_to_uav_conflict,
X5_comm_failure_to_pad_contention, X6_crowd_evacuation_to_airspace_lockdown.
The effective record type is ``arm_predicate_truth_ticks``.  Although the
reader can distinguish a row without ``predicate_id``, the X producer below
only writes rows that carry it.

Producer authority for this effective type is
``Dataset/tools/x_arm_pipeline.py``: ``run_arm`` (line 254) calls
``calculate(config, run)`` and ``lines(out/'objective/predicate_truth_ticks.jsonl',
dense)`` (line 277, name ``predicate_truth_ticks``) persists the whole ``dense``
list verbatim — no digest, no post-filter.  ``arm_window_semantics.py`` has no
X ticks writer; no other producer writes this file.

``arm_predicate_truth_ticks``: every row is the fixed seven-key literal of
``add`` (``calculate`` line 148-149) — ``tick``, ``predicate_id``,
``bindings``, ``value``, ``measurements``, ``authority``, ``source_refs``.
``value`` is always one of ``true``/``false``/``unknown`` (string literals,
never booleans; ``None`` inputs map to ``unknown``).  ``calculate`` spans
lines 146-237 and contains exactly ten row-emitting call sites; the schema's
root oneOf mirrors them branch by branch:

1. ``environment.rain_active`` / ``environment.fog_active`` (line 169-170,
   ``reg``) — the only rows over the weather row ``w``; ``measurements`` is
   ``{'weather': {key: w[key]}}`` with ``key`` in {rain, fog_density} and the
   weather-key membership test (``if key in w``) makes the inner map exactly
   one key of a number leaf; ``authority`` is the governed registry anchor.
2. ``x.runtime.<family>.<field>`` (line 180, ``add``) — ``bindings`` is
   ``{'entity': …}``; ``measurements`` is ``{family: {field: value}}`` where
   ``family`` ranges over the six-emitter tuple (line 156) and the leaf is
   the authored boolean or ``None`` (unknown).
3. ``x.motion.moving`` / ``x.motion.descending`` (lines 182-183, ``add``) —
   ``measurements`` is ``{'speed_mps': speed}`` (Euclidean norm, >= 0.0) or
   ``{'vz_mps': vel_mps[2]}`` (signed).
4. ``x.pedestrian.fallen`` (lines 184-186, ``add``) — ``measurements`` is
   ``{'activity_type': str, 'posture': str, 'pedestrian_state':
   row.get('pedestrian_state')}`` (nullable object or null).
5. ``x.facility.multiple_requests`` (lines 187-188, ``add``) — the only
   branch whose value can be ``unknown`` besides x.runtime;
   ``measurements`` is ``{'request_count': number|None,
   'requester_ids': [str]|None}`` (copied verbatim from the row's
   ``facility_state``; no non-negativity evidence exists for the count).
6. ``x.geometry.<trigger_id>`` (lines 212-218, ``add``) — emitted for every
   authored ``entity_proximity`` trigger of the source script regardless of
   endpoint presence; ``bindings`` is ``{'first','second'}``; when either
   endpoint is absent from the tick frame the measurement map is empty
   (``measure={}``) and the value is ``unknown``; otherwise it carries
   ``distance_3d_m``/``distance_xy_m``/``distance_z_m`` (>= 0.0) plus the
   verbatim trigger dict.  Its ``xy_plus_z`` metric reads both
   ``horizontal_distance_m`` and ``vertical_distance_m``; every other or
   absent metric reads ``distance_m``.  The conditional schema requires
   the selected thresholds.  No sampled X tick has this geometry branch.
7. ``safety.aircraft_pair_below_separation`` (lines 189-194, ``reg``) —
   ``bindings`` ``{'first_aircraft','second_aircraft'}``; ``measurements``
   is ``{'derived': {'nearest_aircraft_distance_m': …}}`` (equivalent
   separation distance, >= 0.0; line 260 re-checks the formula).
8. ``emergency.temporary_no_fly_zone_active`` (lines 198-204, ``reg``) —
   ``bindings`` ``{'region': region_id}``; ``measurements`` is
   ``{'scene': {'restricted_region_active': bool}}``; ``source_refs`` gets a
   second entry ``<scene path>#entity=<region_id>`` (line 204).
9. ``airspace.aircraft_inside_restricted_airspace`` (lines 205-209, ``reg``)
   — ``bindings`` ``{'actor','restricted_region'}``; ``measurements`` is
   ``{'geometry': {'restricted_region_active': bool,
   'minimum_restricted_boundary_distance_m': >= 0.0}}``; same second
   source_ref (line 209).
10. No other id is emitted: the registry pair emitters for
    ``airspace.airspace_corridor_over_capacity``,
    ``airspace.aircraft_at_home_pad``, ``facility.landing_pad_contention``
    and ``safety.aircraft_near_pedestrian`` are absent from ``calculate``;
    the contract-side ``x_names`` vocabulary (``Dataset/world_model/graph/
    contract.py`` lines 362-378) confirms the same X row space.

Branches 2-6 share the ``X business observation`` authority; 1, 7-9 share
``Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json#<id>``
(``reg``, line 150).  ``source_refs`` entries are ``raw/…#…`` fragments; the
fallback is the weather row for ``environment.*`` and the trajectory row
otherwise (line 149).

Typed dynamic maps (the runtime family/field containers) carry anchored
``propertyNames`` patterns per the current compiler (``structure_schema.py``
``_key_rule``); leaves keep producer types, with ``x-source-value-types``
restricted to the enumerated raw-type names.

``calculate`` appends only ``add``/``reg`` rows, both of which always write
``predicate_id``.  No X domain-state tick record is produced by this writer.
``prepare`` bounds every X window end at 900; ``calculate`` emits dense ticks
within the window, so a tick need not be divisible by five.
"""

from __future__ import annotations

import re
from functools import cache
from typing import Any, Mapping

from Dataset.world_model.graph.structure_schema import (
    ClosedRecordGrammar,
    compile_record_grammar,
    validate_record,
)

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_ID_TICKS = (
    "https://aeroworld.local/schema/arm_predicate_truth_ticks_x.record.schema.json"
)
_REGISTRY_AUTHORITY_PREFIX = (
    "Dataset/semantic_rules/predicates/"
    "core_semantic_predicate_templates.json#"
)
_BUSINESS_AUTHORITY = "X business observation"
_X_PROFILE = r"X[1-6]_[a-z0-9_]+"
_SOURCE_REF_ITEM = {
    "type": "string",
    # Fallback refs are raw/... (line 149); restricted-airspace and no-fly
    # rows append the scene path verbatim (lines 204/209): repo-relative.
    "pattern": r"^[A-Za-z0-9_./-]+#[^\s]*$",
}
_VALUE_ENUM = ["true", "false", "unknown"]
_NONEMPTY_STRING = {"type": "string", "minLength": 1}
# The runtime families the X emitter iterates (x_arm_pipeline.py line 156).
_RUNTIME_FAMILIES = (
    "communication_state",
    "control_state",
    "facility_state",
    "incident_state",
    "mission_state",
    "navigation_state",
)
_TRIGGER_PROXIMITY = {
    "type": "object",
    "additionalProperties": False,
    "required": ["trigger_id", "type", "entity_a", "entity_b"],
    # calculate :217 uses both planar and vertical thresholds only for
    # xy_plus_z; its other branch indexes distance_m.
    "if": {
        "properties": {"metric": {"const": "xy_plus_z"}},
        "required": ["metric"],
    },
    "then": {"required": ["horizontal_distance_m", "vertical_distance_m"]},
    "else": {"required": ["distance_m"]},
    "properties": {
        "trigger_id": _NONEMPTY_STRING,
        "type": {"const": "entity_proximity"},
        "entity_a": _NONEMPTY_STRING,
        "entity_b": _NONEMPTY_STRING,
        "metric": {"enum": ["xy", "xy_plus_z", "3d"]},
        "operator": {"type": "string", "minLength": 1},
        "distance_m": {"type": "number", "minimum": 0.0},
        "horizontal_distance_m": {"type": "number", "minimum": 0.0},
        "vertical_distance_m": {"type": "number", "minimum": 0.0},
        "min_true_ticks": {"type": "integer", "minimum": 0},
        "sustain_ticks": {"type": "integer", "minimum": 0},
    },
}


def _object(
    properties: Mapping[str, Mapping[str, Any]], *,
    required: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    names = tuple(properties) if required is None else required
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(names),
        "properties": {name: dict(schema) for name, schema in properties.items()},
    }


def _entity_ref() -> dict[str, Any]:
    """Entity ids are roster-open strings (no closed id vocabulary exists)."""

    return {**_NONEMPTY_STRING, "x-source-value-types": ["string"]}


def _region_ref() -> dict[str, Any]:
    """Scene region ids are open strings (profile-roster data, not structure)."""

    return {**_NONEMPTY_STRING, "x-source-value-types": ["string"]}


def _distance(minimum: float = 0.0) -> dict[str, Any]:
    """``math.dist``/prism/separation distances: finite, never negative."""

    return {"type": "number", "minimum": minimum, "x-source-value-types": ["int", "float"]}


def _map(pattern: str, leaf: Mapping[str, Any], *, min_keys: int = 1,
         max_keys: int | None = 1) -> dict[str, Any]:
    """Typed dynamic map: anchored ``propertyNames`` + typed extras."""

    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "minProperties": min_keys,
        "propertyNames": {"type": "string", "pattern": pattern},
    }
    if max_keys is not None:
        schema["maxProperties"] = max_keys
    schema["additionalProperties"] = dict(leaf)
    return schema


def _row(predicate_pattern: str, *, bindings: Mapping[str, Any],
         measurements: Mapping[str, Any]) -> dict[str, Any]:
    return _object({
        "tick": {"type": "integer", "minimum": 0, "maximum": 900,
                 "x-source-value-types": ["int"]},
        "predicate_id": {"type": "string", "pattern": predicate_pattern},
        "bindings": _object(bindings),
        "value": {"enum": _VALUE_ENUM},
        "measurements": measurements,
        "authority": {
            "type": "string",
            "pattern": "^("
            + re.escape(_REGISTRY_AUTHORITY_PREFIX) + r"[^\s#]+"
            + "|" + re.escape(_BUSINESS_AUTHORITY)
            + ")$",
        },
        "source_refs": {
            "type": "array",
            "items": dict(_SOURCE_REF_ITEM),
            "minItems": 1,
            "uniqueItems": True,
        },
    })


def _weather_row() -> dict[str, Any]:
    return _row(
        r"^environment\.(rain_active|fog_active)$",
        bindings={"region": {
            "type": "string",
            "pattern": r"^" + _X_PROFILE + r"\.authored_weather_domain$",
        }},
        measurements=_object({
            "weather": _map(
                r"^(rain|fog_density)$",
                _distance(),
                min_keys=1, max_keys=1,
            ),
        }),
    )


def _runtime_row() -> dict[str, Any]:
    families = "|".join(_RUNTIME_FAMILIES)
    leaf = {
        "anyOf": [
            {"type": "boolean", "x-source-value-types": ["bool"]},
            {"type": "null", "x-source-value-types": ["null"]},
        ],
    }
    family_map = _map(
        r"^(" + families + r")$",
        _map(r"^[a-z][a-z0-9_]*$", leaf, min_keys=1, max_keys=1),
        min_keys=1, max_keys=1,
    )
    return _row(
        r"^x\.runtime\.(" + families + r")\.[a-z][a-z0-9_]*$",
        bindings={"entity": _entity_ref()},
        measurements=family_map,
    )


def _motion_rows() -> list[dict[str, Any]]:
    """Separate rows: the writer's measurement keys are disjoint per id."""

    return [
        _row(
            r"^x\.motion\.moving$",
            bindings={"entity": _entity_ref()},
            measurements=_object({"speed_mps": _distance()}),
        ),
        _row(
            r"^x\.motion\.descending$",
            bindings={"entity": _entity_ref()},
            measurements=_object({
                "vz_mps": {"type": "number",
                           "x-source-value-types": ["int", "float"]},
            }),
        ),
    ]


def _pedestrian_row() -> dict[str, Any]:
    return _row(
        r"^x\.pedestrian\.fallen$",
        bindings={"entity": _entity_ref()},
        measurements=_object({
            "activity_type": _NONEMPTY_STRING,
            "posture": _NONEMPTY_STRING,
            "pedestrian_state": {
                "anyOf": [
                    {"type": "null", "x-source-value-types": ["null"]},
                    {
                        "type": "object",
                        "propertyNames": {
                            "type": "string",
                            "pattern": r"^[a-z][a-z0-9_]*$",
                        },
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "boolean",
                                 "x-source-value-types": ["bool"]},
                                {"type": "string", "minLength": 1,
                                 "x-source-value-types": ["string"]},
                            ],
                        },
                    },
                ],
            },
        }),
    )


def _facility_row() -> dict[str, Any]:
    return _row(
        r"^x\.facility\.multiple_requests$",
        bindings={"entity": _entity_ref()},
        measurements=_object({
            "request_count": {
                "anyOf": [
                    {"type": "number", "x-source-value-types": ["int", "float"]},
                    {"type": "null", "x-source-value-types": ["null"]},
                ],
            },
            "requester_ids": {
                "anyOf": [
                    {"type": "array", "items": _NONEMPTY_STRING},
                    {"type": "null", "x-source-value-types": ["null"]},
                ],
            },
        }),
    )


def _geometry_row() -> dict[str, Any]:
    return _row(
        r"^x\.geometry\.[^\s]+$",
        bindings={"first": _entity_ref(), "second": _entity_ref()},
        measurements={
            "anyOf": [
                {"type": "object", "additionalProperties": False,
                 "maxProperties": 0},
                _object({
                    "distance_3d_m": _distance(),
                    "distance_xy_m": _distance(),
                    "distance_z_m": _distance(),
                    "trigger": _TRIGGER_PROXIMITY,
                }),
            ],
        },
    )


def _separation_row() -> dict[str, Any]:
    return _row(
        r"^safety\.aircraft_pair_below_separation$",
        bindings={
            "first_aircraft": _entity_ref(),
            "second_aircraft": _entity_ref(),
        },
        measurements=_object({
            "derived": _object({
                "nearest_aircraft_distance_m": _distance(),
            }),
        }),
    )


def _no_fly_row() -> dict[str, Any]:
    return _row(
        r"^emergency\.temporary_no_fly_zone_active$",
        bindings={"region": _region_ref()},
        measurements=_object({
            "scene": _object({
                "restricted_region_active": {
                    "type": "boolean", "x-source-value-types": ["bool"]},
            }),
        }),
    )


def _restricted_row() -> dict[str, Any]:
    return _row(
        r"^airspace\.aircraft_inside_restricted_airspace$",
        bindings={"actor": _entity_ref(), "restricted_region": _region_ref()},
        measurements=_object({
            "geometry": _object({
                "restricted_region_active": {
                    "type": "boolean", "x-source-value-types": ["bool"]},
                "minimum_restricted_boundary_distance_m": _distance(),
            }),
        }),
    )


def arm_predicate_truth_ticks_schema(layer: str = "X") -> dict[str, Any]:
    """Return the closed schema for one X dense predicate tick row.

    One oneOf branch per ``calculate`` emitter (the ten call sites listed in
    the module docstring); each branch closes the full seven-key row and its
    ``bindings``/``measurements`` sub-objects.
    """

    if layer != "X":
        raise ValueError(
            f"arm_predicate_truth_ticks declares layer X only, got {layer!r}"
        )
    return {
        "$schema": _DRAFT,
        "$id": _SCHEMA_ID_TICKS,
        "title": "X ARM dense predicate tick record",
        "oneOf": [
            _weather_row(),
            _runtime_row(),
            *_motion_rows(),
            _pedestrian_row(),
            _facility_row(),
            _geometry_row(),
            _separation_row(),
            _no_fly_row(),
            _restricted_row(),
        ],
    }


@cache
def arm_predicate_truth_ticks_grammar() -> ClosedRecordGrammar:
    return compile_record_grammar(
        arm_predicate_truth_ticks_schema(),
        schema_ref=(
            "Dataset/world_model/graph/arm_predicate_ticks_x_structure.py:"
            "arm_predicate_truth_ticks_schema"
        ),
        authority="Dataset/tools/x_arm_pipeline.py:calculate",
    )


def validate_predicate_truth_tick(
    raw: Mapping[str, Any], *, source_path: str | None = None
) -> None:
    """Validate one ``arm_predicate_truth_ticks`` row against its grammar."""

    validate_record(
        arm_predicate_truth_ticks_grammar(),
        raw,
        source_path=source_path,
    )


__all__ = [
    "arm_predicate_truth_ticks_schema",
    "arm_predicate_truth_ticks_grammar",
    "validate_predicate_truth_tick",
]
