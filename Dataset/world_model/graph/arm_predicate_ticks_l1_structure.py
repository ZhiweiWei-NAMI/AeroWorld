"""Closed record structure for L1 ARM dense predicate tick rows.

Source family ``arm_predicate_truth_ticks``; profiles L1-1_v1, L1-1_v2,
L1-2_v1, L1-3_v1, L1-3_v2, L1-4_v1, L1-4_v2.  Producer chain: the arm entry
(``scripts/arms/L1-*/.../arm_factual.py``) calls
``Dataset/tools/l1_window.py:execute_config``, which calls
``Dataset/tools/l1_semantics.py:project`` (line 375), keeps the dense rows
``start <= tick <= end`` (line 449) and writes them verbatim to
``objective/predicate_truth_ticks.jsonl`` (line 466).  Each dense row is the
literal built at ``Dataset/tools/l1_semantics.py:129``:

    {"tick": tick, "predicate_id": pid, "bindings": b, "value": value,
     "operands": operands, "source_refs": refs, **extra}

- ``tick``: every tick of the projected prefix ``range(max_tick + 1)``
  (lines 89, 129) — the dense cadence, not the formal 5-tick grid.
- ``predicate_id``: closed to the eight predicates the projector can emit:
  the target-family primaries ``airspace.aircraft_inside_restricted_airspace``
  (bindings lines 73-74), ``airspace.aircraft_outside_assigned_altitude_band``
  (line 75), ``safety.aircraft_pair_below_separation`` (lines 76-78),
  ``airspace.airspace_corridor_over_capacity`` (lines 79-80), and the four
  lifecycle predicates always emitted together for lifecycle targets
  (lines 81-85): ``airspace.aircraft_airborne``,
  ``airspace.aircraft_at_home_pad``, ``airspace.aircraft_touchdown``,
  ``airspace.aircraft_in_landing_zone``.  Anything else raises at line 86.
- ``bindings``: exact roles per predicate (lines 73-85); values are
  whitespace-free identifiers.
- ``value``: ``"true"`` / ``"false"`` (line 126) when the required operands
  are present, ``"out_of_scope"`` (line 127) when a needed entity is absent
  from the tick; ``unknown`` is unreachable at L1 (line 125 raises instead).
- ``operands``: the dot-keyed flattening of the branch context
  (line 128) — ``geometry.*`` (lines 100, 104, 120), ``derived.*`` (line
  109) or ``scene.*`` (line 114).  Leaves are booleans, numbers, integers
  and identifier strings.  When a needed entity is absent, the projector
  skips context construction and writes an empty map instead.
- ``source_refs``: trajectory refs with ``#tick=…&entity=…`` for each
  needed entity (line 96).  Capacity writes a scene ``#entity=…`` ref first,
  followed by one trajectory ref per occupant (line 116).
- extras: ``pair_positions_enu_m`` (two ``[x, y, z]`` number arrays, line
  110) and ``separation_model`` (the compiled model dict of
  ``_aircraft_pair_separation_models`` or JSON ``null``, line 110) for an
  in-scope pair row; both are absent when a pair entity is missing.
  ``occupied_aircraft_ids`` and ``corridor_active`` (line 115) belong to the
  capacity branch.  No other extras exist at L1.

The L1 writer has no branch for ``arm_domain_state_ticks`` rows in this
physical JSONL.  Its only effective family is ``arm_predicate_truth_ticks``.

Registry wiring into ``structure_closure.py`` belongs to the integration
stage; this module exports pure schema constructors, imports no data and
performs no scans at import time.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["arm_predicate_truth_ticks_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# Entity refs are one trajectory row; the capacity row begins with its scene
# source and then appends one trajectory ref per corridor occupant.
_ENTITY_SOURCE_REF_PATTERN = (
    r"^[^#]+\.jsonl#tick=(?:0|[1-9][0-9]*)&entity=[^#]+$"
)
_SCENE_SOURCE_REF_PATTERN = r"^[^#]+\.json#entity=[^#]+$"
# Binding values and entity ids: non-empty whitespace-free identifiers.
_IDENTIFIER_PATTERN = r"^(?!$)[^\s]+$"
# Dotted operand keys of l1_semantics.py:128 over context groups
# geometry / derived / scene (lines 100, 104, 109, 114, 120).
_OPERAND_KEY_PATTERN = r"^(?:geometry|derived|scene)\.[A-Za-z0-9_]+$"

_L1_VALUES = ("true", "false")

_RESTRICTED = "airspace.aircraft_inside_restricted_airspace"
_ALTITUDE = "airspace.aircraft_outside_assigned_altitude_band"
_PAIR = "safety.aircraft_pair_below_separation"
_CAPACITY = "airspace.airspace_corridor_over_capacity"
_AIRBORNE = "airspace.aircraft_airborne"
_HOME = "airspace.aircraft_at_home_pad"
_TOUCHDOWN = "airspace.aircraft_touchdown"
_LANDING = "airspace.aircraft_in_landing_zone"
_L1_PREDICATES = (_RESTRICTED, _ALTITUDE, _PAIR, _CAPACITY,
                  _AIRBORNE, _HOME, _TOUCHDOWN, _LANDING)

_PAIR_DISTANCE_METRICS = (
    "anisotropic_safety_volume_equivalent_distance",
    "euclidean_3d",
)


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


def _pattern_string(pattern: str) -> dict[str, Any]:
    return {"type": "string", "pattern": pattern}


def _integer() -> dict[str, Any]:
    return {"type": "integer", "minimum": 0}


def _number() -> dict[str, Any]:
    return {"type": "number"}


def _boolean() -> dict[str, Any]:
    return {"type": "boolean"}


def _bindings_schema(roles: tuple[str, ...]) -> dict[str, Any]:
    return _object(
        {role: _pattern_string(_IDENTIFIER_PATTERN) for role in roles},
        required=roles,
    )


def _source_refs(predicate_id: str) -> dict[str, Any]:
    entity_ref = _pattern_string(_ENTITY_SOURCE_REF_PATTERN)
    if predicate_id == _CAPACITY:
        return {
            "type": "array",
            "prefixItems": [_pattern_string(_SCENE_SOURCE_REF_PATTERN)],
            "items": entity_ref,
            "minItems": 1,
        }
    count = 2 if predicate_id == _PAIR else 1
    return {
        "type": "array",
        "items": entity_ref,
        "minItems": count,
        "maxItems": count,
    }


def _operands_schema(
    leaves: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Dot-keyed operand map with the branch's required leaves pinned.

    ``operands`` (line 128) flattens exactly one context group per
    predicate, so the branch's keys are a subset of the anchored
    ``geometry.|derived.|scene.`` key space.  The listed leaves are always
    present after the needed entities are resolved; the out-of-scope branch
    writes an empty object instead.
    """

    return {
        "type": "object",
        "propertyNames": {"pattern": _OPERAND_KEY_PATTERN},
        "additionalProperties": False,
        "required": list(leaves),
        "properties": {name: dict(schema) for name, schema in leaves.items()},
    }


def _restricted_operands() -> dict[str, Any]:
    """``geometry`` of l1_semantics.py:100."""

    return _operands_schema({
        "geometry.restricted_region_active": _boolean(),
        "geometry.minimum_restricted_boundary_distance_m": _number(),
    })


def _altitude_operands() -> dict[str, Any]:
    """``geometry`` of l1_semantics.py:104."""

    return _operands_schema({
        "geometry.position_z_m": _number(),
        "geometry.assigned_altitude_m": _number(),
    })


def _pair_operands() -> dict[str, Any]:
    """``derived`` of l1_semantics.py:109."""

    return _operands_schema({
        "derived.nearest_aircraft_distance_m": _number(),
    })


def _capacity_operands() -> dict[str, Any]:
    """``scene`` of l1_semantics.py:114."""

    return _operands_schema({
        "scene.maximum_corridor_occupancy_count": _integer(),
        "scene.maximum_corridor_capacity": _integer(),
    })


def _lifecycle_operands() -> dict[str, Any]:
    """``geometry`` of l1_semantics.py:120 (airborne/home/touchdown/landing)."""

    return _operands_schema({
        "geometry.z_agl_m": _number(),
        "geometry.local_ground_reference_z_m": _number(),
        "geometry.speed_mps": _number(),
        "geometry.xy_distance_to_home_pad_m": _number(),
        "geometry.home_pad_id": _pattern_string(_IDENTIFIER_PATTERN),
        "geometry.xy_distance_to_assigned_landing_zone_m": _number(),
        "geometry.assigned_landing_zone_id": _pattern_string(_IDENTIFIER_PATTERN),
    })


def _separation_model() -> dict[str, Any]:
    """Compiled model of _aircraft_pair_separation_models (2063-2117) or null.

    With a model: ``metric`` is the fixed equivalent-distance name, both
    limits are positive finite numbers, and ``separation_margin_m`` is the
    governed margin (lines 2105-2111).  Without one the projector writes
    JSON ``null`` (l1_semantics.py:110).
    """

    return {
        "oneOf": [
            _object({
                "metric": {"const": _PAIR_DISTANCE_METRICS[0]},
                "horizontal_limit_m": _number(),
                "vertical_limit_m": _number(),
                "separation_margin_m": _number(),
                "trigger_id": {"type": "string"},
            }),
            {"type": "null"},
        ],
    }


def _pair_positions() -> dict[str, Any]:
    """``[list(a), list(c)]`` (l1_semantics.py:110) — two [x, y, z] arrays."""

    return {
        "type": "array",
        "items": {
            "type": "array",
            "items": _number(),
            "minItems": 3,
            "maxItems": 3,
        },
        "minItems": 2,
        "maxItems": 2,
    }


def _extras_for(predicate_id: str) -> dict[str, Mapping[str, Any]]:
    """``**extra`` keys of the row literal (l1_semantics.py:110, 115)."""

    if predicate_id == _PAIR:
        return {
            "pair_positions_enu_m": _pair_positions(),
            "separation_model": _separation_model(),
        }
    if predicate_id == _CAPACITY:
        return {
            "occupied_aircraft_ids": {
                "type": "array",
                "items": _pattern_string(_IDENTIFIER_PATTERN),
                "minItems": 0,
            },
            "corridor_active": _boolean(),
        }
    return {}


def _dense_row(predicate_id: str, *, out_of_scope: bool = False) -> dict[str, Any]:
    if predicate_id == _RESTRICTED:
        bindings = ("actor", "restricted_region")
        operands = _restricted_operands()
    elif predicate_id == _ALTITUDE:
        bindings = ("actor",)
        operands = _altitude_operands()
    elif predicate_id == _PAIR:
        bindings = ("first_aircraft", "second_aircraft")
        operands = _pair_operands()
    elif predicate_id == _CAPACITY:
        bindings = ("corridor",)
        operands = _capacity_operands()
    elif predicate_id == _HOME:
        bindings = ("aircraft", "home_pad")
        operands = _lifecycle_operands()
    elif predicate_id == _LANDING:
        bindings = ("aircraft", "landing_zone")
        operands = _lifecycle_operands()
    else:  # _AIRBORNE, _TOUCHDOWN
        bindings = ("aircraft",)
        operands = _lifecycle_operands()
    if out_of_scope and predicate_id == _CAPACITY:
        raise ValueError("capacity rows have no needed entity and cannot be out of scope")
    return _object({
        "tick": _integer(),
        "predicate_id": {"const": predicate_id},
        "bindings": _bindings_schema(bindings),
        "value": ({"const": "out_of_scope"} if out_of_scope else
                  {"type": "string", "enum": list(_L1_VALUES)}),
        "operands": _object({}) if out_of_scope else operands,
        "source_refs": _source_refs(predicate_id),
        **({} if out_of_scope else _extras_for(predicate_id)),
    })


def arm_predicate_truth_ticks_schema(layer: str = "L1") -> dict[str, Any]:
    """Return the closed schema for one L1 dense predicate tick row.

    The eight predicate branches are exactly the projector's emit set; each
    pins ``predicate_id`` with ``const`` and closes its bindings, operands and
    extras.  Seven predicates distinguish complete geometry from an absent
    needed entity: the latter writes ``out_of_scope`` with empty operands and
    no extras.  Capacity has no needed entity, so it has only the complete
    branch.  These rows carry no integrity-keyed field.
    """

    if layer != "L1":
        raise ValueError(
            f"arm_predicate_truth_ticks declares layer L1 only, got {layer!r}"
        )
    return {
        "$schema": _DRAFT,
        "$id": (
            "https://aeroworld.local/schema/"
            "arm_predicate_truth_ticks_l1.record.schema.json"
        ),
        "title": "L1 ARM dense predicate tick record",
        "oneOf": [
            _dense_row(predicate_id) if predicate_id == _CAPACITY else
            {"oneOf": [
                _dense_row(predicate_id),
                _dense_row(predicate_id, out_of_scope=True),
            ]}
            for predicate_id in _L1_PREDICATES
        ],
    }
