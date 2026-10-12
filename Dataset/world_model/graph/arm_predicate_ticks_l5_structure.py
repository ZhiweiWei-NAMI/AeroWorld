"""Closed record structure for L5 ARM dense predicate tick rows.

Source family ``arm_predicate_truth_ticks``; L5-1_v1/v2/v3, L5-2_v1/v2,
and L5-3_v1/v2 have current ARM outputs. ``Dataset/tools/l5_arm_pipeline.py``
(``execute_config``) writes ``semantic["predicate_truth_ticks"]`` verbatim to
``objective/predicate_truth_ticks.jsonl``; the rows are built by
``Dataset/tools/l5_arm_semantics.py:calculate`` (``add``, lines 127-139) as
one fixed nine-key dict literal per configured predicate per tick:

    {"tick": tick, "tuple_id": tuple_id, "predicate_id": pid,
     "bindings": bindings, "value": truth, "context": context,
     "missing_source_record": missing, "source_refs": refs,
     "authority_ref": "Dataset/semantic_rules/predicates/"
                      "core_semantic_predicate_templates.json#" + pid}

- ``tick``: the L5 window grid ``range(start_tick, end_tick + 1)``;
  configuration bounds the window to ticks 0..900, and ``calculate`` raises
  ``weather grid incomplete`` unless the weather rows cover that range.
- ``tuple_id``: ``"l5_tuple:" + object_hash([arm_key, pid, bindings])
  .split(":")[1][:24]`` (line 134) — literal prefix plus 24 lowercase hex
  characters.
- ``predicate_id``: closed to the predicates the writer can emit:
  ``emergency.mission_aborted`` (only when configured, line 151), the six
  ``environment.*`` entries of ``ENV_PREDICATES`` (lines 24-27, 156-158),
  the five ``airspace.*`` entries of ``MAIN_PREDICATES`` (lines 28-31,
  159-169) and ``safety.aircraft_near_pedestrian`` (``NEAR``, line 32,
  170-177).  ``add`` raises on any binding mismatch against the registry
  template (line 129), so the branch set below is exactly the producer set.
- ``bindings``: exact registry argument roles per predicate
  (``template["argument_roles"]``), written as fixed object keys by
  ``add``; values are non-empty whitespace-free identifiers.
- ``value``: ``true`` / ``false`` / ``unknown`` from the template
  evaluation path (lines 130-133); ``out_of_scope`` is not reachable at L5.
- ``context``: exactly one of the writer's context payloads per branch —
  ``{"control": {"mission_abort_active": bool}}`` (line 155, the runtime
  ``mission_state`` boolean), ``{"weather": <engine weather row>}`` (line
  158, keys per the accepted weather contract
  ``Dataset/tools/weather_fields.py``: ``tick`` + ``condition`` and the six
  numeric inputs read unconditionally by ``calculate`` while building
  mechanisms, plus optional override keys), and the
  geometry block of ``local_geometry`` (lines 79-98) for the main
  predicates; the unobserved-primary-actor variant writes an empty
  ``geometry`` object plus the ``raw/trajectories.jsonl#
  primary_actor_not_observed`` missing-source entry.  Near-miss rows carry
  ``{"derived": {"nearest_pedestrian_distance_m": float | None}}``
  (lines 170-177); ``None`` makes the row ``unknown`` through the
  state-contract required-field path (line 131).
- ``missing_source_record`` lists missing fields from the predicate template's
  required state contract in template order: empty when the required context
  values are present; every required geometry field when geometry is empty;
  the distance field when the near-pedestrian distance is ``None``.
  ``source_refs`` carries the fixed per-predicate references assembled by
  the writer, with decimal tick fragments bounded to 0..900.
- ``authority_ref``: the fixed registry fragment (line 138).

Other-layer dense-row shapes are deliberately not accepted: the
``arm_window_semantics.py`` branch-window writer rows (``schema_name`` /
``schema_version`` / ``annotation_layer`` / ``evidence`` / ``truth_id`` /
``tuple_id=world_predicate_tuple:...``) target ``predicate_truth.jsonl``,
not this file, and the L3 six-key dense rows are a different producer.  The
nine keys above carry no integrity-keyed field, so the publication
projection (``structure_schema._PUBLICATION_EXCLUDED_PATHS``) excludes
nothing from these records and raw validation needs no carve-out.

This module exports the predicate-row schema constructor and performs no
source scan at import time.
"""

from __future__ import annotations

from typing import Any, Mapping

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates,
)

__all__ = ["arm_predicate_truth_ticks_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# Dense-row identity (Dataset/tools/l5_arm_semantics.py:134).
_TUPLE_ID_PATTERN = r"^l5_tuple:[0-9a-f]{24}$"
_AUTHORITY_REF_PREFIX = (
    "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json#"
)
_REF_TICK = r"(?:0|[1-9][0-9]?|[1-8][0-9]{2}|900)"
_TRAJECTORY_REF_PATTERN = (
    rf"^raw/trajectories\.jsonl#tick={_REF_TICK}&entity=[\s\S]+$"
)
_WEATHER_REF_PATTERN = rf"^raw/weather\.jsonl#tick={_REF_TICK}$"
_GEOMETRY_REF_PATTERN = rf"^raw/geometry\.jsonl#tick={_REF_TICK}$"
# Binding values and authored entity identifiers are nonempty strings without
# whitespace. Missing-source entries come from each predicate's required fields.
_IDENTIFIER_PATTERN = r"^(?!$)[^\s]+$"
_ONTOLOGY_CLASS_PATTERN = r"^world:[^\s]+$"

_L5_VALUES = ("true", "false", "unknown")

# Producer predicate sets, mirroring Dataset/tools/l5_arm_semantics.py
# (ENV_PREDICATES / MAIN_PREDICATES / NEAR / the mission-abort branch).
_MISSION_ABORTED = "emergency.mission_aborted"
_ENV_PREDICATES = (
    "environment.rain_active",
    "environment.fog_active",
    "environment.visibility_below_threshold",
    "environment.wind_speed_above_threshold",
    "environment.low_illumination_active",
    "environment.temperature_outside_operating_range",
)
_MAIN_PREDICATES = (
    "airspace.aircraft_outside_assigned_altitude_band",
    "airspace.aircraft_airborne",
    "airspace.aircraft_at_home_pad",
    "airspace.aircraft_touchdown",
    "airspace.aircraft_in_landing_zone",
)
_NEAR = "safety.aircraft_near_pedestrian"
_L5_PREDICATES = (_MISSION_ABORTED,) + _ENV_PREDICATES + _MAIN_PREDICATES + (_NEAR,)


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
    return {"type": "integer", "minimum": 0, "maximum": 900}


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
    trajectory = _pattern_string(_TRAJECTORY_REF_PATTERN)
    if predicate_id == _MISSION_ABORTED:
        refs = [trajectory]
    elif predicate_id in _ENV_PREDICATES:
        refs = [_pattern_string(_WEATHER_REF_PATTERN)]
    elif predicate_id == _NEAR:
        refs = [trajectory, _pattern_string(_TRAJECTORY_REF_PATTERN)]
    else:
        refs = [
            trajectory,
            _pattern_string(_GEOMETRY_REF_PATTERN),
            {"const": "raw/binding_catalog.json"},
        ]
    return {
        "type": "array",
        "prefixItems": refs,
        "items": False,
        "minItems": len(refs),
        "maxItems": len(refs),
    }


def _missing_source_record(fields: tuple[str, ...]) -> dict[str, Any]:
    schema = {
        "type": "array",
        "items": False,
        "minItems": len(fields),
        "maxItems": len(fields),
    }
    if fields:
        schema["prefixItems"] = [{"const": field} for field in fields]
    return schema


def _geometry_block(*, observed: bool) -> dict[str, Any]:
    """``geometry`` payload of ``l5_arm_semantics.local_geometry`` rows.

    All fields are produced together when the primary actor is observed
    (lines 88-95); the primary-actor-unobserved variant writes an empty
    object in ``local_geometry`` (lines 82-83); the dense row's missing list
    is then derived from the predicate template's required geometry fields.
    """

    if not observed:
        return _object({})
    return _object({
        "position_z_m": _number(),
        "assigned_altitude_m": _number(),
        "z_agl_m": _number(),
        "local_ground_reference_z_m": _number(),
        "home_pad_id": _pattern_string(_IDENTIFIER_PATTERN),
        "home_pad_ontology_class_id": _pattern_string(_ONTOLOGY_CLASS_PATTERN),
        "assigned_landing_zone_id": _pattern_string(_IDENTIFIER_PATTERN),
        "assigned_landing_zone_ontology_class_id": _pattern_string(
            _ONTOLOGY_CLASS_PATTERN
        ),
        "speed_mps": _number(),
        "xy_distance_to_home_pad_m": _number(),
        "xy_distance_to_assigned_landing_zone_m": _number(),
    })


def _control_context() -> dict[str, Any]:
    """``{'control': …}`` for the mission-abort predicate branch (line 155)."""

    return _object({
        "control": _object({
            "mission_abort_active": _boolean(),
        }),
    })


def _weather_context(required_field: str) -> dict[str, Any]:
    """``{'weather': …}`` from the L0 weather payload (line 158).

    Keys follow the accepted weather contract in
    ``Dataset/tools/weather_fields.py``.  The six predicate inputs are
    present in every completed L5 output because ``calculate`` also reads
    all six unconditionally while building mechanisms (line 184).
    """

    numeric = (
        "rain", "wetness", "fog", "fog_density", "dust", "wind_speed",
        "wind_direction_deg", "visibility", "visibility_m", "temperature_c",
        "illumination_lux", "hazard_concentration_ppm", "hazard_radius_m",
    )
    if not required_field.startswith("weather."):
        raise ValueError(f"not a weather field: {required_field}")
    name = required_field.removeprefix("weather.")
    if name not in numeric:
        raise ValueError(f"unregistered weather field: {required_field}")
    properties = {
        "tick": _integer(),
        "condition": {"type": "string", "minLength": 1},
        "temperature_source": {"type": "string"},
        "hazard_source_active": _boolean(),
        **{key: _number() for key in numeric},
    }
    required = (
        "tick", "condition", "rain", "fog_density", "visibility_m",
        "wind_speed", "temperature_c", "illumination_lux",
    )
    return _object({"weather": _object(properties, required=required)})


def _derived_context(*, observed: bool) -> dict[str, Any]:
    """``{'derived': …}`` for the near-pedestrian branch (lines 174-177).

    ``nearest_pedestrian_distance_m`` is ``None`` when either party is
    unobserved at the tick; the row then turns ``unknown`` through the
    state-contract required-field path (line 131).
    """

    return _object({
        "derived": _object({
            "nearest_pedestrian_distance_m": (
                _number() if observed else {"type": "null"}
            ),
        }),
    })


def _context_for(
    predicate_id: str, required_fields: tuple[str, ...], *, missing: bool
) -> dict[str, Any]:
    if predicate_id == _MISSION_ABORTED:
        return _control_context()
    if predicate_id in _ENV_PREDICATES:
        return _weather_context(required_fields[0])
    if predicate_id == _NEAR:
        return _derived_context(observed=not missing)
    if predicate_id in _MAIN_PREDICATES:
        return _object({"geometry": _geometry_block(observed=not missing)})
    raise ValueError(f"unregistered L5 predicate: {predicate_id}")


def _dense_row(predicate_id: str) -> dict[str, Any]:
    template = next(
        item for item in get_core_predicate_templates()
        if item["id"] == predicate_id
    )
    roles = tuple(str(role["key"]) for role in template["argument_roles"])
    required_fields = tuple(
        item["field"] for item in template["state_contract"]["required_fields"]
    )
    if (not required_fields or
            any(not isinstance(field, str) or not field for field in required_fields) or
            len(required_fields) != len(set(required_fields))):
        raise ValueError(f"invalid required fields for {predicate_id}")

    def variant(*, missing: bool) -> dict[str, Any]:
        return _object({
            "tick": _integer(),
            "tuple_id": _pattern_string(_TUPLE_ID_PATTERN),
            "predicate_id": {"const": predicate_id},
            "bindings": _bindings_schema(roles),
            "value": ({"const": "unknown"} if missing else
                      {"type": "string", "enum": list(_L5_VALUES)}),
            "context": _context_for(
                predicate_id, required_fields, missing=missing
            ),
            "missing_source_record": _missing_source_record(
                required_fields if missing else ()
            ),
            "source_refs": _source_refs(predicate_id),
            "authority_ref": {"const": _AUTHORITY_REF_PREFIX + predicate_id},
        })

    if predicate_id == _MISSION_ABORTED or predicate_id in _ENV_PREDICATES:
        return variant(missing=False)
    return {"oneOf": [variant(missing=False), variant(missing=True)]}


def arm_predicate_truth_ticks_schema(layer: str = "L5") -> dict[str, Any]:
    """Return the closed schema for one L5 dense predicate tick row.

    Argument roles are read from the governed registry at call time (the
    same authority ``add`` checks bindings against); no state is cached or
    shared with other modules.
    """

    if layer != "L5":
        raise ValueError(
            f"arm_predicate_truth_ticks declares layer L5 only, got {layer!r}"
        )
    return {
        "$schema": _DRAFT,
        "$id": (
            "https://aeroworld.local/schema/"
            "arm_predicate_truth_ticks_l5.record.schema.json"
        ),
        "title": "L5 ARM dense predicate tick record",
        "oneOf": [_dense_row(predicate_id) for predicate_id in _L5_PREDICATES],
    }
