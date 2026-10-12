"""Closed frozen L5 ARM ``objective/predicate_truth.jsonl`` row grammar.

``l5_arm_pipeline.execute_config`` writes ``calculate``'s sampled truth rows
verbatim.  ``l5_arm_semantics.calculate`` constructs the L5 observation
contexts and adapts assertions through ``_adapt_assertions``.  Frozen L5 rows
are 3.0.0 with all three root digests, no update ticks, and one local context
(``weather``, ``geometry``, ``control``, or ``derived``) in the evidence value.

The current adapter asks for ``truth_state_update_tick`` and
``evidence_update_tick`` in each observation context, but L5 ``add`` supplies
neither.  Its current 3.1.0 output is blocked at that lookup; this grammar
does not claim that an unproduced 3.1.0 L5 row exists.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates,
)
from Dataset.tools.weather_fields import WEATHER_ALIASES, WEATHER_NUMERIC_FIELDS
from Dataset.world_model.graph.predicate_structure import (
    formal_predicate_truth_schema,
)
from Dataset.world_model.graph.structure_fragments import (
    stripped_lower_phrase_string_rule,
)


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
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
_ABORT = "emergency.mission_aborted"
_NEAR = "safety.aircraft_near_pedestrian"
_PREDICATES = (_ABORT, *_ENV_PREDICATES, *_MAIN_PREDICATES, _NEAR)
_EPISODE_ID_PATTERN = (
    r"^(?:L5-1_v[123]|L5-2_v[12]|L5-3_v[12])__seed(?:00|01|02)"
    r"::(?:L5-1_v[123]|L5-2_v[12]|L5-3_v[12])/seed(?:00|01|02)"
    r"/grp_[a-z0-9_]+/[^/\s]+$"
)


def _object(properties: dict[str, Any], *, required: tuple[str, ...] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties) if required is None else list(required),
    }


def _number() -> dict[str, str]:
    return {"type": "number"}


def _nonempty() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _weather() -> dict[str, Any]:
    # batch_generate.weather_payload starts from one of the seven fixed
    # WEATHER_PROFILES, then applies only weather_fields override keys.
    # Every profile has the same base keys; light smoke and an explicit
    # override can additionally carry dust.
    numeric = WEATHER_NUMERIC_FIELDS | WEATHER_ALIASES.keys()
    fields: dict[str, Any] = {
        "tick": {"type": "integer", "minimum": 0, "maximum": 900},
        "condition": stripped_lower_phrase_string_rule((
            "clear", "rain", "fog", "wind", "dusk", "heat", "light smoke",
        )),
        "hazard_source_active": {"type": "boolean"},
    }
    fields.update({name: _number() for name in sorted(numeric)})
    return _object(fields, required=tuple(name for name in fields if name != "dust"))


def _geometry() -> dict[str, Any]:
    # local_geometry emits either no geometry for an unobserved primary actor
    # or all eleven fields together for an observed one.
    complete = _object({
        "position_z_m": _number(),
        "assigned_altitude_m": _number(),
        "z_agl_m": _number(),
        "local_ground_reference_z_m": _number(),
        "home_pad_id": _nonempty(),
        "home_pad_ontology_class_id": {"const": "world:LandingPad"},
        "assigned_landing_zone_id": _nonempty(),
        "assigned_landing_zone_ontology_class_id": {
            "enum": ["world:LandingArea", "world:LandingPad"],
        },
        "speed_mps": _number(),
        "xy_distance_to_home_pad_m": _number(),
        "xy_distance_to_assigned_landing_zone_m": _number(),
    })
    return {"oneOf": [_object({}), complete]}


def _context(predicate_id: str) -> tuple[str, dict[str, Any]]:
    if predicate_id == _ABORT:
        return "control", _object({"mission_abort_active": {"type": "boolean"}})
    if predicate_id in _ENV_PREDICATES:
        return "weather", _weather()
    if predicate_id == _NEAR:
        return "derived", _object({
            "nearest_pedestrian_distance_m": {
                "oneOf": [_number(), {"type": "null"}],
            },
        })
    return "geometry", _geometry()


_TRAJECTORY_REF = (
    r"^raw/trajectories\.jsonl#tick=[0-9]+&entity=[^\s]+$"
)
_WEATHER_REF = r"^raw/weather\.jsonl#tick=[0-9]+$"
_GEOMETRY_REF = r"^raw/geometry\.jsonl#tick=[0-9]+$"


def _source_refs(predicate_id: str) -> dict[str, Any]:
    if predicate_id == _ABORT:
        members = [{"type": "string", "pattern": _TRAJECTORY_REF}]
    elif predicate_id in _ENV_PREDICATES:
        members = [{"type": "string", "pattern": _WEATHER_REF}]
    elif predicate_id == _NEAR:
        members = [{"type": "string", "pattern": _TRAJECTORY_REF}] * 2
    else:
        members = [
            {"type": "string", "pattern": _TRAJECTORY_REF},
            {"type": "string", "pattern": _GEOMETRY_REF},
            {"const": "raw/binding_catalog.json"},
        ]
    return {
        "type": "array", "prefixItems": members, "items": False,
        "minItems": len(members), "maxItems": len(members),
    }


def _variant(formal_variant: dict[str, Any], template: dict[str, Any],
             predicate_id: str) -> dict[str, Any]:
    branch = deepcopy(formal_variant)
    evidence = branch["properties"]["evidence"]
    evidence["properties"]["source_refs"] = _source_refs(predicate_id)
    fields = [entry["field"] for entry in template["state_contract"]["required_fields"]]
    evidence["properties"]["missing_requirements"]["items"] = {"enum": fields}
    observations = evidence["properties"]["observations"]
    observations["minItems"] = 1
    observations["maxItems"] = 1
    observation = observations["items"]["oneOf"][0]
    value = observation["properties"]["value"]
    roles = branch["properties"]["bindings"]
    classes = value["properties"]["binding_ontology_classes"]
    if predicate_id == _ABORT:
        # This local producer binds its UAV to the registry's broader
        # MobileAgent role and writes the observed specific class.
        classes["properties"]["actor"] = {"const": "world:UnmannedAircraft"}
    if predicate_id == "airspace.aircraft_in_landing_zone":
        classes["properties"]["landing_zone"] = {
            "enum": ["world:LandingArea", "world:LandingPad"],
        }
    context_key, context_schema = _context(predicate_id)
    observation["properties"]["value"] = _object({
        "bindings": deepcopy(roles),
        "binding_ontology_classes": deepcopy(classes),
        "world_truth_value": {"enum": ["true", "false", "unknown"]},
        context_key: context_schema,
    })
    return branch


def arm_predicate_truth_schema() -> dict[str, Any]:
    """Return the frozen 3.0.0 L5 whole-record grammar."""
    formal = formal_predicate_truth_schema()
    registry = {entry["id"]: entry for entry in get_core_predicate_templates()}
    variants = {
        branch["properties"]["predicate_id"]["const"]: branch
        for branch in formal["allOf"][0]["oneOf"]
    }
    if any(predicate_id not in variants or predicate_id not in registry
           for predicate_id in _PREDICATES):
        raise ValueError("L5 predicate producer differs from the formal registry")
    schema = deepcopy(formal)
    schema["$id"] = "https://aeroworld.local/schema/arm_predicate_truth_l5.record.schema.json"
    schema["title"] = "Frozen L5 ARM predicate truth JSONL record"
    properties = schema["properties"]
    required = schema["required"]
    properties["schema_version"] = {"const": "3.0.0"}
    properties["episode_id"] = {"type": "string", "pattern": _EPISODE_ID_PATTERN}
    properties["tuple_id"] = {"type": "string", "pattern": r"^l5_tuple:[0-9a-f]{24}$"}
    properties["truth_id"] = {"type": "string", "pattern": r"^truth:[0-9a-f]{24}$"}
    properties["tick"] = {
        "type": "integer", "minimum": 0, "maximum": 900, "multipleOf": 5,
    }
    properties["value"] = {"enum": ["true", "false", "unknown"]}
    properties["source_class"] = {"const": "deterministic_derived"}
    properties["mechanism_id"] = {"const": None}
    required.append("mechanism_id")
    for field in ("truth_state_update_tick", "evidence_update_tick"):
        properties.pop(field)
        required.remove(field)
    schema["allOf"] = [{"oneOf": [
        _variant(variants[predicate_id], registry[predicate_id], predicate_id)
        for predicate_id in _PREDICATES
    ]}]
    return schema
