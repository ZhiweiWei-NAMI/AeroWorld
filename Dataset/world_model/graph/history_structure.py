"""Closed L6 pre-window history root and its original selected-raw projection.

``window_semantics`` in ``Dataset/tools/l6_v2/observations.py`` writes one
gzip JSON root with seven raw arrays and two objective arrays.  The frozen
3.0.0 rows have two exact lineage shapes: L6-1/L6-2 retain digest fields,
while L6-3/L6-4/L6-5 do not.  The current predicate adapter emits 3.1.0
and requires update ticks absent from the frozen history; this schema covers
the frozen history, not an unverified regenerated 3.1.0 archive.

``l6_history_subtree_schema`` and ``validate_history_subtrees`` retain their
selected-domain/control interface for callers that only inspect those arrays.
``arm_semantic_history_schema`` closes the complete root for an explicit L6
profile and is intentionally separate from global dispatch and registration.
"""

from __future__ import annotations

from copy import deepcopy
from functools import cache
import re
from typing import Any, Mapping

from Dataset.semantic_simulation.control_response_state import (
    SUPPORTED_API_VALUE_KEYS,
)
from Dataset.semantic_simulation.domain_state import SOURCE_CLASSES
from Dataset.tools.weather_fields import WEATHER_NUMERIC_FIELDS
from Dataset.world_model.graph.arm_predicate_transition_l6_structure import (
    arm_predicate_transitions_schema,
)
from Dataset.world_model.graph.arm_trajectories_l6_structure import (
    arm_trajectories_schema,
)
from Dataset.world_model.graph.compute_comm_structure import (
    communication_state_record_schema,
)
from Dataset.world_model.graph.predicate_structure import formal_predicate_truth_schema
from Dataset.world_model.graph.structure_schema import (
    compile_record_grammar,
    validate_record,
)


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_UNKNOWN = {"const": "unknown"}
_NONEMPTY_STRING = {"type": "string", "minLength": 1}
_SOURCE_CLASSES = {"enum": sorted(SOURCE_CLASSES)}
_DIGEST = {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}
_OBSERVATION_SAMPLING_SCOPE = {
    "enum": ["formal", "probe_without_advancing_formal_memory"],
}
_L6_PROFILES = frozenset(
    f"L6-{group}_v{version}"
    for group in range(1, 6)
    for version in (1, 2)
)
# Frozen cohort distinction, confirmed with one real archive from each of
# the ten declared profiles.  It is not inferred from a filename at runtime.
_DIGEST_PROFILES = frozenset(
    f"L6-{group}_v{version}"
    for group in (1, 2)
    for version in (1, 2)
)
_FRAME_RUNTIME_FAMILIES = (
    "control_state", "communication_state", "navigation_state",
    "security_state", "mission_state",
)  # Dataset/tools/l6_v2/common.py:RUNTIME_FAMILIES


def _array(items: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": dict(items), "minItems": 0}


def _object(properties: Mapping[str, Mapping[str, Any]], *,
           required: tuple[str, ...] | None = None) -> dict[str, Any]:
    names = tuple(properties) if required is None else required
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(names),
        "properties": {name: dict(schema) for name, schema in properties.items()},
    }


def _string(*, non_empty: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "string"}
    if non_empty:
        result["minLength"] = 1
    return result


def _string_list(*, min_items: int = 0) -> dict[str, Any]:
    return {
        "type": "array",
        "items": _string(),
        "minItems": min_items,
    }


def _nullable_string() -> dict[str, Any]:
    return {"oneOf": [_NONEMPTY_STRING, {"type": "null"}]}


def _unknown_or(*schemas: Mapping[str, Any]) -> dict[str, Any]:
    return {"oneOf": [dict(schema) for schema in schemas] + [_UNKNOWN]}


def _unknown_or_number() -> dict[str, Any]:
    return _unknown_or({"type": "number"})


def _unknown_or_bool() -> dict[str, Any]:
    return _unknown_or({"type": "boolean"})


def _vector3() -> dict[str, Any]:
    return {
        "type": "array",
        "prefixItems": [{"type": "number"}] * 3,
        "items": False,
        "minItems": 3,
        "maxItems": 3,
    }


def _unknown_or_vector3() -> dict[str, Any]:
    return _unknown_or(_vector3())


def _quality() -> dict[str, Any]:
    base = _object({"status": _string(non_empty=True)})
    gnss = _object({
        "status": _string(non_empty=True),
        "numeric_fsm_applied": {"type": "boolean"},
        "rain_visibility_not_used_as_event_driver": {"type": "boolean"},
    })
    return {"oneOf": [base, gnss]}


def _security_values() -> dict[str, Any]:
    return _object({
        "actor_id": _nullable_string(),
        "actor_ontology_class_id": _nullable_string(),
        "ground_control_station_id": _nullable_string(),
        "ground_control_station_ontology_class_id": _nullable_string(),
        "communication_station_id": _nullable_string(),
        "communication_station_ontology_class_id": _nullable_string(),
        "operational_state": _string(),
        "auth_score": _unknown_or_number(),
        "auth_state": _string(),
        "command_integrity_score": _unknown_or_number(),
        "command_integrity": _unknown_or_bool(),
        "command_source_valid": _unknown_or_bool(),
        "jamming_indicator": _unknown_or_bool(),
        "spectrum_interference_ratio": _unknown_or_number(),
        "lockout_active": _unknown_or_bool(),
        "lockout_duration_ticks": _unknown_or_number(),
        "missing_inputs": _string_list(),
    })


def _gnss_values() -> dict[str, Any]:
    return _object({
        "quality_level": _string(),
        "truth_position_enu_m": _unknown_or_vector3(),
        "reported_position_enu_m": _unknown_or_vector3(),
        "position_error_m": _unknown_or_number(),
        "drift_offset_enu_m": _unknown_or_vector3(),
        "operational_state": _string(),
        "multipath_warning": _unknown_or_bool(),
        "visual_relocalization": _unknown_or_bool(),
        "gnss_spoofed": _unknown_or_bool(),
        "geofence_alert": _unknown_or_bool(),
        "mission_recovered": _unknown_or_bool(),
        "fsm_state": _string(),
        "recovery_decay": _unknown_or_number(),
        "missing_inputs": _string_list(),
        "actor_id": _NONEMPTY_STRING,
        "actor_ontology_class_id": _NONEMPTY_STRING,
        "gnss_service_id": _NONEMPTY_STRING,
        "spoofing_signal_id": _NONEMPTY_STRING,
    })


def _domain_security_record(*, with_digests: bool) -> dict[str, Any]:
    props = {
        "schema_name": {"const": "domain_state_observation"},
        "schema_version": _NONEMPTY_STRING,
        "episode_id": _NONEMPTY_STRING,
        "tick": {"type": "integer", "minimum": 0},
        "observation_family": {"const": "security_command"},
        "subject_id": _NONEMPTY_STRING,
        "subject_category": {"enum": ["uav", "ground_station"]},
        "source_class": _SOURCE_CLASSES,
        "rule_id": _NONEMPTY_STRING,
        "rule_version": _NONEMPTY_STRING,
        "model_id": _NONEMPTY_STRING,
        "model_version": _NONEMPTY_STRING,
        "source_refs": _string_list(min_items=1),
        "producer_source_refs": _string_list(min_items=1),
        "values": _security_values(),
        "quality": _quality(),
        "observation_id": _NONEMPTY_STRING,
        "sampling_scope": _OBSERVATION_SAMPLING_SCOPE,
    }
    if with_digests:
        props.update({name: dict(_DIGEST) for name in
                      ("input_digest", "parameter_digest", "seed_digest")})
    return _object(props)


def _domain_gnss_record(*, with_digests: bool) -> dict[str, Any]:
    # This branch mirrors the L6 producer's hand-built GNSS row.  It does not
    # invent the metadata fields used by domain_state._observation_row.
    props = {
        "tick": {"type": "integer", "minimum": 0},
        "subject_id": _NONEMPTY_STRING,
        "observation_family": {"const": "gnss_navigation"},
        "values": _gnss_values(),
        "quality": _quality(),
        "source_class": _SOURCE_CLASSES,
        "model_id": _NONEMPTY_STRING,
        "model_version": _NONEMPTY_STRING,
        "sampling_scope": _OBSERVATION_SAMPLING_SCOPE,
        "source_refs": _string_list(min_items=1),
    }
    if with_digests:
        props["seed_digest"] = dict(_DIGEST)
    return _object(props)


def _supported_api_values() -> dict[str, Any]:
    return _object({
        key: {"const": value}
        for key, value in sorted(SUPPORTED_API_VALUE_KEYS.items())
    })


def _control_values() -> dict[str, Any]:
    bool_unknown = _unknown_or_bool()
    numeric_unknown = _unknown_or_number()
    return _object({
        "supported_api_value_keys": _supported_api_values(),
        "rth_active": bool_unknown,
        "kinematic_rth_active": bool_unknown,
        "alternate_rth_active": bool_unknown,
        "reroute_active": bool_unknown,
        "alternate_reroute_active": bool_unknown,
        "altitude_corrective_maneuver_active": bool_unknown,
        "deconfliction_active": bool_unknown,
        "evasion_active": bool_unknown,
        "pull_up_active": bool_unknown,
        "safe_hold_active": bool_unknown,
        "hold_active": bool_unknown,
        "resequence_active": bool_unknown,
        "divert_active": bool_unknown,
        "diversion_active": bool_unknown,
        "slowdown_active": bool_unknown,
        "mission_abort_active": bool_unknown,
        "station_unavailable": bool_unknown,
        "backup_link_active": bool_unknown,
        "path_deviation": bool_unknown,
        "route_recovered": bool_unknown,
        "sustained_drift": bool_unknown,
        "route_distance_m": numeric_unknown,
        "previous_route_distance_m": numeric_unknown,
        "home_distance_m": numeric_unknown,
        "previous_home_distance_m": numeric_unknown,
        "speed_delta_mps": numeric_unknown,
        "altitude_error_m": numeric_unknown,
        "gcs_compromised": bool_unknown,
        "jamming_active": bool_unknown,
        "unauthorized_command": bool_unknown,
        "command_integrity_violation": bool_unknown,
        "command_lockout": bool_unknown,
    })


def _control_record() -> dict[str, Any]:
    return _object({
        "tick": {"type": "integer", "minimum": 0},
        "entity_id": _NONEMPTY_STRING,
        "values": _control_values(),
        "model_id": {
            "const": "control_response_state.structured_truth_route_comm_kinematic_rth_v5"
        },
        "sampling_scope": _OBSERVATION_SAMPLING_SCOPE,
        "source_refs": _string_list(min_items=1),
    })


def _selected_arrays(*, with_digests: bool) -> dict[str, Any]:
    return {
        "domain_observations": _array({
            "oneOf": [
                _domain_security_record(with_digests=with_digests),
                _domain_gnss_record(with_digests=with_digests),
            ],
        }),
        "control_observations": _array(_control_record()),
    }


def l6_history_subtree_schema() -> dict[str, Any]:
    """Return the original two-array projection for both frozen L6 cohorts.

    The two lineage shapes are exact alternatives per record, not an open
    union of optional digest keys.  This is still only the selected raw
    projection; ``arm_semantic_history_schema`` covers the complete archive.
    """

    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/l6_history.selected_subtrees.schema.json",
        "title": "L6 measured history selected raw subtrees",
        **_object({
            "domain_observations": _array({"oneOf": [
                builder(with_digests=version)
                for version in (False, True)
                for builder in (_domain_security_record, _domain_gnss_record)
            ]}),
            "control_observations": _array(_control_record()),
        }),
    }


@cache
def l6_history_subtree_grammar():
    """Compile the selected-array grammar for structural projection."""

    return compile_record_grammar(
        l6_history_subtree_schema(),
        schema_ref="Dataset/world_model/graph/history_structure.py:l6_history_subtree_schema",
        authority=(
            "Dataset/tools/l6_v2/semantics.py:producer_observations;"
            "Dataset/semantic_simulation/domain_state.py:_observation_row;"
            "Dataset/semantic_simulation/control_response_state.py:_observation_row"
        ),
    )


def selected_history_subtrees(history: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """Extract the two declared arrays without inspecting other raw arrays."""

    if not isinstance(history, Mapping):
        raise TypeError(f"{source}: history root is not an object")
    raw = history.get("raw")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: history raw container is absent or not an object")
    selected: dict[str, Any] = {}
    for name in ("domain_observations", "control_observations"):
        if name not in raw:
            raise ValueError(f"{source}: history raw.{name} is absent")
        selected[name] = raw[name]
    return selected


def validate_history_subtrees(history: Mapping[str, Any], *, source: str) -> None:
    """Validate only the declared domain/control raw arrays in one archive."""

    selected = selected_history_subtrees(history, source=source)
    validate_record(l6_history_subtree_grammar(), selected, source_path=source)


def _embedded(schema: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(schema))
    for name in ("$schema", "$id", "title", "description"):
        result.pop(name, None)
    return result


def _episode_id_rule(profile: str) -> dict[str, Any]:
    return {"type": "string", "pattern": rf"^{re.escape(profile)}__seed[0-9]+$"}


def _arm_key_rule(profile: str) -> dict[str, Any]:
    return {"type": "string", "pattern":
            rf"^{re.escape(profile)}/seed[0-9]+/grp_[^/\s]+/[^/\s]+$"}


def _semantic_episode_rule(profile: str) -> dict[str, Any]:
    return {"type": "string", "pattern":
            rf"^{re.escape(profile)}__seed[0-9]+::"
            rf"{re.escape(profile)}/seed[0-9]+/grp_[^/\s]+/[^/\s]+$"}


def _local_frame_record(profile: str, trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """Project only the keys ``semantics.local_frames`` copies from a row."""

    categories = {
        "semantic.asset_anchor": "ground_station",
        "facility.landing_pad.visible.v1": "facility",
        "facility.radio.base_tower.v1": "communication_station",
        "uav.airsim.flying_pawn.v1": "uav",
        "uav.inspect.quad.v1": "uav",
    }
    branches = trajectory["oneOf"]
    if {branch["properties"]["asset_id"]["const"] for branch in branches} != set(categories):
        raise ValueError("L6 trajectory identity branches differ from history frame scope")
    entities = []
    copied = (
        "entity_id", "entity_kind", "entity_type", "assigned_altitude_m",
        "planned_route_waypoints_enu_m", *_FRAME_RUNTIME_FAMILIES,
    )
    for branch in branches:
        source = branch["properties"]
        props = {name: deepcopy(source[name]) for name in copied if name in source}
        props["entity_category"] = {"const": categories[source["asset_id"]["const"]]}
        props["truth_pose"] = _object({
            "position_enu_m": deepcopy(source["pos_enu"]),
            "velocity_enu_mps": deepcopy(source["vel_mps"]),
        })
        required = tuple(name for name in props if name in branch["required"])
        projected = _object(props, required=(*required, "entity_category", "truth_pose"))
        # The two UAV asset identities disappear from local_frames; their
        # projected branches are identical and must not double-match oneOf.
        if projected not in entities:
            entities.append(projected)
    return _object({
        "episode_id": _episode_id_rule(profile),
        "tick": {"type": "integer", "minimum": 0, "maximum": 900},
        "sim_time_s": {"type": "number", "x-source-value-types": ["float"]},
        "entities": {"type": "array", "items": {"oneOf": entities}, "minItems": 1},
        "scope": {"const": "current_source_script_selected_entities_only"},
    })


def _raw_reference() -> dict[str, Any]:
    """Exact reference families written by ``semantics.evaluate``."""

    return {"oneOf": [
        {"type": "string", "pattern":
         r"^raw/trajectories\.jsonl#tick=[0-9]+&entity=[^\s]+$"},
        {"type": "string", "pattern":
         r"^raw/communication_state\.jsonl#tick=[0-9]+&entity=[^\s]+$"},
        {"type": "string", "pattern":
         r"^raw/communication_ticks\.jsonl#tick=[0-9]+&entity=[^\s]+$"},
        {"type": "string", "pattern":
         r"^raw/domain_observations\.jsonl#tick=[0-9]+&subject=[^\s&]+"
         r"&family=(gnss_navigation|security_command)$"},
        {"type": "string", "pattern":
         r"^raw/control_observations\.jsonl#tick=[0-9]+&entity=[^\s]+$"},
    ]}


def _communication_record(*, with_digests: bool, formal: bool) -> dict[str, Any]:
    """Adapt published comm value branches to the ARM-local row envelope."""

    published = communication_state_record_schema()
    selected = published["oneOf"][:2]
    if len(selected) != 2 or any(
        row["properties"]["station_source"].get("enum") !=
        ["observed_simulator_truth", "simulated_from_facility_anchor"]
        or row["properties"]["station_id"].get("const") == "unknown"
        for row in selected
    ) or {
        row["properties"]["requirement"]["properties"]["status"]["const"]
        for row in selected
    } != {"not_required", "source_required"}:
        raise ValueError("published communication selected-station branches changed")
    branches = []
    frame_ref = {"type": "string", "pattern":
                 r"^raw/local_frames\.jsonl#tick=[0-9]+&entity=[^\s]+$"}
    weather_ref = {"type": "string", "pattern":
                   r"^raw/weather\.jsonl#tick=[0-9]+$"}
    for published_row in selected:
        row = _embedded(published_row)
        props = row["properties"]
        props["producer_source_refs"] = props["source_refs"]
        props["source_refs"] = {
            "type": "array", "prefixItems": [frame_ref, frame_ref, weather_ref],
            "items": False, "minItems": 3, "maxItems": 3,
        }
        props["sampling_scope"] = {
            "const": "formal" if formal else "off_grid_diagnostic_sequence",
        }
        # The frozen history rows precede the published 2.0.0 envelope.
        props["schema_version"] = {"const": "1.0.0"}
        props["station_source"] = {"const": "observed_simulator_truth"}
        row["required"].extend(("producer_source_refs", "sampling_scope"))
        if with_digests:
            for name in ("input_digest", "parameter_digest", "seed_digest"):
                props[name] = dict(_DIGEST)
                row["required"].append(name)
        branches.append(row)
    return {"oneOf": branches}


def _weather_record() -> dict[str, Any]:
    """``EpisodeStateEngine`` weather row, including the two live aliases."""

    numeric = sorted(WEATHER_NUMERIC_FIELDS | {"fog", "visibility"})
    fields = {name: {"type": "number"} for name in numeric}
    for name in ("rain", "fog", "fog_density"):
        fields[name] = {"type": "number", "minimum": 0, "maximum": 1}
    fields.update({
        "tick": {"type": "integer", "minimum": 0, "maximum": 900},
        "condition": {"type": "string", "pattern": r"\S"},
        "hazard_source_active": {"type": "boolean"},
    })
    required = tuple(name for name in fields if name != "dust")
    return _object(fields, required=required)


_L6_PREDICATES = (
    "airspace.aircraft_airborne", "airspace.aircraft_at_home_pad",
    "airspace.aircraft_in_landing_zone",
    "airspace.aircraft_outside_assigned_altitude_band",
    "airspace.aircraft_touchdown", "communication.link_degraded",
    "communication.link_unavailable", "emergency.aircraft_in_return_to_home",
    "emergency.aircraft_in_safe_hold",
    "positioning.aircraft_deviates_from_planned_route",
    "positioning.gnss_quality_degraded", "positioning.gnss_spoofing_active",
    "positioning.navigation_error_exceeds_tolerance",
    "security.communication_jamming_impact_active",
    "security.ground_control_station_compromised",
)
_BASE_PREDICATES = _L6_PREDICATES[:10]
_PROFILE_ADDITIONS = {
    "1": (),
    "2": (),
    "3": _L6_PREDICATES[10:13],
    "4": (_L6_PREDICATES[13],),
    "5": (_L6_PREDICATES[14],),
}  # arm_runtime.ArmSpec.predicate_ids: base + primary + positioning additions


def _profile_predicates(profile: str) -> tuple[str, ...]:
    group = profile.split("-", 1)[1].split("_", 1)[0]
    return (*_BASE_PREDICATES, *_PROFILE_ADDITIONS[group])


def _truth_context(predicate_id: str) -> dict[str, Any]:
    if predicate_id.startswith("communication."):
        return {"direct_predicate_value": {"enum": ["true", "false", "unknown"]}}
    if predicate_id in (
        "positioning.gnss_quality_degraded", "positioning.gnss_spoofing_active",
        "positioning.navigation_error_exceeds_tolerance",
    ):
        return {"domain": _object({"gnss_navigation": _gnss_values()})}
    if predicate_id.startswith("security."):
        return {"domain": _object({"security_command": _security_values()})}
    if predicate_id == "positioning.aircraft_deviates_from_planned_route":
        return {"navigation": _object({"path_deviation": _unknown_or_bool()})}
    if predicate_id.startswith("emergency."):
        return {
            "control": _control_values(),
            "plan": _object({"rth_mode_active": _unknown_or_bool()}),
        }
    if predicate_id.startswith("airspace."):
        geometry = {
            "position_z_m": {"type": "number"},
            "assigned_altitude_m": {"type": "number"},
            "local_ground_reference_z_m": {"type": "number"},
            "z_agl_m": {"type": "number"},
            "speed_mps": {"type": "number"},
            "home_pad_id": _NONEMPTY_STRING,
            "assigned_landing_zone_id": _NONEMPTY_STRING,
            "xy_distance_to_home_pad_m": {"type": "number"},
            "xy_distance_to_assigned_landing_zone_m": {"type": "number"},
        }
        base = tuple(name for name in geometry if not name.startswith("xy_distance_"))
        return {"geometry": _object(geometry, required=base)}
    raise ValueError(f"L6 history predicate has no producer context: {predicate_id}")


def _predicate_truth_record(profile: str, *, with_digests: bool) -> dict[str, Any]:
    """Frozen 3.0.0 formal rows, with L6-local evidence and arm stamps."""

    schema = _embedded(formal_predicate_truth_schema())
    props, required = schema["properties"], schema["required"]
    props["schema_version"] = {"const": "3.0.0"}
    props["truth_id"] = {"type": "string", "pattern": r"^truth:[0-9a-f]{24}$"}
    props["tuple_id"] = {"type": "string", "pattern": r"^arm_tuple:[0-9a-f]{24}$"}
    props["tick"] = {"type": "integer", "minimum": 0, "maximum": 900,
                     "multipleOf": 5}
    props["episode_id"] = _semantic_episode_rule(profile)
    props["parent_episode_id"] = _episode_id_rule(profile)
    props["arm_key"] = _arm_key_rule(profile)
    props["source_class"] = {"const": "simulated_derived"}
    props["mechanism_id"] = {"const": None}
    required.extend(("parent_episode_id", "arm_key", "mechanism_id"))
    for name in ("truth_state_update_tick", "evidence_update_tick"):
        props.pop(name)
        required.remove(name)
    if not with_digests:
        for name in ("rule_digest", "parameter_digest", "input_digest"):
            props.pop(name)
            required.remove(name)
    formal_variants = {
        variant["properties"]["predicate_id"]["const"]: variant
        for variant in schema["allOf"][0]["oneOf"]
    }
    if not set(_L6_PREDICATES) <= formal_variants.keys():
        raise ValueError("L6 history predicate is absent from the formal registry")
    variants = []
    for predicate_id in _profile_predicates(profile):
        variant = deepcopy(formal_variants[predicate_id])
        evidence = variant["properties"]["evidence"]
        eprops = evidence["properties"]
        eprops["source_refs"] = {
            "type": "array", "minItems": 1,
            "prefixItems": [_raw_reference()["oneOf"][0]],
            "items": _raw_reference(),
        }
        observation = eprops["observations"]["items"]["oneOf"][0]
        value_props = observation["properties"]["value"]["properties"]
        base = {
            "bindings": deepcopy(value_props["bindings"]),
            "binding_ontology_classes": deepcopy(value_props["binding_ontology_classes"]),
            "world_truth_value": {"enum": ["true", "false", "unknown", "out_of_scope"]},
        }
        observation["properties"]["value"] = {"oneOf": [
            _object(base), _object({**base, **_truth_context(predicate_id)}),
        ]}
        eprops["observations"] = {
            "type": "array", "items": observation, "minItems": 1, "maxItems": 1,
        }
        variants.append(variant)
    schema["allOf"] = [{"oneOf": variants}]
    return schema


def _predicate_transition_record(profile: str, *, with_digests: bool) -> dict[str, Any]:
    """Reuse the L6 transition rule, keeping the frozen history envelope."""

    schema = _embedded(arm_predicate_transitions_schema())
    props, required = schema["properties"], schema["required"]
    if set(props["predicate_id"]["enum"]) != set(_L6_PREDICATES):
        raise ValueError("L6 transition predicate set differs from history writer")
    allowed = set(_profile_predicates(profile))
    props["predicate_id"] = {"enum": sorted(allowed)}
    schema["allOf"][0]["oneOf"] = [
        branch for branch in schema["allOf"][0]["oneOf"]
        if branch["properties"]["predicate_id"]["const"] in allowed
    ]
    props["episode_id"] = _semantic_episode_rule(profile)
    props["parent_episode_id"] = _episode_id_rule(profile)
    props["arm_key"] = _arm_key_rule(profile)
    props["source_refs"] = {
        "type": "array", "minItems": 1, "uniqueItems": True,
        "items": _raw_reference(),
    }
    if with_digests:
        required.append("rule_digest")
    else:
        del props["rule_digest"]
    return schema


def arm_semantic_history_schema(profile: str) -> dict[str, Any]:
    """Return the complete frozen L6 gzip-JSON root for one exact profile.

    The history file has one record at ordinal zero.  Global source dispatch
    and pending-status promotion are separate integration work.
    """

    if profile not in _L6_PROFILES:
        raise ValueError(f"arm_semantic_history has no reviewed L6 profile: {profile!r}")
    digests = profile in _DIGEST_PROFILES
    trajectory = _embedded(arm_trajectories_schema())
    raw = {
        "local_frames": _array(_local_frame_record(profile, trajectory)),
        "communication_state": _array(
            _communication_record(with_digests=digests, formal=True)
        ),
        "communication_ticks": _array(
            _communication_record(with_digests=digests, formal=False)
        ),
        **_selected_arrays(with_digests=digests),
        "trajectories": _array(trajectory),
        "weather": _array(_weather_record()),
    }
    objective = {
        "predicate_truth": _array(_predicate_truth_record(profile, with_digests=digests)),
        "predicate_transitions": _array(
            _predicate_transition_record(profile, with_digests=digests)
        ),
    }
    return {
        "$schema": _DRAFT,
        "$id": f"https://aeroworld.local/schema/arm_semantic_history_{profile}.root.schema.json",
        "title": f"Frozen L6 ARM pre-window semantic history root: {profile}",
        **_object({
            "schema": {"const": "l6_measured_pre_window_evidence"},
            "episode_id": _episode_id_rule(profile),
            "arm_key": _arm_key_rule(profile),
            "end_exclusive_tick": {"type": "integer", "minimum": 0, "maximum": 900},
            "computation": {"const": "continuous execution and producer memory from tick zero"},
            "raw": _object(raw),
            "objective": _object(objective),
        }),
    }


__all__ = [
    "arm_semantic_history_schema",
    "l6_history_subtree_schema",
    "l6_history_subtree_grammar",
    "selected_history_subtrees",
    "validate_history_subtrees",
]
