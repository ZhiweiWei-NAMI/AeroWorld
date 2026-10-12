"""Closed record structure for L6 ARM dense predicate tick rows.

Source family ``arm_predicate_truth_ticks``; producer:
``Dataset/tools/l6_v2/``
(``arm_runtime`` writes ``computed["objective"]["predicate_truth_ticks"]``
verbatim to ``objective/predicate_truth_ticks.jsonl`` through ``write_rows``,
``Dataset/tools/l6_v2/arm_runtime.py:281-282``).  Rows are built by
``Dataset/tools/l6_v2/semantics.py:evaluate`` as one fixed eight-key dict
literal per configured predicate per tick (lines 345-348):

    {"tick": tick, "predicate_id": pid, "bindings": binding,
     "value": value, "context": context, "missing_inputs": missing,
     "source_refs": refs,
     "sampling_scope": "formal" if tick % 5 == 0 else "diagnostic_off_grid"}

- ``tick``: every tick present in the trajectory stream (``evaluate``
  iterates ``sorted({r["tick"] for r in trajectories})``, line 252); rows on
  the tick % 5 grid are ``formal``, the others ``diagnostic_off_grid``.
- ``predicate_id``: closed to ``ArmSpec.predicate_ids()``
  (``Dataset/tools/l6_v2/arm_runtime.py:111-117`` = sorted union of
  ``BASE_PREDICATES`` lines 45-50, the profile primary, and the
  ``positioning.`` prefix additions that ``PREDICATE_ADDITIONS``
  lines 51-54 inject when the primary predicate has that prefix).  Across
  the validated L6 profiles
  that union is the fifteen identifiers enumerated in ``_L6_PREDICATES``
  below.  ``binding_universe`` (``semantics.py:200-235``) iterates exactly
  ``config["predicate_ids"]``, so no other identifier can reach the writer.
- ``bindings``: exact registry argument roles per predicate
  (``template["argument_roles"]`` via ``get_core_predicate_templates()``),
  written as fixed object keys by ``binding_universe``; values are
  non-empty whitespace-free identifiers (scope entity ids or configured
  service/signal ids).
- ``value`` (lines 337-343): ``'out_of_scope'`` when the actor-station
  binding does not match the active communication assignment (lines 273-274,
  294-296), ``'unknown'`` when the tick-entity row is missing (lines
  263-265) or any state-contract required field is missing (lines 334-336),
  else the template evaluation coerced to ``'true'`` / ``'false'``
  (bool, lines 340-341) or ``'unknown'`` (``None``, lines 342-343).
- ``context``: for every predicate the producer writes either ``{}`` (the
  line 257 initialization, kept whenever the branch's producer row is
  absent; communication station mismatches also leave it empty) or exactly
  one branch payload
  (lines 266-333, quoted at ``_context_for`` below).  Each context node is
  therefore the ``oneOf`` of the empty closed object and the branch payload.
- ``missing_inputs``: always a ``list[str]`` (lines 263-272, 334-336): the
  missing tick-entity marker, the communication producer-absent marker, the
  actor-absent marker, and dotted state-contract required-field paths.
- ``source_refs``: always non-empty; first element is the primary actor's
  trajectory reference (line 344), branch references follow (lines 280-282,
  288-289, 297-298, 303, 309, 315-316, 329).

The rows carry no digest/hash/integrity field, so the publication projection
(``structure_schema._PUBLICATION_EXCLUDED_PATHS``, whose only string-key
entry is ``("raw", "domain_observations", "seed_digest")``) excludes nothing
from these records.  ``seed_digest`` lives on the L6 ``raw/`` writer's
domain-observation rows (``semantics.py:176``); that file is written under
``raw/`` but is not a declared arm source, so no exemption is added here.

L6 writes only predicate rows (``evaluate`` returns only predicate
collections under ``objective``, ``semantics.py:393-401``).  This module
therefore declares only the predicate-row grammar.

Registry wiring into ``structure_closure.py`` (the ``GRAMMAR_SOURCE_FAMILIES``
entry plus a dispatch branch for ``arm_predicate_truth_ticks``) belongs to
the integration stage; this module exports pure schema constructors, imports
no data and performs no scans at import time.
"""

from __future__ import annotations

from typing import Any, Mapping

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates,
)

__all__ = ["arm_predicate_truth_ticks_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# References are assembled at semantics.py:280-329,344.  The producer
# interpolates tick and entity identifiers into these exact file paths.
_TICK_REF = r"(?:0|[1-9][0-9]*)"
_ENTITY_REF = r"[^\s]+"
_TRAJECTORY_REF = (
    rf"^raw/trajectories\.jsonl#tick={_TICK_REF}&entity={_ENTITY_REF}$"
)
# Missing-input entries are producer markers or dotted state-contract paths
# (semantics.py:265, 268, 272, 334-336); binding values are producer
# identifiers.  Both are non-empty whitespace-free strings.
_IDENTIFIER_PATTERN = r"^(?!$)[^\s]+$"

_L6_VALUES = ("true", "false", "unknown", "out_of_scope")
_COMM_DIRECT_VALUES = ("true", "false", "unknown")
_SAMPLING_SCOPES = ("formal", "diagnostic_off_grid")

# Producer predicate set: BASE_PREDICATES + the two security primaries of
# the validated L6 profiles + the PREDICATE_ADDITIONS prefix additions
# (Dataset/tools/l6_v2/arm_runtime.py:45-54 with predicate_ids 111-117).
_BASE_PREDICATES = (
    "airspace.aircraft_airborne",
    "airspace.aircraft_touchdown",
    "airspace.aircraft_at_home_pad",
    "airspace.aircraft_in_landing_zone",
    "airspace.aircraft_outside_assigned_altitude_band",
    "communication.link_degraded",
    "communication.link_unavailable",
    "emergency.aircraft_in_return_to_home",
    "emergency.aircraft_in_safe_hold",
    "positioning.aircraft_deviates_from_planned_route",
)
_GNSS_ADDITION_PREDICATES = (
    "positioning.gnss_quality_degraded",
    "positioning.gnss_spoofing_active",
    "positioning.navigation_error_exceeds_tolerance",
)
_SECURITY_PRIMARY_PREDICATES = (
    "security.communication_jamming_impact_active",
    "security.ground_control_station_compromised",
)
_L6_PREDICATES = (
    *_BASE_PREDICATES,
    *_GNSS_ADDITION_PREDICATES,
    *_SECURITY_PRIMARY_PREDICATES,
)

# The fixed string-to-string mapping the producer embeds as
# ``supported_api_value_keys``
# (Dataset/semantic_simulation/control_response_state.py:31-57, written
# sorted at line 348).
_SUPPORTED_API_VALUE_KEYS = (
    ("communication.backup_link_active", "backup_link_active"),
    ("communication.station_unavailable", "station_unavailable"),
    ("control.alternate_reroute_active", "alternate_reroute_active"),
    ("control.alternate_rth_active", "alternate_rth_active"),
    ("control.altitude_corrective_maneuver_active",
     "altitude_corrective_maneuver_active"),
    ("control.deconfliction_active", "deconfliction_active"),
    ("control.diversion_active", "diversion_active"),
    ("control.divert_active", "divert_active"),
    ("control.evasion_active", "evasion_active"),
    ("control.hold_active", "hold_active"),
    ("control.pull_up_active", "pull_up_active"),
    ("control.reroute_active", "reroute_active"),
    ("control.resequence_active", "resequence_active"),
    ("control.rth_active", "rth_active"),
    ("control.safe_hold_active", "safe_hold_active"),
    ("control.slowdown_active", "slowdown_active"),
    ("mission.abort_active", "mission_abort_active"),
    ("navigation.path_deviation", "path_deviation"),
    ("navigation.route_recovered", "route_recovered"),
    ("navigation.sustained_drift", "sustained_drift"),
    ("security.command_integrity_violation", "command_integrity_violation"),
    ("security.command_lockout_active", "command_lockout"),
    ("security.gcs_compromised", "gcs_compromised"),
    ("security.jamming", "jamming_active"),
    ("security.unauthorized_command", "unauthorized_command"),
)

_GNSS_BRANCH_PREFIXES = (
    "positioning.gnss",
    "positioning.navigation_error_exceeds_tolerance",
)


def _object(
    properties: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    required: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    names = tuple(properties or ()) if required is None else required
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(names),
        "properties": {
            name: dict(schema) for name, schema in (properties or {}).items()
        },
    }


def _pattern_string(pattern: str) -> dict[str, Any]:
    return {"type": "string", "pattern": pattern}


def _integer() -> dict[str, Any]:
    return {"type": "integer", "minimum": 0}


def _bindings_schema(roles: tuple[str, ...]) -> dict[str, Any]:
    return _object(
        {role: _pattern_string(_IDENTIFIER_PATTERN) for role in roles},
        required=roles,
    )


def _source_refs(predicate_id: str) -> dict[str, Any]:
    if predicate_id.startswith("communication."):
        second = (
            rf"^raw/communication_(?:state|ticks)\.jsonl#tick={_TICK_REF}"
            rf"&entity={_ENTITY_REF}$"
        )
        minimum = 1
    elif predicate_id.startswith(_GNSS_BRANCH_PREFIXES):
        second = (
            rf"^raw/domain_observations\.jsonl#tick={_TICK_REF}"
            rf"&subject={_ENTITY_REF}&family=gnss_navigation$"
        )
        minimum = 2
    elif predicate_id.startswith("security."):
        second = (
            rf"^raw/domain_observations\.jsonl#tick={_TICK_REF}"
            rf"&subject={_ENTITY_REF}&family=security_command$"
        )
        minimum = 2
    elif predicate_id == "positioning.aircraft_deviates_from_planned_route" or predicate_id.startswith("emergency."):
        second = (
            rf"^raw/control_observations\.jsonl#tick={_TICK_REF}"
            rf"&entity={_ENTITY_REF}$"
        )
        minimum = 2
    elif predicate_id.startswith("airspace."):
        second = _TRAJECTORY_REF
        minimum = 1
    else:
        raise ValueError(f"unregistered L6 predicate: {predicate_id}")
    return {
        "type": "array",
        "prefixItems": [
            _pattern_string(_TRAJECTORY_REF),
            _pattern_string(second),
        ],
        "items": False,
        "minItems": minimum,
        "maxItems": 2,
    }


def _row_missing_inputs(predicate_id: str, template: Mapping[str, Any]) -> dict[str, Any]:
    markers = ["raw/trajectories.jsonl#missing_tick_entity"]
    if predicate_id.startswith("communication."):
        markers.extend((
            "communication_state.producer_absent_no_authored_station",
            "communication_state.actor",
        ))
    markers.extend(
        field["field"] for field in template["state_contract"]["required_fields"]
    )
    return {
        "type": "array",
        "items": {"type": "string", "enum": markers},
        "uniqueItems": True,
    }


def _bool_or_unknown() -> dict[str, Any]:
    """``bool | "unknown"`` as explicit branches; a bare boolean type would
    also accept ``true``/``false`` where the producer writes the string."""
    return {
        "oneOf": [
            {"type": "boolean", "x-source-value-types": ["bool"]},
            {"const": "unknown"},
        ],
    }


def _number_or_unknown() -> dict[str, Any]:
    return {
        "oneOf": [
            {"type": "number", "x-source-value-types": ["float"]},
            {"const": "unknown"},
        ],
    }


def _vector3_or_unknown() -> dict[str, Any]:
    """``_round_list`` output: a 3-float array or the ``"unknown"`` string."""
    return {
        "oneOf": [
            {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "number", "x-source-value-types": ["float"]},
            },
            {"const": "unknown"},
        ],
    }


def _control_values() -> dict[str, Any]:
    """Values payload of ``control_response_state._derive_values``.

    The dict literal (``Dataset/semantic_simulation/control_response_state.py:
    347-393``) is fixed at 33 keys: ``supported_api_value_keys`` (the sorted
    25-entry mapping of ``SUPPORTED_API_VALUE_KEYS``, lines 31-57), 26
    boolean-or-``"unknown"`` fields, and 6 polled floats rounded to six
    decimals by ``_round`` (lines 1037-1038), which maps ``None`` to
    ``"unknown"`` — never to ``null``.
    """

    return _object({
        "supported_api_value_keys": _object({
            key: {"const": value} for key, value in _SUPPORTED_API_VALUE_KEYS
        }),
        "rth_active": _bool_or_unknown(),
        "kinematic_rth_active": _bool_or_unknown(),
        "alternate_rth_active": _bool_or_unknown(),
        "reroute_active": _bool_or_unknown(),
        "alternate_reroute_active": _bool_or_unknown(),
        "altitude_corrective_maneuver_active": _bool_or_unknown(),
        "deconfliction_active": _bool_or_unknown(),
        "evasion_active": _bool_or_unknown(),
        "pull_up_active": _bool_or_unknown(),
        "safe_hold_active": _bool_or_unknown(),
        "hold_active": _bool_or_unknown(),
        "resequence_active": _bool_or_unknown(),
        "divert_active": _bool_or_unknown(),
        "diversion_active": _bool_or_unknown(),
        "slowdown_active": _bool_or_unknown(),
        "mission_abort_active": _bool_or_unknown(),
        "station_unavailable": _bool_or_unknown(),
        "backup_link_active": _bool_or_unknown(),
        "path_deviation": _bool_or_unknown(),
        "route_recovered": _bool_or_unknown(),
        "sustained_drift": _bool_or_unknown(),
        "gcs_compromised": _bool_or_unknown(),
        "jamming_active": _bool_or_unknown(),
        "unauthorized_command": _bool_or_unknown(),
        "command_integrity_violation": _bool_or_unknown(),
        "command_lockout": _bool_or_unknown(),
        "route_distance_m": _number_or_unknown(),
        "previous_route_distance_m": _number_or_unknown(),
        "home_distance_m": _number_or_unknown(),
        "previous_home_distance_m": _number_or_unknown(),
        "speed_delta_mps": _number_or_unknown(),
        "altitude_error_m": _number_or_unknown(),
    })


def _control_context() -> dict[str, Any]:
    """``{"control": values, "plan": {"rth_mode_active": values["rth_active"]}}``
    for the ``emergency.*`` branch (``semantics.py:307-308``)."""

    return _object({
        "control": _control_values(),
        "plan": _object({
            "rth_mode_active": _bool_or_unknown(),
        }),
    })


def _gnss_navigation_values() -> dict[str, Any]:
    """Values payload of the embedded ``gnss_navigation`` row.

    Producer dict ``domain_state._gnss_values``
    (``Dataset/semantic_simulation/domain_state.py:3742-3759``) minus the
    popped ``_next_error_range_m`` (``semantics.py:163``), plus the four L6
    identity keys (``semantics.py:166-169``).  The original pipeline adds
    six keys at ``domain_state.py:1249-1269`` including two
    ``*_source_ref`` strings that ``producer_observations`` does not
    reproduce.  ``quality_level`` / ``operational_state`` / ``fsm_state``
    come from the candidate-vote collapse (lines 3724-3738, 3663-3679) and
    fall back to ``"unknown"``; vectors and ``position_error_m`` come from
    ``_round`` / ``_round_list`` (lines 5084-5095, always float or
    ``"unknown"``).
    """

    return _object({
        "quality_level": {
            "type": "string",
            "enum": ["nominal", "degraded", "spoofing_suspect", "unknown"],
        },
        "truth_position_enu_m": _vector3_or_unknown(),
        "reported_position_enu_m": _vector3_or_unknown(),
        "position_error_m": _number_or_unknown(),
        "drift_offset_enu_m": _vector3_or_unknown(),
        "operational_state": {
            "type": "string",
            "enum": [
                "nominal", "multipath_degraded", "spoofed", "geofence_alert",
                "relocalizing", "recovered", "unknown",
            ],
        },
        "multipath_warning": _bool_or_unknown(),
        "visual_relocalization": _bool_or_unknown(),
        "gnss_spoofed": _bool_or_unknown(),
        "geofence_alert": _bool_or_unknown(),
        "mission_recovered": _bool_or_unknown(),
        "fsm_state": {
            "type": "string",
            "enum": [
                "nominal", "multipath_degraded", "spoofed", "geofence_alert",
                "relocalizing", "recovered", "unknown",
            ],
        },
        "recovery_decay": {
            "type": "number",
            "x-source-value-types": ["float"],
        },
        "missing_inputs": {
            "type": "array",
            "items": {"type": "string", "enum": [
                "truth_frames.entities[].truth_pose.position_enu_m",
                "truth_frames.entities[].navigation_state",
                "truth_frames.entities[].navigation_state.multipath_warning",
                "truth_frames.entities[].navigation_state.visual_relocalization|visual_relocalization_active",
                "truth_frames.entities[].navigation_state.gnss_spoofed|spoofing_active",
                "truth_frames.entities[].navigation_state.geofence_alert|geofence_violation",
                "truth_frames.entities[].navigation_state.mission_recovered|relocalization_complete",
                "truth_frames.entities[].navigation_state.operational_state|gnss_mode|navigation_mode|recovery_state",
            ]},
            "uniqueItems": True,
        },
        "actor_id": _pattern_string(_IDENTIFIER_PATTERN),
        "actor_ontology_class_id": {"const": "world:UnmannedAircraft"},
        "gnss_service_id": _pattern_string(_IDENTIFIER_PATTERN),
        "spoofing_signal_id": _pattern_string(_IDENTIFIER_PATTERN),
    })


def _security_command_values() -> dict[str, Any]:
    """Values payload of the embedded ``security_command`` row.

    Producer dict ``domain_state._security_rows``
    (``Dataset/semantic_simulation/domain_state.py:2950-2985``): actor/GCS
    identities are exclusive and null-able, ``operational_state`` comes
    from ``_exact_enum`` over ("mode", "status", "threat", "condition")
    against the nine-value set (lines 2855-2869) falling back to
    ``"unknown"``, scores are ``_round``-rounded floats or ``"unknown"``
    (0.25/1.0 and 0.1/1.0/None, lines 2903-2914),
    ``spectrum_interference_ratio`` is 1.0/0.0/``"unknown"``
    (lines 2922-2924), and ``lockout_duration_ticks`` is a non-negative int
    or ``"unknown"`` (lines 2979-2981).
    """

    return _object({
        "actor_id": {
            "oneOf": [
                _pattern_string(_IDENTIFIER_PATTERN),
                {"type": "null"},
            ],
        },
        "actor_ontology_class_id": {
            "oneOf": [
                {"const": "world:UnmannedAircraft"},
                {"type": "null"},
            ],
        },
        "ground_control_station_id": {
            "oneOf": [
                _pattern_string(_IDENTIFIER_PATTERN),
                {"type": "null"},
            ],
        },
        "ground_control_station_ontology_class_id": {
            "oneOf": [
                {"const": "world:GroundControlStation"},
                {"type": "null"},
            ],
        },
        "communication_station_id": {
            "oneOf": [
                _pattern_string(_IDENTIFIER_PATTERN),
                {"type": "null"},
            ],
        },
        "communication_station_ontology_class_id": {
            "oneOf": [
                {"const": "world:CommunicationStation"},
                {"type": "null"},
            ],
        },
        "operational_state": {
            "type": "string",
            "enum": [
                "nominal", "normal", "clear", "none", "gcs_compromised",
                "jamming", "unauthorized_command",
                "command_integrity_violation", "command_lockout", "unknown",
            ],
        },
        "auth_score": _number_or_unknown(),
        "auth_state": {
            "type": "string",
            "enum": [
                "authenticated", "locked_or_recovering", "compromised",
                "unknown",
            ],
        },
        "command_integrity_score": _number_or_unknown(),
        "command_integrity": _bool_or_unknown(),
        "command_source_valid": _bool_or_unknown(),
        "jamming_indicator": _bool_or_unknown(),
        "spectrum_interference_ratio": _number_or_unknown(),
        "lockout_active": _bool_or_unknown(),
        "lockout_duration_ticks": {
            "oneOf": [
                {
                    "type": "integer",
                    "minimum": 0,
                    "x-source-value-types": ["int"],
                },
                {"const": "unknown"},
            ],
        },
        "missing_inputs": {
            "type": "array",
            "items": {"const": "truth_frames.entities[].security_state"},
            "maxItems": 1,
        },
    })


def _gnss_domain_context() -> dict[str, Any]:
    """``{"domain": {"gnss_navigation": values}}``
    (``semantics.py:287``)."""

    return _object({
        "domain": _object({
            "gnss_navigation": _gnss_navigation_values(),
        }),
    })


def _security_domain_context() -> dict[str, Any]:
    """``{"domain": {"security_command": values}}``
    (``semantics.py:293``)."""

    return _object({
        "domain": _object({
            "security_command": _security_command_values(),
        }),
    })


def _navigation_context() -> dict[str, Any]:
    """``{"navigation": {"path_deviation": ...}}`` for the route-deviation
    branch (``semantics.py:301-302``)."""

    return _object({
        "navigation": _object({
            "path_deviation": _bool_or_unknown(),
        }),
    })


def _geometry_block() -> dict[str, Any]:
    """``geometry`` payload of the ``airspace.*`` branch
    (``semantics.py:320-332``).

    The seven base fields are written whenever the primary-actor trajectory
    row exists (line 317 guard); the two ``xy_distance_*`` fields are
    appended together only when the home-pad trajectory row exists
    (lines 328-332).

    The six-file bounded sample observed the nine-field form; the seven-field
    form is declared from the writer condition and remains unobserved in that
    sample.  ``assigned_altitude_m`` and ``local_ground_reference_z_m``
    come from the profile config (plain numbers); the remaining numerics
    are code-guaranteed floats (``row["pos_enu"]`` arithmetic and
    ``math.dist``).
    """

    base = {
        "position_z_m": {"type": "number", "x-source-value-types": ["float"]},
        "assigned_altitude_m": {"type": "number"},
        "local_ground_reference_z_m": {"type": "number"},
        "z_agl_m": {"type": "number", "x-source-value-types": ["float"]},
        "speed_mps": {"type": "number", "x-source-value-types": ["float"]},
        "home_pad_id": _pattern_string(_IDENTIFIER_PATTERN),
        "assigned_landing_zone_id": _pattern_string(_IDENTIFIER_PATTERN),
    }
    distances = {
        "xy_distance_to_home_pad_m": {
            "type": "number",
            "x-source-value-types": ["float"],
        },
        "xy_distance_to_assigned_landing_zone_m": {
            "type": "number",
            "x-source-value-types": ["float"],
        },
    }
    return {"oneOf": [_object(base), _object({**base, **distances})]}


def _context_node(payload: dict[str, Any]) -> dict[str, Any]:
    """Branch context: ``{}`` when the branch's producer row is absent,
    else exactly the payload
    (``semantics.py:257``, 266-333)."""

    return {"oneOf": [_object(), payload]}


def _context_for(predicate_id: str) -> dict[str, Any]:
    """Producer branch payloads, mirroring ``semantics.evaluate`` 266-333.

    - ``communication.*`` (266-282): ``{"direct_predicate_value": value}``
      with value from ``compute_comm._truth_from_link_degraded``
      (2498-2506) / ``_truth_from_link_unavailable`` (2509-2517), both
      returning only ``'true'`` / ``'false'`` / ``'unknown'``.
    - ``positioning.gnss*`` / ``positioning.navigation_error_exceeds_tolerance``
      (283-289): ``{"domain": {"gnss_navigation": values}}`` when the
      observation row exists.
    - ``security.*`` (290-298): ``{"domain": {"security_command": values}}``
      when the observation row exists.
    - ``positioning.aircraft_deviates_from_planned_route`` (299-303):
      ``{"navigation": {"path_deviation": ...}}`` when the control row
      exists.
    - ``emergency.*`` (304-309): ``{"control": values,
      "plan": {"rth_mode_active": values["rth_active"]}}`` when the control
      row exists.
    - ``airspace.*`` with an observed primary row (317-333):
      ``{"geometry": {...}}``.
    - Otherwise the branch is skipped and ``context`` stays ``{}``
      (line 257).
    """

    if predicate_id.startswith("communication."):
        return _context_node(_object({
            "direct_predicate_value": {
                "type": "string",
                "enum": list(_COMM_DIRECT_VALUES),
            },
        }))
    if predicate_id == "positioning.aircraft_deviates_from_planned_route":
        return _context_node(_navigation_context())
    if predicate_id.startswith(_GNSS_BRANCH_PREFIXES):
        return _context_node(_gnss_domain_context())
    if predicate_id.startswith("security."):
        return _context_node(_security_domain_context())
    if predicate_id.startswith("emergency."):
        return _context_node(_control_context())
    if predicate_id.startswith("airspace."):
        return _context_node(_object({"geometry": _geometry_block()}))
    raise ValueError(f"unregistered L6 predicate: {predicate_id}")


def _dense_row(predicate_id: str) -> dict[str, Any]:
    template = next(
        item for item in get_core_predicate_templates()
        if item["id"] == predicate_id
    )
    roles = tuple(str(role["key"]) for role in template["argument_roles"])
    return _object({
        "tick": _integer(),
        "predicate_id": {"const": predicate_id},
        "bindings": _bindings_schema(roles),
        "value": {"type": "string", "enum": list(_L6_VALUES)},
        "context": _context_for(predicate_id),
        "missing_inputs": _row_missing_inputs(predicate_id, template),
        "source_refs": _source_refs(predicate_id),
        "sampling_scope": {
            "type": "string",
            "enum": list(_SAMPLING_SCOPES),
        },
    })


def arm_predicate_truth_ticks_schema(layer: str = "L6") -> dict[str, Any]:
    """Return the closed schema for one L6 dense predicate tick row.

    Argument roles are read from the governed registry at call time (the
    same authority ``binding_universe`` consumes); no state is cached or
    shared with other modules.
    """

    if layer != "L6":
        raise ValueError(
            f"arm_predicate_truth_ticks declares layer L6 only, got {layer!r}"
        )
    return {
        "$schema": _DRAFT,
        "$id": (
            "https://aeroworld.local/schema/"
            "arm_predicate_truth_ticks_l6.record.schema.json"
        ),
        "title": "L6 ARM dense predicate tick record",
        "oneOf": [_dense_row(predicate_id) for predicate_id in _L6_PREDICATES],
    }
