"""Closed record grammar for L4 ARM dense predicate and domain tick rows.

Source family ``arm_predicate_truth_ticks``; the L4 profile set is
L4-1_v1 .. L4-11_v2 (24 profiles).  Every window writes
``arms/<profile>/<seed>/<group>/<arm_id>/objective/predicate_truth_ticks.jsonl``.

Writer (read this session)
    ``Dataset/tools/l4_window.py::execute_config`` is the only writer:
    lines 651-652 select ``semantic["predicate_truth"]`` and
    ``semantic["l0_predicate_state"]`` rows with ``start <= tick <= end``
    and lines 665-666 persist both lists verbatim (``write_jsonl``
    serialises each row with ``json.dumps(row, ensure_ascii=False)`` and
    adds no key).  The family dispatcher splits the file on
    ``"predicate_id" in record``: rows without ``predicate_id`` are the
    ``arm_domain_state_ticks`` effective type, rows with it are
    ``arm_predicate_truth_ticks``.

Effective type 1 - ``arm_domain_state_ticks`` (formal branch)
    ``semantic["l0_predicate_state"]`` is the ``_state_row`` literal of
    ``Dataset/semantic_simulation/predicate_state_computers.py`` (lines
    1229-1279): 22 fixed envelope keys plus the family payload ``values``,
    the correlated ``missing_source_record`` list and ``quality``, and the
    per-family ``observation_family`` / ``subject_kind`` / ``rule_id``.
    The 14 families this producer emits on L4 are closed below as one
    discriminated variant each: ``observation_family`` const +
    ``subject_kind`` enum + ``rule_id`` const + fully closed ``values``
    object.  Leaves the producer reads through ``values.get(field)``
    (the per-family ``_missing`` sets) are declared nullable; every other
    leaf keeps its produced JSON type.  Aircraft-pair geometry has two
    closed ``values`` branches: ``euclidean_3d`` omits the separation-limit
    keys, while the anisotropic metric requires both numeric limits, as the
    producer's ``if separation_model is not None`` block writes them.

    Integrity-key drift: the current producer emits ``input_digest`` /
    ``parameter_digest`` / ``seed_digest``, but frozen L4 windows predate
    the digest-bearing ``_state_row`` (rows in
    ``arms/L4-1_v1/seed00/grp_converging_approach/.../predicate_truth_ticks.jsonl``
    carry ``model_id``/``model_version`` and none of the three digests).
    The frozen envelope omits all three; the current ``_state_row`` writes
    all three together (``digest_object`` sha256 for ``input_digest`` and
    ``seed_digest``; the governed-defaults path or a sha256 digest for
    ``parameter_digest``).  The grammar admits only these two envelope
    shapes.  The current three-key shape is 尚无实例验证 in the stored L4
    windows inspected here.  The
    publication projection (``structure_schema._PUBLICATION_EXCLUDED_PATHS``)
    already excludes the exact root paths ``("input_digest",)`` and
    ``("parameter_digest",)``; ``seed_digest`` is not in the exclusion set
    and keeps its pattern.  ``validate_record`` consumes the full
    unchanged schema either way, so validation needs no carve-out.

Effective type 2 - ``arm_predicate_truth_ticks`` (business branch)
    ``semantic["predicate_truth"]`` on L4 is
    ``Dataset/tools/l4_business_semantics.py::evaluate``'s row literal
    (lines 336-343): 13 fixed keys.  ``predicate_id`` comes from
    ``rule["predicate_ids"]`` of
    ``Dataset/semantic_rules/profiles/l4_arm_business_semantics.json``
    (loaded fresh at evaluate time; ``l4_window.load_config`` lines
    132-135 enforce that the window config's ids equal the contract
    rule), so the closed enums below restate exactly that contract,
    grouped by the five mechanisms that select the ``metrics`` payload.
    A tick whose sample set is incomplete writes value ``"unknown"`` with
    ``metrics == {}`` (line 222 initialises, lines 335-343 emit), so each
    mechanism's ``metrics`` is declared as the empty object or the full
    measured payload - never a partial set - and the frozen incomplete
    rows (``arms/L4-9_v1/seed00/...``, ``"metrics": {}``,
    bindings spelling ``{"vehicles": [...]}``) stay inside the grammar.
    ``rule_digest`` is the sha256 of the authority file (line 164):
    emitted by the current writer (line 340), absent in frozen L4-9 rows
    (earlier revision) - declared optional like the domain digests; the
    publication projection already excludes the exact root path
    ``("rule_digest",)``.
    For ``uav_pedestrian`` full metrics, ``landing_zone_centre_enu_m`` is
    always a two-number array: ``evaluate`` sets it from the authored source
    descent move before entering the tick loop and may later replace it with
    the executed descent target.  Incomplete ticks carry ``metrics == {}``.

Discrimination.  Both roots discriminate through
``allOf: [{"oneOf": [...]}]`` whose branches constrain
``observation_family``/``rule_id``/``values`` (domain) or
``predicate_id``/``metrics`` (business).  ``predicate_id`` must carry the
discrimination because an incomplete tick emits ``metrics == {}`` for
every mechanism; five metrics-only branches would all match that row and
break the oneOf exactly-one rule.

Quality correlation is deliberately not modeled: ``quality`` and
``missing_source_record`` (domain) and ``value`` and
``observation_status`` (business) are two projections of the same
producer fact, but the grammar keeps the leaves declared plainly instead
of adding if/then coupling.

Sibling shapes are rejected by closure: the L3 six-key dense rows, the
``arm_window_semantics.py`` branch-window rows and the L6
``context``/``sampling_scope`` rows come from different writers and do
not carry these envelopes.  Registry wiring into ``structure_closure.py``
belongs to the integration stage; this module exports pure schema
constructors, imports nothing from the corpus and performs no scans at
import time.  Every call builds fresh objects (no shared mutable state).
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "arm_predicate_truth_ticks_schema",
    "arm_domain_state_ticks_schema",
]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"

_DOMAIN_SCHEMA_NAME = "domain_state_observation"
_DOMAIN_SCHEMA_VERSION = "3.0.0"
_BUSINESS_SCHEMA_NAME = "l4_business_predicate_truth"
_BUSINESS_SCHEMA_VERSION = "1.0.0"
# Verbatim authority_scope of
# Dataset/semantic_rules/profiles/l4_arm_business_semantics.json (line 3),
# copied by evaluate into every row (l4_business_semantics.py line 337).
_BUSINESS_AUTHORITY_SCOPE = (
    "L4 ARM observed business response; not a formal ontology event family"
)

# digest_object(...) output (Dataset/semantic_truth/provenance.py).
_DIGEST_PATTERN = "^sha256:[0-9a-f]{64}$"
# stable_identifier(...) prefix + 24 lowercase hex digits
# (Dataset/semantic_truth/provenance.py lines 42-44).
_OBSERVATION_ID_PATTERN = "^predicate_contract_state:[0-9a-f]{24}$"
# identifier("l4_truth", ...) (Dataset/tools/l4_business_semantics.py
# lines 38-40): literal prefix + sha256[:24] hex digits.
_TRUTH_ID_PATTERN = "^l4_truth:[0-9a-f]{24}$"


def _string(*, non_empty: bool = False) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if non_empty:
        schema["minLength"] = 1
    return schema


def _boolean() -> dict[str, Any]:
    return {"type": "boolean"}


def _number() -> dict[str, Any]:
    return {"type": "number"}


def _integer() -> dict[str, Any]:
    return {"type": "integer", "minimum": 0}


# The compiler's _effective_type rejects "type" arrays ("use explicit
# branches for type alternatives"), so every nullable leaf is a oneOf of
# the value branch and an explicit null branch, as in the house style.
def _nullable_number() -> dict[str, Any]:
    return {"oneOf": [{"type": "number"}, {"type": "null"}]}


def _nullable_integer() -> dict[str, Any]:
    return {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]}


def _nullable_string() -> dict[str, Any]:
    return {
        "oneOf": [
            {"type": "string", "minLength": 1},
            {"type": "null"},
        ],
    }


def _nullable_enum(values: tuple[str, ...]) -> dict[str, Any]:
    """A string enum whose members the producer may also emit as null."""

    return {"oneOf": [{"enum": list(values)}, {"type": "null"}]}


def _sha256_digest() -> dict[str, Any]:
    return {"type": "string", "pattern": _DIGEST_PATTERN}


def _string_list() -> dict[str, Any]:
    return {
        "type": "array",
        "items": _string(non_empty=True),
    }


def _vector(count: int, kind: str = "number") -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": kind},
        "minItems": count,
        "maxItems": count,
    }


def _nullable_vector(count: int) -> dict[str, Any]:
    return {
        "oneOf": [
            {
                "type": "array",
                "items": {"type": "number"},
                "minItems": count,
                "maxItems": count,
            },
            {"type": "null"},
        ],
    }


def _nullable_polygon() -> dict[str, Any]:
    """A boundary polygon of xy points, nullable per the producer's
    ``_missing`` set for the region family."""

    return {
        "oneOf": [
            {
                "type": "array",
                "items": _vector(2),
                "minItems": 3,
            },
            {"type": "null"},
        ],
    }


def _object(
    properties: Mapping[str, Mapping[str, Any]], *,
    required: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties) if required is None else list(required),
        "properties": {str(key): dict(value) for key, value in properties.items()},
    }


def _speed_map() -> dict[str, Any]:
    """Entity-id keyed measured speed map of the two-entity mechanisms
    (intersection, ambulance).

    Keys are the bound state samples' entity ids (unbounded set, so no
    enum), each value the measured speed; a complete sample tick always
    writes at least the two bound entities.
    """

    return {
        "type": "object",
        "minProperties": 1,
        "propertyNames": {"pattern": "^.+$"},
        "additionalProperties": {"type": "number"},
    }


# ---------------------------------------------------------------------------
# arm_domain_state_ticks (formal branch rows of _state_row)
# ---------------------------------------------------------------------------


def _domain_envelope_properties() -> dict[str, dict[str, Any]]:
    """The fixed 22-key ``_state_row`` envelope.

    ``observation_family`` / ``subject_kind`` / ``rule_id`` / ``values``
    keep their base shape here and are closed by the discriminating
    ``oneOf`` in :func:`arm_domain_state_ticks_schema`.
    """

    return {
        "schema_name": {"const": _DOMAIN_SCHEMA_NAME},
        "schema_version": {"const": _DOMAIN_SCHEMA_VERSION},
        "annotation_layer": {"const": "L0"},
        "observation_id": {
            "type": "string",
            "pattern": _OBSERVATION_ID_PATTERN,
        },
        "episode_id": _string(non_empty=True),
        "tick": {"type": "integer", "minimum": 0},
        "observation_family": _string(non_empty=True),
        "subject_id": _string(non_empty=True),
        "subject_kind": _string(non_empty=True),
        # _state_row writes subject_category = subject_kind.
        "subject_category": _string(non_empty=True),
        "source_class": {"const": "derived_from_observed"},
        "rule_id": _string(non_empty=True),
        "rule_version": {"const": _DOMAIN_SCHEMA_VERSION},
        "model_id": _string(non_empty=True),
        "model_version": {"const": _DOMAIN_SCHEMA_VERSION},
        # These keys are absent together in frozen rows and written together
        # by the current _state_row.  The envelope branches below correlate
        # their presence.
        "input_digest": _sha256_digest(),
        "parameter_digest": {
            "oneOf": [
                {"const": "registry/governed_defaults"},
                _sha256_digest(),
            ],
        },
        "seed_digest": _sha256_digest(),
        "source_refs": _string_list(),
        # Closed per family by the discriminating oneOf below.
        "values": {"type": "object"},
        # Normalised "missing_source_record:<ref>&field=<field>" strings;
        # kept as non-empty strings because frozen rows predate the
        # normalised spelling.
        "missing_source_record": _string_list(),
        "quality": {"enum": ["complete", "unknown_missing_source"]},
    }


def _domain_variant(
    family: str,
    subject_kinds: tuple[str, ...],
    rule_id: str,
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "properties": {
            "observation_family": {"const": family},
            "subject_kind": {"enum": list(subject_kinds)},
            "rule_id": {"const": rule_id},
            "values": _object(values),
        },
        "required": ["observation_family", "subject_kind", "rule_id", "values"],
    }


def _aircraft_pair_variant() -> dict[str, Any]:
    # predicate_state_computers.py writes the two limit keys together only
    # when a separation model exists.  Euclidean rows omit both keys; the
    # anisotropic branch writes two numbers (never null).
    values = {
        "first_aircraft_id": _string(non_empty=True),
        "first_aircraft_ontology_class_id": {
            "const": "world:UnmannedAircraft",
        },
        "second_aircraft_id": _string(non_empty=True),
        "second_aircraft_ontology_class_id": {
            "const": "world:UnmannedAircraft",
        },
        "pair_entity_ids": _vector(2, "string"),
        "pair_distance_m": _number(),
        "pair_euclidean_distance_m": _number(),
        "pair_horizontal_distance_m": _number(),
        "pair_vertical_distance_m": _number(),
    }
    variant = _domain_variant(
        "predicate_contract_aircraft_pair_geometry",
        ("uav_pair",),
        "geometric_computer.aircraft_pair_distance",
        values,
    )
    euclidean = _object({
        **values,
        "pair_distance_metric": {"const": "euclidean_3d"},
    })
    anisotropic = _object({
        **values,
        "pair_distance_metric": {
            "const": "anisotropic_safety_volume_equivalent_distance",
        },
        "pair_horizontal_limit_m": _number(),
        "pair_vertical_limit_m": _number(),
    })
    variant["properties"]["values"] = {"oneOf": [euclidean, anisotropic]}
    return variant


def _domain_variants() -> list[dict[str, Any]]:
    """One variant per ``predicate_contract_*`` family the L4 producer
    emits (Dataset/semantic_simulation/predicate_state_computers.py)."""

    return [
        # Lines 480-513: aircraft state with local-ground and landing pads.
        _domain_variant(
            "predicate_contract_aircraft_geometry",
            ("uav",),
            "geometric_computer.aircraft_state",
            {
                # Written as list(position) (line 442); a missing position
                # raises before the row exists, so this leaf is never null.
                "position_enu_m": _vector(3),
                "velocity_enu_mps": _nullable_vector(3),
                "speed_mps": _nullable_number(),
                "z_agl_m": _nullable_number(),
                "local_ground_reference_z_m": _nullable_number(),
                # Not in this family's _missing read set (lines 489-497):
                # the producer always resolves the takeoff ground plane.
                "takeoff_ground_z_m": _number(),
                "home_pad_id": _nullable_string(),
                "home_pad_ontology_class_id": _nullable_enum(
                    ("world:LandingPad",)
                ),
                "home_pad_pose_enu_m": _nullable_vector(3),
                "xy_distance_to_home_pad_m": _nullable_number(),
                "assigned_landing_zone_id": _nullable_string(),
                "assigned_landing_zone_ontology_class_id": _nullable_enum(
                    ("world:LandingArea", "world:LandingPad")
                ),
                "assigned_landing_zone_pose_enu_m": _nullable_vector(3),
                "xy_distance_to_assigned_landing_zone_m": _nullable_number(),
                "assigned_altitude_m": _nullable_number(),
                "altitude_deviation_m": _nullable_number(),
                "aircraft_inside_corridor": _boolean(),
                "containing_corridor_ids": _string_list(),
                # _entity_role (lines 3297-3298) copies the scene-authored
                # role or uav_corridor_role; no governed enum exists.
                "entity_role": _string(non_empty=True),
            },
        ),
        # Lines 515-568: restricted-region prism distances.
        _domain_variant(
            "predicate_contract_aircraft_region_geometry",
            ("uav_region",),
            "geometric_computer.restricted_region_distance",
            {
                "aircraft_id": _string(non_empty=True),
                "aircraft_ontology_class_id": {
                    "const": "world:UnmannedAircraft",
                },
                "restricted_region_id": _string(non_empty=True),
                "restricted_region_ontology_class_id": {
                    "const": "world:RestrictedAirspaceRegion",
                },
                "restricted_region_active": _boolean(),
                "boundary_polygon_enu_m": _nullable_polygon(),
                "distance_to_boundary_m": _nullable_number(),
                "horizontal_distance_to_boundary_m": _number(),
                "signed_distance_to_boundary_m": _number(),
                "inside_restricted_region": _boolean(),
            },
        ),
        # Lines 570-603: protected airspace containment.
        _domain_variant(
            "predicate_contract_aircraft_protected_region_geometry",
            ("uav_protected_region",),
            "geometric_computer.protected_region_containment",
            {
                "aircraft_id": _string(non_empty=True),
                "aircraft_ontology_class_id": {
                    "const": "world:UnmannedAircraft",
                },
                "protected_region_id": _string(non_empty=True),
                "protected_region_ontology_class_id": {
                    "const": "world:ProtectedAirspace",
                },
                "inside_protected_airspace": _boolean(),
                "distance_to_protected_airspace_m": _number(),
            },
        ),
        _domain_variant(
            "predicate_contract_aircraft_corridor_geometry",
            ("uav_corridor",),
            "geometric_computer.aircraft_corridor_membership",
            {
                "aircraft_id": _string(non_empty=True),
                "aircraft_ontology_class_id": {
                    "const": "world:UnmannedAircraft",
                },
                "corridor_id": _string(non_empty=True),
                "corridor_ontology_class_id": {
                    "const": "world:AirspaceCorridor",
                },
                "aircraft_inside_corridor": _boolean(),
                "speed_mps": _nullable_number(),
            },
        ),
        _aircraft_pair_variant(),
        _domain_variant(
            "predicate_contract_corridor_geometry",
            ("airspace_corridor",),
            "geometric_computer.corridor_occupancy_and_capacity",
            {
                "corridor_id": _string(non_empty=True),
                "corridor_ontology_class_id": {
                    "const": "world:AirspaceCorridor",
                },
                "corridor_occupancy_count": _integer(),
                "corridor_capacity": _integer(),
                "occupying_aircraft_ids": _string_list(),
                "corridor_center_enu_m": _vector(3),
                "corridor_extent_m": _vector(3),
                "corridor_yaw_deg": _number(),
                "corridor_cross_section_size_m": _vector(2),
                "minimum_center_separation_m": _number(),
                "capacity_model": {
                    "const": "independent_cross_section_lanes",
                },
            },
        ),
        _domain_variant(
            "predicate_contract_region_runtime_state",
            ("restricted_region",),
            "runtime_region_state_computer.region_activity",
            {
                "no_fly_zone_id": _string(non_empty=True),
                "no_fly_zone_ontology_class_id": {
                    "const": "world:NoFlyZone",
                },
                "restricted_region_active": _boolean(),
                "boundary_margin_m": _number(),
            },
        ),
        _domain_variant(
            "predicate_contract_restricted_airspace_state",
            ("uav",),
            "plan_window_computer.restricted_airspace_effective_range",
            {
                "restricted_region_active": _boolean(),
                "minimum_restricted_boundary_distance_m": _nullable_number(),
                "boundary_margin_m": _number(),
                "active_restricted_region_ids": _string_list(),
            },
        ),
        _domain_variant(
            "predicate_contract_aircraft_plan",
            ("uav",),
            "plan_window_computer.aircraft_control_and_compliance",
            {
                "rth_mode_active": _boolean(),
                # _entity_role: scene-authored role, no governed enum.
                "entity_role": _string(non_empty=True),
                "cooperative_flag": _boolean(),
                "restricted_region_id": _nullable_string(),
            },
        ),
        # Fixed 12-key nearest-neighbour payload of
        # geometric_computer.agent_proximity; all keys are present in
        # every row (verified against frozen L4-1_v1 rows).
        _domain_variant(
            "predicate_contract_agent_proximity_geometry",
            ("uav", "vehicle", "pedestrian"),
            "geometric_computer.agent_proximity",
            {
                "nearest_aircraft_distance_m": _nullable_number(),
                "nearest_aircraft_id": _nullable_string(),
                "nearest_ground_vehicle_distance_m": _nullable_number(),
                "nearest_ground_vehicle_id": _nullable_string(),
                "nearest_vehicle_distance_m": _nullable_number(),
                "nearest_vehicle_id": _nullable_string(),
                "nearest_pedestrian_distance_m": _nullable_number(),
                "nearest_pedestrian_id": _nullable_string(),
                "nearest_population_distance_m": _nullable_number(),
                "nearest_population_id": _nullable_string(),
                "nearest_building_distance_m": _nullable_number(),
                "nearest_building_id": _nullable_string(),
            },
        ),
        # append_pair rows (lines 2369-2423): values are
        # {first_role}_id, {second_role}_id, distance_m and the two role
        # ontology classes from the fixed role_classes map (lines
        # 2396-2401).
        _domain_variant(
            "predicate_contract_aircraft_structure_geometry",
            ("uav_structure",),
            "geometric_computer.full_grounded_pair_distance",
            {
                "aircraft_id": _string(non_empty=True),
                "aircraft_ontology_class_id": {
                    "const": "world:UnmannedAircraft",
                },
                "structure_id": _string(non_empty=True),
                "structure_ontology_class_id": {
                    "const": "world:BuildingStructure",
                },
                "distance_m": _number(),
            },
        ),
        _domain_variant(
            "predicate_contract_aircraft_vehicle_geometry",
            ("uav_vehicle",),
            "geometric_computer.full_grounded_pair_distance",
            {
                "aircraft_id": _string(non_empty=True),
                "aircraft_ontology_class_id": {
                    "const": "world:UnmannedAircraft",
                },
                "vehicle_id": _string(non_empty=True),
                "vehicle_ontology_class_id": {
                    "const": "world:GroundVehicle",
                },
                "distance_m": _number(),
            },
        ),
        _domain_variant(
            "predicate_contract_aircraft_pedestrian_geometry",
            ("uav_pedestrian",),
            "geometric_computer.full_grounded_pair_distance",
            {
                "aircraft_id": _string(non_empty=True),
                "aircraft_ontology_class_id": {
                    "const": "world:UnmannedAircraft",
                },
                "pedestrian_id": _string(non_empty=True),
                "pedestrian_ontology_class_id": {
                    "const": "world:Pedestrian",
                },
                "distance_m": _number(),
            },
        ),
        _domain_variant(
            "predicate_contract_vehicle_pedestrian_geometry",
            ("vehicle_pedestrian",),
            "geometric_computer.full_grounded_pair_distance",
            {
                "vehicle_id": _string(non_empty=True),
                "vehicle_ontology_class_id": {
                    "const": "world:GroundVehicle",
                },
                "pedestrian_id": _string(non_empty=True),
                "pedestrian_ontology_class_id": {
                    "const": "world:Pedestrian",
                },
                "distance_m": _number(),
            },
        ),
    ]


def arm_domain_state_ticks_schema() -> dict[str, Any]:
    """Closed schema for one formal domain-state tick row.

    The record is the ``_state_row`` envelope closed over the 14
    ``predicate_contract_*`` families; frozen rows omit all three integrity
    keys, while the current writer supplies all three together.
    """

    properties = _domain_envelope_properties()
    required = (
        "schema_name",
        "schema_version",
        "annotation_layer",
        "observation_id",
        "episode_id",
        "tick",
        "observation_family",
        "subject_id",
        "subject_kind",
        "subject_category",
        "source_class",
        "rule_id",
        "rule_version",
        "model_id",
        "model_version",
        "source_refs",
        "values",
        "missing_source_record",
        "quality",
    )
    schema = _object(
        properties,
        required=required,
    )
    schema["$schema"] = _DRAFT
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "arm_domain_state_ticks.l4.record.schema.json"
    )
    schema["title"] = "L4 arm domain state tick record"
    schema["allOf"] = [
        {"oneOf": _domain_variants()},
        {"oneOf": [
            # The current _state_row emits the complete three-key envelope.
            {"required": ["input_digest", "parameter_digest", "seed_digest"]},
            # The frozen row has exactly the required keys above.  Since the
            # root is closed, this maximum forbids partial envelopes.
            {"maxProperties": len(required)},
        ]},
    ]
    return schema


# ---------------------------------------------------------------------------
# arm_predicate_truth_ticks (business branch rows of l4_business_semantics)
# ---------------------------------------------------------------------------

# predicate_id groups per mechanism, verbatim from
# Dataset/semantic_rules/profiles/l4_arm_business_semantics.json
# (profiles L4-4_v1/v2, L4-5_v1/v2/v3, L4-9_v1/v2, L4-10_v1/v2,
# L4-11_v1/v2).  The group is the discriminator: evaluate writes exactly
# dict(zip(rule["predicate_ids"], vals)) per mechanism.
_PEDESTRIAN_PREDICATE_IDS = (
    "l4_business.uav_low_altitude",
    "l4_business.aircraft_near_pedestrian",
    "l4_business.landing_zone_occupied",
    "l4_business.pedestrian_clear_of_landing_zone",
    "l4_business.uav_hold_active",
    "l4_business.uav_landed",
)
_VEHICLE_PREDICATE_IDS = (
    "l4_business.uav_vehicle_proximity",
    "l4_business.uav_vehicle_contact",
    "l4_business.uav_damaged",
    "l4_business.uav_airborne",
    "l4_business.vehicle_emergency_stop_active",
    "l4_business.vehicle_physically_stopped",
)
_INTERSECTION_PREDICATE_IDS = (
    "l4_business.near_pair",
    "l4_business.braking_active",
    "l4_business.yielding_active",
    "l4_business.braking_stopped",
)
_AMBULANCE_PREDICATE_IDS = (
    "l4_business.ambulance_approaching_vehicle",
    "l4_business.priority_request_active",
    "l4_business.priority_request_acknowledged",
    "l4_business.civilian_yielding",
    "l4_business.yield_clearance_satisfied",
    "l4_business.ambulance_geometry_passed",
    "l4_business.priority_passage_completed",
)
_AV_PREDICATE_IDS = (
    "l4_business.sensor_fault",
    "l4_business.minimal_risk_maneuver",
    "l4_business.declared_stopped_safe",
    "l4_business.physically_stopped_safe",
)


def _business_envelope_properties() -> dict[str, dict[str, Any]]:
    """The fixed 13-key ``l4_business_semantics.evaluate`` row literal
    (lines 336-343)."""

    return {
        "schema_name": {"const": _BUSINESS_SCHEMA_NAME},
        "schema_version": {"const": _BUSINESS_SCHEMA_VERSION},
        "authority_scope": {"const": _BUSINESS_AUTHORITY_SCOPE},
        "episode_id": _string(non_empty=True),
        # Closed to the authority contract's 27 ids by the oneOf below.
        "predicate_id": _string(non_empty=True),
        "tick": {"type": "integer", "minimum": 0},
        "value": {"enum": ["true", "false", "unknown"]},
        # Closed to the two writer spellings by the oneOf below.
        "bindings": {"type": "object"},
        # sha256 of the authority file (line 164); absent in frozen L4-9
        # rows from the earlier writer revision: declared optional.
        "rule_digest": _sha256_digest(),
        "truth_id": {"type": "string", "pattern": _TRUTH_ID_PATTERN},
        "source_refs": _string_list(),
        # Closed per mechanism by the oneOf below.
        "metrics": {"type": "object"},
        "observation_status": {"enum": ["observed", "inactive_lifecycle"]},
    }


def _metrics_pedestrian() -> dict[str, dict[str, Any]]:
    return {
        "height_agl_m": _number(),
        "horizontal_distance_m": _number(),
        "vertical_distance_m": _number(),
        # The source descent move fixes the centre before the tick loop.
        "landing_zone_centre_enu_m": _vector(2),
        "landing_zone_radius_m": _number(),
        "horizontal_limit_m": _number(),
        "vertical_limit_m": _number(),
        "low_altitude_limit_m": _number(),
        "speed_mps": _number(),
    }


def _metrics_vehicle() -> dict[str, dict[str, Any]]:
    return {
        "distance_3d_m": _number(),
        "proximity_limit_m": _number(),
        "contact_held": _boolean(),
        "height_agl_m": _number(),
        "speed_mps": _number(),
        "stationary_tolerance_mps": _number(),
    }


def _metrics_intersection_base() -> dict[str, dict[str, Any]]:
    return {
        "distance_m": _number(),
        "threshold_m": _number(),
        "near_dwell_ticks": _integer(),
        "required_dwell_ticks": _integer(),
        "speeds_mps": _speed_map(),
    }


def _metrics_ambulance() -> dict[str, dict[str, Any]]:
    metrics = _metrics_intersection_base()
    metrics.update({
        "closing_speed_mps": _number(),
        "downstream_offset_m": _number(),
        "lateral_separation_m": _number(),
        "min_yield_center_separation_m": _number(),
        "approach_direction_xy": _vector(2),
        "approach_normal_xy": _vector(2),
        "approach_move_action_id": _string(non_empty=True),
        "clearance_established": _boolean(),
        # None until the dispatch sample is observed (lines 300, 319).
        "approach_dispatch_tick": _nullable_integer(),
    })
    return metrics


def _metrics_av() -> dict[str, dict[str, Any]]:
    return {
        "speed_mps": _number(),
        "stationary_tolerance_mps": _number(),
    }


def _business_variant(
    predicate_ids: tuple[str, ...],
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """One mechanism branch: the mechanism's predicate ids plus its
    complete-or-empty metrics payload.

    The metrics union is required because an incomplete sample tick emits
    value "unknown" with ``metrics == {}`` for the same predicate ids;
    a full-only payload would reject every frozen incomplete row.
    """

    return {
        "properties": {
            "predicate_id": {"enum": list(predicate_ids)},
            "metrics": {
                "oneOf": [
                    # Incomplete sample tick: {"metrics": {}} (line 222).
                    # maxProperties: 0 keeps this branch disjoint from the
                    # full payload branch so exactly one matches a row.
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "maxProperties": 0,
                    },
                    # Complete sample tick: the full measured payload.
                    _object(metrics),
                ],
            },
        },
        "required": ["predicate_id", "metrics"],
    }


def _business_variants() -> list[dict[str, Any]]:
    return [
        # uav_pedestrian (L4-5 profiles).
        _business_variant(_PEDESTRIAN_PREDICATE_IDS, _metrics_pedestrian()),
        # uav_vehicle (L4-4 profiles).
        _business_variant(_VEHICLE_PREDICATE_IDS, _metrics_vehicle()),
        # intersection (L4-9 profiles).
        _business_variant(
            _INTERSECTION_PREDICATE_IDS, _metrics_intersection_base(),
        ),
        # ambulance (L4-10 profiles).
        _business_variant(_AMBULANCE_PREDICATE_IDS, _metrics_ambulance()),
        # av (L4-11 profiles).
        _business_variant(_AV_PREDICATE_IDS, _metrics_av()),
    ]


def _bindings_variant(keys: tuple[str, ...]) -> dict[str, Any]:
    """One closed bindings spelling of non-empty entity ids.

    Current writer: ``{"entities": sorted(entities)}`` (line 340); frozen
    L4-9 rows store the earlier ``{"vehicles": [...]}`` spelling.
    """

    entity_list = {
        "type": "array",
        "items": _string(non_empty=True),
        "minItems": 1,
    }
    return {
        "properties": {
            "bindings": _object(
                {key: entity_list for key in keys},
                required=keys,
            ),
        },
        "required": ["bindings"],
    }


def arm_predicate_truth_ticks_schema() -> dict[str, Any]:
    """Closed schema for one L4 business predicate truth tick row.

    The record is the 13-key ``evaluate`` row closed over the five
    mechanisms (predicate_id group + metrics payload, oneOf across
    mechanisms; bindings oneOf across the two writer spellings).
    """

    properties = _business_envelope_properties()
    schema = _object(
        properties,
        required=(
            "schema_name",
            "schema_version",
            "authority_scope",
            "episode_id",
            "predicate_id",
            "tick",
            "value",
            "bindings",
            "truth_id",
            "source_refs",
            "metrics",
            "observation_status",
        ),
    )
    schema["$schema"] = _DRAFT
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "arm_predicate_truth_ticks.l4.record.schema.json"
    )
    schema["title"] = "L4 arm predicate truth tick record"
    schema["allOf"] = [
        # Mechanism discrimination: the predicate-id group selects the
        # metrics payload; exactly one branch matches a valid row.
        {"oneOf": _business_variants()},
        {
            "oneOf": [
                _bindings_variant(("entities",)),
                _bindings_variant(("vehicles",)),
            ]
        },
    ]
    return schema
