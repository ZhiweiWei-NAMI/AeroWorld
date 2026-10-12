"""Closed L2 ARM ``raw/states.jsonl`` row grammar.

``l2_window.state_rows`` writes an exact four-key tombstone when an entity
has no trajectory sample.  Otherwise it copies the current engine row and
adds ``observed: true`` and ``record_provenance``.  The observed branch is
split by staged asset identity so that UAV lifecycle, facility scope and
event-actor fields cannot migrate between entity types.

The L2 configuration schema permits a ``set_runtime_state`` intervention
with a governed patch for any of the ten executable runtime families.  Those
families are optional on an observed row and use the shared, producer-matched
raw-value and source-guard fragments.  Fixed L2 scene and event-script
carriers provide the non-runtime preserved fields declared in each branch.
No module import reads data or changes integration wiring.
"""

from __future__ import annotations

import copy

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_FIELDS,
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
    _CONTROL_ENUMS,
    _MISSION_ENUMS,
    _SECURITY_ENUMS,
)
from Dataset.world_model.graph.structure_fragments import (
    nonblank_python_string_rule,
    runtime_enum_string_rule,
    runtime_state_source_guard_rule,
)

__all__ = ["arm_states_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

# The L2 intervention action schema admits a state_patch object; the engine
# validates it against all ten RUNTIME_STATE_FIELDS before merging it onto
# the addressed entity.  Scene carriers use a subset of these families.
_L2_RUNTIME_FAMILIES = RUNTIME_STATE_FIELDS

# label_class values written by the L2 producers (typed_entity_labels from
# Config/LowAltitude/asset_catalog.json); the L2 corpus binds exactly these.
_LABEL_CLASSES = ("charging_pile", "landing_pad", "radio_tower",
                  "traffic_light", "uav", "vehicle")

# The contract selects a command-member domain by family name.
_FAMILY_COMMAND_ENUMS = {
    "control_state": _CONTROL_ENUMS,
    "mission_state": _MISSION_ENUMS,
    "security_state": _SECURITY_ENUMS,
}

# uav_corridor.altitude_layers_m comes from UAV_ALTITUDE_LAYERS_M in
# Dataset/tools/uav_corridor_planner.py:27, stamped as list(UAV_ALTITUDE_
# LAYERS_M) by _set_uav_altitude_metadata.  The literal triple is a
# producer constant, so the sequence is closed in that order.
_ALTITUDE_LAYERS_M = (30.0, 50.0, 80.0)

# Event actors are staged at DEFAULT_DURATION_TICKS + 1 (180 s script grid
# -> tick 901) and spawn via the event script; every in-window L2 entity
# other than the L2-5 event actors activates at tick 0.
_EVENT_ACTOR_ACTIVATION_TICK = 901


def _enum(values) -> dict:
    return {"const": values[0]} if len(values) == 1 else {"enum": list(values)}


def _string_enum(values) -> dict:
    return {"type": "string", **_enum(values)}


def _object(properties: dict, required=None) -> dict:
    """Closed object: every key declared, nothing else allowed."""
    schema = {
        "type": "object",
        "properties": copy.deepcopy(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def _nonempty_string() -> dict:
    return {"type": "string", "minLength": 1}


def _number3() -> dict:
    """Fixed-length 3-vector of numbers (compiler strict-checks int|float)."""
    return {
        "type": "array",
        "prefixItems": [{"type": "number"}, {"type": "number"},
                        {"type": "number"}],
        "minItems": 3,
        "maxItems": 3,
        # compiler requires a typed tail; maxItems makes it unreachable
        "items": False,
    }


def _waypoint() -> dict:
    """One authored route waypoint; preserve integer and float leaves."""
    return {
        "type": "array",
        "prefixItems": [{"type": "number"}, {"type": "number"},
                        {"type": "number"}],
        "minItems": 3,
        "maxItems": 3,
        "items": False,
    }


def _planned_route_waypoints_enu_m() -> dict:
    """planned_route_waypoints_enu_m: stamped nonempty for every UAV
    (regenerate_boundary_scenarios.py 7576 raises when the staged planned
    route holds fewer than 2 waypoints)."""
    return {
        "type": "array",
        "minItems": 1,
        "items": _waypoint(),
    }


def _metadata_rule() -> dict:
    """The executable contract accepts strings nonblank after strip()."""
    return nonblank_python_string_rule()


def _command_rule(family: str) -> dict:
    member_rule = runtime_enum_string_rule(_FAMILY_COMMAND_ENUMS[family])
    return {"oneOf": [member_rule,
                      {"type": "array", "minItems": 1, "items": member_rule}]}


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state field, from the executable patch
    contract (``invalid_runtime_state_value_paths`` in
    ``Dataset/tools/runtime_state_contract.py``); raw value types are kept
    as the producer writes them."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return _metadata_rule()
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        # members = [value] if isinstance(value, str) else value: one string
        # or a nonempty flat sequence, each member inside the family domain
        # the contract selects.
        return _command_rule(family)
    if family == "facility_state" and field == "requester_ids":
        # The producer accepts an empty list; only non-string members fail.
        return {"type": "array", "items": {"type": "string", "minLength": 1}}
    if family == "facility_state" and field in ("capacity", "request_count"):
        return {"type": "number", "minimum": 0}
    if family == "facility_state" and field in (
            "priority_granted_to", "arbitration_winner_id"):
        # the contract rejects a value without a non-whitespace character
        # ("not value.strip()", runtime_state_contract.py 600-601) - the
        # same clause _metadata_rule encodes for the provenance pair, not a
        # plain nonempty check: minLength would admit a whitespace-only
        # string the producer rejects.
        return nonblank_python_string_rule()
    if family == "communication_state" and field in (
            "availability", "link_availability"):
        return {"type": "number", "minimum": 0, "maximum": 1}
    if family == "incident_state" and field in (
            "hazard_concentration_ppm", "hazard_radius_m"):
        return {"type": "number", "minimum": 0}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        return runtime_enum_string_rule(
            RUNTIME_STATE_ENUM_FIELDS[(family, field)])
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _runtime_state_family(family: str) -> dict:
    """One recorded family value on an observed row.

    Production shape: ``deep_merge_mapping`` of validated carriers and
    patches onto the engine entity, which the executable contract keeps a
    flat nonempty mapping of contract fields to scalars.  Key closure is the
    contract itself (consumed fields plus the admitted provenance pair), so
    an L2-authored contract field the current scripts have not yet written
    stays declared, and an unknown key stays rejected.  Family presence on
    the row means a carrier or patch reached the entity
    (``validate_runtime_state_patch`` rejects empty family patches), so
    ``minProperties: 1``.
    """
    fields = sorted(
        RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
        **runtime_state_source_guard_rule("engine_whole"),
    }


def _lifecycle() -> dict:
    """lifecycle: the scene lifecycle contract merged with the auto-lifecycle
    stamps (regenerate_boundary_scenarios.py 7231 + altitude repair 8021).
    ``home_pad_entity_id`` is a scene pad entity id string;
    ``home_hover_enu_m`` / ``mission_start_enu_m`` are the 3-number pad /
    mission-start positions."""
    return _object(
        {
            "assigned_altitude_m": {"type": "number"},
            "auto_lifecycle_applied": {"const": True},
            "home_hover_enu_m": _number3(),
            "home_pad_entity_id": _nonempty_string(),
            "mission_start_enu_m": _number3(),
            "requires_landing_or_terminal_resolution": {"const": True},
            "requires_takeoff": {"const": True},
            "waypoint_repair_applied": {"const": True},
        },
        required=(
            "assigned_altitude_m",
            "auto_lifecycle_applied",
            "home_hover_enu_m",
            "home_pad_entity_id",
            "mission_start_enu_m",
            "requires_landing_or_terminal_resolution",
            "requires_takeoff",
            "waypoint_repair_applied",
        ),
    )


def _observer_lifecycle() -> dict:
    """observer_lifecycle: the corridor-observer lifecycle stamp
    (regenerate_boundary_scenarios.py 4737): presence for the whole episode,
    the airborne hold terminal behavior and the 18.0 m terminal boundary
    distance producer constant."""
    return _object(
        {
            "capture_boundary_id": _nonempty_string(),
            "maximum_terminal_boundary_distance_m": {"type": "number"},
            "presence": {"const": "episode_full_duration"},
            "terminal_behavior": {
                "const": "airborne_hold_inside_runtime_boundary"},
        },
        required=("capture_boundary_id",
                  "maximum_terminal_boundary_distance_m",
                  "presence",
                  "terminal_behavior"),
    )


def _uav_corridor() -> dict:
    """uav_corridor: _set_uav_altitude_metadata stamp (7975)."""
    return _object(
        {
            "assigned_altitude_m": {"type": "number"},
            "altitude_layers_m": {
                "type": "array",
                "prefixItems": [{"const": v} for v in _ALTITUDE_LAYERS_M],
                "minItems": len(_ALTITUDE_LAYERS_M),
                "maxItems": len(_ALTITUDE_LAYERS_M),
                "uniqueItems": True,
                "items": False,
            },
            "used_lateral_bypass": {"type": "boolean"},
        },
        required=("assigned_altitude_m", "altitude_layers_m",
                  "used_lateral_bypass"),
    )


def _semantic_scope(label_class: str) -> dict:
    """The facility scope stamped for this L2 source asset."""
    identity = {
        "radio_tower": ("communication_base_station", "world:CommunicationStation"),
        "charging_pile": ("charging_station", "world:ChargingStation"),
        "landing_pad": ("landing_pad", "world:LandingPad"),
    }
    subtype, ontology_class = identity[label_class]
    fields = {
        "authority_id": {"const": "facility_scope_contract.v1"},
        "ontology_class_id": {"const": ontology_class},
        "scope_subtype": {"const": subtype},
        "scope_type": {"const": "facility"},
    }
    if label_class != "radio_tower":
        fields["service_capacity"] = {"type": "integer"}
    return _object(fields, required=tuple(fields))


def _sensor_profile() -> dict:
    """contract_inspect_uav.sensor_profile: the UE capture camera profile
    copied from Config/LowAltitude/uav_sensor_profile.json (source and
    hfov_deg carry the pointer to that file); the fixed rotation offset is
    the producer's 3-number block."""
    profile = {
        "camera_name": {"const": "bottom_center"},
        "capture_backend": {"const": "editor_hook"},
        "fixed_rotation_offset_deg": _object(
            {
                "pitch_deg": {"type": "number"},
                "roll_deg": {"type": "number"},
                "yaw_deg": {"type": "number"},
            },
            required=("pitch_deg", "roll_deg", "yaw_deg"),
        ),
        "height": {"type": "integer"},
        "hfov_deg": {"type": "number"},
        "source": {"const": "Config/LowAltitude/uav_sensor_profile.json"},
        "width": {"type": "integer"},
    }
    return _object(profile, required=sorted(profile))


def _contract_inspect_uav() -> dict:
    """contract_inspect_uav: the inspect-UAV capture contract stamped by the
    L2/U-inspect builder (regenerate_boundary_scenarios.py 5174).  The route
    arrays hold loop, planned and repaired waypoint lists."""
    routes = {
        "type": "array",
        "minItems": 1,
        "items": _waypoint(),
    }
    contract = {
        "corridor_policy": {"const": "inspect_capture_corridor"},
        "corridor_role": {"const": "inspect_observer"},
        "fixed_altitude_loop": {"type": "boolean"},
        "inspect_altitude_code": _string_enum(("I18", "I30", "I35")),
        "inspect_altitude_m": {"type": "number"},
        "inspect_fov_coverage_required": {"type": "boolean"},
        "loop_route_enu_m": routes,
        "min_path_length_m": {"type": "number"},
        "motion_policy": {
            "const": "fixed_altitude_loop_observes_capture_boundary"},
        "observed_capture_boundary_id": _nonempty_string(),
        "planned_route_enu_m": routes,
        "policy": {"const": "semantic_event_contract_v1"},
        "repaired_path_length_m": {"type": "number"},
        "repaired_route_enu_m": routes,
        "sensor_fov_deg": {"type": "number"},
        "sensor_fov_source": {
            "const": "Config/LowAltitude/uav_sensor_profile.json#hfov_deg"},
        "sensor_profile": _sensor_profile(),
    }
    return _object(contract, required=sorted(contract))


def _path_deviation_contract() -> dict:
    """path_deviation_contract: the L2-2 UAV route-deviation contract
    carrier (predicate_id is the governed predicate; the minimum route
    distance is a producer number)."""
    contract = {
        "minimum_route_distance_m": {"type": "number"},
        "predicate_id": {"const": "positioning.aircraft_deviates_from_planned_route"},
    }
    return _object(contract, required=sorted(contract))


def _event_actor_motion_contract(*, continuous: bool) -> dict:
    """The civilian lane-entry or police dispatch contract, by asset."""
    base = {
        "schema": {"const": "event_actor_motion_contract_v1"},
        "policy": {"const": ("offstage_then_continuous_lane_entry" if continuous
                              else "offstage_until_semantic_dispatch")},
        "semantic_event_id": {"const": ("traffic_confusion" if continuous
                                         else "police_arrival")},
        "pre_activation_presence": {"const": "absent"},
        "stationary_preplacement_allowed": {"const": False},
        "yaw_authority": {"const": "scene_lane_then_velocity"},
    }
    if not continuous:
        return _object(base, required=tuple(base))
    entry = {
        "entry_event_id": _string_enum(("civilian_car_a_fault_entry",
                                        "civilian_car_b_fault_entry")),
        "entry_tick": {"type": "integer"},
        "arrival_tick": {"type": "integer"},
        "route_length_m": {"type": "number"},
        "entry_speed_mps": {"type": "number"},
        "duration_model": {
            "const": "sum_segment_ceil_distance_over_speed_ticks"},
        "minimum_visible_motion_ratio_before_event": {"const": 1.0},
        "entry_capture_boundary_clearance_m": {"type": "number"},
    }
    # event_actor_entry writes seven entry fields.  A later relocation pass
    # may append entry_capture_boundary_clearance_m for visible placement.
    return _object({**base, **entry}, required=tuple(
        field for field in (*base, *entry)
        if field != "entry_capture_boundary_clearance_m"
    ))


def _single_state_sequence(*values: str) -> dict:
    return {
        "type": "array", "minItems": 1, "maxItems": 1,
        "prefixItems": [_string_enum(values) if values else _nonempty_string()],
        "items": False,
    }


def _rounded_float3() -> dict:
    # EpisodeStateEngine._row_for_entity applies round(float(...), 6).
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number", "x-source-value-types": ["float"]},
    }


def _observed_branch() -> dict:
    """Closed L2 engine rows, split by the staged entity identity."""
    core = {
        "tick": {"type": "integer", "minimum": 0, "maximum": 900},
        "entity_id": _nonempty_string(),
        "observed": {"const": True},
        "record_provenance": {"const": "current_engine_execution"},
        "label_class": _string_enum(_LABEL_CLASSES),
        "asset_id": _nonempty_string(),
        "pos_enu": _rounded_float3(),
        "vel_mps": _rounded_float3(),
        "yaw_deg": {"type": "number", "x-source-value-types": ["float"]},
        "state": {"type": "string"},
        "activity_type": {"type": "string"},
        "animation_hint": {"type": "string"},
        "posture": {"const": "standing"},
        "social_state": {"const": "solo"},
        "task_id": _nonempty_string(),
        "role": _string_enum(("U_inspect", "mission_uav",
                              "semantic_facility", "semantic_vehicle")),
        "state_sequence": {"type": "array", "minItems": 1,
                           "items": _nonempty_string()},
        "semantic_role": {"type": "string"},
        "activation_tick": {"type": "integer"},
        "category": _string_enum(("facility", "traffic_light", "uav", "vehicle")),
        "route_waypoints_enu_m": {"type": "array", "items": _waypoint()},
    }
    runtime = {family: _runtime_state_family(family)
               for family in _L2_RUNTIME_FAMILIES}
    uav_fields = {
        "assigned_altitude_m": {"type": "number"},
        "ground_reference_z_m": {"type": "number"},
        "ground_reference_source": _string_enum(
            ("home_hover_pose_origin_v1", "map_ground_reference_v1")),
        "uav_corridor_role": _string_enum(
            ("inspect_observer", "mission_uav", "observer")),
        "planned_route_waypoints_enu_m": _planned_route_waypoints_enu_m(),
        "uav_corridor": _uav_corridor(),
    }
    uav_required = tuple(uav_fields)

    def entity_branch(*, label: str, role: str, category: str,
                      assets: tuple[str, ...], extra: dict,
                      required_extra: tuple[str, ...] = (),
                      activation_tick: int = 0,
                      uav: bool = False,
                      sequence: dict | None = None) -> dict:
        properties = {**copy.deepcopy(core), **copy.deepcopy(runtime)}
        if uav:
            properties.update(copy.deepcopy(uav_fields))
            properties["route_waypoints_enu_m"] = {
                "type": "array", "minItems": 1, "items": _waypoint(),
            }
        else:
            properties["route_waypoints_enu_m"] = {
                "type": "array", "minItems": 0, "maxItems": 0,
                "items": False,
            }
        properties.update(copy.deepcopy(extra))
        properties["label_class"] = {"const": label}
        properties["role"] = {"const": role}
        properties["category"] = {"const": category}
        properties["asset_id"] = _string_enum(assets)
        properties["activation_tick"] = {"const": activation_tick}
        if sequence is not None:
            properties["state_sequence"] = sequence
        required = tuple(core) + (uav_required if uav else ()) + required_extra
        return _object(properties, required=required)

    inspect_fields = {
        "contract_inspect_uav": _contract_inspect_uav(),
        "full_episode_presence": {"const": True},
        "inspect_altitude_code": _string_enum(("I18", "I30", "I35")),
        "min_path_length_m": {"type": "number"},
    }
    facility_assets = {
        "radio_tower": ("facility.radio.base_tower.v1", "facility.base_station"),
        "charging_pile": ("facility.charger.cityops.v1", "facility.charger"),
        "landing_pad": ("facility.landing_pad.visible.v1", "facility.landing_pad"),
    }
    branches = [
        entity_branch(
            label="uav", role="U_inspect", category="uav",
            assets=("uav.inspect.quad.v1",), uav=True,
            extra={**inspect_fields, "uav_corridor_role": {"const": "inspect_observer"}},
            required_extra=tuple(inspect_fields),
            sequence={
                "type": "array", "minItems": 4, "maxItems": 4,
                "prefixItems": [{"const": state} for state in
                                ("takeoff", "inspect_racetrack", "orbit", "land")],
                "items": False,
            },
        ),
        entity_branch(
            label="uav", role="mission_uav", category="uav",
            assets=("uav.airsim.cv_pawn.v1", "uav.airsim.flying_pawn.v1",
                    "uav.inspect.quad.v1"), uav=True,
            extra={"uav_corridor_role": {"const": "mission_uav"},
                   "lifecycle": _lifecycle(),
                   "path_deviation_contract": _path_deviation_contract()},
            required_extra=("lifecycle",),
            sequence=_single_state_sequence(),
        ),
        entity_branch(
            label="uav", role="mission_uav", category="uav",
            assets=("uav.airsim.cv_pawn.v1", "uav.airsim.flying_pawn.v1"),
            uav=True,
            extra={"uav_corridor_role": {"const": "observer"},
                   "observer_lifecycle": _observer_lifecycle()},
            required_extra=("observer_lifecycle",),
            sequence=_single_state_sequence("corridor_observer"),
        ),
    ]
    for label, (asset, entity_kind) in facility_assets.items():
        branches.append(entity_branch(
            label=label, role="semantic_facility", category="facility",
            assets=(asset,),
            extra={"entity_kind": {"const": entity_kind},
                   "semantic_scope": _semantic_scope(label)},
            required_extra=("entity_kind", "semantic_scope"),
            sequence=_single_state_sequence("online" if label == "radio_tower"
                                            else "available"),
        ))
    branches.append(entity_branch(
        label="traffic_light", role="semantic_facility",
        category="traffic_light",
        assets=("prop.traffic_control.signal_light.v1",), extra={},
        sequence=_single_state_sequence("green_cycle"),
    ))
    for asset, continuous, sequence in (
        ("vehicle.ground.boxcar.v1", True, "moving"),
        ("vehicle.emergency.police_suv.v1", False, "response"),
    ):
        branches.append(entity_branch(
            label="vehicle", role="semantic_vehicle", category="vehicle",
            assets=(asset,), activation_tick=_EVENT_ACTOR_ACTIVATION_TICK,
            extra={
                "spawn_policy": {"const": "event_script_only"},
                "event_actor_motion_contract":
                    _event_actor_motion_contract(continuous=continuous),
            },
            required_extra=("spawn_policy", "event_actor_motion_contract"),
            sequence=_single_state_sequence(sequence),
        ))
    return {"oneOf": branches}


def _tombstone_branch() -> dict:
    """The exact literal shape ``state_rows`` writes for a configured
    (tick, entity) pair absent from the window trajectory index
    (``l2_window.py`` 280-281): ``observed`` False and the literal missing
    reason."""
    return {
        "type": "object", "additionalProperties": False,
        "required": ["tick", "entity_id", "observed", "missing_reason"],
        "properties": {
            "tick": {"type": "integer", "minimum": 0, "maximum": 900},
            "entity_id": _nonempty_string(),
            "observed": {"const": False},
            "missing_reason": {
                "const": "no recorded sample for this entity at this tick"},
        },
    }


def arm_states_schema() -> dict:
    """Return the closed record schema for one ``arm_states`` L2 JSONL row.

    ``oneOf`` over the two producer shapes of ``state_rows``: the observed
    engine-row shape and the ``observed: False`` tombstone.  ``observed`` is
    the branch discriminator.  This pure constructor performs no I/O.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_states_l2.record.schema.json",
        "title": "arm_states L2 whole-record grammar (raw/states.jsonl row)",
        "oneOf": [_observed_branch(), _tombstone_branch()],
    }
