"""Closed whole-row grammar for L4 ARM raw trajectory JSONL records.

``Dataset/tools/l4_window.py`` writes its window-scoped engine ``trajectories``
list to ``raw/trajectories.jsonl`` and ``raw/states.jsonl``.  Rows come from
``EpisodeStateEngine._row_for_entity``: fixed pose and activity keys followed
by the preserved scene fields and any validated runtime-state patch.  The
schema covers the L4 scoped UAV, pedestrian, vehicle and asset-anchor records,
including the L4 event-actor motion contracts and UAV lifecycle stamps.

Runtime family names, fields and value domains come from the executable
``runtime_state_contract`` and the shared string/source-guard fragments.
The root closes every declared key; dynamic entity IDs remain strings rather
than an observed ID list.  No integrity metadata is written on these rows.
Registry wiring and source selection belong to the integration stage.
"""

from __future__ import annotations

import copy

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
    RUNTIME_STATE_FIELDS,
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

__all__ = ["arm_trajectories_schema"]


_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# Scoped L4 scene roles and their declared assets.
_LABEL_CLASSES = ("pedestrian", "uav", "vehicle", "asset_anchor")
_CATEGORIES = ("pedestrian", "uav", "vehicle", "facade_anchor")
_ROLES = ("semantic_pedestrian", "mission_uav", "semantic_vehicle",
          "semantic_context")
_ASSETS = (
    "uav.inspect.quad.v1",
    "uav.airsim.flying_pawn.v1",
    "vehicle.ground.boxcar.v1",
    "vehicle.emergency.ambulance.v1",
    "vehicle.emergency.suv.v1",
    "pedestrian.cityops.basic.v1",
    "semantic.asset_anchor",
)
# uav_corridor.altitude_layers_m is list(UAV_ALTITUDE_LAYERS_M) written at
# regenerate_boundary_scenarios.py:7995; UAV_ALTITUDE_LAYERS_M is the
# producer constant (30.0, 50.0, 80.0) in Dataset/tools/uav_corridor_planner
# .py:27, so the sequence is closed with that order and no extras.
_ALTITUDE_LAYERS_M = (30.0, 50.0, 80.0)

# The L4 home-hover UAV builder writes this source; the map source belongs to
# other layers' scene builders and is outside the L4 trajectory input chain.
_GROUND_REFERENCE_SOURCES = ("home_hover_pose_origin_v1",)

_FAMILY_COMMAND_ENUMS = {
    "control_state": _CONTROL_ENUMS,
    "mission_state": _MISSION_ENUMS,
    "security_state": _SECURITY_ENUMS,
}

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


def _float3() -> dict:
    """3-vector whose members the producer always writes as floats."""
    vector = _number3()
    vector["prefixItems"] = [
        {"type": "number", "x-source-value-types": ["float"]}
        for _ in range(3)
    ]
    return vector


def _float() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """The raw value accepted by the executable runtime-state contract."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return nonblank_python_string_rule()
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        member = runtime_enum_string_rule(_FAMILY_COMMAND_ENUMS[family])
        return {"oneOf": [
            member,
            {"type": "array", "minItems": 1, "items": member},
        ]}
    if family == "facility_state" and field == "requester_ids":
        return {"type": "array", "items": {"type": "string", "minLength": 1}}
    if family == "facility_state" and field in ("capacity", "request_count"):
        return {"type": "number", "minimum": 0}
    if family == "facility_state" and field in (
            "priority_granted_to", "arbitration_winner_id"):
        return nonblank_python_string_rule()
    if family == "communication_state" and field in (
            "availability", "link_availability"):
        return {"type": "number", "minimum": 0, "maximum": 1}
    if family == "incident_state" and field in (
            "hazard_concentration_ppm", "hazard_radius_m"):
        return {"type": "number", "minimum": 0}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        return runtime_enum_string_rule(RUNTIME_STATE_ENUM_FIELDS[(family, field)])
    raise ValueError(f"runtime state field outside contract: {family}.{field}")


def _runtime_state_family(family: str) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family]
                    | RUNTIME_STATE_METADATA_FIELDS)
    return {
        **_object({name: _runtime_state_field_rule(family, name)
                   for name in fields}),
        "minProperties": 1,
        **runtime_state_source_guard_rule("engine_whole"),
    }


def _event_actor_motion_contract() -> dict:
    """event_actor_motion_contract: oneOf the two producer-stamped variants.

    ``stage_event_actor`` writes the six-key dispatch variant.  The
    continuous-lane entry adds seven keys; its lane-placement pass may add
    ``entry_capture_boundary_clearance_m`` later.
    """
    base = {
        "schema": {"const": "event_actor_motion_contract_v1"},
        "semantic_event_id": {"type": "string", "minLength": 1},
        "pre_activation_presence": {"const": "absent"},
        "stationary_preplacement_allowed": {"const": False},
        "yaw_authority": {"const": "scene_lane_then_velocity"},
    }
    entry = {
        "entry_event_id": {"type": "string", "minLength": 1},
        "entry_tick": {"type": "integer", "minimum": 0},
        "arrival_tick": {"type": "integer", "minimum": 1},
        "route_length_m": _float(),
        "entry_speed_mps": _float(),
        "duration_model": {"const": "sum_segment_ceil_distance_over_speed_ticks"},
        "minimum_visible_motion_ratio_before_event": {"const": 1.0},
        "entry_capture_boundary_clearance_m": _float(),
    }
    return {
        "oneOf": [
            _object(
                {**base, "policy": {"const": "offstage_until_semantic_dispatch"}},
                required=tuple(sorted({*base, "policy"})),
            ),
            _object(
                {**base, **entry,
                 "policy": {"const": "offstage_then_continuous_lane_entry"}},
                required=tuple(sorted((set(base) | set(entry) | {"policy"})
                                      - {"entry_capture_boundary_clearance_m"})),
            ),
        ]
    }


def _waypoint() -> dict:
    """One ENU waypoint: exactly three float members (q / round(..., 6))."""
    return {
        "type": "array",
        "prefixItems": [{"type": "number", "x-source-value-types": ["float"]}
                        for _ in range(3)],
        "minItems": 3,
        "maxItems": 3,
        "items": False,
    }


def _route_waypoints_enu_m() -> dict:
    """An empty route or a nonempty list of authored ENU waypoints."""
    return {
        "oneOf": [
            # exactly-empty route; the compiler expands container consts into
            # this shape, written out here for the grammar to stay explicit
            {"type": "array", "minItems": 0, "maxItems": 0, "items": False},
            {"type": "array", "minItems": 1, "items": _waypoint()},
        ]
    }


def _planned_route_waypoints_enu_m() -> dict:
    return {
        "type": "array",
        "minItems": 1,
        "items": _waypoint(),
    }


def _lifecycle() -> dict:
    """lifecycle: producer-stamped block; +terminal_flight_state only on L4-4.

    The scene builder writes the six-key lifecycle block; its altitude pass
    adds two keys, and the L4-4 terminal stage alone adds
    ``terminal_flight_state``.  It is optional for the other L4 profiles.
    """
    return _object(
        {
            "auto_lifecycle_applied": {"const": True},
            "home_pad_entity_id": {"type": "string", "minLength": 1},
            "home_hover_enu_m": _float3(),
            "mission_start_enu_m": _float3(),
            "requires_takeoff": {"const": True},
            "requires_landing_or_terminal_resolution": {"const": True},
            "waypoint_repair_applied": {"const": True},
            "assigned_altitude_m": _float(),
            "terminal_flight_state": {"const": True},
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


def _uav_corridor() -> dict:
    """uav_corridor: _set_uav_altitude_metadata stamp (7993-7997)."""
    return _object(
        {
            "assigned_altitude_m": _float(),
            "altitude_layers_m": {
                "type": "array",
                "prefixItems": [{"const": value} for value in _ALTITUDE_LAYERS_M],
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


def arm_trajectories_schema() -> dict:
    """Whole-record JSON Schema for one L4 raw/trajectories.jsonl row.

    Pure schema constructor: no I/O, no registry or compiler interaction.
    The record root is entity-scoped and time-anchored (arm_trajectories is
    a DIRECT_TICK_FAMILIES member; root tick is the window time_point).
    """
    row_core = {
        "tick": {"type": "integer", "minimum": 0, "maximum": 900},
        "entity_id": {"type": "string", "minLength": 1},
        "label_class": _string_enum(_LABEL_CLASSES),
        "asset_id": _string_enum(_ASSETS),
        "pos_enu": _float3(),
        "vel_mps": _float3(),
        "yaw_deg": _float(),
        "state": {"type": "string"},
        "activity_type": {"type": "string"},
        "animation_hint": {"type": "string"},
        "posture": {"type": "string"},
        "social_state": {"type": "string"},
        "task_id": {"type": "string"},
        "role": _string_enum(_ROLES),
        "state_sequence": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "semantic_role": {"type": "string"},
        "activation_tick": {"type": "integer"},
        "category": _string_enum(_CATEGORIES),
        "route_waypoints_enu_m": _route_waypoints_enu_m(),
    }

    # row_activity_payload (batch_generate.py:1856): pedestrian rows carry
    # the activity catalog (get_activity), every other label carries the
    # state string in all four fields with posture standing / social_state
    # solo. The catalog closes posture to fallen/standing and social_state
    # to group/observer/solo.
    pedestrian_payload = _object(
        {
            "activity_type": {"type": "string"},
            "animation_hint": {"type": "string"},
            "posture": _string_enum(("fallen", "standing")),
            "social_state": _string_enum(("group", "observer", "solo")),
        }
    )
    non_pedestrian_payload = _object(
        {
            "activity_type": {"type": "string"},
            "animation_hint": {"type": "string"},
            "posture": {"const": "standing"},
            "social_state": {"const": "solo"},
        }
    )

    # preserved_fields_from(PRESERVED_ENTITY_FIELDS, batch_generate.py:215):
    # per-key presence follows the engine entity, so keys are optional and
    # never null. row.update(preserved_fields_from(entity)) applies them
    # after row_activity_payload. The four preserved keys that carry the
    # scene entity verbatim (spawn_policy / event_actor_motion_contract /
    # lifecycle / uav_corridor) are bound to their own closed subschemas.
    preserved = {
        "assigned_altitude_m": _float(),
        "ground_reference_z_m": _float(),
        "ground_reference_source": _string_enum(_GROUND_REFERENCE_SOURCES),
        "uav_corridor_role": {"const": "mission_uav"},
        "planned_route_waypoints_enu_m": _planned_route_waypoints_enu_m(),
        "lifecycle": _lifecycle(),
        "uav_corridor": _uav_corridor(),
        "spawn_policy": {"const": "event_script_only"},
        "event_actor_motion_contract": _event_actor_motion_contract(),
    }

    families = {
        family: _runtime_state_family(family) for family in RUNTIME_STATE_FIELDS
    }

    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/arm_trajectories_l4.record.schema.json",
        "title": "L4 ARM raw trajectory record (raw/trajectories.jsonl row)",
        "type": "object",
        "properties": {
            **row_core,
            **preserved,
            **families,
        },
        "required": sorted(row_core),
        "additionalProperties": False,
        # route_waypoints_enu_m is nonempty iff planned_route_waypoints_enu_m
        # is present (both written from the same entity route; the planned
        # copy is stamped for UAVs at regenerate_boundary_scenarios.py).
        # These conditions live at the record root because a subschema
        # condition on the array can only constrain the array itself.
        "allOf": [
            {
                "if": {
                    "properties": {
                        "route_waypoints_enu_m": {"minItems": 1},
                    },
                    "required": ["route_waypoints_enu_m"],
                },
                "then": {"required": ["planned_route_waypoints_enu_m"]},
            },
            {
                "if": {"required": ["planned_route_waypoints_enu_m"]},
                "then": {
                    "properties": {
                        "route_waypoints_enu_m": {"minItems": 1},
                    },
                },
            },
            # _set_uav_altitude_metadata writes the same altitude into the
            # row, the corridor and the lifecycle block.
            {
                # _set_uav_altitude_metadata (7987-7997) stamps the row's
                # assigned_altitude_m (via initial_state) and uav_corridor in
                # the same call and early-returns before both when the
                # repaired route is empty, so the two slots are
                # present-together. The value itself is not pinned: the
                # altitude is chosen per scene (_choose_uav_altitude 7774,
                # fixed_altitudes 8254, U_inspect branch 8246), so an
                # observed altitude list would not be a producer bound.
                # The same stamp copies the value into
                # uav_corridor.assigned_altitude_m and - when a lifecycle
                # block exists - into lifecycle.assigned_altitude_m
                # (8018-8026); cross-key value equality is a producer
                # invariant the schema cannot express, stated here instead.
                "if": {"required": ["uav_corridor"]},
                "then": {"required": ["assigned_altitude_m"]},
            },
            {
                "if": {"required": ["assigned_altitude_m"]},
                "then": {"required": ["uav_corridor"]},
            },
        ],
        # Producer branch the row belongs to.  Branch 1 binds the
        # pedestrian tuple; branch 2 covers the other labels and binds each
        # label<->role<->category<->asset_id tuple the producer derives from
        # Config/LowAltitude/asset_catalog.json. Each branch is decisive
        # (label_class is a const / disjoint enum in the branch), so a
        # pedestrian-labeled row satisfies only branch 1 and every other row
        # exactly one label branch - no vacuous then without else. The
        # non-pedestrian payload constants (posture standing / social_state
        # solo) ride along with branch 2 instead of being duplicated per
        # label.
        "oneOf": [
            {
                "properties": {
                    "label_class": {"const": "pedestrian"},
                    "role": {"const": "semantic_pedestrian"},
                    "category": {"const": "pedestrian"},
                    "asset_id": {"const": "pedestrian.cityops.basic.v1"},
                    **pedestrian_payload["properties"],
                },
            },
            {
                "properties": {
                    "label_class": {"enum": ["asset_anchor", "uav", "vehicle"]},
                    **non_pedestrian_payload["properties"],
                },
                "oneOf": [
                    {
                        "properties": {
                            "label_class": {"const": "asset_anchor"},
                            "role": {"const": "semantic_context"},
                            "category": {"const": "facade_anchor"},
                            "asset_id": {"const": "semantic.asset_anchor"},
                        },
                    },
                    {
                        "properties": {
                            "label_class": {"const": "uav"},
                            "role": {"const": "mission_uav"},
                            "category": {"const": "uav"},
                            "asset_id": {"enum": ["uav.inspect.quad.v1",
                                                  "uav.airsim.flying_pawn.v1"]},
                        },
                    },
                    {
                        "properties": {
                            "label_class": {"const": "vehicle"},
                            "role": {"const": "semantic_vehicle"},
                            "category": {"const": "vehicle"},
                            "asset_id": {"enum": ["vehicle.emergency.ambulance.v1",
                                                  "vehicle.emergency.suv.v1",
                                                  "vehicle.ground.boxcar.v1"]},
                        },
                    },
                ],
            },
        ],
    }
