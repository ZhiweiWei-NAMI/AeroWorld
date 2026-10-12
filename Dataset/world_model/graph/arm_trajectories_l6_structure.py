"""Closed record structure for L6 ARM raw trajectory rows.

Source family ``arm_trajectories`` (source_family == effective_family; one
JSONL row per record); profiles L6-1_v1, L6-1_v2, L6-2_v1, L6-2_v2, L6-3_v1,
L6-3_v2, L6-4_v1, L6-4_v2, L6-5_v1, L6-5_v2.  Producer:
``Dataset/tools/l6_v2/arm_runtime.py::run_arm`` replays the profile's scene
and event script on the shipped engine (``Dataset/tools/l6_v2/engine.py``
wrapping ``EpisodeStateEngine`` from ``Dataset/tools/batch_generate.py``),
filters the engine's ``trajectory_rows`` to the declared scope
(``Dataset/tools/l6_v2/scope.py::select_scope``) and to
``[window_start_tick, window_end_tick]``, and writes them verbatim through
``write_rows(out / 'raw/trajectories.jsonl', trajectories)``
(arm_runtime.py:265).  ``raw/states.jsonl`` is written from the same rows
through a different template (arm_runtime.py:266-269) and is the separate
``arm_states`` family closed by ``arm_states_l6_structure``; the two files are
not byte-identical, so no cross-family reuse delegation applies.

Declared inventory (recount at generation time): 168 arm branches across the
ten profiles, 168 ``raw/trajectories.jsonl`` present, 0 missing, structure
status ``field_scan_defined``.  Each arm's ``manifest.json`` carries schema
``aero_l6_v2_arm_manifest``; every L6 dedicated entry under
``scripts/arms/L6-*/`` imports ``Dataset.tools.l6_v2.arm_runtime``, so this
runtime is the single writer anchor for all ten profiles.

Five closed record branches share the engine row of
``EpisodeStateEngine._row_for_entity`` (batch_generate.py:2395): fixed base
keys ``tick/entity_id/label_class/asset_id/pos_enu/vel_mps/yaw_deg/state``
plus ``row_activity_payload``'s unconditional ``activity_type/
animation_hint/posture/social_state`` (batch_generate.py:1856; the
non-pedestrian stamp ``standing``/``solo``; no L6 scope entity is a
pedestrian: the profile scopes are mission UAVs, the home pads, the base
station, the L6-2 contract facility and the L6-5 ground-control anchor, whose
asset catalog labels are ``uav``/``landing_pad``/``radio_tower``/
``asset_anchor``), then ``preserved_fields_from(entity)``
(batch_generate.py:2414), which copies a field only when its value is neither
``None`` nor ``""``.  The base twelve and the preserved fields present on
every fixed L6 scope entity are required in their matching branch.

Preserved-slot closure: ``build_entities`` stores the scene entity under
``scene_setup`` and spreads ``preserved_fields_from(scene_entity)``, so the
row space closes to what the ten L6 scene setups
(``Dataset/scenarios/L6_digital_layer/failure/<profile>/scene_setup.json``)
author on the scope entities plus the runtime families a ``set_runtime_state``
patch deep-merges onto them (``_apply_pending_runtime_state_patches`` +
``deep_merge_mapping``).  Executed over the ten profile scenes with the
producer's own functions, the union row-key universe is exactly the property
set declared below: ``activation_tick/category/entity_kind/role/
state_sequence/semantic_role/task_id/route_waypoints_enu_m/
semantic_scope/assigned_altitude_m/ground_reference_z_m/
ground_reference_source/lifecycle/uav_corridor/uav_corridor_role/
planned_route_waypoints_enu_m/path_deviation_contract/contract_facility``
plus runtime-state families.  The fixed ten scene builders yield 35 scoped
entities in five identity and keyset families.  Non-preserved scene keys the
scope entities carry
(``motion_contract``, ``scene_occupancy``, ``validation_tags``,
``ground_flow_contract``, ``background_vehicle``, ``background_pedestrian``,
``contract_inspect_uav``, ``observer_lifecycle``, ``boundary_crossing_injected``,
``query_tags``, ``enabled``) are not in ``PRESERVED_ENTITY_FIELDS`` and
are not declared here.

Runtime-state families: ``l6_v2/engine.py`` dispatches
``set_runtime_state`` to the shared ``EpisodeStateEngine`` handler.  The
handler accepts all ten ``RUNTIME_STATE_FIELDS`` from
``Dataset/tools/runtime_state_contract.py``, merges an accepted patch into
the entity, and ``_row_for_entity`` preserves that family in the raw JSONL
row.  ``l6_v2/common.py::RUNTIME_FAMILIES`` lists five fields for L6
comparison; it does not restrict the raw-row writer.  All ten family slots
are therefore declared as optional closed objects on each identity branch.
Their consumed keys and values follow ``validate_runtime_state_patch`` and
``invalid_runtime_state_value_paths``: nonblank metadata strings, booleans,
enum fields as strings under ``value.strip().lower()`` membership, and
command fields as one string or a nonempty flat string array.  The shared
source guard checks each actual family subtree with the engine's
``engine_whole`` policy.  Six checked trajectory files contain instances of
``control_state``, ``communication_state``, ``navigation_state`` and
``security_state``; ``mission_state``, ``sensor_state``, ``facility_state``,
``incident_state``, ``vehicle_state`` and ``pedestrian_state`` are declared
from the producer with no instance in that bounded check.

Nested structures: ``semantic_scope`` is the scene facility_scope contract
with ``authority_id`` hard-wired to ``facility_scope_contract.v1``;
``service_capacity`` rides on the scoped landing pads and is absent on the
base-station and ground-control scopes.  ``uav_corridor`` carries the scene's
three authored altitude layers.  ``lifecycle`` is the eight-key engine
auto-lifecycle object.  ``path_deviation_contract`` is the L6-3 scene's
two-key authoring copied verbatim onto the GNSS-anomaly UAV rows.
``contract_facility`` is the L6-2_v1 scene's two-key event-contract stamp
copied verbatim onto its contract facility rows.  ``route_waypoints_enu_m``
and ``planned_route_waypoints_enu_m`` are producer pass-throughs of the
authored arrays; empty lists are a declared status (the facilities author
``[]`` and ``preserved_fields_from`` copies them).  Waypoint members are the
scene's positional x/y/z triples: on L6-3 the planned route's z component is
authored int, while x/y stay float everywhere
and the runtime route stays float on every profile.  ``state_sequence`` is
the authored single-label pass-through.  ``activation_tick`` is the authored
scene value (0 on every L6 scope entity).  Motion semantics, units, event
truth and trajectory authenticity are later-stage concerns and are not
encoded here.

No L6 trajectory row carries an integrity, digest, or hash field, so no
publication-exclusion path is required.  Registry wiring into
``structure_closure.py`` / ``validate_source_record`` belongs to the
integration stage; this module only exports the pure schema constructor and
scans no data at import.
"""

from __future__ import annotations

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
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

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

# row_activity_payload stamps these two verbatim for every non-pedestrian
# row; the L6 scope carries no pedestrian entity.
_POSTURE = "standing"
_SOCIAL_STATE = "solo"

# l6_v2/engine.py dispatches set_runtime_state directly to the shared
# EpisodeStateEngine.  That handler accepts RUNTIME_STATE_FIELDS, not the
# smaller l6_v2/common.py comparison-field list.
_CARRIED_FAMILIES = tuple(RUNTIME_STATE_FIELDS)

# Entity pairing actually written on the L6 scope.  ``build_entities``
# derives ``label_class = label_class_for(entity_id, logical_asset_id,
# category)`` (batch_generate.py:1966), which is
# ``label_for_declared_asset(logical_asset_id)`` over
# Config/LowAltitude/asset_catalog.json (batch_generate.py:1521-1524 ->
# typed_entity_labels.py:24-30) and stores ``asset_id = logical_asset_id``
# (batch_generate.py:2000); ``_row_for_entity`` writes both slots verbatim
# (batch_generate.py:2406-2407).  ``role`` is the scene-authored
# ``initial_state.role``/``role`` preserved verbatim, so the pairing is
# author-determined per scope entity.  The ten fixed scene builders and
# select_scope yield these five combinations across 35 scoped entities.
# The fourth member is the authored entity_kind.  UAV scene entries do not
# author a nonempty kind, so preserved_fields_from never writes that key.
_L6_ENTITY_PAIRINGS = (
    ("asset_anchor", "semantic.asset_anchor", "semantic_context",
     "ground_station.ground_control_station"),
    ("landing_pad", "facility.landing_pad.visible.v1", "semantic_facility",
     "facility.landing_pad"),
    ("radio_tower", "facility.radio.base_tower.v1", "semantic_facility",
     "facility.base_station"),
    ("uav", "uav.airsim.flying_pawn.v1", "mission_uav", None),
    ("uav", "uav.inspect.quad.v1", "mission_uav", None),
)


def _identity_branches() -> list[dict]:
    """Five closed root branches for the fixed L6 scope builders.

    Each branch declares its whole keyset.  The engine can add any of its
    runtime-state families to a scoped entity through a validated
    ``set_runtime_state`` action, so those family slots remain optional on
    every branch.  Other preserved fields follow the ten authored scene
    definitions and the producer's ``preserved_fields_from`` behavior.
    """
    common_required = [
        "tick", "entity_id", "label_class", "asset_id", "pos_enu",
        "vel_mps", "yaw_deg", "state", "activity_type", "animation_hint",
        "posture", "social_state", "role", "category",
        "route_waypoints_enu_m", "semantic_role", "state_sequence",
        "task_id", "activation_tick",
    ]
    uav_required = [
        "assigned_altitude_m", "ground_reference_source",
        "ground_reference_z_m", "lifecycle",
        "planned_route_waypoints_enu_m", "uav_corridor",
        "uav_corridor_role",
    ]
    branches = []
    for label_class, asset_id, role, entity_kind in _L6_ENTITY_PAIRINGS:
        allowed = set(common_required) | set(_CARRIED_FAMILIES)
        required = list(common_required)
        if entity_kind is None:
            allowed.update(uav_required)
            required.extend(uav_required)
            if asset_id == "uav.inspect.quad.v1":
                allowed.add("path_deviation_contract")
        else:
            allowed.update(("entity_kind", "semantic_scope"))
            required.extend(("entity_kind", "semantic_scope"))
            if label_class == "radio_tower":
                allowed.add("contract_facility")
        source_properties = _row_properties()
        properties = {name: source_properties[name] for name in sorted(allowed)}
        properties.update({
            "asset_id": {"const": asset_id},
            "label_class": {"const": label_class},
            "role": {"const": role},
            "category": {"const": "uav" if entity_kind is None else "facility"},
        })
        if entity_kind is not None:
            properties["entity_kind"] = {"const": entity_kind}
            scope = _semantic_scope()
            if label_class == "landing_pad":
                scope["required"].append("service_capacity")
            else:
                del scope["properties"]["service_capacity"]
            properties["semantic_scope"] = scope
        branches.append({
            "type": "object", "additionalProperties": False,
            "required": required, "properties": properties,
        })
    return branches


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _float3() -> dict:
    # ``_row_for_entity`` writes ``round(float(value), 6)`` per component
    # (batch_generate.py:2408-2409), so every recorded member is a float
    # source value; the JSON type stays ``number``.
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number", "x-source-value-types": ["float"]},
    }


def _waypoints() -> dict:
    # Producer pass-through of the authored runtime route; empty lists are a
    # declared structural status for this family.  Members are the scene's
    # positional x/y/z triples, all float on the L6 rows.
    return {"type": "array", "items": _float3()}


def _planned_route_waypoints() -> dict:
    # Same producer pass-through; the L6-3 scenes author the planned route's
    # z component as int while x/y stay float, so z keeps the default
    # number source types.
    member = {
        "type": "array", "minItems": 3, "maxItems": 3,
        "prefixItems": [
            {"type": "number", "x-source-value-types": ["float"]},
            {"type": "number", "x-source-value-types": ["float"]},
            {"type": "number"},
        ],
        "items": False,
    }
    return {"type": "array", "items": member}


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state patch field, from the executable
    producer contract (``validate_runtime_state_patch`` plus
    ``invalid_runtime_state_value_paths`` in runtime_state_contract.py)."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return nonblank_python_string_rule()
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        # Commands validate every member against the family enum after
        # ``strip().lower()`` (runtime_state_contract.py:557-578).
        member = runtime_enum_string_rule(_command_enum(family))
        return {"oneOf": [
            member,
            {"type": "array", "minItems": 1, "items": member},
        ]}
    if family == "facility_state" and field == "requester_ids":
        # The producer accepts an empty list; only non-string members fail.
        return {"type": "array", "items": _identifier()}
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
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _command_enum(family: str) -> frozenset[str]:
    # Command fields validate their members against the family enum after
    # ``strip().lower()`` (runtime_state_contract.py:569-578); the mapping is
    # hard-wired in the contract, so mirror it here.
    return {
        "control_state": _CONTROL_ENUMS,
        "mission_state": _MISSION_ENUMS,
        "security_state": _SECURITY_ENUMS,
    }[family]


def _runtime_state_family_patch(family: str) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family]
                    | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object", "additionalProperties": False, "minProperties": 1,
        **runtime_state_source_guard_rule("engine_whole"),
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
    }


def _runtime_state_families() -> dict:
    """Closed rule per carried family; deep-merge patches may create or
    extend any of them, so each stays an optional row slot.  Six families
    have no instance in the six-file check."""
    return {family: _runtime_state_family_patch(family)
            for family in _CARRIED_FAMILIES}


def _lifecycle() -> dict:
    # Eight-key object built from the scene lifecycle contract and the engine
    # auto-lifecycle stamps; carried by every scoped mission UAV.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "assigned_altitude_m", "auto_lifecycle_applied",
            "home_hover_enu_m", "home_pad_entity_id", "mission_start_enu_m",
            "requires_landing_or_terminal_resolution", "requires_takeoff",
            "waypoint_repair_applied",
        ],
        "properties": {
            "assigned_altitude_m": _float_leaf(),
            "auto_lifecycle_applied": {"type": "boolean"},
            "home_hover_enu_m": _float3(),
            "home_pad_entity_id": _identifier(),
            "mission_start_enu_m": _float3(),
            "requires_landing_or_terminal_resolution": {"type": "boolean"},
            "requires_takeoff": {"type": "boolean"},
            "waypoint_repair_applied": {"type": "boolean"},
        },
    }


def _uav_corridor() -> dict:
    # Engine carries the scene's uav_corridor verbatim; the L6 scenes author
    # exactly three altitude layers.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["altitude_layers_m", "assigned_altitude_m",
                     "used_lateral_bypass"],
        "properties": {
            "altitude_layers_m": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": _float_leaf(),
            },
            "assigned_altitude_m": _float_leaf(),
            "used_lateral_bypass": {"type": "boolean"},
        },
    }


def _semantic_scope() -> dict:
    # Scene facility_scope contract copied verbatim; service_capacity rides
    # on the scoped landing pads and is absent on the base-station and
    # ground-control scopes, so it stays optional.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["authority_id", "ontology_class_id", "scope_subtype",
                     "scope_type"],
        "properties": {
            "authority_id": {"const": "facility_scope_contract.v1"},
            "ontology_class_id": _identifier(),
            "scope_subtype": _identifier(),
            "scope_type": _identifier(),
            "service_capacity": {"type": "integer", "minimum": 0},
        },
    }


def _path_deviation_contract() -> dict:
    # L6-3 scene authoring copied verbatim onto the GNSS-anomaly UAV rows.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["minimum_route_distance_m", "predicate_id"],
        "properties": {
            "minimum_route_distance_m": _float_leaf(),
            "predicate_id": _identifier(),
        },
    }


def _contract_facility() -> dict:
    # L6-2_v1 scene authoring copied verbatim onto its contract facility
    # rows; the policy stamp is a producer-visible authored constant.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["contract_scenario_id", "policy"],
        "properties": {
            "contract_scenario_id": _identifier(),
            "policy": {"const": "semantic_event_contract_v1"},
        },
    }


def _row_properties() -> dict:
    families = _runtime_state_families()
    return {
        # _row_for_entity base keys plus the unconditional activity payload
        # (batch_generate.py:2395-2413); written on every row.  The identity
        # quintuple keys are declared here with their row shapes and their
        # per-asset pairing is pinned through the root oneOf branches.
        "tick": {"type": "integer", "minimum": 0},
        "entity_id": _identifier(),
        "label_class": _identifier(),
        "asset_id": _identifier(),
        "pos_enu": _float3(),
        "vel_mps": _float3(),
        "yaw_deg": _float_leaf(),
        "state": _identifier(),
        "activity_type": _identifier(),
        "animation_hint": _identifier(),
        "posture": {"const": _POSTURE},
        "social_state": {"const": _SOCIAL_STATE},
        "role": _identifier(),
        "category": _identifier(),
        "entity_kind": _identifier(),
        # Preserved slots: written only when the scene value is neither None
        # nor "" (preserved_fields_from).  role/category and the slots listed
        # in each branch's ``required`` are carried by every corresponding
        # scoped scene entity.  The engine drops a slot for a None/""
        # authored value.  entity_kind is required on facility branches and
        # absent from the UAV branches by their closed keysets.
        "activation_tick": {"type": "integer", "minimum": 0},
        "assigned_altitude_m": _float_leaf(),
        "ground_reference_source": _identifier(),
        "ground_reference_z_m": _float_leaf(),
        "lifecycle": _lifecycle(),
        "planned_route_waypoints_enu_m": _planned_route_waypoints(),
        "route_waypoints_enu_m": _waypoints(),
        "semantic_role": _identifier(),
        "semantic_scope": _semantic_scope(),
        "state_sequence": {"type": "array", "minItems": 1,
                           "items": _identifier()},
        "task_id": _identifier(),
        "uav_corridor": _uav_corridor(),
        "uav_corridor_role": _identifier(),
        "path_deviation_contract": _path_deviation_contract(),
        "contract_facility": _contract_facility(),
        # Runtime families preserved_fields_from deep-copies from the scene
        # entity or deep-merges from a set_runtime_state patch.
        **families,
    }


def arm_trajectories_schema() -> dict:
    """Return the closed L6 raw trajectory record schema.

    The ``EpisodeStateEngine._row_for_entity`` base shape
    ``l6_v2/arm_runtime.run_arm`` writes for both factual and intervention
    arms of the ten declared profiles.  Required are the twelve always-written
    base keys plus the seven preserved slots every scope entity carries on
    every row of every L6 file (``role``, ``category``,
    ``route_waypoints_enu_m``, ``semantic_role``, ``state_sequence``,
    ``task_id``, ``activation_tick``).  The five root branches declare
    exact identity, facility and UAV keysets from the fixed L6 scope
    builders.  Runtime-family slots stay optional because a validated
    action can patch them on a scoped entity.  All ten runtime families
    accepted by the shared engine are declared; six have no instance in
    the six-file check.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_trajectories_l6.record.schema.json",
        "title": "L6 ARM raw trajectory record",
        "oneOf": _identity_branches(),
    }
