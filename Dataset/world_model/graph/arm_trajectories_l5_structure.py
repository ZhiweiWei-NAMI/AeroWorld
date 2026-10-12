"""Closed record structure for L5 ARM raw trajectory rows.

Source family ``arm_trajectories`` (source_family == effective_family; one
JSONL row per record); profiles L5-1_v1, L5-1_v2, L5-1_v3, L5-2_v1, L5-2_v2,
L5-3_v1, L5-3_v2, all carrying ``grp_l5_env_branch_v1`` arm groups (36
declared arm configs under ``interventions/L5-*/seed*/`` — 21
``arm_factual`` + 15 ``arm_time_delay`` — 36 present ``raw/trajectories.jsonl``
files, 0 missing, declared ``structure_status`` ``field_scan_defined``;
21238 present rows).  Producer: ``Dataset/tools/l5_arm_pipeline.py::
execute_config`` resolves one ``trajectories`` list from
``Dataset/tools/l5_arm_engine.py::execute`` — ``EpisodeStateEngine.
_row_for_entity`` rows (``Dataset/tools/batch_generate.py``) of the
configured ``entity_ids`` inside ``[start_tick, end_tick]``
(l5_arm_pipeline.py:109) — and writes that list verbatim to
``raw/trajectories.jsonl`` (``dump_rows``, compact ``ensure_ascii=False``
JSON; l5_arm_pipeline.py:124).

Two closed record branches, split by the producer's declared asset
identity.  ``_row_for_entity`` writes ``entity["label_class"]`` and
``entity["asset_id"]`` verbatim (batch_generate.py:2401-2402);
``build_entities`` sets both from the scene's ``logical_asset_id`` through
``label_class_for`` -> ``label_for_declared_asset``
(Config/LowAltitude/asset_catalog.json, batch_generate.py:1966,
typed_entity_labels.py), so the label/asset pairing closes to the two
declared catalog rows and the cross pairings are unproducible.  Which
branch a row belongs to is therefore a producer fact, not a spelling
convention (``label_class_for`` itself states that identity spelling does
not override the declared label).

- UAV-carrier branch: the one configured entity per L5-1/L5-2 config and
  the UAV of each L5-3 config (``uav_weather_l5_<n>_v<k>``, asset
  ``uav.inspect.quad.v1`` -> ``label_class`` ``uav``).  16606 rows.
- Delivery-bag attachment branch: the attached payload of each L5-3 config
  (``payload_bag_l5_<n>_v<k>``, asset ``prop.service.delivery_bag.v1`` ->
  ``label_class`` ``delivery_bag``).  4632 rows.  ``l5_arm_engine.execute``
  rewrites the payload row's ``pos_enu``/``vel_mps`` from the parent sample
  plus the tether vector (l5_arm_engine.py:66-85) — recorded values change,
  the record structure does not.

Both branches write the same twelve always-written base keys
(``_row_for_entity``, batch_generate.py:2395-2413): ``tick`` unconditionally,
``entity_id``/``label_class``/``asset_id`` from the entity,
``pos_enu``/``vel_mps`` as ``round(float(value), 6)`` per component,
``yaw_deg`` from the last-motion yaw, ``state`` from the keyframe sample,
then ``row_activity_payload``'s unconditional
``activity_type``/``animation_hint``/``posture``/``social_state`` — every L5
configured entity is non-pedestrian, so the stamp is the constant
``standing``/``solo`` on all 21238 rows (batch_generate.py:1856-1868).

Conditional slots: ``preserved_fields_from(entity)``
(batch_generate.py:1737) copies a field only when its value is neither
``None`` nor ``""`` — from the entity, or from the entity's nested
``initial_state``/``visual_state`` — so every preserved key is a
conditional slot.  The engine entity carries ``PRESERVED_ENTITY_FIELDS``
copied from the arm's branch scene (``sources.script``/``sources.scene`` of
the intervention config; the L5 entry scripts bind
``design/<group>/<arm>.script.json``/``.scene.json``).  A census of all 36
branch scenes shows the seeded keysets are identical across the seeds of
each profile, so each branch's authorable keyset closes to:

- both configured roles: ``activation_tick`` (entity) and
  ``category``/``role``/``semantic_role``/``state_sequence``/``task_id``
  (``initial_state``) and ``route_waypoints_enu_m`` (entity);
- the UAV carrier additionally: ``uav_corridor_role`` (``initial_state``)
  and ``uav_corridor``/``assigned_altitude_m`` (``initial_state`` of the
  altitude-staging pass, hoisted to the entity)/``ground_reference_z_m``/
  ``ground_reference_source``/``lifecycle``/``planned_route_waypoints_enu_m``
  (entity);
- the delivery-bag carrier: none of the UAV-carrier fields.  Its scene
  authors no ``lifecycle``/``uav_corridor*``/``assigned_altitude_m``/
  ``ground_reference_*``/``planned_route_*`` key, no ``set_runtime_state``
  patch and no scene preservation path creates them on it in the current
  corpus, so the bag branch does not declare them.  Deep-merge patches are
  the one producer path that can add a runtime family to a bag row (below),
  so only the ten family slots stay optional on that branch.

Required are therefore, per branch, the twelve base keys plus that
branch's scene-closed slots; the ten runtime-state families and nothing
else stay optional.  The branches are mutually exclusive through the
required ``asset_id``/``label_class`` consts: a row stripped of its branch
markers no longer validates, and a bag row cannot validate against the
UAV branch (or vice versa) because both consts would have to match at
once.

Runtime-state families: a family is present on a row when the scene
carrier declares it or an event-script ``set_runtime_state`` patch creates
or extends it.  The patch path is keyed by entity name, not by branch (the
L5 entry registers ``set_runtime_state`` among the action handlers,
``l5_arm_engine.py:28-30``; the handler validates the payload and queues it
for the named entity, batch_generate.py:2275-2309;
``EpisodeStateEngine._apply_pending_runtime_state_patches``,
batch_generate.py:2312-2319, deep-merges it there), so a bag row may carry
a family and the family slots stay optional on both branches.  Payloads
are gated by ``validate_runtime_state_patch`` (batch_generate.py:849-892),
which admits only the ten ``RUNTIME_STATE_FIELDS``
members, their consumed fields per ``RUNTIME_STATE_CONSUMED_FIELDS``, the
``rule_id``/``mechanism_id`` metadata strings, and the boolean/enum/command
value rules of ``invalid_runtime_state_value_paths`` — the executable
contract in ``Dataset/tools/runtime_state_contract.py`` that this module
mirrors per family.  L5 sources author only boolean fields of
``control_state``, ``mission_state`` and ``navigation_state``
(``l5_l6_x_runtime_state_schedule.json`` scenarios L5-1_v1..L5-3_v2; every
L5 ``set_runtime_state`` action payload carries the schedule's
``state_patch`` verbatim; 80 patch actions in the corpus, all targeting the
configured UAV), so those three families are closed to the full executable
rules and are instance-verified at their observed key sets
(``control_state.slowdown_active``/``rth_active``,
``mission_state.unsafe``/``mission_abort_active``/``termination_active``,
``navigation_state.route_degraded``/``visual_navigation_degraded``/
``visual_navigation_failed``/``visual_navigation_recovered`` — booleans
with both values where the count allows); the remaining seven families are
producer-allowed without any L5 scene carrier, patch or row —
尚无实例验证, as are the non-boolean contract fields of the three
carried families and the ``rule_id``/``mechanism_id`` metadata strings.
Enum membership and case normalisation stay producer-side; the grammar
keeps the written string.

Tick: the writer keeps engine rows of every configured entity inside the
declared closed window (``start<=tick<=end``, l5_arm_pipeline.py:109) and
the engine samples active entities at every integer tick, so L5 rows cover
each declared window densely (all 36 files: per-entity coverage of every
tick in ``[start_tick, end_tick]``; no L5 writer path fills pre-activation
rows — ``_entity_active_at``, batch_generate.py:2085-2092, drops them).
The declared window is bounded by the L5 config contract
``0 <= start_tick < fork_tick < end_tick <= 900`` on the 5-tick grid
(``load_config``, Dataset/tools/l5_arm_common.py:102-105; ``rules/l5_arm_config.schema.json``
bounds ``window.end_tick`` at 900), so the row ``tick`` closes to
``0 <= tick <= 900``: the declared window maximum is the row maximum.  No
source declares a narrower structural bound (the 36 windows start at
170/180/200 and end at 515..895), so the schema keeps the producer bound
and does not invent a profile-specific one.

Boundary: an L5 window records only rows of the configured ``entity_ids``
(one ``uav_weather_l5_<profile>`` per L5-1/L5-2 profile, that UAV plus its
attached ``payload_bag_l5_<profile>`` for L5-3); background vehicles,
pedestrians, pads, observers and inspect UAVs are authored scene members
but are never configured into the window, and no writer path for them
exists here.  No row of any of the 36 declared sources carries an
integrity or digest field at any path (leaf scan), so no
publication-exclusion path is required for this family.  Registry wiring
into ``structure_closure.py`` and dispatch through ``structure_sources.py``
belong to the integration stage; this module only exports the pure schema
constructor and scans no data at import.
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
# row; all L5 configured entities are uav/prop carriers.
_POSTURE = "standing"
_SOCIAL_STATE = "solo"

# The two declared catalog rows the L5 configured entities carry
# (Config/LowAltitude/asset_catalog.json; label_for_declared_asset maps the
# logical_asset_id to its declared label_class).
_UAV_ASSET_ID = "uav.inspect.quad.v1"
_UAV_LABEL_CLASS = "uav"
_BAG_ASSET_ID = "prop.service.delivery_bag.v1"
_BAG_LABEL_CLASS = "delivery_bag"

_FAMILY_COMMAND_ENUMS = {
    "control_state": _CONTROL_ENUMS,
    "mission_state": _MISSION_ENUMS,
    "security_state": _SECURITY_ENUMS,
}


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _float3() -> dict:
    # ``_row_for_entity`` writes ``round(float(value), 6)`` per component
    # (batch_generate.py:2408-2409) and the tether rewrite recomputes from
    # floats (l5_arm_engine.py:66-85); all 21238 L5 rows carry three float
    # members in pos_enu and vel_mps.
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number", "x-source-value-types": ["float"]},
    }


def _waypoints() -> dict:
    # Producer pass-through of the authored route; the empty list is the
    # declared shape of the L5-3 attached payloads.  Members are the scene's
    # positional x/y/z triples; all 24943 row route members and all 24943
    # planned-route members, plus all 314 scene-authored members, are floats.
    return {"type": "array", "items": _float3()}


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state family field, from the executable
    producer contract (``validate_runtime_state_patch`` plus
    ``invalid_runtime_state_value_paths`` in runtime_state_contract.py)."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return nonblank_python_string_rule()
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        member = runtime_enum_string_rule(_FAMILY_COMMAND_ENUMS[family])
        return {"oneOf": [
            dict(member),
            {"type": "array", "minItems": 1, "items": member},
        ]}
    if family == "facility_state" and field == "requester_ids":
        # The producer accepts an empty list; only non-string members fail.
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
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _runtime_state_family_rule(family: str) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object", "additionalProperties": False, "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
        **runtime_state_source_guard_rule("engine_whole"),
    }


def _runtime_state_families() -> dict:
    """Closed rule per executable family; a deep-merged patch may create or
    extend any of them, so each stays an optional row slot on both
    branches."""
    return {family: _runtime_state_family_rule(family)
            for family in RUNTIME_STATE_FIELDS}


def _lifecycle() -> dict:
    # Eight-key object built from the scene lifecycle contract and the
    # engine's auto-lifecycle stamps (regenerate_boundary_scenarios.py:7232-7237,
    # 8022-8025); identical on all 16606 UAV rows, pose members all floats
    # (49818 row members per vector).
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "assigned_altitude_m", "auto_lifecycle_applied", "home_hover_enu_m",
            "home_pad_entity_id", "mission_start_enu_m",
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
    # Engine carries the scene's uav_corridor verbatim; the L5 scenes author
    # exactly three altitude layers, and all 16606 UAV rows carry three
    # floats (49818 row members).
    return {
        "type": "object", "additionalProperties": False,
        "required": ["altitude_layers_m", "assigned_altitude_m", "used_lateral_bypass"],
        "properties": {
            "altitude_layers_m": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": _float_leaf(),
            },
            "assigned_altitude_m": _float_leaf(),
            "used_lateral_bypass": {"type": "boolean"},
        },
    }


def _base_properties() -> dict:
    """Base keys written on every row (``_row_for_entity`` +
    ``row_activity_payload``), the slots both configured roles author
    (``activation_tick``/``category``/``role``/``semantic_role``/
    ``state_sequence``/``task_id``/``route_waypoints_enu_m`` on all 48
    scene carriers and all 21238 rows), and the ten runtime-state family
    slots: ``_handle_set_runtime_state`` admits a patch for any declared
    family (``validate_runtime_state_patch``, batch_generate.py:849-892,
    checks only the executable family/field/value contract) and
    ``_apply_pending_runtime_state_patches`` deep-merges it into the named
    entity, so every family is an optional slot on both branches."""
    return {
        # _row_for_entity base keys (always written).
        "tick": {"type": "integer", "minimum": 0, "maximum": 900},
        "entity_id": _identifier(),
        "pos_enu": _float3(),
        "vel_mps": _float3(),
        "yaw_deg": _float_leaf(),
        "state": _identifier(),
        "activity_type": _identifier(),
        "animation_hint": _identifier(),
        "posture": {"const": _POSTURE},
        "social_state": {"const": _SOCIAL_STATE},
        # Preserved slots authored on both configured roles; written only
        # when the scene value is neither None nor "" (preserved_fields_from).
        "activation_tick": {"type": "integer", "minimum": 0},
        "category": _identifier(),
        "role": _identifier(),
        "route_waypoints_enu_m": _waypoints(),
        "semantic_role": _identifier(),
        "state_sequence": {"type": "array", "minItems": 1,
                           "items": _identifier()},
        "task_id": _identifier(),
        # Runtime-state family slots (optional on both branches; values
        # closed to the executable producer contract).
        **_runtime_state_families(),
    }


def _uav_branch() -> dict:
    """UAV-carrier branch: the configured mission UAV row.

    Requires the base keys plus the UAV-carrier slots its 36 scene carriers
    author (``uav_corridor_role``/``assigned_altitude_m`` via
    ``initial_state``; ``ground_reference_z_m``/``ground_reference_source``/
    ``lifecycle``/``planned_route_waypoints_enu_m``/``uav_corridor`` via the
    entity); present on all 16606 UAV rows.  The runtime-state family slots
    live in the base properties (optional on both branches); the UAV branch
    adds only the scene-closed carrier slots above.
    """
    properties = {
        **_base_properties(),
        "asset_id": {"const": _UAV_ASSET_ID},
        "label_class": {"const": _UAV_LABEL_CLASS},
        "assigned_altitude_m": _float_leaf(),
        "ground_reference_source": _identifier(),
        "ground_reference_z_m": _float_leaf(),
        "lifecycle": _lifecycle(),
        "planned_route_waypoints_enu_m": _waypoints(),
        "uav_corridor": _uav_corridor(),
        "uav_corridor_role": _identifier(),
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(set(properties) - set(RUNTIME_STATE_FIELDS)),
        "properties": properties,
    }


def _bag_branch() -> dict:
    """Delivery-bag attachment branch: the attached payload row.

    Requires the base keys plus the seven slots its 12 scene carriers
    author; present on all 4632 bag rows.  The UAV-carrier slots
    (``lifecycle``/``uav_corridor*``/``assigned_altitude_m``/
    ``ground_reference_*``/``planned_route_*``) are authored on no bag
    carrier and are not declared.  The ten runtime-family slots come from
    the base properties and stay optional on this branch.
    """
    properties = {
        **_base_properties(),
        "asset_id": {"const": _BAG_ASSET_ID},
        "label_class": {"const": _BAG_LABEL_CLASS},
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(set(properties) - set(RUNTIME_STATE_FIELDS)),
        "properties": properties,
    }


def arm_trajectories_schema() -> dict:
    """Compile the closed L5 raw trajectory record schema.

    Two record branches, ``oneOf``-split by the producer's declared asset
    identity: the UAV-carrier branch (``asset_id``
    ``uav.inspect.quad.v1`` / ``label_class`` ``uav``, required with its
    14 scene-closed slots) and the delivery-bag attachment branch
    (``asset_id`` ``prop.service.delivery_bag.v1`` / ``label_class``
    ``delivery_bag``, required with its 7 scene-closed slots).  The consts
    make the cross pairings unproducible and keep a bag row out of the UAV
    branch and vice versa.  Required are the twelve always-written base
    keys plus each branch's scene-closed slots; optional are only the ten
    runtime-state families (``_handle_set_runtime_state`` admits any
    declared family for the named entity and the engine deep-merges it
    there, so a patch can create a family on either branch).

    Observed per-profile family sets (all 36 declared sources, 21238 rows):
    L5-1_v1/L5-1_v2 UAVs carry ``control_state``/``mission_state``/
    ``navigation_state``; L5-1_v3 carries ``control_state``/
    ``mission_state``; L5-2_v1/L5-2_v2 carry ``mission_state``/
    ``navigation_state``; L5-3_v1 UAV and payload carry none; L5-3_v2 UAV
    carries ``mission_state``.  ``security_state``, ``sensor_state``,
    ``facility_state``, ``communication_state``, ``incident_state``,
    ``vehicle_state`` and ``pedestrian_state`` remain producer-allowed
    without an L5 instance (尚无实例验证), as do the non-boolean contract
    fields of the three L5-carried families and the
    ``rule_id``/``mechanism_id`` metadata strings.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_trajectories_l5.record.schema.json",
        "title": "L5 ARM raw trajectory record",
        # Branch schemas carry their own closed boundaries; the wrapper
        # declares only the type and the two alternatives (a parent-level
        # additionalProperties: False would see no parent annotations and
        # reject every branch key — the same shape
        # arm_trajectories_l1_structure closes its contracts with).
        "type": "object",
        "oneOf": [_uav_branch(), _bag_branch()],
    }
