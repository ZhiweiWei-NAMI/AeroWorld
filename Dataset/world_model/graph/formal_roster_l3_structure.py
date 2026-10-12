"""Closed whole-JSON grammar for L3 formal entity rosters.

``convert_to_render_ready.py`` writes one JSON object with only ``entities``.
``structure_sources`` reads that root before projecting ``entities[]`` for the
field catalog.  The converter branches shared with L4 reuse fresh, pure
entity schemas from ``formal_roster_l4_structure``.  The five L3 authored
scene profiles add roadwork props, no-fly zones, hazmat anchors, an authored
ambulance and pedestrians.  The converter's service-facility repair adds a
paired placement/repair block to a global-flow pad, which the shared pad row
already accepts optionally; no separate repaired-pad branch is registered.

This module does not read sources or compile a grammar at import time.
"""

from __future__ import annotations

from Dataset.world_model.graph import formal_roster_l4_structure as _shared
from Dataset.world_model.graph.structure_fragments import (
    nonblank_python_string_rule,
    runtime_state_source_guard_rule,
)
from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
)

__all__ = ["formal_roster_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


def _scene_fields() -> dict:
    # Preserved authored fields plus the converter's runtime crop stamp.
    return {
        "activation_tick": {"type": "integer", "minimum": 0},
        "role": _shared._identifier(),
        "route_waypoints_enu_m": _shared._waypoints(),
        "runtime_visibility": _shared._runtime_visibility(),
        "semantic_role": _shared._identifier(),
        "state_sequence": {"type": "array", "items": _shared._identifier()},
        "task_id": _shared._identifier(),
    }


_SCENE_REQUIRED = [
    "activation_tick", "category", "mode", "role",
    "route_waypoints_enu_m", "runtime_visibility", "semantic_role",
    "state_sequence", "task_id",
]


def _roadwork_prop_row(asset_id: str, entity_kind: str) -> dict:
    # L3-1_v1 authors barrier, traffic-cone and construction-fence entities
    # with event_script_only spawn policy; the converter preserves that
    # marker and maps each logical asset to its fixed profile.
    return _shared._branch(
        {
            **_scene_fields(),
            "asset_id": {"const": asset_id},
            "logical_asset_id": {"const": asset_id},
            "label_class": {"const": "prop"},
            "entity_category": {"const": "prop"},
            "entity_kind": {"const": entity_kind},
            "entity_type": {"const": entity_kind},
            "category": {"const": "prop"},
            "mode": {"const": "scene_sync"},
            "role": {"const": "semantic_context"},
            "spawn_policy": {"const": "event_script_only"},
        },
        _SCENE_REQUIRED + ["spawn_policy"],
    )


def _no_fly_zone_row(asset_id: str) -> dict:
    # L3-2_v1/v2 temporary zone and L3-3_v1/v2 hazmat zone have the same
    # closed row shape but distinct authored assets.  The converter retains
    # their initial incident_state and facility_scope contract.
    incident = _shared._runtime_state_family("incident_state")
    incident.update(runtime_state_source_guard_rule("engine_whole"))
    for name in RUNTIME_STATE_METADATA_FIELDS:
        incident["properties"][name] = nonblank_python_string_rule()
    scope = _shared._semantic_scope()
    del scope["properties"]["service_capacity"]
    return _shared._branch(
        {
            **_scene_fields(),
            "asset_id": {"const": asset_id},
            "logical_asset_id": {"const": asset_id},
            "label_class": {"const": "no_fly_zone"},
            "entity_category": {"const": "facility"},
            "entity_kind": {"const": "facility.no_fly_zone"},
            "entity_type": {"const": "facility.no_fly_zone"},
            "category": {"const": "facility"},
            "incident_state": incident,
            "mode": {"const": "metadata_only"},
            "role": {"const": "semantic_context"},
            "semantic_scope": scope,
            "spawn_policy": {"const": "event_script_only"},
        },
        _SCENE_REQUIRED + [
            "incident_state", "semantic_scope", "spawn_policy",
        ],
    )


def _hazmat_anchor_row(category: str, semantic_role: str) -> dict:
    # Both L3-3 versions author semantic.spawn_zone for the ambulance arrival
    # and pedestrian safe-zone anchors.  logical_only_profile preserves the
    # authored category and derives its entity kind from label no_fly_zone.
    return _shared._branch(
        {
            **_scene_fields(),
            "asset_id": {"const": "semantic.spawn_zone"},
            "logical_asset_id": {"const": "semantic.spawn_zone"},
            "label_class": {"const": "no_fly_zone"},
            "entity_category": {"const": category},
            "entity_kind": {"const": f"{category}.no_fly_zone"},
            "entity_type": {"const": f"{category}.no_fly_zone"},
            "proxy_template_id": {"const": ""},
            "category": {"const": category},
            "mode": {"const": "logical_contract"},
            "role": {"const": "semantic_context"},
            "semantic_role": {"const": semantic_role},
        },
        _SCENE_REQUIRED,
    )


def _hazmat_ambulance_row() -> dict:
    # The L3-3 scene entries dispatch this ambulance via the event script,
    # with no event_actor_motion_contract.  The converter preserves the
    # source fields instead of fabricating that contract.
    return _shared._branch(
        {
            **_scene_fields(),
            "asset_id": {"const": "vehicle.emergency.ambulance.v1"},
            "logical_asset_id": {"const": "vehicle.emergency.ambulance.v1"},
            "label_class": {"const": "vehicle"},
            "entity_category": {"const": "vehicle"},
            "entity_kind": {"const": "vehicle.car"},
            "entity_type": {"const": "vehicle.car"},
            "proxy_template_id": {"const": "vehicle.sedan"},
            "category": {"const": "vehicle"},
            "mode": {"const": "scene_sync"},
            "role": {"const": "semantic_vehicle"},
            "spawn_policy": {"const": "event_script_only"},
        },
        _SCENE_REQUIRED + ["spawn_policy"],
    )


def _hazmat_pedestrian_row() -> dict:
    # The two authored L3-3 pedestrians are source-roster entities with no
    # background_role; the converter's pedestrian profile is deterministic.
    return _shared._branch(
        {
            **_scene_fields(),
            "asset_id": {"const": "pedestrian.cityops.basic.v1"},
            "logical_asset_id": {"const": "pedestrian.cityops.basic.v1"},
            "label_class": {"const": "pedestrian"},
            "entity_category": {"const": "pedestrian"},
            "entity_kind": {"const": "pedestrian.person"},
            "entity_type": {"const": "pedestrian.person"},
            "proxy_template_id": {"const": "human.walker"},
            "category": {"const": "pedestrian"},
            "mode": {"const": "pedestrian_managed"},
            "role": {"const": "semantic_pedestrian"},
        },
        _SCENE_REQUIRED,
    )


def _guard_runtime_states(branch: dict) -> None:
    if "oneOf" in branch:
        for variant in branch["oneOf"]:
            _guard_runtime_states(variant)
        return
    for family in RUNTIME_STATE_FIELDS:
        rule = branch["properties"].get(family)
        if rule is not None:
            rule.update(runtime_state_source_guard_rule("engine_whole"))
            for name in RUNTIME_STATE_METADATA_FIELDS:
                if name in rule["properties"]:
                    rule["properties"][name] = nonblank_python_string_rule()


def _l3_event_actor_vehicle_row() -> dict:
    # L3 scene production uses event_actor_entry.  The shared L4 factory
    # also declares dispatch, which only the L2/L4 writers can produce.
    shared = _shared._event_actor_vehicle_row()
    (entry,) = (
        branch for branch in shared["oneOf"]
        if branch["properties"]["event_actor_motion_contract"]
        ["properties"]["policy"]["const"]
        == "offstage_then_continuous_lane_entry"
    )
    return entry


def _entity_schema() -> dict:
    # Shared branches are selected by the L3 scene/converter production
    # paths; L4-only facade-anchor, ordinary-pedestrian and extra corridor
    # branches are not silently carried into this level.
    shared_factories = (
        _shared._sumo_vehicle_row,
        _l3_event_actor_vehicle_row,
        _shared._global_flow_uav_row,
        _shared._global_pad_row,
        _shared._inspect_uav_row,
        _shared._mission_uav_row,
        _shared._observer_uav_row,
        _shared._semantic_facility_row,
        _shared._contract_facility_row,
        _shared._repaired_facility_row,
        _shared._pedestrian_row,
        _shared._corridor_row,
    )
    branches = [factory() for factory in shared_factories]
    branches.extend((
        _roadwork_prop_row("prop.roadwork.barrier.v1", "prop.roadwork_barrier"),
        _roadwork_prop_row("prop.roadwork.traffic_cone.v1", "prop.traffic_cone"),
        _roadwork_prop_row("prop.roadwork.construction_fence.v1",
                           "prop.construction_fence"),
        _no_fly_zone_row("trigger.no_fly.box.v1"),
        _no_fly_zone_row("trigger.hazard.generic.box.v1"),
        _hazmat_anchor_row("vehicle_anchor", "hazmat_arrival_reference"),
        _hazmat_anchor_row("crowd_anchor", "safe_zone_reference"),
        _hazmat_ambulance_row(),
        _hazmat_pedestrian_row(),
    ))
    for branch in branches:
        _guard_runtime_states(branch)
    return {"oneOf": branches}


def formal_roster_schema() -> dict:
    """Return the closed whole-JSON L3 formal_roster schema."""
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_roster_l3.record.schema.json",
        "title": "AERO_WORLD formal_roster L3 JSON root record schema",
        "type": "object", "additionalProperties": False,
        "required": ["entities"],
        "properties": {
            "entities": {"type": "array", "items": _entity_schema()},
        },
    }
