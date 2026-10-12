"""Closed whole-JSON grammar for X-layer formal entity rosters.

``convert_to_render_ready.py`` writes the roster as ``{"entities": [...]}``;
the capture filter retains the JSON root.  Shared roster builders supply
SUMO traffic, global UAVs and pads, and scene-authored actors.  The X3
scene adds an ambulance, an injured pedestrian and police tape.  X2 and
X6 author no-fly facilities with distinct spawn-policy presence, while X6
also authors three logical crowd anchors.  The constructors read no files.
"""

from __future__ import annotations

from Dataset.world_model.graph import formal_roster_l4_structure as _shared
from Dataset.world_model.graph import formal_roster_l6_structure as _runtime

__all__ = ["formal_roster_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


def _scene_fields() -> dict:
    # The converter preserves authored scene fields and stamps runtime
    # visibility on each retained entity.
    return {
        "activation_tick": {"type": "integer", "minimum": 0},
        "role": _shared._identifier(),
        "route_waypoints_enu_m": _shared._waypoints(),
        "runtime_visibility": _runtime._runtime_visibility(),
        "semantic_role": _shared._identifier(),
        "state_sequence": {"type": "array", "items": _shared._identifier()},
        "task_id": _shared._identifier(),
    }


_SCENE_REQUIRED = [
    "activation_tick", "category", "mode", "role",
    "route_waypoints_enu_m", "runtime_visibility", "semantic_role",
    "state_sequence", "task_id",
]


def _sumo_vehicle_row() -> dict:
    # The SUMO builder may merge a validated source runtime-state patch.
    row = _shared._sumo_vehicle_row()
    row["properties"].update(_runtime._runtime_state_properties())
    return row


def _repaired_global_pad_row() -> dict:
    # The service-facility reanchor pass also reaches global landing pads.
    row = _runtime._global_pad_row()
    row["properties"]["placement"] = _runtime._placement()
    row["properties"]["scene_occupancy_repair"] = (
        _runtime._scene_occupancy_repair()
    )
    row["required"].extend(("placement", "scene_occupancy_repair"))
    return row


def _ordinary_pedestrian_row() -> dict:
    # X1's crowd and X6's evacuees have no background contract and carry no
    # runtime state; X3's scene_setup authors incident_state only on its
    # injured actor.  Dropping the shared optional incident_state keeps this
    # row disjoint from _fall_pedestrian_row under oneOf, since both rows are
    # additionalProperties-false.  Source pedestrian state is still checked by
    # the shared executable-contract fragment.
    row = _shared._ordinary_pedestrian_row()
    row["properties"]["pedestrian_state"] = (
        _runtime._runtime_state_family("pedestrian_state")
    )
    row["properties"].pop("incident_state", None)
    return row


def _fall_pedestrian_row() -> dict:
    # X3 scene_setup authors both state families on its injured actor.
    row = _ordinary_pedestrian_row()
    row["properties"].update({
        "asset_id": {"const": "pedestrian.cityops.basic.v1"},
        "logical_asset_id": {"const": "pedestrian.cityops.basic.v1"},
        "entity_category": {"const": "pedestrian"},
        "entity_kind": {"const": "pedestrian.person"},
        "entity_type": {"const": "pedestrian.person"},
        "label_class": {"const": "pedestrian"},
        "proxy_template_id": {"const": "human.walker"},
        "incident_state": _runtime._runtime_state_family("incident_state"),
    })
    row["required"].extend(("incident_state", "pedestrian_state"))
    return row


def _ambulance_row() -> dict:
    # X3's authored ambulance has no L4 event-actor dispatch contract.
    return _shared._branch(
        {
            **_scene_fields(),
            "asset_id": {"const": "vehicle.emergency.ambulance.v1"},
            "logical_asset_id": {"const": "vehicle.emergency.ambulance.v1"},
            "entity_category": {"const": "vehicle"},
            "entity_kind": {"const": "vehicle.car"},
            "entity_type": {"const": "vehicle.car"},
            "label_class": {"const": "vehicle"},
            "proxy_template_id": {"const": "vehicle.sedan"},
            "category": {"const": "vehicle"},
            "mode": {"const": "scene_sync"},
            "role": {"const": "semantic_vehicle"},
        },
        _SCENE_REQUIRED,
    )


def _police_tape_row() -> dict:
    # X3's scene author marks the prop as event-script-only.
    return _shared._branch(
        {
            **_scene_fields(),
            "asset_id": {"const": "prop.incident.police_tape.v1"},
            "logical_asset_id": {"const": "prop.incident.police_tape.v1"},
            "entity_category": {"const": "prop"},
            "entity_kind": {"const": "prop.police_tape"},
            "entity_type": {"const": "prop.police_tape"},
            "label_class": {"const": "prop"},
            "proxy_template_id": {"const": "prop.incident.police_tape.v1"},
            "category": {"const": "prop"},
            "mode": {"const": "scene_sync"},
            "role": {"const": "semantic_context"},
            "spawn_policy": {"const": "event_script_only"},
        },
        _SCENE_REQUIRED + ["spawn_policy"],
    )


def _crowd_anchor_row() -> dict:
    # X6 scene_setup authors one crowd marker and two destination anchors.
    return _shared._branch(
        {
            **_scene_fields(),
            "asset_id": {"const": "semantic.spawn_zone"},
            "logical_asset_id": {"const": "semantic.spawn_zone"},
            "entity_category": {"const": "crowd_anchor"},
            "entity_kind": {"const": "crowd_anchor.no_fly_zone"},
            "entity_type": {"const": "crowd_anchor.no_fly_zone"},
            "label_class": {"const": "no_fly_zone"},
            "proxy_template_id": {"const": ""},
            "category": {"const": "crowd_anchor"},
            "mode": {"const": "logical_contract"},
            "role": {"const": "semantic_context"},
            "semantic_role": {"enum": [
                "idle", "reroute_exit_reference", "safe_zone_reference",
            ]},
        },
        _SCENE_REQUIRED,
    )


def _no_fly_zone_row(*, event_script_only: bool) -> dict:
    # X2 authors the zone without spawn_policy; X6 explicitly sets it.
    scope = _runtime._semantic_scope()
    scope["properties"].pop("service_capacity", None)
    scope["properties"]["ontology_class_id"] = {"const": "world:NoFlyZone"}
    scope["properties"]["scope_subtype"] = {"const": "no_fly_zone"}
    properties = {
        **_scene_fields(),
        "asset_id": {"const": "trigger.no_fly.box.v1"},
        "logical_asset_id": {"const": "trigger.no_fly.box.v1"},
        "entity_category": {"const": "facility"},
        "entity_kind": {"const": "facility.no_fly_zone"},
        "entity_type": {"const": "facility.no_fly_zone"},
        "label_class": {"const": "no_fly_zone"},
        "proxy_template_id": {"const": "proxy.facility_observation_point"},
        "category": {"const": "facility"},
        "mode": {"const": "metadata_only"},
        "role": {"const": "semantic_context"},
        "semantic_scope": scope,
    }
    required = _SCENE_REQUIRED + ["semantic_scope"]
    if event_script_only:
        properties["spawn_policy"] = {"const": "event_script_only"}
        required.append("spawn_policy")
    return _shared._branch(properties, required)


def _entity_schema() -> dict:
    # Each factory returns a detached schema.  L4-only event actor, facade,
    # and shared-corridor branches and L6-only GCS anchors are not X sources.
    return {"oneOf": [
        _sumo_vehicle_row(),
        _runtime._global_flow_uav_row(),
        _runtime._global_pad_row(),
        _repaired_global_pad_row(),
        _runtime._inspect_uav_row(),
        _runtime._mission_uav_row(),
        _runtime._observer_uav_row(),
        _runtime._semantic_facility_row(),
        _runtime._contract_facility_row(),
        _runtime._repaired_facility_row(),
        _runtime._pedestrian_row(),
        _ordinary_pedestrian_row(),
        _fall_pedestrian_row(),
        _runtime._corridor_row(),
        _ambulance_row(),
        _police_tape_row(),
        _crowd_anchor_row(),
        _no_fly_zone_row(event_script_only=False),
        _no_fly_zone_row(event_script_only=True),
    ]}


def formal_roster_schema() -> dict:
    """Return the closed whole-JSON X-layer formal roster schema."""
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_roster_x.record.schema.json",
        "title": "AERO_WORLD formal_roster X JSON root record schema",
        "type": "object", "additionalProperties": False,
        "required": ["entities"],
        "properties": {
            "entities": {"type": "array", "minItems": 1,
                         "items": _entity_schema()},
        },
    }
