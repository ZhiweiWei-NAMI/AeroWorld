"""Closed whole-JSON grammar for L2 formal entity rosters.

The converter writes one ``{"entities": [...]}`` root.  Shared SUMO, UAV,
facility, pedestrian and corridor entities use pure factories for the same
producer path.  The L2-5 scene adds an event-dispatched police unit and a
traffic signal; a service-facility repair can also affect a global pad.

Only source-side entity rows receive optional validated runtime-state
families.  Schema construction reads no source files.
"""

from __future__ import annotations

from Dataset.world_model.graph import formal_roster_l4_structure as _shared_l4
from Dataset.world_model.graph import formal_roster_l6_structure as _shared_l6

__all__ = ["formal_roster_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


def _source_runtime_states(row: dict) -> dict:
    # preserved_fields_from copies validated state families from authored
    # source rows; this factory returns a new object on every call.
    row["properties"].update(_shared_l6._runtime_state_properties())
    return row


def _l2_facility_row(factory) -> dict:
    row = factory()
    row["properties"]["mode"] = {"const": "scene_sync"}
    return row


def _event_actor_rows() -> tuple[dict, dict]:
    variants = _shared_l4._event_actor_vehicle_row()["oneOf"]
    (entry,) = (
        row for row in variants
        if row["properties"]["event_actor_motion_contract"]
        ["properties"]["policy"]["const"]
        == "offstage_then_continuous_lane_entry"
    )
    (dispatch,) = (
        row for row in variants
        if row["properties"]["event_actor_motion_contract"]
        ["properties"]["policy"]["const"]
        == "offstage_until_semantic_dispatch"
    )
    _source_runtime_states(entry)
    _source_runtime_states(dispatch)
    # L2-5 dispatches the police unit; its authored incident-state patch
    # survives the converter and the actor is removed after flow recovery.
    dispatch["properties"].update({
        "asset_id": {"const": "vehicle.emergency.police_suv.v1"},
        "logical_asset_id": {"const": "vehicle.emergency.police_suv.v1"},
        "label_class": {"const": "vehicle"},
        "entity_category": {"const": "vehicle"},
        "entity_kind": {"const": "vehicle.car"},
        "entity_type": {"const": "vehicle.car"},
        "proxy_template_id": {"const": "vehicle.sedan"},
        "role": {"const": "semantic_vehicle"},
    })
    dispatch["properties"]["incident_state"]["required"] = [
        "manual_control_active",
    ]
    dispatch["required"].extend(("deactivation_tick", "incident_state"))
    return entry, dispatch


def _repaired_global_pad_row() -> dict:
    row = _shared_l6._global_pad_row()
    row["properties"]["placement"] = _shared_l6._placement()
    row["properties"]["scene_occupancy_repair"] = (
        _shared_l6._scene_occupancy_repair()
    )
    row["required"].extend(("placement", "scene_occupancy_repair"))
    return row


def _traffic_signal_row() -> dict:
    # L2-5's source-side signal has no semantic_scope; the authored
    # facility_state.fault patch is present at the formal roster boundary.
    props = {
        **_shared_l6._runtime_state_properties(),
        "activation_tick": {"type": "integer", "minimum": 0},
        "asset_id": {"const": "prop.traffic_control.signal_light.v1"},
        "logical_asset_id": {"const": "prop.traffic_control.signal_light.v1"},
        "label_class": {"const": "traffic_light"},
        "entity_category": {"const": "traffic_light"},
        "entity_kind": {"const": "traffic_light.signal"},
        "entity_type": {"const": "traffic_light.signal"},
        "proxy_template_id": {"const": "proxy.traffic_light_signal"},
        "category": {"const": "traffic_light"},
        "mode": {"const": "scene_sync"},
        "role": {"const": "semantic_facility"},
        "route_waypoints_enu_m": _shared_l4._waypoints(),
        "runtime_visibility": _shared_l4._runtime_visibility(),
        "semantic_role": {"const": "green_cycle"},
        "state_sequence": {
            "type": "array", "items": _shared_l4._identifier(),
        },
        "task_id": _shared_l4._identifier(),
    }
    props["facility_state"]["required"] = ["fault"]
    return _shared_l4._branch(
        props,
        ["activation_tick", "category", "facility_state", "mode", "role",
         "route_waypoints_enu_m", "runtime_visibility", "semantic_role",
         "state_sequence", "task_id"],
    )


def _entity_schema() -> dict:
    sumo = _shared_l4._sumo_vehicle_row()
    sumo["properties"].update(_shared_l6._runtime_state_properties())
    entry, police = _event_actor_rows()
    facade = _source_runtime_states(_shared_l4._logical_facade_anchor_row())
    return {"oneOf": [
        sumo,
        entry,
        police,
        _shared_l6._global_flow_uav_row(),
        _shared_l6._global_pad_row(),
        _repaired_global_pad_row(),
        _shared_l6._inspect_uav_row(),
        _shared_l6._mission_uav_row(),
        _shared_l6._observer_uav_row(),
        _l2_facility_row(_shared_l6._semantic_facility_row),
        _l2_facility_row(_shared_l6._contract_facility_row),
        _l2_facility_row(_shared_l6._repaired_facility_row),
        _shared_l6._pedestrian_row(),
        _shared_l6._corridor_row(),
        facade,
        _traffic_signal_row(),
    ]}


def formal_roster_schema() -> dict:
    """Return the closed L2 formal_roster JSON-root schema."""
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_roster_l2.record.schema.json",
        "title": "AERO_WORLD formal_roster L2 JSON root record schema",
        "type": "object", "additionalProperties": False,
        "required": ["entities"],
        "properties": {
            "entities": {"type": "array", "minItems": 1,
                         "items": _entity_schema()},
        },
    }
