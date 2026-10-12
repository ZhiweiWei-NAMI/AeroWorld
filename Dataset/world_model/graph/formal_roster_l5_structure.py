"""Closed whole-JSON grammar for L5 formal entity rosters.

The converter and capture writer preserve the ``{"entities": [...]}`` root.
L5 uses the shared SUMO, UAV, facility, pedestrian and corridor producers.
Its authored weather scenes add an entry-stage car and a delivery-bag prop.
Schema construction reads no source files.
"""

from __future__ import annotations

from Dataset.world_model.graph import formal_roster_l4_structure as _shared_l4
from Dataset.world_model.graph import formal_roster_l6_structure as _shared_l6

__all__ = ["formal_roster_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


def _l5_entry_actor_row() -> dict:
    # L5-1 calls event_actor_entry; its scene does not call the dispatch
    # producer represented by the other arm of the L4 factory.
    (entry,) = (
        row for row in _shared_l4._event_actor_vehicle_row()["oneOf"]
        if row["properties"]["event_actor_motion_contract"]
        ["properties"]["policy"]["const"]
        == "offstage_then_continuous_lane_entry"
    )
    entry["properties"].update(_shared_l6._runtime_state_properties())
    return entry


def _repaired_global_pad_row() -> dict:
    # The common service-facility repair can re-anchor a global pad.
    row = _shared_l6._global_pad_row()
    row["properties"]["placement"] = _shared_l6._placement()
    row["properties"]["scene_occupancy_repair"] = (
        _shared_l6._scene_occupancy_repair()
    )
    row["required"].extend(("placement", "scene_occupancy_repair"))
    return row


def _l5_facility_row(factory) -> dict:
    row = factory()
    row["properties"]["mode"] = {"const": "scene_sync"}
    return row


def _l5_mission_uav_row() -> dict:
    row = _shared_l6._mission_uav_row()
    # Only L2/L6 author path-deviation contracts; L5 weather scenes do not.
    del row["properties"]["path_deviation_contract"]
    return row


def _delivery_bag_row() -> dict:
    # L5-3 authors a service delivery bag through world_entity.  The
    # converter's asset profile fixes its identity and scene mode.
    asset = "prop.service.delivery_bag.v1"
    properties = {
        **_shared_l6._runtime_state_properties(),
        "activation_tick": {"type": "integer", "minimum": 0},
        "asset_id": {"const": asset},
        "logical_asset_id": {"const": asset},
        "label_class": {"const": "prop"},
        "entity_category": {"const": "prop"},
        "entity_kind": {"const": "prop.delivery_bag"},
        "entity_type": {"const": "prop.delivery_bag"},
        "proxy_template_id": {"const": asset},
        "category": {"const": "prop"},
        "mode": {"const": "scene_sync"},
        "role": {"const": "semantic_context"},
        "route_waypoints_enu_m": _shared_l4._waypoints(),
        "runtime_visibility": _shared_l4._runtime_visibility(),
        "semantic_role": {"const": "attached"},
        "state_sequence": {
            "type": "array", "items": _shared_l4._identifier(),
        },
        "task_id": _shared_l4._identifier(),
    }
    return _shared_l4._branch(
        properties,
        ["activation_tick", "category", "mode", "role",
         "route_waypoints_enu_m", "runtime_visibility", "semantic_role",
         "state_sequence", "task_id"],
    )


def _entity_schema() -> dict:
    sumo = _shared_l4._sumo_vehicle_row()
    sumo["properties"].update(_shared_l6._runtime_state_properties())
    ordinary_pedestrian = _shared_l4._ordinary_pedestrian_row()
    ordinary_pedestrian["properties"].update(
        _shared_l6._runtime_state_properties()
    )
    return {"oneOf": [
        sumo,
        _l5_entry_actor_row(),
        _shared_l6._global_flow_uav_row(),
        _shared_l6._global_pad_row(),
        _repaired_global_pad_row(),
        _shared_l6._inspect_uav_row(),
        _l5_mission_uav_row(),
        _shared_l6._observer_uav_row(),
        _l5_facility_row(_shared_l6._semantic_facility_row),
        _l5_facility_row(_shared_l6._contract_facility_row),
        _l5_facility_row(_shared_l6._repaired_facility_row),
        _shared_l6._pedestrian_row(),
        ordinary_pedestrian,
        _shared_l6._corridor_row(),
        _delivery_bag_row(),
    ]}


def formal_roster_schema() -> dict:
    """Return the closed L5 formal_roster JSON-root schema."""
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_roster_l5.record.schema.json",
        "title": "AERO_WORLD formal_roster L5 JSON root record schema",
        "type": "object", "additionalProperties": False,
        "required": ["entities"],
        "properties": {
            "entities": {"type": "array", "minItems": 1,
                         "items": _entity_schema()},
        },
    }
