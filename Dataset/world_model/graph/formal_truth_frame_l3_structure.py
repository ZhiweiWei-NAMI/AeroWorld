"""Closed whole-frame grammar for L3 formal truth JSONL rows.

``convert_to_render_ready.py`` writes the same 26-key frame envelope for
every layer, and ``filter_render_ready_truth_for_capture.py`` only crops the
entity list and rebuilds its dependent summaries.  One JSONL line is the
record root: ``formal_truth_frame`` is not an entity-projected source in
``structure_sources.py``.

The frame envelope and the SUMO/global-UAV entity writers are shared with
the L1 producer, so this module calls their pure schema factories and
replaces only layer-dependent branches.  No returned schema object is
cached or mutated across calls.  L3 adds its authored roadwork, no-fly-zone,
hazmat, facility and UAV-lifecycle rows; their nested contracts reuse the
same-layer ``arm_trajectories_l3_structure`` fragments produced from the
L3 scene/engine contract.  Runtime-state objects receive the episode-level
``engine_whole`` source guard used by the converter.

Dynamic maps stay instance-bound.  ``entity_motion_state`` keys are entity
ids, ``sumo_traffic_light_states`` keys are controller ids, and
``roster_summary.by_category`` keys are category tokens; none is expanded
into one property per observed instance.

The bounded L3 truth-frame records carry three runtime families (control,
incident and mission).  The executable runtime contract allows all ten
families, so the other seven and the unobserved value members remain
declared without L3 truth-frame instance verification.  Both presence forms
of service-facility and global-pad occupancy repair occur in the bounded
records.  Nonempty global-UAV ``active_event_ids`` remains producer-backed
without an instance there.  Truth frames contain no integrity/digest/hash
field, so no publication-exclusion path applies.
"""

from __future__ import annotations

from Dataset.tools.runtime_state_contract import RUNTIME_STATE_FIELDS
from Dataset.world_model.graph import arm_trajectories_l3_structure as _l3
from Dataset.world_model.graph import formal_truth_frame_l1_structure as _frame
from Dataset.world_model.graph.structure_fragments import (
    runtime_state_source_guard_rule,
)

__all__ = ["formal_truth_frame_schema"]


_EPISODE_PATTERN = r"^(?:L3-1_v1|L3-2_v[12]|L3-3_v[12])__seed(?:00|01|02)$"
_FRAME_PATTERN = (
    r"^(?:L3-1_v1|L3-2_v[12]|L3-3_v[12])__seed(?:00|01|02)_tick_[0-9]+$"
)

_SOURCE_CORE_REQUIRED = (
    "annotations",
    "entity_category",
    "entity_id",
    "entity_kind",
    "entity_type",
    "label_class",
    "logical_asset_id",
    "proxy_template_id",
    "render_presence",
    "roi_id",
    "runtime_visibility",
    "site_id",
    "state",
    "state_revision",
    "tags",
    "truth_pose",
    "visual_revision",
)


def _source_properties() -> dict:
    """Fresh L3 preserved-field rules for converter source entities."""
    row = _l3._row_properties()
    names = {
        "activation_tick",
        "assigned_altitude_m",
        "background_pedestrian",
        "background_role",
        "category",
        "contract_facility",
        "contract_inspect_uav",
        "control_state",
        "deactivation_tick",
        "event_actor_motion_contract",
        "full_episode_presence",
        "ground_flow_contract",
        "ground_reference_source",
        "ground_reference_z_m",
        "incident_state",
        "lifecycle",
        "min_path_length_m",
        "mission_state",
        "observer_lifecycle",
        "planned_route_waypoints_enu_m",
        "role",
        "route_waypoints_enu_m",
        "semantic_role",
        "semantic_scope",
        "spawn_policy",
        "state_sequence",
        "task_id",
        "uav_corridor",
        "uav_corridor_role",
        *RUNTIME_STATE_FIELDS,
    }
    properties = {name: row[name] for name in names}
    properties.update({
        # These are in convert_to_render_ready.PRESERVED_ENTITY_FIELDS but
        # not in the ARM trajectory writer's narrower L3 row projection.
        "contract_scenario_id": _frame._identifier(),
        "inspect_altitude_code": _frame._identifier(),
        "inspect_altitude_m": _frame._float_leaf(),
        "inspect_fov_coverage_required": {"type": "boolean"},
        "scene_occupancy_repair": _frame._scene_occupancy_repair(),
    })
    for family in RUNTIME_STATE_FIELDS:
        properties[family].update(runtime_state_source_guard_rule("engine_whole"))
    return properties


def _source_branch(
    required_extra: tuple[str, ...],
    *,
    optional_extra: tuple[str, ...] = (),
    uav: bool = False,
    allow_empty_proxy: bool = False,
) -> dict:
    core = _frame._entity_core_properties()
    if not allow_empty_proxy:
        core["proxy_template_id"] = _frame._identifier()
    properties = {name: core[name] for name in _SOURCE_CORE_REQUIRED}
    if uav:
        properties["uav_visibility"] = core["uav_visibility"]

    extra = _source_properties()
    allowed = set(required_extra) | set(optional_extra) | set(RUNTIME_STATE_FIELDS)
    properties.update({name: extra[name] for name in allowed})
    required = [*_SOURCE_CORE_REQUIRED, *required_extra]
    if uav:
        required.append("uav_visibility")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _source_rows() -> list[dict]:
    route_actor = (
        "activation_tick",
        "category",
        "role",
        "route_waypoints_enu_m",
        "semantic_role",
        "state_sequence",
        "task_id",
    )
    mission_uav = (
        "activation_tick",
        "assigned_altitude_m",
        "category",
        "ground_reference_source",
        "ground_reference_z_m",
        "lifecycle",
        "planned_route_waypoints_enu_m",
        "role",
        "route_waypoints_enu_m",
        "semantic_role",
        "state_sequence",
        "task_id",
        "uav_corridor",
        "uav_corridor_role",
    )
    return [
        # Generated airspace-corridor carrier.
        _source_branch(("activation_tick", "category")),
        # Continuous background pedestrian.
        _source_branch((
            "activation_tick",
            "background_pedestrian",
            "background_role",
            "category",
            "contract_scenario_id",
            "ground_flow_contract",
            "role",
            "route_waypoints_enu_m",
            "semantic_role",
            "state_sequence",
            "task_id",
        )),
        # Authored pedestrians plus logical crowd/vehicle anchors.
        _source_branch(route_actor, allow_empty_proxy=True),
        # Contract facilities; the placement repair is conditional.
        _source_branch(
            (*route_actor, "contract_facility", "contract_scenario_id",
             "semantic_scope"),
            optional_extra=("scene_occupancy_repair",),
        ),
        # Roadwork props and the staged L3-3 ambulance.
        _source_branch((*route_actor, "spawn_policy")),
        # Mission landing-pad facility.
        _source_branch((*route_actor, "semantic_scope")),
        # Long-lived inspect UAV.
        _source_branch((
            "activation_tick",
            "assigned_altitude_m",
            "category",
            "contract_inspect_uav",
            "full_episode_presence",
            "ground_reference_source",
            "ground_reference_z_m",
            "inspect_altitude_code",
            "inspect_altitude_m",
            "inspect_fov_coverage_required",
            "min_path_length_m",
            "planned_route_waypoints_enu_m",
            "role",
            "route_waypoints_enu_m",
            "semantic_role",
            "state_sequence",
            "task_id",
            "uav_corridor",
            "uav_corridor_role",
        ), uav=True),
        # Corridor-observer UAV.
        _source_branch((
            "activation_tick",
            "assigned_altitude_m",
            "category",
            "ground_reference_source",
            "ground_reference_z_m",
            "observer_lifecycle",
            "planned_route_waypoints_enu_m",
            "role",
            "route_waypoints_enu_m",
            "semantic_role",
            "state_sequence",
            "task_id",
            "uav_corridor",
            "uav_corridor_role",
        ), uav=True),
        # Temporary NFZ / hazmat-zone facility.
        _source_branch((*route_actor, "incident_state", "semantic_scope",
                        "spawn_policy")),
        # Mission UAV.  Runtime families are optional producer-governed
        # state, which covers the L3-1/L3-2/L3-3 factual variants without
        # turning their concrete keys into schema properties.
        _source_branch(mission_uav, uav=True),
        # L3-1 moving event actor.
        _source_branch((*route_actor, "deactivation_tick",
                        "event_actor_motion_contract", "spawn_policy")),
    ]


def _sumo_semantic_metadata() -> dict:
    schema = _frame._semantic_metadata()
    schema["oneOf"][0]["properties"]["logical_asset_id"] = {
        "enum": [
            "vehicle.emergency.ambulance.v1",
            "vehicle.emergency.suv.v1",
            "vehicle.ground.boxcar.v1",
        ],
    }
    schema["oneOf"][1]["properties"]["logical_asset_id"] = {
        "enum": [
            "vehicle.emergency.suv.v1",
            "vehicle.ground.boxcar.v1",
        ],
    }
    return schema


def _sumo_row(*, required_lifecycle: bool, retained: bool) -> dict:
    row = _frame._sumo_row(
        required_lifecycle=required_lifecycle,
        retained=retained,
    )
    row["properties"]["sumo_vehicle"]["properties"][
        "semantic_metadata"
    ] = _sumo_semantic_metadata()
    vehicle_type = {
        "type": "string",
        "pattern": (
            "^(aero_passenger|aero_emergency|aero_ambulance)"
            "(@[^@\\s]+)?$"
        ),
    }
    row["properties"]["background_vehicle"]["properties"][
        "vehicle_type"
    ] = vehicle_type
    row["properties"]["sumo_vehicle"]["properties"][
        "vehicle_type"
    ] = vehicle_type
    return row


def _uav_flow_row() -> dict:
    row = _frame._uav_flow_row()
    if "planned_route_waypoints_enu_m" not in row["required"]:
        row["required"].append("planned_route_waypoints_enu_m")
    return row


def _pad_rows() -> list[dict]:
    base = _frame._pad_row()
    base["properties"].pop("scene_occupancy_repair", None)
    base["required"] = [
        name for name in base["required"] if name != "scene_occupancy_repair"
    ]
    if "semantic_scope" not in base["required"]:
        base["required"].append("semantic_scope")

    repaired = _frame._pad_row()
    repaired["properties"]["scene_occupancy_repair"] = (
        _frame._scene_occupancy_repair()
    )
    for name in ("semantic_scope", "scene_occupancy_repair"):
        if name not in repaired["required"]:
            repaired["required"].append(name)
    return [base, repaired]


def _entities() -> dict:
    return {
        "type": "array",
        "items": {
            "oneOf": [
                _sumo_row(required_lifecycle=False, retained=False),
                _sumo_row(required_lifecycle=True, retained=False),
                _sumo_row(required_lifecycle=True, retained=True),
                _uav_flow_row(),
                *_pad_rows(),
                *_source_rows(),
            ],
        },
    }


def formal_truth_frame_schema() -> dict:
    """Return the closed schema for one L3 ``truth_frames.jsonl`` row."""
    schema = _frame.formal_truth_frame_schema()
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "formal_truth_frame_l3.record.schema.json"
    )
    schema["title"] = "AERO_WORLD formal_truth_frame L3 frame record schema"
    schema["properties"]["episode_id"] = {
        "type": "string",
        "pattern": _EPISODE_PATTERN,
    }
    schema["properties"]["frame_id"] = {
        "type": "string",
        "pattern": _FRAME_PATTERN,
    }
    schema["properties"]["entities"] = _entities()
    return schema
