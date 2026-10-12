"""Closed whole-row grammar for L5 formal truth frames.

``convert_to_render_ready`` writes one complete frame object per JSONL row.
The capture filter only crops ``entities`` and rebuilds the three summaries
derived from them.  The root therefore remains the 26-key frame written by
the converter.

The source-authored entity arms reuse the compiled L5 formal-roster schema:
``preserved_fields_from`` copies only ``PRESERVED_ENTITY_FIELDS`` from a
trajectory sample and its roster entry, while this module adds the truth
pose, render presence, annotations and visibility written per frame.  The
SUMO, global-UAV and global-pad arms have separate writer-owned payloads.
The composition is pure: it reads no records and mutates no shared schema.

``entity_motion_state`` and ``sumo_traffic_light_states`` remain dynamic
maps.  Their anchored key rules preserve instance identity instead of
enumerating entity or controller ids from one episode.
"""

from __future__ import annotations

from copy import deepcopy

from Dataset.world_model.graph.formal_roster_l5_structure import (
    formal_roster_schema as _formal_roster_schema,
)

__all__ = ["formal_truth_frame_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

_RUNTIME_STATE_FIELDS = frozenset({
    "communication_state",
    "control_state",
    "facility_state",
    "incident_state",
    "mission_state",
    "navigation_state",
    "pedestrian_state",
    "security_state",
    "sensor_state",
    "vehicle_state",
})

# convert_to_render_ready.preserved_fields_from copies the declared
# PRESERVED_ENTITY_FIELDS from the trajectory and roster; it delegates
# runtime families to runtime_state_fields_from_sources.  Generated visibility
# is absent here because the per-frame writer owns that value.
_PRESERVED_ENTITY_FIELDS = frozenset({
    "activation_tick",
    "assigned_altitude_m",
    "background_pedestrian",
    "background_role",
    "background_vehicle",
    "capture_boundary",
    "capture_boundary_id",
    "capture_contract",
    "category",
    "contract_facility",
    "contract_inspect_uav",
    "contract_logical_sidecar",
    "contract_scenario_id",
    "deactivation_tick",
    "entity_kind",
    "event_actor_motion_contract",
    "full_episode_presence",
    "ground_flow_contract",
    "ground_reference_source",
    "ground_reference_z_m",
    "inspect_altitude_code",
    "inspect_altitude_m",
    "inspect_capture_corridor",
    "inspect_fov_coverage_required",
    "inspect_observes_boundary",
    "lifecycle",
    "min_path_length_m",
    "mission_boundary_crossing",
    "motion_contract",
    "observer_lifecycle",
    "pad_boundary_policy",
    "path_deviation_contract",
    "planned_route_waypoints_enu_m",
    "role",
    "route_waypoints_enu_m",
    "semantic_role",
    "semantic_scope",
    "spawn_policy",
    "state_sequence",
    "task_id",
    "task_kind",
    "task_state",
    "uav_boundary_crossing_required",
    "uav_corridor",
    "uav_corridor_role",
    "uav_crosses_boundary",
    *_RUNTIME_STATE_FIELDS,
})

_ENTITY_KEY_PATTERN = r"^[A-Za-z0-9_.:-]+$"
_CONTROLLER_KEY_PATTERN = r"^[A-Za-z0-9_.:-]+$"


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _float_vector(length: int) -> dict:
    return {
        "type": "array",
        "minItems": length,
        "maxItems": length,
        "items": _float_leaf(),
    }


def _nullable_float() -> dict:
    return {"oneOf": [_float_leaf(), {"type": "null"}]}


def _runtime_visibility(*, retained: bool) -> dict:
    properties = {
        "distance_to_capture_boundary_m": _nullable_float(),
        "expanded_boundary_padding_m": {"const": 60.0},
        "policy": {"const": "roi_polygon_expanded_60m_runtime_truth_crop_v1"},
        "runtime_visible": {"type": "boolean"},
    }
    required = list(properties)
    if retained:
        properties.update({
            "retained_outside_runtime_boundary": {"const": True},
            "retention_policy": {
                "const": "required_sumo_vehicle_preserve_source_lifecycle_v1",
            },
        })
        required.extend((
            "retained_outside_runtime_boundary",
            "retention_policy",
        ))
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _uav_visibility() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "camera_call_policy",
            "capture_roi_policy",
            "inspect_observation_distance_m",
            "roi_capture_distance_m",
            "roi_capture_eligible",
            "selected_for_capture_truth",
        ],
        "properties": {
            "camera_call_policy": {
                "const": "do_not_call_camera_when_roi_capture_eligible_is_false",
            },
            "capture_roi_policy": {
                "const": "camera_only_while_uav_position_inside_capture_roi_polygon_v1",
            },
            "inspect_observation_distance_m": _float_leaf(),
            "roi_capture_distance_m": _nullable_float(),
            "roi_capture_eligible": {"type": "boolean"},
            "selected_for_capture_truth": {"type": "boolean"},
        },
    }


def _truth_pose() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "authority_mode",
            "authority_owner",
            "coordinate_contract_id",
            "position_enu_m",
            "rotation_deg",
            "velocity_enu_mps",
        ],
        "properties": {
            "authority_mode": {"const": "authoritative_input"},
            "authority_owner": {"const": "dataset_converter"},
            "coordinate_contract_id": {"const": "coord.external_enu_m.v1"},
            "position_enu_m": _float_vector(3),
            "rotation_deg": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pitch_deg", "roll_deg", "yaw_deg"],
                "properties": {
                    "pitch_deg": {"const": 0.0},
                    "roll_deg": {"const": 0.0},
                    "yaw_deg": _float_leaf(),
                },
            },
            "velocity_enu_mps": _float_vector(3),
        },
    }


def _annotations() -> dict:
    activity = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "activity_type",
            "animation_hint",
            "posture",
            "social_state",
        ],
        "properties": {
            "activity_type": _identifier(),
            "animation_hint": _identifier(),
            "posture": _identifier(),
            "social_state": _identifier(),
        },
    }
    network = {
        "type": "object",
        "additionalProperties": False,
        "required": ["latency_ms", "packet_loss", "status"],
        "properties": {
            "latency_ms": {"const": 0.0},
            "packet_loss": {"const": 0.0},
            "status": {"const": "nominal"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["activity_type", "speed_mps", "state_facets"],
        "properties": {
            "activity_type": _identifier(),
            "speed_mps": _float_leaf(),
            "state_facets": {
                "type": "object",
                "additionalProperties": False,
                "required": ["activity", "network"],
                "properties": {"activity": activity, "network": network},
            },
        },
    }


_TRUTH_CORE_REQUIRED = (
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
    "state_revision",
    "tags",
    "truth_pose",
    "visual_revision",
)


def _truth_core_properties(
    roster_properties: dict, *, retained: bool = False
) -> dict:
    def roster_rule(name: str, fallback: dict) -> dict:
        return deepcopy(roster_properties.get(name, fallback))

    entity_kind = roster_rule("entity_kind", _identifier())
    return {
        "annotations": _annotations(),
        "entity_category": roster_rule("entity_category", _identifier()),
        "entity_id": roster_rule("entity_id", _identifier()),
        "entity_kind": entity_kind,
        "entity_type": roster_rule("entity_type", entity_kind),
        "label_class": roster_rule("label_class", _identifier()),
        "logical_asset_id": roster_rule("logical_asset_id", _identifier()),
        "proxy_template_id": roster_rule("proxy_template_id", _identifier()),
        "render_presence": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "global_roster",
                "offstage",
                "offstage_reason",
                "roi_membership",
                "submission_state",
                "visibility_state",
            ],
            "properties": {
                "global_roster": {"const": True},
                "offstage": {"const": False},
                "offstage_reason": {"const": "none"},
                "roi_membership": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": _identifier(),
                },
                "submission_state": {"const": "submit_to_ue"},
                "visibility_state": {"const": "visible"},
            },
        },
        "roi_id": roster_rule("roi_id", _identifier()),
        "runtime_visibility": _runtime_visibility(retained=retained),
        "site_id": roster_rule("site_id", _identifier()),
        "state_revision": {"type": "integer", "minimum": 1},
        "tags": roster_rule(
            "tags", {"type": "array", "items": _identifier()}
        ),
        "truth_pose": _truth_pose(),
        "visual_revision": {"const": 1},
    }


def _closed_entity(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(dict.fromkeys(required)),
        "properties": properties,
    }


def _roster_branches() -> tuple[dict, dict, dict, list[dict]]:
    rows = deepcopy(
        _formal_roster_schema()["properties"]["entities"]["items"]["oneOf"]
    )
    sumo_rows: list[dict] = []
    uav_rows: list[dict] = []
    pad_rows: list[dict] = []
    source_rows: list[dict] = []
    for row in rows:
        properties = row["properties"]
        source = properties.get("source", {}).get("const")
        if source == "sumo_traci":
            sumo_rows.append(row)
        elif source == "uav_global_flow" and "uav_global_flow" in properties:
            uav_rows.append(row)
        elif source == "uav_global_flow" and "uav_global_pad" in properties:
            pad_rows.append(row)
        elif {"placement", "scene_occupancy_repair"}.issubset(row["required"]):
            # Those two roster-only repair markers are not preserved into a
            # frame.  Its resulting facility arms are already represented by
            # the no-contract and contract-facility rows.
            continue
        else:
            source_rows.append(row)
    if len(sumo_rows) != 1 or len(uav_rows) != 1 or len(source_rows) != 10:
        raise ValueError("L5 formal-roster branches diverged from truth-frame writer")
    if not pad_rows:
        raise ValueError("L5 formal roster has no global-pad producer branch")
    pad = pad_rows[0]
    for other in pad_rows[1:]:
        for name in (
            "background_role",
            "motion_contract",
            "semantic_scope",
            "source",
            "uav_global_pad",
        ):
            if other["properties"][name] != pad["properties"][name]:
                raise ValueError(
                    f"L5 global-pad truth property diverged across roster arms: {name}"
                )
    return sumo_rows[0], uav_rows[0], pad, source_rows


def _sumo_vehicle_schema(
    roster_row: dict, *, lifecycle_required: bool
) -> dict:
    schema = deepcopy(roster_row["properties"]["sumo_vehicle"])
    properties = schema["properties"]
    properties["controlling_signal_state"] = {
        "enum": ["green", "mixed", "none", "off", "red", "red_yellow", "yellow"],
    }
    properties.update({
        "accel_mps2": _float_leaf(),
        "dimensions_m": {
            "type": "object",
            "additionalProperties": False,
            "required": ["height", "length", "width"],
            "properties": {
                "height": _float_leaf(),
                "length": _float_leaf(),
                "width": _float_leaf(),
            },
        },
        "signals": {"type": "integer"},
        "source_alpha": _float_leaf(),
        "source_next_time_s": _float_leaf(),
        "source_prev_time_s": _float_leaf(),
        "speed_mps": _float_leaf(),
        "sumo_angle_deg": _float_leaf(),
        "sumo_front_bumper_xy_m": _float_vector(2),
        "sumo_xy_m": _float_vector(2),
        "truth_front_bumper_enu_m": _float_vector(3),
    })
    if lifecycle_required:
        properties["required_lifecycle_preservation_policy"] = {
            "const": "required_sumo_vehicle_preserve_source_lifecycle_v1",
        }
    schema["required"] = list(properties)
    return schema


def _sumo_row(
    roster_row: dict, *, lifecycle_required: bool, retained: bool
) -> dict:
    roster_properties = roster_row["properties"]
    properties = _truth_core_properties(roster_properties, retained=retained)
    for name in (
        "background_role",
        "background_vehicle",
        "ground_flow_contract",
        "source",
        "sumo_segment",
    ):
        properties[name] = deepcopy(roster_properties[name])
    for family in _RUNTIME_STATE_FIELDS:
        if family in roster_properties:
            properties[family] = deepcopy(roster_properties[family])
    properties.update({
        "capture_boundary_id": _identifier(),
        "state": _identifier(),
        "sumo_vehicle": _sumo_vehicle_schema(
            roster_row, lifecycle_required=lifecycle_required
        ),
        "sumo_visibility": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "inspect_observation_distance_m",
                "selected_for_capture_truth",
            ],
            "properties": {
                "inspect_observation_distance_m": _float_leaf(),
                "selected_for_capture_truth": {"const": True},
            },
        },
    })
    return _closed_entity(
        properties,
        [
            *_TRUTH_CORE_REQUIRED,
            "background_role",
            "background_vehicle",
            "capture_boundary_id",
            "ground_flow_contract",
            "source",
            "state",
            "sumo_segment",
            "sumo_vehicle",
            "sumo_visibility",
        ],
    )


def _uav_global_flow_payload(roster_row: dict) -> dict:
    payload = deepcopy(roster_row["properties"]["uav_global_flow"])
    payload["properties"].update({
        "active_event_ids": {"type": "array", "items": _identifier()},
        "source_alpha": _float_leaf(),
        "source_next_time_s": _float_leaf(),
        "source_prev_time_s": _float_leaf(),
    })
    payload["required"] = list(payload["properties"])
    return payload


def _uav_flow_row(roster_row: dict) -> dict:
    roster_properties = roster_row["properties"]
    properties = _truth_core_properties(roster_properties)
    for name in (
        "background_role",
        "motion_contract",
        "planned_route_waypoints_enu_m",
        "semantic_role",
        "source",
        "uav_segment",
    ):
        properties[name] = deepcopy(roster_properties[name])
    properties.update({
        "mission_type": _identifier(),
        "task_id": _identifier(),
        "uav_global_flow": _uav_global_flow_payload(roster_row),
        "uav_visibility": _uav_visibility(),
    })
    return _closed_entity(
        properties,
        [
            *_TRUTH_CORE_REQUIRED,
            "background_role",
            "motion_contract",
            "planned_route_waypoints_enu_m",
            "semantic_role",
            "source",
            "uav_global_flow",
            "uav_segment",
            "uav_visibility",
        ],
    )


def _pad_row(roster_row: dict) -> dict:
    roster_properties = roster_row["properties"]
    properties = _truth_core_properties(roster_properties)
    for name in (
        "background_role",
        "motion_contract",
        "semantic_scope",
        "source",
        "uav_global_pad",
    ):
        properties[name] = deepcopy(roster_properties[name])
    return _closed_entity(
        properties,
        [
            *_TRUTH_CORE_REQUIRED,
            "background_role",
            "motion_contract",
            "semantic_scope",
            "source",
            "uav_global_pad",
        ],
    )


def _source_row(roster_row: dict) -> dict:
    roster_properties = roster_row["properties"]
    properties = _truth_core_properties(roster_properties)
    required = list(_TRUTH_CORE_REQUIRED)
    roster_required = set(roster_row["required"])
    for name in sorted(_PRESERVED_ENTITY_FIELDS & roster_properties.keys()):
        properties[name] = deepcopy(roster_properties[name])
        if name in roster_required:
            required.append(name)
    properties["state"] = _identifier()
    category = roster_properties.get("category", {}).get("const")
    if category == "uav":
        properties["uav_visibility"] = _uav_visibility()
        required.append("uav_visibility")
    return _closed_entity(properties, required)


def _entities_schema(
    sumo_row: dict, uav_row: dict, pad_row: dict, source_rows: list[dict]
) -> dict:
    return {
        "type": "array",
        "items": {
            "oneOf": [
                _sumo_row(
                    sumo_row, lifecycle_required=False, retained=False
                ),
                _sumo_row(
                    sumo_row, lifecycle_required=True, retained=False
                ),
                _sumo_row(
                    sumo_row, lifecycle_required=True, retained=True
                ),
                _uav_flow_row(uav_row),
                _pad_row(pad_row),
                *[_source_row(row) for row in source_rows],
            ],
        },
    }


def _incident_row() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "accident_class",
            "affected_vehicle_ids",
            "anchor",
            "end_s",
            "episode_event_id",
            "episode_scenario_id",
            "incident_id",
            "injection_method",
            "intent",
            "intent_stage",
            "start_s",
        ],
        "properties": {
            "accident_class": _identifier(),
            "affected_vehicle_ids": {
                "type": "array",
                "items": _identifier(),
            },
            "anchor": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "geometry_source",
                    "sumo_edge_id",
                    "sumo_lane_id",
                    "truth_position_enu_m",
                ],
                "properties": {
                    "geometry_source": _identifier(),
                    "sumo_edge_id": _identifier(),
                    "sumo_lane_id": _identifier(),
                    "truth_position_enu_m": _float_vector(3),
                },
            },
            "end_s": _float_leaf(),
            "episode_event_id": _identifier(),
            "episode_scenario_id": _identifier(),
            "incident_id": _identifier(),
            "injection_method": _identifier(),
            "intent": _identifier(),
            "intent_stage": _identifier(),
            "start_s": _float_leaf(),
            "traffic_light_id": _identifier(),
        },
    }


def _traffic_light_state() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "controlled_links",
            "next_switch_s",
            "phase_index",
            "program_id",
            "state",
        ],
        "properties": {
            "controlled_links": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 3,
                        "items": {"type": "string"},
                    },
                },
            },
            "next_switch_s": _float_leaf(),
            "phase_index": {"type": "integer", "minimum": 0},
            "program_id": {"type": "string"},
            "state": _identifier(),
        },
    }


def _sumo_semantics(segment_schema: dict) -> dict:
    properties = {
        "absolute_time_s": _float_leaf(),
        "active_incident_count": {"type": "integer", "minimum": 0},
        "active_incidents": {"type": "array", "items": _incident_row()},
        "active_selected_vehicle_count": {"type": "integer", "minimum": 0},
        "enabled": {"const": True},
        "max_observation_distance_m": {"const": 60.0},
        "retained_outside_boundary_sumo_vehicle_count": {
            "type": "integer",
            "minimum": 0,
        },
        "runtime_boundary_visible_selected_vehicle_count": {
            "type": "integer",
            "minimum": 0,
        },
        "scenario_id": _identifier(),
        "segment": deepcopy(segment_schema),
        "selected_vehicle_count": {"type": "integer", "minimum": 0},
        "semantic_lifecycle_outside_runtime_boundary_count": {
            "type": "integer",
            "minimum": 0,
        },
        "source_alpha": _float_leaf(),
        "source_next_time_s": _float_leaf(),
        "source_prev_time_s": _float_leaf(),
        "source_vehicle_count": {"type": "integer", "minimum": 0},
        "traffic_light_count": {"type": "integer", "minimum": 0},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _uav_frame_payload(segment_schema: dict) -> dict:
    properties = {
        "absolute_time_s": _float_leaf(),
        "active_selected_uav_count": {"type": "integer", "minimum": 0},
        "baseline_active_uavs": {"type": "integer", "minimum": 0},
        "camera_capture_roi_policy": {
            "const": "camera_only_while_uav_position_inside_capture_roi_polygon_v1",
        },
        "enabled": {"const": True},
        "minimum_active_uavs_required": {"type": "integer", "minimum": 0},
        "roi_capture_eligible_uav_count": {"type": "integer", "minimum": 0},
        "segment": deepcopy(segment_schema),
        "selected_uav_count": {"type": "integer", "minimum": 0},
        "source_alpha": _float_leaf(),
        "source_next_time_s": _float_leaf(),
        "source_prev_time_s": _float_leaf(),
        "source_uav_count": {"type": "integer", "minimum": 0},
        "task_count": {"type": "integer", "minimum": 0},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _empty_array(item_schema: dict) -> dict:
    return {
        "type": "array",
        "maxItems": 0,
        "items": item_schema,
    }


def _empty_object() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "maxProperties": 0,
    }


def formal_truth_frame_schema() -> dict:
    """Return the closed schema for one L5 ``truth_frames.jsonl`` row."""
    sumo_row, uav_row, pad_row, source_rows = _roster_branches()
    sumo_segment = deepcopy(sumo_row["properties"]["sumo_segment"])
    uav_segment = deepcopy(uav_row["properties"]["uav_segment"])
    incident = _incident_row()

    # The converter writes the SUMO and UAV field groups together.  The
    # absent arms have no current L5 instance, but are explicit writer paths.
    sumo_presence = {
        "if": {
            "properties": {"sumo_segment": {"type": "null"}},
            "required": ["sumo_segment"],
        },
        "then": {
            "properties": {
                "sumo_active_incidents": _empty_array(incident),
                "sumo_segment": {"type": "null"},
                "sumo_semantics": {"type": "null"},
                "sumo_traffic_light_states": _empty_object(),
            },
        },
        "else": {
            "properties": {
                "sumo_active_incidents": {
                    "type": "array",
                    "items": incident,
                },
                "sumo_segment": sumo_segment,
                "sumo_semantics": _sumo_semantics(sumo_segment),
                "sumo_traffic_light_states": {
                    "type": "object",
                    "additionalProperties": _traffic_light_state(),
                    "propertyNames": {
                        "type": "string",
                        "pattern": _CONTROLLER_KEY_PATTERN,
                    },
                },
            },
        },
    }
    uav_presence = {
        "if": {
            "properties": {"uav_segment": {"type": "null"}},
            "required": ["uav_segment"],
        },
        "then": {
            "properties": {
                "uav_global_flow": {"type": "null"},
                "uav_segment": {"type": "null"},
            },
        },
        "else": {
            "properties": {
                "uav_global_flow": _uav_frame_payload(uav_segment),
                "uav_segment": uav_segment,
            },
        },
    }

    properties = {
        "active_roi_id": _identifier(),
        "active_site_id": _identifier(),
        "capture_boundary_id": _identifier(),
        "dt_s": {"const": 0.1},
        "entities": _entities_schema(sumo_row, uav_row, pad_row, source_rows),
        "entity_motion_state": {
            "type": "object",
            "additionalProperties": _identifier(),
            "propertyNames": {
                "type": "string",
                "pattern": _ENTITY_KEY_PATTERN,
            },
        },
        "episode_id": _identifier(),
        "frame_id": {"type": "string", "pattern": r"^.+_tick_[0-9]+$"},
        "frame_seq": {"type": "integer", "minimum": 0},
        "inspect_observes_boundary": {"type": "boolean"},
        "map_id": _identifier(),
        "pad_boundary_policy": {
            "type": "object",
            "additionalProperties": False,
            "required": ["default", "inside_required_for"],
            "properties": {
                "default": _identifier(),
                "inside_required_for": {
                    "type": "array",
                    "items": _identifier(),
                },
            },
        },
        "render_mode": {"const": "ue_pie"},
        "roster_summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["by_category", "total"],
            "properties": {
                "by_category": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "integer",
                        "minimum": 0,
                    },
                    "propertyNames": {
                        "type": "string",
                        "pattern": r"^[a-z][a-z0-9_]*$",
                    },
                },
                "total": {"type": "integer", "minimum": 0},
            },
        },
        "schema_name": {"const": "truth_frame"},
        "schema_version": {"const": "v1"},
        "sim_time_s": _float_leaf(),
        "tick": {"type": "integer", "minimum": 0},
        "tick_hz": {"const": 10},
        "uav_crosses_boundary": {"type": "boolean"},
    }
    required = [
        "active_roi_id",
        "active_site_id",
        "capture_boundary_id",
        "dt_s",
        "entities",
        "entity_motion_state",
        "episode_id",
        "frame_id",
        "frame_seq",
        "inspect_observes_boundary",
        "map_id",
        "pad_boundary_policy",
        "render_mode",
        "roster_summary",
        "schema_name",
        "schema_version",
        "sim_time_s",
        "sumo_active_incidents",
        "sumo_segment",
        "sumo_semantics",
        "sumo_traffic_light_states",
        "tick",
        "tick_hz",
        "uav_crosses_boundary",
        "uav_global_flow",
        "uav_segment",
    ]
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_truth_frame_l5.record.schema.json",
        "title": "AERO_WORLD formal_truth_frame L5 JSONL row schema",
        "type": "object",
        "unevaluatedProperties": False,
        "required": required,
        "properties": properties,
        "allOf": [sumo_presence, uav_presence],
    }
