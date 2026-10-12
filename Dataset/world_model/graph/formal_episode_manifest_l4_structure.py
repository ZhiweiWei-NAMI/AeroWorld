"""Closed record structure for one formal episode manifest, L4 layer.

Source family ``formal_episode_manifest``; profiles L4-1_v1, L4-1_v2,
L4-2_v1, L4-2_v2, L4-3_v1, L4-3_v2, L4-3_v3, L4-4_v1, L4-4_v2, L4-5_v1,
L4-5_v2, L4-5_v3, L4-6_v1, L4-6_v2, L4-7_v1, L4-7_v2, L4-8_v1, L4-8_v2,
L4-9_v1, L4-9_v2, L4-10_v1, L4-10_v2, L4-11_v1, L4-11_v2 (24 profiles x
3 seeds = 72 declarations, one ``episode_manifest.json`` JSON root each).

Record identity: ``structure_sources.py`` does not list this family in
``_ENTITY_FILES`` and ``iter_source_records`` reads ``.json`` roots whole
(:167-171), so the record root of this grammar is the complete file root.
The family is in ``EXTENSION_TAIL_FAMILIES`` (:37-41) and is declared
``source_schema_pending`` in ``_declared_structure_statuses()``
(``structure_closure.py``).

Producer chain, verified against current source this session:

1. ``Dataset/tools/batch_generate.py:generate_episode`` writes the source
   manifest (:2539-2570) with the 12-key base block; ``manifest[
   "artifacts"].update`` (:2590-2602) adds the three review artifacts;
   ``vehicle_source_policy`` and the 6-key ``sumo_explicit_vehicle_plan``
   summary (:2603-2611) are rewritten (:2612).
2. ``Dataset/tools/convert_to_render_ready.py`` deep-copies that manifest
   (:5673 ``render_manifest = copy.deepcopy(manifest)``) and ``.update``
   (:5674-5726) rewrites: identity keys (:5676-5682), the 7-key
   ``runtime_spatial_crop`` payload (``runtime_spatial_crop_payload``
   :3504-3513), the ``source_vehicle_authority`` payload
   (``source_vehicle_authority_payload`` :3478-3501, its 9 keys built at
   :5463-5467, extended by ``vehicle_source_policy`` and the 7-key
   ``explicit_vehicle_plan`` block at :5468-5478), ``sumo_traffic``/
   ``uav_global_flow`` summaries (:5540-5582, consumed at :5685-5690), the
   2-key ``generation`` payload (:5691-5694), ``record_counts``/
   ``canonical_record_counts`` (:5633-5696), ``node_counts``
   (:5697-5715), ``artifacts``/``canonical_artifacts`` (:5644-5672,
   :5716-5717), ``time_range`` (:5718-5723) and ``validation_summary``
   (:5724); write (:5743). ``sumo_traffic.explicit_vehicle_plan`` deep
   copies the authority block (:5546-5548), so both embedded plan blocks
   share one 7-key structure.
3. ``Dataset/tools/scene_occupancy.py:audit_and_attach_episode`` (:666-677,
   called :5744) attaches the 5-key ``scene_occupancy`` block and the
   artifact keys (``update_episode_manifests_with_scene_occupancy``
   :635-663).
4. ``Dataset/tools/filter_render_ready_truth_for_capture.py`` rewrites
   record counts, generation and artifact paths and adds
   ``capture_visible_truth_filter``/``capture_truth_sync``.  The frozen
   L4-2_v1 transfer-boundary root also contains
   ``source_event_script_digest``; the current writer chain has no assignment
   for that key, so the frozen record is its structural authority here.

Integrity metadata: the current converter writes no
``generation.source_contract_hash``; frozen roots may carry it, so it is
optional in the raw grammar.  The optional root
``source_event_script_digest`` is likewise retained from frozen roots.
Both exact paths are in ``structure_schema._PUBLICATION_EXCLUDED_PATHS``
and are excluded only by the publication projection.

The frozen ``canonical_asset_records`` members carry
``asset_conflict_count``/``asset_conflict_fields``/``asset_conflicts`` as
data fields.  Their raw string and array types remain in the schema; conflict
interpretation and source repair are outside this grammar.

The constructor performs no data reads or compilation at import.
"""

from __future__ import annotations

_PATTERN_VEHICLE_ID = (
    "^(?:bg_vehicle_[A-Za-z0-9_]+__seed[0-9]{2}|traffic_bg_[A-Za-z0-9_]+)$"
)
_PATTERN_UAV_ID = "^uav_[A-Za-z0-9_]+$"
_PATTERN_UAV_ENTITY_ID = "^global_uav_uav_[A-Za-z0-9_]+$"
# Selection task ids are copied from generate_uav_flow.py.  Its task-id
# prefixes differ from the mission-type names in task_count_by_type.
_PATTERN_UAV_TASK_ID = (
    "^(?:delivery_p[0-9]+_[0-9]{3}|edge_relay_[0-9]{2}|"
    "incident_response_[0-9]{2}|infra_inspect_p[0-9]+_[0-9]{2}|"
    "intersection_inspect_[0-9]{2}|pad_patrol_[A-Za-z0-9_]+)$"
)
_PATTERN_INCIDENT_INJECTION = "^[A-Za-z0-9_.]+(?:\\+[A-Za-z0-9_.]+)*$"
_UAV_TASK_TYPES = (
    "edge_compute_relay",
    "incident_response_inspection",
    "infrastructure_inspection",
    "intersection_inspect",
    "logistics_delivery",
    "pad_patrol",
)

_SUMO_TYPES = (
    "aero_ambulance", "aero_delivery", "aero_emergency",
    "aero_passenger", "aero_police",
)

_L4_PROFILES = (
    "L4-1_v1", "L4-1_v2", "L4-2_v1", "L4-2_v2", "L4-3_v1",
    "L4-3_v2", "L4-3_v3", "L4-4_v1", "L4-4_v2", "L4-5_v1",
    "L4-5_v2", "L4-5_v3", "L4-6_v1", "L4-6_v2", "L4-7_v1",
    "L4-7_v2", "L4-8_v1", "L4-8_v2", "L4-9_v1", "L4-9_v2",
    "L4-10_v1", "L4-10_v2", "L4-11_v1", "L4-11_v2",
)


def _str() -> dict[str, str]:
    return {"type": "string"}


def _nonblank_str() -> dict[str, str]:
    return {"type": "string", "minLength": 1}


def _int(minimum: int | None = None, maximum: int | None = None) -> dict[str, int]:
    rule: dict[str, int] = {"type": "integer"}
    if minimum is not None:
        rule["minimum"] = minimum
    if maximum is not None:
        rule["maximum"] = maximum
    return rule


def _number() -> dict[str, str]:
    return {"type": "number"}


def _bool() -> dict[str, str]:
    return {"type": "boolean"}


def _obj(properties: dict[str, dict], required: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _array_of(item: dict) -> dict[str, object]:
    return {"type": "array", "items": item}


def _string_id_map(value: dict) -> dict[str, object]:
    return {
        "type": "object",
        "patternProperties": {"^.+$": value},
        "minProperties": 0,
        "additionalProperties": False,
    }


def _vehicle_id_map(value: dict) -> dict[str, object]:
    """Dynamic key map whose keys are planned SUMO vehicle ids."""
    return {
        "type": "object",
        "patternProperties": {_PATTERN_VEHICLE_ID: value},
        "additionalProperties": False,
    }


def _int_vehicle_id_map() -> dict[str, object]:
    return _vehicle_id_map(_int())


def _float_vehicle_id_map() -> dict[str, object]:
    return _vehicle_id_map(_number())


def _string_vehicle_id_map() -> dict[str, object]:
    return _vehicle_id_map(_str())


def _seed_semantics() -> dict[str, object]:
    return {
        "type": "object",
        "patternProperties": {
            "^seed[0-9]{2}$": {"type": "string"},
        },
        "required": ["seed00", "seed01", "seed02"],
        "additionalProperties": False,
    }


def _direction_distribution() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "inbound": _number(),
            "outbound": _number(),
            "cross": _number(),
        },
        "required": ["inbound", "outbound", "cross"],
        "additionalProperties": False,
    }


def _traffic_profile() -> dict[str, object]:
    """The developer-constant traffic profile dict (batch_generate)."""
    return _obj(
        {
            "profile_id": _nonblank_str(),
            "seed_label": _nonblank_str(),
            "flow_mode": _nonblank_str(),
            "schedule_policy": _nonblank_str(),
            "slot_count": _int(minimum=0),
            "slot_duration_ticks": _int(minimum=0),
            "core_crossing_duration_ticks": _int(minimum=0),
            "background_vehicles_per_slot": _int(minimum=0),
            "accepted_background_vehicle_target": _int(minimum=0),
            "seed_semantics": _seed_semantics(),
            "direction_distribution": _direction_distribution(),
            "target_speed_mps": _number(),
            "slot_entry_jitter_ticks": _array_of(_int()),
        },
        [
            "profile_id", "seed_label", "flow_mode", "schedule_policy",
            "slot_count", "slot_duration_ticks", "core_crossing_duration_ticks",
            "background_vehicles_per_slot", "accepted_background_vehicle_target",
            "seed_semantics", "direction_distribution", "target_speed_mps",
            "slot_entry_jitter_ticks",
        ],
    )


def _seed_profile() -> dict[str, object]:
    return _obj(
        {
            "seed_index": _int(minimum=0),
            "seed_label": _nonblank_str(),
            "profile_id": _nonblank_str(),
            "direction_bias": _nonblank_str(),
        },
        ["seed_index", "seed_label", "profile_id", "direction_bias"],
    )


def _sumo_explicit_vehicle_plan_summary() -> dict[str, object]:
    """batch_generate's 6-key summary (:2604-2611)."""
    return _obj(
        {
            "schema": {"const": "aero_explicit_sumo_vehicle_plan_v2"},
            "vehicle_count": _int(minimum=0),
            "minimum_vehicle_count": _int(minimum=0),
            "seed_profile": _seed_profile(),
            "traffic_profile": _traffic_profile(),
            "review": {"const": "vehicle_plan_review.json"},
        },
        [
            "schema", "vehicle_count", "minimum_vehicle_count",
            "seed_profile", "traffic_profile", "review",
        ],
    )


def _selection_audit(policy_const: bool) -> dict[str, object]:
    """The converter's selection audit dict (:4793-4815, 15 keys)."""
    policy: dict[str, str] = (
        {"const": "semantic_hard_sumo_authoritative_background_tolerant_selection_v1"}
        if policy_const
        else _nonblank_str()
    )
    properties: dict[str, object] = {
        "policy": policy,
        "planned_vehicle_count": _int(minimum=0),
        "planned_semantic_vehicle_count": _int(minimum=0),
        "planned_source_presence_required_vehicle_count": _int(minimum=0),
        "planned_capture_required_vehicle_count": _int(minimum=0),
        "realized_vehicle_count": _int(minimum=0),
        "selected_runtime_visible_vehicle_count": _int(minimum=0),
        "extra_realized_background_vehicle_count": _int(minimum=0),
        "extra_realized_background_vehicle_ids": _array_of(_nonblank_str()),
        "missing_background_vehicle_count": _int(minimum=0),
        "missing_background_vehicle_ids": _array_of(_nonblank_str()),
        "extra_visible_background_vehicle_count": _int(minimum=0),
        "extra_visible_background_vehicle_ids": _array_of(_nonblank_str()),
        "missing_visible_background_vehicle_count": _int(minimum=0),
        "missing_visible_background_vehicle_ids": _array_of(_nonblank_str()),
    }
    return _obj(properties, sorted(properties))


def _embedded_explicit_plan() -> dict[str, object]:
    """The converter's full seven-key embedded explicit vehicle plan."""
    plan = _obj(
        {
            "schema": {"const": "aero_explicit_sumo_vehicle_plan_v2"},
            "episode_id": _nonblank_str(),
            "vehicle_count": _int(minimum=0),
            "vehicle_ids": _array_of(_nonblank_str()),
            "selection_audit": _selection_audit(policy_const=True),
            "seed_profile": _seed_profile(),
            "traffic_profile": _traffic_profile(),
        },
        [
            "schema", "episode_id", "vehicle_count", "vehicle_ids",
            "selection_audit", "seed_profile", "traffic_profile",
        ],
    )
    plan["properties"]["vehicle_ids"]["minItems"] = 1
    return plan


def _coordinate_mapper() -> dict[str, object]:
    return _obj(
        {
            "matrix": {
                "type": "array",
                "items": _array_of(_number()),
            },
            "max_error_m": _number(),
            "mean_error_m": _number(),
            "pair_count": _int(minimum=0),
            "source": _nonblank_str(),
        },
        ["matrix", "max_error_m", "mean_error_m", "pair_count", "source"],
    )


def _sumo_traffic_source() -> dict[str, object]:
    """The frozen sumo source dict (:4801-4837), 19 keys."""
    return _obj(
        {
            "output_dir": _str(),
            "frames": _str(),
            "manifest": _str(),
            "incident_plan": _str(),
            "map_id": _nonblank_str(),
            "duration_s": _number(),
            "warmup_s": _number(),
            "capture_start_s": _number(),
            "capture_duration_s": _number(),
            "time_axis_policy": _nonblank_str(),
            "step_length_s": _number(),
            "sample_period_s": _number(),
            "sample_every_ticks": _int(),
            "sample_count": _int(minimum=0),
            "max_vehicles": _int(minimum=0),
            "net_xml": _str(),
            "road_derived_network_manifest": _str(),
            "traffic_light_count": _int(minimum=0),
            "coordinate_mapper": _coordinate_mapper(),
        },
        [
            "output_dir", "frames", "manifest", "incident_plan", "map_id",
            "duration_s", "warmup_s", "capture_start_s", "capture_duration_s",
            "time_axis_policy", "step_length_s", "sample_period_s",
            "sample_every_ticks", "sample_count", "max_vehicles", "net_xml",
            "road_derived_network_manifest", "traffic_light_count",
            "coordinate_mapper",
        ],
    )


def _segment() -> dict[str, object]:
    return _obj(
        {
            "seed_index": _int(minimum=0),
            "seed_label": _nonblank_str(),
            "segment_start_s": _number(),
            "segment_end_s": _number(),
            "duration_s": _number(),
            "phase_name": _nonblank_str(),
        },
        [
            "seed_index", "seed_label", "segment_start_s", "segment_end_s",
            "duration_s", "phase_name",
        ],
    )


def _visibility_geometry() -> dict[str, object]:
    return _obj(
        {
            "capture_polygon_points": _int(minimum=0),
            "inspect_route_points": _int(minimum=0),
            "hfov_deg": _number(),
            "vfov_deg": _number(),
            "altitude_m": _number(),
            "padding_m": _number(),
            "footprint_half_width_m": _number(),
            "footprint_half_height_m": _number(),
        },
        [
            "capture_polygon_points", "inspect_route_points", "hfov_deg",
            "vfov_deg", "altitude_m", "padding_m", "footprint_half_width_m",
            "footprint_half_height_m",
        ],
    )


def _sumo_selection() -> dict[str, object]:
    """The frozen selection dict (:4841-4871), 12 keys."""
    return _obj(
        {
            "vehicle_ids": _array_of(_str()),
            "entity_ids": _string_vehicle_id_map(),
            "candidate_count": _int(minimum=0),
            "selected_count": _int(minimum=0),
            "scenario_vehicle_ids": _array_of(_str()),
            "min_visible_vehicle_target": _int(minimum=0),
            "max_visible_vehicle_target": _int(minimum=0),
            "expanded_to_nearest": _bool(),
            "min_distance_m_by_vehicle_id": _float_vehicle_id_map(),
            "frames_seen_by_vehicle_id": _int_vehicle_id_map(),
            "motion_span_m_by_vehicle_id": _float_vehicle_id_map(),
            "max_speed_mps_by_vehicle_id": _float_vehicle_id_map(),
        },
        [
            "vehicle_ids", "entity_ids", "candidate_count", "selected_count",
            "scenario_vehicle_ids", "min_visible_vehicle_target",
            "max_visible_vehicle_target", "expanded_to_nearest",
            "min_distance_m_by_vehicle_id", "frames_seen_by_vehicle_id",
            "motion_span_m_by_vehicle_id", "max_speed_mps_by_vehicle_id",
        ],
    )


def _source_vehicle_authority() -> dict[str, object]:
    """``source_vehicle_authority_payload`` (:3478-3501) plus the
    converter-attached ``vehicle_source_policy`` and ``explicit_vehicle_plan``
    keys (:5468-5478)."""
    return _obj(
        {
            "enabled": _bool(),
            "policy": {"const": "sumo_only_vehicle_authority_replaces_source_vehicle_truth_v2"},
            "authority": {"const": "episode_local_sumo_traci_after_warmup"},
            "reason": _nonblank_str(),
            "replaced_source_vehicle_count": _int(minimum=0),
            "replaced_source_vehicle_ids": _array_of(_str()),
            "replaced_source_vehicle_counts_by_role": _string_id_map(
                _int(minimum=0)
            ),
            "event_reference_policy": _nonblank_str(),
            "vehicle_source_policy": {
                "const": "explicit_episode_traffic_schedule_only_v2"
            },
            "explicit_vehicle_plan": _embedded_explicit_plan(),
        },
        [
            "enabled", "policy", "authority", "reason",
            "replaced_source_vehicle_count", "replaced_source_vehicle_ids",
            "replaced_source_vehicle_counts_by_role", "event_reference_policy",
            "vehicle_source_policy", "explicit_vehicle_plan",
        ],
    )


def _runtime_spatial_crop() -> dict[str, object]:
    """``runtime_spatial_crop_payload`` (:3504-3513)."""
    return _obj(
        {
            "enabled": {"const": True},
            "policy": {"const": "roi_polygon_expanded_60m_runtime_truth_crop_v1"},
            "capture_boundary_id": _nonblank_str(),
            "expanded_boundary_padding_m": {"const": 60.0},
            "bbox_enu_m": {
                "type": "array",
                "items": _number(),
                "minItems": 4,
                "maxItems": 4,
            },
            "dynamic_frame_rule": _nonblank_str(),
            "uav_camera_rule": _nonblank_str(),
        },
        [
            "enabled", "policy", "capture_boundary_id",
            "expanded_boundary_padding_m", "bbox_enu_m", "dynamic_frame_rule",
            "uav_camera_rule",
        ],
    )


def _canonical_asset_record() -> dict[str, object]:
    """Current eight-key writer row or frozen eleven-key conflict row."""
    properties = {
        "logical_asset_id": _nonblank_str(),
        "authority": _nonblank_str(),
        "source_entity_id": _str(),
        "source_presence_required": _bool(),
        # convert_to_render_ready.SUMO_TYPE_LOGICAL_ASSET_BY_TYPE_ID.
        "sumo_type_id": {"enum": list(_SUMO_TYPES)},
        "source_roster_logical_asset_id": _str(),
        "explicit_type_id": _str(),
        "sumo_record_type_id": _nonblank_str(),
    }
    current = _obj(properties, list(properties))
    frozen_properties = {
        **properties,
        "asset_conflict_count": {
            "type": "string",
            "pattern": "^[0-9]+$",
        },
        "asset_conflict_fields": _str(),
        "asset_conflicts": _array_of(
            _obj(
                {
                    "field": _nonblank_str(),
                    "logical_asset_id": _nonblank_str(),
                    "resolved_by": _nonblank_str(),
                },
                ["field", "logical_asset_id", "resolved_by"],
            )
        ),
    }
    frozen = _obj(frozen_properties, list(frozen_properties))
    return {"oneOf": [current, frozen]}


def _required_lifecycle_policy() -> dict[str, object]:
    compact_properties = {
        "policy": {"const": "required_sumo_vehicle_preserve_source_lifecycle_v1"},
        "required_vehicle_count": _int(minimum=0),
        "clipped_required_vehicle_count": _int(minimum=0),
    }
    compact = _obj(compact_properties, list(compact_properties))
    full_properties = {
        **compact_properties,
        "source_lifecycle_by_vehicle_id": _vehicle_id_map(_obj(
            {
                "vehicle_id": _nonblank_str(),
                "first_tick": _int(minimum=0),
                "last_tick": _int(minimum=0),
                "source_frame_count": _int(minimum=0),
            },
            ["vehicle_id", "first_tick", "last_tick", "source_frame_count"],
        )),
        "truth_lifecycle_by_vehicle_id": _vehicle_id_map(_obj(
            {
                "vehicle_id": _nonblank_str(),
                "entity_id": _nonblank_str(),
                "first_tick": _int(minimum=0),
                "last_tick": _int(minimum=0),
                "truth_frame_count": _int(minimum=0),
                "retained_outside_runtime_boundary_count": _int(minimum=0),
            },
            [
                "vehicle_id", "entity_id", "first_tick", "last_tick",
                "truth_frame_count", "retained_outside_runtime_boundary_count",
            ],
        )),
    }
    return {"oneOf": [compact, _obj(full_properties, list(full_properties))]}


def _sumo_traffic() -> dict[str, object]:
    """``compiled_summary["sumo_traffic"]`` (:5540-5568), 13 keys."""
    incident_item = _obj(
        {
            "incident_id": _nonblank_str(),
            "episode_event_id": _nonblank_str(),
            "accident_class": _nonblank_str(),
            "start_s": _number(),
            "end_s": _number(),
            "injection_method": {
                "type": "string",
                "pattern": _PATTERN_INCIDENT_INJECTION,
            },
        },
        [
            "incident_id", "episode_event_id", "accident_class", "start_s",
            "end_s", "injection_method",
        ],
    )
    return _obj(
        {
            "enabled": _bool(),
            "source": _sumo_traffic_source(),
            "segment": _segment(),
            "visibility_geometry": _visibility_geometry(),
            "selection": _sumo_selection(),
            "explicit_vehicle_plan": _embedded_explicit_plan(),
            "explicit_selection_audit": _selection_audit(policy_const=True),
            "canonical_asset_policy": _obj(
                {
                    "policy": {"const": "canonical_sumo_vehicle_asset_once_per_lifecycle_v1"},
                    "checked_sumo_vehicle_entity_count": _int(minimum=0),
                    "logical_asset_id_drift": _int(minimum=0),
                },
                [
                    "policy", "checked_sumo_vehicle_entity_count",
                    "logical_asset_id_drift",
                ],
            ),
            "canonical_asset_records": _vehicle_id_map(
                _canonical_asset_record()
            ),
            "required_vehicle_lifecycle_policy": _required_lifecycle_policy(),
            "runtime_spatial_crop": _runtime_spatial_crop(),
            "pedestrian_vehicle_clearance_filter": _obj(
                {
                    "enabled": _bool(),
                    "policy": {"const": "validator_reports_pedestrian_vehicle_clearance_no_sumo_selection_deletion_v1"},
                    "reason": _nonblank_str(),
                },
                ["enabled", "policy", "reason"],
            ),
            "scenario_incidents": _array_of(incident_item),
        },
        [
            "enabled", "source", "segment", "visibility_geometry",
            "selection", "explicit_vehicle_plan", "explicit_selection_audit",
            "canonical_asset_policy", "canonical_asset_records",
            "required_vehicle_lifecycle_policy", "runtime_spatial_crop",
            "pedestrian_vehicle_clearance_filter", "scenario_incidents",
        ],
    )


def _uav_task_type() -> dict[str, str]:
    return {"type": "string", "enum": list(_UAV_TASK_TYPES)}


def _uav_global_flow() -> dict[str, object]:
    """``compiled_summary["uav_global_flow"]`` (:5574-5582), 7 keys."""
    return _obj(
        {
            "enabled": _bool(),
            "source": _obj(
                {
                    "output_dir": _str(),
                    "frames": _str(),
                    "manifest": _str(),
                    "task_plan": _str(),
                    "map_id": _nonblank_str(),
                    "duration_s": _number(),
                    "sample_period_s": _number(),
                    "sample_count": _int(minimum=0),
                    "minimum_active_uavs_required": _int(minimum=0),
                    "baseline_active_uavs": _int(minimum=0),
                    "task_count": _int(minimum=0),
                    "task_count_by_type": _string_id_map(_int(minimum=0)),
                },
                [
                    "output_dir", "frames", "manifest", "task_plan",
                    "map_id", "duration_s", "sample_period_s", "sample_count",
                    "minimum_active_uavs_required", "baseline_active_uavs",
                    "task_count", "task_count_by_type",
                ],
            ),
            "segment": _segment(),
            "selection": _obj(
                {
                    "uav_ids": _array_of(_str()),
                    "entity_ids": {
                        "type": "object",
                        "patternProperties": {
                            _PATTERN_UAV_ID: {
                                "type": "string",
                                "pattern": _PATTERN_UAV_ENTITY_ID,
                            },
                        },
                        "additionalProperties": False,
                    },
                    "task_ids": {
                        "type": "object",
                        "patternProperties": {
                            _PATTERN_UAV_ID: {
                                "type": "string",
                                "pattern": _PATTERN_UAV_TASK_ID,
                            },
                        },
                        "additionalProperties": False,
                    },
                    "mission_type_by_uav_id": {
                        "type": "object",
                        "patternProperties": {
                            _PATTERN_UAV_ID: _uav_task_type(),
                        },
                        "additionalProperties": False,
                    },
                    "selected_count": _int(minimum=0),
                    "active_count_min": _int(minimum=0),
                    "active_count_max": _int(minimum=0),
                    "active_count_mean": _number(),
                    "candidate_count": _int(minimum=0),
                    "observable_candidate_count": _int(minimum=0),
                    "visibility_padding_m": _number(),
                    "min_distance_m_by_uav_id": {
                        "type": "object",
                        "patternProperties": {_PATTERN_UAV_ID: _number()},
                        "additionalProperties": False,
                    },
                    "frames_seen_by_uav_id": {
                        "type": "object",
                        "patternProperties": {
                            _PATTERN_UAV_ID: _int(minimum=0)
                        },
                        "additionalProperties": False,
                    },
                    "motion_span_m_by_uav_id": {
                        "type": "object",
                        "patternProperties": {_PATTERN_UAV_ID: _number()},
                        "additionalProperties": False,
                    },
                    "max_speed_mps_by_uav_id": {
                        "type": "object",
                        "patternProperties": {_PATTERN_UAV_ID: _number()},
                        "additionalProperties": False,
                    },
                },
                [
                    "uav_ids", "entity_ids", "task_ids",
                    "mission_type_by_uav_id", "selected_count",
                    "active_count_min", "active_count_max", "active_count_mean",
                    "candidate_count", "observable_candidate_count",
                    "visibility_padding_m", "min_distance_m_by_uav_id",
                    "frames_seen_by_uav_id", "motion_span_m_by_uav_id",
                    "max_speed_mps_by_uav_id",
                ],
            ),
            "pad_count": _int(minimum=0),
            "runtime_spatial_crop": _runtime_spatial_crop(),
            "camera_capture_roi_policy": {
                "const": "camera_only_while_uav_position_inside_capture_roi_polygon_v1"
            },
        },
        [
            "enabled", "source", "segment", "selection", "pad_count",
            "runtime_spatial_crop", "camera_capture_roi_policy",
        ],
    )


def _scene_occupancy() -> dict[str, object]:
    """``update_episode_manifests_with_scene_occupancy`` (:645-651)."""
    return _obj(
        {
            "authority": {"const": "low_altitude_scene_occupancy_authority_v1"},
            "geometry_authority": {"const": "road_geojson_topology_plus_ue_roadway_mesh"},
            "lane_geometry_source": {"const": "ue_roadway_mesh"},
            "validation_summary": _validation_summary_counts(),
            "artifact": _nonblank_str(),
        },
        [
            "authority", "geometry_authority", "lane_geometry_source",
            "validation_summary", "artifact",
        ],
    )


def _validation_summary_counts() -> dict[str, object]:
    return _obj(
        {
            "ok": _bool(),
            "errors": _array_of(_str()),
            "warnings": _array_of(_str()),
            "error_count": _int(minimum=0),
            "warning_count": _int(minimum=0),
        },
        ["ok", "errors", "warnings", "error_count", "warning_count"],
    )


def _road_semantics() -> dict[str, object]:
    """``SceneRoadSemantics.as_dict()`` (road_semantic_rules.py:133-152)."""
    return _obj(
        {
            "authority": {"const": "road_semantic_rules_v1"},
            "policy": {"const": "static_road_blocking_asset_closes_matched_lane_edges_v1"},
            "episode_id": _nonblank_str(),
            "road_blocking_assets": _array_of(_nonblank_str()),
            "road_closure_clearance_m": _number(),
            "road_blocking_asset_radius_m": _string_id_map(_number()),
            "closed_edge_count": _int(minimum=0),
            "closed_lane_count": _int(minimum=0),
            "closed_source_road_count": _int(minimum=0),
            "closed_edges": _array_of(_nonblank_str()),
            "closed_lanes": _array_of(_nonblank_str()),
            "closed_source_roads": _array_of(_nonblank_str()),
            "closures": _array_of(
                _obj(
                    {
                        "entity_id": _nonblank_str(),
                        "asset_id": _nonblank_str(),
                        "position_enu_m": {
                            "type": "array",
                            "items": _number(),
                            "minItems": 3,
                            "maxItems": 3,
                        },
                        "nearest_lane_clearance_m": _number(),
                        "edge_id": _nonblank_str(),
                        "lane_id": _nonblank_str(),
                        "source_road_id": _nonblank_str(),
                        "direction_role": _nonblank_str(),
                        "physical_lane_index": _int(minimum=0),
                        "matched_lane_distance_m": _number(),
                        "matched_sample_s_m": _number(),
                        "policy": {"const": "static_road_blocking_asset_closes_matched_lane_edges_v1"},
                    },
                    [
                        "entity_id", "asset_id", "position_enu_m",
                        "nearest_lane_clearance_m", "edge_id", "lane_id",
                        "source_road_id", "direction_role",
                        "physical_lane_index", "matched_lane_distance_m",
                        "matched_sample_s_m", "policy",
                    ],
                )
            ),
            "off_road_blocking_assets": _array_of(
                _obj(
                    {
                        # road_semantic_rules.resolve_scene_road_semantics
                        # writes these five members when the nearest lane
                        # exceeds the closure clearance (lines 285-295).
                        "entity_id": _nonblank_str(),
                        "asset_id": {"enum": [
                            "facility.barrier.basic",
                            "prop.roadwork.barrier.v1",
                            "prop.roadwork.construction_fence.v1",
                            "prop.roadwork.traffic_cone.v1",
                        ]},
                        "position_enu_m": {
                            "type": "array", "minItems": 3, "maxItems": 3,
                            "items": {"type": "number",
                                      "x-source-value-types": ["float"]},
                        },
                        "nearest_lane_clearance_m": {
                            "type": "number", "x-source-value-types": ["float"],
                        },
                        "classification": {"const": "off_road_blocking_context"},
                    },
                    ["entity_id", "asset_id", "position_enu_m",
                     "nearest_lane_clearance_m", "classification"],
                )
            ),
        },
        [
            "authority", "policy", "episode_id", "road_blocking_assets",
            "road_closure_clearance_m", "road_blocking_asset_radius_m",
            "closed_edge_count", "closed_lane_count",
            "closed_source_road_count", "closed_edges", "closed_lanes",
            "closed_source_roads", "closures", "off_road_blocking_assets",
        ],
    )


def _filter_stats() -> dict[str, object]:
    """``capture_truth_stats`` (:647-733)."""
    stats: dict[str, object] = {
        "input_truth_entities": _int(minimum=0),
        "output_truth_entities": _int(minimum=0),
        "removed_truth_entities": _int(minimum=0),
        "input_dynamic_pvu_records": _int(minimum=0),
        "output_dynamic_pvu_records": _int(minimum=0),
        "removed_dynamic_pvu_records": _int(minimum=0),
        "dynamic_pvu_records": _int(minimum=0),
        "trajectory_rows": _int(minimum=0),
        "source_truth_entity_id_count": _int(minimum=0),
        "truth_entity_id_count": _int(minimum=0),
        "roster_entities": _int(minimum=0),
        "removed_entity_id_count": _int(minimum=0),
        "removed_entity_ids_sample": _array_of(_str()),
        "removed_jitter_vehicle_records": _int(minimum=0),
        "removed_road_closure_vehicle_records": _int(minimum=0),
        "retained_required_lifecycle_outside_runtime_records": _int(minimum=0),
        "road_closed_vehicle_id_count": _int(minimum=0),
        "road_closed_vehicle_ids_sample": _array_of(_str()),
        "jitter_filtered_vehicle_id_count": _int(minimum=0),
        "jitter_filtered_vehicle_ids_sample": _array_of(_str()),
        "jitter_filter_policy": {"const": "drop_sumo_background_vehicle_with_adjacent_velocity_cosine_below_threshold_v1"},
        "road_semantic_closure_filter_policy": {"const": "drop_vehicle_records_on_scene_closed_road_lanes_v1"},
        "road_semantics": _road_semantics(),
        "jitter_filter_cos_threshold": _number(),
        "jitter_filter_min_speed_mps": _number(),
        "jitter_filter_adjacent_tick_delta": _int(minimum=0),
        "dynamic_filter_rule": _nonblank_str(),
        "source_context_rule": _nonblank_str(),
        "generator_authority_rule": _nonblank_str(),
        "required_lifecycle_preservation_policy": {"const": "required_sumo_vehicle_preserve_source_lifecycle_v1"},
    }
    return _obj(stats, sorted(stats))


def _capture_truth_block() -> dict[str, object]:
    """``capture_truth_payload`` (:657-674) with ``stats`` (:675-679)."""
    return _obj(
        {
            "policy": {"const": "capture_boundary_plus_60m_runtime_truth_filter_v1"},
            "filtering_enabled": _bool(),
            "dynamic_pvu_filtering_enabled": _bool(),
            "coordinate_policy": {"const": "copy_truth_pose_map_enu_without_transform"},
            "visibility_policy": _nonblank_str(),
            "visibility_padding_policy": {"const": "capture_boundary_polygon_expanded_60m_v1"},
            "visibility_padding_m": _number(),
            "visibility_geometry": _visibility_geometry(),
            "formal_roi_bbox_enu_m": {
                "type": "array",
                "items": _number(),
                "minItems": 5,
                "maxItems": 5,
            },
            "stats": _filter_stats(),
        },
        [
            "policy", "filtering_enabled", "dynamic_pvu_filtering_enabled",
            "coordinate_policy", "visibility_policy",
            "visibility_padding_policy", "visibility_padding_m",
            "visibility_geometry", "formal_roi_bbox_enu_m", "stats",
        ],
    )


def formal_episode_manifest_schema() -> dict[str, object]:
    """Closed schema for one complete ``episode_manifest.json`` root, L4."""
    artifacts_keys = [
        "scenario_plan", "global_entity_roster", "truth_frames",
        "event_trace", "event_realization", "trajectories", "weather_meta",
        "dynamic_labels", "episode_manifest", "scene_occupancy_manifest",
    ]
    count_keys = [
        "scenario_plan", "global_entity_roster", "truth_frames",
        "event_trace", "event_realization", "trajectories", "weather_meta",
        "dynamic_labels", "episode_manifest",
    ]
    properties: dict[str, object] = {
        # batch_generate source block (:2539-2551), rewritten by the converter.
        "episode_id": _nonblank_str(),
        "scenario_id": {"enum": list(_L4_PROFILES)},
        "duration_ticks": _int(minimum=0),
        "seed": _int(minimum=0),
        "n_events": _int(minimum=0),
        "n_event_realizations": _int(minimum=0),
        "n_entities": _int(minimum=0),
        "source_event_script_path": _nonblank_str(),
        # The capture filter writes this raw provenance field on records it
        # rewrites; L4-2_v1__seed00 carries it, while other checked roots do not.
        "source_event_script_digest": _str(),
        "source_scene_setup_path": _nonblank_str(),
        "generator": {"const": "Dataset/tools/batch_generate.py"},
        "artifacts": _obj(
            {
                "scenario_plan": _nonblank_str(),
                "global_entity_roster": _nonblank_str(),
                "truth_frames": _nonblank_str(),
                "event_trace": _nonblank_str(),
                "event_realization": _nonblank_str(),
                "trajectories": _nonblank_str(),
                "weather_meta": _nonblank_str(),
                "dynamic_labels": _nonblank_str(),
                "episode_manifest": _nonblank_str(),
                "scene_occupancy_manifest": _nonblank_str(),
            },
            artifacts_keys,
        ),
        "vehicle_source_policy": {"const": "explicit_episode_traffic_schedule_only_v2"},
        "sumo_explicit_vehicle_plan": _sumo_explicit_vehicle_plan_summary(),
        # Converter identity/contract keys (:5676-5682).
        "map_id": _nonblank_str(),
        "capture_boundary_id": _nonblank_str(),
        "uav_crosses_boundary": _bool(),
        "inspect_observes_boundary": _bool(),
        "pad_boundary_policy": _obj(
            {
                "default": _nonblank_str(),
                "inside_required_for": _array_of(_nonblank_str()),
            },
            ["default", "inside_required_for"],
        ),
        "runtime_spatial_crop": _runtime_spatial_crop(),
        "source_vehicle_authority": _source_vehicle_authority(),
        "sumo_traffic": _sumo_traffic(),
        "uav_global_flow": _uav_global_flow(),
        # Converter generation payload (:5691-5694) plus capture-filter
        # additions (filter_render_ready_truth_for_capture.py:804-820).
        "generation": _obj(
            {
                "generator": {"const": "Dataset/tools/convert_to_render_ready.py"},
                "source_episode_dir": _nonblank_str(),
                "source_contract_hash": _str(),
                "truth_filter_source_root": _nonblank_str(),
                "truth_filter_contract": {"const": "capture_boundary_plus_60m_runtime_truth_filter_v1"},
                "capture_truth_sync_policy": {"const": "capture_boundary_plus_60m_runtime_truth_filter_v1"},
                "trajectory_source": {"const": "capture_boundary_plus_60m_filtered_truth_frames"},
                "trajectory_contract": {"const": "trajectory_rows_rewritten_from_filtered_truth_v1"},
            },
            [
                "generator", "source_episode_dir",
                "truth_filter_source_root", "truth_filter_contract",
                "capture_truth_sync_policy", "trajectory_source",
                "trajectory_contract",
            ],
        ),
        "record_counts": _obj(
            {key: _int(minimum=0) for key in count_keys},
            count_keys,
        ),
        "canonical_record_counts": _obj(
            {key: _int(minimum=0) for key in count_keys},
            count_keys,
        ),
        "node_counts": _obj(
            {
                "all_nodes": _int(minimum=0),
                "dynamic_nodes": _int(minimum=0),
                "static_nodes": _int(minimum=0),
            },
            ["all_nodes", "dynamic_nodes", "static_nodes"],
        ),
        "canonical_artifacts": _obj(
            {key: _nonblank_str() for key in artifacts_keys},
            artifacts_keys,
        ),
        "time_range": _obj(
            {
                "tick_start": _int(minimum=0),
                "tick_end": _int(minimum=0),
                "sim_time_start": _number(),
                "sim_time_end": _number(),
            },
            ["tick_start", "tick_end", "sim_time_start", "sim_time_end"],
        ),
        "validation_summary": _obj(
            {
                "ok": _bool(),
                "errors": _array_of(_str()),
                "warnings": _array_of(_str()),
            },
            ["ok", "errors", "warnings"],
        ),
        "scene_occupancy": _scene_occupancy(),
        "capture_visible_truth_filter": _capture_truth_block(),
        "capture_truth_sync": _capture_truth_block(),
    }
    # The converter writes generation and the capture filter extends it.
    base_required = [
        "episode_id", "scenario_id", "duration_ticks", "seed", "n_events",
        "n_event_realizations", "n_entities", "source_event_script_path",
        "source_scene_setup_path", "generator", "artifacts",
        "vehicle_source_policy", "sumo_explicit_vehicle_plan", "map_id",
        "capture_boundary_id", "uav_crosses_boundary",
        "inspect_observes_boundary", "pad_boundary_policy",
        "runtime_spatial_crop", "source_vehicle_authority", "sumo_traffic",
        "uav_global_flow", "record_counts", "canonical_record_counts",
        "node_counts", "canonical_artifacts", "time_range",
        "validation_summary", "scene_occupancy",
        "capture_visible_truth_filter", "capture_truth_sync", "generation",
    ]
    return _obj(properties, base_required)
