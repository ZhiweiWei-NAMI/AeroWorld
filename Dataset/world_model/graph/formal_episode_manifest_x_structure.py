"""Closed record structure for the formal episode manifest, X layer.

Source family ``formal_episode_manifest``; declared X profiles
X1_rain_to_c2loss_to_forced_landing, X2_gnss_spoof_to_geofence_violation,
X3_pedestrian_fall_to_emergency_response, X4_fog_to_uav_conflict,
X5_comm_failure_to_pad_contention, X6_crowd_evacuation_to_airspace_lockdown,
each with seeds 00-02: 18 present records, 0 missing at the
``aw_data/render_ready_episodes_capture_filtered`` transfer boundary.

Record identity: the record root is the whole JSON document of one
episode ``episode_manifest.json`` (18 files observed at the transfer
boundary, one per episode; the family has no per-row JSONL split), and
every root carries the profile in ``scenario_id``, so the X layer is
closed as one root schema with the profile id enum-enumerated from the
producer registry.

Producer chain, all references re-read this session:
- ``Dataset/tools/batch_generate.py`` authors the source manifest
  (episode_id, scenario_id, duration_ticks, seed, n_events,
  n_event_realizations, n_entities, source_event_script_path,
  source_scene_setup_path, generator, map_id, sumo_explicit_vehicle_plan
  with review/vehicle_count/minimum_vehicle_count).
- ``Dataset/tools/convert_to_render_ready.py`` ``convert_episode``
  deep-copies it (:5673) and ``update``s the render keys (:5674-5726:
  capture_boundary_id, uav_crosses_boundary, inspect_observes_boundary,
  pad_boundary_policy from the source contract, runtime_spatial_crop,
  source_vehicle_authority :5463-5478 (its ``explicit_vehicle_plan``
  copy drops the source head's ``review``/``minimum_vehicle_count`` and
  adds episode_id/vehicle_ids/selection_audit), sumo_traffic/
  uav_global_flow from ``compiled_summary`` :5540-5582, generation
  :5691, record_counts :5633-5643, node_counts :5697,
  artifacts/canonical_artifacts :5644-5717, time_range :5718,
  validation_summary :5724) then writes ``episode_manifest.json``
  (:5743) and attaches the scene-occupancy
  audit via ``audit_and_attach_episode`` (:5744 →
  ``scene_occupancy.update_episode_manifests_with_scene_occupancy``
  :635-663, which adds exactly the five-key root ``scene_occupancy``
  block (:645-651) and the ``scene_occupancy_manifest`` artifact slots
  (:652-657); the service-facility repair payloads of
  ``repair_service_facility_placements`` (:5437) live on roster rows and
  in ``scene_occupancy_manifest.json``, never as a manifest key).
- ``Dataset/tools/filter_render_ready_truth_for_capture.py`` ``update_manifest``
  (:576-623) refreshes record_counts/canonical_record_counts (:589-594),
  recomputes node_counts (:595-600), extends ``generation`` with the
  truth-filter keys (:601-607), writes the identical
  ``capture_visible_truth_filter``/``capture_truth_sync`` payloads
  (:608-621) and rewrites both artifact maps to the capture-filtered root
  (:504-518); ``filter_episode`` then re-attaches the scene-occupancy
  audit (:847).  This is the final writer of every formal byte.

X-layer profile evidence: the converter registry drives
``Dataset/scenarios/X_cross_layer/<profile>/event_script.json`` for exactly
the six profiles above; the six-profile enum below is the profile id set,
and every record's ``scenario_id`` is one of them.  ``capture_boundary_id``
is producer-written per record (the checked seed00 roots write
``capture_boundary_<profile>`` except X6, which writes
``nfz_x6_lockdown``), so the schema pins no boundary-id pattern.

Type-system decisions:
- The current converter writes eight fields per canonical asset record.
  Frozen transfer roots have three additional real data fields:
  ``asset_conflict_count`` (string), ``asset_conflict_fields`` (string),
  and ``asset_conflicts[]`` (closed records with ``field``,
  ``logical_asset_id``, ``resolved_by``).  Both complete shapes are closed.
- Booleans stay boolean and never merge with integers (``seed`` is
  integer 0-2; ``uav_crosses_boundary``/``inspect_observes_boundary``/
  ``enabled``/``ok``/``filtering_enabled``/``dynamic_pvu_filtering_enabled``/
  ``source_presence_required`` are booleans); per-key numeric/string
  leaves keep one raw type.
- Producer-written dynamic maps (``selection.entity_ids``,
  ``selection.task_ids``/``mission_type_by_uav_id``, the per-id metric
  maps and ``sumo_traffic.canonical_asset_records``)
  are closed with a dict-typed ``additionalProperties`` member schema —
  the compiler's closed-boundary form for typed dynamic maps — plus an
  anchored ``propertyNames`` pattern; every member value is validated
  (e.g. ``canonical_asset_records`` members against the full closed
  asset-record schema).
- ``oneOf`` separates the two canonical asset shapes and the three/five
  key lifecycle policy shapes from the converter's capture-required branch.
  The two explicit-plan forms are distinct keys
  (root ``sumo_explicit_vehicle_plan`` vs
  ``source_vehicle_authority.explicit_vehicle_plan``), each closed to
  its own keyset, instead of a variant under one key.  List keys whose
  every observed instance is empty (``scenario_incidents``,
  ``road_semantics.closures``, ``off_road_blocking_assets``) keep one
  closed item record without instance verification of that item.

Authoritative manifests: frozen formal records at
``aw_data/render_ready_episodes_capture_filtered``.  The producer also
nominally defaults to ``Dataset/render_ready_episodes``, which is not the
formal capture handoff (AGENTS.md boundary 5) and carries no X-layer
declaration — it is not a record branch of this grammar.

Publication exclusion: frozen X roots may carry
``generation.source_contract_hash`` and the root
``source_event_script_digest``.  The current converter and capture filter
write neither field, so the frozen roots provide structural authority and
both raw fields are optional.  The shared publication projection already
excludes the exact paths ``("generation", "source_contract_hash")`` and
``("source_event_script_digest",)``; no global rule change is required.
``validation_summary``/``scene_occupancy.validation_summary`` are producer
result payloads, not integrity metadata, and stay ordinary data fields.

Boundary: this module exports the pure schema constructor only; it scans
no data and compiles nothing at import.  ``formal_episode_manifest`` is
not registered in ``structure_closure.py`` (it is currently
``source_schema_pending`` in the inventory); registration/dispatch wiring
belongs to G1, and the G2 runtime source-string exclusion work is a
separate integration dependency.
"""

from __future__ import annotations

__all__ = ["formal_episode_manifest_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

_X_PROFILES = (
    "X1_rain_to_c2loss_to_forced_landing",
    "X2_gnss_spoof_to_geofence_violation",
    "X3_pedestrian_fall_to_emergency_response",
    "X4_fog_to_uav_conflict",
    "X5_comm_failure_to_pad_contention",
    "X6_crowd_evacuation_to_airspace_lockdown",
)


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _int_leaf() -> dict:
    return {"type": "integer", "x-source-value-types": ["int"]}


def _string_array() -> dict:
    return {"type": "array", "items": {"type": "string"}}


def _number_array() -> dict:
    return {"type": "array", "items": {"type": "number"}}


def _id_map(member: dict) -> dict:
    # Closed producer-written dynamic map (per-id slots; ids are producer
    # evidence, no id regex is pinned).  The dict-typed
    # ``additionalProperties`` is the compiler's closed-boundary form for
    # a dynamic map: every member value is validated against ``member``
    # and every key against the anchored propertyNames pattern.
    return {
        "type": "object",
        "additionalProperties": member,
        "propertyNames": {"type": "string", "pattern": "^.+$"},
    }


def _closed(**properties: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _profile_const() -> dict:
    # Single-type string enum over the six producer-registered profile
    # ids.  The compiler rejects const on non-scalar types
    # (structure_schema._effective_type: container const/enum requires an
    # explicit structure), and _strict_enum compares each member with
    # strict type + equality.
    return {"enum": list(_X_PROFILES)}


def _seed_label() -> dict:
    return {"type": "string", "pattern": "^seed[0-9]{2}$"}


def _artifact_map() -> dict:
    # The ten artifact slots written by the converter and the capture
    # filter (update_artifact_paths) with repo-relative targets.
    return _closed(
        episode_manifest=_identifier(),
        dynamic_labels=_identifier(),
        event_realization=_identifier(),
        event_trace=_identifier(),
        global_entity_roster=_identifier(),
        scenario_plan=_identifier(),
        scene_occupancy_manifest=_identifier(),
        trajectories=_identifier(),
        truth_frames=_identifier(),
        weather_meta=_identifier(),
    )


def _record_counts() -> dict:
    # record_counts / canonical_record_counts: the converter writes the
    # nine generator slots; the capture filter refreshes truth_frames,
    # trajectories and global_entity_roster and keeps the rest.
    return _closed(
        dynamic_labels=_int_leaf(),
        episode_manifest=_int_leaf(),
        event_realization=_int_leaf(),
        event_trace=_int_leaf(),
        global_entity_roster=_int_leaf(),
        scenario_plan=_int_leaf(),
        trajectories=_int_leaf(),
        truth_frames=_int_leaf(),
        weather_meta=_int_leaf(),
    )


def _node_counts() -> dict:
    return _closed(
        all_nodes=_int_leaf(),
        dynamic_nodes=_int_leaf(),
        static_nodes=_int_leaf(),
    )


def _time_range() -> dict:
    return _closed(
        sim_time_end=_float_leaf(),
        sim_time_start=_float_leaf(),
        tick_end=_int_leaf(),
        tick_start=_int_leaf(),
    )


def _validation_summary() -> dict:
    # The converter writes the root audit payload with empty
    # error/warning lists (:5724); the attached scene-occupancy summaries
    # carry the nonempty producer warning/error strings.
    return _closed(
        ok={"type": "boolean"},
        errors=_string_array(),
        warnings=_string_array(),
    )


def _runtime_spatial_crop() -> dict:
    return _closed(
        bbox_enu_m={
            "type": "array", "minItems": 4, "maxItems": 4,
            "items": {"type": "number"},
        },
        capture_boundary_id=_identifier(),
        dynamic_frame_rule=_identifier(),
        enabled={"type": "boolean"},
        expanded_boundary_padding_m=_float_leaf(),
        policy=_identifier(),
        uav_camera_rule=_identifier(),
    )


def _pad_boundary_policy() -> dict:
    # Written verbatim from the source contract
    # (convert_to_render_ready.py :5508); the formal X events author the
    # inside_required_for member.
    return _closed(
        default=_identifier(),
        inside_required_for=_string_array(),
    )


def _traffic_profile() -> dict:
    return _closed(
        accepted_background_vehicle_target=_int_leaf(),
        background_vehicles_per_slot=_int_leaf(),
        core_crossing_duration_ticks=_int_leaf(),
        direction_distribution=_closed(
            cross=_float_leaf(),
            inbound=_float_leaf(),
            outbound=_float_leaf(),
        ),
        flow_mode=_identifier(),
        profile_id=_identifier(),
        schedule_policy=_identifier(),
        seed_label=_seed_label(),
        seed_semantics=_closed(
            seed00=_identifier(),
            seed01=_identifier(),
            seed02=_identifier(),
        ),
        slot_count=_int_leaf(),
        slot_duration_ticks=_int_leaf(),
        slot_entry_jitter_ticks={
            "type": "array", "items": {"type": "integer"},
        },
        target_speed_mps=_float_leaf(),
    )


def _seed_profile() -> dict:
    return _closed(
        direction_bias=_identifier(),
        profile_id=_identifier(),
        seed_index=_int_leaf(),
        seed_label=_seed_label(),
    )


def _source_explicit_plan() -> dict:
    # Root ``sumo_explicit_vehicle_plan``: the batch_generate-authored
    # head, deep-copied unchanged through converter and capture filter.
    # Frozen X1 evidence (:25-69); the capture-filter deepcopy preserves
    # the other 17 records' heads the same way.  No episode_id,
    # vehicle_ids or selection audit on this form.
    return _closed(
        minimum_vehicle_count=_int_leaf(),
        review=_identifier(),
        schema={"const": "aero_explicit_sumo_vehicle_plan_v2"},
        seed_profile=_seed_profile(),
        traffic_profile=_traffic_profile(),
        vehicle_count=_int_leaf(),
    )


def _authority_explicit_plan() -> dict:
    # The converter's 7-key explicit-plan copy (schema/episode_id/
    # vehicle_count/vehicle_ids/selection_audit/seed_profile/
    # traffic_profile); it carries neither ``review`` nor
    # ``minimum_vehicle_count``.  Used by both carriers:
    # ``source_vehicle_authority.explicit_vehicle_plan`` (converter
    # :5470-5478) and ``sumo_traffic.explicit_vehicle_plan``
    # (compiled_summary payload; record-verified keyset).
    return _closed(
        episode_id=_identifier(),
        schema={"const": "aero_explicit_sumo_vehicle_plan_v2"},
        seed_profile=_seed_profile(),
        selection_audit=_selection_audit(),
        traffic_profile=_traffic_profile(),
        vehicle_count=_int_leaf(),
        vehicle_ids=_string_array(),
    )


def _selection_audit() -> dict:
    return _closed(
        policy=_identifier(),
        planned_vehicle_count=_int_leaf(),
        planned_semantic_vehicle_count=_int_leaf(),
        planned_source_presence_required_vehicle_count=_int_leaf(),
        planned_capture_required_vehicle_count=_int_leaf(),
        realized_vehicle_count=_int_leaf(),
        selected_runtime_visible_vehicle_count=_int_leaf(),
        extra_realized_background_vehicle_count=_int_leaf(),
        extra_realized_background_vehicle_ids=_string_array(),
        missing_background_vehicle_count=_int_leaf(),
        missing_background_vehicle_ids=_string_array(),
        extra_visible_background_vehicle_count=_int_leaf(),
        extra_visible_background_vehicle_ids=_string_array(),
        missing_visible_background_vehicle_count=_int_leaf(),
        missing_visible_background_vehicle_ids=_string_array(),
    )


def _asset_conflict() -> dict:
    # The current converter writes eight keys.  The frozen transfer roots
    # additionally carry three real conflict fields; each shape is closed.
    current = _closed(
        authority=_identifier(),
        explicit_type_id=_identifier(),
        logical_asset_id=_identifier(),
        source_entity_id={"type": "string"},
        source_presence_required={"type": "boolean"},
        source_roster_logical_asset_id={"type": "string"},
        sumo_record_type_id={"type": "string", "minLength": 1},
        sumo_type_id={"enum": [
            "aero_ambulance", "aero_delivery", "aero_emergency",
            "aero_passenger", "aero_police",
        ]},
    )
    frozen = _closed(
        **current["properties"],
        asset_conflict_count={"type": "string", "pattern": "^[0-9]+$"},
        asset_conflict_fields={"type": "string"},
        asset_conflicts={
            "type": "array",
            "items": _closed(
                field=_identifier(),
                logical_asset_id=_identifier(),
                resolved_by=_identifier(),
            ),
        },
    )
    return {"oneOf": [current, frozen]}


def _sumo_source() -> dict:
    return _closed(
        capture_duration_s=_float_leaf(),
        capture_start_s=_float_leaf(),
        coordinate_mapper=_closed(
            matrix={
                # Identity 2x2 + translation rows (record :349-363).
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
            },
            max_error_m=_float_leaf(),
            mean_error_m=_float_leaf(),
            pair_count=_int_leaf(),
            source=_identifier(),
        ),
        duration_s=_float_leaf(),
        frames=_identifier(),
        incident_plan=_identifier(),
        manifest=_identifier(),
        map_id=_identifier(),
        max_vehicles=_int_leaf(),
        net_xml=_identifier(),
        output_dir=_identifier(),
        road_derived_network_manifest=_identifier(),
        sample_count=_int_leaf(),
        sample_every_ticks=_int_leaf(),
        sample_period_s=_float_leaf(),
        step_length_s=_float_leaf(),
        time_axis_policy=_identifier(),
        traffic_light_count=_int_leaf(),
        warmup_s=_float_leaf(),
    )


def _sumo_segment() -> dict:
    return _closed(
        duration_s=_float_leaf(),
        phase_name={"const": "episode_local_vehicle_authority_after_warmup"},
        seed_index=_int_leaf(),
        seed_label=_seed_label(),
        segment_end_s=_float_leaf(),
        segment_start_s=_float_leaf(),
    )


def _visibility_geometry() -> dict:
    return _closed(
        altitude_m=_float_leaf(),
        capture_polygon_points=_int_leaf(),
        footprint_half_height_m=_float_leaf(),
        footprint_half_width_m=_float_leaf(),
        hfov_deg=_float_leaf(),
        inspect_route_points=_int_leaf(),
        padding_m=_float_leaf(),
        vfov_deg=_float_leaf(),
    )


def _sumo_selection() -> dict:
    # Keyset read from the X1 record selection block (:388-862): the
    # per-vehicle maps min_distance/frames_seen/motion_span/max_speed
    # carry one member per selected vehicle id.
    return _closed(
        candidate_count=_int_leaf(),
        entity_ids=_id_map(_identifier()),
        expanded_to_nearest={"type": "boolean"},
        frames_seen_by_vehicle_id=_id_map(_int_leaf()),
        max_visible_vehicle_target=_int_leaf(),
        max_speed_mps_by_vehicle_id=_id_map(_float_leaf()),
        min_distance_m_by_vehicle_id=_id_map(_float_leaf()),
        min_visible_vehicle_target=_int_leaf(),
        motion_span_m_by_vehicle_id=_id_map(_float_leaf()),
        scenario_vehicle_ids=_string_array(),
        selected_count=_int_leaf(),
        vehicle_ids=_string_array(),
    )


def _required_lifecycle_policy() -> dict:
    compact = _closed(
        clipped_required_vehicle_count=_int_leaf(),
        policy=_identifier(),
        required_vehicle_count=_int_leaf(),
    )
    full = _closed(
        **compact["properties"],
        source_lifecycle_by_vehicle_id=_id_map(_closed(
            first_tick=_int_leaf(),
            last_tick=_int_leaf(),
            source_frame_count=_int_leaf(),
            vehicle_id=_identifier(),
        )),
        truth_lifecycle_by_vehicle_id=_id_map(_closed(
            entity_id=_identifier(),
            first_tick=_int_leaf(),
            last_tick=_int_leaf(),
            retained_outside_runtime_boundary_count=_int_leaf(),
            truth_frame_count=_int_leaf(),
            vehicle_id=_identifier(),
        )),
    )
    return {"oneOf": [compact, full]}


def _pedestrian_vehicle_clearance_filter() -> dict:
    return _closed(
        enabled={"type": "boolean"},
        policy=_identifier(),
        reason=_identifier(),
    )


def _uav_source() -> dict:
    return _closed(
        baseline_active_uavs=_int_leaf(),
        duration_s=_float_leaf(),
        frames=_identifier(),
        manifest=_identifier(),
        map_id=_identifier(),
        minimum_active_uavs_required=_int_leaf(),
        output_dir=_identifier(),
        sample_count=_int_leaf(),
        sample_period_s=_float_leaf(),
        task_count=_int_leaf(),
        task_count_by_type=_closed(
            edge_compute_relay=_int_leaf(),
            incident_response_inspection=_int_leaf(),
            infrastructure_inspection=_int_leaf(),
            intersection_inspect=_int_leaf(),
            logistics_delivery=_int_leaf(),
            pad_patrol=_int_leaf(),
        ),
        task_plan=_identifier(),
    )


def _uav_selection() -> dict:
    # Keyset read from the X1 record selection block (:2194-2313): the
    # per-uav maps min_distance/frames_seen/motion_span/max_speed carry
    # one member per selected uav id.
    return _closed(
        active_count_max=_int_leaf(),
        active_count_mean=_float_leaf(),
        active_count_min=_int_leaf(),
        candidate_count=_int_leaf(),
        entity_ids=_id_map(_identifier()),
        frames_seen_by_uav_id=_id_map(_int_leaf()),
        max_speed_mps_by_uav_id=_id_map(_float_leaf()),
        min_distance_m_by_uav_id=_id_map(_float_leaf()),
        mission_type_by_uav_id=_id_map(_identifier()),
        motion_span_m_by_uav_id=_id_map(_float_leaf()),
        observable_candidate_count=_int_leaf(),
        selected_count=_int_leaf(),
        task_ids=_id_map(_identifier()),
        uav_ids=_string_array(),
        visibility_padding_m=_float_leaf(),
    )


def _filter_stats() -> dict:
    # Exactly the 29-key keyset of the record's ``stats`` block
    # (:2475-2497): everything except the nested ``road_semantics``
    # object.
    stats = {
        "dynamic_filter_rule": _identifier(),
        "dynamic_pvu_records": _int_leaf(),
        "generator_authority_rule": _identifier(),
        "input_dynamic_pvu_records": _int_leaf(),
        "input_truth_entities": _int_leaf(),
        "jitter_filter_adjacent_tick_delta": _int_leaf(),
        "jitter_filter_cos_threshold": _float_leaf(),
        "jitter_filter_min_speed_mps": _float_leaf(),
        "jitter_filter_policy": _identifier(),
        "jitter_filtered_vehicle_id_count": _int_leaf(),
        "jitter_filtered_vehicle_ids_sample": _string_array(),
        "output_dynamic_pvu_records": _int_leaf(),
        "output_truth_entities": _int_leaf(),
        "removed_dynamic_pvu_records": _int_leaf(),
        "removed_entity_id_count": _int_leaf(),
        "removed_entity_ids_sample": _string_array(),
        "removed_jitter_vehicle_records": _int_leaf(),
        "removed_road_closure_vehicle_records": _int_leaf(),
        "removed_truth_entities": _int_leaf(),
        "required_lifecycle_preservation_policy": _identifier(),
        "retained_required_lifecycle_outside_runtime_records": _int_leaf(),
        "road_closed_vehicle_id_count": _int_leaf(),
        "road_closed_vehicle_ids_sample": _string_array(),
        "road_semantic_closure_filter_policy": _identifier(),
        "roster_entities": _int_leaf(),
        "source_context_rule": _identifier(),
        "source_truth_entity_id_count": _int_leaf(),
        "trajectory_rows": _int_leaf(),
        "truth_entity_id_count": _int_leaf(),
    }
    return _closed(
        **stats,
        road_semantics={
            # The nested road-semantics block of the stats payload
            # (:2498-2523): SceneRoadSemantics.as_dict() verbatim
            # (road_semantic_rules.py :133-152).
            "type": "object",
            "additionalProperties": False,
            "required": [
                "authority", "closed_edge_count", "closed_edges",
                "closed_lane_count", "closed_lanes",
                "closed_source_road_count", "closed_source_roads",
                "closures", "episode_id", "off_road_blocking_assets",
                "policy", "road_blocking_asset_radius_m",
                "road_blocking_assets", "road_closure_clearance_m",
            ],
            "properties": {
                "authority": _identifier(),
                "policy": _identifier(),
                "episode_id": {"type": "string"},
                "road_blocking_assets": _string_array(),
                "road_closure_clearance_m": _float_leaf(),
                "road_blocking_asset_radius_m": _id_map(_float_leaf()),
                "closed_edge_count": _int_leaf(),
                "closed_lane_count": _int_leaf(),
                "closed_source_road_count": _int_leaf(),
                "closed_edges": _string_array(),
                "closed_lanes": _string_array(),
                "closed_source_roads": _string_array(),
                "closures": {
                    # RoadClosureRecord.as_dict() items
                    # (road_semantic_rules.py :86-100); the X corpus
                    # writes an empty list on every record — a nonempty
                    # member is still-without-instance-verification.
                    "type": "array",
                    "items": _closed(
                        asset_id=_identifier(),
                        direction_role={"type": "string"},
                        edge_id={"type": "string"},
                        entity_id={"type": "string"},
                        lane_id={"type": "string"},
                        matched_lane_distance_m=_float_leaf(),
                        matched_sample_s_m=_float_leaf(),
                        nearest_lane_clearance_m=_float_leaf(),
                        physical_lane_index=_int_leaf(),
                        policy=_identifier(),
                        position_enu_m=_number_array(),
                        source_road_id={"type": "string"},
                    ),
                },
                "off_road_blocking_assets": {
                    # resolve_scene_road_semantics off-road member
                    # (:287-295), also without an X instance.
                    "type": "array",
                    "items": _closed(
                        asset_id=_identifier(),
                        classification={"const": "off_road_blocking_context"},
                        entity_id={"type": "string"},
                        nearest_lane_clearance_m=_float_leaf(),
                        position_enu_m=_number_array(),
                    ),
                },
            },
        },
    )


def _capture_sync_payload() -> dict:
    # The writer emits capture_visible_truth_filter and capture_truth_sync
    # as deepcopy-identical payloads (update_manifest :608-621).
    return _closed(
        coordinate_policy=_identifier(),
        dynamic_pvu_filtering_enabled={"type": "boolean"},
        filtering_enabled={"type": "boolean"},
        formal_roi_bbox_enu_m={
            "type": "array", "items": _float_leaf(),
            "minItems": 5, "maxItems": 5,
        },
        policy=_identifier(),
        stats=_filter_stats(),
        visibility_geometry=_visibility_geometry(),
        visibility_padding_m=_float_leaf(),
        visibility_padding_policy=_identifier(),
        visibility_policy=_identifier(),
    )


def _generation() -> dict:
    schema = _closed(
        capture_truth_sync_policy=_identifier(),
        generator=_identifier(),
        source_contract_hash={
            # Integrity metadata: exact publication-excluded path
            # ("generation", "source_contract_hash") already present in
            # structure_schema._PUBLICATION_EXCLUDED_PATHS.
            "type": "string", "x-source-value-types": ["string"],
        },
        source_episode_dir=_identifier(),
        trajectory_contract=_identifier(),
        trajectory_source=_identifier(),
        truth_filter_contract=_identifier(),
        truth_filter_source_root=_identifier(),
    )
    schema["required"].remove("source_contract_hash")
    return schema


def _scene_occupancy() -> dict:
    return _closed(
        artifact=_identifier(),
        authority=_identifier(),
        geometry_authority=_identifier(),
        lane_geometry_source=_identifier(),
        validation_summary={
            "type": "object", "additionalProperties": False,
            "required": ["error_count", "errors", "ok", "warning_count",
                         "warnings"],
            "properties": {
                "error_count": _int_leaf(),
                "errors": _string_array(),
                "ok": {"type": "boolean"},
                "warning_count": _int_leaf(),
                "warnings": _string_array(),
            },
        },
    )


def formal_episode_manifest_schema() -> dict:
    """Return the closed record schema for one X-layer manifest root."""
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_episode_manifest_x.record.schema.json",
        "title": "AERO_WORLD formal_episode_manifest X record schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "artifacts", "canonical_artifacts", "canonical_record_counts",
            "capture_boundary_id", "capture_truth_sync",
            "capture_visible_truth_filter", "duration_ticks", "episode_id",
            "generator", "generation", "inspect_observes_boundary", "map_id",
            "n_entities", "n_event_realizations", "n_events",
            "node_counts", "pad_boundary_policy",
            "record_counts", "runtime_spatial_crop", "scenario_id",
            "scene_occupancy", "seed", "source_event_script_path",
            "source_scene_setup_path", "source_vehicle_authority",
            "sumo_explicit_vehicle_plan", "sumo_traffic", "time_range",
            "uav_crosses_boundary",
            "uav_global_flow", "validation_summary", "vehicle_source_policy",
        ],
        "properties": {
            "artifacts": _artifact_map(),
            "canonical_artifacts": _artifact_map(),
            "canonical_record_counts": _record_counts(),
            "capture_boundary_id": _identifier(),
            "capture_truth_sync": _capture_sync_payload(),
            "capture_visible_truth_filter": _capture_sync_payload(),
            "duration_ticks": _int_leaf(),
            "episode_id": _identifier(),
            "generator": {
                # Root-level sibling: batch_generate.py authors it, the
                # converter deep-copies the manifest head unchanged, the
                # capture filter rewrites it (:580); confirmed on every
                # checked root.  Distinct from generation.generator
                # ("Dataset/tools/convert_to_render_ready.py").
                "type": "string", "x-source-value-types": ["string"],
            },
            "generation": _generation(),
            "inspect_observes_boundary": {"type": "boolean"},
            "map_id": _identifier(),
            "n_entities": _int_leaf(),
            "n_event_realizations": _int_leaf(),
            "n_events": _int_leaf(),
            "node_counts": _node_counts(),
            "pad_boundary_policy": _pad_boundary_policy(),
            "record_counts": _record_counts(),
            "runtime_spatial_crop": _runtime_spatial_crop(),
            "scenario_id": _profile_const(),
            "scene_occupancy": _scene_occupancy(),
            "seed": {"type": "integer", "minimum": 0},
            "source_event_script_digest": {
                # Frozen roots may carry this integrity field; current
                # converter and capture-filter writers do not add it.
                # X1 seed00 lacks it, so it remains optional at the root.
                "type": "string", "x-source-value-types": ["string"],
            },
            "source_event_script_path": _identifier(),
            "source_scene_setup_path": _identifier(),
            "source_vehicle_authority": _closed(
                authority=_identifier(),
                enabled={"type": "boolean"},
                event_reference_policy=_identifier(),
                explicit_vehicle_plan=_authority_explicit_plan(),
                policy=_identifier(),
                reason=_identifier(),
                replaced_source_vehicle_count=_int_leaf(),
                replaced_source_vehicle_counts_by_role=_closed(
                    semantic_background_vehicle=_int_leaf(),
                ),
                replaced_source_vehicle_ids=_string_array(),
                vehicle_source_policy=_identifier(),
            ),
            "sumo_explicit_vehicle_plan": _source_explicit_plan(),
            "sumo_traffic": _closed(
                canonical_asset_policy=_closed(
                    checked_sumo_vehicle_entity_count=_int_leaf(),
                    logical_asset_id_drift=_int_leaf(),
                    policy=_identifier(),
                ),
                canonical_asset_records=_id_map(_asset_conflict()),
                enabled={"type": "boolean"},
                explicit_selection_audit=_selection_audit(),
                explicit_vehicle_plan=_authority_explicit_plan(),
                pedestrian_vehicle_clearance_filter=_pedestrian_vehicle_clearance_filter(),
                required_vehicle_lifecycle_policy=_required_lifecycle_policy(),
                runtime_spatial_crop=_runtime_spatial_crop(),
                scenario_incidents={
                    # The three-key projection the converter writes per
                    # incident (:5557-5566) plus the manifest's own id
                    # slots (traffic manifest rows; the X corpus writes
                    # an empty list on every record — a nonempty member
                    # is still-without-instance-verification).
                    "type": "array",
                    "items": _closed(
                        accident_class=_identifier(),
                        episode_event_id=_identifier(),
                        incident_id=_identifier(),
                        injection_method=_identifier(),
                        end_s=_float_leaf(),
                        start_s=_float_leaf(),
                    ),
                },
                selection=_sumo_selection(),
                segment=_sumo_segment(),
                source=_sumo_source(),
                visibility_geometry=_visibility_geometry(),
            ),
            "time_range": _time_range(),
            "uav_crosses_boundary": {"type": "boolean"},
            "uav_global_flow": _closed(
                camera_capture_roi_policy=_identifier(),
                enabled={"type": "boolean"},
                pad_count=_int_leaf(),
                runtime_spatial_crop=_runtime_spatial_crop(),
                segment=_closed(
                    duration_s=_float_leaf(),
                    phase_name=_identifier(),
                    seed_index=_int_leaf(),
                    seed_label=_seed_label(),
                    segment_end_s=_float_leaf(),
                    segment_start_s=_float_leaf(),
                ),
                selection=_uav_selection(),
                source=_uav_source(),
            ),
            "validation_summary": _validation_summary(),
            "vehicle_source_policy": _identifier(),
        },
    }
