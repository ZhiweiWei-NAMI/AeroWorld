"""Closed record structure for the formal episode manifest, L6 layer.

Source family ``formal_episode_manifest``; declared L6 profiles
L6-1_v1, L6-1_v2, L6-2_v1, L6-2_v2, L6-3_v1, L6-3_v2, L6-4_v1, L6-4_v2,
L6-5_v1, L6-5_v2, each with seeds 00-02: 30 present records, 0 missing at
the ``aw_data/render_ready_episodes_capture_filtered`` transfer boundary.
The family is ``source_schema_pending`` in the declared source inventory
(``field_catalog.declared_source_inventory()``); this module supplies the
producer grammar that registration (G1 wiring) will attach.

Record identity: the record root is the whole JSON document of one
episode ``episode_manifest.json`` (one file per episode; the family has
no per-row JSONL split), and every root carries the profile in
``scenario_id``, so the L6 layer is closed as one root schema with the
profile id enum-listed from the declared profile set.

Producer chain, all references re-read this session:
- ``Dataset/tools/batch_generate.py`` authors the source manifest
  (episode_id, scenario_id, duration_ticks, seed, n_events,
  n_event_realizations, n_entities, source_event_script_path,
  source_scene_setup_path, generator, map_id, sumo_explicit_vehicle_plan
  with schema/vehicle_count/minimum_vehicle_count/seed_profile/
  traffic_profile/review).  Frozen transfer roots can additionally carry
  the root ``source_event_script_digest``.
- ``Dataset/tools/convert_to_render_ready.py`` ``convert_episode``
  deep-copies it (:5673) and ``update``s the render keys (:5674-5726:
  capture_boundary_id, uav_crosses_boundary, inspect_observes_boundary,
  pad_boundary_policy from the source contract :5508, runtime_spatial_crop,
  source_vehicle_authority :5463-5478 (its ``explicit_vehicle_plan`` copy
  drops ``review``/``minimum_vehicle_count`` and adds episode_id/
  vehicle_ids/selection_audit), sumo_traffic from ``compiled_summary``
  :5540-5568, uav_global_flow :5574-5582, generation :5691, record_counts
  :5633-5643, artifacts/canonical_artifacts :5644-5717, node_counts :5697,
  time_range :5718, validation_summary :5724), writes the manifest
  (:5743), then attaches the scene-occupancy audit via
  ``scene_occupancy.update_episode_manifests_with_scene_occupancy``
  (:635-663), which adds the root ``scene_occupancy`` key and the
  ``scene_occupancy_manifest`` artifact slot.
- ``Dataset/tools/filter_render_ready_truth_for_capture.py``
  ``update_manifest`` (:576-623) refreshes record_counts and
  canonical_record_counts (:589-594), recomputes node_counts (:595-600),
  extends ``generation`` with the truth-filter keys (:601-607), writes
  the deepcopy-identical ``capture_visible_truth_filter`` and
  ``capture_truth_sync`` payloads (:608-621) and rewrites both artifact
  maps to the capture-filtered root (:504-518); ``filter_episode``
  re-attaches the scene-occupancy audit (:847).  This is the final
  writer of every formal byte.  ``update_render_host_config`` writes a
  sibling ``render_host_config.json`` and does not touch the manifest.

L6 profile evidence: ``scenario_id`` is one of the ten declared profile
ids above (inventory profile set for producer L6; frozen records re-read this
session cover L6-1_v1__seed00, L6-3_v2__seed01, L6-4_v2__seed02,
L6-5_v2__seed00 heads plus the X2 cross-check).  ``capture_boundary_id``
is producer-derived (``capture_boundary_`` + lowercased profile id,
L6-1_v1__seed00 :72), not a grammar pattern.

Type-system decisions:
- The current converter writes eight fields per canonical asset record.
  Frozen transfer roots have three additional real data fields:
  ``asset_conflict_count`` (string), ``asset_conflict_fields`` (string),
  and ``asset_conflicts[]`` (closed records with ``field``,
  ``logical_asset_id``, ``resolved_by``).  Both complete shapes are closed.
- Booleans stay boolean and never merge with integers (``seed`` is an
  integer; ``uav_crosses_boundary``/``inspect_observes_boundary``/
  ``enabled``/``ok``/``filtering_enabled``/``dynamic_pvu_filtering_enabled``/
  ``source_presence_required``/``expanded_to_nearest`` are booleans);
  per-key numeric/string leaves keep one raw type.
- Producer-written dynamic maps (``selection.entity_ids``, the per-id
  metric maps and ``sumo_traffic.canonical_asset_records``) are closed
  with a dict-typed ``additionalProperties`` member schema — the
  compiler's closed-boundary form for typed dynamic maps — plus an
  anchored ``propertyNames`` pattern; every member value is validated
  (``canonical_asset_records`` members against the full closed
  asset-record schema).
- ``oneOf`` separates the two canonical asset shapes and the three/five
  key lifecycle policy shapes from the converter's capture-required branch.
  No nullable leaf is observed.
  ``const`` appears only on producer constants (explicit-plan schema id,
  sumo segment phase name); ``scenario_id`` carries the declared
  profile enum list instead of a const.
- ``sumo_type_id`` vocabulary is declared as a closed identifier and
  ``sumo_record_type_id`` as the anchored ``type[@instance]`` pattern
  instead of the X sibling's two-value enum: the L6 corpus carries
  ``aero_delivery`` service records (L6-5_v2__seed00 :871), so a
  two-value enum would be incomplete producer evidence.
- Integrity metadata vs data: the two integrity-keyed fields of this
  record are ``source_event_script_digest`` at the root and
  ``generation.source_contract_hash``.  Frozen transfer roots provide
  their structural authority; the current writer does not add them.  Raw
  validation never excludes integrity paths (compiler comment,
  ``structure_schema.py`` :28-30), so both stay optional raw fields here;
  publication exclusion is carried by the existing
  ``_PUBLICATION_EXCLUDED_PATHS`` entries ``("source_event_script_digest",)``
  (:45) and ``("generation", "source_contract_hash")`` (:46) — no rule
  edit is required and this module adds none.
  ``validation_summary``/``scene_occupancy.validation_summary`` are
  producer result payloads, not integrity metadata, and stay ordinary
  data fields.

The only arrays with producer-typed item schemas are
``scenario_incidents[]`` (the converter's six-key projection,
:5557-5566), ``asset_conflicts[]``, and the ``closures[]``/
``off_road_blocking_assets[]`` members of the nested road-semantics
block — the L6 corpus writes empty lists for all four, so their nonempty
members are declared from the producer payload shapes but remain
without an L6 instance (尚无实例验证).

Boundary: this module exports the pure schema constructor only; it scans
no data and compiles nothing at import.  ``formal_episode_manifest`` is
not registered in ``structure_closure.py`` (it sits in
``EXTENSION_TAIL_FAMILIES`` with inventory status
``source_schema_pending``); registration/dispatch wiring belongs to G1,
and ``structure_sources.iter_source_records`` gates this family on the
inventory status flip to ``field_scan_defined`` before any record can be
validated at runtime.
"""

from __future__ import annotations

__all__ = ["formal_episode_manifest_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

_L6_PROFILES = (
    "L6-1_v1",
    "L6-1_v2",
    "L6-2_v1",
    "L6-2_v2",
    "L6-3_v1",
    "L6-3_v2",
    "L6-4_v1",
    "L6-4_v2",
    "L6-5_v1",
    "L6-5_v2",
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
    return {"enum": list(_L6_PROFILES)}


def _seed_label() -> dict:
    return {"type": "string", "pattern": "^seed[0-9]{2}$"}


def _digest_string() -> dict:
    # Frozen integrity metadata stays in raw validation; the shared
    # publication projection excludes its exact paths.
    return {"type": "string", "x-source-value-types": ["string"]}


def _artifact_map() -> dict:
    # The ten artifact slots written by the converter and the capture
    # filter (update_artifact_paths, :504-518) with repo-relative
    # targets; both artifact maps share the keyset.
    return _closed(
        dynamic_labels=_identifier(),
        episode_manifest=_identifier(),
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
    # nine generator slots (:5633-5643); the capture filter refreshes
    # truth_frames, trajectories and global_entity_roster (:589-594).
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
    # The root converter-written audit payload (:5724); the empty-string
    # members are the initial empty lists.
    return _closed(
        errors=_string_array(),
        ok={"type": "boolean"},
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
    # Written verbatim from the source contract (:5508).
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
    # L6 evidence (L6-1_v1__seed00 :25-70): schema/vehicle_count/
    # minimum_vehicle_count/seed_profile/traffic_profile/review; no
    # episode_id, vehicle_ids or selection audit on this form.
    return _closed(
        minimum_vehicle_count=_int_leaf(),
        review=_identifier(),
        schema={"const": "aero_explicit_sumo_vehicle_plan_v2"},
        seed_profile=_seed_profile(),
        traffic_profile=_traffic_profile(),
        vehicle_count=_int_leaf(),
    )


def _selection_audit() -> dict:
    # Keyset from L6-1_v1__seed00 :190-234 and L6-5_v2__seed00 :727-772;
    # identical to the producer audit payload.
    return _closed(
        extra_realized_background_vehicle_count=_int_leaf(),
        extra_realized_background_vehicle_ids=_string_array(),
        extra_visible_background_vehicle_count=_int_leaf(),
        extra_visible_background_vehicle_ids=_string_array(),
        missing_background_vehicle_count=_int_leaf(),
        missing_background_vehicle_ids=_string_array(),
        missing_visible_background_vehicle_count=_int_leaf(),
        missing_visible_background_vehicle_ids=_string_array(),
        planned_capture_required_vehicle_count=_int_leaf(),
        planned_semantic_vehicle_count=_int_leaf(),
        planned_source_presence_required_vehicle_count=_int_leaf(),
        planned_vehicle_count=_int_leaf(),
        policy=_identifier(),
        realized_vehicle_count=_int_leaf(),
        selected_runtime_visible_vehicle_count=_int_leaf(),
    )


def _asset_conflict() -> dict:
    # The current converter writes eight keys.  Frozen records add three
    # real conflict fields; both complete shapes remain closed.
    current = _closed(
        authority=_identifier(),
        explicit_type_id=_identifier(),
        logical_asset_id=_identifier(),
        source_entity_id={"type": "string"},
        source_presence_required={"type": "boolean"},
        source_roster_logical_asset_id={"type": "string"},
        sumo_record_type_id={
            "type": "string", "pattern": "^[^@\\s]+(@[^@\\s]+)?$",
        },
        sumo_type_id=_identifier(),
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
    # SumoTrafficDataset.source_summary() keyset
    # (truth_integration.py :230-653 range) — 18 scalars + the
    # coordinate_mapper block (identity 2x2 + translation rows,
    # L6-1_v1__seed00 :298-317).
    return _closed(
        capture_duration_s=_float_leaf(),
        capture_start_s=_float_leaf(),
        coordinate_mapper=_closed(
            matrix={
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
    # SumoSegment dict for the episode-local authority run; the phase
    # name is the producer constant (L6-1_v1__seed00 :325,
    # L6-5_v2__seed00 :326).
    return _closed(
        duration_s=_float_leaf(),
        phase_name={"const": "episode_local_vehicle_authority_after_warmup"},
        seed_index=_int_leaf(),
        seed_label=_seed_label(),
        segment_end_s=_float_leaf(),
        segment_start_s=_float_leaf(),
    )


def _visibility_geometry() -> dict:
    # VisibilityGeometry.as_dict() — eight rounded scalars
    # (sumo_ground_flow/truth_integration.py :293-303).
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
    # VehicleSelection.as_dict() keyset (:322-342): the per-vehicle
    # maps carry one member per selected vehicle id.
    return _closed(
        candidate_count=_int_leaf(),
        entity_ids=_id_map(_identifier()),
        expanded_to_nearest={"type": "boolean"},
        frames_seen_by_vehicle_id=_id_map(_int_leaf()),
        max_speed_mps_by_vehicle_id=_id_map(_float_leaf()),
        max_visible_vehicle_target=_int_leaf(),
        min_distance_m_by_vehicle_id=_id_map(_float_leaf()),
        min_visible_vehicle_target=_int_leaf(),
        motion_span_m_by_vehicle_id=_id_map(_float_leaf()),
        scenario_vehicle_ids=_string_array(),
        selected_count=_int_leaf(),
        vehicle_ids=_string_array(),
    )


def _required_lifecycle_policy() -> dict:
    # The converter writes the compact block for zero capture-required
    # vehicles and two per-id lifecycle maps when the set is nonempty.
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
    # UavGlobalFlowDataset.source_summary() keyset
    # (uav_global_flow/truth_integration.py :289-303).
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
    # UavSelection.as_dict() keyset (:60-309 range; record evidence
    # L6-1_v1__seed00 :1548-1715): the per-uav maps carry one member per
    # selected uav id.
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
    # The 29 scalar keys of the stats block (record evidence
    # L6-1_v1__seed00 :1889-2027, L6-5_v2__seed00 :1889-2028) plus the
    # nested ``road_semantics`` object.
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
            # SceneRoadSemantics.as_dict() shape; the L6 corpus writes
            # empty closures/off-road lists on every read record, so
            # their item schemas come from the producer payload shapes
            # without an L6 instance (尚无实例验证).
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
                "closed_edge_count": _int_leaf(),
                "closed_lane_count": _int_leaf(),
                "closed_source_road_count": _int_leaf(),
                "closed_edges": _string_array(),
                "closed_lanes": _string_array(),
                "closed_source_roads": _string_array(),
                "closures": {
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
                "episode_id": {"type": "string"},
                "off_road_blocking_assets": {
                    "type": "array",
                    "items": _closed(
                        asset_id=_identifier(),
                        classification={"const": "off_road_blocking_context"},
                        entity_id={"type": "string"},
                        nearest_lane_clearance_m=_float_leaf(),
                        position_enu_m=_number_array(),
                    ),
                },
                "policy": _identifier(),
                "road_blocking_asset_radius_m": _id_map(_float_leaf()),
                "road_blocking_assets": _string_array(),
                "road_closure_clearance_m": _float_leaf(),
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
        formal_roi_bbox_enu_m=_number_array(),
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
        source_contract_hash=_digest_string(),
        source_episode_dir=_identifier(),
        trajectory_contract=_identifier(),
        trajectory_source=_identifier(),
        truth_filter_contract=_identifier(),
        truth_filter_source_root=_identifier(),
    )
    schema["required"].remove("source_contract_hash")
    return schema


def _scene_occupancy() -> dict:
    # The five-key payload written by
    # scene_occupancy.update_episode_manifests_with_scene_occupancy
    # (:635-663); its validation_summary carries the two count slots the
    # root converter summary does not.
    return _closed(
        artifact=_identifier(),
        authority=_identifier(),
        geometry_authority=_identifier(),
        lane_geometry_source=_identifier(),
        validation_summary=_closed(
            error_count=_int_leaf(),
            errors=_string_array(),
            ok={"type": "boolean"},
            warning_count=_int_leaf(),
            warnings=_string_array(),
        ),
    )


def formal_episode_manifest_schema() -> dict:
    """Return the closed record schema for one L6 manifest root."""
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_episode_manifest_l6.record.schema.json",
        "title": "AERO_WORLD formal_episode_manifest L6 record schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "artifacts", "canonical_artifacts", "canonical_record_counts",
            "capture_boundary_id", "capture_truth_sync",
            "capture_visible_truth_filter", "duration_ticks", "episode_id",
            "generation", "inspect_observes_boundary", "map_id",
            "n_entities", "n_event_realizations", "n_events",
            "generator", "node_counts", "pad_boundary_policy",
            "record_counts",
            "runtime_spatial_crop", "scenario_id", "scene_occupancy",
            "seed",
            "source_event_script_path", "source_scene_setup_path",
            "source_vehicle_authority", "sumo_explicit_vehicle_plan",
            "sumo_traffic", "time_range", "uav_crosses_boundary",
            "uav_global_flow", "validation_summary",
            "vehicle_source_policy",
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
            "generation": _generation(),
            "generator": _identifier(),
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
            "source_event_script_digest": _digest_string(),
            "source_event_script_path": _identifier(),
            "source_scene_setup_path": _identifier(),
            "source_vehicle_authority": _closed(
                authority=_identifier(),
                enabled={"type": "boolean"},
                event_reference_policy=_identifier(),
                explicit_vehicle_plan=_closed(
                    episode_id=_identifier(),
                    schema={"const": "aero_explicit_sumo_vehicle_plan_v2"},
                    seed_profile=_seed_profile(),
                    selection_audit=_selection_audit(),
                    traffic_profile=_traffic_profile(),
                    vehicle_count=_int_leaf(),
                    vehicle_ids=_string_array(),
                ),
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
                explicit_vehicle_plan=_closed(
                    episode_id=_identifier(),
                    schema={"const": "aero_explicit_sumo_vehicle_plan_v2"},
                    seed_profile=_seed_profile(),
                    selection_audit=_selection_audit(),
                    traffic_profile=_traffic_profile(),
                    vehicle_count=_int_leaf(),
                    vehicle_ids=_string_array(),
                ),
                pedestrian_vehicle_clearance_filter=_pedestrian_vehicle_clearance_filter(),
                required_vehicle_lifecycle_policy=_required_lifecycle_policy(),
                runtime_spatial_crop=_runtime_spatial_crop(),
                scenario_incidents={
                    # The six-key projection the converter writes per
                    # incident (:5557-5566); the L6 corpus writes an
                    # empty list on every read record — a nonempty
                    # member is declared from the producer shape but is
                    # without an L6 instance (尚无实例验证).
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
