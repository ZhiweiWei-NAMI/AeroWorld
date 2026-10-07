"""Source field mappings and the generated semantic field vocabulary.

Only declared source rules carry unit, coordinate, or shared-field authority.
Other source paths remain distinct and numeric conversion stays unavailable
until a source declaration establishes the unit and its instance context.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from Dataset.semantic_truth.formal_clock import validate_formal_frame_clock
from Dataset.world_model.graph.field_paths import (
    WORLD_OBSERVATION_VALUE_PATH, catalog_source_branch,
    world_observation_numeric_contracts, world_observation_path_key,
)


REPO = Path(__file__).resolve().parents[3]
DEFAULT_DEFINITIONS = REPO / "design/belief/field_definitions.jsonl"

# These fields describe the record envelope, independent of its payload owner.
# Sharing their definition never merges records or their source-specific values.
SHARED_METADATA = {
    "schema_name": "Name of the source record schema",
    "schema_version": "Version string of the source record schema",
    "episode_id": "Source episode identifier",
    "model_id": "Identifier of the source producer model",
    "model_version": "Version string of the source producer model",
    "source_refs[]": "One source reference carried by the record",
}
RECORD_TICK_SEMANTIC_ID = "clock:record_tick"
FORMAL_COORDINATE_CONTRACT_ID = "coord.external_enu_m.v1"
FORMAL_POSE_COORDINATES = {
    ("formal_truth_frame", "entities[].truth_pose.position_enu_m[]"): ("m", "position"),
    ("formal_truth_frame", "entities[].truth_pose.velocity_enu_mps[]"): ("m/s", "velocity"),
}
FORMAL_COORDINATE_BASIS = (
    "Dataset/tools/convert_to_render_ready.py:truth_pose + "
    "Dataset/semantic_truth/l0_supplement.py:coordinate_contract_id + "
    "episode_manifest.json:source_scene_setup_path -> scene_setup.json:map_ref + "
    "Config/LowAltitude/Maps/donghu_road_topo/map_context.json"
)

EXACT_MEANINGS = {
    ("formal_truth_frame", "entities[].truth_pose.position_enu_m[]"):
        "One component of the entity's actual map-local east, north, up position at this frame tick; array position selects the axis",
    ("formal_truth_frame", "entities[].truth_pose.velocity_enu_mps[]"):
        "One component of the entity's actual map-local east, north, up velocity at this frame tick; array position selects the axis",
    ("formal_truth_frame", "entities[].truth_pose.coordinate_contract_id"):
        "Producer coordinate contract identifier carried by this entity's truth pose",
    ("formal_truth_frame", "dt_s"):
        "Duration of one source simulation tick in seconds",
    ("formal_truth_frame", "tick_hz"):
        "Number of source simulation ticks per second",
    ("formal_truth_frame", "sim_time_s"):
        "Source frame time in seconds from the episode clock origin",
    ("domain_state", "values.red_duration_ticks_by_controller.{entity}"):
        "Consecutive simulation ticks classified as an all_red_fault for this controller",
    ("domain_state", "values.queue_sampling_window.stopped_vehicle_count_by_lane.{entity}"):
        "Number of known vehicles at or below the source stop-speed threshold on this SUMO lane",
    ("domain_state", "values.position_error_m"):
        "Horizontal GNSS position error radius assigned to the subject UAV by the domain-state rule",
    ("compute_state", "deadline.queue_started_tick"):
        "First observed tick of the current continuous nonempty queue segment for this compute node",
    ("compute_state", "deadline.deadline_tick"):
        "Current queue segment's waiting deadline: its first observed tick plus the configured slack in ticks",
    ("compute_state", "deadline.slack_ticks"):
        "Configured waiting budget in simulation ticks for a continuous nonempty compute queue segment",
    ("compute_state", "deadline.status"):
        "Whether the current queue segment is inactive, within its waiting budget, overdue, or unknown",
    ("compute_state", "deadline.requirement_status"):
        "Availability or applicability of the source inputs for the queue-segment waiting deadline",
    ("communication_state", "bandwidth.requested_mbps"):
        "Requested data rate for this UAV communication flow before capacity allocation",
    ("communication_state", "channel.nominal_capacity_mbps"):
        "Profile nominal data rate cap copied to this flow; no station aggregate budget is modeled",
    ("communication_state", "channel.capacity_mbps"):
        "Effective data rate cap for this flow after its capacity factor",
    ("communication_state", "bandwidth.allocated_mbps"):
        "Data rate allocated to this flow as the lesser of requested rate and effective capacity",
    ("communication_state", "channel.allocated_bandwidth_mbps"):
        "Data rate allocated to this flow as the lesser of requested rate and effective capacity",
    ("communication_state", "bandwidth.dropped_mbps"):
        "Unmet requested data rate for this flow; not measured packet loss",
    ("communication_state", "channel.queued_bandwidth_mbps"):
        "Unmet requested data rate for this flow; not measured packet loss",
    ("communication_state", "link_quality.latency_ms"):
        "Entity-to-station link latency estimate calculated from the current quality score and channel latency profile",
    ("communication_state", "quality_thresholds.latency_ms"):
        "Latency limit selected by the UAV flow profile for the link-latency predicate",
    ("formal_weather_meta", "visibility_m"):
        "Visible range supplied by the weather state for this episode tick",
    ("formal_weather_meta", "fog_density"):
        "Fog density setting supplied by the weather state for this episode tick",
    ("formal_weather_meta", "rain"):
        "Rain intensity fraction supplied by the weather state for this episode tick",
    ("formal_weather_meta", "wind_speed"):
        "Wind speed supplied by the weather state for this episode tick",
    ("formal_weather_meta", "illumination_lux"):
        "Illuminance supplied by the weather state for this episode tick",
    ("formal_weather_meta", "temperature_c"):
        "Air temperature supplied by the weather state for this episode tick",
    ("formal_weather_meta", "hazard_radius_m"):
        "Planar radius used to locate the active hazard source around its declared center",
    ("arm_predicate_transitions", "from_tick"):
        "Earlier sampled endpoint coordinate referenced by this transition record",
    ("arm_predicate_transitions", "to_tick"):
        "Later sampled endpoint coordinate and record time of this transition",
    ("formal_event_realization", "render_truth_snapshots_by_tick.{tick}.{entity}.tick"):
        "Render snapshot time on the source episode's simulation clock",
    ("formal_event_realization", "source_truth_snapshots_by_tick.{tick}.{entity}.tick"):
        "Source snapshot time on the source episode's simulation clock",
}

EXACT_BRANCH_MEANINGS = {
    ("domain_state", "signal_queue", "values.queue_vehicle_count"):
        "Largest known stopped-vehicle count on one lane at the sample tick",
    ("domain_state", "signal_queue_lane_state", "values.queue_vehicle_count"):
        "Known stopped-vehicle count on the subject SUMO lane at the sample tick",
}

# These mappings were checked against the frame writer, branch state producers,
# and the action handlers. Planned positions and handler success are kept apart
# from observed motion and measured effects.
EXACT_MEANINGS.update({
    ("formal_truth_frame", "frame_seq"): "Zero-based serialized frame ordinal; this is a record index, not elapsed time.",
    ("formal_truth_frame", "frame_id"): "Episode-local identifier of the serialized truth frame.",
    ("formal_truth_frame", "entities[].entity_id"): "Roster entity whose actual state is reported by this frame item.",
    ("formal_truth_frame", "entities[].truth_pose.authority_mode"): "Authority mode governing the stored truth pose.",
    ("formal_truth_frame", "entities[].truth_pose.authority_owner"): "Producer subsystem owning the stored truth pose.",
    ("arm_states", "entity_id"): "Branch-sample entity whose state is reported at this row tick.",
    ("arm_trajectories", "entity_id"): "Branch-sample entity whose trajectory sample is reported at this row tick.",
    ("arm_states", "sim_time_s"): "Episode-local sample time copied from the formal capture frame, on the formal-derived row form.",
    ("arm_trajectories", "sim_time_s"): "Episode-local sample time copied from the formal capture frame, on the formal-derived row form.",
    ("arm_states", "pos_enu[]"): "Component of this state row's observed position in the producer's x/y/z order; no shared map origin is declared.",
    ("arm_states", "values.pos_enu[]"): "Component of the observed position in the L6 present-values carrier, in producer x/y/z order; no shared map origin is declared.",
    ("arm_states", "vel_mps[]"): "Component of observed translational velocity at the state-row tick, in producer x/y/z order.",
    ("arm_states", "values.vel_mps[]"): "Component of observed translational velocity in the L6 present-values carrier, in producer x/y/z order.",
    ("arm_trajectories", "pos_enu[]"): "Component of sampled trajectory position at this row tick, in producer x/y/z order; no shared map origin is declared.",
    ("arm_trajectories", "vel_mps[]"): "Component of sampled translational velocity at this row tick, in producer x/y/z order.",
    ("arm_states", "observed"): "Whether the state producer found an actual trajectory sample; false denotes an explicit missing row.",
    ("arm_states", "observation_status"): "Producer observation availability, including an inactive branch engine; this is not a business state.",
    ("arm_states", "execution_active"): "Whether the L5 branch engine has a sample for this entity at this tick.",
    ("arm_states", "present"): "Whether the L6 values carrier represents a present trajectory sample.",
    ("arm_states", "missing_reason"): "Reason for a missing state sample; null denotes a present sample.",
    ("arm_states", "source_ref"): "Reference to the trajectory row projected into this state sample.",
    ("arm_actions", "origin"): "Dispatcher source of the audit row: source_script or explicit intervention, not the world actor.",
    ("arm_actions", "sequence"): "Zero-based append ordinal in the action-audit stream, not a time coordinate.",
    ("arm_actions", "action.action_id"): "Branch-local authored command identifier carried into the dispatch audit.",
    ("arm_actions", "action.type"): "Dispatched simulator or control operation type.",
    ("arm_actions", "action.entity_id"): "Target entity of an entity-scoped command, not its actor.",
    ("arm_actions", "action.camera_id"): "Scene-local camera configuration requested for capture, not an asset identifier or actor.",
    ("arm_actions", "action.waypoints_enu_m[][]"): "Requested move waypoint component; axes are waypoint order then producer x/y/z, not observed position.",
    ("arm_actions", "action.position_enu_m[]"): "Requested spawn-position component in producer x/y/z order, not observed position.",
    ("arm_actions", "action.velocity_mps"): "Configured scalar move-schedule speed, not an observed velocity vector.",
    ("arm_actions", "action.delay_ticks"): "Configured dispatch-to-runtime-state delay in simulation ticks.",
    ("arm_actions", "result.status"): "Handler disposition: ok denotes documented immediate handler success; omitted dispositions mean no handler call.",
    ("arm_actions", "result.tick"): "Tick when the successful handler processed the command; absent for an omitted command.",
    ("arm_actions", "result.scheduled_tick"): "Producer-selected start tick of a move schedule or queued runtime-state patch.",
    ("arm_actions", "result.effective_tick"): "First tick declared effective for a runtime-state patch or entity removal.",
    ("arm_actions", "result.entity_id"): "Resolved target returned by a successful entity-scoped handler.",
    ("arm_actions", "result.path_length_m"): "Length from dispatch-time position through requested move waypoints; planned length, not measured travel.",
    ("arm_actions", "result.mode"): "Visual mode returned by a successful visual-state handler.",
    ("arm_actions", "result.activity_type"): "Normalized activity returned by a successful pedestrian-activity handler.",
    ("arm_actions", "result.profile"): "Weather preset selected by the handler, not evidence of a later measured weather change.",
    ("arm_actions", "result.state_families[]"): "One runtime-state family queued by a successful state handler.",
    ("arm_actions", "result.capture_id"): "Handler registration-token echo of action_id or camera_id, not an image asset or proof of capture.",
})

# A source row can legitimately omit a value kind in one episode. These types
# are declared by the current producer rather than inferred from one scan.
EXACT_ALLOWED_VALUE_TYPES = {
    ("compute_state", "deadline.queue_started_tick"): ["int", "null", "string"],
    ("compute_state", "deadline.deadline_tick"): ["int", "null", "string"],
    ("compute_state", "deadline.slack_ticks"): ["int", "string"],
}

# The two queue-depth carriers are one projected count. The top-level field can
# also carry the explicit unknown sentinel, while the nested numeric dictionary
# serializes its present integer count as a float.
EXACT_ALLOWED_VALUE_TYPES.update({
    ("compute_state", "queue_depth"): ["int", "float", "string"],
    ("compute_state", "resource_balance.queue_depth"): ["int", "float", "string"],
})

EXACT_MISSING_RULES = {
    ("formal_truth_frame", "entities[].truth_pose.position_enu_m[]"): {
        "absent": "source pose component is absent; no position is projected",
        "invalid": "non-finite or incomplete pose vector is rejected",
    },
    ("formal_truth_frame", "entities[].truth_pose.velocity_enu_mps[]"): {
        "absent": "source pose component is absent; no velocity is projected",
        "invalid": "non-finite or incomplete pose vector is rejected",
    },
    ("compute_state", "deadline.queue_started_tick"): {
        "null": "no active nonempty queue segment",
        "unknown": "source inputs for queue continuity are missing",
    },
    ("compute_state", "deadline.deadline_tick"): {
        "null": "no active nonempty queue segment",
        "unknown": "source inputs or configured waiting budget are missing",
    },
    ("compute_state", "deadline.slack_ticks"): {
        "unknown": "configured waiting budget is missing",
    },
    ("compute_state", "deadline.status"): {
        "inactive": "no active nonempty queue segment",
        "unknown": "source inputs or configured waiting budget are missing",
    },
    ("compute_state", "deadline.requirement_status"): {
        "not_applicable": "no active nonempty queue segment",
        "missing_source_record": "source inputs or configured waiting budget are missing",
    },
}

UNIT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "index": {"dimension": "ordinal_index", "canonical_unit": "index", "conversion": "identity"},
    "count": {"dimension": "discrete_count", "canonical_unit": "count", "conversion": "identity"},
    "deg": {"dimension": "plane_angle", "canonical_unit": "deg", "conversion": "identity"},
    "pixel": {"dimension": "image_axis_pixel_count", "canonical_unit": "pixel", "conversion": "identity"},
    "service_slot": {"dimension": "service_capacity", "canonical_unit": "service_slot", "conversion": "identity"},
    "source_frame": {"dimension": "source_frame_count", "canonical_unit": "source_frame", "conversion": "identity"},
    "simulation_tick": {"dimension": "simulation_time_step", "canonical_unit": "simulation_tick",
                        "conversion": "identity", "context": "sample_tick_clock"},
    "m": {"dimension": "length", "canonical_unit": "m", "conversion": "identity"},
    "m/s": {"dimension": "speed", "canonical_unit": "m/s", "conversion": "identity"},
    "s": {"dimension": "time", "canonical_unit": "s", "conversion": "identity"},
    "Hz": {"dimension": "frequency", "canonical_unit": "Hz", "conversion": "identity"},
    "vehicle": {"dimension": "entity_count", "canonical_unit": "vehicle", "conversion": "identity"},
    "aircraft": {"dimension": "entity_count", "canonical_unit": "aircraft", "conversion": "identity"},
    "Mbps": {"dimension": "data_rate", "canonical_unit": "Mbps", "conversion": "identity"},
    "ms": {"dimension": "time", "canonical_unit": "ms", "conversion": "identity"},
    "ratio": {"dimension": "dimensionless_ratio", "canonical_unit": "ratio", "conversion": "identity"},
    "lux": {"dimension": "illuminance", "canonical_unit": "lux", "conversion": "identity"},
    "degC": {"dimension": "temperature", "canonical_unit": "degC", "conversion": "identity"},
    "bitmask": {"dimension": "discrete_bitmask", "canonical_unit": "bitmask", "conversion": "identity"},
    "cpu_core": {"dimension": "processor_capacity", "canonical_unit": "cpu_core", "conversion": "identity"},
    "gpu_unit": {"dimension": "accelerator_capacity", "canonical_unit": "gpu_unit", "conversion": "identity"},
    "MB": {"dimension": "data_size", "canonical_unit": "MB", "conversion": "identity"},
    "percent": {"dimension": "percentage", "canonical_unit": "percent", "conversion": "identity"},
    "m/s^2": {"dimension": "acceleration", "canonical_unit": "m/s^2", "conversion": "identity"},
    "ppm": {"dimension": "parts_per_million", "canonical_unit": "ppm", "conversion": "identity"},
    "deg/simulation_tick": {"dimension": "angular_rate_per_simulation_tick", "canonical_unit": "deg/simulation_tick", "conversion": "identity"},
}

SEMANTIC_ALIASES = {
    ("compute_state", "resource_balance.queue_depth"):
        "source:compute_state:queue_depth",
    ("communication_state", "channel.allocated_bandwidth_mbps"):
        "source:communication_state:bandwidth.allocated_mbps",
    ("communication_state", "channel.queued_bandwidth_mbps"):
        "source:communication_state:bandwidth.dropped_mbps",
}

ARM_TRANSITION_TICK_BASIS = (
    "Dataset/semantic_truth/minimal_semantics.py:build_transitions + "
    "Dataset/semantic_truth/minimal_semantics_adapter.py:_adapt_transitions + "
    "Dataset/tools/l4_business_semantics.py:evaluate + "
    "Dataset/tools/x_arm_pipeline.py:transition construction"
)

# Exact source mappings. The producer path is evidence for the *specific*
# field; the field spelling alone is not a unit declaration.
UNIT_RULES: dict[tuple[str, str], tuple[str, str, str | None, str]] = {
    ("formal_truth_frame", "entities[].truth_pose.position_enu_m[]"):
        ("m", FORMAL_COORDINATE_BASIS, "position", "map_enu"),
    ("formal_truth_frame", "entities[].truth_pose.velocity_enu_mps[]"):
        ("m/s", FORMAL_COORDINATE_BASIS, "velocity", "map_enu"),
    ("formal_truth_frame", "dt_s"):
        ("s", "Dataset/tools/convert_to_render_ready.py:truth_frames.dt_s", "duration", None),
    ("formal_truth_frame", "tick_hz"):
        ("Hz", "Dataset/tools/convert_to_render_ready.py:truth_frames.tick_hz", "frequency", None),
    ("formal_truth_frame", "sim_time_s"):
        ("s", "Dataset/tools/convert_to_render_ready.py:truth_frames.sim_time_s", "time_point", None),
    ("domain_state", "values.red_duration_ticks_by_controller.{entity}"):
        ("simulation_tick", "Dataset/semantic_simulation/domain_state.py:red_duration", "duration", None),
    ("domain_state", "values.queue_sampling_window.stopped_vehicle_count_by_lane.{entity}"):
        ("vehicle", "Dataset/semantic_simulation/domain_state.py:load_episode_inputs.queue_window_by_tick", "count", None),
    ("domain_state", "values.queue_vehicle_count"):
        ("vehicle", "Dataset/semantic_simulation/domain_state.py:_traffic_rows.stopped", "count", None),
    ("domain_state", "values.position_error_m"):
        ("m", "Dataset/semantic_rules/predicates/world_truth_predicate_contracts.yaml:positioning.navigation_error_exceeds_tolerance", "length", None),
    ("compute_state", "deadline.queue_started_tick"):
        ("simulation_tick", "Dataset/semantic_simulation/compute_comm.py:_build_compute_rows.queue_started_tick_by_entity", "time_point", None),
    ("compute_state", "deadline.deadline_tick"):
        ("simulation_tick", "Dataset/semantic_simulation/compute_comm.py:_deadline", "time_point", None),
    ("compute_state", "deadline.slack_ticks"):
        ("simulation_tick", "Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json:compute.task_profiles.deadline_slack_ticks", "duration", None),
    ("communication_state", "bandwidth.requested_mbps"):
        ("Mbps", "Dataset/semantic_simulation/compute_comm.py:_build_communication_rows.bandwidth_request", "data_rate", None),
    ("communication_state", "channel.nominal_capacity_mbps"):
        ("Mbps", "Dataset/semantic_simulation/compute_comm.py:_build_communication_rows.nominal_bandwidth_capacity", "data_rate", None),
    ("communication_state", "channel.capacity_mbps"):
        ("Mbps", "Dataset/semantic_simulation/compute_comm.py:_build_communication_rows.bandwidth_capacity", "data_rate", None),
    ("communication_state", "bandwidth.allocated_mbps"):
        ("Mbps", "Dataset/semantic_simulation/compute_comm.py:_build_communication_rows.allocated_bandwidth", "data_rate", None),
    ("communication_state", "channel.allocated_bandwidth_mbps"):
        ("Mbps", "Dataset/semantic_simulation/compute_comm.py:_build_communication_rows.allocated_bandwidth", "data_rate", None),
    ("communication_state", "bandwidth.dropped_mbps"):
        ("Mbps", "Dataset/semantic_simulation/compute_comm.py:_build_communication_rows.queued_bandwidth", "data_rate", None),
    ("communication_state", "channel.queued_bandwidth_mbps"):
        ("Mbps", "Dataset/semantic_simulation/compute_comm.py:_build_communication_rows.queued_bandwidth", "data_rate", None),
    ("communication_state", "link_quality.latency_ms"):
        ("ms", "Dataset/semantic_simulation/compute_comm.py:_link_quality + Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json:communication.channel", "latency_estimate", None),
    ("communication_state", "quality_thresholds.latency_ms"):
        ("ms", "Dataset/semantic_simulation/compute_comm.py:_build_communication_rows + Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json:communication.flow_profiles.uav.latency_threshold_ms", "decision_threshold", None),
    ("formal_weather_meta", "visibility_m"):
        ("m", "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json:environment.visibility_below_threshold", "visible_range", None),
    ("formal_weather_meta", "fog_density"):
        ("ratio", "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json:environment.fog_active", "fog_density", None),
    ("formal_weather_meta", "rain"):
        ("ratio", "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json:environment.rain_active", "rain_intensity", None),
    ("formal_weather_meta", "wind_speed"):
        ("m/s", "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json:environment.wind_speed_above_threshold", "speed", None),
    ("formal_weather_meta", "illumination_lux"):
        ("lux", "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json:environment.low_illumination_active", "illuminance", None),
    ("formal_weather_meta", "temperature_c"):
        ("degC", "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json:environment.temperature_outside_operating_range", "temperature", None),
    ("formal_weather_meta", "hazard_radius_m"):
        ("m", "Dataset/scenarios/L3_dynamic_constraints/interaction/L3-3_v2/spec.py:hazard_radius_m + Dataset/semantic_simulation/observable_state_completion.py:_hazmat_response_rows", "radius", None),
    ("arm_predicate_transitions", "from_tick"):
        ("simulation_tick", ARM_TRANSITION_TICK_BASIS, "time_point", None),
    ("arm_predicate_transitions", "to_tick"):
        ("simulation_tick", ARM_TRANSITION_TICK_BASIS, "time_point", None),
    ("formal_event_realization", "render_truth_snapshots_by_tick.{tick}.{entity}.tick"):
        ("simulation_tick", "Dataset/tools/convert_to_render_ready.py:render_truth_snapshots_by_tick", "time_point", None),
    ("formal_event_realization", "source_truth_snapshots_by_tick.{tick}.{entity}.tick"):
        ("simulation_tick", "Dataset/tools/convert_to_render_ready.py:source_truth_snapshots_by_tick", "time_point", None),
}

ARM_MOTION_BASIS = (
    "Dataset/tools/filter_render_ready_truth_for_capture.py:trajectory_row_from_entity + "
    "Dataset/tools/batch_generate.py:EpisodeStateEngine._row_for_entity + "
    "Dataset/tools/l5_arm_common.py:states_from_rows + "
    "Dataset/tools/l6_v2/arm_runtime.py:states projection"
)
for _family, _path, _unit, _role in (
        ("arm_states", "pos_enu[]", "m", "position"),
        ("arm_states", "values.pos_enu[]", "m", "position"),
        ("arm_states", "vel_mps[]", "m/s", "velocity"),
        ("arm_states", "values.vel_mps[]", "m/s", "velocity"),
        ("arm_trajectories", "pos_enu[]", "m", "position"),
        ("arm_trajectories", "vel_mps[]", "m/s", "velocity"),
        ("arm_states", "sim_time_s", "s", "time_point"),
        ("arm_trajectories", "sim_time_s", "s", "time_point")):
    UNIT_RULES[(_family, _path)] = (_unit, ARM_MOTION_BASIS, _role, None)

UNIT_RULES.update({
    ("formal_truth_frame", "frame_seq"): ("index", "Dataset/tools/convert_to_render_ready.py:truth_frames.frame_seq", "record_sequence", None),
    ("arm_actions", "sequence"): ("index", "Dataset/tools/l5_arm_common.py:action_audit", "audit_sequence", None),
    ("arm_actions", "result.tick"): ("simulation_tick", "Dataset/tools/batch_generate.py:EpisodeStateEngine action handlers", "time_point", None),
    ("arm_actions", "result.scheduled_tick"): ("simulation_tick", "Dataset/tools/batch_generate.py:_handle_move_entity + _handle_set_runtime_state", "time_point", None),
    ("arm_actions", "result.effective_tick"): ("simulation_tick", "Dataset/tools/batch_generate.py:_handle_set_runtime_state + _handle_remove_entity", "time_point", None),
    ("arm_actions", "action.delay_ticks"): ("simulation_tick", "Dataset/tools/batch_generate.py:_handle_set_runtime_state", "duration", None),
    ("arm_actions", "result.path_length_m"): ("m", "Dataset/tools/batch_generate.py:_handle_move_entity:path_length_m", "planned_path_length", None),
    ("arm_actions", "action.waypoints_enu_m[][]"): ("m", "Dataset/tools/batch_generate.py:_handle_move_entity", "planned_position", None),
    ("arm_actions", "action.position_enu_m[]"): ("m", "Dataset/tools/batch_generate.py:_handle_spawn_entity", "planned_position", None),
    ("arm_actions", "action.velocity_mps"): ("m/s", "Dataset/tools/batch_generate.py:_handle_move_entity", "configured_speed", None),
})

def _declare_quantity(family: str, path: str, unit: str, basis: str,
                      role: str, meaning: str) -> None:
    """Compose only the exact producer rules enumerated below."""
    key = family, path
    if key in UNIT_RULES or key in EXACT_MEANINGS:
        raise ValueError(f"duplicate source quantity declaration: {key}")
    UNIT_RULES[key] = (unit, basis, role, None)
    EXACT_MEANINGS[key] = meaning


_VISIBILITY_BASIS = "Dataset/tools/inspect_observation_contract.py:VisibilityGeometry + Dataset/tools/convert_to_render_ready.py:uav_camera_capture_visibility_payload"
for _path, _meaning in (
    ("effective_tick", "Scheduled effective time beyond the observed window; the action has not executed in this dataset window."),
    ("window_end_tick", "Last physical tick available to the runtime-state materializer."),
):
    _declare_quantity("formal_objective_manifest", "runtime_state_materialization.pending_outside_window[]." + _path,
        "simulation_tick", "Dataset/semantic_simulation/predicate_state_computers.py:PlanWindowComputer.materialize_runtime_state_truth",
        "time_point", _meaning)
for _path, _meaning in {
    "offstage_source.path": "Actual optional original physical episode source path; null when that source is missing.",
    "offstage_source.role": "Role of the optional original physical source in extending the capture-filtered scope.",
    "offstage_source.status": "Presence of the optional original physical source, independent of event nonoccurrence.",
    "offstage_source.missing_policy": "Explicit treatment of missing offstage physical observations.",
    "runtime_state_materialization.pending_outside_window": "Empty scheduled-action list permitted by the runtime summary grammar.",
    "runtime_state_materialization.pending_outside_window[].action_id": "Declared action identifier for an action effective only after the observation ends.",
    "runtime_state_materialization.pending_outside_window[].entity_id": "Actual target entity of the deferred scheduled action.",
    "runtime_state_materialization.pending_outside_window[].event_id": "Actual fired script event that scheduled the deferred action.",
    "runtime_state_materialization.pending_outside_window[].source_script": "Current executable script source of the deferred action.",
    "runtime_state_materialization.pending_outside_window[].status": "Scheduled beyond the measured window; does not assert execution or an outcome.",
}.items():
    EXACT_MEANINGS["formal_objective_manifest", _path] = _meaning
for _name in ("episode_manifest.json", "global_entity_roster.json", "trajectories.jsonl", "truth_frames.jsonl", "weather_meta.jsonl"):
    EXACT_MEANINGS["formal_objective_manifest", "input_files." + _name + ".path"] = "Physical input path actually read by the objective producer: " + _name
for _name in ("compute_predicate_truth", "compute_predicate_matrix", "compute_events"):
    _declare_quantity("formal_objective_manifest", "business_event_truth.record_counts." + _name,
        "count", "Dataset/semantic_truth/compute_event_truth.py:manifest_entry", "generated_record_count",
        "Number of real records recomputed from the saved execution states in " + _name + ".jsonl.")
for _path, _meaning in {
    "producer": "Executable function that recomputes the current business predicates and events.",
    "state_sources[]": "Saved same-episode execution-state files actually consumed by the business semantic producer.",
    "source_path_base": "Directory against which the relative execution-state paths are resolved.",
    "profile_source": "Current executable business predicate and rising-event rule profile.",
    "basis": "Actual execution-state authority used by the business truth recomputation.",
    "matrix_population": "Exact population of same-tick scopes represented in the predicate matrix; absence is not out_of_scope.",
    "recovery_scope": "Recovery definition availability; rising-onset rules alone do not define recovery.",
}.items():
    EXACT_MEANINGS["formal_objective_manifest", "business_event_truth." + _path] = _meaning
for _family, _prefix in (("arm_states", ""), ("arm_trajectories", ""),
                          ("formal_truth_frame", "entities[].")):
    _declare_quantity(_family, _prefix + "uav_visibility.inspect_observation_distance_m",
        "m", _VISIBILITY_BASIS, "observability_proxy_distance",
        "Planar distance to the capture polygon or inspect camera footprint; zero inside either region.")
    _declare_quantity(_family, _prefix + "uav_visibility.roi_capture_distance_m",
        "m", _VISIBILITY_BASIS, "planar_roi_distance",
        "Planar distance to the capture polygon; zero inside its boundary.")
for _family, _path in (("arm_states", "yaw_deg"), ("arm_states", "values.yaw_deg"),
                        ("arm_trajectories", "yaw_deg")):
    _declare_quantity(_family, _path, "deg", ARM_MOTION_BASIS, "heading_angle",
        "Entity horizontal heading at this row's tick; the state engine updates it from XY velocity while moving.")
for _axis in ("pitch", "roll", "yaw"):
    _declare_quantity("formal_truth_frame", "entities[].truth_pose.rotation_deg." + _axis + "_deg",
        "deg", "Dataset/tools/convert_to_render_ready.py:truth_pose", "pose_rotation_angle",
        "Producer truth-pose " + _axis + " angle; pitch and roll are explicitly constructed as zero, while yaw carries pose heading.")

_ENTITY_CARRIERS = (("arm_branch_roster", ""), ("arm_scene_setup", ""),
    ("arm_states", ""), ("arm_trajectories", ""), ("formal_roster", ""),
    ("formal_source_scene_setup", "entities[]."), ("formal_truth_frame", "entities[]."))
_ROUTE_BASIS = "Dataset/tools/batch_generate.py:build_entities + Dataset/tools/regenerate_boundary_scenarios.py:apply_uav_corridor_contract"
_INSPECT_BASIS = "Dataset/tools/regenerate_boundary_scenarios.py:_find_inspect_route/_add_contract_inspect_uav + Dataset/tools/inspect_observation_contract.py + Config/LowAltitude/uav_sensor_profile.json"
_INSPECT_QUANTITIES = (
    ("planned_route_enu_m[][]", "m", "inspect_planned_waypoint", "One component of the generated inspect-route plan."),
    ("repaired_route_enu_m[][]", "m", "inspect_repaired_waypoint", "One component of the inspect route after building-clearance repair."),
    ("loop_route_enu_m[][]", "m", "inspect_loop_waypoint", "One component of the closed or repeated fixed-altitude route used by inspect visibility geometry."),
    ("inspect_altitude_m", "m", "planned_route_altitude", "Fixed z altitude selected for the inspect-route contract."),
    ("min_path_length_m", "m", "required_path_length", "Contract lower bound on inspect-route length."),
    ("repaired_path_length_m", "m", "repaired_path_length", "Generator-computed length of the repaired inspect route."),
    ("sensor_fov_deg", "deg", "horizontal_field_of_view", "Sensor horizontal field of view copied from the UAV sensor profile."),
    ("sensor_profile.hfov_deg", "deg", "horizontal_field_of_view", "Sensor horizontal field of view carried by the copied UAV sensor profile."),
    ("sensor_profile.width", "pixel", "image_axis_pixel_count", "Configured sensor image width in pixels."),
    ("sensor_profile.height", "pixel", "image_axis_pixel_count", "Configured sensor image height in pixels."),
    ("sensor_profile.fixed_rotation_offset_deg.pitch_deg", "deg", "fixed_camera_rotation_offset", "Sensor fixed pitch offset used to select the downward-camera footprint rule."),
    ("sensor_profile.fixed_rotation_offset_deg.yaw_deg", "deg", "fixed_camera_rotation_offset", "Configured sensor fixed yaw offset."),
    ("sensor_profile.fixed_rotation_offset_deg.roll_deg", "deg", "fixed_camera_rotation_offset", "Configured sensor fixed roll offset."),
)
for _family, _prefix in _ENTITY_CARRIERS:
    for _path, _role, _meaning in (
        ("route_waypoints_enu_m[][]", "authored_route_waypoint", "One component of the authored route used to initialize an eligible entity's movement schedule."),
        ("planned_route_waypoints_enu_m[][]", "planned_route_waypoint", "One component of the entity's carried route plan."),
    ):
        _declare_quantity(_family, _prefix + _path, "m", _ROUTE_BASIS, _role, _meaning)
    for _path, _unit, _role, _meaning in _INSPECT_QUANTITIES:
        _declare_quantity(_family, _prefix + "contract_inspect_uav." + _path,
                          _unit, _INSPECT_BASIS, _role, _meaning)

_INITIAL_BASIS = "Dataset/tools/convert_to_render_ready.py:global_entity_roster + Dataset/tools/filter_render_ready_truth_for_capture.py + ARM branch roster copying"
for _family in ("formal_roster", "arm_branch_roster"):
    _declare_quantity(_family, "initial_position_enu_m[]", "m", _INITIAL_BASIS,
        "source_dependent_initial_position", "One component of the source-selected initial or reference position; its reference time depends on the roster producer.")
    _declare_quantity(_family, "initial_yaw_deg", "deg", _INITIAL_BASIS,
        "source_dependent_initial_heading", "Source-selected initial or reference heading; its reference time depends on the roster producer.")

_GROUND_BASIS = "Dataset/tools/regenerate_boundary_scenarios.py:_ground_flow_contract/_annotate_physical_vehicle_lane + Dataset/tools/convert_to_render_ready.py:ground flow visibility"
_LIFECYCLE_BASIS = "Dataset/tools/batch_generate.py:build_entities + Dataset/tools/convert_to_render_ready.py:entity activation filtering + Dataset/tools/regenerate_boundary_scenarios.py:UAV lifecycle"
_CARRIED_QUANTITIES = (
    ("activation_tick", "simulation_tick", "time_point", "Scenario tick at which the entity becomes eligible for materialization.", _LIFECYCLE_BASIS),
    ("deactivation_tick", "simulation_tick", "time_point", "Scenario tick at which the entity stops being eligible for materialization; null leaves the stop undeclared.", _LIFECYCLE_BASIS),
    ("ground_flow_contract.speed_mps", "m/s", "configured_speed", "Configured speed of the continuous ground-flow route.", _GROUND_BASIS),
    ("ground_flow_contract.route_duration_ticks", "simulation_tick", "duration", "Planned route duration used to bound required ground-flow visibility.", _GROUND_BASIS),
    ("ground_flow_contract.planned_path_length_m", "m", "planned_path_length", "Length computed over the contract's deduplicated route points.", _GROUND_BASIS),
    ("ground_flow_contract.min_xy_span_m", "m", "required_planar_motion_span", "Contract lower bound on planar route motion span.", _GROUND_BASIS),
    ("ground_flow_contract.min_visible_motion_ratio", "ratio", "required_visible_motion_fraction", "Contract lower bound on the fraction of visible motion.", _GROUND_BASIS),
    ("ground_flow_contract.physical_lane_lateral_m", "m", "physical_lane_lateral_offset", "Selected route offset from the physical road-lane reference.", _GROUND_BASIS),
    ("background_vehicle.physical_lane_lateral_m", "m", "physical_lane_lateral_offset", "Selected physical lane offset copied into background-vehicle metadata.", _GROUND_BASIS),
    ("lifecycle.home_hover_enu_m[]", "m", "planned_home_hover_position", "One component of the planned home-pad hover position.", _LIFECYCLE_BASIS),
    ("lifecycle.mission_start_enu_m[]", "m", "planned_mission_start_position", "One component of the mission start position before lifecycle insertion.", _LIFECYCLE_BASIS),
    ("ground_reference_z_m", "m", "lifecycle_z_reference", "Source-selected z reference used to classify takeoff and landing.", _LIFECYCLE_BASIS),
    ("lifecycle.assigned_altitude_m", "m", "planned_route_altitude", "Selected z layer carried by the UAV lifecycle plan.", _LIFECYCLE_BASIS),
    ("uav_corridor.assigned_altitude_m", "m", "planned_route_altitude", "Selected z layer carried by the UAV corridor plan.", _ROUTE_BASIS),
    ("uav_corridor.altitude_layers_m[]", "m", "allowed_route_altitude", "One allowed z layer in the UAV corridor plan.", _ROUTE_BASIS),
    ("observer_lifecycle.maximum_terminal_boundary_distance_m", "m", "maximum_planar_terminal_boundary_distance", "Maximum allowed planar distance from the terminal waypoint to the capture polygon boundary.", _LIFECYCLE_BASIS),
    ("semantic_scope.service_capacity", "service_slot", "declared_service_capacity", "Concurrent charging or landing service slots declared by the facility scope contract.", "Dataset/semantic_truth/facility_scope.py + Dataset/semantic_rules/profiles/facility_scope_contract.json"),
)
for _family, _prefix in _ENTITY_CARRIERS:
    for _path, _unit, _role, _meaning, _basis in _CARRIED_QUANTITIES:
        _declare_quantity(_family, _prefix + _path, _unit, _basis, _role, _meaning)
for _family, _prefix in (("arm_branch_roster", ""), ("arm_states", ""),
                          ("arm_trajectories", ""), ("formal_roster", ""),
                          ("formal_truth_frame", "entities[].")):
    _declare_quantity(_family, _prefix + "assigned_altitude_m", "m", _ROUTE_BASIS,
        "planned_route_altitude", "Selected z altitude layer carried by the entity route plan.")

_SUMO_BASIS = "Dataset/tools/sumo_ground_flow/run_traffic.py:_vehicle_body_center_pose + Dataset/tools/convert_to_render_ready.py:SUMO roster segment + Dataset/tools/sumo_ground_flow/explicit_vehicle_plan.py"
for _family, _prefix in (("formal_roster", ""), ("arm_branch_roster", ""),
                          ("formal_truth_frame", "entities[].")):
    for _carrier in ("sumo_segment.", "ground_flow_contract.segment."):
        for _path, _unit, _role, _meaning in (
            ("segment_start_s", "s", "time_point", "Absolute source time at the beginning of the selected SUMO segment."),
            ("segment_end_s", "s", "time_point", "Absolute source time at the end of the selected SUMO segment."),
            ("duration_s", "s", "duration", "Duration of the selected SUMO segment."),
            ("seed_index", "index", "source_seed_index", "Ordinal seed index of the selected SUMO segment."),
        ):
            _declare_quantity(_family, _prefix + _carrier + _path, _unit, _SUMO_BASIS, _role, _meaning)
    for _path, _unit, _role, _meaning in (
        ("ground_flow_contract.sample_period_s", "s", "sample_interval", "Sampling interval of the saved ground-flow segment."),
        ("sumo_vehicle.lane_position_m", "m", "lane_front_position", "Longitudinal position of the vehicle front bumper measured from the SUMO lane start."),
        ("sumo_vehicle.center_lane_position_m", "m", "lane_center_position", "Longitudinal position of the vehicle body center measured from the SUMO lane start."),
        ("sumo_vehicle.allowed_speed_mps", "m/s", "allowed_source_speed", "Allowed lane or vehicle speed carried by this SUMO source row."),
    ):
        _declare_quantity(_family, _prefix + _path, _unit, _SUMO_BASIS, _role, _meaning)
for _family in ("formal_roster", "arm_branch_roster"):
    _declare_quantity(_family, "sumo_visibility.frames_seen_in_segment", "source_frame", _SUMO_BASIS,
        "segment_visibility_frame_count", "Number of source frames with a valid truth position in the selected SUMO segment.")
    _declare_quantity(_family, "sumo_visibility.min_observation_distance_m", "m", _SUMO_BASIS,
        "segment_observability_proxy_distance", "Minimum observability proxy distance carried by the segment selector.")

_BOUNDS_BASIS = "Dataset/tools/regenerate_boundary_scenarios.py:local_bounds_for_bundle"
_BOX_BASIS = "Dataset/tools/regenerate_boundary_scenarios.py:_corridor_entity_from_segment + Dataset/semantic_simulation/predicate_state_computers.py:_airspace_corridors"
for _family, _prefix in (("arm_scene_setup", ""), ("formal_source_scene_setup", "entities[].")):
    _declare_quantity(_family, "local_bounds.center_enu_m[]", "m", _BOUNDS_BASIS,
        "export_envelope_center", "One component of the export-envelope center formed from authored bounds; z is constructed as zero.")
    _declare_quantity(_family, "local_bounds.radius_m", "m", _BOUNDS_BASIS,
        "export_envelope_radius", "Export-envelope radius formed from authored planar bounds and the generator margin.")
    _declare_quantity(_family, _prefix + "placement.extent_m[]", "m", _BOX_BASIS,
        "local_box_half_extent", "One local x/y/z half dimension of the authored box.")
    for _axis in ("pitch", "yaw", "roll"):
        _declare_quantity(_family, _prefix + "placement.rotation_deg." + _axis + "_deg",
            "deg", _BOX_BASIS, "authored_placement_rotation", "Authored entity-placement " + _axis + " angle.")
    for _axis in ("pitch", "yaw"):
        _declare_quantity(_family, "cameras[].placement.rotation_deg." + _axis + "_deg",
            "deg", "Dataset/tools/regenerate_boundary_scenarios.py:scene cameras",
            "authored_static_camera_rotation", "Authored static-camera " + _axis + " orientation.")

_RESTRICTED_BASIS = "Dataset/semantic_simulation/predicate_state_computers.py:_signed_distance_to_restricted_prism"
for _family, _prefix in (("arm_predicate_truth", "evidence.observations[].value.geometry."),
    ("arm_predicate_truth", "measurements.geometry."), ("arm_predicate_truth_ticks", "operands.geometry."),
    ("arm_predicate_truth_ticks", "measurements.geometry."), ("arm_domain_state_ticks", "values.")):
    _declare_quantity(_family, _prefix + "minimum_restricted_boundary_distance_m", "m",
        _RESTRICTED_BASIS, "signed_restricted_boundary_distance",
        "Signed distance to the restricted prism boundary; interior points are negative and boundary points are zero.")
for _path, _role, _meaning, _unit in (
    ("corridor_center_enu_m[]", "authored_corridor_center", "One component of the active authored corridor-box center.", "m"),
    ("corridor_extent_m[]", "local_box_half_extent", "One local x/y/z half dimension of the active corridor box.", "m"),
    ("corridor_yaw_deg", "authored_corridor_yaw", "Yaw used to transform source ENU deltas into corridor-local axes.", "deg"),
    ("corridor_cross_section_size_m[]", "corridor_full_cross_section", "Full cross-section dimension; index zero is lateral width and index one is vertical height.", "m"),
    ("minimum_center_separation_m", "governed_center_separation", "Governed aircraft-center separation used to derive corridor capacity.", "m"),
):
    _declare_quantity("arm_domain_state_ticks", "values." + _path, _unit, _BOX_BASIS, _role, _meaning)

_SCRIPT_ROUTE_BASIS = "Dataset/tools/regenerate_boundary_scenarios.py:apply_uav_corridor_contract + Dataset/tools/uav_corridor_planner.py"
for _family in ("arm_script_plan", "formal_source_script", "script_category_source"):
    for _path, _role, _meaning in (
        ("events[].actions[].waypoints_enu_m[][]", "planned_command_waypoint", "One component of the requested command route after corridor repair."),
        ("events[].actions[].uav_corridor_validation.original_waypoints_enu_m[][]", "original_command_waypoint", "One component of the authored command route before corridor repair."),
        ("events[].actions[].uav_corridor_validation.assigned_altitude_m", "planned_route_altitude", "Selected command-route altitude carried by corridor validation."),
        ("parameters.uav_corridor_segment_details[].segment_start_enu_m[]", "planned_segment_endpoint", "One component of a consecutive repaired-route segment's start."),
        ("parameters.uav_corridor_segment_details[].segment_end_enu_m[]", "planned_segment_endpoint", "One component of a consecutive repaired-route segment's end."),
        ("parameters.uav_corridor_segment_details[].assigned_altitude_m", "planned_route_altitude", "Selected z altitude layer of the repaired-route segment."),
    ):
        _declare_quantity(_family, _path, "m", _SCRIPT_ROUTE_BASIS, _role, _meaning)
    _declare_quantity(_family, "parameters.uav_corridor_segment_details[].segment_index", "index",
        _SCRIPT_ROUTE_BASIS, "planned_segment_index", "Ordinal index of this repaired-route segment.")
for _path in ("parameters.uav_assigned_altitudes_m.{entity}",
              "parameters.fixed_uav_assigned_altitudes_m.{entity}", "parameters.uav_altitude_layers_m[]"):
    _declare_quantity("arm_script_plan", _path, "m", _SCRIPT_ROUTE_BASIS,
        "planned_route_altitude", "Selected or allowed z altitude layer in the authored UAV route plan.")
for _path, _unit, _role, _meaning in (
    ("created_schedule.start_pos_enu[]", "m", "dispatch_start_position", "One component of the engine's actual position when it created this move schedule."),
    ("created_schedule.waypoints_enu_m[][]", "m", "resolved_schedule_waypoint", "One component of the varied or resolved movement schedule."),
    ("created_schedule.velocity_mps", "m/s", "configured_speed", "Scalar speed selected for the created move schedule."),
    ("created_schedule.tick", "simulation_tick", "time_point", "Start tick selected for the created move schedule."),
    ("created_schedule.source_event_tick", "simulation_tick", "time_point", "Dispatch tick of the source event that created the move schedule."),
):
    _declare_quantity("arm_actions", _path, _unit,
        "Dataset/tools/batch_generate.py:_handle_move_entity + Dataset/tools/l1_window.py:created_schedule",
        _role, _meaning)

_UAV_FLOW_BASIS = (
    "Dataset/tools/uav_global_flow/generate_uav_flow.py:UavTask + "
    "Dataset/tools/uav_global_flow/truth_integration.py:UavSegment + "
    "Dataset/tools/convert_to_render_ready.py:build_uav_roster_entries"
)
_UAV_FLOW_QUANTITIES = (
    ("motion_contract.altitude_layer_m", "m", "planned_route_altitude",
     "Planned map-ENU z layer of the replayed global UAV task; it is not an observed or above-ground height."),
    ("motion_contract.route_length_m", "m", "planned_route_length",
     "Three-dimensional polyline length of the global UAV task's planned route."),
    ("motion_contract.segment.duration_s", "s", "duration",
     "Duration of the selected global UAV-flow source segment."),
    ("motion_contract.segment.seed_index", "index", "source_segment_index",
     "Ordinal seed segment selected from the global UAV-flow source."),
    ("motion_contract.segment.segment_end_s", "s", "source_time_point",
     "End coordinate of the selected segment on the global UAV-flow source clock."),
    ("motion_contract.segment.segment_start_s", "s", "source_time_point",
     "Start coordinate of the selected segment on the global UAV-flow source clock."),
    ("motion_contract.speed_mps", "m/s", "configured_speed",
     "Configured scalar cruise speed of the replayed global UAV task."),
    ("uav_global_flow.altitude_layer_m", "m", "planned_route_altitude",
     "Planned map-ENU z layer carried by the selected global UAV record; it is not an observed or above-ground height."),
    ("uav_global_flow.ground_reference_z_m", "m", "lifecycle_z_reference",
     "Map-ENU z reference selected from the origin pad when present, otherwise from the global UAV task plan."),
    ("uav_global_flow.sample_period_s", "s", "sample_interval",
     "Sampling interval declared by the global UAV-flow manifest."),
    ("uav_segment.duration_s", "s", "duration",
     "Duration of the selected global UAV-flow source segment."),
    ("uav_segment.seed_index", "index", "source_segment_index",
     "Ordinal seed segment selected from the global UAV-flow source."),
    ("uav_segment.segment_end_s", "s", "source_time_point",
     "End coordinate of the selected segment on the global UAV-flow source clock."),
    ("uav_segment.segment_start_s", "s", "source_time_point",
     "Start coordinate of the selected segment on the global UAV-flow source clock."),
)
for _family in ("formal_roster", "arm_branch_roster"):
    for _path, _unit, _role, _meaning in _UAV_FLOW_QUANTITIES:
        _declare_quantity(_family, _path, _unit, _UAV_FLOW_BASIS, _role, _meaning)
_declare_quantity("arm_branch_roster", "min_path_length_m", "m", _INSPECT_BASIS,
    "required_path_length", "Contract lower bound on the inspect UAV's planned route length.")

_SUMO_PLAN_BASIS = (
    "Dataset/tools/sumo_ground_flow/explicit_vehicle_plan.py:_vehicle_plan_record + "
    "Dataset/tools/sumo_ground_flow/run_traffic.py + "
    "Dataset/tools/convert_to_render_ready.py"
)
for _family in ("formal_roster", "arm_branch_roster"):
    for _path, _unit, _role, _meaning in (
        ("sumo_vehicle.semantic_metadata.expected_core_entry_tick", "simulation_tick", "time_point",
         "Planned episode-local tick at which the SUMO vehicle enters the core region."),
        ("sumo_vehicle.semantic_metadata.expected_core_exit_tick", "simulation_tick", "time_point",
         "Planned episode-local tick at which the SUMO vehicle leaves the core region."),
        ("sumo_vehicle.semantic_metadata.release_tick", "simulation_tick", "time_point",
         "Planned vehicle release coordinate on the episode clock; negative values denote release during formal warm-up before tick zero."),
        ("sumo_vehicle.semantic_metadata.seed_profile.seed_index", "index", "source_seed_profile_index",
         "Ordinal seed profile used to construct this explicit SUMO vehicle plan."),
        ("sumo_vehicle.semantic_metadata.traffic_slot_index", "index", "planned_traffic_slot_index",
         "Ordinal traffic slot assigned by the explicit SUMO vehicle plan."),
    ):
        _declare_quantity(_family, _path, _unit, _SUMO_PLAN_BASIS, _role, _meaning)

_EVENT_OCCURRENCE_BASIS = (
    "Dataset/semantic_truth/minimal_semantics.py + "
    "Dataset/semantic_truth/minimal_semantics_adapter.py:adapt_event_occurrences"
)
for _path, _unit, _role, _meaning in (
    ("trigger_tick", "simulation_tick", "time_point",
     "Formal-grid tick of the objective trigger-predicate onset."),
    ("detection_tick", "simulation_tick", "time_point",
     "Formal-grid tick at which the complete support proof confirms the event."),
    ("start_tick", "simulation_tick", "time_point",
     "Occurrence start copied exactly from trigger_tick by the producer."),
    ("end_tick", "simulation_tick", "time_point",
     "Occurrence end copied exactly from detection_tick by the producer; it is not the outcome terminal tick."),
    ("event_level", "index", "event_abstraction_level",
     "Event abstraction-level code fixed to two for an L2 event occurrence; it is not severity."),
    ("event_phases[].tick", "simulation_tick", "time_point",
     "The supporting predicate transition's to_tick for this event phase."),
):
    _declare_quantity("arm_event_occurrences", _path, _unit,
                      _EVENT_OCCURRENCE_BASIS, _role, _meaning)

_EVENT_OUTCOME_BASIS = (
    "Dataset/semantic_truth/minimal_semantics.py:build_outcomes + "
    "Dataset/semantic_truth/minimal_semantics_adapter.py:adapt_event_outcomes"
)
for _path, _meaning in (
    ("terminal_tick", "Last formal-grid tick of the exact continuous terminal-predicate hold for a successful outcome."),
    ("lifecycle_phase_evidence[].start_tick", "First formal-grid tick of the exact continuous terminal-predicate hold."),
    ("lifecycle_phase_evidence[].end_tick", "Last formal-grid tick of the exact continuous terminal-predicate hold."),
):
    _declare_quantity("arm_event_outcomes", _path, "simulation_tick",
                      _EVENT_OUTCOME_BASIS, "time_point", _meaning)

_SCRIPT_EVENT_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:event + "
    "Dataset/tools/spec_compiler.py:EventStepSpec"
)
_declare_quantity("arm_script_plan", "events[].priority", "index", _SCRIPT_EVENT_BASIS,
    "event_stage_code",
    "Authored event stage or ordering code also carried into intent_stage; local producers do not establish higher- or lower-value runtime precedence.")
_declare_quantity("arm_script_plan", "events[].max_fire_count", "count", _SCRIPT_EVENT_BASIS,
    "maximum_event_firings", "Maximum number of times the authored event may fire.")

_TERMINAL_FEASIBILITY_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:enforce_uav_terminal_feasibility + "
    "Dataset/tools/batch_generate.py:_handle_move_entity"
)
for _path, _unit, _role, _meaning in (
    ("events[].actions[].delay_ticks", "simulation_tick", "duration",
     "Configured event-dispatch delay of a queued runtime-state patch; zero is applied on the following tick by the state engine."),
    ("events[].actions[].terminal_feasibility.available_ticks", "simulation_tick", "duration",
     "Terminal-route tick budget remaining after its dispatch bound, final capture step, and touchdown dwell."),
    ("events[].actions[].terminal_feasibility.dispatch_upper_bound_tick", "simulation_tick", "time_point",
     "Deterministic latest dispatch coordinate derived from the authored event chain."),
    ("events[].actions[].terminal_feasibility.home_pose_origin_z_m", "m", "lifecycle_z_reference",
     "Map-ENU z reference copied from the terminal UAV scene entity's ground reference."),
    ("events[].actions[].terminal_feasibility.landing_reference_enu_m[]", "m", "planned_landing_position",
     "One component of the final terminal waypoint used as the explicit landing reference."),
    ("events[].actions[].terminal_feasibility.max_speed_mps", "m/s", "configured_speed_limit",
     "Configured UAV speed limit used by the terminal-feasibility proof."),
    ("events[].actions[].terminal_feasibility.required_speed_mps", "m/s", "minimum_required_constant_speed",
     "Terminal route length divided by the available terminal-route duration."),
    ("events[].actions[].terminal_feasibility.route_length_m", "m", "planned_terminal_route_length",
     "Three-dimensional polyline length of the authored terminal route."),
    ("events[].actions[].terminal_feasibility.touchdown_altitude_tolerance_m", "m", "landing_altitude_tolerance",
     "Allowed terminal z separation from the explicit landing reference."),
    ("events[].actions[].terminal_feasibility.touchdown_dwell_ticks", "simulation_tick", "duration",
     "Touchdown dwell reserved after completion of the terminal route."),
    ("events[].actions[].velocity_mps", "m/s", "commanded_speed",
     "Scalar move-command speed after terminal-feasibility adjustment; it is not an observed velocity vector."),
):
    _declare_quantity("arm_script_plan", _path, _unit,
                      _TERMINAL_FEASIBILITY_BASIS, _role, _meaning)

_ACTION_CORRIDOR_BASIS = "Dataset/tools/regenerate_boundary_scenarios.py:_replace_uav_action_waypoints"
_declare_quantity("arm_actions", "action.uav_corridor_validation.assigned_altitude_m", "m",
    _ACTION_CORRIDOR_BASIS, "planned_route_altitude",
    "Map-ENU z layer selected while repairing this UAV command route.")
_declare_quantity("arm_actions", "action.uav_corridor_validation.original_waypoints_enu_m[][]", "m",
    _ACTION_CORRIDOR_BASIS, "original_command_waypoint",
    "One component of the authored map-ENU command route saved before corridor repair; it is not an observed trajectory.")

_CAPTURE_BOUNDARY_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:_capture_boundary_from_points + "
    "Dataset/tools/roi_contract.py:source_capture_polygon"
)
for _prefix in ("parameters.capture_boundary.",
                "parameters.semantic_event_contract.capture_boundary."):
    _declare_quantity("arm_script_plan", _prefix + "center_enu_m[]", "m",
        _CAPTURE_BOUNDARY_BASIS, "capture_boundary_center",
        "One component of the final map-ENU capture-boundary center selected from the ROI contract, scene entity, or event envelope.")
    _declare_quantity("arm_script_plan", _prefix + "polygon_enu_m[][]", "m",
        _CAPTURE_BOUNDARY_BASIS, "capture_boundary_vertex",
        "One x or y component of a final map-ENU capture-polygon vertex.")

_L1_BOUNDARY_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:build_l1 + uav_home_pad_pose + "
    "_shortest_boundary_clear_path_xy"
)
for _path, _unit, _role, _meaning in (
    ("parameters.approach_tick", "simulation_tick", "time_point",
     "Authored dispatch tick passed to the approach_boundary tick trigger."),
    ("parameters.boundary_response_distance_m", "m", "trigger_radius",
     "L1-1 entity-proximity response radius for boundary_conflict; it is deliberately distinct from conflict_distance_m."),
    ("parameters.conflict_distance_m", "m", "boundary_clearance_threshold",
     "Boundary threshold used to derive L1-1 takeoff and landing clearances; in L1-1 it is not the response trigger radius."),
    ("parameters.resolution_tick", "simulation_tick", "time_point",
     "Authored nominal resolution coordinate stored by the L1 producer; current L1 production and execution code does not consume it, so it is not an observed outcome time."),
):
    _declare_quantity("arm_script_plan", _path, _unit,
                      _L1_BOUNDARY_BASIS, _role, _meaning)
for _prefix in ("parameters.", "parameters.capture_boundary.",
                "parameters.semantic_event_contract.capture_boundary."):
    for _suffix, _unit, _role, _meaning in (
        ("pre_mission_takeoff_boundary_clearance_m", "m", "required_planar_boundary_clearance",
         "Minimum planar clearance required between the restricted polygon and the home pad or repaired takeoff route."),
        ("post_conflict_landing_boundary_clearance_m", "m", "required_planar_boundary_clearance",
         "Minimum planar clearance required between the restricted polygon and the terminal landing route."),
        ("pre_mission_takeoff_max_route_length_m", "m", "maximum_planned_route_length",
         "Maximum takeoff-route length allowed before the first mission event."),
        ("pre_mission_takeoff_speed_mps", "m/s", "configured_takeoff_speed",
         "Configured takeoff speed used to derive the pre-mission route-length limit."),
    ):
        _declare_quantity("arm_script_plan", _prefix + _suffix, _unit,
                          _L1_BOUNDARY_BASIS, _role, _meaning)
for _path, _unit, _role, _meaning in (
    ("parameters.pre_mission_takeoff_boundary_clearance_model.boundary_conflict_threshold_m", "m", "boundary_clearance_threshold",
     "Boundary threshold used by the pre-mission clearance derivation."),
    ("parameters.pre_mission_takeoff_boundary_clearance_model.first_mission_event_tick", "simulation_tick", "time_point",
     "Dispatch tick of the first authored mission event in the pre-mission budget."),
    ("parameters.pre_mission_takeoff_boundary_clearance_model.max_displacement_per_tick_m", "m", "maximum_one_tick_displacement",
     "Maximum distance the configured UAV can travel during one simulation tick."),
    ("parameters.pre_mission_takeoff_boundary_clearance_model.maximum_takeoff_route_length_m", "m", "maximum_planned_route_length",
     "Maximum takeoff-route length derived from speed and the pre-mission tick budget."),
    ("parameters.pre_mission_takeoff_boundary_clearance_model.pre_mission_ticks", "simulation_tick", "duration",
     "Tick budget from takeoff entry to the first authored mission event."),
    ("parameters.pre_mission_takeoff_boundary_clearance_model.required_clearance_m", "m", "required_planar_boundary_clearance",
     "Required pre-mission planar clearance: boundary threshold plus one-tick maximum displacement."),
    ("parameters.pre_mission_takeoff_boundary_clearance_model.simulation_tick_hz", "Hz", "frequency",
     "Simulation clock rate used by the pre-mission clearance derivation."),
    ("parameters.pre_mission_takeoff_boundary_clearance_model.takeoff_entry_tick", "simulation_tick", "time_point",
     "Authored takeoff-entry coordinate used as the start of the pre-mission budget."),
    ("parameters.pre_mission_takeoff_boundary_clearance_model.takeoff_speed_mps", "m/s", "configured_takeoff_speed",
     "Configured takeoff speed used by the pre-mission route-length derivation."),
    ("parameters.pre_mission_takeoff_boundary_clearance_model.uav_max_speed_mps", "m/s", "configured_speed_limit",
     "Configured UAV speed limit used by the pre-mission clearance derivation."),
    ("parameters.post_conflict_landing_boundary_clearance_model.boundary_conflict_threshold_m", "m", "boundary_clearance_threshold",
     "Boundary threshold used by the post-conflict landing-clearance derivation."),
    ("parameters.post_conflict_landing_boundary_clearance_model.max_displacement_per_tick_m", "m", "maximum_one_tick_displacement",
     "Maximum distance the configured UAV can travel during one simulation tick."),
    ("parameters.post_conflict_landing_boundary_clearance_model.required_clearance_m", "m", "required_planar_boundary_clearance",
     "Required post-conflict planar clearance: boundary threshold plus one-tick maximum displacement."),
    ("parameters.post_conflict_landing_boundary_clearance_model.routing_numerical_margin_m", "m", "routing_numerical_margin",
     "Extra polygon-buffer distance used by the boundary visibility-graph router."),
    ("parameters.post_conflict_landing_boundary_clearance_model.simulation_tick_hz", "Hz", "frequency",
     "Simulation clock rate used by the post-conflict clearance derivation."),
    ("parameters.post_conflict_landing_boundary_clearance_model.uav_max_speed_mps", "m/s", "configured_speed_limit",
     "Configured UAV speed limit used by the post-conflict clearance derivation."),
):
    _declare_quantity("arm_script_plan", _path, _unit,
                      _L1_BOUNDARY_BASIS, _role, _meaning)

_TRAFFIC_TEMPLATE_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:DeterministicTrafficTemplate + "
    "deterministic_traffic_template_payload"
)
_declare_quantity("arm_script_plan", "parameters.deterministic_sumo_traffic_template.min_vehicle_count",
    "vehicle", _TRAFFIC_TEMPLATE_BASIS, "declared_minimum_vehicle_count",
    "Minimum vehicle count stored by the authored traffic template; current explicit vehicle-plan production does not consume this field, so it is not an observed episode count.")
for _direction in ("cross", "inbound", "outbound"):
    _declare_quantity("arm_script_plan",
        "parameters.deterministic_sumo_traffic_template.seed_profiles.{seed_profile}.direction_weights." + _direction,
        "ratio", _TRAFFIC_TEMPLATE_BASIS, "directional_allocation_weight",
        "Dimensionless authored " + _direction + " allocation weight; current explicit vehicle-plan production does not consume this field, so it is not an observed traffic probability.")

_SEMANTIC_CONTRACT_BASIS = (
    "Dataset/tools/semantic_event_contract.py:EpisodeContract.counts/_default_inspect_contract + "
    "Dataset/tools/regenerate_boundary_scenarios.py:apply_semantic_event_contract/_scene_counts"
)
for _category in ("facility", "logical", "pedestrian", "uav", "vehicle"):
    _unit = "vehicle" if _category == "vehicle" else "count"
    _declare_quantity("arm_script_plan",
        "parameters.semantic_event_contract.exact_counts." + _category,
        _unit, _SEMANTIC_CONTRACT_BASIS, "required_scene_entity_count",
        "Exact required number of " + _category + " scene entities under the low-altitude event-chain contract; logical counts corridor, trigger, airspace, hazard, and crowd-anchor sidecars rather than propositions.")
    _declare_quantity("arm_script_plan", "parameters.target_" + _category + "_count",
        _unit, _SEMANTIC_CONTRACT_BASIS, "generation_target_entity_count",
        "Scene-entity target adopted by the generator for category " + _category + "; it is a generation target rather than an observed count.")
_declare_quantity("arm_script_plan", "parameters.semantic_event_contract.inspect.altitude_m", "m",
    _SEMANTIC_CONTRACT_BASIS, "planned_route_altitude",
    "Fixed map-ENU z layer required for the contract inspect UAV; it is not an above-ground height.")
_declare_quantity("arm_script_plan", "parameters.semantic_event_contract.inspect.min_path_length_m", "m",
    _SEMANTIC_CONTRACT_BASIS, "required_path_length",
    "Contract lower bound on the inspect UAV's planned closed-loop route length.")

_SPATIAL_GRID_BASIS = (
    "Dataset/tools/sumo_ground_flow/spatial_event_grid.py:SpatialGridCell/SpatialAssignment + "
    "Dataset/tools/regenerate_boundary_scenarios.py:build_spatially_assigned_bundle"
)
for _path, _role, _meaning in (
    ("parameters.spatial_grid_assignment.actual_capture_center_enu_m[]", "actual_assignment_center",
     "One component of the post-translation map-ENU mean of capture focus points, excluding ground context."),
    ("parameters.spatial_grid_assignment.applied_world_origin_enu_m[]", "applied_world_origin",
     "One component of the map-ENU translation origin supplied to authored local p(x,y,z) coordinates."),
    ("parameters.spatial_grid_assignment.expanded_capture_center_enu_m[]", "final_capture_boundary_center",
     "One component of the final map-ENU capture-boundary center, which may include the expanded event envelope."),
    ("parameters.spatial_grid_assignment.grid_cell.center_enu_m[]", "grid_cell_center",
     "One component of the map-ENU arithmetic mean of main-road shape-segment midpoints assigned to the grid cell."),
    ("parameters.spatial_grid_assignment.grid_cell.representative_enu_m[]", "grid_cell_representative",
     "One component of the map-ENU main-road segment midpoint nearest the grid-cell center."),
    ("parameters.spatial_grid_assignment.provisional_capture_center_enu_m[]", "provisional_assignment_center",
     "One component of the pre-translation map-ENU mean of capture focus points, excluding ground context."),
    ("parameters.spatial_grid_assignment.target_center_enu_m[]", "assignment_target_center",
     "One component of the planner-selected map-ENU target, which may be an incident anchor or a grid representative."),
):
    _declare_quantity("arm_script_plan", _path, "m", _SPATIAL_GRID_BASIS, _role, _meaning)
for _path, _unit, _role, _meaning in (
    ("parameters.spatial_grid_assignment.exclusion_radius_m", "m", "incident_exclusion_radius",
     "Radius used to keep nontraffic assignments away from reserved incident points."),
    ("parameters.spatial_grid_assignment.grid_cell.incident_count", "count", "stored_grid_incident_count",
     "Stored grid-cell incident count; the current builder leaves this field at its default zero and does not compute a nonzero incident total."),
    ("parameters.spatial_grid_assignment.grid_cell.main_edge_count", "count", "distinct_vehicle_edge_count",
     "Number of distinct vehicle-capable SUMO edge identifiers represented in the grid cell."),
    ("parameters.spatial_grid_assignment.grid_cell.max_speed_mps", "m/s", "maximum_edge_speed",
     "Maximum SUMO speed among the main-road edges represented in the grid cell."),
    ("parameters.spatial_grid_assignment.grid_cell.total_main_edge_length_m", "m", "allocated_main_edge_length",
     "Accumulated equal-per-shape-segment allocation of main-edge polyline length to this cell; it is not an exact geometric clip length."),
    ("parameters.spatial_grid_assignment.target_error_m", "m", "planar_assignment_error",
     "Planar distance between the post-translation assignment center and its selected target center."),
):
    _declare_quantity("arm_script_plan", _path, _unit,
                      _SPATIAL_GRID_BASIS, _role, _meaning)

_UAV_CORRIDOR_SUMMARY_BASIS = "Dataset/tools/regenerate_boundary_scenarios.py:ensure_uav_corridor_population"
_declare_quantity("arm_script_plan", "parameters.uav_corridor_segment_count", "count",
    _UAV_CORRIDOR_SUMMARY_BASIS, "nondegenerate_route_segment_count",
    "Total number of nondegenerate adjacent point pairs across the repaired UAV routes; it is not a UAV count.")
_declare_quantity("arm_script_plan", "parameters.uav_corridor_segments[].point_count", "count",
    _UAV_CORRIDOR_SUMMARY_BASIS, "planned_route_point_count",
    "Number of points in this UAV's repaired route.")
_declare_quantity("arm_script_plan", "parameters.uav_corridor_segments[].assigned_altitude_m", "m",
    _UAV_CORRIDOR_SUMMARY_BASIS, "planned_route_altitude",
    "Map-ENU z layer assigned to this UAV's repaired route.")

_UTM_PLAN_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:build_utm_service_plan + "
    "Dataset/semantic_simulation/utm_state.py:_load_utm_service_plan/_validate_complete_schedule"
)
for _path, _role, _meaning in (
    ("parameters.utm_service_plan.global_uav_flow_policy.intent_start_tick", "time_point",
     "Start of the default operational-intent interval for roster UAVs without an explicit UAV plan."),
    ("parameters.utm_service_plan.global_uav_flow_policy.intent_end_tick", "time_point",
     "End of the default operational-intent interval for roster UAVs without an explicit UAV plan."),
    ("parameters.utm_service_plan.uav_plans[].intent_start_tick", "time_point",
     "Inclusive start tick of this UAV's explicit operational-intent interval."),
    ("parameters.utm_service_plan.uav_plans[].intent_end_tick", "time_point",
     "Inclusive end tick of this UAV's explicit operational-intent interval."),
    ("parameters.utm_service_plan.uav_plans[].authorization_schedule[].start_tick", "time_point",
     "Inclusive start tick of an airspace-authorization state interval."),
    ("parameters.utm_service_plan.uav_plans[].authorization_schedule[].end_tick", "time_point",
     "Inclusive end tick of an airspace-authorization state interval."),
    ("parameters.utm_service_plan.uav_plans[].flight_plan_schedule[].start_tick", "time_point",
     "Inclusive start tick of a flight-plan approval state interval."),
    ("parameters.utm_service_plan.uav_plans[].flight_plan_schedule[].end_tick", "time_point",
     "Inclusive end tick of a flight-plan approval state interval."),
):
    _declare_quantity("arm_script_plan", _path, "simulation_tick",
                      _UTM_PLAN_BASIS, _role, _meaning)


# Formal-source numeric meanings reviewed against the current render, simulator,
# domain-state, compute, and communication producers. Nested ENU values retain
# metre units but no coordinate-frame declaration because those carriers do not
# repeat the formal truth-pose coordinate contract needed by the binder.
_FORMAL_RENDER_NUMERIC_BASIS = 'Dataset/tools/convert_to_render_ready.py:build_annotations + runtime_visibility_payload + truth entity builders'
_FORMAL_SUMO_NUMERIC_BASIS = 'Dataset/tools/sumo_ground_flow/run_traffic.py:_vehicle_records/_traffic_light_records + Dataset/tools/sumo_ground_flow/truth_integration.py:_interpolate_vehicle/_traffic_light_states + Dataset/tools/convert_to_render_ready.py:sumo_vehicle_truth_entity'
_FORMAL_SUMO_PLAN_NUMERIC_BASIS = 'Dataset/tools/sumo_ground_flow/explicit_vehicle_plan.py:_vehicle_plan_record + Dataset/tools/sumo_ground_flow/run_traffic.py:_vehicle_records + Dataset/tools/convert_to_render_ready.py:sumo_vehicle_truth_entity'
_FORMAL_UAV_NUMERIC_BASIS = 'Dataset/tools/uav_global_flow/generate_uav_flow.py + Dataset/tools/uav_global_flow/truth_integration.py:UavSegment/sample/_interpolate_uav + Dataset/tools/convert_to_render_ready.py:build_uav_roster_entries/uav_global_truth_entity'
_FORMAL_ROSTER_NUMERIC_BASIS = 'Dataset/tools/convert_to_render_ready.py:global roster visibility accumulation and inspect route contract projection'
_COMPUTE_RESOURCE_NUMERIC_BASIS = 'Dataset/semantic_simulation/compute_comm.py:_build_compute_rows/_compute_demand/_allocate_compute/_compute_offload'
_COMMUNICATION_NUMERIC_BASIS = 'Dataset/semantic_simulation/compute_comm.py:_build_communication_rows/_select_station/_apply_handover_policy/_station_operational_state/_link_quality/_heartbeat_age_ms/_retransmission_count'
_FORMAL_WEATHER_NUMERIC_BASIS = 'Dataset/tools/convert_to_render_ready.py:normalize_weather_row + source weather generator fields'
_COMPUTE_EVENT_NUMERIC_BASIS = 'Dataset/semantic_simulation/compute_comm.py:_build_event_rows'
_FORMAL_REALIZATION_NUMERIC_BASIS = 'Dataset/tools/batch_generate.py:build_event_realizations/first_motion_tick/first_capture_motion_tick/terminal_reached_tick + Dataset/tools/convert_to_render_ready.py:rewrite_event_realizations'
_DOMAIN_STATE_NUMERIC_BASIS = 'Dataset/semantic_simulation/domain_state.py:_uav_rows/_forced_landing_rows/_payload_energy_rows/_medical_response_rows/_crowd_row/_facility_rows/_traffic_rows/_road_closure_row/_av_safe_stop_rows/_ambulance_priority_row/_gnss_values'

for _path, _unit, _basis, _role, _meaning in (
    ('handover_policy.candidate_distance_m', 'm', _COMMUNICATION_NUMERIC_BASIS, 'candidate_station_distance', 'Three-dimensional distance from the UAV to the nearest currently available candidate station.'),
    ('handover_policy.handover_margin_m', 'm', _COMMUNICATION_NUMERIC_BASIS, 'handover_distance_margin', 'Configured distance advantage that a candidate station must exceed before a geometric switch is allowed.'),
    ('handover_policy.minimum_dwell_ticks', 'simulation_tick', _COMMUNICATION_NUMERIC_BASIS, 'duration', 'Configured minimum elapsed ticks since the last station switch before another geometric switch is allowed.'),
    ('heartbeat_age_ms', 'ms', _COMMUNICATION_NUMERIC_BASIS, 'heartbeat_age', 'Elapsed time since the last successful heartbeat under a known unavailable link; zero on a usable link.'),
    ('link_quality.availability', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'station_availability_factor', 'Station availability factor used multiplicatively in the link-quality score.'),
    ('link_quality.capacity_factor', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'station_capacity_factor', 'Station factor used to scale nominal channel bandwidth capacity.'),
    ('link_quality.distance_m', 'm', _COMMUNICATION_NUMERIC_BASIS, 'selected_station_distance', 'Three-dimensional distance from the UAV to the station selected by the handover policy.'),
    ('link_quality.effective_range_m', 'm', _COMMUNICATION_NUMERIC_BASIS, 'effective_link_range', 'Station nominal range multiplied by its current range factor.'),
    ('link_quality.operational_factor', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'station_operational_factor', 'Station operational factor used multiplicatively in the link-quality score.'),
    ('link_quality.packet_loss_ratio', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'modeled_packet_loss_fraction', 'Modeled packet-loss fraction derived from the link-quality score and configured maximum packet loss.'),
    ('link_quality.quality_score', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'link_quality_score', 'Clamped link-quality score derived from range, weather attenuation, station operation, and availability.'),
    ('link_quality.range_factor', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'station_range_factor', 'Station factor used to scale its nominal communication range.'),
    ('link_quality.weather_attenuation', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'weather_attenuation_factor', 'Clamped multiplicative link attenuation computed from rain, fog density, dust, and channel coefficients.'),
    ('quality_thresholds.packet_loss_ratio', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'decision_threshold', 'Configured packet-loss fraction threshold selected by the UAV flow profile.'),
    ('retransmission.count', 'count', _COMMUNICATION_NUMERIC_BASIS, 'retransmission_attempt_count', 'Deterministically modeled retransmission count for the current link state and tick.'),
    ('route.distance_m', 'm', _COMMUNICATION_NUMERIC_BASIS, 'selected_station_distance', 'Three-dimensional distance from the UAV to the station selected by the handover policy.'),
    ('station_operational_state.availability', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'station_availability_factor', 'Station availability factor derived from the governed profile and structured runtime state.'),
    ('station_operational_state.capacity_factor', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'station_capacity_factor', 'Station bandwidth-capacity factor derived from the governed profile and structured runtime state.'),
    ('station_operational_state.operational_factor', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'station_operational_factor', 'Station link-quality operational factor derived from the governed profile and structured runtime state.'),
    ('station_operational_state.range_factor', 'ratio', _COMMUNICATION_NUMERIC_BASIS, 'station_range_factor', 'Station nominal-range factor derived from the governed profile and structured runtime state.'),
    ('handover_policy.current_station_distance_m', 'm', _COMMUNICATION_NUMERIC_BASIS, 'current_station_distance', 'Three-dimensional distance from the UAV to its previously associated station.'),
    ('handover_policy.distance_advantage_m', 'm', _COMMUNICATION_NUMERIC_BASIS, 'candidate_distance_advantage', 'Current-station distance minus candidate-station distance; positive values favor the candidate.'),
    ('handover_policy.dwell_elapsed_ticks', 'simulation_tick', _COMMUNICATION_NUMERIC_BASIS, 'duration', 'Elapsed simulation ticks since the recorded last station switch.'),
    ('handover_policy.last_switch_tick', 'simulation_tick', _COMMUNICATION_NUMERIC_BASIS, 'time_point', 'Episode tick recorded as the most recent station association switch or initial association.'),
):
    _declare_quantity('communication_state', _path, _unit, _basis, _role, _meaning)

for _path, _unit, _basis, _role, _meaning in (
    ('allocation.cpu_cores', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'locally_allocated_cpu', 'CPU capacity allocated on the subject compute node as the lesser of requested and effective local capacity.'),
    ('allocation.gpu_units', 'gpu_unit', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'locally_allocated_gpu', 'GPU capacity allocated on the subject compute node as the lesser of requested and effective local capacity.'),
    ('allocation.memory_mb', 'MB', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'locally_allocated_memory', 'Memory capacity allocated on the subject compute node as the lesser of requested and effective local capacity.'),
    ('capacity.cpu_cores', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'effective_cpu_capacity', 'CPU capacity currently available on the node after the deterministic availability schedule is applied.'),
    ('capacity.gpu_units', 'gpu_unit', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'effective_gpu_capacity', 'GPU capacity currently available on the node after the deterministic availability schedule is applied.'),
    ('capacity.max_queue_depth', 'count', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'queue_depth_limit', 'Configured maximum queue depth used to classify capacity overflow.'),
    ('capacity.memory_mb', 'MB', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'effective_memory_capacity', 'Memory capacity currently available on the node after the deterministic availability schedule is applied.'),
    ('declared_capacity.cpu_cores', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'declared_cpu_capacity', 'CPU capacity declared by the compute node profile before availability is applied.'),
    ('declared_capacity.gpu_units', 'gpu_unit', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'declared_gpu_capacity', 'GPU capacity declared by the compute node profile before availability is applied.'),
    ('declared_capacity.max_queue_depth', 'count', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'queue_depth_limit', 'Maximum queue depth declared by the compute node profile.'),
    ('declared_capacity.memory_mb', 'MB', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'declared_memory_capacity', 'Memory capacity declared by the compute node profile before availability is applied.'),
    ('inbound_allocation.cpu_cores', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'accepted_inbound_cpu', 'CPU demand accepted on this node from other nodes during the current tick.'),
    ('memory_used_mb', 'MB', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'locally_allocated_memory', 'Memory allocated to the subject task on the local node; the producer stores the allocation as memory used.'),
    ('migration.transferred_cpu_cores', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'migration_cpu_transfer', 'CPU demand transferred by an active migration; zero when migration is inactive.'),
    ('offload.accepted_cpu_cores', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'accepted_offload_cpu', 'Queued CPU demand accepted by the selected offload target at this tick.'),
    ('offload.requested_cpu_cores', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'requested_offload_cpu', 'Local queued CPU demand offered for offload before target residual capacity is applied.'),
    ('queue_depth', 'count', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'remaining_queue_depth', 'Ceiling of CPU demand still queued after offload, expressed as the producer queue-depth count.'),
    ('resource_balance.queue_depth', 'count', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'remaining_queue_depth', 'Ceiling of CPU demand still queued after offload, expressed as the producer queue-depth count.'),
    ('resource_balance.cpu_allocated', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'locally_allocated_cpu', 'CPU demand allocated on the local node.'),
    ('resource_balance.cpu_offloaded', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'accepted_offload_cpu', 'CPU demand accepted by an offload target.'),
    ('resource_balance.cpu_queued', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'remaining_queued_cpu', 'CPU demand left queued after local allocation and offload.'),
    ('resource_balance.cpu_requested', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'requested_cpu', 'CPU demand produced from the task profile, speed, load window, and deterministic variation.'),
    ('resource_balance.gpu_allocated', 'gpu_unit', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'locally_allocated_gpu', 'GPU demand allocated on the local node.'),
    ('resource_balance.gpu_queued', 'gpu_unit', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'queued_gpu', 'GPU demand exceeding effective local GPU capacity.'),
    ('resource_balance.gpu_requested', 'gpu_unit', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'requested_gpu', 'GPU demand produced from the task profile, load window, and deterministic variation.'),
    ('resource_balance.memory_allocated', 'MB', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'locally_allocated_memory', 'Memory demand allocated on the local node.'),
    ('resource_balance.memory_queued', 'MB', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'queued_memory', 'Memory demand exceeding effective local memory capacity.'),
    ('resource_balance.memory_requested', 'MB', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'requested_memory', 'Memory demand produced from the task profile, load window, and deterministic variation.'),
    ('resource_request.cpu_cores', 'cpu_core', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'requested_cpu', 'CPU demand produced from the task profile, speed, load window, and deterministic variation.'),
    ('resource_request.gpu_units', 'gpu_unit', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'requested_gpu', 'GPU demand produced from the task profile, load window, and deterministic variation.'),
    ('resource_request.memory_mb', 'MB', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'requested_memory', 'Memory demand produced from the task profile, load window, and deterministic variation.'),
    ('cpu_utilization_pct', 'percent', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'cpu_utilization', 'Local CPU allocation divided by effective CPU capacity and multiplied by 100.'),
    ('gpu_utilization_pct', 'percent', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'gpu_utilization', 'Local GPU allocation divided by effective GPU capacity and multiplied by 100.'),
    ('resource_balance.cpu_utilization_pct', 'percent', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'cpu_utilization', 'Local CPU allocation divided by effective CPU capacity and multiplied by 100.'),
    ('resource_balance.gpu_utilization_pct', 'percent', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'gpu_utilization', 'Local GPU allocation divided by effective GPU capacity and multiplied by 100.'),
    ('resource_balance.memory_utilization_pct', 'percent', _COMPUTE_RESOURCE_NUMERIC_BASIS, 'memory_utilization', 'Local memory allocation divided by effective memory capacity and multiplied by 100.'),
):
    _declare_quantity('compute_state', _path, _unit, _basis, _role, _meaning)

for _path, _unit, _basis, _role, _meaning in (
    ('values.active_closure_count', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'active_road_closure_count', 'Number of current SUMO incidents classified as road-closing incidents.'),
    ('values.affected_vehicle_count', 'vehicle', _DOMAIN_STATE_NUMERIC_BASIS, 'affected_vehicle_count', 'Number of vehicles identified as affected by the current domain condition.'),
    ('values.altitude_delta_to_landing_zone_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'signed_vertical_separation', 'Subject UAV map-local z coordinate minus the emergency landing-zone endpoint z coordinate.'),
    ('values.altitude_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'map_local_z_coordinate', 'Subject UAV map-local z coordinate copied from its truth position; it is not height above ground.'),
    ('values.capacity', 'service_slot', _DOMAIN_STATE_NUMERIC_BASIS, 'facility_service_capacity', 'Service capacity declared by the facility semantic-scope contract.'),
    ('values.civilian_mean_speed_mps', 'm/s', _DOMAIN_STATE_NUMERIC_BASIS, 'mean_vehicle_speed', 'Mean scalar speed of civilian vehicles in the ambulance-priority observation.'),
    ('values.cohort_member_count', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'declared_crowd_member_count', 'Number of entity IDs in the frozen crowd cohort definition.'),
    ('values.continuous_movement_count', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'continuous_mover_count', 'Number of observed cohort members meeting both the per-step displacement and evacuation-speed conditions.'),
    ('values.controller_count', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'traffic_controller_count', 'Number of current scene and SUMO traffic-signal controller states.'),
    ('values.detection_dwell_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Continuous ticks for which the medical subject satisfies the UAV detection condition.'),
    ('values.detour_candidate_vehicle_count', 'vehicle', _DOMAIN_STATE_NUMERIC_BASIS, 'detour_candidate_count', 'Number of closure-affected vehicles treated as detour candidates.'),
    ('values.energy_charged_ratio', 'ratio', _DOMAIN_STATE_NUMERIC_BASIS, 'state_of_charge_increment', 'Battery state-of-charge fraction added during charging ticks in the current update interval.'),
    ('values.energy_consumed_ratio', 'ratio', _DOMAIN_STATE_NUMERIC_BASIS, 'state_of_charge_decrement', 'Battery state-of-charge fraction consumed during the current update interval.'),
    ('values.evacuating_state_count', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'structured_evacuation_count', 'Number of observed cohort members whose structured pedestrian state declares evacuation active.'),
    ('values.fall_persistence_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Continuous ticks for which the medical subject remains in a corroborated fallen state.'),
    ('values.handoff_dwell_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Continuous ticks for which the responder arrival or handoff condition is satisfied.'),
    ('values.heartbeat_age_ms', 'ms', _DOMAIN_STATE_NUMERIC_BASIS, 'heartbeat_age', 'Communication heartbeat age copied into the forced-landing observation.'),
    ('values.landing_zone_approach_enu_m[]', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'planned_landing_approach_position', 'One component of the penultimate map-local ENU route point used to validate the emergency landing approach.'),
    ('values.landing_zone_center_enu_m[]', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'planned_landing_endpoint_position', 'One component of the final map-local ENU route point used as the emergency landing-zone center.'),
    ('values.lane_vehicle_count', 'vehicle', _DOMAIN_STATE_NUMERIC_BASIS, 'lane_vehicle_count', 'Number of currently observed vehicles on the subject SUMO lane.'),
    ('values.link_unavailable_duration_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Continuous simulation ticks for which the communication link has remained unavailable.'),
    ('values.max_red_duration_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Maximum current all-red-fault duration across known traffic-signal controllers.'),
    ('values.moving_count', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'moving_crowd_member_count', 'Number of observed cohort members whose scalar speed exceeds the crowd moving threshold.'),
    ('values.nearest_pad_altitude_delta_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'signed_vertical_separation', 'Subject UAV map-local z coordinate minus the nearest pad z coordinate.'),
    ('values.nearest_pad_delta_enu_m[]', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'uav_minus_pad_displacement', 'One component of the subject UAV position minus the nearest pad position in map-local ENU order.'),
    ('values.nearest_pad_distance_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'three_dimensional_distance', 'Three-dimensional distance between the subject UAV and its nearest pad.'),
    ('values.nearest_pad_xy_distance_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'planar_distance', 'Planar distance between the subject UAV and its nearest pad.'),
    ('values.nearest_uav_distance_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'three_dimensional_distance', 'Three-dimensional distance from the medical subject to the nearest observed UAV.'),
    ('values.network_mean_speed_mps', 'm/s', _DOMAIN_STATE_NUMERIC_BASIS, 'mean_vehicle_speed', 'Mean scalar speed of all vehicles in the current traffic observation.'),
    ('values.observed_cohort_member_count', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'observed_crowd_member_count', 'Number of frozen cohort IDs present in the current truth frame.'),
    ('values.observed_pedestrian_count_total', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'observed_pedestrian_count', 'Total number of pedestrian entities present in the current truth frame.'),
    ('values.payload_swing_angle_deg', 'deg', _DOMAIN_STATE_NUMERIC_BASIS, 'payload_swing_angle', 'Modeled payload swing angle derived from wind, UAV speed, and payload state.'),
    ('values.payload_swing_rate_deg_per_tick', 'deg/simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'payload_swing_rate', 'Change in modeled payload swing angle divided by authoritative tick step.'),
    ('values.pedestrian_count', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'declared_crowd_member_count', 'Number of entity IDs in the frozen crowd cohort definition.'),
    ('values.planned_route_distance_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'planned_route_length', 'Three-dimensional length of the subject UAV route waypoint polyline.'),
    ('values.position_enu_m[]', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'subject_truth_position', 'One component of the subject entity truth position in map-local ENU order; the domain row does not repeat the formal coordinate contract ID.'),
    ('values.power_derating_ratio', 'ratio', _DOMAIN_STATE_NUMERIC_BASIS, 'power_derating_fraction', 'Modeled power-derating fraction derived from observed temperature and the payload-energy profile.'),
    ('values.predicted_range_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'predicted_flight_range', 'Modeled remaining UAV range from nominal range, state of charge, and derating scale.'),
    ('values.queue_depth', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'facility_request_queue_depth', 'Number of charging requesters beyond the facility service capacity.'),
    ('values.queue_mean_speed_mps', 'm/s', _DOMAIN_STATE_NUMERIC_BASIS, 'mean_queued_vehicle_speed', 'Mean scalar speed of vehicles classified as stopped in the selected queue lane.'),
    ('values.queue_sampling_window.peak_source_tick', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'time_point', 'Truth-frame tick from which the stored queue-window measurement was taken.'),
    ('values.queue_sampling_window.queue_vehicle_count', 'vehicle', _DOMAIN_STATE_NUMERIC_BASIS, 'queued_vehicle_count', 'Largest stopped-vehicle count on a lane at the stored queue-window source tick.'),
    ('values.queue_sampling_window.window_end_tick', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'time_point', 'Inclusive end tick of the stored queue sampling window.'),
    ('values.queue_sampling_window.window_start_tick', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'time_point', 'Inclusive start tick of the stored queue sampling window.'),
    ('values.recovery_decay', 'ratio', _DOMAIN_STATE_NUMERIC_BASIS, 'gnss_error_recovery_decay', 'Configured multiplicative carry-over factor used when modeled GNSS error recovers toward its target.'),
    ('values.responder_arrival_dwell_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Continuous ticks for which the responder arrival condition is satisfied.'),
    ('values.response_rule_parameters.fall_confirmation_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Configured fall-persistence duration required to confirm a medical incident.'),
    ('values.response_rule_parameters.handoff_required_dwell_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Configured responder handoff dwell required by the medical-response rule.'),
    ('values.response_rule_parameters.responder_approach_min_delta_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'minimum_approach_progress', 'Configured minimum distance reduction used to recognize responder approach progress.'),
    ('values.response_rule_parameters.responder_arrival_radius_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'arrival_radius', 'Configured responder-to-subject distance threshold for arrival.'),
    ('values.response_rule_parameters.uav_detection_radius_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'detection_radius', 'Configured UAV-to-subject distance threshold for detection.'),
    ('values.response_rule_parameters.uav_detection_required_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Configured continuous detection duration required by the medical-response rule.'),
    ('values.route_deviation_vehicle_count', 'vehicle', _DOMAIN_STATE_NUMERIC_BASIS, 'route_deviation_vehicle_count', 'Number of currently observed vehicles whose structured SUMO metadata marks route deviation active.'),
    ('values.safe_zone_condition_dwell_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Continuous ticks for which the complete observed crowd cohort satisfies the physical safe-zone count condition.'),
    ('values.safe_zone_distance_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'safe_zone_radius', 'Configured minimum planar distance from the fixed hazard reference used to classify a cohort member in the safe zone.'),
    ('values.safe_zone_reached_fraction', 'ratio', _DOMAIN_STATE_NUMERIC_BASIS, 'required_safe_zone_fraction', 'Configured cohort fraction used to derive the required number of members in the safe zone.'),
    ('values.safe_zone_required_count', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'required_safe_zone_member_count', 'Minimum cohort member count derived from cohort size and the configured safe-zone fraction.'),
    ('values.safe_zone_required_dwell_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Configured continuous duration required for the physical safe-zone condition.'),
    ('values.service_contact_distance_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'facility_service_distance', 'Three-dimensional distance between a charging facility and its selected service UAV.'),
    ('values.speed_mps', 'm/s', _DOMAIN_STATE_NUMERIC_BASIS, 'subject_speed', 'Scalar magnitude of the subject entity truth velocity.'),
    ('values.state_of_charge_ratio', 'ratio', _DOMAIN_STATE_NUMERIC_BASIS, 'battery_state_of_charge', 'Modeled remaining battery state-of-charge fraction after interval consumption and charging.'),
    ('values.static_duration_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Continuous ticks for which the medical subject remains below the static-motion threshold.'),
    ('values.stop_dwell_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Continuous ticks for which the minimal-risk maneuver has at least one affected vehicle stopped.'),
    ('values.stopped_vehicle_count', 'vehicle', _DOMAIN_STATE_NUMERIC_BASIS, 'stopped_vehicle_count', 'Number of affected vehicles at or below the safe-stop speed threshold.'),
    ('values.structured_evacuation_state_count', 'count', _DOMAIN_STATE_NUMERIC_BASIS, 'structured_evacuation_count', 'Number of observed cohort members whose structured pedestrian state declares evacuation active.'),
    ('values.temperature_c', 'degC', _DOMAIN_STATE_NUMERIC_BASIS, 'observed_temperature', 'Weather temperature used by the payload-energy derating model.'),
    ('values.temperature_energy_factor', 'ratio', _DOMAIN_STATE_NUMERIC_BASIS, 'energy_consumption_multiplier', 'Dimensionless multiplier applied to energy consumption as a function of temperature-driven derating.'),
    ('values.touchdown_dwell_ticks', 'simulation_tick', _DOMAIN_STATE_NUMERIC_BASIS, 'duration', 'Continuous ticks for which the physical touchdown candidate conditions are satisfied.'),
    ('values.truth_position_enu_m[]', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'subject_truth_position', 'One component of the subject UAV truth position in map-local ENU order before modeled GNSS error is added.'),
    ('values.velocity_enu_mps[]', 'm/s', _DOMAIN_STATE_NUMERIC_BASIS, 'subject_truth_velocity', 'One component of the subject UAV truth velocity in map-local ENU order.'),
    ('values.vertical_speed_mps', 'm/s', _DOMAIN_STATE_NUMERIC_BASIS, 'vertical_velocity', 'Up-axis component of subject UAV truth velocity, inferred from successive positions when the stored component is zero.'),
    ('values.wind_speed_mps', 'm/s', _DOMAIN_STATE_NUMERIC_BASIS, 'wind_speed', 'Weather wind speed used by the payload swing and energy model.'),
    ('values.xy_distance_to_landing_zone_m', 'm', _DOMAIN_STATE_NUMERIC_BASIS, 'planar_distance', 'Planar distance between the subject UAV and the emergency landing-zone endpoint.'),
    ('values.yielding_vehicle_count', 'vehicle', _DOMAIN_STATE_NUMERIC_BASIS, 'yielding_vehicle_count', 'Number of nearby civilian vehicles classified as yielding during ambulance priority.'),
):
    _declare_quantity('domain_state', _path, _unit, _basis, _role, _meaning)

for _path, _unit, _basis, _role, _meaning in (
    ('transition.from_tick', 'simulation_tick', _COMPUTE_EVENT_NUMERIC_BASIS, 'time_point', 'Earlier predicate-sample tick in the rising transition that produced this compute or communication event.'),
    ('transition.to_tick', 'simulation_tick', _COMPUTE_EVENT_NUMERIC_BASIS, 'time_point', 'Later predicate-sample tick in the rising transition that produced this compute or communication event.'),
    ('trigger_tick', 'simulation_tick', _COMPUTE_EVENT_NUMERIC_BASIS, 'time_point', 'Event trigger tick copied from the later predicate sample of the rising transition.'),
):
    _declare_quantity('events', _path, _unit, _basis, _role, _meaning)

for _path, _unit, _basis, _role, _meaning in (
    ('source_truth_snapshots_by_tick.{tick}.{entity}.position_enu_m[]', 'm', _FORMAL_REALIZATION_NUMERIC_BASIS, 'source_truth_position', 'One component of the source trajectory position captured for an event evidence snapshot in map-local ENU order.'),
    ('source_truth_snapshots_by_tick.{tick}.{entity}.velocity_enu_mps[]', 'm/s', _FORMAL_REALIZATION_NUMERIC_BASIS, 'source_truth_velocity', 'One component of the source trajectory velocity captured for an event evidence snapshot in map-local ENU order.'),
    ('render_truth_snapshots_by_tick.{tick}.{entity}.position_enu_m[]', 'm', _FORMAL_REALIZATION_NUMERIC_BASIS, 'render_truth_position', 'One component of the render-ready truth position captured for an event evidence snapshot in map-local ENU order.'),
    ('render_truth_snapshots_by_tick.{tick}.{entity}.velocity_enu_mps[]', 'm/s', _FORMAL_REALIZATION_NUMERIC_BASIS, 'render_truth_velocity', 'One component of the render-ready truth velocity captured for an event evidence snapshot in map-local ENU order.'),
    ('action_realizations[].terminal_enu_m[]', 'm', _FORMAL_REALIZATION_NUMERIC_BASIS, 'planned_terminal_waypoint', 'One component of the final authored action waypoint used as the terminal target; it is not an observed terminal position.'),
    ('action_realizations[].dispatch_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'Tick at which this action realization was processed or dispatched.'),
    ('action_realizations[].scheduled_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'Producer-selected tick at which the dispatched action schedule begins.'),
    ('action_realizations[].result_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'Tick selected by the realization producer as the action result coordinate.'),
    ('action_realizations[].evidence_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'Capture-grid tick selected to carry evidence for this action result.'),
    ('sequence_no', 'index', _FORMAL_REALIZATION_NUMERIC_BASIS, 'event_log_sequence', 'One-based ordinal of the source event-log entry in the realization stream.'),
    ('dispatch_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'Tick at which the source event was dispatched.'),
    ('result_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'Latest result tick among this event realization actions, or the dispatch tick when no action result exists.'),
    ('evidence_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'Latest action evidence tick clamped to the episode duration, or the capture-aligned event result tick.'),
    ('before_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'Capture-grid evidence coordinate immediately before the event evidence tick, clamped to episode start.'),
    ('after_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'Capture-grid evidence coordinate immediately after the event evidence tick, clamped to episode end.'),
    ('capture_grid_step', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'duration', 'Spacing in simulation ticks between capture-grid evidence samples.'),
    ('action_realizations[].path_length_m', 'm', _FORMAL_REALIZATION_NUMERIC_BASIS, 'planned_action_path_length', 'Planned path length returned by the move handler for this realized action.'),
    ('action_realizations[].first_motion_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'First trajectory tick at or after scheduling whose position differs from the dispatch baseline.'),
    ('action_realizations[].first_capture_motion_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'First capture-grid tick at which motion from the dispatch baseline is observed.'),
    ('action_realizations[].terminal_tick', 'simulation_tick', _FORMAL_REALIZATION_NUMERIC_BASIS, 'time_point', 'First trajectory tick at which the action terminal waypoint is reached within the producer tolerance.'),
):
    _declare_quantity('formal_event_realization', _path, _unit, _basis, _role, _meaning)

for _path, _unit, _basis, _role, _meaning in (
    ('runtime_visibility.first_visible_tick', 'simulation_tick', _FORMAL_ROSTER_NUMERIC_BASIS, 'time_point', 'First formal episode tick at which this roster entity is retained by the runtime visibility policy.'),
    ('runtime_visibility.last_visible_tick', 'simulation_tick', _FORMAL_ROSTER_NUMERIC_BASIS, 'time_point', 'Last formal episode tick at which this roster entity is retained by the runtime visibility policy.'),
    ('runtime_visibility.visible_tick_count', 'count', _FORMAL_ROSTER_NUMERIC_BASIS, 'observed_tick_count', 'Number of formal episode ticks at which this roster entity is retained by the runtime visibility policy.'),
    ('runtime_visibility.expanded_boundary_padding_m', 'm', _FORMAL_ROSTER_NUMERIC_BASIS, 'runtime_crop_padding', 'Fixed planar padding added around the capture polygon for runtime truth retention.'),
    ('inspect_altitude_m', 'm', _FORMAL_ROSTER_NUMERIC_BASIS, 'planned_map_z_layer', 'Fixed map-local z coordinate selected for the inspect-UAV route contract; it is not height above ground.'),
    ('min_path_length_m', 'm', _FORMAL_ROSTER_NUMERIC_BASIS, 'required_path_length', 'Minimum route length required by the inspect-UAV contract.'),
):
    _declare_quantity('formal_roster', _path, _unit, _basis, _role, _meaning)

for _path, _unit, _basis, _role, _meaning in (
    ('entities[].annotations.speed_mps', 'm/s', _FORMAL_RENDER_NUMERIC_BASIS, 'annotation_speed', 'Magnitude of the entity truth velocity copied into its render annotation.'),
    ('entities[].annotations.state_facets.network.latency_ms', 'ms', _FORMAL_RENDER_NUMERIC_BASIS, 'nominal_render_annotation_latency', 'Nominal render annotation value fixed to zero by the converter; it is not a measured communication-state latency.'),
    ('entities[].annotations.state_facets.network.packet_loss', 'ratio', _FORMAL_RENDER_NUMERIC_BASIS, 'nominal_render_annotation_packet_loss', 'Nominal render annotation fraction fixed to zero by the converter; it is not measured packet loss.'),
    ('entities[].state_revision', 'index', _FORMAL_RENDER_NUMERIC_BASIS, 'state_payload_revision', 'One-based state payload revision constructed as frame tick plus one.'),
    ('entities[].visual_revision', 'index', _FORMAL_RENDER_NUMERIC_BASIS, 'visual_payload_revision', 'Visual payload revision fixed to one by the render-ready converter; it is not a time coordinate.'),
    ('entities[].runtime_visibility.expanded_boundary_padding_m', 'm', _FORMAL_RENDER_NUMERIC_BASIS, 'runtime_crop_padding', 'Fixed planar padding added around the capture polygon for runtime truth retention.'),
    ('entities[].runtime_visibility.distance_to_capture_boundary_m', 'm', _FORMAL_RENDER_NUMERIC_BASIS, 'planar_capture_boundary_distance', 'Planar distance from the entity position to the capture polygon, with zero for a point inside the polygon.'),
    ('entities[].sumo_vehicle.truth_front_bumper_enu_m[]', 'm', _FORMAL_SUMO_NUMERIC_BASIS, 'truth_front_bumper_position', 'One component of the SUMO vehicle front-bumper position after mapping into truth map-local ENU coordinates; the nested carrier does not repeat the coordinate contract ID.'),
    ('entities[].sumo_vehicle.sumo_xy_m[]', 'm', _FORMAL_SUMO_NUMERIC_BASIS, 'sumo_front_bumper_coordinate', 'One component of the raw SUMO-network XY position returned for the vehicle front bumper; this is not map ENU.'),
    ('entities[].sumo_vehicle.sumo_front_bumper_xy_m[]', 'm', _FORMAL_SUMO_NUMERIC_BASIS, 'sumo_front_bumper_coordinate', 'One component of the explicit raw SUMO-network front-bumper XY position; this is not map ENU.'),
    ('sumo_traffic_light_states.{entity}.phase_index', 'index', _FORMAL_SUMO_NUMERIC_BASIS, 'sumo_signal_phase_index', 'Zero-based active SUMO traffic-light program phase index.'),
    ('sumo_traffic_light_states.{entity}.next_switch_s', 's', _FORMAL_SUMO_NUMERIC_BASIS, 'source_time_point', 'Absolute SUMO simulation time at which the traffic-light program next switches phase.'),
    ('entities[].sumo_vehicle.sumo_angle_deg', 'deg', _FORMAL_SUMO_NUMERIC_BASIS, 'sumo_heading_angle', 'SUMO vehicle heading angle returned by TraCI before conversion to truth yaw.'),
    ('entities[].sumo_vehicle.speed_mps', 'm/s', _FORMAL_SUMO_NUMERIC_BASIS, 'vehicle_speed', 'SUMO vehicle scalar speed, interpolated between source traffic samples when needed.'),
    ('entities[].sumo_vehicle.accel_mps2', 'm/s^2', _FORMAL_SUMO_NUMERIC_BASIS, 'vehicle_acceleration', 'SUMO vehicle scalar longitudinal acceleration, interpolated between source traffic samples when needed.'),
    ('entities[].sumo_vehicle.signals', 'bitmask', _FORMAL_SUMO_NUMERIC_BASIS, 'sumo_vehicle_signal_mask', 'Integer bitmask returned by TraCI for the vehicle signal state.'),
    ('entities[].sumo_vehicle.dimensions_m.height', 'm', _FORMAL_SUMO_NUMERIC_BASIS, 'vehicle_body_height', 'SUMO vehicle body height reported by TraCI.'),
    ('entities[].sumo_vehicle.dimensions_m.length', 'm', _FORMAL_SUMO_NUMERIC_BASIS, 'vehicle_body_length', 'SUMO vehicle body length reported by TraCI.'),
    ('entities[].sumo_vehicle.dimensions_m.width', 'm', _FORMAL_SUMO_NUMERIC_BASIS, 'vehicle_body_width', 'SUMO vehicle body width reported by TraCI.'),
    ('entities[].sumo_vehicle.source_prev_time_s', 's', _FORMAL_SUMO_NUMERIC_BASIS, 'source_time_point', 'Earlier SUMO source-frame time bracketing this formal sample.'),
    ('entities[].sumo_vehicle.source_next_time_s', 's', _FORMAL_SUMO_NUMERIC_BASIS, 'source_time_point', 'Later SUMO source-frame time bracketing this formal sample.'),
    ('entities[].sumo_vehicle.source_alpha', 'ratio', _FORMAL_SUMO_NUMERIC_BASIS, 'interpolation_fraction', 'Clamped interpolation fraction between the bracketing SUMO source frames.'),
    ('entities[].sumo_visibility.inspect_observation_distance_m', 'm', _FORMAL_RENDER_NUMERIC_BASIS, 'observability_proxy_distance', 'Planar distance from the SUMO vehicle position to the inspect observation geometry; zero inside the observed region.'),
    ('entities[].sumo_vehicle.semantic_metadata.expected_core_entry_tick', 'simulation_tick', _FORMAL_SUMO_PLAN_NUMERIC_BASIS, 'time_point', 'Authored episode tick at which the explicit vehicle plan expects entry into its semantic core region.'),
    ('entities[].sumo_vehicle.semantic_metadata.expected_core_exit_tick', 'simulation_tick', _FORMAL_SUMO_PLAN_NUMERIC_BASIS, 'time_point', 'Authored episode tick at which the explicit vehicle plan expects exit from its semantic core region.'),
    ('entities[].sumo_vehicle.semantic_metadata.release_tick', 'simulation_tick', _FORMAL_SUMO_PLAN_NUMERIC_BASIS, 'time_point', 'Authored vehicle release coordinate on the episode clock; negative values refer to the formal warm-up interval.'),
    ('entities[].sumo_vehicle.semantic_metadata.seed_profile.seed_index', 'index', _FORMAL_SUMO_PLAN_NUMERIC_BASIS, 'seed_ordinal', 'Zero-based deterministic traffic seed-profile ordinal carried by the explicit vehicle plan.'),
    ('entities[].sumo_vehicle.semantic_metadata.traffic_slot_index', 'index', _FORMAL_SUMO_PLAN_NUMERIC_BASIS, 'traffic_schedule_slot', 'Ordinal traffic-flow slot assigned by the explicit vehicle plan; it is not a simulation tick.'),
    ('entities[].uav_global_flow.altitude_layer_m', 'm', _FORMAL_UAV_NUMERIC_BASIS, 'planned_map_z_layer', 'Fixed map-local z layer assigned by the global UAV task plan; it is not height above ground.'),
    ('entities[].uav_global_flow.ground_reference_z_m', 'm', _FORMAL_UAV_NUMERIC_BASIS, 'map_ground_reference_z', 'Map-local z coordinate selected from the origin pad or global UAV task plan as the ground reference.'),
    ('entities[].uav_global_flow.sample_period_s', 's', _FORMAL_UAV_NUMERIC_BASIS, 'duration', 'Sampling interval of the global UAV source dataset.'),
    ('entities[].uav_global_flow.source_prev_time_s', 's', _FORMAL_UAV_NUMERIC_BASIS, 'source_time_point', 'Earlier global-UAV source-frame time bracketing this formal sample.'),
    ('entities[].uav_global_flow.source_next_time_s', 's', _FORMAL_UAV_NUMERIC_BASIS, 'source_time_point', 'Later global-UAV source-frame time bracketing this formal sample.'),
    ('entities[].uav_global_flow.source_alpha', 'ratio', _FORMAL_UAV_NUMERIC_BASIS, 'interpolation_fraction', 'Clamped interpolation fraction between the bracketing global-UAV source frames.'),
    ('entities[].motion_contract.segment.seed_index', 'index', _FORMAL_UAV_NUMERIC_BASIS, 'seed_ordinal', 'Zero-based global-UAV segment ordinal selected for this episode seed.'),
    ('entities[].motion_contract.segment.segment_start_s', 's', _FORMAL_UAV_NUMERIC_BASIS, 'source_time_point', 'Start time of the selected segment on the global-UAV source timeline.'),
    ('entities[].motion_contract.segment.segment_end_s', 's', _FORMAL_UAV_NUMERIC_BASIS, 'source_time_point', 'End time of the selected segment on the global-UAV source timeline.'),
    ('entities[].motion_contract.segment.duration_s', 's', _FORMAL_UAV_NUMERIC_BASIS, 'duration', 'Duration of the selected global-UAV source segment.'),
    ('entities[].motion_contract.speed_mps', 'm/s', _FORMAL_UAV_NUMERIC_BASIS, 'configured_route_speed', 'Scalar speed configured by the global UAV task whose route is replayed.'),
    ('entities[].motion_contract.route_length_m', 'm', _FORMAL_UAV_NUMERIC_BASIS, 'planned_route_length', 'Three-dimensional length of the global UAV task route.'),
    ('entities[].motion_contract.altitude_layer_m', 'm', _FORMAL_UAV_NUMERIC_BASIS, 'planned_map_z_layer', 'Fixed map-local z layer carried by the replay motion contract; it is not height above ground.'),
    ('entities[].uav_segment.seed_index', 'index', _FORMAL_UAV_NUMERIC_BASIS, 'seed_ordinal', 'Zero-based global-UAV segment ordinal selected for this episode seed.'),
    ('entities[].uav_segment.segment_start_s', 's', _FORMAL_UAV_NUMERIC_BASIS, 'source_time_point', 'Start time of the selected segment on the global-UAV source timeline.'),
    ('entities[].uav_segment.segment_end_s', 's', _FORMAL_UAV_NUMERIC_BASIS, 'source_time_point', 'End time of the selected segment on the global-UAV source timeline.'),
    ('entities[].uav_segment.duration_s', 's', _FORMAL_UAV_NUMERIC_BASIS, 'duration', 'Duration of the selected global-UAV source segment.'),
    ('entities[].inspect_altitude_m', 'm', _FORMAL_RENDER_NUMERIC_BASIS, 'planned_map_z_layer', 'Fixed map-local z coordinate carried by the inspect-UAV route contract; it is not height above ground.'),
    ('entities[].min_path_length_m', 'm', _FORMAL_RENDER_NUMERIC_BASIS, 'required_path_length', 'Minimum inspect-route path length required by the carried route contract.'),
):
    _declare_quantity('formal_truth_frame', _path, _unit, _basis, _role, _meaning)

# Numeric carriers that are copied from authored runtime state into both the
# formal roster and formal truth entities.  The executable state contract fixes
# the ranges and consumers; these values are source state, not measurements
# made by the graph builder.
_FORMAL_RUNTIME_STATE_NUMERIC_BASIS = (
    "Dataset/tools/runtime_state_contract.py:invalid_runtime_state_value_paths + "
    "Dataset/tools/convert_to_render_ready.py:runtime_state_fields_from_sources/"
    "preserved_fields_from"
)
for _family, _prefix in (
    ("formal_roster", ""),
    ("formal_truth_frame", "entities[]."),
):
    for _path, _unit, _role, _meaning in (
        ("communication_state.availability", "ratio",
         "authored_communication_availability",
         "Authored runtime communication-availability scalar, constrained to the interval from zero to one and preserved as predicate input; it is not measured link quality."),
        ("communication_state.link_availability", "ratio",
         "authored_link_availability",
         "Authored runtime link-availability scalar, constrained to the interval from zero to one and preserved as predicate input; it is not a measured delivery rate."),
        ("facility_state.capacity", "service_slot",
         "authored_facility_service_capacity",
         "Nonnegative concurrent service capacity carried by the governed runtime facility state."),
        ("facility_state.request_count", "count",
         "authored_facility_request_count",
         "Nonnegative number of service requests carried by the governed runtime facility state."),
        ("incident_state.hazard_concentration_ppm", "ppm",
         "authored_hazard_concentration",
         "Nonnegative hazard concentration carried by the governed runtime incident state."),
        ("incident_state.hazard_radius_m", "m",
         "authored_hazard_radius",
         "Nonnegative planar hazard radius carried by the governed runtime incident state."),
        ("path_deviation_contract.minimum_route_distance_m", "m",
         "route_deviation_decision_threshold",
         "Governed minimum planar distance from the planned route required by positioning.aircraft_deviates_from_planned_route."),
    ):
        _declare_quantity(
            _family, _prefix + _path, _unit,
            _FORMAL_RUNTIME_STATE_NUMERIC_BASIS, _role, _meaning,
        )

# Event actors are authored offstage and enter on a quantized physical route.
# The route builder computes every quantity below before the converter copies
# the contract into the roster and each visible truth entity.
_EVENT_ACTOR_MOTION_NUMERIC_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:"
    "event_actor_entry/relocate_event_actor_entries_outside_capture_boundary/"
    "quantized_path_duration_ticks/path_length_m"
)
for _family, _prefix in (
    ("formal_roster", ""),
    ("formal_truth_frame", "entities[]."),
):
    for _path, _unit, _role, _meaning in (
        ("event_actor_motion_contract.entry_tick", "simulation_tick",
         "time_point",
         "Episode tick at which the quantized offstage-to-event entry route begins."),
        ("event_actor_motion_contract.arrival_tick", "simulation_tick",
         "time_point",
         "Episode tick at which the event actor is planned to reach the semantic arrival point."),
        ("event_actor_motion_contract.entry_capture_boundary_clearance_m", "m",
         "planar_capture_boundary_clearance",
         "Planar polygon distance from the relocated entry-route start to the capture boundary."),
        ("event_actor_motion_contract.entry_speed_mps", "m/s",
         "configured_entry_speed",
         "Scalar route speed used to quantize the event actor's entry motion."),
        ("event_actor_motion_contract.minimum_visible_motion_ratio_before_event", "ratio",
         "required_pre_event_visible_motion_ratio",
         "Required fraction of the entry motion visible before the semantic event; the current producer fixes it to one."),
        ("event_actor_motion_contract.route_length_m", "m",
         "planned_entry_route_length",
         "Three-dimensional polyline length of the event actor's quantized entry route."),
    ):
        _declare_quantity(
            _family, _prefix + _path, _unit,
            _EVENT_ACTOR_MOTION_NUMERIC_BASIS, _role, _meaning,
        )

# ARM scene records may carry one entity at the root, so these placement
# fields use root paths.  The lane writers take longitudinal coordinates
# from LaneSample.s_m; the authored hint remains a separate quantity.
# Metre authority here does not bind these carriers to the formal pose frame.
_ARM_SCENE_PLACEMENT_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:lane_entity/"
    "sidewalk_entity/_physical_vehicle_lane_audit/align_vehicle_lane_motion + "
    "Dataset/tools/map_spatial_index.py:LaneSample/plan_sidewalk_anchor"
)
for _path, _unit, _role, _meaning in (
    ("placement.lane_half_width_m", "m", "authored_lane_half_width",
     "Lane half-width used by the scene author to place lane and shoulder entities."),
    ("placement.lane_index", "index", "anchor_lane_index",
     "Lane index of the selected traffic-bundle anchor sample on its edge."),
    ("placement.lateral_offset_m", "m", "authored_lateral_offset",
     "Authored lateral offset: vehicles store their selected physical-lane offset, while other lane entities retain the requested offset before shoulder adjustment."),
    ("placement.longitudinal_s", "m", "resolved_lane_longitudinal_coordinate",
     "Longitudinal coordinate in metres of the resolved anchor sample on its traffic-bundle edge."),
    ("placement.physical_lane_center_offsets_m[]", "m", "physical_lane_center_offset",
     "One signed physical vehicle-lane center offset computed from the selected road's lane count and width."),
    ("placement.resolved_lateral_from_center_m", "m", "resolved_lane_lateral_offset",
     "Signed lateral offset actually applied to the selected lane sample to resolve the authored entity position."),
    ("placement.road_lane_count", "count", "placement_road_lane_count",
     "Road lane count used by the physical-lane placement calculation; the producer uses one when road metadata is unavailable."),
    ("placement.road_width_m", "m", "placement_road_width",
     "Road width used by the physical-lane placement calculation; when road metadata is unavailable the producer uses twice its authored lane half-width."),
    ("placement.selected_physical_lane_offset_m", "m", "selected_physical_lane_offset",
     "Signed physical-lane center offset selected for this vehicle's placement."),
    ("placement.source_longitudinal_s_hint", "m", "authored_lane_longitudinal_hint",
     "Requested longitudinal coordinate in metres passed to the lane-placement resolver; it is distinct from the resolved anchor coordinate."),
):
    _declare_quantity("arm_scene_setup", _path, _unit,
                      _ARM_SCENE_PLACEMENT_BASIS, _role, _meaning)
for _path, _role, _meaning, _writer in (
    ("placement.position_enu_m[]", "authored_entity_position_component",
     "One component of the authored world-pose entity position; array position selects the producer's x, y, or z component.",
     "world_entity"),
    ("placement.resolved_position_enu_m[]", "resolved_entity_position_component",
     "One component of the entity position resolved by the scene placement writer; array position selects the producer's x, y, or z component.",
     "world_entity/lane_entity/sidewalk_entity/box_entity"),
    ("placement.center_enu_m[]", "authored_box_center_component",
     "One component of the authored box-volume center in the producer's x, y, or z order.",
     "box_entity"),
):
    _declare_quantity("arm_scene_setup", _path, "m",
        "Dataset/tools/regenerate_boundary_scenarios.py:" + _writer +
        " + Dataset/tools/x_arm_pipeline.py:prepare + Dataset/scenarios/X_cross_layer",
        _role, _meaning)

# Service-facility repair uses the traffic bundle's lane samples.  Position
# components therefore have metre authority, while these nested carriers do
# not repeat the formal pose coordinate contract and remain unbound to a
# coordinate frame by coordinate_policy.
_SCENE_OCCUPANCY_REPAIR_NUMERIC_BASIS = (
    "Dataset/tools/map_spatial_index.py:LaneSample/plan_sidewalk_anchor/"
    "nearest_lane_clearance + Dataset/tools/convert_to_render_ready.py:"
    "_plan_service_facility_repair/repair_service_facility_placements"
)
_SCENE_OCCUPANCY_REPAIR_QUANTITIES = (
    ("anchor_lane_s_m", "m", "anchor_lane_longitudinal_coordinate",
     "Longitudinal coordinate of the selected anchor sample on its traffic-bundle lane edge."),
    ("from_position_enu_m[]", "m", "pre_repair_position_component",
     "One x, y, or z component of the service facility position before render-ready reanchoring."),
    ("offset_from_curb_m", "m", "repair_search_offset",
     "Repair search offset: the normal sidewalk branch stores distance beyond the fixed lane half-width, while the open-space fallback stores its radial search distance."),
    ("required_clearance_m", "m", "required_lane_center_clearance",
     "Asset-specific minimum planar center-to-lane clearance required before accepting the repaired position."),
    ("resolved_lateral_from_center_m", "m", "signed_lane_center_lateral_offset",
     "Signed lateral displacement from the selected lane-center sample to the repaired position; the open-space fallback stores the unsigned clearance."),
    ("road_clearance_m", "m", "nearest_lane_center_clearance",
     "Planar distance from the repaired position to the nearest sampled lane center."),
    ("to_position_enu_m[]", "m", "post_repair_position_component",
     "One x, y, or z component of the accepted service facility position after render-ready reanchoring."),
)
for _family, _prefix in (
    ("formal_roster", "scene_occupancy_repair."),
    ("formal_truth_frame", "entities[].scene_occupancy_repair."),
):
    for _path, _unit, _role, _meaning in _SCENE_OCCUPANCY_REPAIR_QUANTITIES:
        _declare_quantity(
            _family, _prefix + _path, _unit,
            _SCENE_OCCUPANCY_REPAIR_NUMERIC_BASIS, _role, _meaning,
        )
for _path, _unit, _role, _meaning in (
    ("placement.anchor_lane_s_m", "m", "anchor_lane_longitudinal_coordinate",
     "Longitudinal coordinate of the accepted repair anchor on its traffic-bundle lane edge."),
    ("placement.offset_from_curb_m", "m", "repair_search_offset",
     "Repair search offset copied into placement: normally distance beyond the fixed lane half-width, or the radial search distance for the open-space fallback."),
    ("placement.resolved_lateral_from_center_m", "m",
     "signed_lane_center_lateral_offset",
     "Signed lateral displacement from the selected lane-center sample copied from the accepted repair; the open-space fallback stores the unsigned clearance."),
    ("placement.road_clearance_m", "m", "nearest_lane_center_clearance",
     "Planar distance from the accepted placement to the nearest sampled lane center."),
):
    _declare_quantity(
        "formal_roster", _path, _unit,
        _SCENE_OCCUPANCY_REPAIR_NUMERIC_BASIS, _role, _meaning,
    )

# Pad phase values are positive, dimensionless ranking weights.  They are not
# probabilities: the current Gaussian-plus-offset producer can emit values
# above one and multiplies origin and destination weights when ranking pairs.
_UAV_GLOBAL_PAD_NUMERIC_BASIS = (
    "Dataset/tools/uav_global_flow/generate_uav_flow.py:"
    "_phase_centers/_phase_weight/_candidate_pad_for_cell/ranked_pad_pairs"
)
for _family, _prefix in (
    ("formal_roster", ""),
    ("formal_truth_frame", "entities[]."),
):
    for _path, _unit, _role, _meaning in (
        ("uav_global_pad.lane_s_m", "m", "pad_anchor_lane_longitudinal_coordinate",
         "Longitudinal coordinate of the pad's sidewalk anchor on its traffic-bundle lane edge."),
        ("uav_global_pad.phase_origin_weights[]", "ratio",
         "delivery_phase_origin_ranking_weight",
         "One unnormalized Gaussian-plus-offset spatial weight used to rank this pad as an origin; array position selects one of three delivery phases."),
        ("uav_global_pad.phase_destination_weights[]", "ratio",
         "delivery_phase_destination_ranking_weight",
         "One unnormalized Gaussian-plus-offset spatial weight used to rank this pad as a destination; array position selects one of three reversed delivery phases."),
    ):
        _declare_quantity(
            _family, _prefix + _path, _unit,
            _UAV_GLOBAL_PAD_NUMERIC_BASIS, _role, _meaning,
        )

# The branch roster is the nearest identity authority used by ARM graph
# materialization.  Background identities are copied intact from the formal
# roster.  Engine identities are rebuilt from raw/ue_roster.json and may carry
# only the producer's explicit preserve list.  Reuse the already adjudicated
# formal quantity semantics while recording which of those two copy paths can
# put each field into ue/global_entity_roster.json.
_ARM_BRANCH_FORMAL_ROSTER_COPY_BASIS = (
    "Dataset/tools/arm_ue_export.py:export_arm canonical_roster/"
    "output_roster + Dataset/world_model/graph/arm.py:materialize_arm "
    "roster_paths/branch_roster"
)
_ARM_BRANCH_ENGINE_PRESERVE_BASIS = (
    "Dataset/tools/batch_generate.py:PRESERVED_ENTITY_FIELDS/"
    "preserved_fields_from/write_engine_inputs + Dataset/tools/"
    "arm_ue_export.py:export_arm raw/ue_roster/output_roster + "
    "Dataset/world_model/graph/arm.py:materialize_arm roster_paths/"
    "branch_roster"
)

# These fields are absent from the engine-row allowlist (runtime-state fields
# are explicitly removed), so an ARM branch roster can receive them only from
# an untouched formal-roster identity.
for _path in (
    "communication_state.availability",
    "communication_state.link_availability",
    "facility_state.capacity",
    "facility_state.request_count",
    "incident_state.hazard_concentration_ppm",
    "incident_state.hazard_radius_m",
    "placement.anchor_lane_s_m",
    "placement.offset_from_curb_m",
    "placement.resolved_lateral_from_center_m",
    "placement.road_clearance_m",
    "runtime_visibility.expanded_boundary_padding_m",
    "runtime_visibility.first_visible_tick",
    "runtime_visibility.last_visible_tick",
    "runtime_visibility.visible_tick_count",
    "scene_occupancy_repair.anchor_lane_s_m",
    "scene_occupancy_repair.from_position_enu_m[]",
    "scene_occupancy_repair.offset_from_curb_m",
    "scene_occupancy_repair.required_clearance_m",
    "scene_occupancy_repair.resolved_lateral_from_center_m",
    "scene_occupancy_repair.road_clearance_m",
    "scene_occupancy_repair.to_position_enu_m[]",
    "uav_global_pad.lane_s_m",
    "uav_global_pad.phase_destination_weights[]",
    "uav_global_pad.phase_origin_weights[]",
):
    _unit, _basis, _role, _coordinate = UNIT_RULES[("formal_roster", _path)]
    if _coordinate is not None:
        raise ValueError(f"unexpected formal roster coordinate binding: {_path}")
    _declare_quantity(
        "arm_branch_roster", _path, _unit,
        _basis + " + " + _ARM_BRANCH_FORMAL_ROSTER_COPY_BASIS,
        _role, EXACT_MEANINGS[("formal_roster", _path)],
    )

# These contracts are on the engine preserve list as well as on untouched
# formal identities.  Their physical definitions remain those of the formal
# producer; the ARM tools copy them without recomputation.
for _path in (
    "event_actor_motion_contract.arrival_tick",
    "event_actor_motion_contract.entry_capture_boundary_clearance_m",
    "event_actor_motion_contract.entry_speed_mps",
    "event_actor_motion_contract.entry_tick",
    "event_actor_motion_contract.minimum_visible_motion_ratio_before_event",
    "event_actor_motion_contract.route_length_m",
    "inspect_altitude_m",
    "path_deviation_contract.minimum_route_distance_m",
):
    _unit, _basis, _role, _coordinate = UNIT_RULES[("formal_roster", _path)]
    if _coordinate is not None:
        raise ValueError(f"unexpected formal roster coordinate binding: {_path}")
    _declare_quantity(
        "arm_branch_roster", _path, _unit,
        _basis + " + " + _ARM_BRANCH_ENGINE_PRESERVE_BASIS,
        _role, EXACT_MEANINGS[("formal_roster", _path)],
    )

_declare_quantity(
    "formal_truth_frame",
    "entities[].sumo_vehicle.semantic_metadata.source_activation_tick",
    "simulation_tick",
    "Dataset/tools/sumo_ground_flow/build_semantic_vehicle_plans.py:"
    "build_vehicle_base_instructions",
    "time_point",
    "Authored source-entity activation tick retained as SUMO semantic metadata; the SUMO vehicle itself is inserted at warm-up tick zero.",
)

# The domain-state rows preserve UNKNOWN beside numeric values.  These rules
# govern only a present numeric member and retain the producer's exact role.
for _path, _unit, _basis, _role, _meaning in (
    ("values.ambulance_mean_speed_mps", "m/s",
     "Dataset/semantic_simulation/domain_state.py:_ambulance_priority_row/_speed/_mean",
     "mean_ambulance_speed",
     "Mean scalar speed of all entities classified by the producer as ambulances at this tick."),
    ("values.ambulance_travel_since_priority_m", "m",
     "Dataset/semantic_simulation/domain_state.py:_ambulance_priority_row",
     "planar_travel_since_priority_activation",
     "Planar displacement of the sole primary ambulance from its position saved when the current priority interval activated."),
    ("values.clearance_gap_m", "m",
     "Dataset/semantic_simulation/domain_state.py:_ambulance_priority_row",
     "nearest_civilian_planar_clearance",
     "Minimum planar distance from the sole primary ambulance to a civilian vehicle inside the configured yield-observation radius."),
    ("values.priority_activation_tick", "simulation_tick",
     "Dataset/semantic_simulation/domain_state.py:_ambulance_priority_row:priority_state",
     "time_point",
     "First episode tick of the current uninterrupted ambulance-priority interval."),
    ("values.fault_start_tick", "simulation_tick",
     "Dataset/semantic_simulation/domain_state.py:_av_safe_stop_rows:fault_start_tick",
     "time_point",
     "First episode tick of the current uninterrupted AV sensor-fault interval; the producer clears it when the fault becomes false."),
    ("values.mean_affected_accel_mps2", "m/s^2",
     "Dataset/semantic_simulation/domain_state.py:_av_safe_stop_rows/_vehicle_accel/_mean",
     "mean_affected_vehicle_longitudinal_acceleration",
     "Mean SUMO longitudinal acceleration of vehicles selected as affected by the AV safe-stop incident."),
    ("values.mean_affected_speed_mps", "m/s",
     "Dataset/semantic_simulation/domain_state.py:_av_safe_stop_rows/_speed/_mean",
     "mean_affected_vehicle_speed",
     "Mean scalar speed of vehicles selected as affected by the AV safe-stop incident."),
    ("values.crowd_centroid_enu_m[]", "m",
     "Dataset/semantic_simulation/domain_state.py:_crowd_row/_centroid",
     "current_crowd_centroid_component",
     "One x, y, or z component of the arithmetic centroid of currently observed frozen-cohort pedestrian positions."),
    ("values.hazard_reference_enu_m[]", "m",
     "Dataset/semantic_simulation/domain_state.py:_crowd_row:crowd_initial",
     "initial_crowd_hazard_reference_component",
     "One x, y, or z component of the hazard reference fixed to the frozen cohort's first authoritative-position centroid."),
    ("values.hazard_zone_count", "count",
     "Dataset/semantic_simulation/domain_state.py:_crowd_row",
     "pedestrian_count_within_hazard_radius",
     "Number of currently observed frozen-cohort pedestrians whose planar distance from the initial hazard reference is at most the configured hazard radius."),
    ("values.pedestrian_displacement_m.{entity}", "m",
     "Dataset/semantic_simulation/domain_state.py:_crowd_row:displacement_by_pedestrian",
     "pedestrian_planar_displacement_from_initial_position",
     "Planar distance from the named frozen-cohort pedestrian's current position to its first authoritative position."),
    ("values.safe_zone_count", "count",
     "Dataset/semantic_simulation/domain_state.py:_crowd_row",
     "pedestrian_count_at_safe_distance",
     "Number of currently observed frozen-cohort pedestrians whose planar distance from the initial hazard reference is at least the configured safe-zone distance."),
    ("values.forced_descent_start_tick", "simulation_tick",
     "Dataset/semantic_simulation/domain_state.py:_forced_landing_rows:forced_descent_start_tick",
     "time_point",
     "First episode tick at which the current forced descent was physically observed and latched."),
    ("values.link_unavailable_since_tick", "simulation_tick",
     "Dataset/semantic_simulation/domain_state.py:_forced_landing_rows:link_unavailable_since_tick",
     "time_point",
     "First episode tick of the current uninterrupted known-unavailable communication-link interval."),
    ("values.drift_offset_enu_m[]", "m",
     "Dataset/semantic_simulation/domain_state.py:_gnss_values",
     "modeled_gnss_error_vector_component",
     "One x, y, or z component of the modeled GNSS error vector added to the truth position."),
    ("values.reported_position_enu_m[]", "m",
     "Dataset/semantic_simulation/domain_state.py:_gnss_values",
     "modeled_gnss_reported_position_component",
     "One x, y, or z component of truth position plus the modeled GNSS error vector."),
    ("values.nearest_responder_distance_m", "m",
     "Dataset/semantic_simulation/domain_state.py:_medical_rows/_nearest_entity",
     "nearest_responder_three_dimensional_distance",
     "Three-dimensional Euclidean distance from the medical subject to the nearest responder with an observed truth position."),
    ("values.responder_eta_s", "s",
     "Dataset/semantic_simulation/domain_state.py:_medical_rows",
     "nominal_responder_eta",
     "Nearest-responder distance divided by the configured positive nominal responder speed."),
    ("values.auth_score", "ratio",
     "Dataset/semantic_simulation/domain_state.py:_security_rows",
     "authentication_score",
     "Authentication score constructed as 0.25 for explicit compromise or unauthorized command and 1.0 for explicit absence of both conditions."),
    ("values.command_integrity_score", "ratio",
     "Dataset/semantic_simulation/domain_state.py:_security_rows",
     "command_integrity_score",
     "Command-integrity score constructed as 0.1 for an explicit integrity violation and 1.0 for its explicit absence."),
    ("values.lockout_duration_ticks", "simulation_tick",
     "Dataset/semantic_simulation/domain_state.py:_security_rows:lockout_duration",
     "duration",
     "Accumulated authoritative tick duration of the current uninterrupted explicit command-lockout interval."),
    ("values.spectrum_interference_ratio", "ratio",
     "Dataset/semantic_simulation/domain_state.py:_security_rows",
     "jamming_indicator_ratio",
     "Modeled spectrum-interference indicator: one for explicit active jamming and zero for explicit inactive jamming."),
):
    _declare_quantity("domain_state", _path, _unit, _basis, _role, _meaning)

# Formal L1/L2 time carriers are direct projections of the minimal-semantics
# state machine.  Their ARM counterparts are separate source families and do
# not supply authority for these formal rows.
_FORMAL_MINIMAL_SEMANTICS_NUMERIC_BASIS = (
    "Dataset/semantic_truth/world_truth.py:evaluate_world_truth/replay_world_truth + "
    "Dataset/semantic_truth/minimal_semantics.py:build_transitions + "
    "Dataset/semantic_truth/minimal_semantics_adapter.py:"
    "_adapt_assertions/_adapt_transitions/adapt_event_occurrences"
)
for _path, _meaning in (
    ("truth_state_update_tick",
     "Episode tick at which this assertion's truth value last changed, or its initial materialization tick when it has not changed."),
    ("evidence_update_tick",
     "Episode tick at which this assertion's evidence payload last changed or was refreshed."),
    ("evidence.observations[].value.truth_state_update_tick",
     "Truth-state update tick copied into the normalized world-observation evidence carried by this assertion."),
    ("evidence.observations[].value.evidence_update_tick",
     "Evidence update tick copied into the normalized world-observation evidence carried by this assertion."),
):
    _declare_quantity(
        "formal_predicate_truth", _path, "simulation_tick",
        _FORMAL_MINIMAL_SEMANTICS_NUMERIC_BASIS, "time_point", _meaning,
    )
for _path, _meaning in (
    ("from_tick", "Earlier adjacent formal predicate-sample tick referenced by this transition."),
    ("to_tick", "Later adjacent formal predicate-sample tick at which this transition is materialized."),
):
    _declare_quantity(
        "formal_predicate_transitions", _path, "simulation_tick",
        _FORMAL_MINIMAL_SEMANTICS_NUMERIC_BASIS, "time_point", _meaning,
    )
for _path, _unit, _role, _meaning in (
    ("trigger_tick", "simulation_tick", "time_point",
     "Formal-grid tick of the objective trigger-predicate onset."),
    ("detection_tick", "simulation_tick", "time_point",
     "Formal-grid tick at which the complete support proof confirms the event."),
    ("start_tick", "simulation_tick", "time_point",
     "Occurrence start copied exactly from trigger_tick by the producer."),
    ("end_tick", "simulation_tick", "time_point",
     "Occurrence end copied exactly from detection_tick; it is not an outcome terminal tick."),
    ("event_level", "index", "event_abstraction_level",
     "Event abstraction-level code fixed to two for an L2 occurrence; it is not severity."),
    ("event_phases[].tick", "simulation_tick", "time_point",
     "The supporting predicate transition's to_tick saved by _event_phase_entry for this occurrence phase."),
):
    _declare_quantity(
        "formal_event_occurrences", _path, _unit,
        _FORMAL_MINIMAL_SEMANTICS_NUMERIC_BASIS, _role, _meaning,
    )

for _path, _meaning in (
    ("from_tick", "Earlier observed predicate endpoint of a continuity break; intervening truth is not inferred."),
    ("to_tick", "Later observed predicate endpoint of a continuity break; the gap differs from the formal sampling step."),
):
    _declare_quantity("formal_predicate_continuity_breaks", _path, "simulation_tick",
                      _FORMAL_MINIMAL_SEMANTICS_NUMERIC_BASIS, "time_point", _meaning)

for _path, _role, _meaning in (
    ("render_truth_reconciliation.changes[].old_result_tick", "time_point",
     "Action result tick saved before render-truth terminal reconciliation; null means the source action had no result tick."),
    ("render_truth_reconciliation.changes[].old_terminal_tick", "time_point",
     "Action terminal tick saved before render-truth terminal reconciliation; null means the source action had no terminal tick."),
    ("render_truth_reconciliation.changes[].render_result_tick", "time_point",
     "Action result tick replaced with the first render-truth tick that reaches the required terminal waypoint."),
    ("render_truth_reconciliation.changes[].render_terminal_tick", "time_point",
     "First render-truth tick at which the action reaches its required terminal waypoint."),
    ("render_truth_trigger_reconciliation.source_dispatch_tick", "time_point",
     "Event dispatch tick before a SUMO-replaced vehicle trigger is reconciled against render truth."),
    ("render_truth_trigger_reconciliation.render_dispatch_tick", "time_point",
     "Event dispatch tick recomputed from render truth for a SUMO-replaced vehicle trigger or its dependent event."),
):
    _declare_quantity(
        "formal_event_realization", _path, "simulation_tick",
        "Dataset/tools/convert_to_render_ready.py:"
        "render_truth_event_trigger_reconciliation/reconcile_event_realizations_with_render_truth",
        _role, _meaning,
    )

for _path, _unit, _basis, _role, _meaning in (
    ('wetness', 'ratio', _FORMAL_WEATHER_NUMERIC_BASIS, 'surface_wetness_fraction', 'Source road or surface wetness fraction constrained to the interval from zero to one.'),
    ('dust', 'ratio', _FORMAL_WEATHER_NUMERIC_BASIS, 'dust_intensity_fraction', 'Source atmospheric dust intensity fraction constrained to the interval from zero to one.'),
    ('wind_direction_deg', 'deg', _FORMAL_WEATHER_NUMERIC_BASIS, 'wind_direction_angle', 'Wind direction angle supplied by the formal weather source.'),
    ('hazard_concentration_ppm', 'ppm', _FORMAL_WEATHER_NUMERIC_BASIS, 'hazard_concentration', 'Hazard-source concentration supplied by the formal weather source in parts per million.'),
):
    _declare_quantity('formal_weather_meta', _path, _unit, _basis, _role, _meaning)

# The formal label graph now keeps the author ScriptEvent and ActionPlan rows.
# These leaves are the same authored commands consumed by EpisodeStateEngine;
# only explicitly enumerated paths reuse the reviewed ARM script declaration.
_FORMAL_AUTHOR_EVENT_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:event/enforce_uav_terminal_feasibility + "
    "Dataset/tools/spec_compiler.py:EventStepSpec + Dataset/tools/batch_generate.py:EpisodeStateEngine"
)
for _path in (
    "events[].actions[].delay_ticks",
    "events[].actions[].terminal_feasibility.available_ticks",
    "events[].actions[].terminal_feasibility.dispatch_upper_bound_tick",
    "events[].actions[].terminal_feasibility.home_pose_origin_z_m",
    "events[].actions[].terminal_feasibility.landing_reference_enu_m[]",
    "events[].actions[].terminal_feasibility.max_speed_mps",
    "events[].actions[].terminal_feasibility.required_speed_mps",
    "events[].actions[].terminal_feasibility.route_length_m",
    "events[].actions[].terminal_feasibility.touchdown_altitude_tolerance_m",
    "events[].actions[].terminal_feasibility.touchdown_dwell_ticks",
    "events[].actions[].velocity_mps",
    "events[].max_fire_count",
    "events[].priority",
):
    _source_key = "arm_script_plan", _path
    _unit, _source_basis, _role, _frame = UNIT_RULES[_source_key]
    _declare_quantity(
        "formal_source_script", _path, _unit,
        _FORMAL_AUTHOR_EVENT_BASIS + "; " + _source_basis,
        _role, EXACT_MEANINGS[_source_key],
    )

for _path, _unit, _role, _meaning, _basis in (
    ("events[].actions[].continuity_repair.original_start_enu_m[]", "m",
     "original_command_start_position",
     "One component of the authored first move waypoint saved before continuity repair replaced it with the entity's current position.",
     "Dataset/tools/regenerate_boundary_scenarios.py:enforce_move_start_continuity"),
    ("events[].actions[].continuity_repair.start_gap_m", "m",
     "command_start_continuity_gap",
     "Three-dimensional distance from the authored first move waypoint to the entity position expected at dispatch order.",
     "Dataset/tools/regenerate_boundary_scenarios.py:enforce_move_start_continuity"),
    ("events[].actions[].overrides.fog", "ratio", "configured_weather_override",
     "Authored fog-intensity fraction passed to the deterministic weather handler; the handler mirrors it to fog_density.",
     "Dataset/tools/batch_generate.py:weather_payload"),
    ("events[].actions[].overrides.rain", "ratio", "configured_weather_override",
     "Authored rain-intensity fraction passed to the deterministic weather handler.",
     "Dataset/tools/batch_generate.py:weather_payload"),
    ("events[].actions[].overrides.visibility_m", "m", "configured_weather_override",
     "Authored visibility-distance override passed to the deterministic weather handler.",
     "Dataset/tools/batch_generate.py:weather_payload"),
    ("events[].actions[].overrides.wetness", "ratio", "configured_weather_override",
     "Authored surface-wetness fraction passed to the deterministic weather handler.",
     "Dataset/tools/batch_generate.py:weather_payload"),
    ("events[].actions[].overrides.wind_speed", "m/s", "configured_weather_override",
     "Authored scalar wind-speed override passed to the deterministic weather handler.",
     "Dataset/tools/batch_generate.py:weather_payload"),
    ("events[].actions[].position_enu_m[]", "m", "planned_spawn_position",
     "One component of the authored spawn position consumed by the spawn_entity handler; it is a command target, not an observed pose.",
     "Dataset/tools/batch_generate.py:_handle_spawn_entity"),
    ("events[].actions[].rotation_deg.yaw_deg", "deg", "planned_spawn_heading",
     "Authored spawn yaw consumed by the spawn_entity handler.",
     "Dataset/tools/batch_generate.py:_handle_spawn_entity"),
    ("events[].actions[].state_patch.communication_state.availability", "ratio",
     "scheduled_runtime_link_availability",
     "Authored runtime communication availability fraction queued by set_runtime_state and constrained to the interval from zero to one.",
     "Dataset/tools/runtime_state_contract.py + Dataset/tools/batch_generate.py:_handle_set_runtime_state"),
    ("events[].actions[].state_patch.facility_state.request_count", "count",
     "scheduled_facility_request_count",
     "Authored nonnegative facility request count queued by set_runtime_state.",
     "Dataset/tools/runtime_state_contract.py + Dataset/tools/batch_generate.py:_handle_set_runtime_state"),
):
    _declare_quantity("formal_source_script", _path, _unit, _basis, _role, _meaning)

# These L4 rows are written by the discriminated geometric-state producers in
# predicate_state_computers.py. Each declaration below names the exact stored
# quantity; nullable fields retain the same unit when a numeric value is present.
_ARM_DOMAIN_GEOMETRY_BASIS = (
    "Dataset/semantic_simulation/predicate_state_computers.py:"
    "GeometricStateComputer.compute/PlanWindowComputer.compute"
)
for _path, _unit, _role, _meaning in (
    ("values.altitude_deviation_m", "m", "absolute_altitude_deviation",
     "Absolute difference between the aircraft position z component and its assigned altitude."),
    ("values.assigned_altitude_m", "m", "planned_route_altitude",
     "Assigned map-local z altitude selected from the aircraft roster corridor, global-flow, or motion contract."),
    ("values.assigned_landing_zone_pose_enu_m[]", "m", "assigned_landing_position",
     "One component of the resolved assigned landing-zone or landing-pad position."),
    ("values.boundary_margin_m", "m", "governed_boundary_margin",
     "Governed restricted-airspace boundary margin read from the source script contract."),
    ("values.boundary_polygon_enu_m[][]", "m", "restricted_boundary_vertex",
     "One coordinate component of a restricted-region boundary polygon vertex."),
    ("values.corridor_capacity", "count", "corridor_aircraft_capacity",
     "Aircraft capacity declared by the active corridor geometry."),
    ("values.corridor_occupancy_count", "count", "corridor_aircraft_occupancy",
     "Number of aircraft whose sampled positions occupy the corridor at this tick."),
    ("values.distance_m", "m", "grounded_pair_distance",
     "Three-dimensional Euclidean distance between the two explicitly grounded entities in this relation row."),
    ("values.distance_to_boundary_m", "m", "unsigned_restricted_boundary_distance",
     "Unsigned three-dimensional distance from the aircraft position to the restricted-region prism boundary."),
    ("values.distance_to_protected_airspace_m", "m", "unsigned_protected_airspace_distance",
     "Unsigned three-dimensional distance from the aircraft position to the protected-airspace prism."),
    ("values.home_pad_pose_enu_m[]", "m", "home_pad_position",
     "One component of the resolved home-pad position used by the aircraft geometry row."),
    ("values.horizontal_distance_to_boundary_m", "m", "planar_restricted_boundary_distance",
     "Planar distance from the aircraft position to the restricted-region boundary polygon."),
    ("values.local_ground_reference_z_m", "m", "local_ground_reference_z",
     "Local map z reference selected from landing contact, the nearest contacted pad, or the ENU ground surface."),
    ("values.nearest_aircraft_distance_m", "m", "nearest_aircraft_separation_distance",
     "Minimum governed aircraft separation distance from the subject, using the pair model when one is declared."),
    ("values.nearest_building_distance_m", "m", "nearest_building_euclidean_distance",
     "Minimum three-dimensional Euclidean distance from the subject to a building structure."),
    ("values.nearest_ground_vehicle_distance_m", "m", "nearest_ground_vehicle_euclidean_distance",
     "Minimum three-dimensional Euclidean distance from the subject to a ground vehicle."),
    ("values.nearest_pedestrian_distance_m", "m", "nearest_pedestrian_euclidean_distance",
     "Minimum three-dimensional Euclidean distance from the subject to a pedestrian."),
    ("values.nearest_population_distance_m", "m", "nearest_population_euclidean_distance",
     "Minimum three-dimensional Euclidean distance from the subject to either a vehicle or pedestrian."),
    ("values.nearest_vehicle_distance_m", "m", "nearest_vehicle_euclidean_distance",
     "Minimum three-dimensional Euclidean distance from the subject to a vehicle."),
    ("values.pair_distance_m", "m", "governed_aircraft_pair_distance",
     "Aircraft-pair separation value used by the predicate contract: Euclidean distance without a pair model, otherwise its anisotropic equivalent."),
    ("values.pair_euclidean_distance_m", "m", "aircraft_pair_euclidean_distance",
     "Three-dimensional Euclidean distance between the two aircraft."),
    ("values.pair_horizontal_distance_m", "m", "aircraft_pair_horizontal_distance",
     "Planar XY distance between the two aircraft."),
    ("values.pair_horizontal_limit_m", "m", "aircraft_pair_horizontal_limit",
     "Horizontal separation limit declared by the aircraft pair's anisotropic safety model."),
    ("values.pair_vertical_distance_m", "m", "aircraft_pair_vertical_distance",
     "Absolute z separation between the two aircraft."),
    ("values.pair_vertical_limit_m", "m", "aircraft_pair_vertical_limit",
     "Vertical separation limit declared by the aircraft pair's anisotropic safety model."),
    ("values.position_enu_m[]", "m", "subject_truth_position",
     "One component of the subject aircraft truth position copied from its trajectory sample."),
    ("values.signed_distance_to_boundary_m", "m", "signed_restricted_boundary_distance",
     "Restricted-prism boundary distance with negative sign only while the point is inside both the polygon and vertical extent."),
    ("values.speed_mps", "m/s", "subject_speed",
     "Magnitude of the subject aircraft velocity vector at this tick."),
    ("values.takeoff_ground_z_m", "m", "takeoff_ground_reference_z",
     "Map z coordinate of the aircraft takeoff ground plane resolved by the geometry producer."),
    ("values.velocity_enu_mps[]", "m/s", "subject_velocity",
     "One component of the subject aircraft truth velocity copied from its trajectory sample."),
    ("values.xy_distance_to_assigned_landing_zone_m", "m", "planar_landing_zone_distance",
     "Planar XY distance from the aircraft position to its assigned landing-zone position."),
    ("values.xy_distance_to_home_pad_m", "m", "planar_home_pad_distance",
     "Planar XY distance from the aircraft position to its home-pad position."),
    ("values.z_agl_m", "m", "height_above_local_ground",
     "Aircraft position z minus the producer-selected local ground-reference z."),
):
    _declare_quantity("arm_domain_state_ticks", _path, _unit,
                      _ARM_DOMAIN_GEOMETRY_BASIS, _role, _meaning)

# L6 history is a frozen prefix of the same in-memory records produced by
# semantics.evaluate. The following tables enumerate those copies and the
# values constructed directly by that producer; no path-name inference is used.
_HISTORY_BASIS = (
    "Dataset/tools/l6_v2/semantics.py:evaluate/producer_observations/local_frames + "
    "Dataset/tools/l6_v2/observations.py:window_semantics"
)

def _reuse_history_quantity(target_path: str, source_family: str,
                            source_path: str) -> None:
    source_key = source_family, source_path
    unit, basis, role, _frame = UNIT_RULES[source_key]
    _declare_quantity(
        "arm_semantic_history", target_path, unit,
        basis + "; " + _HISTORY_BASIS, role, EXACT_MEANINGS[source_key],
    )

_declare_quantity(
    "arm_semantic_history", "end_exclusive_tick", "simulation_tick",
    "Dataset/tools/l6_v2/observations.py:window_semantics", "time_point",
    "Exclusive upper tick of the saved pre-window history; it equals the ARM window start tick.",
)
for _path in ("from_tick", "to_tick"):
    _reuse_history_quantity(
        "objective.predicate_transitions[]." + _path,
        "arm_predicate_transitions", _path,
    )

for _prefix in ("raw.communication_state[].", "raw.communication_ticks[]."):
    _declare_quantity(
        "arm_semantic_history", _prefix + "tick", "simulation_tick",
        _HISTORY_BASIS, "time_point",
        "Episode tick at which this saved communication-state row was evaluated.",
    )
    for _path in (
        "bandwidth.allocated_mbps", "bandwidth.dropped_mbps",
        "bandwidth.requested_mbps", "channel.allocated_bandwidth_mbps",
        "channel.capacity_mbps", "channel.nominal_capacity_mbps",
        "channel.queued_bandwidth_mbps", "handover_policy.candidate_distance_m",
        "handover_policy.current_station_distance_m",
        "handover_policy.distance_advantage_m",
        "handover_policy.dwell_elapsed_ticks",
        "handover_policy.handover_margin_m",
        "handover_policy.last_switch_tick",
        "handover_policy.minimum_dwell_ticks", "heartbeat_age_ms",
        "link_quality.availability", "link_quality.capacity_factor",
        "link_quality.distance_m", "link_quality.effective_range_m",
        "link_quality.latency_ms", "link_quality.operational_factor",
        "link_quality.packet_loss_ratio", "link_quality.quality_score",
        "link_quality.range_factor", "link_quality.weather_attenuation",
        "quality_thresholds.latency_ms",
        "quality_thresholds.packet_loss_ratio", "retransmission.count",
        "route.distance_m", "station_operational_state.availability",
        "station_operational_state.capacity_factor",
        "station_operational_state.operational_factor",
        "station_operational_state.range_factor",
    ):
        _reuse_history_quantity(_prefix + _path, "communication_state", _path)

_HISTORY_CONTROL_QUANTITIES = (
    ("altitude_error_m", "m", "signed_altitude_error",
     "Current position z minus assigned altitude; negative values are below the assigned altitude."),
    ("home_distance_m", "m", "current_planar_home_distance",
     "Current planar XY distance from the subject to its home position."),
    ("previous_home_distance_m", "m", "previous_planar_home_distance",
     "Planar XY distance from the prior formal sample to the subject's home position."),
    ("previous_route_distance_m", "m", "previous_planar_route_distance",
     "Minimum planar XY distance from the prior formal sample to the planned route polyline."),
    ("route_distance_m", "m", "current_planar_route_distance",
     "Minimum planar XY distance from the current sample to the planned route polyline."),
    ("speed_delta_mps", "m/s", "sample_to_sample_speed_decrease",
     "Previous scalar speed minus current scalar speed; positive values denote slowing."),
)
for _name, _unit, _role, _meaning in _HISTORY_CONTROL_QUANTITIES:
    for _prefix in (
        "raw.control_observations[].values.",
        "objective.predicate_truth[].evidence.observations[].value.control.",
    ):
        _declare_quantity("arm_semantic_history", _prefix + _name, _unit,
                          _HISTORY_BASIS, _role, _meaning)
_declare_quantity(
    "arm_semantic_history", "raw.control_observations[].tick", "simulation_tick",
    _HISTORY_BASIS, "time_point",
    "Episode tick at which this saved control-response observation was evaluated.",
)

_HISTORY_GNSS_QUANTITIES = (
    ("drift_offset_enu_m[]", "m", "modeled_gnss_position_offset",
     "One component of the modeled GNSS error vector added to truth position."),
    ("position_error_m", "m", "modeled_gnss_position_error",
     "Magnitude of the modeled GNSS horizontal position error."),
    ("recovery_decay", "ratio", "gnss_error_recovery_decay",
     "Configured multiplicative carry-over factor used when modeled GNSS error recovers toward its target."),
    ("reported_position_enu_m[]", "m", "modeled_gnss_reported_position",
     "One component of truth position plus the modeled GNSS error vector."),
    ("truth_position_enu_m[]", "m", "subject_truth_position",
     "One component of the subject UAV truth position before modeled GNSS error is added."),
)
_HISTORY_SECURITY_QUANTITIES = (
    ("auth_score", "ratio", "authentication_score",
     "Command-source authentication score derived from explicit compromise and unauthorized-command state."),
    ("command_integrity_score", "ratio", "command_integrity_score",
     "Command-integrity score derived from the explicit integrity-violation state."),
    ("lockout_duration_ticks", "simulation_tick", "duration",
     "Accumulated formal-sample tick duration for which command lockout has remained active."),
    ("spectrum_interference_ratio", "ratio", "spectrum_interference_indicator_ratio",
     "Modeled interference ratio: one for explicit jamming, zero for explicit no jamming, otherwise unknown."),
)
for _name, _unit, _role, _meaning in _HISTORY_GNSS_QUANTITIES:
    for _prefix in (
        "raw.domain_observations[].values.",
        "objective.predicate_truth[].evidence.observations[].value.domain.gnss_navigation.",
    ):
        _declare_quantity("arm_semantic_history", _prefix + _name, _unit,
                          _HISTORY_BASIS, _role, _meaning)
for _name, _unit, _role, _meaning in _HISTORY_SECURITY_QUANTITIES:
    for _prefix in (
        "raw.domain_observations[].values.",
        "objective.predicate_truth[].evidence.observations[].value.domain.security_command.",
    ):
        _declare_quantity("arm_semantic_history", _prefix + _name, _unit,
                          _HISTORY_BASIS, _role, _meaning)
_declare_quantity(
    "arm_semantic_history", "raw.domain_observations[].tick", "simulation_tick",
    _HISTORY_BASIS, "time_point",
    "Episode tick at which this saved domain-state observation was evaluated.",
)

for _path, _unit, _role, _meaning in (
    ("assigned_altitude_m", "m", "planned_route_altitude",
     "Assigned aircraft z altitude copied from the L6 geometry contract."),
    ("local_ground_reference_z_m", "m", "local_ground_reference_z",
     "Aircraft ground-reference z copied from the L6 geometry contract."),
    ("position_z_m", "m", "subject_position_z",
     "Z component of the subject trajectory position at the predicate tick."),
    ("speed_mps", "m/s", "subject_speed",
     "Magnitude of the subject trajectory velocity vector at the predicate tick."),
    ("xy_distance_to_assigned_landing_zone_m", "m", "planar_landing_zone_distance",
     "Planar XY distance from the subject to its assigned landing zone at the predicate tick."),
    ("xy_distance_to_home_pad_m", "m", "planar_home_pad_distance",
     "Planar XY distance from the subject to its home pad at the predicate tick."),
    ("z_agl_m", "m", "height_above_local_ground",
     "Subject trajectory z minus the L6 geometry contract's ground-reference z."),
):
    _declare_quantity(
        "arm_semantic_history",
        "objective.predicate_truth[].evidence.observations[].value.geometry." + _path,
        _unit, _HISTORY_BASIS, _role, _meaning,
    )
_declare_quantity(
    "arm_semantic_history", "objective.predicate_truth[].tick", "simulation_tick",
    _HISTORY_BASIS, "time_point",
    "Formal-grid episode tick of the saved predicate assertion.",
)

for _target, _source in (
    ("raw.local_frames[].entities[].assigned_altitude_m", "assigned_altitude_m"),
    ("raw.local_frames[].entities[].planned_route_waypoints_enu_m[][]",
     "planned_route_waypoints_enu_m[][]"),
    ("raw.local_frames[].entities[].truth_pose.position_enu_m[]", "pos_enu[]"),
    ("raw.local_frames[].entities[].truth_pose.velocity_enu_mps[]", "vel_mps[]"),
):
    _reuse_history_quantity(_target, "arm_trajectories", _source)
for _path in (
    "raw.local_frames[].entities[].communication_state.availability",
    "raw.local_frames[].entities[].communication_state.link_availability",
):
    _declare_quantity(
        "arm_semantic_history", _path, "ratio", _HISTORY_BASIS,
        "runtime_link_availability",
        "Runtime communication availability fraction carried from the trajectory row and constrained to the interval from zero to one.",
    )
_declare_quantity(
    "arm_semantic_history", "raw.local_frames[].sim_time_s", "s",
    "Dataset/tools/l6_v2/semantics.py:local_frames", "time_point",
    "Episode-local time computed as tick divided by the fixed ten-hertz L6 engine clock.",
)
_declare_quantity(
    "arm_semantic_history", "raw.local_frames[].tick", "simulation_tick",
    _HISTORY_BASIS, "time_point",
    "Episode tick represented by this saved selected-entity local frame.",
)

for _path in (
    "activation_tick", "assigned_altitude_m", "ground_reference_z_m",
    "lifecycle.assigned_altitude_m", "lifecycle.home_hover_enu_m[]",
    "lifecycle.mission_start_enu_m[]", "planned_route_waypoints_enu_m[][]",
    "pos_enu[]", "route_waypoints_enu_m[][]",
    "semantic_scope.service_capacity", "uav_corridor.altitude_layers_m[]",
    "uav_corridor.assigned_altitude_m", "vel_mps[]", "yaw_deg",
):
    _reuse_history_quantity("raw.trajectories[]." + _path,
                            "arm_trajectories", _path)
for _path, _unit, _role, _meaning in (
    ("communication_state.availability", "ratio", "runtime_link_availability",
     "Runtime communication availability fraction constrained to the interval from zero to one."),
    ("communication_state.link_availability", "ratio", "runtime_link_availability",
     "Runtime link-availability fraction constrained to the interval from zero to one."),
    ("facility_state.capacity", "service_slot", "facility_service_capacity",
     "Nonnegative concurrent service capacity carried by the validated runtime facility state."),
    ("facility_state.request_count", "count", "facility_request_count",
     "Nonnegative request count carried by the validated runtime facility state."),
    ("incident_state.hazard_concentration_ppm", "ppm", "hazard_concentration",
     "Nonnegative hazard concentration carried by the validated runtime incident state."),
    ("incident_state.hazard_radius_m", "m", "hazard_radius",
     "Nonnegative hazard radius carried by the validated runtime incident state."),
    ("path_deviation_contract.minimum_route_distance_m", "m",
     "route_deviation_threshold",
     "Minimum planar route distance configured by the L6 path-deviation contract."),
):
    _declare_quantity("arm_semantic_history", "raw.trajectories[]." + _path,
                      _unit, _HISTORY_BASIS, _role, _meaning)
_declare_quantity(
    "arm_semantic_history", "raw.trajectories[].tick", "simulation_tick",
    _HISTORY_BASIS, "time_point",
    "Episode tick of the trajectory row saved in the pre-window history.",
)

for _target, _source in (
    ("dust", "dust"), ("fog", "fog_density"),
    ("fog_density", "fog_density"),
    ("hazard_concentration_ppm", "hazard_concentration_ppm"),
    ("hazard_radius_m", "hazard_radius_m"),
    ("illumination_lux", "illumination_lux"), ("rain", "rain"),
    ("temperature_c", "temperature_c"),
    ("visibility", "visibility_m"), ("visibility_m", "visibility_m"),
    ("wetness", "wetness"), ("wind_direction_deg", "wind_direction_deg"),
    ("wind_speed", "wind_speed"),
):
    _reuse_history_quantity(
        "raw.weather[]." + _target, "formal_weather_meta", _source,
    )
_declare_quantity(
    "arm_semantic_history", "raw.weather[].tick", "simulation_tick",
    _HISTORY_BASIS, "time_point",
    "Episode tick of the weather state saved in the pre-window history.",
)

# Remaining ARM numeric meanings are tied to the concrete producers below.
# Generic visual-state builder leaves and weather-trigger values stay
# undeclared because one source path can carry unrelated quantities.
_ARM_REMAINING_STATE_BASIS = (
    "Dataset/tools/batch_generate.py:preserved_fields_from/_row_for_entity + "
    "Dataset/tools/l5_arm_common.py:states_from_rows + "
    "Dataset/tools/runtime_state_contract.py:invalid_runtime_state_value_paths"
)
for _family in ("arm_states", "arm_trajectories"):
    for _path, _unit, _role, _meaning in (
        ("communication_state.availability", "ratio", "runtime_link_availability",
         "Runtime communication availability fraction constrained to the interval from zero to one."),
        ("communication_state.link_availability", "ratio", "runtime_link_availability",
         "Runtime link-availability fraction constrained to the interval from zero to one."),
        ("facility_state.capacity", "service_slot", "facility_service_capacity",
         "Nonnegative concurrent service capacity carried by the validated runtime facility state."),
        ("facility_state.request_count", "count", "facility_request_count",
         "Nonnegative request count carried by the validated runtime facility state."),
        ("incident_state.hazard_concentration_ppm", "ppm", "hazard_concentration",
         "Nonnegative hazard concentration carried by the validated runtime incident state."),
        ("incident_state.hazard_radius_m", "m", "hazard_radius",
         "Nonnegative hazard radius carried by the validated runtime incident state."),
    ):
        _declare_quantity(_family, _path, _unit, _ARM_REMAINING_STATE_BASIS, _role, _meaning)
    _declare_quantity(_family, "inspect_altitude_m", "m", _INSPECT_BASIS,
        "planned_route_altitude", "Fixed map-ENU z altitude selected for the inspect-route contract.")
    _declare_quantity(_family, "min_path_length_m", "m", _INSPECT_BASIS,
        "required_path_length", "Contract lower bound on the inspect entity's planned route length.")
    _declare_quantity(
        _family, "path_deviation_contract.minimum_route_distance_m", "m",
        "Dataset/tools/regenerate_boundary_scenarios.py:build_l2/build_l6 path_deviation_contract + "
        "Dataset/tools/batch_generate.py:preserved_fields_from/_row_for_entity",
        "route_deviation_threshold",
        "Minimum planar distance from the planned route required by the carried path-deviation contract.")

_EVENT_ACTOR_MOTION_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:"
    "event_actor_entry/relocate_event_actor_entries_outside_capture_boundary + "
    "Dataset/tools/batch_generate.py:preserved_fields_from/_row_for_entity"
)
for _family in ("arm_states", "arm_trajectories"):
    for _path, _unit, _role, _meaning in (
        ("event_actor_motion_contract.arrival_tick", "simulation_tick", "time_point",
         "Authored semantic-arrival tick used to solve the continuous event-actor entry."),
        ("event_actor_motion_contract.entry_capture_boundary_clearance_m", "m", "entry_boundary_clearance",
         "Planar distance from the staged entry position to the capture-polygon boundary."),
        ("event_actor_motion_contract.entry_speed_mps", "m/s", "configured_entry_speed",
         "Constant route speed selected so the event actor reaches its semantic arrival at the authored tick."),
        ("event_actor_motion_contract.entry_tick", "simulation_tick", "time_point",
         "Tick-quantized start coordinate of the event actor's continuous entry route."),
        ("event_actor_motion_contract.minimum_visible_motion_ratio_before_event", "ratio",
         "required_visible_motion_fraction",
         "Required fraction of the entry motion that must be visible before the semantic event."),
        ("event_actor_motion_contract.route_length_m", "m", "planned_entry_route_length",
         "Polyline length of the generated event-actor entry route."),
    ):
        _declare_quantity(_family, _path, _unit, _EVENT_ACTOR_MOTION_BASIS, _role, _meaning)

for _path in (
    "runtime_state.communication_state.availability",
    "runtime_state.communication_state.link_availability",
    "values.communication_state.availability",
    "values.communication_state.link_availability",
):
    _declare_quantity("arm_states", _path, "ratio", _ARM_REMAINING_STATE_BASIS,
        "runtime_link_availability",
        "Runtime communication availability fraction constrained to the interval from zero to one.")

_LOCAL_BUSINESS_BASIS = "Dataset/tools/l4_incident_observations.py:evaluate"
for _path, _unit, _role, _meaning in (
    ("metrics.arrival_dispatch_tick", "simulation_tick", "time_point",
     "Observed dispatch tick of the action that captures ambulance arrival."),
    ("metrics.destination_distance_m.{entity}", "m", "destination_distance",
     "Three-dimensional distance from one evacuation subject's measured position to its authored destination."),
    ("metrics.destinations_enu_m.{entity}[]", "m", "planned_destination_position",
     "One component of an evacuation subject's authored destination."),
    ("metrics.dispatch_action_tick", "simulation_tick", "time_point",
     "Observed dispatch tick of the ambulance movement action."),
    ("metrics.distance_limit_m", "m", "proximity_threshold",
     "Authored three-dimensional ambulance-to-patient proximity limit."),
    ("metrics.distance_m", "m", "measured_entity_distance",
     "Measured three-dimensional ambulance-to-patient distance when both subjects are present."),
    ("metrics.handoff_delay_ticks", "simulation_tick", "duration",
     "Authored delay from ambulance arrival to responder handoff."),
    ("metrics.handoff_dispatch_tick", "simulation_tick", "time_point",
     "Observed dispatch tick of the action that captures responder handoff."),
    ("metrics.horizontal_distance_m", "m", "measured_horizontal_distance",
     "Measured planar separation between the aircraft and pedestrian."),
    ("metrics.horizontal_limit_m", "m", "horizontal_proximity_threshold",
     "Authored horizontal separation limit of the proximity trigger."),
    ("metrics.landing_dispatch_tick", "simulation_tick", "time_point",
     "Observed dispatch tick of the aircraft landing movement."),
    ("metrics.near_dwell_ticks", "simulation_tick", "duration",
     "Consecutive simulation ticks for which the measured subjects remain within the proximity limit."),
    ("metrics.recovery_dwell_ticks", "simulation_tick", "duration",
     "Consecutive simulation ticks outside the response volume after it has first been entered."),
    ("metrics.required_dwell_ticks", "simulation_tick", "duration",
     "Authored uninterrupted proximity dwell required by the trigger."),
    ("metrics.vertical_distance_m", "m", "measured_vertical_distance",
     "Absolute vertical separation between the aircraft and pedestrian."),
    ("metrics.vertical_limit_m", "m", "vertical_proximity_threshold",
     "Authored vertical separation limit of the proximity trigger."),
    ("values.proximity_dwell_ticks", "simulation_tick", "duration",
     "Consecutive simulation ticks for which the aircraft and pedestrian remain inside the response volume."),
):
    _declare_quantity("arm_local_business_truth", _path, _unit, _LOCAL_BUSINESS_BASIS, _role, _meaning)

_declare_quantity(
    "arm_event_outcomes", "hold_samples", "count",
    "Dataset/tools/x_arm_pipeline.py:calculate recovery outcome",
    "recovery_sample_count",
    "Number of consecutive false samples on the five-tick X predicate grid that confirmed recovery.",
)
_declare_quantity(
    "arm_event_outcomes", "terminal_hold_samples", "count",
    "Dataset/tools/l4_business_semantics.py:evaluate + "
    "Dataset/tools/l4_incident_observations.py:evaluate",
    "terminal_sample_count",
    "Number of consecutive producer-selected terminal observation samples required for the local L4 outcome.",
)
_declare_quantity(
    "arm_event_outcomes", "window_censor_tick", "simulation_tick",
    "Dataset/semantic_truth/window_outcome.py:arm_window_outcome",
    "time_point",
    "Inclusive ARM window end tick at which the event outcome is right-censored.",
)

_X_GEOMETRY_BASIS = (
    "Dataset/tools/x_arm_pipeline.py:calculate entity_proximity measurement + "
    "Dataset/world_model/graph/arm.py:measured_geometry_support projection"
)
for _path, _role, _meaning in (
    ("measurements.distance_3d_m", "measured_entity_distance",
     "Measured three-dimensional separation between the two trigger entities."),
    ("measurements.distance_xy_m", "measured_horizontal_distance",
     "Measured planar separation between the two trigger entities."),
    ("measurements.distance_z_m", "measured_vertical_distance",
     "Absolute vertical separation between the two trigger entities."),
    ("measurements.trigger.distance_m", "proximity_threshold",
     "Authored three-dimensional proximity limit copied with the supporting measurement."),
    ("measurements.trigger.horizontal_distance_m", "horizontal_proximity_threshold",
     "Authored horizontal separation limit copied with an xy-plus-z trigger."),
    ("measurements.trigger.vertical_distance_m", "vertical_proximity_threshold",
     "Authored vertical separation limit copied with an xy-plus-z trigger."),
):
    _declare_quantity("arm_chain_nodes:measured_geometry_support", _path, "m",
                      _X_GEOMETRY_BASIS, _role, _meaning)
_declare_quantity("arm_chain_nodes:measured_geometry_support",
    "measurements.trigger.min_true_ticks", "simulation_tick", _X_GEOMETRY_BASIS, "duration",
    "Authored uninterrupted true duration required by the entity-proximity trigger.")
_declare_quantity("arm_chain_nodes:measured_geometry_support", "tick", "simulation_tick",
    _X_GEOMETRY_BASIS, "time_point",
    "Episode tick of the measured geometry evidence projected into the causal chain.")
_declare_quantity("arm_chain_nodes:script_emission", "tick", "simulation_tick",
    "Dataset/tools/x_arm_pipeline.py:script emission chain node + "
    "Dataset/world_model/graph/arm.py:script_emission projection", "time_point",
    "Episode tick at which the authored script event emitted.")


_ACTION_AUDIT_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:enforce_move_start_continuity + "
    "Dataset/tools/l5_arm_engine.py:execute.dispatch + "
    "Dataset/tools/batch_generate.py:EpisodeStateEngine action handlers"
)
for _path, _unit, _role, _meaning in (
    ("action.continuity_repair.original_start_enu_m[]", "m", "original_command_start_position",
     "One component of the first authored move waypoint saved before continuity repair."),
    ("action.continuity_repair.start_gap_m", "m", "command_start_continuity_gap",
     "Three-dimensional gap between the authored first waypoint and expected dispatch position."),
    ("action.rotation_deg.pitch_deg", "deg", "authored_spawn_rotation",
     "Authored spawn pitch retained in the action audit; the current entity-spawn handler does not consume rotation_deg."),
    ("action.rotation_deg.roll_deg", "deg", "authored_spawn_rotation",
     "Authored spawn roll retained in the action audit; the current entity-spawn handler does not consume rotation_deg."),
    ("action.rotation_deg.yaw_deg", "deg", "authored_spawn_heading",
     "Authored spawn yaw retained in the action audit; the current entity-spawn handler does not consume rotation_deg."),
):
    _declare_quantity("arm_actions", _path, _unit, _ACTION_AUDIT_BASIS, _role, _meaning)

_ARM_WEATHER_FIELDS = (
    ("dust", "ratio", "dust_intensity_fraction",
     "Atmospheric dust intensity fraction constrained to the interval from zero to one."),
    ("fog", "ratio", "fog_intensity_fraction",
     "Fog-intensity alias mirrored with fog_density by the weather handler."),
    ("fog_density", "ratio", "fog_density",
     "Fog density fraction constrained to the interval from zero to one."),
    ("hazard_concentration_ppm", "ppm", "hazard_concentration",
     "Hazard-source concentration in parts per million."),
    ("hazard_radius_m", "m", "hazard_radius", "Radius of the active environmental hazard."),
    ("illumination_lux", "lux", "illuminance", "Scene illuminance in lux."),
    ("rain", "ratio", "rain_intensity_fraction", "Normalized rain-intensity fraction."),
    ("temperature_c", "degC", "temperature", "Ambient temperature in degrees Celsius."),
    ("visibility", "m", "visible_range",
     "Visibility-distance alias mirrored with visibility_m by the weather handler."),
    ("visibility_m", "m", "visible_range", "Atmospheric visibility distance."),
    ("wetness", "ratio", "surface_wetness_fraction", "Normalized surface-wetness fraction."),
    ("wind_direction_deg", "deg", "wind_direction_angle", "Authored wind-direction angle."),
    ("wind_speed", "m/s", "wind_speed", "Scalar wind speed."),
)
for _prefix in ("action.overrides.", "effect.weather_before.", "effect.weather_after."):
    for _name, _unit, _role, _meaning in _ARM_WEATHER_FIELDS:
        _declare_quantity("arm_actions", _prefix + _name, _unit,
            "Dataset/tools/batch_generate.py:weather_payload + "
            "Dataset/tools/l5_arm_engine.py:execute.dispatch weather snapshots", _role, _meaning)

_RUNTIME_NUMERIC_FIELDS = (
    ("communication_state.availability", "ratio", "runtime_link_availability",
     "Runtime communication availability fraction constrained to the interval from zero to one."),
    ("communication_state.link_availability", "ratio", "runtime_link_availability",
     "Runtime link-availability fraction constrained to the interval from zero to one."),
    ("facility_state.capacity", "service_slot", "facility_service_capacity",
     "Nonnegative concurrent service capacity carried by a validated runtime-state patch."),
    ("facility_state.request_count", "count", "facility_request_count",
     "Nonnegative request count carried by a validated runtime-state patch."),
    ("incident_state.hazard_concentration_ppm", "ppm", "hazard_concentration",
     "Nonnegative hazard concentration carried by a validated runtime-state patch."),
    ("incident_state.hazard_radius_m", "m", "hazard_radius",
     "Nonnegative hazard radius carried by a validated runtime-state patch."),
)
for _prefix in (
    "action.runtime_state.",
    "action.state_patch.",
    "action.visual_state.",
    "action.visual_state.initial_state.",
    "action.visual_state.initial_state.runtime_state.",
    "action.visual_state.runtime_state.",
    "action.visual_state.visual_state.",
    "action.visual_state.visual_state.runtime_state.",
):
    for _suffix, _unit, _role, _meaning in _RUNTIME_NUMERIC_FIELDS:
        _declare_quantity("arm_actions", _prefix + _suffix, _unit,
            "Dataset/tools/runtime_state_contract.py:runtime_state_patch_from_action/"
            "invalid_runtime_state_value_paths + Dataset/tools/batch_generate.py:"
            "_handle_set_runtime_state/preserved_fields_from", _role, _meaning)

for _path in (
    "action.terminal_feasibility.available_ticks",
    "action.terminal_feasibility.dispatch_upper_bound_tick",
    "action.terminal_feasibility.home_pose_origin_z_m",
    "action.terminal_feasibility.landing_reference_enu_m[]",
    "action.terminal_feasibility.max_speed_mps",
    "action.terminal_feasibility.required_speed_mps",
    "action.terminal_feasibility.route_length_m",
    "action.terminal_feasibility.touchdown_altitude_tolerance_m",
    "action.terminal_feasibility.touchdown_dwell_ticks",
):
    _source_path = "events[].actions[]." + _path.removeprefix("action.")
    _unit, _basis, _role, _frame = UNIT_RULES["arm_script_plan", _source_path]
    _declare_quantity("arm_actions", _path, _unit, _ACTION_AUDIT_BASIS + "; " + _basis,
                      _role, EXACT_MEANINGS["arm_script_plan", _source_path])

for _path, _unit, _role, _meaning in (
    ("effect.added_schedule.source_event_tick", "simulation_tick", "time_point",
     "Dispatch tick of the source event that created the recorded schedule."),
    ("effect.added_schedule.start_pos_enu[]", "m", "dispatch_start_position",
     "One component of the entity position from which the recorded move schedule starts."),
    ("effect.added_schedule.tick", "simulation_tick", "time_point",
     "Start tick of the schedule appended by the action handler."),
    ("effect.added_schedule.velocity_mps", "m/s", "configured_speed",
     "Scalar speed stored on the appended move schedule."),
    ("effect.added_schedule.waypoints_enu_m[][]", "m", "resolved_schedule_waypoint",
     "One component of a waypoint stored on the appended movement schedule."),
    ("effect.keyframes_after[][]", "simulation_tick", "time_point",
     "Numeric member of each engine keyframe tuple: its episode tick; the sibling activity member is a string."),
    ("effect.keyframes_after[][][]", "m", "keyframe_position",
     "One position component of an engine keyframe after the action."),
    ("effect.motion_model.configured_speed_mps", "m/s", "configured_speed",
     "Scalar speed copied from the appended move schedule."),
    ("effect.motion_model.continuous_lower_bound_ticks", "simulation_tick", "duration",
     "Unrounded route travel duration computed from total distance, configured speed, and engine tick rate."),
    ("effect.motion_model.nominal_completion_tick", "simulation_tick", "time_point",
     "Schedule start tick plus the sum of per-segment quantized durations."),
    ("effect.motion_model.nominal_duration_ticks", "simulation_tick", "duration",
     "Sum of the per-segment durations after ceiling each nonzero segment."),
    ("effect.motion_model.segments[].distance_m", "m", "segment_length",
     "Three-dimensional distance of one nonzero scheduled movement segment."),
    ("effect.motion_model.segments[].ticks", "simulation_tick", "duration",
     "Ceiling-quantized duration of one nonzero scheduled movement segment."),
    ("effect.motion_model.segments[].velocity_mps[]", "m/s", "segment_velocity",
     "One component of the constant velocity that reaches the segment endpoint in its quantized duration."),
    ("effect.previous_motion_owner.dispatch_tick", "simulation_tick", "time_point",
     "Dispatch tick of the motion command that owned the entity before this action."),
    ("effect.previous_motion_owner.nominal_completion_tick", "simulation_tick", "time_point",
     "Nominal completion tick of the prior motion owner."),
    ("effect.previous_motion_owner.scheduled_tick", "simulation_tick", "time_point",
     "Schedule start tick of the prior motion owner."),
    ("motion_schedule.endpoint_enu_m[]", "m", "scheduled_terminal_position",
     "One component of the terminal keyframe position selected for the L6 movement effect."),
    ("motion_schedule.terminal_tick", "simulation_tick", "time_point",
     "Terminal keyframe tick selected for the L6 movement effect."),
):
    _declare_quantity("arm_actions", _path, _unit,
        _ACTION_AUDIT_BASIS + "; Dataset/tools/l6_v2/engine.py:motion_effect", _role, _meaning)

# The ARM ActionPlan owns these authored leaves. Their current producers were
# already reviewed for the formal author-script projection.
for _path in (
    "events[].actions[].continuity_repair.original_start_enu_m[]",
    "events[].actions[].continuity_repair.start_gap_m",
    "events[].actions[].overrides.fog",
    "events[].actions[].overrides.rain",
    "events[].actions[].overrides.visibility_m",
    "events[].actions[].overrides.wetness",
    "events[].actions[].overrides.wind_speed",
    "events[].actions[].position_enu_m[]",
    "events[].actions[].state_patch.communication_state.availability",
    "events[].actions[].state_patch.facility_state.request_count",
):
    _unit, _basis, _role, _frame = UNIT_RULES["formal_source_script", _path]
    _declare_quantity("arm_script_plan", _path, _unit, _basis, _role,
                      EXACT_MEANINGS["formal_source_script", _path])
_declare_quantity(
    "arm_script_plan", "events[].actions[].rotation_deg.yaw_deg", "deg",
    "Dataset/tools/regenerate_boundary_scenarios.py:rot/spawn_from_scene + "
    "Dataset/tools/batch_generate.py:_handle_spawn_entity",
    "authored_spawn_heading",
    "Authored spawn yaw retained in the action plan; the current entity-spawn handler does not consume rotation_deg.",
)


_ARM_SCRIPT_PARAMETER_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:scenario builders + "
    "Dataset/tools/spec_compiler.py:TriggerSpec/EventStepSpec"
)
for _path, _meaning in (
    ("parameters.anomaly_tick", "Authored episode tick at which the anomaly begins."),
    ("parameters.failure_tick", "Authored episode tick at which the failure begins."),
    ("parameters.incident_tick", "Authored episode tick at which the incident begins."),
    ("parameters.recovery_tick", "Authored nominal recovery tick retained by the script parameters."),
    ("parameters.weather_threshold_tick", "Authored episode tick associated with the weather-threshold stage."),
):
    _declare_quantity("arm_script_plan", _path, "simulation_tick",
                      _ARM_SCRIPT_PARAMETER_BASIS, "time_point", _meaning)
_declare_quantity("arm_script_plan", "parameters.contention_distance_m", "m",
    _ARM_SCRIPT_PARAMETER_BASIS, "contention_distance_threshold",
    "Authored spatial distance used to define contention in the scenario.")

_L1_CORRIDOR_CONGESTION_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:build_l1 "
    "l1_4_corridor_congestion_contract"
)
for _path, _unit, _role, _meaning in (
    ("parameters.l1_4_corridor_congestion_contract.capacity", "count",
     "corridor_aircraft_capacity",
     "Maximum number of mission aircraft permitted in the shared corridor cell."),
    ("parameters.l1_4_corridor_congestion_contract.inside_sample_positions_enu_m.{entity}[]",
     "m", "planned_inside_corridor_position",
     "One component of a mission aircraft's authored position inside the shared corridor cell."),
    ("parameters.l1_4_corridor_congestion_contract.minimum_center_separation_m",
     "m", "governed_center_separation",
     "Minimum aircraft-center separation used by the shared corridor congestion contract."),
    ("parameters.l1_4_corridor_congestion_contract.required_true_samples", "count",
     "required_true_sample_count",
     "Number of consecutive formal truth samples required by the congestion contract."),
):
    _declare_quantity("arm_script_plan", _path, _unit,
                      _L1_CORRIDOR_CONGESTION_BASIS, _role, _meaning)

_L4_6_CHAIN_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:build_l4 "
    "l4_6_physical_chain"
)
for _path, _unit, _role, _meaning in (
    ("parameters.l4_6_physical_chain.conflict_position_enu_m[]", "m",
     "planned_conflict_position",
     "One component of the SUMO-lane point selected as the pedestrian-vehicle conflict position."),
    ("parameters.l4_6_physical_chain.pedestrian_safe_clearance_m", "m",
     "required_terminal_clearance",
     "Required terminal pedestrian-to-vehicle clearance after retreat."),
    ("parameters.l4_6_physical_chain.pedestrian_safe_retreat_enu_m[]", "m",
     "planned_safe_retreat_position",
     "One component of the authored sidewalk retreat position."),
    ("parameters.l4_6_physical_chain.source_proximity_distance_m", "m",
     "proximity_threshold",
     "Three-dimensional proximity threshold copied from the source conflict trigger."),
    ("parameters.l4_6_physical_chain.source_proximity_required_ticks", "simulation_tick",
     "duration", "Required uninterrupted duration of the source pedestrian-vehicle proximity trigger."),
    ("parameters.l4_6_physical_chain.vehicle_approach_speed_mps", "m/s",
     "configured_vehicle_speed", "Configured vehicle speed on the physical approach route."),
    ("parameters.l4_6_physical_chain.vehicle_braking_speed_mps", "m/s",
     "configured_vehicle_speed", "Configured vehicle speed on the braking route."),
    ("parameters.l4_6_physical_chain.vehicle_max_sumo_projection_error_m", "m",
     "maximum_lane_projection_error",
     "Maximum allowed separation between the vehicle pose and its directed SUMO lane projection."),
    ("parameters.l4_6_physical_chain.vehicle_physical_lane_lateral_m", "m",
     "physical_lane_lateral_offset", "Selected vehicle-body center offset from the SUMO lane center."),
    ("parameters.l4_6_physical_chain.vehicle_reposition_start_s_m", "m",
     "lane_longitudinal_position", "SUMO lane-s coordinate used to reposition the staged vehicle entry."),
    ("parameters.l4_6_physical_chain.vehicle_start_s_m", "m",
     "lane_longitudinal_position", "SUMO lane-s coordinate of the authored vehicle start."),
    ("parameters.l4_6_physical_chain.vehicle_stop_s_m", "m",
     "lane_longitudinal_position", "SUMO lane-s coordinate of the vehicle's planned stop."),
    ("parameters.l4_6_physical_chain.vehicle_stops_short_of_conflict_m", "m",
     "planned_stop_clearance",
     "Longitudinal distance by which the vehicle stops short of the selected conflict point."),
):
    _declare_quantity("arm_script_plan", _path, _unit, _L4_6_CHAIN_BASIS, _role, _meaning)

_declare_quantity("arm_script_plan", "parameters.path_deviation_min_m", "m",
    "Dataset/tools/regenerate_boundary_scenarios.py:build_l2/build_l6",
    "route_deviation_threshold",
    "Minimum planar distance from the planned route used to construct the deviation scenario.")
for _path, _role, _meaning in (
    ("parameters.proximity_reachability_normalization[].distance_m",
     "normalized_proximity_threshold",
     "Reachable three-dimensional proximity threshold after deterministic route normalization."),
    ("parameters.proximity_reachability_normalization[].horizontal_distance_m",
     "normalized_horizontal_proximity_threshold",
     "Reachable horizontal threshold after deterministic xy-plus-z trigger normalization."),
    ("parameters.proximity_reachability_normalization[].vertical_distance_m",
     "normalized_vertical_proximity_threshold",
     "Reachable vertical threshold after deterministic xy-plus-z trigger normalization."),
):
    _declare_quantity("arm_script_plan", _path, "m",
        "Dataset/tools/regenerate_boundary_scenarios.py:normalize_proximity_trigger_reachability",
        _role, _meaning)

_RELEASE_SPATIAL_BASIS = (
    "Dataset/tools/regenerate_boundary_scenarios.py:"
    "build_roi_bound_bundle/_assignment_outside_roi_m"
)
for _path, _unit, _role, _meaning in (
    ("parameters.release_spatial_authority.actual_assignment_center_enu_m[]", "m",
     "actual_assignment_center",
     "One component of the generated event bundle's actual assignment center."),
    ("parameters.release_spatial_authority.assignment_outside_roi_m", "m",
     "assignment_outside_roi_distance",
     "Planar distance by which the actual assignment center lies outside the formal ROI polygon; zero when covered."),
    ("parameters.release_spatial_authority.attempts[].actual_assignment_center_enu_m[]", "m",
     "actual_assignment_center",
     "One component of the actual assignment center recorded for this placement attempt."),
    ("parameters.release_spatial_authority.attempts[].assignment_outside_roi_m", "m",
     "assignment_outside_roi_distance",
     "Planar distance outside the formal ROI polygon for this placement attempt."),
    ("parameters.release_spatial_authority.attempts[].attempt", "index",
     "assignment_attempt_index", "One-based ordinal of the deterministic spatial placement attempt."),
    ("parameters.release_spatial_authority.attempts[].roi_contract_center_enu_m[]", "m",
     "roi_contract_center", "One component of the formal ROI contract center recorded for this attempt."),
    ("parameters.release_spatial_authority.attempts[].target_assignment_center_enu_m[]", "m",
     "assignment_target_center", "One component of the target assignment center used for this attempt."),
    ("parameters.release_spatial_authority.roi_contract_center_enu_m[]", "m",
     "roi_contract_center", "One component of the formal ROI contract center used for release binding."),
):
    _declare_quantity("arm_script_plan", _path, _unit, _RELEASE_SPATIAL_BASIS, _role, _meaning)

_ROI_NUMERIC_BASIS = (
    "Dataset/tools/roi_contract.py:RoiContract.contract_payload/load_formal_roi_contract + "
    "Dataset/tools/regenerate_boundary_scenarios.py:build_l3 ROI minimal change evidence"
)
for _path, _unit, _role, _meaning in (
    ("parameters.roi_contract.baseline_capture_polygon_enu_m[][]", "m",
     "baseline_capture_polygon_vertex",
     "One planar component of a vertex in the baseline formal capture polygon."),
    ("parameters.roi_contract.capture_frames_per_episode", "source_frame",
     "capture_frame_count", "Number of formal capture frames declared for one episode."),
    ("parameters.roi_contract.capture_polygon_enu_m[][]", "m",
     "capture_polygon_vertex", "One planar component of a vertex in the active formal capture polygon."),
    ("parameters.roi_contract.capture_step_ticks", "simulation_tick",
     "sample_interval", "Tick interval between consecutive formal capture frames."),
    ("parameters.roi_contract.expanded_boundary_padding_m", "m",
     "runtime_crop_padding", "Distance by which the capture polygon is padded for the runtime truth crop."),
    ("parameters.roi_contract.minimal_change_evidence.camera_cross_footprint_guard_m", "m",
     "camera_footprint_guard", "Additional cross-footprint guard used by the ROI feasibility calculation."),
    ("parameters.roi_contract.minimal_change_evidence.camera_footprint_half_extents_m[]", "m",
     "camera_footprint_half_extent",
     "One half-extent of the camera footprint used in the ROI feasibility calculation."),
    ("parameters.roi_contract.minimal_change_evidence.center_translation_enu_m[]", "m",
     "roi_center_translation", "One planar component of the selected ROI center translation."),
    ("parameters.roi_contract.minimal_change_evidence.center_translation_norm_m", "m",
     "roi_center_translation_distance", "Euclidean norm of the selected ROI center translation."),
    ("parameters.roi_contract.minimal_change_evidence.entry_capture_boundary_clearance_m", "m",
     "entry_boundary_clearance",
     "Planar clearance between the event-actor entry position and capture boundary."),
    ("parameters.roi_contract.minimal_change_evidence.entry_route_length_m", "m",
     "planned_entry_route_length",
     "Length of the event-actor route used by the minimal-change feasibility proof."),
    ("parameters.roi_contract.minimal_change_evidence.entry_ticks_at_speed_ceiling",
     "simulation_tick", "duration",
     "Ceiling-quantized travel duration of the entry route at the ground-vehicle speed ceiling."),
    ("parameters.roi_contract.minimal_change_evidence.event_stage_interior_margin_m", "m",
     "event_stage_interior_margin",
     "Required interior margin between the semantic event stage and the active ROI boundary."),
    ("parameters.roi_contract.minimal_change_evidence.ground_vehicle_speed_ceiling_mps", "m/s",
     "configured_speed_limit",
     "Ground-vehicle speed ceiling used by the entry feasibility proof."),
    ("parameters.roi_contract.minimal_change_evidence.inspect_altitude_m", "m",
     "planned_route_altitude",
     "Fixed map-ENU z altitude used by the inspect-camera feasibility proof."),
    ("parameters.roi_contract.minimal_change_evidence.observation_center_clearance_m", "m",
     "observation_center_clearance",
     "Clearance retained around the observation center by the feasibility proof."),
    ("parameters.roi_contract.minimal_change_evidence.preserved_dimensions_m[]", "m",
     "preserved_roi_dimension",
     "One active ROI dimension preserved by the minimal-change replacement."),
    ("parameters.roi_contract.minimal_change_evidence.roadwork_anchor_s_m", "m",
     "lane_longitudinal_position", "SUMO lane-s coordinate selected for the roadwork anchor."),
    ("parameters.roi_contract.minimal_change_evidence.rounded_contract_min_feasibility_margin_m", "m",
     "minimum_feasibility_margin",
     "Rounded minimum geometric feasibility margin recorded by the ROI replacement proof."),
    ("parameters.roi_contract.minimal_change_evidence.search_step_m", "m",
     "translation_search_step", "Translation step used by the bounded ROI feasibility search."),
    ("parameters.roi_contract.minimal_change_evidence.semantic_stage_span_m", "m",
     "semantic_stage_span",
     "Planar span of the semantic event stage used in the ROI feasibility proof."),
    ("parameters.roi_contract.minimal_change_evidence.sensor_hfov_deg", "deg",
     "horizontal_field_of_view",
     "Horizontal field of view used by the inspect-camera footprint calculation."),
    ("parameters.roi_contract.minimal_change_evidence.sensor_resolution[]", "pixel",
     "image_axis_pixel_count", "One configured sensor-resolution axis in pixels."),
    ("parameters.roi_contract.minimal_change_evidence.translation_feasible_set_interior_guard_m", "m",
     "translation_feasible_set_guard", "Interior guard applied to the feasible translation set."),
    ("parameters.roi_contract.minimal_change_evidence.unsearched_edge_translation_lower_bound_m", "m",
     "unsearched_translation_lower_bound",
     "Lower bound on an unsearched edge translation recorded by the feasibility proof."),
    ("parameters.roi_contract.runtime_crop_bbox_enu_m[]", "m",
     "runtime_crop_bound", "One planar bound of the map-ENU runtime crop box."),
    ("parameters.roi_contract.tick_end", "simulation_tick",
     "time_point", "Inclusive final tick covered by the formal ROI contract."),
    ("parameters.roi_contract.tick_start", "simulation_tick",
     "time_point", "Inclusive first tick covered by the formal ROI contract."),
):
    _declare_quantity("arm_script_plan", _path, _unit, _ROI_NUMERIC_BASIS, _role, _meaning)


_X1_CHAIN_BASIS = "Dataset/tools/regenerate_boundary_scenarios.py:build_x x1_physical_chain"
for _path, _unit, _role, _meaning in (
    ("parameters.x1_physical_chain.crowd_clearance_radius_m", "m",
     "crowd_clearance_radius",
     "Authored clearance radius around the forced-landing point used to relocate crowd members."),
    ("parameters.x1_physical_chain.crowd_clearance_targets_enu_m.{entity}[]", "m",
     "planned_crowd_clearance_position",
     "One component of a crowd member's authored clearance target."),
    ("parameters.x1_physical_chain.forced_descent_z_m[]", "m",
     "planned_descent_altitude",
     "One map-ENU z coordinate in the authored forced-descent sequence."),
    ("parameters.x1_physical_chain.landing_point_enu_m[]", "m",
     "planned_landing_position", "One component of the authored forced-landing point."),
    ("parameters.x1_physical_chain.rain_threshold", "ratio",
     "rain_intensity_threshold", "Normalized rain-intensity threshold used by the X1 physical chain."),
    ("parameters.x1_physical_chain.tower_degraded_availability", "ratio",
     "runtime_link_availability", "Tower availability fraction during the degraded-link stage."),
    ("parameters.x1_physical_chain.tower_nominal_availability", "ratio",
     "runtime_link_availability", "Tower availability fraction during the nominal-link stage."),
    ("parameters.x1_physical_chain.tower_unavailable_availability", "ratio",
     "runtime_link_availability", "Tower availability fraction during the unavailable-link stage."),
):
    _declare_quantity("arm_script_plan", _path, _unit, _X1_CHAIN_BASIS, _role, _meaning)

_X3_CHAIN_BASIS = "Dataset/tools/regenerate_boundary_scenarios.py:build_x x3_physical_chain"
for _path, _unit, _role, _meaning in (
    ("parameters.x3_physical_chain.ambulance_lane_arrival_s_m", "m",
     "lane_longitudinal_position", "SUMO lane-s coordinate of the ambulance's planned arrival."),
    ("parameters.x3_physical_chain.ambulance_lane_start_s_m", "m",
     "lane_longitudinal_position", "SUMO lane-s coordinate of the ambulance's planned start."),
    ("parameters.x3_physical_chain.ambulance_route_sample_spacing_m", "m",
     "route_sample_spacing",
     "Longitudinal spacing used to sample the ambulance route along its SUMO lane."),
    ("parameters.x3_physical_chain.uav_detection_radius_m", "m",
     "detection_radius", "Three-dimensional UAV-to-patient radius required for physical detection."),
    ("parameters.x3_physical_chain.uav_detection_required_ticks", "simulation_tick",
     "duration",
     "Required uninterrupted UAV-to-patient proximity duration for physical detection."),
):
    _declare_quantity("arm_script_plan", _path, _unit, _X3_CHAIN_BASIS, _role, _meaning)

_X6_CHAIN_BASIS = "Dataset/tools/regenerate_boundary_scenarios.py:build_x x6_physical_chain"
for _path, _unit, _role, _meaning in (
    ("parameters.x6_physical_chain.crowd_hazard_radius_m", "m",
     "crowd_hazard_radius", "Authored radius of the initial crowd hazard-side region."),
    ("parameters.x6_physical_chain.hazard_reference_enu_m[]", "m",
     "hazard_reference_position", "One component of the fixed initial hazard reference position."),
    ("parameters.x6_physical_chain.minimum_route_target_distance_from_origin_m", "m",
     "minimum_evacuation_route_distance",
     "Minimum distance required between an evacuation route target and its origin."),
    ("parameters.x6_physical_chain.reroute_exit_anchor_enu_m[]", "m",
     "reroute_exit_position", "One component of the authored UAV reroute-exit anchor position."),
    ("parameters.x6_physical_chain.reroute_exit_radius_m", "m",
     "reroute_exit_radius", "Horizontal proximity radius of the reroute-exit anchor."),
    ("parameters.x6_physical_chain.reroute_exit_required_dwell_ticks", "simulation_tick",
     "duration", "Required uninterrupted dwell at the reroute-exit anchor."),
    ("parameters.x6_physical_chain.safe_targets_enu_m.{entity}[]", "m",
     "planned_safe_target_position",
     "One component of a pedestrian cohort member's authored safe target."),
    ("parameters.x6_physical_chain.safe_zone_anchor_enu_m[]", "m",
     "safe_zone_anchor_position", "One component of the authored safe-zone anchor position."),
    ("parameters.x6_physical_chain.safe_zone_anchor_radius_m", "m",
     "safe_zone_anchor_radius", "Proximity radius of the pedestrian safe-zone anchor."),
    ("parameters.x6_physical_chain.safe_zone_distance_m", "m",
     "safe_zone_displacement",
     "Authored displacement from the hazard-side crowd origin to the safe zone."),
    ("parameters.x6_physical_chain.safe_zone_reached_fraction", "ratio",
     "required_population_fraction",
     "Fraction of the explicit evacuation cohort required to reach the safe zone."),
    ("parameters.x6_physical_chain.safe_zone_required_dwell_ticks", "simulation_tick",
     "duration",
     "Required uninterrupted dwell of each cohort member at the safe-zone anchor."),
):
    _declare_quantity("arm_script_plan", _path, _unit, _X6_CHAIN_BASIS, _role, _meaning)

for _path, _unit, _role, _meaning in (
    ("triggers[].delay_ticks", "simulation_tick", "duration",
     "Authored delay after a referenced event before the trigger becomes true."),
    ("triggers[].distance_m", "m", "proximity_threshold",
     "Authored proximity threshold for a three-dimensional or planar entity-proximity trigger."),
    ("triggers[].horizontal_distance_m", "m", "horizontal_proximity_threshold",
     "Authored horizontal threshold of an xy-plus-z proximity trigger."),
    ("triggers[].min_true_ticks", "simulation_tick", "duration",
     "Required uninterrupted true duration of the trigger condition."),
    ("triggers[].sustain_ticks", "simulation_tick", "duration",
     "Required duration for which a weather comparison must remain satisfied."),
    ("triggers[].tick", "simulation_tick", "time_point",
     "Authored episode tick of an absolute tick trigger."),
    ("triggers[].vertical_distance_m", "m", "vertical_proximity_threshold",
     "Authored vertical threshold of an xy-plus-z proximity trigger."),
):
    _declare_quantity("arm_script_plan", _path, _unit,
                      _ARM_SCRIPT_PARAMETER_BASIS, _role, _meaning)



# ARM predicate rows below store fixed quantitative fields from six executable
# producers.  Sampled truth may wrap the same producer context under evidence,
# while dense truth retains the producer-specific context, operands, metrics, or
# measurements carrier.  Each path is enumerated because an unqualified numeric
# value has no unit authority.
_ARM_PREDICATE_ADAPTER_BASIS = (
    "Dataset/semantic_truth/minimal_semantics_adapter.py:"
    "_project_grounded_window_state/_adapt_assertions"
)
for _path, _role, _meaning in (
    ("truth_state_update_tick", "time_point",
     "Episode tick at which the stored predicate truth state was last set or changed."),
    ("evidence_update_tick", "time_point",
     "Episode tick at which the evidence supporting the stored predicate truth was last refreshed."),
):
    for _target in (_path, "evidence.observations[].value." + _path):
        _declare_quantity(
            "arm_predicate_truth", _target, "simulation_tick",
            _ARM_PREDICATE_ADAPTER_BASIS, _role, _meaning,
        )

# L6 exposes the same reviewed control, GNSS, and security observations in its
# dense context and in the sampled predicate evidence.
for _name, _unit, _role, _meaning in _HISTORY_CONTROL_QUANTITIES:
    for _family, _prefix in (
        ("arm_predicate_truth", "evidence.observations[].value.control."),
        ("arm_predicate_truth_ticks", "context.control."),
    ):
        _declare_quantity(
            _family, _prefix + _name, _unit, _HISTORY_BASIS, _role, _meaning,
        )
for _quantities, _context_name in (
    (_HISTORY_GNSS_QUANTITIES, "gnss_navigation"),
    (_HISTORY_SECURITY_QUANTITIES, "security_command"),
):
    for _name, _unit, _role, _meaning in _quantities:
        for _family, _prefix in (
            ("arm_predicate_truth",
             "evidence.observations[].value.domain." + _context_name + "."),
            ("arm_predicate_truth_ticks",
             "context.domain." + _context_name + "."),
        ):
            _declare_quantity(
                _family, _prefix + _name, _unit,
                _HISTORY_BASIS, _role, _meaning,
            )

_ARM_PREDICATE_GEOMETRY_QUANTITIES = (
    ("assigned_altitude_m", "m", "planned_route_altitude",
     "Assigned map-local z altitude used by the grounded aircraft predicate."),
    ("local_ground_reference_z_m", "m", "local_ground_reference_z",
     "Producer-selected local map z reference used to measure aircraft height above ground."),
    ("position_z_m", "m", "subject_position_z",
     "Map-local z component of the subject aircraft position at the predicate tick."),
    ("speed_mps", "m/s", "subject_speed",
     "Magnitude of the subject aircraft velocity vector at the predicate tick."),
    ("xy_distance_to_assigned_landing_zone_m", "m", "planar_landing_zone_distance",
     "Planar XY distance from the subject aircraft to its assigned landing zone."),
    ("xy_distance_to_home_pad_m", "m", "planar_home_pad_distance",
     "Planar XY distance from the subject aircraft to its home pad."),
    ("z_agl_m", "m", "height_above_local_ground",
     "Subject aircraft position z minus the producer-selected local ground-reference z."),
)
for _family, _prefix, _basis in (
    ("arm_predicate_truth", "evidence.observations[].value.geometry.",
     "Dataset/tools/l1_semantics.py:project + "
     "Dataset/tools/l2_arm_semantics.py:_ground_contexts/recompute + "
     "Dataset/tools/l5_arm_semantics.py:local_geometry/calculate + "
     "Dataset/tools/l6_v2/semantics.py:evaluate + " + _ARM_PREDICATE_ADAPTER_BASIS),
    ("arm_predicate_truth_ticks", "context.geometry.",
     "Dataset/tools/l2_arm_semantics.py:_ground_contexts/recompute + "
     "Dataset/tools/l5_arm_semantics.py:local_geometry/calculate + "
     "Dataset/tools/l6_v2/semantics.py:evaluate"),
    ("arm_predicate_truth_ticks", "operands.geometry.",
     "Dataset/tools/l1_semantics.py:project"),
):
    for _path, _unit, _role, _meaning in _ARM_PREDICATE_GEOMETRY_QUANTITIES:
        _declare_quantity(
            _family, _prefix + _path, _unit, _basis, _role, _meaning,
        )

_ARM_NEAREST_AIRCRAFT_MEANING = (
    "Aircraft separation supplied to the predicate: Euclidean distance unless "
    "the producer applies the governed anisotropic pair model, whose equivalent "
    "distance remains expressed in metres."
)
for _family, _path, _basis in (
    ("arm_predicate_truth",
     "evidence.observations[].value.derived.nearest_aircraft_distance_m",
     "Dataset/tools/l1_semantics.py:project + "
     "Dataset/tools/l2_arm_semantics.py:recompute + "
     "Dataset/tools/l6_v2/semantics.py:evaluate + " + _ARM_PREDICATE_ADAPTER_BASIS),
    ("arm_predicate_truth_ticks",
     "context.derived.nearest_aircraft_distance_m",
     "Dataset/tools/l2_arm_semantics.py:recompute + "
     "Dataset/tools/l6_v2/semantics.py:evaluate"),
    ("arm_predicate_truth_ticks",
     "operands.derived.nearest_aircraft_distance_m",
     "Dataset/tools/l1_semantics.py:project"),
    ("arm_predicate_truth", "measurements.derived.nearest_aircraft_distance_m",
     "Dataset/tools/x_arm_pipeline.py:calculate + "
     "Dataset/semantic_simulation/predicate_state_computers.py:"
     "_equivalent_aircraft_separation_distance"),
    ("arm_predicate_truth_ticks",
     "measurements.derived.nearest_aircraft_distance_m",
     "Dataset/tools/x_arm_pipeline.py:calculate + "
     "Dataset/semantic_simulation/predicate_state_computers.py:"
     "_equivalent_aircraft_separation_distance"),
):
    _declare_quantity(
        _family, _path, "m", _basis,
        "governed_aircraft_pair_distance", _ARM_NEAREST_AIRCRAFT_MEANING,
    )
for _name, _role, _meaning, _producer in (
    ("nearest_building_distance_m", "nearest_building_euclidean_distance",
     "Minimum three-dimensional Euclidean distance from the subject aircraft to a building structure.",
     "Dataset/tools/l2_arm_semantics.py:recompute"),
    ("nearest_pedestrian_distance_m", "nearest_pedestrian_euclidean_distance",
     "Minimum three-dimensional Euclidean distance from the subject aircraft to a pedestrian.",
     "Dataset/tools/l5_arm_semantics.py:calculate"),
):
    _declare_quantity(
        "arm_predicate_truth",
        "evidence.observations[].value.derived." + _name,
        "m", _producer + " + " + _ARM_PREDICATE_ADAPTER_BASIS,
        _role, _meaning,
    )
    _declare_quantity(
        "arm_predicate_truth_ticks", "context.derived." + _name,
        "m", _producer, _role, _meaning,
    )

_L2_PREDICATE_BASIS = "Dataset/tools/l2_arm_semantics.py:recompute"
for _family, _prefix in (
    ("arm_predicate_truth", "evidence.observations[].value.domain.pad_facility."),
    ("arm_predicate_truth_ticks", "context.domain.pad_facility."),
):
    _declare_quantity(
        _family, _prefix + "capacity", "aircraft",
        _L2_PREDICATE_BASIS, "facility_service_capacity",
        "Concurrent service capacity declared by the grounded pad or charging facility.",
    )
    _declare_quantity(
        _family, _prefix + "requester_count", "aircraft",
        _L2_PREDICATE_BASIS, "facility_request_count",
        "Number of active service requesters reported for the grounded pad or charging facility.",
    )

for _family, _prefix, _basis in (
    ("arm_predicate_truth", "evidence.observations[].value.scene.",
     _L2_PREDICATE_BASIS + " + Dataset/tools/l1_semantics.py:project + "
     + _ARM_PREDICATE_ADAPTER_BASIS),
    ("arm_predicate_truth_ticks", "context.scene.", _L2_PREDICATE_BASIS),
    ("arm_predicate_truth_ticks", "operands.scene.",
     "Dataset/tools/l1_semantics.py:project"),
):
    _declare_quantity(
        _family, _prefix + "maximum_corridor_capacity", "aircraft",
        _basis, "corridor_aircraft_capacity",
        "Aircraft capacity declared by the active corridor geometry.",
    )
    _declare_quantity(
        _family, _prefix + "maximum_corridor_occupancy_count", "aircraft",
        _basis, "corridor_aircraft_occupancy",
        "Number of aircraft whose sampled positions occupy the active corridor at this tick.",
    )

# L5 predicate weather contexts copy one complete row from the formal weather
# source.  The two aliases retain the exact unit of their canonical source key.
for _target, _source in (
    ("dust", "dust"),
    ("fog", "fog_density"),
    ("fog_density", "fog_density"),
    ("hazard_concentration_ppm", "hazard_concentration_ppm"),
    ("hazard_radius_m", "hazard_radius_m"),
    ("illumination_lux", "illumination_lux"),
    ("rain", "rain"),
    ("temperature_c", "temperature_c"),
    ("visibility", "visibility_m"),
    ("visibility_m", "visibility_m"),
    ("wetness", "wetness"),
    ("wind_direction_deg", "wind_direction_deg"),
    ("wind_speed", "wind_speed"),
):
    _source_key = ("formal_weather_meta", _source)
    _unit, _basis, _role, _frame = UNIT_RULES[_source_key]
    _meaning = EXACT_MEANINGS[_source_key]
    for _family, _prefix in (
        ("arm_predicate_truth", "evidence.observations[].value.weather."),
        ("arm_predicate_truth_ticks", "context.weather."),
    ):
        _declare_quantity(
            _family, _prefix + _target, _unit,
            _basis + " + Dataset/tools/l5_arm_semantics.py:calculate",
            _role, _meaning,
        )
for _family, _path in (
    ("arm_predicate_truth", "evidence.observations[].value.weather.tick"),
    ("arm_predicate_truth_ticks", "context.weather.tick"),
):
    _declare_quantity(
        _family, _path, "simulation_tick",
        "Dataset/tools/l5_arm_semantics.py:calculate",
        "time_point", "Episode tick of the copied formal weather row.",
    )

_X_PREDICATE_BASIS = "Dataset/tools/x_arm_pipeline.py:calculate"
for _family in ("arm_predicate_truth", "arm_predicate_truth_ticks"):
    for _path, _unit, _role, _meaning in (
        ("measurements.distance_3d_m", "m", "pair_euclidean_distance",
         "Three-dimensional Euclidean separation of the two entities at this tick."),
        ("measurements.distance_xy_m", "m", "pair_planar_distance",
         "Planar XY Euclidean separation of the two entities at this tick."),
        ("measurements.distance_z_m", "m", "pair_vertical_distance",
         "Absolute z separation of the two entities at this tick."),
        ("measurements.request_count", "count", "facility_request_count",
         "Observed facility request count carried by the branch runtime state."),
        ("measurements.speed_mps", "m/s", "subject_speed",
         "Magnitude of the observed entity velocity vector at this tick."),
        ("measurements.vz_mps", "m/s", "subject_vertical_velocity",
         "Signed z component of the observed entity velocity at this tick."),
        ("measurements.trigger.distance_m", "m", "authored_proximity_limit",
         "Authored three-dimensional distance limit of the evaluated entity-proximity trigger."),
        ("measurements.trigger.horizontal_distance_m", "m", "authored_horizontal_proximity_limit",
         "Authored planar distance limit of the evaluated entity-proximity trigger."),
        ("measurements.trigger.min_true_ticks", "simulation_tick", "duration",
         "Required consecutive true duration of the evaluated proximity trigger."),
        ("measurements.trigger.sustain_ticks", "simulation_tick", "duration",
         "Configured duration for which the evaluated trigger stays active after its raw condition clears."),
        ("measurements.trigger.vertical_distance_m", "m", "authored_vertical_proximity_limit",
         "Authored absolute vertical distance limit of the evaluated entity-proximity trigger."),
        ("measurements.weather.fog_density", "ratio", "fog_density",
         "Fog-density fraction supplied by the branch weather row."),
        ("measurements.weather.rain", "ratio", "rain_intensity",
         "Rain-intensity fraction supplied by the branch weather row."),
    ):
        _declare_quantity(
            _family, _path, _unit, _X_PREDICATE_BASIS, _role, _meaning,
        )

_L4_BUSINESS_BASIS = "Dataset/tools/l4_business_semantics.py:evaluate"
_L4_BUSINESS_QUANTITIES = (
    ("approach_direction_xy[]", "ratio", "unit_approach_direction_component",
     "One component of the normalized authored priority-vehicle approach direction."),
    ("approach_dispatch_tick", "simulation_tick", "time_point",
     "Episode tick at which the successful priority-approach move action was dispatched."),
    ("approach_normal_xy[]", "ratio", "unit_approach_normal_component",
     "One component of the normalized planar normal to the priority-vehicle approach direction."),
    ("closing_speed_mps", "m/s", "signed_closing_speed",
     "Relative speed projected onto the vehicle separation vector; positive values denote closing."),
    ("distance_3d_m", "m", "pair_euclidean_distance",
     "Three-dimensional Euclidean distance between the grounded aircraft and vehicle."),
    ("distance_m", "m", "trigger_metric_distance",
     "Euclidean entity separation evaluated in the two or three dimensions selected by the authored proximity trigger."),
    ("downstream_offset_m", "m", "signed_downstream_offset",
     "Priority-vehicle position relative to the yielding vehicle, projected onto the authored approach direction; nonnegative values denote passage."),
    ("height_agl_m", "m", "height_above_local_ground",
     "Aircraft position z minus its roster-declared local ground-reference z."),
    ("horizontal_distance_m", "m", "pair_planar_distance",
     "Planar XY distance between the grounded aircraft and pedestrian."),
    ("horizontal_limit_m", "m", "authored_horizontal_proximity_limit",
     "Authored planar distance limit of the evaluated entity-proximity trigger."),
    ("landing_zone_centre_enu_m[]", "m", "landing_zone_center_position",
     "One planar component of the landing-zone centre selected from the source descent target."),
    ("landing_zone_radius_m", "m", "landing_zone_radius",
     "Radius used to classify pedestrian occupancy of the landing zone."),
    ("lateral_separation_m", "m", "signed_lateral_separation",
     "Yielding-vehicle offset from the priority vehicle projected onto the approach normal."),
    ("low_altitude_limit_m", "m", "low_altitude_threshold",
     "Maximum height above local ground allowed by the low-altitude predicate."),
    ("min_yield_center_separation_m", "m", "yield_clearance_threshold",
     "Minimum absolute lateral centre separation required for measured yield clearance."),
    ("near_dwell_ticks", "simulation_tick", "duration",
     "Current consecutive duration for which the grounded proximity condition has remained true."),
    ("proximity_limit_m", "m", "authored_proximity_limit",
     "Authored three-dimensional distance limit of the aircraft-vehicle proximity trigger."),
    ("required_dwell_ticks", "simulation_tick", "duration",
     "Authored consecutive true duration required by the grounded proximity trigger."),
    ("speed_mps", "m/s", "subject_speed",
     "Magnitude of the observed subject velocity used by the business predicate."),
    ("speeds_mps.{entity}", "m/s", "entity_speed",
     "Magnitude of this grounded entity's observed velocity at the predicate tick."),
    ("stationary_tolerance_mps", "m/s", "stationary_speed_threshold",
     "Maximum observed speed accepted as stationary by the business predicate."),
    ("threshold_m", "m", "authored_proximity_limit",
     "Authored distance limit of the evaluated vehicle proximity trigger."),
    ("vertical_distance_m", "m", "pair_vertical_distance",
     "Absolute z separation between the grounded aircraft and pedestrian."),
    ("vertical_limit_m", "m", "authored_vertical_proximity_limit",
     "Authored absolute vertical distance limit of the evaluated entity-proximity trigger."),
)
for _family in ("arm_predicate_truth", "arm_predicate_truth_ticks"):
    for _path, _unit, _role, _meaning in _L4_BUSINESS_QUANTITIES:
        _declare_quantity(
            _family, "metrics." + _path, _unit,
            _L4_BUSINESS_BASIS, _role, _meaning,
        )

_L1_PAIR_BASIS = (
    "Dataset/tools/l1_semantics.py:project + "
    "Dataset/semantic_simulation/predicate_state_computers.py:"
    "_aircraft_pair_separation_models/_equivalent_aircraft_separation_distance"
)
for _path, _role, _meaning in (
    ("pair_positions_enu_m[][]", "aircraft_pair_position",
     "One map-local ENU component of either grounded aircraft position at the predicate tick."),
    ("separation_model.horizontal_limit_m", "aircraft_pair_horizontal_limit",
     "Horizontal separation limit declared by the grounded aircraft pair's anisotropic safety model."),
    ("separation_model.separation_margin_m", "governed_separation_margin",
     "Governed scalar separation margin used to express the anisotropic equivalent distance in metres."),
    ("separation_model.vertical_limit_m", "aircraft_pair_vertical_limit",
     "Vertical separation limit declared by the grounded aircraft pair's anisotropic safety model."),
):
    _declare_quantity(
        "arm_predicate_truth_ticks", _path, "m",
        _L1_PAIR_BASIS, _role, _meaning,
    )

# The raw tagged-union carrier remains the source field for boolean and string
# observations.  Numeric branches receive path-qualified meanings and units
# from the executable predicate contracts in ``unit_policy`` below.
EXACT_MEANINGS[("arm_predicate_truth", WORLD_OBSERVATION_VALUE_PATH)] = (
    "Boolean or categorical world-observation value carried with its sibling path in sampled ARM predicate evidence."
)

_WORLD_OBSERVATION_UNIT_BASIS = (
    "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json:"
    "templates[].state_contract.required_fields + "
    "Dataset/tools/arm_window_semantics.py:predicate_rows.evidence.world_observations"
)


def _world_observation_unit_rule(source_family: str, field_path: str
                                 ) -> tuple[str, str, str, None] | None:
    path_key = world_observation_path_key(source_family, field_path)
    if path_key is None:
        return None
    contract = world_observation_numeric_contracts()[path_key]
    return contract["unit"], _WORLD_OBSERVATION_UNIT_BASIS, "predicate_world_observation", None


def _world_observation_meaning(source_family: str, field_path: str) -> str | None:
    path_key = world_observation_path_key(source_family, field_path)
    if path_key is None:
        return None
    contract = world_observation_numeric_contracts()[path_key]
    predicates = ", ".join(contract["predicate_ids"])
    sources = ", ".join(contract["sources"])
    return (
        f"Numeric {path_key} world observation carried by sampled ARM predicate evidence; "
        f"the executable predicate contracts assign unit {contract['unit']} for {predicates}, "
        f"from {sources}."
    )

# A record's direct `tick` is the sample clock coordinate. Nested tick-like
# names need their own declaration because they can mean a duration or a bound.
DIRECT_TICK_FAMILIES = frozenset({
    "formal_truth_frame", "formal_predicate_truth", "formal_world_truth_graph_deltas",
    "predicate_truth", "predicate_truth_matrix", "compute_state", "communication_state",
    "domain_state", "arm_actions", "arm_states", "arm_trajectories",
    "arm_predicate_truth", "arm_predicate_truth_ticks", "arm_domain_state_ticks",
    "arm_event_occurrences",
    "arm_local_business_truth", "formal_weather_meta",
})

# The common clock axis does not make a dispatched action, an event onset, and
# a graph delta the same kind of record. Keep that role on each source mapping.
DIRECT_TICK_SOURCE_ROLES = {
    "arm_actions": "action_dispatch",
    "arm_event_occurrences": "event_occurrence",
    "formal_world_truth_graph_deltas": "graph_delta_application",
}


def source_time_role(source_family: str, field_path: str) -> str | None:
    if field_path != "tick" or source_family not in DIRECT_TICK_FAMILIES:
        return None
    return DIRECT_TICK_SOURCE_ROLES.get(source_family, "sample_record")


def coordinate_policy(source_family: str, field_path: str) -> dict[str, Any]:
    if (source_family, field_path) in FORMAL_POSE_COORDINATES:
        return {"coordinate_status": "declared",
                "coordinate_contract_id": FORMAL_COORDINATE_CONTRACT_ID,
                "coordinate_frame": "map_enu",
                "coordinate_axes": ["east", "north", "up"],
                "coordinate_basis": FORMAL_COORDINATE_BASIS}
    return {"coordinate_status": "not_declared", "coordinate_contract_id": None,
            "coordinate_frame": None, "coordinate_axes": None,
            "coordinate_basis": None}


def unit_policy(source_family: str, field_path: str,
                value_types: Iterable[str]) -> dict[str, Any]:
    """Return a source unit policy; unregistered numbers cannot be converted."""
    observed = set(value_types)
    if field_path == "tick" and source_family in DIRECT_TICK_FAMILIES and observed != {"int"}:
        raise ValueError(f"direct record tick must be an integer: {source_family}:{field_path}: {sorted(observed)}")
    if not observed & {"int", "float"}:
        return {"unit": None, "source_unit": None, "canonical_unit": None,
                "unit_status": "not_applicable", "unit_basis": None,
                "quantity_role": None, "time_role": None, "coordinate_frame": None}
    rule = UNIT_RULES.get((source_family, field_path))
    if rule is None:
        rule = _world_observation_unit_rule(source_family, field_path)
    if rule is None and field_path == "tick" and source_family in DIRECT_TICK_FAMILIES:
        rule = ("simulation_tick", "Dataset/world_model/graph/contract.py:label_temporal_graph.time_unit",
                "time_point", None)
    if rule is None:
        return {"unit": None, "source_unit": None, "canonical_unit": None,
                "unit_status": "unresolved", "unit_basis": None,
                "quantity_role": None, "time_role": None, "coordinate_frame": None}
    source_unit, basis, quantity_role, frame = rule
    return {"unit": source_unit, "source_unit": source_unit,
            "canonical_unit": UNIT_DEFINITIONS[source_unit]["canonical_unit"],
            "unit_status": "declared", "unit_basis": basis,
            "quantity_role": quantity_role,
            "time_role": quantity_role if quantity_role in {"time_point", "duration"} else None,
            "coordinate_frame": frame}


def canonical_numeric(value: int | float, mapping: dict[str, Any],
                      *, sample_clock: dict[str, Any] | None = None,
                      coordinate_context: Mapping[str, Any] | None = None) -> tuple[float | None, str | None]:
    """Convert a value only when the source and canonical units are proven."""
    if type(value) not in (int, float):
        raise TypeError(f"numeric conversion received {type(value).__name__}")
    if mapping["unit_status"] != "declared":
        return None, "source unit has no declaration"
    source, target = mapping["source_unit"], mapping["canonical_unit"]
    if source not in UNIT_DEFINITIONS or target not in UNIT_DEFINITIONS:
        raise ValueError(f"unit absent from registry: {source}->{target}")
    if mapping.get("coordinate_frame") is not None:
        if coordinate_context is None:
            return None, "source coordinate context is missing"
        if (coordinate_context.get("coordinate_frame") != mapping["coordinate_frame"] or
                coordinate_context.get("coordinate_contract_id") != mapping.get("coordinate_contract_id")):
            raise ValueError("source coordinate context differs from the field declaration")
    if source == "simulation_tick" and sample_clock is None:
        return None, "sample tick clock is missing"
    if source == target:
        return float(value), None
    if source == "simulation_tick" and target == "s":
        if sample_clock is None or type(sample_clock.get("tick_hz")) not in (int, float) or sample_clock["tick_hz"] <= 0:
            return None, "sample tick_hz is missing"
        if mapping.get("time_role") == "time_point":
            if type(sample_clock.get("origin_s")) not in (int, float):
                return None, "sample clock origin_s is missing"
            return sample_clock["origin_s"] + value / sample_clock["tick_hz"], None
        if mapping.get("time_role") == "duration":
            return value / sample_clock["tick_hz"], None
        return None, "tick time role is unresolved"
    raise ValueError(f"no declared conversion: {source}->{target}")


@lru_cache(maxsize=8)
def sample_tick_clock(episode_id: str) -> dict[str, Any]:
    """Read the actual episode clock declared by its first render-ready frame."""
    path = REPO / "aw_data/render_ready_episodes_capture_filtered" / episode_id / "truth_frames.jsonl"
    with path.open(encoding="utf-8") as stream:
        first = json.loads(next(stream))
    if first.get("episode_id") != episode_id:
        raise ValueError(f"source frame episode differs from its path: {path}:1")
    hz = validate_formal_frame_clock(first, expected_tick=0, source=f"{path}:1")
    tick, step, sim_time = (first[name] for name in ("tick", "dt_s", "sim_time_s"))
    return {"tick_hz": hz, "dt_s": step, "origin_s": sim_time - tick / hz,
            "source_file": path.relative_to(REPO).as_posix(), "source_line": 1}


@lru_cache(maxsize=210)
def formal_coordinate_context(episode_id: str) -> dict[str, Any]:
    """Resolve the native map frame from the current formal episode and map package."""
    episode = REPO / "aw_data/render_ready_episodes_capture_filtered" / episode_id
    manifest_path = episode / "episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    map_id = manifest.get("map_id")
    if not isinstance(map_id, str) or not map_id:
        raise ValueError(f"formal map coordinate declaration differs: {episode_id}")
    scene_path = REPO / manifest["source_scene_setup_path"]
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_map = scene.get("map_ref")
    if (not isinstance(scene_map, Mapping) or scene_map.get("map_id") != map_id or
            scene_map.get("coordinate_frame") != "ENU"):
        raise ValueError(f"formal source scene map frame differs: {episode_id}")
    package_path = REPO / "Config/LowAltitude/Maps" / map_id / "map_package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    context_path = REPO / package["map_context"]
    context = json.loads(context_path.read_text(encoding="utf-8"))
    origin = context.get("world_origin_cm")
    if (package.get("map_id") != map_id or context.get("map_id") != map_id or
            context.get("local_frame") != "ENU" or
            scene_map.get("geo_reference") != context.get("geo_reference") or
            context.get("world_origin_policy") != "fixed" or
            not isinstance(origin, list) or len(origin) != 3 or
            any(type(component) not in (int, float) or not math.isfinite(component)
                for component in origin)):
        raise ValueError(f"map ENU origin or axes are undeclared: {map_id}")
    return {
        "coordinate_contract_id": FORMAL_COORDINATE_CONTRACT_ID,
        "coordinate_frame": "map_enu", "map_id": map_id,
        "axis_order": ["east", "north", "up"],
        "world_origin_cm": list(origin),
        "height_reference": "map_local_up",
        "source_files": [p.relative_to(REPO).as_posix() for p in
                         (manifest_path, scene_path, package_path, context_path)],
    }


def coordinate_context_for_value(source_family: str, field_path: str,
                                 source_file: str, raw_record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Check a pose's own contract ID before admitting its native map coordinates."""
    if (source_family, field_path) not in FORMAL_POSE_COORDINATES:
        return None
    path = Path(source_file)
    if (len(path.parts) != 4 or path.parts[0:2] !=
            ("aw_data", "render_ready_episodes_capture_filtered") or
            path.name != "truth_frames.jsonl"):
        raise ValueError(f"formal coordinate source path is invalid: {source_file}")
    pose = raw_record.get("truth_pose")
    if not isinstance(pose, Mapping) or pose.get("coordinate_contract_id") != FORMAL_COORDINATE_CONTRACT_ID:
        raise ValueError(f"truth pose lacks {FORMAL_COORDINATE_CONTRACT_ID}: {source_file}")
    for name in ("position_enu_m", "velocity_enu_mps"):
        vector = pose.get(name)
        if (not isinstance(vector, list) or len(vector) != 3 or
                any(type(component) not in (int, float) or not math.isfinite(component)
                    for component in vector)):
            raise ValueError(f"truth pose has invalid {name}: {source_file}")
    return formal_coordinate_context(path.parts[2])


_CURRENT_DOMAIN_QUANTITIES = {
    "accel_mps2": ("m/s^2", "Observed longitudinal acceleration from the previous vehicle speed and elapsed sample time"),
    "altitude_deviation_m": ("m", "Absolute difference between observed aircraft z and its assigned altitude"),
    "altitude_error_m": ("m", "Observed aircraft altitude error used by the control response computer"),
    "ambulance_distance_m": ("m", "Current planar ambulance distance to the observed hazard center"),
    "ambulance_previous_distance_m": ("m", "Previous sampled planar ambulance distance to the hazard center"),
    "ambulance_perimeter_tolerance_m": ("m", "Allowed planar ambulance offset from the hazard perimeter for arrival"),
    "arrival_dwell_ticks": ("simulation_tick", "Accumulated sampled arrival-condition dwell on the episode clock"),
    "arrival_required_dwell_ticks": ("simulation_tick", "Required arrival-condition dwell before responder arrival"),
    "assigned_altitude_m": ("m", "Aircraft altitude assigned by its roster or corridor motion contract"),
    "assigned_landing_zone_pose_enu_m[]": ("m", "Component of the assigned landing target position, selected by array axis"),
    "associated_patient_distance_m": ("m", "Planar distance between the observed isolation-control center and associated fallen pedestrian"),
    "boundary_margin_m": ("m", "Governed restricted-airspace boundary comparison margin"),
    "boundary_polygon_enu_m[][]": ("m", "Horizontal coordinate of a restricted-region polygon vertex; axes are vertex then x/y"),
    "corridor_capacity": ("count", "Integer aircraft capacity computed from independent corridor cross-section lanes"),
    "corridor_center_enu_m[]": ("m", "Component of the airspace corridor center"),
    "corridor_cross_section_size_m[]": ("m", "Component of the airspace corridor cross-section size"),
    "corridor_extent_m[]": ("m", "Component of the airspace corridor extent"),
    "corridor_occupancy_count": ("count", "Number of sampled aircraft occupying this corridor at the tick"),
    "corridor_yaw_deg": ("deg", "Corridor horizontal orientation used by the geometry computer"),
    "deployed_controller_count": ("count", "Number of observed deployed physical isolation controllers"),
    "distance_m": ("m", "Three-dimensional Euclidean separation of the two explicitly bound pair participants"),
    "distance_to_boundary_m": ("m", "Unsigned geometric distance to the restricted-region boundary"),
    "distance_to_protected_airspace_m": ("m", "Observed aircraft distance to the protected-airspace boundary"),
    "eta_window_ticks": ("simulation_tick", "Producer pad-approach comparison window on the episode clock"),
    "geometry.base_z_m": ("m", "Base z of the polygon prism constructed around sampled isolation controls"),
    "geometry.height_m": ("m", "Governed height of the isolation-control polygon prism"),
    "geometry.polygon_enu_m[][]": ("m", "Horizontal coordinate of a constructed isolation-prism vertex; axes are vertex then x/y"),
    "gust_window_ticks": ("simulation_tick", "Backward sampled wind-history window used to detect a gust"),
    "handoff_required_dwell_ticks": ("simulation_tick", "Required responder handoff dwell for the hazmat outcome"),
    "hazard_center_enu_m[]": ("m", "Component of the hazard center observed in a dynamic pose or lawful static placement"),
    "hazard_concentration_ppm": ("ppm", "Observed weather hazard concentration in parts per million"),
    "hazard_radius_m": ("m", "Observed hazard radius used by physical safety and response conditions"),
    "home_distance_m": ("m", "Current aircraft distance to its home target used by the control response computer"),
    "previous_home_distance_m": ("m", "Previous sampled aircraft distance to its home target"),
    "home_pad_pose_enu_m[]": ("m", "Component of the home landing-pad position"),
    "horizontal_distance_to_boundary_m": ("m", "Horizontal aircraft distance to the restricted-region polygon boundary"),
    "illumination_lux": ("lux", "Observed illumination copied from the weather frame"),
    "local_ground_reference_z_m": ("m", "Local ground-reference z used to compute aircraft height above ground"),
    "lockdown_radius_m": ("m", "Governed horizontal buffer radius around observed isolation controls"),
    "minimum_center_separation_m": ("m", "Aircraft center-separation margin used to compute corridor capacity"),
    "minimum_restricted_boundary_distance_m": ("m", "Minimum signed aircraft boundary distance over active restricted regions; interior points are negative and the aggregate is zero when none are active"),
    "nearest_aircraft_distance_m": ("m", "Nearest sampled aircraft separation; governed anisotropic equivalent when a pair model exists"),
    "nearest_building_distance_m": ("m", "Minimum three-dimensional distance to an available building position"),
    "nearest_ground_vehicle_distance_m": ("m", "Minimum three-dimensional distance to a sampled ground vehicle"),
    "nearest_pedestrian_distance_m": ("m", "Minimum three-dimensional distance to a sampled pedestrian"),
    "nearest_population_distance_m": ("m", "Minimum three-dimensional distance to a sampled ground vehicle or pedestrian"),
    "nearest_vehicle_distance_m": ("m", "Minimum three-dimensional distance to a sampled vehicle"),
    "pair_distance_m": ("m", "Aircraft-pair comparison distance; metric field distinguishes Euclidean and governed anisotropic equivalent"),
    "pair_euclidean_distance_m": ("m", "Aircraft-pair three-dimensional Euclidean separation"),
    "pair_horizontal_distance_m": ("m", "Aircraft-pair horizontal Euclidean separation"),
    "pair_horizontal_limit_m": ("m", "Declared horizontal component limit of the aircraft-pair separation model"),
    "pair_vertical_distance_m": ("m", "Absolute aircraft-pair vertical separation"),
    "pair_vertical_limit_m": ("m", "Declared vertical component limit of the aircraft-pair separation model"),
    "patient_association_max_distance_m": ("m", "Maximum physical distance allowed when associating a fallen pedestrian with isolation controls"),
    "previous_route_distance_m": ("m", "Previous sampled distance from the aircraft to its assigned route"),
    "rain_rate": ("ratio", "Normalized rain-intensity value copied from the formal weather carrier; it is not precipitation per hour"),
    "request_count": ("count", "Explicit pad request count, or count of its explicit requester identities"),
    "risk_distance_m": ("m", "Planar distance between a pedestrian and its nearest sampled risk vehicle"),
    "route_distance_m": ("m", "Current distance from the aircraft to its assigned route"),
    "safe_radius_margin_m": ("m", "Additional safe distance beyond the observed hazard radius"),
    "safe_required_dwell_ticks": ("simulation_tick", "Required pedestrian safe-distance dwell for hazmat resolution"),
    "signed_distance_to_boundary_m": ("m", "Signed aircraft distance to the restricted-region boundary"),
    "speed_delta_mps": ("m/s", "Change in sampled aircraft speed used by the slowdown response detector"),
    "surface_wetness": ("ratio", "Normalized road wetness copied from the formal weather carrier"),
    "takeoff_ground_z_m": ("m", "Ground-reference z at the observed aircraft takeoff origin"),
    "target_pedestrian_distances_m.{entity}": ("m", "Current planar distance from the identified target pedestrian to the hazard center"),
    "target_pedestrian_safe_dwell_ticks.{entity}": ("simulation_tick", "Accumulated safe-distance dwell of the identified target pedestrian"),
    "visibility_m": ("m", "Observed visible range copied from the formal weather frame"),
    "wind_direction_deg": ("deg", "Observed weather wind direction in degrees"),
    "xy_distance_to_assigned_landing_zone_m": ("m", "Horizontal aircraft distance to the assigned landing target"),
    "xy_distance_to_home_pad_m": ("m", "Horizontal aircraft distance to its home landing pad"),
    "z_agl_m": ("m", "Observed aircraft z minus its current local ground-reference z"),
}
_CURRENT_DOMAIN_BASIS = (
    "Dataset/semantic_simulation/control_response_state.py + "
    "Dataset/semantic_simulation/predicate_state_computers.py + "
    "Dataset/semantic_simulation/observable_state_completion.py; "
    "Dataset/semantic_truth/objective_pipeline.py current formal physical inputs"
)
for _path, (_unit, _meaning) in _CURRENT_DOMAIN_QUANTITIES.items():
    _full_path = "values." + _path
    _declare_quantity("domain_state", _full_path, _unit, _CURRENT_DOMAIN_BASIS,
                      "duration" if _unit == "simulation_tick" else "source_quantity", _meaning)
EXACT_BRANCH_MEANINGS[("domain_state", "observable_pad_facility", "values.fault")] = (
    "Exact source fault scalar retained as boolean, number or string by _exact_state_scalar; no physical unit is established for a numeric code"
)


def classify_catalog(rows: list[dict[str, Any]], version: str) -> list[dict[str, Any]]:
    """Attach one definition reference and unit policy to every source field."""
    seen: set[str] = set()
    matched_rules: set[tuple[str, str]] = set()
    for row in rows:
        row["projection_owners"] = list(projection_owners(row))
        source_id = row["id"]
        if source_id in seen:
            raise ValueError(f"duplicate source field: {source_id}")
        seen.add(source_id)
        path = row["field_path"]
        branch = catalog_source_branch(row)
        key = (row["source_family"], path)
        observation_meaning = _world_observation_meaning(row["source_family"], path)
        declared_types = EXACT_ALLOWED_VALUE_TYPES.get(key)
        if declared_types is not None:
            observed_types = row.get("observed_value_types", row["value_types"])
            if not set(observed_types) <= set(declared_types):
                raise ValueError(f"source value differs from declared types: {source_id}: {observed_types}")
            row["observed_value_types"] = observed_types
            row["value_types"] = declared_types
            row["value_kind"] = "mixed"
        if key in UNIT_RULES:
            matched_rules.add(key)
        row.update(unit_policy(row["source_family"], path, row["value_types"]))
        if key in FORMAL_POSE_COORDINATES:
            expected_unit, expected_role = FORMAL_POSE_COORDINATES[key]
            if (row["source_unit"] != expected_unit or row["quantity_role"] != expected_role or
                    row["coordinate_frame"] != "map_enu"):
                raise ValueError(f"formal coordinate rule differs from unit rule: {source_id}")
        elif row["coordinate_frame"] is not None:
            raise ValueError(f"unit rule declares a coordinate without a source binding: {source_id}")
        row.update(coordinate_policy(row["source_family"], path))
        row["source_time_role"] = source_time_role(row["source_family"], path)
        row["semantic_id"] = (SEMANTIC_ALIASES[key] if key in SEMANTIC_ALIASES else
                              "metadata:" + path if path in SHARED_METADATA
                              and row["value_types"] == ["string"] else
                              RECORD_TICK_SEMANTIC_ID if path == "tick"
                              and row["source_family"] in DIRECT_TICK_FAMILIES
                              and row["value_types"] == ["int"] else "source:" + source_id)
        row["definition_status"] = ("shared_metadata" if row["semantic_id"].startswith("metadata:")
                                    else "shared_clock" if row["semantic_id"] == RECORD_TICK_SEMANTIC_ID
                                    else "unit_evidence_gap" if row["unit_status"] == "unresolved"
                                    else "source_specific")
        row["meaning_status"] = ("declared" if row["semantic_id"] == RECORD_TICK_SEMANTIC_ID
                                 or path in SHARED_METADATA and row["value_types"] == ["string"]
                                 or key in EXACT_MEANINGS
                                 or (row["source_family"], branch, path) in EXACT_BRANCH_MEANINGS
                                 or observation_meaning is not None
                                 else "unresolved")
        row["array_rank"] = path.count("[]")
        if key in EXACT_MISSING_RULES:
            row["missing_rule"] = {"declaration_status": "source_declared",
                                   **EXACT_MISSING_RULES[key]}
        row["contract_version"] = version
    if matched_rules != set(UNIT_RULES):
        raise ValueError(f"unit mappings absent from field catalog: {sorted(set(UNIT_RULES) - matched_rules)}")
    return rows


def projection_owners(row: dict[str, Any]) -> tuple[str, ...]:
    """An ARM outcome's node type follows its joined occurrence, not its keys."""
    if row["source_family"] == "arm_event_outcomes":
        return "EventOutcome", "PredicateEpisodeOutcome"
    return (row["owner"],)


def _source_meaning(source: dict[str, Any]) -> str | None:
    semantic_id = source["semantic_id"]
    if semantic_id == RECORD_TICK_SEMANTIC_ID:
        return "Tick coordinate on the sample simulation clock; source fields retain their record roles"
    if semantic_id.startswith("metadata:"):
        return SHARED_METADATA[source["field_path"]]
    return _world_observation_meaning(source["source_family"], source["field_path"]) or \
        EXACT_BRANCH_MEANINGS.get((source["source_family"],
                                      catalog_source_branch(source), source["field_path"])) or \
        EXACT_MEANINGS.get((source["source_family"], source["field_path"]))


def build_definitions(rows: list[dict[str, Any]], version: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    indices: dict[str, int] = {}
    for row in rows:
        if row.get("contract_version") != version or not row.get("semantic_id"):
            raise ValueError(f"field lacks current semantic mapping: {row['id']}")
        expected_unit = unit_policy(row["source_family"], row["field_path"], row["value_types"])
        if any(row.get(key) != value for key, value in expected_unit.items()):
            raise ValueError(f"source unit mapping differs from registry: {row['id']}")
        semantic_id, index = row["semantic_id"], row.get("semantic_index")
        if type(index) is not int or index < 0:
            raise ValueError(f"field lacks a declared semantic index: {row['id']}")
        if semantic_id in indices and indices[semantic_id] != index:
            raise ValueError(f"semantic field has conflicting indices: {semantic_id}")
        indices[semantic_id] = index
        groups[semantic_id].append(row)
    if (len(set(indices.values())) != len(indices) or
            set(indices.values()) != set(range(len(indices)))):
        raise ValueError("semantic field indices are not unique and contiguous")
    definitions = []
    for semantic_id in sorted(groups, key=indices.__getitem__):
        index, sources = indices[semantic_id], groups[semantic_id]
        types = {tuple(source["value_types"]) for source in sources}
        shapes = {(source["array_rank"], source.get("shape_length"),
                   source.get("shape_status") or "not_declared",
                   tuple(tuple(axis) for axis in source.get("observed_axis_lengths", ())))
                  for source in sources}
        units = {(source["canonical_unit"], source.get("quantity_role"), source.get("time_role"),
                  source.get("coordinate_frame"), source.get("coordinate_status"),
                  source.get("coordinate_contract_id"),
                  tuple(source.get("coordinate_axes") or ()))
                 for source in sources}
        meaning_statuses = {source["meaning_status"] for source in sources}
        meanings = {_source_meaning(source) for source in sources}
        missing_rules = {json.dumps(source.get("missing_rule"), sort_keys=True) for source in sources}
        if any(len(values) != 1 for values in (types, shapes, units, meaning_statuses, meanings, missing_rules)):
            raise ValueError(f"shared definition has incompatible type, shape, unit or meaning: {semantic_id}")
        meaning = next(iter(meanings))
        if (meaning is None) != (sources[0]["meaning_status"] == "unresolved"):
            raise ValueError(f"field meaning status disagrees with declaration: {semantic_id}")
        array_rank, shape_length, shape_status, observed_axes = next(iter(shapes))
        canonical_unit, _, _, _, _, _, _ = next(iter(units))
        value_types = sorted({item for source in sources for item in source["value_types"]})
        definition = {
            "kind": "field_definition", "id": semantic_id, "index": index,
            "contract_version": version,
            "definition_status": sources[0]["definition_status"],
            "meaning": meaning,
            "meaning_status": sources[0]["meaning_status"],
            "value_types": value_types, "shape": {"array_rank": array_rank,
                "fixed_length": shape_length, "length_status": shape_status,
                "observed_axis_lengths": [list(axis) for axis in observed_axes] if observed_axes else None},
            "canonical_unit": canonical_unit, "time_role": sources[0].get("time_role"),
            "quantity_role": sources[0].get("quantity_role"),
            "coordinate_frame": sources[0].get("coordinate_frame"),
            "coordinate_status": sources[0].get("coordinate_status"),
            "coordinate_contract_id": sources[0].get("coordinate_contract_id"),
            "coordinate_axes": sources[0].get("coordinate_axes"),
            "missing_rule": sources[0].get("missing_rule") or {
                "null_observed": "null" in value_types,
                "empty_list_observed": "empty_list" in value_types,
                "empty_object_observed": "empty_object" in value_types,
                "declaration_status": "observed_types_only"},
            "mixed_value_policy": ({"numeric": "convert_only_int_or_float_with_declared_unit",
                                    "other": "preserve_source_value_and_status_without_numeric_substitution"}
                                   if {"int", "float"} & set(value_types) and
                                   set(value_types) - {"int", "float", "null"} else None),
            "source_fields": sorted(source["id"] for source in sources),
            "source_roles": sorted({source["source_family"] for source in sources}),
            "owners": sorted({owner for source in sources for owner in projection_owners(source)}),
        }
        definitions.append(definition)
    return definitions


def write_definitions(path: Path, rows: list[dict[str, Any]], version: str) -> dict[str, int]:
    definitions = build_definitions(rows, version)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in definitions:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return {"source_fields": len(rows), "definitions": len(definitions),
            "shared_groups": sum(len(row["source_fields"]) > 1 for row in definitions),
            "unit_evidence_gaps": sum(row["unit_status"] == "unresolved" for row in rows)}
