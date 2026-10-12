"""Closed producer schema for the formal objective manifest JSON root.

Source family ``formal_objective_manifest``; the L1 inventory contains the
seven declared profiles L1-1_v1, L1-1_v2, L1-2_v1, L1-3_v1, L1-3_v2,
L1-4_v1, and L1-4_v2 (three seeds each).  ``structure_sources.py`` does not
place this family in ``_ENTITY_FILES`` and reads a ``.json`` source as one
record, so this schema describes the complete ``manifest.json`` root.

No objective manifest currently exists under ``aw_data/objective_semantic_truth``.
The schema is therefore compiled from the current writers and has no instance
validation yet.  It does not generate a missing objective artifact.

Producer authority, checked against the current source:

* ``objective_pipeline._build_manifest`` (:2295-2522) builds the base root,
  including the two trajectory-union branches.
* ``build_episode_objective_artifacts`` (:526-610) then adds all eight
  simulation-log declarations, simulation-log reconciliation, the L0 source
  contract artifact, event-family mapping, and both materialization summaries;
  it also extends ``record_counts`` and the L0 artifact list.
* The copied summaries retain their own producer shapes:
  ``input_adapter.build_tick_contexts`` (:282-300),
  ``minimal_semantics_adapter`` (:220-239), ``world_truth`` (:1167-1196),
  ``utm_state`` (:292-305), ``simulation_logs`` (:275-304),
  ``l0_supplement`` (:378-442), and ``PlanWindowComputer`` (:976-993).
  The world-truth registry-owned subtrees reuse the already closed
  ``world_truth_base_schema`` fragments rather than copying registry values.

The writer has no profile-level structure switch, so the returned schema does
not pin ``episode_id``, ``epi_id``, or runtime schedule identifiers to L1
values.  Cross-level reuse remains valid only while those producer paths and
the executable world-truth registry remain shared.

Raw integrity metadata remains in this grammar and is checked only by type and
format.  ``_PUBLICATION_INTEGRITY_PATHS_REQUIRED`` records the exact leaf paths
of raw integrity metadata the publication projection leaves out.  All 52
now intersect ``structure_schema._PUBLICATION_EXCLUDED_PATHS`` (DSH G4 PASS);
this module neither edits that global policy nor removes a raw manifest field.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types,
    get_core_predicate_ids,
)
from Dataset.world_model.graph.world_truth_structure import world_truth_base_schema
from Dataset.world_model.graph.published_structure import published_record_schema

__all__ = ["formal_objective_manifest_schema"]


_SCHEMA = "https://json-schema.org/draft/2020-12/schema"
_DIGEST_PATTERN = "^sha256:[0-9a-f]{64}$"
_SIMULATION_LOG_NAMES = (
    "compute_log.jsonl",
    "communication_log.jsonl",
    "charging_log.jsonl",
    "battery_log.jsonl",
    "utm_log.jsonl",
    "gnss_log.jsonl",
    "weather_log.jsonl",
    "facility_log.jsonl",
)
_SORTED_SIMULATION_LOG_NAMES = tuple(sorted(_SIMULATION_LOG_NAMES))
_GLOBAL_DETECTION_NAMES = (
    "semantic_state_observations",
    "predicate_truth",
    "predicate_transitions",
    "predicate_continuity_breaks",
    "event_occurrences",
    "event_outcomes",
)
_INPUT_FILE_NAMES = (
    "episode_manifest.json",
    "global_entity_roster.json",
    "trajectories.jsonl",
    "truth_frames.jsonl",
    "weather_meta.jsonl",
)
_UTM_RECORD_NAMES = (
    "airspace_authorization_log.jsonl",
    "operational_intent_log.jsonl",
    "operational_intent_pair_log.jsonl",
    "flight_plan_log.jsonl",
    "deconfliction_log.jsonl",
)
_L0_CONTRACT_NAME = "l0_predicate_source_availability.json"


# These paths describe integrity leaves in the raw producer record.  They are
# deliberately not consumed here: publication projection is owned by
# structure_schema.py and integration is outside this task.
_PUBLICATION_INTEGRITY_PATHS_REQUIRED: tuple[tuple[str, ...], ...] = (
    *(("input_files", name) for name in _INPUT_FILE_NAMES),
    ("domain_profile", "sha256"),
    ("compute_profile", "sha256"),
    ("contract_profile", "sha256"),
    ("stage_acceptance_profile", "sha256"),
    ("domain_input_digest",),
    ("context_input_digest",),
    ("static_geometry_authority", "resolved_source_sha256"),
    ("projection", "input_digest"),
    ("projection", "parameter_digest"),
    *(("global_detection_record_digests", name)
      for name in _GLOBAL_DETECTION_NAMES),
    ("world_truth_summary", "input_digest"),
    ("world_truth_summary", "parameter_digest"),
    ("utm_supplement_summary", "input_digest"),
    ("utm_supplement_summary", "parameter_digest"),
    ("trajectory_union", "world_source", "sha256"),
    ("trajectory_union", "scenario_source", "sha256"),
    *(("simulation_logs", name, "sha256")
      for name in _SIMULATION_LOG_NAMES),
    *((
        "simulation_log_reconciliation",
        "source_digest_multiset_by_log",
        name,
        "multiset_digest",
    ) for name in _SIMULATION_LOG_NAMES),
    ("l0_contract_artifacts", _L0_CONTRACT_NAME, "sha256"),
    ("event_family_mapping", "sha256"),
    ("l0_state_materialization", "sumo_roster_never_active_digest"),
    ("l0_state_materialization", "render_vehicle_authority_conflict_digest"),
    ("l0_state_materialization", "l0_state_profile_sha256"),
    ("l0_state_materialization", "sumo_authority_digest"),
    ("l0_state_materialization", "state_digest"),
    ("runtime_state_materialization", "runtime_schedule_sha256"),
    ("runtime_state_materialization", "l0_state_profile_sha256"),
    ("runtime_state_materialization", "event_fire_tick_digest"),
)


def _string() -> dict[str, Any]:
    return {"type": "string"}


def _nonblank_string() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _digest() -> dict[str, Any]:
    return {"type": "string", "pattern": _DIGEST_PATTERN}


def _integer(
    minimum: int | None = 0,
    maximum: int | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer"}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _object(
    properties: Mapping[str, Any],
    *,
    required: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties) if required is None else list(required),
        "additionalProperties": False,
    }


def _array(
    items: Mapping[str, Any],
    *,
    min_items: int | None = None,
    max_items: int | None = None,
    unique: bool = False,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": dict(items)}
    if min_items is not None:
        schema["minItems"] = min_items
    if max_items is not None:
        schema["maxItems"] = max_items
    if unique:
        schema["uniqueItems"] = True
    return schema


def _fixed_string_array(values: Sequence[str]) -> dict[str, Any]:
    values = tuple(values)
    schema: dict[str, Any] = {
        "type": "array",
        "items": False,
        "minItems": len(values),
        "maxItems": len(values),
    }
    if values:
        schema["prefixItems"] = [{"const": value} for value in values]
    return schema


def _fixed_integer_map(
    names: Sequence[str],
    *,
    minimum: int = 0,
) -> dict[str, Any]:
    return _object({name: _integer(minimum) for name in names})


def _enum_key_map(
    names: Sequence[str],
    value_schema: Mapping[str, Any],
    *,
    exact: bool,
) -> dict[str, Any]:
    names = tuple(names)
    schema: dict[str, Any] = {
        "type": "object",
        "propertyNames": {"enum": list(names)},
        "additionalProperties": dict(value_schema),
        "maxProperties": len(names),
    }
    if exact:
        schema["minProperties"] = len(names)
    return schema


def _profile_reference() -> dict[str, Any]:
    return _object({"path": _nonblank_string(), "sha256": _digest()})


def _static_geometry_authority() -> dict[str, Any]:
    return {
        "oneOf": [
            _object({
                "used": {"const": False},
                "source_type": {"const": "explicit_static_geometry"},
                "source_scene_setup_path": {"type": "null"},
                "resolved_source_sha256": {"type": "null"},
            }),
            _object({
                "used": {"const": True},
                "source_type": {
                    "const": "scenario_authority_scene_setup_polygon_prism"
                },
                "source_scene_setup_path": _nonblank_string(),
                "resolved_source_sha256": _digest(),
            }),
        ]
    }


def _projection(core_predicate_ids: Sequence[str], event_type_count: int) -> dict[str, Any]:
    core_predicate_ids = tuple(core_predicate_ids)
    return _object({
        "catalog_id": {"const": "aeroworld.ontology_predicates.v3"},
        "catalog_version": {"const": "3.0.0"},
        "projection_mode": {
            "const": "global_detection_with_epi_view_acceptance"
        },
        "event_detection_scope": {"const": "global_contract_event_families"},
        "durable_event_view": {"const": "full_detection_rows"},
        "durable_output_policy": {"const": "full_detection_rows_v1"},
        "global_detection_persistence": {
            "const": "durable_rows_with_manifest_counts_and_sources"
        },
        "selected_api_ids": _array(
            {"enum": list(core_predicate_ids)},
            max_items=len(core_predicate_ids),
            unique=True,
        ),
        "selected_api_count": _integer(0, len(core_predicate_ids)),
        "global_projected_api_ids": _fixed_string_array(core_predicate_ids),
        "global_projected_api_count": {"const": len(core_predicate_ids)},
        "catalog_gate": _object({
            "passed": {"const": True},
            "template_count": {"const": len(core_predicate_ids)},
            "event_occurrence_type_count": {"const": event_type_count},
        }),
        "input_digest": _digest(),
        "parameter_digest": _digest(),
    })


def _annotation_layers(predicate_count: int) -> dict[str, Any]:
    l0_artifacts = (
        "truth_frames.jsonl",
        "trajectories.jsonl",
        "weather_meta.jsonl",
        "l0_predicate_state.jsonl",
        "compute_comm_supplement",
        "domain_state_supplement",
        "utm_supplement",
        *(f"simulation_logs/{name}" for name in _SORTED_SIMULATION_LOG_NAMES),
        _L0_CONTRACT_NAME,
    )
    return _object({
        "L0": _object({
            "role": {
                "const": "raw_and_deterministic_supplement_truth_all_entities_all_ticks"
            },
            "artifacts": _fixed_string_array(l0_artifacts),
            "materialization": {
                "const": "authoritative_input_plus_deterministic_l0_builders"
            },
        }),
        "L1": _object({
            "role": {"const": "complete_grounded_world_truth_predicate_candidates"},
            "artifacts": _fixed_string_array((
                "world_truth_graph_base.json",
                "world_truth_graph_deltas.jsonl",
                "world_truth_summary.json",
            )),
            # world_truth_base currently has no ``scope_types`` member, so
            # ``world_truth_base.get('scope_types', ())`` writes an empty list.
            "scope_types": _fixed_string_array(()),
            "predicate_count": {"const": predicate_count},
            "truth_values": _fixed_string_array(
                ("true", "false", "unknown", "out_of_scope")
            ),
            "materialization": {
                "const": "tick0_complete_matrix_base_plus_value_change_delta"
            },
            "full_state_snapshot_per_change": {"const": False},
            "replayable_at_arbitrary_tick": {"const": True},
        }),
        "L2": _object({
            "role": {"const": "semantic_event_graph_roi_or_event_entities"},
            "artifacts": _fixed_string_array((
                "predicate_truth.jsonl",
                "predicate_transitions.jsonl",
                "predicate_continuity_breaks.jsonl",
                "semantic_graph_base.json",
                "semantic_graph_deltas.jsonl",
                "semantic_truth_graphs.jsonl",
            )),
            "representation": {
                "const": "tick0_base_plus_predicate_delta_plus_event_window_projection"
            },
            "full_state_snapshot_per_change": {"const": False},
            "replayable_at_arbitrary_tick": {"const": True},
        }),
    })


def _record_counts() -> dict[str, Any]:
    names = (
        "l0_predicate_state",
        "semantic_state_observations",
        "predicate_truth",
        "predicate_transitions",
        "predicate_continuity_breaks",
        "event_occurrences",
        "event_outcomes",
        "world_truth_graph_deltas",
        "world_truth_graph_delta_operations",
        "world_truth_initial_assertions",
        "semantic_graph_bases",
        "semantic_graph_deltas",
        "semantic_truth_graphs",
        "evidence_observations",
    )
    properties = {name: _integer() for name in names}
    properties["world_truth_graph_bases"] = {"const": 1}
    properties["simulation_log_records"] = _integer(1)
    return _object(properties)


def _intermediate_record_counts() -> dict[str, Any]:
    return _fixed_integer_map((
        "domain_state_observations",
        "objective_supplement_domain_state_observations",
        "compute_state",
        "communication_state",
        "compute_communication_predicate_truth",
        "utm_state",
    ))


def _truth_value_count_map() -> dict[str, Any]:
    return _object(
        {
            value: _integer()
            for value in ("true", "false", "unknown", "out_of_scope")
        },
        required=(),
    )


def _world_truth_summary(world_base: Mapping[str, Any]) -> dict[str, Any]:
    base_properties = world_base["properties"]
    base_summary = base_properties["summary"]["properties"]
    candidate_counts = deepcopy(
        base_summary["initial_candidate_counts_by_predicate"]
    )
    executable_ids = tuple(candidate_counts["propertyNames"]["enum"])
    operation_counts = _object(
        {
            name: _integer()
            for name in (
                "add_predicate_assertion",
                "remove_predicate_assertion",
                "set_predicate_value",
                "refresh_predicate_evidence",
            )
        },
        required=(),
    )
    return _object({
        "schema_name": {"const": "world_truth_evaluation_summary"},
        "schema_version": deepcopy(base_properties["schema_version"]),
        "episode_id": _nonblank_string(),
        "representation": {"const": "grounded_predicate_candidates"},
        "candidate_set_complete": {"const": True},
        "candidate_set_scope": {"const": "executable_l1_predicates_only"},
        "formal_tick_count": {"const": 181},
        "predicate_count": deepcopy(base_summary["predicate_count"]),
        "executable_predicate_count": deepcopy(
            base_summary["executable_predicate_count"]
        ),
        "non_executable_predicate_count": deepcopy(
            base_summary["non_executable_predicate_count"]
        ),
        "non_executable_predicates": deepcopy(
            base_properties["non_executable_predicates"]
        ),
        "evaluated_candidate_count": _integer(),
        "evaluated_candidate_counts_by_predicate": candidate_counts,
        "value_counts": _truth_value_count_map(),
        "value_counts_by_predicate": _enum_key_map(
            executable_ids,
            _truth_value_count_map(),
            exact=True,
        ),
        "base_assertion_count": _integer(),
        "delta_batch_count": _integer(),
        "delta_operation_count": _integer(),
        "delta_operation_counts": operation_counts,
        "input_digest": _digest(),
        "parameter_digest": _digest(),
        "governed_parameters": deepcopy(base_properties["governed_parameters"]),
    })


def _utm_supplement_summary() -> dict[str, Any]:
    return _object({
        "schema_name": {"const": "utm_supplement_summary"},
        "schema_version": {"const": "1.0.0"},
        "episode_id": _nonblank_string(),
        "scenario_id": _string(),
        "input_digest": _digest(),
        "parameter_digest": _digest(),
        "model_id": _string(),
        "model_version": _string(),
        "formal_tick_count": {"const": 181},
        "uav_count": _integer(1),
        "record_counts": _fixed_integer_map(_UTM_RECORD_NAMES),
        "source_policy": {"const": "scenario_contract_plus_entity_roster"},
    })


def _trajectory_union() -> dict[str, Any]:
    common = {
        "world_source": _object({
            "path": _nonblank_string(),
            "sha256": _digest(),
        }),
        "key": _fixed_string_array(("tick", "entity_id")),
        "overlap_authority": {"const": "world_source"},
    }
    return {
        "oneOf": [
            _object({
                **deepcopy(common),
                "scenario_source": _object({
                    "path": _nonblank_string(),
                    "sha256": _digest(),
                }),
                "scenario_source_role": {"const": "missing_offstage_rows_only"},
            }),
            _object({
                **deepcopy(common),
                "scenario_source": {"type": "null"},
                "scenario_source_role": {"const": "unavailable_explicit"},
                "missing_source_record": _object({
                    "path": {"oneOf": [_nonblank_string(), {"type": "null"}]},
                    "reason": {
                        "const": (
                            "formal scenario trajectory source is absent; "
                            "world trajectory remains authoritative"
                        )
                    },
                }),
            }),
        ]
    }


def _simulation_logs() -> dict[str, Any]:
    return _object({
        name: _object({
            "path": {"const": f"simulation_logs/{name}"},
            "record_count": _integer(),
            "sha256": _digest(),
        })
        for name in _SIMULATION_LOG_NAMES
    })


def _simulation_log_reconciliation() -> dict[str, Any]:
    per_log_digest = _object({
        "record_count": _integer(),
        "multiset_digest": _digest(),
    })
    return _object({
        "schema_name": {
            "const": "simulation_log_source_coverage_reconciliation"
        },
        "schema_version": {"const": "2.0.0"},
        "status": {"const": "PASS"},
        "record_count": _integer(1),
        "record_count_by_log": _fixed_integer_map(_SIMULATION_LOG_NAMES),
        "source_digest_multiset_by_log": _object({
            name: deepcopy(per_log_digest) for name in _SIMULATION_LOG_NAMES
        }),
        "log_count": {"const": len(_SIMULATION_LOG_NAMES)},
    })


def _l0_contract_artifacts() -> dict[str, Any]:
    return _object({
        _L0_CONTRACT_NAME: _object({
            "path": {"const": _L0_CONTRACT_NAME},
            "record_count": _integer(),
            "sha256": _digest(),
        })
    })


def _l0_state_materialization() -> dict[str, Any]:
    return _object({
        "schema_name": {"const": "l0_state_materialization_summary"},
        "schema_version": {"const": "1.0.0"},
        "episode_id": _nonblank_string(),
        "scenario_id": _string(),
        "formal_source_tick_count": {"const": 901},
        "formal_semantic_tick_count": {"const": 181},
        "active_vehicle_tick_count": _integer(),
        "active_uav_tick_count": _integer(),
        "trajectory_entity_tick_count": _integer(),
        "regulation_field_count": _integer(),
        "weather_temperature_count": _integer(),
        "source_unavailability_count": _integer(),
        "sumo_roster_vehicle_count": _integer(),
        "sumo_vehicle_ids_seen": _integer(),
        "sumo_roster_never_active_count": _integer(),
        "sumo_roster_never_active_digest": _digest(),
        "render_vehicle_tick_removed_count": _integer(),
        "render_vehicle_tick_pose_reconciled_count": _integer(),
        "render_vehicle_authority_conflict_count": _integer(),
        "render_vehicle_authority_conflict_digest": _digest(),
        "runtime_state_field_normalization_count": _integer(),
        "road_lane_count": _integer(),
        "road_traffic_light_controller_count": _integer(),
        "road_signalized_junction_count": _integer(),
        "road_crossing_count": _integer(),
        "l0_state_profile_id": {"const": "aeroworld_l0_state_supplement_v1"},
        "l0_state_profile_sha256": _digest(),
        "sumo_authority_digest": _digest(),
        "state_digest": _digest(),
    })


def _runtime_state_materialization() -> dict[str, Any]:
    schema = _object({
        "episode_id": _nonblank_string(),
        "runtime_schedule_profile_id": _string(),
        "runtime_schedule_source": _string(),
        "runtime_schedule_sha256": _digest(),
        "l0_state_profile_id": {"const": "aeroworld_l0_state_supplement_v1"},
        "l0_state_profile_sha256": _digest(),
        "runtime_event_count": _integer(),
        "resolved_runtime_event_count": _integer(),
        "unresolved_runtime_event_ids": _array(_string(), unique=True),
        "applied_runtime_action_count": _integer(),
        "materialized_entity_tick_count": _integer(),
        "event_fire_ticks": {
            "type": "object",
            "patternProperties": {
                "^.+$": _integer(0, 900),
            },
            "additionalProperties": False,
        },
        "event_fire_tick_digest": _digest(),
    })
    schema["properties"]["pending_outside_window"] = _array(_object({
        "event_id": _nonblank_string(), "entity_id": _nonblank_string(),
        "action_id": _nonblank_string(), "effective_tick": _integer(901),
        "window_end_tick": {"const": 900}, "source_script": _nonblank_string(),
        "status": {"const": "scheduled_beyond_observation_window"},
    }))
    return schema


def formal_objective_manifest_schema() -> dict[str, Any]:
    """Return the closed whole-record schema for an objective manifest.

    The constructor reads only the executable semantic registry through the
    existing world-truth schema factory.  It does not inspect episode data or
    objective output directories.
    """

    world_base = world_truth_base_schema()
    world_summary = world_base["properties"]["summary"]["properties"]
    predicate_count = int(world_summary["predicate_count"]["const"])
    core_predicate_ids = tuple(get_core_predicate_ids())
    event_type_count = len(get_core_event_occurrence_types())

    schema = {
        "$schema": _SCHEMA,
        "$id": (
            "https://aeroworld.local/schema/"
            "formal_objective_manifest_l1.record.schema.json"
        ),
        "title": "AERO_WORLD formal_objective_manifest L1 JSON root record schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_name",
            "schema_version",
            "episode_id",
            "epi_id",
            "seed_label",
            "episode_root",
            "offstage_source",
            "output_dir",
            "input_files",
            "domain_profile",
            "compute_profile",
            "contract_profile",
            "stage_acceptance_profile",
            "domain_input_digest",
            "context_input_digest",
            "static_geometry_authority",
            "projection",
            "annotation_layers",
            "record_counts",
            "global_detection_record_counts",
            "global_detection_record_digests",
            "durable_record_counts_match_global_detection",
            "intermediate_record_counts",
            "world_truth_summary",
            "utm_supplement_summary",
            "compact_graph_rejected_record_count",
            "closure_status",
            "forbidden_inputs_used",
            "trajectory_union",
            "simulation_logs",
            "simulation_log_reconciliation",
            "l0_contract_artifacts",
            "event_family_mapping",
            "l0_state_materialization",
            "runtime_state_materialization",
        ],
        "properties": {
            "schema_name": {"const": "objective_semantic_episode_manifest"},
            "schema_version": {"const": "4.0.0"},
            "episode_id": _nonblank_string(),
            "epi_id": _nonblank_string(),
            "seed_label": {"enum": ["seed00", "seed01", "seed02"]},
            "episode_root": _nonblank_string(),
            "output_dir": _nonblank_string(),
            "input_files": _object({
                name: _object({"path": _nonblank_string()}) for name in _INPUT_FILE_NAMES
            }),
            "offstage_source": _object({
                "path": {"oneOf": [_nonblank_string(), {"type": "null"}]},
                "status": {"enum": ["present", "source_missing", "not_declared"]},
                "role": {"const": "offstage_scenario_roster_and_trajectories"},
                "missing_policy": {"const": "retain_unknown_preflight_and_absent_pose_evidence"},
            }),
            "domain_profile": _profile_reference(),
            "compute_profile": _profile_reference(),
            "contract_profile": _profile_reference(),
            "stage_acceptance_profile": _profile_reference(),
            "domain_input_digest": _digest(),
            "context_input_digest": _digest(),
            "static_geometry_authority": _static_geometry_authority(),
            "projection": _projection(core_predicate_ids, event_type_count),
            "annotation_layers": _annotation_layers(predicate_count),
            "record_counts": _record_counts(),
            "global_detection_record_counts": _fixed_integer_map(
                _GLOBAL_DETECTION_NAMES
            ),
            "global_detection_record_digests": _object({
                name: _digest() for name in _GLOBAL_DETECTION_NAMES
            }),
            "durable_record_counts_match_global_detection": {"const": True},
            "intermediate_record_counts": _intermediate_record_counts(),
            "world_truth_summary": _world_truth_summary(world_base),
            "utm_supplement_summary": _utm_supplement_summary(),
            "compact_graph_rejected_record_count": _integer(),
            "closure_status": {"enum": ["PASS", "FAIL", "UNKNOWN"]},
            "forbidden_inputs_used": _array(_string(), min_items=0, max_items=0),
            "trajectory_union": _trajectory_union(),
            "simulation_logs": _simulation_logs(),
            "simulation_log_reconciliation": _simulation_log_reconciliation(),
            "l0_contract_artifacts": _l0_contract_artifacts(),
            "event_family_mapping": _object({
                "path": _nonblank_string(),
                "sha256": _digest(),
            }),
            "l0_state_materialization": _l0_state_materialization(),
            "runtime_state_materialization": _runtime_state_materialization(),
            "business_event_truth": _object({
                "producer": {"const": "Dataset/semantic_truth/compute_event_truth.py:derive"},
                "state_sources": _fixed_string_array(("compute_state.jsonl", "communication_state.jsonl")),
                "source_path_base": {"const": "objective episode directory"},
                "profile_source": {"const": "Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json"},
                "record_counts": _fixed_integer_map(("compute_predicate_truth", "compute_predicate_matrix", "compute_events")),
                "basis": {"const": "current saved execution states; predicates and events recomputed by the current business rules"},
                "matrix_population": {"const": "only scopes with a same-tick execution record; absent scopes are not converted to out_of_scope"},
                "recovery_scope": {"const": "event rules declare rising onsets; no recovery is inferred without a declared terminal rule"},
            }),
        },
    }
    return published_record_schema(schema)
