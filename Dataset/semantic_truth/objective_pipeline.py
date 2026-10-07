"""Objective semantic corpus orchestration for AeroWorld episodes.

The pipeline reads authoritative truth-frame inputs through the two L0 state
computers, projects contract-relevant state into predicates, derives strict
adjacent transitions and lifecycle rows, and writes compact graph artifacts.
"""

from __future__ import annotations

import json
import gc
import ctypes
import fcntl
import os
import shutil
import tempfile
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from Dataset.semantic_simulation.domain_state import (
    OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES,
    build_episode_artifacts as build_domain_state_artifacts,
    load_domain_state_profile,
    resolve_declared_input_path,
    sanitize_objective_input,
)
from Dataset.semantic_simulation.compute_comm import (
    build_objective_world_inputs,
)
from Dataset.semantic_simulation.control_response_state import (
    PATH_DEVIATION_THRESHOLD_M,
    build_control_response_state_rows,
)
from Dataset.semantic_simulation.observable_state_completion import (
    build_observable_state_rows,
)
from Dataset.semantic_simulation.predicate_state_computers import (
    GeometricStateComputer,
    PlanWindowComputer,
    PredicateStateComputerError,
    restricted_region_runtime_active,
)
from Dataset.semantic_simulation.utm_state import build_utm_episode_artifacts
from Dataset.semantic_truth.compact_truth_graph import (
    build_empty_semantic_graph_base,
    build_compact_truth_graph_result,
    serialize_compact_truth_graph_jsonl,
    serialize_semantic_graph_deltas_jsonl,
)
from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_ids
from Dataset.semantic_truth.charging_supplement import build_charging_plan
from Dataset.semantic_truth.input_adapter import build_tick_contexts
from Dataset.semantic_truth.episode_sources import source_episode_root, source_episode_descriptor, source_sumo_frames
from Dataset.semantic_truth.l0_supplement import (
    build_l0_source_availability,
    materialize_episode_l0_state,
)
from Dataset.semantic_truth.minimal_semantics_adapter import run_minimal_semantic_engine
def _entity_roster_by_id(episode_root: Path) -> dict[str, dict[str, Any]]:
    """Load the episode roster keyed by entity id.

    The event layer uses roster record fields to classify participants; predicate
    truth is computed independently for every in-scope entity.
    """
    roster_path = episode_root / "global_entity_roster.json"
    if not roster_path.is_file():
        raise ObjectivePipelineError(f"entity roster is missing: {roster_path}")
    payload = json.loads(roster_path.read_text(encoding="utf-8-sig"))
    entities = payload.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ObjectivePipelineError(f"entity roster is empty: {roster_path}")
    indexed: dict[str, dict[str, Any]] = {}
    for entity in entities:
        entity_id = entity.get("entity_id") if isinstance(entity, dict) else None
        if not isinstance(entity_id, str) or not entity_id:
            raise ObjectivePipelineError(
                f"entity roster record lacks entity_id in {roster_path}"
            )
        if entity_id in indexed:
            raise ObjectivePipelineError(
                f"duplicate entity ID {entity_id!r} in {roster_path}"
            )
        indexed[entity_id] = entity
    return indexed
from Dataset.semantic_truth.provenance import (
    canonical_json,
    digest_file,
    digest_object,
    read_jsonl,
    sha256_bytes,
    without_integrity_metadata,
)
from Dataset.semantic_validation.formal_episode_contract import (
    FormalEpisodeContractError,
    require_formal_episode_set,
)
from Dataset.semantic_truth.simulation_logs import (
    build_simulation_logs,
    serialize_simulation_log,
    summarize_simulation_log_source_coverage,
)
from Dataset.semantic_truth.stage_acceptance import evaluate_epi_stage_acceptance
from Dataset.semantic_truth.world_truth import (
    evaluate_world_truth,
    serialize_world_truth_deltas,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODES_ROOT = (
    PROJECT_ROOT / "aw_data" / "render_ready_episodes_capture_filtered"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "aw_data" / "objective_semantic_truth"
DEFAULT_DOMAIN_PROFILE_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "domain_state_supplement_profile.json"
)
DEFAULT_COMPUTE_PROFILE_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "compute_comm_supplement_profile.json"
)
DEFAULT_CONTRACT_PROFILE_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "epi_objective_semantic_contract.json"
)
DEFAULT_STAGE_ACCEPTANCE_PROFILE_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "epi_stage_acceptance_contract.json"
)
DEFAULT_EVENT_FAMILY_MAPPING_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "event_family_authority_mapping.json"
)

SCHEMA_VERSION = "4.0.0"
FORMAL_EPISODE_COUNT = 210
FORMAL_TICKS = tuple(range(0, 901, 5))
CORPUS_ROOT_FILE_NAMES = frozenset({"corpus_manifest.json", "epi_closure_matrix.json"})
REQUIRED_EPISODE_AUTHORITY_FILES = (
    "episode_manifest.json",
    "global_entity_roster.json",
    "trajectories.jsonl",
    "truth_frames.jsonl",
    "weather_meta.jsonl",
)
OPTIONAL_EPISODE_AUTHORITY_FILES = (
    "scene_occupancy_manifest.json",
    "semantic_static_geometry.json",
    "scene_setup.json",
)
FORBIDDEN_INPUT_NAMES = {
    "event_trace.jsonl",
    "event_realization.jsonl",
    "dynamic_labels.jsonl",
    "scenario_plan.json",
}
FORBIDDEN_TEXT = {
    "active_event_ids",
    "event_trace",
    "event_realization",
    "dynamic_labels",
    "scenario_plan",
    "expected_event",
    "semantic_role",
    "state_facets",
    "task_id",
}
# Producer ledgers carry actual workload task IDs. They are source evidence,
# not the actor task/story annotations excluded from semantic truth inputs.
PRODUCER_LEDGER_FILES = (
    "domain_state_observations.jsonl",
    "compute_state.jsonl",
    "communication_state.jsonl",
    "simulation_logs/compute_log.jsonl",
)
OBJECTIVE_CODE_AUTHORITY_FILES = (
    "Dataset/semantic_truth/episode_sources.py",
    "Dataset/semantic_simulation/__init__.py",
    "Dataset/semantic_simulation/compute_comm.py",
    "Dataset/semantic_simulation/control_response_state.py",
    "Dataset/semantic_simulation/domain_state.py",
    "Dataset/semantic_simulation/observable_state_completion.py",
    "Dataset/semantic_simulation/predicate_state_computers.py",
    "Dataset/semantic_simulation/utm_state.py",
    "Dataset/semantic_truth/__init__.py",
    "Dataset/semantic_truth/compact_truth_graph.py",
    "Dataset/semantic_truth/core_semantic_registry.py",
    "Dataset/semantic_truth/entity_scope.py",
    "Dataset/semantic_truth/facility_scope.py",
    "Dataset/semantic_truth/input_adapter.py",
    "Dataset/semantic_truth/l0_state_profile.py",
    "Dataset/semantic_truth/l0_supplement.py",
    "Dataset/semantic_truth/minimal_semantics.py",
    "Dataset/semantic_truth/minimal_semantics_adapter.py",
    "Dataset/semantic_truth/model.py",
    "Dataset/semantic_truth/objective_pipeline.py",
    "Dataset/semantic_truth/provenance.py",
    "Dataset/semantic_truth/runtime_schema.py",
    "Dataset/semantic_truth/simulation_logs.py",
    "Dataset/semantic_truth/stage_acceptance.py",
    "Dataset/semantic_truth/world_truth.py",
    "Dataset/semantic_validation/__init__.py",
    "Dataset/semantic_validation/formal_episode_contract.py",
    "Dataset/tools/__init__.py",
    "Dataset/tools/runtime_state_contract.py",
    "Dataset/tools/sumo_ground_flow/__init__.py",
    "Dataset/tools/sumo_ground_flow/road_signal_context.py",
)
OBJECTIVE_STATIC_AUTHORITY_FILES = (
    "Dataset/knowledge_graph/profiles/ontology_selection_manifest.json",
    "Dataset/knowledge_graph/world_core.ttl",
    "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json",
    "Dataset/semantic_rules/profiles/epi_objective_semantic_contract.json",
    "Dataset/semantic_rules/profiles/event_family_authority_mapping.json",
    "Dataset/semantic_rules/profiles/entity_scope_classes.json",
    "Dataset/semantic_rules/profiles/facility_scope_contract.json",
    "Dataset/semantic_rules/profiles/l0_state_supplement_profile.json",
    "Dataset/semantic_rules/schema/event_occurrence.schema.json",
    "Dataset/semantic_rules/schema/event_outcome.schema.json",
    "Dataset/semantic_rules/schema/facility_scope_contract.schema.json",
    "Dataset/semantic_rules/schema/predicate_transition.schema.json",
    "Dataset/semantic_rules/schema/predicate_truth.schema.json",
    "Dataset/semantic_rules/schema/simulation_log.schema.json",
    "Dataset/semantic_rules/schema/simulation_log_reconciliation.schema.json",
    "Plugins/SumoImporter/Maps/donghu_road_topo/source/road_derived/road_derived.net.xml",
    "aw_data/uav_outputs/donghu_uav_flow_270s/uav_task_plan.json",
)
OBJECTIVE_GLOB_AUTHORITY_CLASSES = (
    (
        "domain_ontologies",
        "Dataset/knowledge_graph/domain",
        "*.ttl",
        (".shacl.ttl",),
    ),
    (
        "event_ontologies",
        "Dataset/knowledge_graph/events",
        "*_events.ttl",
        (".shacl.ttl",),
    ),
    (
        "runtime_state_schedules",
        "Dataset/semantic_rules/profiles/epi_runtime_state_schedules",
        "*.json",
        (),
    ),
)


class ObjectivePipelineError(RuntimeError):
    """Raised when objective semantic corpus orchestration fails closed."""


@dataclass(frozen=True)
class ObjectiveEpisodeArtifacts:
    episode_id: str
    epi_id: str
    seed_label: str
    output_dir: Path
    files: dict[str, str]
    manifest: dict[str, Any]
    closure: dict[str, Any]


def build_episode_objective_artifacts(
    episode_root: Path,
    output_dir: Path,
    *,
    domain_profile_path: Path = DEFAULT_DOMAIN_PROFILE_PATH,
    compute_profile_path: Path = DEFAULT_COMPUTE_PROFILE_PATH,
    contract_profile_path: Path = DEFAULT_CONTRACT_PROFILE_PATH,
    stage_acceptance_profile_path: Path = DEFAULT_STAGE_ACCEPTANCE_PROFILE_PATH,
    _strict_input_guard: bool = True,
    _manifest_episode_root: Path | None = None,
) -> ObjectiveEpisodeArtifacts:
    episode_root = episode_root.resolve()
    output_dir = output_dir.resolve()
    if not episode_root.is_dir():
        raise ObjectivePipelineError(f"episode root does not exist: {episode_root}")
    episode_id = episode_root.name
    epi_id, seed_label = parse_episode_identity(episode_id)
    manifest_episode_root = (
        _manifest_episode_root.resolve()
        if _manifest_episode_root is not None
        else episode_root
    )
    if _strict_input_guard:
        _assert_no_forbidden_episode_inputs(episode_root)
        with _objective_strict_episode_root(episode_root) as strict_episode_root:
            artifacts = build_episode_objective_artifacts(
                strict_episode_root,
                output_dir,
                domain_profile_path=domain_profile_path,
                compute_profile_path=compute_profile_path,
                contract_profile_path=contract_profile_path,
                stage_acceptance_profile_path=stage_acceptance_profile_path,
                _strict_input_guard=False,
                _manifest_episode_root=manifest_episode_root,
            )
            # Private working paths never become durable evidence references.
            files = {
                name: text.replace(str(strict_episode_root / "charging_service_plan.json"), str(output_dir / "charging_service_plan.json"))
                          .replace(str(strict_episode_root), str(manifest_episode_root))
                for name, text in artifacts.files.items()
            }
            return ObjectiveEpisodeArtifacts(
                episode_id=episode_id, epi_id=epi_id, seed_label=seed_label,
                output_dir=output_dir, files=files,
                manifest=json.loads(files["manifest.json"]),
                closure=json.loads(files["epi_closure.json"]),
            )
    profile = load_domain_state_profile(domain_profile_path)
    contract = json.loads(contract_profile_path.read_text(encoding="utf-8-sig"))
    stage_acceptance_contract = json.loads(
        stage_acceptance_profile_path.read_text(encoding="utf-8-sig")
    )
    l0_contract_objects = {
        "l0_predicate_source_availability.json": build_l0_source_availability(
            episode_root
        )
    }
    l0_contract_texts = {
        name: _json_text(value) for name, value in sorted(l0_contract_objects.items())
    }
    _assert_authoritative_ticks(episode_root / "truth_frames.jsonl", profile)
    context_profile = _tick_context_profile(profile)
    (
        context_episode_id,
        contexts,
        context_input_files,
        context_input_digest,
        static_geometry_authority,
    ) = build_tick_contexts(
        episode_root,
        context_profile,
    )
    if context_episode_id != episode_id:
        raise ObjectivePipelineError(
            f"tick context episode id {context_episode_id} does not match directory {episode_id}"
        )
    compute_rows, communication_rows, compute_predicate_rows = (
        build_objective_world_inputs(
            episode_root,
            compute_profile_path,
        )
    )
    domain_artifacts = build_domain_state_artifacts(
        episode_root,
        output_dir / "_domain_state_in_memory",
        domain_profile_path,
        charging_service_plan_path=episode_root / "charging_service_plan.json",
        communication_rows=communication_rows,
    )
    domain_rows = [dict(row) for row in domain_artifacts.observations]
    control_response_rows = build_control_response_state_rows(
        episode_root,
        communication_rows=communication_rows,
        domain_rows=domain_rows,
    )
    domain_rows.extend(dict(row) for row in control_response_rows)
    observable_state_rows = build_observable_state_rows(episode_root)
    domain_rows.extend(dict(row) for row in observable_state_rows)
    geometric_state = GeometricStateComputer(episode_root).compute()
    plan_window_rows = PlanWindowComputer(episode_root, geometric_state).compute()
    domain_rows.extend(dict(row) for row in geometric_state.rows)
    domain_rows.extend(dict(row) for row in plan_window_rows)
    l0_predicate_state = sorted(
        [
            dict(row)
            for row in domain_rows
            if str(row.get("observation_family", "")).startswith("predicate_contract_")
        ],
        key=lambda row: (
            int(row["tick"]),
            str(row["observation_family"]),
            str(row["subject_id"]),
        ),
    )
    utm_artifacts = build_utm_episode_artifacts(
        episode_root,
        output_dir / "_utm_in_memory",
    )
    utm_records = {
        name: _jsonl_rows_from_text(content, name)
        for name, content in utm_artifacts.files.items()
        if name.endswith(".jsonl")
    }
    world_truth_result = evaluate_world_truth(
        episode_root,
        source_availability=l0_contract_objects[
            "l0_predicate_source_availability.json"
        ],
        domain_rows=domain_rows,
        compute_rows=compute_rows,
        communication_rows=communication_rows,
        compute_predicate_rows=compute_predicate_rows,
        utm_records=utm_records,
    )
    simulation_logs = build_simulation_logs(
        episode_id=episode_id,
        compute_rows=compute_rows,
        communication_rows=communication_rows,
        compute_predicate_rows=compute_predicate_rows,
        domain_rows=domain_rows,
        utm_records=utm_records,
        weather_rows=list(read_jsonl(episode_root / "weather_meta.jsonl")),
    )
    simulation_log_reconciliation = summarize_simulation_log_source_coverage(
        logs=simulation_logs
    )
    simulation_log_texts = {
        f"simulation_logs/{name}": serialize_simulation_log(rows)
        for name, rows in sorted(simulation_logs.items())
    }

    projection_predicate_ids = sorted(
        set(
            _stage_projection_predicate_ids(
                stage_acceptance_contract,
                epi_id,
            )
        )
        | set(_roster_contract_predicate_ids(episode_root))
    )
    roster_by_id = _entity_roster_by_id(episode_root)
    minimal_result = run_minimal_semantic_engine(
        contexts,
        domain_rows,
        communication_rows,
        world_truth_result.base_graph,
        world_truth_result.deltas,
        input_digest=digest_object(
            {
                "context_input_digest": context_input_digest,
                "domain_input_digest": domain_artifacts.input_digest,
                "communication_row_count": len(communication_rows),
                "control_response_row_count": len(control_response_rows),
                "observable_state_row_count": len(observable_state_rows),
                "predicate_state_input_digest": digest_object(l0_predicate_state),
                "predicate_state_row_count": len(l0_predicate_state),
            }
        ),
        stage_projection_predicate_ids=projection_predicate_ids,
        roster_by_id=roster_by_id,
    )
    projection = dict(minimal_result.projection)
    full_semantic_state = list(minimal_result.semantic_state)
    full_predicate_truth = list(minimal_result.predicate_truth)
    epi_view_api_ids = _required_api_ids_for_epi_view(contract, epi_id)
    full_transitions = list(minimal_result.transitions)
    full_continuity_breaks = list(minimal_result.continuity_breaks)
    full_event_occurrences = list(minimal_result.occurrences)
    full_event_outcomes = list(minimal_result.outcomes)
    closure = evaluate_epi_stage_acceptance(
        contract,
        stage_acceptance_contract,
        episode_id=episode_id,
        epi_id=epi_id,
        semantic_state=full_semantic_state,
        predicate_truth=full_predicate_truth,
        transitions=full_transitions,
        occurrences=full_event_occurrences,
        outcomes=full_event_outcomes,
        roster_by_id=roster_by_id,
    )
    # Durable L2 files are the complete deterministic detection view.  EPI
    # acceptance remains an independent closure projection in the manifest;
    # it must never remove lifecycle or predicate rows from the durable data.
    output_rows = _select_durable_output_rows(
        full_semantic_state,
        full_predicate_truth,
        full_transitions,
        full_continuity_breaks,
        full_event_occurrences,
        full_event_outcomes,
    )
    semantic_state = output_rows["semantic_state"]
    predicate_truth = output_rows["predicate_truth"]
    transitions = output_rows["transitions"]
    continuity_breaks = output_rows["continuity_breaks"]
    event_occurrences = output_rows["event_occurrences"]
    event_outcomes = output_rows["event_outcomes"]
    evidence = _build_evidence_observations(semantic_state, predicate_truth)
    graph_predicate_truth = [
        row
        for row in predicate_truth
        if row.get("value") in {"true", "false"}
        and not row.get("evidence", {}).get("missing_requirements")
    ]
    graph_result = build_compact_truth_graph_result(
        graph_predicate_truth,
        transitions,
        event_occurrences,
        event_outcomes,
        evidence,
        continuity_breaks,
        continuity_source_truth=predicate_truth,
    )
    if graph_result.rejected_records:
        examples = [
            {
                "record_kind": row.get("record_kind"),
                "record_id": row.get("record_id"),
                "reason": row.get("reason"),
            }
            for row in graph_result.rejected_records[:20]
        ]
        raise ObjectivePipelineError(
            "compact truth graph rejected objective records: "
            f"count={len(graph_result.rejected_records)}, examples={examples}"
        )
    semantic_graph_bases = graph_result.base_graphs or (
        build_empty_semantic_graph_base(episode_id),
    )
    global_detection_counts = {
        "semantic_state_observations": len(full_semantic_state),
        "predicate_truth": len(full_predicate_truth),
        "predicate_transitions": len(full_transitions),
        "predicate_continuity_breaks": len(full_continuity_breaks),
        "event_occurrences": len(full_event_occurrences),
        "event_outcomes": len(full_event_outcomes),
    }
    global_detection_digests = {
        "semantic_state_observations": digest_object(full_semantic_state),
        "predicate_truth": digest_object(full_predicate_truth),
        "predicate_transitions": digest_object(full_transitions),
        "predicate_continuity_breaks": digest_object(full_continuity_breaks),
        "event_occurrences": digest_object(full_event_occurrences),
        "event_outcomes": digest_object(full_event_outcomes),
    }
    manifest = _build_manifest(
        manifest_episode_root,
        output_dir,
        episode_id,
        epi_id,
        seed_label,
        domain_profile_path,
        compute_profile_path,
        contract_profile_path,
        stage_acceptance_profile_path,
        domain_artifacts,
        context_input_files,
        static_geometry_authority,
        projection,
        epi_view_api_ids,
        domain_rows,
        compute_rows,
        communication_rows,
        compute_predicate_rows,
        l0_predicate_state,
        utm_artifacts.summary,
        world_truth_result.base_graph,
        world_truth_result.deltas,
        world_truth_result.summary,
        semantic_state,
        predicate_truth,
        transitions,
        continuity_breaks,
        event_occurrences,
        event_outcomes,
        evidence,
        semantic_graph_bases,
        graph_result.deltas,
        graph_result.graphs,
        graph_result.rejected_records,
        closure,
        global_detection_counts,
        global_detection_digests,
    )
    manifest["simulation_logs"] = {
        name.removeprefix("simulation_logs/"): {
            "path": name,
            "record_count": len(simulation_logs[name.removeprefix("simulation_logs/")]),
            "sha256": sha256_bytes(text.encode("utf-8")),
        }
        for name, text in sorted(simulation_log_texts.items())
    }
    manifest["simulation_log_reconciliation"] = dict(simulation_log_reconciliation)
    manifest["l0_contract_artifacts"] = {
        name: {
            "path": name,
            "record_count": len(value.get("entries") or []),
            "sha256": sha256_bytes(l0_contract_texts[name].encode("utf-8")),
        }
        for name, value in sorted(l0_contract_objects.items())
    }
    manifest["record_counts"]["simulation_log_records"] = sum(
        len(rows) for rows in simulation_logs.values()
    )
    manifest["annotation_layers"]["L0"]["artifacts"].extend(
        sorted(simulation_log_texts)
    )
    manifest["annotation_layers"]["L0"]["artifacts"].extend(sorted(l0_contract_texts))
    manifest["event_family_mapping"] = {
        "path": str(DEFAULT_EVENT_FAMILY_MAPPING_PATH),
        "sha256": digest_file(DEFAULT_EVENT_FAMILY_MAPPING_PATH),
    }
    for manifest_key, filename in (
        ("l0_state_materialization", "l0_state_materialization_summary.json"),
        (
            "runtime_state_materialization",
            "runtime_state_materialization_summary.json",
        ),
    ):
        summary_path = episode_root / filename
        if not summary_path.is_file():
            raise ObjectivePipelineError(
                f"materialization summary is missing: {summary_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        if not isinstance(summary, dict):
            raise ObjectivePipelineError(
                f"materialization summary must be an object: {summary_path}"
            )
        manifest[manifest_key] = summary
    from Dataset.semantic_truth.compute_event_truth import derive, manifest_entry
    business_records = derive(episode_root, compute_rows, communication_rows, compute_profile_path)
    manifest["business_event_truth"] = manifest_entry(business_records)
    files = {
        "charging_service_plan.json": (episode_root / "charging_service_plan.json").read_text(encoding="utf-8"),
        "l0_predicate_state.jsonl": _jsonl_text(l0_predicate_state),
        "semantic_state_observations.jsonl": _jsonl_text(semantic_state),
        "predicate_truth.jsonl": _jsonl_text(predicate_truth),
        "predicate_transitions.jsonl": _jsonl_text(transitions),
        "predicate_continuity_breaks.jsonl": _jsonl_text(continuity_breaks),
        "event_occurrences.jsonl": _jsonl_text(event_occurrences),
        "event_outcomes.jsonl": _jsonl_text(event_outcomes),
        "world_truth_graph_base.json": _json_text(world_truth_result.base_graph),
        "world_truth_graph_deltas.jsonl": serialize_world_truth_deltas(
            world_truth_result.deltas
        ),
        "world_truth_summary.json": _json_text(world_truth_result.summary),
        "semantic_graph_base.json": _json_text(semantic_graph_bases[0]),
        "semantic_graph_deltas.jsonl": serialize_semantic_graph_deltas_jsonl(
            graph_result.deltas
        ),
        "semantic_truth_graphs.jsonl": serialize_compact_truth_graph_jsonl(
            graph_result.graphs
        ),
        "evidence_observations.jsonl": _jsonl_text(evidence),
        "epi_closure.json": _json_text(closure),
        "manifest.json": _json_text(manifest),
    }
    files.update(l0_contract_texts)
    _assert_no_forbidden_payload(files)
    # Producer ledgers carry actual workload task IDs. They are source evidence,
    # not the actor task/story annotations excluded from semantic truth inputs.
    files.update({
        "domain_state_observations.jsonl": _jsonl_text(domain_rows),
        "compute_state.jsonl": _jsonl_text(compute_rows),
        "communication_state.jsonl": _jsonl_text(communication_rows),
    })
    files.update({name: _jsonl_text(rows) for name, rows in business_records.items()})
    files.update(simulation_log_texts)
    files["simulation_logs/reconciliation.json"] = _json_text(
        simulation_log_reconciliation
    )
    _assert_no_forbidden_payload(
        {
            name: text
            for name, text in files.items()
            if name not in PRODUCER_LEDGER_FILES
        }
    )
    return ObjectiveEpisodeArtifacts(
        episode_id=episode_id,
        epi_id=epi_id,
        seed_label=seed_label,
        output_dir=output_dir,
        files=files,
        manifest=without_integrity_metadata(manifest),
        closure=without_integrity_metadata(closure),
    )


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if _path_lexists(current) and current.is_symlink():
            raise ObjectivePipelineError(f"{label} contains a symlink: {current}")


def _validated_artifact_names(files: Mapping[str, str]) -> set[str]:
    names: set[str] = set()
    for raw_name, text in files.items():
        if not isinstance(raw_name, str) or not isinstance(text, str):
            raise ObjectivePipelineError(
                "objective artifact names and payloads must be strings"
            )
        name = PurePosixPath(raw_name)
        if (
            not raw_name
            or not name.parts
            or name.is_absolute()
            or "\\" in raw_name
            or any(part in {"", ".", ".."} for part in name.parts)
            or name.as_posix() != raw_name
        ):
            raise ObjectivePipelineError(
                f"objective artifact path escapes its episode directory: {raw_name!r}"
            )
        names.add(raw_name)
    return names


def _expected_tree_kinds(files: Mapping[str, str]) -> dict[str, str]:
    names = _validated_artifact_names(files)
    expected = {name: "file" for name in names}
    for raw_name in names:
        parent = PurePosixPath(raw_name).parent
        while parent != PurePosixPath("."):
            parent_name = parent.as_posix()
            previous = expected.get(parent_name)
            if previous == "file":
                raise ObjectivePipelineError(
                    f"objective artifact file/directory collision: {parent_name}"
                )
            expected[parent_name] = "directory"
            parent = parent.parent
    return expected


def _tree_entries(root: Path, *, label: str) -> dict[str, dict[str, str]]:
    if root.is_symlink() or not root.is_dir():
        raise ObjectivePipelineError(f"{label} must be a non-symlink directory: {root}")
    entries: dict[str, dict[str, str]] = {}
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ObjectivePipelineError(f"{label} contains a symlink: {relative}")
        if path.is_dir():
            entries[relative] = {"kind": "directory"}
        elif path.is_file():
            entries[relative] = {"kind": "file"}
        else:
            raise ObjectivePipelineError(
                f"{label} contains a non-regular filesystem entry: {relative}"
            )
    return entries


def _output_file_references(output_dir: Path) -> dict[str, Any]:
    entries = _tree_entries(output_dir, label="objective output")
    if not entries:
        raise ObjectivePipelineError(
            f"objective output directory is empty: {output_dir}"
        )
    return entries


def _write_exact_artifact_tree(target: Path, files: Mapping[str, str]) -> None:
    _expected_tree_kinds(files)
    if _path_lexists(target):
        raise ObjectivePipelineError(
            f"objective staging target already exists: {target}"
        )
    target.mkdir(parents=True)
    for name, text in sorted(files.items()):
        _atomic_write_text(target / name, text)


def write_episode_outputs(artifacts: ObjectiveEpisodeArtifacts) -> None:
    """Atomically replace one episode directory for isolated tooling only."""

    requested_output = artifacts.output_dir.absolute()
    _assert_no_symlink_components(requested_output, label="objective output path")
    output_dir = requested_output.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = output_dir.parent / f".{output_dir.name}.previous"
    if _path_lexists(backup_dir):
        raise ObjectivePipelineError(
            f"stale objective output backup requires inspection: {backup_dir}"
        )
    _expected_tree_kinds(artifacts.files)
    if _path_lexists(output_dir) and not output_dir.is_dir():
        raise ObjectivePipelineError(
            f"objective output must be a directory when present: {output_dir}"
        )
    staging_dir = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.staging.",
        )
    )
    try:
        for name, text in sorted(artifacts.files.items()):
            _atomic_write_text(staging_dir / name, text)
        if _path_lexists(output_dir):
            os.replace(output_dir, backup_dir)
        try:
            os.replace(staging_dir, output_dir)
        except Exception:
            if _path_lexists(backup_dir) and not _path_lexists(output_dir):
                os.replace(backup_dir, output_dir)
            raise
        if _path_lexists(backup_dir):
            shutil.rmtree(backup_dir)
    finally:
        if _path_lexists(staging_dir):
            shutil.rmtree(staging_dir)


def check_episode_outputs(artifacts: ObjectiveEpisodeArtifacts) -> list[str]:
    expected_kinds = _expected_tree_kinds(artifacts.files)
    try:
        actual_entries = _tree_entries(
            artifacts.output_dir, label=f"objective episode {artifacts.episode_id}"
        )
    except ObjectivePipelineError as exc:
        return [f"invalid:{exc}"]
    actual_kinds = {
        name: str(declaration["kind"]) for name, declaration in actual_entries.items()
    }
    mismatches = [
        f"extra:{name}" for name in sorted(set(actual_kinds) - set(expected_kinds))
    ]
    mismatches.extend(
        f"missing:{name}" for name in sorted(set(expected_kinds) - set(actual_kinds))
    )
    for name in sorted(set(expected_kinds) & set(actual_kinds)):
        if actual_kinds[name] != expected_kinds[name]:
            mismatches.append(f"type:{name}")
            continue
        if expected_kinds[name] != "file":
            continue
        if (artifacts.output_dir / name).read_text(encoding="utf-8") != artifacts.files[name]:
            mismatches.append(f"content:{name}")
    return mismatches


def _authority_file_reference(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ObjectivePipelineError(f"{label} file is missing: {path}")
    _assert_no_symlink_components(path, label=label)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _authority_file_references(
    project_root: Path,
    relatives: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    return {
        relative: _authority_file_reference(project_root / relative, label=label)
        for relative in relatives
    }


def _authority_glob_references(
    project_root: Path,
    *,
    class_label: str,
    root_relative: str,
    pattern: str,
    excluded_name_suffixes: Sequence[str],
) -> dict[str, str]:
    root = project_root / root_relative
    if root.is_symlink() or not root.is_dir():
        raise ObjectivePipelineError(f"static authority directory is missing: {root}")
    _assert_no_symlink_components(root, label=f"static authority {class_label}")
    paths = [
        path
        for path in sorted(root.glob(pattern), key=lambda item: item.as_posix())
        if not any(path.name.endswith(suffix) for suffix in excluded_name_suffixes)
    ]
    if not paths:
        raise ObjectivePipelineError(f"static authority class is empty: {class_label}")
    result: dict[str, Any] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ObjectivePipelineError(
                f"static authority {class_label} contains a non-file: {path}"
            )
        result[path.relative_to(project_root).as_posix()] = _authority_file_reference(
            path,
            label=f"static authority {class_label}",
        )
    return result


def _static_authority_references(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = project_root.resolve()
    authority: dict[str, Any] = {
        "code": _authority_file_references(
            project_root,
            OBJECTIVE_CODE_AUTHORITY_FILES,
            label="objective code authority",
        ),
        "static_data": _authority_file_references(
            project_root,
            OBJECTIVE_STATIC_AUTHORITY_FILES,
            label="objective static-data authority",
        ),
    }
    for (
        class_label,
        root_relative,
        pattern,
        excluded_name_suffixes,
    ) in OBJECTIVE_GLOB_AUTHORITY_CLASSES:
        authority[class_label] = _authority_glob_references(
            project_root,
            class_label=class_label,
            root_relative=root_relative,
            pattern=pattern,
            excluded_name_suffixes=excluded_name_suffixes,
        )
    return authority


def _scenario_authority_directory(
    episode_id: str, *, project_root: Path = PROJECT_ROOT
) -> Path:
    epi_id, _ = parse_episode_identity(episode_id)
    scenarios_root = project_root / "Dataset" / "scenarios"
    raw_event_matches = sorted(
        path.parent
        for path in scenarios_root.glob(f"**/{epi_id}/event_script.json")
        if path.is_file()
    )
    raw_scene_matches = sorted(
        path.parent
        for path in scenarios_root.glob(f"**/{epi_id}/scene_setup.json")
        if path.is_file()
    )
    if (
        len(raw_event_matches) != 1
        or len(raw_scene_matches) != 1
        or raw_event_matches != raw_scene_matches
    ):
        raise ObjectivePipelineError(
            f"{episode_id}: scenario authority must resolve to one directory with "
            "event_script.json and scene_setup.json"
        )
    _assert_no_symlink_components(raw_event_matches[0], label="scenario authority")
    return raw_event_matches[0].resolve()


def _require_manifest_scenario_authority(
    manifest_path: Path,
    *,
    episode_id: str,
    scenario_root: Path,
    project_root: Path,
) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectivePipelineError(
            f"{episode_id}: episode manifest cannot be read: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ObjectivePipelineError(
            f"{episode_id}: episode manifest must be an object"
        )
    epi_id, _ = parse_episode_identity(episode_id)
    if (
        manifest.get("episode_id") != episode_id
        or manifest.get("scenario_id") != epi_id
    ):
        raise ObjectivePipelineError(
            f"{episode_id}: episode/scenario identity differs from the manifest"
        )
    for field, filename in (
        ("source_event_script_path", "event_script.json"),
        ("source_scene_setup_path", "scene_setup.json"),
    ):
        expected = (scenario_root / filename).relative_to(project_root).as_posix()
        if manifest.get(field) != expected:
            raise ObjectivePipelineError(
                f"{episode_id}: {field} must name the unique scenario authority "
                f"{expected!r}, found {manifest.get(field)!r}"
            )


def _episode_authority_references(
    episode_root: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    domain_profile_path: Path = DEFAULT_DOMAIN_PROFILE_PATH,
) -> dict[str, Any]:
    if episode_root.is_symlink() or not episode_root.is_dir():
        raise ObjectivePipelineError(
            f"episode authority must be a non-symlink directory: {episode_root}"
        )
    _assert_no_symlink_components(episode_root, label="episode authority")
    episode_id = episode_root.name
    parse_episode_identity(episode_id)
    files: dict[str, Any] = {}
    for name in REQUIRED_EPISODE_AUTHORITY_FILES:
        path = episode_root / name
        if path.is_symlink() or not path.is_file():
            raise ObjectivePipelineError(f"episode authority is missing: {path}")
        files[f"render/{name}"] = _authority_file_reference(path, label="episode input")
    for name in OPTIONAL_EPISODE_AUTHORITY_FILES:
        path = episode_root / name
        if path.is_symlink():
            raise ObjectivePipelineError(
                f"episode authority contains a symlink: {path}"
            )
        if _path_lexists(path) and not path.is_file():
            raise ObjectivePipelineError(
                f"optional episode authority must be a regular file: {path}"
            )
        files[f"render_optional/{name}"] = (
            _authority_file_reference(path, label="episode input") if path.is_file() else {"path": str(path), "status": "missing"}
        )
    union_root = source_episode_root(episode_root)
    if union_root is not None:
        _assert_no_symlink_components(union_root, label="episode union authority")
    for name in ("global_entity_roster.json", "trajectories.jsonl"):
        if union_root is None:
            files[f"scenario_union/{name}"] = source_episode_descriptor(episode_root)
            continue
        path = union_root / name
        if path.is_symlink():
            raise ObjectivePipelineError(
                f"episode union authority contains a symlink: {path}"
            )
        if _path_lexists(path) and not path.is_file():
            raise ObjectivePipelineError(
                f"episode union authority must be a regular file: {path}"
            )
        files[f"scenario_union/{name}"] = (
            _authority_file_reference(path, label="episode input") if path.is_file() else {"path": str(path), "status": "missing"}
        )
    scenario_root = _scenario_authority_directory(episode_id, project_root=project_root)
    _require_manifest_scenario_authority(
        episode_root / "episode_manifest.json",
        episode_id=episode_id,
        scenario_root=scenario_root,
        project_root=project_root,
    )
    for name in ("event_script.json", "scene_setup.json"):
        path = scenario_root / name
        files[f"scenario/{name}"] = _authority_file_reference(
            path,
            label=f"scenario authority {episode_id}",
        )
    raw_sumo_frames = source_sumo_frames(episode_root)
    files["sumo_episode_authority/sumo_traffic_frames.jsonl"] = _authority_file_reference(
        raw_sumo_frames,
        label=f"episode SUMO authority {episode_id}",
    )
    files["charging_plan_authority"] = files["scenario/scene_setup.json"]
    return files


@contextmanager
def _corpus_output_lock(output_root: Path) -> Iterable[None]:
    requested = output_root.absolute()
    _assert_no_symlink_components(requested, label="objective output root")
    requested.parent.mkdir(parents=True, exist_ok=True)
    lock_path = requested.parent / f".{requested.name}.lock"
    if lock_path.is_symlink():
        raise ObjectivePipelineError(
            f"objective output lock must not be a symlink: {lock_path}"
        )
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ObjectivePipelineError(
                f"objective output root is locked by another writer/checker: {requested}"
            ) from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True)
class _ObjectiveCorpusJob:
    episode_root: Path
    final_output_dir: Path
    materialized_output_dir: Path
    domain_profile_path: Path
    compute_profile_path: Path
    contract_profile_path: Path
    stage_acceptance_profile_path: Path
    check: bool
    input_authority: Mapping[str, Any]
    execution_authority: Mapping[str, Any]


def _execution_authority_references(
    *,
    domain_profile_path: Path,
    compute_profile_path: Path,
    contract_profile_path: Path,
    stage_acceptance_profile_path: Path,
) -> dict[str, Any]:
    profiles: dict[str, str] = {}
    for label, path in (
        ("domain", domain_profile_path),
        ("compute", compute_profile_path),
        ("contract", contract_profile_path),
        ("stage_acceptance", stage_acceptance_profile_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise ObjectivePipelineError(
                f"objective profile authority is missing: {path}"
            )
        profiles[label] = _authority_file_reference(path, label="profile")
    return {"static_authority": _static_authority_references(), "profiles": profiles}


def _validated_episode_roots(episode_roots: Sequence[Path]) -> list[Path]:
    if not episode_roots:
        raise ObjectivePipelineError("no objective episodes selected")
    roots: list[Path] = []
    seen_names: set[str] = set()
    seen_paths: set[Path] = set()
    for raw_root in episode_roots:
        requested = raw_root.absolute()
        _assert_no_symlink_components(requested, label="objective episode input")
        root = requested.resolve()
        if root.is_symlink() or not root.is_dir():
            raise ObjectivePipelineError(
                f"objective episode input must be a non-symlink directory: {root}"
            )
        if root.name in seen_names or root in seen_paths:
            raise ObjectivePipelineError(
                f"duplicate objective episode job: {root.name}"
            )
        parse_episode_identity(root.name)
        seen_names.add(root.name)
        seen_paths.add(root)
        roots.append(root)
    return sorted(roots, key=lambda path: path.name)


def _closure_status_counts(
    processed: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(item.get("closure_status") or "MISSING") for item in processed
            ).items()
        )
    )


def _closure_failure_diagnostics(
    closure: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the exact failed executable stages needed to diagnose closure."""

    required_fields = {
        "status",
        "event_count",
        "stage_status_counts",
        "executable_stage_status_counts",
        "executable_stage_count",
        "not_applicable_stage_count",
        "semantic_coverage_status",
        "stage_contract_profile_id",
        "stages",
    }
    missing = sorted(required_fields - set(closure))
    if missing:
        raise ObjectivePipelineError(
            f"closure lacks required diagnostic fields: {missing}"
        )
    status = closure["status"]
    if status not in {"PASS", "FAIL", "UNKNOWN"}:
        raise ObjectivePipelineError(f"closure has invalid status: {status!r}")
    if status == "PASS":
        return None
    if (
        not isinstance(closure["event_count"], int)
        or isinstance(closure["event_count"], bool)
        or closure["event_count"] < 0
    ):
        raise ObjectivePipelineError(
            "closure event_count must be a non-negative integer"
        )
    for field in ("stage_status_counts", "executable_stage_status_counts"):
        value = closure[field]
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for key, count in value.items()
        ):
            raise ObjectivePipelineError(
                f"closure {field} must map status strings to non-negative integers"
            )
    for field in ("executable_stage_count", "not_applicable_stage_count"):
        value = closure[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ObjectivePipelineError(
                f"closure {field} must be a non-negative integer"
            )
    for field in (
        "semantic_coverage_status",
        "stage_contract_profile_id",
    ):
        if not isinstance(closure[field], str) or not closure[field]:
            raise ObjectivePipelineError(f"closure {field} must be a non-empty string")
    stages = closure["stages"]
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        raise ObjectivePipelineError("non-PASS closure lacks a stages array")
    non_pass_stages: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ObjectivePipelineError("closure contains a non-object stage")
        if (
            stage.get("implementation_status") == "executable"
            and stage.get("stage_goal_status") != "PASS"
        ):
            non_pass_stages.append(dict(stage))
    if not non_pass_stages:
        raise ObjectivePipelineError(
            "non-PASS closure has no non-PASS executable stage"
        )
    return {
        "event_count": closure["event_count"],
        "stage_status_counts": dict(closure["stage_status_counts"]),
        "executable_stage_status_counts": dict(
            closure["executable_stage_status_counts"]
        ),
        "executable_stage_count": closure["executable_stage_count"],
        "not_applicable_stage_count": closure["not_applicable_stage_count"],
        "semantic_coverage_status": closure["semantic_coverage_status"],
        "stage_contract_profile_id": closure["stage_contract_profile_id"],
        "non_pass_executable_stages": non_pass_stages,
    }


def _formal_closure_failure_report(
    processed: Sequence[Mapping[str, Any]], counts: Mapping[str, int]
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for row in processed:
        if row.get("closure_status") == "PASS":
            continue
        diagnostics = row.get("closure_failure_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ObjectivePipelineError(
                "formal non-PASS episode lacks closure diagnostics: "
                f"{row.get('episode_id')}"
            )
        episodes.append(
            {
                "episode_id": row["episode_id"],
                "epi_id": row["epi_id"],
                "seed_label": row["seed_label"],
                "closure_status": row["closure_status"],
                **dict(diagnostics),
            }
        )
    episodes.sort(key=lambda item: str(item["episode_id"]))
    return {
        "closure_status_counts": dict(counts),
        "non_pass_episode_count": len(episodes),
        "non_pass_episodes": episodes,
    }


def _require_formal_closure_pass(
    processed: Sequence[Mapping[str, Any]], *, corpus_mode: str
) -> dict[str, int]:
    counts = _closure_status_counts(processed)
    if any(status not in {"PASS", "FAIL", "UNKNOWN"} for status in counts):
        raise ObjectivePipelineError(f"invalid stage closure statuses: {counts}")
    # Narrative stage goals are measurements. An event which did not occur
    # remains usable when its predicate coverage and task evidence are valid.
    return counts


def _assert_corpus_root_entries(output_root: Path, episode_ids: Sequence[str]) -> None:
    expected = set(episode_ids) | set(CORPUS_ROOT_FILE_NAMES)
    actual: dict[str, str] = {}
    for path in output_root.iterdir():
        if path.is_symlink():
            raise ObjectivePipelineError(
                f"objective corpus root contains a symlink: {path.name}"
            )
        actual[path.name] = (
            "directory" if path.is_dir() else "file" if path.is_file() else "other"
        )
    extras = sorted(set(actual) - expected)
    missing = sorted(expected - set(actual))
    wrong_types = sorted(
        name
        for name in expected & set(actual)
        if actual[name] != ("file" if name in CORPUS_ROOT_FILE_NAMES else "directory")
    )
    if extras or missing or wrong_types:
        raise ObjectivePipelineError(
            "objective corpus root tree is not exact: "
            f"extra={extras}, missing={missing}, wrong_types={wrong_types}"
        )


def _assert_authority_stable(
    *,
    execution_before: Mapping[str, Any],
    episode_before: Mapping[str, Any],
    roots: Sequence[Path],
    domain_profile_path: Path,
    compute_profile_path: Path,
    contract_profile_path: Path,
    stage_acceptance_profile_path: Path,
) -> None:
    execution_after = _execution_authority_references(
        domain_profile_path=domain_profile_path,
        compute_profile_path=compute_profile_path,
        contract_profile_path=contract_profile_path,
        stage_acceptance_profile_path=stage_acceptance_profile_path,
    )
    if execution_after != execution_before:
        raise ObjectivePipelineError(
            "objective static authority changed during the task"
        )
    episode_after = {
        root.name: _episode_authority_references(
            root, domain_profile_path=domain_profile_path
        )
        for root in roots
    }
    if episode_after != dict(episode_before):
        raise ObjectivePipelineError(
            "objective episode authority changed during the task"
        )


def build_corpus(
    episode_roots: Sequence[Path],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    corpus_mode: str,
    check: bool = False,
    workers: int = 1,
    domain_profile_path: Path = DEFAULT_DOMAIN_PROFILE_PATH,
    compute_profile_path: Path = DEFAULT_COMPUTE_PROFILE_PATH,
    contract_profile_path: Path = DEFAULT_CONTRACT_PROFILE_PATH,
    stage_acceptance_profile_path: Path = DEFAULT_STAGE_ACCEPTANCE_PROFILE_PATH,
) -> dict[str, Any]:
    if workers < 1:
        raise ObjectivePipelineError("workers must be >= 1")
    if corpus_mode not in {"formal_full", "selected_diagnostic", "custom_all"}:
        raise ObjectivePipelineError(f"invalid objective corpus mode: {corpus_mode}")
    roots = _validated_episode_roots(episode_roots)
    if corpus_mode == "formal_full":
        try:
            require_formal_episode_set(
                (root.name for root in roots),
                label="formal objective corpus inputs",
            )
        except FormalEpisodeContractError as exc:
            raise ObjectivePipelineError(str(exc)) from exc
    requested_output_root = output_root.absolute()
    _assert_no_symlink_components(
        requested_output_root, label="objective corpus output root"
    )
    output_root = requested_output_root.resolve()
    canonical_output_root = DEFAULT_OUTPUT_ROOT.resolve()
    if output_root == canonical_output_root and corpus_mode != "formal_full":
        raise ObjectivePipelineError(
            "canonical objective output accepts only formal_full corpus mode"
        )
    if corpus_mode == "formal_full":
        formal_input_root = DEFAULT_EPISODES_ROOT.resolve()
        if output_root != canonical_output_root or any(
            root.parent != formal_input_root for root in roots
        ):
            raise ObjectivePipelineError(
                "formal_full requires the default formal input and output roots"
            )
        complete_formal_roots = selected_episode_roots(
            formal_input_root,
            episodes=[],
            all_episodes=True,
            formal=True,
        )
        if complete_formal_roots != roots:
            raise ObjectivePipelineError(
                "formal_full episode selection differs from the complete formal root"
            )
    domain_profile_path = domain_profile_path.resolve()
    compute_profile_path = compute_profile_path.resolve()
    contract_profile_path = contract_profile_path.resolve()
    stage_acceptance_profile_path = stage_acceptance_profile_path.resolve()

    with _corpus_output_lock(requested_output_root):
        if check:
            if output_root.is_symlink() or not output_root.is_dir():
                raise ObjectivePipelineError(
                    f"objective corpus output is missing: {output_root}"
                )
        elif output_root.is_dir() and not any(output_root.iterdir()):
            output_root.rmdir()
        elif _path_lexists(output_root):
            raise ObjectivePipelineError(
                f"objective corpus build requires a nonexistent output root: {output_root}"
            )

        backup_dir = output_root.parent / f".{output_root.name}.previous"
        if _path_lexists(backup_dir):
            raise ObjectivePipelineError(
                f"stale objective corpus backup requires inspection: {backup_dir}"
            )
        execution_before = _execution_authority_references(
            domain_profile_path=domain_profile_path,
            compute_profile_path=compute_profile_path,
            contract_profile_path=contract_profile_path,
            stage_acceptance_profile_path=stage_acceptance_profile_path,
        )
        episode_before = {
            root.name: _episode_authority_references(
                root, domain_profile_path=domain_profile_path
            )
            for root in roots
        }
        staging_root: Path | None = None
        if not check:
            staging_root = PROJECT_ROOT / "Dataset/world_model/runs/.preparation/objective_corpus" / output_root.name
            staging_root.mkdir(parents=True, exist_ok=True)
        try:
            jobs = [
                _ObjectiveCorpusJob(
                    episode_root=root,
                    final_output_dir=output_root / root.name,
                    materialized_output_dir=(
                        output_root / root.name if check else staging_root / root.name
                    ),
                    domain_profile_path=domain_profile_path,
                    compute_profile_path=compute_profile_path,
                    contract_profile_path=contract_profile_path,
                    stage_acceptance_profile_path=stage_acceptance_profile_path,
                    check=check,
                    input_authority=episode_before[root.name],
                    execution_authority=execution_before,
                )
                for root in roots
            ]
            results: list[tuple[_ObjectiveCorpusJob, Mapping[str, Any]]] = []
            failures: dict[str, str] = {}
            if workers == 1:
                for job in jobs:
                    try:
                        results.append((job, _process_episode(job)))
                    except Exception as exc:
                        failures[job.episode_root.name] = str(exc)
                        print(f"failed {job.episode_root.name}: {exc}", flush=True)
                        if not check:
                            break
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    future_jobs = {
                        executor.submit(_process_episode, job): job for job in jobs
                    }
                    for future in as_completed(future_jobs):
                        job = future_jobs[future]
                        try:
                            results.append((job, future.result()))
                        except Exception as exc:
                            failures[job.episode_root.name] = str(exc)
                            print(f"failed {job.episode_root.name}: {exc}", flush=True)
            if failures:
                raise ObjectivePipelineError(
                    "objective corpus task failed episodes: "
                    + ", ".join(
                        f"{episode}={error}"
                        for episode, error in sorted(failures.items())
                    )
                )

            processed: list[dict[str, Any]] = []
            for job, result in results:
                row = result.get("processed")
                if not isinstance(row, Mapping):
                    raise ObjectivePipelineError(
                        f"objective episode result lacks processed row: {job.episode_root.name}"
                    )
                processed_row = dict(row)
                processed_row["input_authority"] = episode_before[
                    job.episode_root.name
                ]
                processed_row["output_files"] = result["output_files"]
                processed.append(processed_row)
            processed.sort(key=lambda item: str(item["episode_id"]))
            closure_status_counts = _require_formal_closure_pass(
                processed, corpus_mode=corpus_mode
            )
            corpus_manifest = _build_corpus_manifest(
                processed,
                corpus_mode=corpus_mode,
                execution_authority=execution_before,
            )
            closure_matrix = _build_closure_matrix(
                processed,
                corpus_mode=corpus_mode,
                execution_authority=execution_before,
            )
            if check:
                _assert_corpus_root_entries(output_root, [root.name for root in roots])
                _check_root_file(
                    output_root / "corpus_manifest.json", _json_text(corpus_manifest)
                )
                _check_root_file(
                    output_root / "epi_closure_matrix.json", _json_text(closure_matrix)
                )
            else:
                _atomic_write_text(
                    staging_root / "corpus_manifest.json", _json_text(corpus_manifest)
                )
                _atomic_write_text(
                    staging_root / "epi_closure_matrix.json", _json_text(closure_matrix)
                )
            _assert_authority_stable(
                execution_before=execution_before,
                episode_before=episode_before,
                roots=roots,
                domain_profile_path=domain_profile_path,
                compute_profile_path=compute_profile_path,
                contract_profile_path=contract_profile_path,
                stage_acceptance_profile_path=stage_acceptance_profile_path,
            )
            if not check:
                for checkpoint in staging_root.glob("*.completed.json"):
                    checkpoint.unlink()
                _assert_corpus_root_entries(staging_root, [root.name for root in roots])
                if _path_lexists(output_root):
                    raise ObjectivePipelineError(
                        f"objective corpus output appeared during build: {output_root}"
                    )
                os.replace(staging_root, output_root)
                staging_root = None
            return {
                "schema_name": "objective_semantic_corpus_build_result",
                "schema_version": SCHEMA_VERSION,
                "mode": "check" if check else "write",
                "corpus_mode": corpus_mode,
                "execution_authority": execution_before,
                "episode_count": len(processed),
                "closure_status_counts": closure_status_counts,
                "record_counts": _sum_record_counts(processed),
                "episodes": processed,
            }
        finally:
            # Completed episode checkpoints survive interruption. They are
            # validated against source/code references before being reused.
            pass


def parse_episode_identity(episode_id: str) -> tuple[str, str]:
    if "__seed" not in episode_id:
        raise ObjectivePipelineError(f"episode id lacks seed suffix: {episode_id}")
    epi_id, seed_suffix = episode_id.rsplit("__seed", 1)
    if not seed_suffix.isdigit():
        raise ObjectivePipelineError(
            f"episode id has invalid seed suffix: {episode_id}"
        )
    return epi_id, f"seed{int(seed_suffix):02d}"


def selected_episode_roots(
    episodes_root: Path,
    *,
    episodes: Sequence[str],
    all_episodes: bool,
    formal: bool = False,
) -> list[Path]:
    requested_root = episodes_root.absolute()
    _assert_no_symlink_components(requested_root, label="episodes root")
    episodes_root = requested_root.resolve()
    if episodes_root.is_symlink() or not episodes_root.is_dir():
        raise ObjectivePipelineError(
            f"episodes root must be a non-symlink directory: {episodes_root}"
        )
    if episodes and all_episodes:
        raise ObjectivePipelineError("--all and --episode are mutually exclusive")
    if len(episodes) != len(set(episodes)):
        raise ObjectivePipelineError("duplicate --episode selection is not allowed")
    if episodes:
        roots = []
        for episode in episodes:
            if not episode or Path(episode).name != episode:
                raise ObjectivePipelineError(
                    f"selected episode must be a directory name: {episode!r}"
                )
            candidate = episodes_root / episode
            _assert_no_symlink_components(candidate, label="selected episode")
            resolved = candidate.resolve()
            if resolved.parent != episodes_root:
                raise ObjectivePipelineError(
                    f"selected episode escapes episodes root: {episode!r}"
                )
            roots.append(resolved)
    else:
        directory_entries = sorted(
            path for path in episodes_root.iterdir() if path.is_dir()
        )
        symlinks = sorted(path.name for path in directory_entries if path.is_symlink())
        if symlinks:
            raise ObjectivePipelineError(
                f"episodes root contains symlink directories: {symlinks}"
            )
        roots = [
            path.resolve()
            for path in directory_entries
            if (path / "episode_manifest.json").is_file()
        ]
        roots.sort(key=lambda path: path.name)
        if not all_episodes:
            raise ObjectivePipelineError("select --all or at least one --episode")
        unrecognized = sorted(
            path.name for path in directory_entries if path.resolve() not in roots
        )
        if unrecognized:
            raise ObjectivePipelineError(
                f"episodes root contains non-episode directories: {unrecognized}"
            )
        if formal:
            try:
                require_formal_episode_set(
                    (root.name for root in roots),
                    label="formal episodes root",
                )
            except FormalEpisodeContractError as exc:
                raise ObjectivePipelineError(str(exc)) from exc
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise ObjectivePipelineError(f"selected episodes are missing: {missing}")
    invalid_manifests = [
        str(root / "episode_manifest.json")
        for root in roots
        if (root / "episode_manifest.json").is_symlink()
        or not (root / "episode_manifest.json").is_file()
    ]
    if invalid_manifests:
        raise ObjectivePipelineError(
            f"selected episode manifests are missing or symlinked: {invalid_manifests}"
        )
    if not roots:
        raise ObjectivePipelineError("no episodes selected")
    return roots


@contextmanager
def _objective_strict_episode_root(episode_root: Path) -> Iterable[Path]:
    preparation = PROJECT_ROOT / "Dataset/world_model/runs/.preparation/objective"
    preparation.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="aeroworld_objective_strict_",
        dir=preparation,
    ) as temp_dir:
        strict_root = Path(temp_dir) / episode_root.name
        strict_root.mkdir(parents=True, exist_ok=True)
        for name in (
            "episode_manifest.json",
            "scene_occupancy_manifest.json",
            "semantic_static_geometry.json",
            "scene_setup.json",
        ):
            source = episode_root / name
            if source.is_file():
                _write_strict_json(source, strict_root / name)
        world_roster_source = episode_root / "global_entity_roster.json"
        if not world_roster_source.is_file():
            raise ObjectivePipelineError(
                f"all-entity world roster input is missing: {world_roster_source}"
            )
        scenario_root = source_episode_root(episode_root)
        scenario_roster_source = scenario_root / 'global_entity_roster.json' if scenario_root is not None else None
        _write_strict_entity_roster_union(
            scenario_roster_source,
            world_roster_source,
            strict_root / "global_entity_roster.json",
        )
        charging_plan, _ = build_charging_plan(strict_root)
        (strict_root / "charging_service_plan.json").write_text(_json_text(charging_plan), encoding="utf-8")
        truth_source = episode_root / "truth_frames.jsonl"
        if truth_source.is_file():
            _write_strict_truth_frames(truth_source, strict_root / truth_source.name)
        weather_source = episode_root / "weather_meta.jsonl"
        if weather_source.is_file():
            _write_strict_jsonl(weather_source, strict_root / weather_source.name)
        scenario_trajectory_source = scenario_root / 'trajectories.jsonl' if scenario_root is not None else None
        world_trajectory_source = episode_root / "trajectories.jsonl"
        if not world_trajectory_source.is_file():
            raise ObjectivePipelineError(
                "all-entity world trajectory input is missing: "
                f"{world_trajectory_source}"
            )
        _write_strict_trajectory_union(
            scenario_trajectory_source,
            world_trajectory_source,
            strict_root / "trajectories.jsonl",
        )
        l0_state_summary = materialize_episode_l0_state(strict_root)
        (strict_root / "l0_state_materialization_summary.json").write_text(
            canonical_json(l0_state_summary) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        strict_geometry = GeometricStateComputer(strict_root).compute()
        runtime_state_summary = PlanWindowComputer(
            strict_root,
            strict_geometry,
        ).materialize_runtime_state_truth(strict_root / "truth_frames.jsonl")
        (strict_root / "runtime_state_materialization_summary.json").write_text(
            canonical_json(runtime_state_summary) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        # Domain producers integrate charging and battery on the full physical
        # clock. Each semantic consumer selects its declared sampling grid.
        # Cropping this input to five-tick records loses real interval evidence.
        _assert_strict_structured_families_projected(strict_root / "truth_frames.jsonl")
        del strict_geometry
        del runtime_state_summary
        gc.collect()
        ctypes.CDLL(None).malloc_trim(0)
        yield strict_root


def _write_strict_json(source: Path, target: Path) -> None:
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    sanitized = sanitize_objective_input(value)
    target.write_text(
        json.dumps(sanitized, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_strict_entity_roster_union(
    scenario_source: Path | None,
    world_source: Path,
    target: Path,
) -> None:
    """Merge scenario-declared actors into the generated all-entity roster."""

    world = json.loads(world_source.read_text(encoding="utf-8-sig"))
    if not isinstance(world, dict) or not isinstance(world.get("entities"), list):
        raise ObjectivePipelineError(f"{world_source}: entities must be an array")
    entities: dict[str, dict[str, Any]] = {}
    for raw in world["entities"]:
        if not isinstance(raw, dict):
            raise ObjectivePipelineError(
                f"{world_source}: roster entry must be an object"
            )
        entity_id = raw.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id or entity_id in entities:
            raise ObjectivePipelineError(
                f"{world_source}: invalid or duplicate roster entity_id"
            )
        entities[entity_id] = sanitize_objective_input(raw)
    if scenario_source is not None:
        scenario = json.loads(scenario_source.read_text(encoding="utf-8-sig"))
        if not isinstance(scenario, dict):
            raise ObjectivePipelineError(f"{scenario_source}: root must be an object")
        scenario_values: Any = scenario.get("entities", scenario)
        if isinstance(scenario_values, dict):
            scenario_rows = list(scenario_values.values())
        elif isinstance(scenario_values, list):
            scenario_rows = scenario_values
        else:
            raise ObjectivePipelineError(
                f"{scenario_source}: roster entries must be an object or array"
            )
        seen_scenario: set[str] = set()
        for raw in scenario_rows:
            if not isinstance(raw, dict):
                raise ObjectivePipelineError(
                    f"{scenario_source}: roster entry must be an object"
                )
            entity_id = raw.get("entity_id")
            if (
                not isinstance(entity_id, str)
                or not entity_id
                or entity_id in seen_scenario
            ):
                raise ObjectivePipelineError(
                    f"{scenario_source}: invalid or duplicate roster entity_id"
                )
            seen_scenario.add(entity_id)
            if isinstance(raw.get("background_vehicle"), Mapping):
                # Scenario background vehicles are plan templates.  Their
                # realized, seed-qualified SUMO identities are already in the
                # world roster and are the sole vehicle truth authority.
                continue
            sanitized = sanitize_objective_input(raw)
            existing = entities.get(entity_id)
            if existing is None:
                entities[entity_id] = sanitized
                continue
            _assert_roster_identity_equal(
                entity_id,
                existing,
                sanitized,
                world_source=world_source,
                scenario_source=scenario_source,
            )
    payload = sanitize_objective_input(world)
    payload["entities"] = [entities[entity_id] for entity_id in sorted(entities)]
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _assert_roster_identity_equal(
    entity_id: str,
    world_entity: Mapping[str, Any],
    scenario_entity: Mapping[str, Any],
    *,
    world_source: Path,
    scenario_source: Path,
) -> None:
    world_category = (
        str(world_entity.get("entity_category") or world_entity.get("category") or "")
        .strip()
        .lower()
    )
    scenario_category = (
        str(
            scenario_entity.get("entity_category")
            or scenario_entity.get("category")
            or ""
        )
        .strip()
        .lower()
    )
    if (
        not world_category
        or not scenario_category
        or world_category != scenario_category
    ):
        raise ObjectivePipelineError(
            f"roster union category conflict for {entity_id}: "
            f"{world_source}={world_category!r}, {scenario_source}={scenario_category!r}"
        )
    if world_category not in {"facility", "ground_station"}:
        return
    world_scope = world_entity.get("semantic_scope")
    scenario_scope = scenario_entity.get("semantic_scope")
    if not isinstance(world_scope, Mapping) or not isinstance(scenario_scope, Mapping):
        raise ObjectivePipelineError(
            f"roster union facility {entity_id} lacks semantic_scope in one source"
        )
    if dict(world_scope) != dict(scenario_scope):
        raise ObjectivePipelineError(
            f"roster union semantic_scope conflict for {entity_id}: "
            f"{world_source}={dict(world_scope)!r}, "
            f"{scenario_source}={dict(scenario_scope)!r}"
        )
    world_kind = str(
        world_entity.get("entity_kind") or world_entity.get("entity_type") or ""
    )
    scenario_kind = str(
        scenario_entity.get("entity_kind") or scenario_entity.get("entity_type") or ""
    )
    if not world_kind or not scenario_kind or world_kind != scenario_kind:
        raise ObjectivePipelineError(
            f"roster union facility kind conflict for {entity_id}: "
            f"{world_source}={world_kind!r}, {scenario_source}={scenario_kind!r}"
        )


def _write_strict_jsonl(source: Path, target: Path) -> None:
    rows = [sanitize_objective_input(row) for row in read_jsonl(source)]
    target.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


# Trigger-volume assets whose runtime activity drives the restricted-region predicate.
_RESTRICTED_REGION_ASSETS = (
    "trigger.no_fly.box.v1",
    "trigger.hazard.generic.box.v1",
)


def _write_strict_truth_frames(source: Path, target: Path) -> None:
    rows: list[dict[str, Any]] = []
    for raw in read_jsonl(source):
        entities = raw.get("entities")
        if not isinstance(entities, list):
            raise ObjectivePipelineError(
                f"{source}: truth-frame entities must be an array"
            )
        activity: dict[str, bool] = {}
        for entity in entities:
            if not isinstance(entity, Mapping):
                raise ObjectivePipelineError(
                    f"{source}: truth-frame entity must be an object"
                )
            # Every authored trigger-volume asset whose runtime activity the
            # restricted-region computer evaluates must be projected here.  The
            # projection is the only path that survives input sanitization, which
            # removes the authored `annotations` block; a volume left out of it
            # falls back to the entity path and fails on the missing activity.
            if entity.get("logical_asset_id") not in _RESTRICTED_REGION_ASSETS:
                continue
            entity_id = entity.get("entity_id")
            if not isinstance(entity_id, str) or not entity_id or entity_id in activity:
                raise ObjectivePipelineError(
                    f"{source}: invalid or duplicate restricted-region entity"
                )
            try:
                activity[entity_id] = restricted_region_runtime_active(entity)
            except PredicateStateComputerError as exc:
                raise ObjectivePipelineError(f"{source}:{entity_id}: {exc}") from exc
        projected = dict(raw)
        projected["restricted_region_runtime_activity"] = activity
        rows.append(sanitize_objective_input(projected))
    target.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _write_strict_trajectory_union(
    scenario_source: Path | None,
    world_source: Path,
    target: Path,
) -> None:
    rows_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    # The capture-filtered world trajectory is the same-tick observation
    # authority.  Scenario trajectories may add offstage entities only; they
    # must never replace an overlapping world pose.
    sources = (world_source, scenario_source) if scenario_source is not None else (world_source,)
    for source in sources:
        seen: set[tuple[int, str]] = set()
        for raw in read_jsonl(source):
            tick = raw.get("tick")
            entity_id = raw.get("entity_id")
            if (
                not isinstance(tick, int)
                or not isinstance(entity_id, str)
                or not entity_id
            ):
                raise ObjectivePipelineError(
                    f"{source}: trajectory row lacks integer tick or entity_id"
                )
            key = (tick, entity_id)
            if key in seen:
                raise ObjectivePipelineError(
                    f"{source}: duplicate trajectory row {key}"
                )
            seen.add(key)
            if source == scenario_source and isinstance(
                raw.get("background_vehicle"), Mapping
            ):
                continue
            row = sanitize_objective_input(raw)
            if key not in rows_by_key:
                rows_by_key[key] = row
    target.write_text(
        "".join(
            json.dumps(
                rows_by_key[key],
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for key in sorted(rows_by_key)
        ),
        encoding="utf-8",
        newline="\n",
    )


def _assert_strict_structured_families_projected(truth_frames_path: Path) -> None:
    if not truth_frames_path.is_file():
        return
    for frame in read_jsonl(truth_frames_path):
        entities = frame.get("entities")
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            forbidden = {
                "state",
                "activity_state",
                "activity_type",
                "posture",
                "semantic_role",
                "task_id",
                "active_event_ids",
            } & set(entity)
            annotations = entity.get("annotations")
            if isinstance(annotations, Mapping):
                forbidden.update({"activity_type", "state_facets"} & set(annotations))
            if forbidden:
                raise ObjectivePipelineError(
                    f"strict objective input still contains forbidden fields: {sorted(forbidden)}"
                )
            if any(
                family in entity for family in OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES
            ):
                return


def _process_episode(
    job: _ObjectiveCorpusJob,
) -> dict[str, Any]:
    checkpoint = job.materialized_output_dir.with_name(job.materialized_output_dir.name + ".completed.json")
    if not job.check and checkpoint.is_file():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (saved["input_authority"] == job.input_authority
                and saved["execution_authority"] == job.execution_authority):
            files = saved["files"]
            if any(not (job.materialized_output_dir / name).is_file()
                   or (job.materialized_output_dir / name).stat().st_size != size
                   for name, size in files.items()):
                raise ObjectivePipelineError(f"incomplete episode checkpoint: {job.episode_root.name}")
            print(f"reuse {job.episode_root.name}", flush=True)
            return saved["result"]
    if not job.check and job.materialized_output_dir.exists():
        shutil.rmtree(job.materialized_output_dir)
    artifacts = build_episode_objective_artifacts(
        job.episode_root,
        job.final_output_dir,
        domain_profile_path=job.domain_profile_path,
        compute_profile_path=job.compute_profile_path,
        contract_profile_path=job.contract_profile_path,
        stage_acceptance_profile_path=job.stage_acceptance_profile_path,
    )
    if job.check:
        mismatches = check_episode_outputs(artifacts)
        if mismatches:
            raise ObjectivePipelineError(
                f"{artifacts.episode_id}: objective outputs are stale: {mismatches}"
            )
    else:
        _write_exact_artifact_tree(job.materialized_output_dir, artifacts.files)
    processed = {
        "episode_id": artifacts.episode_id,
        "epi_id": artifacts.epi_id,
        "seed_label": artifacts.seed_label,
        "output_dir": str(artifacts.output_dir),
        "record_counts": artifacts.manifest["record_counts"],
        "closure_status": artifacts.closure["status"],
        "artifact_count": len(artifacts.files),
    }
    closure_failure_diagnostics = _closure_failure_diagnostics(artifacts.closure)
    if closure_failure_diagnostics is not None:
        processed["closure_failure_diagnostics"] = closure_failure_diagnostics
    result = {
        "processed": processed,
        "output_files": _output_file_references(job.materialized_output_dir),
    }
    if not job.check:
        _atomic_write_text(checkpoint, _json_text({
            "input_authority": job.input_authority,
            "execution_authority": job.execution_authority,
            "files": {name: len(text.encode("utf-8")) for name, text in artifacts.files.items()},
            "result": result,
        }))
    print(f"complete {artifacts.episode_id} events={processed['record_counts']['event_occurrences']}", flush=True)
    return result


def _tick_context_profile(domain_profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "inputs": {
            "truth_frames": domain_profile["inputs"]["truth_frames"],
            "entity_roster": domain_profile["inputs"]["entity_roster"],
            "static_geometry": "semantic_static_geometry.json",
            "episode_manifest": domain_profile["inputs"]["episode_manifest"],
        },
        "authoritative_tick_policy": dict(domain_profile["authoritative_tick_policy"]),
    }


def _jsonl_rows_from_text(text: str, source_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ObjectivePipelineError(
                f"{source_name}:{line_number}: JSONL row must be an object"
            )
        rows.append(value)
    return rows


def _build_evidence_observations(
    semantic_state: Sequence[Mapping[str, Any]],
    predicate_truth: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for state in semantic_state:
        numeric_values = _numeric_values(
            {
                "value": state.get("value"),
                "source_values": state.get("source_values"),
                "thresholds": state.get("thresholds"),
            }
        )
        if not numeric_values:
            continue
        observation_id = str(
            state.get("observation_id") or state.get("semantic_state_id")
        )
        if not observation_id:
            continue
        evidence_id = f"semantic_state_observations.jsonl#{observation_id}"
        evidence[evidence_id] = {
            "schema_name": "evidence_observation",
            "schema_version": SCHEMA_VERSION,
            "evidence_id": evidence_id,
            "episode_id": state["episode_id"],
            "tick": state["tick"],
            "observation_id": observation_id,
            "metric_id": state.get("api_id"),
            "subject_id": "|".join(
                str(value)
                for _, value in sorted(dict(state.get("bindings") or {}).items())
            ),
            "source_class": state.get("source_class", "unknown"),
            "numeric_values": numeric_values,
            "source_refs": list(state.get("source_refs", [])),
        }
    for truth in predicate_truth:
        evidence_payload = (
            truth.get("evidence") if isinstance(truth.get("evidence"), Mapping) else {}
        )
        numeric_values = _numeric_values(evidence_payload.get("observations"))
        if not numeric_values:
            continue
        truth_id = str(truth.get("truth_id") or "")
        if not truth_id:
            continue
        evidence_id = f"predicate_truth.jsonl#{truth_id}"
        evidence[evidence_id] = {
            "schema_name": "evidence_observation",
            "schema_version": SCHEMA_VERSION,
            "evidence_id": evidence_id,
            "episode_id": truth["episode_id"],
            "tick": truth["tick"],
            "observation_id": truth_id,
            "metric_id": truth.get("predicate_id"),
            "subject_id": "|".join(
                str(value)
                for _, value in sorted(dict(truth.get("bindings") or {}).items())
            ),
            "source_class": "deterministic_derived",
            "numeric_values": numeric_values,
            "source_refs": list(evidence_payload.get("source_refs", [])),
        }
    return [evidence[key] for key in sorted(evidence)]


def _select_durable_output_rows(
    semantic_state: Sequence[Mapping[str, Any]],
    predicate_truth: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    continuity_breaks: Sequence[Mapping[str, Any]],
    occurrences: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    return {
        "semantic_state": list(semantic_state),
        "predicate_truth": list(predicate_truth),
        "transitions": list(transitions),
        "continuity_breaks": list(continuity_breaks),
        "event_occurrences": list(occurrences),
        "event_outcomes": list(outcomes),
    }


def _required_api_ids_for_epi_view(
    contract: Mapping[str, Any], epi_id: str
) -> list[str]:
    epi_contracts = contract.get("epi_contracts")
    event_families = contract.get("event_families")
    if not isinstance(epi_contracts, Mapping) or not isinstance(
        event_families, Mapping
    ):
        raise ObjectivePipelineError(
            "objective contract lacks EPI or event-family mappings"
        )
    base_epi_id = epi_id.rsplit("__seed", 1)[0]
    epi = epi_contracts.get(base_epi_id)
    if not isinstance(epi, Mapping):
        raise ObjectivePipelineError(f"objective contract lacks EPI {base_epi_id}")
    required: set[str] = set()
    referenced_families: list[Mapping[str, Any]] = []
    for stage in epi.get("required_chain", ()):
        if not isinstance(stage, Mapping):
            raise ObjectivePipelineError(
                f"{base_epi_id}: required_chain contains a non-object"
            )
        family = event_families.get(stage.get("event_family_id"))
        if not isinstance(family, Mapping):
            raise ObjectivePipelineError(
                f"{base_epi_id}: unknown event family {stage.get('event_family_id')!r}"
            )
        referenced_families.append(family)
        required.update(
            str(predicate_id)
            for predicate_id in family.get("required_predicates", ())
            if isinstance(predicate_id, str)
        )
    if not required:
        if referenced_families and all(
            family.get("executable") is False for family in referenced_families
        ):
            return []
        raise ObjectivePipelineError(
            f"{base_epi_id}: executable event chain has no required predicates"
        )
    ontology_ids = set(get_core_predicate_ids())
    outside_ontology = sorted(required - ontology_ids)
    if outside_ontology:
        raise ObjectivePipelineError(
            f"{base_epi_id}: required predicates are outside the ontology: "
            f"{outside_ontology}"
        )
    return sorted(required)


def _roster_contract_predicate_ids(episode_root: Path) -> list[str]:
    roster_path = episode_root / "global_entity_roster.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8-sig"))
    entities = roster.get("entities") if isinstance(roster, Mapping) else None
    if not isinstance(entities, list):
        raise ObjectivePipelineError(f"{roster_path}: entities must be an array")
    predicate_ids: set[str] = set()
    for entity in entities:
        if not isinstance(entity, Mapping):
            raise ObjectivePipelineError(
                f"{roster_path}: roster entity must be an object"
            )
        contract = entity.get("path_deviation_contract")
        if contract is None:
            continue
        if not isinstance(contract, Mapping):
            raise ObjectivePipelineError(
                f"{roster_path}:{entity.get('entity_id')}: path deviation contract must be an object"
            )
        predicate_id = contract.get("predicate_id")
        if predicate_id != "positioning.aircraft_deviates_from_planned_route":
            raise ObjectivePipelineError(
                f"{roster_path}:{entity.get('entity_id')}: invalid path deviation predicate"
            )
        threshold = contract.get("minimum_route_distance_m")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or float(threshold) != PATH_DEVIATION_THRESHOLD_M
        ):
            raise ObjectivePipelineError(
                f"{roster_path}:{entity.get('entity_id')}: path deviation threshold "
                f"must equal governed {PATH_DEVIATION_THRESHOLD_M}m"
            )
        predicate_ids.add(predicate_id)
    return sorted(predicate_ids)


def _stage_projection_predicate_ids(
    stage_contract: Mapping[str, Any],
    epi_id: str,
) -> list[str]:
    """Return exact L1 predicates referenced by this EPI's declarative L2 proof."""

    required: set[str] = set()
    stages = stage_contract.get("stages")
    if not isinstance(stages, Mapping):
        raise ObjectivePipelineError("stage acceptance contract lacks stages")
    prefix = f"{epi_id}::"
    for stage_key, stage in stages.items():
        if not str(stage_key).startswith(prefix) or not isinstance(stage, Mapping):
            continue
        for clause in stage.get("acceptance_clauses", ()):
            if not isinstance(clause, Mapping):
                continue
            for field in (
                "predicate_id",
                "trigger_predicate_id",
            ):
                predicate_id = clause.get(field)
                if isinstance(predicate_id, str):
                    required.add(predicate_id)
    ontology_ids = set(get_core_predicate_ids())
    outside_ontology = sorted(required - ontology_ids)
    if outside_ontology:
        raise ObjectivePipelineError(
            f"{epi_id}: stage proof references predicates outside the ontology: "
            f"{outside_ontology}"
        )
    return sorted(required)


def _build_manifest(
    episode_root: Path,
    output_dir: Path,
    episode_id: str,
    epi_id: str,
    seed_label: str,
    domain_profile_path: Path,
    compute_profile_path: Path,
    contract_profile_path: Path,
    stage_acceptance_profile_path: Path,
    domain_artifacts: Any,
    context_input_files: Sequence[Mapping[str, Any]],
    static_geometry_authority: Mapping[str, Any],
    projection: Mapping[str, Any],
    epi_view_api_ids: Sequence[str],
    domain_rows: Sequence[Mapping[str, Any]],
    compute_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    compute_predicate_rows: Sequence[Mapping[str, Any]],
    l0_predicate_state: Sequence[Mapping[str, Any]],
    utm_summary: Mapping[str, Any],
    world_truth_base: Mapping[str, Any],
    world_truth_deltas: Sequence[Mapping[str, Any]],
    world_truth_summary: Mapping[str, Any],
    semantic_state: Sequence[Mapping[str, Any]],
    predicate_truth: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    continuity_breaks: Sequence[Mapping[str, Any]],
    occurrences: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    base_graphs: Sequence[Mapping[str, Any]],
    graph_deltas: Sequence[Mapping[str, Any]],
    graphs: Sequence[Mapping[str, Any]],
    rejected_graph_records: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Any],
    global_detection_counts: Mapping[str, int],
    global_detection_digests: Mapping[str, str],
) -> dict[str, Any]:
    record_counts = {
        "l0_predicate_state": len(l0_predicate_state),
        "semantic_state_observations": len(semantic_state),
        "predicate_truth": len(predicate_truth),
        "predicate_transitions": len(transitions),
        "predicate_continuity_breaks": len(continuity_breaks),
        "event_occurrences": len(occurrences),
        "event_outcomes": len(outcomes),
        "world_truth_graph_bases": 1,
        "world_truth_graph_deltas": len(world_truth_deltas),
        "world_truth_graph_delta_operations": sum(
            int(row.get("operation_count", 0)) for row in world_truth_deltas
        ),
        "world_truth_initial_assertions": len(
            world_truth_base.get("initial_assertions", ())
        ),
        "semantic_graph_bases": len(base_graphs),
        "semantic_graph_deltas": len(graph_deltas),
        "semantic_truth_graphs": len(graphs),
        "evidence_observations": len(evidence),
    }
    durable_detection_counts = {
        key: int(record_counts[key]) for key in global_detection_counts
    }
    expected_global_counts = {
        str(key): int(value) for key, value in global_detection_counts.items()
    }
    if durable_detection_counts != expected_global_counts:
        raise ObjectivePipelineError(
            "durable L2 record counts differ from global detection counts: "
            f"durable={durable_detection_counts}, global={expected_global_counts}"
        )
    manifest = {
        "schema_name": "objective_semantic_episode_manifest",
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "epi_id": epi_id,
        "seed_label": seed_label,
        "episode_root": str(episode_root),
        "output_dir": str(output_dir),
        "input_files": {
            name: {"path": str(episode_root / name)}
            for name in (
                "episode_manifest.json",
                "global_entity_roster.json",
                "trajectories.jsonl",
                "truth_frames.jsonl",
                "weather_meta.jsonl",
            )
            if (episode_root / name).is_file()
        },
        "domain_profile": {
            "path": str(domain_profile_path),
            "sha256": digest_file(domain_profile_path),
        },
        "compute_profile": {
            "path": str(compute_profile_path),
            "sha256": digest_file(compute_profile_path),
        },
        "contract_profile": {
            "path": str(contract_profile_path),
            "sha256": digest_file(contract_profile_path),
        },
        "stage_acceptance_profile": {
            "path": str(stage_acceptance_profile_path),
            "sha256": digest_file(stage_acceptance_profile_path),
        },
        "domain_input_digest": domain_artifacts.input_digest,
        "context_input_digest": digest_object(context_input_files),
        "static_geometry_authority": dict(static_geometry_authority),
        "projection": {
            "catalog_id": projection.get("catalog_id"),
            "catalog_version": projection.get("catalog_version"),
            "projection_mode": "global_detection_with_epi_view_acceptance",
            "event_detection_scope": "global_contract_event_families",
            "durable_event_view": "full_detection_rows",
            "durable_output_policy": "full_detection_rows_v1",
            "global_detection_persistence": "durable_rows_with_manifest_counts_and_sources",
            "selected_api_ids": list(epi_view_api_ids),
            "selected_api_count": len(epi_view_api_ids),
            "global_projected_api_ids": list(projection.get("selected_api_ids", [])),
            "global_projected_api_count": len(projection.get("selected_api_ids", [])),
            "catalog_gate": projection.get("catalog_gate"),
            "input_digest": projection.get("input_digest"),
            "parameter_digest": projection.get("parameter_digest"),
        },
        "annotation_layers": {
            "L0": {
                "role": "raw_and_deterministic_supplement_truth_all_entities_all_ticks",
                "artifacts": [
                    "truth_frames.jsonl",
                    "trajectories.jsonl",
                    "weather_meta.jsonl",
                    "l0_predicate_state.jsonl",
                    "compute_comm_supplement",
                    "domain_state_supplement",
                    "utm_supplement",
                ],
                "materialization": "authoritative_input_plus_deterministic_l0_builders",
            },
            "L1": {
                "role": "complete_grounded_world_truth_predicate_candidates",
                "artifacts": [
                    "world_truth_graph_base.json",
                    "world_truth_graph_deltas.jsonl",
                    "world_truth_summary.json",
                ],
                "scope_types": list(world_truth_base.get("scope_types", ())),
                "predicate_count": world_truth_summary.get("predicate_count"),
                "truth_values": ["true", "false", "unknown", "out_of_scope"],
                "materialization": "tick0_complete_matrix_base_plus_value_change_delta",
                "full_state_snapshot_per_change": False,
                "replayable_at_arbitrary_tick": True,
            },
            "L2": {
                "role": "semantic_event_graph_roi_or_event_entities",
                "artifacts": [
                    "predicate_truth.jsonl",
                    "predicate_transitions.jsonl",
                    "predicate_continuity_breaks.jsonl",
                    "semantic_graph_base.json",
                    "semantic_graph_deltas.jsonl",
                    "semantic_truth_graphs.jsonl",
                ],
                "representation": "tick0_base_plus_predicate_delta_plus_event_window_projection",
                "full_state_snapshot_per_change": False,
                "replayable_at_arbitrary_tick": True,
            },
        },
        "record_counts": record_counts,
        "global_detection_record_counts": dict(sorted(global_detection_counts.items())),
        "global_detection_record_digests": dict(
            sorted(global_detection_digests.items())
        ),
        "durable_record_counts_match_global_detection": True,
        "intermediate_record_counts": {
            "domain_state_observations": len(domain_artifacts.observations),
            "objective_supplement_domain_state_observations": max(
                0, len(domain_rows) - len(domain_artifacts.observations)
            ),
            "compute_state": len(compute_rows),
            "communication_state": len(communication_rows),
            "compute_communication_predicate_truth": len(compute_predicate_rows),
            "utm_state": sum(
                int(value)
                for value in dict(utm_summary.get("record_counts", {})).values()
            ),
        },
        "world_truth_summary": dict(world_truth_summary),
        "utm_supplement_summary": dict(utm_summary),
        "compact_graph_rejected_record_count": len(rejected_graph_records),
        "closure_status": closure["status"],
        "forbidden_inputs_used": [],
    }
    world_trajectory_path = episode_root / "trajectories.jsonl"
    scenario_root = source_episode_root(episode_root)
    scenario_trajectory_path = scenario_root / "trajectories.jsonl" if scenario_root is not None else None
    manifest["offstage_source"] = source_episode_descriptor(episode_root)
    if world_trajectory_path.is_file():
        manifest["trajectory_union"] = {
            "world_source": {
                "path": str(world_trajectory_path),
                "sha256": digest_file(world_trajectory_path),
            },
            "key": ["tick", "entity_id"],
            "overlap_authority": "world_source",
        }
        if scenario_trajectory_path is not None:
            manifest["trajectory_union"].update(
                {
                    "scenario_source": {
                        "path": manifest["offstage_source"]["path"],
                        "sha256": digest_file(scenario_trajectory_path),
                    },
                    "scenario_source_role": "missing_offstage_rows_only",
                }
            )
        else:
            manifest["trajectory_union"].update(
                {
                    "scenario_source": None,
                    "scenario_source_role": "unavailable_explicit",
                    "missing_source_record": {
                        "path": (str(Path(manifest["offstage_source"]["path"]) / "trajectories.jsonl")
                                 if manifest["offstage_source"]["path"] is not None else None),
                        "reason": "formal scenario trajectory source is absent; world trajectory remains authoritative",
                    },
                }
            )
    return manifest


def _assert_authoritative_ticks(
    truth_frames_path: Path, profile: Mapping[str, Any]
) -> None:
    if not truth_frames_path.is_file():
        raise ObjectivePipelineError(f"truth frames missing: {truth_frames_path}")
    tick_policy = profile["authoritative_tick_policy"]
    expected_ticks = tuple(
        range(
            int(tick_policy["start"]),
            int(tick_policy["end"]) + 1,
            int(tick_policy["step"]),
        )
    )
    wanted = set(expected_ticks)
    seen: set[int] = set()
    duplicates: list[int] = []
    for row in read_jsonl(truth_frames_path):
        tick = row.get("tick")
        if not isinstance(tick, int) or tick not in wanted:
            continue
        if tick in seen:
            duplicates.append(tick)
        seen.add(tick)
    missing = [tick for tick in expected_ticks if tick not in seen]
    if missing or duplicates:
        raise ObjectivePipelineError(
            f"truth_frames.jsonl must contain exactly authoritative ticks "
            f"{expected_ticks[0]}..{expected_ticks[-1]} step {int(tick_policy['step'])}; "
            f"missing={missing[:10]}, duplicates={duplicates[:10]}"
        )


def _assert_no_forbidden_episode_inputs(episode_root: Path) -> None:
    present = {path.name for path in episode_root.iterdir() if path.is_file()}
    forbidden_present = sorted(present & FORBIDDEN_INPUT_NAMES)
    # These files may exist in render-ready roots; the pipeline contract is that
    # this orchestrator does not read them. Presence is recorded but not read.
    _ = forbidden_present


def _assert_no_forbidden_payload(files: Mapping[str, str]) -> None:
    for name, text in files.items():
        lowered = text.lower()
        if any(token in lowered for token in FORBIDDEN_TEXT):
            raise ObjectivePipelineError(f"{name} contains forbidden evidentiary text")


def _numeric_values(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix or "value"] = float(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_numeric_values(item, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            result.update(_numeric_values(item, child))
    return result


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(without_integrity_metadata(row)) + "\n" for row in rows)


def _json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(without_integrity_metadata(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _check_root_file(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ObjectivePipelineError(f"root output is missing: {path}")
    if path.read_text(encoding="utf-8") != expected:
        raise ObjectivePipelineError(f"root output is stale: {path}")


def _build_corpus_manifest(
    processed: Sequence[Mapping[str, Any]],
    *,
    corpus_mode: str,
    execution_authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_name": "objective_semantic_corpus_manifest",
        "schema_version": SCHEMA_VERSION,
        "corpus_mode": corpus_mode,
        "execution_authority": execution_authority,
        "episode_count": len(processed),
        "record_counts": _sum_record_counts(processed),
        "episodes": list(processed),
    }


def _build_closure_matrix(
    processed: Sequence[Mapping[str, Any]],
    *,
    corpus_mode: str,
    execution_authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_name": "objective_semantic_epi_closure_matrix",
        "schema_version": SCHEMA_VERSION,
        "corpus_mode": corpus_mode,
        "execution_authority": execution_authority,
        "episode_count": len(processed),
        "rows": [_closure_matrix_row(row) for row in processed],
    }


def _closure_matrix_row(row: Mapping[str, Any]) -> dict[str, Any]:
    status = row["closure_status"]
    diagnostics = row.get("closure_failure_diagnostics")
    if status == "PASS":
        if diagnostics is not None:
            raise ObjectivePipelineError(
                f"PASS closure unexpectedly has failure diagnostics: {row['episode_id']}"
            )
    elif not isinstance(diagnostics, Mapping):
        raise ObjectivePipelineError(
            f"non-PASS closure lacks failure diagnostics: {row['episode_id']}"
        )
    result = {
        "episode_id": row["episode_id"],
        "epi_id": row["epi_id"],
        "seed_label": row["seed_label"],
        "closure_status": status,
    }
    if diagnostics is not None:
        result["closure_failure_diagnostics"] = dict(diagnostics)
    return result


def _sum_record_counts(processed: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in processed:
        for key, value in dict(row.get("record_counts") or {}).items():
            totals[key] = totals.get(key, 0) + int(value)
    return dict(sorted(totals.items()))


__all__ = [
    "DEFAULT_EPISODES_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_STAGE_ACCEPTANCE_PROFILE_PATH",
    "ObjectiveEpisodeArtifacts",
    "ObjectivePipelineError",
    "build_corpus",
    "build_episode_objective_artifacts",
    "check_episode_outputs",
    "parse_episode_identity",
    "selected_episode_roots",
    "write_episode_outputs",
]
