"""Enumerate source leaf paths and their observed value types without inventing units."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from Dataset.world_model.graph.contract import (
    CONTRACT_VERSION, REPO, _declared_script_paths, declared_samples, iter_jsonl, read_json, source_path,
)
from Dataset.world_model.graph.entity_identity import IdentityResolver, read_roster
from Dataset.world_model.graph.controller_scope import controller_evidence, controller_namespaces_for_record
from Dataset.world_model.graph.structure_schema import is_publication_excluded_path
from Dataset.world_model.graph.field_contract import (
    DEFAULT_DEFINITIONS, EXACT_MISSING_RULES, build_definitions, classify_catalog,
    projection_owners, unit_policy, write_definitions,
)
from Dataset.world_model.graph.field_paths import (
    ENTITY_KEYED_OBJECTS, SNAPSHOT_DICTIONARIES, FACILITY_SUBTYPE_KEY,
    WORLD_OBSERVATION_VALUE_PATH,
    catalog_source_branch, catalog_source_subtype,
    manifest_source_ref_dimension,
    source_branch_for_record, source_field_id, source_subtype_for_record,
    source_value_at_tokens, world_observation_numeric_contracts,
    world_observation_path_key, world_observation_variant_field_path,
)
from Dataset.world_model.graph.source_projection import project_values, source_value_type as _value_type
from Dataset.world_model.graph.field_structure import (
    extra_structure_bindings, iter_structure_bindings,
    IMPLEMENTATION_SOURCES_GAP_SOURCE_ID,
    is_declared_implementation_sources_gap,
)
from Dataset.world_model.graph.structure_reuse import rules_for_source
from Dataset.world_model.graph.sample_identity import (
    arm_chain_row_family, inventory_scope, manifest_reference, validate_record_identity,
)
from Dataset.tools.l4_incident_observations import PROFILES as LOCAL_BUSINESS_PROFILES


_WORLD_OBSERVATION_COUNT_PENDING = "pending_structure_reprojection"
_WORLD_OBSERVATION_VARIANT_BASIS = (
    "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json:"
    "templates[].state_contract.required_fields + "
    "Dataset/tools/arm_window_semantics.py:predicate_rows.evidence.world_observations"
)


def _array_prefixes(tokens: tuple[str | int, ...], field_path: str) -> list[tuple[str | int, ...]]:
    prefixes = [tokens[:index] for index, token in enumerate(tokens) if type(token) is int]
    if len(prefixes) != field_path.count("[]"):
        raise ValueError(f"source array positions differ: {tokens}:{field_path}")
    return prefixes


def _owner(source_kind: str, path: str) -> str:
    if source_kind.startswith("arm_chain_nodes:") or source_kind.startswith("arm_chain_events:"):
        row_kind = source_kind.split(":", 1)[1]
        return "ScriptEvent" if row_kind == "script_emission" else "ContextAnchor" if row_kind == "measured_geometry_support" else "source_only"
    if source_kind in {"arm_chain_relations", "formal_event_chain_relations"}:
        return "edge_attribute"
    if source_kind in {"arm_manifest", "formal_episode_manifest", "formal_objective_manifest",
                       "arm_initial_events", "formal_source_scene_setup",
                       "script_category_source"}:
        return "source_only"
    if source_kind == "arm_semantic_history":
        if path.startswith("objective.predicate_truth[]."):
            return "PredicateAssertion"
        if path.startswith("objective.predicate_transitions[]."):
            return "PredicateTransition"
        return "source_only"
    if source_kind == "formal_weather_meta":
        return "source_only" if path == "tick" else "ContextAnchor"
    if source_kind in {"formal_roster", "arm_branch_roster", "arm_scene_setup"}:
        return "EntityIdentity"
    if source_kind in {"arm_script_plan", "formal_source_script"}:
        if path.startswith("events[].actions[]."):
            return "ActionPlan"
        return "ScriptEvent" if path.startswith("events[].") else (
            "source_only" if source_kind == "formal_source_script" else "ContextAnchor")
    if source_kind == "formal_world_truth_graph_base":
        if path.startswith("initial_assertions[]"):
            return "PredicateAssertion"
        if path.startswith("scope_entities[]"):
            return "EntityIdentity"
        return "ContextAnchor"
    if source_kind == "formal_world_truth_graph_deltas":
        return "source_only" if path.startswith("operations[]") else "TimePoint" if path == "tick" else "source_only"
    if source_kind in {"formal_predicate_truth", "predicate_truth"}:
        return "PredicateAssertion"
    if source_kind in {"formal_predicate_transitions", "arm_predicate_transitions"}:
        return "PredicateTransition"
    if source_kind in {"formal_predicate_continuity_breaks"}:
        return "ContextAnchor"
    if source_kind in {"formal_event_occurrences", "events", "arm_event_occurrences"}:
        return "EventPhase" if path.startswith("event_phases[]") else "EventOccurrence"
    if source_kind in {"formal_event_outcomes", "arm_event_outcomes"}:
        return "EventOutcome"
    if source_kind == "formal_event_realization":
        return "ScriptEvent"
    if source_kind == "formal_truth_frame":
        if path.startswith("entities[]."):
            return "EntityState"
        if path.startswith("sumo_traffic_light_states.{entity}."):
            return "ContextAnchor"
        return "TimePoint" if path == "tick" or path == "entity_motion_state" or path.startswith("entity_motion_state.") else "source_only"
    if source_kind in {"arm_states", "arm_trajectories"}:
        return "EntityState"
    if source_kind == "arm_actions":
        return "ContextAnchor"
    if source_kind == "arm_predicate_truth":
        return "PredicateAssertion"
    if source_kind == "arm_predicate_truth_ticks":
        return "PredicateAssertion"
    if source_kind == "arm_domain_state_ticks":
        return "ContextAnchor"
    if source_kind == "arm_local_business_truth":
        return "ContextAnchor"
    if source_kind == "predicate_truth_matrix":
        return "ContextAnchor"
    if source_kind in {"domain_state", "compute_state", "communication_state"}:
        return "ContextAnchor"
    raise ValueError(f"field source family has no owner: {source_kind}:{path}")


def _time_granularity(source_kind: str, field_path: str) -> str:
    if source_kind == "arm_predicate_transitions" and field_path in {"from_tick", "to_tick"}:
        return "transition_from_endpoint" if field_path == "from_tick" else "transition_to_endpoint"
    if source_kind == "arm_predicate_truth_ticks" and field_path == "metrics.approach_dispatch_tick":
        return "action_dispatch_tick"
    if source_kind == "arm_local_business_truth":
        if field_path in {
            "metrics.arrival_dispatch_tick", "metrics.handoff_dispatch_tick",
            "metrics.landing_dispatch_tick", "metrics.dispatch_action_tick",
        }:
            return "action_dispatch_tick"
        if field_path.startswith("metrics.destinations_enu_m.") or field_path in {
            "metrics.distance_limit_m", "metrics.horizontal_limit_m",
            "metrics.vertical_limit_m", "metrics.handoff_delay_ticks",
            "metrics.required_dwell_ticks",
        }:
            return "planned_parameter"
    if source_kind == "formal_roster" and field_path in {
        "initial_position_enu_m[]", "initial_yaw_deg",
    }:
        return "source_dependent_initial_pose"
    if source_kind in {"formal_roster", "arm_branch_roster", "arm_scene_setup", "arm_script_plan", "arm_manifest"}:
        return "static_or_plan"
    if source_kind == "formal_world_truth_graph_base":
        if field_path.startswith("scope_entities[]"):
            return "episode_scope_static"
        if field_path.startswith("initial_assertions[]"):
            return "base_truth_tick"
        if field_path == "tick":
            return "base_tick"
        return "source_metadata"
    if source_kind in {"formal_truth_frame", "arm_states", "arm_trajectories", "domain_state",
                       "compute_state", "communication_state", "arm_domain_state_ticks", "formal_weather_meta"}:
        return "per_source_tick"
    if source_kind in {"formal_predicate_truth", "predicate_truth", "arm_predicate_truth"}:
        return "sampled_truth_tick"
    if source_kind == "arm_predicate_truth_ticks":
        return "dense_truth_tick"
    if source_kind.startswith("arm_chain_") or source_kind.endswith("event_occurrences") or source_kind.endswith("event_outcomes"):
        return "event_or_interval"
    return "source_row_time"


@dataclass(frozen=True)
class SourceDeclaration:
    source_family: str
    path: Path
    sample: str
    structure_status: str = "field_scan_defined"
    read_relation: str = "direct_read"
    read_condition: str | None = None
    reference_samples: tuple[str, ...] = ()

    @property
    def presence(self) -> str:
        return "present" if self.path.is_file() else "missing"

    @property
    def identity_status(self) -> str:
        return ("not_applicable" if self.source_family in {
            "formal_source_script", "formal_source_scene_setup",
            "script_category_source", "arm_scene_setup"} else "sample")


def declared_source_inventory() -> Iterable[SourceDeclaration]:
    """List expected source paths and their structural read roles, including absent files.

    A pending structure has a known source path, not a declared field schema.
    A manifest reference is read directly only when its read condition holds.
    """
    formal, arm = declared_samples()
    script_references: dict[Path, set[str]] = {}
    for episode_id in formal:
        episode = REPO / "aw_data/render_ready_episodes_capture_filtered" / episode_id
        episode_manifest = episode / "episode_manifest.json"
        yield SourceDeclaration("formal_episode_manifest", episode_manifest,
                                episode_id, "field_scan_defined")
        manifest = read_json(episode_manifest)
        script_references.setdefault(REPO / manifest["source_event_script_path"], set()).add(
            "formal:" + episode_id)
        yield SourceDeclaration("formal_source_script", REPO / manifest["source_event_script_path"],
                                episode_id, "field_scan_defined", "manifest_reference",
                                "capture_screenshot_action_realization",
                                ("formal:" + episode_id,))
        yield SourceDeclaration("formal_source_scene_setup", REPO / manifest["source_scene_setup_path"],
                                episode_id, "field_scan_defined", "manifest_reference",
                                "capture_screenshot_action_realization",
                                ("formal:" + episode_id,))
        for name, kind in (("global_entity_roster.json", "formal_roster"),
                           ("truth_frames.jsonl", "formal_truth_frame"),
                           ("event_realization.jsonl", "formal_event_realization")):
            yield SourceDeclaration(kind, episode / name, episode_id)
        objective = REPO / "aw_data/objective_semantic_truth" / episode_id
        yield SourceDeclaration("formal_objective_manifest", objective / "manifest.json",
                                episode_id, "field_scan_defined")
        for name in ("world_truth_graph_base.json", "world_truth_graph_deltas.jsonl",
                     "predicate_truth.jsonl", "predicate_transitions.jsonl",
                     "predicate_continuity_breaks.jsonl", "event_occurrences.jsonl", "event_outcomes.jsonl"):
            path = objective / name
            yield SourceDeclaration("formal_" + path.stem, path, episode_id)
        for name, family in (("domain_state_observations.jsonl", "domain_state"),
                             ("compute_state.jsonl", "compute_state"),
                             ("communication_state.jsonl", "communication_state")):
            yield SourceDeclaration(family, objective / name, episode_id)
        for name, family in (("compute_predicate_truth.jsonl", "predicate_truth"),
                             ("compute_predicate_matrix.jsonl", "predicate_truth_matrix"),
                             ("compute_events.jsonl", "events")):
            yield SourceDeclaration(family, objective / name, episode_id)
        yield SourceDeclaration("formal_weather_meta", episode / "weather_meta.jsonl", episode_id)
        yield SourceDeclaration("formal_event_chain_relations",
                                REPO / "aw_data/event_chains" / episode_id / "event_chain_relations.jsonl",
                                episode_id)

    scene_references: dict[Path, set[str]] = {}
    for branch in arm:
        sample = source_path(branch)
        sample_key = "arm:" + branch.relative_to(REPO / "arms").as_posix()
        script_references.setdefault(branch / "ue/event_script.json", set()).add(sample_key)
        manifest_path = branch / "manifest.json"
        yield SourceDeclaration("arm_manifest", manifest_path, sample)
        for folder, names in (("raw", ("states.jsonl", "trajectories.jsonl", "actions.jsonl")),
                              ("objective", ("predicate_truth.jsonl", "predicate_truth_ticks.jsonl",
                                             "predicate_transitions.jsonl", "local_business_truth.jsonl",
                                             "event_occurrences.jsonl", "event_outcomes.jsonl"))):
            for name in names:
                path = branch / folder / name
                yield SourceDeclaration("arm_" + name.removesuffix(".jsonl"), path, sample)
        for chain_name, kind in (("nodes.jsonl", "arm_chain_nodes"),
                                 ("events.jsonl", "arm_chain_events"),
                                 ("relations.jsonl", "arm_chain_relations")):
            yield SourceDeclaration(kind, branch / "chain" / chain_name, sample)
        yield SourceDeclaration("arm_branch_roster", branch / "ue/global_entity_roster.json", sample)
        yield SourceDeclaration("arm_script_plan", branch / "ue/event_script.json", sample)
        manifest = read_json(manifest_path)
        script_ref = manifest_reference(manifest, "script")
        if script_ref is not None:
            script_references.setdefault(REPO / script_ref, set()).add(sample_key)
        scene_ref = manifest_reference(manifest, "scene")
        if scene_ref is not None:
            scene_path = REPO / scene_ref
            scene_references.setdefault(scene_path, set()).add(sample_key)
        producer = _arm_producer_family(branch.relative_to(REPO / "arms").parts[0])
        if producer in {"L3", "L6"}:
            yield SourceDeclaration("arm_initial_events", branch / "objective/initial_events.json",
                                    sample)
        if producer == "L6":
            history_ref = manifest_reference(manifest, "semantic_history")
            if history_ref is not None:
                yield SourceDeclaration("arm_semantic_history", REPO / history_ref,
                                        sample, "field_scan_defined")

    for scene_path, sample_keys in sorted(scene_references.items()):
        yield SourceDeclaration("arm_scene_setup", scene_path, source_path(scene_path),
                                reference_samples=tuple(sorted(sample_keys)))

    for script_path in _declared_script_paths(arm):
        yield SourceDeclaration("script_category_source", script_path, source_path(script_path),
                                "field_scan_defined",
                                reference_samples=tuple(sorted(script_references.get(script_path, ()))))


def _sources() -> Iterable[tuple[str, Path, str]]:
    """Scan only present files with a defined field traversal."""
    for declaration in declared_source_inventory():
        if declaration.structure_status != "field_scan_defined" or declaration.presence != "present":
            continue
        kind = declaration.source_family
        if kind == "arm_predicate_truth_ticks":
            first = next(iter_jsonl(declaration.path), None)
            if first is not None and "predicate_id" not in first[1]:
                kind = "arm_domain_state_ticks"
        yield kind, declaration.path, declaration.sample


def _arm_producer_family(profile: str) -> str:
    level = re.fullmatch(r"L([1-6])-[0-9]+_v[0-9]+", profile)
    if level is not None:
        return "L" + level.group(1)
    if re.fullmatch(r"X[1-6]_[a-z0-9_]+", profile):
        return "X"
    raise ValueError(f"unknown ARM producer profile: {profile}")


def _arm_source_applicable(declaration: SourceDeclaration) -> bool:
    profile = declaration.path.relative_to(REPO / "arms").parts[0]
    producer = _arm_producer_family(profile)
    if declaration.source_family == "arm_chain_events":
        return producer in {"L1", "L2", "L4", "L5"}
    if declaration.source_family == "arm_chain_nodes":
        return producer in {"L3", "L6", "X"}
    if declaration.source_family == "arm_local_business_truth":
        return profile in LOCAL_BUSINESS_PROFILES
    raise ValueError(f"no ARM applicability rule: {declaration.source_family}")


def _source_rebuild_preflight(families: set[str] | None = None) -> None:
    """Refuse a rebuild that would erase required or zero-instance sources."""
    declarations = list(declared_source_inventory())
    issues: dict[tuple[str, str], list[str]] = defaultdict(list)
    conditional = {"arm_chain_events", "arm_chain_nodes", "arm_local_business_truth"}
    for declaration in declarations:
        if families is not None and declaration.source_family not in families:
            continue
        applicable = (_arm_source_applicable(declaration)
                      if declaration.source_family in conditional else True)
        if declaration.structure_status != "field_scan_defined":
            reason = "structure_pending"
        elif declaration.presence != "present":
            if not applicable:
                continue
            reason = "source_missing"
        elif declaration.path.stat().st_size == 0:
            reason = "zero_instance_without_complete_source_schema"
        else:
            continue
        issues[(reason, declaration.source_family)].append(source_path(declaration.path))

    initial_selected = families is None or "arm_initial_events" in families
    history_selected = families is None or "arm_semantic_history" in families
    if initial_selected or history_selected:
        declared = {(item.source_family, item.path) for item in declarations}
        for declaration in declarations:
            if declaration.source_family != "arm_manifest":
                continue
            branch = declaration.path.parent
            producer = _arm_producer_family(branch.relative_to(REPO / "arms").parts[0])
            if initial_selected and producer in {"L3", "L6"}:
                path = branch / "objective/initial_events.json"
                if ("arm_initial_events", path) not in declared:
                    issues[("source_declaration_missing", "arm_initial_events")].append(source_path(path))
            if history_selected and producer == "L6":
                reference = manifest_reference(read_json(declaration.path), "semantic_history")
                if reference is None:
                    issues[("source_reference_missing", "arm_semantic_history")].append(
                        source_path(declaration.path))
                elif ("arm_semantic_history", REPO / reference) not in declared:
                    issues[("source_declaration_missing", "arm_semantic_history")].append(reference)
    if issues:
        details = "; ".join(
            f"{reason}:{family}={len(paths)} [{', '.join(paths[:2])}]"
            for (reason, family), paths in sorted(issues.items()))
        raise ValueError("source structures are incomplete before catalog write: " + details)


def _read_current_catalog(output: Path) -> list[dict[str, Any]]:
    if not output.is_file():
        raise FileNotFoundError(output)
    rows = [row for _, row in iter_jsonl(output)]
    if (not rows or any(row.get("contract_version") != CONTRACT_VERSION for row in rows)
            or len({row.get("id") for row in rows}) != len(rows)):
        raise ValueError(f"catalog is not a unique current-version source: {output}")
    validate_normalizations(rows)
    return rows


def _observe_normalization(row: dict[str, Any], projected: Any) -> None:
    key = projected.normalized_id, projected.binding_slot
    bindings = row["normalizations"]
    found = bindings.get(key)
    if found is None:
        found = {"normalized_id": projected.normalized_id,
                 "binding_slot": projected.binding_slot,
                 "declared_shape": projected.declared_shape,
                 "observed_source_value_types": set(),
                 "observed_value_types": set(),
                 "observed_value_statuses": set(),
                 "observed_structure_statuses": set(),
                 "observed_shapes": set(), "instance_count": 0}
        bindings[key] = found
    elif found["declared_shape"] != projected.declared_shape:
        raise ValueError(f"structural shape changed within source field: {row['id']}:{key}")
    for source in projected.sources:
        if source.source_field_id == row["id"]:
            found["observed_source_value_types"].add(_value_type(source.field_value.value))
    if projected.value_type is not None:
        found["observed_value_types"].add(projected.value_type)
    found["observed_value_statuses"].add(projected.value_status)
    found["observed_structure_statuses"].add(projected.structure_status)
    if projected.observed_shape is not None:
        found["observed_shapes"].add(projected.observed_shape)
    found["instance_count"] += 1


@lru_cache(maxsize=1)
def _structure_bindings_by_source() -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for binding in iter_structure_bindings():
        grouped[binding.source_field_id].append(binding)
    for source_id, bindings in grouped.items():
        keys = [(binding.normalized_id, binding.binding_slot) for binding in bindings]
        if len(keys) != len(set(keys)):
            raise ValueError(f"source structure has duplicate binding slots: {source_id}")
    return {source_id: tuple(bindings) for source_id, bindings in grouped.items()}


def _finished_normalizations(row: dict[str, Any]) -> list[dict[str, Any]]:
    observed = row.pop("normalizations")
    declared = {}
    for binding in _structure_bindings_by_source().get(row["id"], ()):
        key = binding.normalized_id, binding.binding_slot
        declared[key] = {"normalized_id": binding.normalized_id,
                         "binding_slot": binding.binding_slot,
                         "selector": binding.selector,
                         "value_types": list(binding.value_types),
                         "value_statuses": list(binding.value_statuses),
                         "structure_statuses": list(binding.structure_statuses),
                         "declared_shape": list(binding.declared_shape),
                         "source_value_types": list(binding.source_value_types),
                         "declaration_basis": binding.declaration_basis,
                         "declaration_status": "structure_declared"}
    for binding in extra_structure_bindings(row["id"]):
        key = binding["normalized_id"], binding["binding_slot"]
        if key in declared:
            raise ValueError(f"source normalization declarations overlap: {row['id']}:{key}")
        declared[key] = {**binding, "declaration_status": "structure_declared"}
    for rule in rules_for_source(row["id"]):
        if rule.structure_mode != "source_only":
            continue
        key = rule.normalized_id, rule.target_binding_slot
        if key in declared:
            raise ValueError(f"source reuse declarations overlap: {row['id']}:{key}")
        declared[key] = {
            "normalized_id": rule.normalized_id,
            "binding_slot": rule.target_binding_slot,
            "selector": "reuse:" + rule.condition,
            "source_value_types": rule.source_value_types,
            "value_types": rule.value_types,
            "value_statuses": rule.value_statuses,
            "structure_statuses": ["source_only"],
            "declared_shape": None,
            "declaration_basis": rule.producer_basis,
            "declaration_status": "structure_declared",
        }
    keys = set(observed) | set(declared)
    finished = []
    for key in sorted(keys, key=lambda item: (item[0], item[1] or "")):
        actual = observed.get(key)
        rule = declared.get(key)
        if actual is None:
            actual = {"normalized_id": key[0], "binding_slot": key[1],
                      "declared_shape": (tuple(rule["declared_shape"])
                                         if rule is not None and rule["declared_shape"] is not None
                                         else None),
                      "observed_source_value_types": set(),
                      "observed_value_types": set(), "observed_value_statuses": set(),
                      "observed_structure_statuses": set(), "observed_shapes": set(),
                      "instance_count": 0}
        shape = list(actual["declared_shape"]) if actual["declared_shape"] is not None else None
        if rule is not None and shape != rule["declared_shape"]:
            raise ValueError(f"source structural declaration differs from observation: {row['id']}:{key}")
        types = (set(rule["value_types"]) if rule is not None and
                 rule["value_types"] is not None else
                 None if rule is not None else set(actual["observed_value_types"]))
        source_types = (set(rule["source_value_types"]) if rule is not None and
                        rule["source_value_types"] is not None else
                        None if rule is not None else set(actual["observed_source_value_types"]))
        statuses = (set(rule["value_statuses"]) if rule is not None and
                    rule["value_statuses"] is not None else
                    None if rule is not None else set(actual["observed_value_statuses"]))
        structures = (set(rule["structure_statuses"]) if rule is not None and
                      rule["structure_statuses"] is not None else
                      None if rule is not None else set(actual["observed_structure_statuses"]))
        if (source_types is not None and not actual["observed_source_value_types"] <= source_types or
                types is not None and not actual["observed_value_types"] <= types or
                statuses is not None and not actual["observed_value_statuses"] <= statuses or
                structures is not None and not actual["observed_structure_statuses"] <= structures):
            raise ValueError(f"source structural observation exceeds declaration: {row['id']}:{key}")
        if (row["source_family"], row["field_path"]) in EXACT_MISSING_RULES and \
                "absent" in EXACT_MISSING_RULES[(row["source_family"], row["field_path"])] and \
                statuses is not None:
            statuses.add("absent")
        finished.append({"normalized_id": key[0], "binding_slot": key[1],
                         "selector": rule["selector"] if rule is not None else
                             "sample_identity" if key[1] == "sample_identity" else
                             "observed_arm_pose" if key[1] == "observed_pose" else
                             "source_leaf",
                         "source_value_types": sorted(source_types) if source_types is not None else None,
                         "observed_source_value_types": sorted(actual["observed_source_value_types"]),
                         "value_types": sorted(types) if types is not None else None,
                         "value_statuses": sorted(statuses) if statuses is not None else None,
                         "structure_statuses": sorted(structures) if structures is not None else None,
                         "declared_shape": shape,
                         "observed_value_types": sorted(actual["observed_value_types"]),
                         "observed_value_statuses": sorted(actual["observed_value_statuses"]),
                         "observed_structure_statuses": sorted(actual["observed_structure_statuses"]),
                         "observed_shapes": [list(item) for item in sorted(
                             actual["observed_shapes"], key=lambda item: json.dumps(item))],
                         "instance_count": actual["instance_count"],
                         "declaration_status": rule["declaration_status"] if rule is not None else
                             "observed_only",
                         "declaration_basis": rule.get("declaration_basis") if rule is not None else None})
    if not finished:
        raise ValueError(f"source field has no structural binding: {row['id']}")
    return finished


def _scan_group(group: list[tuple[str, Path, str]]) -> dict[tuple[str, str, str | None, str | None], dict[str, Any]]:
    aggregates: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = {}
    for kind, path, sample in group:
        scope = inventory_scope(kind, path, sample)
        resolver = _source_identity_resolver(kind, path, sample) if kind in ENTITY_MERGE_BASIS else None
        if kind in {"formal_roster", "arm_branch_roster", "arm_scene_setup"}:
            source_record = read_roster(path)
            records = ((None, row, ("entities", index), source_record)
                       for index, row in enumerate(source_record["entities"]))
        elif path.suffix == ".json":
            source_record = read_json(path)
            records = ((None, source_record, (), source_record),)
        else:
            records = ((line, row, (), row) for line, row in iter_jsonl(path))
        for line_no, record, prefix_tokens, source_record in records:
            if kind in {"arm_predicate_truth_ticks", "arm_domain_state_ticks"}:
                is_predicate = "predicate_id" in record
                if is_predicate != (kind == "arm_predicate_truth_ticks"):
                    raise ValueError(f"mixed predicate/state rows in {path}:{line_no}")
            effective_kind = (arm_chain_row_family(kind, record)
                              if kind in {"arm_chain_nodes", "arm_chain_events"} else kind)
            source_ref_mode = manifest_source_ref_dimension(record) if kind == "arm_manifest" else None
            namespaces = _controller_namespaces(kind, path, record)
            for projected, source in ((projected, source)
                                      for projected in project_values(
                                          record, source_family=effective_kind,
                                          source_prefix_tokens=prefix_tokens,
                                          entity_key_namespaces=namespaces,
                                          scope=scope, source_file=source_path(path),
                                          source_line=line_no)
                                      for source in projected.sources):
                leaf = source.field_value
                field_path, value = leaf.field_path, leaf.value
                source_value = source_value_at_tokens(source_record, leaf.source_tokens)
                if type(source_value) is not type(value) or source_value != value:
                    raise ValueError(f"source leaf location differs: {path}:{line_no}:{leaf.raw_path}")
                if not field_path:
                    continue
                key = effective_kind, field_path, leaf.source_branch, leaf.source_subtype
                row = aggregates.get(key)
                if row is None:
                    row = {"kind": "state_field", "id": source_field_id(*key),
                           "source_family": effective_kind, "field_path": field_path,
                           "owner": _owner(effective_kind, field_path), "value_types": set(),
                           "source_scope_status": "sample" if scope is not None else "not_applicable",
                           "source_files": set(), "instance_count": 0,
                           "normalizations": {},
                           "payload_ranks": set(), "envelope_array_ranks": set(),
                           "first_source_file": source_path(path), "first_source_line": line_no,
                           "time_granularity": _time_granularity(effective_kind, field_path)}
                    if leaf.source_branch is not None:
                        row["source_branch"] = {"observation_family": leaf.source_branch}
                        if leaf.source_subtype is not None:
                            row["source_branch"][FACILITY_SUBTYPE_KEY] = leaf.source_subtype
                    if kind == "arm_manifest" and field_path.startswith("source_refs."):
                        row["source_ref_key_mode"] = ("source_path" if source_ref_mode == "source_path"
                                                      else "role")
                    if kind == "formal_world_truth_graph_deltas" and field_path.startswith("operations[]"):
                        row["projection_targets"] = ["PredicateAssertion", "ContextAnchor"]
                    if "{entity}" in field_path:
                        row.update(entity_identities=set(), unresolved_value_count=0,
                                   unresolved_reasons={}, entity_merge_basis=ENTITY_MERGE_BASIS[kind])
                        if field_path.endswith("[]"):
                            if kind == "formal_truth_frame" and field_path.startswith("sumo_traffic_light_states.{entity}."):
                                row["array_axis_lengths"] = [set() for _ in range(field_path.count("[]"))]
                            else:
                                row["array_lengths"] = set()
                    aggregates[key] = row
                _observe_normalization(row, projected)
                row["value_types"].add(_value_type(value))
                row["payload_ranks"].add(len(projected.payload_shape))
                row["envelope_array_ranks"].add(sum(type(token) is int
                                                     for token in projected.envelope_tokens))
                row["source_files"].add(source_path(path))
                row["instance_count"] += 1
                if "entity_identities" in row:
                    assert resolver is not None
                    if "array_lengths" in row:
                        array_tokens = leaf.source_tokens[:-1]
                        source_array = source_value_at_tokens(record, array_tokens)
                        if not isinstance(source_array, list):
                            raise ValueError(f"entity array source is not a list: {path}:{line_no}:{array_tokens}")
                        row["array_lengths"].add(len(source_array))
                    if "array_axis_lengths" in row:
                        for axis, array_tokens in enumerate(_array_prefixes(leaf.source_tokens, field_path)):
                            source_array = source_value_at_tokens(record, array_tokens)
                            if not isinstance(source_array, list):
                                raise ValueError(f"SUMO controller axis is not a list: {path}:{line_no}:{array_tokens}")
                            row["array_axis_lengths"][axis].add(len(source_array))
                    resolved = resolver.resolve(leaf.entity_id, source_file=source_path(path),
                        source_line=line_no, source_path=leaf.raw_path,
                        namespace=leaf.entity_namespace or "sample",
                        source_status=leaf.identity_status, source_reason=leaf.identity_reason)
                    if resolved.status == "resolved":
                        row["entity_identities"].add((resolved.sample_key, resolved.namespace,
                                                       resolved.lifecycle_id, resolved.raw_id))
                    else:
                        row["unresolved_value_count"] += 1
                        reason = resolved.reason or resolved.status
                        reasons = row["unresolved_reasons"]
                        reasons[reason] = reasons.get(reason, 0) + 1
    return aggregates


def _merge_group(aggregates: dict[tuple[str, str, str | None, str | None], dict[str, Any]],
                 found: dict[tuple[str, str, str | None, str | None], dict[str, Any]]) -> None:
    for key, row in found.items():
        old = aggregates.get(key)
        if old is None:
            aggregates[key] = {**row, "sample_count": 1}
            continue
        if old["owner"] != row["owner"]:
            raise ValueError(f"field owner changed across sources: {key}")
        if old.get("source_scope_status") != row.get("source_scope_status"):
            raise ValueError(f"field identity projection changed across sources: {key}")
        if old.get("source_ref_key_mode") != row.get("source_ref_key_mode"):
            raise ValueError(f"ARM source reference key mode changed across sources: {key}")
        old["value_types"].update(row["value_types"])
        for binding_key, incoming in row["normalizations"].items():
            current = old["normalizations"].get(binding_key)
            if current is None:
                old["normalizations"][binding_key] = incoming
                continue
            if current["declared_shape"] != incoming["declared_shape"]:
                raise ValueError(f"field normalized shape changed across sources: {key}:{binding_key}")
            for name in ("observed_source_value_types", "observed_value_types", "observed_value_statuses",
                         "observed_structure_statuses", "observed_shapes"):
                current[name].update(incoming[name])
            current["instance_count"] += incoming["instance_count"]
        old["payload_ranks"].update(row["payload_ranks"])
        old["envelope_array_ranks"].update(row["envelope_array_ranks"])
        old["source_files"].update(row["source_files"])
        old["sample_count"] += 1
        old["instance_count"] += row["instance_count"]
        if "entity_identities" in row:
            old["entity_identities"].update(row["entity_identities"])
            old["unresolved_value_count"] += row["unresolved_value_count"]
            if "array_lengths" in row:
                old.setdefault("array_lengths", set()).update(row["array_lengths"])
            if "array_axis_lengths" in row:
                axes = old.setdefault("array_axis_lengths", [set() for _ in row["array_axis_lengths"]])
                if len(axes) != len(row["array_axis_lengths"]):
                    raise ValueError(f"SUMO controller array rank differs: {key}")
                for target_axis, source_axis in zip(axes, row["array_axis_lengths"]):
                    target_axis.update(source_axis)
            for reason, count in row["unresolved_reasons"].items():
                old["unresolved_reasons"][reason] = old["unresolved_reasons"].get(reason, 0) + count
        first = (old["first_source_file"], old["first_source_line"] or 0)
        incoming = (row["first_source_file"], row["first_source_line"] or 0)
        if incoming < first:
            old["first_source_file"] = row["first_source_file"]
            old["first_source_line"] = row["first_source_line"]


def _finished_row(aggregate: dict[str, Any]) -> dict[str, Any]:
    row = {**aggregate}
    row["value_types"] = sorted(row["value_types"])
    row["normalizations"] = _finished_normalizations(row)
    row["observed_payload_ranks"] = sorted(row.pop("payload_ranks"))
    row["observed_envelope_array_ranks"] = sorted(row.pop("envelope_array_ranks"))
    nonnull = set(row["value_types"]) - {"null"}
    if not nonnull:
        row["value_kind"] = "null"
    elif nonnull <= {"int", "float"}:
        row["value_kind"] = "numeric"
    elif nonnull == {"bool"}:
        row["value_kind"] = "boolean"
    elif nonnull == {"string"}:
        row["value_kind"] = "categorical"
    elif nonnull <= {"empty_list", "empty_object"}:
        row["value_kind"] = "container"
    else:
        row["value_kind"] = "mixed"
    row.update(unit_policy(row["source_family"], row["field_path"], row["value_types"]))
    row["source_file_count"] = len(row.pop("source_files"))
    if "entity_identities" in row:
        row["entity_count"] = len(row.pop("entity_identities"))
        row["unresolved_reasons"] = dict(sorted(row["unresolved_reasons"].items()))
        if "array_lengths" in row:
            lengths = row.pop("array_lengths")
            if len(lengths) != 1:
                if not (row["source_family"] == "formal_truth_frame" and
                        row["field_path"].startswith("sumo_traffic_light_states.{entity}.")):
                    raise ValueError(f"entity field has conflicting array lengths: {row['id']}: {sorted(lengths)}")
                row["shape_length"] = None
                row["shape_status"] = "variable_observed"
                row["observed_array_lengths"] = sorted(lengths)
            else:
                row["shape_length"] = next(iter(lengths))
                row["shape_status"] = "fixed_observed"
        if "array_axis_lengths" in row:
            axes = [sorted(lengths) for lengths in row.pop("array_axis_lengths")]
            if any(not lengths for lengths in axes):
                raise ValueError(f"SUMO controller array axis lacks observation: {row['id']}")
            row["observed_axis_lengths"] = axes
            row["shape_length"] = None
            row["shape_status"] = ("variable_observed" if any(len(lengths) > 1 for lengths in axes)
                                   else "fixed_observed")
    row["contract_version"] = CONTRACT_VERSION
    return row


def _world_observation_variant_binding(
        normalized_id: str, path_key: str | None,
        observed: dict[str, Any] | None = None, *, pending: bool = False,
        ) -> dict[str, Any]:
    """Declare one tagged-union branch without inventing observation counts."""
    numeric = path_key is not None
    allowed_types = ["float", "int"] if numeric else ["bool", "string"]
    if observed is None:
        observed_source_types: list[str] = []
        observed_types: list[str] = []
        observed_statuses: list[str] = []
        observed_structures: list[str] = []
        observed_shapes: list[list[int | None]] = []
        normalized_shapes: list[list[int | None]] = []
        instance_count: int | None = None if pending else 0
    else:
        observed_source_types = sorted(observed.get("observed_source_value_types", ()))
        observed_types = sorted(observed.get("observed_value_types", ()))
        observed_statuses = sorted(observed.get("observed_value_statuses", ()))
        observed_structures = sorted(observed.get("observed_structure_statuses", ()))
        observed_shapes = deepcopy(observed.get("observed_shapes", ()))
        normalized_shapes = deepcopy(observed.get("normalized_shapes", ()))
        instance_count = observed.get("instance_count")
    if (not set(observed_source_types) <= set(allowed_types) or
            not set(observed_types) <= set(allowed_types)):
        raise ValueError(
            f"world-observation branch has incompatible observed types: "
            f"{path_key}:{observed_source_types}:{observed_types}")
    if pending:
        if instance_count is not None:
            raise ValueError("pending world-observation branch has an observed count")
    elif type(instance_count) is not int or instance_count < 0:
        raise ValueError("observed world-observation branch lacks an exact count")
    selector = ("world_observation.path=" + path_key if path_key is not None
                else "world_observation.nonnumeric_value")
    result = {
        "normalized_id": normalized_id, "binding_slot": None,
        "selector": selector,
        "declaration_status": "contract_declared_variant",
        "declaration_basis": _WORLD_OBSERVATION_VARIANT_BASIS,
        "declared_shape": None,
        "source_value_types": allowed_types, "value_types": allowed_types,
        "value_statuses": ["present"], "structure_statuses": ["source_only"],
        "observed_source_value_types": observed_source_types,
        "observed_value_types": observed_types,
        "observed_value_statuses": observed_statuses,
        "observed_structure_statuses": observed_structures,
        "observed_shapes": observed_shapes,
        "normalized_shapes": normalized_shapes,
        "instance_count": instance_count,
        "record_family": "arm_predicate_truth",
    }
    if pending:
        result["instance_count_status"] = _WORLD_OBSERVATION_COUNT_PENDING
    return result


def _world_observation_variant_row(
        carrier: dict[str, Any], path_key: str | None,
        observed_binding: dict[str, Any] | None, *, pending: bool,
        ) -> dict[str, Any]:
    field_path = (WORLD_OBSERVATION_VALUE_PATH if path_key is None else
                  world_observation_variant_field_path(
                      "arm_predicate_truth", WORLD_OBSERVATION_VALUE_PATH, path_key))
    source_id = source_field_id("arm_predicate_truth", field_path)
    normalized_id = "structural:source:" + source_id
    row = deepcopy(carrier)
    row.update({
        "id": source_id, "field_path": field_path,
        "value_types": ["bool", "string"] if path_key is None else ["float", "int"],
        "observed_value_types": (
            sorted(observed_binding.get("observed_source_value_types", ()))
            if observed_binding is not None else []),
        "value_kind": "mixed" if path_key is None else "numeric",
        "instance_count": None if pending else (
            observed_binding["instance_count"] if observed_binding is not None else 0),
        "normalizations": [_world_observation_variant_binding(
            normalized_id, path_key, observed_binding, pending=pending)],
    })
    for name in (
            "projection_owners", "unit", "source_unit", "canonical_unit", "unit_status",
            "unit_basis", "quantity_role", "time_role", "coordinate_frame",
            "coordinate_status", "coordinate_contract_id", "coordinate_axes",
            "coordinate_basis", "source_time_role", "semantic_id", "definition_status",
            "meaning_status", "array_rank", "contract_version", "index", "semantic_index"):
        row.pop(name, None)
    row.pop("instance_count_status", None)
    if pending:
        row["instance_count_status"] = _WORLD_OBSERVATION_COUNT_PENDING
    if path_key is None:
        row.pop("source_path_key", None)
        row.pop("predicate_ids", None)
    else:
        contract = world_observation_numeric_contracts()[path_key]
        row["source_path_key"] = path_key
        row["predicate_ids"] = list(contract["predicate_ids"])
    row["structure_source_field_id"] = source_field_id(
        "arm_predicate_truth", WORLD_OBSERVATION_VALUE_PATH)
    row["structure_declaration_basis"] = sorted(set(
        row.get("structure_declaration_basis", ())) | {_WORLD_OBSERVATION_VARIANT_BASIS})
    return row


def _validate_expanded_world_observation_rows(
        carrier: dict[str, Any], variants: dict[str, dict[str, Any]]) -> None:
    contracts = world_observation_numeric_contracts()
    if set(variants) != set(contracts):
        raise ValueError(
            "world-observation catalog branch roster differs: "
            f"missing={sorted(set(contracts) - set(variants))}, "
            f"extra={sorted(set(variants) - set(contracts))}")
    expected_carrier_id = source_field_id(
        "arm_predicate_truth", WORLD_OBSERVATION_VALUE_PATH)
    if (carrier["id"] != expected_carrier_id or
            carrier.get("value_types") != ["bool", "string"] or
            carrier.get("source_path_key") is not None or
            carrier.get("predicate_ids") is not None):
        raise ValueError("world-observation nonnumeric carrier differs")
    for path_key, row in variants.items():
        field_path = world_observation_variant_field_path(
            "arm_predicate_truth", WORLD_OBSERVATION_VALUE_PATH, path_key)
        expected_id = source_field_id("arm_predicate_truth", field_path)
        expected_normalized = "structural:source:" + expected_id
        bindings = row.get("normalizations")
        if (row.get("id") != expected_id or row.get("field_path") != field_path or
                row.get("source_path_key") != path_key or
                row.get("predicate_ids") != list(contracts[path_key]["predicate_ids"]) or
                row.get("value_types") != ["float", "int"] or
                not isinstance(bindings, list) or len(bindings) != 1 or
                bindings[0].get("normalized_id") != expected_normalized):
            raise ValueError(f"world-observation catalog branch differs: {path_key}")


def _expand_world_observation_variants(rows: list[dict[str, Any]]) -> None:
    """Split the raw tagged union into contract-qualified model fields."""
    carrier_id = source_field_id("arm_predicate_truth", WORLD_OBSERVATION_VALUE_PATH)
    carriers = [row for row in rows if row.get("id") == carrier_id]
    if len(carriers) != 1:
        raise ValueError("field catalog lacks one world-observation value carrier")
    carrier = carriers[0]
    variants: dict[str, dict[str, Any]] = {}
    for row in rows:
        path_key = world_observation_path_key(
            row.get("source_family", ""), row.get("field_path", ""))
        if path_key is not None:
            if path_key in variants:
                raise ValueError(f"duplicate world-observation catalog branch: {path_key}")
            variants[path_key] = row
    if variants:
        contracts = world_observation_numeric_contracts()
        extra = set(variants) - set(contracts)
        if extra:
            raise ValueError(f"unknown world-observation catalog branches: {sorted(extra)}")
        for path_key in sorted(set(contracts) - set(variants)):
            row = _world_observation_variant_row(carrier, path_key, None, pending=True)
            rows.append(row)
            variants[path_key] = row
        _validate_expanded_world_observation_rows(carrier, variants)
        return

    bindings = carrier.get("normalizations")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("world-observation carrier has no structural bindings")
    expected_ids = {
        path_key: "structural:source:" + source_field_id(
            "arm_predicate_truth", world_observation_variant_field_path(
                "arm_predicate_truth", WORLD_OBSERVATION_VALUE_PATH, path_key))
        for path_key in world_observation_numeric_contracts()
    }
    by_normalized_id = {binding.get("normalized_id"): binding for binding in bindings}
    if len(by_normalized_id) != len(bindings):
        raise ValueError("world-observation carrier has duplicate structural bindings")
    base_normalized_id = "structural:source:" + carrier_id
    unknown = set(by_normalized_id) - {base_normalized_id, *expected_ids.values()}
    if unknown or base_normalized_id not in by_normalized_id:
        raise ValueError(f"world-observation carrier has unexpected structural bindings: {sorted(unknown)}")
    base_binding = by_normalized_id[base_normalized_id]
    old_unsplit = bool(set(base_binding.get("observed_source_value_types", ())) & {"int", "float"})
    if old_unsplit and len(bindings) != 1:
        raise ValueError("world-observation carrier mixes legacy and qualified bindings")

    base_observed = base_binding
    if old_unsplit:
        base_observed = deepcopy(base_binding)
        base_observed["observed_source_value_types"] = sorted(
            set(base_binding.get("observed_source_value_types", ())) - {"int", "float"})
        base_observed["observed_value_types"] = sorted(
            set(base_binding.get("observed_value_types", ())) - {"int", "float"})
        base_observed["instance_count"] = None
    replacement = [_world_observation_variant_row(
        carrier, None, base_observed, pending=old_unsplit)]
    for path_key, normalized_id in sorted(expected_ids.items()):
        observed = None if old_unsplit else by_normalized_id.get(normalized_id)
        replacement.append(_world_observation_variant_row(
            carrier, path_key, observed, pending=old_unsplit))
    rows[:] = [row for row in rows if row is not carrier] + replacement


def _pending_world_observation_counts(row: dict[str, Any]) -> bool:
    status = row.get("instance_count_status")
    if status is None:
        return False
    if status != _WORLD_OBSERVATION_COUNT_PENDING:
        raise ValueError(f"unknown field instance count status: {row['id']}:{status}")
    path = row.get("field_path")
    if (row.get("source_family") != "arm_predicate_truth" or
            path != WORLD_OBSERVATION_VALUE_PATH and
            world_observation_path_key("arm_predicate_truth", path) is None):
        raise ValueError(f"pending instance count is inapplicable: {row['id']}")
    return True


def _normalized_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["kind"] == "rule_required_field":
            if row.get("normalizations"):
                raise ValueError(f"rule input is not a source projection: {row['id']}")
            continue
        if row["kind"] != "state_field":
            raise ValueError(f"unknown field catalog row kind: {row['id']}")
        bindings = row.get("normalizations")
        if not isinstance(bindings, list) or not bindings:
            raise ValueError(f"source field has no structural bindings: {row['id']}")
        pending_counts = _pending_world_observation_counts(row)
        if ((pending_counts and (row.get("instance_count") is not None or
                                 any(binding.get("instance_count") is not None or
                                     binding.get("instance_count_status") !=
                                     _WORLD_OBSERVATION_COUNT_PENDING
                                     for binding in bindings))) or
                (not pending_counts and
                 sum(binding.get("instance_count", -1) for binding in bindings) !=
                 row.get("instance_count")) or
                {raw_type for binding in bindings
                 for raw_type in binding.get("observed_source_value_types", ())} !=
                set(row.get("observed_value_types", row["value_types"]))):
            raise ValueError(f"source instances or raw types are missing structural bindings: {row['id']}")
        seen_slots: set[tuple[str, str | None]] = set()
        for binding in bindings:
            normalized_id = binding.get("normalized_id")
            slot = binding.get("binding_slot")
            if (not isinstance(normalized_id, str) or not normalized_id.startswith("structural:")
                    or slot is not None and not isinstance(slot, str)
                    or (normalized_id, slot) in seen_slots):
                raise ValueError(f"source normalization key is invalid or duplicated: {row['id']}")
            seen_slots.add((normalized_id, slot))
            shape = binding.get("declared_shape")
            if (shape is not None and (not isinstance(shape, list) or
                    any(axis is not None and (type(axis) is not int or axis < 0) for axis in shape))):
                raise ValueError(f"normalized declaration has invalid shape: {row['id']}:{normalized_id}")
            optional_allowed = {"source_value_types", "value_types",
                                "value_statuses", "structure_statuses"}
            for name in ("source_value_types", "observed_source_value_types",
                         "value_types", "value_statuses", "structure_statuses",
                         "observed_value_types", "observed_value_statuses",
                         "observed_structure_statuses"):
                values = binding.get(name)
                if (values is None and name in optional_allowed and
                        binding.get("declaration_status") == "structure_declared" and
                        str(binding.get("selector", "")).startswith("reuse:")):
                    continue
                if (not isinstance(values, list) or
                        any(not isinstance(value, str) for value in values) or
                        values != sorted(set(values))):
                    raise ValueError(f"normalized {name} differs: {row['id']}:{normalized_id}")
            for observed_name, allowed_name in (
                    ("observed_source_value_types", "source_value_types"),
                    ("observed_value_types", "value_types"),
                    ("observed_value_statuses", "value_statuses"),
                    ("observed_structure_statuses", "structure_statuses")):
                if (binding[allowed_name] is not None and
                        not set(binding[observed_name]) <= set(binding[allowed_name])):
                    raise ValueError(f"normalized observation exceeds declaration: {row['id']}:{normalized_id}")
            if (not pending_counts and
                    (type(binding.get("instance_count")) is not int or
                     binding["instance_count"] < 0)):
                raise ValueError(f"normalized binding instance count differs: {row['id']}:{normalized_id}")
            observed_shapes = binding.get("observed_shapes")
            declared_gap = (row["id"] == IMPLEMENTATION_SOURCES_GAP_SOURCE_ID
                            and shape == [None]
                            and binding["observed_value_types"] == []
                            and binding["observed_source_value_types"] == ["empty_object"]
                            and binding["observed_value_statuses"] == ["empty_object"]
                            and binding["observed_structure_statuses"] == ["declared_only"]
                            and is_declared_implementation_sources_gap(
                                normalized_id, None, "empty_object", "declared_only"))
            if (not isinstance(observed_shapes, list) or
                    any(not isinstance(observed_shape, list) or
                        shape is not None and len(observed_shape) != len(shape) or
                        any(axis is not None and (type(axis) is not int or axis < 0)
                            for axis in observed_shape) or
                        shape is not None and any(
                            declared_axis is not None and observed_axis is not None and
                            declared_axis != observed_axis
                            for declared_axis, observed_axis in zip(shape, observed_shape))
                        for observed_shape in observed_shapes
                        if not (declared_gap and observed_shape is None))):
                raise ValueError(f"normalized observed shapes differ: {row['id']}:{normalized_id}")
            group = groups.get(normalized_id)
            if group is None:
                group = {"source_fields": set(), "owners": set(), "value_types": set(),
                         "value_statuses": set(), "structure_statuses": set(),
                         "open_allowed": set(), "declared_shape": shape}
                groups[normalized_id] = group
            elif group["declared_shape"] != shape:
                raise ValueError(f"normalized field has conflicting declared shapes: {normalized_id}")
            group["source_fields"].add(row["id"])
            group["owners"].update(projection_owners(row))
            for name in ("value_types", "value_statuses", "structure_statuses"):
                if binding[name] is None:
                    group["open_allowed"].add(name)
                else:
                    group[name].update(binding[name])
    for normalized_id, group in groups.items():
        if (len(group["owners"]) != 1
                and not normalized_id.startswith("structural:sample:")
                and not (group["owners"] == {"EventOutcome", "PredicateEpisodeOutcome"}
                         and all(source_id.startswith("arm_event_outcomes:")
                                 for source_id in group["source_fields"]))):
            raise ValueError(f"normalized field spans incompatible owners: {normalized_id}")
    return groups


def _normalized_metadata(normalized_id: str, index: int,
                         group: dict[str, Any]) -> dict[str, Any]:
    return {"normalized_index": index,
            "normalized_source_fields": sorted(group["source_fields"]),
            "normalized_owners": sorted(group["owners"]),
            "normalized_value_types": (None if "value_types" in group["open_allowed"]
                                       else sorted(group["value_types"])),
            "normalized_value_statuses": (None if "value_statuses" in group["open_allowed"]
                                          else sorted(group["value_statuses"])),
            "normalized_structure_statuses": (None if "structure_statuses" in group["open_allowed"]
                                              else sorted(group["structure_statuses"])),
            "normalized_declared_shape": group["declared_shape"]}


def validate_normalizations(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Verify frozen structural indices and recompute all shared memberships."""
    groups = _normalized_groups(rows)
    codes: dict[str, int] = {}
    for row in rows:
        for binding in row.get("normalizations", ()):
            normalized_id = binding["normalized_id"]
            index = binding.get("normalized_index")
            if type(index) is not int or index < 0:
                raise ValueError(f"normalized index is not an integer: {row['id']}:{normalized_id}")
            if normalized_id in codes and codes[normalized_id] != index:
                raise ValueError(f"normalized field has conflicting indices: {normalized_id}")
            codes[normalized_id] = index
    if (len(set(codes.values())) != len(codes) or
            set(codes.values()) != set(range(len(codes)))):
        raise ValueError("normalized indices are not unique and contiguous")
    for row in rows:
        for binding in row.get("normalizations", ()):
            normalized_id = binding["normalized_id"]
            expected = _normalized_metadata(normalized_id, codes[normalized_id], groups[normalized_id])
            if any(binding.get(name) != value for name, value in expected.items()):
                raise ValueError(f"normalized index or membership differs: {row['id']}:{normalized_id}")
    return codes


def _extend_index_codes(ids: Iterable[str], existing: dict[str, int],
                        label: str) -> dict[str, int]:
    """Append new declarations without renumbering existing model fields."""
    ids = set(ids)
    if (any(type(index) is not int or index < 0 for index in existing.values()) or
            len(set(existing.values())) != len(existing) or
            set(existing.values()) != set(range(len(existing)))):
        raise ValueError(f"{label} indices are not unique and contiguous")
    removed = set(existing) - ids
    if removed:
        raise ValueError(f"{label} declarations removed from the current index: {sorted(removed)}")
    codes = dict(existing)
    for key in sorted(ids - set(existing)):
        codes[key] = len(codes)
    return codes


def _write_compact_catalog(output: Path, rows: list[dict[str, Any]], *,
                           definition_indices: dict[str, int] | None = None,
                           rebuild_indices: bool = False) -> None:
    rows = [row for row in rows if row["id"] != "arm_chain_relations:causal_claim"]
    previous = _read_current_catalog(output) if output.is_file() and not rebuild_indices else []
    source_indices = {row["id"]: row["index"] for row in previous}
    normalized_indices = validate_normalizations(previous) if previous else {}
    if definition_indices is None:
        definition_indices = {}
        for row in previous:
            semantic_id, index = row["semantic_id"], row["semantic_index"]
            if semantic_id in definition_indices and definition_indices[semantic_id] != index:
                raise ValueError(f"semantic field has conflicting indices: {semantic_id}")
            definition_indices[semantic_id] = index
    _expand_world_observation_variants(rows)
    rows.sort(key=lambda row: row["id"])
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("field catalog has duplicate IDs")
    classify_catalog(rows, CONTRACT_VERSION)
    semantic_codes = _extend_index_codes(
        (row["semantic_id"] for row in rows), definition_indices, "semantic")
    for row in rows:
        row["semantic_index"] = semantic_codes[row["semantic_id"]]
    build_definitions(rows, CONTRACT_VERSION)
    groups = _normalized_groups(rows)
    codes = _extend_index_codes(groups, normalized_indices, "normalized")
    for row in rows:
        for binding in row.get("normalizations", ()):
            binding.update(_normalized_metadata(binding["normalized_id"],
                                                codes[binding["normalized_id"]],
                                                groups[binding["normalized_id"]]))
    validate_normalizations(rows)
    source_codes = _extend_index_codes((row["id"] for row in rows), source_indices, "source")
    for row in rows:
        row["contract_version"] = CONTRACT_VERSION
        row["index"] = source_codes[row["id"]]
        if "{entity}" in row["field_path"]:
            row["array_rank"] = row["field_path"].count("[]")
            row.setdefault("shape_status", "element_positions_only")
            row["shape_basis"] = "Dataset/world_model/graph/field_paths.py:structural_array_positions"
            row["entity_namespace"] = ("sumo_lane" if row["source_family"] == "domain_state"
                and row["field_path"].startswith("values.queue_sampling_window.stopped_vehicle_count_by_lane.{entity}")
                else "sumo_tls_controller" if row["source_family"] == "formal_truth_frame"
                and row["field_path"].startswith("sumo_traffic_light_states.{entity}.")
                else "source_scoped" if row["source_family"] == "domain_state"
                and row["field_path"].startswith("values.red_duration_ticks_by_controller.{entity}")
                else "sample")
            if row["entity_namespace"] == "source_scoped":
                row["entity_namespaces"] = ["sample", "sumo_tls_controller"]
            row["entity_count_scope"] = "graph_sample"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)
    if output.resolve() == (REPO / "design/belief/field_catalog.jsonl").resolve():
        write_definitions(DEFAULT_DEFINITIONS, rows, CONTRACT_VERSION)


def _check_snapshot_types(rows: list[dict[str, Any]]) -> None:
    prefixes = tuple(root + ".{tick}" for root in SNAPSHOT_DICTIONARIES)
    for row in rows:
        if (row["source_family"] == "formal_event_realization"
                and row["field_path"].startswith(prefixes)
                and row["value_kind"] == "mixed"):
            raise ValueError(f"snapshot field has conflicting value types: {row['id']}: {row['value_types']}")


ENTITY_MERGE_BASIS = {
    "formal_event_realization": "Dataset/tools/batch_generate.py;Dataset/tools/convert_to_render_ready.py;Dataset/tools/arm_ue_export.py",
    "formal_truth_frame": "Dataset/tools/convert_to_render_ready.py;Dataset/tools/arm_ue_export.py",
    "arm_local_business_truth": "Dataset/tools/l4_incident_observations.py",
    "arm_predicate_truth": "Dataset/tools/l4_business_semantics.py",
    "arm_predicate_truth_ticks": "Dataset/tools/l4_business_semantics.py",
    "domain_state": "Dataset/semantic_simulation/domain_state.py",
    "arm_script_plan": "arms/*/ue/event_script.json;Dataset/tools/arm_ue_export.py",
}


def _source_identity_resolver(kind: str, path: Path, sample: str) -> IdentityResolver:
    if kind in {"formal_event_realization", "formal_truth_frame", "domain_state"}:
        episode = path.parent.name
        sample_key = "formal:" + episode
        roster_paths = [(REPO / "aw_data/render_ready_episodes_capture_filtered" / episode /
                         "global_entity_roster.json", False)]
    elif kind == "arm_script_plan":
        branch = path.parent.parent
        if not branch.is_relative_to(REPO / "arms") or path != branch / "ue/event_script.json":
            raise ValueError(f"ARM script plan is not a branch execution script: {path}")
        sample_key = "arm:" + source_path(branch).removeprefix("arms/")
        roster_paths = [(branch / "ue/global_entity_roster.json", False)]
    else:
        branch = path.parent.parent
        sample_key = "arm:" + source_path(branch).removeprefix("arms/")
        manifest = read_json(branch / "manifest.json")
        roster_paths = [(branch / "ue/global_entity_roster.json", False)]
        parent = manifest_reference(manifest, "roster")
        if parent is not None:
            roster_paths.append((REPO / parent, True))
    resolver = IdentityResolver(sample_key)
    for roster_path, inherited in roster_paths:
        if not roster_path.is_file():
            raise FileNotFoundError(roster_path)
        for index, item in enumerate(read_roster(roster_path)["entities"]):
            resolver.declare(item, source_file=source_path(roster_path),
                             source_path=f"entities[{index}]", authority="parent_roster" if inherited else "roster",
                             inherited=inherited)
    if kind in {"domain_state", "formal_truth_frame"}:
        evidence = controller_evidence(path.parent.name)
        for controller_id, line in evidence.sumo_first_line.items():
            resolver.declare({"entity_id": controller_id, "category": "sumo_tls_controller"},
                source_file=evidence.source_file,
                source_path=f"truth_frames.jsonl:{line}:sumo_traffic_light_states.{controller_id}",
                authority="truth_frame_sumo_tls", namespace="sumo_tls_controller")
    return resolver


def _controller_namespaces(kind: str, path: Path, record: dict[str, Any]) -> dict[str, str] | None:
    if kind != "domain_state":
        return None
    return controller_namespaces_for_record(path, record)


def _source_subtrees(kind: str, record: dict[str, Any]):
    if kind == "formal_event_realization":
        yield "", record
        return
    for parent in sorted(ENTITY_KEYED_OBJECTS[kind]):
        item: Any = record
        for segment in parent.split("."):
            if not isinstance(item, dict) or segment not in item:
                break
            item = item[segment]
        else:
            yield parent, item


def _direct_source_leaves(item: Any, raw_path: str,
                          tokens: tuple[str | int, ...] = (),
                          array_lengths: dict[tuple[str | int, ...], int] | None = None):
    """Independent raw-location walk used to check every projected source leaf."""
    if isinstance(item, dict):
        if not item:
            yield tokens, raw_path, item
        for key, child in item.items():
            if is_publication_excluded_path(tokens + (key,)):
                continue
            child_path = f"{raw_path}.{key}" if raw_path else key
            yield from _direct_source_leaves(child, child_path, tokens + (key,), array_lengths)
    elif isinstance(item, list):
        if array_lengths is not None:
            array_lengths[tokens] = len(item)
        if not item:
            yield tokens, raw_path, item
        for index, child in enumerate(item):
            yield from _direct_source_leaves(child, f"{raw_path}[{index}]", tokens + (index,), array_lengths)
    else:
        yield tokens, raw_path, item


def _scan_entity_group(group: list[tuple[str, Path, str]]) -> dict[tuple[str, str, str | None, str | None], dict[str, Any]]:
    aggregates: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = {}
    for kind, path, sample in group:
        scope = inventory_scope(kind, path, sample)
        resolver = _source_identity_resolver(kind, path, sample)
        records = ((None, read_json(path)),) if path.suffix == ".json" else iter_jsonl(path)
        for line_no, record in records:
            if scope is not None:
                validate_record_identity(record, scope, source_family=kind,
                                         source_file=source_path(path), source_line=line_no)
            source_branch = source_branch_for_record(kind, record)
            source_subtype = source_subtype_for_record(kind, record)
            for parent, subtree in _source_subtrees(kind, record):
                parent_tokens = tuple(parent.split(".")) if parent else ()
                array_lengths: dict[tuple[str | int, ...], int] = {}
                direct = {tokens: (value, raw) for tokens, raw, value in
                          _direct_source_leaves(subtree, parent, parent_tokens, array_lengths)
                          if kind == "formal_event_realization" or tokens != parent_tokens}
                namespaces = _controller_namespaces(kind, path, record)
                for projected, source in ((projected, source)
                                          for projected in project_values(
                                              subtree, source_family=kind,
                                              field_prefix=parent,
                                              source_prefix_tokens=parent_tokens,
                                              entity_key_namespaces=namespaces,
                                              source_branch=source_branch,
                                              source_subtype=source_subtype,
                                              scope=scope, source_file=source_path(path),
                                              source_line=line_no)
                                          for source in projected.sources):
                    leaf = source.field_value
                    if not leaf.field_path:
                        continue
                    if kind != "formal_event_realization" and "{entity}" not in leaf.field_path:
                        continue
                    raw_value = direct.pop(leaf.source_tokens, None)
                    source_value = source_value_at_tokens(record, leaf.source_tokens)
                    if (raw_value is None or type(raw_value[0]) is not type(leaf.value)
                            or raw_value[0] != leaf.value or raw_value[1] != leaf.raw_path or
                            type(source_value) is not type(leaf.value) or source_value != leaf.value):
                        raise ValueError(f"source leaf differs from shared field walk: {path}:{line_no}:{leaf.raw_path}")
                    if tuple(token for token in leaf.source_tokens if type(token) is int) != leaf.element_positions:
                        raise ValueError(f"source array positions differ: {path}:{line_no}:{leaf.raw_path}")
                    if kind == "formal_event_realization":
                        root = leaf.source_tokens[0] if leaf.source_tokens else None
                        if root in SNAPSHOT_DICTIONARIES and len(leaf.source_tokens) > 1:
                            tokens = leaf.source_tokens
                            if (leaf.snapshot_tick != int(tokens[1]) or
                                    (leaf.entity_id != tokens[2] if len(tokens) >= 3 else leaf.entity_id is not None)):
                                raise ValueError(f"snapshot time or entity differs: {path}:{line_no}:{leaf.raw_path}")
                        elif leaf.snapshot_tick is not None:
                            raise ValueError(f"unexpected snapshot time: {path}:{line_no}:{leaf.raw_path}")
                    elif leaf.entity_id != leaf.source_tokens[len(parent_tokens)]:
                        raise ValueError(f"entity key differs from raw path: {path}:{line_no}:{leaf.raw_path}")
                    key = kind, leaf.field_path, leaf.source_branch, leaf.source_subtype
                    row = aggregates.get(key)
                    if row is None:
                        row = {"kind": "state_field", "id": source_field_id(*key),
                               "source_family": kind, "field_path": leaf.field_path,
                               "owner": _owner(kind, leaf.field_path), "value_types": set(),
                               "source_scope_status": "sample" if scope is not None else "not_applicable",
                               "source_files": set(), "instance_count": 0,
                               "normalizations": {},
                               "payload_ranks": set(), "envelope_array_ranks": set(),
                               "first_source_file": source_path(path), "first_source_line": line_no,
                               "time_granularity": _time_granularity(kind, leaf.field_path)}
                        if leaf.source_branch is not None:
                            row["source_branch"] = {"observation_family": leaf.source_branch}
                            if leaf.source_subtype is not None:
                                row["source_branch"][FACILITY_SUBTYPE_KEY] = leaf.source_subtype
                        if "{entity}" in leaf.field_path:
                            row.update(entity_identities=set(), unresolved_value_count=0,
                                       unresolved_reasons={}, entity_merge_basis=ENTITY_MERGE_BASIS[kind])
                            if leaf.field_path.endswith("[]"):
                                if kind == "formal_truth_frame" and leaf.field_path.startswith("sumo_traffic_light_states.{entity}."):
                                    row["array_axis_lengths"] = [set() for _ in range(leaf.field_path.count("[]"))]
                                else:
                                    row["array_lengths"] = set()
                        aggregates[key] = row
                    _observe_normalization(row, projected)
                    row["value_types"].add(_value_type(leaf.value))
                    row["payload_ranks"].add(len(projected.payload_shape))
                    row["envelope_array_ranks"].add(sum(type(token) is int
                                                         for token in projected.envelope_tokens))
                    row["source_files"].add(source_path(path))
                    row["instance_count"] += 1
                    if "entity_identities" in row:
                        if "array_lengths" in row:
                            array_tokens = leaf.source_tokens[:-1]
                            length = array_lengths.get(array_tokens)
                            if length is None:
                                raise ValueError(f"entity array source missing: {path}:{line_no}:{array_tokens}")
                            row["array_lengths"].add(length)
                        if "array_axis_lengths" in row:
                            for axis, array_tokens in enumerate(_array_prefixes(leaf.source_tokens, leaf.field_path)):
                                length = array_lengths.get(array_tokens)
                                if length is None:
                                    raise ValueError(f"SUMO controller array axis missing: {path}:{line_no}:{array_tokens}")
                                row["array_axis_lengths"][axis].add(length)
                        resolved = resolver.resolve(leaf.entity_id, source_file=source_path(path),
                            source_line=line_no, source_path=leaf.raw_path,
                            namespace=leaf.entity_namespace or "sample",
                            source_status=leaf.identity_status, source_reason=leaf.identity_reason)
                        if resolved.status == "resolved":
                            row["entity_identities"].add((resolved.sample_key, resolved.namespace,
                                                           resolved.lifecycle_id, resolved.raw_id))
                        else:
                            row["unresolved_value_count"] += 1
                            reason = resolved.reason or resolved.status
                            reasons = row["unresolved_reasons"]
                            reasons[reason] = reasons.get(reason, 0) + 1
                if direct:
                    raise ValueError(f"source leaves missing from shared field walk: {path}:{line_no}:{list(direct)[:5]}")
    return aggregates


def _entity_template(kind: str, field_path: str) -> str:
    if kind == "formal_event_realization":
        parents = [root + ".{tick}" for root in SNAPSHOT_DICTIONARIES]
    else:
        parents = ENTITY_KEYED_OBJECTS.get(kind, ())
    for parent in parents:
        if not field_path.startswith(parent + "."):
            continue
        rest = field_path[len(parent) + 1:]
        match = re.fullmatch(r"([^\.\[]+)(.*)", rest)
        if match and match.group(1) != "{entity}":
            return parent + ".{entity}" + match.group(2)
    return field_path


def rebuild_entity_field_catalog(output: Path, *, workers: int = 6,
                                 families: set[str] | None = None) -> dict[str, int]:
    """Rescan only confirmed entity-keyed source structures and freeze a new catalog."""
    previous = _read_current_catalog(output)
    kinds = set(ENTITY_MERGE_BASIS) if families is None else set(families)
    if not kinds or not kinds <= set(ENTITY_MERGE_BASIS):
        raise ValueError(f"unsupported entity source family selection: {kinds}")
    _source_rebuild_preflight(kinds)
    old_groups: dict[tuple[str, str, str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
    retained = []
    for row in previous:
        kind, path = row["source_family"], row["field_path"]
        source_branch = catalog_source_branch(row)
        source_subtype = catalog_source_subtype(row)
        template = _entity_template(kind, path)
        if kind in kinds and (kind == "formal_event_realization" or template != path
                              or "{entity}" in path):
            old_groups[(kind, template, source_branch, source_subtype)].append(row)
        else:
            retained.append(row)
    for (kind, template, branch, subtype), group in old_groups.items():
        declarations = {(row["owner"], tuple(row["value_types"]), row["value_kind"],
                         row["unit"], row["unit_status"], row["unit_basis"],
                         row["time_granularity"], row["field_path"].count("[]"),
                         "ENU" if "_enu_m" in row["field_path"] else None)
                        for row in group}
        if len(declarations) != 1:
            raise ValueError(f"entity field declarations conflict: {kind}:{branch}:{subtype}:{template}: "
                             f"{[row['id'] for row in group]}")
    source_rows = [source for source in _sources() if source[0] in kinds]
    groups = [list(items) for _, items in groupby(
        sorted(source_rows, key=lambda item: item[2]), key=lambda item: item[2])]
    aggregates: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = {}
    if workers == 1:
        for group in groups:
            _merge_group(aggregates, _scan_entity_group(group))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_scan_entity_group, group) for group in groups]
            for future in as_completed(futures):
                _merge_group(aggregates, future.result())
    rebuilt = [_finished_row(row) for row in aggregates.values()]
    current = {(row["source_family"], row["field_path"], catalog_source_branch(row),
                catalog_source_subtype(row)): row
               for row in rebuilt}
    for key, old_rows in old_groups.items():
        new = current.get(key)
        if new is None:
            raise ValueError(f"source field disappeared during entity rebuild: {key}")
        old = old_rows[0]
        fields = ("value_types", "value_kind", "unit", "unit_status", "unit_basis", "time_granularity")
        if not (key[0] == "formal_truth_frame" and key[1].startswith("sumo_traffic_light_states.{entity}.")):
            fields = ("owner",) + fields
        elif old["owner"] != "source_only" or new["owner"] != "ContextAnchor":
            raise ValueError(f"SUMO controller field owner migration differs: {key}")
        for field in fields:
            if old[field] != new[field]:
                raise ValueError(f"source field declaration changed: {key}:{field}")
    if set(current) != set(old_groups):
        raise ValueError(f"entity rebuild found unreviewed fields: {sorted(set(current) - set(old_groups))[:20]}")
    rows = retained + rebuilt
    _check_snapshot_types(rows)
    _write_compact_catalog(output, rows)
    return {"source_files": len(source_rows), "source_groups": len(groups),
            "field_rows": len(rows), "rebuilt_fields": len(rebuilt),
            "entity_fields": sum("{entity}" in row["field_path"] for row in rebuilt),
            "entity_value_records": sum(row["instance_count"] for row in rebuilt if "{entity}" in row["field_path"]),
            "unresolved_value_records": sum(row.get("unresolved_value_count", 0) for row in rebuilt)}


def from_structure(output: Path, *,
                   structure_path: Path = REPO / "design/belief/structure_fields.jsonl",
                   inventory_path: Path = REPO / "design/belief/structure_source_inventory.jsonl",
                   acceptance_path: Path = REPO / "design/belief/structure_acceptance.json"
                   ) -> dict[str, int]:
    """Reuse published grammar bindings and counts, including unobserved fields.

    This does not infer meaning, units, identity counts, or per-field file counts.
    Identity and array positions are checked against the authentic record when
    graph nodes are written. Missing sources keep their inventory status.
    """
    from Dataset.world_model.graph.structure_catalog import (
        CatalogRows, _digest_map_tokens, validate_catalog_rows,
    )

    acceptance = read_json(acceptance_path)
    structure_rows = [row for _, row in iter_jsonl(structure_path)]
    inventory_rows = [row for _, row in iter_jsonl(inventory_path)]
    header = structure_rows[0]
    summaries = [row for row in inventory_rows
                 if row["kind"] == "structure_source_inventory_summary"]
    if (header.get("kind") != "structure_fields_header" or len(summaries) != 1
            or summaries[0].get("source_schema_closure") != "complete"
            or acceptance.get("source_schema_closure") != "complete"):
        raise ValueError("structural delivery has not completed publication acceptance")
    source_rows = tuple(row for row in structure_rows if row["kind"] == "source_field")
    normalized_rows = tuple(row for row in structure_rows if row["kind"] == "normalized_field")
    validate_catalog_rows(CatalogRows(source_rows, normalized_rows))
    declarations = [row for row in inventory_rows if row["kind"] == "source_declaration"]
    if len(declarations) != summaries[0]["declarations"]:
        raise ValueError("structural source inventory count differs")
    if any(row["structure_status"] != "field_scan_defined" for row in declarations):
        raise ValueError("structural inventory contains a pending source")

    rows = []
    excluded = 0
    for source in source_rows:
        path = source["field_path"]
        integrity_tokens = _digest_map_tokens(path)
        if (is_publication_excluded_path(tuple(path.replace("[]", "").split(".")))
                or integrity_tokens is not None and is_publication_excluded_path(integrity_tokens)):
            excluded += 1
            continue
        bindings = deepcopy(source["normalizations"])
        value_types = sorted(set(source["observed_source_value_types"]) | {
            value_type for binding in bindings
            for value_type in binding["source_value_types"] or ()})
        if not value_types:
            raise ValueError(f"source grammar declares no value types: {source['id']}")
        nonnull = set(value_types) - {"null"}
        value_kind = ("null" if not nonnull else
                      "numeric" if nonnull <= {"int", "float"} else
                      "boolean" if nonnull == {"bool"} else
                      "categorical" if nonnull == {"string"} else
                      "container" if nonnull <= {"empty_list", "empty_object"} else "mixed")
        family = source["source_family"]
        row = {"kind": "state_field", "id": source["id"],
               "source_family": family, "field_path": path,
               "owner": _owner(family, path), "value_types": value_types,
               "observed_value_types": source["observed_source_value_types"],
               "value_kind": value_kind, "instance_count": source["instance_count"],
               "normalizations": bindings,
               "structure_source_index": source["source_index"],
               "structure_catalog_ref": source_path(structure_path),
               "source_inventory_ref": source_path(inventory_path),
               "source_file_count": None, "first_source_file": None,
               "first_source_line": None,
               "provenance_count_status": "per_record_graph_validation",
               "source_scope_status": "declared_by_source_inventory",
               "time_granularity": _time_granularity(family, path),
               "shape_status": "grammar_binding", "shape_length": None,
               "structure_declaration_basis": sorted({binding["declaration_basis"] for binding in bindings}),
               "missing_rule": {"declaration_status": "closed_record_grammar",
                   "basis": "Each source field retains its own normalizations and structure_declaration_basis.",
                   "absent": "No value is imputed; applicability and presence follow the source record grammar.",
                   "null": "Preserve a permitted source null without replacing it by a numeric value."}}
        if source["source_branch"] is not None:
            row["source_branch"] = {"observation_family": source["source_branch"]}
            if source["source_subtype"] is not None:
                row["source_branch"][FACILITY_SUBTYPE_KEY] = source["source_subtype"]
        if "{entity}" in path:
            row["entity_count"] = None
            row["unresolved_value_count"] = None
            row["identity_observation_status"] = "per_record_graph_validation"
        rows.append(row)
    _write_compact_catalog(output, rows)
    return {"source_fields": len(rows), "excluded_historical_integrity_fields": excluded,
            "declarations": len(declarations),
            "source_missing": summaries[0]["status_counts"]["source_missing"],
            "zero_instance": summaries[0]["status_counts"]["zero_instance"],
            "meaning_unresolved": sum(row["meaning_status"] == "unresolved" for row in rows),
            "numeric_unit_unresolved": sum(row["unit_status"] == "unresolved" for row in rows)}


def refresh_objective_manifest_fields(output: Path) -> dict[str, int]:
    """Rebuild this changed JSON family from its grammar and all current roots.

    This is a new graph release: removed manifest fields disappear and indices
    are rebuilt. Other families retain their labelled observation snapshot.
    """
    from Dataset.world_model.graph.structure_grammar_catalog import compile_catalog_grammars

    declarations = [d for d in declared_source_inventory()
                    if d.source_family == "formal_objective_manifest"]
    if len(declarations) != 210 or any(not d.path.is_file() for d in declarations):
        raise ValueError("the current objective manifest corpus is incomplete")
    plan = compile_catalog_grammars(declarations)
    aggregates = _scan_group([(d.source_family, d.path, d.sample) for d in declarations])
    rules = {(r.source_family, r.field_path, r.source_branch, r.source_subtype): r
             for r in plan.rules}
    if set(aggregates) - set(rules):
        raise ValueError(f"manifest observations exceed the grammar fields: {sorted(set(aggregates) - set(rules))}")
    rows = _read_current_catalog(output)
    rows = [r for r in rows if r["source_family"] != "formal_objective_manifest"]
    for row in rows:
        row["observation_count_basis"] = "published structure-1 observation snapshot"
    for key, rule in sorted(rules.items()):
        observed = aggregates.get(key)
        bindings = list(observed["normalizations"].values()) if observed is not None else []
        raw_bindings = [b for b in bindings if b["normalized_id"] == "structural:source:" + rule.source_field_id
                        and b["binding_slot"] is None]
        binding = {**rule.catalog_rule,
            "declaration_status": "closed_record_grammar",
            "observed_source_value_types": sorted(set().union(*(b["observed_source_value_types"] for b in raw_bindings))),
            "observed_value_types": sorted(set().union(*(b["observed_value_types"] for b in raw_bindings))),
            "observed_value_statuses": sorted(set().union(*(b["observed_value_statuses"] for b in raw_bindings))),
            "observed_structure_statuses": sorted(set().union(*(b["observed_structure_statuses"] for b in raw_bindings))),
            "observed_shapes": [], "instance_count": sum(b["instance_count"] for b in raw_bindings)}
        for name in ("source_value_types", "value_types", "value_statuses", "structure_statuses"):
            binding[name] = sorted(binding[name])
        types = sorted(rule.source_value_types)
        nonnull = set(types) - {"null"}
        kind = ("null" if not nonnull else "numeric" if nonnull <= {"int", "float"}
                else "boolean" if nonnull == {"bool"} else "categorical" if nonnull == {"string"}
                else "container" if nonnull <= {"empty_list", "empty_object"} else "mixed")
        completed = _finished_row(observed) if observed else None
        normalized = [b for b in completed["normalizations"]
                      if (b["normalized_id"], b["binding_slot"]) != (binding["normalized_id"], None)] if completed else []
        normalized.append(binding)
        row = {"kind": "state_field", "id": rule.source_field_id,
            "source_family": rule.source_family, "field_path": rule.field_path,
            "owner": _owner(rule.source_family, rule.field_path), "value_types": types,
            "observed_value_types": completed["value_types"] if completed else [], "value_kind": kind,
            "instance_count": observed["instance_count"] if observed else 0, "normalizations": normalized,
            "source_file_count": len(observed["source_files"]) if observed else 0,
            "first_source_file": observed["first_source_file"] if observed else None,
            "first_source_line": observed["first_source_line"] if observed else None,
            "source_scope_status": "sample", "source_inventory_ref": "design/belief/source_inventory.jsonl",
            "observation_count_basis": "all 210 current objective JSON manifests",
            "structure_declaration_basis": list(rule.schema_refs),
            "time_granularity": _time_granularity(rule.source_family, rule.field_path),
            "shape_status": "grammar_binding", "shape_length": None,
            "missing_rule": {"declaration_status": "closed_record_grammar",
                "basis": "Each source field retains its own normalizations and structure_declaration_basis.",
                "absent": "No value is imputed; applicability and presence follow the source record grammar.",
                "null": "Preserve a permitted source null without replacing it by a numeric value."}}
        if rule.source_branch is not None:
            row["source_branch"] = {"observation_family": rule.source_branch}
            if rule.source_subtype is not None:
                row["source_branch"][FACILITY_SUBTYPE_KEY] = rule.source_subtype
        rows.append(row)
    _write_compact_catalog(output, rows, rebuild_indices=True)
    return {"manifests": len(declarations), "manifest_fields": len(rules), "source_fields": len(rows)}


def append_current_domain_fields(output: Path) -> dict[str, int]:
    """Declare the current writers without inventing observed instance counts."""
    from Dataset.world_model.graph.structure_grammar_catalog import compile_catalog_grammars

    declaration = next(d for d in declared_source_inventory() if d.source_family == "domain_state")
    plan = compile_catalog_grammars([declaration])
    rows = _read_current_catalog(output)
    rows = [row for row in rows if row.get("source_scope_status") != "declared_by_current_producers"]
    existing = {row["id"] for row in rows}
    added = 0
    for rule in plan.rules:
        if rule.source_field_id in existing:
            continue
        binding = {**rule.catalog_rule, "declaration_status": "closed_record_grammar",
            "observed_source_value_types": [], "observed_value_types": [],
            "observed_value_statuses": [], "observed_structure_statuses": [],
            "observed_shapes": [], "instance_count": 0}
        for name in ("source_value_types", "value_types", "value_statuses", "structure_statuses"):
            binding[name] = list(binding[name])
        types = list(rule.source_value_types)
        nonnull = set(types) - {"null"}
        kind = ("null" if not nonnull else "numeric" if nonnull <= {"int", "float"}
                else "boolean" if nonnull == {"bool"} else "categorical" if nonnull == {"string"}
                else "container" if nonnull <= {"empty_list", "empty_object"} else "mixed")
        row = {"kind": "state_field", "id": rule.source_field_id,
            "source_family": rule.source_family, "field_path": rule.field_path,
            "source_branch": {"observation_family": rule.source_branch},
            "owner": _owner(rule.source_family, rule.field_path), "value_types": types,
            "observed_value_types": [], "value_kind": kind, "instance_count": 0,
            "normalizations": [binding], "source_file_count": None,
            "first_source_file": None, "first_source_line": None,
            "source_scope_status": "declared_by_current_producers",
            "source_inventory_ref": "design/belief/source_inventory.jsonl",
            "observation_count_basis": "not enumerated; zero denotes no counted observations",
            "provenance_count_status": "per_record_graph_validation",
            "structure_declaration_basis": list(rule.schema_refs),
            "time_granularity": _time_granularity(rule.source_family, rule.field_path),
            "shape_status": "grammar_binding", "shape_length": None,
            "missing_rule": {"declaration_status": "closed_record_grammar",
                "basis": "Each source field retains its own normalizations and structure_declaration_basis.",
                "absent": "No value is imputed; applicability and presence follow the source record grammar.",
                "null": "Preserve a permitted source null without replacing it by a numeric value."}}
        rows.append(row)
        added += 1
    _write_compact_catalog(output, rows, rebuild_indices=True)
    return {"added_domain_fields": added, "source_fields": len(rows),
            "domain_branches": len({r.source_branch for r in plan.rules})}


def scan(output: Path, *, workers: int = 12) -> dict[str, int]:
    _source_rebuild_preflight()
    source_rows = list(_sources())
    groups = [list(items) for _, items in groupby(
        sorted(source_rows, key=lambda item: item[2]), key=lambda item: item[2])]
    aggregates: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = {}

    if workers == 1:
        for group in groups:
            _merge_group(aggregates, _scan_group(group))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_scan_group, group) for group in groups]
            for future in as_completed(futures):
                _merge_group(aggregates, future.result())
    rows = _rule_field_rows() + [_finished_row(aggregates[key]) for key in sorted(aggregates)]
    _check_snapshot_types(rows)
    _write_compact_catalog(output, rows)
    return {"source_files": len(source_rows), "samples": len(groups),
            "field_rows": len(rows)}


def rebuild_source_families(output: Path, *, families: set[str], workers: int = 6) -> dict[str, int]:
    """Replace complete families whose authoritative source files changed."""
    if families != {"arm_script_plan"}:
        raise ValueError(f"unreviewed source-family rebuild: {sorted(families)}")
    previous = _read_current_catalog(output)
    _source_rebuild_preflight(families)
    source_rows = [source for source in _sources() if source[0] in families]
    if not source_rows:
        raise ValueError("selected source families have no source files")
    groups = [list(items) for _, items in groupby(
        sorted(source_rows, key=lambda item: item[2]), key=lambda item: item[2])]
    aggregates: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = {}
    if workers == 1:
        for group in groups:
            _merge_group(aggregates, _scan_group(group))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_scan_group, group) for group in groups]
            for future in as_completed(futures):
                _merge_group(aggregates, future.result())
    rebuilt = [_finished_row(aggregate) for aggregate in aggregates.values()]
    retained = [row for row in previous if row["source_family"] not in families]
    _write_compact_catalog(output, retained + rebuilt)
    return {"source_files": len(source_rows), "samples": len(groups),
            "rebuilt_fields": len(rebuilt), "field_rows": len(retained) + len(rebuilt)}


def reclassify_owners(output: Path) -> dict[str, int]:
    """Apply the current source-to-graph ownership map to a catalog."""
    rows = _read_current_catalog(output)
    changed = 0
    for row in rows:
        if row.get("source_family") == "rule_contract":
            continue
        owner = _owner(row["source_family"], row["field_path"])
        if row["owner"] != owner:
            row["owner"] = owner
            changed += 1
        row["time_granularity"] = _time_granularity(row["source_family"], row["field_path"])
        value_types = row.get("value_types") or []
        row.update(unit_policy(row["source_family"], row["field_path"], value_types))
        if row["source_family"] == "formal_world_truth_graph_deltas" and row["field_path"].startswith("operations[]"):
            row["projection_targets"] = ["PredicateAssertion", "ContextAnchor"]
    _write_compact_catalog(output, rows)
    return {"field_rows": len(rows), "owner_changes": changed}


def rebuild_contract_mappings(output: Path) -> dict[str, int]:
    """Reclassify retained source rows without replaying unchanged source scans."""
    rows = _read_current_catalog(output)
    _write_compact_catalog(output, rows)
    definitions = build_definitions(rows, CONTRACT_VERSION)
    return {"field_rows": len(rows), "definition_rows": len(definitions),
            "shared_groups": sum(len(row["source_fields"]) > 1 for row in definitions),
            "unit_evidence_gaps": sum(row["unit_status"] == "unresolved" for row in rows)}


def _rule_field_rows() -> list[dict[str, Any]]:
    """Build rule-input declarations from the current executable predicate source."""
    source = REPO / "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json"
    requirements: dict[str, dict[str, Any]] = {}
    for predicate in read_json(source)["templates"]:
        contract = predicate.get("state_contract", {})
        unknown_fields = set(contract.get("unknown_if_missing", []))
        for field in contract.get("required_fields", []):
            name = field["field"]
            rule = requirements.setdefault(name, {"unit": field.get("unit"), "source": field.get("source"),
                                              "predicate_ids": [], "unknown_if_missing_predicates": []})
            if (rule["unit"], rule["source"]) != (field.get("unit"), field.get("source")):
                raise ValueError(f"rule field source or unit conflict: {name}")
            rule["predicate_ids"].append(predicate["id"])
            if name in unknown_fields:
                rule["unknown_if_missing_predicates"].append(predicate["id"])
        undeclared = unknown_fields - {field["field"] for field in contract.get("required_fields", [])}
        if undeclared:
            raise ValueError(f"unknown_if_missing field absent from required_fields: {predicate['id']}: {undeclared}")
    rows: list[dict[str, Any]] = []
    for name, requirement in sorted(requirements.items()):
        field_id = "rule_contract:" + name
        unit = requirement["unit"]
        value_kind = "boolean" if unit == "bool" else "categorical" if unit in {"four-value", "state-code", "identifier"} else "numeric"
        value_types = ["bool"] if value_kind == "boolean" else ["string"] if value_kind == "categorical" else ["int", "float"]
        row = {"kind": "rule_required_field", "id": field_id, "source_family": "rule_contract",
               "field_path": name, "owner": "rule_input", "value_types": value_types,
               "source_scope_status": "not_applicable",
               "value_kind": value_kind, "unit": unit, "unit_status": "declared",
               "unit_basis": source_path(source) + ":state_contract.required_fields",
               "source": requirement["source"], "source_file_count": 0,
               "first_source_file": source_path(source), "first_source_line": None,
               "instance_count": None, "graph_instance_count": 0, "sample_count": 0,
               "source_mapping_status": "requires_path_level_reconciliation",
               "time_granularity": "rule_evaluation",
               "predicate_ids": requirement["predicate_ids"],
               "unknown_if_missing_predicates": requirement["unknown_if_missing_predicates"]}
        rows.append(row)
    return rows


def append_rule_fields(output: Path) -> dict[str, int]:
    """Register current rule inputs on a current-version source catalog."""
    rows = _read_current_catalog(output)
    existing = {row["id"]: row for row in rows}
    declared = _rule_field_rows()
    for row in declared:
        old = existing.get(row["id"])
        if old is not None:
            for name in ("predicate_ids", "unknown_if_missing_predicates", "unit", "source"):
                if old.get(name) != row[name]:
                    raise ValueError(f"rule field differs from current predicate source: {row['id']}:{name}")
        else:
            rows.append(row)
    _write_compact_catalog(output, rows)
    return {"rule_fields": len(declared), "field_rows": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "design/belief/field_catalog.jsonl")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--rule-fields", action="store_true")
    parser.add_argument("--reclassify", action="store_true")
    parser.add_argument("--entity-rebuild", action="store_true")
    parser.add_argument("--source-rebuild", action="store_true")
    parser.add_argument("--contract-rebuild", action="store_true")
    parser.add_argument("--from-structure", action="store_true")
    parser.add_argument("--source-family", action="append", choices=sorted(ENTITY_MERGE_BASIS))
    args = parser.parse_args()
    if args.from_structure:
        print(from_structure(args.output))
    elif args.source_rebuild:
        print(rebuild_source_families(args.output, workers=args.workers,
              families=set(args.source_family or ())))
    elif args.entity_rebuild:
        print(rebuild_entity_field_catalog(args.output, workers=args.workers,
              families=set(args.source_family) if args.source_family else None))
    elif args.rule_fields:
        print(append_rule_fields(args.output))
    elif args.reclassify:
        print(reclassify_owners(args.output))
    elif args.contract_rebuild:
        print(rebuild_contract_mappings(args.output))
    else:
        print(scan(args.output, workers=args.workers))
