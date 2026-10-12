"""Verify the frozen label-graph catalog, vocabulary, fields and partitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from Dataset.world_model.graph.contract import CONTRACT_VERSION, EDGE_ENDPOINTS, NODE_TYPES, REPO, compile_catalog, iter_jsonl
from Dataset.world_model.graph.field_paths import (
    DYNAMIC_KEYED_OBJECTS, SNAPSHOT_DICTIONARIES, catalog_source_branch,
)
from Dataset.world_model.graph.field_contract import (
    DEFAULT_DEFINITIONS, build_definitions, coordinate_policy, source_time_role, unit_policy,
)
from Dataset.world_model.graph.field_catalog import _entity_template, ENTITY_MERGE_BASIS, validate_normalizations
from Dataset.world_model.graph.records import iter_partitions, link_node_attrs, load_scope
from Dataset.world_model.graph.validator import GraphValidator
from Dataset.world_model.graph.vocabulary import DEFAULT_VOCAB, script_source_provenance


def _verify_dynamic_key_structure(row: dict) -> None:
    family, path = row["source_family"], row["field_path"]
    allowed_predicate_template = any(
        family == declared_family and path == f"{parent}.{{predicate}}"
        for declared_family, parents in DYNAMIC_KEYED_OBJECTS.items()
        for parent, dimension in parents.items() if dimension == "predicate")
    if "{predicate}" in path and not allowed_predicate_template:
        raise ValueError(f"predicate template has no declared parent: {row['id']}")
    if "{source_path}" in path and (family != "arm_manifest" or path != "source_refs.{source_path}"):
        raise ValueError(f"source path template has no declared parent: {row['id']}")
    for parent, dimension in DYNAMIC_KEYED_OBJECTS.get(family, {}).items():
        if path.startswith(parent + ".") and path != f"{parent}.{{{dimension}}}":
            raise ValueError(f"dynamic predicate map has a concrete key: {row['id']}")
    if family == "arm_manifest" and path.startswith("source_refs."):
        mode = row.get("source_ref_key_mode")
        if mode == "source_path":
            if path != "source_refs.{source_path}":
                raise ValueError(f"X ARM source reference has a concrete path key: {row['id']}")
        elif mode == "role":
            if path == "source_refs.{source_path}":
                raise ValueError(f"role-key source reference uses the X ARM template: {row['id']}")
        else:
            raise ValueError(f"ARM source reference lacks its producer key mode: {row['id']}")


def verify_partition(partition: Path, validator: GraphValidator) -> dict[str, int]:
    scope = load_scope(partition)
    if scope["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"partition contract version differs: {partition}")

    def nodes():
        for batch in pq.ParquetFile(partition / "nodes.parquet").iter_batches(batch_size=8192):
            for row in batch.to_pylist():
                row["attrs"] = json.loads(row.pop("attrs_json"))
                if row["sample_key"] != scope["sample_key"]:
                    raise ValueError(f"node sample key differs: {row['node_id']}")
                validator.validate_node(row)
                row["attrs"] = link_node_attrs(row)
                yield row

    def edges():
        for batch in pq.ParquetFile(partition / "edges.parquet").iter_batches(batch_size=8192):
            for row in batch.to_pylist():
                row["attrs"] = json.loads(row.pop("attrs_json"))
                if row["sample_key"] != scope["sample_key"]:
                    raise ValueError(f"edge sample key differs: {row['edge_id']}")
                yield row

    counts = validator.validate_graph(nodes(), edges(), nodes_prevalidated=True)
    if counts != {key: scope["counts"][key] for key in ("nodes", "edges")}:
        raise ValueError(f"partition row count differs from scope: {partition}: {counts}")
    return counts


def verify(*, graphs: bool = False, partition: Path | None = None) -> dict[str, int]:
    contract_path = REPO / "design/belief/contract.jsonl"
    field_path = REPO / "design/belief/field_catalog.jsonl"
    for path in (contract_path, DEFAULT_VOCAB, field_path, DEFAULT_DEFINITIONS):
        if not path.is_file():
            raise ValueError(f"required graph index missing: {path}")
    contract = [row for _, row in iter_jsonl(contract_path)]
    vocabulary = [row for _, row in iter_jsonl(DEFAULT_VOCAB)]
    fields = [row for _, row in iter_jsonl(field_path)]
    definitions = [row for _, row in iter_jsonl(DEFAULT_DEFINITIONS)]
    if not contract or contract[0].get("kind") != "schema" or contract[0].get("version") != CONTRACT_VERSION:
        raise ValueError("contract header version differs")
    compiled = json.loads(json.dumps(compile_catalog(), ensure_ascii=False))
    if contract != compiled:
        raise ValueError("frozen contract differs from current source compiler")
    if not vocabulary or vocabulary[0].get("kind") != "vocabulary" or vocabulary[0].get("contract_version") != CONTRACT_VERSION:
        raise ValueError("vocabulary header version differs")
    if {row["id"] for row in contract if row["kind"] == "node_type"} != set(NODE_TYPES):
        raise ValueError("node types differ from contract compiler")
    if {row["id"] for row in contract if row["kind"] == "edge_type"} != set(EDGE_ENDPOINTS):
        raise ValueError("edge types differ from contract compiler")
    categories = [row for row in vocabulary if row["kind"] == "node_category"]
    category_keys = [(row["node_type"], row["category_id"]) for row in categories]
    if len(category_keys) != len(set(category_keys)) or len({row["index"] for row in categories}) != len(categories):
        raise ValueError("duplicate node category or index")
    if {row["index"] for row in categories} != set(range(len(categories))):
        raise ValueError("node category indices are not contiguous")
    category_set = set(category_keys)
    source_references = script_source_provenance(contract)
    frozen_references = {row["category_id"]: {key: row.get(key) for key in
                           ("source_scripts", "source_realizations", "category_status", "single_scenario")}
                           for row in categories if row["node_type"] == "ScriptEvent"
                           and row["category_id"] is not None}
    if frozen_references != source_references:
        raise ValueError("ScriptEvent intent sources changed; bump contract version and rebuild")
    category_owners = {
        "predicate": ("PredicateAssertion", "PredicateTransition", "PredicateEpisodeOutcome"),
        "event": ("EventOccurrence", "EventOutcome"),
        "entity_category": ("EntityIdentity", "EntityState"),
        "script_event_category": ("ScriptEvent",),
        "command_category": ("ActionPlan", "CommandDispatch"),
        "context_category": ("ContextAnchor",),
        "event_phase_category": ("EventPhase",),
        "predicate_episode_outcome_category": ("PredicateEpisodeOutcome",),
        "time_category": ("TimePoint",),
    }
    for row in contract:
        for owner in category_owners.get(row["kind"], ()):
            if (owner, row["id"]) not in category_set:
                raise ValueError(f"category missing index: {owner}:{row['id']}")
    role_rows = [row for row in vocabulary if row["kind"] == "edge_role"]
    if len({row["role"] for row in role_rows}) != len(role_rows) or len({row["index"] for row in role_rows}) != len(role_rows):
        raise ValueError("duplicate role or role index")
    if {row["index"] for row in role_rows} != set(range(len(role_rows))):
        raise ValueError("role indices are not contiguous")
    declared_roles = {role["key"] for row in contract if row["kind"] in {"predicate", "event"}
                           for role in row.get("roles", [])}
    declared_roles.update(row["id"] for row in contract if row["kind"] == "graph_edge_role")
    if not declared_roles <= {row["role"] for row in role_rows}:
        raise ValueError("declared role missing vocabulary index")
    field_ids = [row["id"] for row in fields]
    if not field_ids or len(field_ids) != len(set(field_ids)):
        raise ValueError("field catalog empty or contains duplicate IDs")
    field_indices = [row.get("index") for row in fields]
    if any(not isinstance(index, int) or index < 0 for index in field_indices) or len(set(field_indices)) != len(fields):
        raise ValueError("field catalog has absent or duplicate indices")
    if set(field_indices) != set(range(len(fields))):
        raise ValueError("field catalog indices are not contiguous")
    normalized_codes = validate_normalizations(fields)
    structure_conflicts = [row["id"] for row in fields
                           if any("structure_conflict" in binding.get("observed_structure_statuses", [])
                                  for binding in row.get("normalizations", []))]
    if structure_conflicts:
        raise ValueError(f"source structures remain in conflict: {len(structure_conflicts)}; "
                         f"{structure_conflicts[:5]}")
    if definitions != build_definitions(fields, CONTRACT_VERSION):
        raise ValueError("field definitions differ from source mappings")
    unresolved_meanings = [row["id"] for row in definitions if row.get("meaning_status") != "declared"]
    if unresolved_meanings:
        raise ValueError(f"field meanings remain unfinished: {len(unresolved_meanings)}; {unresolved_meanings[:5]}")
    unresolved_numeric_units = [row["id"] for row in fields
                                if {"int", "float"} & set(row["value_types"])
                                and row.get("unit_status") == "unresolved"]
    if unresolved_numeric_units:
        raise ValueError(f"numeric source units remain unfinished: {len(unresolved_numeric_units)}; "
                         f"{unresolved_numeric_units[:5]}")
    undeclared_nulls = [row["id"] for row in definitions
                        if "null" in row["value_types"] and
                        row["missing_rule"].get("declaration_status") == "observed_types_only"]
    if undeclared_nulls:
        raise ValueError(f"null source meanings remain unfinished: {len(undeclared_nulls)}; "
                         f"{undeclared_nulls[:5]}")
    for row in fields:
        if row.get("contract_version") != CONTRACT_VERSION:
            raise ValueError(f"field catalog version differs: {row['id']}")
        catalog_source_branch(row)
        _verify_dynamic_key_structure(row)
        if row["source_family"] == "formal_event_realization":
            root, separator, suffix = row["field_path"].partition(".")
            if root in SNAPSHOT_DICTIONARIES and separator and suffix.split(".", 1)[0] != "{tick}":
                raise ValueError(f"snapshot field has concrete tick key: {row['id']}")
        if row["source_family"] in ENTITY_MERGE_BASIS and _entity_template(
                row["source_family"], row["field_path"]) != row["field_path"]:
            raise ValueError(f"entity-keyed field has a concrete entity ID: {row['id']}")
        if "{entity}" in row["field_path"]:
            if row.get("entity_count") is None or row.get("unresolved_value_count") is None:
                raise ValueError(f"entity field lacks provenance counts: {row['id']}")
            if row["entity_count"] < 0 or row["unresolved_value_count"] < 0:
                raise ValueError(f"entity field has invalid identity counts: {row['id']}")
            expected_namespace = ("sumo_lane" if row["source_family"] == "domain_state"
                                  and row["field_path"].startswith("values.queue_sampling_window.stopped_vehicle_count_by_lane.")
                                  else "sumo_tls_controller" if row["source_family"] == "formal_truth_frame"
                                  and row["field_path"].startswith("sumo_traffic_light_states.{entity}.")
                                  else "source_scoped" if row["source_family"] == "domain_state"
                                  and row["field_path"].startswith("values.red_duration_ticks_by_controller.")
                                  else "sample")
            if row.get("entity_namespace") != expected_namespace:
                raise ValueError(f"entity field namespace differs: {row['id']}")
            if expected_namespace == "source_scoped" and row.get("entity_namespaces") != ["sample", "sumo_tls_controller"]:
                raise ValueError(f"controller field namespace sources differ: {row['id']}")
            if row["field_path"].endswith("[]"):
                if expected_namespace == "sumo_tls_controller":
                    axes = row.get("observed_axis_lengths")
                    if (not isinstance(axes, list) or len(axes) != row["array_rank"] or
                            any(not isinstance(axis, list) or not axis or
                                any(type(length) is not int or length < 0 for length in axis)
                                for axis in axes) or
                            row.get("shape_status") != ("variable_observed" if any(len(axis) > 1 for axis in axes)
                                                         else "fixed_observed") or
                            row.get("shape_length") is not None):
                        raise ValueError(f"SUMO controller array axes lack source check: {row['id']}")
                elif (row.get("shape_status") != "fixed_observed" or
                      type(row.get("shape_length")) is not int or row["shape_length"] < 0):
                    raise ValueError(f"entity array shape has not been checked: {row['id']}")
        if row.get("owner") not in {*NODE_TYPES, "RelativeState", "rule_input", "source_only", "edge_attribute"}:
            raise ValueError(f"field owner unresolved: {row['id']}: {row.get('owner')}")
        if row.get("owner") == "rule_input" and row.get("kind") != "rule_required_field":
            raise ValueError(f"rule input is not a rule field: {row['id']}")
        if row.get("unit_status") == "declared" and (not row.get("unit") or not row.get("unit_basis")):
            raise ValueError(f"declared field unit has no authority: {row['id']}")
        expected_unit = unit_policy(row["source_family"], row["field_path"], row["value_types"])
        if any(row.get(key) != value for key, value in expected_unit.items()):
            raise ValueError(f"field unit differs from source registry: {row['id']}")
        expected_coordinate = coordinate_policy(row["source_family"], row["field_path"])
        if any(row.get(key) != value for key, value in expected_coordinate.items()):
            raise ValueError(f"field coordinate declaration differs from source registry: {row['id']}")
        if row.get("source_time_role") != source_time_role(row["source_family"], row["field_path"]):
            raise ValueError(f"field source time role differs from source registry: {row['id']}")
        if row.get("value_kind") not in {"numeric", "categorical", "boolean", "null", "container", "mixed"}:
            raise ValueError(f"field value kind missing: {row['id']}")
    statuses = Counter()
    partitions = [partition] if partition is not None else list(iter_partitions())
    if partition is None and vocabulary[0]["materialized_samples"] != len(partitions):
        raise ValueError("vocabulary materialized sample count differs from partitions")
    validator = GraphValidator() if graphs else None
    for path in partitions:
        scope = load_scope(path)
        if scope["contract_version"] != CONTRACT_VERSION:
            raise ValueError(f"partition contract version differs: {path}")
        statuses[scope["materialization_status"]] += 1
        if validator:
            verify_partition(path, validator)
    return {"contract_rows": len(contract), "node_categories": len(categories),
            "role_categories": len(role_rows), "field_rows": len(fields),
            "normalized_fields": len(normalized_codes),
            "field_definitions": len(definitions),
            "partitions": sum(statuses.values()), "complete_partitions": statuses["complete"],
            "validated_partitions": sum(statuses.values()) if graphs else 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--partition", type=Path)
    args = parser.parse_args()
    print(verify(graphs=args.graphs, partition=args.partition))
