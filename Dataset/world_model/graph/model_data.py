"""Convert canonical Parquet windows into PyG HeteroData; labels stay separate."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch_geometric.data import HeteroData

from Dataset.world_model.graph.contract import CONTRACT_VERSION, NODE_TYPES, REPO, iter_jsonl
from Dataset.world_model.graph.entity_identity import resolve_value_identity
from Dataset.world_model.graph.field_contract import DEFAULT_DEFINITIONS, canonical_numeric, sample_tick_clock
from Dataset.world_model.graph.field_catalog import projection_owners, validate_normalizations
from Dataset.world_model.graph.records import load_scope
from Dataset.world_model.graph.field_paths import (
    WORLD_OBSERVATION_VALUE_PATH, source_value_at_tokens,
    world_observation_path_key,
)
from Dataset.world_model.graph.sample_identity import (
    identity_envelope, node_source_scope,
)
from Dataset.world_model.graph.source_projection import (
    _requires_source_projection, _source_family, project_node_values,
    projected_catalog_binding, projected_numeric_authority,
)
from Dataset.world_model.graph.vocabulary import DEFAULT_VOCAB


TRUTH_CODE = {"false": 0, "true": 1, "unknown": 2, "out_of_scope": 3}


class TargetGraph(HeteroData):
    """Keep global field codes while batching local numeric references."""

    def __inc__(self, key, value, store=None, *args, **kwargs):
        if key == "numeric_owner_index":
            return store.num_nodes
        if key == "numeric_entity_index":
            return self["EntityIdentity"].num_nodes if "EntityIdentity" in self.node_types else 0
        return super().__inc__(key, value, store, *args, **kwargs)


def _numeric_fields(node: dict, normalized_codes: dict[str, int],
                    catalog: dict[str, dict], scope: dict):
    family, _ = _source_family(node)
    source_scope = node_source_scope(node, family, scope, require_stored=True)
    for projected in project_node_values(node, catalog=catalog, scope=source_scope,
                                         owner_scope=scope, require_stored_scope=True,
                                         numeric_only=True):
        if (projected.value_status != "present" or
                projected.structure_status not in {"source_only", "resolved"} or
                type(projected.value) not in {int, float}):
            continue
        if not math.isfinite(projected.value):
            raise ValueError(f"nonfinite normalized numeric value: {projected.normalized_id}")
        normalized_index = projected_catalog_binding(projected, catalog)
        if normalized_codes.get(projected.normalized_id) != normalized_index:
            raise ValueError(f"normalized field index differs: {projected.normalized_id}")
        owners = set()
        for source in projected.sources:
            leaf = source.field_value
            key = source.source_field_id
            if key not in catalog:
                raise ValueError(f"numeric field absent from frozen catalog: {key}")
            if (family == "arm_predicate_truth" and
                    (leaf.field_path == WORLD_OBSERVATION_VALUE_PATH or
                     leaf.source_path_key is not None)):
                branch = world_observation_path_key(family, leaf.field_path)
                declared = catalog[key]
                if (branch is None or branch != leaf.source_path_key or
                        declared.get("source_path_key") != branch or
                        leaf.predicate_id not in declared.get("predicate_ids", ())):
                    raise ValueError(
                        f"numeric world observation lacks one qualified model field: "
                        f"{leaf.predicate_id}:{leaf.source_path_key}:{key}")
            allowed_owners = projection_owners(catalog[key])
            owner = node["node_type"] if node["node_type"] in allowed_owners else catalog[key]["owner"]
            owners.add(owner)
            if (_requires_source_projection(node, family, leaf.field_path)
                    and owner != node["node_type"]):
                raise ValueError(f"numeric source field owner differs from projection: {key}")
        if node["node_type"] in owners:
            if owners != {node["node_type"]}:
                raise ValueError(f"normalized source owners conflict: {projected.normalized_id}")
            source = projected.sources[0]
            yield normalized_index, projected.value, source.field_value, projected, source_scope


def load_window(partition: Path, start_tick: int, end_tick: int) -> tuple[dict, list[dict], list[dict]]:
    scope = load_scope(partition)
    nodes: list[dict] = []
    tick = ds.field("tick")
    node_filter = tick.is_null() | ((tick >= start_tick) & (tick <= end_tick))
    scanner = ds.dataset(partition / "nodes.parquet", format="parquet").scanner(
        filter=node_filter, batch_size=8192)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            row["attrs"] = json.loads(row.pop("attrs_json"))
            nodes.append(row)
    complete_window = len(nodes) == scope["counts"]["nodes"]
    endpoint_ids = (None if complete_window else
                    pa.array(sorted(row["node_id"] for row in nodes), type=pa.string()))
    edges: list[dict] = []
    for batch in pq.ParquetFile(partition / "edges.parquet").iter_batches(batch_size=8192):
        if not complete_window:
            endpoints_in_window = pc.and_(pc.is_in(batch.column("source"), value_set=endpoint_ids),
                                          pc.is_in(batch.column("target"), value_set=endpoint_ids))
            batch = batch.filter(endpoints_in_window)
        for row in batch.to_pylist():
            row["attrs"] = json.loads(row.pop("attrs_json"))
            edges.append(row)
    return scope, nodes, edges


def convert_window(partition: Path, start_tick: int, end_tick: int,
                   *, vocab_path: Path = DEFAULT_VOCAB,
                   field_catalog_path: Path | None = None) -> tuple[HeteroData, list[dict], dict]:
    partition = partition.resolve()
    scope, nodes, edges = load_window(partition, start_tick, end_tick)
    if scope["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"partition contract version differs: {partition}")
    vocabulary = [row for _, row in iter_jsonl(vocab_path)]
    header = [row for row in vocabulary if row["kind"] == "vocabulary"]
    if len(header) != 1 or header[0]["contract_version"] != CONTRACT_VERSION:
        raise ValueError("category vocabulary missing or contract version differs")
    category_codes = {(row["node_type"], row["category_id"]): row["index"]
                      for row in vocabulary if row["kind"] == "node_category"}
    role_codes = {row["role"]: row["index"] for row in vocabulary if row["kind"] == "edge_role"}
    context_rows = [row for _, row in iter_jsonl(REPO / "design/belief/contract.jsonl")]
    if not context_rows or context_rows[0].get("version") != CONTRACT_VERSION:
        raise ValueError("graph contract version differs")
    anchor_kind_codes = {row["id"]: row["index"] for row in context_rows
                         if row["kind"] == "context_anchor_kind"}
    context_source_family_codes = {row["id"]: row["index"] for row in context_rows
                                   if row["kind"] == "context_category"}
    if field_catalog_path is None or not field_catalog_path.is_file():
        raise ValueError("model conversion requires the frozen field catalog")
    sample_clock = sample_tick_clock(scope["episode_id"])
    field_rows = [row for _, row in iter_jsonl(field_catalog_path)]
    if not field_rows or any(row.get("contract_version") != CONTRACT_VERSION for row in field_rows):
        raise ValueError("field catalog version differs")
    field_codes = {row["id"]: row["index"] for row in field_rows}
    normalized_codes = validate_normalizations(field_rows)
    catalog_rows = {row["id"]: row for row in field_rows}
    definition_rows = [row for _, row in iter_jsonl(DEFAULT_DEFINITIONS)]
    if not definition_rows or any(row.get("contract_version") != CONTRACT_VERSION for row in definition_rows):
        raise ValueError("field definitions version differs")
    definition_codes = {row["id"]: row["index"] for row in definition_rows}
    if (len(definition_codes) != len(definition_rows) or
            set(definition_codes.values()) != set(range(len(definition_rows))) or
            {row["semantic_id"] for row in field_rows} - set(definition_codes)):
        raise ValueError("field definition references or indices differ")
    if len(field_codes) != len(field_rows) or len(set(field_codes.values())) != len(field_rows):
        raise ValueError("field catalog IDs or indices are not unique")
    if set(field_codes.values()) != set(range(len(field_rows))):
        raise ValueError("field catalog indices are not contiguous")
    field_units: list[str | None] = [None] * len(normalized_codes)
    field_source_units: list[str | None] = [None] * len(normalized_codes)
    unit_sources: dict[int, set[tuple[str | None, str | None]]] = defaultdict(set)
    for row in field_rows:
        for binding in row.get("normalizations", []):
            unit_sources[binding["normalized_index"]].add(
                (row["canonical_unit"], row["source_unit"]))
    for normalized_index, units in unit_sources.items():
        if len(units) == 1:
            field_units[normalized_index], field_source_units[normalized_index] = next(iter(units))
    grouped_nodes: dict[str, list[dict]] = defaultdict(list)
    for row in nodes:
        grouped_nodes[row["node_type"]].append(row)
    identity_nodes = sorted(grouped_nodes.get("EntityIdentity", []),
                            key=lambda row: (row["tick"] if row["tick"] is not None else -1, row["node_id"]))
    identity_by_raw_id: dict[tuple[str, str], tuple[int, dict]] = {}
    for identity_index, identity_node in enumerate(identity_nodes):
        raw_id = identity_node["attrs"].get("entity_id")
        if not isinstance(raw_id, str) or not raw_id:
            raise ValueError(f"identity node has no raw entity ID: {identity_node['node_id']}")
        namespace = identity_node["attrs"].get("identity_namespace")
        if namespace not in {"sample", "compute_node", "sumo_tls_controller"}:
            raise ValueError(f"identity node lacks a declared namespace: {identity_node['node_id']}")
        key = namespace, raw_id
        if key in identity_by_raw_id:
            raise ValueError(f"sample has ambiguous scoped entity ID: {key}")
        identity_by_raw_id[key] = identity_index, identity_node
    data = TargetGraph()
    index_rows: list[dict] = []
    node_index: dict[str, tuple[str, int]] = {}
    token_sequences: dict[tuple, tuple] = {}
    unresolved_numeric: Counter[tuple[str, str]] = Counter()
    numeric_count = canonical_count = 0

    def tokens(sequence):
        key = tuple(sys.intern(token) if isinstance(token, str) else token
                    for token in sequence)
        return token_sequences.setdefault(key, key)

    for node_type in NODE_TYPES:
        group = sorted(grouped_nodes.get(node_type, []), key=lambda row: (row["tick"] if row["tick"] is not None else -1, row["node_id"]))
        if not group:
            continue
        store = data[node_type]
        store.num_nodes = len(group)
        store.category_id = torch.tensor([category_codes[(node_type, row["category_id"])] for row in group], dtype=torch.long)
        store.tick = torch.tensor([row["tick"] if row["tick"] is not None else -1 for row in group], dtype=torch.long)
        store.tick_present = torch.tensor([row["tick"] is not None for row in group], dtype=torch.bool)
        store.record_id = [row["node_id"] for row in group]
        store.source_file = [row["source_file"] for row in group]
        store.source_line = [row["source_line"] for row in group]
        store.source_replay_status = [row["source_replay_status"] for row in group]
        store.payload_json = [json.dumps(row["attrs"], ensure_ascii=False, separators=(",", ":")) for row in group]
        if node_type == "ContextAnchor":
            store.anchor_kind_id = torch.tensor(
                [anchor_kind_codes[row["attrs"]["anchor_kind"]] for row in group], dtype=torch.long)
            store.source_family_id = torch.tensor(
                [context_source_family_codes[row["attrs"]["source_family"]] for row in group], dtype=torch.long)
        if node_type == "PredicateAssertion":
            store.target_y = torch.tensor([TRUTH_CODE[row["attrs"]["truth_status"]] for row in group], dtype=torch.long)
        owner_index: list[int] = []
        field_index: list[int] = []
        element_index: list[int] = []
        element_positions: list[list[int]] = []
        raw_paths: list[str] = []
        source_tokens: list[list[str | int]] = []
        input_tokens: list[list[str | int]] = []
        predicate_ids: list[str | None] = []
        source_path_keys: list[str | None] = []
        source_branches: list[str | None] = []
        source_subtypes: list[str | None] = []
        snapshot_ticks: list[int] = []
        snapshot_tick_present: list[bool] = []
        value_ticks: list[int] = []
        value_tick_present: list[bool] = []
        time_roles: list[str | None] = []
        endpoint_roles: list[str | None] = []
        source_values: list[int | float] = []
        value_types: list[str] = []
        normalized_ids: list[str] = []
        normalized_values: list[int | float] = []
        normalized_value_types: list[str] = []
        value_statuses: list[str] = []
        structure_statuses: list[str] = []
        binding_slots: list[str | None] = []
        declared_shapes: list[list[int | None] | None] = []
        observed_shapes: list[list[int | None] | None] = []
        normalized_shapes: list[list[int | None] | None] = []
        record_observed: list[bool | None] = []
        payload_tokens: list[list[str | int]] = []
        payload_positions: list[list[int]] = []
        payload_shapes: list[list[int]] = []
        envelope_tokens: list[list[str | int]] = []
        all_sources: list[list[dict]] = []
        source_field_indices: list[list[int]] = []
        field_value: list[float] = []
        canonical_present: list[bool] = []
        conversion_reason: list[str | None] = []
        semantic_index: list[int] = []
        semantic_present: list[bool] = []
        semantic_binding_statuses: list[str] = []
        entity_index: list[int] = []
        entity_present: list[bool] = []
        entity_raw_id: list[str | None] = []
        entity_namespace: list[str | None] = []
        identity_status: list[str] = []
        identity_reason: list[str | None] = []
        identity_basis: list[str | None] = []
        identity_basis_path: list[str | None] = []
        target_scopes: list[str | None] = []
        numeric_source_file: list[str] = []
        numeric_source_line: list[int | None] = []
        numeric_record_id: list[str] = []
        numeric_source_scope: list[dict | None] = []
        for index, row in enumerate(group):
            node_index[row["node_id"]] = node_type, index
            index_rows.append({"record_kind": "node", "record_id": row["node_id"],
                               "tensor_type": node_type, "local_index": index})
            for numeric_index, value, leaf, projected, source_scope in _numeric_fields(
                    row, normalized_codes, catalog_rows, scope):
                owner_index.append(index)
                field_index.append(numeric_index)
                element_index.append(leaf.element_positions[-1] if leaf.element_positions else -1)
                element_positions.append(tokens(leaf.element_positions))
                raw_paths.append(sys.intern(leaf.raw_path))
                source_tokens.append(tokens(leaf.source_tokens))
                input_tokens.append(tokens(leaf.input_tokens))
                predicate_ids.append(leaf.predicate_id)
                source_path_keys.append(leaf.source_path_key)
                source_branches.append(leaf.source_branch)
                source_subtypes.append(leaf.source_subtype)
                snapshot_ticks.append(leaf.snapshot_tick if leaf.snapshot_tick is not None else -1)
                snapshot_tick_present.append(leaf.snapshot_tick is not None)
                resolved_tick = (leaf.value_tick if leaf.value_tick is not None or leaf.time_role is not None
                                 else row["tick"])
                value_ticks.append(resolved_tick if resolved_tick is not None else -1)
                value_tick_present.append(resolved_tick is not None)
                time_roles.append(leaf.time_role or ("owner_node_tick" if row["tick"] is not None else None))
                endpoint_roles.append(leaf.endpoint_role)
                source_values.append(leaf.value)
                value_types.append(type(leaf.value).__name__)
                normalized_ids.append(sys.intern(projected.normalized_id))
                normalized_values.append(value)
                normalized_value_types.append(projected.value_type)
                value_statuses.append(projected.value_status)
                structure_statuses.append(projected.structure_status)
                binding_slots.append(projected.binding_slot)
                declared_shapes.append(tokens(projected.declared_shape)
                                       if projected.declared_shape is not None else None)
                observed_shapes.append(tokens(projected.observed_shape)
                                       if projected.observed_shape is not None else None)
                normalized_shapes.append(tokens(projected.normalized_shape)
                                         if projected.normalized_shape is not None else None)
                record_observed.append(projected.record_observed)
                payload_tokens.append(tokens(projected.payload_tokens))
                payload_positions.append(tokens(projected.payload_positions))
                payload_shapes.append(tokens(projected.payload_shape))
                envelope_tokens.append(tokens(projected.envelope_tokens))
                # Field declarations live in the shared catalog. Retain only
                # each original value and its exact binding here, including
                # secondary sources of a joined normalization.
                source_rows = [{
                    "source_field_id": catalog_rows[source.source_field_id]["id"],
                    "source_index": field_codes[source.source_field_id],
                    "source_tokens": tokens(source.field_value.source_tokens),
                    "input_tokens": tokens(source.field_value.input_tokens),
                    "value": source.field_value.value,
                } for source in projected.sources]
                all_sources.append(source_rows)
                source_field_indices.append([source_row["source_index"] for source_row in source_rows])
                mapping, coordinate_context, semantic_id, semantic_status = projected_numeric_authority(
                    projected, catalog_rows, row["source_file"], row["attrs"]["raw"])
                if semantic_status == "declared":
                    converted, reason = canonical_numeric(
                        value, mapping, sample_clock=sample_clock,
                        coordinate_context=coordinate_context)
                else:
                    converted, reason = None, "source meaning has no shared declaration"
                field_value.append(converted if converted is not None else float("nan"))
                canonical_present.append(converted is not None)
                conversion_reason.append(reason)
                numeric_count += 1
                canonical_count += converted is not None
                if converted is None:
                    for source in projected.sources:
                        unresolved_numeric[(source.source_field_id, reason)] += 1
                semantic_index.append(definition_codes[semantic_id] if semantic_id is not None else -1)
                semantic_present.append(semantic_id is not None)
                semantic_binding_statuses.append(semantic_status)
                identity = resolve_value_identity(leaf, row, identity_by_raw_id)
                entity_index.append(identity.local_index if identity.local_index is not None else -1)
                entity_present.append(identity.status == "resolved")
                entity_raw_id.append(leaf.entity_id)
                entity_namespace.append(leaf.entity_namespace)
                identity_status.append(identity.status)
                identity_reason.append(identity.reason)
                identity_basis.append(identity.basis)
                identity_basis_path.append(identity.basis_path)
                target_scopes.append(leaf.target_scope)
                numeric_source_file.append(row["source_file"])
                numeric_source_line.append(row["source_line"])
                numeric_record_id.append(row["node_id"])
                numeric_source_scope.append(source_scope)
        store.numeric_owner_index = torch.tensor(owner_index, dtype=torch.long)
        store.numeric_field_index = torch.tensor(field_index, dtype=torch.long)
        store.numeric_element_index = torch.tensor(element_index, dtype=torch.long)
        store.numeric_element_positions = element_positions
        store.numeric_raw_path = raw_paths
        store.numeric_source_tokens = source_tokens
        store.numeric_input_tokens = input_tokens
        store.numeric_predicate_id = predicate_ids
        store.numeric_source_path_key = source_path_keys
        store.numeric_source_branch = source_branches
        store.numeric_source_subtype = source_subtypes
        store.numeric_snapshot_tick = torch.tensor(snapshot_ticks, dtype=torch.long)
        store.numeric_snapshot_tick_present = torch.tensor(snapshot_tick_present, dtype=torch.bool)
        store.numeric_value_tick = torch.tensor(value_ticks, dtype=torch.long)
        store.numeric_value_tick_present = torch.tensor(value_tick_present, dtype=torch.bool)
        store.numeric_time_role = time_roles
        store.numeric_endpoint_role = endpoint_roles
        store.numeric_source_value = source_values
        store.numeric_value_type = value_types
        store.numeric_normalized_id = normalized_ids
        store.numeric_normalized_value = normalized_values
        store.numeric_normalized_value_type = normalized_value_types
        store.numeric_value_status = value_statuses
        store.numeric_structure_status = structure_statuses
        store.numeric_binding_slot = binding_slots
        store.numeric_declared_shape = declared_shapes
        store.numeric_observed_shape = observed_shapes
        store.numeric_normalized_shape = normalized_shapes
        store.numeric_record_observed = record_observed
        store.numeric_payload_tokens = payload_tokens
        store.numeric_payload_positions = payload_positions
        store.numeric_payload_shape = payload_shapes
        store.numeric_envelope_tokens = envelope_tokens
        store.numeric_sources = all_sources
        store.numeric_source_field_indices = source_field_indices
        store.numeric_value = torch.tensor(field_value, dtype=torch.float64)
        store.numeric_value_present = torch.tensor(canonical_present, dtype=torch.bool)
        store.numeric_conversion_reason = conversion_reason
        store.numeric_semantic_index = torch.tensor(semantic_index, dtype=torch.long)
        store.numeric_semantic_present = torch.tensor(semantic_present, dtype=torch.bool)
        store.numeric_semantic_binding_status = semantic_binding_statuses
        store.numeric_entity_index = torch.tensor(entity_index, dtype=torch.long)
        store.numeric_entity_present = torch.tensor(entity_present, dtype=torch.bool)
        store.numeric_entity_raw_id = entity_raw_id
        store.numeric_entity_namespace = entity_namespace
        store.numeric_identity_status = identity_status
        store.numeric_identity_reason = identity_reason
        store.numeric_identity_basis = identity_basis
        store.numeric_identity_basis_path = identity_basis_path
        store.numeric_target_scope = target_scopes
        store.numeric_source_file = numeric_source_file
        store.numeric_source_line = numeric_source_line
        store.numeric_record_id = numeric_record_id
        store.numeric_source_scope = numeric_source_scope
    grouped_edges: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in edges:
        s_type, _ = node_index[row["source"]]
        t_type, _ = node_index[row["target"]]
        grouped_edges[(s_type, row["edge_type"], t_type)].append(row)
    for edge_key, group in sorted(grouped_edges.items()):
        store = data[edge_key]
        pairs = [[node_index[row["source"]][1], node_index[row["target"]][1]] for row in group]
        store.edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()
        store.role_id = torch.tensor([role_codes[row["role"]] if row["role"] is not None else -1 for row in group], dtype=torch.long)
        store.delta_tick = torch.tensor([(row["target_tick"] - row["source_tick"])
                                         if row["source_tick"] is not None and row["target_tick"] is not None else 0
                                         for row in group], dtype=torch.long)
        store.delta_tick_present = torch.tensor([row["source_tick"] is not None and row["target_tick"] is not None
                                                 for row in group], dtype=torch.bool)
        store.record_id = [row["edge_id"] for row in group]
        store.source_file = [row["source_file"] for row in group]
        store.source_line = [row["source_line"] for row in group]
        store.source_replay_status = [row["source_replay_status"] for row in group]
        store.payload_json = [json.dumps(row["attrs"], ensure_ascii=False, separators=(",", ":")) for row in group]
        store.numeric_source_value = [row["attrs"].get("value") for row in group]
        store.numeric_unit = [row["attrs"].get("unit") for row in group]
        store.numeric_value = torch.tensor([float(row["attrs"]["value"])
                                            if type(row["attrs"].get("value")) in {int, float} else float("nan")
                                            for row in group], dtype=torch.float64)
        store.numeric_value_present = torch.tensor([type(row["attrs"].get("value")) in {int, float}
                                                    for row in group], dtype=torch.bool)
        for index, row in enumerate(group):
            index_rows.append({"record_kind": "edge", "record_id": row["edge_id"],
                               "tensor_type": list(edge_key), "local_index": index})
    metadata = {"contract_version": CONTRACT_VERSION,
                "sample_key": scope["sample_key"],
                "sample_identity": identity_envelope(scope),
                "source_graph": partition.relative_to(REPO).as_posix(),
                "source_materialization_status": scope["materialization_status"],
                "source_unresolved": scope["unresolved"],
                "source_graph_counts": scope["counts"],
                "outside_window_nodes": scope["counts"]["nodes"] - len(nodes),
                "outside_window_edges": scope["counts"]["edges"] - len(edges),
                "start_tick": start_tick, "end_tick": end_tick,
                "nodes": len(nodes), "edges": len(edges),
                "tensor_index_kinds": ["node", "edge"],
                "numeric_trace": "numeric_owner_index + numeric_field_index + numeric_sources",
                "numeric_counts": {"values": numeric_count, "canonical": canonical_count,
                                   "unresolved": numeric_count - canonical_count},
                "unresolved_numeric_fields": [
                    {"source_field_id": key, "reason": reason, "values": count}
                    for (key, reason), count in sorted(unresolved_numeric.items())],
                "category_codes": [{"node_type": kind, "category_id": category, "index": index}
                                   for (kind, category), index in category_codes.items()],
                "anchor_kind_codes": anchor_kind_codes,
                "context_source_family_codes": context_source_family_codes,
                "role_codes": role_codes, "field_codes": normalized_codes,
                "source_field_codes": field_codes,
                "field_units": field_units, "field_source_units": field_source_units,
                "definition_codes": definition_codes,
                "sample_tick_clock": sample_clock,
                "supervision_status": ("saved_branch_labels" if scope["source_kind"] == "arm" else
                    "partial_world_labels" if any(item.get("relation", {}).get("source_family") ==
                        "formal_world_truth_graph_base" for item in scope["unresolved"])
                    else "formal_world_labels"),
                "vocabulary_materialized_samples": header[0]["materialized_samples"]}
    data.graph_metadata = metadata
    return data, index_rows, metadata


def validate_saved_tensor(output: Path) -> dict[str, int]:
    """Read the persisted PyG object and its source/index correspondences."""
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    data = torch.load(output / "target_graph.pt", map_location="cpu", weights_only=False,
                      mmap=True)
    # Input file stamps describe the external resume cache.  An append-only
    # catalog extension can preserve a sample's payload while updating those
    # stamps after proving that all of its used declarations are unchanged.
    # Every payload metadata field still has to match the persisted tensor.
    payload_metadata = {key: value for key, value in metadata.items()
                        if key != "artifact_inputs"}
    tensor_metadata = {key: value for key, value in data.graph_metadata.items()
                       if key != "artifact_inputs"}
    if payload_metadata != tensor_metadata or metadata["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"tensor metadata differs: {output}")
    if data.num_nodes != metadata["nodes"] or data.num_edges != metadata["edges"]:
        raise ValueError(f"tensor node/edge counts differ: {output}")
    data.validate(raise_on_error=True)
    numeric = present = 0
    for node_type in data.node_types:
        store = data[node_type]
        if len(store.source_replay_status) != store.num_nodes:
            raise ValueError(f"tensor node source status count differs: {node_type}")
        payloads = [json.loads(payload) for payload in store.payload_json]
        count = store.numeric_value.numel()
        numeric += count
        present += int(store.numeric_value_present.sum())
        owners = store.numeric_owner_index.tolist()
        field_indices = store.numeric_field_index.tolist()
        values = store.numeric_value.tolist()
        value_present = store.numeric_value_present.tolist()
        for index in range(count):
            owner = owners[index]
            if not 0 <= owner < store.num_nodes:
                raise ValueError(f"numeric owner index differs: {node_type}:{index}")
            source_value = source_value_at_tokens(
                payloads[owner]["raw"], tuple(store.numeric_input_tokens[index]))
            saved_value = store.numeric_source_value[index]
            if type(source_value) is not type(saved_value) or source_value != saved_value:
                raise ValueError(f"saved numeric source value differs: {node_type}:{index}")
            field_index = field_indices[index]
            if metadata["field_codes"].get(store.numeric_normalized_id[index]) != field_index:
                raise ValueError(f"numeric field index differs: {node_type}:{index}")
            value = values[index]
            if value_present[index] != math.isfinite(value):
                raise ValueError(f"numeric value and presence differ: {node_type}:{index}")
            if not value_present[index] and not store.numeric_conversion_reason[index]:
                raise ValueError(f"unresolved numeric value has no reason: {node_type}:{index}")
            sources = store.numeric_sources[index]
            if [source["source_index"] for source in sources] != store.numeric_source_field_indices[index]:
                raise ValueError(f"numeric source indices differ: {node_type}:{index}")
            for source in sources:
                if metadata["source_field_codes"].get(source["source_field_id"]) != source["source_index"]:
                    raise ValueError(f"numeric original field index differs: {node_type}:{index}")
                original = source_value_at_tokens(payloads[owner]["raw"], tuple(source["input_tokens"]))
                if type(original) is not type(source["value"]) or original != source["value"]:
                    raise ValueError(f"numeric original source value differs: {node_type}:{index}")
    for edge_type in data.edge_types:
        store = data[edge_type]
        for name in ("source_file", "source_line", "source_replay_status"):
            if len(store[name]) != store.num_edges:
                raise ValueError(f"tensor edge source count differs: {edge_type}:{name}")
        values = store.numeric_value.tolist()
        edge_present = store.numeric_value_present.tolist()
        for index, available in enumerate(edge_present):
            if not available:
                continue
            attrs = json.loads(store.payload_json[index])
            original = store.numeric_source_value[index]
            if (type(original) is not type(attrs["value"]) or original != attrs["value"] or
                    float(original) != values[index]):
                raise ValueError(f"tensor edge numeric source differs: {edge_type}:{index}")
            if attrs.get("unit_status") != "declared" or attrs.get("unit") != store.numeric_unit[index]:
                raise ValueError(f"tensor edge numeric unit differs: {edge_type}:{index}")
    observed_index = set()
    for _, row in iter_jsonl(output / "tensor_index.jsonl"):
        tensor_type = tuple(row["tensor_type"]) if isinstance(row["tensor_type"], list) else row["tensor_type"]
        key = row["record_kind"], tensor_type, row["local_index"]
        if key in observed_index:
            raise ValueError(f"duplicate tensor index: {key}")
        if ((row["record_kind"] == "node" and tensor_type not in data.node_types) or
                (row["record_kind"] == "edge" and tensor_type not in data.edge_types) or
                row["record_kind"] not in {"node", "edge"}):
            raise ValueError(f"tensor index has an unknown record type: {key}")
        records = data[tensor_type].record_id
        index = row["local_index"]
        if type(index) is not int or not 0 <= index < len(records) or records[index] != row["record_id"]:
            raise ValueError(f"tensor/source record index differs: {key}")
        observed_index.add(key)
    if len(observed_index) != data.num_nodes + data.num_edges:
        raise ValueError(f"tensor/source record index differs: {output}")
    if metadata["numeric_counts"] != {"values": numeric, "canonical": present,
                                     "unresolved": numeric - present}:
        raise ValueError(f"tensor numeric counts differ: {output}")
    return {"nodes": metadata["nodes"], "edges": metadata["edges"],
            "numeric_values": numeric, "canonical_values": present,
            "unresolved_numeric_values": numeric - present}


def _tensor_inputs(partition: Path, field_catalog_path: Path) -> list[dict[str, Any]]:
    paths = [partition / name for name in ("scope.parquet", "nodes.parquet", "edges.parquet")]
    paths.extend((field_catalog_path, DEFAULT_DEFINITIONS, DEFAULT_VOCAB,
                  REPO / "design/belief/contract.jsonl",
                  Path(__file__), *(REPO / "Dataset/world_model/graph" / name for name in
                      ("field_contract.py", "field_paths.py", "field_structure.py",
                       "source_projection.py", "sample_identity.py", "entity_identity.py"))))
    return [{"path": source.resolve().relative_to(REPO).as_posix(),
             "bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}
            for source in paths]


def export_partition(partition: Path, output: Path, *,
                     field_catalog_path: Path = REPO / "design/belief/field_catalog.jsonl",
                     start_tick: int | None = None, end_tick: int | None = None,
                     replace_generated: bool = False
                     ) -> dict[str, Any]:
    """Publish one complete sample atomically; completed samples can be resumed."""
    partition = partition.resolve()
    scope = load_scope(partition)
    start = scope["start_tick"] if start_tick is None else start_tick
    end = scope["end_tick"] if end_tick is None else end_tick
    inputs = _tensor_inputs(partition, field_catalog_path)
    backup = output.with_name(output.name + ".previous")
    if backup.exists() and not output.exists():
        backup.replace(output)
    if output.exists():
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if (metadata["sample_key"] != scope["sample_key"] or
                metadata["start_tick"] != start or metadata["end_tick"] != end):
            raise ValueError(f"existing tensor belongs to another sample or window: {output}")
        if metadata.get("artifact_inputs") != inputs and not replace_generated:
            raise ValueError(f"tensor input artifacts changed; rebuild this sample: {output}")
        if not replace_generated:
            return {"sample_key": scope["sample_key"], "resumed": True,
                    **validate_saved_tensor(output)}
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        # Only the owned, incomplete sample output is discarded on resumption.
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    data, index_rows, metadata = convert_window(
        partition, start, end, field_catalog_path=field_catalog_path)
    metadata["artifact_inputs"] = inputs
    torch.save(data, temporary / "target_graph.pt")
    with (temporary / "tensor_index.jsonl").open("w", encoding="utf-8") as stream:
        for row in index_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    del data, index_rows
    gc.collect()
    result = validate_saved_tensor(temporary)
    if output.exists():
        if backup.exists():
            shutil.rmtree(backup)
        output.replace(backup)
    temporary.replace(output)
    if backup.exists():
        shutil.rmtree(backup)
    return {"sample_key": scope["sample_key"], "resumed": False, **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field-catalog", type=Path, default=REPO / "design/belief/field_catalog.jsonl")
    parser.add_argument("--replace-generated", action="store_true")
    args = parser.parse_args()
    print(export_partition(args.partition, args.output, field_catalog_path=args.field_catalog,
                           start_tick=args.start, end_tick=args.end,
                           replace_generated=args.replace_generated))


if __name__ == "__main__":
    main()
