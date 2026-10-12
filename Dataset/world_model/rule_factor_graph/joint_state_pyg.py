"""Standalone tensor adapter for bindings produced after observation isolation.

Source identity, source evidence and dependency/cache declarations are returned
as external metadata. Learner tensors contain anonymous local references only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch_geometric.data import HeteroData

from .joint_state_binding import BindingError, BindingResult, STATUSES
from .state_dependencies import NODE_KINDS, RELATIONS
from Dataset.world_model.graph.prediction import PredictionGraph, to_pyg


class JointStateGraph(PredictionGraph):
    """Standard heterogeneous edges offset all local cell/frame/node references.

    PredictionGraph already offsets its query_entity_index. Global semantic
    codes use *_code names, so PyG does not increment their numeric values.
    """


@dataclass(frozen=True)
class JointStatePygResult:
    graph: JointStateGraph
    metadata: Mapping[str, Any]


def _edges(pairs):
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()


def to_fixed_rule_result_pyg(source: JointStatePygResult, evaluations):
    """Project evaluated truths; exact identities remain external metadata."""
    from .registry_runtime import FixedRuleEvaluation
    from .unified_contracts import TRUTH_DOMAIN

    if not isinstance(source, JointStatePygResult):
        raise TypeError("evaluated projection requires JointStatePygResult")
    rows = tuple(evaluations)
    if not rows or any(not isinstance(row, FixedRuleEvaluation) for row in rows):
        raise TypeError("evaluated projection requires nonempty FixedRuleEvaluation records")
    binding = source.metadata["binding"]
    graph = source.graph.clone()
    if "FixedRuleResult" in graph.node_types:
        raise BindingError("evaluated result store already exists")

    predicates = tuple(sorted({row.predicate_id for row in rows}))
    roles = tuple(sorted({item.role for row in rows for item in row.bindings}))
    classes = tuple(sorted({item.entity.ontology_class for row in rows for item in row.bindings}))
    aliases = {alias: index for index, alias in enumerate(binding.aliases)}
    edges, positions, role_codes, class_codes, operands = [], [], [], [], []
    for index, row in enumerate(rows):
        if row.output.tick != binding.cutoff_tick or row.truth not in TRUTH_DOMAIN:
            raise BindingError("evaluated result has the wrong current sample or truth")
        if tuple(item.entity for item in row.bindings) != row.output.owner:
            raise BindingError("evaluated result owners differ from ordered role bindings")
        for position, item in enumerate(row.bindings):
            edges.append((aliases[item.entity.entity_id], index))
            positions.append(position)
            role_codes.append(roles.index(item.role))
            class_codes.append(classes.index(item.entity.ontology_class))
        operands.extend((cell_index, index) for cell_index in row.operand_cell_indices)

    store = graph["FixedRuleResult"]
    store.num_nodes = len(rows)
    store.node_present = torch.ones(len(rows), dtype=torch.bool)
    store.predicate_code = torch.tensor([predicates.index(row.predicate_id) for row in rows], dtype=torch.long)
    store.truth_code = torch.tensor([TRUTH_DOMAIN.index(row.truth) for row in rows], dtype=torch.long)
    store.log_support = torch.tensor([row.log_support for row in rows], dtype=torch.float64)
    store.sample_relative_tick = torch.tensor([row.output.tick - binding.cutoff_tick for row in rows], dtype=torch.long)
    store.availability_present = torch.tensor([row.availability_tick is not None for row in rows], dtype=torch.bool)
    store.availability_relative_tick = torch.tensor([
        -1 if row.availability_tick is None else row.availability_tick - binding.cutoff_tick
        for row in rows], dtype=torch.long)
    owners = graph["ObservedEntity", "binds_evaluated_role", "FixedRuleResult"]
    owners.edge_index = _edges(edges)
    owners.role_position = torch.tensor(positions, dtype=torch.long)
    owners.role_code = torch.tensor(role_codes, dtype=torch.long)
    owners.owner_class_code = torch.tensor(class_codes, dtype=torch.long)
    graph["SourceCell", "supports_evaluated_result", "FixedRuleResult"].edge_index = _edges(operands)
    return JointStatePygResult(graph, {
        **source.metadata, "fixed_rule_results": rows,
        "result_predicates": predicates, "result_roles": roles,
        "result_owner_classes": classes, "result_truth_domain": TRUTH_DOMAIN,
        "result_policy": "evaluated_fixed_receiver_rules_no_labels",
    })


def _numeric(value):
    if value is None or isinstance(value, Mapping):
        return (), False
    values = tuple(value) if isinstance(value, tuple) else (value,)
    if any(type(item) not in (bool, int, float) for item in values):
        return (), False
    if len(values) not in (1, 3):
        raise BindingError("tensor adapter supports scalar or three-component source quantities")
    try:
        numeric = tuple(float(item) for item in values)
    except (OverflowError, ValueError) as exc:
        raise BindingError("source/parameter value cannot be encoded as float32") from exc
    limit = torch.finfo(torch.float32).max
    if any(not math.isfinite(item) or abs(item) > limit for item in numeric):
        raise BindingError("source/parameter value is outside finite float32 range")
    return numeric, True


def _plain_input(value):
    """Materialize fresh containers from the immutable observation context."""
    if isinstance(value, Mapping):
        return {key: _plain_input(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_input(item) for item in value]
    return value


def _same_input(actual, expected):
    if isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
                and actual.shape == expected.shape
                and (not actual.is_floating_point() or bool(torch.isfinite(actual).all()))
                and torch.equal(actual.detach().cpu(), expected))
    if isinstance(expected, Mapping):
        return (isinstance(actual, Mapping) and set(actual) == set(expected)
                and all(_same_input(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_same_input(a, b) for a, b in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def _bound_prediction_graph(binding, base_graph):
    try:
        expected = to_pyg({"input": _plain_input(binding.isolated_input), "trace": None})
    except (OverflowError, RuntimeError) as exc:
        raise BindingError("isolated observations cannot be encoded as finite prediction tensors") from exc
    if base_graph is None:
        base_graph = expected
    if not isinstance(base_graph, PredictionGraph):
        raise BindingError("base_graph must be an isolated PredictionGraph, never a target graph")
    actual = base_graph.to_dict()
    canonical = expected.to_dict()
    if set(actual) != set(canonical):
        raise BindingError("prediction graph stores differ from immutable isolated input")
    for store_key, store in canonical.items():
        if not _same_input(actual[store_key], store):
            raise BindingError(f"prediction graph store {store_key!r} differs from immutable isolated input")
    # from_dict retains all globals, including query indices and their __inc__ semantics.
    return JointStateGraph.from_dict(base_graph.clone().to_dict())


def to_joint_state_pyg(
    binding: BindingResult, *, base_graph: HeteroData | None = None,
) -> JointStatePygResult:
    """Add bound dependency stores to a cloned isolated prediction graph.

    Target graphs and graphs already carrying these stores are rejected.
    External metadata retains the exact plan, all edge/cache policies, cell
    identities and coverage; it is never attached as a tensor feature.
    """
    if not isinstance(binding, BindingResult):
        raise TypeError("the adapter requires an isolated BindingResult")
    graph = _bound_prediction_graph(binding, base_graph)
    # PyG needs a per-node tensor to reconstruct stores when unbatching;
    # num_nodes alone is not represented in its slice dictionary.
    graph["ObservedEntity"].node_present = torch.ones(len(binding.aliases), dtype=torch.bool)
    node_ids = tuple(sorted(binding.plan.nodes))
    codes = {node_id: code for code, node_id in enumerate(node_ids)}
    nodes = [binding.plan.nodes[node_id] for node_id in node_ids]
    data_types = tuple(sorted({node.data_type for node in nodes}))
    domains = tuple(sorted({node.domain.kind for node in nodes}))
    units = tuple(sorted({node.unit_status for node in nodes}))
    graph["DependencyNode"].num_nodes = len(nodes)
    graph["DependencyNode"].semantic_code = torch.arange(len(nodes), dtype=torch.long)
    graph["DependencyNode"].kind_code = torch.tensor([NODE_KINDS.index(node.kind) for node in nodes])
    graph["DependencyNode"].data_type_code = torch.tensor([data_types.index(node.data_type) for node in nodes])
    graph["DependencyNode"].domain_code = torch.tensor([domains.index(node.domain.kind) for node in nodes])
    graph["DependencyNode"].unit_status_code = torch.tensor([units.index(node.unit_status) for node in nodes])
    dependency = graph["DependencyNode", "dependency", "DependencyNode"]
    dependency.edge_index = _edges([(codes[edge.source], codes[edge.target]) for edge in binding.plan.edges])
    dependency.relation_code = torch.tensor([RELATIONS.index(edge.relation) for edge in binding.plan.edges])
    dependency.use_code = torch.tensor([("value", "guard").index(edge.use) for edge in binding.plan.edges])
    policy_names = ("time_grid", "temporal_lookup", "update_policy", "absence_policy")
    policy_codes = {}
    for name in policy_names:
        vocabulary = tuple(sorted({getattr(edge, name) for edge in binding.plan.edges}))
        dependency[name + "_code"] = torch.tensor([vocabulary.index(getattr(edge, name))
                                                   for edge in binding.plan.edges])
        policy_codes[name] = vocabulary
    graph["SourceCell"].num_nodes = len(binding.cells)
    values = []
    present = []
    encoded = []
    read_nodes = {}
    for read in binding.reads:
        read_nodes.setdefault(read.cell_index, []).append(binding.plan.nodes[read.node_id])
    for index, cell in enumerate(binding.cells):
        if cell.status == "admitted" and any(
                node.data_type == "numeric" for node in read_nodes.get(index, ())):
            if type(cell.value) not in (int, float):
                raise BindingError("admitted numeric SourceCell requires a finite scalar value")
        numeric, supported = _numeric(cell.value)
        values.append([*numeric, *([0.] * (3 - len(numeric)))])
        present.append([i < len(numeric) for i in range(3)])
        encoded.append(supported)
    graph["SourceCell"].x = torch.tensor(values, dtype=torch.float32).reshape(-1, 3)
    # Preserve comparison precision separately from float32 encoder features.
    graph["SourceCell"].exact_x = torch.tensor(values, dtype=torch.float64).reshape(-1, 3)
    categories = tuple(sorted({cell.value for cell in binding.cells if isinstance(cell.value, str)}))
    graph["SourceCell"].category_code = torch.tensor([
        categories.index(cell.value) if isinstance(cell.value, str) else -1 for cell in binding.cells],
        dtype=torch.long)
    if not bool(torch.isfinite(graph["SourceCell"].x).all()):
        raise BindingError("encoded source/parameter tensors must be finite")
    graph["SourceCell"].value_present = torch.tensor(present, dtype=torch.bool).reshape(-1, 3)
    graph["SourceCell"].numeric_encoding_supported = torch.tensor(encoded, dtype=torch.bool)
    graph["SourceCell"].status_code = torch.tensor([STATUSES.index(cell.status) for cell in binding.cells])
    graph["SourceCell"].relative_tick = torch.tensor([cell.key.tick - binding.cutoff_tick
                                                     for cell in binding.cells], dtype=torch.long)
    graph["SourceCell"].availability_relative_tick = torch.tensor([
        -1 if cell.availability_tick is None else cell.availability_tick - binding.cutoff_tick
        for cell in binding.cells], dtype=torch.long)
    graph["SourceCell"].availability_present = torch.tensor([
        cell.availability_tick is not None for cell in binding.cells], dtype=torch.bool)
    # Project admitted container leaves through exact receiver feature fields.
    # Their owner, source clock and status remain on the containing SourceCell.
    from Dataset.world_model.graph.observation_logs import ENUM_VALUES, FEATURE_UNITS, FEATURES
    primitive_units = tuple(sorted(set(FEATURE_UNITS.values())))
    receiver_containers = {field.rsplit(".", depth)[0]
                           for field in FEATURES for depth in range(1, field.count(".") + 1)}
    primitive_records = []
    primitive_values = []
    primitive_unit_codes = []
    primitive_category_codes = []
    primitive_present = []
    for index, cell in enumerate(binding.cells):
        if cell.status != "admitted" or not isinstance(cell.value, Mapping):
            continue
        if cell.key.field not in FEATURE_UNITS and cell.key.field not in receiver_containers:
            continue
        pending = [(cell.key.field, cell.value)]
        projected_fields = set()
        while pending:
            path, value = pending.pop(0)
            if isinstance(value, Mapping):
                if path in FEATURE_UNITS:
                    raise BindingError(f"declared receiver feature is not a primitive: {path}")
                if path in receiver_containers:
                    if not value:
                        raise BindingError(f"admitted receiver container has no primitive evidence: {path}")
                    for name in value:
                        if type(name) is not str or not name or "." in name:
                            raise BindingError(f"receiver container has a noncanonical field key: {path}")
                        child = f"{path}.{name}"
                        if child not in FEATURE_UNITS and child not in receiver_containers:
                            raise BindingError(f"receiver container has an undeclared field: {child}")
                pending.extend((f"{path}.{name}" if path else str(name), item)
                               for name, item in value.items())
                continue
            if path in receiver_containers:
                raise BindingError(f"declared receiver container is not a mapping: {path}")
            if path not in FEATURE_UNITS:
                continue
            if path in projected_fields:
                raise BindingError(f"primitive evidence fields collide: {path}")
            projected_fields.add(path)
            unit = FEATURE_UNITS[path]
            if unit == "bool":
                if type(value) is not bool:
                    raise BindingError(f"primitive evidence field is not boolean: {path}")
                scalar, category = (1.0 if value else 0.0), -1
            elif unit in ENUM_VALUES:
                if not isinstance(value, str) or value not in ENUM_VALUES[unit]:
                    raise BindingError(f"primitive evidence field is not a declared enum value: {path}")
                scalar, category = 0.0, ENUM_VALUES[unit].index(value)
            else:
                if type(value) not in (int, float):
                    raise BindingError(f"primitive evidence field is not numeric: {path}")
                numeric, supported = _numeric(value)
                if not supported or len(numeric) != 1:
                    raise BindingError(f"primitive evidence field is not a finite scalar: {path}")
                scalar, category = numeric[0], -1
            primitive_records.append({"cell_index": index, "field": path, "unit": unit, "value": value})
            primitive_values.append(scalar)
            primitive_unit_codes.append(primitive_units.index(unit))
            primitive_category_codes.append(category)
            primitive_present.append(True)
    graph["PrimitiveEvidence"].num_nodes = len(primitive_records)
    graph["PrimitiveEvidence"].node_present = torch.ones(len(primitive_records), dtype=torch.bool)
    graph["PrimitiveEvidence"].field_code = torch.tensor(
        [FEATURES.index(record["field"]) for record in primitive_records], dtype=torch.long)
    graph["PrimitiveEvidence"].unit_code = torch.tensor(primitive_unit_codes, dtype=torch.long)
    graph["PrimitiveEvidence"].category_code = torch.tensor(primitive_category_codes, dtype=torch.long)
    graph["PrimitiveEvidence"].x = torch.tensor(primitive_values, dtype=torch.float32).reshape(-1, 1)
    graph["PrimitiveEvidence"].exact_x = torch.tensor(primitive_values, dtype=torch.float64).reshape(-1, 1)
    graph["PrimitiveEvidence"].value_present = torch.tensor(primitive_present, dtype=torch.bool)
    graph["SourceCell", "projects_primitive", "PrimitiveEvidence"].edge_index = _edges([
        (record["cell_index"], position) for position, record in enumerate(primitive_records)])
    graph["PrimitiveEvidence", "rev_projects_primitive", "SourceCell"].edge_index = _edges([
        (position, record["cell_index"]) for position, record in enumerate(primitive_records)])
    if not bool(torch.isfinite(graph["PrimitiveEvidence"].x).all()):
        raise BindingError("encoded source/parameter tensors must be finite")
    aliases = {alias: i for i, alias in enumerate(binding.aliases)}
    graph["ObservedEntity", "owns_source_cell", "SourceCell"].edge_index = _edges([
        (aliases[owner], i) for i, cell in enumerate(binding.cells) for owner in cell.key.owner_aliases])
    frames = tuple(dict.fromkeys(cell.frame_owner for cell in binding.cells if cell.frame_owner is not None))
    frame_codes = {key: i for i, key in enumerate(frames)}
    graph["CoherentFrame"].num_nodes = len(frames)
    graph["CoherentFrame"].relative_tick = torch.tensor([key.tick - binding.cutoff_tick for key in frames], dtype=torch.long)
    graph["CoherentFrame", "contains", "SourceCell"].edge_index = _edges([
        (frame_codes[cell.frame_owner], i) for i, cell in enumerate(binding.cells) if cell.frame_owner is not None])
    graph["JointScene"].num_nodes = 1
    graph["JointScene"].node_present = torch.ones(1, dtype=torch.bool)
    graph["JointScene", "owns", "CoherentFrame"].edge_index = _edges([(0, i) for i in range(len(frames))])
    graph["JointScene", "owns", "SourceCell"].edge_index = _edges([(0, i) for i in range(len(binding.cells))])
    graph["SourceCell", "binds", "DependencyNode"].edge_index = _edges(sorted({
        (read.cell_index, codes[read.node_id]) for read in binding.reads}))
    graph["SourceCell", "upstream_of", "DependencyNode"].edge_index = _edges(sorted({
        (read.cell_index, codes[binding.plan.specs[read.predicate_id].predicate_node_id])
        for read in binding.reads}))
    # Keep external trace/episode/path/branch identity out of the graph.
    external = {
        "node_ids": node_ids, "data_types": data_types, "domains": domains,
        "unit_statuses": units, "statuses": STATUSES, "temporal_policy_codes": policy_codes,
        "plan": binding.plan, "cells": binding.cells, "reads": binding.reads,
        "selected_predicates": binding.selected_predicates, "coverage": binding.coverage(),
        "alias_order": binding.aliases, "frame_owners": frames,
        "facility_projections": binding.facility_projections,
        "pose_artifact": binding.pose_artifact, "populations": binding.populations,
        "weather_projection": binding.weather_projection,
        "communication_profile": binding.communication_profile,
        "primitive_evidence": primitive_records, "primitive_units": primitive_units,
        "source_categories": categories, "binding": binding,
    }
    return JointStatePygResult(graph, external)
