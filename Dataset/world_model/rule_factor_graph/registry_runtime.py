"""One contract-grounded graph runtime with shared candidate neural factors."""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque
from itertools import combinations, product
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults
from .factor_network import ParameterContext, SharedFactorNetwork, operator_features
from .generic_grounding import (CandidateVariable, EntityClassCatalog, GroundedRule,
    OutputBindingAuthority, ParameterProfile, TypedEntity, VariableKey, bind_roles,
    ground_contract, soft_scope_candidates)
from .unified_contracts import (ContractError, FieldSchema, TRUTH_DOMAIN,
    canonical_class_uri, canonical_state_field, field_owner_roles, load_unified_contracts)
from .unified_inference import (FactorGraph, FiniteVariable, InferenceBudget, LogFactor,
    first_opposite_pmf, infer, joint_log_probability)
from .variable_encoder import SchemaVocabulary, SharedVariableEncoder, pack_variables


@dataclass(frozen=True)
class FixedRuleEvaluation:
    predicate_id: str
    output: VariableKey
    bindings: tuple
    truth: str
    log_support: tuple[float, ...]
    availability_tick: int | None
    operand_cell_indices: tuple[int, ...]


def evaluate_fixed_receiver_rules(fixed: SupervisedGraph, source):
    """Evaluate singleton receiver operands with recorded availability."""
    from .joint_state_pyg import JointStatePygResult

    if not isinstance(source, JointStatePygResult):
        raise TypeError("fixed evaluation requires JointStatePygResult")
    binding = source.metadata["binding"]
    if fixed.labels or fixed.anchor_tick != binding.cutoff_tick:
        raise ContractError("fixed evaluation requires label-free current receiver rules")

    cell_indices = {}
    for read in binding.reads:
        cell = binding.cells[read.cell_index]
        identity = (read.predicate_id, cell.key.artifact, cell.key.field,
                    cell.key.tick, cell.key.owners)
        cell_indices.setdefault(identity, set()).add(read.cell_index)

    evaluated = []
    for rule in fixed.rules:
        if rule.producer_binding is None or rule.inputs or rule.previous_inputs:
            raise ContractError("receiver evaluation requires explicit producer operands")
        assignment, indexes = {}, []
        _, nodes = _direct_fields(rule.contract)
        for node, (field, key) in zip(nodes, rule.producer_inputs, strict=True):
            if field != node.source_cell.field_path:
                raise ContractError("evaluated operand differs from its declared field")
            variable = fixed.variables[key]
            if len(variable.domain) != 1:
                raise ContractError("fixed receiver evaluation requires singleton operands")
            assignment[key] = variable.domain[0]
            identity = (rule.contract.predicate_id, node.source_cell.artifact,
                        node.source_cell.field_path, key.tick,
                        tuple(owner.entity_id for owner in key.owner))
            matches = cell_indices.get(identity, ())
            if len(matches) != 1:
                raise ContractError("evaluated operand requires one exact bound source cell")
            indexes.append(next(iter(matches)))

        supports = tuple(rule.log_support({**assignment, rule.output: truth})
                         for truth in TRUTH_DOMAIN)
        supported = tuple(truth for truth, value in zip(TRUTH_DOMAIN, supports)
                          if value == 0.)
        if len(supported) != 1 or any(value not in (0., -math.inf) for value in supports):
            raise ContractError("fixed rule must support one four-valued truth")

        cells = tuple(binding.cells[index] for index in indexes)
        times = tuple(cell.availability_tick for cell in cells)
        availability = (max(times) if times
                        and all(cell.status == "admitted" for cell in cells)
                        and all(time is not None for time in times) else None)
        if availability is not None and availability > binding.cutoff_tick:
            raise ContractError("evaluated receiver result exceeds admitted availability")
        evaluated.append(FixedRuleEvaluation(rule.contract.predicate_id, rule.output,
                                            rule.bindings, supported[0], supports,
                                            availability, tuple(indexes)))
    return tuple(evaluated)


@lru_cache(maxsize=1)
def _contracts():
    return load_unified_contracts()


_DIRECT_FIELDS = {}


def _direct_fields(contract):
    """Declared producer operands are fixed per contract; derive each once per process."""
    if contract.predicate_id not in _DIRECT_FIELDS:
        from .state_dependencies import direct_monitor_fields
        _DIRECT_FIELDS[contract.predicate_id] = direct_monitor_fields(contract)
    return _DIRECT_FIELDS[contract.predicate_id]


@lru_cache(maxsize=None)
def _governed_flow_profile(communication_profile, episode_id):
    from .state_dependencies import communication_row_actor_category
    return communication_profile.values(episode_id)[communication_row_actor_category()]


def _declared_owner(rule: GroundedRule, roles: Sequence[str]):
    entities = {binding.role: binding.entity for binding in rule.bindings}
    roles = tuple(roles)
    owners = [entities[role] for role in roles]
    for unordered in rule.contract.grounding_spec["unordered_role_sets"]:
        positions = [index for index, role in enumerate(roles) if role in unordered]
        if len(positions) == len(unordered):
            for index, entity in zip(positions, sorted(owners[index] for index in positions)):
                owners[index] = entity
    return tuple(owners)


def validate_rule_binding(rule: GroundedRule, variables: Mapping[VariableKey, CandidateVariable],
                          authoritative_parameters: Mapping[str, Any]) -> None:
    """Reject wrong output scopes, shared owners, times, profiles and parameters."""
    rule.validate_output_binding()
    if rule.producer_binding is not None:
        from .state_dependencies import direct_monitor_fields
        _, producer_nodes = direct_monitor_fields(rule.contract)
        for node, (field, key) in zip(producer_nodes, rule.producer_inputs):
            if key not in variables or (variables[key].schema.data_type, variables[key].schema.unit) != (
                    node.data_type, node.unit):
                raise ContractError("fixed row operand schema differs from its producer authority")
            if node.source_cell.form == "governed_row_parameter":
                value = rule.parameter_profile.parameters[field]
                if variables[key].domain != (value,) or not variables[key].observed:
                    raise ContractError("fixed row threshold differs from its governed profile")
    for field, key in rule.producer_inputs:
        if key not in variables or variables[key].schema.source != "bound_source_cell":
            raise ContractError(f"fixed direct row operand is absent from the typed registry: {field}")
    from .generic_grounding import bind_roles
    roles = {binding.role: binding.entity for binding in rule.bindings}
    if bind_roles(rule.contract, roles) != rule.bindings:
        raise ContractError("grounded argument roles differ from their declared ordered scope")
    expected_output_owner = _declared_owner(rule, tuple(role.key for role in rule.contract.roles))
    if rule.output.owner != expected_output_owner or rule.output.field != f"predicate:{rule.contract.predicate_id}":
        raise ContractError("grounded rule targets the wrong declared consequent")
    if rule.output not in variables:
        raise ContractError("declared consequent is absent from the shared registry")
    expected_fields = () if rule.contract.semantic_mode == "direct_observation" else tuple(
        field.field for field in rule.contract.required_fields)
    if tuple(field for field, _ in rule.inputs) != expected_fields:
        raise ContractError("grounded rule omitted or reordered a declared required field")
    identity = (rule.output.tick, rule.output.episode_id, rule.output.branch_id)
    for field, key in rule.inputs:
        if key not in variables or key.field != canonical_state_field(rule.contract, field):
            raise ContractError("grounded input is absent or has the wrong field")
        if key.owner != _declared_owner(rule, field_owner_roles(rule.contract, field)):
            raise ContractError("grounded input has the wrong typed role owner")
        if (key.tick, key.episode_id, key.branch_id) != identity:
            raise ContractError("grounded input crosses declared time, episode or profile")
    for field, key in rule.previous_inputs:
        if (key not in variables or key.field != canonical_state_field(rule.contract, field)
                or key.tick >= rule.output.tick):
            raise ContractError("previous dependency is absent or does not precede its output")
        if key.owner != _declared_owner(rule, field_owner_roles(rule.contract, field)) or (
                key.episode_id, key.branch_id) != (rule.output.episode_id, rule.output.branch_id):
            raise ContractError("previous dependency has the wrong typed owner or profile")
    declared_parameters = tuple(rule.contract.threshold_parameters)
    if tuple(rule.parameters) != declared_parameters:
        raise ContractError("grounded rule parameters differ from its exact contract")
    for name in declared_parameters:
        if name not in authoritative_parameters or rule.parameters[name] != authoritative_parameters[name]:
            raise ContractError(f"grounded parameter differs from its governing profile: {name}")


TRUTH_OUTPUT_ATTRIBUTE = "grounded_truth_output"


def declared_attribute_symbols(rules: Sequence[GroundedRule],
        variables: Mapping[VariableKey, CandidateVariable], *,
        source_attributes: Mapping[VariableKey, str] | None = None) -> Mapping[VariableKey, str]:
    """Canonical declared attributes, with opaque instance identities excluded.

    Bound source symbols are established from SourceCell metadata by the joint
    adapter. Ordinary inputs follow canonical_state_field, including proven
    shared aliases. Only a grounded consequent with predicate_contract schema
    receives the common truth-output symbol.
    """
    attributes = {}
    def assign(key, symbol):
        if key not in variables or not isinstance(symbol, str) or not symbol:
            raise ContractError("attribute declaration has an absent variable or invalid symbol")
        if key in attributes and attributes[key] != symbol:
            raise ContractError("one shared variable has conflicting declared attributes")
        attributes[key] = symbol
    for rule in rules:
        output = variables[rule.output]
        if (output.schema.source != "predicate_contract" or output.schema.field != rule.output.field
                or output.schema.data_type != "four_value" or output.schema.unit != "four-value"
                or output.schema.categories != TRUTH_DOMAIN):
            raise ContractError("neutral truth attribute requires the exact grounded consequent schema")
        assign(rule.output, TRUTH_OUTPUT_ATTRIBUTE)
        schemas = {schema.field: schema for schema in rule.contract.required_fields}
        for field, key in (*rule.inputs, *rule.previous_inputs):
            canonical = canonical_state_field(rule.contract, field)
            variable, schema = variables[key], schemas[field]
            if (key.field != canonical or (variable.schema.data_type, variable.schema.unit)
                    != (schema.data_type, schema.unit)):
                raise ContractError("ordinary attribute differs from its canonical contract declaration")
            assign(key, canonical)
    for key, symbol in ({} if source_attributes is None else source_attributes).items():
        if key not in variables:
            raise ContractError("source attribute references an absent variable")
        variable = variables[key]
        if (variable.schema.source != "bound_source_cell" or variable.schema.field != key.field
                or symbol != variable.schema.field):
            raise ContractError("source attribute differs from its validated bound SourceCell declaration")
        assign(key, symbol)
    if set(attributes) != set(variables):
        raise ContractError("each variable requires an explicit canonical attribute declaration")
    return MappingProxyType(attributes)


def _truth_inference_domain(variable: CandidateVariable) -> tuple:
    # Only a declared Boolean truth domain is event-compatible. Unknown/OOS
    # candidates remain four-valued strings and never become false evidence.
    if variable.key.field.startswith("predicate:") and variable.domain == ("false", "true"):
        return (False, True)
    return variable.domain


@dataclass(frozen=True)
class RuntimeGraph:
    graph: FactorGraph
    rules: tuple[GroundedRule, ...]
    variables: tuple[CandidateVariable, ...]
    candidate_tokens: torch.Tensor
    domain_mask: torch.Tensor
    soft_scopes: tuple[tuple[VariableKey, ...], ...]
    report: Mapping[str, Any]


@dataclass(frozen=True)
class DependencyIncidence:
    """Explicit local source/consequent scope; indices contain no source IDs."""
    scope: tuple[VariableKey, ...]
    role_slots: tuple[int, ...]
    unordered_groups: tuple[int, ...]
    parameter_values: tuple[float, ...]
    parameter_contexts: tuple[ParameterContext, ...]

    def __post_init__(self):
        if (not isinstance(self.scope, tuple) or len(self.scope) < 2
                or len(set(self.scope)) != len(self.scope)
                or len(self.role_slots) != len(self.scope)
                or len(self.unordered_groups) != len(self.scope)):
            raise ContractError("dependency incidence needs distinct variables and exact operand metadata")
        if any(type(slot) is not int or slot < 0 for slot in self.role_slots):
            raise ContractError("dependency operand slots must be nonnegative integers")
        if any(type(group) is not int or group < -1 for group in self.unordered_groups):
            raise ContractError("dependency groups must be integer IDs or ordered -1")
        if not isinstance(self.parameter_values, tuple) or any(
                type(value) is not float or not math.isfinite(value) for value in self.parameter_values):
            raise ContractError("dependency parameters must be an explicit tuple of finite floats")
        if (not isinstance(self.parameter_contexts, tuple)
                or len(self.parameter_contexts) != len(self.parameter_values)
                or any(not isinstance(item, ParameterContext) for item in self.parameter_contexts)):
            raise ContractError("dependency parameters require paired typed semantic contexts")


def _threshold_parameter_contexts(contract, name):
    """Read physical units and consuming operands from the contract AST."""
    from .unified_contracts import walk_expression
    schemas = {field.field: field for field in contract.required_fields}
    contexts = []
    for expression in walk_expression(contract.evaluation_spec):
        for operand in ("threshold", "minimum", "maximum"):
            if expression.get(operand) != name:
                continue
            fields = tuple(expression[key] for key in ("field", "left", "right") if key in expression)
            if not fields or any(field not in schemas for field in fields):
                raise ContractError("threshold operand lacks a declared consuming field")
            units = {schemas[field].unit for field in fields}
            if len(units) != 1:
                raise ContractError("threshold operand has contradictory consuming units")
            contexts.append(ParameterContext(name, units.pop(), "established",
                "predicate_evaluation." + expression["op"],
                operand + ":" + ",".join(canonical_state_field(contract, field) for field in fields), "threshold"))
    if not contexts:
        raise ContractError(f"ordinary parameter lacks declared consuming semantics: {name}")
    return tuple(contexts)


def _dependency_parameter_contexts(spec, node):
    """Preserve every direct VALUE consumer and its exact selector context."""
    if node.node_id.startswith("threshold:"):
        return _threshold_parameter_contexts(load_unified_contracts()[spec.predicate_id],
                                              node.node_id.removeprefix("threshold:"))
    field = node.node_id.removeprefix("param:")
    contexts = []
    for edge in spec.edges:
        if edge.source != node.node_id or edge.use != "value":
            continue
        function, separator, output = edge.target.partition("#")
        if not separator or ".py:" not in function or not output:
            raise ContractError(f"parameter consumer has no declared function/operand: {edge.target}")
        operand = "input:" + field + ";output:" + output
        if edge.selector:
            operand += ";selector:" + edge.selector
        if edge.index_bindings:
            operand += ";index:" + repr(edge.index_bindings)
        contexts.append(ParameterContext(field, node.unit, node.unit_status,
                                         function, operand, edge.relation))
    if not contexts:
        raise ContractError(f"parameter has no declared VALUE consumer: {node.node_id}")
    return tuple(contexts)


def _value_parameter_nodes(spec):
    """Shortest value-only depth, then semantic node order; guards stay out."""
    from .state_dependencies import PARAMETER
    node_by_id = {node.node_id: node for node in spec.nodes}
    depths = {spec.predicate_node_id: 0}
    pending = deque((spec.predicate_node_id,))
    while pending:
        target = pending.popleft()
        source = node_by_id[target].source_cell
        if source is not None and source.form in (
                "published_monitor", "governed_row_parameter", "publisher_requirement"):
            continue
        for edge in spec.edges:
            if edge.target == target and edge.use == "value" and edge.source not in depths:
                depths[edge.source] = depths[target]+1
                pending.append(edge.source)
    return tuple(sorted((node for node in spec.nodes if node.kind == PARAMETER and node.node_id in depths),
                        key=lambda node: (depths[node.node_id], node.node_id)))


@dataclass(frozen=True)
class SupervisedGraph:
    rules: tuple[GroundedRule, ...]
    variables: Mapping[VariableKey, CandidateVariable]
    attribute_symbols: Mapping[VariableKey, str]
    incidences: tuple[DependencyIncidence, ...]
    labels: Mapping[VariableKey, int]
    anchor_tick: int
    report: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "variables", MappingProxyType(dict(self.variables)))
        object.__setattr__(self, "attribute_symbols", MappingProxyType(dict(self.attribute_symbols)))
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))


@lru_cache(maxsize=1)
def _canonical_declared_plan(predicate_ids):
    """Keep one canonical plan shape, never a per-record authority cache."""
    from .joint_state_binding import compile_dependency_plan
    from .state_dependencies import load_dependency_specs
    specs = load_dependency_specs()
    return compile_dependency_plan({pid: specs[pid] for pid in predicate_ids})


def _declared_scalar_context(result, binding, contract, role_bindings):
    """Resolve only immediate identity-projected scalar receiver operands."""
    from .ast_compiler import set_field
    from Dataset.semantic_truth.world_truth import _missing_value
    spec = binding.plan.specs[contract.predicate_id]
    context, domains, observed, references = {}, {}, [], []
    if tuple(field.field for field in spec.fields) != tuple(
            schema.field for schema in contract.required_fields):
        raise ContractError("declared scalar fields differ from the exact contract")
    store = result.graph["SourceCell"]
    for field in spec.fields:
        projection = field.schema_projection
        if (field.owner_status != "resolved" or field.projection_status != "resolved"
                or projection is None or projection.type_relation != "identical"
                or projection.unit_relation not in ("identical", "nonmeasurement_code_alias")
                or (projection.contract_data_type, projection.contract_unit) != (field.data_type, field.unit)
                or not field.producer_role_projections
                or tuple(item.owner_role for item in field.producer_role_projections) != field.owner_roles
                or any(not (
                    item.selector == "same_identifier" and item.form in ("index_identity", "manifest_identity")
                    or item.form == "scope_source_lookup"
                    and item.selector == "producer_id=scope.source_entity_id_or_scope.entity_id;role_id=scope.entity_id"
                    and set(contract.template["scope_types"]) <= {"uav", "vehicle", "pedestrian", "facility"})
                    for item in field.producer_role_projections)):
            raise ContractError("declared scalar requires established identity/schema projections")
        source_ids = {item.source_node for item in field.producer_role_projections}
        if len(source_ids) != 1:
            raise ContractError("declared scalar has ambiguous producer identity paths")
        source_id = source_ids.pop()
        node = binding.plan.nodes[source_id]
        target = field.node_id if source_id != field.node_id else source_id
        incoming = tuple(edge for edge in spec.edges if edge.target == target and edge.use == "value")
        if (len(incoming) != 1 or "contract_field_projection" not in incoming[0].selector.split(";")
                or incoming[0].index_bindings or incoming[0].collection_keys):
            raise ContractError("declared scalar requires one exact contract field projection")
        if node.source_cell is None:
            node = binding.plan.nodes[incoming[0].source]
        elif incoming[0].source != source_id:
            raise ContractError("declared scalar projection differs from its immediate source")
        if node.source_cell is None:
            from .state_dependencies import DETERMINISTIC
            if (incoming[0].source != source_id or node.kind != DETERMINISTIC
                    or field.data_type not in ("numeric", "integer", "boolean", "categorical")
                    or (node.data_type, node.unit, node.domain) != (
                        projection.producer_data_type, projection.producer_unit, projection.producer_domain)
                    or tuple(item.index_key for item in field.producer_role_projections) != node.index.keys):
                raise ContractError("declared derived scalar differs from its exact producer schema/owner")
            indexes = {read.cell_index for read in binding.reads
                       if read.predicate_id == contract.predicate_id and read.node_id == node.node_id
                       and binding.cells[read.cell_index].key.tick == binding.cutoff_tick}
            if len(indexes) != 1:
                raise ContractError("declared derived scalar requires one exact cutoff dependency cell")
            cell = binding.cells[indexes.pop()]
            expected_owners = tuple(role_bindings[role].entity_id for role in field.owner_roles)
            if (cell.key.artifact != "dependency" or cell.key.field != node.node_id
                    or cell.key.owner_aliases != expected_owners
                    or cell.key.episode != binding.cells[0].key.episode
                    or cell.key.branch != binding.cells[0].key.branch
                    or cell.status != "unsupported" or cell.value is not None):
                raise ContractError("unexecuted derived scalar requires explicit unavailable producer evidence")
            set_field(context, field.field, None)
            domains[field.field] = (None,)
            references.extend((*node.schema_source, field.schema_source))
            continue
        source = node.source_cell
        if (source is None or source.form not in ("published_monitor", "canonical_field_root")
                or (node.data_type, node.unit) != (projection.producer_data_type, projection.producer_unit)
                or node.domain != projection.producer_domain
                or field.data_type not in ("numeric", "integer", "boolean", "categorical")):
            raise ContractError("declared formula operand is not an immediate typed scalar SourceCell")
        indexes = {read.cell_index for read in binding.reads
                   if read.predicate_id == contract.predicate_id and read.node_id == node.node_id
                   and binding.cells[read.cell_index].key.tick == binding.cutoff_tick}
        if len(indexes) != 1:
            raise ContractError("declared scalar requires one exact cutoff SourceCell")
        index = indexes.pop()
        cell = binding.cells[index]
        expected_owners = tuple(role_bindings[role].entity_id for role in field.owner_roles)
        if ((cell.key.artifact, cell.key.field) != (source.artifact, source.field_path)
                or cell.key.owner_aliases != expected_owners
                or cell.key.episode != binding.cells[0].key.episode
                or cell.key.branch != binding.cells[0].key.branch):
            raise ContractError("declared scalar source artifact, owner or time authority differs")
        if cell.status not in ("admitted", "missing", "latent"):
            raise ContractError("declared scalar source is unsupported")
        value = cell.value
        if cell.status != "admitted" or _missing_value(value):
            value = None
        elif field.data_type == "categorical":
            code = int(store.category_code[index])
            categories = result.metadata["source_categories"]
            if (type(value) is not str or not 0 <= code < len(categories)
                    or categories[code] != value or value not in node.domain.values):
                raise ContractError("declared category differs from exact SourceCell encoding/domain")
        else:
            valid = (type(value) in (int, float) if field.data_type == "numeric"
                     else type(value) is (bool if field.data_type == "boolean" else int))
            if (not valid or not math.isfinite(value) or not bool(store.numeric_encoding_supported[index])
                    or store.value_present[index].tolist() != [True, False, False]
                    or int(store.category_code[index]) != -1 or float(store.exact_x[index, 0]) != value):
                raise ContractError("declared scalar differs from exact SourceCell encoding")
        set_field(context, field.field, value)
        domains[field.field] = (value,)
        if value is not None:
            observed.append(field.field)
        references.extend((*node.schema_source, field.schema_source))
    return context, domains, tuple(observed), tuple(dict.fromkeys(references))


def joint_state_fixed_rule_graph(result, *, episode_id: str, branch_id: str,
        source_owners: Mapping[str, TypedEntity], class_catalog: EntityClassCatalog,
        parameter_profiles: Mapping[str, ParameterProfile] | None = None,
        output_domain: tuple[str, ...] = TRUTH_DOMAIN,
        applicable: bool = True, source_out_of_scope: bool = False) -> SupervisedGraph:
    """Ground direct producers or declared formulas from exact cutoff tensors.

    Declared formulas require caller profiles, including empty parameter sets.
    No labels enter this conversion. Future candidates require another interface.
    """
    from .joint_state_pyg import JointStatePygResult
    from .joint_state_binding import BindingResult
    if not isinstance(result, JointStatePygResult):
        raise TypeError("fixed rule conversion requires JointStatePygResult")
    binding = result.metadata.get("binding")
    if not isinstance(binding, BindingResult) or class_catalog.episode_id != episode_id:
        raise ContractError("fixed rule conversion requires original typed binding and episode catalog")
    if any(cell.key.episode != episode_id for cell in binding.cells):
        raise ContractError("fixed rule sources belong to another episode")
    contracts = _contracts()
    declared = tuple(pid for pid in binding.selected_predicates
                     if contracts[pid].semantic_mode == "declared_formula")
    if declared:
        authority_plan = _canonical_declared_plan(tuple(binding.plan.specs))
        relevant_nodes = set()
        for pid in declared:
            spec = authority_plan.specs[pid]
            if binding.plan.specs[pid] != spec:
                raise ContractError("declared formula spec differs from canonical dependency authority")
            relevant_nodes.update(node.node_id for node in spec.nodes)
        if any(binding.plan.nodes.get(node_id) != authority_plan.nodes[node_id]
               for node_id in relevant_nodes):
            raise ContractError("declared formula node differs from canonical dependency authority")
        if (tuple(edge for edge in binding.plan.edges if edge.target in relevant_nodes)
                != tuple(edge for edge in authority_plan.edges if edge.target in relevant_nodes)
                or any(binding.plan.families.get(node_id) != family
                       for node_id, family in authority_plan.families.items() if node_id in relevant_nodes)):
            raise ContractError("declared formula edges/families differ from canonical dependency authority")
    if (type(applicable) is not bool or type(source_out_of_scope) is not bool
            or not output_domain or any(value not in TRUTH_DOMAIN for value in output_domain)):
        raise ContractError("fixed row rule requires explicit applicability and truth domain")
    if any(alias != owner.entity_id or alias not in binding.aliases for alias, owner in source_owners.items()):
        raise ContractError("fixed rule owners must be exact isolated typed aliases")
    owners = {alias: class_catalog.resolve(owner) for alias, owner in source_owners.items()}
    variables, rules = {}, []
    cutoff = binding.cutoff_tick
    query = binding.isolated_input["query"]
    store = result.graph["SourceCell"]
    for pid in binding.selected_predicates:
        contract = contracts[pid]
        if contract.semantic_mode == "declared_formula":
            if parameter_profiles is None or pid not in parameter_profiles or not isinstance(
                    parameter_profiles[pid], ParameterProfile):
                raise ContractError("declared formula requires an explicit caller parameter profile")
            profile = parameter_profiles[pid]
            if set(profile.parameters) != set(contract.threshold_parameters):
                raise ContractError("declared formula profile differs from exact governed parameters")
            role_bindings = {role.key: owners[query["bindings"][role.key]] for role in contract.roles}
            resolved = bind_roles(contract, role_bindings, class_catalog=class_catalog)
            context, domains, observed, source_refs = _declared_scalar_context(
                result, binding, contract, role_bindings)
            source_refs = (*source_refs, profile.source_ref)
            authority = OutputBindingAuthority(pid, resolved, cutoff, episode_id, branch_id,
                profile.profile_id, profile.parameters, source_refs)
            rules.append(ground_contract(contract, role_bindings, context, profile.parameters,
                tick=cutoff, episode_id=episode_id, branch_id=branch_id, variable_registry=variables,
                domain_by_field=domains, observed_fields=observed, source_refs=source_refs,
                output_domain=output_domain, parameter_profile=profile, output_authority=authority,
                class_catalog=class_catalog, applicable=applicable, source_out_of_scope=source_out_of_scope))
            continue
        direct, nodes = _direct_fields(contract)
        receiver_authority = "Dataset/world_model/graph/observation_logs.py:generate"
        if any(node.source_cell.form == "governed_row_parameter" for node in nodes):
            if binding.communication_profile is None:
                raise ContractError("governed row parameters require named communication profile authority")
            flow_profile = _governed_flow_profile(binding.communication_profile, episode_id)
            profile = ParameterProfile(profile_id=binding.communication_profile.profile_id,
                parameters={"quality_thresholds.latency_ms": round(float(flow_profile["latency_threshold_ms"]), 6),
                            "quality_thresholds.packet_loss_ratio": round(float(flow_profile["packet_loss_threshold_ratio"]), 6)},
                source_ref=binding.communication_profile.profile_ref)
            source_refs = (binding.communication_profile.manifest_ref, receiver_authority)
        else:
            from .state_dependencies import DIRECT_PREDICATE_PRODUCERS
            producer = DIRECT_PREDICATE_PRODUCERS[contract.grounding_spec["candidate_authority"]["source"]]
            producer_ref = f"{producer.module}:{direct.function}"
            profile = ParameterProfile(profile_id=producer_ref, parameters={}, source_ref=producer_ref)
            source_refs = (producer_ref, receiver_authority)
        role_bindings = {role.key: owners[query["bindings"][role.key]] for role in contract.roles}
        resolved = bind_roles(contract, role_bindings, class_catalog=class_catalog)
        authority = OutputBindingAuthority(pid, resolved, cutoff, episode_id, branch_id,
            profile.profile_id, profile.parameters, source_refs)
        rule = ground_contract(contract, role_bindings, {}, {}, tick=cutoff,
            episode_id=episode_id, branch_id=branch_id, variable_registry=variables,
            output_domain=output_domain, parameter_profile=profile, output_authority=authority,
            class_catalog=class_catalog, applicable=applicable, source_out_of_scope=source_out_of_scope)
        producer_inputs = []
        for node in nodes:
            indexes = {read.cell_index for read in binding.reads if read.predicate_id == pid
                       and read.node_id == node.node_id
                       and binding.cells[read.cell_index].key.tick == cutoff}
            if len(indexes) != 1:
                raise ContractError("fixed row requires one exact current operand per declared field")
            index = indexes.pop()
            cell = binding.cells[index]
            if (cell.key.artifact, cell.key.field) != (node.source_cell.artifact, node.source_cell.field_path):
                raise ContractError("fixed row operand has no established current producer owner/field")
            if not cell.key.owners or any(alias not in owners for alias in cell.key.owners):
                raise ContractError("fixed row requires all typed ordered producer owners")
            typed_owners = tuple(owners[alias] for alias in cell.key.owners)
            if (cell.key.artifact == "communication_state.jsonl"
                    and any(owner.ontology_class == canonical_class_uri("world:Facility") for owner in typed_owners)):
                raise ContractError("communication row station identity cannot be its facility anchor")
            if cell.status not in ("admitted", "missing") and not (
                    cell.status == "latent" and node.source_cell.form == "unobserved_monitor"):
                raise ContractError("fixed current row operand must be admitted or explicitly missing")
            field = node.source_cell.field_path
            key = VariableKey(typed_owners, f"source:{cell.key.artifact}:{field}", cutoff, episode_id, branch_id)
            def value_at(sample_index):
                sample = binding.cells[sample_index]
                if sample.status == "missing" or (
                        sample.status == "latent" and node.source_cell.form == "unobserved_monitor"):
                    return None
                if node.data_type in ("numeric", "boolean", "integer"):
                    valid_type = (type(sample.value) in (int, float) if node.data_type == "numeric"
                                  else type(sample.value) is (bool if node.data_type == "boolean" else int))
                    if (not valid_type or not math.isfinite(sample.value)
                            or not bool(store.numeric_encoding_supported[sample_index])
                            or store.value_present[sample_index].tolist() != [True, False, False]
                            or int(store.category_code[sample_index]) != -1):
                        raise ContractError("fixed typed operand requires supported scalar PyG encoding")
                    scalar = float(store.exact_x[sample_index, 0])
                    if scalar != sample.value:
                        raise ContractError("fixed scalar PyG value differs from the exact receiver value")
                    if node.data_type == "boolean":
                        return bool(scalar)
                    if node.data_type == "integer":
                        return int(scalar)
                    return scalar
                if node.data_type != "categorical" or type(sample.value) is not str:
                    raise ContractError("fixed operand has no declared typed receiver conversion")
                code = int(store.category_code[sample_index])
                categories = result.metadata["source_categories"]
                if not 0 <= code < len(categories):
                    raise ContractError("categorical fixed operand lacks an exact PyG category")
                value = categories[code]
                if value != sample.value or value not in node.domain.values:
                    raise ContractError("categorical fixed operand differs from its declared producer domain")
                return value
            value = value_at(index)
            history = sorted({(sample.key.tick, value_at(read.cell_index))
                for read in binding.reads if read.predicate_id == pid and read.node_id == node.node_id
                for sample in (binding.cells[read.cell_index],)
                if sample.key.owners == cell.key.owners and sample.key.tick < cutoff
                and sample.status == "admitted"})
            variable = CandidateVariable(key, (value,), FieldSchema(key.field, node.data_type,
                node.unit, "bound_source_cell", node.domain.values), observed=value is not None,
                observed_value=value, history=tuple(history), source_refs=node.schema_source,
                abstraction="fixed_current_receiver_row")
            if key in variables and variables[key] != variable:
                raise ContractError("shared fixed row operand declarations disagree")
            variables[key] = variable
            producer_inputs.append((field, key))
        rules.append(replace(rule, producer_inputs=tuple(producer_inputs), producer_binding=direct))
    attributes = declared_attribute_symbols(rules, variables,
        source_attributes={key: key.field for rule in rules for _, key in rule.producer_inputs})
    return SupervisedGraph(tuple(rules), variables, attributes, (), {}, cutoff,
        {"abstraction": "fixed_current_receiver_row", "label_policy": "no_labels",
         "numeric_future_model": False, "producer_functions": tuple(
             rule.producer_binding.function for rule in rules if rule.producer_binding is not None)})


@dataclass(frozen=True)
class NumericCandidateOutcomeEvaluator:
    """Compiled declared direct producers for one query: float64 `value > threshold` per predicate."""
    current: Any
    columns: Any
    thresholds: Any
    anchors: Any

    def truth(self, values):
        """Declared producer truth [..., T, P] in rule order; all operands are admitted and not_required."""
        return values[..., self.columns] > self.thresholds

    def __call__(self, candidate_values):
        """First opposite-to-anchor tick index 0..9, or 10 beyond the horizon, per candidate and predicate."""
        opposite = self.truth(candidate_values[:, 1:]) != self.anchors
        return np.where(opposite.any(1), opposite.argmax(1), 10).astype(np.int64)


@lru_cache(maxsize=None)
def _terminal_numeric_gt_paths(function):
    """Accept only the complete declared guarded numeric-GT producer shape (checked once per function)."""
    import ast
    import inspect
    import textwrap

    def definition(target):
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
        except (TypeError, OSError, SyntaxError) as error:
            raise ContractError("numeric batch producer source is unavailable") from error
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ContractError("numeric batch producer must be one ordinary function")
        node = tree.body[0]
        args = node.args
        if (node.decorator_list or len(args.args) != 1 or args.posonlyargs or args.vararg
                or args.kwonlyargs or args.kwarg or args.defaults or args.kw_defaults):
            raise ContractError("numeric batch producer signature drift")
        return node, args.args[0].arg

    node, row = definition(function)
    if len(node.body) != 5:
        raise ContractError("numeric batch producer body drift")
    paths, names = [], []
    for statement in node.body[1:3]:
        if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)):
            raise ContractError("numeric batch operands must be direct assignments")
        call = statement.value
        if (not isinstance(call, ast.Call) or call.keywords or len(call.args) != 1
                or not isinstance(call.func, ast.Attribute) or call.func.attr != "get"
                or not isinstance(call.func.value, ast.Subscript)
                or not isinstance(call.func.value.value, ast.Name) or call.func.value.value.id != row
                or not isinstance(call.func.value.slice, ast.Constant) or not isinstance(call.func.value.slice.value, str)
                or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str)):
            raise ContractError("numeric batch operands require exact row section/get field paths")
        paths.append((call.func.value.slice.value, call.args[0].value))
        names.append(statement.targets[0].id)
    if len(set(names)) != 2 or row in names:
        raise ContractError("numeric batch operand names drift")
    value, threshold = names
    expected = ast.parse(f"""
if _source_requirement_status({row}) == "source_required":
    return "unknown"
{value} = {row}[{paths[0][0]!r}].get({paths[0][1]!r})
{threshold} = {row}[{paths[1][0]!r}].get({paths[1][1]!r})
if not isinstance({value}, (int, float)) or not isinstance({threshold}, (int, float)):
    return "unknown"
return "true" if {value} > {threshold} else "false"
""").body
    if [ast.dump(n, include_attributes=False) for n in node.body] != [ast.dump(n, include_attributes=False) for n in expected]:
        raise ContractError("numeric batch producer guard/type/terminal AST drift")
    guard = function.__globals__.get("_source_requirement_status")
    guard_node, guard_row = definition(guard)
    guard_expected = ast.parse(f"""
if "requirement" in {guard_row}:
    return str({guard_row}["requirement"].get("status"))
if "failure" in {guard_row}:
    return str({guard_row}["failure"].get("requirement_status"))
return "not_required"
""").body
    if [ast.dump(n, include_attributes=False) for n in guard_node.body] != [ast.dump(n, include_attributes=False) for n in guard_expected]:
        raise ContractError("numeric batch source-requirement helper AST drift")
    return tuple(".".join(path) for path in paths)


def numeric_candidate_outcome_evaluator(result, *, prefix, receiver_bank,
        source_owners, class_catalog, branch_id):
    """Compile the actual declared direct producers for one query's typed receiver row.

    Candidate order is the fitted law's eligible residual/donor order. Donor guards
    were established once when the TRAIN donor bank was built. Chosen branches use
    the same compiled float64 comparison, so Ours and RuleMC share every draw and rule.
    """
    import numpy as np
    from importlib import import_module
    from .ast_compiler import set_field
    from .state_dependencies import DIRECT_PREDICATE_PRODUCERS
    from .communication_data import PREDICATES
    current = joint_state_fixed_rule_graph(result, episode_id=prefix.episode,
        branch_id=branch_id, source_owners=source_owners, class_catalog=class_catalog)
    columns = {(f.artifact, f.field): i for i, f in enumerate(prefix.fields)}
    specs = receiver_bank.field_specs
    if tuple((f.artifact, f.field, f.unit, f.owner_roles) for f in prefix.fields) != tuple(
            (s.artifact, s.field_path, s.unit, s.owner_keys) for s in specs):
        raise ContractError("candidate fields differ from exact receiver authority")
    predicates = tuple(rule.contract.predicate_id for rule in current.rules)
    if len(predicates) != 2 or set(predicates) != set(PREDICATES):
        raise ContractError("extrema design requires the exact two communication predicates")
    producers = {}
    for rule in current.rules:
        if not rule.applicable or rule.source_out_of_scope:
            raise ContractError("candidate direct predicate is outside known scope")
        declaration = DIRECT_PREDICATE_PRODUCERS[rule.contract.grounding_spec["candidate_authority"]["source"]]
        function = getattr(import_module(declaration.module[:-3].replace("/", ".")), rule.producer_binding.function)
        value_path, threshold_path = _terminal_numeric_gt_paths(function)
        _, nodes = _direct_fields(rule.contract)
        if len(nodes) != len(rule.producer_inputs):
            raise ContractError("candidate operand descriptors and producer inputs differ")
        anchor_row = {}
        declared_operands = {}
        for node, (field, key) in zip(nodes, rule.producer_inputs):
            form = node.source_cell.form
            anchor_value = current.variables[key].domain[0]
            set_field(anchor_row, field, anchor_value)
            col = None
            if form == "published_monitor":
                col = columns[(node.source_cell.artifact, field)]
                spec = specs[col]
                if (node.data_type, node.unit, node.owner_keys) != (spec.data_type, spec.unit, spec.owner_keys):
                    raise ContractError("candidate operand type/unit/owner declaration drift")
                if anchor_value != float(prefix.values[-1, col]):
                    raise ContractError("candidate producer anchor differs from exact receiver prefix")
                initial_value = anchor_value
            elif form == "governed_row_parameter":
                initial_value = rule.parameter_profile.parameters[field]
                if anchor_value != initial_value:
                    raise ContractError("candidate anchor parameter differs from its fixed profile")
            elif form == "publisher_requirement" and field == "requirement.status":
                initial_value = anchor_value
            else:
                raise ContractError("candidate operand lacks declared authority")
            declared_operands[field] = (form, initial_value, col)
        anchor_truth = function(anchor_row)
        if anchor_truth not in ("true", "false"):
            raise ContractError("candidate hard predicate anchor is unknown")
        if (value_path not in declared_operands or threshold_path not in declared_operands
                or declared_operands[value_path][0] != "published_monitor"
                or declared_operands[threshold_path][0] != "governed_row_parameter"
                or set(declared_operands) != {value_path, threshold_path, "requirement.status"}
                or declared_operands["requirement.status"][:2] != ("publisher_requirement", "not_required")):
            raise ContractError("numeric batch AST operands differ from exact typed producer declarations")
        threshold = declared_operands[threshold_path][1]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
            raise ContractError("numeric batch threshold must be finite declared numeric data")
        producers[rule.contract.predicate_id] = (declared_operands[value_path][2], float(threshold), anchor_truth == "true")
    ordered = [producers[pid] for pid in PREDICATES]
    return NumericCandidateOutcomeEvaluator(current,
        np.array([c for c, _, _ in ordered], dtype=np.int64),
        np.array([t for _, t, _ in ordered], dtype=np.float64),
        np.array([a for _, _, a in ordered], dtype=bool))


def joint_state_supervised_graph(result, *, episode_id: str, branch_id: str,
        flow: TypedEntity, label_flow_id: str,
        source_owners: Mapping[str, TypedEntity], owner_slots: Mapping[str, int],
        class_catalog: EntityClassCatalog, label_records: Sequence[tuple[Mapping[str, Any], str]],
        parameter_profile: ParameterProfile,
        source_branch: int = 0,
        output_domain: tuple[str, ...] = TRUTH_DOMAIN) -> SupervisedGraph:
    """Condition direct truth outputs on admitted, typed PyG source cells.

    This is a supervised observed-prefix abstraction, not producer execution or
    numerical future inference. Labels are used only as likelihood indices.
    ObservedLog, native traces and isolated_input are never learner inputs here.
    The caller supplies independently established alias types and operand slots.
    """
    from .joint_state_pyg import JointStatePygResult
    from .joint_state_binding import CellKey, STATUSES, FormalWeatherRegionProjection, _formal_weather_authority
    from .state_dependencies import NODE_KINDS, RELATIONS
    if not isinstance(result, JointStatePygResult):
        raise TypeError("supervised conversion requires JointStatePygResult")
    if class_catalog.episode_id != episode_id or set(source_owners) != set(owner_slots):
        raise ContractError("source owner authority differs from episode or operand slots")
    if not source_owners or len(set(owner_slots.values())) != len(owner_slots):
        raise ContractError("source owners need distinct explicit operand slots")
    if any(type(slot) is not int or slot < 0 for slot in owner_slots.values()):
        raise ContractError("source owner slots must be nonnegative integers")
    for alias, owner in source_owners.items():
        if alias != owner.entity_id:
            raise ContractError("source owner must use its exact isolated alias")
        class_catalog.resolve(owner)
    class_catalog.resolve(flow)
    if (not isinstance(output_domain, tuple) or not output_domain
            or len(set(output_domain)) != len(output_domain)
            or any(value not in TRUTH_DOMAIN for value in output_domain)):
        raise ContractError("output domain must explicitly declare distinct four-value truth states")
    metadata, graph = result.metadata, result.graph
    cells, reads, plan = metadata["cells"], metadata["reads"], metadata["plan"]
    aliases, node_ids = tuple(metadata["alias_order"]), tuple(metadata["node_ids"])
    if node_ids != tuple(sorted(plan.nodes)) or flow.entity_id not in aliases:
        raise ContractError("dependency node or flow alias authority differs from tensor indexing")
    if not set(source_owners) <= set(aliases):
        raise ContractError("typed source owners are absent from the isolated graph")
    codes = {node: index for index, node in enumerate(node_ids)}
    if type(source_branch) is not int or source_branch < 0:
        raise ContractError("source branch must explicitly identify a nonnegative binder branch")
    if len({cell.key for cell in cells}) != len(cells):
        raise ContractError("source metadata contains duplicate physical cell identities")
    upstream_nodes = {pid: frozenset(spec.upstream_closure()) for pid, spec in plan.specs.items()}
    for read in reads:
        if (type(read.cell_index) is not int or not 0 <= read.cell_index < len(cells)
                or read.node_id not in plan.nodes or read.predicate_id not in plan.specs
                or read.predicate_id not in metadata["selected_predicates"]
                or read.node_id not in upstream_nodes[read.predicate_id]):
            raise ContractError("source read differs from its exact predicate/node/cell authority")
    def check_tensor(actual, expected, description):
        if actual.dtype != expected.dtype or not torch.equal(actual.detach().cpu(), expected):
            raise ContractError(f"dependency tensor differs from metadata: {description}")
    nodes = [plan.nodes[node] for node in node_ids]
    if (tuple(metadata["data_types"]) != tuple(sorted({node.data_type for node in nodes}))
            or tuple(metadata["domains"]) != tuple(sorted({node.domain.kind for node in nodes}))
            or tuple(metadata["unit_statuses"]) != tuple(sorted({node.unit_status for node in nodes}))
            or tuple(metadata["statuses"]) != STATUSES):
        raise ContractError("dependency schema code dictionaries differ from the compiled plan")
    for name, expected in {
        "semantic_code": list(range(len(nodes))),
        "kind_code": [NODE_KINDS.index(node.kind) for node in nodes],
        "data_type_code": [metadata["data_types"].index(node.data_type) for node in nodes],
        "domain_code": [metadata["domains"].index(node.domain.kind) for node in nodes],
        "unit_status_code": [metadata["unit_statuses"].index(node.unit_status) for node in nodes],
    }.items():
        check_tensor(graph["DependencyNode"][name], torch.tensor(expected, dtype=torch.long), name)
    dependency = graph["DependencyNode", "dependency", "DependencyNode"]
    check_tensor(dependency.edge_index, torch.tensor([(codes[edge.source], codes[edge.target])
        for edge in plan.edges], dtype=torch.long).reshape(-1, 2).T, "plan edges")
    for name, expected in {
        "relation_code": [RELATIONS.index(edge.relation) for edge in plan.edges],
        "use_code": [("value", "guard").index(edge.use) for edge in plan.edges],
    }.items():
        check_tensor(dependency[name], torch.tensor(expected, dtype=torch.long), name)
    for name in ("time_grid", "temporal_lookup", "update_policy", "absence_policy"):
        vocabulary = tuple(sorted({getattr(edge, name) for edge in plan.edges}))
        if tuple(metadata["temporal_policy_codes"][name]) != vocabulary:
            raise ContractError("dependency temporal dictionary differs from the compiled plan")
        check_tensor(dependency[name+"_code"], torch.tensor([vocabulary.index(getattr(edge, name))
            for edge in plan.edges], dtype=torch.long), name)
    store = graph["SourceCell"]
    if (store.num_nodes != len(cells) or store.x.shape != (len(cells), 3)
            or store.value_present.shape != store.x.shape
            or store.numeric_encoding_supported.shape != (len(cells),)
            or store.status_code.shape != (len(cells),)
            or store.relative_tick.shape != (len(cells),)
            or graph["ObservedEntity"].num_nodes != len(aliases)
            or graph["DependencyNode"].num_nodes != len(node_ids)):
        raise ContractError("dependency/cell tensor shapes differ from metadata")
    if (store.x.dtype != torch.float32 or store.value_present.dtype != torch.bool
            or store.numeric_encoding_supported.dtype != torch.bool
            or store.status_code.dtype != torch.long or store.relative_tick.dtype != torch.long
            or not bool(torch.isfinite(store.x).all())):
        raise ContractError("source tensor encodings must retain finite values and exact types")
    cutoffs = {cell.key.tick - int(store.relative_tick[index]) for index, cell in enumerate(cells)}
    if len(cutoffs) != 1:
        raise ContractError("source tensor times differ from one isolated cutoff")
    cutoff = cutoffs.pop()
    if type(cutoff) is not int or cutoff < 0:
        raise ContractError("source cutoff must be a nonnegative integer")
    weather_projection = metadata["weather_projection"]
    if weather_projection is not None:
        from pathlib import Path
        if not isinstance(weather_projection, FormalWeatherRegionProjection):
            raise ContractError("weather metadata requires a typed formal region projection")
        weather_episode, raw_region, manifest_ref = _formal_weather_authority(Path(weather_projection.partition))
        region = source_owners.get(weather_projection.region_alias)
        if ((weather_projection.episode_id, weather_projection.raw_region_id, weather_projection.manifest_ref)
                != (weather_episode, raw_region, manifest_ref) or weather_episode != episode_id
                or weather_projection.source_authority != "episode_manifest.json#capture_boundary_id"
                or weather_projection.ontology_class != "world:OperationalRegion"
                or type(weather_projection.availability_tick) is not int
                or not 0 <= weather_projection.availability_tick <= cutoff
                or region is None or region.ontology_class != canonical_class_uri("world:OperationalRegion")):
            raise ContractError("weather ownership differs from exact typed formal manifest authority")

    def check_edges(key, pairs):
        expected = torch.tensor(sorted(set(pairs)), dtype=torch.long).reshape(-1, 2).T
        actual = graph[key].edge_index.detach().cpu()
        if actual.dtype != torch.long or not torch.equal(actual, expected):
            raise ContractError(f"dependency tensor edges differ from metadata: {key}")

    check_edges(("SourceCell", "binds", "DependencyNode"),
                ((read.cell_index, codes[read.node_id]) for read in reads))
    check_edges(("SourceCell", "upstream_of", "DependencyNode"),
                ((read.cell_index, codes[plan.specs[read.predicate_id].predicate_node_id]) for read in reads))
    frames = tuple(dict.fromkeys(cell.frame_owner for cell in cells if cell.frame_owner is not None))
    if tuple(metadata["frame_owners"]) != frames or graph["CoherentFrame"].num_nodes != len(frames):
        raise ContractError("coherent source frames differ from original cell metadata")
    check_tensor(graph["CoherentFrame"].relative_tick,
                 torch.tensor([frame.tick - cutoff for frame in frames], dtype=torch.long),
                 "coherent frame times")
    frame_codes = {frame: index for index, frame in enumerate(frames)}
    check_tensor(graph["CoherentFrame", "contains", "SourceCell"].edge_index,
                 torch.tensor([(frame_codes[cell.frame_owner], index)
                               for index, cell in enumerate(cells) if cell.frame_owner is not None],
                              dtype=torch.long).reshape(-1, 2).T,
                 "coherent frame containment")
    # Ownership serialization follows cell order, rather than sorted pair order.
    ownership = [(aliases.index(owner), index) for index, cell in enumerate(cells)
                 for owner in cell.key.owner_aliases]
    expected_owner = torch.tensor(ownership, dtype=torch.long).reshape(-1, 2).T
    if not torch.equal(graph["ObservedEntity", "owns_source_cell", "SourceCell"].edge_index.detach().cpu(), expected_owner):
        raise ContractError("source-cell ownership edges differ from explicit aliases")
    selected = tuple(metadata["selected_predicates"])
    contracts = load_unified_contracts()
    if not selected or len(selected) != len(set(selected)):
        raise ContractError("supervised predicates must be explicit and distinct")
    by_predicate = {}
    for row, source_ref in label_records:
        pid = row["predicate_id"]
        if pid not in selected or pid in by_predicate:
            raise ContractError("independent supervision must name each selected predicate exactly once")
        if (row["episode_id"] != episode_id or row["tick"] != cutoff
                or row["bindings"] != {"flow": label_flow_id}
                or set(row["binding_ontology_classes"]) != {"flow"}
                or canonical_class_uri(row["binding_ontology_classes"]["flow"]) != flow.ontology_class
                or (row["schema_name"], row["schema_version"]) != ("compute_comm_predicate_truth", "2.0.0")
                or not isinstance(source_ref, str) or not source_ref):
            raise ContractError("independent predicate source has wrong binding, time, type or schema")
        if row["value"] not in output_domain:
            raise ContractError("supervision value is outside the explicitly declared output domain")
        by_predicate[pid] = (row, source_ref)
    if set(by_predicate) != set(selected):
        raise ContractError("independent supervision is missing a selected predicate")
    variables, rules, labels = {}, [], {}
    for pid in selected:
        contract = contracts[pid]
        if contract.semantic_mode != "direct_observation" or tuple(role.key for role in contract.roles) != ("flow",):
            raise ContractError("observed-prefix supervised conversion requires direct flow predicates")
        bindings = bind_roles(contract, {"flow": flow}, class_catalog=class_catalog)
        authority = OutputBindingAuthority(pid, bindings, cutoff, episode_id, branch_id,
            parameter_profile.profile_id, parameter_profile.parameters, (by_predicate[pid][1],))
        rule = ground_contract(contract, {"flow": flow}, {}, {}, tick=cutoff,
            episode_id=episode_id, branch_id=branch_id, variable_registry=variables,
            output_domain=output_domain, parameter_profile=parameter_profile,
            output_authority=authority, class_catalog=class_catalog)
        rules.append(rule)
        labels[rule.output] = output_domain.index(by_predicate[pid][0]["value"])
    reads_by_cell = {}
    for read in reads:
        if read.cell_index not in reads_by_cell:
            reads_by_cell[read.cell_index] = []
        reads_by_cell[read.cell_index].append(read)
    source_keys, source_series, exclusions = {}, {}, {}
    def exclude(reason):
        exclusions[reason] = exclusions.get(reason, 0) + 1
    for index, cell in enumerate(cells):
        if cell.key.artifact == "weather_meta.jsonl":
            expected_owner = None if weather_projection is None else weather_projection.region_alias
            if cell.key.entity != expected_owner:
                raise ContractError("weather cell owner differs from the formal region projection")
            if (weather_projection is not None and cell.status == "admitted"
                    and cell.key.tick < weather_projection.availability_tick):
                raise ContractError("weather cell predates typed ownership availability")
        if int(store.status_code[index]) != STATUSES.index(cell.status):
            raise ContractError("source tensor status differs from immutable metadata")
        value = cell.value
        numeric = (() if value is None or isinstance(value, Mapping) else
                   value if isinstance(value, tuple) else (value,))
        supported = bool(numeric) and all(type(item) in (bool, int, float) for item in numeric)
        if bool(store.numeric_encoding_supported[index]) != supported:
            raise ContractError("source numeric support differs from immutable metadata")
        if supported:
            if len(numeric) not in (1, 3):
                raise ContractError("source scalar/vector encoding has unsupported component count")
            expected_values = torch.tensor((*numeric, *((0.,) * (3-len(numeric)))), dtype=torch.float32)
            expected_present = torch.arange(3) < len(numeric)
            if (not torch.equal(store.x[index].detach().cpu(), expected_values)
                    or not torch.equal(store.value_present[index].detach().cpu(), expected_present)):
                raise ContractError("source tensor values differ from immutable metadata")
        elif bool(store.value_present[index].any()) or bool(store.x[index].any()):
            raise ContractError("non-numeric source must retain empty tensor encoding")
        if (cell.key.episode != episode_id or cell.key.branch != source_branch
                or type(cell.key.tick) is not int or cell.key.tick < 0):
            raise ContractError("source metadata crosses the requested episode or binder branch")
        if cell.status != "admitted":
            exclude(cell.status)
            continue
        if cell.key.tick > cutoff:
            raise ContractError("future source cell cannot be admitted to supervised input")
        if not cell.key.owner_aliases:
            exclude("global_source")
            continue
        if any(alias not in source_owners for alias in cell.key.owner_aliases):
            exclude("owner_outside_declared_subset")
            continue
        nodes = [plan.nodes[read.node_id] for read in reads_by_cell.get(index, ())]
        if not nodes or any(node.source_cell is None or
                (node.source_cell.artifact, node.source_cell.field_path) !=
                (cell.key.artifact, cell.key.field) for node in nodes):
            raise ContractError("admitted source artifact/field differs from its declared SourceCell metadata")
        schemas = {(node.data_type, node.unit, node.unit_status) for node in nodes}
        if len(schemas) != 1:
            raise ContractError("shared source cell has absent or contradictory schema authority")
        source_type, unit, unit_status = schemas.pop()
        # Dependency metadata names a three-component physical vector vector3;
        # CandidateVariable's existing encoder schema names that type vector.
        candidate_types = {"numeric": "numeric", "vector3": "vector", "boolean": "boolean"}
        if source_type not in candidate_types or not supported:
            exclude("unsupported_type")
            continue
        kind = candidate_types[source_type]
        if unit is None:
            raise ContractError("admitted source must have an established physical unit")
        authority = []
        for node in sorted({node.node_id: node for node in nodes}.values(), key=lambda node: node.node_id):
            source = node.source_cell
            root = plan.nodes[plan.families[node.node_id]].source_cell
            if (root is None or source.coordinate_keys.count("episode_id") != 1
                    or source.coordinate_keys.count("tick") != 1
                    or len(set(source.coordinate_keys)) != len(source.coordinate_keys)
                    or not source.form or node.schema_status != "established"
                    or not node.schema_source):
                raise ContractError("admitted source lacks exact coordinate/form/schema authority")
            family, ancestors = node.node_id, set()
            while plan.nodes[family].source_cell.coherent_parent is not None:
                if family in ancestors:
                    raise ContractError("admitted source coherent-parent authority contains a cycle")
                ancestors.add(family)
                family = plan.nodes[family].source_cell.coherent_parent
                if (family not in plan.nodes or plan.nodes[family].source_cell is None
                        or plan.nodes[family].source_cell.artifact != cell.key.artifact):
                    raise ContractError("admitted source coherent-parent authority is absent or crosses artifacts")
            if family != plan.families[node.node_id]:
                raise ContractError("admitted source coherent parent differs from its compiled family")
            expected_frame = (CellKey(root.artifact, episode_id, None, cell.key.tick, "", source_branch)
                              if root.form == "coherent_frame_root" else None)
            if cell.frame_owner != expected_frame:
                raise ContractError("admitted source coherent parent differs from declared family authority")
            authority.append((node.node_id, source, root, node.schema_status,
                              node.schema_source, node.domain))
        authority = tuple(authority)
        encoded_value = store.x[index][store.value_present[index]].detach().cpu().tolist()
        if len(encoded_value) != (3 if kind == "vector" else 1):
            raise ContractError("source tensor component count differs from declared datatype")
        candidate = tuple(encoded_value) if kind == "vector" else encoded_value[0]
        if kind == "boolean":
            if candidate not in (0., 1.):
                raise ContractError("Boolean source tensor is not 0/1")
            candidate = bool(candidate)
        owner = tuple(class_catalog.resolve(source_owners[alias]) for alias in cell.key.owner_aliases)
        identity = (tuple((item.entity_id, item.ontology_class, item.ontology_classes) for item in owner),
                    episode_id, source_branch, branch_id, cell.key.artifact, cell.key.field)
        schema = (source_type, unit, unit_status)
        if identity not in source_series:
            source_series[identity] = (owner, kind, schema, authority, {})
        _, _, series_schema, series_authority, samples = source_series[identity]
        if schema != series_schema or authority != series_authority:
            raise ContractError("source:artifact:field series has conflicting physical or schema authority")
        if cell.key.tick in samples:
            raise ContractError("source series contains duplicate admitted sample ticks")
        samples[cell.key.tick] = (index, candidate)
    series_provenance = {}
    for identity in sorted(source_series):
        owner, kind, schema, authority, samples = source_series[identity]
        ordered = tuple(sorted(samples.items()))
        latest_tick, (_, candidate) = ordered[-1]
        key = VariableKey(owner, f"source:{identity[-2]}:{identity[-1]}",
                          latest_tick, episode_id, branch_id)
        if key in variables:
            raise ContractError("source:artifact:field authority collides with an existing variable")
        sample_indices = frozenset(index for _, (index, _) in ordered)
        predicate_reads = {}
        for pid in selected:
            members = frozenset(read.cell_index for read in reads
                                if read.predicate_id == pid and read.cell_index in sample_indices)
            if members:
                if members != sample_indices:
                    raise ContractError("shared source series requires equal exact per-predicate temporal read sets")
                predicate_reads[pid] = tuple(index for _, (index, _) in ordered)
        variables[key] = CandidateVariable(key, (candidate,),
            FieldSchema(key.field, kind, schema[1], "bound_source_cell"), observed=True,
            observed_value=candidate, history=tuple((tick, value) for tick, (_, value) in ordered[:-1]),
            source_refs=tuple(ref for item in authority for ref in item[4]),
            abstraction="admitted_observed_series")
        for _, (index, _) in ordered:
            source_keys[index] = key
        series_provenance[key] = MappingProxyType({
            "sample_cells": tuple((index, cells[index]) for _, (index, _) in ordered),
            "sample_reads": tuple((index, tuple(reads_by_cell[index])) for _, (index, _) in ordered),
            "source_authority": authority, "schema_authority": schema,
            "predicate_read_sets": MappingProxyType(predicate_reads),
            "binder_branch": source_branch, "model_branch": branch_id})
    incidences, parameter_report = [], {}
    for rule in rules:
        members = sorted({read.cell_index for read in reads
                          if read.predicate_id == rule.contract.predicate_id and read.cell_index in source_keys})
        if not members:
            raise ContractError("selected predicate has no admitted typed source dependency")
        member_keys = {source_keys[index] for index in members}
        scope_sources = tuple(key for key in series_provenance if key in member_keys)
        scope = (rule.output, *scope_sources)
        slots = tuple(owner_slots[key.owner[0].entity_id]+1 for key in scope_sources)
        constants, constant_ids, constant_contexts, nonnumeric = [], [], [], []
        for node in _value_parameter_nodes(plan.specs[rule.contract.predicate_id]):
            matches = {read.cell_index for read in reads if read.predicate_id == rule.contract.predicate_id
                       and read.node_id == node.node_id}
            if len(matches) != 1:
                raise ContractError(f"value parameter requires one exact supplied cell/read: {node.node_id}")
            cell = cells[matches.pop()]
            if (cell.status != "parameter" or cell.key.artifact != "dependency"
                    or cell.key.field != node.node_id or cell.key.entity is not None or cell.key.tick != cutoff):
                raise ContractError(f"value parameter binding differs from exact authority: {node.node_id}")
            if cell.value is None:
                raise ContractError(f"required producer parameter was not supplied: {node.node_id}")
            if node.data_type not in ("numeric", "integer"):
                nonnumeric.append(node.node_id)
                continue
            if type(cell.value) not in (int, float) or not math.isfinite(cell.value):
                raise ContractError(f"required producer parameter must be finite numeric: {node.node_id}")
            if node.node_id == "clock:tick" and cell.value != cutoff:
                raise ContractError("producer clock parameter differs from the causal cutoff")
            for context in _dependency_parameter_contexts(plan.specs[rule.contract.predicate_id], node):
                constants.append(float(cell.value))
                constant_ids.append(node.node_id)
                constant_contexts.append(context)
        incidences.append(DependencyIncidence(scope, (0, *slots), (-1, *slots),
                                             tuple(constants), tuple(constant_contexts)))
        parameter_report[rule.contract.predicate_id] = {
            "numeric_node_ids": tuple(constant_ids), "numeric_values": tuple(constants),
            "semantic_contexts": tuple(constant_contexts),
            "nonnumeric_excluded": tuple(nonnumeric), "numeric_count": len(constants)}
    attributes = declared_attribute_symbols(rules, variables,
        source_attributes={key: key.field for key in source_keys.values()})
    return SupervisedGraph(tuple(rules), variables, attributes, tuple(incidences), labels, cutoff, {
        "abstraction": "observed_prefix_supervised_truth", "source_cell_count": len(source_keys),
        "source_series_count": len(series_provenance),
        "source_history_sample_count": sum(len(variables[key].history) for key in series_provenance),
        "source_series_provenance": MappingProxyType(series_provenance),
        "source_cell_to_series": tuple((index, cell, source_keys.get(index)) for index, cell in enumerate(cells)),
        "source_read_to_series": tuple((read, source_keys.get(read.cell_index)) for read in reads),
        "excluded_cells": exclusions, "output_domain": output_domain,
        "dependency_scope_sizes": tuple(len(item.scope) for item in incidences),
        "producer_parameter_conditioning": parameter_report,
        "label_policy": "external_likelihood_only", "global_weather": (
            "formal_manifest_region_owned" if weather_projection is not None else
            "excluded_without_verified_formal_projection"),
        "parameter_cells": "outside_state_registry;numeric_value_closure_conditions_shared_scorer",
        "feature_policy": "admitted_entity_owned_SourceCell_tensors_only",
        "producer_execution": False, "numeric_future_model": False})


class UnifiedRuntime:
    """Reuse one encoder and one local scorer throughout all graph instances."""

    def __init__(self, vocabulary: SchemaVocabulary, *, encoder: SharedVariableEncoder | None = None,
                 scorer: SharedFactorNetwork | None = None, dtype: torch.dtype = torch.float64,
                 authoritative_parameters: Mapping[str, Any] | None = None,
                 inference_budget: InferenceBudget | None = None, local_assignment_budget: int = 100000):
        self.vocabulary = vocabulary
        self.encoder = SharedVariableEncoder(vocabulary) if encoder is None else encoder
        self.scorer = SharedFactorNetwork() if scorer is None else scorer
        self.encoder.to(dtype=dtype)
        self.scorer.to(dtype=dtype)
        self.dtype = dtype
        self.authoritative_parameters = dict(get_governed_parameter_defaults() if authoritative_parameters is None
                                             else authoritative_parameters)
        self.inference_budget = InferenceBudget() if inference_budget is None else inference_budget
        if type(local_assignment_budget) is not int or local_assignment_budget < 1:
            raise ValueError("local_assignment_budget must be an explicit positive integer")
        self.local_assignment_budget = local_assignment_budget

    def parameters(self):
        return (*self.encoder.parameters(), *self.scorer.parameters())

    def build(self, rules: Sequence[GroundedRule], variables: Mapping[VariableKey, CandidateVariable], *,
              attribute_symbols: Mapping[VariableKey, str],
              anchor_tick: int = 0, legal_lags: tuple[int, ...] = (5,),
              shared_role_associations: bool = True,
              dependency_incidences: Sequence[DependencyIncidence] = ()) -> RuntimeGraph:
        rules = tuple(rules)
        if not rules or not variables:
            raise ContractError("a runtime request requires grounded rules and explicit candidate variables")
        output_keys = tuple(rule.output for rule in rules)
        if len(set(output_keys)) != len(output_keys):
            raise ContractError("duplicate grounded consequent instances")
        for rule in rules:
            validate_rule_binding(rule, variables, self.authoritative_parameters)
        expected_attributes = declared_attribute_symbols(rules, variables,
            source_attributes={key: symbol for key, symbol in attribute_symbols.items()
                               if key in variables and variables[key].schema.source == "bound_source_cell"})
        if dict(attribute_symbols) != dict(expected_attributes):
            raise ContractError("packed attributes differ from exact grounded/source declarations")
        ordered_variables = tuple(variables.values())
        if any(key != variable.key for key, variable in variables.items()):
            raise ContractError("registry indexing differs from exact shared variable keys")
        parameter = next(self.encoder.parameters())
        batch = pack_variables(ordered_variables, self.vocabulary,
                               attribute_symbols=expected_attributes, anchor_tick=anchor_tick,
                               tick_seconds=0.1, dtype=self.dtype, device=parameter.device)
        encoded = self.encoder(batch)
        indices = {variable.key: index for index, variable in enumerate(ordered_variables)}
        finite = tuple(FiniteVariable(variable.key, _truth_inference_domain(variable),
                                     variable.observed_index, variable.abstraction)
                       for variable in ordered_variables)
        factors = []
        hard_scopes = []
        for index, rule in enumerate(rules):
            if rule.contract.semantic_mode == "direct_observation" and rule.producer_binding is None:
                continue
            scope = rule.scope
            cardinalities = tuple(len(variables[key].domain) for key in scope)
            assignment_count = math.prod(cardinalities)
            if assignment_count > self.local_assignment_budget:
                raise ValueError(f"blocked_budget: hard factor requires {assignment_count} assignments")
            values = [rule.log_support(dict(zip(scope, assignment))) for assignment in
                      product(*(variables[key].domain for key in scope))]
            table = torch.tensor(values, dtype=self.dtype, device=parameter.device).reshape(cardinalities)
            factors.append(LogFactor(scope, table, f"hard:{index}", hard=True))
            hard_scopes.append(scope)
        scopes = list(soft_scope_candidates(rules, variables, legal_lags=legal_lags))
        # Connections use the typed participants already declared by contracts.
        # Optional same-time associations never remove mandatory rule scopes.
        if shared_role_associations:
            for left, right in combinations(rules, 2):
                lk, rk = left.output, right.output
                if (lk.tick, lk.episode_id, lk.branch_id) == (rk.tick, rk.episode_id, rk.branch_id) and set(lk.owner).intersection(rk.owner):
                    scopes.append((lk, rk))
        scopes = tuple(dict.fromkeys(scopes))
        rule_by_scope = {rule.scope: rule for rule in rules}
        for index, scope in enumerate(scopes):
            rule = rule_by_scope.get(scope)
            if rule is not None:
                operators = rule.contract.operators
                paired = tuple((float(rule.parameters[name]), context)
                    for name in rule.contract.threshold_parameters
                    for context in _threshold_parameter_contexts(rule.contract, name))
                constants = tuple(value for value, _ in paired)
                constant_contexts = tuple(context for _, context in paired)
            else:
                temporal = len(scope) == 2 and scope[0].owner == scope[1].owner and scope[0].field == scope[1].field
                operators = ("transition",) if temporal else ("dependency",)
                constants = ()
                constant_contexts = ()
            scores = self.scorer.score_scope(encoded.tokens, encoded.domain_mask,
                tuple(indices[key] for key in scope), role_slots=tuple(range(len(scope))),
                unordered_groups=(-1,) * len(scope), operators=operators,
                parameter_values=constants, parameter_contexts=constant_contexts,
                assignment_budget=self.local_assignment_budget)
            factors.append(LogFactor(scope, scores, f"soft:{index}"))
        for index, incidence in enumerate(dependency_incidences):
            if not isinstance(incidence, DependencyIncidence) or any(key not in indices for key in incidence.scope):
                raise ContractError("dependency incidence references an absent typed variable")
            identities = {(key.episode_id, key.branch_id) for key in incidence.scope}
            if len(identities) != 1 or max(key.tick for key in incidence.scope) > anchor_tick:
                raise ContractError("dependency incidence crosses sample identity or causal cutoff")
            output, *source_keys = incidence.scope
            if output not in output_keys or variables[output].observed:
                raise ContractError("conditioned incidence must begin with an unobserved declared output")
            if any(not variables[key].observed or len(variables[key].domain) != 1 for key in source_keys):
                raise ContractError("observed-prefix incidence conditioning requires singleton admitted sources")
            candidates = len(variables[output].domain)
            if candidates > self.local_assignment_budget:
                raise ValueError("blocked_budget: conditioned dependency exceeds assignment budget")
            output_tokens = encoded.tokens[indices[output], :candidates]
            source_tokens = [encoded.tokens[indices[key], 0].expand(candidates, -1) for key in source_keys]
            tokens = torch.stack((output_tokens, *source_tokens), dim=1).unsqueeze(0)
            scores = self.scorer(tokens,
                role_mask=torch.ones(tokens.shape[:-1], dtype=torch.bool, device=parameter.device),
                role_slots=torch.tensor((incidence.role_slots,), dtype=torch.long, device=parameter.device),
                unordered_groups=torch.tensor((incidence.unordered_groups,), dtype=torch.long, device=parameter.device),
                operator_context=operator_features(("dependency",), dtype=self.dtype, device=parameter.device)[None],
                parameter_values=torch.tensor((incidence.parameter_values,), dtype=self.dtype, device=parameter.device),
                parameter_mask=torch.ones((1, len(incidence.parameter_values)), dtype=torch.bool, device=parameter.device),
                parameter_contexts=(incidence.parameter_contexts,))
            # The physical source domains are observed singletons. Conditioning
            # them before exact inference leaves only the output axis, while
            # every actor/station token participates in the same attention call.
            factors.append(LogFactor((output,), scores[0], f"dependency:{index}"))
        graph = FactorGraph(finite, tuple(factors))
        return RuntimeGraph(graph, rules, ordered_variables, encoded.tokens, encoded.domain_mask, scopes, {
            "variable_count": len(finite), "factor_count": len(factors), "hard_factor_count": len(hard_scopes),
            "factor_arities": [len(factor.scope) for factor in factors],
            "domain_cardinalities": [len(variable.domain) for variable in finite],
            "legal_lags": legal_lags, "shared_role_associations": shared_role_associations,
            "local_assignment_budget": self.local_assignment_budget,
            "encoder_instance": id(self.encoder), "scorer_instance": id(self.scorer),
            "direct_observation_rules": sum(rule.contract.semantic_mode == "direct_observation" for rule in rules),
            "dependency_incidence_count": len(dependency_incidences),
            "dependency_conditioning": "observed_singletons_before_factor_table",
            "dependency_scopes": tuple(item.scope for item in dependency_incidences),
            "hard_consequent_scope_complete": all(scope in scopes for scope in hard_scopes),
            "exactness": "declared_finite_candidate_abstraction",
        })

    def marginal(self, instance: RuntimeGraph, query: Sequence[VariableKey]):
        return infer(instance.graph, query=query, budget=self.inference_budget)

    def supervised_nll(self, instance: RuntimeGraph, labels: Mapping[VariableKey, int]) -> torch.Tensor:
        """Exact joint conditional NLL; labels never enter candidate encoding."""
        variables = {variable.key: variable for variable in instance.variables}
        outputs = {rule.output for rule in instance.rules}
        if not labels or any(key not in outputs or variables[key].observed for key in labels):
            raise ContractError("supervision must label unobserved declared consequents only")
        return -joint_log_probability(instance.graph, labels, budget=self.inference_budget)

    def event_pmf(self, instance: RuntimeGraph, sequences: Mapping[Any, Sequence[VariableKey]], *,
                  anchors: Mapping[Any, bool] | None = None):
        return first_opposite_pmf(instance.graph, sequences, anchors=anchors, budget=self.inference_budget)
