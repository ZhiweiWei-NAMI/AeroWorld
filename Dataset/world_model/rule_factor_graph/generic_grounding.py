"""Typed grounding, explicit finite candidates, and dependency scopes."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, MutableMapping

from Dataset.semantic_truth.world_truth import MISSING, _missing_value, _value_at_path
from .ast_compiler import CompiledRule, compile_rule, set_field
from .unified_contracts import (
    ContractError, FieldSchema, PredicateContract, TRUTH_DOMAIN, canonical_state_field,
    canonical_class_is_a, canonical_class_uri, disjoint_class_pairs, field_owner_roles,
)


@dataclass(frozen=True, order=True)
class TypedEntity:
    entity_id: str
    ontology_class: str = field(compare=False)
    ontology_classes: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.entity_id or not isinstance(self.entity_id, str):
            raise ContractError("entity identity must be an explicit nonempty string")
        if not isinstance(self.ontology_class, str):
            raise ContractError("entity class must be an explicit ontology class")
        source_class = canonical_class_uri(self.ontology_class)
        classes = tuple(sorted(set(canonical_class_uri(value) for value in
            (self.ontology_classes or (source_class,)))))
        if not any(canonical_class_is_a(value, source_class) for value in classes):
            raise ContractError("source class is not supported by the explicit instance type assertions")
        conflicts = disjoint_class_pairs(classes)
        if conflicts:
            raise ContractError(f"instance has explicitly disjoint ontology types: {conflicts}")
        object.__setattr__(self, "ontology_class", source_class)
        object.__setattr__(self, "ontology_classes", classes)

    def supports_class(self, expected: str) -> bool:
        return any(canonical_class_is_a(value, expected) for value in self.ontology_classes)


@dataclass(frozen=True)
class RoleBinding:
    role: str
    entity: TypedEntity

    @property
    def entity_id(self):
        return self.entity.entity_id

    @property
    def ontology_class(self):
        return self.entity.ontology_class


@dataclass(frozen=True)
class EntityClassCatalog:
    """Explicit per-episode instance classes and their source references."""
    episode_id: str
    classes: Mapping[str, tuple[str, ...]]
    source_refs: Mapping[str, tuple[str, ...]]

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ContractError("instance class catalog requires an exact episode")
        canonical = {}
        for entity_id, classes in self.classes.items():
            if not classes or not self.source_refs.get(entity_id):
                raise ContractError("every instance catalog type requires explicit source evidence")
            canonical[entity_id] = tuple(dict.fromkeys(canonical_class_uri(value) for value in classes))
        object.__setattr__(self, "classes", MappingProxyType(canonical))
        object.__setattr__(self, "source_refs", MappingProxyType(dict(self.source_refs)))

    def resolve(self, entity: TypedEntity) -> TypedEntity:
        if entity.entity_id not in self.classes:
            raise ContractError(f"entity is absent from the explicit class catalog: {entity.entity_id}")
        classes = self.classes[entity.entity_id]
        if not any(canonical_class_is_a(value, entity.ontology_class) for value in classes):
            raise ContractError(f"binding class conflicts with instance catalog: {entity.entity_id}")
        return TypedEntity(entity.entity_id, entity.ontology_class, classes)


@dataclass(frozen=True)
class VariableKey:
    owner: tuple[TypedEntity, ...]
    field: str
    tick: int
    episode_id: str
    branch_id: str = "factual"

    def __post_init__(self):
        if not self.owner or any(not isinstance(owner, TypedEntity) for owner in self.owner):
            raise ContractError("variable owner must be an explicit typed entity or role tuple")
        if not isinstance(self.field, str) or not self.field:
            raise ContractError("variable semantic field must be explicit")
        if type(self.tick) is not int or self.tick < 0:
            raise ContractError("variable tick must be a nonnegative integer")
        if not all(isinstance(v, str) and v for v in (self.episode_id, self.branch_id)):
            raise ContractError("variable episode and branch provenance must be explicit")


def _candidate_signature(value):
    if isinstance(value, tuple):
        return (tuple, tuple(_candidate_signature(item) for item in value))
    return (type(value), value)


@dataclass(frozen=True)
class CandidateVariable:
    key: VariableKey
    domain: tuple[Any, ...]
    schema: FieldSchema
    observed: bool = False
    observed_value: Any = None
    history: tuple[tuple[int, Any], ...] = ()
    forecast_context: tuple[Any, ...] = ()
    source_refs: tuple[str, ...] = ()
    observable: bool = True
    abstraction: str = "finite_declared"
    preserved_rules: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.domain, tuple) or not self.domain:
            raise ContractError("every variable requires an explicit nonempty finite candidate domain")
        signatures = tuple(_candidate_signature(value) for value in self.domain)
        if len(set(signatures)) != len(signatures):
            raise ContractError("candidate domain contains duplicate typed values")
        for value in self.domain:
            self._validate_value(value)
        if type(self.observed) is not bool or type(self.observable) is not bool:
            raise ContractError("observed/observable metadata must be Boolean")
        if self.observed and _candidate_signature(self.observed_value) not in signatures:
            raise ContractError("observed value is outside the declared domain")
        if not self.abstraction:
            raise ContractError("finite candidate abstraction must be named")

    def _validate_value(self, value):
        if value is None:
            return  # Explicit missing candidate; hard evaluation remains unknown.
        kind = self.schema.data_type
        if kind == "numeric" and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ContractError(f"numeric candidate is invalid for {self.key.field}: {value!r}")
        if kind == "boolean" and type(value) is not bool:
            raise ContractError(f"Boolean candidate is invalid for {self.key.field}: {value!r}")
        if kind == "integer" and type(value) is not int:
            raise ContractError(f"integer candidate is invalid for {self.key.field}: {value!r}")
        if kind in {"categorical", "identifier"} and not isinstance(value, str):
            raise ContractError(f"categorical candidate is invalid for {self.key.field}: {value!r}")
        if kind == "four_value" and value not in TRUTH_DOMAIN:
            raise ContractError(f"truth candidate is invalid: {value!r}")
        if kind == "vector" and (not isinstance(value, tuple) or not value or
                any(type(item) not in (int, float) or not math.isfinite(item) for item in value)):
            raise ContractError(f"vector candidate is invalid for {self.key.field}")
        if kind not in {"numeric", "integer", "boolean", "categorical", "identifier", "four_value", "vector"}:
            raise ContractError(f"unknown declared data type: {kind}")

    @property
    def observed_index(self) -> int | None:
        if not self.observed:
            return None
        target = _candidate_signature(self.observed_value)
        return next(i for i, value in enumerate(self.domain) if _candidate_signature(value) == target)


@dataclass(frozen=True)
class ParameterProfile:
    profile_id: str
    parameters: Mapping[str, Any]
    source_ref: str

    def __post_init__(self):
        if not all(isinstance(value, str) and value for value in (self.profile_id, self.source_ref)):
            raise ContractError("parameter profile requires an exact identity and source")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in self.parameters.values()):
            raise ContractError("governed parameter values must be finite numeric constants")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True)
class OutputBindingAuthority:
    """Independent record metadata against which a grounded output is checked."""
    predicate_id: str
    bindings: tuple[RoleBinding, ...]
    tick: int
    episode_id: str
    branch_id: str
    parameter_profile_id: str
    parameters: Mapping[str, Any]
    source_refs: tuple[str, ...]

    def __post_init__(self):
        if not self.source_refs:
            raise ContractError("output authority must cite independent source metadata")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True)
class GroundedRule:
    contract: PredicateContract
    bindings: tuple[RoleBinding, ...]
    inputs: tuple[tuple[str, VariableKey], ...]
    output: VariableKey
    parameters: Mapping[str, Any]
    previous_inputs: tuple[tuple[str, VariableKey], ...] = ()
    applicable: bool = True
    source_out_of_scope: bool = False
    previous_absent_fields: tuple[str, ...] = ()
    parameter_profile: ParameterProfile | None = None
    output_authority: OutputBindingAuthority | None = None
    class_catalog: EntityClassCatalog | None = None
    producer_inputs: tuple[tuple[str, VariableKey], ...] = ()
    producer_binding: Any = None
    producer_nullable_fields: frozenset[str] = field(default=frozenset(), init=False, repr=False)
    _compiled_rule: CompiledRule | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.validate_output_binding()
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        if self.contract.semantic_mode == "declared_formula" and self.producer_binding is None:
            object.__setattr__(self, "_compiled_rule", compile_rule(self.contract))

    def validate_output_binding(self):
        if self.parameter_profile is None or self.output_authority is None:
            raise ContractError("every grounded rule needs an explicit parameter profile and independent output authority")
        expected_roles = tuple(role.key for role in self.contract.roles)
        if tuple(binding.role for binding in self.bindings) != expected_roles:
            raise ContractError("grounded output roles differ from the exact contract roles")
        # Recheck distinctness and ontology classes for manually constructed rules.
        bind_roles(self.contract, {binding.role: binding.entity for binding in self.bindings},
                   class_catalog=self.class_catalog)
        if self.class_catalog is not None and self.class_catalog.episode_id != self.output.episode_id:
            raise ContractError("instance class catalog belongs to another episode")
        if self.output.field != f"predicate:{self.contract.predicate_id}":
            raise ContractError("grounded output predicate identity differs from its rule")
        if self.output.owner != _owner(self.contract, self.bindings, expected_roles):
            raise ContractError("grounded output owner differs from all declared role bindings")
        expected_fields = (() if self.contract.semantic_mode == "direct_observation"
                           else tuple(field.field for field in self.contract.required_fields))
        if tuple(field for field, _ in self.inputs) != expected_fields:
            raise ContractError("grounded input fields differ from the exact state contract")
        if self.producer_binding is not None:
            from .state_dependencies import COMPUTE_COMM, DIRECT_PREDICATE_PRODUCERS, direct_monitor_fields
            binding, nodes = direct_monitor_fields(self.contract)
            if self.producer_binding != binding or tuple(field for field, _ in self.producer_inputs) != tuple(
                    node.source_cell.field_path for node in nodes):
                raise ContractError("fixed direct rule differs from its declared producer binding/operands")
            object.__setattr__(self, "producer_nullable_fields", frozenset(
                node.source_cell.field_path for node in nodes
                if node.source_cell.form == "published_monitor"
                and node.source_cell.coherent_parent is not None
                and node.data_type == "numeric"
                and node.domain.kind == "numeric_or_unknown"))
            owners = {key.owner for _, key in self.producer_inputs}
            if len(owners) != 1 or len(next(iter(owners))) != len(nodes[0].index.keys):
                raise ContractError("direct row operands require one ordered producer owner tuple")
            authority = self.contract.grounding_spec["candidate_authority"]["source"]
            producer = DIRECT_PREDICATE_PRODUCERS[authority]
            if producer.module != COMPUTE_COMM:
                raise ContractError("direct row owner gate requires the declared compute_comm producer")
            owner_gates = {
                "_build_communication_rows": (("actor", "station"), {
                    "actor": ("world:UnmannedAircraft",),
                    "station": ("world:CommunicationStation",)}),
                "_build_compute_rows": (("actor",), {
                    "actor": ("world:UnmannedAircraft", "world:GroundVehicle")}),
            }
            if binding.row_builder not in owner_gates:
                raise ContractError("direct row builder has no declared owner gate")
            coordinates, classes = owner_gates[binding.row_builder]
            if any(node.index.keys != coordinates for node in nodes):
                raise ContractError("direct row coordinates differ from the declared producer owner gate")
            for coordinate, owner in zip(nodes[0].index.keys, next(iter(owners))):
                if not any(owner.supports_class(allowed) for allowed in classes[coordinate]):
                    raise ContractError("direct row owner type conflicts with the producer coordinate")
                if self.class_catalog is None or self.class_catalog.resolve(owner) != owner:
                    raise ContractError("direct row owner requires independent catalog type authority")
            for node, (field, key) in zip(nodes, self.producer_inputs):
                if (key.field, key.tick, key.episode_id, key.branch_id) != (
                        f"source:{node.source_cell.artifact}:{field}", self.output.tick,
                        self.output.episode_id, self.output.branch_id):
                    raise ContractError("direct row operand field or sample identity differs from producer")
        elif self.producer_inputs:
            raise ContractError("producer operands require the declared direct binding")
        for field, key in self.inputs + self.previous_inputs:
            if key.owner != _owner(self.contract, self.bindings, field_owner_roles(self.contract, field)):
                raise ContractError(f"grounded state owner differs from field authority: {field}")
            if key.field != canonical_state_field(self.contract, field):
                raise ContractError(f"grounded state semantic field differs from source: {field}")
            if (key.episode_id, key.branch_id) != (self.output.episode_id, self.output.branch_id):
                raise ContractError("grounded state crosses output episode or branch")
        if any(key.tick != self.output.tick for _, key in self.inputs):
            raise ContractError("grounded current state has the wrong output time")
        if any(key.tick >= self.output.tick for _, key in self.previous_inputs):
            raise ContractError("grounded previous state is not earlier than the output")
        if set(self.parameters) != set(self.contract.threshold_parameters):
            raise ContractError("grounded threshold parameter names differ from contract")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in self.parameters.values()):
            raise ContractError("grounded thresholds must be finite numeric governed constants")
        for name, value in self.parameters.items():
            if name not in self.parameter_profile.parameters or value != self.parameter_profile.parameters[name]:
                raise ContractError(f"threshold differs from the declared parameter profile: {name}")
        authority = self.output_authority
        if (authority.predicate_id, canonical_role_bindings(self.contract, authority.bindings, self.class_catalog), authority.tick,
            authority.episode_id, authority.branch_id) != (
            self.contract.predicate_id, canonical_role_bindings(self.contract, self.bindings, self.class_catalog), self.output.tick,
            self.output.episode_id, self.output.branch_id):
            raise ContractError("grounded output differs from independent record binding authority")
        if authority.parameter_profile_id != self.parameter_profile.profile_id:
            raise ContractError("grounded output has the wrong governed parameter profile")
        if dict(authority.parameters) != dict(self.parameter_profile.parameters):
            raise ContractError("grounded output parameter snapshot differs from independent authority")

    @property
    def scope(self) -> tuple[VariableKey, ...]:
        return tuple(dict.fromkeys(key for _, key in self.inputs + self.previous_inputs + self.producer_inputs)) + (self.output,)

    def contexts(self, assignment: Mapping[VariableKey, Any]) -> tuple[dict, dict]:
        context, previous = {}, {}
        for field, key in self.inputs:
            set_field(context, field, assignment[key])
        for field, key in self.previous_inputs:
            if field not in self.previous_absent_fields:
                set_field(previous, field, assignment[key])
        return context, previous

    def log_support(self, assignment: Mapping[VariableKey, Any]) -> float:
        if self.producer_binding is not None:
            if assignment[self.output] not in TRUTH_DOMAIN:
                raise ContractError("fixed producer output requires four-valued truth")
            if not self.applicable or self.source_out_of_scope:
                truth = "out_of_scope"
            else:
                from .state_dependencies import DIRECT_PREDICATE_PRODUCERS
                from importlib import import_module
                authority = self.contract.grounding_spec["candidate_authority"]["source"]
                producer = DIRECT_PREDICATE_PRODUCERS[authority]
                operand_values = tuple(assignment[key] for _, key in self.producer_inputs)
                if any(value is None and field not in self.producer_nullable_fields
                       for (field, _), value in zip(self.producer_inputs, operand_values)):
                    truth = "unknown"
                else:
                    row = {}
                    for (field, _), value in zip(self.producer_inputs, operand_values):
                        set_field(row, field, value)
                    truth = getattr(import_module(producer.module[:-3].replace("/", ".")),
                                    self.producer_binding.function)(row)
            return 0.0 if assignment[self.output] == truth else -math.inf
        if self.contract.semantic_mode == "direct_observation":
            raise ContractError("direct-observation future labels have no declared physical hard factor")
        context, previous = self.contexts(assignment)
        return self._compiled_rule.log_support(assignment[self.output], context, previous,
            self.parameters, applicable=self.applicable, source_out_of_scope=self.source_out_of_scope)


def bind_roles(contract: PredicateContract, bindings: Mapping[str, Any], *,
               class_catalog: EntityClassCatalog | None = None) -> tuple[RoleBinding, ...]:
    if set(bindings) != {role.key for role in contract.roles}:
        raise ContractError("binding roles differ from the exact argument-role contract")
    resolved = []
    for role in contract.roles:
        raw = bindings[role.key]
        if isinstance(raw, TypedEntity):
            entity = raw
        elif isinstance(raw, Mapping):
            entity = TypedEntity(raw["entity_id"], raw["ontology_class"])
        elif isinstance(raw, tuple) and len(raw) == 2:
            entity = TypedEntity(*raw)
        else:
            raise ContractError("each role binding requires a typed identity")
        if class_catalog is not None:
            entity = class_catalog.resolve(entity)
        if not entity.supports_class(role.declared_class_uri or role.ontology_class):
            raise ContractError(f"role {role.key} class {entity.ontology_class} violates {role.ontology_class}")
        resolved.append(RoleBinding(role.key, entity))
    by_role = {item.role: item.entity for item in resolved}
    for group in contract.grounding_spec["distinct_role_sets"]:
        if len({by_role[role].entity_id for role in group}) != len(group):
            raise ContractError(f"distinct role constraint violated: {group}")
    return tuple(resolved)


def _owner(contract, bindings, owner_roles):
    by_role = {binding.role: binding.entity for binding in bindings}
    roles = list(owner_roles)
    owners = [by_role[role] for role in roles]
    for group in contract.grounding_spec["unordered_role_sets"]:
        positions = [i for i, role in enumerate(roles) if role in group]
        if len(positions) == len(group):
            ordered = sorted(owners[i] for i in positions)
            for i, entity in zip(positions, ordered):
                owners[i] = entity
    return tuple(owners)


def canonical_role_bindings(contract: PredicateContract, bindings: tuple[RoleBinding, ...],
                            class_catalog: EntityClassCatalog | None = None) -> tuple[RoleBinding, ...]:
    """Canonicalize only role permutations explicitly declared unordered."""
    roles = tuple(role.key for role in contract.roles)
    if tuple(binding.role for binding in bindings) != roles:
        raise ContractError("authority roles differ from the exact contract roles")
    if class_catalog is not None:
        bindings = bind_roles(contract, {binding.role: binding.entity for binding in bindings},
                              class_catalog=class_catalog)
    owners = _owner(contract, bindings, roles)
    return tuple(RoleBinding(role, entity) for role, entity in zip(roles, owners))


def _insert(registry: MutableMapping[VariableKey, CandidateVariable], variable: CandidateVariable):
    for key in registry:
        if (key.field, key.tick, key.episode_id, key.branch_id,
            tuple(owner.entity_id for owner in key.owner)) == (
            variable.key.field, variable.key.tick, variable.key.episode_id, variable.key.branch_id,
            tuple(owner.entity_id for owner in variable.key.owner)) and key.owner != variable.key.owner:
            raise ContractError("one state owner identity has conflicting canonical class declarations")
    if variable.key in registry:
        existing = registry[variable.key]
        # Sources can differ when one shared state appears in several factors.
        # Candidate/evidence disagreement is contract divergence, never fallback.
        if (existing.domain, existing.schema.data_type, existing.schema.unit,
            existing.observed, existing.observed_value, existing.history,
            existing.forecast_context, existing.observable, existing.abstraction) != (
            variable.domain, variable.schema.data_type, variable.schema.unit,
            variable.observed, variable.observed_value, variable.history,
            variable.forecast_context, variable.observable, variable.abstraction):
            raise ContractError(f"shared variable declarations disagree: {variable.key}")
        source_refs = tuple(dict.fromkeys(existing.source_refs + variable.source_refs))
        if source_refs != existing.source_refs:
            registry[variable.key] = replace(existing, source_refs=source_refs)
    else:
        registry[variable.key] = variable


def ground_contract(contract: PredicateContract, bindings: Mapping[str, Any],
                    context: Mapping[str, Any], parameters: Mapping[str, Any], *, tick: int,
                    episode_id: str, branch_id: str = "factual",
                    domain_by_field: Mapping[str, tuple[Any, ...]] | None = None,
                    observed_fields: Iterable[str] = (), previous_context: Mapping[str, Any] | None = None,
                    previous_tick: int | None = None, predicate_evidence: str | None = None,
                    output_domain: tuple[str, ...] = TRUTH_DOMAIN,
                    variable_registry: MutableMapping[VariableKey, CandidateVariable] | None = None,
                    history_by_field: Mapping[str, tuple[tuple[int, Any], ...]] | None = None,
                    forecast_by_field: Mapping[str, tuple[Any, ...]] | None = None,
                    source_refs: tuple[str, ...] = (), applicable: bool = True,
                    source_out_of_scope: bool = False, parameter_profile: ParameterProfile | None = None,
                    output_authority: OutputBindingAuthority | None = None,
                    class_catalog: EntityClassCatalog | None = None) -> GroundedRule:
    """Ground explicit candidates; forecast context never becomes truth evidence.

    Observed state can use a singleton domain.  Every unobserved physical field
    needs an explicit finite domain from the caller's abstraction contract.
    Direct-observation labels are one predicate variable; future source labels
    are never introduced as a second input variable.
    """
    compiled = compile_rule(contract)
    if class_catalog is not None and class_catalog.episode_id != episode_id:
        raise ContractError("instance class catalog belongs to another episode")
    resolved = bind_roles(contract, bindings, class_catalog=class_catalog)
    resolved_by_role = {binding.role: binding.entity_id for binding in resolved}
    for role, source in contract.grounding_spec["role_sources"].items():
        if source["kind"] == "context":
            value = _value_at_path(context, source["path"])
            if not _missing_value(value) and value != resolved_by_role[role]:
                raise ContractError(f"known context role identity disagrees with its binding: {role}")
    if parameter_profile is None or output_authority is None:
        raise ContractError("grounding requires an explicit parameter profile and independent output authority")
    # A failed binding must not leave partially inserted variables in the graph.
    registry = {} if variable_registry is None else dict(variable_registry)
    domains = {} if domain_by_field is None else domain_by_field
    observed = set(observed_fields)
    history = {} if history_by_field is None else history_by_field
    forecast = {} if forecast_by_field is None else forecast_by_field
    absent = set(contract.threshold_parameters) - set(parameters)
    if absent:
        raise ContractError(f"missing governed threshold parameters: {sorted(absent)}")
    inputs, previous_inputs, previous_absent = [], [], []
    if contract.semantic_mode != "direct_observation":
        for schema in contract.required_fields:
            value = _value_at_path(context, schema.field)
            is_observed = schema.field in observed and not _missing_value(value)
            if schema.field in domains:
                domain = tuple(domains[schema.field])
            elif is_observed:
                domain = (value,)
            else:
                raise ContractError(f"unobserved state needs a declared finite domain: {schema.field}")
            key = VariableKey(_owner(contract, resolved, field_owner_roles(contract, schema.field)),
                              canonical_state_field(contract, schema.field), tick, episode_id, branch_id)
            variable = CandidateVariable(key, domain, schema, is_observed,
                value if is_observed else None, tuple(history.get(schema.field, ())),
                tuple(forecast.get(schema.field, ())), source_refs,
                abstraction="finite_declared" if len(domain) > 1 else "observed_singleton")
            _insert(registry, variable)
            inputs.append((schema.field, key))
            if schema.field in compiled.previous_fields:
                if previous_tick is None or previous_tick >= tick:
                    raise ContractError("changed requires an explicit earlier dependency tick")
                prev_value = _value_at_path({} if previous_context is None else previous_context, schema.field)
                # The evaluator's missing previous value yields False for changed.
                if prev_value is MISSING:
                    previous_absent.append(schema.field)
                    prev_value = None
                prev_key = VariableKey(key.owner, key.field, previous_tick, episode_id, branch_id)
                _insert(registry, CandidateVariable(prev_key, (prev_value,), schema,
                    observed=prev_value is not None, observed_value=prev_value,
                    source_refs=source_refs, abstraction="declared_previous_evidence"))
                previous_inputs.append((schema.field, prev_key))
    output = VariableKey(_owner(contract, resolved, tuple(role.key for role in contract.roles)),
                         f"predicate:{contract.predicate_id}", tick, episode_id, branch_id)
    if contract.semantic_mode == "direct_observation" and predicate_evidence is None:
        direct_field = contract.required_fields[0].field
        if direct_field in observed:
            predicate_evidence = compiled.evaluate(context, previous_context, parameters,
                applicable=applicable, source_out_of_scope=source_out_of_scope)
    schema = FieldSchema(output.field, "four_value", "four-value", "predicate_contract", TRUTH_DOMAIN)
    _insert(registry, CandidateVariable(output, tuple(output_domain), schema,
        observed=predicate_evidence is not None, observed_value=predicate_evidence,
        history=tuple(history.get(output.field, ())),
        forecast_context=tuple(forecast.get(output.field, ())), source_refs=source_refs,
        abstraction="finite_truth_domain"))
    grounded = GroundedRule(contract, resolved, tuple(inputs), output,
        {name: parameters[name] for name in contract.threshold_parameters},
        tuple(previous_inputs), applicable, source_out_of_scope, tuple(previous_absent),
        parameter_profile, output_authority, class_catalog)
    if variable_registry is not None:
        variable_registry.update(registry)
    return grounded


def dependency_closure(rules: Iterable[GroundedRule], query_keys: Iterable[VariableKey]) -> tuple[GroundedRule, ...]:
    """Return complete connected rule closure, including shared-state neighbors."""
    rules = tuple(rules)
    reached = set(query_keys)
    included: set[int] = set()
    changed = True
    while changed:
        changed = False
        for i, rule in enumerate(rules):
            if i not in included and reached.intersection(rule.scope):
                included.add(i)
                reached.update(rule.scope)
                changed = True
    return tuple(rule for i, rule in enumerate(rules) if i in included)


def soft_scope_candidates(rules: Iterable[GroundedRule], variables: Mapping[VariableKey, CandidateVariable],
                          *, legal_lags: tuple[int, ...] = (5,)) -> tuple[tuple[VariableKey, ...], ...]:
    """Optional learned scopes follow hard scopes and declared same-state lags.

    lag=5 is the explicit formal capture cadence.  It declares a modeling
    dependency and does not invent an occurrence of the AST changed operator.
    """
    if any(type(lag) is not int or lag <= 0 for lag in legal_lags):
        raise ContractError("soft temporal lags must be explicit positive ticks")
    scopes = [rule.scope for rule in rules]
    for key in variables:
        for lag in legal_lags:
            if key.tick >= lag:
                previous = VariableKey(key.owner, key.field, key.tick-lag, key.episode_id, key.branch_id)
                if previous in variables:
                    scopes.append((previous, key))
    return tuple(dict.fromkeys(scopes))
