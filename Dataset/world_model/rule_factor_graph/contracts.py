"""Typed variables and provenance for the bounded corridor rule graph."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch


class GroundingError(ValueError):
    """An identity, executable rule, or source contract is inconsistent."""


class UnsupportedGroundingError(GroundingError):
    """A record is outside the graph's declared grounding scope."""


class VariableKind(str, Enum):
    EXOGENOUS = "Exogenous"
    STATE = "State"
    PREDICATE = "Predicate"
    EVENT = "Event"


@dataclass(frozen=True)
class StateKey:
    owner_id: str
    field: str
    episode_id: str
    tick: int
    branch_id: str

    def __post_init__(self):
        for name in ("owner_id", "field", "episode_id", "branch_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise GroundingError(f"{name} must be a nonempty identity")
        if type(self.tick) is not int or self.tick < 0:
            raise GroundingError("tick must be a nonnegative integer")


@dataclass(frozen=True)
class RoleBinding:
    role: str
    entity_id: str
    ontology_class: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value
               for value in (self.role, self.entity_id, self.ontology_class)):
            raise GroundingError("role binding needs role, identity, and ontology class")


@dataclass(frozen=True)
class Variable:
    key: StateKey
    kind: VariableKind
    value: bool | float | None = None
    observed: bool = False
    role_bindings: tuple[RoleBinding, ...] = ()
    source_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.key, StateKey) or not isinstance(self.kind, VariableKind):
            raise GroundingError("variable requires a typed key and kind")
        if type(self.observed) is not bool or (self.observed and self.value is None):
            raise GroundingError("observed variables need an explicit value")
        if len({binding.role for binding in self.role_bindings}) != len(self.role_bindings):
            raise GroundingError("duplicate variable role bindings")


@dataclass(frozen=True)
class FactorTemplate:
    template_id: str
    factor_type: Literal["deterministic_rule", "observation", "transition", "event"]
    target_field: str
    input_fields: tuple[str, ...]
    role_classes: tuple[tuple[str, str], ...]
    source_refs: tuple[str, ...]
    input_owner_roles: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self):
        if self.factor_type not in {"deterministic_rule", "observation", "transition", "event"}:
            raise GroundingError(f"unsupported factor type: {self.factor_type}")
        if not self.template_id or not self.target_field or not self.input_fields:
            raise GroundingError("factor template must declare its target and inputs")
        if len(self.input_owner_roles) != len(self.input_fields):
            raise GroundingError("factor template must declare ownership for every input field")


@dataclass(frozen=True)
class GroundFactor:
    factor_id: str
    template: FactorTemplate
    inputs: tuple[StateKey, ...]
    output: StateKey
    bindings: tuple[RoleBinding, ...]
    source_refs: tuple[str, ...]

    def __post_init__(self):
        if tuple(key.field for key in self.inputs) != self.template.input_fields:
            raise GroundingError("ground factor input fields differ from its template")
        if self.output.field != self.template.target_field:
            raise GroundingError("ground factor target differs from its template")
        if tuple((b.role, b.ontology_class) for b in self.bindings) != self.template.role_classes:
            raise GroundingError("ground factor role classes differ from its template")
        identity = (self.output.episode_id, self.output.tick, self.output.branch_id)
        if any((key.episode_id, key.tick, key.branch_id) != identity for key in self.inputs):
            raise GroundingError("factor inputs cross episode, tick, or branch identities")
        if self.output.owner_id != "|".join(b.entity_id for b in self.bindings):
            raise GroundingError("ground factor output has the wrong role owner")
        role_ids = {binding.role: binding.entity_id for binding in self.bindings}
        for key, owners in zip(self.inputs, self.template.input_owner_roles):
            try:
                owner_id = "|".join(role_ids[role] for role in owners) if owners else self.output.episode_id
            except KeyError as exc:
                raise GroundingError("factor input declares a missing owner role") from exc
            if key.owner_id != owner_id:
                raise GroundingError("ground factor input has the wrong shared State owner")


@dataclass(frozen=True)
class EdgeProvenance:
    source: StateKey
    target: StateKey
    factor_id: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class PredicateEvidence:
    predicate_id: str
    episode_id: str
    tick: int
    branch_id: str
    bindings: tuple[RoleBinding, ...]
    value: Literal["true", "false", "unknown", "out_of_scope"]
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class GroundedSlice:
    episode_id: str
    branch_id: str
    tick: int
    actor_id: str
    corridor_id: str
    owner_id: str
    actor_class: str
    corridor_class: str
    predicate_values: tuple[bool | None, bool | None, bool | None]
    predicate_observed: tuple[bool, bool, bool]
    inside: bool | None
    speed_mps: float | None
    tau_mps: float
    applicable: bool
    source_refs: tuple[str, ...]
    variables: tuple[Variable, ...] = ()
    factors: tuple[GroundFactor, ...] = ()
    edges: tuple[EdgeProvenance, ...] = ()
    variable_source_refs: tuple[tuple[str, tuple[str, ...]], ...] = ()
    authority_source_refs: tuple[str, ...] = ()

    def __post_init__(self):
        StateKey(self.owner_id, "slice", self.episode_id, self.tick, self.branch_id)
        if self.owner_id != f"{self.actor_id}|{self.corridor_id}":
            raise GroundingError("slice owner differs from actor/corridor roles")
        if len(self.predicate_values) != 3 or len(self.predicate_observed) != 3:
            raise GroundingError("predicate order must be O, M, S")
        if any(value is not None and type(value) is not bool for value in self.predicate_values):
            raise GroundingError("predicate values must be Boolean or unknown")
        if any(type(value) is not bool for value in self.predicate_observed):
            raise GroundingError("predicate observation mask must be Boolean")
        if any(mask != (value is not None)
               for value, mask in zip(self.predicate_values, self.predicate_observed)):
            raise GroundingError("predicate observation mask differs from evidence")
        if self.variables or self.factors or self.edges:
            self._validate_graph()

    def _validate_graph(self):
        by_key = {variable.key: variable for variable in self.variables}
        if len(by_key) != len(self.variables):
            raise GroundingError("duplicate variable keys in slice")
        identity = (self.episode_id, self.tick, self.branch_id)
        if any((key.episode_id, key.tick, key.branch_id) != identity for key in by_key):
            raise GroundingError("variable identity differs from slice")
        factor_ids = {factor.factor_id for factor in self.factors}
        if len(factor_ids) != len(self.factors):
            raise GroundingError("duplicate factor IDs")
        expected_edges = set()
        for factor in self.factors:
            if factor.output not in by_key or any(key not in by_key for key in factor.inputs):
                raise GroundingError("factor references a missing variable")
            if by_key[factor.output].kind == VariableKind.EXOGENOUS:
                raise GroundingError("exogenous variables cannot have incoming factors")
            if factor.template.factor_type == "deterministic_rule":
                if by_key[factor.output].kind != VariableKind.PREDICATE:
                    raise GroundingError("deterministic rules must produce predicates")
                if not any(by_key[key].kind == VariableKind.STATE for key in factor.inputs):
                    raise GroundingError("predicate rules require shared state inputs")
            expected_edges.update((key, factor.output, factor.factor_id) for key in factor.inputs)
        actual_edges = {(edge.source, edge.target, edge.factor_id) for edge in self.edges}
        if actual_edges != expected_edges or len(actual_edges) != len(self.edges):
            raise GroundingError("graph edges differ from grounded factor dependencies")


@dataclass(frozen=True, init=False)
class BranchState:
    """Frozen upstream state samples; q is not a predicate probability tensor.

    Columns correspond to exact State keys. Values are [N] for one sample or
    [K,N] for K samples. Observed entries are finite; absent entries are NaN.
    Construction and access copy tensors, and all samples are detached.
    """
    variables: tuple[Variable, ...]
    source_refs: tuple[str, ...]
    _sample_values: torch.Tensor
    _observed_mask: torch.Tensor

    def __init__(self, variables, sample_values, observed_mask, source_refs=()):
        variables = tuple(variables)
        if not variables or any(v.kind != VariableKind.STATE for v in variables):
            raise GroundingError("BranchState contains only typed State variables")
        if len({v.key for v in variables}) != len(variables):
            raise GroundingError("BranchState has duplicate state keys")
        if not isinstance(sample_values, torch.Tensor) or not sample_values.is_floating_point():
            raise GroundingError("state samples must be a floating-point tensor")
        if sample_values.ndim not in (1, 2) or sample_values.shape[-1] != len(variables):
            raise GroundingError("state sample columns must match variables")
        if not isinstance(observed_mask, torch.Tensor) or observed_mask.dtype != torch.bool:
            raise GroundingError("state observation mask must be a Boolean tensor")
        if observed_mask.shape not in (sample_values.shape, torch.Size((len(variables),))):
            raise GroundingError("state observation mask shape differs from samples")
        if observed_mask.device != sample_values.device:
            raise GroundingError("state samples and mask must share a device")
        mask = observed_mask.expand_as(sample_values)
        if not torch.isfinite(sample_values[mask]).all().item():
            raise GroundingError("observed state samples must be finite")
        if not torch.isnan(sample_values[~mask]).all().item():
            raise GroundingError("unknown state samples must be explicit NaN")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "source_refs", tuple(source_refs))
        object.__setattr__(self, "_sample_values", sample_values.detach().clone())
        object.__setattr__(self, "_observed_mask", observed_mask.detach().clone())

    @property
    def sample_values(self):
        return self._sample_values.clone()

    @property
    def q(self):
        return self.sample_values

    @property
    def observed_mask(self):
        return self._observed_mask.clone()

    def detached(self):
        return BranchState(self.variables, self._sample_values, self._observed_mask, self.source_refs)

    def values_for(self, keys: tuple[StateKey, ...]):
        indices = {variable.key: i for i, variable in enumerate(self.variables)}
        try:
            selected = [indices[key] for key in keys]
        except KeyError as exc:
            raise GroundingError(f"state sample binding is absent: {exc.args[0]}") from exc
        return self._sample_values[..., selected].clone()
