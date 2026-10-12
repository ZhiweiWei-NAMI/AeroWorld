"""Exact finite-domain factor graphs and events from their joint partition.

Exactness applies to the declared candidate domains, not a continuous-world
distribution. Variable identities are opaque keys; tensor axes follow the
declared factor scope, and no candidate assignments are deduplicated.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

import torch


class ZeroSupportError(ValueError):
    """The complete conditioned dependency closure has no supported assignment."""


class BlockedBudgetError(RuntimeError):
    """Exact inference exceeded its declared allocation budget."""

    def __init__(self, message: str, report: "InferenceReport") -> None:
        super().__init__(f"blocked_budget: {message}")
        self.report = report


@dataclass(frozen=True)
class FiniteVariable:
    key: Hashable
    domain: tuple[Any, ...]
    observed_index: int | None = None
    abstraction: str = "finite_declared"

    def __post_init__(self) -> None:
        if not isinstance(self.key, Hashable):
            raise TypeError("variable keys must be hashable")
        if not isinstance(self.domain, tuple) or not self.domain:
            raise ValueError("a variable must declare a nonempty tuple candidate domain")
        if not isinstance(self.abstraction, str) or not self.abstraction:
            raise ValueError("a variable must name its finite candidate abstraction")
        if self.observed_index is not None:
            _check_index(self.observed_index, len(self.domain), "observed_index")


@dataclass(frozen=True)
class LogFactor:
    scope: tuple[Hashable, ...]
    log_values: torch.Tensor
    factor_id: str
    hard: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.scope, tuple) or len(set(self.scope)) != len(self.scope):
            raise ValueError("factor scope must be a tuple of distinct variable keys")
        if not isinstance(self.factor_id, str) or not self.factor_id:
            raise ValueError("a factor requires a nonempty provenance identifier")
        if type(self.hard) is not bool:
            raise TypeError("hard must be Boolean")
        if not isinstance(self.log_values, torch.Tensor) or not self.log_values.is_floating_point():
            raise TypeError("factor tables must be floating tensors")
        if self.log_values.ndim != len(self.scope):
            raise ValueError("factor table rank must equal its scope arity")
        if bool((torch.isnan(self.log_values) | torch.isposinf(self.log_values)).any()):
            raise ValueError("factor tables cannot contain NaN or positive infinity")
        if self.hard and not bool(((self.log_values == 0) | torch.isneginf(self.log_values)).all()):
            raise ValueError("hard factor tables must contain only 0 and negative infinity")


@dataclass(frozen=True)
class FactorGraph:
    variables: tuple[FiniteVariable, ...]
    factors: tuple[LogFactor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.variables, tuple) or not isinstance(self.factors, tuple):
            raise TypeError("graph variables and factors must be tuples")
        if any(not isinstance(variable, FiniteVariable) for variable in self.variables):
            raise TypeError("graph variables must use FiniteVariable")
        if any(not isinstance(factor, LogFactor) for factor in self.factors):
            raise TypeError("graph factors must use LogFactor")
        by_key = {variable.key: variable for variable in self.variables}
        if len(by_key) != len(self.variables):
            raise ValueError("a shared variable key must occur exactly once")
        if len({factor.factor_id for factor in self.factors}) != len(self.factors):
            raise ValueError("duplicate factor provenance identifiers")
        devices = {factor.log_values.device for factor in self.factors}
        if len(devices) > 1:
            raise ValueError("all factor tables must be on one device")
        for factor in self.factors:
            if any(key not in by_key for key in factor.scope):
                raise ValueError("factor scope references a missing variable")
            expected = tuple(len(by_key[key].domain) for key in factor.scope)
            if factor.log_values.shape != expected:
                raise ValueError(f"factor {factor.factor_id!r} table shape differs from its declared domains")


@dataclass(frozen=True)
class InferenceBudget:
    max_table_entries: int = 1_000_000
    max_total_entries: int = 20_000_000
    max_active_variables: int = 10_000

    def __post_init__(self) -> None:
        for name in ("max_table_entries", "max_total_entries", "max_active_variables"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class InferenceReport:
    status: str
    algorithm: str
    variable_count: int
    active_variable_count: int
    factor_count: int
    hard_factor_count: int
    component_count: int
    conditioned_component_count: int
    factor_arities: tuple[int, ...]
    conditioned_factor_arities: tuple[int, ...]
    domain_cardinalities: tuple[tuple[Hashable, int], ...]
    elimination_order: tuple[Hashable, ...]
    induced_treewidth: int
    max_table_entries: int
    total_table_entries: int
    exactness: str = "exact_over_declared_finite_candidate_domains"


@dataclass(frozen=True)
class InferenceResult:
    logZ: torch.Tensor
    probabilities: torch.Tensor
    query: tuple[Hashable, ...]
    report: InferenceReport


@dataclass(frozen=True)
class EventConstraint:
    kind: str
    step: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("first_flip", "right_censored", "none"):
            raise ValueError("event kind must be first_flip, right_censored, or none")
        if self.kind == "none":
            if self.step is not None:
                raise ValueError("none does not take an event step")
        elif type(self.step) is not int:
            raise TypeError("first_flip and right_censored need an integer step")


@dataclass(frozen=True)
class _Table:
    scope: tuple[Hashable, ...]
    values: torch.Tensor


@dataclass(frozen=True)
class _Prepared:
    variables: tuple[FiniteVariable, ...]
    factors: tuple[LogFactor, ...]
    tables: tuple[_Table, ...]
    evidence: Mapping[Hashable, int]
    offset: torch.Tensor
    reference: torch.Tensor


def _check_index(value: int, cardinality: int, name: str) -> None:
    if type(value) is not int or not 0 <= value < cardinality:
        raise ValueError(f"{name} must be an integer candidate index in 0..{cardinality - 1}")


def dependency_closure(graph: FactorGraph, query: Sequence[Hashable]) -> FactorGraph:
    """Keep every factor touching a query, recursively, before conditioning.

    Disconnected components outside the query cancel in normalized queries.
    An empty query requests the complete graph and all scalar factors.
    """
    query = tuple(query)
    by_key = {variable.key: variable for variable in graph.variables}
    if len(set(query)) != len(query) or any(key not in by_key for key in query):
        raise ValueError("query keys must be distinct declared variables")
    if not query:
        return graph
    adjacent: dict[Hashable, list[int]] = {key: [] for key in by_key}
    for index, factor in enumerate(graph.factors):
        for key in factor.scope:
            adjacent[key].append(index)
    keys, selected, pending = set(query), set(), list(query)
    while pending:
        for factor_index in adjacent[pending.pop()]:
            if factor_index in selected:
                continue
            selected.add(factor_index)
            for key in graph.factors[factor_index].scope:
                if key not in keys:
                    keys.add(key)
                    pending.append(key)
    # Scalar factors are part of the same partition even though they have no edge.
    selected.update(index for index, factor in enumerate(graph.factors) if not factor.scope)
    return FactorGraph(tuple(variable for variable in graph.variables if variable.key in keys),
                       tuple(factor for index, factor in enumerate(graph.factors) if index in selected))


def _prepare(graph: FactorGraph, evidence: Mapping[Hashable, int] | None,
             closure_keys: Sequence[Hashable]) -> _Prepared:
    if not isinstance(graph, FactorGraph):
        raise TypeError("inference requires a FactorGraph")
    by_key = {variable.key: variable for variable in graph.variables}
    if evidence is not None and not isinstance(evidence, Mapping):
        raise TypeError("evidence must map variable keys to candidate indexes")
    supplied = {} if evidence is None else dict(evidence)
    for key, index in supplied.items():
        if key not in by_key:
            raise ValueError("evidence references a missing variable")
        _check_index(index, len(by_key[key].domain), "evidence")
    conditioned = {variable.key: variable.observed_index for variable in graph.variables
                   if variable.observed_index is not None}
    for key, index in supplied.items():
        if key in conditioned and conditioned[key] != index:
            raise ZeroSupportError("evidence conflicts with a declared observed candidate")
        conditioned[key] = index
    selected = dependency_closure(graph, closure_keys)
    conditioned = {key: index for key, index in conditioned.items() if key in {v.key for v in selected.variables}}
    reference = (selected.factors[0].log_values if selected.factors
                 else torch.tensor(0.0, dtype=torch.float64)).to(dtype=torch.float64)
    offset = reference.new_zeros(())
    tables = []
    for factor in selected.factors:
        values = factor.log_values.to(dtype=torch.float64)
        scope = list(factor.scope)
        for axis in range(len(scope) - 1, -1, -1):
            if scope[axis] in conditioned:
                values = values.select(axis, conditioned[scope[axis]])
                del scope[axis]
        finite = torch.isfinite(values)
        maximum = values[finite].max().detach() if bool(finite.any()) else values.new_zeros(())
        values = torch.where(finite, values - maximum, values)
        offset = offset + maximum
        tables.append(_Table(tuple(scope), values))
    return _Prepared(selected.variables, selected.factors, tuple(tables), conditioned, offset, reference)


def _safe_logsumexp(values: torch.Tensor, dim: int) -> torch.Tensor:
    reachable = torch.isfinite(values).any(dim=dim, keepdim=True)
    safe = torch.where(reachable, values, torch.zeros_like(values))
    result = torch.logsumexp(safe, dim=dim)
    return torch.where(reachable.squeeze(dim), result, torch.full_like(result, -torch.inf))


def _neighbors(keys: Sequence[Hashable], scopes: Sequence[Sequence[Hashable]]) -> dict[Hashable, set[Hashable]]:
    adjacent = {key: set() for key in keys}
    for scope in scopes:
        for key in scope:
            adjacent[key].update(other for other in scope if other != key)
    return adjacent


def _component_count(adjacent: Mapping[Hashable, set[Hashable]]) -> int:
    unseen, count = set(adjacent), 0
    while unseen:
        count += 1
        pending = [unseen.pop()]
        while pending:
            for neighbor in adjacent[pending.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
    return count


def _plan(prepared: _Prepared, retain: tuple[Hashable, ...], extras: tuple[_Table, ...],
          budget: InferenceBudget, output_query: tuple[Hashable, ...] = ()) -> InferenceReport:
    active = tuple(variable.key for variable in prepared.variables if variable.key not in prepared.evidence)
    cardinalities = {variable.key: len(variable.domain) for variable in prepared.variables}
    rank = {key: index for index, key in enumerate(active)}
    tables = prepared.tables + extras
    scopes = [table.scope for table in tables]
    adjacent = _neighbors(active, scopes)
    conditioned_components = _component_count(adjacent)
    components = _component_count(_neighbors(tuple(cardinalities), tuple(f.scope for f in prepared.factors)))
    remaining = set(active) - set(retain)
    order, width = [], 0
    maximum = max([prod(cardinalities[key] for key in scope) for scope in scopes] + [1])
    total = sum(prod(cardinalities[key] for key in scope) for scope in scopes)
    while remaining:
        def priority(key: Hashable) -> tuple[int, int, int]:
            neighbors = adjacent[key]
            fill = sum(other not in adjacent[neighbor] for index, neighbor in enumerate(neighbors)
                       for other in list(neighbors)[index + 1:])
            entries = cardinalities[key] * prod(cardinalities[neighbor] for neighbor in neighbors)
            return fill, entries, rank[key]
        key = min(remaining, key=priority)
        neighbors = adjacent[key]
        width = max(width, len(neighbors))
        entries = cardinalities[key] * prod(cardinalities[neighbor] for neighbor in neighbors)
        maximum, total = max(maximum, entries), total + entries
        for neighbor in neighbors:
            adjacent[neighbor].update(neighbors - {neighbor})
            adjacent[neighbor].discard(key)
        del adjacent[key]
        remaining.remove(key)
        order.append(key)
    final_entries = prod(cardinalities[key] for key in (output_query or retain))
    maximum, total = max(maximum, final_entries), total + final_entries
    width = max(width, len(retain) - 1)
    status = "blocked_budget" if (len(active) > budget.max_active_variables
             or maximum > budget.max_table_entries or total > budget.max_total_entries) else "passed"
    report = InferenceReport(status, "exact_variable_elimination", len(prepared.variables), len(active),
                             len(prepared.factors), sum(f.hard for f in prepared.factors), components, conditioned_components,
                             tuple(len(f.scope) for f in prepared.factors),
                             tuple(len(t.scope) for t in prepared.tables),
                             tuple(cardinalities.items()), tuple(order), width, maximum, total)
    if status != "passed":
        raise BlockedBudgetError(f"{len(active)} active variables, largest table {maximum} entries, "
                                 f"total planned table entries {total}", report)
    return report


def _align(table: _Table, scope: tuple[Hashable, ...], cardinalities: Mapping[Hashable, int]) -> torch.Tensor:
    present = tuple(key for key in scope if key in table.scope)
    permutation = tuple(table.scope.index(key) for key in present)
    values = table.values.permute(permutation) if permutation else table.values
    return values.reshape(tuple(cardinalities[key] if key in table.scope else 1 for key in scope))


def _combine(tables: Sequence[_Table], scope: tuple[Hashable, ...],
             cardinalities: Mapping[Hashable, int], reference: torch.Tensor) -> torch.Tensor:
    values = reference.new_zeros(tuple(cardinalities[key] for key in scope))
    for table in tables:
        values = values + _align(table, scope, cardinalities)
    return values


def _solve(prepared: _Prepared, retain: tuple[Hashable, ...], budget: InferenceBudget,
           extras: tuple[_Table, ...] = (), output_query: tuple[Hashable, ...] = ()) -> tuple[torch.Tensor, InferenceReport]:
    report = _plan(prepared, retain, extras, budget, output_query)
    cardinalities = {variable.key: len(variable.domain) for variable in prepared.variables}
    ranks = {variable.key: index for index, variable in enumerate(prepared.variables)}
    tables = list(prepared.tables + extras)
    for key in report.elimination_order:
        bucket = [table for table in tables if key in table.scope]
        tables = [table for table in tables if key not in table.scope]
        scope = tuple(sorted({other for table in bucket for other in table.scope} | {key}, key=ranks.__getitem__))
        values = _combine(bucket, scope, cardinalities, prepared.reference)
        axis = scope.index(key)
        tables.append(_Table(tuple(other for other in scope if other != key), _safe_logsumexp(values, axis)))
    values = _combine(tables, retain, cardinalities, prepared.reference)
    return values, report


def infer(graph: FactorGraph, query: Sequence[Hashable] = (), *,
          evidence: Mapping[Hashable, int] | None = None,
          budget: InferenceBudget | None = None) -> InferenceResult:
    """Return a normalized exact query joint and its closure log partition.

    Observed variables remain in output axes with one supported candidate.
    Empty query returns a scalar probability one and the complete logZ.
    """
    query = tuple(query)
    prepared = _prepare(graph, evidence, query)
    retain = tuple(key for key in query if key not in prepared.evidence)
    values, report = _solve(prepared, retain, InferenceBudget() if budget is None else budget, output_query=query)
    flat = values.reshape(-1)
    log_partition = _safe_logsumexp(flat, 0)
    if bool(torch.isneginf(log_partition)):
        raise ZeroSupportError("conditioned dependency closure has zero total support")
    probabilities = torch.exp(values - log_partition)
    cardinalities = {variable.key: len(variable.domain) for variable in prepared.variables}
    # Reinsert fixed query axes without discarding multiplicity on latent axes.
    for axis, key in enumerate(query):
        if key in prepared.evidence:
            indexes = torch.arange(cardinalities[key], device=probabilities.device)
            mask = (indexes == prepared.evidence[key]).to(dtype=probabilities.dtype)
            shape = [1] * (probabilities.ndim + 1)
            shape[axis] = cardinalities[key]
            probabilities = probabilities.unsqueeze(axis) * mask.reshape(shape)
    restored = log_partition + prepared.offset
    if not bool(torch.isfinite(restored)):
        raise ValueError("graph arithmetic overflowed the log partition")
    return InferenceResult(restored, probabilities, query, report)


def _index_mask(prepared: _Prepared, key: Hashable, index: int) -> _Table:
    by_key = {variable.key: variable for variable in prepared.variables}
    if key not in by_key:
        raise ValueError("constraint references a missing closure variable")
    _check_index(index, len(by_key[key].domain), "constraint")
    if key in prepared.evidence:
        value = 0.0 if prepared.evidence[key] == index else -torch.inf
        return _Table((), prepared.reference.new_tensor(value))
    values = prepared.reference.new_full((len(by_key[key].domain),), -torch.inf)
    values[index] = 0.0
    return _Table((key,), values)


def joint_log_probability(graph: FactorGraph, labels: Mapping[Hashable, int], *,
                          evidence: Mapping[Hashable, int] | None = None,
                          budget: InferenceBudget | None = None) -> torch.Tensor:
    """Marginalize all unlabelled variables in a partial joint observation."""
    if not isinstance(labels, Mapping) or not labels:
        raise ValueError("partial labels must be a nonempty key-to-index mapping")
    prepared = _prepare(graph, evidence, tuple(labels))
    selected_budget = InferenceBudget() if budget is None else budget
    denominator, _ = _solve(prepared, (), selected_budget)
    if bool(torch.isneginf(denominator)):
        raise ZeroSupportError("conditioned dependency closure has zero total support")
    numerator, _ = _solve(prepared, (), selected_budget,
                          tuple(_index_mask(prepared, key, index) for key, index in labels.items()))
    return numerator - denominator


def _truth_index(variable: FiniteVariable, value: bool) -> int:
    matches = [index for index, candidate in enumerate(variable.domain)
               if type(candidate) is bool and candidate is value]
    if len(variable.domain) != 2 or len(matches) != 1 or any(type(v) is not bool for v in variable.domain):
        raise ValueError("event truth variables must have a distinct Boolean candidate domain")
    return matches[0]


def _event_prepare(graph: FactorGraph, sequences: Mapping[Hashable, Sequence[Hashable]],
                   queried: Sequence[Hashable], anchors: Mapping[Hashable, bool] | None,
                   evidence: Mapping[Hashable, int] | None) -> tuple[_Prepared, dict[Hashable, bool]]:
    if not isinstance(sequences, Mapping) or not sequences:
        raise ValueError("event sequences must be a nonempty label-to-truth-keys mapping")
    by_key = {variable.key: variable for variable in graph.variables}
    anchor_values, anchor_evidence, closure = {}, {} if evidence is None else dict(evidence), []
    horizons = set()
    # Explicit known anchors condition the joint even when their event label is
    # absent from a partial event observation. Unknown unqueried anchors impose
    # no condition and cannot manufacture an event target.
    conditioning_labels = list(queried)
    if anchors is not None:
        if not isinstance(anchors, Mapping):
            raise TypeError("anchors must map event labels to Boolean truth")
        for label, value in anchors.items():
            if value is None and label not in queried:
                continue
            if type(value) is not bool:
                raise ValueError("unknown anchor truth cannot define an event target")
            if label not in conditioning_labels:
                conditioning_labels.append(label)
    for label in conditioning_labels:
        if label not in sequences:
            raise ValueError("event query has no declared truth sequence")
        sequence = tuple(sequences[label])
        if not 2 <= len(sequence) <= 11 or len(set(sequence)) != len(sequence):
            raise ValueError("event sequences require distinct anchor and 1..10 future truth variables")
        if any(key not in by_key for key in sequence):
            raise ValueError("event sequence references a missing truth variable")
        for key in sequence:
            _truth_index(by_key[key], False)
            _truth_index(by_key[key], True)
        horizons.add(len(sequence))
        first = by_key[sequence[0]]
        if anchors is not None and label in anchors:
            value = anchors[label]
            if type(value) is not bool:
                raise ValueError("unknown anchor truth cannot define an event target")
        else:
            index = anchor_evidence.get(first.key, first.observed_index)
            if index is None:
                raise ValueError("unknown anchor truth cannot define an event target")
            value = first.domain[index]
        index = _truth_index(first, value)
        if first.key in anchor_evidence and anchor_evidence[first.key] != index:
            raise ZeroSupportError("declared event anchor conflicts with evidence")
        anchor_evidence[first.key] = index
        anchor_values[label] = value
        closure.extend(sequence)
    if len(horizons) > 1:
        raise ValueError("joint event sequences must use one prediction horizon")
    return _prepare(graph, anchor_evidence, tuple(dict.fromkeys(closure))), anchor_values


def _event_masks(prepared: _Prepared, sequences: Mapping[Hashable, Sequence[Hashable]],
                 constraints: Mapping[Hashable, EventConstraint], anchors: Mapping[Hashable, bool]) -> tuple[_Table, ...]:
    by_key = {variable.key: variable for variable in prepared.variables}
    masks = []
    for label, event in constraints.items():
        if not isinstance(event, EventConstraint):
            raise TypeError("event constraints must use EventConstraint")
        sequence, initial = tuple(sequences[label]), anchors[label]
        horizon = len(sequence) - 1
        if event.kind == "first_flip":
            if not 1 <= event.step <= horizon:
                raise ValueError("first flip must lie inside the future horizon")
            for key in sequence[:event.step]:
                masks.append(_index_mask(prepared, key, _truth_index(by_key[key], initial)))
            key = sequence[event.step]
            masks.append(_index_mask(prepared, key, _truth_index(by_key[key], not initial)))
        else:
            final = horizon if event.kind == "none" else event.step
            if not 0 <= final <= horizon:
                raise ValueError("right censor step must lie inside anchor..horizon")
            for key in sequence[:final + 1]:
                masks.append(_index_mask(prepared, key, _truth_index(by_key[key], initial)))
    return tuple(masks)


def event_log_probability(graph: FactorGraph, sequences: Mapping[Hashable, Sequence[Hashable]],
                          constraints: Mapping[Hashable, EventConstraint], *,
                          anchors: Mapping[Hashable, bool] | None = None,
                          evidence: Mapping[Hashable, int] | None = None,
                          partial_labels: Mapping[Hashable, int] | None = None,
                          budget: InferenceBudget | None = None) -> torch.Tensor:
    """One normalized joint event probability, including partial censoring.

    Every event numerator uses the same centered factor tables as its
    denominator. A first flip leaves all later truth values unconstrained.
    Partial labels join the event in the numerator and are marginalized when
    absent. Labels are supervision and must not be in forecast inputs.
    """
    if not isinstance(constraints, Mapping) or not constraints:
        raise ValueError("event constraints must be a nonempty mapping")
    queried = tuple(constraints)
    prepared, initial = _event_prepare(graph, sequences, queried, anchors, evidence)
    if partial_labels:
        # Include every dependency of additional supervision in both partitions.
        closure = tuple(dict.fromkeys(tuple(v.key for v in prepared.variables) + tuple(partial_labels)))
        expanded_evidence = {} if evidence is None else dict(evidence)
        expanded_evidence.update(prepared.evidence)
        prepared = _prepare(graph, expanded_evidence, closure)
    selected_budget = InferenceBudget() if budget is None else budget
    denominator, _ = _solve(prepared, (), selected_budget)
    if bool(torch.isneginf(denominator)):
        raise ZeroSupportError("event anchor has zero joint support")
    masks = _event_masks(prepared, sequences, constraints, initial)
    if partial_labels:
        masks += tuple(_index_mask(prepared, key, index) for key, index in partial_labels.items())
    numerator, _ = _solve(prepared, (), selected_budget, masks)
    return numerator - denominator


def first_opposite_pmf(graph: FactorGraph, sequences: Mapping[Hashable, Sequence[Hashable]], *,
                       anchors: Mapping[Hashable, bool] | None = None,
                       evidence: Mapping[Hashable, int] | None = None,
                       budget: InferenceBudget | None = None) -> torch.Tensor:
    """Return [P, T] rows in mapping order, first flip bins then none.

    T=11 is the five-second horizon of ten half-second future bins. All rows
    come from one joint graph conditioned on the declared anchor truths.
    """
    queried = tuple(sequences)
    prepared, initial = _event_prepare(graph, sequences, queried, anchors, evidence)
    selected_budget = InferenceBudget() if budget is None else budget
    denominator, _ = _solve(prepared, (), selected_budget)
    if bool(torch.isneginf(denominator)):
        raise ZeroSupportError("event anchors have zero joint support")
    rows = []
    for label in queried:
        outcomes = [EventConstraint("first_flip", step) for step in range(1, len(sequences[label]))]
        outcomes.append(EventConstraint("none"))
        values = []
        for outcome in outcomes:
            numerator, _ = _solve(prepared, (), selected_budget,
                                  _event_masks(prepared, sequences, {label: outcome}, initial))
            values.append(torch.exp(numerator - denominator))
        rows.append(torch.stack(values))
    return torch.stack(rows)


def observation_event_constraint(values: Sequence[bool | None | str]) -> EventConstraint:
    """Use the observed prefix; later missing rows cannot erase a known flip."""
    values = tuple(values)
    if not 2 <= len(values) <= 11:
        raise ValueError("event observations require anchor and 1..10 future bins")
    if type(values[0]) is not bool:
        raise ValueError("unknown anchor truth cannot define an event target")
    for step, value in enumerate(values[1:], 1):
        if value is None or value in ("unknown", "out_of_scope", "missing"):
            return EventConstraint("right_censored", step - 1)
        if type(value) is not bool:
            raise TypeError("event observations must be Boolean or explicit missing status")
        if value != values[0]:
            return EventConstraint("first_flip", step)
    return EventConstraint("none")


def _mixture_weights(reference: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    branches = reference.shape[0]
    if weights is None:
        return reference.new_full((branches,), 1.0 / branches)
    if not isinstance(weights, torch.Tensor) or weights.shape != (branches,):
        raise ValueError("branch weights must have shape [K]")
    if not weights.is_floating_point() or weights.device != reference.device:
        raise ValueError("branch weights must be floating on the branch device")
    total = weights.sum()
    if (not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())
            or not bool(torch.isfinite(total)) or not bool(total > 0)):
        raise ValueError("branch weights need finite nonnegative positive total mass")
    return weights / total


def mixture_first_opposite_pmf(branch_pmfs: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Mix branch PMFs only after each branch has normalized its partition."""
    if not isinstance(branch_pmfs, torch.Tensor) or branch_pmfs.ndim != 3 or branch_pmfs.shape[0] < 1:
        raise ValueError("branch PMFs must have nonempty shape [K,P,T]")
    if not 2 <= branch_pmfs.shape[-1] <= 11 or branch_pmfs.shape[1] < 1:
        raise ValueError("branch PMFs require predicates and 1..10 future bins plus none")
    if not branch_pmfs.is_floating_point() or not bool(torch.isfinite(branch_pmfs).all()) or bool((branch_pmfs < 0).any()):
        raise ValueError("branch PMFs must be finite nonnegative floating probabilities")
    totals = branch_pmfs.sum(-1)
    if not torch.allclose(totals, torch.ones_like(totals), rtol=1e-6, atol=1e-8):
        raise ValueError("every branch PMF must already sum to one")
    return (_mixture_weights(branch_pmfs, weights)[:, None, None] * branch_pmfs).sum(0)


def mixture_event_nll(branch_log_probabilities: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Mix normalized joint event probabilities before taking their NLL."""
    if (not isinstance(branch_log_probabilities, torch.Tensor) or branch_log_probabilities.ndim != 1
            or branch_log_probabilities.shape[0] < 1 or not branch_log_probabilities.is_floating_point()):
        raise ValueError("branch event log probabilities must have nonempty floating shape [K]")
    if bool((torch.isnan(branch_log_probabilities) | torch.isposinf(branch_log_probabilities)).any()):
        raise ValueError("branch probabilities cannot contain NaN or positive infinity")
    if bool((branch_log_probabilities > 1e-6).any()):
        raise ValueError("branch event log probabilities must already be normalized")
    normalized = _mixture_weights(branch_log_probabilities, weights)
    positive = normalized > 0
    safe = torch.where(positive, normalized, torch.ones_like(normalized))
    log_weights = torch.where(positive, safe.log(), torch.full_like(safe, -torch.inf))
    return -_safe_logsumexp(branch_log_probabilities + log_weights, 0)
