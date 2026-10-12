"""Shared local assignment potentials over explicit graph factor scopes."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from .variable_encoder import TOKEN_DIM, _numeric_features, _position_features

# Exact operator names and generic graph relations, independent of predicate IDs.
OPERATORS = ("all", "any", "not", "truthy", "falsy", "direct_truth", "lt", "lte",
             "gt", "gte", "equals", "in", "present", "gt_fields", "gt_constant",
             "absolute_difference_gt", "outside_range", "changed", "prior",
             "observation", "transition", "dependency", "event")


@dataclass(frozen=True)
class ParameterContext:
    """Declared parameter semantics; instance identities are never symbols.

    An unknown unit stays None or source_unspecified with its exact unit_status.
    Operand names describe a consuming read, rather than parameter row order.
    """
    field: str
    unit: str | None
    unit_status: str
    consuming_function: str
    operand: str
    relation: str

    def __post_init__(self):
        for name in ("field", "unit_status", "consuming_function", "operand", "relation"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"parameter context requires a declared {name}")
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit):
            raise ValueError("parameter context unit must be declared or explicitly absent")

    @property
    def symbols(self):
        return (self.field, "<undeclared_unit>" if self.unit is None else self.unit,
                self.unit_status, self.consuming_function, self.operand, self.relation)


def operator_features(operators: Iterable[str], *, dtype=torch.float32, device=None) -> torch.Tensor:
    result = torch.zeros(len(OPERATORS), dtype=dtype, device=device)
    for operator in operators:
        if operator not in OPERATORS:
            raise ValueError(f"unsupported declared neural operator context: {operator!r}")
        result[OPERATORS.index(operator)] += 1
    if not bool(result.any()):
        raise ValueError("a factor needs explicit operator or relation context")
    return result


class SharedFactorNetwork(nn.Module):
    """One role-aware set attention and MLP for every soft factor assignment.

    No factor/predicate/entity identity is accepted. Ordered operand slots are
    explicit incidence metadata. Every member of an unordered group must use
    the same slot, so swapping those members preserves the assignment score.
    Scores have a fixed range and zero mean over each factor's valid assignments;
    there is no trainable whole-factor scale or graph-level prediction head.
    """

    def __init__(self, token_dim: int = TOKEN_DIM, *, attention_heads: int = 4,
                 score_bound: float = 4.0):
        super().__init__()
        if token_dim != TOKEN_DIM or token_dim % attention_heads:
            raise ValueError("factor tokens must have 128 dimensions divisible by attention_heads")
        if not math.isfinite(score_bound) or score_bound <= 0:
            raise ValueError("score_bound must be a finite positive declared constant")
        self.token_dim, self.score_bound = token_dim, float(score_bound)
        self.role_adapter = nn.Linear(8, token_dim, bias=False)
        self.context_adapter = nn.Sequential(nn.Linear(len(OPERATORS), token_dim), nn.SiLU())
        self.parameter_symbol_embedding = nn.Embedding(256, token_dim)
        self.parameter_character_position = nn.Linear(8, token_dim, bias=False)
        self.parameter_adapter = nn.Sequential(nn.Linear(4 + 6 * token_dim, token_dim),
                                               nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.attention = nn.MultiheadAttention(token_dim, attention_heads, dropout=0.0, batch_first=True)
        self.assignment_norm = nn.LayerNorm(token_dim)
        self.local_potential = nn.Sequential(nn.Linear(3 * token_dim + 1, token_dim), nn.SiLU(),
                                             nn.Linear(token_dim, token_dim // 2), nn.SiLU(),
                                             nn.Linear(token_dim // 2, 1))

    def forward(self, assignment_tokens: torch.Tensor, *, role_mask: torch.Tensor,
                role_slots: torch.Tensor, unordered_groups: torch.Tensor,
                operator_context: torch.Tensor, assignment_mask: torch.Tensor | None = None,
                parameter_values: torch.Tensor | None = None,
                parameter_mask: torch.Tensor | None = None,
                parameter_contexts: Sequence[Sequence[ParameterContext]] | None = None) -> torch.Tensor:
        """Return scores [F,A] for tokens [F,A,R,128]; padding returns zero.

        unordered_groups [F,R] uses -1 for ordered operands and nonnegative IDs
        to establish membership and validate equal slots within each group.
        Each ordered role is a separate singleton. Group IDs are not
        neural inputs. Parameters [F,P] are ordinary constants, not variables.
        """
        if not isinstance(assignment_tokens, torch.Tensor) or assignment_tokens.ndim != 4 or assignment_tokens.shape[-1] != self.token_dim:
            raise ValueError("assignment_tokens must have shape [F,A,R,128]")
        factors, assignments, roles, _ = assignment_tokens.shape
        if min(factors, assignments, roles) < 1:
            raise ValueError("factor, assignment and role dimensions must be nonempty")
        parameter = self.role_adapter.weight
        if assignment_tokens.dtype != parameter.dtype or assignment_tokens.device != parameter.device:
            raise ValueError("factor tokens must match model dtype and device")
        if role_mask.shape != (factors, assignments, roles) or role_mask.dtype != torch.bool:
            raise ValueError("role_mask must be Boolean [F,A,R]")
        if role_slots.shape != (factors, roles) or unordered_groups.shape != (factors, roles):
            raise ValueError("role slots and unordered groups must have shape [F,R]")
        if role_slots.dtype != torch.long or unordered_groups.dtype != torch.long:
            raise TypeError("role slot/group metadata must use integer tensors")
        if bool((role_slots < 0).any()) or bool((unordered_groups < -1).any()):
            raise ValueError("operand slots must be nonnegative and group padding is -1")
        if operator_context.shape != (factors, len(OPERATORS)) or operator_context.dtype != parameter.dtype:
            raise ValueError("operator_context must be floating [F,len(OPERATORS)]")
        if not bool(torch.isfinite(operator_context).all()) or bool((operator_context < 0).any()) or not bool(operator_context.any(-1).all()):
            raise ValueError("each factor needs finite nonnegative explicit operator context")
        if assignment_mask is None:
            assignment_mask = torch.ones((factors, assignments), dtype=torch.bool, device=parameter.device)
        if assignment_mask.shape != (factors, assignments) or assignment_mask.dtype != torch.bool:
            raise ValueError("assignment_mask must be Boolean [F,A]")
        if not bool(assignment_mask.any(-1).all()):
            raise ValueError("every factor needs at least one declared assignment")
        tensors = (role_mask, role_slots, unordered_groups, operator_context, assignment_mask)
        if any(tensor.device != parameter.device for tensor in tensors):
            raise ValueError("factor metadata must share the model device")
        if bool((assignment_mask & ~role_mask.any(-1)).any()):
            raise ValueError("a valid assignment must contain an explicit scope")
        if not bool(torch.isfinite(assignment_tokens[role_mask & assignment_mask[..., None]]).all()):
            raise ValueError("active candidate tokens must be finite")
        for factor in range(factors):
            active_roles = role_mask[factor, assignment_mask[factor]].any(0)
            for group in torch.unique(unordered_groups[factor, active_roles]).tolist():
                if group >= 0:
                    slots = role_slots[factor, active_roles & (unordered_groups[factor] == group)]
                    if slots.unique().numel() != 1:
                        raise ValueError("unordered operands must share the same incidence slot")
        if (parameter_values is None) != (parameter_mask is None):
            raise ValueError("constant parameter values and masks must be supplied together")
        if parameter_values is None and parameter_contexts is not None:
            raise ValueError("parameter semantics require their paired values and masks")
        context = self.context_adapter(operator_context.detach())
        if parameter_values is not None:
            if parameter_values.ndim != 2 or parameter_values.shape[0] != factors or parameter_mask.shape != parameter_values.shape:
                raise ValueError("ordinary parameters need [F,P] values and masks")
            if parameter_values.dtype != parameter.dtype or parameter_mask.dtype != torch.bool or parameter_values.device != parameter.device or parameter_mask.device != parameter.device:
                raise ValueError("ordinary parameters must match model dtype/device and Boolean mask")
            if not bool(torch.isfinite(parameter_values[parameter_mask]).all()):
                raise ValueError("declared ordinary parameters must be finite")
            if (parameter_contexts is None or len(parameter_contexts) != factors
                    or any(len(row) != parameter_values.shape[1] for row in parameter_contexts)
                    or any(not isinstance(item, ParameterContext) for row in parameter_contexts for item in row)):
                raise ValueError("every parameter value requires an exact typed semantic context")
            safe_values = torch.where(parameter_mask, parameter_values.detach(), torch.zeros_like(parameter_values))
            # Character positions belong to each declared symbol, never to the
            # parameter rows. Nonlinear byte/position pairing preserves spelling.
            symbol_tokens = []
            for row in parameter_contexts:
                pairs = []
                for item in row:
                    channels = []
                    for symbol in item.symbols:
                        codes = torch.tensor(tuple(symbol.encode("utf-8")), dtype=torch.long,
                                             device=parameter.device)
                        positions = torch.arange(len(codes), dtype=parameter.dtype, device=parameter.device)
                        chars = self.parameter_symbol_embedding(codes)
                        chars = torch.nn.functional.silu(chars + self.parameter_character_position(
                            _position_features(positions)))
                        channels.append(chars.mean(0))
                    pairs.append(torch.cat(channels))
                symbol_tokens.append(torch.stack(pairs) if pairs else parameter.new_empty((0, 6 * self.token_dim)))
            semantic = torch.stack(symbol_tokens)
            parameter_tokens = self.parameter_adapter(torch.cat((_numeric_features(safe_values), semantic), dim=-1))
            context = context + (parameter_tokens * parameter_mask[..., None]).sum(-2) / parameter_mask.sum(-1).clamp_min(1)[..., None]
        valid_role_mask = role_mask & assignment_mask[..., None]
        safe_tokens = torch.where(valid_role_mask[..., None], assignment_tokens, torch.zeros_like(assignment_tokens))
        role_features = self.role_adapter(_position_features(role_slots.to(parameter.dtype)))
        tokens = safe_tokens + role_features[:, None] + context[:, None, None]
        flat_tokens = tokens.reshape(factors * assignments, roles, self.token_dim)
        flat_mask = valid_role_mask.reshape(factors * assignments, roles)
        # MHA needs one unmasked position even on padded assignments. Its result
        # is subsequently masked out; no fake assignment enters graph inference.
        attention_mask = flat_mask.clone()
        attention_mask[~attention_mask.any(-1), 0] = True
        attended, _ = self.attention(flat_tokens, flat_tokens, flat_tokens,
                                    key_padding_mask=~attention_mask, need_weights=False)
        attended = self.assignment_norm(flat_tokens + attended).reshape(factors, assignments, roles, self.token_dim)
        # Group IDs establish membership only. Each ordered incidence remains
        # a separate singleton, even when two incidences share an operand slot.
        memberships = []
        for factor in range(factors):
            groups = []
            unordered = {}
            for role, group in enumerate(unordered_groups[factor].tolist()):
                if group == -1:
                    groups.append([role])
                elif group in unordered:
                    groups[unordered[group]].append(role)
                else:
                    unordered[group] = len(groups)
                    groups.append([role])
            memberships.append(groups)
        group_roles = torch.zeros((factors, max(map(len, memberships)), roles),
                                  dtype=parameter.dtype, device=parameter.device)
        for factor, groups in enumerate(memberships):
            for group, members in enumerate(groups):
                group_roles[factor, group, members] = 1
        member_count = torch.einsum("fgr,far->fag", group_roles,
                                   valid_role_mask.to(parameter.dtype))
        active_groups = member_count > 0
        group_count = active_groups.sum(-1, keepdim=True)

        def group_mean(values):
            safe_values = torch.where(valid_role_mask[..., None], values,
                                      torch.zeros_like(values))
            means = torch.einsum("fgr,fard->fagd", group_roles, safe_values)
            means = means / member_count.clamp_min(1)[..., None]
            return means.sum(-2) / group_count.clamp_min(1)

        pooled = group_mean(attended)
        raw_mean = group_mean(safe_tokens)
        encoded_context = context[:, None].expand(factors, assignments, self.token_dim)
        arity = group_count.to(parameter.dtype).log1p()
        raw_scores = self.local_potential(torch.cat((pooled, raw_mean, encoded_context, arity), dim=-1)).squeeze(-1)
        bounded = self.score_bound * raw_scores.tanh()
        mean = (bounded * assignment_mask).sum(-1, keepdim=True) / assignment_mask.sum(-1, keepdim=True)
        return torch.where(assignment_mask, bounded - mean, torch.zeros_like(bounded))

    def score_scope(self, candidate_tokens: torch.Tensor, domain_mask: torch.Tensor,
                    scope_indices: Sequence[int], *, role_slots: Sequence[int],
                    unordered_groups: Sequence[int], operators: Iterable[str],
                    parameter_values: Sequence[float] = (),
                    parameter_contexts: Sequence[ParameterContext] = (),
                    assignment_budget: int = 100000) -> torch.Tensor:
        """Gather a scope's exact Cartesian assignments, preserving every row."""
        if candidate_tokens.ndim != 3 or candidate_tokens.shape[-1] != self.token_dim or domain_mask.shape != candidate_tokens.shape[:-1] or domain_mask.dtype != torch.bool:
            raise ValueError("scope scoring requires tokens[N,D,128] and Boolean domain_mask[N,D]")
        scope_indices, role_slots, unordered_groups = tuple(scope_indices), tuple(role_slots), tuple(unordered_groups)
        if not scope_indices or len(scope_indices) != len(set(scope_indices)):
            raise ValueError("a factor scope needs distinct explicit variable indices")
        if len(role_slots) != len(scope_indices) or len(unordered_groups) != len(scope_indices):
            raise ValueError("every scope incidence needs slot and unordered-group metadata")
        if any(type(index) is not int or not 0 <= index < candidate_tokens.shape[0] for index in scope_indices):
            raise ValueError("factor scope references an absent variable")
        domains = [domain_mask[index].nonzero().flatten() for index in scope_indices]
        cardinalities = tuple(domain.numel() for domain in domains)
        assignments = math.prod(cardinalities)
        if type(assignment_budget) is not int or assignment_budget < 1 or assignments > assignment_budget:
            raise ValueError(f"blocked_budget: local factor requires {assignments} assignments, limit {assignment_budget}")
        combinations = tuple(itertools.product(*(domain.tolist() for domain in domains)))
        if not combinations:
            raise ValueError("factor scope contains an empty candidate domain")
        selectors = torch.tensor(combinations, dtype=torch.long, device=candidate_tokens.device)
        variables = torch.tensor(scope_indices, dtype=torch.long, device=candidate_tokens.device)
        tokens = candidate_tokens[variables[None], selectors].unsqueeze(0)
        kwargs = dict(role_mask=torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device),
            role_slots=torch.tensor((role_slots,), dtype=torch.long, device=tokens.device),
            unordered_groups=torch.tensor((unordered_groups,), dtype=torch.long, device=tokens.device),
            operator_context=operator_features(operators, dtype=tokens.dtype, device=tokens.device).unsqueeze(0))
        if len(parameter_contexts) != len(parameter_values):
            raise ValueError("ordinary constants require one typed context per value")
        if parameter_values:
            kwargs["parameter_values"] = torch.tensor((tuple(parameter_values),), dtype=tokens.dtype, device=tokens.device)
            kwargs["parameter_mask"] = torch.ones_like(kwargs["parameter_values"], dtype=torch.bool)
            kwargs["parameter_contexts"] = (tuple(parameter_contexts),)
        return self(tokens, **kwargs)[0].reshape(cardinalities)
