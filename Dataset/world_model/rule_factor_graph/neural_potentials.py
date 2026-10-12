"""Neural association potentials over a fixed, identifiable mixed-state basis.

The network produces coefficients, never predicate probabilities or dense state
tables.  A single chain partition normalizes all state and predicate assignments.
Structure is a fixed binary choice made outside weight optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

UNARY_NAMES = ("A", "L", "P")
LOCAL_ASSOCIATION_NAMES = ("AL", "AP")
TEMPORAL_ASSOCIATION_NAMES = ("AA", "QQ", "Aprev_Pnext", "Lprev_Anext")
ASSOCIATION_NAMES = LOCAL_ASSOCIATION_NAMES + TEMPORAL_ASSOCIATION_NAMES
STRUCTURES = {"none": (0, 0), "AL": (1, 0), "AP": (0, 1), "AL_AP": (1, 1)}


def potential_bases(*, dtype=torch.float64, device=None):
    """Return additive, local and temporal bases in the declared six-state order.

    Centers use the uniform reference domain, not a learned distribution:
    a=A-1/2, l=L-1/3, p=P-2/3.  Each basis has fixed scale one.  The QQ
    persistence basis is equality minus 1/3.  Temporal bases have zero row and
    column means, so additive node scores cannot absorb their interaction.
    """
    a = torch.tensor([0, 0, 0, 1, 1, 1], dtype=dtype, device=device) - 0.5
    l = torch.tensor([0, 0, 1, 0, 0, 1], dtype=dtype, device=device) - 1 / 3
    p = torch.tensor([0, 1, 1, 0, 1, 1], dtype=dtype, device=device) - 2 / 3
    q = torch.tensor([0, 1, 2, 0, 1, 2], device=device)
    unary = torch.stack((a, l, p))
    local = torch.stack((a * l, a * p))
    temporal = torch.stack((
        a[:, None] * a[None, :],
        (q[:, None] == q[None, :]).to(dtype=dtype) - 1 / 3,
        a[:, None] * p[None, :],
        l[:, None] * a[None, :],
    ))
    return unary, local, temporal


def basis_diagnostics(support: torch.Tensor | None = None) -> dict:
    unary, local, temporal = potential_bases()
    design = torch.cat((torch.ones((1, 6), dtype=torch.float64), unary, local)).T
    report = {
        "state_order": [[a, q] for a in (0, 1) for q in (0, 1, 2)],
        "local_columns": ["1", "A-1/2", "L-1/3", "P-2/3", "a*l", "a*p"],
        "local_rank": int(torch.linalg.matrix_rank(design).item()),
        "temporal_columns": list(TEMPORAL_ASSOCIATION_NAMES),
        "temporal_rank": int(torch.linalg.matrix_rank(temporal.reshape(4, -1).T).item()),
        "temporal_max_abs_row_mean": float(temporal.mean(-1).abs().max().item()),
        "temporal_max_abs_column_mean": float(temporal.mean(-2).abs().max().item()),
        "normalization": "fixed scale 1; uniform-domain centering",
        "structure_selection": "fixed binary local AL/AP mask; no trainable gates",
        "interpretation": "association scores; no causal interpretation",
        "identifiability_limit": "domain basis rank does not identify every context-conditioned coefficient; a fixed anchor collapses first-edge interactions into next-node main effects",
    }
    if support is not None:
        if not isinstance(support, torch.Tensor) or support.ndim != 2 or support.shape[-1] != 6 or support.dtype != torch.bool:
            raise ValueError("rank diagnostics require Boolean support[T,6]")
        if not bool(support.any(-1).all()):
            raise ValueError("rank diagnostics require nonempty support at every time")
        support = support.detach().cpu()
        node_reports, edge_reports = [], []
        for time, selected in enumerate(support):
            active = design[selected]
            main, interactions = active[:, :4], active[:, 4:]
            residual = interactions - main @ torch.linalg.pinv(main) @ interactions
            node_reports.append({
                "time": time, "active_states": int(selected.sum()),
                "design_rank": int(torch.linalg.matrix_rank(active, atol=1e-10).item()),
                "interaction_rank_after_main_effects": int(torch.linalg.matrix_rank(residual, atol=1e-10).item()),
            })
        for time in range(support.shape[0] - 1):
            left, right = support[time].nonzero().flatten(), support[time + 1].nonzero().flatten()
            active = temporal[:, left[:, None], right[None, :]].reshape(4, -1).T
            rows = torch.eye(left.numel(), dtype=torch.float64).repeat_interleave(right.numel(), dim=0)
            columns = torch.eye(right.numel(), dtype=torch.float64).repeat(left.numel(), 1)
            additive = torch.cat((rows, columns), dim=-1)
            residual = active - additive @ torch.linalg.pinv(additive) @ active
            edge_reports.append({
                "time": time, "active_pairs": int(left.numel() * right.numel()),
                "basis_rank": int(torch.linalg.matrix_rank(active, atol=1e-10).item()),
                "interaction_rank_after_main_effects": int(torch.linalg.matrix_rank(residual, atol=1e-10).item()),
            })
        report["active_node_support"] = node_reports
        report["active_edge_support"] = edge_reports
    return report


@dataclass(frozen=True)
class MixedScores:
    unary: torch.Tensor
    transitions: torch.Tensor
    coefficients: Mapping[str, torch.Tensor]
    raw_coefficients: Mapping[str, torch.Tensor]
    structure_mask: tuple[int, int]

    def window(self) -> "MixedScores":
        if self.unary.ndim != 4 or self.unary.shape[0] != 1:
            raise ValueError("window extraction requires one batched window")
        return MixedScores(
            self.unary[0], self.transitions[0],
            {key: value[0] for key, value in self.coefficients.items()},
            {key: value[0] for key, value in self.raw_coefficients.items()},
            self.structure_mask,
        )


def _validate_mask(structure_mask):
    mask = tuple(structure_mask)
    if len(mask) != 2 or any(type(value) is not int or value not in (0, 1) for value in mask):
        raise ValueError("structure_mask must contain exactly two fixed binary integers")
    return mask


def _validate_knockout(knockout: Sequence[str]):
    names = tuple(knockout)
    if len(set(names)) != len(names) or any(name not in ASSOCIATION_NAMES for name in names):
        raise ValueError("knockout must name unique declared association factors")
    return frozenset(names)


def score_from_coefficients(
    unary_coefficients: torch.Tensor, local_coefficients: torch.Tensor,
    temporal_coefficients: torch.Tensor, *, structure_mask=(1, 1), knockout=(),
) -> MixedScores:
    """Compile fixed-basis coefficient tensors into joint chain potentials.

    Arbitrary leading dimensions are retained. Last dimensions are [T,3],
    [T,2], and [T-1,4]. This function also supports explicit hand-scored graphs.
    """
    tensors = (unary_coefficients, local_coefficients, temporal_coefficients)
    if any(not isinstance(value, torch.Tensor) or not value.is_floating_point() for value in tensors):
        raise TypeError("all coefficient arrays must be floating-point tensors")
    if unary_coefficients.ndim < 2 or unary_coefficients.shape[-1] != 3:
        raise ValueError("unary coefficients must have shape [..., T, 3]")
    time_steps = unary_coefficients.shape[-2]
    prefix = unary_coefficients.shape[:-2]
    if not 1 <= time_steps <= 11:
        raise ValueError("mixed chain time budget is 1..11")
    if local_coefficients.shape != (*prefix, time_steps, 2):
        raise ValueError("local coefficients must have shape [..., T, 2]")
    if temporal_coefficients.shape != (*prefix, time_steps - 1, 4):
        raise ValueError("temporal coefficients must have shape [..., T-1, 4]")
    if any(value.dtype != tensors[0].dtype or value.device != tensors[0].device for value in tensors):
        raise ValueError("coefficient arrays must share dtype and device")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("neural coefficient arrays must be finite")
    mask = _validate_mask(structure_mask)
    disabled = _validate_knockout(knockout)
    local_mask = local_coefficients.new_tensor([
        mask[index] * int(name not in disabled) for index, name in enumerate(LOCAL_ASSOCIATION_NAMES)
    ])
    temporal_mask = temporal_coefficients.new_tensor([
        int(name not in disabled) for name in TEMPORAL_ASSOCIATION_NAMES
    ])
    effective_local = local_coefficients * local_mask
    effective_temporal = temporal_coefficients * temporal_mask
    unary_basis, local_basis, temporal_basis = potential_bases(
        dtype=unary_coefficients.dtype, device=unary_coefficients.device,
    )
    unary = unary_coefficients @ unary_basis + effective_local @ local_basis
    transitions = torch.einsum("...tc,csr->...tsr", effective_temporal, temporal_basis)
    raw = {name: unary_coefficients[..., index] for index, name in enumerate(UNARY_NAMES)}
    raw.update({name: local_coefficients[..., index] for index, name in enumerate(LOCAL_ASSOCIATION_NAMES)})
    raw.update({name: temporal_coefficients[..., index] for index, name in enumerate(TEMPORAL_ASSOCIATION_NAMES)})
    effective = {name: raw[name] for name in UNARY_NAMES}
    effective.update({name: effective_local[..., index] for index, name in enumerate(LOCAL_ASSOCIATION_NAMES)})
    effective.update({name: effective_temporal[..., index] for index, name in enumerate(TEMPORAL_ASSOCIATION_NAMES)})
    return MixedScores(unary, transitions, effective, raw, mask)


class MixedNeuralPotentials(nn.Module):
    """Encode available history and frozen shared motion q into factor scores.

    Public forward takes [B,H,F] history and [B,K,T,D] branch context. Future
    communication observations and labels have no argument in this interface.
    Histories and branch features are detached; encoder and coefficient-head
    parameters are trainable. Branch weights remain the supplied uniform q.
    """

    def __init__(self, history_dim: int, branch_dim: int, hidden_dim: int = 32, *, structure="AL_AP"):
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (history_dim, branch_dim, hidden_dim)):
            raise ValueError("history, branch, and hidden dimensions must be positive integers")
        if structure not in STRUCTURES:
            raise ValueError(f"unknown fixed structure: {structure!r}")
        self.history_dim, self.branch_dim, self.hidden_dim = history_dim, branch_dim, hidden_dim
        self.structure = structure
        self.structure_mask = STRUCTURES[structure]
        self.history_encoder = nn.GRU(history_dim, hidden_dim, batch_first=True)
        self.context_encoder = nn.Sequential(nn.Linear(hidden_dim + branch_dim, hidden_dim), nn.Tanh())
        self.unary_coefficients = nn.Linear(hidden_dim, len(UNARY_NAMES))
        self.local_coefficients = nn.Linear(hidden_dim, len(LOCAL_ASSOCIATION_NAMES))
        self.temporal_coefficients = nn.Linear(2 * hidden_dim, len(TEMPORAL_ASSOCIATION_NAMES))

    def forward(self, history: torch.Tensor, branch_features: torch.Tensor, *, knockout=()) -> MixedScores:
        if not isinstance(history, torch.Tensor) or not isinstance(branch_features, torch.Tensor):
            raise TypeError("history and branch features must be tensors")
        if history.ndim != 3 or history.shape[-1] != self.history_dim or history.shape[1] < 1:
            raise ValueError("history must have shape [B,H,history_dim], H>=1")
        if branch_features.ndim != 4 or branch_features.shape[-1] != self.branch_dim:
            raise ValueError("branch_features must have shape [B,K,T,branch_dim]")
        if history.shape[0] != branch_features.shape[0] or history.shape[0] < 1 or branch_features.shape[1] < 1:
            raise ValueError("history and branch batches must match and contain branches")
        if not 1 <= branch_features.shape[2] <= 11:
            raise ValueError("mixed chain time budget is 1..11")
        parameter = self.unary_coefficients.weight
        if any(value.dtype != parameter.dtype or value.device != parameter.device for value in (history, branch_features)):
            raise ValueError("model inputs must match parameter dtype and device")
        if not bool(torch.isfinite(history).all() & torch.isfinite(branch_features).all()):
            raise ValueError("model inputs must be finite available-history and q features")
        _validate_knockout(knockout)
        _, hidden = self.history_encoder(history.detach())
        prefix = hidden[-1][:, None, None, :].expand(*branch_features.shape[:3], self.hidden_dim)
        encoded = self.context_encoder(torch.cat((prefix, branch_features.detach()), dim=-1))
        temporal_context = torch.cat((encoded[:, :, :-1], encoded[:, :, 1:]), dim=-1)
        return score_from_coefficients(
            self.unary_coefficients(encoded), self.local_coefficients(encoded),
            self.temporal_coefficients(temporal_context),
            structure_mask=self.structure_mask, knockout=knockout,
        )

    def score_window(self, history: torch.Tensor, branch_features: torch.Tensor, *, knockout=()) -> MixedScores:
        if history.ndim != 2 or branch_features.ndim != 3:
            raise ValueError("score_window requires history[H,F] and branch_features[K,T,D]")
        return self(history.unsqueeze(0), branch_features.unsqueeze(0), knockout=knockout).window()

    def association_manifest(self) -> dict:
        return {
            "structure": self.structure,
            "structure_mask": list(self.structure_mask),
            "hard_factors_trainable": False,
            "q_trainable": False,
            "edges": [
                {"name": name, "kind": "local", "scope": list(name),
                 "selected": bool(self.structure_mask[index])}
                for index, name in enumerate(LOCAL_ASSOCIATION_NAMES)
            ] + [
                {"name": name, "kind": "temporal", "scope": scope, "selected": True}
                for name, scope in zip(TEMPORAL_ASSOCIATION_NAMES, (
                    ["A_t", "A_t+1"], ["Q_t", "Q_t+1"], ["A_t", "P_t+1"], ["L_t", "A_t+1"],
                ))
            ],
            "basis": basis_diagnostics(),
        }
