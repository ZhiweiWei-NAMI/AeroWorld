"""Neural emissions and associations for a shared-speed corridor chain.

Forecast trajectories are noisy evidence. Their features and the observed
prefix are detached inputs; B and I are latent states normalized by the exact
joint chain kernel. This module has no future truth or label argument.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .corridor_state_space import state_assignments


HISTORY_FEATURE_NAMES = (
    "speed_over_tau", "speed_delta_over_tau", "velocity_x_over_tau",
    "velocity_y_over_tau", "velocity_z_over_tau", "inside_c1", "inside_c2",
    "signed_distance_c1_over_extent_scale", "signed_distance_c2_over_extent_scale",
    "history_fraction",
)
ACTOR_FEATURE_NAMES = ("speed_over_tau", "speed_delta_over_tau", "horizon_fraction")
CORRIDOR_FEATURE_NAMES = (
    "signed_distance_over_extent_scale", "log_extent_x", "log_extent_y",
    "log_extent_z", "horizon_fraction",
)
EMISSION_NAMES = ("B", "I1", "I2")
PERSISTENCE_NAMES = ("lambda_B", "lambda_I1", "lambda_I2")
ASSOCIATION_NAMES = ("gamma1", "gamma2")
FACTOR_NAMES = EMISSION_NAMES + PERSISTENCE_NAMES + ASSOCIATION_NAMES
MODES = ("emissions", "persist", "gamma")


@dataclass(frozen=True)
class CorridorScores:
    unary: torch.Tensor
    transitions: torch.Tensor
    coefficients: Mapping[str, torch.Tensor]
    raw_coefficients: Mapping[str, torch.Tensor]
    mode: str
    separate_b: bool

    def window(self) -> "CorridorScores":
        if self.unary.ndim != 4 or self.unary.shape[0] != 1:
            raise ValueError("window extraction requires one batched window")
        return CorridorScores(
            self.unary[0], self.transitions[0],
            {name: value[0] for name, value in self.coefficients.items()},
            {name: value[0] for name, value in self.raw_coefficients.items()},
            self.mode, self.separate_b,
        )


def _check_mode(mode: str, separate_b: bool, knockout: Sequence[str]):
    if mode not in MODES:
        raise ValueError(f"unknown corridor factor mode: {mode!r}")
    if type(separate_b) is not bool:
        raise TypeError("separate_b must be a fixed Boolean structure choice")
    names = tuple(knockout)
    if len(names) != len(set(names)) or any(
        name not in PERSISTENCE_NAMES + ASSOCIATION_NAMES for name in names
    ):
        raise ValueError("knockout must name unique corridor persistence or gamma factors")
    return frozenset(names)


def corridor_potential_bases(*, separate_b=False, dtype=torch.float64, device=None):
    """Return fixed main-effect, agreement and requested raw gamma bases.

    Shared state order is lexicographic (B,I1,I2), with no deduplication of
    identical projected predicate rows. The ablation uses (B1,B2,I1,I2).
    Equality has fixed scale one. Raw gamma indicators are linearly independent
    of the centered main effects on the full cube; no extra gate is learned.
    """
    _check_mode("gamma", separate_b, ())
    states = state_assignments(separate_b=separate_b, device=device).to(dtype=dtype)
    if separate_b:
        b_values, i_values = states[:, :2], states[:, 2:]
        b_unary = (b_values - .5).sum(dim=-1)
        b_agreement = (b_values[:, None, :] == b_values[None, :, :]).to(dtype).sum(-1)
    else:
        b_values, i_values = states[:, :1].expand(-1, 2), states[:, 1:]
        b_unary = states[:, 0] - .5
        b_agreement = (states[:, None, 0] == states[None, :, 0]).to(dtype)
    main = torch.cat((b_unary[:, None], i_values - .5), dim=-1).T
    gamma = ((1 - b_values) * i_values).T
    agreement_i = (i_values[:, None, :] == i_values[None, :, :]).to(dtype)
    temporal = torch.cat((b_agreement[..., None], agreement_i), dim=-1).permute(2, 0, 1)
    return main, gamma, temporal


def corridor_basis_diagnostics(support: torch.Tensor | None = None, *, separate_b=False) -> dict:
    main, gamma, temporal = corridor_potential_bases(separate_b=separate_b)
    design = torch.cat((torch.ones_like(main[:1]), main, gamma), dim=0).T
    report = {
        "state_order": state_assignments(separate_b=separate_b).tolist(),
        "node_columns": ["1", "sum(Bc-1/2)" if separate_b else "B-1/2",
                         "I1-1/2", "I2-1/2", "(1-B1)*I1" if separate_b else "(1-B)*I1",
                         "(1-B2)*I2" if separate_b else "(1-B)*I2"],
        "node_rank": int(torch.linalg.matrix_rank(design).item()),
        "temporal_columns": list(PERSISTENCE_NAMES),
        "temporal_rank": int(torch.linalg.matrix_rank(temporal.flatten(1).T).item()),
        "fixed_scale": 1,
        "main_effect_center": "binary uniform reference 1/2",
        "gamma_basis": "requested raw indicator (1-B)*Ic; no trainable gate or extra scale",
        "identifiability_limit": "full-domain rank does not identify every context-dependent coefficient; geometry, anchor clamps and observed events can restrict effective rank",
    }
    if support is not None:
        if support.ndim != 2 or support.shape[-1] != design.shape[0] or support.dtype != torch.bool:
            raise ValueError("support rank diagnostics require Boolean [T,S]")
        if not bool(support.any(-1).all()):
            raise ValueError("support must contain at least one state at every time")
        selected = support.detach().cpu()
        rows = []
        for time, active in enumerate(selected):
            current = design[active]
            additive, interaction = current[:, :4], current[:, 4:]
            residual = interaction - additive @ torch.linalg.pinv(additive) @ interaction
            rows.append({
                "time": time, "active_states": int(active.sum()),
                "node_rank": int(torch.linalg.matrix_rank(current).item()),
                "gamma_rank_after_main_effects": int(torch.linalg.matrix_rank(residual, atol=1e-10).item()),
            })
        report["active_support_rank"] = rows
    return report


def corridor_scores_from_coefficients(
    b: torch.Tensor, inside: torch.Tensor, persistence_b: torch.Tensor,
    persistence_i: torch.Tensor, gamma: torch.Tensor, *, mode="gamma",
    separate_b=False, knockout=(),
) -> CorridorScores:
    """Compile named coefficients without a learned dense-state bypass.

    Inputs retain arbitrary leading dimensions and end in [T], [T,2],
    [T-1], [T-1,2], [T,2]. In the separate-B ablation, the same B head and
    lambda head supply both independent B variables, retaining model capacity.
    """
    disabled = _check_mode(mode, separate_b, knockout)
    tensors = (b, inside, persistence_b, persistence_i, gamma)
    if any(not isinstance(value, torch.Tensor) or not value.is_floating_point() for value in tensors):
        raise TypeError("corridor coefficients must be floating point tensors")
    if b.ndim < 1 or not 1 <= b.shape[-1] <= 11:
        raise ValueError("B coefficients must end in T with 1<=T<=11")
    prefix, steps = b.shape[:-1], b.shape[-1]
    expected = ((*prefix, steps, 2), (*prefix, steps - 1),
                (*prefix, steps - 1, 2), (*prefix, steps, 2))
    if any(value.shape != shape for value, shape in zip(tensors[1:], expected)):
        raise ValueError("I, lambda_B, lambda_I, gamma coefficient shapes disagree with B")
    if any(value.dtype != b.dtype or value.device != b.device for value in tensors):
        raise ValueError("all corridor coefficients must share dtype and device")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("corridor coefficients must be finite")
    if bool((persistence_b < 0).any()) or bool((persistence_i < 0).any()):
        raise ValueError("agreement coefficients must be nonnegative")
    raw = {
        "B": b, "I1": inside[..., 0], "I2": inside[..., 1],
        "lambda_B": persistence_b, "lambda_I1": persistence_i[..., 0],
        "lambda_I2": persistence_i[..., 1], "gamma1": gamma[..., 0], "gamma2": gamma[..., 1],
    }
    effective = dict(raw)
    for name in PERSISTENCE_NAMES:
        if mode == "emissions" or name in disabled:
            effective[name] = torch.zeros_like(raw[name])
    for name in ASSOCIATION_NAMES:
        if mode != "gamma" or name in disabled:
            effective[name] = torch.zeros_like(raw[name])
    main, local, edge = corridor_potential_bases(separate_b=separate_b, dtype=b.dtype, device=b.device)
    main_coefficients = torch.stack([effective[name] for name in EMISSION_NAMES], dim=-1)
    gamma_coefficients = torch.stack([effective[name] for name in ASSOCIATION_NAMES], dim=-1)
    persistence_coefficients = torch.stack([effective[name] for name in PERSISTENCE_NAMES], dim=-1)
    unary = main_coefficients @ main + gamma_coefficients @ local
    transitions = torch.einsum("...tf,fsr->...tsr", persistence_coefficients, edge)
    return CorridorScores(unary, transitions, effective, raw, mode, separate_b)


class CorridorNeuralFactors(nn.Module):
    """A prefix GRU and factor-specific MLPs output conditioned coefficients.

    The single B coefficient is computed from sampled actor speed and history
    once per actor/time. I heads share weights across both corridors. Lambda
    heads use consecutive encoded contexts and softplus; gamma is signed.
    Structure choices are fixed outside optimization.
    """

    def __init__(self, history_dim=10, hidden_dim=16, *, mode="gamma", separate_b=False):
        super().__init__()
        _check_mode(mode, separate_b, ())
        if any(type(value) is not int or value <= 0 for value in (history_dim, hidden_dim)):
            raise ValueError("history and hidden dimensions must be positive integers")
        self.history_dim, self.hidden_dim = history_dim, hidden_dim
        self.actor_dim, self.corridor_dim = len(ACTOR_FEATURE_NAMES), len(CORRIDOR_FEATURE_NAMES)
        self.mode, self.separate_b = mode, separate_b
        self.history_encoder = nn.GRU(history_dim, hidden_dim, batch_first=True)
        self.b_context = nn.Sequential(nn.Linear(hidden_dim + self.actor_dim, hidden_dim), nn.Tanh())
        self.i_context = nn.Sequential(nn.Linear(hidden_dim + self.corridor_dim, hidden_dim), nn.Tanh())
        self.b_emission_head = nn.Linear(hidden_dim, 1)
        self.i_emission_head = nn.Linear(hidden_dim, 1)
        self.b_persistence_head = nn.Linear(2 * hidden_dim, 1)
        self.i_persistence_head = nn.Linear(2 * hidden_dim, 1)
        self.gamma_head = nn.Linear(2 * hidden_dim, 1)
        if mode == "emissions":
            self.b_persistence_head.requires_grad_(False)
            self.i_persistence_head.requires_grad_(False)
        if mode != "gamma":
            self.gamma_head.requires_grad_(False)

    def forward(self, history, actor_features, corridor_features, *, knockout=()) -> CorridorScores:
        _check_mode(self.mode, self.separate_b, knockout)
        if any(not isinstance(value, torch.Tensor) for value in (history, actor_features, corridor_features)):
            raise TypeError("history, actor evidence and corridor evidence must be tensors")
        if history.ndim != 3 or history.shape[-1] != self.history_dim or history.shape[1] < 1:
            raise ValueError("history must have shape [B,H,history_dim] with H>=1")
        if actor_features.ndim != 4 or actor_features.shape[-1] != self.actor_dim:
            raise ValueError("actor_features must have shape [B,K,T,3]")
        batch, branches, steps = actor_features.shape[:3]
        if batch != history.shape[0] or batch < 1 or branches < 1 or not 1 <= steps <= 11:
            raise ValueError("history and evidence need matching nonempty batches, K>=1 and 1<=T<=11")
        if corridor_features.shape != (batch, branches, steps, 2, self.corridor_dim):
            raise ValueError("corridor_features must have shape [B,K,T,2,5]")
        parameter = self.b_emission_head.weight
        tensors = (history, actor_features, corridor_features)
        if any(value.dtype != parameter.dtype or value.device != parameter.device for value in tensors):
            raise ValueError("model inputs must share parameter dtype and device")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("model inputs must be finite prefix and frozen-q evidence")
        _, hidden = self.history_encoder(history.detach())
        h = hidden[-1][:, None, None, :].expand(batch, branches, steps, self.hidden_dim)
        b_context = self.b_context(torch.cat((h, actor_features.detach()), dim=-1))
        i_h = h[..., None, :].expand(batch, branches, steps, 2, self.hidden_dim)
        i_context = self.i_context(torch.cat((i_h, corridor_features.detach()), dim=-1))
        b = self.b_emission_head(b_context).squeeze(-1)
        inside = self.i_emission_head(i_context).squeeze(-1)
        b_edge = torch.cat((b_context[:, :, :-1], b_context[:, :, 1:]), dim=-1)
        i_edge = torch.cat((i_context[:, :, :-1], i_context[:, :, 1:]), dim=-1)
        persistence_b = F.softplus(self.b_persistence_head(b_edge).squeeze(-1))
        persistence_i = F.softplus(self.i_persistence_head(i_edge).squeeze(-1))
        paired_b = b_context[..., None, :].expand_as(i_context)
        gamma = self.gamma_head(torch.cat((paired_b, i_context), dim=-1)).squeeze(-1)
        return corridor_scores_from_coefficients(
            b, inside, persistence_b, persistence_i, gamma,
            mode=self.mode, separate_b=self.separate_b, knockout=knockout,
        )

    def score_window(self, history, actor_features, corridor_features, *, knockout=()) -> CorridorScores:
        if history.ndim != 2 or actor_features.ndim != 3 or corridor_features.ndim != 4:
            raise ValueError("score_window requires history[H,F], actor[K,T,3], corridor[K,T,2,5]")
        return self(history[None], actor_features[None], corridor_features[None], knockout=knockout).window()

    def parameter_manifest(self) -> dict:
        names = list(HISTORY_FEATURE_NAMES) if self.history_dim == len(HISTORY_FEATURE_NAMES) else None
        factor_paths = {
            "B": "b_emission_head", "I1": "i_emission_head", "I2": "i_emission_head",
            "lambda_B": "b_persistence_head", "lambda_I1": "i_persistence_head",
            "lambda_I2": "i_persistence_head", "gamma1": "gamma_head", "gamma2": "gamma_head",
        }
        active = set(EMISSION_NAMES)
        if self.mode != "emissions":
            active.update(PERSISTENCE_NAMES)
        if self.mode == "gamma":
            active.update(ASSOCIATION_NAMES)
        return {
            "class": type(self).__name__, "mode": self.mode, "separate_b": self.separate_b,
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "history_shape": ["B", "H", self.history_dim],
            "actor_evidence_shape": ["B", "K", "T", self.actor_dim],
            "corridor_evidence_shape": ["B", "K", "T", 2, self.corridor_dim],
            "history_feature_names": names,
            "actor_feature_names": list(ACTOR_FEATURE_NAMES),
            "corridor_feature_names": list(CORRIDOR_FEATURE_NAMES),
            "layers": {
                "history_encoder": {"kind": "GRU", "input": self.history_dim, "hidden": self.hidden_dim,
                                    "layers": 1, "activations": "PyTorch GRU sigmoid reset/update gates and tanh candidate"},
                "b_context": {"kind": "Linear,Tanh", "input": self.hidden_dim + self.actor_dim, "output": self.hidden_dim},
                "i_context": {"kind": "Linear,Tanh", "input": self.hidden_dim + self.corridor_dim, "output": self.hidden_dim},
                "b_emission_head": {"kind": "Linear", "input": self.hidden_dim, "output": 1},
                "i_emission_head": {"kind": "Linear", "input": self.hidden_dim, "output": 1},
                "b_persistence_head": {"kind": "Linear,Softplus", "input": 2 * self.hidden_dim, "output": 1},
                "i_persistence_head": {"kind": "Linear,Softplus", "input": 2 * self.hidden_dim, "output": 1},
                "gamma_head": {"kind": "Linear", "input": 2 * self.hidden_dim, "output": 1},
            },
            "factor_tensors": [
                {"name": name, "shape": ["B", "K", "T-1" if name in PERSISTENCE_NAMES else "T"],
                 "parameter_path": factor_paths[name] + ".weight/.bias", "active": name in active,
                 "owners": ["actor:a", "corridor:c1" if name.endswith("1") else "corridor:c2"]
                 if name.startswith("I") or name.startswith("gamma") or name.startswith("lambda_I") else ["actor:a"]}
                for name in FACTOR_NAMES
            ],
            "formula": "u(z_t)=theta_B*(B-.5)+sum_c theta_Ic*(Ic-.5)+sum_c gamma_c*(1-B)*Ic; edge=sum_v lambda_v*1[v_t=v_(t+1)]",
            "conditioning": "all eight named coefficients are network-conditioned; lambda=softplus(raw); B emission appears once per actor/time in shared mode",
            "q_trainable": False, "input_detached": True, "hard_rules_trainable": False,
            "future_truth_forward_argument": False, "trainable_gates": False,
            "association_interpretation": "predictive hypothesis; no causal claim",
            "basis": corridor_basis_diagnostics(separate_b=self.separate_b),
            "parameter_tensors": {name: {"shape": list(p.shape), "trainable": p.requires_grad}
                                  for name, p in self.named_parameters()},
        }
