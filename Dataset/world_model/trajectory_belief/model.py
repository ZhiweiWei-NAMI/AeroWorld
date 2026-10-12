"""A shared conditional hazard head over fixed trajectory draws."""
from __future__ import annotations

import math

import torch
from torch import nn

from Dataset.world_model.transition_prediction.model import censored_nll, log_distribution


HISTORY_STEPS = 21
FUTURE_STEPS = 10


def marginal_log_distribution(branch_logp: torch.Tensor) -> torch.Tensor:
    """Integrate complete first-event distributions, preserving none mass."""
    if branch_logp.ndim != 3 or branch_logp.shape[-1] != 11 or branch_logp.shape[1] < 1:
        raise ValueError("branch log probabilities must have shape [N,K,11], K >= 1")
    return torch.logsumexp(branch_logp, dim=1) - math.log(branch_logp.shape[1])


def _kinematics(state: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """Relative positions/velocities and pair motion, without absolute location."""
    extra = state.ndim - 3
    shape = (len(parameters),) + (1,) * extra
    horizontal = parameters[:, 1].reshape(shape)
    vertical = parameters[:, 2].reshape(shape)
    margin = parameters[:, 3].reshape(shape)
    delta = state[..., 1, :] - state[..., 0, :]
    scale = torch.stack([horizontal, horizontal, vertical], dim=-1)
    position = delta[..., :3] / scale
    velocity = delta[..., 3:] / scale
    pair_velocity = state[..., :, 3:].mean(dim=-2) / scale
    horizontal_ratio = torch.linalg.vector_norm(delta[..., :2], dim=-1) / horizontal
    vertical_ratio = delta[..., 2].abs() / vertical
    euclidean_ratio = torch.linalg.vector_norm(delta[..., :3], dim=-1) / margin
    return torch.cat([position, velocity, pair_velocity, horizontal_ratio[..., None],
                      vertical_ratio[..., None], euclidean_ratio[..., None]], dim=-1)


class ConditionalTrajectoryHead(nn.Module):
    """Calibrate an imperfect fixed q with equal capacity for every q mode.

    Exact sampled rule truth is an input calculated by the authoritative rule
    evaluator. Labels, donor identifiers, plans and factual future truth are
    absent from this interface. No gradient reaches sampled trajectories.
    """

    def __init__(self, width: int = 24):
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        self.width = width
        self.history_in = nn.Sequential(nn.Linear(13, width), nn.SiLU())
        self.history_gru = nn.GRU(width, width, batch_first=True)
        self.future_in = nn.Sequential(nn.Linear(15, width), nn.SiLU())
        self.future_gru = nn.GRU(2 * width + 5, width, batch_first=True)
        self.hazard = nn.Linear(width, 1)
        nn.init.zeros_(self.hazard.weight)
        nn.init.zeros_(self.hazard.bias)
        self.register_buffer("prior_logits", torch.zeros(2, FUTURE_STEPS))

    def forward(self, inputs: dict[str, torch.Tensor], return_branches: bool = False):
        history = inputs["history"].detach()
        future = inputs["sampled_future"].detach()
        parameters = inputs["rule_parameters"].detach()
        current = inputs["current"].long()
        history_truth = inputs["history_rule_truth"].detach()
        future_truth = inputs["sampled_rule_truth"].detach()
        n = len(history)
        if history.shape != (n, HISTORY_STEPS, 2, 6):
            raise ValueError("history must have shape [N,21,2,6]")
        if future.ndim != 5 or future.shape[0] != n or future.shape[2:] != (10, 2, 6):
            raise ValueError("sampled future must have shape [N,K,10,2,6]")
        k = future.shape[1]
        if k < 1 or parameters.shape != (n, 4) or current.shape != (n,):
            raise ValueError("invalid branch count or static input shape")
        if history_truth.shape != (n, 21) or future_truth.shape != (n, k, 10):
            raise ValueError("rule evaluations must match history and sampled future")
        if not all(torch.isfinite(value).all() for value in (history, future, parameters)):
            raise ValueError("trajectory head inputs must be finite")
        if (parameters[:, 1:] <= 0).any() or not torch.isin(parameters[:, 0], parameters.new_tensor([0, 1])).all():
            raise ValueError("rule parameters require metric 0/1 and positive H,V,margin")
        if not torch.isin(current, current.new_tensor([0, 1])).all():
            raise ValueError("current truth must be coded 0/1")
        history_features = torch.cat([_kinematics(history, parameters), history_truth[..., None].float()], dim=-1)
        _, hidden = self.history_gru(self.history_in(history_features))
        h = hidden[-1]
        time = torch.linspace(0.1, 1.0, 10, device=future.device, dtype=future.dtype)
        time = time[None, None, :, None].expand(n, k, -1, -1)
        opposite = future_truth != current[:, None, None].bool()
        future_features = torch.cat([_kinematics(future, parameters), future_truth[..., None].float(),
                                     opposite[..., None].float(), time], dim=-1)
        z = self.future_in(future_features)
        static = torch.cat([parameters[:, :1], parameters[:, 1:].log() / 5.0, current[:, None].float()], dim=-1)
        history_context = h[:, None, None].expand(-1, k, 10, -1)
        static_context = static[:, None, None].expand(-1, k, 10, -1)
        sequence = torch.cat([z, history_context, static_context], dim=-1).reshape(n * k, 10, -1)
        output, _ = self.future_gru(sequence)
        logits = self.hazard(output).reshape(n, k, 10) + self.prior_logits[current, None]
        branch_logp = log_distribution(logits.reshape(n * k, 10)).reshape(n, k, 11)
        logp = marginal_log_distribution(branch_logp)
        return (logp, branch_logp) if return_branches else logp


def analytical_ceiling(width: int, batch_size: int, k: int) -> dict:
    """Conservative dense arithmetic/storage estimates, not measured throughput."""
    model = ConditionalTrajectoryHead(width)
    parameters = sum(p.numel() for p in model.parameters())
    # Multiply-add counts: projection + GRU three gates + scalar projection.
    history_macs = 21 * (13 * width + 3 * (width * width + width * width))
    branch_macs = 10 * (15 * width + 3 * ((2 * width + 5) * width + width * width) + width)
    forward_flops = 2 * batch_size * (history_macs + k * branch_macs)
    # AdamW parameter/gradient/two-moment storage; activations include gate buffers.
    optimizer_bytes = parameters * 4 * 4
    activation_bytes = batch_size * (21 * 10 * width + k * 10 * 16 * width) * 4
    return dict(kind="analytical_estimate", width=width, batch_size=batch_size, k=k,
                parameters=parameters, forward_dense_flops=forward_flops,
                estimated_training_flops_per_batch=3 * forward_flops,
                adam_parameter_bytes=optimizer_bytes, activation_bytes=activation_bytes,
                assumptions="float32; GRU dense matrix operations; backward estimated at twice forward; "
                            "activation allowance covers gate buffers; excludes framework workspace, "
                            "sample storage, rule evaluation and device transfers; no throughput claim")


__all__ = ["ConditionalTrajectoryHead", "marginal_log_distribution", "analytical_ceiling", "censored_nll"]
