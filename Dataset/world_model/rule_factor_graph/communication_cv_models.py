"""Current numeric communication heads for the clean comparable rerun."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


def _feature_keys(keys, name):
    if not isinstance(keys, tuple) or not keys or len(set(keys)) != len(keys):
        raise ValueError(f"{name} requires explicit unique feature names")
    return keys


class NumericHistoryContext(nn.Module):
    """Shared per-predicate 32-feature context, derived only from the prefix.

    Order: own8, companion8, margin/threshold, lastdiff/threshold, crossing,
    quadratic first-opposite onehot11, quadratic a/threshold, fit RMS/threshold.
    Crossing is capped at20; flat/wrong-trend extrapolation gives20.
    """
    def __init__(self):
        super().__init__()
        times = torch.arange(-3, 1, dtype=torch.float64)
        future = torch.arange(1, 11, dtype=torch.float64)
        design = torch.stack((times.square(), times, torch.ones_like(times)), -1)
        self.register_buffer("fit_design", design)
        self.register_buffer("fit_pinv", torch.linalg.pinv(design))
        self.register_buffer("future_design", torch.stack((future.square(), future, torch.ones_like(future)), -1))

    def forward(self, history, raw_history, thresholds):
        rows = history.transpose(1, 2)
        anchor = raw_history[:, -1]
        diff = anchor - raw_history[:, -2]
        above = anchor > thresholds
        margin = anchor - thresholds
        toward = (above & (diff < 0)) | (~above & (diff > 0))
        divisor = torch.where(toward, diff, torch.ones_like(diff))
        crossing = torch.where(toward, (-margin / divisor).clamp(0, 20), torch.full_like(diff, 20))
        trend = torch.stack((margin / thresholds, diff / thresholds, crossing), -1)
        recent = raw_history[:, -4:].transpose(1, 2)
        coefficients = recent @ self.fit_pinv.T
        fitted = coefficients @ self.fit_design.T
        projected = coefficients @ self.future_design.T
        opposite = (projected > thresholds[:, :, None]) != above[:, :, None]
        first = opposite.long().argmax(-1)
        bins = torch.where(opposite.any(-1), first, torch.full_like(first, 10))
        outcome = torch.nn.functional.one_hot(bins, 11).to(history.dtype)
        curvature = coefficients[..., 0] / thresholds
        rms = (fitted-recent).square().mean(-1).sqrt() / thresholds
        return torch.cat((rows, rows.flip(1), trend, outcome, curvature[..., None], rms[..., None]), -1)


def linear_forecast_bins(raw_history, thresholds):
    anchor = raw_history[:, -1]
    diff = anchor - raw_history[:, -2]
    times = torch.arange(1, 11, device=anchor.device, dtype=anchor.dtype)
    projected = anchor[:, None] + diff[:, None] * times[None, :, None]
    opposite = (projected > thresholds[:, None]) != (anchor > thresholds)[:, None]
    first = opposite.long().argmax(1)
    return torch.where(opposite.any(1), first, torch.full_like(first, 10))


class HistoryOnlyLogitHead(nn.Module):
    """Feature-matched shared MLP with the fixed linear forecast8I prior."""
    def __init__(self, hidden=32):
        super().__init__()
        self.model_name = "history_mlp"
        self.context = NumericHistoryContext()
        self.head = nn.Sequential(nn.Linear(32, hidden), nn.Tanh(), nn.Linear(hidden, 11))
        self.linear_forecast_skip = nn.Linear(11, 11, bias=False)
        with torch.no_grad():
            self.linear_forecast_skip.weight.copy_(8.0 * torch.eye(11))
        self.linear_forecast_skip.weight.requires_grad_(False)

    def forward(self, history, raw_history, thresholds):
        features = self.context(history, raw_history, thresholds)
        bins = linear_forecast_bins(raw_history, thresholds)
        outcome = torch.nn.functional.one_hot(bins, 11).to(history.dtype)
        return torch.log_softmax(self.head(features) + self.linear_forecast_skip(outcome), -1)


class IndependentEventHeads(nn.Module):
    """Numeric mean-state baseline with direct L/P outcome heads.

    HistoryGRU2, full common32x2 context, and per-step numeric2+rule2 inputs.
    The estimated-mean trajectory yields independent L/P eleven-bin logits.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.model_name = "numeric_direct_heads"
        self.history_encoder = nn.GRU(2, hidden, batch_first=True)
        self.context_encoder = nn.Sequential(nn.Linear(hidden + 64 + 4, hidden), nn.Tanh())
        self.latency_head = nn.Linear(hidden, 11)
        self.packet_loss_head = nn.Linear(hidden, 11)

    def forward(self, history, branches, truth, context):
        batch, count, steps = branches.shape[:3]
        _, hidden = self.history_encoder(history)
        prefix = hidden[-1][:, None, None].expand(batch, count, steps, -1)
        extra = context.flatten(1)[:, None, None].expand(batch, count, steps, -1)
        encoded = self.context_encoder(torch.cat((prefix, extra, branches, truth.to(branches.dtype)), -1)).mean(2)
        return torch.stack((self.latency_head(encoded), self.packet_loss_head(encoded)), 1)


class TrajectoryLogitHead(nn.Module):
    """Eleven outcome logits for EACH sampled trajectory and predicate: [B,P,K,11].

    One shared encoder/head reads the anchor-to-future sampled numeric states, one predicate's fixed-rule
    trace and the trajectory's rule-derived outcome bin. Weights are shared across branches and predicates;
    there is no branch pooling, scalar branch gate or predicate expert. Log-softmax over the eleven bins gives
    every trajectory full support. The caller uses the experiment's sampling weights;
    the ordinary K8 variant takes an arithmetic mean of the eight distributions.
    """

    def __init__(self, feature_keys, hidden_dim=32):
        super().__init__()
        self._feature_keys = _feature_keys(feature_keys, "feature_keys")
        self.model_name = "trajectory_logit_head"
        self.history_dim = len(feature_keys)
        self.hidden_dim = hidden_dim
        self.history_encoder = nn.GRU(self.history_dim, hidden_dim, batch_first=True)
        self.branch_encoder = nn.GRU(self.history_dim + 1, hidden_dim, batch_first=True)
        self.context_encoder = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.residual = nn.Sequential(nn.Linear(hidden_dim + 11, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 11))
        self.outcome_skip = nn.Linear(11, 11, bias=False)
        with torch.no_grad():
            self.residual[2].weight.zero_()
            self.residual[2].bias.zero_()
            self.outcome_skip.weight.copy_(4.0 * torch.eye(11))

    @property
    def feature_keys(self):
        return self._feature_keys

    def forward(self, history, branch_features, truth):
        if truth.dtype != torch.bool or truth.ndim != 4 or truth.shape[:3] != branch_features.shape[:3]:
            raise ValueError("rule traces must be Boolean [B,K,11,P] aligned with sampled states")
        batch, branches, steps, width = branch_features.shape
        predicates = truth.shape[-1]
        _, history_hidden = self.history_encoder(history)
        trace = truth.permute(0, 3, 1, 2).to(branch_features.dtype)
        states = branch_features[:, None].expand(batch, predicates, branches, steps, width)
        channels = [states, trace[..., None]]
        sequence = torch.cat(channels, -1)
        _, branch_hidden = self.branch_encoder(sequence.reshape(-1, steps, sequence.shape[-1]))
        prefix = history_hidden[-1][:, None, None, :].expand(batch, predicates, branches, self.hidden_dim)
        context = self.context_encoder(torch.cat((prefix, branch_hidden[-1].reshape(batch, predicates, branches, -1)), -1))
        anchor = truth[:, :1, :1, :]
        opposite = truth[:, :, 1:, :] != anchor
        classes = torch.where(opposite.any(2), opposite.long().argmax(2), torch.full_like(opposite[:, :, 0, :], 10, dtype=torch.long))
        outcome = torch.nn.functional.one_hot(classes.permute(0, 2, 1), 11).to(branch_features.dtype)
        features, logits = [context, outcome], self.outcome_skip(outcome)
        return torch.log_softmax(logits + self.residual(torch.cat(features, -1)), dim=-1)



class HistoryTrendTrajectoryHead(TrajectoryLogitHead):
    """Current Ours: trajectory logits plus shared prefix MLP/fixed linear prior."""
    def __init__(self, feature_keys, hidden_dim=32):
        super().__init__(feature_keys, hidden_dim)
        self.model_name = "history_trend_trajectory_head"
        self.history_trend_head = HistoryOnlyLogitHead(hidden_dim)

    def forward(self, history, branch_features, truth, *, raw_history, thresholds):
        trajectory = super().forward(history, branch_features, truth)
        prefix = self.history_trend_head(history, raw_history, thresholds)
        return torch.log_softmax(trajectory + prefix[:, :, None], -1)


class ObservedHistoryContext(nn.Module):
    """Twenty observed-prefix features; no forecast/category computation."""
    def __init__(self, source_context):
        super().__init__()
        self.register_buffer("fit_design", source_context.fit_design.detach().clone())
        self.register_buffer("fit_pinv", source_context.fit_pinv.detach().clone())

    def forward(self, history, raw_history, thresholds):
        rows = history.transpose(1, 2)
        anchor = raw_history[:, -1]
        margin = (anchor-thresholds)/thresholds
        diff = (anchor-raw_history[:, -2])/thresholds
        recent = raw_history[:, -4:].transpose(1, 2)
        coefficients = recent @ self.fit_pinv.T
        fitted = coefficients @ self.fit_design.T
        curvature = coefficients[..., 0]/thresholds
        rms = (fitted-recent).square().mean(-1).sqrt()/thresholds
        return torch.cat((rows, rows.flip(1), margin[..., None], diff[..., None],
                          curvature[..., None], rms[..., None]), -1)


class OursHistoryOnly(nn.Module):
    """History-only ablation sliced from a freshly seeded current Ours.

    Retains the history GRU, history columns of fusion/residual layers, and
    observed columns of the shared prefix MLP. Removed modules never enter
    the optimizer or forward; common weights/biases copy the seeded source.
    This ablates the complete future-prediction/rule package, including priors.
    """
    def __init__(self, source_ours):
        super().__init__()
        self.model_name = "OursHistoryOnly"
        hidden = source_ours.hidden_dim
        self.history_encoder = deepcopy(source_ours.history_encoder)
        self.context_encoder = deepcopy(source_ours.context_encoder)
        self.context_encoder[0].weight = nn.Parameter(source_ours.context_encoder[0].weight[:, :hidden].detach().clone())
        self.context_encoder[0].in_features = hidden
        self.residual = deepcopy(source_ours.residual)
        self.residual[0].weight = nn.Parameter(source_ours.residual[0].weight[:, :hidden].detach().clone())
        self.residual[0].in_features = hidden
        self.context = ObservedHistoryContext(source_ours.history_trend_head.context)
        self.prefix_head = deepcopy(source_ours.history_trend_head.head)
        columns = list(range(18)) + [30, 31]
        self.prefix_head[0].weight = nn.Parameter(source_ours.history_trend_head.head[0].weight[:, columns].detach().clone())
        self.prefix_head[0].in_features = 20

    def forward(self, history, raw_history, thresholds):
        _, hidden = self.history_encoder(history)
        context = self.context_encoder(hidden[-1])
        features = self.context(history, raw_history, thresholds)
        logits = self.residual(context)[:, None] + self.prefix_head(features)
        return torch.log_softmax(logits, -1)


class ControlledSamplingTrajectoryHead(nn.Module):
    """Identically initialized sampling-on/off arms with full trajectory widths.

    Both use only the observed20 prefix features, with no prefix forecast prior.
    The enabled branch retains numeric/rule encoding and the trainable4I outcome
    skip. Disabled forward never reads branch/rule inputs; fixed zeros replace
    branch hidden/outcome/skip contribution, while inactive modules stay present.
    """
    def __init__(self, source_ours, *, sampling_enabled):
        super().__init__()
        self.sampling_enabled = sampling_enabled
        self.model_name = "OursSamplingHybrid" if sampling_enabled else "OursNoSamplingHybrid"
        self.hidden_dim = source_ours.hidden_dim
        self.history_dim = source_ours.history_dim
        for name in ("history_encoder", "branch_encoder", "context_encoder", "residual", "outcome_skip"):
            setattr(self, name, deepcopy(getattr(source_ours, name)))
        self.context = ObservedHistoryContext(source_ours.history_trend_head.context)
        self.prefix_head = deepcopy(source_ours.history_trend_head.head)
        columns = list(range(18)) + [30, 31]
        self.prefix_head[0].weight = nn.Parameter(source_ours.history_trend_head.head[0].weight[:, columns].detach().clone())
        self.prefix_head[0].in_features = 20

    def forward(self, history, branch_features=None, truth=None, *, raw_history, thresholds):
        if self.sampling_enabled:
            trajectory = TrajectoryLogitHead.forward(self, history, branch_features, truth)
        else:
            _, hidden = self.history_encoder(history)
            prefix = hidden[-1][:, None, None].expand(history.shape[0], 2, 8, self.hidden_dim)
            branch_hidden = torch.zeros_like(prefix)
            context = self.context_encoder(torch.cat((prefix, branch_hidden), -1))
            outcome = context.new_zeros((*context.shape[:-1], 11))
            trajectory = torch.log_softmax(self.residual(torch.cat((context, outcome), -1)), -1)
        features = self.context(history, raw_history, thresholds)
        prefix_logs = torch.log_softmax(self.prefix_head(features), -1)
        return torch.log_softmax(trajectory + prefix_logs[:, :, None], -1)

    def inactive_capacity(self):
        modules, columns = {}, {}
        if not self.sampling_enabled:
            modules = {name: sum(p.numel() for p in getattr(self, name).parameters())
                       for name in ("branch_encoder", "outcome_skip")}
            columns = {"context_encoder.0.weight": list(range(self.hidden_dim, 2*self.hidden_dim)),
                       "residual.0.weight": list(range(self.hidden_dim, self.hidden_dim+11))}
        parameters = dict(self.named_parameters())
        column_counts = {name: parameters[name].shape[0]*len(indices) for name, indices in columns.items()}
        inactive = sum(modules.values()) + sum(column_counts.values())
        total = sum(p.numel() for p in self.parameters())
        return dict(sampling_enabled=self.sampling_enabled, instantiated_parameters=total,
                    likelihood_inactive_parameters=inactive, likelihood_active_parameters=total-inactive,
                    inactive_modules=modules, inactive_input_columns=columns,
                    inactive_column_parameters=column_counts,
                    inactive_column_updates="zero data gradient; AdamW weight decay can still update matrix columns")


def parameter_manifest(model):
    manifest = {"model": model.model_name,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "parameter_shapes": {name: list(p.shape) for name, p in model.named_parameters()},
            "frozen_parameters": [name for name, p in model.named_parameters() if not p.requires_grad]}
    if isinstance(model, ControlledSamplingTrajectoryHead):
        manifest.update(model.inactive_capacity())
    return manifest
