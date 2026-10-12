"""Cumulative probability representation for the current numeric comparison.

The cache-only runner derives these logs from marginal eleven-bin PMFs.
Probability accumulation and current losses are implemented in that runner.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CumulativeBranchLogs:
    """Normalized log-CDF and log-survival tensors [B,P,10]."""
    log_cdf: torch.Tensor
    log_survival: torch.Tensor
