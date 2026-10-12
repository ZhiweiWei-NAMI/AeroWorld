"""Official TabM, adapted to two marginal eleven-bin censored event PMFs.

Input[B,108] = common numeric prefix32x2 + estimatedQmean states11x2
+ governed rule trace11x2. Native members are parameter-efficient ensembles,
not Q draws. Train member-wise censored NLL, infer mean member probabilities.
Official implementation: https://github.com/yandex-research/tabm
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from importlib import metadata
import torch
from torch import nn

TABM_VERSION = "0.0.3"
NUM_EMBEDDINGS_VERSION = "0.0.12"
INPUT_FEATURES = 108


@dataclass(frozen=True)
class TabMConfig:
    """Native no-embedding defaults fixed before any outer evaluation."""

    n_blocks: int = 3
    d_block: int = 512
    ensemble_members: int = 32
    dropout: float = 0.1
    learning_rate: float = 0.002
    weight_decay: float = 0.0003

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.n_blocks, self.d_block, self.ensemble_members)):
            raise ValueError("TabM blocks, width and member count must be positive integers")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("TabM dropout must be finite in [0,1)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("TabM learning rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("TabM weight decay must be nonnegative and finite")


class NumericTabM(nn.Module):
    """input_features selects the numeric input width; 108 stays the declared default."""

    def __init__(self, config=TabMConfig(), input_features=INPUT_FEATURES):
        super().__init__()
        for package, required in (("tabm", TABM_VERSION), ("rtdl-num-embeddings", NUM_EMBEDDINGS_VERSION)):
            installed = metadata.version(package)
            if installed != required:
                raise RuntimeError(f"{package} requires {required}, found {installed}")
        if type(input_features) is not int or input_features < 1:
            raise ValueError("TabM numeric input width must be a positive integer")
        from tabm import TabM
        self.config = config
        self.model_name = "TabM"
        self.input_features = input_features
        self.network = TabM.make(n_num_features=input_features, cat_cardinalities=None,
            d_out=22, num_embeddings=None, n_blocks=config.n_blocks, d_block=config.d_block,
            dropout=config.dropout, activation="ReLU", k=config.ensemble_members,
            arch_type="tabm", start_scaling_init="random-signs").to(dtype=torch.float32)

    def forward(self, features):
        logits = self.network(features.to(torch.float32))
        return torch.log_softmax(logits.to(torch.float64).reshape(len(features), self.config.ensemble_members, 2, 11), -1)


def make_tabm_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=model.config.learning_rate,
                            weight_decay=model.config.weight_decay)


def tabm_member_censored_nll(member_logs, compatible):
    observed = torch.where(compatible[:, None], member_logs, -torch.inf)
    return -torch.logsumexp(observed, -1).mean()


def tabm_log_pmf(model, features):
    members = model(features)
    return torch.logsumexp(members, 1) - math.log(members.shape[1])


def tabm_parameter_manifest(model):
    return dict(model="official TabM numeric marginal adaptation", package=f"tabm=={TABM_VERSION}",
                config=asdict(model.config), input_shape=["B", model.input_features],
                output_shape=["B", model.config.ensemble_members, 2, 11],
                training_objective="mean member-wise marginal censored NLL",
                inference="mean member probabilities", q_branch_axis=False,
                network_dtype="float32", probability_loss_dtype="float64",
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
