"""Schema-derived addresses, typed scalar declarations and TRAIN scale records."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
from Dataset.world_model.contracts.schema import FIELD_REGISTRY

@dataclass(frozen=True, order=True)
class QueryAddress:
    """Exact identity of one requested future entity-field cell."""

    episode_id: str
    entity_id: str
    field_family: str
    target_tick: int
    component: int | None = None

@dataclass(frozen=True)
class FieldTarget:
    """One scalar target component derived exactly from registry metadata."""

    family: str
    component: int | None
    logical_dtype: str
    unit: str | None
    frame: str | None
    axis_role: str | None
    vector_length: int
    circular: bool
    lower: float | None
    upper: float | None
    enum_values: tuple[Any, ...]
    allowed_non_present: tuple[str, ...]

    @property
    def class_count(self) -> int:
        return len(self.enum_values)

@dataclass(frozen=True)
class InputScale:
    """The one TRAIN-window normalization for a real-valued scalar component.

    Forward standardization is used for input encoding and supervision.  Typed
    laws are built directly in physical units in ``distributions.law()`` from
    the physical anchor and ``forecast_std``, so no inverse helper is needed.
    Angles stay circular, categoricals unnormalized and missing states stay
    status cells, so they never pass through this class.
    """

    family: str
    component: int | None
    mean: float
    std: float

    def transform(self, value: float) -> float:
        return (float(value) - self.mean) / self.std


@dataclass(frozen=True)
class FeatureScale:
    """The same normalization path for a real-valued feature vector/plane."""

    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std

    def inverse(self, standardized: torch.Tensor) -> torch.Tensor:
        """Recover the physical feature plane from its standardized value."""
        return standardized * self.std + self.mean


def fit_input_scale(family: str, component: int | None, values) -> InputScale:
    """Fit one scalar scaler on the caller-declared training-window values."""
    values = [float(value) for value in values]
    if not values:
        raise ValueError(f"no training-window values to fit {family}:{component}")
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return InputScale(family, component, mean, std if std else 1.)


def fit_feature_scale(value: torch.Tensor) -> FeatureScale:
    """Fit one tensor scaler over the caller-declared training-window values."""
    x = value.detach().float()
    std = x.std(0, unbiased=False)
    return FeatureScale(x.mean(0), torch.where(std > 0, std, torch.ones_like(std)))

def modelable_field_targets(
    registry: Mapping[str, Mapping[str, Any]] = FIELD_REGISTRY,
) -> tuple[tuple[FieldTarget, ...], tuple[str, ...]]:
    """Compile frame-state targets from exact declarations.

    Real scalars/vectors, booleans, and strings with a finite declared enum
    are modelable.  Open string vocabularies are reported and excluded before
    any future value is inspected.
    """
    targets: list[FieldTarget] = []
    excluded: list[str] = []
    for family, declaration in sorted(registry.items()):
        if family.startswith("archive.") or "frame" not in declaration["records"]:
            continue
        dtype = declaration["dtype"]
        enum = declaration.get("enum")
        if dtype == "real":
            logical_dtype = "real"
            enum_values: tuple[Any, ...] = ()
        elif dtype == "bool":
            logical_dtype = "bool"
            enum_values = (False, True)
        elif dtype == "string" and enum:
            logical_dtype = "enum"
            enum_values = tuple(enum)
        else:
            if dtype == "string":
                excluded.append(family)
            continue
        vector_length = int(declaration.get("vector_length") or 1)
        components: tuple[int | None, ...] = (
            tuple(range(vector_length)) if vector_length > 1 else (None,)
        )
        bounded = declaration.get("bounded") or (None, None)
        for component in components:
            targets.append(FieldTarget(
                family=family,
                component=component,
                logical_dtype=logical_dtype,
                unit=declaration.get("unit"),
                frame=declaration.get("frame"),
                axis_role=declaration.get("axis_role"),
                vector_length=vector_length,
                circular=bool(declaration.get("circular")),
                lower=None if bounded[0] is None else float(bounded[0]),
                upper=None if bounded[1] is None else float(bounded[1]),
                enum_values=enum_values,
                allowed_non_present=tuple(declaration.get("non_present_kinds") or ()),
            ))
    return tuple(targets), tuple(excluded)


import math
import torch
from torch import nn

# Per-query numeric features built by ``causal_input.query_tensors``: 12 address
# and history descriptors plus 3 exact history anchor-atom features.
QUERY_FEATURES = 15


class AddressQueryAttention(nn.Module):
    """Shared address reads over registered slots; no entity-specific parameters."""
    def __init__(self, hidden=2048, width=256):
        super().__init__()
        from Dataset.world_model.model.causal_input import COMM
        self.field_ids = {name: i for i, name in enumerate(sorted(FIELD_REGISTRY.keys() | COMM.keys()))}
        self.field = nn.Embedding(len(self.field_ids), width)
        self.component = nn.Embedding(16, width)
        self.query_projection = nn.Sequential(nn.Linear(QUERY_FEATURES, width), nn.GELU(), nn.Linear(width, width))
        self.identity_projection = nn.Linear(32, width)
        self.hidden_projection = nn.Linear(hidden, width)
        self.global_projection = nn.Linear(hidden, width)
        self.entity_attention = nn.MultiheadAttention(width, 8, dropout=0, batch_first=True)
        self.field_attention = nn.MultiheadAttention(width, 8, dropout=0, batch_first=True)
        self.entity_output = nn.Linear(width, hidden)
        self.field_output = nn.Linear(width, hidden)

    def forward(self, query, hidden, token_entities, token_fields, global_context):
        if hidden.ndim != 2 or len(token_entities) != len(hidden) or len(token_fields) != len(hidden):
            raise ValueError("address registration and hidden slots differ")
        device, dtype = hidden.device, self.field.weight.dtype
        features = query["features"].to(device=device, dtype=dtype)
        q = self.query_projection(features)
        fields = torch.tensor([self.field_ids[a.field_family] for a in query["addresses"]], device=device)
        components = torch.tensor([0 if a.component is None else a.component+1 for a in query["addresses"]], device=device)
        ordinals = (query["entity_ordinals"]+1).to(device=device, dtype=dtype)
        phase = ordinals[:, None]*torch.exp(-torch.arange(16, device=device, dtype=dtype)*math.log(10000)/16)
        entity_query = q+self.identity_projection(torch.cat((phase.sin(), phase.cos()), dim=-1))
        field_query = entity_query+self.field(fields)+self.component(components)
        h = self.hidden_projection(hidden.to(dtype=dtype))
        global_context = global_context.to(device=device, dtype=dtype).reshape(-1, hidden.shape[-1]).mean(0)
        global_ = self.global_projection(global_context)
        # Group identical address supports to avoid padding to total history length.
        entity_slots = {}
        pair_slots = {}
        for slot, (entities, families) in enumerate(zip(token_entities, token_fields)):
            for entity in entities:
                if entity not in entity_slots:
                    entity_slots[entity] = []
                entity_slots[entity].append(slot)
                for family in families:
                    key = entity, family
                    if key not in pair_slots:
                        pair_slots[key] = []
                    pair_slots[key].append(slot)
        entity_groups, field_groups = {}, {}
        for row, address in enumerate(query["addresses"]):
            if address.entity_id not in entity_slots:
                raise ValueError(f"unregistered causal query entity {address.entity_id}")
            if address.entity_id not in entity_groups:
                entity_groups[address.entity_id] = []
            entity_groups[address.entity_id].append(row)
            key = address.entity_id, address.field_family
            if key not in field_groups:
                field_groups[key] = []
            field_groups[key].append(row)
        entity_result, field_result = torch.zeros_like(q), torch.zeros_like(q)
        for entity, rows in entity_groups.items():
            indices = torch.tensor(rows, device=device)
            kv = h[entity_slots[entity]][None]
            entity_result[indices] = self.entity_attention(entity_query[indices][None], kv, kv, need_weights=False)[0][0]
        for key, rows in field_groups.items():
            indices = torch.tensor(rows, device=device)
            # An absent historical field remains a registered address. Its support is
            # the causal entity slot plus global memory, never an invented field cell.
            support = pair_slots[key] if key in pair_slots else entity_slots[key[0]]
            kv = torch.cat((h[support], global_[None]), dim=0)[None]
            field_result[indices] = self.field_attention(field_query[indices][None], kv, kv, need_weights=False)[0][0]
        return self.entity_output(entity_query+entity_result+global_), self.field_output(field_query+field_result+global_)
