"""Dynamic entity-field queries with one shared typed prediction head.

The query inventory is compiled only from the causal input branch and the
field registry.  Future records attach supervision after addresses exist;
they never enter ``DynamicQueryBatch`` or choose its rows.  Entity ids are
therefore join keys, not learned embeddings, and every query row passes
through the same encoder, trunk, and output projection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from p01v4.contracts.schema import COORDINATE_FRAMES, FIELD_REGISTRY


STATUS_KINDS = (
    "present",
    "missing",
    "inapplicable",
    "absent",
    "source_unknown",
    "unrecorded",
)
NON_PRESENT_KINDS = STATUS_KINDS[1:5]
LOGICAL_DTYPES = ("real", "enum", "bool")


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
    """TRAIN-prefix normalization for one real-valued target component."""

    family: str
    component: int | None
    mean: float
    std: float
    empirical_std: float
    count: int
    max_tick: int

    def transform(self, value: float) -> float:
        return (float(value) - self.mean) / self.std


@dataclass(frozen=True)
class QueryInventory:
    """Causal address inventory and the registry decisions behind it."""

    episode_id: str
    entities: tuple[str, ...]
    fields: tuple[FieldTarget, ...]
    target_ticks: tuple[int, ...]
    addresses: tuple[QueryAddress, ...]
    excluded_open_fields: tuple[str, ...]


@dataclass(frozen=True)
class FeatureLayout:
    """Registry-wide shared feature vocabulary; it contains no entity ids."""

    units: tuple[str | None, ...]
    frames: tuple[str | None, ...]
    axis_roles: tuple[str | None, ...]
    components: tuple[int | None, ...]
    names: tuple[str, ...]

    @classmethod
    def from_fields(cls, fields: Sequence[FieldTarget]) -> "FeatureLayout":
        units = tuple(sorted({f.unit for f in fields}, key=lambda x: "" if x is None else x))
        frames = tuple(sorted({f.frame for f in fields}, key=lambda x: "" if x is None else x))
        axis_roles = tuple(sorted({f.axis_role for f in fields}, key=lambda x: "" if x is None else x))
        components = tuple(sorted({f.component for f in fields}, key=lambda x: -1 if x is None else x))
        names = (
            "history.latest", "history.previous", "history.delta", "history.mean",
            "history.log_count", "history.age", "history.present_fraction",
            *(f"history.status.{kind}" for kind in STATUS_KINDS),
            "horizon.linear", "horizon.log",
            *(f"dtype.{kind}" for kind in LOGICAL_DTYPES),
            *(f"unit.{unit}" for unit in units),
            *(f"frame.{frame}" for frame in frames),
            *(f"axis_role.{role}" for role in axis_roles),
            *(f"component.{component}" for component in components),
            "schema.circular", "schema.has_lower", "schema.has_upper",
            "schema.vector_length", "schema.enum_size",
            *(f"schema.allows.{kind}" for kind in NON_PRESENT_KINDS),
        )
        return cls(units, frames, axis_roles, components, tuple(names))


@dataclass
class DynamicQueryBatch:
    """Forward inputs for a variable Q axis.  No target data lives here."""

    addresses: tuple[QueryAddress, ...]
    features: Tensor
    scale_mean: Tensor
    scale_std: Tensor
    lower: Tensor
    upper: Tensor
    has_lower: Tensor
    has_upper: Tensor
    circular: Tensor
    logical_dtype: Tensor
    class_count: Tensor
    feature_names: tuple[str, ...]

    def index_select(self, indices: Tensor) -> "DynamicQueryBatch":
        rows = indices.detach().cpu().tolist()
        return DynamicQueryBatch(
            addresses=tuple(self.addresses[i] for i in rows),
            features=self.features.index_select(0, indices),
            scale_mean=self.scale_mean.index_select(0, indices),
            scale_std=self.scale_std.index_select(0, indices),
            lower=self.lower.index_select(0, indices),
            upper=self.upper.index_select(0, indices),
            has_lower=self.has_lower.index_select(0, indices),
            has_upper=self.has_upper.index_select(0, indices),
            circular=self.circular.index_select(0, indices),
            logical_dtype=self.logical_dtype.index_select(0, indices),
            class_count=self.class_count.index_select(0, indices),
            feature_names=self.feature_names,
        )


@dataclass
class TypedLabels:
    """Future supervision kept outside the model's forward inputs."""

    status: Tensor
    numeric: Tensor
    categorical: Tensor
    numeric_mask: Tensor
    categorical_mask: Tensor

    def status_masks(self) -> dict[str, Tensor]:
        return {kind: self.status == i for i, kind in enumerate(STATUS_KINDS)}


@dataclass
class TypedQueryOutput:
    numeric: Tensor
    class_logits: Tensor
    status_logits: Tensor

    def index_select(self, indices: Tensor) -> "TypedQueryOutput":
        return TypedQueryOutput(
            self.numeric.index_select(0, indices),
            self.class_logits.index_select(0, indices),
            self.status_logits.index_select(0, indices),
        )


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


def compile_query_inventory(
    typed_input: Mapping[str, Any], *, episode_id: str, target_ticks: Sequence[int],
    registry: Mapping[str, Mapping[str, Any]] = FIELD_REGISTRY,
) -> QueryInventory:
    """Compile Q without consulting target records or future membership."""
    entities = {
        entity["entity_id"] for entity in typed_input["entities"].values()
        if isinstance(entity.get("entity_id"), str)
    }
    entities.update(
        state["entity_id"] for state in typed_input["states"]
        if isinstance(state.get("entity_id"), str)
    )
    fields, excluded = modelable_field_targets(registry)
    ticks = tuple(sorted({int(t) for t in target_ticks}))
    addresses = tuple(
        QueryAddress(episode_id, entity_id, field.family, tick, field.component)
        for entity_id in sorted(entities)
        for tick in ticks
        for field in fields
    )
    return QueryInventory(
        episode_id=episode_id,
        entities=tuple(sorted(entities)),
        fields=fields,
        target_ticks=ticks,
        addresses=addresses,
        excluded_open_fields=excluded,
    )


def _cell_status_value(cell: Any, component: int | None) -> tuple[str, Any]:
    if isinstance(cell, Mapping) and set(cell) == {"kind"}:
        return str(cell["kind"]), None
    value = cell
    if isinstance(cell, Mapping) and "value" in cell:
        value = cell["value"]
    if component is not None:
        value = value[component]
    return "present", value


def _state_history(typed_input: Mapping[str, Any], fields: Sequence[FieldTarget]) -> dict[tuple[str, str, int | None], list[tuple[int, str, Any]]]:
    by_family: dict[str, list[FieldTarget]] = {}
    for field in fields:
        by_family.setdefault(field.family, []).append(field)
    histories: dict[tuple[str, str, int | None], list[tuple[int, str, Any]]] = {}
    for state in typed_input["states"]:
        entity_id = state.get("entity_id")
        tick = state.get("tick")
        if not isinstance(entity_id, str) or not isinstance(tick, int):
            continue
        state_fields = state.get("fields") or {}
        for family, components in by_family.items():
            if family not in state_fields:
                continue
            for field in components:
                status, value = _cell_status_value(state_fields[family], field.component)
                histories.setdefault((entity_id, family, field.component), []).append(
                    (tick, status, value))
    for history in histories.values():
        history.sort(key=lambda row: row[0])
    return histories


def fit_input_scales(
    typed_input: Mapping[str, Any], fields: Sequence[FieldTarget],
    *, overrides: Mapping[tuple[str, int | None], InputScale] | None = None,
) -> dict[tuple[str, int | None], InputScale]:
    """Fit real component statistics only on the causal input branch."""
    histories = _state_history(typed_input, fields)
    scales: dict[tuple[str, int | None], InputScale] = {}
    for field in fields:
        if field.logical_dtype != "real":
            continue
        key = (field.family, field.component)
        values: list[float] = []
        ticks: list[int] = []
        for (_entity, family, component), history in histories.items():
            if (family, component) != key:
                continue
            for tick, status, value in history:
                if status == "present":
                    values.append(float(value))
                    ticks.append(tick)
        if not values:
            raise ValueError(f"no causal input values for modelable field component {key}")
        mean = sum(values) / len(values)
        empirical_std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        # A constant declared field has a well-defined zero-centred value; a
        # unit scale keeps it representable without inventing variation.
        std = empirical_std if empirical_std > 0.0 else 1.0
        scales[key] = InputScale(
            family=field.family,
            component=field.component,
            mean=mean,
            std=std,
            empirical_std=empirical_std,
            count=len(values),
            max_tick=max(ticks),
        )
    if overrides:
        scales.update(overrides)
    return scales


def _one_hot(value: Any, vocabulary: Sequence[Any]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocabulary]


def _history_features(
    history: Sequence[tuple[int, str, Any]], field: FieldTarget,
    scale: InputScale | None, cutoff: int,
) -> tuple[list[float], str]:
    latest_status = history[-1][1] if history else "unrecorded"
    present = [(tick, value) for tick, status, value in history if status == "present"]
    encoded: list[tuple[int, float]] = []
    for tick, value in present:
        if field.logical_dtype == "real":
            assert scale is not None
            encoded.append((tick, scale.transform(float(value))))
        else:
            encoded.append((tick, float(field.enum_values.index(value))))
    if encoded:
        latest_tick, latest = encoded[-1]
        previous = encoded[-2][1] if len(encoded) > 1 else latest
        mean = sum(value for _, value in encoded) / len(encoded)
        age = (cutoff - latest_tick) / max(1, cutoff)
    else:
        latest = previous = mean = age = 0.0
    total = len(history)
    features = [
        latest,
        previous,
        latest - previous,
        mean,
        math.log1p(len(encoded)) / math.log1p(max(1, cutoff + 1)),
        age,
        len(encoded) / total if total else 0.0,
        *_one_hot(latest_status, STATUS_KINDS),
    ]
    return features, latest_status


def compile_query_batch(
    inventory: QueryInventory, typed_input: Mapping[str, Any],
    scales: Mapping[tuple[str, int | None], InputScale], *, cutoff: int,
) -> DynamicQueryBatch:
    """Encode every address with shared causal history and schema features."""
    fields = {(field.family, field.component): field for field in inventory.fields}
    layout = FeatureLayout.from_fields(inventory.fields)
    histories = _state_history(typed_input, inventory.fields)
    max_vector = max(field.vector_length for field in inventory.fields)
    max_enum = max((field.class_count for field in inventory.fields), default=1)
    max_horizon = max(tick - cutoff for tick in inventory.target_ticks)

    rows: list[list[float]] = []
    means: list[float] = []
    stds: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    has_lowers: list[bool] = []
    has_uppers: list[bool] = []
    circular: list[bool] = []
    dtype_ids: list[int] = []
    class_counts: list[int] = []
    for address in inventory.addresses:
        field = fields[(address.field_family, address.component)]
        scale = scales.get((field.family, field.component))
        history = histories.get((address.entity_id, field.family, field.component), ())
        history_features, _ = _history_features(history, field, scale, cutoff)
        horizon = address.target_tick - cutoff
        row = [
            *history_features,
            horizon / max_horizon,
            math.log1p(horizon) / math.log1p(max_horizon),
            *_one_hot(field.logical_dtype, LOGICAL_DTYPES),
            *_one_hot(field.unit, layout.units),
            *_one_hot(field.frame, layout.frames),
            *_one_hot(field.axis_role, layout.axis_roles),
            *_one_hot(field.component, layout.components),
            float(field.circular),
            float(field.lower is not None),
            float(field.upper is not None),
            field.vector_length / max_vector,
            field.class_count / max_enum if field.class_count else 0.0,
            *(float(kind in field.allowed_non_present) for kind in NON_PRESENT_KINDS),
        ]
        rows.append(row)
        means.append(scale.mean if scale is not None else 0.0)
        stds.append(scale.std if scale is not None else 1.0)
        lowers.append(field.lower if field.lower is not None else 0.0)
        uppers.append(field.upper if field.upper is not None else 0.0)
        has_lowers.append(field.lower is not None)
        has_uppers.append(field.upper is not None)
        circular.append(field.circular)
        dtype_ids.append(LOGICAL_DTYPES.index(field.logical_dtype))
        class_counts.append(field.class_count)
    features = torch.tensor(rows, dtype=torch.float32)
    if features.shape[1] != len(layout.names):
        raise RuntimeError(
            f"feature layout mismatch: tensor has {features.shape[1]}, names has {len(layout.names)}")
    return DynamicQueryBatch(
        addresses=inventory.addresses,
        features=features,
        scale_mean=torch.tensor(means, dtype=torch.float32),
        scale_std=torch.tensor(stds, dtype=torch.float32),
        lower=torch.tensor(lowers, dtype=torch.float32),
        upper=torch.tensor(uppers, dtype=torch.float32),
        has_lower=torch.tensor(has_lowers, dtype=torch.bool),
        has_upper=torch.tensor(has_uppers, dtype=torch.bool),
        circular=torch.tensor(circular, dtype=torch.bool),
        logical_dtype=torch.tensor(dtype_ids, dtype=torch.long),
        class_count=torch.tensor(class_counts, dtype=torch.long),
        feature_names=layout.names,
    )


def attach_typed_labels(
    batch: DynamicQueryBatch, typed_targets: Mapping[str, Any],
    fields: Sequence[FieldTarget], scales: Mapping[tuple[str, int | None], InputScale],
) -> TypedLabels:
    """Attach future values after Q is fixed; labels never enter forward."""
    field_map = {(field.family, field.component): field for field in fields}
    states: dict[tuple[str, int], Mapping[str, Any]] = {}
    for state in typed_targets["states"]:
        key = (state.get("entity_id"), state.get("tick"))
        if key in states:
            raise ValueError(f"duplicate target state for {key}")
        states[key] = state

    statuses: list[int] = []
    numeric: list[float] = []
    categorical: list[int] = []
    numeric_mask: list[bool] = []
    categorical_mask: list[bool] = []
    for address in batch.addresses:
        field = field_map[(address.field_family, address.component)]
        state = states.get((address.entity_id, address.target_tick))
        if state is None or field.family not in (state.get("fields") or {}):
            status, value = "unrecorded", None
        else:
            status, value = _cell_status_value(state["fields"][field.family], field.component)
        statuses.append(STATUS_KINDS.index(status))
        is_numeric = status == "present" and field.logical_dtype == "real"
        is_categorical = status == "present" and field.logical_dtype in ("enum", "bool")
        if is_numeric:
            numeric.append(scales[(field.family, field.component)].transform(float(value)))
        else:
            numeric.append(0.0)
        if is_categorical:
            categorical.append(field.enum_values.index(value))
        else:
            categorical.append(-1)
        numeric_mask.append(is_numeric)
        categorical_mask.append(is_categorical)
    return TypedLabels(
        status=torch.tensor(statuses, dtype=torch.long),
        numeric=torch.tensor(numeric, dtype=torch.float32),
        categorical=torch.tensor(categorical, dtype=torch.long),
        numeric_mask=torch.tensor(numeric_mask, dtype=torch.bool),
        categorical_mask=torch.tensor(categorical_mask, dtype=torch.bool),
    )


class SharedTypedQueryHead(nn.Module):
    """One row-wise parameter set for every entity, field, and query."""

    def __init__(self, feature_dim: int, *, hidden_dim: int = 64, max_classes: int = 3):
        super().__init__()
        self.max_classes = max_classes
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        # A single projection is interpreted by declared type metadata.  It
        # is shared rather than selected by entity id, field name, or query.
        self.output = nn.Linear(hidden_dim, 1 + max_classes + len(STATUS_KINDS))

    def forward(self, batch: DynamicQueryBatch) -> TypedQueryOutput:
        projected = self.output(self.trunk(batch.features))
        numeric_z = projected[:, 0]
        raw = numeric_z * batch.scale_std + batch.scale_mean
        lower_only = batch.has_lower & ~batch.has_upper
        upper_only = batch.has_upper & ~batch.has_lower
        both = batch.has_lower & batch.has_upper
        raw = torch.where(
            lower_only,
            batch.lower + F.softplus(raw - batch.lower),
            raw,
        )
        raw = torch.where(
            upper_only,
            batch.upper - F.softplus(batch.upper - raw),
            raw,
        )
        bounded = batch.lower + (batch.upper - batch.lower) * torch.sigmoid(raw)
        raw = torch.where(both, bounded, raw)
        radians = torch.deg2rad(raw)
        wrapped = torch.rad2deg(torch.atan2(torch.sin(radians), torch.cos(radians)))
        raw = torch.where(batch.circular, wrapped, raw)
        numeric = (raw - batch.scale_mean) / batch.scale_std
        start = 1
        class_logits = projected[:, start:start + self.max_classes]
        status_logits = projected[:, start + self.max_classes:]
        return TypedQueryOutput(numeric, class_logits, status_logits)


def typed_supervision_loss(
    output: TypedQueryOutput, labels: TypedLabels, batch: DynamicQueryBatch,
) -> dict[str, Tensor]:
    """Typed value and explicit availability losses over the shared output."""
    status_loss = F.cross_entropy(output.status_logits, labels.status)

    numeric_rows = labels.numeric_mask
    difference = output.numeric[numeric_rows] - labels.numeric[numeric_rows]
    if bool(numeric_rows.any()):
        circular = batch.circular[numeric_rows]
        raw_difference = difference * batch.scale_std[numeric_rows]
        radians = torch.deg2rad(raw_difference)
        wrapped = torch.rad2deg(torch.atan2(torch.sin(radians), torch.cos(radians)))
        difference = torch.where(circular, wrapped / batch.scale_std[numeric_rows], difference)
        numeric_loss = F.smooth_l1_loss(difference, torch.zeros_like(difference))
    else:
        numeric_loss = output.numeric.sum() * 0.0

    categorical_rows = labels.categorical_mask
    if bool(categorical_rows.any()):
        logits = output.class_logits[categorical_rows]
        class_count = batch.class_count[categorical_rows]
        class_ids = torch.arange(logits.shape[-1], device=logits.device)
        logits = logits.masked_fill(class_ids[None, :] >= class_count[:, None], -torch.inf)
        categorical_loss = F.cross_entropy(
            logits, labels.categorical[categorical_rows])
    else:
        categorical_loss = output.class_logits.sum() * 0.0
    total = status_loss + numeric_loss + categorical_loss
    return {
        "total": total,
        "status": status_loss,
        "numeric": numeric_loss,
        "categorical": categorical_loss,
    }


def declared_component_axis(field: FieldTarget) -> str | None:
    """Return the exact coordinate axis declared for a vector component."""
    if field.component is None or field.frame is None:
        return None
    return str(COORDINATE_FRAMES[field.frame]["axes"][field.component])
