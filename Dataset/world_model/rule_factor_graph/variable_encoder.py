"""One schema-based encoder for every finite candidate-valued graph variable.

Identity strings remain in grounding. A variable is encoded once; roles belong
to factor incidences and are supplied to the shared scorer. Forecast values are
conditional inputs, separate from candidate assignments of latent true state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .generic_grounding import CandidateVariable

DATA_TYPES = ("numeric", "categorical", "boolean", "identifier", "four_value", "vector", "integer")
TOKEN_DIM = 128


@dataclass(frozen=True)
class SchemaVocabulary:
    """Exact schema symbols fixed when the model is constructed.

    Categories are state codes; attributes are declared canonical fields, never
    predicate or instance identifiers. There is no unknown-symbol
    bucket: a schema or categorical value absent from the declaration is an error.
    """

    units: tuple[str, ...]
    classes: tuple[str, ...]
    categories: tuple[str, ...]
    attributes: tuple[str, ...]

    def __post_init__(self):
        for name in ("units", "classes", "categories", "attributes"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != len(set(values)):
                raise ValueError(f"{name} must be a tuple of distinct declared symbols")
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain nonempty strings")
        if not self.units or not self.classes or not self.attributes:
            raise ValueError("the schema vocabulary needs units, owner classes and declared attributes")

    def index(self, kind: str, value: str) -> int:
        if kind not in {"units", "classes", "categories", "attributes"}:
            raise ValueError(f"unsupported schema vocabulary: {kind}")
        try:
            return getattr(self, kind).index(value)
        except ValueError as exc:
            raise ValueError(f"undeclared {kind} symbol: {value!r}") from exc

    @classmethod
    def from_variables(cls, variables: Sequence["CandidateVariable"], *,
                       attribute_symbols: Mapping) -> "SchemaVocabulary":
        """Build a declaration from explicit domains, schema and available context.

        Call once on the entire model's schema/domain catalog, before constructing
        the encoder. Changing a predicate/entity identity cannot change this map.
        """
        units, classes, categories, attributes = set(), set(), set(), set()
        if set(attribute_symbols) != {variable.key for variable in variables}:
            raise ValueError("attribute declarations must match the exact variable catalog")
        for variable in variables:
            symbol = attribute_symbols[variable.key]
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("each variable needs a nonempty declared attribute symbol")
            attributes.add(symbol)
            units.add(variable.schema.unit)
            classes.update(owner.ontology_class for owner in variable.key.owner)
            if variable.schema.data_type in {"categorical", "four_value"}:
                values = (*variable.domain, *(value for _, value in variable.history),
                          *variable.forecast_context)
                if variable.observed_value is not None:
                    values += (variable.observed_value,)
                if any(value is not None and not isinstance(value, str) for value in values):
                    raise TypeError("categorical state values must be exact strings")
                categories.update(value for value in values if value is not None)
        return cls(tuple(sorted(units)), tuple(sorted(classes)), tuple(sorted(categories)),
                   tuple(sorted(attributes)))


@dataclass(frozen=True)
class CandidateBatch:
    """Packed candidates with arbitrary leading batch/branch/time dimensions.

    values [...,N,D,C] and value_mask describe scalar/vector components.
    category_ids [...,N,D] contains exact state-code indices or -1.
    All remaining scalar metadata is [...,N]. owner_classes [...,N,L]
    contains schema class indices or -1 for padding; identities are excluded.
    attribute_ids contains the separately declared canonical field indices.
    History is [...,N,H,C], forecast is [...,N,Q,C], each with masks,
    categories and physical relative times. These are available context only.
    """

    values: torch.Tensor
    value_mask: torch.Tensor
    category_ids: torch.Tensor
    domain_mask: torch.Tensor
    type_ids: torch.Tensor
    unit_ids: torch.Tensor
    attribute_ids: torch.Tensor
    owner_classes: torch.Tensor
    time_seconds: torch.Tensor
    observed: torch.Tensor
    missing: torch.Tensor
    observable: torch.Tensor
    history_values: torch.Tensor
    history_value_mask: torch.Tensor
    history_category_ids: torch.Tensor
    history_times: torch.Tensor
    history_mask: torch.Tensor
    forecast_values: torch.Tensor
    forecast_value_mask: torch.Tensor
    forecast_category_ids: torch.Tensor
    forecast_times: torch.Tensor
    forecast_mask: torch.Tensor


@dataclass(frozen=True)
class EncodedCandidates:
    tokens: torch.Tensor
    domain_mask: torch.Tensor


def _numeric_features(values: torch.Tensor) -> torch.Tensor:
    return torch.stack((values.sign() * values.abs().log1p(), values.tanh(),
                        values / (1 + values.abs()), values.atan()), dim=-1)


def _position_features(values: torch.Tensor) -> torch.Tensor:
    # Continuous time and component/operand positions use no identity vocabulary.
    scales = values.new_tensor((1.0, 0.1, 0.01, 0.001))
    phase = values[..., None] * scales
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class SharedVariableEncoder(nn.Module):
    """Type adapters and context aggregation shared across all variables."""

    def __init__(self, vocabulary: SchemaVocabulary, token_dim: int = TOKEN_DIM):
        super().__init__()
        if token_dim != TOKEN_DIM:
            raise ValueError(f"the unified token contract is {TOKEN_DIM} dimensions")
        self.vocabulary, self.token_dim = vocabulary, token_dim
        self.type_embedding = nn.Embedding(len(DATA_TYPES), token_dim)
        self.unit_embedding = nn.Embedding(len(vocabulary.units), token_dim)
        self.attribute_embedding = nn.Embedding(len(vocabulary.attributes), token_dim)
        self.class_embedding = nn.Embedding(len(vocabulary.classes), token_dim)
        # An empty category registry still allocates one inaccessible parameter row.
        self.category_embedding = nn.Embedding(max(1, len(vocabulary.categories)), token_dim)
        self.missing_value_token = nn.Parameter(torch.zeros(token_dim))
        self.scalar_adapter = nn.Sequential(nn.Linear(4, token_dim), nn.SiLU(),
                                            nn.Linear(token_dim, token_dim))
        self.component_position = nn.Linear(8, token_dim, bias=False)
        self.time_adapter = nn.Sequential(nn.Linear(8, token_dim), nn.SiLU())
        self.observation_adapter = nn.Linear(4, token_dim)
        self.history_adapter = nn.Sequential(nn.Linear(2 * token_dim, token_dim), nn.SiLU())
        self.forecast_adapter = nn.Sequential(nn.Linear(2 * token_dim, token_dim), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(6 * token_dim, token_dim), nn.SiLU(),
                                    nn.Linear(token_dim, token_dim), nn.LayerNorm(token_dim))

    def _value_tokens(self, values, value_mask, categories, active):
        values = torch.where(value_mask, values.detach(), torch.zeros_like(values))
        component_count = values.shape[-1]
        component_positions = torch.arange(component_count, dtype=values.dtype, device=values.device)
        components = self.scalar_adapter[0](_numeric_features(values))
        components = self.scalar_adapter[1](
            components + self.component_position(_position_features(component_positions)))
        components = self.scalar_adapter[2](components)
        count = value_mask.sum(-1).clamp_min(1)[..., None]
        numeric = (components * value_mask[..., None]).sum(-2) / count
        categorical_mask = active & (categories >= 0)
        categorical = self.category_embedding(categories.clamp_min(0))
        missing_candidate = active & ~value_mask.any(-1) & (categories < 0)
        return (numeric + categorical * categorical_mask[..., None]
                + self.missing_value_token * missing_candidate[..., None])

    def _validate(self, batch: CandidateBatch):
        if not isinstance(batch, CandidateBatch):
            raise TypeError("the shared encoder requires CandidateBatch")
        values = batch.values
        if not isinstance(values, torch.Tensor) or values.ndim < 3 or not values.is_floating_point():
            raise ValueError("candidate values need shape [...,N,D,C] and a floating dtype")
        if any(size < 1 for size in values.shape[-3:]):
            raise ValueError("candidate variable/domain/component dimensions must be nonempty")
        shape, scalar_shape = values.shape[:-1], values.shape[:-2]
        parameter = self.type_embedding.weight
        floating_names = ("values", "time_seconds", "history_values", "history_times",
                          "forecast_values", "forecast_times")
        for name in CandidateBatch.__dataclass_fields__:
            tensor = getattr(batch, name)
            if not isinstance(tensor, torch.Tensor) or tensor.device != parameter.device:
                raise ValueError(f"{name} must be a tensor on the model device")
            if name in floating_names and tensor.dtype != parameter.dtype:
                raise ValueError(f"{name} must match model floating dtype")
        for name in ("domain_mask", "category_ids"):
            if getattr(batch, name).shape != shape:
                raise ValueError(f"{name} shape differs from candidate domains")
        if batch.value_mask.shape != values.shape:
            raise ValueError("value_mask shape differs from candidate components")
        for name in ("type_ids", "unit_ids", "attribute_ids", "time_seconds", "observed", "missing", "observable"):
            if getattr(batch, name).shape != scalar_shape:
                raise ValueError(f"{name} shape differs from variable metadata")
        if batch.owner_classes.shape[:-1] != scalar_shape or batch.owner_classes.shape[-1] < 1:
            raise ValueError("owner_classes needs [...,N,L] declared class indices")
        for name in ("value_mask", "domain_mask", "observed", "missing", "observable",
                     "history_value_mask", "history_mask", "forecast_value_mask", "forecast_mask"):
            if getattr(batch, name).dtype != torch.bool:
                raise TypeError(f"{name} must be Boolean")
        for name, bound, allowed_padding in (("type_ids", len(DATA_TYPES), False),
                ("unit_ids", len(self.vocabulary.units), False),
                ("attribute_ids", len(self.vocabulary.attributes), False),
                ("owner_classes", len(self.vocabulary.classes), True),
                ("category_ids", len(self.vocabulary.categories), True)):
            tensor = getattr(batch, name)
            if tensor.dtype != torch.long or bool(((tensor < (-1 if allowed_padding else 0)) |
                                                   (tensor >= bound)).any()):
                raise ValueError(f"{name} has an undeclared schema index")
        if not bool(batch.domain_mask.any(-1).all()):
            raise ValueError("every variable needs at least one declared candidate")
        if not bool((batch.owner_classes >= 0).any(-1).all()):
            raise ValueError("every variable needs at least one typed owner")
        if bool((batch.observed & batch.missing).any()):
            raise ValueError("an observed variable cannot have missing evidence")
        if bool((batch.value_mask & ~batch.domain_mask[..., None]).any()):
            raise ValueError("padded candidates cannot carry numeric components")
        if bool(((batch.category_ids >= 0) & ~batch.domain_mask).any()):
            raise ValueError("padded candidates cannot carry categorical values")
        if not bool(torch.isfinite(values[batch.value_mask]).all()) or not bool(torch.isfinite(batch.time_seconds).all()):
            raise ValueError("candidate values and physical times must be finite")
        for prefix in ("history", "forecast"):
            context = getattr(batch, prefix + "_values")
            mask = getattr(batch, prefix + "_mask")
            component_mask = getattr(batch, prefix + "_value_mask")
            categories = getattr(batch, prefix + "_category_ids")
            times = getattr(batch, prefix + "_times")
            if context.ndim != values.ndim or context.shape[:-2] != scalar_shape or context.shape[-1] != values.shape[-1]:
                raise ValueError(f"{prefix} values need [...,N,H_or_Q,C]")
            if component_mask.shape != context.shape or any(t.shape != context.shape[:-1] for t in (mask, categories, times)):
                raise ValueError(f"{prefix} masks/categories/times have inconsistent shapes")
            if categories.dtype != torch.long or bool(((categories < -1) | (categories >= len(self.vocabulary.categories))).any()):
                raise ValueError(f"{prefix} contains an undeclared category")
            if bool((component_mask & ~mask[..., None]).any()) or bool(((categories >= 0) & ~mask).any()):
                raise ValueError(f"padded {prefix} entries carry a value")
            if not bool(torch.isfinite(context[component_mask]).all()) or not bool(torch.isfinite(times[mask]).all()):
                raise ValueError(f"available {prefix} values and times must be finite")
            if prefix == "history" and bool((mask & (times > 0)).any()):
                raise ValueError("history contains future evidence relative to the anchor")

    def forward(self, batch: CandidateBatch) -> EncodedCandidates:
        self._validate(batch)
        value = self._value_tokens(batch.values, batch.value_mask, batch.category_ids, batch.domain_mask)
        class_mask = batch.owner_classes >= 0
        classes = self.class_embedding(batch.owner_classes.clamp_min(0))
        class_mean = (classes * class_mask[..., None]).sum(-2) / class_mask.sum(-1).clamp_min(1)[..., None]
        metadata = (self.type_embedding(batch.type_ids) + self.unit_embedding(batch.unit_ids)
                    + self.attribute_embedding(batch.attribute_ids) + class_mean)
        flags = torch.stack((batch.observed, batch.missing, batch.observable,
                             batch.owner_classes.ge(0).sum(-1).to(batch.values.dtype)), dim=-1).to(batch.values.dtype)
        observation = self.observation_adapter(flags.detach())
        time = self.time_adapter(_position_features(batch.time_seconds.detach()))
        contexts = []
        for prefix, adapter in (("history", self.history_adapter), ("forecast", self.forecast_adapter)):
            mask = getattr(batch, prefix + "_mask")
            context_value = self._value_tokens(getattr(batch, prefix + "_values"),
                getattr(batch, prefix + "_value_mask"), getattr(batch, prefix + "_category_ids"), mask)
            times = getattr(batch, prefix + "_times")
            safe_times = torch.where(mask, times.detach(), torch.zeros_like(times))
            context = adapter(torch.cat((context_value, self.time_adapter(_position_features(safe_times))), dim=-1))
            contexts.append((context * mask[..., None]).sum(-2) / mask.sum(-1).clamp_min(1)[..., None])
        domain_size = batch.values.shape[-2]
        channels = (metadata, observation, time, *contexts)
        expanded = [tensor.unsqueeze(-2).expand(*tensor.shape[:-1], domain_size, self.token_dim) for tensor in channels]
        encoded = self.output(torch.cat((value, *expanded), dim=-1))
        encoded = encoded * batch.domain_mask[..., None]
        return EncodedCandidates(encoded, batch.domain_mask)


def pack_variables(variables: Sequence["CandidateVariable"], vocabulary: SchemaVocabulary,
                   *, attribute_symbols: Mapping, anchor_tick: int, tick_seconds: float, dtype=torch.float32,
                   device=None) -> CandidateBatch:
    """Pack explicit finite domains without exposing identity strings to the model."""
    variables = tuple(variables)
    if not variables or type(anchor_tick) is not int or not math.isfinite(tick_seconds) or tick_seconds <= 0:
        raise ValueError("packing needs variables, an explicit integer anchor and positive seconds per tick")
    if len({variable.key for variable in variables}) != len(variables):
        raise ValueError("shared variables must be interned before encoding")
    if set(attribute_symbols) != {variable.key for variable in variables}:
        raise ValueError("attribute declarations must match the exact packed variables")
    count, domains = len(variables), max(len(variable.domain) for variable in variables)
    components = max((len(value) if isinstance(value, tuple) else 1)
        for variable in variables for value in (*variable.domain,
            *(value for _, value in variable.history), *variable.forecast_context))
    owners = max(len(variable.key.owner) for variable in variables)
    history_size = max(1, max(len(variable.history) for variable in variables))
    forecast_size = max(1, max(len(variable.forecast_context) for variable in variables))
    floating = lambda shape: torch.zeros(shape, dtype=dtype, device=device)
    integer = lambda shape: torch.full(shape, -1, dtype=torch.long, device=device)
    boolean = lambda shape: torch.zeros(shape, dtype=torch.bool, device=device)
    values, value_mask, category_ids = floating((count, domains, components)), boolean((count, domains, components)), integer((count, domains))
    domain_mask = boolean((count, domains))
    type_ids, unit_ids, attribute_ids = integer((count,)), integer((count,)), integer((count,))
    owner_classes = integer((count, owners))
    times = floating((count,))
    observed, missing, observable = boolean((count,)), boolean((count,)), boolean((count,))
    history_values, history_value_mask = floating((count, history_size, components)), boolean((count, history_size, components))
    history_categories, history_times, history_mask = integer((count, history_size)), floating((count, history_size)), boolean((count, history_size))
    forecast_values, forecast_value_mask = floating((count, forecast_size, components)), boolean((count, forecast_size, components))
    forecast_categories, forecast_times, forecast_mask = integer((count, forecast_size)), floating((count, forecast_size)), boolean((count, forecast_size))

    def put(value, data_type, output, output_mask, output_category):
        if value is None:
            return  # An explicit absent candidate has a learned shared missing token.
        if data_type in {"categorical", "four_value"}:
            if not isinstance(value, str):
                raise TypeError("declared categorical candidates must be strings")
            output_category.fill_(vocabulary.index("categories", value))
        elif data_type == "identifier":
            if not isinstance(value, str) or not value:
                raise TypeError("identifier candidates must be nonempty strings")
            # Presence is invariant to identifier renumbering. Equality relations
            # belong to the compiler or explicit incidence relation features.
            output[0], output_mask[0] = 1.0, True
        else:
            sequence = value if data_type == "vector" else (value,)
            if not isinstance(sequence, tuple) or not sequence:
                raise TypeError("numeric vector values must be nonempty tuples")
            if data_type == "boolean" and any(type(item) is not bool for item in sequence):
                raise TypeError("Boolean schema candidates must be Boolean")
            if data_type == "integer" and any(type(item) is not int for item in sequence):
                raise TypeError("Integer schema candidates must be exact int")
            if any(not isinstance(item, (float, int)) or not math.isfinite(item) for item in sequence):
                raise TypeError("numeric candidates must be finite real values")
            output[:len(sequence)] = torch.tensor(sequence, dtype=dtype, device=device)
            output_mask[:len(sequence)] = True

    for row, variable in enumerate(variables):
        schema = variable.schema
        if schema.data_type not in DATA_TYPES:
            raise ValueError(f"unsupported schema data type: {schema.data_type}")
        type_ids[row] = DATA_TYPES.index(schema.data_type)
        unit_ids[row] = vocabulary.index("units", schema.unit)
        attribute_ids[row] = vocabulary.index("attributes", attribute_symbols[variable.key])
        for column, owner in enumerate(variable.key.owner):
            owner_classes[row, column] = vocabulary.index("classes", owner.ontology_class)
        times[row] = (variable.key.tick - anchor_tick) * tick_seconds
        observed[row] = variable.observed
        missing[row] = variable.observed_value is None
        observable[row] = variable.observable
        for candidate, value in enumerate(variable.domain):
            domain_mask[row, candidate] = True
            put(value, schema.data_type, values[row, candidate], value_mask[row, candidate], category_ids[row, candidate])
        for index, (tick, value) in enumerate(variable.history):
            if type(tick) is not int or tick > anchor_tick:
                raise ValueError("history must contain only integer ticks in the observed prefix")
            history_mask[row, index] = True
            history_times[row, index] = (tick - anchor_tick) * tick_seconds
            put(value, schema.data_type, history_values[row, index], history_value_mask[row, index], history_categories[row, index])
        for index, value in enumerate(variable.forecast_context):
            forecast_mask[row, index] = True
            forecast_times[row, index] = times[row]
            put(value, schema.data_type, forecast_values[row, index], forecast_value_mask[row, index], forecast_categories[row, index])
    return CandidateBatch(values, value_mask, category_ids, domain_mask, type_ids, unit_ids, attribute_ids,
        owner_classes, times, observed, missing, observable, history_values, history_value_mask,
        history_categories, history_times, history_mask, forecast_values, forecast_value_mask,
        forecast_categories, forecast_times, forecast_mask)
