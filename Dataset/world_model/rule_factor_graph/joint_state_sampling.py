"""Typed receiver-observed TRAIN numeric laws with exact query anchors.

Generator-score innovation uses an explicit channel prior, leave-component-out
ridge mean and whole-trajectory residual bootstrap. Reflection maps proposals
to the closed score domain before the producer's Python round6 decoding.
"""

from dataclasses import dataclass, field
from itertools import combinations
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping
from threading import local
import numpy as np
import torch

INNOVATION_RIDGE_LAMBDA = .001


def _text(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty string")
    return value


def _integer(value, lower, upper=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Expected an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError("Integer outside allowed range")
    return value


def _immutable(array):
    array = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


# Fixed degree-2 least-squares geometry, independent of TRAIN or query values.
_QUADRATIC_PAST = np.vander(np.arange(-3, 1, dtype=np.float64), 3, increasing=True)
_QUADRATIC_FUTURE = np.vander(np.arange(1, 11, dtype=np.float64), 3, increasing=True)
_QUADRATIC_SCORE_MAP = _immutable(_QUADRATIC_FUTURE @ np.linalg.solve(
    _QUADRATIC_PAST.T @ _QUADRATIC_PAST, _QUADRATIC_PAST.T))


def _grid(ticks, lengths):
    ticks = tuple(_integer(t, 0) for t in ticks)
    if len(ticks) not in lengths:
        raise ValueError("Unexpected sequence length")
    if any(b - a != 5 for a, b in zip(ticks, ticks[1:])):
        raise ValueError("Ticks must form a five-tick grid")
    return ticks


def _refs(refs, length):
    refs = tuple(refs)
    if len(refs) != length:
        raise ValueError("Source references must align with ticks")
    for ref in refs:
        _text(ref)
    return refs


@dataclass(frozen=True)
class NumericField:
    artifact: str
    field: str
    unit: str
    owner_roles: tuple
    lower: float = None
    upper: float = None
    transform: str = "identity"

    def __post_init__(self):
        if self.transform not in ("identity", "log_positive", "logit_open_unit"):
            raise ValueError("Numeric transform must be explicitly declared")
        for value in (self.artifact, self.field, self.unit):
            _text(value)
        roles = tuple(self.owner_roles)
        for role in roles:
            _text(role)
        if not roles or len(set(roles)) != len(roles):
            raise ValueError("Field owner roles must be nonempty and unique")
        object.__setattr__(self, "owner_roles", roles)
        for name in ("lower", "upper"):
            bound = getattr(self, name)
            if bound is not None:
                if isinstance(bound, bool) or not isinstance(bound, Real):
                    raise TypeError("Bounds must be real numbers")
                if not np.isfinite(bound):
                    raise ValueError("Bounds must be finite")
                object.__setattr__(self, name, float(bound))
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("Lower bound exceeds upper bound")
        if self.transform == "log_positive" and (self.lower != 0 or self.upper is not None):
            raise ValueError("log_positive requires an explicitly declared nonnegative unbounded field")
        if self.transform == "logit_open_unit" and (self.lower != 0 or self.upper != 1):
            raise ValueError("logit_open_unit requires explicitly declared unit-interval bounds")



@dataclass(frozen=True)
class GeneratorScoreProfile:
    """Declared channel prior; N/W/M and ordered receiver field identities."""
    nominal_latency_ms: float
    worst_case_latency_ms: float
    maximum_packet_loss_ratio: float
    fields: tuple
    source_ref: str
    producer_version: str

    def __post_init__(self):
        _text(self.source_ref)
        _text(self.producer_version)
        for name in ("nominal_latency_ms", "worst_case_latency_ms", "maximum_packet_loss_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError("Profile parameters must be finite real numbers")
            object.__setattr__(self, name, float(value))
        n, w, m = self.nominal_latency_ms, self.worst_case_latency_ms, self.maximum_packet_loss_ratio
        if not (0 <= n < w and 0 < m <= 1):
            raise ValueError("Profile requires 0<=N<W and 0<M<=1")
        fields = tuple(self.fields)
        if (len(fields) != 2 or any(not isinstance(f, NumericField) for f in fields)
                or tuple(f.transform for f in fields) != ("identity", "identity")
                or (fields[0].lower, fields[0].upper, fields[1].lower, fields[1].upper) != (n, w, 0., m)):
            raise ValueError("Profile requires ordered identity fields with closed N/W and 0/M bounds")
        object.__setattr__(self, "fields", fields)

    def decode(self, u):
        u = np.asarray(u, dtype=np.float64)
        if not np.isfinite(u).all() or np.any(u < 0) or np.any(u > 1):
            raise ValueError("Generator badness must lie in closed [0,1]")
        n, span, m = self.nominal_latency_ms, self.worst_case_latency_ms-self.nominal_latency_ms, self.maximum_packet_loss_ratio
        # Python round(float,6) is the producer's _round authority.
        flat = u.ravel().tolist()
        latency = np.array([round(n+span*v, 6) for v in flat], dtype=np.float64)
        loss = np.array([round(m*v**2, 6) for v in flat], dtype=np.float64)
        return _immutable(np.stack((latency, loss), -1).reshape(u.shape+(2,)))

    def encode(self, values):
        if not isinstance(values, np.ndarray) or values.ndim < 1 or values.shape[-1] != 2:
            raise ValueError("Generator observations require ordered latency/loss pairs")
        values = _values(values, values.shape, self.fields)
        n, span, m = self.nominal_latency_ms, self.worst_case_latency_ms-self.nominal_latency_ms, self.maximum_packet_loss_ratio
        latency, loss = values[..., 0], values[..., 1]
        # Intersect both six-decimal rounding cells with the declared closed domain.
        lo = np.maximum(0., np.maximum((latency-.0000005-n)/span, np.sqrt(np.maximum(0., loss-.0000005)/m)))
        hi = np.minimum(1., np.minimum((latency+.0000005-n)/span, np.sqrt((loss+.0000005)/m)))
        if np.any(lo > hi):
            raise ValueError("Observed pair has no shared generator-score rounding interval")
        u = (lo+hi)/2
        for endpoint in (0., 1.):
            pair = self.decode(np.array(endpoint))
            u = np.where(np.all(values == pair, axis=-1), endpoint, u)
        if not np.array_equal(self.decode(u), values):
            raise ValueError("Observed pair disagrees with producer Python round6 authority")
        return _immutable(u)


def _metadata(owners, fields):
    owners = tuple(tuple(owner) for owner in owners)
    if not owners or any(len(owner) != 3 for owner in owners):
        raise ValueError("Owners must contain (role, id, class) triples")
    for owner in owners:
        for value in owner:
            _text(value)
    roles = tuple(owner[0] for owner in owners)
    if len(set(roles)) != len(roles):
        raise ValueError("Owner roles must be unique")
    fields = tuple(fields)
    if not fields or any(not isinstance(f, NumericField) for f in fields):
        raise TypeError("Fields must be NumericField instances")
    keys = tuple((f.artifact, f.field) for f in fields)
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate numeric fields")
    if any(not set(f.owner_roles).issubset(roles) for f in fields):
        raise ValueError("A field references an absent owner role")
    return owners, fields


def _values(values, shape, fields):
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float64):
        raise TypeError("Values must be a float64 ndarray")
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("Values must have the declared shape and be finite")
    for i, field in enumerate(fields):
        column = values[..., i]
        if field.lower is not None and np.any(column < field.lower):
            raise ValueError("Value below field lower bound")
        if field.upper is not None and np.any(column > field.upper):
            raise ValueError("Value above field upper bound")
    return _immutable(values)


@dataclass(frozen=True)
class NumericSequence:
    episode: str
    component: str
    owners: tuple
    fields: tuple
    ticks: tuple
    availability_ticks: tuple
    values: np.ndarray
    source_refs: tuple

    def __post_init__(self):
        _text(self.episode)
        _text(self.component)
        owners, fields = _metadata(self.owners, self.fields)
        ticks = _grid(self.ticks, (8, 18))
        available = tuple(_integer(t, 0) for t in self.availability_ticks)
        if available != ticks:
            raise ValueError("Receiver availability must equal observation ticks")
        for name, value in (
            ("owners", owners), ("fields", fields), ("ticks", ticks),
            ("availability_ticks", available),
            ("source_refs", _refs(self.source_refs, len(ticks))),
            ("values", _values(self.values, (len(ticks), len(fields)), fields)),
        ):
            object.__setattr__(self, name, value)


def _schema(sequence):
    return sequence.fields, tuple((role, cls) for role, _, cls in sequence.owners)


def _identity(sequence):
    return sequence.episode, sequence.component, sequence.owners, sequence.ticks[0]


@dataclass(frozen=True)
class ExtremaSamplingDesign:
    """Disjoint full-membership atoms covering both outcome-column extrema."""
    atoms: tuple
    members: tuple
    counts: tuple
    population_size: int

    def __post_init__(self):
        population = _integer(self.population_size, 1)
        atoms = tuple(tuple(tuple(bit for bit in atom) for atom in group) for group in self.atoms)
        members = tuple(tuple(group) for group in self.members)
        counts = tuple(_integer(n, 1, 16) for n in self.counts)
        if not atoms or len(atoms) != len(members) or len(atoms) != len(counts):
            raise ValueError("Stratum atoms, members and counts must align")
        if any(not group for group in members):
            raise ValueError("Strata must partition every candidate exactly once")
        flat = np.concatenate([np.asarray(group) for group in members])
        if flat.dtype.kind != "i" or not np.array_equal(np.sort(flat), np.arange(population)):
            raise ValueError("Strata must partition every candidate exactly once")
        flat_atoms = tuple(atom for group in atoms for atom in group)
        if (any(len(atom) != 4 or any(type(bit) is not bool for bit in atom) for atom in flat_atoms)
                or len(set(flat_atoms)) != len(flat_atoms) or any(not group for group in atoms)):
            raise ValueError("Strata require unique four-Boolean membership atoms")
        if sum(counts) > 16:
            raise ValueError("Stratified branch budget exceeds16")
        for name, value in (("atoms", atoms), ("members", members), ("counts", counts), ("population_size", population)):
            object.__setattr__(self, name, value)


def extrema_sampling_design(outcomes, K):
    """Exact minimum-cardinality lexicographic atom cover, plus remainder.

    outcomes is the complete candidate population's two hard-rule first-flip
    indices:0..9 for horizons5..50,10 for beyond prediction horizon. No target
    labels enter this design. Surplus uses largest remainder after one draw
    per nonempty stratum; ties follow the canonical stratum order.
    """
    K = _integer(K, 1, 16)
    if (not isinstance(outcomes, np.ndarray) or outcomes.ndim != 2 or outcomes.shape[1] != 2
            or not len(outcomes) or outcomes.dtype.kind not in "iu"
            or np.any(outcomes < 0) or np.any(outcomes > 10)):
        raise ValueError("Hard first-flip outcomes must be integer [N,2] indices0..10")
    minimum, maximum = outcomes.min(0), outcomes.max(0)
    membership = np.stack((outcomes[:, 0] == minimum[0], outcomes[:, 0] == maximum[0],
                           outcomes[:, 1] == minimum[1], outcomes[:, 1] == maximum[1]), axis=1)
    # Packed codes (first column most significant) order exactly like tuples of Booleans.
    codes = membership @ np.array([8, 4, 2, 1])
    atoms = tuple(tuple(bool(code >> shift & 1) for shift in (3, 2, 1, 0)) for code in np.unique(codes).tolist())
    cover = None
    for size in range(1, len(atoms)+1):
        for candidate in combinations(atoms, size):
            if all(any(atom[column] for atom in candidate) for column in range(4)):
                cover = candidate
                break
        if cover is not None:
            break
    if cover is None:
        raise ValueError("No atom cover of candidate extrema")
    remaining = tuple(atom for atom in atoms if atom not in cover)
    groups = tuple((atom,) for atom in cover) + ((remaining,) if remaining else ())
    group_codes = tuple([sum(bit << shift for bit, shift in zip(atom, (3, 2, 1, 0))) for atom in group] for group in groups)
    members = tuple(tuple(np.flatnonzero(np.isin(codes, code_list)).tolist()) for code_list in group_codes)
    if K < len(groups):
        raise ValueError("Branch budget cannot reserve every positive nonempty stratum")
    surplus = K-len(groups)
    sizes = tuple(len(group) for group in members)
    # Integer quotients/remainders avoid floating-point allocation ties.
    quotients = tuple(surplus*size//len(codes) for size in sizes)
    remainders = tuple(surplus*size % len(codes) for size in sizes)
    counts = [1+count for count in quotients]
    for index in sorted(range(len(groups)), key=lambda i: (-remainders[i], i))[:K-sum(counts)]:
        counts[index] += 1
    return ExtremaSamplingDesign(groups, members, tuple(counts), len(codes))


@dataclass(frozen=True)
class OrdinaryDesign:
    """One stratum holding the whole eligible pool: eight ordinary draws, every weight 1/K."""
    population_size: int
    counts: tuple
    members: tuple

    @classmethod
    def whole_pool(cls, population, K):
        return cls(_integer(population, 1), (_integer(K, 1, 16),), (range(population),))


@dataclass(frozen=True)
class NumericBranches:
    values: np.ndarray
    ticks: tuple
    fields: tuple
    query_owners: tuple
    query_episode: str
    query_component: str
    donor_ids: tuple
    donor_components: tuple
    donor_refs: tuple
    bank_source_refs: tuple
    prior_weights: np.ndarray
    q_law: str
    query_source_refs: tuple
    fit_source_refs: tuple
    outcome_classes: np.ndarray
    branch_strata: tuple
    sampling_design: object
    fit_context_excluded_component: str | None
    law_description: str
    score_profile: GeneratorScoreProfile | None = None
    folded_proposal_diagnostics: tuple = ()

    def __post_init__(self):
        _text(self.query_episode)
        _text(self.query_component)
        owners, fields = _metadata(self.query_owners, self.fields)
        ticks = _grid(self.ticks, (11,))
        ids = tuple(tuple(identity) for identity in self.donor_ids)
        K = _integer(len(ids), 1, 16)
        components = tuple(self.donor_components)
        if len(components) != K:
            raise ValueError("Donor components must align with branches")
        for identity, component in zip(ids, components):
            if len(identity) != 4:
                raise ValueError("Invalid donor identity")
            episode, identified_component, donor_owners, start = identity
            _text(episode)
            _text(component)
            donor_owners, _ = _metadata(donor_owners, fields)
            _integer(start, 0)
            if identified_component != component:
                raise ValueError("Donor identity/component mismatch")
            if episode == self.query_episode or component == self.query_component:
                raise ValueError("Query episode/component cannot be a donor")
            if tuple((r, c) for r, _, c in donor_owners) != tuple((r, c) for r, _, c in owners):
                raise ValueError("Donor ROLE+CLASS schema differs")
        ids = tuple((e, c, tuple(tuple(o) for o in oo), int(t)) for e, c, oo, t in ids)
        refs = tuple(_refs(ref, 10) for ref in self.donor_refs)
        bank_refs = tuple(self.bank_source_refs)  # fit validated each TRAIN path once
        if len(refs) != K or not bank_refs:
            raise ValueError("Missing donor or bank source references")
        if self.q_law not in ("absolute_train_donor", "conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation"):
            raise ValueError("Unknown explicitly declared numeric Q law")
        if self.q_law in ("generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation"):
            if not isinstance(self.score_profile, GeneratorScoreProfile) or self.score_profile.fields != fields:
                raise ValueError("Generator branch profile/field provenance missing or drifting")
            diagnostics = tuple(tuple(row) for row in self.folded_proposal_diagnostics)
            if (len(diagnostics) != K or any(len(row) != 3 or not 0 <= row[0] <= 10
                    or not all(np.isfinite(v) for v in row) or row[1] > row[2] for row in diagnostics)):
                raise ValueError("Folded proposal diagnostics must contain count/min/max per branch")
            object.__setattr__(self, "folded_proposal_diagnostics", diagnostics)
        elif self.score_profile is not None or self.folded_proposal_diagnostics:
            raise ValueError("Generator provenance is only valid for generator law")
        _text(self.law_description)
        if self.fit_context_excluded_component is not None:
            _text(self.fit_context_excluded_component)
            if self.fit_context_excluded_component != self.query_component:
                raise ValueError("Excluded fit context must match query component")
        query_refs = _refs(self.query_source_refs, 8)
        fit_refs = tuple(self.fit_source_refs)  # InnovationContext validated and indexed these once
        if not fit_refs:
            raise ValueError("Fit lineage must contain actual complete TRAIN paths")
        design = self.sampling_design
        if not isinstance(design, (ExtremaSamplingDesign, OrdinaryDesign)) or sum(design.counts) != K:
            raise ValueError("Sampling design must match branch budget")
        strata = tuple(_integer(i, 0, len(design.counts)-1) for i in self.branch_strata)
        if len(strata) != K or tuple(strata.count(i) for i in range(len(design.counts))) != design.counts:
            raise ValueError("Branch stratum assignments differ from design")
        weights = self.prior_weights
        if (not isinstance(weights, np.ndarray) or weights.dtype != np.float64 or weights.shape != (K,)
                or not np.isfinite(weights).all() or np.any(weights <= 0)):
            raise ValueError("Prior weights must be positive finite float64 [K]")
        expected = np.array([len(design.members[i])/(design.population_size*design.counts[i]) for i in strata])
        if not np.array_equal(weights, expected) or not np.isclose(weights.sum(), 1., rtol=1e-14, atol=1e-14):
            raise ValueError("Prior weights must equal stratum size/(population*draw count)")
        outcomes = self.outcome_classes
        if (not isinstance(outcomes, np.ndarray) or outcomes.shape != (K, 2) or outcomes.dtype.kind not in "iu"
                or np.any(outcomes < 0) or np.any(outcomes > 10)):
            raise ValueError("Sampled hard outcomes must be integer [K,2] indices0..10")
        outcomes = np.frombuffer(np.ascontiguousarray(outcomes, dtype=np.int64).tobytes(), dtype=np.int64).reshape(K, 2)
        for name, value in (
            ("query_owners", owners), ("fields", fields), ("ticks", ticks),
            ("donor_ids", ids), ("donor_components", components),
            ("donor_refs", refs), ("bank_source_refs", bank_refs),
            ("values", _values(self.values, (K, 11, len(fields)), fields)),
            ("prior_weights", _immutable(weights)), ("query_source_refs", query_refs),
            ("fit_source_refs", fit_refs), ("outcome_classes", outcomes), ("branch_strata", strata),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class InnovationContext:
    excluded_component: str | None
    bank_indices: tuple
    history_mean: np.ndarray
    history_scale: np.ndarray
    normalized_histories: np.ndarray
    coefficients: np.ndarray
    residuals: np.ndarray
    source_refs: tuple
    forecast_law: str = "ridge_mean"

    def __post_init__(self):
        if self.forecast_law not in ("ridge_mean", "last_difference", "quadratic_last4"):
            raise ValueError("Unknown innovation context forecast law")
        if self.excluded_component is not None:
            _text(self.excluded_component)
        indices = tuple(_integer(i, 0) for i in self.bank_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Innovation context needs unique complete TRAIN paths")
        dimension = self.history_mean.shape[0]
        if dimension not in (8, 16):
            raise ValueError("Innovation context requires explicit 8D score or 16D joint history")
        output = 10 if dimension == 8 else 20
        shapes = {"history_mean": (dimension,), "history_scale": (dimension,),
                  "normalized_histories": (len(indices), dimension),
                  "coefficients": (dimension+1, output), "residuals": (len(indices), output)}
        for name, shape in shapes.items():
            value = getattr(self, name)
            if name == "coefficients" and self.forecast_law in ("last_difference", "quadratic_last4"):
                if value is not None or dimension != 8:
                    raise ValueError("Explicit trend context requires no regression coefficients and score H8")
                continue
            if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                    or value.shape != shape or not np.isfinite(value).all()):
                raise ValueError("Invalid finite float64 innovation context parameter: " + name)
            object.__setattr__(self, name, _immutable(value))
        if np.any(self.history_scale <= 0):
            raise ValueError("Innovation history scales must be positive")
        refs = tuple(_refs(ref, 18) for ref in self.source_refs)
        if len(refs) != len(indices):
            raise ValueError("Innovation fit lineage must align with context paths")
        object.__setattr__(self, "bank_indices", indices)
        object.__setattr__(self, "source_refs", refs)


def _transform_values(values, fields, device):
    if len(fields) != 2 or tuple(f.transform for f in fields) != ("log_positive", "logit_open_unit"):
        raise ValueError("Innovation law requires explicitly ordered log-positive and logit-open-unit declarations")
    x = torch.tensor(np.array(values, copy=True), dtype=torch.float64, device=device)
    if not bool(torch.isfinite(x).all()) or bool((x[..., 0] <= 0).any() | (x[..., 1] <= 0).any() | (x[..., 1] >= 1).any()):
        raise ValueError("Innovation transform requires latency>0 and0<loss<1; endpoints are unsupported")
    z = torch.stack((x[..., 0].log(), x[..., 1].log()-torch.log1p(-x[..., 1])), -1)
    if not bool(torch.isfinite(z).all()):
        raise FloatingPointError("Nonfinite transformed innovation input")
    return z


def _history16(transformed):
    return torch.cat((transformed[..., -1, :], torch.diff(transformed, dim=-2).flatten(-2)), -1)


def _inverse_values(transformed):
    x = torch.stack((transformed[..., 0].exp(), transformed[..., 1].sigmoid()), -1)
    if (not bool(torch.isfinite(transformed).all() & torch.isfinite(x).all())
            or bool((x[..., 0] <= 0).any() | (x[..., 1] <= 0).any() | (x[..., 1] >= 1).any())):
        raise FloatingPointError("Innovation inverse overflow/underflow or rounded boundary; no clipping applied")
    return x


def _fit_innovation_context(bank, indices, excluded, device, score_profile=None, *, trend_centered=False, quadratic_map=None):
    indices = tuple(indices)
    if not indices:
        raise ValueError("Empty leave-component-out innovation context")
    with torch.no_grad():
        values = np.stack([bank[i].values for i in indices])
        if score_profile is None:
            z = _transform_values(values, bank[0].fields, device)
            h = _history16(z[:, :8])
        else:
            z = torch.tensor(score_profile.encode(values).copy(), dtype=torch.float64, device=device)
            h = torch.cat((z[:, 7:8], torch.diff(z[:, :8], dim=1)), 1)
        mean, scale = h.mean(0), h.std(0, correction=0)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        normalized = (h-mean)/scale
        if quadratic_map is not None:
            prior = z[:, 4:8] @ quadratic_map.T
            coefficients = None
            residuals = z[:, 8:]-prior
        elif trend_centered:
            steps = torch.arange(1, 11, dtype=torch.float64, device=device)
            prior = z[:, 7:8] + (z[:, 7:8]-z[:, 6:7])*steps
            coefficients = None
            residuals = z[:, 8:]-prior
        else:
            design = torch.cat((torch.ones((len(indices), 1), dtype=torch.float64, device=device), normalized), 1)
            target = (z[:, 8:]-z[:, 7:8]).flatten(1)
            penalty = torch.eye(design.shape[1], dtype=torch.float64, device=device)*INNOVATION_RIDGE_LAMBDA
            penalty[0, 0] = 0
            coefficients = torch.linalg.solve(design.T@design/len(indices) + penalty, design.T@target/len(indices))
            residuals = target-design@coefficients
        arrays = tuple(None if t is None else t.detach().cpu().numpy()
                       for t in (mean, scale, normalized, coefficients, residuals))
    return InnovationContext(excluded, indices, *arrays, tuple(bank[i].source_refs for i in indices),
                             "quadratic_last4" if quadratic_map is not None else "last_difference" if trend_centered else "ridge_mean")


@dataclass(frozen=True, init=False)
class JointNumericStateQ:
    bank: tuple
    train_components: tuple
    field_mean: np.ndarray
    field_scale: np.ndarray
    normalized_histories: np.ndarray
    bank_source_refs: tuple
    neighbor_pool: int | None
    q_law: str
    innovation_contexts: tuple
    fit_device: str | None
    score_profile: GeneratorScoreProfile | None
    episode_component: Mapping = field(repr=False, compare=False)
    context_pools: tuple = field(repr=False, compare=False)
    context_episodes: tuple = field(repr=False, compare=False)
    context_tensors: tuple = field(repr=False, compare=False)
    context_positions: tuple = field(repr=False, compare=False)
    quadratic_score_map: torch.Tensor | None = field(repr=False, compare=False)
    _query_candidates: local = field(repr=False, compare=False)

    @classmethod
    def fit(cls, donors, train_components, neighbor_pool=64, *, q_law="absolute_train_donor", device=None, score_profile=None):
        """Fit the explicitly named law; conditional regression requires CUDA.

        None selects the full eligible empirical residual/donor pool. A finite
        neighbor_pool changes that population to nearest standardized histories.
        Conditional contexts fit all TRAIN and each leave-one-component-out bank.
        """
        if q_law not in ("absolute_train_donor", "conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation"):
            raise ValueError("Unknown explicit Q law")
        if q_law in ("conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation"):
            if device is None:
                raise ValueError("Conditional innovation fit requires an explicit CUDA device")
            device = torch.device(device)
            if device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError("Conditional innovation fit/query operations require CUDA")
            device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        elif device is not None:
            raise ValueError("Absolute empirical law has no regression fit device")
        if q_law in ("generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation"):
            if not isinstance(score_profile, GeneratorScoreProfile):
                raise TypeError("Generator innovation requires an explicit GeneratorScoreProfile")
        elif score_profile is not None:
            raise ValueError("Score profile requires generator_score_innovation law")
        pool = None if neighbor_pool is None else _integer(neighbor_pool, 1)
        if q_law in ("generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation") and pool != 128:
            raise ValueError("Polynomial empirical innovation explicitly requires 128 nearest TRAIN histories")
        if isinstance(train_components, str):
            raise TypeError("TRAIN components must be an iterable of identifiers")
        train = tuple(sorted(set(_text(c) for c in train_components)))
        bank = tuple(donors)
        if not bank:
            raise ValueError("A nonempty donor bank is required")
        if any(not isinstance(d, NumericSequence) for d in bank):
            raise TypeError("Donors must be NumericSequence instances")
        if any(len(d.ticks) != 18 or d.component not in train for d in bank):
            raise ValueError("Every donor must be an 18-point TRAIN sequence")
        if any(_schema(d) != _schema(bank[0]) for d in bank):
            raise ValueError("Donor field order or ROLE+CLASS schema differs")
        if len({_identity(d) for d in bank}) != len(bank):
            raise ValueError("Duplicate donor identities")
        bank = tuple(sorted(bank, key=_identity))
        if {d.component for d in bank} != set(train):
            raise ValueError("Every declared TRAIN component must have donors")
        episode_components = {}
        for donor in bank:
            if donor.episode in episode_components and episode_components[donor.episode] != donor.component:
                raise ValueError("One source episode cannot belong to two components")
            episode_components[donor.episode] = donor.component
        if score_profile is not None:
            if bank[0].fields != score_profile.fields:
                raise ValueError("Bank ordered field identities differ from governing score profile")
            score_profile.encode(np.stack([d.values for d in bank]))
        quadratic_map = (torch.tensor(_QUADRATIC_SCORE_MAP.copy(), dtype=torch.float64, device=device)
                         if q_law == "generator_quadratic_empirical_innovation" else None)
        contexts = ()
        mean = scale = normalized = None
        if q_law in ("conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation"):
            contexts = tuple(_fit_innovation_context(bank,
                tuple(i for i, d in enumerate(bank) if d.component != excluded), excluded, device, score_profile,
                trend_centered=q_law == "generator_trend_empirical_innovation", quadratic_map=quadratic_map)
                for excluded in (None, *train))
        else:
            history = np.stack([d.values[:8] for d in bank])
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                mean = history.mean(axis=(0, 1))
                scale = history.std(axis=(0, 1))
                scale[scale == 0] = 1
                normalized = (history - mean) / scale
            if not all(np.isfinite(a).all() for a in (mean, scale, normalized)):
                raise ValueError("Nonfinite TRAIN normalization statistics")
        # Eligible component/episode groups and device tensors are fixed TRAIN-only data: built once.
        context_pools = tuple(tuple(bank[i] for i in c.bank_indices) for c in contexts)
        context_episodes = tuple(frozenset(d.episode for d in group) for group in context_pools)
        if q_law in ("generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation") and any(len(group) < 128 for group in context_pools):
            raise ValueError("Every polynomial-law LOCO context must provide at least 128 eligible TRAIN histories")
        context_positions = tuple(MappingProxyType({_identity(d): row for row, d in enumerate(group)})
                                  for group in context_pools)
        context_tensors = tuple(tuple(None if c.forecast_law in ("last_difference", "quadratic_last4") and name == "coefficients"
                                     else torch.tensor(getattr(c, name).copy(), dtype=torch.float64, device=device)
            for name in ("history_mean", "history_scale", "coefficients", "residuals")) for c in contexts)
        result = object.__new__(cls)
        for name, value in (
            ("episode_component", MappingProxyType(episode_components)), ("context_pools", context_pools),
            ("context_episodes", context_episodes), ("context_tensors", context_tensors),
            ("context_positions", context_positions), ("quadratic_score_map", quadratic_map),
            ("bank", bank), ("train_components", train), ("neighbor_pool", pool),
            ("q_law", q_law), ("innovation_contexts", contexts),
            ("score_profile", score_profile),
            ("_query_candidates", local()),
            ("fit_device", str(device) if device is not None else None),
            ("field_mean", _immutable(mean) if mean is not None else None),
            ("field_scale", _immutable(scale) if scale is not None else None),
            ("normalized_histories", _immutable(normalized) if normalized is not None else None),
            ("bank_source_refs", tuple(d.source_refs for d in bank)),
        ):
            object.__setattr__(result, name, value)
        return result

    @property
    def law_manifest(self):
        if self.score_profile is not None:
            p = self.score_profile
            manifest = {"q_law": self.q_law, "fit_device": self.fit_device,
                "neighbor_pool": self.neighbor_pool,
                "population": "full eligible empirical pool" if self.neighbor_pool is None else "nearest standardized-prefix empirical pool",
                "transforms": tuple(f.transform for f in p.fields),
                "injected_deterministic_prior": {"source_ref": p.source_ref,
                    "producer_version": p.producer_version,
                    "fields": tuple({"artifact": f.artifact, "field": f.field, "unit": f.unit,
                        "owner_roles": f.owner_roles, "lower": f.lower, "upper": f.upper,
                        "transform": f.transform} for f in p.fields),
                    "nominal_latency_ms": p.nominal_latency_ms, "worst_case_latency_ms": p.worst_case_latency_ms,
                    "maximum_packet_loss_ratio": p.maximum_packet_loss_ratio,
                    "decode": "Python round6(N+(W-N)*u), Python round6(M*u*u); closed u in [0,1]"},
                "fit_assumptions": ("linear conditional mean from score H8", "whole10D TRAIN residual bootstrap", "reflection onto closed [0,1]"),
                "ridge_nonintercept_lambda": INNOVATION_RIDGE_LAMBDA,
                "ridge_objective": "mean sum-of10 squared residuals + lambda sum-ofnonintercept squared coefficients",
                "history": "u anchor + seven chronological du",
                "future": "decode(1-abs(remainder(query u anchor + fitted10D mean + whole10D TRAIN residual,2)-1)); exact raw query anchor",
                "fit_contexts": "allTRAIN for held-out components; exclude ownTRAIN component from stats/coefs/residuals",
                "draw_design": "minimum lexicographic full-extrema atom cover + remainder; reserve1 each; integer largest-remainder surplus",
                "draw_prior": "stratum size/(eligible population size * stratum draw count)"}
            if self.q_law == "generator_trend_empirical_innovation":
                manifest.update({"population": "128 nearest eligible TRAIN histories by standardized score H8",
                    "fit_assumptions": ("last chronological score difference continues linearly",
                        "whole10D nearest-TRAIN empirical residual bootstrap", "reflection onto closed [0,1]"),
                    "injected_linear_prior": "u_anchor + k*(u_anchor-u_previous), k=1..10",
                    "ridge_nonintercept_lambda": None, "ridge_objective": None,
                    "future": "decode(reflect(query trend prior + whole10D TRAIN actual-minus-donor-trend residual)); exact raw query anchor",
                    "fit_contexts": "allTRAIN for held-out; exclude ownTRAIN component from normalization/neighbors/residuals",
                    "draw_design": "ordinary independent draws with replacement from 128 nearest TRAIN residual trajectories",
                    "draw_prior": "1/K for ordinary draws"})
            if self.q_law == "generator_quadratic_empirical_innovation":
                manifest.update({"population": "128 nearest eligible TRAIN histories by standardized score H8",
                    "fit_assumptions": ("degree2 least-squares polynomial fitted to last4 scores extrapolates",
                        "whole10D nearest-TRAIN empirical residual bootstrap", "reflection onto closed [0,1]"),
                    "injected_quadratic_prior": {"degree": 2, "past_times": (-3, -2, -1, 0),
                        "future_times": tuple(range(1, 11)), "fit": "unweighted least squares on four observed scores",
                        "interpretation": "provisional data-fitted state trend; not a physical-law claim"},
                    "ridge_nonintercept_lambda": None, "ridge_objective": None,
                    "future": "decode(reflect(query last4 quadratic LS prior + whole10D TRAIN actual-minus-donor-quadratic residual)); exact raw query anchor",
                    "fit_contexts": "allTRAIN for held-out; exclude ownTRAIN component from normalization/neighbors/residuals",
                    "draw_design": "ordinary independent draws with replacement from 128 nearest TRAIN residual trajectories",
                    "draw_prior": "1/K for ordinary draws"})
            return MappingProxyType(manifest)
        return MappingProxyType({"q_law": self.q_law, "fit_device": self.fit_device,
            "neighbor_pool": self.neighbor_pool,
            "population": "full eligible empirical pool" if self.neighbor_pool is None else "nearest standardized-prefix empirical pool",
            "transforms": tuple(f.transform for f in self.bank[0].fields),
            "ridge_nonintercept_lambda": INNOVATION_RIDGE_LAMBDA if self.q_law in ("conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation") else None,
            "ridge_objective": "mean sum-of20 squared residuals + lambda sum-ofnonintercept squared coefficients" if self.q_law in ("conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation") else None,
            "history": "transformed anchor2 + seven chronological2D differences" if self.q_law in ("conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation") else "raw eight-row history",
            "future": "inverse(query transformed anchor + fitted20D mean + whole20D TRAIN residual)" if self.q_law in ("conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation") else "absolute complete TRAIN donor future with exact query anchor",
            "fit_contexts": "allTRAIN for held-out components; exclude ownTRAIN component from stats/coefs/residuals" if self.q_law in ("conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation") else "TRAIN prefix normalization",
            "draw_design": "minimum lexicographic full-extrema atom cover + remainder; reserve1 each; integer largest-remainder surplus",
            "draw_prior": "stratum size/(eligible population size * stratum draw count)"})

    def _check_prefix(self, prefix):
        if not isinstance(prefix, NumericSequence) or len(prefix.ticks) != 8:
            raise ValueError("Query must be an eight-point NumericSequence")
        if _schema(prefix) != _schema(self.bank[0]):
            raise ValueError("Query field order or ROLE+CLASS schema differs")
        if self.episode_component.get(prefix.episode, prefix.component) != prefix.component:
            raise ValueError("Query/source episode component assignment drift")

    def _context_index(self, prefix):
        return 1+self.train_components.index(prefix.component) if prefix.component in self.train_components else 0

    def context(self, prefix):
        """Return immutable TRAIN regression context for this query."""
        self._check_prefix(prefix)
        if self.q_law not in ("conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation"):
            raise ValueError("Absolute law has no conditional regression context")
        index = self._context_index(prefix)
        if prefix.episode in self.context_episodes[index]:
            raise ValueError("Query episode cannot enter conditional fit context")
        return self.innovation_contexts[index]

    def _innovation_query(self, prefix, index):
        mean, scale, coefficients, _ = self.context_tensors[index]
        with torch.no_grad():
            if self.score_profile is None:
                z = _transform_values(prefix.values, prefix.fields, self.fit_device)
                h = _history16(z)
            else:
                z = torch.tensor(self.score_profile.encode(prefix.values).copy(), dtype=torch.float64, device=self.fit_device)
                h = torch.cat((z[-1:], torch.diff(z)))
            normalized = (h-mean)/scale
            design = torch.cat((torch.ones(1, dtype=torch.float64, device=self.fit_device), normalized))
            if self.q_law == "generator_quadratic_empirical_innovation":
                prediction = self.quadratic_score_map @ z[-4:]
            elif self.q_law == "generator_trend_empirical_innovation":
                prediction = (z[-1]-z[-2])*torch.arange(1, 11, dtype=torch.float64, device=self.fit_device)
            else:
                prediction = design@coefficients
            if not bool(torch.isfinite(normalized).all() & torch.isfinite(prediction).all()):
                raise FloatingPointError("Nonfinite conditional query history or20D mean")
        return z[-1], normalized, prediction

    def _query_state(self, prefix):
        """One thread-local query transform shared by pool, reconstruction and fold diagnostics."""
        cached = getattr(self._query_candidates, "query", None)
        if cached is not None and cached[0] is prefix:
            return cached[1:]
        self.context(prefix)
        index = self._context_index(prefix)
        state = (index, *self._innovation_query(prefix, index))
        self._query_candidates.query = (prefix, *state)
        return state

    def _pool(self, prefix):
        self._check_prefix(prefix)
        if self.q_law in ("conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation"):
            self.context(prefix)
            index = self._context_index(prefix)
            if self.neighbor_pool is None:
                return self.context_pools[index]
            _, _, normalized, _ = self._query_state(prefix)
            with torch.no_grad():
                histories = torch.tensor(self.innovation_contexts[index].normalized_histories.copy(),
                                         dtype=torch.float64, device=self.fit_device)
                distances = (histories-normalized).square().sum(1)
                if not bool(torch.isfinite(distances).all()):
                    raise FloatingPointError("Nonfinite conditional residual-neighborhood distance")
                order = torch.argsort(distances, stable=True)[:self.neighbor_pool].cpu().tolist()
            return tuple(self.bank[self.innovation_contexts[index].bank_indices[i]] for i in order)
        eligible = np.array([i for i, d in enumerate(self.bank)
            if d.component != prefix.component and d.episode != prefix.episode], dtype=np.int64)
        if not len(eligible):
            raise ValueError("No eligible donors after component/episode exclusion")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            query = (prefix.values - self.field_mean) / self.field_scale
            distances = np.square(self.normalized_histories[eligible] - query).sum(axis=(1, 2))
        if not np.isfinite(distances).all():
            raise ValueError("Nonfinite history distances")
        order = np.argsort(distances, kind="stable")[:self.neighbor_pool]
        return tuple(self.bank[int(i)] for i in eligible[order])

    def _residual_rows(self, index, donors, rows=None):
        """Device residuals for donors; the full context pool is already in row order."""
        residuals = self.context_tensors[index][3]
        if donors is self.context_pools[index]:
            return residuals
        if rows is not None and self.neighbor_pool is None:
            return residuals[list(rows)]
        positions = self.context_positions[index]
        return residuals[[positions[_identity(d)] for d in donors]]

    def _reconstruct_donors(self, prefix, donors):
        if self.q_law == "absolute_train_donor":
            values = np.stack([np.concatenate((prefix.values[-1:], d.values[8:]), axis=0) for d in donors])
        else:
            index, anchor, _, prediction = self._query_state(prefix)
            with torch.no_grad():
                residuals = self._residual_rows(index, donors)
                if self.score_profile is None:
                    future = _inverse_values(anchor + (prediction + residuals).reshape(-1, 10, 2)).cpu().numpy()
                else:
                    raw = (prediction+residuals if self.q_law == "generator_quadratic_empirical_innovation"
                           else anchor+prediction+residuals)
                    folded = 1-torch.abs(torch.remainder(raw, 2)-1)
                    future = self.score_profile.decode(folded.cpu().numpy())
            values = np.concatenate((np.broadcast_to(prefix.values[-1:], (len(donors), 1, 2)), future), 1)
        return _values(np.asarray(values, dtype=np.float64), (len(donors), 11, len(prefix.fields)), prefix.fields)

    def _fold_diagnostics(self, prefix, donors, rows=None):
        if self.score_profile is None:
            return ()
        index, anchor, _, prediction = self._query_state(prefix)
        with torch.no_grad():
            residuals = self._residual_rows(index, donors, rows)
            raw = (prediction+residuals if self.q_law == "generator_quadratic_empirical_innovation"
                   else anchor+prediction+residuals).cpu().numpy()
        return tuple((int(np.count_nonzero((row < 0) | (row > 1))), float(row.min()), float(row.max())) for row in raw)

    def _candidate_population(self, prefix):
        """One query's complete candidate population, computed once per thread-local prefix."""
        cached = getattr(self._query_candidates, "entry", None)
        if cached is not None and cached[0] is prefix:
            return cached[1], cached[2]
        pool = self._pool(prefix)
        values = self._reconstruct_donors(prefix, pool)
        self._query_candidates.entry = (prefix, pool, values)
        return pool, values

    def candidate_paths(self, prefix):
        """Immutable complete eligible paths in the same order as _pool(prefix)."""
        return self._candidate_population(prefix)[1]

    def reconstruct(self, prefix, donor_ids):
        """Rederive selected latent values from exact query prefix and fitted TRAIN law."""
        pool = {_identity(d): d for d in self._pool(prefix)}
        identities = tuple(donor_ids)
        if not identities or any(identity not in pool for identity in identities):
            raise ValueError("Reconstruction donor is outside the eligible empirical pool")
        return self._reconstruct_donors(prefix, tuple(pool[identity] for identity in identities))

    def sample(self, prefix, K, seed, *, outcome_evaluator, draw="stratified"):
        """Draw weighted strata from actual hard candidate outcomes, without labels."""
        K = _integer(K, 1, 16)
        seed = _integer(seed, 0)
        pool, candidates = self._candidate_population(prefix)
        if not callable(outcome_evaluator):
            raise TypeError("Sampling requires an actual hard-rule outcome callback")
        outcomes = outcome_evaluator(candidates)
        if not isinstance(outcomes, np.ndarray) or outcomes.shape != (len(pool), 2):
            raise ValueError("Outcome callback must cover EVERY eligible candidate and both predicates")
        if draw not in ("stratified", "ordinary"):
            raise ValueError("Unknown draw design")
        identity = (prefix.episode, prefix.component, prefix.owners, prefix.ticks)
        if draw == "ordinary":  # i.i.d. uniform draws from the fitted law's eligible pool, no outcome information
            design = OrdinaryDesign.whole_pool(len(pool), K)
            rng = np.random.default_rng(np.random.SeedSequence([seed, *map(ord, "ordinary"), *map(ord, repr(identity))]))
            selected, strata = rng.integers(len(pool), size=K).tolist(), [0]*K
        else:
            design = extrema_sampling_design(outcomes, K)
            rng = np.random.default_rng(np.random.SeedSequence([seed, *map(ord, repr(identity))]))
            selected, strata = [], []
            for stratum, (members, count) in enumerate(zip(design.members, design.counts)):
                selected.extend(int(members[int(i)]) for i in rng.integers(len(members), size=count))
                strata.extend([stratum]*count)
        chosen = tuple(pool[i] for i in selected)
        values = candidates[selected]
        weights = np.array([len(design.members[i])/(len(pool)*design.counts[i]) for i in strata], dtype=np.float64)
        context = self.context(prefix) if self.q_law in ("conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation") else None
        fit_refs = context.source_refs if context is not None else self.bank_source_refs
        return NumericBranches(values, tuple(prefix.ticks[-1] + 5 * i for i in range(11)),
            prefix.fields, prefix.owners, prefix.episode, prefix.component,
            tuple(_identity(d) for d in chosen), tuple(d.component for d in chosen),
            tuple(d.source_refs[8:] for d in chosen), self.bank_source_refs,
            weights, self.q_law, prefix.source_refs, fit_refs, outcomes[selected], tuple(strata), design,
            context.excluded_component if context is not None else None,
            self.law_manifest["future"] + "; " + self.law_manifest["population"],
            self.score_profile, self._fold_diagnostics(prefix, chosen, selected))

    def validate(self, prefix, branches, *, outcome_evaluator):
        """Verify carrier, exact reconstruction, fit lineage and hard-rule sampling design."""
        if not isinstance(branches, NumericBranches):
            raise TypeError("Validation requires NumericBranches")
        if (branches.query_episode, branches.query_component, branches.query_owners, branches.fields,
                branches.query_source_refs, branches.bank_source_refs, branches.q_law) != (
                prefix.episode, prefix.component, prefix.owners, prefix.fields,
                prefix.source_refs, self.bank_source_refs, self.q_law):
            raise ValueError("Sampled carrier identity/schema/law/source drift")
        if branches.ticks != tuple(prefix.ticks[-1]+5*i for i in range(11)):
            raise ValueError("Sampled carrier time grid drift")
        if not np.array_equal(branches.values, self.reconstruct(prefix, branches.donor_ids)):
            raise ValueError("Sampled values disagree with exact fitted-law reconstruction")
        pool, candidates = self._candidate_population(prefix)
        ids = tuple(_identity(d) for d in pool)
        outcomes = outcome_evaluator(candidates)
        if not isinstance(outcomes, np.ndarray) or outcomes.shape != (len(pool), 2):
            raise ValueError("Validation outcome callback must cover every candidate")
        design = (OrdinaryDesign.whole_pool(len(pool), len(branches.donor_ids))
                  if isinstance(branches.sampling_design, OrdinaryDesign)
                  else extrema_sampling_design(outcomes, len(branches.donor_ids)))
        if branches.sampling_design != design:
            raise ValueError("Sampled design differs from current hard-rule outcomes")
        selected = tuple(ids.index(identity) for identity in branches.donor_ids)
        if not np.array_equal(branches.outcome_classes, outcomes[list(selected)]):
            raise ValueError("Sampled hard outcomes drift")
        for k, (index, stratum) in enumerate(zip(selected, branches.branch_strata)):
            if index not in design.members[stratum] or branches.donor_refs[k] != pool[index].source_refs[8:]:
                raise ValueError("Sampled donor/stratum/source drift")
        context = self.context(prefix) if self.q_law in ("conditional_joint_innovation", "generator_score_innovation", "generator_trend_empirical_innovation", "generator_quadratic_empirical_innovation") else None
        if (branches.fit_source_refs != (context.source_refs if context is not None else self.bank_source_refs)
                or branches.fit_context_excluded_component != (context.excluded_component if context is not None else None)
                or branches.law_description != self.law_manifest["future"] + "; " + self.law_manifest["population"]):
            raise ValueError("Sampled derived fit provenance drift")
        if (branches.score_profile != self.score_profile
                or branches.folded_proposal_diagnostics != self._fold_diagnostics(prefix, tuple(pool[i] for i in selected), selected)):
            raise ValueError("Generator profile or folded proposal provenance drift")
        return True

    def conditional_mean(self, prefix):
        """Return immutable [10,F] mean over the selected neighbor pool."""
        with np.errstate(over="raise", invalid="raise"):
            mean = self.candidate_paths(prefix)[:, 1:].mean(axis=0)
        if not np.isfinite(mean).all():
            raise ValueError("Nonfinite conditional mean")
        return _immutable(mean)
