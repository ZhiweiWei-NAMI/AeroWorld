"""Generic noisy-state trajectory oracle over declared episode contracts.

This module turns an assembled state sampler output into fixed-rule trajectory
supervision for arbitrary F fields, P predicates and K candidates. It consumes
only declared generic data: EpisodeContract, CandidateTrajectory, exact
source-index arrays, explicit per-field noise/domain/transform declarations and
explicit supervision declarations. No RULES tuple, slice names, field count,
predicate count, candidate count or TTL constant exists here; every physical
domain, noise law, transform and censoring semantic enters through an adapter.
An assembled state graph of any slice remains an assembly artifact; the dense
shared trajectory forward is the neural backbone.

Truth codes are the existing four-valued domain. UNKNOWN stays UNKNOWN and is
never read as a false observation. Compatibility keeps the declared censoring:
after the first known value opposite to a known anchor the later cumulative outcome is
known to have flipped, so later event bins become incompatible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .predicate_trajectory_data import (
    EpisodeContract, CandidateTrajectory, FuturePredicateEvaluator,
    STATE_OFFSETS, TRUTH_CODES, assemble_trajectory_batch,
)
from .unified_contracts import ContractError

ANCHOR_STEPS = len(STATE_OFFSETS)
FUTURE_OFFSETS = STATE_OFFSETS[1:]
FUTURE_STEPS = len(FUTURE_OFFSETS)
OUTCOME_BINS = ANCHOR_STEPS  # ten timed flip bins plus the tail beyond prediction horizon T>50, may be infinite
TIMED_BINS = OUTCOME_BINS - 1
# Declared types sampled as raw physical cells.  Numeric gaussian fields keep
# the float fast path; every other declared type passes through sampling,
# CandidateTrajectory and the fixed rules without any float coercion.  The
# assembled typed projection encodes them for the model afterwards.
TYPED_DATA_TYPES = ("integer", "boolean", "categorical", "identifier", "four_value",
                    "vector", "vector3")


def _frozen_array(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class StateFieldNoise:
    """Declared per-field noise law for one raw physical state cell.

    The law identity is exact: canonical field, declared authority/path,
    type and unit, plus the typed owners/roles it applies to. A law without
    owner_entity_ids resolves only when the episode declares that canonical
    field once; repeats require either per-owner pinning or the explicit
    shared=True law, which maps one declared law onto the fields' independent
    value slots (it never merges their topology or values).

    method "gaussian" perturbs known continuous numeric cells by the declared
    sigma in physical units times standard normal draws; only states
    explicitly declared perturbable carry Gaussian noise, the anchor row is
    never perturbed. Gaussian integer fields must declare their exact rounding
    law through the sampler's field_laws callback; there is no silent scalar
    coercion, and an identifier that merely looks numeric stays identity.
    "identity" retains the raw cell exactly through the fixed rules,
    "discrete_law" resamples the declared discrete code through the caller's
    callable, and "component_law" perturbs through explicit per-component
    scales via the caller's callable, preserving a declared ENU vector as
    components rather than a norm; boolean, category, four-value and vector
    laws stay explicit declarations. The declared projection is an explicit
    ascending physical interval per numeric field or empty for an
    adapter-supplied per-row bound; no universal bounds are imposed on
    discrete codes or signed vectors.
    """
    field: str
    data_type: str
    unit: str
    method: str = "gaussian"
    sigma: tuple[float, ...] = ()
    projection: tuple[tuple[float, float], ...] = ()
    source_ref: str = ""
    owner_entity_ids: tuple[str, ...] = ()
    authority: str = ""
    shared: bool = False

    def __post_init__(self):
        if not self.field or self.data_type not in (
                "numeric", "integer", "boolean", "categorical", "identifier", "four_value",
                "vector", "vector3"):
            raise ContractError(f"noise declaration needs an explicit field and data type: {self.field!r}")
        if self.method not in ("gaussian", "identity", "discrete_law", "component_law"):
            raise ContractError(f"undeclared noise method for {self.field}: {self.method}")
        if self.method == "gaussian":
            if self.data_type not in ("numeric", "integer"):
                raise ContractError(f"gaussian noise needs a numeric or integer field: {self.field}")
            if len(self.sigma) != 1 or not np.isfinite(self.sigma).all() or self.sigma[0] <= 0:
                raise ContractError(f"gaussian noise needs one positive declared sigma: {self.field}")
        if self.method == "identity":
            if self.sigma:
                raise ContractError(f"identity law declares no sigma: {self.field}")
        if self.method == "component_law":
            if self.data_type not in ("vector", "vector3"):
                raise ContractError(
                    f"component law needs a declared vector type, not {self.data_type}: {self.field}")
            if len(self.sigma) < 1:
                raise ContractError(f"component law needs declared per-component scales: {self.field}")
        if self.method == "discrete_law" and self.data_type in ("vector", "vector3"):
            raise ContractError(f"discrete law does not apply to declared vectors: {self.field}")
        if self.data_type == "identifier" and self.method != "identity":
            raise ContractError(f"identifier fields keep the identity law: {self.field}")
        if self.projection and self.data_type != "numeric":
            raise ContractError(
                f"declared projection applies to numeric fields only: {self.field}")
        if self.projection and (len(self.projection) != 1 or self.projection[0][0] > self.projection[0][1]):
            raise ContractError(f"declared projection needs one ascending interval: {self.field}")
        if self.source_ref and not isinstance(self.source_ref, str):
            raise ContractError("noise declaration source must be a string")
        if self.shared and (self.owner_entity_ids or self.authority):
            raise ContractError(
                f"shared law must not also declare owners or authority: {self.field}")
        if self.owner_entity_ids and any(not item for item in self.owner_entity_ids):
            raise ContractError(f"owner entity ids must be explicit nonempty strings: {self.field}")

    @property
    def typed(self):
        """True when the field keeps raw physical cells instead of the float table."""
        return self.data_type in TYPED_DATA_TYPES

    def sigma_for(self):
        return np.asarray(self.sigma, dtype=float)


@dataclass(frozen=True)
class FieldTransform:
    """Declared per-field numeric transform for neural inputs.

    The transform is (value - mean) / scale on known physical cells and 0 on
    unknown cells. A caller choosing the identity transform supplies mean 0 /
    scale 1; the oracle invents no default. owner_entity_ids pins the transform
    to one exact typed-owner occurrence of the field (each occurrence keeps its
    independent value slot); a transform without them applies to every
    occurrence of that field name (an explicit shared law mapped onto the
    distinct slots, never a merge of their topology).
    """
    field: str
    mean: float = 0.0
    scale: float = 1.0
    source_ref: str = ""
    owner_entity_ids: tuple[str, ...] = ()
    shared: bool = False

    def __post_init__(self):
        if not self.field:
            raise ContractError("field transform needs an explicit field")
        if not np.isfinite([self.mean, self.scale]).all() or self.scale <= 0:
            raise ContractError(f"field transform needs finite mean and positive scale: {self.field}")
        if self.shared and self.owner_entity_ids:
            raise ContractError(
                f"shared transform must not also declare owners: {self.field}")
        if self.owner_entity_ids and any(not item for item in self.owner_entity_ids):
            raise ContractError(f"owner entity ids must be explicit nonempty strings: {self.field}")


@dataclass(frozen=True)
class SupervisionDeclaration:
    """Explicit supervision authority for one generic trajectory cohort.

    reference_declaration names the evidence the oracle mean reads (for a
    source-anchored noisy-future law, the observed source operand table). It is
    never inferred from a predicate identifier. fit_episode_ids is the explicit
    TRAIN-fitting population of the declared transform, frozen before any
    validation use.
    """
    name: str
    reference_declaration: str
    fit_episode_ids: tuple[str, ...]
    source_ref: str

    def __post_init__(self):
        if not self.name or not self.reference_declaration or not self.source_ref:
            raise ContractError("supervision declaration needs name, reference and source")
        if not self.fit_episode_ids or any(not episode for episode in self.fit_episode_ids):
            raise ContractError("supervision declaration needs explicit fitting episodes")

    def as_dict(self):
        return {"name": self.name, "reference_declaration": self.reference_declaration,
                "fit_episode_ids": tuple(self.fit_episode_ids), "source_ref": self.source_ref}


class OracleNoiseLawDeclaration:
    """Named declared noise law; the caller's protocol records this object."""
    def __init__(self, name, law, *, sigma_fraction=None, reference=""):
        if not name or not law or not isinstance(law, str):
            raise ContractError("noise law declaration needs a name and a law string")
        self.name = name
        self.law = law
        self.sigma_fraction = None if sigma_fraction is None else float(sigma_fraction)
        self.reference = reference

    def as_dict(self):
        return {"name": self.name, "law": self.law,
                "sigma_fraction_of_declared_scale": self.sigma_fraction,
                "reference": self.reference}


def _field_identity(item):
    """Exact (field, data_type, unit) identity of one declared field."""
    schema = getattr(item, "schema", item)
    return (schema.field, schema.data_type, schema.unit)


def _coordinate_identity(item):
    """Exact typed-owner coordinate identity of one episode field.

    FieldCoordinate carries the full law identity: canonical field, declared
    type, unit, the exact typed owner (entity id and ontology classes) of
    every declared role, and the declared candidate authority/path. A plain
    FieldSchema has no owner or authority and matches by field/type/unit
    alone, which the episode-level checks below then reject for repeats.
    """
    schema = getattr(item, "schema", item)
    owners = tuple((entity.entity_id, entity.ontology_classes)
                   for entity in getattr(item, "owner", ()) or ())
    return (schema.field, schema.data_type, schema.unit, owners,
            getattr(item, "authority", ""))


def _declaration_key(item):
    """Declared scope of one law/transform entry.

    owner_entity_ids nonempty pins the entry to that exact typed-owner
    occurrence of the field (identity includes the entity ids; episode checks
    also confirm the declared ontology classes and authority match the
    coordinate). owner_entity_ids empty with shared=True applies the same
    declared law to every occurrence of that field name, mapped onto their
    independent value slots. An entry with neither is owner-scoped only when
    the episode never repeats the field; repetition without an explicit
    declaration is a collision error.
    """
    owners = getattr(item, "owner_entity_ids", ()) or ()
    if owners:
        return (item.field, tuple(owners))
    if getattr(item, "shared", False):
        return (item.field, None)
    return (item.field, ())


def declared_field_transforms(noise, transforms):
    """Order per-declared-field transforms, validating the exact field set.

    transforms is a mapping from field name to (mean, scale), or a sequence of
    FieldTransform or (field, mean, scale) entries. The returned tuple follows
    the declared noise order.
    """
    by_field = {}
    items = transforms.items() if isinstance(transforms, Mapping) else transforms
    for item in items:
        if isinstance(item, FieldTransform):
            by_field[item.field] = item
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            field, (mean, scale) = item
            by_field[field] = FieldTransform(field, float(mean), float(scale))
        else:
            field, mean, scale = item
            by_field[field] = FieldTransform(field, float(mean), float(scale))
    names = [item.field for item in noise]
    missing = [field for field in names if field not in by_field]
    extra = [field for field in by_field if field not in set(names)]
    if missing or extra:
        raise ContractError(f"declared transform/field mismatch: missing={missing} extra={extra}")
    return tuple(by_field[field] for field in names)


def check_field_declarations(fields, noise, transforms=None):
    """Validate one declared field axis against its noise/transform adapters.

    fields are exact FieldCoordinate tuples or FieldSchema values. Every exact
    typed-owner coordinate keeps its own value slot and resolves to exactly one
    noise declaration and, when transforms are supplied, one transform. The
    law identity is exact: canonical field, declared authority, type and unit,
    plus the coordinate's typed owners/roles. A field name repeated on
    different typed owners is a collision unless every entry for that name
    either pins its occurrence through owner_entity_ids (validated against the
    coordinate's declared owner identities and authority) or declares the
    explicit shared law (shared=True) mapped onto the independent slots;
    partial coverage, ambiguous repeats and guessed name matches are errors.
    The contract declares no epoch, generation or version fields and none are
    consulted. Returns the noise declarations and ordered transforms aligned
    with the declared field order.
    """
    declared = [_coordinate_identity(item) for item in fields]
    scalar_keys = [_field_identity(item) for item in fields]
    noise = check_noise_declarations(noise)
    # An owner-id pin names exactly one occurrence: if several coordinates
    # share the id tuple but differ in declared ontology class or authority,
    # the pin is ambiguous and must be replaced by full-identity pinning or
    # an explicit shared law.
    id_pin_uses = {}
    for identity in declared:
        field, owners = identity[0], identity[3]
        if owners:
            id_pin_uses.setdefault((field, tuple(item[0] for item in owners)), set()).add(identity)
    ambiguous_pins = {key: len(identities) for key, identities in id_pin_uses.items()
                      if len(identities) > 1}
    by_key = {}
    for item in noise:
        key = _declaration_key(item)
        if key in by_key:
            raise ContractError(
                f"noise declaration declares {item.field} twice for the same scope")
        by_key[key] = item
    resolved, missing, collisions, identity_mismatches = [], [], [], []
    for identity, scalar in zip(declared, scalar_keys):
        field, owners, authority = identity[0], identity[3], identity[4]

        def exact_identity(item, _identity=identity, _authority=authority):
            """A resolved law must name the slot's exact canonical identity.

            Field name, data type, unit and (when the law declares one) the
            authority must equal the coordinate's declared identity after the
            resolution branch; a numeric/ms law never covers a boolean/bool
            slot and a mismatched authority is never silently accepted.
            """
            return (item.field == _identity[0] and item.data_type == _identity[1]
                    and item.unit == _identity[2]
                    and (not item.authority or item.authority == _authority))

        pinned = by_key.get((field, tuple(item[0] for item in owners)) if owners else (field, ()))
        shared = by_key.get((field, None))
        unpinned = by_key.get((field, ()))
        if pinned is not None and owners:
            if ambiguous_pins.get((field, tuple(item[0] for item in owners))):
                collisions.append(field)
                continue
            if not exact_identity(pinned):
                identity_mismatches.append(field)
                continue
            if shared is not None or unpinned is not None:
                collisions.append(field)
                continue
            resolved.append(pinned)
        elif shared is not None and unpinned is None:
            if not exact_identity(shared):
                identity_mismatches.append(field)
                continue
            resolved.append(shared)
        elif unpinned is not None:
            if not exact_identity(unpinned):
                identity_mismatches.append(field)
                continue
            repeats = scalar_keys.count(scalar)
            same_coordinate = sum(1 for item in fields
                                  if _coordinate_identity(item) == identity)
            if repeats == 1 or same_coordinate == repeats:
                resolved.append(unpinned)
            else:
                collisions.append(field)
        else:
            missing.append(field)
    if identity_mismatches:
        raise ContractError(
            "resolved law does not match the exact declared identity of "
            f"{sorted(set(identity_mismatches))}: canonical field, data type, "
            "unit and declared authority must match after every resolution branch")
    if collisions:
        raise ContractError(
            "field declarations collide on repeated fields without an explicit "
            f"shared law or exact owner pinning: {sorted(set(collisions))}")
    if missing:
        raise ContractError(f"noise declarations differ from the exact declared field axis: "
                            f"missing={sorted(set(missing))}")
    if transforms is None:
        return tuple(resolved), ()
    ordered_transforms = []
    transform_items = transforms.items() if isinstance(transforms, Mapping) else transforms
    if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
           and not isinstance(item[1], FieldTransform) for item in transform_items) \
            and len({item[0] for item in transform_items}) == len(transform_items) \
            and len(scalar_keys) == len(set(scalar_keys)):
        # Name-keyed transforms keep the released single-owner L/P interface.
        ordered_transforms = declared_field_transforms(resolved, transforms)
        return tuple(resolved), ordered_transforms
    by_transform_key = {}
    for item in transform_items:
        if not isinstance(item, FieldTransform):
            field, mean, scale = item
            item = FieldTransform(field, float(mean), float(scale))
        key = _declaration_key(item)
        if key in by_transform_key:
            raise ContractError(
                f"transform declares {item.field} twice for the same scope")
        by_transform_key[key] = item
    resolved_transforms, missing_t, collisions_t = [], [], []
    for identity in declared:
        field, owners = identity[0], identity[3]
        pinned = by_transform_key.get((field, tuple(item[0] for item in owners)) if owners else (field, ()))
        shared = by_transform_key.get((field, None))
        unpinned = by_transform_key.get((field, ()))
        if pinned is not None:
            if shared is not None or unpinned is not None:
                collisions_t.append(field)
                continue
            resolved_transforms.append(pinned)
        elif shared is not None and unpinned is None:
            resolved_transforms.append(shared)
        elif unpinned is not None:
            repeats = sum(1 for item in declared if item[0] == field and item[3] == owners)
            if repeats == 1 or repeats == sum(1 for item in declared
                                              if item[:4] == identity[:4]):
                resolved_transforms.append(unpinned)
            else:
                collisions_t.append(field)
        else:
            missing_t.append(field)
    if collisions_t:
        raise ContractError(
            "transforms collide on repeated fields without an explicit shared "
            f"law or exact owner pinning: {sorted(set(collisions_t))}")
    if missing_t:
        raise ContractError(f"declared transform/field mismatch: missing={sorted(set(missing_t))}")
    return tuple(resolved), tuple(resolved_transforms)


def oracle_seed_default(seed, query):
    """Existing declared K-stream formula: one stream per seed and query."""
    return int(seed) * 1_000_003 + int(query)


class PhysicalOperandProjector:
    """Declared physical projection of noisy operands, once per cohort.

    A field with a declared projection interval clips to that fixed physical
    interval; a field without one clips to [0, upper] with the caller's
    per-row, per-field upper domain bound (for example an adapter-declared
    query-domain bound). Exact endpoints, no epsilon. Pre-projection violations
    and endpoint mass are counted per drawn sample over the known mask, so all
    fractions share one denominator and stay comparable across K.
    """

    def __init__(self, noise):
        """Compile from a declaration tuple or a compiled law-to-slot mapping.

        Two scopes may share one field name exactly when they pin different
        typed owners or repeat one identical explicit shared law object; two
        separately written unscoped duplicates are a rejected collision.
        """
        self.noise = tuple(noise)
        if not self.noise:
            raise ContractError("operand projector needs at least one declared field")
        keys = [(_declaration_key(item), item) for item in self.noise]
        if len(set(key for key, _ in keys)) != len(keys) and not _shared_law_alignment(keys):
            raise ContractError("declared noise identity contains duplicate scopes")
        for item in self.noise:
            if item.projection:
                low, high = item.projection[0]
                if not (np.isfinite(low) and np.isfinite(high) and low <= high):
                    raise ContractError(f"invalid declared projection for {item.field}")

    def bounds(self, upper_by_row):
        """Validate declared [N,F] or single-row [F] upper numeric bounds.

        Only numeric columns with no declared interval consult upper_by_row.
        Typed columns (integer laws, booleans, category/identifier codes,
        four-value codes and declared vectors) project only through their own
        declared interval or callback, never through a universal bound.
        """
        numeric = [not item.typed for item in self.noise]
        upper = np.asarray(upper_by_row, dtype=float)
        if upper_by_row is None:
            upper = np.zeros((0, 0))
        elif upper.ndim not in (1, 2) or upper.shape[-1] != len(self.noise):
            raise ContractError("declared upper domain bounds must be [N,F] or [F]")
        elif not np.isfinite(upper).all() or (upper < 0).any():
            raise ContractError("declared upper domain bounds must be finite and nonnegative")
        low = np.array([item.projection[0][0] if item.projection else 0. for item in self.noise])
        high = np.array([item.projection[0][1] if item.projection else np.nan
                         for item in self.noise])
        return low, high, upper, numeric

    def project(self, values, known, upper_by_row=None):
        """Project one query's operands; return the projected table and counts.

        values is [K,11,F] (or [N,11,F]/[N,K,11,F]) and may hold object cells
        for typed fields; no float coercion happens here. A numeric field with
        a declared interval clips to it for every row; a numeric field without
        one clips to [0, upper] with the declared per-row numeric bound. Typed
        fields pass through untouched unless they declare their own interval.
        Cells outside any declared projection are counted as pre-projection
        violations; counts cover known numeric projection cells only.
        """
        known = np.asarray(known, dtype=bool)
        if values.ndim not in (3, 4) or values.shape != known.shape:
            raise ContractError("projected operands need an aligned [N,11,F] or [N,K,11,F] table")
        rows, fields = values.shape[0], values.shape[-1]
        if fields != len(self.noise):
            raise ContractError("projected operand width differs from declared fields")
        low, fixed_high, upper, numeric = self.bounds(upper_by_row)
        if upper.ndim == 1:
            if upper.shape != (fields,):
                raise ContractError("declared upper domain bounds must be [N,F] or [F]")
            upper = upper.reshape(1, fields)
        elif upper.ndim == 2 and upper.shape[0] == 0:
            pass  # no per-row numeric bounds were supplied
        elif upper.ndim == 2 and upper.shape != (rows, fields):
            raise ContractError("declared upper domain bounds must be [N,F]")
        projected = np.array(values, copy=True)
        counts = {}
        for name in ("cells", "below", "above", "endpoint_low", "endpoint_high"):
            counts[name] = 0
            counts[name + "_by_field"] = [0] * fields
        for column, item in enumerate(self.noise):
            if not numeric[column]:
                # Raw typed cells: identity, discrete law or declared callback
                # output passes through without universal numeric bounds.
                mask = known[..., column]
                counts["cells"] += int(mask.sum())
                counts["cells_by_field"][column] += int(mask.sum())
                continue
            column_values = np.asarray(projected[..., column], dtype=float)
            low_c, high_c = float(low[column]), float(fixed_high[column])
            if np.isnan(high_c):
                if upper.ndim == 2 and upper.shape[0] == rows:
                    upper_c = upper[:, column]
                elif upper.ndim == 2 and upper.shape[0] == 1:
                    upper_c = np.full(rows, upper[0, column])
                else:
                    upper_c = np.full(rows, np.nan)
                if not np.isfinite(upper_c).all():
                    raise ContractError(
                        f"numeric field {item.field} lacks its declared projection bound")
                high_c = upper_c
            low_t = np.broadcast_to(np.asarray(low_c, dtype=float), column_values.shape)
            high_t = np.broadcast_to(np.asarray(high_c, dtype=float), column_values.shape) \
                if np.ndim(high_c) == 0 else np.broadcast_to(
                    np.asarray(high_c, dtype=float)[:, None], column_values.shape)
            mask = known[..., column]
            below = (column_values < low_t) & mask
            above = (column_values > high_t) & mask
            clipped = np.where(mask, np.clip(column_values, low_t, high_t), column_values)
            low_hit = (clipped == low_t) & mask
            high_hit = (clipped == high_t) & mask
            projected[..., column] = clipped
            for name, matrix in (("cells", mask), ("below", below), ("above", above),
                                 ("endpoint_low", low_hit), ("endpoint_high", high_hit)):
                counts[name] += int(matrix.sum())
                counts[name + "_by_field"][column] += int(matrix.sum())
        return projected, counts


class OracleStateSampler:
    """Declared per-field noisy-state sampler over assembled state tables.

    The source table is the caller's explicit operand supply with shape
    [N,11,F] physical values (nan where unknown), known flags and in-scope
    flags. Candidate draws read one declared stream per query; draw k of that
    stream is exactly what every smaller candidate count reuses, so candidate
    arms differ only in sample count. Only known future cells carry noise; the
    anchor row stays the observed prefix evidence. The declared physical
    projection applies after the draw with exact endpoints and no epsilon.
    """

    def __init__(self, noise, *, seed_formula=oracle_seed_default, sigma_scale=1.0,
                 projector=None, field_laws=None):
        """Compile one sampler over declared laws mapped onto value slots.

        noise may be the raw declaration tuple (single-owner episodes) or the
        exact law-to-slot mapping compiled once by check_field_declarations,
        where one explicit shared law object repeats across the distinct value
        slots. Declaration-scope uniqueness and the aligned shared mapping are
        different notions: an unscoped duplicate declaration is still a
        rejected collision, only the same shared law object is tolerated on
        several slots.
        """
        self.noise = check_noise_declarations(noise)
        self.field_laws = {} if field_laws is None else dict(field_laws)
        for item in self.noise:
            if item.method in ("discrete_law", "component_law") and not callable(self.field_laws.get(item.field)):
                raise ContractError(f"{item.field} needs its declared callable {item.method}")
            if item.method == "gaussian" and item.data_type == "integer" \
                    and not callable(self.field_laws.get(item.field)):
                raise ContractError(
                    f"gaussian integer field {item.field} must declare its exact "
                    "rounding law through field_laws")
        self.seed_formula = seed_formula
        if not callable(self.seed_formula):
            raise ContractError("oracle sampler needs a callable declared seed formula")
        self.sigma_scale = float(sigma_scale)
        if self.sigma_scale < 0 or not np.isfinite(self.sigma_scale):
            raise ContractError("declared noise sigma scale must be finite and nonnegative")
        self.projector = PhysicalOperandProjector(self.noise) if projector is None else projector

    def sigma_vector(self):
        """Sigma over the declared numeric gaussian columns; typed columns are 0."""
        sigma = np.zeros(len(self.noise))
        for index, item in enumerate(self.noise):
            if item.method != "gaussian":
                continue
            sigma[index] = item.sigma_for()[0] * self.sigma_scale
        return sigma

    def sample(self, source_table, *, k, seed, upper_by_row=None):
        """Return sampled values/known [N,K,11,F] plus declared count dict.

        values may be a float table for declared numeric columns or an object
        table holding raw typed cells (category/identifier strings, declared
        ENU component tuples). Numeric gaussian columns take the declared
        float fast path with the declared draw stream, so existing numeric L/P
        outputs, dtypes and random streams stay bit-identical. Typed columns
        keep their raw physical cells through the identity law or the exact
        declared callback; identifiers never carry gaussian noise; gaussian
        integer fields must declare their rounding law, or the draw fails.
        upper_by_row is the declared per-row, per-field numeric bound [N,F]
        for numeric fields without their own projection interval. The contract
        clock is fixed: source tables must supply all eleven instants.
        """
        if not isinstance(source_table, tuple) or len(source_table) != 3:
            raise ContractError("source table must be a (values, known, scope) tuple")
        values, known, scope = source_table
        values = np.asarray(values)
        known = np.asarray(known, dtype=bool)
        scope = np.asarray(scope, dtype=bool)
        n = values.shape[0]
        fields = len(self.noise)
        if values.ndim != 3 or values.shape[1:] != (ANCHOR_STEPS, fields) \
                or known.shape != values.shape or scope.shape != values.shape:
            raise ContractError("source table must supply [N,11,F] values, known and scope")
        numeric = [not item.typed for item in self.noise]
        if upper_by_row is None:
            upper = np.zeros((n, 0))
        else:
            upper = np.asarray(upper_by_row, dtype=float)
            if upper.shape != (n, fields):
                raise ContractError("declared upper domain bounds must be [N,F]")
        if type(k) is not int or k < 1:
            raise ContractError("candidate count must be a positive integer")
        sampled = np.empty((n, k, ANCHOR_STEPS, fields), dtype=object)
        sampled_known = np.zeros((n, k, ANCHOR_STEPS, fields), dtype=bool)
        counts = {"queries": n, "K": k,
                  "unknown_future_cells": int((~known[:, 1:]).sum()),
                  "out_of_scope_future_cells": int((~scope[:, 1:]).sum())}
        for name in ("cells", "below", "above", "endpoint_low", "endpoint_high"):
            counts[name] = 0
            counts[name + "_by_field"] = [0] * fields
        numeric_draws = not any(item.typed for item in self.noise)
        sigma = self.sigma_vector()
        for query in range(n):
            generator = np.random.default_rng(self.seed_formula(seed, query))
            draws = generator.standard_normal((k, FUTURE_STEPS, fields))
            anchor = values[query, 0]
            anchor_known = known[query, 0]
            if numeric_draws:
                # Existing float fast path: one vectorized draw application,
                # identical arithmetic and stream to the released numeric law.
                future = np.broadcast_to(values[query, 1:].astype(float, copy=False),
                                         (k, FUTURE_STEPS, fields)).copy()
                future += sigma * draws
            else:
                future = np.empty((k, FUTURE_STEPS, fields), dtype=object)
                future[:] = np.broadcast_to(values[query, 1:], (k, FUTURE_STEPS, fields))
                for column, item in enumerate(self.noise):
                    if item.method == "identity":
                        # Declared identity: raw physical cells pass through
                        # for every declared type, numeric included.
                        continue
                    if not numeric[column]:
                        # Gaussian integer fields always carry their declared
                        # rounding law (validated at construction); discrete and
                        # component laws apply through the same callback. One raw
                        # physical cell per candidate: the law sees and returns
                        # declared typed cells (category/identifier strings,
                        # declared component tuples, native integer codes).
                        changed = self.field_laws[item.field](
                            list(future[:, :, column]), known[query, 1:, column],
                            draws[..., column], item)
                        if len(changed) != k:
                            raise ContractError(f"declared field law must return one cell per candidate: {item.field}")
                        column_new = np.empty((k, FUTURE_STEPS), dtype=object)
                        for sample in range(k):
                            row = list(changed[sample])
                            if len(row) != FUTURE_STEPS:
                                raise ContractError(
                                    f"declared field law must return one cell per instant: {item.field}")
                            for time in range(FUTURE_STEPS):
                                column_new[sample, time] = row[time]
                        future[:, :, column] = column_new
                    elif item.method == "gaussian":
                        column_values = np.asarray(future[:, :, column], dtype=float)
                        column_values += item.sigma[0] * self.sigma_scale * draws[..., column]
                        future[:, :, column] = column_values
                    else:
                        raise ContractError(
                            f"numeric field {item.field} supports the declared gaussian "
                            f"and identity laws only, not {item.method}")
            future_known = known[query, 1:]
            drawn_known = np.broadcast_to(future_known, (k,) + future_known.shape)
            projected, drawn = self.projector.project(
                future, drawn_known, upper[query] if upper.size else None)
            for sample in range(k):
                for column, item in enumerate(self.noise):
                    if numeric[column]:
                        sampled[query, sample, 0, column] = \
                            float(anchor[column]) if anchor_known[column] else 0.
                    else:
                        sampled[query, sample, 0, column] = anchor[column]
                for time in range(1, ANCHOR_STEPS):
                    for column, item in enumerate(self.noise):
                        if numeric[column]:
                            sampled[query, sample, time, column] = \
                                float(projected[sample, time - 1, column]) \
                                if future_known[time - 1, column] else 0.
                        else:
                            sampled[query, sample, time, column] = projected[sample, time - 1, column]
            sampled_known[query, :, 0] = anchor_known
            sampled_known[query, :, 1:] = future_known
            for name, total in drawn.items():
                if name.endswith("_by_field"):
                    continue
                counts[name] += total
                counts[name + "_by_field"] = [count + part for count, part
                                              in zip(counts[name + "_by_field"], drawn[name + "_by_field"])]
        if numeric_draws:
            # Released numeric contract: float table, zeros on unknown cells.
            numeric_table = np.zeros((n, k, ANCHOR_STEPS, fields))
            for column, is_num in enumerate(numeric):
                if is_num:
                    numeric_table[..., column] = np.vectorize(
                        lambda cell: float(cell))(sampled[..., column])
                else:
                    numeric_table[..., column] = np.asarray(
                        [[sampled[query, sample, time, column]
                          for time in range(ANCHOR_STEPS)]
                         for query in range(n) for sample in range(k)],
                        dtype=float).reshape(n, k, ANCHOR_STEPS)
            return numeric_table, sampled_known, counts
        return sampled, sampled_known, counts


def _shared_law_alignment(keys):
    """True when every repeated scope key repeats one identical shared law.

    keys pairs each declaration's scope key with the declaration object. Two
    distinct declarations may carry the same scope key only when they pin
    different owners; one explicit shared law appearing on several value slots
    is one object repeated, which is exactly the declared shared mapping onto
    independent slots — not a duplicate declaration.
    """
    by_key = {}
    for key, item in keys:
        group = by_key.setdefault(key, [])
        if group and (item is not group[0] or not getattr(item, "shared", False)):
            return False
        group.append(item)
    return True


def check_noise_declarations(noise):
    """Accept an ordered noise declaration tuple with distinct declared scopes.

    Two entries may share a canonical field name only when they pin different
    typed owners through owner_entity_ids, or when exactly one explicit shared
    law object is mapped onto several value slots. Any other repeat, in
    particular two separately written unscoped duplicates, is a collision.
    """
    noise = tuple(noise)
    if not noise:
        raise ContractError("declared noise law needs at least one field")
    keys = [(_declaration_key(item), item) for item in noise]
    if len(set(key for key, _ in keys)) != len(keys) and not _shared_law_alignment(keys):
        raise ContractError("declared noise identity contains duplicate scopes")
    by_field = {}
    for item in noise:
        by_field.setdefault(item.field, []).append(_declaration_key(item)[1])
    for field, scopes in by_field.items():
        if len(scopes) > 1 and None not in scopes:
            owners = [scope for scope in scopes if scope]
            if len(owners) != len(scopes):
                raise ContractError(
                    f"field {field} mixes owner-pinned and unpinned declarations; "
                    "declare an explicit shared law or pin every occurrence")
    return noise


def sample_oracle_states(source_table, noise, *, k, seed, upper_by_row,
                         seed_formula=oracle_seed_default, sigma_scale=1.0, field_laws=None):
    """One-call declared-state sampling with an explicit seed formula."""
    sampler = OracleStateSampler(noise, seed_formula=seed_formula, sigma_scale=sigma_scale,
                                 field_laws=field_laws)
    return sampler.sample(source_table, k=k, seed=seed, upper_by_row=upper_by_row)


def apply_field_transforms(values, known, transforms):
    """Declared per-field transform on known encoded numeric cells, 0 elsewhere.

    values are already encoded numeric model inputs (after the assembled typed
    projection); the transform never touches raw typed physical operands.
    """
    values = np.asarray(values, dtype=float)
    known = np.asarray(known, dtype=bool)
    if values.shape != known.shape:
        raise ContractError("transformed operands need an aligned known mask")
    if len(transforms) != values.shape[-1]:
        raise ContractError("transform count differs from the declared field axis")
    result = np.zeros(values.shape, dtype=float)
    for index, item in enumerate(transforms):
        mask = known[..., index]
        result[..., index] = np.where(mask, (values[..., index] - item.mean) / item.scale, 0.)
    return result


class _ScopedEvaluator(FuturePredicateEvaluator):
    """Reuse one compiled immutable plan set under a different scope overlay."""

    def __init__(self, template, applicable, episode=None):
        self.episode = template.episode if episode is None else episode
        self.plans = template.plans
        self.predicate_field_incidence = template.predicate_field_incidence
        self.expanded_predicate_field_incidence = template.expanded_predicate_field_incidence
        self.applicable = _frozen_array(applicable, bool)

    def evaluate(self, record):
        codes = super().evaluate(record)
        if self.applicable.shape != codes.shape:
            raise ContractError("declared applicability differs from fixed-rule result axes")
        return _frozen_array(np.where(self.applicable, codes, TRUTH_CODES["out_of_scope"]), np.int8)


class OracleRuleEvaluator:
    """Reusable fixed-rule evaluation for one contract and its candidate rows.

    The exact field/owner/rule plans are compiled once per EpisodeContract and
    reused for every query window of that contract and every candidate count;
    no per-query scan, graph rebuild or provenance re-check runs in the loop.
    scoped() overlays an explicit [K,11,P] applicability table that keeps
    query-specific in-scope availability distinct from missing operands:
    out-of-scope cells return the declared code, never false.
    """

    def __init__(self, episode):
        if not isinstance(episode, EpisodeContract):
            raise ContractError("oracle rule evaluation requires an EpisodeContract")
        self.episode = episode
        self._evaluator = FuturePredicateEvaluator(episode)

    @property
    def predicate_field_incidence(self):
        return self._evaluator.predicate_field_incidence

    def scoped(self, applicable, *, episode=None):
        """Return a reusable evaluator for one record's declared scope table."""
        return _ScopedEvaluator(self._evaluator, applicable, episode)

    @property
    def flow_code(self):
        """Exact per-flow plan identity: field/query declarations and scope.

        Callers compile one evaluator per distinct flow code and reuse it at
        every anchor of that flow; anchors differ only in their declared scope
        overlay and candidate operands, never in compiled plans.
        """
        return (self.episode.episode_id, self.episode.branch_id,
                tuple((item.schema.field, item.schema.data_type, item.schema.unit,
                       tuple(owner.ontology_class for owner in item.owner))
                      for item in self.episode.fields),
                tuple(item.contract.id for item in self.episode.queries),
                tuple(item.applicable for item in self.episode.queries))

    def evaluate(self, record, *, applicable=None):
        """Evaluate fixed rules for one record under the episode contract."""
        if applicable is None:
            return self._evaluator.evaluate(record)
        return self.scoped(applicable).evaluate(record)


def evaluate_contract_batch(contract_episode, records, *, applicable=None):
    """Assemble a fixed-rule batch for candidate records of one contract.

    One compiled plan set serves the whole cohort. applicable, when supplied,
    is [N,K,11,P], one explicit scope table per record.
    """
    records = tuple(records)
    if not records:
        raise ContractError("cannot assemble an empty contract batch")
    episode = records[0].episode
    if episode is not contract_episode:
        raise ContractError("contract batch differs from its declared episode")
    if any(record.episode is not episode for record in records):
        raise ContractError("contract batch records must share one episode contract")
    evaluator = OracleRuleEvaluator(episode)
    if applicable is None:
        evaluators = [evaluator.scoped(np.ones((record.sampled_states.shape[0],) +
                     record.sampled_states.shape[1:-1] + (len(episode.queries),), bool))
                     for record in records]
    else:
        table = np.asarray(applicable, dtype=bool)
        if table.shape != (len(records),) + records[0].sampled_states.shape[:-1] + (len(episode.queries),):
            raise ContractError("declared applicability must be [N,K,11,P]")
        evaluators = [evaluator.scoped(table[index]) for index in range(len(records))]
    return assemble_trajectory_batch(records, evaluators=tuple(evaluators))


def compute_first_flip_supervision(truth_codes):
    """Cumulative first-opposite supervision from exact four-valued codes.

    truth_codes [N,P,11] carries the declared codes at the anchor and the ten
    future instants. For each predicate with a known anchor: after the first
    known value opposite to the anchor the later outcome is known (target 1 and
    all later bins incompatible); before any opposite, a fully observed
    prefix keeps the cumulative target 0 known (earlier timed bins incompatible).
    Unknown and out-of-scope codes never count as false observations and stay
    masked; targets are nan where unknown. Returns targets/known [N,P,10] and
    compatible [N,P,11].
    """
    codes = np.asarray(truth_codes, dtype=np.int8)
    if codes.ndim != 3 or codes.shape[-1] != ANCHOR_STEPS:
        raise ContractError("first-flip supervision needs [N,P,11] exact truth codes")
    observed = (codes == TRUTH_CODES["false"]) | (codes == TRUTH_CODES["true"])
    n, predicates = codes.shape[0], codes.shape[1]
    targets = np.full((n, predicates, TIMED_BINS), np.nan)
    known = np.zeros((n, predicates, TIMED_BINS), bool)
    compatible = np.ones((n, predicates, OUTCOME_BINS), bool)
    for i in range(n):
        for p in range(predicates):
            anchor = codes[i, p, 0]
            if anchor not in (TRUTH_CODES["false"], TRUTH_CODES["true"]):
                continue
            opposite_seen, complete = False, True
            for j in range(1, ANCHOR_STEPS):
                step_observed = observed[i, p, j]
                complete = complete and step_observed
                opposite_seen = opposite_seen or (step_observed and codes[i, p, j] != anchor)
                if opposite_seen or complete:
                    known[i, p, j - 1] = True
                    targets[i, p, j - 1] = float(opposite_seen)
                    if opposite_seen:
                        compatible[i, p, j:] = False
                    else:
                        compatible[i, p, :j] = False
    return targets, known, compatible


def first_flip_event_mass(truth_codes):
    """Unnormalized candidate-mean fixed-rule first-flip event mass.

    Codes [K,11,P]: a known anchor starts the trace; the first known opposite
    places unit mass at its timed bin, a fully observed trace without opposite
    places mass on the tail bin, and unknown cells stop the trace unresolved.
    Returns the candidate-mean mass [P,11] and the missing-code count [P].
    """
    codes = np.asarray(truth_codes, dtype=np.int8)
    if codes.ndim != 3 or codes.shape[1] != ANCHOR_STEPS:
        raise ContractError("first-flip event mass needs [K,11,P] exact truth codes")
    k, steps, fields = codes.shape
    observed = (codes == TRUTH_CODES["false"]) | (codes == TRUTH_CODES["true"])
    mass = np.zeros((k, fields, OUTCOME_BINS))
    for f in range(fields):
        for d in range(k):
            if not observed[d, 0, f]:
                continue
            for t in range(1, steps):
                if not observed[d, t, f]:
                    break
                if codes[d, t, f] != codes[d, 0, f]:
                    mass[d, f, t - 1] = 1
                    break
            else:
                mass[d, f, OUTCOME_BINS - 1] = 1
    return mass.mean(0), (~observed).sum((0, 1))


class OracleTrajectoryPreparer:
    """Generic cohort assembly from sampled states to fixed-rule arrays.

    One preparer compiles the contract evaluator exactly once for the declared
    episode contract, then evaluates every candidate record of that contract.
    Scope tables, cumulative supervision and the neural transform enter
    explicitly; the module invents no supervision from sampled operands and no
    transform default. Prepared numeric arrays keep the shared interface:
    history[B,H,F], sampled states[B,K,11,F], known masks, rule truth
    [B,K,11,P], incidence and per-predicate outcome logits.
    """

    def __init__(self, episode, *, noise, transforms=None, fields=None):
        fields = episode.fields if fields is None else tuple(fields)
        self.noise, self.transforms = check_field_declarations(fields, noise, transforms)
        self.episode = episode
        self.evaluator = OracleRuleEvaluator(episode)

    def sampler(self, *, seed_formula=oracle_seed_default, sigma_scale=1.0,
                field_laws=None):
        """One sampler over the exact compiled law-to-slot mapping.

        The mapping is compiled once by check_field_declarations: an explicit
        shared law repeats as one object across the distinct value slots, and
        the sampler consumes that alignment without revalidating scopes.
        """
        return OracleStateSampler(self.noise, seed_formula=seed_formula,
                                  sigma_scale=sigma_scale, field_laws=field_laws)

    def truth_codes_for(self, records, *, applicable=None):
        """Fixed-rule codes [N,K,11,P] for candidate records of one contract."""
        records = tuple(records)
        codes = np.stack([self.evaluator.evaluate(record) for record in records])
        if applicable is not None:
            table = np.asarray(applicable, dtype=bool)
            if table.shape != codes.shape:
                raise ContractError("declared applicability differs from fixed-rule result axes")
            codes = np.where(table, codes, TRUTH_CODES["out_of_scope"]).astype(np.int8)
        return codes

    def normalized_states(self, values, known):
        """Declared per-field transform on known physical cells, 0 elsewhere."""
        return apply_field_transforms(values, known, self.transforms)

    def assemble(self, records, *, applicable):
        """Existing fixed-rule batch assembly with explicit per-record scope."""
        return evaluate_contract_batch(self.episode, records, applicable=applicable)


class OracleSourceIndexAdapter:
    """Exact typed source join for declared producer rows.

    source_index maps ProducerRowKey to producer row mappings; lookups use the
    exact typed key only. Missing keys yield None rows, which the fixed-rule
    evaluator keeps as UNKNOWN operands. No substring or payload scan selects a
    row and no provenance recheck runs inside the numerical loop.
    """

    def __init__(self, source_index):
        self.index = source_index

    def rows(self, episode, query_index, candidate_count):
        from .predicate_trajectory_data import producer_rows_from_index
        return producer_rows_from_index(episode, query_index, candidate_count, self.index)


def build_candidate_record(episode, history, history_known, sampled_states,
                           sampled_known, *, producer_rows=()):
    """Existing immutable CandidateTrajectory from generic declared arrays.

    sampled_states may hold raw typed physical cells (object table); the
    assembled typed projection encodes them for the model afterwards.
    """
    return CandidateTrajectory(episode, history, history_known, sampled_states,
                               sampled_known, producer_rows=tuple(producer_rows))


def tensorize_cohort(batch, *, split, device, targets=None, target_known=None,
                     compatible=None, prior_logits=None, field_type_keys=None):
    """Delegate to the existing training preparation for fixed tensors.

    The generic module supplies assembled fixed-rule arrays; the shared dense
    forward consumes the existing PreparedTrajectorySplit tensors unchanged.
    """
    from .predicate_trajectory_training import prepare_assembled_batch
    return prepare_assembled_batch(batch, split=split, device=device, targets=targets,
        target_known=target_known, compatible=compatible, prior_logits=prior_logits,
        field_type_keys=field_type_keys)
