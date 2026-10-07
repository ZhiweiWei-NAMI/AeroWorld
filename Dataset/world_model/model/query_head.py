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
