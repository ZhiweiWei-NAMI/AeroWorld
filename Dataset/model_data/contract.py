"""Split-neutral contracts for the AERO-210 state/predicate projection."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DATASET_SCHEMA_NAME = "aero_unified_graph_dataset"
DATASET_SCHEMA_VERSION = "2.1.0"
EPISODE_SCHEMA_NAME = "aero_unified_graph_episode"
EPISODE_SCHEMA_VERSION = "2.1.0"

FORMAL_TICKS = tuple(range(0, 901, 5))
TICK_TO_INDEX = {tick: index for index, tick in enumerate(FORMAL_TICKS)}
FORMAL_TICK_COUNT = len(FORMAL_TICKS)
SIMULATION_TICK_SECONDS = 0.1
FORMAL_FRAME_SECONDS = 0.5

PREDICATE_TRUTH_CODES = {
    "false": 0,
    "true": 1,
    "unknown": 2,
    "out_of_scope": 3,
}
INACTIVE_PREDICATE_CODE = 4

EDIT_CODES = {
    "retain": 0,
    "birth": 1,
    "death": 2,
    "value_change": 3,
}

ALLOWED_HISTORY_LENGTHS = frozenset({4, 8, 16})
ALLOWED_HORIZONS = frozenset({1, 5, 10, 20})
MATERIALIZED_WINDOWS = ((1, 1), (8, 10))

UNASSIGNED_SPLIT = "unassigned"

_EPISODE_PATTERN = re.compile(r"^(?P<profile>.+)__seed(?P<seed>00|01|02)$")

BOOLEAN_FIELDS = frozenset(
    {
        "aircraft_inside_corridor",
        "cooperative_flag",
        "crossed_stop_line",
        "inside_restricted_region",
        "restricted_region_active",
        "rth_mode_active",
    }
)

CATEGORICAL_FIELDS = frozenset(
    {
        "capacity_model",
        "controlling_signal_state",
        "entity_role",
        "pair_distance_metric",
    }
)

DIMENSIONLESS_CONTINUOUS_FIELDS = frozenset(
    {
        "corridor_capacity",
        "corridor_occupancy_count",
    }
)

VECTOR_WIDTHS = {
    "assigned_landing_zone_pose_enu_m": 3,
    "corridor_center_enu_m": 3,
    "corridor_cross_section_size_m": 2,
    "corridor_extent_m": 3,
    "home_pad_pose_enu_m": 3,
    "position_enu_m": 3,
    "velocity_enu_mps": 3,
}

RAGGED_FIELD_KINDS = {
    "active_restricted_region_ids": "multi_reference",
    "boundary_polygon_enu_m": "ragged_continuous",
    "containing_corridor_ids": "multi_reference",
    "occupying_aircraft_ids": "multi_reference",
    "pair_entity_ids": "multi_reference",
}


class ModelDataError(ValueError):
    """Raised when source truth cannot be projected without ambiguity."""


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for identifiers and provenance."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def stable_identifier(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def stable_int63(namespace: str, value: str) -> int:
    payload = canonical_json((namespace, value)).encode("utf-8")
    code = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
    if code == 0:
        raise ModelDataError(f"reserved zero code generated for {namespace}:{value}")
    return code


def parse_episode_id(episode_id: str) -> tuple[str, str]:
    match = _EPISODE_PATTERN.fullmatch(episode_id)
    if match is None:
        raise ModelDataError(f"invalid formal episode id: {episode_id!r}")
    return match.group("profile"), match.group("seed")


def expected_field_kind(field_name: str) -> tuple[str, int]:
    if field_name in BOOLEAN_FIELDS:
        return "boolean", 1
    if field_name in CATEGORICAL_FIELDS:
        return "categorical", 1
    if field_name in RAGGED_FIELD_KINDS:
        return RAGGED_FIELD_KINDS[field_name], 0
    if field_name.endswith("_id"):
        return "reference", 1
    if field_name in VECTOR_WIDTHS:
        return "continuous", VECTOR_WIDTHS[field_name]
    if field_name in DIMENSIONLESS_CONTINUOUS_FIELDS:
        return "continuous", 1
    if field_name.endswith(("_m", "_mps", "_deg")):
        return "continuous", 1
    raise ModelDataError(f"field lacks an explicit state type contract: {field_name}")


def field_unit(field_name: str, value_kind: str) -> str | None:
    if value_kind not in {"continuous", "ragged_continuous"}:
        return None
    if field_name.endswith("_mps"):
        return "m/s"
    if field_name.endswith("_m"):
        return "m"
    if field_name.endswith("_deg"):
        return "degree"
    if field_name in DIMENSIONLESS_CONTINUOUS_FIELDS:
        return "1"
    raise ModelDataError(f"numeric field lacks an explicit state unit: {field_name}")


def is_missing_state_value(value: Any) -> bool:
    return value is None or value == "unknown"


def validate_state_value(field_name: str, value: Any) -> tuple[str, int]:
    expected_kind, expected_width = expected_field_kind(field_name)
    if is_missing_state_value(value):
        return expected_kind, expected_width
    if expected_kind == "boolean":
        if not isinstance(value, bool):
            raise ModelDataError(
                f"{field_name} requires bool, found {type(value).__name__}"
            )
    elif expected_kind == "categorical":
        if not isinstance(value, str):
            raise ModelDataError(
                f"{field_name} requires str, found {type(value).__name__}"
            )
    elif expected_kind == "reference":
        if not isinstance(value, str) or not value:
            raise ModelDataError(f"{field_name} requires a non-empty reference id")
    elif expected_kind == "continuous":
        if expected_width == 1:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ModelDataError(f"{field_name} requires a finite scalar")
            if not math.isfinite(float(value)):
                raise ModelDataError(f"{field_name} contains a non-finite scalar")
        else:
            if not isinstance(value, list) or len(value) != expected_width:
                raise ModelDataError(
                    f"{field_name} requires a width-{expected_width} numeric vector"
                )
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in value
            ):
                raise ModelDataError(f"{field_name} contains a non-finite vector value")
    elif expected_kind == "multi_reference":
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ModelDataError(f"{field_name} requires a list of reference ids")
    elif expected_kind == "ragged_continuous":
        if not isinstance(value, list) or any(
            not isinstance(point, list)
            or len(point) not in {2, 3}
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in point
            )
            for point in value
        ):
            raise ModelDataError(f"{field_name} requires a finite point sequence")
    else:
        raise ModelDataError(f"unsupported state kind: {expected_kind}")
    return expected_kind, expected_width


def pack_bits(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype=np.bool_).reshape(-1)
    return np.packbits(array, bitorder="little").tobytes()


def unpack_bits(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    size = math.prod(shape)
    unpacked = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8), bitorder="little", count=size
    )
    return unpacked.astype(np.bool_, copy=False).reshape(shape)


def predicate_masks(codes: np.ndarray) -> Mapping[str, np.ndarray]:
    values = np.asarray(codes, dtype=np.uint8)
    if np.any(values > INACTIVE_PREDICATE_CODE):
        raise ModelDataError("predicate tensor contains an unknown truth code")
    return {
        "activation": values == PREDICATE_TRUTH_CODES["true"],
        "supervision": (values == PREDICATE_TRUTH_CODES["false"])
        | (values == PREDICATE_TRUTH_CODES["true"]),
        "scope": (values == PREDICATE_TRUTH_CODES["false"])
        | (values == PREDICATE_TRUTH_CODES["true"])
        | (values == PREDICATE_TRUTH_CODES["unknown"]),
        "candidate_active": values != INACTIVE_PREDICATE_CODE,
    }
