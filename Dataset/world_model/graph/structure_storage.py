"""Bounded Parquet storage for observed structural source projections.

The three tables retain one normalized value per projection and one original
value per permitted source leaf. A record points back to its complete source
root through its declaration and physical line or root ordinal. Integrity-key
subtrees are counted but are not copied into these tables.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from Dataset.world_model.graph.field_paths import source_value_at_tokens
from Dataset.world_model.graph.field_structure import (
    IMPLEMENTATION_SOURCES_GAP_NORMALIZED_ID,
    is_declared_implementation_sources_gap,
    is_declared_implementation_sources_gap_projection,
)
from Dataset.world_model.graph.structure_catalog import (
    CatalogFragment, CatalogIndexMaps, StructureCatalog, _validate_fragment,
)
from Dataset.world_model.graph.structure_sources import project_record

if TYPE_CHECKING:
    from Dataset.world_model.graph.source_projection import ProjectedField
    from Dataset.world_model.graph.structure_catalog import CatalogIndices
    from Dataset.world_model.graph.structure_sources import SourceRecord


DEFAULT_TARGET_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_LEAVES = 80_000
_TABLE_NAMES = ("records.parquet", "projections.parquet", "source_leaves.parquet")
_VALUE_TYPES = frozenset({
    "null", "bool", "int", "float", "string", "empty_list", "empty_object",
})
_STRUCTURE_STATUSES = frozenset({
    "source_only", "resolved", "declared_only", "structure_conflict",
})


RECORD_SCHEMA = pa.schema([
    ("record_index", pa.int64()),
    ("declaration_index", pa.int64()),
    ("source_line", pa.int64()),
    ("record_ordinal", pa.int64()),
    ("effective_family", pa.string()),
    ("root_kind", pa.string()),
    ("record_tick_present", pa.bool_()),
    ("record_tick_type", pa.string()),
    ("record_tick_json", pa.binary()),
    ("excluded_integrity_subtrees", pa.int64()),
    ("projection_start", pa.int64()),
    ("projection_count", pa.int64()),
    ("source_leaf_start", pa.int64()),
    ("source_leaf_count", pa.int64()),
    ("oversized_record", pa.bool_()),
])

PROJECTION_SCHEMA = pa.schema([
    ("projection_index", pa.int64()),
    ("record_index", pa.int64()),
    ("normalized_index", pa.int64()),
    ("normalized_id", pa.string()),
    ("normalized_value_json", pa.binary()),
    ("value_type", pa.string()),
    ("value_status", pa.string()),
    ("structure_status", pa.string()),
    ("record_observed", pa.bool_()),
    ("binding_slot", pa.string()),
    ("declared_shape_json", pa.binary()),
    ("observed_shape_json", pa.binary()),
    ("normalized_shape_json", pa.binary()),
    ("envelope_tokens_json", pa.binary()),
    ("payload_tokens_json", pa.binary()),
    ("payload_positions", pa.list_(pa.int64())),
    ("payload_shape_json", pa.binary()),
    ("source_leaf_start", pa.int64()),
    ("source_leaf_count", pa.int64()),
])

SOURCE_LEAF_SCHEMA = pa.schema([
    ("source_leaf_index", pa.int64()),
    ("projection_index", pa.int64()),
    ("source_index", pa.int64()),
    ("source_field_id", pa.string()),
    ("field_path", pa.string()),
    ("raw_path", pa.string()),
    ("raw_value_json", pa.binary()),
    ("raw_value_type", pa.string()),
    ("element_positions", pa.list_(pa.int64())),
    ("source_tokens_json", pa.binary()),
    ("input_tokens_json", pa.binary()),
    ("snapshot_tick", pa.int64()),
    ("entity_id", pa.string()),
    ("identity_status", pa.string()),
    ("identity_reason", pa.string()),
    ("declared_entity_id", pa.string()),
    ("entity_namespace", pa.string()),
    ("target_scope", pa.string()),
    ("value_tick", pa.int64()),
    ("time_role", pa.string()),
    ("endpoint_role", pa.string()),
    ("source_branch", pa.string()),
    ("source_subtype", pa.string()),
    ("predicate_id", pa.string()),
    ("source_path_key", pa.string()),
])


@dataclass(frozen=True)
class RecordLocation:
    shard_id: int
    record_index: int


@dataclass(frozen=True)
class ShardSummary:
    shard_id: int
    relative_path: str
    records: int
    projections: int
    source_leaves: int
    first_declaration_index: int
    last_declaration_index: int
    excluded_integrity_subtrees: int
    oversized_records: int
    compressed_bytes: int


@dataclass(frozen=True)
class StorageSummary:
    shards: tuple[ShardSummary, ...]
    records: int
    projections: int
    source_leaves: int
    excluded_integrity_subtrees: int
    oversized_records: int


@dataclass(frozen=True)
class StorageResumePoint:
    shard_id: int
    record_index: int
    declaration_index: int
    record_ordinal: int
    source_line: int | None


@dataclass(frozen=True)
class EncodedRecordSummary:
    declaration_index: int
    record_ordinal: int
    source_line: int | None
    source_file: str
    projection_count: int
    source_leaf_count: int
    structure_conflict_projections: int
    excluded_integrity_subtrees: int
    estimated_bytes: int


@dataclass(frozen=True)
class EncodedStructureBatch:
    """Worker output with batch-local indexes and complete record boundaries."""

    catalog_fragment: CatalogFragment
    record_table: pa.Table
    projection_table: pa.Table
    source_leaf_table: pa.Table
    record_summaries: tuple[EncodedRecordSummary, ...]


def _nonnegative(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} is not a nonnegative integer: {value!r}")
    return value


def _record_child_counts(row: dict[str, Any]) -> tuple[int, int]:
    projections = _nonnegative(row["projection_count"], "projection_count")
    leaves = _nonnegative(row["source_leaf_count"], "source_leaf_count")
    if not projections and leaves:
        raise ValueError("record with no projections has source leaves")
    return projections, leaves


def _check_projection_reference(row: dict[str, Any], expected_index: int,
                                record_index: int, start: int, end: int) -> None:
    if (row["projection_index"] != expected_index or
            row["record_index"] != record_index or
            not start <= expected_index < end):
        raise ValueError("projection does not reference its record")


def _checked_projection_leaf_end(row: dict[str, Any], expected_start: int,
                                 record_start: int, record_end: int) -> int:
    if row["source_leaf_start"] != expected_start:
        raise ValueError("projection source leaf ranges are not consecutive")
    count = _nonnegative(row["source_leaf_count"], "source_leaf_count")
    if count < 1:
        raise ValueError("projection has no original source leaf")
    end = expected_start + count
    if expected_start < record_start or end > record_end:
        raise ValueError("projection source leaves fall outside their record")
    return end


def _check_source_leaf_reference(row: dict[str, Any], expected_index: int,
                                 projection_index: int, start: int, end: int) -> None:
    if (row["source_leaf_index"] != expected_index or
            row["projection_index"] != projection_index or
            not start <= expected_index < end):
        raise ValueError("source leaf does not reference its projection")


def _optional_int(value: Any, name: str) -> int | None:
    if value is not None:
        if type(value) is not int:
            raise TypeError(f"{name} is not an integer: {value!r}")
        return value
    return None


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        return "float"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "empty_list"
    if type(value) is dict:
        return "empty_object"
    raise TypeError(f"source value has unsupported type: {type(value).__name__}")


def _json_type(value: Any) -> str:
    tag = _value_type(value)
    return {"empty_list": "array", "empty_object": "object"}.get(tag, tag)


def _assert_json_value(value: Any) -> None:
    if value is None or type(value) in (bool, int, float, str):
        return
    if type(value) is list:
        for item in value:
            _assert_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"JSON object key is not a string: {key!r}")
            _assert_json_value(item)
        return
    raise TypeError(f"value is not JSON data: {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    _assert_json_value(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _tokens_json(tokens: tuple[str | int, ...], name: str) -> bytes:
    if not isinstance(tokens, tuple):
        raise TypeError(f"{name} is not a token tuple")
    for token in tokens:
        if type(token) is int:
            _nonnegative(token, name)
        elif type(token) is not str:
            raise TypeError(f"{name} contains a non-key, non-index token: {token!r}")
    return _json_bytes(list(tokens))


def _positions(values: tuple[int, ...], name: str) -> list[int]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} is not a tuple")
    return [_nonnegative(value, name) for value in values]


def _shape_json(shape: tuple[int | None, ...] | None, name: str) -> bytes | None:
    if shape is None:
        return None
    if not isinstance(shape, tuple):
        raise TypeError(f"{name} is not a shape tuple")
    for axis in shape:
        if axis is not None:
            _nonnegative(axis, name)
    return _json_bytes(list(shape))


def _same_typed(first: Any, second: Any) -> bool:
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return (first.keys() == second.keys() and
                all(_same_typed(first[key], second[key]) for key in first))
    if isinstance(first, list):
        return (len(first) == len(second) and
                all(_same_typed(left, right) for left, right in zip(first, second)))
    return first == second


def _row_size(row: dict[str, Any]) -> int:
    """Bound Python row buffering; this is not an on-disk size estimate."""
    total = 160
    for value in row.values():
        if isinstance(value, (bytes, str)):
            total += len(value)
        elif isinstance(value, list):
            total += 8 * len(value)
    return total


def _is_json_root(record: SourceRecord) -> bool:
    """One complete JSON root rather than one physical JSONL row.

    Mirrors the classification in ``structure_sources.iter_source_records`` so
    storage accepts exactly the boundaries that reader yields: a ``.json``
    document root, or the registered ``arm_semantic_history`` gzip root, which
    is a single JSON document behind a ``.json.gz`` name.  Every other
    compressed or unknown suffix stays outside this predicate and is rejected
    by the caller.
    """
    return (record.path.suffix == ".json" or
            (record.source_family == "arm_semantic_history" and
             record.path.suffixes[-2:] == [".json", ".gz"]))


def _encoded_record(record: SourceRecord,
                    projections: Sequence[ProjectedField],
                    bindings: Sequence[CatalogIndices]
                    ) -> tuple[dict[str, Any], list[dict[str, Any]],
                               list[dict[str, Any]], int]:
    _nonnegative(record.declaration_index, "declaration_index")
    _nonnegative(record.record_ordinal, "record_ordinal")
    _nonnegative(record.excluded_integrity_subtrees, "excluded_integrity_subtrees")
    if type(record.raw_root) is not dict:
        raise TypeError(f"source root is not a JSON object: {record.source_file}")
    if not isinstance(record.effective_family, str) or not record.effective_family:
        raise ValueError(f"source record has no effective family: {record.source_file}")
    if _is_json_root(record):
        if record.source_line is not None or record.record_ordinal != 0:
            raise ValueError(f"JSON root has a source line or nonzero ordinal: {record.source_file}")
    elif record.path.suffix == ".jsonl":
        if type(record.source_line) is not int or record.source_line < 1:
            raise ValueError(f"JSONL row lacks its physical line: {record.source_file}")
    else:
        raise ValueError(f"source has no JSON record boundary: {record.source_file}")
    if type(record.record_tick_present) is not bool or record.record_tick_present != (
            "tick" in record.raw_root):
        raise ValueError(f"record tick presence differs from source root: {record.source_file}")
    if record.record_tick_present and not _same_typed(
            record.record_tick, record.raw_root["tick"]):
        raise ValueError(f"record tick differs from source root: {record.source_file}")
    if len(projections) != len(bindings):
        raise ValueError(f"projection and catalog index counts differ: {record.source_file}")

    projection_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    seen_tokens: set[tuple[str | int, ...]] = set()
    for projection_index, (projected, binding) in enumerate(zip(projections, bindings)):
        normalized_index = _nonnegative(binding.normalized_index, "normalized_index")
        declared_gap = is_declared_implementation_sources_gap(
            projected.normalized_id, projected.value_type,
            projected.value_status, projected.structure_status,
        )
        if declared_gap and not is_declared_implementation_sources_gap_projection(projected):
            raise ValueError("implementation_sources metadata gap has an invalid original value")
        if (projected.value_type is not None and
                projected.value_type != _value_type(projected.value)):
            raise ValueError(f"normalized value type differs: {projected.normalized_id}")
        if (projected.value_type is None and
                (projected.value is not None or
                 projected.structure_status != "structure_conflict" and not declared_gap)):
            raise ValueError(f"untyped normalized value lacks a declared gap: {projected.normalized_id}")
        if projected.structure_status not in _STRUCTURE_STATUSES:
            raise ValueError(f"unknown structure status: {projected.structure_status}")
        if not isinstance(projected.value_status, str) or not projected.value_status:
            raise ValueError(f"normalized value has no status: {projected.normalized_id}")
        if not isinstance(projected.normalized_id, str) or not projected.normalized_id:
            raise ValueError("normalized field has no ID")
        if not projected.sources or projected.absent_source_field_id is not None:
            raise ValueError(f"projection lacks actual source leaves: {projected.normalized_id}")
        if len(projected.sources) != len(binding.source_indices):
            raise ValueError(f"source index count differs: {projected.normalized_id}")
        if (projected.record_observed is not None and
                type(projected.record_observed) is not bool):
            raise TypeError(f"record_observed is not a boolean: {projected.normalized_id}")

        leaf_start = len(source_rows)
        projection_rows.append({
            "projection_index": projection_index,
            "record_index": 0,
            "normalized_index": normalized_index,
            "normalized_id": projected.normalized_id,
            "normalized_value_json": _json_bytes(projected.value),
            "value_type": projected.value_type,
            "value_status": projected.value_status,
            "structure_status": projected.structure_status,
            "record_observed": projected.record_observed,
            "binding_slot": projected.binding_slot,
            "declared_shape_json": _shape_json(projected.declared_shape, "declared_shape"),
            "observed_shape_json": _shape_json(projected.observed_shape, "observed_shape"),
            "normalized_shape_json": _shape_json(projected.normalized_shape, "normalized_shape"),
            "envelope_tokens_json": _tokens_json(projected.envelope_tokens, "envelope_tokens"),
            "payload_tokens_json": _tokens_json(projected.payload_tokens, "payload_tokens"),
            "payload_positions": _positions(projected.payload_positions, "payload_positions"),
            "payload_shape_json": _shape_json(projected.payload_shape, "payload_shape"),
            "source_leaf_start": leaf_start,
            "source_leaf_count": len(projected.sources),
        })
        for source, source_index in zip(projected.sources, binding.source_indices):
            leaf = source.field_value
            _nonnegative(source_index, "source_index")
            tokens = leaf.source_tokens
            if tokens in seen_tokens:
                raise ValueError(f"source leaf is repeated within one root: {record.source_file}:{tokens}")
            seen_tokens.add(tokens)
            from_root = source_value_at_tokens(record.raw_root, tokens)
            if not _same_typed(from_root, leaf.value):
                raise ValueError(f"source leaf differs from its raw root: {record.source_file}:{tokens}")
            raw_type = _value_type(leaf.value)
            if (raw_type == "empty_list" and leaf.value or
                    raw_type == "empty_object" and leaf.value):
                raise ValueError(f"nonempty container reached a source leaf: {record.source_file}:{tokens}")
            positions = _positions(leaf.element_positions, "element_positions")
            if positions != [token for token in tokens if type(token) is int]:
                raise ValueError(f"source leaf array positions differ: {record.source_file}:{tokens}")
            if not isinstance(source.source_field_id, str) or not source.source_field_id:
                raise ValueError(f"source leaf has no field ID: {record.source_file}:{tokens}")
            source_rows.append({
                "source_leaf_index": len(source_rows),
                "projection_index": projection_index,
                "source_index": source_index,
                "source_field_id": source.source_field_id,
                "field_path": leaf.field_path,
                "raw_path": leaf.raw_path,
                "raw_value_json": _json_bytes(leaf.value),
                "raw_value_type": raw_type,
                "element_positions": positions,
                "source_tokens_json": _tokens_json(tokens, "source_tokens"),
                "input_tokens_json": _tokens_json(leaf.input_tokens, "input_tokens"),
                "snapshot_tick": _optional_int(leaf.snapshot_tick, "snapshot_tick"),
                "entity_id": leaf.entity_id,
                "identity_status": leaf.identity_status,
                "identity_reason": leaf.identity_reason,
                "declared_entity_id": leaf.declared_entity_id,
                "entity_namespace": leaf.entity_namespace,
                "target_scope": leaf.target_scope,
                "value_tick": _optional_int(leaf.value_tick, "value_tick"),
                "time_role": leaf.time_role,
                "endpoint_role": leaf.endpoint_role,
                "source_branch": leaf.source_branch,
                "source_subtype": leaf.source_subtype,
                "predicate_id": leaf.predicate_id,
                "source_path_key": leaf.source_path_key,
            })

    record_row = {
        "record_index": 0,
        "declaration_index": record.declaration_index,
        "source_line": record.source_line,
        "record_ordinal": record.record_ordinal,
        "effective_family": record.effective_family,
        "root_kind": "object",
        "record_tick_present": record.record_tick_present,
        "record_tick_type": (_json_type(record.record_tick)
                             if record.record_tick_present else None),
        "record_tick_json": (_json_bytes(record.record_tick)
                             if record.record_tick_present else None),
        "excluded_integrity_subtrees": record.excluded_integrity_subtrees,
        "projection_start": 0,
        "projection_count": len(projection_rows),
        "source_leaf_start": 0,
        "source_leaf_count": len(source_rows),
        "oversized_record": False,
    }
    estimated_bytes = (_row_size(record_row) +
                       sum(map(_row_size, projection_rows)) +
                       sum(map(_row_size, source_rows)))
    return record_row, projection_rows, source_rows, estimated_bytes


def encode_structure_batch(
        records: Sequence[SourceRecord],
        projected_records: Sequence[Sequence[ProjectedField]] | None = None,
) -> EncodedStructureBatch:
    """Project and encode complete roots in a worker, using local catalog indexes."""
    if not records:
        raise ValueError("structure batch is empty")
    if projected_records is not None and len(projected_records) != len(records):
        raise ValueError("projected record count differs from source record count")
    catalog = StructureCatalog()
    record_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    summaries: list[EncodedRecordSummary] = []
    for position, record in enumerate(records):
        projected = (project_record(record) if projected_records is None
                     else projected_records[position])
        bindings = tuple(catalog.observe(value, source_family=record.effective_family)
                         for value in projected)
        record_row, record_projections, record_sources, estimated_bytes = (
            _encoded_record(record, projected, bindings))
        record_index = len(record_rows)
        projection_start = len(projection_rows)
        source_start = len(source_rows)
        record_row["record_index"] = record_index
        record_row["projection_start"] = projection_start
        record_row["source_leaf_start"] = source_start
        for row in record_projections:
            row["projection_index"] += projection_start
            row["record_index"] = record_index
            row["source_leaf_start"] += source_start
        for row in record_sources:
            row["source_leaf_index"] += source_start
            row["projection_index"] += projection_start
        record_rows.append(record_row)
        projection_rows.extend(record_projections)
        source_rows.extend(record_sources)
        summaries.append(EncodedRecordSummary(
            declaration_index=record.declaration_index,
            record_ordinal=record.record_ordinal,
            source_line=record.source_line,
            source_file=record.source_file,
            projection_count=len(record_projections),
            source_leaf_count=len(record_sources),
            structure_conflict_projections=sum(
                value.structure_status == "structure_conflict" for value in projected),
            excluded_integrity_subtrees=record.excluded_integrity_subtrees,
            estimated_bytes=estimated_bytes,
        ))
    fragment = catalog.export_fragment()
    _validate_fragment(fragment)
    return EncodedStructureBatch(
        catalog_fragment=fragment,
        record_table=pa.Table.from_pylist(record_rows, schema=RECORD_SCHEMA),
        projection_table=pa.Table.from_pylist(projection_rows, schema=PROJECTION_SCHEMA),
        source_leaf_table=pa.Table.from_pylist(source_rows, schema=SOURCE_LEAF_SCHEMA),
        record_summaries=tuple(summaries),
    )


def _replace_column(table: pa.Table, name: str,
                    values: pa.Array | pa.ChunkedArray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"structure table has no {name} column")
    return table.set_column(index, name, values)


def _offset_column(table: pa.Table, name: str, delta: int) -> pa.Table:
    if not delta:
        return table
    return _replace_column(
        table, name,
        pc.add_checked(table.column(name), pa.scalar(delta, type=pa.int64())),
    )


def _mapped_index_column(table: pa.Table, *, index_name: str, id_name: str,
                         local_rows: tuple[dict[str, Any], ...],
                         global_indexes: tuple[int, ...]) -> pa.Table:
    if len(local_rows) != len(global_indexes):
        raise ValueError(f"{index_name} map differs from catalog fragment")
    local_ids = []
    for position, (row, global_index) in enumerate(zip(local_rows, global_indexes)):
        if row[index_name] != position:
            raise ValueError(f"{index_name} is not local and consecutive")
        _nonnegative(global_index, index_name)
        local_ids.append(row["id"])
    local_indexes = table.column(index_name)
    if table.num_rows:
        selected_ids = pc.take(pa.array(local_ids, type=pa.string()), local_indexes)
        matches = pc.fill_null(pc.equal(selected_ids, table.column(id_name)), False)
        if pc.all(matches).as_py() is not True:
            raise ValueError(f"{index_name} and {id_name} differ")
    mapped = pc.take(pa.array(global_indexes, type=pa.int64()), local_indexes)
    if table.num_rows and pc.any(pc.is_null(mapped)).as_py():
        raise ValueError(f"{index_name} map has an unresolved index")
    return _replace_column(table, index_name, mapped)


class StructureStorageWriter:
    """Write three ZSTD Parquet tables per bounded, complete-record shard."""

    def __init__(self, root: Path, *, target_bytes: int = DEFAULT_TARGET_BYTES,
                 max_leaves: int = DEFAULT_MAX_LEAVES) -> None:
        self._initialize(root, target_bytes=target_bytes, max_leaves=max_leaves)
        self.shards_root.mkdir(parents=True, exist_ok=False)

    def _initialize(self, root: Path, *, target_bytes: int,
                    max_leaves: int) -> None:
        self.root = Path(root)
        self.target_bytes = _nonnegative(target_bytes, "target_bytes")
        self.max_leaves = _nonnegative(max_leaves, "max_leaves")
        if not self.target_bytes or not self.max_leaves:
            raise ValueError("shard thresholds must be positive")
        self.shards_root = self.root / "shards"
        self._record_tables: list[pa.Table] = []
        self._projection_tables: list[pa.Table] = []
        self._source_tables: list[pa.Table] = []
        self._buffer_records = 0
        self._buffer_projections = 0
        self._buffer_leaves = 0
        self._buffer_bytes = 0
        self._buffer_first_declaration: int | None = None
        self._buffer_last_declaration: int | None = None
        self._buffer_excluded = 0
        self._buffer_oversized = 0
        self._next_shard_id = 0
        self._shard_summaries: list[ShardSummary] = []
        self._last_declaration: int | None = None
        self._last_ordinal: int | None = None
        self._last_line: int | None = None
        self._finished: StorageSummary | None = None

    @classmethod
    def resume_from_validated(
            cls, root: Path, validated: StorageSummary, *,
            target_bytes: int = DEFAULT_TARGET_BYTES,
            max_leaves: int = DEFAULT_MAX_LEAVES,
            ) -> tuple[StructureStorageWriter, StorageResumePoint]:
        """Append after a validated, complete shard prefix without changing it.

        ``validated`` must come from storage validation. This method checks that
        its directories, table sizes, summary totals, and final record still
        match disk; it does not repeat the full row-level validation.
        """
        if type(validated) is not StorageSummary or type(validated.shards) is not tuple:
            raise TypeError("validated structure storage summary is required")
        if not validated.shards:
            raise ValueError("validated structure storage has no shards")
        root = Path(root)
        directories = _shard_dirs(root)
        if len(directories) != len(validated.shards):
            raise ValueError("validated shard count differs from storage")
        for (shard_id, shard_dir), summary in zip(directories, validated.shards):
            if type(summary) is not ShardSummary or summary.shard_id != shard_id:
                raise ValueError(f"validated shard ID differs: {shard_dir}")
            if summary.relative_path != shard_dir.relative_to(root).as_posix():
                raise ValueError(f"validated shard path differs: {shard_dir}")
            for name in ("records", "projections", "source_leaves",
                         "first_declaration_index", "last_declaration_index",
                         "excluded_integrity_subtrees", "oversized_records",
                         "compressed_bytes"):
                _nonnegative(getattr(summary, name), name)
            if not summary.records:
                raise ValueError(f"validated shard is empty: {shard_dir}")
            if (summary.first_declaration_index > summary.last_declaration_index or
                    summary.oversized_records > summary.records):
                raise ValueError(f"validated shard bounds are invalid: {shard_dir}")
            if {item.name for item in shard_dir.iterdir()} != set(_TABLE_NAMES):
                raise ValueError(f"validated shard tables differ: {shard_dir}")
            if sum((shard_dir / name).stat().st_size for name in _TABLE_NAMES) != summary.compressed_bytes:
                raise ValueError(f"validated shard bytes differ: {shard_dir}")
        for name in ("records", "projections", "source_leaves",
                     "excluded_integrity_subtrees", "oversized_records"):
            _nonnegative(getattr(validated, name), name)
            if sum(getattr(shard, name) for shard in validated.shards) != getattr(validated, name):
                raise ValueError(f"validated storage {name} differs from shards")

        last = validated.shards[-1]
        record_file = directories[-1][1] / "records.parquet"
        parquet = pq.ParquetFile(record_file)
        if parquet.schema_arrow != RECORD_SCHEMA or parquet.metadata.num_rows != last.records:
            raise ValueError(f"validated final record table differs: {record_file}")
        records = pq.read_table(record_file, columns=[
            "record_index", "declaration_index", "record_ordinal", "source_line",
        ])
        if records.num_rows != last.records:
            raise ValueError(f"validated final record count differs: {record_file}")
        record = records.slice(records.num_rows - 1, 1).to_pylist()[0]
        point = StorageResumePoint(
            shard_id=last.shard_id,
            record_index=_nonnegative(record["record_index"], "record_index"),
            declaration_index=_nonnegative(record["declaration_index"], "declaration_index"),
            record_ordinal=_nonnegative(record["record_ordinal"], "record_ordinal"),
            source_line=record["source_line"],
        )
        if (point.record_index != last.records - 1 or
                point.declaration_index != last.last_declaration_index or
                point.source_line is not None and
                (type(point.source_line) is not int or point.source_line < 1) or
                point.source_line is None and point.record_ordinal != 0):
            raise ValueError(f"validated final record position differs: {record_file}")

        writer = cls.__new__(cls)
        writer._initialize(root, target_bytes=target_bytes, max_leaves=max_leaves)
        writer._shard_summaries = list(validated.shards)
        writer._next_shard_id = len(validated.shards)
        writer._last_declaration = point.declaration_index
        writer._last_ordinal = point.record_ordinal
        writer._last_line = point.source_line
        return writer, point

    def _check_order(self, summary: EncodedRecordSummary) -> None:
        if (self._last_declaration is None or
                summary.declaration_index > self._last_declaration):
            if summary.record_ordinal != 0:
                raise ValueError(f"first source record ordinal is not zero: {summary.source_file}")
            return
        if summary.declaration_index < self._last_declaration:
            raise ValueError(f"source declaration order reversed: {summary.source_file}")
        assert self._last_ordinal is not None
        if summary.record_ordinal != self._last_ordinal + 1:
            raise ValueError(f"source record ordinal is not consecutive: {summary.source_file}")
        if (summary.source_line is None or self._last_line is None or
                summary.source_line <= self._last_line):
            raise ValueError(f"physical JSONL lines are not increasing: {summary.source_file}")

    def _append_segment(self, batch: EncodedStructureBatch,
                        record_table: pa.Table, start: int, stop: int,
                        projection_starts: list[int], leaf_starts: list[int],
                        oversized: tuple[bool, ...]) -> None:
        if stop == start:
            return
        projection_start, projection_stop = projection_starts[start], projection_starts[stop]
        leaf_start, leaf_stop = leaf_starts[start], leaf_starts[stop]
        records = record_table.slice(start, stop - start)
        projections = batch.projection_table.slice(
            projection_start, projection_stop - projection_start)
        leaves = batch.source_leaf_table.slice(leaf_start, leaf_stop - leaf_start)
        record_delta = self._buffer_records - start
        projection_delta = self._buffer_projections - projection_start
        leaf_delta = self._buffer_leaves - leaf_start
        for name, delta in (("record_index", record_delta),
                            ("projection_start", projection_delta),
                            ("source_leaf_start", leaf_delta)):
            records = _offset_column(records, name, delta)
        for name, delta in (("projection_index", projection_delta),
                            ("record_index", record_delta),
                            ("source_leaf_start", leaf_delta)):
            projections = _offset_column(projections, name, delta)
        for name, delta in (("source_leaf_index", leaf_delta),
                            ("projection_index", projection_delta)):
            leaves = _offset_column(leaves, name, delta)
        self._record_tables.append(records)
        self._projection_tables.append(projections)
        self._source_tables.append(leaves)
        summaries = batch.record_summaries[start:stop]
        if self._buffer_first_declaration is None:
            self._buffer_first_declaration = summaries[0].declaration_index
        self._buffer_last_declaration = summaries[-1].declaration_index
        self._buffer_records += stop - start
        self._buffer_projections += projection_stop - projection_start
        self._buffer_leaves += leaf_stop - leaf_start
        self._buffer_bytes += sum(summary.estimated_bytes for summary in summaries)
        self._buffer_excluded += sum(
            summary.excluded_integrity_subtrees for summary in summaries)
        self._buffer_oversized += sum(oversized[start:stop])

    def write_batch(self, batch: EncodedStructureBatch,
                    index_maps: CatalogIndexMaps) -> tuple[RecordLocation, ...]:
        if self._finished is not None:
            raise ValueError("structure storage is already finished")
        summaries = batch.record_summaries
        if not summaries:
            raise ValueError("structure batch is empty")
        if (batch.record_table.schema != RECORD_SCHEMA or
                batch.projection_table.schema != PROJECTION_SCHEMA or
                batch.source_leaf_table.schema != SOURCE_LEAF_SCHEMA):
            raise ValueError("structure batch table schema differs")
        projection_starts = [0]
        leaf_starts = [0]
        for summary in summaries:
            projection_starts.append(
                projection_starts[-1] + _nonnegative(summary.projection_count, "projection_count"))
            leaf_starts.append(
                leaf_starts[-1] + _nonnegative(summary.source_leaf_count, "source_leaf_count"))
            _nonnegative(summary.estimated_bytes, "estimated_bytes")
        if (batch.record_table.num_rows != len(summaries) or
                batch.projection_table.num_rows != projection_starts[-1] or
                batch.source_leaf_table.num_rows != leaf_starts[-1]):
            raise ValueError("structure batch table counts differ from record summaries")
        projections = _mapped_index_column(
            batch.projection_table, index_name="normalized_index",
            id_name="normalized_id",
            local_rows=batch.catalog_fragment.normalized_rows,
            global_indexes=index_maps.normalized_indices)
        leaves = _mapped_index_column(
            batch.source_leaf_table, index_name="source_index",
            id_name="source_field_id",
            local_rows=batch.catalog_fragment.source_rows,
            global_indexes=index_maps.source_indices)
        batch = EncodedStructureBatch(
            batch.catalog_fragment, batch.record_table, projections, leaves, summaries)
        oversized = tuple(
            summary.estimated_bytes > self.target_bytes or
            summary.source_leaf_count > self.max_leaves for summary in summaries)
        record_table = _replace_column(
            batch.record_table, "oversized_record", pa.array(oversized, type=pa.bool_()))
        locations = []
        segment_start = 0
        buffered_records = self._buffer_records
        buffered_bytes = self._buffer_bytes
        buffered_leaves = self._buffer_leaves
        for position, (summary, is_oversized) in enumerate(zip(summaries, oversized)):
            self._check_order(summary)
            if buffered_records and (
                    buffered_bytes + summary.estimated_bytes > self.target_bytes or
                    buffered_leaves + summary.source_leaf_count > self.max_leaves):
                self._append_segment(batch, record_table, segment_start, position,
                                     projection_starts, leaf_starts, oversized)
                self._flush()
                segment_start = position
                buffered_records = buffered_bytes = buffered_leaves = 0
            locations.append(RecordLocation(self._next_shard_id, buffered_records))
            buffered_records += 1
            buffered_bytes += summary.estimated_bytes
            buffered_leaves += summary.source_leaf_count
            self._last_declaration = summary.declaration_index
            self._last_ordinal = summary.record_ordinal
            self._last_line = summary.source_line
            if (is_oversized or buffered_bytes >= self.target_bytes or
                    buffered_leaves >= self.max_leaves):
                self._append_segment(batch, record_table, segment_start, position + 1,
                                     projection_starts, leaf_starts, oversized)
                self._flush()
                segment_start = position + 1
                buffered_records = buffered_bytes = buffered_leaves = 0
        self._append_segment(batch, record_table, segment_start, len(summaries),
                             projection_starts, leaf_starts, oversized)
        return tuple(locations)

    def _flush(self) -> None:
        if not self._buffer_records:
            return
        shard_id = self._next_shard_id
        shard_dir = self.shards_root / f"part-{shard_id:06d}"
        tables = (
            pa.concat_tables(self._record_tables),
            pa.concat_tables(self._projection_tables),
            pa.concat_tables(self._source_tables),
        )
        if (tuple(table.num_rows for table in tables) !=
                (self._buffer_records, self._buffer_projections, self._buffer_leaves) or
                self._buffer_first_declaration is None or
                self._buffer_last_declaration is None):
            raise ValueError("structure shard tables differ from buffered record counts")
        shard_dir.mkdir(exist_ok=False)
        dictionaries = (
            ["effective_family", "root_kind", "record_tick_type"],
            ["normalized_id", "value_type", "value_status", "structure_status",
             "binding_slot"],
            ["source_field_id", "field_path", "raw_value_type", "identity_status",
             "time_role", "endpoint_role", "source_branch", "source_subtype"],
        )
        for name, table, dictionary_columns in zip(_TABLE_NAMES, tables, dictionaries):
            pq.write_table(table, shard_dir / name, compression="zstd",
                           use_dictionary=dictionary_columns)
        summary = ShardSummary(
            shard_id=shard_id,
            relative_path=shard_dir.relative_to(self.root).as_posix(),
            records=self._buffer_records,
            projections=self._buffer_projections,
            source_leaves=self._buffer_leaves,
            first_declaration_index=self._buffer_first_declaration,
            last_declaration_index=self._buffer_last_declaration,
            excluded_integrity_subtrees=self._buffer_excluded,
            oversized_records=self._buffer_oversized,
            compressed_bytes=sum((shard_dir / name).stat().st_size for name in _TABLE_NAMES),
        )
        self._shard_summaries.append(summary)
        self._next_shard_id += 1
        self._record_tables = []
        self._projection_tables = []
        self._source_tables = []
        self._buffer_records = self._buffer_projections = self._buffer_leaves = 0
        self._buffer_bytes = 0
        self._buffer_first_declaration = self._buffer_last_declaration = None
        self._buffer_excluded = self._buffer_oversized = 0

    def finish(self) -> StorageSummary:
        if self._finished is None:
            self._flush()
            shards = tuple(self._shard_summaries)
            self._finished = StorageSummary(
                shards=shards,
                records=sum(shard.records for shard in shards),
                projections=sum(shard.projections for shard in shards),
                source_leaves=sum(shard.source_leaves for shard in shards),
                excluded_integrity_subtrees=sum(
                    shard.excluded_integrity_subtrees for shard in shards),
                oversized_records=sum(shard.oversized_records for shard in shards),
            )
        return self._finished


def _shard_dirs(root: Path) -> list[tuple[int, Path]]:
    shards_root = Path(root) / "shards"
    if not shards_root.is_dir():
        raise FileNotFoundError(shards_root)
    result = []
    for path in shards_root.iterdir():
        if not path.is_dir() or not path.name.startswith("part-") or not path.name[5:].isdigit():
            raise ValueError(f"unexpected structure shard entry: {path}")
        result.append((int(path.name[5:]), path))
    result.sort()
    if [shard_id for shard_id, _ in result] != list(range(len(result))):
        raise ValueError("structure shard IDs are not consecutive")
    return result


def _decoded_row(row: dict[str, Any]) -> dict[str, Any]:
    decoded = {}
    for name, value in row.items():
        if name.endswith("_json"):
            decoded[name[:-5]] = json.loads(value) if value is not None else None
        else:
            decoded[name] = value
    return decoded


def _check_inspected_value(row: dict[str, Any], *, original: bool) -> None:
    key = "raw_value" if original else "normalized_value"
    type_key = "raw_value_type" if original else "value_type"
    value_type = row[type_key]
    if value_type is None:
        declared_gap = (not original and is_declared_implementation_sources_gap(
            row["normalized_id"], value_type,
            row["value_status"], row["structure_status"],
        ))
        if (original or row[key] is not None or
                row["structure_status"] != "structure_conflict" and not declared_gap):
            raise ValueError(f"stored {key} has no valid explicit type")
    elif value_type != _value_type(row[key]):
        raise ValueError(f"stored {key} type differs from JSON value")
    if original and value_type in {"empty_list", "empty_object"} and row[key]:
        raise ValueError(f"stored {key} is a nonempty source container")


def _check_decoded_tokens(tokens: Any, name: str) -> None:
    if not isinstance(tokens, list) or any(
            type(token) not in (str, int) or
            (type(token) is int and token < 0) for token in tokens):
        raise ValueError(f"stored {name} loses JSON key/index types")


def inspect_structure_record(root: Path, declaration_index: int,
                             record_ordinal: int | None = None, *,
                             source_line: int | None = None,
                             normalized_index: int | None = None,
                             source_index: int | None = None,
                             shard_id: int | None = None) -> dict[str, Any]:
    """Read one record and its typed values; optional field indexes narrow the view."""
    _nonnegative(declaration_index, "declaration_index")
    if record_ordinal is not None:
        _nonnegative(record_ordinal, "record_ordinal")
    if source_line is not None:
        if type(source_line) is not int or source_line < 1:
            raise ValueError(f"source_line is not a positive physical line: {source_line!r}")
    if record_ordinal is None and source_line is None:
        raise ValueError("record_ordinal or source_line is required")
    if normalized_index is not None:
        _nonnegative(normalized_index, "normalized_index")
    if source_index is not None:
        _nonnegative(source_index, "source_index")
    dirs = _shard_dirs(Path(root))
    if shard_id is not None:
        _nonnegative(shard_id, "shard_id")
        dirs = [(index, path) for index, path in dirs if index == shard_id]
        if not dirs:
            raise KeyError(f"structure shard is absent: {shard_id}")
    matches: list[tuple[int, Path, dict[str, Any]]] = []
    filters = [("declaration_index", "=", declaration_index)]
    if record_ordinal is not None:
        filters.append(("record_ordinal", "=", record_ordinal))
    if source_line is not None:
        filters.append(("source_line", "=", source_line))
    for index, path in dirs:
        rows = pq.read_table(path / "records.parquet", filters=filters).to_pylist()
        matches.extend((index, path, row) for row in rows)
    if len(matches) != 1:
        raise KeyError(f"expected one structural source record, found {len(matches)}: "
                       f"{declaration_index}:{source_line}:{record_ordinal}")
    index, path, record = matches[0]
    if source_line is not None and record["source_line"] != source_line:
        raise ValueError(f"source line differs: {declaration_index}:{record_ordinal}")
    record_index = _nonnegative(record["record_index"], "record_index")
    projection_start = _nonnegative(record["projection_start"], "projection_start")
    projection_count, leaf_count = _record_child_counts(record)
    projection_end = projection_start + projection_count
    leaf_start = _nonnegative(record["source_leaf_start"], "source_leaf_start")
    leaf_end = leaf_start + leaf_count
    projection_rows = (pq.read_table(
        path / "projections.parquet",
        filters=[("projection_index", ">=", projection_start),
                 ("projection_index", "<", projection_end)]).to_pylist()
        if projection_end > projection_start else [])
    if len(projection_rows) != projection_count:
        raise ValueError("record projection count differs from Parquet")
    leaf_rows = (pq.read_table(
        path / "source_leaves.parquet",
        filters=[("source_leaf_index", ">=", leaf_start),
                 ("source_leaf_index", "<", leaf_end)]).to_pylist()
        if leaf_end > leaf_start else [])
    if len(leaf_rows) != leaf_count:
        raise ValueError("record source leaf count differs from Parquet")
    leaves_by_projection: dict[int, list[dict[str, Any]]] = {}
    for raw_leaf in leaf_rows:
        leaf = _decoded_row(raw_leaf)
        _check_inspected_value(leaf, original=True)
        for token_name in ("source_tokens", "input_tokens"):
            _check_decoded_tokens(leaf[token_name], token_name)
        leaves_by_projection.setdefault(leaf["projection_index"], []).append(leaf)
    selected = []
    expected_leaf = leaf_start
    for offset, raw_projection in enumerate(projection_rows):
        projection = _decoded_row(raw_projection)
        projection_index = projection_start + offset
        _check_projection_reference(
            projection, projection_index, record_index,
            projection_start, projection_end)
        source_end = _checked_projection_leaf_end(
            projection, expected_leaf, leaf_start, leaf_end)
        _check_inspected_value(projection, original=False)
        for token_name in ("envelope_tokens", "payload_tokens"):
            _check_decoded_tokens(projection[token_name], token_name)
        sources = leaves_by_projection.get(projection_index, [])
        if len(sources) != projection["source_leaf_count"]:
            raise ValueError("projection source leaf count differs from Parquet")
        for source_offset, leaf in enumerate(sources):
            _check_source_leaf_reference(
                leaf, expected_leaf + source_offset, projection_index,
                expected_leaf, source_end)
        expected_leaf = source_end
        if (normalized_index is not None and
                projection["normalized_index"] != normalized_index):
            continue
        if source_index is not None and not any(
                leaf["source_index"] == source_index for leaf in sources):
            continue
        selected.append({"projection": projection, "source_leaves": sources})
    if expected_leaf != leaf_end:
        raise ValueError("record and projection source leaf counts differ")
    decoded_record = _decoded_row(record)
    if decoded_record["record_tick_present"]:
        if _json_type(decoded_record["record_tick"]) != decoded_record["record_tick_type"]:
            raise ValueError("stored record tick type differs from JSON value")
    elif decoded_record["record_tick"] is not None or decoded_record["record_tick_type"] is not None:
        raise ValueError("absent record tick carries a value or type")
    return {"shard_id": index, "record": decoded_record, "projections": selected}


def _rows(parquet: pq.ParquetFile, columns: list[str]):
    for batch in parquet.iter_batches(columns=columns, batch_size=8192):
        yield from batch.to_pylist()


def _int_batch(column: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Keep nullable Parquet integers in Arrow until the batch comparison."""
    missing = pc.is_null(column).to_numpy(zero_copy_only=False)
    values = pc.fill_null(column, 0).to_numpy(zero_copy_only=False)
    return values, missing


def _allowed_batch(column: pa.Array, allowed: pa.Array) -> np.ndarray:
    return pc.fill_null(pc.is_in(column, value_set=allowed), False).to_numpy(
        zero_copy_only=False)


def _check_batch_field_indices(ids: pa.Array, indices: np.ndarray,
                               seen: dict[str | None, int], *,
                               first_error: str, repeated_error: str) -> None:
    """Apply the existing first-observation index rule once per distinct batch ID."""
    encoded = pc.dictionary_encode(ids)
    codes = pc.fill_null(encoded.indices, -1).to_numpy(zero_copy_only=False)
    names = encoded.dictionary.to_pylist()
    distinct, first_positions = np.unique(codes, return_index=True)
    expected_by_code = np.empty(len(names) + 1, dtype=np.int64)
    first_by_code = np.empty(len(names) + 1, dtype=np.int64)
    pending: dict[str | None, int] = {}
    for position in np.argsort(first_positions):
        code = int(distinct[position])
        field_id = None if code == -1 else names[code]
        if field_id in seen:
            expected = seen[field_id]
        elif field_id in pending:
            expected = pending[field_id]
        else:
            expected = len(seen) + len(pending)
            pending[field_id] = expected
        expected_by_code[code + 1] = expected
        first_by_code[code + 1] = first_positions[position]
    bad = np.flatnonzero(indices != expected_by_code[codes + 1])
    if bad.size:
        row = int(bad[0])
        code = int(codes[row])
        field_id = None if code == -1 else names[code]
        if field_id in pending and row == first_by_code[code + 1]:
            raise ValueError(first_error)
        raise ValueError(repeated_error)
    seen.update(pending)


def validate_structure_storage(root: Path) -> StorageSummary:
    """Stream narrow columns to verify shard order, references, and type labels.

    This does not decode every JSON value or re-read the original source corpus.
    Use ``inspect_structure_record`` and the source locator for targeted value
    round trips.
    """
    root = Path(root)
    summaries: list[ShardSummary] = []
    normalized_indices: dict[str | None, int] = {}
    source_indices: dict[str | None, int] = {}
    last_declaration: int | None = None
    last_ordinal: int | None = None
    last_line: int | None = None
    allowed_value_types = pa.array(sorted(_VALUE_TYPES), type=pa.string())
    allowed_structure_statuses = pa.array(sorted(_STRUCTURE_STATUSES), type=pa.string())
    for shard_id, shard_dir in _shard_dirs(root):
        if {item.name for item in shard_dir.iterdir()} != set(_TABLE_NAMES):
            raise ValueError(f"structure shard has missing or extra tables: {shard_dir}")
        record_file, projection_file, leaf_file = (
            shard_dir / name for name in _TABLE_NAMES)
        record_parquet = pq.ParquetFile(record_file)
        projection_parquet = pq.ParquetFile(projection_file)
        leaf_parquet = pq.ParquetFile(leaf_file)
        if record_parquet.schema_arrow != RECORD_SCHEMA:
            raise ValueError(f"record schema differs: {record_file}")
        if projection_parquet.schema_arrow != PROJECTION_SCHEMA:
            raise ValueError(f"projection schema differs: {projection_file}")
        if leaf_parquet.schema_arrow != SOURCE_LEAF_SCHEMA:
            raise ValueError(f"source leaf schema differs: {leaf_file}")
        record_ranges: list[tuple[int, int, int, int, int]] = []
        first_declaration: int | None = None
        last_shard_declaration: int | None = None
        record_count = projection_count = leaf_count = 0
        excluded_count = oversized_count = 0
        expected_projection = expected_leaf = 0
        for row in _rows(record_parquet, [
                "record_index", "declaration_index", "source_line", "record_ordinal",
                "effective_family", "root_kind", "record_tick_present",
                "record_tick_type", "excluded_integrity_subtrees", "projection_start",
                "projection_count", "source_leaf_start", "source_leaf_count",
                "oversized_record"]):
            if row["record_index"] != record_count:
                raise ValueError(f"record index is not consecutive: {shard_dir}")
            declaration = _nonnegative(row["declaration_index"], "declaration_index")
            ordinal = _nonnegative(row["record_ordinal"], "record_ordinal")
            if (last_declaration is None or declaration > last_declaration):
                if ordinal != 0:
                    raise ValueError("first record ordinal is not zero")
            elif declaration == last_declaration:
                if ordinal != last_ordinal + 1:
                    raise ValueError("record ordinal is not consecutive")
                if (row["source_line"] is None or last_line is None or
                        row["source_line"] <= last_line):
                    raise ValueError("physical JSONL lines are not increasing")
            else:
                raise ValueError("source declaration order is reversed")
            if row["source_line"] is not None:
                if type(row["source_line"]) is not int or row["source_line"] < 1:
                    raise ValueError("physical JSONL line is invalid")
            elif ordinal != 0:
                raise ValueError("JSON root has a nonzero ordinal")
            if not row["effective_family"] or row["root_kind"] != "object":
                raise ValueError("record family or root type is invalid")
            if (row["record_tick_present"] and row["record_tick_type"] not in
                    {"null", "bool", "int", "float", "string", "array", "object"}):
                raise ValueError("record tick has no explicit JSON type")
            if not row["record_tick_present"] and row["record_tick_type"] is not None:
                raise ValueError("absent record tick has a type")
            if (row["projection_start"] != expected_projection or
                    row["source_leaf_start"] != expected_leaf):
                raise ValueError("record table ranges are not consecutive")
            p_count, l_count = _record_child_counts(row)
            record_ranges.append((record_count, expected_projection,
                                  expected_projection + p_count, expected_leaf,
                                  expected_leaf + l_count))
            expected_projection += p_count
            expected_leaf += l_count
            excluded_count += _nonnegative(
                row["excluded_integrity_subtrees"], "excluded_integrity_subtrees")
            if row["oversized_record"]:
                oversized_count += 1
            first_declaration = declaration if first_declaration is None else first_declaration
            last_shard_declaration = declaration
            last_declaration, last_ordinal, last_line = (
                declaration, ordinal, row["source_line"])
            record_count += 1
        if not record_count or (oversized_count and record_count != 1):
            raise ValueError(f"empty or mixed oversized shard: {shard_dir}")

        if (expected_projection > np.iinfo(np.int64).max or
                expected_leaf > np.iinfo(np.int64).max):
            raise ValueError("record and projection counts differ")
        record_bounds = np.asarray(record_ranges, dtype=np.int64)
        projection_leaf_end_batches: list[np.ndarray] = []
        expected_source_leaf = 0
        for batch in projection_parquet.iter_batches(columns=[
                "projection_index", "record_index", "normalized_index",
                "normalized_id", "value_type", "value_status", "structure_status",
                "source_leaf_start", "source_leaf_count"], batch_size=8192):
            positions = np.arange(projection_count, projection_count + batch.num_rows,
                                  dtype=np.int64)
            owners = np.searchsorted(record_bounds[:, 2], positions, side="right")
            if np.any(owners == record_count):
                raise ValueError("projection has no record range")
            stored_projection, missing_projection = _int_batch(batch.column(0))
            stored_record, missing_record = _int_batch(batch.column(1))
            if np.any(missing_projection | missing_record |
                      (stored_projection != positions) | (stored_record != owners) |
                      (positions < record_bounds[owners, 1]) |
                      (positions >= record_bounds[owners, 2])):
                raise ValueError("projection does not reference its record")
            normalized_index, missing_index = _int_batch(batch.column(2))
            invalid_index = missing_index | (normalized_index < 0)
            if np.any(invalid_index):
                first = int(np.flatnonzero(invalid_index)[0])
                _nonnegative(batch.column(2)[first].as_py(), "normalized_index")
            _check_batch_field_indices(
                batch.column(3), normalized_index, normalized_indices,
                first_error="normalized indexes are not first-seen consecutive",
                repeated_error="one normalized ID has differing indexes")
            value_type = batch.column(4)
            null_type = pc.is_null(value_type).to_numpy(zero_copy_only=False)
            if np.any(~_allowed_batch(value_type, allowed_value_types) & ~null_type):
                raise ValueError("normalized value has invalid type label")
            structure_status = batch.column(6)
            conflict = pc.fill_null(
                pc.equal(structure_status, "structure_conflict"), False).to_numpy(
                    zero_copy_only=False)
            declared_gap = (
                pc.fill_null(pc.equal(batch.column(3),
                                      IMPLEMENTATION_SOURCES_GAP_NORMALIZED_ID), False)
                .to_numpy(zero_copy_only=False)
                & pc.fill_null(pc.equal(batch.column(5), "empty_object"), False)
                .to_numpy(zero_copy_only=False)
                & pc.fill_null(pc.equal(structure_status, "declared_only"), False)
                .to_numpy(zero_copy_only=False)
            )
            if np.any(null_type & ~conflict & ~declared_gap):
                raise ValueError("untyped normalized value lacks a declared gap")
            nonempty_status = pc.fill_null(
                pc.not_equal(batch.column(5), ""), False).to_numpy(
                    zero_copy_only=False)
            if np.any(~nonempty_status |
                      ~_allowed_batch(structure_status, allowed_structure_statuses)):
                raise ValueError("projection status is missing or invalid")
            source_starts, missing_start = _int_batch(batch.column(7))
            source_counts, missing_count = _int_batch(batch.column(8))
            invalid_count = missing_count | (source_counts < 0)
            if np.any(invalid_count):
                first = int(np.flatnonzero(invalid_count)[0])
                _nonnegative(batch.column(8)[first].as_py(), "source_leaf_count")
            if np.any(source_counts == 0):
                raise ValueError("projection has no original source leaf")
            prefix = np.cumsum(source_counts, dtype=np.int64)
            if (prefix[-1] < 0 or np.any(prefix[1:] < prefix[:-1]) or
                    expected_source_leaf > np.iinfo(np.int64).max - int(prefix[-1])):
                raise ValueError("projection source leaves fall outside their record")
            expected_starts = expected_source_leaf + prefix - source_counts
            if np.any(missing_start | (source_starts != expected_starts)):
                raise ValueError("projection source leaf ranges are not consecutive")
            source_ends = expected_source_leaf + prefix
            if np.any((expected_starts < record_bounds[owners, 3]) |
                      (source_ends > record_bounds[owners, 4])):
                raise ValueError("projection source leaves fall outside their record")
            projection_leaf_end_batches.append(source_ends)
            expected_source_leaf = int(source_ends[-1])
            projection_count += batch.num_rows
        if projection_count != expected_projection or expected_source_leaf != expected_leaf:
            raise ValueError("record and projection counts differ")

        projection_leaf_ends = (np.concatenate(projection_leaf_end_batches)
                                if projection_leaf_end_batches else np.empty(0, dtype=np.int64))
        for batch in leaf_parquet.iter_batches(columns=[
                "source_leaf_index", "projection_index", "source_index",
                "source_field_id", "raw_value_type"], batch_size=8192):
            positions = np.arange(leaf_count, leaf_count + batch.num_rows,
                                  dtype=np.int64)
            owners = np.searchsorted(projection_leaf_ends, positions, side="right")
            if np.any(owners == projection_count):
                raise ValueError("source leaf has no projection range")
            stored_leaf, missing_leaf = _int_batch(batch.column(0))
            stored_projection, missing_projection = _int_batch(batch.column(1))
            starts = np.where(owners == 0, 0,
                              projection_leaf_ends[np.maximum(owners - 1, 0)])
            if np.any(missing_leaf | missing_projection |
                      (stored_leaf != positions) | (stored_projection != owners) |
                      (positions < starts) |
                      (positions >= projection_leaf_ends[owners])):
                raise ValueError("source leaf does not reference its projection")
            source_index, missing_index = _int_batch(batch.column(2))
            invalid_index = missing_index | (source_index < 0)
            if np.any(invalid_index):
                first = int(np.flatnonzero(invalid_index)[0])
                _nonnegative(batch.column(2)[first].as_py(), "source_index")
            _check_batch_field_indices(
                batch.column(3), source_index, source_indices,
                first_error="source indexes are not first-seen consecutive",
                repeated_error="one source field has differing indexes")
            if np.any(~_allowed_batch(batch.column(4), allowed_value_types)):
                raise ValueError("source leaf has invalid explicit type label")
            leaf_count += batch.num_rows
        if leaf_count != expected_leaf:
            raise ValueError("projection and source leaf counts differ")
        actual_counts = tuple(parquet.metadata.num_rows for parquet in (
            record_parquet, projection_parquet, leaf_parquet))
        if actual_counts != (record_count, projection_count, leaf_count):
            raise ValueError("Parquet metadata counts differ from streamed rows")
        summaries.append(ShardSummary(
            shard_id=shard_id,
            relative_path=shard_dir.relative_to(root).as_posix(),
            records=record_count,
            projections=projection_count,
            source_leaves=leaf_count,
            first_declaration_index=first_declaration,
            last_declaration_index=last_shard_declaration,
            excluded_integrity_subtrees=excluded_count,
            oversized_records=oversized_count,
            compressed_bytes=sum(path.stat().st_size for path in (
                record_file, projection_file, leaf_file)),
        ))
    shards = tuple(summaries)
    return StorageSummary(
        shards=shards,
        records=sum(shard.records for shard in shards),
        projections=sum(shard.projections for shard in shards),
        source_leaves=sum(shard.source_leaves for shard in shards),
        excluded_integrity_subtrees=sum(
            shard.excluded_integrity_subtrees for shard in shards),
        oversized_records=sum(shard.oversized_records for shard in shards),
    )
