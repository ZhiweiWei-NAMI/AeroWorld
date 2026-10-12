"""Recover observed catalog fragments and progress from completed structure shards.

Only narrow Parquet columns are read. Arrow joins and hash aggregations consume
projection and leaf rows; Python handles records and aggregate groups.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from Dataset.world_model.graph.field_structure import is_declared_implementation_sources_gap
from Dataset.world_model.graph.structure_catalog import (
    CatalogFragment, StructureCatalog, _shape, _validate_fragment,
)
from Dataset.world_model.graph.structure_storage import (
    PROJECTION_SCHEMA, RECORD_SCHEMA, SOURCE_LEAF_SCHEMA,
    ShardSummary, StorageResumePoint, _STRUCTURE_STATUSES, _VALUE_TYPES,
)


_RECORD_COLUMNS = (
    "record_index", "declaration_index", "record_ordinal", "source_line",
    "effective_family", "excluded_integrity_subtrees", "projection_start",
    "projection_count", "source_leaf_start", "source_leaf_count",
    "oversized_record",
)
_PROJECTION_COLUMNS = (
    "projection_index", "record_index", "normalized_index", "normalized_id",
    "value_type", "value_status", "structure_status", "binding_slot",
    "declared_shape_json", "observed_shape_json", "normalized_shape_json",
    "source_leaf_start", "source_leaf_count",
)
_LEAF_COLUMNS = (
    "source_leaf_index", "projection_index", "source_index",
    "source_field_id", "field_path", "raw_value_type", "source_branch",
    "source_subtype",
)
_PROJECTION_GROUP_KEYS = (
    "normalized_id", "normalized_index", "declared_shape_json", "value_type",
    "value_status", "structure_status", "observed_shape_json",
    "normalized_shape_json",
)
_LEAF_GROUP_KEYS = (
    "source_field_id", "source_index", "effective_family", "field_path",
    "source_branch", "source_subtype", "normalized_id", "binding_slot",
    "raw_value_type", "value_type", "value_status", "structure_status",
    "observed_shape_json", "normalized_shape_json",
)


@dataclass(frozen=True)
class RecoveredDeclaration:
    declaration_index: int
    records: int
    projections: int
    source_leaves: int
    structure_conflict_projections: int
    excluded_integrity_subtrees: int
    effective_families: tuple[str, ...]
    first_record_index: int
    last_record_index: int
    first_record_ordinal: int
    last_record_ordinal: int
    first_source_line: int | None
    last_source_line: int | None


@dataclass(frozen=True)
class RecoveredStructureShard:
    catalog_fragment: CatalogFragment
    source_global_indices: tuple[int, ...]
    normalized_global_indices: tuple[int, ...]
    declarations: tuple[RecoveredDeclaration, ...]
    resume_point: StorageResumePoint
    shard_summary: ShardSummary


def _nonnegative(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} is not a nonnegative integer: {value!r}")
    return value


def _shape_from_json(value: bytes | None) -> tuple[int | None, ...] | None:
    if value is None:
        return None
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        raise ValueError(f"structural shape is not a JSON array: {decoded!r}")
    result = tuple(decoded)
    _shape(result)
    return result


def _all_true(values: pa.Array | pa.ChunkedArray) -> bool:
    return len(values) == 0 or pc.all(pc.fill_null(values, False)).as_py() is True


def _dense_index(table: pa.Table, name: str) -> None:
    column = table.column(name)
    count = table.num_rows
    if count and (
            pc.min(column).as_py() != 0 or
            pc.max(column).as_py() != count - 1 or
            pc.count_distinct(column).as_py() != count):
        raise ValueError(f"{name} is not consecutive")


def _read_shard_tables(
        shard_dir: Path, use_threads: bool,
        ) -> tuple[pa.Table, pa.Table, pa.Table, int]:
    names = ("records.parquet", "projections.parquet", "source_leaves.parquet")
    if {path.name for path in shard_dir.iterdir()} != set(names):
        raise ValueError(f"structure shard has missing or extra tables: {shard_dir}")
    files = tuple(shard_dir / name for name in names)
    parquet = tuple(pq.ParquetFile(path) for path in files)
    for path, reader, schema in zip(
            files, parquet, (RECORD_SCHEMA, PROJECTION_SCHEMA, SOURCE_LEAF_SCHEMA)):
        if reader.schema_arrow != schema:
            raise ValueError(f"structure table schema differs: {path}")
    tables = tuple(pq.read_table(path, columns=list(columns), use_threads=use_threads)
                   for path, columns in zip(
        files, (_RECORD_COLUMNS, _PROJECTION_COLUMNS, _LEAF_COLUMNS)))
    for path, reader, table in zip(files, parquet, tables):
        if reader.metadata.num_rows != table.num_rows:
            raise ValueError(f"Parquet row count differs: {path}")
    return (*tables, sum(path.stat().st_size for path in files))


def _record_progress(
        records: pa.Table, projection_count: int, leaf_count: int,
        shard_id: int, relative_path: str, compressed_bytes: int,
        ) -> tuple[list[dict[str, Any]], StorageResumePoint, ShardSummary]:
    if not records.num_rows:
        raise ValueError("structure shard has no records")
    declaration_rows: dict[int, dict[str, Any]] = {}
    previous_declaration: int | None = None
    previous_ordinal: int | None = None
    previous_line: int | None = None
    projection_cursor = leaf_cursor = excluded = oversized = 0
    record_rows = records.to_pylist()
    for expected_index, row in enumerate(record_rows):
        if row["record_index"] != expected_index:
            raise ValueError("record index is not consecutive")
        declaration = _nonnegative(row["declaration_index"], "declaration_index")
        ordinal = _nonnegative(row["record_ordinal"], "record_ordinal")
        line = row["source_line"]
        if line is not None and (type(line) is not int or line < 1):
            raise ValueError("record has an invalid physical source line")
        if previous_declaration is None:
            if shard_id == 0 and ordinal != 0:
                raise ValueError("first source record ordinal is not zero")
        elif declaration > previous_declaration:
            if ordinal != 0:
                raise ValueError("first source record ordinal is not zero")
        elif declaration == previous_declaration:
            if (ordinal != previous_ordinal + 1 or line is None or
                    previous_line is None or line <= previous_line):
                raise ValueError("source record order differs")
        else:
            raise ValueError("source declaration order is reversed")
        if line is None and ordinal != 0:
            raise ValueError("JSON root has a nonzero record ordinal")
        family = row["effective_family"]
        if not isinstance(family, str) or not family:
            raise ValueError("record has no effective family")
        projections = _nonnegative(row["projection_count"], "projection_count")
        leaves = _nonnegative(row["source_leaf_count"], "source_leaf_count")
        if (row["projection_start"] != projection_cursor or
                row["source_leaf_start"] != leaf_cursor or
                not projections and leaves):
            raise ValueError("record child ranges differ")
        projection_cursor += projections
        leaf_cursor += leaves
        excluded_item = _nonnegative(
            row["excluded_integrity_subtrees"], "excluded_integrity_subtrees")
        excluded += excluded_item
        if type(row["oversized_record"]) is not bool:
            raise ValueError("record oversized flag is not boolean")
        oversized += row["oversized_record"]
        item = declaration_rows.get(declaration)
        if item is None:
            item = {
                "declaration_index": declaration, "records": 0, "projections": 0,
                "source_leaves": 0, "structure_conflict_projections": 0,
                "excluded_integrity_subtrees": 0, "effective_families": set(),
                "first_record_index": expected_index, "last_record_index": expected_index,
                "first_record_ordinal": ordinal, "last_record_ordinal": ordinal,
                "first_source_line": line, "last_source_line": line,
            }
            declaration_rows[declaration] = item
        item["records"] += 1
        item["projections"] += projections
        item["source_leaves"] += leaves
        item["excluded_integrity_subtrees"] += excluded_item
        item["effective_families"].add(family)
        item["last_record_index"] = expected_index
        item["last_record_ordinal"] = ordinal
        item["last_source_line"] = line
        previous_declaration, previous_ordinal, previous_line = declaration, ordinal, line
    if projection_cursor != projection_count or leaf_cursor != leaf_count:
        raise ValueError("record and child table counts differ")
    if oversized and records.num_rows != 1:
        raise ValueError("oversized record is mixed with other records")
    assert previous_declaration is not None and previous_ordinal is not None
    point = StorageResumePoint(
        shard_id, records.num_rows - 1, previous_declaration,
        previous_ordinal, previous_line,
    )
    summary = ShardSummary(
        shard_id, relative_path, records.num_rows, projection_count, leaf_count,
        record_rows[0]["declaration_index"], previous_declaration,
        excluded, oversized, compressed_bytes,
    )
    return list(declaration_rows.values()), point, summary


def _joined_projections(
        records: pa.Table, projections: pa.Table, use_threads: bool,
        ) -> pa.Table:
    record_refs = records.select((
        "record_index", "declaration_index", "effective_family", "projection_start",
        "projection_count", "source_leaf_start", "source_leaf_count",
    )).rename_columns((
        "record_index", "declaration_index", "effective_family", "projection_start",
        "projection_count", "record_leaf_start", "record_leaf_count",
    ))
    joined = projections.join(record_refs, keys="record_index",
                              join_type="left outer", use_threads=use_threads)
    if joined.num_rows != projections.num_rows:
        raise ValueError("projection record join changed row count")
    if not _all_true(pc.and_(
            pc.is_valid(joined["effective_family"]),
            pc.and_(
                pc.greater_equal(joined["projection_index"], joined["projection_start"]),
                pc.less(joined["projection_index"], pc.add_checked(
                    joined["projection_start"], joined["projection_count"]))))):
        raise ValueError("projection does not belong to its record")
    if not _all_true(pc.and_(
            pc.greater(joined["source_leaf_count"], 0),
            pc.and_(
                pc.greater_equal(joined["source_leaf_start"], joined["record_leaf_start"]),
                pc.less_equal(pc.add_checked(
                    joined["source_leaf_start"], joined["source_leaf_count"]),
                    pc.add_checked(joined["record_leaf_start"],
                                   joined["record_leaf_count"]))))):
        raise ValueError("projection source leaves fall outside their record")
    counts = joined.group_by("record_index", use_threads=use_threads).aggregate([
        ("projection_index", "count")])
    checked = records.select(("record_index", "projection_count")).join(
        counts, keys="record_index", join_type="left outer", use_threads=use_threads)
    if not _all_true(pc.equal(
            checked["projection_count"],
            pc.fill_null(checked["projection_index_count"], 0))):
        raise ValueError("record projection count differs")
    return joined


def _joined_leaves(
        leaves: pa.Table, projections: pa.Table, use_threads: bool,
        ) -> pa.Table:
    joined = leaves.join(projections.select((
        "projection_index", "record_index", "effective_family", "normalized_id",
        "binding_slot", "value_type", "value_status", "structure_status",
        "observed_shape_json", "normalized_shape_json", "source_leaf_start",
        "source_leaf_count",
    )), keys="projection_index", join_type="left outer", use_threads=use_threads)
    if joined.num_rows != leaves.num_rows:
        raise ValueError("source leaf projection join changed row count")
    if not _all_true(pc.and_(
            pc.is_valid(joined["normalized_id"]),
            pc.and_(
                pc.greater_equal(joined["source_leaf_index"], joined["source_leaf_start"]),
                pc.less(joined["source_leaf_index"], pc.add_checked(
                    joined["source_leaf_start"], joined["source_leaf_count"]))))):
        raise ValueError("source leaf does not belong to its projection")
    counts = joined.group_by("projection_index", use_threads=use_threads).aggregate([
        ("source_leaf_index", "count")])
    checked = projections.select(("projection_index", "source_leaf_count")).join(
        counts, keys="projection_index", join_type="left outer", use_threads=use_threads)
    if not _all_true(pc.equal(
            checked["source_leaf_count"],
            pc.fill_null(checked["source_leaf_index_count"], 0))):
        raise ValueError("projection source leaf count differs")
    duplicate = joined.group_by(
        ("projection_index", "source_field_id"), use_threads=use_threads).aggregate([
        ("source_leaf_index", "count")])
    if duplicate.num_rows and pc.max(duplicate["source_leaf_index_count"]).as_py() != 1:
        raise ValueError("projection repeats a source field ID")
    return joined


def _fragment_from_groups(
        projection_groups: pa.Table, leaf_groups: pa.Table,
        projection_count: int, leaf_count: int,
        ) -> tuple[CatalogFragment, tuple[int, ...], tuple[int, ...]]:
    normalized_meta: dict[str, tuple[int, tuple[int | None, ...] | None, int]] = {}
    source_meta: dict[str, tuple[int, str, str, str | None, str | None, int]] = {}
    normalized_id_by_index: dict[int, str] = {}
    source_id_by_index: dict[int, str] = {}
    projection_rows = projection_groups.to_pylist()
    leaf_rows = leaf_groups.to_pylist()
    if (sum(row["projection_index_count"] for row in projection_rows) != projection_count or
            sum(row["source_leaf_index_count"] for row in leaf_rows) != leaf_count):
        raise ValueError("catalog groups do not cover the shard")
    for row in projection_rows:
        field_id = row["normalized_id"]
        index = _nonnegative(row["normalized_index"], "normalized_index")
        first = _nonnegative(row["projection_index_min"], "projection_index_min")
        shape = _shape_from_json(row["declared_shape_json"])
        if (not isinstance(field_id, str) or not field_id.startswith("structural:") or
                row["value_type"] not in _VALUE_TYPES | {None} or
                row["value_type"] is None and row["structure_status"] != "structure_conflict"
                and not is_declared_implementation_sources_gap(
                    field_id, row["value_type"], row["value_status"],
                    row["structure_status"],
                ) or
                not isinstance(row["value_status"], str) or not row["value_status"] or
                row["structure_status"] not in _STRUCTURE_STATUSES):
            raise ValueError(f"invalid normalized observation: {field_id!r}")
        descriptor = (index, shape)
        previous = normalized_meta.get(field_id)
        if previous is not None and previous[:2] != descriptor:
            raise ValueError(f"normalized field description differs: {field_id}")
        other_id = normalized_id_by_index.get(index)
        if other_id is not None and other_id != field_id:
            raise ValueError(f"normalized index identifies two fields: {index}")
        normalized_id_by_index[index] = field_id
        normalized_meta[field_id] = (index, shape, min(previous[2], first)
                                     if previous is not None else first)
    for row in leaf_rows:
        field_id = row["source_field_id"]
        index = _nonnegative(row["source_index"], "source_index")
        first = _nonnegative(row["source_leaf_index_min"], "source_leaf_index_min")
        descriptor = (index, row["effective_family"], row["field_path"],
                      row["source_branch"], row["source_subtype"])
        if (not isinstance(field_id, str) or not field_id or
                row["raw_value_type"] not in _VALUE_TYPES or
                row["normalized_id"] not in normalized_meta):
            raise ValueError(f"invalid source observation: {field_id!r}")
        previous = source_meta.get(field_id)
        if previous is not None and previous[:5] != descriptor:
            raise ValueError(f"source field description differs: {field_id}")
        other_id = source_id_by_index.get(index)
        if other_id is not None and other_id != field_id:
            raise ValueError(f"source index identifies two fields: {index}")
        source_id_by_index[index] = field_id
        source_meta[field_id] = (*descriptor, min(previous[5], first)
                                 if previous is not None else first)

    catalog = StructureCatalog()
    source_global = []
    for field_id, (index, family, path, branch, subtype, _) in sorted(
            source_meta.items(), key=lambda item: item[1][5]):
        catalog._source(field_id, family, path, branch, subtype)
        source_global.append(index)
    normalized_global = []
    for field_id, (index, shape, _) in sorted(
            normalized_meta.items(), key=lambda item: item[1][2]):
        catalog._normalized(field_id, shape)
        normalized_global.append(index)

    for row in sorted(projection_rows, key=lambda item: item["projection_index_min"]):
        normalized = catalog._normalized_rows[catalog.normalized_index_by_id[
            row["normalized_id"]]]
        count = row["projection_index_count"]
        normalized["instance_count"] += count
        normalized["value_status_counts"][row["value_status"]] = (
            normalized["value_status_counts"].get(row["value_status"], 0) + count)
        normalized["structure_status_counts"][row["structure_status"]] = (
            normalized["structure_status_counts"].get(row["structure_status"], 0) + count)
        if row["value_type"] is not None:
            normalized["observed_value_types"].add(row["value_type"])
        normalized["observed_shapes"].add(_shape_from_json(row["observed_shape_json"]))
        normalized["normalized_shapes"].add(
            _shape_from_json(row["normalized_shape_json"]))

    for row in sorted(leaf_rows, key=lambda item: item["source_leaf_index_min"]):
        source = catalog._source_rows[catalog.source_index_by_id[row["source_field_id"]]]
        normalized = catalog._normalized_rows[catalog.normalized_index_by_id[
            row["normalized_id"]]]
        count = row["source_leaf_index_count"]
        source["instance_count"] += count
        source["observed_source_value_types"].add(row["raw_value_type"])
        normalized["source_indices"].add(source["source_index"])
        binding = catalog._binding(source, normalized, row["binding_slot"])
        binding["instance_count"] += count
        binding["observed_source_value_types"].add(row["raw_value_type"])
        if row["value_type"] is not None:
            binding["observed_value_types"].add(row["value_type"])
        binding["observed_value_statuses"].add(row["value_status"])
        binding["observed_structure_statuses"].add(row["structure_status"])
        binding["observed_shapes"].add(_shape_from_json(row["observed_shape_json"]))
        binding["normalized_shapes"].add(
            _shape_from_json(row["normalized_shape_json"]))
    fragment = catalog.export_fragment()
    _validate_fragment(fragment)
    return fragment, tuple(source_global), tuple(normalized_global)


def recover_structure_tables(
        records: pa.Table, projections: pa.Table, leaves: pa.Table,
        shard_id: int, relative_path: str, compressed_bytes: int, *,
        use_threads: bool = False,
        ) -> RecoveredStructureShard:
    """Recover one shard from in-memory tables using only structural columns.

    The returned indexes are aligned with fragment-local rows. Tables with
    local indexes therefore return local indexes in first-observation order.
    """
    shard_id = _nonnegative(shard_id, "shard_id")
    if type(use_threads) is not bool:
        raise TypeError("use_threads is not a boolean")
    records = records.select(_RECORD_COLUMNS)
    projections = projections.select(_PROJECTION_COLUMNS)
    leaves = leaves.select(_LEAF_COLUMNS)
    _dense_index(projections, "projection_index")
    _dense_index(leaves, "source_leaf_index")
    declaration_rows, resume_point, shard_summary = _record_progress(
        records, projections.num_rows, leaves.num_rows, shard_id,
        relative_path, compressed_bytes)
    projection_records = _joined_projections(records, projections, use_threads)
    joined_leaves = _joined_leaves(leaves, projection_records, use_threads)
    projection_groups = projections.group_by(
        _PROJECTION_GROUP_KEYS, use_threads=use_threads).aggregate([
        ("projection_index", "count"), ("projection_index", "min")])
    leaf_groups = joined_leaves.group_by(
        _LEAF_GROUP_KEYS, use_threads=use_threads).aggregate([
        ("source_leaf_index", "count"), ("source_leaf_index", "min")])
    fragment, source_global, normalized_global = _fragment_from_groups(
        projection_groups, leaf_groups, projections.num_rows, leaves.num_rows)
    conflicts = projection_records.filter(pc.equal(
        projection_records["structure_status"], "structure_conflict"))
    conflict_groups = conflicts.group_by(
        "declaration_index", use_threads=use_threads).aggregate([
        ("projection_index", "count")])
    by_declaration = {row["declaration_index"]: row for row in declaration_rows}
    for row in conflict_groups.to_pylist():
        by_declaration[row["declaration_index"]]["structure_conflict_projections"] = (
            row["projection_index_count"])
    declarations = []
    for row in declaration_rows:
        row["effective_families"] = tuple(sorted(row["effective_families"]))
        declarations.append(RecoveredDeclaration(**row))
    return RecoveredStructureShard(
        fragment, source_global, normalized_global,
        tuple(declarations), resume_point, shard_summary,
    )


def recover_structure_shard(
        root: Path, shard_id: int, *, use_threads: bool = False,
        ) -> RecoveredStructureShard:
    """Rebuild one shard's observed catalog rows without Python leaf iteration.

    The returned global indexes are aligned with fragment-local rows. A caller
    replaying shards in order must compare them with ``merge_fragment`` output.
    """
    shard_id = _nonnegative(shard_id, "shard_id")
    if type(use_threads) is not bool:
        raise TypeError("use_threads is not a boolean")
    root = Path(root)
    shard_dir = root / "shards" / f"part-{shard_id:06d}"
    records, projections, leaves, compressed_bytes = _read_shard_tables(
        shard_dir, use_threads)
    return recover_structure_tables(
        records, projections, leaves, shard_id,
        shard_dir.relative_to(root).as_posix(), compressed_bytes,
        use_threads=use_threads,
    )
