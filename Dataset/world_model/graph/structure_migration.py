"""Migrate the staged structural Parquet prefix and continue the source delivery."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError
from contextlib import nullcontext
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from Dataset.world_model.graph.contract import REPO, source_path
from Dataset.world_model.graph.structure_catalog import StructureCatalog, _validate_fragment
from Dataset.world_model.graph.structure_delivery import (
    FIELDS, INVENTORY, STAGED_FIELDS, STAGED_INVENTORY, STAGED_VALUES, VALUES,
    _accept_recovered_declaration, _check_resume_source, _declaration_row,
    _line, _resume_validated_prefix, _write_jsonl,
)
from Dataset.world_model.graph.structure_migration_transform import (
    MigratedShardTables, migrate_shard_tables,
)
from Dataset.world_model.graph.structure_sources import ordered_source_declarations
from Dataset.world_model.graph.structure_storage import (
    StorageResumePoint, StorageSummary, StructureStorageWriter, _mapped_index_column,
    _shard_dirs, validate_structure_storage,
)


MIGRATING = VALUES.with_name(VALUES.name + ".migrating")
BACKUP = VALUES.with_name(VALUES.name + ".pre_migration")
MAPPING = REPO / "design/belief/structure_migration_mapping.jsonl"
MAPPING_TMP = MAPPING.with_name(MAPPING.name + ".tmp")
_CHANGE_COLUMNS = frozenset({
    "normalized_id", "source_field_id", "field_path", "binding_slot",
    "declared_shape_json", "structure_status",
})
_PROJECTION_DICTIONARIES = [
    "normalized_id", "value_type", "value_status", "structure_status",
    "binding_slot",
]
_LEAF_DICTIONARIES = [
    "source_field_id", "field_path", "raw_value_type", "identity_status",
    "time_role", "endpoint_role", "source_branch", "source_subtype",
]


class _Progress:
    def __init__(self, started: float, total: int) -> None:
        self.started = started
        self.last_report = started
        self.total = total
        self.completed = 0
        self.records = 0
        self.projections = 0
        self.source_leaves = 0

    def report(self, phase: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_report < 60:
            return
        print(_line({
            "phase": phase,
            "shards_completed": self.completed,
            "shards_total": self.total,
            "records": self.records,
            "projections": self.projections,
            "source_leaves": self.source_leaves,
            "elapsed_seconds": round(now - self.started, 3),
        }).rstrip(), flush=True)
        self.last_report = now


class _Relations:
    def __init__(self, kind: str) -> None:
        if kind not in {"source", "normalized"}:
            raise ValueError(f"unknown migration relation kind: {kind}")
        self.kind = kind
        self.old_id_by_index: dict[int, str] = {}
        self.old_index_by_id: dict[str, int] = {}
        self.counts: dict[tuple[int, str, int, str], int] = {}
        self.instances = 0
        self.index_changed = 0

    def accept(self, relation: pa.Table, global_indices: tuple[int, ...],
               catalog_rows: list[dict[str, Any]], expected_instances: int) -> None:
        names = ("old_index", "old_id", "new_local_index", "new_id", "instance_count")
        if relation.column_names != list(names):
            raise ValueError(f"{self.kind} migration relation columns differ")
        if relation.num_rows:
            local = relation["new_local_index"]
            if pc.any(pc.is_null(local)).as_py():
                raise ValueError(f"{self.kind} migration has a missing local index")
            if pc.any(pc.greater_equal(local, len(global_indices))).as_py() or (
                    pc.any(pc.less(local, 0)).as_py()):
                raise ValueError(f"{self.kind} migration has an out-of-range local index")
            global_column = pc.take(pa.array(global_indices, type=pa.int64()), local)
            rows = relation.append_column("new_index", global_column).to_pylist()
        else:
            rows = []
        instance_total = 0
        for row in rows:
            old_index = row["old_index"]
            old_id = row["old_id"]
            new_index = row["new_index"]
            new_id = row["new_id"]
            count = row["instance_count"]
            if (type(old_index) is not int or old_index < 0 or
                    not isinstance(old_id, str) or not old_id or
                    type(new_index) is not int or
                    not isinstance(new_id, str) or not new_id or
                    type(count) is not int or count < 1):
                raise ValueError(f"{self.kind} migration has an invalid relation")
            index_name = self.kind + "_index"
            if (not 0 <= new_index < len(catalog_rows) or
                    catalog_rows[new_index][index_name] != new_index or
                    catalog_rows[new_index]["id"] != new_id):
                raise ValueError(f"{self.kind} migration index differs from catalog")
            seen_id = self.old_id_by_index.get(old_index)
            if seen_id is not None and seen_id != old_id:
                raise ValueError(f"one old {self.kind} index identifies multiple IDs")
            seen_index = self.old_index_by_id.get(old_id)
            if seen_index is not None and seen_index != old_index:
                raise ValueError(f"one old {self.kind} ID has multiple indexes")
            self.old_id_by_index[old_index] = old_id
            self.old_index_by_id[old_id] = old_index
            key = old_index, old_id, new_index, new_id
            self.counts[key] = self.counts.get(key, 0) + count
            self.index_changed += count * (old_index != new_index)
            instance_total += count
        if instance_total != expected_instances:
            raise ValueError(f"{self.kind} migration relations lose rows")
        self.instances += instance_total

    def verify(self, expected_instances: int, catalog_rows: list[dict[str, Any]]) -> None:
        if self.instances != expected_instances:
            raise ValueError(f"{self.kind} migration instance total differs")
        if sorted(self.old_id_by_index) != list(range(len(self.old_id_by_index))):
            raise ValueError(f"old {self.kind} indexes are not dense")
        new_indices = {key[2] for key in self.counts}
        if new_indices != set(range(len(catalog_rows))):
            raise ValueError(f"new {self.kind} indexes are not covered")
        if sum(self.counts.values()) != expected_instances:
            raise ValueError(f"{self.kind} migration relation counts differ")

    def rows(self) -> Iterator[dict[str, Any]]:
        for (old_index, old_id, new_index, new_id), count in sorted(self.counts.items()):
            yield {
                "kind": self.kind + "_index_mapping",
                "old_index": old_index, "old_id": old_id,
                "new_index": new_index, "new_id": new_id,
                "instance_count": count,
            }


def _migrate_one(old_root: Path, shard_id: int) -> MigratedShardTables:
    shard_dir = old_root / "shards" / f"part-{shard_id:06d}"
    filenames = ("records.parquet", "projections.parquet", "source_leaves.parquet")
    if {path.name for path in shard_dir.iterdir()} != set(filenames):
        raise ValueError(f"old structure shard has missing or extra tables: {shard_dir}")
    records, projections, leaves = (
        pq.read_table(shard_dir / name, use_threads=False) for name in filenames
    )
    for name, table in zip(filenames, (records, projections, leaves)):
        if pq.ParquetFile(shard_dir / name).metadata.num_rows != table.num_rows:
            raise ValueError(f"old structure table row count differs: {shard_dir / name}")
    return migrate_shard_tables(
        records, projections, leaves, shard_id=shard_id,
        relative_path=f"shards/part-{shard_id:06d}",
    )


def _written_shard(old_root: Path, shard_id: int,
                   projections: pa.Table, leaves: pa.Table) -> int:
    name = f"part-{shard_id:06d}"
    source_record = old_root / "shards" / name / "records.parquet"
    pending = MIGRATING / "shards" / (name + ".tmp")
    published = MIGRATING / "shards" / name
    if pending.exists() or published.exists():
        raise FileExistsError(f"migration shard destination exists: {name}")
    pending.mkdir(exist_ok=False)
    linked_record = pending / "records.parquet"
    os.link(source_record, linked_record)
    original_stat = source_record.stat()
    linked_stat = linked_record.stat()
    if ((original_stat.st_dev, original_stat.st_ino, original_stat.st_size) !=
            (linked_stat.st_dev, linked_stat.st_ino, linked_stat.st_size)):
        raise ValueError(f"migration record hard link differs: {name}")
    projection_path = pending / "projections.parquet"
    leaf_path = pending / "source_leaves.parquet"
    pq.write_table(projections, projection_path, compression="zstd",
                   use_dictionary=_PROJECTION_DICTIONARIES)
    pq.write_table(leaves, leaf_path, compression="zstd",
                   use_dictionary=_LEAF_DICTIONARIES)
    if (not pq.read_table(projection_path, use_threads=True).equals(projections) or
            not pq.read_table(leaf_path, use_threads=True).equals(leaves)):
        raise ValueError(f"migration Parquet readback differs: {name}")
    if ((source_record.stat().st_dev, source_record.stat().st_ino,
         source_record.stat().st_size) !=
            (linked_record.stat().st_dev, linked_record.stat().st_ino,
             linked_record.stat().st_size)):
        raise ValueError(f"migration record hard link changed: {name}")
    compressed_bytes = sum(path.stat().st_size for path in (
        linked_record, projection_path, leaf_path))
    os.replace(pending, published)
    return compressed_bytes


class _Migration:
    def __init__(self, declarations: tuple[tuple[int, Any], ...],
                 progress: _Progress) -> None:
        self.declarations = declarations
        self.rows = [_declaration_row(index, declaration)
                     for index, declaration in declarations]
        self.catalog = StructureCatalog()
        self.source_relations = _Relations("source")
        self.normalized_relations = _Relations("normalized")
        self.changes: Counter[str] = Counter()
        self.shards = []
        self.point: StorageResumePoint | None = None
        self.progress = progress

    def accept(self, old_root: Path, shard_id: int,
               result: MigratedShardTables) -> None:
        if type(result) is not MigratedShardTables:
            raise TypeError(f"migration worker returned a different type: {shard_id}")
        recovered = result.recovered
        summary = recovered.shard_summary
        point = recovered.resume_point
        if (summary.shard_id != shard_id or
                summary.relative_path != f"shards/part-{shard_id:06d}" or
                summary.compressed_bytes != 0 or
                point.shard_id != shard_id or not recovered.declarations or
                (summary.records, summary.projections, summary.source_leaves) != (
                    result.records.num_rows, result.projections.num_rows,
                    result.source_leaves.num_rows)):
            raise ValueError(f"migration worker shard metadata differs: {shard_id}")
        if (recovered.source_global_indices !=
                tuple(range(len(recovered.catalog_fragment.source_rows))) or
                recovered.normalized_global_indices !=
                tuple(range(len(recovered.catalog_fragment.normalized_rows)))):
            raise ValueError(f"migration worker indexes are not local: {shard_id}")
        first = recovered.declarations[0]
        if self.point is not None:
            if first.declaration_index < self.point.declaration_index:
                raise ValueError(f"migration declaration order reversed: {shard_id}")
            if first.declaration_index == self.point.declaration_index:
                if (first.first_record_ordinal != self.point.record_ordinal + 1 or
                        first.first_source_line is None or
                        self.point.source_line is None or
                        first.first_source_line <= self.point.source_line):
                    raise ValueError(f"migration source record gap: {shard_id}")
            elif first.first_record_ordinal != 0:
                raise ValueError(f"migration declaration starts after zero: {shard_id}")
        maps = self.catalog._merge_validated_fragment(recovered.catalog_fragment)
        projections = _mapped_index_column(
            result.projections, index_name="normalized_index", id_name="normalized_id",
            local_rows=recovered.catalog_fragment.normalized_rows,
            global_indexes=maps.normalized_indices)
        leaves = _mapped_index_column(
            result.source_leaves, index_name="source_index", id_name="source_field_id",
            local_rows=recovered.catalog_fragment.source_rows,
            global_indexes=maps.source_indices)
        self.source_relations.accept(
            result.source_index_relations, maps.source_indices,
            self.catalog._source_rows, result.source_leaves.num_rows)
        self.normalized_relations.accept(
            result.normalized_index_relations, maps.normalized_indices,
            self.catalog._normalized_rows, result.projections.num_rows)
        if set(result.column_change_counts) != _CHANGE_COLUMNS:
            raise ValueError(f"migration change categories differ: {shard_id}")
        for name, count in result.column_change_counts.items():
            if type(count) is not int or not 0 <= count <= (
                    result.source_leaves.num_rows if name in {
                        "source_field_id", "field_path"} else result.projections.num_rows):
                raise ValueError(f"migration change count is invalid: {shard_id}:{name}")
            self.changes[name] += count
        compressed_bytes = _written_shard(old_root, shard_id, projections, leaves)
        for item in recovered.declarations:
            _accept_recovered_declaration(self.declarations, self.rows, item, shard_id)
        self.shards.append(type(summary)(
            shard_id=summary.shard_id,
            relative_path=summary.relative_path,
            records=summary.records,
            projections=summary.projections,
            source_leaves=summary.source_leaves,
            first_declaration_index=summary.first_declaration_index,
            last_declaration_index=summary.last_declaration_index,
            excluded_integrity_subtrees=summary.excluded_integrity_subtrees,
            oversized_records=summary.oversized_records,
            compressed_bytes=compressed_bytes,
        ))
        self.point = point
        self.progress.completed += 1
        self.progress.records += summary.records
        self.progress.projections += summary.projections
        self.progress.source_leaves += summary.source_leaves
        self.progress.report("migrate")

    def expected_storage(self) -> StorageSummary:
        shards = tuple(self.shards)
        return StorageSummary(
            shards=shards,
            records=sum(shard.records for shard in shards),
            projections=sum(shard.projections for shard in shards),
            source_leaves=sum(shard.source_leaves for shard in shards),
            excluded_integrity_subtrees=sum(
                shard.excluded_integrity_subtrees for shard in shards),
            oversized_records=sum(shard.oversized_records for shard in shards),
        )

    def validate_inventory_and_catalog(self, storage: StorageSummary) -> None:
        if self.point is None or not storage.shards:
            raise ValueError("migration has no complete shard")
        _validate_fragment(self.catalog.export_fragment())
        for name in ("records", "projections", "source_leaves",
                     "excluded_integrity_subtrees"):
            if sum(row[name] for row in self.rows) != getattr(storage, name):
                raise ValueError(f"migration inventory {name} differs from storage")
        if (sum(row["instance_count"] for row in self.catalog._source_rows) !=
                storage.source_leaves or
                sum(row["instance_count"] for row in self.catalog._normalized_rows) !=
                storage.projections or
                sum(row["structure_status_counts"].get("structure_conflict", 0)
                    for row in self.catalog._normalized_rows) !=
                sum(row["structure_conflict_projections"] for row in self.rows)):
            raise ValueError("migration catalog observations differ from inventory")
        self.source_relations.verify(storage.source_leaves, self.catalog._source_rows)
        self.normalized_relations.verify(
            storage.projections, self.catalog._normalized_rows)


def _wait_for(future: Future[Any], progress: _Progress,
              phase: str, shard_id: int | None = None) -> Any:
    while True:
        try:
            return future.result(timeout=60)
        except TimeoutError:
            if future.done():
                return future.result()
            progress.report(phase, force=True)
        except Exception as exc:
            if shard_id is None:
                raise
            raise RuntimeError(f"migration shard worker failed: {shard_id}") from exc


def _mapping_rows(state: _Migration,
                  storage: StorageSummary) -> Iterator[dict[str, Any]]:
    assert state.point is not None
    changes = dict(sorted(state.changes.items()))
    changes["normalized_index"] = state.normalized_relations.index_changed
    changes["source_index"] = state.source_relations.index_changed
    yield {
        "kind": "structure_migration_mapping_header",
        "format_version": "structure-1",
        "scope": "migrated_prefix",
        "values_root": source_path(VALUES),
        "inventory_path": source_path(INVENTORY),
        "fields_path": source_path(FIELDS),
        "prefix_end": asdict(state.point),
        "shards": len(storage.shards),
        "records": storage.records,
        "projections": storage.projections,
        "source_leaves": storage.source_leaves,
        "column_change_counts": changes,
    }
    yield from state.source_relations.rows()
    yield from state.normalized_relations.rows()
    yield {
        "kind": "structure_migration_mapping_summary",
        "source_relations": len(state.source_relations.counts),
        "normalized_relations": len(state.normalized_relations.counts),
        "source_instances": state.source_relations.instances,
        "normalized_instances": state.normalized_relations.instances,
    }


def _preflight() -> list[tuple[int, Path]]:
    if not STAGED_VALUES.is_dir():
        raise FileNotFoundError(f"old staged structure root is absent: {STAGED_VALUES}")
    occupied = tuple(path for path in (
        VALUES, FIELDS, INVENTORY, STAGED_FIELDS, STAGED_INVENTORY,
        MIGRATING, BACKUP, MAPPING, MAPPING_TMP) if path.exists())
    if occupied:
        raise FileExistsError("migration output or previous cutover state exists: " +
                              ", ".join(map(str, occupied)))
    if STAGED_VALUES.stat().st_dev != MIGRATING.parent.stat().st_dev:
        raise ValueError("migration root is on a different filesystem")
    directories = _shard_dirs(STAGED_VALUES)
    if not directories:
        raise ValueError("old structure staging has no complete shard")
    if any(path.name != f"part-{shard_id:06d}" for shard_id, path in directories):
        raise ValueError("old structure shard names differ from writer output")
    return directories


def migrate(*, workers: int) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    directories = _preflight()
    started = time.monotonic()
    progress = _Progress(started, len(directories))
    declarations = ordered_source_declarations()
    state = _Migration(declarations, progress)
    MIGRATING.mkdir(exist_ok=False)
    (MIGRATING / "shards").mkdir(exist_ok=False)
    progress.report("migrate", force=True)
    with (ProcessPoolExecutor(max_workers=workers) if workers > 1
          else nullcontext(None)) as executor:
        pending: deque[tuple[int, Future[MigratedShardTables]]] = deque()

        def consume_one() -> None:
            shard_id, future = pending.popleft()
            state.accept(
                STAGED_VALUES, shard_id,
                _wait_for(future, progress, "migrate", shard_id))

        for shard_id, _ in directories:
            if executor is None:
                state.accept(STAGED_VALUES, shard_id,
                             _migrate_one(STAGED_VALUES, shard_id))
            else:
                pending.append((
                    shard_id, executor.submit(_migrate_one, STAGED_VALUES, shard_id)))
                if len(pending) >= 2 * workers:
                    consume_one()
        while pending:
            consume_one()

    expected = state.expected_storage()
    progress.report("validate_storage", force=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(validate_structure_storage, MIGRATING)
        validated = _wait_for(future, progress, "validate_storage")
    if validated != expected:
        raise ValueError("migrated Parquet storage differs from written shard summaries")
    state.validate_inventory_and_catalog(validated)
    _, writer_point = StructureStorageWriter.resume_from_validated(
        MIGRATING, validated)
    if state.point is None or writer_point != state.point:
        raise ValueError("migrated tail differs from storage writer")
    progress.report("check_source_tail", force=True)
    _check_resume_source(
        MIGRATING, declarations[writer_point.declaration_index][1],
        state.rows[writer_point.declaration_index], writer_point)
    _write_jsonl(MAPPING_TMP, _mapping_rows(state, validated))
    if (not STAGED_VALUES.is_dir() or BACKUP.exists() or
            not MIGRATING.is_dir() or MAPPING.exists()):
        raise ValueError("migration cutover paths changed during validation")
    os.replace(STAGED_VALUES, BACKUP)
    os.replace(MIGRATING, STAGED_VALUES)
    os.replace(MAPPING_TMP, MAPPING)
    progress.report("resume", force=True)
    summary = _resume_validated_prefix(
        declarations, state.rows, state.catalog, validated, writer_point,
        workers=workers, started=started)
    if (not VALUES.is_dir() or not FIELDS.is_file() or
            not INVENTORY.is_file() or not MAPPING.is_file() or
            STAGED_VALUES.exists() or not BACKUP.is_dir()):
        raise ValueError("migration publication is incomplete; old stage was preserved")
    progress.report("cleanup", force=True)
    shutil.rmtree(BACKUP)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    migrate(workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
