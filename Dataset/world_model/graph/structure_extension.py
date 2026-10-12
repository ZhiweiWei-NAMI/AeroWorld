"""Extend the published structure delivery with reviewed record grammars.

The existing projection and source-leaf Parquet files are linked into a staging
root. Existing record tables change only their declaration indexes; the exact
L6 implementation-sources metadata gap changes only its structure status.
Newly readable source records are then appended. Publication follows validation of
the complete staged delivery.

``stage`` builds that staging root from scratch. ``resume-stage`` finishes an
interrupted one: it verifies the retained prefix in place, recovers the appended
shards into the catalog and inventory, and continues any tail below the last
persisted record. It never copies the published delivery again and never
publishes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from dataclasses import replace
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from Dataset.world_model.graph.contract import source_path
from Dataset.world_model.graph.field_structure import (
    IMPLEMENTATION_SOURCES_GAP_NORMALIZED_ID,
    IMPLEMENTATION_SOURCES_GAP_SOURCE_ID,
)
from Dataset.world_model.graph.structure_catalog import (
    CatalogRows, StructureCatalog, validate_catalog_rows,
)
from Dataset.world_model.graph.structure_audit_checkpoint import (
    AuditCheckpoint, SourceIdentity, clear_default, stat_identity,
)
from Dataset.world_model.graph.structure_closure import (
    declared_effective_families, reviewed_source_grammar_declaration,
    source_grammar_for_declaration,
)
from Dataset.world_model.graph.structure_delivery import (
    FIELDS, INVENTORY, STAGED_FIELDS, STAGED_INVENTORY, STAGED_VALUES, VALUES,
    _Progress, _check_anchors, _check_resume_source, _declaration_row, _field_rows,
    _inventory_rows, _line, _load_delivery, _recover_prefix, _scan_declaration,
    _summary, _validate_record_grammar_source, _write_jsonl,
)
from Dataset.world_model.graph.structure_grammar_catalog import grammar_declaration_status
from Dataset.world_model.graph.structure_recovery import recover_structure_shard
from Dataset.world_model.graph.structure_sources import (
    EXTENSION_TAIL_FAMILIES, SourceResumeCursor, ordered_source_declarations,
)
from Dataset.world_model.graph.structure_storage import (
    ShardSummary, StorageSummary, StructureStorageWriter, _shard_dirs,
    validate_structure_storage,
)


MAPPING = INVENTORY.with_name("structure_migration_mapping.jsonl")
STAGED_MAPPING = MAPPING.with_name(MAPPING.name + ".tmp")
ACCEPTANCE = INVENTORY.with_name("structure_acceptance.json")
STAGED_ACCEPTANCE = ACCEPTANCE.with_name(ACCEPTANCE.name + ".tmp")
_EXISTING_FINALS = (VALUES, FIELDS, INVENTORY, MAPPING)
_FINALS = (*_EXISTING_FINALS, ACCEPTANCE)
_STAGED = (STAGED_VALUES, STAGED_FIELDS, STAGED_INVENTORY,
           STAGED_MAPPING, STAGED_ACCEPTANCE)
_BACKUPS = tuple(path.with_name(path.name + ".extend_backup")
                 for path in _EXISTING_FINALS)
_RECORD_DICTIONARIES = ("effective_family", "root_kind", "record_tick_type")
_PROJECTION_DICTIONARIES = (
    "normalized_id", "value_type", "value_status", "structure_status", "binding_slot",
)


def _role(declaration: Any) -> tuple[Any, ...]:
    return (
        declaration.source_family, source_path(declaration.path),
        declaration.sample, declaration.read_relation,
        declaration.read_condition, declaration.reference_samples,
    )


def _published_role(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["source_family"], row["source_file"], row["declared_sample"],
        row["read_relation"], row["read_condition"],
        tuple(row["reference_samples"]),
    )


def _observed_catalog_prefix(published: CatalogRows) -> CatalogRows:
    """Keep stored field indexes dense while new records are first observed."""
    source_count = sum(row["instance_count"] > 0 for row in published.source_fields)
    normalized_count = sum(
        row["instance_count"] > 0 for row in published.normalized_fields)
    for name, fields, count in (
            ("source", published.source_fields, source_count),
            ("normalized", published.normalized_fields, normalized_count)):
        if (any(row["instance_count"] < 1 for row in fields[:count]) or
                any(row["instance_count"] != 0 for row in fields[count:])):
            raise ValueError(f"published zero-instance {name} fields are not a suffix")
    sources = []
    for old in published.source_fields[:source_count]:
        row = deepcopy(old)
        row["normalizations"] = [
            binding for binding in row["normalizations"]
            if binding["instance_count"] > 0
        ]
        sources.append(row)
    normalized = []
    for old in published.normalized_fields[:normalized_count]:
        row = deepcopy(old)
        row["source_members"] = [
            member for member in row["source_members"]
            if member["source_index"] < source_count
        ]
        normalized.append(row)
    prefix = CatalogRows(tuple(sources), tuple(normalized))
    validate_catalog_rows(prefix)
    return prefix


def _verify_rebuilt_catalog(published: CatalogRows, current: CatalogRows) -> None:
    """Check stored indexes and every deferred zero-instance field reference."""
    for name, old_fields, new_fields in (
            ("source", published.source_fields, current.source_fields),
            ("normalized", published.normalized_fields, current.normalized_fields)):
        index_name = name + "_index"
        observed = (row for row in old_fields if row["instance_count"] > 0)
        for old in observed:
            index = old[index_name]
            if (index >= len(new_fields) or
                    new_fields[index][index_name] != index or
                    new_fields[index]["id"] != old["id"]):
                raise ValueError(f"published observed {name} index changed: {old['id']}")
        new_by_id = {row["id"]: row for row in new_fields}
        for old in old_fields:
            if old["instance_count"] == 0 and old["id"] not in new_by_id:
                raise ValueError(f"published zero-instance {name} field disappeared: {old['id']}")
    new_bindings = {
        (source["id"], binding["normalized_id"], binding["binding_slot"])
        for source in current.source_fields
        for binding in source["normalizations"]
    }
    for source in published.source_fields:
        for binding in source["normalizations"]:
            if binding["instance_count"] == 0 and (
                    source["id"], binding["normalized_id"],
                    binding["binding_slot"]) not in new_bindings:
                raise ValueError(
                    f"published zero-instance field binding disappeared: "
                    f"{source['id']}:{binding['normalized_id']}:{binding['binding_slot']}")


def _current_rows(
    published: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[tuple[int, Any], ...], tuple[int | None, ...]]:
    current = ordered_source_declarations()
    retained = [row for row in published
                if row["source_family"] not in EXTENSION_TAIL_FAMILIES]
    removed = [row for row in published
               if row["source_family"] in EXTENSION_TAIL_FAMILIES]
    if (not removed or
            any(row["structure_status"] != "source_schema_pending" or
                row["records"] or row["projections"] or row["source_leaves"] or
                row["shard_spans"] or row["structure_conflict_projections"]
                for row in removed)):
        raise ValueError("published schema-pending declarations are not empty")
    tail = current[len(retained):]
    if (len(current) <= len(retained) or
            any(declaration.source_family not in EXTENSION_TAIL_FAMILIES
                for _, declaration in tail)):
        raise ValueError("current schema-pending roles are not a final append block")
    old_tail = {_published_role(row): row for row in removed}
    new_tail = {_role(declaration) for _, declaration in tail}
    if len(old_tail) != len(removed) or not set(old_tail) <= new_tail:
        raise ValueError("published schema-pending source role disappeared")
    old_to_new: list[int | None] = [None] * len(published)
    rows = []
    for index, ((_, declaration), old) in enumerate(zip(current, retained)):
        if (_role(declaration) != _published_role(old) or
                declaration.structure_status != old["structure_status"] or
                declaration.presence != old["presence"] or
                (declaration.path.stat().st_size if declaration.presence == "present" else None)
                != old["input_bytes"]):
            raise ValueError(f"published source role or bytes changed: {old['source_file']}")
        row = dict(old)
        row["declaration_index"] = index
        row["record_grammar_status"] = grammar_declaration_status(declaration)
        row["grammar_validated_records"] = 0
        row["grammar_validated_effective_records"] = {}
        if row["structure_conflict_projections"]:
            if (row["source_family"] != "arm_manifest" or
                    row["structure_conflict_projections"] != 1 or
                    row["records"] != 1):
                raise ValueError(f"unrelated published structure conflict: {row['source_file']}")
            row["structure_conflict_projections"] = 0
        rows.append(row)
        old_to_new[old["declaration_index"]] = index
    for index, declaration in tail:
        old = old_tail.get(_role(declaration))
        if old is None:
            if (declaration.source_family != "arm_initial_events" or
                    declaration.structure_status != "field_scan_defined" or
                    declaration.presence != "present"):
                raise ValueError(f"unexpected new source role: {declaration.path}")
        elif (declaration.presence != old["presence"] or
              (declaration.path.stat().st_size
               if declaration.presence == "present" else None) != old["input_bytes"] or
              declaration.structure_status not in {
                  old["structure_status"], "field_scan_defined"
              }):
            raise ValueError(f"schema-pending source role or bytes changed: {declaration.path}")
        rows.append(_declaration_row(index, declaration))
    if len(rows) != len(current):
        raise ValueError("extended declaration count differs from current inventory")
    return rows, current, tuple(old_to_new)


def _new_index(old_index: int, old_to_new: tuple[int | None, ...]) -> int:
    if type(old_index) is not int or not 0 <= old_index < len(old_to_new):
        raise ValueError(f"published declaration index is invalid: {old_index}")
    current = old_to_new[old_index]
    if current is None:
        raise ValueError(f"stored record points to a schema-pending declaration: {old_index}")
    return current


def _storage_summary(raw: dict[str, Any]) -> StorageSummary:
    return StorageSummary(
        shards=tuple(ShardSummary(**item) for item in raw["shards"]),
        records=raw["records"], projections=raw["projections"],
        source_leaves=raw["source_leaves"],
        excluded_integrity_subtrees=raw["excluded_integrity_subtrees"],
        oversized_records=raw["oversized_records"],
    )


def _replace_table(path: Path, table: pa.Table, dictionaries: tuple[str, ...]) -> None:
    pending = path.with_name(path.name + ".tmp")
    if pending.exists():
        raise FileExistsError(pending)
    pq.write_table(table, pending, compression="zstd", use_dictionary=list(dictionaries))
    os.replace(pending, path)


def _verify_record_table(original: pa.Table, staged_path: Path,
                         indices: list[int], label: Path) -> None:
    """Require the staged record table to differ in ``declaration_index`` alone."""
    actual = pq.read_table(staged_path)
    unchanged = [name for name in original.column_names
                 if name != "declaration_index"]
    if (actual.schema != original.schema or
            actual.column("declaration_index").to_pylist() != indices or
            not actual.select(unchanged).equals(original.select(unchanged))):
        raise ValueError(f"rewritten record table changed another column: {label}")


def _record_shard_rewrite(
    shard: ShardSummary, old_to_new: tuple[int | None, ...], *,
    write: bool,
) -> tuple[bool, ShardSummary]:
    """Rebuild one record shard's declaration indexes and bounds.

    Returns whether its indexes moved and the rebuilt summary.  With ``write``
    false the staged table is only checked: it must already carry the expected
    indexes and must differ from the published table in that column alone.
    """
    old_path = VALUES / shard.relative_path / "records.parquet"
    staged_path = STAGED_VALUES / shard.relative_path / "records.parquet"
    first = _new_index(shard.first_declaration_index, old_to_new)
    last = _new_index(shard.last_declaration_index, old_to_new)
    moved = (first, last) != (shard.first_declaration_index,
                              shard.last_declaration_index)
    if moved:
        original = pq.read_table(old_path)
        old_indices = original.column("declaration_index").to_pylist()
        new_indices = [
            _new_index(value, old_to_new)
            for value in old_indices
        ]
        if new_indices != sorted(new_indices):
            raise ValueError(f"rewritten declaration order differs: {old_path}")
        if write:
            replacement = original.set_column(
                original.schema.get_field_index("declaration_index"),
                "declaration_index", pa.array(new_indices, type=pa.int64()),
            )
            _replace_table(staged_path, replacement, _RECORD_DICTIONARIES)
        _verify_record_table(original, staged_path, new_indices, old_path)
    elif not os.path.samefile(old_path, staged_path):
        raise ValueError(f"unchanged record shard was not linked: {old_path}")
    compressed = sum(
        (STAGED_VALUES / shard.relative_path / name).stat().st_size
        for name in ("records.parquet", "projections.parquet", "source_leaves.parquet")
    )
    return moved, replace(
        shard, first_declaration_index=first, last_declaration_index=last,
        compressed_bytes=compressed)


def _rewrite_records(
    shards: list[ShardSummary], old_to_new: tuple[int | None, ...], *,
    write: bool = True,
) -> int:
    """Reorder record declaration indexes and rebuild prefix shard bounds.

    ``write=False`` is the resume path's check of the very indexes and bounds
    this function produced earlier: it reads the staged root and changes nothing.
    """
    changed = 0
    last_report = time.monotonic()
    for index, shard in enumerate(shards):
        moved, rebuilt = _record_shard_rewrite(shard, old_to_new, write=write)
        changed += moved
        shards[index] = rebuilt
        now = time.monotonic()
        if now - last_report >= 600:
            print(_line({"phase": "rewrite_record_indexes", "shards_done": index + 1,
                         "shards_total": len(shards)}).rstrip(), flush=True)
            last_report = now
    return changed


def _gap_declarations(
    published: list[dict[str, Any]], old_to_new: tuple[int | None, ...],
) -> dict[int, set[int]]:
    grouped: dict[int, set[int]] = {}
    for row in published:
        if not row["structure_conflict_projections"]:
            continue
        if (row["source_family"] != "arm_manifest" or
                row["structure_conflict_projections"] != 1 or
                row["records"] != 1 or len(row["shard_spans"]) != 1):
            raise ValueError(f"unrelated structure conflict: {row['source_file']}")
        span = row["shard_spans"][0]
        if span["first_ordinal"] != 0 or span["last_ordinal"] != 0:
            raise ValueError(f"metadata gap has no JSON root ordinal: {row['source_file']}")
        grouped.setdefault(span["shard_id"], set()).add(
            _new_index(row["declaration_index"], old_to_new))
    if not grouped:
        raise ValueError("published delivery has no implementation-sources metadata gaps")
    return grouped


def _metadata_gap_position(
    stage_dir: Path, record: dict[str, Any],
    projections: list[dict[str, Any]], leaves: list[dict[str, Any]],
) -> int:
    """Return one record's unique implementation-sources gap projection.

    Both the position and the published row are checked against the current
    rule, so a rewrite and its verification can only ever settle on the same
    projection and cannot reclassify anything else.
    """
    if record["record_ordinal"] != 0:
        raise ValueError(f"metadata-gap record position differs: {stage_dir}")
    matches = [i for i in range(
        record["projection_start"],
        record["projection_start"] + record["projection_count"],
    ) if projections[i]["normalized_id"] == IMPLEMENTATION_SOURCES_GAP_NORMALIZED_ID]
    if len(matches) != 1:
        raise ValueError(f"metadata-gap projection is not unique: {stage_dir}")
    position = matches[0]
    projection = projections[position]
    if (projection["structure_status"] != "structure_conflict" or
            projection["value_type"] is not None or
            projection["value_status"] != "empty_object" or
            projection["normalized_value_json"] != b"null" or
            json.loads(projection["declared_shape_json"]) != [None] or
            projection["observed_shape_json"] is not None or
            projection["normalized_shape_json"] is not None or
            projection["source_leaf_count"] != 1):
        raise ValueError(f"published metadata gap differs from current rule: {stage_dir}")
    leaf = leaves[projection["source_leaf_start"]]
    if (leaf["source_field_id"] != IMPLEMENTATION_SOURCES_GAP_SOURCE_ID or
            leaf["field_path"] != "provenance.implementation_sources" or
            leaf["raw_path"] != "provenance.implementation_sources" or
            leaf["raw_value_json"] != b"{}" or
            leaf["raw_value_type"] != "empty_object" or
            json.loads(leaf["source_tokens_json"]) !=
            ["provenance", "implementation_sources"]):
        raise ValueError(f"published original metadata value differs: {stage_dir}")
    return position


def _verify_projection_table(original: pa.Table, staged_path: Path,
                             statuses: list[str], label: Path) -> None:
    """Require the staged projection table to differ in ``structure_status`` alone."""
    actual = pq.read_table(staged_path)
    unchanged = [name for name in original.column_names
                 if name != "structure_status"]
    if (actual.schema != original.schema or
            actual.column("structure_status").to_pylist() != statuses or
            not actual.select(unchanged).equals(original.select(unchanged))):
        raise ValueError(f"rewritten metadata projection changed another column: {label}")


def _metadata_gap_shard_rewrite(
    shard: ShardSummary, declarations: set[int], *,
    write: bool,
) -> tuple[int, ShardSummary]:
    """Settle one shard's manifest gaps on the reclassified structure status.

    Returns how many gap projections were reclassified and the rebuilt summary.
    With ``write`` false the staged table is only checked: the reclassified
    status must already be there and every other column must still equal the
    published table.
    """
    old_dir = VALUES / shard.relative_path
    stage_dir = STAGED_VALUES / shard.relative_path
    records = pq.read_table(stage_dir / "records.parquet").to_pylist()
    original = pq.read_table(old_dir / "projections.parquet")
    projections = original.to_pylist()
    leaves = pq.read_table(old_dir / "source_leaves.parquet", columns=[
        "source_field_id", "field_path", "raw_path", "raw_value_json",
        "raw_value_type", "source_tokens_json",
    ]).to_pylist()
    statuses = original.column("structure_status").to_pylist()
    seen: set[int] = set()
    changed = 0
    for record in records:
        declaration_index = record["declaration_index"]
        if declaration_index not in declarations:
            continue
        if declaration_index in seen:
            raise ValueError(f"metadata-gap record position differs: {stage_dir}")
        seen.add(declaration_index)
        statuses[_metadata_gap_position(
            stage_dir, record, projections, leaves)] = "declared_only"
        changed += 1
    if seen != declarations:
        raise ValueError(f"metadata-gap declarations are absent from shard: {stage_dir}")
    if write:
        replacement = original.set_column(
            original.schema.get_field_index("structure_status"),
            "structure_status", pa.array(statuses, type=pa.string()),
        )
        _replace_table(stage_dir / "projections.parquet", replacement,
                       _PROJECTION_DICTIONARIES)
    _verify_projection_table(original, stage_dir / "projections.parquet",
                             statuses, stage_dir)
    return changed, replace(
        shard,
        compressed_bytes=sum((stage_dir / name).stat().st_size for name in (
            "records.parquet", "projections.parquet", "source_leaves.parquet")),
    )


def _rewrite_metadata_gaps(
    shards: list[ShardSummary], grouped: dict[int, set[int]],
    *, write: bool = True,
) -> int:
    """Reclassify the exact L6 implementation-sources metadata gaps.

    ``write=False`` is the resume path's check of the very status corrections
    this function produced earlier: it reads the staged root and changes nothing.
    """
    changed = 0
    for shard_id, declarations in sorted(grouped.items()):
        item, rebuilt = _metadata_gap_shard_rewrite(
            shards[shard_id], declarations, write=write)
        changed += item
        shards[shard_id] = rebuilt
    return changed


def _mapping_point(root: Path, storage: dict[str, Any], point: dict[str, Any]) -> None:
    shard_id = point["shard_id"]
    if type(shard_id) is not int or not 0 <= shard_id < len(storage["shards"]):
        raise ValueError("migration prefix shard is absent")
    shard = storage["shards"][shard_id]
    if shard["shard_id"] != shard_id:
        raise ValueError("migration prefix shard ID differs")
    table = pq.read_table(root / shard["relative_path"] / "records.parquet",
                          columns=["record_index", "declaration_index",
                                   "record_ordinal", "source_line"])
    index = point["record_index"]
    if type(index) is not int or not 0 <= index < table.num_rows:
        raise ValueError("migration prefix record is absent")
    record = table.slice(index, 1).to_pylist()[0]
    if any(point[key] != record[key] for key in record):
        raise ValueError("migration prefix does not resolve to its stored record")


def _historical_mapping_header(storage: dict[str, Any]) -> dict[str, Any]:
    with MAPPING.open("rb") as stream:
        header = json.loads(stream.readline())
    if (header.get("kind") != "structure_migration_mapping_header" or
            header.get("scope") != "migrated_prefix" or
            header.get("values_root") != source_path(VALUES) or
            header.get("inventory_path") != source_path(INVENTORY) or
            header.get("fields_path") != source_path(FIELDS) or
            "current_prefix_end" in header):
        raise ValueError("published migration mapping header differs")
    _mapping_point(VALUES, storage, header["prefix_end"])
    return header


def _stage_mapping(header: dict[str, Any],
                   old_to_new: tuple[int | None, ...]) -> None:
    with MAPPING.open("rb") as source, STAGED_MAPPING.open("xb") as output:
        if json.loads(source.readline()) != header:
            raise ValueError("published migration mapping changed during staging")
        header = dict(header)
        historical = header["prefix_end"]
        header["prefix_end_scope"] = "historical_declaration_index"
        header["current_prefix_end"] = {
            **historical,
            "declaration_index": _new_index(historical["declaration_index"],
                                             old_to_new),
        }
        output.write(_line(header).encode("utf-8"))
        shutil.copyfileobj(source, output)


def _validate_staged_mapping(storage: dict[str, Any],
                             staged_storage: dict[str, Any],
                             old_to_new: tuple[int | None, ...]) -> None:
    historical = _historical_mapping_header(storage)
    with MAPPING.open("rb") as source, STAGED_MAPPING.open("rb") as staged:
        if json.loads(source.readline()) != historical:
            raise ValueError("historical migration mapping changed")
        updated = json.loads(staged.readline())
        expected = dict(historical)
        expected["prefix_end_scope"] = "historical_declaration_index"
        expected["current_prefix_end"] = {
            **historical["prefix_end"],
            "declaration_index": _new_index(
                historical["prefix_end"]["declaration_index"], old_to_new),
        }
        if updated != expected:
            raise ValueError("staged migration mapping header differs")
        _mapping_point(STAGED_VALUES, staged_storage, updated["current_prefix_end"])
        last_kind = None
        for original_line, staged_line in zip_longest(source, staged):
            if original_line is None or staged_line != original_line:
                raise ValueError("staged migration mapping changed historical field maps")
            last_kind = json.loads(original_line)["kind"]
        if last_kind != "structure_migration_mapping_summary":
            raise ValueError("migration mapping has no final summary")


def _audit_one_source(index: int, declaration: Any,
                      row: dict[str, Any]) -> dict[str, Any]:
    return _validate_record_grammar_source(index, declaration, row)


def _audit_identity(index: int, declaration: Any,
                    row: dict[str, Any]) -> SourceIdentity:
    """Everything a reused audit result of this source has to agree with."""
    return SourceIdentity(
        source_file=row["source_file"],
        declaration_index=index,
        source_family=declaration.source_family,
        role=_role(declaration),
        declared_sample=declaration.sample,
        structure_status=declaration.structure_status,
        presence=row["presence"],
        records=row["records"],
        input_bytes=row["input_bytes"],
        row_effective=tuple(row["effective_families"]),
        declared_effective=tuple(declared_effective_families(declaration)),
        shard_spans=tuple(row["shard_spans"]),
        stat=stat_identity(declaration.path),
    )


def _audit_schemas(declaration: Any) -> tuple[tuple[str, str, str], ...]:
    """The complete canonical record grammar schema of every effective type.

    A declaration with several effective types or a profile resolves one grammar
    per type, so all of them have to hold for a reuse.  The third element is the
    grammar's own canonical schema text, compared byte for byte rather than by
    reference, so a changed schema invalidates only the sources that use it.
    """
    return tuple(
        (effective, grammar.schema_ref, grammar._schema_json)
        for effective in declared_effective_families(declaration)
        for grammar in (source_grammar_for_declaration(
            declaration, effective_family=effective),))


def _audit_existing_sources(
    rows: list[dict[str, Any]], current: tuple[tuple[int, Any], ...],
    retained_count: int, workers: int, checkpoint: AuditCheckpoint,
) -> list[dict[str, Any]]:
    """Audit the retained sources, reusing only still-matching completed ones."""
    if workers < 1:
        raise ValueError("grammar audit worker count must be positive")
    selected = [(index, declaration, rows[index])
                for index, declaration in current[:retained_count]
                if rows[index]["records"] and
                reviewed_source_grammar_declaration(declaration)]
    expected = sum(row["records"] for _, _, row in selected)
    results: list[dict[str, Any]] = []
    scanned: list[tuple[int, Any, dict[str, Any], SourceIdentity]] = []
    cached_records = 0
    for index, declaration, row in selected:
        identity = _audit_identity(index, declaration, row)
        # Only a completed result whose declaration role, published position and
        # record count, source stat identity and complete grammar schema still
        # match is replayed.  A source that failed or was interrupted has no row
        # here and is audited again.
        cached = checkpoint.reuse(identity, lambda d=declaration: _audit_schemas(d))
        if cached is None:
            scanned.append((index, declaration, row, identity))
        else:
            results.append(cached)
            cached_records += cached["records"]
    print(_line({"phase": "grammar_audit", "sources": len(selected),
                 "cached_sources": len(results), "scanned_sources": len(scanned),
                 "expected_records": expected}).rstrip(), flush=True)
    failures: list[tuple[str, str]] = []
    validated = cached_records
    done_sources = 0
    last_report = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(_audit_one_source, index, declaration, row):
                (identity, declaration)
            for index, declaration, row, identity in scanned
        }
        while pending:
            done, _ = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            for future in done:
                identity, declaration = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    detail = str(exc)
                    failures.append((identity.source_file, detail))
                    print(_line({"phase": "grammar_audit_failure",
                                 "source_file": identity.source_file,
                                 "detail": detail}).rstrip(), flush=True)
                    done_sources += 1
                    continue
                # Commit inside the main process, one transaction per success, so
                # an interrupt keeps every source already audited.
                checkpoint.commit(identity, result, _audit_schemas(declaration))
                results.append(result)
                validated += result["records"]
                done_sources += 1
            now = time.monotonic()
            if now - last_report >= 600:
                print(_line({"phase": "grammar_audit",
                             "cached_sources": len(results) - (done_sources - len(failures)),
                             "scanned_sources_done": done_sources,
                             "scanned_sources_total": len(scanned),
                             "sources_failed": len(failures),
                             "validated_records": validated}).rstrip(), flush=True)
                last_report = now
    if failures:
        raise RuntimeError(
            f"grammar audit failed for {len(failures)} source(s); "
            f"validated {validated} of {expected} expected records")
    if validated != expected:
        raise ValueError("grammar audit count differs from published source inventory")
    # Every selected source now points at the version it was audited with, so any
    # schema version left over from a builder change has no remaining holder and
    # can be dropped.  Keeps the stored version count equal to what is in use.
    reaped_versions = checkpoint.reap_unreferenced()
    print(_line({"phase": "grammar_audit_complete", "sources": len(selected),
                 "cached_sources": len(selected) - len(scanned),
                 "scanned_sources": len(scanned),
                 "cached_records": cached_records,
                 "scanned_records": validated - cached_records,
                 "validated_records": validated,
                 "reaped_schema_versions": reaped_versions}).rstrip(), flush=True)
    return sorted(results, key=lambda item: item["declaration_index"])


def _stage_acceptance(
    summary: dict[str, Any], audits: list[dict[str, Any]],
    *, rewritten_records: int, gap_count: int, appended_records: int,
    migrated_records: int,
) -> None:
    if migrated_records + appended_records != summary["records"]:
        raise ValueError("migrated and appended records do not sum to the delivery total")
    row = {
        "kind": "structure_acceptance",
        "inventory_path": source_path(INVENTORY),
        "fields_path": source_path(FIELDS),
        "values_root": source_path(VALUES),
        "migration_mapping_path": source_path(MAPPING),
        "declarations": summary["declarations"],
        "records": summary["records"],
        "source_schema_closure": summary["source_schema_closure"],
        "schema_coverage": summary["schema_coverage"],
        "grammar_validated_records": summary["grammar_validated_records"],
        "validated_sources": audits,
        "record_index_shards_rewritten": rewritten_records,
        "historical_metadata_gaps_reclassified": gap_count,
        # G7 keeps the two volumes apart: records carried over from the
        # published delivery (declaration indexes rewritten where the append
        # block shifted them) versus records newly scanned from the appended
        # declarations.  Their sum is the delivery total, checked above.  The
        # appended block spans seven source families, so it is named for the
        # volume it is and not for one family.
        "migrated_records": migrated_records,
        "appended_records": appended_records,
        "storage_validation": "complete_staged_parquet_scan",
    }
    with STAGED_ACCEPTANCE.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, indent=2) + "\n")


def _restore_cutover() -> None:
    if ACCEPTANCE.exists():
        ACCEPTANCE.unlink()
    for final, backup in zip(_EXISTING_FINALS, _BACKUPS):
        if not backup.exists():
            continue
        if final.is_dir():
            shutil.rmtree(final)
        elif final.exists():
            final.unlink()
        os.replace(backup, final)


def _deliver_staged(
    *, started: float, rows: list[dict[str, Any]],
    current: tuple[tuple[int, Any], ...], old_to_new: tuple[int | None, ...],
    catalog: StructureCatalog, published_catalog: CatalogRows,
    baseline: dict[str, Any], mapping_header: dict[str, Any],
    retained_count: int, storage: StorageSummary, prefix: StorageSummary,
    audits: list[dict[str, Any]], anchors: dict[str, Any],
    rewritten_records: int, rewritten_gaps: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Finish and verify one complete staged delivery, without publishing it.

    This is the shared second half of a stage: the appended records are already
    on disk in ``STAGED_VALUES``, so it only closes the catalog, verifies the
    staged Parquet scan against the appended summary and writes the four staged
    indexes that publication will read.
    """
    fields = catalog.finish(
        declarations=tuple(declaration for _, declaration in current))
    _verify_rebuilt_catalog(published_catalog, fields)
    if catalog.grammar_declaration_statuses != tuple(
            row["record_grammar_status"] for row in rows):
        raise ValueError("catalog grammar declarations differ from source inventory")
    for row in rows[retained_count:]:
        if row["records"]:
            if row["record_grammar_status"] != "closed_record_grammar":
                raise ValueError(f"appended source has no closed grammar: {row['source_file']}")
            span = row["shard_spans"][-1]
            audits.append({
                "declaration_index": row["declaration_index"],
                "source_family": row["source_family"],
                "source_file": row["source_file"],
                "records": row["grammar_validated_records"],
                "effective_records": row["grammar_validated_effective_records"],
                "last_record_ordinal": span["last_ordinal"],
                "last_source_line": span["last_line"],
            })
    audits.sort(key=lambda item: item["declaration_index"])
    expected_audits = {
        row["declaration_index"] for row in rows
        if row["record_grammar_status"] == "closed_record_grammar" and row["records"]
    }
    if ({item["declaration_index"] for item in audits} != expected_audits or
            len(audits) != len(expected_audits)):
        raise ValueError("grammar audits differ from reviewed source declarations")
    summary = _summary(rows, fields, storage, time.monotonic() - started)
    actual = validate_structure_storage(STAGED_VALUES)
    if actual != storage:
        raise ValueError("staged structure storage differs from appended summary")
    _check_anchors(STAGED_VALUES, anchors)
    _write_jsonl(STAGED_FIELDS, _field_rows(fields))
    _write_jsonl(STAGED_INVENTORY, _inventory_rows(rows, summary))
    _load_delivery(STAGED_INVENTORY, STAGED_FIELDS, STAGED_VALUES,
                   scan_storage=False)
    _stage_mapping(mapping_header, old_to_new)
    _validate_staged_mapping(baseline["storage"], summary["storage"],
                             old_to_new)
    appended_records = sum(row["records"] for row in rows[retained_count:])
    migrated_records = prefix.records
    _stage_acceptance(summary, audits, rewritten_records=rewritten_records,
                      gap_count=rewritten_gaps, appended_records=appended_records,
                      migrated_records=migrated_records)
    result = {key: value for key, value in summary.items() if key != "storage"}
    result.update(rewritten_record_shards=rewritten_records,
                  reclassified_metadata_gaps=rewritten_gaps,
                  migrated_records=migrated_records,
                  appended_records=appended_records,
                  shards=len(storage.shards), phase="staged")
    if extra:
        result.update(extra)
    print(_line(result).rstrip(), flush=True)
    return result


def stage_extension(*, workers: int = 16) -> dict[str, Any]:
    """Build and verify an extension without changing published paths."""
    occupied = [path for path in (*_STAGED, *_BACKUPS) if path.exists()]
    if occupied:
        raise FileExistsError("structure extension stage or backup already exists: " +
                              ", ".join(map(str, occupied)))
    if ACCEPTANCE.exists() or any(not path.exists() for path in _EXISTING_FINALS):
        raise FileNotFoundError("published structure delivery or migration mapping is absent")
    started = time.monotonic()
    baseline, published, published_catalog = _load_delivery(
        INVENTORY, FIELDS, VALUES, scan_storage=False,
        published_baseline=True)
    rows, current, old_to_new = _current_rows(published)
    mapping_header = _historical_mapping_header(baseline["storage"])
    retained_count = sum(index is not None for index in old_to_new)
    gap_shards = _gap_declarations(published, old_to_new)
    gap_count = sum(len(items) for items in gap_shards.values())
    if gap_count != baseline["structure_conflict_projections"]:
        raise ValueError("metadata gaps do not account for published structure conflicts")
    catalog = StructureCatalog.resume_from_published(
        _observed_catalog_prefix(published_catalog))
    catalog.reclassify_implementation_sources_gap(gap_count)
    checkpoint = AuditCheckpoint.open_default(VALUES.parent)
    try:
        audits = _audit_existing_sources(rows, current, retained_count, workers,
                                         checkpoint)
    finally:
        # A failed or interrupted stage keeps the checkpoint so the next run
        # resumes the audit; only a successful publish removes it.
        checkpoint.close()
    for audit in audits:
        row = rows[audit["declaration_index"]]
        row["grammar_validated_records"] = audit["records"]
        row["grammar_validated_effective_records"] = audit["effective_records"]
    shutil.copytree(VALUES, STAGED_VALUES, copy_function=os.link)
    shards = list(_storage_summary(baseline["storage"]).shards)
    rewritten_records = _rewrite_records(shards, old_to_new)
    rewritten_gaps = _rewrite_metadata_gaps(shards, gap_shards)
    if rewritten_gaps != gap_count:
        raise ValueError("staged metadata-gap count differs from published inventory")
    prefix = StorageSummary(
        shards=tuple(shards),
        records=baseline["records"], projections=baseline["projections"],
        source_leaves=baseline["source_leaves"],
        excluded_integrity_subtrees=baseline["excluded_integrity_subtrees"],
        oversized_records=baseline["storage"]["oversized_records"],
    )
    writer, point = StructureStorageWriter.resume_from_validated(STAGED_VALUES, prefix)
    last_published = max(index for index, row in enumerate(rows[:retained_count])
                         if row["records"])
    if point.declaration_index != last_published:
        raise ValueError("rewritten storage tail differs from declaration inventory")
    progress = _Progress(started, time.monotonic(),
                         declarations=retained_count,
                         records=prefix.records, projections=prefix.projections,
                         source_leaves=prefix.source_leaves)
    anchors = {}
    for index, declaration in current[retained_count:]:
        row = rows[index]
        _scan_declaration(index, declaration, row, catalog, writer,
                          None, 1, anchors, progress)
        progress.declarations += 1
        progress.report()
    storage = writer.finish()
    return _deliver_staged(
        started=started, rows=rows, current=current, old_to_new=old_to_new,
        catalog=catalog, published_catalog=published_catalog,
        baseline=baseline, mapping_header=mapping_header,
        retained_count=retained_count, storage=storage, prefix=prefix,
        audits=audits, anchors=anchors,
        rewritten_records=rewritten_records, rewritten_gaps=rewritten_gaps)


def _verify_staged_prefix_links(
    shards: list[ShardSummary], gap_shards: dict[int, set[int]],
) -> int:
    """Require the staged prefix to still hold the published table links.

    Only the record table of a reordered shard and the projection table of a
    manifest-gap shard were rewritten in place; every other table is still the
    published file's own link.  Returns how many links were checked.
    """
    checked = 0
    for shard in shards:
        old_dir = VALUES / shard.relative_path
        stage_dir = STAGED_VALUES / shard.relative_path
        if {item.name for item in stage_dir.iterdir()} != {
                "records.parquet", "projections.parquet", "source_leaves.parquet"}:
            raise ValueError(f"staged shard tables differ: {stage_dir}")
        if not os.path.samefile(old_dir / "source_leaves.parquet",
                                stage_dir / "source_leaves.parquet"):
            raise ValueError(f"staged source leaves are not the published link: {stage_dir}")
        checked += 1
        if shard.shard_id not in gap_shards:
            if not os.path.samefile(old_dir / "projections.parquet",
                                    stage_dir / "projections.parquet"):
                raise ValueError(f"staged projections are not the published link: {stage_dir}")
            checked += 1
    return checked


def _audit_appended_records(
    rows: list[dict[str, Any]], current: tuple[tuple[int, Any], ...],
    retained_count: int, *, complete: bool, start: int, stop: int,
) -> int:
    """Re-read appended sources in ``current[start:stop]`` for their grammar counts.

    Every appended source that holds records is read through the same validator
    as the delivery, so its ``grammar_validated_records`` are the audited ones
    and never a value carried over from the scan.  Returns how many records were
    validated.
    """
    validated = 0
    for index, declaration in current[max(retained_count, start):stop]:
        row = rows[index]
        if not row["records"]:
            continue
        if row["record_grammar_status"] != "closed_record_grammar":
            raise ValueError(f"appended source has no closed grammar: {row['source_file']}")
        audit = _validate_record_grammar_source(
            index, declaration, row, complete=complete)
        row["grammar_validated_records"] = audit["records"]
        row["grammar_validated_effective_records"] = audit["effective_records"]
        validated += audit["records"]
    return validated


def resume_stage_extension(*, workers: int = 16) -> dict[str, Any]:
    """Finish a staged extension from its retained shards without re-copying.

    The staging root already holds the rewritten published prefix and the whole
    appended block, so nothing is copied again and no source is rescanned into
    new records.  The staged prefix is verified in place, the appended shards
    are recovered into the catalog and the source inventory, and any tail left
    below the last persisted record continues after its own cursor.  The run
    stops on the staged indexes exactly like a stage; it never publishes.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    occupied = [path for path in _BACKUPS if path.exists()]
    if occupied:
        raise FileExistsError("structure extension backup is occupied: " +
                              ", ".join(map(str, occupied)))
    delivered = [path for path in (STAGED_FIELDS, STAGED_INVENTORY,
                                   STAGED_MAPPING, STAGED_ACCEPTANCE)
                 if path.exists()]
    if delivered:
        raise FileExistsError("structure extension stage is already delivered: " +
                              ", ".join(map(str, delivered)))
    if not STAGED_VALUES.is_dir():
        raise FileNotFoundError(f"staged structure values root is absent: {STAGED_VALUES}")
    if ACCEPTANCE.exists() or any(not path.exists() for path in _EXISTING_FINALS):
        raise FileNotFoundError("published structure delivery or migration mapping is absent")
    started = time.monotonic()
    baseline, published, published_catalog = _load_delivery(
        INVENTORY, FIELDS, VALUES, scan_storage=False,
        published_baseline=True)
    rows, current, old_to_new = _current_rows(published)
    mapping_header = _historical_mapping_header(baseline["storage"])
    retained_count = sum(index is not None for index in old_to_new)
    gap_shards = _gap_declarations(published, old_to_new)
    gap_count = sum(len(items) for items in gap_shards.values())
    if gap_count != baseline["structure_conflict_projections"]:
        raise ValueError("metadata gaps do not account for published structure conflicts")
    catalog = StructureCatalog.resume_from_published(
        _observed_catalog_prefix(published_catalog))
    catalog.reclassify_implementation_sources_gap(gap_count)
    checkpoint = AuditCheckpoint.open_default(VALUES.parent)
    try:
        audits = _audit_existing_sources(rows, current, retained_count, workers,
                                         checkpoint)
    finally:
        # A failed or interrupted stage keeps the checkpoint so the next run
        # resumes the audit; only a successful publish removes it.
        checkpoint.close()
    for audit in audits:
        row = rows[audit["declaration_index"]]
        row["grammar_validated_records"] = audit["records"]
        row["grammar_validated_effective_records"] = audit["effective_records"]

    # The staged prefix is checked, never rewritten: the same expected indexes
    # and bounds are recomputed and compared against what is already on disk.
    shards = list(_storage_summary(baseline["storage"]).shards)
    verified_records = _rewrite_records(shards, old_to_new, write=False)
    verified_gaps = _rewrite_metadata_gaps(shards, gap_shards, write=False)
    if verified_gaps != gap_count:
        raise ValueError("staged metadata-gap count differs from published inventory")
    verified_links = _verify_staged_prefix_links(shards, gap_shards)
    prefix = StorageSummary(
        shards=tuple(shards),
        records=baseline["records"], projections=baseline["projections"],
        source_leaves=baseline["source_leaves"],
        excluded_integrity_subtrees=baseline["excluded_integrity_subtrees"],
        oversized_records=baseline["storage"]["oversized_records"],
    )
    directories = _shard_dirs(STAGED_VALUES)
    if len(directories) < len(prefix.shards):
        raise ValueError("staged structure storage lost published shards")
    if [shard_id for shard_id, _ in directories[:len(prefix.shards)]] != list(
            shard.shard_id for shard in prefix.shards):
        raise ValueError("staged structure prefix shard names differ from the summary")
    if [shard_id for shard_id, _ in directories] != list(range(len(directories))):
        raise ValueError("staged structure shard names are not consecutive")
    appended_ids = tuple(shard_id for shard_id, _ in directories[len(prefix.shards):])
    if not appended_ids:
        raise ValueError("staged structure storage has no appended shard")

    tail_shard = recover_structure_shard(STAGED_VALUES, appended_ids[-1])
    point = tail_shard.resume_point
    if not retained_count <= point.declaration_index < len(current):
        raise ValueError("appended block covers a retained or absent declaration")
    declaration = current[point.declaration_index][1]
    row = rows[point.declaration_index]
    source_record, tail_anchor = _check_resume_source(
        STAGED_VALUES, declaration, row, point)

    retained_totals = [
        (row["records"], row["projections"], row["source_leaves"],
         len(row["shard_spans"]))
        for row in rows[:retained_count]
    ]
    progress = _Progress(started, time.monotonic(),
                         records=prefix.records, projections=prefix.projections,
                         source_leaves=prefix.source_leaves)
    recovered_storage, last_point = _recover_prefix(
        current, rows, catalog, appended_ids, tail_shard, progress, workers)
    if last_point != point:
        raise ValueError("recovered source cursor differs from the storage tail")
    if any(
        (rows[index]["records"], rows[index]["projections"],
         rows[index]["source_leaves"], len(rows[index]["shard_spans"])) != totals
        for index, totals in enumerate(retained_totals)
    ):
        raise ValueError("appended shards hold records of a published declaration")
    # A recovered row is never scanned again, and _accept_recovered_declaration
    # carries no delivery status, so its declaration row is still the initial
    # zero_instance.  Set the projected state the scan would have set, for the
    # block that ends before the tail: the tail row itself is re-derived by
    # _scan_declaration, complete or not.  A recovered row without projections
    # keeps its zero_instance, and a missing or pending one keeps its own state.
    for index in range(retained_count, point.declaration_index):
        row = rows[index]
        if row["projections"]:
            row["delivery_status"] = "projected"

    combined_shards = prefix.shards + recovered_storage.shards
    storage = StorageSummary(
        shards=combined_shards,
        records=prefix.records + recovered_storage.records,
        projections=prefix.projections + recovered_storage.projections,
        source_leaves=prefix.source_leaves + recovered_storage.source_leaves,
        excluded_integrity_subtrees=(prefix.excluded_integrity_subtrees +
                                    recovered_storage.excluded_integrity_subtrees),
        oversized_records=(prefix.oversized_records +
                           recovered_storage.oversized_records),
    )
    writer, writer_point = StructureStorageWriter.resume_from_validated(
        STAGED_VALUES, storage)
    if writer_point != point:
        raise ValueError("recovered source cursor differs from the storage writer")

    # The persisted block is audited against its real sources before anything is
    # appended, so a partial tail is only marked complete once it really is.
    persisted_records = _audit_appended_records(
        rows, current, retained_count, complete=True,
        start=retained_count, stop=point.declaration_index)
    _audit_appended_records(
        rows, current, retained_count, complete=False,
        start=point.declaration_index, stop=point.declaration_index + 1)
    tail_records = rows[point.declaration_index]["records"]
    anchors = {declaration.source_family: tail_anchor}
    resume_after = SourceResumeCursor(
        point.record_ordinal, point.source_line, source_record.effective_family)
    progress.declarations = point.declaration_index
    for index, scanned in current[point.declaration_index:]:
        row = rows[index]
        progress.source_file = row["source_file"]
        progress.record_ordinal = (point.record_ordinal
                                   if index == point.declaration_index else None)
        progress.source_line = (point.source_line
                                if index == point.declaration_index else None)
        _scan_declaration(index, scanned, row, catalog, writer, None, 1,
                          anchors, progress,
                          resume_after=(resume_after
                                        if index == point.declaration_index else None))
        progress.declarations += 1
        progress.report()
    # Anything the tail scan added still has to be audited; a block that already
    # covered its last declaration passes here with nothing left to read.  This
    # re-reads the tail declaration complete together with every declaration
    # after it, so it is the tail block's final count and not the appended
    # block's.
    tail_onward_records = _audit_appended_records(
        rows, current, retained_count, complete=True,
        start=point.declaration_index, stop=len(current))
    # The two audit ranges partition the appended block: everything before the
    # tail declaration was stored complete, and everything from the tail on was
    # just re-read complete.  What was already on disk is that persisted range
    # plus the tail's own stored records; what this run wrote is the tail
    # block's final count minus its stored records.  The three must close
    # against the appended block's own record count, and a shortfall or a
    # negative is a real accounting error rather than a count to round away.
    all_audited_records = persisted_records + tail_onward_records
    persisted_block_records = persisted_records + tail_records
    written_after_resume = tail_onward_records - tail_records
    appended_block_records = sum(row["records"] for row in rows[retained_count:])
    if (all_audited_records != appended_block_records or
            persisted_block_records + written_after_resume != all_audited_records or
            written_after_resume < 0 or persisted_block_records < 0):
        raise ValueError(
            f"appended record accounting differs: audited={all_audited_records} "
            f"block={appended_block_records} persisted={persisted_block_records} "
            f"written={written_after_resume}")
    storage = writer.finish()
    result = _deliver_staged(
        started=started, rows=rows, current=current, old_to_new=old_to_new,
        catalog=catalog, published_catalog=published_catalog,
        baseline=baseline, mapping_header=mapping_header,
        retained_count=retained_count, storage=storage, prefix=prefix,
        audits=audits, anchors=anchors,
        rewritten_records=verified_records, rewritten_gaps=verified_gaps,
        extra={
            # The verification counts are reported, not implied: the prefix was
            # checked in place, the appended block was recovered, and the tail
            # continued after its own cursor without rewriting a stored record.
            "verified_record_index_shards": verified_records,
            "verified_metadata_gaps": verified_gaps,
            "verified_published_table_links": verified_links,
            "recovered_appended_shards": len(appended_ids),
            "recovered_appended_records": recovered_storage.records,
            "audited_persisted_records": persisted_block_records,
            "audited_appended_records": all_audited_records,
            "records_appended_after_resume": written_after_resume,
            "resumed_after_declaration": point.declaration_index,
            "resumed_after_record_ordinal": point.record_ordinal,
            "resumed_after_source_line": point.source_line,
        })
    return result


def publish_extension() -> dict[str, Any]:
    """Switch an already validated stage after its code and results are reviewed."""
    if any(not path.exists() for path in (*_EXISTING_FINALS, *_STAGED)):
        raise FileNotFoundError("published delivery or complete extension stage is absent")
    if ACCEPTANCE.exists() or any(path.exists() for path in _BACKUPS):
        raise FileExistsError("extension has already been published or backup is occupied")
    stage_summary, stage_rows, _ = _load_delivery(
        STAGED_INVENTORY, STAGED_FIELDS, STAGED_VALUES, scan_storage=False)
    if stage_summary["source_schema_closure"] != "complete":
        raise ValueError("staged source schema closure is incomplete")
    baseline, published, _ = _load_delivery(INVENTORY, FIELDS, VALUES,
                                           scan_storage=False,
                                           published_baseline=True)
    expected_rows, current, old_to_new = _current_rows(published)
    for (index, declaration), expected, staged in zip(
            current, expected_rows, stage_rows, strict=True):
        if (_role(declaration) != _published_role(staged) or
                index != staged["declaration_index"] or
                expected["structure_status"] != staged["structure_status"] or
                expected["presence"] != staged["presence"] or
                expected["input_bytes"] != staged["input_bytes"]):
            raise ValueError("staged declaration identity differs from current sources")
    _validate_staged_mapping(baseline["storage"], stage_summary["storage"],
                             old_to_new)
    with STAGED_ACCEPTANCE.open("r", encoding="utf-8") as stream:
        acceptance = json.load(stream)
    if (acceptance.get("kind") != "structure_acceptance" or
            acceptance.get("inventory_path") != source_path(INVENTORY) or
            acceptance.get("fields_path") != source_path(FIELDS) or
            acceptance.get("values_root") != source_path(VALUES) or
            acceptance.get("migration_mapping_path") != source_path(MAPPING) or
            acceptance.get("declarations") != stage_summary["declarations"] or
            acceptance.get("records") != stage_summary["records"] or
            acceptance.get("source_schema_closure") != stage_summary["source_schema_closure"] or
            acceptance.get("schema_coverage") != stage_summary["schema_coverage"] or
            acceptance.get("grammar_validated_records") !=
            stage_summary["grammar_validated_records"] or
            acceptance.get("storage_validation") != "complete_staged_parquet_scan"):
        raise ValueError("staged acceptance differs from the delivered inventory")
    # G7 requires the two volumes to stay separate and to agree with the real
    # delivery, not only with the counters that produced them.  Re-read them
    # from the published baseline and the staged inventory here, and reject the
    # retired family-specific names so a stale acceptance cannot relabel the
    # seven-family appended block as initial events.
    if ("initial_event_records_appended" in acceptance or
            "appended_initial_event_records" in acceptance):
        raise ValueError("staged acceptance carries a retired volume name")
    retained_count = sum(index is not None for index in old_to_new)
    migrated_records = acceptance.get("migrated_records")
    appended_records = acceptance.get("appended_records")
    if type(migrated_records) is not int or type(appended_records) is not int:
        raise ValueError("staged acceptance has no migrated/appended volume split")
    real_migrated = sum(row["records"] for row in stage_rows[:retained_count])
    real_appended = sum(row["records"] for row in stage_rows[retained_count:])
    if (migrated_records != baseline["records"] or
            migrated_records != real_migrated or
            appended_records != real_appended or
            migrated_records + appended_records != stage_summary["records"]):
        raise ValueError("staged acceptance volumes differ from the delivered inventory")
    staged_by_index = {row["declaration_index"]: row for row in stage_rows}
    expected_indexes = {
        index for index, row in staged_by_index.items()
        if row["record_grammar_status"] == "closed_record_grammar" and row["records"]
    }
    if type(acceptance.get("validated_sources")) is not list:
        raise ValueError("staged grammar audits are absent")
    validated = Counter()
    seen: set[int] = set()
    for item in acceptance["validated_sources"]:
        if type(item) is not dict or type(item.get("declaration_index")) is not int:
            raise ValueError("staged grammar audit has no declaration index")
        index = item["declaration_index"]
        if index in seen or index not in expected_indexes:
            raise ValueError(f"staged grammar audit declaration differs: {index}")
        row = staged_by_index[index]
        span = row["shard_spans"][-1]
        expected = {
            "declaration_index": index,
            "source_family": row["source_family"],
            "source_file": row["source_file"],
            "records": row["grammar_validated_records"],
            "effective_records": row["grammar_validated_effective_records"],
            "last_record_ordinal": span["last_ordinal"],
            "last_source_line": span["last_line"],
        }
        if item != expected or item["records"] != row["records"]:
            raise ValueError(f"staged grammar audit source differs: {row['source_file']}")
        seen.add(index)
        validated[item["source_family"]] += item["records"]
    if (seen != expected_indexes or
            dict(sorted(validated.items())) != stage_summary["grammar_validated_records"]):
        raise ValueError("staged grammar audit counts differ from source inventory")
    # _deliver_staged writes acceptance only after the full Parquet scan and
    # value round trips have succeeded. Reuse that result as resume-stage does:
    # check every shard path and table size plus the final real record, rather
    # than repeating the same row-level scan before an otherwise read-only
    # cutover. Catalog, declaration, mapping and audit checks above still run.
    StructureStorageWriter.resume_from_validated(
        STAGED_VALUES, _storage_summary(stage_summary["storage"]))
    try:
        for final, backup in zip(_EXISTING_FINALS, _BACKUPS):
            os.replace(final, backup)
        for staged, final in zip(_STAGED, _FINALS):
            os.replace(staged, final)
    except BaseException:
        _restore_cutover()
        raise
    for backup in _BACKUPS:
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()
    # Published baseline has changed, so the audit checkpoint keyed on the
    # previous one is dropped here and only here.  A failed or interrupted stage
    # keeps it for the next run.
    clear_default(VALUES.parent)
    result = {key: value for key, value in stage_summary.items() if key != "storage"}
    result.update(rewritten_record_shards=acceptance["record_index_shards_rewritten"],
                  reclassified_metadata_gaps=acceptance["historical_metadata_gaps_reclassified"],
                  migrated_records=acceptance["migrated_records"],
                  appended_records=acceptance["appended_records"],
                  shards=len(stage_summary["storage"]["shards"]), phase="published")
    print(_line(result).rstrip(), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--workers", type=int, default=16)
    resume = commands.add_parser("resume-stage")
    resume.add_argument("--workers", type=int, default=16)
    commands.add_parser("publish")
    args = parser.parse_args()
    if args.command == "stage":
        stage_extension(workers=args.workers)
    elif args.command == "resume-stage":
        resume_stage_extension(workers=args.workers)
    else:
        publish_extension()
