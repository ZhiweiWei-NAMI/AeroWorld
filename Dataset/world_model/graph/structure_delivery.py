"""Build and inspect the structural source delivery.

The build reads each declared source once. The inventory is published last;
its summary is the marker that the three Parquet tables and field catalog were
validated together. This module assigns no field meaning, unit, or coordinate.
"""

from __future__ import annotations

from Dataset.semantic_truth.provenance import open_artifact_binary
import argparse
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterator

import orjson

from Dataset.world_model.graph.contract import REPO, source_path
from Dataset.world_model.graph.structure_closure import (
    declaration_not_applicable, declared_effective_families,
    reviewed_source_grammar_declaration,
)
from Dataset.world_model.graph.field_catalog import SourceDeclaration
from Dataset.world_model.graph.sample_identity import identity_envelope, inventory_scope
from Dataset.world_model.graph.structure_catalog import (
    CatalogRows, StructureCatalog, validate_catalog_rows,
)
from Dataset.world_model.graph.structure_recovery import (
    RecoveredDeclaration, RecoveredStructureShard, recover_structure_shard,
)
from Dataset.world_model.graph.structure_sources import (
    SourceRecord, SourceResumeCursor, _source_record, iter_source_records,
    ordered_source_declarations, project_record,
)
from Dataset.world_model.graph.structure_grammar_catalog import grammar_declaration_status
from Dataset.world_model.graph.structure_storage import (
    EncodedRecordSummary, EncodedStructureBatch, RecordLocation, StorageSummary,
    StorageResumePoint, StructureStorageWriter, _decoded_row, _is_json_root,
    _same_typed, _shard_dirs, encode_structure_batch, inspect_structure_record,
    validate_structure_storage,
)


FORMAT_VERSION = "structure-1"
INVENTORY = REPO / "design/belief/structure_source_inventory.jsonl"
FIELDS = REPO / "design/belief/structure_fields.jsonl"
VALUES = REPO / "Dataset/world_model/runs/belief_structure"
STAGED_INVENTORY = INVENTORY.with_name(INVENTORY.name + ".tmp")
STAGED_FIELDS = FIELDS.with_name(FIELDS.name + ".tmp")
STAGED_VALUES = VALUES.with_name(VALUES.name + ".tmp")
ACCEPTANCE = INVENTORY.with_name("structure_acceptance.json")
_OUTPUT_PATHS = (INVENTORY, FIELDS, VALUES)
_STAGED_PATHS = (STAGED_INVENTORY, STAGED_FIELDS, STAGED_VALUES)
_REBUILD_BACKUPS = tuple(
    path.with_name(path.name + ".rebuild_backup") for path in _OUTPUT_PATHS)
_SOURCE_STATUSES = frozenset({
    "projected", "zero_instance", "source_missing", "structure_pending",
    "structure_pending_missing",
})
_BATCH_16_FAMILIES = frozenset({
    "arm_actions", "arm_chain_events", "arm_chain_nodes", "arm_chain_relations",
    "arm_event_occurrences", "arm_event_outcomes", "arm_local_business_truth",
    "arm_predicate_transitions", "arm_predicate_truth",
    "arm_predicate_truth_ticks", "arm_states", "arm_trajectories",
    "communication_state", "compute_state", "domain_state", "events",
    "formal_event_chain_relations", "formal_event_realization",
    "formal_weather_meta", "predicate_truth", "predicate_truth_matrix",
})


@dataclass(frozen=True)
class _Anchor:
    declaration_index: int
    record_ordinal: int
    source_line: int | None
    shard_id: int
    projection_count: int
    source_leaf_count: int


@dataclass
class _Progress:
    started: float
    last_report: float
    declarations: int = 0
    records: int = 0
    projections: int = 0
    source_leaves: int = 0
    source_file: str | None = None
    record_ordinal: int | None = None
    source_line: int | None = None

    def report(self) -> None:
        now = time.monotonic()
        if now - self.last_report < 600:
            return
        print(_line({
            "declarations_completed": self.declarations,
            "records": self.records,
            "projections": self.projections,
            "source_leaves": self.source_leaves,
            "current_source": self.source_file,
            "last_completed_record_ordinal": self.record_ordinal,
            "last_completed_source_line": self.source_line,
            "elapsed_seconds": round(now - self.started, 3),
        }).rstrip(), flush=True)
        self.last_report = now


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def _write_jsonl(path: Path, rows: Iterator[dict[str, Any]], *,
                 overwrite: bool = False) -> None:
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_line(row))


def _declaration_row(index: int, declaration: Any) -> dict[str, Any]:
    presence = declaration.presence
    if declaration.structure_status == "field_scan_defined":
        status = "source_missing" if presence == "missing" else "zero_instance"
    elif declaration.structure_status == "source_schema_pending":
        status = ("structure_pending_missing" if presence == "missing"
                  else "structure_pending")
    else:
        raise ValueError(f"unknown source structure status: {declaration.structure_status}")
    return {
        "kind": "source_declaration", "declaration_index": index,
        "source_family": declaration.source_family,
        "source_file": source_path(declaration.path),
        "declared_sample": declaration.sample,
        "identity_status": declaration.identity_status,
        "source_scope": None,
        "reference_samples": list(declaration.reference_samples),
        "structure_status": declaration.structure_status,
        "presence": presence, "read_relation": declaration.read_relation,
        "read_condition": declaration.read_condition,
        "input_bytes": declaration.path.stat().st_size if presence == "present" else None,
        "delivery_status": status,
        "effective_families": [], "records": 0, "projections": 0,
        "record_grammar_status": grammar_declaration_status(declaration),
        "grammar_validated_records": 0,
        "grammar_validated_effective_records": {},
        "source_leaves": 0, "structure_conflict_projections": 0,
        "excluded_integrity_subtrees": 0, "shard_spans": [],
    }


def _accept_record(row: dict[str, Any], record: SourceRecord,
                   summary: EncodedRecordSummary, location: RecordLocation) -> None:
    if row["declaration_index"] != record.declaration_index:
        raise ValueError(f"source record and declaration differ: {record.source_file}")
    if row["source_scope"] is None and record.source_scope is not None:
        row["source_scope"] = identity_envelope(record.source_scope)
    row["records"] += 1
    if row["record_grammar_status"] == "closed_record_grammar":
        row["grammar_validated_records"] += 1
        effective_counts = row["grammar_validated_effective_records"]
        effective_counts[record.effective_family] = (
            effective_counts.get(record.effective_family, 0) + 1)
    row["projections"] += summary.projection_count
    row["source_leaves"] += summary.source_leaf_count
    row["structure_conflict_projections"] += summary.structure_conflict_projections
    row["excluded_integrity_subtrees"] += summary.excluded_integrity_subtrees
    if record.effective_family not in row["effective_families"]:
        row["effective_families"].append(record.effective_family)
    spans = row["shard_spans"]
    if spans and spans[-1]["shard_id"] == location.shard_id:
        spans[-1]["last_ordinal"] = record.record_ordinal
        spans[-1]["last_line"] = record.source_line
    else:
        spans.append({
            "shard_id": location.shard_id,
            "first_ordinal": record.record_ordinal,
            "last_ordinal": record.record_ordinal,
            "first_line": record.source_line,
            "last_line": record.source_line,
        })


def _consume_batch(records: tuple[SourceRecord, ...], batch: EncodedStructureBatch,
                   catalog: StructureCatalog, writer: StructureStorageWriter,
                   row: dict[str, Any], anchors: dict[str, _Anchor],
                   progress: _Progress) -> None:
    if len(batch.record_summaries) != len(records):
        raise ValueError(f"encoded batch record count differs: {row['source_file']}")
    for record, summary in zip(records, batch.record_summaries):
        if ((summary.declaration_index, summary.record_ordinal,
             summary.source_line, summary.source_file,
             summary.excluded_integrity_subtrees) !=
            (record.declaration_index, record.record_ordinal,
             record.source_line, record.source_file,
             record.excluded_integrity_subtrees)):
            raise ValueError(f"encoded batch record differs: {record.source_file}:"
                             f"{record.source_line}")
    index_maps = catalog._merge_validated_fragment(batch.catalog_fragment)
    locations = writer.write_batch(batch, index_maps)
    if len(locations) != len(records):
        raise ValueError(f"stored batch record count differs: {row['source_file']}")
    for record, summary, location in zip(records, batch.record_summaries, locations):
        _accept_record(row, record, summary, location)
        progress.records += 1
        progress.projections += summary.projection_count
        progress.source_leaves += summary.source_leaf_count
        progress.record_ordinal = record.record_ordinal
        progress.source_line = record.source_line
        progress.report()
        anchor = _Anchor(record.declaration_index, record.record_ordinal,
                         record.source_line, location.shard_id,
                         summary.projection_count, summary.source_leaf_count)
        anchors.setdefault(record.source_family, anchor)
        if summary.structure_conflict_projections:
            anchors.setdefault("structure_conflict", anchor)


def _scan_declaration(index: int, declaration: Any, row: dict[str, Any],
                      catalog: StructureCatalog, writer: StructureStorageWriter,
                      executor: ProcessPoolExecutor | None, workers: int,
                      anchors: dict[str, _Anchor], progress: _Progress, *,
                      resume_after: SourceResumeCursor | None = None) -> None:
    if (declaration.structure_status != "field_scan_defined" or
            declaration.presence != "present"):
        return
    pending: deque[tuple[tuple[SourceRecord, ...], Future[EncodedStructureBatch]]] = deque()

    def consume_one() -> None:
        records, future = pending.popleft()
        try:
            batch = future.result()
            _consume_batch(records, batch, catalog, writer, row, anchors, progress)
        except Exception as exc:
            first, last = records[0], records[-1]
            raise RuntimeError(f"structure batch failed: {first.source_file}:"
                               f"records {first.record_ordinal}..{last.record_ordinal} "
                               f"lines {first.source_line}..{last.source_line}") from exc

    try:
        source_records = iter_source_records(index, declaration, resume_after=resume_after)
        batch_size = (16 if declaration.source_family in _BATCH_16_FAMILIES and
                      declaration.path.suffix == ".jsonl" else 1)
        while True:
            if len(pending) >= 2 * workers:
                consume_one()
            records = tuple(islice(source_records, batch_size))
            if not records:
                break
            projected_records = (tuple(project_record(record) for record in records)
                                 if declaration.source_family == "domain_state" else None)
            if executor is None:
                batch = encode_structure_batch(records, projected_records)
                _consume_batch(records, batch, catalog, writer, row, anchors,
                               progress)
            else:
                pending.append((records, executor.submit(
                    encode_structure_batch, records, projected_records)))
        while pending:
            consume_one()
    except Exception as exc:
        raise RuntimeError(f"structure source failed: {row['source_file']}: {exc}") from exc
    row["effective_families"].sort()
    if row["projections"]:
        row["delivery_status"] = "projected"


def _declaration_from_row(row: dict[str, Any]) -> SourceDeclaration:
    return SourceDeclaration(
        row["source_family"], REPO / row["source_file"],
        row["declared_sample"], row["structure_status"],
        row["read_relation"], row["read_condition"],
        tuple(row["reference_samples"]),
    )


def _validate_record_grammar_source(
    index: int, declaration: SourceDeclaration, row: dict[str, Any], *,
    complete: bool = True,
) -> dict[str, Any]:
    """Re-read one real source through the same validator as the delivery."""
    if (row["declaration_index"] != index or
            row["source_family"] != declaration.source_family or
            row["source_file"] != source_path(declaration.path) or
            row["presence"] != "present" or row["records"] < 1 or
            not reviewed_source_grammar_declaration(declaration)):
        raise ValueError(f"grammar audit received an unreviewed source: {row['source_file']}")
    if (not declaration.path.is_file() or
            declaration.path.stat().st_size != row["input_bytes"]):
        raise ValueError(f"grammar source bytes or presence changed: {row['source_file']}")
    permitted = set(declared_effective_families(declaration))
    effective_counts: Counter[str] = Counter()
    records = 0
    last_line: int | None = None
    source_records = iter_source_records(index, declaration)
    try:
        for record in islice(source_records, row["records"]):
            if (record.record_ordinal != records or
                    record.source_file != row["source_file"] or
                    record.effective_family not in permitted):
                raise ValueError(
                    f"{row['source_file']}:record_ordinal={record.record_ordinal},"
                    f"source_line={record.source_line}: grammar audit role differs")
            effective_counts[record.effective_family] += 1
            records += 1
            last_line = record.source_line
        if complete:
            extra = next(source_records, None)
            if extra is not None:
                raise ValueError(
                    f"{row['source_file']}:record_ordinal={extra.record_ordinal},"
                    f"source_line={extra.source_line}: source has more records than inventory")
    except Exception as exc:
        detail = str(exc)
        if "record_ordinal=" not in detail:
            source_line = ("None" if declaration.path.suffix in {".json", ".gz"}
                           else "unknown")
            detail = (f"{row['source_file']}:record_ordinal={records},"
                      f"source_line={source_line}: {detail}")
        raise ValueError(f"grammar validation failed: {detail}") from exc
    if records != row["records"] or set(effective_counts) != set(row["effective_families"]):
        raise ValueError(
            f"{row['source_file']}:record_ordinal={records},source_line={last_line}: "
            "grammar audit count or effective types differ from inventory")
    if row["shard_spans"]:
        span = row["shard_spans"][-1]
        if (span["last_ordinal"] != records - 1 or
                span["last_line"] != last_line):
            raise ValueError(
                f"{row['source_file']}:record_ordinal={records - 1},"
                f"source_line={last_line}: stored record position differs")
    return {
        "declaration_index": index,
        "source_family": declaration.source_family,
        "source_file": row["source_file"],
        "records": records,
        "effective_records": dict(sorted(effective_counts.items())),
        "last_record_ordinal": records - 1,
        "last_source_line": last_line,
    }


def _declared_profile(row: dict[str, Any]) -> str | None:
    sample = row["declared_sample"]
    if sample.startswith("arms/"):
        parts = sample.split("/")
        if len(parts) != 5 or not all(parts[1:]):
            raise ValueError(f"ARM declaration sample is malformed: {sample}")
        return parts[1]
    profile, separator, seed = sample.rpartition("__")
    if separator:
        if profile and len(seed) == 6 and seed.startswith("seed") and seed[4:].isdigit():
            return profile
        raise ValueError(f"declaration sample has a malformed profile: {sample}")
    if row["source_family"] in {"formal_roster", "formal_weather_meta"}:
        raise ValueError(f"formal declaration has no profile: {sample}")
    return None


def _schema_coverage(
    rows: list[dict[str, Any]], conflicts: int, *,
    published_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute current closure or the explicit pre-extension published batch."""
    published = published_coverage is not None
    families: dict[str, dict[str, Any]] = {}
    uncovered = unvalidated = missing = without_instances = not_applicable = 0
    for row in rows:
        declaration = _declaration_from_row(row)
        family = declaration.source_family
        status = row["record_grammar_status"]
        if not published and status != grammar_declaration_status(declaration):
            raise ValueError(f"record grammar status differs: {row['source_file']}")
        if status == "not_applicable":
            if (published or not declaration_not_applicable(declaration) or
                    row["presence"] != "missing" or row["records"]):
                raise ValueError(f"invalid non-applicable source: {row['source_file']}")
            not_applicable += 1
        validated = row["grammar_validated_records"]
        effective_counts = row["grammar_validated_effective_records"]
        if (type(validated) is not int or not 0 <= validated <= row["records"] or
                type(effective_counts) is not dict or
                any(type(count) is not int or count < 1
                    for count in effective_counts.values()) or
                sum(effective_counts.values()) != validated or
                (status != "closed_record_grammar" and
                 (validated or effective_counts))):
            raise ValueError(f"invalid grammar validation counts: {row['source_file']}")
        item = families.setdefault(family, {
            "declarations": 0, "records": 0, "grammar_validated_records": 0,
            "pending_declarations": 0, "missing_declarations": 0,
            "zero_instance_declarations": 0, "declared_without_instances": 0,
            "grammar_statuses": {}, "profiles": {}, "effective_families": {},
        })
        if not published:
            item.setdefault("not_applicable_declarations", 0)
        item["declarations"] += 1
        item["records"] += row["records"]
        item["grammar_validated_records"] += validated
        statuses = item["grammar_statuses"]
        statuses[status] = statuses.get(status, 0) + 1
        if status == "not_applicable":
            item["not_applicable_declarations"] += 1
        elif status != "closed_record_grammar":
            uncovered += 1
        else:
            unvalidated += row["records"] - validated
        if row["structure_status"] != "field_scan_defined":
            item["pending_declarations"] += 1
        if row["presence"] == "missing":
            item["missing_declarations"] += 1
            missing += 1
        if row["delivery_status"] == "zero_instance":
            item["zero_instance_declarations"] += 1
        if (status == "closed_record_grammar" and
                row["presence"] == "present" and row["records"] == 0):
            item["declared_without_instances"] += 1
            without_instances += 1
        profile = _declared_profile(row) or "shared"
        profile_item = item["profiles"].setdefault(profile, {
            "declarations": 0, "records": 0, "grammar_validated_records": 0,
            "grammar_statuses": {}, "effective_families": {},
        })
        profile_item["declarations"] += 1
        profile_item["records"] += row["records"]
        profile_item["grammar_validated_records"] += validated
        profile_statuses = profile_item["grammar_statuses"]
        profile_statuses[status] = profile_statuses.get(status, 0) + 1
        if status == "closed_record_grammar":
            if published:
                try:
                    effective_families = tuple(
                        published_coverage["families"][family]["profiles"]
                        [profile]["effective_families"])
                except (KeyError, TypeError) as exc:
                    raise ValueError(
                        f"published effective types are absent: {family}:{profile}") from exc
            else:
                effective_families = declared_effective_families(declaration)
        else:
            effective_families = tuple(row["effective_families"])
        if status == "closed_record_grammar" and (
                set(row["effective_families"]) - set(effective_families) or
                set(effective_counts) != set(row["effective_families"])):
            raise ValueError(f"validated effective types differ: {row['source_file']}")
        for effective in effective_families:
            effective_item = item["effective_families"].setdefault(effective, {
                "declarations": 0, "observed_declarations": 0,
                "grammar_validated_records": 0,
            })
            effective_item["declarations"] += 1
            effective_item["observed_declarations"] += (
                effective in row["effective_families"])
            effective_item["grammar_validated_records"] += effective_counts.get(
                effective, 0)
            profile_effective = profile_item["effective_families"].setdefault(
                effective, {
                    "declarations": 0, "observed_declarations": 0,
                    "grammar_validated_records": 0,
                })
            profile_effective["declarations"] += 1
            profile_effective["observed_declarations"] += (
                effective in row["effective_families"])
            profile_effective["grammar_validated_records"] += effective_counts.get(
                effective, 0)
    for item in families.values():
        statuses = item["grammar_statuses"]
        item["schema_status"] = (next(iter(statuses)) if len(statuses) == 1
                                 else "mixed")
        item["grammar_statuses"] = dict(sorted(statuses.items()))
        item["profiles"] = dict(sorted(item["profiles"].items()))
        item["effective_families"] = dict(sorted(item["effective_families"].items()))
        for profile in item["profiles"].values():
            profile["grammar_statuses"] = dict(sorted(profile["grammar_statuses"].items()))
            profile["effective_families"] = dict(sorted(profile["effective_families"].items()))
    result = {
        "families": dict(sorted(families.items())),
        "uncovered_declarations": uncovered,
        "unvalidated_closed_grammar_records": unvalidated,
        "missing_declarations": missing,
        "declared_without_instances": without_instances,
        "structure_conflict_projections": conflicts,
        "closure": ("complete" if not (uncovered or unvalidated or
                                       (missing if published else 0) or conflicts)
                    else "incomplete"),
    }
    if not published:
        result["not_applicable_declarations"] = not_applicable
    return result


def _summary(rows: list[dict[str, Any]], catalog: CatalogRows,
             storage: StorageSummary, elapsed: float) -> dict[str, Any]:
    counts = Counter(row["delivery_status"] for row in rows)
    totals = {name: sum(row[name] for row in rows) for name in (
        "records", "projections", "source_leaves",
        "structure_conflict_projections", "excluded_integrity_subtrees")}
    if (totals["records"], totals["projections"], totals["source_leaves"],
            totals["excluded_integrity_subtrees"]) != (
            storage.records, storage.projections, storage.source_leaves,
            storage.excluded_integrity_subtrees):
        raise ValueError("source inventory totals differ from structure storage")
    if (sum(row["instance_count"] for row in catalog.normalized_fields)
            != storage.projections or
            sum(row["instance_count"] for row in catalog.source_fields)
            != storage.source_leaves):
        raise ValueError("structure catalog observations differ from storage")
    if (sum(row["structure_status_counts"].get("structure_conflict", 0)
            for row in catalog.normalized_fields)
            != totals["structure_conflict_projections"]):
        raise ValueError("structure conflicts differ from catalog observations")
    if not storage.records or not counts["projected"]:
        raise ValueError("no readable structural source was materialized")
    coverage = _schema_coverage(rows, totals["structure_conflict_projections"])
    if coverage["unvalidated_closed_grammar_records"]:
        raise ValueError("scanned records lack complete grammar validation")
    validated = Counter()
    for row in rows:
        validated[row["source_family"]] += row["grammar_validated_records"]
    return {
        "kind": "structure_source_inventory_summary",
        "format_version": FORMAT_VERSION,
        "delivery_status": "all_defined_present_sources_scanned",
        "source_schema_closure": coverage["closure"],
        "grammar_validated_records": dict(sorted(
            (family, count) for family, count in validated.items() if count)),
        "schema_coverage": coverage,
        "declarations": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "pending_declarations": counts["structure_pending"] + counts["structure_pending_missing"],
        "missing_declarations": counts["source_missing"] + counts["structure_pending_missing"],
        "zero_instance_declarations": counts["zero_instance"],
        **totals,
        "catalog_source_fields": len(catalog.source_fields),
        "catalog_normalized_fields": len(catalog.normalized_fields),
        "storage": _jsonable(asdict(storage)),
        "elapsed_seconds": round(elapsed, 3),
    }


def _inventory_rows(rows: list[dict[str, Any]],
                    summary: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield {
        "kind": "structure_source_inventory_header",
        "format_version": FORMAT_VERSION,
        "fields_path": source_path(FIELDS), "values_root": source_path(VALUES),
        "source_leaf_coverage": "each permitted JSON leaf once per physical record",
        "integrity_exclusion": "integrity-key subtrees counted separately",
    }
    yield from rows
    yield summary


def _field_rows(catalog: CatalogRows) -> Iterator[dict[str, Any]]:
    yield {
        "kind": "structure_fields_header", "format_version": FORMAT_VERSION,
        "inventory_path": source_path(INVENTORY),
        "values_root": source_path(VALUES),
        "index_policy": "first_observation_then_exact_zero_instance_bindings",
        "source_fields": len(catalog.source_fields),
        "normalized_fields": len(catalog.normalized_fields),
    }
    yield from catalog.source_fields
    yield from catalog.normalized_fields


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if any(type(row) is not dict for row in rows):
        raise ValueError(f"structure index has a non-object row: {path}")
    return rows


def _load_delivery(inventory_path: Path, fields_path: Path,
                   values_root: Path, *, scan_storage: bool,
                   published_baseline: bool = False,
                   ) -> tuple[dict[str, Any], list[dict[str, Any]], CatalogRows]:
    inventory = _read_jsonl(inventory_path)
    field_rows = _read_jsonl(fields_path)
    if len(inventory) < 2 or not field_rows:
        raise ValueError("structure delivery has no final inventory or field header")
    header, summary = inventory[0], inventory[-1]
    field_header = field_rows[0]
    if (header.get("kind") != "structure_source_inventory_header" or
            header.get("format_version") != FORMAT_VERSION or
            header.get("fields_path") != source_path(FIELDS) or
            header.get("values_root") != source_path(VALUES) or
            summary.get("kind") != "structure_source_inventory_summary" or
            summary.get("format_version") != FORMAT_VERSION or
            summary.get("delivery_status") != "all_defined_present_sources_scanned" or
            summary.get("source_schema_closure") not in {"complete", "incomplete"} or
            field_header.get("kind") != "structure_fields_header" or
            field_header.get("format_version") != FORMAT_VERSION or
            field_header.get("inventory_path") != source_path(INVENTORY) or
            field_header.get("values_root") != source_path(VALUES)):
        raise ValueError("structure delivery headers or completion summary differ")
    declarations = inventory[1:-1]
    if (len(declarations) != summary.get("declarations") or
            any(row.get("kind") != "source_declaration" or
                row.get("declaration_index") != index or
                row.get("delivery_status") not in _SOURCE_STATUSES
                for index, row in enumerate(declarations))):
        raise ValueError("structure declaration inventory is incomplete or unordered")
    for row in declarations:
        structure, presence = row["structure_status"], row["presence"]
        if (structure not in {"field_scan_defined", "source_schema_pending"} or
                presence not in {"present", "missing"}):
            raise ValueError("structure declaration has an unknown source state")
        if structure == "source_schema_pending":
            expected = ("structure_pending" if presence == "present"
                        else "structure_pending_missing")
        elif presence == "missing":
            expected = "source_missing"
        else:
            expected = "projected" if row["projections"] else "zero_instance"
        if row["delivery_status"] != expected:
            raise ValueError(f"structure source status differs: {row['source_file']}")
        for name in ("records", "projections", "source_leaves",
                     "structure_conflict_projections", "excluded_integrity_subtrees"):
            if type(row[name]) is not int or row[name] < 0:
                raise ValueError(f"structure source count is invalid: {row['source_file']}:{name}")
        if (row["structure_conflict_projections"] > row["projections"] or
                row["source_leaves"] < row["projections"]):
            raise ValueError(f"structure source field counts differ: {row['source_file']}")
        if expected in {"source_missing", "structure_pending", "structure_pending_missing"} and (
                row["records"] or row["projections"] or row["source_leaves"] or
                row["shard_spans"]):
            raise ValueError(f"unread source has stored values: {row['source_file']}")
        next_ordinal = 0
        for span in row["shard_spans"]:
            if (span["first_ordinal"] != next_ordinal or
                    span["last_ordinal"] < next_ordinal or
                    type(span["shard_id"]) is not int or span["shard_id"] < 0):
                raise ValueError(f"source shard spans are not ordered: {row['source_file']}")
            next_ordinal = span["last_ordinal"] + 1
        if next_ordinal != row["records"]:
            raise ValueError(f"source shard spans lose records: {row['source_file']}")
    counts = Counter(row["delivery_status"] for row in declarations)
    if dict(sorted(counts.items())) != summary.get("status_counts"):
        raise ValueError("structure declaration status counts differ")
    if (counts["structure_pending"] + counts["structure_pending_missing"]
            != summary.get("pending_declarations") or
            counts["source_missing"] + counts["structure_pending_missing"]
            != summary.get("missing_declarations") or
            counts["zero_instance"] != summary.get("zero_instance_declarations")):
        raise ValueError("structure declaration gap counts differ")
    for name in ("records", "projections", "source_leaves",
                 "structure_conflict_projections", "excluded_integrity_subtrees"):
        if sum(row[name] for row in declarations) != summary.get(name):
            raise ValueError(f"structure declaration {name} differs from summary")
    if "schema_coverage" in summary:
        for row in declarations:
            if any(name not in row for name in (
                    "record_grammar_status", "grammar_validated_records",
                    "grammar_validated_effective_records")):
                raise ValueError(f"record grammar audit is absent: {row['source_file']}")
        coverage = _schema_coverage(
            declarations, summary["structure_conflict_projections"],
            published_coverage=(summary["schema_coverage"]
                                if published_baseline else None))
        if coverage["unvalidated_closed_grammar_records"]:
            raise ValueError("delivered records lack complete grammar validation")
        validated = Counter()
        for row in declarations:
            validated[row["source_family"]] += row["grammar_validated_records"]
        expected_validated = dict(sorted(
            (family, count) for family, count in validated.items() if count))
        if (summary["source_schema_closure"] != coverage["closure"] or
                summary["schema_coverage"] != coverage or
                summary.get("grammar_validated_records") != expected_validated):
            raise ValueError("source schema closure differs from declared coverage")
    elif (summary["source_schema_closure"] != "incomplete" or
          "grammar_validated_records" in summary or
          any("record_grammar_status" in row or
              "grammar_validated_records" in row or
              "grammar_validated_effective_records" in row
              for row in declarations)):
        raise ValueError("published source inventory lacks its grammar audit")
    sources = tuple(row for row in field_rows[1:] if row.get("kind") == "source_field")
    normalized = tuple(row for row in field_rows[1:] if row.get("kind") == "normalized_field")
    if (len(sources) + len(normalized) != len(field_rows) - 1 or
            len(sources) != field_header.get("source_fields") or
            len(normalized) != field_header.get("normalized_fields") or
            len(sources) != summary.get("catalog_source_fields") or
            len(normalized) != summary.get("catalog_normalized_fields")):
        raise ValueError("structure field index is incomplete")
    catalog = CatalogRows(sources, normalized)
    validate_catalog_rows(catalog)
    if (sum(row["instance_count"] for row in normalized) != summary["projections"] or
            sum(row["instance_count"] for row in sources) != summary["source_leaves"]):
        raise ValueError("structure field observations differ from inventory")
    if (sum(row["structure_status_counts"].get("structure_conflict", 0)
            for row in normalized) != summary["structure_conflict_projections"]):
        raise ValueError("structure conflict counts differ from field index")
    if not values_root.is_dir():
        raise FileNotFoundError(values_root)
    if scan_storage:
        actual = _jsonable(asdict(validate_structure_storage(values_root)))
        if actual != summary["storage"]:
            raise ValueError("structure Parquet summary differs from published inventory")
    return summary, declarations, catalog


def _check_anchors(root: Path, anchors: dict[str, _Anchor]) -> None:
    for anchor in anchors.values():
        result = inspect_structure_record(
            root, anchor.declaration_index, anchor.record_ordinal,
            source_line=anchor.source_line, shard_id=anchor.shard_id)
        record = result["record"]
        if (record["record_ordinal"] != anchor.record_ordinal or
                record["projection_count"] != anchor.projection_count or
                record["source_leaf_count"] != anchor.source_leaf_count or
                len(result["projections"]) != anchor.projection_count):
            raise ValueError("targeted structural value round trip differs")


def _matches_source_family(source_family: str, effective_family: str) -> bool:
    if source_family in {"arm_chain_nodes", "arm_chain_events"}:
        return (effective_family.startswith(source_family + ":") and
                len(effective_family) > len(source_family) + 1)
    if source_family == "arm_predicate_truth_ticks":
        return effective_family in {"arm_predicate_truth_ticks", "arm_domain_state_ticks"}
    return effective_family == source_family


def _accept_recovered_declaration(
        declarations: tuple[tuple[int, Any], ...], rows: list[dict[str, Any]],
        item: RecoveredDeclaration, shard_id: int) -> None:
    index = item.declaration_index
    if not 0 <= index < len(declarations):
        raise ValueError(f"stored declaration index is absent: {index}")
    declaration = declarations[index][1]
    row = rows[index]
    if (declaration.structure_status != "field_scan_defined" or
            declaration.presence != "present" or
            item.records < 1 or item.first_record_ordinal != row["records"] or
            item.last_record_ordinal != item.first_record_ordinal + item.records - 1 or
            any(not _matches_source_family(declaration.source_family, family)
                for family in item.effective_families)):
        raise ValueError(f"stored records differ from current declaration: {row['source_file']}")
    # Same root/row classification as structure_sources.iter_source_records: a
    # .json document root, or the registered arm_semantic_history gzip root.
    if _is_json_root(declaration):
        if (item.records != 1 or item.first_source_line is not None or
                item.last_source_line is not None):
            raise ValueError(f"stored JSON root position differs: {row['source_file']}")
    elif declaration.path.suffix == ".jsonl":
        if (item.first_source_line is None or item.last_source_line is None or
                item.first_source_line > item.last_source_line):
            raise ValueError(f"stored JSONL lines differ: {row['source_file']}")
        if (row["shard_spans"] and
                item.first_source_line <= row["shard_spans"][-1]["last_line"]):
            raise ValueError(f"stored JSONL lines are not increasing: {row['source_file']}")
    else:
        raise ValueError(f"stored source has no structural reader: {row['source_file']}")
    scope = inventory_scope(
        declaration.source_family, declaration.path, declaration.sample)
    expected_scope = identity_envelope(scope) if scope is not None else None
    if row["source_scope"] is not None and row["source_scope"] != expected_scope:
        raise ValueError(f"stored source scope differs: {row['source_file']}")
    row["source_scope"] = expected_scope
    for name in ("records", "projections", "source_leaves",
                 "structure_conflict_projections", "excluded_integrity_subtrees"):
        row[name] += getattr(item, name)
    for family in item.effective_families:
        if family not in row["effective_families"]:
            row["effective_families"].append(family)
    row["shard_spans"].append({
        "shard_id": shard_id,
        "first_ordinal": item.first_record_ordinal,
        "last_ordinal": item.last_record_ordinal,
        "first_line": item.first_source_line,
        "last_line": item.last_source_line,
    })


def _recover_prefix(
        declarations: tuple[tuple[int, Any], ...], rows: list[dict[str, Any]],
        catalog: StructureCatalog, shard_ids: tuple[int, ...],
        tail_shard: RecoveredStructureShard, progress: _Progress, workers: int
        ) -> tuple[StorageSummary, StorageResumePoint]:
    last_point: StorageResumePoint | None = None
    shard_summaries = []

    def consume(shard_id: int, recovered: RecoveredStructureShard) -> None:
        nonlocal last_point
        if (recovered.shard_summary.shard_id != shard_id or
                recovered.shard_summary.relative_path != f"shards/part-{shard_id:06d}" or
                recovered.resume_point.shard_id != shard_id or
                not recovered.declarations):
            raise ValueError(f"recovered shard identity differs: {shard_id}")
        first = recovered.declarations[0]
        if last_point is not None:
            if first.declaration_index < last_point.declaration_index:
                raise ValueError(f"recovered declaration order reversed: {shard_id}")
            if first.declaration_index == last_point.declaration_index:
                if (first.first_record_ordinal != last_point.record_ordinal + 1 or
                        first.first_source_line is None or
                        last_point.source_line is None or
                        first.first_source_line <= last_point.source_line):
                    raise ValueError(f"recovered source record gap: {shard_id}")
            elif first.first_record_ordinal != 0:
                raise ValueError(f"recovered declaration starts after ordinal zero: {shard_id}")
        index_maps = catalog._merge_validated_fragment(recovered.catalog_fragment)
        if (index_maps.source_indices != recovered.source_global_indices or
                index_maps.normalized_indices != recovered.normalized_global_indices):
            raise ValueError(f"recovered field indexes differ: {shard_id}")
        for item in recovered.declarations:
            _accept_recovered_declaration(declarations, rows, item, shard_id)
        last_point = recovered.resume_point
        shard_summaries.append(recovered.shard_summary)
        progress.declarations = last_point.declaration_index
        progress.records += recovered.shard_summary.records
        progress.projections += recovered.shard_summary.projections
        progress.source_leaves += recovered.shard_summary.source_leaves
        progress.source_file = rows[last_point.declaration_index]["source_file"]
        progress.record_ordinal = last_point.record_ordinal
        progress.source_line = last_point.source_line
        progress.report()

    if workers == 1:
        for shard_id in shard_ids:
            recovered = (tail_shard if shard_id == tail_shard.resume_point.shard_id
                         else recover_structure_shard(STAGED_VALUES, shard_id))
            consume(shard_id, recovered)
    else:
        pending: deque[tuple[int, Future[RecoveredStructureShard]]] = deque()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for shard_id in shard_ids:
                if len(pending) >= 2 * workers:
                    completed_id, future = pending.popleft()
                    try:
                        consume(completed_id, future.result())
                    except Exception as exc:
                        raise RuntimeError(f"structure shard recovery failed: {completed_id}") from exc
                if shard_id == tail_shard.resume_point.shard_id:
                    cached: Future[RecoveredStructureShard] = Future()
                    cached.set_result(tail_shard)
                    pending.append((shard_id, cached))
                else:
                    pending.append((shard_id, executor.submit(
                        recover_structure_shard, STAGED_VALUES, shard_id)))
            while pending:
                shard_id, future = pending.popleft()
                try:
                    consume(shard_id, future.result())
                except Exception as exc:
                    raise RuntimeError(f"structure shard recovery failed: {shard_id}") from exc
    if last_point is None:
        raise ValueError("structure recovery has no persisted record")
    for row in rows:
        row["effective_families"].sort()
    shards = tuple(shard_summaries)
    summary = StorageSummary(
        shards=shards,
        records=sum(shard.records for shard in shards),
        projections=sum(shard.projections for shard in shards),
        source_leaves=sum(shard.source_leaves for shard in shards),
        excluded_integrity_subtrees=sum(
            shard.excluded_integrity_subtrees for shard in shards),
        oversized_records=sum(shard.oversized_records for shard in shards),
    )
    return summary, last_point


def _source_record_at_resume(
        index: int, declaration: Any, point: StorageResumePoint) -> SourceRecord:
    # iter_source_records already opens the arm_semantic_history gzip root and
    # yields its single record at ordinal 0 with no source line.
    if _is_json_root(declaration):
        record = next(iter_source_records(index, declaration))
    elif declaration.path.suffix == ".jsonl":
        if point.source_line is None:
            raise ValueError(f"stored JSONL source line is absent: {declaration.path}")
        with open_artifact_binary(declaration.path) as stream:
            for line_no, line in enumerate(stream, 1):
                if line_no == point.source_line:
                    if not line.strip():
                        raise ValueError(f"stored source line is blank: {declaration.path}:{line_no}")
                    raw_root = orjson.loads(line)
                    break
            else:
                raise ValueError(f"stored source line is absent: {declaration.path}:{point.source_line}")
        predicate_kind = None
        if declaration.source_family == "arm_predicate_truth_ticks":
            if type(raw_root) is not dict:
                raise TypeError(f"source record is not an object: {declaration.path}:{point.source_line}")
            predicate_kind = ("arm_predicate_truth_ticks" if "predicate_id" in raw_root
                              else "arm_domain_state_ticks")
        record = _source_record(
            index, declaration, raw_root, source_line=point.source_line,
            record_ordinal=point.record_ordinal,
            source_scope=inventory_scope(
                declaration.source_family, declaration.path, declaration.sample),
            predicate_tick_kind=predicate_kind,
        )
    else:
        raise ValueError(f"stored source has no structural reader: {declaration.path}")
    if ((record.declaration_index, record.record_ordinal, record.source_line,
         record.source_file) !=
        (index, point.record_ordinal, point.source_line, source_path(declaration.path))):
        raise ValueError(f"stored source identity differs: {declaration.path}")
    return record


def _without_indexes(row: dict[str, Any], indexes: frozenset[str]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in indexes}


def _check_resume_source(
        root: Path, declaration: Any, row: dict[str, Any],
        point: StorageResumePoint) -> tuple[SourceRecord, _Anchor]:
    record = _source_record_at_resume(point.declaration_index, declaration, point)
    if record.source_file != row["source_file"]:
        raise ValueError(f"resumed declaration source file differs: {record.source_file}")
    actual = inspect_structure_record(
        root, point.declaration_index, point.record_ordinal,
        source_line=point.source_line, shard_id=point.shard_id)
    expected = encode_structure_batch((record,))
    expected_record = _decoded_row(expected.record_table.to_pylist()[0])
    if not _same_typed(
            _without_indexes(expected_record, frozenset({
                "record_index", "projection_start", "source_leaf_start",
                "oversized_record"})),
            _without_indexes(actual["record"], frozenset({
                "record_index", "projection_start", "source_leaf_start",
                "oversized_record"}))):
        raise ValueError(f"resumed source record differs from stored values: {record.source_file}")
    expected_projections = expected.projection_table.to_pylist()
    expected_leaves = expected.source_leaf_table.to_pylist()
    if len(expected_projections) != len(actual["projections"]):
        raise ValueError(f"resumed source projection count differs: {record.source_file}")
    for raw_projection, stored in zip(expected_projections, actual["projections"]):
        expected_projection = _decoded_row(raw_projection)
        if not _same_typed(
                _without_indexes(expected_projection, frozenset({
                    "projection_index", "record_index", "normalized_index",
                    "source_leaf_start"})),
                _without_indexes(stored["projection"], frozenset({
                    "projection_index", "record_index", "normalized_index",
                    "source_leaf_start"}))):
            raise ValueError(f"resumed source projection differs: {record.source_file}")
        first = raw_projection["source_leaf_start"]
        leaves = expected_leaves[first:first + raw_projection["source_leaf_count"]]
        if len(leaves) != len(stored["source_leaves"]):
            raise ValueError(f"resumed source leaf count differs: {record.source_file}")
        for raw_leaf, stored_leaf in zip(leaves, stored["source_leaves"]):
            if not _same_typed(
                    _without_indexes(_decoded_row(raw_leaf), frozenset({
                        "source_leaf_index", "projection_index", "source_index"})),
                    _without_indexes(stored_leaf, frozenset({
                        "source_leaf_index", "projection_index", "source_index"}))):
                raise ValueError(f"resumed source leaf differs: {record.source_file}")
    stored_record = actual["record"]
    anchor = _Anchor(
        point.declaration_index, point.record_ordinal, point.source_line,
        point.shard_id, stored_record["projection_count"],
        stored_record["source_leaf_count"],
    )
    return record, anchor


def _build_into(staging_values: Path, staging_fields: Path,
                staging_inventory: Path, *, workers: int) -> dict[str, Any]:
    """Materialize a fresh delivery into a staging root and verify it there.

    This never reads a published or previously staged shard. It assumes the
    three staging paths are absent and leaves them in place for the caller to
    publish.
    """
    started = time.monotonic()
    progress = _Progress(started, started)
    declarations = ordered_source_declarations()
    rows: list[dict[str, Any]] = []
    catalog = StructureCatalog()
    anchors: dict[str, _Anchor] = {}
    writer = StructureStorageWriter(staging_values)
    with ProcessPoolExecutor(max_workers=workers) if workers > 1 else _NoExecutor() as executor:
        for index, declaration in declarations:
            row = _declaration_row(index, declaration)
            progress.source_file = row["source_file"]
            progress.record_ordinal = None
            progress.source_line = None
            _scan_declaration(index, declaration, row, catalog, writer,
                              executor, workers, anchors, progress)
            rows.append(row)
            progress.declarations += 1
            progress.report()
    storage = writer.finish()
    field_catalog = catalog.finish(
        declarations=tuple(declaration for _, declaration in declarations))
    if catalog.grammar_declaration_statuses != tuple(
            row["record_grammar_status"] for row in rows):
        raise ValueError("catalog grammar declarations differ from source inventory")
    summary = _summary(rows, field_catalog, storage, time.monotonic() - started)
    actual = validate_structure_storage(staging_values)
    if _jsonable(asdict(actual)) != summary["storage"]:
        raise ValueError("written Parquet tables differ from their summary")
    _check_anchors(staging_values, anchors)
    _write_jsonl(staging_fields, _field_rows(field_catalog))
    _write_jsonl(staging_inventory, _inventory_rows(rows, summary))
    _load_delivery(staging_inventory, staging_fields, staging_values,
                   scan_storage=False)
    return summary


def build(*, workers: int) -> dict[str, Any]:
    """Materialize every defined, present declaration in fixed source order."""
    if workers < 1:
        raise ValueError("workers must be positive")
    occupied = [path for path in (*_OUTPUT_PATHS, *_STAGED_PATHS) if path.exists()]
    if occupied:
        raise FileExistsError("structure output or staging already exists: " +
                              ", ".join(map(str, occupied)))
    summary = _build_into(STAGED_VALUES, STAGED_FIELDS, STAGED_INVENTORY,
                          workers=workers)
    for staged, final in ((STAGED_VALUES, VALUES), (STAGED_FIELDS, FIELDS),
                          (STAGED_INVENTORY, INVENTORY)):
        os.replace(staged, final)
    print(_line(summary).rstrip(), flush=True)
    return summary


def _restore_rebuild_backups() -> None:
    for backup, final in zip(_REBUILD_BACKUPS, _OUTPUT_PATHS):
        if not backup.exists():
            continue
        if final.is_dir():
            shutil.rmtree(final)
        elif final.exists():
            final.unlink()
        os.replace(backup, final)


def rebuild(*, workers: int) -> dict[str, Any]:
    """Discard the published delivery and rebuild it from the current sources.

    Old shards are never read as input. A fresh staging is built and verified
    while the published artifacts stay in place; only then are the published
    paths moved to same-filesystem ``.rebuild_backup`` siblings and the staged
    paths published. A failed build leaves the published delivery untouched; a
    failed cutover restores the backups. Backups are removed on success.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    occupied = [path for path in (*_STAGED_PATHS, *_REBUILD_BACKUPS) if path.exists()]
    if occupied:
        raise FileExistsError("structure staging or rebuild backup already exists: " +
                              ", ".join(map(str, occupied)))
    if not any(path.exists() for path in _OUTPUT_PATHS):
        raise FileNotFoundError("structure delivery has no published artifact to rebuild")
    try:
        summary = _build_into(STAGED_VALUES, STAGED_FIELDS, STAGED_INVENTORY,
                              workers=workers)
    except BaseException:
        for path in _STAGED_PATHS:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        raise
    try:
        for final, backup in zip(_OUTPUT_PATHS, _REBUILD_BACKUPS):
            if final.exists():
                os.replace(final, backup)
        for staged, final in ((STAGED_VALUES, VALUES), (STAGED_FIELDS, FIELDS),
                              (STAGED_INVENTORY, INVENTORY)):
            os.replace(staged, final)
    except BaseException:
        _restore_rebuild_backups()
        raise
    for backup in _REBUILD_BACKUPS:
        if backup.is_dir():
            shutil.rmtree(backup)
        elif backup.exists():
            backup.unlink()
    print(_line(summary).rstrip(), flush=True)
    return summary


def _complete_interrupted_publication() -> dict[str, Any]:
    if (not VALUES.is_dir() or STAGED_VALUES.exists() or INVENTORY.exists() or
            not STAGED_INVENTORY.is_file() or
            (FIELDS.exists() and STAGED_FIELDS.exists())):
        raise ValueError("structure publication has an inconsistent partial state")
    fields_path = FIELDS if FIELDS.is_file() else STAGED_FIELDS
    if not fields_path.is_file():
        raise ValueError("structure publication has no completed field index")
    summary, _, _ = _load_delivery(
        STAGED_INVENTORY, fields_path, VALUES, scan_storage=True)
    if not FIELDS.exists():
        os.replace(STAGED_FIELDS, FIELDS)
    os.replace(STAGED_INVENTORY, INVENTORY)
    print(_line(summary).rstrip(), flush=True)
    return summary


def _resume_validated_prefix(
        declarations: tuple[tuple[int, Any], ...], rows: list[dict[str, Any]],
        catalog: StructureCatalog, storage: StorageSummary,
        point: StorageResumePoint, *, workers: int, started: float,
        ) -> dict[str, Any]:
    """Continue from a validated staged prefix and publish its final delivery."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if len(rows) != len(declarations) or not 0 <= point.declaration_index < len(declarations):
        raise ValueError("stored tail declaration index is absent")
    declaration = declarations[point.declaration_index][1]
    source_record, tail_anchor = _check_resume_source(
        STAGED_VALUES, declaration, rows[point.declaration_index], point)
    writer, writer_point = StructureStorageWriter.resume_from_validated(
        STAGED_VALUES, storage)
    if writer_point != point:
        raise ValueError("recovered source cursor differs from storage writer")
    for index, row in enumerate(rows):
        row["effective_families"].sort()
        if index < point.declaration_index and row["projections"]:
            row["delivery_status"] = "projected"
        if row["records"] and row["record_grammar_status"] == "closed_record_grammar":
            audit = _validate_record_grammar_source(
                index, declarations[index][1], row,
                complete=index < point.declaration_index)
            row["grammar_validated_records"] = audit["records"]
            row["grammar_validated_effective_records"] = audit["effective_records"]
    anchors = {declaration.source_family: tail_anchor}
    resume_after = SourceResumeCursor(
        point.record_ordinal, point.source_line, source_record.effective_family)
    progress = _Progress(
        started, time.monotonic(), declarations=point.declaration_index,
        records=storage.records, projections=storage.projections,
        source_leaves=storage.source_leaves,
    )

    with ProcessPoolExecutor(max_workers=workers) if workers > 1 else _NoExecutor() as executor:
        for index, declaration in declarations[point.declaration_index:]:
            row = rows[index]
            progress.source_file = row["source_file"]
            progress.record_ordinal = point.record_ordinal if index == point.declaration_index else None
            progress.source_line = point.source_line if index == point.declaration_index else None
            _scan_declaration(
                index, declaration, row, catalog, writer, executor, workers,
                anchors, progress,
                resume_after=resume_after if index == point.declaration_index else None)
            progress.declarations += 1
            progress.report()

    storage = writer.finish()
    field_catalog = catalog.finish(
        declarations=tuple(declaration for _, declaration in declarations))
    if catalog.grammar_declaration_statuses != tuple(
            row["record_grammar_status"] for row in rows):
        raise ValueError("catalog grammar declarations differ from source inventory")
    summary = _summary(rows, field_catalog, storage, time.monotonic() - started)
    actual = validate_structure_storage(STAGED_VALUES)
    if _jsonable(asdict(actual)) != summary["storage"]:
        raise ValueError("resumed Parquet tables differ from their summary")
    _check_anchors(STAGED_VALUES, anchors)
    _write_jsonl(STAGED_FIELDS, _field_rows(field_catalog), overwrite=True)
    _write_jsonl(STAGED_INVENTORY, _inventory_rows(rows, summary), overwrite=True)
    _load_delivery(STAGED_INVENTORY, STAGED_FIELDS, STAGED_VALUES,
                   scan_storage=False)
    if any(path.exists() for path in _OUTPUT_PATHS):
        raise FileExistsError("structure final output appeared during recovery")
    for staged, final in ((STAGED_VALUES, VALUES), (STAGED_FIELDS, FIELDS),
                          (STAGED_INVENTORY, INVENTORY)):
        os.replace(staged, final)
    print(_line(summary).rstrip(), flush=True)
    return summary


def resume(*, workers: int) -> dict[str, Any]:
    """Continue a staged structural delivery from complete persisted shards."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if INVENTORY.exists():
        raise FileExistsError(f"structure inventory is already published: {INVENTORY}")
    if VALUES.exists():
        return _complete_interrupted_publication()
    if FIELDS.exists():
        raise FileExistsError(f"structure fields exist without published values: {FIELDS}")
    if not STAGED_VALUES.is_dir():
        raise FileNotFoundError(STAGED_VALUES)
    for path in (STAGED_FIELDS, STAGED_INVENTORY):
        if path.exists() and not path.is_file():
            raise ValueError(f"structure staged index is not a file: {path}")
    directories = _shard_dirs(STAGED_VALUES)
    if not directories:
        raise ValueError("structure staging has no complete shard")
    if any(path.name != f"part-{shard_id:06d}" for shard_id, path in directories):
        raise ValueError("structure staged shard names differ from writer output")

    started = time.monotonic()
    progress = _Progress(started, started)
    declarations = ordered_source_declarations()
    rows = [_declaration_row(index, declaration) for index, declaration in declarations]
    tail_shard = recover_structure_shard(STAGED_VALUES, directories[-1][0])
    point = tail_shard.resume_point
    if not 0 <= point.declaration_index < len(declarations):
        raise ValueError("stored tail declaration index is absent")
    declaration = declarations[point.declaration_index][1]
    _check_resume_source(STAGED_VALUES, declaration, rows[point.declaration_index], point)
    catalog = StructureCatalog()
    recovered_storage, recovered_point = _recover_prefix(
        declarations, rows, catalog, tuple(shard_id for shard_id, _ in directories),
        tail_shard, progress, workers)
    if recovered_point != point:
        raise ValueError("recovered source cursor differs from storage writer")
    return _resume_validated_prefix(
        declarations, rows, catalog, recovered_storage, point,
        workers=workers, started=started)


class _NoExecutor:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: Any) -> None:
        return None


def inspect(*, status: str | None = None,
            structure_status: str | None = None) -> None:
    summary, declarations, catalog = _load_delivery(
        INVENTORY, FIELDS, VALUES,
        scan_storage=status is None and structure_status is None,
        published_baseline=not ACCEPTANCE.exists())
    if status is not None:
        for row in declarations:
            if row["delivery_status"] == status:
                print(_line(row).rstrip())
    elif structure_status is not None:
        for row in catalog.normalized_fields:
            if row["structure_status"] == structure_status:
                print(_line(row).rstrip())
    else:
        print(_line({key: value for key, value in summary.items()
                     if key != "storage"} | {
                         "shards": len(summary["storage"]["shards"]),
                         "oversized_records": summary["storage"]["oversized_records"],
                     }).rstrip())


def _find_shard(row: dict[str, Any], record_ordinal: int | None,
                source_line: int | None) -> int:
    for span in row["shard_spans"]:
        if (record_ordinal is not None and
                span["first_ordinal"] <= record_ordinal <= span["last_ordinal"] or
                source_line is not None and span["first_line"] is not None and
                span["first_line"] <= source_line <= span["last_line"]):
            return span["shard_id"]
    raise KeyError(f"source record is absent: {row['declaration_index']}:"
                   f"{record_ordinal}:{source_line}")


def read(declaration_index: int, *, record_ordinal: int | None = None,
         source_line: int | None = None, normalized_id: str | None = None,
         source_field_id: str | None = None) -> dict[str, Any]:
    _, declarations, catalog = _load_delivery(
        INVENTORY, FIELDS, VALUES, scan_storage=False,
        published_baseline=not ACCEPTANCE.exists())
    if not 0 <= declaration_index < len(declarations):
        raise IndexError(f"declaration index is absent: {declaration_index}")
    row = declarations[declaration_index]
    normalized_index = None
    source_index = None
    if normalized_id is not None:
        matches = [field["normalized_index"] for field in catalog.normalized_fields
                   if field["id"] == normalized_id]
        if not matches:
            raise KeyError(f"normalized field is absent: {normalized_id}")
        normalized_index = matches[0]
    if source_field_id is not None:
        matches = [field["source_index"] for field in catalog.source_fields
                   if field["id"] == source_field_id]
        if not matches:
            raise KeyError(f"source field is absent: {source_field_id}")
        source_index = matches[0]
    shard_id = _find_shard(row, record_ordinal, source_line)
    value = inspect_structure_record(
        VALUES, declaration_index, record_ordinal, source_line=source_line,
        normalized_index=normalized_index, source_index=source_index,
        shard_id=shard_id)
    value["source_file"] = row["source_file"]
    value["source_scope"] = row["source_scope"]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="materialize all readable declarations")
    build_parser.add_argument("--workers", type=int, default=16)
    rebuild_parser = commands.add_parser(
        "rebuild", help="discard published artifacts and rebuild from current sources")
    rebuild_parser.add_argument("--workers", type=int, default=16)
    resume_parser = commands.add_parser("resume", help="continue completed staged shards")
    resume_parser.add_argument("--workers", type=int, default=16)
    inspect_parser = commands.add_parser("inspect", help="validate and summarize final delivery")
    filters = inspect_parser.add_mutually_exclusive_group()
    filters.add_argument("--status", choices=sorted(_SOURCE_STATUSES))
    filters.add_argument("--structure-status")
    read_parser = commands.add_parser("read", help="read one published physical record")
    read_parser.add_argument("--declaration-index", type=int, required=True)
    locator = read_parser.add_mutually_exclusive_group(required=True)
    locator.add_argument("--record-ordinal", type=int)
    locator.add_argument("--source-line", type=int)
    read_parser.add_argument("--normalized-id")
    read_parser.add_argument("--source-field-id")
    args = parser.parse_args(argv)
    if args.command == "build":
        build(workers=args.workers)
    elif args.command == "rebuild":
        rebuild(workers=args.workers)
    elif args.command == "resume":
        resume(workers=args.workers)
    elif args.command == "inspect":
        inspect(status=args.status, structure_status=args.structure_status)
    else:
        print(_line(read(
            args.declaration_index, record_ordinal=args.record_ordinal,
            source_line=args.source_line, normalized_id=args.normalized_id,
            source_field_id=args.source_field_id)).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
