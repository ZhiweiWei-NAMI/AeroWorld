"""Read declared structure sources one physical record at a time.

This module keeps source locations and observed leaves independent of field
meaning, units, graph ownership, and the semantic field catalog. Missing and
schema-pending declarations remain in the inventory; callers must not pass
them to the value reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import gzip
import json
import orjson

from Dataset.world_model.graph.contract import read_json, source_path
from Dataset.world_model.graph.controller_scope import controller_namespaces_for_record
from Dataset.world_model.graph.entity_identity import read_roster
from Dataset.world_model.graph.field_catalog import (
    SourceDeclaration, _direct_source_leaves, declared_source_inventory,
)
from Dataset.world_model.graph.field_paths import source_value_at_tokens
from Dataset.world_model.graph.sample_identity import arm_chain_row_family, inventory_scope
from Dataset.world_model.graph.source_projection import ProjectedField, project_values
from Dataset.world_model.graph.structure_closure import (
    GRAMMAR_SOURCE_FAMILIES, PROFILE_GRAMMAR_SOURCE_FAMILIES,
    validate_source_record,
)
from Dataset.world_model.graph.structure_schema import is_publication_excluded_path


_ENTITY_FILES = frozenset({"formal_roster", "arm_branch_roster", "arm_scene_setup"})
_CHAIN_KINDS = frozenset({"arm_chain_nodes", "arm_chain_events"})
EXTENSION_TAIL_FAMILIES = frozenset({
    "arm_initial_events", "arm_semantic_history", "formal_episode_manifest",
    "formal_objective_manifest", "formal_source_scene_setup",
    "formal_source_script", "script_category_source",
})


@dataclass(frozen=True)
class SourceRecord:
    """One complete JSON root or nonblank JSONL row from one declaration."""

    declaration_index: int
    source_family: str
    effective_family: str
    path: Path
    source_file: str
    source_line: int | None
    record_ordinal: int
    raw_root: Any
    source_scope: Mapping[str, Any] | None
    record_tick_present: bool
    record_tick: Any
    excluded_integrity_subtrees: int


@dataclass(frozen=True)
class SourceResumeCursor:
    """Last persisted record of a declaration."""

    record_ordinal: int
    source_line: int | None
    effective_family: str


def ordered_source_declarations() -> tuple[tuple[int, SourceDeclaration], ...]:
    """Give each declared source role a repeatable inventory position."""
    declarations = sorted(
        declared_source_inventory(),
        key=lambda item: (
            item.source_family in EXTENSION_TAIL_FAMILIES,
            item.source_family, source_path(item.path), item.sample,
            item.read_relation, item.read_condition or "", item.reference_samples,
        ),
    )
    keys = [(
        item.source_family, source_path(item.path), item.sample,
        item.read_relation, item.read_condition, item.reference_samples,
    ) for item in declarations]
    if len(keys) != len(set(keys)):
        raise ValueError("source inventory has duplicate declaration roles")
    return tuple(enumerate(declarations))


def _excluded_integrity_subtrees(raw_root: Any) -> int:
    """Count subtrees under the exact publication-excluded paths, without
    writing their names or values. This is the same matcher the value walk
    uses, over complete original source token paths."""
    count = 0
    pending = [((), raw_root)]
    while pending:
        tokens, item = pending.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"source object key is not a string: {key!r}")
                if is_publication_excluded_path(tokens + (key,)):
                    count += 1
                else:
                    pending.append((tokens + (key,), child))
        elif isinstance(item, list):
            pending.extend((tokens + (index,), child)
                           for index, child in enumerate(item))
    return count


def _effective_family(source_family: str, raw_root: Any,
                      predicate_tick_kind: str | None) -> str:
    if source_family in _CHAIN_KINDS:
        return arm_chain_row_family(source_family, raw_root)
    if source_family == "arm_predicate_truth_ticks":
        if predicate_tick_kind is None:
            raise ValueError("predicate/state source row has no resolved family")
        return predicate_tick_kind
    return source_family


def _source_record(declaration_index: int, declaration: SourceDeclaration,
                   raw_root: Any, *, source_line: int | None,
                   record_ordinal: int, source_scope: Mapping[str, Any] | None,
                   predicate_tick_kind: str | None) -> SourceRecord:
    family = declaration.source_family
    source_file = source_path(declaration.path)
    location = (f"{source_file}:record_ordinal={record_ordinal},"
                f"source_line={source_line}")
    if not isinstance(raw_root, Mapping):
        raise TypeError(f"source record is not an object: {location}")
    try:
        effective_family = _effective_family(family, raw_root, predicate_tick_kind)
        if family in GRAMMAR_SOURCE_FAMILIES | PROFILE_GRAMMAR_SOURCE_FAMILIES:
            validate_source_record(
                family, raw_root, source_file,
                profile=source_scope["profile"] if source_scope is not None else None,
                effective_family=effective_family,
            )
    except Exception as exc:
        raise ValueError(f"{location}: {exc}") from exc
    return SourceRecord(
        declaration_index=declaration_index,
        source_family=family,
        effective_family=effective_family,
        path=declaration.path,
        source_file=source_file,
        source_line=source_line,
        record_ordinal=record_ordinal,
        raw_root=raw_root,
        source_scope=source_scope,
        record_tick_present="tick" in raw_root,
        record_tick=raw_root.get("tick"),
        excluded_integrity_subtrees=_excluded_integrity_subtrees(raw_root),
    )


def iter_source_records(declaration_index: int, declaration: SourceDeclaration, *,
                        resume_after: SourceResumeCursor | None = None
                        ) -> Iterator[SourceRecord]:
    """Stream records after the last persisted ordinal and physical source line."""
    if declaration.structure_status != "field_scan_defined":
        raise ValueError(f"source structure is pending: {source_path(declaration.path)}")
    if declaration.presence != "present":
        raise FileNotFoundError(declaration.path)
    json_root = (declaration.path.suffix == ".json" or
                 (declaration.source_family == "arm_semantic_history" and
                  declaration.path.suffixes[-2:] == [".json", ".gz"]))
    if not json_root and declaration.path.suffix != ".jsonl":
        raise ValueError(f"source has no executable structure reader: {declaration.path}")
    if resume_after is not None:
        if not isinstance(resume_after, SourceResumeCursor):
            raise TypeError(f"invalid source resume cursor: {resume_after!r}")
        if type(resume_after.record_ordinal) is not int or resume_after.record_ordinal < 0:
            raise ValueError(f"invalid resumed record ordinal: {resume_after.record_ordinal!r}")
        if not isinstance(resume_after.effective_family, str) or not resume_after.effective_family:
            raise ValueError(f"invalid resumed record family: {resume_after.effective_family!r}")
        family = declaration.source_family
        if family in _CHAIN_KINDS:
            valid_family = (resume_after.effective_family.startswith(family + ":") and
                            len(resume_after.effective_family) > len(family) + 1)
        elif family == "arm_predicate_truth_ticks":
            valid_family = resume_after.effective_family in {
                "arm_predicate_truth_ticks", "arm_domain_state_ticks"}
        else:
            valid_family = resume_after.effective_family == family
        if not valid_family:
            raise ValueError(f"resumed record family differs: {declaration.path}")
        if json_root:
            if resume_after.record_ordinal != 0 or resume_after.source_line is not None:
                raise ValueError(f"invalid JSON root resume cursor: {declaration.path}")
        elif (type(resume_after.source_line) is not int or
              resume_after.source_line < 1 or
              resume_after.record_ordinal >= resume_after.source_line):
            raise ValueError(f"invalid JSONL resume cursor: {declaration.path}")

    source_scope = inventory_scope(
        declaration.source_family, declaration.path, declaration.sample,
    )
    if json_root:
        if resume_after is not None:
            return
        if declaration.path.suffix == ".gz":
            with gzip.open(declaration.path, "rt", encoding="utf-8") as stream:
                raw_root = json.load(stream)
        else:
            raw_root = (read_roster(declaration.path)
                        if declaration.source_family in _ENTITY_FILES else
                        read_json(declaration.path))
        yield _source_record(
            declaration_index, declaration, raw_root, source_line=None,
            record_ordinal=0, source_scope=source_scope,
            predicate_tick_kind=None,
        )
        return

    predicate_tick_kind = (resume_after.effective_family
                           if resume_after is not None and
                           declaration.source_family == "arm_predicate_truth_ticks" else None)
    ordinal = resume_after.record_ordinal + 1 if resume_after is not None else 0
    skipped_records = 0
    boundary_found = False
    with declaration.path.open("rb") as stream:
        for line_no, line in enumerate(stream, 1):
            if resume_after is not None and line_no <= resume_after.source_line:
                if line.strip():
                    skipped_records += 1
                if line_no == resume_after.source_line:
                    if not line.strip() or skipped_records != ordinal:
                        raise ValueError(f"source resume boundary differs: {declaration.path}:{line_no}")
                    boundary_found = True
                continue
            if not line.strip():
                continue
            try:
                raw_root = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(
                    f"{source_path(declaration.path)}:record_ordinal={ordinal},"
                    f"source_line={line_no}: invalid JSONL record") from exc
            if declaration.source_family == "arm_predicate_truth_ticks":
                if not isinstance(raw_root, Mapping):
                    raise TypeError(
                        f"predicate/state row is not an object: "
                        f"{source_path(declaration.path)}:record_ordinal={ordinal},"
                        f"source_line={line_no}")
                row_kind = ("arm_predicate_truth_ticks" if "predicate_id" in raw_root else
                            "arm_domain_state_ticks")
                if predicate_tick_kind is None:
                    predicate_tick_kind = row_kind
                elif predicate_tick_kind != row_kind:
                    raise ValueError(
                        f"mixed predicate/state rows in {source_path(declaration.path)}:"
                        f"record_ordinal={ordinal},source_line={line_no}")
            yield _source_record(
                declaration_index, declaration, raw_root, source_line=line_no,
                record_ordinal=ordinal, source_scope=source_scope,
                predicate_tick_kind=predicate_tick_kind,
            )
            ordinal += 1
    if resume_after is not None and not boundary_found:
        raise ValueError(f"source resume line is absent: {declaration.path}:{resume_after.source_line}")


def _project_full_or_entity_root(record: SourceRecord) -> tuple[ProjectedField, ...]:
    raw_root = record.raw_root
    family = record.effective_family
    arguments = {
        "source_family": family,
        "scope": record.source_scope,
        "source_file": record.source_file,
        "source_line": record.source_line,
    }
    if family not in _ENTITY_FILES:
        namespaces = (controller_namespaces_for_record(record.path, raw_root)
                      if family == "domain_state" else None)
        return project_values(raw_root, entity_key_namespaces=namespaces, **arguments)

    entities = raw_root["entities"]
    if not isinstance(entities, list):
        raise TypeError(f"entity source has no entity list: {record.source_file}")
    projected: list[ProjectedField] = []
    for key, child in raw_root.items():
        if is_publication_excluded_path((key,)):
            continue
        if key == "entities" and entities:
            for index, entity in enumerate(entities):
                if not isinstance(entity, Mapping):
                    raise TypeError(f"source entity is not an object: {record.source_file}:entities[{index}]")
                projected.extend(project_values(
                    entity, source_prefix_tokens=("entities", index), **arguments,
                ))
        else:
            projected.extend(project_values(
                {key: child}, **arguments,
            ))
    return tuple(projected)


def project_record(record: SourceRecord) -> tuple[ProjectedField, ...]:
    """Project and independently reconcile every permitted raw leaf in a record."""
    projected = _project_full_or_entity_root(record)
    leaves: dict[tuple[str | int, ...], Any] = {}
    for value in projected:
        for source in value.sources:
            leaf = source.field_value
            tokens = leaf.source_tokens
            if tokens in leaves:
                raise ValueError(f"source leaf was projected more than once: {record.source_file}:{record.source_line}:{tokens}")
            from_root = source_value_at_tokens(record.raw_root, tokens)
            if type(from_root) is not type(leaf.value) or from_root != leaf.value:
                raise ValueError(f"projected source value differs from raw record: {record.source_file}:{record.source_line}:{tokens}")
            leaves[tokens] = leaf
    for tokens, raw_path, raw_value in _direct_source_leaves(record.raw_root, ""):
        leaf = leaves.pop(tokens, None)
        if leaf is None:
            raise ValueError(f"raw source leaf was not projected: {record.source_file}:{record.source_line}:{tokens}")
        if (leaf.raw_path != raw_path or type(leaf.value) is not type(raw_value)
                or leaf.value != raw_value):
            raise ValueError(f"projected leaf differs from raw source: {record.source_file}:{record.source_line}:{tokens}")
    if leaves:
        raise ValueError(f"projected source has no raw leaf: {record.source_file}:{record.source_line}:{next(iter(leaves))}")
    return projected
