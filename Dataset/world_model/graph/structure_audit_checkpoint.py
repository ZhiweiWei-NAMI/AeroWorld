"""Persistent per-source checkpoints for the existing-source grammar audit.

The G5 audit re-reads every reviewed published source on each stage run.  This
module keeps each source's completed audit result so a later run re-audits only
the sources whose declaration role, published position and record count, source
file bytes, or applicable record-grammar schema actually changed.  A source that
failed or was interrupted is never reused: only a completed result whose whole
identity still matches is replayed.

Grammar schemas are versioned by their complete text, never by a summary.  One schema_ref can
resolve to several complete schema texts over time -- the ref is a builder call
whose arguments stay put while the builder body changes -- so each distinct
(schema_ref, schema_json) pair keeps its own schema_id.  An audit result points
at the exact version it saw: after a builder edit the old sources still hold the
old text and are re-audited, while freshly audited sources hold the new text and
are reused.  Nothing here reduces a schema to a fixed-length value, and acceptance is
never summary-based.

Storage is one SQLite database in a run directory outside the published Parquet
root (Dataset/world_model/runs is gitignored).  Only the main process touches it
and every success is committed on its own transaction, so an interrupt keeps all
sources already audited.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Callable, Sequence

CHECKPOINT_DIR_NAME = "structure_audit_checkpoint"
DATABASE_NAME = "grammar_audit.sqlite3"

# Full source identity of the file at audit completion.  An inode change or a
# different mtime_ns/ctime_ns on identical bytes still forces a re-audit, which
# is the conservative side of a stale reuse.
_STAT_NAMES = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")

# A declaration role is allowed to read one file more than once, so the record
# key is the pair (declaration_index, source_file) rather than the path alone:
# two roles of one path must never overwrite each other's result.  A declaration
# index is only reused for a different path after a rebuild, which the key also
# excludes.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS audit_source (
    declaration_index INTEGER NOT NULL,
    source_file TEXT NOT NULL,
    source_family TEXT NOT NULL,
    role_json TEXT NOT NULL,
    declared_sample TEXT NOT NULL,
    structure_status TEXT NOT NULL,
    presence TEXT NOT NULL,
    records INTEGER NOT NULL,
    input_bytes INTEGER NOT NULL,
    row_effective_json TEXT NOT NULL,
    declared_effective_json TEXT NOT NULL,
    shard_spans_json TEXT NOT NULL,
    stat_dev INTEGER NOT NULL,
    stat_ino INTEGER NOT NULL,
    stat_size INTEGER NOT NULL,
    stat_mtime_ns INTEGER NOT NULL,
    stat_ctime_ns INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (declaration_index, source_file)
);
CREATE TABLE IF NOT EXISTS grammar_schema (
    schema_id INTEGER PRIMARY KEY,
    schema_ref TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    UNIQUE (schema_ref, schema_json)
);
CREATE TABLE IF NOT EXISTS audit_schema (
    declaration_index INTEGER NOT NULL,
    source_file TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    effective_family TEXT NOT NULL,
    schema_id INTEGER NOT NULL,
    PRIMARY KEY (declaration_index, source_file, ordinal)
);
"""


def clear_default(runs_root: Path) -> None:
    """Drop a checkpoint without opening one; used after a successful publish."""
    shutil.rmtree(runs_root / CHECKPOINT_DIR_NAME, ignore_errors=True)


def stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    """The five stat fields recorded when a source audit completed."""
    info = os.stat(path)
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns)


@dataclass(frozen=True)
class SourceIdentity:
    """Everything a reused audit result has to agree with, minus the schemas."""

    source_file: str
    declaration_index: int
    source_family: str
    role: tuple[Any, ...]
    declared_sample: str
    structure_status: str
    presence: str
    records: int
    input_bytes: int
    row_effective: tuple[str, ...]
    declared_effective: tuple[str, ...]
    shard_spans: tuple[Any, ...]
    stat: tuple[int, int, int, int, int]


def _json_normal(value: Any) -> Any:
    """Tuples and lists compare equal after a JSON round trip."""
    if isinstance(value, (list, tuple)):
        return [_json_normal(item) for item in value]
    return value


def _decode(value: str, name: str) -> Any:
    try:
        return json.loads(value)
    except ValueError as exc:  # corrupt checkpoint must not pass for a reuse
        raise ValueError(f"audit checkpoint column is not valid JSON: {name}") from exc


def _tuple_of_strings(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or any(type(v) is not str for v in values):
        raise ValueError(f"audit checkpoint {name} is not a string sequence")
    return tuple(values)


class AuditCheckpoint:
    """Per-source grammar-audit results, written only by the main process."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / DATABASE_NAME
        directory.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode=WAL")
        # FULL so an interrupted run survives the loss of the process as well as
        # the machine; each source commit is one fsynced transaction.
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(_SCHEMA_SQL)

    @classmethod
    def open_default(cls, runs_root: Path) -> "AuditCheckpoint":
        """Open the checkpoint beside, never inside, the published Parquet root."""
        return cls(runs_root / CHECKPOINT_DIR_NAME)

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()

    def __enter__(self) -> "AuditCheckpoint":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def clear(self) -> None:
        """Drop every checkpoint file; called only after a successful publish."""
        self.close()
        shutil.rmtree(self.directory, ignore_errors=True)

    def _schema_id(self, schema_ref: str, schema_json: str) -> int:
        """Resolve one content version of a schema_ref, adding it when new.

        A builder edit changes the text under a stable ref, so the same ref holds
        several complete versions.  The version is matched on the full text, not
        on a summary of the text, and never overwrites an older version: sources still
        pointing at the older text keep it for their own reuse test.
        """
        connection = self._connection
        row = connection.execute(
            "SELECT schema_id FROM grammar_schema WHERE schema_ref = ? AND schema_json = ?",
            (schema_ref, schema_json)).fetchone()
        if row is not None:
            return int(row[0])
        cursor = connection.execute(
            "INSERT INTO grammar_schema (schema_ref, schema_json) VALUES (?,?)",
            (schema_ref, schema_json))
        return int(cursor.lastrowid)

    def reuse(self, identity: SourceIdentity,
              schemas: Callable[[], Sequence[tuple[str, str, str]]],
              ) -> dict[str, Any] | None:
        """Return the stored result when its whole identity still holds.

        Cheap declaration and file checks run first so a changed role or a
        rewritten source never pays for grammar compilation.  The applicable
        record-grammar schemas are compared last, each against the exact stored
        version it was audited with, so a schema change invalidates only the
        sources whose results predate it.
        """
        connection = self._connection
        row = connection.execute(
            "SELECT source_family, role_json, declared_sample, structure_status,"
            " presence, records, input_bytes, row_effective_json,"
            " declared_effective_json, shard_spans_json, stat_dev, stat_ino,"
            " stat_size, stat_mtime_ns, stat_ctime_ns, result_json"
            " FROM audit_source WHERE declaration_index = ? AND source_file = ?",
            (identity.declaration_index, identity.source_file)).fetchone()
        if row is None:
            return None
        (source_family, role_json, declared_sample, structure_status, presence,
         records, input_bytes, row_effective_json, declared_effective_json,
         shard_spans_json, stat_dev, stat_ino, stat_size, stat_mtime_ns,
         stat_ctime_ns, result_json) = row
        if (source_family != identity.source_family or
                _decode(role_json, "role_json") != _json_normal(identity.role) or
                declared_sample != identity.declared_sample or
                structure_status != identity.structure_status or
                presence != identity.presence or
                records != identity.records or
                input_bytes != identity.input_bytes or
                _tuple_of_strings(_decode(row_effective_json, "row_effective_json"),
                                  "row_effective_json") != identity.row_effective or
                _tuple_of_strings(_decode(declared_effective_json, "declared_effective_json"),
                                  "declared_effective_json") != identity.declared_effective or
                _decode(shard_spans_json, "shard_spans_json")
                != _json_normal(identity.shard_spans) or
                (stat_dev, stat_ino, stat_size, stat_mtime_ns, stat_ctime_ns)
                != identity.stat):
            return None
        result = _decode(result_json, "result_json")
        if (type(result) is not dict or
                result.get("declaration_index") != identity.declaration_index or
                result.get("source_family") != identity.source_family or
                result.get("source_file") != identity.source_file):
            return None
        expected = tuple(schemas())
        stored = connection.execute(
            "SELECT ordinal, effective_family, schema_id FROM audit_schema"
            " WHERE declaration_index = ? AND source_file = ? ORDER BY ordinal",
            (identity.declaration_index, identity.source_file)).fetchall()
        if not expected or len(stored) != len(expected):
            return None
        for position, (ordinal, effective_family, schema_id) in enumerate(stored):
            want_family, want_ref, want_json = expected[position]
            if ordinal != position or effective_family != want_family:
                return None
            version = connection.execute(
                "SELECT schema_ref, schema_json FROM grammar_schema WHERE schema_id = ?",
                (schema_id,)).fetchone()
            # The referenced version is compared in full.  An audited-with text
            # that no longer equals the current text misses here even though the
            # old version is still stored and still reachable by its schema_id.
            if version is None or version[0] != want_ref or version[1] != want_json:
                return None
        return result

    def commit(self, identity: SourceIdentity, result: dict[str, Any],
               schemas: Sequence[tuple[str, str, str]]) -> None:
        """Record one completed audit in a single transaction."""
        if not schemas:
            raise ValueError("a completed source audit has no applicable grammar")
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT OR REPLACE INTO audit_source (declaration_index, source_file,"
                " source_family, role_json, declared_sample, structure_status, presence,"
                " records, input_bytes, row_effective_json, declared_effective_json,"
                " shard_spans_json, stat_dev, stat_ino, stat_size, stat_mtime_ns,"
                " stat_ctime_ns, result_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (identity.declaration_index, identity.source_file, identity.source_family,
                 json.dumps(_json_normal(identity.role), ensure_ascii=False, allow_nan=False),
                 identity.declared_sample, identity.structure_status, identity.presence,
                 identity.records, identity.input_bytes,
                 json.dumps(_json_normal(identity.row_effective), ensure_ascii=False,
                            allow_nan=False),
                 json.dumps(_json_normal(identity.declared_effective), ensure_ascii=False,
                            allow_nan=False),
                 json.dumps(_json_normal(identity.shard_spans), ensure_ascii=False,
                            allow_nan=False),
                 *identity.stat,
                 json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True)))
            connection.execute(
                "DELETE FROM audit_schema WHERE declaration_index = ? AND source_file = ?",
                (identity.declaration_index, identity.source_file))
            for ordinal, (effective_family, schema_ref, schema_json) in enumerate(schemas):
                connection.execute(
                    "INSERT OR REPLACE INTO audit_schema"
                    " (declaration_index, source_file, ordinal, effective_family, schema_id)"
                    " VALUES (?,?,?,?,?)",
                    (identity.declaration_index, identity.source_file, ordinal,
                     effective_family, self._schema_id(schema_ref, schema_json)))
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise

    def reap_unreferenced(self) -> int:
        """Drop schema versions no audited source points at any more.

        A version becomes unreachable only after every source that used it has
        been re-audited, so removing it cannot invalidate a live result.  Runs
        after a completed audit keeps the stored version count meaningful.
        """
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = connection.execute(
                "DELETE FROM grammar_schema WHERE schema_id NOT IN"
                " (SELECT schema_id FROM audit_schema)")
            connection.execute("COMMIT")
            return int(cursor.rowcount)
        except BaseException:
            connection.execute("ROLLBACK")
            raise

    def counts(self) -> tuple[int, int, int]:
        """Stored sources, stored schema versions, and versions still referenced.

        Versions are (schema_ref, schema_json) pairs, so one ref may hold several
        while a builder change is still being worked through.  The third number
        is what the sources actually point at; the difference is only the
        residue reap_unreferenced() clears.
        """
        connection = self._connection
        sources = connection.execute(
            "SELECT COUNT(*) FROM audit_source").fetchone()[0]
        versions = connection.execute(
            "SELECT COUNT(*) FROM grammar_schema").fetchone()[0]
        referenced = connection.execute(
            "SELECT COUNT(DISTINCT schema_id) FROM audit_schema").fetchone()[0]
        return sources, versions, referenced
