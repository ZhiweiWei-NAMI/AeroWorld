"""Read back one materialized partition and check source traceability."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

from Dataset.world_model.graph.contract import REPO
from Dataset.world_model.graph.records import load_scope
from Dataset.world_model.graph.validator import GraphValidator
from Dataset.world_model.graph.verify_contract import verify_partition


FILES = ("nodes.parquet", "edges.parquet", "scope.parquet")


@lru_cache(maxsize=None)
def _line_count(path: Path) -> int:
    count = 0
    last = b""
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    return count + (last not in {b"", b"\n"})


def audit_partition(partition: Path) -> dict:
    scope = load_scope(partition)
    files = {name: (partition / name).stat().st_size if (partition / name).is_file() else 0
             for name in FILES}
    validated = verify_partition(partition, GraphValidator())
    node_ids: set[str] = set()
    node_types: Counter[str] = Counter()
    null_lines: Counter[str] = Counter()
    null_line_extensions: Counter[str] = Counter()
    source_missing = 0
    source_line_bad = 0
    null_line_disallowed = 0
    source_max_line: dict[Path, int] = defaultdict(int)
    for batch in pq.ParquetFile(partition / "nodes.parquet").iter_batches(
            columns=["node_id", "node_type", "source_file", "source_line"], batch_size=8192):
        for row in batch.to_pylist():
            node_ids.add(row["node_id"])
            node_type = row["node_type"]
            node_types[node_type] += 1
            source_file = row["source_file"]
            source = REPO / source_file if isinstance(source_file, str) else None
            if source is None or not source.is_file():
                source_missing += 1
            line = row["source_line"]
            extension = Path(source_file).suffix if isinstance(source_file, str) else "<missing>"
            if line is None:
                null_lines[node_type] += 1
                null_line_extensions[extension] += 1
                if extension != ".json":
                    null_line_disallowed += 1
            elif extension not in {".json", ".jsonl"}:
                source_line_bad += 1
            elif type(line) is not int or line < 1:
                source_line_bad += 1
            elif source is not None and source.is_file():
                source_max_line[source] = max(source_max_line[source], line)
    for source, maximum in source_max_line.items():
        if maximum > _line_count(source):
            for batch in pq.ParquetFile(partition / "nodes.parquet").iter_batches(
                    columns=["source_file", "source_line"], batch_size=8192):
                source_line_bad += sum(row["source_line"] is not None and
                    REPO / row["source_file"] == source and row["source_line"] > _line_count(source)
                    for row in batch.to_pylist())
    dangling = 0
    for batch in pq.ParquetFile(partition / "edges.parquet").iter_batches(
            columns=["source", "target"], batch_size=8192):
        for row in batch.to_pylist():
            dangling += row["source"] not in node_ids or row["target"] not in node_ids
    unresolved_reasons: Counter[str] = Counter()
    unresolved_invalid = 0
    unresolved_null_lines = 0
    for row in scope["unresolved"]:
        if not isinstance(row, dict) or not all(key in row for key in
                ("source_file", "source_line", "reason")) or not row.get("source_file") or not row.get("reason"):
            unresolved_invalid += 1
            continue
        unresolved_reasons[row["reason"]] += 1
        unresolved_null_lines += row["source_line"] is None
    gates = {
        "traceability": source_missing == 0 and source_line_bad == 0,
        "null_line_format": null_line_disallowed == 0,
        "status": scope["materialization_status"] in {"complete", "complete_with_unresolved"},
        "no_dangling": dangling == 0,
        "unresolved_enumerated": unresolved_invalid == 0,
        "three_nonempty_files": all(files.values()),
    }
    return {"sample_key": scope["sample_key"], "status": scope["materialization_status"],
            "counts": validated, "file_bytes": files, "gates": gates,
            "source_missing": source_missing, "source_line_bad": source_line_bad,
            "null_line_disallowed": null_line_disallowed,
            "null_line_extensions": dict(sorted(null_line_extensions.items())),
            "null_lines": dict(sorted(null_lines.items())),
            "node_types": dict(sorted(node_types.items())), "dangling": dangling,
            "unresolved_count": len(scope["unresolved"]), "unresolved_invalid": unresolved_invalid,
            "unresolved_null_lines": unresolved_null_lines,
            "unresolved_reasons": dict(sorted(unresolved_reasons.items()))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", type=Path)
    args = parser.parse_args()
    result = audit_partition(args.partition)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not all(result["gates"].values()):
        raise SystemExit(1)
