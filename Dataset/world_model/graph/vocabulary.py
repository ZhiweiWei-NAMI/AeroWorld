"""Freeze graph category and role indices across all materialized samples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from Dataset.world_model.graph.contract import CONTRACT_VERSION, REPO, iter_jsonl
from Dataset.world_model.graph.records import DEFAULT_ROOT, iter_partitions, load_scope


DEFAULT_VOCAB = REPO / "design/belief/category_index.jsonl"


def script_source_provenance(catalog: list[dict]) -> dict[str, dict[str, object]]:
    """Freeze script definitions and identify saved-only intent extensions."""
    provenance: dict[str, dict[str, object]] = {}
    for row in catalog:
        if row["kind"] != "script_event_category":
            continue
        scripts = row["source_scripts"]
        paths = scripts or sorted({source.rsplit(":", 1)[0]
                                   for source in row["realization_sources"]})
        entries = []
        for path in sorted(paths):
            if not (REPO / path).is_file():
                raise FileNotFoundError(REPO / path)
            entries.append({"source_file": path})
        scenarios = {path.rsplit("/", 1)[0] for path in scripts
                     if path.startswith("Dataset/scenarios/")}
        provenance[row["id"]] = {
            "source_scripts": entries if scripts else [],
            "source_realizations": entries if not scripts else [],
            "category_status": row["execution_status"],
            "single_scenario": len(scenarios) == 1 if scripts else None,
        }
    return provenance


def build_vocabulary(output: Path = DEFAULT_VOCAB, root: Path = DEFAULT_ROOT,
                     *, contract_path: Path = REPO / "design/belief/contract.jsonl") -> dict[str, int]:
    category_counts: dict[tuple[str, str | None], int] = defaultdict(int)
    role_counts: dict[str, int] = defaultdict(int)
    samples = 0
    for path in iter_partitions(root):
        scope = load_scope(path)
        if scope["contract_version"] != CONTRACT_VERSION:
            raise ValueError(f"contract version differs in {path}")
        samples += 1
        for batch in pq.ParquetFile(path / "nodes.parquet").iter_batches(
                columns=["node_type", "category_id"], batch_size=65536):
            for row in pa.Table.from_batches([batch]).group_by(
                    ["node_type", "category_id"]).aggregate([("node_type", "count")]).to_pylist():
                category_counts[(row["node_type"], row["category_id"])] += row["node_type_count"]
        for batch in pq.ParquetFile(path / "edges.parquet").iter_batches(
                columns=["role"], batch_size=65536):
            for item in pc.value_counts(batch.column(0)).to_pylist():
                if item["values"] is not None:
                    role_counts[item["values"]] += item["counts"]
    catalog = [row for _, row in iter_jsonl(contract_path)]
    if not catalog or catalog[0].get("version") != CONTRACT_VERSION:
        raise ValueError("contract version differs while building vocabulary")
    script_sources = script_source_provenance(catalog)
    for row in catalog:
        if row["kind"] == "predicate":
            category_counts.setdefault(("PredicateAssertion", row["id"]), 0)
            category_counts.setdefault(("PredicateTransition", row["id"]), 0)
            category_counts.setdefault(("PredicateEpisodeOutcome", row["id"]), 0)
        elif row["kind"] == "event":
            category_counts.setdefault(("EventOccurrence", row["id"]), 0)
            category_counts.setdefault(("EventOutcome", row["id"]), 0)
        elif row["kind"] == "entity_category":
            category_counts.setdefault(("EntityIdentity", row["id"]), 0)
            category_counts.setdefault(("EntityState", row["id"]), 0)
        elif row["kind"] == "script_event_category":
            category_counts.setdefault(("ScriptEvent", row["id"]), 0)
        elif row["kind"] == "command_category":
            category_counts.setdefault(("ActionPlan", row["id"]), 0)
            category_counts.setdefault(("CommandDispatch", row["id"]), 0)
        elif row["kind"] == "context_category":
            category_counts.setdefault(("ContextAnchor", row["id"]), 0)
        elif row["kind"] == "event_phase_category":
            category_counts.setdefault(("EventPhase", row["id"]), 0)
        elif row["kind"] == "predicate_episode_outcome_category":
            category_counts.setdefault(("PredicateEpisodeOutcome", row["id"]), 0)
        elif row["kind"] == "time_category":
            category_counts.setdefault(("TimePoint", row["id"]), 0)
        elif row["kind"] == "graph_edge_role":
            role_counts.setdefault(row["id"], 0)
        if row["kind"] in {"predicate", "event"}:
            for role in row.get("roles", []):
                role_counts.setdefault(role["key"], 0)
    category_counts.setdefault(("EntityIdentity", None), 0)
    category_counts.setdefault(("EntityState", None), 0)
    category_counts.setdefault(("ScriptEvent", None), 0)
    existing_categories: dict[tuple[str, str | None], int] = {}
    existing_roles: dict[str, int] = {}
    if output.exists():
        old = [row for _, row in iter_jsonl(output)]
        if not old or old[0].get("contract_version") != CONTRACT_VERSION:
            raise ValueError("existing vocabulary has another contract version; full rebuild required")
        old_categories = [row for row in old if row["kind"] == "node_category"]
        old_roles = [row for row in old if row["kind"] == "edge_role"]
        if (len({(row["node_type"], row["category_id"]) for row in old_categories}) != len(old_categories)
                or len({row["role"] for row in old_roles}) != len(old_roles)):
            raise ValueError("existing vocabulary has duplicate category or role keys")
        existing_categories = {(row["node_type"], row["category_id"]): row["index"]
                               for row in old_categories}
        existing_roles = {row["role"]: row["index"] for row in old_roles}
        if len(set(existing_categories.values())) != len(existing_categories) or len(set(existing_roles.values())) != len(existing_roles):
            raise ValueError("existing vocabulary has duplicate indices")
        removed = set(existing_categories) - set(category_counts)
        if removed:
            raise ValueError(f"category removal requires contract version bump: {removed}")
        old_scripts = {row["category_id"]: {key: row.get(key) for key in
                       ("source_scripts", "source_realizations", "category_status", "single_scenario")}
                       for row in old_categories if row["node_type"] == "ScriptEvent" and row["category_id"] is not None}
        if set(old_scripts) != set(script_sources):
            raise ValueError("ScriptEvent intent vocabulary changed; bump contract version and rebuild all partitions")
        for intent, provenance in script_sources.items():
            if old_scripts[intent] != provenance:
                raise ValueError(f"ScriptEvent source changed for {intent}; bump contract version and rebuild all partitions")
        for role in existing_roles:
            role_counts.setdefault(role, 0)
    next_category = max(existing_categories.values(), default=-1) + 1
    for key in sorted(set(category_counts) - set(existing_categories), key=lambda item: (item[0], item[1] or "")):
        existing_categories[key] = next_category
        next_category += 1
    next_role = max(existing_roles.values(), default=-1) + 1
    for key in sorted(set(role_counts) - set(existing_roles)):
        existing_roles[key] = next_role
        next_role += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": "vocabulary", "contract_version": CONTRACT_VERSION,
                                 "materialized_samples": samples}, ensure_ascii=False) + "\n")
        for (node_type, category), index in sorted(existing_categories.items(), key=lambda item: item[1]):
            count = category_counts[(node_type, category)]
            stream.write(json.dumps({"kind": "node_category", "node_type": node_type,
                "category_id": category, "index": index, "instance_count": count,
                **(script_sources[category]
                   if node_type == "ScriptEvent" and category is not None else {})},
                ensure_ascii=False) + "\n")
        for role, index in sorted(existing_roles.items(), key=lambda item: item[1]):
            count = role_counts[role]
            stream.write(json.dumps({"kind": "edge_role", "role": role, "index": index,
                                     "instance_count": count}, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return {"samples": samples, "node_categories": len(category_counts), "edge_roles": len(role_counts)}


def extend_declared_roles(output: Path = DEFAULT_VOCAB, *,
                          contract_path: Path = REPO / "design/belief/contract.jsonl") -> int:
    """Add executable zero-instance roles without claiming current counts."""
    rows = [row for _, row in iter_jsonl(output)]
    if rows[0]["contract_version"] != CONTRACT_VERSION:
        raise ValueError("vocabulary version differs")
    known = {row["role"] for row in rows if row["kind"] == "edge_role"}
    next_index = max(row["index"] for row in rows if row["kind"] == "edge_role") + 1
    added = 0
    for _, declaration in iter_jsonl(contract_path):
        if declaration["kind"] != "graph_edge_role" or declaration["id"] in known:
            continue
        rows.append({"kind": "edge_role", "role": declaration["id"], "index": next_index,
            "instance_count": None, "count_status": "not_enumerated_current_release",
            "declaration_source": declaration["source"]})
        known.add(declaration["id"])
        next_index += 1
        added += 1
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return added


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_VOCAB)
    args = parser.parse_args()
    print(build_vocabulary(args.output))
