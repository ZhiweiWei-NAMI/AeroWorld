"""Current source presence, separate from the published structure observation snapshot."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from Dataset.world_model.graph.contract import CONTRACT_VERSION, REPO, read_json, source_path
from Dataset.world_model.graph.field_catalog import declared_source_inventory

CURRENT_INVENTORY = REPO / "design/belief/source_inventory.jsonl"


def build_inventory(path: Path = CURRENT_INVENTORY) -> dict:
    declarations = list(declared_source_inventory())
    manifests = {}
    statuses = Counter()
    rows = []
    for index, declaration in enumerate(declarations):
        source = declaration.path
        present = source.is_file()
        count, count_basis = None, "not recounted; source presence is checked independently"
        if present and source.suffix == ".json":
            count, count_basis = 1, "one JSON root"
        if present and source.is_relative_to(REPO / "aw_data/objective_semantic_truth"):
            episode = source.parent.name
            if episode not in manifests:
                manifests[episode] = read_json(source.parent / "manifest.json")
            producer_counts = manifests[episode]["record_counts"]
            producer_counts = {**producer_counts, **(manifests[episode].get("business_event_truth") or {}).get("record_counts", {})}
            stem = source.stem
            producer_key = {"world_truth_graph_deltas": "world_truth_graph_deltas"}.get(stem, stem)
            if source.suffix == ".jsonl" and producer_key in producer_counts:
                count = producer_counts[producer_key]
                count_basis = source_path(source.parent / "manifest.json") + ":record_counts." + producer_key
        status = "source_present" if present else "source_missing"
        statuses[status] += 1
        rows.append({"kind": "source_declaration", "declaration_index": index,
            "contract_version": CONTRACT_VERSION, "source_family": declaration.source_family,
            "source_file": source_path(source), "declared_sample": declaration.sample,
            "identity_status": declaration.identity_status, "presence": declaration.presence,
            "delivery_status": status, "input_bytes": source.stat().st_size if present else None,
            "current_record_count": count, "record_count_basis": count_basis,
            "record_grammar_status": "declared; validated when read by the graph builder",
            "read_relation": declaration.read_relation, "read_condition": declaration.read_condition,
            "reference_samples": list(declaration.reference_samples)})
    summary = {"kind": "source_inventory_summary", "contract_version": CONTRACT_VERSION,
        "declarations": len(rows), "status_counts": dict(statuses),
        "structural_observation_snapshot": "design/belief/structure_source_inventory.jsonl",
        "observation_scope": "Current presence and producer counts; historical projected value counts are not relabelled."}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in [summary, *rows]:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return summary


if __name__ == "__main__":
    print(json.dumps(build_inventory(), ensure_ascii=False))
