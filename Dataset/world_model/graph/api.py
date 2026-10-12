"""Browse saved label graphs, temporal values, PyG indices and contract drafts."""

from __future__ import annotations

import json
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from Dataset.world_model.graph.contract import CONTRACT_VERSION, EDGE_ENDPOINTS, REPO, iter_jsonl
from Dataset.world_model.graph.entity_identity import resolve_value_identity
from Dataset.world_model.graph.field_contract import (
    DEFAULT_DEFINITIONS, canonical_numeric, coordinate_context_for_value, sample_tick_clock,
)
from Dataset.world_model.graph.field_catalog import projection_owners, validate_normalizations
from Dataset.world_model.graph.source_projection import (
    _requires_source_projection, _source_family, project_node_values,
    projected_catalog_binding, projected_numeric_authority, projected_source_rows,
    source_value_type,
)
from Dataset.world_model.graph.sample_identity import (
    identity_envelope, node_source_scope,
)
from Dataset.world_model.graph.records import DEFAULT_ROOT, iter_partitions, load_scope, read_window
from Dataset.world_model.graph.validator import ContractError, GraphValidator
from Dataset.semantic_truth.entity_scope import participant_binding_allowed


app = FastAPI(title="Belief Label Graph")


@lru_cache(maxsize=1)
def _validator() -> GraphValidator:
    return GraphValidator()


def _current_index_rows(path: Path, version_key: str) -> tuple[list[dict], str]:
    if not path.is_file():
        return [], "missing"
    rows = [row for _, row in iter_jsonl(path)]
    if not rows:
        return [], "missing"
    if any(row.get(version_key) != CONTRACT_VERSION for row in rows):
        return [], "version_mismatch"
    return rows, "available"


def _generated_index_status() -> str:
    required = (
        (REPO / "design/belief/contract.jsonl", "version"),
        (REPO / "design/belief/category_index.jsonl", "contract_version"),
        (REPO / "design/belief/field_catalog.jsonl", "contract_version"),
        (DEFAULT_DEFINITIONS, "contract_version"),
    )
    for path, version_key in required:
        if not path.is_file():
            return "missing"
        first = next(iter_jsonl(path), None)
        if first is None:
            return "missing"
        if first[1].get(version_key) != CONTRACT_VERSION:
            return "version_mismatch"
    return "available"
DRAFT_ROOT = DEFAULT_ROOT / "drafts"


def _graph_stamp() -> tuple[int, int]:
    path = DEFAULT_ROOT / "coverage.json"
    return path.stat().st_mtime_ns, path.stat().st_size


@lru_cache(maxsize=1)
def _sample_snapshot(stamp: tuple[int, int]) -> dict[str, tuple[Path, dict]]:
    scopes = [(path, load_scope(path)) for path in iter_partitions() if "drafts" not in path.parts]
    return {scope["sample_key"]: (path, scope) for path, scope in scopes}


def _samples() -> dict[str, Path]:
    return {key: value[0] for key, value in _sample_snapshot(_graph_stamp()).items()}


@lru_cache(maxsize=1)
def _cached_field_catalog(stamp: tuple[int, int]) -> dict[str, dict]:
    rows = {row["id"]: row for _, row in iter_jsonl(REPO / "design/belief/field_catalog.jsonl")}
    validate_normalizations(list(rows.values()))
    return rows


def _field_catalog() -> dict[str, dict]:
    path = REPO / "design/belief/field_catalog.jsonl"
    return _cached_field_catalog((path.stat().st_mtime_ns, path.stat().st_size))


def _partition(sample_key: str) -> Path:
    status = _generated_index_status()
    if status != "available":
        raise HTTPException(409, f"label-graph indices are {status}")
    found = _samples().get(sample_key)
    if found is None:
        raise HTTPException(404, f"unknown sample {sample_key}")
    if load_scope(found)["contract_version"] != CONTRACT_VERSION:
        raise HTTPException(409, "label-graph partition version differs")
    return found


def _draft_file(sample_key: str, name: str) -> Path:
    if sample_key.startswith("formal:"):
        relative = Path("formal") / sample_key.removeprefix("formal:")
    elif sample_key.startswith("arm:"):
        relative = Path("arm") / sample_key.removeprefix("arm:")
    else:
        raise HTTPException(400, "invalid sample key")
    if any(part in {"..", "."} for part in relative.parts):
        raise HTTPException(400, "invalid sample path")
    return DRAFT_ROOT / relative / name


def _draft_rows(sample_key: str, name: str) -> list[dict]:
    path = _draft_file(sample_key, name)
    return [row for _, row in iter_jsonl(path)] if path.exists() else []


def _append_draft(sample_key: str, name: str, row: dict) -> None:
    path = _draft_file(sample_key, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _find_node(partition: Path, node_id: str) -> dict | None:
    return _find_nodes(partition, {node_id}).get(node_id)


def _tensor_record(partition: Path, record_id: str) -> dict:
    output = REPO / "Dataset/world_model/runs/belief_pyg" / partition.relative_to(DEFAULT_ROOT)
    metadata_path = output / "metadata.json"
    if not metadata_path.is_file():
        return {"status": "not_generated"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    from Dataset.world_model.graph.model_data import _tensor_inputs
    if metadata["artifact_inputs"] != _tensor_inputs(
            partition, REPO / "design/belief/field_catalog.jsonl"):
        return {"status": "stale"}
    position = next((row for _, row in iter_jsonl(output / "tensor_index.jsonl")
                     if row["record_id"] == record_id), None)
    return {"status": "available" if position is not None else "not_in_saved_tensor",
            "start_tick": metadata["start_tick"], "end_tick": metadata["end_tick"],
            "position": position}


def _find_nodes(partition: Path, node_ids: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    scope = load_scope(partition)
    for batch in pq.ParquetFile(partition / "nodes.parquet").iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            if row["node_id"] in node_ids:
                row["attrs"] = json.loads(row.pop("attrs_json"))
                family, _ = _source_family(row)
                node_source_scope(row, family, scope, require_stored=True)
                found[row["node_id"]] = row
        if len(found) == len(node_ids):
            break
    return found


@lru_cache(maxsize=128)
def _outgoing_edges(partition: Path, source_id: str) -> tuple[dict, ...]:
    found = []
    for batch in pq.ParquetFile(partition / "edges.parquet").iter_batches(
            columns=["edge_type", "source", "role"], batch_size=65536):
        found.extend(row for row in batch.to_pylist() if row["source"] == source_id)
    return tuple(found)


def _current_outgoing(sample_key: str, partition: Path, source_id: str) -> list[dict]:
    return [*_outgoing_edges(partition, source_id),
            *(row for row in _draft_rows(sample_key, "edges.jsonl") if row["source"] == source_id)]


@app.get("/")
def index():
    return FileResponse(REPO / "design/belief/index.html")


@app.get("/event-view.js")
def event_view_script():
    return FileResponse(REPO / "design/belief/event-view.js", media_type="application/javascript")


@app.get("/graph-view.js")
def graph_view_script():
    return FileResponse(REPO / "design/belief/graph-view.js", media_type="application/javascript")


@app.get("/graph-view.css")
def graph_view_style():
    return FileResponse(REPO / "design/belief/graph-view.css", media_type="text/css")


@app.get("/api/event-view")
def focused_event_view(sample_key: str, event_id: str = "", entity_id: str = "",
                       start_tick: int | None = None, end_tick: int | None = None,
                       offset: int = Query(0, ge=0), limit: int = Query(2000, ge=1, le=10000)):
    from Dataset.world_model.graph.event_view import event_window
    partition = _partition(sample_key)
    try:
        result = event_window(partition, event_id=event_id, entity_id=entity_id,
            start_tick=start_tick, end_tick=end_tick, offset=offset, limit=limit)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    result["tensor"] = _tensor_summary(partition)
    return result


@app.get("/api/prediction-input")
def prediction_input(sample_key: str, cutoff_tick: int, history_ticks: int = Query(100, ge=0),
                     observation: str = "structured_logs"):
    from Dataset.world_model.graph.prediction import read_prediction_view, to_pyg
    try:
        view = read_prediction_view(_partition(sample_key), cutoff_tick,
                                    history_ticks=history_ticks, observation=observation)
        tensor = to_pyg(view)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(400, str(error)) from error
    return {**view, "tensor": {"node_counts": {kind: tensor[kind].num_nodes for kind in tensor.node_types},
        "edge_counts": {"/".join(kind): tensor[kind].edge_index.shape[1] for kind in tensor.edge_types},
        "state_features": tensor.input_metadata["state_features"],
        "log_features": tensor.input_metadata["log_features"],
        "log_feature_units": tensor.input_metadata["log_feature_units"]}}


@app.get("/api/intervention-pair")
def intervention_pair(sample_key: str):
    path = REPO / "design/belief/intervention_pairs.json"
    if not path.is_file():
        return {"status": "not_generated"}
    catalog = json.loads(path.read_text())
    if catalog.get("graph_version") != CONTRACT_VERSION:
        return {"status": "version_mismatch"}
    entries = [r for r in catalog["pairs"] if sample_key in {r["sample_key"], r["factual_sample"]}]
    return {"status": "available", "pairs": [json.loads((REPO / r["pair"]).read_text()) for r in entries]}


@app.get("/api/event-tasks")
def event_task_catalog():
    path = REPO / "design/belief/event_tasks.json"
    return json.loads(path.read_text()) if path.is_file() else {"status": "not_generated"}


@app.get("/api/catalog")
def catalog(kind: str | None = None):
    path = REPO / "design/belief/contract.jsonl"
    status = _generated_index_status()
    rows = [row for _, row in iter_jsonl(path)] if status == "available" else []
    if rows and rows[0].get("kind") != "schema":
        rows, status = [], "version_mismatch"
    if kind:
        rows = [row for row in rows if row["kind"] == kind]
    return {"rows": rows, "status": status}


@app.get("/api/fields")
def fields(limit: int = Query(200, ge=1, le=2000), offset: int = Query(0, ge=0),
           search: str = "", owner: str | None = None):
    path = REPO / "design/belief/field_catalog.jsonl"
    status = _generated_index_status()
    rows, status = _current_index_rows(path, "contract_version") if status == "available" else ([], status)
    if status == "available":
        try:
            validate_normalizations(rows)
        except ValueError as error:
            return {"rows": [], "total": 0, "status": "structure_incomplete", "reason": str(error)}
    rows = [row for row in rows if (search.lower() in row["id"].lower() or
            any(search.lower() in binding["normalized_id"].lower()
                for binding in row.get("normalizations", ())))
            and (owner is None or owner in projection_owners(row))]
    return {"rows": rows[offset:offset + limit], "total": len(rows), "status": status}


@app.get("/api/field-definitions")
def field_definitions(limit: int = Query(200, ge=1, le=2000), offset: int = Query(0, ge=0),
                      search: str = ""):
    status = _generated_index_status()
    rows, status = (_current_index_rows(DEFAULT_DEFINITIONS, "contract_version")
                    if status == "available" else ([], status))
    rows = [row for row in rows if search.lower() in row["id"].lower()]
    return {"rows": rows[offset:offset + limit], "total": len(rows), "status": status}


@app.get("/api/values")
def values(sample_key: str, start_tick: int, end_tick: int,
           entity_id: str = "", entity_namespace: str = "", field_id: str = "",
           field_index: int | None = Query(None, ge=0),
           limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0)):
    return _value_rows(sample_key, start_tick, end_tick, entity_id, entity_namespace,
                       field_id, field_index, limit, offset)


def _value_rows(sample_key: str, start_tick: int, end_tick: int,
                entity_id: str, entity_namespace: str, field_id: str,
                field_index: int | None, limit: int, offset: int,
                record_ids: set[str] | None = None):
    if start_tick > end_tick:
        raise HTTPException(400, "start_tick exceeds end_tick")
    if entity_namespace and entity_namespace not in {"sample", "compute_node", "sumo_tls_controller", "sumo_lane"}:
        raise HTTPException(400, "unknown entity namespace")
    partition = _partition(sample_key)
    scope = load_scope(partition)
    sample_clock = sample_tick_clock(scope["episode_id"])
    owner_sample = identity_envelope(scope)
    catalog_rows = _field_catalog()
    identities: dict[tuple[str, str], tuple[None, dict]] = {}
    parquet = pq.ParquetFile(partition / "nodes.parquet")
    for batch in parquet.iter_batches(columns=["node_id", "node_type", "category_id", "attrs_json"], batch_size=8192):
        batch = batch.filter(pc.equal(batch.column("node_type"), "EntityIdentity"))
        for row in batch.to_pylist():
            if row["node_type"] != "EntityIdentity":
                continue
            attrs = json.loads(row["attrs_json"])
            raw_id = attrs.get("entity_id")
            namespace = attrs.get("identity_namespace")
            if namespace not in {"sample", "compute_node", "sumo_tls_controller"}:
                raise ValueError(f"identity node lacks a declared namespace: {row['node_id']}")
            key = namespace, raw_id
            if key in identities:
                raise ValueError(f"sample identity ID is ambiguous: {sample_key}:{key}")
            identities[key] = None, {"node_id": row["node_id"], "attrs": attrs,
                                     "category_id": row.get("category_id")}
    result = []
    matching = 0
    wanted_records = pa.array(sorted(record_ids), type=pa.string()) if record_ids is not None else None
    for batch in parquet.iter_batches(batch_size=2048):
        if wanted_records is not None:
            batch = batch.filter(pc.is_in(batch.column("node_id"), value_set=wanted_records))
        for stored in batch.to_pylist():
            node = {**stored, "attrs": json.loads(stored["attrs_json"])}
            family, _ = _source_family(node)
            raw = node["attrs"].get("raw")
            if family is None or (raw is None and node["attrs"].get("fields") is None):
                continue
            source_scope = node_source_scope(node, family, scope, require_stored=True)
            for projected in project_node_values(node, catalog=catalog_rows,
                                                 scope=source_scope, owner_scope=scope,
                                                 require_stored_scope=True,
                                                 include_absent=True):
                sources = projected.sources
                catalog_ids = ([source.source_field_id for source in sources] if sources else
                               [projected.absent_source_field_id])
                declared_sources = [catalog_rows.get(source_id) for source_id in catalog_ids]
                if any(declared is None for declared in declared_sources):
                    raise ValueError(f"value source field is absent from catalog: {catalog_ids}")
                owner_sets = {projection_owners(declared) for declared in declared_sources}
                if len(owner_sets) != 1:
                    raise ValueError(f"normalized value sources have conflicting owners: "
                                     f"{projected.normalized_id}:{sorted(owner_sets)}")
                owners = set(next(iter(owner_sets)))
                for source, declared in zip(sources, declared_sources):
                    leaf = source.field_value
                    if _requires_source_projection(node, family, leaf.field_path) and (
                            node["node_type"] not in projection_owners(declared)):
                        raise ValueError(f"source field owner differs from value projection: {source.source_field_id}")
                if node["node_type"] not in owners:
                    continue
                normalized_index = projected_catalog_binding(projected, catalog_rows)
                declared = declared_sources[0]
                assert declared is not None
                leaf = sources[0].field_value if sources else None
                catalog_id = catalog_ids[0]
                tick = (leaf.value_tick if leaf.value_tick is not None or leaf.time_role is not None
                        else node["tick"]) if leaf is not None else node["tick"]
                query_tick = tick if tick is not None else node["tick"]
                if query_tick is not None and not start_tick <= query_tick <= end_tick:
                    continue
                projected_entity_id = leaf.entity_id if leaf else node["attrs"].get("entity_id")
                projected_namespace = (leaf.entity_namespace if leaf else
                                       node["attrs"].get("identity_namespace"))
                if entity_id and projected_entity_id != entity_id:
                    continue
                if entity_namespace and projected_namespace != entity_namespace:
                    continue
                if field_id and field_id not in {*catalog_ids, projected.normalized_id}:
                    continue
                if field_index is not None and field_index != normalized_index:
                    continue
                identity = resolve_value_identity(leaf, node, identities) if leaf is not None else None
                semantic_pairs = {(item["semantic_id"], item["meaning_status"])
                                  for item in declared_sources}
                semantic_pair = next(iter(semantic_pairs)) if len(semantic_pairs) == 1 else None
                semantic_id = (semantic_pair[0] if semantic_pair is not None and
                               semantic_pair[1] == "declared" and
                               projected.structure_status != "structure_conflict" else None)
                semantic_status = "declared" if semantic_id is not None else "unbound"
                coordinate_context = (coordinate_context_for_value(
                    family, leaf.field_path, node["source_file"], raw)
                    if leaf is not None else None)
                if (projected.value_status != "present" or
                        projected.structure_status not in {"source_only", "resolved"}):
                    canonical, conversion_reason = None, "source structural value is not present"
                elif type(projected.value) in {int, float}:
                    mapping, coordinate_context, semantic_id, semantic_status = projected_numeric_authority(
                        projected, catalog_rows, node["source_file"], raw)
                    canonical, conversion_reason = (
                        canonical_numeric(projected.value, mapping, sample_clock=sample_clock,
                                          coordinate_context=coordinate_context)
                        if semantic_status == "declared" else
                        (None, "source meaning has no shared declaration"))
                else:
                    canonical, conversion_reason = None, "source value is not numeric"
                if matching >= offset:
                    result.append({"field_id": projected.normalized_id,
                        "field_index": normalized_index,
                        "source_field_id": catalog_id,
                        "source_field_indices": [item["index"] for item in declared_sources],
                        "owner_record_id": node["node_id"],
                        "owner_sample": owner_sample, "source_scope": source_scope,
                        "semantic_id": semantic_id,
                        "semantic_binding_status": semantic_status,
                        "source_semantic_ids": [item["semantic_id"] for item in declared_sources],
                        "meaning_status": declared["meaning_status"],
                        "normalized_id": projected.normalized_id,
                        "normalized_index": normalized_index,
                        "normalized_value": projected.value,
                        "normalized_value_type": projected.value_type,
                        "value_status": projected.value_status,
                        "structure_status": projected.structure_status,
                        "binding_slot": projected.binding_slot,
                        "declared_shape": list(projected.declared_shape)
                            if projected.declared_shape is not None else None,
                        "observed_shape": list(projected.observed_shape)
                            if projected.observed_shape is not None else None,
                        "normalized_shape": list(projected.normalized_shape)
                            if projected.normalized_shape is not None else None,
                        "source_field_ids": catalog_ids,
                        "sources": projected_source_rows(projected, catalog=catalog_rows),
                        "envelope_tokens": list(projected.envelope_tokens),
                        "payload_tokens": list(projected.payload_tokens),
                        "payload_positions": list(projected.payload_positions),
                        "payload_shape": list(projected.payload_shape),
                        "source_branch": leaf.source_branch if leaf else None,
                        "source_subtype": leaf.source_subtype if leaf else None,
                        "predicate_id": leaf.predicate_id if leaf else None,
                        "source_path_key": leaf.source_path_key if leaf else None,
                        "entity_id": projected_entity_id,
                        "entity_namespace": projected_namespace,
                        "target_scope": leaf.target_scope if leaf else None,
                        "identity_node_id": identity.node_id if identity else node["attrs"].get("identity_ref"),
                        "identity_status": identity.status if identity else node["attrs"].get("identity_status"),
                        "identity_reason": identity.reason if identity else None,
                        "identity_basis": identity.basis if identity else node["attrs"].get("identity_basis"),
                        "identity_basis_path": identity.basis_path if identity else node["attrs"].get("identity_basis_path"),
                        "snapshot_tick": leaf.snapshot_tick if leaf else None, "value_tick": tick,
                        "time_role": (leaf.time_role if leaf else None) or ("owner_node_tick" if node["tick"] is not None else None),
                        "endpoint_role": leaf.endpoint_role if leaf else None,
                        "record_observed": projected.record_observed,
                        "value": leaf.value if leaf else None,
                        "value_type": source_value_type(leaf.value) if leaf else None,
                        "source_unit": declared["source_unit"],
                        "canonical_unit": declared["canonical_unit"],
                        "coordinate_status": declared["coordinate_status"],
                        "coordinate_frame": declared["coordinate_frame"],
                        "canonical_value": canonical,
                        "coordinate_context": coordinate_context,
                        "conversion_reason": conversion_reason,
                        "element_positions": list(leaf.element_positions) if leaf else [],
                        "raw_path": leaf.raw_path if leaf else None,
                        "source_tokens": list(leaf.source_tokens) if leaf else None,
                        "input_tokens": list(leaf.input_tokens) if leaf else None,
                        "source_file": node["source_file"], "source_line": node["source_line"]})
                    if len(result) > limit:
                        return {"rows": result[:limit], "has_more": True, "status": "available"}
                matching += 1
    return {"rows": result, "has_more": False, "status": "available"}


@app.get("/api/samples")
def samples():
    status = _generated_index_status()
    if status != "available":
        return {"samples": [], "status": status}
    scopes = [value[1] for value in _sample_snapshot(_graph_stamp()).values()]
    if any(scope["contract_version"] != CONTRACT_VERSION for scope in scopes):
        raise HTTPException(409, "label-graph partition version differs")
    return {"samples": scopes, "status": "available"}


@app.get("/api/window")
def window(sample_key: str, start_tick: int, end_tick: int,
           limit: int = Query(2000, ge=1, le=10000), offset: int = Query(0, ge=0),
           node_type: str | None = None, search: str = ""):
    partition = _partition(sample_key)
    result = read_window(partition, start_tick, end_tick, limit=limit, offset=offset,
                         node_type=node_type, search=search)
    draft_nodes = [row for row in _draft_rows(sample_key, "nodes.jsonl")
                   if row.get("tick") is None or start_tick <= row["tick"] <= end_tick]
    draft_ids = {row["node_id"] for row in draft_nodes}
    ids = {row["node_id"] for row in result["nodes"]} | draft_ids
    result["nodes"].extend(draft_nodes)
    result["edges"].extend(row for row in _draft_rows(sample_key, "edges.jsonl")
                           if row["source"] in ids and row["target"] in ids)
    return result


@app.get("/api/record")
def record(sample_key: str, record_id: str):
    partition = _partition(sample_key)
    node = _find_node(partition, record_id)
    if node is None:
        node = next((row for row in _draft_rows(sample_key, "nodes.jsonl") if row["node_id"] == record_id), None)
    if node is None:
        raise HTTPException(404, "record not found")
    return {"record": node, "tensor": _tensor_record(partition, record_id)}


@app.get("/api/tensor-record")
def tensor_record(sample_key: str, record_id: str):
    return _tensor_record(_partition(sample_key), record_id)


@app.post("/api/allowed-targets")
def allowed_targets(request: dict[str, Any]):
    sample_key = request["sample_key"]
    partition = _partition(sample_key)
    source_id = request["source_id"]
    candidate_ids = set(request.get("candidate_ids", []))
    found = _find_nodes(partition, candidate_ids | {source_id})
    drafts = {row["node_id"]: row for row in _draft_rows(sample_key, "nodes.jsonl")}
    source = found.get(source_id) or drafts.get(source_id)
    if source is None:
        raise HTTPException(404, "source node not found")
    candidates = []
    for candidate_id in request.get("candidate_ids", []):
        target = found.get(candidate_id) or drafts.get(candidate_id)
        if target is not None:
            candidates.append(target)
    return {"targets": _validator().allowed_targets(source, request["edge_type"], request.get("role"),
                                                   candidates, _current_outgoing(sample_key, partition, source_id))}


@app.post("/api/draft/node")
def draft_node(request: dict[str, Any]):
    sample_key = request["sample_key"]
    scope = load_scope(_partition(sample_key))
    tick = request.get("tick")
    if tick is not None and not scope["start_tick"] <= tick <= scope["end_tick"]:
        raise HTTPException(400, "draft tick outside sample window")
    node = {"node_id": sample_key + "|draft:" + str(uuid.uuid4()), "sample_key": sample_key,
            "node_type": request["node_type"], "category_id": request.get("category_id"),
            "tick": tick, "attrs": request.get("attrs", {}), "source_file": "draft",
            "source_line": None, "rule_ref": request.get("rule_ref"),
            "source_replay_status": "draft"}
    if node["node_type"] == "EntityIdentity":
        entity = node["attrs"].get("entity_id")
        if entity:
            node["attrs"]["event_actor_allowed"] = participant_binding_allowed(entity, "actor", {})
            node["attrs"]["event_non_actor_allowed"] = participant_binding_allowed(entity, "region", {})
    for field in node["attrs"].get("fields", []):
        if field.get("value_type") == "float" and type(field.get("value")) is int:
            field["value"] = float(field["value"])
    try:
        _validator().validate_node(node, draft=True)
    except ContractError as exc:
        raise HTTPException(422, str(exc)) from exc
    _append_draft(sample_key, "nodes.jsonl", node)
    return node


@app.post("/api/draft/edge")
def draft_edge(request: dict[str, Any]):
    sample_key = request["sample_key"]
    partition = _partition(sample_key)
    draft_nodes = {row["node_id"]: row for row in _draft_rows(sample_key, "nodes.jsonl")}
    source = draft_nodes.get(request["source"]) or _find_node(partition, request["source"])
    target = draft_nodes.get(request["target"]) or _find_node(partition, request["target"])
    if source is None or target is None:
        raise HTTPException(404, "edge endpoint absent")
    edge = {"edge_id": sample_key + "|draft:e:" + str(uuid.uuid4()), "sample_key": sample_key,
            "edge_type": request["edge_type"], "source": request["source"], "target": request["target"],
            "role": request.get("role"), "attrs": request.get("attrs", {}),
            "source_file": "draft", "source_line": None, "rule_ref": request.get("rule_ref"),
            "source_replay_status": "draft"}
    try:
        _validator().validate_edge(edge, source, target, draft=True)
        choice = _validator().allowed_targets(source, edge["edge_type"], edge["role"], [target],
                                            _current_outgoing(sample_key, partition, source["node_id"]))[0]
        if not choice["allowed"]:
            raise ContractError(choice["reason"])
    except (ContractError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    _append_draft(sample_key, "edges.jsonl", edge)
    return edge


@app.post("/api/draft/validate")
def validate_draft(request: dict[str, Any]):
    sample_key = request["sample_key"]
    partition = _partition(sample_key)
    nodes = _draft_rows(sample_key, "nodes.jsonl")
    edges = _draft_rows(sample_key, "edges.jsonl")
    involved = {edge["source"] for edge in edges} | {edge["target"] for edge in edges}
    draft_ids = {row["node_id"] for row in nodes}
    existing_ids = involved - draft_ids
    existing_map = _find_nodes(partition, existing_ids)
    existing = [existing_map.get(node_id) for node_id in existing_ids]
    if any(row is None for row in existing):
        raise HTTPException(422, "draft edge endpoint missing from graph")
    try:
        result = _validator().validate_graph(nodes + existing, edges, draft=False,
                                          complete_node_ids=draft_ids)
    except ContractError as exc:
        return {"valid": False, "reason": str(exc)}
    return {"valid": True, "counts": result, "draft_only": True}


def main() -> None:
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


def _decoded(stored: dict) -> dict:
    row = dict(stored)
    row["attrs"] = json.loads(row.pop("attrs_json"))
    return row


@lru_cache(maxsize=8)
def _identity_records(partition: Path, stamp: tuple[int, int]) -> tuple[dict, ...]:
    table = pq.read_table(partition / "nodes.parquet", filters=[("node_type", "=", "EntityIdentity")])
    return tuple(_decoded(row) for row in table.to_pylist())


def _tensor_summary(partition: Path) -> dict:
    output = REPO / "Dataset/world_model/runs/belief_pyg" / partition.relative_to(DEFAULT_ROOT)
    path = output / "metadata.json"
    if not path.is_file():
        return {"status": "not_generated", "nodes": None, "edges": None,
                "numeric_counts": None, "supervision_status": None}
    from Dataset.world_model.graph.model_data import _tensor_inputs
    metadata = json.loads(path.read_text(encoding="utf-8"))
    status = "available" if metadata["artifact_inputs"] == _tensor_inputs(
        partition, REPO / "design/belief/field_catalog.jsonl") else "stale"
    return {"status": status, **{key: metadata[key] for key in (
        "nodes", "edges", "numeric_counts", "supervision_status")}}


@app.get("/api/temporal-view")
def temporal_view(sample_key: str, start_tick: int, end_tick: int,
                  entity_id: str = "", entity_namespace: str = "", field_id: str = "",
                  limit: int = Query(1200, ge=10, le=3000)):
    partition = _partition(sample_key)
    scope = load_scope(partition)
    if not scope["start_tick"] <= start_tick <= end_tick <= scope["end_tick"]:
        raise HTTPException(400, "window exceeds saved sample scope")
    if entity_namespace and entity_namespace not in {"sample", "compute_node", "sumo_tls_controller", "sumo_lane"}:
        raise HTTPException(400, "unknown entity namespace")
    catalog = _field_catalog()
    if field_id and field_id not in catalog and field_id not in validate_normalizations(list(catalog.values())):
        raise HTTPException(400, "unknown field ID")
    node_path = partition / "nodes.parquet"
    identity_rows = _identity_records(partition, (node_path.stat().st_mtime_ns, node_path.stat().st_size))
    selected_identities = [row for row in identity_rows if
        (not entity_id or row["attrs"]["entity_id"] == entity_id) and
        (not entity_namespace or row["attrs"]["identity_namespace"] == entity_namespace)]
    entity_filter = bool(entity_id or entity_namespace)
    identity_ids = {row["node_id"] for row in selected_identities}
    related = set(identity_ids)
    if entity_filter:
        def neighbors(seeds: set[str], kinds: set[str], *, target_only: bool = False) -> set[str]:
            ids = pa.array(sorted(seeds), type=pa.string())
            types = pa.array(sorted(kinds), type=pa.string())
            found = set(seeds)
            for batch in pq.ParquetFile(partition / "edges.parquet").iter_batches(
                    columns=["edge_type", "source", "target"], batch_size=32768):
                touches = pc.is_in(batch.column("target"), value_set=ids)
                if not target_only:
                    touches = pc.or_(touches, pc.is_in(batch.column("source"), value_set=ids))
                mask = pc.and_(touches, pc.is_in(batch.column("edge_type"), value_set=types))
                for row in batch.filter(mask).to_pylist():
                    found.update((row["source"], row["target"]))
            return found

        # Arguments and participants can bind to an actual state, rather than its identity.
        owned = neighbors(identity_ids, {"identity_state"})
        bindings = {"predicate_argument", "event_participant", "action_target",
                    "supplement_subject", "controller_state", "compute_state_node", "matrix_scope_entity"}
        related = neighbors(owned, bindings, target_only=True)
        # Preserve recorded transitions, phases, outcomes and support around these records.
        context_kinds = set(EDGE_ENDPOINTS) - bindings - {
            "identity_state", "compute_node_host", "at_time", "same_entity_next",
            "same_predicate_next", "RelativeState"}
        related = neighbors(neighbors(related, context_kinds), context_kinds)
        related.update(neighbors(owned, {"RelativeState"}))
    quotas = {"time": max(1, limit // 5), "identity": max(1, limit // 10),
              "state": max(1, limit // 4), "predicate": max(1, limit // 4),
              "transition": max(1, limit // 20), "event": max(1, limit // 10),
              "context": max(1, limit // 20)}
    groups = {key: [] for key in quotas}
    truncated_groups = set()
    timepoints = []
    for batch in pq.ParquetFile(node_path).iter_batches(batch_size=4096):
        ticks = batch.column("tick")
        mask = pc.or_(pc.is_null(ticks), pc.fill_null(
            pc.and_(pc.greater_equal(ticks, start_tick), pc.less_equal(ticks, end_tick)), False))
        for stored in batch.filter(mask).to_pylist():
            kind = stored["node_type"]
            group = ("time" if kind == "TimePoint" else "identity" if kind == "EntityIdentity" else
                     "state" if kind == "EntityState" else "predicate" if kind == "PredicateAssertion" else
                     "transition" if kind == "PredicateTransition" else "context" if kind == "ContextAnchor" else "event")
            if group == "time":
                timepoints.append(stored["tick"])
            elif entity_filter and stored["node_id"] not in related:
                continue
            if len(groups[group]) == quotas[group]:
                truncated_groups.add(group)
                continue
            groups[group].append(_decoded(stored))
    nodes = [row for rows in groups.values() for row in rows]
    nodes.sort(key=lambda row: (row["tick"] is None, row["tick"] or 0, row["node_type"], row["node_id"]))
    node_ids = {row["node_id"] for row in nodes}
    ids = pa.array(sorted(node_ids), type=pa.string())
    edges = []
    edges_truncated = False
    for batch in pq.ParquetFile(partition / "edges.parquet").iter_batches(batch_size=32768):
        sources = pc.is_in(batch.column("source"), value_set=ids)
        targets = pc.is_in(batch.column("target"), value_set=ids)
        if pc.any(pc.and_(pc.or_(sources, targets), pc.invert(pc.and_(sources, targets)))).as_py():
            edges_truncated = True
        for row in batch.filter(pc.and_(sources, targets)).to_pylist():
            if len(edges) >= limit * 3:
                edges_truncated = True
                break
            edges.append(_decoded(row))
    predicates = [{"record_id": row["node_id"], "predicate_id": row["category_id"],
                   "tick": row["tick"], "truth_status": row["attrs"]["truth_status"],
                   "bindings": row["attrs"]["bindings"], "source_file": row["source_file"],
                   "source_line": row["source_line"]} for row in nodes
                   if row["node_type"] == "PredicateAssertion"]
    event_ids = {row["node_id"] for row in groups["event"]}
    events = [{"record_id": row["node_id"], **{key: row[key] for key in (
        "node_type", "category_id", "tick", "attrs", "source_file", "source_line")}}
        for row in nodes if row["node_id"] in event_ids]
    series_nodes = {row["node_id"] for row in nodes if
        (field_id or row["node_type"] == "EntityState" or
         (row["node_type"] == "ContextAnchor" and row["tick"] is not None))}
    projected = _value_rows(sample_key, start_tick, end_tick, entity_id, entity_namespace,
                            field_id, None, limit * 4, 0, series_nodes)
    series = {}
    series_truncated = projected["has_more"] or bool(truncated_groups & {"state", "context"})
    for value in projected["rows"]:
        declared = catalog[value["source_field_id"]]
        if not set(declared["value_types"]) & {"int", "float"}:
            continue
        if not field_id and declared.get("time_role") == "time_point":
            continue
        key = (value["field_id"], value["source_field_id"], value["entity_namespace"],
               value["entity_id"], tuple(value["element_positions"]))
        if key not in series:
            if len(series) >= (limit if field_id else 6):
                series_truncated = True
                continue
            series[key] = {"id": json.dumps(key, ensure_ascii=False, separators=(",", ":")),
                "field_id": value["field_id"], "source_field_id": value["source_field_id"],
                "label": declared["field_path"], "canonical_unit": value["canonical_unit"],
                "entity_id": value["entity_id"], "entity_namespace": value["entity_namespace"],
                "element_positions": value["element_positions"], "points": []}
        series[key]["points"].append({"tick": value["value_tick"], "value": value["canonical_value"],
            "value_status": value["value_status"], "record_id": value["owner_record_id"],
            "conversion_reason": value["conversion_reason"],
            "source_file": value["source_file"], "source_line": value["source_line"]})
    for item in series.values():
        item["points"].sort(key=lambda point: (point["tick"] is None, point["tick"] or 0, point["record_id"]))
    entities = [{"node_id": row["node_id"], "entity_id": row["attrs"]["entity_id"],
        "entity_namespace": row["attrs"]["identity_namespace"], "category_id": row["category_id"],
        "identity_status": row["attrs"].get("identity_status")} for row in selected_identities[:limit]]
    return {"scope": scope, "window": {"start_tick": start_tick, "end_tick": end_tick},
        "entities": entities, "timepoints": sorted(set(timepoints)), "nodes": nodes,
        "edges": edges, "series": list(series.values()), "predicates": predicates, "events": events,
        "truncation": {"nodes": bool(truncated_groups) or len(selected_identities) > limit,
                       "edges": edges_truncated, "series": series_truncated,
                       "predicates": bool(truncated_groups & {"predicate", "transition"}),
                       "events": "event" in truncated_groups}, "tensor": _tensor_summary(partition)}


if __name__ == "__main__":
    main()
