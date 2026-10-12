"""Partitioned Parquet graph records and filtered window reads."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from Dataset.world_model.graph.contract import CONTRACT_VERSION, REPO, strip_source_digests
from Dataset.world_model.graph.entity_identity import business_object_binding_allowed, identity_node_id
from Dataset.world_model.graph.source_projection import _source_family
from Dataset.world_model.graph.sample_identity import (
    arm_identity, identity_envelope, node_source_scope, read_formal_identity,
    validate_scope_identity,
)
from Dataset.world_model.graph.validator import GraphValidator, ContractError, _same_typed
from Dataset.semantic_truth.entity_scope import entity_scope_contract
from Dataset.semantic_truth.facility_scope import ontology_class_is_a


DEFAULT_ROOT = REPO / "Dataset/world_model/runs/belief_label_graph"


@lru_cache(maxsize=1024)
def _source_classes_compatible(first: str, second: str) -> bool:
    return ontology_class_is_a(first, second) or ontology_class_is_a(second, first)

SCOPE_SCHEMA = pa.schema([
    ("sample_key", pa.string()), ("source_kind", pa.string()), ("episode_id", pa.string()),
    ("profile", pa.string()), ("seed", pa.string()), ("group", pa.string()),
    ("branch", pa.string()), ("arm_key", pa.string()),
    ("identity_sources_json", pa.large_string()),
    ("start_tick", pa.int32()), ("end_tick", pa.int32()),
    ("contract_version", pa.string()), ("source_root", pa.string()),
    ("materialization_status", pa.string()), ("counts_json", pa.large_string()),
    ("unresolved_json", pa.large_string()),
])
NODE_SCHEMA = pa.schema([
    ("node_id", pa.string()), ("sample_key", pa.string()), ("node_type", pa.string()),
    ("category_id", pa.string()), ("tick", pa.int32()), ("attrs_json", pa.large_string()),
    ("source_file", pa.string()), ("source_line", pa.int32()),
    ("rule_ref", pa.string()), ("source_replay_status", pa.string()),
])
EDGE_SCHEMA = pa.schema([
    ("edge_id", pa.string()), ("sample_key", pa.string()), ("edge_type", pa.string()),
    ("source", pa.string()), ("target", pa.string()), ("role", pa.string()),
    ("source_tick", pa.int32()), ("target_tick", pa.int32()),
    ("attrs_json", pa.large_string()), ("source_file", pa.string()),
    ("source_line", pa.int32()), ("rule_ref", pa.string()),
    ("source_replay_status", pa.string()),
])


def partition_for(scope: dict[str, Any], root: Path = DEFAULT_ROOT) -> Path:
    if scope["source_kind"] == "formal":
        return root / "formal" / scope["episode_id"]
    return root / "arm" / scope["profile"] / scope["seed"] / scope["group"] / scope["branch"]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def link_node_attrs(node: dict[str, Any]) -> dict[str, Any]:
    """Retain the same source-backed link evidence in writer and partition checks."""
    attrs = node["attrs"]
    keys = ("entity_id", "entity_category", "identity_status", "identity_ref",
        "identity_basis", "identity_basis_path", "identity_inherited_from", "identity_reason",
        "identity_namespace", "source_scope_type", "source_host_entity_id",
        "ontology_class", "classification_status", "raw_category_status", "ontology_status",
        "source_completeness", "background_role", "event_actor_allowed", "event_non_actor_allowed",
        "uav_role_classification", "bindings", "binding_ontology_classes", "truth_status",
        "raw_predicate_id", "assertion_layer", "from_value", "to_value", "from_tick", "to_tick", "anchor_kind", "source_family",
        "category_status", "target_entity_id", "native_action_id", "native_event_id", "source_object_field",
        "source_action_index", "audit_status", "audit_source_file", "audit_source_line",
        "dispatch_status", "origin", "controller_ids")
    result = {key: attrs.get(key) for key in keys}
    if node["node_type"] == "EntityIdentity" and isinstance(attrs.get("raw"), dict):
        result["_participant_scope"] = {key: value for key, value in attrs["raw"].items()
            if key in {"entity_id", "entity_category", "background_role", "role",
                       "uav_corridor_role", "semantic_role", "contract_inspect_uav"}}
    if node["node_type"] == "ScriptEvent" and node["source_file"].endswith("/event_realization.jsonl"):
        result["_action_refs"] = attrs["raw"]["action_realizations"]
    if node["node_type"] == "ActionPlan" and node["source_file"].endswith("/ue/event_script.json"):
        result["_action_payload"] = attrs["raw"]
    if node["node_type"] == "ContextAnchor" and node["category_id"] == "arm_action_audit":
        result["_action_payload"] = attrs["raw"]["action"]
    if node["node_type"] == "CommandDispatch":
        result["_action_payload"] = attrs["source_action"]
    if node["node_type"] == "ContextAnchor" and result["anchor_kind"] == "compute_comm_matrix":
        result["_scope_values"] = [
            (item["scope_type"], item["scope_entity_id"])
            for item in attrs["raw"]["scope_values"]]
    if (node["node_type"] == "ContextAnchor" and result["anchor_kind"] == "supplement_state"
            and attrs.get("source_layer") == "compute_comm_supplement"
            and "node_id" in attrs["raw"]):
        raw = attrs["raw"]
        result["_compute_ref"] = (raw["node_id"], raw["entity_id"])
    return result


class GraphWriter:
    """One-sample writer. Only the current sample's slim node index is retained."""

    def __init__(self, scope: dict[str, Any], root: Path = DEFAULT_ROOT, *,
                 validate: bool = True, replace_generated: bool = False):
        validate_scope_identity(scope)
        self.scope = scope
        self.path = partition_for(scope, root)
        if (self.path / "scope.parquet").exists():
            if not replace_generated:
                raise FileExistsError(f"partition already materialized: {self.path}")
            existing = load_scope(self.path)
            if (existing["sample_key"] != scope["sample_key"] or
                    existing["source_root"] != scope["source_root"]):
                raise ValueError(f"refusing to replace a different partition: {self.path}")
            expected = {"scope.parquet", "nodes.parquet", "edges.parquet"}
            if {item.name for item in self.path.iterdir()} != expected:
                raise ValueError(f"partition contains other files: {self.path}")
            for name in expected:
                (self.path / name).unlink()
        elif self.path.exists() and any(self.path.iterdir()):
            present = {item.name for item in self.path.iterdir()}
            if not replace_generated or not present <= {"nodes.parquet", "edges.parquet"}:
                raise ValueError(f"incomplete partition requires explicit replacement: {self.path}")
            for name in present:
                (self.path / name).unlink()
        self.path.mkdir(parents=True, exist_ok=True)
        self.validator = GraphValidator() if validate else None
        self.nodes = pq.ParquetWriter(self.path / "nodes.parquet", NODE_SCHEMA, compression="zstd")
        self.edges = pq.ParquetWriter(self.path / "edges.parquet", EDGE_SCHEMA, compression="zstd")
        self._node_batch: list[dict[str, Any]] = []
        self._edge_batch: list[dict[str, Any]] = []
        self.node_index: dict[str, dict[str, Any]] = {}
        self.edge_ids: set[str] = set()
        self.argument_roles: dict[str, Counter[str]] = {}
        self.event_roles: dict[str, Counter[str]] = {}
        self.controller_targets: dict[str, Counter[str]] = {}
        self.plan_links: dict[str, Counter[str]] = {}
        self.observed_dispatches: dict[str, Counter[int]] = {}
        self.observed_targets: Counter[str] = Counter()
        self.action_targets: Counter[str] = Counter()
        self.time_anchors: Counter[str] = Counter()
        self.counts: dict[str, int] = {"nodes": 0, "edges": 0}
        self.unresolved: list[dict[str, Any]] = []
        self._unresolved_role_classes: set[tuple[str, str, str]] = set()
        self._conflicting_role_classes: set[tuple[str, str, str, str, str]] = set()

    def add_node(self, node_id: str, node_type: str, category_id: str, tick: int | None,
                 attrs: dict[str, Any], source_file: str, source_line: int | None = None,
                 rule_ref: str | None = None, source_replay_status: str = "saved_unchecked") -> str:
        if isinstance(attrs.get("raw"), dict):
            family, _ = _source_family({"node_type": node_type, "attrs": attrs,
                                        "source_file": source_file})
            source_scope = node_source_scope(
                {"node_id": node_id, "sample_key": self.scope["sample_key"],
                 "node_type": node_type, "attrs": attrs, "source_file": source_file},
                family, self.scope)
            if source_scope is not None:
                attrs = {**attrs, "source_scope": source_scope}
        node = {"node_id": node_id, "sample_key": self.scope["sample_key"], "node_type": node_type,
                "category_id": category_id, "tick": tick,
                "attrs": {key: strip_source_digests(value) if key in {"raw", "source_action"}
                          else value for key, value in attrs.items()},
                "source_file": source_file, "source_line": source_line, "rule_ref": rule_ref,
                "source_replay_status": source_replay_status}
        if node_id in self.node_index:
            raise ContractError(f"duplicate node {node_id}")
        if self.validator:
            self.validator.validate_node(node)
        # The index retains the fields needed to validate links, not the raw state payload.
        self.node_index[node_id] = {**node, "attrs": link_node_attrs(node)}
        self._node_batch.append({**{key: value for key, value in node.items() if key != "attrs"},
                                 "attrs_json": _json(node["attrs"])})
        self.counts["nodes"] += 1
        if len(self._node_batch) >= 2048:
            self._flush_nodes()
        return node_id

    def add_edge(self, edge_type: str, source: str, target: str, *, role: str | None = None,
                 attrs: dict[str, Any] | None = None, source_file: str,
                 source_line: int | None = None, rule_ref: str | None = None,
                 source_replay_status: str = "saved_unchecked",
                 native_relation_id: str | None = None) -> str:
        if source not in self.node_index or target not in self.node_index:
            raise ContractError(f"dangling {edge_type}: {source} -> {target}")
        endpoints = (self.node_index[source], self.node_index[target])
        unresolved_identity = any(
            item["node_type"] in {"EntityIdentity", "EntityState"}
            and item["attrs"].get("identity_status") != "resolved"
            for item in endpoints)
        if unresolved_identity:
            if edge_type in {"identity_state", "same_entity_next"}:
                raise ContractError(f"{edge_type} requires resolved entity identity")
            source_replay_status = "unresolved_identity"
            self.unresolved.append({"source_file": source_file, "source_line": source_line,
                                    "relation": {"source": source, "target": target,
                                                 "edge_type": edge_type},
                                    "reason": "edge endpoint identity unresolved"})
        # Source-local ordinal makes parallel relations different even with identical endpoints.
        serial = self.counts["edges"]
        edge_id = f"{self.scope['sample_key']}|e:{serial}"
        edge = {"edge_id": edge_id, "sample_key": self.scope["sample_key"], "edge_type": edge_type,
                "source": source, "target": target, "role": role, "source_file": source_file,
                "source_line": source_line, "rule_ref": rule_ref,
                "source_replay_status": source_replay_status,
                "attrs": {**(attrs or {}), **({"native_relation_id": native_relation_id} if native_relation_id else {})}}
        if edge_type == "event_participant" and self.node_index[source]["node_type"] == "EventOccurrence":
            kinds = entity_scope_contract()["role_role_kinds"]
            target_attrs = self.node_index[target]["attrs"]
            event = self.validator.events[self.node_index[source]["category_id"]] if self.validator else None
            role_classes = {item["key"]: item["class"] for item in event["roles"]} if event else {}
            if business_object_binding_allowed(self.node_index[target],
                    self.node_index[source]["attrs"].get("bindings", {}).get(role),
                    role_classes.get(role, ""), edge["attrs"].get("source_object_evidence"),
                    self.node_index[source]["tick"]):
                allowed = True
            elif role_classes.get(role, "").removeprefix("world:") == "ComputeNode":
                allowed = (self.node_index[target]["category_id"] == "compute_node" and
                           target_attrs.get("identity_namespace") == "compute_node")
                if target_attrs.get("identity_status") != "resolved":
                    allowed = None
            elif (event or {}).get("role_policy", {}).get(role) == "actor" or role in kinds["actor_roles"]:
                allowed = target_attrs.get("event_actor_allowed")
            elif role in kinds["non_actor_roles"]:
                allowed = target_attrs.get("event_non_actor_allowed")
            else:
                allowed = None
            if allowed is None:
                edge["attrs"]["role_policy_status"] = "unresolved"
                if not unresolved_identity:
                    edge["source_replay_status"] = "unresolved_role_policy"
                self.unresolved_relation(source_file=source_file, source_line=source_line,
                    relation={"edge_type": edge_type, "source": source, "target": target, "role": role},
                    reason="event role has no governed actor/non-actor classification")
            elif allowed is False:
                edge["attrs"]["role_policy_status"] = "source_conflict"
                if not unresolved_identity:
                    edge["source_replay_status"] = "source_role_policy_conflict"
                self.unresolved_relation(source_file=source_file, source_line=source_line,
                    relation={"edge_type": edge_type, "source": source, "target": target, "role": role},
                    reason="saved event participant conflicts with roster binding policy")
        if edge_type in {"predicate_argument", "event_participant"}:
            parent = self.node_index[source]
            bound_class = (parent["attrs"].get("binding_ontology_classes") or {}).get(role)
            target_attrs = self.node_index[target]["attrs"]
            target_class = target_attrs.get("ontology_class")
            if bound_class and target_class and not _source_classes_compatible(bound_class, target_class):
                edge["attrs"]["ontology_claim_status"] = "source_conflict"
                edge["attrs"]["binding_ontology_class"] = bound_class
                edge["attrs"]["target_ontology_class"] = target_class
                if edge["source_replay_status"] == "saved_unchecked":
                    edge["source_replay_status"] = "source_role_class_conflict"
                self.counts["source_role_class_conflict_edges"] = self.counts.get("source_role_class_conflict_edges", 0) + 1
                issue = (parent["category_id"], str(role), str(target_attrs.get("entity_id") or target),
                         bound_class, target_class)
                if issue not in self._conflicting_role_classes:
                    self._conflicting_role_classes.add(issue)
                    self.unresolved_relation(source_file=source_file, source_line=source_line,
                        relation={"edge_type": edge_type, "category_id": parent["category_id"],
                                  "role": role, "entity_id": target_attrs.get("entity_id"),
                                  "binding_ontology_class": bound_class,
                                  "target_ontology_class": target_class},
                        reason="saved binding class conflicts with source-backed entity class")
            if not (bound_class or target_class):
                edge["attrs"]["ontology_role_status"] = "unresolved"
                if edge["source_replay_status"] == "saved_unchecked":
                    edge["source_replay_status"] = "unresolved_role_class"
                self.counts["unresolved_role_class_edges"] = self.counts.get("unresolved_role_class_edges", 0) + 1
                issue = (parent["category_id"], str(role), str(target_attrs.get("entity_id") or target))
                if issue not in self._unresolved_role_classes:
                    self._unresolved_role_classes.add(issue)
                    self.unresolved_relation(source_file=source_file, source_line=source_line,
                        relation={"edge_type": edge_type, "category_id": parent["category_id"],
                                  "role": role, "entity_id": target_attrs.get("entity_id")},
                        reason="bound entity has no source-backed ontology class")
        if self.validator:
            self.validator.validate_edge(edge, self.node_index[source], self.node_index[target])
        if edge_type == "predicate_argument":
            self.argument_roles.setdefault(source, Counter())[role] += 1
        if edge_type == "event_participant" and self.node_index[source]["node_type"] == "EventOccurrence":
            self.event_roles.setdefault(source, Counter())[role] += 1
        if edge_type == "controller_state":
            self.controller_targets.setdefault(source, Counter())[target] += 1
        if edge_type in {"action_plan_audit", "action_plan_dispatch"}:
            self.plan_links.setdefault(source, Counter())[edge_type] += 1
        if edge_type == "observed_dispatch" and self.node_index[source]["source_file"].endswith(
                "/event_realization.jsonl"):
            self.observed_dispatches.setdefault(source, Counter())[
                self.node_index[target]["attrs"]["source_action_index"]] += 1
            self.observed_targets[target] += 1
        if edge_type == "action_target" and self.node_index[source]["node_type"] == "CommandDispatch":
            self.action_targets[source] += 1
        if edge_type == "at_time":
            self.time_anchors[source] += 1
        source_tick, target_tick = self.node_index[source]["tick"], self.node_index[target]["tick"]
        self._edge_batch.append({**{key: value for key, value in edge.items() if key != "attrs"},
                                 "source_tick": source_tick, "target_tick": target_tick,
                                 "attrs_json": _json(edge["attrs"])})
        self.edge_ids.add(edge_id)
        self.counts["edges"] += 1
        if len(self._edge_batch) >= 4096:
            self._flush_edges()
        return edge_id

    def unresolved_relation(self, *, source_file: str, source_line: int | None, relation: Any,
                            reason: str) -> None:
        self.unresolved.append({"source_file": source_file, "source_line": source_line,
                                "relation": strip_source_digests(relation), "reason": reason})

    def _flush_nodes(self) -> None:
        if self._node_batch:
            self.nodes.write_table(pa.Table.from_pylist(self._node_batch, schema=NODE_SCHEMA))
            self._node_batch.clear()

    def _flush_edges(self) -> None:
        if self._edge_batch:
            self.edges.write_table(pa.Table.from_pylist(self._edge_batch, schema=EDGE_SCHEMA))
            self._edge_batch.clear()

    def close(self, status: str = "complete") -> dict[str, int]:
        if self.validator:
            for node in self.node_index.values():
                if (node["node_type"] in {"EntityState", "PredicateAssertion", "EventPhase", "CommandDispatch"}
                        or node["node_type"] == "ContextAnchor" and node["category_id"] in
                        {"formal_weather_meta", "sumo_controller_state", "arm_action_audit"}):
                    if self.time_anchors[node["node_id"]] != 1:
                        raise ContractError(f"node time anchor count differs: {node['node_id']}")
                if node["node_type"] == "ContextAnchor" and node["category_id"] == "sumo_controller_state":
                    expected = Counter({identity_node_id(self.scope["sample_key"], "sumo_tls_controller", item): 1
                                        for item in node["attrs"]["controller_ids"]})
                    if self.controller_targets.get(node["node_id"], Counter()) != expected:
                        raise ContractError(f"SUMO controller state edges differ: {node['node_id']}")
                if node["node_type"] == "ActionPlan":
                    audit_status = node["attrs"]["audit_status"]
                    expected = Counter({"action_plan_audit": 1, "action_plan_dispatch": 1}) if audit_status == "ok" else (
                        Counter({"action_plan_audit": 1}) if audit_status != "not_recorded" else Counter())
                    if self.plan_links.get(node["node_id"], Counter()) != expected:
                        raise ContractError(f"action plan audit/dispatch edges differ: {node['node_id']}")
                if node["node_type"] == "ScriptEvent" and node["source_file"].endswith(
                        "/event_realization.jsonl"):
                    expected = Counter(range(len(node["attrs"]["_action_refs"])))
                    if self.observed_dispatches.get(node["node_id"], Counter()) != expected:
                        raise ContractError(f"formal observed dispatch edges differ: {node['node_id']}")
                if node["node_type"] == "CommandDispatch":
                    expected = 1 if node["attrs"].get("target_entity_id") else 0
                    if self.action_targets[node["node_id"]] != expected:
                        raise ContractError(f"command target edges differ: {node['node_id']}")
                    if (node["source_file"].endswith("/event_realization.jsonl") and
                            self.observed_targets[node["node_id"]] != 1):
                        raise ContractError(f"formal command source edge differs: {node['node_id']}")
                if node["node_type"] == "PredicateAssertion":
                    signature = self.validator.predicates[node["category_id"]]
                    expected = Counter(item["key"] for item in signature["roles"])
                    actual = self.argument_roles.get(node["node_id"], Counter())
                    if actual != expected:
                        raise ContractError(f"predicate arguments incomplete: {node['node_id']}: {actual} != {expected}")
                    bindings = node["attrs"].get("bindings", {})
                    for group in signature.get("distinct_role_sets", []):
                        if len({bindings.get(role) for role in group}) != len(group):
                            raise ContractError(f"predicate distinct roles collide: {node['node_id']}")
                    for group in signature.get("unordered_role_sets", []):
                        if [bindings.get(role) for role in group] != sorted(bindings.get(role) for role in group):
                            raise ContractError(f"predicate unordered roles differ: {node['node_id']}")
                if node["node_type"] == "EventOccurrence":
                    signature = self.validator.events[node["category_id"]]
                    if signature["roles"]:
                        expected = Counter(item["key"] for item in signature["roles"])
                        actual = self.event_roles.get(node["node_id"], Counter())
                        if actual != expected:
                            raise ContractError(f"event participants incomplete: {node['node_id']}: {actual} != {expected}")
        self._flush_nodes()
        self._flush_edges()
        self.nodes.close()
        self.edges.close()
        row = {"sample_key": self.scope["sample_key"], "source_kind": self.scope["source_kind"],
               "episode_id": self.scope["episode_id"], "profile": self.scope.get("profile"),
               "seed": self.scope.get("seed"), "group": self.scope.get("group"), "branch": self.scope.get("branch"),
               "arm_key": self.scope.get("arm_key"),
               "identity_sources_json": _json(self.scope["identity_sources"]),
               "start_tick": self.scope.get("start_tick"), "end_tick": self.scope.get("end_tick"),
               "contract_version": CONTRACT_VERSION, "source_root": self.scope["source_root"],
               "materialization_status": status, "counts_json": _json(self.counts),
               "unresolved_json": _json(self.unresolved)}
        pq.write_table(pa.Table.from_pylist([row], schema=SCOPE_SCHEMA), self.path / "scope.parquet", compression="zstd")
        return self.counts


def load_scope(path: Path) -> dict[str, Any]:
    row = pq.read_table(path / "scope.parquet").to_pylist()[0]
    if row["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"partition contract version differs: {path}")
    row["identity_sources"] = json.loads(row.pop("identity_sources_json"))
    validate_scope_identity(row)
    expected = (read_formal_identity(row["episode_id"]) if row["source_kind"] == "formal"
                else arm_identity(REPO / "arms" / row["arm_key"]))
    for name in ("sample_key", "source_kind", "episode_id", "profile", "seed",
                 "group", "branch", "arm_key", "identity_sources"):
        if not _same_typed(row[name], expected[name]):
            raise ValueError(f"partition sample identity differs from source: {path}:{name}")
    suffix = (("formal", row["episode_id"]) if row["source_kind"] == "formal"
              else ("arm", row["profile"], row["seed"], row["group"], row["branch"]))
    if path.parts[-len(suffix):] != suffix:
        raise ValueError(f"partition path differs from sample identity: {path}")
    row["identity_envelope"] = identity_envelope(row)
    row["counts"] = json.loads(row.pop("counts_json"))
    row["unresolved"] = json.loads(row.pop("unresolved_json"))
    return row


def iter_partitions(root: Path = DEFAULT_ROOT) -> Iterable[Path]:
    yield from sorted(path.parent for path in root.rglob("scope.parquet"))


def read_window(path: Path, start_tick: int, end_tick: int, *, limit: int = 2000,
                offset: int = 0, node_type: str | None = None, search: str = "",
                neighbor_limit: int = 300) -> dict[str, Any]:
    """Read a page of matching nodes and linked neighbors from one sample."""
    if start_tick > end_tick or limit < 1 or limit > 10000 or offset < 0:
        raise ValueError("invalid window or page")
    scope = load_scope(path)
    node_file = pq.ParquetFile(path / "nodes.parquet")
    priority = {kind: 0 for kind in ("EventOccurrence", "EventPhase", "EventOutcome",
        "ScriptEvent", "ActionPlan", "CommandDispatch", "PredicateEpisodeOutcome", "PredicateTransition")}
    priority.update({"PredicateAssertion": 1, "ContextAnchor": 2, "EntityState": 3,
                     "EntityIdentity": 4, "TimePoint": 5})
    matching: list[tuple[int, int, str]] = []
    search_lower = search.lower()
    for batch in node_file.iter_batches(columns=["node_id", "node_type", "category_id", "tick"], batch_size=8192):
        for row in batch.to_pylist():
            tick = row["tick"]
            if tick is not None and not start_tick <= tick <= end_tick:
                continue
            if node_type and row["node_type"] != node_type:
                continue
            if search_lower and search_lower not in row["node_id"].lower() and search_lower not in (row["category_id"] or "").lower():
                continue
            matching.append((priority[row["node_type"]], tick if tick is not None else -1, row["node_id"]))
    matching.sort()
    page_ids = [item[2] for item in matching[offset:offset + limit]]
    ids = set(page_ids)
    neighbor_ids: set[str] = set()
    relevant_edges: list[dict[str, Any]] = []
    if ids:
        for batch in pq.ParquetFile(path / "edges.parquet").iter_batches(batch_size=8192):
            for row in batch.to_pylist():
                if row["source"] in ids or row["target"] in ids:
                    relevant_edges.append(row)
                    other = row["target"] if row["source"] in ids else row["source"]
                    if other not in ids and len(neighbor_ids) < neighbor_limit:
                        neighbor_ids.add(other)
    selected_by_id: dict[str, dict[str, Any]] = {}
    needed = ids | neighbor_ids
    if needed:
        for batch in node_file.iter_batches(batch_size=8192):
            for row in batch.to_pylist():
                if row["node_id"] in needed:
                    row["attrs"] = json.loads(row.pop("attrs_json"))
                    family, _ = _source_family(row)
                    node_source_scope(row, family, scope, require_stored=True)
                    row["neighbor_outside_window"] = (row["node_id"] in neighbor_ids and
                        row["tick"] is not None and not start_tick <= row["tick"] <= end_tick)
                    selected_by_id[row["node_id"]] = row
    selected = [selected_by_id[node_id] for node_id in page_ids]
    selected.extend(selected_by_id[node_id] for node_id in sorted(neighbor_ids))
    visible = {row["node_id"] for row in selected}
    edges = []
    for row in relevant_edges:
        if row["source"] in visible and row["target"] in visible:
            row["attrs"] = json.loads(row.pop("attrs_json"))
            edges.append(row)
    return {"scope": scope, "nodes": selected, "edges": edges,
            "page": {"offset": offset, "limit": limit, "matched_total": len(matching),
                     "has_more": offset + limit < len(matching)}}


def export_window_jsonl(path: Path, start_tick: int, end_tick: int, output: Path,
                        *, limit: int = 10000) -> None:
    data = read_window(path, start_tick, end_tick, limit=limit)
    output.mkdir(parents=True, exist_ok=True)
    (output / "scope.json").write_text(_json(data["scope"]) + "\n", encoding="utf-8")
    for name in ("nodes", "edges"):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in data[name]:
                stream.write(_json(row) + "\n")
