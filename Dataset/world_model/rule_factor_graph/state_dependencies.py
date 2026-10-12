"""Predicate-to-state dependency specifications for the executable contracts.

Every executable contract passes through one mechanism.  The contract supplies
its immediate inputs (typed fields, role bindings, owner roles, thresholds).
Upstream dependencies come from two source-checked layers:

* AST readers over the producer code: the predicate-to-function binding of a
  ``direct_predicate_row`` authority and the row fields each truth function
  reads, plus the ``context`` projection tables of the truth builder.
* Declarations (``_declarations()``) keyed by exact node identity.  A node identity is
  a producer function plus output (``path:function#output``), a ``record:``
  authority plus field path, or a ``context:`` field path.  Every declaration
  cites ``file:line`` locations that are resolved from the producer source at
  load time; a missing function or anchor raises ``ContractError``.

Node identities never contain predicate names. Canonical source families share
artifact/field identity; concrete cells additionally bind episode, tick and
the source key. Separate deterministic role selectors preserve each reading
occurrence. Whole truth frames and contained scalar/container views form one
coherent observation, never independent candidate samples. Node kinds mark candidates only:
``stochastic_candidate`` is an exogenous or upstream state a later model may
treat as random, ``deterministic`` is computed by a producer from its inputs,
``parameter`` is a threshold or configured constant, and ``unresolved`` has no
established producer.  No probability law is attached or implied here; laws
are estimated from data. A spec is ``complete`` only when its reachable nodes
have anchored declarations, established types/domains/units and producer
indexes, its field/role/schema projections and cache timing policies resolve,
and it has no unresolved items.

Every node also carries a ``ProducerIndex``: the producer loop or dictionary key
that stores the value, as a tuple of ``PRODUCER_INDEX_KEYS`` (``("actor",
"station")`` for a communication row, ``()`` for a profile constant).  Nodes
named ``record:`` or ``context:`` are indexed by the owner roles
of the contract field that reads them (``FieldBinding.owner_roles``, contract
role keys) and carry the ``contract_roles`` marker instead of producer keys.
An index the source does not establish is marked ``unestablished``.  For a
``direct_predicate_row`` contract each ``RoleSlot`` also exports the producer row
key it reads and the ordered arguments that row key composes, mapped to index keys.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from .unified_contracts import (
    ContractError, FIELD_OWNER_DECLARATIONS, FieldSchema, PredicateContract, TRUTH_DOMAIN, canonical_state_field,
    expression_fields, field_owner_roles, field_owner_source_refs, load_unified_contracts,
    walk_expression,
)

ROOT = Path(__file__).resolve().parents[3]
COMPUTE_COMM = "Dataset/semantic_simulation/compute_comm.py"
STATE_COMPUTERS = "Dataset/semantic_simulation/predicate_state_computers.py"
WORLD_TRUTH = "Dataset/semantic_truth/world_truth.py"
DOMAIN_STATE = "Dataset/semantic_simulation/domain_state.py"
CONTROL_STATE = "Dataset/semantic_simulation/control_response_state.py"
OBSERVABLE_STATE = "Dataset/semantic_simulation/observable_state_completion.py"
UTM_STATE = "Dataset/semantic_simulation/utm_state.py"
EXECUTABLE_COUNT = 72

DETERMINISTIC, STOCHASTIC_CANDIDATE, PARAMETER, UNRESOLVED = (
    "deterministic", "stochastic_candidate", "parameter", "unresolved")
NODE_KINDS = (DETERMINISTIC, STOCHASTIC_CANDIDATE, PARAMETER, UNRESOLVED)
COMPLETE, PARTIAL, DIRECT_ONLY = "complete", "partial", "direct_only"
RELATIONS = ("reads", "threshold", "derives", "guards", "derives_previous_tick")

INDEX_ESTABLISHED, INDEX_CONTRACT_ROLES, INDEX_UNESTABLISHED = "established", "contract_roles", "unestablished"
# Node ids whose index is the owner roles of the contract field that reads them.
CONTRACT_FIELD_PREFIXES = ("record:", "context:", "canonical:")
# Producer vocabulary of index keys.  These are not contract role keys: a contract
# role (``flow``, ``session``) is related to them only through ``RoleSlot.producer``.
PRODUCER_INDEX_KEYS = {
    "entity_id": "actual entity_id within one source artifact/episode/tick; used only for entity-based source-cell families",
    "actor": "entity (UAV, or vehicle in compute rows) of a compute_comm row, keyed by entity_id",
    "station": "communication station, keyed by the station entity_id",
    "tick": "episode tick alone: shared by every entity at that tick",
    "aircraft": "UAV of predicate_state_computers, keyed by uav_id",
    "corridor": "airspace corridor of predicate_state_computers, keyed by corridor_id",
    "entity": "world_truth scope source entity of any category, keyed by entity_id",
    "first_aircraft": "first aircraft of an ordered producer pair",
    "second_aircraft": "second aircraft of an ordered producer pair",
    "vehicle": "vehicle entity_id",
    "pedestrian": "pedestrian entity_id",
    "structure": "building structure entity_id",
    "region": "region entity_id; shared across aircraft referring to that region",
    "facility": "facility entity_id",
    "lane": "SUMO lane_id",
    "signal": "scene or SUMO controller signal_id",
    "cohort": "crowd cohort identity and its member set",
    "controller": "lockdown controller entity_id",
    "authorization": "authored authorization_id",
    "plan": "authored plan_id",
    "intent": "authored intent_id",
    "first_intent": "first authored intent_id of a pair",
    "second_intent": "second authored intent_id of a pair",
    "system": "authored UTM system_id",
}
# An empty key tuple is a global value: constant over the episode and shared by all entities.


# --------------------------------------------------------------------------
# Public records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StateDomain:
    """Value domain of a field.  ``referenced_values`` lists only the values
    the evaluation AST tests; the full code set is not declared.  Numeric and
    identifier domains stay ``unspecified``; ranges are never guessed."""
    kind: str
    values: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "values": list(self.values)}


@dataclass(frozen=True)
class ProducerIndex:
    """Where a producer stores a value, in ``PRODUCER_INDEX_KEYS`` vocabulary.

    ``established``: ``keys`` is the producer loop or dictionary key, ``()`` meaning
    global; ``provenance`` cites it.  ``contract_roles``: the node is a contract
    field and its index is ``FieldBinding.owner_roles`` (contract role keys),
which is not repeated here. The shared canonical region activity retains its
source region index. ``unestablished``: the source does not establish
    the index; ``reason`` says why."""
    status: str
    keys: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status == INDEX_ESTABLISHED:
            unknown = [key for key in self.keys if key not in PRODUCER_INDEX_KEYS]
            if unknown or len(set(self.keys)) != len(self.keys) or not self.provenance:
                raise ContractError(f"invalid established producer index {self.keys} "
                                    f"(unknown {unknown}, provenance {self.provenance})")
        elif self.status in (INDEX_CONTRACT_ROLES, INDEX_UNESTABLISHED):
            if self.keys or self.provenance or not self.reason:
                raise ContractError(f"a {self.status} producer index has no keys or provenance and needs a reason")
        else:
            raise ContractError(f"unknown producer index status {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "keys": list(self.keys),
                "provenance": list(self.provenance), "reason": self.reason}


CONTRACT_ROLES_INDEX = ProducerIndex(
    INDEX_CONTRACT_ROLES, reason="indexed by the owner roles (contract role keys) of the field that reads it")


@dataclass(frozen=True)
class CompositionArgument:
    """One argument of a composed producer row key."""
    expression: str  # source text of the argument
    row_key: str | None  # row key whose value is this local name
    index_keys: tuple[str, ...]  # producer index keys of that row key
    status: str  # "resolved" or "unresolved"
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "index_keys": list(self.index_keys)}


@dataclass(frozen=True)
class ProducerRoleBinding:
    """The producer row key a ``direct_predicate_row`` role reads.

    ``form``: ``row_key`` (the row key is one local name), ``stable_identifier``
    or ``f_string``.  ``constant_parts`` are the literal strings of the key
    (``stable_identifier``: its prefix) and ``excluded_arguments`` the non-entity
    arguments dropped from ``composition`` (``inputs.episode_id``).
    ``index_keys`` are the resolved arguments' producer index keys in order."""
    row_key: str
    row_builder: str
    form: str
    constant_parts: tuple[str, ...]
    excluded_arguments: tuple[str, ...]
    composition: tuple[CompositionArgument, ...]
    index_keys: tuple[str, ...]
    status: str  # "resolved" or "unresolved"
    unresolved: tuple[str, ...]
    provenance: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "constant_parts": list(self.constant_parts),
                "excluded_arguments": list(self.excluded_arguments),
                "composition": [item.to_dict() for item in self.composition],
                "index_keys": list(self.index_keys), "unresolved": list(self.unresolved),
                "provenance": list(self.provenance)}


@dataclass(frozen=True)
class RoleSlot:
    """``producer`` is set for ``direct_predicate_row`` contracts and ``None`` for
    the others, whose roles are not bound through a producer row key."""
    key: str
    ontology_class: str
    declared_class_uri: str
    source_kind: str
    source_path: str | None
    producer: ProducerRoleBinding | None

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "producer": None if self.producer is None else self.producer.to_dict()}


@dataclass(frozen=True)
class ProducerRoleProjection:
    """Exact identity bridge at a contract read, separate from time selection.

    ``index_identity`` equates a producer key with a grounded role identifier.
    ``referenced_identifier`` selects a role from an identifier-valued context
    cell; it does not equate that role with the context's entity index.
    ``manifest_identity`` attaches episode-wide observations to the manifest
    region without treating the tick as an entity identifier.
    """
    source_node: str
    index_key: str | None
    owner_role: str
    identifier_source: str
    form: str
    provenance: tuple[str, ...]
    selector: str = "same_identifier"

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "provenance": list(self.provenance)}


@dataclass(frozen=True)
class SchemaProjection:
    """Producer metadata retained alongside a compatible contract reading."""
    producer_data_type: str
    producer_unit: str | None
    contract_data_type: str
    contract_unit: str
    type_relation: str
    unit_relation: str
    provenance: tuple[str, ...]
    producer_domain: StateDomain
    contract_domain: StateDomain
    domain_relation: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "provenance": list(self.provenance),
                "producer_domain": self.producer_domain.to_dict(),
                "contract_domain": self.contract_domain.to_dict()}


@dataclass(frozen=True)
class FieldBinding:
    """One contract field: typed schema, shared node identity and owners."""
    field: str
    node_id: str
    data_type: str
    unit: str
    domain: StateDomain
    source: str
    schema_source: str
    owner_roles: tuple[str, ...]
    owner_status: str  # "resolved" or "unresolved"
    owner_error: str | None
    owner_provenance: tuple[str, ...]
    use: str  # "evaluated" (read by the AST) or "availability_guard" (unknown_if_missing only)
    producer_role_projections: tuple[ProducerRoleProjection, ...] = ()
    projection_status: str = "resolved"
    schema_projection: SchemaProjection | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "domain": self.domain.to_dict(),
                "owner_roles": list(self.owner_roles),
                "owner_provenance": list(self.owner_provenance),
                "producer_role_projections": [item.to_dict() for item in self.producer_role_projections],
                "schema_projection": None if self.schema_projection is None else self.schema_projection.to_dict()}


@dataclass(frozen=True)
class SourceCell:
    """A family template or coherent view, not one globally shared value.

    Concrete cells are bound by artifact, episode, tick and the declared source
    key. Role selection occurrences remain separate even when families match.
    """
    artifact: str
    field_path: str
    coordinate_keys: tuple[str, ...]
    form: str
    coherent_parent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "coordinate_keys": list(self.coordinate_keys)}


@dataclass(frozen=True)
class DependencyNode:
    node_id: str
    kind: str
    label: str
    unit: str | None
    provenance: tuple[str, ...]
    index: ProducerIndex
    unresolved: tuple[str, ...] = ()
    note: str = ""
    data_type: str = "unknown"
    domain: StateDomain = StateDomain("unspecified")
    schema_status: str = "unestablished"
    schema_source: tuple[str, ...] = ()
    unit_status: str = "unestablished"
    owner_status: str = "unestablished"
    owner_keys: tuple[str, ...] = ()
    source_cell: SourceCell | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "provenance": list(self.provenance),
                "index": self.index.to_dict(), "unresolved": list(self.unresolved),
                "domain": self.domain.to_dict(), "schema_source": list(self.schema_source),
                "owner_keys": list(self.owner_keys),
                "source_cell": None if self.source_cell is None else self.source_cell.to_dict()}


@dataclass(frozen=True)
class DependencyEdge:
    """``source`` is an input of ``target``; ``relation`` is one of RELATIONS."""
    source: str
    target: str
    relation: str
    provenance: tuple[str, ...]
    owner_roles: tuple[str, ...] = ()
    use: str = "value"  # value, guard; provenance-only fields are outside predicate closure
    index_bindings: tuple[tuple[str, str], ...] = ()
    collection_keys: tuple[str, ...] = ()
    selector: str = ""
    role_projections: tuple[ProducerRoleProjection, ...] = ()
    time_grid: str = ""
    temporal_lookup: str = ""
    update_policy: str = ""
    absence_policy: str = ""
    temporal_provenance: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.relation)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "provenance": list(self.provenance),
                "owner_roles": list(self.owner_roles),
                "index_bindings": [list(item) for item in self.index_bindings],
                "collection_keys": list(self.collection_keys),
                "role_projections": [item.to_dict() for item in self.role_projections],
                "temporal_provenance": list(self.temporal_provenance)}


@dataclass(frozen=True)
class UnresolvedItem:
    subject: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PredicateDependencySpec:
    predicate_id: str
    predicate_node_id: str
    semantic_mode: str
    operators: tuple[str, ...]
    evaluation: Mapping[str, Any]
    roles: tuple[RoleSlot, ...]
    fields: tuple[FieldBinding, ...]
    thresholds: tuple[str, ...]  # parameter node ids
    nodes: tuple[DependencyNode, ...]
    edges: tuple[DependencyEdge, ...]
    unresolved: tuple[UnresolvedItem, ...]
    completeness: str

    def node(self, node_id: str) -> DependencyNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise ContractError(f"{self.predicate_id} has no node {node_id}")

    def node_ids(self, kind: str | None = None) -> tuple[str, ...]:
        """Upstream node ids (the predicate node excluded), optionally one kind."""
        return tuple(node.node_id for node in self.nodes
                     if node.node_id != self.predicate_node_id and kind in (None, node.kind))

    @property
    def upstream_node_ids(self) -> frozenset[str]:
        return frozenset(self.node_ids())

    def upstream_closure(self, edge_uses: tuple[str, ...] = ("value", "guard")) -> frozenset[str]:
        """Reachable upstream states for selected edge uses, including carried state.

        ``("value",)`` excludes branches that only gate UNKNOWN/applicability.
        The default is the full predicate dependency closure.
        """
        if not edge_uses or not set(edge_uses) <= {"value", "guard"}:
            raise ContractError(f"invalid dependency edge uses {edge_uses}")
        reached, pending = set(), [self.predicate_node_id]
        while pending:
            target = pending.pop()
            for edge in self.edges:
                if edge.target == target and edge.use in edge_uses and edge.source not in reached:
                    reached.add(edge.source)
                    pending.append(edge.source)
        return frozenset(reached - {self.predicate_node_id})

    @property
    def stochastic_candidate_roots(self) -> tuple[str, ...]:
        return self.node_ids(STOCHASTIC_CANDIDATE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id, "predicate_node_id": self.predicate_node_id,
            "semantic_mode": self.semantic_mode, "operators": list(self.operators),
            "evaluation": _plain(self.evaluation),
            "roles": [item.to_dict() for item in self.roles],
            "fields": [item.to_dict() for item in self.fields],
            "thresholds": list(self.thresholds),
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "unresolved": [item.to_dict() for item in self.unresolved],
            "completeness": self.completeness,
        }


@dataclass(frozen=True)
class DependencyGraph:
    """Union of per-contract graphs; a shared node has one identity."""
    nodes: Mapping[str, DependencyNode]
    edges: tuple[DependencyEdge, ...]
    predicate_ids: tuple[str, ...]
    consumers: Mapping[str, tuple[str, ...]]  # upstream node id -> predicate ids

    def kind_counts(self) -> dict[str, int]:
        return {kind: sum(node.kind == kind for node in self.nodes.values()) for kind in NODE_KINDS}

    def shared_upstream_nodes(self, minimum_predicates: int = 2) -> dict[str, tuple[str, ...]]:
        return {node_id: ids for node_id, ids in self.consumers.items()
                if len(ids) >= minimum_predicates}

    def to_dict(self) -> dict[str, Any]:
        return {"predicate_ids": list(self.predicate_ids),
                "nodes": [self.nodes[key].to_dict() for key in sorted(self.nodes)],
                "edges": [edge.to_dict() for edge in self.edges],
                "consumers": {key: list(value) for key, value in sorted(self.consumers.items())}}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# --------------------------------------------------------------------------
# Producer source access
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _source(path: str) -> tuple[tuple[str, ...], ast.Module]:
    text = (ROOT / path).read_text(encoding="utf-8")
    return tuple(text.splitlines()), ast.parse(text, filename=path)


def _function(path: str, qualname: str) -> ast.FunctionDef:
    scope: list[ast.stmt] = _source(path)[1].body
    node: ast.stmt | None = None
    for part in qualname.split("."):
        node = next((item for item in scope if isinstance(item, (ast.FunctionDef, ast.ClassDef))
                     and item.name == part), None)
        if node is None:
            raise ContractError(f"{path} has no function {qualname}")
        scope = node.body
    if not isinstance(node, ast.FunctionDef):
        raise ContractError(f"{path}:{qualname} is not a function")
    return node


def _locate(path: str, function: str, anchor: str) -> str:
    """Return ``path:line`` of the first line of ``function`` containing ``anchor``."""
    lines, _ = _source(path)
    first, last = (1, len(lines)) if not function else (
        _function(path, function).lineno, _function(path, function).end_lineno)
    for number in range(first, last + 1):
        if anchor in lines[number - 1]:
            return f"{path}:{number}"
    raise ContractError(f"producer drift: {path}:{function or '<module>'} lacks {anchor!r}")


def _str_const(node: ast.AST, what: str) -> str:
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        raise ContractError(f"producer drift: {what} is not a string literal")
    return node.value


def _row_access(node: ast.AST, param: str) -> tuple[tuple[str, ...], ast.Name] | None:
    """Constant-key access chain rooted at ``param``: ``row["a"].get("b")``.

    A non-constant key truncates the path at its position."""
    keys: list[str | None] = []
    while True:
        if isinstance(node, ast.Subscript):
            keys.append(node.slice.value if isinstance(node.slice, ast.Constant)
                        and isinstance(node.slice.value, str) else None)
            node = node.value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "get" and node.args
              and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            keys.append(node.args[0].value)
            node = node.func.value
        else:
            break
    if not (isinstance(node, ast.Name) and node.id == param and keys):
        return None
    path: list[str] = []
    for key in reversed(keys):
        if key is None:
            break
        path.append(key)
    return (tuple(path), node) if path else None


@lru_cache(maxsize=None)
def _row_reads(path: str, name: str, param: str) -> tuple[dict[tuple[str, ...], int], tuple[tuple[str, int], ...]]:
    """Row paths read by ``name`` (path -> first line) and callees receiving the row."""
    function = _function(path, name)
    module_functions = {item.name for item in _source(path)[1].body if isinstance(item, ast.FunctionDef)}
    reads: dict[tuple[str, ...], int] = {}
    callees: list[tuple[str, int]] = []
    consumed: set[int] = set()

    def visit(node: ast.AST) -> None:
        access = _row_access(node, param)
        if access is not None:
            row_path, root = access
            consumed.add(id(root))
            if row_path not in reads:
                reads[row_path] = node.lineno
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for index, arg in enumerate(node.args):
                if isinstance(arg, ast.Name) and arg.id == param:
                    if node.func.id not in module_functions:
                        raise ContractError(f"{path}:{node.lineno} passes the row to {node.func.id}, "
                                            "which is not a module function; analysis is undefined")
                    consumed.add(id(arg))
                    if (node.func.id, index) not in callees:
                        callees.append((node.func.id, index))
        if (isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
                and any(isinstance(item, ast.Name) and item.id == param for item in node.comparators)):
            consumed.update(id(item) for item in node.comparators)  # presence guard
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in function.body:
        visit(statement)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id == param and id(node) not in consumed:
            raise ContractError(f"{path}:{node.lineno} uses {param} outside constant-key reads")
    return reads, tuple(callees)


@lru_cache(maxsize=None)
def _row_keys(path: str, builder: str) -> frozenset[str]:
    """Top-level keys the row builder emits through ``<row>.update({...})``."""
    keys: set[str] = set()
    for node in ast.walk(_function(path, builder)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update" and node.args and isinstance(node.args[0], ast.Dict)):
            keys.update(_str_const(key, f"{builder} row key") for key in node.args[0].keys)
    if not keys:
        raise ContractError(f"{path}:{builder} emits no row.update literal")
    return frozenset(keys)


def _string_set(path: str, function: str) -> tuple[str, ...]:
    """Field names of the set literal in a structured-state accessor."""
    for node in ast.walk(_function(path, function)):
        if isinstance(node, ast.Set):
            return tuple(sorted(_str_const(item, f"{function} field") for item in node.elts))
    raise ContractError(f"{path}:{function} has no field set literal")


# --------------------------------------------------------------------------
# Direct-predicate producer binding (AST over the producer's own wiring)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DirectProducer:
    module: str
    predicate_row_builder: str  # builds one predicate row per binding
    value_maps: tuple[str, ...]  # functions returning {predicate_id: truth(row)}
    artifact_anchor: str  # line proving the module writes the authority artifact
    artifact_function: str


# Keyed by the contract's candidate_authority source, never by predicate name.
DIRECT_PREDICATE_PRODUCERS = {
    "compute_comm_supplement/predicate_truth.jsonl": DirectProducer(
        COMPUTE_COMM, "_build_predicate_rows",
        ("_compute_predicate_values", "_communication_predicate_values"),
        'files["predicate_truth.jsonl"]', "build_episode_artifacts"),
}


@dataclass(frozen=True)
class RoleRowRead:
    """``{role: row["row_key"]}`` entry of a ``_predicate_row`` bindings dict."""
    role: str
    row_key: str
    line: int


@dataclass(frozen=True)
class DirectBinding:
    predicate_id: str
    function: str
    row_builder: str
    row_arg_index: int
    role_keys: tuple[str, ...]
    provenance: tuple[str, ...]
    role_row_reads: tuple[RoleRowRead, ...]


def communication_row_actor_category() -> str:
    """Producer category gate selecting the communication flow profile."""
    builder = _function(COMPUTE_COMM, "_build_communication_rows")
    categories = [node.test.comparators[0].value for node in ast.walk(builder)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "category"
        and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.NotEq)
        and isinstance(node.test.comparators[0], ast.Constant)
        and isinstance(node.test.comparators[0].value, str)
        and len(node.body) == 1 and isinstance(node.body[0], ast.Continue)]
    if len(categories) != 1:
        raise ContractError("communication producer has no single exact actor category gate")
    return categories[0]


def direct_monitor_fields(contract: PredicateContract) -> tuple[DirectBinding, tuple[DependencyNode, ...]]:
    """Immediate declared row operands of a receiver-grounded producer rule."""
    authority = contract.grounding_spec["candidate_authority"]
    if authority["kind"] != "direct_predicate_row":
        raise ContractError("a fixed producer rule requires direct row authority")
    binding = direct_predicate_bindings(authority["source"])[contract.predicate_id]
    spec = predicate_dependency_spec(contract)
    node_by_id = {node.node_id: node for node in spec.nodes}
    upstream: dict[str, list[str]] = {}
    for edge in spec.edges:
        upstream.setdefault(edge.target, []).append(edge.source)
    monitor_projections: dict[str, list[str]] = {}
    for node in spec.nodes:
        if (node.source_cell is not None and node.source_cell.form == "published_monitor"
                and node.source_cell.coherent_parent is not None):
            monitor_projections.setdefault(node.source_cell.coherent_parent, []).append(node.node_id)
    pending, visited, fields = [spec.predicate_node_id], set(), {}
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = node_by_id[node_id]
        if node.source_cell is not None and node.source_cell.form in (
                "published_monitor", "governed_row_parameter", "publisher_requirement",
                "unobserved_monitor"):
            fields[node.source_cell.field_path] = node
        elif node_id in monitor_projections:
            pending.extend(monitor_projections[node_id])
        else:
            if node_id != spec.predicate_node_id and not upstream.get(node_id):
                raise ContractError(f"direct rule operand has no immediate receiver authority: {node_id}")
            pending.extend(upstream.get(node_id, ()))
    if not fields or not any(node.source_cell.form in (
            "published_monitor", "unobserved_monitor") for node in fields.values()):
        raise ContractError("fixed producer rule requires declared monitor fields")
    return binding, tuple(fields[field] for field in sorted(fields))


def _call_name(node: ast.AST) -> str | None:
    return node.func.id if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else None


@lru_cache(maxsize=None)
def direct_predicate_bindings(authority: str) -> Mapping[str, DirectBinding]:
    """Predicate id -> producer truth function, parsed from the producer module."""
    if authority not in DIRECT_PREDICATE_PRODUCERS:
        raise ContractError(f"no producer is declared for direct authority {authority}")
    producer = DIRECT_PREDICATE_PRODUCERS[authority]
    module, tree = producer.module, _source(producer.module)[1]
    _locate(module, producer.artifact_function, producer.artifact_anchor)
    builder = _function(module, producer.predicate_row_builder)
    params = [arg.arg for arg in builder.args.args]
    row_signature = [arg.arg for arg in _function(module, "_predicate_row").args.args]
    index = {name: row_signature.index(name) for name in ("predicate_id", "bindings", "value")}

    # Row builder behind each iterated parameter: ``X = _build_*_rows(...)`` at
    # every call site of the predicate-row builder.
    row_builders: dict[str, str] = {}
    for caller in (item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef)):
        assigned = {target.id: _call_name(stmt.value) for stmt in ast.walk(caller)
                    if isinstance(stmt, ast.Assign) and _call_name(stmt.value)
                    for target in stmt.targets if isinstance(target, ast.Name)}
        for call in (item for item in ast.walk(caller) if _call_name(item) == producer.predicate_row_builder):
            for position, arg in enumerate(call.args):
                if isinstance(arg, ast.Name) and arg.id in assigned and position < len(params):
                    name = assigned[arg.id]
                    if row_builders.get(params[position], name) != name:
                        raise ContractError(f"{module}: call sites bind {params[position]} to different builders")
                    row_builders[params[position]] = name

    bindings: dict[str, DirectBinding] = {}
    for loop in (item for item in ast.walk(builder) if isinstance(item, ast.For)):
        if not (isinstance(loop.target, ast.Name) and isinstance(loop.iter, ast.Name)
                and loop.iter.id in params):
            continue
        if loop.iter.id not in row_builders:
            raise ContractError(f"{module}: no row builder is bound to {loop.iter.id}")
        for call in (item for item in ast.walk(loop) if _call_name(item) == "_predicate_row"):
            pid_node = call.args[index["predicate_id"]]
            predicate_id = _str_const(pid_node, "predicate id")
            roles = call.args[index["bindings"]]
            value = call.args[index["value"]]
            if not isinstance(roles, ast.Dict) or _call_name(value) is None:
                raise ContractError(f"{module}:{call.lineno} has an unparsable predicate row call")
            positions = [i for i, arg in enumerate(value.args)
                         if isinstance(arg, ast.Name) and arg.id == loop.target.id]
            if len(positions) != 1:
                raise ContractError(f"{module}:{value.lineno} must pass the row exactly once")
            reads = []
            for key, read in zip(roles.keys, roles.values):
                role = _str_const(key, "role key")
                if not (isinstance(read, ast.Subscript) and isinstance(read.value, ast.Name)
                        and read.value.id == loop.target.id):
                    raise ContractError(f"{module}:{read.lineno} binds role {role} to something other than "
                                        f"a {loop.target.id}[...] read")
                reads.append(RoleRowRead(role, _str_const(read.slice, f"{predicate_id}.{role} row key"),
                                         read.lineno))
            bindings[predicate_id] = DirectBinding(
                predicate_id, _call_name(value), row_builders[loop.iter.id], positions[0],
                tuple(sorted(item.role for item in reads)),
                (f"{module}:{pid_node.lineno}",), tuple(sorted(reads, key=lambda item: item.role)))

    # The matrix functions repeat the binding; both must agree exactly.
    declared: dict[str, str] = {}
    for name in producer.value_maps:
        function = _function(module, name)
        for node in ast.walk(function):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    predicate_id = _str_const(key, f"{name} key")
                    if predicate_id in declared:
                        raise ContractError(f"{module}: {predicate_id} is bound in two value maps")
                    declared[predicate_id] = _call_name(value) or ""
                    if predicate_id in bindings:
                        bindings[predicate_id] = replace(
                            bindings[predicate_id],
                            provenance=bindings[predicate_id].provenance + (f"{module}:{key.lineno}",))
    if {key: item.function for key, item in bindings.items()} != declared:
        raise ContractError(f"{module}: predicate-row binding disagrees with {producer.value_maps}")
    return bindings


# --------------------------------------------------------------------------
# Producer indexes: row keys, key composition and row identity
# --------------------------------------------------------------------------

def _index(keys: tuple[str, ...], *at: tuple[str, str, str]) -> ProducerIndex:
    """Established index; every anchor is resolved to ``file:line`` or raises."""
    return ProducerIndex(INDEX_ESTABLISHED, keys, tuple(dict.fromkeys(_locate(*item) for item in at)))


def _extend_index(index: ProducerIndex, *at: tuple[str, str, str]) -> ProducerIndex:
    return replace(index, provenance=tuple(dict.fromkeys(index.provenance + tuple(_locate(*item) for item in at))))


@dataclass(frozen=True)
class RowKeyIndex:
    index: ProducerIndex
    source_node: str | None  # declared node whose value the row key carries


@lru_cache(maxsize=1)
def row_key_indexes() -> Mapping[str, RowKeyIndex]:
    """Producer row keys that are declared as index keys, once, with anchors.

    ``entity_id`` identifies the UAV or vehicle the row is about; ``station_id``
    is the station chosen by ``_apply_handover_policy`` (``source_node``)."""
    ROW, COMPUTE = "_build_communication_rows", "_build_compute_rows"
    return {
        "entity_id": RowKeyIndex(_index(
            ("actor",), (COMPUTE_COMM, ROW, '"entity_id": entity_id,'),
            (COMPUTE_COMM, ROW, 'entity_id = _string_or_unknown(entity.get("entity_id"))'),
            (COMPUTE_COMM, COMPUTE, 'rows.sort(key=lambda row: (row["tick"], row["entity_id"], row["node_id"]))'),
            (COMPUTE_COMM, COMPUTE, '"node_id": f"compute_node:{entity_id}",')), None),
        "station_id": RowKeyIndex(_index(
            ("station",), (COMPUTE_COMM, ROW, '"station_id": station_id,'),
            (COMPUTE_COMM, ROW, 'station_id = station.get("entity_id") if station else "unknown"')),
            _cc("_apply_handover_policy", "station")),
        "tick": RowKeyIndex(_index(
            ("tick",), (COMPUTE_COMM, ROW, '"tick": tick,'), (COMPUTE_COMM, COMPUTE, '"tick": tick,'),
            (COMPUTE_COMM, ROW, 'tick = int(frame["tick"])')), None),
    }


@lru_cache(maxsize=None)
def _row_key_values(path: str, builder: str) -> Mapping[str, ast.expr]:
    """Value expression of each key of the builder's ``row.update({...})`` literal."""
    values: dict[str, ast.expr] = {}
    for node in ast.walk(_function(path, builder)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "row"
                and node.args and isinstance(node.args[0], ast.Dict)):
            for key, value in zip(node.args[0].keys, node.args[0].values):
                name = _str_const(key, f"{builder} row key")
                if name in values:
                    raise ContractError(f"{path}:{builder} sets row key {name} twice")
                values[name] = value
    if not values:
        raise ContractError(f"{path}:{builder} emits no row.update literal")
    return values


@dataclass(frozen=True)
class KeyComposition:
    row_key: str
    form: str
    constant_parts: tuple[str, ...]
    excluded_arguments: tuple[str, ...]
    arguments: tuple[ast.expr, ...]
    line: int


@lru_cache(maxsize=None)
def row_key_composition(path: str, builder: str, row_key: str) -> KeyComposition:
    """How the builder's ``row.update`` literal builds ``row_key``.

    Parsed forms: one local name, ``stable_identifier("prefix", inputs.episode_id,
    *arguments)`` and an f-string of literal text and formatted expressions.
    Anything else raises ``ContractError``."""
    values = _row_key_values(path, builder)
    if row_key not in values:
        raise ContractError(f"{path}:{builder} does not emit row key {row_key}")
    value = values[row_key]
    if isinstance(value, ast.Name):
        return KeyComposition(row_key, "row_key", (), (), (value,), value.lineno)
    if _call_name(value) == "stable_identifier":
        if value.keywords or len(value.args) < 2 or any(isinstance(arg, ast.Starred) for arg in value.args):
            raise ContractError(f"{path}:{value.lineno} has an unparsable stable_identifier call for {row_key}")
        prefix = _str_const(value.args[0], f"{row_key} identifier prefix")
        if ast.unparse(value.args[1]) != "inputs.episode_id":
            raise ContractError(f"{path}:{value.lineno} {row_key} does not compose inputs.episode_id second")
        parts, arguments = [prefix], []
        for arg in value.args[2:]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                parts.append(arg.value)
            else:
                arguments.append(arg)
        return KeyComposition(row_key, "stable_identifier", tuple(parts), ("inputs.episode_id",),
                              tuple(arguments), value.lineno)
    if isinstance(value, ast.JoinedStr):
        parts, arguments = [], []
        for item in value.values:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                parts.append(item.value)
            elif isinstance(item, ast.FormattedValue) and item.conversion == -1 and item.format_spec is None:
                arguments.append(item.value)
            else:
                raise ContractError(f"{path}:{value.lineno} has an unparsable f-string for {row_key}")
        return KeyComposition(row_key, "f_string", tuple(parts), (), tuple(arguments), value.lineno)
    raise ContractError(f"{path}:{value.lineno} builds {row_key} with an unparsable form "
                        f"{type(value).__name__}")


def _argument_binding(path: str, builder: str, argument: ast.expr) -> CompositionArgument:
    """Map one composition argument to the row key holding that local name."""
    text = ast.unparse(argument)
    if not isinstance(argument, ast.Name):
        return CompositionArgument(text, None, (), "unresolved", f"{text} is not a local name of a row key")
    holders = sorted(key for key, value in _row_key_values(path, builder).items()
                     if isinstance(value, ast.Name) and value.id == argument.id)
    if len(holders) != 1:
        return CompositionArgument(text, None, (), "unresolved",
                                   f"{text} is held by {len(holders)} row keys of {builder}: {holders}")
    declared = row_key_indexes().get(holders[0])
    if declared is None:
        return CompositionArgument(text, holders[0], (), "unresolved",
                                   f"row key {holders[0]} has no declared producer index")
    return CompositionArgument(text, holders[0], declared.index.keys, "resolved", "")


def role_producer_binding(path: str, builder: str, read: RoleRowRead) -> ProducerRoleBinding:
    """Producer row key, composed arguments and index keys behind one contract role."""
    composition = row_key_composition(path, builder, read.row_key)
    arguments = tuple(_argument_binding(path, builder, argument) for argument in composition.arguments)
    problems = tuple(f"{item.expression}: {item.reason}" for item in arguments if item.status == "unresolved")
    keys = tuple(dict.fromkeys(key for item in arguments if item.status == "resolved" for key in item.index_keys))
    return ProducerRoleBinding(
        read.row_key, builder, composition.form, composition.constant_parts, composition.excluded_arguments,
        arguments, keys, "unresolved" if problems or not arguments else "resolved",
        problems if arguments else (f"{read.row_key} composes no arguments",),
        (f"{path}:{read.line}", f"{path}:{composition.line}"))


@lru_cache(maxsize=None)
def builder_row_index(path: str, builder: str) -> ProducerIndex:
    """Index of one row of ``builder``: its ``rows.sort(key=lambda row: (...))`` key.

    The leading ``tick`` is the time axis and is stripped; every other part maps
    to producer index keys through ``row_key_indexes`` or through the arguments
    its composition resolves."""
    sorts = [node for node in ast.walk(_function(path, builder))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sort"
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "rows"]
    if len(sorts) != 1 or len(sorts[0].keywords) != 1 or sorts[0].keywords[0].arg != "key":
        raise ContractError(f"{path}:{builder} must sort rows once with a key lambda")
    key = sorts[0].keywords[0].value
    if not (isinstance(key, ast.Lambda) and isinstance(key.body, ast.Tuple) and len(key.args.args) == 1):
        raise ContractError(f"{path}:{sorts[0].lineno} sort key is not a lambda returning a tuple")
    param, keys, provenance = key.args.args[0].arg, [], [f"{path}:{sorts[0].lineno}"]
    parts = []
    for item in key.body.elts:
        if not (isinstance(item, ast.Subscript) and isinstance(item.value, ast.Name) and item.value.id == param):
            raise ContractError(f"{path}:{item.lineno} sort key part is not a {param}[...] read")
        parts.append(_str_const(item.slice, f"{builder} sort key"))
    if not parts or parts[0] != "tick":
        raise ContractError(f"{path}:{sorts[0].lineno} sort key must lead with tick")
    for part in parts[1:]:
        declared = row_key_indexes().get(part)
        if declared is not None:
            keys.extend(declared.index.keys)
            provenance.extend(declared.index.provenance)
            continue
        composition = row_key_composition(path, builder, part)
        for argument in composition.arguments:
            bound = _argument_binding(path, builder, argument)
            if bound.status != "resolved":
                raise ContractError(f"{path}:{builder} sort key part {part}: {bound.reason}")
            keys.extend(bound.index_keys)
        provenance.append(f"{path}:{composition.line}")
    return ProducerIndex(INDEX_ESTABLISHED, tuple(dict.fromkeys(keys)), tuple(dict.fromkeys(provenance)))


# --------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Decl:
    kind: str
    label: str
    inputs: tuple[str, ...]
    lagged_inputs: tuple[str, ...]
    provenance: tuple[str, ...]
    index: ProducerIndex
    unit: str | None = None
    unresolved: tuple[str, ...] = ()
    note: str = ""
    guards: tuple[str, ...] = ()
    data_type: str = "unknown"
    domain: StateDomain = StateDomain("unspecified")
    source_cell: SourceCell | None = None


class _Table:
    """Declaration table with eager validation of every cited location."""

    def __init__(self) -> None:
        self.items: dict[str, _Decl] = {}

    def add(self, node_id: str, kind: str, label: str, *at: tuple[str, str, str],
            index: ProducerIndex | None = None,
            inputs: Iterable[str] = (), lag: Iterable[str] = (), unit: str | None = None,
            unresolved: Iterable[str] = (), note: str = "", provenance: Iterable[str] = (),
            guards: Iterable[str] = (), data_type: str = "unknown",
            domain: StateDomain = StateDomain("unspecified")) -> str:
        if node_id in self.items:
            raise ContractError(f"duplicate dependency declaration: {node_id}")
        if index is None:
            raise ContractError(f"{node_id}: declaration lacks a producer index")
        located = tuple(provenance) + tuple(_locate(*item) for item in at)
        self.items[node_id] = _Decl(kind, label, tuple(dict.fromkeys(inputs)),
                                    tuple(dict.fromkeys(lag)), located, index, unit,
                                    tuple(unresolved), note, tuple(dict.fromkeys(guards)), data_type, domain)
        return node_id

    def root(self, node_id: str, label: str, *at: tuple[str, str, str], index: ProducerIndex | None = None,
             provenance: Iterable[str] = (), unit: str | None = None,
             data_type: str = "unknown", domain: StateDomain = StateDomain("unspecified"),
             note: str = "") -> str:
        """Exogenous state; repeated declarations add provenance to one node."""
        located = tuple(provenance) + tuple(_locate(*item) for item in at)
        existing = self.items.get(node_id)
        if existing is None:
            return self.add(node_id, STOCHASTIC_CANDIDATE, label, index=index, provenance=located,
                            unit=unit, data_type=data_type, domain=domain, note=note)
        if existing.kind != STOCHASTIC_CANDIDATE:
            raise ContractError(f"{node_id} is declared with two kinds")
        if index is None or (existing.index.status, existing.index.keys) != (index.status, index.keys):
            raise ContractError(f"{node_id} is declared with two producer indexes")
        if (unit is not None and existing.unit is not None and existing.unit != unit
                or data_type != "unknown" and existing.data_type != "unknown" and existing.data_type != data_type
                or domain.kind != "unspecified" and existing.domain.kind != "unspecified" and existing.domain != domain):
            raise ContractError(f"{node_id} is declared with conflicting root metadata")
        self.items[node_id] = replace(
            existing, provenance=existing.provenance + located,
            unit=unit if existing.unit is None else existing.unit,
            data_type=data_type if existing.data_type == "unknown" else existing.data_type,
            domain=domain if existing.domain.kind == "unspecified" else existing.domain,
            index=replace(existing.index, provenance=tuple(dict.fromkeys(existing.index.provenance + index.provenance))))
        return node_id

    def validate(self) -> None:
        for node_id, decl in self.items.items():
            if decl.kind not in NODE_KINDS:
                raise ContractError(f"{node_id}: unknown kind {decl.kind}")
            if not decl.provenance:
                raise ContractError(f"{node_id}: declaration lacks provenance")
            if (node_id.startswith(("record:", "context:")) != (decl.index.status == INDEX_CONTRACT_ROLES)
                    and not node_id.startswith("canonical:")):
                raise ContractError(f"{node_id}: only record:, context: and canonical: nodes take the "
                                    f"{INDEX_CONTRACT_ROLES} index")
            if decl.kind in (STOCHASTIC_CANDIDATE, PARAMETER) and (decl.inputs or decl.lagged_inputs or decl.guards):
                raise ContractError(f"{node_id}: a {decl.kind} root cannot have inputs")
            if decl.kind == DETERMINISTIC and not (decl.inputs or decl.lagged_inputs or decl.guards):
                raise ContractError(f"{node_id}: a deterministic node needs inputs")
            if decl.kind == UNRESOLVED and not decl.unresolved:
                raise ContractError(f"{node_id}: an unresolved node needs a reason")
            for source in decl.inputs + decl.lagged_inputs + decl.guards:
                if source not in self.items:
                    raise ContractError(f"{node_id}: input {source} has no declaration")
        state: dict[str, int] = {}

        def check(node_id: str) -> None:  # same-tick edges must be acyclic
            if state.get(node_id) == 2:
                return
            if state.get(node_id) == 1:
                raise ContractError(f"same-tick dependency cycle through {node_id}")
            state[node_id] = 1
            for source in self.items[node_id].inputs + self.items[node_id].guards:
                check(source)
            state[node_id] = 2

        for node_id in self.items:
            check(node_id)


def _cc(function: str, output: str) -> str:
    return f"{COMPUTE_COMM}:{function}#{output}"


def _profile(path: str) -> str:
    return f"param:compute_comm_profile#{path}"


# Context namespaces that world_truth copies verbatim from one L0 row.
IDENTITY_CONTEXT_NAMESPACES = {
    "weather": ("weather_meta.jsonl", (WORLD_TRUTH, "_build_tick_contexts", '"weather": dict(weather_row),'),
                ("tick",), (WORLD_TRUTH, "_evaluate_grounded_world_truth", "weather_row=weather[tick],")),
}  # namespace -> (L0 file, copy anchor, producer index keys of the L0 row, anchor proving them)


def _declare_clock(table: _Table) -> None:
    table.add("clock:tick", PARAMETER, "episode tick index of the row (time index, not a state)",
              (COMPUTE_COMM, "_build_communication_rows", 'tick = int(frame["tick"])'),
              (STATE_COMPUTERS, "GeometricStateComputer.compute", "for tick in FORMAL_TICKS:"),
              index=_index(("tick",), (COMPUTE_COMM, "_build_communication_rows", 'tick = int(frame["tick"])'),
                           (STATE_COMPUTERS, "GeometricStateComputer.compute", "for tick in FORMAL_TICKS:")))


def _declare_communication(table: _Table) -> None:
    """compute_comm.py: communication rows and the link-quality chain."""
    add, root = table.add, table.root
    ROW, LQ, WA = "_build_communication_rows", "_link_quality", "_weather_attenuation"
    SOS, SEL, AHP = "_station_operational_state", "_select_station", "_apply_handover_policy"
    row = lambda output: _cc(ROW, output)  # noqa: E731
    sos = lambda output: _cc(SOS, output)  # noqa: E731
    lq = lambda output: _cc(LQ, output)  # noqa: E731
    station_choice, distance = _cc(AHP, "station"), _cc(AHP, "distance_m")

    tick = "clock:tick"  # declared once by _declare_clock
    # Producer indexes.  ``row_index`` is the index of one communication row (its sort key).
    row_index = _extend_index(builder_row_index(COMPUTE_COMM, ROW), (COMPUTE_COMM, ROW, "for entity in frame_entities:"),
                              (COMPUTE_COMM, ROW, "rows.append(row)"))
    if row_index.keys != ("actor", "station"):
        raise ContractError(f"{COMPUTE_COMM}:{ROW} rows are indexed by {row_index.keys}, not (actor, station)")

    def at_row(*at: tuple[str, str, str]) -> ProducerIndex:
        return _extend_index(row_index, *at)

    profile_index = _index((), (COMPUTE_COMM, ROW, 'comm_profile = profile["communication"]'))
    actor_index = _index(("actor",), (COMPUTE_COMM, ROW, "for entity in frame_entities:"),
                         (COMPUTE_COMM, SEL, "entity_pos = _position(entity)"))
    station_index = _index(("station",), (COMPUTE_COMM, SEL, "for station in stations:"),
                           (COMPUTE_COMM, SEL, "station_pos = _position(station)"))
    tick_index = _index(("tick",), (COMPUTE_COMM, ROW, "for frame in inputs.frames:"),
                        (COMPUTE_COMM, ROW, "weather = inputs.weather_by_tick.get(tick)"))
    actor_state_index = _index(("actor",), (COMPUTE_COMM, SOS, "_structured_communication_input(actor),"))
    station_state_index = _index(
        ("station",), (COMPUTE_COMM, SOS, '("station", station_communication, _structured_security_input(station)),'))

    def carried(name: str, annotation: str, keys: tuple[str, ...], write: str) -> ProducerIndex:
        """State kept in a ``<name>: dict[...]`` that the row loop writes at ``write``."""
        return _index(keys, (COMPUTE_COMM, ROW, f"{name}: {annotation} = {{}}"), (COMPUTE_COMM, ROW, write))

    # Configured profile constants.
    flow = {name: add(_profile(f"communication.flow_profiles.uav.{name}"), PARAMETER,
                      f"flow profile {name}", (COMPUTE_COMM, ROW, 'if category != "uav":'),
                      (COMPUTE_COMM, ROW, anchor), index=profile_index)
            for name, anchor in (("latency_threshold_ms", '"latency_threshold_ms",'),
                                 ("packet_loss_threshold_ratio", '"packet_loss_threshold_ratio",'),
                                 ("bandwidth_mbps", 'flow_profile.get("bandwidth_mbps")'))}
    channel = {name: add(_profile(f"communication.channel.{name}"), PARAMETER, f"channel profile {name}",
                         (COMPUTE_COMM, where, anchor), index=profile_index)
               for name, where, anchor in (
                   ("nominal_latency_ms", LQ, 'channel_profile.get("nominal_latency_ms"'),
                   ("worst_case_latency_ms", LQ, 'channel_profile.get("worst_case_latency_ms"'),
                   ("maximum_packet_loss_ratio", LQ, 'channel_profile.get("maximum_packet_loss_ratio"'),
                   ("deterministic_retransmission_probability_floor", "_retransmission_count",
                    'channel_profile.get("deterministic_retransmission_probability_floor")'))}
    coefficient = {name: add(_profile(f"communication.channel.weather_attenuation.{name}"), PARAMETER,
                             f"weather attenuation coefficient {name}",
                             (COMPUTE_COMM, WA, f'_number(coeff.get("{name}"), 0.0)'), index=profile_index)
                   for name in ("rain", "fog_density", "dust")}
    nominal_range = add(_profile("communication.station_profiles.base_station.nominal_range_m"), PARAMETER,
                        "station nominal range", (COMPUTE_COMM, LQ, 'station_profile.get("nominal_range_m")'),
                        index=profile_index)
    nominal_capacity = add(_profile("communication.station_profiles.base_station.channel_bandwidth_mbps"),
                           PARAMETER, "station channel bandwidth",
                           (COMPUTE_COMM, ROW, 'station_profile.get("channel_bandwidth_mbps")'),
                           index=profile_index)
    backup_station = add(_profile("communication.station_profiles.base_station.backup_station_id"), PARAMETER,
                         "backup station id", (COMPUTE_COMM, ROW, 'backup_station_id = station_profile.get("backup_station_id")'),
                         index=profile_index)
    flow_entry = add(_profile("communication.flow_profiles.uav"), PARAMETER, "flow profile entry presence",
                     (COMPUTE_COMM, ROW, "flow_profile = flow_profiles.get(category)"), index=profile_index)
    station_entry = add(_profile("communication.station_profiles.base_station"), PARAMETER,
                        "station profile entry presence", (COMPUTE_COMM, ROW, "station_profile = base_station_profile"),
                        index=profile_index)
    margin = add(_profile("communication.handover_policy.handover_margin_m"), PARAMETER, "handover margin",
                 (COMPUTE_COMM, ROW, 'handover_policy.get("handover_margin_m")'), index=profile_index)
    dwell = add(_profile("communication.handover_policy.minimum_dwell_ticks"), PARAMETER, "minimum dwell",
                (COMPUTE_COMM, ROW, 'handover_policy.get("minimum_dwell_ticks")'), index=profile_index)
    operational = {key: add(_profile(f"communication.station_profiles.base_station.operational_states[mode].{key}"),
                            PARAMETER, f"station operational-state baseline {key} for the observed mode",
                            (COMPUTE_COMM, SOS, "value = _number(baseline_profile.get(key))"),
                            index=profile_index,
                            note="profile table keyed by the station mode; the mode selects the cell "
                                 "through the observed_mode edge")
                   for key in ("operational_factor", "capacity_factor", "range_factor", "availability")}

    # Exogenous L0 state (candidate roots).
    actor_position = root("l0:truth_frames.jsonl#actor.truth_pose.position_enu_m", "UAV position_enu_m",
                          (COMPUTE_COMM, SEL, "entity_pos = _position(entity)"),
                          (COMPUTE_COMM, "_position", 'pose.get("position_enu_m")'), index=actor_index)
    station_position = root("l0:truth_frames.jsonl#station.position_enu_m",
                            "station position (truth pose, else initial position; roster facility anchor "
                            "when no station is observed)",
                            (COMPUTE_COMM, "_position", 'entity.get("initial_position_enu_m")'),
                            (COMPUTE_COMM, "_simulated_station_from_facility_anchor",
                             '"initial_position_enu_m": _position(anchor),'), index=station_index)
    roster = root("l0:truth_frames.jsonl#station.roster", "station entities present in the frame",
                  (COMPUTE_COMM, ROW, "stations = _station_entities(frame_entities)"), index=tick_index)
    weather_index = _index(("tick",), (COMPUTE_COMM, ROW, "weather = inputs.weather_by_tick.get(tick)"))
    weather = {key: root(f"l0:weather_meta.jsonl#{key}", f"weather_meta.jsonl {key}",
                         (COMPUTE_COMM, "load_episode_inputs", 'read_jsonl(paths["weather"])'),
                         (COMPUTE_COMM, WA, f'weather.get("{key}")'), index=weather_index)
               for key in ("rain", "fog_density", "dust")}
    comm_fields = _string_set(COMPUTE_COMM, "_structured_communication_input")
    security_fields = _string_set(COMPUTE_COMM, "_structured_security_input")
    structured: dict[tuple[str, str, str], str] = {}
    for role, role_index in (("station", station_state_index), ("actor", actor_state_index)):
        for family, fields, accessor in (("communication_state", comm_fields, "_structured_communication_input"),
                                         ("security_state", security_fields, "_structured_security_input")):
            for field in fields:
                structured[(role, family, field)] = root(
                    f"l0:truth_frames.jsonl#{role}.{family}.{field}", f"{role} {family}.{field}",
                    (COMPUTE_COMM, accessor, f'"{field}",'), index=role_index)

    def state(family: str, fields: Iterable[str], roles: Iterable[str] = ("station", "actor")) -> list[str]:
        result = []
        for role in roles:
            for field in fields:
                if (role, family, field) not in structured:
                    raise ContractError(f"{COMPUTE_COMM}: {family} has no field {field}")
                result.append(structured[(role, family, field)])
        return result

    station_mode = state("communication_state", ("mode",), ("station",))[0]
    actor_handover = state("communication_state", ("handover_active",), ("actor",))[0]

    # Station selection and handover policy.
    select_index = _index(("actor",), (COMPUTE_COMM, ROW, "candidate_station, candidate_distance_m = _select_station(entity, stations)"))
    policy_index = _index(("actor",), (COMPUTE_COMM, ROW, "station, distance_m, handover_evidence = _apply_handover_policy("),
                          (COMPUTE_COMM, ROW, "previous_station_id=previous_station_by_entity.get(entity_id),"))
    candidate = add(_cc(SEL, "candidate_station"), DETERMINISTIC, "nearest station",
                    (COMPUTE_COMM, SEL, "if best_distance is None or distance < best_distance:"),
                    inputs=(actor_position, station_position, roster), index=select_index)
    candidate_distance = add(_cc(SEL, "candidate_distance_m"), DETERMINISTIC, "distance to the nearest station",
                             (COMPUTE_COMM, SEL, "distance = math.dist(entity_pos[:3], station_pos[:3])"),
                             inputs=(actor_position, station_position, roster), unit="m", index=select_index)
    last_switch_id, logical_id = row("last_switch_tick"), row("logical_handover")
    previous_station = add(row("previous_station_id"), DETERMINISTIC,
                           "last known station selected for the actor (carried state)",
                           (COMPUTE_COMM, ROW, "previous_station_by_entity[entity_id] = str(station_id)"),
                           lag=(station_choice, row("previous_station_id")),
                           index=carried("previous_station_by_entity", "dict[str, str]", ("actor",),
                                         "previous_station_by_entity[entity_id] = str(station_id)"))
    last_switch = add(last_switch_id, DETERMINISTIC, "tick of the last station switch (carried state)",
                      (COMPUTE_COMM, ROW, "last_switch_tick_by_entity[entity_id] = tick"),
                      lag=(station_choice, previous_station, tick, last_switch_id), unit="tick",
                      index=carried("last_switch_tick_by_entity", "dict[str, int]", ("actor",),
                                    "last_switch_tick_by_entity[entity_id] = tick"))
    switch_allowed = add(_cc(AHP, "switch_allowed"), DETERMINISTIC, "handover policy allows a station switch",
                         (COMPUTE_COMM, AHP, "switch_allowed = ("),
                         inputs=(candidate, candidate_distance, previous_station, last_switch, tick, margin, dwell,
                                 actor_position, station_position, roster), unit="bool", index=policy_index)
    add(station_choice, DETERMINISTIC, "station selected after the handover policy",
        (COMPUTE_COMM, AHP, "selected_station = candidate_station if switch_allowed else current_station"),
        inputs=(candidate, switch_allowed, previous_station, roster), index=policy_index,
        note="one selected station per actor and tick; the station is the value, the actor is the index")
    add(distance, DETERMINISTIC, "distance to the selected station",
        (COMPUTE_COMM, AHP, "selected_distance = candidate_distance_m if switch_allowed else current_distance_m"),
        inputs=(candidate_distance, switch_allowed, previous_station, actor_position, station_position),
        unit="m", index=at_row((COMPUTE_COMM, ROW, "station, distance_m, handover_evidence = _apply_handover_policy(")))

    # Station operational state.
    sos_index = at_row((COMPUTE_COMM, ROW, "station_operational_state = _station_operational_state("))
    observed_mode = add(sos("observed_mode"), DETERMINISTIC, "observed station mode",
                        (COMPUTE_COMM, SOS, 'observed_mode = station_communication.get("mode", "online")'),
                        inputs=(station_mode, station_choice), index=sos_index)
    unavailable = add(sos("unavailable"), DETERMINISTIC, "structured state marks the link unavailable",
                      (COMPUTE_COMM, SOS, 'for field in ("station_unavailable", "communication_unavailable"):'),
                      (COMPUTE_COMM, SOS, "if availability_value <= 0.0:"),
                      (COMPUTE_COMM, SOS, 'if _structured_boolean(security_state, "gcs_compromised") is True:'),
                      inputs=state("communication_state", ("station_unavailable", "communication_unavailable",
                                                          "available", "link_available", "availability"))
                      + state("security_state", ("gcs_compromised",)), unit="bool", index=sos_index)
    degraded = add(sos("degraded"), DETERMINISTIC, "structured state marks the link degraded",
                   (COMPUTE_COMM, SOS, "elif availability_value < 1.0:"),
                   (COMPUTE_COMM, SOS, 'for field in ("jamming", "jamming_active"):'),
                   inputs=state("communication_state", ("availability",))
                   + state("security_state", ("jamming", "jamming_active")), unit="bool", index=sos_index)
    explicit_availability = add(sos("explicit_availability"), DETERMINISTIC, "minimum explicit availability",
                                (COMPUTE_COMM, SOS, "explicit_availability_values.append(availability_value)"),
                                inputs=state("communication_state", ("availability",)), unit="ratio",
                                index=sos_index)
    has_structured = add(sos("has_structured_state"), DETERMINISTIC, "any structured state present",
                         (COMPUTE_COMM, SOS, "if communication_state or security_state:"),
                         inputs=state("communication_state", comm_fields) + state("security_state", security_fields),
                         unit="bool", index=sos_index)
    sos_missing = add(sos("missing_parameters"), DETERMINISTIC, "missing station-state parameters",
                      (COMPUTE_COMM, SOS, 'missing_parameters.append("truth_frames.entities.station")'),
                      (COMPUTE_COMM, SOS, "selected = operational_states.get(observed_mode)"),
                      inputs=(station_choice, observed_mode, *operational.values()), index=sos_index)
    caps = {"operational_factor": '0.22', "capacity_factor": '0.6', "range_factor": '0.75', "availability": '0.85'}
    for key, cap in caps.items():
        add(sos(key), DETERMINISTIC, f"station {key}",
            (COMPUTE_COMM, SOS, f'values["{key}"] = min(values["{key}"], {cap})'),
            (COMPUTE_COMM, SOS, "values[key] = max(0.0, min(1.0, value))"),
            inputs=(unavailable, degraded, observed_mode, operational[key],
                    *((explicit_availability,) if key == "availability" else ())), unit="ratio", index=sos_index)
        table.items[sos(key)] = replace(table.items[sos(key)], guards=(sos_missing,),
            note="Structured presence selects descriptive state_mode/state_basis only; numeric overrides "
                 "already depend on unavailable, degraded and explicit availability.")

    # Link quality.
    lq_index = at_row((COMPUTE_COMM, ROW, "quality = _link_quality("))
    attenuation = add(_cc(WA, "weather_attenuation"), DETERMINISTIC, "weather attenuation factor",
                      (COMPUTE_COMM, WA, "return max(0.05, 1.0 - rain * rain_coeff"),
                      inputs=(*weather.values(), *coefficient.values()), unit="ratio", index=lq_index)
    effective_range = add(lq("effective_range_m"), DETERMINISTIC, "effective range",
                          (COMPUTE_COMM, LQ, "effective_range = nominal_range * max(0.0, range_factor)"),
                          inputs=(nominal_range, sos("range_factor")), unit="m", index=lq_index)
    range_score = add(lq("range_score"), DETERMINISTIC, "range score (distance / effective range)",
                      (COMPUTE_COMM, LQ, "range_score = max(0.0, min(1.0, 1.0 - distance_m / effective_range))"),
                      inputs=(distance, effective_range, sos("operational_factor"), sos("availability")),
                      unit="ratio", index=lq_index)
    score = add(lq("quality_score"), DETERMINISTIC, "link quality score",
                (COMPUTE_COMM, LQ, "range_score * weather_attenuation * operational_factor * availability,"),
                (COMPUTE_COMM, LQ, "or capacity_factor is None"),
                inputs=(range_score, attenuation, sos("operational_factor"), sos("availability")),
                guards=(sos("capacity_factor"), distance, *weather.values()), unit="ratio", index=lq_index,
                note="the unknown-guard also gates on capacity_factor, distance and weather presence")
    level = add(lq("quality_level"), DETERMINISTIC, "link quality level",
                (COMPUTE_COMM, LQ, "if score >= 0.75:"), inputs=(score,), unit="state-code", index=lq_index)
    latency = add(lq("latency_ms"), DETERMINISTIC, "link latency",
                  (COMPUTE_COMM, LQ, "latency_ms = nominal_latency + (1.0 - score)"),
                  inputs=(score, channel["nominal_latency_ms"], channel["worst_case_latency_ms"]), unit="ms",
                  index=lq_index)
    loss = add(lq("packet_loss_ratio"), DETERMINISTIC, "link packet loss",
               (COMPUTE_COMM, LQ, "packet_loss_ratio = max(0.0, min(1.0, (1.0 - score) ** 2"),
               inputs=(score, channel["maximum_packet_loss_ratio"]), unit="ratio", index=lq_index)

    # Declared handover chain.
    declared_index = carried("previous_declared_handover_by_entity", "dict[str, bool | None]", ("actor",),
                             "previous_declared_handover_by_entity[entity_id] = (")
    declared = add(row("declared_handover_active"), DETERMINISTIC, "declared handover flag of the actor",
                   (COMPUTE_COMM, ROW, "declared_handover_active = _structured_boolean("), inputs=(actor_handover,),
                   unit="bool", index=declared_index)
    rising = add(row("declared_handover_rising"), DETERMINISTIC, "declared handover flag rising edge",
                 (COMPUTE_COMM, ROW, "declared_handover_rising = ("), inputs=(declared,), lag=(declared,), unit="bool",
                 index=declared_index)
    entered = add(row("station_entered_backup_link"), DETERMINISTIC, "station entered backup link",
                  (COMPUTE_COMM, ROW, "station_entered_backup_link = {"), inputs=(station_mode,),
                  lag=(station_mode,), unit="bool",
                  index=carried("previous_station_mode_by_id", "dict[str, str]", ("station",),
                                "and previous_station_mode_by_id.get(station_id) not in {"))
    started = add(row("logical_handover_started"), DETERMINISTIC, "logical handover starts",
                  (COMPUTE_COMM, ROW, "logical_handover_started = ("),
                  inputs=(station_choice, entered, rising), unit="bool", index=row_index)
    logical = add(logical_id, DETERMINISTIC, "logical handover pair (carried state)",
                  (COMPUTE_COMM, ROW, "logical_handover_by_entity[entity_id] = ("),
                  (COMPUTE_COMM, ROW, "or backup_station_id not in station_ids"),
                  inputs=(started, declared, backup_station, station_choice, roster), lag=(logical_id,),
                  index=carried("logical_handover_by_entity", "dict[str, tuple[str, str]]", ("actor",),
                                "logical_handover_by_entity[entity_id] = ("))
    logical_active = add(row("logical_handover_active"), DETERMINISTIC, "logical handover active",
                         (COMPUTE_COMM, ROW, "logical_handover_active = ("), inputs=(declared, logical), unit="bool",
                         index=row_index)
    geometric = add(row("geometric_handover_active"), DETERMINISTIC, "station switch handover active",
                    (COMPUTE_COMM, ROW, "geometric_handover_active = ("),
                    inputs=(station_choice, previous_station), unit="bool", index=row_index)
    handover = add(row("handover_active"), DETERMINISTIC, "handover active before the requirement gate",
                   (COMPUTE_COMM, ROW, "handover_active = geometric_handover_active or logical_handover_active"),
                   inputs=(geometric, logical_active), unit="bool", index=row_index)

    # Requirement gate.
    missing = add(row("missing_parameters"), DETERMINISTIC, "missing communication inputs",
                  (COMPUTE_COMM, ROW, "missing_parameters: list[str] = []"),
                  (COMPUTE_COMM, ROW, 'missing_parameters.append("inputs.weather.tick")'),
                  (COMPUTE_COMM, ROW, "if _number(channel_profile.get(parameter_name)) is None:"),
                  (COMPUTE_COMM, ROW, 'missing_parameters.append("communication.current_station_distance")'),
                  (COMPUTE_COMM, ROW, "station_operational_state.pop(\"missing_parameters\")"),
                  (COMPUTE_COMM, ROW, '"communication.station_profiles.base_station.backup_station_id"'),
                  inputs=(*weather.values(), *flow.values(), channel["nominal_latency_ms"],
                          channel["worst_case_latency_ms"], channel["maximum_packet_loss_ratio"],
                          nominal_capacity, flow_entry, station_entry, roster, distance, station_choice,
                          sos_missing, started, backup_station), unit="nonmeasurement", index=row_index,
                  data_type="array", domain=StateDomain("source_field_paths"))

    # Row fields read by the truth functions.
    add(row("requirement.status"), DETERMINISTIC, "requirement status",
        (COMPUTE_COMM, ROW, '"status": "source_required"'), inputs=(missing,), unit="state-code", index=row_index,
        data_type="categorical", domain=StateDomain("enumeration", ("source_required", "not_required")))
    for name, node in (("quality_level", level), ("latency_ms", latency), ("packet_loss_ratio", loss)):
        add(row(f"link_quality.{name}"), DETERMINISTIC, f"row link_quality.{name}",
            (COMPUTE_COMM, ROW, '"link_quality": quality,'), (COMPUTE_COMM, ROW, "quality = _link_quality("),
            inputs=(node,), index=row_index)
    add(row("quality_thresholds.latency_ms"), DETERMINISTIC, "row latency threshold",
        (COMPUTE_COMM, ROW, '_number(flow_profile.get("latency_threshold_ms"))'),
        inputs=(flow["latency_threshold_ms"],), unit="ms", index=row_index)
    add(row("quality_thresholds.packet_loss_ratio"), DETERMINISTIC, "row packet-loss threshold",
        (COMPUTE_COMM, ROW, '_number(flow_profile.get("packet_loss_threshold_ratio"))'),
        inputs=(flow["packet_loss_threshold_ratio"],), unit="ratio", index=row_index)
    # These immediate row views have exact receiver/profile adapters. Their
    # producer derivation remains metadata, rather than an observation demand.
    for field, form in (
        ("link_quality.latency_ms", "published_monitor"),
        ("link_quality.packet_loss_ratio", "published_monitor"),
        ("quality_thresholds.latency_ms", "governed_row_parameter"),
        ("quality_thresholds.packet_loss_ratio", "governed_row_parameter"),
        ("requirement.status", "publisher_requirement"),
    ):
        node_id = row(field)
        table.items[node_id] = replace(table.items[node_id], source_cell=SourceCell(
            "communication_state.jsonl", field, ("episode_id", "tick", *row_index.keys), form))
    add(row("handover.active"), DETERMINISTIC, "row handover.active",
        (COMPUTE_COMM, ROW, '"active": handover_active if not source_required else "unknown",'),
        inputs=(handover, missing), unit="bool", index=row_index)
    capacity = add(row("bandwidth_capacity"), DETERMINISTIC, "bandwidth capacity",
                   (COMPUTE_COMM, ROW, "else max(0.0, nominal_bandwidth_capacity * capacity_factor)"),
                   inputs=(nominal_capacity, sos("capacity_factor")), unit="Mbps", index=row_index)
    allocated = add(row("allocated_bandwidth"), DETERMINISTIC, "allocated bandwidth",
                    (COMPUTE_COMM, ROW, "else min(bandwidth_request, bandwidth_capacity)"),
                    inputs=(flow["bandwidth_mbps"], capacity), unit="Mbps", index=row_index)
    queued = add(row("queued_bandwidth"), DETERMINISTIC, "queued (dropped) bandwidth",
                 (COMPUTE_COMM, ROW, "else max(0.0, bandwidth_request - allocated_bandwidth)"),
                 inputs=(flow["bandwidth_mbps"], allocated), unit="Mbps", index=row_index)
    add(row("bandwidth.dropped_mbps"), DETERMINISTIC, "row bandwidth.dropped_mbps",
        (COMPUTE_COMM, ROW, '"dropped_mbps": _round(queued_bandwidth),'), inputs=(queued,), unit="Mbps",
        index=row_index)
    count = add(_cc("_retransmission_count", "count"), DETERMINISTIC, "retransmission count",
                (COMPUTE_COMM, "_retransmission_count", "return max(base_count, 1 if sample < probability else 0)"),
                inputs=(level, score, latency, loss, distance, tick,
                        channel["deterministic_retransmission_probability_floor"]), unit="count",
                index=at_row((COMPUTE_COMM, ROW, "retransmission_count = _retransmission_count(")),
                note="the sample is a deterministic digest of the listed inputs (_unit_interval), "
                     "not a random draw")
    add(row("retransmission.count"), DETERMINISTIC, "row retransmission.count",
        (COMPUTE_COMM, ROW, '"count": retransmission_count'), inputs=(count, missing), unit="count",
        index=row_index)
    for field in ("bandwidth.dropped_mbps", "retransmission.count", "handover.active"):
        node_id = row(field)
        table.items[node_id] = replace(table.items[node_id], source_cell=SourceCell(
            "communication_state.jsonl", field, ("episode_id", "tick", *row_index.keys),
            "published_monitor"))


def _declare_aircraft_geometry(table: _Table) -> None:
    """predicate_state_computers.py: aircraft geometry and corridor membership."""
    add, root = table.add, table.root
    GC = "GeometricStateComputer.compute"
    geometry = lambda output: f"{STATE_COMPUTERS}:{GC}#aircraft_geometry.{output}"  # noqa: E731
    aircraft = _index(("aircraft",), (STATE_COMPUTERS, GC, "for uav_id in present_uavs:"),
                      (STATE_COMPUTERS, GC, "position = tick_positions[uav_id]"),
                      (STATE_COMPUTERS, GC, '"predicate_contract_aircraft_geometry",'))
    trajectory = _extend_index(aircraft, (STATE_COMPUTERS, "load_trajectory_vectors",
                                          "positions: dict[int, dict[str, tuple[float, float, float]]] = {}"))
    position = root("l0:trajectories.jsonl#aircraft.position_enu_m", "aircraft position_enu_m",
                    (STATE_COMPUTERS, GC, "load_trajectory_vectors("), index=trajectory)
    velocity = root("l0:trajectories.jsonl#aircraft.velocity_enu_mps", "aircraft velocity_enu_mps",
                    (STATE_COMPUTERS, GC, "load_trajectory_vectors("), index=trajectory)
    add(geometry("speed_mps"), DETERMINISTIC, "aircraft speed",
        (STATE_COMPUTERS, GC, '"speed_mps": math.dist((0.0, 0.0, 0.0), velocity)'), inputs=(velocity,), unit="m/s",
        index=aircraft)
    add(geometry("position_enu_m"), DETERMINISTIC, "aircraft position",
        (STATE_COMPUTERS, GC, '"position_enu_m": list(position),'), inputs=(position,), unit="m", index=aircraft)
    ground = add(geometry("local_ground_reference_z_m"), DETERMINISTIC, "local ground reference height",
                 (STATE_COMPUTERS, GC, "local_ground_z = aircraft_ground.local_ground_reference_by_tick[tick]["),
                 inputs=(position,), unit="m",
                 index=_extend_index(aircraft, (STATE_COMPUTERS, "resolve_aircraft_ground_references",
                                                "local_ground_reference_by_tick: dict[int, dict[str, float]] = {}")),
                 unresolved=("pad contact references, pad poses and the global ground reference feeding "
                             "_local_aircraft_ground_reference_z are not decomposed",))
    add(geometry("z_agl_m"), DETERMINISTIC, "height above local ground",
        (STATE_COMPUTERS, GC, "position[2] - local_ground_z"), inputs=(position, ground), unit="m", index=aircraft)
    roster_index = _index(("aircraft",), (STATE_COMPUTERS, GC, "roster = roster_by_id[uav_id]"))
    altitude_sources = tuple(
        add(f"param:global_entity_roster.json#uav.{name}", PARAMETER, f"static roster attribute {name}",
            (STATE_COMPUTERS, GC, anchor), index=roster_index)
        for name, anchor in (
            ("assigned_altitude_m", 'roster.get("assigned_altitude_m")'),
            ("uav_corridor.assigned_altitude_m", 'roster["uav_corridor"].get("assigned_altitude_m")'),
            ("uav_global_flow.altitude_layer_m", 'roster["uav_global_flow"].get("altitude_layer_m")'),
            ("motion_contract.altitude_layer_m", 'roster["motion_contract"].get("altitude_layer_m")')))
    add(geometry("assigned_altitude_m"), DETERMINISTIC, "assigned altitude (first declared roster source)",
        (STATE_COMPUTERS, GC, "assigned_altitude = _number(roster.get(\"assigned_altitude_m\"))"),
        inputs=altitude_sources, unit="m", index=aircraft)
    for name in ("speed_mps", "z_agl_m", "local_ground_reference_z_m", "assigned_altitude_m"):
        add(f"context:geometry.{name}", DETERMINISTIC, f"context geometry.{name}",
            (WORLD_TRUTH, "_build_tick_contexts", 'context["geometry"].update(_values(aircraft_geometry))'),
            inputs=(geometry(name),), index=CONTRACT_ROLES_INDEX)
    add("context:geometry.position_z_m", DETERMINISTIC, "context geometry.position_z_m",
        (WORLD_TRUTH, "_build_tick_contexts", 'context["geometry"]["position_z_m"] = position[2]'),
        inputs=(geometry("position_enu_m"),), unit="m", index=CONTRACT_ROLES_INDEX)

    scene = add("param:scene_setup.json#entities", PARAMETER, "static scene entities",
                (STATE_COMPUTERS, GC, "scene_by_id = _index_entities(scene.get(\"entities\"), scene_path)"),
                index=_index((), (STATE_COMPUTERS, GC, "scene_by_id = _index_entities(scene.get(\"entities\"), scene_path)")))
    margin = add("param:governed_parameter_defaults#separation_margin_m", PARAMETER, "governed separation margin",
                 (STATE_COMPUTERS, GC, 'governed_parameters["separation_margin_m"]'),
                 index=_index((), (STATE_COMPUTERS, GC, 'separation_margin_m = float(governed_parameters["separation_margin_m"])')))
    tick = "clock:tick"  # declared once by _declare_clock
    corridors = add(f"{STATE_COMPUTERS}:{GC}#corridors", DETERMINISTIC, "corridor geometry",
                    (STATE_COMPUTERS, GC, "corridors = _airspace_corridors(scene_by_id, separation_margin_m)"),
                    inputs=(scene, margin),
                    index=_index(("corridor",), (STATE_COMPUTERS, GC, "for corridor_id, corridor in corridors.items()"),
                                 (STATE_COMPUTERS, "_airspace_corridors", "result[entity_id] = {")))
    prefix = "record:predicate_contract_aircraft_corridor_geometry#values."
    add(prefix + "aircraft_inside_corridor", DETERMINISTIC, "aircraft inside corridor",
        (STATE_COMPUTERS, GC, "inside_corridor = uav_id in occupants_by_corridor[corridor_id]"),
        (STATE_COMPUTERS, GC, "if _point_in_oriented_corridor_box(tick_positions[uav_id], corridor)"),
        (STATE_COMPUTERS, GC, "if _corridor_active_at_tick(corridor, tick)"),
        inputs=(position, corridors, tick), unit="bool", index=CONTRACT_ROLES_INDEX)
    add(prefix + "speed_mps", DETERMINISTIC, "corridor row speed",
        (STATE_COMPUTERS, GC, '"speed_mps": values["speed_mps"],'), inputs=(geometry("speed_mps"),), unit="m/s",
        index=CONTRACT_ROLES_INDEX)


def _declare_truth_projection(table: _Table) -> None:
    """world_truth.py: the ``truth`` context namespace, read from its projection tables."""
    projection = _function(WORLD_TRUTH, "_truth_projection")
    families: dict[str, str] = {}  # local container name -> entity family key
    for node in ast.walk(projection):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "get" and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "entity" and node.value.args):
            families[node.targets[0].id] = _str_const(node.value.args[0], "entity family")

    def copied(stmt: ast.AST) -> tuple[str, str] | None:  # ``result[...] = container[...]``
        if (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Subscript)
                and isinstance(stmt.value.value, ast.Name) and stmt.value.value.id in families):
            return families[stmt.value.value.id], stmt.value.value.id
        return None

    sources: dict[str, list[tuple[str, str, int]]] = {}  # target -> [(family, source key, line)]

    def record(target: str, family: str, source: str, line: int) -> None:
        sources[target] = sources.get(target, []) + [(family, source, line)]

    for loop in (item for item in ast.walk(projection) if isinstance(item, ast.For)):
        copy = next((stmt for stmt in ast.walk(loop) if copied(stmt)), None)
        if copy is None or not isinstance(loop.iter, ast.Tuple):
            continue
        for item in loop.iter.elts:
            source, target = ((_str_const(item.elts[0], "source key"), _str_const(item.elts[1], "target key"))
                              if isinstance(item, ast.Tuple) else
                              (_str_const(item, "field"), _str_const(item, "field")))
            record(target, copied(copy)[0], source, copy.lineno)
    for stmt in ast.walk(projection):  # literal fallbacks: result["k"] = container["k2"]
        if copied(stmt) and isinstance(stmt.targets[0], ast.Subscript) \
                and isinstance(stmt.targets[0].slice, ast.Constant):
            record(_str_const(stmt.targets[0].slice, "target key"), copied(stmt)[0],
                   _str_const(stmt.value.slice, "source key"), stmt.lineno)
    entity_index = _index(("entity",), (WORLD_TRUTH, "_build_tick_contexts", "entity = entities_at_tick.get(source_id)"),
                          (WORLD_TRUTH, "_build_tick_contexts", '"truth": _truth_projection(entity),'))
    for target, items in sorted(sources.items()):
        roots = tuple(table.root(f"l0:truth_frames.jsonl#entity.{family}.{source}",
                                 f"truth_frames entity {family}.{source}", provenance=(f"{WORLD_TRUTH}:{line}",),
                                 index=entity_index)
                      for family, source, line in items)
        table.add(f"context:truth.{target}", DETERMINISTIC, f"context truth.{target}",
                  provenance=tuple(f"{WORLD_TRUTH}:{line}" for _, _, line in items), inputs=roots,
                  index=CONTRACT_ROLES_INDEX)
    speed_inputs = tuple(
        table.root(f"l0:truth_frames.jsonl#entity.{name}", f"truth_frames entity {name}",
                   (WORLD_TRUTH, "_speed", anchor), index=entity_index)
        for name, anchor in (("sumo_vehicle.speed_mps", 'sumo.get("speed_mps")'),
                             ("annotations.speed_mps", 'annotations.get("speed_mps")'),
                             ("truth_pose.velocity_enu_mps", 'pose.get("velocity_enu_mps")')))
    speed = table.add(f"{WORLD_TRUTH}:_speed#speed_mps", DETERMINISTIC, "entity speed (first available source)",
                      (WORLD_TRUTH, "_speed", "def _speed("), inputs=speed_inputs, unit="m/s",
                      index=_extend_index(entity_index, (WORLD_TRUTH, "_truth_projection", "speed = _speed(entity)")))
    table.add("context:truth.speed_mps", DETERMINISTIC, "context truth.speed_mps",
              (WORLD_TRUTH, "_truth_projection", 'result["speed_mps"] = speed'), inputs=(speed,), unit="m/s",
              index=CONTRACT_ROLES_INDEX)


def _declare_direct_producers(table: _Table) -> None:
    """Truth functions of every direct authority, with their row reads."""
    building: set[str] = set()

    def value_node(module: str, function: str, builder: str, index: int) -> str:
        node_id = f"{module}:{function}@{builder}#value"
        if node_id in table.items:
            return node_id
        if node_id in building:
            raise ContractError(f"{module}:{function} is recursive")
        building.add(node_id)
        node = _function(module, function)
        reads, callees = _row_reads(module, function, node.args.args[index].arg)
        keys, inputs, guards, skipped = _row_keys(module, builder), [], [], []
        row_index = builder_row_index(module, builder)  # every row field is stored once per builder row
        for path, line in reads.items():
            field = ".".join(path)
            if path[0] not in keys:
                skipped.append(field)
                continue
            row_id = f"{module}:{builder}#{field}"
            if row_id not in table.items:
                table.add(row_id, UNRESOLVED, f"{builder} row field {field}", provenance=(f"{module}:{line}",),
                          index=row_index,
                          unresolved=(f"no producer declaration for row field {field} of {builder}",))
            # This helper validates the authored enum, then always returns
            # UNKNOWN: queue waiting supplies no task completion evidence.
            if module == COMPUTE_COMM and function == "_truth_from_deadline_missed" and path == ("deadline", "status"):
                _locate(module, function, 'return "unknown"')
                guards.append(row_id)
            else:
                inputs.append(row_id)
        for callee, callee_index in callees:
            output = value_node(module, callee, builder, callee_index)
            (guards if callee == "_source_requirement_status" else inputs).append(output)
        building.discard(node_id)
        status_output = function == "_source_requirement_status"
        return table.add(node_id, DETERMINISTIC, f"{function} on {builder} rows", provenance=(f"{module}:{node.lineno}",),
                         inputs=inputs, guards=guards, unit="state-code" if status_output else "four-value", index=row_index,
                         data_type="categorical" if status_output else "four_value",
                         domain=StateDomain("enumeration", ("source_required", "missing_source_record", "not_required"))
                         if status_output else StateDomain("four_value", TRUTH_DOMAIN),
                         note=(f"reads not emitted by {builder}: {', '.join(skipped)}" if skipped else ""))

    for authority in DIRECT_PREDICATE_PRODUCERS:
        for binding in direct_predicate_bindings(authority).values():
            value_node(DIRECT_PREDICATE_PRODUCERS[authority].module, binding.function, binding.row_builder,
                       binding.row_arg_index)


def _declare_identity_namespaces(table: _Table) -> None:
    for contract in load_unified_contracts(executable_only=True).values():
        for field, source in contract.grounding_spec["field_sources"].items():
            if not source.startswith("context:"):
                continue
            namespace, _, key = source[len("context:"):].partition(".")
            if namespace not in IDENTITY_CONTEXT_NAMESPACES:
                continue
            file, where, row_keys, row_anchor = IDENTITY_CONTEXT_NAMESPACES[namespace]
            schema_source = next(item.source for item in contract.required_fields if item.field == field)
            if not schema_source.endswith("/" + file):
                raise ContractError(f"{contract.predicate_id}.{field}: source {schema_source} is not {file}")
            node_id = f"context:{namespace}.{key}"
            if node_id not in table.items:
                table.add(node_id, DETERMINISTIC, f"context {namespace}.{key} (verbatim {file} field)", where,
                          inputs=(table.root(f"l0:{file}#{key}", f"{file} {key}", where,
                                             index=_index(row_keys, row_anchor)),),
                          index=CONTRACT_ROLES_INDEX)


class _ProducerDeclarations:
    """Source-indexed declaration helpers; identities are producer outputs or source cells."""

    def __init__(self, table: _Table, module: str, function: str,
                 keys: tuple[str, ...], anchor: str):
        self.table, self.module, self.function = table, module, function
        self.index = _index(keys, (module, function, anchor))

    def root(self, source: str, anchor: str, data_type: str = "object", unit: str = "nonmeasurement",
             *, function: str | None = None, keys: tuple[str, ...] | None = None,
             kind: str = STOCHASTIC_CANDIDATE, domain: StateDomain | None = None,
             note: str = "") -> str:
        function = self.function if function is None else function
        index = self.index if keys is None else _index(keys, (self.module, function, anchor))
        node_id = source
        if node_id in self.table.items:
            existing = self.table.items[node_id]
            if (existing.kind != kind or existing.index.keys != index.keys
                    or existing.data_type != "unknown" and existing.data_type != data_type
                    or existing.unit is not None and existing.unit != unit
                    or existing.domain.kind != "unspecified" and existing.domain != (domain or _typed_domain(data_type))):
                raise ContractError(f"conflicting producer root declaration for {node_id}")
            self.table.items[node_id] = replace(existing, data_type=data_type, unit=unit,
                domain=domain or _typed_domain(data_type), provenance=tuple(dict.fromkeys(
                    existing.provenance + (_locate(self.module, function, anchor),))))
            return node_id
        return self.table.add(node_id, kind, source, (self.module, function, anchor), index=index,
                              unit=unit, data_type=data_type, domain=domain or _typed_domain(data_type), note=note)

    def value(self, output: str, anchor: str, inputs: Iterable[str], *, function: str | None = None,
              data_type: str = "object", unit: str = "nonmeasurement", guards: Iterable[str] = (),
              lag: Iterable[str] = (), domain: StateDomain | None = None, keys: tuple[str, ...] | None = None,
              note: str = "") -> str:
        function = self.function if function is None else function
        index = self.index if keys is None else _index(keys, (self.module, function, anchor))
        return self.table.add(f"{self.module}:{function}#{output}", DETERMINISTIC, output,
                              (self.module, function, anchor), index=index, inputs=inputs, guards=guards,
                              lag=lag, unit=unit, data_type=data_type, domain=domain or _typed_domain(data_type), note=note)

    def project(self, node_id: str, source: str, anchor: str, *, module: str | None = None,
                function: str | None = None) -> str:
        declaration = self.table.items[source]
        return self.table.add(node_id, DETERMINISTIC, node_id,
                              (module or self.module, self.function if function is None else function, anchor),
                              index=CONTRACT_ROLES_INDEX if node_id.startswith(("record:", "context:")) else declaration.index,
                              inputs=(source,), unit=declaration.unit,
                              data_type=declaration.data_type, domain=declaration.domain)


def _typed_domain(data_type: str) -> StateDomain:
    return StateDomain({"boolean": "boolean_or_unknown", "numeric": "numeric_or_unknown",
                        "integer": "integer_or_unknown", "identifier": "identifier_or_missing",
                        "array": "source_declared_array", "object": "source_declared_object",
                        "vector3": "enu_vector3_or_missing", "categorical": "source_declared_codes"}[data_type],
                       (False, True, "unknown") if data_type == "boolean" else ())


def _declare_compute(table: _Table) -> None:
    p = _ProducerDeclarations(table, COMPUTE_COMM, "_build_compute_rows", ("actor",), "for record in frame_records:")
    tick = "clock:tick"
    entities = p.root("l0:truth_frames.jsonl#entities", 'entities = frame.get("entities")', function="_current_entities", keys=("tick",), data_type="array")
    roster = p.value("compute_entities", "compute_entities = sorted(", (entities,), data_type="array", keys=("tick",),
                     note="Current UAV/vehicle entities sorted by entity_id; per-category rank is computed from this set.")
    category = p.value("category", "category = _entity_category(entity)", (roster,), data_type="categorical")
    rank = p.value("entity_rank", "for rank, entity in enumerate(category_entities)", (roster, category),
                   data_type="integer", unit="count")
    profile = p.root(_profile("compute.node_profiles[category]"), "node_profile = node_profiles.get(category)",
                     kind=PARAMETER, keys=())
    task = p.root(_profile("compute.task_profiles[category]"), "task_profile = task_profiles.get(category)",
                  kind=PARAMETER, keys=())
    schedule = p.root(_profile("compute.node_unavailability_schedule"), 'schedule = compute_profile.get("node_unavailability_schedule")',
                      function="_compute_node_available", data_type="array", kind=PARAMETER, keys=())
    defaults = p.root(_profile("compute.max_queue_depth_default"), "max_queue_depth_default =", kind=PARAMETER, keys=(),
                      data_type="numeric", unit="count")
    declared = p.value("declared_capacity", "declared_capacity = {", (profile, category, defaults),
                       note="cpu_cores, gpu_units, memory_mb and max_queue_depth; missing numeric values are explicit.")
    available = p.value("available", "return False", (schedule, tick, rank, category), function="_compute_node_available",
                        data_type="boolean")
    capacity = p.value("capacity", "capacity = dict(declared_capacity)", (declared, available))
    speed_inputs = tuple(p.root(f"l0:truth_frames.jsonl#actor.{field}", anchor, function="_entity_speed",
                                data_type=data_type, unit="m/s") for field, anchor, data_type in (
                                    ("sumo_vehicle.speed_mps", '("sumo_vehicle", "speed_mps")', "numeric"),
                                    ("annotations.speed_mps", '("annotations", "speed_mps")', "numeric"),
                                    ("truth_pose.velocity_enu_mps", 'velocity = pose.get("velocity_enu_mps")', "vector3")))
    speed = p.value("speed_mps", "return math.sqrt(float(velocity[0])", speed_inputs, function="_entity_speed", data_type="numeric", unit="m/s",
                    note="First known SUMO or annotations speed; else XY velocity magnitude; missing speed becomes 0 in demand only.")
    demand = p.value("demand", '"cpu_cores": None', (task, speed, tick, category), function="_compute_demand",
                     note="Base cpu/gpu/memory, cpu_per_mps, load_windows and deterministic_variation. "
                          "_unit_interval uses category, tick and rounded speed; _seed_digest is unused.")
    allocation = p.value("allocation_and_balance", "allocated = min(requested, cap)", (demand, capacity),
                         function="_allocate_compute")
    residual = p.value("residual_cpu_by_node", "residual_cpu_by_node = {", (capacity, allocation, roster),
                       keys=("tick",), note="Shared frame-wide residual CPU is decremented once for each accepted offload.")
    seed = p.root("param:compute_comm_common#seed_digest", 'seed_digest=common["seed_digest"]',
                  data_type="identifier", kind=PARAMETER, keys=())
    target = p.value("target_node_id", "candidates = sorted(", (residual, available, roster, seed, tick),
                     function="_select_compute_offload_target", data_type="identifier",
                     note="Sorted available peer node candidates with residual CPU; deterministic digest offset.")
    accepted = p.value("accepted_cpu", "accepted_cpu = min(queued_cpu, residual_cpu_by_node[target_node_id])",
                       (target, residual, allocation), data_type="numeric", unit="cpu_cores")
    offload = p.value("offload", "enabled = accepted_cpu_cores > 0.0", (accepted, target, allocation),
                      function="_compute_offload")
    queue = p.value("queue_depth", 'balance["queue_depth"] = _compute_queue_depth(balance)',
                    (allocation, accepted), data_type="integer", unit="count",
                    note="After offloading: ceil CPU unmet demand, or one task for unmet GPU/memory; maximum of these counts.")
    missing = p.value("missing_source_record", "missing_source_record = bool(missing_parameters)",
                      (profile, task, declared, demand), data_type="boolean")
    carried_id = _cc("_build_compute_rows", "queue_started_tick")
    started = p.value("queue_started_tick", "queue_started_tick_by_entity[entity_id] = tick", (queue, missing, tick, roster),
                      lag=(carried_id,), data_type="integer", unit="tick",
                      note="Retain start while queued; clear when missing, unqueued or actor leaves frame.")
    deadline = p.value("deadline", "deadline_tick = queue_started_tick + slack", (task, tick, started),
                       function="_deadline", guards=(missing,),
                       note="Inactive when no queue start; invalid/missing slack or missing source yields unknown; >= marks overdue.")
    execution = p.value("execution_state", 'execution_state = "unknown"', (available, queue, capacity, offload),
                        guards=(missing,), data_type="categorical",
                        domain=StateDomain("enumeration", ("unknown", "failed", "migrating", "queued", "running_offloaded", "running")))
    migration = p.value("migration_active", "migration_active = bool(", (available, offload), data_type="boolean")
    # These identities are the exact fields consumed by parsed truth functions.
    for field, source, anchor in (
        ("resource_request", demand, '"resource_request": _numeric_or_unknown_dict(demand)'),
        ("node_availability.available", available, '"available": node_available,'),
        ("offload.enabled", offload, '"offload": record["offload"],'),
        ("migration.active", migration, '"active": migration_active,'),
        ("deadline.status", deadline, '"deadline": deadline,'),
        ("failure.requirement_status", missing, '"requirement_status": "missing_source_record"')):
        projected = p.project(_cc("_build_compute_rows", field), source, anchor)
        scalar_schemas = {
            "offload.enabled": ("boolean", "nonmeasurement", _typed_domain("boolean")),
            "deadline.status": ("categorical", "state-code", StateDomain("enumeration", ("inactive", "within_budget", "overdue", "unknown"))),
            "failure.requirement_status": ("categorical", "state-code", StateDomain("enumeration", ("missing_source_record", "not_required"))),
        }
        if field in scalar_schemas:
            data_type, unit, domain = scalar_schemas[field]
            table.items[projected] = replace(table.items[projected], data_type=data_type, unit=unit, domain=domain)
    # Keep the object producers and project their exact published numeric leaves.
    for group, anchor in (
        ("capacity", '"capacity": _numeric_or_unknown_dict(capacity)'),
        ("resource_request", '"resource_request": _numeric_or_unknown_dict(demand)'),
    ):
        parent = _cc("_build_compute_rows", group)
        table.items[parent] = replace(table.items[parent], source_cell=SourceCell(
            "compute_state.jsonl", group, ("episode_id", "tick", *p.index.keys),
            "coherent_monitor_container"))
        for resource, unit in (("cpu_cores", "cores"), ("gpu_units", "gpu_units"), ("memory_mb", "MB")):
            field_path = f"{group}.{resource}"
            leaf = p.project(_cc("_build_compute_rows", field_path), parent, anchor)
            table.items[leaf] = replace(table.items[leaf], data_type="numeric", unit=unit,
                domain=_typed_domain("numeric"), source_cell=SourceCell(
                    "compute_state.jsonl", field_path, ("episode_id", "tick", *p.index.keys),
                    "published_monitor", parent))
    # capacity, queue_depth and execution_state are already the exact row identities.
    # offload.enabled, migration.active and deadline.status are exact published
    # compute_state.jsonl row fields with the _build_compute_rows anchors
    # ('"offload": record["offload"],' / '"active": migration_active,' /
    # '"deadline": deadline,'); their producer derivation stays metadata, not
    # an observation demand, and roles/cutoff/missing semantics are unchanged.
    for field, form, anchor in (
        ("node_availability.available", "published_monitor", None),
        ("queue_depth", "published_monitor", None),
        ("failure.requirement_status", "publisher_requirement", None),
        ("offload.enabled", "published_monitor", '"offload": record["offload"],'),
        ("migration.active", "published_monitor", '"active": migration_active,'),
        ("deadline.status", "published_monitor", '"deadline": deadline,'),
        ("execution_state", "published_monitor", '"execution_state": execution_state,'),
    ):
        node_id = _cc("_build_compute_rows", field)
        provenance = table.items[node_id].provenance
        if anchor is not None:
            provenance = tuple(dict.fromkeys(provenance + (
                _locate(COMPUTE_COMM, "_build_compute_rows", anchor),)))
        extra = {}
        if field == "execution_state":
            extra = {"data_type": "categorical", "unit": "nonmeasurement",
                     "domain": StateDomain("enumeration",
                                           ("unknown", "failed", "migrating", "queued",
                                            "running_offloaded", "running"))}
        table.items[node_id] = replace(table.items[node_id], source_cell=SourceCell(
            "compute_state.jsonl", field, ("episode_id", "tick", *p.index.keys), form),
            provenance=provenance, **extra)


def _declare_geometry_completion(table: _Table) -> None:
    GC = "GeometricStateComputer.compute"
    p = _ProducerDeclarations(table, STATE_COMPUTERS, GC, ("aircraft",), "for uav_id in present_uavs:")
    position = "l0:trajectories.jsonl#aircraft.position_enu_m"
    velocity = "l0:trajectories.jsonl#aircraft.velocity_enu_mps"
    scene = "param:scene_setup.json#entities"
    roster = p.root("param:global_entity_roster.json#entities", 'roster_by_id = _index_entities(roster_value.get("entities")',
                    kind=PARAMETER, keys=())
    global_plan = p.root("param:global_uav_task_plan#pad_and_ground_policy", "GLOBAL_UAV_TASK_PLAN_PATH",
                         function="resolve_aircraft_ground_references", kind=PARAMETER, keys=())
    ground_p = _ProducerDeclarations(table, STATE_COMPUTERS, "resolve_aircraft_ground_references", ("aircraft",), "for uav_id in uav_ids:")
    pads = ground_p.value("pads", "pads.update(global_uav_pads)", (scene, roster, global_plan), keys=("facility",))
    home = ground_p.value("home_by_uav", "home_by_uav[uav_id] = (", (roster, scene, pads, global_plan),
                          note="Lifecycle home pad/hover, global flow origin/target, assigned landing zone and explicit ground sources preserve source precedence.")
    activities = ground_p.root("l0:trajectories.jsonl#aircraft.activity", "activities_by_tick,", data_type="categorical")
    radius = ground_p.root("param:governed_parameter_defaults#landing_zone_radius_m", "landing_zone_radius_m = float(",
                           kind=PARAMETER, keys=(), data_type="numeric", unit="m")
    contact_speed = ground_p.root("param:predicate_state_computers#AIRCRAFT_STATIONARY_SPEED_THRESHOLD_MPS", "AIRCRAFT_STATIONARY_SPEED_THRESHOLD_MPS",
                                  function="_landing_pad_contact_references_by_tick", kind=PARAMETER, keys=(), data_type="numeric", unit="m/s")
    contacts = ground_p.value("pad_contact_references", "pad_contact_references_by_tick = _landing_pad_contact_references_by_tick(",
                              (pads, position, velocity, activities, radius, contact_speed), keys=("facility",),
                              lag=(f"{STATE_COMPUTERS}:resolve_aircraft_ground_references#pad_contact_references",),
                              note="Actual landing/parked contact samples and velocity gates feed each pad's contact reference.")
    old_ground_id = f"{STATE_COMPUTERS}:{GC}#aircraft_geometry.local_ground_reference_z_m"
    old = table.items[old_ground_id]
    table.items[old_ground_id] = replace(old, inputs=(position, home, pads, contacts, radius, global_plan), unresolved=(),
        note="Assigned landing contact; else own home contact; else nearest contacted pad within radius; else ENU ground surface.")
    for field, anchor, data_type, unit in (
        ("home_pad_id", '"home_pad_id": home_pad_id,', "identifier", "nonmeasurement"),
        ("assigned_landing_zone_id", '"assigned_landing_zone_id": landing_zone_id,', "identifier", "nonmeasurement"),
        ("xy_distance_to_home_pad_m", '"xy_distance_to_home_pad_m": xy_home,', "numeric", "m"),
        ("xy_distance_to_assigned_landing_zone_m", '"xy_distance_to_assigned_landing_zone_m": xy_landing,', "numeric", "m")):
        value = p.value("aircraft_geometry." + field, anchor, (home, position) if data_type == "numeric" else (home,),
                        data_type=data_type, unit=unit)
        p.project("context:geometry." + field, value, 'context["geometry"].update(_values(aircraft_geometry))',
                  module=WORLD_TRUTH, function="_build_tick_contexts")
    margin = "param:governed_parameter_defaults#separation_margin_m"
    event_script = p.root("param:event_script.json#separation_models", "event_script = _load_object(event_script_path)",
                          kind=PARAMETER, keys=())
    separation = p.value("separation_models", "separation_models = _aircraft_pair_separation_models(",
                         (event_script, scene, roster, margin), keys=("first_aircraft", "second_aircraft"),
                         note="Source _equivalent_aircraft_separation_distance uses horizontal/vertical limits for anisotropic safety volumes.")
    pair_positions = []
    for role, anchor in (("first_aircraft", "tick_positions[first], tick_positions[second]"),
                         ("second_aircraft", "tick_positions[first], tick_positions[second]")):
        pair_positions.append(p.root(f"l0:trajectories.jsonl#{role}.position_enu_m", anchor, "vector3", "m", keys=(role,)))
    pair = p.value("pair_distance_m", "distance = _equivalent_aircraft_separation_distance(",
                   (*pair_positions, separation, margin), data_type="numeric", unit="m",
                   keys=("first_aircraft", "second_aircraft"))
    p.project("record:predicate_contract_aircraft_pair_geometry#values.pair_distance_m", pair, '"pair_distance_m": distance,')
    regions = p.value("restricted_regions", "regions = _restricted_regions(scene_by_id)", (scene,), keys=("region",))
    protected = p.value("protected_regions", "protected_regions = _protected_regions(scene_by_id)", (scene,), keys=("region",))
    for authority, output, anchor, geometry in (
        ("predicate_contract_aircraft_region_geometry", "signed_distance_to_boundary_m", "signed_distance = _signed_distance_to_restricted_prism(position, region)", regions),
        ("predicate_contract_aircraft_protected_region_geometry", "inside_protected_airspace", "inside_protected, containment = _protected_prism_membership(position, region)", protected)):
        value = p.value(output, anchor, (position, geometry), keys=("aircraft", "region"),
                        data_type="boolean" if output.startswith("inside") else "numeric",
                        unit="nonmeasurement" if output.startswith("inside") else "m")
        p.project(f"record:{authority}#values.{output}", value, anchor)
    corridors = f"{STATE_COMPUTERS}:{GC}#corridors"
    occupants = p.value("occupants_by_corridor", "occupants_by_corridor = {", (position, corridors, "clock:tick"),
                        data_type="array", keys=("corridor",),
                        note="Only active corridors; membership uses oriented box and current present UAV positions.")
    count = p.value("corridor_occupancy_count", '"corridor_occupancy_count": len(occupants),', (occupants,),
                    data_type="integer", unit="count", keys=("corridor",))
    capacity = p.value("corridor_capacity", '"corridor_capacity": int(corridor["capacity"]),', (corridors,),
                       data_type="integer", unit="count", keys=("corridor",),
                       note="Independent cross-section lanes computed by _airspace_corridors from corridor extent and separation margin.")
    for field, value in (("corridor_occupancy_count", count), ("corridor_capacity", capacity)):
        p.project(f"record:predicate_contract_corridor_geometry#values.{field}", value, f'"{field}":')
    # Exact generic pair-row wiring; each role receives its own position cell.
    pair_p = _ProducerDeclarations(table, STATE_COMPUTERS, "_append_full_pair_geometry_rows", ("aircraft",), "for aircraft_id in sorted(uav_ids):")
    for first, second, authority in (("aircraft", "structure", "predicate_contract_aircraft_structure_geometry"),
                                     ("aircraft", "vehicle", "predicate_contract_aircraft_vehicle_geometry"),
                                     ("aircraft", "pedestrian", "predicate_contract_aircraft_pedestrian_geometry"),
                                     ("vehicle", "pedestrian", "predicate_contract_vehicle_pedestrian_geometry")):
        sources_list = []
        for role in (first, second):
            output_id = f"{STATE_COMPUTERS}:_position_at_tick_or_static#{role}.position_enu_m"
            if output_id in table.items:
                sources_list.append(output_id)
                continue
            dynamic = pair_p.root(f"l0:trajectories.jsonl#{role}.position_enu_m", "_position_at_tick_or_static(", "vector3", "m", keys=(role,))
            static = pair_p.value(f"{role}.static_position", "entity_metadata.get(entity_id, {})", (scene, roster),
                                  function="_position_at_tick_or_static", keys=(role,), data_type="vector3", unit="m")
            sources_list.append(pair_p.value(f"{role}.position_enu_m", "return tick_positions.get(entity_id)", (dynamic, static),
                                            function="_position_at_tick_or_static", keys=(role,), data_type="vector3", unit="m",
                                            note="Current tick position; static metadata permitted by exact helper only."))
        sources = tuple(sources_list)
        value = pair_p.value(f"{first}_{second}.distance_m", '"distance_m": math.dist(first_position, second_position)',
                             sources, data_type="numeric", unit="m", keys=(first, second))
        pair_p.project(f"record:{authority}#values.distance_m", value, f'"{authority}",')
    activity_p = _ProducerDeclarations(table, STATE_COMPUTERS, "restricted_region_activity_by_tick", ("region",),
                                      "for region_id, region in sorted(regions.items()):")
    observed = activity_p.root("l0:truth_frames.jsonl#restricted_region_runtime_activity[region_id]", 'projected_activity = row.get("restricted_region_runtime_activity")',
                               data_type="boolean", note="Exact region-indexed projected activity when present.")
    region_entity = activity_p.root("l0:truth_frames.jsonl#region", "entity = entity_by_id.get(region_id)",
                                    note="Exact region entity presence/runtime fields and logical_asset_id validation.")
    runtime = activity_p.value("active", "tick_activity[region_id] = tick >= activation_tick", (regions, observed, region_entity, "clock:tick"),
                               data_type="boolean", note="Static activation/deactivation interval; dynamic projected truth or restricted_region_runtime_active(entity).")
    table.add("canonical:state.restricted_region_active", DETERMINISTIC, "region-indexed restricted activity",
              (STATE_COMPUTERS, "restricted_region_activity_by_tick", "result[tick] = tick_activity"),
              inputs=(runtime,), index=activity_p.index, data_type="boolean", unit="nonmeasurement", domain=_typed_domain("boolean"))


def _declare_plan_and_control(table: _Table) -> None:
    function = "PlanWindowComputer.compute"
    p = _ProducerDeclarations(table, STATE_COMPUTERS, function, ("aircraft",), "windows = rth_windows.get(uav_id, ())")
    observed = p.root("l0:truth_frames.jsonl#aircraft.control_state.cooperative_flag", 'raw_value = control["cooperative_flag"]',
                      function="_observed_aircraft_cooperation", data_type="boolean",
                      note="Strict boolean when present; absent/null or missing entity stays unknown.")
    cooperation = p.value("cooperative_flag", '"cooperative_flag": cooperation[(tick, uav_id)]["value"]', (observed,), data_type="boolean")
    script = p.root("param:event_script.json#rth_and_landing_events", 'script = _load_object(event_script_path)', kind=PARAMETER, keys=())
    fire = p.value("fire_ticks", "fire_ticks = _resolve_event_fire_ticks(",
                   (script, "l0:trajectories.jsonl#aircraft.position_enu_m", f"{STATE_COMPUTERS}:GeometricStateComputer.compute#restricted_regions"),
                   note="Resolve required RTH and related landing event fire ticks against positions and regions.")
    windows = p.value("rth_windows", "rth_windows = _control_windows(events, fire_ticks, rth_event_ids)", (script, fire), data_type="array")
    landing = p.value("landing_starts", "landing_starts = _landing_start_ticks(events, fire_ticks)", (script, fire))
    active = p.value("rth_mode_active", "rth_active = any(", (windows, landing, "clock:tick"), data_type="boolean",
                     note="RTH is active from any control-window start until landing start; absent landing uses last formal tick + step.")
    for name, value in (("cooperative_flag", cooperation), ("rth_mode_active", active)):
        p.project(f"context:plan.{name}", value, 'context["plan"].update(_values(aircraft_plan))',
                  module=WORLD_TRUTH, function="_build_tick_contexts")
    c = _ProducerDeclarations(table, CONTROL_STATE, "_derive_values", ("entity",), "control_sources = (")
    for output, family, helper in (("safe_hold_active", "control_state", "_control_bool"),
                                   ("mission_abort_active", "mission_state", "_mission_bool")):
        truth = c.root(f"l0:truth_frames.jsonl#entity.{family}", f'{family}=_mapping(entity.get("{family}"))', function="_sample_entity")
        domain = c.root(f"l0:l0_predicate_state.jsonl#entity.values.{family}",
                        "control_sources = (" if family == "control_state" else "mission_sources = (",
                        note="Only the explicitly typed domain family and its exact API namespace participate; arbitrary row values cannot act as control/mission state.")
        modes = c.root(f"param:control_response_state#{family}.enum_rules", "enum_values=CONTROL_MODE_VALUES" if family == "control_state" else "enum_values=MISSION_MODE_VALUES",
                       function=helper, kind=PARAMETER, keys=())
        value = c.value(output, f'"{output}": {helper}("{output}",', (truth, domain, modes),
                        data_type="boolean", note="Source ordering: sampled structured family, domain structured family, exact domain API family. "
                        "_structured_state_bool checks explicit bool, active commands, enums then inactive evidence; otherwise unknown.")
        c.project(f"context:control.{output}", value, 'context["control"].update(values)',
                  module=WORLD_TRUTH, function="_project_control_response_values")
    position = c.root("l0:truth_frames.jsonl#entity.truth_pose.position_enu_m", 'position = _vector3(pose.get("position_enu_m"))',
                      function="_sample_entity", data_type="vector3", unit="m")
    route = c.root("l0:truth_frames.jsonl#entity.planned_route_waypoints_enu_m", 'entity.get("planned_route_waypoints_enu_m", ())',
                   function="_sample_entity", data_type="array", unit="m")
    distance = c.value("route_distance_m", "return min(", (position, route), function="_route_distance", data_type="numeric", unit="m",
                       note="Minimum XY distance to route segments; absent position or fewer than two waypoints stays missing.")
    threshold = c.root("param:control_response_state#PATH_DEVIATION_THRESHOLD_M", "PATH_DEVIATION_THRESHOLD_M = 15.0",
                       function="", kind=PARAMETER, keys=(), data_type="numeric", unit="m")
    deviation = c.value("path_deviation", "route_distance > PATH_DEVIATION_THRESHOLD_M", (distance, threshold), data_type="boolean")
    c.project("context:navigation.path_deviation", deviation, '"navigation.path_deviation": "path_deviation"',
              module=CONTROL_STATE, function="")


def _declare_facility_energy(table: _Table) -> None:
    p = _ProducerDeclarations(table, DOMAIN_STATE, "_facility_rows", ("facility",), "for facility in facilities:")
    observed = p.root("l0:truth_frames.jsonl#facility.facility_state", 'facility_state = _nested_mapping(facility, ("facility_state",))')
    scope = p.root("param:global_entity_roster.json#facility.semantic_scope", "semantic_scope = validate_roster_facility_scope(facility)", kind=PARAMETER)
    subtype = p.value("facility_subtype", 'subtype = str(semantic_scope["scope_subtype"])', (scope,), data_type="categorical")
    plan = p.root("param:charging_service_plan.json#facilities", "plan = inputs.charging_service_plan", function="_charging_service_state", kind=PARAMETER)
    charging = p.value("charging_service_state", "facility_plan = matches[0]", (plan, "clock:tick", scope), function="_charging_service_state",
                       note="Exactly one plan per facility and one availability interval at tick; requests ordered by arrival and UAV identity; roster/capacity binding validated.")
    availability = p.value("availability", 'availability = str(charging_state["availability"])', (observed, charging, subtype),
                           data_type="categorical", domain=StateDomain("enumeration", ("available", "unavailable", "reserved", "unknown")),
                           note="Charging availability from service plan; landing pad exact observed enum; observed fault overrides to unavailable.")
    capacity = p.value("capacity", "int(contract_capacity) if isinstance(contract_capacity, int) else None", (scope,), guards=(observed, charging),
                       data_type="integer", unit="count", note="Authoritative semantic_scope.service_capacity; explicit state/charging plan capacity conflicts raise.")
    uavs = p.root("l0:truth_frames.jsonl#aircraft", "requesters = _uav_landing_requesters(facility, uavs)",
                  data_type="object", keys=("aircraft",),
                  note="Exact requester identities bind each aircraft's lifecycle home pad and global-flow origin/target IDs to facility entity_id/uav_pad_id aliases.")
    requesters = p.value("requester_ids", 'explicit_requesters = facility_state.get("requester_ids")', (observed, charging, subtype, uavs, scope),
                         data_type="array", note="Charging plan requesters; else explicit string ID list; else landing-pad requester producer; otherwise empty.")
    position = p.root("l0:truth_frames.jsonl#facility.position_enu_m", 'pad_pos = _position(pad)', function="_uav_pad_occupiers", data_type="vector3", unit="m")
    current_pose = p.root("l0:truth_frames.jsonl#aircraft.truth_pose.position_enu_m", 'position = _position(uav)',
                          function="_uav_pad_occupiers", data_type="vector3", unit="m", keys=("aircraft",))
    preflight = p.root("l0:source_episode/trajectories.jsonl#aircraft", 'raw_rows: dict[tuple[int, str], Mapping[str, Any]] = {}',
                       function="_load_pad_preflight_uavs", keys=("aircraft",),
                       note="Declared source_episode_dir/trajectories.jsonl and manifest; identity/coverage conflicts raise, missing artifacts produce explicit gaps.")
    occupancy_positions = p.value("occupancy_uav_positions", "occupancy_uavs = [*uavs, *inputs.preflight_uavs_by_tick.get(tick, [])]",
                                 (current_pose, preflight, scope), data_type="array", unit="m", keys=("aircraft",),
                                 note="Current UAV poses plus declared source preflight poses; retain coverage/visibility gaps.")
    params = p.root("param:domain_state_profile#pad_facility.occupancy_geometry", 'params = profile["domain_models"]["pad_facility"]', kind=PARAMETER, keys=())
    occupiers = p.value("occupier_ids", "occupiers.append(str(uav[\"entity_id\"]))", (position, occupancy_positions, params),
                        function="_uav_pad_occupiers", data_type="array", note="XY radius and altitude tolerance define physically occupying aircraft IDs.")
    occupied = p.value("occupied", 'occupied=True if occupiers else UNKNOWN', (occupiers,), data_type="boolean",
                       note="Producer reports true when known occupiers exist; empty evidence remains unknown.")
    for field, source in (("availability", availability), ("capacity", capacity), ("occupied", occupied)):
        p.project(f"context:domain.pad_facility.{field}", source, 'result[family] = dict(_values(row))',
                  module=WORLD_TRUTH, function="_domain_projection")
    count = table.add(
        f"{WORLD_TRUTH}:_domain_projection#pad_facility.requester_count", DETERMINISTIC, "pad requester count",
        (WORLD_TRUTH, "_domain_projection", 'pad["requester_count"] = len(requesters)'), inputs=(requesters,),
        index=p.index, data_type="integer", unit="count", domain=_typed_domain("integer"))
    p.project("context:domain.pad_facility.requester_count", count, 'pad["requester_count"] = len(requesters)', module=WORLD_TRUTH, function="_domain_projection")
    contention = p.value("contention", "contention = len(requesters) > capacity", (requesters, capacity), data_type="boolean")
    pair = p.value("contention_pairs.pair_contention", '"pair_contention": bool(', (requesters, contention, scope), data_type="boolean",
                   keys=("facility", "first_aircraft", "second_aircraft"),
                   note="Each roster aircraft pair shares the pad's requester set and contention; both aircraft must belong to that exact pad's set.")
    p.project("record:pad_facility:values.contention_pairs#pair_contention", pair, '"pair_contention": bool(')
    e = _ProducerDeclarations(table, DOMAIN_STATE, "_payload_rows", ("aircraft",), "previous_soc = battery_soc_by_uav.get(uav_id)")
    energy_params = e.root("param:domain_state_profile#payload_energy", 'params = profile["domain_models"]["payload_energy"]', kind=PARAMETER, keys=())
    weather = e.root("l0:weather_meta.jsonl#temperature_c", '_number(weather.get("temperature_c"))', data_type="numeric", unit="degC", keys=("tick",))
    sumo_speed = e.root("l0:truth_frames.jsonl#aircraft.sumo_vehicle.speed_mps", 'number = _number(_nested_value(entity, ("sumo_vehicle", "speed_mps")))',
                        function="_speed", data_type="numeric", unit="m/s")
    energy_velocity = e.root("l0:truth_frames.jsonl#aircraft.truth_pose.velocity_enu_mps", 'velocity = pose.get("velocity_enu_mps")',
                             function="_velocity", data_type="vector3", unit="m/s")
    speed = e.value("speed_mps", "return math.sqrt(velocity[0] ** 2", (sumo_speed, energy_velocity), function="_speed", data_type="numeric", unit="m/s")
    activation = e.root("param:global_entity_roster.json#aircraft.activation_tick", 'roster_entity.get("activation_tick")', kind=PARAMETER, data_type="integer", unit="tick")
    history = e.value("preflight_history_coverage", 'preflight_gap = preflight_energy_gap_by_uav.get(uav_id)',
                      (preflight, activation), data_type="object",
                      note="Preflight calls advance the same battery SOC recurrence from source poses; missing pre-visible history remains explicit, never an invented observed SOC.")
    policy_start = e.root("param:domain_state_profile#authoritative_tick_policy.start",
                          'int(profile["authoritative_tick_policy"]["start"])',
                          kind=PARAMETER, keys=(), data_type="integer", unit="tick")
    policy_step = e.root("param:domain_state_profile#authoritative_tick_policy.step",
                         'tick_step = int(profile["authoritative_tick_policy"]["step"])',
                         kind=PARAMETER, keys=(), data_type="integer", unit="tick")
    initial_slots = e.root("param:domain_state#battery_soc_by_uav.initial_membership",
                           'battery_soc_by_uav: dict[str, float | None] = {}',
                           function="_build_observation_rows", kind=PARAMETER, keys=(),
                           domain=StateDomain("empty_mapping"))
    entry_id = f"{DOMAIN_STATE}:_payload_rows#battery_soc_entry_present"
    entry = e.value("battery_soc_entry_present", 'battery_soc_by_uav[uav_id] = soc if isinstance(soc, float) else None',
                    (uavs, history, initial_slots), lag=(entry_id,), data_type="boolean",
                    note="Carried dictionary-key membership, distinct from SOC value: a written None still has an entry. "
                         "Caller preflight-gap assignments can also establish an unresolved entry before visible payload processing.")
    prior_entry = e.value("prior_battery_soc_entry_present", 'if uav_id not in battery_soc_by_uav:',
                          (initial_slots, history), lag=(entry,), data_type="boolean",
                          note="Membership before this payload call; include caller preflight updates and preserve membership even when the prior SOC is None.")
    first_seen = e.value("first_energy_call", 'if uav_id not in battery_soc_by_uav:', (prior_entry,), data_type="boolean",
                         note="Exact absence of a carried dictionary entry, not render visibility or missing numeric SOC.")
    interval_start = e.value("interval_start", 'interval_start = max(',
                             ("clock:tick", policy_start, policy_step, first_seen, activation),
                             data_type="integer", unit="tick",
                             note="Normally max(policy.start, tick-policy.step). On first energy call, valid activation in [normal start,tick] replaces the start; "
                                  "earlier activation leaves missing pre-visible history explicit, later activation raises, invalid activation remains missing.")
    elapsed = e.value("elapsed_ticks", 'elapsed_ticks = tick - interval_start',
                      ("clock:tick", interval_start), data_type="integer", unit="tick")
    charge_truth = e.root("l0:truth_frames.jsonl#entities", "truth_entities in sorted(inputs.charging_truth_by_tick.items())",
                         function="_charging_activity_by_tick", data_type="array", keys=("tick",))
    plan_tick = e.value("plan_tick", 'plan_tick = start + ((tick - start) // step) * step',
                        ("clock:tick", policy_start, policy_step), function="_charging_activity_by_tick",
                        keys=("tick",), data_type="integer", unit="tick",
                        note="Select charging-service schedule on the policy grid; physical contact/operational evidence remains at the current source tick.")
    charging_active = e.value("charging_activity_by_tick", 'statuses[tick] = active',
                              (plan, plan_tick, charge_truth, params, "clock:tick"), function="_charging_activity_by_tick",
                              data_type="boolean", note="Plan selects service aircraft; formal operational charging states, facility fault and physical contact gate each source-tick claim; conflicting/unknown claims remain unknown.")
    interval_charging = e.value("interval_charging", 'for source_tick in range(interval_start, tick)',
                                (charging_active, interval_start, "clock:tick"), data_type="array",
                                note="Aircraft-specific charging statuses over [interval_start,tick), across source ticks, not just the current status.")
    soc_id = f"{DOMAIN_STATE}:_payload_rows#state_of_charge_ratio"
    previous = e.value("previous_soc", "previous_soc = battery_soc_by_uav.get(uav_id)", (history, activation, energy_params),
                       lag=(soc_id,), data_type="numeric", unit="ratio", note="Use carried SOC; first visible activation initializes only when no pre-visible history is required.")
    consumption = e.value("consumed", 'float(params["soc_consumption_per_tick"])', (energy_params, weather, speed, elapsed),
                          data_type="numeric", unit="ratio", note="Consumption per elapsed tick times speed and temperature-derating factors; missing temperature/speed stays unknown.")
    charged = e.value("charged", 'float(params["soc_charge_per_tick"])', (energy_params, interval_charging),
                      data_type="numeric", unit="ratio", note="Sum True charging states over [interval_start,tick); any unknown state makes charged unknown.")
    soc = e.value("state_of_charge_ratio", "soc = (", (previous, consumption, charged, energy_params),
                  data_type="numeric", unit="ratio", guards=(history, activation),
                  note="Clamp previous SOC - consumed + charged to minimum_soc..1; missing prior state or charging history stays unknown.")
    e.project("context:domain.payload_energy.state_of_charge_ratio", soc, 'result[family] = dict(_values(row))',
              module=WORLD_TRUTH, function="_domain_projection")


def _declare_observed_domains(table: _Table) -> None:
    g = _ProducerDeclarations(table, DOMAIN_STATE, "_gnss_values", ("aircraft",), "raw_navigation = uav.get(\"navigation_state\")")
    navigation = g.root("l0:truth_frames.jsonl#aircraft.navigation_state", 'raw_navigation = uav.get("navigation_state")', data_type="object")
    flags = g.value("source_flags", "input_keys = {", (navigation,), data_type="object",
                   note="Exact input_keys aliases: multipath, visual relocalization, spoofed, geofence, recovery. Conflicting or malformed aliases raise; absent evidence is unknown.")
    modes = g.value("operational_state", "operational_keys = (", (navigation,), data_type="categorical",
                   domain=StateDomain("enumeration", ("nominal", "multipath_degraded", "spoofed", "geofence_alert", "relocalizing", "recovered", "unknown")))
    params = g.root("param:domain_state_profile#gnss", 'base_error = float(params["base_error_m"])', kind=PARAMETER, keys=())
    seed = g.root("param:domain_state_common#seed_digest", '_unit_interval("gnss_error_angle", seed_digest, uav_id, tick)',
                  kind=PARAMETER, keys=(), data_type="identifier")
    initial = g.root("param:gnss#initial_error_range", "previous_gnss_error: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))",
                     function="_build_observation_rows", kind=PARAMETER, data_type="array", unit="m")
    candidates_id = f"{DOMAIN_STATE}:_gnss_values#candidates"
    previous = g.value("previous_error_range", 'previous_gnss_error[uav_id] = gnss_values.pop("_next_error_range_m")', (initial,),
                       function="_uav_rows", lag=(candidates_id,), data_type="array", unit="m",
                       note="Carry min/max candidate error from preceding sampled tick; validate finite nonnegative ordered range.")
    candidates = g.value("candidates", "for flags in product(*choices):", (flags, params, previous, seed, "clock:tick"),
                         data_type="array", note="Enumerate missing-flag possibilities and previous range endpoints. FSM precedence: multipath, spoofed, geofence, relocalization, recovered. "
                         "Error decay conditional on decreasing magnitude without spoof/geofence; direction is deterministic digest or configured spoof angle.")
    spoofed = g.value("gnss_spoofed", '"gnss_spoofed": source_flags["gnss_spoofed"]', (flags,), data_type="boolean")
    error = g.value("position_error_m", "error = error_values[0] if all(", (candidates,), data_type="numeric", unit="m",
                    note="Emit only unanimous numeric magnitude across candidates, else unknown. Actual pose does not drive error magnitude.")
    quality = g.value("quality_level", "quality = quality_values[0] if all(", (candidates,), data_type="categorical",
                      domain=StateDomain("enumeration", ("nominal", "degraded", "spoofing_suspect", "unknown")))
    for field, value in (("gnss_spoofed", spoofed), ("position_error_m", error), ("quality_level", quality)):
        g.project(f"record:gnss_navigation#values.{field}", value, f'"{field}":')
    # modes affect row provenance/resolved quality bookkeeping, not these outputs.
    c = _ProducerDeclarations(table, DOMAIN_STATE, "_crowd_row", ("cohort",), "structured_evacuation_values = [")
    pedestrians = c.root("l0:truth_frames.jsonl#entities", "for pedestrian in cohort_pedestrians", data_type="array", keys=("tick",))
    members = c.value("cohort_member_ids", 'crowd_initial["cohort_member_ids"] = sorted(', (pedestrians,), data_type="array",
                      lag=(f"{DOMAIN_STATE}:_crowd_row#cohort_member_ids",),
                      note="Freeze observed non-background pedestrian roster at first authoritative tick; use exact cohort member IDs thereafter.")
    observed = c.root("l0:truth_frames.jsonl#pedestrian.pedestrian_state.evacuation_active", '("evacuation_active",)', data_type="boolean", keys=("pedestrian",))
    active = c.value("evacuation_active", "evacuation_active = _truth_or(*structured_evacuation_values)", (members, observed),
                     data_type="boolean", note="Observed cohort OR: true if any true; false if any known false and no true; otherwise unknown. Continuous movement is not evacuation evidence.")
    c.project("record:crowd_evacuation#values.evacuation_active", active, '"evacuation_active": evacuation_active,')
    s = _ProducerDeclarations(table, DOMAIN_STATE, "_security_rows", ("entity",), 'entity_id = str(entity["entity_id"])')
    compromised = s.root("l0:truth_frames.jsonl#entity.security_state.gcs_compromised", 'compromised = _state_bool(structured, ("gcs_compromised",))', data_type="boolean")
    auth = s.value("auth_state", "auth_state = _auth_state_from_score(auth_score)", (compromised,), data_type="categorical",
                   note="Observed compromised true->auth_score .25, false->1, unknown->None; _auth_state_from_score classifies.")
    s.project("record:security_command#values.auth_state", auth, '"auth_state": auth_state,')
    jamming = s.root("l0:truth_frames.jsonl#actor.security_state.jamming_active", 'jamming = _state_bool(structured, ("jamming_active",))',
                     data_type="boolean", keys=("actor",))
    station_id = table.add(_cc("_build_communication_rows", "station_id"), DETERMINISTIC, "row station identity",
                           (COMPUTE_COMM, "_build_communication_rows", '"station_id": station_id,'),
                           inputs=(_cc("_apply_handover_policy", "station"),), index=builder_row_index(COMPUTE_COMM, "_build_communication_rows"),
                           data_type="identifier", unit="nonmeasurement", domain=_typed_domain("identifier"))
    station = s.value("communication_station_id", 'communication.get("station_id")', (station_id,), data_type="identifier", keys=("actor",),
                     note="Exact communication actor row lookup binds observed jamming flag to station identity; this is semantic authority, not spatial propagation.")
    jam = s.value("jamming_indicator", '"jamming_indicator": jamming,', (jamming,), guards=(station,), data_type="boolean", keys=("actor", "station"),
                  note="Observed actor jamming flag; active aircraft must carry an explicit station binding. No station-specific physical propagation model is implemented.")
    s.project("record:security_command#values.jamming_indicator", jam, '"jamming_indicator": jamming,')
    t = _ProducerDeclarations(table, DOMAIN_STATE, "_traffic_rows", ("lane",), "for lane_id, lane_vehicles in sorted(vehicles_by_lane.items()):")
    vehicles = t.root("l0:truth_frames.jsonl#vehicle", 'lane_id = _nested_value(vehicle, ("sumo_vehicle", "sumo_lane_id"))',
                      data_type="object", keys=("vehicle",))
    speed_threshold = t.root("param:domain_state_profile#ground_traffic.queue_speed_threshold_mps", 'float(params["queue_speed_threshold_mps"])',
                             kind=PARAMETER, keys=(), data_type="numeric", unit="m/s")
    stopped = t.value("stopped_by_lane", 'stopped_by_lane[lane_id].append(str(vehicle["entity_id"]))', (vehicles, speed_threshold),
                      data_type="array", note="Exact lane IDs and known vehicle speed <= threshold; no rolling/global queue-window substitution.")
    count = t.value("queue_vehicle_count", '"queue_vehicle_count": len(queued_ids)', (stopped,), data_type="integer", unit="count")
    t.project("record:signal_queue_lane_state#values.queue_vehicle_count", count, '"queue_vehicle_count": len(queued_ids)')
    signal = t.root("l0:truth_frames.jsonl#signal", 'structured_fault = _state_bool(facility_state, ("fault",))', keys=("signal",))
    sumo = t.root("l0:truth_frames.jsonl#sumo_traffic_light_states[signal_id]", 'sumo_traffic_lights = frame.get("sumo_traffic_light_states")', keys=("signal",))
    incidents = t.root("l0:truth_frames.jsonl#sumo_active_incidents", 'active_incidents = frame.get("sumo_active_incidents")', data_type="array", keys=("tick",))
    classified = t.value("signal_states", "for signal_id, state in sumo_traffic_lights.items():", (signal, sumo, incidents),
                         data_type="categorical", keys=("signal",),
                         domain=StateDomain("enumeration", ("all_red_fault", "nominal", "sumo_cycle", "unknown")),
                         note="Scene facility fault plus declared state; SUMO signal string validated, exact incident target required for all_red_fault; key collisions/missing target states raise.")
    health = t.value("controller_health", '"controller_health": (', (classified,), data_type="categorical", keys=("signal",),
                     domain=StateDomain("enumeration", ("all_red_fault", "nominal", "unknown")),
                     note="Per-signal classification, not traffic-system aggregate health.")
    t.project("record:traffic_signal_state#values.controller_health", health, '"controller_health": (')
    props = t.root("l0:truth_frames.jsonl#entities", "barrier_props = _barrier_props(props)", data_type="array", keys=("tick",))
    barriers = t.value("barrier_props", "barrier_props = _barrier_props(props)", (props,), data_type="array", keys=("tick",))
    anchors = t.root("param:scene_setup.json#road_segment_anchors", "barrier_segment_ids = _barrier_segment_ids(scene_setup)", kind=PARAMETER, keys=())
    closed = t.value("closed_segment_ids", "closed_segment_ids = frozenset(", (incidents,), data_type="array", keys=("tick",),
                     note="Only exact SUMO edge anchors of road-closing incidents establish known closure; affected vehicles and present barriers do not prove lane obstruction.")
    uncertainty = t.value("uncertain_barrier_segment_ids", "uncertain_barrier_segment_ids = {", (barriers, anchors), data_type="array", keys=("tick",))
    closure = t.value("road_closed", 'True if segment_id in closed_segment_ids', (closed,), guards=(incidents, uncertainty, anchors),
                      data_type="boolean", keys=("region",), note="Exact segment member true; unknown if closure scope unknown or segment has uncertain barrier; otherwise false.")
    t.project("record:road_segment_state#values.road_closed", closure, '"road_closed": (')
    # Attach receiver views at established producer coordinates. Record views
    # retain their contract-role indexes and their original projection edges.
    for node_id, field in ((closure, "values.road_closed"),
                           (health, "values.controller_health")):
        table.items[node_id] = replace(table.items[node_id], source_cell=SourceCell(
            "domain_state_supplement/domain_state_observations.jsonl", field,
            ("episode_id", "tick", *table.items[node_id].index.keys), "published_monitor"))


def _declare_lockdown_utm(table: _Table) -> None:
    p = _ProducerDeclarations(table, OBSERVABLE_STATE, "_lockdown_region_rows", ("region",), "for entity in entities:")
    entities = p.root("l0:truth_frames.jsonl#entities", 'static_entity = static_entities.get(entity_id, {})',
                      data_type="array", keys=("tick",), note="Explicit region incident/constraint state; controller state, render presence, identity and usable pose.")
    static = p.root("param:scene_and_roster#lockdown.static_entities", 'static_entity = static_entities.get(entity_id, {})', kind=PARAMETER, keys=())
    explicit = p.value("explicit_region_active", "active = _constraint_active(entity)", (entities, static), data_type="boolean",
                       note="Exact emergency-isolation zone entity; incident_state.temporary_lockdown_active precedes constraint_state.active.")
    deployed = p.value("controller_deployed", "deployed = _lockdown_control_deployed(entity)", (entities,), keys=("controller",), data_type="boolean",
                       note="Usable position required; governed active bool precedes state/render/physical identity. Exclude explicit airspace regions from controllers.")
    derived = p.value("derived_region_active", "if active_controls:", (deployed, static), data_type="boolean",
                      note="Aggregate exact deployed controllers: any true ->true, unknown/no known ->unknown, otherwise false. Region ID is composed from sorted controller_ids.")
    active = p.value("temporary_lockdown_active", '"temporary_lockdown_active": lockdown_active,', (explicit, derived), data_type="boolean",
                     note="Two separately emitted row branches: explicit zone entity identity, or stable identifier over controller set; each resolves its own exact zone binding.")
    p.project("record:lockdown_region_state#values.temporary_lockdown_active", active, '"temporary_lockdown_active": lockdown_active,')
    function = "build_utm_episode_artifacts"
    u = _ProducerDeclarations(table, UTM_STATE, function, ("aircraft",), "for plan in service_plan[\"uav_plans\"]:")
    plan = u.root("param:event_script.json#parameters.utm_service_plan", 'for plan in service_plan["uav_plans"]:', kind=PARAMETER, keys=(),
                  note="Validated complete authorization/flight-plan schedules, active intent intervals, UAV/corridor IDs and service model/system identity.")
    roster = u.root("param:global_entity_roster.json#utm.uav_and_corridor_bindings", '"global_entity_roster.json",', kind=PARAMETER, keys=())
    declarations = u.value("declarations", "declarations.append(", (plan, roster),
                           note="authorization=episode,UAV,airspace; plan/intent=episode,UAV,corridor; system=model_id,model_version.")
    authorization = u.value("authorization.status", 'declaration["authorization_schedule"]', (declarations, "clock:tick"),
                            data_type="categorical", keys=("authorization",),
                            note="_status_at_tick selects exactly one interval inclusive start/end; invalid coverage raises.")
    flight = u.value("flight_plan.approval_status", 'declaration["flight_plan_schedule"]', (declarations, "clock:tick"),
                     data_type="categorical", keys=("plan",))
    active_intents = u.value("active_intents", "active_intents = {", (declarations, "clock:tick"), keys=("intent",),
                             note="Intent interval inclusive start/end keyed by exact UAV identity.")
    conflicts = u.value("conflicts_by_uav", 'if first["corridor_id"] != second["corridor_id"]:', (active_intents, declarations),
                         data_type="array", keys=("aircraft",), note="Pairs of active declarations conflict exactly when corridor IDs match.")
    resolution = u.value("resolution_status", '"resolution_status": "active"', (conflicts, declarations),
                          data_type="categorical", keys=("system", "aircraft"), domain=StateDomain("enumeration", ("active", "not_required")))
    pair = u.value("pair.conflict_active", "conflict_active = bool(", (active_intents, declarations), data_type="boolean",
                    keys=("first_intent", "second_intent"),
                    note="Both exact intents active and same corridor; intent identities bind their UAVs independently.")
    for authority, field, value, anchor in (
        ("airspace_authorization_log.jsonl", "status", authorization, '"status": _status_at_tick('),
        ("flight_plan_log.jsonl", "approval_status", flight, '"approval_status": _status_at_tick('),
        ("deconfliction_log.jsonl", "resolution_status", resolution, '"resolution_status": "active"'),
        ("operational_intent_pair_log.jsonl", "conflict_active", pair, '"conflict_active": conflict_active,')):
        u.project(f"record:{authority}#{field}", value, anchor)


@lru_cache(maxsize=1)
def _count_unit_aliases() -> Mapping[str, tuple[str, tuple[str, ...]]]:
    """Counts of the exact entity collection named by each contract unit."""
    rows = (
        ("context:domain.pad_facility.capacity", "aircraft", DOMAIN_STATE, "_facility_rows", 'int(contract_capacity) if isinstance(contract_capacity, int) else None'),
        ("context:domain.pad_facility.requester_count", "aircraft", WORLD_TRUTH, "_domain_projection", 'pad["requester_count"] = len(requesters)'),
        ("record:predicate_contract_corridor_geometry#values.corridor_occupancy_count", "aircraft", STATE_COMPUTERS, "GeometricStateComputer.compute", '"corridor_occupancy_count": len(occupants),'),
        ("record:predicate_contract_corridor_geometry#values.corridor_capacity", "aircraft", STATE_COMPUTERS, "GeometricStateComputer.compute", '"corridor_capacity": int(corridor["capacity"]),'),
        ("record:signal_queue_lane_state#values.queue_vehicle_count", "vehicle", DOMAIN_STATE, "_traffic_rows", '"queue_vehicle_count": len(queued_ids),'),
    )
    return {node_id: (unit, (_locate(module, function, anchor),))
            for node_id, unit, module, function, anchor in rows}


def _declare_energy_caller_reset(table: _Table) -> None:
    p = _ProducerDeclarations(table, DOMAIN_STATE, "_build_observation_rows", ("aircraft",),
                              'battery_soc_by_uav[uav_id] = None')
    visible = p.value("visible_uav_ids", 'visible_uav_ids = {str(uav["entity_id"]) for uav in uavs}',
                      ("l0:truth_frames.jsonl#entities",), data_type="array", keys=("tick",))
    reset = p.value("invisible_charging_soc_reset", 'if state == UNKNOWN and uav_id not in visible_uav_ids:',
                    (f"{DOMAIN_STATE}:_charging_activity_by_tick#charging_activity_by_tick", visible,
                     "param:domain_state_profile#authoritative_tick_policy.start",
                     "param:domain_state_profile#authoritative_tick_policy.step", "clock:tick"),
                    data_type="boolean", note="For each non-visible UAV, any UNKNOWN charging status in the inclusive "
                    "[max(policy.start,tick-policy.step),tick] writes a dictionary entry with SOC None before payload calls.")
    for output in ("battery_soc_entry_present", "prior_battery_soc_entry_present", "previous_soc"):
        node_id = f"{DOMAIN_STATE}:_payload_rows#{output}"
        old = table.items[node_id]
        table.items[node_id] = replace(old, guards=old.guards + (reset,), provenance=old.provenance +
            (_locate(DOMAIN_STATE, "_build_observation_rows", 'battery_soc_by_uav[uav_id] = None'),))


def _canonicalize_source_cells(table: _Table) -> None:
    """One source family per artifact/field; retain each role-reading occurrence.

    Whole truth-frame and entity-container inputs overlap their scalar fields.
    Every such input is therefore a coherent projection of one frame root,
    rather than an additional independent stochastic candidate.
    """
    frame = "l0:truth_frames.jsonl#frame"
    frame_refs = (_locate(COMPUTE_COMM, "load_episode_inputs", 'for row in read_jsonl(paths["truth_frames"]):'),
                  _locate(DOMAIN_STATE, "load_episode_inputs", 'for raw_row in read_jsonl(paths["truth_frames"]):'))
    table.add(frame, STOCHASTIC_CANDIDATE, "coherent truth-frame observation at its source tick",
              provenance=frame_refs, index=_index(("tick",), (COMPUTE_COMM, "load_episode_inputs", 'tick = row.get("tick")')),
              data_type="object", unit="nonmeasurement", domain=_typed_domain("object"),
              note="One observed frame contains all entity fields and frame-level runtime projections; contained views are not independent candidate roots.")
    table.items[frame] = replace(table.items[frame], source_cell=SourceCell("truth_frames.jsonl", "", ("episode_id", "tick"), "coherent_frame_root"))
    entity_roles = {"actor", "aircraft", "entity", "station", "facility", "vehicle", "pedestrian", "structure", "first_aircraft", "second_aircraft"}

    def family(artifact: str, path: str, old: _Decl) -> str:
        node_id = f"l0:{artifact}#entity_id.{path}"
        known = table.items.get(node_id)
        index = ProducerIndex(INDEX_ESTABLISHED, ("entity_id",), old.index.provenance)
        if known is not None:
            if (known.data_type, known.unit, known.domain) != (old.data_type, old.unit, old.domain):
                raise ContractError(f"source-cell schema conflict for {node_id}")
            table.items[node_id] = replace(known, provenance=tuple(dict.fromkeys(known.provenance + old.provenance)),
                index=replace(known.index, provenance=tuple(dict.fromkeys(known.index.provenance + old.index.provenance))))
            return node_id
        if artifact == "truth_frames.jsonl":
            table.add(node_id, DETERMINISTIC, f"truth-frame entity field {path}", provenance=old.provenance,
                      inputs=(frame,), index=index, data_type=old.data_type, unit=old.unit, domain=old.domain)
        else:
            table.add(node_id, STOCHASTIC_CANDIDATE, f"trajectory entity field {path}", provenance=old.provenance,
                      index=index, data_type=old.data_type, unit=old.unit, domain=old.domain)
        table.items[node_id] = replace(table.items[node_id], source_cell=SourceCell(
            artifact, path, ("episode_id", "tick", "entity_id"), "canonical_cell_family",
            frame if artifact == "truth_frames.jsonl" else None))
        return node_id

    original = tuple(table.items.items())
    for node_id, old in original:
        if old.kind != STOCHASTIC_CANDIDATE or node_id == frame:
            continue
        if node_id.startswith("l0:truth_frames.jsonl#"):
            path = node_id.split("#", 1)[1]
            role, separator, field = path.partition(".")
            if role in entity_roles and separator and role in old.index.keys:
                # The legacy position accessors also admit initial-position
                # and (for synthetic stations) authored roster-anchor sources.
                accessor = field == "position_enu_m" or node_id == "l0:truth_frames.jsonl#actor.truth_pose.position_enu_m"
                raw_field = "truth_pose.position_enu_m" if field == "position_enu_m" else field
                inputs = (family("truth_frames.jsonl", raw_field, old),)
                if accessor:
                    inputs += (family("truth_frames.jsonl", "initial_position_enu_m", old),)
                if role == "station" and field == "position_enu_m":
                    inputs += ("param:global_entity_roster.json#entities",)
                cell = SourceCell("truth_frames.jsonl", raw_field, ("episode_id", "tick", role),
                                  "producer_position_accessor" if accessor else "role_selection", inputs[0])
            else:
                inputs, cell = (frame,), SourceCell("truth_frames.jsonl", path,
                    ("episode_id", "tick") + tuple(key for key in old.index.keys if key != "tick"), "coherent_frame_projection", frame)
                if path == "station.roster":
                    inputs += ("param:global_entity_roster.json#entities",)
            table.items[node_id] = replace(old, kind=DETERMINISTIC, inputs=inputs, source_cell=cell,
                note=old.note + " Coherent source view; never sampled independently of its frame/canonical source family.")
        elif node_id.startswith("l0:trajectories.jsonl#"):
            path = node_id.split("#", 1)[1]
            role, separator, field = path.partition(".")
            if role in entity_roles and separator and role in old.index.keys:
                canonical = family("trajectories.jsonl", field, old)
                table.items[node_id] = replace(old, kind=DETERMINISTIC, inputs=(canonical,),
                    source_cell=SourceCell("trajectories.jsonl", field, ("episode_id", "tick", role), "role_selection", canonical),
                    note=old.note + " Role selection from the canonical trajectory cell family; artifact remains distinct from truth_frames.")
        elif node_id.startswith("l0:weather_meta.jsonl#"):
            table.items[node_id] = replace(old, source_cell=SourceCell("weather_meta.jsonl", node_id.split("#", 1)[1],
                ("episode_id", "tick"), "canonical_field_root"))


def _schema_projection(node_id: str, decl: _Decl, schema: FieldSchema) -> SchemaProjection:
    type_relation, unit_relation = "identical", "identical"
    refs = decl.provenance + (schema.source,)
    count_alias = _count_unit_aliases().get(node_id)
    if decl.data_type != schema.data_type:
        if decl.data_type == "integer" and schema.data_type == "numeric" and count_alias is not None:
            type_relation = "integer_is_numeric"
        else:
            raise ContractError(f"{node_id}: producer type {decl.data_type} conflicts with contract {schema.data_type}")
    if decl.unit != schema.unit:
        code_unit = {"boolean": "bool", "identifier": "identifier", "categorical": "state-code"}.get(decl.data_type)
        if code_unit is not None and {decl.unit, schema.unit} <= {"nonmeasurement", code_unit}:
            unit_relation = "nonmeasurement_code_alias"
        elif decl.unit == "count" and count_alias is not None and count_alias[0] == schema.unit:
            unit_relation = "count_of_contract_entity"
            refs += count_alias[1]
        else:
            raise ContractError(f"{node_id}: producer unit {decl.unit} conflicts with contract {schema.unit}")
    contract_domain = _domain(schema)
    finite_kinds = {"enumeration", "four_value", "boolean", "boolean_or_unknown"}
    domain_relation = "producer_domain_not_fully_declared"
    if decl.domain.kind in finite_kinds:
        referenced = schema.categories or contract_domain.values
        invalid = tuple(value for value in referenced if not any(
            type(value) is type(authored) and value == authored for authored in decl.domain.values))
        if invalid:
            raise ContractError(f"{node_id}: contract categories {invalid} are outside producer domain {decl.domain.values}")
        domain_relation = "contract_categories_subset" if referenced else "finite_producer_no_contract_referenced_codes"
    return SchemaProjection(decl.data_type, decl.unit, schema.data_type, schema.unit,
                            type_relation, unit_relation, tuple(dict.fromkeys(refs)),
                            decl.domain, contract_domain, domain_relation)


def _annotate_declarations(table: _Table) -> None:
    """Attach schemas from contract projection authority and producer expressions.

    This does not infer a physical unit from a numerical type. Unspecified
    source units remain explicit metadata gaps and prevent completeness.
    """
    def annotate(node_id: str, data_type: str, unit: str, domain: StateDomain | None = None) -> None:
        old = table.items[node_id]
        proposed = FieldSchema(node_id, data_type, unit, "producer_annotation", ())
        if old.data_type != "unknown" and old.unit is not None:
            _schema_projection(node_id, old, proposed)
        elif old.data_type != "unknown" and old.data_type != data_type:
            raise ContractError(f"{node_id}: conflicting producer type annotation")
        table.items[node_id] = replace(old, data_type=data_type if old.data_type == "unknown" else old.data_type,
                                       unit=unit if old.unit is None else old.unit,
                                       domain=(domain or _typed_domain(data_type)) if old.domain.kind == "unspecified" else old.domain)

    # Exact primitive output names and structured accessor semantics.
    primitive = {
        "clock:tick": ("integer", "tick"),
        "l0:truth_frames.jsonl#actor.truth_pose.position_enu_m": ("vector3", "m"),
        "l0:truth_frames.jsonl#station.position_enu_m": ("vector3", "m"),
        "l0:truth_frames.jsonl#station.roster": ("array", "nonmeasurement"),
        "l0:trajectories.jsonl#aircraft.position_enu_m": ("vector3", "m"),
        "l0:trajectories.jsonl#aircraft.velocity_enu_mps": ("vector3", "m/s"),
        "param:scene_setup.json#entities": ("object", "nonmeasurement"),
        "param:governed_parameter_defaults#separation_margin_m": ("numeric", "m"),
        _cc("_select_station", "candidate_station"): ("object", "nonmeasurement"),
        _cc("_apply_handover_policy", "station"): ("object", "nonmeasurement"),
        _cc("_build_communication_rows", "previous_station_id"): ("identifier", "nonmeasurement"),
        _cc("_build_communication_rows", "logical_handover"): ("array", "nonmeasurement"),
        _cc("_station_operational_state", "observed_mode"): ("categorical", "nonmeasurement"),
        _cc("_station_operational_state", "missing_parameters"): ("array", "nonmeasurement"),
        f"{STATE_COMPUTERS}:GeometricStateComputer.compute#corridors": ("object", "nonmeasurement"),
        f"{STATE_COMPUTERS}:GeometricStateComputer.compute#aircraft_geometry.position_enu_m": ("vector3", "m"),
    }
    for node_id, (data_type, unit) in primitive.items():
        annotate(node_id, data_type, unit)
    for role in ("actor", "station"):
        for family, accessor in (("communication_state", "_structured_communication_input"),
                                  ("security_state", "_structured_security_input")):
            for field in _string_set(COMPUTE_COMM, accessor):
                data_type, unit = (("categorical", "nonmeasurement") if field == "mode" else
                                   ("numeric", "ratio") if field == "availability" else ("boolean", "nonmeasurement"))
                annotate(f"l0:truth_frames.jsonl#{role}.{family}.{field}", data_type, unit)
    profile_units = {
        "communication.flow_profiles.uav.latency_threshold_ms": "ms",
        "communication.flow_profiles.uav.packet_loss_threshold_ratio": "ratio",
        "communication.flow_profiles.uav.bandwidth_mbps": "Mbps",
        "communication.channel.nominal_latency_ms": "ms",
        "communication.channel.worst_case_latency_ms": "ms",
        "communication.channel.maximum_packet_loss_ratio": "ratio",
        "communication.channel.deterministic_retransmission_probability_floor": "ratio",
        "communication.station_profiles.base_station.nominal_range_m": "m",
        "communication.station_profiles.base_station.channel_bandwidth_mbps": "Mbps",
        "communication.handover_policy.handover_margin_m": "m",
        "communication.handover_policy.minimum_dwell_ticks": "tick",
    }
    for field, unit in profile_units.items():
        annotate(_profile(field), "integer" if unit == "tick" else "numeric", unit)
    for field in ("operational_factor", "capacity_factor", "range_factor", "availability"):
        annotate(_profile(f"communication.station_profiles.base_station.operational_states[mode].{field}"), "numeric", "ratio")
    for node_id, data_type in ((_profile("communication.station_profiles.base_station.backup_station_id"), "identifier"),
                               (_profile("communication.flow_profiles.uav"), "object"),
                               (_profile("communication.station_profiles.base_station"), "object")):
        annotate(node_id, data_type, "nonmeasurement")
    for node_id in table.items:
        if node_id.startswith("param:global_entity_roster.json#uav."):
            annotate(node_id, "numeric", "m")
    # Field schema authority propagates only across literal identity projections.
    for contract in load_unified_contracts(executable_only=True).values():
        for schema in contract.required_fields:
            node_id = field_node_id(contract, schema.field)
            if node_id not in table.items:
                continue
            old = table.items[node_id]
            if old.data_type == "unknown":
                annotate(node_id, schema.data_type, schema.unit,
                         _domain(schema) if _domain(schema).kind != "unspecified" else None)
            _schema_projection(node_id, table.items[node_id], schema)
            if node_id.startswith(("context:weather.", "context:truth.")):
                for source in old.inputs:
                    if source.startswith("l0:"):
                        annotate(source, schema.data_type, schema.unit,
                                 _domain(schema) if _domain(schema).kind != "unspecified" else None)
    # Remaining truth projection cells are copied without type conversion; the
    # exact field forms identify authored IDs, booleans and numeric quantities.
    truth_units = {"accel_mps2": ("numeric", "m/s²"), "speed_mps": ("numeric", "m/s"),
                   "velocity_enu_mps": ("vector3", "m/s"), "allowed_speed_mps": ("numeric", "m/s"),
                   "following_distance_m": ("numeric", "m"), "crossed_stop_line": ("boolean", "nonmeasurement"),
                   "in_crosswalk": ("boolean", "nonmeasurement"), "controlling_signal_state": ("categorical", "nonmeasurement")}
    for node_id, old in tuple(table.items.items()):
        if old.data_type != "unknown":
            continue
        if node_id.startswith("l0:truth_frames.jsonl#entity."):
            field = node_id.rsplit(".", 1)[-1]
            if field.endswith("_id"):
                annotate(node_id, "identifier", "nonmeasurement")
            elif field in truth_units:
                annotate(node_id, *truth_units[field])
        elif old.unit is not None:
            data_type = ("boolean" if old.unit == "bool" else "categorical" if old.unit == "state-code" else
                         "four_value" if old.unit == "four-value" else "integer" if old.unit in ("tick", "count") else "numeric")
            annotate(node_id, data_type, old.unit,
                     StateDomain("four_value", TRUTH_DOMAIN) if data_type == "four_value" else None)
    for key in ("rain", "fog_density", "dust"):
        source = f"l0:weather_meta.jsonl#{key}"
        if key == "dust" and table.items[source].data_type == "unknown":
            annotate(source, "numeric", "ratio")
        if key == "dust":
            old = table.items[source]
            table.items[source] = replace(old, provenance=old.provenance + (
                _locate("Dataset/world_model/graph/field_contract.py", "", "('dust', 'ratio', _FORMAL_WEATHER_NUMERIC_BASIS"),))
        coefficient = _profile(f"communication.channel.weather_attenuation.{key}")
        annotate(coefficient, "numeric", "source_unspecified")
    # One-input projection nodes inherit the producer schema when no authored
    # contract gives a schema. Never propagate an aggregate object's schema to
    # a scalar field selected from it.
    for _ in range(len(table.items)):
        changed = False
        for node_id, old in tuple(table.items.items()):
            if old.data_type == "unknown" and len(old.inputs) == 1:
                source = table.items[old.inputs[0]]
                if source.data_type != "unknown":
                    table.items[node_id] = replace(old, data_type=source.data_type, unit=source.unit, domain=source.domain)
                    changed = True
        if not changed:
            break
    for node_id, old in tuple(table.items.items()):
        gaps = list(old.unresolved)
        if old.data_type == "unknown":
            gaps.append("source-backed data type/domain is not established")
        if old.unit is None or old.unit == "source_unspecified":
            gaps.append("source does not declare a measurement unit")
        table.items[node_id] = replace(old, unresolved=tuple(dict.fromkeys(gaps)))


@lru_cache(maxsize=1)
def _declarations() -> Mapping[str, _Decl]:
    table = _Table()
    _declare_clock(table)
    _declare_communication(table)
    _declare_aircraft_geometry(table)
    _declare_compute(table)
    _declare_geometry_completion(table)
    _declare_plan_and_control(table)
    _declare_facility_energy(table)
    _declare_energy_caller_reset(table)
    _declare_observed_domains(table)
    _declare_lockdown_utm(table)
    _declare_truth_projection(table)
    _declare_identity_namespaces(table)
    _declare_direct_producers(table)
    _annotate_declarations(table)
    _canonicalize_source_cells(table)
    table.validate()
    for key, declared in row_key_indexes().items():
        if declared.source_node is not None and declared.source_node not in table.items:
            raise ContractError(f"row key {key} names source node {declared.source_node}, which is not declared")
    return table.items


def declared_nodes_with_unestablished_index() -> tuple[str, ...]:
    """Declared node ids whose producer index the source does not establish."""
    return tuple(node_id for node_id, decl in _declarations().items()
                 if decl.index.status == INDEX_UNESTABLISHED)


# --------------------------------------------------------------------------
# Spec construction
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _declared_node(node_id: str) -> DependencyNode:
    decl = _declarations()[node_id]
    return DependencyNode(node_id, decl.kind, decl.label, decl.unit, decl.provenance, decl.index,
                          decl.unresolved, decl.note, decl.data_type, decl.domain,
                          "established" if decl.data_type != "unknown" else "unestablished",
                          decl.provenance, "not_declared" if decl.unit == "source_unspecified" else
                          "nonmeasurement" if decl.unit in ("nonmeasurement", "identifier", "bool", "four-value", "state-code") else
                          "established" if decl.unit is not None else "unestablished",
                          decl.index.status, decl.index.keys, decl.source_cell)


def field_node_id(contract: PredicateContract, field: str) -> str:
    """Canonical identity of the state a contract field reads."""
    authority = contract.grounding_spec["candidate_authority"]
    source = contract.grounding_spec["field_sources"][field]
    if authority["kind"] == "direct_predicate_row":
        binding = direct_predicate_bindings(authority["source"]).get(contract.predicate_id)
        if binding is None:
            raise ContractError(f"{contract.predicate_id} has no producer binding in {authority['source']}")
        if tuple(sorted(role.key for role in contract.roles)) != binding.role_keys:
            raise ContractError(f"{contract.predicate_id}: contract roles disagree with the producer bindings")
        module = DIRECT_PREDICATE_PRODUCERS[authority["source"]].module
        return f"{module}:{binding.function}@{binding.row_builder}#value"
    canonical = canonical_state_field(contract, field)
    if canonical != field:
        return f"canonical:{canonical}"
    if source.startswith("context:"):
        return source  # one context cell per scope, whichever authority names it
    if source.startswith("record:"):
        return f"record:{authority['source']}#{source[len('record:'):]}"
    raise ContractError(f"{contract.predicate_id}.{field}: unsupported field source {source!r}")


def _domain(schema: FieldSchema) -> StateDomain:
    if schema.unit == "four-value":
        if schema.categories != TRUTH_DOMAIN:
            raise ContractError(f"{schema.field}: four-value field lacks the truth domain")
        return StateDomain("four_value", TRUTH_DOMAIN)
    if schema.data_type == "boolean":
        return StateDomain("boolean", (False, True))
    if schema.categories:
        return StateDomain("referenced_values", schema.categories)
    return StateDomain("unspecified")


def _threshold_names(contract: PredicateContract) -> tuple[str, ...]:
    names = []
    for node in walk_expression(contract.evaluation_spec):
        for key in ("threshold", "minimum", "maximum"):
            if key in node:
                if not isinstance(node[key], str):
                    raise ContractError(f"{contract.predicate_id}: threshold must name a declared parameter")
                names.append(node[key])
    if set(names) != set(contract.threshold_parameters):
        raise ContractError(f"{contract.predicate_id}: AST thresholds {sorted(set(names))} disagree with "
                            f"threshold_parameters {list(contract.threshold_parameters)}")
    return tuple(contract.threshold_parameters)


def _threshold_unit(contract: PredicateContract, name: str) -> str:
    schemas = {field.field: field for field in contract.required_fields}
    units = {schemas[node[field]].unit for node in walk_expression(contract.evaluation_spec)
             if name in (node.get("threshold"), node.get("minimum"), node.get("maximum"))
             for field in ("field", "left", "right") if field in node}
    if len(units) != 1:
        raise ContractError(f"{contract.predicate_id}: threshold {name} has inconsistent compared units {units}")
    return next(iter(units))


@lru_cache(maxsize=1)
def _record_index_identities() -> Mapping[str, Mapping[str, tuple[str, tuple[str, ...]]]]:
    """Producer-key to emitted identifier paths, keyed by record authority.

    These declarations cite identifier assignments in the row producer, not
    predicate names or resemblance between role names/ontology classes.
    """
    result: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {}

    def add(authority: str, key: str, path: str, module: str, function: str, *anchors: str) -> None:
        refs = tuple(_locate(module, function, anchor) for anchor in anchors)
        result[authority] = {**result.get(authority, {}), key: (path, refs)}

    geometry = "GeometricStateComputer.compute"
    for authority, entries in (
        ("predicate_contract_aircraft_region_geometry", (("aircraft", "values.aircraft_id", '"aircraft_id": uav_id,'),
                                                          ("region", "values.restricted_region_id", '"restricted_region_id": region_id,'))),
        ("predicate_contract_aircraft_protected_region_geometry", (("aircraft", "values.aircraft_id", '"aircraft_id": uav_id,'),
                                                                    ("region", "values.protected_region_id", '"protected_region_id": region_id,'))),
        ("predicate_contract_aircraft_corridor_geometry", (("aircraft", "values.aircraft_id", '"aircraft_id": uav_id,'),
                                                          ("corridor", "values.corridor_id", '"corridor_id": corridor_id,'))),
        ("predicate_contract_corridor_geometry", (("corridor", "values.corridor_id", '"corridor_id": corridor_id,'),)),
        ("predicate_contract_aircraft_pair_geometry", (("first_aircraft", "values.first_aircraft_id", '"first_aircraft_id": first,'),
                                                      ("second_aircraft", "values.second_aircraft_id", '"second_aircraft_id": second,'))),
    ):
        for key, path, anchor in entries:
            add(authority, key, path, STATE_COMPUTERS, geometry, anchor, f'"{authority}",')
    for first, second in (("aircraft", "structure"), ("aircraft", "vehicle"),
                          ("aircraft", "pedestrian"), ("vehicle", "pedestrian")):
        authority = f"predicate_contract_{first}_{second}_geometry"
        for key, slot in ((first, "first"), (second, "second")):
            add(authority, key, f"values.{key}_id", STATE_COMPUTERS, "_append_full_pair_geometry_rows",
                f'f"{{{slot}_role}}_id": {slot}_id,', f'"{authority}",')
    add("predicate_contract_region_runtime_state", "region", "values.no_fly_zone_id", STATE_COMPUTERS,
        "PlanWindowComputer.compute", '"no_fly_zone_id": region_id,')
    for authority, key, path, function, anchor in (
        ("signal_queue_lane_state", "lane", "values.queue_lane_id", "_traffic_rows", '"queue_lane_id": lane_id,'),
        ("road_segment_state", "region", "values.road_segment_id", "_traffic_rows", '"road_segment_id": segment_id,'),
        ("traffic_signal_state", "signal", "values.signal_id", "_traffic_rows", '"signal_id": signal_id,'),
        ("crowd_evacuation", "cohort", "values.crowd_id", "_crowd_row", 'stable_identifier("crowd", inputs.episode_id, cohort_ids)'),
        ("security_command", "entity", "values.ground_control_station_id", "_security_rows", 'entity_id if subject_category == "ground_station" else None'),
        ("security_command", "actor", "values.actor_id", "_security_rows", '"actor_id": entity_id if subject_category == "uav" else None,'),
        ("security_command", "station", "values.communication_station_id", "_security_rows", '"communication_station_id": communication_station_id,'),
        ("pad_facility:values.contention_pairs", "facility", "pad_id", "_facility_rows", '"pad_id": facility_id,'),
        ("pad_facility:values.contention_pairs", "first_aircraft", "first_aircraft_id", "_facility_rows", '"first_aircraft_id": first_aircraft_id,'),
        ("pad_facility:values.contention_pairs", "second_aircraft", "second_aircraft_id", "_facility_rows", '"second_aircraft_id": second_aircraft_id,'),
    ):
        add(authority, key, path, DOMAIN_STATE, function, anchor)
    add("gnss_navigation", "aircraft", "subject_id", DOMAIN_STATE, "_uav_rows",
        'uav_id = str(uav["entity_id"])', '"gnss_navigation",')
    gnss_calls = [node for node in ast.walk(_function(DOMAIN_STATE, "_uav_rows"))
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "_observation_row" and len(node.args) > 4
                  and isinstance(node.args[3], ast.Constant) and node.args[3].value == "gnss_navigation"]
    if len(gnss_calls) != 1 or not isinstance(gnss_calls[0].args[4], ast.Name) or gnss_calls[0].args[4].id != "uav_id":
        raise ContractError("gnss_navigation producer subject_id no longer reads uav_id")
    # The explicit region and derived controller/barrier branches both emit
    # the same zone identifier path; neither is a global scene sample.
    add("lockdown_region_state", "region", "values.emergency_isolation_zone_id", OBSERVABLE_STATE,
        "_lockdown_region_rows", '"emergency_isolation_zone_id": entity_id,',
        '"emergency_isolation_zone_id": region_id,')
    for authority, key, path, anchor in (
        ("airspace_authorization_log.jsonl", "authorization", "authorization_id", '"authorization_id": declaration["authorization_id"],'),
        ("flight_plan_log.jsonl", "plan", "plan_id", '"plan_id": declaration["plan_id"],'),
        ("operational_intent_pair_log.jsonl", "first_intent", "first_intent_id", '"first_intent_id": first["intent_id"],'),
        ("operational_intent_pair_log.jsonl", "second_intent", "second_intent_id", '"second_intent_id": second["intent_id"],'),
        ("deconfliction_log.jsonl", "system", "system_id", '"system_id": utm_system_id,'),
        ("deconfliction_log.jsonl", "aircraft", "uav_id", '"uav_id": declaration["uav_id"],'),
    ):
        add(authority, key, path, UTM_STATE, "build_utm_episode_artifacts", anchor)
    return result


def _field_role_projections(contract: PredicateContract, node_id: str, owners: tuple[str, ...]
                            ) -> tuple[tuple[ProducerRoleProjection, ...], tuple[str, ...]]:
    """Resolve identity at the producer/contract boundary using authored paths."""
    grounding, declarations = contract.grounding_spec, _declarations()
    authority = grounding["candidate_authority"]
    if authority["kind"] == "direct_predicate_row":
        return (), ()  # Exact row composition is already exported in RoleSlot.producer.
    frontier: dict[str, _Decl] = {}

    def visit(source: str) -> None:
        decl = declarations[source]
        if decl.index.status == INDEX_ESTABLISHED:
            frontier[source] = decl
        elif decl.index.status == INDEX_CONTRACT_ROLES:
            for item in decl.inputs + decl.guards:
                visit(item)
        else:
            frontier[source] = decl

    visit(node_id)
    role_sources = grounding["role_sources"]
    scope_roles = tuple(role for role, source in role_sources.items() if source["kind"] == "scope_entity")
    projections, gaps = [], []
    for source, decl in frontier.items():
        for key in decl.index.keys:
            if key == "tick":
                continue
            form, selector = "index_identity", "same_identifier"
            if scope_roles:
                # world_truth looks up the exact source entity for this scope;
                # upstream producer index provenance establishes its identifier.
                if len(scope_roles) != 1 or key not in ("aircraft", "entity", "facility"):
                    gaps.append(f"no exact scope-entity projection for {source} index {key}")
                    continue
                role = scope_roles[0]
                identity = role_sources[role]["identity_provenance"]["identifier_source"]
                refs = decl.index.provenance + (
                    _locate(WORLD_TRUTH, "_build_tick_contexts", "source_id = scope.source_entity_id or scope.entity_id"),
                    _locate(WORLD_TRUTH, "_build_tick_contexts", "entity = entities_at_tick.get(source_id)"))
                form = "scope_source_lookup"
                selector = "producer_id=scope.source_entity_id_or_scope.entity_id;role_id=scope.entity_id"
            else:
                entry = _record_index_identities().get(authority["source"], {}).get(key)
                if entry is None:
                    gaps.append(f"no producer identifier declaration for {authority['source']} index {key}")
                    continue
                identity, refs = entry
                matches = tuple(role for role, role_source in role_sources.items()
                                if role_source["kind"] == "record" and role_source["path"] == identity)
                if len(matches) != 1:
                    gaps.append(f"producer index {key} identifier {identity} does not resolve one contract role")
                    continue
                role = matches[0]
                refs = decl.index.provenance + refs
                if key == "cohort":
                    form, selector = "composed_identifier", 'stable_identifier("crowd", inputs.episode_id, cohort_ids)'
            projections.append(ProducerRoleProjection(source, key, role, identity, form, refs, selector))
    for role in owners:
        role_source = role_sources[role]
        if role_source["kind"] == "context":
            identity = role_source["path"]
            identity_node = "context:" + identity
            if identity_node not in declarations:
                gaps.append(f"no producer declaration for referenced role {role} identifier {identity}")
                continue
            projections.append(ProducerRoleProjection(identity_node, None, role, identity,
                                                       "referenced_identifier", declarations[identity_node].provenance + (
                    _locate(WORLD_TRUTH, "_candidate_from_authority_record", 'raw_id = _value_at_path(record, path)'),),
                                                       "identifier_value_from_context_cell"))
        elif role_source["kind"] == "manifest":
            identity = role_source["path"]
            if authority != {"kind": "episode_operational_region", "source": "episode_manifest.json#capture_boundary_id"}:
                gaps.append(f"no exact manifest projection for role {role}")
                continue
            projections.append(ProducerRoleProjection(node_id, None, role, identity, "manifest_identity",
                contract.source_refs + (_locate(WORLD_TRUTH, "_build_tick_contexts", '"weather": dict(weather_row),'),)))
    covered = {item.owner_role for item in projections}
    gaps.extend(f"owner role {role} has no producer identity projection" for role in owners if role not in covered)
    return tuple(projections), tuple(dict.fromkeys(gaps))


def _carried_state_policy(target: str) -> tuple[str, str, str, str, tuple[str, ...]]:
    """Exact cache timing/update/absence policies, keyed by producer output."""
    comm = _cc("_build_communication_rows", "")
    sampled_comm = (_locate(COMPUTE_COMM, "load_episode_inputs", 'if tick not in wanted_ticks:'),)
    sampled_domain = (_locate(DOMAIN_STATE, "load_episode_inputs", 'if tick not in wanted_ticks:'),)
    policies = {
        comm + "previous_station_id": ("last_write_before_current_actor_call", "write_only_when_station_id_known;otherwise_retain", "retain_actor_when_absent",
            COMPUTE_COMM, "_build_communication_rows", ('if station_id != "unknown":', 'previous_station_by_entity[entity_id] = str(station_id)')),
        comm + "last_switch_tick": ("last_write_before_current_actor_call", "write_on_known_station_switch_or_first_known_station;otherwise_retain", "retain_actor_when_absent",
            COMPUTE_COMM, "_build_communication_rows", ('last_switch_tick_by_entity[entity_id] = tick',)),
        comm + "declared_handover_rising": ("last_write_before_current_actor_call", "overwrite_declared_flag_on_each_present_actor_call", "retain_actor_when_absent",
            COMPUTE_COMM, "_build_communication_rows", ('previous_declared_handover_by_entity[entity_id] = (',)),
        comm + "station_entered_backup_link": ("previous_sampled_frame_map", "replace_entire_station_mode_map_after_frame", "clear_missing_station_on_map_replacement",
            COMPUTE_COMM, "_build_communication_rows", ('previous_station_mode_by_id = station_modes',)),
        comm + "logical_handover": ("last_retained_entry_before_current_actor_call", "write_valid_backup_pair_on_start;clear_when_declared_flag_not_true;otherwise_retain", "retain_actor_when_absent",
            COMPUTE_COMM, "_build_communication_rows", ('logical_handover_by_entity[entity_id] = (', 'logical_handover_by_entity.pop(entity_id, None)')),
        _cc("_build_compute_rows", "queue_started_tick"): ("retained_queue_entry_before_current_sample", "write_at_queue_start;clear_on_missing_or_zero_queue", "delete_absent_actor",
            COMPUTE_COMM, "_build_compute_rows", ('del queue_started_tick_by_entity[entity_id]', 'queue_started_tick_by_entity.pop(entity_id, None)')),
        f"{DOMAIN_STATE}:_uav_rows#previous_error_range": ("last_write_before_current_uav_call", "overwrite_with_min_max_candidate_range_on_present_uav_call", "retain_uav_when_absent",
            DOMAIN_STATE, "_uav_rows", ('previous_gnss_error[uav_id] = gnss_values.pop("_next_error_range_m")',)),
        f"{DOMAIN_STATE}:_crowd_row#cohort_member_ids": ("first_authoritative_sample_frozen_cohort", "initialize_once_even_if_empty;retain_exact_members", "retain_frozen_members_when_unobserved",
            DOMAIN_STATE, "_crowd_row", ('if "cohort_member_ids" not in crowd_initial:',)),
        f"{STATE_COMPUTERS}:resolve_aircraft_ground_references#pad_contact_references": ("observed_prefix_contact_cache", "minimum_valid_contact_height_per_pad;else_declared_geometry", "retain_contact_when_no_new_contact",
            STATE_COMPUTERS, "_landing_pad_contact_references_by_tick", ('for tick in sorted(positions):', 'contacts[pad_id] = min(contacts[pad_id], height) if pad_id in contacts else height')),
    }
    payload = f"{DOMAIN_STATE}:_payload_rows#"
    if target in tuple(payload + key for key in ("battery_soc_entry_present", "prior_battery_soc_entry_present", "previous_soc")):
        refs = sampled_domain + tuple(_locate(DOMAIN_STATE, function, anchor) for function, anchor in (
            ("_build_observation_rows", 'preflight_uavs = inputs.preflight_uavs_by_tick.get(tick, [])'),
            ("_build_observation_rows", 'if state == UNKNOWN and uav_id not in visible_uav_ids:'),
            ("_build_observation_rows", 'battery_soc_by_uav[uav_id] = None'),
            ("_payload_rows", 'battery_soc_by_uav[uav_id] = soc if isinstance(soc, float) else None')))
        return ("authoritative_sampled_ticks_with_preflight_payload_calls", "retained_key_before_payload_call_including_current_caller_updates",
                "payload_writes_numeric_or_None;caller_preflight_or_invisible_unknown_charging_writes_None_and_creates_entry",
                "retain_unobserved_uav_except_explicit_caller_null_reset", refs)
    if target not in policies:
        raise ContractError(f"no source-backed carried-state timing policy for {target}")
    lookup, update, absence, module, function, anchors = policies[target]
    refs = tuple(_locate(module, function, anchor) for anchor in anchors)
    if module == STATE_COMPUTERS:
        return "ordered_trajectory_position_ticks", lookup, update, absence, refs
    return "authoritative_sampled_ticks", lookup, update, absence, refs + (sampled_comm if module == COMPUTE_COMM else sampled_domain)


def _producer_edge(source: str, target: str, relation: str, decl: _Decl) -> DependencyEdge:
    """Export collection reduction and selected-entity lookup separately."""
    source_index = _declarations()[source].index
    target_index = decl.index
    bindings, collection, selector = (), (), ""
    if source_index.status == INDEX_ESTABLISHED:
        if not source_index.keys:
            selector = "global_parameter"
        elif target_index.status == INDEX_CONTRACT_ROLES:
            selector = "contract_field_projection;bind_via_consuming_reads_edge.role_projections"
            if "tick" in source_index.keys:
                selector += ";same_episode_tick"
        else:
            bindings = tuple((key, key) for key in source_index.keys if key in target_index.keys)
            collection = tuple(key for key in source_index.keys if key not in target_index.keys and key != "tick")
            selector = "producer_collection" if collection else "same_index_identity"
            if "tick" in source_index.keys:
                selector += ";same_episode_tick"
        if "station" in source_index.keys and target.startswith(COMPUTE_COMM + ":"):
            if target.startswith(_cc("_select_station", "")):
                bindings, collection = tuple(item for item in bindings if item[0] != "station"), ("station",)
                selector = "all_station_entities_in_current_frame;reduce_min_valid_distance_for_actor"
            elif target.startswith(_cc("_station_operational_state", "")):
                collection = ()
                bindings = tuple(item for item in bindings if item[0] != "station") + (("station", _cc("_apply_handover_policy", "station") + ".entity_id"),)
                selector = "selected_station_lookup_for_same_actor_station_flow"
            elif target in (_cc("_apply_handover_policy", "distance_m"), _cc("_apply_handover_policy", "switch_allowed")):
                bindings = tuple(item for item in bindings if item[0] != "station")
                collection = ("station",)
                selector = "station_entity_id_in_candidate_or_previous_station;actor_handover_policy"
        if (source_index.keys == ("actor", "station") and target_index.keys in (("actor",), ("entity",))
                and target in (_cc("_build_communication_rows", "logical_handover"),
                               f"{DOMAIN_STATE}:_security_rows#communication_station_id")):
            bindings = (("actor", target_index.keys[0]), ("station", _cc("_apply_handover_policy", "station") + ".entity_id"))
            collection = ()
            selector = "single_communication_row_for_actor_and_sampled_tick;selected_station_function_of_actor"
    else:
        selector = "FieldBinding.owner_roles"
    if (target == f"{DOMAIN_STATE}:_payload_rows#interval_charging"
            and source == f"{DOMAIN_STATE}:_charging_activity_by_tick#charging_activity_by_tick"):
        bindings, collection = (("aircraft", "aircraft"),), ("tick",)
        selector = "source_tick_in_[interval_start,clock:tick);same_aircraft"
    temporal = ("", "", "", "", ())
    if source == f"{DOMAIN_STATE}:_charging_activity_by_tick#charging_activity_by_tick":
        temporal = ("dense_truth_source_ticks", "same_source_tick", "", "", (
            _locate(DOMAIN_STATE, "load_episode_inputs", 'charging_truth_by_tick[tick] = _charging_truth_entities('),))
        if target == f"{DOMAIN_STATE}:_payload_rows#interval_charging":
            temporal = (temporal[0], "half_open_source_tick_interval", "", "", temporal[4])
        elif target == f"{DOMAIN_STATE}:_build_observation_rows#invisible_charging_soc_reset":
            bindings, collection = (("aircraft", "aircraft"),), ("tick",)
            selector = "source_tick_in_[max(policy.start,clock:tick-policy.step),clock:tick];same_aircraft_not_visible"
            temporal = (temporal[0], "inclusive_source_tick_interval", "write_SOC_None_and_create_entry_if_any_UNKNOWN", "applies_only_to_nonvisible_uav", temporal[4] + (
                _locate(DOMAIN_STATE, "_build_observation_rows", 'for source_tick in range(previous_tick, tick + 1):'),))
    source_cell, target_cell = _declarations()[source].source_cell, decl.source_cell
    if source_cell is not None and source_cell.form == "canonical_cell_family" and target_cell is not None:
        physical_keys = tuple(key for key in target_index.keys if key != "tick")
        if len(physical_keys) != 1:
            raise ContractError(f"{target}: canonical entity cell requires one exact role-selection key")
        bindings, collection = (("entity_id", physical_keys[0]),), ()
        selector = "bind_entity_id_to_exact_producer_role_key;" + target_cell.form + ";same_artifact_episode_tick"
    elif source_cell is not None and source_cell.form == "coherent_frame_root" and target_cell is not None:
        collection = ()
        selector = "coherent_frame_projection;" + target_cell.field_path + ";same_artifact_episode_tick"
    if target_cell is not None and target_cell.form == "producer_position_accessor":
        selector += ";truth_pose_position_else_initial_position"
        if target == "l0:truth_frames.jsonl#station.position_enu_m":
            selector += ";synthetic_station_from_roster_facility_anchor_when_source_policy_requires"
    if relation == "derives_previous_tick":
        temporal = _carried_state_policy(target)
        selector += ";" + temporal[1]
    return DependencyEdge(source, target, relation, decl.provenance,
                          use="guard" if relation == "guards" else "value",
                          index_bindings=bindings, collection_keys=collection, selector=selector,
                          time_grid=temporal[0], temporal_lookup=temporal[1], update_policy=temporal[2],
                          absence_policy=temporal[3], temporal_provenance=temporal[4])


def predicate_dependency_spec(contract: PredicateContract) -> PredicateDependencySpec:
    """Dependency specification of one executable contract."""
    if not contract.executable:
        raise ContractError(f"{contract.predicate_id} is not executable")
    grounding = contract.grounding_spec
    evaluated = set(expression_fields(contract.evaluation_spec))
    guards = set(contract.template["state_contract"]["unknown_if_missing"])
    if set(grounding["field_sources"]) != {item.field for item in contract.required_fields} \
            or not evaluated <= set(grounding["field_sources"]) \
            or set(grounding["field_sources"]) - evaluated - guards:
        raise ContractError(f"{contract.predicate_id}: AST fields, required fields and field sources disagree")
    if {role.key for role in contract.roles} != set(grounding["role_sources"]):
        raise ContractError(f"{contract.predicate_id}: roles and role_sources disagree")
    declarations = _declarations()
    monitor_projections: dict[str, list[str]] = {}
    for node_id, declaration in declarations.items():
        if (declaration.source_cell is not None and declaration.source_cell.form == "published_monitor"
                and declaration.source_cell.coherent_parent is not None):
            monitor_projections.setdefault(declaration.source_cell.coherent_parent, []).append(node_id)
    predicate_node = f"predicate:{contract.predicate_id}"
    nodes: dict[str, DependencyNode] = {predicate_node: DependencyNode(
        predicate_node, DETERMINISTIC, f"{contract.predicate_id} truth value", "four-value", contract.source_refs,
        ProducerIndex(INDEX_CONTRACT_ROLES, reason="indexed by the roles of this contract (RoleSlot keys)"),
        note=f"{contract.semantic_mode}; operators {', '.join(contract.operators)}",
        data_type="four_value", domain=StateDomain("four_value", TRUTH_DOMAIN), schema_status="established",
        schema_source=contract.source_refs, unit_status="nonmeasurement", owner_status=INDEX_CONTRACT_ROLES,
        owner_keys=tuple(role.key for role in contract.roles))}
    edges: dict[tuple[str, str, str], DependencyEdge] = {}

    def expand(node_id: str) -> None:
        if node_id in nodes:
            return
        decl = declarations.get(node_id)  # declared nodes are shared objects across specs
        if decl is None:
            nodes[node_id] = DependencyNode(
                node_id, UNRESOLVED, f"state {node_id}", None, contract.source_refs,
                CONTRACT_ROLES_INDEX if node_id.startswith(CONTRACT_FIELD_PREFIXES) else ProducerIndex(
                    INDEX_UNESTABLISHED, reason="no upstream declaration for this state"),
                ("no upstream declaration for this state field",))
            return
        nodes[node_id] = _declared_node(node_id)
        for projected in monitor_projections.get(node_id, ()):
            expand(projected)
        for source in decl.inputs:
            expand(source)
            edges[(source, node_id, "derives")] = _producer_edge(source, node_id, "derives", decl)
            if decl.data_type == "four_value":
                for projected in monitor_projections.get(source, ()):
                    edges[(projected, node_id, "derives")] = _producer_edge(
                        projected, node_id, "derives", decl)
        for source in decl.lagged_inputs:
            expand(source)
            edges[(source, node_id, "derives_previous_tick")] = _producer_edge(source, node_id, "derives_previous_tick", decl)
        for source in decl.guards:
            expand(source)
            edges[(source, node_id, "guards")] = _producer_edge(source, node_id, "guards", decl)

    bindings, unresolved = [], []
    producer_bindings: dict[str, ProducerRoleBinding] = {}
    if grounding["candidate_authority"]["kind"] == "direct_predicate_row":
        authority = grounding["candidate_authority"]["source"]
        direct = direct_predicate_bindings(authority)[contract.predicate_id]
        for read in direct.role_row_reads:
            producer_bindings[read.role] = role_producer_binding(
                DIRECT_PREDICATE_PRODUCERS[authority].module, direct.row_builder, read)
            unresolved.extend(UnresolvedItem(f"role:{contract.predicate_id}.{read.role}", reason)
                              for reason in producer_bindings[read.role].unresolved)
        if set(producer_bindings) != {role.key for role in contract.roles}:
            raise ContractError(f"{contract.predicate_id}: contract roles disagree with the producer bindings")
    roles = tuple(RoleSlot(role.key, role.ontology_class, role.declared_class_uri,
                           grounding["role_sources"][role.key]["kind"],
                           grounding["role_sources"][role.key].get("path"),
                           producer_bindings.get(role.key)) for role in contract.roles)
    for schema in contract.required_fields:
        node_id = field_node_id(contract, schema.field)
        authority_source = grounding["candidate_authority"]["source"]
        field_source = grounding["field_sources"][schema.field]
        if (producer_bindings and len(contract.roles) > 1
                and schema.field not in grounding.get("field_owner_roles", {})
                and (authority_source, field_source) not in FIELD_OWNER_DECLARATIONS):
            # The truth row's exact authored bindings, verified against contract roles above.
            owners = direct.role_keys
        else:
            owners = field_owner_roles(contract, schema.field)
        status, error = "resolved", None
        owner_refs = field_owner_source_refs(contract, schema.field)
        if producer_bindings:
            owner_refs = tuple(dict.fromkeys(owner_refs + tuple(
                ref for role in owners for ref in producer_bindings[role].provenance)))
        projections, projection_gaps = _field_role_projections(contract, node_id, owners)
        unresolved.extend(UnresolvedItem(f"binding:{node_id}", reason) for reason in projection_gaps)
        bindings.append(FieldBinding(
            schema.field, node_id, schema.data_type, schema.unit, _domain(schema),
            grounding["field_sources"][schema.field], schema.source, owners, status, error, owner_refs,
            "evaluated" if schema.field in evaluated else "availability_guard", projections,
            "unresolved" if projection_gaps else "resolved", _schema_projection(node_id, declarations[node_id], schema)))
        expand(node_id)
        provenance = owner_refs
        if grounding["candidate_authority"]["kind"] == "direct_predicate_row":
            provenance = direct_predicate_bindings(grounding["candidate_authority"]["source"])[
                contract.predicate_id].provenance + owner_refs
        edges[(node_id, predicate_node, "reads")] = DependencyEdge(
            node_id, predicate_node, "reads", provenance, owners,
            "value" if schema.field in evaluated else "guard",
            selector="direct_RoleSlot.producer" if producer_bindings else "exact_FieldBinding.producer_role_projections",
            role_projections=projections)
        for projection in projections:
            if projection.form != "referenced_identifier":
                continue
            # The grounded relation requires this identifier cell. It is an
            # applicability/identity dependency, not a numeric formula input.
            expand(projection.source_node)
            key = (projection.source_node, predicate_node, "guards")
            previous = edges.get(key)
            edge_projections = (() if previous is None else previous.role_projections) + (projection,)
            edge_owners = tuple(dict.fromkeys(item.owner_role for item in edge_projections))
            edges[key] = DependencyEdge(projection.source_node, predicate_node, "guards", projection.provenance,
                                         edge_owners, "guard", selector="grounded_context_role_identifier",
                                         role_projections=tuple(dict.fromkeys(edge_projections)))
    thresholds = []
    for name in _threshold_names(contract):
        node_id = f"threshold:{name}"
        nodes[node_id] = DependencyNode(
            node_id, PARAMETER, f"threshold parameter {name}", _threshold_unit(contract, name), contract.source_refs,
            ProducerIndex(INDEX_ESTABLISHED, (), contract.source_refs), data_type="numeric", domain=_typed_domain("numeric"),
            schema_status="established", schema_source=contract.source_refs, unit_status="established", owner_status=INDEX_ESTABLISHED)
        edges[(node_id, predicate_node, "threshold")] = DependencyEdge(
            node_id, predicate_node, "threshold", contract.source_refs)
        thresholds.append(node_id)
    for node in nodes.values():
        unresolved.extend(UnresolvedItem(node.node_id, reason) for reason in node.unresolved)
    declared = any(nodes[item.node_id].kind != UNRESOLVED for item in bindings)
    completeness = COMPLETE if not unresolved else (PARTIAL if declared else DIRECT_ONLY)
    return PredicateDependencySpec(
        contract.predicate_id, predicate_node, contract.semantic_mode, contract.operators,
        _plain(contract.evaluation_spec), roles, tuple(bindings), tuple(thresholds),
        tuple(nodes.values()), tuple(edges.values()), tuple(unresolved), completeness)


def load_dependency_specs() -> dict[str, PredicateDependencySpec]:
    """Specs of all executable contracts; fails unless exactly 72 are loaded."""
    specs = {pid: predicate_dependency_spec(contract)
             for pid, contract in load_unified_contracts(executable_only=True).items()}
    if len(specs) != EXECUTABLE_COUNT:
        raise ContractError(f"expected {EXECUTABLE_COUNT} executable contracts, found {len(specs)}")
    return specs


def _without_provenance(node: DependencyNode) -> tuple[Any, ...]:
    return (node.node_id, node.kind, node.label, node.unit, node.index.status, node.index.keys,
            node.index.reason, node.unresolved, node.note, node.data_type, node.domain, node.schema_status, node.unit_status,
            node.owner_status, node.owner_keys, node.source_cell)


def merged_dependency_graph(specs: Mapping[str, PredicateDependencySpec] | Iterable[PredicateDependencySpec]
                            ) -> DependencyGraph:
    """Union of spec graphs, deduplicated by node id and checked for conflicts."""
    items = tuple(specs.values()) if isinstance(specs, Mapping) else tuple(specs)
    nodes: dict[str, DependencyNode] = {}
    edges: dict[tuple[str, str, str], DependencyEdge] = {}
    consumers: dict[str, list[str]] = {}  # upstream node id -> predicate ids
    field_types: dict[str, tuple[str, str, str]] = {}
    for spec in items:
        for node in spec.nodes:
            known = nodes.get(node.node_id)
            if known is None:
                nodes[node.node_id] = node
            elif known != node:  # only contract-derived nodes may differ, by contract provenance
                if _without_provenance(known) != _without_provenance(node):
                    raise ContractError(f"conflicting declarations for node {node.node_id}")
                nodes[node.node_id] = replace(
                    known, provenance=tuple(dict.fromkeys(known.provenance + node.provenance)),
                    schema_source=tuple(dict.fromkeys(known.schema_source + node.schema_source)),
                    index=replace(known.index, provenance=tuple(dict.fromkeys(
                        known.index.provenance + node.index.provenance))))
            if node.node_id != spec.predicate_node_id:
                consumers[node.node_id] = consumers.get(node.node_id, []) + [spec.predicate_id]
        for edge in spec.edges:
            known = edges.get(edge.key)
            if known is not None and (known.owner_roles, known.relation, known.use, known.index_bindings,
                                      known.collection_keys, known.selector, known.role_projections, known.time_grid,
                                      known.temporal_lookup, known.update_policy, known.absence_policy) != (
                                      edge.owner_roles, edge.relation, edge.use, edge.index_bindings,
                                      edge.collection_keys, edge.selector, edge.role_projections, edge.time_grid,
                                      edge.temporal_lookup, edge.update_policy, edge.absence_policy):
                raise ContractError(f"conflicting edge {edge.key}")
            if known is None:
                edges[edge.key] = edge
            elif known != edge:
                edges[edge.key] = replace(known, provenance=tuple(dict.fromkeys(known.provenance + edge.provenance)),
                                         temporal_provenance=tuple(dict.fromkeys(known.temporal_provenance + edge.temporal_provenance)))
        for binding in spec.fields:
            signature = (binding.data_type, binding.unit, binding.domain.kind)
            if field_types.get(binding.node_id, signature) != signature:
                raise ContractError(f"field type conflict on shared node {binding.node_id}")
            field_types[binding.node_id] = signature
    return DependencyGraph(nodes, tuple(edges.values()), tuple(spec.predicate_id for spec in items),
                           {key: tuple(value) for key, value in consumers.items()})
