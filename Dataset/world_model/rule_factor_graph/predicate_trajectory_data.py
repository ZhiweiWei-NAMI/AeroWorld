"""Contract-driven sampled state and fixed predicate trajectory assembly.

This adapter consumes an already fitted state sampler. It neither fits a state
law nor reads supervision, and admits TRAIN/VALID records only. The eleven
state instants include the anchor; the PMF tail bin is never a state instant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import math

import numpy as np

from .ast_compiler import compile_rule, set_field
from .generic_grounding import (ParameterProfile, RoleBinding, TypedEntity,
    bind_roles, canonical_role_bindings, _owner)
from .unified_contracts import (ContractError, FieldSchema, PredicateContract,
    TRUTH_DOMAIN, canonical_state_field, field_owner_roles)
from .state_dependencies import DIRECT_PREDICATE_PRODUCERS, direct_predicate_bindings, builder_row_index

STATE_OFFSETS = tuple(range(0, 51, 5))
TRUTH_CODES = MappingProxyType({name: index for index, name in enumerate(TRUTH_DOMAIN)})

# Declared vector components, keyed by exact contract data type.  The ENU
# ordering is authoritative: Dataset/knowledge_graph/world_core.ttl declares
# positionEastM/positionNorthM/positionUpM and velocityEastMps/velocityNorthMps/
# velocityUpMps as the east/north/up components of the objective ENU position
# and velocity.  No norm, mean or other collapse of a vector is computed.
DECLARED_VECTORS = MappingProxyType({"vector": ("east", "north", "up"),
                                     "vector3": ("east", "north", "up")})

# Declared exact category vocabularies, keyed by (declared candidate authority,
# exact declared field source).  The second key element is the grounding
# contract's exact field_sources value for formula fields (for example
# "record:status") and the declared producer node path for direct row fields.
# Each tuple is the field's complete declared code set: the enumeration domain
# the dependency graph itself declares where one exists, otherwise the
# producer's exact emitted set.  The evaluation-AST subset is never the
# vocabulary; AST-tested codes outside a declared set remain an error.  No
# code is invented.
DECLARED_CATEGORIES = MappingProxyType({
    ("traffic_signal_state", "record:values.controller_health"):
        # state_dependencies.py:2104-2105 declares the record enumeration
        # ("all_red_fault", "nominal", "unknown").
        ("all_red_fault", "nominal", "unknown"),
    ("truth_frames.jsonl#sumo_vehicle.stop_line_id", "context:truth.controlling_signal_state"):
        # Dataset/tools/sumo_ground_flow/road_signal_context.py: per-link
        # normalization (:525-534), the no-signal code (:31-32) and the mixed
        # aggregate (:481).
        ("none", "red", "yellow", "green", "red_yellow", "off", "mixed"),
    ("global_entity_roster.json#facility", "context:domain.pad_facility.availability"):
        # state_dependencies.py:1935 declares ("available", "unavailable",
        # "reserved", "unknown"); domain_state.py:1735-1757 emits within it.
        ("available", "unavailable", "reserved", "unknown"),
    ("gnss_navigation", "record:values.quality_level"):
        # state_dependencies.py:2059 declares ("nominal", "degraded",
        # "spoofing_suspect", "unknown"); domain_state.py:3730-3738.
        ("nominal", "degraded", "spoofing_suspect", "unknown"),
    ("security_command", "record:values.auth_state"):
        # Dataset/semantic_simulation/domain_state.py:4719-4726: the producer's
        # exact emitted set including its UNKNOWN string.
        ("unknown", "authenticated", "locked_or_recovering", "compromised"),
    ("airspace_authorization_log.jsonl", "record:status"):
        # The executable registry ASTs declare exactly these codes for the
        # field ('active' tested by airspace_authorization_active, 'revoked' by
        # airspace_authorization_revoked); utm_state.py:180-183,555-568 reads
        # each declared authorization_schedule status at the tick.
        ("active", "revoked"),
    ("flight_plan_log.jsonl", "record:approval_status"):
        # utm_state.py:211,555-568 reads each declared flight_plan_schedule
        # approval_status; the declared schedules and the registry AST carry
        # exactly 'approved'.
        ("approved",),
    ("deconfliction_log.jsonl", "record:resolution_status"):
        # state_dependencies.py:2157 declares ("active", "not_required");
        # utm_state.py:244-247 emits within it.
        ("active", "not_required"),
    # Direct row fields key on (source artifact, exact declared producer cell
    # field_path); the full producer node path is cited for traceability.
    ("communication_state.jsonl", "requirement.status"):
        # state_dependencies.py:1349-1351 declares the row enumeration
        # ("source_required", "not_required"); the L/P direct receiver slice
        # publishes this status as a governed row parameter
        # (compute_comm.py:_build_communication_rows#requirement.status).
        ("source_required", "not_required"),
    ("compute_state.jsonl", "failure.requirement_status"):
        # state_dependencies.py:1737 declares the row enumeration
        # ("missing_source_record", "not_required"); compute_comm.py:1128-1132
        # emits exactly these two values for the six direct compute predicates
        # (_build_compute_rows#failure.requirement_status).
        ("missing_source_record", "not_required"),
    ("compute_state.jsonl", "execution_state"):
        # state_dependencies.py:1721-1723 declares the row enumeration
        # ("unknown", "failed", "migrating", "queued", "running_offloaded",
        # "running"); this receiver cell belongs to the
        # compute_task_execution_failed slice.
        ("unknown", "failed", "migrating", "queued", "running_offloaded", "running"),
    ("compute_state.jsonl", "deadline.status"):
        # state_dependencies.py:1736 declares the row enumeration
        # ("inactive", "within_budget", "overdue", "unknown");
        # compute_comm.py:1141 publishes the exact deadline object
        # (_build_compute_rows#deadline.status).
        ("inactive", "within_budget", "overdue", "unknown"),
})

# Declared identifier vocabularies, keyed by (authority, source).  An absent
# entry keeps the existing presence-only encoding: identity equality is
# decided by the rule compiler, never by an invented numeric code.
DECLARED_IDENTIFIERS = MappingProxyType({})

# Field identities that carry a declared ENU vector value: (declared candidate
# authority, exact canonical field).  state_dependencies.py declares each with
# data_type "vector3": l0 truth-frame and trajectory positions/velocities
# (:1913-1914,1921-1924,1944-1946,1976-1977), the aircraft geometry record
# (:2331-2345) and truth_units velocity (:2399).
DECLARED_VECTOR_FIELD_IDENTITIES = frozenset({
    ("l0:truth_frames.jsonl", "truth_pose.position_enu_m"),
    ("l0:truth_frames.jsonl", "truth_pose.velocity_enu_mps"),
    ("l0:truth_frames.jsonl", "station.position_enu_m"),
    ("l0:truth_frames.jsonl", "facility.position_enu_m"),
    ("l0:truth_frames.jsonl", "aircraft.truth_pose.position_enu_m"),
    ("l0:truth_frames.jsonl", "aircraft.truth_pose.velocity_enu_mps"),
    ("l0:trajectories.jsonl", "aircraft.position_enu_m"),
    ("l0:trajectories.jsonl", "aircraft.velocity_enu_mps"),
    ("l0:trajectories.jsonl", "first_aircraft.position_enu_m"),
    ("l0:trajectories.jsonl", "second_aircraft.position_enu_m"),
    ("l0:trajectories.jsonl", "aircraft.position_enu_m"),
    ("l0:source_episode/trajectories.jsonl", "aircraft.position_enu_m"),
    ("predicate_contract_aircraft_geometry", "geometry.position_enu_m"),
    ("predicate_contract_aircraft_structure_geometry", "aircraft.position_enu_m"),
    ("predicate_contract_aircraft_structure_geometry", "structure.position_enu_m"),
    ("predicate_contract_aircraft_vehicle_geometry", "aircraft.position_enu_m"),
    ("predicate_contract_aircraft_vehicle_geometry", "vehicle.position_enu_m"),
    ("predicate_contract_aircraft_pedestrian_geometry", "aircraft.position_enu_m"),
    ("predicate_contract_aircraft_pedestrian_geometry", "pedestrian.position_enu_m"),
    ("predicate_contract_vehicle_pedestrian_geometry", "vehicle.position_enu_m"),
    ("predicate_contract_vehicle_pedestrian_geometry", "pedestrian.position_enu_m"),
    ("predicate_contract_aircraft_pair_geometry", "first_aircraft.position_enu_m"),
    ("predicate_contract_aircraft_pair_geometry", "second_aircraft.position_enu_m"),
    ("predicate_contract_aircraft_corridor_geometry", "actor.position_enu_m"),
    ("predicate_contract_aircraft_region_geometry", "actor.position_enu_m"),
    ("predicate_contract_aircraft_protected_region_geometry", "aircraft.position_enu_m"),
})


def _frozen(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_frozen(item) for item in value)
    return value


def _array(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class FieldCoordinate:
    """Exact state identity, with an optional declared producer-row projection.

    ``authority`` is the exact declared candidate authority that produced the
    field: the formula contract's candidate authority or the direct producer's
    source artifact.  ``declared_source`` is the contract's exact grounding
    field_sources value (for example ``record:approval_status``) when one is
    declared, else empty.  Typed vocabulary lookups key on
    ``(authority, declared_source or schema.source)``: the resolved artifact
    path alone cannot distinguish two declared fields of one authority.  No
    lookup ever uses field-name similarity.
    """
    schema: FieldSchema
    owner: tuple[TypedEntity, ...]
    producer_artifact: str = ""
    producer_row_path: str = ""
    owner_roles: tuple[str, ...] = ()
    authority: str = ""
    declared_source: str = ""

    def __post_init__(self):
        if not self.owner or any(not isinstance(item, TypedEntity) for item in self.owner):
            raise ContractError("a field needs exact typed owners")
        if self.owner_roles and len(self.owner_roles) != len(self.owner):
            raise ContractError("owner role count differs from typed field owner count")
        if bool(self.producer_artifact) != bool(self.producer_row_path):
            raise ContractError("producer row projection needs both artifact and field path")
        if self.producer_artifact and self.schema.field != (
                f"source:{self.producer_artifact}:{self.producer_row_path}"):
            raise ContractError("producer field identity differs from exact source projection")

    @property
    def key(self):
        return self.schema.field, self.owner

    @property
    def components(self):
        """Declared encoded components: exact vector axes or the scalar."""
        return DECLARED_VECTORS.get(self.schema.data_type, ("",))


@dataclass(frozen=True)
class ProducerOwnerBinding:
    """Caller-established prefix authority linking predicate roles to row owners.

    coordinates follow the exact declared builder row index; role_bindings are
    the exact predicate argument bindings. source_ref names the admitted relation
    evidence and availability_tick states when that relation became observable.
    Supplied operand rows cannot override this owner authority.
    """
    row_builder: str
    coordinates: tuple[RoleBinding, ...]
    role_bindings: tuple[RoleBinding, ...]
    source_ref: str
    availability_tick: int

    def __post_init__(self):
        if not self.source_ref or type(self.availability_tick) is not int or self.availability_tick < 0:
            raise ContractError("producer owner relation requires source and availability time")
        if len({item.role for item in self.coordinates}) != len(self.coordinates):
            raise ContractError("duplicate producer owner coordinate")

    @property
    def owner(self):
        return tuple(item.entity for item in self.coordinates)


@dataclass(frozen=True)
class PredicateQuery:
    contract: PredicateContract
    bindings: tuple[RoleBinding, ...]
    parameter_profile: ParameterProfile
    applicable: bool = True
    source_out_of_scope: bool = False
    producer_owner_binding: ProducerOwnerBinding | None = None

    def __post_init__(self):
        if not self.contract.executable:
            raise ContractError("trajectory query requires an executable contract")
        resolved = bind_roles(self.contract, {item.role: item.entity for item in self.bindings})
        object.__setattr__(self, "bindings", canonical_role_bindings(self.contract, resolved))
        if type(self.applicable) is not bool or type(self.source_out_of_scope) is not bool:
            raise ContractError("query scope status must be explicit booleans")
        absent = set(self.contract.threshold_parameters) - set(self.parameter_profile.parameters)
        if absent:
            raise ContractError(f"missing governed parameters: {sorted(absent)}")

        relation = self.producer_owner_binding
        if relation is not None:
            authority = self.contract.grounding_spec["candidate_authority"]["source"]
            if authority not in DIRECT_PREDICATE_PRODUCERS:
                raise ContractError("owner row relation requires a declared direct producer")
            producer = DIRECT_PREDICATE_PRODUCERS[authority]
            binding = direct_predicate_bindings(authority)[self.contract.id]
            if relation.row_builder != binding.row_builder or relation.role_bindings != self.bindings:
                raise ContractError("producer row relation differs from exact predicate role bindings")
            if tuple(item.role for item in relation.coordinates) != builder_row_index(
                    producer.module, binding.row_builder).keys:
                raise ContractError("producer owner coordinates differ from declared row index")
            from .state_dependencies import COMPUTE_COMM
            if producer.module != COMPUTE_COMM:
                raise ContractError("producer owner classes lack an executable binding authority")
            class_gates = {
                "_build_communication_rows": {
                    "actor": ("world:UnmannedAircraft",), "station": ("world:CommunicationStation",)},
                "_build_compute_rows": {
                    "actor": ("world:UnmannedAircraft", "world:GroundVehicle")},
            }
            if binding.row_builder not in class_gates:
                raise ContractError("producer row has no declared owner type gate")
            if any(not any(item.entity.supports_class(expected) for expected in
                    class_gates[binding.row_builder][item.role]) for item in relation.coordinates):
                raise ContractError("producer owner type conflicts with its declared coordinate")

    @property
    def owner(self):
        return tuple(item.entity for item in self.bindings)

    @property
    def producer_owner(self):
        return None if self.producer_owner_binding is None else self.producer_owner_binding.owner


def formula_fields(query: PredicateQuery) -> tuple[FieldCoordinate, ...]:
    """Resolve formula fields once from exact contract ownership declarations.

    Each coordinate carries the contract's exact grounding field_sources value
    as ``declared_source``; the template's resolved artifact path stays on
    ``schema.source`` as provenance only, never as a vocabulary key.
    """
    if query.contract.semantic_mode != "declared_formula":
        raise ContractError("formula field assembly requires a declared formula")
    authority = query.contract.grounding_spec["candidate_authority"]["source"]
    sources = query.contract.grounding_spec["field_sources"]
    return tuple(FieldCoordinate(
        FieldSchema(canonical_state_field(query.contract, schema.field), schema.data_type,
                    schema.unit, schema.source, schema.categories),
        _owner(query.contract, query.bindings, field_owner_roles(query.contract, schema.field)),
        owner_roles=field_owner_roles(query.contract, schema.field), authority=authority,
        declared_source=sources[schema.field])
        for schema in query.contract.required_fields)


def direct_fields(query: PredicateQuery) -> tuple[FieldCoordinate, ...]:
    """Use the existing immediate receiver monitor contract, without guessing.

    A producer with no immediate receiver projection must instead supply an
    explicit ProducerRowTrajectory. Missing projections remain visible errors.
    """
    from .state_dependencies import direct_monitor_fields
    if query.producer_owner_binding is None:
        raise ContractError("direct fields require a prefix-established producer owner binding")
    _, nodes = direct_monitor_fields(query.contract)
    result = []
    for node in nodes:
        cell = node.source_cell
        result.append(FieldCoordinate(
            FieldSchema(f"source:{cell.artifact}:{cell.field_path}", node.data_type,
                        node.unit or "nonmeasurement", node.node_id,
                        tuple(node.domain.values)),
            tuple(dict((item.role, item.entity) for item in query.producer_owner_binding.coordinates)[role]
                  for role in node.owner_keys),
            cell.artifact, cell.field_path, tuple(node.owner_keys), cell.artifact, cell.field_path))
    return tuple(result)


@dataclass(frozen=True)
class EpisodeContract:
    """Positional episode/branch/time authority for all supplied trajectories.

    History axis positions map to history_ticks. Candidate time positions map
    to anchor_tick + STATE_OFFSETS: position zero is the observed anchor, and
    positions 1..10 are future instants 5..50 ticks later. No PMF tail timestamp
    is appended. Embedded operand-row metadata does not override this authority.
    """
    episode_id: str
    branch_id: str
    split: str
    history_ticks: tuple[int, ...]
    anchor_tick: int
    fields: tuple[FieldCoordinate, ...]
    queries: tuple[PredicateQuery, ...]

    def __post_init__(self):
        if not self.episode_id or not self.branch_id or self.split not in {"train", "validation"}:
            raise ContractError("trajectory assembly admits explicit TRAIN/VALID episodes only")
        if (type(self.anchor_tick) is not int or self.anchor_tick < 0 or not self.history_ticks
                or tuple(sorted(set(self.history_ticks))) != self.history_ticks
                or self.history_ticks[-1] != self.anchor_tick
                or any(type(tick) is not int or tick < 0 for tick in self.history_ticks)):
            raise ContractError("observed history must end at its nonnegative anchor")
        if not self.fields or not self.queries:
            raise ContractError("episode must declare fields and predicate queries")
        if any(query.producer_owner_binding is not None and
               query.producer_owner_binding.availability_tick > self.anchor_tick for query in self.queries):
            raise ContractError("producer owner relation is unavailable in the observed prefix")
        if len({item.key for item in self.fields}) != len(self.fields):
            raise ContractError("duplicate exact field and owner coordinate")

    @property
    def future_ticks(self):
        return tuple(self.anchor_tick + offset for offset in STATE_OFFSETS)


@dataclass(frozen=True)
class ProducerRowTrajectory:
    """Complete declared producer row operands for one fixed query.

    Missing rows are None. Missing required operands retain UNKNOWN; supplied
    values retain the producer's own validation and missing-value semantics. These are
    operand rows, never future predicate labels. The outer axis is candidates;
    each inner axis follows EpisodeContract.future_ticks exactly. The positional
    episode clock and typed ProducerOwnerBinding are authoritative; embedded
    operand-row episode/tick/identity columns are not a second binding channel.
    """
    query_index: int
    artifact: str
    owner: tuple[TypedEntity, ...]
    rows: tuple[tuple[Mapping[str, Any] | None, ...], ...]
    provenance: str = "predicted_future_operands"

    def __post_init__(self):
        if self.provenance != "predicted_future_operands":
            raise ContractError("producer trajectories require predicted operand provenance")
        if type(self.query_index) is not int or self.query_index < 0 or not self.artifact:
            raise ContractError("producer operand query and artifact must be explicit")
        object.__setattr__(self, "rows", _frozen(self.rows))


@dataclass(frozen=True)
class CandidateTrajectory:
    episode: EpisodeContract
    history: np.ndarray
    history_known: np.ndarray
    sampled_states: np.ndarray
    sampled_known: np.ndarray
    producer_rows: tuple[ProducerRowTrajectory, ...] = ()

    def __post_init__(self):
        for name in ("history", "history_known", "sampled_states", "sampled_known"):
            object.__setattr__(self, name, _array(getattr(self, name),
                bool if name.endswith("known") else None))
        h, f = len(self.episode.history_ticks), len(self.episode.fields)
        if self.history.shape != (h, f) or self.history_known.shape != (h, f):
            raise ContractError("history shape differs from episode time/field contract")
        if (self.sampled_states.ndim != 3 or self.sampled_states.shape[1:] != (11, f)
                or self.sampled_states.shape[0] < 1
                or self.sampled_known.shape != self.sampled_states.shape):
            raise ContractError("sampled states must be [K,11,F] with the same known mask")
        # The anchor is observed prefix evidence, never a freshly sampled state.
        # Object cells (declared vector tuples) compare by exact value equality.
        for column in range(f):
            if self.history_known[-1, column]:
                same = all(self.sampled_states[sample, 0, column] == self.history[-1, column]
                           for sample in range(self.sampled_states.shape[0]))
                if not np.all(self.sampled_known[:, 0, column]) or not same:
                    raise ContractError("sampled anchor differs from the observed prefix")
            elif np.any(self.sampled_known[:, 0, column]):
                raise ContractError("missing observed anchor cannot become predicted truth evidence")
        if len({item.query_index for item in self.producer_rows}) != len(self.producer_rows):
            raise ContractError("duplicate producer row trajectory")
        for item in self.producer_rows:
            if item.query_index >= len(self.episode.queries):
                raise ContractError("producer trajectory query is outside episode contract")
            query = self.episode.queries[item.query_index]
            authority = query.contract.grounding_spec["candidate_authority"]["source"]
            if item.artifact != authority or item.owner != query.producer_owner:
                raise ContractError("producer operand authority/owner differs from query binding")
            if len(item.rows) != self.sampled_states.shape[0] or any(len(row) != 11 for row in item.rows):
                raise ContractError("producer rows must share candidate count and state clock")
            if query.producer_owner_binding is None:
                raise ContractError("producer rows require a prefix-established owner relation")
            if any(row[0] is not None for row in item.rows):
                raise ContractError("predicted producer rows cannot establish observed anchor evidence")


@dataclass(frozen=True)
class ProducerRowKey:
    """Exact typed source index; raw row columns do not select its identity."""
    candidate_index: int
    episode_id: str
    branch_id: str
    tick: int
    owner: tuple[TypedEntity, ...]

    def __post_init__(self):
        if (type(self.candidate_index) is not int or self.candidate_index < 0
                or type(self.tick) is not int or self.tick < 0
                or not self.episode_id or not self.branch_id or not self.owner
                or any(not isinstance(entity, TypedEntity) for entity in self.owner)):
            raise ContractError("producer source index requires exact candidate/episode/time/typed owners")


def producer_rows_from_index(episode: EpisodeContract, query_index: int,
        candidate_count: int, rows: Mapping[ProducerRowKey, Mapping[str, Any]]) -> ProducerRowTrajectory:
    """Join already indexed source operands to the immutable episode clock.

    The source loader owns the typed source index. This adapter performs direct
    lookups of expected keys, never scans row payloads or source repositories.
    Missing exact keys yield missing operands; another episode/time/owner cannot
    enter a query via an approximate join. The anchor remains prefix evidence.
    """
    if type(query_index) is not int or not 0 <= query_index < len(episode.queries):
        raise ContractError("producer join query is outside the episode contract")
    if type(candidate_count) is not int or candidate_count < 1:
        raise ContractError("producer join requires a positive candidate count")
    query = episode.queries[query_index]
    if query.producer_owner_binding is None:
        raise ContractError("producer join requires declared typed source owner coordinates")
    return ProducerRowTrajectory(query_index,
        query.contract.grounding_spec["candidate_authority"]["source"], query.producer_owner,
        tuple((None,) + tuple(rows.get(ProducerRowKey(candidate, episode.episode_id,
            episode.branch_id, tick, query.producer_owner)) for tick in episode.future_ticks[1:])
            for candidate in range(candidate_count)))


@dataclass(frozen=True)
class _QueryPlan:
    query: PredicateQuery
    compiled: Any
    inputs: tuple[tuple[str, int | None], ...]
    previous_fields: tuple[str, ...]
    producer_function: Any = None
    producer_inputs: tuple[tuple[str, int | None], ...] = ()
    required_row_keys: tuple[str, ...] = ()
    receiver_complete: bool = True


class FuturePredicateEvaluator:
    """Prepare source bindings once; evaluate only fixed numeric rule operations."""
    def __init__(self, episode: EpisodeContract):
        self.episode = episode
        by_key = {item.key: index for index, item in enumerate(episode.fields)}
        plans, incidence = [], np.zeros((len(episode.queries), len(episode.fields)), bool)
        for p, query in enumerate(episode.queries):
            compiled = compile_rule(query.contract)
            inputs, producer_inputs, function, required_row_keys, receiver_complete = [], [], None, (), True
            if query.contract.semantic_mode == "declared_formula":
                for schema in query.contract.required_fields:
                    coordinate = (canonical_state_field(query.contract, schema.field),
                        _owner(query.contract, query.bindings, field_owner_roles(query.contract, schema.field)))
                    index = by_key.get(coordinate)
                    inputs.append((schema.field, index))
                    if index is not None:
                        incidence[p, index] = True
            else:
                authority = query.contract.grounding_spec["candidate_authority"]["source"]
                if authority not in DIRECT_PREDICATE_PRODUCERS:
                    raise ContractError(f"direct predicate has no declared producer binding: {authority}")
                producer = DIRECT_PREDICATE_PRODUCERS[authority]
                binding = direct_predicate_bindings(authority)[query.contract.id]
                function = getattr(import_module(producer.module[:-3].replace("/", ".")), binding.function)
                nodes, receiver_complete = _receiver_operand_nodes(query.contract)
                for node in nodes:
                    cell = node.source_cell
                    index = by_key.get((f"source:{cell.artifact}:{cell.field_path}", query.producer_owner))
                    producer_inputs.append((cell.field_path, index))
                    if index is not None:
                        incidence[p, index] = True
                required_row_keys = _declared_row_keys(producer.module, binding.function)
            plans.append(_QueryPlan(query, compiled, tuple(inputs), compiled.previous_fields,
                                    function, tuple(producer_inputs), required_row_keys, receiver_complete))
        self.plans = tuple(plans)
        self.predicate_field_incidence = _array(incidence)
        self.expanded_predicate_field_incidence = _expanded_incidence(
            incidence, episode.fields)

    def evaluate(self, record: CandidateTrajectory):
        if record.episode is not self.episode:
            raise ContractError("evaluator requires its prepared immutable episode contract")
        k = record.sampled_states.shape[0]
        codes = np.full((k, 11, len(self.plans)), TRUTH_CODES["unknown"], np.int8)
        rows = {item.query_index: item for item in record.producer_rows}
        for p, plan in enumerate(self.plans):
            query = plan.query
            if not query.applicable or query.source_out_of_scope:
                codes[:, :, p] = TRUTH_CODES["out_of_scope"]
                continue
            for sample in range(k):
                for time in range(11):
                    if plan.producer_function is not None:
                        if p in rows and time > 0:
                            row = rows[p].rows[sample][time]
                        elif plan.receiver_complete and plan.producer_inputs and all(index is not None and record.sampled_known[sample, time, index]
                                for _, index in plan.producer_inputs):
                            row = {}
                            for path, index in plan.producer_inputs:
                                set_field(row, path, record.sampled_states[sample, time, index].item()
                                    if isinstance(record.sampled_states[sample, time, index], np.generic)
                                    else record.sampled_states[sample, time, index])
                        else:
                            row = None
                        truth = "unknown" if row is None or any(key not in row for key in plan.required_row_keys) else plan.producer_function(row)
                    else:
                        context, previous = {}, {}
                        for path, index in plan.inputs:
                            if index is not None and record.sampled_known[sample, time, index]:
                                value = record.sampled_states[sample, time, index]
                                set_field(context, path, value.item() if isinstance(value, np.generic) else value)
                            if path in plan.previous_fields and index is not None:
                                if time > 0 and record.sampled_known[sample, time-1, index]:
                                    set_field(previous, path, record.sampled_states[sample, time-1, index].item()
                                        if isinstance(record.sampled_states[sample, time-1, index], np.generic)
                                        else record.sampled_states[sample, time-1, index])
                                elif time == 0 and len(record.episode.history_ticks) > 1 and record.history_known[-2, index]:
                                    set_field(previous, path, record.history[-2, index].item()
                                        if isinstance(record.history[-2, index], np.generic) else record.history[-2, index])
                        # Missing previous observations censor changed instead of inventing False.
                        truth = "unknown" if any(path not in _flatten(previous) for path in plan.previous_fields) else (
                            plan.compiled.evaluate(context, previous, query.parameter_profile.parameters))
                    if truth not in TRUTH_CODES:
                        raise ContractError("declared producer returned invalid four-valued truth")
                    codes[sample, time, p] = TRUTH_CODES[truth]
        return _array(codes)


def _receiver_operand_nodes(contract):
    """Resolve immediate receiver operands from the existing dependency graph.

    A source root lacking a receiver projection leaves the rule unavailable to
    the array adapter. Explicit full producer rows can still invoke its function.
    """
    from .state_dependencies import predicate_dependency_spec
    spec = predicate_dependency_spec(contract)
    nodes = {node.node_id: node for node in spec.nodes}
    upstream, projections = {}, {}
    for edge in spec.edges:
        upstream.setdefault(edge.target, []).append(edge.source)
    for node in spec.nodes:
        cell = node.source_cell
        if cell is not None and cell.form == "published_monitor" and cell.coherent_parent is not None:
            projections.setdefault(cell.coherent_parent, []).append(node.node_id)
    pending, visited, operands, complete = [spec.predicate_node_id], set(), {}, True
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = nodes[node_id]
        cell = node.source_cell
        if cell is not None and cell.form in (
                "published_monitor", "governed_row_parameter", "publisher_requirement", "unobserved_monitor"):
            operands[cell.field_path] = node
        elif node_id in projections:
            pending.extend(projections[node_id])
        else:
            if node_id != spec.predicate_node_id and not upstream.get(node_id):
                complete = False
            pending.extend(upstream.get(node_id, ()))
    return tuple(operands[key] for key in sorted(operands)), complete


def _declared_row_keys(module, function):
    """Extract required row keys from the already parsed declared producer AST.

    This determines operand availability only. Truth remains computed by the
    original producer function, including its own optional/missing semantics.
    """
    import ast
    from .state_dependencies import _function
    keys, visited = set(), set()
    def visit(name):
        if name in visited:
            return
        visited.add(name)
        node = _function(module, name)
        for child in ast.walk(node):
            if (isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name)
                    and child.value.id == "row" and isinstance(child.slice, ast.Constant)
                    and isinstance(child.slice.value, str)):
                keys.add(child.slice.value)
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id.startswith("_truth_from_")
                    and any(isinstance(arg, ast.Name) and arg.id == "row" for arg in child.args)):
                visit(child.func.id)
    visit(function)
    return tuple(sorted(keys))


def _flatten(mapping, prefix=""):

    result = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            result.update(_flatten(value, path))
        else:
            result[path] = value
    return result


def _scalar_value(coordinate: FieldCoordinate, value) -> float:
    """Type-aware scalar encoding of one known observed value."""
    schema = coordinate.schema
    if schema.data_type == "numeric":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
            raise ContractError(f"known numeric state is invalid: {schema.field}")
        return float(value)
    if schema.data_type == "integer":
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise ContractError(f"known integer state is invalid: {schema.field}")
        return float(value)
    if schema.data_type == "boolean":
        if not isinstance(value, (bool, np.bool_)):
            raise ContractError(f"known boolean state is invalid: {schema.field}")
        return float(value)
    if schema.data_type == "categorical":
        vocabulary = _declared_categories(coordinate)
        if not isinstance(value, str) or value not in vocabulary:
            raise ContractError(f"categorical state lacks an exact declared code: {schema.field}")
        return float(vocabulary.index(value))
    if schema.data_type == "identifier":
        return _identifier_value(coordinate, value)
    if schema.data_type == "four_value":
        if value not in TRUTH_DOMAIN:
            raise ContractError(f"four-valued state lacks an exact declared code: {schema.field}")
        return float(TRUTH_DOMAIN.index(value))
    raise ContractError(f"scalar encoding requires a scalar state type: {schema.data_type}")


def _category_key(coordinate: FieldCoordinate) -> tuple[str, str]:
    """Exact typed vocabulary identity for one field coordinate."""
    return (coordinate.authority, coordinate.declared_source or coordinate.schema.source)


def _declared_categories(coordinate: FieldCoordinate) -> tuple[str, ...]:
    """Exact declared category vocabulary for one categorical field identity."""
    key = _category_key(coordinate)
    if key not in DECLARED_CATEGORIES:
        raise ContractError(f"categorical field has no declared vocabulary: {key}")
    declared = DECLARED_CATEGORIES[key]
    tested = coordinate.schema.categories
    if any(code not in declared for code in tested):
        raise ContractError(f"evaluation AST tests a code outside the declared vocabulary: {key}")
    return declared


def _identifier_value(coordinate: FieldCoordinate, value) -> float:
    """Declared-vocabulary identifier encoding; presence only when undeclared.

    Identity equality stays a rule-compiler concern: without a declared
    vocabulary the encoding is the invariant presence feature 1.0, and the
    equality semantics remain those of the original value held by the rule
    evaluator. Renumbering an identifier never changes the encoding.
    """
    if not isinstance(value, str) or not value:
        raise ContractError(f"known identifier state is invalid: {coordinate.schema.field}")
    vocabulary = DECLARED_IDENTIFIERS.get(
        (coordinate.authority, coordinate.declared_source or coordinate.schema.source))
    if vocabulary is None:
        return 1.0
    if value not in vocabulary:
        raise ContractError(f"identifier state lacks an exact declared identity: {coordinate.schema.field}")
    return float(vocabulary.index(value))


def project_typed_fields(values: np.ndarray, known: np.ndarray,
        fields: Sequence[FieldCoordinate]) -> tuple[np.ndarray, np.ndarray]:
    """Setup-time typed projection of canonical states to encoded components.

    Returns (component_values, component_known).  Each field contributes its
    declared components: three ENU axes for a declared vector, one scalar for
    every other declared type.  A known vector cell must be an exact declared
    component tuple; any missing/unknown component censors all components of
    that cell (a vector is one declared state, never a per-axis truth).
    Unknown cells stay NaN with known=False; UNKNOWN is never observed 0 and
    is never filled, imputed or reduced to a guessed norm.
    """
    if values.shape != known.shape:
        raise ContractError("typed projection needs aligned value and known shapes")
    if values.shape[-1] != len(fields):
        raise ContractError("typed projection field axis differs from declared fields")
    widths = [_component_width(field) for field in fields]
    offsets = np.cumsum([0] + widths[:-1])
    total = int(sum(widths))
    projected = np.full(values.shape[:-1] + (total,), np.nan, np.float64)
    projected_known = np.zeros(values.shape[:-1] + (total,), bool)
    for index, coordinate in enumerate(fields):
        width = widths[index]
        vector = coordinate.schema.data_type in DECLARED_VECTORS
        for position in np.ndindex(values.shape[:-1]):
            at = position + (index,)
            if not known[at]:
                continue
            value = values[at]
            if vector:
                if _is_vector_identity(coordinate) and not (
                        isinstance(value, (tuple, list)) and len(value) == len(
                        DECLARED_VECTORS[coordinate.schema.data_type])
                        and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                                and math.isfinite(item) for item in value)):
                    raise ContractError(
                        f"known vector state is not a declared finite component tuple: {schema_field(coordinate)}")
                components = tuple(float(item) for item in value)
            else:
                components = (_scalar_value(coordinate, value),)
            base = offsets[index]
            for axis, item in enumerate(components):
                projected[position + (base + axis,)] = item
                projected_known[position + (base + axis,)] = True
    return _array(projected), _array(projected_known)


def schema_field(coordinate: FieldCoordinate) -> str:
    return coordinate.schema.field


def _component_width(field: FieldCoordinate) -> int:
    if field.schema.data_type in DECLARED_VECTORS:
        if not _is_vector_identity(field):
            raise ContractError(
                f"vector state lacks a declared component identity: {field.schema.field}")
        return len(DECLARED_VECTORS[field.schema.data_type])
    return 1


def _is_vector_identity(field: FieldCoordinate) -> bool:
    return (field.authority, field.schema.field) in DECLARED_VECTOR_FIELD_IDENTITIES


def field_component_layout(fields: Sequence[FieldCoordinate]) -> tuple[
        tuple[tuple[str, str, tuple[str, ...]], ...], tuple[int, ...]]:
    """Compiled per-field component identity and value-axis offsets.

    The identity is the exact (canonical field, data type, declared component
    axes); empty axes mean the scalar encoding. Offsets are per canonical
    field, before expansion.
    """
    return tuple((field.schema.field, field.schema.data_type, field.components)
                 for field in fields), tuple(np.cumsum([0] + [_component_width(f) for f in fields])[:-1])


def expanded_field_coordinates(fields: Sequence[FieldCoordinate]) -> tuple[FieldCoordinate, ...]:
    """Per-component field coordinates with the original canonical identity.

    Scalar fields pass through unchanged.  Each vector field contributes one
    coordinate per declared ENU axis; owner, roles and authority are the
    original field's, so exact owner/role identity is preserved per component.
    """
    result: list[FieldCoordinate] = []
    for field in fields:
        if field.schema.data_type in DECLARED_VECTORS:
            if not _is_vector_identity(field):
                raise ContractError(
                    f"vector state lacks a declared component identity: {field.schema.field}")
            for axis in DECLARED_VECTORS[field.schema.data_type]:
                result.append(FieldCoordinate(
                    FieldSchema(f"{field.schema.field}#{axis}", "numeric", field.schema.unit,
                                field.schema.source, ()),
                    field.owner, field.producer_artifact, field.producer_row_path,
                    field.owner_roles, field.authority, field.declared_source))
        else:
            result.append(field)
    return tuple(result)


def _numeric(values, known, fields):
    """Scalar contract encoding, identical to the previous released behavior.

    L/P cached receiver fields are declared numeric scalars; their encodings
    and widths are unchanged.  Non-scalar declared types use the typed
    projection's exact codes.
    """
    result = np.full(values.shape, np.nan, np.float64)
    for index, coordinate in enumerate(fields):
        schema = coordinate.schema
        if schema.data_type in DECLARED_VECTORS:
            raise ContractError("vector states need the typed component projection: "
                                + schema_field(coordinate))
        for position in np.ndindex(values.shape[:-1]):
            at = position + (index,)
            if not known[at]:
                continue
            result[at] = _scalar_value(coordinate, values[at])
    return _array(result)


@dataclass(frozen=True)
class AssembledTrajectoryBatch:
    arrays: Mapping[str, np.ndarray]
    episodes: tuple[EpisodeContract, ...]
    encoded_fields: tuple[FieldCoordinate, ...] = ()

    @property
    def fields(self):
        return self.episodes[0].fields

    @property
    def component_fields(self):
        """Encoded per-component fields; equals the canonical fields without vectors."""
        return self.encoded_fields or self.episodes[0].fields

    def as_mapping(self):
        return {**self.arrays, "episodes": self.episodes,
            "field_metadata": self.episodes[0].fields,
            "component_field_metadata": self.component_fields,
            "query_metadata": self.episodes[0].queries,
            "truth_domain": TRUTH_DOMAIN, "state_offsets": STATE_OFFSETS}


def _expanded_incidence(incidence: np.ndarray, fields: Sequence[FieldCoordinate]) -> np.ndarray:
    """Expand predicate-to-field incidence over encoded vector components."""
    widths = [_component_width(field) for field in fields]
    if all(width == 1 for width in widths):
        return incidence
    expanded = np.zeros((incidence.shape[0], int(sum(widths))), bool)
    offsets = np.cumsum([0] + widths[:-1])
    for index, width in enumerate(widths):
        expanded[:, offsets[index]:offsets[index] + width] = incidence[:, index, None]
    return expanded


def assemble_trajectory_batch(records: Sequence[CandidateTrajectory], *,
        evaluators: Sequence[FuturePredicateEvaluator] | None = None) -> AssembledTrajectoryBatch:
    if not records:
        raise ContractError("cannot assemble an empty trajectory batch")
    records = tuple(records)
    if len({record.episode.split for record in records}) != 1:
        raise ContractError("TRAIN and VALID must be assembled separately")
    evaluators = tuple(FuturePredicateEvaluator(record.episode) for record in records) if evaluators is None else tuple(evaluators)
    if len(evaluators) != len(records):
        raise ContractError("prepared evaluator count differs from records")
    first = records[0]
    signature = lambda record: (tuple((item.schema, item.producer_artifact, item.producer_row_path,
        tuple(owner.ontology_class for owner in item.owner), item.authority) for item in record.episode.fields),
        tuple(item.contract.id for item in record.episode.queries), record.history.shape, record.sampled_states.shape)
    if any(signature(record) != signature(first) for record in records):
        raise ContractError("batch schema and candidate/time dimensions differ")
    incidences = tuple(evaluator.predicate_field_incidence for evaluator in evaluators)
    if any(not np.array_equal(item, incidences[0]) for item in incidences):
        raise ContractError("batch predicate-field topology differs")
    fields = first.episode.fields
    projected_history = [project_typed_fields(r.history, r.history_known, fields) for r in records]
    projected_states = [project_typed_fields(r.sampled_states, r.sampled_known, fields) for r in records]
    expanded_fields = expanded_field_coordinates(fields)
    incidence = _expanded_incidence(incidences[0], fields)
    codes = _array(np.stack([evaluator.evaluate(record) for evaluator, record in zip(evaluators, records)]))
    known = _array((codes == TRUTH_CODES["true"]) | (codes == TRUTH_CODES["false"]))
    arrays = {
        "history": _array(np.stack([item[0] for item in projected_history])),
        "sampled_states": _array(np.stack([item[0] for item in projected_states])),
        "history_known": _array(np.stack([item[1] for item in projected_history])),
        "state_known": _array(np.stack([item[1] for item in projected_states])),
        "raw_history_known": _array(np.stack([r.history_known for r in records])),
        "raw_state_known": _array(np.stack([r.sampled_known for r in records])),
        "sampled_truth": _array(codes == TRUTH_CODES["true"]), "truth_codes": codes,
        "truth_known": known, "applicable": _array(codes != TRUTH_CODES["out_of_scope"]),
        "censored": _array(~known), "predicate_field_incidence": incidence,
        "field_component_identity": _array(np.array([list(identity) for identity in
            field_component_layout(fields)[0]], dtype=object)) if fields else _array(np.empty((0, 3), object)),
        "field_component_offsets": _array(field_component_layout(fields)[1]),
    }
    return AssembledTrajectoryBatch(MappingProxyType(arrays), tuple(r.episode for r in records),
                                    expanded_fields)


@dataclass(frozen=True)
class TrajectoryPygResult:
    graph: Any
    metadata: Mapping[str, Any]


def to_trajectory_pyg(record: CandidateTrajectory, *, evaluator: FuturePredicateEvaluator | None = None) -> TrajectoryPygResult:
    """Create an in-memory typed incidence graph from the same fixed-rule batch.

    This graph adapter is callable independently of the dense training forward.
    Identities and declared schema/roles stay in metadata. No graph is saved.
    Encoded vector components appear as their own FieldCoordinate nodes with
    the exact canonical field identity plus the declared axis suffix, sharing
    the original typed owner and roles.
    """
    import torch
    from .joint_state_pyg import JointStateGraph, _edges
    evaluator = FuturePredicateEvaluator(record.episode) if evaluator is None else evaluator
    batch = assemble_trajectory_batch((record,), evaluators=(evaluator,)).as_mapping()
    episode = record.episode
    k, t, f = record.sampled_states.shape
    component_fields = batch["component_field_metadata"]
    c = len(component_fields)
    p = len(episode.queries)
    graph = JointStateGraph()
    owners = tuple(dict.fromkeys(owner for item in episode.fields for owner in item.owner))
    owners += tuple(owner for query in episode.queries for owner in query.owner if owner not in owners)
    owners = tuple(dict.fromkeys(owners))
    owner_index = {owner: index for index, owner in enumerate(owners)}
    roles = tuple(sorted({role for item in episode.fields for role in item.owner_roles}
        | {item.role for query in episode.queries for item in query.bindings}))
    graph["TypedOwner"].num_nodes = len(owners)
    graph["FieldCoordinate"].num_nodes = c
    graph["PredicateQuery"].num_nodes = p
    pairs, positions, role_codes = [], [], []
    for index, item in enumerate(component_fields):
        for position, owner in enumerate(item.owner):
            pairs.append((owner_index[owner], index)); positions.append(position)
            role_codes.append(roles.index(item.owner_roles[position]) if item.owner_roles else -1)
    edge = graph["TypedOwner", "owns_field", "FieldCoordinate"]
    edge.edge_index = _edges(pairs)
    edge.role_position = torch.tensor(positions, dtype=torch.long)
    edge.role_code = torch.tensor(role_codes, dtype=torch.long)
    pairs, positions, role_codes = [], [], []
    for index, query in enumerate(episode.queries):
        for position, binding in enumerate(query.bindings):
            pairs.append((owner_index[binding.entity], index)); positions.append(position)
            role_codes.append(roles.index(binding.role))
    edge = graph["TypedOwner", "binds_predicate_role", "PredicateQuery"]
    edge.edge_index = _edges(pairs)
    edge.role_position = torch.tensor(positions, dtype=torch.long)
    edge.role_code = torch.tensor(role_codes, dtype=torch.long)
    graph["FieldCoordinate", "operand_of", "PredicateQuery"].edge_index = _edges(
        [(component_index, query_index) for query_index, component_index in
         np.argwhere(evaluator.expanded_predicate_field_incidence)])
    state = graph["CandidateState"]
    state.num_nodes = k*t*c
    state.value = torch.tensor(batch["sampled_states"][0].reshape(-1), dtype=torch.float64)
    state.known = torch.tensor(batch["state_known"][0].reshape(-1), dtype=torch.bool)
    state.candidate_position = torch.tensor(np.repeat(np.arange(k),t*c), dtype=torch.long)
    state.sample_relative_tick = torch.tensor(np.tile(np.repeat(STATE_OFFSETS,c),k), dtype=torch.long)
    state.predicted_future = state.sample_relative_tick > 0
    graph["FieldCoordinate", "has_candidate_state", "CandidateState"].edge_index = _edges(
        [(field_index,index) for index,field_index in enumerate(np.tile(np.arange(c),k*t))])
    result = graph["FixedRuleResult"]
    result.num_nodes = k*t*p
    result.truth_code = torch.tensor(batch["truth_codes"][0].reshape(-1), dtype=torch.long)
    result.known = torch.tensor(batch["truth_known"][0].reshape(-1), dtype=torch.bool)
    result.applicable = torch.tensor(batch["applicable"][0].reshape(-1), dtype=torch.bool)
    result.censored = torch.tensor(batch["censored"][0].reshape(-1), dtype=torch.bool)
    result.candidate_position = torch.tensor(np.repeat(np.arange(k),t*p), dtype=torch.long)
    result.sample_relative_tick = torch.tensor(np.tile(np.repeat(STATE_OFFSETS,p),k), dtype=torch.long)
    graph["PredicateQuery", "has_fixed_result", "FixedRuleResult"].edge_index = _edges(
        [(query_index,index) for index,query_index in enumerate(np.tile(np.arange(p),k*t))])
    graph["CandidateState", "supports_fixed_result", "FixedRuleResult"].edge_index = _edges(
        [((sample*t+time)*c+field_index,(sample*t+time)*p+query_index)
         for sample in range(k) for time in range(t)
         for query_index,field_index in np.argwhere(evaluator.expanded_predicate_field_incidence)])
    observed = graph["ObservedPrefixState"]
    observed.num_nodes = batch["history"][0].size
    observed.value = torch.tensor(batch["history"][0].reshape(-1), dtype=torch.float64)
    observed.known = torch.tensor(batch["history_known"][0].reshape(-1), dtype=torch.bool)
    observed.sample_relative_tick = torch.tensor(np.repeat(
        np.asarray(episode.history_ticks)-episode.anchor_tick,c), dtype=torch.long)
    graph["FieldCoordinate", "has_observed_prefix", "ObservedPrefixState"].edge_index = _edges(
        [(field_index,index) for index,field_index in enumerate(np.tile(np.arange(c),len(episode.history_ticks)))])
    return TrajectoryPygResult(graph, MappingProxyType({"episode":episode,
        "owners":owners,"roles":roles,"fields":episode.fields,
        "component_fields":component_fields,
        "field_component_identity":batch["field_component_identity"],
        "field_component_offsets":batch["field_component_offsets"],
        "queries":episode.queries,
        "branch_id":episode.branch_id,"state_ticks":episode.future_ticks,
        "truth_domain":TRUTH_DOMAIN,"producer_rows":record.producer_rows}))


def check_typed_projection_contract() -> dict:
    """Reusable structural regression for the typed assembly contract (CPU only).

    Exercises declared-vector component projection, exact declared category
    and identifier encodings, UNKNOWN censoring and the unchanged scalar L/P
    numeric encoding on a real executable contract.  No neural forward, no
    data files, no source scans beyond module imports.
    """
    import numpy as np
    from .unified_contracts import FieldSchema, load_unified_contracts
    from .generic_grounding import ParameterProfile, RoleBinding, TypedEntity

    contract = load_unified_contracts(executable_only=True)["ground.vehicle_moving_in_lane"]
    vehicle = TypedEntity("veh-check", "world:GroundVehicle")
    lane = TypedEntity("lane-check", "world:RoadLane")
    query = PredicateQuery(contract, (RoleBinding("vehicle", vehicle), RoleBinding("lane", lane)),
                           ParameterProfile("check", {"vehicle_stopped_speed_threshold_mps": 0.5}, "check"))
    vector = FieldCoordinate(FieldSchema("truth_pose.position_enu_m", "vector", "m",
                                         "truth/truth_frames.jsonl", ()), (vehicle,),
                             owner_roles=("vehicle",), authority="l0:truth_frames.jsonl")
    fields = formula_fields(query) + (vector,)
    values = np.empty((1, 2, 3), dtype=object)
    values[0, 0] = ("lane-9", 1.25, (1.0, 2.0, 3.0))
    values[0, 1] = ("lane-9", 1.25, None)
    known = np.array([[[True, True, True], [True, True, False]]])
    projected, projected_known = project_typed_fields(values, known, fields)
    assert projected.shape == (1, 2, 5)
    assert tuple(projected[0, 0]) == (1.0, 1.25, 1.0, 2.0, 3.0), projected[0, 0]
    assert projected_known[0, 1].tolist() == [True, True, False, False, False]
    assert not bool((projected_known & np.isnan(projected)).any())
    encoded = _numeric(np.array([[1.25]]), np.array([[True]]), (fields[1],))
    assert encoded[0, 0] == 1.25
    try:
        _numeric(np.array([[(1.0, 2.0, 3.0)]]), np.array([[True]]), (vector,))
        raise AssertionError("scalar path must reject vector cells")
    except ContractError:
        pass
    expanded = expanded_field_coordinates(fields)
    assert [item.schema.field for item in expanded] == ["truth.lane_id", "truth.speed_mps",
        "truth_pose.position_enu_m#east", "truth_pose.position_enu_m#north", "truth_pose.position_enu_m#up"]
    return {"projected_width": int(projected.shape[-1]),
            "scalar_width": 1,
            "unknown_censored": bool(not projected_known[0, 1].any() or projected_known[0, 1, 2:].sum() == 0),
            "component_fields": [item.schema.field for item in expanded]}
