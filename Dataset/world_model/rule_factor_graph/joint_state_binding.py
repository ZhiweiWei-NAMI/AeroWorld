"""Typed dependency bindings over isolated prefix observations.

This layer executes no producers and attaches no probability law. All executable
specifications have metadata plans; runtime observation coverage is separate.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .state_dependencies import (
    DETERMINISTIC, PARAMETER, DependencyEdge, DependencyNode, PredicateDependencySpec,
    ProducerRoleBinding, SourceCell, load_dependency_specs, merged_dependency_graph,
)

STATUSES = ("admitted", "latent", "missing", "unsupported", "parameter")
# Source artifact -> (position field, velocity field). An isolated physical
# state carries no artifact identity, so the caller declares which artifact the
# admitted poses came from; no cross-artifact equivalence is asserted.
POSE_FIELDS = {"truth_frames.jsonl": ("truth_pose.position_enu_m", "truth_pose.velocity_enu_mps"),
               "trajectories.jsonl": ("position_enu_m", "velocity_enu_mps")}
# Producer row field -> receiver reference role, as paired by
# Dataset/world_model/graph/observation_logs.py:generate. A composed role
# identity is resolved only through one row that carries that exact reference.
ROW_KEY_REFERENCE_ROLE = {
    "entity_id": "subject", "station_id": "station", "session_id": "session",
    "flow_id": "flow", "retransmission_id": "retransmission_monitor",
    "handover_id": "handover_monitor", "node_id": "compute_node",
    "task_id": "task", "queue_id": "queue", "resource_id": "resource",
}
# Publisher kind/field namespace, ordered owner references and the producer's
# source-requirement guard. Compute truth helpers test missing_source_record;
# generate() reports source_required by overriding the complete field group.
RECEIVER_ROWS = {
    "communication_state.jsonl": ("gateway_diagnostic", "", ("subject", "station"), "source_required"),
    "compute_state.jsonl": ("compute_request_register", "compute.", ("subject",), "missing_source_record"),
    "domain_state_supplement/domain_state_observations.jsonl": (
        "traffic_manager_status", "", ("subject",), None),
}
RECEIVER_FIELD_NAMES = {
    ("domain_state_supplement/domain_state_observations.jsonl", "values.road_closed"): "road.road_closed",
    ("domain_state_supplement/domain_state_observations.jsonl", "values.controller_health"): "signal.controller_health",
}
# compute_comm rows construct these identities from episode and the declared
# composition, without tick (1071-1096, 1429-1454). A handover's composition is
# actor only; its changing selected/previous/candidate station is not identity.
IMMUTABLE_COMPOSED_ROWS = frozenset((
    "session_id", "flow_id", "retransmission_id", "handover_id",
    "node_id", "task_id", "queue_id", "resource_id",
))
# Dataset/semantic_truth/world_truth.py:_build_scope_inventory constructs equal
# entity_id/source_entity_id for exactly these scope types; compute_node
# source entities can differ. The contract template's scope_types is the type.
SCOPE_IDENTITY_TYPES = ("uav", "vehicle", "pedestrian", "facility")
# Collection keys whose members the isolated view can type. compute_comm._is_station_entity
# tests entity_kind text or the communication_station category; isolated states carry
# neither (that category is outside ENTITY_CATEGORIES), so only caller-asserted
# SimulatedStationFacilityProjection stations are typed members, a lower bound.
POPULATION_KEYS = ("station",)
POPULATION_REASON = ("members are caller-typed simulated stations only; the producer population also "
                     "depends on unobserved frame entities and entity_kind, so completeness is not established")
SIMULATED_STATION_AUTHORITY = (
    "Dataset/semantic_simulation/compute_comm.py:_simulated_station_from_facility_anchor"
)


class BindingError(ValueError):
    """An observation envelope or exact identity projection is invalid."""


@dataclass(frozen=True)
class FormalCommunicationProfile:
    """Named immutable episode manifest and its governing compute profile."""
    episode_id: str
    manifest_ref: str
    profile_ref: str
    profile_id: str

    @classmethod
    def from_formal_episode(cls, episode_id: str):
        from .state_dependencies import ROOT
        if not isinstance(episode_id, str) or Path(episode_id).name != episode_id:
            raise BindingError("formal episode requires an exact path component")
        manifest_ref = f"aw_data/objective_semantic_truth/{episode_id}/manifest.json"
        manifest = json.loads((ROOT / manifest_ref).read_text())
        profile_path = Path(manifest["compute_profile"]["path"])
        profile_ref = str(profile_path.relative_to(ROOT))
        profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        if manifest["episode_id"] != episode_id:
            raise BindingError("communication profile manifest belongs to another episode")
        return cls(episode_id, manifest_ref, profile_ref, profile["profile_id"])

    def values(self, episode_id: str):
        from .state_dependencies import ROOT
        if self != self.from_formal_episode(episode_id):
            raise BindingError("communication profile differs from episode manifest authority")
        profile = json.loads((ROOT / self.profile_ref).read_text(encoding="utf-8-sig"))
        return profile["communication"]["flow_profiles"]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class DependencyPlan:
    specs: Mapping[str, PredicateDependencySpec]
    nodes: Mapping[str, DependencyNode]
    edges: tuple[DependencyEdge, ...]
    families: Mapping[str, str]

    @property
    def predicate_ids(self) -> tuple[str, ...]:
        return tuple(self.specs)


@dataclass(frozen=True)
class CellKey:
    """Reading roles never enter physical identity; entity is an isolated alias."""
    artifact: str
    episode: str
    entity: str | None
    tick: int
    field: str
    branch: int
    owners: tuple[str, ...] = ()

    @property
    def owner_aliases(self) -> tuple[str, ...]:
        return self.owners if self.owners else (() if self.entity is None else (self.entity,))


@dataclass(frozen=True)
class BoundCell:
    key: CellKey
    frame_owner: CellKey | None
    status: str
    value: Any
    reason: str
    availability_tick: int | None = None


@dataclass(frozen=True)
class BoundRead:
    node_id: str
    predicate_id: str
    cell_index: int


@dataclass(frozen=True)
class SimulatedStationFacilityProjection:
    """Caller-asserted typed anchor relation; isolated logs do not declare it.

    The caller establishes this relation from the named producer before passing
    anonymous aliases. Registry geometry supplies position only, never station
    velocity or operational state. Availability is the relation's admitted clock.
    """
    station_alias: str
    facility_alias: str
    source_authority: str
    availability_tick: int


def _formal_weather_authority(partition: Path):
    from Dataset.world_model.graph.records import load_scope
    from Dataset.world_model.graph.contract import REPO
    scope = load_scope(partition)
    if scope["source_kind"] != "formal":
        raise BindingError("formal weather ownership cannot be applied to an ARM scope")
    episode = scope["episode_id"]
    manifest_path = REPO / "aw_data/render_ready_episodes_capture_filtered" / episode / "episode_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["episode_id"] != episode:
        raise BindingError("formal weather manifest episode differs from the partition")
    identity = manifest["capture_boundary_id"]
    if not isinstance(identity, str) or not identity:
        raise BindingError("formal manifest capture_boundary_id must be an exact identity")
    return episode, identity, str(manifest_path.relative_to(REPO))


@dataclass(frozen=True)
class FormalWeatherRegionProjection:
    """Verified formal manifest identity; raw references remain external metadata.

    Construct through from_formal_partition using the isolated view's external
    identity trace. The binder rereads the canonical authority and admits values
    only after the explicit projection availability tick.
    """
    partition: str
    episode_id: str
    region_alias: str
    raw_region_id: str
    manifest_ref: str
    availability_tick: int
    ontology_class: str = "world:OperationalRegion"
    source_authority: str = "episode_manifest.json#capture_boundary_id"

    @classmethod
    def from_formal_partition(cls, partition: Path, view: Mapping[str, Any], *,
                              availability_tick: int):
        episode, identity, manifest_ref = _formal_weather_authority(Path(partition))
        references = view["trace"]["entity_references"]
        aliases = [alias for alias, raw in references.items() if raw == identity]
        if len(aliases) != 1:
            raise BindingError("formal manifest region must have one exact isolated identity alias")
        result = cls(str(Path(partition)), episode, aliases[0], identity, manifest_ref, availability_tick)
        _weather_projection(view["input"], result, episode, references)
        return result


def _weather_projection(payload, projection, episode_id, entity_references):
    if projection is None:
        return None
    if not isinstance(projection, FormalWeatherRegionProjection):
        raise BindingError("weather ownership requires a typed formal region projection")
    episode, identity, manifest_ref = _formal_weather_authority(Path(projection.partition))
    if ((projection.episode_id, projection.raw_region_id, projection.manifest_ref)
            != (episode, identity, manifest_ref) or episode != episode_id
            or projection.ontology_class != "world:OperationalRegion"
            or projection.source_authority != "episode_manifest.json#capture_boundary_id"
            or projection.region_alias not in payload["entities"]
            or not isinstance(entity_references, Mapping)
            or entity_references.get(projection.region_alias) != identity
            or sum(raw == identity for raw in entity_references.values()) != 1):
        raise BindingError("formal weather projection differs from exact manifest identity/type authority")
    if (type(projection.availability_tick) is not int or projection.availability_tick < 0
            or projection.availability_tick > payload["cutoff_tick"]):
        raise BindingError("formal weather relation is not available within the causal prefix")
    logs = [log for log in payload["logs"] if log["kind"] == "weather_report"]
    if (not logs or any(log["references"]["subject"] != [projection.region_alias] for log in logs)
            or min(log["availability_tick"] for log in logs) > projection.availability_tick):
        raise BindingError("formal weather publisher subject must precede typed relation availability")
    return projection


@dataclass(frozen=True)
class CollectionPopulation:
    """Typed members admitted at one tick for a producer collection read."""
    predicate_id: str
    node_id: str
    key: str
    tick: int
    members: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class BindingResult:
    plan: DependencyPlan
    cells: tuple[BoundCell, ...]
    reads: tuple[BoundRead, ...]
    selected_predicates: tuple[str, ...]
    cutoff_tick: int
    aliases: tuple[str, ...]
    isolated_input: Mapping[str, Any]
    runtime_reasons: Mapping[str, str]
    facility_projections: tuple[SimulatedStationFacilityProjection, ...] = ()
    pose_artifact: str | None = None
    populations: tuple[CollectionPopulation, ...] = ()
    weather_projection: FormalWeatherRegionProjection | None = None
    communication_profile: FormalCommunicationProfile | None = None

    def coverage(self) -> dict[str, Any]:
        return {"metadata_predicates": len(self.plan.specs),
                "selected_predicates": len(self.selected_predicates),
                "runtime_cell_status_counts": {
                    status: sum(cell.status == status for cell in self.cells)
                    for status in STATUSES},
                "runtime_reasons": dict(self.runtime_reasons),
                "producer_execution": False, "probability_model": False}


def compile_dependency_plan(
    specs: Mapping[str, PredicateDependencySpec] | None = None,
) -> DependencyPlan:
    """Compile the exact API, defaulting to all 72 executable specifications."""
    loaded = load_dependency_specs() if specs is None else dict(specs)
    if not loaded or any(pid != spec.predicate_id for pid, spec in loaded.items()):
        raise BindingError("plan keys must equal declared predicate IDs")
    graph = merged_dependency_graph(loaded)
    families: dict[str, str] = {}
    for node_id, node in graph.nodes.items():
        if node.source_cell is None:
            continue
        current, visited = node_id, set()
        while graph.nodes[current].source_cell.coherent_parent is not None:
            if current in visited:
                raise BindingError(f"cyclic coherent source family: {node_id}")
            visited.add(current)
            parent = graph.nodes[current].source_cell.coherent_parent
            if parent not in graph.nodes or graph.nodes[parent].source_cell is None:
                raise BindingError(f"missing coherent source parent: {node_id}")
            current = parent
        families[node_id] = current
    return DependencyPlan(MappingProxyType(dict(loaded)),
                          MappingProxyType(dict(graph.nodes)), graph.edges,
                          MappingProxyType(families))


def _validate_input(view: Mapping[str, Any]) -> Mapping[str, Any]:
    from Dataset.world_model.graph.prediction import (
        INPUT_VERSION, OBSERVATIONS, OBSERVATION_POLICY,
    )
    from Dataset.world_model.graph.contract import ENTITY_CATEGORIES
    from Dataset.world_model.graph.observation_logs import (
        LOG_VERSION, FEATURES, KINDS, MISSING_REASONS, POLICY,
        CHANNELS, RECEIVERS, REFERENCE_ROLES, PERSISTENT_KINDS,
    )
    if not isinstance(view, Mapping) or set(view) != {"input", "trace"}:
        raise BindingError("expected isolated-view envelope; target graphs are inadmissible")
    # Observation validation uses no trace; the optional formal ownership
    # projection separately checks only the external entity_references map.
    payload = view["input"]
    required = {"version", "cutoff_tick", "history_start_tick", "observation",
                "query", "entities", "observed_cohorts", "observation_policy",
                "log_policy", "log_version", "logs", "states"}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise BindingError("isolated input keys differ from the observation contract")
    if (payload["version"] != INPUT_VERSION or payload["observation"] not in OBSERVATIONS
            or payload["observation_policy"] != OBSERVATION_POLICY):
        raise BindingError("isolated observation authority differs")
    cutoff, start = payload["cutoff_tick"], payload["history_start_tick"]
    if type(cutoff) is not int or type(start) is not int or start > cutoff:
        raise BindingError("invalid cutoff/history clock")
    aliases = payload["entities"]
    if (not isinstance(aliases, list)
            or any(not isinstance(alias, str) or alias != f"entity:{i}"
                   for i, alias in enumerate(aliases))):
        raise BindingError("entities must be ordered unique anonymous aliases")
    if not isinstance(payload["states"], list) or not isinstance(payload["logs"], list):
        raise BindingError("isolated states and logs must be record lists")
    cohorts = payload["observed_cohorts"]
    if (not isinstance(cohorts, Mapping)
            or any(group not in aliases or not isinstance(members, list)
                   or any(member not in aliases for member in members)
                   for group, members in cohorts.items())):
        raise BindingError("observed cohort references must be known aliases")
    for state in payload["states"]:
        if (not isinstance(state, Mapping)
                or set(state) != {"id", "entity_id", "entity_category", "tick", "values"}):
            raise BindingError("isolated state keys differ")
        if (state["entity_id"] not in aliases or state["entity_category"] not in ENTITY_CATEGORIES
                or type(state["tick"]) is not int or not start <= state["tick"] <= cutoff
                or state["id"] != f"state:{state['entity_id']}@{state['tick']}"
                or not isinstance(state["values"], list) or len(state["values"]) != 7):
            raise BindingError("isolated state identity, category, clock or shape differs")
        if any(value is not None and (type(value) not in (int, float)
                   or not math.isfinite(value)) for value in state["values"]):
            raise BindingError("physical values must be finite or explicitly missing")
    uses_logs = payload["observation"].endswith("_logs")
    if (payload["log_version"] != (LOG_VERSION if uses_logs else None)
            or payload["log_policy"] != (POLICY if uses_logs else None)
            or (not uses_logs and payload["logs"])):
        raise BindingError("receiver log authority differs from observation condition")
    for log in payload["logs"]:
        required_log = {"channel", "kind", "receiver_kind", "sample_tick",
                        "availability_tick", "references", "values", "missing", "id"}
        if (not isinstance(log, Mapping) or set(log) != required_log
                or log["kind"] not in KINDS or log["channel"] not in CHANNELS
                or log["receiver_kind"] not in RECEIVERS
                or not isinstance(log["id"], str)):
            raise BindingError("isolated log keys or vocabulary differ")
        if (type(log["sample_tick"]) is not int or type(log["availability_tick"]) is not int
                or log["sample_tick"] != log["availability_tick"]
                or log["availability_tick"] > cutoff
                or (log["availability_tick"] < start and log["kind"] not in PERSISTENT_KINDS)):
            raise BindingError("log availability is outside its admitted prefix")
        if (not isinstance(log["references"], Mapping)
                or any(role not in REFERENCE_ROLES or not isinstance(refs, list)
                       or any(alias not in aliases for alias in refs)
                       for role, refs in log["references"].items())):
            raise BindingError("log reference role or alias differs")
        if (not isinstance(log["values"], Mapping) or not isinstance(log["missing"], Mapping)
                or set(log["values"]) - set(FEATURES)
                or set(log["missing"]) - set(log["values"])
                or any(reason not in MISSING_REASONS for reason in log["missing"].values())):
            raise BindingError("log field or missing-reason authority differs")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in log["values"].values() if value is not None):
            raise BindingError("log values must be finite or explicitly missing")
        if any((value is None) != (field in log["missing"])
               for field, value in log["values"].items()):
            raise BindingError("log value and missing status disagree")
        _validate_log_publisher(log)
    query = payload["query"]
    if (not isinstance(query, Mapping)
            or set(query) != {"task_id", "query", "bindings", "horizon_ticks"}
            or not isinstance(query["task_id"], str) or not query["task_id"]
            or query["query"] not in ("future_onset", "current_hidden_predicate", "future_terminal")
            or not isinstance(query["bindings"], Mapping)
            or type(query["horizon_ticks"]) is not int or query["horizon_ticks"] < 0):
        raise BindingError("exact isolated query and horizon are required")
    if any(not isinstance(role, str) or alias not in aliases
           for role, alias in query["bindings"].items()):
        raise BindingError("query role references an unknown alias")
    return payload


def _validate_log_publisher(log):
    """Validate the exact observation_logs._record/generate publishing contract."""
    from Dataset.world_model.graph.observation_logs import (
        KINDS, RECEIVERS, FIELD_GROUPS, ENUM_VALUES,
    )

    kind = log["kind"]
    index = KINDS.index(kind)
    channel = ("weather", "fault", "request", "request", "communication", "communication",
               "request", "fault", "context", "fault", "request")[index]
    fields = FIELD_GROUPS[index]
    if (log["channel"] != channel or log["receiver_kind"] != RECEIVERS[index]
            or set(log["values"]) != set(fields)):
        raise BindingError(f"{kind}: publisher channel, receiver or complete field group differs")
    refs = log["references"]
    required, optional, lists = {"subject"}, set(), set()
    if kind in ("facility_request_register", "local_link_status"):
        optional, lists = {"requester"}, {"requester"}
    elif kind == "gateway_diagnostic":
        optional = {"station", "session", "flow", "retransmission_monitor", "handover_monitor"}
    elif kind == "compute_request_register":
        required |= {"compute_node", "task", "queue", "resource"}
        optional = {"offload_destination"}
    elif kind == "facility_manager_register":
        required, lists = {"subject", "requester"}, {"requester"}
    elif kind == "public_registry":
        optional, lists = {"member"}, {"member"}
        cohort = log["values"]["map.kind"] == ENUM_VALUES["map_kind_code"].index("cohort")
        if cohort != ("member" in refs) or (cohort and not refs["member"]):
            raise BindingError("public_registry: cohort member reference shape differs")
    if not required <= set(refs) or set(refs) - required - optional:
        raise BindingError(f"{kind}: publisher reference roles differ")
    if any(len(aliases) != len(set(aliases)) or (role not in lists and len(aliases) != 1)
           for role, aliases in refs.items()):
        raise BindingError(f"{kind}: publisher reference cardinality or uniqueness differs")
    if kind == "facility_request_register" and "requester" in refs:
        count = log["values"]["facility_state.request_count"]
        if count is not None and count != len(refs["requester"]):
            raise BindingError("facility_request_register: request count differs from its register")
    for field, unit in fields.items():
        value = log["values"][field]
        if value is None:
            continue
        if unit == "bool" and value not in (0., 1.):
            raise BindingError(f"{kind}: boolean publisher field is not 0/1: {field}")
        if unit in ENUM_VALUES and value not in range(len(ENUM_VALUES[unit])):
            raise BindingError(f"{kind}: publisher enum code differs: {field}")
    forced = set(log["missing"].values()) & {"diagnostic_source_absent", "source_required"}
    if forced:
        allowed = {"source_required"} if kind == "compute_request_register" else (
            {"source_required", "diagnostic_source_absent"} if kind == "gateway_diagnostic" else set())
        if (not forced <= allowed or len(forced) != 1
                or set(log["missing"]) != set(fields)
                or set(log["missing"].values()) != forced):
            raise BindingError(f"{kind}: publisher source-missing override differs")
    if kind == "gateway_diagnostic" and forced == {"diagnostic_source_absent"} and set(refs) != {"subject"}:
        raise BindingError("gateway_diagnostic: absent source has subject reference only")


def _scope_types() -> Mapping[str, frozenset[str]]:
    from .unified_contracts import load_unified_contracts

    return MappingProxyType({pid: frozenset(contract.template["scope_types"])
                             for pid, contract in load_unified_contracts().items()})


def _role_indexes(spec: PredicateDependencySpec, payload: Mapping[str, Any],
                  scope_types: frozenset[str]):
    """Tick-independent index identities, and composed roles resolved per tick."""
    bindings = payload["query"]["bindings"]
    if set(bindings) != {role.key for role in spec.roles}:
        raise BindingError(f"{spec.predicate_id}: query roles differ from executable roles")
    static: dict[str, str] = {}
    composed: list[tuple[str, ProducerRoleBinding]] = []

    def assign(key: str, alias: str) -> None:
        if key in static and static[key] != alias:
            raise BindingError(f"{spec.predicate_id}: ambiguous producer index {key}")
        static[key] = alias

    for role in spec.roles:
        producer = role.producer
        if producer is not None:
            if producer.status != "resolved":
                raise BindingError(f"{spec.predicate_id}: unresolved producer binding")
            if producer.form == "row_key":
                # The role identity is the index entity itself.
                if len(producer.index_keys) != 1:
                    raise BindingError(f"{spec.predicate_id}: row-key role must name one index")
                assign(producer.index_keys[0], bindings[role.key])
            else:
                composed.append((bindings[role.key], producer))
        for field in spec.fields:
            for projection in field.producer_role_projections:
                if projection.owner_role != role.key or projection.index_key is None:
                    continue
                if projection.form == "index_identity":
                    assign(projection.index_key, bindings[role.key])
                elif (projection.form == "scope_source_lookup" and role.source_kind == "scope_entity"
                      and scope_types <= set(SCOPE_IDENTITY_TYPES)):
                    # Identity equals the source entity for these scope types.
                    assign(projection.index_key, bindings[role.key])
    return static, tuple(composed)


def _composed_keys(composed) -> frozenset[str]:
    return frozenset(arg.index_keys[0] for _, producer in composed
                     if producer.row_key in ROW_KEY_REFERENCE_ROLE
                     for arg in producer.composition
                     if arg.row_key in ROW_KEY_REFERENCE_ROLE and len(arg.index_keys) == 1)


def _composed_indexes(spec, composed, payload: Mapping[str, Any], tick: int) -> dict[str, str]:
    """Resolve known immutable components from causally available receiver rows.

    Identity can persist into gaps/future; observations do not persist with it.
    Partial rows never combine to manufacture a complete identity tuple.
    """
    resolved: dict[str, str] = {}
    for alias, producer in composed:
        reference = ROW_KEY_REFERENCE_ROLE.get(producer.row_key)
        if reference is None:
            continue
        rows = set()
        known = {}
        immutable = producer.row_key in IMMUTABLE_COMPOSED_ROWS
        for log in payload["logs"]:
            if alias not in log["references"].get(reference, ()):
                continue
            if immutable:
                if log["availability_tick"] > min(tick, payload["cutoff_tick"]):
                    continue
            elif log["sample_tick"] != tick:
                continue
            components = {}
            for arg in producer.composition:
                role = ROW_KEY_REFERENCE_ROLE.get(arg.row_key)
                if role is None:
                    continue
                if len(arg.index_keys) != 1:
                    raise BindingError(f"{spec.predicate_id}: composition argument must name one index")
                aliases = log["references"].get(role, ())
                if len(aliases) > 1:
                    raise BindingError(f"{spec.predicate_id}: ambiguous composed component {arg.index_keys[0]}")
                if aliases:
                    key, component = arg.index_keys[0], aliases[0]
                    if key in known and known[key] != component:
                        raise BindingError(f"{spec.predicate_id}: receiver rows disagree on known composed component {key}")
                    known[key] = component
                    components[key] = component
            if components:
                rows.add(tuple(sorted(components.items())))
        # A stronger row may establish an omitted component, but disjoint rows
        # never become a synthetic tuple. With publisher-required subject, a
        # station-bearing gateway row jointly establishes its actor and station.
        maximal = [set(row) for row in rows if not any(set(row) < set(other) for other in rows)]
        joint = set.intersection(*maximal) if maximal else set()
        for key, component in sorted(joint):
            if resolved.get(key, component) != component:
                raise BindingError(f"{spec.predicate_id}: ambiguous composed role {key}")
            resolved[key] = component
    return resolved


def _facility_registry(payload, facility_alias):
    from Dataset.world_model.graph.observation_logs import ENUM_VALUES

    facility_code = ENUM_VALUES["map_kind_code"].index("facility_point")
    return [log for log in payload["logs"]
            if log["kind"] == "public_registry" and log["channel"] == "context"
            and log["receiver_kind"] == "public_registry"
            and facility_alias in log["references"].get("subject", ())
            and log["values"].get("map.kind") == facility_code]


def _facility_projections(payload, projections):
    validated = {}
    projections = tuple(projections)
    if ({item.station_alias for item in projections if isinstance(item, SimulatedStationFacilityProjection)}
            & {item.facility_alias for item in projections if isinstance(item, SimulatedStationFacilityProjection)}):
        raise BindingError("a simulated station cannot also be a projected facility anchor")
    for projection in projections:
        if not isinstance(projection, SimulatedStationFacilityProjection):
            raise BindingError("facility projections must be typed producer relations")
        if (projection.source_authority != SIMULATED_STATION_AUTHORITY
                or projection.station_alias not in payload["entities"]
                or projection.facility_alias not in payload["entities"]
                or projection.station_alias == projection.facility_alias):
            raise BindingError("simulated station projection authority or distinct aliases differ")
        if (type(projection.availability_tick) is not int
                or projection.availability_tick > payload["cutoff_tick"]):
            raise BindingError("simulated station relation is not available in the admitted prefix")
        if projection.station_alias in validated:
            raise BindingError("simulated station has duplicate or contradictory facility anchors")
        for state in payload["states"]:
            expected = {projection.station_alias: "communication_station",
                        projection.facility_alias: "facility"}.get(state["entity_id"])
            if expected is not None and state["entity_category"] != expected:
                raise BindingError("simulated station/facility projection conflicts with observed category")
        registry = _facility_registry(payload, projection.facility_alias)
        gateway = [log for log in payload["logs"]
                   if log["kind"] == "gateway_diagnostic" and log["channel"] == "communication"
                   and log["receiver_kind"] == "local_gateway_monitor"
                   and projection.station_alias in log["references"].get("station", ())]
        if (not registry or not gateway
                or min(log["availability_tick"] for log in registry) > projection.availability_tick
                or min(log["availability_tick"] for log in gateway) > projection.availability_tick):
            raise BindingError("facility geometry and station reference must precede relation availability")
        fields = ("map.e_m", "map.n_m", "map.u_m")
        geometry = {tuple(log["values"].get(field) for field in fields) for log in registry}
        if len(geometry) != 1:
            raise BindingError("conflicting public facility geometry for simulated station")
        validated[projection.station_alias] = projection
    return validated


def bind_isolated_view(
    plan: DependencyPlan, view: Mapping[str, Any], *, episode_id: str,
    predicate_ids: Sequence[str], branch: int = 0,
    future_ticks: Sequence[int] = (),
    parameters: Mapping[str, Any] | None = None,
    facility_projections: Sequence[SimulatedStationFacilityProjection] = (),
    pose_artifact: str | None = None,
    weather_projection: FormalWeatherRegionProjection | None = None,
    communication_profile: FormalCommunicationProfile | None = None,
) -> BindingResult:
    """Bind isolated prefix evidence and explicit future tick requests.

    episode_id is an external sharing key, never a learner feature. Future
    times must be declared by the caller; per-edge timing/cache policies remain
    exact metadata and are not replaced by one inferred sampling grid.
    pose_artifact names the source artifact of the admitted poses (isolated
    states carry none); poses bind only that artifact's cells, never another.
    Isolated velocity is a backward difference and binds no source velocity cell.
    """
    if pose_artifact is not None and pose_artifact not in POSE_FIELDS:
        raise BindingError("pose_artifact must be a declared physical-pose artifact")
    payload = _validate_input(view)
    projections = _facility_projections(payload, facility_projections)
    if not isinstance(episode_id, str) or not episode_id or type(branch) is not int or branch < 0:
        raise BindingError("external episode identity and nonnegative branch are required")
    # Only a requested formal identity projection inspects this one external
    # alias map. Trace observations/targets never enter source values or tokens.
    references = None if weather_projection is None else view["trace"]["entity_references"]
    weather_projection = _weather_projection(payload, weather_projection, episode_id, references)
    selected = tuple(predicate_ids)
    if not selected or len(set(selected)) != len(selected) or any(pid not in plan.specs for pid in selected):
        raise BindingError("selected predicates must be unique declared predicates")
    supplied = {} if parameters is None else dict(parameters)
    if any(node_id not in plan.nodes or plan.nodes[node_id].kind != PARAMETER for node_id in supplied):
        raise BindingError("supplied values must name declared parameter nodes")
    cutoff = payload["cutoff_tick"]
    if communication_profile is not None and not isinstance(communication_profile, FormalCommunicationProfile):
        raise BindingError("communication thresholds require a typed formal profile authority")
    flow_profiles = None if communication_profile is None else communication_profile.values(episode_id)
    future = tuple(future_ticks)
    if (any(type(tick) is not int or not cutoff < tick <= cutoff + payload["query"]["horizon_ticks"]
            for tick in future) or future != tuple(sorted(set(future)))):
        raise BindingError("future ticks must be ordered unique integers within query horizon")
    times = sorted({cutoff, *future, *(state["tick"] for state in payload["states"]),
                    *(log["sample_tick"] for log in payload["logs"])})
    state_lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    for state in payload["states"]:
        key = state["entity_id"], state["tick"]
        if key in state_lookup and state_lookup[key] != state:
            raise BindingError("conflicting physical observations")
        state_lookup[key] = state
    subjects = {alias for log in payload["logs"] for alias in log["references"].get("subject", ())}
    physical_entities = tuple(sorted({state["entity_id"] for state in payload["states"]}
                                     | subjects | set(projections)))
    cells: list[BoundCell] = []
    reads: list[BoundRead] = []
    cell_lookup: dict[CellKey, int] = {}
    reasons: dict[str, str] = {}
    populations: dict[tuple[str, str, int], CollectionPopulation] = {}
    scope_types = _scope_types()
    from Dataset.world_model.graph.observation_logs import FEATURE_UNITS
    # Weather rows by sample tick; other rows by sample tick and subject alias.
    weather_logs: defaultdict[int, list] = defaultdict(list)
    subject_logs: defaultdict[tuple[int, str], list] = defaultdict(list)
    for log in payload["logs"]:
        if log["kind"] == "weather_report":
            weather_logs[log["sample_tick"]].append(log)
        for alias in log["references"].get("subject", ()):
            subject_logs[(log["sample_tick"], alias)].append(log)
    log_index = (weather_logs, subject_logs)

    def intern(key: CellKey, owner: CellKey | None, status: str, value: Any, reason: str,
               availability_tick: int | None = None) -> int:
        cell = BoundCell(key, owner, status, _freeze(value), reason, availability_tick)
        if key in cell_lookup:
            index = cell_lookup[key]
            if cells[index] != cell:
                raise BindingError(f"conflicting canonical physical cell: {key}")
            return index
        index = len(cells)
        cells.append(cell)
        cell_lookup[key] = index
        return index

    for pid in selected:
        spec = plan.specs[pid]
        static, composed = _role_indexes(spec, payload, scope_types[pid])
        receiver_bindings = {
            ROW_KEY_REFERENCE_ROLE[role.producer.row_key]: payload["query"]["bindings"][role.key]
            for role in spec.roles if role.producer is not None
            and role.producer.row_key in ROW_KEY_REFERENCE_ROLE
        }
        establishable = frozenset(static) | _composed_keys(composed)
        collected: defaultdict[str, set[str]] = defaultdict(set)
        for edge in spec.edges:
            collected[edge.source].update(edge.collection_keys)
        tick_indexes = {}
        for tick in times:
            components = _composed_indexes(spec, composed, payload, tick)
            if any(key in static and static[key] != alias for key, alias in components.items()):
                raise BindingError(f"{pid}: composed identity conflicts with explicit query index")
            tick_indexes[tick] = {**static, **components}
        # Immediate receiver and governed row views terminate observation
        # traversal; their producer ancestors remain in the compiled metadata.
        closure, pending = set(), [spec.predicate_node_id]
        upstream = defaultdict(list)
        for edge in spec.edges:
            upstream[edge.target].append(edge.source)
        while pending:
            node_id = pending.pop()
            if node_id in closure:
                continue
            closure.add(node_id)
            source = plan.nodes[node_id].source_cell
            if source is None or source.form not in (
                    "published_monitor", "governed_row_parameter", "publisher_requirement",
                    "unobserved_monitor"):
                pending.extend(upstream[node_id])
        closure.discard(spec.predicate_node_id)
        for node_id in sorted(closure):
            node = plan.nodes[node_id]
            source = node.source_cell
            if source is None:
                status = "parameter" if node.kind == PARAMETER else "unsupported"
                reason = (("caller-supplied fixed parameter" if node_id in supplied else
                           "fixed parameter authority only; value not supplied")
                          if status == "parameter" else
                          "producer execution is outside the observation binding layer")
                owners = ()
                if node.kind == DETERMINISTIC and node.index.keys:
                    for field in spec.fields:
                        operand_projections = field.producer_role_projections
                        if (field.data_type not in ("numeric", "integer", "boolean", "categorical")
                                or not operand_projections or not any(item.source_node == node_id for item in operand_projections)):
                            continue
                        if (field.owner_status != "resolved" or field.projection_status != "resolved"
                                or any(item.source_node != node_id for item in operand_projections)
                                or tuple(item.owner_role for item in operand_projections) != field.owner_roles
                                or tuple(item.index_key for item in operand_projections) != node.index.keys):
                            raise BindingError("unavailable contract operand has no exact declared index owners")
                        declared_owners = tuple(tick_indexes[cutoff].get(item.index_key) for item in operand_projections)
                        if any(owner is None for owner in declared_owners) or (
                                owners and owners != declared_owners):
                            raise BindingError("unavailable contract operand has unresolved or conflicting owners")
                        owners = declared_owners
                index = intern(CellKey("dependency", episode_id, None, cutoff, node_id, branch, owners),
                               None, status, supplied.get(node_id), reason)
                reads.append(BoundRead(node_id, pid, index))
                if status == "unsupported":
                    reasons[node_id] = reason
                continue
            entity_keys = tuple(key for key in source.coordinate_keys if key not in ("episode_id", "tick"))
            root = plan.nodes[plan.families[node_id]].source_cell
            if source.form in ("published_monitor", "governed_row_parameter", "publisher_requirement",
                               "unobserved_monitor"):
                if entity_keys != node.index.keys:
                    raise BindingError("published row coordinates differ from producer ownership")
                if (source.artifact != "communication_state.jsonl"
                        and not set(entity_keys) <= establishable):
                    raise BindingError("published row coordinates differ from producer ownership")
                for tick in times:
                    current = _receiver_indexes(source, entity_keys, tick, payload, receiver_bindings)
                    components = tick_indexes[tick]
                    if any(key in components and components[key] != alias
                           for key, alias in current.items()):
                        raise BindingError("current receiver coordinates conflict with producer ownership")
                    coordinates = {**components, **current}
                    owners = tuple(coordinates.get(key) for key in entity_keys)
                    if any(owner is None for owner in owners):
                        index = intern(CellKey("unbound", episode_id, None, tick, node_id, branch),
                                       None, "latent" if tick > cutoff else "missing", None,
                                       "no complete receiver row establishes the ordered producer owners")
                    else:
                        status, value, reason = _observe_receiver(
                            source, node.data_type, owners, tick, cutoff, payload, flow_profiles,
                            receiver_bindings)
                        available = [log["availability_tick"] for log in
                                     _receiver_rows(source, owners, tick, payload, receiver_bindings)]
                        index = intern(CellKey(source.artifact, episode_id, None, tick,
                                               source.field_path, branch, owners),
                                       None, status, value, reason, min(available) if available else None)
                    reads.append(BoundRead(node_id, pid, index))
                continue
            key = entity_keys[0] if len(entity_keys) == 1 else None
            reads_collection = key in collected[node_id]
            typed_population = reads_collection and key in POPULATION_KEYS
            if reads_collection and not typed_population:
                # The indexed owner still binds for same-index reads; the collection
                # read has no admitted membership and is never replaced by that owner.
                reason = "producer collection population has no typed membership authority"
                reasons[node_id] = reason
                index = intern(CellKey("unbound", episode_id, None, cutoff, node_id + "#collection", branch),
                               None, "unsupported", None, reason)
                reads.append(BoundRead(node_id, pid, index))
            if len(entity_keys) > 1 or (key not in (None, "entity_id") and key not in establishable
                                        and not typed_population):
                reason = "source owner coordinates are not established by isolated query projections"
                reasons[node_id] = reason
                index = intern(CellKey("unbound", episode_id, None, cutoff, node_id, branch),
                               None, "unsupported", None, reason)
                reads.append(BoundRead(node_id, pid, index))
                continue
            for tick in times:
                gap = None
                if not entity_keys:
                    entities = ((weather_projection.region_alias,) if source.artifact == "weather_meta.jsonl"
                                and weather_projection is not None else (None,))
                elif key == "entity_id":
                    entities = physical_entities
                else:
                    selected_entity = tick_indexes[tick].get(key)
                    entities = () if selected_entity is None else (selected_entity,)
                    if typed_population:
                        members = () if tick > cutoff else tuple(sorted(
                            alias for alias, item in projections.items() if item.availability_tick <= tick))
                        populations[(pid, node_id, tick)] = CollectionPopulation(
                            pid, node_id, key, tick, members, POPULATION_REASON)
                        entities = tuple(sorted({*entities, *members}))
                    if not entities:
                        gap = (("latent", "future owner or population is not observed")
                               if tick > cutoff else
                               ("missing", "no complete receiver row or typed member establishes the owner at this tick"))
                if gap is not None:
                    index = intern(CellKey("unbound", episode_id, None, tick, node_id, branch),
                                   None, gap[0], None, gap[1])
                    reads.append(BoundRead(node_id, pid, index))
                    continue
                for entity in entities:
                    owner = (CellKey(root.artifact, episode_id, None, tick, "", branch)
                             if root.form == "coherent_frame_root" else None)
                    status, value, reason = _observe(source, node.data_type, entity, tick, cutoff, payload,
                                                     state_lookup, projections, pose_artifact, log_index,
                                                     weather_projection)
                    index = intern(CellKey(source.artifact, episode_id, entity, tick, source.field_path, branch),
                                   owner, status, value, reason,
                                   tick if status == "admitted" and source.artifact == "truth_frames.jsonl"
                                   and any(tuple(field.split(".")[:-1]) == tuple(source.field_path.split("."))
                                           for field in FEATURE_UNITS) else None)
                    reads.append(BoundRead(node_id, pid, index))
    return BindingResult(plan, tuple(cells), tuple(reads), selected, cutoff,
                         tuple(payload["entities"]), _freeze(payload), MappingProxyType(reasons),
                         tuple(projections.values()), pose_artifact, tuple(populations.values()), weather_projection,
                         communication_profile)


def _receiver_indexes(source, entity_keys, tick, payload, receiver_bindings):
    """Read ordered current owners from one exact, complete publisher row."""
    kind, _, references, _ = RECEIVER_ROWS[source.artifact]
    if len(entity_keys) != len(references):
        raise BindingError("receiver owner coordinates differ from the declared row")
    rows = set()
    if tick > payload["cutoff_tick"] or not receiver_bindings:
        return {}
    for log in payload["logs"]:
        if (log["kind"] != kind or log["sample_tick"] != tick
                or log["availability_tick"] > tick
                or any(log["references"].get(role) != [alias]
                       for role, alias in receiver_bindings.items())):
            continue
        aliases = tuple(log["references"].get(role, ()) for role in references)
        if any(len(owners) != 1 for owners in aliases):
            continue
        rows.add(tuple(owners[0] for owners in aliases))
    if len(rows) > 1:
        raise BindingError("conflicting complete current receiver owner tuples")
    return dict(zip(entity_keys, rows.pop())) if rows else {}


def _receiver_rows(source, owners, tick, payload, receiver_bindings):
    kind, _, references, _ = RECEIVER_ROWS[source.artifact]
    if len(owners) != len(references):
        raise BindingError("receiver owner coordinates differ from the declared row")
    return [log for log in payload["logs"] if log["kind"] == kind
            and log["sample_tick"] == tick and log["availability_tick"] <= tick
            and all(log["references"].get(role) == [owner]
                    for role, owner in zip(references, owners))
            and all(log["references"].get(role) == [alias]
                    for role, alias in receiver_bindings.items())]


def _observe_receiver(source, data_type, owners, tick, cutoff, payload, flow_profiles,
                      receiver_bindings):
    if tick > cutoff:
        return "latent", None, "uncertain future; no observation is admitted"
    actor = owners[0]
    rows = _receiver_rows(source, owners, tick, payload, receiver_bindings)
    kind, prefix, _, required_status = RECEIVER_ROWS[source.artifact]
    if source.form == "governed_row_parameter":
        if flow_profiles is None:
            raise BindingError("row thresholds require the named governing communication profile")
        from .state_dependencies import communication_row_actor_category
        category = communication_row_actor_category()
        categories = {state["entity_category"] for state in payload["states"]
                      if state["entity_id"] == actor and state["tick"] <= tick}
        if categories and categories != {category}:
            raise BindingError("observed actor category conflicts with the producer category gate")
        profile_field = {"quality_thresholds.latency_ms": "latency_threshold_ms",
                         "quality_thresholds.packet_loss_ratio": "packet_loss_threshold_ratio"}[source.field_path]
        value = flow_profiles[category][profile_field]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise BindingError("governed communication threshold must be finite numeric")
        return "admitted", round(float(value), 6), "episode manifest governing communication flow profile"
    if not rows:
        return "missing", None, "exact receiver row is absent or not yet available at the sample tick"
    if source.form == "unobserved_monitor":
        return "latent", None, "declared producer scalar is not published by this receiver"
    if source.form == "publisher_requirement":
        # generate() admits exactly these two requirement states. A required
        # source overrides every published field, including fields otherwise missing.
        values = {required_status if "source_required" in log["missing"].values()
                  else "not_required" for log in rows}
        if len(values) != 1:
            raise BindingError("receiver rows disagree on source requirement")
        return "admitted", values.pop(), "observation_logs.generate requirement gate"
    if source.artifact == "domain_state_supplement/domain_state_observations.jsonl":
        field = RECEIVER_FIELD_NAMES[(source.artifact, source.field_path)]
    else:
        field = prefix + source.field_path
    matches = [(log["values"][field], log["missing"].get(field)) for log in rows]
    if any(item != matches[0] for item in matches[1:]):
        raise BindingError(f"conflicting exact receiver field: {field}")
    value, missing = matches[0]
    if missing is not None or value is None:
        return "missing", None, f"published receiver field missing: {missing}"
    if type(value) not in (int, float) or not math.isfinite(value):
        raise BindingError(f"published receiver field requires a finite scalar: {field}")
    if data_type == "boolean":
        if value not in (0., 1.):
            raise BindingError(f"published boolean receiver field is not 0/1: {field}")
        value = bool(value)
    elif data_type == "integer":
        if value != int(value) or value < 0:
            raise BindingError(f"published receiver count is not a nonnegative integer: {field}")
        value = int(value)
    elif data_type == "categorical":
        from Dataset.world_model.graph.observation_logs import ENUM_VALUES, TRAFFIC_FIELDS
        codes = ENUM_VALUES[TRAFFIC_FIELDS[field]]
        if value != int(value) or not 0 <= value < len(codes):
            raise BindingError(f"published receiver enum has no declared code: {field}")
        value = codes[int(value)]
    elif data_type != "numeric":
        raise BindingError(f"receiver scalar type has no adapter: {data_type}")
    return "admitted", value, f"exact published {kind} typed field"


def _observe(source: SourceCell, data_type: str, entity: str | None, tick: int, cutoff: int,
             payload: Mapping[str, Any], states: Mapping[tuple[str, int], Mapping[str, Any]],
             projections: Mapping[str, SimulatedStationFacilityProjection], pose_artifact: str | None,
             log_index, weather_projection):
    if tick > cutoff:
        return "latent", None, "uncertain future; no observation is admitted"
    if source.form == "coherent_frame_root":
        return "latent", None, "whole truth frame is unobserved; coherent field views may be admitted"
    if (source.artifact == "truth_frames.jsonl" and entity in projections
            and source.field_path in ("truth_pose.position_enu_m", "initial_position_enu_m")):
        projection = projections[entity]
        if tick < projection.availability_tick:
            return "latent", None, "typed facility anchor relation is not available at this tick"
        registry = [log for log in _facility_registry(payload, projection.facility_alias)
                    if log["availability_tick"] <= tick]
        fields = ("map.e_m", "map.n_m", "map.u_m")
        values = [tuple(log["values"].get(field) for field in fields) for log in registry]
        value = values[0]
        if any(component is None for component in value):
            return "missing", None, "public facility geometry component missing"
        state = states.get((entity, tick))
        if state is not None and source.field_path == "truth_pose.position_enu_m":
            observed = tuple(state["values"][:3])
            if all(component is not None for component in observed) and observed != value:
                raise BindingError("simulated station position conflicts with typed facility geometry")
        return "admitted", value, "public facility geometry through caller-supplied typed producer relation"
    if source.artifact in POSE_FIELDS and source.field_path in POSE_FIELDS[source.artifact]:
        if source.field_path != POSE_FIELDS[source.artifact][0]:
            return ("latent", None, "isolated velocity is a backward difference of observed poses, "
                    "not the producer-reported velocity; no equivalence is asserted")
        if pose_artifact != source.artifact:
            return ("unsupported", None, "admitted poses are not declared to originate from this "
                    "artifact; no cross-artifact equivalence is asserted")
        state = states.get((entity, tick))
        if state is None:
            return "missing", None, "physical observation absent at this tick"
        value = tuple(state["values"][:3])
        if any(component is None for component in value):
            return "missing", None, "physical observation component missing"
        return "admitted", value, "isolated physical observation"
    if source.artifact not in ("truth_frames.jsonl", "weather_meta.jsonl"):
        return "unsupported", None, "source artifact has no observation adapter"
    if (source.artifact == "weather_meta.jsonl" and weather_projection is not None
            and tick < weather_projection.availability_tick):
        return "latent", None, "formal weather ownership relation is not yet available"
    candidates = (log_index[0].get(tick, ()) if source.artifact == "weather_meta.jsonl"
                  else log_index[1].get((tick, entity), ()))
    if source.artifact == "truth_frames.jsonl":
        from Dataset.world_model.graph.observation_logs import (
            FEATURE_UNITS, FIELD_GROUPS, KINDS, ENUM_VALUES,
        )
        parent = tuple(source.field_path.split("."))
        fields = {field: unit for field, unit in FEATURE_UNITS.items()
                  if tuple(field.split(".")[:-1]) == parent}
        if fields:
            container = {}
            observed_monitor = False
            for field, unit in fields.items():
                kinds = {kind for group, kind in zip(FIELD_GROUPS, KINDS) if field in group}
                observations = [(log["values"][field], log["missing"].get(field))
                                for log in candidates if log["kind"] in kinds]
                if not observations:
                    continue
                observed_monitor = True
                if any(item != observations[0] for item in observations[1:]):
                    raise BindingError(f"conflicting receiver observations: {field}")
                value, missing = observations[0]
                if missing is not None or value is None:
                    continue
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise BindingError(f"published monitor field is not finite numeric: {field}")
                if unit == "bool":
                    if value not in (0.0, 1.0):
                        raise BindingError(f"boolean monitor field is not 0/1: {field}")
                    value = bool(value)
                elif unit == "count":
                    if value < 0 or value != int(value):
                        raise BindingError(f"monitor count is not a nonnegative integer: {field}")
                    value = int(value)
                elif unit in ENUM_VALUES:
                    codes = ENUM_VALUES[unit]
                    if value != int(value) or not 0 <= value < len(codes):
                        raise BindingError(f"published monitor enum has no declared code: {field}")
                    value = codes[int(value)]
                container[field.split(".")[-1]] = value
            if container:
                return "admitted", container, "partial registered container from exact published monitor leaves"
            if observed_monitor:
                return "missing", None, "no known registered container leaf at this tick"
            return "latent", None, "registered container monitor is not observed at this tick"
    matches = [(log["values"][source.field_path], log["missing"].get(source.field_path))
               for log in candidates if source.field_path in log["values"]
               and (source.artifact != "weather_meta.jsonl" or weather_projection is None
                    or log["references"]["subject"] == [entity])]
    if matches:
        if any(item != matches[0] for item in matches[1:]):
            raise BindingError(f"conflicting receiver observations: {source.field_path}")
        value, missing = matches[0]
        if missing is not None or value is None:
            return "missing", None, f"admitted monitor field missing: {missing}"
        if data_type == "boolean":
            # Receiver logs encode booleans as 0/1; the node domain is boolean.
            if value not in (0.0, 1.0):
                raise BindingError(f"boolean monitor field is not 0/1: {source.field_path}")
            value = bool(value)
        return "admitted", value, "isolated receiver observation"
    return "latent", None, "field is not admitted by observation policy"
