"""Generic all-family episode source and operand adapter for the fixed rules.

One contract-driven layer over already declared unified contracts, dependency
specifications, owner projections, governed parameters and actual episode rows.
It compiles the exact ``EpisodeContract``/``FieldCoordinate``/``PredicateQuery``
set, exposes immutable typed state-row access, and binds each executable
predicate to its declared authority: the compute_comm direct producer rows for
the 16 direct predicates, receiver monitor adapters for the published row
fields, and the existing world_truth projections for scope/record formulas.
Nothing here invents a formula, an observation, or a truth label. Missing
authority is explicit; UNKNOWN never becomes false.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .generic_grounding import (EntityClassCatalog, ParameterProfile, RoleBinding,
    TypedEntity)
from .joint_state_binding import (
    RECEIVER_FIELD_NAMES, RECEIVER_ROWS, FormalCommunicationProfile)
from .predicate_trajectory_data import (
    CandidateTrajectory, EpisodeContract, FieldCoordinate, ProducerOwnerBinding,
    ProducerRowKey, PredicateQuery, FuturePredicateEvaluator,
    _receiver_operand_nodes, direct_fields, formula_fields,
    producer_rows_from_index)
from .state_dependencies import (
    DIRECT_PREDICATE_PRODUCERS,
    builder_row_index, direct_predicate_bindings, load_dependency_specs)
from .unified_contracts import (
    ContractError, FieldSchema, PredicateContract, TRUTH_DOMAIN,
    field_owner_roles, load_unified_contracts)

REPO_ROOT = Path(__file__).resolve().parents[3]
OBJECTIVE_ROOT = "aw_data/objective_semantic_truth"
RENDER_ROOT = "aw_data/render_ready_episodes"
COMMUNICATION_ARTIFACT = "communication_state.jsonl"
COMPUTE_ARTIFACT = "compute_state.jsonl"
# Canonical published objective episode streams. The objective episode root
# publishes both domain streams; the supplement directory is not a source.
OBJECTIVE_DOMAIN_FILES = ("domain_state_observations.jsonl", "l0_predicate_state.jsonl")
DIRECT_ARTIFACT = "compute_predicate_truth.jsonl"
UTM_LOG_ARTIFACT = "simulation_logs/utm_log.jsonl"
FRAME_ARTIFACT = "truth_frames.jsonl"
WEATHER_ARTIFACT = "weather_meta.jsonl"
RENDER_FILES = ("truth_frames.jsonl", "scenario_plan.json", "weather_meta.jsonl")
OBJECTIVE_FILES = ("communication_state.jsonl", "compute_state.jsonl", "manifest.json",
                   DIRECT_ARTIFACT, UTM_LOG_ARTIFACT, *OBJECTIVE_DOMAIN_FILES)

# The four statuses of one typed source lookup, plus the admitted case.
RECORD_STATUSES = ("admitted", "missing_record", "unavailable_at_cutoff",
                   "absent_projection", "non_executable_authority")


class SourceAuthorityError(ContractError):
    """An admitted source row, projection or producer identity is invalid."""


@dataclass(frozen=True)
class RowHeader:
    """Typed index entry of one source line; the payload stays on disk."""
    artifact: str
    episode_id: str
    tick: int
    owners: tuple[str, ...]
    line_number: int
    offset: int
    scope_active: bool = True
    availability_tick: int | None = None

    def __post_init__(self):
        if type(self.tick) is not int or self.tick < 0 or not self.episode_id:
            raise SourceAuthorityError("source header requires an exact nonnegative tick and episode")
        if self.availability_tick is not None and (type(self.availability_tick) is not int
                                                  or self.availability_tick < self.tick):
            raise SourceAuthorityError("source availability must be an integer at or after sample tick")

    @property
    def source_ref(self) -> str:
        return f"{self.artifact}#line={self.line_number}"


@dataclass(frozen=True)
class OperandCell:
    """One typed operand read with its exact availability identity."""
    artifact: str
    field_path: str
    owners: tuple[str, ...]
    tick: int
    status: str
    value: Any
    reason: str
    source_ref: str = ""
    availability_tick: int | None = None

    def __post_init__(self):
        if self.status not in RECORD_STATUSES:
            raise ContractError(f"unknown operand record status: {self.status}")
        if self.status != "admitted" and self.value is not None:
            raise ContractError("a non-admitted operand keeps no value")

    @property
    def known(self) -> bool:
        return self.status == "admitted"


@dataclass(frozen=True)
class ProducerOperandRow:
    """Declared producer operand row for one query/tick, or an explicit gap."""
    query_index: int
    tick: int
    row: Mapping[str, Any] | None
    cells: tuple[OperandCell, ...]
    reason: str = ""

    @property
    def known(self) -> bool:
        return self.row is not None


class _LineFile:
    """Byte-offset index of one JSONL artifact; exact-line reads only."""

    def __init__(self, path: Path, artifact: str):
        if not path.is_file():
            raise SourceAuthorityError(f"declared source artifact is missing: {path}")
        self.path, self.artifact = path, artifact
        self._offsets: list[int] = []
        offset = 0
        with path.open("rb") as stream:
            for line in stream:
                self._offsets.append(offset)
                offset += len(line)
        self._offsets = tuple(self._offsets)
        self._count = len(self._offsets)

    def payload(self, header: RowHeader) -> dict[str, Any]:
        if header.line_number < 1 or header.line_number > self._count:
            raise SourceAuthorityError(f"indexed line is outside {self.path}")
        if header.offset != self._offsets[header.line_number - 1]:
            raise SourceAuthorityError("supplied offset differs from the indexed line")
        with self.path.open("rb") as stream:
            stream.seek(header.offset)
            line = stream.readline()
        row = json.loads(line.decode("utf-8"))
        if not isinstance(row, dict):
            raise SourceAuthorityError(f"{self.path} line {header.line_number} is not an object")
        return row


def _immutable(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    return value


def _episode_of(row: Mapping[str, Any], episode_id: str, artifact: str) -> bool:
    # The current weather schema is episode-scoped by its indexed artifact,
    # and publishes tick/weather fields without an embedded episode_id.
    if artifact == WEATHER_ARTIFACT and "episode_id" not in row:
        return True
    return row.get("episode_id") == episode_id


def sources_payload_equal(store_a: "_LineFile", header_a: RowHeader,
                          store_b: "_LineFile", line_b: int) -> bool:
    """Exact payload equality of two indexed lines (cross-stream duplicates)."""
    return store_a.payload(header_a) == store_b.payload(
        RowHeader(store_b.artifact, header_a.episode_id, header_a.tick, header_a.owners,
                  line_b, store_b._offsets[line_b - 1]))

class EpisodeSources:
    """Immutable typed header indexes over one admitted episode's files.

    The loader performs one bounded header pass per declared artifact of one
    episode and never scans a repository, a corpus, or row payloads. Explicit
    supplied indexes (``from_explicit``) admit other episodes/families without
    re-deriving anything here.
    """

    def __init__(self, episode_id: str, *, files: Mapping[str, _LineFile],
                 communication_by_tick_actor: Mapping[tuple[int, str], RowHeader],
                 communication_by_identity: Mapping[tuple[str, int, str], RowHeader],
                 compute_by_tick_node: Mapping[tuple[int, str], RowHeader],
                 domain_by_tick_family_subject: Mapping[tuple[int, str, str], RowHeader],
                 direct_by_tick_predicate: Mapping[tuple[int, str, tuple[tuple[str, str], ...]], RowHeader],
                 frames_by_tick: Mapping[int, RowHeader],
                 weather_by_tick: Mapping[int, RowHeader],
                 manifest: Mapping[str, Any],
                 utm_by_tick_family_key: Mapping[tuple[int, str, tuple[tuple[str, str], ...]], RowHeader] =
                 MappingProxyType({}),
                 supplied_families: Mapping[str, str] = MappingProxyType({})):
        self.episode_id = episode_id
        self._files = MappingProxyType(dict(files))
        self._communication_by_tick_actor = MappingProxyType(dict(communication_by_tick_actor))
        self._communication_by_identity = MappingProxyType(dict(communication_by_identity))
        self._compute_by_tick_node = MappingProxyType(dict(compute_by_tick_node))
        self._domain_by_tick_family_subject = MappingProxyType(dict(domain_by_tick_family_subject))
        self._direct_by_tick_predicate = MappingProxyType(dict(direct_by_tick_predicate))
        self._frames_by_tick = MappingProxyType(dict(frames_by_tick))
        self._weather_by_tick = MappingProxyType(dict(weather_by_tick))
        self._utm_by_tick_family_key = MappingProxyType(dict(utm_by_tick_family_key))
        self.manifest = _immutable(manifest)
        self.supplied_families = MappingProxyType(dict(supplied_families))
        domain_ticks = defaultdict(dict)
        domain_families = defaultdict(list)
        for key, header in self._domain_by_tick_family_subject.items():
            domain_ticks[key[0]][key] = header
            domain_families[(key[0], key[1])].append(header)
        self._domain_ticks = MappingProxyType({tick: MappingProxyType(rows)
                                               for tick, rows in domain_ticks.items()})
        self._domain_families = MappingProxyType({key: tuple(rows)
                                                  for key, rows in domain_families.items()})
        by_tick = defaultdict(dict)
        for key, rows in self._domain_families.items():
            by_tick[key[0]][key] = rows
        self._domain_families_by_tick = MappingProxyType({
            tick: MappingProxyType(rows) for tick, rows in by_tick.items()})
        self._compute_by_actor = MappingProxyType({(header.tick, header.owners[0]): header
                                                   for header in compute_by_tick_node.values()})
        # Payload-derived views are lazily built once from the indexed headers
        # on first use, never re-scanned per row at query time.
        self._lazy_compute_by_identity = None
        self._lazy_record_by_binding = None
        self._lazy_domain_by_observation_id = None
        self._record_contracts = tuple(
            (contract.id, contract.grounding_spec["candidate_authority"]["source"],
             tuple((role.key, contract.grounding_spec["role_sources"][role.key]["path"])
                   for role in contract.roles))
            for contract in load_unified_contracts().values()
            if contract.executable
            and all(item["kind"] == "record"
                    for item in contract.grounding_spec["role_sources"].values()))

    def _view(self, name):
        cached = getattr(self, f"_lazy_{name}")
        if cached is not None:
            return cached
        built = self._build_view(name)
        object.__setattr__(self, f"_lazy_{name}", built)
        return built

    def _build_view(self, name):
        from Dataset.semantic_truth.world_truth import _value_at_path, MISSING
        if name == "compute_by_identity":
            view = {}
            for header in self._compute_by_tick_node.values():
                payload = self.payload(header)
                for field in ("node_id", "task_id", "queue_id", "resource_id"):
                    identity = payload.get(field)
                    if isinstance(identity, str):
                        view[(field, header.tick, identity)] = header
            return MappingProxyType(view)
        if name == "record_by_binding":
            view = {}
            for pid, family, role_paths in self._record_contracts:
                for (tick, row_family), rows in self._domain_families.items():
                    if row_family != family:
                        continue
                    for header in rows:
                        payload = self.payload(header)
                        values = tuple((role, _value_at_path(payload, path))
                                       for role, path in role_paths)
                        if any(value is MISSING or value is None for _, value in values):
                            continue
                        key = (tick, pid, values)
                        if key in view:
                            raise SourceAuthorityError(f"duplicate exact authority record {key}")
                        view[key] = header
            return MappingProxyType(view)
        if name == "domain_by_observation_id":
            view = {}
            for header in self._domain_by_tick_family_subject.values():
                observation_id = self.payload(header).get("observation_id")
                if isinstance(observation_id, str) and observation_id:
                    key = (header.tick, observation_id)
                    if key in view:
                        raise SourceAuthorityError(f"duplicate exact observation identity {key}")
                    view[key] = header
            return MappingProxyType(view)
        raise SourceAuthorityError(f"undeclared lazy view: {name}")



    @classmethod
    def index_episode(cls, episode_id: str, *, repo_root: Path = REPO_ROOT,
                      objective_root: str = OBJECTIVE_ROOT,
                      render_root: str = RENDER_ROOT) -> "EpisodeSources":
        """Header-index the declared single-episode file set.

        Allowed objective reads: communication_state.jsonl, compute_state.jsonl
        and manifest.json of the objective truth episode; allowed render reads:
        truth_frames.jsonl, scenario_plan.json and weather_meta.jsonl of the
        same-episode render package. The domain observation family is read from
        the declared same-episode supplement path when it exists.
        """
        if Path(episode_id).name != episode_id or not episode_id:
            raise SourceAuthorityError("episode identity must be one exact path component")
        objective = Path(repo_root) / objective_root / episode_id
        render = Path(repo_root) / render_root / episode_id
        for name in OBJECTIVE_FILES:
            if not (objective / name).is_file():
                raise SourceAuthorityError(f"objective source is missing: {objective / name}")
        for name in RENDER_FILES:
            if not (render / name).is_file():
                raise SourceAuthorityError(f"render source is missing: {render / name}")
        files: dict[str, _LineFile] = {}
        manifest = json.loads((objective / "manifest.json").read_text())
        if manifest.get("episode_id") != episode_id:
            raise SourceAuthorityError("objective manifest belongs to another episode")
        comm_actor: dict[tuple[int, str], RowHeader] = {}
        comm_identity: dict[tuple[str, int, str], RowHeader] = {}
        compute_rows: dict[tuple[int, str], RowHeader] = {}
        domain_rows: dict[tuple[int, str, str], RowHeader] = {}
        direct_rows: dict[tuple[int, str, tuple[tuple[str, str], ...]], RowHeader] = {}
        utm_rows: dict[tuple[int, str, tuple[tuple[str, str], ...]], RowHeader] = {}
        frames: dict[int, RowHeader] = {}
        weather: dict[int, RowHeader] = {}
        comm_keys = ("session_id", "flow_id", "retransmission_id",
                     "handover_id", "channel_allocation_id", "transmission_attempt_id")
        direct_template_roles = _direct_template_roles()
        for artifact, path in ((COMMUNICATION_ARTIFACT, objective / COMMUNICATION_ARTIFACT),
                               (COMPUTE_ARTIFACT, objective / COMPUTE_ARTIFACT)):
            store = _LineFile(path, artifact)
            files[artifact] = store
            with path.open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    if not isinstance(row, dict) or not _episode_of(row, episode_id, artifact):
                        raise SourceAuthorityError(f"{artifact} line {line_number} differs from its episode authority")
                    tick = row.get("tick")
                    actor = row.get("entity_id")
                    if type(tick) is not int or not isinstance(actor, str) or not actor:
                        raise SourceAuthorityError(f"{artifact} line {line_number} lacks exact tick/entity")
                    header = RowHeader(artifact, episode_id, tick, (actor,), line_number,
                                       store._offsets[line_number - 1])
                    if artifact == COMMUNICATION_ARTIFACT:
                        if (tick, actor) in comm_actor:
                            raise SourceAuthorityError(f"duplicate communication row {actor}@{tick}")
                        comm_actor[(tick, actor)] = header
                        for key in comm_keys:
                            identity = row.get(key)
                            if isinstance(identity, str) and identity:
                                slot = (key, tick, identity)
                                if slot in comm_identity and comm_identity[slot] != header:
                                    raise SourceAuthorityError(f"conflicting communication identity {slot}")
                                comm_identity[slot] = header
                    else:
                        node = row.get("node_id")
                        if not isinstance(node, str) or not node:
                            raise SourceAuthorityError(f"{artifact} line {line_number} lacks node_id")
                        if (tick, node) in compute_rows:
                            raise SourceAuthorityError(f"duplicate compute row {node}@{tick}")
                        compute_rows[(tick, node)] = header
        # Both canonical published domain streams are indexed once; each header
        # keeps its physical artifact (file name) and payload observation
        # family/subject identity. The two streams overlap exactly (verified:
        # duplicate keys across streams carry identical payloads), so one
        # combined (tick, family, subject) index admits every published family
        # and never groups an unfamiliar artifact name as a family.
        for stream in OBJECTIVE_DOMAIN_FILES:
            path = objective / stream
            if not path.is_file():
                raise SourceAuthorityError(f"canonical objective domain stream is missing: {path}")
            store = _LineFile(path, stream)
            files[stream] = store
            with path.open("rb") as stream_bytes:
                for line_number, line in enumerate(stream_bytes, 1):
                    row = json.loads(line)
                    if not isinstance(row, dict) or not _episode_of(row, episode_id, stream):
                        raise SourceAuthorityError(f"{stream} line {line_number} differs from its episode authority")
                    tick = row.get("tick")
                    family = row.get("observation_family")
                    subject = row.get("subject_id")
                    if type(tick) is not int or not isinstance(family, str) or not isinstance(subject, str):
                        raise SourceAuthorityError(f"{stream} line {line_number} lacks exact family identity")
                    key = (tick, family, subject)
                    known = domain_rows.get(key)
                    if known is not None:
                        # The two canonical streams overlap exactly; a repeated
                        # key must carry an identical payload, never a conflict.
                        if known.artifact == stream:
                            continue
                        prior_row = files[known.artifact].payload(known)
                        if prior_row == row:
                            continue
                        raise SourceAuthorityError(f"conflicting domain observation {key} in {stream}")
                    domain_rows[key] = RowHeader(stream, episode_id, tick,
                                                 (family, subject), line_number,
                                                 store._offsets[line_number - 1])
        direct_path = objective / DIRECT_ARTIFACT
        if not direct_path.is_file():
            raise SourceAuthorityError(f"canonical direct producer truth is missing: {direct_path}")
        store = _LineFile(direct_path, DIRECT_ARTIFACT)
        files[DIRECT_ARTIFACT] = store
        with direct_path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if not isinstance(row, dict) or not _episode_of(row, episode_id, DIRECT_ARTIFACT):
                    raise SourceAuthorityError("direct producer row differs from its episode authority")
                tick = row.get("tick")
                pid = row.get("predicate_id")
                bindings = row.get("bindings")
                if type(tick) is not int or not isinstance(pid, str) or not isinstance(bindings, Mapping):
                    raise SourceAuthorityError(f"{DIRECT_ARTIFACT} line {line_number} lacks exact direct identity")
                roles = direct_template_roles.get(pid)
                if roles is None or set(bindings) != set(roles):
                    raise SourceAuthorityError(f"{DIRECT_ARTIFACT} line {line_number} role order differs from the registry")
                canonical = tuple((role, str(bindings[role])) for role in roles)
                key = (tick, pid, canonical)
                if key in direct_rows:
                    raise SourceAuthorityError(f"duplicate direct truth row {key}")
                direct_rows[key] = RowHeader(DIRECT_ARTIFACT, episode_id, tick,
                                             (pid, *(value for _, value in canonical)),
                                             line_number, store._offsets[line_number - 1])
        # UTM physical container view: one outer simulation log whose
        # source_record_kind names the declared inner family exactly
        # (family + ".jsonl", verified against the manifest
        # utm_supplement_summary.record_counts), with each inner state payload
        # in row["state"] carrying its own episode/tick/role identifiers.
        utm_path = objective / UTM_LOG_ARTIFACT
        if not utm_path.is_file():
            raise SourceAuthorityError(f"canonical UTM state container is missing: {utm_path}")
        utm_store = _LineFile(utm_path, UTM_LOG_ARTIFACT)
        files[UTM_LOG_ARTIFACT] = utm_store
        summary = manifest.get("utm_supplement_summary")
        counts = summary.get("record_counts") if isinstance(summary, Mapping) else None
        if not isinstance(counts, Mapping) or not counts:
            raise SourceAuthorityError("objective manifest lacks declared utm_supplement_summary.record_counts")
        declared_utm = {str(name): int(value) for name, value in counts.items()}
        utm_observed: dict[str, int] = {}
        with utm_path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if not isinstance(row, dict) or not _episode_of(row, episode_id, UTM_LOG_ARTIFACT):
                    raise SourceAuthorityError(f"{UTM_LOG_ARTIFACT} line {line_number} differs from its episode authority")
                kind = row.get("source_record_kind")
                family = f"{kind}.jsonl" if isinstance(kind, str) and kind else None
                if family not in declared_utm:
                    raise SourceAuthorityError(f"{UTM_LOG_ARTIFACT} line {line_number} names an undeclared UTM family {family!r}")
                utm_observed[family] = utm_observed.get(family, 0) + 1
                state = row.get("state")
                tick = state.get("tick") if isinstance(state, Mapping) else None
                if type(tick) is not int:
                    raise SourceAuthorityError(f"{UTM_LOG_ARTIFACT} line {line_number} state lacks an exact tick")
                if isinstance(state, Mapping) and state.get("episode_id") not in (None, episode_id):
                    raise SourceAuthorityError(f"{UTM_LOG_ARTIFACT} line {line_number} state belongs to another episode")
                key = (tick, family, tuple(sorted((role, str(value)) for role, value
                                                  in state.items() if role.endswith("_id")
                                                  and isinstance(value, str))))
                if key in utm_rows:
                    raise SourceAuthorityError(f"duplicate UTM state identity {key[:2]}")
                utm_rows[key] = RowHeader(UTM_LOG_ARTIFACT, episode_id, tick,
                                          (family, *(value for _, value in key[2])),
                                          line_number, utm_store._offsets[line_number - 1])
        observed_utm = {family: count for family, count in utm_observed.items()}
        if observed_utm != declared_utm:
            raise SourceAuthorityError(f"UTM container counts differ from the manifest: {observed_utm} vs {declared_utm}")
        for artifact, path, sink in ((FRAME_ARTIFACT, render / FRAME_ARTIFACT, frames),
                                     (WEATHER_ARTIFACT, render / WEATHER_ARTIFACT, weather)):
            store = _LineFile(path, artifact)
            files[artifact] = store
            with path.open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    if not isinstance(row, dict) or not _episode_of(row, episode_id, artifact):
                        raise SourceAuthorityError(f"{artifact} line {line_number} differs from its episode authority")
                    tick = row.get("tick")
                    if type(tick) is not int:
                        raise SourceAuthorityError(f"{artifact} line {line_number} lacks an integer tick")
                    if tick in sink:
                        raise SourceAuthorityError(f"duplicate {artifact} tick {tick}")
                    sink[tick] = RowHeader(artifact, episode_id, tick, (), line_number,
                                           store._offsets[line_number - 1])
        return cls(episode_id, files=files, communication_by_tick_actor=comm_actor,
                   communication_by_identity=comm_identity, compute_by_tick_node=compute_rows,
                   domain_by_tick_family_subject=domain_rows, direct_by_tick_predicate=direct_rows,
                   frames_by_tick=frames, weather_by_tick=weather, manifest=manifest,
                   utm_by_tick_family_key=utm_rows)

    @classmethod
    def from_explicit(cls, episode_id: str, *, files: Mapping[str, _LineFile],
                      headers: Mapping[str, Sequence[RowHeader]],
                      manifest: Mapping[str, Any],
                      supplied_families: Mapping[str, str]) -> "EpisodeSources":
        """Admit caller-supplied typed indexes for other episodes/families.

        Headers must already carry exact episode/tick/owner identities. The
        adapters below never re-scan payloads or re-derive any projection.
        """
        communication_by_tick_actor: dict[tuple[int, str], RowHeader] = {}
        communication_by_identity: dict[tuple[str, int, str], RowHeader] = {}
        compute_by_tick_node: dict[tuple[int, str], RowHeader] = {}
        domain_by_tick_family_subject: dict[tuple[int, str, str], RowHeader] = {}
        direct_by_tick_predicate: dict[tuple[int, str, tuple[tuple[str, str], ...]], RowHeader] = {}
        frames_by_tick: dict[int, RowHeader] = {}
        weather_by_tick: dict[int, RowHeader] = {}
        for artifact, rows in headers.items():
            if artifact not in files:
                raise SourceAuthorityError(f"supplied artifact has no indexed file: {artifact}")
            for header in rows:
                if not isinstance(header, RowHeader) or header.artifact != artifact:
                    raise SourceAuthorityError("supplied header differs from its declared artifact")
                if header.episode_id != episode_id or type(header.tick) is not int or header.tick < 0:
                    raise SourceAuthorityError("supplied header has an invalid episode/tick")
                row = files[artifact].payload(header)
                if not _episode_of(row, episode_id, artifact) or row.get("tick") != header.tick:
                    raise SourceAuthorityError("supplied payload differs from its header episode/tick")
                if artifact == COMMUNICATION_ARTIFACT:
                    sink, key = communication_by_tick_actor, (header.tick, row["entity_id"])
                    if header.owners != (row["entity_id"],):
                        raise SourceAuthorityError("communication header differs from its actor")
                    for field in ("session_id", "flow_id", "retransmission_id",
                                  "handover_id", "channel_allocation_id", "transmission_attempt_id"):
                        identity = row.get(field)
                        if isinstance(identity, str) and identity:
                            slot = (field, header.tick, identity)
                            if slot in communication_by_identity:
                                raise SourceAuthorityError(f"duplicate supplied communication identity {slot}")
                            communication_by_identity[slot] = header
                elif artifact == COMPUTE_ARTIFACT:
                    sink, key = compute_by_tick_node, (header.tick, row["node_id"])
                    if header.owners != (row["entity_id"],):
                        raise SourceAuthorityError("compute header differs from its actor")
                elif artifact == DIRECT_ARTIFACT:
                    roles = _direct_template_roles()[row["predicate_id"]]
                    sink = direct_by_tick_predicate
                    key = (header.tick, row["predicate_id"],
                           tuple((role, row["bindings"][role]) for role in roles))
                elif artifact in (FRAME_ARTIFACT, WEATHER_ARTIFACT):
                    sink = frames_by_tick if artifact == FRAME_ARTIFACT else weather_by_tick
                    key = header.tick
                elif artifact in OBJECTIVE_DOMAIN_FILES or artifact in supplied_families:
                    domain = artifact in OBJECTIVE_DOMAIN_FILES
                    family = row["observation_family"] if domain else artifact
                    subject = row["subject_id"] if domain else header.owners[0]
                    sink, key = domain_by_tick_family_subject, (header.tick, family, subject)
                else:
                    raise SourceAuthorityError(f"supplied artifact has no declared adapter: {artifact}")
                if key in sink:
                    raise SourceAuthorityError(f"duplicate supplied source identity {key}")
                sink[key] = header
        return cls(episode_id, files=files,
                   communication_by_tick_actor=communication_by_tick_actor,
                   communication_by_identity=communication_by_identity,
                   compute_by_tick_node=compute_by_tick_node,
                   domain_by_tick_family_subject=domain_by_tick_family_subject,
                   direct_by_tick_predicate=direct_by_tick_predicate,
                   frames_by_tick=frames_by_tick, weather_by_tick=weather_by_tick,
                   manifest=manifest, supplied_families=supplied_families)

    def payload(self, header: RowHeader) -> dict[str, Any]:
        store = self._files.get(header.artifact)
        if store is None:
            raise SourceAuthorityError(f"artifact has no indexed file: {header.artifact}")
        return store.payload(header)

    # -- exact typed row access ------------------------------------------------
    def communication_row(self, tick: int, actor: str) -> RowHeader | None:
        return self._communication_by_tick_actor.get((tick, actor))

    def communication_row_by_identity(self, row_key: str, tick: int, identity: str) -> RowHeader | None:
        return self._communication_by_identity.get((row_key, tick, identity))

    def compute_row(self, tick: int, node_id: str) -> RowHeader | None:
        return self._compute_by_tick_node.get((tick, node_id))

    def domain_row(self, tick: int, family: str, subject_id: str) -> RowHeader | None:
        return self._domain_by_tick_family_subject.get((tick, family, subject_id))

    def direct_row(self, tick: int, predicate_id: str, bindings: Mapping[str, str],
                   declared_roles: Sequence[str]) -> RowHeader | None:
        canonical = tuple((role, str(bindings[role])) for role in declared_roles)
        return self._direct_by_tick_predicate.get((tick, predicate_id, canonical))

    def frame_row(self, tick: int) -> RowHeader | None:
        return self._frames_by_tick.get(tick)

    def weather_row(self, tick: int) -> RowHeader | None:
        return self._weather_by_tick.get(tick)

    def utm_state_row(self, tick: int, family: str,
                      role_identities: Mapping[str, str]) -> RowHeader | None:
        """Exact UTM state view keyed by tick, inner family and role identities.

        A state row is selected only when every supplied declared role path
        matches that row's identity exactly; the remaining identity columns of
        the row are not part of the lookup key, so no approximate field match
        ever selects a row and no role product is enumerated.
        """
        exact = (tick, family,
                 tuple(sorted((role, str(value)) for role, value in role_identities.items())))
        header = self._utm_by_tick_family_key.get(exact)
        if header is not None:
            return header
        if not role_identities:
            return None
        row_identities = frozenset((role, str(value)) for role, value in role_identities.items())
        for key, header in self._utm_by_tick_family_key.items():
            if key[0] != tick or key[1] != family:
                continue
            if row_identities <= frozenset(key[2]):
                return header
        return None


def _producer_header(sources, contract, bindings, tick):
    authority = contract.grounding_spec["candidate_authority"]["source"]
    binding = direct_predicate_bindings(authority)[contract.id]
    reads = binding.role_row_reads
    primary = next((read for read in reads if read.row_key != "station_id"), None)
    if primary is None:
        raise SourceAuthorityError("producer relation lacks a unique row identity")
    identity = bindings[primary.role].entity_id
    if binding.row_builder == "_build_communication_rows":
        header = (sources.communication_row(tick, identity) if primary.row_key == "entity_id"
                  else sources.communication_row_by_identity(primary.row_key, tick, identity))
    else:
        header = sources._view('compute_by_identity').get((primary.row_key, tick, identity))
    if header is None:
        return None
    payload = sources.payload(header)
    if any(payload.get(read.row_key) != bindings[read.role].entity_id for read in reads):
        return None
    return header



def _producer_actor(sources, row, tick, class_catalog):
    actor_id = row["entity_id"]
    if class_catalog is not None:
        classes = class_catalog.classes.get(actor_id)
        if not classes:
            raise SourceAuthorityError(f"producer actor {actor_id} has no typed instance authority")
        return class_catalog.resolve(TypedEntity(actor_id, classes[0], classes))
    # The communication producer's exact actor category gate is UAV. Compute
    # actors require an observed frame's entity category, not identifier guesses.
    frame = sources.frame_row(tick)
    if frame is not None and frame.availability_tick is not None and frame.availability_tick > tick:
        raise SourceAuthorityError("producer actor type is unavailable in the observed prefix")
    entities = sources.payload(frame)["entities"] if frame is not None else ()
    entity = next((item for item in entities if item["entity_id"] == actor_id), None)
    from Dataset.semantic_truth.world_truth import _ScopeEntity, _scope_ontology_class, _entity_category
    ontology_class = _scope_ontology_class(_ScopeEntity(_entity_category(entity), actor_id)) if entity else None
    if ontology_class is None:
        raise SourceAuthorityError(f"producer actor {actor_id} lacks an observed category authority")
    return TypedEntity(actor_id, ontology_class)


def _direct_template_roles() -> Mapping[str, tuple[str, ...]]:
    """Registry role order per direct predicate, from the frozen templates."""
    from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates
    return {item["id"]: tuple(role["key"] for role in item["argument_roles"])
            for item in get_core_predicate_templates()}


# --------------------------------------------------------------------------
# Record status classification and receiver operand adapters
# --------------------------------------------------------------------------

def classify_record(header: RowHeader | None, *, cutoff_tick: int,
                    projection_form: str | None = None) -> tuple[str, str]:
    """Exact status of one typed row lookup.

    ``missing_record``: no indexed row. ``unavailable_at_cutoff``: the row's
    sample tick is after the causal cutoff. ``absent_projection``: the row
    exists but the declared receiver does not publish this operand. A
    non-executable authority never reaches this adapter.
    """
    if header is None:
        return "missing_record", "no indexed source row carries this exact identity"
    if header.tick > cutoff_tick or (header.availability_tick is not None and header.availability_tick > cutoff_tick):
        return "unavailable_at_cutoff", f"source row tick {header.tick} is after cutoff {cutoff_tick}"
    if projection_form == "unobserved_monitor":
        return "absent_projection", "declared receiver projection is unobserved_monitor"
    if projection_form is None:
        return "missing_record", "no declared receiver projection for this field"
    return "admitted", f"declared receiver projection {projection_form}"


def _scalar(value: Any, *, data_type: str, where: str) -> Any:
    if data_type in ("categorical", "identifier"):
        if not isinstance(value, str) or not value:
            raise SourceAuthorityError(f"published symbolic value requires a nonempty string: {where}")
        return value
    if data_type == "boolean":
        if isinstance(value, bool):
            return value
        if value in (0, 1):
            return bool(value)
        raise SourceAuthorityError(f"published boolean is not 0/1: {where}")
    if data_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SourceAuthorityError(f"published count is not a nonnegative integer: {where}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise SourceAuthorityError(f"published scalar requires a finite number: {where}")
    return value


def receiver_operand_cell(sources: EpisodeSources, *, artifact: str, field_path: str,
                          owners: tuple[str, ...], tick: int, cutoff_tick: int,
                          data_type: str, parameter_profile: ParameterProfile) -> OperandCell:
    """One declared receiver operand through the exact published-field adapter.

    communication_state.jsonl fields read their exact row sections; the two
    quality thresholds come from the named governing communication profile;
    compute_state.jsonl fields read their ``compute.`` section; the traffic
    supplement fields map through RECEIVER_FIELD_NAMES. Unpublished compute
    operands stay ``absent_projection``; unknown marker values stay missing.
    """
    if tick > cutoff_tick:
        return OperandCell("causal_cutoff", "", (), tick, "unavailable_at_cutoff", None,
                           f"sample tick {tick} is after cutoff {cutoff_tick}")

    _, _, reference_roles, _ = RECEIVER_ROWS[artifact]
    header: RowHeader | None
    row: dict[str, Any] | None = None
    if artifact == COMMUNICATION_ARTIFACT:
        header = sources.communication_row(tick, owners[0])
        if header is not None and len(owners) != len(reference_roles):
            raise SourceAuthorityError("communication owners differ from the declared row roles")
        if header is not None:
            row = sources.payload(header)
            if row.get("station_id") != owners[1]:
                return OperandCell(artifact, field_path, owners, tick, "missing_record", None,
                                   "communication row station identity differs from the typed owner",
                                   header.source_ref)
    elif artifact == COMPUTE_ARTIFACT:
        header = sources._compute_by_actor.get((tick, owners[0]))
        if header is not None:
            row = sources.payload(header)
    else:
        family = "road_segment_state" if field_path == "values.road_closed" else "traffic_signal_state"
        mapped = RECEIVER_FIELD_NAMES.get((artifact, field_path))
        if mapped is None:
            raise SourceAuthorityError(f"traffic supplement has no declared receiver field: {field_path}")
        header = sources.domain_row(tick, family, owners[0])
        if header is not None:
            status, reason = classify_record(header, cutoff_tick=cutoff_tick, projection_form="published_monitor")
            if status != "admitted" or not header.scope_active:
                return OperandCell(artifact, field_path, owners, tick, "unavailable_at_cutoff", None,
                                   reason, header.source_ref, header.availability_tick)
            row = sources.payload(header)
            values = row
            path = field_path
            current: Any = values
            for part in path.split("."):
                if not isinstance(current, Mapping) or part not in current:
                    current = None
                    break
                current = current[part]
            from Dataset.semantic_truth.world_truth import _missing_value
            if _missing_value(current):
                return OperandCell(artifact, field_path, owners, tick, "missing_record", None,
                                   f"published receiver field missing at {header.source_ref}",
                                   header.source_ref)
            return OperandCell(artifact, field_path, owners, tick, "admitted",
                               _scalar(current, data_type=data_type, where=field_path),
                               "declared traffic-manager monitor field", header.source_ref, header.availability_tick if header.availability_tick is not None else tick)
        status, reason = classify_record(None, cutoff_tick=cutoff_tick)
        return OperandCell(artifact, field_path, owners, tick, status, None, reason)
    if header is not None and ((header.availability_tick is not None and header.availability_tick > cutoff_tick)
                               or not header.scope_active):
        return OperandCell(header.artifact, "", header.owners, tick, "unavailable_at_cutoff", None,
                           "exact source record is unavailable at cutoff", header.source_ref,
                           header.availability_tick)
    if header is None or row is None:
        status, reason = classify_record(header, cutoff_tick=cutoff_tick)
        return OperandCell(artifact, field_path, owners, tick, status, None, reason,
                           header.source_ref if header is not None else "")
    if row.get("scope_active") is False:
        return OperandCell(artifact, field_path, owners, tick, "unavailable_at_cutoff", None,
                           "published row marks its scope inactive", header.source_ref)
    if field_path in ("quality_thresholds.latency_ms", "quality_thresholds.packet_loss_ratio"):
        name = "latency_threshold_ms" if field_path.endswith("latency_ms") else "packet_loss_threshold_ratio"
        from .state_dependencies import communication_row_actor_category
        profile = parameter_profile.parameters.get(name)
        if profile is None or type(profile) not in (int, float):
            raise SourceAuthorityError(f"governing communication profile lacks {name}")
        return OperandCell(artifact, field_path, owners, tick, "admitted", round(float(profile), 6),
                           "episode manifest governing communication flow profile",
                           parameter_profile.source_ref, tick)
    if field_path == "requirement.status":
        status = str(row.get("requirement", {}).get("status"))
        if status not in ("source_required", "not_required"):
            raise SourceAuthorityError("communication row requirement status is undeclared")
        return OperandCell(artifact, field_path, owners, tick, "admitted", status,
                           "observation_logs.generate requirement gate", header.source_ref, header.availability_tick if header.availability_tick is not None else tick)
    if field_path == "failure.requirement_status":
        status = str(row.get("failure", {}).get("requirement_status"))
        if status not in ("missing_source_record", "not_required"):
            raise SourceAuthorityError("compute row requirement status is undeclared")
        return OperandCell(artifact, field_path, owners, tick, "admitted", status,
                           "compute receiver requirement gate", header.source_ref, header.availability_tick if header.availability_tick is not None else tick)
    if field_path == "handover.active":
        active = row.get("handover", {}).get("active")
        if active == "unknown" or active is None:
            return OperandCell(artifact, field_path, owners, tick, "missing_record", None,
                               "declared producer reports handover unknown", header.source_ref)
        return OperandCell(artifact, field_path, owners, tick, "admitted", _scalar(active, data_type="boolean", where=field_path),
                           "published communication handover flag", header.source_ref, header.availability_tick if header.availability_tick is not None else tick)
    current: Any = row
    for part in field_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            current = None
            break
        current = current[part]
    from Dataset.semantic_truth.world_truth import _missing_value
    if _missing_value(current):
        return OperandCell(artifact, field_path, owners, tick, "missing_record", None,
                           "published receiver field missing", header.source_ref)
    if field_path == "node_availability.available":
        return OperandCell(artifact, field_path, owners, tick, "admitted", _scalar(current, data_type="boolean", where=field_path),
                           "published compute availability flag", header.source_ref, header.availability_tick if header.availability_tick is not None else tick)
    if field_path == "execution_state":
        # Exact _build_compute_rows publisher authority ('"execution_state":
        # execution_state,'), string field with the declared producer domain
        # unknown/failed/migrating/queued/running_offloaded/running.
        if not isinstance(current, str) or not current:
            raise SourceAuthorityError("published compute execution_state is not a nonempty string")
        return OperandCell(artifact, field_path, owners, tick, "admitted", current,
                           "published compute row execution_state", header.source_ref,
                           header.availability_tick if header.availability_tick is not None else tick)
    return OperandCell(artifact, field_path, owners, tick, "admitted",
                       _scalar(current, data_type=data_type, where=field_path),
                       "declared published monitor field", header.source_ref, header.availability_tick if header.availability_tick is not None else tick)


def direct_truth_cell(sources: EpisodeSources, contract: PredicateContract,
                      bindings: Mapping[str, str], tick: int, *, cutoff_tick: int) -> OperandCell:
    """Bind the declared direct producer's own truth row, when stored.

    Legitimate direct_truth stays bound to its producer row index
    (compute_comm_supplement/predicate_truth.jsonl). An absent row means the
    operand is missing, never false; an incomplete binding row is explicit.
    """
    if tick > cutoff_tick:
        return OperandCell("causal_cutoff", "", (), tick, "unavailable_at_cutoff", None,
                           f"sample tick {tick} is after cutoff {cutoff_tick}")

    authority = contract.grounding_spec["candidate_authority"]
    if authority["kind"] != "direct_predicate_row":
        raise ContractError(f"{contract.id} has no direct producer authority")
    if authority["source"] not in DIRECT_PREDICATE_PRODUCERS:
        return OperandCell(DIRECT_ARTIFACT, "value", tuple(bindings.values()), tick,
                           "non_executable_authority", None,
                           f"no declared producer for authority {authority['source']}")
    roles = tuple(role.key for role in contract.roles)
    header = sources.direct_row(tick, contract.id, bindings, roles)
    if header is None:
        status, reason = classify_record(None, cutoff_tick=cutoff_tick)
        return OperandCell(DIRECT_ARTIFACT, "value", tuple(bindings[role] for role in roles),
                           tick, status, None, reason)
    if header is not None and ((header.availability_tick is not None and header.availability_tick > cutoff_tick)
                               or not header.scope_active):
        return OperandCell(header.artifact, "", header.owners, tick, "unavailable_at_cutoff", None,
                           "exact source record is unavailable at cutoff", header.source_ref,
                           header.availability_tick)
    row = sources.payload(header)
    if row.get("binding_status") != "complete":
        return OperandCell(DIRECT_ARTIFACT, "value", tuple(bindings[role] for role in roles),
                           tick, "missing_record", None,
                           f"direct producer row binding_status is {row.get('binding_status')!r}",
                           header.source_ref)
    value = row.get("value")
    if value not in TRUTH_DOMAIN:
        raise SourceAuthorityError(f"direct producer row carries an invalid truth {value!r}")
    return OperandCell(DIRECT_ARTIFACT, "value", tuple(bindings[role] for role in roles),
                       tick, "admitted", value, "declared direct producer truth row",
                       header.source_ref, header.availability_tick if header.availability_tick is not None else tick)


def producer_operand_row(sources: EpisodeSources, contract: PredicateContract,
                         bindings: Mapping[str, TypedEntity], tick: int, *,
                         cutoff_tick: int,
                         parameter_profile: ParameterProfile) -> ProducerOperandRow:
    """Assemble the declared producer's operand row from receiver monitors.

    Fields come from direct_monitor_fields order through the receiver adapters;
    the two governed thresholds come from the named profile; the four exact
    unpublished compute operands stay absent, so the original producer function
    receives a row without them and keeps its own UNKNOWN semantics. No label
    is consulted and no category or threshold is invented.
    """
    from .ast_compiler import set_field
    from .state_dependencies import direct_monitor_fields
    binding = direct_predicate_bindings(contract.grounding_spec["candidate_authority"]["source"])[contract.id]
    nodes, receiver_complete = _receiver_operand_nodes(contract)
    header = _producer_header(sources, contract, bindings, tick)
    if header is None:
        return ProducerOperandRow(-1, tick, None, (), "missing exact producer owner relation")
    if tick > cutoff_tick or (header.availability_tick is not None and header.availability_tick > cutoff_tick):
        return ProducerOperandRow(-1, tick, None, (), "producer record unavailable at cutoff")
    payload = sources.payload(header)
    owners = ((payload["entity_id"], payload["station_id"])
              if binding.row_builder == "_build_communication_rows" else (payload["entity_id"],))
    cells: list[OperandCell] = []
    row: dict[str, Any] = {}
    complete = True
    reason_parts: list[str] = []
    for node in nodes:
        cell = node.source_cell
        if cell.form == "governed_row_parameter":
            name = ("latency_threshold_ms" if cell.field_path.endswith("latency_ms")
                    else "packet_loss_threshold_ratio")
            value = parameter_profile.parameters.get(name)
            if value is None:
                raise SourceAuthorityError(f"governing profile lacks {cell.field_path}")
            set_field(row, cell.field_path, round(float(value), 6))
            cells.append(OperandCell(cell.artifact, cell.field_path, owners, tick, "admitted",
                                     round(float(value), 6), "governing communication flow profile",
                                     parameter_profile.source_ref, tick))
            continue
        data_type = {n.source_cell.field_path: n.data_type for n in nodes}[cell.field_path]
        comm_cell = receiver_operand_cell(sources, artifact=cell.artifact,
            field_path=cell.field_path, owners=owners, tick=tick, cutoff_tick=cutoff_tick,
            data_type=data_type, parameter_profile=parameter_profile)
        cells.append(comm_cell)
        if comm_cell.known:
            set_field(row, cell.field_path, comm_cell.value)
        else:
            complete = False
            reason_parts.append(f"{cell.field_path}: {comm_cell.status} ({comm_cell.reason})")
    if not receiver_complete:
        if tick > cutoff_tick:
            return ProducerOperandRow(-1, tick, None, tuple(cells), "producer row unavailable at cutoff")
        row = dict(payload)
        complete = True
    query_index = -1
    return ProducerOperandRow(query_index, tick, row if complete else None, tuple(cells),
                              "; ".join(reason_parts))


# --------------------------------------------------------------------------
# World-truth context / record operand adapters (scope and record formulas)
# --------------------------------------------------------------------------

def scope_record_cell(sources: EpisodeSources, contract: PredicateContract, schema: FieldSchema,
                      bindings: Mapping[str, TypedEntity], tick: int, *,
                      cutoff_tick: int) -> OperandCell:
    """Read the existing world_truth context or exact declared record binding."""
    from Dataset.semantic_truth import world_truth as wt
    grounding = contract.grounding_spec
    authority = grounding["candidate_authority"]
    source_path = grounding["field_sources"][schema.field]
    owners = tuple(bindings[role].entity_id for role in field_owner_roles(contract, schema.field))
    artifact = authority["source"]
    if tick > cutoff_tick:
        return OperandCell(artifact, source_path, owners, tick, "unavailable_at_cutoff", None,
                           f"sample tick {tick} is after cutoff {cutoff_tick}")
    role_sources = grounding["role_sources"]
    if all(item["kind"] == "record" for item in role_sources.values()):
        if authority["kind"] == "utm_record":
            family = authority["source"]
            header = sources.utm_state_row(tick, family,
                                           {role_sources[role.key]["path"]: bindings[role.key].entity_id
                                            for role in contract.roles})
            record = None
            if header is not None:
                record = sources.payload(header)["state"]
        else:
            key = (tick, contract.id, tuple((role.key, bindings[role.key].entity_id)
                                          for role in contract.roles))
            header = sources._view('record_by_binding').get(key)
            record = sources.payload(header) if header is not None else None
    else:
        scope_roles = [role for role, item in role_sources.items() if item["kind"] == "scope_entity"]
        scope_type = str(contract.template["scope_types"][0])
        entity_id = bindings[scope_roles[0]].entity_id if scope_roles else owners[0]
        header = sources.frame_row(tick)
        entities = {item["entity_id"]: item for item in sources.payload(header)["entities"]} if header else {}
        domain = {key: sources.payload(item) for key, item in sources._domain_ticks.get(tick, {}).items()}
        families = {key: tuple(sources.payload(item) for item in items)
                    for key, items in sources._domain_families_by_tick.get(tick, {}).items()}
        weather = sources.weather_row(tick)
        weather_row = (sources.payload(weather) if weather is not None and weather.scope_active
                       and (weather.availability_tick is None or weather.availability_tick <= cutoff_tick) else {})
        contexts = wt._build_tick_contexts(
            episode_id=sources.episode_id, tick=tick,
            scopes=(wt._ScopeEntity(scope_type, entity_id),), roster_entities={},
            entities_at_tick=entities, weather_row=weather_row, domain_index=domain,
            domain_by_tick_family=families, compute_by_node={}, utm_index={})
        record = contexts[(scope_type, entity_id)]
        origin = record["_observation_sources"].get(source_path.split(":", 1)[1])
        if isinstance(origin, str) and "#observation_id=" in origin:
            observation_id = origin.split("#observation_id=", 1)[1]
            header = sources._view('domain_by_observation_id')[(tick, observation_id)]

        if authority["kind"] == "episode_operational_region":
            header = weather
        for role, item in role_sources.items():
            if item["kind"] == "context" and wt._value_at_path(record, item["path"]) != bindings[role].entity_id:
                return OperandCell(artifact, source_path, owners, tick, "missing_record", None,
                                   f"declared context role {role} differs from its exact binding")
    if header is not None and ((header.availability_tick is not None and header.availability_tick > cutoff_tick)
                               or not header.scope_active):
        return OperandCell(header.artifact, "", header.owners, tick, "unavailable_at_cutoff", None,
                           "exact source record is unavailable at cutoff", header.source_ref,
                           header.availability_tick)
    if record is None:
        return OperandCell(artifact, source_path, owners, tick, "missing_record", None,
                           f"no exact {artifact} record for the declared role tuple")
    if header is not None and not header.scope_active:
        return OperandCell(artifact, source_path, owners, tick, "unavailable_at_cutoff", None,
                           "indexed authority marks its scope inactive", header.source_ref)
    if not source_path.startswith(("record:", "context:")):
        raise SourceAuthorityError(f"unsupported declared field source: {source_path}")
    value = wt._value_at_path(record, source_path.split(":", 1)[1])
    if value is wt.MISSING or wt._missing_value(value):
        return OperandCell(artifact, source_path, owners, tick, "missing_record", None,
                           f"existing world_truth authority lacks {source_path}",
                           header.source_ref if header else "")
    return OperandCell(artifact, source_path, owners, tick, "admitted", value,
                       "existing world_truth projection or exact declared record",
                       header.source_ref if header else "",
                       header.availability_tick if header is not None and header.availability_tick is not None else tick)


# --------------------------------------------------------------------------
# Contract/episode compilation
# --------------------------------------------------------------------------

def governed_parameter_profile(episode_id: str, contracts: Sequence[PredicateContract], *,
                               manifest: Mapping[str, Any] | None = None) -> ParameterProfile:
    """Governed thresholds for the requested contracts; exact names only.

    Direct-communication governed row parameters resolve through the episode
    manifest's declared compute profile; all other thresholds come from the
    governed parameter defaults of the frozen registry. A missing governed
    name is an explicit error, never a substituted value.
    """
    from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults
    parameters: dict[str, float] = {}
    communication_profile: FormalCommunicationProfile | None = None
    profile_values: Mapping[str, Any] = {}
    if manifest is not None and isinstance(manifest.get("compute_profile"), Mapping):
        communication_profile = FormalCommunicationProfile.from_formal_episode(episode_id)
        flow = communication_profile.values(episode_id)
        from .state_dependencies import communication_row_actor_category
        profile_values = flow[communication_row_actor_category()]
    defaults = get_governed_parameter_defaults()
    for contract in contracts:
        names = list(contract.threshold_parameters)
        if contract.semantic_mode == "direct_observation":
            nodes, _ = _receiver_operand_nodes(contract)
            names.extend("latency_threshold_ms" if node.source_cell.field_path.endswith("latency_ms")
                         else "packet_loss_threshold_ratio" for node in nodes
                         if node.source_cell.form == "governed_row_parameter")
        for name in names:
            if name in parameters:
                continue
            if name in profile_values:
                value = profile_values[name]
                if type(value) not in (int, float) or not np.isfinite(value):
                    raise SourceAuthorityError(f"governing profile value is not finite: {name}")
                parameters[name] = round(float(value), 6)
            elif name in defaults:
                value = defaults[name]
                if type(value) not in (int, float) or not np.isfinite(value):
                    raise SourceAuthorityError(f"governed default is not finite: {name}")
                parameters[name] = round(float(value), 6)
            else:
                raise SourceAuthorityError(
                    f"{contract.id}: governed threshold {name} has no declared authority")
    source_ref = (communication_profile.manifest_ref if communication_profile is not None
                  else "Dataset/semantic_truth/core_semantic_registry.py#governed_parameter_defaults")
    profile_id = (f"objective:{episode_id}:governed+{communication_profile.profile_id}"
                  if communication_profile is not None else f"objective:{episode_id}:governed")
    return ParameterProfile(profile_id, parameters, source_ref)


def _scope_active(sources: EpisodeSources, contract: PredicateContract,
                  bindings: Mapping[str, TypedEntity], tick: int, cutoff_tick: int) -> bool | None:
    """Observed frame presence of the declared scope entity, or None when the
    predicate has no scope authority (record/row authorities)."""
    scope_roles = [role for role, item in contract.grounding_spec["role_sources"].items()
                   if item["kind"] == "scope_entity"]
    if not scope_roles:
        return None
    entity_id = bindings[scope_roles[0]].entity_id
    frame = sources.frame_row(tick)
    if frame is None or (frame.availability_tick is not None and frame.availability_tick > cutoff_tick):
        return None
    payload = sources.payload(frame)
    return any(isinstance(item, Mapping) and item.get("entity_id") == entity_id
               for item in payload.get("entities", ()))


def build_episode_contract(sources: EpisodeSources, predicate_ids: Sequence[str], *,
                           branch_id: str, split: str, anchor_tick: int,
                           history_ticks: Sequence[int],
                           bindings_by_predicate: Mapping[str, Mapping[str, TypedEntity]],
                           cutoff_tick: int,
                           contracts: Mapping[str, PredicateContract] | None = None,
                           class_catalog: EntityClassCatalog | None = None) -> EpisodeContract:
    """Compile the exact episode contract for the requested executable set.

    Every requested predicate must be executable (``non_executable_authority``
    is an error, not a silent skip). Formulas receive ``formula_fields``
    coordinates; direct producers with a complete receiver projection receive
    ``direct_fields`` coordinates bound through the typed producer owner
    relation established at the anchor.
    """
    contracts = load_unified_contracts() if contracts is None else contracts
    requested = tuple(dict.fromkeys(predicate_ids))
    non_executable = tuple(pid for pid in requested
                           if pid not in contracts or not contracts[pid].executable)
    if non_executable:
        raise ContractError(f"non_executable_authority: {sorted(non_executable)}")
    profile = governed_parameter_profile(sources.episode_id,
                                         [contracts[pid] for pid in requested],
                                         manifest=sources.manifest)
    fields: list[FieldCoordinate] = []
    queries: list[PredicateQuery] = []
    for pid in requested:
        contract = contracts[pid]
        raw = bindings_by_predicate[pid]
        if set(raw) != {role.key for role in contract.roles}:
            raise ContractError(f"{pid}: supplied role bindings differ from the contract roles")
        if class_catalog is not None:
            bindings = tuple(RoleBinding(role.key, class_catalog.resolve(raw[role.key]))
                             for role in contract.roles)
        else:
            bindings = tuple(RoleBinding(role.key, raw[role.key]) for role in contract.roles)
        scope = _scope_active(sources, contract, {item.role: item.entity for item in bindings}, anchor_tick, cutoff_tick)
        applicable = True if scope is None else bool(scope)
        producer_binding: ProducerOwnerBinding | None = None
        if contract.semantic_mode == "direct_observation":
            authority = contract.grounding_spec["candidate_authority"]["source"]
            if authority not in DIRECT_PREDICATE_PRODUCERS:
                raise ContractError(f"{pid}: direct authority has no declared producer")
            from .state_dependencies import direct_predicate_bindings as _dpb
            binding = _dpb(authority)[pid]
            header = _producer_header(sources, contract, raw, anchor_tick)
            if header is None or anchor_tick > cutoff_tick or (header.availability_tick is not None and header.availability_tick > cutoff_tick):
                raise SourceAuthorityError(f"{pid}: no available prefix producer owner relation")
            row = sources.payload(header)
            coordinates = [RoleBinding("actor", _producer_actor(sources, row, anchor_tick, class_catalog))]
            if binding.row_builder == "_build_communication_rows":
                coordinates.append(RoleBinding("station", TypedEntity(row["station_id"], "world:CommunicationStation")))
            producer_binding = ProducerOwnerBinding(binding.row_builder, tuple(coordinates),
                                                    bindings, header.source_ref, anchor_tick)
            query = PredicateQuery(contract, bindings, profile, applicable=applicable,
                                   producer_owner_binding=producer_binding)
            nodes, receiver_complete = _receiver_operand_nodes(contract)
            if receiver_complete:
                fields.extend(direct_fields(query))
            else:
                owners = {item.role: item.entity for item in coordinates}
                for node in nodes:
                    cell = node.source_cell
                    fields.append(FieldCoordinate(
                        FieldSchema(f"source:{cell.artifact}:{cell.field_path}", node.data_type,
                                    node.unit or "nonmeasurement", node.node_id, tuple(node.domain.values)),
                        tuple(owners[role] for role in node.owner_keys), cell.artifact,
                        cell.field_path, tuple(node.owner_keys), cell.artifact))
        else:
            query = PredicateQuery(contract, bindings, profile, applicable=applicable)
            fields.extend(formula_fields(query))
        queries.append(query)
    fields = list({item.key: item for item in fields}.values())
    return EpisodeContract(sources.episode_id, branch_id, split,
                           tuple(history_ticks), anchor_tick, tuple(fields), tuple(queries))


def _formula_bindings(episode):
    result = {}
    for query in episode.queries:
        if query.contract.semantic_mode == "declared_formula":
            bindings = {item.role: item.entity for item in query.bindings}
            for coordinate, schema in zip(formula_fields(query), query.contract.required_fields):
                result[coordinate.key] = (query.contract, schema, bindings)
    return result


def observed_history(sources: EpisodeSources, episode: EpisodeContract, *,
                     cutoff_tick: int, parameter_profile: ParameterProfile,
                     class_catalog: EntityClassCatalog | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """History [H,F] values and known mask from the declared operand adapters.

    The last history position is the anchor; UNKNOWN statuses become a False
    known flag and are never numeric evidence.
    """
    from .ast_compiler import set_field
    formula_bindings = _formula_bindings(episode)
    h, f = len(episode.history_ticks), len(episode.fields)
    values = np.zeros((h, f), dtype=object)
    known = np.zeros((h, f), dtype=bool)
    for position, tick in enumerate(episode.history_ticks):
        for column, coordinate in enumerate(episode.fields):
            cell = _operand_cell(sources, coordinate, tick, cutoff_tick=cutoff_tick,
                                 parameter_profile=parameter_profile,
                                 formula_binding=formula_bindings.get(coordinate.key))
            if cell.known:
                values[position, column] = cell.value
                known[position, column] = True
    return values, known


def _operand_cell(sources: EpisodeSources, coordinate: FieldCoordinate, tick: int, *,
                  cutoff_tick: int, parameter_profile: ParameterProfile, formula_binding=None) -> OperandCell:
    if formula_binding is not None:
        contract, schema, bindings = formula_binding
        return scope_record_cell(sources, contract, schema, bindings, tick, cutoff_tick=cutoff_tick)
    artifact, path = coordinate.producer_artifact, coordinate.producer_row_path
    owners = tuple(entity.entity_id for entity in coordinate.owner)
    if artifact in RECEIVER_ROWS:
        return receiver_operand_cell(sources, artifact=artifact, field_path=path,
                                     owners=owners, tick=tick, cutoff_tick=cutoff_tick,
                                     data_type=coordinate.schema.data_type,
                                     parameter_profile=parameter_profile)
    if artifact == DIRECT_ARTIFACT:
        return OperandCell(artifact, path, owners, tick, "missing_record", None,
                           "direct producer operand columns enter through producer_operand_row")
    return scope_record_cell_from_coordinate(sources, coordinate, tick,
                                             cutoff_tick=cutoff_tick)


def scope_record_cell_from_coordinate(sources: EpisodeSources, coordinate: FieldCoordinate,
                                      tick: int, *, cutoff_tick: int) -> OperandCell:
    """Formula reads require the complete declared predicate role binding."""
    raise SourceAuthorityError(
        f"{coordinate.schema.field}: formula coordinate alone does not establish "
        "the full candidate role tuple; use observed_history or scope_record_cell")


# --------------------------------------------------------------------------
# Candidate assembly and the 72-predicate admission report
# --------------------------------------------------------------------------

def producer_row_index(sources: EpisodeSources, episode: EpisodeContract, query_index: int,
                       candidate_count: int, *, cutoff_tick: int,
                       parameter_profile: ParameterProfile
                       ) -> Mapping[ProducerRowKey, Mapping[str, Any]]:
    """Typed source index of predicted-future producer operands, ticks 5..50.

    Rows are looked up by exact (candidate, episode, branch, tick, owner)
    keys and carry only declared receiver-published fields; the four exact
    unpublished compute operands stay absent, so the producer's own UNKNOWN
    semantics apply. The anchor row is never supplied here.
    """
    query = episode.queries[query_index]
    rows: dict[ProducerRowKey, Mapping[str, Any]] = {}
    for tick in episode.future_ticks[1:]:
        if query.contract.semantic_mode != "direct_observation":
            raise SourceAuthorityError("formula operands enter as FieldCoordinates, not producer rows")
        produced = producer_operand_row(sources, query.contract,
            {item.role: item.entity for item in query.bindings}, tick,
            cutoff_tick=cutoff_tick, parameter_profile=parameter_profile)
        if produced.row is not None:
            payload = _immutable(produced.row)
            for candidate in range(candidate_count):
                rows[ProducerRowKey(candidate, episode.episode_id, episode.branch_id, tick,
                                    query.producer_owner)] = payload
    return _immutable(rows)


def build_candidate_trajectory(sources: EpisodeSources, episode: EpisodeContract, *,
                               cutoff_tick: int, parameter_profile: ParameterProfile,
                               candidate_count: int,
                               evaluator: FuturePredicateEvaluator | None = None
                               ) -> CandidateTrajectory:
    """Assemble one record: observed prefix plus joined predicted operand rows.

    Producer trajectories join the immutable source index by exact typed keys;
    missing keys yield missing operands. No raw data is written anywhere.
    """
    history, history_known = observed_history(sources, episode, cutoff_tick=cutoff_tick,
                                              parameter_profile=parameter_profile)
    formula_bindings = _formula_bindings(episode)
    f = len(episode.fields)
    states = np.zeros((candidate_count, 11, f), dtype=object)
    states_known = np.zeros((candidate_count, 11, f), dtype=bool)
    for column, coordinate in enumerate(episode.fields):
        for time, tick in enumerate(episode.future_ticks):
            cell = _operand_cell(sources, coordinate, tick, cutoff_tick=cutoff_tick,
                                 parameter_profile=parameter_profile,
                                 formula_binding=formula_bindings.get(coordinate.key))
            if cell.known:
                for candidate in range(candidate_count):
                    states[candidate, time, column] = cell.value
                    states_known[candidate, time, column] = True
    producer_rows = []
    for query_index, query in enumerate(episode.queries):
        if query.contract.semantic_mode != "direct_observation":
            continue
        index = producer_row_index(sources, episode, query_index, candidate_count,
                                   cutoff_tick=cutoff_tick, parameter_profile=parameter_profile)
        producer_rows.append(producer_rows_from_index(episode, query_index, candidate_count, index))
    return CandidateTrajectory(episode, history, history_known, states, states_known,
                               tuple(producer_rows))


def admission_report(predicate_ids: Sequence[str] | None = None, *,
                     contracts: Mapping[str, PredicateContract] | None = None
                     ) -> list[dict[str, Any]]:
    """All 72 executable source-contract admission/adapter outcomes.

    Each row separates declared projected operands from
    declaration-unknown ones (absent projections, governed thresholds are
    known-by-profile). No file is read and no truth is evaluated here.
    """
    contracts = load_unified_contracts() if contracts is None else contracts
    specs = load_dependency_specs()
    requested = sorted(contracts) if predicate_ids is None else list(dict.fromkeys(predicate_ids))
    executable = [pid for pid in requested if contracts[pid].executable]
    if len(executable) != 72 and predicate_ids is None:
        raise ContractError(f"expected 72 executable contracts, found {len(executable)}")
    rows = []
    for pid in executable:
        contract, spec = contracts[pid], specs[pid]
        authority = contract.grounding_spec["candidate_authority"]
        binding = None
        producer_fields: list[dict[str, Any]] = []
        missing_operands: list[str] = []
        known_operands: list[str] = []
        if contract.semantic_mode == "direct_observation":
            from .state_dependencies import direct_monitor_fields
            direct = direct_predicate_bindings(authority["source"])[pid]
            binding = {"function": direct.function, "row_builder": direct.row_builder,
                       "row_index_keys": list(builder_row_index(
                           DIRECT_PREDICATE_PRODUCERS[authority["source"]].module,
                           direct.row_builder).keys),
                       "role_row_reads": {read.role: read.row_key for read in direct.role_row_reads}}
            nodes, receiver_flag = _receiver_operand_nodes(contract)
            for node in nodes:
                cell = node.source_cell
                producer_fields.append({"field": cell.field_path, "artifact": cell.artifact,
                                        "form": cell.form, "owner_roles": list(node.owner_keys),
                                        "data_type": node.data_type, "unit": node.unit})
                if cell.form in ("published_monitor", "publisher_requirement"):
                    known_operands.append(f"{cell.artifact}:{cell.field_path}")
                elif cell.form == "governed_row_parameter":
                    known_operands.append(f"profile:{cell.field_path}")
                else:
                    missing_operands.append(f"{cell.artifact}:{cell.field_path} ({cell.form})")
            if not receiver_flag:
                missing_operands.append(
                    "immediate_receiver_projection_incomplete: weather_meta.jsonl:dust; "
                    "the declared producer row remains executable")
            complete = receiver_flag and all(
                item["form"] in ("published_monitor", "governed_row_parameter",
                                 "publisher_requirement") for item in producer_fields)
        else:
            complete = spec.completeness == "complete"
            for schema in contract.required_fields:
                owner_roles = field_owner_roles(contract, schema.field)
                source = contract.grounding_spec["field_sources"][schema.field]
                kind = ("weather_row" if source.startswith("context:weather.")
                        else "frame_projection" if source.startswith("context:")
                        else "record_row" if source.startswith("record:")
                        else "unknown_source_kind")
                producer_fields.append({"field": schema.field, "source": source,
                                        "owner_roles": list(owner_roles), "adapter_kind": kind,
                                        "data_type": schema.data_type, "unit": schema.unit})
                (known_operands if kind != "unknown_source_kind" else missing_operands).append(
                    f"{authority['source']}:{schema.field}" if kind != "unknown_source_kind"
                    else f"{authority['source']}:{schema.field} (undeclared source adapter)")
        rows.append({
            "predicate_id": pid,
            "semantic_mode": contract.semantic_mode,
            "authority_kind": authority["kind"],
            "authority_source": authority["source"],
            "dependency_completeness": spec.completeness,
            "roles": [role.key for role in contract.roles],
            "threshold_parameters": list(contract.threshold_parameters),
            "producer_binding": binding,
            "operand_adapter": ("declared_producer_rows" if binding else "world_truth_context_or_record"),
            "adapter_fields": tuple(producer_fields),
            "receiver_complete": bool(complete),
            "admission": ("admitted" if (complete or contract.semantic_mode == "declared_formula")
                          else "admitted_with_declared_unknown_operands"),
            "declared_projected_operands": tuple(known_operands),
            "declaration_unknown_operands": tuple(missing_operands),
        })
    return rows


__all__ = [
    "EpisodeSources", "OperandCell", "ProducerOperandRow",
    "RECORD_STATUSES", "RowHeader", "SourceAuthorityError", "admission_report",
    "build_candidate_trajectory", "build_episode_contract", "classify_record",
    "direct_truth_cell", "governed_parameter_profile", "observed_history",
    "producer_operand_row", "producer_row_index", "receiver_operand_cell",
    "scope_record_cell",
]
