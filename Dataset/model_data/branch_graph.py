"""Read-only parent/branch evidence composition for heterogeneous graph projection.

Branch objective artifacts certify a target predicate, not a complete world
snapshot. Evidence coverage is therefore separate from the five truth codes.
An unobserved cell is neither false nor an observed inactive predicate. This
boundary must not silently export partial branch evidence as an EpisodeGraphSequence.
"""

from __future__ import annotations

from Dataset.semantic_truth.provenance import open_artifact_binary
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import orjson

from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates
from Dataset.tools.branch_trajectory_view import compose_trajectory
from Dataset.tools.structured_branch_source import (
    PROJECT_ROOT,
    BranchSource,
    project_path,
    read_object,
    resolve_branch_source,
)

from .contract import (
    FORMAL_TICKS,
    FORMAL_TICK_COUNT,
    ModelDataError,
    PREDICATE_TRUTH_CODES,
    TICK_TO_INDEX,
    stable_identifier,
    stable_int63,
    validate_state_value,
)
from .dataset import EpisodeGraphSequence, UnifiedGraphFrame


BRANCH_ACTION_FEATURE_CONTRACT = "command_only_available_at_tick_v1"


class BranchActionFeatureProvider(Protocol):
    feature_contract: str

    def __call__(
        self,
        branch_key: str,
        history_ticks: Sequence[int],
        target_ticks: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]: ...


class BranchGraphError(ModelDataError):
    """Branch evidence cannot satisfy a requested graph contract."""


@dataclass(frozen=True)
class ParentTrajectory:
    path: Path
    samples: Mapping[str, Mapping[int, Mapping[str, Any]]]

    @classmethod
    def load(cls, path: Path) -> "ParentTrajectory":
        # Graphs use the formal capture grid. Validate every raw row before
        # filtering so duplicate/noninteger samples cannot be hidden by it.
        return cls(path.resolve(), _trajectory_index(_objects(path), formal_only=True))


@dataclass(frozen=True)
class BranchPredicateProjection:
    ticks: np.ndarray
    truth: np.ndarray
    evidence_present: np.ndarray
    unmapped_rows: tuple[Mapping[str, Any], ...]

    @property
    def known_truth(self) -> np.ndarray:
        return self.evidence_present & ((self.truth == 0) | (self.truth == 1))

    def require_complete(self) -> None:
        if self.unmapped_rows or not self.evidence_present.all():
            raise BranchGraphError(
                "target-window evidence is not a complete branch world graph; "
                "recompute branch-local L0/world truth before full-graph supervision"
            )


@dataclass(frozen=True)
class BranchGraphEvidence:
    source: BranchSource
    predicate_id: str
    trajectory: Mapping[str, Mapping[int, Mapping[str, Any]]]
    local_trajectory: Mapping[str, Mapping[int, Mapping[str, Any]]]
    lifecycle: tuple[Mapping[str, Any], ...]
    predicate_rows: tuple[Mapping[str, Any], ...]

    @property
    def branch_key(self) -> str:
        return str(self.source.manifest["branch_key"])

    def trajectory_at(self, tick: int) -> dict[str, Mapping[str, Any]]:
        self._check_tick(tick)
        return {
            entity: samples[tick]
            for entity, samples in self.trajectory.items()
            if tick in samples
        }

    def parent_frame(self, parent: EpisodeGraphSequence, tick: int) -> UnifiedGraphFrame:
        """Expose exact pre-fork graphs; never substitute a post-fork parent."""
        self._check_parent(parent)
        self._check_tick(tick)
        if tick >= self.source.fork:
            raise BranchGraphError("parent graph is not branch truth at/after the fork")
        return parent.frame(FORMAL_TICKS.index(tick))

    def project_predicates(self, parent: EpisodeGraphSequence) -> BranchPredicateProjection:
        """Align exact identities only, retaining every unmatched local record.

        The output roster is the parent's fixed graph roster, not a union with
        entities selected from future branch outcomes. A differently bound or
        aggregated local predicate is retained in unmapped_rows for explicit
        rematerialization, never matched to the first predicate of that type.
        """
        self._check_parent(parent)
        ticks = np.asarray([tick for tick in FORMAL_TICKS if tick <= self.source.end])
        shape = (len(ticks), len(parent.predicate_slots))
        truth = np.full(shape, PREDICATE_TRUTH_CODES["unknown"], dtype=np.uint8)
        present = np.zeros(shape, dtype=np.bool_)
        prefix_count = int(np.sum(ticks < self.source.fork))
        truth[:prefix_count] = parent.predicate_truth[:prefix_count]
        present[:prefix_count] = True
        entity_ids = {int(row["entity_index"]): str(row["entity_id"]) for row in parent.entities}
        slots = {}
        for slot in parent.predicate_slots:
            roles, indices = slot["binding_roles"], slot["binding_entity_indices"]
            if len(roles) != len(indices):
                raise BranchGraphError("parent predicate binding arity mismatch")
            bindings = {role: entity_ids[int(index)] for role, index in zip(roles, indices)}
            key = (str(slot["predicate_id"]), str(slot["slot_id"]), _binding_key(bindings))
            if key in slots:
                raise BranchGraphError("duplicate parent predicate slot identity")
            slots[key] = int(slot["slot_index"])
        unmapped = []
        for row in self.predicate_rows:
            key = (str(row["predicate_id"]), str(row["tuple_id"]), _binding_key(row["bindings"]))
            column = slots.get(key)
            if column is None:
                unmapped.append(row)
                continue
            index = int(row["tick"]) // 5
            if present[index, column]:
                raise BranchGraphError("duplicate branch predicate graph assignment")
            truth[index, column] = PREDICATE_TRUTH_CODES[str(row["value"])]
            present[index, column] = True
        return BranchPredicateProjection(ticks, truth, present, tuple(unmapped))

    def _check_tick(self, tick: int) -> None:
        if type(tick) is not int or tick not in FORMAL_TICKS or tick > self.source.end:
            raise BranchGraphError(f"tick lies outside measured formal branch frames: {tick}")

    def _check_parent(self, parent: EpisodeGraphSequence) -> None:
        if parent.episode_id != self.source.manifest["episode_id"]:
            raise BranchGraphError("branch and parent graph episode identities differ")
        if parent.metadata["profile_id"] != self.source.manifest["profile_id"]:
            raise BranchGraphError("branch and parent graph profile identities differ")


def load_branch_graph_evidence(
    group: Mapping[str, Any],
    arm: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    parent: ParentTrajectory | None = None,
) -> BranchGraphEvidence:
    source = resolve_branch_source(group, arm, project_root=project_root)
    if "source_manifest" in arm and project_path(str(arm["source_manifest"]), project_root) != source.manifest_path:
        raise BranchGraphError("catalog source manifest differs from resolved branch root")
    identity = {
        key: source.manifest[key]
        for key in ("episode_id", "profile_id", "seed_id", "group_id", "arm_id", "branch_key")
    }
    expected_key = "/".join(identity[key] for key in (
        "profile_id", "episode_id", "seed_id", "group_id", "arm_id",
    ))
    if identity["branch_key"] != expected_key:
        raise BranchGraphError("branch key does not encode its declared identity")
    target = group.get("target")
    if not isinstance(target, Mapping) or not isinstance(target.get("predicate_id"), str):
        raise BranchGraphError("group lacks target predicate authority")
    predicate_id = target["predicate_id"]
    manifest = read_object(source.objective_file("manifest.json"))
    semantics = read_object(source.objective_file("window_semantics.json"))
    closure = read_object(source.objective_file("window_closure.json"))
    reference = read_object(source.objective_file("parent_reference.json"))
    for key, expected in identity.items():
        if manifest.get(key) != expected or reference.get(key) != expected:
            raise BranchGraphError(f"objective branch identity mismatch: {key}")
    window = {"fork_tick": source.fork, "window_end_tick": source.end}
    if manifest.get("window") != window or reference.get("window") != window:
        raise BranchGraphError("objective/parent-reference window mismatch")
    if any(semantics.get(key) != value for key, value in window.items()):
        raise BranchGraphError("branch semantic window mismatch")
    if manifest.get("predicate_id") != predicate_id or semantics.get("predicate_id") != predicate_id:
        raise BranchGraphError("branch target predicate differs from catalog")
    if closure.get("status") != "PASS" or semantics.get("missing_source_record"):
        raise BranchGraphError("branch semantic closure is not complete")
    if reference.get("capture_episode_root") != source.manifest["parent_episode_ref"]:
        raise BranchGraphError("objective capture parent reference mismatch")
    if reference.get("objective_root") != f"aw_data/objective_semantic_truth/{identity['episode_id']}":
        raise BranchGraphError("objective world parent reference mismatch")

    parent_path = source.parent / "trajectories.jsonl"
    if parent is None:
        parent = ParentTrajectory.load(parent_path)
    if parent.path != parent_path.resolve():
        raise BranchGraphError("cached parent trajectory belongs to another episode")
    delta = _trajectory_index(_objects(source.raw_file("trajectories.jsonl")), formal_only=False)
    lifecycle = tuple(_objects(source.raw_file("entity_lifecycle_delta.jsonl")))
    for row in lifecycle:
        if not isinstance(row.get("entity_id"), str) or not row["entity_id"] or type(row.get("tick")) is not int:
            raise BranchGraphError("lifecycle identity/tick is malformed")
    # Lifecycle deltas describe a measured presence difference at each exact
    # tick, not an indefinitely persistent tombstone. Use the shared authority.
    composed = compose_trajectory(parent.samples, delta, list(lifecycle), source.manifest)
    trajectory = {
        entity: {
            tick: row
            for tick, row in samples.items()
            if tick in FORMAL_TICKS and tick <= source.end
        }
        for entity, samples in composed.items()
    }
    trajectory = {entity: samples for entity, samples in trajectory.items() if samples}
    trajectory = _apply_lifecycle_presence(trajectory, lifecycle)
    local_trajectory = {
        entity: {
            tick: row
            for tick, row in samples.items()
            if tick in FORMAL_TICKS and tick <= source.end
        }
        for entity, samples in delta.items()
        if samples
    }
    predicate_rows = tuple(_objects(source.objective_file("predicate_truth.jsonl")))
    world_rows = tuple(_objects(source.objective_file("world_truth_graph_deltas.jsonl")))
    _validate_predicate_evidence(predicate_rows, world_rows, source, predicate_id)
    return BranchGraphEvidence(
        source, predicate_id, trajectory, local_trajectory, lifecycle, predicate_rows
    )


def _apply_lifecycle_presence(
    trajectory: Mapping[str, Mapping[int, Mapping[str, Any]]],
    lifecycle: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    """Carry measured presence changes forward on the formal capture grid."""

    by_entity: dict[str, list[tuple[int, bool]]] = {}
    for row in lifecycle:
        by_entity.setdefault(str(row["entity_id"]), []).append(
            (int(row["tick"]), bool(row["present"]))
        )
    result: dict[str, dict[int, Mapping[str, Any]]] = {}
    for entity, samples in trajectory.items():
        changes = sorted(by_entity.get(entity, ()))
        filtered: dict[int, Mapping[str, Any]] = {}
        for tick, row in samples.items():
            state = None
            for change_tick, present in changes:
                if change_tick <= tick:
                    state = present
                else:
                    break
            if state is not False:
                filtered[tick] = row
        if filtered:
            result[entity] = filtered
    return result


def _validate_predicate_evidence(
    truth_rows: tuple[Mapping[str, Any], ...],
    world_rows: tuple[Mapping[str, Any], ...],
    source: BranchSource,
    predicate_id: str,
) -> None:
    truth_by_id = {}
    ticks = set()
    assignments = set()
    for row in truth_rows:
        tick, truth_id = row.get("tick"), row.get("truth_id")
        if type(tick) is not int or tick not in FORMAL_TICKS or not source.fork <= tick <= source.end:
            raise BranchGraphError("predicate evidence tick is outside the formal branch window")
        if row.get("episode_id") != source.manifest["episode_id"] or row.get("predicate_id") != predicate_id:
            raise BranchGraphError("predicate evidence identity mismatch")
        if not isinstance(truth_id, str) or not truth_id or truth_id in truth_by_id:
            raise BranchGraphError("missing/duplicate predicate truth id")
        if row.get("value") not in PREDICATE_TRUTH_CODES:
            raise BranchGraphError("invalid predicate truth value")
        if not isinstance(row.get("tuple_id"), str) or not row["tuple_id"]:
            raise BranchGraphError("predicate evidence tuple identity is missing")
        binding = _binding_key(row.get("bindings"))
        assignment = (tick, row["tuple_id"], binding)
        if assignment in assignments:
            raise BranchGraphError("duplicate predicate evidence assignment")
        if not isinstance(row.get("source_refs"), list) or not row["source_refs"]:
            raise BranchGraphError("predicate truth has no source references")
        assignments.add(assignment)
        ticks.add(tick)
        truth_by_id[truth_id] = row
    expected_ticks = {tick for tick in FORMAL_TICKS if source.fork <= tick <= source.end}
    if ticks != expected_ticks or not expected_ticks:
        raise BranchGraphError("predicate evidence has incomplete formal tick coverage")
    seen = set()
    for row in world_rows:
        if row.get("operation") != "upsert_target_predicate_truth":
            raise BranchGraphError("unsupported branch world graph operation")
        if row.get("branch_key") != source.manifest["branch_key"]:
            raise BranchGraphError("world delta belongs to another branch")
        truth_id = row.get("source_truth_id")
        truth = truth_by_id.get(truth_id)
        if truth is None or truth_id in seen:
            raise BranchGraphError("world delta references absent/duplicate local truth")
        for key in ("episode_id", "predicate_id", "tuple_id", "tick", "bindings", "source_refs", "missing_source_record"):
            if row.get(key) != truth.get(key):
                raise BranchGraphError(f"world delta disagrees with local truth: {key}")
        if row.get("truth_value") != truth["value"]:
            raise BranchGraphError("world delta disagrees with local truth value")
        seen.add(truth_id)
    if seen != set(truth_by_id):
        raise BranchGraphError("world delta is missing local predicate evidence")


def _binding_key(value: Any) -> bytes:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BranchGraphError("predicate bindings must be a string-keyed object")
    # Some local targets are aggregates (e.g. vehicle_ids). Preserve their exact
    # identity instead of coercing lists into entity ids or guessed binary roles.
    return orjson.dumps(dict(value), option=orjson.OPT_SORT_KEYS)


def _trajectory_index(
    rows: Iterable[Mapping[str, Any]], *, formal_only: bool,
) -> dict[str, dict[int, dict[str, Any]]]:
    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    seen: dict[str, set[int]] = {}
    for row in rows:
        entity, tick = row.get("entity_id"), row.get("tick")
        if not isinstance(entity, str) or not entity or type(tick) is not int or not 0 <= tick <= 900:
            raise BranchGraphError("trajectory has malformed entity/tick")
        if entity not in seen:
            seen[entity] = set()
        if tick in seen[entity]:
            raise BranchGraphError(f"duplicate trajectory sample: {entity}@{tick}")
        seen[entity].add(tick)
        if formal_only and tick not in FORMAL_TICKS:
            continue
        for field in ("pos_enu", "vel_mps"):
            value = row.get(field)
            if value is not None and (
                not isinstance(value, list) or len(value) != 3
                or any(type(item) not in (int, float) or not np.isfinite(item) for item in value)
            ):
                raise BranchGraphError(f"malformed measured trajectory field: {field}")
        if entity not in indexed:
            indexed[entity] = {}
        indexed[entity][tick] = dict(row)
    return indexed


def _objects(path: Path) -> Iterable[dict[str, Any]]:
    # orjson keeps full-cohort source composition bounded by disk I/O rather
    # than repeatedly decoding large parent episodes with the Python decoder.
    with open_artifact_binary(path) as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = orjson.loads(line)
            if not isinstance(value, dict):
                raise BranchGraphError(f"expected JSON object: {path}:{number}")
            yield value


class BranchGraphEvidenceStore:
    """Bounded caches keyed by exact branch and parent artifact identities."""

    def __init__(self, catalog_path: Path, *, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = project_root.resolve()
        catalog = read_object(catalog_path)
        groups = catalog.get("groups")
        if not isinstance(groups, list) or not groups:
            raise BranchGraphError("catalog lacks a nonempty group list")
        self.records = {}
        self._sources = {}
        roots = set()
        for group in groups:
            arms = group.get("arms")
            if not isinstance(arms, list) or not arms:
                raise BranchGraphError("catalog group lacks Arms")
            for arm in arms:
                if arm.get("structured_training_eligible") is not True:
                    raise BranchGraphError("fixed cohort contains an ineligible Arm; refusing to filter")
                source = resolve_branch_source(group, arm, project_root=self.project_root)
                key = str(arm["branch_key"])
                if key in self.records or source.root in roots:
                    raise BranchGraphError("catalog repeats a branch identity/root")
                self.records[key] = (group, arm)
                self._sources[key] = source
                roots.add(source.root)
        counts = catalog.get("counts")
        if not isinstance(counts, Mapping) or counts.get("arms") != len(self.records) or counts.get("groups") != len(groups):
            raise BranchGraphError("catalog cohort counts disagree with records")
        self._parents: OrderedDict[Path, ParentTrajectory] = OrderedDict()
        self._last: tuple[str, BranchGraphEvidence] | None = None

    def get_source(self, branch_key: str) -> BranchSource:
        try:
            return self._sources[branch_key]
        except KeyError as exc:
            raise KeyError(branch_key) from exc

    def get_branch(self, branch_key: str) -> BranchGraphEvidence:
        if self._last is not None and self._last[0] == branch_key:
            return self._last[1]
        group, arm = self.records[branch_key]
        path = self._sources[branch_key].parent / "trajectories.jsonl"
        parent = self._parents.pop(path, None)
        if parent is None:
            parent = ParentTrajectory.load(path)
        self._parents[path] = parent
        if len(self._parents) > 1:
            self._parents.popitem(last=False)
        evidence = load_branch_graph_evidence(group, arm, project_root=self.project_root, parent=parent)
        self._last = (branch_key, evidence)
        return evidence


class BranchGraphSequenceStore:
    """In-memory store adapter for existing tensor-contract builders.

    The parent store supplies the already-authorized manifest vocabulary and
    statistics metadata. Branch bytes remain external and split-unassigned;
    this object does not synthesize an episode index or alter split manifests.
    """

    def __init__(self, parent_store: Any, sequences: Iterable[BranchGraphSequence]) -> None:
        self.manifest = parent_store.manifest
        self._sequences = {}
        self._branches = {}
        for branch in sequences:
            if branch.episode_id in self._sequences:
                raise BranchGraphError(f"duplicate branch graph sequence: {branch.episode_id}")
            self._sequences[branch.episode_id] = branch.sequence
            self._branches[branch.episode_id] = branch
        self.episodes = {
            episode_id: {
                "episode_id": episode_id,
                "profile_id": sequence.metadata["profile_id"],
                "split": "unassigned",
            }
            for episode_id, sequence in self._sequences.items()
        }

    def get_episode(self, episode_id: str) -> EpisodeGraphSequence:
        try:
            return self._sequences[str(episode_id)]
        except KeyError as exc:
            raise KeyError(episode_id) from exc

    def get_window(self, episode_id: str, start_frame_index: int, history_length: int, horizon: int):
        return self._branches[episode_id].window(start_frame_index, history_length, horizon)


@dataclass(frozen=True)
class BranchGraphSequence:
    """A branch graph view with an explicit partial-truth boundary."""

    sequence: EpisodeGraphSequence
    branch_key: str
    target_predicate_id: str
    target_predicate_slot_index: int | None
    target_evidence_mask: np.ndarray
    local_entity_ids: tuple[str, ...]
    source_manifest_path: str

    @property
    def episode_id(self) -> str:
        return self.sequence.episode_id

    def window(
        self,
        start_frame_index: int,
        history_length: int,
        horizon: int,
        *,
        action_provider: BranchActionFeatureProvider | None = None,
    ):
        if self.target_predicate_slot_index is None:
            raise BranchGraphError(
                "unresolved target binding cannot enter a graph training window"
            )
        window = self.sequence.window(start_frame_index, history_length, horizon)
        start, end = int(window.target_ticks[0]), int(window.target_ticks[-1])
        if start < self.source_fork_tick or end > self.source_end_tick:
            raise BranchGraphError(
                f"window is outside local branch evidence: {self.branch_key} "
                f"[{start}, {end}]"
            )
        if action_provider is None:
            return window
        if getattr(action_provider, "feature_contract", None) != BRANCH_ACTION_FEATURE_CONTRACT:
            raise BranchGraphError("branch action provider lacks the command-only contract")
        try:
            history_actions, target_actions = action_provider(
                self.branch_key, window.history_ticks, window.target_ticks
            )
        except Exception as exc:
            raise BranchGraphError("branch action provider failed") from exc
        history_actions = np.asarray(history_actions, dtype=np.float32)
        target_actions = np.asarray(target_actions, dtype=np.float32)
        if history_actions.ndim != 2 or target_actions.ndim != 2:
            raise BranchGraphError("branch action features must be rank-two arrays")
        if history_actions.shape[0] != len(window.history_ticks) or target_actions.shape[0] != len(window.target_ticks):
            raise BranchGraphError("branch action feature rows do not match window ticks")
        if history_actions.shape[1] != target_actions.shape[1] or not np.isfinite(history_actions).all() or not np.isfinite(target_actions).all():
            raise BranchGraphError("branch action features have invalid dimensions or values")
        return replace(
            window,
            history_action_features=history_actions,
            target_action_features=target_actions,
        )

    @property
    def source_fork_tick(self) -> int:
        return int(self.sequence.metadata["fork_tick"])

    @property
    def source_end_tick(self) -> int:
        return int(self.sequence.metadata["window_end_tick"])


def compose_branch_graph_sequence(
    parent: EpisodeGraphSequence,
    evidence: BranchGraphEvidence,
) -> BranchGraphSequence:
    """Compose local measured state and target predicate evidence.

    This adapter deliberately masks every non-target predicate after the fork.
    It is therefore safe for inspection and target-window training only; callers
    must use the returned masks and may not label it as a complete branch graph.
    """
    evidence._check_parent(parent)
    source = evidence.source
    branch_key = evidence.branch_key
    branch_id = stable_identifier("branch_episode", branch_key)
    entity_rows = [dict(row) for row in parent.entities]
    entity_index = {str(row["entity_id"]): int(row["entity_index"]) for row in entity_rows}
    local_ids = set(evidence.local_trajectory) | {
        str(row["entity_id"]) for row in evidence.lifecycle
    }
    binding_status = branch_target_binding_status(evidence.predicate_rows)
    for row in evidence.predicate_rows:
        bindings = row["bindings"]
        if any(not isinstance(value, str) or not value for value in bindings.values()):
            continue
        local_ids.update(bindings.values())
    for entity_id in sorted(local_ids):
        if entity_id not in entity_index:
            entity_index[entity_id] = len(entity_rows)
            row = _new_branch_entity(entity_id, evidence)
            row["entity_index"] = entity_index[entity_id]
            entity_rows.append(row)
    arrays = {
        name: getattr(parent, name).copy()
        for name in (
            "continuous", "continuous_scope", "continuous_valid",
            "boolean", "boolean_scope", "boolean_valid",
            "categorical", "categorical_scope", "categorical_valid",
            "reference", "reference_scope", "reference_valid",
        )
    }
    state_slots = [dict(row) for row in parent.state_slots]
    _mask_branch_state(arrays, state_slots, local_ids, source.fork, source.end)
    state_slots, arrays = _overlay_branch_state(
        state_slots, arrays, evidence.local_trajectory, entity_index, branch_id
    )
    for index, row in enumerate(state_slots):
        row["slot_index"] = index
        row["episode_id"] = branch_id
        row["subject_entity_index"] = entity_index[str(row["subject_id"])]
    predicate_truth = parent.predicate_truth.copy()
    fork_index = _formal_index_at_or_after(source.fork)
    predicate_truth[fork_index:] = PREDICATE_TRUTH_CODES["unknown"]
    predicate_slots = [dict(row) for row in parent.predicate_slots]
    if binding_status != "mapped_exact":
        target_index = None
    else:
        target_index, predicate_slots, predicate_truth = _append_branch_predicate(
            predicate_slots, predicate_truth, evidence.predicate_rows, entity_index, parent, source, branch_id
        )
    for index, row in enumerate(predicate_slots):
        row["slot_index"] = index
        row["episode_id"] = branch_id
    for row in entity_rows:
        row["episode_id"] = branch_id
    metadata = dict(parent.metadata)
    metadata.update({
        "episode_id": branch_id, "split": "unassigned",
        "parent_episode_id": parent.episode_id, "branch_key": branch_key,
        "fork_tick": source.fork, "window_end_tick": source.end,
        "branch_graph_completeness": "target_window_only",
        "target_binding_status": binding_status,
        "entity_count": len(entity_rows), "state_slot_count": len(state_slots),
        "predicate_slot_count": len(predicate_slots),
    })
    sequence = EpisodeGraphSequence(
        metadata=metadata, entities=tuple(entity_rows), state_slots=tuple(state_slots),
        predicate_slots=tuple(predicate_slots), predicate_truth=predicate_truth,
        **arrays,
    )
    mask = np.zeros(FORMAL_TICK_COUNT, dtype=np.bool_)
    for row in evidence.predicate_rows:
        mask[TICK_TO_INDEX[int(row["tick"])]] = True
    return BranchGraphSequence(
        sequence=sequence, branch_key=branch_key,
        target_predicate_id=evidence.predicate_id,
        target_predicate_slot_index=target_index,
        target_evidence_mask=mask,
        local_entity_ids=tuple(sorted(local_ids)),
        source_manifest_path=str(source.manifest_path),
    )


def _new_branch_entity(entity_id: str, evidence: BranchGraphEvidence) -> dict[str, Any]:
    sample = next(iter(evidence.local_trajectory.get(entity_id, {}).values()), {})
    category = str(sample.get("entity_category") or sample.get("category") or "").strip().lower()
    if category == "uav":
        ontology, scope, observer = ["world:UnmannedAircraft"], ["uav"], True
    elif category == "pedestrian":
        ontology, scope, observer = ["world:Pedestrian"], ["pedestrian"], False
    elif category == "vehicle":
        ontology, scope, observer = ["world:GroundVehicle"], ["vehicle"], False
    else:
        ontology, scope, observer = [], [], False
    return {
        "entity_id": entity_id, "entity_index": -1,
        "entity_code": stable_int63("graph_entity", entity_id),
        "scope_types": scope, "scope_subtypes": [], "ontology_class_ids": ontology,
        "subject_categories": [category] if category else [], "subject_kinds": [],
        "source_kinds": ["branch_trajectory", "branch_lifecycle"],
        "observer_capable": observer,
    }


def _mask_branch_state(
    arrays: Mapping[str, np.ndarray], slots: Iterable[Mapping[str, Any]],
    local_ids: set[str], fork: int, end: int,
) -> None:
    first, last = _formal_index_at_or_after(fork), _formal_index_at_or_before(end)
    for slot in slots:
        if str(slot["subject_id"]) not in local_ids:
            continue
        kind, start, width = str(slot["value_kind"]), int(slot["storage_index"]), int(slot["width"])
        if kind not in {"continuous", "boolean", "categorical", "reference"}:
            continue
        arrays[f"{kind}_scope"][first:last + 1, start:start + width] = False
        arrays[f"{kind}_valid"][first:last + 1, start:start + width] = False
        arrays[kind][first:last + 1, start:start + width] = {
            "continuous": np.nan, "boolean": False, "categorical": 0, "reference": -1,
        }[kind]


def _overlay_branch_state(
    slots: list[dict[str, Any]], arrays: dict[str, np.ndarray],
    trajectory: Mapping[str, Mapping[int, Mapping[str, Any]]],
    entity_index: Mapping[str, int], branch_id: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    aliases = {"position_enu_m": ("pos_enu", "position_enu_m"),
               "velocity_enu_mps": ("vel_mps", "velocity_enu_mps"),
               "yaw_deg": ("yaw_deg",)}
    by_field: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for slot in slots:
        by_field.setdefault((str(slot["subject_id"]), str(slot["field_name"])), []).append(slot)
    # Do not create branch-only state slots: their statistic keys and priors
    # would have no training-vocabulary authority. Spawned entities remain in
    # the graph roster and can still participate in explicitly bound predicates.
    for entity_id, rows in trajectory.items():
        for tick, row in rows.items():
            if tick not in TICK_TO_INDEX:
                continue
            index = TICK_TO_INDEX[tick]
            for field, names in aliases.items():
                value = next((row[name] for name in names if name in row), None)
                if value is None:
                    continue
                try:
                    kind, width = validate_state_value(field, value)
                except ModelDataError:
                    continue
                for slot in by_field.get((entity_id, field), ()):
                    if str(slot["value_kind"]) != kind or int(slot["width"]) != width:
                        continue
                    start = int(slot["storage_index"])
                    arrays[kind][index, start:start + width] = np.asarray(value, dtype=float).reshape(width)
                    arrays[f"{kind}_scope"][index, start:start + width] = True
                    arrays[f"{kind}_valid"][index, start:start + width] = True
    return slots, arrays


def branch_target_binding_status(rows: Iterable[Mapping[str, Any]]) -> str:
    """Check the whole query series, not just its first binding.

    A nearest-pair query may select different instances over time. Its truth
    series cannot supervise the first pair as though the query were fixed.
    These statuses describe an adapter limitation, not invalid source truth.
    """
    identities = set()
    nonscalar = False
    empty = False
    for row in rows:
        bindings = row.get("bindings")
        encoded = _binding_key(bindings)
        predicate_id, tuple_id = row.get("predicate_id"), row.get("tuple_id")
        if not isinstance(predicate_id, str) or not predicate_id or not isinstance(tuple_id, str) or not tuple_id:
            raise BranchGraphError("target predicate/tuple identity is missing")
        identities.add((predicate_id, tuple_id, encoded))
        empty |= not bindings
        nonscalar |= any(not isinstance(value, str) or not value for value in bindings.values())
    if not identities:
        raise BranchGraphError("target evidence is empty")
    if nonscalar:
        return "unmapped_aggregate"
    if empty:
        return "unmapped_empty_binding"
    if len(identities) != 1:
        return "unmapped_changing_binding"
    return "mapped_exact"


def _append_branch_predicate(
    slots: list[dict[str, Any]], values: np.ndarray, rows: Sequence[Mapping[str, Any]],
    entity_index: Mapping[str, int], parent: EpisodeGraphSequence, source: BranchSource,
    branch_id: str,
) -> tuple[int, list[dict[str, Any]], np.ndarray]:
    first = rows[0]
    bindings = first["bindings"]
    roles = tuple(str(key) for key in bindings)
    ids = tuple(str(bindings[key]) for key in roles)
    parent_entity_ids = {
        int(row["entity_index"]): str(row["entity_id"])
        for row in parent.entities
    }
    exact_slot = None
    for candidate in parent.predicate_slots:
        if str(candidate["predicate_id"]) != str(first["predicate_id"]):
            continue
        candidate_bindings = {
            str(role): parent_entity_ids[int(index)]
            for role, index in zip(
                candidate.get("binding_roles", ()),
                candidate.get("binding_entity_indices", ()),
            )
        }
        if str(candidate["slot_id"]) == str(first["tuple_id"]) and _binding_key(candidate_bindings) == _binding_key(bindings):
            exact_slot = int(candidate["slot_index"])
            break
    if exact_slot is not None:
        for row in rows:
            values[TICK_TO_INDEX[int(row["tick"])] , exact_slot] = PREDICATE_TRUTH_CODES[str(row["value"])]
        return exact_slot, slots, values
    column = np.full(FORMAL_TICK_COUNT, PREDICATE_TRUTH_CODES["unknown"], dtype=np.uint8)
    for row in rows:
        column[TICK_TO_INDEX[int(row["tick"])]] = PREDICATE_TRUTH_CODES[str(row["value"])]
    classes = _predicate_role_classes(str(first["predicate_id"]), roles)
    slot = {
        "episode_id": branch_id, "slot_id": str(first["tuple_id"]), "slot_index": len(slots),
        "assertion_id": stable_identifier("branch_predicate", branch_id, first["tuple_id"]),
        "predicate_id": str(first["predicate_id"]), "binding_roles": list(roles),
        "binding_entity_indices": [entity_index[entity] for entity in ids],
        "binding_ontology_classes": classes,
    }
    return len(slots), slots + [slot], np.column_stack((values, column))


def _predicate_role_classes(predicate_id: str, roles: Sequence[str]) -> list[str]:
    templates = {
        str(template["id"]): template
        for template in get_core_predicate_templates()
    }
    template = templates.get(predicate_id)
    if template is None:
        raise BranchGraphError(f"target predicate lacks registry authority: {predicate_id}")
    role_classes = {
        str(row["key"]): f"world:{row['class']}"
        for row in template["argument_roles"]
    }
    if any(role not in role_classes for role in roles):
        raise BranchGraphError(
            f"target binding roles disagree with predicate registry: {predicate_id}"
        )
    return [role_classes[role] for role in roles]


def _formal_index_at_or_after(tick: int) -> int:
    for index, value in enumerate(FORMAL_TICKS):
        if value >= tick:
            return index
    raise BranchGraphError(f"no formal frame at/after tick {tick}")


def _formal_index_at_or_before(tick: int) -> int:
    for index in range(len(FORMAL_TICKS) - 1, -1, -1):
        if FORMAL_TICKS[index] <= tick:
            return index
    raise BranchGraphError(f"no formal frame at/before tick {tick}")
