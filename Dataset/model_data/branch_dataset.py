"""Exact fixed-cohort window index for partial branch graph evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
from pathlib import Path
from typing import Any, Mapping

import orjson

from .branch_graph import (
    BranchActionFeatureProvider,
    BranchGraphError,
    BranchGraphEvidenceStore,
    BranchGraphSequence,
    compose_branch_graph_sequence,
    branch_target_binding_status,
)
from .contract import FORMAL_TICKS, ModelDataError, TICK_TO_INDEX
from .dataset import GraphWindow
from Dataset.tools.structured_branch_source import PROJECT_ROOT, project_path


@dataclass(frozen=True)
class BranchWindowRecord:
    branch_key: str
    split: str
    start_frame_index: int
    origin_tick: int
    target_tick: int
    predicate_id: str
    binding_status: str
    source_row_number: int


@dataclass(frozen=True)
class BranchGraphWindowSample:
    branch: BranchGraphSequence
    window: GraphWindow
    record: BranchWindowRecord


class BranchGraphWindowDataset:
    """Index accepted structured windows without shrinking the ARM denominator.

    List-valued target bindings stay in ``index`` and are marked blocked. Access
    to such a row fails closed instead of silently removing it from training.
    """

    def __init__(
        self,
        catalog_path: Path,
        windows_path: Path,
        parent_store: Any,
        *,
        split: str,
        project_root: Path = PROJECT_ROOT,
        history_length: int = 8,
        horizon: int = 5,
        expected_window_count: int | None = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ModelDataError(f"invalid branch split: {split}")
        if history_length != 8 or horizon != 5:
            raise ModelDataError("accepted branch windows require history=8 and horizon=5")
        self.split = split
        self.history_length = history_length
        self.horizon = horizon
        self.project_root = project_root.resolve()
        self.parent_store = parent_store
        self.evidence_store = BranchGraphEvidenceStore(
            catalog_path.resolve(), project_root=self.project_root
        )
        by_identity: dict[tuple[str, str, str, str, str], tuple[str, Mapping[str, Any], Mapping[str, Any]]] = {}
        binding_status: dict[str, str] = {}
        for branch_key, (group, arm) in self.evidence_store.records.items():
            identity = (
                str(group["profile_id"]), str(group["episode_id"]),
                str(group["seed_id"]), str(group["group_id"]), str(arm["arm_id"]),
            )
            if identity in by_identity:
                raise BranchGraphError("catalog repeats a structured window identity")
            by_identity[identity] = (branch_key, group, arm)
            binding_status[branch_key] = _binding_status(
                self.evidence_store.get_source(branch_key).objective_file("predicate_truth.jsonl")
            )

        records: list[BranchWindowRecord] = []
        represented: set[str] = set()
        total = 0
        with gzip.open(windows_path.resolve(), "rb") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                total += 1
                value = orjson.loads(line)
                if not isinstance(value, dict):
                    raise BranchGraphError(f"structured window row is not an object: {number}")
                identity = (
                    str(value.get("profile_id")), str(value.get("parent_episode_id")),
                    str(value.get("seed_id")), str(value.get("group_id")),
                    str(value.get("variant")),
                )
                authority = by_identity.get(identity)
                if authority is None:
                    raise BranchGraphError(f"structured window has no catalog ARM: row {number}")
                branch_key, group, arm = authority
                _validate_window_row(
                    value, number, group, arm, branch_key, parent_store,
                    self.project_root, history_length, horizon,
                    self.evidence_store.get_source(branch_key).end,
                )
                represented.add(branch_key)
                if value["split"] == split:
                    origin = int(value["origin_tick"])
                    records.append(BranchWindowRecord(
                        branch_key=branch_key,
                        split=split,
                        start_frame_index=TICK_TO_INDEX[origin] - history_length + 1,
                        origin_tick=origin,
                        target_tick=int(value["target_tick"]),
                        predicate_id=str(value["predicate_id"]),
                        binding_status=binding_status[branch_key],
                        source_row_number=number,
                    ))
        if expected_window_count is not None and total != expected_window_count:
            raise BranchGraphError(
                f"structured window count mismatch: {total} != {expected_window_count}"
            )
        missing = set(self.evidence_store.records) - represented
        if missing:
            raise BranchGraphError(
                f"structured fixed cohort lacks windows for {len(missing)} Arms"
            )
        self.index = tuple(records)
        self.total_window_count = total
        self.blocked_branch_keys = frozenset(
            key for key, status in binding_status.items() if status != "mapped_exact"
        )
        self.blocked_window_count = sum(
            row.binding_status != "mapped_exact" for row in self.index
        )
        self.mappable_window_count = len(self.index) - self.blocked_window_count

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> GraphWindow:
        return self.load_sample(index).window

    def load_sample(
        self,
        index: int,
        *,
        action_provider: BranchActionFeatureProvider | None = None,
    ) -> BranchGraphWindowSample:
        record = self.index[index]
        if record.binding_status != "mapped_exact":
            raise BranchGraphError(
                f"branch window target is explicitly blocked: {record.branch_key}"
            )
        evidence = self.evidence_store.get_branch(record.branch_key)
        parent = self.parent_store.get_episode(evidence.source.manifest["episode_id"])
        branch = compose_branch_graph_sequence(parent, evidence)
        window = branch.window(
            record.start_frame_index,
            self.history_length,
            self.horizon,
            action_provider=action_provider,
        )
        if int(window.history_ticks[-1]) != record.origin_tick or int(window.target_ticks[-1]) != record.target_tick:
            raise BranchGraphError("composed graph window disagrees with structured index ticks")
        return BranchGraphWindowSample(
            branch=branch,
            window=replace(window, split=self.split),
            record=record,
        )


def _binding_status(path: Path) -> str:
    rows = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = orjson.loads(line)
            if not isinstance(row, Mapping):
                raise BranchGraphError(f"target row is malformed: {path}")
            rows.append(row)
    return branch_target_binding_status(rows)


def _validate_window_row(
    row: Mapping[str, Any],
    number: int,
    group: Mapping[str, Any],
    arm: Mapping[str, Any],
    branch_key: str,
    parent_store: Any,
    project_root: Path,
    history_length: int,
    horizon: int,
    branch_end_tick: int,
) -> None:
    origin, target = row.get("origin_tick"), row.get("target_tick")
    if type(origin) is not int or type(target) is not int or origin not in TICK_TO_INDEX or target not in TICK_TO_INDEX:
        raise BranchGraphError(f"structured window ticks are malformed: row {number}")
    if target - origin != 5 * horizon:
        raise BranchGraphError(f"structured window horizon is malformed: row {number}")
    if TICK_TO_INDEX[origin] < history_length - 1:
        raise BranchGraphError(f"structured window lacks formal history: row {number}")
    if row.get("history_frames") != history_length or row.get("future_frames") != horizon:
        raise BranchGraphError(f"structured window shape disagrees with graph contract: row {number}")
    expected_ticks = {
        "history_ticks": 5 * history_length,
        "future_horizon_ticks": 5 * horizon,
        "state_target_horizon_ticks": 5,
        "predicate_event_target_horizon_ticks": 5 * horizon,
    }
    if any(row.get(key) != value for key, value in expected_ticks.items()):
        raise BranchGraphError(f"structured window tick contract mismatch: row {number}")
    if row.get("action_known_at_origin") is not True:
        raise BranchGraphError(f"branch action is not available at origin: row {number}")
    if row.get("predicate_id") != group["target"]["predicate_id"]:
        raise BranchGraphError(f"structured target predicate mismatch: row {number}")
    if row.get("source_branch_root") != arm["branch_root"]:
        source = project_path(str(row.get("source_branch_root")), project_root)
        authority = project_path(str(arm["branch_root"]), project_root)
        if source != authority:
            raise BranchGraphError(f"structured source branch mismatch: row {number}")
    expected_key = "/".join((
        str(group["profile_id"]), str(group["episode_id"]), str(group["seed_id"]),
        str(group["group_id"]), str(arm["arm_id"]),
    ))
    if branch_key != expected_key:
        raise BranchGraphError(f"catalog branch key is noncanonical: row {number}")
    split = row.get("split")
    if split not in {"train", "validation", "test"}:
        raise BranchGraphError(f"structured split is invalid: row {number}")
    parent = parent_store.episodes.get(str(group["episode_id"]))
    if not isinstance(parent, Mapping) or parent.get("split") != split:
        raise BranchGraphError(f"structured split differs from parent graph: row {number}")
    if origin + 5 not in FORMAL_TICKS or target > branch_end_tick:
        raise BranchGraphError(f"structured window escapes branch evidence: row {number}")
