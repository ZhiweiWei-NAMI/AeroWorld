"""Random-access NumPy/PyTorch reader for the unified AERO graph dataset."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import orjson
import pyarrow.parquet as pq

from .contract import (
    ALLOWED_HISTORY_LENGTHS,
    ALLOWED_HORIZONS,
    DATASET_SCHEMA_NAME,
    DATASET_SCHEMA_VERSION,
    EDIT_CODES,
    FORMAL_TICK_COUNT,
    FORMAL_TICKS,
    INACTIVE_PREDICATE_CODE,
    ModelDataError,
    predicate_masks,
    unpack_bits,
)


@dataclass(frozen=True)
class ObserverBundle:
    """One real observer/sensor/modality record at one tick."""

    dataset_id: str
    episode_id: str
    tick: int
    observer_uav_id: str
    sensor_id: str
    modality: str
    feature_tensor: np.ndarray
    token_or_point_coordinates: np.ndarray
    sensor_intrinsics: np.ndarray
    sensor_extrinsics: np.ndarray
    presence_mask: bool
    encoder_name: str
    encoder_weight_hash: str
    preprocess_hash: str
    observation_to_entity_associations: tuple[str, ...]
    source_provenance: tuple[str, ...]


@dataclass(frozen=True)
class StateFrame:
    continuous: np.ndarray
    continuous_scope: np.ndarray
    continuous_valid: np.ndarray
    boolean: np.ndarray
    boolean_scope: np.ndarray
    boolean_valid: np.ndarray
    categorical: np.ndarray
    categorical_scope: np.ndarray
    categorical_valid: np.ndarray
    reference: np.ndarray
    reference_scope: np.ndarray
    reference_valid: np.ndarray


@dataclass(frozen=True)
class UnifiedGraphFrame:
    episode_id: str
    tick: int
    state: StateFrame
    predicate_truth: np.ndarray
    observer_bundles: tuple[ObserverBundle, ...]


@dataclass(frozen=True)
class GraphWindow:
    episode_id: str
    split: str
    history_ticks: np.ndarray
    target_ticks: np.ndarray
    history_state: Mapping[str, np.ndarray]
    target_state: Mapping[str, np.ndarray]
    history_predicate_truth: np.ndarray
    target_predicate_truth: np.ndarray
    target_state_edits: Mapping[str, np.ndarray]
    target_state_edit_supervision: Mapping[str, np.ndarray]
    target_predicate_edits: np.ndarray
    topology: Mapping[str, object]
    observer_bundles: tuple[tuple[ObserverBundle, ...], ...]
    history_action_features: np.ndarray | None = None
    target_action_features: np.ndarray | None = None


ObserverProvider = Callable[[str, int], tuple[ObserverBundle, ...]]
ActionFeatureProvider = Callable[
    [str, Sequence[int], Sequence[int]], tuple[np.ndarray, np.ndarray]
]


@dataclass(frozen=True)
class EpisodeGraphSequence:
    metadata: Mapping[str, Any]
    entities: tuple[Mapping[str, Any], ...]
    state_slots: tuple[Mapping[str, Any], ...]
    predicate_slots: tuple[Mapping[str, Any], ...]
    continuous: np.ndarray
    continuous_scope: np.ndarray
    continuous_valid: np.ndarray
    boolean: np.ndarray
    boolean_scope: np.ndarray
    boolean_valid: np.ndarray
    categorical: np.ndarray
    categorical_scope: np.ndarray
    categorical_valid: np.ndarray
    reference: np.ndarray
    reference_scope: np.ndarray
    reference_valid: np.ndarray
    predicate_truth: np.ndarray

    @property
    def episode_id(self) -> str:
        return str(self.metadata["episode_id"])

    @property
    def split(self) -> str:
        return str(self.metadata["split"])

    def frame(
        self,
        frame_index: int,
        observer_bundles: tuple[ObserverBundle, ...] = (),
    ) -> UnifiedGraphFrame:
        if not 0 <= frame_index < FORMAL_TICK_COUNT:
            raise IndexError(frame_index)
        return UnifiedGraphFrame(
            episode_id=self.episode_id,
            tick=FORMAL_TICKS[frame_index],
            state=StateFrame(
                continuous=self.continuous[frame_index],
                continuous_scope=self.continuous_scope[frame_index],
                continuous_valid=self.continuous_valid[frame_index],
                boolean=self.boolean[frame_index],
                boolean_scope=self.boolean_scope[frame_index],
                boolean_valid=self.boolean_valid[frame_index],
                categorical=self.categorical[frame_index],
                categorical_scope=self.categorical_scope[frame_index],
                categorical_valid=self.categorical_valid[frame_index],
                reference=self.reference[frame_index],
                reference_scope=self.reference_scope[frame_index],
                reference_valid=self.reference_valid[frame_index],
            ),
            predicate_truth=self.predicate_truth[frame_index],
            observer_bundles=observer_bundles,
        )

    def static_topology(self) -> Mapping[str, Any]:
        return self._cached_static_topology

    @cached_property
    def _cached_static_topology(self) -> Mapping[str, Any]:
        entity_count = len(self.entities)
        state_count = len(self.state_slots)
        sources: list[int] = []
        targets: list[int] = []
        edge_types: list[str] = []
        edge_roles: list[str] = []
        for slot in self.state_slots:
            sources.append(int(slot["subject_entity_index"]))
            targets.append(entity_count + int(slot["slot_index"]))
            edge_types.append("has_state")
            edge_roles.append(str(slot["field_name"]))
        predicate_offset = entity_count + state_count
        for slot in self.predicate_slots:
            predicate_node = predicate_offset + int(slot["slot_index"])
            roles = slot["binding_roles"]
            bindings = slot["binding_entity_indices"]
            if len(roles) != len(bindings):
                raise ModelDataError("predicate role/binding arity mismatch")
            for role, entity_index in zip(roles, bindings):
                sources.append(predicate_node)
                targets.append(int(entity_index))
                edge_types.append("predicate_argument")
                edge_roles.append(str(role))
        return {
            "node_partition_offsets": {
                "entity": 0,
                "state_slot": entity_count,
                "predicate_slot": predicate_offset,
            },
            "node_partition_counts": {
                "entity": entity_count,
                "state_slot": state_count,
                "predicate_slot": len(self.predicate_slots),
            },
            "edge_index": np.asarray([sources, targets], dtype=np.int64),
            "edge_types": tuple(edge_types),
            "edge_roles": tuple(edge_roles),
        }

    def window(
        self,
        start_frame_index: int,
        history_length: int,
        horizon: int,
        *,
        observer_provider: ObserverProvider | None = None,
    ) -> GraphWindow:
        _validate_window_shape(history_length, horizon)
        history_end = start_frame_index + history_length
        target_end = history_end + horizon
        if start_frame_index < 0 or target_end > FORMAL_TICK_COUNT:
            raise IndexError(
                f"window [{start_frame_index}, {target_end}) exceeds episode frames"
            )
        history = slice(start_frame_index, history_end)
        target = slice(history_end, target_end)
        transition = slice(history_end - 1, target_end)
        state_arrays = self._state_arrays()
        target_edits: dict[str, np.ndarray] = {}
        target_edit_supervision: dict[str, np.ndarray] = {}
        for kind in ("continuous", "boolean", "categorical", "reference"):
            edits, supervision = derive_state_edits(
                state_arrays[kind][transition],
                state_arrays[f"{kind}_scope"][transition],
                state_arrays[f"{kind}_valid"][transition],
            )
            target_edits[kind] = edits
            target_edit_supervision[kind] = supervision
        observations = tuple(
            observer_provider(self.episode_id, tick) if observer_provider else ()
            for tick in FORMAL_TICKS[start_frame_index:history_end]
        )
        return GraphWindow(
            episode_id=self.episode_id,
            split=self.split,
            history_ticks=np.asarray(
                FORMAL_TICKS[start_frame_index:history_end], dtype=np.int16
            ),
            target_ticks=np.asarray(
                FORMAL_TICKS[history_end:target_end], dtype=np.int16
            ),
            history_state={key: value[history] for key, value in state_arrays.items()},
            target_state={key: value[target] for key, value in state_arrays.items()},
            history_predicate_truth=self.predicate_truth[history],
            target_predicate_truth=self.predicate_truth[target],
            target_state_edits=target_edits,
            target_state_edit_supervision=target_edit_supervision,
            target_predicate_edits=derive_predicate_edits(
                self.predicate_truth[transition]
            ),
            topology=self.static_topology(),
            observer_bundles=observations,
        )

    def _state_arrays(self) -> dict[str, np.ndarray]:
        return {
            "continuous": self.continuous,
            "continuous_scope": self.continuous_scope,
            "continuous_valid": self.continuous_valid,
            "boolean": self.boolean,
            "boolean_scope": self.boolean_scope,
            "boolean_valid": self.boolean_valid,
            "categorical": self.categorical,
            "categorical_scope": self.categorical_scope,
            "categorical_valid": self.categorical_valid,
            "reference": self.reference,
            "reference_scope": self.reference_scope,
            "reference_valid": self.reference_valid,
        }


class AeroGraphStore:
    """Read a completed dataset with a small per-process episode cache."""

    def __init__(
        self,
        root: Path,
        *,
        observer_provider: ObserverProvider | None = None,
        cache_size: int = 2,
        split_manifest: Path | None = None,
    ) -> None:
        if cache_size < 1:
            raise ModelDataError("cache_size must be positive")
        self.root = root.resolve()
        self.manifest = _load_json(self.root / "manifest.json")
        _validate_dataset_document(self.manifest)
        self.schema = self.manifest["logical_schema"]
        self.observer_provider = observer_provider
        observer_status = self.manifest["observer_bundle_contract"]["status"]
        if (
            observer_status != "not_materialized_in_server_handoff"
            and observer_provider is None
        ):
            raise ModelDataError(
                "materialized observer evidence requires an ObserverProvider"
            )
        episode_index_columns = [
            "episode_id",
            "profile_id",
            "split",
            "capture_boundary_id",
            "row_group_index",
            "entity_count",
            "state_slot_count",
            "continuous_width",
            "boolean_width",
            "categorical_width",
            "reference_width",
            "predicate_slot_count",
        ]
        episodes_table = pq.read_table(
            self.root / "episodes.parquet", columns=episode_index_columns
        )
        rows = episodes_table.to_pylist()
        if split_manifest is not None:
            from .external_split import load_external_episode_splits
            if any(row["split"] != "unassigned" for row in rows):
                raise ModelDataError("external split cannot override embedded assignments")
            assignments = load_external_episode_splits(
                split_manifest, {row["episode_id"]: row["profile_id"] for row in rows},
            )
            rows = [{**row, "split": assignments[row["episode_id"]]} for row in rows]
        self.episodes = {str(row["episode_id"]): row for row in rows}
        if len(self.episodes) != int(self.manifest["episode_count"]):
            raise ModelDataError("episode index count disagrees with manifest")
        self._episode_file = pq.ParquetFile(self.root / "episodes.parquet")
        self._entity_file = pq.ParquetFile(self.root / "entities.parquet")
        self._state_slot_file = pq.ParquetFile(self.root / "state_slots.parquet")
        self._predicate_slot_file = pq.ParquetFile(
            self.root / "predicate_slots.parquet"
        )
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EpisodeGraphSequence] = OrderedDict()

    def get_episode(self, episode_id: str) -> EpisodeGraphSequence:
        cached = self._cache.pop(episode_id, None)
        if cached is not None:
            self._cache[episode_id] = cached
            return cached
        metadata = self.episodes.get(episode_id)
        if metadata is None:
            raise KeyError(episode_id)
        sequence = self._decode_episode(metadata)
        self._cache[episode_id] = sequence
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return sequence

    def get_frame(self, episode_id: str, tick: int) -> UnifiedGraphFrame:
        try:
            frame_index = FORMAL_TICKS.index(tick)
        except ValueError as exc:
            raise ModelDataError(f"tick is not formal: {tick}") from exc
        observations = (
            self.observer_provider(episode_id, tick)
            if self.observer_provider is not None
            else ()
        )
        return self.get_episode(episode_id).frame(frame_index, observations)

    def get_window(
        self,
        episode_id: str,
        start_frame_index: int,
        history_length: int,
        horizon: int,
    ) -> GraphWindow:
        return self.get_episode(episode_id).window(
            start_frame_index,
            history_length,
            horizon,
            observer_provider=self.observer_provider,
        )

    def ragged_at(self, episode_id: str, tick: int) -> tuple[Mapping[str, Any], ...]:
        if tick not in FORMAL_TICKS:
            raise ModelDataError(f"tick is not formal: {tick}")
        table = pq.read_table(
            self.root / "ragged_state.parquet",
            filters=[("episode_id", "=", episode_id), ("tick", "=", tick)],
        )
        return tuple(table.to_pylist())

    def _decode_episode(self, metadata: Mapping[str, Any]) -> EpisodeGraphSequence:
        row_group = int(metadata["row_group_index"])
        entities = tuple(self._entity_file.read_row_group(row_group).to_pylist())
        state_slots = tuple(self._state_slot_file.read_row_group(row_group).to_pylist())
        predicate_slots = tuple(
            self._predicate_slot_file.read_row_group(row_group).to_pylist()
        )
        tensor_columns = [
            "episode_id",
            "continuous_values",
            "continuous_scope",
            "continuous_valid",
            "boolean_values",
            "boolean_scope",
            "boolean_valid",
            "categorical_values",
            "categorical_scope",
            "categorical_valid",
            "reference_values",
            "reference_scope",
            "reference_valid",
            "predicate_truth_codes",
        ]
        tensor_table = self._episode_file.read_row_group(
            row_group, columns=tensor_columns
        )
        if tensor_table.num_rows != 1:
            raise ModelDataError("episode tensor row group must contain one row")
        payload = {
            name: tensor_table[name][0].as_py() for name in tensor_table.column_names
        }
        episode_id = str(metadata["episode_id"])
        if payload["episode_id"] != episode_id:
            raise ModelDataError("episode tensor row group is misaligned")
        continuous_width = int(metadata["continuous_width"])
        boolean_width = int(metadata["boolean_width"])
        categorical_width = int(metadata["categorical_width"])
        reference_width = int(metadata["reference_width"])
        predicate_width = int(metadata["predicate_slot_count"])
        continuous_shape = (FORMAL_TICK_COUNT, continuous_width)
        boolean_shape = (FORMAL_TICK_COUNT, boolean_width)
        categorical_shape = (FORMAL_TICK_COUNT, categorical_width)
        reference_shape = (FORMAL_TICK_COUNT, reference_width)
        predicate_shape = (FORMAL_TICK_COUNT, predicate_width)
        continuous = _decode_numeric(
            payload["continuous_values"], "<f8", continuous_shape
        )
        boolean = unpack_bits(payload["boolean_values"], boolean_shape)
        categorical = _decode_numeric(
            payload["categorical_values"], "<i8", categorical_shape
        )
        reference = _decode_numeric(payload["reference_values"], "<i4", reference_shape)
        predicate_truth = _decode_numeric(
            payload["predicate_truth_codes"], "<u1", predicate_shape
        )
        if len(entities) != int(metadata["entity_count"]):
            raise ModelDataError(f"{episode_id}: entity registry count mismatch")
        if len(state_slots) != int(metadata["state_slot_count"]):
            raise ModelDataError(f"{episode_id}: state slot registry count mismatch")
        if len(predicate_slots) != predicate_width:
            raise ModelDataError(
                f"{episode_id}: predicate slot registry count mismatch"
            )
        predicate_masks(predicate_truth)
        return EpisodeGraphSequence(
            metadata=metadata,
            entities=entities,
            state_slots=state_slots,
            predicate_slots=predicate_slots,
            continuous=continuous,
            continuous_scope=unpack_bits(payload["continuous_scope"], continuous_shape),
            continuous_valid=unpack_bits(payload["continuous_valid"], continuous_shape),
            boolean=boolean,
            boolean_scope=unpack_bits(payload["boolean_scope"], boolean_shape),
            boolean_valid=unpack_bits(payload["boolean_valid"], boolean_shape),
            categorical=categorical,
            categorical_scope=unpack_bits(
                payload["categorical_scope"], categorical_shape
            ),
            categorical_valid=unpack_bits(
                payload["categorical_valid"], categorical_shape
            ),
            reference=reference,
            reference_scope=unpack_bits(payload["reference_scope"], reference_shape),
            reference_valid=unpack_bits(payload["reference_valid"], reference_shape),
            predicate_truth=predicate_truth,
        )


class AeroGraphWindowDataset:
    """Graph windows with external splits; stored projection bytes stay neutral."""

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        history_length: int = 8,
        horizon: int = 10,
        observer_provider: ObserverProvider | None = None,
        split_manifest: Path | None = None,
        action_feature_provider: ActionFeatureProvider | None = None,
    ) -> None:
        _validate_window_shape(history_length, horizon)
        if split not in {"train", "validation", "test"}:
            raise ModelDataError(f"invalid split: {split}")
        self.root = root.resolve()
        self.split = split
        self.history_length = history_length
        self.horizon = horizon
        self.observer_provider = observer_provider
        self.action_feature_provider = action_feature_provider
        manifest = _load_json(self.root / "manifest.json")
        from .external_split import load_external_episode_splits

        split_contract = manifest.get("split_contract")
        if not isinstance(split_contract, Mapping) or split_contract.get("status") != "unassigned":
            raise ModelDataError("graph projection must remain split-unassigned")
        if split_manifest is None:
            raise ModelDataError("training windows require an external assigned split manifest; projection is split-unassigned")
        self.split_manifest = split_manifest.resolve()
        graph_episodes = pq.read_table(
            self.root / "episodes.parquet", columns=["episode_id", "profile_id", "split"],
        ).to_pylist()
        if any(row["split"] != "unassigned" for row in graph_episodes):
            raise ModelDataError("graph projection contains an embedded split")
        assignments = load_external_episode_splits(
            self.split_manifest, {row["episode_id"]: row["profile_id"] for row in graph_episodes},
        )
        episodes = [row for row in graph_episodes if assignments[row["episode_id"]] == split]
        count = FORMAL_TICK_COUNT - history_length - horizon + 1
        self.index = tuple(
            (str(episode["episode_id"]), start)
            for episode in episodes
            for start in range(count)
        )
        self._store: AeroGraphStore | None = None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> GraphWindow:
        episode_id, start = self.index[index]
        if self._store is None:
            self._store = AeroGraphStore(
                self.root,
                observer_provider=self.observer_provider,
                split_manifest=self.split_manifest,
            )
        window = self._store.get_window(
            episode_id,
            start,
            self.history_length,
            self.horizon,
        )
        if self.action_feature_provider is not None:
            history_actions, target_actions = self.action_feature_provider(
                window.episode_id,
                window.history_ticks,
                window.target_ticks,
            )
            window = replace(
                window,
                history_action_features=history_actions,
                target_action_features=target_actions,
            )
        return replace(window, split=self.split)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_store"] = None
        return state


def derive_state_edits(
    values: np.ndarray,
    scope: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if values.shape != scope.shape or values.shape != valid.shape:
        raise ModelDataError("state edit arrays must have identical shapes")
    if values.shape[0] < 2:
        raise ModelDataError("state edit derivation needs at least two frames")
    previous_scope = scope[:-1]
    current_scope = scope[1:]
    previous_valid = valid[:-1]
    current_valid = valid[1:]
    edits = np.full(previous_scope.shape, EDIT_CODES["retain"], dtype=np.uint8)
    birth = ~previous_scope & current_scope
    death = previous_scope & ~current_scope
    comparable = previous_valid & current_valid
    changed = comparable & (values[:-1] != values[1:])
    edits[birth] = EDIT_CODES["birth"]
    edits[death] = EDIT_CODES["death"]
    edits[changed] = EDIT_CODES["value_change"]
    supervision = birth | death | comparable
    return edits, supervision


def derive_predicate_edits(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ModelDataError("predicate edit derivation needs a [T,P] tensor")
    previous = values[:-1]
    current = values[1:]
    previous_active = previous != INACTIVE_PREDICATE_CODE
    current_active = current != INACTIVE_PREDICATE_CODE
    edits = np.full(previous.shape, EDIT_CODES["retain"], dtype=np.uint8)
    edits[~previous_active & current_active] = EDIT_CODES["birth"]
    edits[previous_active & ~current_active] = EDIT_CODES["death"]
    edits[previous_active & current_active & (previous != current)] = EDIT_CODES[
        "value_change"
    ]
    return edits


def collate_graph_windows(samples: Sequence[GraphWindow]) -> Mapping[str, Any]:
    import torch

    if not samples:
        raise ModelDataError("cannot collate an empty graph-window batch")
    history_length = len(samples[0].history_ticks)
    horizon = len(samples[0].target_ticks)
    if any(
        len(sample.history_ticks) != history_length
        or len(sample.target_ticks) != horizon
        for sample in samples
    ):
        raise ModelDataError("batch windows must share K and H")
    batch: dict[str, Any] = {
        "episode_id": [sample.episode_id for sample in samples],
        "split": [sample.split for sample in samples],
        "history_ticks": torch.from_numpy(
            np.stack([sample.history_ticks for sample in samples])
        ),
        "target_ticks": torch.from_numpy(
            np.stack([sample.target_ticks for sample in samples])
        ),
        "history_state": {},
        "target_state": {},
        "target_state_edits": {},
        "target_state_edit_supervision": {},
        "topology": [sample.topology for sample in samples],
        "observer_bundles": [sample.observer_bundles for sample in samples],
        "state_slot_padding_mask": {},
    }
    for kind, fill_value in (
        ("continuous", np.nan),
        ("boolean", False),
        ("categorical", 0),
        ("reference", -1),
    ):
        width = max(sample.history_state[kind].shape[1] for sample in samples)
        for section, length in (
            ("history_state", history_length),
            ("target_state", horizon),
        ):
            for suffix, fill in (
                ("", fill_value),
                ("_scope", False),
                ("_valid", False),
            ):
                arrays = [getattr(sample, section)[kind + suffix] for sample in samples]
                padded = _pad_last_dimension(arrays, length, width, fill)
                batch[section][kind + suffix] = torch.from_numpy(padded)
        edit_arrays = [sample.target_state_edits[kind] for sample in samples]
        edit_masks = [sample.target_state_edit_supervision[kind] for sample in samples]
        batch["target_state_edits"][kind] = torch.from_numpy(
            _pad_last_dimension(edit_arrays, horizon, width, EDIT_CODES["retain"])
        )
        batch["target_state_edit_supervision"][kind] = torch.from_numpy(
            _pad_last_dimension(edit_masks, horizon, width, False)
        )
        padding = np.zeros((len(samples), width), dtype=np.bool_)
        for index, sample in enumerate(samples):
            padding[index, : sample.history_state[kind].shape[1]] = True
        batch["state_slot_padding_mask"][kind] = torch.from_numpy(padding)
    predicate_width = max(sample.history_predicate_truth.shape[1] for sample in samples)
    batch["history_predicate_truth"] = torch.from_numpy(
        _pad_last_dimension(
            [sample.history_predicate_truth for sample in samples],
            history_length,
            predicate_width,
            INACTIVE_PREDICATE_CODE,
        )
    )
    batch["target_predicate_truth"] = torch.from_numpy(
        _pad_last_dimension(
            [sample.target_predicate_truth for sample in samples],
            horizon,
            predicate_width,
            INACTIVE_PREDICATE_CODE,
        )
    )
    batch["target_predicate_edits"] = torch.from_numpy(
        _pad_last_dimension(
            [sample.target_predicate_edits for sample in samples],
            horizon,
            predicate_width,
            EDIT_CODES["retain"],
        )
    )
    predicate_padding = np.zeros((len(samples), predicate_width), dtype=np.bool_)
    for index, sample in enumerate(samples):
        predicate_padding[index, : sample.history_predicate_truth.shape[1]] = True
    batch["predicate_slot_padding_mask"] = torch.from_numpy(predicate_padding)
    action_history = [sample.history_action_features for sample in samples]
    action_target = [sample.target_action_features for sample in samples]
    if any(value is not None for value in action_history + action_target):
        if not all(value is not None for value in action_history + action_target):
            raise ModelDataError("action features must be present for every batch window")
        assert all(value is not None for value in action_history)
        assert all(value is not None for value in action_target)
        if any(
            value.ndim != 2
            or value.shape[0] != history_length
            or not np.isfinite(value).all()
            for value in action_history
        ) or any(
            value.ndim != 2
            or value.shape[0] != horizon
            or not np.isfinite(value).all()
            for value in action_target
        ):
            raise ModelDataError("action feature arrays have invalid shape or values")
        widths = {
            int(value.shape[1]) for value in action_history + action_target
        }
        if len(widths) != 1 or not widths:
            raise ModelDataError("action feature widths disagree across windows")
        batch["history_action_features"] = torch.from_numpy(
            np.stack(action_history).astype(np.float32, copy=False)
        )
        batch["target_action_features"] = torch.from_numpy(
            np.stack(action_target).astype(np.float32, copy=False)
        )
    return batch


def _pad_last_dimension(
    arrays: Sequence[np.ndarray],
    time_length: int,
    width: int,
    fill_value: Any,
) -> np.ndarray:
    result = np.full(
        (len(arrays), time_length, width),
        fill_value,
        dtype=arrays[0].dtype,
    )
    for index, array in enumerate(arrays):
        if array.shape[0] != time_length:
            raise ModelDataError("array time dimension disagrees with window")
        result[index, :, : array.shape[1]] = array
    return result


def _decode_numeric(payload: bytes, dtype: str, shape: tuple[int, int]) -> np.ndarray:
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(payload) != expected:
        raise ModelDataError(
            f"tensor payload has {len(payload)} bytes, expected {expected}"
        )
    return np.frombuffer(payload, dtype=dtype).reshape(shape)


def _validate_window_shape(history_length: int, horizon: int) -> None:
    if history_length not in ALLOWED_HISTORY_LENGTHS | {1}:
        raise ModelDataError(
            f"history length is outside the dataset contract: {history_length}"
        )
    if horizon not in ALLOWED_HORIZONS:
        raise ModelDataError(f"horizon is outside the dataset contract: {horizon}")


def _load_json(path: Path) -> dict[str, Any]:
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ModelDataError(f"{path}: root must be an object")
    return value


def _validate_dataset_document(manifest: Mapping[str, Any]) -> None:
    expected = (DATASET_SCHEMA_NAME, DATASET_SCHEMA_VERSION)
    actual = (manifest.get("schema_name"), manifest.get("schema_version"))
    if actual != expected:
        raise ModelDataError(f"manifest contract mismatch: {actual} != {expected}")
    schema = manifest.get("logical_schema")
    if not isinstance(schema, Mapping):
        raise ModelDataError("manifest lacks the logical schema")
    schema_actual = (schema.get("schema_name"), schema.get("schema_version"))
    if schema_actual != expected:
        raise ModelDataError(
            f"logical schema contract mismatch: {schema_actual} != {expected}"
        )
