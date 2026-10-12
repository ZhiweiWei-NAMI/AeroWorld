"""Leakage-safe fixed-width features for authored action commands."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .action_contract import (
    ACTION_SCHEMA_NAME,
    ACTION_SCHEMA_VERSION,
    ActionCommandRecord,
    TargetBindingRecord,
)
from .contract import ModelDataError


ACTION_TYPE_VOCAB = (
    "move_entity",
    "set_runtime_state",
    "set_visual_state",
    "capture_screenshot",
    "spawn_entity",
    "remove_entity",
    "set_weather",
    "set_pedestrian_activity",
)
ACTION_FAMILY_VOCAB = (
    "motion",
    "runtime_mutation",
    "visual_state",
    "capture_request",
    "entity_lifecycle",
    "environment",
    "pedestrian_activity",
)
ACTION_SCOPE_VOCAB = ("entity", "sensor", "global")
ACTION_FEATURE_DIM = (
    len(ACTION_TYPE_VOCAB)
    + len(ACTION_FAMILY_VOCAB)
    + len(ACTION_SCOPE_VOCAB)
    + 7
)


class ActionFeatureStore:
    """Index command-only features at formal ticks.

    History features describe the command issued at the preceding formal tick.
    Target features include only commands available by the final history tick,
    never commands actually issued later in the rollout. A zero target vector
    means no newly issued command known at the origin, not proof that no future
    action will occur. This issue-pulse summary does not encode persistent active
    processes or a complete candidate action plan. No process, outcome, terminal,
    or evidence field is read when constructing these vectors.
    """

    feature_dim = ACTION_FEATURE_DIM

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest = self._object(self.root / "manifest.json")
        if (manifest.get("schema_name"), manifest.get("schema_version")) != (
            ACTION_SCHEMA_NAME,
            ACTION_SCHEMA_VERSION,
        ):
            raise ModelDataError("action feature source schema mismatch")
        split_contract = manifest.get("split_contract")
        if not isinstance(split_contract, Mapping) or split_contract.get("status") != "unassigned":
            raise ModelDataError("action feature source must remain split-unassigned")
        self._verify_artifact(manifest, "episodes.jsonl")
        self._verify_artifact(manifest, "commands.jsonl")
        episode_rows = self._jsonl(self.root / "episodes.jsonl")
        command_rows = self._jsonl(self.root / "commands.jsonl")
        expected_count = int((manifest.get("record_counts") or {}).get("commands", -1))
        if len(command_rows) != expected_count:
            raise ModelDataError("action command count disagrees with its manifest")
        self.episode_ids = frozenset(str(row["episode_id"]) for row in episode_rows)
        if len(self.episode_ids) != len(episode_rows):
            raise ModelDataError("action episode index contains duplicate episodes")
        by_episode: dict[str, list[ActionCommandRecord]] = defaultdict(list)
        for row in command_rows:
            command = self._decode_command(row)
            if command.episode_id not in self.episode_ids:
                raise ModelDataError(f"command references unknown episode {command.episode_id}")
            by_episode[command.episode_id].append(command)
        self._commands: dict[str, dict[int, tuple[ActionCommandRecord, ...]]] = {}
        for row in episode_rows:
            episode_id = str(row["episode_id"])
            commands = by_episode[episode_id]
            if len(commands) != int(row.get("command_count", -1)):
                raise ModelDataError(f"{episode_id}: command count disagrees with index")
            by_tick: dict[int, list[ActionCommandRecord]] = defaultdict(list)
            for command in commands:
                by_tick[int(command.issued_tick)].append(command)
            self._commands[episode_id] = {
                tick: tuple(values) for tick, values in by_tick.items()
            }

    def window_features(
        self,
        episode_id: str,
        history_ticks: Sequence[int],
        target_ticks: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if episode_id not in self._commands:
            raise ModelDataError(f"action source lacks episode {episode_id}")
        history = tuple(int(value) for value in history_ticks)
        target = tuple(int(value) for value in target_ticks)
        if not history or not target:
            raise ModelDataError("action features require non-empty history and target")
        if any(next_tick - tick != 5 for tick, next_tick in zip(history, history[1:])):
            raise ModelDataError("history ticks are not on the formal five-tick grid")
        if any(next_tick - tick != 5 for tick, next_tick in zip(target, target[1:])):
            raise ModelDataError("target ticks are not on the formal five-tick grid")
        expected_first = history[-1] + 5
        if target[0] != expected_first:
            raise ModelDataError("target does not immediately follow history")
        history_features = np.stack(
            [self._at_tick(episode_id, tick - 5, available_by_tick=tick) for tick in history]
        ).astype(np.float32, copy=False)
        target_features = np.stack(
            [self._at_tick(episode_id, tick - 5, available_by_tick=history[-1]) for tick in target]
        ).astype(np.float32, copy=False)
        return history_features, target_features

    def __call__(
        self,
        episode_id: str,
        history_ticks: Sequence[int],
        target_ticks: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.window_features(episode_id, history_ticks, target_ticks)

    def _at_tick(
        self, episode_id: str, tick: int, *, available_by_tick: int
    ) -> np.ndarray:
        result = np.zeros(self.feature_dim, dtype=np.float32)
        commands = tuple(
            command for command in self._commands[episode_id].get(int(tick), ())
            if command.availability_tick <= available_by_tick
        )
        type_offset = 0
        family_offset = type_offset + len(ACTION_TYPE_VOCAB)
        scope_offset = family_offset + len(ACTION_FAMILY_VOCAB)
        scalar_offset = scope_offset + len(ACTION_SCOPE_VOCAB)
        if not commands:
            return result
        for command in commands:
            try:
                result[type_offset + ACTION_TYPE_VOCAB.index(command.action_type)] += 1.0
                result[family_offset + ACTION_FAMILY_VOCAB.index(command.action_family)] += 1.0
                result[scope_offset + ACTION_SCOPE_VOCAB.index(command.scope)] += 1.0
            except ValueError as exc:
                raise ModelDataError(
                    f"unsupported action vocabulary item at {episode_id}:{tick}"
                ) from exc
        target_count = sum(len(command.target_bindings) for command in commands)
        actor_count = sum(command.actor_id is not None for command in commands)
        controller_count = sum(command.controller_id is not None for command in commands)
        duration_values = [
            command.planned_duration_ticks
            for command in commands
            if command.planned_duration_ticks is not None
        ]
        start_offsets = [
            command.planned_start_tick - command.issued_tick for command in commands
        ]
        result[scalar_offset : scalar_offset + 7] = np.asarray(
            (
                math.log1p(len(commands)),
                math.log1p(target_count),
                math.log1p(actor_count),
                math.log1p(controller_count),
                math.log1p(len(duration_values)),
                float(np.mean(start_offsets)),
                math.log1p(float(np.mean(duration_values))) if duration_values else 0.0,
            ),
            dtype=np.float32,
        )
        return result

    @staticmethod
    def _decode_command(value: Mapping[str, object]) -> ActionCommandRecord:
        payload = dict(value)
        bindings = payload.get("target_bindings")
        if not isinstance(bindings, list):
            raise ModelDataError("action command target bindings are not a list")
        payload["target_bindings"] = tuple(
            TargetBindingRecord(
                target_id=str(item["target_id"]),
                roles=tuple(str(role) for role in item["roles"]),
                sources=tuple(str(source) for source in item["sources"]),
            )
            for item in bindings
        )
        declared_roles = payload.get("declared_event_target_roles")
        if not isinstance(declared_roles, list):
            raise ModelDataError("action command target roles are not a list")
        payload["declared_event_target_roles"] = tuple(
            str(role) for role in declared_roles
        )
        return ActionCommandRecord(**payload)

    def _verify_artifact(self, manifest: Mapping[str, object], name: str) -> None:
        artifacts = manifest.get("artifacts")
        metadata = artifacts.get(name) if isinstance(artifacts, Mapping) else None
        path = self.root / name
        if not isinstance(metadata, Mapping) or not path.is_file():
            raise ModelDataError(f"action feature source lacks artifact {name}")
        if path.stat().st_size != int(metadata["bytes"]):
            raise ModelDataError(f"action feature artifact byte count mismatch: {name}")

    @staticmethod
    def _object(path: Path) -> dict[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelDataError(f"{path}: invalid JSON object") from exc
        if not isinstance(value, dict):
            raise ModelDataError(f"{path}: root must be an object")
        return value

    @staticmethod
    def _jsonl(path: Path) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ModelDataError(f"{path}: action index row is not an object")
                    rows.append(value)
        return rows
