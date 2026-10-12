"""Atomic storage for the split-neutral AERO action/process projection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence

from .action_contract import (
    ACTION_SCHEMA_NAME,
    ACTION_SCHEMA_VERSION,
    ActionCommandRecord,
    ActionProjectionDocument,
    OutcomeRecord,
    ProcessRecord,
    TargetBindingRecord,
    record_dict,
)
from .action_projection import (
    discover_episode_roots,
    discover_scenario_scripts,
    project_action_episode,
)
from .contract import ModelDataError, canonical_json, parse_episode_id, stable_identifier


ACTION_DATA_FILES = (
    "episodes.jsonl",
    "commands.jsonl",
    "processes.jsonl",
    "outcomes.jsonl",
)


@dataclass(frozen=True)
class ActionBuildResult:
    output_root: Path
    manifest: Mapping[str, Any]


def build_action_dataset(
    episodes_root: Path,
    scenario_root: Path,
    output_root: Path,
    *,
    episode_ids: Sequence[str] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> ActionBuildResult:
    """Build a deterministic projection without assigning any dataset split."""

    all_roots = discover_episode_roots(episodes_root)
    scripts = discover_scenario_scripts(scenario_root)
    profiles = {parse_episode_id(path.name)[0] for path in all_roots}
    missing_scripts = sorted(profiles - set(scripts))
    if missing_scripts:
        raise ModelDataError(f"profiles lack event scripts: {missing_scripts}")
    by_id = {path.name: path for path in all_roots}
    selected_ids = sorted(episode_ids) if episode_ids else sorted(by_id)
    if not selected_ids:
        raise ModelDataError("at least one episode must be selected")
    if len(selected_ids) != len(set(selected_ids)):
        raise ModelDataError("episode selection contains duplicates")
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise ModelDataError(f"unknown episode ids: {unknown}")

    output_root = output_root.resolve()
    if output_root.exists():
        raise ModelDataError(f"output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    if staging.exists():
        raise ModelDataError(f"staging path already exists: {staging}")
    staging.mkdir()

    handles: dict[str, Any] = {}
    try:
        handles = {
            name: (staging / name).open("w", encoding="utf-8", newline="\n")
            for name in ACTION_DATA_FILES
        }
        counts = Counter()
        action_types = Counter()
        process_statuses = Counter()
        target_sizes = Counter()
        self_only_targets = 0
        profiles_seen: set[str] = set()
        for index, episode_id in enumerate(selected_ids, start=1):
            profile_id, seed_label = parse_episode_id(episode_id)
            document = project_action_episode(
                by_id[episode_id], event_script_path=scripts[profile_id]
            )
            command_rows = [record_dict(value) for value in document.commands]
            process_rows = [record_dict(value) for value in document.processes]
            outcome_rows = [record_dict(value) for value in document.outcomes]
            episode_row = {
                "schema_name": ACTION_SCHEMA_NAME,
                "schema_version": ACTION_SCHEMA_VERSION,
                "episode_id": episode_id,
                "profile_id": profile_id,
                "seed_label": seed_label,
                "split": "unassigned",
                "command_count": len(command_rows),
                "process_count": len(process_rows),
                "outcome_count": len(outcome_rows),
            }
            _write_row(handles["episodes.jsonl"], episode_row)
            for row in command_rows:
                _write_row(handles["commands.jsonl"], row)
            for row in process_rows:
                _write_row(handles["processes.jsonl"], row)
            for row in outcome_rows:
                _write_row(handles["outcomes.jsonl"], row)

            profiles_seen.add(profile_id)
            counts["episodes"] += 1
            counts["commands"] += len(command_rows)
            counts["processes"] += len(process_rows)
            counts["outcomes"] += len(outcome_rows)
            for command in document.commands:
                action_types[command.action_type] += 1
                target_ids = tuple(
                    binding.target_id for binding in command.target_bindings
                )
                target_sizes[len(target_ids)] += 1
                self_only_targets += int(
                    command.actor_id is not None
                    and target_ids == (command.actor_id,)
                )
            process_statuses.update(
                process.terminal_status for process in document.processes
            )
            if progress is not None:
                progress(index, len(selected_ids), episode_id)
        for handle in handles.values():
            handle.close()
        handles = {}

        artifacts = {
            name: {
                "bytes": (staging / name).stat().st_size,
            }
            for name in ACTION_DATA_FILES
        }
        dataset_id = stable_identifier(
            "aero_action_dataset",
            ACTION_SCHEMA_VERSION,
            selected_ids,
        )
        manifest = {
            "schema_name": ACTION_SCHEMA_NAME,
            "schema_version": ACTION_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "build_mode": (
                "formal_full" if len(selected_ids) == len(all_roots) else "selected_canary"
            ),
            "episode_count": counts["episodes"],
            "profile_count": len(profiles_seen),
            "record_counts": {
                "commands": counts["commands"],
                "processes": counts["processes"],
                "outcomes": counts["outcomes"],
            },
            "action_type_counts": dict(sorted(action_types.items())),
            "process_terminal_status_counts": dict(sorted(process_statuses.items())),
            "target_count_distribution": {
                str(key): value for key, value in sorted(target_sizes.items())
            },
            "self_only_target_count": self_only_targets,
            "split_contract": {
                "status": "unassigned",
                "embedded_split": False,
                "group_key": "profile_id",
                "all_three_seeds_stay_together": True,
                "external_manifest_required_before_training": True,
            },
            "visibility_contract": {
                "command_fields": "available at issued_tick",
                "planned_duration": (
                    "0 only for instantaneous actions; null for movement when no "
                    "origin-authoritative plan duration was persisted"
                ),
                "realized_lifecycle": (
                    "stored in process/outcome records and exposed only at each "
                    "field's availability tick"
                ),
                "future_outcome_in_command": False,
            },
            "lifecycle_contract": {
                "movement_completion": "event_realization.terminal_tick",
                "movement_cancellation": (
                    "a later move for the same actor supersedes unfinished movement; "
                    "the replacement semantics are explicit in batch_generate.py"
                ),
                "duration_interval": (
                    "inclusive realized_start_tick..active_last_tick for durative processes"
                ),
                "instantaneous_duration_ticks": 0,
            },
            "artifacts": artifacts,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
        return ActionBuildResult(output_root=output_root, manifest=manifest)
    except BaseException:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_action_dataset(
    dataset_root: Path,
    *,
    episodes_root: Path | None = None,
    scenario_root: Path | None = None,
    deep: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Validate artifacts, linkage, and optional source round-trips."""

    dataset_root = dataset_root.resolve()
    manifest = _load_object(dataset_root / "manifest.json")
    if (manifest.get("schema_name"), manifest.get("schema_version")) != (
        ACTION_SCHEMA_NAME,
        ACTION_SCHEMA_VERSION,
    ):
        raise ModelDataError("action dataset schema mismatch")
    split_contract = manifest.get("split_contract")
    if not isinstance(split_contract, Mapping) or split_contract.get(
        "status"
    ) != "unassigned" or split_contract.get("embedded_split") is not False:
        raise ModelDataError("action projection contains an assigned split")
    expected_files = {"manifest.json", *ACTION_DATA_FILES}
    actual_files = {path.name for path in dataset_root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ModelDataError(
            f"action dataset file set mismatch: {sorted(actual_files)}"
        )
    for name in ACTION_DATA_FILES:
        artifact = manifest.get("artifacts", {}).get(name)
        if not isinstance(artifact, Mapping) or set(artifact) != {"bytes"}:
            raise ModelDataError(f"manifest lacks artifact metadata for {name}")
        path = dataset_root / name
        if path.stat().st_size != int(artifact["bytes"]):
            raise ModelDataError(f"artifact byte count mismatch: {name}")

    episode_rows = _load_jsonl(dataset_root / "episodes.jsonl")
    episode_columns = {
        "schema_name", "schema_version", "episode_id", "profile_id", "seed_label",
        "split", "command_count", "process_count", "outcome_count",
    }
    if any(set(row) != episode_columns for row in episode_rows):
        raise ModelDataError("episode projection contains unsupported metadata")
    command_rows = _load_jsonl(dataset_root / "commands.jsonl")
    process_rows = _load_jsonl(dataset_root / "processes.jsonl")
    outcome_rows = _load_jsonl(dataset_root / "outcomes.jsonl")
    if len(episode_rows) != int(manifest["episode_count"]):
        raise ModelDataError("episode count disagrees with manifest")
    expected_counts = manifest["record_counts"]
    for name, rows in (
        ("commands", command_rows),
        ("processes", process_rows),
        ("outcomes", outcome_rows),
    ):
        if len(rows) != int(expected_counts[name]):
            raise ModelDataError(f"{name} count disagrees with manifest")

    commands = [_decode_command(value) for value in command_rows]
    processes = [_decode_process(value) for value in process_rows]
    outcomes = [_decode_outcome(value) for value in outcome_rows]
    command_keys = {(value.episode_id, value.action_id) for value in commands}
    if len(command_keys) != len(commands):
        raise ModelDataError("action ids are not episode-unique")
    if {(value.episode_id, value.action_id) for value in processes} != command_keys:
        raise ModelDataError("action/process linkage is incomplete")
    if {(value.episode_id, value.action_id) for value in outcomes} != command_keys:
        raise ModelDataError("action/outcome linkage is incomplete")
    if any(row.get("split") != "unassigned" for row in episode_rows):
        raise ModelDataError("episode projection embeds an assigned split")

    grouped_commands = _group_records(commands)
    grouped_processes = _group_records(processes)
    grouped_outcomes = _group_records(outcomes)
    for row in episode_rows:
        episode_id = str(row["episode_id"])
        if len(grouped_commands[episode_id]) != int(row["command_count"]):
            raise ModelDataError(f"{episode_id}: command count mismatch")
        if len(grouped_processes[episode_id]) != int(row["process_count"]):
            raise ModelDataError(f"{episode_id}: process count mismatch")
        if len(grouped_outcomes[episode_id]) != int(row["outcome_count"]):
            raise ModelDataError(f"{episode_id}: outcome count mismatch")

    if deep:
        if episodes_root is None or scenario_root is None:
            raise ModelDataError("deep validation requires both source roots")
        episode_paths = {path.name: path for path in discover_episode_roots(episodes_root)}
        scripts = discover_scenario_scripts(scenario_root)
        for index, row in enumerate(episode_rows, start=1):
            episode_id = str(row["episode_id"])
            profile_id, _seed = parse_episode_id(episode_id)
            rebuilt = project_action_episode(
                episode_paths[episode_id], event_script_path=scripts[profile_id]
            )
            if [record_dict(value) for value in rebuilt.commands] != [
                record_dict(value) for value in grouped_commands[episode_id]
            ]:
                raise ModelDataError(f"{episode_id}: command source round-trip differs")
            if [record_dict(value) for value in rebuilt.processes] != [
                record_dict(value) for value in grouped_processes[episode_id]
            ]:
                raise ModelDataError(f"{episode_id}: process source round-trip differs")
            if [record_dict(value) for value in rebuilt.outcomes] != [
                record_dict(value) for value in grouped_outcomes[episode_id]
            ]:
                raise ModelDataError(f"{episode_id}: outcome source round-trip differs")
            if progress is not None:
                progress(index, len(episode_rows), episode_id)

    return {
        "status": "PASS",
        "dataset_id": manifest["dataset_id"],
        "episode_count": len(episode_rows),
        "profile_count": int(manifest["profile_count"]),
        "command_count": len(commands),
        "process_count": len(processes),
        "outcome_count": len(outcomes),
        "deep_source_roundtrip": deep,
    }


def load_action_document(dataset_root: Path, episode_id: str) -> ActionProjectionDocument:
    """Load one episode document from a validated projection root."""

    dataset_root = dataset_root.resolve()
    episode_rows = [
        row
        for row in _load_jsonl(dataset_root / "episodes.jsonl")
        if row.get("episode_id") == episode_id
    ]
    if len(episode_rows) != 1:
        raise KeyError(episode_id)
    episode = episode_rows[0]
    commands = tuple(
        _decode_command(row)
        for row in _load_jsonl(dataset_root / "commands.jsonl")
        if row.get("episode_id") == episode_id
    )
    processes = tuple(
        _decode_process(row)
        for row in _load_jsonl(dataset_root / "processes.jsonl")
        if row.get("episode_id") == episode_id
    )
    outcomes = tuple(
        _decode_outcome(row)
        for row in _load_jsonl(dataset_root / "outcomes.jsonl")
        if row.get("episode_id") == episode_id
    )
    return ActionProjectionDocument(
        schema_name=ACTION_SCHEMA_NAME,
        schema_version=ACTION_SCHEMA_VERSION,
        episode_id=episode_id,
        profile_id=str(episode["profile_id"]),
        seed_label=str(episode["seed_label"]),
        commands=commands,
        processes=processes,
        outcomes=outcomes,
    )


def _decode_command(value: Mapping[str, Any]) -> ActionCommandRecord:
    payload = dict(value)
    payload["target_bindings"] = tuple(
        TargetBindingRecord(
            target_id=str(item["target_id"]),
            roles=tuple(str(role) for role in item["roles"]),
            sources=tuple(str(source) for source in item["sources"]),
        )
        for item in payload["target_bindings"]
    )
    payload["declared_event_target_roles"] = tuple(
        str(role) for role in payload["declared_event_target_roles"]
    )
    return ActionCommandRecord(**payload)


def _decode_process(value: Mapping[str, Any]) -> ProcessRecord:
    return ProcessRecord(**dict(value))


def _decode_outcome(value: Mapping[str, Any]) -> OutcomeRecord:
    return OutcomeRecord(**dict(value))


def _group_records(values: Iterable[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for value in values:
        grouped[value.episode_id].append(value)
    return grouped


def _write_row(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(canonical_json(value))
    handle.write("\n")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelDataError(f"{path}: invalid JSON object") from exc
    if not isinstance(value, dict):
        raise ModelDataError(f"{path}: root must be an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ModelDataError(f"{path}: cannot open JSONL") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ModelDataError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ModelDataError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows
