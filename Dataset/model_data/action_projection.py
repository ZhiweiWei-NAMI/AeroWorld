"""Project authored events and L0 realizations into action/process records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .action_contract import (
    ACTION_SCHEMA_NAME,
    ACTION_SCHEMA_VERSION,
    ActionCommandRecord,
    ActionProjectionDocument,
    OutcomeRecord,
    ProcessRecord,
    TargetBindingRecord,
)
from .contract import (
    ModelDataError,
    parse_episode_id,
    stable_identifier,
)


_ACTION_FAMILIES = {
    "move_entity": "motion",
    "set_runtime_state": "runtime_mutation",
    "set_visual_state": "visual_state",
    "capture_screenshot": "capture_request",
    "spawn_entity": "entity_lifecycle",
    "remove_entity": "entity_lifecycle",
    "set_weather": "environment",
    "set_pedestrian_activity": "pedestrian_activity",
}
_ENTITY_ACTION_TYPES = frozenset(
    {
        "move_entity",
        "set_runtime_state",
        "set_visual_state",
        "spawn_entity",
        "remove_entity",
        "set_pedestrian_activity",
    }
)
_PARAMETER_EXCLUSIONS = frozenset(
    {"type", "action_id", "entity_id", "ped_id"}
)
_REALIZATION_DETAIL_EXCLUSIONS = frozenset(
    {
        "action_id",
        "action_type",
        "entity_id",
        "camera_id",
        "dispatch_tick",
        "scheduled_tick",
        "status",
        "result_tick",
        "evidence_tick",
        "first_motion_tick",
        "first_capture_motion_tick",
        "terminal_tick",
        "terminal_required",
    }
)


def discover_scenario_scripts(scenario_root: Path) -> dict[str, Path]:
    """Index one authoritative event script for every scenario profile."""

    scripts: dict[str, Path] = {}
    for path in sorted(scenario_root.resolve().rglob("event_script.json")):
        value = _load_object(path)
        profile_id = _required_text(value, "scenario_id", str(path))
        previous = scripts.get(profile_id)
        if previous is not None:
            raise ModelDataError(
                f"duplicate event scripts for {profile_id}: {previous} and {path}"
            )
        scripts[profile_id] = path
    return scripts


def discover_episode_roots(episodes_root: Path) -> list[Path]:
    """Discover the complete three-seed raw episode corpus."""

    roots = sorted(
        path
        for path in episodes_root.resolve().iterdir()
        if path.is_dir()
        and (path / "event_realization.jsonl").is_file()
        and (path / "event_trace.jsonl").is_file()
        and (path / "global_entity_roster.json").is_file()
    )
    profiles: dict[str, set[str]] = {}
    for path in roots:
        profile_id, seed = parse_episode_id(path.name)
        profiles.setdefault(profile_id, set()).add(seed)
    if len(roots) != 210 or len(profiles) != 70:
        raise ModelDataError(
            f"raw action projection requires 210 episodes and 70 profiles; found "
            f"{len(roots)} episodes and {len(profiles)} profiles"
        )
    incomplete = {
        profile: sorted(seeds)
        for profile, seeds in profiles.items()
        if seeds != {"00", "01", "02"}
    }
    if incomplete:
        raise ModelDataError(f"profiles lack the exact three seeds: {incomplete}")
    return roots


def project_action_episode(
    episode_root: Path,
    *,
    event_script_path: Path,
) -> ActionProjectionDocument:
    """Project one episode without deriving future labels into command fields."""

    episode_root = episode_root.resolve()
    episode_id = episode_root.name
    profile_id, seed_label = parse_episode_id(episode_id)

    event_script = _load_object(event_script_path)
    if event_script.get("scenario_id") != profile_id:
        raise ModelDataError(f"{episode_id}: event script profile mismatch")
    trace_path = episode_root / "event_trace.jsonl"
    realization_path = episode_root / "event_realization.jsonl"
    roster_path = episode_root / "global_entity_roster.json"
    manifest_path = episode_root / "episode_manifest.json"
    trace_rows = _load_jsonl(trace_path)
    realization_rows = _load_jsonl(realization_path)
    roster = _load_object(roster_path)
    episode_manifest = _load_object(manifest_path)

    authored_events = _unique_by_id(
        event_script.get("events"), "event_id", f"{event_script_path}:events"
    )
    triggers = _unique_by_id(
        event_script.get("triggers"), "trigger_id", f"{event_script_path}:triggers"
    )
    traces = _unique_by_id(trace_rows, "chain_id", str(trace_path))
    realizations = _unique_by_id(realization_rows, "event_id", str(realization_path))
    if set(authored_events) != set(realizations) or set(authored_events) != set(traces):
        raise ModelDataError(f"{episode_id}: authored/trace/realized event ids disagree")

    action_rows: list[dict[str, Any]] = []
    seen_action_ids: set[str] = set()
    for event_index, event in enumerate(event_script.get("events") or []):
        event_id = _required_text(event, "event_id", str(event_script_path))
        realization = realizations[event_id]
        trace = traces[event_id]
        dispatch_tick = _required_tick(realization, "dispatch_tick", episode_id)
        if int(trace.get("tick", -1)) != dispatch_tick:
            raise ModelDataError(f"{episode_id}:{event_id}: trace dispatch mismatch")
        trigger_id = _required_text(event, "trigger_ref", f"{episode_id}:{event_id}")
        trigger = triggers.get(trigger_id)
        if trigger is None:
            raise ModelDataError(f"{episode_id}:{event_id}: trigger is absent")

        authored_actions = _unique_by_id(
            event.get("actions"), "action_id", f"{event_script_path}:{event_id}:actions"
        )
        realized_actions = _unique_by_id(
            realization.get("action_realizations"),
            "action_id",
            f"{realization_path}:{event_id}:action_realizations",
        )
        if set(authored_actions) != set(realized_actions):
            raise ModelDataError(f"{episode_id}:{event_id}: action ids disagree")
        overlap = seen_action_ids & set(authored_actions)
        if overlap:
            raise ModelDataError(f"{episode_id}: action ids are not episode-unique: {overlap}")
        seen_action_ids.update(authored_actions)

        target_ids = tuple(str(value) for value in realization.get("target_ids") or [])
        expected_targets = _authored_event_targets(event)
        if target_ids != expected_targets:
            raise ModelDataError(
                f"{episode_id}:{event_id}: realized targets differ from authored authority"
            )
        if not target_ids:
            raise ModelDataError(f"{episode_id}:{event_id}: event has no semantic targets")
        declared_target_ids = {
            str(value)
            for value in (event.get("log_event") or {}).get("target_ids") or []
            if str(value)
        }
        declared_roles = tuple(
            sorted(
                set(
                    str(value)
                    for value in (
                        (event.get("log_event") or {}).get("target_roles")
                        or event.get("target_roles")
                        or trace.get("target_roles")
                        or []
                    )
                    if str(value)
                )
            )
        )
        event_action_actors = {
            actor
            for authored_action in authored_actions.values()
            if (actor := _action_actor(authored_action)) is not None
        }
        primary_target = str((trace.get("scope") or {}).get("target_id") or "")

        for action_index, authored in enumerate(event.get("actions") or []):
            action_id = _required_text(authored, "action_id", f"{episode_id}:{event_id}")
            action_type = _required_text(authored, "type", f"{episode_id}:{action_id}")
            if action_type not in _ACTION_FAMILIES:
                raise ModelDataError(f"{episode_id}:{action_id}: unsupported action type")
            realized = realized_actions[action_id]
            if realized.get("action_type") != action_type:
                raise ModelDataError(f"{episode_id}:{action_id}: action type drift")
            if int(realized.get("dispatch_tick", -1)) != dispatch_tick:
                raise ModelDataError(f"{episode_id}:{action_id}: dispatch tick drift")
            actor_id = _action_actor(authored)
            realized_actor = _realized_actor(realized)
            if action_type in _ENTITY_ACTION_TYPES and realized_actor is None:
                raise ModelDataError(f"{episode_id}:{action_id}: realized actor is absent")
            if realized_actor is not None and actor_id != realized_actor:
                raise ModelDataError(f"{episode_id}:{action_id}: action actor drift")
            scope = _action_scope(action_type)
            bindings = _target_bindings(
                target_ids,
                actor_id=actor_id,
                primary_target=primary_target,
                event_action_actors=event_action_actors,
                declared_target_ids=declared_target_ids,
                roster=roster,
            )
            planned_start_tick = _required_tick(
                realized, "scheduled_tick", f"{episode_id}:{action_id}"
            )
            snapshot = (realization.get("source_truth_snapshots_by_tick") or {}).get(
                str(dispatch_tick)
            )
            if isinstance(snapshot, Mapping):
                snapshot_status = "persisted_dispatch_snapshot"
            else:
                snapshot_status = "not_persisted_at_dispatch"
            controller_id = str(trace.get("agent_id") or "").strip() or None
            parameters = {
                key: value
                for key, value in authored.items()
                if key not in _PARAMETER_EXCLUSIONS
            }
            source_refs = {
                "event_script": _portable_source_path(event_script_path),
                "event_index": event_index,
                "action_index": action_index,
                "event_realization": f"{episode_id}/event_realization.jsonl",
                "event_trace": f"{episode_id}/event_trace.jsonl",
            }
            command = ActionCommandRecord(
                episode_id=episode_id,
                profile_id=profile_id,
                event_id=event_id,
                action_id=action_id,
                action_family=_ACTION_FAMILIES[action_type],
                action_type=action_type,
                scope=scope,
                actor_id=actor_id,
                target_bindings=bindings,
                declared_event_target_roles=declared_roles,
                controller_id=controller_id,
                controller_provenance=(
                    "event_trace.agent_id" if controller_id else "not_persisted"
                ),
                trigger_id=trigger_id,
                trigger_type=_required_text(
                    trigger, "type", f"{episode_id}:{trigger_id}"
                ),
                trigger_definition={
                    key: value for key, value in trigger.items() if key != "trigger_id"
                },
                trigger_state_at_issue="satisfied",
                trigger_margin=None,
                parameters=parameters,
                issued_tick=dispatch_tick,
                planned_start_tick=planned_start_tick,
                planned_duration_ticks=(0 if action_type != "move_entity" else None),
                availability_tick=dispatch_tick,
                precondition_snapshot_status=snapshot_status,
                source_refs=source_refs,
            )
            action_rows.append(
                {
                    "sort_key": (dispatch_tick, event_index, action_index, action_id),
                    "command": command,
                    "realized": realized,
                }
            )

    action_rows.sort(key=lambda value: value["sort_key"])
    _validate_manifest_action_count(episode_manifest, len(action_rows), episode_id)
    successor_moves = _successor_moves(action_rows)
    commands: list[ActionCommandRecord] = []
    processes: list[ProcessRecord] = []
    outcomes: list[OutcomeRecord] = []
    for row in action_rows:
        command = row["command"]
        realized = row["realized"]
        process, outcome = _project_lifecycle(
            command,
            realized,
            successor_move=successor_moves.get(command.action_id),
        )
        commands.append(command)
        processes.append(process)
        outcomes.append(outcome)

    return ActionProjectionDocument(
        schema_name=ACTION_SCHEMA_NAME,
        schema_version=ACTION_SCHEMA_VERSION,
        episode_id=episode_id,
        profile_id=profile_id,
        seed_label=seed_label,
        commands=tuple(commands),
        processes=tuple(processes),
        outcomes=tuple(outcomes),
    )


def _project_lifecycle(
    command: ActionCommandRecord,
    realized: Mapping[str, Any],
    *,
    successor_move: tuple[int, str] | None,
) -> tuple[ProcessRecord, OutcomeRecord]:
    action_id = command.action_id
    raw_status = _required_text(realized, "status", action_id)
    result_tick = _required_tick(realized, "result_tick", action_id)
    evidence_tick = _required_tick(realized, "evidence_tick", action_id)
    details = {
        key: value
        for key, value in realized.items()
        if key not in _REALIZATION_DETAIL_EXCLUSIONS
    }

    if command.action_type != "move_entity":
        if command.action_type == "capture_screenshot" and raw_status == "ok":
            if realized.get("metadata_only") is not True:
                raise ModelDataError(f"{action_id}: capture result lacks metadata-only request evidence")
            terminal_status = "request_recorded"
        else:
            terminal_status = "completed" if raw_status == "ok" else "failed"
        terminal_tick = result_tick
        process = ProcessRecord(
            episode_id=command.episode_id,
            process_id=stable_identifier(
                "action_process", command.episode_id, action_id
            ),
            action_id=action_id,
            process_kind="instantaneous",
            available_tick=command.availability_tick,
            planned_start_tick=command.planned_start_tick,
            realized_start_tick=result_tick,
            start_available_tick=result_tick,
            pending_last_tick=(
                result_tick - 1 if command.availability_tick < result_tick else None
            ),
            terminal_tick=terminal_tick,
            terminal_status=terminal_status,
            status_available_tick=terminal_tick,
            active_last_tick=None,
            realized_duration_ticks=0,
            censor_code=None,
            terminal_evidence="event_realization.result_tick",
        )
    else:
        first_motion_tick = realized.get("first_motion_tick")
        if not isinstance(first_motion_tick, int):
            raise ModelDataError(f"{action_id}: movement lacks first_motion_tick")
        observed_terminal = realized.get("terminal_tick")
        if raw_status != "ok":
            terminal_status = "failed"
            terminal_tick = result_tick
            active_last_tick = max(first_motion_tick, terminal_tick)
            terminal_evidence = "event_realization.status"
        elif isinstance(observed_terminal, int):
            terminal_status = "completed"
            terminal_tick = observed_terminal
            active_last_tick = terminal_tick
            terminal_evidence = "event_realization.terminal_tick"
        else:
            if successor_move is None:
                raise ModelDataError(
                    f"{action_id}: unfinished movement lacks a superseding command"
                )
            terminal_status = "cancelled"
            terminal_tick, successor_action_id = successor_move
            active_last_tick = terminal_tick - 1
            terminal_evidence = f"superseded_by:{successor_action_id}"
            details = {**details, "superseded_by_action_id": successor_action_id}
        realized_duration = active_last_tick - first_motion_tick + 1
        if realized_duration <= 0:
            raise ModelDataError(f"{action_id}: non-positive movement duration")
        process = ProcessRecord(
            episode_id=command.episode_id,
            process_id=stable_identifier(
                "action_process", command.episode_id, action_id
            ),
            action_id=action_id,
            process_kind="durative",
            available_tick=command.availability_tick,
            planned_start_tick=command.planned_start_tick,
            realized_start_tick=first_motion_tick,
            start_available_tick=first_motion_tick,
            pending_last_tick=(
                first_motion_tick - 1
                if command.availability_tick < first_motion_tick
                else None
            ),
            terminal_tick=terminal_tick,
            terminal_status=terminal_status,
            status_available_tick=terminal_tick,
            active_last_tick=active_last_tick,
            realized_duration_ticks=realized_duration,
            censor_code=("superseded" if terminal_status == "cancelled" else None),
            terminal_evidence=terminal_evidence,
        )

    outcome_evidence_tick = max(evidence_tick, process.status_available_tick)
    if outcome_evidence_tick != evidence_tick:
        details = {**details, "initial_action_evidence_tick": evidence_tick}
    outcome = OutcomeRecord(
        episode_id=command.episode_id,
        action_id=action_id,
        status=process.terminal_status,
        available_tick=process.status_available_tick,
        terminal_tick=process.terminal_tick,
        raw_execution_status=raw_status,
        result_tick=result_tick,
        evidence_tick=outcome_evidence_tick,
        censor_code=process.censor_code,
        details=details,
    )
    return process, outcome


def _successor_moves(
    action_rows: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[int, str]]:
    by_actor: dict[str, list[ActionCommandRecord]] = {}
    for row in action_rows:
        command = row["command"]
        if command.action_type == "move_entity" and command.actor_id is not None:
            by_actor.setdefault(command.actor_id, []).append(command)
    result: dict[str, tuple[int, str]] = {}
    for commands in by_actor.values():
        commands.sort(key=lambda value: (value.issued_tick, value.action_id))
        for current, successor in zip(commands, commands[1:]):
            result[current.action_id] = (successor.issued_tick, successor.action_id)
    return result


def _target_bindings(
    target_ids: tuple[str, ...],
    *,
    actor_id: str | None,
    primary_target: str,
    event_action_actors: set[str],
    declared_target_ids: set[str],
    roster: Mapping[str, Any],
) -> tuple[TargetBindingRecord, ...]:
    bindings: list[TargetBindingRecord] = []
    for target_id in target_ids:
        roles = {"event_semantic_target"}
        sources = {"event_realization.target_ids"}
        if target_id in declared_target_ids:
            sources.add("event_script.log_event.target_ids")
        if target_id == actor_id:
            roles.add("action_actor")
        if target_id == primary_target:
            roles.add("event_primary_target")
            sources.add("event_trace.scope.target_id")
        if target_id in event_action_actors:
            roles.add("event_action_actor")
            sources.add("event_script.actions.actor")
        metadata = roster.get(target_id)
        if isinstance(metadata, Mapping):
            for prefix, value in (
                ("entity_role", metadata.get("role")),
                ("entity_class", metadata.get("label_class")),
                ("entity_category", metadata.get("category")),
                ("entity_kind", metadata.get("entity_kind")),
            ):
                text = str(value or "").strip()
                if text:
                    roles.add(f"{prefix}:{text}")
            semantic_scope = metadata.get("semantic_scope")
            if isinstance(semantic_scope, Mapping):
                for prefix, key in (
                    ("scope_type", "scope_type"),
                    ("scope_subtype", "scope_subtype"),
                    ("ontology_class", "ontology_class_id"),
                ):
                    text = str(semantic_scope.get(key) or "").strip()
                    if text:
                        roles.add(f"{prefix}:{text}")
            sources.add("global_entity_roster")
        bindings.append(
            TargetBindingRecord(
                target_id=target_id,
                roles=tuple(sorted(roles)),
                sources=tuple(sorted(sources)),
            )
        )
    return tuple(bindings)


def _authored_event_targets(event: Mapping[str, Any]) -> tuple[str, ...]:
    targets: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in targets:
            targets.append(text)

    for value in (event.get("log_event") or {}).get("target_ids") or []:
        add(value)
    for action in event.get("actions") or []:
        if isinstance(action, Mapping):
            add(action.get("entity_id") or action.get("ped_id"))
    return tuple(targets)


def _action_actor(action: Mapping[str, Any]) -> str | None:
    action_type = str(action.get("type") or "")
    if action_type in _ENTITY_ACTION_TYPES:
        return str(action.get("entity_id") or action.get("ped_id") or "").strip() or None
    if action_type == "set_weather":
        return None
    return None


def _realized_actor(action: Mapping[str, Any]) -> str | None:
    action_type = str(action.get("action_type") or "")
    if action_type in _ENTITY_ACTION_TYPES:
        return str(action.get("entity_id") or action.get("ped_id") or "").strip() or None
    return None


def _action_scope(action_type: str) -> str:
    if action_type == "set_weather":
        return "global"
    if action_type == "capture_screenshot":
        return "sensor"
    return "entity"


def _validate_manifest_action_count(
    manifest: Mapping[str, Any], actual: int, episode_id: str
) -> None:
    counts = manifest.get("record_counts")
    if isinstance(counts, Mapping) and "event_action_realizations" in counts:
        if int(counts["event_action_realizations"]) != actual:
            raise ModelDataError(f"{episode_id}: manifest action count mismatch")


def _unique_by_id(
    values: Any,
    key: str,
    context: str,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(values, list):
        raise ModelDataError(f"{context}: expected a list")
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ModelDataError(f"{context}[{index}]: expected an object")
        identifier = _required_text(value, key, f"{context}[{index}]")
        if identifier in result:
            raise ModelDataError(f"{context}: duplicate {key} {identifier!r}")
        result[identifier] = value
    return result


def _required_text(value: Mapping[str, Any], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ModelDataError(f"{context}: {key} must be a non-empty string")
    return item


def _required_tick(value: Mapping[str, Any], key: str, context: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= 900:
        raise ModelDataError(f"{context}: {key} must be an integer in 0..900")
    return item


def _portable_source_path(path: Path) -> str:
    parts = path.resolve().parts
    try:
        index = parts.index("Dataset")
    except ValueError:
        return path.name
    return Path(*parts[index:]).as_posix()


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
