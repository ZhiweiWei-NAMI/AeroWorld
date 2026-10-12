"""Split-neutral action/process contract for the AERO-210 L0 projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .contract import ModelDataError, canonical_json


ACTION_SCHEMA_NAME = "aero_action_process_projection"
ACTION_SCHEMA_VERSION = "1.3.0"
EPISODE_END_TICK = 900
ACTION_SCOPES = frozenset({"entity", "sensor", "global"})
PROCESS_KINDS = frozenset({"instantaneous", "durative"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "censored", "request_recorded"})


@dataclass(frozen=True)
class TargetBindingRecord:
    target_id: str
    roles: tuple[str, ...]
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.target_id, "target_id")
        if not self.roles or any(not value for value in self.roles):
            raise ModelDataError(f"{self.target_id}: target binding lacks typed roles")
        if tuple(sorted(set(self.roles))) != self.roles:
            raise ModelDataError(f"{self.target_id}: target roles are not canonical")
        if not self.sources or tuple(sorted(set(self.sources))) != self.sources:
            raise ModelDataError(f"{self.target_id}: target sources are not canonical")


@dataclass(frozen=True)
class ActionCommandRecord:
    episode_id: str
    profile_id: str
    event_id: str
    action_id: str
    action_family: str
    action_type: str
    scope: str
    actor_id: str | None
    target_bindings: tuple[TargetBindingRecord, ...]
    declared_event_target_roles: tuple[str, ...]
    controller_id: str | None
    controller_provenance: str
    trigger_id: str
    trigger_type: str
    trigger_definition: Mapping[str, Any]
    trigger_state_at_issue: str
    trigger_margin: float | None
    parameters: Mapping[str, Any]
    issued_tick: int
    planned_start_tick: int
    planned_duration_ticks: int | None
    availability_tick: int
    precondition_snapshot_status: str
    source_refs: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "profile_id",
            "event_id",
            "action_id",
            "action_family",
            "action_type",
            "controller_provenance",
            "trigger_id",
            "trigger_type",
            "trigger_state_at_issue",
            "precondition_snapshot_status",
        ):
            _require_text(getattr(self, name), name)
        if self.scope not in ACTION_SCOPES:
            raise ModelDataError(f"{self.action_id}: invalid action scope {self.scope!r}")
        if self.scope == "entity" and not self.actor_id:
            raise ModelDataError(f"{self.action_id}: entity command lacks an entity reference")
        if self.action_type == "capture_screenshot" and (
                self.action_family != "capture_request" or self.scope != "sensor" or
                self.actor_id is not None or
                not isinstance(self.parameters.get("camera_id"), str) or
                not self.parameters["camera_id"]):
            raise ModelDataError(f"{self.action_id}: capture request lacks a camera configuration reference")
        if not self.target_bindings:
            raise ModelDataError(f"{self.action_id}: action lacks semantic targets")
        target_ids = tuple(binding.target_id for binding in self.target_bindings)
        if len(target_ids) != len(set(target_ids)):
            raise ModelDataError(f"{self.action_id}: duplicate semantic target")
        if self.declared_event_target_roles != tuple(
            sorted(set(self.declared_event_target_roles))
        ):
            raise ModelDataError(f"{self.action_id}: declared target roles are not canonical")
        _validate_tick(self.issued_tick, "issued_tick", self.action_id)
        _validate_tick(self.planned_start_tick, "planned_start_tick", self.action_id)
        _validate_tick(self.availability_tick, "availability_tick", self.action_id)
        if self.availability_tick != self.issued_tick:
            raise ModelDataError(f"{self.action_id}: command must be visible at issue")
        if self.planned_start_tick < self.issued_tick:
            raise ModelDataError(f"{self.action_id}: planned start precedes issue")
        if self.planned_duration_ticks is not None and self.planned_duration_ticks < 0:
            raise ModelDataError(f"{self.action_id}: planned duration is negative")


@dataclass(frozen=True)
class ProcessRecord:
    episode_id: str
    process_id: str
    action_id: str
    process_kind: str
    available_tick: int
    planned_start_tick: int
    realized_start_tick: int
    start_available_tick: int
    pending_last_tick: int | None
    terminal_tick: int
    terminal_status: str
    status_available_tick: int
    active_last_tick: int | None
    realized_duration_ticks: int
    censor_code: str | None
    terminal_evidence: str

    def __post_init__(self) -> None:
        for name in ("episode_id", "process_id", "action_id", "terminal_evidence"):
            _require_text(getattr(self, name), name)
        if self.process_kind not in PROCESS_KINDS:
            raise ModelDataError(f"{self.action_id}: invalid process kind")
        if self.terminal_status not in TERMINAL_STATUSES:
            raise ModelDataError(f"{self.action_id}: invalid terminal status")
        for name in (
            "available_tick",
            "planned_start_tick",
            "realized_start_tick",
            "start_available_tick",
            "terminal_tick",
            "status_available_tick",
        ):
            _validate_tick(getattr(self, name), name, self.action_id)
        if self.start_available_tick < self.realized_start_tick:
            raise ModelDataError(f"{self.action_id}: start is visible before it occurs")
        expected_pending_last = (
            self.realized_start_tick - 1
            if self.available_tick < self.realized_start_tick
            else None
        )
        if self.pending_last_tick != expected_pending_last:
            raise ModelDataError(f"{self.action_id}: invalid pending interval")
        if self.status_available_tick < self.terminal_tick:
            raise ModelDataError(f"{self.action_id}: terminal status is visible too early")
        if self.realized_duration_ticks < 0:
            raise ModelDataError(f"{self.action_id}: realized duration is negative")
        if self.process_kind == "instantaneous":
            if self.realized_duration_ticks != 0 or self.active_last_tick is not None:
                raise ModelDataError(f"{self.action_id}: instantaneous process has duration")
        else:
            if self.active_last_tick is None:
                raise ModelDataError(f"{self.action_id}: durative process lacks active end")
            expected = self.active_last_tick - self.realized_start_tick + 1
            if expected != self.realized_duration_ticks or expected <= 0:
                raise ModelDataError(f"{self.action_id}: invalid realized active interval")


@dataclass(frozen=True)
class OutcomeRecord:
    episode_id: str
    action_id: str
    status: str
    available_tick: int
    terminal_tick: int
    raw_execution_status: str
    result_tick: int
    evidence_tick: int
    censor_code: str | None
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text(self.episode_id, "episode_id")
        _require_text(self.action_id, "action_id")
        _require_text(self.raw_execution_status, "raw_execution_status")
        if self.status not in TERMINAL_STATUSES:
            raise ModelDataError(f"{self.action_id}: invalid outcome status")
        for name in ("available_tick", "terminal_tick", "result_tick", "evidence_tick"):
            _validate_tick(getattr(self, name), name, self.action_id)
        if self.available_tick < self.terminal_tick:
            raise ModelDataError(f"{self.action_id}: outcome is visible before terminal")


@dataclass(frozen=True)
class ActionProjectionDocument:
    schema_name: str
    schema_version: str
    episode_id: str
    profile_id: str
    seed_label: str
    commands: tuple[ActionCommandRecord, ...]
    processes: tuple[ProcessRecord, ...]
    outcomes: tuple[OutcomeRecord, ...]

    def __post_init__(self) -> None:
        if (self.schema_name, self.schema_version) != (
            ACTION_SCHEMA_NAME,
            ACTION_SCHEMA_VERSION,
        ):
            raise ModelDataError(f"{self.episode_id}: action schema mismatch")
        action_ids = tuple(command.action_id for command in self.commands)
        if len(action_ids) != len(set(action_ids)):
            raise ModelDataError(f"{self.episode_id}: duplicate action id")
        if {process.action_id for process in self.processes} != set(action_ids):
            raise ModelDataError(f"{self.episode_id}: action/process linkage is incomplete")
        if {outcome.action_id for outcome in self.outcomes} != set(action_ids):
            raise ModelDataError(f"{self.episode_id}: action/outcome linkage is incomplete")
        if len(self.processes) != len(action_ids) or len(self.outcomes) != len(action_ids):
            raise ModelDataError(f"{self.episode_id}: duplicate action process or outcome")
        processes = {process.action_id: process for process in self.processes}
        outcomes = {outcome.action_id: outcome for outcome in self.outcomes}
        for command in self.commands:
            process = processes[command.action_id]
            outcome = outcomes[command.action_id]
            if any(record.episode_id != self.episode_id for record in (command, process, outcome)):
                raise ModelDataError(f"{command.action_id}: action record episode mismatch")
            if command.profile_id != self.profile_id:
                raise ModelDataError(f"{command.action_id}: action profile mismatch")
            if process.terminal_status != outcome.status:
                raise ModelDataError(f"{command.action_id}: process and outcome status differ")
            request_recorded = process.terminal_status == "request_recorded"
            capture_ok = (
                command.action_type == "capture_screenshot"
                and outcome.raw_execution_status == "ok"
            )
            if request_recorded != capture_ok:
                raise ModelDataError(
                    f"{command.action_id}: capture request status contradicts action and realization"
                )


def record_dict(record: Any) -> dict[str, Any]:
    """Return a JSON-compatible dataclass record."""

    value = asdict(record)
    if not isinstance(value, dict):
        raise TypeError(type(record).__name__)
    return value


def command_payload(record: ActionCommandRecord) -> dict[str, Any]:
    """Return only dispatch-visible command fields used by model inputs."""

    return {
        "episode_id": record.episode_id,
        "event_id": record.event_id,
        "action_id": record.action_id,
        "action_family": record.action_family,
        "action_type": record.action_type,
        "scope": record.scope,
        "actor_id": record.actor_id,
        "target_bindings": [record_dict(value) for value in record.target_bindings],
        "declared_event_target_roles": list(record.declared_event_target_roles),
        "controller_id": record.controller_id,
        "controller_provenance": record.controller_provenance,
        "trigger_id": record.trigger_id,
        "trigger_type": record.trigger_type,
        "trigger_definition": record.trigger_definition,
        "trigger_state_at_issue": record.trigger_state_at_issue,
        "trigger_margin": record.trigger_margin,
        "parameters": record.parameters,
        "issued_tick": record.issued_tick,
        "planned_start_tick": record.planned_start_tick,
        "planned_duration_ticks": record.planned_duration_ticks,
        "availability_tick": record.availability_tick,
        "precondition_snapshot_status": record.precondition_snapshot_status,
    }


def serialize_origin_model_input(
    document: ActionProjectionDocument,
    origin_tick: int,
) -> bytes:
    """Serialize the action/process view known no later than ``origin_tick``.

    Terminal labels remain absent until their own availability ticks. Changing a
    future terminal or outcome therefore cannot change earlier model-input bytes.
    """

    _validate_tick(origin_tick, "origin_tick", document.episode_id)
    commands = [
        command
        for command in document.commands
        if command.availability_tick <= origin_tick
    ]
    process_by_action = {process.action_id: process for process in document.processes}
    outcome_by_action = {outcome.action_id: outcome for outcome in document.outcomes}
    rows: list[dict[str, Any]] = []
    for command in commands:
        process = process_by_action[command.action_id]
        process_view: dict[str, Any] = {
            "process_id": process.process_id,
            "action_id": process.action_id,
            "process_kind": process.process_kind,
            "planned_start_tick": process.planned_start_tick,
        }
        if process.status_available_tick <= origin_tick:
            phase = process.terminal_status
            process_view.update(
                {
                    "phase": phase,
                    "realized_start_tick": process.realized_start_tick,
                    "terminal_tick": process.terminal_tick,
                    "terminal_status": process.terminal_status,
                    "realized_duration_ticks": process.realized_duration_ticks,
                    "censor_code": process.censor_code,
                }
            )
        elif process.start_available_tick <= origin_tick:
            phase = "active" if process.process_kind == "durative" else "executing"
            process_view.update(
                {
                    "phase": phase,
                    "realized_start_tick": process.realized_start_tick,
                    "elapsed_ticks": origin_tick - process.realized_start_tick + 1,
                }
            )
        else:
            process_view["phase"] = "pending"
        row: dict[str, Any] = {
            "command": command_payload(command),
            "process": process_view,
        }
        outcome = outcome_by_action[command.action_id]
        if max(outcome.available_tick, outcome.terminal_tick, outcome.result_tick,
               outcome.evidence_tick) <= origin_tick:
            row["outcome"] = record_dict(outcome)
        rows.append(row)
    payload = {
        "schema_name": ACTION_SCHEMA_NAME,
        "schema_version": ACTION_SCHEMA_VERSION,
        "episode_id": document.episode_id,
        "origin_tick": origin_tick,
        "actions": rows,
    }
    return canonical_json(payload).encode("utf-8")


def _require_text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ModelDataError(f"{field} must be a non-empty string")


def _validate_tick(value: Any, field: str, context: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= EPISODE_END_TICK:
        raise ModelDataError(f"{context}: {field} is outside 0..{EPISODE_END_TICK}")
