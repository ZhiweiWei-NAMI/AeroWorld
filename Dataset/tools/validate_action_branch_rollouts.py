#!/usr/bin/env python3
"""Independently validate deterministic action-branch rollout artifacts.

The rollout command writes evidence; this command is a separate read-only
acceptance boundary.  It never trusts a status string in a candidate document
or in ``canary_report.json``.  Every digest, trajectory grid, ledger relation,
objective target and branch-specific image is re-read from disk and checked
against the authored source and the executable candidate design.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = PROJECT_ROOT / "Dataset" / "tools"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_action_branch_rollouts import (  # noqa: E402
    BRANCH_SCHEMA_NAME,
    FORMAL_DURATION_TICKS,
    BranchRolloutError,
    _compare_target,
    _objective_digest,
    _profile_arm,
    _profile_design,
    _read_json,
    _read_jsonl,
    _sha256_file,
    _target_summary,
    _write_json,
    canonical_json,
)


RAW_REQUIRED_FILES = (
    "branch_manifest.json",
    "branch_event_ledger.jsonl",
    "event_script.json",
    "event_trace.jsonl",
    "event_realization.jsonl",
    "global_entity_roster.json",
    "trajectories.jsonl",
    "weather_meta.jsonl",
)
OBJECTIVE_REQUIRED_FILES = (
    "manifest.json",
    "epi_closure.json",
    "predicate_truth.jsonl",
    "predicate_transitions.jsonl",
    "event_occurrences.jsonl",
    "event_outcomes.jsonl",
    "semantic_graph_base.json",
    "semantic_graph_deltas.jsonl",
    "semantic_truth_graphs.jsonl",
    "world_truth_graph_base.json",
    "world_truth_graph_deltas.jsonl",
    "world_truth_summary.json",
)


class ValidationError(RuntimeError):
    """A release artifact cannot be accepted."""


def _finding(
    findings: list[dict[str, Any]],
    *,
    code: str,
    path: str,
    message: str,
    severity: str = "P1",
) -> None:
    findings.append(
        {
            "severity": severity,
            "code": code,
            "path": path,
            "message": message,
        }
    )


def _safe_relative(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    candidate = Path(value)
    if any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved


def _check_regular_files(
    root: Path,
    names: Sequence[str],
    findings: list[dict[str, Any]],
    section: str,
) -> bool:
    valid = True
    for name in names:
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            _finding(
                findings,
                code="missing_or_empty_artifact",
                path=str(path.relative_to(PROJECT_ROOT)),
                message=f"{section} artifact must be a non-empty regular file",
            )
            valid = False
    return valid


def _finite_vector(value: Any, width: int = 3) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < width:
        return False
    try:
        return all(math.isfinite(float(value[index])) for index in range(width))
    except (TypeError, ValueError):
        return False


def _validate_trajectory_grid(path: Path, findings: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _read_jsonl(path)
    by_entity: dict[str, set[int]] = defaultdict(set)
    duplicate_count = 0
    invalid_pose_count = 0
    for row in rows:
        entity_id = str(row.get("entity_id") or "")
        try:
            tick = int(row["tick"])
        except (KeyError, TypeError, ValueError):
            _finding(
                findings,
                code="invalid_trajectory_tick",
                path=str(path.relative_to(PROJECT_ROOT)),
                message=f"trajectory row has no integer tick: {row!r}",
            )
            continue
        if not entity_id or tick < 0 or tick > FORMAL_DURATION_TICKS:
            _finding(
                findings,
                code="trajectory_grid_out_of_range",
                path=str(path.relative_to(PROJECT_ROOT)),
                message=f"trajectory key outside 0..{FORMAL_DURATION_TICKS}: {tick}/{entity_id!r}",
            )
        if tick in by_entity[entity_id]:
            duplicate_count += 1
        by_entity[entity_id].add(tick)
        if not _finite_vector(row.get("pos_enu")) or not _finite_vector(row.get("vel_mps")):
            invalid_pose_count += 1
    if duplicate_count:
        _finding(
            findings,
            code="duplicate_trajectory_key",
            path=str(path.relative_to(PROJECT_ROOT)),
            message=f"{duplicate_count} duplicate (tick, entity_id) keys",
        )
    if invalid_pose_count:
        _finding(
            findings,
            code="nonfinite_trajectory_pose",
            path=str(path.relative_to(PROJECT_ROOT)),
            message=f"{invalid_pose_count} rows lack finite 3D pose/velocity",
        )
    incomplete = {
        entity: sorted(ticks.symmetric_difference(set(range(FORMAL_DURATION_TICKS + 1))))
        for entity, ticks in by_entity.items()
        if ticks != set(range(FORMAL_DURATION_TICKS + 1))
    }
    if incomplete:
        _finding(
            findings,
            code="incomplete_trajectory_grid",
            path=str(path.relative_to(PROJECT_ROOT)),
            message=f"entities do not cover every engine tick: {list(incomplete)[:5]}",
        )
    return {
        "row_count": len(rows),
        "entity_count": len(by_entity),
        "duplicate_key_count": duplicate_count,
        "invalid_pose_count": invalid_pose_count,
        "entity_tick_counts": {key: len(value) for key, value in sorted(by_entity.items())},
    }


def _event_id(row: Mapping[str, Any]) -> str:
    payload = row.get("payload")
    if isinstance(payload, Mapping):
        return str(
            row.get("event_id")
            or payload.get("event_id")
            or row.get("source_event_id")
            or ""
        )
    return str(row.get("event_id") or row.get("source_event_id") or "")


def _event_tick(row: Mapping[str, Any]) -> int | None:
    for key in ("activated_tick", "tick", "dispatch_tick", "evidence_tick"):
        value = row.get(key)
        if value is None and isinstance(row.get("payload"), Mapping):
            value = row["payload"].get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _validate_ledger(
    raw_root: Path,
    *,
    source_script: Mapping[str, Any],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = _read_jsonl(raw_root / "branch_event_ledger.jsonl")
    if len(rows) != 1:
        _finding(
            findings,
            code="ledger_cardinality",
            path=str((raw_root / "branch_event_ledger.jsonl").relative_to(PROJECT_ROOT)),
            message=f"expected exactly one branch ledger row, got {len(rows)}",
        )
        return {}
    ledger = rows[0]
    if ledger.get("status") != "applied":
        _finding(
            findings,
            code="ledger_not_applied",
            path=str((raw_root / "branch_event_ledger.jsonl").relative_to(PROJECT_ROOT)),
            message=f"ledger status is {ledger.get('status')!r}",
        )
    authored_ids = {
        str(event.get("event_id"))
        for event in source_script.get("events") or []
        if isinstance(event, Mapping) and event.get("event_id")
    }
    suppressed = [str(value) for value in ledger.get("suppressed_event_ids") or []]
    unknown = sorted(set(suppressed) - authored_ids)
    if unknown:
        _finding(
            findings,
            code="ledger_unknown_suppression",
            path=str((raw_root / "branch_event_ledger.jsonl").relative_to(PROJECT_ROOT)),
            message=f"suppressed event ids are not authored: {unknown}",
        )
    trace_rows = _read_jsonl(raw_root / "event_trace.jsonl")
    trace_ids = {_event_id(row) for row in trace_rows if _event_id(row)}
    leaked = sorted(trace_ids.intersection(suppressed))
    if leaked:
        _finding(
            findings,
            code="suppressed_event_fired",
            path=str((raw_root / "event_trace.jsonl").relative_to(PROJECT_ROOT)),
            message=f"suppressed events appear in branch trace: {leaked}",
        )
    derived_script = _read_json(raw_root / "event_script.json")
    trigger_by_event = {
        str(event.get("event_id")): str(event.get("trigger_ref") or "")
        for event in derived_script.get("events") or []
        if isinstance(event, Mapping) and event.get("event_id")
    }
    trigger_by_id = {
        str(trigger.get("trigger_id")): trigger
        for trigger in derived_script.get("triggers") or []
        if isinstance(trigger, Mapping) and trigger.get("trigger_id")
    }
    bad_trigger_ids: list[str] = []
    for event_id in suppressed:
        trigger_id = trigger_by_event.get(event_id)
        trigger = trigger_by_id.get(trigger_id or "")
        if not isinstance(trigger, Mapping) or int(trigger.get("tick", -1)) != FORMAL_DURATION_TICKS + 1:
            bad_trigger_ids.append(event_id)
    if bad_trigger_ids:
        _finding(
            findings,
            code="suppression_not_materialized",
            path=str((raw_root / "event_script.json").relative_to(PROJECT_ROOT)),
            message=f"suppressed events do not have impossible tick {FORMAL_DURATION_TICKS + 1}: {bad_trigger_ids}",
        )
    return {
        "fork_tick": ledger.get("fork_tick"),
        "suppressed_event_ids": suppressed,
        "trace_event_count": len(trace_rows),
        "trace_event_ids": sorted(trace_ids),
    }


def _validate_branch(
    *,
    output_root: Path,
    record: Mapping[str, Any],
    design: Mapping[str, Any],
    arm_record: Mapping[str, Any],
    source_script: Mapping[str, Any],
    candidate_design_path: Path,
    source_script_path: Path,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    branch_rel = _safe_relative(output_root, record.get("branch_root"))
    if branch_rel is None:
        _finding(
            findings,
            code="unsafe_branch_root",
            path="canary_report.json.records.branch_root",
            message=f"branch_root must be a safe relative path: {record.get('branch_root')!r}",
            severity="P0",
        )
        return {"arm_id": record.get("arm_id"), "status": "FAIL"}
    raw_root = branch_rel / "raw"
    objective_root = branch_rel / "objective"
    before_count = len(findings)
    _check_regular_files(raw_root, RAW_REQUIRED_FILES, findings, "raw")
    _check_regular_files(objective_root, OBJECTIVE_REQUIRED_FILES, findings, "objective")
    plot_path = branch_rel / "branch_plot.png"
    if not plot_path.is_file() or plot_path.is_symlink() or plot_path.stat().st_size == 0:
        _finding(
            findings,
            code="missing_branch_plot",
            path=str(plot_path.relative_to(PROJECT_ROOT)),
            message="each executed arm requires a non-empty branch-specific PNG",
        )
    elif plot_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        _finding(
            findings,
            code="invalid_branch_plot",
            path=str(plot_path.relative_to(PROJECT_ROOT)),
            message="branch_plot.png is not a PNG file",
        )
    if not (raw_root / "branch_manifest.json").is_file() or not (objective_root / "manifest.json").is_file():
        return {"arm_id": record.get("arm_id"), "status": "FAIL"}

    raw_manifest = _read_json(raw_root / "branch_manifest.json")
    if raw_manifest.get("schema_name") != BRANCH_SCHEMA_NAME:
        _finding(
            findings,
            code="raw_manifest_schema",
            path=str((raw_root / "branch_manifest.json").relative_to(PROJECT_ROOT)),
            message=f"unexpected branch schema {raw_manifest.get('schema_name')!r}",
        )
    arm = arm_record.get("decision_arm") or {}
    expected_arm_id = str(arm.get("arm_id") or "")
    if raw_manifest.get("arm_id") != expected_arm_id or raw_manifest.get("arm_kind") != arm.get("arm_kind"):
        _finding(
            findings,
            code="raw_manifest_identity",
            path=str((raw_root / "branch_manifest.json").relative_to(PROJECT_ROOT)),
            message="branch manifest arm identity does not match the candidate arm",
        )
    source_authority = raw_manifest.get("source_authority") or {}
    trajectory_stats = _validate_trajectory_grid(raw_root / "trajectories.jsonl", findings)
    ledger_stats = _validate_ledger(raw_root, source_script=source_script, findings=findings)
    objective_manifest = _read_json(objective_root / "manifest.json")
    closure = _read_json(objective_root / "epi_closure.json")
    if objective_manifest.get("closure_status") != closure.get("status"):
        _finding(
            findings,
            code="objective_closure_disagreement",
            path=str((objective_root / "manifest.json").relative_to(PROJECT_ROOT)),
            message="objective manifest and epi_closure disagree",
        )
    try:
        expected_target = _target_summary(
            objective_root=objective_root,
            design=design,
            episode_id=str(record.get("episode_id")),
            fork_tick=int((design.get("window_policy") or {}).get("fork_tick") or record.get("fork_tick") or 0),
            end_tick=int((design.get("window_policy") or {}).get("end_tick") or FORMAL_DURATION_TICKS),
        )
    except (BranchRolloutError, OSError, ValueError, KeyError) as exc:
        _finding(
            findings,
            code="target_recompute_failed",
            path=str(objective_root.relative_to(PROJECT_ROOT)),
            message=str(exc),
        )
        expected_target = {}
    declared_target = record.get("target")
    if expected_target and canonical_json(expected_target) != canonical_json(declared_target or {}):
        _finding(
            findings,
            code="target_summary_stale",
            path="canary_report.json.records.target",
            message="reported target summary differs from independently recomputed objective rows",
        )
    arm = arm_record.get("decision_arm") or {}
    measured_objective_digest = _objective_digest(objective_root)
    measured_plot_digest = _sha256_file(plot_path) if plot_path.is_file() else None
    return {
        "arm_id": record.get("arm_id"),
        "arm_kind": arm.get("arm_kind"),
        "branch_root": str(branch_rel.relative_to(PROJECT_ROOT)),
        "formal_objective_closure_status": closure.get("status"),
        "trajectory": trajectory_stats,
        "ledger": ledger_stats,
        "target": expected_target,
        "objective_digest": measured_objective_digest,
        "plot_sha256": measured_plot_digest,
        "artifact_status": "PASS" if len(findings) == before_count else "FAIL",
    }


def validate_rollout(
    *,
    rollout_root: Path,
    candidate_root: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    report_path = rollout_root / "canary_report.json"
    if not report_path.is_file():
        raise ValidationError(f"canary report is missing: {report_path}")
    report = _read_json(report_path)
    scope = report.get("scope") or {}
    profile_id = str(scope.get("profile_id") or "")
    episode_id = str(scope.get("episode_id") or "")
    if not profile_id or not episode_id:
        raise ValidationError("canary report scope lacks profile_id/episode_id")
    design = _profile_design(candidate_root, profile_id)
    scenario_id = episode_id.rsplit("__seed", 1)[0]
    source_script_path = next(
        iter(
            sorted(
                path
                for path in (PROJECT_ROOT / "Dataset" / "scenarios").rglob("event_script.json")
                if str((_read_json(path)).get("scenario_id") or path.parent.name) == scenario_id
            )
        ),
        None,
    )
    if source_script_path is None:
        raise ValidationError(f"authored event script is missing for {scenario_id}")
    source_script = _read_json(source_script_path)
    findings: list[dict[str, Any]] = []
    records = report.get("records")
    requested_ids = scope.get("arm_ids_requested")
    if not isinstance(requested_ids, list) or not requested_ids:
        _finding(
            findings,
            code="missing_requested_arm_set",
            path="canary_report.json.scope.arm_ids_requested",
            message="the canary must declare a non-empty explicit arm selection",
            severity="P0",
        )
        requested_ids = []
    elif any(not isinstance(value, str) or not value for value in requested_ids):
        _finding(
            findings,
            code="malformed_requested_arm_set",
            path="canary_report.json.scope.arm_ids_requested",
            message="every requested arm id must be a non-empty string",
            severity="P0",
        )
    requested_string_ids = [value for value in requested_ids if isinstance(value, str)]
    if len(set(requested_string_ids)) != len(requested_string_ids):
        _finding(
            findings,
            code="duplicate_requested_arm",
            path="canary_report.json.scope.arm_ids_requested",
            message="requested arm ids must be unique",
            severity="P0",
        )
    if not isinstance(records, list) or not records:
        _finding(findings, code="empty_rollout_report", path="canary_report.json.records", message="report has no executed arm records", severity="P0")
        records = []
    record_ids = [
        str(record.get("arm_id") or "")
        for record in records
        if isinstance(record, Mapping)
    ]
    requested_for_compare = (
        requested_string_ids if len(requested_string_ids) == len(requested_ids) else []
    )
    if sorted(record_ids) != sorted(requested_for_compare):
        _finding(
            findings,
            code="requested_execution_set_mismatch",
            path="canary_report.json.records",
            message=(
                "record arm ids must exactly equal the explicitly requested set; "
                f"requested={sorted(requested_for_compare)!r}, records={sorted(record_ids)!r}"
            ),
            severity="P0",
        )
    validated: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            _finding(findings, code="malformed_record", path="canary_report.json.records", message=f"record is not an object: {record!r}")
            continue
        arm_id = str(record.get("arm_id") or "")
        try:
            arm_record = _profile_arm(candidate_root, profile_id, arm_id)
        except BranchRolloutError as exc:
            _finding(findings, code="arm_not_in_candidate", path="canary_report.json.records.arm_id", message=str(exc), severity="P0")
            continue
        if record.get("status") != "EXECUTED":
            _finding(findings, code="arm_not_executed", path=f"canary_report.json.records[{arm_id}]", message=f"status is {record.get('status')!r}")
            continue
        validated.append(
            _validate_branch(
                output_root=rollout_root,
                record=record,
                design=design,
                arm_record=arm_record,
                source_script=source_script,
                candidate_design_path=candidate_root / profile_id / "normalized_design.json",
                source_script_path=source_script_path,
                findings=findings,
            )
        )

    factual = next((row for row in validated if row.get("arm_kind") == "factual"), None)
    alternatives: list[dict[str, Any]] = []
    goal_mode = str((design.get("semantic_goal") or {}).get("event_change_mode") or "")
    for row in validated:
        if row.get("arm_kind") == "factual":
            continue
        comparison = _compare_target(
            factual.get("target", {}) if factual else {},
            row.get("target", {}),
            goal_mode,
            str(row.get("arm_kind") or ""),
        )
        row["comparison"] = comparison
        alternatives.append(row)
        if comparison.get("status") != "PASS":
            _finding(
                findings,
                code="target_comparison_failed",
                path=f"branches/{profile_id}/{episode_id}/{row.get('arm_id')}",
                message=str(comparison.get("contract_issue") or comparison.get("classification") or "target comparison failed"),
            )
    if factual is None:
        _finding(findings, code="missing_factual_comparator", path="canary_report.json.records", message="exactly one executed factual comparator is required", severity="P0")
    if not alternatives:
        _finding(findings, code="missing_alternative", path="canary_report.json.records", message="at least one executed alternative is required", severity="P0")
    if sum(row.get("arm_kind") == "factual" for row in validated) != 1:
        _finding(findings, code="factual_cardinality", path="canary_report.json.records", message="factual comparator cardinality is not exactly one", severity="P0")
    expected_counts = {
        "requested": len(requested_ids),
        "executed": sum(
            isinstance(record, Mapping) and record.get("status") == "EXECUTED"
            for record in records
        ),
        "rejected_fail_closed": sum(
            isinstance(record, Mapping)
            and record.get("status") == "REJECTED_FAIL_CLOSED"
            for record in records
        ),
        "passing_alternatives": sum(
            row.get("comparison", {}).get("status") == "PASS" for row in alternatives
        ),
    }
    declared_counts = report.get("counts")
    if not isinstance(declared_counts, Mapping) or any(
        declared_counts.get(key) != value for key, value in expected_counts.items()
    ):
        _finding(
            findings,
            code="report_counts_stale",
            path="canary_report.json.counts",
            message=f"declared counts {declared_counts!r} != measured counts {expected_counts!r}",
        )
    measured_status = "PASS" if not findings else "NO-GO"
    declared_status = str(report.get("status") or "")
    if declared_status != measured_status:
        _finding(
            findings,
            code="report_status_stale",
            path="canary_report.json.status",
            message=f"declared report status {declared_status!r} != independently measured {measured_status!r}",
        )
    if report.get("frontend_release_status") != measured_status:
        _finding(
            findings,
            code="frontend_status_stale",
            path="canary_report.json.frontend_release_status",
            message=(
                f"declared frontend status {report.get('frontend_release_status')!r} "
                f"!= measured {measured_status!r}"
            ),
        )
    formal_pass = bool(validated) and all(
        row.get("formal_objective_closure_status") == "PASS" for row in validated
    )
    expected_training_status = "PASS" if measured_status == "PASS" and formal_pass else "NO-GO"
    if report.get("training_release_status") != expected_training_status:
        _finding(
            findings,
            code="training_status_stale",
            path="canary_report.json.training_release_status",
            message=(
                f"declared training status {report.get('training_release_status')!r} "
                f"!= measured {expected_training_status!r}"
            ),
        )
    # Declaration mismatches are findings too; recompute the final release
    # status after all checks so a stale PASS can never survive validation.
    status = "PASS" if not findings else "NO-GO"
    if status != "PASS":
        expected_training_status = "NO-GO"
    result = {
        "schema_name": "aeroworld_action_branch_validation_report",
        "schema_version": "1.0.0",
        "validator_version": "action_branch_validator_v1",
        "status": status,
        "frontend_release_status": status,
        "training_release_status": expected_training_status,
        "scope": {
            "profile_id": profile_id,
            "episode_id": episode_id,
            "candidate_root": str(candidate_root.relative_to(PROJECT_ROOT)),
            "rollout_root": str(rollout_root.relative_to(PROJECT_ROOT)),
            "source_event_script": str(source_script_path.relative_to(PROJECT_ROOT)),
        },
        "measured_authority": {
            "candidate_design_sha256": _sha256_file(candidate_root / profile_id / "normalized_design.json"),
            "event_script_sha256": _sha256_file(source_script_path),
            "input_report_sha256": _sha256_file(report_path),
        },
        "records": validated,
        "findings": findings,
        "counts": {
            "records": len(validated),
            "alternatives": len(alternatives),
            "artifact_pass": sum(row.get("artifact_status") == "PASS" for row in validated),
            "findings": len(findings),
        },
    }
    if output_path is None:
        output_path = rollout_root / "validation_report.json"
    _write_json(output_path, result)
    if status == "PASS":
        catalog = {
            "schema_name": "aeroworld_action_branch_frontend_catalog",
            "schema_version": "1.0.0",
            "release_status": "PASS",
            "training_release_status": result["training_release_status"],
            "source_validation_report": str(output_path.relative_to(PROJECT_ROOT)),
            "profile_id": profile_id,
            "episode_id": episode_id,
            "window": {
                "fork_tick": int((design.get("window_policy") or {}).get("fork_tick") or 0),
                "end_tick": int((design.get("window_policy") or {}).get("end_tick") or FORMAL_DURATION_TICKS),
            },
            "branches": [
                {
                    "arm_id": row.get("arm_id"),
                    "arm_kind": row.get("arm_kind"),
                    "branch_root": row.get("branch_root"),
                    "formal_objective_closure_status": row.get("formal_objective_closure_status"),
                    "review_only": row.get("formal_objective_closure_status") != "PASS",
                    "comparison": row.get("comparison"),
                    "target": row.get("target"),
                    "plot_path": f"{row.get('branch_root')}/branch_plot.png",
                    "plot_sha256": row.get("plot_sha256"),
                    "objective_digest": row.get("objective_digest"),
                }
                for row in validated
            ],
        }
        _write_json(rollout_root / "frontend_catalog.json", catalog)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/branches/aero_branch_candidate_review_20260902" / "generated" / "final_branches",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = validate_rollout(
        rollout_root=args.rollout_root.resolve(),
        candidate_root=args.candidate_root.resolve(),
        output_path=args.output.resolve() if args.output else None,
    )
    print(canonical_json(result))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, BranchRolloutError, OSError, ValueError) as exc:
        print(f"action branch validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
