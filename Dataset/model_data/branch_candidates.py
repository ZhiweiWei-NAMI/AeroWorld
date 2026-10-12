"""Resolve exact pre-dispatch candidate authority without reading outcome labels.

This is a provenance object, NOT a neural input. Candidate descriptions and
mechanism hypotheses must not be embedded. A later control compiler must retain
only requested controls and bind their handles to the observed graph. In
particular, this loader does not certify the full branch world state at fork.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contract import FORMAL_TICKS, ModelDataError
from Dataset.tools.structured_branch_source import (
    PROJECT_ROOT, BranchSource, project_path, resolve_branch_source,
)


@dataclass(frozen=True)
class AuthoredBranchCandidate:
    source: BranchSource
    candidate_path: Path
    candidate_sha256: str
    group_path: Path
    group_sha256: str
    record: Mapping[str, Any]
    decision_tick: int
    observation_cutoff_tick: int | None
    observation_phase: str
    predicate_id: str
    operation: str

    def require_available(self, decision_tick: int) -> None:
        if type(decision_tick) is not int or decision_tick != self.decision_tick:
            raise ModelDataError("candidate belongs to its declared pre-dispatch decision, not an earlier origin")


def _object(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ModelDataError(f"candidate authority is not a JSON object: {path}")
    return value, hashlib.sha256(data).hexdigest()


def _equal(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ModelDataError(f"candidate authority disagrees with {label}: {actual!r} != {expected!r}")


def load_authored_branch_candidate(
    group: Mapping[str, Any],
    arm: Mapping[str, Any],
    *,
    candidate_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> AuthoredBranchCandidate:
    """Require an explicit candidate source and exact identity/plan agreement.

    The executed ledger is used only to check source alignment, never as a
    replacement for requested parameters. No event realization, predicate truth,
    trajectory value, future graph registry, or legacy kind mapping is read.
    Candidate availability is a declared pre-dispatch experimental assumption,
    not a measured deployment arrival timestamp.
    """
    source = resolve_branch_source(group, arm, project_root=project_root)
    candidate_path = project_path(str(candidate_path), project_root=project_root)
    record, candidate_digest = _object(candidate_path)
    group_path = candidate_path.parent / "group_manifest.json"
    authored_group, group_digest = _object(group_path)
    for field, expected in (
        ("episode_id", group["episode_id"]),
        ("scenario_id", group["profile_id"]),
        ("seed_id", group["seed_id"]),
        ("branch_group_id", group["branch_group_id"]),
        ("anchor_event_id", group["anchor_event_id"]),
        ("fork_tick", source.fork),
        ("target_predicate_id", group["target"]["predicate_id"]),
    ):
        _equal(authored_group[field], expected, f"group.{field}")
    _equal(candidate_path.parent.name, group["group_id"], "group directory")
    _equal(record["profile_id"], group["profile_id"], "profile_id")
    _equal(record["arm_id"], arm["arm_id"], "arm_id")
    _equal(record["arm_kind"], arm["arm_kind"], "arm_kind")
    anchor = record["anchor"]
    for field, expected in (
        ("episode_id", group["episode_id"]), ("profile_id", group["profile_id"]),
        ("seed_id", group["seed_id"]), ("event_id", group["anchor_event_id"]),
        ("dispatch_tick", source.fork),
    ):
        _equal(anchor[field], expected, f"anchor.{field}")
    decision = record["decision_arm"]
    for field in ("arm_id", "arm_kind"):
        _equal(decision[field], arm[field], f"decision_arm.{field}")
    declared = [row for row in authored_group["arms"] if row["arm_id"] == arm["arm_id"]]
    if len(declared) != 1:
        raise ModelDataError("candidate group does not declare this exact ARM once")
    for field in ("arm_kind", "event_operation"):
        _equal(decision[field], declared[0][field], f"declared ARM {field}")
    _equal(record["fork_contract"]["phase"], "pre_dispatch", "fork phase")
    _equal(record["fork_contract"]["target_bundle_status"], "not_committed", "pending target bundle")
    _equal(record["provenance"]["input_cutoff_tick"], source.fork, "declared input cutoff")
    _equal(record["provenance"]["post_fork_factual_refs"], [], "post-fork factual input refs")
    _equal(record["shared_context"]["context_mode"], "preserve_parent", "shared context mode")
    _equal(record["shared_context"]["common_across_decision_arms"], True, "shared context")
    _equal(record["window_policy"]["chosen_before_rollout"], True, "candidate window selection")
    predicate = record["semantic_goal"]["primary_predicate"]["predicate_id"]
    _equal(predicate, group["target"]["predicate_id"], "predeclared query")

    ledger_path = source.raw_file("branch_event_ledger.jsonl")
    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    if len(ledger_rows) != 1 or not isinstance(ledger_rows[0], dict):
        raise ModelDataError("candidate alignment requires one explicit branch operation ledger")
    ledger = ledger_rows[0]
    required = {"arm_id", "arm_kind", "anchor_event_id", "fork_tick", "operation"}
    if decision["event_operation"] == "delay_causal_event":
        required.add("delay_ticks")
    missing = required - ledger.keys()
    if missing:
        raise ModelDataError(f"branch operation ledger missing required fields: {sorted(missing)}")
    for field, expected in (
        ("arm_id", arm["arm_id"]), ("arm_kind", arm["arm_kind"]),
        ("anchor_event_id", group["anchor_event_id"]), ("fork_tick", source.fork),
        ("operation", decision["event_operation"]),
    ):
        _equal(ledger[field], expected, f"executed ledger {field}")
    if decision["event_operation"] == "delay_causal_event":
        steps = [step for step in decision["mechanism_steps"]
                 if step["step_kind"] == "postpone_source_event"]
        if not steps:
            raise ModelDataError("delayed candidate has no explicit postponement")
        for step in steps:
            delay = step["delay_ticks"]
            if type(delay) is not int or delay <= 0:
                raise ModelDataError("candidate delay must be explicit positive integer ticks")
            _equal(delay, ledger["delay_ticks"], "executed delay ticks")
            _equal(step["source_event_id"], group["anchor_event_id"], "postponed source event")

    past = [int(tick) for tick in FORMAL_TICKS if tick < source.fork]
    if past:
        cutoff, phase = past[-1], "strict_prefix"
    elif source.fork == 0:
        # batch_generate records tick 0 before interpreter.tick(0). That row is
        # a genuine pre-dispatch observation at the decision, not a missing prefix.
        cutoff, phase = 0, "pre_dispatch"
    else:
        cutoff, phase = None, "missing_prefix"
    return AuthoredBranchCandidate(
        source=source, candidate_path=candidate_path, candidate_sha256=candidate_digest,
        group_path=group_path, group_sha256=group_digest, record=record,
        decision_tick=source.fork, observation_cutoff_tick=cutoff,
        observation_phase=phase,
        predicate_id=predicate, operation=decision["event_operation"],
    )
